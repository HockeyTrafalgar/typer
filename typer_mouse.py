"""Magic Mouse double-tap-and-hold gesture detector.

Reads raw multitouch frames from the private MultitouchSupport
framework (the same one BetterTouchTool uses), watches for a
"2-finger tap → release → 2-finger press → hold → release" pattern
on the Magic Mouse surface, and fires user-supplied callbacks at the
hold start and the hold release.

Press-to-talk semantics — like Right-⌘, but on the mouse. Doesn't
conflict with normal mouse use because incidental contact is almost
always a single finger.

Caveats:
  - MultitouchSupport is a PRIVATE Apple framework. The symbol
    interface has been stable for years but Apple can break it at any
    minor macOS release. If that happens, this module fails gracefully:
    .start() returns False and we just log a warning — the rest of
    Typer keeps working with the keyboard hotkeys.
  - We filter to physical mice by sensor-grid size (< 250 cells).
    Laptop trackpads and external Magic Trackpads have much larger
    grids and get skipped, so the gesture won't fire from them.
  - macOS may also synthesize its own gesture from these touches
    (e.g. Mission Control on two-finger double-tap). We just observe
    raw contacts — we don't suppress the OS gesture. If you have it
    bound, disable it in System Settings → Trackpad → More Gestures.
"""
import ctypes
import ctypes.util
import logging
import threading
import time
from ctypes import (
    c_int, c_uint, c_double, c_float, c_void_p,
    POINTER, Structure, CFUNCTYPE,
)

log = logging.getLogger("typer.mouse")

# ── ctypes bindings ─────────────────────────────────────────────────────────
_MT_PATH = (
    "/System/Library/PrivateFrameworks/MultitouchSupport.framework/"
    "MultitouchSupport"
)

try:
    _mt = ctypes.CDLL(_MT_PATH)
    _cf = ctypes.CDLL(ctypes.util.find_library("CoreFoundation"))
    _MT_AVAILABLE = True
except OSError as e:
    log.warning("MultitouchSupport not loadable: %s — mouse trigger disabled", e)
    _mt = None
    _cf = None
    _MT_AVAILABLE = False

if _MT_AVAILABLE:
    _cf.CFArrayGetCount.restype = c_int
    _cf.CFArrayGetCount.argtypes = [c_void_p]
    _cf.CFArrayGetValueAtIndex.restype = c_void_p
    _cf.CFArrayGetValueAtIndex.argtypes = [c_void_p, c_int]
    _cf.CFRunLoopRunInMode.restype = c_int
    _cf.CFRunLoopRunInMode.argtypes = [c_void_p, c_double, c_int]
    _kCFRunLoopDefaultMode = c_void_p.in_dll(_cf, "kCFRunLoopDefaultMode")

    class _MTPoint(Structure):
        _fields_ = [("x", c_float), ("y", c_float)]

    class _MTReadout(Structure):
        _fields_ = [("pos", _MTPoint), ("vel", _MTPoint)]

    class _Finger(Structure):
        _fields_ = [
            ("frame", c_int),
            ("timestamp", c_double),
            ("identifier", c_int),
            ("state", c_int),
            ("foo3", c_int),
            ("foo4", c_int),
            ("normalized", _MTReadout),
            ("size", c_float),
            ("zero1", c_int),
            ("angle", c_float),
            ("majorAxis", c_float),
            ("minorAxis", c_float),
            ("absolute", _MTReadout),
            ("zero2", c_int * 2),
            ("z_density", c_float),
        ]

    _MTContactCallbackFunction = CFUNCTYPE(
        c_int, c_void_p, POINTER(_Finger), c_int, c_double, c_int
    )

    _mt.MTDeviceCreateList.restype = c_void_p
    _mt.MTDeviceCreateList.argtypes = []
    _mt.MTDeviceGetFamilyID.restype = c_int
    _mt.MTDeviceGetFamilyID.argtypes = [c_void_p, POINTER(c_int)]
    _mt.MTRegisterContactFrameCallback.restype = c_int
    _mt.MTRegisterContactFrameCallback.argtypes = [c_void_p, _MTContactCallbackFunction]
    # Newer-hardware-compatible variant. Same callback signature, takes
    # an extra void* refcon (we pass NULL). Used by BetterTouchTool to
    # subscribe to the Apple Silicon Magic Mouse 2 / USB-C variants
    # where the no-refcon entry point returns rc=1.
    try:
        _mt.MTRegisterContactFrameCallbackWithRefcon.restype = c_int
        _mt.MTRegisterContactFrameCallbackWithRefcon.argtypes = [
            c_void_p, _MTContactCallbackFunction, c_void_p
        ]
        _HAS_REFCON_REG = True
    except AttributeError:
        _HAS_REFCON_REG = False
    _mt.MTDeviceStart.restype = c_int
    _mt.MTDeviceStart.argtypes = [c_void_p, c_int]
    _mt.MTDeviceStop.restype = c_int
    _mt.MTDeviceStop.argtypes = [c_void_p]
    # Optional — only present on some macOS versions, guarded at call site.
    try:
        _mt.MTDeviceGetSensorDimensions.restype = c_int
        _mt.MTDeviceGetSensorDimensions.argtypes = [
            c_void_p, POINTER(c_int), POINTER(c_int)
        ]
        _HAS_SENSOR_DIMS = True
    except AttributeError:
        _HAS_SENSOR_DIMS = False


# Community-known family IDs that have shown up as Magic Mouse on
# various macOS releases. We treat any of these as "definitely a
# mouse." If the actual hardware has a different family ID, we fall
# back to the sensor-dimension heuristic below.
_KNOWN_MOUSE_FAMILIES = {102, 112, 113, 117, 128, 129}
# Trackpads have larger sensor grids; anything below this cell-count
# threshold is treated as a mouse. Magic Mouse 2 reports ~150 cells
# (10x15); a Magic Trackpad reports >400.
_SENSOR_GRID_MOUSE_CAP = 250


class GestureDetector:
    """Watches Magic Mouse multitouch contacts and fires
    on_press_start / on_press_end around a "double-tap-then-hold"
    gesture.

    Gesture state machine over contact-count edge transitions:

        IDLE        → 2 fingers down                       → FIRST_TAP
        FIRST_TAP   → <2 fingers within TAP_MAX_MS         → GAP
                    → still down after TAP_MAX_MS          → (long press — ignored)
        GAP         → 2 fingers down within DOUBLE_TAP_GAP → HOLDING (fire start)
                    → timeout                              → IDLE
        HOLDING     → <2 fingers                           → IDLE (fire end)
    """

    def __init__(
        self,
        on_press_start,
        on_press_end,
        tap_max_ms=260,
        double_tap_gap_ms=350,
    ):
        self._on_press_start = on_press_start
        self._on_press_end = on_press_end
        self._tap_max_ms = tap_max_ms
        self._double_tap_gap_ms = double_tap_gap_ms

        self._state = "IDLE"
        self._t_down1 = 0.0
        self._t_up1 = 0.0
        self._last_count = 0
        self._lock = threading.Lock()

        # ctypes objects we MUST keep references to so they aren't GC'd
        # out from under the framework's callback table.
        self._cb_refs = []
        self._devices = []
        self._thread = None
        self._stop = threading.Event()

    # ── public API ──────────────────────────────────────────────────────────
    def start(self):
        """Spin up a dedicated CFRunLoop thread that will open the
        multitouch framework, find the Magic Mouse, and register the
        frame callback. Always returns True if MT itself is available
        — the actual device discovery happens on the worker thread
        and only logs (a failure there is non-fatal)."""
        if not _MT_AVAILABLE:
            return False
        self._thread = threading.Thread(
            target=self._run_loop, daemon=True, name="mt-touch"
        )
        self._thread.start()
        return True

    def stop(self):
        self._stop.set()
        for d in self._devices:
            try:
                _mt.MTDeviceStop(d)
            except Exception:
                pass

    # ── device discovery ────────────────────────────────────────────────────
    def _attach_devices(self):
        devs = _mt.MTDeviceCreateList()
        if not devs:
            log.warning("MTDeviceCreateList returned NULL")
            return False
        n = _cf.CFArrayGetCount(devs)
        log.info("mouse trigger: found %d multitouch device(s)", n)
        # FIRST PASS — only look at family IDs, defer sensor-dimension
        # probing until AFTER we've identified the mouse. Calling
        # MTDeviceGetSensorDimensions on a device we then don't register
        # appears to leave the framework in a state where the NEXT
        # device's MTRegisterContactFrameCallback returns rc=1.
        for i in range(n):
            d = _cf.CFArrayGetValueAtIndex(devs, i)
            fam = c_int(-1)
            _mt.MTDeviceGetFamilyID(d, ctypes.byref(fam))
            if fam.value in _KNOWN_MOUSE_FAMILIES:
                self._register(d, fam.value, grid=-1)
        if self._devices:
            return True
        # FALLBACK — no family matched; try sensor-dimension heuristic
        # to catch hardware whose family ID we haven't seen yet.
        log.info(
            "mouse trigger: no known-family mouse found, trying sensor-dim heuristic"
        )
        for i in range(n):
            d = _cf.CFArrayGetValueAtIndex(devs, i)
            fam = c_int(-1)
            _mt.MTDeviceGetFamilyID(d, ctypes.byref(fam))
            if fam.value in _KNOWN_MOUSE_FAMILIES:
                continue  # already tried in pass 1
            grid = self._sensor_cells(d)
            if 0 < grid <= _SENSOR_GRID_MOUSE_CAP:
                self._register(d, fam.value, grid)
            else:
                log.debug(
                    "mouse trigger: skipping device family=%d grid=%d (not a mouse)",
                    fam.value, grid,
                )
        return bool(self._devices)

    @staticmethod
    def _sensor_cells(device):
        if not _HAS_SENSOR_DIMS:
            return -1
        cols = c_int(0)
        rows = c_int(0)
        try:
            _mt.MTDeviceGetSensorDimensions(
                device, ctypes.byref(cols), ctypes.byref(rows)
            )
        except Exception:
            return -1
        return cols.value * rows.value

    @classmethod
    def _looks_like_mouse(cls, family_id, grid_size):
        if family_id in _KNOWN_MOUSE_FAMILIES:
            return True
        # Unknown family — accept only if the grid is small enough to
        # plausibly be a Magic Mouse, not a trackpad.
        if 0 < grid_size <= _SENSOR_GRID_MOUSE_CAP:
            return True
        return False

    def _register(self, device, family_id, grid):
        log.info(
            "mouse trigger: registering device family=%d grid=%d",
            family_id, grid,
        )
        def _cb(d, fingers_ptr, n_fingers, timestamp, frame):
            try:
                self._on_frame(int(n_fingers))
            except Exception:
                log.exception("frame callback failed")
            return 0
        ref = _MTContactCallbackFunction(_cb)
        self._cb_refs.append(ref)  # keepalive
        # Try the WithRefcon variant first (works for modern Magic Mouse
        # on Apple Silicon); fall back to the plain variant for older
        # hardware / older macOS where the WithRefcon symbol is absent.
        rc = 1
        if _HAS_REFCON_REG:
            rc = _mt.MTRegisterContactFrameCallbackWithRefcon(device, ref, None)
            if rc != 0:
                log.debug(
                    "MTRegisterContactFrameCallbackWithRefcon rc=%d, "
                    "falling back to plain MTRegisterContactFrameCallback",
                    rc,
                )
        if rc != 0:
            rc = _mt.MTRegisterContactFrameCallback(device, ref)
        if rc != 0:
            log.warning(
                "register-contact-frame-callback failed rc=%d (device family=%d). "
                "Probably an entitlement requirement Apple added in macOS 26 "
                "(Tahoe) for MultitouchSupport — the PyObjC-bundled Python "
                "binary lacks the private entitlement. Keyboard hotkeys "
                "continue to work.",
                rc, family_id,
            )
            return
        rc = _mt.MTDeviceStart(device, 0)
        if rc != 0:
            log.warning(
                "MTDeviceStart failed rc=%d (device family=%d)",
                rc, family_id,
            )
            return
        self._devices.append(device)
        log.info(
            "mouse trigger: attached device family=%d grid=%d",
            family_id, grid,
        )

    # ── run loop ────────────────────────────────────────────────────────────
    def _run_loop(self):
        # Device discovery + registration must happen on the SAME
        # thread that will pump the CFRunLoop where MT delivers
        # contact frames. Doing this on the main thread (the old code
        # path) caused MTRegisterContactFrameCallback to return rc=1
        # because the registration is bound to the calling thread's
        # runloop, which never gets pumped (the main thread runs
        # AppKit's runloop, not the default-mode one MT uses).
        if not self._attach_devices():
            log.info("mouse trigger: no Magic Mouse detected")
            return
        log.info(
            "mouse trigger active: %d device(s), tap≤%dms gap≤%dms",
            len(self._devices), self._tap_max_ms, self._double_tap_gap_ms,
        )
        # Drive a short-timeout CFRunLoop. The frame callbacks get
        # dispatched here. The short timeout also lets us tear down
        # cleanly when stop() is called and run periodic state-machine
        # timeouts that don't depend on an incoming frame.
        while not self._stop.is_set():
            _cf.CFRunLoopRunInMode(_kCFRunLoopDefaultMode, 0.1, False)
            self._check_state_timeouts()

    # ── state machine ──────────────────────────────────────────────────────
    def _now_ms(self):
        return time.monotonic() * 1000.0

    def _on_frame(self, n_fingers):
        prev = self._last_count
        self._last_count = n_fingers
        # Edge: contact count crossed the "two or more fingers" boundary.
        going_down = n_fingers >= 2 and prev < 2
        going_up = n_fingers < 2 and prev >= 2
        if not (going_down or going_up):
            return
        now = self._now_ms()
        with self._lock:
            state = self._state
            if going_down:
                if state == "IDLE":
                    self._state = "FIRST_TAP"
                    self._t_down1 = now
                elif state == "GAP":
                    if (now - self._t_up1) <= self._double_tap_gap_ms:
                        self._state = "HOLDING"
                        self._fire(self._on_press_start)
                    else:
                        self._state = "FIRST_TAP"
                        self._t_down1 = now
            elif going_up:
                if state == "FIRST_TAP":
                    if (now - self._t_down1) <= self._tap_max_ms:
                        self._state = "GAP"
                        self._t_up1 = now
                    else:
                        # First "tap" was actually a long press —
                        # not part of a double-tap gesture.
                        self._state = "IDLE"
                elif state == "HOLDING":
                    self._state = "IDLE"
                    self._fire(self._on_press_end)
                else:
                    self._state = "IDLE"

    def _check_state_timeouts(self):
        """Run periodically from the CFRunLoop thread to expire stale
        GAP states even if no new frames arrive."""
        now = self._now_ms()
        with self._lock:
            if (self._state == "GAP"
                    and now - self._t_up1 > self._double_tap_gap_ms):
                self._state = "IDLE"

    @staticmethod
    def _fire(cb):
        try:
            cb()
        except Exception:
            log.exception("gesture callback raised")
