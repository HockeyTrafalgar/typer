#!/usr/bin/env python3
"""
Typer — Hold-to-Talk Live Typing (Parakeet-MLX, Apple Silicon native)

Hold Right-⌘ → speak → words stream into the focused field in real time;
release to stop. Or tap F19 to toggle dictation on/off.

Model: nvidia/parakeet-tdt-0.6b-v3 via parakeet-mlx — multilingual (25
       European languages incl. Russian + English) with per-utterance
       automatic language detection, true streaming, native Apple Silicon.

Requirements: parakeet-mlx, pynput, pyperclip, sounddevice, numpy
Permissions: Microphone + Accessibility + Input Monitoring on the Python binary
"""

import threading
import sys
import os
import time
import queue
import atexit
import signal
import subprocess
import logging
import logging.handlers
import traceback
from pathlib import Path

import collections

import numpy as np
import pyperclip
import sounddevice as sd
import mlx.core as mx
import torch

from pynput import keyboard
import parakeet_mlx
from parakeet_mlx.parakeet import DecodingConfig, Greedy, Beam
from silero_vad import load_silero_vad

# ── CONFIG ──────────────────────────────────────────────────────────────────
MODEL_ID = os.environ.get("TYPER_MODEL", "mlx-community/parakeet-tdt-0.6b-v3")
SAMPLE_RATE = 16000

# How often to push audio into the streaming model + poll for output.
# Smaller = lower latency, more CPU. Larger = more right-context per forward
# pass, so fewer visible "wrong then corrected" revisions.
LIVE_PUSH_INTERVAL_S = float(os.environ.get("TYPER_PUSH_S", "0.5"))

# Streaming-quality knobs for parakeet_mlx.transcribe_stream().
# depth: how many encoder layers carry exact KV-cache across chunks. Higher
#   = streaming output more closely matches a full non-streaming pass (fewer
#   revisions), at the cost of more compute/memory per chunk. Default in
#   the library is 1; we push to 4 for noticeably more stable partials.
STREAM_DEPTH = int(os.environ.get("TYPER_STREAM_DEPTH", "8"))
# (left_context, right_context) attention window in encoder frames. Larger
#   right_context = the model sees further ahead before committing tokens,
#   directly reducing revisions (but adds inherent latency).
STREAM_LEFT_CTX  = int(os.environ.get("TYPER_STREAM_LEFT_CTX", "256"))
STREAM_RIGHT_CTX = int(os.environ.get("TYPER_STREAM_RIGHT_CTX", "384"))
# Preserve original (non-local) attention during streaming. More accurate
#   but heavier; off by default in the library.
STREAM_KEEP_ORIG_ATTN = os.environ.get("TYPER_STREAM_KEEP_ORIG_ATTN", "0") == "1"

# Beam search decoding (vs greedy). Beam explores multiple hypotheses per
#   step and picks the best — lower WER, especially on accents / noisy
#   audio / homophones, at higher compute per chunk. Set BEAM_SIZE=1 to
#   fall back to greedy.
BEAM_SIZE         = int(os.environ.get("TYPER_BEAM_SIZE", "3"))
BEAM_LEN_PENALTY  = float(os.environ.get("TYPER_BEAM_LEN_PENALTY", "1.0"))
BEAM_PATIENCE     = float(os.environ.get("TYPER_BEAM_PATIENCE", "1.0"))
# TDT-only: how strongly to reward longer-duration emissions. The library
#   default is 0.7; tweak if you see the model rushing or stalling.
BEAM_DUR_REWARD   = float(os.environ.get("TYPER_BEAM_DUR_REWARD", "0.7"))

def _build_decoding_config():
    if BEAM_SIZE <= 1:
        log.info("decoding: greedy")
        return DecodingConfig(decoding=Greedy())
    log.info("decoding: beam size=%d len_penalty=%.2f patience=%.2f dur_reward=%.2f",
             BEAM_SIZE, BEAM_LEN_PENALTY, BEAM_PATIENCE, BEAM_DUR_REWARD)
    return DecodingConfig(decoding=Beam(
        beam_size=BEAM_SIZE,
        length_penalty=BEAM_LEN_PENALTY,
        patience=BEAM_PATIENCE,
        duration_reward=BEAM_DUR_REWARD,
    ))

# Single keyboard controller for synthetic keystrokes. We use pynput rather
# than pyautogui for injection: pyautogui's hotkey() leaves the Command
# modifier stuck-down on macOS under rapid-fire use, which turns the next real
# keypress into a shortcut (e.g. Cmd+Space → Spotlight) and steals focus.
_kbd = keyboard.Controller()

# Modifiers we forcibly release after each injection to guarantee no stale
# state leaks into the user's real keystrokes. ONLY the keys we actually
# synthesize (left Command, used by the ⌘V paste) — crucially NOT cmd_r:
# injecting a cmd_r release would be seen by our own listener as the user
# letting go of the Right-⌘ push-to-talk key, instantly stopping dictation.
_MODIFIER_KEYS = (keyboard.Key.cmd,)

def _release_modifiers():
    for k in _MODIFIER_KEYS:
        try:
            _kbd.release(k)
        except Exception:
            pass

# ── VAD CONFIG ──────────────────────────────────────────────────────────────
# Silero VAD operates on fixed 512-sample frames at 16 kHz (= 32 ms).
VAD_FRAME = 512
# Probability above which a frame is considered speech. Higher = stricter
# (fewer false positives / less hallucination). Default 0.6 is conservative.
VAD_THRESHOLD = float(os.environ.get("TYPER_VAD_THRESHOLD", "0.75"))
# Lead-in flushed at speech onset so we don't clip the first phoneme.
VAD_PRE_ROLL_MS = int(os.environ.get("TYPER_VAD_PRE_ROLL_MS", "200"))
# How long silence must persist before we mark the end of an utterance.
VAD_HANGOVER_MS = int(os.environ.get("TYPER_VAD_HANGOVER_MS", "400"))
# Minimum RMS energy required for a frame to even be considered for VAD —
# rejects very-low-level noise/breath that the model occasionally
# misclassifies as speech.
VAD_RMS_FLOOR = float(os.environ.get("TYPER_VAD_RMS_FLOOR", "0.012"))

# ── LOGGING ──────────────────────────────────────────────────────────────────
LOG_DIR = Path(os.environ.get("TYPER_LOG_DIR", Path(__file__).resolve().parent / "logs"))
LOG_LEVEL = os.environ.get("TYPER_LOG_LEVEL", "DEBUG").upper()

def _setup_logging():
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / "typer.log"
    fmt = logging.Formatter(
        "%(asctime)s.%(msecs)03d %(levelname)-8s [%(threadName)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    root = logging.getLogger()
    root.setLevel(LOG_LEVEL)
    # Clear any existing handlers (re-entry safety).
    for h in list(root.handlers):
        root.removeHandler(h)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    sh.setLevel(LOG_LEVEL)
    root.addHandler(sh)
    fh = logging.handlers.RotatingFileHandler(
        log_path, maxBytes=5 * 1024 * 1024, backupCount=5, encoding="utf-8"
    )
    fh.setFormatter(fmt)
    fh.setLevel(LOG_LEVEL)
    root.addHandler(fh)
    # Route uncaught exceptions through logging.
    def _excepthook(exc_type, exc, tb):
        logging.getLogger("typer").critical(
            "Uncaught exception", exc_info=(exc_type, exc, tb)
        )
    sys.excepthook = _excepthook
    def _thread_excepthook(args):
        logging.getLogger("typer").error(
            "Uncaught thread exception in %s", args.thread.name,
            exc_info=(args.exc_type, args.exc_value, args.exc_traceback),
        )
    threading.excepthook = _thread_excepthook
    logging.getLogger("typer").info("Logging to %s (level=%s)", log_path, LOG_LEVEL)
    return log_path

_setup_logging()
log = logging.getLogger("typer")

# ── STATE ────────────────────────────────────────────────────────────────────
model            = None
model_loaded     = threading.Event()  # set on success OR failure; check `model` for actual readiness

live_active      = False
live_stop_event  = threading.Event()
live_state_lock  = threading.Lock()  # serializes start/stop transitions

f19_held         = False  # edge-detect F19 press (toggles live mode)
rcmd_held        = False  # tracks Right-⌘ hold-to-talk state

# Single inference thread owns the model — MLX tensors are bound to the thread
# that created them, so we cannot load on one thread and infer on another.
inference_q = queue.Queue()

# ── INFERENCE THREAD ─────────────────────────────────────────────────────────
def inference_worker():
    """Owns the Parakeet model. Loads it, then services live jobs forever."""
    global model
    log.info("inference worker starting")
    try:
        mx.set_default_device(mx.gpu)
        _ = (mx.zeros((1,)) + 1).item()
        log.debug("MLX GPU warm-up ok")
    except Exception:
        log.exception("MLX init failed")
    log.info("Loading Parakeet model: %s", MODEL_ID)
    log.info("(first run downloads ~1.2 GB; cached after)")
    t0 = time.time()
    try:
        model = parakeet_mlx.from_pretrained(MODEL_ID)
        log.info("Parakeet ready in %.1fs", time.time() - t0)
        try:
            _get_vad_model()
        except Exception:
            log.exception("VAD load failed (continuing without VAD)")
    except Exception:
        log.exception("model load failed — dictation will be unavailable")
    finally:
        # Always release the main thread; it checks `model is not None` to know
        # whether load actually succeeded.
        model_loaded.set()
    if model is None:
        return
    while True:
        job = inference_q.get()
        log.debug("inference job: %r", job)
        try:
            if job[0] == "live":
                _do_live_stream()
        except Exception:
            log.exception("inference job error")

# ── VAD GATE ─────────────────────────────────────────────────────────────────
class VadGate:
    """Streaming Silero-VAD gate.

    Feed it raw 16 kHz float32 audio chunks of any size via .process(samples)
    and it returns ONLY the voiced portion (with pre-roll lead-in and
    post-speech hangover), suitable for forwarding to an ASR streaming model
    without poisoning it with silence-hallucinations.
    """
    def __init__(self, model, threshold, pre_roll_ms, hangover_ms, rms_floor):
        self.model = model
        self.threshold = threshold
        self.pre_roll_frames = max(1, pre_roll_ms * 16000 // 1000 // VAD_FRAME)
        self.hangover_frames = max(1, hangover_ms * 16000 // 1000 // VAD_FRAME)
        self.rms_floor = rms_floor
        self.tail = np.zeros(0, dtype=np.float32)
        self.pre_roll = collections.deque(maxlen=self.pre_roll_frames)
        self.in_speech = False
        self.silent_run = 0
        self.stats = {"frames": 0, "voiced": 0, "emitted": 0,
                      "speech_starts": 0, "speech_ends": 0}

    def reset(self):
        self.tail = np.zeros(0, dtype=np.float32)
        self.pre_roll.clear()
        self.in_speech = False
        self.silent_run = 0
        try:
            self.model.reset_states()
        except Exception:
            pass

    def _is_voiced(self, frame):
        rms = float(np.sqrt(np.mean(frame * frame)))
        if rms < self.rms_floor:
            return False, 0.0, rms
        with torch.no_grad():
            prob = float(self.model(torch.from_numpy(frame), 16000).item())
        return prob >= self.threshold, prob, rms

    def process(self, samples):
        if len(samples) == 0:
            return np.zeros(0, dtype=np.float32)
        buf = np.concatenate([self.tail, samples]) if len(self.tail) else samples
        n = len(buf) // VAD_FRAME
        out = []
        for i in range(n):
            frame = buf[i * VAD_FRAME:(i + 1) * VAD_FRAME].astype(np.float32, copy=False)
            voiced, prob, rms = self._is_voiced(frame)
            self.stats["frames"] += 1
            if voiced:
                self.stats["voiced"] += 1
            if voiced:
                if not self.in_speech:
                    self.in_speech = True
                    self.stats["speech_starts"] += 1
                    log.debug("VAD ▶ speech start (p=%.2f rms=%.4f, flushing %d pre-roll frames)",
                              prob, rms, len(self.pre_roll))
                    for pr in self.pre_roll:
                        out.append(pr)
                    self.pre_roll.clear()
                self.silent_run = 0
                out.append(frame)
            else:
                if self.in_speech:
                    out.append(frame)  # forward through hangover window
                    self.silent_run += 1
                    if self.silent_run >= self.hangover_frames:
                        self.in_speech = False
                        self.silent_run = 0
                        self.stats["speech_ends"] += 1
                        log.debug("VAD ■ speech end (hangover %d frames)",
                                  self.hangover_frames)
                else:
                    self.pre_roll.append(frame)
        self.tail = buf[n * VAD_FRAME:].copy()
        if out:
            arr = np.concatenate(out)
            self.stats["emitted"] += len(arr)
            return arr
        return np.zeros(0, dtype=np.float32)


vad_model = None
vad_model_lock = threading.Lock()

def _get_vad_model():
    global vad_model
    with vad_model_lock:
        if vad_model is None:
            log.info("loading Silero VAD (onnx)")
            t0 = time.time()
            vad_model = load_silero_vad(onnx=True)
            log.info("Silero VAD ready in %.2fs", time.time() - t0)
        return vad_model


# ── LIVE STREAMING ────────────────────────────────────────────────────────────
def _paste_text(text):
    """Copy `text` to clipboard and emit ⌘V at the focused field.

    Uses pynput's Controller with a `pressed` context manager so the Command
    modifier is *always* released, even on exception — preventing the stuck-
    modifier focus-stealing seen with pyautogui.hotkey on macOS.
    """
    if not text:
        return
    log.debug("paste %d chars: %r", len(text), text)
    pyperclip.copy(text)
    time.sleep(0.03)
    _release_modifiers()
    with _kbd.pressed(keyboard.Key.cmd):
        _kbd.press('v')
        _kbd.release('v')

def _emit_diff(current, last):
    """Reconcile what's in the field (last) with the new transcript (current).

    Backspaces over any diverged tail in `last`, then pastes the new tail
    from `current`. Returns the canonical text now in the field.
    """
    n = min(len(current), len(last))
    i = 0
    while i < n and current[i] == last[i]:
        i += 1
    to_delete = len(last) - i
    to_add = current[i:]
    if to_delete or to_add:
        log.debug("diff: -%d +%d (common=%d)", to_delete, len(to_add), i)
    if to_delete > 0:
        _release_modifiers()
        for _ in range(to_delete):
            _kbd.press(keyboard.Key.backspace)
            _kbd.release(keyboard.Key.backspace)
    if to_add:
        _paste_text(to_add)
    return current

def _do_live_stream():
    """True streaming via Parakeet's StreamingParakeet context.

    Runs ON THE INFERENCE THREAD (model owner). sounddevice's audio callback
    runs on its own thread but only enqueues raw samples; all MLX ops stay here.
    """
    log.info("live stream begin")
    audio_q = queue.Queue()
    cb_stats = {"frames": 0, "calls": 0, "status_warns": 0}

    def _audio_cb(indata, frames, time_info, status):
        cb_stats["calls"] += 1
        cb_stats["frames"] += frames
        if status:
            cb_stats["status_warns"] += 1
            log.warning("audio callback status: %s", status)
        audio_q.put(indata[:, 0].copy())

    gate = None
    try:
        gate = VadGate(_get_vad_model(), VAD_THRESHOLD,
                       VAD_PRE_ROLL_MS, VAD_HANGOVER_MS, VAD_RMS_FLOOR)
        log.info("VAD active (threshold=%.2f pre_roll=%dms hangover=%dms rms_floor=%.4f)",
                 VAD_THRESHOLD, VAD_PRE_ROLL_MS, VAD_HANGOVER_MS, VAD_RMS_FLOOR)
    except Exception:
        log.exception("VAD init failed, falling back to raw audio (hallucinations possible)")

    last_pasted = ""
    iters = 0

    def _drain_and_emit(stream):
        nonlocal last_pasted
        buf = []
        while True:
            try:
                buf.append(audio_q.get_nowait())
            except queue.Empty:
                break
        if not buf:
            return
        samples = np.concatenate(buf)
        raw_n = len(samples)
        if gate is not None:
            samples = gate.process(samples)
        if len(samples):
            log.debug("drain: raw=%d voiced=%d (%.2fs voiced)",
                      raw_n, len(samples), len(samples) / SAMPLE_RATE)
            stream.add_audio(mx.array(samples))
        current = stream.result.text or ""
        if current != last_pasted:
            log.debug("transcript update: %r", current)
            last_pasted = _emit_diff(current, last_pasted)

    t_start = time.time()
    try:
        log.info("stream cfg: depth=%d context=(%d,%d) keep_orig_attn=%s push=%.2fs",
                 STREAM_DEPTH, STREAM_LEFT_CTX, STREAM_RIGHT_CTX,
                 STREAM_KEEP_ORIG_ATTN, LIVE_PUSH_INTERVAL_S)
        with model.transcribe_stream(
            context_size=(STREAM_LEFT_CTX, STREAM_RIGHT_CTX),
            depth=STREAM_DEPTH,
            keep_original_attention=STREAM_KEEP_ORIG_ATTN,
            decoding_config=_build_decoding_config(),
        ) as stream, sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
            blocksize=int(SAMPLE_RATE * 0.1),  # 100ms callback granularity
            callback=_audio_cb,
        ):
            log.debug("audio input opened sr=%d", SAMPLE_RATE)
            while not live_stop_event.is_set():
                time.sleep(LIVE_PUSH_INTERVAL_S)
                iters += 1
                t_tick = time.time()
                _drain_and_emit(stream)
                tick_ms = (time.time() - t_tick) * 1000.0
                try:
                    active_mb = mx.get_active_memory() / (1024 * 1024)
                    peak_mb = mx.get_peak_memory() / (1024 * 1024)
                    log.debug("tick %d: drain+infer=%.0fms mlx_active=%.0fMB peak=%.0fMB",
                              iters, tick_ms, active_mb, peak_mb)
                except Exception:
                    log.debug("tick %d: drain+infer=%.0fms", iters, tick_ms)
            # Final flush — capture the tail spoken in the last interval.
            log.debug("final flush after %d iters", iters)
            _drain_and_emit(stream)
    except Exception:
        log.exception("stream error")
    finally:
        vad_stats = gate.stats if gate else {}
        if gate:
            gate.reset()
        log.info(
            "live stream end: dur=%.2fs iters=%d audio_calls=%d audio_frames=%d status_warns=%d vad=%s final_text=%r",
            time.time() - t_start, iters, cb_stats["calls"], cb_stats["frames"],
            cb_stats["status_warns"], vad_stats, last_pasted,
        )

# ── LIVE START/STOP ──────────────────────────────────────────────────────────
def start_live_dictation():
    global live_active
    with live_state_lock:
        if live_active:
            log.debug("start_live_dictation: already active, ignoring")
            return
        if not model_loaded.is_set() or model is None:
            log.warning("model not loaded; ignoring start")
            return
        live_stop_event.clear()
        live_active = True
    log.info("🎤 Live — speak now")
    inference_q.put(("live",))

def stop_live_dictation():
    global live_active
    with live_state_lock:
        if not live_active:
            log.debug("stop_live_dictation: not active, ignoring")
            return
        live_active = False
        live_stop_event.set()
    log.info("✅ stopped")

# ── KEYBOARD LISTENER ────────────────────────────────────────────────────────
def _toggle_live():
    if live_active:
        stop_live_dictation()
    else:
        start_live_dictation()

def on_press(key):
    global f19_held, rcmd_held
    # Right-⌘ → hold-to-talk: start on press, stop on release.
    if key == keyboard.Key.cmd_r:
        if not rcmd_held:         # ignore macOS auto-repeat while held
            rcmd_held = True
            log.debug("right-cmd press → hold-to-talk start")
            threading.Thread(target=start_live_dictation, daemon=True,
                             name="start-live").start()
    # F19 → toggle: tap to start, tap again to stop.
    elif key == keyboard.Key.f19:
        if not f19_held:          # edge-detect: one toggle per physical press
            f19_held = True
            log.debug("F19 press → toggle (active=%s)", live_active)
            threading.Thread(target=_toggle_live, daemon=True,
                             name="toggle-live").start()

def on_release(key):
    global f19_held, rcmd_held
    if key == keyboard.Key.cmd_r:
        rcmd_held = False
        log.debug("right-cmd release → hold-to-talk stop")
        threading.Thread(target=stop_live_dictation, daemon=True,
                         name="stop-live").start()
    elif key == keyboard.Key.f19:
        f19_held = False
        log.debug("F19 release (no-op for toggle)")

# ── MAIN ─────────────────────────────────────────────────────────────────────
def main():
    print("""
╔══════════════════════════════════════════════╗
║   Typer — Parakeet-MLX live typing          ║
╠══════════════════════════════════════════════╣
║  Hold Right-⌘ → speak → release to stop     ║
║  Tap F19 → start, tap again → stop          ║
║  Ctrl+C to quit                              ║
╚══════════════════════════════════════════════╝
""")
    log.info("Typer starting (pid=%d, python=%s)", os.getpid(), sys.version.split()[0])
    threading.Thread(target=inference_worker, daemon=True,
                     name="inference").start()
    log.info("loading model in background...")
    model_loaded.wait()
    if model is None:
        log.critical("model load failed — exiting")
        sys.exit(1)
    log.info("🟢 ready. Hold Right-⌘ to talk, or tap F19 to toggle.")

    def _cleanup():
        log.info("cleanup")
        if live_active:
            stop_live_dictation()
    atexit.register(_cleanup)
    signal.signal(signal.SIGTERM, lambda *a: (log.info("SIGTERM"), _cleanup(), sys.exit(0)))

    with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
        try:
            listener.join()
        except KeyboardInterrupt:
            log.info("shutting down (KeyboardInterrupt)")
            _cleanup()

if __name__ == "__main__":
    try:
        main()
    except Exception:
        log.critical("fatal in main", exc_info=True)
        raise
