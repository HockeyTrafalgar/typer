#!/usr/bin/env python3
"""
Typer — Hold-to-Talk Live Typing (Whisper-MLX, Apple Silicon native)

Hold Right-⌘ → speak → words stream into the focused field in near real time;
release to stop. Or tap F19 to toggle dictation on/off.

Model: mlx-community/whisper-large-v3-mlx via mlx-whisper — multilingual
       (~99 languages) with automatic language detection, native Apple
       Silicon. Whisper is NOT a true streaming model the way Parakeet
       is, so we approximate streaming by re-transcribing a growing
       voiced-audio buffer every push interval and diff-pasting the new
       suffix into the focused field. This is heavier per tick than
       Parakeet but typically more accurate, especially on accented or
       noisy speech.

Requirements: mlx-whisper, pynput, pyperclip, sounddevice, numpy,
              silero-vad, onnxruntime, torch
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
from pathlib import Path

import collections

import numpy as np
import pyperclip
import sounddevice as sd
import mlx.core as mx
import torch

# ── Hugging Face cache short-circuit ─────────────────────────────────────────
# mlx_whisper internally calls huggingface_hub.snapshot_download(), which
# by default issues an HTTP GET to https://huggingface.co/api/models/...
# every time we warm up — even when the weights are already on disk —
# to check whether the cached revision is current. For an app that loads
# the same pinned model on every launch, that costs us a ~200-800 ms
# round-trip on each start and breaks the app when offline.
#
# Set HF_HUB_OFFLINE=1 BEFORE the mlx_whisper import (which is what
# pulls in huggingface_hub) so the snapshot resolver skips the API
# check and uses the cache directly. We do this automatically when the
# repo is already cached; first run leaves it online so the initial
# download works. TYPER_HF_OFFLINE=0 forces the online path even when
# cached (useful if you want to pick up a newer revision).
def _maybe_enable_hf_offline():
    if os.environ.get("HF_HUB_OFFLINE") is not None:
        return  # respect user override either direction
    if os.environ.get("TYPER_HF_OFFLINE", "1") == "0":
        return
    repo = os.environ.get("TYPER_MODEL", "mlx-community/whisper-large-v3-mlx")
    cache_root = Path(os.environ.get("HF_HOME") or
                      Path.home() / ".cache" / "huggingface") / "hub"
    cache_dir = cache_root / f"models--{repo.replace('/', '--')}"
    if cache_dir.is_dir():
        os.environ["HF_HUB_OFFLINE"] = "1"
        # Older huggingface_hub releases honored this constant instead.
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
_maybe_enable_hf_offline()

from pynput import keyboard
import mlx_whisper
from silero_vad import load_silero_vad

# Magic Mouse double-tap-and-hold gesture detector. Lives in its own
# module because it pokes a private Apple framework via ctypes and is
# allowed to fail without breaking the rest of the app.
from typer_mouse import GestureDetector as _MouseGestureDetector

# AppKit (PyObjC) drives the live-transcription overlay HUD. NSPanel with
# becomesKeyOnlyIfNeeded + nonactivating behavior is the only way to get a
# floating window on macOS that NEVER steals focus from the user's app —
# critical here since we're synthesizing keystrokes into that app.
import objc
from AppKit import (
    NSApplication, NSApp, NSPanel, NSScreen, NSEvent, NSColor, NSFont,
    NSTextField, NSView, NSMakeRect, NSMakePoint, NSMakeSize,
    NSBackingStoreBuffered, NSFloatingWindowLevel,
    NSWindowCollectionBehaviorCanJoinAllSpaces,
    NSWindowCollectionBehaviorStationary,
    NSWindowCollectionBehaviorIgnoresCycle,
    NSVisualEffectView,
    NSVisualEffectMaterialHUDWindow, NSVisualEffectMaterialPopover,
    NSVisualEffectBlendingModeBehindWindow,
    NSVisualEffectStateActive,
    NSFontWeightRegular,
    NSViewWidthSizable, NSViewHeightSizable,
    NSAppearance, NSAppearanceNameVibrantDark, NSAppearanceNameVibrantLight,
    NSImage, NSBezierPath, NSEdgeInsetsMake, NSImageResizingModeStretch,
    NSImageView, NSImageSymbolConfiguration, NSTextAlignmentCenter,
    NSFontAttributeName,
    NSLineBreakByWordWrapping, NSStringDrawingUsesLineFragmentOrigin,
)
# Liquid Glass APIs — shipped in macOS 26 (Tahoe, WWDC '25). When
# present we use NSGlassEffectView for proper refractive glass that
# adapts to whatever's behind the panel; otherwise we keep the older
# NSVisualEffectView path (which uses blur+vibrancy instead of true
# refraction). Import is wrapped so the script still loads on older
# macOS, where these symbols are absent.
try:
    from AppKit import (
        NSGlassEffectView,
        NSGlassEffectViewStyleRegular,
        NSGlassEffectViewStyleClear,
    )
    _LIQUID_GLASS_AVAILABLE = True
except ImportError:
    NSGlassEffectView = None
    NSGlassEffectViewStyleRegular = None
    NSGlassEffectViewStyleClear = None
    _LIQUID_GLASS_AVAILABLE = False
from Quartz import kCACornerCurveContinuous
from Foundation import NSObject
from PyObjCTools import AppHelper

# Accessibility API — used to locate the text caret of the focused field
# so the overlay can sit next to it instead of next to the mouse cursor.
# Requires the user to grant Accessibility permission to the Python binary
# (System Settings → Privacy & Security → Accessibility). Apps that don't
# expose AXSelectedTextRange / AXBoundsForRange (some Electron apps, GPU-
# rendered terminals) silently fall back to the mouse-cursor position.
from ApplicationServices import (
    AXUIElementCreateSystemWide,
    AXUIElementCopyAttributeValue,
    AXUIElementCopyParameterizedAttributeValue,
    AXValueGetValue,
    kAXFocusedUIElementAttribute,
    kAXSelectedTextRangeAttribute,
    kAXValueCGRectType,
)

# NSPanel-only style mask that gives us a borderless, non-activating HUD.
# AppKit doesn't always re-export this constant by name in PyObjC, so we
# define it manually (NSWindowStyleMaskNonactivatingPanel = 1<<7).
NSWindowStyleMaskNonactivatingPanel = 1 << 7

# ── CONFIG ──────────────────────────────────────────────────────────────────
# Best available Whisper on MLX. Use "mlx-community/whisper-large-v3-turbo"
# for ~2-3× faster inference at slightly lower accuracy, or
# "mlx-community/whisper-medium-mlx" / "...small-mlx" for lower memory.
MODEL_ID = os.environ.get("TYPER_MODEL", "mlx-community/whisper-large-v3-mlx")
SAMPLE_RATE = 16000

# Typing mode:
#   "batch"  — accumulate audio while held, transcribe once on release,
#              paste the full final text in one shot. No diff churn, no
#              backspaces, no risk of partial pastes landing in the wrong
#              window. Live transcription still runs every push interval
#              and updates the floating overlay so you see what's coming.
#   "stream" — legacy behavior: re-transcribe every push interval and
#              diff-paste the new suffix directly into the focused field.
TYPER_MODE = os.environ.get("TYPER_MODE", "batch").lower()

# How often to re-transcribe for the overlay preview (batch mode) or for
# the diff-paste tick (stream mode). Whisper is not natively streaming,
# so each tick re-runs the model on the full voiced audio captured so
# far. Larger = lower CPU/GPU load, more lag in the overlay preview.
#
# Note: this is the MINIMUM wait between ticks; if inference takes
# longer than the interval (very likely with whisper-large-v3 once the
# buffer grows past ~5 s), the real cadence is bounded by inference time.
# If you want truly snappy feedback, switch to a faster model:
#   TYPER_MODEL=mlx-community/whisper-large-v3-turbo  # ~3× faster
#   TYPER_MODEL=mlx-community/whisper-small-mlx       # ~10× faster
LIVE_PUSH_INTERVAL_S = float(os.environ.get("TYPER_PUSH_S", "0.35"))

# Optional explicit language code (e.g. "en", "ru"). Empty = auto-detect
# per inference call. Setting this skips detection and saves a bit of time.
WHISPER_LANGUAGE = os.environ.get("TYPER_WHISPER_LANGUAGE", "") or None

# Decoding knobs forwarded to mlx_whisper.transcribe(). mlx-whisper does
# NOT yet implement beam search (raises NotImplementedError), so we use
# greedy decode at T=0 with temperature-fallback sampling at higher T when
# the output looks degenerate. Set TYPER_BEAM_SIZE > 1 only if a future
# mlx-whisper release adds beam decoding.
WHISPER_BEAM_SIZE   = int(os.environ.get("TYPER_BEAM_SIZE", "1"))
WHISPER_BEST_OF     = int(os.environ.get("TYPER_BEST_OF", "5"))
# Comma-separated fallback temperatures. Whisper starts at the first;
# if the result fails the compression_ratio / logprob thresholds below,
# it retries at the next higher temperature (where best_of kicks in).
# Single value (e.g. "0.0") = pure greedy, no fallback (faster, more
# fragile to hallucinations).
WHISPER_TEMPERATURE = os.environ.get("TYPER_TEMPERATURE", "0.0,0.2,0.4,0.6,0.8,1.0")
# Fallback triggers — Whisper's defaults, exposed for tuning.
# compression_ratio_threshold: gzip(text)/len(text) above this → likely
#   stuck in a repetition loop → retry at higher T. Lower = stricter.
WHISPER_COMPRESSION_RATIO_THRESHOLD = float(os.environ.get("TYPER_COMPRESSION_RATIO", "2.4"))
# logprob_threshold: avg token log-prob below this → low-confidence
#   output → retry at higher T. Less negative = stricter.
WHISPER_LOGPROB_THRESHOLD = float(os.environ.get("TYPER_LOGPROB_THRESHOLD", "-1.0"))
# no_speech_threshold: <|nospeech|> prob above this AND logprob below
#   threshold → treat segment as silence (emit nothing). Helps with the
#   classic "Thanks for watching!" hallucination on silence.
WHISPER_NO_SPEECH_THRESHOLD = float(os.environ.get("TYPER_NO_SPEECH_THRESHOLD", "0.6"))
# Suspected-hallucination silence detection (seconds of word-level
# silence that flags a segment as hallucinated). Requires word
# timestamps, which add some compute. Set to "" to disable.
WHISPER_HALLUCINATION_SILENCE_S = os.environ.get("TYPER_HALLUCINATION_SILENCE_S", "2.0")
# Whether the model sees its own prior output as conditioning. ON gives
# better long-form coherence (consistent spelling/style across the
# buffer) but can compound errors and amplify hallucinations. We re-feed
# the same buffer every tick, so the risk is lower than in normal
# long-form transcription.
WHISPER_CONDITION_ON_PREV = os.environ.get("TYPER_CONDITION_ON_PREV", "1") == "1"
# Optional vocabulary/style prime fed to every transcription call. Use
# this to bias toward names, jargon, or formatting Whisper otherwise
# mangles (e.g. "Tokens used: pynput, MLX, Parakeet, Whisper.").
WHISPER_INITIAL_PROMPT = os.environ.get("TYPER_INITIAL_PROMPT", "") or None

# Hard cap on the rolling buffer fed to Whisper each tick. Whisper's
# context is 30 s; beyond that the model windows internally and gets
# slower per call. We cap a bit under that to keep latency bounded.
MAX_BUFFER_S = float(os.environ.get("TYPER_MAX_BUFFER_S", "28.0"))

# Minimum voiced-audio duration before we run a PREVIEW transcribe.
# Whisper happily hallucinates plausible text on tiny chunks ("Thank
# you.", "Bye."), so until we have ≥ this many seconds of voiced
# audio the HUD just stays on "Dictating…" without committing to a
# guess. The final commit-on-release path uses its own much lower
# threshold so short legitimate utterances ("yes", "no") still paste.
MIN_PREVIEW_AUDIO_S = float(os.environ.get("TYPER_MIN_PREVIEW_S", "1.0"))

# Single keyboard controller for synthetic keystrokes.
_kbd = keyboard.Controller()

_MODIFIER_KEYS = (keyboard.Key.cmd,)

def _release_modifiers():
    for k in _MODIFIER_KEYS:
        try:
            _kbd.release(k)
        except Exception:
            pass

# ── VAD CONFIG ──────────────────────────────────────────────────────────────
VAD_FRAME = 512
# Lower threshold = triggers earlier on soft phoneme attacks (s/f/th,
# unvoiced consonants) at the cost of more false positives. Silero's
# author-recommended default is 0.5; we use it here to avoid clipping
# word onsets.
VAD_THRESHOLD = float(os.environ.get("TYPER_VAD_THRESHOLD", "0.5"))
# Lead-in flushed at speech onset. Must cover the gap between actual
# phoneme onset and Silero crossing the threshold (typically 100-300 ms
# for soft attacks). 500 ms is generous; reduce if you see leading
# breath/click noise in the transcript.
VAD_PRE_ROLL_MS = int(os.environ.get("TYPER_VAD_PRE_ROLL_MS", "500"))
VAD_HANGOVER_MS = int(os.environ.get("TYPER_VAD_HANGOVER_MS", "400"))
# Minimum RMS energy required to even run the VAD on a frame. Too high
# and quiet speech onsets get gated out before Silero ever sees them.
# 0.005 keeps room ambience out but admits soft speech.
VAD_RMS_FLOOR = float(os.environ.get("TYPER_VAD_RMS_FLOOR", "0.005"))

# ── OVERLAY CONFIG ──────────────────────────────────────────────────────────
# Floating HUD that previews the live transcript next to the mouse
# cursor. Updates every push interval via main-thread dispatch.
OVERLAY_ENABLED   = os.environ.get("TYPER_OVERLAY", "1") == "1"
# Light glass HUD dimensions, matching the reference CSS:
#   min-height: 72px, padding: 0 28px, border-radius: 32px,
#   font-size: 22px (medium weight), color rgba(0,0,0,0.72)
# Wide pill shape that scales with viewport in the CSS spec — we pick
# a fixed pixel width that gives the same proportions on a 13–16" mac.
OVERLAY_W            = int(os.environ.get("TYPER_OVERLAY_W", "780"))
# OVERLAY_H is the SINGLE-LINE height — the resting state. The panel
# grows past this as the transcript wraps, capped at OVERLAY_MAX_H.
OVERLAY_H            = int(os.environ.get("TYPER_OVERLAY_H", "72"))
OVERLAY_MAX_H        = int(os.environ.get("TYPER_OVERLAY_MAX_H", "320"))
OVERLAY_FONT_SIZE    = float(os.environ.get("TYPER_OVERLAY_FONT_SIZE", "21"))
# Spotlight uses a fully-rounded pill (radius = H/2). Default to that.
OVERLAY_CORNER_RADIUS = float(os.environ.get("TYPER_OVERLAY_CORNER", "36"))
OVERLAY_PAD_X         = float(os.environ.get("TYPER_OVERLAY_PAD_X", "28"))
OVERLAY_PAD_Y         = float(os.environ.get("TYPER_OVERLAY_PAD_Y", "14"))
# Use Apple's Liquid Glass material (macOS 26 Tahoe+) instead of the
# older NSVisualEffectView blur. Set TYPER_OVERLAY_GLASS=0 to force the
# legacy path even on Tahoe (useful if a specific build of macOS has
# a Liquid Glass regression).
OVERLAY_USE_GLASS     = (
    os.environ.get("TYPER_OVERLAY_GLASS", "1") == "1" and _LIQUID_GLASS_AVAILABLE
)
# Whole-panel alpha applied on top of the vibrancy material. 1.0 = the
# material's native opacity (still translucent thanks to blur); lower
# values fade everything (background AND text) toward fully see-through.
OVERLAY_ALPHA         = float(os.environ.get("TYPER_OVERLAY_ALPHA", "1.0"))
#   "caret"  — sit next to the focused text-input caret via macOS
#              Accessibility API (best UX; requires AX permission).
#   "cursor" — anchor to the mouse cursor position.
#   "bottom" — centered at the bottom of the active screen.
OVERLAY_PLACEMENT = os.environ.get("TYPER_OVERLAY_PLACEMENT", "caret")

# ── LOGGING ──────────────────────────────────────────────────────────────────
LOG_DIR = Path(os.environ.get("TYPER_LOG_DIR", Path(__file__).resolve().parent / "logs"))
LOG_LEVEL = os.environ.get("TYPER_LOG_LEVEL", "DEBUG").upper()

def _setup_logging():
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / "typer_whisper.log"
    fmt = logging.Formatter(
        "%(asctime)s.%(msecs)03d %(levelname)-8s [%(threadName)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    root = logging.getLogger()
    root.setLevel(LOG_LEVEL)
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
    # Silence noisy third-party loggers that don't add user-visible
    # signal at DEBUG. numba dumps its SSA IR for every JIT compilation
    # (triggered by silero-vad's onnx path); httpx/httpcore dump every
    # HTTP round-trip even when we never make any. Pin them to WARNING
    # regardless of TYPER_LOG_LEVEL so our own DEBUG output stays usable.
    for noisy in (
        "numba", "httpx", "httpcore",
        "urllib3", "huggingface_hub",
        "matplotlib", "PIL",
    ):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    logging.getLogger("typer").info("Logging to %s (level=%s)", log_path, LOG_LEVEL)
    return log_path

_setup_logging()
log = logging.getLogger("typer")

# ── STATE ────────────────────────────────────────────────────────────────────
model_ready      = threading.Event()  # whisper weights warmed
model_ok         = False              # whether warm-up actually succeeded

live_active      = False
live_stop_event   = threading.Event()
# Set when the user cancels mid-dictation (e.g. Esc). Causes the final
# commit to skip the paste step entirely so nothing lands in the field.
live_cancel_event = threading.Event()
live_state_lock  = threading.Lock()

f19_held         = False
rcmd_held        = False

# Single inference thread services jobs; MLX tensors stay on one thread.
inference_q = queue.Queue()

# ── INFERENCE THREAD ─────────────────────────────────────────────────────────
def _parse_temperature(spec):
    parts = [p.strip() for p in spec.split(",") if p.strip()]
    vals = tuple(float(p) for p in parts)
    return vals[0] if len(vals) == 1 else vals

_WHISPER_TEMPERATURE_PARSED = _parse_temperature(WHISPER_TEMPERATURE)
_WHISPER_HALLUCINATION_SILENCE_PARSED = (
    float(WHISPER_HALLUCINATION_SILENCE_S) if WHISPER_HALLUCINATION_SILENCE_S else None
)

def _whisper_transcribe(audio_f32):
    """Run mlx_whisper.transcribe on a float32 mono 16 kHz array, return text."""
    kwargs = dict(
        path_or_hf_repo=MODEL_ID,
        temperature=_WHISPER_TEMPERATURE_PARSED,
        compression_ratio_threshold=WHISPER_COMPRESSION_RATIO_THRESHOLD,
        logprob_threshold=WHISPER_LOGPROB_THRESHOLD,
        no_speech_threshold=WHISPER_NO_SPEECH_THRESHOLD,
        condition_on_previous_text=WHISPER_CONDITION_ON_PREV,
        fp16=True,
        verbose=None,
    )
    if WHISPER_LANGUAGE:
        kwargs["language"] = WHISPER_LANGUAGE
    if WHISPER_INITIAL_PROMPT:
        kwargs["initial_prompt"] = WHISPER_INITIAL_PROMPT
    if _WHISPER_HALLUCINATION_SILENCE_PARSED is not None:
        kwargs["hallucination_silence_threshold"] = _WHISPER_HALLUCINATION_SILENCE_PARSED
        kwargs["word_timestamps"] = True  # required for silence detection
    # mlx_whisper exposes decoder knobs via DecodingOptions kwargs surfaced
    # through transcribe(); beam_size triggers beam search when > 1.
    if WHISPER_BEAM_SIZE > 1:
        kwargs["beam_size"] = WHISPER_BEAM_SIZE
    else:
        kwargs["best_of"] = WHISPER_BEST_OF
    res = mlx_whisper.transcribe(audio_f32, **kwargs)
    return (res.get("text") or "").strip()

def inference_worker():
    """Owns MLX state. Warms the model, then services live jobs forever."""
    global model_ok
    log.info("inference worker starting")
    try:
        mx.set_default_device(mx.gpu)
        _ = (mx.zeros((1,)) + 1).item()
        log.debug("MLX GPU warm-up ok")
    except Exception:
        log.exception("MLX init failed")
    log.info("Warming Whisper model: %s", MODEL_ID)
    log.info("(first run downloads ~1.5 GB for large-v3; cached after)")
    t0 = time.time()
    try:
        # 1 s of silence forces weight load + compile graph.
        _ = _whisper_transcribe(np.zeros(SAMPLE_RATE, dtype=np.float32))
        model_ok = True
        log.info("Whisper ready in %.1fs", time.time() - t0)
        try:
            _get_vad_model()
        except Exception:
            log.exception("VAD load failed (continuing without VAD)")
    except Exception:
        log.exception("model warm-up failed — dictation will be unavailable")
    finally:
        model_ready.set()
    if not model_ok:
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
    """Streaming Silero-VAD gate. Returns only voiced audio with pre-roll
    lead-in and post-speech hangover, so we don't feed silence to Whisper
    (which loves to hallucinate on silence)."""
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
                    out.append(frame)
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

# ── Persistent audio input ──────────────────────────────────────────────────
# CoreAudio takes ~100-300 ms to open an InputStream the first time the
# device is touched, plus a smaller open-cost on each subsequent open.
# Opening fresh per-session (the old behavior) means the first samples
# after Right-⌘ press are silently dropped while the device spins up.
# Solution: open once at app startup and keep the stream live for the
# process lifetime. The callback discards samples between sessions so
# we don't grow the queue or eat CPU during idle.
_persistent_audio_q = queue.Queue()
_audio_session_active = threading.Event()
_audio_input_stream = None
_audio_cb_stats = {"calls": 0, "frames": 0, "status_warns": 0}

def _open_persistent_audio_input():
    """Open the mic once on app startup and keep it open for the
    lifetime of the process. Safe to call repeatedly — only the first
    call actually opens the device."""
    global _audio_input_stream
    if _audio_input_stream is not None:
        return _audio_input_stream

    def _cb(indata, frames, time_info, status):
        _audio_cb_stats["calls"] += 1
        _audio_cb_stats["frames"] += frames
        if status:
            _audio_cb_stats["status_warns"] += 1
            log.warning("audio callback status: %s", status)
        # Cheap no-op between sessions — no allocation, no enqueue.
        if not _audio_session_active.is_set():
            return
        _persistent_audio_q.put(indata[:, 0].copy())

    try:
        _audio_input_stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
            blocksize=int(SAMPLE_RATE * 0.1),
            callback=_cb,
        )
        _audio_input_stream.start()
        log.info("audio input opened (persistent, sr=%d)", SAMPLE_RATE)
    except Exception:
        log.exception("failed to open persistent audio input — "
                      "will fall back to per-session open in _do_live_stream")
        _audio_input_stream = None
    return _audio_input_stream

def _drain_persistent_audio_q():
    """Discard any audio sitting in the queue. Called at session start
    so we don't process stale pre-session samples (shouldn't happen
    because the callback discards when inactive, but cheap insurance)."""
    while True:
        try:
            _persistent_audio_q.get_nowait()
        except queue.Empty:
            return

def _get_vad_model():
    global vad_model
    with vad_model_lock:
        if vad_model is None:
            log.info("loading Silero VAD (onnx)")
            t0 = time.time()
            vad_model = load_silero_vad(onnx=True)
            log.info("Silero VAD ready in %.2fs", time.time() - t0)
        return vad_model


# ── OVERLAY (floating HUD) ───────────────────────────────────────────────────
# Module-level system-wide AX element. Created lazily and reused: cheap,
# but cache it anyway to avoid the repeated cross-process IPC.
_ax_system = AXUIElementCreateSystemWide()


def _rounded_mask_image(radius):
    """Build a 9-slice NSImage that clips an NSVisualEffectView to a
    rounded-rect shape.

    Why this exists: NSVisualEffectView's vibrancy material is composited
    OUTSIDE the layer tree, so the usual `layer.cornerRadius` does not
    actually clip it — the material spills into the rectangle's corners
    and shows up as bright triangles against dark wallpapers (the bug
    the user reported). The documented Apple fix is `setMaskImage_`
    with a 9-part resizable rounded-corner image: the corners are
    drawn 1:1, the edges and center stretch to fill any panel size.
    """
    diameter = radius * 2 + 1
    img = NSImage.alloc().initWithSize_(NSMakeSize(diameter, diameter))
    img.lockFocus()
    path = NSBezierPath.bezierPathWithRoundedRect_xRadius_yRadius_(
        NSMakeRect(0, 0, diameter, diameter), radius, radius
    )
    NSColor.blackColor().setFill()
    path.fill()
    img.unlockFocus()
    img.setCapInsets_(NSEdgeInsetsMake(radius, radius, radius, radius))
    img.setResizingMode_(NSImageResizingModeStretch)
    return img

def _caret_screen_rect():
    """Return (x, y, w, h) of the focused-app's text caret in NSScreen
    coordinates (origin bottom-left of primary display), or None when
    unavailable.

    Failure modes that legitimately return None:
      • No Accessibility permission for our Python binary.
      • Focused element isn't a text field (no AXSelectedTextRange).
      • App doesn't implement AXBoundsForRange (some Electron / GPU UIs).
      • Caret is in an off-screen scrolled region.
    Callers must fall back to the mouse cursor.
    """
    try:
        err, focused = AXUIElementCopyAttributeValue(
            _ax_system, kAXFocusedUIElementAttribute, None
        )
        if err or focused is None:
            return None
        err, range_val = AXUIElementCopyAttributeValue(
            focused, kAXSelectedTextRangeAttribute, None
        )
        if err or range_val is None:
            return None
        # AXBoundsForRange takes the AXValue<CFRange> we just retrieved
        # and returns an AXValue<CGRect>. The selection's range is fine
        # even when length=0 (pure caret): bounds are reported as a
        # zero-width rect at the caret's pixel position.
        err, bounds_val = AXUIElementCopyParameterizedAttributeValue(
            focused, "AXBoundsForRange", range_val, None
        )
        if err or bounds_val is None:
            return None
        ok, rect = AXValueGetValue(bounds_val, kAXValueCGRectType, None)
        if not ok:
            return None
        # Some apps (and the case of "no text field focused at all")
        # return a degenerate (0,0,0,0) rect rather than an error. Treat
        # zero-sized rects as "no caret available" so we fall back.
        if rect.size.width == 0 and rect.size.height == 0:
            return None
        # AX uses the global display coordinate space whose origin sits
        # at the top-left of the primary display (y grows down). NSScreen
        # uses the primary display's bottom-left (y grows up). Flip Y
        # relative to the primary screen's height.
        main = NSScreen.mainScreen()
        if main is None:
            return None
        main_h = main.frame().size.height
        x = rect.origin.x
        # Subtract caret height too so y points to the BOTTOM of the
        # caret in NS coords (which is where we want to anchor the
        # overlay's top edge before going down by OVERLAY_H).
        y = main_h - rect.origin.y - rect.size.height
        return (float(x), float(y), float(rect.size.width), float(rect.size.height))
    except Exception:
        log.debug("caret lookup failed", exc_info=True)
        return None


class _OverlayController(NSObject):
    """Owns a non-activating NSPanel that floats above all windows and
    shows the live transcript. Methods are invoked via
    performSelectorOnMainThread_ from any worker thread so AppKit calls
    always land on the main run loop, which is the only thread that may
    mutate the view hierarchy safely.

    NSWindowStyleMaskNonactivatingPanel + becomesKeyOnlyIfNeeded ensures
    the user's currently-focused app NEVER loses key-window status when
    we show/hide/move the panel — critical because we're synthesizing
    keystrokes into that app.
    """

    def init(self):
        self = objc.super(_OverlayController, self).init()
        if self is None:
            return None
        # ── Panel ──────────────────────────────────────────────────────
        rect = NSMakeRect(0, 0, OVERLAY_W, OVERLAY_H)
        style = NSWindowStyleMaskNonactivatingPanel
        self.panel = NSPanel.alloc().initWithContentRect_styleMask_backing_defer_(
            rect, style, NSBackingStoreBuffered, False
        )
        self.panel.setLevel_(NSFloatingWindowLevel)
        self.panel.setOpaque_(False)
        self.panel.setBackgroundColor_(NSColor.clearColor())
        self.panel.setAlphaValue_(OVERLAY_ALPHA)
        self.panel.setHasShadow_(True)
        self.panel.setBecomesKeyOnlyIfNeeded_(True)
        self.panel.setHidesOnDeactivate_(False)
        self.panel.setIgnoresMouseEvents_(True)
        self.panel.setCollectionBehavior_(
            NSWindowCollectionBehaviorCanJoinAllSpaces
            | NSWindowCollectionBehaviorStationary
            | NSWindowCollectionBehaviorIgnoresCycle
        )

        # ── Background surface ─────────────────────────────────────────
        # Two paths:
        #   Liquid Glass (macOS 26+) — true refractive material that
        #     bends light from the wallpaper behind it. Has first-class
        #     cornerRadius and a real contentView, so no mask-image hack.
        #   Vibrancy fallback — NSVisualEffectView + vibrant-light, with
        #     a 9-slice mask image to clip the blur (cornerRadius alone
        #     doesn't actually clip the vibrancy material).
        # Both surfaces present the same `content` view to the rest of
        # init() so the subview construction below is identical.
        if OVERLAY_USE_GLASS:
            bg = NSGlassEffectView.alloc().initWithFrame_(rect)
            # Regular style is Apple's everyday Liquid Glass material —
            # already light and refractive on its own. Clear was making
            # the panel read DARKER as we added white tint (the tint
            # interacts oddly with the Clear style's compositing).
            try:
                bg.setStyle_(NSGlassEffectViewStyleRegular)
            except Exception:
                pass
            bg.setCornerRadius_(OVERLAY_CORNER_RADIUS)
            # No tint — let the Regular material show through at its
            # native brightness. If we ever want to bias the surface
            # toward a particular hue, set a near-clearColor here.
            bg.setTintColor_(NSColor.clearColor())
            bg.setAutoresizingMask_(NSViewWidthSizable | NSViewHeightSizable)
            # NSGlassEffectView ships with no contentView by default —
            # we must install one explicitly. All our subviews (dot,
            # text, mic, esc) sit on this plain NSView so they render
            # ABOVE the glass material, not refracted by it.
            content = NSView.alloc().initWithFrame_(rect)
            content.setAutoresizingMask_(NSViewWidthSizable | NSViewHeightSizable)
            bg.setContentView_(content)
            self.panel.setContentView_(bg)
            log.debug("overlay background: NSGlassEffectView (Liquid Glass)")
        else:
            bg = NSVisualEffectView.alloc().initWithFrame_(rect)
            bg.setMaterial_(NSVisualEffectMaterialPopover)
            bg.setBlendingMode_(NSVisualEffectBlendingModeBehindWindow)
            bg.setState_(NSVisualEffectStateActive)
            bg.setAppearance_(
                NSAppearance.appearanceNamed_(NSAppearanceNameVibrantLight)
            )
            bg.setAutoresizingMask_(NSViewWidthSizable | NSViewHeightSizable)
            bg.setMaskImage_(_rounded_mask_image(OVERLAY_CORNER_RADIUS))
            bg.setWantsLayer_(True)
            layer = bg.layer()
            layer.setCornerRadius_(OVERLAY_CORNER_RADIUS)
            try:
                layer.setCornerCurve_(kCACornerCurveContinuous)
            except Exception:
                pass
            layer.setMasksToBounds_(True)
            layer.setBorderWidth_(1.0)
            layer.setBorderColor_(
                NSColor.colorWithCalibratedWhite_alpha_(0.0, 0.08).CGColor()
            )
            self.panel.setContentView_(bg)
            content = bg  # legacy path: subviews go straight on the VE view
            log.debug("overlay background: NSVisualEffectView (legacy vibrancy)")
        self.panel.invalidateShadow()
        self._bg = bg
        self._content = content

        # ── Leading mic icon (SF Symbol) ───────────────────────────────
        # Spotlight-style layout: one icon on the left, text fills the
        # rest. No right-side controls. SF Symbol "mic" at body weight,
        # tinted to match the placeholder text color.
        mic_size = 26.0
        try:
            mic_cfg = NSImageSymbolConfiguration.configurationWithPointSize_weight_(
                mic_size, NSFontWeightRegular
            )
            mic_img = NSImage.imageWithSystemSymbolName_accessibilityDescription_(
                "mic", "Dictation"
            )
            if mic_img is not None:
                mic_img = mic_img.imageWithSymbolConfiguration_(mic_cfg)
        except Exception:
            mic_img = None
        mic_view_w = mic_size + 4
        mic_view_h = mic_size + 4
        mic_x = OVERLAY_PAD_X
        mic_y = (OVERLAY_H - mic_view_h) / 2.0
        mic_view = NSImageView.alloc().initWithFrame_(
            NSMakeRect(mic_x, mic_y, mic_view_w, mic_view_h)
        )
        if mic_img is not None:
            mic_view.setImage_(mic_img)
        try:
            mic_view.setContentTintColor_(
                NSColor.colorWithCalibratedWhite_alpha_(0.0, 0.55)
            )
        except Exception:
            pass
        content.addSubview_(mic_view)
        self._mic = mic_view

        # ── Transcript label ───────────────────────────────────────────
        # Spotlight-style placeholder typography: SF Pro Regular at the
        # body size, muted near-black. Active transcript uses the same
        # weight at full strength so the visual rhythm doesn't change
        # mid-sentence.
        text_x = mic_x + mic_view_w + 12.0
        text_w = OVERLAY_W - text_x - OVERLAY_PAD_X
        line_h = OVERLAY_FONT_SIZE * 1.35
        text_y = (OVERLAY_H - line_h) / 2.0
        self.label = NSTextField.alloc().initWithFrame_(
            NSMakeRect(text_x, text_y, text_w, line_h)
        )
        self.label.setEditable_(False)
        self.label.setSelectable_(False)
        self.label.setBezeled_(False)
        self.label.setDrawsBackground_(False)
        # Regular weight, matching the Spotlight Search placeholder.
        self.label.setFont_(
            NSFont.systemFontOfSize_weight_(OVERLAY_FONT_SIZE, NSFontWeightRegular)
        )
        self.label.setTextColor_(
            NSColor.colorWithCalibratedWhite_alpha_(0.0, 0.55)
        )
        self.label.setStringValue_("")
        cell = self.label.cell()
        cell.setLineBreakMode_(NSLineBreakByWordWrapping)
        cell.setWraps_(True)
        cell.setUsesSingleLineMode_(False)
        cell.setScrollable_(False)
        content.addSubview_(self.label)
        self._text_x = text_x
        self._text_max_w = text_w
        self._text_line_h = line_h

        # State for show_/setText_ transitions.
        self._current_text = ""
        return self

    # ── main-thread entry points (called via performSelectorOnMainThread_) ──
    def show_(self, status):
        """First display when a dictation session begins. The status
        string ('Listening…') is set as muted placeholder text."""
        # Reset to single-line height BEFORE positioning. _position_near
        # _cursor uses OVERLAY_H to compute the anchor point, but the
        # panel may still be at the (taller) frame from a previous
        # session. Without this reset, the panel snaps to the right
        # X but is vertically misaligned relative to the caret.
        cur = self.panel.frame()
        if int(round(cur.size.height)) != OVERLAY_H:
            top_y = cur.origin.y + cur.size.height
            self.panel.setFrame_display_(
                NSMakeRect(cur.origin.x, top_y - OVERLAY_H,
                           cur.size.width, float(OVERLAY_H)),
                False,
            )
        self._position_near_cursor()
        self._current_text = ""
        self._render(status or "Listening…", placeholder=True)
        self.panel.orderFrontRegardless()

    def hide_(self, _ignored):
        self.panel.orderOut_(None)
        self._current_text = ""

    def setText_(self, text):
        """Live transcript update. Empty/None → 'Dictating…' placeholder
        (we know capture is active, we just don't have words yet)."""
        if text:
            self._current_text = text
            self._render(text, placeholder=False)
        else:
            self._current_text = ""
            self._render("Dictating…", placeholder=True)

    def setState_(self, state):
        """Swap the placeholder text. 'listening' → 'Listening…',
        anything else → 'Dictating…'. No-op if real transcript text is
        already showing."""
        if self._current_text:
            return
        self._render(
            "Listening…" if state == "listening" else "Dictating…",
            placeholder=True,
        )

    @objc.python_method
    def _render(self, text, placeholder):
        # Both placeholder and live-transcript text share the SAME hue
        # as the mic icon (black @ 55% alpha) for a single, calm
        # typographic voice across the whole HUD.
        color = NSColor.colorWithCalibratedWhite_alpha_(0.0, 0.55)
        self.label.setTextColor_(color)
        self.label.setStringValue_(text)
        self._resize_to_fit(text)

    @objc.python_method
    def _measure_text_height(self, text):
        """Wrap-aware height for `text` rendered into a column of width
        self._text_max_w in the label's current font. Returns at least
        one line's worth of height even for empty text."""
        line_h = self._text_line_h
        if not text:
            return line_h
        try:
            attrs = {NSFontAttributeName: self.label.font()}
            s = objc.lookUpClass("NSString").stringWithString_(text)
            rect = s.boundingRectWithSize_options_attributes_(
                NSMakeSize(self._text_max_w, 1.0e6),
                NSStringDrawingUsesLineFragmentOrigin,
                attrs,
            )
            # Round up so we never under-allocate by a sub-pixel.
            return max(line_h, float(rect.size.height) + 1.0)
        except Exception:
            log.debug("text measure failed", exc_info=True)
            return line_h

    @objc.python_method
    def _resize_to_fit(self, text):
        """Grow the panel vertically to fit the wrapped text. Stays at
        OVERLAY_H for single-line content, expands up to OVERLAY_MAX_H
        as the transcript wraps to more lines. The panel's TOP edge is
        the anchor — growth happens downward so the caret-anchored
        position above the focused input stays stable."""
        line_h = self._text_line_h
        # Vertical padding on either side of the first line (used to
        # vertically center the resting single-line state).
        top_pad = (OVERLAY_H - line_h) / 2.0
        bot_pad = top_pad

        measured_h = self._measure_text_height(text)
        needed = top_pad + measured_h + bot_pad
        new_h = int(round(max(float(OVERLAY_H), min(float(OVERLAY_MAX_H), needed))))

        old_frame = self.panel.frame()
        old_top_y = old_frame.origin.y + old_frame.size.height
        if int(round(old_frame.size.height)) != new_h:
            new_frame = NSMakeRect(
                old_frame.origin.x, old_top_y - new_h,
                old_frame.size.width, float(new_h),
            )
            # No animate=True — text updates land every ~0.35 s, and
            # animating each one creates visible churn. Instant resize
            # reads as the panel growing naturally with the content.
            self.panel.setFrame_display_(new_frame, True)

        # Anchor mic icon to the FIRST line so it doesn't drift down
        # the panel as the transcript grows.
        if self._mic is not None:
            mic_frame = self._mic.frame()
            mic_h = mic_frame.size.height
            mic_top_pad = (OVERLAY_H - mic_h) / 2.0
            self._mic.setFrame_(NSMakeRect(
                mic_frame.origin.x,
                new_h - mic_top_pad - mic_h,
                mic_frame.size.width, mic_h,
            ))

        # Resize the label to span from top-pad to bottom-pad. Text
        # starts at the top of the frame and wraps downward, so the
        # frame's TOP must align with where the first line lives.
        label_h = max(line_h, new_h - top_pad - bot_pad)
        self.label.setFrame_(NSMakeRect(
            self._text_x,
            bot_pad,
            self._text_max_w,
            label_h,
        ))

    @objc.python_method
    def _position_near_cursor(self):
        screen = NSScreen.mainScreen()
        vf = (screen or NSScreen.screens()[0]).visibleFrame()

        if OVERLAY_PLACEMENT == "bottom":
            x = vf.origin.x + (vf.size.width - OVERLAY_W) / 2
            y = vf.origin.y + 40
            self.panel.setFrameOrigin_(NSMakePoint(x, y))
            return

        if OVERLAY_PLACEMENT == "caret":
            caret = _caret_screen_rect()
            if caret is not None:
                # caret = (x, y_bottom, w, h) in NSScreen coords; we want
                # the overlay to hang JUST BELOW the caret line.
                cx, cy, cw, ch = caret
                # Anchor horizontally to the caret column, but indent a
                # bit so the overlay's left edge doesn't sit on top of
                # the caret itself.
                x = cx + 6
                # Overlay's bottom = caret's bottom - small pad - OVERLAY_H
                y = cy - 6 - OVERLAY_H
                # If we'd clip below the screen, place the overlay ABOVE
                # the caret line instead (top of overlay sits above the
                # caret's top edge by a small pad).
                if y < vf.origin.y:
                    y = cy + ch + 6
                # Horizontal clamp.
                if x + OVERLAY_W > vf.origin.x + vf.size.width:
                    x = vf.origin.x + vf.size.width - OVERLAY_W - 8
                if x < vf.origin.x:
                    x = vf.origin.x + 8
                self.panel.setFrameOrigin_(NSMakePoint(x, y))
                return
            # Caret lookup failed → fall through to cursor placement so
            # the overlay still appears somewhere useful.
            log.debug("caret unavailable, falling back to mouse cursor")

        # "cursor" placement (or caret fallback).
        loc = NSEvent.mouseLocation()
        x = loc.x + 18
        y = loc.y - OVERLAY_H - 18
        if x + OVERLAY_W > vf.origin.x + vf.size.width:
            x = loc.x - OVERLAY_W - 18
        if y < vf.origin.y:
            y = loc.y + 18
        self.panel.setFrameOrigin_(NSMakePoint(x, y))


_overlay_controller = None  # set in main() on the main thread

def _overlay_call(selector, arg=None):
    """Dispatch an overlay method to the main run loop. Safe to call
    from any thread; returns immediately (waitUntilDone=False)."""
    if _overlay_controller is None or not OVERLAY_ENABLED:
        return
    _overlay_controller.performSelectorOnMainThread_withObject_waitUntilDone_(
        selector, arg, False
    )

def overlay_show(status=None):
    _overlay_call("show:", status)

def overlay_hide():
    _overlay_call("hide:", None)

def overlay_set_text(text):
    _overlay_call("setText:", text or "")

def overlay_set_state(state):
    """state ∈ {'listening','dictating'} — only affects the placeholder
    text. Ignored once real transcript text is showing."""
    _overlay_call("setState:", state)


# ── LIVE STREAMING ────────────────────────────────────────────────────────────
def _paste_text(text):
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
    """Backspace over the diverged tail of `last`, paste the new tail of
    `current`. Returns the canonical text now in the field."""
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
    """Whisper-MLX pseudo-streaming.

    Captures voiced audio into a rolling buffer; every LIVE_PUSH_INTERVAL_S
    re-runs Whisper on the whole buffer and diff-pastes the new suffix.
    Runs ON THE INFERENCE THREAD (MLX owner).

    Audio comes from the PERSISTENT InputStream opened once at app
    startup (see _open_persistent_audio_input). Falls back to opening
    a per-session stream only if the persistent one failed to come up.
    """
    log.info("live stream begin")
    use_persistent = _audio_input_stream is not None
    if use_persistent:
        audio_q = _persistent_audio_q
        # Discard anything captured between sessions (the callback
        # should already be no-oping, but cheap insurance against any
        # in-flight buffer at the moment we flip the flag).
        _drain_persistent_audio_q()
        cb_stats = _audio_cb_stats
        fallback_cb = None
    else:
        # Fallback path: open a fresh stream just for this session.
        # ~100-300 ms of mic open latency, used only if the persistent
        # stream failed at startup (e.g. permissions denied at that
        # moment and then granted afterward — user restart fixes it).
        audio_q = queue.Queue()
        cb_stats = {"frames": 0, "calls": 0, "status_warns": 0}
        def fallback_cb(indata, frames, time_info, status):
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

    last_pasted = ""       # only used in stream mode (diff state)
    last_preview = ""      # what's currently shown in the overlay
    iters = 0
    max_buf_samples = int(MAX_BUFFER_S * SAMPLE_RATE)
    voiced_buf = np.zeros(0, dtype=np.float32)

    def _ingest_audio():
        """Drain the audio queue into voiced_buf (with VAD gating).
        Returns True if any new voiced audio was added this call."""
        nonlocal voiced_buf
        buf = []
        while True:
            try:
                buf.append(audio_q.get_nowait())
            except queue.Empty:
                break
        if not buf:
            return False
        samples = np.concatenate(buf)
        raw_n = len(samples)
        if gate is not None:
            samples = gate.process(samples)
        if not len(samples):
            return False
        log.debug("drain: raw=%d voiced=%d (%.2fs voiced)",
                  raw_n, len(samples), len(samples) / SAMPLE_RATE)
        voiced_buf = np.concatenate([voiced_buf, samples])
        if len(voiced_buf) > max_buf_samples:
            drop = len(voiced_buf) - max_buf_samples
            log.debug("buffer cap hit, dropping %d oldest samples", drop)
            voiced_buf = voiced_buf[drop:]
        return True

    def _transcribe_buffer(min_audio_s=0.25):
        """Run Whisper on the current voiced_buf. Returns the text, or
        None if the buffer is shorter than `min_audio_s` (whisper
        hallucinates on tiny chunks) or inference failed."""
        if len(voiced_buf) < int(min_audio_s * SAMPLE_RATE):
            return None
        t_infer = time.time()
        try:
            text = _whisper_transcribe(voiced_buf)
        except Exception:
            log.exception("whisper transcribe failed this tick")
            return None
        infer_ms = (time.time() - t_infer) * 1000.0
        log.debug("whisper tick: buf=%.2fs infer=%.0fms text=%r",
                  len(voiced_buf) / SAMPLE_RATE, infer_ms, text)
        return text

    # Tracked so _commit_final can decide whether to rerun whisper. If
    # the user releases right after a preview tick finished, the buffer
    # has barely grown and we can paste the cached preview text instead
    # of paying ~1-2 s for another large-v3 inference. Threshold below
    # is in samples (0.4 s @ 16 kHz).
    state = {
        "transitioned": False,
        "preview_buf_samples": 0,   # voiced_buf length at last successful preview
    }
    REUSE_PREVIEW_THRESHOLD_SAMPLES = int(0.4 * SAMPLE_RATE)

    def _tick_preview():
        """Per-interval tick during recording. Ingests audio, runs a
        whisper preview, and updates the overlay only. NEVER touches the
        user's input field — that happens once on release in batch mode,
        or via the legacy diff path in stream mode."""
        nonlocal last_preview, last_pasted
        _ingest_audio()
        # Flip the placeholder Listening… → Dictating… on the first tick
        # where VAD has emitted any voiced audio. Signals to the user
        # that we've heard them even before whisper produces words.
        if not state["transitioned"] and len(voiced_buf) > 0:
            state["transitioned"] = True
            overlay_set_state("dictating")
        # Hold off on running whisper until we have at least
        # MIN_PREVIEW_AUDIO_S of voiced audio. Whisper produces
        # plausible-but-wrong text on shorter chunks ("Thank you.",
        # "Bye.", "Subtitles by …") and showing that in the HUD just
        # confuses the user. Final commit-on-release uses its own much
        # lower threshold so short legitimate utterances still paste.
        text = _transcribe_buffer(min_audio_s=MIN_PREVIEW_AUDIO_S)
        if text is None:
            return
        state["preview_buf_samples"] = len(voiced_buf)
        if TYPER_MODE == "stream":
            # Legacy: diff-paste into the focused field on every tick.
            if text != last_pasted:
                last_pasted = _emit_diff(text, last_pasted)
            return
        # batch mode: update overlay only.
        if text != last_preview:
            last_preview = text
            overlay_set_text(text)

    def _commit_final():
        """Called after the user releases. In batch mode we want the
        paste to feel instant: if the most recent preview already saw
        essentially all the audio we captured, reuse its text instead
        of paying ~1-2 s for another whisper pass on the same buffer.
        Only retranscribe when meaningful new audio has come in since."""
        nonlocal last_preview, last_pasted
        # If the user hit Esc, drop everything: no transcribe, no paste,
        # no overlay revisions — just hide the HUD and bail.
        if live_cancel_event.is_set():
            log.info("commit-final: cancelled, skipping paste")
            overlay_hide()
            return
        _ingest_audio()
        new_samples = len(voiced_buf) - state["preview_buf_samples"]
        if (TYPER_MODE != "stream"
                and last_preview
                and new_samples < REUSE_PREVIEW_THRESHOLD_SAMPLES):
            text = last_preview
            log.info(
                "commit-final: reusing preview (%.2fs new audio < %.2fs threshold)",
                new_samples / SAMPLE_RATE,
                REUSE_PREVIEW_THRESHOLD_SAMPLES / SAMPLE_RATE,
            )
        else:
            t0 = time.time()
            text = _transcribe_buffer()
            log.info(
                "commit-final: re-ran whisper (%.2fs new audio, %.0fms infer)",
                max(0, new_samples) / SAMPLE_RATE,
                (time.time() - t0) * 1000.0,
            )
        if text is None:
            overlay_hide()
            return
        if TYPER_MODE == "stream":
            if text != last_pasted:
                last_pasted = _emit_diff(text, last_pasted)
        else:
            # batch mode: paste the entire final transcript in one shot.
            # Overlay shows it briefly so the user can confirm before it
            # disappears. We hide AFTER the paste so a slow paste doesn't
            # leave a stale-looking overlay.
            #
            # Append a trailing space so two back-to-back dictation
            # sessions don't stick together (e.g. "hello there" then
            # "how are you" would otherwise paste as
            # "hello therehow are you"). Strip first to avoid
            # double-spacing if whisper already produced a trailing
            # space, then add exactly one.
            payload = text.rstrip() + " "
            overlay_set_text(text)
            _paste_text(payload)
            last_pasted = payload
        overlay_hide()

    t_start = time.time()
    try:
        log.info("stream cfg: mode=%s model=%s push=%.2fs beam=%d temp=%s "
                 "best_of=%d cond_prev=%s lang=%s halluc_silence=%s prompt=%r",
                 TYPER_MODE, MODEL_ID, LIVE_PUSH_INTERVAL_S, WHISPER_BEAM_SIZE,
                 WHISPER_TEMPERATURE, WHISPER_BEST_OF,
                 WHISPER_CONDITION_ON_PREV, WHISPER_LANGUAGE or "auto",
                 _WHISPER_HALLUCINATION_SILENCE_PARSED, WHISPER_INITIAL_PROMPT)
        overlay_show("Listening…")
        # Activate the persistent callback so it starts enqueueing
        # samples; the device itself is already open and primed, so
        # the FIRST samples after this flag flip are real speech, not
        # CoreAudio open-stream silence.
        _audio_session_active.set()
        try:
            if not use_persistent:
                # Fallback: spin up a per-session stream now.
                fallback_stream = sd.InputStream(
                    samplerate=SAMPLE_RATE,
                    channels=1,
                    dtype="float32",
                    blocksize=int(SAMPLE_RATE * 0.1),
                    callback=fallback_cb,
                )
                fallback_stream.start()
                log.debug("audio input opened (per-session fallback) sr=%d", SAMPLE_RATE)
            else:
                log.debug("audio input enabled (persistent stream)")
            while not live_stop_event.is_set():
                # wait() returns immediately when the user releases, instead
                # of finishing the full LIVE_PUSH_INTERVAL_S sleep — keeps
                # the final whisper pass close to the release event.
                if live_stop_event.wait(LIVE_PUSH_INTERVAL_S):
                    break
                iters += 1
                t_tick = time.time()
                _tick_preview()
                tick_ms = (time.time() - t_tick) * 1000.0
                try:
                    active_mb = mx.get_active_memory() / (1024 * 1024)
                    peak_mb = mx.get_peak_memory() / (1024 * 1024)
                    log.debug("tick %d: preview=%.0fms mlx_active=%.0fMB peak=%.0fMB",
                              iters, tick_ms, active_mb, peak_mb)
                except Exception:
                    log.debug("tick %d: preview=%.0fms", iters, tick_ms)
            log.debug("commit-final after %d iters", iters)
            _commit_final()
        finally:
            # Stop the callback from collecting more samples; the
            # persistent stream itself stays OPEN for the next session.
            _audio_session_active.clear()
            if not use_persistent and 'fallback_stream' in dir():
                try:
                    fallback_stream.stop()
                    fallback_stream.close()
                except Exception:
                    log.debug("fallback stream close failed", exc_info=True)
    except Exception:
        log.exception("stream error")
        overlay_hide()
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
        if not model_ready.is_set() or not model_ok:
            log.warning("model not loaded; ignoring start")
            return
        live_stop_event.clear()
        live_cancel_event.clear()
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

def cancel_live_dictation():
    """Stop dictation AND mark this session as cancelled so the final
    commit skips the paste. Used by Esc keybinding so the user can back
    out of a hold-to-talk session without pasting whatever they said."""
    if not live_active:
        return
    log.info("✖ cancel")
    live_cancel_event.set()
    stop_live_dictation()

# ── KEYBOARD LISTENER ────────────────────────────────────────────────────────
def _toggle_live():
    if live_active:
        stop_live_dictation()
    else:
        start_live_dictation()

def on_press(key):
    global f19_held, rcmd_held
    if key == keyboard.Key.cmd_r:
        if not rcmd_held:
            rcmd_held = True
            log.debug("right-cmd press → hold-to-talk start")
            threading.Thread(target=start_live_dictation, daemon=True,
                             name="start-live").start()
    elif key == keyboard.Key.f19:
        if not f19_held:
            f19_held = True
            log.debug("F19 press → toggle (active=%s)", live_active)
            threading.Thread(target=_toggle_live, daemon=True,
                             name="toggle-live").start()
    elif key == keyboard.Key.esc and live_active:
        # The "esc" pill in the HUD advertises this — cancel the
        # in-flight session, drop the captured audio, don't paste.
        log.debug("Esc press → cancel dictation")
        threading.Thread(target=cancel_live_dictation, daemon=True,
                         name="cancel-live").start()

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
║   Typer — Whisper-MLX live typing           ║
╠══════════════════════════════════════════════╣
║  Hold Right-⌘ → speak → release to stop     ║
║  Tap F19 → start, tap again → stop          ║
║  Ctrl+C to quit                              ║
╚══════════════════════════════════════════════╝
""")
    log.info("Typer (whisper) starting (pid=%d, python=%s)", os.getpid(), sys.version.split()[0])
    threading.Thread(target=inference_worker, daemon=True,
                     name="inference").start()
    log.info("loading model in background...")
    model_ready.wait()
    if not model_ok:
        log.critical("model load failed — exiting")
        sys.exit(1)
    # Open the mic ONCE now and keep it open for the rest of the
    # process. CoreAudio's open-stream latency (~100-300 ms first call)
    # otherwise gets paid on every Right-⌘ press and silently swallows
    # the first words of each session.
    _open_persistent_audio_input()

    # Magic Mouse double-tap-and-hold trigger. Same semantics as
    # Right-⌘: hold to dictate, release to stop. start() returns False
    # if no Magic Mouse is connected or MultitouchSupport can't load —
    # we just log it and keep the keyboard triggers running.
    def _mouse_press_start():
        log.debug("mouse double-tap-hold start")
        threading.Thread(target=start_live_dictation, daemon=True,
                         name="mouse-start-live").start()
    def _mouse_press_end():
        log.debug("mouse double-tap-hold release")
        threading.Thread(target=stop_live_dictation, daemon=True,
                         name="mouse-stop-live").start()
    mouse_detector = _MouseGestureDetector(_mouse_press_start, _mouse_press_end)
    mouse_detector.start()

    log.info("🟢 ready (mode=%s). Hold Right-⌘ to talk, or tap F19 to toggle.",
             TYPER_MODE)

    def _cleanup():
        log.info("cleanup")
        if live_active:
            stop_live_dictation()
        overlay_hide()
        # Stop the persistent input stream so CoreAudio releases the
        # device promptly on exit. os._exit (used by our SIGINT handler)
        # would skip this otherwise.
        if _audio_input_stream is not None:
            try:
                _audio_input_stream.stop()
                _audio_input_stream.close()
            except Exception:
                log.debug("audio stream close failed", exc_info=True)
        try:
            mouse_detector.stop()
        except Exception:
            log.debug("mouse detector stop failed", exc_info=True)
    atexit.register(_cleanup)

    # Hard SIGINT / SIGTERM handlers. AppHelper.runEventLoop(installInterrupt
    # =True) installs a Mach-port-based handler that only stops the run loop
    # on the NEXT main-thread event — when the loop is idle (our normal
    # state, since we get no UI events until the user touches the HUD), the
    # signal is queued but never delivered, so Ctrl+C appears to do nothing.
    # We bypass that by installing our own handlers that force-exit. We do
    # the kill via os._exit AFTER cleanup so atexit handlers still run and
    # we don't get stuck waiting for the run loop to wake up.
    def _force_exit(signame):
        def _h(*_):
            log.info("%s — force exit", signame)
            try:
                _cleanup()
            finally:
                # os._exit skips Python's normal shutdown (which would try
                # to join the AppKit run loop thread — i.e. ourselves — and
                # hang). atexit handlers we care about already ran above.
                os._exit(0)
        return _h
    signal.signal(signal.SIGINT,  _force_exit("SIGINT"))
    signal.signal(signal.SIGTERM, _force_exit("SIGTERM"))

    # The keyboard.Listener spawns its own Quartz event-tap thread, so we
    # can start it and leave the main thread free to run AppKit's run
    # loop — which is required for the floating overlay panel. Updates
    # are marshalled onto this main loop via performSelectorOnMainThread_.
    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()

    global _overlay_controller
    if OVERLAY_ENABLED:
        # NSApplication must be initialized on the main thread before any
        # AppKit objects are created. The controller's window is built in
        # init(), so it lives on the main thread from birth.
        NSApplication.sharedApplication()
        _overlay_controller = _OverlayController.alloc().init()
        log.info("overlay initialized (%dx%d, placement=%s)",
                 OVERLAY_W, OVERLAY_H, OVERLAY_PLACEMENT)
        # Schedule a low-frequency timer that yields back to the Python
        # interpreter on the main thread. CFRunLoopRun otherwise sleeps in
        # mach_msg, where pending Python signal handlers can't fire. The
        # timer body is empty — its only purpose is the periodic wake-up.
        try:
            from Foundation import NSTimer
            NSTimer.scheduledTimerWithTimeInterval_repeats_block_(
                0.25, True, lambda _t: None
            )
        except Exception:
            log.debug("signal-yield timer install failed", exc_info=True)
        try:
            AppHelper.runEventLoop(installInterrupt=True)
        except KeyboardInterrupt:
            log.info("shutting down (KeyboardInterrupt)")
        finally:
            _cleanup()
    else:
        # Overlay disabled — fall back to the original simple wait.
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
