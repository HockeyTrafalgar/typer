#!/usr/bin/env python3
"""
Typer — Hold-to-Talk Dictation (Whisper-MLX, Apple Silicon native)

Hold Right-⌘ → speak → release; the whole utterance is transcribed once and
pasted into the focused field. Or tap F19 to toggle dictation on/off.

Voice pipeline mirrors Handy (the most popular open-source local dictation
app): hold the key → accumulate ONLY Silero-VAD voiced audio (threshold 0.3
with onset/hangover smoothing) → on release transcribe the whole utterance
ONCE with a plain greedy decode (no prior-text context, Whisper's own
no-speech gate) → strip filler words and stutters → paste in one shot. No
live/streaming mode, no energy gate, and no confidence/phrase hallucination
post-filter — the VAD is the defense, exactly as in Handy.

Model: mlx-community/whisper-large-v3-turbo via mlx-whisper — multilingual
       (~99 languages) with automatic language detection, native Apple
       Silicon. Override with TYPER_MODEL.

Requirements: mlx-whisper, pynput, pyperclip, sounddevice, numpy,
              silero-vad, onnxruntime, torch
Permissions: Microphone + Accessibility + Input Monitoring on the Python binary
"""

import threading
import sys
import os
import re
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
    repo = os.environ.get("TYPER_MODEL", "mlx-community/whisper-large-v3-turbo")
    cache_root = Path(os.environ.get("HF_HOME") or
                      Path.home() / ".cache" / "huggingface") / "hub"
    cache_dir = cache_root / f"models--{repo.replace('/', '--')}"
    if cache_dir.is_dir():
        os.environ["HF_HUB_OFFLINE"] = "1"
        # Older huggingface_hub releases honored this constant instead.
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
_maybe_enable_hf_offline()

from pynput import keyboard
from pynput import mouse as _pynput_mouse
import mlx_whisper
from silero_vad import load_silero_vad

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
    NSVisualEffectMaterialPopover,
    NSVisualEffectBlendingModeBehindWindow,
    NSVisualEffectStateActive,
    NSFontWeightRegular,
    NSViewWidthSizable, NSViewHeightSizable,
    NSAppearance, NSAppearanceNameVibrantLight,
    NSImage, NSBezierPath, NSEdgeInsetsMake, NSImageResizingModeStretch,
    NSImageView, NSImageSymbolConfiguration,
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
    )
    _LIQUID_GLASS_AVAILABLE = True
except ImportError:
    NSGlassEffectView = None
    NSGlassEffectViewStyleRegular = None
    _LIQUID_GLASS_AVAILABLE = False
from Quartz import kCACornerCurveContinuous
from Foundation import NSObject, NSUserDefaults, NSNumber
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
    AXUIElementCopyElementAtPosition,
    AXUIElementIsAttributeSettable,
    AXValueGetValue,
    kAXFocusedUIElementAttribute,
    kAXSelectedTextRangeAttribute,
    kAXRoleAttribute,
    kAXValueAttribute,
    kAXValueCGRectType,
)

# NSPanel-only style mask that gives us a borderless, non-activating HUD.
# AppKit doesn't always re-export this constant by name in PyObjC, so we
# define it manually (NSWindowStyleMaskNonactivatingPanel = 1<<7).
NSWindowStyleMaskNonactivatingPanel = 1 << 7

# ── CONFIG ──────────────────────────────────────────────────────────────────
# Default Whisper on MLX. large-v3-turbo has 4 decoder layers vs large-v3's
# 32 → roughly half the per-utterance inference time with negligible English
# accuracy loss, and it stays multilingual (~99 languages). That speed is
# what makes the single on-release pass feel instant, so it's the default.
# Set TYPER_MODEL="mlx-community/whisper-large-v3-mlx" for maximum (esp.
# non-English) accuracy at ~2× the latency, or "...medium-mlx"/"...small-mlx"
# for lower memory.
MODEL_ID = os.environ.get("TYPER_MODEL", "mlx-community/whisper-large-v3-turbo")
SAMPLE_RATE = 16000

# ── SPEECH PIPELINE (Handy-style) ────────────────────────────────────────────
# We mirror Handy's voice pipeline: hold the key -> accumulate ONLY Silero-VAD
# voiced audio -> transcribe the whole utterance ONCE on release -> strip
# fillers/stutters -> paste. There is deliberately no live/streaming mode, no
# energy gate, and no confidence/phrase hallucination post-filter: Handy gets
# its reliability from the VAD plus a plain greedy decode, and so do we. The
# handful of values below are the only speech settings; everything else is
# hardcoded to Handy's choices (see _whisper_transcribe and the VAD config).

# Optional explicit language (e.g. "en", "ru"). Empty = Whisper auto-detects.
WHISPER_LANGUAGE = os.environ.get("TYPER_WHISPER_LANGUAGE", "") or None

# Optional vocabulary prime fed to Whisper as initial_prompt to bias toward
# names/jargon -- exactly how Handy passes its "custom words".
CUSTOM_WORDS = os.environ.get("TYPER_INITIAL_PROMPT", "").strip() or None

# Whisper's own no-speech probability gate. Handy/transcribe-rs uses 0.2;
# segments Whisper itself flags as non-speech above this are dropped inside
# mlx_whisper.transcribe. This is the ONLY hallucination guard we keep, and it
# is Whisper's built-in one -- no custom post-filtering on top.
NO_SPEECH_THRESHOLD = 0.2

# How often the capture loop drains the mic queue while the key is held
# (seconds). Pure audio plumbing -- no transcription runs until release.
CAPTURE_TICK_S = 0.1

# Short-clip zero-pad (Handy audio.rs): pad an utterance with under 1.0 s of
# voiced audio up to 1.25 s so Whisper's encoder stays stable on one-word
# answers like "yes"/"no".
PAD_BELOW_SAMPLES = SAMPLE_RATE            # under 1.0 s of audio
PAD_TO_SAMPLES    = SAMPLE_RATE * 5 // 4   # -> 1.25 s

# Clipboard handling for the Cmd+V paste.
#
# The transcription is delivered by copying it to the clipboard and
# synthesizing Cmd+V, which clobbers whatever the user had copied. With
# restore ON (the default) we snapshot the clipboard ONCE when a dictation
# session starts and put it back ONCE after the session's final paste has
# landed — so anything you had copied before dictating is still there to
# paste afterwards.
#
# Why session-level and not per-paste: in live/both mode we paste on every
# tick, so a per-paste snapshot would capture OUR OWN previous paste, not the
# user's original clipboard. Snapshotting at session start is the only point
# where the clipboard still holds the user's content.
#
# Safety: the restore runs on a background thread after a delay and only if
# OUR pasted text is still on the clipboard (so it never clobbers something
# the user copied after dictating, and the delay clears the window in which
# the target app reads the clipboard for the synthetic Cmd+V). Set
# TYPER_RESTORE_CLIPBOARD=0 to disable and just leave the transcript on the
# clipboard (the old behavior). Bump the delay if a slow target app pastes
# the restored content instead of the transcript.
RESTORE_CLIPBOARD = os.environ.get("TYPER_RESTORE_CLIPBOARD", "1") == "1"
CLIPBOARD_RESTORE_DELAY_S = float(
    os.environ.get("TYPER_CLIPBOARD_RESTORE_DELAY_S", "1.0")
)

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
VAD_FRAME = 512  # samples per Silero frame @ 16 kHz (~32 ms)
# Speech-probability threshold. Handy uses 0.3 (below Silero's 0.5 default) to
# catch soft consonant onsets; the onset debounce below rejects the noise
# blips that a low threshold would otherwise let in.
VAD_THRESHOLD = 0.3
# Pre-roll flushed at speech onset, and trailing silence kept after speech
# ends. Handy uses ~450 ms for each (15 frames at its 30 ms frame size).
VAD_PRE_ROLL_MS = 450
VAD_HANGOVER_MS = 450
# Onset debounce: require this many CONSECUTIVE voiced frames before opening
# the gate, so a single noise blip can't start a capture (Handy onset = 2).
VAD_ONSET_FRAMES = 2

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
# Live mic-level meter (a small EQ-style bar cluster between the mic icon and
# the transcript) so you can SEE the microphone is picking up sound. Driven by
# the RAW input level ~10×/s, so it reacts to any audio — not just speech that
# passes the VAD gate. Set TYPER_OVERLAY_LEVEL_METER=0 to hide it.
OVERLAY_LEVEL_METER = os.environ.get("TYPER_OVERLAY_LEVEL_METER", "1") == "1"
# RMS that maps to a full-scale meter. Normal speech sits around 0.02–0.08
# RMS; 0.12 keeps loud speech near the top without pinning. Lower = more
# sensitive (bars rise on quieter input).
MIC_LEVEL_REF = float(os.environ.get("TYPER_MIC_LEVEL_REF", "0.12"))

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
# Wall-clock time the user released the key / requested stop. Used purely to
# log the true release→paste latency (which includes any in-flight whisper
# tick we have to wait out before the final commit runs).
live_stop_requested_at = 0.0

f19_held         = False
f18_held         = False
rcmd_held        = False
# "Latched"/permanent mode: set when F19 is tapped while F18 is still held.
# Once latched, releasing F18 does NOT stop dictation — only another F19 tap
# does. Reset whenever a session starts or stops.
latched          = False

# Single inference thread services jobs; MLX tensors stay on one thread.
inference_q = queue.Queue()

# ── INFERENCE THREAD ─────────────────────────────────────────────────────────


# ── TEXT CLEANUP (ported from Handy's text.rs) ───────────────────────────────
# Language-aware filler lists. "um"/"eh"/"ha" are real words in some languages
# (Portuguese "um" = a/an, Spanish "ha" = has), so they're only treated as
# fillers where they genuinely are; unknown languages get a list that omits
# those.
_FILLER_WORDS = {
    "en": ["uh", "um", "uhm", "umm", "uhh", "uhhh", "ah", "hmm", "hm", "mmm",
           "mm", "mh", "eh", "ehh", "ha"],
    "es": ["ehm", "mmm", "hmm", "hm"],
    "pt": ["ahm", "hmm", "mmm", "hm"],
    "fr": ["euh", "hmm", "hm", "mmm"],
    "de": ["äh", "ähm", "hmm", "hm", "mmm"],
    "it": ["ehm", "hmm", "mmm", "hm"],
    "cs": ["ehm", "hmm", "mmm", "hm"],
    "pl": ["hmm", "mmm", "hm"],
    "tr": ["hmm", "mmm", "hm"],
    "ru": ["хм", "ммм", "hmm", "mmm"],
    "uk": ["хм", "ммм", "hmm", "mmm"],
    "ar": ["hmm", "mmm"],
    "ja": ["hmm", "mmm"],
    "ko": ["hmm", "mmm"],
    "vi": ["hmm", "mmm", "hm"],
    "zh": ["hmm", "mmm"],
}
_FILLER_FALLBACK = ["uh", "uhm", "umm", "uhh", "uhhh", "ah", "hmm", "hm",
                    "mmm", "mm", "mh", "ehh"]


def _filler_words_for_language(lang):
    base = (lang or "en").split("-")[0].split("_")[0]
    return _FILLER_WORDS.get(base, _FILLER_FALLBACK)


def _collapse_stutters(text):
    """Collapse 3+ consecutive identical (case-insensitive) alphabetic words to
    one: "wh wh wh wh" -> "wh", "I I I I" -> "I". Two repeats are left alone."""
    words = text.split()
    if not words:
        return text
    result = []
    i = 0
    while i < len(words):
        word = words[i]
        if word.lower().isalpha():
            count = 1
            while i + count < len(words) and words[i + count].lower() == word.lower():
                count += 1
            result.append(word)
            i += count if count >= 3 else 1
        else:
            result.append(word)
            i += 1
    return " ".join(result)


def _filter_transcription_output(text, lang):
    """Handy's filter_transcription_output: remove filler words, collapse
    stutters, normalize whitespace. The ONLY post-processing we do."""
    filtered = text
    for word in _filler_words_for_language(lang):
        filtered = re.sub(r"(?i)\b%s\b[,.]?" % re.escape(word), "", filtered)
    filtered = _collapse_stutters(filtered)
    filtered = re.sub(r"\s{2,}", " ", filtered)
    return filtered.strip()


def _whisper_transcribe(audio_f32):
    """Transcribe a float32 mono 16 kHz array the way Handy does: a single
    plain decode (no temperature ladder, no prior-text context, Whisper's own
    no-speech gate), then filler/stutter cleanup. No energy gate and no
    confidence/phrase hallucination post-filter -- the Silero VAD upstream is
    the real defense against silence captions."""
    kwargs = dict(
        path_or_hf_repo=MODEL_ID,
        temperature=0.0,
        condition_on_previous_text=False,
        no_speech_threshold=NO_SPEECH_THRESHOLD,
        fp16=True,
        verbose=None,
    )
    if WHISPER_LANGUAGE:
        kwargs["language"] = WHISPER_LANGUAGE
    if CUSTOM_WORDS:
        kwargs["initial_prompt"] = CUSTOM_WORDS
    res = mlx_whisper.transcribe(audio_f32, **kwargs)
    text = (res.get("text") or "").strip()
    lang = WHISPER_LANGUAGE or res.get("language") or "en"
    cleaned = _filter_transcription_output(text, lang)
    if cleaned != text:
        log.debug("cleanup: %r -> %r", text, cleaned)
    return cleaned

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
    """Streaming Silero-VAD gate with onset debounce, pre-roll and hangover --
    a port of Handy's SmoothedVad. Emits only voiced audio (plus a pre-roll
    lead-in and a short hangover tail), so Whisper never sees silence."""

    def __init__(self, model, threshold, pre_roll_ms, hangover_ms, onset_frames):
        self.model = model
        self.threshold = threshold
        self.prefill_frames = max(1, pre_roll_ms * SAMPLE_RATE // 1000 // VAD_FRAME)
        self.hangover_frames = max(1, hangover_ms * SAMPLE_RATE // 1000 // VAD_FRAME)
        self.onset_frames = max(1, onset_frames)
        self.tail = np.zeros(0, dtype=np.float32)
        # Most-recent frames, flushed as pre-roll at speech onset.
        self.frame_buffer = collections.deque(maxlen=self.prefill_frames + 1)
        self.in_speech = False
        self.hangover_counter = 0
        self.onset_counter = 0
        self.stats = {"frames": 0, "voiced": 0, "emitted": 0,
                      "speech_starts": 0, "speech_ends": 0}

    def reset(self):
        self.tail = np.zeros(0, dtype=np.float32)
        self.frame_buffer.clear()
        self.in_speech = False
        self.hangover_counter = 0
        self.onset_counter = 0
        try:
            self.model.reset_states()
        except Exception:
            pass

    def _is_voiced(self, frame):
        with torch.no_grad():
            prob = float(self.model(torch.from_numpy(frame), 16000).item())
        return prob >= self.threshold, prob

    def process(self, samples):
        if len(samples) == 0:
            return np.zeros(0, dtype=np.float32)
        buf = np.concatenate([self.tail, samples]) if len(self.tail) else samples
        n = len(buf) // VAD_FRAME
        out = []
        for i in range(n):
            frame = buf[i * VAD_FRAME:(i + 1) * VAD_FRAME].astype(np.float32, copy=False)
            # Buffer every frame for possible pre-roll (Handy frame_buffer).
            self.frame_buffer.append(frame)
            voiced, prob = self._is_voiced(frame)
            self.stats["frames"] += 1
            if voiced:
                self.stats["voiced"] += 1
            if not self.in_speech and voiced:
                # Potential onset: need onset_frames consecutive voiced frames.
                self.onset_counter += 1
                if self.onset_counter >= self.onset_frames:
                    self.in_speech = True
                    self.hangover_counter = self.hangover_frames
                    self.onset_counter = 0
                    self.stats["speech_starts"] += 1
                    log.debug("VAD start (p=%.2f, flushing %d pre-roll frames)",
                              prob, len(self.frame_buffer))
                    out.extend(self.frame_buffer)
                    self.frame_buffer.clear()
            elif self.in_speech and voiced:
                self.hangover_counter = self.hangover_frames
                out.append(frame)
            elif self.in_speech and not voiced:
                if self.hangover_counter > 0:
                    self.hangover_counter -= 1
                    out.append(frame)
                else:
                    self.in_speech = False
                    self.stats["speech_ends"] += 1
                    log.debug("VAD end (hangover elapsed)")
            else:
                # Silence or broken onset run.
                self.onset_counter = 0
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
# Last seen callback frame count, used by the heartbeat thread to
# detect a stream that stopped delivering samples without raising an
# error (CoreAudio occasionally drops a stream silently — we want to
# know about it before the next dictation attempt fails).
_audio_last_seen_frames = 0
_audio_last_seen_at = 0.0

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
        ch0 = indata[:, 0]
        _persistent_audio_q.put(ch0.copy())
        # Drive the overlay mic-level meter from the RAW input so the user
        # sees the mic is picking up sound even below the VAD threshold.
        _emit_mic_level(ch0)

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

# ── HEARTBEAT ────────────────────────────────────────────────────────────────
# Periodic background log that records the health of subsystems we rely
# on long-running. Specifically what bit me last night: the persistent
# CoreAudio stream silently stopped delivering callbacks at some point
# during an overnight run — restart of Typer fixed it. Now we'll see
# it BEFORE the user tries to dictate.
HEARTBEAT_INTERVAL_S = float(os.environ.get("TYPER_HEARTBEAT_S", "60"))
_heartbeat_stop = threading.Event()

def _heartbeat():
    """Every HEARTBEAT_INTERVAL_S, log a one-line health snapshot.
    Watches the audio stream for two failure modes:
      • stream.active flipped to False (CoreAudio dropped it)
      • stream.active still True but no new callback frames in N
        seconds (stream is alive but silently producing nothing)
    Either way, try restarting the stream so the NEXT dictation works."""
    global _audio_last_seen_frames, _audio_last_seen_at
    _audio_last_seen_frames = _audio_cb_stats["frames"]
    _audio_last_seen_at = time.time()
    while not _heartbeat_stop.wait(HEARTBEAT_INTERVAL_S):
        try:
            _do_heartbeat()
        except Exception:
            log.exception("heartbeat tick failed")

def _do_heartbeat():
    global _audio_last_seen_frames, _audio_last_seen_at
    now = time.time()
    frames_now = _audio_cb_stats["frames"]
    calls_now = _audio_cb_stats["calls"]
    dframes = frames_now - _audio_last_seen_frames
    age_s = now - _audio_last_seen_at
    stream_active = (
        _audio_input_stream is not None
        and getattr(_audio_input_stream, "active", False)
    )
    log.info(
        "heartbeat: stream_active=%s calls=%d frames=%d Δframes=%d "
        "(over %.1fs) status_warns=%d session_active=%s queue=%d",
        stream_active, calls_now, frames_now, dframes, age_s,
        _audio_cb_stats["status_warns"],
        _audio_session_active.is_set(),
        _persistent_audio_q.qsize(),
    )
    # Restart logic: only intervene when the stream looks unhealthy.
    # A perfectly healthy idle stream WILL still tick its callback at
    # ~10 Hz (we set blocksize to 0.1 s of audio) — if dframes is zero
    # over a full heartbeat interval, the stream's broken.
    needs_restart = False
    if _audio_input_stream is None:
        needs_restart = True
        log.warning("heartbeat: persistent audio stream is None — restarting")
    elif not stream_active:
        needs_restart = True
        log.warning("heartbeat: persistent audio stream went inactive — restarting")
    elif dframes == 0 and age_s >= HEARTBEAT_INTERVAL_S * 0.9:
        needs_restart = True
        log.warning(
            "heartbeat: persistent audio stream stalled "
            "(no callback frames in %.1fs) — restarting", age_s,
        )
    if needs_restart:
        # Never swap the stream out from under an in-flight dictation —
        # _do_live_stream holds a reference to the live stream/queue and a
        # mid-session restart would drop the audio it's actively draining.
        # The session is delivering frames anyway, so a genuine stall here
        # is unlikely; defer any restart to the next idle heartbeat.
        if _audio_session_active.is_set():
            log.warning("heartbeat: stream looks unhealthy but a dictation "
                        "session is active — deferring restart")
        else:
            _restart_persistent_audio_input()
    _audio_last_seen_frames = frames_now
    _audio_last_seen_at = now

def _restart_persistent_audio_input():
    """Close the current persistent audio stream (if any) and re-open
    a fresh one. Used by the heartbeat watchdog when CoreAudio drops
    the stream out from under us."""
    global _audio_input_stream
    old = _audio_input_stream
    _audio_input_stream = None
    if old is not None:
        try:
            old.stop()
            old.close()
        except Exception:
            log.debug("old stream close failed", exc_info=True)
    _open_persistent_audio_input()
    if _audio_input_stream is not None:
        log.info("audio stream restarted successfully")
    else:
        log.error("audio stream restart FAILED — next session will use fallback")

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


def _overlay_is_dark():
    """True when macOS is in Dark Mode. The Liquid Glass surface inherits
    the system appearance, so on a dark desktop the HUD material turns
    dark and our near-black ink would be invisible against it.

    AppleInterfaceStyle is "Dark" in Dark Mode and absent (None) in Light
    Mode — the canonical, view-independent signal. We fall back to the
    app's effective appearance name if the default can't be read."""
    try:
        style = NSUserDefaults.standardUserDefaults().stringForKey_(
            "AppleInterfaceStyle"
        )
        if style is not None:
            return "dark" in style.lower()
    except Exception:
        pass
    try:
        name = NSApp.effectiveAppearance().name()
        return "Dark" in str(name)
    except Exception:
        return False


def _overlay_ink_color(alpha=0.55):
    """Ink color for the HUD text + mic icon. Near-black on a light
    surface, near-white on a dark one, so the transcript stays legible
    regardless of the system theme."""
    white = 1.0 if _overlay_is_dark() else 0.0
    return NSColor.colorWithCalibratedWhite_alpha_(white, alpha)

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


# Accessibility roles used to classify a click target as a place you can type.
# Editable text inputs across native/web toolkits report one of these roles…
_AX_EDITABLE_ROLES = {
    "AXTextField", "AXTextArea", "AXComboBox", "AXSearchField",
}
# …and these are unambiguously NON-text controls — clicking one should never
# start dictation. (We block on these even in permissive mode.)
_AX_NON_EDITABLE_ROLES = {
    "AXButton", "AXMenuButton", "AXPopUpButton", "AXMenuItem", "AXMenuBarItem",
    "AXLink", "AXStaticText", "AXImage", "AXCheckBox", "AXRadioButton",
    "AXSlider", "AXIncrementor", "AXStepper", "AXDisclosureTriangle",
    "AXColorWell", "AXScrollBar", "AXTabGroup", "AXToolbar",
    "AXSegmentedControl", "AXRadioGroup", "AXDockItem",
}


def _ax_role(el):
    try:
        err, role = AXUIElementCopyAttributeValue(el, kAXRoleAttribute, None)
        if err or role is None:
            return None
        return str(role)
    except Exception:
        return None


def _ax_element_editability(el):
    """Classify an AX element as 'editable', 'non_editable', or 'unknown'.

    'editable' = a place keystrokes can land: a text-input role, OR it exposes
    a selected-text range (every real text input does — the same signal the
    caret overlay uses), OR its AXValue is settable. 'non_editable' = a known
    non-text control. Anything else (generic containers, web groups, no AX
    data) is 'unknown' so the caller can decide how cautious to be."""
    if el is None:
        return "unknown"
    role = _ax_role(el)
    if role in _AX_EDITABLE_ROLES:
        return "editable"
    # A live selected-text range means it's a text input regardless of role
    # (covers contenteditable web areas that report generic roles).
    try:
        err, rng = AXUIElementCopyAttributeValue(
            el, kAXSelectedTextRangeAttribute, None)
        if not err and rng is not None:
            return "editable"
    except Exception:
        pass
    # A settable AXValue is an editable control (read-only labels aren't).
    try:
        err, settable = AXUIElementIsAttributeSettable(el, kAXValueAttribute, None)
        if not err and settable:
            return "editable"
    except Exception:
        pass
    if role in _AX_NON_EDITABLE_ROLES:
        return "non_editable"
    return "unknown"


def _click_should_start_dictation(x, y):
    """Editability gate for the mouse trigger. Returns True if we should start
    dictation for a left-click long-press at screen point (x, y).

    Strategy: classify the element directly UNDER the click first (that's what
    the user pointed at — a button vs a field). If that's inconclusive, fall
    back to the currently FOCUSED element (where typing would actually land).
    Block only when something is POSITIVELY non-editable; when AX can't tell
    (apps that don't expose accessibility — Electron, some web fields, GPU
    terminals), start anyway so the trigger keeps working there.

    pynput delivers (x, y) in the global top-left screen space, which is
    exactly what AXUIElementCopyElementAtPosition expects — no flip needed."""
    # 1) What did they click ON?
    pt_state = "unknown"
    try:
        err, el = AXUIElementCopyElementAtPosition(
            _ax_system, float(x), float(y), None)
        if not err and el is not None:
            pt_state = _ax_element_editability(el)
    except Exception:
        log.debug("AX point lookup failed", exc_info=True)
    if pt_state == "editable":
        return True
    if pt_state == "non_editable":
        log.debug("mouse trigger gated: click target non-editable")
        return False
    # 2) Point inconclusive → where would typing go (focused element)?
    foc_state = "unknown"
    try:
        err, foc = AXUIElementCopyAttributeValue(
            _ax_system, kAXFocusedUIElementAttribute, None)
        if not err and foc is not None:
            foc_state = _ax_element_editability(foc)
    except Exception:
        log.debug("AX focus lookup failed", exc_info=True)
    if foc_state == "editable":
        return True
    if foc_state == "non_editable":
        log.debug("mouse trigger gated: focused element non-editable")
        return False
    # 3) Both inconclusive → start anyway (permissive).
    log.debug("mouse trigger: editability unknown — starting (permissive)")
    return True


class _LevelMeterView(NSView):
    """A tiny EQ-style mic-activity meter: a cluster of rounded vertical bars
    whose heights track a 0..1 level. Center bars are weighted taller so the
    cluster reads as an organic 'listening' visual rather than a flat row.
    The ink color follows the system theme (via _overlay_ink_color), and each
    bar's opacity rises with the level so it's faint at silence and bold when
    the mic is hearing you."""

    _BARS = 5
    _WEIGHTS = (0.5, 0.8, 1.0, 0.8, 0.5)
    _BAR_W = 3.0
    _GAP = 3.5
    _MIN_H = 4.0

    def initWithFrame_(self, frame):
        self = objc.super(_LevelMeterView, self).initWithFrame_(frame)
        if self is None:
            return None
        self._level = 0.0
        return self

    @objc.python_method
    def set_level(self, level):
        try:
            lvl = max(0.0, min(1.0, float(level)))
        except Exception:
            return
        # Skip redraws for imperceptible changes to avoid needless main-thread
        # work, but always honor a return-to-zero so the meter settles.
        if abs(lvl - self._level) < 0.01 and not (lvl == 0.0 and self._level != 0.0):
            return
        self._level = lvl
        self.setNeedsDisplay_(True)

    def drawRect_(self, rect):
        bounds = self.bounds()
        n = self._BARS
        bar_w = self._BAR_W
        gap = self._GAP
        total_w = n * bar_w + (n - 1) * gap
        x0 = (bounds.size.width - total_w) / 2.0
        cy = bounds.size.height / 2.0
        max_h = bounds.size.height
        for i in range(n):
            lvl = self._level * self._WEIGHTS[i]
            h = self._MIN_H + lvl * (max_h - self._MIN_H)
            x = x0 + i * (bar_w + gap)
            y = cy - h / 2.0
            r = NSMakeRect(x, y, bar_w, h)
            path = NSBezierPath.bezierPathWithRoundedRect_xRadius_yRadius_(
                r, bar_w / 2.0, bar_w / 2.0
            )
            _overlay_ink_color(0.22 + 0.63 * min(1.0, lvl)).setFill()
            path.fill()


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
                NSAppearance.appearanceNamed_(
                    NSAppearanceNameVibrantDark if _overlay_is_dark()
                    else NSAppearanceNameVibrantLight
                )
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

        # ── Leading mic-activity level meter ───────────────────────────
        # Spotlight-style layout: one leading element, text fills the rest.
        # Instead of a static mic icon we show an animated EQ-style level
        # meter so the user can see the microphone is picking up sound. It
        # sits on the first line and is anchored there during vertical growth.
        self._mic = None  # no static icon; the meter is the only leading glyph
        self._meter = None
        if OVERLAY_LEVEL_METER:
            meter_w = 34.0
            meter_h = 30.0
            meter_x = OVERLAY_PAD_X
            meter_y = (OVERLAY_H - meter_h) / 2.0
            meter = _LevelMeterView.alloc().initWithFrame_(
                NSMakeRect(meter_x, meter_y, meter_w, meter_h)
            )
            content.addSubview_(meter)
            self._meter = meter
            text_x = meter_x + meter_w + 12.0
        else:
            # Meter disabled and no icon → text starts at the leading pad.
            text_x = OVERLAY_PAD_X

        # ── Transcript label ───────────────────────────────────────────
        # Spotlight-style placeholder typography: SF Pro Regular at the
        # body size, muted near-black. Active transcript uses the same
        # weight at full strength so the visual rhythm doesn't change
        # mid-sentence.
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
        self.label.setTextColor_(_overlay_ink_color(0.55))
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
        if getattr(self, "_meter", None) is not None:
            self._meter.set_level(0.0)
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

    def setProcessing_(self, _ignored):
        """Shown the instant the user releases the key, while the final
        whisper pass + paste run. Replaces the old behavior of hiding the
        HUD on release: the panel stays visible with a 'Processing…'
        placeholder so the user knows their text is still on the way, and
        commit-final hides it only AFTER the paste lands. The preview ticks
        have already stopped by now, so nothing overwrites this until
        commit-final calls setText_ with the final transcript."""
        self._current_text = ""
        # Capture has stopped; settle the meter to zero so it doesn't look
        # like the mic is still hearing input while we transcribe.
        if getattr(self, "_meter", None) is not None:
            self._meter.set_level(0.0)
        self._render("Processing…", placeholder=True)

    def setLevel_(self, num):
        """Update the mic-activity meter (0..1). Called ~10×/s from the audio
        callback via performSelectorOnMainThread_."""
        meter = getattr(self, "_meter", None)
        if meter is None:
            return
        try:
            meter.set_level(float(num))
        except Exception:
            pass

    @objc.python_method
    def _render(self, text, placeholder):
        # Both placeholder and live-transcript text share the SAME hue
        # as the mic icon (black @ 55% alpha) for a single, calm
        # typographic voice across the whole HUD.
        color = _overlay_ink_color(0.55)
        self.label.setTextColor_(color)
        # Re-tint the mic each render too: the icon's tint is otherwise
        # fixed at init, so a live Light↔Dark theme switch (or launching
        # in a different theme than the current one) would leave it stale.
        if getattr(self, "_mic", None) is not None:
            try:
                self._mic.setContentTintColor_(color)
            except Exception:
                pass
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

        # Anchor mic icon (and the level meter) to the FIRST line so they
        # don't drift down the panel as the transcript grows.
        for sub in (self._mic, getattr(self, "_meter", None)):
            if sub is None:
                continue
            sf = sub.frame()
            sh = sf.size.height
            stop_pad = (OVERLAY_H - sh) / 2.0
            sub.setFrame_(NSMakeRect(
                sf.origin.x,
                new_h - stop_pad - sh,
                sf.size.width, sh,
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

def overlay_processing():
    """Show 'Processing…' on release; the HUD stays up until commit-final
    finishes the final transcribe + paste and then hides it."""
    _overlay_call("setProcessing:", None)

def overlay_set_level(level):
    """Push a 0..1 mic level to the overlay's activity meter. NSNumber so it
    bridges cleanly through performSelectorOnMainThread_withObject_."""
    if not OVERLAY_LEVEL_METER:
        return
    _overlay_call("setLevel:", NSNumber.numberWithDouble_(float(level)))

# Smoothed mic level (module-level so the audio callback can keep state
# between frames). Fast attack / slow release so the meter snaps up on speech
# onset and eases back down, instead of flickering frame-to-frame.
_mic_level_smoothed = 0.0
_MIC_LEVEL_FLOOR = 0.004          # subtract noise floor so silence reads ~0
_MIC_LEVEL_ATTACK = 0.6
_MIC_LEVEL_RELEASE = 0.3

def _emit_mic_level(samples):
    """Compute a smoothed 0..1 level from a raw audio frame and push it to the
    overlay meter. Called from the audio callback (~10×/s). Cheap and
    exception-safe so it can never disturb audio capture."""
    global _mic_level_smoothed
    if not (OVERLAY_ENABLED and OVERLAY_LEVEL_METER):
        return
    try:
        rms = float(np.sqrt(np.mean(samples * samples)))
        eff = max(0.0, rms - _MIC_LEVEL_FLOOR)
        # sqrt gives a more perceptual response so normal speech visibly
        # fills the meter rather than hugging the bottom.
        target = min(1.0, (eff / MIC_LEVEL_REF) ** 0.5) if MIC_LEVEL_REF > 0 else 0.0
        coeff = _MIC_LEVEL_ATTACK if target > _mic_level_smoothed else _MIC_LEVEL_RELEASE
        _mic_level_smoothed += coeff * (target - _mic_level_smoothed)
        overlay_set_level(_mic_level_smoothed)
    except Exception:
        pass

def _reset_mic_level():
    """Zero the meter at the start of a session so a stale level from the
    previous session doesn't linger when the HUD reappears."""
    global _mic_level_smoothed
    _mic_level_smoothed = 0.0
    overlay_set_level(0.0)


# ── LIVE STREAMING ────────────────────────────────────────────────────────────
# Per-session clipboard save/restore state. We snapshot the user's clipboard
# once at the START of a dictation session (the only moment it still holds
# THEIR content, since we paste over it during the session) and put it back
# once after the session's final paste. `last_write` is the exact string we
# last placed on the clipboard, used as the "is it still ours?" guard so the
# restore can never clobber something copied after dictating.
_clipboard_session = {"original": None, "last_write": None, "wrote": False}

def _clipboard_session_begin():
    """Snapshot the user's clipboard at the start of a dictation session."""
    _clipboard_session["last_write"] = None
    _clipboard_session["wrote"] = False
    _clipboard_session["original"] = None
    if not RESTORE_CLIPBOARD:
        return
    try:
        _clipboard_session["original"] = pyperclip.paste()
        log.debug("clipboard snapshot taken (%d chars)",
                  len(_clipboard_session["original"] or ""))
    except Exception:
        log.debug("clipboard snapshot failed", exc_info=True)

def _clipboard_session_restore():
    """Restore the pre-dictation clipboard, once, after the final paste.

    Runs on a background thread after CLIPBOARD_RESTORE_DELAY_S and only if
    our pasted text is STILL on the clipboard — so it neither races the
    target app's read of the synthetic Cmd+V nor clobbers something the user
    copied after dictating. No-op if restore is off, nothing was pasted, or
    the original clipboard already matched what we pasted."""
    if not RESTORE_CLIPBOARD or not _clipboard_session["wrote"]:
        return
    original = _clipboard_session["original"]
    last_write = _clipboard_session["last_write"]
    if original is None or original == last_write:
        return
    def _restore(original=original, pasted=last_write):
        time.sleep(CLIPBOARD_RESTORE_DELAY_S)
        try:
            if pyperclip.paste() == pasted:
                pyperclip.copy(original)
                log.debug("clipboard restored to pre-dictation contents "
                          "(%d chars)", len(original))
            else:
                log.debug("clipboard changed since our paste — not restoring")
        except Exception:
            log.debug("clipboard restore failed", exc_info=True)
    threading.Thread(target=_restore, daemon=True, name="clip-restore").start()

def _paste_text(text):
    if not text:
        return
    log.debug("paste %d chars: %r", len(text), text)
    # Put the transcription on the clipboard and paste it. We deliberately do
    # NOT restore inline here: in live/both mode this runs every tick, and the
    # restore must happen ONCE at session end (see _clipboard_session_restore)
    # against the user's ORIGINAL clipboard, not our own previous paste.
    pyperclip.copy(text)
    _clipboard_session["last_write"] = text
    _clipboard_session["wrote"] = True
    time.sleep(0.03)
    _release_modifiers()
    with _kbd.pressed(keyboard.Key.cmd):
        _kbd.press('v')
        _kbd.release('v')

def _do_live_stream():
    """Capture + transcribe for one dictation session (Handy-style). Runs ON
    THE INFERENCE THREAD (MLX owner).

    While the key is held we accumulate ONLY Silero-VAD voiced audio; no
    transcription happens during capture. On release we transcribe the whole
    utterance ONCE, clean fillers/stutters, and paste it in one shot.

    Audio comes from the PERSISTENT InputStream opened once at app startup
    (see _open_persistent_audio_input). Falls back to opening a per-session
    stream only if the persistent one failed to come up.
    """
    log.info("live stream begin")
    # Health check the persistent stream BEFORE recording. If CoreAudio
    # silently dropped it since the last session, restart inline so
    # this session captures audio instead of falling into a 0-frames
    # session and confusing the user.
    if (_audio_input_stream is not None
            and not getattr(_audio_input_stream, "active", False)):
        log.warning(
            "live stream begin: persistent stream is inactive — restarting inline"
        )
        _restart_persistent_audio_input()
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
            ch0 = indata[:, 0]
            audio_q.put(ch0.copy())
            _emit_mic_level(ch0)

    gate = None
    try:
        gate = VadGate(_get_vad_model(), VAD_THRESHOLD,
                       VAD_PRE_ROLL_MS, VAD_HANGOVER_MS, VAD_ONSET_FRAMES)
        log.info("VAD active (threshold=%.2f pre_roll=%dms hangover=%dms onset=%d)",
                 VAD_THRESHOLD, VAD_PRE_ROLL_MS, VAD_HANGOVER_MS, VAD_ONSET_FRAMES)
    except Exception:
        log.exception("VAD init failed -- feeding raw audio (Handy relies on VAD)")

    iters = 0
    voiced_buf = np.zeros(0, dtype=np.float32)
    state = {"transitioned": False}

    def _ingest_audio():
        """Drain the mic queue, VAD-gate it, and append voiced audio to the
        buffer. No transcription here -- we just accumulate while held."""
        nonlocal voiced_buf
        chunks = []
        while True:
            try:
                chunks.append(audio_q.get_nowait())
            except queue.Empty:
                break
        if not chunks:
            return
        samples = np.concatenate(chunks)
        if gate is not None:
            samples = gate.process(samples)
        if len(samples):
            voiced_buf = np.concatenate([voiced_buf, samples])
            if not state["transitioned"]:
                state["transitioned"] = True
                overlay_set_state("dictating")

    def _finalize():
        """Run once after release: drain the last audio, transcribe the whole
        utterance ONCE, clean it, and paste it."""
        nonlocal voiced_buf
        _ingest_audio()
        if live_cancel_event.is_set():
            log.info("finalize: cancelled, skipping paste")
            overlay_hide()
            return
        audio = voiced_buf
        # Short-clip zero-pad (Handy): keep Whisper stable on one-word answers.
        if 0 < len(audio) < PAD_BELOW_SAMPLES:
            audio = np.concatenate(
                [audio, np.zeros(PAD_TO_SAMPLES - len(audio), dtype=np.float32)])
        if not len(audio):
            log.info("finalize: no voiced audio captured")
            overlay_hide()
            return
        t0 = time.time()
        try:
            text = _whisper_transcribe(audio)
        except Exception:
            log.exception("whisper transcribe failed")
            overlay_hide()
            return
        infer_ms = (time.time() - t0) * 1000.0
        total_ms = ((time.time() - live_stop_requested_at) * 1000.0
                    if live_stop_requested_at else infer_ms)
        log.info("finalize: buf=%.2fs infer=%.0fms total(release->pasted)=%.0fms text=%r",
                 len(voiced_buf) / SAMPLE_RATE, infer_ms, total_ms, text)
        if not text:
            overlay_hide()
            return
        # Trailing space so consecutive dictations don't run together.
        overlay_set_text(text)
        _paste_text(text + " ")
        overlay_hide()

    t_start = time.time()
    try:
        log.info("stream cfg: model=%s lang=%s no_speech=%.2f prompt=%r",
                 MODEL_ID, WHISPER_LANGUAGE or "auto", NO_SPEECH_THRESHOLD,
                 CUSTOM_WORDS)
        overlay_show("Listening…")
        # Snapshot the user's clipboard now, before any paste overwrites it,
        # so we can restore it after this session's final paste.
        _clipboard_session_begin()
        # Zero the mic-activity meter for a clean start.
        _reset_mic_level()
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
                # wait() returns immediately on release, so the final pass runs
                # right after the key comes up.
                if live_stop_event.wait(CAPTURE_TICK_S):
                    break
                iters += 1
                _ingest_audio()
            log.debug("finalize after %d iters", iters)
            _finalize()
        finally:
            # Stop the callback from collecting more samples; the
            # persistent stream itself stays OPEN for the next session.
            _audio_session_active.clear()
            if not use_persistent and 'fallback_stream' in locals():
                try:
                    fallback_stream.stop()
                    fallback_stream.close()
                except Exception:
                    log.debug("fallback stream close failed", exc_info=True)
    except Exception:
        log.exception("stream error")
        overlay_hide()
    finally:
        # Restore the pre-dictation clipboard once the session's final paste
        # has landed (runs on a background thread after a safe delay; no-op if
        # nothing was pasted or restore is disabled).
        _clipboard_session_restore()
        vad_stats = gate.stats if gate else {}
        if gate:
            gate.reset()
        log.info(
            "live stream end: dur=%.2fs iters=%d audio_calls=%d audio_frames=%d status_warns=%d vad=%s",
            time.time() - t_start, iters, cb_stats["calls"], cb_stats["frames"],
            cb_stats["status_warns"], vad_stats,
        )

# ── LIVE START/STOP ──────────────────────────────────────────────────────────
def start_live_dictation():
    global live_active, latched
    with live_state_lock:
        if live_active:
            log.debug("start_live_dictation: already active, ignoring")
            return
        latched = False  # fresh session is never latched
        if not model_ready.is_set() or not model_ok:
            log.warning("model not loaded; ignoring start")
            return
        live_stop_event.clear()
        live_cancel_event.clear()
        live_active = True
    log.info("🎤 Live — speak now")
    inference_q.put(("live",))

def stop_live_dictation():
    global live_active, latched, live_stop_requested_at
    with live_state_lock:
        if not live_active:
            log.debug("stop_live_dictation: not active, ignoring")
            return
        live_active = False
        latched = False
        live_stop_requested_at = time.time()
        live_stop_event.set()
    log.info("✅ stopped (release latency clock started)")
    # Keep the HUD up with a 'Processing…' placeholder while commit-final
    # runs the final transcribe + paste; it hides the panel itself once the
    # text has landed. Skip for cancel (Esc) — that path hides immediately.
    if not live_cancel_event.is_set():
        overlay_processing()

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
    global f19_held, f18_held, rcmd_held, latched
    # NOTE: start/stop/toggle are invoked INLINE on the listener thread (not
    # dispatched to short-lived daemon threads). They only flip an Event and
    # enqueue a job — microsecond operations — so they never stall the event
    # tap. Running them inline guarantees that a press's start always happens
    # before the matching release's stop; the old per-event-thread dispatch
    # could reorder them and leave a session stuck "on".
    if key == keyboard.Key.cmd_r:
        if not rcmd_held:
            rcmd_held = True
            log.debug("right-cmd press → hold-to-talk start")
            start_live_dictation()
    elif key == keyboard.Key.f18:
        # Hold-to-talk, like right-cmd: hold F18 to dictate, release to stop.
        if not f18_held:
            f18_held = True
            log.debug("F18 press → hold-to-talk start")
            start_live_dictation()
    elif key == keyboard.Key.f19:
        if not f19_held:
            f19_held = True
            if f18_held and not latched:
                # F19 tapped while F18 is still held → latch the active
                # hold-to-talk session into permanent mode. Releasing F18
                # will no longer stop it; only another F19 tap will.
                latched = True
                log.debug("F19 while F18 held → permanent (latched) mode")
            elif latched:
                # Second F19 tap ends a latched session.
                log.debug("F19 press → stop (was latched)")
                stop_live_dictation()
            else:
                log.debug("F19 press → toggle (active=%s)", live_active)
                _toggle_live()
    elif key == keyboard.Key.esc and live_active:
        # The "esc" pill in the HUD advertises this — cancel the
        # in-flight session, drop the captured audio, don't paste.
        log.debug("Esc press → cancel dictation")
        cancel_live_dictation()

def on_release(key):
    global f19_held, f18_held, rcmd_held
    if key == keyboard.Key.cmd_r:
        rcmd_held = False
        log.debug("right-cmd release → hold-to-talk stop")
        stop_live_dictation()
    elif key == keyboard.Key.f18:
        f18_held = False
        if latched:
            # Session was latched into permanent mode via F19 — keep
            # dictating; only another F19 tap stops it.
            log.debug("F18 release ignored (latched/permanent mode)")
        else:
            # Show 'Processing…' on release (handled inside stop_live_
            # dictation) and keep the HUD up until _commit_final has pasted
            # the final text, then it hides the panel itself.
            log.debug("F18 release → hold-to-talk stop")
            stop_live_dictation()
    elif key == keyboard.Key.f19:
        f19_held = False
        log.debug("F19 release (no-op for toggle)")

# ── MOUSE LISTENER (left-click long-press) ──────────────────────────────────
# Hold the LEFT mouse button for MOUSE_HOLD_MS without moving the cursor
# more than MOUSE_MOVE_THRESHOLD_PX = start dictation. Release = stop.
# A movement past the threshold within the hold window cancels the
# trigger so normal click-and-drag (text selection, window drag) keeps
# working unchanged. Uses pynput's public CGEventTap-backed listener —
# no private API, no entitlements needed.
# Left-click long-press trigger is OFF by default — it competes with normal
# clicking. Set TYPER_MOUSE_CLICK_TRIGGER=1 to re-enable it.
MOUSE_CLICK_TRIGGER_ENABLED = os.environ.get("TYPER_MOUSE_CLICK_TRIGGER", "0") == "1"
MOUSE_HOLD_MS = int(os.environ.get("TYPER_MOUSE_HOLD_MS", "700"))
MOUSE_MOVE_THRESHOLD_PX = float(os.environ.get("TYPER_MOUSE_MOVE_PX", "5"))

_mouse_click_lock = threading.Lock()
_mouse_click_state = {
    "down": False,
    "x0": 0,
    "y0": 0,
    "moved": False,
    "fired_start": False,
    "timer": None,
}

def _mouse_hold_timer_fire():
    """Runs MOUSE_HOLD_MS after a left-down event. Fires the start
    callback only if the button is still down, the cursor hasn't drifted
    past the movement threshold, AND the click landed on / focuses an
    editable text field. Firing at the END of the hold (not at mouse-down)
    is deliberate: focus has settled by now, which is when the AX
    editability check is reliable."""
    with _mouse_click_lock:
        if not _mouse_click_state["down"]:
            return  # released before threshold
        if _mouse_click_state["moved"]:
            return  # turned into a drag, don't hijack it
        if _mouse_click_state["fired_start"]:
            return  # defensive — shouldn't happen
        x0 = _mouse_click_state["x0"]
        y0 = _mouse_click_state["y0"]
    # Editability gate runs OUTSIDE the lock — it does Accessibility IPC that
    # can take a few ms, and we must not block the mouse listener thread.
    if not _click_should_start_dictation(x0, y0):
        log.debug("mouse left-click long-press: target not editable — not starting")
        return
    with _mouse_click_lock:
        # Re-validate after the AX lookup: the button may have been released
        # or dragged while we were querying. Only commit if still a clean
        # pending hold, and only THEN mark fired_start so the release handler
        # knows a session is live (and a gated no-start can't trigger a stop).
        if (not _mouse_click_state["down"] or _mouse_click_state["moved"]
                or _mouse_click_state["fired_start"]):
            return
        _mouse_click_state["fired_start"] = True
    log.debug("mouse left-click long-press → hold-to-talk start")
    start_live_dictation()

def _mouse_on_click(x, y, button, pressed):
    if button != _pynput_mouse.Button.left:
        return
    if pressed:
        with _mouse_click_lock:
            # Cancel any previous timer (defensive — only happens if
            # someone clicks twice without a release event somehow).
            old = _mouse_click_state["timer"]
            if old is not None:
                old.cancel()
            _mouse_click_state["down"] = True
            _mouse_click_state["x0"] = x
            _mouse_click_state["y0"] = y
            _mouse_click_state["moved"] = False
            _mouse_click_state["fired_start"] = False
            t = threading.Timer(MOUSE_HOLD_MS / 1000.0, _mouse_hold_timer_fire)
            t.daemon = True
            _mouse_click_state["timer"] = t
            t.start()
    else:
        with _mouse_click_lock:
            fired = _mouse_click_state["fired_start"]
            t = _mouse_click_state["timer"]
            if t is not None:
                t.cancel()
            _mouse_click_state["down"] = False
            _mouse_click_state["timer"] = None
            _mouse_click_state["fired_start"] = False
        if fired:
            log.debug("mouse release → hold-to-talk stop")
            stop_live_dictation()

def _mouse_on_move(x, y):
    # Hot path: on_move fires constantly as the cursor wanders. Bail
    # out fast when the button isn't down.
    if not _mouse_click_state["down"]:
        return
    with _mouse_click_lock:
        if not _mouse_click_state["down"] or _mouse_click_state["moved"]:
            return
        dx = x - _mouse_click_state["x0"]
        dy = y - _mouse_click_state["y0"]
        if dx * dx + dy * dy > MOUSE_MOVE_THRESHOLD_PX ** 2:
            _mouse_click_state["moved"] = True
            t = _mouse_click_state["timer"]
            if t is not None:
                t.cancel()
                _mouse_click_state["timer"] = None
            log.debug("mouse moved %.1fpx — canceling hold-to-talk",
                      (dx * dx + dy * dy) ** 0.5)

# ── MAIN ─────────────────────────────────────────────────────────────────────
def main():
    print("""
╔══════════════════════════════════════════════╗
║   Typer — Whisper-MLX live typing           ║
╠══════════════════════════════════════════════╣
║  Hold Right-⌘ or F18 → speak → release      ║
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

    # Heartbeat watchdog — logs subsystem health every minute and
    # restarts the persistent audio stream if it silently dies.
    threading.Thread(target=_heartbeat, daemon=True, name="heartbeat").start()
    log.info("heartbeat: interval=%.0fs", HEARTBEAT_INTERVAL_S)

    log.info("🟢 ready. Hold Right-⌘ or F18 to talk, or tap F19 to toggle.")

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
        _heartbeat_stop.set()
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

    # pynput's keyboard and mouse listeners EACH call
    # HIServices.AXIsProcessTrusted() from their own thread the instant they
    # start. PyObjC resolves that lazy framework attribute with a DESTRUCTIVE
    # funcmap.pop() that isn't thread-safe: if both listener threads hit the
    # first access concurrently, the loser raises `KeyError:
    # 'AXIsProcessTrusted'` and its listener thread dies on startup — so on a
    # given launch either the hotkeys or the mouse trigger silently wouldn't
    # work. Touch it ONCE here on the main thread first; that populates
    # PyObjC's attribute cache (HIServices.__dict__) so the listener threads
    # hit the cached function and never enter the racy resolver.
    try:
        import HIServices
        HIServices.AXIsProcessTrusted()
    except Exception:
        log.debug("AXIsProcessTrusted pre-resolve failed", exc_info=True)

    # The keyboard.Listener spawns its own Quartz event-tap thread, so we
    # can start it and leave the main thread free to run AppKit's run
    # loop — which is required for the floating overlay panel. Updates
    # are marshalled onto this main loop via performSelectorOnMainThread_.
    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()

    # Mouse listener for the left-click long-press trigger. Same
    # CGEventTap-backed approach as the keyboard listener, so it runs
    # in its own background thread and doesn't block the main runloop.
    # Disabled by default (TYPER_MOUSE_CLICK_TRIGGER) so it doesn't catch
    # ordinary clicks.
    mouse_listener = None
    if MOUSE_CLICK_TRIGGER_ENABLED:
        mouse_listener = _pynput_mouse.Listener(
            on_click=_mouse_on_click,
            on_move=_mouse_on_move,
        )
        mouse_listener.start()
        log.info(
            "mouse trigger: left-click hold≥%dms (move<%.0fpx)",
            MOUSE_HOLD_MS, MOUSE_MOVE_THRESHOLD_PX,
        )
    else:
        log.info("mouse trigger: left-click long-press DISABLED "
                 "(set TYPER_MOUSE_CLICK_TRIGGER=1 to enable)")

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
