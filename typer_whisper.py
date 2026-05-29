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

from pynput import keyboard
import mlx_whisper
from silero_vad import load_silero_vad

# ── CONFIG ──────────────────────────────────────────────────────────────────
# Best available Whisper on MLX. Use "mlx-community/whisper-large-v3-turbo"
# for ~2-3× faster inference at slightly lower accuracy, or
# "mlx-community/whisper-medium-mlx" / "...small-mlx" for lower memory.
MODEL_ID = os.environ.get("TYPER_MODEL", "mlx-community/whisper-large-v3-mlx")
SAMPLE_RATE = 16000

# How often to re-transcribe the growing buffer + paste the new suffix.
# Whisper is not natively streaming, so each tick re-runs the model on the
# full voiced audio captured so far. Larger = fewer revisions, more latency.
LIVE_PUSH_INTERVAL_S = float(os.environ.get("TYPER_PUSH_S", "1.0"))

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
    logging.getLogger("typer").info("Logging to %s (level=%s)", log_path, LOG_LEVEL)
    return log_path

_setup_logging()
log = logging.getLogger("typer")

# ── STATE ────────────────────────────────────────────────────────────────────
model_ready      = threading.Event()  # whisper weights warmed
model_ok         = False              # whether warm-up actually succeeded

live_active      = False
live_stop_event  = threading.Event()
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
    max_buf_samples = int(MAX_BUFFER_S * SAMPLE_RATE)
    voiced_buf = np.zeros(0, dtype=np.float32)

    def _drain_and_emit():
        nonlocal last_pasted, voiced_buf
        buf = []
        while True:
            try:
                buf.append(audio_q.get_nowait())
            except queue.Empty:
                break
        if buf:
            samples = np.concatenate(buf)
            raw_n = len(samples)
            if gate is not None:
                samples = gate.process(samples)
            if len(samples):
                log.debug("drain: raw=%d voiced=%d (%.2fs voiced)",
                          raw_n, len(samples), len(samples) / SAMPLE_RATE)
                voiced_buf = np.concatenate([voiced_buf, samples])
                if len(voiced_buf) > max_buf_samples:
                    # Drop the oldest audio so Whisper's 30s window doesn't
                    # start sliding mid-utterance. The already-pasted text
                    # before the cut won't be revisited by future diffs.
                    drop = len(voiced_buf) - max_buf_samples
                    log.debug("buffer cap hit, dropping %d oldest samples", drop)
                    voiced_buf = voiced_buf[drop:]
        if len(voiced_buf) < SAMPLE_RATE // 4:  # < 0.25s — not worth a call
            return
        t_infer = time.time()
        try:
            current = _whisper_transcribe(voiced_buf)
        except Exception:
            log.exception("whisper transcribe failed this tick")
            return
        infer_ms = (time.time() - t_infer) * 1000.0
        log.debug("whisper tick: buf=%.2fs infer=%.0fms text=%r",
                  len(voiced_buf) / SAMPLE_RATE, infer_ms, current)
        if current != last_pasted:
            last_pasted = _emit_diff(current, last_pasted)

    t_start = time.time()
    try:
        log.info("stream cfg: model=%s push=%.2fs beam=%d temp=%s best_of=%d "
                 "cond_prev=%s lang=%s halluc_silence=%s prompt=%r",
                 MODEL_ID, LIVE_PUSH_INTERVAL_S, WHISPER_BEAM_SIZE,
                 WHISPER_TEMPERATURE, WHISPER_BEST_OF,
                 WHISPER_CONDITION_ON_PREV, WHISPER_LANGUAGE or "auto",
                 _WHISPER_HALLUCINATION_SILENCE_PARSED, WHISPER_INITIAL_PROMPT)
        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
            blocksize=int(SAMPLE_RATE * 0.1),
            callback=_audio_cb,
        ):
            log.debug("audio input opened sr=%d", SAMPLE_RATE)
            while not live_stop_event.is_set():
                time.sleep(LIVE_PUSH_INTERVAL_S)
                iters += 1
                t_tick = time.time()
                _drain_and_emit()
                tick_ms = (time.time() - t_tick) * 1000.0
                try:
                    active_mb = mx.get_active_memory() / (1024 * 1024)
                    peak_mb = mx.get_peak_memory() / (1024 * 1024)
                    log.debug("tick %d: drain+infer=%.0fms mlx_active=%.0fMB peak=%.0fMB",
                              iters, tick_ms, active_mb, peak_mb)
                except Exception:
                    log.debug("tick %d: drain+infer=%.0fms", iters, tick_ms)
            log.debug("final flush after %d iters", iters)
            _drain_and_emit()
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
        if not model_ready.is_set() or not model_ok:
            log.warning("model not loaded; ignoring start")
            return
        live_stop_event.clear()
        live_active = True
    log.info("🎤 Live — speak now")
    threading.Thread(target=lambda: subprocess.run(
        ['afplay', '/System/Library/Sounds/Blow.aiff'],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL),
        daemon=True, name="sound-start").start()
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
    threading.Thread(target=lambda: subprocess.run(
        ['afplay', '/System/Library/Sounds/Funk.aiff'],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL),
        daemon=True, name="sound-stop").start()

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
