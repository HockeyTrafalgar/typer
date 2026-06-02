# Typer

Hold-to-talk voice dictation for macOS. Apple Silicon native, powered by
[`mlx-whisper`](https://github.com/ml-explore/mlx-examples/tree/main/whisper)
(`mlx-community/whisper-large-v3-mlx`) — state-of-the-art accuracy across
~99 languages.

> **Hold Right-⌘ → speak → release → text gets pasted into the focused
> field.** Tap F19 to toggle on/off if you'd rather not hold a key.

A small floating glass HUD appears next to the focused text caret while
you talk, showing the live transcript. Press **Esc** to cancel without
pasting.

Whisper isn't a true streaming model, so Typer captures audio into a
rolling buffer while you hold the key, shows live previews in the HUD,
and pastes the result in one shot on release (batch mode) — avoiding the
diff-flicker and stuck-modifier edge cases of true streaming.

---

## Setup

Requires **macOS with Apple Silicon** and **Python ≥ 3.10**.

```bash
git clone https://github.com/HockeyTrafalgar/typer.git
cd typer

/opt/homebrew/bin/python3.12 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
```

### macOS permissions

Grant the **venv Python binary** (`./.venv/bin/python3.12`) the
following permissions in **System Settings → Privacy & Security**:

| Permission | Why |
|---|---|
| **Microphone** | Recording your voice |
| **Accessibility** | Synthetic keystrokes for paste / backspace, and reading the text-caret position for HUD placement |
| **Input Monitoring** | Global hotkey listener (Right-⌘ / F19) |

The system will prompt the first time each permission is needed —
accept, then restart the script.

---

## Run

```bash
./run_whisper.sh
```

Hotkeys:
- **Hold Right-⌘** (or **F18**) — record while held, paste on release.
- **Tap F19** — toggle on/off (start with first tap, stop with second).
- **Press Esc** while recording — cancel the session, skip the paste.
- **Ctrl+C in the terminal** — quit.

Typer uses Apple's **Liquid Glass** (`NSGlassEffectView`, macOS 26+) for
the floating HUD when available, falling back to the older
`NSVisualEffectView` blur on earlier macOS.

---

## Configuration

Everything is environment-variable driven so you can iterate without
touching the source. Most useful knobs:

### Model / accuracy

| Var | Default | Notes |
|---|---|---|
| `TYPER_MODEL` | `…/whisper-large-v3-mlx` | Any MLX-hosted Whisper model id (e.g. `…/whisper-large-v3-turbo` for ~3× faster) |
| `TYPER_MODE` | `batch` | `batch` — paste once on release; `stream` — diff-paste live |
| `TYPER_WHISPER_LANGUAGE` | unset | `en`, `ru`, … — skip auto-detect |
| `TYPER_INITIAL_PROMPT` | unset | Vocabulary/style prime — best lever for names & jargon |
| `TYPER_BEST_OF` | `5` | Number of samples to draw at each fallback temperature |
| `TYPER_TEMPERATURE` | `0.0,0.2,…,1.0` | Whisper fallback temperatures |
| `TYPER_HF_OFFLINE` | `1` when cache exists | Skip HuggingFace API check on startup |

### Latency

| Var | Default | Notes |
|---|---|---|
| `TYPER_PUSH_S` | `0.35` | Minimum interval between transcribe ticks |
| `TYPER_MAX_BUFFER_S` | `28.0` | Cap on rolling audio buffer (Whisper context is 30 s) |

### VAD (voice activity detection)

| Var | Default | Notes |
|---|---|---|
| `TYPER_VAD_THRESHOLD` | `0.5` | Silero speech-probability threshold |
| `TYPER_VAD_PRE_ROLL_MS` | `500` | Audio retained before speech onset |
| `TYPER_VAD_HANGOVER_MS` | `400` | Silence required to mark end of utterance |
| `TYPER_VAD_RMS_FLOOR` | `0.005` | Minimum RMS energy before VAD even runs |

### Overlay HUD

| Var | Default | Notes |
|---|---|---|
| `TYPER_OVERLAY` | `1` | Set to `0` to disable the HUD entirely |
| `TYPER_OVERLAY_PLACEMENT` | `caret` | `caret`, `cursor`, or `bottom` |
| `TYPER_OVERLAY_GLASS` | `1` | Set to `0` to force the legacy vibrancy material |
| `TYPER_OVERLAY_W` | `780` | |
| `TYPER_OVERLAY_H` | `72` | Single-line height; HUD grows past this as text wraps |
| `TYPER_OVERLAY_MAX_H` | `320` | Cap on growth |
| `TYPER_OVERLAY_CORNER` | `36` | Corner radius |
| `TYPER_OVERLAY_FONT_SIZE` | `19` | |
| `TYPER_OVERLAY_ALPHA` | `1.0` | Whole-panel transparency |

### Logging

| Var | Default | Notes |
|---|---|---|
| `TYPER_LOG_LEVEL` | `DEBUG` | One of `DEBUG`, `INFO`, `WARNING`, `ERROR` |
| `TYPER_LOG_DIR` | `./logs` | Log files rotate at 5 MB × 5 |

Read the top of [`typer_whisper.py`](typer_whisper.py) for the full list
of knobs and what they do.

---

## How it works

Whisper isn't a true streaming model, so we
approximate it: while you hold Right-⌘, audio is captured into a
rolling buffer and a Liquid-Glass HUD shows live previews from
re-transcribing the buffer every tick. On release we either reuse the
most recent preview (when no significant new audio arrived) or run one
final whisper pass, then paste the result. Batch-paste avoids the
diff-flicker and stuck-modifier edge cases of true streaming. Caret
position for the HUD comes from the macOS Accessibility API
(`AXBoundsForRange` on the focused text element), with a graceful
fallback to the mouse cursor when the focused app doesn't expose it.

---

## Caveats

- While holding Right-⌘, don't click into a different field — synthetic
  keystrokes always go wherever the cursor is.
- The HUD anchors to the caret on most native macOS text fields
  (Safari, Chrome inputs, Notes, TextEdit, terminals, most Electron
  apps). Some GPU-rendered text views don't expose `AXBoundsForRange`;
  the HUD falls back to mouse-cursor placement.
- If F19 triggers macOS Help / a media action instead of reaching the
  app, enable **System Settings → Keyboard → "Use F1, F2, etc. keys as
  standard function keys"** (or remap to a key you don't use, e.g. F13).
- Tail words you spoke in the last ~0.4 s before release might be cut
  from the paste in batch mode — adjust `TYPER_PUSH_S` lower if this
  bites you.

---

## Acknowledgements & disclosure

Built incrementally in pair-programming sessions with
[Claude Code](https://claude.com/claude-code).
A lot of the code, comments, and commit messages were drafted by the
model and then reviewed and integrated by hand. If you're spelunking
the diff you'll see Co-Authored-By trailers crediting the assist.

Upstream credit:
- [`mlx-whisper`](https://github.com/ml-explore/mlx-examples/tree/main/whisper) — Whisper on MLX
- [`silero-vad`](https://github.com/snakers4/silero-vad) — voice activity detection

## License

[MIT](LICENSE).
