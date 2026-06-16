# Typer

Hold-to-talk voice dictation for macOS. Apple Silicon native, powered by
[`mlx-whisper`](https://github.com/ml-explore/mlx-examples/tree/main/whisper)
(`mlx-community/whisper-large-v3-turbo`) — fast, accurate transcription
across ~99 languages.

> **Hold Right-⌘ → speak → release → text gets pasted into the focused
> field.** Tap F19 to toggle on/off if you'd rather not hold a key.

A small floating glass HUD appears next to the focused text caret while
you talk, with a live mic-activity meter so you can see the microphone is
picking up sound. Press **Esc** to cancel without pasting.

Whisper isn't a streaming model, so by default Typer is **true
push-to-talk**: while you hold the key it just accumulates voice-activity-
gated audio (no transcription), then transcribes the whole utterance **once**
on release and pastes it. This is the design popular local Whisper dictation
tools use (Handy, WhisperWriter) — it sidesteps the silence hallucinations
("Thank you", "Thanks for watching") and post-release lag that come from
re-transcribing a growing buffer. A live-streaming mode is available too
(see `TYPER_MODE`).

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
| `TYPER_MODEL` | `…/whisper-large-v3-turbo` | Any MLX-hosted Whisper model id (e.g. `…/whisper-large-v3-mlx` for max accuracy at ~2× latency) |
| `TYPER_MODE` | `release` | `release` — accumulate while held, transcribe once on release, paste (recommended); `live` — re-transcribe & diff-paste as you speak; `both` — stream live, then an authoritative correction pass on release. (`batch`→`release`, `stream`→`live` still accepted) |
| `TYPER_WHISPER_LANGUAGE` | unset | `en`, `ru`, … — skip auto-detect |
| `TYPER_INITIAL_PROMPT` | unset | Vocabulary/style prime — best lever for names & jargon |
| `TYPER_CONDITION_ON_PREV` | `0` | Feed prior output back as conditioning. Off by default — on amplifies repetition loops/hallucinations |
| `TYPER_HALLUCINATION_FILTER` | `0` | Master switch for the DOWNSTREAM text/segment hallucination filters (per-segment confidence drop + stock-phrase blocklist). **Off by default** — these were discarding legitimate short utterances ("yes", "no", "you", "bye"). Silence is still suppressed upstream by VAD + `TYPER_MIN_SPEECH_RMS` and Whisper's own `no_speech_threshold`. Set `1` to restore aggressive post-filtering for very noisy mics |
| `TYPER_STRIP_HALLUCINATIONS` | `1` | Drop a transcription whose ENTIRE text is a stock hallucination phrase ("Thank you", "Thanks for watching", …). Only consulted when `TYPER_HALLUCINATION_FILTER=1` |
| `TYPER_BEST_OF` | `5` | Number of samples to draw at each fallback temperature |
| `TYPER_TEMPERATURE` | `0.0,0.2` | Whisper fallback temperatures (kept short to bound worst-case re-decode latency) |
| `TYPER_HF_OFFLINE` | `1` when cache exists | Skip HuggingFace API check on startup |

### Latency

| Var | Default | Notes |
|---|---|---|
| `TYPER_PUSH_S` | `0.35` | Capture-loop tick interval (re-transcribe cadence in `live`/`both`; just audio-drain in `release`) |
| `TYPER_MAX_BUFFER_S` | `28.0` | Cap on rolling audio buffer (Whisper context is 30 s) |

### VAD (voice activity detection)

| Var | Default | Notes |
|---|---|---|
| `TYPER_VAD_THRESHOLD` | `0.5` | Silero speech-probability threshold |
| `TYPER_VAD_PRE_ROLL_MS` | `500` | Audio retained before speech onset |
| `TYPER_VAD_HANGOVER_MS` | `400` | Silence required to mark end of utterance |
| `TYPER_VAD_RMS_FLOOR` | `0.005` | Minimum RMS energy before VAD even runs |

### Mouse trigger (optional alternative to the hotkeys)

Left-click-and-hold to start dictation, release to stop. Off by default.

| Var | Default | Notes |
|---|---|---|
| `TYPER_MOUSE_CLICK_TRIGGER` | `0` | `1` enables left-click long-press as a trigger |
| `TYPER_MOUSE_HOLD_MS` | `700` | How long the click must be held before dictation starts |
| `TYPER_MOUSE_MOVE_PX` | `5` | Pointer drift that cancels the trigger (keeps click-drag working) |

The mouse trigger only starts dictation when the click lands on / focuses an
editable text field (via the Accessibility API), so clicking buttons, links,
or empty areas won't start it. It's permissive — in apps that don't expose
accessibility (Electron, some web fields, GPU terminals) it still starts,
since it can't positively rule out a text field.

### Overlay HUD

| Var | Default | Notes |
|---|---|---|
| `TYPER_OVERLAY` | `1` | Set to `0` to disable the HUD entirely |
| `TYPER_OVERLAY_LEVEL_METER` | `1` | Live mic-activity meter (EQ bars) so you can see the mic is hearing you; `0` hides it |
| `TYPER_MIC_LEVEL_REF` | `0.12` | RMS mapped to a full meter; lower = more sensitive |
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

While you hold Right-⌘, audio is captured and passed through a Silero
**voice-activity-detection (VAD) gate** that strips silence so Whisper
never sees non-speech (the main source of hallucinations). In the default
`release` mode that's *all* that happens during the hold — no Whisper runs
until you let go, at which point a single pass transcribes the whole
silence-trimmed utterance and the text is pasted in one shot. Because no
per-tick transcription ever ran, there's no in-flight inference to wait out
on release and no partial-window hallucinations to filter. For dictation
longer than Whisper's ~30 s context window, the buffer is flushed at VAD
silence boundaries and each chunk is transcribed exactly once.

`live`/`both` modes additionally re-transcribe the growing buffer every
tick and diff-paste the new suffix for real-time feedback (`both` then runs
one authoritative pass on release to correct it). Decoding uses greedy
`temperature=0` with a short fallback and `condition_on_previous_text=False`
(the most-cited anti-repetition lever); degenerate segments are dropped via
Whisper's own `no_speech_prob`/`avg_logprob`/`compression_ratio` metrics.

Text is delivered by copying it to the clipboard and synthesizing Cmd+V.
That would clobber whatever you had copied, so by default Typer snapshots
the clipboard when a dictation session starts and restores it once the final
paste has landed — anything you'd copied before dictating is still there to
paste afterwards. It only restores if our pasted text is still on the
clipboard, so it never overwrites something you copied in the meantime.
Disable with `TYPER_RESTORE_CLIPBOARD=0`.

Caret position for the HUD comes from the macOS Accessibility API
(`AXBoundsForRange` on the focused text element), with a graceful fallback
to the mouse cursor when the focused app doesn't expose it.

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
- In the default `release` mode the whole utterance is transcribed on
  release, so no tail words are lost. In `live` mode, words spoken in the
  last ~0.4 s before release can occasionally be cut from the diff — use
  `release` (or `both`) if that bites you.

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
