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

The **voice pipeline is fixed in code to mirror [Handy](https://github.com/cjpais/Handy)** —
Silero VAD (threshold 0.3, onset/hangover smoothing) → one plain greedy
decode on release (no prior-text context, Whisper's own no-speech gate) →
filler/stutter cleanup → paste. There is nothing to tune there, and no
live/streaming mode. Only a handful of settings remain:

| Var | Default | Notes |
|---|---|---|
| `TYPER_MODEL` | `…/whisper-large-v3-turbo` | Any MLX-hosted Whisper model id (e.g. `…/whisper-large-v3-mlx` for slightly better non-English accuracy at ~2× latency) |
| `TYPER_WHISPER_LANGUAGE` | unset | `en`, `ru`, … — pin a language instead of auto-detecting |
| `TYPER_INITIAL_PROMPT` | unset | Custom words fed to Whisper so it spells your names/jargon right (Handy's "custom words") |
| `TYPER_HF_OFFLINE` | `1` when cache exists | Skip the HuggingFace API check on startup |

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

The voice pipeline is a direct port of [Handy](https://github.com/cjpais/Handy),
the most popular open-source local dictation app:

1. While you hold the key, audio passes through a Silero **VAD gate**
   (threshold 0.3) with Handy's onset/hangover smoothing — a 2-frame onset
   debounce so a single noise blip can't open the gate, ~450 ms of pre-roll
   so soft word onsets aren't clipped, and ~450 ms of hangover. Only voiced
   audio is accumulated; Whisper never sees silence.
2. **No transcription runs while you hold.** On release, the whole utterance
   is transcribed *once* with a plain greedy decode (`temperature=0`,
   `condition_on_previous_text=False`, Whisper's own `no_speech` gate at
   0.2). A clip under 1 s is zero-padded to 1.25 s so short answers decode
   cleanly.
3. The text is cleaned with Handy's language-aware **filler/stutter filter**
   (drops "uh"/"um"-type fillers for the spoken language, collapses 3+
   repeated words) and pasted in one shot.

That's the whole defense against hallucinations: the VAD keeps silence out
of Whisper. There is deliberately no energy gate and no confidence/phrase
post-filter — they were discarding legitimate short words, and Handy proves
the VAD alone is enough.

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
