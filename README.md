# Typer

Hold-to-talk live voice typing for macOS. Apple Silicon native.

**Hold F1 → speak → words stream into the focused field in real time.
Release F1 → stop.**

## Model

`nvidia/parakeet-tdt-0.6b-v2` via [parakeet-mlx](https://github.com/senstella/parakeet-mlx)
— state-of-the-art ASR (beats Whisper large-v3 on word-error-rate), runs natively
on the Apple Silicon GPU through MLX. First run downloads ~1.2 GB to `~/.cache/`.

## Setup

```bash
cd ~/work/typer
/opt/homebrew/bin/python3.12 -m venv .venv   # needs Python ≥ 3.10
. .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
./run.sh
```

## Permissions (System Settings → Privacy & Security)

Grant the venv Python binary (`~/work/typer/.venv/bin/python3.12`):
- **Microphone**
- **Accessibility** (keystroke injection for paste/backspace)
- **Input Monitoring** (global F1 hotkey)

## Config (env vars)

- `TYPER_PUSH_S=0.4` — how often audio is pushed to the model + the field
  updated. Lower = snappier, more CPU.
- `TYPER_MODEL=mlx-community/parakeet-tdt-0.6b-v2` — any parakeet-mlx model id.

## How it works

True streaming: a `sounddevice` mic stream feeds raw audio to Parakeet's
incremental decoder. Each tick, the new full transcript is diffed against
what's already in the field — diverged tails are erased with backspaces and
the corrected tail is pasted. So self-corrections by the decoder are reflected
in place rather than duplicated.

## Notes / caveats

- While holding F1, don't click into a different field — the backspace/paste
  keystrokes go wherever the cursor is.
- If F1 triggers macOS Help or a media action instead, enable
  **System Settings → Keyboard → "Use F1, F2, etc. keys as standard function
  keys"** (or remap as preferred).
