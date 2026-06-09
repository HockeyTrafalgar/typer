#!/bin/bash
cd "$(dirname "$0")"
. .venv/bin/activate
# Load local overrides from .env (if present). `set -a` exports every
# assignment so the Python process inherits them; commented lines are
# ignored. Pre-existing environment variables still take precedence is NOT
# guaranteed here — .env wins, which is what you want when tuning.
if [ -f .env ]; then
  set -a
  . ./.env
  set +a
fi
exec python typer_whisper.py
