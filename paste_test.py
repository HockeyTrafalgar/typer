#!/usr/bin/env python3
"""Isolated test of typer's paste path: clipboard copy + synthetic Cmd+V.

Run it, then within 4 seconds click into any text field (Notes, a browser
box, this terminal). If "TYPER_PASTE_OK <n>" appears, synthetic keystroke
posting works. If nothing appears, macOS is blocking synthetic keystrokes
for the process that launched this — re-grant Accessibility (see below).
"""
import time
import pyperclip
from pynput import keyboard

_kbd = keyboard.Controller()
MARKER = "TYPER_PASTE_OK 42"

print("Focus a text field now. Pasting in 4s...")
time.sleep(4)

pyperclip.copy(MARKER)
print(f"clipboard set to: {MARKER!r}")
time.sleep(0.05)
# release any stray cmd, then send Cmd+V exactly like typer does
try:
    _kbd.release(keyboard.Key.cmd)
except Exception:
    pass
with _kbd.pressed(keyboard.Key.cmd):
    _kbd.press('v')
    _kbd.release('v')
print("Cmd+V sent. Did the marker appear in your text field?")
