#!/usr/bin/env python3
"""Prints every key press/release pynput sees. Ctrl+C to quit.
Hold F1 (and try other keys) to see what they report."""
from pynput import keyboard

def on_press(key):
    print(f"PRESS   {key!r}")

def on_release(key):
    print(f"RELEASE {key!r}")
    if key == keyboard.Key.esc:
        return False

print("Press keys (hold F1, try F2..F12, etc). Esc or Ctrl+C to quit.")
with keyboard.Listener(on_press=on_press, on_release=on_release) as l:
    l.join()
