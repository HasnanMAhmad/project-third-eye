#!/usr/bin/env python3
"""
Audio Diagnostic Utility for Project Third Eye
Plays back the primary audio directives to verify system speaker levels and TTS output.
"""

import time
import sys

try:
    import pyttsx3
except ImportError:
    print("Error: pyttsx3 is not installed in the virtual environment.", file=sys.stderr)
    sys.exit(1)

def run_audio_diagnostic():
    print("=" * 60)
    print(" PROJECT THIRD EYE - AUDIO DIAGNOSTIC TEST")
    print("=" * 60)
    print("Initializing Microsoft SAPI5 speech engine...")

    try:
        engine = pyttsx3.init()
        engine.setProperty("rate", 165)
        engine.setProperty("volume", 1.0)
    except Exception as e:
        print(f"Failed to initialize pyttsx3: {e}", file=sys.stderr)
        return

    phrases = [
        "Project Third Eye audio diagnostic test. System online.",
        "Obstacle ahead. Veer Left.",
        "STOP! Obstacle directly ahead!"
    ]

    for i, phrase in enumerate(phrases, 1):
        print(f"[{i}/3] Speaking: \"{phrase}\" ...", flush=True)
        try:
            engine.say(phrase)
            engine.runAndWait()
        except Exception as err:
            print(f"Error during playback: {err}", file=sys.stderr)
        time.sleep(1.0)

    print("\nDiagnostic complete! If you heard all 3 phrases, your audio is 100% operational.")

if __name__ == "__main__":
    run_audio_diagnostic()
