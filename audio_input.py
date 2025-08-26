import sounddevice as sd
import numpy as np

def record_audio(duration=2, fs=44100):
    print("🎙️ Gravant...")
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1)
    sd.wait()
    print("✅ Gravació completa")
    return recording[:, 0], fs
