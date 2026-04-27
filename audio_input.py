import sounddevice as sd
import numpy as np

def get_input_devices():
    """Retorna una llista d'entrades d'àudio disponibles amb el seu índex i nom."""
    devices = sd.query_devices()
    return [(idx, dev) for idx, dev in enumerate(devices) if dev['max_input_channels'] > 0]

def record_audio(duration=2, fs=44100, device=None):
    print("🎙️ Gravant...")
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='float32', device=device)
    sd.wait()
    print("✅ Gravació completa")
    return recording[:, 0], fs
