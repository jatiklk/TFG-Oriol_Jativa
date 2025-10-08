# main.py
import audio_dspy
from audio_dspy.farina import generate_farina_sweep
from audio_input import record_audio
from analyzer import plot_fft, extract_fundamental_note, extract_impulse_response
from gui_app import AudioApp
import matplotlib.pyplot as plt

def main_console():
    signal, fs = record_audio(duration=2)
    plot_fft(signal, fs)

    # Extracció de la nota fonamental
    fundamental_freq, note_name = extract_fundamental_note(signal, fs)
    print(f"Nota fonamental: {note_name} ({fundamental_freq:.2f} Hz)")

    # Extracció de la impulse response
    impulse_response = extract_impulse_response(signal, fs)
    print("Impulse response extreta.")

    # Mostra la impulse response amb un gràfic
    plt.plot(impulse_response)
    plt.title("Impulse Response")
    plt.xlabel("Mostres")
    plt.ylabel("Amplitud")
    plt.show()

if __name__ == "__main__":
    USE_GUI = True  # Canvia a False per executar en mode consola

    if USE_GUI:
        app = AudioApp()
        app.mainloop()
    else:
        main_console()
