import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import numpy as np
from audio_input import record_audio
from analyzer import plot_fft, extract_impulse_response

class AudioApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Audio Analyzer")
        self.geometry("900x700")

        self.record_btn = ttk.Button(self, text="Enregistra àudio", command=self.record_and_plot)
        self.record_btn.pack(pady=20)

        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(8, 8))
        self.fig.tight_layout(pad=4.0)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def record_and_plot(self):
        self.ax1.clear()
        self.ax2.clear()
        signal, fs = record_audio(duration=2)
        
        # FFT plot
        N = len(signal)
        spectrum = abs(np.fft.fft(signal)[:N//2])
        freqs = np.fft.fftfreq(N, 1/fs)[:N//2]
        self.ax1.plot(freqs, spectrum)
        self.ax1.set_title("Resposta en freqüència (FFT)")
        self.ax1.set_xlabel("Freqüència (Hz)")
        self.ax1.set_ylabel("Amplitud")

        # Impulse response plot
        impulse_response = extract_impulse_response(signal, fs)
        self.ax2.plot(impulse_response)
        self.ax2.set_title("Impulse Response")
        self.ax2.set_xlabel("Mostres")
        self.ax2.set_ylabel("Amplitud")

        self.canvas.draw()

if __name__ == "__main__":
    app = AudioApp()
    app.mainloop()
