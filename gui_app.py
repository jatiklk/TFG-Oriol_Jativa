import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import numpy as np
from audio_input import record_audio
from analyzer_THD import plot_fft, extract_impulse_response, compute_THD

class AudioApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Audio Analyzer")
        self.geometry("900x700")

        self.record_btn = ttk.Button(self, text="Enregistra àudio", command=self.record_and_plot)
        self.record_btn.pack(pady=20)

        self.upload_btn = ttk.Button(self, text="Carrega àudio (WAV)", command=self.upload_and_analyze)
        self.upload_btn.pack(pady=10)

        self.thd_label = ttk.Label(self, text="THD: -")
        self.thd_label.pack(pady=10)

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
        # Calcula i mostra el THD de la gravació
        thd = compute_THD(signal, fs)
        self.thd_label.config(text=f"THD (gravació): {thd:.2f}%")

    def upload_and_analyze(self):
        self.ax1.clear()
        self.ax2.clear()
        file_path = filedialog.askopenfilename(filetypes=[("WAV files", "*.wav")])
        if not file_path:
            return
        try:
            from scipy.io import wavfile
            fs, data = wavfile.read(file_path)
            # Si l'àudio és estèreo, agafa només un canal
            if len(data.shape) > 1:
                data = data[:, 0]
            # Normalitza si és enter
            if data.dtype != np.float32 and data.dtype != np.float64:
                data = data.astype(np.float32) / np.max(np.abs(data))
            signal = data
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
            # Calcula i mostra el THD
            thd = compute_THD(signal, fs)
            self.thd_label.config(text=f"THD (fitxer): {thd:.2f}%")
        except Exception as e:
            self.thd_label.config(text=f"Error: {e}")

if __name__ == "__main__":
    app = AudioApp()
    app.mainloop()
