import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
from tkinter import simpledialog
from tkinter import messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile
from audio_input import record_audio
from analyzer_THD import plot_fft, extract_impulse_response, compute_THD_F
from analyzer_IMD import compute_IMD_smpte, compute_IMD_ccif
from generator import sine, sweep_generator, sweep_generator_linear

class AudioApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Audio Analyzer")
        self.geometry("900x700")
        self.selected_analysis_type = None
        self.selected_thd_type = None
        self.signal = None
        self.fs = None
        
        # Mostrar la pantalla inicial de selecció
        self.show_selection_screen()
    
    def show_selection_screen(self):
        """Mostra la pantalla inicial per seleccionar entre THD i IMD"""
        # Esborrar els widgets anteriors
        for widget in self.winfo_children():
            widget.destroy()
        
        # Frame principal
        main_frame = ttk.Frame(self, padding="20")
        main_frame.pack(expand=True, fill=tk.BOTH)
        
        # Títol
        title_label = ttk.Label(main_frame, text="Selecciona el tipus d'anàlisi", 
                                font=("Arial", 16, "bold"))
        title_label.pack(pady=20)
        
        # Botó THD
        thd_btn = ttk.Button(main_frame, text="THD", 
                            command=lambda: self.select_analysis_type("THD"),
                            width=20)
        thd_btn.pack(pady=10)
        
        # Botó IMD
        imd_btn = ttk.Button(main_frame, text="IMD", 
                            command=lambda: self.select_analysis_type("IMD"),
                            width=20)
        imd_btn.pack(pady=10)
    
    def select_analysis_type(self, analysis_type):
        """Gestiona la selecció del tipus d'anàlisi"""
        self.selected_analysis_type = analysis_type
        
        if analysis_type == "THD":
            self.show_thd_selection_screen()
        elif analysis_type == "IMD":
            self.show_imd_screen()
    
    def show_thd_selection_screen(self):
        """Mostra la pantalla de selecció del tipus de THD"""
        # Esborrar els widgets anteriors
        for widget in self.winfo_children():
            widget.destroy()
        
        # Frame principal
        main_frame = ttk.Frame(self, padding="20")
        main_frame.pack(expand=True, fill=tk.BOTH)
        
        # Botó per tornar enrere
        back_btn = ttk.Button(main_frame, text="← Tornar", 
                             command=self.show_selection_screen)
        back_btn.pack(anchor=tk.NW, pady=(0, 20))
        
        # Títol
        title_label = ttk.Label(main_frame, text="Selecciona el tipus de THD", 
                               font=("Arial", 14, "bold"))
        title_label.pack(pady=20)
        
        # Frame per al desplegable
        dropdown_frame = ttk.Frame(main_frame)
        dropdown_frame.pack(pady=20)
        
        # Etiqueta del desplegable
        label = ttk.Label(dropdown_frame, text="Tipus de THD:")
        label.pack(side=tk.LEFT, padx=5)
        
        # Desplegable amb les opcions
        thd_options = ["THD_F", "THD_RMS", "THD_N", "THD_SWEEP"]
        self.thd_dropdown = ttk.Combobox(dropdown_frame, values=thd_options, 
                                         state="readonly", width=20)
        self.thd_dropdown.set(thd_options[0])  # Valor per defecte
        self.thd_dropdown.pack(side=tk.LEFT, padx=5)
        
        # Botó per continuar
        continue_btn = ttk.Button(main_frame, text="Continuar", 
                                 command=self.proceed_with_thd_analysis,
                                 width=20)
        continue_btn.pack(pady=20)
    
    def proceed_with_thd_analysis(self):
        """Continua amb l'anàlisi de THD seleccionat"""
        self.selected_thd_type = self.thd_dropdown.get()
        
        # Si és THD_SWEEP, anar directament a la pantalla d'anàlisi
        # Si és THD_F, THD_N o THD_RMS, mostrar opcions de generar o importar
        if self.selected_thd_type == "THD_SWEEP":
            self.show_analysis_screen()
        else:
            self.show_signal_input_screen()
    
    def show_signal_input_screen(self):
        """Mostra la pantalla per seleccionar generar o importar una senyal"""
        # Esborrar els widgets anteriors
        for widget in self.winfo_children():
            widget.destroy()
        
        # Frame principal
        main_frame = ttk.Frame(self, padding="20")
        main_frame.pack(expand=True, fill=tk.BOTH)
        
        # Botó per tornar enrere
        back_btn = ttk.Button(main_frame, text="← Tornar", 
                             command=self.show_thd_selection_screen)
        back_btn.pack(anchor=tk.NW, pady=(0, 20))
        
        # Títol
        title_label = ttk.Label(main_frame, text="Selecciona la font de la senyal", 
                               font=("Arial", 14, "bold"))
        title_label.pack(pady=20)
        
        # Botó generar sinusoide
        generate_btn = ttk.Button(main_frame, text="Generar sinusoide", 
                                 command=self.show_sine_generation_dialog,
                                 width=25)
        generate_btn.pack(pady=10)
        
        # Botó importar WAV
        import_btn = ttk.Button(main_frame, text="Importar WAV", 
                               command=self.import_wav_signal,
                               width=25)
        import_btn.pack(pady=10)
    
    def show_sine_generation_dialog(self):
        """Mostra un diàleg per generar una sinusoide"""
        # Crear una finestra secundària
        dialog = tk.Toplevel(self)
        dialog.title("Generar sinusoide")
        dialog.geometry("400x300")
        dialog.transient(self)
        dialog.grab_set()
        
        # Centrar la finestra respecte a la principal
        dialog.update_idletasks()
        x = self.winfo_x() + (self.winfo_width() // 2) - (dialog.winfo_width() // 2)
        y = self.winfo_y() + (self.winfo_height() // 2) - (dialog.winfo_height() // 2)
        dialog.geometry(f"+{x}+{y}")
        
        # Frame principal
        main_frame = ttk.Frame(dialog, padding="15")
        main_frame.pack(expand=True, fill=tk.BOTH)
        
        # Freqüència
        freq_label = ttk.Label(main_frame, text="Freqüència (Hz):")
        freq_label.grid(row=0, column=0, sticky=tk.W, pady=10)
        freq_entry = ttk.Entry(main_frame, width=20)
        freq_entry.insert(0, "1000")
        freq_entry.grid(row=0, column=1, pady=10)
        
        # Amplitud
        amp_label = ttk.Label(main_frame, text="Amplitud (0-1):")
        amp_label.grid(row=1, column=0, sticky=tk.W, pady=10)
        amp_entry = ttk.Entry(main_frame, width=20)
        amp_entry.insert(0, "0.9")
        amp_entry.grid(row=1, column=1, pady=10)
        
        # Durada
        dur_label = ttk.Label(main_frame, text="Durada (segons):")
        dur_label.grid(row=2, column=0, sticky=tk.W, pady=10)
        dur_entry = ttk.Entry(main_frame, width=20)
        dur_entry.insert(0, "2")
        dur_entry.grid(row=2, column=1, pady=10)
        
        # Freqüència de mostratge
        fs_label = ttk.Label(main_frame, text="Freq. mostratge (Hz):")
        fs_label.grid(row=3, column=0, sticky=tk.W, pady=10)
        fs_options = ["44100", "48000", "96000", "192000"]
        fs_dropdown = ttk.Combobox(main_frame, values=fs_options, state="readonly", width=17)
        fs_dropdown.set("48000")
        fs_dropdown.grid(row=3, column=1, pady=10)
        
        def generate_and_save():
            try:
                freq = float(freq_entry.get())
                amp = float(amp_entry.get())
                dur = float(dur_entry.get())
                fs = int(fs_dropdown.get())
                
                # Validacions
                if amp < 0 or amp > 1:
                    messagebox.showerror("Error", "L'amplitud ha de ser entre 0 i 1")
                    return
                if freq <= 0:
                    messagebox.showerror("Error", "La freqüència ha de ser positiva")
                    return
                if dur <= 0:
                    messagebox.showerror("Error", "La durada ha de ser positiva")
                    return
                
                # Generar sinusoide
                signal = sine(fs, dur, freq, amp)
                
                # Demanar on guardar
                file_path = filedialog.asksaveasfilename(
                    defaultextension=".wav",
                    filetypes=[("WAV files", "*.wav"), ("All files", "*.*")],
                    initialfile=f"sine_{freq}Hz_{amp}amp.wav"
                )
                
                if file_path:
                    # Guardar el fitxer
                    wavfile.write(file_path, fs, (signal * 32767).astype(np.int16))
                    messagebox.showinfo("Èxit", f"Sinusoide guardada a:\n{file_path}")
                    self.signal = signal
                    self.fs = fs
                    dialog.destroy()
                    self.show_analysis_screen()
            except ValueError:
                messagebox.showerror("Error", "Els paràmetres introduïts no són vàlids")
        
        # Botó generar
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=4, column=0, columnspan=2, pady=20)
        
        generate_btn = ttk.Button(button_frame, text="Generar i guardar", 
                                 command=generate_and_save)
        generate_btn.pack(side=tk.LEFT, padx=5)
        
        cancel_btn = ttk.Button(button_frame, text="Cancelar", 
                               command=dialog.destroy)
        cancel_btn.pack(side=tk.LEFT, padx=5)
    
    def import_wav_signal(self):
        """Importa un fitxer WAV"""
        file_path = filedialog.askopenfilename(filetypes=[("WAV files", "*.wav")])
        if not file_path:
            return
        try:
            fs, data = wavfile.read(file_path)
            # Si l'àudio és estèreo, agafa només un canal
            if len(data.shape) > 1:
                data = data[:, 0]
            # Normalitza si és enter
            if data.dtype != np.float32 and data.dtype != np.float64:
                data = data.astype(np.float32) / np.max(np.abs(data))
            self.signal = data.astype(np.float64)
            self.fs = fs
            self.show_analysis_screen()
        except Exception as e:
            messagebox.showerror("Error", f"Error al carregar el fitxer: {e}")
    
    def show_sweep_generation_dialog(self):
        """Mostra un diàleg per generar un sweep"""
        # Crear una finestra secundària
        dialog = tk.Toplevel(self)
        dialog.title("Generar sweep")
        dialog.geometry("400x350")
        dialog.transient(self)
        dialog.grab_set()
        
        # Centrar la finestra respecte a la principal
        dialog.update_idletasks()
        x = self.winfo_x() + (self.winfo_width() // 2) - (dialog.winfo_width() // 2)
        y = self.winfo_y() + (self.winfo_height() // 2) - (dialog.winfo_height() // 2)
        dialog.geometry(f"+{x}+{y}")
        
        # Frame principal
        main_frame = ttk.Frame(dialog, padding="15")
        main_frame.pack(expand=True, fill=tk.BOTH)
        
        # Freqüència inicial
        f_start_label = ttk.Label(main_frame, text="Freq. inicial (Hz):")
        f_start_label.grid(row=0, column=0, sticky=tk.W, pady=10)
        f_start_entry = ttk.Entry(main_frame, width=20)
        f_start_entry.insert(0, "20")
        f_start_entry.grid(row=0, column=1, pady=10)
        
        # Freqüència final
        f_end_label = ttk.Label(main_frame, text="Freq. final (Hz):")
        f_end_label.grid(row=1, column=0, sticky=tk.W, pady=10)
        f_end_entry = ttk.Entry(main_frame, width=20)
        f_end_entry.insert(0, "20000")
        f_end_entry.grid(row=1, column=1, pady=10)
        
        # Durada
        dur_label = ttk.Label(main_frame, text="Durada (segons):")
        dur_label.grid(row=2, column=0, sticky=tk.W, pady=10)
        dur_entry = ttk.Entry(main_frame, width=20)
        dur_entry.insert(0, "5")
        dur_entry.grid(row=2, column=1, pady=10)
        
        # Freqüència de mostratge
        fs_label = ttk.Label(main_frame, text="Freq. mostratge (Hz):")
        fs_label.grid(row=3, column=0, sticky=tk.W, pady=10)
        fs_options = ["44100", "48000", "96000", "192000"]
        fs_dropdown = ttk.Combobox(main_frame, values=fs_options, state="readonly", width=17)
        fs_dropdown.set("48000")
        fs_dropdown.grid(row=3, column=1, pady=10)
        
        # Tipus de sweep (logarítmic o lineal)
        sweep_type_label = ttk.Label(main_frame, text="Tipus de sweep:")
        sweep_type_label.grid(row=4, column=0, sticky=tk.W, pady=10)
        sweep_options = ["Logarítmic", "Lineal"]
        sweep_dropdown = ttk.Combobox(main_frame, values=sweep_options, state="readonly", width=17)
        sweep_dropdown.set("Logarítmic")
        sweep_dropdown.grid(row=4, column=1, pady=10)
        
        def generate_and_save():
            try:
                f_start = float(f_start_entry.get())
                f_end = float(f_end_entry.get())
                dur = float(dur_entry.get())
                fs = int(fs_dropdown.get())
                sweep_type = sweep_dropdown.get()
                
                # Validacions
                if f_start <= 0:
                    messagebox.showerror("Error", "La freqüència inicial ha de ser positiva")
                    return
                if f_end <= 0:
                    messagebox.showerror("Error", "La freqüència final ha de ser positiva")
                    return
                if dur <= 0:
                    messagebox.showerror("Error", "La durada ha de ser positiva")
                    return
                
                # Generar sweep
                if sweep_type == "Logarítmic":
                    signal = sweep_generator(fs, dur, f_start, f_end)
                else:
                    signal = sweep_generator_linear(fs, dur, f_start, f_end)
                
                # Demanar on guardar
                file_path = filedialog.asksaveasfilename(
                    defaultextension=".wav",
                    filetypes=[("WAV files", "*.wav"), ("All files", "*.*")],
                    initialfile=f"sweep_{f_start}Hz-{f_end}Hz_{dur}s.wav"
                )
                
                if file_path:
                    # Guardar el fitxer
                    wavfile.write(file_path, fs, (signal * 32767).astype(np.int16))
                    messagebox.showinfo("Èxit", f"Sweep guardat a:\n{file_path}")
                    self.signal = signal
                    self.fs = fs
                    dialog.destroy()
                    self.show_analysis_screen()
            except ValueError:
                messagebox.showerror("Error", "Els paràmetres introduïts no són vàlids")
        
        # Botó generar
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=5, column=0, columnspan=2, pady=20)
        
        generate_btn = ttk.Button(button_frame, text="Generar i guardar", 
                                 command=generate_and_save)
        generate_btn.pack(side=tk.LEFT, padx=5)
        
        cancel_btn = ttk.Button(button_frame, text="Cancelar", 
                               command=dialog.destroy)
        cancel_btn.pack(side=tk.LEFT, padx=5)
    
    
    def show_imd_screen(self):
        """Mostra la pantalla per seleccionar generar sweep o importar una senyal"""
        # Esborrar els widgets anteriors
        for widget in self.winfo_children():
            widget.destroy()
        
        # Frame principal
        main_frame = ttk.Frame(self, padding="20")
        main_frame.pack(expand=True, fill=tk.BOTH)
        
        # Botó per tornar enrere
        back_btn = ttk.Button(main_frame, text="← Tornar", 
                             command=self.show_selection_screen)
        back_btn.pack(anchor=tk.NW, pady=(0, 20))
        
        # Títol
        title_label = ttk.Label(main_frame, text="Selecciona la font del sweep", 
                               font=("Arial", 14, "bold"))
        title_label.pack(pady=20)
        
        # Botó generar sweep
        generate_btn = ttk.Button(main_frame, text="Generar sweep", 
                                 command=self.show_sweep_generation_dialog,
                                 width=25)
        generate_btn.pack(pady=10)
        
        # Botó importar WAV
        import_btn = ttk.Button(main_frame, text="Importar WAV", 
                               command=self.import_wav_signal,
                               width=25)
        import_btn.pack(pady=10)
    
    def show_analysis_screen(self):
        """Mostra la pantalla principal d'anàlisi"""
        # Esborrar els widgets anteriors
        for widget in self.winfo_children():
            widget.destroy()
        
        # Frame principal
        main_frame = ttk.Frame(self, padding="10")
        main_frame.pack(expand=True, fill=tk.BOTH)
        
        # Botó per tornar enrere
        back_btn = ttk.Button(main_frame, text="← Tornar", 
                             command=self.show_selection_screen)
        back_btn.pack(anchor=tk.NW, pady=(0, 10))
        
        # Títol - determinar quin tipus d'anàlisi
        if self.selected_analysis_type == "THD":
            analysis_title = f"Anàlisi: {self.selected_thd_type}"
        else:
            analysis_title = "Anàlisi: IMD"
        
        title_label = ttk.Label(main_frame, 
                               text=analysis_title, 
                               font=("Arial", 14, "bold"))
        title_label.pack(pady=10)

        # Botons d'entrada d'àudio (només per THD_SWEEP)
        if self.selected_analysis_type == "THD" and self.selected_thd_type == "THD_SWEEP":
            button_frame = ttk.Frame(main_frame)
            button_frame.pack(pady=10)
            
            self.record_btn = ttk.Button(button_frame, text="Enregistra àudio", 
                                        command=self.record_and_plot)
            self.record_btn.pack(side=tk.LEFT, padx=5)

            self.upload_btn = ttk.Button(button_frame, text="Carrega àudio (WAV)", 
                                        command=self.upload_and_analyze)
            self.upload_btn.pack(side=tk.LEFT, padx=5)

        # Etiqueta de resultat
        self.thd_label = ttk.Label(main_frame, text="Resultat: -")
        self.thd_label.pack(pady=10)

        # Gràfics
        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(8, 6))
        self.fig.tight_layout(pad=4.0)
        self.canvas = FigureCanvasTkAgg(self.fig, master=main_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Si tenim una senyal carregada o generada, analitzar-la
        if self.signal is not None and self.fs is not None:
            self.analyze_signal(self.signal, self.fs)
        elif self.selected_analysis_type == "THD" and self.selected_thd_type == "THD_SWEEP":
            # Per THD_SWEEP, mostrar gràfics en blanc fins que l'usuari cargui una senyal
            self.ax1.clear()
            self.ax2.clear()
            self.ax1.text(0.5, 0.5, "Carrega una senyal per analitzar", 
                         ha='center', va='center', transform=self.ax1.transAxes)
            self.ax1.set_title("Resposta en freqüència (FFT)")
            self.canvas.draw()
    
    def analyze_signal(self, signal, fs):
        """Analitza una senyal i mostra els gràfics"""
        self.ax1.clear()
        self.ax2.clear()
        
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
        
        # Calcula i mostra el resultat segons el tipus d'anàlisi
        if self.selected_analysis_type == "THD":
            thd = compute_THD_F(signal, fs)
            self.thd_label.config(text=f"THD ({self.selected_thd_type}): {thd:.2f}%")
        elif self.selected_analysis_type == "IMD":
            try:
                imd_smpte = compute_IMD_smpte(signal, fs)
                self.thd_label.config(text=f"IMD (SMPTE): {imd_smpte:.2f}%")
            except Exception as e:
                self.thd_label.config(text=f"Error al calcular IMD: {str(e)}")
    
    def record_and_plot(self):
        """Grava àudio i l'analitza"""
        try:
            self.signal, self.fs = record_audio(duration=2)
            self.analyze_signal(self.signal, self.fs)
        except Exception as e:
            messagebox.showerror("Error", f"Error al gravar: {e}")

    def upload_and_analyze(self):
        """Carrega un fitxer WAV i l'analitza"""
        file_path = filedialog.askopenfilename(filetypes=[("WAV files", "*.wav")])
        if not file_path:
            return
        try:
            self.fs, data = wavfile.read(file_path)
            # Si l'àudio és estèreo, agafa només un canal
            if len(data.shape) > 1:
                data = data[:, 0]
            # Normalitza si és enter
            if data.dtype != np.float32 and data.dtype != np.float64:
                data = data.astype(np.float32) / np.max(np.abs(data))
            self.signal = data.astype(np.float64)
            self.analyze_signal(self.signal, self.fs)
        except Exception as e:
            messagebox.showerror("Error", f"Error al carregar el fitxer: {e}")

if __name__ == "__main__":
    app = AudioApp()
    app.mainloop()
