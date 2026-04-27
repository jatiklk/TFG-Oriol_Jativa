import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
from tkinter import simpledialog
from tkinter import messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import numpy as np
import threading
import time
from scipy.io import wavfile
from audio_input import record_audio, get_input_devices
from analyzer_THD import plot_fft, extract_impulse_response, compute_THD_F, compute_THD_rms, compute_THDN, compute_THD_sweep
from analyzer_IMD import compute_IMD_smpte, compute_IMD_ccif
from generator import sine, sweep_generator, sweep_generator_linear, soft_clip_tanh, apply_nonlinear_distortion, synth_tone_with_thd, synth_tone_with_thd_and_noise

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
        
        # Botó Simulator
        simulator_btn = ttk.Button(main_frame, text="Simulator", 
                                   command=self.show_simulator_screen,
                                   width=20)
        simulator_btn.pack(pady=10)
    
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
        
        # Si és THD_SWEEP, mostrar opcions de generar o importar sweep
        # Si és THD_F, THD_N o THD_RMS, mostrar opcions de generar sinusoide o importar
        if self.selected_thd_type == "THD_SWEEP":
            self.show_sweep_input_screen()
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
        
        # Botó gravar àudio
        record_btn = ttk.Button(main_frame, text="Enregistra àudio", 
                                 command=self.show_record_audio_screen,
                                 width=25)
        record_btn.pack(pady=10)
        
        # Botó importar WAV
        import_btn = ttk.Button(main_frame, text="Importar WAV", 
                               command=self.import_wav_signal,
                               width=25)
        import_btn.pack(pady=10)
    
    def show_record_audio_screen(self):
        """Mostra una pantalla per triar l'entrada del sistema i gravar àudio"""
        for widget in self.winfo_children():
            widget.destroy()

        self.recording_active = False
        self.record_start_time = None

        main_frame = ttk.Frame(self, padding="20")
        main_frame.pack(expand=True, fill=tk.BOTH)

        back_btn = ttk.Button(main_frame, text="← Tornar", 
                             command=self.show_signal_input_screen)
        back_btn.pack(anchor=tk.NW, pady=(0, 20))

        title_label = ttk.Label(main_frame, text="Enregistra àudio", 
                               font=("Arial", 14, "bold"))
        title_label.pack(pady=10)

        device_frame = ttk.Frame(main_frame)
        device_frame.pack(fill=tk.X, pady=5)

        device_label = ttk.Label(device_frame, text="Entrada del sistema:")
        device_label.pack(side=tk.LEFT, padx=(0, 10))

        input_devices = get_input_devices()
        device_names = [f"{idx}: {device['name']}" for idx, device in input_devices]
        self.device_dropdown = ttk.Combobox(device_frame, values=device_names, state="readonly", width=50)
        if device_names:
            self.device_dropdown.set(device_names[0])
        self.device_dropdown.pack(side=tk.LEFT, fill=tk.X, expand=True)

        duration_frame = ttk.Frame(main_frame)
        duration_frame.pack(fill=tk.X, pady=5)

        dur_label = ttk.Label(duration_frame, text="Durada (0-5 s):")
        dur_label.pack(side=tk.LEFT, padx=(0, 10))
        self.duration_entry = ttk.Entry(duration_frame, width=10)
        self.duration_entry.insert(0, "2")
        self.duration_entry.pack(side=tk.LEFT)

        fs_frame = ttk.Frame(main_frame)
        fs_frame.pack(fill=tk.X, pady=5)

        fs_label = ttk.Label(fs_frame, text="Freq. mostratge:")
        fs_label.pack(side=tk.LEFT, padx=(0, 10))
        fs_options = ["44100", "48000"]
        self.fs_dropdown = ttk.Combobox(fs_frame, values=fs_options, state="readonly", width=18)
        self.fs_dropdown.set("48000")
        self.fs_dropdown.pack(side=tk.LEFT)

        status_frame = ttk.Frame(main_frame)
        status_frame.pack(fill=tk.X, pady=15)

        status_label = ttk.Label(status_frame, text="Estat:")
        status_label.pack(side=tk.LEFT, padx=(0, 10))

        self.record_indicator = tk.Canvas(status_frame, width=18, height=18, highlightthickness=0)
        self.record_indicator.pack(side=tk.LEFT)
        self.record_indicator_circle = self.record_indicator.create_oval(2, 2, 16, 16, fill="grey")

        self.timer_label = ttk.Label(status_frame, text="Temps: 0.0 s")
        self.timer_label.pack(side=tk.LEFT, padx=(15, 0))

        button_frame = ttk.Frame(main_frame)
        button_frame.pack(pady=15)

        self.start_record_button = ttk.Button(button_frame, text="Inicia gravació", 
                                             command=lambda: self.start_recording(input_devices))
        self.start_record_button.pack(side=tk.LEFT, padx=5)

        self.cancel_record_button = ttk.Button(button_frame, text="Cancelar", 
                                              command=self.show_signal_input_screen)
        self.cancel_record_button.pack(side=tk.LEFT, padx=5)

    def start_recording(self, input_devices):
        if self.recording_active:
            return

        try:
            dur = float(self.duration_entry.get())
            if dur <= 0 or dur > 5:
                messagebox.showerror("Error", "La durada ha de ser entre 0 i 5 segons")
                return
            fs = int(self.fs_dropdown.get())
        except ValueError:
            messagebox.showerror("Error", "Els paràmetres han de ser numèrics")
            return

        if not input_devices:
            messagebox.showerror("Error", "No s'ha trobat cap dispositiu d'entrada")
            return

        selected_device = self.device_dropdown.get()
        if not selected_device:
            messagebox.showerror("Error", "Selecciona un dispositiu d'entrada")
            return

        device_index = int(selected_device.split(":", 1)[0])
        self.recording_active = True
        self.record_start_time = time.monotonic()
        self.record_indicator.itemconfig(self.record_indicator_circle, fill="red")
        self.start_record_button.config(state=tk.DISABLED)
        self.timer_label.config(text="Temps: 0.0 s")
        self.update_recording_timer()

        record_thread = threading.Thread(target=self._record_audio_thread, 
                                         args=(dur, fs, device_index), daemon=True)
        record_thread.start()

    def update_recording_timer(self):
        if self.recording_active:
            elapsed = time.monotonic() - self.record_start_time
            self.timer_label.config(text=f"Temps: {elapsed:.1f} s")
            self.after(100, self.update_recording_timer)

    def _record_audio_thread(self, duration, fs, device):
        try:
            self.signal, self.fs = record_audio(duration=duration, fs=fs, device=device)
            self.after(0, lambda: self._finish_recording(success=True))
        except Exception as e:
            self.after(0, lambda: self._finish_recording(success=False, error=e))

    def _finish_recording(self, success, error=None):
        self.recording_active = False
        self.record_indicator.itemconfig(self.record_indicator_circle, fill="grey")
        self.start_record_button.config(state=tk.NORMAL)
        if success:
            messagebox.showinfo("Èxit", "Gravació completa")
            self.show_analysis_screen()
        else:
            messagebox.showerror("Error", f"Error al gravar: {error}")

    def show_sweep_input_screen(self):
        """Mostra la pantalla per seleccionar generar o importar una sweep"""
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

    def import_wav_signal(self):
        """Importa un fitxer WAV"""
        file_path = filedialog.askopenfilename(filetypes=[("WAV files", "*.wav")])
        if not file_path:
            return
        try:
            fs, data = wavfile.read(file_path)
            if len(data.shape) > 1:
                data = data[:, 0]
            if data.dtype != np.float32 and data.dtype != np.float64:
                data = data.astype(np.float32) / np.max(np.abs(data))
            self.signal = data.astype(np.float64)
            self.fs = fs
            self.show_analysis_screen()
        except Exception as e:
            messagebox.showerror("Error", f"Error al carregar el fitxer: {e}")

    def import_wav_for_simulation(self):
        """Importa WAV i permet triar distorsió"""
        file_path = filedialog.askopenfilename(filetypes=[("WAV files", "*.wav")])
        if not file_path:
            return
        try:
            fs, data = wavfile.read(file_path)
            if len(data.shape) > 1:
                data = data[:, 0]
            if data.dtype != np.float32 and data.dtype != np.float64:
                data = data.astype(np.float32) / np.max(np.abs(data))
            self.signal = data.astype(np.float64)
            self.fs = fs
            self.show_distortion_selection_screen()
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

    def show_distortion_selection_screen(self):
        """Permet seleccionar distorsió per al WAV importat"""
        for widget in self.winfo_children():
            widget.destroy()

        main_frame = ttk.Frame(self, padding="20")
        main_frame.pack(expand=True, fill=tk.BOTH)

        back_btn = ttk.Button(main_frame, text="← Tornar", 
                              command=self.show_simulator_screen)
        back_btn.pack(anchor=tk.NW, pady=(0, 20))

        title_label = ttk.Label(main_frame, text="Selecciona el tipus de distorsió", 
                               font=("Arial", 14, "bold"))
        title_label.pack(pady=20)

        methods = ["soft_clip_tanh", "apply_nonlinear_distortion"]
        self.distortion_dropdown = ttk.Combobox(main_frame, values=methods, state="readonly", width=30)
        self.distortion_dropdown.set(methods[0])
        self.distortion_dropdown.pack(pady=10)

        params_label = ttk.Label(main_frame, text="Paràmetres (dist, alpha):")
        params_label.pack(pady=6)

        self.param_entry = ttk.Entry(main_frame, width=20)
        self.param_entry.insert(0, "0.1")
        self.param_entry.pack(pady=5)

        apply_btn = ttk.Button(main_frame, text="Aplicar distorsió", 
                               command=self.apply_selected_distortion)
        apply_btn.pack(pady=10)

    def apply_selected_distortion(self):
        method = self.distortion_dropdown.get()
        try:
            p = float(self.param_entry.get())
        except ValueError:
            messagebox.showerror("Error", "El paràmetre ha de ser numèric")
            return

        if method == "soft_clip_tanh":
            self.signal = soft_clip_tanh(self.signal, dist=p)
        else:
            self.signal = apply_nonlinear_distortion(self.signal, alpha=p)

        self.show_analysis_screen()

    def show_generator_distortion_dialog(self):
        """Mostra un diàleg per generar una senyal distorsionada directament"""
        for widget in self.winfo_children():
            widget.destroy()

        main_frame = ttk.Frame(self, padding="20")
        main_frame.pack(expand=True, fill=tk.BOTH)

        back_btn = ttk.Button(main_frame, text="← Tornar", 
                              command=self.show_simulator_screen)
        back_btn.pack(anchor=tk.NW, pady=(0, 20))

        title_label = ttk.Label(main_frame, text="Generador distorsionat", 
                               font=("Arial", 14, "bold"))
        title_label.pack(pady=20)

        methods = ["sine", "synth_tone_with_thd", "synth_tone_with_thd_and_noise", "sweep_generator", "sweep_generator_linear"]
        self.generator_dropdown = ttk.Combobox(main_frame, values=methods, state="readonly", width=30)
        self.generator_dropdown.set(methods[0])
        self.generator_dropdown.pack(pady=10)

        fs_label = ttk.Label(main_frame, text="Freq. mostratge (Hz):")
        fs_label.pack(pady=3)
        fs_options = ["44100", "48000", "96000", "192000"]
        self.generator_fs_dropdown = ttk.Combobox(main_frame, values=fs_options, state="readonly", width=17)
        self.generator_fs_dropdown.set("48000")
        self.generator_fs_dropdown.pack(pady=3)

        dur_label = ttk.Label(main_frame, text="Durada (segons):")
        dur_label.pack(pady=3)
        self.generator_dur_entry = ttk.Entry(main_frame, width=20)
        self.generator_dur_entry.insert(0, "2")
        self.generator_dur_entry.pack(pady=3)

        f0_label = ttk.Label(main_frame, text="Freq. fonamental (Hz):")
        f0_label.pack(pady=3)
        self.generator_f0_entry = ttk.Entry(main_frame, width=20)
        self.generator_f0_entry.insert(0, "1000")
        self.generator_f0_entry.pack(pady=3)

        amp_label = ttk.Label(main_frame, text="Amplitud (0-1):")
        amp_label.pack(pady=3)
        self.generator_amp_entry = ttk.Entry(main_frame, width=20)
        self.generator_amp_entry.insert(0, "0.9")
        self.generator_amp_entry.pack(pady=3)

        thd_label = ttk.Label(main_frame, text="THD target (0-1):")
        thd_label.pack(pady=3)
        self.generator_thd_entry = ttk.Entry(main_frame, width=20)
        self.generator_thd_entry.insert(0, "0.05")
        self.generator_thd_entry.pack(pady=3)

        noise_label = ttk.Label(main_frame, text="Noise level (synth_tone_with_thd_and_noise):")
        noise_label.pack(pady=3)
        self.generator_noise_entry = ttk.Entry(main_frame, width=20)
        self.generator_noise_entry.insert(0, "0.01")
        self.generator_noise_entry.pack(pady=3)

        dist_label = ttk.Label(main_frame, text="Distorsió (soft_clip/apply_nonlin):")
        dist_label.pack(pady=3)
        self.generator_dist_entry = ttk.Entry(main_frame, width=20)
        self.generator_dist_entry.insert(0, "0.1")
        self.generator_dist_entry.pack(pady=3)

        apply_btn = ttk.Button(main_frame, text="Generar i analitzar", 
                               command=self.generate_distorted_signal)
        apply_btn.pack(pady=10)

    def generate_distorted_signal(self):
        method = self.generator_dropdown.get()
        try:
            fs = int(self.generator_fs_dropdown.get())
            dur = float(self.generator_dur_entry.get())
            f0 = float(self.generator_f0_entry.get())
            amp = float(self.generator_amp_entry.get())
            thd_target = float(self.generator_thd_entry.get())
            noise_level = float(self.generator_noise_entry.get())
            dist = float(self.generator_dist_entry.get())
        except ValueError:
            messagebox.showerror("Error", "Els paràmetres han de ser numèrics")
            return

        if method == "sine":
            self.signal = sine(fs, dur, f0, amp)
        elif method == "synth_tone_with_thd":
            self.signal = synth_tone_with_thd(fs, dur, f0, thd_target)
        elif method == "synth_tone_with_thd_and_noise":
            self.signal = synth_tone_with_thd_and_noise(fs, dur, f0, thd_target, noise_level=noise_level)
        elif method == "sweep_generator":
            self.signal = sweep_generator(fs, dur, f0, float(self.generator_f0_entry.get() or 20000))
        else:
            self.signal = sweep_generator_linear(fs, dur, f0, float(self.generator_f0_entry.get() or 20000))

        self.fs = fs
        self.show_analysis_screen()

    def show_imd_screen(self):
        """Mostra la pantalla per seleccionar generar sweep o importar una senyal"""
        # Esborrar widgets existents
        for widget in self.winfo_children():
            widget.destroy()

        main_frame = ttk.Frame(self, padding="20")
        main_frame.pack(expand=True, fill=tk.BOTH)

        back_btn = ttk.Button(main_frame, text="← Tornar", 
                              command=self.show_selection_screen)
        back_btn.pack(anchor=tk.NW, pady=(0, 20))

        title_label = ttk.Label(main_frame, text="Selecciona la font del sweep", 
                               font=("Arial", 14, "bold"))
        title_label.pack(pady=20)

        generate_btn = ttk.Button(main_frame, text="Generar sweep", 
                                 command=self.show_sweep_generation_dialog,
                                 width=25)
        generate_btn.pack(pady=10)

        import_btn = ttk.Button(main_frame, text="Importar WAV", 
                               command=self.import_wav_signal,
                               width=25)
        import_btn.pack(pady=10)

    def show_simulator_screen(self):
        """Mostra opcions de simular distorsió"""
        for widget in self.winfo_children():
            widget.destroy()

        main_frame = ttk.Frame(self, padding="20")
        main_frame.pack(expand=True, fill=tk.BOTH)

        back_btn = ttk.Button(main_frame, text="← Tornar", 
                              command=self.show_selection_screen)
        back_btn.pack(anchor=tk.NW, pady=(0, 20))

        title_label = ttk.Label(main_frame, text="Simulator de distorsió", 
                               font=("Arial", 14, "bold"))
        title_label.pack(pady=20)

        import_btn = ttk.Button(main_frame, text="Pujar WAV i aplicar distorsió", 
                                command=self.import_wav_for_simulation,
                                width=30)
        import_btn.pack(pady=10)

        generator_btn = ttk.Button(main_frame, text="Generar senyal distorsionada", 
                                   command=self.show_generator_distortion_dialog,
                                   width=30)
        generator_btn.pack(pady=10)

    
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

        # Títol i resultat en una fila superior
        top_frame = ttk.Frame(main_frame)
        top_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Títol a l'esquerra
        title_label = ttk.Label(top_frame, 
                               text=analysis_title, 
                               font=("Arial", 14, "bold"))
        title_label.pack(side=tk.LEFT)
        
        # Etiqueta de resultat a la dreta, més gran
        self.thd_label = ttk.Label(top_frame, text="Resultat: -", 
                                  font=("Arial", 18, "bold"), foreground="#1a237e")
        self.thd_label.pack(side=tk.RIGHT)

        # Gràfics: només un eix per FFT / THD
        self.fig, self.ax1 = plt.subplots(figsize=(10, 6))
        self.fig.tight_layout(pad=4.0)
        self.canvas = FigureCanvasTkAgg(self.fig, master=main_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Si tenim una senyal carregada o generada, analitzar-la
        if self.signal is not None and self.fs is not None:
            self.analyze_signal(self.signal, self.fs)
        elif self.selected_analysis_type == "THD" and self.selected_thd_type == "THD_SWEEP":
            # Per THD_SWEEP, mostrar gràfics en blanc fins que l'usuari carregui una senyal
            self.ax1.clear()
            self.ax1.text(0.5, 0.5, "Carrega una senyal per analitzar", 
                         ha='center', va='center', transform=self.ax1.transAxes)
            self.ax1.set_title("Resposta en freqüència (FFT)")
            self.canvas.draw()
    
    def analyze_signal(self, signal, fs):
        """Analitza una senyal i mostra els gràfics"""
        self.ax1.clear()
        
        # FFT plot
        N = len(signal)
        spectrum = abs(np.fft.fft(signal)[:N//2])
        freqs = np.fft.fftfreq(N, 1/fs)[:N//2]
        self.ax1.plot(freqs, spectrum, color="#1e88e5", linewidth=1.5)
        self.ax1.set_title("Resposta en freqüència (FFT)", fontsize=16, fontweight="bold")
        self.ax1.set_xlabel("Freqüència (Hz)", fontsize=12)
        self.ax1.set_ylabel("Amplitud", fontsize=12)
        self.ax1.grid(True, linestyle="--", alpha=0.4)
        self.ax1.tick_params(axis='both', labelsize=10)

        self.canvas.draw()
        
        # Calcula i mostra el resultat segons el tipus d'anàlisi
        if self.selected_analysis_type == "THD":
            try:
                if self.selected_thd_type == "THD_F":
                    thd = compute_THD_F(signal, fs, len(signal))
                    self.thd_label.config(text=f"THD_F: {thd:.2f}%")
                elif self.selected_thd_type == "THD_RMS":
                    thd = compute_THD_rms(signal, fs, len(signal))
                    self.thd_label.config(text=f"THD_RMS: {thd:.2f}%")
                elif self.selected_thd_type == "THD_N":
                    thd = compute_THDN(signal, fs, len(signal))
                    self.thd_label.config(text=f"THD+N: {thd:.2f}%")
                elif self.selected_thd_type == "THD_SWEEP":
                    thd_values, freqs = compute_THD_sweep(signal, num_partitions=10)
                    self.thd_label.config(text=f"THD_SWEEP (mitjana): {np.mean(thd_values):.2f}%")
                    self.ax1.clear()
                    self.ax1.plot(freqs, thd_values)
                    self.ax1.set_title("THD per segment")
                    self.ax1.set_xlabel("Freqüència (Hz)")
                    self.ax1.set_ylabel("THD (%)")
                    self.ax1.set_xscale('log')
                    self.ax1.grid(True)
                    self.canvas.draw()
                    return
                else:
                    self.thd_label.config(text="THD: Tipus desconegut")
            except Exception as e:
                self.thd_label.config(text=f"Error al calcular THD: {e}")
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
