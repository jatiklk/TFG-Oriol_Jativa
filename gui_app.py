import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
from tkinter import simpledialog
from tkinter import messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import numpy as np
import threading
import time
import os
from scipy.io import wavfile
import sounddevice as sd
from audio_input import record_audio, get_input_devices
from analyzer_THD import plot_fft, extract_impulse_response, compute_THD_F, compute_THD_rms, compute_THDN, compute_THD_sweep, compute_THD_harmonic
from analyzer_IMD import compute_IMD_smpte, compute_IMD_ccif
from generator import sine, sweep_generator, sweep_generator_linear, soft_clip_tanh, apply_nonlinear_distortion, synth_tone_with_thd, synth_tone_with_thd_and_noise, synth_two_tones, sweep_log
from custom_farina import Farina

class AudioApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("DistorLab")
        self.geometry("900x700")
        self.selected_analysis_type = None
        self.selected_thd_type = None
        self.selected_imd_method = "SMPTE"
        self.signal = None
        self.fs = None
        self.is_recording_source = False
        self.distortion_applied = False
        self.thd_diagram_image = self.load_diagram_image("thd_diagram.png", max_width=320)
        self.imd_diagram_image = self.load_diagram_image("IMD_diagram.PNG", max_width=480)
        self.protocol("WM_DELETE_WINDOW", self.on_close)
        
        # Mostrar la pantalla inicial de selecció
        self.show_selection_screen()
    
    def load_thd_diagram_image(self, max_width=640):
        """Carrega la imatge THD amb l'amplada màxima indicada."""
        return self.load_diagram_image("thd_diagram.png", max_width)
    
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
    
    def load_diagram_image(self, file_name, max_width=640):
        """Carrega una imatge de diagrama si existeix a la carpeta de l'aplicació.

        Si l'amplada de la imatge és més gran que `max_width`, es torna una versió
        reduïda utilitzant `subsample` per evitar dependre de PIL.
        """
        filename = os.path.join(os.path.dirname(__file__), file_name)
        if os.path.exists(filename):
            try:
                img = tk.PhotoImage(file=filename)
            except Exception:
                return None

            try:
                width = img.width()
                if width > max_width and max_width > 0:
                    factor = max(1, int(round(width / float(max_width))))
                    return img.subsample(factor, factor)
                return img
            except Exception:
                return img
        return None

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

        # Mostrar diagrama THD una mica més gran (un pas abans de gravar/importar)
        thd_small = self.thd_diagram_image
        if thd_small is not None:
            image_small_frame = ttk.Frame(main_frame)
            image_small_frame.pack(fill=tk.BOTH, expand=False, pady=(0, 10))
            img_label_small = ttk.Label(image_small_frame, image=thd_small)
            img_label_small.image = thd_small
            img_label_small.pack()

        thd_options = ["THD_F", "THD_RMS", "THD_N", "THD_SWEEP", "Farina"]

        # Frame per al desplegable
        dropdown_frame = ttk.Frame(main_frame)
        dropdown_frame.pack(pady=20)
        
        # Etiqueta del desplegable
        label = ttk.Label(dropdown_frame, text="Tipus de THD:")
        label.pack(side=tk.LEFT, padx=5)
        
        # Desplegable amb les opcions
        self.thd_dropdown = ttk.Combobox(dropdown_frame, values=thd_options, 
                                         state="readonly", width=20)
        self.thd_dropdown.set(thd_options[0])  # Valor per defecte
        self.thd_dropdown.pack(side=tk.LEFT, padx=5)
        
        # Botó per continuar
        continue_btn = ttk.Button(main_frame, text="Continuar", 
                                 command=self.proceed_with_thd_analysis,
                                 width=20)
        continue_btn.pack(pady=20)
        
        # Botó generar senyal prova
        generate_btn = ttk.Button(main_frame, text="Generar senyal prova", 
                                 command=self.show_thd_test_signal_dialog,
                                 width=20)
        generate_btn.pack(pady=10)
    
    def proceed_with_thd_analysis(self):
        """Continua amb l'anàlisi de THD seleccionat"""
        self.selected_thd_type = self.thd_dropdown.get()
        
        # Si és THD_SWEEP, mostrar opcions de generar o importar sweep
        # Si és THD_F, THD_N o THD_RMS, mostrar opcions de generar sinusoide o importar
        if self.selected_thd_type == "THD_SWEEP":
            self.show_sweep_input_screen()
        elif self.selected_thd_type == "Farina":
            self.show_farina_screen()
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
        
        
    
    def show_record_audio_screen(self, from_farina=False):
        """Mostra una pantalla per triar l'entrada del sistema i gravar àudio"""
        for widget in self.winfo_children():
            widget.destroy()

        self.recording_active = False
        self.record_start_time = None

        main_frame = ttk.Frame(self, padding="20")
        main_frame.pack(expand=True, fill=tk.BOTH)

        back_command = self.show_farina_screen if from_farina else self.show_signal_input_screen
        back_btn = ttk.Button(main_frame, text="← Tornar", 
                             command=back_command)
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
                                              command=back_command)
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
            self.is_recording_source = True
            messagebox.showinfo("Èxit", "Gravació completa")
            if self.selected_analysis_type == "THD" and self.selected_thd_type == "Farina":
                try:
                    dur = getattr(self, 'farina_record_dur', 5.0)
                    A = getattr(self, 'farina_record_A', 0.9)
                    fs = getattr(self, 'farina_record_fs', self.fs)
                    far = Farina(A=A, duration=dur, fs=fs)
                    far.process_measurement(self.signal, log=False)
                    self.current_farina = far
                except Exception as e:
                    messagebox.showerror("Error", f"Error processant la gravació amb Farina: {e}")
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

    def show_farina_screen(self):
        """Mostra la pantalla per generar/descarregar la sweep usada pel mètode Farina i processar-la"""
        for widget in self.winfo_children():
            widget.destroy()

        main_frame = ttk.Frame(self, padding="20")
        main_frame.pack(expand=True, fill=tk.BOTH)

        back_btn = ttk.Button(main_frame, text="← Tornar", 
                             command=self.show_thd_selection_screen)
        back_btn.pack(anchor=tk.NW, pady=(0, 20))

        title_label = ttk.Label(main_frame, text="Farina - Sweep Logarítmic", 
                               font=("Arial", 14, "bold"))
        title_label.pack(pady=10)

        info_label = ttk.Label(main_frame, text="S'utilitzen paràmetres fixos per Farina: durada=5s, amplitud=0.9, fs=48000",
                               font=("Arial", 10), foreground="#333333", wraplength=760)
        info_label.pack(pady=(0, 15))

        btn_frame = ttk.Frame(main_frame)
        btn_frame.pack(pady=15)

        gen_btn = ttk.Button(btn_frame, text="Descarregar senyal de test", 
                      command=self._download_farina_probe_default)
        gen_btn.pack(side=tk.LEFT, padx=5)

        record_btn = ttk.Button(btn_frame, text="Enregistra WAV", 
                       command=self._prepare_farina_record)
        record_btn.pack(side=tk.LEFT, padx=5)

        import_btn = ttk.Button(btn_frame, text="Importar WAV", 
                     command=self._import_and_analyze_farina)
        import_btn.pack(side=tk.LEFT, padx=5)

    def _generate_and_process_farina(self):
        # deprecated: use dedicated generate method
        self._generate_farina_probe()

    def _generate_farina_probe(self):
        dur = 5.0
        A = 0.9
        fs = 48000

        try:
            far = Farina(A=A, duration=dur, fs=fs)
            # Process the probe against itself to produce the far_response/IRs
            far.process_measurement(far.probe, log=False)
            # Store for later analysis
            self.current_farina = far
            # Set the probe as current signal so user can inspect FFT
            self.signal = far.probe.astype(np.float64)
            self.fs = fs
            self.selected_analysis_type = "THD"
            self.selected_thd_type = "Farina"
            self.is_recording_source = False
            # Show analysis screen which will call analyze_signal
            self.show_analysis_screen()
        except Exception as e:
            messagebox.showerror("Error", f"No s'ha pogut generar la sweep Farina: {e}")

    def _prepare_farina_record(self):
        self.farina_record_dur = 5.0
        self.farina_record_A = 0.9
        self.farina_record_fs = 48000
        self.show_record_audio_screen(from_farina=True)

    def _download_farina_probe_default(self):
        """Genera i descarrega la probe Farina amb paràmetres per defecte, sense analitzar."""
        # Paràmetres per defecte (sense opció a l'usuari)
        dur = 5.0
        A = 0.9
        fs = 48000
        try:
            far = Farina(A=A, duration=dur, fs=fs)
            sweep = far.probe
        except Exception as e:
            messagebox.showerror("Error", f"No s'ha pogut generar la sweep Farina: {e}")
            return

        path = filedialog.asksaveasfilename(defaultextension='.wav', filetypes=[('WAV files', '*.wav')], initialfile='farina_probe.wav')
        if not path:
            return

        try:
            wavfile.write(path, fs, sweep.astype(np.float32))
            messagebox.showinfo("Desat", f"Sweep Farina (test) desada a: {path}")
        except Exception as e:
            messagebox.showerror("Error", f"No s'ha pogut desar el fitxer: {e}")

    def _save_farina_sweep(self):
        dur = 5.0
        A = 0.9
        fs = 48000
        # Crear un objecte Farina amb els paràmetres i desar la seva 'probe' exacta
        try:
            far = Farina(A=A, duration=dur, fs=fs)
            sweep = far.probe
        except Exception as e:
            messagebox.showerror("Error", f"No s'ha pogut generar la sweep Farina: {e}")
            return

        path = filedialog.asksaveasfilename(defaultextension='.wav', filetypes=[('WAV files', '*.wav')])
        if not path:
            return

        try:
            wavfile.write(path, fs, sweep.astype(np.float32))
            messagebox.showinfo("Desat", f"Sweep Farina desada a: {path}")
        except Exception as e:
            messagebox.showerror("Error", f"No s'ha pogut desar el fitxer: {e}")

    def _import_and_analyze_farina(self):
        # Ask user for WAV file and analyze with Farina processing
        file_path = filedialog.askopenfilename(filetypes=[("WAV files", "*.wav")])
        if not file_path:
            return
        try:
            fs, data = wavfile.read(file_path)
            if len(data.shape) > 1:
                data = data[:, 0]
            # normalize to float
            if data.dtype != np.float32 and data.dtype != np.float64:
                data = data.astype(np.float32) / np.max(np.abs(data))
            measurement = data.astype(np.float64)
        except Exception as e:
            messagebox.showerror("Error", f"Error al carregar el fitxer: {e}")
            return

        dur = 5.0
        A = 0.9
        try:
            # Create Farina object with same params and process measurement
            far = Farina(A=A, duration=dur, fs=fs)
            far.process_measurement(measurement, log=False)
            self.current_farina = far
            self.signal = measurement
            self.fs = fs
            self.selected_analysis_type = "THD"
            self.selected_thd_type = "Farina"
            self.is_recording_source = False
            self.show_analysis_screen()
        except Exception as e:
            messagebox.showerror("Error", f"Error processant amb Farina: {e}")

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
            self.is_recording_source = False
            self.show_analysis_screen()
            if self.selected_analysis_type == "IMD":
                self.analyze_signal(self.signal, self.fs)
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
    
    def show_distortion_selection_screen(self):
        """Mostra opcions de distorsió per aplicar al WAV carregat"""
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

        # Selecció del mètode de distorsió
        method_label = ttk.Label(main_frame, text="Mètode de distorsió:")
        method_label.pack(pady=5)
        
        methods = ["soft_clip_tanh", "apply_nonlinear_distortion"]
        self.distortion_dropdown = ttk.Combobox(main_frame, values=methods, state="readonly", width=30)
        self.distortion_dropdown.set(methods[0])
        self.distortion_dropdown.pack(pady=5)

        # Paràmetre de distorsió
        param_label = ttk.Label(main_frame, text="Paràmetre de distorsió (0.0 - 1.0):")
        param_label.pack(pady=5)
        
        self.param_entry = ttk.Entry(main_frame, width=20)
        self.param_entry.insert(0, "0.5")
        self.param_entry.pack(pady=5)

        # Botó per aplicar distorsió
        apply_btn = ttk.Button(main_frame, text="Aplicar distorsió", 
                               command=self.apply_selected_distortion,
                               width=30)
        apply_btn.pack(pady=20)
    
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
                    if self.selected_analysis_type == "IMD":
                        self.show_imd_screen()
                    else:
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
    
    def show_thd_test_signal_dialog(self):
        """Mostra un diàleg per generar una senyal de prova per THD (sine o amb THD)."""
        dialog = tk.Toplevel(self)
        dialog.title("Generar senyal prova")
        dialog.geometry("380x320")
        dialog.transient(self)
        dialog.grab_set()

        # Centrar la finestra respecte a la principal
        dialog.update_idletasks()
        x = self.winfo_x() + (self.winfo_width() // 2) - (dialog.winfo_width() // 2)
        y = self.winfo_y() + (self.winfo_height() // 2) - (dialog.winfo_height() // 2)
        dialog.geometry(f"+{x}+{y}")

        main_frame = ttk.Frame(dialog, padding="12")
        main_frame.pack(expand=True, fill=tk.BOTH)

        method_label = ttk.Label(main_frame, text="Tipus de senyal:")
        method_label.grid(row=0, column=0, sticky=tk.W, pady=6)
        methods = ["Sine pura", "Sweep logarítmic", "Sweep lineal"]
        method_dropdown = ttk.Combobox(main_frame, values=methods, state="readonly", width=22)
        method_dropdown.set(methods[0])
        method_dropdown.grid(row=0, column=1, pady=6)

        f0_label = ttk.Label(main_frame, text="Freq. fonamental (Hz):")
        f0_label.grid(row=1, column=0, sticky=tk.W, pady=6)
        f0_entry = ttk.Entry(main_frame, width=20)
        f0_entry.insert(0, "1000")
        f0_entry.grid(row=1, column=1, pady=6)

        dur_label = ttk.Label(main_frame, text="Durada (s):")
        dur_label.grid(row=2, column=0, sticky=tk.W, pady=6)
        dur_entry = ttk.Entry(main_frame, width=20)
        dur_entry.insert(0, "2")
        dur_entry.grid(row=2, column=1, pady=6)

        fs_label = ttk.Label(main_frame, text="Freq. mostratge (Hz):")
        fs_label.grid(row=3, column=0, sticky=tk.W, pady=6)
        fs_dropdown = ttk.Combobox(main_frame, values=["44100","48000","96000"], state="readonly", width=17)
        fs_dropdown.set("48000")
        fs_dropdown.grid(row=3, column=1, pady=6)

        amp_label = ttk.Label(main_frame, text="Amplitud (0-1):")
        amp_label.grid(row=4, column=0, sticky=tk.W, pady=6)
        amp_entry = ttk.Entry(main_frame, width=20)
        amp_entry.insert(0, "0.9")
        amp_entry.grid(row=4, column=1, pady=6)

        # No es sol·liciten paràmetres de distorsió — només senyals netes
        # (si s'escull sweep, s'utilitzen f0 com a freq. inicial i f_end a l'entrada de f0)
        thd_entry = None
        noise_entry = None

        def generate_and_set():
            try:
                sel = method_dropdown.get()
                f0 = float(f0_entry.get())
                dur = float(dur_entry.get())
                fs = int(fs_dropdown.get())
                amp = float(amp_entry.get())
            except ValueError:
                messagebox.showerror("Error", "Els paràmetres han de ser numèrics")
                return

            if sel == "Sine pura":
                sig = sine(fs, dur, f0, amp)
            elif sel == "Sweep logarítmic":
                # use f0 as start, ask user to reuse f0 field for end frequency via simpledialog
                f_end = simpledialog.askfloat("Freq. final", "Introdueix la freqüència final (Hz):", initialvalue=20000, parent=dialog)
                if f_end is None:
                    return
                sig = sweep_generator(fs=fs, dur=dur, f_start=f0, f_end=f_end)
            elif sel == "Sweep lineal":
                f_end = simpledialog.askfloat("Freq. final", "Introdueix la freqüència final (Hz):", initialvalue=20000, parent=dialog)
                if f_end is None:
                    return
                sig = sweep_generator_linear(fs=fs, dur=dur, f_start=f0, f_end=f_end)
            else:
                messagebox.showerror("Error", "Tipus de senyal desconegut")
                return

            # Assignar la senyal i mostrar la pantalla per reproduir/descarregar (sense analitzar)
            self.signal = sig.astype(np.float64)
            self.fs = fs
            self.is_recording_source = False
            dialog.destroy()
            self.show_test_signal_ready_screen()

        btn_frame = ttk.Frame(main_frame)
        btn_frame.grid(row=7, column=0, columnspan=2, pady=12)
        gen_btn = ttk.Button(btn_frame, text="Generar", command=generate_and_set)
        gen_btn.pack(side=tk.LEFT, padx=6)
        cancel_btn = ttk.Button(btn_frame, text="Cancelar", command=dialog.destroy)
        cancel_btn.pack(side=tk.LEFT, padx=6)
    
    
    def show_imd_screen(self):
        """Mostra la pantalla per seleccionar generar sweep o importar una senyal"""
        for widget in self.winfo_children():
            widget.destroy()

        main_frame = ttk.Frame(self, padding="20")
        main_frame.pack(expand=True, fill=tk.BOTH)

        back_btn = ttk.Button(main_frame, text="← Tornar", 
                             command=self.show_selection_screen)
        back_btn.pack(anchor=tk.NW, pady=(0, 20))

        title_label = ttk.Label(main_frame, text="Anàlisi IMD: genera senyal o importa WAV", 
                               font=("Arial", 14, "bold"))
        title_label.pack(pady=20)

        method_frame = ttk.Frame(main_frame)
        method_frame.pack(pady=(0, 20), fill=tk.X)

        method_label = ttk.Label(method_frame, text="Mètode IMD:", font=("Arial", 12))
        method_label.pack(side=tk.LEFT, padx=(0, 10))

        imd_methods = ["SMPTE", "CCIF"]
        self.imd_dropdown = ttk.Combobox(method_frame, values=imd_methods, state="readonly", width=20)
        self.imd_dropdown.set(self.selected_imd_method)
        self.imd_dropdown.bind("<<ComboboxSelected>>", self.on_imd_method_changed)
        self.imd_dropdown.pack(side=tk.LEFT)

        # Mostrar diagrama IMD una mica més avall, després de triar el tipus d'anàlisi
        imd_small = self.imd_diagram_image if self.imd_diagram_image is not None else None
        if imd_small is not None:
            image_small_frame = ttk.Frame(main_frame)
            image_small_frame.pack(fill=tk.BOTH, expand=False, pady=(10, 10))
            img_label_small = ttk.Label(image_small_frame, image=imd_small)
            img_label_small.image = imd_small
            img_label_small.pack()

        info_label = ttk.Label(main_frame, 
                               text="Prem un botó per generar la senyal amb el mètode triat, o importa un WAV.",
                               font=("Arial", 11), foreground="#333333", wraplength=760)
        info_label.pack(pady=(0, 20))

        ccif_btn = ttk.Button(main_frame, text="Generar CCIF", 
                              command=lambda: self.generate_imd_signal("CCIF"), width=25)
        ccif_btn.pack(pady=10)

        smpte_btn = ttk.Button(main_frame, text="Generar SMPTE", 
                               command=lambda: self.generate_imd_signal("SMPTE"), width=25)
        smpte_btn.pack(pady=10)

        import_btn = ttk.Button(main_frame, text="Importar WAV", 
                               command=self.import_wav_signal,
                               width=25)
        import_btn.pack(pady=10)

    def select_imd_method(self, method):
        self.selected_imd_method = method
        if hasattr(self, 'imd_dropdown'):
            self.imd_dropdown.set(self.selected_imd_method)

    def on_imd_method_changed(self, event=None):
        self.selected_imd_method = self.imd_dropdown.get()

    def generate_imd_signal(self, method):
        self.selected_imd_method = method
        if hasattr(self, 'imd_method_label'):
            self.imd_method_label.config(text=f"Mètode seleccionat: {self.selected_imd_method}")

        fs = 48000
        if method == "SMPTE":
            self.signal = synth_two_tones(fs=fs, dur=2.0, f1=60.0, f2=7000.0, amp1=0.8, amp2=0.2)
            default_name = "IMD_SMPTE_signal.wav"
        else:
            self.signal = synth_two_tones(fs=fs, dur=2.0, f1=7000.0, f2=7600.0, amp1=0.45, amp2=0.45)
            default_name = "IMD_CCIF_signal.wav"

        self.fs = fs

        file_path = filedialog.asksaveasfilename(
            defaultextension=".wav",
            filetypes=[("WAV files", "*.wav"), ("All files", "*.*")],
            initialfile=default_name
        )
        if not file_path:
            return

        try:
            wavfile.write(file_path, self.fs, (self.signal * 32767).astype(np.int16))
            messagebox.showinfo("Èxit", f"Fitxer IMD generat i desat a:\n{file_path}")
        except Exception as e:
            messagebox.showerror("Error", f"No s'ha pogut desar el fitxer: {e}")

    def show_test_signal_ready_screen(self):
        """Pantalla que mostra només opcions per reproduir o desar la senyal de prova (sense analitzar)."""
        for widget in self.winfo_children():
            widget.destroy()

        main_frame = ttk.Frame(self, padding="20")
        main_frame.pack(expand=True, fill=tk.BOTH)

        back_btn = ttk.Button(main_frame, text="← Tornar", 
                              command=self.show_signal_input_screen)
        back_btn.pack(anchor=tk.NW, pady=(0, 20))

        title_label = ttk.Label(main_frame, text="Senyal de prova llesta", 
                               font=("Arial", 14, "bold"))
        title_label.pack(pady=20)

        info_label = ttk.Label(main_frame, text="Aquesta senyal és neta (sense distorsió). Pots reproduir-la o descarregar-la.", 
                               font=("Arial", 12), wraplength=760)
        info_label.pack(pady=10)

        play_btn = ttk.Button(main_frame, text="Reproduir senyal", 
                               command=self.play_signal, width=30)
        play_btn.pack(pady=10)

        save_btn = ttk.Button(main_frame, text="Descarregar WAV", 
                               command=self.save_signal, width=30)
        save_btn.pack(pady=10)

        home_btn = ttk.Button(main_frame, text="Anar a l'inici", 
                               command=self.show_selection_screen, width=30)
        home_btn.pack(pady=10)

        self.simulator_status_label = ttk.Label(main_frame, text="", font=("Arial", 11))
        self.simulator_status_label.pack(pady=8)

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

        self.distortion_applied = True
        self.show_simulator_result_screen()

    def show_generator_distortion_dialog(self):
        """Mostra un menú amb botons per seleccionar el tipus de senyal a generar"""
        for widget in self.winfo_children():
            widget.destroy()

        main_frame = ttk.Frame(self, padding="20")
        main_frame.pack(expand=True, fill=tk.BOTH)

        back_btn = ttk.Button(main_frame, text="← Tornar", 
                              command=self.show_simulator_screen)
        back_btn.pack(anchor=tk.NW, pady=(0, 20))

        title_label = ttk.Label(main_frame, text="Selecciona el tipus de senyal", 
                               font=("Arial", 14, "bold"))
        title_label.pack(pady=20)

        # Botó Sinusoïdal
        sine_btn = ttk.Button(main_frame, text="Sinusoïdal pur", 
                             command=self.show_sine_dialog, width=35)
        sine_btn.pack(pady=8)

        # Botó THD
        thd_btn = ttk.Button(main_frame, text="To amb THD", 
                            command=self.show_synth_thd_dialog, width=35)
        thd_btn.pack(pady=8)

        # Botó THD + Soroll
        thd_noise_btn = ttk.Button(main_frame, text="To amb THD + Soroll", 
                                  command=self.show_synth_thd_noise_dialog, width=35)
        thd_noise_btn.pack(pady=8)

        # Botó Sweep
        sweep_btn = ttk.Button(main_frame, text="Sweep (logarítmic/lineal)", 
                              command=self.show_sweep_dialog, width=35)
        sweep_btn.pack(pady=8)

    def show_sine_dialog(self):
        """Diàleg per generar un sinusoïdal pur"""
        for widget in self.winfo_children():
            widget.destroy()

        main_frame = ttk.Frame(self, padding="20")
        main_frame.pack(expand=True, fill=tk.BOTH)

        back_btn = ttk.Button(main_frame, text="← Tornar", 
                              command=self.show_generator_distortion_dialog)
        back_btn.pack(anchor=tk.NW, pady=(0, 20))

        title_label = ttk.Label(main_frame, text="Sinusoïdal pur", 
                               font=("Arial", 14, "bold"))
        title_label.pack(pady=20)

        # Freq. mostratge
        fs_label = ttk.Label(main_frame, text="Freq. mostratge (Hz):")
        fs_label.pack(pady=5)
        fs_options = ["44100", "48000", "96000", "192000"]
        self.sine_fs = ttk.Combobox(main_frame, values=fs_options, state="readonly", width=20)
        self.sine_fs.set("48000")
        self.sine_fs.pack(pady=5)

        # Durada
        dur_label = ttk.Label(main_frame, text="Durada (segons):")
        dur_label.pack(pady=5)
        self.sine_dur = ttk.Entry(main_frame, width=20)
        self.sine_dur.insert(0, "2.0")
        self.sine_dur.pack(pady=5)

        # Freqüència
        f0_label = ttk.Label(main_frame, text="Freqüència (Hz):")
        f0_label.pack(pady=5)
        self.sine_f0 = ttk.Entry(main_frame, width=20)
        self.sine_f0.insert(0, "1000")
        self.sine_f0.pack(pady=5)

        # Amplitud
        amp_label = ttk.Label(main_frame, text="Amplitud (0-1):")
        amp_label.pack(pady=5)
        self.sine_amp = ttk.Entry(main_frame, width=20)
        self.sine_amp.insert(0, "0.9")
        self.sine_amp.pack(pady=5)

        generate_btn = ttk.Button(main_frame, text="Generar", 
                                 command=self.generate_sine_signal, width=30)
        generate_btn.pack(pady=20)

    def generate_sine_signal(self):
        try:
            fs = int(self.sine_fs.get())
            dur = float(self.sine_dur.get())
            f0 = float(self.sine_f0.get())
            amp = float(self.sine_amp.get())
            
            if dur <= 0 or f0 <= 0 or amp <= 0:
                messagebox.showerror("Error", "Els paràmetres han de ser positius")
                return
                
            self.signal = sine(fs, dur, f0, amp)
            self.original_signal = self.signal.copy()
            self.fs = fs
            self.distortion_applied = False
            self.show_simulator_result_screen()
        except ValueError:
            messagebox.showerror("Error", "Els paràmetres han de ser numèrics")

    def show_synth_thd_dialog(self):
        """Diàleg per generar un to amb THD"""
        for widget in self.winfo_children():
            widget.destroy()

        main_frame = ttk.Frame(self, padding="20")
        main_frame.pack(expand=True, fill=tk.BOTH)

        back_btn = ttk.Button(main_frame, text="← Tornar", 
                              command=self.show_generator_distortion_dialog)
        back_btn.pack(anchor=tk.NW, pady=(0, 20))

        title_label = ttk.Label(main_frame, text="To amb THD", 
                               font=("Arial", 14, "bold"))
        title_label.pack(pady=20)

        # Freq. mostratge
        fs_label = ttk.Label(main_frame, text="Freq. mostratge (Hz):")
        fs_label.pack(pady=5)
        fs_options = ["44100", "48000", "96000", "192000"]
        self.thd_fs = ttk.Combobox(main_frame, values=fs_options, state="readonly", width=20)
        self.thd_fs.set("48000")
        self.thd_fs.pack(pady=5)

        # Durada
        dur_label = ttk.Label(main_frame, text="Durada (segons):")
        dur_label.pack(pady=5)
        self.thd_dur = ttk.Entry(main_frame, width=20)
        self.thd_dur.insert(0, "2.0")
        self.thd_dur.pack(pady=5)

        # Freqüència
        f0_label = ttk.Label(main_frame, text="Freqüència fonamental (Hz):")
        f0_label.pack(pady=5)
        self.thd_f0 = ttk.Entry(main_frame, width=20)
        self.thd_f0.insert(0, "1000")
        self.thd_f0.pack(pady=5)

        # THD target
        thd_label = ttk.Label(main_frame, text="THD target (0.0 - 1.0):")
        thd_label.pack(pady=5)
        self.thd_target = ttk.Entry(main_frame, width=20)
        self.thd_target.insert(0, "0.05")
        self.thd_target.pack(pady=5)

        generate_btn = ttk.Button(main_frame, text="Generar", 
                                 command=self.generate_synth_thd_signal, width=30)
        generate_btn.pack(pady=20)

    def generate_synth_thd_signal(self):
        try:
            fs = int(self.thd_fs.get())
            dur = float(self.thd_dur.get())
            f0 = float(self.thd_f0.get())
            thd_target = float(self.thd_target.get())
            
            if dur <= 0 or f0 <= 0 or thd_target < 0:
                messagebox.showerror("Error", "Els paràmetres han de ser positius")
                return
                
            self.signal = synth_tone_with_thd(fs, dur, f0, thd_target)
            self.original_signal = self.signal.copy()
            self.fs = fs
            self.distortion_applied = False
            self.show_simulator_result_screen()
        except ValueError:
            messagebox.showerror("Error", "Els paràmetres han de ser numèrics")

    def show_synth_thd_noise_dialog(self):
        """Diàleg per generar un to amb THD i soroll"""
        for widget in self.winfo_children():
            widget.destroy()

        main_frame = ttk.Frame(self, padding="20")
        main_frame.pack(expand=True, fill=tk.BOTH)

        back_btn = ttk.Button(main_frame, text="← Tornar", 
                              command=self.show_generator_distortion_dialog)
        back_btn.pack(anchor=tk.NW, pady=(0, 20))

        title_label = ttk.Label(main_frame, text="To amb THD + Soroll", 
                               font=("Arial", 14, "bold"))
        title_label.pack(pady=20)

        # Freq. mostratge
        fs_label = ttk.Label(main_frame, text="Freq. mostratge (Hz):")
        fs_label.pack(pady=5)
        fs_options = ["44100", "48000", "96000", "192000"]
        self.thd_noise_fs = ttk.Combobox(main_frame, values=fs_options, state="readonly", width=20)
        self.thd_noise_fs.set("48000")
        self.thd_noise_fs.pack(pady=5)

        # Durada
        dur_label = ttk.Label(main_frame, text="Durada (segons):")
        dur_label.pack(pady=5)
        self.thd_noise_dur = ttk.Entry(main_frame, width=20)
        self.thd_noise_dur.insert(0, "2.0")
        self.thd_noise_dur.pack(pady=5)

        # Freqüència
        f0_label = ttk.Label(main_frame, text="Freqüència fonamental (Hz):")
        f0_label.pack(pady=5)
        self.thd_noise_f0 = ttk.Entry(main_frame, width=20)
        self.thd_noise_f0.insert(0, "1000")
        self.thd_noise_f0.pack(pady=5)

        # THD target
        thd_label = ttk.Label(main_frame, text="THD target (0.0 - 1.0):")
        thd_label.pack(pady=5)
        self.thd_noise_target = ttk.Entry(main_frame, width=20)
        self.thd_noise_target.insert(0, "0.05")
        self.thd_noise_target.pack(pady=5)

        # Noise level
        noise_label = ttk.Label(main_frame, text="Nivell de soroll (0.0 - 0.1):")
        noise_label.pack(pady=5)
        self.thd_noise_level = ttk.Entry(main_frame, width=20)
        self.thd_noise_level.insert(0, "0.01")
        self.thd_noise_level.pack(pady=5)

        generate_btn = ttk.Button(main_frame, text="Generar", 
                                 command=self.generate_synth_thd_noise_signal, width=30)
        generate_btn.pack(pady=20)

    def generate_synth_thd_noise_signal(self):
        try:
            fs = int(self.thd_noise_fs.get())
            dur = float(self.thd_noise_dur.get())
            f0 = float(self.thd_noise_f0.get())
            thd_target = float(self.thd_noise_target.get())
            noise_level = float(self.thd_noise_level.get())
            
            if dur <= 0 or f0 <= 0 or thd_target < 0 or noise_level < 0:
                messagebox.showerror("Error", "Els paràmetres han de ser positius")
                return
                
            self.signal = synth_tone_with_thd_and_noise(fs, dur, f0, thd_target, noise_level=noise_level)
            self.original_signal = self.signal.copy()
            self.fs = fs
            self.distortion_applied = False
            self.show_simulator_result_screen()
        except ValueError:
            messagebox.showerror("Error", "Els paràmetres han de ser numèrics")

    def show_sweep_dialog(self):
        """Diàleg per generar un sweep logarítmic o lineal"""
        for widget in self.winfo_children():
            widget.destroy()

        main_frame = ttk.Frame(self, padding="20")
        main_frame.pack(expand=True, fill=tk.BOTH)

        back_btn = ttk.Button(main_frame, text="← Tornar", 
                              command=self.show_generator_distortion_dialog)
        back_btn.pack(anchor=tk.NW, pady=(0, 20))

        title_label = ttk.Label(main_frame, text="Sweep", 
                               font=("Arial", 14, "bold"))
        title_label.pack(pady=20)

        # Freq. mostratge
        fs_label = ttk.Label(main_frame, text="Freq. mostratge (Hz):")
        fs_label.pack(pady=5)
        fs_options = ["44100", "48000", "96000", "192000"]
        self.sweep_fs = ttk.Combobox(main_frame, values=fs_options, state="readonly", width=20)
        self.sweep_fs.set("48000")
        self.sweep_fs.pack(pady=5)

        # Durada
        dur_label = ttk.Label(main_frame, text="Durada (segons):")
        dur_label.pack(pady=5)
        self.sweep_dur = ttk.Entry(main_frame, width=20)
        self.sweep_dur.insert(0, "5.0")
        self.sweep_dur.pack(pady=5)

        # Freqüència inicial
        f_start_label = ttk.Label(main_frame, text="Freqüència inicial (Hz):")
        f_start_label.pack(pady=5)
        self.sweep_f_start = ttk.Entry(main_frame, width=20)
        self.sweep_f_start.insert(0, "20")
        self.sweep_f_start.pack(pady=5)

        # Freqüència final
        f_end_label = ttk.Label(main_frame, text="Freqüència final (Hz):")
        f_end_label.pack(pady=5)
        self.sweep_f_end = ttk.Entry(main_frame, width=20)
        self.sweep_f_end.insert(0, "20000")
        self.sweep_f_end.pack(pady=5)

        # Tipus de sweep
        sweep_type_label = ttk.Label(main_frame, text="Tipus de sweep:")
        sweep_type_label.pack(pady=5)
        sweep_options = ["Logarítmic", "Lineal"]
        self.sweep_type = ttk.Combobox(main_frame, values=sweep_options, state="readonly", width=20)
        self.sweep_type.set("Logarítmic")
        self.sweep_type.pack(pady=5)

        generate_btn = ttk.Button(main_frame, text="Generar", 
                                 command=self.generate_sweep_signal, width=30)
        generate_btn.pack(pady=20)

    def generate_sweep_signal(self):
        try:
            fs = int(self.sweep_fs.get())
            dur = float(self.sweep_dur.get())
            f_start = float(self.sweep_f_start.get())
            f_end = float(self.sweep_f_end.get())
            sweep_type = self.sweep_type.get()
            
            if dur <= 0 or f_start <= 0 or f_end <= 0:
                messagebox.showerror("Error", "Els paràmetres han de ser positius")
                return
            if f_start >= f_end:
                messagebox.showerror("Error", "La freqüència inicial ha de ser menor a la final")
                return
                
            if sweep_type == "Logarítmic":
                self.signal = sweep_generator(fs, dur, f_start, f_end)
            else:
                self.signal = sweep_generator_linear(fs, dur, f_start, f_end)
            
            self.original_signal = self.signal.copy()
            self.fs = fs
            self.distortion_applied = False
            self.show_simulator_result_screen()
        except ValueError:
            messagebox.showerror("Error", "Els paràmetres han de ser numèrics")

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

    def show_simulator_result_screen(self):
        """Mostra opcions per reproduir o desar la senyal simulada"""
        for widget in self.winfo_children():
            widget.destroy()

        main_frame = ttk.Frame(self, padding="20")
        main_frame.pack(expand=True, fill=tk.BOTH)

        back_btn = ttk.Button(main_frame, text="← Tornar", 
                              command=self.show_simulator_screen)
        back_btn.pack(anchor=tk.NW, pady=(0, 20))

        title_label = ttk.Label(main_frame, text="Senyal simulada llesta", 
                               font=("Arial", 14, "bold"))
        title_label.pack(pady=20)

        info_label = ttk.Label(main_frame, text="Ara pots reproduir la senyal o desar-la com a WAV.",
                               font=("Arial", 12))
        info_label.pack(pady=10)

        if not self.distortion_applied:
            distort_btn = ttk.Button(main_frame, text="Aplicar distorsió",
                                     command=self.show_distortion_options_dialog, width=30)
            distort_btn.pack(pady=10)

        play_btn = ttk.Button(main_frame, text="Reproduir senyal", 
                               command=self.play_signal, width=30)
        play_btn.pack(pady=10)

        save_btn = ttk.Button(main_frame, text="Descarregar WAV", 
                               command=self.save_signal, width=30)
        save_btn.pack(pady=10)

        home_btn = ttk.Button(main_frame, text="Anar a l'inici", 
                               command=self.show_selection_screen, width=30)
        home_btn.pack(pady=10)

        self.simulator_status_label = ttk.Label(main_frame, text="", font=("Arial", 11))
        self.simulator_status_label.pack(pady=8)

    def show_distortion_options_dialog(self):
        """Diàleg per seleccionar el tipus de distorsió a aplicar a la senyal generada"""
        for widget in self.winfo_children():
            widget.destroy()

        main_frame = ttk.Frame(self, padding="20")
        main_frame.pack(expand=True, fill=tk.BOTH)

        back_btn = ttk.Button(main_frame, text="← Tornar", 
                              command=self.show_simulator_result_screen)
        back_btn.pack(anchor=tk.NW, pady=(0, 20))

        title_label = ttk.Label(main_frame, text="Aplicar distorsió", 
                               font=("Arial", 14, "bold"))
        title_label.pack(pady=20)

        # Selecció del mètode de distorsió
        method_label = ttk.Label(main_frame, text="Mètode de distorsió:")
        method_label.pack(pady=10)
        
        methods = ["soft_clip_tanh", "apply_nonlinear_distortion"]
        self.dist_method_var = tk.StringVar(value=methods[0])
        
        for method in methods:
            rb = ttk.Radiobutton(main_frame, text=method, variable=self.dist_method_var, value=method)
            rb.pack(anchor=tk.W, padx=40, pady=5)

        # Paràmetre de distorsió
        param_label = ttk.Label(main_frame, text="Paràmetre de distorsió:", 
                               font=("Arial", 11))
        param_label.pack(pady=10)

        self.dist_param_scale = ttk.Scale(main_frame, from_=0.1, to=5.0, orient=tk.HORIZONTAL, length=300)
        self.dist_param_scale.set(0.5)
        self.dist_param_scale.pack(pady=5)

        param_value_label = ttk.Label(main_frame, text="0.5")
        param_value_label.pack(pady=5)
        
        def update_param_label(event=None):
            value = float(self.dist_param_scale.get())
            param_value_label.config(text=f"{value:.2f}")
        
        self.dist_param_scale.bind("<B1-Motion>", update_param_label)
        self.dist_param_scale.bind("<Button-1>", update_param_label)

        # Botó per aplicar
        apply_btn = ttk.Button(main_frame, text="Aplicar distorsió", 
                               command=self.apply_distortion_to_generated_signal,
                               width=30)
        apply_btn.pack(pady=20)

    def apply_distortion_to_generated_signal(self):
        """Aplica la distorsió seleccionada a la senyal generada"""
        try:
            method = self.dist_method_var.get()
            param = float(self.dist_param_scale.get())
            
            if self.signal is None or self.fs is None:
                messagebox.showerror("Error", "No hi ha senyal per distorsionar")
                return
            
            # Aplicar distorsió
            if method == "soft_clip_tanh":
                self.signal = soft_clip_tanh(self.signal, dist=param)
            else:
                self.signal = apply_nonlinear_distortion(self.signal, alpha=param)
            
            # Mostrar pantalla de resultado amb la senyal distorsionada
            self.distortion_applied = True
            self.show_simulator_result_screen()
            
        except ValueError:
            messagebox.showerror("Error", "El paràmetre ha de ser numèric")
    
    def play_signal(self):
        """Reprodueix la senyal simulada utilitzant sounddevice."""
        if self.signal is None or self.fs is None:
            messagebox.showerror("Error", "No hi ha cap senyal per reproduir.")
            return

        def playback():
            try:
                sd.play(self.signal, self.fs)
                sd.wait()
            except Exception as e:
                messagebox.showerror("Error", f"No s'ha pogut reproduir la senyal: {e}")

        threading.Thread(target=playback, daemon=True).start()

    def save_signal(self):
        """Desa la senyal simulada a un fitxer WAV."""
        if self.signal is None or self.fs is None:
            messagebox.showerror("Error", "No hi ha cap senyal per desar.")
            return

        file_path = filedialog.asksaveasfilename(
            defaultextension=".wav",
            filetypes=[("WAV files", "*.wav"), ("All files", "*.*")],
            initialfile="simulated_signal.wav"
        )
        if not file_path:
            return

        try:
            signal_to_save = self.signal
            if signal_to_save.dtype != np.int16:
                signal_to_save = np.clip(signal_to_save, -1.0, 1.0)
                signal_to_save = (signal_to_save * 32767).astype(np.int16)
            wavfile.write(file_path, self.fs, signal_to_save)
            messagebox.showinfo("Èxit", f"Fitxer guardat a:\n{file_path}")
        except Exception as e:
            messagebox.showerror("Error", f"No s'ha pogut desar el fitxer: {e}")

    def show_analysis_screen(self):
        """Mostra la pantalla principal d'anàlisi"""
        # Esborrar els widgets anteriors
        for widget in self.winfo_children():
            widget.destroy()
        
        # Frame principal
        main_frame = ttk.Frame(self, padding="10")
        main_frame.pack(expand=True, fill=tk.BOTH)
        
        # Frame de botons de control
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(anchor=tk.NW, fill=tk.X, pady=(0, 10))
        
        # Botó per tornar enrere
        back_btn = ttk.Button(button_frame, text="← Tornar", 
                             command=self.show_selection_screen)
        back_btn.pack(side=tk.LEFT, padx=(0, 10))
        
        # Botó per descarregar WAV si és gravació
        if self.is_recording_source and self.signal is not None and self.fs is not None:
            download_btn = ttk.Button(button_frame, text="Descarregar WAV", 
                                     command=self.download_recorded_wav)
            download_btn.pack(side=tk.LEFT)
        
        # Botó per descarregar informe en PDF (sempre disponible si hi ha senyal)
        if self.signal is not None and self.fs is not None:
            pdf_btn = ttk.Button(button_frame, text="Descarregar PDF",
                                 command=self.export_result_pdf)
            pdf_btn.pack(side=tk.LEFT, padx=(6, 0))

        # Botó de detall Farina (només per a anàlisi Farina)
        if (self.selected_analysis_type == "THD" and self.selected_thd_type == "Farina"
                and hasattr(self, 'current_farina') and self.current_farina is not None):
            farina_detail_btn = ttk.Button(button_frame, text="Detall Farina",
                                           command=lambda: self.current_farina.plot_far_response())
            farina_detail_btn.pack(side=tk.LEFT, padx=(6, 0))
        
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

        if self.selected_analysis_type == "IMD":
            selected_method_text = f"Mètode IMD seleccionat: {self.selected_imd_method}"
            selected_method_label = ttk.Label(main_frame, text=selected_method_text, 
                                              font=("Arial", 12), foreground="#333333")
            selected_method_label.pack(fill=tk.X, pady=(0, 10))

            hint_text = (
                "CCIF: senyal amb dos tons d'alta freqüència i mateixa amplitud. "
                "SMPTE: dos tons, un de baixa freqüència / alta amplitud i un d'alta freqüència / baixa amplitud."
            )
            hint_label = ttk.Label(main_frame, text=hint_text, font=("Arial", 10), foreground="#333333", wraplength=760)
            hint_label.pack(fill=tk.X, pady=(0, 10))

        # Gràfics: un o dos eixos segons el tipus d'anàlisi
        if self.selected_analysis_type == "THD" and self.selected_thd_type == "Farina":
            self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(12, 6))
            self.ax2.set_title("Impulse Response", fontsize=12, fontweight="bold")
            self.ax2.set_xlabel("Mostres")
            self.ax2.set_ylabel("Amplitud")
        else:
            self.fig, self.ax1 = plt.subplots(figsize=(10, 6))
            self.ax2 = None
        self.fig.tight_layout(pad=4.0)
        self.canvas = FigureCanvasTkAgg(self.fig, master=main_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        plt.close(self.fig)  # desregistra del gestor pyplot per evitar que plt.show() el mostri com a finestra separada
        
        if self.selected_analysis_type == "IMD":
            if self.signal is not None and self.fs is not None:
                note_text = f"Anàlisi IMD automàtica amb mètode {self.selected_imd_method}."
            else:
                note_text = "Carrega o genera una senyal CCIF/SMPTE abans de poder analitzar IMD."
            note_label = ttk.Label(main_frame, text=note_text, 
                                   font=("Arial", 11), foreground="#333333")
            note_label.pack(pady=(5, 10))

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
    
    def imd_method_selected(self, event=None):
        self.selected_imd_method = self.imd_dropdown.get()
        if self.selected_imd_method == "CCIF":
            reminder = (
                "Recordatori CCIF:\n" 
                "Utilitza una senyal amb dos tons d'alta freqüència i mateixa amplitud."
            )
        else:
            reminder = (
                "Recordatori SMPTE:\n"
                "Utilitza una senyal amb dos tons, un de baixa freqüència i alta amplitud i un d'alta freqüència i baixa amplitud."
            )
        messagebox.showinfo("Recordatori IMD", reminder)

    def analyze_signal(self, signal, fs):
        """Analitza una senyal i mostra els gràfics"""
        if signal is None or fs is None:
            messagebox.showerror("Error", "No hi ha cap senyal per analitzar.")
            return
        self.ax1.clear()
        if self.ax2 is not None:
            self.ax2.clear()
        
        # FFT plot
        N = len(signal)
        spectrum = abs(np.fft.fft(signal)[:N//2])
        freqs = np.fft.fftfreq(N, 1/fs)[:N//2]
        positive = freqs > 0
        freqs = freqs[positive]
        spectrum = spectrum[positive]
        if len(freqs) == 0:
            self.ax1.text(0.5, 0.5, "No hi ha dades FFT vàlides per mostrar.",
                          ha='center', va='center', transform=self.ax1.transAxes)
        else:
            # Convertir l'amplitud a decibels relatius
            eps = 1e-12
            spectrum_db = 20 * np.log10(np.maximum(spectrum / np.max(spectrum), eps))
            self.ax1.semilogx(freqs, spectrum_db, color="#1e88e5", linewidth=1.5)
            self.ax1.set_xscale('log', nonpositive='clip')
            self.ax1.set_xlim([20, 20000])
            self.ax1.set_ylim([np.min(spectrum_db) - 5, np.max(spectrum_db) + 5])
        self.ax1.set_title("Resposta en freqüència (FFT)", fontsize=16, fontweight="bold")
        self.ax1.set_xlabel("Freqüència (Hz)", fontsize=12)
        self.ax1.set_ylabel("Amplitud (dB rel.)", fontsize=12)
        self.ax1.grid(True, which='both', linestyle="--", alpha=0.4)
        self.ax1.tick_params(axis='both', labelsize=10)

        self.canvas.draw()
        
        # Calcula i mostra el resultat segons el tipus d'anàlisi
        if self.selected_analysis_type == "THD":
            try:
                if self.selected_thd_type == "THD_F":
                    thd = compute_THD_F(signal, fs, len(signal))
                    self.thd_label.config(text=f"THD_F: {thd:.2f}%")
                    self.draw_harmonic_thd_table(signal, fs)
                elif self.selected_thd_type == "THD_RMS":
                    thd = compute_THD_rms(signal, fs, len(signal))
                    self.thd_label.config(text=f"THD_RMS: {thd:.2f}%")
                    self.draw_harmonic_thd_table(signal, fs)
                elif self.selected_thd_type == "THD_N":
                    thd = compute_THDN(signal, fs, len(signal))
                    self.thd_label.config(text=f"THD+N: {thd:.2f}%")
                elif self.selected_thd_type == "Farina":
                    # Farina analysis: expect a Farina object in self.current_farina
                    if hasattr(self, 'current_farina') and self.current_farina is not None:
                        try:
                            thd = self.current_farina.getTHD()
                            self.thd_label.config(text=f"Farina THD: {thd:.2f}%")
                            if self.ax2 is not None:
                                try:
                                    ir = self.current_farina.get_IR()
                                    self.ax2.plot(ir, color='#d32f2f')
                                    self.ax2.set_xlim([0, len(ir)])
                                    self.ax2.set_ylim([np.min(ir) - 0.01, np.max(ir) + 0.01])
                                    self.ax2.set_title('Impulse Response', fontsize=12, fontweight='bold')
                                    self.ax2.set_xlabel('Mostres')
                                    self.ax2.set_ylabel('Amplitud')
                                    self.ax2.grid(True, linestyle='--', alpha=0.4)
                                except Exception:
                                    pass
                        except Exception as e:
                            self.thd_label.config(text=f"Error Farina THD: {e}")
                    else:
                        self.thd_label.config(text="Farina: objecte no disponible")
                elif self.selected_thd_type == "THD_SWEEP":
                    thd_values, freqs = compute_THD_sweep(signal, fs, num_partitions=10)
                    self.thd_label.config(text=f"THD_SWEEP (mitjana): {np.mean(thd_values):.2f}%")
                    self.ax1.clear()
                    self.ax1.semilogx(freqs, thd_values, color="#1e88e5", linewidth=1.5, marker='o', markersize=4)
                    self.ax1.set_title("THD per segment")
                    self.ax1.set_xlabel("Freqüència (Hz)")
                    self.ax1.set_ylabel("THD (%)")
                    self.ax1.set_xscale('log', nonpositive='clip')
                    self.ax1.set_xlim([20, 20000])
                    self.ax1.grid(True, which='both', linestyle='--', alpha=0.4)
                    self.canvas.draw()
                    return
                else:
                    self.thd_label.config(text="THD: Tipus desconegut")
            except Exception as e:
                self.thd_label.config(text=f"Error al calcular THD: {e}")
        elif self.selected_analysis_type == "IMD":
            try:
                if self.selected_imd_method == "CCIF":
                    imd_value = compute_IMD_ccif(signal, fs, f1=7000.0, f2=7600.0)
                    self.thd_label.config(text=f"IMD (CCIF): {imd_value:.2f}%")
                else:
                    imd_value = compute_IMD_smpte(signal, fs, f1=60.0, f2=7000.0)
                    self.thd_label.config(text=f"IMD (SMPTE): {imd_value:.2f}%")
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

    def draw_harmonic_thd_table(self, signal, fs, H=6):
        """Dibuixa una taula amb els valors de THD per cada harmonic."""
        _, harmonics = compute_THD_harmonic(signal, fs, len(signal), H=H)
        if not harmonics:
            return

        if self.selected_thd_type == "THD_F":
            col_labels = ["Harmònic", "Freq (Hz)", "THD_F (%)"]
            table_data = [
                [str(h["h"]), f"{h['freq']:.0f}", f"{h['thd_f_pct']:.2f}"]
                for h in harmonics
            ]
        else:
            col_labels = ["Harmònic", "Freq (Hz)", "THD_RMS (%)"]
            table_data = [
                [str(h["h"]), f"{h['freq']:.0f}", f"{h['thd_rms_pct']:.2f}"]
                for h in harmonics
            ]

        table = self.ax1.table(
            cellText=table_data,
            colLabels=col_labels,
            loc="upper right",
            cellLoc="center",
            colLoc="center",
            bbox=[0.70, 0.55, 0.27, 0.30],
        )
        table.auto_set_font_size(False)
        table.set_fontsize(7)
        table.scale(0.95, 0.95)
        self.ax1.figure.subplots_adjust(right=0.80)

    def download_recorded_wav(self):
        """Descarrega el WAV de la gravació realitzada"""
        if self.signal is None or self.fs is None:
            messagebox.showerror("Error", "No hi ha senyal per descarregar")
            return
        
        # Demanar on guardar
        file_path = filedialog.asksaveasfilename(
            defaultextension=".wav",
            filetypes=[("WAV files", "*.wav"), ("All files", "*.*")],
            initialfile=f"recorded_audio_{self.selected_analysis_type}.wav"
        )
        
        if not file_path:
            return
        
        try:
            # Convertir a int16 per guardar com a WAV estàndard
            signal_int16 = (self.signal * 32767).astype(np.int16)
            wavfile.write(file_path, self.fs, signal_int16)
            messagebox.showinfo("Èxit", f"Gravació descarregada correctament a:\n{file_path}")
        except Exception as e:
            messagebox.showerror("Error", f"No s'ha pogut descarregar el fitxer: {e}")

    def export_result_pdf(self):
        """Genera i desa un PDF amb el resultat i el gràfic actual."""
        if self.signal is None or self.fs is None:
            messagebox.showerror("Error", "No hi ha senyal per exportar")
            return

        file_path = filedialog.asksaveasfilename(
            defaultextension=".pdf",
            filetypes=[("PDF files", "*.pdf"), ("All files", "*.*")],
            initialfile=f"informe_{self.selected_analysis_type}.pdf"
        )
        if not file_path:
            return

        try:
            result_text = self.thd_label.cget("text") if hasattr(self, 'thd_label') else "Resultat no disponible"
            is_farina = (self.selected_analysis_type == "THD"
                         and self.selected_thd_type == "Farina"
                         and hasattr(self, 'current_farina') and self.current_farina is not None)

            with PdfPages(file_path) as pdf:
                # --- Pàgina 1: FFT + IR (Farina) o només FFT (resta) ---
                if is_farina:
                    fig, (ax_fft, ax_ir) = plt.subplots(1, 2, figsize=(12, 5))
                else:
                    fig, ax_fft = plt.subplots(figsize=(10, 5))

                header = f"DistorLab – Informe d'anàlisi\nResultat: {result_text}"
                fig.suptitle(header, fontsize=12, fontweight='bold')

                # FFT
                N = len(self.signal)
                spectrum = abs(np.fft.fft(self.signal)[:N//2])
                freqs = np.fft.fftfreq(N, 1/self.fs)[:N//2]
                positive = freqs > 0
                freqs_p = freqs[positive]
                spectrum_p = spectrum[positive]
                if len(freqs_p) == 0:
                    ax_fft.text(0.5, 0.5, "No hi ha dades FFT vàlides.", ha='center')
                else:
                    eps = 1e-12
                    spectrum_db = 20 * np.log10(spectrum_p / (np.max(spectrum_p) + eps) + eps)
                    ax_fft.plot(freqs_p, spectrum_db, color="#1e88e5", linewidth=1.5)
                    ax_fft.set_xscale('log', nonpositive='clip')
                    ax_fft.set_xlim([20, 20000])
                    ax_fft.set_ylim([np.min(spectrum_db) - 5, np.max(spectrum_db) + 5])
                ax_fft.set_title("Resposta en freqüència (FFT)", fontsize=11)
                ax_fft.set_xlabel("Freqüència (Hz)")
                ax_fft.set_ylabel("Amplitud (dB rel.)")
                ax_fft.grid(True, linestyle="--", alpha=0.4)

                # IR (només Farina)
                if is_farina:
                    ir = self.current_farina.get_IR()
                    ax_ir.plot(ir, color='#d32f2f', linewidth=0.9)
                    ax_ir.set_title("Resposta Impulsional (H1)", fontsize=11)
                    ax_ir.set_xlabel("Mostres")
                    ax_ir.set_ylabel("Amplitud")
                    ax_ir.grid(True, linestyle="--", alpha=0.4)

                fig.tight_layout(rect=[0, 0, 1, 0.92])
                pdf.savefig(fig, bbox_inches='tight')
                plt.close(fig)

                # --- Pàgina 2 (Farina): subplots harmònics ---
                if is_farina:
                    harm_responses = self.current_farina.harm_responses
                    n_harms = len(harm_responses)
                    colors = ['#e53935', '#43a047', '#1e88e5', '#8e24aa', '#fb8c00',
                              '#00acc1', '#6d4c41', '#e91e63', '#00897b', '#fdd835']

                    fig2, axes = plt.subplots(n_harms, 1, figsize=(12, 2.5 * n_harms))
                    if n_harms == 1:
                        axes = [axes]
                    fig2.suptitle("Farina – Respostes harmòniques", fontsize=13, fontweight="bold")

                    for i, resp in enumerate(harm_responses):
                        ax = axes[i]
                        label = "IR fonamental (H1)" if i == 0 else f"Harmònic H{i + 1}"
                        color = colors[i % len(colors)] if i > 0 else '#455a64'
                        ax.plot(resp, color=color, linewidth=0.9)
                        ax.set_title(label, fontsize=10)
                        ax.set_xlabel("Mostres")
                        ax.set_ylabel("Amplitud")
                        ax.grid(True, linestyle='--', alpha=0.4)

                    fig2.tight_layout(rect=[0, 0, 1, 0.96], h_pad=3.0)
                    pdf.savefig(fig2, bbox_inches='tight')
                    plt.close(fig2)

            messagebox.showinfo("Èxit", f"Informe PDF desat a:\n{file_path}")
        except Exception as e:
            messagebox.showerror("Error", f"No s'ha pogut generar el PDF: {e}")

    def on_close(self):
        """Tanca l'aplicació netament i allibera recursos d'àudio."""
        try:
            sd.stop()
        except Exception:
            pass
        try:
            plt.close('all')
        except Exception:
            pass
        self.destroy()

if __name__ == "__main__":
    app = AudioApp()
    app.mainloop()
