import numpy as np
import audio_dspy as adsp
import scipy.signal as signal
from functools import wraps
from generator import sweep_log
import matplotlib.pyplot as plt

# aqui he important manualment tot farina, aixi puc editar el que vulgui i millorar el que necessiti,
#  marcare les meves modificacions amb comentaris com aquests d'inici a fi
class Farina:
    """
    Class that implements Alberto Farina's method [1]_ for
    simultaneously measuring frequency response and harmonic
    distortion of weakly nonlinear systems.

    References
    ----------
    .. [1] A. Farina "Simultaneous Measurement of Impulse Response
           and Distortion with a Swept-Sine Technique", Audio
           Engineering Society Convention 108, Feb. 2000
    """

    def __init__(self, A, duration, fs, f0=20, f1=20000):
        """
        Creates an object to create and process a Farina-style measurement.

        Parameters
        ----------
        duration : float
            length [seconds] of the desired measurement signal
        fs : float
            sample rate [Hz] of the measurement signal
        f0 : float
            Frequency [Hz] at which to start the measurement
        f1 : float
            Frequency [Hz] at which to end the measurement
        """
        N = int(duration * fs)
        self.fs = fs

        # create probe and inverse probe
        self.probe = sweep_log(f0, f1, duration, fs, A)
        R = np.log(f1 / f0)
        k = np.exp(np.arange(N) * R / N)
        print(f"tamany de la k: {len(k)}")
        print(f"tamany de la probe: {len(self.probe)}")
        self.inv_probe = np.flip(self.probe) / k

        # @TEST: test that probe convolved with inverse has flat spectrum,
        # and impulse-like response

        # determin times to look for harmonics
        self.far_response = None
        self.harm_times = [0]
        mult = 1
        while True:
            mult += 1
            delta_n = int(N * np.log(mult) / np.log(f1/f0))
            self.harm_times.append(delta_n)
            # CORRECIÓ: compara la diferència entre harmònics consecutius
            # relativa a la longitud de la senyal (5% de N)
            if self.harm_times[-1] - self.harm_times[-2] < N * 0.03: #0.05
                break

    def process_measurement(self, measurement, fft_size=None, normalize=True, log = False):
        """
        Processes a measurement made using the probe signal
        for this object.
        """
        # ir_in = self.far_response = signal.convolve(self.probe, self.inv_probe)
        # ir_out = self.far_response = signal.convolve(measurement, self.inv_probe)
        # h_t = ir_in / ir_out
        # H_w = fft(h_t, n=fft_size)
        # fer plot de les sub IRs i posar les marques on es fan els talls

        self.far_response = signal.convolve(measurement, self.inv_probe)
        if normalize:
            self.far_response = adsp.normalize(self.far_response)

        amax = np.argmax(self.far_response)
        level = adsp.level_detect(self.far_response, self.fs)
        off = int(self.fs/10)
        amin = np.argwhere(level[amax-off:amax] < 0.05)[-1][0]
        amax = amax - (off - amin)
        end = amax + np.argwhere(level[amax:] < 10**(-60/20))[0][0]

        self.harm_responses = [self.far_response[amax:end]]

        figure, ax = plt.subplots(len(self.harm_times)+1, 1, figsize=(10, 2*len(self.harm_times))) 
        if log:
            ax[0].plot(20*np.log10(np.abs(self.far_response)))
            ax[0].set_ylim([-90,0])
        else:
            ax[0].plot(self.far_response)
            x[0].set_ylim([-1,1])
        colors = ['r', 'g', 'b', 'm', 'c', 'y', 'k', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan', 'navy', 'teal', 'coral', 'gold', 'indigo', 'lime']

        for i in range(1, len(self.harm_times)):
            start = amax - self.harm_times[i]
            end = amax - self.harm_times[i-1]
            self.harm_responses.append(self.far_response[start:end])
            # start = amax - self.harm_times[i]
            # end = amax - self.harm_times[i-1]
            # # Add guard band to reduce spillover from adjacent harmonics (15% on each side)
            # guard = int((end - start) * 0.15)
            # start = start + guard
            # end = end - guard
            # if end > start:  # Only add if window is still valid
            #     self.harm_responses.append(self.far_response[start:end])
            # else:
            #     # If guard band eliminates window, create minimal response
            #     self.harm_responses.append(np.array([0]))
            if log:
                ax[i].plot(20*np.log10(np.abs(self.harm_responses[i-1])))
                ax[i].set_ylim([-90,0])
            else:
                ax[i].plot(self.harm_responses[i-1])
                ax[i].set_ylim([-1,1])
            ax[0].axvline(x=start, color=colors[i], linestyle='--')
            ax[0].axvline(x=end, color=colors[i], linestyle=':')
        

        plt.show()
           
        
    def _check_meas(func):
        """
        Decorator to make sure the measurement has been
        processed before attempting to access anything that
        depends on it.
        """
        @wraps(func)
        def checker(self, *args, **kwargs):
            if self.far_response is None:
                assert False, 'You must process a measurement before calling this function'
            return func(self, *args, **kwargs)
        return checker

    @_check_meas
    def get_harm_response(self, harm_num):
        """
        Returns the impulse response for a certain harmonic
        of the system. Note that the fundamental is the 1st harmonic.
        """
        assert harm_num > 0, 'Harmonic number must be greater than zero!'
        assert harm_num < len(self.harm_times), 'Harmonic number too large!'
        return self.harm_responses[harm_num-1]

    @_check_meas
    def get_IR(self):
        """
        Returns the impulse response for the linear
        part of the system.
        """
        return self.get_harm_response(1)

    @_check_meas
    def getTHD(self, harms=9):
        """
        Returns the estimated total harmonic distortion for the system.
        """
        rms_vals = np.zeros(harms)
        for idx, response in enumerate(self.harm_responses[:harms]):
            r_corr = signal.convolve(response, self.get_IR())
            rms_vals[idx] = np.sqrt(np.mean(r_corr**2))
        rms_vals /= rms_vals[0]
        return (np.sqrt(np.sum(rms_vals[1:]**2)) / rms_vals[0])*100.0
    
    