# primer de tot començare amb el CCIF i mes tard implementaré l'IMD SMPTE
import numpy as np

def compute_IMD_ccif(signal, fs, f1, f2, bins=2, include_3rd=True):
    """
    IMD calculat CCIF que mesura |f2-f1| i opcionalment els productes de 3er ordre.
    Return IMD en %
    """
    x = np.asarray(signal, dtype=np.float64)
    x = x - np.mean(x)  
    N = len(x)
    # windowing
    w = np.hanning(N)
    X = np.fft.rfft(x * w)
    mag = np.abs(X) / (N/2) 
    freqs = np.fft.rfftfreq(N, 1/fs)

    def local_peak_at(f):
        k = int(np.argmin(np.abs(freqs - f)))
        k0 = max(0, k - bins)
        k1 = min(len(mag) - 1, k + bins)
        return float(np.max(mag[k0:k1+1]))

    A1 = local_peak_at(f1)
    A2 = local_peak_at(f2)
    if A1 < 1e-15 and A2 < 1e-15:
        return 0.0

    # referencia
    Aref = np.sqrt(A1**2 + A2**2) + 1e-15

    # component 2n ordre
    fd = abs(f2 - f1)
    Ad = local_peak_at(fd)

    im_components = [Ad]

    # productes 3er ordre
    if include_3rd:
        f_im1 = abs(2*f1 - f2)
        f_im2 = abs(2*f2 - f1)
        if f_im1 < fs/2: im_components.append(local_peak_at(f_im1))
        if f_im2 < fs/2: im_components.append(local_peak_at(f_im2))

    im_rms = np.sqrt(np.sum(np.square(im_components)))
    imd = (im_rms / Aref) * 100.0
    return float(imd)
