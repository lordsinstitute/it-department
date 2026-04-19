import numpy as np

def mse(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    return float(np.mean((a - b) ** 2))

def psnr(a: np.ndarray, b: np.ndarray, max_val: float = 255.0) -> float:
    m = mse(a, b)
    if m <= 1e-12:
        return 99.0
    import math
    return float(20.0 * math.log10(max_val) - 10.0 * math.log10(m))

def changed_ratio(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(a != b))

def audio_snr_db(original: np.ndarray, stego: np.ndarray) -> float:
    """
    Simple SNR estimate:
      10log10(signal_power / noise_power)
    """
    o = original.astype(np.float64)
    s = stego.astype(np.float64)
    noise = s - o
    sp = np.mean(o ** 2) + 1e-12
    npow = np.mean(noise ** 2) + 1e-12
    import math
    return float(10.0 * math.log10(sp / npow))