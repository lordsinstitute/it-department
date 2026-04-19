import os
import wave
import numpy as np
from utils.metrics import audio_snr_db

def _bytes_to_bits(data: bytes) -> np.ndarray:
    arr = np.frombuffer(data, dtype=np.uint8)
    return np.unpackbits(arr).astype(np.uint8)

def _apply_redundancy(bits: np.ndarray, r: int) -> np.ndarray:
    r = max(1, int(r))
    if r == 1:
        return bits
    return np.repeat(bits, r).astype(np.uint8)

def encode_wav_lsb(carrier_path: str, payload: bytes, redundancy_r: int = 1):
    """
    WAV PCM LSB encoding:
    - 1 bit per sample (safe default)
    - redundancy repeats each bit R times
    """
    with wave.open(carrier_path, "rb") as wf:
        params = wf.getparams()
        frames = wf.readframes(wf.getnframes())

    sampwidth = params.sampwidth
    if sampwidth not in (1, 2):
        raise ValueError("Only 8-bit or 16-bit PCM WAV supported.")

    dtype = np.uint8 if sampwidth == 1 else np.int16
    orig = np.frombuffer(frames, dtype=dtype)
    samples = orig.copy()

    bits = _bytes_to_bits(payload)
    bits = _apply_redundancy(bits, redundancy_r)

    needed_bits = int(bits.size)
    capacity_bits = int(samples.size)  # 1 bit per sample

    if needed_bits > capacity_bits:
        raise ValueError(
            f"Insufficient WAV capacity. Need {needed_bits} bits (redundancy included), have {capacity_bits} bits."
        )

    if sampwidth == 1:
        samples[:needed_bits] = (samples[:needed_bits] & 0xFE) | bits
    else:
        samples[:needed_bits] = (samples[:needed_bits] & ~1) | bits

    base, _ = os.path.splitext(carrier_path)
    out_path = f"{base}_stego.wav"

    with wave.open(out_path, "wb") as wf_out:
        wf_out.setparams(params)
        wf_out.writeframes(samples.tobytes())

    used_pct = (needed_bits / max(1, capacity_bits)) * 100.0

    try:
        snr = audio_snr_db(orig.astype(np.float64), samples.astype(np.float64))
    except Exception:
        snr = 0.0

    stats = {
        "carrier": os.path.basename(carrier_path),
        "output_file": os.path.basename(out_path),
        "channels": int(params.nchannels),
        "framerate": int(params.framerate),
        "sampwidth": int(params.sampwidth),
        "nframes": int(params.nframes),
        "duration_sec": round(params.nframes / max(1, params.framerate), 3),
        "redundancy_r": int(redundancy_r),
        "capacity_bits": int(capacity_bits),
        "payload_bits_effective": int(needed_bits),
        "payload_bytes_raw": int(len(payload)),
        "usage_percent": round(float(used_pct), 4),
        "distortion": {"snr_db_est": round(float(snr), 4)},
        "recommendations": [
            "Use PCM WAV (uncompressed) for stable embedding.",
            "Keep usage < 30% for better transparency.",
        ]
    }

    if redundancy_r > 1:
        stats["recommendations"].append(f"Redundancy R={redundancy_r} improves robustness but reduces capacity.")
    if used_pct > 50:
        stats["recommendations"].append("Warning: High density can increase audible artifacts and detectability.")

    return out_path, stats