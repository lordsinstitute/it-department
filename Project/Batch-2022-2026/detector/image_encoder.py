import os
import numpy as np
from PIL import Image
from utils.metrics import psnr, mse, changed_ratio

def _bytes_to_bits(data: bytes) -> np.ndarray:
    arr = np.frombuffer(data, dtype=np.uint8)
    return np.unpackbits(arr).astype(np.uint8)

def _apply_redundancy(bits: np.ndarray, r: int) -> np.ndarray:
    r = max(1, int(r))
    if r == 1:
        return bits
    return np.repeat(bits, r).astype(np.uint8)

def _edge_priority_indices(rgb: np.ndarray) -> np.ndarray:
    """
    Compute simple edge strength from grayscale gradient magnitude.
    Returns pixel indices sorted by descending edge score.
    """
    # grayscale
    gray = (0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]).astype(np.float32)

    # gradients (cheap central differences)
    gx = np.zeros_like(gray)
    gy = np.zeros_like(gray)
    gx[:, 1:-1] = gray[:, 2:] - gray[:, :-2]
    gy[1:-1, :] = gray[2:, :] - gray[:-2, :]

    mag = np.abs(gx) + np.abs(gy)  # L1 magnitude
    flat = mag.reshape(-1)

    # sort indices by strength desc
    return np.argsort(-flat)

def encode_image_lsb(
    carrier_path: str,
    payload: bytes,
    bits_per_channel: int = 1,
    adaptive: bool = True,
    redundancy_r: int = 1
):
    """
    LSB encoding in RGB channels with:
    - adaptive embedding: edges-first pixel ordering
    - redundancy: repeat each bit R times
    - output always PNG
    """
    if bits_per_channel not in (1, 2):
        raise ValueError("bits_per_channel must be 1 or 2.")

    img = Image.open(carrier_path).convert("RGB")
    orig = np.array(img, dtype=np.uint8)
    arr = orig.copy()

    h, w, c = arr.shape  # c=3
    capacity_bits = h * w * c * bits_per_channel

    bits = _bytes_to_bits(payload)
    bits = _apply_redundancy(bits, redundancy_r)
    needed_bits = int(bits.size)

    if needed_bits > capacity_bits:
        raise ValueError(
            f"Insufficient capacity. Need {needed_bits} bits (redundancy included), have {capacity_bits} bits."
        )

    # Determine write order
    if adaptive:
        pix_order = _edge_priority_indices(arr)  # length h*w
    else:
        pix_order = np.arange(h * w)

    # We embed across RGB bytes for each pixel in chosen order.
    # Create a flat view per pixel: (h*w, 3)
    flat_rgb = arr.reshape(-1, 3)

    # Prepare a flat stream of carrier bytes in embedding order:
    ordered = flat_rgb[pix_order].reshape(-1)  # length (h*w*3)

    # Now embed into ordered bytes
    if bits_per_channel == 1:
        ordered[:needed_bits] = (ordered[:needed_bits] & 0xFE) | bits
    else:
        # 2-bit embedding
        if needed_bits % 2 != 0:
            bits = np.pad(bits, (0, 1), mode="constant")
            needed_bits += 1
        pairs = bits.reshape(-1, 2)
        two_vals = (pairs[:, 0] << 1) | pairs[:, 1]
        n = two_vals.size
        ordered[:n] = (ordered[:n] & 0xFC) | two_vals

    # Write back ordered stream to flat_rgb in pix_order
    flat_rgb[pix_order] = ordered.reshape(-1, 3)

    base, _ = os.path.splitext(carrier_path)
    out_path = f"{base}_stego.png"
    Image.fromarray(arr).save(out_path, format="PNG")

    # metrics
    m = mse(orig, arr)
    p = psnr(orig, arr)
    ch = changed_ratio(orig, arr)

    used_pct = (needed_bits / max(1, capacity_bits)) * 100.0

    stats = {
        "carrier": os.path.basename(carrier_path),
        "output_file": os.path.basename(out_path),
        "width": int(w),
        "height": int(h),
        "channels": int(c),
        "bits_per_channel": int(bits_per_channel),
        "adaptive_embedding": bool(adaptive),
        "redundancy_r": int(redundancy_r),
        "capacity_bits": int(capacity_bits),
        "payload_bits_effective": int(needed_bits),
        "payload_bytes_raw": int(len(payload)),
        "usage_percent": round(float(used_pct), 4),
        "distortion": {
            "mse": round(float(m), 6),
            "psnr_db": round(float(p), 4),
            "changed_ratio": round(float(ch), 6),
        },
        "recommendations": [
            "Adaptive embedding enabled: edges-first to reduce visible artifacts.",
            "Prefer PNG carriers for stable LSB embedding (avoid lossy formats).",
            "Keep usage < 30% for minimal distortion and detectability.",
        ]
    }

    if bits_per_channel == 2:
        stats["recommendations"].append("2 bits/channel increases capacity but also distortion and detectability.")
    if redundancy_r > 1:
        stats["recommendations"].append(f"Redundancy R={redundancy_r} increases robustness but reduces capacity.")
    if used_pct > 50:
        stats["recommendations"].append("Warning: High embedding density can be detectable and may degrade quality.")

    return out_path, stats