import os
from typing import List, Dict, Tuple

from utils.crypto import pack_payload_batch, password_strength_label
from utils.risk import risk_score_encoding
from detector.image_encoder import encode_image_lsb
from detector.audio_encoder import encode_wav_lsb

def estimate_required_bits(packed_payload_bytes: int, redundancy_r: int) -> int:
    # payload bytes -> bits; redundancy multiplies bits
    return int(packed_payload_bytes * 8 * max(1, int(redundancy_r)))

def advise_min_image_dims(required_bits: int, bits_per_channel: int) -> Tuple[int, int]:
    # capacity_bits = w*h*3*bpc -> choose near-square
    import math
    denom = 3 * max(1, int(bits_per_channel))
    pixels = math.ceil(required_bits / max(1, denom))
    side = math.ceil(math.sqrt(pixels))
    return int(side), int(side)

def advise_min_wav_duration(required_bits: int, framerate: int = 44100, channels: int = 2) -> float:
    # capacity_bits = frames * channels (1 bit per sample)
    # frames = duration * framerate
    denom = max(1, int(framerate) * max(1, int(channels)))
    return float(required_bits / denom)

def encode_message_batch_to_carrier(
    carrier_path: str,
    records: List[Dict[str, str]],
    password: str,
    bits_per_unit: int,
    allow_compress: bool,
    adaptive_embedding: bool,
    redundancy_r: int,
):
    """
    Encoding-only:
    - Packs multiple records into one container
    - Optional compression, optional encryption+HMAC
    - Tamper-evident watermark_id embedded in payload header
    - Redundancy repeats each bit R times (majority-vote-ready)
    - Adaptive embedding for images: edges first
    """
    ext = os.path.splitext(carrier_path)[1].lower().strip(".")
    if ext not in ("png", "bmp", "wav"):
        raise ValueError("Unsupported carrier type. Use PNG/BMP/WAV.")

    media_type = "audio" if ext == "wav" else "image"

    encode_meta = {
        "media_type": media_type,
        "bits_per_unit": int(bits_per_unit),
        "redundancy_r": int(redundancy_r),
        "adaptive_embedding": bool(adaptive_embedding) if media_type == "image" else False,
        "allow_compress": bool(allow_compress),
    }

    pack = pack_payload_batch(
        records=records,
        password=password,
        original_filename="batch.txt",
        allow_compress=allow_compress,
        encode_meta=encode_meta
    )

    # Embed depending on carrier
    if media_type == "image":
        if bits_per_unit not in (1, 2):
            raise ValueError("Image bits-per-channel must be 1 or 2.")
        out_path, embed_stats = encode_image_lsb(
            carrier_path=carrier_path,
            payload=pack.packed,
            bits_per_channel=bits_per_unit,
            adaptive=adaptive_embedding,
            redundancy_r=redundancy_r
        )
    else:
        # WAV safe default: 1 bit/sample
        if bits_per_unit != 1:
            raise ValueError("For WAV, bits-per-unit must be 1 (safe default).")
        out_path, embed_stats = encode_wav_lsb(
            carrier_path=carrier_path,
            payload=pack.packed,
            redundancy_r=redundancy_r
        )

    usage_percent = float(embed_stats.get("usage_percent", 0.0))
    encrypted = bool(pack.encrypted)
    compressed = bool(pack.compressed)
    pw_strength = password_strength_label(password)

    risk_score = risk_score_encoding(
        usage_percent=usage_percent,
        bits_per_unit=bits_per_unit,
        encrypted=encrypted,
        has_password=bool(password),
        compressed=compressed,
        password_strength=pw_strength
    )

    findings = {
        "mode": "encode",
        "media_type": media_type,
        "batch": {
            "record_count": int(len(records)),
            "labels": [r.get("label", "message") for r in records],
        },
        "payload": {
            "magic": "STEGTK2",
            "version": 3,
            "plaintext_container": "RECv1!",
            "plaintext_len": pack.plaintext_len,
            "body_len": pack.body_len,
            "compressed": compressed,
            "encrypted": encrypted,
            "password_strength": pw_strength,
            "crc32": pack.crc32_hex,
            "hmac_sha256": pack.hmac_hex,
            "packed_total_bytes": len(pack.packed),
            "watermark_id": pack.watermark_id,
        },
        "encoding_options": {
            "bits_per_unit": int(bits_per_unit),
            "adaptive_embedding": bool(adaptive_embedding) if media_type == "image" else False,
            "redundancy_r": int(redundancy_r),
            "allow_compress": bool(allow_compress),
        },
        "embedding": embed_stats,
        "risk_score": int(risk_score),
    }
    return out_path, findings, pack