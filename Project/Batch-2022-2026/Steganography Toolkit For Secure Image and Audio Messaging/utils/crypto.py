import os
import zlib
import hmac
import hashlib
import binascii
from dataclasses import dataclass
from typing import List, Dict, Tuple

MAGIC = b"STEGTK2"
VERSION = 3  # bumped due to batching + watermark metadata

@dataclass
class PackResult:
    packed: bytes
    compressed: bool
    encrypted: bool
    plaintext_len: int
    body_len: int
    hmac_hex: str
    crc32_hex: str
    watermark_id: str  # new: embedded in header

def _pbkdf2(password: str, salt: bytes, length: int = 64) -> bytes:
    return hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 150_000, dklen=length)

def _keystream(key: bytes, nonce: bytes, n: int) -> bytes:
    out = bytearray()
    counter = 0
    while len(out) < n:
        block = hashlib.sha256(key + nonce + counter.to_bytes(4, "big")).digest()
        out.extend(block)
        counter += 1
    return bytes(out[:n])

def _xor_stream(data: bytes, key: bytes, nonce: bytes) -> bytes:
    ks = _keystream(key, nonce, len(data))
    return bytes([a ^ b for a, b in zip(data, ks)])

def password_strength_label(pw: str) -> str:
    if not pw:
        return "none"
    score = 0
    if len(pw) >= 12: score += 2
    elif len(pw) >= 8: score += 1
    if any(c.islower() for c in pw): score += 1
    if any(c.isupper() for c in pw): score += 1
    if any(c.isdigit() for c in pw): score += 1
    if any(not c.isalnum() for c in pw): score += 1
    if score <= 2:
        return "weak"
    if score <= 4:
        return "medium"
    return "strong"

def _u16(n: int) -> bytes:
    return int(n).to_bytes(2, "big")

def _u32(n: int) -> bytes:
    return int(n).to_bytes(4, "big")

def _pack_records(records: List[Dict[str, str]]) -> bytes:
    """
    Container format (plaintext, before compress/encrypt):
      REC_MAGIC(6) = b"RECv1!"
      COUNT(2)
      for each record:
        LABEL_LEN(2) + LABEL(utf-8)
        TYPE_LEN(2) + TYPE(utf-8)
        DATA_LEN(4) + DATA(bytes)
    """
    rec_magic = b"RECv1!"
    out = bytearray()
    out.extend(rec_magic)
    out.extend(_u16(len(records)))

    for r in records:
        label = (r.get("label") or "message").encode("utf-8", errors="replace")[:200]
        ctype = (r.get("content_type") or "text/plain; charset=utf-8").encode("utf-8", errors="replace")[:200]
        data = (r.get("data") or "").encode("utf-8")

        out.extend(_u16(len(label))); out.extend(label)
        out.extend(_u16(len(ctype))); out.extend(ctype)
        out.extend(_u32(len(data))); out.extend(data)

    return bytes(out)

def compute_watermark_id(payload_plain: bytes, encode_meta: Dict) -> str:
    """
    Tamper-evident watermark:
    watermark_id = SHA256( payload_plain + canonical_encode_meta_bytes )
    This is embedded in header and also stored in ledger evidence.
    """
    # canonical-ish meta ordering (stable keys)
    keys = ["media_type", "bits_per_unit", "redundancy_r", "adaptive_embedding", "allow_compress", "ts"]
    meta_bytes = bytearray()
    for k in keys:
        v = encode_meta.get(k, "")
        meta_bytes.extend(str(k).encode("utf-8") + b"=" + str(v).encode("utf-8") + b";")
    return hashlib.sha256(payload_plain + bytes(meta_bytes)).hexdigest()

def pack_payload_batch(
    records: List[Dict[str, str]],
    password: str,
    original_filename: str,
    allow_compress: bool,
    encode_meta: Dict
) -> PackResult:
    """
    Payload layout (binary):
      TOTLEN(4) excluding TOTLEN itself
      MAGIC(7) + VERSION(1)
      FLAGS(1): bit0=compressed, bit1=encrypted
      FNLEN(1) + FNAME(variable)
      TS(8)
      PTLEN(4) plaintext length
      BDLEN(4) body length
      WATERMARK_ID(32 bytes raw)  <-- new, fixed-length
      SALT(16) + NONCE(16)
      BODY(variable)
      CRC32(4) of plaintext
      HMAC(32) over header_without_hmac + body + crc32
    """
    if records is None or len(records) == 0:
        records = [{"label": "message", "content_type": "text/plain; charset=utf-8", "data": ""}]

    plaintext = _pack_records(records)

    fn = (original_filename or "batch.txt").encode("utf-8", errors="replace")[:200]
    fnlen = len(fn)

    ts = int(__import__("time").time())
    encode_meta = dict(encode_meta or {})
    encode_meta["ts"] = ts

    ptlen = len(plaintext)
    crc32 = binascii.crc32(plaintext) & 0xFFFFFFFF
    crc32_bytes = crc32.to_bytes(4, "big")
    crc32_hex = f"{crc32:08x}"

    compressed = False
    body = plaintext
    if allow_compress and ptlen > 64:
        cand = zlib.compress(plaintext, level=9)
        if len(cand) < len(plaintext):
            body = cand
            compressed = True
    body_len = len(body)

    encrypted = bool(password)
    salt = os.urandom(16)
    nonce = os.urandom(16)

    # watermark derived from PLAINTEXT container + encode metadata
    watermark_id_hex = compute_watermark_id(plaintext, encode_meta)
    watermark_raw = bytes.fromhex(watermark_id_hex)  # 32 bytes

    if encrypted:
        k = _pbkdf2(password, salt, 64)
        enc_key = k[:32]
        mac_key = k[32:]
        body = _xor_stream(body, enc_key, nonce)
    else:
        mac_key = b"\x00" * 32

    flags = 0
    if compressed: flags |= 1
    if encrypted:  flags |= 2

    header = bytearray()
    header.extend(MAGIC)
    header.extend(bytes([VERSION]))
    header.extend(bytes([flags]))
    header.extend(bytes([fnlen]))
    header.extend(fn)
    header.extend(ts.to_bytes(8, "big"))
    header.extend(ptlen.to_bytes(4, "big"))
    header.extend(body_len.to_bytes(4, "big"))
    header.extend(watermark_raw)
    header.extend(salt)
    header.extend(nonce)

    mac = hmac.new(mac_key, bytes(header) + body + crc32_bytes, hashlib.sha256).digest()
    hmac_hex = mac.hex()

    blob = bytes(header) + body + crc32_bytes + mac
    totlen = len(blob).to_bytes(4, "big")
    packed = totlen + blob

    return PackResult(
        packed=packed,
        compressed=compressed,
        encrypted=encrypted,
        plaintext_len=ptlen,
        body_len=body_len,
        hmac_hex=hmac_hex,
        crc32_hex=crc32_hex,
        watermark_id=watermark_id_hex
    )