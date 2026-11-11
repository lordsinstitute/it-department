from PIL import Image
import os

MAX_BYTES = 50 * 1024  # 50 KB

def extract_lsb_payload(image_path):
    img = Image.open(image_path)
    pixels = img.getdata()

    bits = []
    for px in pixels:
        for c in px[:3]:
            bits.append(c & 1)

    bytes_out = []
    for i in range(0, len(bits), 8):
        byte = bits[i:i+8]
        if len(byte) < 8:
            break
        value = int("".join(map(str, byte)), 2)
        bytes_out.append(value)
        if len(bytes_out) >= MAX_BYTES:
            break

    payload = bytes(bytes_out)

    os.makedirs("extracted", exist_ok=True)
    out = f"extracted/payload_{os.path.basename(image_path)}.bin"
    with open(out, "wb") as f:
        f.write(payload)

    return out