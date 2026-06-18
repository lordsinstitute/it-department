import os
import math
from PIL import Image, ImageDraw

def block_entropy(block):
    freq = {}
    for p in block:
        freq[p] = freq.get(p, 0) + 1

    entropy = 0
    for count in freq.values():
        p = count / len(block)
        entropy -= p * math.log2(p)
    return entropy

def generate_entropy_heatmap(image_path, block_size=16):
    img = Image.open(image_path).convert("L")
    w, h = img.size
    pixels = img.load()

    heatmap = Image.new("RGB", (w, h))
    draw = ImageDraw.Draw(heatmap)

    for y in range(0, h, block_size):
        for x in range(0, w, block_size):
            block = [
                pixels[i, j]
                for i in range(x, min(x + block_size, w))
                for j in range(y, min(y + block_size, h))
            ]

            ent = block_entropy(block)
            intensity = int(min(255, ent * 32))
            draw.rectangle(
                [x, y, x + block_size, y + block_size],
                fill=(intensity, 0, 255 - intensity)
            )

    os.makedirs("static/heatmaps", exist_ok=True)
    out = f"static/heatmaps/heatmap_{os.path.basename(image_path)}.png"
    heatmap.save(out)
    return out