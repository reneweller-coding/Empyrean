"""The icon: the panel's logo (Plugin/Theme.cpp, drawLogo) -- twenty-four partials round a ring, their sizes from a
fixed die, lit from the top, green turning violet, a small light in the centre -- on the night's blue.

    python Deploy/make_icon.py      -> Deploy/additiv.ico (16 .. 256 px), additiv.png (256), additiv_512.png

The .ico stays committed so the release build needs no Python.
"""
import math
from pathlib import Path
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent
BACK = (7, 9, 15, 255)
GREEN = (127, 224, 196)
VIOLET = (182, 156, 240)


def mix(a, b, t):
    """The colour between a and b."""
    return tuple(int(round(a[i] + (b[i] - a[i]) * t)) for i in range(3))


def draw(size):
    """The logo at one size, drawn four times larger and scaled down (smooth edges)."""
    s = size * 4
    img = Image.new("RGBA", (s, s), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.rounded_rectangle([0, 0, s - 1, s - 1], radius=s * 0.18, fill=BACK)
    c = s / 2.0
    radius = 0.36 * s
    count = 24 if size >= 48 else 12   # small icons: fewer, larger partials
    for i in range(count):
        a = 2.0 * math.pi * i / count
        k = (i * (24 // count) * 2654435761) & 0xFFFFFFFF
        k ^= k >> 15
        dot = radius * (0.08 + 0.12 * (k % 1000) / 1000.0) * (1.6 if size < 48 else 1.0)
        lit = 0.35 + 0.65 * 0.5 * (1.0 + math.cos(a))
        col = mix(GREEN, VIOLET, 0.5 * (1.0 - math.cos(a)))
        x, y = c + math.sin(a) * radius, c - math.cos(a) * radius
        d.ellipse([x - dot / 2, y - dot / 2, x + dot / 2, y + dot / 2], fill=col + (int(255 * lit),))
    core = radius * 0.22 * (1.5 if size < 48 else 1.0)
    d.ellipse([c - core / 2, c - core / 2, c + core / 2, c + core / 2], fill=GREEN + (205,))
    return img.resize((size, size), Image.LANCZOS)


def main():
    sizes = [16, 24, 32, 48, 64, 128, 256]
    images = [draw(n) for n in sizes]
    images[-1].save(ROOT / "additiv.ico", sizes=[(n, n) for n in sizes], append_images=images[:-1])
    images[-1].save(ROOT / "additiv.png")
    draw(512).save(ROOT / "additiv_512.png")
    print("Deploy/additiv.ico, additiv.png, additiv_512.png")


if __name__ == "__main__":
    main()
