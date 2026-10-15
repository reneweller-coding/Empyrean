"""The icon: the panel's logo (Plugin/Theme.cpp, drawLogo) -- the overtone ladder, the partials of one tone as lights on
stalks, their heights falling as the harmonic series does, mint turning violet, in a soft glow -- on the night's blue.

    python Deploy/make_icon.py      -> Deploy/empyrean.ico (16 .. 256 px), empyrean.png (256), empyrean_512.png

The .ico stays committed so the release build needs no Python.
"""
from pathlib import Path
from PIL import Image, ImageDraw, ImageFilter

ROOT = Path(__file__).resolve().parent
BACK = (7, 9, 15, 255)
GREEN = (127, 224, 196)
VIOLET = (182, 156, 240)


def mix(a, b, t):
    """The colour between a and b."""
    return tuple(int(round(a[i] + (b[i] - a[i]) * t)) for i in range(3))


def ladder(s, count):
    """The ladder on a transparent square of s pixels: count stalks (small icons get fewer, broader ones)."""
    img = Image.new("RGBA", (s, s), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    base, broad = 0.76 * s, (10.0 / count) ** 0.6
    for i in range(count):
        t = i / (count - 1)
        x = (0.2 + 0.6 * t) * s
        top = base - 0.52 * s / (i + 1) ** 0.55
        col = mix(GREEN, VIOLET, t)
        d.line([(x, base), (x, top)], fill=col + (150,), width=max(1, int(0.018 * s * broad)))
        r = (0.05 - 0.022 * t) * s * broad
        d.ellipse([x - r, top - r, x + r, top + r], fill=col + (255,))
    return img


def glow(layer, s):
    """The layer with a soft glow under it, as the panel draws its lights."""
    blur = layer.filter(ImageFilter.GaussianBlur(s * 0.022))
    r, g, b, a = blur.split()
    blur = Image.merge("RGBA", (r, g, b, a.point(lambda v: min(255, int(v * 1.4)))))
    out = Image.new("RGBA", layer.size, (0, 0, 0, 0))
    out.alpha_composite(blur)
    out.alpha_composite(layer)
    return out


def draw(size):
    """The icon at one size, drawn four times larger and scaled down (smooth edges)."""
    s = size * 4
    img = Image.new("RGBA", (s, s), (0, 0, 0, 0))
    ImageDraw.Draw(img).rounded_rectangle([0, 0, s - 1, s - 1], radius=s * 0.18, fill=BACK)
    count = 10 if size >= 48 else 6 if size >= 24 else 5
    img.alpha_composite(glow(ladder(s, count), s))
    return img.resize((size, size), Image.LANCZOS)


def main():
    sizes = [16, 24, 32, 48, 64, 128, 256]
    images = [draw(n) for n in sizes]
    images[-1].save(ROOT / "empyrean.ico", sizes=[(n, n) for n in sizes], append_images=images[:-1])
    images[-1].save(ROOT / "empyrean.png")
    draw(512).save(ROOT / "empyrean_512.png")
    print("Deploy/empyrean.ico, empyrean.png, empyrean_512.png")


if __name__ == "__main__":
    main()
