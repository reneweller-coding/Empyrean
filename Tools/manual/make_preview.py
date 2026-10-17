"""Empyrean -- the picture GitHub shows when the project is linked: docs/social-preview.png (1280 x 640).

    python Tools/manual/make_preview.py

The logo, the name, what it is in a few lines, and the Cloud page (docs/screenshot.png, from the release build). GitHub
takes it by hand: Settings, Social preview. After Totality's make_preview.py.
"""
import os
from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))
W, H = 1280, 640


def font(name, size):
    try:
        return ImageFont.truetype(os.path.join("C:/Windows/Fonts", name), size)
    except OSError:
        return ImageFont.load_default()


img = Image.new("RGB", (W, H), (7, 9, 15))
d = ImageDraw.Draw(img)
shot = Image.open(os.path.join(ROOT, "docs", "screenshot.png")).convert("RGB")
sw, sh = 760, int(760 * shot.height / shot.width)
if sh > 520:
    sh, sw = 520, int(520 * shot.width / shot.height)
crop = shot.resize((sw, sh), Image.LANCZOS)
x0, y0 = W - sw - 30, (H - sh) // 2
img.paste(crop, (x0, y0))
d.rectangle([x0 - 1, y0 - 1, x0 + sw, y0 + sh], outline=(40, 49, 73), width=2)
logo = Image.open(os.path.join(ROOT, "Deploy", "empyrean_512.png")).convert("RGBA").resize((140, 140), Image.LANCZOS)
img.paste(logo, (56, 64), logo)
d.text((56, 222), "EMPYREAN", font=font("segoeuib.ttf", 50), fill=(127, 224, 196))
y = 296
for line in ("An additive ambient synth on", "the GPU: every voice twenty", "thousand sine waves, a", "conductor that plays all night."):
    d.text((58, y), line, font=font("segoeui.ttf", 25), fill=(226, 236, 242))
    y += 35
for line in ("VST3 and standalone for Windows", "with an NVIDIA GeForce RTX card"):
    d.text((58, y + 18), line, font=font("segoeui.ttf", 19), fill=(135, 148, 168))
    y += 26
out = os.path.join(ROOT, "docs", "social-preview.png")
img.save(out)
print("wrote", os.path.relpath(out, ROOT), img.size)
