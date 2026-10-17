"""Empyrean -- the signal flow as a picture: docs/flow.png, for the manual and the project's front page.

    python Tools/manual/make_flow.py

After Totality's Tools/manual/make_flow.py (01.10.2026): every unit a box in the colour of its family (the panel's,
Plugin/Theme.cpp: the aurora's green the sources, gold the filter, violet the envelopes and modulation, blue what
moves by itself, ice blue the rooms and the mix), the groups as frames, the buses as arrows. Drawn with Pillow at twice
the size it is shown at, so it stays sharp on a screen and in the PDF. What it shows is Engine::renderBlock's signal
flow (Core/src/Engine.cpp, its steps 1 to 8) and the CUDA graph of Core/src/Cloud.cu.
"""
import math
import os
from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))
S = 2                                   # drawn at twice the size
W, H = 1400, 850                        # in the size it is shown at

BG = (7, 9, 15)
BOX = (17, 23, 37)
INK = (226, 236, 242)
DIM = (135, 148, 168)
SOURCE, FILTER, ENVELOPE, MOTION, SPACE = (127, 224, 196), (224, 179, 106), (182, 156, 240), (111, 180, 255), (159, 208, 230)
ROOT_GOLD = (232, 181, 106)


def font(size, bold=False):
    names = ["segoeuib.ttf" if bold else "segoeui.ttf", "arialbd.ttf" if bold else "arial.ttf",
             "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"]
    for n in names:
        for folder in ("C:/Windows/Fonts", "/usr/share/fonts/truetype/dejavu"):
            p = os.path.join(folder, n)
            if os.path.exists(p):
                return ImageFont.truetype(p, size * S)
    return ImageFont.load_default()


img = Image.new("RGB", (W * S, H * S), BG)
d = ImageDraw.Draw(img)
F_BOX, F_SMALL, F_TITLE, F_NOTE = font(13), font(11), font(13, True), font(12)


def wrap(text, f, width):
    lines = []
    for para in text.split("\n"):
        line = ""
        for word in para.split(" "):
            test = (line + " " + word).strip()
            if d.textlength(test, font=f) <= width * S:
                line = test
            else:
                lines.append(line)
                line = word
        lines.append(line)
    return lines


def box(x, y, w, h, text, colour, f=None):
    f = f or F_BOX
    d.rounded_rectangle([x * S, y * S, (x + w) * S, (y + h) * S], radius=7 * S, fill=BOX, outline=colour, width=2 * S)
    lines = wrap(text, f, w - 16)
    lh = f.size * 1.3
    ty = y * S + (h * S - lh * len(lines)) / 2
    for ln in lines:
        tw = d.textlength(ln, font=f)
        d.text((x * S + (w * S - tw) / 2, ty), ln, font=f, fill=INK)
        ty += lh


def group(x, y, w, h, title, colour):
    d.rounded_rectangle([x * S, y * S, (x + w) * S, (y + h) * S], radius=10 * S, outline=colour, width=2 * S)
    d.text(((x + 14) * S, (y + 8) * S), title, font=F_TITLE, fill=colour)


def arrow(points, colour, dashed=False, head=True):
    pts = [(px * S, py * S) for px, py in points]
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        if dashed:
            n = max(1, int(((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5 / (12 * S)))
            for i in range(0, n, 2):
                a, b = i / n, min(1.0, (i + 1) / n)
                d.line([(x0 + (x1 - x0) * a, y0 + (y1 - y0) * a), (x0 + (x1 - x0) * b, y0 + (y1 - y0) * b)],
                       fill=colour, width=2 * S)
        else:
            d.line([(x0, y0), (x1, y1)], fill=colour, width=2 * S)
    if not head:
        return
    (x0, y0), (x1, y1) = pts[-2], pts[-1]
    ang = math.atan2(y1 - y0, x1 - x0)
    size = 8 * S
    d.polygon([(x1, y1), (x1 - size * math.cos(ang - 0.45), y1 - size * math.sin(ang - 0.45)),
               (x1 - size * math.cos(ang + 0.45), y1 - size * math.sin(ang + 0.45))], fill=colour)


def note(x, y, text, colour=DIM, f=None):
    d.text((x * S, y * S), text, font=f or F_NOTE, fill=colour)


# What plays it (Engine::renderBlock, steps 1 and 2; the modulation and the sound).
ins = [("The conductor: Free or Chords, a chord in just intonation that changes one voice at a time; Step; Brain 2 "
        "deep in the background", MOTION),
       ("Played notes, each on its exact sample, and MPE: every note its own bend, pressure and slide", ROOT_GOLD),
       ("Modulation: LFOs, envelopes, macros, the Beat, what the instrument hears, a reservoir of 2048 neurons -- "
        "sixteen rows", ENVELOPE),
       ("The sound: 16384 presets on a measured map; glides, morphs and journeys from one into the next", MOTION),
       ("The family jam, the headset's pose and hands, the real night where you are (the sun's height)", SPACE)]
x = 20
for t, c in ins:
    box(x, 16, 262, 60, t, c, F_SMALL)
    x += 274

# The card: one CUDA graph a block (Cloud.cu, and the kernels of Core/src/gpu).
group(20, 100, 660, 600, "THE GPU  --  one CUDA graph per block of 128 samples", SOURCE)
box(36, 132, 628, 70, "16 voices x 20,000 partials: every harmonic a cloud of detuned sines, every partial with its "
    "frequency and glide, its loudness, pitch and pan envelopes, its distance -- the floor's voices under the "
    "conductor keeping their depth", SOURCE)
box(36, 212, 628, 52, "The spectrum per partial (tilt, brightness, odd / even, inharmonic, stretch, shift, comb) and "
    "the filter as its exact response at every partial's frequency", FILTER)
box(36, 274, 628, 52, "Motion: Random, Wave, Pendulum, Breathe, Glissando, Aurora -- a landscape read as time against "
    "frequency: painted, recorded, or animated after the Kaleidoscope", MOTION)
box(36, 336, 628, 52, "Fields on every partial: Scale, Swarm, Scatter, Orbit, the slime mould; dispersion, self-tune, "
    "sync, chimeras, swarmalators, flocks", MOTION)
box(36, 398, 628, 52, "Living systems that move it: Lenia and Turing fields, a fluid (lattice Boltzmann), islands, "
    "the ripple, a gong's turbulence, the melt, the dream", ENVELOPE)
box(36, 460, 628, 52, "Worlds of their own: chimes in a simulated wind, rain, leaves and sea, the night chorus, thunder, "
    "a wave room of 250,000 cells, stars and black holes", SOURCE)
box(36, 522, 628, 52, "Render: a thread per sample, 512 partials a chunk, phases in fixed point; the sum in a fixed "
    "order -- near and far planes, binaural round the head when it is on", SOURCE)
box(36, 584, 628, 52, "The modal hall: 65,536 decaying resonances, tuned to the scale, growing a memory of what it heard; "
    "the spectral echo: 500 bands, each a delay line", SPACE)
note(40, 652, "0.10 .. 0.17 ms of GPU time per block of 2.7 ms on an RTX 5090; Windows' REALTIME GPU class")

# The inputs into the card.
arrow([(151, 76), (151, 100)], MOTION)
arrow([(425, 76), (425, 100)], ROOT_GOLD)
arrow([(699, 76), (699, 88), (560, 88), (560, 100)], ENVELOPE, dashed=True)
arrow([(973, 76), (973, 92), (630, 92), (630, 100)], MOTION, dashed=True)

# The real-time path.
group(20, 718, 660, 112, "THE REAL-TIME PATH", ROOT_GOLD)
box(36, 750, 628, 64, "A worker thread (MMCSS Pro Audio) renders ahead into a ring by two host blocks and ten "
    "milliseconds; the audio callback stamps the notes and copies -- the card never waits in it", ROOT_GOLD)

# The near plane, on the CPU.
group(720, 100, 660, 236, "THE NEAR PLANE  --  on the CPU, Noctuary's", SPACE)
box(736, 132, 628, 52, "the voices' near plane + chimes, weather and night + the wreck, the magnetosphere and the "
    "throats (giant vocal tracts)", SOURCE)
box(736, 194, 628, 36, "two echoes in series, the second receding into the far plane  ->  a small reverb", SPACE)
box(736, 240, 628, 36, "early reflections of a room, from the dry voices", SPACE)
box(736, 286, 628, 36, "the spectral echo and the melt from the card, partly into the far plane", SPACE)
arrow([(664, 540), (700, 540), (700, 158), (736, 158)], SOURCE)   # from the render: the near plane

# The far plane.
group(720, 354, 660, 112, "THE FAR PLANE  --  heard only through the far reverb", SPACE)
box(736, 386, 628, 64, "diffusion (the swell)  ->  far reverb: four networks, pre-delay, freeze  ->  Unmask: the hall "
    "steps aside for the near plane, band by band", SPACE)
arrow([(664, 560), (708, 560), (708, 420), (736, 420)], SOURCE)   # and the far plane

# Together.
group(720, 484, 660, 216, "TOGETHER", SPACE)
box(736, 516, 628, 52, "near + far + the modal hall + the bloom + the wave room  ->  a merger's ringdown, the ocean's "
    "deep sound channel", SPACE)
box(736, 578, 628, 52, "mid / side: the bass made mono, the width (binaural: the two ears left alone)  ->  the phantom "
    "layer between the ears", SPACE)
box(736, 640, 628, 44, "Floor's share measured  ->  master  ->  soft clip", SPACE)
arrow([(664, 610), (716, 610), (716, 542), (736, 542)], SPACE)   # the modal hall, into the sum
arrow([(1380, 300), (1392, 300), (1392, 500), (1380, 500)], SPACE)   # the near plane, past the far one
arrow([(1050, 466), (1050, 484)], SPACE)

# Out.
box(736, 718, 628, 64, "Out: stereo -- VST3 and standalone (ASIO); empy_render; Empyrean Journeys into WAV, FLAC, Ogg "
    "Vorbis or MP3; score cues and chapters beside them for the Kaleidoscope", SPACE, F_SMALL)
arrow([(1050, 700), (1050, 718)], SPACE)
note(720, 796, "Deterministic: the card sums in a fixed order, so a render is bit-identical every time.", DIM)
note(720, 816, "Every preset levelled by its loudness (ITU-R BS.1770), as it is played.", DIM)

out = os.path.join(ROOT, "docs", "flow.png")
img.save(out)
print("written", out)
