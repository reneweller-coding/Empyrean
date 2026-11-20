"""night_plot.py: a night of the night director (Core/include/additiv/NightDirector.h) to look at -- the table and a
picture of the curve against what was chosen.

Runs additiv_night (no card, no sound) and prints, per step: when it begins, the preset and its group, how long it
stays, its stillness and brightness as percentiles of the candidates and the curve's targets there, and its distance in
sound from the step before; then the night's numbers: how closely the stillness follows the curve (Spearman), the mean
step between neighbours against the shipped Whole Nights'. With --png it draws the two curves (stillness above,
brightness below) and the presets chosen as a step line over the hours, the phases marked. The picture needs Pillow
(no matplotlib: none of the venvs on this machine has it); G:\\Tools\\VRAudio\\AmbientSynth\\Tools\\TextureGen\\.venv has
Pillow.

    python Tools/presets/night_plot.py [--hours 8] [--seed 1] [--family N|NAME|bank] [--tsv FILE] [--png FILE]
"""
import argparse
import math
import os
import subprocess
import sys

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
EXES = [os.path.join(ROOT, "bin", "msvc", "additiv_night.exe"),
        os.path.join(ROOT, "build", "msvc", "Release", "additiv_night.exe")]


def clock(seconds, start_h):
    """Seconds into the night as the clock shows them, from start_h o'clock."""
    m = int(round(seconds / 60.0)) + int(round(start_h * 60))
    return "%d:%02d" % ((m // 60) % 24, m % 60)


def spearman(a, b):
    def ranks(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        for k, i in enumerate(order):
            r[i] = float(k)
        return r
    ra, rb = ranks(a), ranks(b)
    m = (len(a) - 1) / 2.0
    num = sum((x - m) * (y - m) for x, y in zip(ra, rb))
    den = math.sqrt(sum((x - m) ** 2 for x in ra) * sum((y - m) ** 2 for y in rb))
    return num / den if den else 0.0


def read(exe, a):
    cmd = [exe, "--hours", str(a.hours), "--seed", str(a.seed)]
    if a.family:
        cmd += ["--family", a.family]
    out = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout
    head, steps, curve, whole = {}, [], [], None
    cols = None
    for line in out.splitlines():
        if line.startswith("# night "):
            parts = line[8:].split(" ")
            head = {"hours": float(parts[0]), "seed": parts[1], "family": " ".join(parts[2:-2]),
                    "candidates": int(parts[-2]), "scale": float(parts[-1])}
        elif line.startswith("# phases "):
            p = line[9:].split()
            head.update(asleep=float(p[0]), morning=float(p[1]), waves=int(p[2]))
        elif line.startswith("# whole_nights "):
            whole = float(line.split()[2])
        elif line.startswith("# curve "):
            curve.append(tuple(float(x) for x in line.split()[2:5]))
        elif line.startswith("step\t"):
            cols = line.split("\t")
        elif cols and line.strip():
            row = dict(zip(cols, line.split("\t")))
            for k in ("at", "stay", "still", "bright", "target_still", "target_bright", "distance", "motion_z", "bright_z"):
                row[k] = float(row[k])
            steps.append(row)
    return head, steps, curve, whole


def table(head, steps, whole, start_h):
    lines = ["Night of %g h (seed %s, %s): %d presets of %d candidates; falling asleep %d min, %d waves, morning %d min" % (
        head["hours"], head["seed"], head["family"], len(steps), head["candidates"],
        round(head["asleep"] * head["hours"] * 60), head["waves"], round(head["morning"] * head["hours"] * 60)), "",
        "%-6s %-34s %-22s %6s  %-14s %-14s %5s" % ("time", "preset", "group", "stay", "stillness/goal", "bright/goal",
                                                   "step"),
        "-" * 112]
    for s in steps:
        lines.append("%-6s %-34s %-22s %6s  %4.0f %% / %3.0f %%  %4.0f %% / %3.0f %%  %5.2f" % (
            clock(s["at"], start_h), s["preset"][:34], s["group"][:22], "%d:%02d" % (s["stay"] // 60, s["stay"] % 60),
            100 * s["still"], 100 * s["target_still"], 100 * s["bright"], 100 * s["target_bright"], s["distance"]))
    rho = spearman([s["still"] for s in steps], [s["target_still"] for s in steps])
    rho_b = spearman([s["bright"] for s in steps], [s["target_bright"] for s in steps])
    rho_m = spearman([s["motion_z"] for s in steps], [s["target_still"] for s in steps])
    mean = sum(s["distance"] for s in steps[1:]) / max(1, len(steps) - 1)
    lines += ["", "stillness follows the curve: Spearman %.2f (brightness %.2f; the measured motion alone %.2f)" % (
        rho, rho_b, rho_m),
        "mean step between neighbours %.2f spreads (the shipped Whole Nights %.2f)" % (mean, whole or 0.0),
        "stillness and brightness are percentiles of the candidates: 0 % the stillest / darkest, 100 % the most moving "
        "/ brightest"]
    return "\n".join(lines)


def plot(head, steps, curve, path, start_h):
    from PIL import Image, ImageDraw
    W, H, L, R, T, B = 1400, 760, 70, 30, 40, 50
    img = Image.new("RGB", (W, H), (16, 20, 30))
    d = ImageDraw.Draw(img)
    total = head["hours"] * 3600.0
    panel_h = (H - T - B - 40) // 2
    ink, dim, goal = (220, 226, 236), (110, 120, 140), (240, 200, 120)
    colours = [(120, 200, 230), (190, 150, 240)]
    d.text((L, 10), "Night director: %g h, seed %s, %s -- %d presets (curve: dashed, chosen: solid)" % (
        head["hours"], head["seed"], head["family"], len(steps)), fill=ink)
    for p, (key, name) in enumerate([("still", "stillness"), ("bright", "brightness")]):
        top = T + p * (panel_h + 40)
        x0, x1, y0, y1 = L, W - R, top, top + panel_h
        d.rectangle([x0, y0, x1, y1], outline=dim)
        d.text((8, y0 + panel_h // 2 - 6), name, fill=ink)
        for q in (0.0, 0.25, 0.5, 0.75, 1.0):
            y = y1 - q * (y1 - y0)
            d.line([x0, y, x1, y], fill=(40, 46, 60))
            d.text((x0 - 34, y - 6), "%3d%%" % (q * 100), fill=dim)
        for xph in (head["asleep"], 1.0 - head["morning"]):
            x = x0 + xph * (x1 - x0)
            d.line([x, y0, x, y1], fill=(80, 90, 110))
        hours = int(head["hours"])
        for h in range(hours + 1):
            x = x0 + (h * 3600.0 / total) * (x1 - x0)
            d.line([x, y1, x, y1 + 4], fill=dim)
            if p == 1:
                d.text((x - 14, y1 + 8), clock(h * 3600.0, start_h), fill=dim)
        pts = [(x0 + c[0] * (x1 - x0), y1 - c[1 + p] * (y1 - y0)) for c in curve]
        for i in range(0, len(pts) - 1, 2):   # dashed
            d.line([pts[i], pts[i + 1]], fill=goal, width=2)
        for s in steps:
            xa = x0 + s["at"] / total * (x1 - x0)
            xb = x0 + min(1.0, (s["at"] + s["stay"]) / total) * (x1 - x0)
            y = y1 - s[key] * (y1 - y0)
            d.line([xa, y, xb, y], fill=colours[p], width=3)
    img.save(path)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--hours", type=float, default=8.0)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--family", default=None, help="a family's number or name, or bank (default: all whole-night groups)")
    ap.add_argument("--start", type=float, default=23.0, help="the clock at the night's start, hours (23)")
    ap.add_argument("--exe", default=None)
    ap.add_argument("--tsv", default=None, help="write the table there too")
    ap.add_argument("--png", default=None, help="draw the curves there (Pillow)")
    a = ap.parse_args()
    exe = os.path.abspath(a.exe) if a.exe else next((e for e in EXES if os.path.exists(e)), None)
    if not exe:
        sys.exit("night_plot: no additiv_night (build it, or pass --exe)")
    head, steps, curve, whole = read(exe, a)
    text = table(head, steps, whole, a.start)
    print(text)
    if a.tsv:
        with open(a.tsv, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(text + "\n")
    if a.png:
        try:
            plot(head, steps, curve, a.png, a.start)
            print("picture: %s" % a.png)
        except ImportError:
            print("no Pillow in this Python: no picture (G:\\Tools\\VRAudio\\AmbientSynth\\Tools\\TextureGen\\.venv has it)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
