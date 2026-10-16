"""grow.py: the map grows itself (Phase 31) -- MAP-Elites over the bank's sound space.

    python Tools/presets/grow.py [--generations G (20)] [--batch B (64)] [--cells N (20)] [--seed S]
                                 [--seconds S (14)] [--jobs J (10)] [--out DIR] [--favorites FILE] [--dry]

An archive of N x N cells over the preset map's two axes -- the brightness (the spectral centroid) and the motion (how
much the spectrum changes) --, each cell holding its best sound, its elite (Mouret and Clune, "Illuminating search
spaces by mapping elites", 2015). It begins with the factory bank, placed by its measured descriptors
(work/presets/descriptors.json). Every generation takes B elites at random, mutates each (one to a few knobs moved in
their own scale, now and then crossed with another elite, matrices too), renders and measures the children as the
presets are measured (additiv_presets --evaluate: the same chord, the loudness in LUFS and the four descriptors), and
keeps a child where its cell is empty or it is better than the elite there.

Better means: nearer the player's stars (the favourites in %APPDATA%/Empyrean/player.settings -- the sounds the player
likes), farther from every factory preset (new), and neither silent nor at the edge of clipping. Over the generations
the archive fills the map's empty stretches and refines its crowded ones.

At the end every elite that is not a factory preset is brought to the bank's level (its master gain, as make_presets.py
levels the bank) and written as an .empyrean user preset into Documents/Empyrean/Presets/Grown, named after where it
stands on the map ("Grown Dusky Drifting 7-12"); the plugin lists them with its own presets. The archive is kept in
work/grow/archive.json and a next run goes on from it.
"""
import argparse
import json
import os
import random
import re
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
WORK = ROOT / "work" / "grow"
BIN = ROOT / "bin" / "msvc"

# knobs that are not the sound: the player's, the blends', the headset's, the tuning's identity, the level
FIXED = {"master_gain", "preset_glide", "morph_active", "morph_pos", "map_active", "map_x", "map_y", "map_radius",
         "map_glide", "route_speed", "brain_on", "mpe", "mpe_bend", "mpe_press", "mpe_press_depth", "mpe_slide",
         "mpe_slide_depth", "dream_journal", "vr_hands", "head_tracking", "scale", "root", "keymap", "ref_pitch",
         "partials", "auto_mode"}
BRIGHT = ["Umber", "Dusky", "Shaded", "Mellow", "Clear", "Bright", "Gleaming", "Radiant"]
MOVING = ["Still", "Hushed", "Drifting", "Swaying", "Moving", "Restless", "Teeming", "Turbulent"]


def params():
    """The parameter table as the renderer dumps it: key -> its description (kind, range, skew, default)."""
    out = subprocess.run([str(BIN / "empy_render.exe"), "--dump-params"], capture_output=True, text=True, check=True)
    table = json.loads(out.stdout[out.stdout.index("["):])
    return {p["key"]: p for p in table}


def bank():
    """The factory presets: name -> (knobs as a dict of floats, matrix text)."""
    text = (ROOT / "Core" / "src" / "Presets.inc").read_text(encoding="utf-8")
    found = {}
    for name, group, knobs, mod in re.findall(r'\{ "([^"]+)", (\d+), "([^"]*)", "([^"]*)" \}', text):
        values = {}
        for kv in knobs.split():
            k, _, v = kv.partition("=")
            values[k] = float(v)
        found[name] = (values, mod)
    return found


def favourites(path):
    """The player's stars from player.settings (a JUCE properties file), or none."""
    if path is None:
        base = os.environ.get("APPDATA")
        if not base:
            return []
        path = Path(base) / "Empyrean" / "player.settings"
    try:
        root = ET.parse(path).getroot()
    except (OSError, ET.ParseError):
        return []
    for v in root.iter("VALUE"):
        if v.get("name") == "favorites":
            return [s for s in (v.get("val") or "").splitlines() if s.strip()]
    return []


def documents():
    """The player's Documents folder as Windows knows it (OneDrive may have moved it), else ~/Documents."""
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes

        class GUID(ctypes.Structure):
            _fields_ = [("Data1", wintypes.DWORD), ("Data2", wintypes.WORD), ("Data3", wintypes.WORD),
                        ("Data4", ctypes.c_ubyte * 8)]

        fid = GUID(0xFDD39AD0, 0x238F, 0x46AF, (ctypes.c_ubyte * 8)(0xAD, 0xB4, 0x6C, 0x85, 0x48, 0x03, 0x69, 0xC7))
        p = ctypes.c_wchar_p()
        if ctypes.windll.shell32.SHGetKnownFolderPath(ctypes.byref(fid), 0, None, ctypes.byref(p)) == 0:
            folder = Path(p.value)
            ctypes.windll.ole32.CoTaskMemFree(p)
            return folder
    return Path.home() / "Documents"


class Space:
    """The knobs' own scales: a knob's value to 0 .. 1 and back, its mutation, the distance of two sounds."""

    def __init__(self, table):
        self.table = table
        self.keys = [k for k, p in table.items() if k not in FIXED]

    def norm(self, k, v):
        p = self.table[k]
        lo, hi = p["min"], p["max"]
        if hi <= lo:
            return 0.0
        x = min(max((v - lo) / (hi - lo), 0.0), 1.0)
        return x ** p["skew"] if p["kind"] == "float" else x

    def value(self, k, n):
        p = self.table[k]
        lo, hi = p["min"], p["max"]
        n = min(max(n, 0.0), 1.0)
        if p["kind"] == "float":
            return lo + (hi - lo) * n ** (1.0 / p["skew"])
        return float(round(lo + (hi - lo) * n))

    def full(self, values):
        """Every knob: the sound's, the defaults for the rest."""
        return {k: values.get(k, p["default"]) for k, p in self.table.items()}

    def mutate(self, rng, values, count):
        """count knobs moved: a float by a normal step in its scale, a whole number or a choice to another value, a
        switch turned."""
        child = dict(values)
        for k in rng.sample(self.keys, count):
            p = self.table[k]
            if p["kind"] == "bool":
                child[k] = 1.0 - round(child[k])
            elif p["kind"] == "choice":
                child[k] = float(rng.randrange(int(p["min"]), int(p["max"]) + 1))
            else:
                child[k] = self.value(k, self.norm(k, child[k]) + rng.gauss(0.0, 0.12))
        return child

    def distance(self, a, b):
        """The mean distance of two sounds' knobs, each in its own scale."""
        return sum(abs(self.norm(k, a[k]) - self.norm(k, b[k])) for k in self.keys) / len(self.keys)


def evaluate(lines, seconds, jobs):
    """Renders and measures sounds (a line each: knobs '|' matrix) -> per line (lufs, b, m, w, n) or None."""
    WORK.mkdir(parents=True, exist_ok=True)
    batch, result = WORK / "batch.txt", WORK / "batch.json"
    batch.write_text("\n".join(lines) + "\n", encoding="utf-8")
    subprocess.run([str(BIN / "additiv_presets.exe"), "--evaluate", str(batch), "--out", str(result), "--jobs",
                    str(jobs), "--seconds", str(seconds)], check=True, capture_output=True)
    return json.loads(result.read_text(encoding="utf-8"))


def line_of(values, mod):
    """A sound as --evaluate reads it."""
    return " ".join(f"{k}={v:.6g}" for k, v in values.items()) + "|" + mod


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--generations", type=int, default=20)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--cells", type=int, default=20)
    ap.add_argument("--seed", type=int, default=31)
    ap.add_argument("--seconds", type=float, default=14.0)
    ap.add_argument("--jobs", type=int, default=10)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--favorites", type=Path, default=None)
    ap.add_argument("--dry", action="store_true", help="grow, but write no presets")
    a = ap.parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import make_presets   # the bank's level (TARGET_DB) and its names

    rng = random.Random(a.seed)
    space = Space(params())
    factory = bank()
    desc = json.loads((ROOT / "work" / "presets" / "descriptors.json").read_text(encoding="utf-8"))
    names = [n for n in factory if n in desc]
    # the axes from the bank: its 1st to 99th percentile of brightness and motion
    bs = sorted(desc[n][0] for n in names)
    ms = sorted(desc[n][1] for n in names)
    lo_b, hi_b = bs[len(bs) // 100], bs[-1 - len(bs) // 100]
    lo_m, hi_m = ms[len(ms) // 100], ms[-1 - len(ms) // 100]
    mean = [sum(desc[n][i] for n in names) / len(names) for i in range(4)]
    sd = [max(1e-6, (sum((desc[n][i] - mean[i]) ** 2 for n in names) / len(names)) ** 0.5) for i in range(4)]

    def cell(d):
        i = min(max(int((d[0] - lo_b) / (hi_b - lo_b) * a.cells), 0), a.cells - 1)
        j = min(max(int((d[1] - lo_m) / (hi_m - lo_m) * a.cells), 0), a.cells - 1)
        return i, j

    stars = [space.full(factory[n][0]) for n in favourites(a.favorites) if n in factory]

    spread = np.array(sd)
    placed = np.array([desc[n][:4] for n in names]) / spread

    def nearest(d, itself=-1):
        """How far a sound stands from the nearest factory preset (the one at index itself left out), in the
        descriptors' spreads."""
        dist = np.sqrt(((placed - np.array(d[:4]) / spread) ** 2).sum(axis=1))
        if itself >= 0:
            dist[itself] = np.inf
        return float(dist.min())

    def fitness(values, lufs, near):
        """Higher is better: near the stars, new (a factory preset measured against the others, so a child has to be
        newer than the elite it would replace, not merely different), neither silent nor clipping."""
        f = 0.0
        if stars:
            f -= 3.0 * min(space.distance(values, s) for s in stars)
        f += 0.5 * min(near, 2.0)
        if lufs < -50.0:
            f -= 10.0
        elif lufs > -3.0:
            f -= 2.0
        return f

    archive_file = WORK / "archive.json"
    archive = {}
    if archive_file.exists():
        for key, e in json.loads(archive_file.read_text(encoding="utf-8")).items():
            archive[tuple(int(x) for x in key.split(","))] = e
        print(f"going on from {len(archive)} elites")
    else:
        for k, n in enumerate(names):
            values = space.full(factory[n][0])
            d = desc[n]
            key = cell(d)
            f = fitness(values, d[4] if len(d) > 4 else -20.0, nearest(d, k))
            if key not in archive or f > archive[key]["fitness"]:
                archive[key] = {"fitness": f, "values": values, "mod": factory[n][1], "lufs": None, "desc": d[:4],
                                "from": n}
        print(f"{len(archive)} of {a.cells * a.cells} cells held by the bank, {len(stars)} stars")

    for g in range(a.generations):
        keys = list(archive)
        children = []
        for _ in range(a.batch):
            p = archive[rng.choice(keys)]
            values, mod = dict(p["values"]), p["mod"]
            if rng.random() < 0.25:   # crossed with another elite, knob by knob (and maybe its matrix)
                q = archive[rng.choice(keys)]
                values = {k: (v if rng.random() < 0.5 else q["values"][k]) for k, v in values.items()}
                if rng.random() < 0.5:
                    mod = q["mod"]
            count = 1
            while count < 6 and rng.random() < 0.5:
                count += 1
            children.append((space.mutate(rng, values, count), mod))
        results = evaluate([line_of(v, m) for v, m in children], a.seconds, a.jobs)
        kept = 0
        for (values, mod), r in zip(children, results):
            if r is None:
                continue
            lufs, d = r[0], r[1:5]
            key = cell(d)
            f = fitness(values, lufs, nearest(d))
            if key not in archive or f > archive[key]["fitness"]:
                archive[key] = {"fitness": f, "values": values, "mod": mod, "lufs": lufs, "desc": d, "from": "grown"}
                kept += 1
        grown = sum(1 for e in archive.values() if e["from"] == "grown")
        print(f"generation {g + 1}/{a.generations}: {kept} of {len(children)} kept; {len(archive)} cells held, "
              f"{grown} grown")
        archive_file.write_text(json.dumps({f"{i},{j}": e for (i, j), e in archive.items()}), encoding="utf-8")

    if a.dry:
        return
    out = a.out or documents() / "Empyrean" / "Presets" / "Grown"
    out.mkdir(parents=True, exist_ok=True)
    written = 0
    for (i, j), e in sorted(archive.items()):
        if e["from"] != "grown" or e["lufs"] is None:
            continue
        values = dict(e["values"])
        values["master_gain"] = round(max(-18.0, min(12.0, make_presets.TARGET_DB - e["lufs"])), 2)
        name = f"Grown {BRIGHT[i * len(BRIGHT) // a.cells]} {MOVING[j * len(MOVING) // a.cells]} {i}-{j}"
        root = ET.Element("EmpyreanPreset", version="grown",
                          values=" ".join(f"{k}={v:.6g}" for k, v in values.items()), mod=e["mod"])
        ET.ElementTree(root).write(out / f"{name}.empyrean", encoding="UTF-8", xml_declaration=True)
        written += 1
    print(f"{written} grown presets -> {out}")


if __name__ == "__main__":
    main()
