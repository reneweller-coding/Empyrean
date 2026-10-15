"""make_journeys.py: the journeys Empyrean comes with -- into Core/src/Journeys.inc.

A journey (Core/include/additiv/Journey.h) is presets in a row, each held for a while drawn from a
range and glided into the next; cyclic, so a night plays itself. After Noctuary's
Tools/make_journeys.py, but from the presets' own parameters (there is no measured library here):
every preset a point in a space of what is heard -- tilt, brightness, the cloud's width, shimmer,
the motion, the envelope, the space, the filter, the effects, the conductor's pace -- each
normalised in its parameter's own (skewed) domain.

For every group, two journeys of twelve:

  <Group> - Journey   twelve presets spread across the group (farthest-point sampling from its
                      medoid, so they are its different faces), ordered as a chain of nearest
                      neighbours, so every glide is a step to something close. Six to twelve
                      minutes each, glides of one to two and a half minutes.
  <Group> - Night     the twelve stillest (the least motion, shimmer, ripple and sync, the darkest,
                      the slowest attacks and the slowest conductor), chained the same way, ten to
                      eighteen minutes each, glides of two to four: the version for falling asleep.

And eight whole nights across groups (two from the sleep concerts, one from the visions, one from the swarms and currents): twenty-four presets, the stillest few of every group in a
family of groups, chained,
twelve to twenty minutes each with glides of two and a half to five -- about six and a half
hours before the first comes round again.

Nothing here listens: a nearest neighbour in this space is a preset that is set alike, which is
the promise of a gentle glide, not of a good one.

    python Tools/presets/make_journeys.py
"""
import math
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = ROOT / "Core" / "src" / "Journeys.inc"
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import make_presets  # noqa: E402

# What is heard, and how much each counts in the distance.
FEATURES = {
    "tilt": 1.5, "brightness": 1.5, "cloud_width": 1.0, "shimmer": 1.0, "drift": 0.5, "inharmonic": 1.0, "odd_even": 0.7,
    "attack": 1.0, "release": 0.7, "depth": 1.0, "far_level": 0.7, "air": 0.8, "presence": 0.5,
    "motion_depth": 1.0, "modal_level": 0.6, "delay_mix": 0.4, "spec_mix": 0.5, "melt_mix": 0.7, "memory_level": 0.4,
    "life_depth": 0.6, "sync_depth": 0.6, "ripple_depth": 0.6, "auto_rate": 0.8, "purity": 0.4,
}
# The stillness: what moves, brightens or hurries (lower is stiller).
STILL = {"motion_depth": 1.0, "shimmer": 1.0, "brightness": 1.0, "ripple_depth": 1.0, "sync_depth": 0.7, "air": 0.5,
         "attack": -1.0, "auto_rate": -1.0, "tilt": -0.8}

WHOLE_NIGHTS = [
    ("Whole Night - Sleep Concert", ["Sleep Concert", "Night Gardens", "Ghosts & Memory", "Choirs", "Strings"]),
    ("Whole Night - Deep Space", ["Deep Space", "Dark Drones", "Melting Spectra", "Resonant Halls", "Shepard & Tides"]),
    ("Whole Night - Living Systems", ["Living Fields", "Turing Gardens", "Synchrony", "Flocks", "Night Gardens", "Reservoir Minds",
                                      "Self-Listening", "Strange Attractors"]),
    ("Whole Night - Tides", ["Waves", "Shepard & Tides", "Ripples", "Pendulums", "Glissandi", "Air & Noise"]),
    ("Whole Night - Lullabies", ["Lullabies", "Cradle", "Slow Breath", "Afterglow", "Twilight Choirs", "Moonlight"]),
    ("Whole Night - Deep Sleep", ["Deep Sleep", "Warm Drones", "Starfield", "Ocean Sleep", "Slow Tides", "Night Rain", "Dream Drift",
                                  "Hypnagogia"]),
    ("Whole Night - Visions", ["Kaleidoscopes", "Spirals", "Ink & Flow", "Feedback Loops", "Raindrops", "Star Streams", "Moire Veils",
                               "Painted Light"]),
    ("Whole Night - Currents & Swarms", ["Currents", "Archipelagos", "Islands", "Chimeras", "Swarms", "Bronze Swells", "Ink Plumes",
                                         "Twin Swarms"]),
    ("Whole Night - Halls & Orbits", ["Cathedral Years", "Planetaria", "Worn Stone", "Growing Halls", "Slow Clocks", "Event Horizons",
                                      "Memory Palaces", "Orbits"]),
]


def param_table():
    """key -> (kind, min, max, default, skew) from Core/src/Params.cpp."""
    text = (ROOT / "Core" / "src" / "Params.cpp").read_text(encoding="utf-8")
    table = {}
    for m in re.finditer(r'\bF\(ParamId::\w+,\s*"(\w+)",\s*"[^"]*",\s*"[^"]*",\s*([-\d.]+)f?,\s*([-\d.]+)f?,\s*([-\d.]+)f?,\s*([-\d.]+)f?', text):
        table[m.group(1)] = ("float", float(m.group(2)), float(m.group(3)), float(m.group(4)), float(m.group(5)))
    for m in re.finditer(r'\bI\(ParamId::\w+,\s*"(\w+)",\s*"[^"]*",\s*"[^"]*",\s*([-\d.]+)f?,\s*([-\d.]+)f?,\s*([-\d.]+)f?', text):
        table[m.group(1)] = ("int", float(m.group(2)), float(m.group(3)), float(m.group(4)), 1.0)
    for m in re.finditer(r'\bC\(ParamId::\w+,\s*"(\w+)",\s*"[^"]*",\s*"[^"]*",\s*\w+,\s*\w+,\s*(\d+)\)', text):
        table[m.group(1)] = ("choice", 0.0, 0.0, float(m.group(2)), 1.0)
    for m in re.finditer(r'\bB\(ParamId::\w+,\s*"(\w+)",\s*"[^"]*",\s*"[^"]*",\s*(true|false)\)', text):
        table[m.group(1)] = ("bool", 0.0, 1.0, 1.0 if m.group(2) == "true" else 0.0, 1.0)
    return table


def norm(table, key, value):
    """A value in its parameter's normalised (skewed) domain, as Params.cpp's toNorm."""
    kind, lo, hi, _, skew = table[key]
    if hi <= lo:
        return 0.0
    x = min(1.0, max(0.0, (value - lo) / (hi - lo)))
    return x if skew == 1.0 else x ** skew


def features(table, params):
    """A preset's point: every feature normalised; an effect or a motion that is off counts as nothing."""
    def get(k):
        return float(params[k]) if k in params else table[k][3]
    out = {}
    for k in FEATURES:
        out[k] = norm(table, k, get(k))
    if get("motion") == 0:
        out["motion_depth"] = 0.0
    if get("life_mode") == 0:
        out["life_depth"] = 0.0
    # A preset made for the hands is played by the conductor in a journey (the instrument keeps it
    # playing), at its own pace: it counts as it is set, not as the stillest.
    return out


def distance(a, b):
    return math.sqrt(sum(FEATURES[k] * (a[k] - b[k]) ** 2 for k in FEATURES))


def stillness(f):
    return sum(w * (f[k] if w > 0 else 1.0 - f[k]) * abs(w) for k, w in STILL.items())


def spread(points, names, count):
    """Farthest-point sampling from the medoid: count names, the group's different faces."""
    medoid = min(names, key=lambda n: sum(distance(points[n], points[m]) for m in names))
    chosen = [medoid]
    while len(chosen) < min(count, len(names)):
        best = max((n for n in names if n not in chosen), key=lambda n: min(distance(points[n], points[c]) for c in chosen))
        chosen.append(best)
    return chosen


def chain(points, names):
    """The names as a chain of nearest neighbours, from the first."""
    left = list(names[1:])
    out = [names[0]]
    while left:
        nxt = min(left, key=lambda n: distance(points[out[-1]], points[n]))
        out.append(nxt)
        left.remove(nxt)
    return out


def journey(title, names, dwell, glide):
    lines = ["# Empyrean journey: <preset> | <dwell m:ss or a range> | <glide m:ss or a range>",
             "# Made by Tools/presets/make_journeys.py from the presets' parameters; copy it and edit it freely.",
             f"journey {title}", "cyclic on"]
    lines += [f"{n} | {dwell} | {glide}" for n in names]
    return "\n".join(lines) + "\n"


def main():
    table = param_table()
    bank = make_presets.build()
    points, groups = {}, {}
    for name, g, params, _ in bank:
        p = dict(kv.split("=", 1) for kv in params.split())
        points[name] = features(table, p)
        groups.setdefault(make_presets.GROUPS[g], []).append(name)
    journeys = []
    for group in make_presets.GROUPS:
        names = groups[group]
        journeys.append((f"{group} - Journey", journey(f"{group} - Journey", chain(points, spread(points, names, 12)), "6:00-12:00", "1:00-2:30")))
        still = sorted(names, key=lambda n: stillness(points[n]))[:12]
        journeys.append((f"{group} - Night", journey(f"{group} - Night", chain(points, still), "10:00-18:00", "2:00-4:00")))
    for title, family in WHOLE_NIGHTS:
        per = math.ceil(24 / len(family))   # the stillest few of every group in the family
        still = [n for g in family for n in sorted(groups[g], key=lambda n: stillness(points[n]))[:per]][:24]
        start = min(still, key=lambda n: stillness(points[n]))
        still.remove(start)
        journeys.append((title, journey(title, chain(points, [start] + still), "12:00-20:00", "2:30-5:00")))
    lines = ["// Generated by Tools/presets/make_journeys.py -- do not edit; edit the script and run it again.",
             "/// The journeys the instrument comes with (Journey.h), each as its file would read.",
             "const char* const Journeys[] = {"]
    for title, text in journeys:
        body = text.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")
        lines.append(f'    "{body}",')
    lines.append("};")
    OUT.write_bytes(("\n".join(lines) + "\n").encode("utf-8"))
    print(f"{len(journeys)} journeys -> {OUT}")


if __name__ == "__main__":
    main()
