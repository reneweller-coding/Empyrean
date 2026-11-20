"""make_journeys.py: the journeys Empyrean comes with -- into Core/src/Journeys.inc.

A journey (Core/include/additiv/Journey.h) is presets in a row, each held for a while drawn from a
range and glided into the next; cyclic, so a night plays itself. After Noctuary's
Tools/make_journeys.py, which stands on its library's measurement. Here every preset is a point in two spaces at once
(19.11.2026; before, in the second alone):
  what it measures -- work/presets/descriptors.json (or its checked-in copy, descriptors.py), what
      additiv_presets --describe hears of a chord of fourteen seconds and the map and the search by example read:
      its brightness, its motion, its width, its noise, each as z-scores over the bank -- two thirds of the distance
      (PARAM_SHARE);
  how it is set -- tilt, brightness, the cloud's width, shimmer, the motion, the envelope, the space, the filter, the
      effects, the conductor's pace, each normalised in its parameter's own (skewed) domain -- the other third: four
      measures do not tell a tuning, a hall or an effect from another, and the parameters do.
A preset the descriptions lack stands on its parameters alone (the run says how many).

For every group, two journeys of twelve:

  <Group> - Journey   twelve presets spread across the group (farthest-point sampling from its
                      medoid, so they are its different faces), ordered as a chain of nearest
                      neighbours, so every glide is a step to something close.
  <Group> - Night     the twelve stillest -- the least measured motion, the darkest, the least noise, and of the
                      parameters the least motion, shimmer, ripple and sync, the slowest attacks and the slowest
                      conductor -- chained the same way: the version for falling asleep.

And nineteen whole nights across groups (two from the sleep concerts, one each from the visions, the swarms and
currents, the halls and orbits, the harmonic lives, the dreams and phantoms, the weather and growth, the rooms and
topology, the night and illusions, the choruses and mirrors, the deep floors, the stars and gravity, the light and depth, the plasma and voice): twenty-four presets, the stillest few
of every group in a family of groups, chained, timed as the nights.

How long a step stays is the preset's own (19.11.2026, as in Noctuary): what moves by itself -- the motion, the life,
sync and ripple, the streams, shimmer and drift, the conductor's pace, the figures -- is held longest, what stands
still is left soon -- half by the measured motion, half by the parameters (a chord of fourteen seconds does not
show a conductor that changes its notes every few minutes, or a figure). The third that moves least is held 1:30-3:00 (nights 3-6 minutes), the third that moves most
6-10 (nights 10-15), the rest between, never two long or two short ones in a row; the glide grows with the stay.
Before, every step of a journey drew from the same six to twelve minutes, of a night ten to eighteen, of a whole
night twelve to twenty.

A nearest neighbour here is a preset that sounds alike in four measures and is set alike, which is the promise of a
gentle glide, not of a good one.

    python Tools/presets/make_journeys.py
"""
import math
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = ROOT / "Core" / "src" / "Journeys.inc"
PARAM_SHARE = 1.0 / 3.0   # of the distance's spread, the parameters'; the measures have the rest
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import descriptors  # noqa: E402
import make_presets  # noqa: E402

# What is heard, and how much each counts in the distance.
FEATURES = {
    "tilt": 1.5, "brightness": 1.5, "cloud_width": 1.0, "shimmer": 1.0, "drift": 0.5, "inharmonic": 1.0, "odd_even": 0.7,
    "attack": 1.0, "release": 0.7, "depth": 1.0, "far_level": 0.7, "air": 0.8, "presence": 0.5,
    "motion_depth": 1.0, "modal_level": 0.6, "delay_mix": 0.4, "spec_mix": 0.5, "melt_mix": 0.7, "memory_level": 0.4,
    "life_depth": 0.6, "sync_depth": 0.6, "ripple_depth": 0.6, "auto_rate": 0.8, "purity": 0.4,
    # Phases 30 and 31: nought in every preset before them, so their journeys stay as they were
    "harp": 0.6, "thunder": 0.5, "ice": 0.5, "stream_depth": 0.5, "risset": 0.5, "night_arc": 0.4, "chorus": 0.6,
    "mirror": 0.5,
    # Phase 32: the floor beside the foundation
    "pedal": 0.4, "pressure": 0.5, "floor_sub": 0.5, "floor_fifth": 0.4,
    # Phase 33: space
    "star": 0.6, "black_hole": 0.5, "merger": 0.4, "redshift": 0.5,
    # Phase 34: light and depth
    "scintillation": 0.5, "lines": 0.6, "absorption": 0.5, "wreck": 0.5, "sofar": 0.5,
    # Phase 35: plasma and voice
    "magneto": 0.5, "dawn_chorus": 0.4, "throat": 0.6, "throat_sub": 0.4,
}
# The stillness: what moves, brightens or hurries (lower is stiller).
STILL = {"motion_depth": 1.0, "shimmer": 1.0, "brightness": 1.0, "ripple_depth": 1.0, "sync_depth": 0.7, "air": 0.5,
         "attack": -1.0, "auto_rate": -1.0, "tilt": -0.8, "thunder": 0.5, "risset": 0.5, "chorus": 0.3}

# What moves by itself (higher moves more): how long a step is held. The conductor's pace and the figures are
# added in self_motion().
MOTION = {"motion_depth": 1.0, "life_depth": 1.0, "sync_depth": 0.7, "ripple_depth": 0.7, "stream_depth": 0.6,
          "shimmer": 0.5, "drift": 0.5, "risset": 0.5, "night_arc": 0.4, "thunder": 0.3, "chorus": 0.3}
TIMINGS = {   # short, middle, long: (dwell, glide)
    "journey": [("1:30-3:00", "0:30-1:00"), ("3:00-6:00", "0:45-1:30"), ("6:00-10:00", "1:00-2:00")],
    "night": [("3:00-6:00", "1:00-2:00"), ("6:00-10:00", "1:30-2:30"), ("10:00-15:00", "2:00-3:30")],
}

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
    ("Whole Night - Harmonic Lives", ["Turning Years", "Night Ecology", "Breathing Grains", "Aurora Curtains", "Glacier Calving",
                                      "Breath Registers", "Subterranea", "Halo Horizons"]),
    ("Whole Night - Dreams & Phantoms", ["Night Memories", "Breathing In", "Inner Voices", "Temple Winds", "Glass Forests",
                                         "Dreaming Rooms", "Bloch Tides", "Lucid Dreams"]),
    ("Whole Night - Weather & Growth", ["Rain on Water", "Seashores", "Mycelium Nights", "Forest Rain", "Rustling Woods",
                                        "Swarm Rings", "Pine Winds", "Tam-Tam Halls"]),
    ("Whole Night - Rooms & Topology", ["Interference Nights", "Chapel Air", "Halo Rings", "Curtained Rooms",
                                        "Tonnetz Walks", "Standing Waves", "Still Halos", "Stone Halls"]),
    ("Whole Night - Night & Illusions", ["Night Arcs", "Frozen Nights", "Distant Storms", "Endless Ritardando",
                                         "Aeolian Harps", "Singing Lakes", "Fusion Choirs", "Harp Gardens"]),
    ("Whole Night - Chorus & Mirrors", ["Cool Nights", "Cricket Meadows", "Frog Ponds", "Night Concerts", "Lake Nights",
                                        "Mirror Chapels", "Time Mirrors", "Nocturnes"]),
    ("Whole Night - Deep Floors", ["Breathing Floors", "Organ Points", "Low Chapels", "Abyssal Plains", "Sub Abysses",
                                   "Night Pressure", "Cavern Drones", "Undertow"]),
    ("Whole Night - Stars & Gravity", ["Quiet Universe", "Sun Songs", "Red Giants", "Falling Light", "Redshift Halls",
                                       "Cosmic Floors", "Star Choirs", "Distant Mergers"]),
    ("Whole Night - Light & Depth", ["Nebula Glow", "Twinkling Stars", "Ocean Crossings", "Whale Roads", "Derelict Hulls",
                                     "Hydrogen Choirs", "Deep Channels", "Thermal Snaps"]),
    ("Whole Night - Plasma & Voice", ["Monk Drones", "Whistler Nights", "Kargyraa Halls", "Plasma Hiss", "Abyssal Choirs",
                                      "Dawn Choruses", "Vowel Tides", "Giant Throats"]),
]


def param_table():
    """key -> (kind, min, max, default, skew) from Core/src/Params.cpp."""
    text = (ROOT / "Core" / "src" / "Params.cpp").read_text(encoding="utf-8")
    table = {}
    for m in re.finditer(r'\bF\(\s*ParamIdE?::\w+,\s*"(\w+)",\s*"[^"]*",\s*"[^"]*",\s*([-\d.]+)f?,\s*([-\d.]+)f?,\s*([-\d.]+)f?,\s*([-\d.]+)f?', text):
        table[m.group(1)] = ("float", float(m.group(2)), float(m.group(3)), float(m.group(4)), float(m.group(5)))
    for m in re.finditer(r'\bI\(\s*ParamIdE?::\w+,\s*"(\w+)",\s*"[^"]*",\s*"[^"]*",\s*([-\d.]+)f?,\s*([-\d.]+)f?,\s*([-\d.]+)f?', text):
        table[m.group(1)] = ("int", float(m.group(2)), float(m.group(3)), float(m.group(4)), 1.0)
    for m in re.finditer(r'\bC\(\s*ParamIdE?::\w+,\s*"(\w+)",\s*"[^"]*",\s*"[^"]*",\s*\w+,\s*\w+,\s*(\d+)\s*\)', text):
        table[m.group(1)] = ("choice", 0.0, 0.0, float(m.group(2)), 1.0)
    for m in re.finditer(r'\bB\(\s*ParamIdE?::\w+,\s*"(\w+)",\s*"[^"]*",\s*"[^"]*",\s*(true|false)\s*\)', text):
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


def measured(d):
    """A description's features (work/presets/descriptors.json, as make_find.py reads it): brightness, log motion,
    width, noise."""
    return [d[0], math.log(max(d[1], 0.0) + 0.01), d[2], d[3]]


def zscores(rows):
    """Each column centred on the bank's mean and scaled by its spread."""
    n, k = len(rows), len(rows[0])
    mean = [sum(r[i] for r in rows) / n for i in range(k)]
    sd = [math.sqrt(sum((r[i] - mean[i]) ** 2 for r in rows) / n) or 1.0 for i in range(k)]
    return [[(r[i] - mean[i]) / sd[i] for i in range(k)] for r in rows]


def points_of(names, params_points, desc):
    """Every preset's point: the four measures (standardised) and the parameters (weighted, scaled so that they carry
    PARAM_SHARE of the bank's spread). A preset without a description stands on its parameters alone."""
    keys = list(FEATURES)
    prow = [[math.sqrt(FEATURES[k]) * params_points[n][k] for k in keys] for n in names]
    have = [n for n in names if n in desc]
    mz = dict(zip(have, zscores([measured(desc[n]) for n in have]))) if have else {}
    pm = [sum(r[i] for r in prow) / len(prow) for i in range(len(keys))]
    pvar = sum(sum((r[i] - pm[i]) ** 2 for r in prow) / len(prow) for i in range(len(keys))) or 1.0
    # the measures' variance is four (z-scores); the parameters are scaled to PARAM_SHARE of the whole
    scale = math.sqrt(4.0 * PARAM_SHARE / (1.0 - PARAM_SHARE) / pvar)
    out = {}
    for n, r in zip(names, prow):
        m = mz.get(n)
        out[n] = (m if m is not None else [0.0] * 4) + [scale * x for x in r]
    return out, mz


def distance(a, b):
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def stillness_by_params(f):
    return sum(w * (f[k] if w > 0 else 1.0 - f[k]) * abs(w) for k, w in STILL.items())


def self_motion(table, params, f):
    """How much a preset moves by itself: its motions, the conductor's pace and the figures."""
    def get(k):
        return float(params[k]) if k in params else table[k][3]
    m = sum(w * f[k] for k, w in MOTION.items())
    pace = "auto_rate" if get("auto_mode") == 1 else "brain_rate"
    m += 1.0 - norm(table, pace, get(pace))
    m += 0.25 * get("figures")
    return m


def timings(names, moving, kind):
    """(dwell, glide) for each of `names`, in their order, by how much each moves by itself."""
    n = len(names)
    rank = sorted(range(n), key=lambda i: moving[names[i]])
    third = max(1, n // 3)
    cls = [1] * n
    for i in rank[:third]:
        cls[i] = 0
    for i in rank[n - third:]:
        cls[i] = 2
    # No two long and no two short ones in a row (the journeys are cyclic: the last meets the first).
    for i in range(n):
        j = (i + 1) % n
        if j != i and cls[i] == cls[j] and cls[j] != 1:
            cls[j] = 1
    return [TIMINGS[kind][c] for c in cls]


def spread(points, names, count):
    """Farthest-point sampling from the medoid: count names, the group's different faces."""
    medoid = min(names, key=lambda n: sum(distance(points[n], points[m]) for m in names))
    chosen = [medoid]
    while len(chosen) < min(count, len(names)):
        best = max((n for n in names if n not in chosen), key=lambda n: min(distance(points[n], points[c]) for c in chosen))
        chosen.append(best)
    return chosen


def chain(points, names):
    """The names as a chain of nearest neighbours, from the first; then untangled (2-opt): a journey is cyclic, and
    the greedy chain's last step back to the first was its longest -- a section turned round wherever that makes the
    whole round shorter, the first staying first."""
    left = list(names[1:])
    out = [names[0]]
    while left:
        nxt = min(left, key=lambda n: distance(points[out[-1]], points[n]))
        out.append(nxt)
        left.remove(nxt)
    d = lambda a, b: distance(points[a], points[b])
    better = True
    while better and len(out) > 3:
        better = False
        for i in range(1, len(out) - 1):
            for j in range(i + 1, len(out)):
                a, b, c, e = out[i - 1], out[i], out[j], out[(j + 1) % len(out)]
                if d(a, c) + d(b, e) < d(a, b) + d(c, e) - 1e-9:
                    out[i:j + 1] = reversed(out[i:j + 1])
                    better = True
    return out


def journey(title, names, moving, kind):
    lines = ["# Empyrean journey: <preset> | <dwell m:ss or a range> | <glide m:ss or a range>",
             "# Made by Tools/presets/make_journeys.py from how the presets measure and how they are set; copy it and edit it freely.",
             f"journey {title}", "cyclic on"]
    lines += [f"{n} | {dwell} | {glide}" for n, (dwell, glide) in zip(names, timings(names, moving, kind))]
    return "\n".join(lines) + "\n"


def standardised(values):
    """name -> value, as z-scores over the bank."""
    names = list(values)
    return dict(zip(names, (r[0] for r in zscores([[values[n]] for n in names]))))


def main():
    table = param_table()
    bank = make_presets.build()
    byparams, groups, pmoving, pstill = {}, {}, {}, {}
    for name, g, params, _ in bank:
        p = dict(kv.split("=", 1) for kv in params.split())
        byparams[name] = features(table, p)
        pmoving[name] = self_motion(table, p, byparams[name])
        pstill[name] = stillness_by_params(byparams[name])
        groups.setdefault(make_presets.GROUPS[g], []).append(name)
    desc = descriptors.load()
    names = [name for name, *_ in bank]
    missing = [n for n in names if n not in desc]
    if missing:
        print(f"the descriptions lack {len(missing)} presets (first: {missing[0]}): they stand on their parameters")
    points, mz = points_of(names, byparams, desc)
    # What moves by itself: the measured motion (the spectrum's change over a quarter of a second to three seconds)
    # and the parameters' (the conductor's pace and the figures, which a chord of fourteen seconds cannot show), half
    # and half. And the stillness: the measured motion, brightness and noise against the parameters' stillness.
    pm, ps = standardised(pmoving), standardised(pstill)
    moving = {n: (0.5 * mz[n][1] + 0.5 * pm[n]) if n in mz else pm[n] for n in names}
    still_score = {n: (0.4 * mz[n][1] + 0.2 * mz[n][0] + 0.1 * mz[n][3] + 0.3 * ps[n]) if n in mz else ps[n]
                   for n in names}

    def stillness(n):
        return still_score[n]

    journeys = []
    for group in make_presets.GROUPS:
        names = groups[group]
        journeys.append((f"{group} - Journey", journey(f"{group} - Journey", chain(points, spread(points, names, 12)), moving, "journey")))
        still = sorted(names, key=lambda n: stillness(n))[:12]
        journeys.append((f"{group} - Night", journey(f"{group} - Night", chain(points, still), moving, "night")))
    for title, family in WHOLE_NIGHTS:
        per = math.ceil(24 / len(family))   # the stillest few of every group in the family
        still = [n for g in family for n in sorted(groups[g], key=lambda n: stillness(n))[:per]][:24]
        start = min(still, key=lambda n: stillness(n))
        still.remove(start)
        journeys.append((title, journey(title, chain(points, [start] + still), moving, "night")))
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
