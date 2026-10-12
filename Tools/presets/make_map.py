"""make_map.py: the preset map and the routes over it -- into Core/src/PresetMap.inc and Core/src/Routes.inc.

The map (after Noctuary's Tools/preset_map.py, but from the presets' own parameters: there is no
measured library here) lays every preset of the bank on a square with two axes one can read:

  across  dark -> bright   the brightness, the tilt (falling), the filter's opening, the air, the presence
  up      still -> moving  the motion's depth and rate, the shimmer, the ripple, the sync, the living
                           landscape, a quick attack, a quick conductor, a travelling comb

Across is taken by rank, up by rank within strips of like brightness (so the presets fill the
square evenly: up means more moving than its neighbours in brightness), then a short repulsion
pass moves points that sit on top of each other apart. Near on the map is set alike -- the promise of
a smooth blend, not of a good one.

The routes are walks over the map (Route.h): waypoints by preset name, so they follow the
presets if the map is laid out again, each with a travel time (the glide to it), a hold time and
a blend radius.

    python Tools/presets/make_map.py
"""
import math
import pathlib
import random
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT_MAP = ROOT / "Core" / "src" / "PresetMap.inc"
OUT_ROUTES = ROOT / "Core" / "src" / "Routes.inc"
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import make_presets  # noqa: E402
from make_journeys import param_table, norm  # noqa: E402


def axes(table, p):
    """A preset's (bright, moving), each 0 .. 1 before ranking."""
    def get(k):
        return float(p[k]) if k in p else table[k][3]

    def n(k):
        return norm(table, k, get(k))
    model = int(get("filter_model"))
    opening = 1.0 if model in (0, 4) else (n("cutoff") if model in (1, 2, 3) else 0.2 + 0.6 * n("cutoff"))
    bright = 0.35 * n("brightness") + 0.25 * (1.0 - n("tilt")) + 0.15 * opening + 0.15 * n("air") + 0.10 * n("presence")
    motion = (n("motion_depth") * (0.5 + 0.5 * n("motion_rate"))) if int(get("motion")) != 0 else 0.0
    life = (n("life_depth") * (0.5 + 0.5 * n("life_rate"))) if int(get("life_mode")) != 0 else 0.0
    conductor = (1.0 - n("auto_rate")) if get("brain_on") >= 0.5 else 0.0
    comb = n("comb_depth") * min(1.0, abs(get("comb_sweep")) / 0.5)
    moving = (0.25 * motion + 0.15 * n("shimmer") + 0.10 * n("ripple_depth") + 0.10 * n("sync_depth") + 0.10 * life
              + 0.15 * (1.0 - n("attack")) + 0.10 * conductor + 0.05 * comb)
    return bright, moving


def rank01(values, rng):
    """Values to their ranks, spread evenly over 0.03 .. 0.97 (ties broken by a die)."""
    order = sorted(range(len(values)), key=lambda i: (values[i], rng.random()))
    out = [0.0] * len(values)
    for r, i in enumerate(order):
        out[i] = 0.03 + 0.94 * (r + 0.5) / len(values)
    return out


def repel(xy, steps=40):
    """Pushes points closer than a share of the mean spacing apart, a little each step."""
    n = len(xy)
    d0 = 0.7 / math.sqrt(n)
    p = np.array(xy, dtype=np.float64)
    for _ in range(steps):
        diff = p[:, None, :] - p[None, :, :]
        dist = np.sqrt((diff ** 2).sum(axis=2)) + np.eye(n)
        push = np.clip(d0 - dist, 0.0, None) / dist
        np.fill_diagonal(push, 0.0)
        p += 0.25 * (diff * push[:, :, None]).sum(axis=1)
        np.clip(p, 0.02, 0.98, out=p)
    return p.tolist()


def nearest(points, names, x, y, avoid=()):
    """The preset nearest a place on the map (not one already in the route)."""
    best, bd = None, 1e9
    for nme in names:
        if nme in avoid:
            continue
        px, py = points[nme]
        d = (px - x) ** 2 + (py - y) ** 2
        if d < bd:
            best, bd = nme, d
    return best


def route_text(points, names, places, rng, travel=(60, 180), hold=(30, 120), radius=(0.03, 0.06)):
    """A route through the presets nearest to places, with times drawn from the ranges."""
    taken, items = [], []
    for k, (x, y) in enumerate(places):
        nme = nearest(points, names, x, y, taken)
        taken.append(nme)
        t = 20 if k == 0 else round(rng.uniform(*travel))
        items.append(f"{nme}|{t}|{round(rng.uniform(*hold))}|{rng.uniform(*radius):.3f}")
    return ";".join(items)


def main():
    table = param_table()
    bank = make_presets.build()
    rng = random.Random(4096)
    raw = []
    for name, g, params, _ in bank:
        p = dict(kv.split("=", 1) for kv in params.split())
        raw.append(axes(table, p))
    # Across by rank; up by rank within each of 32 strips of like brightness -- brightness and motion
    # go together (the sleep concerts are dark and still), so ranked each on its own they would
    # leave whole corners empty, and an empty place on the map has nothing to blend.
    xs = rank01([a for a, _ in raw], rng)
    ys = [0.0] * len(raw)
    order = sorted(range(len(raw)), key=lambda i: xs[i])
    strips = 32
    for s in range(strips):
        idx = order[s * len(order) // strips:(s + 1) * len(order) // strips]
        ranked = rank01([raw[i][1] for i in idx], rng)
        for i, y in zip(idx, ranked):
            ys[i] = y
    xy = repel(list(zip(xs, ys)))
    lines = ["// Generated by Tools/presets/make_map.py -- do not edit; edit the script and run it again.",
             "/// Every factory preset's place on the map, in the bank's order: across dark to bright, up still to moving.",
             "const float kPresetMap[][2] = {"]
    for (x, y) in xy:
        lines.append(f"    {{ {x:.4f}f, {y:.4f}f }},")
    lines.append("};")
    OUT_MAP.write_bytes(("\n".join(lines) + "\n").encode("utf-8"))

    # The routes.
    names = [name for name, *_ in bank]
    points = {name: tuple(xy[i]) for i, name in enumerate(names)}
    section = {name: g // 16 for name, g, *_ in bank}
    by = lambda s: [n for n in names if section[n] == s]   # noqa: E731
    circle = [(0.5 + 0.35 * math.cos(2 * math.pi * k / 8), 0.5 + 0.35 * math.sin(2 * math.pi * k / 8)) for k in range(8)]
    spiral = [(0.5 + (0.42 - 0.05 * k) * math.cos(1.3 * k), 0.5 + (0.42 - 0.05 * k) * math.sin(1.3 * k)) for k in range(8)]
    routes = [
        ("Dawn", names, [(0.08, 0.08), (0.3, 0.25), (0.55, 0.45), (0.75, 0.65), (0.92, 0.9)]),
        ("Dusk", names, [(0.92, 0.9), (0.7, 0.6), (0.45, 0.4), (0.25, 0.2), (0.08, 0.08)]),
        ("Round the Map", names, circle),
        ("Dark Shore", names, [(0.1, 0.1), (0.12, 0.35), (0.1, 0.6), (0.12, 0.85), (0.1, 0.5)]),
        ("Bright Shore", names, [(0.9, 0.1), (0.88, 0.35), (0.9, 0.6), (0.88, 0.85), (0.9, 0.5)]),
        ("Stillness", names, [(0.1, 0.08), (0.35, 0.1), (0.6, 0.08), (0.85, 0.1), (0.5, 0.06)]),
        ("Weather", names, [(0.1, 0.9), (0.35, 0.88), (0.6, 0.92), (0.85, 0.88), (0.5, 0.94)]),
        ("Spiral In", names, spiral),
        ("Sleep Walk", by(2), [(0.15, 0.1), (0.3, 0.2), (0.2, 0.35), (0.4, 0.15), (0.5, 0.3), (0.3, 0.05)]),
        ("Living Tour", by(1), circle[::2] + [(0.5, 0.5)]),
        ("Visions Tour", by(3), circle[1::2] + [(0.5, 0.5)]),
        ("Sounds Tour", by(0), [(0.2, 0.2), (0.8, 0.2), (0.8, 0.8), (0.2, 0.8), (0.5, 0.5)]),
    ]
    lines = ["// Generated by Tools/presets/make_map.py -- do not edit; edit the script and run it again.",
             "/// The routes that come with the instrument (Route.h): a name and the text form.",
             "const RoutePreset kRoutes[] = {"]
    for title, pool, places in routes:
        slow = title in ("Sleep Walk", "Stillness", "Dusk")
        text = route_text(points, pool, places, rng, travel=(120, 300) if slow else (60, 180), hold=(60, 240) if slow else (30, 120))
        body = text.replace("\\", "\\\\").replace('"', '\\"')
        lines.append(f'    {{ "{title}", "{body}" }},')
    lines.append("};")
    OUT_ROUTES.write_bytes(("\n".join(lines) + "\n").encode("utf-8"))
    print(f"{len(xy)} presets on the map -> {OUT_MAP}; {len(routes)} routes -> {OUT_ROUTES}")


if __name__ == "__main__":
    main()
