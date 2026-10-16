"""make_wreck.py: the wreck's modes (Phase 34) -- four shapes of a derelict as finite mass-spring models, their modes by
an eigen-decomposition, written to Core/src/Wreck.inc.

    python Tools/presets/make_wreck.py

Each shape is a lattice of equal masses joined by springs: the members of a truss, or a shell's edges with its
diagonals (shear) and its next-but-one neighbours (bending). Its stiffness matrix K (3 x nodes square, a spring
k n n^T between its ends, n the spring's direction) gives the modes as K phi = omega^2 phi (lumped equal masses);
the six free-body modes (omega near 0) are dropped. Kept per shape: the lowest WreckModes modes' frequencies over the
first's, how strongly each is struck at WreckHits places over the structure (the mode's displacement there over its
largest), and how two pick-up points hear it (its displacement along a fixed direction: the stereo). The engine rings
the modes, strikes them where a micrometeorite or a thermal snap lands, and tunes the lowest to the root.

    Hull   a cylindrical module, ten rings of fourteen, end bulkheads
    Tank   a spherical tank, nine latitudes of sixteen and the poles
    Truss  a square truss, twenty bays: long and thin, many low bending modes
    Plate  a perforated plate, free at its edges
"""
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "Core" / "src" / "Wreck.inc"
MODES = 64
HITS = 16
SHAPES = ["Hull", "Tank", "Truss", "Plate"]


def assemble(nodes, springs):
    """The stiffness matrix of springs (i, j, k) between nodes."""
    n = len(nodes)
    K = np.zeros((3 * n, 3 * n))
    for i, j, k in springs:
        d = nodes[j] - nodes[i]
        u = d / np.linalg.norm(d)
        b = k * np.outer(u, u)
        for a, c, s in ((i, i, 1), (j, j, 1), (i, j, -1), (j, i, -1)):
            K[3 * a:3 * a + 3, 3 * c:3 * c + 3] += s * b
    return K


def near(nodes, i, j, limit):
    return np.linalg.norm(nodes[i] - nodes[j]) <= limit


def hull():
    rings, around, r, length = 10, 14, 1.0, 3.0
    nodes = [np.array([r * np.cos(2 * np.pi * a / around), r * np.sin(2 * np.pi * a / around), length * z / (rings - 1)])
             for z in range(rings) for a in range(around)]
    nodes += [np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, length])]
    nodes = np.array(nodes)
    idx = lambda z, a: z * around + a % around
    springs = []
    for z in range(rings):
        for a in range(around):
            springs.append((idx(z, a), idx(z, a + 1), 1.0))              # round the ring
            springs.append((idx(z, a), idx(z, a + 2), 0.15))             # its bending
            if z + 1 < rings:
                springs.append((idx(z, a), idx(z + 1, a), 1.0))          # along
                springs.append((idx(z, a), idx(z + 1, a + 1), 0.5))      # shear
                springs.append((idx(z, a + 1), idx(z + 1, a), 0.5))
            if z + 2 < rings:
                springs.append((idx(z, a), idx(z + 2, a), 0.15))         # bending along
    for a in range(around):                                              # the bulkheads
        springs.append((len(nodes) - 2, idx(0, a), 0.6))
        springs.append((len(nodes) - 1, idx(rings - 1, a), 0.6))
    return nodes, springs


def tank():
    lats, around = 9, 16
    nodes = [np.array([0.0, 0.0, 1.0])]
    for l in range(1, lats + 1):
        th = np.pi * l / (lats + 1)
        for a in range(around):
            ph = 2 * np.pi * (a + 0.5 * (l % 2)) / around
            nodes.append(np.array([np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)]))
    nodes.append(np.array([0.0, 0.0, -1.0]))
    nodes = np.array(nodes)
    d = np.linalg.norm(nodes[:, None, :] - nodes[None, :, :], axis=2)
    springs = []
    near1 = np.sort(d[d > 0])[: len(nodes) * 4]
    edge = near1.max() * 1.05
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            if d[i, j] <= edge:
                springs.append((i, j, 1.0))
            elif d[i, j] <= 2.0 * edge:
                springs.append((i, j, 0.12))   # bending, the shell's curvature held
    return nodes, springs


def truss():
    bays, w = 20, 0.4
    corners = [(0, 0), (w, 0), (w, w), (0, w)]
    nodes = np.array([[x, y, 0.5 * b] for b in range(bays + 1) for (x, y) in corners])
    idx = lambda b, c: 4 * b + c % 4
    springs = []
    for b in range(bays + 1):
        for c in range(4):
            springs.append((idx(b, c), idx(b, c + 1), 1.0))
        springs.append((idx(b, 0), idx(b, 2), 0.7))
        if b < bays:
            for c in range(4):
                springs.append((idx(b, c), idx(b + 1, c), 1.0))
                springs.append((idx(b, c), idx(b + 1, c + 1), 0.7))
    return nodes, springs


def plate():
    nx, ny = 15, 10
    holes = {(4, 4), (4, 5), (5, 4), (5, 5), (10, 3), (10, 4), (11, 6)}
    keep = [(x, y) for y in range(ny) for x in range(nx) if (x, y) not in holes]
    where = {p: i for i, p in enumerate(keep)}
    rng = np.random.default_rng(34)
    # a plate bends out of its plane: the nodes a little off it (a slight camber), so the springs carry the bending
    nodes = np.array([[0.2 * x, 0.2 * y, 0.02 * np.sin(0.7 * x) * np.cos(0.9 * y) + 0.002 * rng.standard_normal()]
                      for (x, y) in keep])
    springs = []
    for (x, y), i in where.items():
        for dx, dy, k in ((1, 0, 1.0), (0, 1, 1.0), (1, 1, 0.5), (1, -1, 0.5), (2, 0, 0.2), (0, 2, 0.2)):
            j = where.get((x + dx, y + dy))
            if j is not None:
                springs.append((i, j, k))
    # a second layer joined by short struts gives the plate its stiffness against bending
    top = nodes + np.array([0.0, 0.0, 0.08])
    n = len(nodes)
    allnodes = np.vstack([nodes, top])
    for (x, y), i in where.items():
        springs.append((i, i + n, 1.0))
        for dx, dy in ((1, 0), (0, 1)):
            j = where.get((x + dx, y + dy))
            if j is not None:
                springs.append((i + n, j + n, 1.0))
                springs.append((i, j + n, 0.4))
                springs.append((j, i + n, 0.4))
    return allnodes, springs


def modes(nodes, springs):
    K = assemble(nodes, springs)
    w, v = np.linalg.eigh(K)
    keep = w > 1e-5 * w.max()   # the free-body modes and a soft mechanism of the plate (4e-6) left out
    w, v = w[keep], v[:, keep]
    return w[:MODES], v[:, :MODES]


def main():
    out = ["// Generated by Tools/presets/make_wreck.py -- do not edit: the wreck's modes (Phase 34).", ""]
    ratios, hits, picks = [], [], []
    for name, make in zip(SHAPES, (hull, tank, truss, plate)):
        nodes, springs = make()
        w, v = modes(nodes, springs)
        n = len(nodes)
        f = np.sqrt(w / w[0])
        disp = v.reshape(n, 3, -1)                       # node, axis, mode
        amp = np.linalg.norm(disp, axis=1)               # node, mode
        rng = np.random.default_rng(len(name))
        where = np.sort(rng.choice(n, HITS, replace=False))
        h = amp[where] / np.maximum(amp.max(axis=0, keepdims=True), 1e-12)
        centre = nodes.mean(axis=0)
        far = np.argsort(np.linalg.norm(nodes - centre, axis=1))[-2:]
        direction = np.array([0.3, 0.5, 0.81])
        p = np.einsum("nam,a->nm", disp[far], direction)
        p /= np.maximum(np.abs(p).max(), 1e-12)
        ratios.append(f)
        hits.append(h.T)
        picks.append(p.T)
        print(f"{name}: {n} nodes, {len(springs)} springs, modes 1 .. {MODES}: x{f[-1]:.1f} over the first")
    out.append(f"constexpr int WreckShapes = {len(SHAPES)};   ///< hull, tank, truss, plate")
    out.append(f"constexpr int WreckModes = {MODES};    ///< modes a shape")
    out.append(f"constexpr int WreckHits = {HITS};     ///< places a shape is struck")
    out.append("/** @brief Each shape's modes' frequencies over its first. */")
    out.append("constexpr float WreckRatio[WreckShapes][WreckModes] = {")
    for f in ratios:
        out.append("    { " + ", ".join(f"{x:.5f}f" for x in f) + " },")
    out.append("};")
    out.append("/** @brief How strongly each mode is struck at each place (its displacement there over its largest). */")
    out.append("constexpr float WreckHit[WreckShapes][WreckModes][WreckHits] = {")
    for h in hits:
        out.append("    {")
        for row in h:
            out.append("        { " + ", ".join(f"{x:.4f}f" for x in row) + " },")
        out.append("    },")
    out.append("};")
    out.append("/** @brief How the two pick-ups (left, right) hear each mode, signed. */")
    out.append("constexpr float WreckPick[WreckShapes][WreckModes][2] = {")
    for p in picks:
        out.append("    { " + ", ".join(f"{{ {a:.4f}f, {b:.4f}f }}" for a, b in p) + " },")
    out.append("};")
    OUT.write_text("\n".join(out) + "\n", encoding="utf-8", newline="\n")
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
