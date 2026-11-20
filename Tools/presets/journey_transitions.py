"""journey_transitions.py: what happens where a journey glides from one preset into the next (plan D2, 20.11.2026).

make_journeys.py chains a journey's presets so that neighbours measure alike -- a promise of a gentle glide, not a
check of one: two presets can measure alike and stand in different tunings or on different roots. This tool hears
the transitions of the journeys the instrument comes with (Core/src/Journeys.inc, or another file of that form):
every neighbour pair of every journey, the glide into the second as the journey asks for it (the middle of its
glide's range). What a glide does (Engine::updateSound, Morph.h): every number glides, every choice -- the scale,
the root, the conductor's mode -- changes at the glide's middle, and what sounds keeps its key and is retuned into the
new scale with a time constant of a tenth of the glide (at least GlideSeconds, at most six seconds). The conductors are
not restarted: they keep their chord and their own root, which wanders and is never set from a preset.

Three passes, each resumable (results appended to TSVs under work/journeys/, done pairs skipped):

  static   for every pair, without the card: the root distance in fifths (the presets' Root), and how far the keys
           of the first preset's register are retuned when the second's tuning takes over (the largest move in
           cents, from `additiv_brain_audit --tuning`, every key's pitch as each preset tunes it). A pair's tunings
           are compatible when no key of the register moves by a quarter tone (50 cents) or more: every note that
           sounds stays the degree it was; moved_share is the share of the register's keys that move so far.
           Seconds for the whole set. (Measured 20.11.2026: almost no pair is compatible so -- in JI 7-limit, whose
           keys snap, a sixth of the keys move by about a semitone whatever the root, between two scales half.)
  notes    for every pair, without the card (`additiv_brain_audit --pairs`): the first preset for PRE seconds, the
           glide, POST seconds after it. From the notes and the tuning as it changes: the seconds during which a
           note of the outgoing chord (begun before the glide) and a note of the incoming one (begun during or after
           it) -- the conductors and the foundation -- stand a minor second or a minor ninth apart (25 to 150 cents,
           or an octave more: the conductor's own rub; by pitch in Hz, never by key), a note counted until it has
           fallen 20 dB after its release (0.47 of the release time); the same with the major seventh and its
           octave counted too (rub_m7_s: the interval class of a semitone under two octaves); the figures' notes
           against the outgoing chord (rub_fig_s); and the largest retune of a note that sounds at the glide's middle
           and how many such notes move by a quarter tone or more (moved50).
           `--hold-pitch` (plan D2b) is Engine::setHoldPitch, 1 by default as in the instrument since 20.11.2026: a
           note that sounds when a glide changes the tuning keeps its pitch in Hz until it ends (the audit prints
           `H,<s>,<source>,<note>,<Hz>` for each such note, and its pitch is that from then on); the results go to
           notes_hold1<tag>.tsv. `--hold-pitch 0` measures the instrument as it was before, every sounding note
           retuned (notes<tag>.tsv, the files of D2). The two are two instruments: notes, report and export-bad read
           only the files of the one asked for. `--fresh` measures every pair again into this run's own file
           (otherwise a pair any run of the same instrument measured is skipped).
  render   on the graphics card, a sample (the card is shared: chunks of at most --chunk-min minutes, the card
           taken through G:/Tools/Games/gpu_card.py and given back between chunks, all of it within --budget-min):
           `empy_render --preset A --fill --set brain_on=1 --then B --at RPRE --glide G` (about thirty seconds a
           transition), and from the audio the short-term loudness (BS.1770 K weighting, 3 s windows): the steady
           levels before (the last 25 s before the glide) and after (the last 25 s of the render), how far the glide
           and the 15 s after it fall below the lowest or rise above the highest short-term loudness either preset
           has in its steady stretch (LU), and a roughness peak (Vassilakis over the spectrum's 40 strongest peaks,
           every half second) against the larger steady roughness, as a ratio. The curves are kept
           (work/journeys/curves/), the audio is not. --card-sample takes the sample card_sample() draws.
  alone    on the card (plan D2b, 20.11.2026): every rendered pair whose level dips or peaks over the limit has both
           its presets rendered alone, as the pair was (`--fill`, conductor on, the same seconds), and the dip and the
           peak counted again beyond what either preset does by itself over the same stretch (dip_alone_lu,
           peak_alone_lu in alone.tsv; the report and export-bad take those where they exist). The holes of 13 to 53 LU
           found in D2 were the presets' own: played alone, without any glide, Travertine Atrium falls 55 LU under its
           first 25 seconds' lowest level in the same stretch (its conductor strikes at 25 s and next at 61 s, every
           note decaying to nothing in seconds), Seething Eddy Light 43, Answering Reply 22, Scintillant Rigel 16 --
           as deep as with the glide, and a glide of any single knob (empy_render --then-only) gave the same hole. The
           first 25 s after `--fill` are not the preset's whole range. Same budget rules as render; curves of the
           presets alone in work/journeys/curves/alone_*.npz.

  report   joins them into work/journeys/transitions.tsv and prints the counts of broken transitions: a level dip or
           peak over 3 LU, or more than 5 s of semitone rubbing (the plan's acceptance, D2).
  export-bad  every pair measured as broken, from all the runs of the instrument asked for (--hold-pitch), into
           Tools/presets/journey_bad_pairs.tsv, which make_journeys.py keeps apart. A level dip or peak counts as
           measured against both presets alone where that was measured (alone.tsv; plan D2b: the holes the old
           yardstick found were the presets' own); the renders are the old instrument's, whose levels the switch
           does not move (it keeps a pitch, not a level).
  --check  the ctest `journeys` (no card, no program): no neighbour pair of Core/src/Journeys.inc that
           journey_bad_pairs.tsv lists, and no two presets of one scale whose roots stand more than --max-fifths
           (5) fifths apart -- a tritone between the roots rubbed in a third of the transitions, the other root steps
           in a quarter.

    python Tools/presets/journey_transitions.py static
    python Tools/presets/journey_transitions.py notes            (four processes, below normal)
    python Tools/presets/journey_transitions.py render --card-sample --budget-min 52 --chunk-min 15
    python Tools/presets/journey_transitions.py alone --card-sample --budget-min 30 --chunk-min 15
    python Tools/presets/journey_transitions.py export-bad
    python Tools/presets/journey_transitions.py report
    python Tools/presets/journey_transitions.py --check

`--inc FILE` reads another journeys file (a candidate from make_journeys.py --out), `--tag NAME` keeps its results
apart; the pairs are keyed by (from, to, glide), so a candidate's pairs that the shipped journeys share are measured
once.
"""
import argparse
import concurrent.futures as cf
import json
import math
import os
import pathlib
import random
import re
import subprocess
import sys
import tempfile
import time

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import make_presets  # noqa: E402

INC = ROOT / "Core" / "src" / "Journeys.inc"
WORK = ROOT / "work" / "journeys"
BIN = ROOT / "bin" / "msvc"
AUDIT_EXE = BIN / "additiv_brain_audit.exe"
RENDER_EXE = BIN / "empy_render.exe"
GPU_CARD = pathlib.Path(r"G:\Tools\Games")

PRE = 120.0        # notes: the first preset plays this long before the glide (its chord built without --fill)
POST = 60.0        # notes: seconds after the glide's end
RPRE = 40.0        # render: the first preset (its chord filled at once) this long before the glide
RPOST = 40.0       # render: seconds after the glide's end
STEADY = 25.0      # render: the steady levels over this many seconds
COMPATIBLE_C = 50.0  # a key moved by a quarter tone or more becomes another degree
MAX_FIFTHS = 5     # the check: roots of one scale at most five fifths apart (a tritone rubbed most)
MAX_BAD_SHARE = 0.10  # the check: at most this share of the neighbour pairs measured as broken (shipped 19.11.: 20 %)
RUB_LO, RUB_HI = 25.0, 150.0   # a semitone or a major seventh, folded into an octave
TAIL = 0.47        # a released note counts until 20 dB down: the release is five time constants
LU_LIMIT, RUB_LIMIT = 3.0, 5.0  # the plan's acceptance: a level jump over 3 LU, more than 5 s of semitone rubbing

PAIR_COLUMNS = ["journey", "index", "from", "to", "glide"]
STATIC_COLUMNS = ["key", "from", "to", "root_from", "root_to", "fifths", "scale_from", "scale_to", "retune_max_c",
                  "retune_mean_c", "moved_share", "compatible"]
NOTE_COLUMNS = ["key", "rub_s", "rub_m7_s", "rub_fig_s", "retune_sounding_c", "out_notes", "in_notes", "chord_mid",
                "moved50"]
RENDER_COLUMNS = ["key", "level_from", "level_to", "dip_lu", "peak_lu", "jump_lu", "rough_ratio", "card_s", "range_lu"]
ALONE_COLUMNS = ["key", "dip_alone_lu", "peak_alone_lu", "alone_low_lu", "alone_high_lu"]
# (rows without range_lu were measured against the steady stretches' medians, not their extremes: an upper bound of
# the dip and the peak; those over the limit are rendered again)
SCALES = ["12-TET", "JI Major", "JI Minor", "JI 7-limit", "Pythagorean", "JI Pentatonic", "Harmonic 8-16",
          "Subharmonic 16-8", "Slendro", "Bohlen-Pierce", "Otonality 1-11", "User", "Timbre"]
ROOTS = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


# ----------------------------------------------------------------------------------------------------------- pairs
def parse_time(t):
    """m:ss, h:mm:ss or seconds -> seconds."""
    v = 0.0
    for part in t.strip().split(":"):
        v = v * 60.0 + float(part)
    return v


def parse_range(r):
    """'a-b' or 'a' -> (lo, hi) in seconds."""
    r = r.strip()
    i = r.find("-", 1)
    return (parse_time(r), parse_time(r)) if i < 0 else (parse_time(r[:i]), parse_time(r[i + 1:]))


def journeys(inc=INC):
    """[(title, [(preset, (dwell lo, hi), (glide lo, hi))])] from a Journeys.inc."""
    text = pathlib.Path(inc).read_text(encoding="utf-8")
    out = []
    for body in re.findall(r'^    "(.*)",$', text, re.M):
        body = body.replace("\\n", "\n").replace('\\"', '"').replace("\\\\", "\\")
        title = re.search(r"^journey (.*)$", body, re.M).group(1)
        steps = []
        for line in body.split("\n"):
            if " | " not in line or line.startswith("#"):
                continue
            name, dwell, glide = [f.strip() for f in line.split("|")[:3]]
            steps.append((name, parse_range(dwell), parse_range(glide)))
        out.append((title, steps))
    return out


def key_of(a, b, glide):
    return "%s|%s|%.0f" % (a, b, glide)


def pairs(inc=INC):
    """Every neighbour pair of every journey (cyclic: the last into the first), the glide the middle of the second's
    range, as dicts with a 'key'."""
    out = []
    for title, steps in journeys(inc):
        n = len(steps)
        for i in range(n):
            a, b = steps[i], steps[(i + 1) % n]
            glide = round(0.5 * (b[2][0] + b[2][1]))
            out.append({"journey": title, "index": i, "from": a[0], "to": b[0], "glide": glide,
                        "key": key_of(a[0], b[0], glide)})
    return out


# ------------------------------------------------------------------------------------------------------ the bank
_BANK = None


def bank():
    """name -> its parameters as the generator writes them (a dict of strings)."""
    global _BANK
    if _BANK is None:
        _BANK = {name: dict(kv.split("=", 1) for kv in params.split()) for name, _g, params, _r in make_presets.build()}
    return _BANK


def param(name, key, default):
    return float(bank()[name].get(key, default))


def register(name):
    """The keys a preset's chord can sound on: the conductor's register, the foundation's octave from C2 (or from
    Lowest), and the second conductor's register when it plays."""
    lo, hi = int(param(name, "brain_low", 48)), int(param(name, "brain_high", 79))
    lo2, hi2 = lo, hi
    if param(name, "brain2_on", 0) >= 0.5:
        iv = int(param(name, "brain2_interval", 7))
        lo2, hi2 = lo + iv, hi + iv
    return max(0, min(lo, lo2, 36)), min(127, max(hi, hi2))


def fifths(r1, r2):
    """Steps round the circle of fifths between two roots (0 .. 6)."""
    d = (7 * (r2 - r1)) % 12
    return min(d, 12 - d)


# ------------------------------------------------------------------------------------------------------- helpers
def low_priority():
    return 0x00004000 if os.name == "nt" else 0


def read_tsv(path, columns):
    """Rows of a TSV written by this tool, as dicts (an absent file: none)."""
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#") or line.startswith(columns[0] + "\t"):
            continue
        f = line.split("\t")
        if len(f) == len(columns):
            rows.append(dict(zip(columns, f)))
        elif columns in (RENDER_COLUMNS, NOTE_COLUMNS) and len(f) == len(columns) - 1:
            rows.append(dict(zip(columns, f + [""])))   # written before the last column was added
    return rows


def append_tsv(path, columns, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists()
    with open(path, "a", encoding="utf-8", newline="\n") as fh:
        if new:
            fh.write("\t".join(columns) + "\n")
        for r in rows:
            fh.write("\t".join(fmt(r[c]) for c in columns) + "\n")


def fmt(v):
    return ("%.4g" % v) if isinstance(v, float) else str(v)


def cents(hz):
    return 1200.0 * math.log2(hz / 440.0)


def folded(d):
    c = math.fmod(abs(d), 1200.0)
    return min(c, 1200.0 - c)


def rubs(x, y):
    """Two pitches in cents a minor second or a minor ninth apart (the conductor's own rub, ClusterBrain rubsOwn)."""
    d = abs(x - y)
    return RUB_LO < d < RUB_HI or 1200.0 + RUB_LO < d < 1200.0 + RUB_HI


def rubs_m7(x, y):
    """The same, or a major seventh (a semitone short of an octave), under two octaves: the interval class."""
    d = abs(x - y)
    return d < 2400.0 and RUB_LO < folded(d) < RUB_HI


# -------------------------------------------------------------------------------------------------------- static
def tunings(names, exe=AUDIT_EXE, jobs=4):
    """name -> 128 pitches (Hz) as the preset tunes its keys; cached in work/journeys/tuning.tsv."""
    path = WORK / "tuning.tsv"
    have = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            f = line.split("\t")
            if len(f) == 129:
                have[f[0]] = [float(x) for x in f[1:]]
    want = [n for n in dict.fromkeys(names) if n not in have]
    if want:
        def run(chunk):
            fd, tmp = tempfile.mkstemp(suffix=".txt", prefix="jt_tuning_")
            with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as fh:
                fh.write("\n".join(chunk) + "\n")
            try:
                r = subprocess.run([str(exe), "--presets", tmp, "--tuning"], capture_output=True, text=True,
                                   encoding="utf-8", errors="replace", creationflags=low_priority())
            finally:
                os.remove(tmp)
            if r.returncode != 0:
                raise RuntimeError("additiv_brain_audit --tuning failed: " + r.stderr[:300])
            got, cur = {}, None
            for line in r.stdout.splitlines():
                if line.startswith("# preset "):
                    cur = line[len("# preset "):].split(" ", 1)[1]
                elif line.startswith("T,") and cur is not None:
                    got[cur] = [float(x) for x in line.split(",")[2:]]
            return got
        chunks = [want[i:i + 200] for i in range(0, len(want), 200)]
        WORK.mkdir(parents=True, exist_ok=True)
        with cf.ThreadPoolExecutor(jobs) as ex, open(path, "a", encoding="utf-8", newline="\n") as fh:
            for got in ex.map(run, chunks):
                for n, hz in got.items():
                    have[n] = hz
                    fh.write(n + "\t" + "\t".join("%.4f" % x for x in hz) + "\n")
    return have


def static_row(p, tune):
    a, b = p["from"], p["to"]
    ra, rb = int(param(a, "root", 2)), int(param(b, "root", 2))
    lo, hi = register(a)
    moves = [abs(1200.0 * math.log2(tune[b][k] / tune[a][k])) for k in range(lo, hi + 1)
             if tune[a][k] > 0 and tune[b][k] > 0]
    mx = max(moves) if moves else 0.0
    return {"key": p["key"], "from": a, "to": b, "root_from": ROOTS[ra], "root_to": ROOTS[rb], "fifths": fifths(ra, rb),
            "scale_from": SCALES[int(param(a, "scale", 3))], "scale_to": SCALES[int(param(b, "scale", 3))],
            "retune_max_c": mx, "retune_mean_c": sum(moves) / len(moves) if moves else 0.0,
            "moved_share": sum(1 for m in moves if m >= COMPATIBLE_C) / len(moves) if moves else 0.0,
            "compatible": 1 if mx < COMPATIBLE_C else 0}


def static_rows(ps, tag):
    tune = tunings([p["from"] for p in ps] + [p["to"] for p in ps])
    rows, seen = [], set()
    for p in ps:
        if p["key"] in seen:
            continue
        seen.add(p["key"])
        rows.append(static_row(p, tune))
    return rows


# --------------------------------------------------------------------------------------------------------- notes
def note_metrics(text):
    """One pair's audit output -> its note measures."""
    at = glide = 0.0
    rel = {0: 5.0, 1: 5.0}
    tables = []                 # (t, [cents of key 0..127])
    events = []                 # (t, source, on, note)
    pins = {}                   # (source, note) -> [(t, cents)]: --hold-pitch, a sounding note kept at its pitch
    for line in text.splitlines():
        if line.startswith("H,"):
            f = line.split(",")
            pins.setdefault((int(f[2]), int(f[3])), []).append((float(f[1]), cents(float(f[4]))))
        elif line.startswith("E,"):
            f = line.split(",")
            events.append((float(f[1]), int(f[2]), f[3] == "1", int(f[4])))
        elif line.startswith("T,"):
            f = line.split(",")
            tables.append((float(f[1]), [cents(float(x)) if float(x) > 0 else -1e9 for x in f[2:]]))
        elif line.startswith("G,"):
            f = line.split(",")
            at, glide = float(f[1]), float(f[2])
        elif line.startswith("P,"):
            f = line.split(",")
            rel[int(f[1])] = float(f[3])
    mid = at + 0.5 * glide
    end = events[-1][0] if events else at + glide
    end = max(end, at + glide + POST - 1.0)
    # notes: (source, note, on, off+tail)
    notes, open_ = [], {}
    for t, src, on, note in events:
        k = (src, note)
        if on:
            open_[k] = t
        elif k in open_:
            t0 = open_.pop(k)
            notes.append((src, note, t0, t + TAIL * rel[0 if t < mid else 1]))
    for (src, note), t0 in open_.items():
        notes.append((src, note, t0, end))

    def table_at(t):
        cur = tables[0][1]
        for tt, tb in tables:
            if tt <= t:
                cur = tb
            else:
                break
        return cur

    def pin_of(n, t):
        """The pitch a note was kept at, if it was by time t (its own pin: one made while it sounded -- or rang: the
        engine keeps a voice a little longer than the 20 dB this tool counts a release to, hence the second's slack)."""
        for tp, c in pins.get((n[0], n[1]), ()):
            if n[2] <= tp < n[3] + 1.0 and tp <= t:
                return c
        return None

    def pitch(n, t, tb=None):
        c = pin_of(n, t)
        return c if c is not None else (tb if tb is not None else table_at(t))[n[1]]
    chord = [n for n in notes if n[0] in (1, 2, 4)]
    figs = [n for n in notes if n[0] == 5]
    out_ = [n for n in chord if n[2] < at and n[3] > at]
    in_ = [n for n in chord if n[2] >= at]
    in_f = [n for n in figs if n[2] >= at]
    points = sorted({at, end} | {n[2] for n in out_ + in_ + in_f if at <= n[2] <= end} |
                    {n[3] for n in out_ + in_ + in_f if at <= n[3] <= end} | {t for t, _ in tables if at <= t <= end})

    def rub_time(group_in, rub=rubs):
        total = 0.0
        for s, e in zip(points, points[1:]):
            if e <= s:
                continue
            m = 0.5 * (s + e)
            tb = table_at(m)
            o = [pitch(n, m, tb) for n in out_ if n[2] <= m < n[3]]
            i = [pitch(n, m, tb) for n in group_in if n[2] <= m < n[3]]
            if any(rub(x, y) for x in o for y in i):
                total += e - s
        return total
    # the retune at the middle: the largest move of a chord note sounding there
    before, after = table_at(mid - 0.05), table_at(mid + 1.0)
    for tt, tb in tables:          # the table the middle switched to (it may be printed a tenth of a second late)
        if mid - 0.05 < tt <= mid + 1.0:
            after = tb
    sounding = [n for n in chord if n[2] <= mid < n[3]]
    moves = [abs(pitch(n, mid + 1.0, after) - pitch(n, mid - 0.05, before)) for n in sounding]
    retune = max(moves, default=0.0)
    return {"rub_s": rub_time(in_), "rub_m7_s": rub_time(in_, rubs_m7), "rub_fig_s": rub_time(in_f), "retune_sounding_c": retune,
            "out_notes": len(out_), "in_notes": len(in_), "chord_mid": len(sounding),
            "moved50": sum(1 for m in moves if m >= COMPATIBLE_C)}


def notes_files(hold):
    """The note audits of one instrument: with Engine::setHoldPitch (hold 1, notes_hold1*.tsv) or as it was before
    20.11.2026 (hold 0, the other notes*.tsv). A pair measured on one says nothing about the other."""
    if hold:
        return sorted(WORK.glob("notes_hold%d*.tsv" % hold))
    return sorted(f for f in WORK.glob("notes*.tsv") if not f.name.startswith("notes_hold"))


def run_note_chunk(chunk, exe, dt, every, hold=1):
    fd, tmp = tempfile.mkstemp(suffix=".txt", prefix="jt_pairs_")
    with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as fh:
        for p in chunk:
            fh.write("%s|%s|%.1f|%.1f|%.1f\n" % (p["from"], p["to"], PRE, p["glide"], PRE + p["glide"] + POST))
    try:
        r = subprocess.run([str(exe), "--pairs", tmp, "--dt", repr(dt), "--param-every", str(every),
                            "--hold-pitch", str(hold)],
                           capture_output=True, text=True, encoding="utf-8", errors="replace",
                           creationflags=low_priority())
    finally:
        os.remove(tmp)
    if r.returncode != 0:
        raise RuntimeError("additiv_brain_audit --pairs failed: " + r.stderr[:300])
    parts = re.split(r"^# pair .*$", r.stdout, flags=re.M)[1:]
    if len(parts) != len(chunk):
        raise RuntimeError("expected %d pairs, got %d" % (len(chunk), len(parts)))
    return [dict(note_metrics(t), key=p["key"]) for p, t in zip(chunk, parts)]


def notes_pass(ps, args):
    hold = "_hold%d" % args.hold_pitch if args.hold_pitch else ""
    path = WORK / ("notes%s%s.tsv" % (hold, args.tag_suffix))
    if args.fresh:   # this run's own file alone
        done = {r["key"] for r in read_tsv(path, NOTE_COLUMNS)}
    else:            # any run's of the same instrument (with the switch or without it: two instruments)
        done = {r["key"] for f in notes_files(args.hold_pitch) for r in read_tsv(f, NOTE_COLUMNS)}
    todo = list({p["key"]: p for p in ps if p["key"] not in done}.values())
    if args.limit:
        todo = todo[:args.limit]
    print("notes: %d pairs to audit (%d done)" % (len(todo), len(done)), flush=True)
    chunks = [todo[i:i + args.chunk] for i in range(0, len(todo), args.chunk)]
    t0, n = time.time(), 0
    with cf.ThreadPoolExecutor(args.jobs) as ex:
        futs = [ex.submit(run_note_chunk, c, AUDIT_EXE, args.dt, args.every, args.hold_pitch) for c in chunks]
        for f in cf.as_completed(futs):
            rows = f.result()
            append_tsv(path, NOTE_COLUMNS, rows)
            n += len(rows)
            el = time.time() - t0
            print("  %d of %d, %.0f s, about %.0f s to go" % (n, len(todo), el, el / n * (len(todo) - n)), flush=True)
    print("notes: %d pairs in %.0f s -> %s" % (n, time.time() - t0, path))


# -------------------------------------------------------------------------------------------------------- render
K1 = ([1.53512485958697, -2.69169618940638, 1.19839281085285], [1.0, -1.69065929318241, 0.73248077421585])
K2 = ([1.0, -2.0, 1.0], [1.0, -1.99004745483398, 0.99007225036621])


def short_term(x, sr, hop=0.5):
    """BS.1770 short-term loudness (3 s windows) every `hop` seconds: (times of the windows' ends, LUFS)."""
    import numpy as np
    from scipy.signal import lfilter
    z = lfilter(K2[0], K2[1], lfilter(K1[0], K1[1], x, axis=0), axis=0)
    p = (z ** 2).sum(axis=1)
    c = np.concatenate([[0.0], np.cumsum(p)])
    w, h = int(3 * sr), int(hop * sr)
    ends = np.arange(w, len(p) + 1, h)
    ms = (c[ends] - c[ends - w]) / w
    return ends / sr, -0.691 + 10.0 * np.log10(np.maximum(ms, 1e-12))


def roughness(x, sr, hop=0.5, n=8192, peaks=40):
    """Vassilakis roughness of the mono mix over its strongest spectral peaks, every `hop` seconds."""
    import numpy as np
    m = x.mean(axis=1)
    win = np.hanning(n)
    times, vals = [], []
    freqs = np.fft.rfftfreq(n, 1.0 / sr)
    for start in range(0, len(m) - n, int(hop * sr)):
        s = np.abs(np.fft.rfft(m[start:start + n] * win))
        loc = np.where((s[1:-1] > s[:-2]) & (s[1:-1] >= s[2:]) & (freqs[1:-1] > 20.0))[0] + 1
        if len(loc) < 2:
            times.append((start + n) / sr)
            vals.append(0.0)
            continue
        loc = loc[np.argsort(s[loc])[-peaks:]]
        f, a = freqs[loc], s[loc] / (n / 4)
        fi, fj = np.meshgrid(f, f)
        ai, aj = np.meshgrid(a, a)
        fmin, df = np.minimum(fi, fj), np.abs(fi - fj)
        amin, amax = np.minimum(ai, aj), np.maximum(ai, aj)
        sv = 0.24 / (0.0207 * fmin + 18.96)
        r = (ai * aj) ** 0.1 * 0.5 * (2 * amin / (amin + amax + 1e-30)) ** 3.11 * \
            (np.exp(-3.5 * sv * df) - np.exp(-5.75 * sv * df))
        vals.append(float(np.triu(r, 1).sum()))
        times.append((start + n) / sr)
    return np.array(times), np.array(vals)


def render_metrics(wav, glide, curves=None):
    import numpy as np
    from scipy.io import wavfile
    sr, x = wavfile.read(str(wav))
    x = x.astype(np.float64)
    if x.dtype.kind == "f" and np.abs(x).max() > 4.0:
        x /= 2.0 ** 31
    if np.abs(x).max() > 4.0:
        x /= 2.0 ** 31 if np.abs(x).max() > 2.0 ** 24 else 2.0 ** 23
    total = len(x) / sr
    t, st = short_term(x, sr)
    tr, rv = roughness(x, sr)
    if curves is not None:   # kept, so the measures can be worked out again without the card
        np.savez_compressed(str(curves), t=t, st=st, tr=tr, rv=rv, glide=glide, total=total)
    return metrics_of(t, st, tr, rv, glide, total)


def alone_corrected(t, st, glide, total, alone):
    """The dip and the peak of a transition counted beyond what either preset does by itself: the floor is the lowest
    of the steady stretches' lowest levels (as metrics_of) and of each preset's own lowest level over the same window
    when played alone; the ceiling likewise. `alone` is a list of (t, st) curves of the presets alone (same clock).
    Returns (dip, peak, the alone curves' lowest and highest level in the window)."""
    import numpy as np
    a0, a1 = RPRE - STEADY, RPRE
    b0, b1 = total - STEADY, total
    sa, sb = st[(t > a0 + 3) & (t <= a1)], st[(t > b0 + 3) & (t <= b1)]
    lo, hi = (t > RPRE), (t <= RPRE + glide + 15.0)
    w = lo & hi
    low = min(float(sa.min()), float(sb.min()))
    high = max(float(sa.max()), float(sb.max()))
    alow, ahigh = float("inf"), float("-inf")
    for at, ast in alone:
        aw = (at > RPRE) & (at <= RPRE + glide + 15.0)
        if aw.any():
            alow, ahigh = min(alow, float(ast[aw].min())), max(ahigh, float(ast[aw].max()))
    dip = max(0.0, min(low, alow) - float(st[w].min()))
    peak = max(0.0, float(st[w].max()) - max(high, ahigh))
    return dip, peak, alow, ahigh


def yardstick_check():
    """Part of --check (plan D2b): a preset that falls silent by itself between its notes is not a hole of the glide.
    Synthetic curves: the first preset at -20 LUFS for its steady stretch, then a 20 s silence at -80 that it has
    alone too; the glide follows the first preset. metrics_of (the steady stretches alone, as D2 counted) calls that a
    hole of 60 LU; counted against the presets alone it is none. And a hole the glide makes itself (-80 where both
    presets alone stay at -20) is still found. Returns the problems, empty when the yardstick holds."""
    import numpy as np
    glide, total = 90.0, RPRE + 90.0 + RPOST
    t = np.arange(3.0, total + 0.01, 0.5)
    own = np.where((t > 60.0) & (t < 80.0), -80.0, -20.0)        # the first preset's own silence
    flat = np.full_like(t, -20.0)
    made = np.where((t > 60.0) & (t < 80.0), -80.0, -20.0)       # a glide that makes the same hole itself
    problems = []
    old = metrics_of(t, own, t, np.ones_like(t), glide, total)["dip_lu"]
    if old < LU_LIMIT:
        problems.append("the steady stretches alone should count the preset's own silence (%.1f LU)" % old)
    dip, _peak, _l, _h = alone_corrected(t, own, glide, total, [(t, own), (t, flat)])
    if dip > LU_LIMIT:
        problems.append("a preset's own silence counted as a hole of the glide: %.1f LU" % dip)
    dip, _peak, _l, _h = alone_corrected(t, made, glide, total, [(t, flat), (t, flat)])
    if dip < 50.0:
        problems.append("a hole the glide makes itself was not found: %.1f LU" % dip)
    return problems


def metrics_of(t, st, tr, rv, glide, total):
    """The measures from the loudness and roughness curves. A dip or a peak is counted beyond what either preset does
    by itself in its steady stretch (its own lowest and highest short-term loudness there): a preset that is struck
    and falls silent between its notes is not a dip of the glide."""
    import numpy as np
    a0, a1 = RPRE - STEADY, RPRE
    b0, b1 = total - STEADY, total
    sa, sb = st[(t > a0 + 3) & (t <= a1)], st[(t > b0 + 3) & (t <= b1)]
    la, lb = float(np.median(sa)), float(np.median(sb))
    w = (t > RPRE) & (t <= RPRE + glide + 15.0)
    dip = max(0.0, min(float(sa.min()), float(sb.min())) - float(st[w].min()))
    peak = max(0.0, float(st[w].max()) - max(float(sa.max()), float(sb.max())))
    ra = float(np.median(rv[(tr > a0) & (tr <= a1)]))
    rb = float(np.median(rv[(tr > b0) & (tr <= b1)]))
    wr = (tr > RPRE) & (tr <= RPRE + glide + 15.0)
    ratio = float(rv[wr].max()) / max(ra, rb, 1e-12)
    return {"level_from": la, "level_to": lb, "dip_lu": dip, "peak_lu": peak, "jump_lu": lb - la, "rough_ratio": ratio,
            "range_lu": max(float(sa.max() - sa.min()), float(sb.max() - sb.min()))}


def render_one(p, out_dir, n):
    """Renders one transition on the card (the caller holds it); returns (the WAV, the seconds it took)."""
    wav = out_dir / ("transition%d.wav" % (n % 4))
    seconds = RPRE + p["glide"] + RPOST
    env = dict(os.environ, ADDITIV_MUTE="1")
    t0 = time.time()
    r = subprocess.run([str(RENDER_EXE), "--preset", p["from"], "--fill", "--then", p["to"], "--at", "%.1f" % RPRE,
                        "--glide", "%.1f" % p["glide"], "--seconds", "%.1f" % seconds, "--log", "60",
                        "--set", "brain_on=1", "--out",
                        str(wav)], capture_output=True, text=True, encoding="utf-8", errors="replace", env=env)
    took = time.time() - t0
    if r.returncode != 0:
        raise RuntimeError("empy_render failed for %s: %s" % (p["key"], (r.stderr or r.stdout)[-300:]))
    return wav, took


def analyse_one(p, wav, took):
    """The measures of a rendered transition (off the card, while the next one renders); the WAV is deleted."""
    curves = WORK / "curves"
    curves.mkdir(parents=True, exist_ok=True)
    m = render_metrics(wav, p["glide"], curves / (re.sub(r"[^A-Za-z0-9]+", "_", p["key"]) + ".npz"))
    wav.unlink()
    return dict(m, key=p["key"], card_s=took)


def sample_pairs(ps, per, seed=7):
    """A stratified sample: `per` pairs of every journey, spread over its round."""
    rng = random.Random(seed)
    by = {}
    for p in ps:
        by.setdefault(p["journey"], []).append(p)
    out = []
    for title, lst in by.items():
        k = min(per, len(lst))
        off = rng.randrange(len(lst))
        out += [lst[(off + i * len(lst) // k) % len(lst)] for i in range(k)]
    return out


def card_sample(ps, nights=2, groups=25, seed=7):
    """The sample the card can afford (about thirty seconds a transition): `nights` pairs of every Whole Night, and
    one pair each of `groups` group journeys and `groups` group nights drawn at random -- chosen by journey and step,
    so a candidate with the same journeys in another order is sampled at the same places."""
    rng = random.Random(seed)
    by = {}
    for p in ps:
        by.setdefault(p["journey"], []).append(p)
    titles = sorted(by)
    whole = [t for t in titles if t.startswith("Whole Night")]
    jour = [t for t in titles if t.endswith("- Journey")]
    nite = [t for t in titles if t.endswith("- Night") and not t.startswith("Whole Night")]
    out = []
    for t in whole:
        n = len(by[t])
        off = rng.randrange(n)
        out += [by[t][(off + i * n // nights) % n] for i in range(nights)]
    for t in rng.sample(jour, min(groups, len(jour))) + rng.sample(nite, min(groups, len(nite))):
        out.append(by[t][rng.randrange(len(by[t]))])
    return out


def render_pass(ps, args):
    sys.path.insert(0, str(GPU_CARD))
    import gpu_card
    path = WORK / ("render%s.tsv" % args.tag_suffix)
    budget_path = WORK / ("card_budget%s.json" % args.tag_suffix)   # each tag its own budget (D2b)
    used = json.loads(budget_path.read_text())["seconds"] if budget_path.exists() else 0.0
    done = {r["key"] for f in WORK.glob("render*.tsv") for r in read_tsv(f, RENDER_COLUMNS)
            if (r["range_lu"] != "" or (float(r["dip_lu"]) <= LU_LIMIT and float(r["peak_lu"]) <= LU_LIMIT))
            and float(r["level_from"]) > -70.0}   # -120: rendered before brain_on=1 was set, the first one silent
    chosen = card_sample(ps, args.sample_nights, args.sample_groups) if args.card_sample else \
        (sample_pairs(ps, args.sample) if args.sample else ps)
    todo = [p for p in chosen if p["key"] not in done]
    todo = list({p["key"]: p for p in todo}.values())
    random.Random(11).shuffle(todo)   # so that a budget that runs out cuts across the strata, not the last of them
    if args.limit:
        todo = todo[:args.limit]
    print("render: %d pairs to render (%d done), card used %.1f of %.0f min" % (len(todo), len(done), used / 60,
                                                                              args.budget_min), flush=True)
    out_dir = WORK / "render_tmp"
    out_dir.mkdir(parents=True, exist_ok=True)
    pool = cf.ThreadPoolExecutor(2)
    pending = []
    count = 0

    def collect(wait):
        for p, f in list(pending):
            if wait or f.done():
                pending.remove((p, f))
                try:
                    row = f.result()
                except Exception as exc:
                    print("  analysis failed for %s: %s" % (p["key"], exc), flush=True)
                    continue
                append_tsv(path, RENDER_COLUMNS, [row])
                print("  %-70s dip %.1f peak %.1f jump %+.1f rough x%.2f (%.0f s)" % (
                    p["key"][:70], row["dip_lu"], row["peak_lu"], row["jump_lu"], row["rough_ratio"],
                    row["card_s"]), flush=True)
    while todo and used < 60.0 * args.budget_min:
        chunk_left = min(60.0 * args.chunk_min, 60.0 * args.budget_min - used)
        if chunk_left < 40.0:
            break   # no render fits in what is left (it took the card and gave it back every 45 s once: 20.11.2026)
        with gpu_card.hold("Empyrean D2 [journey_transitions]: Uebergaenge rendern"):
            t_hold = time.time()
            while todo and time.time() - t_hold + 30.0 < chunk_left:   # a render takes twenty to forty seconds
                gpu_card.wait_while_paused()
                while len(pending) >= 3:   # the WAVs rotate over four files
                    collect(False)
                    time.sleep(0.2)
                p = todo.pop(0)
                try:
                    wav, took = render_one(p, out_dir, count)
                except Exception as exc:   # one bad render does not end the chunk
                    print("  failed: %s" % exc, flush=True)
                    continue
                count += 1
                pending.append((p, pool.submit(analyse_one, p, wav, took)))
                collect(False)
            used += time.time() - t_hold
            budget_path.write_text(json.dumps({"seconds": used}))
        collect(True)
        print("render: chunk done, card used %.1f of %.0f min, %d left" % (used / 60, args.budget_min, len(todo)),
              flush=True)
        if todo and used < 60.0 * args.budget_min:
            time.sleep(args.pause)   # the card back for whoever waits (gpu_card polls every 20 s)


def alone_curve_path(name, seconds):
    return WORK / "curves" / ("alone_%s_%d.npz" % (re.sub(r"[^A-Za-z0-9]+", "_", name), int(round(seconds))))


def render_alone(name, seconds, out_dir, n):
    """Renders one preset alone as a transition's first preset is rendered (the caller holds the card); its short-term
    loudness is kept, the audio is not."""
    import numpy as np
    from scipy.io import wavfile
    wav = out_dir / ("alone%d.wav" % (n % 4))
    env = dict(os.environ, ADDITIV_MUTE="1")
    t0 = time.time()
    r = subprocess.run([str(RENDER_EXE), "--preset", name, "--fill", "--seconds", "%.1f" % seconds, "--log", "60",
                        "--set", "brain_on=1", "--out", str(wav)], capture_output=True, text=True, encoding="utf-8",
                       errors="replace", env=env, creationflags=low_priority())
    if r.returncode != 0:
        raise RuntimeError("empy_render failed for %s: %s" % (name, (r.stderr or r.stdout)[-300:]))
    took = time.time() - t0
    sr, x = wavfile.read(str(wav))
    x = x.astype(np.float64)
    if np.abs(x).max() > 4.0:   # as render_metrics reads it
        x /= 2.0 ** 31
    if np.abs(x).max() > 4.0:
        x /= 2.0 ** 31 if np.abs(x).max() > 2.0 ** 24 else 2.0 ** 23
    t, st = short_term(x, sr)
    np.savez_compressed(str(alone_curve_path(name, seconds)), t=t, st=st)
    wav.unlink()
    return took


def alone_rows(ps):
    """For every rendered pair whose two presets have their alone curves: the corrected dip and peak."""
    import numpy as np
    rd = rendered()
    rows = []
    for p in {p["key"]: p for p in ps}.values():
        r = rd.get(p["key"])
        cur = WORK / "curves" / (re.sub(r"[^A-Za-z0-9]+", "_", p["key"]) + ".npz")
        if r is None or not cur.exists():
            continue
        d = np.load(str(cur))
        total = float(d["total"])
        alone = []
        for name in (p["from"], p["to"]):
            f = alone_curve_path(name, total)
            if not f.exists():
                break
            a = np.load(str(f))
            alone.append((a["t"], a["st"]))
        if len(alone) < 2:
            continue
        dip, peak, lo, hi = alone_corrected(d["t"], d["st"], float(d["glide"]), total, alone)
        rows.append({"key": p["key"], "dip_alone_lu": dip, "peak_alone_lu": peak, "alone_low_lu": lo,
                     "alone_high_lu": hi})
    return rows


def rendered():
    """key -> the render row any run measured (of two, the one measured against the extremes)."""
    rd = {}
    for f in sorted(WORK.glob("render*.tsv")):
        for r in read_tsv(f, RENDER_COLUMNS):
            if r["key"] not in rd or r["range_lu"] != "" or rd[r["key"]]["range_lu"] == "":
                rd[r["key"]] = r
    return rd


def alone_pass(ps, args):
    """The presets of every rendered pair over the level limit, alone (see the file comment); then alone.tsv."""
    sys.path.insert(0, str(GPU_CARD))
    import gpu_card
    chosen = card_sample(ps, args.sample_nights, args.sample_groups) if args.card_sample else \
        (sample_pairs(ps, args.sample) if args.sample else ps)
    rd = rendered()
    want = {}
    for p in chosen:
        r = rd.get(p["key"])
        if r is None or not (float(r["dip_lu"]) > LU_LIMIT or float(r["peak_lu"]) > LU_LIMIT):
            continue
        seconds = RPRE + p["glide"] + RPOST
        for name in (p["from"], p["to"]):
            if not alone_curve_path(name, seconds).exists():
                want[(name, round(seconds, 1))] = seconds
    todo = list(want.items())
    budget_path = WORK / ("card_budget%s.json" % args.tag_suffix)
    used = json.loads(budget_path.read_text())["seconds"] if budget_path.exists() else 0.0
    print("alone: %d presets to render alone, card used %.1f of %.0f min" % (len(todo), used / 60, args.budget_min),
          flush=True)
    out_dir = WORK / "render_tmp"
    out_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    while todo and used < 60.0 * args.budget_min:
        chunk_left = min(60.0 * args.chunk_min, 60.0 * args.budget_min - used)
        if chunk_left < 40.0:
            break   # no render fits in what is left
        with gpu_card.hold("Empyrean D2b [journey_transitions]: Presets allein rendern"):
            t_hold = time.time()
            while todo and time.time() - t_hold + 30.0 < chunk_left:
                gpu_card.wait_while_paused()
                (name, _s), seconds = todo.pop(0)
                try:
                    took = render_alone(name, seconds, out_dir, count)
                except Exception as exc:
                    print("  failed: %s" % exc, flush=True)
                    continue
                count += 1
                print("  alone: %-40s %.0f s (%.0f s)" % (name, seconds, took), flush=True)
            used += time.time() - t_hold
            budget_path.write_text(json.dumps({"seconds": used}))
        print("alone: chunk done, card used %.1f of %.0f min, %d left" % (used / 60, args.budget_min, len(todo)),
              flush=True)
        if todo and used < 60.0 * args.budget_min:
            time.sleep(args.pause)
    rows = alone_rows(ps)
    path = WORK / "alone.tsv"
    if path.exists():
        path.unlink()
    append_tsv(path, ALONE_COLUMNS, rows)
    print("alone: %d pairs with both presets alone -> %s" % (len(rows), path))


# -------------------------------------------------------------------------------------------------------- report
def joined(ps, tag_suffix="", hold=1):
    """The pairs with whatever any run measured of them (the runs share keys: a pair is the same pair in any file),
    the notes of the instrument `hold` asks for (notes_files); of a render, a row measured against the extremes wins
    over one measured against the medians."""
    st, nt, rd = {}, {}, {}
    for f in sorted(WORK.glob("static*.tsv")):
        st.update({r["key"]: r for r in read_tsv(f, STATIC_COLUMNS)})
    for f in notes_files(hold):
        nt.update({r["key"]: r for r in read_tsv(f, NOTE_COLUMNS)})
    for f in sorted(WORK.glob("render*.tsv")):
        for r in read_tsv(f, RENDER_COLUMNS):
            if r["key"] not in rd or r["range_lu"] != "" or rd[r["key"]]["range_lu"] == "":
                rd[r["key"]] = r
    al = {r["key"]: r for r in read_tsv(WORK / "alone.tsv", ALONE_COLUMNS)}
    rows = []
    for p in ps:
        r = dict(p)
        for d in (st.get(p["key"]), nt.get(p["key"]), rd.get(p["key"]), al.get(p["key"])):
            if d:
                r.update({k: v for k, v in d.items() if k not in ("from", "to")})
        rows.append(r)
    return rows


def broken(r):
    """The plan's acceptance: a level dip or peak over 3 LU, or more than 5 s of semitone rubbing (None: unknown)."""
    lvl = None
    if "dip_lu" in r:
        lvl = float(r["dip_lu"]) > LU_LIMIT or float(r["peak_lu"]) > LU_LIMIT
        if lvl and "dip_alone_lu" in r:   # counted again against the presets alone (plan D2b): that verdict holds
            lvl = float(r["dip_alone_lu"]) > LU_LIMIT or float(r["peak_alone_lu"]) > LU_LIMIT
        if lvl and r.get("range_lu", "") == "":
            lvl = None   # an upper bound (measured against the medians) over the limit: not known
        if float(r["level_from"]) < -70.0:
            lvl = None   # the first preset never sounded (rendered before brain_on=1 was set): not known
    rub = float(r["rub_s"]) > RUB_LIMIT if "rub_s" in r else None
    return lvl, rub


def report(ps, tag_suffix="", hold=1):
    rows = joined(ps, tag_suffix, hold)
    cols = PAIR_COLUMNS + [c for c in STATIC_COLUMNS[3:]] + NOTE_COLUMNS[1:] + RENDER_COLUMNS[1:] + ALONE_COLUMNS[1:]
    path = WORK / ("transitions%s.tsv" % tag_suffix)
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\t".join(cols) + "\n")
        for r in rows:
            fh.write("\t".join(str(r.get(c, "")) for c in cols) + "\n")
    n = len(rows)
    stat = [r for r in rows if "compatible" in r]
    nts = [r for r in rows if "rub_s" in r]
    rds = [r for r in rows if "dip_lu" in r and broken(r)[0] is not None]
    print("%d transitions -> %s (notes %s)" % (n, path, "with sounding notes kept at their pitch (--hold-pitch 1)"
                                               if hold else "as before 20.11.2026, retuned (--hold-pitch 0)"))
    if stat:
        inc = sum(1 for r in stat if r["compatible"] == "0")
        print("  static: %d measured; tunings incompatible (a key of the register moves >= %.0f c): %d (%.1f %%); "
              "root distance in fifths: %s" % (len(stat), COMPATIBLE_C, inc, 100.0 * inc / len(stat),
                                               " ".join("%d:%d" % (f, sum(1 for r in stat if int(r["fifths"]) == f))
                                                        for f in range(7))))
    if nts:
        bad = sum(1 for r in nts if float(r["rub_s"]) > RUB_LIMIT)
        print("  notes: %d audited; more than %.0f s of semitone rubbing between the outgoing and the incoming chord: "
              "%d (%.1f %%); mean %.1f s" % (len(nts), RUB_LIMIT, bad, 100.0 * bad / len(nts),
                                             sum(float(r["rub_s"]) for r in nts) / len(nts)))
    if rds:
        bad = sum(1 for r in rds if broken(r)[0])
        print("  render: %d rendered; a dip or peak over %.0f LU: %d (%.1f %%)" % (len(rds), LU_LIMIT, bad,
                                                                                   100.0 * bad / len(rds)))
    both = [r for r in rows if "rub_s" in r and "dip_lu" in r and broken(r)[0] is not None]
    if both:
        bad = sum(1 for r in both if any(broken(r)))
        print("  rendered and audited: %d; broken (level over %.0f LU or rubbing over %.0f s): %d (%.1f %%)" % (
            len(both), LU_LIMIT, RUB_LIMIT, bad, 100.0 * bad / len(both)))
    return rows


def export_bad(path, hold=1):
    """Every pair measured as broken, in any of the runs of the instrument `hold` asks for (the shipped journeys' and
    the candidates'), into the list make_journeys.py reads (BAD_PAIRS there): from, to, why -- unordered for
    make_journeys, written as measured. A level counts against both presets alone where that was measured (D2b)."""
    bad = {}
    al = {r["key"]: r for r in read_tsv(WORK / "alone.tsv", ALONE_COLUMNS)}
    for f in notes_files(hold):
        for r in read_tsv(f, NOTE_COLUMNS):
            if float(r["rub_s"]) > RUB_LIMIT:
                a, b, _g = r["key"].split("|")
                bad.setdefault((a, b), set()).add("rub %.0f s" % float(r["rub_s"]))
    for f in sorted(WORK.glob("render*.tsv")):
        for r in read_tsv(f, RENDER_COLUMNS):
            if r["range_lu"] == "":
                continue   # measured against the medians (before 20.11.2026, 9 a.m.): an upper bound, rendered again
            dip, peak = float(r["dip_lu"]), float(r["peak_lu"])
            if r["key"] in al:   # counted again against both presets alone (plan D2b): that verdict holds
                dip, peak = float(al[r["key"]]["dip_alone_lu"]), float(al[r["key"]]["peak_alone_lu"])
            if dip > LU_LIMIT or peak > LU_LIMIT:
                a, b, _g = r["key"].split("|")
                bad.setdefault((a, b), set()).add("level dip %.1f peak %.1f LU" % (dip, peak))
    lines = ["# Pairs of presets measured as broken where a journey glides from the first into the second "
             "(Tools/presets/journey_transitions.py export-bad): more than %.0f s of a minor second between the "
             "outgoing and the incoming chord (additiv_brain_audit --pairs --hold-pitch %d, every pair of the "
             "journeys), or a level dip or peak over %.0f LU beyond both presets alone (empy_render --then, a "
             "sample). make_journeys.py keeps them apart." % (RUB_LIMIT, hold, LU_LIMIT), "from\tto\twhy"]
    for (a, b), why in sorted(bad.items()):
        lines.append("%s\t%s\t%s" % (a, b, "; ".join(sorted(why))))
    pathlib.Path(path).write_bytes(("\n".join(lines) + "\n").encode("utf-8"))
    print("export-bad: %d pairs -> %s" % (len(bad), path))


# --------------------------------------------------------------------------------------------------------- check
def check(inc, max_fifths, max_bad_share):
    """The ctest `journeys` (no card, no program: the journeys file, the bank's parameters and the list of measured
    faults): exit 1 when two neighbours of one scale stand on roots more than `max_fifths` fifths apart (by default
    none a tritone apart: the root step that rubbed most, in a third of the glides), or when more than
    `max_bad_share` of the neighbour pairs are ones measured as broken (Tools/presets/journey_bad_pairs.tsv, in the
    direction the journey glides). Not every broken pair can be avoided without leaving the presets' nearest neighbours (make_journeys.py
    weighs the two), so the second is a ceiling that the shipped journeys keep under, not a zero."""
    import make_journeys
    table = make_journeys.param_table()
    tun = {n: make_journeys.tuning_of(table, p) for n, p in bank().items()}
    listed = set()   # as measured: from, to (a glide is not the same either way round)
    if make_journeys.BAD_PAIRS.exists():
        for line in make_journeys.BAD_PAIRS.read_text(encoding="utf-8").splitlines():
            if line and not line.startswith("#") and not line.startswith("from\t"):
                a, b = line.split("\t")[:2]
                listed.add((a, b))
    ps = pairs(inc)
    far, broken_ = [], []
    for p in ps:
        ta, tb = tun[p["from"]], tun[p["to"]]
        f = fifths(ta[1], tb[1]) if ta[0] == tb[0] and ta[2:] == tb[2:] and ta[1] >= 0 else 0
        if f > max_fifths:
            far.append((p, "roots %d fifths apart in %s" % (f, SCALES[ta[0]])))
        if (p["from"], p["to"]) in listed:
            broken_.append((p, "measured as broken"))
    share = len(broken_) / max(1, len(ps))
    yard = yardstick_check()
    for why in yard:
        print("journey_transitions --check: the level's yardstick: %s" % why)
    print("journey_transitions --check: %d neighbour pairs of %d journeys; roots more than %d fifths apart in one "
          "scale: %d (none allowed); measured as broken: %d = %.1f %% (%d pairs listed; at most %.1f %% allowed)" % (
              len(ps), len({p["journey"] for p in ps}), max_fifths, len(far), len(broken_), 100.0 * share,
              len(listed), 100.0 * max_bad_share))
    shown = far[:8] + (broken_[:8] if share > max_bad_share else [])
    for p, why in shown:
        print("  %s, step %d: %s -> %s: %s" % (p["journey"], p["index"] + 1, p["from"], p["to"], why))
    return 1 if far or share > max_bad_share or yard else 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("what", nargs="?", choices=["pairs", "static", "notes", "render", "alone", "report",
                                                      "export-bad"])
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--inc", default=str(INC))
    ap.add_argument("--tag", default="")
    ap.add_argument("--jobs", type=int, default=4)   # at most four: the machine is shared
    ap.add_argument("--chunk", type=int, default=20)
    ap.add_argument("--dt", type=float, default=128 / 48000.0)
    ap.add_argument("--every", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--only", default="", help="a file of pair keys (from|to|glide), one a line: measure those alone")
    ap.add_argument("--sample", type=int, default=0, help="render: this many pairs of every journey")
    ap.add_argument("--card-sample", action="store_true",
                    help="render: the sample the card affords (card_sample: every Whole Night, some groups)")
    ap.add_argument("--sample-nights", type=int, default=2)
    ap.add_argument("--sample-groups", type=int, default=25)
    ap.add_argument("--budget-min", type=float, default=90.0)
    ap.add_argument("--chunk-min", type=float, default=15.0)
    ap.add_argument("--pause", type=float, default=45.0)
    ap.add_argument("--max-fifths", type=int, default=MAX_FIFTHS)
    ap.add_argument("--max-bad-share", type=float, default=MAX_BAD_SHARE)
    ap.add_argument("--hold-pitch", type=int, default=1,
                    help="notes, report, export-bad: Engine::setHoldPitch, 1 as in the instrument (a sounding note "
                         "keeps its pitch), 0 as before 20.11.2026; each has its own TSVs")
    ap.add_argument("--fresh", action="store_true", help="notes: skip only the pairs this run's own file has")
    args = ap.parse_args()
    args.tag_suffix = ("_" + args.tag) if args.tag else ""
    if args.check:
        sys.exit(check(args.inc, args.max_fifths, args.max_bad_share))
    ps = pairs(args.inc)
    if args.only:
        keys = set(pathlib.Path(args.only).read_text(encoding="utf-8").split("\n"))
        ps = [p for p in ps if p["key"] in keys]
    if args.what == "pairs":
        print("%d pairs in %d journeys, %d distinct" % (len(ps), len({p["journey"] for p in ps}),
                                                       len({p["key"] for p in ps})))
    elif args.what == "static":
        t0 = time.time()
        rows = static_rows(ps, args.tag)
        path = WORK / ("static%s.tsv" % args.tag_suffix)
        if path.exists():
            path.unlink()
        append_tsv(path, STATIC_COLUMNS, rows)
        print("static: %d pairs in %.0f s -> %s" % (len(rows), time.time() - t0, path))
    elif args.what == "notes":
        notes_pass(ps, args)
    elif args.what == "render":
        render_pass(ps, args)
    elif args.what == "alone":
        alone_pass(ps, args)
    elif args.what == "report":
        report(ps, args.tag_suffix, args.hold_pitch)
    elif args.what == "export-bad":
        export_bad(ROOT / "Tools" / "presets" / "journey_bad_pairs.tsv", args.hold_pitch)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
