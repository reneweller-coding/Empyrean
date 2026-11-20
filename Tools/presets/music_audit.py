"""The music audit: the faults found by ear in November 2026, counted over the conductors and the figures.

Every measure here began as something Rene heard and nobody had measured (19./20.11.2026): two conductors
that did not hear each other, a conductor that did not check its own notes, the second conductor doubling the
first one's note, notes struck again a few seconds after they were let go, notes released before their own
attack had swelled in, and figures striking the pitch they had just struck. Each was found by a throw-away
script; this is the one tool that keeps measuring them, and `--check` is the ctest `music_audit` that goes red
when one of them comes back. It is Noctuary's Tools/library/music_audit.py (1f83564), on Empyrean's bank and
with the figures.

`additiv_brain_audit` (Engine::auditConductor) steps the engine's conductors, foundation and figures without
the graphics card, with the parameters as a render reads them, and prints every note with the frequency the
tuning gives it. Everything below is measured in cents of that frequency -- never in MIDI keys, because the
same fault came three times from code that compared keys where the tuning decides what sounds.

Per preset (a preset whose conductor is off is skipped), over both conductors (sources 1 and 2):

  own_rub           share of the time a conductor sounds during which two of its own notes are a minor second
                    or minor ninth apart (25..150 or 1225..1350 cents)
  cross_rub         share of the time both sound during which a note of one and a note of the other, less than
                    two octaves apart, are a semitone, major seventh or tritone apart (interval class 25..150 or
                    550..650 cents)
  doubling          onsets within 25 cents of a pitch the other conductor holds at that moment
  restrike          onsets within 25 cents of a pitch the same conductor let go less than 30 s before
  self_unison       onsets within 25 cents of a pitch the same conductor still holds
  short             notes held for less than the preset's own attack (they never reach their level)
  changes_per_hour  onsets of both conductors per hour
  silence_s         seconds with no conductor note sounding, after the first note
  one_note          conductors that play a single pitch for the whole run

and of the figures (source 5):

  figure_ons                  their onsets
  figure_repeat_in_round      onsets within 25 cents of the pitch of the note before them in the same round of the
                              same figure (what Engine::figureRepeats() counts)
  figure_same_as_previous     onsets within 25 cents of the pitch of the previous figure onset of any figure (the
                              rendered survey of 20.11.2026: 121 of 1162)
  figure_rub                  onsets a semitone, a major seventh or a tritone (in any octave, 25..150 or 550..600
                              cents folded, as Engine::rubs) from a sounding conductor note, and not themselves a
                              chord tone (within 25 cents of one)
  figure_shorter_than_attack  figure notes held for less than twice the preset's figure_attack

What the audit cannot see (see Engine::auditConductor): Follow Light over a living field (no preset of the
bank uses Follow Light), Night Arc (the real night pushes the conductor's rate and density; the audit has no
night), and modulation routes from what the instrument hears (they read silence). `--check` and `--update`
step the engine as a render does (one GPU block), but read a knob that the matrix moves and the notes hear every
8 blocks (a render reads it every block; in the Timbre scale that rebuilds the scale each time, and a single such
preset took minutes); `--all` and `--sample` take `--dt` (default 8 GPU blocks, 21 ms) and `--param-every` (4
steps): about eight times faster, and no longer note for note.

Usage:

  music_audit.py --sample 200 --seed 1911          a deterministic sample of the bank
  music_audit.py --all [--estimate]                the whole bank (times the first 200, prints an estimate;
                                                   --estimate stops there)
  music_audit.py --check Tests/golden/music_audit.json    the stored sample against its bands; exit 1 outside
  music_audit.py --update [--golden FILE]          measure the golden sample afresh and write the file

The audit binary is found next to the build (`--exe` overrides it). Workers run at below-normal priority, each
auditing a list of presets in one process (`additiv_brain_audit --presets`). Writes a TSV (one row per preset,
`--tsv`, under work/ by default, which git ignores) and prints a summary, the totals split by Chords/Free and by
scale, and the worst presets per measure. Exit codes: 0 fine, 1 a measure outside its band, 2 a usage or run
error.
"""
import argparse
import concurrent.futures as cf
import json
import math
import os
import random
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
import make_presets  # noqa: E402

GOLDEN = os.path.join(ROOT, "Tests", "golden", "music_audit.json")
EXES = [os.path.join(ROOT, "build", "msvc", "Release", "additiv_brain_audit.exe"),
        os.path.join(ROOT, "bin", "msvc", "additiv_brain_audit.exe")]
GOLDEN_SEED = 2011
GOLDEN_N = 200
BLOCK = 128.0 / 48000.0   # one GPU block, the render's step
GOLDEN_EVERY = 8          # the golden reads a modulated knob the notes hear every 8 blocks (21 ms)
REST_S = 30.0             # the rest after letting a pitch go (ClusterBrain::resting's floor)
SAME_C = 25.0             # two notes within this many cents are one pitch (as ClusterBrain::againstOther)
ROUND_C = 0.1             # the audit's Hz are printed to a thousandth (%.3f): at 20 Hz that moves a difference by up
                          # to 0.09 cents. Two keys exactly 25 cents apart (Purity 0.75 puts neighbouring keys there)
                          # are two pitches to the engine (its test is < 25.0 on the unrounded Hz) and must stay two here


def one_pitch(a, b):
    """Whether two pitches in cents are one, as the engine judges them (< SAME_C), less the printed Hz's rounding."""
    return abs(a - b) < SAME_C - ROUND_C
SCALES = ["12-TET", "JI Major", "JI Minor", "JI 7-limit", "Pythagorean", "JI Pentatonic", "Harmonic 8-16",
          "Subharmonic 16-8", "Slendro", "Bohlen-Pierce", "Otonality 1-11", "User", "Timbre"]
COLUMNS = ["name", "group", "mode", "scale", "attack", "fig_attack", "brain2", "ons", "notes", "restrike",
           "doubling", "short", "self_unison", "own_rub", "cross_rub", "changes_per_hour", "silence_s", "one_note",
           "figure_ons", "figure_repeat_in_round", "figure_same_as_previous", "figure_rub",
           "figure_shorter_than_attack", "sound_s", "both_s"]
FAULTS = ["restrike", "doubling", "short", "self_unison", "own_rub", "cross_rub", "silence_s",
          "figure_repeat_in_round", "figure_same_as_previous", "figure_rub", "figure_shorter_than_attack"]


def load_bank():
    """Every preset whose conductor plays: name -> its facts (group, mode, scale, second conductor)."""
    bank = {}
    for name, g, params, _rows in make_presets.build():
        p = dict(kv.split("=", 1) for kv in params.split())
        if float(p.get("brain_on", 1)) < 0.5:
            continue
        bank[name] = {"group": make_presets.GROUPS[g],
                      "mode": "Chords" if int(float(p.get("auto_mode", 1))) == 1 else "Free",
                      "scale": SCALES[int(float(p.get("scale", 3)))],
                      "brain2": 1 if float(p.get("brain2_on", 0)) >= 0.5 else 0}
    return bank


def cents(hz):
    """Cents above A 440: the pitch as the tuning plays it."""
    return 1200.0 * math.log2(hz / 440.0)


def folded(d):
    """An interval in cents folded into 0 .. 600 (any octave)."""
    c = math.fmod(abs(d), 1200.0)
    return min(c, 1200.0 - c)


def own_rubs(cs):
    """True when two of one conductor's notes are a minor second or ninth apart (cents list)."""
    for i in range(len(cs)):
        for j in range(i + 1, len(cs)):
            d = abs(cs[i] - cs[j])
            if 25.0 < d < 150.0 or 1225.0 < d < 1350.0:
                return True
    return False


def cross_rubs(a, b):
    """True when a note of one and a note of the other, under two octaves apart, rub (cents lists)."""
    for x in a:
        for y in b:
            d = abs(x - y)
            if d >= 2400.0:
                continue
            c = folded(d)
            if 25.0 < c < 150.0 or 550.0 < c < 650.0:
                return True
    return False


def measure(name, facts, attack, fig_attack, lines, hours):
    """One preset's row from its E lines."""
    end = hours * 3600.0
    held = {1: {}, 2: {}}      # conductor -> MIDI note -> (on time, cents)
    offat = {1: [], 2: []}     # conductor -> [(off time, cents)] of the last REST_S seconds
    pitches = {1: set(), 2: set()}
    ons = notes = restrike = doubling = short = self_unison = 0
    sound = {1: 0.0, 2: 0.0}
    own = {1: 0.0, 2: 0.0}
    both = cross = silence = 0.0
    state_own = {1: False, 2: False}
    state_cross = False
    first = None
    last_t = 0.0
    fig_on = {}                # note -> [on times] of figure notes sounding
    fig_last = {}              # figure -> (step, cents) of its last onset
    fig_prev = None            # cents of the last figure onset of any figure
    f_ons = f_round = f_prev = f_rub = f_short = 0

    def advance(t):
        nonlocal both, cross, silence
        dt = t - last_t
        if dt <= 0.0:
            return
        for w in (1, 2):
            if held[w]:
                sound[w] += dt
                if state_own[w]:
                    own[w] += dt
        if held[1] and held[2]:
            both += dt
            if state_cross:
                cross += dt
        if first is not None and not held[1] and not held[2]:
            silence += dt

    def close(w, note, t):
        nonlocal notes, short
        on_t, c = held[w].pop(note)
        notes += 1
        if t - on_t < attack:
            short += 1
        offat[w].append((t, c))

    for line in lines:
        if not line.startswith("E,"):
            continue
        _, ts, ws, on, note, _vel, hzs, figs, steps = line.split(",")
        t, w, note, hz = float(ts), int(ws), int(note), float(hzs)
        if hz <= 0.0:
            continue
        c = cents(hz)
        if w == 5:
            if on == "1":
                f_ons += 1
                fig, step = int(figs), int(steps)
                last = fig_last.get(fig)
                if last is not None and step > last[0] and one_pitch(c, last[1]):
                    f_round += 1
                fig_last[fig] = (step, c)
                if fig_prev is not None and one_pitch(c, fig_prev):
                    f_prev += 1
                fig_prev = c
                chord = [x[1] for x in held[1].values()] + [x[1] for x in held[2].values()]
                tone = any(folded(c - x) <= SAME_C for x in chord)
                rub = any(25.0 < folded(c - x) < 150.0 or folded(c - x) > 550.0 for x in chord)
                if rub and not tone:
                    f_rub += 1
                fig_on.setdefault(note, []).append(t)
            elif fig_on.get(note):
                if t - fig_on[note].pop(0) < 2.0 * fig_attack:
                    f_short += 1
            continue
        if w not in (1, 2):
            continue
        advance(t)
        last_t = t
        if on == "1":
            ons += 1
            if first is None:
                first = t
            o = 3 - w
            if any(one_pitch(c, x[1]) for x in held[o].values()):
                doubling += 1
            offat[w] = [x for x in offat[w] if t - x[0] < REST_S]
            if any(one_pitch(c, x[1]) for x in offat[w]):
                restrike += 1
            if any(one_pitch(c, x[1]) for x in held[w].values()):
                self_unison += 1
            if note in held[w]:
                close(w, note, t)
            held[w][note] = (t, c)
            pitches[w].add(round(c))
        elif note in held[w]:
            close(w, note, t)
        state_own[1] = own_rubs([x[1] for x in held[1].values()])
        state_own[2] = own_rubs([x[1] for x in held[2].values()])
        state_cross = bool(held[1] and held[2]) and cross_rubs([x[1] for x in held[1].values()],
                                                               [x[1] for x in held[2].values()])
    advance(end)
    snd = sound[1] + sound[2]
    one = sum(1 for w in (1, 2) if len(pitches[w]) == 1)
    return {"name": name, "group": facts["group"], "mode": facts["mode"], "scale": facts["scale"],
            "attack": attack, "fig_attack": fig_attack, "brain2": facts["brain2"],
            "ons": ons, "notes": notes, "restrike": restrike, "doubling": doubling, "short": short,
            "self_unison": self_unison,
            "own_rub": (own[1] + own[2]) / snd if snd > 0 else 0.0,
            "cross_rub": cross / both if both > 0 else 0.0,
            "changes_per_hour": ons / hours, "silence_s": silence, "one_note": one,
            "figure_ons": f_ons, "figure_repeat_in_round": f_round, "figure_same_as_previous": f_prev,
            "figure_rub": f_rub, "figure_shorter_than_attack": f_short,
            "sound_s": snd, "both_s": both, "own_s": own[1] + own[2], "cross_s": cross}


def low_priority():
    """Creation flags that start a child at below-normal priority on Windows (0 elsewhere)."""
    return 0x00004000 if os.name == "nt" else 0


def run_chunk(exe, names, bank, hours, dt, every):
    """Audits a list of presets in one additiv_brain_audit process; returns their rows."""
    fd, path = tempfile.mkstemp(suffix=".txt", prefix="music_audit_")
    with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(names) + "\n")
    try:
        r = subprocess.run([exe, "--presets", path, "--hours", str(hours), "--dt", repr(dt), "--param-every",
                            str(every)], capture_output=True, text=True, encoding="utf-8", errors="replace",
                           creationflags=low_priority())
    finally:
        os.remove(path)
    if r.returncode != 0:
        raise RuntimeError("additiv_brain_audit failed (%d): %s" % (r.returncode, r.stderr.strip()[:400]))
    rows, cur, attack, fig_attack, lines = [], None, 6.0, 0.08, []
    for line in r.stdout.splitlines() + ["# preset -2 <end>"]:
        if line.startswith("# preset "):
            if cur is not None:
                rows.append(measure(cur, bank[cur], attack, fig_attack, lines, hours))
            cur, lines = line[len("# preset "):].split(" ", 1)[1], []
            if cur == "<end>":
                break
        elif line.startswith("A,"):
            _, a, fa = line.split(",")
            attack, fig_attack = float(a), float(fa)
        else:
            lines.append(line)
    if len(rows) != len(names):
        raise RuntimeError("expected %d presets from additiv_brain_audit, got %d" % (len(names), len(rows)))
    return rows


def audit(exe, names, bank, hours, dt, every, jobs, chunk, progress=False):
    """All the presets named, in chunks over at most `jobs` processes; rows in the order given."""
    chunks = [names[i:i + chunk] for i in range(0, len(names), chunk)]
    out = [None] * len(chunks)
    done = 0
    t0 = time.time()
    with cf.ThreadPoolExecutor(max(1, jobs)) as ex:
        futs = {ex.submit(run_chunk, exe, c, bank, hours, dt, every): i for i, c in enumerate(chunks)}
        for f in cf.as_completed(futs):
            out[futs[f]] = f.result()
            done += len(chunks[futs[f]])
            if progress and done % (10 * chunk) < chunk:
                el = time.time() - t0
                print("  %d of %d presets, %.0f s, about %.0f s to go" % (done, len(names), el,
                      el / done * (len(names) - done)), flush=True)
    return [r for c in out for r in c]


def totals(rows):
    """The sums over a set of rows: counts summed, shares weighted by time."""
    snd = sum(r["sound_s"] for r in rows)
    both = sum(r["both_s"] for r in rows)
    t = {"presets": len(rows)}
    for k in ("ons", "notes", "restrike", "doubling", "short", "self_unison", "one_note", "figure_ons",
              "figure_repeat_in_round", "figure_same_as_previous", "figure_rub", "figure_shorter_than_attack"):
        t[k] = sum(r[k] for r in rows)
    t["own_rub"] = sum(r["own_s"] for r in rows) / snd if snd else 0.0
    t["cross_rub"] = sum(r["cross_s"] for r in rows) / both if both else 0.0
    t["changes_per_hour"] = sum(r["changes_per_hour"] for r in rows) / max(1, len(rows))
    t["silence_s"] = sum(r["silence_s"] for r in rows)
    t["mute"] = sum(1 for r in rows if r["ons"] == 0)
    return t


def write_tsv(rows, path):
    """One row per preset."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\t".join(COLUMNS) + "\n")
        for r in rows:
            fh.write("\t".join(("%.4f" % r[c]) if isinstance(r[c], float) else str(r[c]) for c in COLUMNS) + "\n")


def line_of(t):
    """The totals of a group of presets on one line."""
    ons, fo = max(1, t["ons"]), max(1, t["figure_ons"])
    return ("%5d presets  %7d onsets  own_rub %.2f %%  cross_rub %.2f %%  doubling %d  restrike %d  short %d  "
            "self_unison %d  %.1f/h  one_note %d  | figures %d: in round %d, same as previous %d (%.1f %%), rub %d, "
            "short %d" % (t["presets"], t["ons"], 100 * t["own_rub"], 100 * t["cross_rub"], t["doubling"],
                          t["restrike"], t["short"], t["self_unison"], t["changes_per_hour"], t["one_note"],
                          t["figure_ons"], t["figure_repeat_in_round"], t["figure_same_as_previous"],
                          100.0 * t["figure_same_as_previous"] / fo, t["figure_rub"],
                          t["figure_shorter_than_attack"])) if ons else ""


def summary(rows, worst=10):
    """The readable totals, the splits, and the worst presets per measure."""
    t = totals(rows)
    ons, notes, fo = max(1, t["ons"]), max(1, t["notes"]), max(1, t["figure_ons"])
    print("presets %d, conductor onsets %d, notes %d, figure onsets %d" % (t["presets"], t["ons"], t["notes"],
                                                                         t["figure_ons"]))
    print("  own_rub    %6.2f %% of the time a conductor sounds" % (100 * t["own_rub"]))
    print("  cross_rub  %6.2f %% of the time both sound" % (100 * t["cross_rub"]))
    print("  doubling   %6d onsets  (%.2f %%)" % (t["doubling"], 100.0 * t["doubling"] / ons))
    print("  restrike   %6d onsets  (%.2f %%)" % (t["restrike"], 100.0 * t["restrike"] / ons))
    print("  short      %6d notes   (%.2f %%)" % (t["short"], 100.0 * t["short"] / notes))
    print("  self_unison %5d onsets  (%.2f %%)" % (t["self_unison"], 100.0 * t["self_unison"] / ons))
    print("  changes    %8.1f onsets per hour, on average" % t["changes_per_hour"])
    print("  silence    %8.0f s in all, %.1f s per preset" % (t["silence_s"], t["silence_s"] / max(1, t["presets"])))
    print("  one_note   %6d conductors on a single pitch; %d presets without a note" % (t["one_note"], t["mute"]))
    print("  figures    %6d onsets: repeat in round %d (%.2f %%), same as previous %d (%.2f %%), rub %d (%.2f %%), "
          "shorter than twice their attack %d (%.2f %%)" % (
              t["figure_ons"], t["figure_repeat_in_round"], 100.0 * t["figure_repeat_in_round"] / fo,
              t["figure_same_as_previous"], 100.0 * t["figure_same_as_previous"] / fo, t["figure_rub"],
              100.0 * t["figure_rub"] / fo, t["figure_shorter_than_attack"],
              100.0 * t["figure_shorter_than_attack"] / fo))
    for key in ("mode", "scale"):
        print("by %s:" % key)
        for v in sorted({r[key] for r in rows}):
            print("  %-16s %s" % (v, line_of(totals([r for r in rows if r[key] == v]))))
    for key in FAULTS + ["one_note"]:
        w = sorted(rows, key=lambda r: -r[key])[:worst]
        w = [r for r in w if r[key] > 0]
        if not w:
            continue
        print("worst by %s:" % key)
        for r in w:
            v = ("%.1f %%" % (100 * r[key])) if key.endswith("rub") and not key.startswith("figure") else (
                "%g" % round(r[key], 1))
            print("  %-30s %-12s %-6s %-14s %10s   (onsets %d, figures %d, attack %g)" % (
                r["name"][:30], r["group"][:12], r["mode"], r["scale"][:14], v, r["ons"], r["figure_ons"],
                r["attack"]))
    return t


def bands_for(t):
    """The tolerance bands stored with the golden totals (see the golden file's header for the reasoning)."""
    def act(v):              # activity: two-sided, 15 %
        return [round(v * 0.85, 4), round(v * 1.15, 4)]

    def fault(v, add, mul):  # a fault: no lower bound, room above for harmless drift
        return [0, round(max(v * mul, v + add), 4)]
    return {"ons": act(t["ons"]), "notes": act(t["notes"]), "changes_per_hour": act(t["changes_per_hour"]),
            "figure_ons": act(t["figure_ons"]),
            "restrike": fault(t["restrike"], 15, 2.0), "doubling": fault(t["doubling"], 15, 2.0),
            "short": fault(t["short"], 40, 1.3), "self_unison": fault(t["self_unison"], 50, 1.3),
            "own_rub": fault(t["own_rub"], 0.003, 2.0), "cross_rub": fault(t["cross_rub"], 0.01, 1.5),
            "silence_s": fault(t["silence_s"], 1800, 1.5), "one_note": fault(t["one_note"], 2, 1.0),
            "mute": fault(t["mute"], 1, 1.0),
            "figure_repeat_in_round": fault(t["figure_repeat_in_round"], 15, 2.0),
            "figure_same_as_previous": fault(t["figure_same_as_previous"], 50, 1.3),
            "figure_rub": fault(t["figure_rub"], 30, 1.5),
            "figure_shorter_than_attack": fault(t["figure_shorter_than_attack"], 50, 1.3)}


def golden_sample(bank):
    """The fixed golden sample: GOLDEN_N names drawn with GOLDEN_SEED from the sorted bank."""
    return random.Random(GOLDEN_SEED).sample(sorted(bank), min(GOLDEN_N, len(bank)))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--sample", type=int, help="audit N presets drawn with --seed")
    mode.add_argument("--all", action="store_true", help="audit the whole bank")
    mode.add_argument("--check", metavar="GOLDEN", help="compare the golden sample with its bands")
    mode.add_argument("--update", action="store_true", help="measure the golden sample and write --golden")
    ap.add_argument("--seed", type=int, default=1911)
    ap.add_argument("--golden", default=GOLDEN)
    ap.add_argument("--hours", type=float, default=None, help="per preset (1 for the golden, 0.5 for --all)")
    ap.add_argument("--dt", type=float, default=8 * BLOCK, help="the step for --all and --sample, s")
    ap.add_argument("--param-every", type=int, default=4,
                    help="--all and --sample: read a modulated knob the notes hear every this many steps")
    ap.add_argument("--exe", default=None)
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--chunk", type=int, default=25, help="presets per audit process")
    ap.add_argument("--tsv", default=os.path.join(ROOT, "work", "music_audit", "music_audit.tsv"))
    ap.add_argument("--worst", type=int, default=10)
    ap.add_argument("--estimate", action="store_true", help="with --all: stop after the first 200 and the estimate")
    a = ap.parse_args()
    exe = a.exe or next((e for e in EXES if os.path.exists(e)), None)
    if not exe or not os.path.exists(exe):
        print("music_audit: no additiv_brain_audit (build it, or pass --exe)")
        return 2
    exe = os.path.abspath(exe)
    bank = load_bank()
    t0 = time.time()
    if a.check or a.update:
        path = a.check or a.golden
        if a.check:
            with open(path, encoding="utf-8") as fh:
                golden = json.load(fh)
            names = golden["presets"]
            hours = golden.get("hours", 1.0)
            dt, every = golden.get("dt", BLOCK), golden.get("param_every", GOLDEN_EVERY)
            missing = [n for n in names if n not in bank]
            if missing:
                print("music_audit: %d golden presets are no longer in the bank (first: %s); "
                      "run music_audit.py --update if that was meant" % (len(missing), missing[0]))
                return 1
        else:
            names, hours, dt, every = golden_sample(bank), a.hours or 1.0, BLOCK, GOLDEN_EVERY
        rows = audit(exe, names, bank, hours, dt, every, a.jobs, a.chunk)
        write_tsv(rows, a.tsv)
        t = summary(rows, a.worst)
        print("(%.0f s, TSV %s)" % (time.time() - t0, a.tsv))
        if a.update:
            golden = {
                "_about": ["Golden totals of Tools/presets/music_audit.py, checked by the ctest music_audit.",
                           "Written by: python Tools/presets/music_audit.py --update (%s). The sample is %d presets "
                           "with a conductor drawn with random.Random(%d) from the sorted bank; their names are "
                           "stored, so the check runs the same presets until this file is written again. Each is "
                           "audited for %g h, stepped as a render steps (one GPU block); a knob the matrix moves and "
                           "the notes hear is read every %d blocks." % (
                               time.strftime("%Y-%m-%d"), len(names), GOLDEN_SEED, hours, every),
                           "Bands: the conductors and the figures are deterministic per preset, so the totals come "
                           "back exactly on the same code. Activity (ons, notes, changes_per_hour, figure_ons) may "
                           "move 15 % either way: a change of pace is not a fault. Faults have no lower bound (fewer "
                           "is better: write the file again to tighten) and an upper one at twice the count or share "
                           "plus a small floor (restrike, doubling and figure_repeat_in_round +15, short x1.3 or "
                           "+40, self_unison, figure_same_as_previous and figure_shorter_than_attack x1.3 or +50, "
                           "figure_rub x1.5 or +30, own_rub +0.3 points, cross_rub x1.5 or +1 point, silence x1.5 "
                           "or +30 min, one_note +2, mute +1).",
                           "It bites: with ClusterBrain::againstOther switched off (20.11.2026) the check failed; "
                           "the numbers are in the commit that wrote this file."],
                "hours": hours, "dt": dt, "param_every": every, "seed": GOLDEN_SEED, "presets": names, "totals": t, "bands": bands_for(t)}
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
            with open(path, "w", encoding="utf-8", newline="\n") as fh:
                json.dump(golden, fh, indent=1)
                fh.write("\n")
            print("music_audit: wrote %s" % path)
            return 0
        bad = 0
        for key, (lo, hi) in golden["bands"].items():
            v = t[key]
            g = golden["totals"].get(key)
            if v < lo or v > hi:
                print("music_audit: FAIL %s = %s outside [%s, %s] (golden %s)" % (key, v, lo, hi, g))
                bad += 1
            elif g is not None and lo == 0 and v < g:
                print("music_audit: %s = %s, better than the golden %s (--update tightens the band)" % (key, v, g))
        if bad:
            print("music_audit: %d measure(s) outside their bands" % bad)
            return 1
        print("music_audit: all measures inside their bands")
        return 0
    names = sorted(bank)
    hours = a.hours or 0.5
    if a.sample:
        names = random.Random(a.seed).sample(names, min(a.sample, len(names)))
        rows = audit(exe, names, bank, hours, a.dt, a.param_every, a.jobs, a.chunk)
    else:
        first = names[:200]
        rows = audit(exe, first, bank, hours, a.dt, a.param_every, a.jobs,
                     max(1, min(a.chunk, 200 // max(1, a.jobs) + 1)))
        el = time.time() - t0
        print("first 200 presets in %.1f s: the whole bank (%d) should take about %.1f min" % (
            el, len(names), el / 200 * len(names) / 60), flush=True)
        if a.estimate:
            return 0
        rows += audit(exe, names[200:], bank, hours, a.dt, a.param_every, a.jobs, a.chunk, progress=True)
    write_tsv(rows, a.tsv)
    summary(rows, a.worst)
    print("(%.0f s, TSV %s)" % (time.time() - t0, a.tsv))
    return 0


if __name__ == "__main__":
    sys.exit(main())
