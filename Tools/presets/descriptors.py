"""descriptors.py: what every preset measures, from wherever it is -- the measurement or its checked-in copy.

`additiv_presets --describe work/presets/descriptors.json` plays each preset a chord of fourteen seconds and describes
its last eight: per preset five numbers, the brightness (the spectral centroid in octaves above 40 Hz), the motion
(how much the spectrum changes over a quarter of a second, a second and three seconds, as a share of itself), the
width (1 - the correlation of left and right), the noisiness (the spectral flatness of the bands) and the level the
chord was measured at (LUFS). The measurement takes hours on the card and lands in work/, which git ignores.

Core/src/Find.inc, Journeys.inc, PresetMap.inc and Nudges.inc are made from it (make_find.py, make_journeys.py,
make_map.py, make_nudges.py), and a fresh clone has no work/. So a copy is checked in beside this script:
Tools/presets/descriptors.tsv, a preset a line in the measurement's order, the numbers as Python reads them (repr,
so they read back bit for bit). The generators call load(): the measurement when it is there, else the copy, and
they say which.

    python Tools/presets/descriptors.py --check     the copy against work/presets/descriptors.json (exit 1 when they
                                                    differ; passes, and says so, when only one of them is there)
    python Tools/presets/descriptors.py --refresh   the copy written anew from work/presets/descriptors.json

Refresh the copy (and commit it with the .inc files) whenever the bank is described again.
"""
import argparse
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
MEASURED = ROOT / "work" / "presets" / "descriptors.json"
COPY = ROOT / "Tools" / "presets" / "descriptors.tsv"
COLUMNS = ["brightness", "motion", "width", "noise", "level"]
HEADER = "# Tools/presets/descriptors.py --refresh from work/presets/descriptors.json (additiv_presets --describe)."


def read_measured():
    """name -> five numbers, from work/presets/descriptors.json; None when it is not there."""
    if not MEASURED.exists():
        return None
    return json.loads(MEASURED.read_text(encoding="utf-8"))


def read_copy():
    """name -> five numbers, from the checked-in copy; None when it is not there."""
    if not COPY.exists():
        return None
    out = {}
    for line in COPY.read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#") or line.startswith("name\t"):
            continue
        name, *values = line.split("\t")
        out[name] = [float(v) for v in values]
    return out


def load(quiet=False):
    """The descriptions: the measurement when it is there, else the checked-in copy; {} when neither is (and said)."""
    d = read_measured()
    where = "work/presets/descriptors.json"
    if d is None:
        d, where = read_copy(), "Tools/presets/descriptors.tsv (the checked-in copy)"
    if d is None:
        print("no descriptions: neither work/presets/descriptors.json nor Tools/presets/descriptors.tsv")
        return {}
    if not quiet:
        print(f"descriptions: {len(d)} presets from {where}")
    return d


def write_copy(d):
    lines = [HEADER, "name\t" + "\t".join(COLUMNS)]
    for name, values in d.items():
        if "\t" in name or len(values) != len(COLUMNS):
            raise SystemExit(f"cannot write {name!r}: a tab in the name or not {len(COLUMNS)} numbers")
        lines.append(name + "\t" + "\t".join(repr(float(v)) for v in values))
    COPY.write_bytes(("\n".join(lines) + "\n").encode("utf-8"))


def check():
    """0 when the copy and the measurement agree (or only one of them is there), 1 when they differ."""
    m, c = read_measured(), read_copy()
    if m is None or c is None:
        print(f"descriptors: only {'the copy' if m is None else 'the measurement'} is there -- nothing to compare")
        return 0 if (m or c) else 1
    only_m = [n for n in m if n not in c]
    only_c = [n for n in c if n not in m]
    differ = [n for n in m if n in c and [float(x) for x in m[n]] != c[n]]
    if not (only_m or only_c or differ):
        print(f"descriptors: the copy agrees with the measurement ({len(m)} presets)")
        return 0
    print(f"descriptors: the copy is stale -- {len(only_m)} presets only measured, {len(only_c)} only in the copy, "
          f"{len(differ)} differ (first: {(only_m + only_c + differ)[0]}); run descriptors.py --refresh")
    return 1


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true", help="the copy against the measurement")
    mode.add_argument("--refresh", action="store_true", help="the copy from the measurement")
    a = ap.parse_args()
    if a.refresh:
        m = read_measured()
        if m is None:
            raise SystemExit(f"{MEASURED} is not there")
        write_copy(m)
        print(f"{len(m)} presets -> {COPY}")
        return 0
    return check()


if __name__ == "__main__":
    sys.exit(main())
