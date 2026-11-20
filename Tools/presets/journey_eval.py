"""journey_eval.py: two journeys files judged by what the presets measure (20.11.2026; plan D1/D2).

For each file (Core/src/Journeys.inc or a candidate written by make_journeys.py --out): the mean, median and 90th
percentile step between neighbours (the four measures as z-scores over the bank: brightness, log motion, width,
noise), the spread of a group's Journey (the mean distance of its presets to their centre -- its different faces),
the nights' measured motion (z, lower is stiller), the median loudness jump between neighbours (LU, as measured)
and the mean stay. Copied from the session's scratchpad (plan_tools_2026-11-20) into the repository, reading the
measures as the generators do (descriptors.load: the measurement, else its checked-in copy).

    python Tools/presets/journey_eval.py BEFORE.inc AFTER.inc
"""
import math
import pathlib
import re
import statistics as st
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import descriptors  # noqa: E402


def measured(d):
    return [d[0], math.log(max(d[1], 0.0) + 0.01), d[2], d[3]]


def zscores(desc):
    names = list(desc)
    rows = [measured(desc[n]) for n in names]
    mean = [sum(r[i] for r in rows) / len(rows) for i in range(4)]
    sd = [math.sqrt(sum((r[i] - mean[i]) ** 2 for r in rows) / len(rows)) for i in range(4)]
    return {n: [(r[i] - mean[i]) / sd[i] for i in range(4)] for n, r in zip(names, rows)}


def dist(a, b):
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def parse(path):
    """title -> [[preset, dwell, glide]] from a Journeys.inc."""
    text = pathlib.Path(path).read_text(encoding="utf-8")
    out = {}
    for body in re.findall(r'^    "(.*)",$', text, re.M):
        body = body.replace("\\n", "\n").replace('\\"', '"')
        title = re.search(r"^journey (.*)$", body, re.M).group(1)
        out[title] = [l.split(" | ") for l in body.split("\n") if " | " in l and not l.startswith("#")]
    return out


def minutes(r):
    a, b = (r.split("-") + [r])[:2]
    f = lambda t: int(t.split(":")[0]) + int(t.split(":")[1]) / 60
    return (f(a) + f(b)) / 2


def evaluate(path, desc, z):
    """The figures of one file, as a dict."""
    js = parse(path)
    step, spread, nightmot, loud, dwell = [], [], [], [], []
    for t, steps in js.items():
        ns = [s[0] for s in steps if s[0] in z]
        cyc = ns + ns[:1]
        step += [dist(z[a], z[b]) for a, b in zip(cyc, cyc[1:])]
        loud += [abs(desc[a][4] - desc[b][4]) for a, b in zip(cyc, cyc[1:])]
        c = [sum(z[n][i] for n in ns) / len(ns) for i in range(4)]
        if t.endswith("- Journey"):
            spread.append(st.mean(dist(z[n], c) for n in ns))
        else:
            nightmot += [z[n][1] for n in ns]
        dwell += [minutes(s[1]) for s in steps]
    return {"journeys": len(js), "step_mean": st.mean(step), "step_median": st.median(step),
            "step_p90": sorted(step)[int(0.9 * len(step))], "spread": st.mean(spread), "night_motion": st.mean(nightmot),
            "loud_median": st.median(loud), "stay": st.mean(dwell)}


def main():
    desc = descriptors.load(quiet=True)
    z = zscores(desc)
    res = []
    for label, path in zip(("before", "after"), sys.argv[1:3]):
        r = evaluate(path, desc, z)
        res.append(r)
        print("%-6s journeys %d | step between neighbours: mean %.3f median %.3f p90 %.3f | spread of a journey %.3f | "
              "nights' motion z %.3f | loudness jump median %.2f | stay %.1f min" % (
                  label, r["journeys"], r["step_mean"], r["step_median"], r["step_p90"], r["spread"],
                  r["night_motion"], r["loud_median"], r["stay"]))
    if len(res) == 2:
        a, b = res
        print("change | step mean %+.1f %% median %+.1f %% p90 %+.1f %% | spread %+.1f %%" % (
            100 * (b["step_mean"] / a["step_mean"] - 1), 100 * (b["step_median"] / a["step_median"] - 1),
            100 * (b["step_p90"] / a["step_p90"] - 1), 100 * (b["spread"] / a["spread"] - 1)))


if __name__ == "__main__":
    main()
