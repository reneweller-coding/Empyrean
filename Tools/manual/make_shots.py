"""The manual's screenshots: every tab as a whole page, from the standalone, muted.

    python Tools/manual/make_shots.py [path\\to\\Empyrean.exe]   -> docs/screenshots/page-N-<tab>.png, <picture>.png

One run of the standalone in the screenshot mode (Plugin/PluginEditor.cpp, shootStep): the preset PRESET loaded,
SETTLE seconds for the clouds to sound, then every tab on its own tick, the window grown until nothing of the page
scrolls (ADDITIV_SHOT_FULL). Muted (ADDITIV_MUTE): the engine runs and is seen, nothing is heard. Besides:
docs/screenshot.png, the first tab at the window's usual size, the picture on the project's front page. And the
pictures a chapter asks for with a line "[name]" in chapters.txt (EXTRAS): the command palette, the preset browser,
the modulation dock with a source dropped on a knob. A tab's name becomes a file name as slug() makes it. Last,
Empyrean Journeys' window (journeys-app.png), from the same build tree.
"""
import glob
import os
import re
import shutil
import subprocess
import sys

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
PRODUCT = "Empyrean"
PRESET = "Velvet Dreaming"   # Sleep Concert: the conductor, Breathe, the far reverb
SETTLE = 8                   # seconds: the preset fills its chord at speed (Engine::requestFill), the attack rises
# The chapters' own pictures: name -> (what the screenshot mode is asked, the tab photographed, by its name: a new tab
# moves the others).
EXTRAS = {
    "palette": ({"ADDITIV_SHOT_PALETTE": "dark calm wide"}, "Focus"),
    "browser": ({"ADDITIV_SHOT_BROWSER": "1"}, "Focus"),
    "dock": ({"ADDITIV_SHOT_DROP": "LFO 2>tilt", "ADDITIV_SHOT_SIZE": "1600x1000"}, "Cloud & Filter"),
    "journey": ({"ADDITIV_SHOT_JOURNEY": "Sleep Concert - Night"}, "Journey"),
}


def slug(tab):
    """A tab's name as a file name: "Cloud & Filter" -> "Cloud-Filter" (make_manual.py reads it so)."""
    return re.sub(r"[^A-Za-z0-9]+", "-", tab).strip("-")


def run(exe, prefix, full, extra=None):
    """One run of the standalone; the pictures land at <prefix>-<n>-<slug of the tab>.png."""
    env = dict(os.environ, ADDITIV_SHOT=prefix, ADDITIV_SHOT_AFTER=str(SETTLE), ADDITIV_PRESET=PRESET, ADDITIV_MUTE="1")
    env.update(extra or {})
    if full:
        env["ADDITIV_SHOT_FULL"] = "1"
    else:
        env.pop("ADDITIV_SHOT_FULL", None)
    for old in glob.glob(prefix + "-*.png"):
        os.remove(old)
    subprocess.run([exe], env=env, timeout=300, check=False)
    named = []
    for p in glob.glob(prefix + "-*.png"):
        n, tab = os.path.splitext(os.path.basename(p))[0][len(os.path.basename(prefix)) + 1:].split("-", 1)
        to = "%s-%s-%s.png" % (prefix, n, slug(tab))
        if to != p:
            os.replace(p, to)
        named.append(to)
    return sorted(named, key=lambda p: int(os.path.basename(p)[len(os.path.basename(prefix)) + 1:].split("-")[0]))


def main():
    exe = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        ROOT, "build", "msvc", "Plugin", PRODUCT + "_artefacts", "Release", "Standalone", PRODUCT + ".exe")
    if not os.path.exists(exe):
        sys.exit("no standalone at " + exe)
    out = os.path.join(ROOT, "docs", "screenshots")
    os.makedirs(out, exist_ok=True)
    pages = run(exe, os.path.join(out, "page"), True)
    if len(pages) < 7:
        sys.exit("the standalone took %d pictures, 7 expected" % len(pages))
    for p in pages:
        print(os.path.relpath(p, ROOT))
    small = run(exe, os.path.join(out, "front"), False)
    if not small:
        sys.exit("no picture for the front page")
    shutil.move(small[0], os.path.join(ROOT, "docs", "screenshot.png"))
    for p in small[1:]:
        os.remove(p)
    print("docs/screenshot.png")
    for name, (extra, tab) in EXTRAS.items():
        got = run(exe, os.path.join(out, "extra"), False, extra)
        shot = [p for p in got if p.endswith("-" + slug(tab) + ".png")]
        if not shot:
            sys.exit("no picture for " + name)
        shutil.move(shot[0], os.path.join(out, name + ".png"))
        for p in got:
            if os.path.exists(p):
                os.remove(p)
        print("docs/screenshots/%s.png" % name)
    # Empyrean Journeys, the program beside it: its window, a whole night chosen (EmpyreanJourneys --shot), from the
    # same build tree as the standalone.
    journeys = os.path.normpath(os.path.join(os.path.dirname(exe), "..", "..", "..", "..", "Tools", "renderapp",
                                             "EmpyreanJourneys_artefacts", "Release", "EmpyreanJourneys.exe"))
    if not os.path.exists(journeys):
        sys.exit("no Empyrean Journeys at " + journeys)
    subprocess.run([journeys, "--shot", os.path.join(out, "journeys-app.png"), "--choose", "Whole Night - Lullabies"],
                   timeout=120, check=True)
    print("docs/screenshots/journeys-app.png")


if __name__ == "__main__":
    main()
