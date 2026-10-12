"""The manual's screenshots: every tab as a whole page, from the standalone, muted.

    python Tools/manual/make_shots.py [path\\to\\Empyrean.exe]   -> docs/screenshots/page-N-<tab>.png

One run of the standalone in the screenshot mode (Plugin/PluginEditor.cpp, shootStep): the preset PRESET loaded,
SETTLE seconds for the clouds to sound, then every tab on its own tick, the window grown until nothing of the page
scrolls (ADDITIV_SHOT_FULL). Muted (ADDITIV_MUTE): the engine runs and is seen, nothing is heard. Besides:
docs/screenshot.png, the first tab at the window's usual size, the picture on the project's front page.
"""
import glob
import os
import shutil
import subprocess
import sys

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
PRODUCT = "Empyrean"
PRESET = "Velvet Dreaming"   # Sleep Concert: the conductor, Breathe, the far reverb
SETTLE = 8                   # seconds: the preset fills its chord at speed (Engine::requestFill), the attack rises


def run(exe, prefix, full):
    """One run of the standalone; the pictures land at <prefix>-<n>-<tab>.png."""
    env = dict(os.environ, ADDITIV_SHOT=prefix, ADDITIV_SHOT_AFTER=str(SETTLE), ADDITIV_PRESET=PRESET, ADDITIV_MUTE="1")
    if full:
        env["ADDITIV_SHOT_FULL"] = "1"
    else:
        env.pop("ADDITIV_SHOT_FULL", None)
    for old in glob.glob(prefix + "-*.png"):
        os.remove(old)
    subprocess.run([exe], env=env, timeout=300, check=False)
    return sorted(glob.glob(prefix + "-*.png"), key=lambda p: int(os.path.basename(p).split("-")[1]))


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


if __name__ == "__main__":
    main()
