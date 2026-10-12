"""The user manual, built out of the program itself (after Totality's Tools/manual).

    python Tools/manual/make_shots.py                      -> docs/screenshots/page-N-<tab>.png
    python Tools/manual/make_manual.py [--render EXE]      -> docs/manual/<product>-Manual.html (and .pdf)

The prose is the help's: Plugin/Resources/chapters.txt, the text the program shows under the question mark, so the
two never disagree. The parameter tables are not written down here: `empy_render --dump-params` writes every row
of the engine's table (Core/src/Params.cpp), and each lands under the chapter of the tab that shows it. A section of
the table that no chapter takes is a hole, and the manual is not printed until it is placed (--allow-holes prints
anyway). The appendix -- MIDI, the presets, what the instrument needs -- is generated too.

The PDF is printed by Edge or Chrome in headless mode; without a browser the HTML is still written and prints by hand.
"""
import argparse
import glob
import html
import json
import os
import re
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, "..", ".."))
PRODUCT = "Empyrean"
TITLE = "Empyrean"   # the cover's

# The sections of the parameter table on each tab (Plugin/PluginEditor.cpp builds the tabs).
TABS = {
    "Cloud": ["Cloud", "Air"],
    "Motion": ["Motion", "Field"],
    "Filter": ["Filter", "Envelope"],
    "Tuning": ["Tuning"],
    "Conductor": ["Conductor"],
    "Space": ["Space", "Master"],
    "Effects": ["Echo", "Room", "Modal"],
    "Modulation": ["Modulation"],
}

CSS = """
@page { size: A4; margin: 17mm 15mm 15mm 15mm; }
body { font: 10.5pt/1.55 "Segoe UI", "Helvetica Neue", Arial, sans-serif; color: #15171c; background: #fff; margin: 0; }
h1 { font-size: 26pt; margin: 0 0 2mm 0; letter-spacing: 1.5pt; color: #2d6f62; }
h2 { font-size: 16pt; margin: 0 0 3mm 0; padding-bottom: 2mm; border-bottom: 1px solid #cfdad6; color: #2d6f62; }
h3 { font-size: 10.5pt; margin: 5mm 0 1mm 0; color: #5d4f8a; letter-spacing: 0.6pt; text-transform: uppercase; }
p { margin: 0 0 3.2mm 0; text-align: justify; hyphens: auto; }
.cover { page-break-after: always; text-align: center; padding-top: 22mm; }
.cover img.logo { width: 34mm; border: none; margin: 0 auto 6mm auto; display: block; }
.cover p { text-align: center; }
.cover img.front { width: 100%; margin-top: 10mm; border: 1px solid #cfdad6; border-radius: 3px; }
.sub { color: #555b66; font-size: 12pt; margin: 0; }
.facts { margin-top: 7mm; color: #555b66; font-size: 9.5pt; }
.toc { page-break-after: always; }
.toc li { margin: 1.2mm 0; }
.chapter { page-break-before: always; }
.shot { width: auto; max-width: 100%; max-height: 240mm; display: block; margin: 2mm auto 4mm auto; border: 1px solid #cfdad6; border-radius: 3px; }
table { border-collapse: collapse; width: 100%; font-size: 8.8pt; margin: 2mm 0 5mm 0; }
th { text-align: left; color: #555b66; font-weight: 600; border-bottom: 1px solid #a9b8b3; padding: 1mm 1.5mm; }
td { border-bottom: 1px solid #e3eae7; padding: 0.8mm 1.5mm; vertical-align: top; }
td.key { font-family: Consolas, monospace; font-size: 8.3pt; color: #555b66; }
.names { columns: 3; column-gap: 6mm; font-size: 8.6pt; margin: 0 0 4mm 0; }
.names div { break-inside: avoid; }
"""


def version():
    """The version from the project() line of CMakeLists.txt, the one source of truth."""
    text = open(os.path.join(ROOT, "CMakeLists.txt"), encoding="utf-8").read()
    m = re.search(r"project\(\s*\w+\s+VERSION\s+([0-9.]+)", text)
    return m.group(1) if m else "?"


def chapters():
    """The help's chapters: [{title, modules: [tab or 'all'], paragraphs: [text]}]."""
    out = []
    for line in open(os.path.join(ROOT, "Plugin", "Resources", "chapters.txt"), encoding="utf-8").read().splitlines():
        m = re.match(r"^==\s*(.*?)\s*\|\s*(.*?)\s*\|\s*(.*?)\s*==\s*$", line)
        if m:
            out.append({"title": m.group(1), "modules": [x.strip() for x in m.group(2).split(",") if x.strip()], "paragraphs": [""]})
        elif out and not line.startswith("#"):
            if line.strip():
                out[-1]["paragraphs"][-1] += (" " if out[-1]["paragraphs"][-1] else "") + line.strip()
            elif out[-1]["paragraphs"][-1]:
                out[-1]["paragraphs"].append("")
    for c in out:
        c["paragraphs"] = [p for p in c["paragraphs"] if p]
    return out


def number(x):
    """A number as a table shows it: short."""
    return ("%.4g" % x) if isinstance(x, float) else str(x)


def range_text(p):
    """A parameter's range: its choices, on / off, or min .. max with the unit."""
    if p["kind"] == "choice":
        return ", ".join(html.escape(c) for c in p["choices"])
    if p["kind"] == "bool":
        return "off / on"
    unit = (" " + p["unit"]) if p["unit"] else ""
    return "%s .. %s%s" % (number(p["min"]), number(p["max"]), html.escape(unit))


def default_text(p):
    """A parameter's default, as its range reads."""
    if p["kind"] == "choice":
        i = int(round(p["default"]))
        return html.escape(p["choices"][i]) if 0 <= i < len(p["choices"]) else str(i)
    if p["kind"] == "bool":
        return "on" if p["default"] >= 0.5 else "off"
    return number(p["default"]) + ((" " + html.escape(p["unit"])) if p["unit"] else "")


def table(section, params):
    """One section's parameters as a table."""
    rows = "".join("<tr><td>%s</td><td class='key'>%s</td><td>%s</td><td>%s</td></tr>"
                   % (html.escape(p["name"]), html.escape(p["key"]), range_text(p), default_text(p)) for p in params)
    return "<h3>%s</h3><table><tr><th>Control</th><th>Key</th><th>Range</th><th>Default</th></tr>%s</table>" % (html.escape(section), rows)


def presets():
    """The factory presets by group, read from the generated bank (Core/src/Presets.inc)."""
    groups = {}
    text = open(os.path.join(ROOT, "Core", "src", "Presets.inc"), encoding="utf-8").read()
    for name, g in re.findall(r'^\s*\{ "([^"]+)", (\d+),', text, re.M):
        groups.setdefault(int(g), []).append(name)
    names = re.search(r"kPresetGroupNames\[kNumPresetGroups\]\s*=\s*\{(.*?)\};",
                      open(os.path.join(ROOT, "Core", "src", "Presets.cpp"), encoding="utf-8").read(), re.S)
    titles = re.findall(r'"([^"]+)"', names.group(1)) if names else []
    return [(titles[g] if g < len(titles) else "Group %d" % g, groups[g]) for g in sorted(groups)]


APPENDIX_MIDI = [
    ("Notes", "play over the conductor (velocity: the note's level, and a source of the matrix)"),
    ("Controller 0, then program change", "a preset: the bank names the group (0 .. 15), the program the preset in it (0 .. 63)"),
    ("Controller 1 (mod wheel)", "a source of the matrix; the factory presets open the filter or the brightness with it"),
    ("Controller 64 (hold pedal)", "keeps notes sounding until it comes up"),
    ("Controller 74", "the filter's cutoff"),
    ("Channel or key pressure", "a source of the matrix; the factory presets lift the presence with it"),
    ("All notes off, all sound off", "every voice is released"),
    ("Any other controller", "MIDI learn: right-click a control, move the controller"),
]

APPENDIX_NEEDS = [
    ("Graphics card", "NVIDIA GeForce RTX 20 or newer (Turing and later) with a current driver; the instrument computes every "
                      "partial there and is silent without one, saying so in the status line"),
    ("System", "Windows 10 or 11, 64-bit, a processor with AVX2"),
    ("Formats", "standalone application and VST3 plug-in; the offline renderer empy_render for the command line"),
    ("Latency", "two audio blocks and ten milliseconds, reported to the host, which compensates for it"),
]


def find_browser():
    """Edge or Chrome, for the PDF."""
    for p in [r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe", r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
              r"C:\Program Files\Google\Chrome\Application\chrome.exe", r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe"]:
        if os.path.exists(p):
            return p
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--render", default=os.path.join(ROOT, "build", "msvc", "Release", "empy_render.exe"))
    ap.add_argument("--out", default=os.path.join(ROOT, "docs", "manual", PRODUCT + "-Manual.html"))
    ap.add_argument("--allow-holes", action="store_true")
    ap.add_argument("--no-pdf", action="store_true")
    a = ap.parse_args()

    params = json.loads(subprocess.run([a.render, "--dump-params"], capture_output=True, text=True, check=True).stdout)
    sections = {}
    for p in params:
        sections.setdefault(p["section"], []).append(p)
    chs = chapters()
    tabs_named = {m for c in chs for m in c["modules"] if m != "all"}
    unknown = sorted(tabs_named - set(TABS))
    if unknown:
        print("chapters name tabs that do not exist: " + ", ".join(unknown))
        return 1
    placed = {s for t in tabs_named for s in TABS[t]}
    holes = [s for s in sections if s not in placed]
    if holes:
        print("sections on no page of the manual: " + ", ".join(holes))
        if not a.allow_holes:
            return 1

    out_dir = os.path.dirname(a.out)
    os.makedirs(out_dir, exist_ok=True)
    rel = lambda path: os.path.relpath(path, out_dir).replace("\\", "/")
    shots = {}
    for path in glob.glob(os.path.join(ROOT, "docs", "screenshots", "page-*-*.png")):
        shots[os.path.splitext(os.path.basename(path))[0].split("-", 2)[2]] = path
    front = os.path.join(ROOT, "docs", "screenshot.png")

    parts = ["<!DOCTYPE html><html><head><meta charset='utf-8'><title>%s Manual</title><style>%s</style></head><body>" % (PRODUCT, CSS)]
    parts.append("<div class='cover'><img class='logo' src='%s'><h1>%s</h1>"
                 "<p class='sub'>An additive ambient synthesizer: twenty thousand partials per voice, on the graphics card</p>"
                 "<p class='facts'>Version %s &middot; %d parameters &middot; %d presets &middot; manual built %s</p>%s</div>"
                 % (rel(os.path.join(ROOT, "Deploy", "empyrean_512.png")), html.escape(TITLE.upper()), version(), len(params),
                    sum(len(n) for _, n in presets()), time.strftime("%d.%m.%Y"),
                    ("<img class='front' src='%s'>" % rel(front)) if os.path.exists(front) else ""))
    appendix = ["MIDI", "The presets", "What it needs", "License"]
    parts.append("<div class='toc'><h2>Contents</h2><ol>%s</ol></div>"
                 % "".join("<li><a href='#c%d'>%s</a></li>" % (i, html.escape(t)) for i, t in enumerate([c["title"] for c in chs] + appendix)))
    for i, c in enumerate(chs):
        parts.append("<div class='chapter' id='c%d'><h2>%s</h2>" % (i, html.escape(c["title"])))
        for tab in c["modules"]:
            if tab in shots:
                parts.append("<img class='shot' src='%s'>" % rel(shots[tab]))
        for text in c["paragraphs"]:
            parts.append("<p>%s</p>" % html.escape(text))
        for tab in c["modules"]:
            for s in TABS.get(tab, []):
                if s in sections:
                    parts.append(table(s, sections[s]))
        parts.append("</div>")
    n = len(chs)
    parts.append("<div class='chapter' id='c%d'><h2>MIDI</h2><table><tr><th>Message</th><th>What it does</th></tr>%s</table></div>"
                 % (n, "".join("<tr><td>%s</td><td>%s</td></tr>" % (html.escape(k), html.escape(v)) for k, v in APPENDIX_MIDI)))
    parts.append("<div class='chapter' id='c%d'><h2>The presets</h2>" % (n + 1))
    for g, (title, names) in enumerate(presets()):
        parts.append("<h3>%d &middot; %s</h3><div class='names'>%s</div>"
                     % (g, html.escape(title), "".join("<div>%d %s</div>" % (i, html.escape(x)) for i, x in enumerate(names))))
    parts.append("</div>")
    parts.append("<div class='chapter' id='c%d'><h2>What it needs</h2><table>%s</table></div>"
                 % (n + 2, "".join("<tr><td>%s</td><td>%s</td></tr>" % (html.escape(k), html.escape(v)) for k, v in APPENDIX_NEEDS)))
    parts.append("<div class='chapter' id='c%d'><h2>License</h2><p>%s is free software under the GNU Affero General Public "
                 "License, version 3 (LICENSE.txt beside the program). It is built with JUCE and NVIDIA's CUDA; the CUDA runtime "
                 "is linked into the program under NVIDIA's license for it.</p></div>" % (n + 3, PRODUCT))
    parts.append("</body></html>")
    with open(a.out, "w", encoding="utf-8") as f:
        f.write("\n".join(parts))
    print("wrote " + a.out)

    if not a.no_pdf:
        browser = find_browser()
        if browser is None:
            print("no Edge or Chrome: the HTML prints by hand")
            return 0
        pdf = os.path.splitext(a.out)[0] + ".pdf"
        url = "file:///" + os.path.abspath(a.out).replace("\\", "/")
        # The old one goes first, so that a PDF there is this one. The browser returns before its child process has
        # written the file: it is waited for until its size holds. A profile of its own, or the call hands the job to
        # a running browser and returns without a file (Totality's lesson).
        if os.path.exists(pdf):
            os.remove(pdf)
        profile = tempfile.mkdtemp(prefix="additiv_manual_")
        subprocess.run([browser, "--headless=new", "--disable-gpu", "--no-pdf-header-footer", "--user-data-dir=" + profile,
                        "--print-to-pdf=" + pdf, url], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=180)
        size = -1
        for _ in range(120):
            now = os.path.getsize(pdf) if os.path.exists(pdf) else -1
            if now > 0 and now == size:
                break
            size = now
            time.sleep(0.5)
        print(("wrote " + pdf) if os.path.exists(pdf) else "the browser wrote no PDF")
    return 0


if __name__ == "__main__":
    sys.exit(main())
