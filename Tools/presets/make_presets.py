"""make_presets.py: the factory bank -- 16 groups of 64 presets -- into Core/src/Presets.inc.

Every group is a recipe: the ranges its sound lives in, drawn by a seeded die, so the bank is the
same on every run. A preset names only what it changes (Presets.h). The four macros mean the same
in every preset: 1 the motion, 2 the brightness (the cutoff where a filter is on), 3 the air, 4 the
space; the mod wheel opens the brightness too, pressure lifts the presence.

Levelling: Tools/presets (additiv_presets) renders every preset and writes work/presets/levels.json;
run this script again and every preset's master_gain brings it to the bank's common level.

    python Tools/presets/make_presets.py            # the bank, levelled if levels.json is there
"""
import json
import pathlib
import random

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = ROOT / "Core" / "src" / "Presets.inc"
LEVELS = ROOT / "work" / "presets" / "levels.json"
TARGET_DB = -20.0   # the bank's common level for the measuring chord (dBFS RMS)

# Choice indices, in the order of the tables in Core/src/Params.cpp.
SCALE = {"12-TET": 0, "JI Major": 1, "JI Minor": 2, "JI 7-limit": 3, "Pythagorean": 4, "JI Pentatonic": 5, "Harmonic": 6,
         "Subharmonic": 7, "Slendro": 8, "Bohlen-Pierce": 9, "Otonality": 10, "Timbre": 12}
MOTION = {"None": 0, "Random": 1, "Wave": 2, "Pendulum": 3, "Breathe": 4, "Glissando": 5, "Aurora": 6}
LAND = {"Aurora": 0, "Shepard": 1, "Tides": 2, "Rain": 3, "Choir": 4}
FILTER = {"Off": 0, "LP 6": 1, "LP 12": 2, "LP 24": 3, "HP 12": 4, "BP 12": 5, "Notch": 6, "Peak": 7, "Formant": 8}
LFO = {"Sine": 0, "Triangle": 1, "Ramp Up": 2, "Ramp Down": 3, "Square": 4, "Random": 5, "Steps": 6}
ENV = {"Swell": 0, "Pulse": 1, "Bloom": 2, "Flicker": 3, "Wobble": 4, "Rise In": 5, "Fall In": 6, "Saw Up": 7}

GROUPS = ["Sleep Concert", "Choirs", "Glass", "Bells", "Strings", "Organs", "Air & Noise", "Deep Space",
          "Glissandi", "Pendulums", "Waves", "Shepard & Tides", "Shimmer", "Dark Drones", "Microtonal", "Keys"]

NAMES = {
    "Sleep Concert": (["Somnolent", "Nocturnal", "Drowsy", "Hushed", "Lunar", "Slow", "Velvet", "Deep"],
                      ["Vigil", "Harbour", "Lantern", "Cradle", "Hours", "Tide", "Dreaming", "Hall"]),
    "Choirs": (["Distant", "Hollow", "Vowel", "Cathedral", "Breathing", "Pale", "Turning", "Silent"],
               ["Chorus", "Voices", "Hymn", "Choir", "Psalm", "Litany", "Chant", "Aria"]),
    "Glass": (["Frosted", "Clear", "Thin", "Cut", "Crystal", "Brittle", "Cold", "Lucid"],
              ["Glass", "Prism", "Facet", "Pane", "Harmonica", "Lens", "Vial", "Spire"]),
    "Bells": (["Bronze", "Temple", "Far", "Struck", "Sunken", "Iron", "Tibetan", "Tolling"],
              ["Bell", "Gong", "Bowl", "Carillon", "Chime", "Tower", "Peal", "Toll"]),
    "Strings": (["Bowed", "Silken", "Warm", "Sustained", "Muted", "Gliding", "Long", "Amber"],
                ["Strings", "Viols", "Cellos", "Ensemble", "Section", "Quartet", "Bows", "Fibres"]),
    "Organs": (["Stopped", "Reed", "Flue", "Open", "Chapel", "Ancient", "Pedal", "Mixture"],
               ["Organ", "Pipes", "Diapason", "Flute", "Bourdon", "Principal", "Celeste", "Nave"]),
    "Air & Noise": (["Wind", "Static", "Dusty", "Rustling", "Salt", "Grain", "Rain", "Ash"],
                    ["Field", "Breath", "Drift", "Veil", "Mist", "Spray", "Hiss", "Weather"]),
    "Deep Space": (["Orbital", "Dark", "Cosmic", "Silent", "Remote", "Event", "Stellar", "Void"],
                   ["Nebula", "Horizon", "Station", "Signal", "Expanse", "Corona", "Abyss", "Drift"]),
    "Glissandi": (["Falling", "Rising", "Gathering", "Converging", "Bending", "Sliding", "Arriving", "Swarming"],
                  ["Glissando", "Choir", "Swarm", "Chord", "Lights", "Voices", "Birds", "Cloud"]),
    "Pendulums": (["Swinging", "Slow", "Patient", "Pacing", "Turning", "Measured", "Clock", "Steady"],
                  ["Pendulum", "Clocks", "Weights", "Escapement", "Metronome", "Orbits", "Gears", "Cycles"]),
    "Waves": (["Rolling", "Climbing", "Long", "Swelling", "Broken", "Calm", "Rising", "Endless"],
              ["Waves", "Swell", "Surf", "Breakers", "Current", "Ocean", "Wake", "Shore"]),
    "Shepard & Tides": (["Endless", "Climbing", "Eternal", "Spiral", "Circling", "Tidal", "Ascending", "Infinite"],
                        ["Staircase", "Tide", "Spiral", "Ascent", "Ladder", "Moon", "Ebb", "Flood"]),
    "Shimmer": (["Glinting", "Bright", "Sparkling", "Silver", "Trembling", "Dappled", "Glowing", "Flickering"],
                ["Shimmer", "Light", "Haze", "Dust", "Halo", "Aurora", "Veil", "Sheen"]),
    "Dark Drones": (["Low", "Heavy", "Buried", "Sombre", "Black", "Subterranean", "Slow", "Brooding"],
                    ["Drone", "Engine", "Cavern", "Basalt", "Undertow", "Furnace", "Monolith", "Crypt"]),
    "Microtonal": (["Stretched", "Foreign", "Tilted", "Crooked", "Seventh", "Eleventh", "Tritave", "Bent"],
                   ["Scale", "Gamelan", "Tuning", "Lattice", "Comma", "Mode", "Series", "Ratio"]),
    "Keys": (["Soft", "Felt", "Played", "Close", "Gentle", "Touched", "Night", "Quiet"],
             ["Keys", "Pad", "Piano", "Hands", "Chords", "Notes", "Touch", "Fingers"]),
}


def f(x, digits=4):
    """A number as the table writes it: short, never in exponent form."""
    s = f"{x:.{digits}f}"
    if "." in s:   # only a fraction's zeros: 180 stays 180
        s = s.rstrip("0").rstrip(".")
    return s if s not in ("", "-0") else "0"


class Recipe:
    """One preset in the making: the parameters it changes and its matrix rows."""

    def __init__(self, rng):
        self.rng = rng
        self.p = {}
        self.rows = []

    def u(self, lo, hi):
        """A uniform draw."""
        return self.rng.uniform(lo, hi)

    def lg(self, lo, hi):
        """A draw even on a log scale (times, rates, frequencies)."""
        import math
        return math.exp(self.rng.uniform(math.log(lo), math.log(hi)))

    def pick(self, *items):
        """One of the items."""
        return self.rng.choice(items)

    def set(self, key, value):
        """A parameter."""
        self.p[key] = value

    def row(self, src, target, depth):
        """A matrix row."""
        self.rows.append(f"{src}>{target}:{f(depth, 3)}")


def common(r, filtered):
    """What every preset shares: the macros' meaning, the wheel, pressure, a slow life in the cloud."""
    r.row("macro1", "motion_depth", 0.45)
    r.row("macro1", "motion_rate", 0.2)
    r.row("macro2", "cutoff" if filtered else "brightness", 0.35)
    r.row("macro3", "air", 0.7)
    r.row("macro4", "depth", 0.35)
    r.row("macro4", "far_level", 0.25)
    r.row("wheel", "cutoff" if filtered else "brightness", 0.3)
    r.row("pressure", "presence", 0.5)
    # A slow LFO on the cloud's width, a random curve on its drift: every preset lives a little.
    r.set("lfo1_rate", f(r.lg(0.008, 0.06)))
    r.row("lfo1", "cloud_width", round(r.u(0.04, 0.12), 3))
    r.row("random", "drift", round(r.u(0.05, 0.15), 3))


def envelope(r, attack, decay, sustain, release):
    """The voice's envelope."""
    r.set("attack", f(attack, 3))
    r.set("decay", f(decay, 3))
    r.set("sustain", f(sustain, 3))
    r.set("release", f(release, 3))


def conductor(r, chords=True, every=(30, 120), density=(4, 6), low=(36, 43), high=(72, 84)):
    """The conductor's pace and register."""
    r.set("auto_mode", 1 if chords else 0)
    r.set("auto_rate", f(r.lg(*every), 1))
    r.set("brain_rate", f(r.lg(15, 60), 1))
    r.set("brain_density", r.rng.randint(*density))
    r.set("brain_low", r.rng.randint(*low))
    r.set("brain_high", r.rng.randint(*high))
    r.set("root", r.rng.randint(0, 11))


def g_sleep(r):
    r.set("scale", r.pick(SCALE["JI 7-limit"], SCALE["Otonality"], SCALE["JI Major"], SCALE["Harmonic"], SCALE["JI Minor"]))
    r.set("purity", 1)
    r.set("motion", r.pick(MOTION["Breathe"], MOTION["Breathe"], MOTION["None"], MOTION["Aurora"]))
    r.set("landscape", r.pick(LAND["Aurora"], LAND["Tides"]))
    r.set("cloud_width", f(r.u(12, 32), 1))
    r.set("tilt", f(r.u(1.2, 1.7), 2))
    r.set("brightness", f(r.u(0.45, 0.85), 2))
    envelope(r, r.lg(10, 30), r.lg(4, 12), r.u(0.8, 1), r.lg(20, 45))
    conductor(r, True, (60, 180), (4, 6))
    r.set("depth", f(r.u(0.6, 0.85), 2))
    r.set("far_decay", f(r.u(30, 60), 1))
    r.set("breath", f(r.u(0.2, 0.5), 2))
    r.set("tide", f(r.u(5, 15), 1))
    r.set("brain_timbre", f(r.u(0, 0.5), 2))
    common(r, False)


def g_choirs(r):
    r.set("filter_model", FILTER["Formant"])
    r.set("cutoff", f(r.lg(300, 4000), 0))
    r.set("resonance", f(r.u(0.4, 0.8), 2))
    r.set("keytrack", f(r.u(0, 0.3), 2))
    r.set("motion", r.pick(MOTION["Aurora"], MOTION["Breathe"]))
    r.set("landscape", LAND["Choir"])
    r.set("cloud_width", f(r.u(30, 60), 1))
    r.set("tilt", f(r.u(0.9, 1.3), 2))
    r.set("shimmer", f(r.u(0.4, 0.8), 2))
    envelope(r, r.lg(3, 12), r.lg(2, 8), r.u(0.8, 1), r.lg(8, 25))
    conductor(r, True, (30, 120), (4, 6), (45, 52), (74, 86))
    r.set("lfo2_shape", LFO["Sine"])
    r.set("lfo2_rate", f(r.lg(0.01, 0.08), 4))
    r.row("lfo2", "cutoff", round(r.u(0.1, 0.3), 3))   # the vowel turns
    common(r, True)


def g_glass(r):
    r.set("inharmonic", f(r.u(0.2, 0.6), 2))
    r.set("tilt", f(r.u(0.6, 1.0), 2))
    r.set("brightness", f(r.u(0.8, 1), 2))
    r.set("cloud_width", f(r.u(3, 10), 1))
    r.set("shimmer", f(r.u(0.1, 0.3), 2))
    r.set("presence", f(r.u(2, 5), 1))
    r.set("motion", r.pick(MOTION["Pendulum"], MOTION["Random"], MOTION["None"]))
    envelope(r, r.lg(0.05, 0.6), r.lg(3, 12), r.u(0.1, 0.4), r.lg(6, 20))
    conductor(r, r.rng.random() < 0.6, (20, 60), (3, 5), (52, 60), (80, 92))
    r.set("env1_shape", ENV["Pulse"])
    r.set("env1_rate", f(r.lg(0.1, 0.5), 3))
    r.row("env1", "brightness", -round(r.u(0.1, 0.25), 3))
    common(r, False)


def g_bells(r):
    r.set("inharmonic", f(r.u(0.5, 1.0), 2))
    r.set("odd_even", f(r.u(-0.6, 0.6), 2))
    r.set("tilt", f(r.u(0.7, 1.1), 2))
    r.set("cloud_width", f(r.u(2, 8), 1))
    r.set("scale", r.pick(SCALE["Timbre"], SCALE["JI Pentatonic"], SCALE["Slendro"], SCALE["JI 7-limit"]))
    r.set("motion", r.pick(MOTION["None"], MOTION["Wave"]))
    envelope(r, r.lg(0.01, 0.1), r.lg(6, 20), r.u(0, 0.2), r.lg(10, 30))
    conductor(r, False, (20, 60), (3, 5), (40, 52), (72, 88))
    r.set("brain_rate", f(r.lg(6, 20), 1))
    r.set("brain_hold_min", f(r.u(5, 10), 1))
    r.set("brain_hold_max", f(r.u(12, 30), 1))
    r.set("far_decay", f(r.u(25, 50), 1))
    common(r, False)


def g_strings(r):
    r.set("tilt", f(r.u(0.8, 1.1), 2))
    r.set("cloud_width", f(r.u(18, 40), 1))
    r.set("drift", f(r.u(5, 12), 1))
    r.set("shimmer", f(r.u(0.5, 0.8), 2))
    r.set("filter_model", FILTER["LP 12"])
    r.set("cutoff", f(r.lg(1200, 4000), 0))
    r.set("keytrack", f(r.u(0.5, 0.8), 2))
    r.set("filter_env", f(r.u(1, 2), 2))
    r.set("resonance", f(r.u(0.05, 0.3), 2))
    r.set("motion", r.pick(MOTION["Breathe"], MOTION["None"]))
    envelope(r, r.lg(1, 6), r.lg(2, 6), r.u(0.7, 0.95), r.lg(4, 12))
    conductor(r, True, (20, 90), (4, 6))
    common(r, True)


def g_organs(r):
    r.set("odd_even", f(r.pick(r.u(-0.9, -0.4), r.u(0.4, 0.9)), 2))
    r.set("cloud_width", f(r.u(2, 8), 1))
    r.set("tilt", f(r.u(0.9, 1.4), 2))
    r.set("shimmer", f(r.u(0.1, 0.3), 2))
    r.set("presence", f(r.u(0, 3), 1))
    r.set("motion", r.pick(MOTION["None"], MOTION["Breathe"]))
    r.set("purity", f(r.u(0.6, 1), 2))
    envelope(r, r.lg(0.3, 2), r.lg(1, 4), r.u(0.85, 1), r.lg(2, 8))
    conductor(r, True, (20, 80), (4, 6))
    r.set("far_size", f(r.u(1.8, 3), 2))
    r.set("far_decay", f(r.u(20, 45), 1))
    common(r, False)


def g_air(r):
    r.set("air", f(r.u(0.5, 1), 2))
    r.set("air_width", f(r.u(1000, 2400), 0))
    r.set("air_jitter", f(r.lg(4, 20), 2))
    r.set("air_center", f(r.u(1, 6), 2))
    r.set("brightness", f(r.u(0.3, 0.7), 2))
    r.set("motion", MOTION["Aurora"])
    r.set("landscape", r.pick(LAND["Rain"], LAND["Tides"], LAND["Aurora"]))
    r.set("cloud_width", f(r.u(20, 50), 1))
    envelope(r, r.lg(4, 16), r.lg(4, 10), r.u(0.8, 1), r.lg(10, 30))
    conductor(r, True, (40, 150), (3, 5))
    r.set("lfo3_shape", LFO["Random"])
    r.set("lfo3_rate", f(r.lg(0.05, 0.3), 3))
    r.row("lfo3", "air_center", round(r.u(0.1, 0.3), 3))
    common(r, False)


def g_space(r):
    r.set("tilt", f(r.u(1.6, 2.4), 2))
    r.set("depth", f(r.u(0.8, 1), 2))
    r.set("far_decay", f(r.u(45, 90), 1))
    r.set("far_level", f(r.u(0.8, 1), 2))
    r.set("doppler", f(r.u(0.5, 1), 2))
    r.set("breath", f(r.u(0.4, 0.9), 2))
    r.set("breath_rate", f(r.lg(0.01, 0.06), 3))
    r.set("air", f(r.u(0.2, 0.4), 2))
    r.set("air_center", f(r.u(1, 2), 2))
    r.set("motion", r.pick(MOTION["Breathe"], MOTION["Aurora"], MOTION["Wave"]))
    r.set("landscape", r.pick(LAND["Tides"], LAND["Shepard"]))
    envelope(r, r.lg(10, 30), r.lg(6, 15), r.u(0.8, 1), r.lg(25, 60))
    conductor(r, True, (90, 240), (3, 5), (24, 36), (55, 67))
    r.set("brain2_on", 1)
    r.set("brain2_interval", r.pick(-12, -24, 7))
    common(r, False)


def g_gliss(r):
    r.set("motion", MOTION["Glissando"])
    r.set("motion_rate", f(r.lg(0.3, 1.5), 3))
    r.set("cloud_width", f(r.u(10, 40), 1))
    r.set("tilt", f(r.u(1.0, 1.5), 2))
    envelope(r, r.lg(4, 12), r.lg(3, 8), r.u(0.8, 1), r.lg(10, 25))
    conductor(r, True, (25, 90), (4, 6))
    common(r, False)


def g_pendulum(r):
    r.set("motion", MOTION["Pendulum"])
    r.set("motion_rate", f(r.lg(0.3, 2), 3))
    r.set("motion_depth", f(r.u(0.6, 1), 2))
    r.set("tilt", f(r.u(0.9, 1.4), 2))
    r.set("cloud_width", f(r.u(5, 25), 1))
    envelope(r, r.lg(2, 8), r.lg(2, 6), r.u(0.8, 1), r.lg(8, 20))
    conductor(r, True, (30, 120), (4, 6))
    common(r, False)


def g_waves(r):
    r.set("motion", MOTION["Wave"])
    r.set("motion_rate", f(r.lg(0.3, 1.5), 3))
    r.set("motion_depth", f(r.u(0.5, 1), 2))
    r.set("tilt", f(r.u(0.9, 1.4), 2))
    r.set("cloud_width", f(r.u(10, 35), 1))
    envelope(r, r.lg(3, 10), r.lg(3, 8), r.u(0.8, 1), r.lg(8, 25))
    conductor(r, True, (30, 120), (4, 6))
    common(r, False)


def g_shepard(r):
    r.set("motion", MOTION["Aurora"])
    r.set("landscape", r.pick(LAND["Shepard"], LAND["Tides"]))
    r.set("motion_rate", f(r.lg(0.5, 2), 3))
    r.set("tilt", f(r.u(0.8, 1.2), 2))
    r.set("cloud_width", f(r.u(15, 40), 1))
    envelope(r, r.lg(4, 12), r.lg(3, 8), r.u(0.8, 1), r.lg(10, 25))
    conductor(r, True, (45, 150), (4, 6))
    common(r, False)


def g_shimmer(r):
    r.set("shimmer", f(r.u(0.7, 1), 2))
    r.set("shimmer_rate", f(r.lg(0.3, 1.5), 3))
    r.set("presence", f(r.u(3, 6), 1))
    r.set("brightness", f(r.u(0.7, 1), 2))
    r.set("cloud_width", f(r.u(20, 50), 1))
    r.set("tilt", f(r.u(0.8, 1.2), 2))
    r.set("motion", r.pick(MOTION["Random"], MOTION["Breathe"]))
    envelope(r, r.lg(3, 10), r.lg(3, 8), r.u(0.8, 1), r.lg(8, 25))
    conductor(r, True, (30, 120), (4, 6), (48, 55), (80, 92))
    common(r, False)


def g_dark(r):
    r.set("filter_model", FILTER["LP 24"])
    r.set("cutoff", f(r.lg(150, 700), 0))
    r.set("keytrack", f(r.u(0.2, 0.5), 2))
    r.set("resonance", f(r.u(0.1, 0.5), 2))
    r.set("tilt", f(r.u(1.8, 2.6), 2))
    r.set("brain2_on", 1)
    r.set("brain2_interval", r.pick(-12, -24, -7))
    r.set("motion", r.pick(MOTION["Breathe"], MOTION["None"], MOTION["Aurora"]))
    r.set("landscape", LAND["Tides"])
    envelope(r, r.lg(8, 25), r.lg(5, 12), r.u(0.8, 1), r.lg(15, 40))
    conductor(r, False, (40, 120), (3, 5), (28, 38), (60, 72))
    r.set("brain_rate", f(r.lg(20, 60), 1))
    common(r, True)


def g_micro(r):
    scale = r.pick("Bohlen-Pierce", "Slendro", "Harmonic", "Subharmonic", "Pythagorean", "Timbre")
    r.set("scale", SCALE[scale])
    if scale == "Bohlen-Pierce":
        r.set("keymap", 1)
    if scale == "Timbre":
        r.set("inharmonic", f(r.u(0.2, 0.6), 2))
        r.set("brain_timbre", f(r.u(0.5, 1), 2))
    r.set("purity", 1)
    r.set("cloud_width", f(r.u(4, 15), 1))
    r.set("tilt", f(r.u(1.0, 1.5), 2))
    r.set("motion", r.pick(MOTION["Breathe"], MOTION["None"], MOTION["Pendulum"]))
    envelope(r, r.lg(3, 12), r.lg(3, 8), r.u(0.8, 1), r.lg(8, 25))
    conductor(r, True, (30, 120), (4, 6))
    common(r, False)


def g_keys(r):
    r.set("brain_on", 0)
    r.set("keys_depth", f(r.u(0.1, 0.3), 2))
    r.set("motion", r.pick(MOTION["None"], MOTION["Breathe"], MOTION["Wave"]))
    r.set("cloud_width", f(r.u(8, 20), 1))
    r.set("filter_model", FILTER["LP 12"])
    r.set("cutoff", f(r.lg(800, 3000), 0))
    r.set("filter_env", f(r.u(1, 3), 2))
    r.set("keytrack", f(r.u(0.4, 0.8), 2))
    envelope(r, r.lg(0.02, 0.5), r.lg(1, 4), r.u(0.5, 0.9), r.lg(2, 8))
    r.set("root", r.rng.randint(0, 11))
    r.row("velocity", "brightness", round(r.u(0.15, 0.35), 3))
    common(r, True)


RECIPES = [g_sleep, g_choirs, g_glass, g_bells, g_strings, g_organs, g_air, g_space,
           g_gliss, g_pendulum, g_waves, g_shepard, g_shimmer, g_dark, g_micro, g_keys]


def build():
    """The bank: per group 64 presets, each a name, its parameters and its matrix."""
    levels = json.loads(LEVELS.read_text()) if LEVELS.exists() else {}
    bank = []
    for g, (group, recipe) in enumerate(zip(GROUPS, RECIPES)):
        adjectives, nouns = NAMES[group]
        names = [f"{a} {n}" for a in adjectives for n in nouns]
        random.Random(1000 + g).shuffle(names)
        for i in range(64):
            r = Recipe(random.Random(g * 1000 + i))
            recipe(r)
            name = names[i]
            if name in levels:   # the measured level, brought to the common one (at most 18 dB either way)
                r.set("master_gain", f(max(-18.0, min(12.0, TARGET_DB - levels[name])), 2))
            bank.append((name, g, " ".join(f"{k}={v}" for k, v in r.p.items()), ";".join(r.rows)))
    return bank


def main():
    bank = build()
    lines = ["// Generated by Tools/presets/make_presets.py -- do not edit; edit the script and run it again.",
             "/// The factory presets: 16 groups of 64 (Presets.h).",
             "const Preset kPresets[] = {"]
    for name, g, params, mod in bank:
        lines.append(f'    {{ "{name}", {g}, "{params}", "{mod}" }},')
    lines.append("};")
    OUT.write_bytes(("\n".join(lines) + "\n").encode("utf-8"))
    levelled = "levelled" if LEVELS.exists() else "not levelled yet (run additiv_presets)"
    print(f"{len(bank)} presets in {len(GROUPS)} groups -> {OUT} ({levelled})")


if __name__ == "__main__":
    main()
