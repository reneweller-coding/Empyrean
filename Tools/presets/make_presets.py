"""make_presets.py: the factory bank -- 32 groups of 64 presets -- into Core/src/Presets.inc.

Every group is a recipe: the ranges its sound lives in, drawn by a seeded die, so the bank is the
same on every run. A preset names only what it changes (Presets.h). The four macros mean the same
in every preset: 1 the motion, 2 the brightness (the cutoff where a filter is on), 3 the air, 4 the
space; the mod wheel opens the brightness too, pressure lifts the presence.

The first sixteen groups are the sounds; the second sixteen (the living systems, 12.10.2026) each
start from one of them and put one of the systems in front: the living landscape, the sync, the
flock, the ripple, the melt, the spectral echo, the modal hall, the memory, the attractors, the
reservoir, what the instrument hears, the field, the dispersion, beating lattices. Phase 10 was
added to the first sixteen too, drawn after everything else, so what they had stays as it was.

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
          "Glissandi", "Pendulums", "Waves", "Shepard & Tides", "Shimmer", "Dark Drones", "Microtonal", "Keys",
          "Living Fields", "Turing Gardens", "Synchrony", "Flocks", "Ripples", "Melting Spectra", "Spectral Cascades",
          "Resonant Halls", "Ghosts & Memory", "Strange Attractors", "Reservoir Minds", "Self-Listening", "Gravity Wells",
          "Smear & Dispersion", "Lattices & Beats", "Night Gardens"]

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
    "Living Fields": (["Growing", "Teeming", "Mossy", "Budding", "Creeping", "Verdant", "Spreading", "Seeded"],
                      ["Meadow", "Colony", "Lichen", "Bloom", "Culture", "Garden", "Reef", "Fern"]),
    "Turing Gardens": (["Spotted", "Striped", "Dappling", "Patterned", "Mottled", "Branching", "Labyrinth", "Coral"],
                       ["Morphogen", "Pattern", "Leopard", "Zebra", "Maze", "Mitosis", "Shell", "Pigment"]),
    "Synchrony": (["Entrained", "Locked", "Pulsing", "Coupled", "Aligned", "Gathered", "Kindred", "Phased"],
                  ["Fireflies", "Crickets", "Hearts", "Lanterns", "Rowers", "Metronomes", "Pulses", "Tremolo"]),
    "Flocks": (["Murmuring", "Wheeling", "Banking", "Scattering", "Soaring", "Flocking", "Darting", "Homing"],
               ["Starlings", "Swifts", "Shoal", "Murmuration", "Geese", "Swallows", "Sparrows", "Herons"]),
    "Ripples": (["Rippling", "Pebbled", "Concentric", "Plucked", "Skimmed", "Dropped", "Quivering", "Lapping"],
                ["Pond", "Rings", "Stones", "Puddle", "Basin", "Lake", "Fountain", "Water"]),
    "Melting Spectra": (["Melting", "Thawing", "Molten", "Dissolving", "Softened", "Liquid", "Fading", "Weeping"],
                        ["Icefield", "Wax", "Snow", "Glacier", "Candle", "Frost", "Sugar", "Icicles"]),
    "Spectral Cascades": (["Cascading", "Splintered", "Echoing", "Refracted", "Tumbling", "Mirrored", "Fractured", "Repeating"],
                          ["Prisms", "Stairs", "Falls", "Fountains", "Spectra", "Rainbows", "Steps", "Mirrors"]),
    "Resonant Halls": (["Resonant", "Vaulted", "Ringing", "Painted", "Columned", "Marble", "Hollowed", "Echoed"],
                       ["Rotunda", "Cistern", "Basilica", "Atrium", "Cloister", "Vault", "Chamber", "Apse"]),
    "Ghosts & Memory": (["Remembered", "Haunted", "Faded", "Lingering", "Spectral", "Recalled", "Former", "Vanished"],
                        ["Rooms", "Letters", "Portraits", "Summers", "Gardens", "Waltz", "Footsteps", "Afternoons"]),
    "Strange Attractors": (["Strange", "Chaotic", "Folded", "Butterfly", "Wandering", "Unstable", "Turbulent", "Looping"],
                           ["Attractor", "Orbit", "Lorenz", "Roessler", "Saddle", "Torus", "Manifold", "Trajectory"]),
    "Reservoir Minds": (["Thinking", "Dreaming", "Listening", "Echo-State", "Neural", "Recurrent", "Pondering", "Musing"],
                        ["Network", "Mind", "Synapse", "Neurons", "Memory", "Thought", "Reverie", "Cortex"]),
    "Self-Listening": (["Attentive", "Responsive", "Self-Aware", "Heeding", "Answering", "Hearing", "Reacting", "Alert"],
                       ["Ear", "Mirror", "Feedback", "Reply", "Echo", "Response", "Witness", "Listener"]),
    "Gravity Wells": (["Gravitating", "Pulled", "Drawn", "Orbiting", "Falling", "Magnetic", "Tethered", "Sinking"],
                      ["Well", "Mass", "Lodestone", "Anchor", "Centre", "Pole", "Core", "Magnet"]),
    "Smear & Dispersion": (["Smeared", "Dispersed", "Blurred", "Smudged", "Streaked", "Diffuse", "Bleeding", "Washed"],
                           ["Ink", "Watercolour", "Chalk", "Pastel", "Charcoal", "Graphite", "Gouache", "Sepia"]),
    "Lattices & Beats": (["Interlocking", "Beating", "Woven", "Latticed", "Gridded", "Knotted", "Braided", "Meshed"],
                         ["Lattice", "Weave", "Tapestry", "Trellis", "Loom", "Mesh", "Grid", "Fabric"]),
    "Night Gardens": (["Moonlit", "Nightblooming", "Starlit", "Dewy", "Midnight", "Dusky", "Owl", "Firefly"],
                      ["Orchard", "Grove", "Arbour", "Glade", "Pergola", "Lawn", "Hedge", "Bower"]),
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

    def row(self, src, target, depth, via=None):
        """A matrix row, its depth scaled by a via source's amount when one is given."""
        self.rows.append(f"{src}>{target}:{f(depth, 3)}" + (f":{via}" if via else ""))


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


# The effects per group (12.10.2026): each entry a probability and ranges. echo: Noctuary's first
# delay (level, left time in s, feedback; the right time 1.2 .. 1.6 times the left); echo2: the
# second, in series; room: early reflections, room size, near reverb, its decay; modal: the modal
# hall on the GPU (level, decay, damping, tuned share, spread in cents); far: diffusion and the
# network (2 Colourless, 3 Rotating). What a group leaves out keeps the table's default.
FX = {
    "Sleep Concert":   dict(echo=(0.6, (0.10, 0.25), (1.2, 3.0), (0.35, 0.60)), echo2=(0.3, (0.08, 0.18)),
                            room=((0.10, 0.25), (10, 25), (0.05, 0.15), (1.5, 3.0)),
                            modal=(0.8, (0.15, 0.35), (10, 25), (0.3, 0.5), (0.3, 0.8), (3, 10)), far=((0.4, 0.8), (2, 3))),
    "Choirs":          dict(echo=(0.3, (0.08, 0.15), (0.8, 2.0), (0.30, 0.50)),
                            room=((0.20, 0.40), (18, 35), (0.15, 0.30), (1.5, 3.5)),
                            modal=(0.6, (0.10, 0.25), (6, 14), (0.3, 0.5), (0.2, 0.6), (4, 12))),
    "Glass":           dict(echo=(0.7, (0.15, 0.30), (0.25, 0.9), (0.40, 0.70)),
                            modal=(0.9, (0.25, 0.45), (8, 20), (0.1, 0.3), (0.6, 1.0), (0, 4))),
    "Bells":           dict(echo=(0.5, (0.10, 0.25), (0.6, 1.8), (0.30, 0.55)),
                            modal=(1.0, (0.25, 0.45), (15, 40), (0.2, 0.4), (0.7, 1.0), (2, 8))),
    "Strings":         dict(echo=(0.3, (0.08, 0.15), (0.6, 1.5), (0.30, 0.50)),
                            room=((0.25, 0.45), (15, 30), (0.15, 0.30), (1.8, 3.5)),
                            modal=(0.5, (0.10, 0.20), (5, 12), (0.3, 0.5), (0.2, 0.5), (4, 12))),
    "Organs":          dict(room=((0.30, 0.50), (25, 40), (0.20, 0.35), (2.5, 5.0)),
                            modal=(0.6, (0.15, 0.30), (8, 18), (0.3, 0.5), (0.0, 0.3), (4, 12))),
    "Air & Noise":     dict(echo=(0.5, (0.10, 0.25), (0.4, 1.6), (0.35, 0.60)),
                            modal=(0.8, (0.20, 0.40), (6, 20), (0.4, 0.7), (0.0, 0.2), (6, 15))),
    "Deep Space":      dict(echo=(0.8, (0.15, 0.30), (1.5, 4.0), (0.50, 0.75)), echo2=(0.5, (0.10, 0.20)),
                            modal=(0.9, (0.25, 0.45), (25, 60), (0.4, 0.7), (0.3, 0.7), (4, 12)), far=((0.6, 0.9), (3, 3))),
    "Glissandi":       dict(echo=(0.5, (0.10, 0.20), (0.5, 1.5), (0.35, 0.55)),
                            modal=(0.7, (0.15, 0.30), (8, 20), (0.3, 0.5), (0.0, 0.4), (4, 12))),
    "Pendulums":       dict(echo=(0.8, (0.15, 0.30), (0.3, 1.2), (0.30, 0.60)),
                            modal=(0.5, (0.10, 0.25), (6, 15), (0.3, 0.5), (0.3, 0.7), (3, 10))),
    "Waves":           dict(echo=(0.5, (0.10, 0.20), (0.8, 2.5), (0.35, 0.60)),
                            modal=(0.8, (0.20, 0.35), (10, 25), (0.3, 0.5), (0.3, 0.7), (3, 10))),
    "Shepard & Tides": dict(echo=(0.4, (0.10, 0.20), (1.0, 2.5), (0.35, 0.60)),
                            modal=(0.9, (0.25, 0.40), (12, 30), (0.3, 0.5), (0.7, 1.0), (2, 8))),
    "Shimmer":         dict(echo=(0.8, (0.15, 0.30), (0.2, 0.8), (0.40, 0.70)),
                            modal=(0.8, (0.20, 0.40), (10, 25), (0.1, 0.3), (0.5, 0.9), (2, 8))),
    "Dark Drones":     dict(echo=(0.4, (0.10, 0.20), (2.0, 4.0), (0.40, 0.65)),
                            modal=(1.0, (0.25, 0.45), (30, 60), (0.5, 0.8), (0.2, 0.6), (4, 12)), far=((0.7, 0.9), (3, 3))),
    "Microtonal":      dict(echo=(0.4, (0.10, 0.20), (0.6, 2.0), (0.30, 0.55)),
                            modal=(1.0, (0.25, 0.45), (10, 30), (0.2, 0.5), (0.8, 1.0), (0, 5))),
    "Keys":            dict(echo=(0.4, (0.10, 0.20), (0.3, 0.9), (0.30, 0.50)),
                            room=((0.20, 0.35), (8, 18), (0.15, 0.25), (1.2, 2.5)),
                            modal=(0.5, (0.10, 0.25), (5, 12), (0.3, 0.5), (0.4, 0.8), (3, 10))),
}


# The field, the dispersion and the hall's breath per group (12.10.2026), drawn after the effects:
# field = (probability, modes, pull, radius in cents, spring, wander); disp = (probability, seconds);
# breath = (probability, amount). Modes: 1 Scale, 2 Swarm, 3 Scatter.
MOTION_FX = {
    "Sleep Concert":   dict(field=(0.2, (2,), (0.2, 0.4), (80, 300), (0.3, 0.6), (0.02, 0.08)), breath=(0.4, (0.2, 0.6))),
    "Glass":           dict(disp=(0.6, (0.1, 0.8))),
    "Bells":           dict(disp=(0.5, (0.1, 0.6))),
    "Air & Noise":     dict(field=(0.3, (3,), (0.4, 0.8), (60, 250), (0.2, 0.5), (0.05, 0.3)), disp=(0.3, (0.2, 1.0))),
    "Deep Space":      dict(field=(0.4, (2, 3), (0.3, 0.7), (100, 400), (0.1, 0.4), (0.02, 0.1)), breath=(0.6, (0.3, 0.8))),
    "Glissandi":       dict(field=(0.4, (2,), (0.4, 0.8), (100, 400), (0.1, 0.4), (0.05, 0.2))),
    "Waves":           dict(field=(0.3, (2,), (0.3, 0.6), (150, 500), (0.2, 0.5), (0.02, 0.08))),
    "Shimmer":         dict(field=(0.4, (3,), (0.4, 0.8), (60, 250), (0.2, 0.5), (0.05, 0.3)), disp=(0.3, (0.1, 0.5))),
    "Dark Drones":     dict(breath=(0.6, (0.3, 0.8))),
    "Microtonal":      dict(field=(0.4, (1,), (0.3, 0.7), (20, 80), (0.0, 0.3), (0.05, 0.1))),
    "Keys":            dict(disp=(0.4, (0.05, 0.4))),
}


def motion_effects(r, group):
    """The group's field, dispersion and hall breath, drawn after its effects."""
    fx = MOTION_FX.get(group, {})
    if "field" in fx:
        p, modes, pull, radius, spring, wander = fx["field"]
        if r.rng.random() < p:
            r.set("field_mode", r.pick(*modes))
            r.set("field_strength", f(r.u(*pull), 2))
            r.set("field_radius", f(r.u(*radius), 0))
            r.set("field_spring", f(r.u(*spring), 2))
            r.set("field_rate", f(r.lg(*wander), 3))
            r.set("field_count", r.rng.randint(2, 6))
    if "disp" in fx:
        p, span = fx["disp"]
        if r.rng.random() < p:
            r.set("dispersion", f(r.u(*span) * (1 if r.rng.random() < 0.8 else -1), 2))
    if "breath" in fx:
        p, amount = fx["breath"]
        if r.rng.random() < p:
            r.set("modal_breath", f(r.u(*amount), 2))


# Phase 9 per group (12.10.2026), drawn last: memory = (probability, level, time s, bloom s);
# tune = (probability, self-tune); paint = (probability, paint), only where the preset has a modal
# hall; spec = (probability, level, time s, tilt, scatter, feedback). Binaural is the listener's and
# never set by a preset (Presets.h keptByPresets).
P9_FX = {
    "Sleep Concert":   dict(memory=(0.5, (0.3, 0.6), (60, 180), (3, 10)), paint=(0.3, (0.3, 0.7))),
    "Choirs":          dict(memory=(0.3, (0.2, 0.5), (40, 120), (2, 6)), tune=(0.4, (0.3, 0.8))),
    "Glass":           dict(spec=(0.4, (0.12, 0.25), (0.3, 1.2), (-0.5, 1.0), (0.1, 0.5), (0.3, 0.6))),
    "Bells":           dict(spec=(0.3, (0.10, 0.20), (0.5, 1.5), (-1.0, 0.5), (0.1, 0.4), (0.3, 0.6))),
    "Strings":         dict(memory=(0.3, (0.2, 0.4), (40, 120), (2, 6)), tune=(0.5, (0.3, 0.8))),
    "Organs":          dict(memory=(0.3, (0.2, 0.4), (40, 100), (2, 5)), tune=(0.4, (0.3, 0.7))),
    "Air & Noise":     dict(paint=(0.4, (0.3, 0.8)), spec=(0.3, (0.12, 0.25), (0.4, 1.5), (-0.5, 1.0), (0.3, 0.8), (0.3, 0.6))),
    "Deep Space":      dict(memory=(0.5, (0.3, 0.6), (90, 240), (5, 15)), spec=(0.4, (0.12, 0.25), (1.5, 3.0), (0.0, 1.0), (0.2, 0.6), (0.4, 0.7))),
    "Glissandi":       dict(spec=(0.4, (0.12, 0.25), (0.5, 1.5), (0.0, 1.0), (0.2, 0.6), (0.3, 0.6))),
    "Pendulums":       dict(spec=(0.3, (0.12, 0.25), (0.3, 1.0), (-0.5, 0.5), (0.0, 0.3), (0.3, 0.6))),
    "Waves":           dict(memory=(0.3, (0.2, 0.5), (60, 150), (3, 10)), paint=(0.4, (0.3, 0.8))),
    "Shepard & Tides": dict(paint=(0.5, (0.4, 0.9))),
    "Shimmer":         dict(spec=(0.5, (0.15, 0.30), (0.3, 1.2), (0.3, 1.0), (0.2, 0.6), (0.3, 0.7))),
    "Dark Drones":     dict(memory=(0.4, (0.3, 0.6), (90, 240), (5, 15))),
    "Microtonal":      dict(memory=(0.3, (0.2, 0.5), (60, 150), (3, 8))),
    "Keys":            dict(tune=(0.3, (0.3, 0.7))),
}


def p9_effects(r, group):
    """The group's memory, self-tuning, painted hall and spectral echo, drawn last."""
    fx = P9_FX.get(group, {})
    if "memory" in fx:
        p, level, life, bloom = fx["memory"]
        if r.rng.random() < p:
            r.set("memory_level", f(r.u(*level), 2))
            r.set("memory_time", f(r.lg(*life), 0))
            r.set("memory_bloom", f(r.lg(*bloom), 1))
            r.set("memory_blur", f(r.u(0.3, 0.8), 2))
            r.set("memory_sink", f(r.u(0.4, 0.9), 2))
    if "tune" in fx:
        p, grav = fx["tune"]
        if r.rng.random() < p:
            r.set("tune_gravity", f(r.u(*grav), 2))
            r.set("tune_anchor", f(r.u(0.05, 0.3), 2))
    if "paint" in fx and "modal_level" in r.p:
        p, amount = fx["paint"]
        if r.rng.random() < p:
            r.set("modal_paint", f(r.u(*amount), 2))
    if "spec" in fx:
        p, level, time, tilt, scatter, fb = fx["spec"]
        if r.rng.random() < p:
            r.set("spec_mix", f(r.u(*level), 3))
            r.set("spec_time", f(r.lg(*time), 3))
            r.set("spec_tilt", f(r.u(*tilt), 2))
            r.set("spec_scatter", f(r.u(*scatter), 2))
            r.set("spec_feedback", f(r.u(*fb), 2))
            r.set("spec_to_far", f(r.u(0.2, 0.7), 2))


def effects(r, group):
    """The group's echoes, room, modal hall and far network, drawn after its recipe."""
    fx = FX[group]
    if "echo" in fx:
        p, mix, time, fb = fx["echo"]
        if r.rng.random() < p:
            tl = r.lg(*time)
            r.set("delay_mix", f(r.u(*mix), 3))
            r.set("delay_time_l", f(tl, 3))
            r.set("delay_time_r", f(min(4.0, tl * r.u(1.2, 1.6)), 3))
            r.set("delay_feedback", f(r.u(*fb), 2))
            r.set("delay_to_far", f(r.u(0.3, 0.8), 2))
    if "echo2" in fx:
        p, mix = fx["echo2"]
        if r.rng.random() < p:
            r.set("delay2_mix", f(r.u(*mix), 3))
            r.set("delay2_time_l", f(r.lg(1.8, 3.2), 3))
            r.set("delay2_time_r", f(r.lg(2.4, 4.0), 3))
    if "room" in fx:
        early, size, near, decay = fx["room"]
        r.set("early_level", f(r.u(*early), 2))
        r.set("early_size", f(r.u(*size), 1))
        r.set("near_mix", f(r.u(*near), 2))
        r.set("near_decay", f(r.u(*decay), 2))
    if "modal" in fx:
        p, level, decay, damp, tune, spread = fx["modal"]
        if r.rng.random() < p:
            r.set("modal_level", f(r.u(*level), 3))
            r.set("modal_decay", f(r.lg(*decay), 1))
            r.set("modal_damp", f(r.u(*damp), 2))
            r.set("modal_tune", f(r.u(*tune), 2))
            r.set("modal_spread", f(r.u(*spread), 1))
            r.set("modal_drift", f(r.u(1.0, 8.0), 1))
    if "far" in fx:
        diffuse, mode = fx["far"]
        r.set("far_diffuse", f(r.u(*diffuse), 2))
        r.set("far_mode", r.rng.randint(*mode))


# ---- Phase 10 (12.10.2026): the living landscape, the sync, the flock, the ripple, the melt, the
# reservoir, and matrix rows from the new sources.

# Rows from the new sources, by kind: (source, target, depth range). Depths are in the target's
# normalised range; a unipolar source (Lenia, what is heard, the piece's) only adds.
POOLS = {
    "chaos": [("lorenz_x", "cloud_width", (0.08, 0.2)), ("lorenz_z", "tilt", (0.04, 0.1)), ("rossler_x", "drift", (0.1, 0.25)),
              ("rossler_y", "shimmer", (0.1, 0.25)), ("lorenz_y", "presence", (0.05, 0.12))],
    "kura": [("kura1", "cloud_width", (0.06, 0.15)), ("kura2", "shimmer", (0.1, 0.25)), ("kura3", "air", (0.1, 0.25)),
             ("kura4", "brightness", (0.06, 0.15))],
    "lenia": [("lenia1", "shimmer", (0.15, 0.35)), ("lenia2", "air", (0.15, 0.35)), ("lenia3", "presence", (0.1, 0.2)),
              ("lenia4", "cloud_width", (0.1, 0.25))],
    "heard": [("centroid", "brightness", (-0.25, -0.1)), ("loudness", "depth", (0.1, 0.25)), ("flux", "shimmer", (0.2, 0.4)),
              ("roughness", "tune_gravity", (0.3, 0.6))],
    "reservoir": [("reservoir1", "brightness", (0.08, 0.2)), ("reservoir2", "cloud_width", (0.1, 0.25)),
                  ("reservoir3", "air_center", (0.1, 0.25)), ("reservoir4", "tilt", (0.04, 0.1))],
    "piece": [("beat", "cloud_width", (0.06, 0.15)), ("cascade", "shimmer", (0.15, 0.3)), ("root_age", "depth", (0.1, 0.25)),
              ("random_note", "drift", (0.1, 0.25)), ("note", "tilt", (-0.1, -0.04))],
}
VIAS = ["root_age", "lenia2", "loudness", "kura3"]   # a via that is never 0 for long (no macro: they rest at 0)


def rows_from(r, pools, count, via_p=0.3):
    """Up to `count` rows from the pools, one per source, some through a via source."""
    used = {row.split(">")[0] for row in r.rows}
    for _ in range(count):
        src, target, (lo, hi) = r.pick(*POOLS[r.pick(*pools)])
        if src in used:
            continue
        used.add(src)
        via = r.pick(*VIAS) if r.rng.random() < via_p else None
        r.row(src, target, round(r.u(lo, hi), 3), via)


def set_life(r, mode, depth, rate, character, zoom):
    """The living landscape: 1 Lenia, 2 Turing."""
    r.set("life_mode", mode)
    r.set("life_depth", f(r.u(*depth), 2))
    r.set("life_rate", f(r.u(*rate), 2))
    r.set("life_character", f(r.u(*character), 2))
    r.set("life_zoom", f(r.u(*zoom), 2))


def set_sync(r, depth, coupling, rate, spread):
    """The partials falling into step."""
    r.set("sync_depth", f(r.u(*depth), 2))
    r.set("sync_coupling", f(r.u(*coupling), 2))
    r.set("sync_rate", f(r.lg(*rate), 3))
    r.set("sync_spread", f(r.u(*spread), 2))


def set_ripple(r, depth, speed, decay):
    """The wave along the spectrum, struck by every note."""
    r.set("ripple_depth", f(r.u(*depth), 2))
    r.set("ripple_speed", f(r.lg(*speed), 3))
    r.set("ripple_decay", f(r.lg(*decay), 2))


def set_melt(r, mix, time, spread):
    """The spectrum held and melting."""
    r.set("melt_mix", f(r.u(*mix), 3))
    r.set("melt_time", f(r.lg(*time), 2))
    r.set("melt_spread", f(r.u(*spread), 2))
    r.set("melt_to_far", f(r.u(0.2, 0.6), 2))


def set_flock(r, strength, wander, spring):
    """The clouds as flocks (the field's fourth mode)."""
    r.set("field_mode", 4)
    r.set("field_strength", f(r.u(*strength), 2))
    r.set("field_rate", f(r.lg(*wander), 3))
    r.set("field_spring", f(r.u(*spring), 2))


def set_reservoir(r, chaos, memory, inp):
    """The reservoir's character (it runs only where a row reads it)."""
    r.set("reservoir_chaos", f(r.u(*chaos), 2))
    r.set("reservoir_memory", f(r.lg(*memory), 2))
    r.set("reservoir_input", f(r.u(*inp), 2))


# Phase 10 in the first sixteen groups: life = (p, modes, depth, rate, character, zoom); sync = (p,
# depth, coupling, rate Hz, spread); ripple = (p, depth, speed oct/s, decay s); melt = (p, mix, time
# s, spread oct); flock = (p, strength, wander, spring), only where no field was drawn; rows = (p,
# pools, count).
P10_FX = {
    "Sleep Concert":   dict(life=(0.3, (1,), (0.3, 0.6), (0.05, 0.3), (0.1, 0.5), (0.5, 1.0)),
                            melt=(0.4, (0.2, 0.4), (15, 40), (0.2, 0.6)), rows=(0.6, ["chaos", "kura", "lenia"], 1)),
    "Choirs":          dict(sync=(0.3, (0.2, 0.5), (0.4, 0.8), (0.2, 1.0), (0.2, 0.5)),
                            melt=(0.3, (0.15, 0.35), (8, 25), (0.2, 0.6)), rows=(0.5, ["heard", "kura"], 1)),
    "Glass":           dict(ripple=(0.5, (0.4, 0.8), (0.5, 2.5), (3, 12)), rows=(0.4, ["heard", "chaos"], 1)),
    "Bells":           dict(ripple=(0.5, (0.4, 0.8), (0.3, 1.5), (4, 15)), melt=(0.3, (0.2, 0.4), (10, 30), (0.3, 0.9)),
                            rows=(0.4, ["kura", "piece"], 1)),
    "Strings":         dict(sync=(0.3, (0.2, 0.4), (0.3, 0.7), (0.2, 0.8), (0.2, 0.5)), flock=(0.3, (0.4, 0.8), (0.05, 0.3), (0.1, 0.4)),
                            rows=(0.5, ["kura", "chaos"], 1)),
    "Organs":          dict(sync=(0.4, (0.15, 0.35), (0.6, 1.0), (4.0, 6.0), (0.05, 0.2)), rows=(0.4, ["heard", "piece"], 1)),
    "Air & Noise":     dict(life=(0.4, (2,), (0.4, 0.7), (0.2, 0.6), (0.0, 1.0), (0.3, 1.0)),
                            melt=(0.4, (0.2, 0.5), (6, 20), (0.8, 1.5)), rows=(0.6, ["reservoir", "lenia"], 1)),
    "Deep Space":      dict(life=(0.3, (1,), (0.3, 0.6), (0.05, 0.2), (0.1, 0.5), (0.5, 1.0)),
                            melt=(0.5, (0.3, 0.6), (20, 60), (0.3, 1.0)), rows=(0.7, ["chaos", "reservoir"], 2)),
    "Glissandi":       dict(flock=(0.3, (0.5, 0.9), (0.1, 0.4), (0.1, 0.3)), ripple=(0.3, (0.3, 0.6), (0.5, 2.0), (4, 12)),
                            rows=(0.4, ["chaos"], 1)),
    "Pendulums":       dict(sync=(0.6, (0.4, 0.8), (0.5, 1.0), (0.3, 1.5), (0.1, 0.4)), rows=(0.4, ["kura"], 1)),
    "Waves":           dict(ripple=(0.6, (0.4, 0.8), (0.3, 1.2), (6, 20)), life=(0.2, (2,), (0.3, 0.6), (0.1, 0.4), (0.2, 0.8), (0.4, 1.0)),
                            rows=(0.4, ["chaos", "kura"], 1)),
    "Shepard & Tides": dict(melt=(0.3, (0.2, 0.4), (10, 30), (0.3, 0.8)), rows=(0.5, ["kura", "piece"], 1)),
    "Shimmer":         dict(sync=(0.3, (0.2, 0.4), (0.4, 0.8), (2.0, 5.0), (0.2, 0.6)),
                            life=(0.3, (2,), (0.3, 0.6), (0.3, 0.8), (0.0, 1.0), (0.3, 1.0)), rows=(0.5, ["heard", "lenia"], 1)),
    "Dark Drones":     dict(melt=(0.5, (0.3, 0.6), (20, 60), (0.2, 0.8)), life=(0.3, (1,), (0.3, 0.6), (0.05, 0.2), (0.1, 0.5), (0.5, 1.0)),
                            rows=(0.6, ["reservoir", "chaos"], 1)),
    "Microtonal":      dict(sync=(0.3, (0.2, 0.5), (0.4, 0.8), (0.3, 2.0), (0.1, 0.4)), rows=(0.6, ["piece"], 1)),
    "Keys":            dict(ripple=(0.6, (0.4, 0.8), (0.5, 2.5), (3, 12)), melt=(0.3, (0.15, 0.35), (6, 20), (0.2, 0.6)),
                            rows=(0.5, ["heard", "piece"], 1)),
}


def p10_effects(r, group):
    """Phase 10 in one of the first sixteen groups, drawn after everything else."""
    fx = P10_FX.get(group, {})
    if "life" in fx:
        p, modes, *ranges = fx["life"]
        if r.rng.random() < p:
            set_life(r, r.pick(*modes), *ranges)
    if "sync" in fx:
        p, *ranges = fx["sync"]
        if r.rng.random() < p:
            set_sync(r, *ranges)
    if "ripple" in fx:
        p, *ranges = fx["ripple"]
        if r.rng.random() < p:
            set_ripple(r, *ranges)
    if "melt" in fx:
        p, *ranges = fx["melt"]
        if r.rng.random() < p:
            set_melt(r, *ranges)
    if "flock" in fx:
        p, *ranges = fx["flock"]
        if r.rng.random() < p and "field_mode" not in r.p:
            set_flock(r, *ranges)
    if "rows" in fx:
        p, pools, count = fx["rows"]
        if r.rng.random() < p:
            rows_from(r, pools, count)
            if any(row.startswith("reservoir") for row in r.rows):
                set_reservoir(r, (0.85, 1.2), (2, 10), (0.5, 1.2))


# ---- The living systems: sixteen groups, each from one of the first sixteen's sounds.

BASE = dict(zip(GROUPS[:16], RECIPES))


def based(r, *bases):
    """A new group's sound: one of the first sixteen groups' recipes with all its effects."""
    group = r.pick(*bases)
    BASE[group](r)
    effects(r, group)
    motion_effects(r, group)
    p9_effects(r, group)
    return group


ALL = tuple(GROUPS[:16])


def n_living(r):
    based(r, "Sleep Concert", "Choirs", "Strings", "Deep Space", "Shimmer", "Shepard & Tides")
    set_life(r, 1, (0.45, 0.85), (0.08, 0.5), (0.1, 0.6), (0.4, 1.0))
    r.set("lenia_rate", f(r.lg(1, 8), 2))
    rows_from(r, ["lenia"], 2)
    rows_from(r, ["kura"], 1)


def n_turing(r):
    based(r, "Glass", "Air & Noise", "Shimmer", "Waves", "Bells", "Choirs")
    set_life(r, 2, (0.45, 0.85), (0.15, 0.7), (0.0, 1.0), (0.3, 1.0))
    rows_from(r, ["lenia", "heard"], 2)


def n_sync(r):
    based(r, "Choirs", "Strings", "Organs", "Sleep Concert", "Pendulums", "Glass")
    set_sync(r, (0.45, 0.9), (0.5, 1.0), (0.2, 3.0), (0.1, 0.5))
    r.row("kura1", "sync_depth", round(r.u(0.1, 0.25), 3))
    rows_from(r, ["kura"], 1)


def n_flock(r):
    based(r, "Glissandi", "Waves", "Strings", "Air & Noise", "Shimmer", "Choirs")
    set_flock(r, (0.5, 1.0), (0.08, 0.6), (0.1, 0.4))
    rows_from(r, ["chaos"], 2)


def n_ripple(r):
    based(r, "Keys", "Bells", "Glass", "Pendulums", "Waves", "Strings")
    set_ripple(r, (0.5, 1.0), (0.3, 2.5), (3, 20))
    if r.p.get("brain_on") != 0:   # the conductor changes notes often: every change strikes the string
        r.set("auto_rate", f(r.lg(8, 30), 1))
    rows_from(r, ["heard", "piece"], 1)


def n_melt(r):
    based(r, "Sleep Concert", "Deep Space", "Dark Drones", "Choirs", "Bells", "Strings")
    set_melt(r, (0.4, 0.8), (10, 50), (0.3, 1.5))
    if r.rng.random() < 0.4:
        r.row("lorenz_x", "melt_spread", round(r.u(0.1, 0.25), 3))
    rows_from(r, ["chaos", "kura"], 1)


def n_cascade(r):
    based(r, "Glass", "Bells", "Shimmer", "Keys", "Pendulums", "Microtonal")
    r.set("spec_mix", f(r.u(0.2, 0.4), 3))
    r.set("spec_time", f(r.lg(0.3, 2.5), 3))
    r.set("spec_tilt", f(r.u(-1, 1), 2))
    r.set("spec_scatter", f(r.u(0.3, 0.9), 2))
    r.set("spec_feedback", f(r.u(0.5, 0.85), 2))
    r.set("spec_damp", f(r.u(0.2, 0.6), 2))
    r.set("spec_to_far", f(r.u(0.2, 0.6), 2))
    if r.rng.random() < 0.4:
        set_melt(r, (0.15, 0.3), (5, 15), (0.2, 0.6))
    rows_from(r, ["piece", "heard"], 1)


def n_halls(r):
    based(r, "Bells", "Organs", "Choirs", "Dark Drones", "Microtonal", "Glass")
    r.set("modal_level", f(r.u(0.3, 0.5), 3))
    r.set("modal_decay", f(r.lg(20, 60), 1))
    r.set("modal_damp", f(r.u(0.2, 0.5), 2))
    r.set("modal_tune", f(r.u(0.5, 1.0), 2))
    r.set("modal_spread", f(r.u(2, 10), 1))
    r.set("modal_breath", f(r.u(0.3, 0.9), 2))
    r.set("modal_paint", f(r.u(0.4, 1.0), 2))
    if r.rng.random() < 0.5:
        r.row("kura2", "modal_level", round(r.u(0.05, 0.15), 3))
    rows_from(r, ["kura", "lenia"], 1)


def n_ghosts(r):
    based(r, "Sleep Concert", "Strings", "Keys", "Choirs", "Deep Space", "Organs")
    r.set("memory_level", f(r.u(0.4, 0.7), 2))
    r.set("memory_time", f(r.lg(90, 300), 0))
    r.set("memory_bloom", f(r.lg(4, 15), 1))
    r.set("memory_blur", f(r.u(0.5, 1.0), 2))
    r.set("memory_sink", f(r.u(0.4, 0.9), 2))
    r.set("tune_gravity", f(r.u(0.2, 0.6), 2))
    r.row("root_age", "memory_level", round(r.u(0.1, 0.25), 3))
    rows_from(r, ["piece"], 1)


def n_attractors(r):
    based(r, *ALL)
    r.set("chaos_period", f(r.lg(20, 200), 1))
    rows_from(r, ["chaos"], 3)


def n_reservoir(r):
    based(r, *ALL)
    set_reservoir(r, (0.85, 1.25), (1, 15), (0.5, 1.5))
    rows_from(r, ["reservoir"], 3, via_p=0.2)
    if not any(row.startswith("reservoir") for row in r.rows):   # the pool's draws may all have been taken
        r.row("reservoir1", "brightness", round(r.u(0.08, 0.2), 3))


def n_listening(r):
    based(r, *ALL)
    rows_from(r, ["heard"], 3, via_p=0.15)


def n_gravity(r):
    based(r, "Microtonal", "Glass", "Organs", "Shimmer", "Choirs", "Bells")
    r.set("field_mode", r.pick(1, 2, 3))
    r.set("field_strength", f(r.u(0.5, 0.9), 2))
    r.set("field_radius", f(r.u(30, 300), 0))
    r.set("field_spring", f(r.u(0.1, 0.5), 2))
    r.set("field_rate", f(r.lg(0.02, 0.2), 3))
    r.set("field_count", r.rng.randint(2, 6))
    r.set("tune_gravity", f(r.u(0.3, 0.8), 2))
    if r.rng.random() < 0.5:
        r.set("dispersion", f(r.u(0.2, 0.8), 2))
    rows_from(r, ["chaos", "kura"], 1)


def n_smear(r):
    based(r, "Glass", "Bells", "Keys", "Air & Noise", "Glissandi", "Shimmer")
    r.set("dispersion", f(r.u(0.5, 1.5) * (1 if r.rng.random() < 0.7 else -1), 2))
    set_melt(r, (0.25, 0.5), (4, 15), (0.8, 2.0))
    rows_from(r, ["heard"], 1)


def n_lattice(r):
    based(r, "Microtonal", "Organs", "Glass", "Strings", "Choirs")
    r.set("scale", r.pick(SCALE["JI Major"], SCALE["JI Minor"], SCALE["JI 7-limit"], SCALE["JI Pentatonic"], SCALE["Otonality"]))
    r.set("purity", f(r.u(0.5, 0.9), 2))   # not quite pure: the beats are what this group plays with
    set_sync(r, (0.2, 0.45), (0.6, 1.0), (2.0, 6.0), (0.05, 0.25))
    rows_from(r, ["piece"], 2)
    if not any(row.startswith("beat") for row in r.rows):
        r.row("beat", "cloud_width", round(r.u(0.06, 0.15), 3))


def n_night(r):
    based(r, "Sleep Concert", "Deep Space", "Shepard & Tides", "Choirs", "Dark Drones")
    if r.rng.random() < 0.6:
        set_life(r, 1, (0.3, 0.6), (0.05, 0.25), (0.1, 0.5), (0.5, 1.0))
    set_melt(r, (0.2, 0.45), (15, 45), (0.2, 0.7))
    r.set("memory_level", f(r.u(0.2, 0.45), 2))
    r.set("memory_time", f(r.lg(60, 200), 0))
    rows_from(r, ["kura", "lenia", "chaos"], 2)


RECIPES = RECIPES + [n_living, n_turing, n_sync, n_flock, n_ripple, n_melt, n_cascade, n_halls,
                     n_ghosts, n_attractors, n_reservoir, n_listening, n_gravity, n_smear, n_lattice, n_night]


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
            if g < 16:   # a new group's recipe brings its sound's effects itself
                effects(r, group)
                motion_effects(r, group)
                p9_effects(r, group)
                p10_effects(r, group)
            assert len(r.rows) <= 16, (group, i, r.rows)
            name = names[i]
            if name in levels:   # the measured level, brought to the common one (at most 18 dB either way)
                r.set("master_gain", f(max(-18.0, min(12.0, TARGET_DB - levels[name])), 2))
            bank.append((name, g, " ".join(f"{k}={v}" for k, v in r.p.items()), ";".join(r.rows)))
    all_names = [name for name, *_ in bank]
    assert len(set(all_names)) == len(all_names), "preset names repeat"
    return bank


def main():
    bank = build()
    lines = ["// Generated by Tools/presets/make_presets.py -- do not edit; edit the script and run it again.",
             "/// The factory presets: 32 groups of 64 (Presets.h).",
             "const Preset kPresets[] = {"]
    for name, g, params, mod in bank:
        lines.append(f'    {{ "{name}", {g}, "{params}", "{mod}" }},')
    lines.append("};")
    OUT.write_bytes(("\n".join(lines) + "\n").encode("utf-8"))
    levelled = "levelled" if LEVELS.exists() else "not levelled yet (run additiv_presets)"
    print(f"{len(bank)} presets in {len(GROUPS)} groups -> {OUT} ({levelled})")


if __name__ == "__main__":
    main()
