"""make_presets.py: the factory bank -- 64 groups of 64 presets -- into Core/src/Presets.inc.

Every group is a recipe: the ranges its sound lives in, drawn by a seeded die, so the bank is the
same on every run. A preset names only what it changes (Presets.h). The four macros mean the same
in every preset: 1 the motion, 2 the brightness (the cutoff where a filter is on), 3 the air, 4 the
space; the mod wheel opens the brightness too, pressure lifts the presence.

The first sixteen groups are the sounds; the second sixteen (the living systems, 12.10.2026) each
start from one of them and put one of the systems in front: the living landscape, the sync, the
flock, the ripple, the melt, the spectral echo, the modal hall, the memory, the attractors, the
reservoir, what the instrument hears, the field, the dispersion, beating lattices. Phase 10 was
added to the first sixteen too, drawn after everything else, so what they had stays as it was.

The third sixteen are sleep concerts (12.10.2026): each again from one of the first sixteen's
sounds, made slow, dark and without anything sudden (sleepy()), and moved by modulations made
for the night -- a breath of five to seven a minute, tides of minutes, a smooth wander, chaos
over half an hour, Kuramoto in step, slow Lenia, the night deepening while the harmony stands
(Root Age), calming answers to what is heard, a reservoir with a long memory.

The fourth sixteen are visions (13.10.2026): the animated pictures of the living landscape
(KaleidoscopeEnhanced's transforms), the spectral comb, Stretch and Shift, each group with one in
front. Phase 13 was added to the first forty-eight too, drawn after everything else.

Then the swarms and currents (Phase 16), the halls and orbits (Phase 17), and the harmonics' lives
(Phase 26, 15.10.2026): registers that overblow, seasons of overtone families, avalanches on a rice
pile, subharmonics pumped from below, grains, the horizon and breathers -- sixteen groups, and the new
means in the groups before where they fit, drawn after everything else (the conductor's floor too).

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
          "Smear & Dispersion", "Lattices & Beats", "Night Gardens",
          "Lullabies", "Slow Breath", "Deep Sleep", "Dream Drift", "Moonlight", "Night Rain", "Ocean Sleep", "Cradle",
          "Starfield", "Hypnagogia", "Twilight Choirs", "Warm Drones", "Snowfall", "Forest Night", "Slow Tides", "Afterglow",
          "Kaleidoscopes", "Tunnels", "Spirals", "Ink & Flow", "Feedback Loops", "Raindrops", "Star Streams", "Moire Veils",
          "Stretched Worlds", "Shifted Metals", "Comb Phasers", "Barber Poles", "Painted Light", "Prisms", "Alien Harmony", "Dream Machines",
          "Swarms", "Splinters", "Chimeras", "Twin Swarms", "Currents", "Vortex Streets", "Ink Plumes", "Islands",
          "Labyrinths", "Archipelagos", "Gong Blooms", "Bronze Swells", "Shimmer Cascades", "Pressure Pads", "Expressive Voices", "Living Storms",
          "Growing Halls", "Eroded Halls", "Cathedral Years", "Worn Stone", "Slow Clocks", "Quickened Air", "Gravity Lenses", "Lingering Waves",
          "Orbits", "Cyclotrons", "Planetaria", "Magnetic Choirs", "Event Horizons", "Memory Palaces", "Celestial Mechanics", "Living Architecture",
          "Overblown Flutes", "Breath Registers", "Four Seasons", "Turning Years", "Avalanches", "Glacier Calving", "Subterranea", "Period Doubling",
          "Halo Horizons", "Aurora Curtains", "Rainfall Grains", "Breathing Grains", "Breather Chains", "Harmonic Ecology", "Night Ecology", "Wild Harmonics"]

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
    "Lullabies": (["Hushing", "Lilting", "Rocking", "Humming", "Tender", "Cooing", "Swaying", "Mild"],
                  ["Lullaby", "Berceuse", "Nursery", "Song", "Blanket", "Pillow", "Hush", "Rhyme"]),
    "Slow Breath": (["Inhaling", "Exhaling", "Breathing", "Resting", "Sighing", "Easing", "Settling", "Calming"],
                    ["Breath", "Lungs", "Rhythm", "Pause", "Exhale", "Swell", "Pulse", "Ease"]),
    "Deep Sleep": (["Dreamless", "Heavy", "Sunken", "Dormant", "Unconscious", "Delta", "Still", "Weighted"],
                   ["Slumber", "Sleep", "Depths", "Hibernation", "Repose", "Stillness", "Rest", "Torpor"]),
    "Dream Drift": (["Dozing", "Wandering", "Floating", "Drifting", "Unmoored", "Weightless", "Hazy", "Meandering"],
                    ["Reverie", "Dreamscape", "Current", "Raft", "Feather", "Haze", "Float", "Balloon"]),
    "Moonlight": (["Silvered", "Pale", "Moonbeam", "Lunar", "Crescent", "Waning", "Waxing", "Gibbous"],
                  ["Glow", "Beam", "Path", "Night", "Silver", "Reflection", "Light", "Halo"]),
    "Night Rain": (["Pattering", "Drizzling", "Misting", "Dripping", "Steady", "Quiet", "Distant", "Warm"],
                   ["Rain", "Drizzle", "Roof", "Window", "Gutter", "Puddles", "Showers", "Downpour"]),
    "Ocean Sleep": (["Undulating", "Lulling", "Swaying", "Briny", "Seaward", "Rolling", "Pelagic", "Abyssal"],
                    ["Sea", "Swells", "Bay", "Lagoon", "Kelp", "Shoreline", "Fathoms", "Seabed"]),
    "Cradle": (["Rocked", "Cradled", "Nested", "Wrapped", "Soothing", "Lulled", "Cosy", "Snug"],
               ["Hammock", "Bassinet", "Crib", "Nest", "Swing", "Arms", "Rocker", "Shelter"]),
    "Starfield": (["Starry", "Twinkling", "Sidereal", "Galactic", "Celestial", "Faraway", "Faint", "Polar"],
                  ["Stars", "Constellation", "Sky", "Milky Way", "Vault", "Firmament", "Meridian", "Zenith"]),
    "Hypnagogia": (["Hypnagogic", "Half-Asleep", "Vanishing", "Liminal", "Drowsing", "Twilit", "Unfocused", "Nodding"],
                   ["Threshold", "Visions", "Images", "Edge", "Doorway", "Murmurs", "Shapes", "Glimpses"]),
    "Twilight Choirs": (["Vesper", "Evening", "Compline", "Dimmed", "Faraway", "Murmured", "Hushed", "Softened"],
                        ["Choir", "Voices", "Hymn", "Chant", "Psalm", "Anthem", "Carol", "Canticle"]),
    "Warm Drones": (["Warm", "Glowing", "Mellow", "Woollen", "Amber", "Hearth", "Embered", "Velvety"],
                    ["Drone", "Hum", "Organ", "Bass", "Ground", "Pedal", "Murmur", "Bed"]),
    "Snowfall": (["Snowy", "Falling", "Powdered", "Silent", "Settling", "Hushed", "White", "Crystalline"],
                 ["Snow", "Flakes", "Snowfall", "Flurry", "Hoarfrost", "Crystals", "Blanket", "Field"]),
    "Forest Night": (["Wooded", "Shadowed", "Ferny", "Rustling", "Silent", "Midnight", "Pine", "Hollow"],
                     ["Forest", "Woods", "Canopy", "Thicket", "Clearing", "Pines", "Undergrowth", "Copse"]),
    "Slow Tides": (["Ebbing", "Flowing", "Receding", "Neap", "Spring", "Lunar", "Turning", "Retreating"],
                   ["Ebb", "Flow", "Estuary", "Mudflats", "Shallows", "Sandbar", "Inlet", "Waterline"]),
    "Afterglow": (["Lingering", "Golden", "Fading", "Embered", "Rosy", "Dimming", "Last", "Sunset"],
                  ["Glow", "Afterglow", "Embers", "Light", "Dusk", "Horizon", "Warmth", "Evening"]),
    "Kaleidoscopes": (["Folded", "Mirrored", "Faceted", "Turning", "Jewelled", "Hexagonal", "Radiant", "Symmetric"],
                      ["Kaleidoscope", "Rosette", "Mandala", "Facets", "Mosaic", "Snowflake", "Medallion", "Lantern"]),
    "Tunnels": (["Endless", "Boring", "Hollow", "Rushing", "Receding", "Curving", "Burrowing", "Spinning"],
                ["Tunnel", "Passage", "Corridor", "Tube", "Shaft", "Wormhole", "Burrow", "Conduit"]),
    "Spirals": (["Coiling", "Winding", "Unwinding", "Whorled", "Nautilus", "Golden", "Logarithmic", "Droste"],
                ["Spiral", "Helix", "Vortex", "Whorl", "Coil", "Shell", "Galaxy", "Fern"]),
    "Ink & Flow": (["Swirling", "Flowing", "Streaming", "Curling", "Eddying", "Pooling", "Seeping", "Drifting"],
                   ["Ink", "Marble", "Smoke", "Silk", "Dye", "Eddies", "Plumes", "Currents"]),
    "Feedback Loops": (["Recursive", "Iterated", "Self-Similar", "Fractal", "Nested", "Regressing", "Returning", "Folding"],
                       ["Loop", "Recursion", "Iteration", "Cascade", "Regress", "Abyss", "Repetition", "Echoes"]),
    "Raindrops": (["Splashing", "Beading", "Trickling", "Rainy", "Glistening", "Wet", "Tapping", "Spattering"],
                  ["Raindrops", "Droplets", "Pond", "Ripples", "Splash", "Rings", "Dew", "Beads"]),
    "Star Streams": (["Streaming", "Warping", "Hyperspace", "Rushing", "Shooting", "Comet", "Stellar", "Interstellar"],
                     ["Stars", "Comets", "Meteors", "Lightspeed", "Stardust", "Photons", "Starlight", "Trails"]),
    "Moire Veils": (["Interfering", "Shimmering", "Overlaid", "Rippled", "Gauzy", "Sheer", "Layered", "Silken"],
                    ["Veil", "Moire", "Gauze", "Organza", "Tulle", "Screen", "Mesh", "Curtain"]),
    "Stretched Worlds": (["Elongated", "Warped", "Alien", "Elastic", "Distended", "Strange", "Otherworldly", "Tensile"],
                         ["World", "Harmony", "Octave", "Planet", "Realm", "Horizon", "Spectrum", "Continent"]),
    "Shifted Metals": (["Shifted", "Metallic", "Ringing", "Clanging", "Brazen", "Tinny", "Steel", "Chrome"],
                       ["Metal", "Gong", "Plate", "Chime", "Anvil", "Wire", "Cymbal", "Girder"]),
    "Comb Phasers": (["Phasing", "Combed", "Notched", "Sweeping", "Swirled", "Whooshing", "Flanged", "Filtered"],
                     ["Phaser", "Comb", "Jet", "Notch", "Sweep", "Wash", "Swirl", "Flanger"]),
    "Barber Poles": (["Ever-Rising", "Ever-Falling", "Unending", "Circling", "Barber", "Escalating", "Perpetual", "Revolving"],
                     ["Pole", "Stair", "Climb", "Descent", "Helix", "Illusion", "Paradox", "Ascension"]),
    "Painted Light": (["Painted", "Brushed", "Gilded", "Washed", "Tinted", "Glazed", "Luminous", "Pastel"],
                      ["Light", "Canvas", "Fresco", "Window", "Glass", "Icon", "Panel", "Lantern"]),
    "Prisms": (["Prismatic", "Refracting", "Splitting", "Chromatic", "Beaming", "Rainbowed", "Angled", "Bent"],
               ["Prisms", "Spectrum", "Rays", "Beams", "Colours", "Bands", "Shards", "Lights"]),
    "Alien Harmony": (["Xenial", "Extraterrestrial", "Foreign", "Unearthly", "Martian", "Exotic", "Uncanny", "Outlandish"],
                      ["Harmony", "Chord", "Choir", "Hymn", "Anthem", "Consonance", "Concord", "Song"]),
    "Dream Machines": (["Dreaming", "Clockwork", "Humming", "Whirring", "Lucid", "Mechanical", "Electric", "Somnambulant"],
                       ["Machine", "Engine", "Automaton", "Device", "Contraption", "Apparatus", "Loom", "Orrery"]),
    # Phase 16
    "Swarms": (["Gyrating", "Rotating", "Wheeling", "Spiralling", "Pirouetting", "Careering", "Looping", "Coursing"],
               ["Swarm", "Gyre", "Carousel", "Maelstrom", "Roundel", "Whirligig", "Merry-Go-Round", "Starling Ring"]),
    "Splinters": (["Splintered", "Shattered", "Sundered", "Cleft", "Parted", "Riven", "Severed", "Fissured"],
                  ["Phases", "Shoals", "Circles", "Arcs", "Fragments", "Facets", "Splinters", "Shards of Light"]),
    "Chimeras": (["Half-Lit", "Twofold", "Janus", "Uneven", "Partial", "Ambivalent", "Two-Faced", "Patchwork"],
                 ["Chimera", "Hybrid", "Kernel", "Mirage", "Sphinx", "Griffin", "Centaur", "Mosaic"]),
    "Twin Swarms": (["Twin", "Paired", "Antiphonal", "Alternating", "Answering", "Doubled", "Opposed", "Reciprocal"],
                    ["Beacons", "Signals", "Pendants", "Lamps", "Calls", "Lanterns", "Torches", "Semaphores"]),
    "Currents": (["Viscous", "Honeyed", "Syrupy", "Languid", "Slow-Moving", "Treacly", "Unhurried", "Oozing"],
                 ["Current", "Stream", "Seep", "Eddy", "Ooze", "Rivulet", "Runnel", "Trickle"]),
    "Vortex Streets": (["Churning", "Roiling", "Seething", "Boiling", "Racing", "Restless", "Tumbling", "Swirling"],
                       ["Vortices", "Street", "Torrent", "Rapids", "Whirlpool", "Race", "Wash", "Overfall"]),
    "Ink Plumes": (["Inked", "Plumed", "Indigo", "Smoky", "Billowing", "Trailing", "Inky", "Wafting"],
                   ["Plume", "Cloudlet", "Wisp", "Tendril", "Inkdrop", "Sepia Cloud", "Smoke Ring", "Pall"]),
    "Islands": (["Gathering", "Merging", "Coalescing", "Settling", "Joining", "Pooling", "Floating", "Ripening"],
                ["Islands", "Isles", "Atolls", "Reefs", "Continents", "Skerries", "Holms", "Sandbanks"]),
    "Labyrinths": (["Winding", "Mazy", "Serpentine", "Tangled", "Meandering", "Coiled", "Intricate", "Veined"],
                   ["Labyrinth", "Maze", "Channels", "Pathways", "Corridors", "Fretwork", "Tracery", "Warren"]),
    "Archipelagos": (["Becalmed", "Misted", "Tranquil", "Hazy", "Slumbering", "Shrouded", "Dim", "Lulled"],
                     ["Archipelago", "Lagoon", "Haven", "Cove", "Inlet", "Shallows", "Estuary", "Mooring"]),
    "Gong Blooms": (["Blooming", "Flaring", "Swelling", "Kindled", "Thundering", "Shivering", "Resounding", "Bursting"],
                    ["Gong", "Tam-Tam", "Plate", "Disc", "Shield", "Cymbal", "Sheet", "Tympan"]),
    "Bronze Swells": (["Gilded", "Brazen", "Burnished", "Tawny", "Coppery", "Auric", "Rusted", "Patinated"],
                      ["Swell", "Surge", "Crescendo", "Billow", "Upwelling", "Heave", "Groundswell", "Tidewater"]),
    "Shimmer Cascades": (["Sparkling", "Glittering", "Showering", "Sifting", "Twinkling", "Spilling", "Raining", "Dripping"],
                         ["Cascade", "Fountain", "Spray", "Spangles", "Glitter", "Motes", "Droplets", "Sequins"]),
    "Pressure Pads": (["Pressed", "Caressed", "Held", "Leaning", "Pushing", "Weighted", "Tender", "Firm"],
                      ["Pad", "Cushion", "Felt", "Palm", "Fingertips", "Hands", "Contact", "Pressure"]),
    "Expressive Voices": (["Singing", "Sighing", "Calling", "Yearning", "Pleading", "Warbling", "Keening", "Crooning"],
                          ["Voice", "Song", "Lament", "Cantus", "Solo", "Plea", "Lilt", "Cantilena"]),
    "Living Storms": (["Storming", "Tempestuous", "Gusting", "Squalling", "Thunderous", "Charged", "Wind-Torn", "Brooding"],
                      ["Storm", "Tempest", "Gale", "Cyclone", "Front", "Monsoon", "Cloudburst", "Thunderhead"]),
    # Phase 17
    "Growing Halls": (["Growing", "Ripening", "Burgeoning", "Deepening", "Widening", "Maturing", "Awakening", "Unfurling"],
                      ["Hall", "Nave", "Rotunda", "Atrium", "Chancel", "Cloister", "Gallery", "Transept"]),
    "Eroded Halls": (["Eroded", "Weathered", "Worn", "Crumbling", "Sandblown", "Wind-Scoured", "Wasted", "Thinned"],
                     ["Ruin", "Colonnade", "Arcade", "Crypt", "Undercroft", "Portico", "Ambulatory", "Apse"]),
    "Cathedral Years": (["Ancient", "Timeless", "Centuried", "Patient", "Ageless", "Venerable", "Enduring", "Abiding"],
                        ["Cathedral", "Basilica", "Minster", "Abbey", "Sanctuary", "Chapel", "Vault", "Dome"]),
    "Worn Stone": (["Smoothed", "Polished", "Rounded", "Softened", "Footworn", "Rainworn", "Mossed", "Lichened"],
                   ["Stone", "Step", "Threshold", "Flagstone", "Pillar", "Lintel", "Plinth", "Cornerstone"]),
    "Slow Clocks": (["Dilated", "Stretched", "Lagging", "Drowsing", "Lengthened", "Unhastened", "Elongated", "Tarrying"],
                    ["Clock", "Hour", "Escapement", "Hourglass", "Dial", "Chronometer", "Metronome", "Sundial"]),
    "Quickened Air": (["Quickened", "Hastened", "Brisk", "Fleet", "Nimble", "Darting", "Flickering", "Skittering"],
                      ["Air", "Breeze", "Zephyr", "Draught", "Flurry", "Updraft", "Whirl", "Sough"]),
    "Gravity Lenses": (["Lensed", "Curved", "Warped", "Bent", "Refracted", "Focused", "Magnified", "Arced"],
                       ["Lens", "Halo", "Ring of Light", "Arc", "Mirage", "Caustic", "Corona", "Glory"]),
    "Lingering Waves": (["Lingering", "Pooling", "Gathered", "Lulled", "Stilled", "Trapped", "Dawdling", "Loitering"],
                        ["Wave", "Ripple", "Swell", "Undulation", "Wavelet", "Billow", "Roller", "Surge"]),
    "Orbits": (["Orbiting", "Circling", "Revolving", "Encircling", "Rounding", "Precessing", "Satellite", "Ringed"],
               ["Orbit", "Ellipse", "Perihelion", "Apogee", "Epicycle", "Ring", "Circuit", "Path"]),
    "Cyclotrons": (["Magnetic", "Ionised", "Spinning", "Whirring", "Gyral", "Accelerated", "Polar", "Electrified"],
                   ["Cyclotron", "Gyrator", "Spindle", "Coil", "Ion", "Particle", "Helix", "Dynamo"]),
    "Planetaria": (["Celestial", "Sidereal", "Planetary", "Heavenly", "Astral", "Starlit", "Zodiacal", "Nightbound"],
                   ["Planetarium", "Orrery", "Sphere", "Moon", "Planet", "Comet", "Equinox", "Solstice"]),
    "Magnetic Choirs": (["Polarised", "Attracted", "Drawn", "Magnetised", "Aligned", "Tugged", "Swaying", "Lodestone"],
                        ["Choir", "Chorale", "Chant", "Antiphon", "Psalm", "Motet", "Canticle", "Vesper"]),
    "Event Horizons": (["Infinite", "Frozen", "Suspended", "Boundless", "Eternal", "Fathomless", "Dark-Bound", "Unending"],
                       ["Horizon", "Singularity", "Threshold", "Edge", "Event", "Abyss", "Rim", "Brink"]),
    "Memory Palaces": (["Remembered", "Recalled", "Haunted", "Treasured", "Storied", "Kept", "Unforgotten", "Familiar"],
                       ["Palace", "Chamber", "Room", "Corridor", "Library", "Archive", "Gallery", "Mansion"]),
    "Celestial Mechanics": (["Clockwork", "Precise", "Elliptic", "Periodic", "Orbital", "Harmonic", "Gravitating", "Tidal"],
                            ["Mechanics", "Ephemeris", "Almanac", "Transit", "Conjunction", "Syzygy", "Eclipse", "Occultation"]),
    "Living Architecture": (["Living", "Breathing", "Growing", "Shifting", "Sentient", "Evolving", "Organic", "Unfolding"],
                            ["Architecture", "Edifice", "Structure", "Tower", "Spire", "Citadel", "Pavilion", "Bastion"]),
    "Overblown Flutes": (["Overblown", "Breathy", "Fluttering", "Reedy", "Windblown", "Piping", "Whistling", "Airy"],
                         ["Flute", "Shakuhachi", "Ney", "Fife", "Pipe", "Recorder", "Ocarina", "Panpipe"]),
    "Breath Registers": (["Rising", "Swelling", "Lungful", "Exhaled", "Inhaled", "Billowing", "Sighing", "Bellowed"],
                         ["Register", "Bellows", "Windchest", "Tremulant", "Ventil", "Wind Line", "Pallet", "Tracker"]),
    "Four Seasons": (["Vernal", "Estival", "Autumnal", "Hibernal", "Equinoctial", "Seasonal", "Ripening", "Budding"],
                     ["Season", "Harvest", "Thaw", "Blossom", "Frost", "Fallow", "Pasture", "Orchard"]),
    "Turning Years": (["Perennial", "Annual", "Cyclic", "Recurring", "Returning", "Yearlong", "Unceasing", "Evergreen"],
                      ["Year", "Almanack", "Calendar", "Solar Wheel", "Rotation", "Round", "Cycle", "Revolution"]),
    "Avalanches": (["Sliding", "Tumbling", "Cascading", "Slumping", "Thundering", "Rushing", "Shearing", "Toppling"],
                   ["Avalanche", "Scree", "Rockfall", "Snowslide", "Landslip", "Talus", "Moraine", "Couloir"]),
    "Glacier Calving": (["Calving", "Groaning", "Creaking", "Splitting", "Cracking", "Drifting", "Booming", "Fracturing"],
                        ["Glacier", "Icefall", "Serac", "Crevasse", "Berg", "Floe", "Icefield", "Ice Shelf"]),
    "Subterranea": (["Subterranean", "Underground", "Cavernous", "Buried", "Nether", "Bedrock", "Hidden", "Chthonic"],
                    ["Floor", "Hypogeum", "Grotto", "Catacomb", "Mineshaft", "Aquifer", "Root Cellar", "Keep"]),
    "Period Doubling": (["Halved", "Doubled", "Bifurcating", "Forking", "Branching", "Cascaded", "Split", "Parametric"],
                        ["Period", "Octave Below", "Fork", "Bifurcation", "Ladder", "Feigenbaum", "Subharmonic", "Faraday"]),
    "Halo Horizons": (["Haloed", "Converging", "Gathering", "Crowded", "Approaching", "Asymptotic", "Brimming", "Compressed"],
                      ["Skyline", "Vanishing Point", "Limit", "Asymptote", "Boundary", "Aureole", "Nimbus", "Firmament"]),
    "Aurora Curtains": (["Wavering", "Draped", "Hanging", "Rippling", "Northern", "Folding", "Glimmering", "Unfurled"],
                        ["Curtain", "Veil", "Drapery", "Arc of Light", "Corona Band", "Streamer", "Ribbon", "Banner"]),
    "Rainfall Grains": (["Pattering", "Drizzling", "Spattering", "Dripping", "Sprinkled", "Misting", "Pelting", "Dappled"],
                        ["Rainfall", "Mizzle", "Sleet", "Hail", "Sprinkle", "Droplet", "Shower", "Spatter"]),
    "Breathing Grains": (["Granular", "Grained", "Sifted", "Powdered", "Milled", "Sanded", "Seeded", "Dusted"],
                         ["Grain", "Pollen", "Seed", "Sand", "Silt", "Flour", "Dust Cloud", "Spore"]),
    "Breather Chains": (["Pulsing", "Throbbing", "Beating", "Heaving", "Lilting", "Swaying", "Rocking", "Nodding"],
                        ["Breather", "Soliton", "Kink", "Pendulum Chain", "Hump", "Pulse", "Lantern Chain", "Bead"]),
    "Harmonic Ecology": (["Grazing", "Foraging", "Competing", "Thriving", "Teeming", "Symbiotic", "Feral", "Ecological"],
                         ["Ecology", "Meadowland", "Biome", "Habitat", "Food Web", "Niche", "Ecosystem", "Commons"]),
    "Night Ecology": (["Nocturne", "Moonlit", "Dusky", "Owlish", "Dew-Laden", "Shadowed", "Whispering", "Sleeping"],
                      ["Marsh", "Heath", "Fen", "Bog", "Moor", "Brake", "Hollow", "Spinney"]),
    "Wild Harmonics": (["Wild", "Untamed", "Unruly", "Restless", "Storm-Torn", "Blustering", "Ragged", "Feverish"],
                       ["Harmonics", "Overtones", "Partials", "Spectra", "Cyclone", "Squall", "Maelstrom", "Whirlwind"]),
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


# ---- The sleep concerts: sixteen groups made for the night.

def num(r, key, default):
    """A parameter as the preset has it so far (its default when it names none)."""
    return float(r.p[key]) if key in r.p else default


def sleepy(r, attack=(6, 25), release=(15, 45)):
    """What every sleep concert keeps: slow in and out, darker, a slow conductor low down, nothing sudden."""
    if num(r, "attack", 2.0) < attack[0]:
        r.set("attack", f(r.lg(*attack), 3))
    if num(r, "release", 4.0) < release[0]:
        r.set("release", f(r.lg(*release), 3))
    r.set("sustain", f(r.u(0.85, 1.0), 3))
    if num(r, "brightness", 1.0) > 0.6:
        r.set("brightness", f(r.u(0.3, 0.6), 2))
    if num(r, "tilt", 1.2) < 1.4:
        r.set("tilt", f(r.u(1.4, 2.0), 2))
    if num(r, "shimmer", 0.4) > 0.5:
        r.set("shimmer", f(r.u(0.2, 0.5), 2))
    if num(r, "shimmer_rate", 0.15) > 0.3:
        r.set("shimmer_rate", f(r.lg(0.03, 0.2), 3))
    if num(r, "presence", 0.0) > 2.0:
        r.set("presence", f(r.u(0.0, 2.0), 1))
    if num(r, "motion_rate", 1.0) > 0.6:
        r.set("motion_rate", f(r.lg(0.1, 0.5), 3))
    # The conductor: on, in chords, slow, few voices, low.
    r.set("brain_on", 1)
    r.set("auto_mode", 1)
    r.set("auto_rate", f(r.lg(90, 300), 1))
    r.set("brain_rate", f(r.lg(30, 90), 1))
    r.set("brain_density", r.rng.randint(3, 5))
    r.set("brain_low", r.rng.randint(31, 40))
    r.set("brain_high", r.rng.randint(62, 74))
    # Nothing sudden: soft echoes, slow low ripples, slow sync.
    if "delay_feedback" in r.p:
        r.set("delay_feedback", f(min(num(r, "delay_feedback", 0.4), 0.5), 2))
    if "delay_mix" in r.p:
        r.set("delay_mix", f(min(num(r, "delay_mix", 0.0), 0.15), 3))
    if "spec_mix" in r.p:
        r.set("spec_mix", f(min(num(r, "spec_mix", 0.0), 0.2), 3))
    if "ripple_depth" in r.p:
        r.set("ripple_depth", f(min(num(r, "ripple_depth", 0.0), 0.4), 2))
        r.set("ripple_speed", f(r.lg(0.1, 0.4), 3))
        r.set("ripple_decay", f(r.lg(12, 30), 2))
    if num(r, "sync_rate", 0.5) > 1.0:
        r.set("sync_rate", f(r.lg(0.05, 0.3), 3))


def setup(r, kind):
    """A kind of sleep modulation's source, set once per preset."""
    done = r.p.setdefault("_sleep", set())
    if kind in done:
        return
    done.add(kind)
    if kind == "breath":    # LFO 4: a breath, five to seven a minute
        r.set("lfo4_shape", LFO["Sine"])
        r.set("lfo4_rate", f(r.u(0.08, 0.12), 4))
    elif kind == "tide":    # LFO 3: a tide, one every two to eight minutes
        r.set("lfo3_shape", r.pick(LFO["Sine"], LFO["Triangle"]))
        r.set("lfo3_rate", f(r.lg(0.002, 0.008), 4))
    elif kind == "wander":  # LFO 2: a smooth random wander
        r.set("lfo2_shape", LFO["Random"])
        r.set("lfo2_rate", f(r.lg(0.01, 0.04), 4))
    elif kind == "chaos":   # the attractors over ten to thirty minutes
        r.set("chaos_period", f(r.lg(600, 1800), 0))
    elif kind == "kura":    # the oscillators well in step
        r.set("coherence", f(r.u(0.5, 0.9), 2))
        r.set("coherence_rate", f(r.u(0.3, 0.7), 2))
    elif kind == "lenia":   # the field slow
        r.set("lenia_rate", f(r.lg(0.5, 2.0), 2))
    elif kind == "reservoir":
        set_reservoir(r, (0.9, 1.05), (10, 30), (0.3, 0.8))


# Sleep modulations, by kind: (source, target, depth range). Slow and smooth only.
SLEEP_POOLS = {
    "breath": [("lfo4", "brightness", (0.03, 0.07)), ("lfo4", "cloud_width", (0.04, 0.1)), ("lfo4", "air", (0.05, 0.12)),
               ("lfo4", "presence", (0.05, 0.1)), ("lfo4", "depth", (0.03, 0.06))],
    "tide": [("lfo3", "depth", (0.08, 0.18)), ("lfo3", "far_level", (0.06, 0.14)), ("lfo3", "tilt", (0.03, 0.07)),
             ("lfo3", "cloud_width", (0.06, 0.14))],
    "wander": [("lfo2", "drift", (0.1, 0.25)), ("lfo2", "shimmer", (0.08, 0.2)), ("lfo2", "motion_depth", (0.08, 0.2)),
               ("lfo2", "air_center", (0.1, 0.2))],
    "chaos": [("lorenz_x", "cloud_width", (0.06, 0.15)), ("lorenz_z", "tilt", (0.03, 0.07)), ("rossler_x", "drift", (0.1, 0.2)),
              ("rossler_y", "shimmer", (0.08, 0.2)), ("lorenz_y", "depth", (0.05, 0.12))],
    "kura": [("kura1", "cloud_width", (0.05, 0.12)), ("kura2", "shimmer", (0.08, 0.2)), ("kura3", "air", (0.08, 0.2)),
             ("kura4", "brightness", (0.04, 0.1))],
    "lenia": [("lenia1", "shimmer", (0.1, 0.25)), ("lenia2", "air", (0.1, 0.25)), ("lenia3", "presence", (0.08, 0.15)),
              ("lenia4", "cloud_width", (0.08, 0.2))],
    "night": [("root_age", "brightness", (-0.2, -0.08)), ("root_age", "depth", (0.1, 0.25)), ("root_age", "far_level", (0.08, 0.2)),
              ("root_age", "memory_level", (0.1, 0.25)), ("root_age", "melt_mix", (0.1, 0.25))],
    "calm": [("loudness", "brightness", (-0.2, -0.08)), ("flux", "shimmer", (-0.2, -0.08)), ("roughness", "tune_gravity", (0.3, 0.6)),
             ("loudness", "depth", (0.08, 0.2))],
    "reservoir": [("reservoir1", "brightness", (0.05, 0.12)), ("reservoir2", "cloud_width", (0.08, 0.18)),
                  ("reservoir3", "air_center", (0.08, 0.2)), ("reservoir4", "tilt", (0.03, 0.07))],
}
SLEEP_VIAS = ["root_age", "lenia2", "kura3"]


def sleep_rows(r, kinds, count, via_p=0.25):
    """Up to `count` sleep modulations from the kinds: each a source and a target not taken yet, some through a via."""
    taken = {tuple(row.split(":")[0].split(">")) for row in r.rows}
    for _ in range(count):
        kind = r.pick(*kinds)
        src, target, (lo, hi) = r.pick(*SLEEP_POOLS[kind])
        if (src, target) in taken:
            continue
        taken.add((src, target))
        setup(r, kind)
        via = r.pick(*SLEEP_VIAS) if r.rng.random() < via_p else None
        r.row(src, target, round(r.u(lo, hi), 3), via)


CALM = ("Sleep Concert", "Choirs", "Strings", "Deep Space", "Dark Drones", "Shepard & Tides", "Organs", "Air & Noise")


def s_lullaby(r):
    based(r, "Sleep Concert", "Strings", "Choirs", "Organs", "Bells")
    sleepy(r)
    r.set("scale", r.pick(SCALE["JI Major"], SCALE["JI Pentatonic"], SCALE["Otonality"]))
    r.set("purity", 1)
    r.set("motion", MOTION["Breathe"])
    r.set("motion_depth", f(r.u(0.3, 0.6), 2))
    sleep_rows(r, ["breath", "kura", "night"], 4)


def s_breath(r):
    based(r, "Sleep Concert", "Choirs", "Strings", "Air & Noise", "Organs")
    sleepy(r)
    rate = r.u(0.08, 0.12)   # one breath, five to seven a minute: the room, the harmonics' pulse and LFO 4 share it
    r.p.setdefault("_sleep", set()).add("breath")
    r.set("lfo4_shape", LFO["Sine"])
    r.set("lfo4_rate", f(rate, 4))
    r.set("breath", f(r.u(0.3, 0.6), 2))
    r.set("breath_rate", f(rate, 4))
    set_sync(r, (0.3, 0.5), (0.7, 1.0), (rate, rate), (0.05, 0.15))
    r.row("lfo4", "brightness", round(r.u(0.04, 0.08), 3))
    r.row("lfo4", "cloud_width", round(r.u(0.05, 0.12), 3))
    sleep_rows(r, ["night", "calm"], 2)


def s_deep(r):
    based(r, "Dark Drones", "Deep Space", "Sleep Concert")
    sleepy(r, attack=(12, 30), release=(25, 60))
    r.set("filter_model", FILTER["LP 24"])
    r.set("cutoff", f(r.lg(150, 500), 0))
    r.set("tilt", f(r.u(2.0, 2.6), 2))
    r.set("brightness", f(r.u(0.2, 0.4), 2))
    r.set("auto_rate", f(r.lg(180, 420), 1))
    r.set("brain_density", r.rng.randint(2, 4))
    r.set("brain_high", r.rng.randint(55, 64))
    sleep_rows(r, ["tide", "night", "chaos"], 4)


def s_dream(r):
    based(r, *CALM)
    sleepy(r)
    if r.rng.random() < 0.5:
        r.set("field_mode", 2)   # Swarm, wandering slowly
        r.set("field_strength", f(r.u(0.3, 0.6), 2))
        r.set("field_radius", f(r.u(100, 300), 0))
        r.set("field_spring", f(r.u(0.2, 0.5), 2))
        r.set("field_rate", f(r.lg(0.005, 0.02), 3))
    if r.rng.random() < 0.4:
        set_melt(r, (0.15, 0.35), (10, 30), (0.2, 0.6))
    sleep_rows(r, ["chaos"], 3)
    sleep_rows(r, ["wander"], 1)


def s_moon(r):
    based(r, "Shimmer", "Glass", "Choirs", "Sleep Concert")
    sleepy(r)
    r.set("shimmer", f(r.u(0.3, 0.5), 2))
    r.set("shimmer_rate", f(r.lg(0.03, 0.12), 3))
    if r.rng.random() < 0.5:
        set_sync(r, (0.15, 0.3), (0.5, 0.8), (0.05, 0.2), (0.2, 0.5))
    sleep_rows(r, ["kura", "breath"], 4)


def s_rain(r):
    based(r, "Air & Noise", "Sleep Concert", "Choirs")
    sleepy(r)
    r.set("air", f(r.u(0.5, 0.9), 2))
    r.set("air_jitter", f(r.lg(6, 20), 2))
    r.set("air_center", f(r.u(2, 5), 2))
    r.set("air_width", f(r.u(1200, 2400), 0))
    if r.rng.random() < 0.5:
        set_life(r, 2, (0.3, 0.5), (0.1, 0.3), (0.0, 1.0), (0.4, 1.0))
    if r.rng.random() < 0.5:
        set_melt(r, (0.2, 0.4), (8, 25), (0.6, 1.4))
    sleep_rows(r, ["lenia", "wander", "night"], 4)


def s_ocean(r):
    based(r, "Waves", "Shepard & Tides", "Sleep Concert")
    sleepy(r)
    r.set("motion", MOTION["Wave"])
    r.set("motion_rate", f(r.lg(0.1, 0.3), 3))
    r.set("motion_depth", f(r.u(0.4, 0.8), 2))
    r.set("tide", f(r.u(5, 15), 1))
    r.set("tide_period", f(r.u(2, 8), 1))
    if r.rng.random() < 0.5:
        set_ripple(r, (0.2, 0.4), (0.1, 0.3), (15, 30))
    sleep_rows(r, ["breath", "tide"], 4)


def s_cradle(r):
    based(r, "Pendulums", "Sleep Concert", "Strings")
    sleepy(r)
    r.set("motion", MOTION["Pendulum"])
    r.set("motion_rate", f(r.lg(0.1, 0.25), 3))
    r.set("motion_depth", f(r.u(0.4, 0.7), 2))
    set_sync(r, (0.2, 0.4), (0.6, 0.9), (0.1, 0.2), (0.1, 0.3))
    sleep_rows(r, ["kura", "breath", "night"], 4)


def s_star(r):
    based(r, "Deep Space", "Glass", "Bells", "Sleep Concert")
    sleepy(r)
    r.set("brain_density", r.rng.randint(2, 3))
    set_melt(r, (0.35, 0.7), (20, 60), (0.2, 0.8))
    sleep_rows(r, ["chaos", "tide", "night"], 4)


def s_hypna(r):
    based(r, *CALM)
    sleepy(r)
    if r.rng.random() < 0.6:
        set_life(r, 1, (0.3, 0.6), (0.05, 0.2), (0.1, 0.5), (0.5, 1.0))
    if r.rng.random() < 0.5:
        set_melt(r, (0.2, 0.4), (10, 30), (0.3, 0.9))
    sleep_rows(r, ["reservoir"], 3, via_p=0.4)
    sleep_rows(r, ["lenia"], 1)
    if not any(row.startswith("reservoir") for row in r.rows):
        setup(r, "reservoir")
        r.row("reservoir1", "brightness", round(r.u(0.05, 0.12), 3))


def s_choirs(r):
    based(r, "Choirs")
    sleepy(r)
    r.set("lfo2_rate", f(r.lg(0.004, 0.02), 4))   # the vowel turns slowly
    if r.rng.random() < 0.5:
        r.set("memory_level", f(r.u(0.3, 0.55), 2))
        r.set("memory_time", f(r.lg(90, 240), 0))
    sleep_rows(r, ["kura", "breath", "night"], 3)


def s_warm(r):
    based(r, "Organs", "Strings", "Dark Drones", "Sleep Concert")
    sleepy(r)
    r.set("brain_density", r.rng.randint(2, 3))
    r.set("brain2_on", 1)
    r.set("brain2_interval", -12)
    r.set("brain2_depth", f(r.u(0.6, 0.9), 2))
    set_sync(r, (0.2, 0.4), (0.6, 1.0), (0.05, 0.15), (0.1, 0.3))
    sleep_rows(r, ["calm", "tide"], 4)


def s_snow(r):
    based(r, "Glass", "Bells", "Shimmer")
    sleepy(r, attack=(2, 8))
    set_melt(r, (0.3, 0.6), (10, 30), (0.3, 1.0))
    if r.rng.random() < 0.5:
        set_ripple(r, (0.15, 0.3), (0.1, 0.3), (10, 25))
    sleep_rows(r, ["lenia", "kura", "calm"], 4)


def s_forest(r):
    based(r, "Air & Noise", "Choirs", "Strings", "Sleep Concert")
    sleepy(r)
    set_life(r, r.pick(1, 2), (0.4, 0.7), (0.05, 0.3), (0.1, 0.8), (0.4, 1.0))
    r.set("air", f(r.u(0.2, 0.5), 2))
    sleep_rows(r, ["lenia", "wander", "chaos"], 4)


def s_tides(r):
    based(r, "Shepard & Tides", "Waves", "Glissandi")
    sleepy(r)
    r.set("motion", MOTION["Aurora"])
    r.set("landscape", LAND["Tides"])
    r.set("motion_rate", f(r.lg(0.1, 0.4), 3))
    r.set("tide", f(r.u(8, 20), 1))
    r.set("tide_period", f(r.u(3, 12), 1))
    sleep_rows(r, ["tide", "night"], 4)


def s_afterglow(r):
    based(r, "Sleep Concert", "Strings", "Choirs", "Keys", "Organs")
    sleepy(r)
    r.set("memory_level", f(r.u(0.4, 0.7), 2))
    r.set("memory_time", f(r.lg(120, 300), 0))
    r.set("memory_bloom", f(r.lg(5, 15), 1))
    r.set("memory_blur", f(r.u(0.5, 1.0), 2))
    r.set("memory_sink", f(r.u(0.5, 0.9), 2))
    if r.rng.random() < 0.5:
        set_melt(r, (0.2, 0.4), (15, 40), (0.2, 0.6))
    sleep_rows(r, ["night", "calm", "kura"], 4)


RECIPES = RECIPES + [s_lullaby, s_breath, s_deep, s_dream, s_moon, s_rain, s_ocean, s_cradle,
                     s_star, s_hypna, s_choirs, s_warm, s_snow, s_forest, s_tides, s_afterglow]


# ---- Phase 13 (13.10.2026): the animated pictures, the spectral comb, Stretch and Shift.

PIC = {"Kaleidoscope": 3, "Tunnel": 4, "Spiral": 5, "Flow": 6, "Feedback": 7, "Rain": 8, "Starfield": 9, "Moire": 10}


def set_picture(r, mode, depth=(0.5, 0.85), rate=(0.1, 0.5), turn=(0.2, 1.5), sides=(3, 12), character=(0.0, 0.6)):
    """An animated picture in the living landscape (over the preset's landscape)."""
    r.set("life_mode", PIC[mode])
    r.set("life_depth", f(r.u(*depth), 2))
    r.set("life_rate", f(r.u(*rate), 2))
    r.set("life_turn", f(r.u(*turn) * r.pick(1, -1), 2))
    r.set("life_sides", r.rng.randint(*sides))
    r.set("life_character", f(r.u(*character), 2))
    r.set("life_zoom", f(r.u(0.6, 1.0), 2))


def set_comb(r, depth=(0.3, 0.7), density=(1, 4), sweep=(0.02, 0.2), stereo=(0.2, 0.8), direction=None):
    """The spectral comb (a sweep up or down, or the given direction)."""
    r.set("comb_depth", f(r.u(*depth), 2))
    r.set("comb_density", f(r.lg(*density), 2))
    r.set("comb_sweep", f(r.lg(*sweep) * (direction if direction else r.pick(1, -1)), 3))
    r.set("comb_stereo", f(r.u(*stereo), 2))


# What the first forty-eight groups take, drawn last: comb = (p, depth, density, sweep, stereo);
# stretch = (p, cents an octave); shift = (p, Hz, either way); picture = (p, modes, depth, rate, turn).
SLOW_PIC = ((0.3, 0.6), (0.03, 0.15), (0.05, 0.3))   # the sleep concerts' pictures: depth, rate, turn
SLOW_COMB = ((0.2, 0.5), (0.5, 2), (0.01, 0.04), (0.2, 0.6))
P13_FX = {
    "Sleep Concert":   dict(picture=(0.15, ["Kaleidoscope", "Spiral", "Flow"], (0.3, 0.5), (0.05, 0.2), (0.1, 0.3))),
    "Choirs":          dict(comb=(0.2, (0.3, 0.6), (1, 3), (0.02, 0.1), (0.3, 0.7))),
    "Glass":           dict(comb=(0.3, (0.4, 0.8), (2, 6), (0.05, 0.3), (0.3, 0.8)), stretch=(0.3, (1180, 1240)), shift=(0.15, (3, 25))),
    "Bells":           dict(stretch=(0.4, (1170, 1250)), shift=(0.2, (2, 20))),
    "Strings":         dict(comb=(0.15, (0.3, 0.5), (1, 2), (0.02, 0.08), (0.2, 0.6))),
    "Organs":          dict(comb=(0.2, (0.3, 0.6), (2, 4), (0.03, 0.1), (0.3, 0.8))),
    "Air & Noise":     dict(comb=(0.3, (0.5, 0.9), (1, 4), (0.05, 0.3), (0.3, 0.8)), picture=(0.2, ["Flow", "Rain"], (0.4, 0.7), (0.1, 0.4), (0.2, 1.0))),
    "Deep Space":      dict(picture=(0.25, ["Starfield", "Spiral", "Tunnel"], (0.3, 0.6), (0.05, 0.25), (0.1, 0.5)), shift=(0.1, (1, 8))),
    "Glissandi":       dict(comb=(0.2, (0.3, 0.6), (1, 3), (0.05, 0.2), (0.3, 0.7)), picture=(0.15, ["Spiral", "Tunnel"], (0.4, 0.7), (0.1, 0.4), (0.2, 1.0))),
    "Pendulums":       dict(picture=(0.15, ["Moire"], (0.4, 0.7), (0.1, 0.3), (0.2, 0.8))),
    "Waves":           dict(picture=(0.2, ["Rain", "Flow"], (0.4, 0.7), (0.1, 0.3), (0.1, 0.5)), comb=(0.15, (0.3, 0.6), (1, 2), (0.02, 0.1), (0.2, 0.6))),
    "Shepard & Tides": dict(comb=(0.35, (0.4, 0.8), (1, 2), (0.03, 0.12), (0.2, 0.6)), picture=(0.2, ["Spiral"], (0.4, 0.7), (0.05, 0.3), (0.1, 0.5))),
    "Shimmer":         dict(comb=(0.3, (0.4, 0.8), (2, 6), (0.05, 0.3), (0.4, 0.9)), picture=(0.2, ["Kaleidoscope", "Moire"], (0.4, 0.7), (0.2, 0.5), (0.3, 1.5))),
    "Dark Drones":     dict(shift=(0.15, (2, 15)), picture=(0.15, ["Feedback", "Flow"], (0.3, 0.6), (0.05, 0.2), (0.1, 0.4))),
    "Microtonal":      dict(stretch=(0.3, (1150, 1260))),
    "Keys":            dict(comb=(0.15, (0.3, 0.5), (1, 3), (0.03, 0.1), (0.2, 0.6)), stretch=(0.25, (1195, 1215))),
    "Living Fields":   dict(comb=(0.15, (0.3, 0.6), (1, 3), (0.02, 0.1), (0.2, 0.6))),
    "Turing Gardens":  dict(comb=(0.15, (0.3, 0.6), (1, 3), (0.02, 0.1), (0.2, 0.6))),
    "Synchrony":       dict(comb=(0.15, (0.3, 0.6), (1, 3), (0.02, 0.1), (0.2, 0.6))),
    "Flocks":          dict(picture=(0.2, ["Flow"], (0.4, 0.7), (0.1, 0.4), (0.2, 1.0))),
    "Ripples":         dict(picture=(0.25, ["Rain"], (0.4, 0.7), (0.1, 0.4), (0.1, 0.5))),
    "Melting Spectra": dict(comb=(0.2, (0.3, 0.6), (1, 3), (0.02, 0.1), (0.2, 0.6)), picture=(0.15, ["Feedback"], (0.3, 0.6), (0.05, 0.2), (0.1, 0.5))),
    "Spectral Cascades": dict(comb=(0.3, (0.4, 0.8), (2, 6), (0.05, 0.3), (0.3, 0.8)), stretch=(0.2, (1180, 1240))),
    "Resonant Halls":  dict(stretch=(0.25, (1170, 1250)), shift=(0.1, (2, 12))),
    "Ghosts & Memory": dict(picture=(0.15, ["Feedback", "Flow"], (0.3, 0.6), (0.05, 0.2), (0.1, 0.4))),
    "Strange Attractors": dict(shift=(0.2, (2, 15)), picture=(0.2, ["Feedback"], (0.4, 0.7), (0.1, 0.3), (0.2, 1.0))),
    "Reservoir Minds": dict(comb=(0.2, (0.3, 0.6), (1, 4), (0.03, 0.15), (0.2, 0.7))),
    "Self-Listening":  dict(comb=(0.2, (0.3, 0.6), (1, 4), (0.03, 0.15), (0.2, 0.7))),
    "Gravity Wells":   dict(stretch=(0.3, (1170, 1250))),
    "Smear & Dispersion": dict(comb=(0.3, (0.4, 0.8), (1, 4), (0.05, 0.25), (0.3, 0.8)), shift=(0.15, (2, 15))),
    "Lattices & Beats": dict(stretch=(0.3, (1180, 1230)), comb=(0.15, (0.3, 0.6), (1, 3), (0.02, 0.1), (0.2, 0.6))),
    "Night Gardens":   dict(picture=(0.3, ["Kaleidoscope", "Flow", "Starfield"], (0.3, 0.6), (0.05, 0.2), (0.1, 0.4))),
    "Lullabies":       dict(picture=(0.2, ["Kaleidoscope", "Moire"]) + SLOW_PIC),
    "Slow Breath":     dict(comb=(0.2,) + SLOW_COMB),
    "Deep Sleep":      dict(picture=(0.2, ["Feedback", "Flow"]) + SLOW_PIC),
    "Dream Drift":     dict(picture=(0.35, ["Flow", "Spiral", "Feedback"]) + SLOW_PIC),
    "Moonlight":       dict(picture=(0.35, ["Kaleidoscope", "Moire"]) + SLOW_PIC, comb=(0.2,) + SLOW_COMB),
    "Night Rain":      dict(picture=(0.5, ["Rain"]) + SLOW_PIC),
    "Ocean Sleep":     dict(picture=(0.35, ["Rain", "Flow"]) + SLOW_PIC),
    "Cradle":          dict(picture=(0.2, ["Moire"]) + SLOW_PIC),
    "Starfield":       dict(picture=(0.6, ["Starfield", "Spiral"]) + SLOW_PIC),
    "Hypnagogia":      dict(picture=(0.4, ["Feedback", "Flow", "Kaleidoscope"]) + SLOW_PIC),
    "Twilight Choirs": dict(comb=(0.2,) + SLOW_COMB),
    "Warm Drones":     dict(comb=(0.2,) + SLOW_COMB, stretch=(0.15, (1190, 1215))),
    "Snowfall":        dict(picture=(0.4, ["Starfield", "Rain"]) + SLOW_PIC, comb=(0.15,) + SLOW_COMB),
    "Forest Night":    dict(comb=(0.15,) + SLOW_COMB),
    "Slow Tides":      dict(picture=(0.3, ["Spiral", "Flow"]) + SLOW_PIC, comb=(0.2,) + SLOW_COMB),
    "Afterglow":       dict(picture=(0.2, ["Feedback"]) + SLOW_PIC),
}


def p13_effects(r, group):
    """Phase 13 in one of the first forty-eight groups, drawn after everything else."""
    fx = P13_FX.get(group, {})
    if "comb" in fx:
        p, depth, density, sweep, stereo = fx["comb"]
        if r.rng.random() < p:
            set_comb(r, depth, density, sweep, stereo)
    if "stretch" in fx:
        p, cents = fx["stretch"]
        if r.rng.random() < p:
            r.set("stretch", f(r.u(*cents), 1))
    if "shift" in fx:
        p, hz = fx["shift"]
        if r.rng.random() < p:
            r.set("shift", f(r.u(*hz) * r.pick(1, -1), 2))
    if "picture" in fx:
        p, modes, depth, rate, turn = fx["picture"]
        if r.rng.random() < p and num(r, "life_mode", 0) == 0:   # a living landscape that grows stays
            set_picture(r, r.pick(*modes), depth, rate, turn, character=(0.0, 0.5))


# ---- The visions: sixteen groups, each from one of the first sixteen's sounds, one new means in front.

VISION_POOLS = {
    "turn": [("lfo2", "life_turn", (0.04, 0.12))],
    "depth": [("kura2", "life_depth", (0.08, 0.18)), ("rossler_x", "life_depth", (0.08, 0.18))],
    "comb": [("kura1", "comb_density", (0.05, 0.12)), ("lorenz_x", "comb_depth", (0.1, 0.2)), ("lfo3", "comb_depth", (0.1, 0.2))],
    "shift": [("lfo3", "shift", (0.008, 0.025))],
}


def vision_rows(r, kinds, count, via_p=0.25):
    """Rows onto the new means' knobs, their sources set slowly (the sleep concerts' setups)."""
    taken = {tuple(row.split(":")[0].split(">")) for row in r.rows}
    for _ in range(count):
        if len(r.rows) >= 16:
            return
        src, target, (lo, hi) = r.pick(*VISION_POOLS[r.pick(*kinds)])
        if (src, target) in taken:
            continue
        taken.add((src, target))
        if src == "lfo2":
            setup(r, "wander")
        elif src == "lfo3":
            setup(r, "tide")
        elif src.startswith("kura"):
            setup(r, "kura")
        elif src.startswith(("lorenz", "rossler")):
            setup(r, "chaos")
        via = r.pick(*SLEEP_VIAS) if r.rng.random() < via_p else None
        r.row(src, target, round(r.u(lo, hi), 3), via)


def landscape_any(r):
    """A factory landscape for a picture to be made of."""
    r.set("landscape", r.pick(*LAND.values()))


def v_pictured(mode, bases, comb_p=0.3, **ranges):
    """A recipe: the sound of one of `bases`, the picture `mode` in front, now and then a slow comb."""
    def recipe(r):
        based(r, *bases)
        landscape_any(r)
        set_picture(r, mode, **ranges)
        if r.rng.random() < comb_p:
            set_comb(r, (0.2, 0.5), (1, 3), (0.02, 0.1), (0.2, 0.6))
        vision_rows(r, ["turn", "depth"], 2)
        rows_from(r, ["kura", "lenia", "chaos"], 1)
    return recipe


v_kaleido = v_pictured("Kaleidoscope", ("Shimmer", "Glass", "Choirs", "Sleep Concert", "Strings", "Air & Noise"))
v_tunnel = v_pictured("Tunnel", ("Deep Space", "Glissandi", "Shepard & Tides", "Dark Drones", "Air & Noise"))
v_spiral = v_pictured("Spiral", ("Shepard & Tides", "Glissandi", "Waves", "Deep Space", "Shimmer"), comb_p=0.4)
v_flow = v_pictured("Flow", ("Air & Noise", "Choirs", "Strings", "Sleep Concert", "Waves"), character=(0.2, 0.8))
v_feedback = v_pictured("Feedback", ("Dark Drones", "Deep Space", "Organs", "Sleep Concert", "Strings"))
v_rain = v_pictured("Rain", ("Glass", "Bells", "Keys", "Air & Noise", "Waves"), comb_p=0.15, rate=(0.2, 0.7), character=(0.2, 0.9))
v_stars = v_pictured("Starfield", ("Deep Space", "Glass", "Shimmer", "Bells", "Sleep Concert"), comb_p=0.15)
v_moire = v_pictured("Moire", ("Organs", "Strings", "Choirs", "Pendulums", "Shimmer"), sides=(3, 8))


def v_stretched(r):
    based(r, "Bells", "Glass", "Microtonal", "Organs", "Strings", "Choirs")
    r.set("stretch", f(r.u(1100, 1300), 1))
    r.set("scale", SCALE["Timbre"])   # the tuning made from the stretched spectrum: consonant again
    r.set("purity", 1)
    if r.rng.random() < 0.4:
        set_sync(r, (0.2, 0.4), (0.5, 0.9), (0.1, 0.8), (0.1, 0.4))
    rows_from(r, ["kura", "chaos"], 2)


def v_metals(r):
    based(r, "Bells", "Glass", "Dark Drones", "Organs", "Keys")
    r.set("shift", f(r.u(5, 80) * r.pick(1, 1, -1), 1))
    if r.rng.random() < 0.4:
        r.set("stretch", f(r.u(1170, 1240), 1))
    vision_rows(r, ["shift"], 1)
    rows_from(r, ["kura", "chaos"], 1)


def v_comb(r):
    based(r, "Organs", "Strings", "Choirs", "Air & Noise", "Shimmer", "Keys")
    set_comb(r, (0.5, 1.0), (1, 6), (0.05, 0.5), (0.3, 1.0))
    vision_rows(r, ["comb"], 2)
    rows_from(r, ["kura", "heard"], 1)


def v_barber(r):
    based(r, "Shepard & Tides", "Glissandi", "Organs", "Choirs", "Strings")
    up = r.pick(1, -1)   # the comb and the spiral travel the same way: a rise (or fall) without end
    set_comb(r, (0.5, 0.9), (0.5, 2), (0.04, 0.2), (0.1, 0.5), direction=up)
    if r.rng.random() < 0.6:
        landscape_any(r)
        set_picture(r, "Spiral", depth=(0.4, 0.7), rate=(0.1, 0.4), turn=(0.1, 0.6))
    vision_rows(r, ["depth"], 1)
    rows_from(r, ["kura"], 1)


def v_painted(r):
    based(r, "Sleep Concert", "Choirs", "Air & Noise", "Shimmer", "Waves")
    r.set("motion", MOTION["Aurora"])
    r.set("motion_depth", f(r.u(0.4, 0.8), 2))
    landscape_any(r)
    set_picture(r, r.pick("Kaleidoscope", "Flow"), depth=(0.3, 0.6), rate=(0.05, 0.3), turn=(0.1, 0.6))
    if r.rng.random() < 0.4:
        set_comb(r, (0.2, 0.5), (1, 3), (0.02, 0.1), (0.2, 0.6))
    vision_rows(r, ["turn"], 1)
    rows_from(r, ["lenia", "kura"], 1)


def v_prism(r):
    based(r, "Glass", "Shimmer", "Bells", "Keys")
    set_comb(r, (0.4, 0.8), (2, 6), (0.03, 0.2), (0.4, 1.0))
    r.set("stretch", f(r.u(1185, 1225), 1))
    landscape_any(r)
    set_picture(r, "Kaleidoscope", depth=(0.4, 0.7), rate=(0.1, 0.4), turn=(0.3, 1.5), sides=(6, 12))
    vision_rows(r, ["comb", "turn"], 2)


def v_alien(r):
    based(r, "Choirs", "Strings", "Organs", "Sleep Concert", "Deep Space")
    r.set("stretch", f(r.u(1120, 1280), 1))
    r.set("scale", SCALE["Timbre"])
    r.set("purity", 1)
    set_sync(r, (0.2, 0.45), (0.5, 0.9), (0.1, 0.6), (0.1, 0.4))
    r.set("memory_level", f(r.u(0.2, 0.5), 2))
    r.set("memory_time", f(r.lg(60, 200), 0))
    rows_from(r, ["kura", "reservoir"], 2)
    if any(row.startswith("reservoir") for row in r.rows):
        set_reservoir(r, (0.9, 1.1), (3, 15), (0.4, 1.0))


def v_machine(r):
    based(r, *ALL)
    landscape_any(r)
    set_picture(r, r.pick(*PIC), depth=(0.3, 0.7), rate=(0.1, 0.5), turn=(0.2, 1.2))
    if r.rng.random() < 0.6:
        set_comb(r, (0.3, 0.7), (1, 4), (0.03, 0.2), (0.2, 0.8))
    if r.rng.random() < 0.4:
        r.set("stretch", f(r.u(1170, 1240), 1))
    if r.rng.random() < 0.3:
        r.set("shift", f(r.u(2, 20) * r.pick(1, -1), 1))
    vision_rows(r, ["turn", "depth", "comb", "shift"], 2)
    rows_from(r, ["reservoir", "kura"], 1)
    if any(row.startswith("reservoir") for row in r.rows):
        set_reservoir(r, (0.9, 1.1), (3, 15), (0.4, 1.0))


RECIPES = RECIPES + [v_kaleido, v_tunnel, v_spiral, v_flow, v_feedback, v_rain, v_stars, v_moire,
                     v_stretched, v_metals, v_comb, v_barber, v_painted, v_prism, v_alien, v_machine]


# ---- Phase 16 (14.10.2026): swarms and chimeras, the Fluid and the Islands, turbulence, MPE.

LIFE_FLUID, LIFE_ISLANDS = 11, 12
MPE_T = {"Off": 0, "Level": 1, "Filter": 2, "Brightness": 3, "Width": 4, "Nearer": 5, "Shimmer": 6, "Motion": 7, "Breath": 8}


def set_swarm(r, state, depth=(0.3, 0.7), rate=(0.1, 1.2), reach=(10, 50)):
    """The partials as swarmalators. Where their states lie (simulated, 14.10.2026): a phase wave
    circling through pitch and field without coupling and with attraction, or with strong coupling
    and repulsion; two groups pulsing against each other with both; broken waves at the borders
    between; each for itself with repulsion and little coupling."""
    r.set("sync_mode", 2)
    r.set("sync_depth", f(r.u(*depth), 2))
    r.set("sync_rate", f(r.lg(*rate), 3))
    r.set("sync_reach", f(r.u(*reach), 1))
    if state == "wave":
        coupling, swarm = (0.0, r.u(0.4, 1.0)) if r.rng.random() < 0.5 else (r.u(0.85, 1.0), r.u(-0.8, -0.3))
        spread = r.u(0.03, 0.2)
    elif state == "twin":
        coupling, swarm, spread = r.u(0.3, 1.0), r.u(0.3, 1.0), r.u(0.03, 0.3)
    elif state == "splinter":
        if r.rng.random() < 0.5:
            coupling, swarm, spread = r.u(0.5, 0.7), r.u(-0.6, -0.4), r.u(0.03, 0.15)
        else:
            coupling, swarm, spread = 0.0, r.u(0.25, 0.4), r.u(0.2, 0.35)
    else:
        coupling, swarm, spread = r.u(0.0, 0.3), r.u(-1.0, -0.3), r.u(0.1, 0.4)
    r.set("sync_coupling", f(coupling, 2))
    r.set("sync_swarm", f(swarm, 2))
    r.set("sync_spread", f(spread, 2))


def set_chimera(r, depth=(0.3, 0.7), rate=(0.1, 1.0)):
    """A chimera: the partials coupled to their neighbours with a lag, alike in their rates."""
    r.set("sync_mode", 1)
    r.set("sync_depth", f(r.u(*depth), 2))
    r.set("sync_coupling", f(r.u(0.6, 1.0), 2))
    r.set("sync_rate", f(r.lg(*rate), 3))
    r.set("sync_spread", f(r.u(0.0, 0.04), 3))
    r.set("sync_lag", f(r.u(0.3, 0.65), 2))


def set_turb(r, amount=(0.4, 1.0), settle=(2, 20)):
    """The energy cascade."""
    r.set("turbulence", f(r.u(*amount), 2))
    r.set("turbulence_time", f(r.lg(*settle), 2))


def set_mpe(r, presses, press_depth, slides, slide_depth):
    """What a played note's pressure and slide move (MPE)."""
    r.set("mpe_press", MPE_T[r.pick(*presses)])
    r.set("mpe_press_depth", f(r.u(*press_depth), 2))
    r.set("mpe_slide", MPE_T[r.pick(*slides)])
    r.set("mpe_slide_depth", f(r.u(*slide_depth), 2))
    if MPE_T["Filter"] in (r.p.get("mpe_press"), r.p.get("mpe_slide")) and int(num(r, "filter_model", 0)) == 0:
        r.set("filter_model", FILTER["LP 12"])   # a filter for the hand to move
        r.set("cutoff", f(r.lg(600, 3000), 0))
        r.set("resonance", f(r.u(0.1, 0.4), 2))


P16_POOLS = {
    "swarm": [("lfo2", "sync_reach", (0.08, 0.2)), ("kura1", "sync_swarm", (0.1, 0.25)), ("lorenz_x", "sync_depth", (0.08, 0.18))],
    "life": [("lfo3", "life_rate", (0.08, 0.2)), ("kura2", "life_depth", (0.08, 0.18))],
    "turb": [("loudness", "turbulence", (0.1, 0.25)), ("lfo3", "turbulence", (0.08, 0.2)), ("rossler_x", "turbulence", (0.08, 0.2))],
}


def p16_rows(r, kinds, count, via_p=0.25):
    """Rows onto the new means' knobs, their sources set slowly."""
    taken = {tuple(row.split(":")[0].split(">")) for row in r.rows}
    for _ in range(count):
        if len(r.rows) >= 16:
            return
        src, target, (lo, hi) = r.pick(*P16_POOLS[r.pick(*kinds)])
        if (src, target) in taken:
            continue
        taken.add((src, target))
        if src == "lfo2":
            setup(r, "wander")
        elif src == "lfo3":
            setup(r, "tide")
        elif src.startswith("kura"):
            setup(r, "kura")
        elif src.startswith(("lorenz", "rossler")):
            setup(r, "chaos")
        via = r.pick(*SLEEP_VIAS) if r.rng.random() < via_p else None
        r.row(src, target, round(r.u(lo, hi), 3), via)


def q_swarms(r):
    based(r, "Choirs", "Strings", "Organs", "Sleep Concert", "Shimmer", "Glass")
    set_swarm(r, "wave")
    p16_rows(r, ["swarm"], 1)
    rows_from(r, ["kura"], 1)


def q_splinters(r):
    based(r, "Glass", "Shimmer", "Bells", "Strings", "Air & Noise", "Keys")
    set_swarm(r, "splinter", rate=(0.3, 2.5))
    p16_rows(r, ["swarm"], 1)
    rows_from(r, ["chaos"], 1)


def q_chimeras(r):
    based(r, "Choirs", "Organs", "Strings", "Sleep Concert", "Deep Space", "Pendulums")
    set_chimera(r)
    rows_from(r, ["kura", "lenia"], 2)


def q_twins(r):
    based(r, "Pendulums", "Bells", "Glass", "Organs", "Keys", "Waves")
    set_swarm(r, "twin", rate=(0.1, 0.8), reach=(20, 60))
    p16_rows(r, ["swarm"], 1)


def q_currents(r):
    based(r, "Sleep Concert", "Choirs", "Strings", "Deep Space", "Air & Noise", "Waves")
    set_life(r, LIFE_FLUID, (0.4, 0.75), (0.05, 0.4), (0.0, 0.4), (0.6, 1.0))
    p16_rows(r, ["life"], 1)


def q_vortex(r):
    based(r, "Air & Noise", "Glass", "Shimmer", "Waves", "Glissandi", "Dark Drones")
    set_life(r, LIFE_FLUID, (0.5, 0.85), (0.4, 1.0), (0.6, 1.0), (0.6, 1.0))
    rows_from(r, ["heard"], 1)


def q_plumes(r):
    based(r, "Bells", "Keys", "Glass", "Pendulums", "Strings")
    set_life(r, LIFE_FLUID, (0.5, 0.9), (0.2, 0.7), (0.3, 0.8), (0.7, 1.0))
    p16_rows(r, ["life"], 1)


def q_islands(r):
    based(r, "Choirs", "Organs", "Strings", "Shimmer", "Deep Space", "Sleep Concert")
    set_life(r, LIFE_ISLANDS, (0.4, 0.8), (0.2, 0.8), (0.6, 1.0), (0.6, 1.0))
    p16_rows(r, ["life"], 1)


def q_labyrinths(r):
    based(r, "Glass", "Air & Noise", "Shimmer", "Organs", "Microtonal")
    set_life(r, LIFE_ISLANDS, (0.4, 0.8), (0.3, 0.9), (0.0, 0.4), (0.5, 1.0))
    rows_from(r, ["kura"], 1)


def q_archipelagos(r):
    based(r, *CALM)
    sleepy(r)
    set_life(r, LIFE_ISLANDS, (0.3, 0.6), (0.1, 0.4), (0.5, 1.0), (0.6, 1.0))
    sleep_rows(r, ["breath", "tide", "night"], 2)


def q_gongs(r):
    based(r, "Bells", "Dark Drones", "Deep Space", "Organs")
    set_turb(r, (0.6, 1.0), (2, 12))
    if r.rng.random() < 0.4:
        r.set("stretch", f(r.u(1180, 1240), 1))
    p16_rows(r, ["turb"], 1)


def q_bronze(r):
    based(r, "Organs", "Strings", "Choirs", "Sleep Concert", "Shepard & Tides")
    set_turb(r, (0.4, 0.8), (6, 30))
    p16_rows(r, ["turb"], 1)
    rows_from(r, ["kura"], 1)


def q_cascades(r):
    based(r, "Glass", "Shimmer", "Bells", "Keys", "Air & Noise")
    set_turb(r, (0.5, 1.0), (1, 8))
    p16_rows(r, ["turb"], 1)


def q_pressure(r):
    based(r, "Keys", "Strings", "Organs", "Choirs", "Glass", "Bells")
    if r.rng.random() < 0.7:
        r.set("brain_on", 0)   # an instrument for the hands
    set_mpe(r, ["Level", "Filter", "Nearer", "Width"], (0.4, 0.9), ["Brightness", "Filter", "Shimmer", "Motion"], (0.4, 0.9))


def q_voices(r):
    based(r, "Choirs", "Strings", "Sleep Concert", "Organs")
    if r.rng.random() < 0.6:
        r.set("brain_on", 0)
    r.set("filter_model", FILTER["Formant"])   # the slide moves the vowel
    r.set("cutoff", f(r.lg(300, 2500), 0))
    r.set("resonance", f(r.u(0.2, 0.6), 2))
    set_mpe(r, ["Level", "Nearer"], (0.4, 0.8), ["Filter"], (0.3, 0.7))


def q_storms(r):
    based(r, *ALL)
    if r.rng.random() < 0.5:
        set_life(r, LIFE_FLUID, (0.4, 0.8), (0.3, 0.9), (0.4, 1.0), (0.6, 1.0))
    else:
        set_life(r, LIFE_ISLANDS, (0.4, 0.8), (0.3, 0.9), (0.2, 1.0), (0.6, 1.0))
    if r.rng.random() < 0.5:
        set_swarm(r, r.pick("wave", "splinter", "twin"))
    else:
        set_chimera(r)
    set_turb(r, (0.3, 0.8), (2, 15))
    p16_rows(r, ["swarm", "life", "turb"], 2)


RECIPES = RECIPES + [q_swarms, q_splinters, q_chimeras, q_twins, q_currents, q_vortex, q_plumes, q_islands,
                     q_labyrinths, q_archipelagos, q_gongs, q_bronze, q_cascades, q_pressure, q_voices, q_storms]


# The new means in the groups before, where they fit, drawn after everything else (what a preset had
# stays): turb = (p, amount, settle); fluid / islands = (p, depth, rate, character), only where no
# living landscape lives yet; swarm = (p, states), making a sync it has a swarm or a chimera (or
# giving one); mpe = (p, presses, slides).
SLOW = ((0.3, 0.55), (0.05, 0.3))
P16_FX = {
    "Glass":             dict(turb=(0.25, (0.3, 0.7), (1, 8))),
    "Bells":             dict(turb=(0.35, (0.4, 0.8), (2, 10))),
    "Strings":           dict(mpe=(0.4, ["Level"], ["Width", "Brightness"])),
    "Organs":            dict(turb=(0.2, (0.2, 0.5), (6, 20))),
    "Air & Noise":       dict(fluid=(0.2, (0.4, 0.7), (0.2, 0.6), (0.3, 0.9))),
    "Deep Space":        dict(fluid=(0.15, (0.3, 0.6), (0.05, 0.3), (0.0, 0.5))),
    "Waves":             dict(fluid=(0.2, (0.4, 0.7), (0.1, 0.5), (0.2, 0.7))),
    "Dark Drones":       dict(turb=(0.2, (0.3, 0.6), (6, 20))),
    "Keys":              dict(mpe=(0.5, ["Level", "Filter"], ["Brightness", "Filter"])),
    "Synchrony":         dict(swarm=(0.4, ["wave", "twin", "splinter", "chimera"])),
    "Flocks":            dict(swarm=(0.2, ["wave", "splinter"])),
    "Ripples":           dict(fluid=(0.2, (0.4, 0.7), (0.2, 0.6), (0.3, 0.8))),
    "Spectral Cascades": dict(turb=(0.4, (0.4, 0.8), (1, 8))),
    "Resonant Halls":    dict(turb=(0.3, (0.3, 0.7), (3, 15))),
    "Night Gardens":     dict(islands=(0.15,) + SLOW + ((0.5, 1.0),)),
    "Deep Sleep":        dict(islands=(0.2,) + SLOW + ((0.6, 1.0),)),
    "Dream Drift":       dict(islands=(0.2,) + SLOW + ((0.4, 1.0),)),
    "Ocean Sleep":       dict(fluid=(0.2,) + SLOW + ((0.0, 0.3),)),
    "Hypnagogia":        dict(islands=(0.2,) + SLOW + ((0.3, 0.9),)),
    "Slow Tides":        dict(fluid=(0.15,) + SLOW + ((0.0, 0.3),)),
    "Afterglow":         dict(islands=(0.2,) + SLOW + ((0.6, 1.0),)),
    "Stretched Worlds":  dict(turb=(0.25, (0.3, 0.7), (2, 12))),
    "Shifted Metals":    dict(turb=(0.3, (0.4, 0.8), (2, 10))),
    "Dream Machines":    dict(swarm=(0.2, ["wave", "twin", "splinter"])),
}


def p16_effects(r, group):
    """Phase 16 in one of the first sixty-four groups, drawn after everything else."""
    fx = P16_FX.get(group, {})
    if "turb" in fx:
        p, amount, settle = fx["turb"]
        if r.rng.random() < p:
            set_turb(r, amount, settle)
    for kind, mode in (("fluid", LIFE_FLUID), ("islands", LIFE_ISLANDS)):
        if kind in fx:
            p, depth, rate, character = fx[kind]
            if r.rng.random() < p and num(r, "life_mode", 0) == 0:   # a living landscape that grows stays
                set_life(r, mode, depth, rate, character, (0.6, 1.0))
    if "swarm" in fx:
        p, states = fx["swarm"]
        if r.rng.random() < p:
            depth = num(r, "sync_depth", 0.0)
            keep = (max(0.25, depth - 0.05), max(0.3, depth + 0.05)) if depth > 0.0 else (0.3, 0.6)
            state = r.pick(*states)
            if state == "chimera":
                set_chimera(r, depth=keep)
            else:
                set_swarm(r, state, depth=keep)
    if "mpe" in fx:
        p, presses, slides = fx["mpe"]
        if r.rng.random() < p:
            set_mpe(r, presses, (0.4, 0.8), slides, (0.4, 0.8))


# ---- Phase 17 (14.10.2026): the hall that grows into what it hears (or erodes), the partials' own
# time, the ripple's bend, the Orbit field.

FIELD_ORBIT = 5


def set_hall(r, level=(0.3, 0.5), decay=(10, 40), tune=(0.5, 1.0)):
    """A modal hall, where the preset has none yet (one it has stays as it is)."""
    if num(r, "modal_level", 0.0) > 0.0:
        return
    r.set("modal_level", f(r.u(*level), 3))
    r.set("modal_decay", f(r.lg(*decay), 1))
    r.set("modal_damp", f(r.u(0.2, 0.5), 2))
    r.set("modal_tune", f(r.u(*tune), 2))
    r.set("modal_spread", f(r.u(2, 10), 1))
    r.set("modal_drift", f(r.u(1.0, 8.0), 1))


def set_grow(r, grow, memory):
    """The hall's memory: growth (positive) or erosion, formed over Memory seconds."""
    r.set("hall_grow", f(r.u(*grow), 2))
    r.set("hall_memory", f(r.lg(*memory), 0))


def set_bend(r, bend, drift=None):
    """The partials' own time: their drift, shimmer and sync slower (positive) or faster where the
    spectrum is dense -- with drift enough to hear it."""
    r.set("time_bend", f(r.u(*bend), 2))
    if drift is not None and num(r, "drift", 3.0) < drift[0]:
        r.set("drift", f(r.u(*drift), 1))


def set_orbit(r, pull=(0.1, 0.5), radius=(15, 60), spring=(0.05, 0.4), wander=(0.01, 0.15)):
    """The Orbit field: Pull the wells' depth, Radius the orbits' size, Spring the damping, Wander
    the magnetic field (how fast they circle)."""
    r.set("field_mode", FIELD_ORBIT)
    r.set("field_strength", f(r.u(*pull), 2))
    r.set("field_radius", f(r.lg(*radius), 0))
    r.set("field_spring", f(r.u(*spring), 2))
    r.set("field_rate", f(r.lg(*wander), 3))


def set_linger(r, bend):
    """The ripple's bend: its waves slower where the notes are."""
    r.set("ripple_bend", f(r.u(*bend), 2))


P17_POOLS = {
    "hall": [("lfo3", "hall_grow", (0.1, 0.3)), ("rossler_x", "hall_grow", (0.1, 0.25)), ("kura2", "modal_level", (0.05, 0.12))],
    "time": [("lfo2", "time_bend", (0.15, 0.35)), ("loudness", "time_bend", (0.15, 0.35)), ("lorenz_x", "time_bend", (0.1, 0.3))],
    "orbit": [("lfo2", "field_radius", (0.06, 0.15)), ("kura1", "field_rate", (0.05, 0.15)), ("lorenz_z", "field_strength", (0.08, 0.2))],
    "ripple": [("lfo3", "ripple_bend", (0.1, 0.3)), ("flux", "ripple_bend", (0.15, 0.3))],
}


def p17_rows(r, kinds, count, via_p=0.25):
    """Rows onto the new knobs, their sources set slowly."""
    taken = {tuple(row.split(":")[0].split(">")) for row in r.rows}
    for _ in range(count):
        if len(r.rows) >= 16:
            return
        src, target, (lo, hi) = r.pick(*P17_POOLS[r.pick(*kinds)])
        if (src, target) in taken:
            continue
        taken.add((src, target))
        if src == "lfo2":
            setup(r, "wander")
        elif src == "lfo3":
            setup(r, "tide")
        elif src.startswith("kura"):
            setup(r, "kura")
        elif src.startswith(("lorenz", "rossler")):
            setup(r, "chaos")
        via = r.pick(*SLEEP_VIAS) if r.rng.random() < via_p else None
        r.row(src, target, round(r.u(lo, hi), 3), via)


def h_growing(r):
    based(r, "Bells", "Organs", "Choirs", "Glass", "Strings", "Dark Drones")
    set_hall(r)
    set_grow(r, (0.5, 1.0), (300, 1800))
    p17_rows(r, ["hall"], 1)
    rows_from(r, ["kura"], 1)


def h_eroded(r):
    based(r, "Bells", "Glass", "Organs", "Microtonal", "Keys", "Air & Noise")
    set_hall(r, decay=(15, 50))
    set_grow(r, (-1.0, -0.5), (300, 1800))
    p17_rows(r, ["hall"], 1)


def h_cathedral(r):
    based(r, *CALM)
    sleepy(r)
    set_hall(r, level=(0.25, 0.45), decay=(20, 60))
    set_grow(r, (0.4, 0.9), (3600, 14400))
    sleep_rows(r, ["breath", "tide"], 2)


def h_worn(r):
    based(r, "Dark Drones", "Deep Space", "Organs", "Sleep Concert", "Shepard & Tides")
    set_hall(r, level=(0.25, 0.45), decay=(20, 60))
    set_grow(r, (-0.9, -0.4), (1800, 7200))
    sleep_rows(r, ["tide", "chaos"], 1)


def t_slow(r):
    based(r, "Choirs", "Strings", "Sleep Concert", "Organs", "Shimmer", "Deep Space")
    set_bend(r, (0.6, 1.0), drift=(6, 20))
    if num(r, "shimmer", 0.4) < 0.3:
        r.set("shimmer", f(r.u(0.3, 0.6), 2))
    p17_rows(r, ["time"], 1)


def t_quick(r):
    based(r, "Air & Noise", "Glass", "Shimmer", "Waves", "Glissandi")
    set_bend(r, (-1.0, -0.5), drift=(4, 15))
    p17_rows(r, ["time"], 1)


def t_lens(r):
    based(r, "Bells", "Glass", "Keys", "Pendulums", "Shimmer", "Microtonal")
    set_bend(r, (0.5, 1.0), drift=(4, 12))
    set_ripple(r, (0.4, 0.8), (0.3, 1.5), (5, 20))
    set_linger(r, (0.5, 1.0))
    if r.rng.random() < 0.5:
        r.set("dispersion", f(r.u(0.2, 0.8), 2))


def t_linger(r):
    based(r, "Keys", "Bells", "Glass", "Pendulums", "Waves", "Strings")
    set_ripple(r, (0.5, 1.0), (0.5, 3.0), (4, 20))
    set_linger(r, (0.6, 1.0))
    if r.p.get("brain_on") != 0:   # the conductor changes notes often: every change strikes the string
        r.set("auto_rate", f(r.lg(8, 30), 1))
    p17_rows(r, ["ripple"], 1)


def o_orbits(r):
    based(r, "Choirs", "Strings", "Organs", "Glass", "Shimmer", "Sleep Concert")
    set_orbit(r)
    p17_rows(r, ["orbit"], 1)


def o_cyclotrons(r):
    based(r, "Glass", "Bells", "Keys", "Air & Noise", "Shimmer", "Microtonal")
    set_orbit(r, pull=(0.4, 0.9), radius=(8, 30), spring=(0.02, 0.2), wander=(0.15, 0.6))
    p17_rows(r, ["orbit"], 1)


def o_planetaria(r):
    based(r, *CALM)
    sleepy(r)
    set_orbit(r, pull=(0.05, 0.25), radius=(20, 80), spring=(0.02, 0.15), wander=(0.005, 0.04))
    sleep_rows(r, ["breath", "wander"], 2)


def o_magnetic(r):
    based(r, "Choirs", "Strings", "Organs", "Sleep Concert")
    set_orbit(r, pull=(0.2, 0.6), radius=(10, 40), spring=(0.1, 0.4), wander=(0.02, 0.2))
    if r.rng.random() < 0.5:
        set_swarm(r, r.pick("wave", "twin"), depth=(0.2, 0.5))
    else:
        rows_from(r, ["kura"], 1)


def x_horizons(r):
    based(r, "Deep Space", "Dark Drones", "Sleep Concert", "Shepard & Tides", "Organs")
    if int(num(r, "filter_model", 0)) in (FILTER["LP 12"], FILTER["LP 24"]) and num(r, "cutoff", 2500.0) < 300.0:
        r.set("cutoff", f(num(r, "cutoff", 2500.0) * 2.5, 0))   # a drone's low pass this deep left one 3.6 dB under the bank's level at +12 dB
    set_bend(r, (0.7, 1.0), drift=(5, 15))
    r.set("dispersion", f(r.u(0.4, 1.2), 2))
    set_hall(r, level=(0.25, 0.45), decay=(25, 60))
    set_grow(r, (0.3, 0.8), (900, 3600))
    p17_rows(r, ["time", "hall"], 1)


def x_palaces(r):
    based(r, "Sleep Concert", "Strings", "Keys", "Choirs", "Organs")
    set_hall(r)
    set_grow(r, (0.4, 1.0), (600, 3600))
    r.set("memory_level", f(r.u(0.3, 0.6), 2))
    r.set("memory_time", f(r.lg(90, 300), 0))
    r.set("memory_bloom", f(r.lg(4, 15), 1))
    rows_from(r, ["piece"], 1)


def x_mechanics(r):
    based(r, "Bells", "Glass", "Pendulums", "Organs", "Choirs", "Microtonal")
    set_orbit(r, pull=(0.2, 0.6), radius=(10, 50), spring=(0.05, 0.3), wander=(0.02, 0.3))
    set_bend(r, (0.3, 0.8), drift=(3, 10))
    p17_rows(r, ["orbit", "time"], 1)


def x_living(r):
    based(r, *ALL)
    kinds = []
    while len(kinds) < 2:
        kinds = []
        if r.rng.random() < 0.6:
            kinds.append("hall")
        if r.rng.random() < 0.6:
            kinds.append("time")
        if r.rng.random() < 0.5 and num(r, "field_mode", 0) == 0:
            kinds.append("orbit")
        if r.rng.random() < 0.4:
            kinds.append("ripple")
    if "hall" in kinds:
        set_hall(r)
        set_grow(r, (0.4, 1.0) if r.rng.random() < 0.6 else (-1.0, -0.4), (600, 3600))
    if "time" in kinds:
        set_bend(r, (0.4, 1.0) if r.rng.random() < 0.6 else (-1.0, -0.4), drift=(4, 15))
    if "orbit" in kinds:
        set_orbit(r)
    if "ripple" in kinds:
        set_ripple(r, (0.4, 0.9), (0.4, 2.5), (4, 20))
        set_linger(r, (0.5, 1.0))
    p17_rows(r, kinds, 2)


RECIPES = RECIPES + [h_growing, h_eroded, h_cathedral, h_worn, t_slow, t_quick, t_lens, t_linger,
                     o_orbits, o_cyclotrons, o_planetaria, o_magnetic, x_horizons, x_palaces, x_mechanics, x_living]


# The new means in the groups before, where they fit, drawn after everything else (what a preset had
# stays): grow = (p, grow, memory), a quiet hall given where there is none; bend = (p, bend);
# linger = (p, bend), only where a ripple runs; orbit = (p, gentle), a Scale field or none becoming
# an orbit (gentle: the sleep concerts' slow wide circles).
P17_FX = {
    "Organs":             dict(grow=(0.15, (0.3, 0.8), (900, 3600))),
    "Dark Drones":        dict(grow=(0.15, (-0.7, -0.3), (1800, 7200))),
    "Shimmer":            dict(bend=(0.15, (-0.8, -0.3))),
    "Glass":              dict(bend=(0.1, (-0.8, -0.3))),
    "Microtonal":         dict(orbit=(0.15, False)),
    "Keys":               dict(linger=(0.4, (0.4, 0.9))),
    "Bells":              dict(linger=(0.3, (0.4, 0.9))),
    "Synchrony":          dict(bend=(0.25, (0.3, 0.8))),
    "Ripples":            dict(linger=(0.45, (0.4, 1.0))),
    "Resonant Halls":     dict(grow=(0.35, (0.3, 0.9), (600, 3600))),
    "Ghosts & Memory":    dict(grow=(0.25, (0.3, 0.8), (600, 3600))),
    "Gravity Wells":      dict(orbit=(0.35, False)),
    "Smear & Dispersion": dict(bend=(0.4, (0.4, 1.0))),
    "Deep Sleep":         dict(grow=(0.15, (0.2, 0.6), (3600, 14400))),
    "Dream Drift":        dict(bend=(0.2, (0.3, 0.8))),
    "Starfield":          dict(orbit=(0.2, True)),
    "Hypnagogia":         dict(bend=(0.2, (0.3, 0.8))),
    "Twilight Choirs":    dict(grow=(0.2, (0.2, 0.6), (1800, 7200))),
    "Warm Drones":        dict(grow=(0.2, (0.2, 0.6), (1800, 7200))),
    "Afterglow":          dict(grow=(0.15, (-0.6, -0.2), (1800, 7200))),
    "Star Streams":       dict(orbit=(0.2, True)),
    "Swarms":             dict(bend=(0.2, (0.3, 0.8))),
    "Chimeras":           dict(bend=(0.2, (0.3, 0.8))),
}


def p17_effects(r, group):
    """Phase 17 in one of the first eighty groups, drawn after everything else."""
    fx = P17_FX.get(group, {})
    if "grow" in fx:
        p, grow, memory = fx["grow"]
        if r.rng.random() < p:
            set_hall(r, level=(0.15, 0.3), decay=(15, 45))
            set_grow(r, grow, memory)
    if "bend" in fx:
        p, bend = fx["bend"]
        if r.rng.random() < p:
            set_bend(r, bend)
    if "linger" in fx:
        p, bend = fx["linger"]
        if r.rng.random() < p and num(r, "ripple_depth", 0.0) > 0.0:
            set_linger(r, bend)
    if "orbit" in fx:
        p, gentle = fx["orbit"]
        if r.rng.random() < p and int(num(r, "field_mode", 0)) in (0, 1):
            if gentle:
                set_orbit(r, pull=(0.05, 0.25), radius=(20, 80), spring=(0.02, 0.15), wander=(0.005, 0.04))
            else:
                set_orbit(r)


# ---- Phase 26 (15.10.2026): the harmonics' own lives -- registers that overblow, seasons of overtone
# families, avalanches on a rice pile, subharmonics pumped from below -- and grains, the horizon, breathers.

SLEEP_GROUPS = tuple(GROUPS[32:48])


def set_overblow(r, blow=(0.2, 0.45), time=(0.3, 1.5)):
    """Overblow near its thresholds (the octave from a breath of about 0.26, the twelfth from 0.6): how hard
    a note is played decides its register, and a note near an edge flickers."""
    r.set("overblow", f(r.u(*blow), 2))
    r.set("overblow_time", f(r.lg(*time), 2))


def set_breath_press(r, depth=(0.4, 0.9)):
    """MPE's pressure as the breath (Overblow's, and the subharmonics' pump)."""
    r.set("mpe_press", MPE_T["Breath"])
    r.set("mpe_press_depth", f(r.u(*depth), 2))


def set_seasons(r, depth=(0.5, 0.9), year=(120, 600)):
    """The seasons of the overtone families, a whole round in Year seconds."""
    r.set("seasons", f(r.u(*depth), 2))
    r.set("season_time", f(r.lg(*year), 0))


def set_avalanche(r, depth=(0.5, 0.9), rate=(0.15, 0.6), glow=(0.3, 1.0)):
    """The rice pile on every note's harmonics: below a grain a second it is mostly still."""
    r.set("avalanche", f(r.u(*depth), 2))
    r.set("avalanche_rate", f(r.lg(*rate), 3))
    r.set("avalanche_glow", f(r.lg(*glow), 2))


def set_subs(r, depth=(0.5, 0.9), time=(0.5, 2.0)):
    """The subharmonics: a loud note grows a floor under it, the slower the nearer its threshold."""
    r.set("subharmonics", f(r.u(*depth), 2))
    r.set("sub_time", f(r.lg(*time), 2))


def set_horizon(r, amount=(0.6, 1.0), height=(2.0, 4.0), colour=True):
    """The harmonics bent towards a horizon `height` octaves up -- with their upper harmonics left in, so the
    halo is heard (colour: a new group's sound may be brightened for it; a preset from before keeps its own)."""
    r.set("horizon", f(r.u(*amount), 2))
    r.set("horizon_height", f(r.u(*height), 2))
    if not colour:
        return
    if num(r, "tilt", 1.2) > 1.1:
        r.set("tilt", f(r.u(0.6, 1.1), 2))
    if num(r, "brightness", 1.0) < 0.8:
        r.set("brightness", f(r.u(0.8, 1.0), 2))


def set_grains(r, share=(0.5, 1.0), length=(0.02, 0.1), density=(0.03, 0.15)):
    """A share of the partials in grains, each born at a new place in its cloud."""
    r.set("grain", f(r.u(*share), 2))
    r.set("grain_length", f(r.lg(*length), 3))
    r.set("grain_density", f(r.lg(*density), 3))


def set_breathers(r, amount=(0.6, 1.0), lengthen=True):
    """The ripple as a sine-Gordon chain: its strikes stay as breathers, long-lived (a ripple is set where there
    is none; one there is kept, its decay lengthened unless the preset is from before and keeps its own)."""
    if num(r, "ripple_depth", 0.0) <= 0.0:
        set_ripple(r, (0.5, 0.9), (0.6, 1.5), (15, 30))
    elif lengthen:
        r.set("ripple_decay", f(max(num(r, "ripple_decay", 6.0), r.lg(15, 30)), 2))
    r.set("ripple_breathers", f(r.u(*amount), 2))


P26_POOLS = {
    "blow": [("lfo4", "overblow", (0.1, 0.25)), ("pressure", "overblow", (0.2, 0.4)), ("lfo2", "overblow", (0.08, 0.2))],
    "season": [("lfo2", "seasons", (0.1, 0.25)), ("root_age", "seasons", (0.1, 0.3))],
    "avalanche": [("lfo2", "avalanche_rate", (0.1, 0.25)), ("loudness", "avalanche", (0.1, 0.3)),
                  ("flux", "avalanche_rate", (0.1, 0.2))],
    "sub": [("pressure", "subharmonics", (0.2, 0.4)), ("lfo3", "subharmonics", (0.1, 0.25)),
            ("root_age", "subharmonics", (0.15, 0.3))],
    "horizon": [("lfo3", "horizon_height", (0.08, 0.2)), ("lfo2", "horizon", (0.1, 0.25)),
                ("lorenz_x", "horizon_height", (0.06, 0.15))],
    "grain": [("lfo2", "grain_density", (0.1, 0.25)), ("lfo4", "grain_length", (0.05, 0.15)), ("kura1", "grain", (0.1, 0.2))],
    "breather": [("lfo3", "ripple_breathers", (0.1, 0.2)), ("flux", "ripple_depth", (0.1, 0.2))],
}


def p26_rows(r, kinds, count, via_p=0.2):
    """Rows onto the new knobs, their sources set slowly."""
    taken = {tuple(row.split(":")[0].split(">")) for row in r.rows}
    for _ in range(count):
        if len(r.rows) >= 16:
            return
        src, target, (lo, hi) = r.pick(*P26_POOLS[r.pick(*kinds)])
        if (src, target) in taken:
            continue
        taken.add((src, target))
        if src == "lfo2":
            setup(r, "wander")
        elif src == "lfo3":
            setup(r, "tide")
        elif src == "lfo4":
            setup(r, "breath")
        elif src.startswith("kura"):
            setup(r, "kura")
        elif src.startswith(("lorenz", "rossler")):
            setup(r, "chaos")
        via = r.pick(*SLEEP_VIAS) if r.rng.random() < via_p else None
        r.row(src, target, round(r.u(lo, hi), 3), via)


def high_register(r, low=(55, 62)):
    """The conductor up high, so a floor can open under it (and no Foundation: brain_low 55 and up)."""
    if r.p.get("brain_on") == 0:
        return
    lo = r.rng.randint(*low)
    r.set("brain_low", lo)
    r.set("brain_high", r.rng.randint(lo + 18, min(lo + 30, 96)))


def e_flutes(r):
    based(r, "Organs", "Air & Noise", "Choirs", "Strings", "Glass")
    set_overblow(r, (0.2, 0.45), (0.2, 1.0))
    if r.rng.random() < 0.6:
        set_breath_press(r)
    p26_rows(r, ["blow"], 1)


def e_registers(r):
    based(r, *CALM)
    set_overblow(r, (0.1, 0.3), (0.6, 3.0))
    setup(r, "breath")
    r.row("lfo4", "overblow", round(r.u(0.15, 0.3), 3))   # the register rises and falls with a slow breath
    sleep_rows(r, ["tide"], 1)


def e_seasons(r):
    based(r, "Choirs", "Strings", "Organs", "Glass", "Shimmer", "Sleep Concert", "Bells")
    set_seasons(r, (0.5, 0.9), (60, 300))
    p26_rows(r, ["season"], 1)


def e_years(r):
    based(r, *CALM)
    sleepy(r)
    set_seasons(r, (0.4, 0.8), (600, 3600))
    sleep_rows(r, ["breath", "tide"], 2)


def e_avalanches(r):
    based(r, "Bells", "Glass", "Keys", "Strings", "Shimmer", "Microtonal")
    set_avalanche(r, (0.5, 0.9), (0.15, 0.6), (0.3, 1.0))
    p26_rows(r, ["avalanche"], 1)


def e_glacier(r):
    based(r, "Dark Drones", "Deep Space", "Shepard & Tides", "Sleep Concert", "Organs")
    set_avalanche(r, (0.6, 1.0), (0.05, 0.2), (1.0, 3.0))
    sleep_rows(r, ["tide"], 1)


def e_subterranea(r):
    based(r, "Glass", "Shimmer", "Choirs", "Strings", "Bells")
    set_subs(r, (0.6, 1.0), (0.8, 3.0))
    high_register(r)
    r.row("root_age", "subharmonics", round(r.u(0.15, 0.3), 3))   # the longer the harmony stands, the deeper
    p26_rows(r, ["sub"], 1)


def e_doubling(r):
    based(r, "Bells", "Organs", "Dark Drones", "Strings", "Choirs")
    set_subs(r, (0.6, 1.0), (0.3, 1.5))
    set_breath_press(r)
    if r.rng.random() < 0.4:   # up and down at once: turbulence above, the subharmonics below
        set_turb(r, amount=(0.3, 0.7))
    p26_rows(r, ["sub"], 1)


def e_halo(r):
    based(r, "Choirs", "Organs", "Strings", "Sleep Concert", "Deep Space", "Shimmer")
    set_horizon(r, (0.7, 1.0), (2.0, 4.0))
    p26_rows(r, ["horizon"], 1)


def e_curtains(r):
    based(r, "Deep Space", "Shepard & Tides", "Sleep Concert", "Choirs", "Shimmer")
    set_horizon(r, (0.5, 1.0), (1.5, 3.5))
    setup(r, "tide")
    r.row("lfo3", "horizon_height", round(r.u(0.1, 0.25), 3))   # the halo travels like a curtain
    p26_rows(r, ["horizon"], 1)


def e_rain(r):
    based(r, "Glass", "Bells", "Keys", "Shimmer", "Air & Noise", "Strings")
    set_grains(r, (0.6, 1.0), (0.015, 0.08), (0.02, 0.12))
    p26_rows(r, ["grain"], 1)


def e_grain_breath(r):
    based(r, "Choirs", "Strings", "Sleep Concert", "Organs", "Deep Space", "Air & Noise")
    set_grains(r, (0.5, 1.0), (0.15, 0.8), (0.25, 0.7))
    p26_rows(r, ["grain"], 1)


def e_breathers(r):
    based(r, "Keys", "Bells", "Glass", "Strings", "Pendulums", "Waves")
    set_breathers(r)
    if r.p.get("brain_on") != 0:   # every change of the conductor's strikes the string
        r.set("auto_rate", f(r.lg(10, 40), 1))
    p26_rows(r, ["breather"], 1)


def e_ecology(r):
    based(r, *ALL)
    kinds = []
    while len(kinds) < 2:
        kinds = [k for k in ("blow", "season", "avalanche", "sub") if r.rng.random() < 0.5]
    if "blow" in kinds:
        set_overblow(r, (0.15, 0.4), (0.3, 1.5))
    if "season" in kinds:
        set_seasons(r, (0.4, 0.8), (90, 480))
    if "avalanche" in kinds:
        set_avalanche(r, (0.4, 0.8), (0.1, 0.4), (0.4, 1.2))
    if "sub" in kinds:
        set_subs(r, (0.5, 0.9), (0.5, 2.0))
    p26_rows(r, kinds, 2)


def e_night(r):
    based(r, *CALM)
    sleepy(r)
    kinds = []
    while len(kinds) < 2:
        kinds = [k for k in ("season", "sub", "breather", "grain") if r.rng.random() < 0.5]
    if "season" in kinds:
        set_seasons(r, (0.3, 0.7), (900, 3600))
    if "sub" in kinds:
        set_subs(r, (0.4, 0.8), (2.0, 6.0))
    if "breather" in kinds:
        set_breathers(r, (0.5, 0.9))
        r.set("ripple_depth", f(min(num(r, "ripple_depth", 0.4), 0.4), 2))
    if "grain" in kinds:
        set_grains(r, (0.3, 0.7), (0.3, 1.0), (0.3, 0.7))
    sleep_rows(r, ["breath", "tide"], 1)


def e_wild(r):
    based(r, *ALL)
    set_overblow(r, (0.3, 0.7), (0.1, 0.5))
    set_avalanche(r, (0.7, 1.0), (0.5, 2.0), (0.2, 0.6))
    if r.rng.random() < 0.5:
        set_subs(r, (0.6, 1.0), (0.3, 1.0))
    if r.rng.random() < 0.4:
        set_grains(r, (0.3, 0.7), (0.02, 0.1), (0.05, 0.2))
    p26_rows(r, ["blow", "avalanche"], 2)


RECIPES = RECIPES + [e_flutes, e_registers, e_seasons, e_years, e_avalanches, e_glacier, e_subterranea, e_doubling,
                     e_halo, e_curtains, e_rain, e_grain_breath, e_breathers, e_ecology, e_night, e_wild]


# The new means in the groups before, where they fit, drawn after everything else (what a preset had stays):
# kind -> the share of the group's presets that get it. The sleep concerts get the gentle kinds, gently.
P26_FX = {
    "Organs": dict(blow=0.2), "Choirs": dict(blow=0.1, season=0.1), "Strings": dict(season=0.1),
    "Bells": dict(sub=0.15, avalanche=0.1), "Glass": dict(avalanche=0.15, grain=0.1), "Keys": dict(avalanche=0.1),
    "Dark Drones": dict(sub=0.2, horizon=0.1), "Deep Space": dict(horizon=0.15), "Shimmer": dict(grain=0.15),
    "Air & Noise": dict(grain=0.2), "Ripples": dict(breather=0.3), "Gong Blooms": dict(sub=0.3),
    "Bronze Swells": dict(sub=0.25), "Pressure Pads": dict(blow=0.25), "Expressive Voices": dict(blow=0.2),
    "Snowfall": dict(grain=0.25), "Night Rain": dict(grain=0.25), "Raindrops": dict(grain=0.2),
    "Warm Drones": dict(season=0.2), "Slow Tides": dict(season=0.2), "Lingering Waves": dict(breather=0.35),
    "Planetaria": dict(season=0.2), "Celestial Mechanics": dict(season=0.2), "Event Horizons": dict(horizon=0.3),
    "Gravity Lenses": dict(horizon=0.2), "Living Storms": dict(avalanche=0.25), "Shimmer Cascades": dict(avalanche=0.2),
}


def p26_effects(r, group):
    """Phase 26 in one of the first ninety-six groups, drawn after everything else (the floor too)."""
    fx = P26_FX.get(group, {})
    gentle = group in SLEEP_GROUPS
    for kind, share in fx.items():
        if r.rng.random() >= share:
            continue
        if kind == "blow":
            set_overblow(r, (0.15, 0.35), (0.5, 2.0))   # by velocity: their pressure keeps what it moved
        elif kind == "season":
            set_seasons(r, (0.3, 0.6) if gentle else (0.4, 0.8), (900, 3600) if gentle else (120, 600))
        elif kind == "avalanche":
            set_avalanche(r, (0.4, 0.8), (0.1, 0.4), (0.4, 1.2))
        elif kind == "sub":
            set_subs(r, (0.4, 0.8), (1.0, 4.0))
        elif kind == "horizon":
            set_horizon(r, (0.4, 0.8), (2.5, 4.5), colour=False)
        elif kind == "grain":
            if gentle:
                set_grains(r, (0.2, 0.5), (0.3, 1.0), (0.3, 0.7))
            else:
                set_grains(r, (0.3, 0.7), (0.02, 0.2), (0.05, 0.3))
        elif kind == "breather":
            set_breathers(r, (0.5, 0.9), lengthen=False)


def foundation(r):
    """The conductor's floor (Foundation, 14.10.2026): most presets hold their root low under the chord;
    a few float, and none whose register has no bass. Drawn last: what the preset had stays as it was."""
    if int(num(r, "brain_low", 36)) >= 55 or r.rng.random() < 0.15:
        return
    r.set("brain_foundation", f(r.u(0.45, 0.85), 2))


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
            if g < 48:   # phase 13 drawn last: what the preset had stays as it was
                p13_effects(r, group)
            if g < 64:   # and phase 16 after it
                p16_effects(r, group)
            if g < 80:   # and phase 17 after that
                p17_effects(r, group)
            foundation(r)   # and the conductor's floor after everything
            if g < 96:   # and phase 26 after it, so what a preset had stays as it was
                p26_effects(r, GROUPS[g])
            r.p.pop("_sleep", None)   # the sleep modulations' bookkeeping, not a parameter
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
             "/// The factory presets: 112 groups of 64 (Presets.h).",
             "const Preset Presets[] = {"]
    for name, g, params, mod in bank:
        lines.append(f'    {{ "{name}", {g}, "{params}", "{mod}" }},')
    lines.append("};")
    OUT.write_bytes(("\n".join(lines) + "\n").encode("utf-8"))
    levelled = "levelled" if LEVELS.exists() else "not levelled yet (run additiv_presets)"
    print(f"{len(bank)} presets in {len(GROUPS)} groups -> {OUT} ({levelled})")


if __name__ == "__main__":
    main()
