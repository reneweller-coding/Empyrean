/**
 * @file Params.h
 * @brief The one parameter table: every parameter with its key, name, section, range, default, skew and unit.
 *
 * The engine, the plugin's parameter layout, the generic editor, presets and the renderer all
 * iterate this table (after Noctuary's Params.h: adding a parameter is one row here and one line
 * where the engine reads it). Also the choice names that have no home elsewhere: the scales,
 * the root names, the conductor's modes and the motions.
 * @author Rene Weller
 */
#pragma once

#include <array>
#include <cstddef>

namespace additiv
{

/** @brief Every parameter, in table order. */
enum class ParamIdE : int
{
    // Master
    MASTER_GAIN,   ///< output level, dB
    // Cloud: the voice's spectrum and its own life
    PARTIALS,       ///< partials per voice
    TILT,           ///< spectral tilt: harmonic h at h^-tilt
    BRIGHTNESS,     ///< 0 .. 1: harmonics above 1 + b^2 * 200 fade out (1 keeps them all)
    ODD_EVEN,       ///< -1 .. 1: odd against even harmonics
    INHARMONIC,     ///< 0 .. 1: the stiff string's stretch
    CLOUD_WIDTH,    ///< the cloud around harmonic 1, cents
    WIDTH_SLOPE,    ///< how the cloud widens with the harmonic number
    DRIFT,          ///< each partial's own pitch wander, cents
    DRIFT_RATE,     ///< its speed, Hz
    SHIMMER,        ///< each partial's own amplitude wander, 0 .. 1
    SHIMMER_RATE,   ///< its speed, Hz
    STEREO,         ///< how far the partials scatter across the field
    // Air: a band of noise made of partials
    AIR,          ///< the band's level against the voice's, 0 .. 1
    AIR_CENTER,   ///< its centre, times the fundamental
    AIR_WIDTH,    ///< its width, cents
    AIR_JITTER,   ///< how fast its partials flicker, Hz
    // Filter: read as a frequency response at every partial
    FILTER_MODEL,   ///< FilterModelNames
    CUTOFF,         ///< Hz
    RESONANCE,      ///< 0 .. 1
    FILTER_ENV,     ///< octaves the voice's envelope moves the cutoff
    KEY_TRACK,      ///< 0 .. 1: how far the cutoff follows the note
    FILTER_DRIFT,   ///< 0 .. 1: the cutoff's own slow wander, up to an octave
    // Motion: the envelope lanes, as a scene
    MOTION,         ///< which motion the partials' envelopes make (MotionNames)
    MOTION_RATE,    ///< how fast the motion runs, times its own speed
    MOTION_DEPTH,   ///< how deep it reaches, times its own depth
    LANDSCAPE,      ///< which landscape the Aurora motion reads (LandscapeNames)
    // Envelope of every voice
    ATTACK,    ///< seconds
    DECAY,     ///< seconds
    SUSTAIN,   ///< 0 .. 1
    RELEASE,   ///< seconds
    // Tuning
    SCALE,         ///< ScaleNames
    ROOT_NOTE,     ///< the scale's root, a pitch class (RootNames)
    KEY_MAP,       ///< Snap to 12 keys or consecutive degrees
    REF_PITCH,     ///< A4 in Hz
    PURITY,        ///< 0 = 12-TET, 1 = the scale; the clouds narrow with it so pure intervals can lock
    TIDE,          ///< cents the whole instrument leans on a minute-scale curve
    TIDE_PERIOD,   ///< minutes per swing
    // Conductor
    BRAIN_ON,           ///< the conductor plays
    AUTO_MODE,          ///< Free or Chords (Autoplay), BrainModeNames
    BRAIN_RATE,         ///< Free: mean seconds between events
    AUTO_RATE,          ///< Chords: seconds between exchanges ("Every")
    BRAIN_DENSITY,      ///< notes sounding at once
    BRAIN_HOLD_MIN,     ///< Free: shortest hold, seconds
    BRAIN_HOLD_MAX,     ///< Free: longest hold, seconds
    BRAIN_LOW,          ///< lowest note the conductor chooses
    BRAIN_HIGH,         ///< highest note
    BRAIN_CONSONANCE,   ///< 0 anything goes .. 1 strictly consonant
    BRAIN_WANDER,       ///< how often the root moves (Free)
    AUTO_LEAD,          ///< Chords: how far the exchanged voice may move, semitones
    AUTO_TENSION,       ///< Chords: 0 = the new note must fit the chord, 1 = anything goes
    AUTO_ROOT_MOVE,     ///< Chords: how often an exchange moves the root too
    BRAIN_TIMBRE,       ///< how much consonance is judged from the cloud's own spectrum (Sethares)
    AUTO_STEP,          ///< Autoplay: a rising edge exchanges a voice now
    BRAIN2_ON,          ///< the second conductor, in the background
    BRAIN2_INTERVAL,    ///< its root against the first one's, semitones
    BRAIN2_DENSITY,     ///< its notes at once
    BRAIN2_RATE,        ///< seconds between its events
    BRAIN2_DEPTH,       ///< where its notes stand, 0 near .. 1 far
    // Space
    DEPTH,          ///< how deep the conductor's notes sit (near/far planes)
    KEYS_DEPTH,     ///< where played notes sit, 0 at the ear .. 1 far
    FAR_LEVEL,      ///< the far plane's reverb level
    FAR_SIZE,       ///< the far reverb's size
    FAR_DECAY,      ///< its decay, seconds
    FAR_DAMP,       ///< its damping
    PRESENCE,       ///< dB: the 2-5 kHz bell on the near plane
    PAD_LOW_CUT,    ///< Hz: below it the clouds fall 12 dB/oct, leaving the bottom free
    BREATH,         ///< 0 .. 1: every voice's distance wanders
    BREATH_RATE,    ///< Hz
    DOPPLER,        ///< 0 .. 1: a voice coming closer rises, one receding falls
    BASS_MONO,      ///< below this the mix is mono, Hz
    STEREO_WIDTH,   ///< the master's width
    // Modulation: the sources' own settings (the matrix rows are data, Modulation.h)
    LFO1_SHAPE,   ///< LfoShapeNames
    LFO1_RATE,    ///< Hz
    LFO2_SHAPE,   ///< LfoShapeNames
    LFO2_RATE,    ///< Hz
    LFO3_SHAPE,   ///< LfoShapeNames
    LFO3_RATE,    ///< Hz
    LFO4_SHAPE,   ///< LfoShapeNames
    LFO4_RATE,    ///< Hz
    ENV1_SHAPE,   ///< a factory shape (Shapes.h)
    ENV1_RATE,    ///< how fast it runs, times its own speed
    ENV2_SHAPE,   ///< a factory shape
    ENV2_RATE,    ///< how fast it runs
    MACRO1,       ///< 0 .. 1
    MACRO2,       ///< 0 .. 1
    MACRO3,       ///< 0 .. 1
    MACRO4,       ///< 0 .. 1
    // Echo: Noctuary's two asymmetric delays, the second in series after the first (12.10.2026)
    DELAY_MIX,         ///< the echoes on the near plane, 0 .. 1
    DELAY_TIME_L,      ///< s
    DELAY_TIME_R,      ///< s
    DELAY_FEEDBACK,    ///< 0 .. 0.95
    DELAY_CROSS,       ///< 0 .. 1: how much of each side's feedback comes from the other
    DELAY_DAMP,        ///< 0 .. 1: the loop's low pass
    DELAY_ABSORB,      ///< 0 .. 1: the loop narrows to a band as the feedback rises
    DELAY_DUCK,        ///< 0 .. 1: the loop darkens while the input is loud
    DELAY_TO_FAR,      ///< 0 .. 1: the echoes receding into the far plane
    DELAY2_MIX,        ///< the second echo, 0 .. 1
    DELAY2_TIME_L,     ///< s
    DELAY2_TIME_R,     ///< s
    DELAY2_FEEDBACK,   ///< 0 .. 0.95
    DELAY2_DAMP,       ///< 0 .. 1
    DELAY2_TO_FAR,     ///< 0 .. 1
    // Room: the near plane's early reflections and small reverb
    EARLY_LEVEL,    ///< 0 .. 1
    EARLY_SIZE,     ///< m
    EARLY_ABSORB,   ///< 0 .. 1
    NEAR_MIX,       ///< the near reverb, 0 .. 1 wet
    NEAR_DECAY,     ///< s
    NEAR_DAMP,      ///< 0 .. 1
    // The far hall, more of Noctuary's
    FAR_DIFFUSE,     ///< 0 .. 1: the swell in front of the far reverb
    FAR_PRE_DELAY,   ///< ms
    FAR_UNMASK,      ///< 0 .. 1: the far hall steps aside for the near plane, band by band
    FAR_MODE,        ///< FarModeNames
    FAR_FREEZE,      ///< the far hall held
    // The modal hall: tens of thousands of decaying resonances on the GPU
    MODAL_LEVEL,    ///< 0 .. 1
    MODAL_DECAY,    ///< s to -60 dB, at and below 250 Hz
    MODAL_DAMP,     ///< 0 .. 1: how much faster the upper octaves die
    MODAL_TUNE,     ///< 0 .. 1: the share of the modes tuned to the scale
    MODAL_SPREAD,   ///< ct: how far a tuned mode lies from its note
    MODAL_DRIFT,    ///< ct: every mode's slow wander
    MODAL_WIDTH,    ///< 0 .. 1: mono .. every mode its own place
    MODAL_LOW,      ///< Hz: the lowest mode
    MODAL_FREEZE,   ///< the modal hall held, its input shut
    MODAL_BREATH,   ///< 0 .. 1: every mode's decay wanders, slowly
    // The field on the partials and the dispersion of their clocks (Cloud.h SpectralField)
    FIELD_MODE,       ///< FieldModeNames
    FIELD_STRENGTH,   ///< 0 .. 1: how fast the partials may move (600 x^2 cents a second)
    FIELD_RADIUS,     ///< ct: an attractor's reach; the Scale field's stiffness
    FIELD_SPRING,     ///< 0 .. 1: the pull back home (4 x^2 per second)
    FIELD_RATE,       ///< how fast the attractors wander (time of the Lorenz system per second)
    FIELD_COUNT,      ///< 1 .. 8 attractors
    DISPERSION,       ///< s: the lowest partials' clock late (positive) or the highest (negative)
    // Phase 9 (12.10.2026)
    BINAURAL,          ///< the near plane placed round the head (headphones)
    BINAURAL_SPAN,     ///< 0 .. 1: the places over the front half .. all round
    BINAURAL_HEIGHT,   ///< 0 .. 1: the elevations' spread
    BINAURAL_ORBIT,    ///< Hz: the whole cloud turning round the head (negative: the other way)
    MEMORY_LEVEL,      ///< 0 .. 1: the ghosts of the notes let go
    MEMORY_TIME,       ///< s: how long a ghost takes to fade by 60 dB
    MEMORY_BLOOM,      ///< s: how long it takes to rise
    MEMORY_BLUR,       ///< 0 .. 1: how far it widens, drifts and darkens as it fades
    MEMORY_SINK,       ///< 0 .. 1: how far it recedes into the far plane
    TUNE_GRAVITY,      ///< 0 .. 1: self-tuning: voices whose harmonics nearly meet pull together
    TUNE_REACH,        ///< ct: how near they must be
    TUNE_ANCHOR,       ///< 0 .. 1: the pull back to the scale
    MODAL_PAINT,       ///< 0 .. 1: the modal hall heard through the landscape
    SPEC_MIX,          ///< 0 .. 1: the spectral echo on the near plane
    SPEC_TIME,         ///< s: its bands' middle delay
    SPEC_TILT,         ///< -1 .. 1: the highs later (positive) or earlier
    SPEC_SCATTER,      ///< 0 .. 1: each band a time of its own
    SPEC_FEEDBACK,     ///< 0 .. 0.95
    SPEC_DAMP,         ///< 0 .. 1: less feedback towards the top
    SPEC_TO_FAR,       ///< 0 .. 1: its share into the far plane
    // Phase 10a: Noctuary's modulators
    COHERENCE,        ///< 0 .. 1: the Kuramoto bank's coupling (free .. in step)
    COHERENCE_RATE,   ///< x: its speed (natural periods 23, 31, 41, 53 s)
    LENIA_RATE,       ///< Hz: Lenia fields per second
    LENIA_GROWTH,     ///< Lenia's growth centre (mu)
    CHAOS_PERIOD,     ///< s: about a Lorenz lobe change and a Roessler turn
    BRAIN_CASCADE,    ///< 0 .. 1: the conductor's events breed events (a Hawkes clock)
    // Phase 10b: the living landscape
    LIFE_MODE,        ///< LifeModeNames
    LIFE_DEPTH,       ///< 0 .. 1: how far the partials' loudness follows the field
    LIFE_RATE,        ///< 0 .. 1: how fast it lives
    LIFE_CHARACTER,   ///< 0 .. 1: Lenia's growth; the Turing field's way from spots to mazes
    LIFE_ZOOM,        ///< 0.25 .. 1: how much of the field's width the range spans
    // Phase 10c: partial sync
    SYNC_DEPTH,      ///< 0 .. 1: how far each partial's loudness follows its oscillator
    SYNC_COUPLING,   ///< 0 .. 1: how strongly the members of a harmonic pull each other into step
    SYNC_RATE,       ///< Hz: the oscillators' middle rate
    SYNC_SPREAD,     ///< 0 .. 1: how far their own rates scatter
    // Phase 10f: the ripple
    RIPPLE_DEPTH,   ///< 0 .. 1: how far the partials' loudness follows the wave
    RIPPLE_SPEED,   ///< octaves a second the wave runs along the spectrum
    RIPPLE_DECAY,   ///< s: how long a strike rings on the string
    // Phase 10g: the reservoir
    RESERVOIR_CHAOS,    ///< its coupling's spectral radius: below 1 it forgets, about 1 the edge of chaos
    RESERVOIR_MEMORY,   ///< s: how slowly its state moves
    RESERVOIR_INPUT,    ///< how strongly it hears the music
    // Phase 10h: the melt
    MELT_MIX,      ///< 0 .. 1: the melt on the near plane
    MELT_TIME,     ///< s: how long the melted sound lingers (to -60 dB)
    MELT_SPREAD,   ///< octaves: how far a held sound has spread after a second
    MELT_TO_FAR,   ///< 0 .. 1: its share into the far plane
    // Phase 11: morph and journeys (the player's: presets never set them, Presets.h keptByPresets)
    MORPH_ACTIVE,   ///< the sound is the blend of the morph's slots A and B (Morph.h)
    MORPH_POS,      ///< 0 (A) .. 1 (B): where between them
    PRESET_GLIDE,   ///< s: a preset chosen glides from the sound before it (0: it comes at once)
    // Phase 13: the partials' ratios
    STRETCH,        ///< cents an octave of the harmonics spans (1200: harmonic; Sethares' stretched timbres)
    SHIFT,          ///< Hz every partial is moved by (a frequency shifter: harmonic ratios become inharmonic)
    LIFE_SIDES,     ///< animated pictures: mirror wedges, spiral arms, grating lines
    LIFE_TURN,      ///< animated pictures: turns a minute (negative: the other way)
    COMB_DEPTH,     ///< 0 .. 1: the spectral comb's notches (0: none)
    COMB_DENSITY,   ///< notches an octave
    COMB_SWEEP,     ///< notches a second the comb travels (positive: up, an endless rise)
    COMB_STEREO,    ///< 0 .. 1: left and right combed apart
    // Phase 14: the preset map (the player's: presets never set them)
    MAP_ACTIVE,    ///< the sound is the blend of the presets round the map's cursor (PresetMap.h)
    MAP_X,         ///< the cursor across, 0 .. 1 (dark to bright)
    MAP_Y,         ///< the cursor up, 0 .. 1 (still to moving)
    MAP_RADIUS,    ///< the blend's width on the map
    MAP_GLIDE,     ///< s: how fast the sound follows the cursor
    ROUTE_SPEED,   ///< times: how fast a route is walked (its travel and hold times divided by it)
    // Phase 16 (14.10.2026): swarms and chimeras, turbulence, MPE
    SYNC_MODE,   ///< how the partials' oscillators couple: Global (Kuramoto), Chimera (to their neighbours), Swarm (swarmalators)
    SYNC_SWARM,   ///< -1 .. 1: Swarm: how a partial's place follows its phase (positive: like phases gather, negative: they part)
    SYNC_REACH,   ///< cents: Swarm: how far a partial wanders in pitch (and with it across the field) on its circle
    SYNC_LAG,     ///< 0 .. 1: Chimera: the coupling's phase lag (more: more of the cloud falls out of step)
    TURBULENCE,   ///< 0 .. 1: the energy cascade: a loud harmonic hands its energy upwards, the louder the more
    TURBULENCE_TIME,   ///< s: how long the cascade takes to settle back when the sound grows soft
    MPE_ON,            ///< MPE: notes on channels 2 .. 16 carry their own bend, pressure and slide (the lower zone)
    MPE_BEND,   ///< semitones: a member channel's full bend (48 by the MPE standard; an RPN 0 from the controller sets it)
    MPE_PRESS,         ///< what a note's pressure moves (MpeTargetNames)
    MPE_PRESS_DEPTH,   ///< 0 .. 1: how far
    MPE_SLIDE,         ///< what a note's slide (CC 74 on its channel) moves
    MPE_SLIDE_DEPTH,   ///< 0 .. 1: how far
    // Phase 17 (14.10.2026): the growing hall, the partials' own time, the ripple's bend
    HALL_GROW,   ///< -1 .. 1: the modal hall's memory of what it heard: positive, those pitches ring longer (growth); negative, shorter (erosion)
    HALL_MEMORY,   ///< s: how slowly the hall's memory forms (a minute .. four hours)
    TIME_BEND,   ///< -1 .. 1: each partial's clock (its drift, shimmer, sync) slower where the spectrum is dense (positive) or faster
    RIPPLE_BEND,   ///< 0 .. 1: the ripple's speed falls where the spectrum is loud: its waves linger at the notes
    // Phase 19 (14.10.2026)
    BRAIN_LIGHT,   ///< 0 .. 1: the conductors prefer notes whose harmonics fall where the landscape (or the living field) is bright
    BRAIN_FOUNDATION,   ///< 0 .. 1: a voice on the conductor's root in the octave from C2 (or Lowest, when lower), held while the root stands (its level; 0 none)
    // Phase 26 (15.10.2026): the harmonics' own lives, grains, the horizon, breathers
    OVERBLOW,   ///< 0 .. 1: the breath: a note's registers (1, 2, 3, 4 times its fundamental) compete for it, a stronger one jumps up
    OVERBLOW_TIME,    ///< s: how long a register takes to take over
    SEASONS,          ///< 0 .. 1: five families of harmonics take turns, one after another, in a slow cycle
    SEASON_TIME,      ///< s: the whole cycle through the five (a year)
    AVALANCHE,        ///< 0 .. 1: a rice pile on every note's harmonics: avalanches of every size run up them as glints
    AVALANCHE_RATE,   ///< grains a second falling on the pile (how restless it is)
    AVALANCHE_GLOW,   ///< s: how long a harmonic glows after it toppled
    SUBHARMONICS,     ///< 0 .. 1: a loud note pumps clouds at half (and a quarter) its harmonics: the period doubles
    SUB_TIME,         ///< s: how fast the subharmonics grow and fade
    HORIZON,          ///< 0 .. 1: the harmonics bend towards a horizon they never pass, crowding below it
    HORIZON_HEIGHT,   ///< octaves: the horizon over the fundamental
    GRAIN,            ///< 0 .. 1: the share of the partials that sound in grains, each born at a new place
    GRAIN_LENGTH,     ///< s: a grain's mean length
    GRAIN_DENSITY,    ///< 0 .. 1: the share of its time a grain partial sounds
    RIPPLE_BREATHERS,   ///< 0 .. 1: the ripple's string becomes a sine-Gordon chain: a note's strike stays as a breather that pulses where it fell
    // Phase 27 (16.10.2026): Talbot timbres, the phantom layer, the chime field, the dream, the premonition
    TALBOT,   ///< 0 .. 1: the first 24 harmonics' amplitudes evolve as a wave function: they fall apart and come back
    TALBOT_MODE,   ///< Revival (the timbre comes back whole after a period) or Bloch (its brightness swings up and down)
    TALBOT_TIME,   ///< s: the period
    PHANTOM,   ///< 0 .. 1: a noise the same in both ears but where pitches turn the phase between them: heard only in the head (headphones)
    PHANTOM_LINE,    ///< what the phantom layer carries: a melody of its own, the echo of notes gone, the chord
    PHANTOM_WIDTH,   ///< octaves over which the ears' phase turns round a pitch
    CHIMES,          ///< 0 .. 1: 2048 tuned chimes over the field, rung by a simulated wind
    CHIME_KIND,      ///< tubes, glass, bamboo, bells
    CHIME_PITCH,     ///< octaves: the chimes' register up or down
    CHIME_DECAY,     ///< s: how long a chime rings
    WIND,            ///< 0 .. 1: the wind's strength
    GUSTS,           ///< 0 .. 1: how much it gusts
    DREAM,         ///< 0 .. 1: the instrument remembers what it heard and dreams it, far off, wandering by association
    DREAM_FOCUS,   ///< 0 .. 1: a blend of memories .. one memory at a time
    DREAM_DRIFT,   ///< 0 .. 1: how restlessly the dream wanders
    DREAM_DEPTH,   ///< 0 .. 1: how far off the dream is heard
    PREMONITION,   ///< 0 .. 1: the conductor's notes are heard coming -- a swell like a reverb backwards into their start
    PREMONITION_TIME,   ///< s: how long before a note its premonition begins
    // Phase 28 (16.10.2026): the weather in the wind, the bloom, the swarmalators, the physarum
    RAIN,             ///< 0 .. 1: rain falls over the field where the showers the wind carries are
    RAIN_RATE,        ///< drops a second over the whole field
    RAIN_SURFACE,     ///< what it falls on: water, leaves, stone, a tin roof
    RAIN_SHOWERS,     ///< 0 .. 1: how patchy the rain is (showers passing with the wind)
    RAIN_TUNE,        ///< 0 .. 1: the share of the bubbles and the roof's ring on the scale's pitches
    LEAVES,           ///< 0 .. 1: leaves rustle where the wind blows
    FOLIAGE,          ///< broad leaves, pine, grass, dry leaves
    SURF,             ///< 0 .. 1: a sea breaks on a shore along the field, its waves raised by the wind
    SURF_SWELL,       ///< s: the swell's period
    SHORE,            ///< sand, pebbles, rocks
    SURF_DISTANCE,    ///< 0 .. 1: how far off the shore is
    BLOOM,            ///< 0 .. 1: a large plate everything rings, its energy climbing into a shimmer
    BLOOM_SIZE,       ///< m: the plate's side
    BLOOM_DECAY,      ///< s: its ring at 200 Hz
    BLOOM_DRIVE,      ///< 0 .. 1: how readily its energy climbs
    BLOOM_GLIDE,      ///< -1 .. 1: its pitch with its energy (an opera gong's glide)
    SWARMAL,          ///< 0 .. 1: the partials ride swarmalators round the listener (place and pulse)
    SWARMAL_GATHER,   ///< J, -1 .. 1: like phases gather (positive) or part
    SWARMAL_SYNC,     ///< K, -1 .. 1: near ones fall into step (positive) or out of it
    SWARMAL_RATE,     ///< Hz: the pulse's rate
    SWARMAL_SPEED,    ///< the swarm's time against the clock's
    SWARMAL_SPREAD,   ///< 0 .. 1: the swarmalators' own rates scattered
    PHYSARUM_SHARE,   ///< 0 .. 1: the share of the partials that crawl (Field: Physarum)
    PHYSARUM_TRAIL,   ///< s: how long the physarum's trail lasts
    // Phase 29 (16.10.2026): the wave room, the topological halo, the quantum walk
    WAVE_ROOM,     ///< 0 .. 1: the voices sound in a room simulated as air (its modes, its standing waves)
    ROOM_SIZE,     ///< 0.3 .. 1: the room, from a chapel to a hall of 10 x 8 x 4.5 m
    ROOM_ABSORB,   ///< 0 .. 1: how much the walls take up
    ROOM_WALK,     ///< 0 .. 1: where the listener stands, from the front to the back
    HALO,   ///< 0 .. 1: a topological halo round the listener: struck where a note stands, it runs round one way only
    HALO_SPEED,       ///< s: once round
    HALO_TOPOLOGY,    ///< 0 .. 1: topological (the halo runs) .. trivial (it stays where it was struck)
    HALO_DISORDER,    ///< 0 .. 1: disorder inside the lattice (the rim runs on regardless)
    HALO_TURN,        ///< which way round: counter-clockwise, clockwise
    QUANTUM,          ///< 0 .. 1: the conductor's root moves as a quantum walk on the Tonnetz
    QUANTUM_TIME,     ///< s: between the walk's looks
    QUANTUM_SPREAD,   ///< 0 .. 1: how far the walk's wave spreads between them
    // Phase 30 (16.10.2026): the night -- the Aeolian harp, thunder, the singing ice
    HARP,               ///< 0 .. 1: an Aeolian harp in the wind: each string locks onto a harmonic and sings it
    HARP_PITCH,         ///< oct: the strings' tuning about 55 .. 110 Hz (to the chord)
    HARP_GAUGE,         ///< 0 .. 1: thin to thick strings (thin ones sing higher harmonics)
    THUNDER,            ///< 0 .. 1: a thunderstorm: how often lightning strikes
    THUNDER_DISTANCE,   ///< km: how far off the storm is
    THUNDER_RUMBLE,     ///< 0 .. 1: how tortuous and branched the channels are (a crack .. a long rumble)
    ICE,                ///< 0 .. 1: a frozen lake that sings as it cracks
    ICE_THICKNESS,      ///< m: the ice's thickness (thin ice sings longer)
    ICE_COLD,           ///< 0 .. 1: how fast the cold builds the stress (how often it cracks)
    STREAMS,            ///< how Streams parts the partials: odd and even, by three, registers, members
    STREAM_DEPTH,       ///< 0 .. 1: each stream's own slow vibrato and tremolo (0: off)
    STREAM_RATE,        ///< Hz: their rate
    FUSION,             ///< 0 .. 1: the streams apart (heard as several sounds) .. moving as one (fused)
    RISSET,             ///< 0 .. 1: the voices pulse in Risset's rhythm, which slows (or quickens) for ever
    RISSET_RATE,        ///< Hz: the pulse in the middle of its octaves
    RISSET_TIME,        ///< s: an octave of slowing
    RISSET_WAY,         ///< slowing, quickening
    NIGHT_ARC,          ///< 0 .. 1: the real night (the sun where you are) darkens, slows and widens the sound
    DREAM_JOURNAL,      ///< the dream's memories kept with the state, and back with it
    VR_HANDS,           ///< 0 .. 1: the headset's hands reach into the cloud (wind, halo, room, physarum)
    HEAD_TRACKING,      ///< the binaural cloud and the thunder stay put as the head turns (the headset)
    // Phase 31 (16.10.2026): the night chorus, the time mirror
    CHORUS,          ///< 0 .. 1: a night chorus of insects and frogs, each kind in its own band, calling in turns
    CHORUS_KIND,     ///< who calls: a meadow, the treetops, a pond, all of them
    CHORUS_WARMTH,   ///< degrees Celsius: the night's warmth (warm insects call faster: Dolbear's law)
    CHORUS_SYNC,     ///< 0 .. 1: how far a call heard resets a neighbour's clock (they fall into step, or into turns)
    CHORUS_TUNE,     ///< 0 .. 1: the callers' carriers pulled to the scale's pitches
    MIRROR,        ///< 0 .. 1: a time-reversal mirror round you in the wave room: what it heard runs back to its source
    MIRROR_TIME,   ///< s: the mirror's window (it listens for one, plays it back reversed during the next)
    COUNT          ///< number of parameters
};

constexpr int NumParams = static_cast<int>( ParamIdE::COUNT );   ///< rows of the table

/** @brief How a parameter's value is read. */
enum class ParamKindE
{
    FLOAT,   ///< continuous, with a skew
    INT,     ///< whole numbers, linear
    BOOL,    ///< 0 or 1
    CHOICE   ///< an index into `choices`
};

/** @brief One row of the parameter table. */
struct ParamDesc
{
    ParamIdE           m_id;           ///< the parameter; rows are ordered by it
    const char        *m_key;          ///< stable identifier (automation, presets)
    const char        *m_name;         ///< display name
    const char        *m_section;      ///< grouping for editors
    ParamKindE         m_kind;         ///< how the value is read
    float              m_min;          ///< the lowest value, in the parameter's unit
    float              m_max;          ///< the highest value
    float              m_def;          ///< the default
    float              m_skew;         ///< 1 = linear, < 1 = more resolution at the low end
    const char        *m_unit;         ///< "Hz", "s", "dB", "ct", "st" or ""
    const char *const *m_choices;      ///< ParamKind::Choice only
    int                m_numChoices;   ///< entries of `choices`
};

/** @brief The table, ordered by ParamId. @return NumParams rows */
const std::array<ParamDesc, NumParams> &paramTable();

/**
 * @brief The row of one parameter.
 * @param[in] id  the parameter
 * @return its description
 */
inline const ParamDesc &paramDesc( ParamIdE id )
{
    return paramTable()[static_cast<size_t>( id )];
}

/**
 * @brief Looks a parameter up by its key.
 * @param[in] key  the stable identifier
 * @return the row, or nullptr
 */
const ParamDesc *findParam( const char *key );

/**
 * @brief A value in a parameter's normalised domain: 0 .. 1 across its range, the skew applied as
 *        the hosts apply it ((v - min) / (max - min))^skew -- where modulation depths are measured, so the
 *        same depth means the same on a cutoff in hertz and on a mix.
 * @param[in] d  the row
 * @param[in] v  the value in its unit
 * @return 0 .. 1
 */
float toNorm( const ParamDesc &d, float v );

/**
 * @brief Back from the normalised domain.
 * @param[in] d  the row
 * @param[in] n  0 .. 1
 * @return the value in its unit
 */
float fromNorm( const ParamDesc &d, float n );

constexpr int NumScaleChoices = 13;    ///< entries of ScaleNames: eleven tables, the user slot and the timbre scale
constexpr int UserScaleIndex = 11;     ///< the Scala slot, the first choice that is not a table (Tuning.cpp checks it)
constexpr int TimbreScaleIndex = 12;   ///< the scale computed from the spectrum (Sethares)
extern const char *const ScaleNames[NumScaleChoices];   ///< "12-TET", "JI Major (Ptolemy)", ... (as in Noctuary)
constexpr int            NumLifeModes = 13;             ///< entries of LifeModeNames
extern const char *const LifeModeNames
    [NumLifeModes];   ///< "Off", "Lenia", "Turing", the animated pictures, "Fluid", "Islands" (Cloud.h LifeMode)
constexpr int            NumSyncModes = 3;              ///< entries of SyncModeNames
extern const char *const SyncModeNames[NumSyncModes];   ///< "Global", "Chimera", "Swarm" (Cloud.h SyncMode)
constexpr int            NumMpeTargets = 9;             ///< entries of MpeTargetNames
extern const char *const MpeTargetNames
    [NumMpeTargets];   ///< "Off", "Level", "Filter", "Brightness", "Width", "Nearer", "Shimmer", "Motion", "Breath"

/** @brief What a note's pressure or slide moves (MPE), in MpeTargetNames' order. */
enum class MpeTargetE : int
{
    OFF = 0,
    LEVEL,
    FILTER,
    BRIGHTNESS,
    WIDTH,
    NEARER,
    SHIMMER,
    MOTION,
    BREATH   ///< Overblow's breath and the subharmonics' pump (Phase 26)
};
constexpr int            NumTalbotModes = 2;                  ///< entries of TalbotModeNames
extern const char *const TalbotModeNames[NumTalbotModes];     ///< "Revival", "Bloch"
constexpr int            NumPhantomLines = 3;                 ///< entries of PhantomLineNames
extern const char *const PhantomLineNames[NumPhantomLines];   ///< "Melody", "Echo", "Chord"
constexpr int            NumChimeKinds = 4;                   ///< entries of ChimeKindNames
extern const char *const ChimeKindNames[NumChimeKinds];   ///< "Tubes", "Glass", "Bamboo", "Bells" (Cloud.h ChimeKind)
constexpr int            NumFieldModes = 7;               ///< entries of FieldModeNames
extern const char *const FieldModeNames
    [NumFieldModes];   ///< "Off", "Scale", "Swarm", "Scatter", "Flock", "Orbit", "Physarum" (Cloud.h FieldMode)
constexpr int            NumRainSurfaces = 4;                 ///< entries of RainSurfaceNames
extern const char *const RainSurfaceNames[NumRainSurfaces];   ///< "Water", "Leaves", "Stone", "Tin Roof"
constexpr int            NumFoliages = 4;                     ///< entries of FoliageNames
extern const char *const FoliageNames[NumFoliages];           ///< "Broadleaf", "Pine", "Grass", "Dry Leaves"
constexpr int            NumHaloTurns = 2;                    ///< entries of HaloTurnNames
extern const char *const HaloTurnNames[NumHaloTurns];         ///< "Counter-clockwise", "Clockwise"
constexpr int            NumStreams = 4;                      ///< entries of StreamsNames (StreamsE)
extern const char *const StreamsNames[NumStreams];            ///< "Odd & Even", "By Three", "Registers", "Members"
constexpr int            NumRissetWays = 2;                   ///< entries of RissetWayNames
extern const char *const RissetWayNames[NumRissetWays];       ///< "Slowing", "Quickening"
constexpr int            NumChorusKinds = 4;                  ///< entries of ChorusKindNames (ChorusKindE)
extern const char *const ChorusKindNames[NumChorusKinds];     ///< "Meadow", "Treetops", "Pond", "Summer Night"
constexpr int            NumShores = 3;                       ///< entries of ShoreNames
extern const char *const ShoreNames[NumShores];               ///< "Sand", "Pebbles", "Rocks"
constexpr int            NumFarModes = 4;                     ///< entries of FarModeNames
extern const char *const FarModeNames
    [NumFarModes];   ///< the far reverb's networks (Effects.h Reverb::setMode): "Classic", "Scattering", "Colourless", "Rotating"
extern const char *const RootNames[12];    ///< "C" .. "B"
extern const char *const KeyMapNames[2];   ///< "Snap to 12 keys", "Consecutive degrees"

constexpr int            NumMotions = 7;            ///< entries of MotionNames
extern const char *const MotionNames[NumMotions];   ///< None, Random, Wave, Pendulum, Breathe, Glissando, Aurora
extern const char *const
    FilterModelNames[];   ///< Off, LP 6, LP 12, LP 24, HP 12, BP 12, Notch, Peak, Formant (Cloud.h FilterModel)
constexpr int            NumLfoShapes = 7;              ///< entries of LfoShapeNames
extern const char *const LfoShapeNames[NumLfoShapes];   ///< Sine, Triangle, Ramp Up, Ramp Down, Square, Random, Steps
extern const char *const EnvShapeNames[];   ///< the factory shapes' names, for the envelopes' choice (Shapes.h)

}   // namespace additiv
