/**
 * @file Cloud.h
 * @brief The GPU partial cloud: every voice a few dozen numbers on the host, tens of thousands of
 *        partials on the GPU.
 *
 * A voice is a harmonic spectrum whose every harmonic is not one sine but a cloud of `members`
 * slightly detuned sines (the PADsynth idea, played live): the cloud's width in cents, how it
 * widens towards the top, and every partial's own slow pitch drift, amplitude shimmer and place in
 * the stereo field. The host only fills CloudVoice; the GPU expands it into partials each block
 * (update kernel), sums them sample by sample (render kernel) and folds the chunks into one
 * block of four channels -- near and far, left and right, the two planes of Noctuary's spatial
 * model -- (reduce kernel). The three kernels and the two copies around them are one CUDA graph,
 * so a block costs one launch.
 *
 * Every partial also has its own envelopes, on three lanes: amplitude, pitch and place in the
 * stereo field. A lane does not store 20,000 envelopes; it says how the partials share a bank of
 * loopable multi-segment shapes (or one landscape of time against frequency): which shape each
 * partial takes, how fast it runs, where in its loop it starts, how late it begins and how deep it
 * reaches -- each as a function of the partial's pitch, harmonic, place in its cloud, parity or
 * chance (MapSource). Envelopes are pure functions of time, evaluated once per block per partial;
 * the render kernel ramps between them.
 *
 * This header carries no CUDA types: the plugin and the tools include it from plain C++.
 * @author Rene Weller
 *
 * @par Compile Flags:
 *   @arg @c __CUDACC__ - set by nvcc: the helpers both sides use are compiled for the host and the GPU
 */
#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "additiv/Spectrum.h"

namespace additiv
{

constexpr int   EnvPoints = 16;          ///< breakpoints per envelope shape
constexpr int   EnvShapes = 64;          ///< shapes in the bank
constexpr int   LandMaxColumns = 512;    ///< frequency columns a landscape may have
constexpr int   LandMaxRows = 8192;      ///< time rows a landscape may have
constexpr float LandLowHz = 20.0f;       ///< a landscape's first column
constexpr float LandHighHz = 20000.0f;   ///< and its last; the columns between are spaced logarithmically

/** @brief What a partial's share of an envelope lane is read from: each source is 0 .. 1 per partial. */
enum class MapSourceE : std::int32_t
{
    NONE = 0,   ///< always 0: every partial the same
    PITCH,      ///< the partial's centre frequency, logarithmically from LandLowHz (0) to LandHighHz (1)
    HARMONIC,   ///< its harmonic within the voice: 0 for the fundamental, 1 for the highest
    MEMBER,     ///< its place in its harmonic's cloud: 0 the lowest member, 0.5 the centre, 1 the highest
    RANDOM,     ///< its own dice
    ODD_EVEN,   ///< 0 on odd harmonics, 1 on even ones
};
constexpr int MapSources = 6;   ///< how many MapSource values there are

/** @brief One breakpoint of an envelope shape. */
struct EnvPoint
{
    float m_time = 0.0f;    ///< seconds from the shape's start, rising from point to point
    float m_level = 0.0f;   ///< 0 .. 1 on the amplitude lane, -1 .. 1 on pitch and pan
    float m_curve = 0.0f;   ///< -1 .. 1, the bend of the segment arriving here (0 straight, + slow start, - fast start)
    float m_pad = 0.0f;     ///< keeps the point 16 bytes
};

/**
 * @brief A multi-segment envelope with an optional loop and a release tail.
 *
 * It runs from point 0; between loopStart and loopEnd it loops while the note is held (a loop
 * from the first to the last point makes it an LFO). At the note's release a shape with points
 * after loopEnd leaves its loop wherever it is and glides through those points; a looping shape
 * without such a tail keeps looping (the voice's own envelope fades it). Without a loop it holds
 * its last level.
 */
struct EnvShape
{
    EnvPoint     m_p[EnvPoints];     ///< the points, `count` of them
    std::int32_t m_count = 0;        ///< points in use, 0 .. EnvPoints (0 = a flat zero)
    std::int32_t m_loopStart = -1;   ///< first point of the loop, -1 = no loop
    std::int32_t m_loopEnd = -1;     ///< last point of the loop
    std::int32_t m_pad = 0;          ///< keeps the shape a multiple of 16 bytes
};

/**
 * @brief One envelope lane of a voice: how its partials share the shapes (or the landscape).
 *
 * Every per-partial value is a base plus a spread read from a MapSource. A lane with amount 0 is
 * skipped and costs nothing.
 */
struct EnvLane
{
    float m_amount = 0.0f;       ///< amplitude: depth 0 .. 1; pitch: cents; pan: share of the field 0 .. 1
    float m_rate = 1.0f;         ///< how fast the shapes run (1 = as written)
    float m_rateSpread = 0.0f;   ///< rate times (1 + rateSpread * source): a linear spread makes a pendulum wave
    float m_phase = 0.0f;        ///< loop cycles added times the source: a travelling wave (negative: the other way)
    float m_delay = 0.0f;        ///< seconds of delay times the source (negative: times 1 - source)
    float m_depthTilt = 0.0f;    ///< -1 .. 1: the amount only where the source is low (-1) or high (+1)
    float m_chaos = 0.0f;   ///< 0 .. 1: from lockstep to every partial for itself (random loop phase, rate +-1 octave)
    std::int32_t m_shape = 0;                       ///< first shape of the bank this lane uses
    std::int32_t m_shapeSpread = 1;                 ///< how many shapes from there the partials spread over
    std::int32_t m_landscape = 0;                   ///< 1 = read the landscape instead of a shape
    MapSourceE   m_shapeBy = MapSourceE::RANDOM;    ///< which source picks the shape
    MapSourceE   m_rateBy = MapSourceE::HARMONIC;   ///< which source spreads the rate
    MapSourceE   m_phaseBy = MapSourceE::PITCH;     ///< which source spreads the loop phase
    MapSourceE   m_delayBy = MapSourceE::PITCH;     ///< which source spreads the delay
    MapSourceE   m_depthBy = MapSourceE::PITCH;     ///< which source tilts the depth
    std::int32_t m_pad = 0;                         ///< keeps the lane 64 bytes
};

// Phase 27 (16.10.2026): the phantom layer, Talbot timbres, the chime field, the dream.
constexpr int PhantomTargets = 8;                          ///< pitches the phantom layer carries at once
constexpr int TalbotHarmonics = 24;                        ///< harmonics a Talbot timbre moves (1 .. 24)
constexpr int ChimeCount = 2048;                           ///< the chime field's chimes
constexpr int ChimeModes = 16;                             ///< modes per chime: eight, each a doublet a few cents apart
constexpr int ChimeModesTotal = ChimeCount * ChimeModes;   ///< every mode of the field
constexpr int ChimePitches = 64;                           ///< pitches the chimes may be tuned to
constexpr int WindW = 128;                                 ///< the wind's lattice from left to right
constexpr int WindH = 64;                                  ///< and from near to far
constexpr int DreamBands = 512;       ///< the dream's spectrum: bands over 20 Hz .. 20 kHz on the log axis
constexpr int DreamMemories = 4096;   ///< snapshots of the heard spectrum the dream keeps
constexpr int DreamRecall = 32;       ///< memories a recall blends at most (the nearest)

// Phase 28 (16.10.2026): the weather in the wind, the bloom, the swarmalators, the physarum.
constexpr int RainSlots = 16384;     ///< drops that may sound at once
constexpr int RainPitches = 64;      ///< pitches tuned rain may sing
constexpr int LeafClusters = 4096;   ///< the leaves' clusters over the field (64 across, 64 from near to far)
constexpr int SurfSegments = 128;    ///< the shore's stretches from left to right
constexpr int SurfWaves = 64;        ///< the sea's wave trains
constexpr int BloomModes = 32768;    ///< the bloom plate's modes
constexpr int BloomBands = 64;       ///< its bands for the cascade, on the log axis from 30 Hz to 18 kHz
constexpr int SwarmN = 1024;         ///< the swarmalators
constexpr int PhysW = 1024;          ///< the physarum's trail across the spectrum: 20 Hz .. 20 kHz on the log axis
constexpr int PhysH = 32;            ///< and across the field, left to right
constexpr int PhysFood = 128;        ///< its food: lines at the notes' harmonics

// Phase 29 (16.10.2026): the topological halo, the wave room.
constexpr int HaloRadius = 12;                        ///< the halo's lattice: a disc of this radius, in sites
constexpr int HaloSites = 441;                        ///< its sites (the whole points of the disc)
constexpr int HaloComponents = 2 * HaloSites;         ///< two orbitals a site (Qi, Wu and Zhang's model)
constexpr int HaloCarriers = 8;                       ///< pitches it carries at once, each its own lattice
constexpr int RoomDecim = 8;                          ///< the wave room runs at the sample rate over this
constexpr int RoomNX = 88;                            ///< its grid across (11.4 cm cells at 48 kHz: 10 m)
constexpr int RoomNY = 70;                            ///< from front to back (8 m)
constexpr int RoomNZ = 40;                            ///< from the floor up (4.5 m)
constexpr int RoomCells = RoomNX * RoomNY * RoomNZ;   ///< its cells
constexpr int RoomSources = 24;                       ///< voices that may sound in it at once

// Phase 30 (16.10.2026): the night -- the Aeolian harp, thunder, the singing ice.
constexpr int   HarpStrings = 12;         ///< the Aeolian harp's strings
constexpr int   HarpModes = 32;           ///< harmonics a string rings (a warp's lanes)
constexpr int   ThunderSegments = 4096;   ///< a lightning channel's segments: its main stroke and its branches
constexpr int   ThunderMain = 2048;       ///< of them the main stroke's
constexpr int   ThunderBranches = 4;      ///< branches off the main stroke
constexpr int   ThunderRing = 1 << 21;    ///< samples the thunder's ring holds an ear (43.7 s at 48 kHz)
constexpr int   ThunderChunks = 16;       ///< blocks a strike's segments are sent into the ring over
constexpr int   IceN = 32;                ///< the ice sheet's stress grid: IceN x IceN cells (12.5 m each)
constexpr float IceShare = 0.2f;          ///< the stress a slipping cell hands each of its neighbours (OFC's alpha)
constexpr int   IceChirps = 64;           ///< chirps that may sound at once

// Phase 31 (16.10.2026): the night chorus, the time mirror.
constexpr int ChorusCallers = 2048;   ///< the night chorus's callers (crickets, katydids, frogs ...)
constexpr int StarModes = 1024;       ///< Star (Phase 33): the star's modes at most
constexpr int ChorusSpecies = 8;      ///< their kinds
constexpr int ChorusPitches = 64;     ///< pitches the callers' carriers may be tuned to
constexpr int RoomMirrors = 64;       ///< the time mirror's cells round the listener in the wave room
constexpr int MirrorMax = 36864;      ///< samples at the room's rate a mirror cell records (6 s at 6 kHz)

/**
 * @brief How Streams parts a voice's partials into streams (Phase 30): each stream moves with a vibrato and a tremolo
 *        of its own, and the ear hears it as a sound of its own (Bregman's common fate).
 */
enum class StreamsE : std::int32_t
{
    ODD_EVEN = 0,   ///< the odd harmonics and the even ones (the subharmonics a third)
    THIRDS,         ///< the harmonics by their remainder over three (the subharmonics a fourth)
    REGISTERS,      ///< the registers: harmonics 1 .. 2, 3 .. 6, 7 .. 16, above
    MEMBERS,        ///< every member of a harmonic's cloud its own (the cloud parts into lines)
    COUNT           ///< number of ways
};

/** @brief What a GPU voice plays: a note, or one of the layers that are not notes (Phase 27). */
enum class VoiceKindE : std::int32_t
{
    NOTE = 0,   ///< a note: its harmonics' clouds and its air
    PHANTOM,    ///< the phantom layer: noise the same in both ears but where pitches turn the phase between them
    DREAM,      ///< the dream: a cloud over DreamBands bands, shaped by what the dream recalls
    HALO        ///< the topological halo (Phase 29): a lattice round the listener, every site a sine per carrier
};

constexpr int VoiceFilters = 4;    ///< filters a voice carries during a morph between different ones (the heaviest)
constexpr int FilterBins = 1024;   ///< the transported filter's table: 20 Hz .. 20 kHz on the log axis

/** @brief One voice as the host describes it; the engine fills the derived fields at launch(). */
struct CloudVoice
{
    float m_hz = 110.0f;     ///< fundamental, Hz
    float m_level = 0.05f;   ///< RMS the voice aims at with gain 1 (all partials together)
    float m_gain = 0.0f;     ///< the envelope at the end of the coming block; the partials ramp to it across the block
    float m_tilt = 1.2f;     ///< spectral tilt: harmonic h at h^-tilt
    float m_width = 25.0f;   ///< cents: the cloud's width (two standard deviations) around harmonic 1
    float m_widthSlope = 0.3f;     ///< how the width grows with the harmonic number: width * h^widthSlope
    float m_drift = 3.0f;          ///< cents each partial wanders on its own
    float m_driftRate = 0.1f;      ///< Hz, the speed of that wander
    float m_shimmer = 0.4f;        ///< 0 .. 1, each partial's own amplitude wander
    float m_shimmerRate = 0.15f;   ///< Hz, the speed of the shimmer
    float m_stereo = 1.0f;         ///< 0 .. 1, how far the partials scatter across the stereo field
    float m_time = 0.0f;         ///< seconds since the note began, the clock of every envelope (advanced by the engine)
    float m_releaseAt = -1.0f;   ///< the note's time of release in seconds, -1 while it is held
    int   m_partials = 20000;    ///< partials this voice renders, 1 .. the engine's partialsPerVoice
    std::uint32_t m_seed = 1;    ///< the note's dice: initial phases, member offsets, drift and shimmer rates, pans
    int           m_active = 0;    ///< 0 = silent: the GPU skips the voice entirely
    int           m_restart = 0;   ///< 1 = a new note: fresh phases, amplitudes from zero, time 0 (cleared by launch())
    EnvLane       m_amp;           ///< the amplitude lane
    EnvLane       m_pitch;         ///< the pitch lane, in cents
    EnvLane       m_pan;           ///< the stereo lane, added to each partial's place
    /**
     * @brief Where the voice stands, 0 at the ear .. 1 deep in the background (Noctuary's planes,
     *        after Robert Rich). Every partial splits between the near and the far output by
     *        cos/sin(d pi/2), loses 6 dB per unit and is darkened by the air (see airAbsorb).
     */
    float m_distance = 0.0f;
    float m_airAbsorb =
        1.0f;   ///< 0 .. 1: how far distance darkens -- at 1 a partial's corner falls 5 octaves per unit from 16 kHz
    int m_onset = 0;   ///< with restart: samples into the block at which the note begins (sample-accurate starts)
    // The spectrum (Noctuary's formula; the voice is normalised after it, so these colour, not level).
    float m_brightness = 1.0f;   ///< 0 .. 1: harmonics above 1 + b^2 * 200 fade out (raised cosine); 1 keeps them all
    float m_oddEven = 0.0f;      ///< -1 .. 1: positive thins the even harmonics, negative the odd ones above the first
    float m_inharmonic = 0.0f;   ///< the stiff string's B: harmonic h at h sqrt(1 + B h^2)
    float m_presenceDb = 0.0f;   ///< 0 .. 6 dB: a bell over 1.8 .. 5.6 kHz, only near the listener (times 1 - distance)
    float m_lowCut = 0.0f;       ///< Hz: partials below fall 12 dB/oct (0 = off)
    // The voice filter, read as a frequency response at every partial (FilterModel).
    int   m_filterModel = 0;    ///< FilterModel; 0 = off
    float m_cutoff = 2500.0f;   ///< Hz, after key tracking, envelope and drift (the engine's)
    float m_filterQ = 0.707f;   ///< the resonance as a Q
    float m_vowel = 0.0f;       ///< Formant: 0 .. 4, u o a e i
    // Air: a share of the partials as a band of noise around a multiple of the fundamental.
    float m_air = 0.0f;          ///< 0 .. 1: the band's level against the voice's
    float m_airCenter = 3.0f;    ///< the band's centre, times the fundamental
    float m_airWidth = 600.0f;   ///< its width in cents (two standard deviations)
    float m_airJitter = 8.0f;    ///< Hz: how fast its partials flicker in pitch and level -- what makes them noise
    // Filled by the engine at launch(), read by the GPU.
    int   m_harmonics = 0;          ///< harmonics below the cutoff, at most `partials`
    int   m_members = 0;            ///< partials per harmonic
    int   m_harmonicPartials = 0;   ///< partials of the harmonic clouds; the rest up to `partials` are air
    float m_scale = 0.0f;           ///< amplitude of harmonic 1's every member, so that the voice meets `level`
    float m_airScale = 0.0f;        ///< amplitude of every air partial
    float m_filterMix =
        1.0f;   ///< 0 .. 1: the filter's gain raised to this (a morph fades one filter out and another in)
    float m_stretch = 1.0f;      ///< harmonic h at h^stretch (partialRatio)
    float m_shift = 0.0f;        ///< Hz added to every partial (below 20 Hz they fade out)
    float m_turbulence = 0.0f;   ///< the energy cascade's drive for this note, per second (0: none; Turbulence)
    // The voice filters during a morph between different ones (Morph.h): with two or more, the
    // filter above is not read; the card transports them into each other (filterTransportKernel).
    int   m_filterCount = 0;                    ///< filters in the lists below (0 or 1: the filter above alone)
    float m_filterWeight[VoiceFilters] = {};    ///< each one's share
    float m_filterModels[VoiceFilters] = {};    ///< its model (FilterModel, as a number)
    float m_filterCutoffs[VoiceFilters] = {};   ///< its cutoff, Hz
    float m_filterQs[VoiceFilters] = {};        ///< its Q
    float m_filterVowels[VoiceFilters] = {};    ///< its vowel (Formant)
    // Phase 26: the harmonics' own lives, worked out by the engine every block (Engine::stepHarmonics)
    // and read by the update kernel at each harmonic. All at rest when `harmonicLife` is 0.
    int   m_harmonicLife = 0;                           ///< which of them act: 1 registers, 2 seasons, 4 avalanche glow
    float m_lifeMakeup = 1.0f;                          ///< the voice's amplitude make-up for them (its power kept)
    float m_register[4] = { 1.0f, 0.0f, 0.0f, 0.0f };   ///< Overblow: the registers' shares (1, 2, 3, 4 times f0)
    float m_season[SeasonFamilies] = { 1.0f, 1.0f, 1.0f, 1.0f, 1.0f };   ///< Seasons: each family's amplitude
    float m_glow[AvalancheSites] = {};   ///< Avalanche: harmonics 2 .. 49, the glow's amplitude added (0 none)
    int   m_subClouds = 0;               ///< Subharmonics: clouds reserved after the harmonics (0 or SubClouds)
    float m_sub[SubClouds] = {};         ///< their amplitudes, against harmonic 1's at rest
    float m_pumpKeep[4] = { 1.0f, 1.0f, 1.0f, 1.0f };   ///< harmonics 1, 3, 5, 7: the amplitude left them
    float m_horizon = 0.0f;        ///< 0 .. 1: the harmonics bent towards the horizon (horizonBend)
    float m_horizonRatio = 8.0f;   ///< the horizon over the fundamental
    float m_grain = 0.0f;          ///< 0 .. 1: the partials sounding as grains (0: continuous)
    float m_grainLength = 0.08f;   ///< s: a grain's mean length
    float m_grainDuty = 0.3f;      ///< 0 .. 1: the share of its time a partial sounds
    float m_grainNorm = 1.0f;      ///< the grains' make-up, so their power is the continuous one's
    // Phase 27
    VoiceKindE m_kind = VoiceKindE::NOTE;   ///< what the voice plays (a note, the phantom layer, the dream)
    int        m_phantomCount = 0;          ///< Phantom: pitches in phantomHz
    float      m_phantomWidth = 0.1f;       ///< Phantom: octaves over which the ears' phase turns once round a pitch
    float      m_phantomHz[PhantomTargets] = {};       ///< Phantom: the pitches heard only between the ears
    float      m_phantomWeight[PhantomTargets] = {};   ///< each one's share of the full turn (it fades in and out)
    float m_talbot[TalbotHarmonics] = {};   ///< Talbot: harmonics 1 .. 24, their amplitude factors (harmonicLife 8)
    // Phase 29
    float m_haloHz[HaloCarriers] = {};   ///< Halo: each carrier's pitch, Hz (0: silent)
    // Phase 30
    int   m_streams = 0;          ///< Streams: how the partials part into streams (StreamsE)
    float m_streamDepth = 0.0f;   ///< Streams: 0 .. 1, each stream's slow vibrato and tremolo (0: none)
    float m_streamRate = 2.0f;    ///< Streams: Hz, their rate
    float m_fusion = 0.0f;        ///< Streams: 0 each stream its own motion (heard apart) .. 1 all as one (fused)
    // Phase 32
    float m_floorHz = 0.0f;     ///< Hz: under it the voice's filter, comb, low cut and presence give way, coming back
                                ///< over the octave above (the foundation keeps its depth; 0: off)
    float m_beatHz = 0.0f;      ///< Hz: Pressure -- a share of the members this far above the rest at the fundamental
                                ///< (the harmonic times it above), so the floor beats slowly (0: off)
    float m_beatShare = 0.0f;   ///< 0 .. 0.5: that share of the members
    float m_floorGain = 1.0f;   ///< Floor: the amplitude under floorHz raised by (1 .. 8), over the octave above back
};

/** @brief The voice filter's models; on the GPU each is its analogue prototype's magnitude at the partial's frequency. */
enum class FilterModelE : int
{
    OFF = 0,   ///< no filter
    LP6,       ///< one-pole low pass
    LP12,      ///< two-pole low pass with resonance
    LP24,      ///< two of them in series
    HP12,      ///< two-pole high pass
    BP12,      ///< band pass, unity at the cutoff
    NOTCH,     ///< band stop
    PEAK,      ///< a resonant bell on a flat line
    FORMANT,   ///< three band passes on the formants of u-o-a-e-i; the cutoff picks the vowel
    COUNT      ///< number of models
};
constexpr int NumFilterModels = static_cast<int>( FilterModelE::COUNT );   ///< entries of FilterModelNames

constexpr int CloudChannels = 4;   ///< the voices' channels on the GPU: near left, near right, far left, far right
/**
 * @brief A finished frame: the four voice channels, the modal hall's left and right, the spectral echo's, the
 *        melt's, the chime field's four (near and far, left and right), the phantom layer's two (heard dry), the
  *        weather's four (near and far, left and right), the bloom's two, the wave room's two and the night's four
 *        (the harp, the thunder, the ice and the chorus; near and far, left and right).
 */
constexpr int CloudOutChannels = 28;
constexpr int ModalModes = 65536;   ///< the modal hall's resonances
constexpr int ModalTuned = 128;     ///< pitches the hall's tuned modes may sit on
constexpr int HallBins =
    1024;   ///< the hall's memory and the spectrum's density: bands over 20 Hz .. 20 kHz on the log axis (11.7 cents)

/**
 * @brief The modal hall's settings for the next block: the host's copy, sent up with the voices.
 *
 * A reverb made the way the instrument is made: ModalModes resonances, each a complex one-pole
 * (a damped sine) driven by the voices of the block -- near and far, left and right -- and heard
 * through a phase of its own in each ear. Its tail is a sum of decaying sines, as the voices are a
 * sum of sustained ones (Abel, Coffin and Spratt 2014, a modal architecture for artificial
 * reverberation). Where the modes stand is the instrument's choice: spread evenly over the range
 * they are a hall; a share `tune` of them on the pitches of the scale and the hall rings in key,
 * like the sympathetic strings of a viola d'amore. Every mode's weight is set so that white noise
 * comes out at the level it went in, whatever the decay; tuned modes are weighted down by how much
 * closer they stand together than spread ones, so turning `tune` colours the hall rather than
 * making it louder.
 */
struct ModalHall
{
    int   m_on = 0;           ///< 0: the hall rests -- silent, its modes cleared, almost no GPU time
    int   m_freeze = 0;       ///< 1: the modes ring on (about 24 minutes to -60 dB) with the input shut
    int   m_tunedCount = 0;   ///< pitches in `tuned`
    int   m_pad = 0;          ///< keeps what follows 16-byte aligned
    float m_decay = 8.0f;     ///< seconds to -60 dB at and below 250 Hz
    float m_damp = 0.35f;     ///< 0 .. 1: every octave above 250 Hz dies 2^damp times faster
    float m_tune = 0.5f;      ///< 0 .. 1: the share of the modes on the scale's pitches
    float m_spread = 6.0f;    ///< cents a tuned mode may lie from its pitch (a triangle)
    float m_drift = 4.0f;     ///< cents of every mode's slow wander (0.02 .. 0.12 Hz)
    float m_width = 1.0f;     ///< 0: both ears hear the same phase of every mode (mono) .. 1: each its own
    float m_low = 80.0f;      ///< Hz: the lowest mode of the spread ones
    float m_time =
        0.0f;   ///< seconds: the drift's clock, wrapped at 1000 s (every drift rate is a whole number of cycles in 1000 s)
    float m_breath =
        0.0f;   ///< 0 .. 1: every mode's decay wanders by up to an octave either way, slowly, each on its own
    float m_paint = 0.0f;       ///< 0 .. 1: every mode heard through the landscape at its frequency (the painted hall)
    float m_paintNorm = 1.0f;   ///< the landscape's make-up (one over its RMS), so the painted hall keeps its level
    float m_paintTime = 0.0f;   ///< seconds into the landscape's loop, for the painting
    float m_grow =
        0.0f;   ///< -1 .. 1: how far the hall's memory lengthens (growth) or shortens (erosion) its modes' decays
    float m_memoryCoef = 0.0f;    ///< the memory's step this block, 1 - exp(-dt / Memory) (0: it does not learn)
    float m_densityCoef = 0.0f;   ///< the spectrum's density followed this block, 1 - exp(-dt / 0.3 s)
    float m_redshift = 0.0f;      ///< 0 .. 1: every mode glides down, reborn at the top (Phase 33: the redshift hall)
    float m_redPhase = 0.0f;      ///< 0 .. 1: the glide's clock (time over an octave's time and the span, wrapped)
    float m_redSpan = 3.0f;       ///< octaves a mode glides through before it is reborn
    float m_rpad = 0.0f;          ///< keeps the tuned pitches 16-byte aligned
    int   m_listen =
        0;   ///< 1: the partials' amplitudes are gathered over the octaves this block (for the memory or the partials' time)
    float m_tuned[ModalTuned] = {};   ///< Hz: the scale's pitches between `low` and 16 kHz
};
static_assert( sizeof( ModalHall ) % 16 == 0, "ModalHall rides in the block header" );

/**
 * @brief The binaural partial cloud: every partial at its own place around the head.
 *
 * Because every partial is a sine, what a head does to it is one complex gain per ear at its
 * frequency: the head's shadow (Brown and Duda 1998, a pole and a zero whose zero moves with the
 * angle of incidence), the pinna's five reflections by azimuth and elevation (the same model), and
 * the time the sound takes round the head (Woodworth) as a phase. No convolution: the update kernel
 * works out the two gains per partial per block, the render kernel adds a cosine to the sine it
 * already has. Only the near plane is placed so; the far plane goes into the far reverb, which is
 * diffuse anyway. The two ears' power is kept where the stereo pan law had it, so switching changes
 * the place, not the level.
 */
struct Binaural
{
    int   m_on = 0;   ///< 1: the near plane binaural (for headphones)
    float m_span =
        0.5f;   ///< 0: the partials' places spread over the front half (+-90 degrees) .. 1: all round (+-180)
    float m_height = 0.3f;   ///< 0 .. 1: elevations spread over up to +-60 degrees
    float m_angle = 0.0f;    ///< radians: the whole cloud turned (the orbit), 0 in front, positive to the right
};
static_assert( sizeof( Binaural ) % 16 == 0, "Binaural rides in the block header" );

/**
 * @brief The spectral echo: a delay line for every frequency band, on the GPU.
 *
 * The voices (near and far) are taken apart every block by a 1024-point FFT (a Hann window, the
 * block as the hop), every band of each ear goes through a delay line of its own with feedback,
 * and the bands are put back together and overlapped (output: channels 6 and 7, 896
 * samples after the input at 128 a block). The bands' times follow `tilt` (positive: the highs
 * later) and `scatter` (each band a time of its own), the feedback falls towards the top by
 * `damp`. Echoes then come back as cascades rather than as copies.
 */
struct SpectralEcho
{
    int   m_on = 0;            ///< 1: running (0: silent, nothing computed)
    int   m_clear = 0;         ///< 1 for one block: the lines emptied first (the engine asks when it switches on)
    int   m_pad0 = 0;          ///< keeps the floats 16-byte aligned
    int   m_pad1 = 0;          ///< likewise
    float m_frames = 187.0f;   ///< the bands' middle delay, in blocks
    float m_tilt = 0.0f;       ///< -1 .. 1: the delay times 2^(2 tilt (place - 0.5)), place 0 at 20 Hz .. 1 at 20 kHz
    float m_scatter = 0.0f;    ///< 0 .. 1: each band's time scattered by up to +-75 %
    float m_feedback = 0.0f;   ///< 0 .. 0.95
    float m_damp = 0.0f;       ///< 0 .. 1: the feedback times (1 - damp place)
    float m_pad2 = 0.0f;       ///< keeps the table 16-byte aligned
    float m_pad3 = 0.0f;       ///< likewise
    float m_pad4 = 0.0f;       ///< likewise
};
static_assert( sizeof( SpectralEcho ) % 16 == 0, "SpectralEcho rides in the block header" );

constexpr int MeltN = 4096;    ///< the melt's FFT size (11.7 Hz bands at 48 kHz)
constexpr int MeltHop = 512;   ///< samples between its frames

/**
 * @brief The melt: a spectral blur on the GPU (output: channels 8 and 9, MeltN - block samples late).
 *
 * Every MeltHop samples the voices (near and far) are taken apart by a MeltN-point FFT. Each band's
 * energy is held -- the new frame mixed in by 1 - `keep` -- and the held spectrum diffuses across the
 * bands, a little every frame (`widen`, in octaves on a log axis), so what has been held longest is
 * spread widest: a chord melts into the space between its notes. A phase vocoder sings it again:
 * each band follows the frequency it hears (its partial's, not the band's centre), is pulled back
 * to its own centre by `pull` as it melts, and is detuned a little, otherwise in each ear.
 */
struct Melt
{
    int   m_on = 0;          ///< 1: running (0: silent, nothing computed)
    int   m_clear = 0;       ///< 1 for one block: everything emptied first (the engine asks when it switches on)
    int   m_pad0 = 0;        ///< keeps the floats 16-byte aligned
    int   m_pad1 = 0;        ///< likewise
    float m_keep = 0.99f;    ///< per frame: the share of the held energy kept
    float m_widen = 0.0f;    ///< per frame: the blur's standard deviation in bands, per band index
    float m_pull = 0.0f;     ///< per frame: each band's frequency pulled towards its centre by this share
    float m_detune = 0.0f;   ///< radians a sample: each band's offset at most (its own in each ear)
};
static_assert( sizeof( Melt ) % 16 == 0, "Melt rides in the block header" );

constexpr int SnapStride = 8;    ///< every eighth partial of every voice is in the picture's snapshot
constexpr int LifeW = 256;       ///< the living field's width, cells (across the spectrum)
constexpr int LifeH = 128;       ///< its height, cells (across a cloud's members)
constexpr int LifeRadius = 10;   ///< Lenia's ring kernel radius, cells
constexpr int LifeJets = 16;     ///< the Fluid's jets: two for each of the eight loudest notes
constexpr int IsleW = 64;        ///< Islands: its grid across (each cell four of the field's)
constexpr int IsleH = 32;        ///< Islands: its grid along

/**
 * @brief The spectral comb: notches evenly spaced per octave, read at every partial's frequency
 *        (a filter only an additive instrument has: no delay line, any spacing, any depth). Its
 *        phase runs on, so the notches travel through the spectrum without end -- a phaser that
 *        rises or falls forever -- and each partial reads it a little further on by its place in
 *        the stereo field, so left and right are combed apart.
 */
struct SpectralComb
{
    float m_depth = 0.0f;     ///< 0 .. 1: the notches' depth (0: no comb)
    float m_density = 2.0f;   ///< notches an octave
    float m_phase = 0.0f;     ///< where the notches stand, in notches (the engine runs it on)
    float m_stereo = 0.0f;    ///< 0 .. 1: half a notch apart from one side of the field to the other
    float m_makeup = 1.0f;    ///< one over the root of the comb's mean square
    float m_pad0 = 0.0f;      ///< keeps the struct a multiple of 16 bytes
    float m_pad1 = 0.0f;      ///< likewise
    float m_pad2 = 0.0f;      ///< likewise
};
static_assert( sizeof( SpectralComb ) % 16 == 0, "SpectralComb rides in the block header" );

/**
 * @brief Turbulence: the energy cascade of a struck gong or tam-tam, along every voice's harmonics.
 *
 * In a plate or a shell the modes are not alone: struck hard they hand their energy on to modes
 * above them, the more the louder they are -- the bloom of a tam-tam a second after the strike
 * (Bilbao's nonlinear plates). Here every harmonic of a voice has a power that starts where its
 * spectrum puts it; each block it hands drive p^2 of it on -- the drive by the cube of the note's and the instrument's loudness -- (seven tenths to the next harmonic,
 * three to its octave), so only the loud ones cascade, and it settles back to the spectrum's own
 * shape over `settle` seconds, the higher harmonics faster. The power moves, it is not made: a soft
 * chord stays pure, a dense loud swell breaks slowly into a shimmering glow above itself without
 * growing louder.
 */
struct Turbulence
{
    float m_on = 0.0f;       ///< 1 when any voice cascades (else nothing is computed)
    float m_settle = 6.0f;   ///< s: how long the harmonics take to settle back
    float m_pad0 = 0.0f;     ///< keeps the struct a multiple of 16 bytes
    float m_pad1 = 0.0f;     ///< likewise
};
static_assert( sizeof( Turbulence ) % 16 == 0, "Turbulence rides in the block header" );

/** @brief The chimes' bodies: the ChimeKind parameter's choices. */
enum class ChimeKindE : std::int32_t
{
    TUBES = 0,   ///< metal tubes: a free bar's modes (1, 2.76, 5.40, 8.93 ..)
    GLASS,       ///< glass bowls: 1, 2.32, 4.25, 6.63 ..
    BAMBOO,      ///< bamboo: a tube's modes, short and woody
    BELLS,       ///< small bells: hum, prime, tierce, quint, nominal ..
    COUNT        ///< number of kinds
};

/**
 * @brief The chime field (Phase 27): ChimeCount tuned chimes over the field -- left to right, near to far -- in a
 *        wind. The wind is a lattice Boltzmann flow (D2Q9) on a WindW x WindH torus, pushed to the right, shedding
 *        vortex streets from three posts; every chime's clapper is a pendulum swung by the wind's sideways and gusting
 *        parts, and where it strikes the chime's modes ring (complex one-poles, as the modal hall's). So gusts cross
 *        the field and the chimes ring in waves, from one side to the other.
 */
struct ChimeField
{
    int   m_on = 0;           ///< 1: the field sounds and its wind blows (0: nothing computed, everything kept)
    int   m_kind = 0;         ///< ChimeKindE
    int   m_pitchCount = 0;   ///< pitches in `pitch`
    int   m_reset = 0;        ///< 1 for one block: the wind and the clappers begin again from rest
    float m_level = 0.0f;     ///< the field's amplitude
    float m_force = 0.0f;     ///< the wind's push this block, lattice units (its gusts in it)
    float m_decay = 4.0f;     ///< s: the lowest mode's ring to -60 dB
    float m_kick = 30.0f;     ///< how hard the wind swings the clappers
    float m_pitch[ChimePitches] = {};   ///< Hz: the pitches the chimes are tuned to, each chime one of them
};
static_assert( sizeof( ChimeField ) % 16 == 0, "ChimeField rides in the block header" );

/**
 * @brief The dream (Phase 27): every few seconds the spectrum the instrument heard, gathered over DreamBands bands,
 *        is kept as a memory if it is new; a dream state wanders among the memories by association -- a modern
 *        Hopfield network (Ramsauer et al. 2020): the nearest memories to the state, a little noise in the query,
 *        blended by a softmax of their likeness -- and a voice of the kind DREAM sings it, far off.
 */
struct Dream
{
    int   m_on = 0;             ///< 1: the dream remembers and recalls (0: nothing computed, the memories kept)
    int   m_store = 0;          ///< 1 for one block: what was gathered is kept as a memory if it is new
    int   m_clear = 0;          ///< 1 for one block: every memory forgotten
    int   m_pad0 = 0;           ///< keeps the floats 16-byte aligned
    float m_focus = 20.0f;      ///< the recall's sharpness (the softmax's beta): low blends, high recalls one memory
    float m_ease = 0.002f;      ///< a block's share of the way from the state to what it recalls
    float m_noise = 0.1f;       ///< the wander in the query (against the state's length 1)
    float m_novelty = 0.985f;   ///< a snapshot this like a memory (cosine) is not kept
};
static_assert( sizeof( Dream ) % 16 == 0, "Dream rides in the block header" );

/** @brief What the rain falls on: the RainSurface parameter's choices. */
enum class RainSurfaceE : std::int32_t
{
    WATER = 0,   ///< a pond: a drop traps a bubble that sings (Minnaert's resonance), rising as it rises
    LEAVES,      ///< leaves: a short tick, a leaf's own ring
    STONE,       ///< stone: a click and a glint
    ROOF,        ///< a tin roof: each drop rings the sheet's modes
    COUNT        ///< number of surfaces
};

/** @brief What the wind moves: the Foliage parameter's choices. */
enum class FoliageE : std::int32_t
{
    BROADLEAF = 0,   ///< broad leaves: a soft rustle
    PINE,            ///< needles: a high, smooth hiss
    GRASS,           ///< tall grass: low and soft
    DRY,             ///< dry leaves: a sparse crackle
    COUNT            ///< number of kinds
};

/** @brief What the sea breaks on: the Shore parameter's choices. */
enum class ShoreE : std::int32_t
{
    SAND = 0,   ///< sand: the foam's hiss
    PEBBLES,    ///< pebbles: the backwash crackles through the stones
    ROCKS,      ///< rocks: a boom as each wave breaks
    COUNT       ///< number of shores
};

/**
 * @brief The weather (Phase 28): rain, leaves and a sea, all in the chime field's wind (ChimeField).
 *
 * Rain: drops fall over the field -- left to right, near to far -- where the showers are: a density carried
 * across by the wind (advected on its lattice), fed at the windward side by slow randoms, so showers pass. On water
 * a drop traps a bubble whose ring is Minnaert's (3.26 kHz over its radius in mm, damped as van den Doel measured,
 * rising as it rises); on leaves it ticks, on stone it clicks, on a tin roof it rings the sheet's modes. Tuned, a
 * share of the bubbles and the roof's modes sing the scale's pitches.
 * Leaves: clusters over the field rustle as the wind where they stand blows -- harder in the gusts --, every leaf
 * a click through its own small resonance.
 * The sea: SurfWaves trains of a swell, their heights by the wind, run onto a shore along the field; where a crest
 * rises past its break it foams, and the foam hisses, the backwash crackles, rocks boom.
 */
struct Weather
{
    int   m_wind = 0;                  ///< 1: the wind blows (for the chimes or the weather; its lattice computed)
    int   m_rainOn = 0;                ///< 1: the rain falls
    int   m_leavesOn = 0;              ///< 1: the leaves rustle
    int   m_surfOn = 0;                ///< 1: the sea breaks
    int   m_surface = 0;               ///< RainSurfaceE
    int   m_foliage = 0;               ///< FoliageE
    int   m_shore = 0;                 ///< ShoreE
    int   m_pitchCount = 0;            ///< pitches in `pitch`
    int   m_reset = 0;                 ///< 1 for one block: the showers, the drops and the sea begin again
    int   m_rainFresh = 0;             ///< 1 for one block: the showers begin even (the rain turned on)
    int   m_windFresh = 0;             ///< 1 for one block: the wind begins from calm air (it began to blow)
    int   m_pad2 = 0;                  ///< keeps the floats 16-byte aligned
    float m_rainLevel = 0.0f;          ///< the rain's amplitude
    float m_rainRate = 600.0f;         ///< drops a second over the whole field
    float m_showers = 0.5f;            ///< 0 .. 1: how patchy the rain is (0 even, 1 passing showers)
    float m_tune = 0.0f;               ///< 0 .. 1: the share of the bubbles and the roof's modes on the scale's pitches
    float m_leavesLevel = 0.0f;        ///< the leaves' amplitude
    float m_windRef = 0.02f;           ///< the wind's speed (lattice units) at which the leaves rustle fully
    float m_surfLevel = 0.0f;          ///< the sea's amplitude
    float m_surfHeight = 1.0f;         ///< the waves' height against the calm swell's (the wind raises it)
    float m_surfPeriod = 10.0f;        ///< s: the swell's period
    float m_surfDistance = 0.6f;       ///< 0 .. 1: how far off the shore is
    float m_time = 0.0f;               ///< s since the sea began (its clock)
    float m_advect = 0.15f;            ///< the showers' drift against the wind's speed
    float m_shower[4] = {};            ///< the windward feed across the field (near .. far): slow randoms, -1 .. 1
    float m_pitch[RainPitches] = {};   ///< Hz: the pitches tuned drops may sing
    int   m_handX[2] = { 0, 0 };       ///< the headset's hands in the wind (Phase 30): each one's cell across
    int   m_handY[2] = { 0, 0 };       ///< and from near to far
    int   m_handR[2] = { 0, 0 };       ///< its radius, cells (0: no hand); the wind flows round it as round a post
    int   m_pad3[2] = { 0, 0 };        ///< keeps the struct 16-byte aligned
};
static_assert( sizeof( Weather ) % 16 == 0, "Weather rides in the block header" );

/**
 * @brief The bloom (Phase 28): a large plate of BloomModes modes (a tam-tam, a thunder sheet) that everything in
 *        the field rings -- the voices, the chimes, the weather -- and whose energy climbs.
 *
 * A thin plate's modes stand evenly in hertz (Weyl), a mode every 3.24 / area Hz for bronze 1.5 mm thick. Linear,
 * it would be a hall; a plate struck hard is not (von Karman's equations): its modes trade energy, and the energy
 * runs up to the high ones -- the shimmer that blooms out of a tam-tam's stroke a moment after it -- the faster the
 * louder (weak wave turbulence: Duering, Josserand and Rica 2006). Here the trade is the kinetic picture of that
 * cascade: per band each block a flux upward, the energy over the band's modes squared, taken from a band as
 * extra damping and given to the next as noise of the same power -- every mode a complex one-pole, the sum of the
 * energies kept. And as in Berger's plate the stretching raises the pitch with the energy, the low modes most:
 * the glide of an opera gong.
 */
struct Bloom
{
    int   m_on = 0;            ///< 1: the plate rings (0: nothing computed, its modes cleared)
    int   m_pad0 = 0;          ///< keeps the floats 16-byte aligned
    int   m_pad1 = 0;          ///< likewise
    int   m_pad2 = 0;          ///< likewise
    float m_level = 0.0f;      ///< the plate's amplitude out
    float m_spacing = 1.65f;   ///< Hz between its modes (the plate's size)
    float m_decay = 8.0f;      ///< s: a mode's ring at 200 Hz (higher ones shorter, as the square root)
    float m_drive = 0.5f;      ///< 0 .. 1: how readily the energy climbs
    float m_glide = 0.3f;      ///< -1 .. 1: the pitch with the energy (up when positive)
    float m_width = 1.0f;      ///< 0 .. 1: how far the ears' phases differ
    float m_pad3 = 0.0f;       ///< keeps the struct 16-byte aligned
    float m_pad4 = 0.0f;       ///< likewise
};
static_assert( sizeof( Bloom ) % 16 == 0, "Bloom rides in the block header" );

/**
 * @brief The swarmalators (Phase 28; O'Keeffe, Ha and Strogatz 2017): SwarmN oscillators that swarm in the plane
 *        around the listener while they synchronise, place and phase pulling each other -- like phases gather
 *        (J), near ones fall into step (K). Every partial rides one of them: its place left to right and near to
 *        far is the swarmalator's, its loudness pulses with its phase. Their states: gathered and pulsing as one;
 *        a disc of phases at rest; a ring whose phase turns round it, so the pulse circles the listener; that ring
 *        broken into turning clusters; and, past a point, the ring itself turning.
 */
struct SwarmSpace
{
    int   m_on = 0;          ///< 1: the swarm moves and the partials ride it
    int   m_reset = 0;       ///< 1 for one block: the swarm scattered afresh
    int   m_pad0 = 0;        ///< keeps the floats 16-byte aligned
    int   m_pad1 = 0;        ///< likewise
    float m_depth = 0.0f;    ///< 0 .. 1: how far the partials follow it (place and pulse)
    float m_j = 0.5f;        ///< J: like phases gather (positive) or part
    float m_k = 0.0f;        ///< K: near ones fall into step (positive) or out of it
    float m_rate = 0.3f;     ///< Hz: the pulse's middle rate
    float m_speed = 1.0f;    ///< the swarm's time against the clock's
    float m_spread = 0.3f;   ///< the oscillators' own rates scattered, against the rate
    float m_pad2 = 0.0f;     ///< keeps the struct 16-byte aligned
    float m_pad3 = 0.0f;     ///< likewise
};
static_assert( sizeof( SwarmSpace ) % 16 == 0, "SwarmSpace rides in the block header" );

/**
 * @brief The physarum (Phase 28; Jones 2010, after the slime mould Physarum polycephalum): a share of the partials
 *        crawl as its agents over a trail across the spectrum and the field. Each senses the trail ahead, ahead
 *        left and ahead right, turns towards the most, moves and leaves trail; the trail spreads and fades, and the
 *        notes' harmonics feed it. So veins grow between the harmonics -- partials gliding along them from one to
 *        the next -- and as the chord changes the network moves to the new food and the old veins starve.
 */
struct Physarum
{
    int   m_on = 0;                      ///< 1: the field is the physarum (FieldModeE::PHYSARUM)
    int   m_fresh = 0;                   ///< 1 when the last block was none: every agent starts where its partial is
    int   m_count = 0;                   ///< food lines in `food`
    int   m_pad0 = 0;                    ///< keeps the floats 16-byte aligned
    float m_share = 0.3f;                ///< 0 .. 1: the share of the partials that crawl
    float m_speed = 0.2f;                ///< trail cells an agent moves a block
    float m_sensor = 5.0f;               ///< cells ahead it senses
    float m_spring = 0.0f;               ///< per second: how fast an agent is drawn back to its partial's own place
    float m_keep = 0.99f;                ///< the trail's share kept a block (its fading)
    float m_spread = 0.15f;              ///< the share of the trail that spreads to the neighbours a block
    float m_pad1 = 0.0f;                 ///< keeps the arrays 16-byte aligned
    float m_pad2 = 0.0f;                 ///< likewise
    float m_food[PhysFood] = {};         ///< food lines: trail cells across (0 .. PhysW)
    float m_foodWeight[PhysFood] = {};   ///< each one's weight
};
static_assert( sizeof( Physarum ) % 16 == 0, "Physarum rides in the block header" );

/**
 * @brief The topological halo (Phase 29): a Chern insulator of sound round the listener -- Qi, Wu and Zhang's model
 *        on a disc of HaloSites sites, two orbitals each, its hopping turning the phase so that its bulk is gapped and
 *        its rim carries states that run one way only. Every carrier (a pitch of the chord) has a lattice of its
 *        own: a note drives the rim where it stands, at the gap's middle, and what it starts runs round the
 *        listener -- the same way whatever is in the way (disorder inside cannot turn it back: topological
 *        protection). Made trivial (Topology), the gap holds no states and the sound stays where it was struck.
 *        Evolved a block at a time (fourth-order Runge-Kutta) in the frame turning with the carrier; a voice of the
 *        kind HALO sings it, each site a sine per carrier, its complex amplitude the site's.
 */
struct Halo
{
    int   m_on = 0;                     ///< 1: the lattices evolve (0: nothing computed, cleared)
    int   m_voice = -1;                 ///< the GPU voice that sings it
    int   m_clear = 0;                  ///< bits: carriers whose lattice is cleared this block (a new pitch on it)
    int   m_pad0 = 0;                   ///< keeps the floats 16-byte aligned
    float m_omega = 10.0f;              ///< rad/s per unit of the lattice's energy: how fast it all runs
    float m_mass = 1.0f;                ///< u: 0 < |u| < 2 topological (its sign the way round), beyond trivial
    float m_disorder = 0.0f;            ///< the inside's random potential, in units of the hopping
    float m_damp = 0.1f;                ///< per second: how fast it fades
    float m_drive[HaloCarriers] = {};   ///< each carrier's drive this block (amplitude)
    int   m_site[HaloCarriers] = {};    ///< the rim site each carrier is driven at
    int   m_defectX[2] = { 99, 99 };    ///< the headset's hands as defects (Phase 30): each one's site across (99 none)
    int   m_defectY[2] = { 99, 99 };    ///< and from front to back
    float m_defectV = 0.0f;             ///< their potential, in units of the hopping (a wall where it is large)
    float m_defectR2 = 4.0f;            ///< their radius squared, sites
    float m_pad1 = 0.0f;                ///< keeps the struct 16-byte aligned
    float m_pad2 = 0.0f;                ///< likewise
};
static_assert( sizeof( Halo ) % 16 == 0, "Halo rides in the block header" );

/**
 * @brief The wave room (Phase 29): a room simulated as air -- the wave equation on a grid of RoomNX x RoomNY x RoomNZ
 *        cells (the standard leapfrog scheme, a Courant number of about a half), at the sample rate over RoomDecim:
 *        its modes, its standing waves and its flutter come of themselves, below about 1.5 kHz. The voices sound in
 *        it where they stand (their side and their distance), each from its own cell; two ears a cell either side of
 *        the listener hear it. The walls take up a share of what meets them (a locally reacting boundary of one
 *        admittance). The voices go down to the room's rate through a low pass, the ears come up through the same.
 */
struct WaveRoom
{
    int   m_on = 0;                      ///< 1: the room runs (0: nothing computed, its air at rest)
    int   m_nx = 0;                      ///< its cells across this block (at most RoomNX)
    int   m_ny = 0;                      ///< from front to back
    int   m_nz = 0;                      ///< from the floor up
    int   m_count = 0;                   ///< sources in `source`
    int   m_ear[2] = { 0, 0 };           ///< the ears' cells (left, right)
    int   m_clear = 0;                   ///< 1 for one block: the air at rest (the room changed size)
    float m_lambda2 = 0.25f;             ///< the Courant number squared
    float m_beta = 0.2f;                 ///< the walls' admittance (0: rigid; 1: as much taken as reflected away)
    float m_level = 0.0f;                ///< the ears' level
    int   m_srcZ = 0;                    ///< the plane the sources stand on (only its cells look for them)
    int   m_source[RoomSources] = {};    ///< each source's cell
    int   m_voiceOf[RoomSources] = {};   ///< and its GPU voice
    float m_gain[RoomSources] = {};      ///< and its gain
    int   m_absorb[2] = { -1, -1 };     ///< the headset's open hands (Phase 30): each one's cell, an absorber (-1 none)
    float m_absorbR2 = 4.0f;            ///< their radius squared, cells
    float m_absorbK = 0.0f;             ///< the share of the pressure they take a step there
    int   m_mirrorOn = 0;               ///< 1: the time mirror records and plays back (Phase 31)
    int   m_mirrorLen = 0;              ///< its window, samples at the room's rate
    int   m_mirrorPos = 0;              ///< where in the window this block's first step is
    int   m_mirrorBuf = 0;              ///< its stroke at this block's first step: 0 listening, 1 sending back reversed
    float m_mirrorGain = 0.0f;          ///< how loud a cell sends the reversed field back in
    float m_pad5 = 0.0f;                ///< keeps the struct 16-byte aligned
    float m_pad6 = 0.0f;                ///< likewise
    float m_pad7 = 0.0f;                ///< likewise
    int   m_mirror[RoomMirrors] = {};   ///< the mirror's cells (a ring on the sources' plane)
};
static_assert( sizeof( WaveRoom ) % 16 == 0, "WaveRoom rides in the block header" );

/**
 * @brief The Aeolian harp (Phase 30): HarpStrings strings standing in the wind (ChimeField's lattice) where the harp
 *        is.
 *
 * A string in the wind sheds a vortex street at Strouhal's frequency, 0.2 times the wind's speed over the string's
 * diameter. The wake is a van der Pol oscillator at that frequency, driving the string's HarpModes harmonics and
 * driven back by their acceleration (Facchinetti, de Langre and Biolley 2004): so a string locks onto the harmonic
 * nearest its vortices and holds it -- the lock-in --, and jumps to the next when the wind has moved far enough. A
 * harp's strings are tuned alike or to a chord and differ in their thickness: each sings another harmonic.
 */
struct AeolianHarp
{
    int   m_on = 0;                       ///< 1: the strings ring in the wind (0: nothing computed)
    int   m_clear = 0;                    ///< 1 for one block: the strings at rest, their wakes begun again
    int   m_x = 64;                       ///< where the harp stands in the wind's lattice: across (cells)
    int   m_y = 32;                       ///< and from near to far
    float m_level = 0.0f;                 ///< the harp's amplitude out
    float m_speed = 120.0f;               ///< m/s of wind per lattice unit of the wind's speed
    float m_eps = 0.3f;                   ///< the wake's van der Pol coefficient
    float m_couple = 12.0f;               ///< A: how strongly a string's acceleration drives its wake
    float m_force = 4.0e-5f;              ///< M: how strongly the wake's lift drives the string (air against gut)
    float m_pad0 = 0.0f;                  ///< keeps the arrays 16-byte aligned
    float m_pad1 = 0.0f;                  ///< likewise
    float m_pad2 = 0.0f;                  ///< likewise
    float m_hz[HarpStrings] = {};         ///< each string's fundamental, Hz
    float m_diameter[HarpStrings] = {};   ///< each string's diameter, m
    float m_place[HarpStrings] = {};      ///< each string's place, -1 (left) .. 1 (right)
};
static_assert( sizeof( AeolianHarp ) % 16 == 0, "AeolianHarp rides in the block header" );

/**
 * @brief Thunder (Phase 30): a strike lays out a lightning channel and every segment of it sends the listener an
 *        N-wave.
 *
 * The channel runs from the cloud's base to the ground, tortuous at every scale (a segment turns 16 degrees from the
 * straight way on average), with branches off its upper half. Each segment's N-wave is as long as the channel's
 * radius says (Few 1969: the energy per metre, about 1e5 J/m), lengthening on its way; it claps where the segment
 * stands across the line of sight and rumbles where it lies along it (its length then takes time to pass), and the air
 * takes the highs of the far ones. The waves arrive by distance -- the near part's crack first, the rumble of the rest
 * for seconds after -- and are summed once into a ring of whole numbers that the blocks read as it comes round.
 */
struct Thunder
{
    int   m_on = 0;         ///< 1: the ring is read and strikes are laid out (0: nothing computed)
    int   m_strike = 0;     ///< 1 for one block: a strike, its channel laid out this block
    int   m_chunk = -1;     ///< which of the channel's ThunderChunks parts goes into the ring this block (-1 none)
    int   m_seed = 0;       ///< the strike's dice
    int   m_read = 0;       ///< where in the ring this block is read
    int   m_origin = 0;     ///< where in the ring the strike's moment is
    int   m_pad0 = 0;       ///< keeps the floats 16-byte aligned
    int   m_pad1 = 0;       ///< likewise
    float m_level = 0.0f;   ///< the thunder's amplitude out
    float m_near = 0.15f;   ///< its share in the near plane
    float m_distance = 3000.0f;   ///< m: the strike's foot from the listener
    float m_azimuth = 0.0f;       ///< rad: its direction, 0 ahead, positive to the right
    float m_height = 3000.0f;     ///< m: the cloud's base, where the channel begins
    float m_tortuosity = 0.28f;   ///< rad: a segment's mean turn from the straight way
    float m_branches = 0.5f;      ///< 0 .. 1: how much the branches carry
    float m_energy = 1.0f;        ///< the strike's energy per metre against a typical one's
    float m_yaw = 0.0f;           ///< rad: the listener's head turned (the headset), positive to the right
    float m_pad2 = 0.0f;          ///< keeps the struct 16-byte aligned
    float m_pad3 = 0.0f;          ///< likewise
    float m_pad4 = 0.0f;          ///< likewise
};
static_assert( sizeof( Thunder ) % 16 == 0, "Thunder rides in the block header" );

/**
 * @brief The singing ice (Phase 30): a frozen lake under the stress the cold builds, heard from its shore.
 *
  * The sheet is a grid of slider blocks (Olami, Feder and Christensen 1992): a cell past its threshold slips and hands
 * a share of its stress to each neighbour, which may slip in turn -- so cracks come in avalanches of every size, the
 * large ones rare. The cold raises every cell alike until the most stressed one reaches its threshold (the slow drive
 * of the model, a crack at a time, as often as the cold says); the sheet begins in its critical state, warmed by many
 * cracks before (setIceStress). A crack is heard across the ice as a chirp falling from the highs:
 * waves of bending are dispersive, their speed growing as the square root of the frequency, so the highs arrive first
 * and the frequency falls about as one over the time squared -- the laser-like song of lake ice. The thinner the ice,
 * the slower the waves and the longer the song.
 */
struct SingingIce
{
    int   m_on = 0;          ///< 1: the sheet sings (0: nothing computed)
    int   m_reset = 0;       ///< 1 for one block: no crack running, every chirp silent (the stress by setIceStress)
    int   m_crack = 0;       ///< 1 for one block: the cold brings the most stressed cell to its threshold
    int   m_pad0 = 0;        ///< keeps the floats 16-byte aligned
    float m_level = 0.0f;    ///< the chirps' amplitude out
    float m_beta = 96.0f;    ///< m^2/s: the sheet's flexural constant sqrt(D / (rho h)), 958 times its thickness
    float m_size = 400.0f;   ///< m: the lake from side to side and from the shore out
    float m_pad1 = 0.0f;     ///< keeps the struct 16-byte aligned
};
static_assert( sizeof( SingingIce ) % 16 == 0, "SingingIce rides in the block header" );

/** @brief Who calls in the night chorus: the ChorusKind parameter's choices. */
enum class ChorusKindE : std::int32_t
{
    MEADOW = 0,     ///< a meadow: field crickets, snowy tree crickets, mole crickets
    TREETOPS,       ///< the treetops: tree crickets and katydids
    POND,           ///< a pond: peepers, tree frogs, bullfrogs, toads
    SUMMER_NIGHT,   ///< all of them
    COUNT           ///< number of kinds
};

/**
 * @brief The night chorus (Phase 31): ChorusCallers insects and frogs over the field, each calling in its own rhythm.
 *
 * Eight kinds, each in its own band of the spectrum and with its own call -- a cricket's chirp of a few pulses at
 * 4.6 kHz, a tree cricket's trill, a katydid's rasp, a peeper's rising peep, a tree frog's pulsed call, a bullfrog's
 * deep drone (each kind its niche: Krause's acoustic niche hypothesis). An insect's rates follow the night's warmth
 * (Dolbear's law: the snowy tree cricket's chirps a minute are about 7 times the degrees Celsius less 30). Every caller
 * is an oscillator that calls when it comes round; a call of its own kind heard close enough resets it towards its
 * start (Greenfield's inhibitory resetting) -- so neighbours fall into step, or into turns --, and a caller about to
 * call into a louder caller of another kind in its band waits a little (a gap in time as well as in pitch).
 */
struct NightChorus
{
    int   m_on = 0;                      ///< 1: the callers call (0: nothing computed)
    int   m_reset = 0;                   ///< 1 for one block: the callers scattered anew over the field
    int   m_kind = 0;                    ///< who calls (ChorusKindE)
    int   m_pitchCount = 0;              ///< pitches in m_pitch
    float m_level = 0.0f;                ///< the chorus's amplitude out
    float m_warmth = 18.0f;              ///< degrees Celsius: the night's temperature (the insects' rates)
    float m_sync = 0.5f;                 ///< 0 .. 1: how far a call heard resets a neighbour's clock
    float m_tune = 0.0f;                 ///< 0 .. 1: the carriers pulled to the scale's pitches
    float m_pitch[ChorusPitches] = {};   ///< Hz: the scale's pitches the carriers may be pulled to
};
static_assert( sizeof( NightChorus ) % 16 == 0, "NightChorus rides in the block header" );

/**
 * @brief Star (Phase 33): a star's oscillations heard -- its modes (asteroseismology) transposed into the audible, each
 *        a damped oscillator kept ringing by the convection's noise: a Lorentzian line of its own width. The modes
 *        come as a table (CloudEngine::setStarModes); here what holds for all of them.
 */
struct StarField
{
    int   m_on = 0;         ///< 1: the star sounds (0: nothing computed)
    int   m_count = 0;      ///< modes in the table (StarModes at most)
    int   m_reset = 0;      ///< 1 for one block: the modes begin from silence (a new star)
    int   m_pad = 0;        ///< keeps the struct 16-byte aligned
    float m_level = 0.0f;   ///< the star's amplitude out
    float m_far = 0.7f;     ///< 0 .. 1: its share in the far plane
    float m_pad1 = 0.0f;    ///< likewise
    float m_pad2 = 0.0f;    ///< likewise
};
static_assert( sizeof( StarField ) % 16 == 0, "StarField rides in the block header" );

/**
 * @brief Black hole (Phase 33): a share of the partials falls towards a hole across the field -- redshifted, dimmed
 *        and drawn to it, frozen at the horizon, reborn far out -- and the rest is lensed: the light behind it bent
 *        into a ring round it and brightened.
 */
struct BlackHole
{
    int   m_on = 0;             ///< 1: the hole bends the cloud (0: nothing)
    int   m_pad = 0;            ///< keeps the struct 16-byte aligned
    float m_share = 0.0f;       ///< 0 .. 1: the share of the partials falling
    float m_fallPhase = 0.0f;   ///< 0 .. 1: the fall's clock (time over a fall's length, wrapped)
    float m_place = 0.3f;       ///< -1 .. 1: where the hole stands across
    float m_ring = 0.0f;        ///< the Einstein ring's radius across the field (0: no lensing)
    float m_start = 30.0f;      ///< where a falling partial begins, in Schwarzschild radii
    float m_pad1 = 0.0f;        ///< likewise
};
static_assert( sizeof( BlackHole ) % 16 == 0, "BlackHole rides in the block header" );

/** @brief What lives in the field: the LifeMode parameter's choices. */
enum class LifeModeE : std::int32_t
{
    OFF = 0,   ///< no field
    LENIA,     ///< Lenia (Chan 2019): Conway's Life made continuous -- blobs that drift, pulse, split, die
    TURING,    ///< Gray-Scott reaction-diffusion (Pearson 1993): spots, mitosis, coral, worms, mazes
    // Animated pictures (Phase 13d): KaleidoscopeEnhanced's continuous transforms, over the landscape
    // (the factory's or the player's own) or drawn from nothing.
    KALEIDOSCOPE,   ///< the landscape folded into turning mirror wedges round a wandering centre
    TUNNEL,         ///< a flight down a tunnel lined with the landscape: rings born in the middle, running out
    SPIRAL,         ///< the landscape wound into a log-polar spiral that zooms for ever (Droste)
    FLOW,           ///< the landscape carried on a flowing noise current, like ink in water
    FEEDBACK,       ///< video feedback: the field fed back into itself, turned and zoomed, the landscape seeping in
    RAIN,           ///< drops on a pond: a damped 2D wave, rings spreading and crossing
    STARFIELD,      ///< stars streaming out of the middle (log-polar: they speed up as they go)
    MOIRE,          ///< two turning line gratings, their slow interference bands
    // Phase 16: two more fields that live
    FLUID,     ///< a fluid (lattice Boltzmann): the notes played blow plumes of ink into it, which roll up into eddies
    ISLANDS,   ///< Cahn-Hilliard separation (Oono-Puri): a fine-grained field gathers into a few large islands that drift
    COUNT      ///< number of modes
};

/** @brief Whether a living landscape is one of the animated pictures (computed, not grown). @param[in] m the mode @return true then */
#if defined( __CUDACC__ )
__host__ __device__
#endif
    inline bool lifeIsPicture( LifeModeE m )
{
    return m >= LifeModeE::KALEIDOSCOPE && m <= LifeModeE::MOIRE;
}

/**
 * @brief The living landscape: a field on a torus the GPU steps every block, and every partial reads.
 *
 * Where the landscapes are pictures that loop, this one lives: a Lenia field or a Gray-Scott
 * field of LifeW x LifeH cells, stepped up to twice a block in the graph before the partials are
 * updated. Each partial reads it at its frequency (across, log, `zoom` of the field spanning the
 * range) and at its place in its cloud (along), so every harmonic is a slice through the living
 * pattern and the pattern's motion is the partials' loudness. Made up by the field's own power
 * (`makeup`, from the GPU's measure of the last block), so a sparse field changes the colour, not
 * the level; a field that died or filled up is seeded again.
 */
struct LifeField
{
    LifeModeE m_mode = LifeModeE::OFF;   ///< what lives in it
    int       m_steps = 0;               ///< steps this block, 0 .. 2
    int       m_reseed = 0;              ///< 1 for one block: seeded again first (the engine asks)
    int       m_salt = 0;                ///< the seeding's dice, counted up at every seeding
    float     m_depth = 0.0f;            ///< 0 .. 1: how far the partials' loudness follows the field
    float     m_zoom = 1.0f;             ///< 0.25 .. 1: how much of the field's width the range spans
    float     m_makeup = 1.0f;           ///< one over the field's RMS (the engine's, from lifeStats)
    float     m_lenMu = 0.15f;           ///< Lenia: the growth bell's centre
    float     m_lenSigma = 0.015f;       ///< Lenia: its width
    float     m_feed = 0.035f;           ///< Turing: the feed rate F
    float     m_kill = 0.065f;           ///< Turing: the kill rate k
    float     m_time = 0.0f;             ///< pictures: how far the flow has run (Life Rate's units)
    float     m_angle = 0.0f;            ///< pictures: how far it has turned, radians (Turn)
    float     m_sides = 6.0f;            ///< pictures: mirror wedges, spiral arms, grating lines (Sides)
    float     m_character = 0.3f;        ///< pictures: how strongly the transform distorts (Character)
    float     m_seconds = 0.0f;          ///< pictures: real seconds, for the centre's slow wander
    float     m_drops = 0.0f;            ///< Rain: the chance a cell is struck this block
    float     m_turnStep = 0.0f;         ///< pictures: the turn this block, radians (Feedback turns by it)
    float     m_fade = 0.999f;           ///< Fluid: the share of the ink kept each step
    float     m_tau = 0.53f;             ///< Fluid: the lattice's relaxation time (its viscosity: (tau - 1/2) / 3)
    float     m_push = 0.004f;           ///< Fluid: the jets' force, lattice units at full strength
    int       m_substeps = 1;            ///< Islands: steps of the separation this block
    float     m_mean = 0.0f;             ///< Islands: the field's mean (0: a labyrinth; below: islands of the minority)
    float m_eps = 0.001f;   ///< Islands: Oono and Puri's long-range term (more: smaller domains, which stop growing)
    float m_drift = 0.0f;   ///< Islands: how far the field has drifted across, cells of its grid
    float m_pad1 = 0.0f;    ///< keeps the table 16-byte aligned
    float m_pad2 = 0.0f;    ///< likewise
    float m_pad3 = 0.0f;    ///< likewise
    float m_jetX[LifeJets] = {};      ///< Fluid: where each jet blows, cells across (the notes' frequencies)
    float m_jetAmt[LifeJets] = {};    ///< Fluid: how strongly, 0 .. 1 (0: none)
    float m_jetSway[LifeJets] = {};   ///< Fluid: its lean, sideways share of its push
};
static_assert( sizeof( LifeField ) % 16 == 0, "LifeField rides in the block header" );

/** @brief How the partials' oscillators couple (PartialSync::mode). */
enum class SyncModeE : std::int32_t
{
    GLOBAL = 0,   ///< Kuramoto: every member to its cloud's mean
    CHIMERA,      ///< to its neighbours round the cloud, with a phase lag: an ordered part beside a disordered one
    SWARM,        ///< swarmalators: the phase and a place on a circle in pitch and pan pull each other
};

/**
 * @brief Partial sync: a Kuramoto oscillator for every partial, coupled to its own cloud.
 *
 * Every partial carries a phase that turns at a rate of its own (`rate`, scattered by `spread`)
 * and is pulled towards the mean phase of the members of its harmonic -- Kuramoto's mean field,
 * dθ = ω + K R sin(ψ - θ), measured by the GPU each block in a fixed order. Its loudness follows
 * the phase by `depth`. Uncoupled, the members flicker each for itself and the flicker averages
 * out in the harmonic; coupled past the critical strength they fall into step and the whole cloud
 * pulses as one -- fireflies, pacemaker cells, applause. The mean power is kept whatever happens:
 * the factor is (1 + depth sin θ) / sqrt(1 + depth^2 / 2).
 *
 * Chimera (Kuramoto and Battogtokh 2002, Abrams and Strogatz 2004): the members sit round a ring in
 * their order across the cloud and pull only their neighbours, by 1 + A cos(distance) with A
 * 0.995, through a phase lag a little under a quarter turn. From a start that is ordered in one
 * part and not in the other, the two stay side by side: a coherent pulsing core in a cloud that
 * flickers -- and in a finite cloud the border wanders. Three mean phasors a cloud carry it.
 *
 * Swarm (O'Keeffe, Hong and Strogatz 2017; on a ring, O'Keeffe, Ceron and Petrungaro 2022): every
 * partial also has a place x on a circle -- its pitch moved by reach cos x, its side by up to half
 * the field with sin x -- and place and phase pull each other: dx = v + J/2 (S+ sin(Φ+ - (x + θ))
 * + S- sin(Φ- - (x - θ))), dθ = ω + K/2 (S+ sin(Φ+ - (x + θ)) - S- sin(Φ- - (x - θ))), with
 * S± e^(iΦ±) the mean of e^(i(x ± θ)) over the cloud, in the frame turning at the rate. Its states:
 * the cloud gathered and pulsing as one; scattered and flickering; a phase wave, where each place
 * has its phase and the loudest part of the cloud circles through pitch and field at the rate; and
 * a phase wave broken into several, pulsing against each other.
 */
struct PartialSync
{
    float     m_depth = 0.0f;               ///< 0 .. 1: how far the loudness follows the phase (0: nothing computed)
    float     m_coupling = 0.0f;            ///< K, rad/s
    float     m_rate = 0.5f;                ///< Hz: the oscillators' middle rate
    float     m_spread = 0.3f;              ///< 0 .. 1: each rate scattered by up to this share either way
    SyncModeE m_mode = SyncModeE::GLOBAL;   ///< how they couple
    float     m_swarm = 0.0f;               ///< Swarm: J, rad/s (positive: like phases gather)
    float     m_reach = 0.0f;               ///< Swarm: cents a partial is moved on its circle
    float     m_lag = 1.39f;                ///< Chimera: the phase lag, radians
};
static_assert( sizeof( PartialSync ) % 16 == 0, "PartialSync rides in the block header" );

constexpr int RippleCells = 256;   ///< the ripple's string: cells over the range, log (20 Hz .. 20 kHz)

/**
 * @brief The ripple: a damped wave along the spectrum, its displacement read by every partial at its
 *        frequency as a share of its loudness. The engine runs the string (Engine::stepRipple) and sends
 *        it up with the voices.
 */
struct Ripple
{
    float m_depth = 0.0f;          ///< 0 .. 1: (1 + depth tanh u) times the make-up (0: nothing read)
    float m_makeup = 1.0f;         ///< one over the root of (1 + depth^2 mean tanh^2 u)
    float m_periodic = 0.0f;       ///< 0 .. 1: read as 2 sin(u / 2) instead of u (Breathers: the string's angle)
    float m_pad1 = 0.0f;           ///< likewise
    float m_u[RippleCells] = {};   ///< the string's displacement, cell by cell
};
static_assert( sizeof( Ripple ) % 16 == 0, "Ripple rides in the block header" );

constexpr int ReservoirN = 2048;      ///< the reservoir's neurons
constexpr int ReservoirInputs = 8;    ///< what it is fed
constexpr int ReservoirOutputs = 4;   ///< what is read from it

/**
 * @brief The reservoir: an echo state network (Jaeger 2001) on the GPU -- ReservoirN neurons, a
 *        fixed random coupling scaled to `chaos` (its spectral radius: below 1 it forgets, about 1
 *        it rings at the edge of chaos, above it runs on by itself), leaking at `leak` a block,
 *        driven by ReservoirInputs numbers about the music, read out by ReservoirOutputs fixed
 *        random projections. Smooth, never periodic, with a memory of what was played.
 */
struct Reservoir
{
    int   m_on = 0;                    ///< 1: stepped this block (a matrix row reads it)
    float m_chaos = 0.95f;             ///< the coupling's spectral radius
    float m_leak = 0.001f;             ///< a block's share of the new state
    float m_input = 0.7f;              ///< the inputs' gain
    float m_u[ReservoirInputs] = {};   ///< the inputs, about -1 .. 1
};
static_assert( sizeof( Reservoir ) % 16 == 0, "Reservoir rides in the block header" );

constexpr int FieldPoints = 128;     ///< the scale's pitches a Scale field may pull to
constexpr int FieldAttractors = 8;   ///< attractors a Swarm or Scatter field may have
constexpr int GainRing = 1024;       ///< blocks of every voice's level the GPU keeps, for the dispersion (2.7 s at 128)

/** @brief What the field does: the FieldMode parameter's choices. */
enum class FieldModeE : std::int32_t
{
    OFF = 0,   ///< every partial at its own place (a field switched off lets them glide home)
    SCALE,     ///< every partial pulled towards the nearest pitch of the scale: the cloud collapses into the key
    SWARM,     ///< a few attractors wander through the spectrum on chaotic paths and gather the partials near them
    SCATTER,   ///< the same attractors push the partials away: moving gaps, like dust parted by a wind
    FLOCK,     ///< every cloud's members as a flock of starlings: cohesion, alignment, separation, a wind
    ORBIT,   ///< the scale's pitches as wells in pitch and side, a magnetic field bending every move: the partials circle them
    PHYSARUM,   ///< a share of the partials crawl as a slime mould over a trail the harmonics feed (Physarum)
    COUNT       ///< number of modes
};

/**
 * @brief Forces on the partials' frequencies, and the dispersion of their clocks: the host's copy,
 *        sent up with the voices.
 *
 * The field. Every partial keeps an offset in cents from its own place (its harmonic, its member,
 * its drift and pitch lane), carried from block to block on the GPU. Each block it moves towards
 * what pulls it -- the nearest pitch of the scale, or the attractors near it, weighted by a bell
 * of `radius` cents -- by at most `strength` cents a second and never past the attractor, and
 * the spring draws it back towards its own place. So a cloud does not snap to a grid: it is
 * gathered, it gives way, it is let go, at the speeds the knobs say. The mean level does not
 * change: partials that meet keep their random phases, so their powers add as before.
 *
 * The dispersion. Each partial's clock runs `dispersion` seconds late at the bottom of the range
 * (20 Hz) and on time at the top when it is positive, the other way round when negative -- its
 * envelopes, the landscape it reads and the voice's level, which the GPU keeps a history of. An
 * attack or a glint then reaches the ear as a cascade, the highs first and the lows after (a
 * falling chirp, as in a dispersive medium -- a long spring, a stiff string, ice), or rising.
 */
struct SpectralField
{
    FieldModeE m_mode = FieldModeE::OFF;   ///< what pulls the partials
    int        m_count =
        0;   ///< Scale: pitches in `pos`; Swarm, Scatter: attractors in `pos` and `mass`; Flock: winds in `pos`
    int m_flockFresh =
        0;   ///< Flock: 1 when the last block was no flock (its state is taken from where the partials are)
    int m_orbitFresh =
        0;   ///< Orbit: 1 when the last block was no orbit (its state is taken from where the partials are)
    float m_strength = 0.0f;   ///< cents a second a partial may move at most
    float m_radius = 60.0f;    ///< cents: an attractor's reach (the bell's width); the Scale field's stiffness
    float m_spring = 0.3f;     ///< per second: how fast a partial is drawn back to its own place
    float m_dispersion =
        0.0f;   ///< seconds: the clock of the lowest partials late (positive) or of the highest (negative)
    float m_orbitWell = 3.0f;    ///< Orbit: the wells' angular frequency, rad/s (their depth)
    float m_orbitField = 6.0f;   ///< Orbit: the magnetic field as the circling's angular frequency, rad/s
    float m_orbitDamp = 0.3f;    ///< Orbit: the damping, per second
    float m_timeBend =
        0.0f;   ///< -1 .. 1: each partial's clock slower (positive) or faster where the spectrum is dense
    float m_pos
        [FieldPoints] = {};   ///< log2 Hz: the scale's pitches ascending (Scale), or the attractors (Swarm, Scatter)
    float m_mass[FieldAttractors] = {};   ///< the attractors' weights; Scatter's are negative
};
static_assert( sizeof( SpectralField ) % 16 == 0, "SpectralField rides in the block header" );

/** @brief How an engine is laid out; fixed between open() and close(). */
struct CloudConfig
{
    double m_sampleRate = 48000.0;       ///< Hz
    int    m_blockSize = 128;            ///< samples per launch(), a power of two, 32 .. 4096
    int    m_voices = 16;                ///< voices the GPU tables hold
    int    m_partialsPerVoice = 20000;   ///< the most partials one voice can render
    bool   m_spinWait = true;            ///< wait() spins on the GPU's event (lowest latency) instead of sleeping on it
};

/**
 * @brief The GPU side: tables of partial state, the kernels and the graph that runs them.
 *
 * Use: open(), setShapes() (and setLandscape()), then per block edit voice(i), launch(), wait()
 * (or poll ready()). Not thread-safe: one thread drives it.
 */
class CloudEngine
{
public:
    CloudEngine();
    ~CloudEngine();
    CloudEngine( const CloudEngine & ) = delete;              ///< owns GPU memory: not copyable
    CloudEngine &operator=( const CloudEngine & ) = delete;   ///< owns GPU memory: not copyable

    /**
     * @brief Picks the first CUDA device, allocates the tables and builds the graph.
     * @param[in]  config  the layout
     * @param[out] error   receives the reason when it fails (may be null)
     * @return true when the engine can render
     */
    bool open( const CloudConfig &config, std::string *error );

    /** @brief Frees everything; open() may be called again. */
    void close();

    /** @brief The layout open() was given. @return the config */
    const CloudConfig &config() const;

    /**
     * @brief Replaces the bank of envelope shapes; waits for a block in flight first.
     * @param[in] shapes  the shapes
     * @param[in] count   how many, at most EnvShapes (more are dropped)
     * @return true when uploaded
     */
    bool setShapes( const EnvShape *shapes, int count );

    /**
     * @brief Replaces the landscape: rows of time, each `columns` amplitudes from LandLowHz to
     *        LandHighHz; it loops over all its rows. Waits for a block in flight first.
     * @param[in] amps        rows * columns amplitudes 0 .. 1, row by row
     * @param[in] columns     2 .. LandMaxColumns
     * @param[in] rows        1 .. LandMaxRows
     * @param[in] rowSeconds  how long one row lasts at rate 1
     * @return true when uploaded (false: outside the limits, nothing changed)
     */
    bool setLandscape( const float *amps, int columns, int rows, float rowSeconds );

    /**
     * @brief The host's copy of a voice, edited freely between wait() and the next launch().
     * @param[in] i  0 .. voices-1
     * @return the voice
     */
    CloudVoice &voice( int i );

    /** @brief The host's copy of the modal hall's settings, edited between wait() and the next launch(). @return the settings */
    ModalHall &modal();

    /** @brief The host's copy of the field and the dispersion, edited between wait() and the next launch(). @return the settings */
    SpectralField &field();

    /** @brief The host's copy of the binaural placement, edited between wait() and the next launch(). @return the settings */
    Binaural &binaural();

    /** @brief The host's copy of the spectral echo's settings, edited between wait() and the next launch(). @return the settings */
    SpectralEcho &spectralEcho();

    /** @brief The host's copy of the living landscape's settings, edited between wait() and the next launch(). @return the settings */
    LifeField &life();

    /** @brief The host's copy of the partial sync's settings, edited between wait() and the next launch(). @return the settings */
    PartialSync &sync();

    /** @brief The host's copy of the ripple, edited between wait() and the next launch(). @return the ripple */
    Ripple &ripple();

    /** @brief The host's copy of the reservoir's settings, edited between wait() and the next launch(). @return the settings */
    Reservoir &reservoir();

    /** @brief The host's copy of the melt's settings, edited between wait() and the next launch(). @return the settings */
    Melt &melt();

    /** @brief The host's copy of the spectral comb, edited between wait() and the next launch(). @return the settings */
    SpectralComb &comb();
    /** @brief The energy cascade's settings for the next block (the voices carry their drives). @return them */
    Turbulence &turbulence();
    /** @brief The chime field's settings for the next block (Phase 27). @return them */
    ChimeField &chimes();
    /** @brief The dream's settings for the next block (Phase 27). @return them */
    Dream &dream();
    /** @brief How many memories the dream keeps, as the last block left it. @return 0 .. DreamMemories */
    int dreamMemories() const;
    /** @brief How many chimes struck in the last block. @return the count */
    int chimeStrikes() const;
    /** @brief The weather's settings for the next block (Phase 28). @return them */
    Weather &weather();
    /** @brief The bloom's settings for the next block (Phase 28). @return them */
    Bloom &bloom();
    /** @brief The swarmalators' settings for the next block (Phase 28). @return them */
    SwarmSpace &swarm();
    /** @brief The physarum's settings and food for the next block (Phase 28). @return them */
    Physarum &physarum();
    /**
     * @brief What the weather, the bloom, the swarm and the physarum did in the last block (Phase 28).
     * @param[out] out  receives 8 values: drops sounding, drops born, the wind's mean speed (lattice units), the
     *                  leaves' mean rustle (0 .. 1), the sea's mean foam, the bloom's share of its energy above 2 kHz,
     *                  the swarm's phase order R and its phase-place order S (each 0 .. 1)
     */
    void weatherStats( float *out ) const;
    /**
     * @brief The physarum's trail as the last block left it: its mean and its spread (standard deviation over the
     *        mean -- 0 for an even trail, large for veins).
     * @param[out] mean    receives the mean
     * @param[out] spread  receives the spread
     */
    void physarumStats( float &mean, float &spread ) const;
    /** @brief The halo's settings and drives for the next block (Phase 29). @return them */
    Halo &halo();
    /** @brief The wave room's settings and sources for the next block (Phase 29). @return them */
    WaveRoom &room();
    /** @brief The Aeolian harp's settings for the next block (Phase 30). @return them */
    AeolianHarp &harp();
    /**
     * @brief Thunder's settings for the next block (Phase 30). A strike asked for (m_strike) while the last one's
     *        channel is still being sent into the ring waits for it; the ring's reading and the strike's moment are
     *        kept here (m_read, m_origin, m_chunk are overwritten).
     * @return them
     */
    Thunder &thunder();
    /** @brief The singing ice's settings for the next block (Phase 30). @return them */
    SingingIce &ice();
    /** @brief The night chorus's settings for the next block (Phase 31). @return them */
    NightChorus &chorus();
    /** @brief The star's settings for the next block (Phase 33). @return them */
    StarField &star();
    /** @brief The black hole's settings for the next block (Phase 33). @return them */
    BlackHole &hole();
    /**
     * @brief Replaces the star's modes at the next launch (Phase 33).
     * @param[in] in     four values a mode: its frequency (Hz), its line width (Hz), its amplitude, its place (-1 .. 1)
     * @param[in] count  how many (StarModes at most)
     */
    void setStarModes( const float *in, int count );
    /**
     * @brief What the night chorus did in the last block (Phase 31).
     * @param[out] out  receives 4 values: the calls begun, the callers calling, how far the most numerous kind's clocks
     *                  stand together (the order parameter, 0 .. 1), the share of the callers waiting for a gap
     */
    void chorusStats( float *out ) const;
    /**
     * @brief The dream's newest memories, brought down from the card (Phase 30, the dream journal); waits for the block
     *        in flight. The thread that launches.
     * @param[out] out   receives up to `most` memories, the newest first, DreamBands values each (each of length 1)
     * @param[in]  most  how many at most
     * @return how many
     */
    int dreamRead( std::vector<float> &out, int most );
    /**
     * @brief Replaces the dream's memories (Phase 30: a journal read back); waits for the block in flight. The thread
     *        that launches.
     * @param[in] in     the memories, the newest first, DreamBands values each
     * @param[in] count  how many (at most DreamMemories)
     */
    void dreamWrite( const float *in, int count );
    /**
     * @brief Replaces the ice sheet's stress at the next launch (Phase 30: a sheet warmed into its critical state).
     * @param[in] in  IceN x IceN stresses, row by row (below 1)
     */
    void setIceStress( const float *in );
    /**
     * @brief What the night did in the last block (Phase 30).
     * @param[out] out  receives 4 values: the harp's strings' mean wake amplitude, the harmonic they sound (by energy,
     *                  averaged over the strings), the ice's last crack's size (slips), the chirps begun since open()
     */
    void nightStats( float *out ) const;
    /**
     * @brief The halo's carriers as the last block left them (Phase 29).
     * @param[out] energy  receives HaloCarriers values: each lattice's energy (the sum of its amplitudes squared)
     * @param[out] angle   receives HaloCarriers values: where round the disc its energy's middle stands, radians
     * @param[out] rim     receives HaloCarriers values: the share of its energy on the rim (the outer two rings)
     */
    void haloStats( float *energy, float *angle, float *rim ) const;
    /**
     * @brief Which GPU voices carry the layers that are not notes (Phase 27): the phantom layer's chunks are summed
     *        apart, into the frame's channels 14 and 15.
     * @param[in] phantomVoice  the phantom layer's voice, -1 none
     * @param[in] dreamVoice    the dream's voice, -1 none
     */
    void setLayers( int phantomVoice, int dreamVoice );

    /**
     * @brief The hall's memory (ModalHall::grow): per band over the octaves, how much was heard there
     *        against the average (1), as the last snapshot brought it down.
     * @param[out] out  receives HallBins values
     * @return true when the memory has come down at least once
     */
    bool hallMemory( float *out ) const;
    /**
     * @brief Replaces the hall's memory at the next launch (a hall loaded, or a new one).
     * @param[in] in  HallBins values (1: nothing learnt)
     */
    void setHallMemory( const float *in );

    /**
     * @brief The reservoir's outputs after the last block, -1 .. 1.
     * @param[out] out  receives ReservoirOutputs values
     */
    void reservoirOut( float *out ) const;

    /**
     * @brief The living field's mean and mean square (of what the partials read, 0 .. 1), measured by
     *        the GPU in the last block.
     * @param[out] mean    receives the mean
     * @param[out] meanSq  receives the mean of the squares
     */
    void lifeStats( float &mean, float &meanSq ) const;

    /**
     * @brief What the instrument heard of its own voices in the last block (an FFT of the last 1024
     *        samples, near and far together): its level, where its energy stands, how much it changed.
     * @param[out] levelDb   receives the level, dB of full scale
     * @param[out] centroid  receives the energy's centre on the log axis, 0 at 20 Hz .. 1 at 20 kHz
     * @param[out] flux      receives the change: the rise of the spectrum's magnitudes over their sum, 0 .. 1
     */
    void heard( float &levelDb, float &centroid, float &flux ) const;

    /**
     * @brief The living field as the last snapshot brought it down: LifeW x LifeH cells, each the value
     *        the partials read (0 .. 1), row after row. Valid after takeSnapshot() returned true.
     * @return  the cells, or null when the field is off
     */
    const float *lifePicture() const;

    /** @brief The next launch() also brings the partials' snapshot down (every SnapStride-th partial's frequency and amplitude). */
    void requestSnapshot();

    /**
     * @brief The snapshot a launch brought down, once: after the wait() of a launch that copied it.
     * @param[in]  dots      receives the points, per voice snapshotPerVoice() of (Hz, amplitude), voices in order
     * @param[out] perVoice  receives how many points each voice has
     * @return true when a new snapshot is there (then valid until the next launch())
     */
    bool takeSnapshot( const float *&dots, int &perVoice );

    /**
     * @brief Sends the voices up and starts one block on the GPU; returns at once. Clears every
     *        voice's restart flag and advances the active voices' time by one block.
     */
    void launch();

    /** @brief Whether the launched block is done (never blocks). @return true when wait() would return at once */
    bool ready() const;

    /**
     * @brief Waits for the launched block.
     * @return  blockSize frames of CloudOutChannels (near L, R, far L, R, modal hall L, R, spectral echo L, R, melt L, R), valid until the next launch()
     */
    const float *wait();

    /** @brief GPU time of the last finished block, from the events around the graph. @return milliseconds */
    float lastGpuMs() const;

    /** @brief The device's name and compute capability, for the logs. @return e.g. "NVIDIA GeForce RTX 5090 (sm_120)" */
    std::string deviceName() const;

private:
    struct Impl;                    ///< the CUDA side, defined in Cloud.cu
    std::unique_ptr<Impl> m_impl;   ///< null while closed
};

}   // namespace additiv
