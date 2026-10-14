/**
 * @file Cloud.cu
 * @brief The partial cloud's kernels and the engine that runs them as one CUDA graph per block.
 *
 * Per block three kernels run, in this order:
 *   - update: one thread per partial. Turns the voice's few numbers into this partial's frequency
 *     (harmonic, place in the cloud, drift, pitch envelope), amplitude (tilt, shimmer, amplitude
 *     envelope, the voice's gain) and place in the field (pan envelope), all for the END of the
 *     block, and writes what the render kernel needs: start phase, phase step and its per-sample
 *     change (a linear glide, so pitch envelopes never step), and the left/right amplitude ramps.
 *   - render: one CUDA block per chunk of kChunk partials, one thread per sample. The chunk's
 *     partials sit in shared memory and every thread of a warp reads the same partial at the same
 *     time (a broadcast), so the inner loop is one __sinf and a handful of FMAs per partial and sample.
 *   - reduce: adds the chunks' partial sums sample by sample, always in the same order, so a block
 *     renders bit for bit the same every time.
 * Phases are 32-bit fixed point: they wrap exactly, so a note may hold for hours without the
 * drift a float phase would gather.
 */
#include "additiv/Cloud.h"
#include "additiv/Spectrum.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <immintrin.h>
#include <vector>

namespace additiv {
namespace {

constexpr int kChunk = 512;     ///< partials per CUDA block of the render kernel
constexpr int kThreads = 256;   ///< threads per CUDA block of the update and render kernels
constexpr int kReduceLanes = 32;    ///< samples per CUDA block of the reduce kernel
constexpr int kReduceGroups = 16;   ///< threads per sample in the reduce kernel, each summing every 16th chunk
constexpr int kModalThreads = 256;                          ///< modes per CUDA block of the modal kernel
constexpr int kModalBlocks = kModalModes / kModalThreads;   ///< CUDA blocks of the modal kernel
constexpr int kModalTile = 128;                             ///< samples the modal kernel sums per pass
static_assert(kModalModes % kModalThreads == 0, "the modes fill whole CUDA blocks");
constexpr int kEchoN = 1024;                 ///< the spectral echo's FFT size
constexpr int kEchoBins = kEchoN / 2 + 1;    ///< its bands, 0 .. N/2
constexpr int kEchoSlots = 2048;             ///< blocks each band's delay line holds (5.4 s at 128)
constexpr int kEchoThreads = kEchoN / 2;     ///< threads of its one CUDA block: one butterfly each
constexpr int kMeltBands = kMeltN / 2;       ///< the melt's bands that live, 0 .. N/2 - 1 (0, the DC, kept silent)
constexpr int kMeltThreads = 1024;           ///< threads of its one CUDA block: two bands each
static_assert(kMeltBands == 2 * kMeltThreads, "two bands a thread");
constexpr int kLifeCells = kLifeW * kLifeH;  ///< cells of the living field
static_assert((kLifeW & (kLifeW - 1)) == 0 && (kLifeH & (kLifeH - 1)) == 0, "the torus wraps by masks");
/** @brief Lenia's ring kernel, normalised, set once at open(). */
__constant__ float cLeniaKernel[(2 * kLifeRadius + 1) * (2 * kLifeRadius + 1)];
constexpr float kAirShare = 0.2f;    ///< the share of a voice's partials that become its air band when Air is up

/** @brief What every kernel needs to know about the block; uploaded in front of the voices. */
struct BlockHeader {
    int   n;               ///< samples in this block
    int   nPrev;           ///< samples of the previous block, by which the phases advance
    float sampleRate;      ///< Hz
    int   voices;          ///< voices in the table
    int   shapeCount;      ///< shapes in the bank
    int   landColumns;     ///< the landscape's frequency columns (0 = none)
    int   landRows;        ///< its time rows
    float landRowSeconds;  ///< how long one row lasts
    std::uint32_t blockIndex;   ///< blocks launched since open(): where this block's levels go in the ring
    std::uint32_t hpad0;        ///< keeps what follows 16-byte aligned
    std::uint32_t hpad1;        ///< likewise
    std::uint32_t hpad2;        ///< likewise
    ModalHall modal;       ///< the modal hall's settings for this block
    SpectralField field;   ///< the field and the dispersion for this block
    Binaural binaural;     ///< the binaural placement for this block
    SpectralEcho echo;     ///< the spectral echo for this block
    LifeField life;        ///< the living landscape for this block
    PartialSync sync;      ///< the partial sync for this block
    Ripple ripple;         ///< the ripple for this block
    Reservoir reservoir;   ///< the reservoir for this block
    Melt melt;             ///< the melt for this block
    SpectralComb comb;     ///< the spectral comb for this block
    Turbulence turb;       ///< the energy cascade for this block
    std::int32_t lifeParity;   ///< which of the two field buffers holds the field before this block's steps
    std::int32_t lpad0;        ///< keeps the header 16-byte aligned
    std::int32_t lpad1;        ///< likewise
    std::int32_t lpad2;        ///< likewise
};
static_assert(sizeof(BlockHeader) % 16 == 0, "the voices behind the header stay 16-byte aligned");
static_assert(sizeof(CloudVoice) % 16 == 0, "CloudVoice is uploaded as an array");
static_assert(sizeof(EnvLane) == 64, "EnvLane keeps its layout");
static_assert(sizeof(EnvShape) % 16 == 0, "EnvShape is uploaded as an array");

/**
 * @brief An integer hash with good avalanche (lowbias32, C. Wellons); the dice of the cloud.
 * @param x  any value
 * @return   its hash
 */
__host__ __device__ __forceinline__ std::uint32_t hash32(std::uint32_t x)
{
    x ^= x >> 16; x *= 0x7feb352dU;
    x ^= x >> 15; x *= 0x846ca68bU;
    x ^= x >> 16;
    return x;
}

/**
 * @brief A uniform number in [0, 1) for one partial of one note.
 * @param seed  the note's seed
 * @param p     the partial within the voice
 * @param salt  which of the partial's dice (offset, drift rates, phases, pan, envelopes, ...)
 * @return      0 .. 1
 */
__device__ __forceinline__ float hash01(std::uint32_t seed, std::uint32_t p, std::uint32_t salt)
{
    return static_cast<float>(hash32(seed * 0x9E3779B9U ^ hash32(p * 0x85EBCA6BU + salt * 0xC2B2AE35U)) >> 8) * (1.0f / 16777216.0f);
}

/**
 * @brief A sine of a slow oscillator at `rate` Hz and phase offset `phase` (cycles) at time `t`,
 *        reduced to one cycle first so the fast __sinf stays accurate after hours.
 * @param rate   Hz
 * @param t      seconds
 * @param phase  0 .. 1
 * @return       -1 .. 1
 */
__device__ __forceinline__ float slowSine(float rate, float t, float phase)
{
    float x = fmaf(rate, t, phase);
    x -= floorf(x);
    return __sinf(6.28318531f * x);
}

/**
 * @brief The bend of an envelope segment.
 * @param x  0 .. 1 through the segment
 * @param c  -1 .. 1: 0 straight, positive starts slowly, negative starts fast
 * @return   0 .. 1
 */
__device__ __forceinline__ float bend(float x, float c)
{
    if (c > 0.001f) return __powf(x, 1.0f + 5.0f * c);
    if (c < -0.001f) return 1.0f - __powf(1.0f - x, 1.0f - 5.0f * c);
    return x;
}

/**
 * @brief A shape's held course (its start, then its loop forever) at shape time `tau`.
 * @param s      the shape
 * @param tau    seconds of shape time (already scaled by the partial's rate)
 * @param phase  loop cycles added inside the loop
 * @return       the level
 */
__device__ float shapeHeld(const EnvShape& s, float tau, float phase)
{
    const int n = s.count;
    if (n <= 0) return 0.0f;
    const bool loop = s.loopStart >= 0 && s.loopEnd > s.loopStart && s.loopEnd < n;
    if (loop) {
        const float t0 = s.p[s.loopStart].time;
        const float len = s.p[s.loopEnd].time - t0;
        if (len > 0.0f && tau >= t0) {
            float x = (tau - t0) / len + phase;
            x -= floorf(x);
            tau = t0 + x * len;
        }
    }
    if (tau <= s.p[0].time) return s.p[0].level;
    const int last = loop ? s.loopEnd : n - 1;
    if (tau >= s.p[last].time) return s.p[last].level;
    int i = 0;
    while (i + 1 < last && s.p[i + 1].time <= tau) ++i;
    const float span = s.p[i + 1].time - s.p[i].time;
    const float x = span > 0.0f ? (tau - s.p[i].time) / span : 1.0f;
    return s.p[i].level + (s.p[i + 1].level - s.p[i].level) * bend(x, s.p[i + 1].curve);
}

/**
 * @brief A shape's level: its held course, or after the release its tail from wherever the loop was.
 * @param s       the shape
 * @param tau     shape time now
 * @param tauOff  shape time of the release, negative while held
 * @param phase   loop cycles added inside the loop
 * @return        the level
 */
__device__ float shapeValue(const EnvShape& s, float tau, float tauOff, float phase)
{
    const bool tail = s.loopStart >= 0 && s.loopEnd > s.loopStart && s.loopEnd + 1 < s.count;
    if (tauOff < 0.0f || !tail || tau < tauOff) return shapeHeld(s, tau, phase);
    float prevX = 0.0f;
    float prevL = shapeHeld(s, tauOff, phase);
    const float x = tau - tauOff;
    const float tEnd = s.p[s.loopEnd].time;
    for (int j = s.loopEnd + 1; j < s.count; ++j) {
        const float xj = s.p[j].time - tEnd;
        if (x < xj) {
            const float span = xj - prevX;
            const float u = span > 0.0f ? (x - prevX) / span : 1.0f;
            return prevL + (s.p[j].level - prevL) * bend(u, s.p[j].curve);
        }
        prevX = xj;
        prevL = s.p[j].level;
    }
    return prevL;
}

/** @brief Everything one partial's envelope lanes are read against. */
struct LaneContext {
    float src[kMapSources];     ///< the partial's map sources, 0 .. 1
    std::uint32_t seed;         ///< the note's seed
    std::uint32_t p;            ///< the partial within the voice
    float t;                    ///< the note's time at the end of the block, seconds
    float releaseAt;            ///< the note's release time, negative while held
    const EnvShape* shapes;     ///< the bank
    int shapeCount;             ///< shapes in it
    const float* land;          ///< the landscape
    int landColumns;            ///< its columns (0 = none)
    int landRows;               ///< its rows
    float landRowSeconds;       ///< seconds per row
    float landColumn;           ///< the partial's column, 0 .. landColumns-1
};

/**
 * @brief The landscape at one column and time, interpolated both ways; it loops over its rows.
 * @param c      the context (landscape and column)
 * @param tau    lane time, seconds
 * @param phase  loop cycles added
 * @return       0 .. 1
 */
__device__ float landValue(const LaneContext& c, float tau, float phase)
{
    const float rows = static_cast<float>(c.landRows);
    float r = tau / c.landRowSeconds + phase * rows;
    r -= floorf(r / rows) * rows;
    int r0 = min(static_cast<int>(r), c.landRows - 1);
    const float fr = r - static_cast<float>(r0);
    const int r1 = r0 + 1 == c.landRows ? 0 : r0 + 1;
    const int c0 = min(static_cast<int>(c.landColumn), c.landColumns - 1);
    const float fc = c.landColumn - static_cast<float>(c0);
    const int c1 = min(c0 + 1, c.landColumns - 1);
    const float* a = c.land + static_cast<size_t>(r0) * c.landColumns;
    const float* b = c.land + static_cast<size_t>(r1) * c.landColumns;
    const float top = a[c0] + (a[c1] - a[c0]) * fc;
    const float bottom = b[c0] + (b[c1] - b[c0]) * fc;
    return top + (bottom - top) * fr;
}

/**
 * @brief One lane's value for one partial: which shape, how fast, where in its loop, how late.
 * @param L        the lane
 * @param c        the partial's context
 * @param salt     the lane's dice (each lane its own)
 * @param bipolar  pitch and pan read the landscape as -1 .. 1
 * @param depth    receives the partial's depth (amount, tilted by its source)
 * @return         the envelope's level
 */
__device__ float laneValue(const EnvLane& L, const LaneContext& c, std::uint32_t salt, bool bipolar, float& depth)
{
    // Random is its own die for every parameter of every lane, so a random shape and a random
    // depth do not come out correlated.
    auto src = [&](MapSource m, std::uint32_t which) {
        const int i = static_cast<int>(m);
        if (m == MapSource::Random) return hash01(c.seed, c.p, salt + 2 + which);
        return i > 0 && i < kMapSources ? c.src[i] : 0.0f;
    };
    const float sDepth = src(L.depthBy, 0);
    depth = L.amount * (L.depthTilt >= 0.0f ? 1.0f - L.depthTilt + L.depthTilt * sDepth : 1.0f + L.depthTilt * sDepth);
    const float rate = L.rate * fmaxf(0.01f, 1.0f + L.rateSpread * src(L.rateBy, 1))
                     * exp2f(L.chaos * (2.0f * hash01(c.seed, c.p, salt) - 1.0f));
    const float phase = L.phase * src(L.phaseBy, 2) + L.chaos * hash01(c.seed, c.p, salt + 1);
    const float sd = src(L.delayBy, 3);
    const float delay = L.delay >= 0.0f ? L.delay * sd : -L.delay * (1.0f - sd);
    const float tau = rate * fmaxf(0.0f, c.t - delay);
    if (L.landscape && c.landColumns > 0 && c.landRows > 0) {
        const float v = landValue(c, tau, phase);
        return bipolar ? 2.0f * v - 1.0f : v;
    }
    if (c.shapeCount <= 0) return 0.0f;
    const int spread = max(1, L.shapeSpread);
    const int pick = min(spread - 1, static_cast<int>(src(L.shapeBy, 4) * static_cast<float>(spread)));
    const int idx = min(max(L.shape + pick, 0), c.shapeCount - 1);
    const float tauOff = c.releaseAt >= 0.0f ? rate * fmaxf(0.0f, c.releaseAt - delay) : -1.0f;
    return shapeValue(c.shapes[idx], tau, tauOff, phase);
}

/** @brief The formants of u, o, a, e, i (Hz), three each. */
__constant__ float kFormants[5][3] = {
    { 350.0f, 600.0f, 2400.0f },
    { 450.0f, 800.0f, 2830.0f },
    { 800.0f, 1150.0f, 2900.0f },
    { 400.0f, 2000.0f, 2550.0f },
    { 270.0f, 2140.0f, 2950.0f },
};

/**
 * @brief A two-pole band pass's magnitude, unity at its centre.
 * @param r  frequency over centre
 * @param q  Q
 * @return   the gain
 */
__device__ inline float bandPass(float r, float q)
{
    const float a = 1.0f - r * r, b = r / q;
    return b * rsqrtf(a * a + b * b + 1.0e-12f);
}

/**
 * @brief The voice filter's response at one partial: the analogue prototype's magnitude.
 * @param model   FilterModel
 * @param f       the partial, Hz
 * @param cutoff  Hz
 * @param q       resonance as a Q (Peak: the bell's gain; Formant: the formants' Q)
 * @param vowel   Formant: 0 .. 4, u o a e i
 * @return        the gain
 */
__device__ float filterGain(int model, float f, float cutoff, float q, float vowel)
{
    if (model <= 0) return 1.0f;
    const float r = f / fmaxf(cutoff, 1.0f);
    const float a = 1.0f - r * r, b = r / fmaxf(q, 0.1f);
    const float den2 = a * a + b * b + 1.0e-12f;
    switch (static_cast<FilterModel>(model)) {
    case FilterModel::LP6:   return rsqrtf(1.0f + r * r);
    case FilterModel::LP12:  return rsqrtf(den2);
    case FilterModel::LP24:  return 1.0f / den2;
    case FilterModel::HP12:  return r * r * rsqrtf(den2);
    case FilterModel::BP12:  return b * rsqrtf(den2);
    case FilterModel::Notch: return fabsf(a) * rsqrtf(den2);
    case FilterModel::Peak:  return 1.0f + (fmaxf(q, 1.0f) - 1.0f) * bandPass(r, 2.0f);
    case FilterModel::Formant: {
        const float v = fminf(fmaxf(vowel, 0.0f), 4.0f);
        const int i0 = min(static_cast<int>(v), 3);
        const float fr = v - static_cast<float>(i0);
        float g = 0.02f;
        const float w[3] = { 1.0f, 0.5f, 0.25f };
        for (int k = 0; k < 3; ++k) {
            const float fk = kFormants[i0][k] * exp2f(fr * log2f(kFormants[i0 + 1][k] / kFormants[i0][k]));
            g += w[k] * bandPass(f / fk, q);
        }
        // Three band passes at unity pass little between them: measured over the factory chord the
        // formant presets sat 18 dB under the rest of the bank. Made up here, not in every preset.
        return 3.0f * g;
    }
    default: return 1.0f;
    }
}

/**
 * @brief One block of the field for one partial: its offset moved towards what pulls it and drawn back home.
 * @param F       the field
 * @param logHz   log2 of where the partial is now (its own place plus the offset)
 * @param off     the offset, cents
 * @param dt      the block's length, seconds
 * @param keep    log2 offset the partial keeps from what pulls it: its place in its cloud's bell,
 *                narrowed to 2 cents a unit -- partials gathered onto one pitch would add up to one
 *                sine of a frozen random level (measured: 3 dB down, the shimmer gone), a narrow
 *                cloud still beats and keeps its power
 * @return        the new offset, cents (at most an octave)
 */
__device__ float fieldStep(const SpectralField& F, float logHz, float off, float dt, float keep)
{
    if (F.mode == FieldMode::Off || F.strength <= 0.0f || F.count <= 0)
        return off * __expf(-fmaxf(F.spring, 1.0f) * dt);   // switched off: home within a second or so
    const float R = fmaxf(F.radius, 1.0f);
    const float maxStep = F.strength * dt;
    const float relax = 1.0f - __expf(-(F.strength / R) * dt);   // towards an attractor, never past it
    float step = 0.0f;
    if (F.mode == FieldMode::Scale) {
        // The nearest pitch of the scale (ascending): a binary search, then the neighbour.
        int lo = 0, hi = F.count - 1;
        while (hi - lo > 1) {
            const int mid = (lo + hi) >> 1;
            if (F.pos[mid] <= logHz) lo = mid; else hi = mid;
        }
        const float a = F.pos[lo], b = F.pos[hi];
        const float q = fabsf(logHz - a) <= fabsf(b - logHz) ? a : b;
        step = 1200.0f * (q + keep - logHz) * relax;
    } else {
        for (int k = 0; k < min(F.count, kFieldAttractors); ++k) {
            const float d = 1200.0f * (F.pos[k] + keep - logHz);
            const float u = d / R;
            const float w = __expf(-0.5f * u * u);
            step += F.mass[k] >= 0.0f ? F.mass[k] * w * d * relax : F.mass[k] * w * copysignf(maxStep, d);
        }
    }
    step = fminf(fmaxf(step, -maxStep), maxStep);
    return fminf(fmaxf((off + step) * __expf(-F.spring * dt), -1200.0f), 1200.0f);
}

/** @brief A frequency's band on the hall's log axis (kHallBins over 20 Hz .. 20 kHz). @param f Hz @return the band */
__device__ __forceinline__ int hallBin(float f)
{
    return min(max(static_cast<int>(log2f(fmaxf(f, 20.0f) * (1.0f / 20.0f)) * (static_cast<float>(kHallBins) / 9.96578428f)), 0), kHallBins - 1);
}

/**
 * @brief One block of the Orbit field for one partial: a charged particle in pitch and side, drawn to
 *        the nearest pitch of the scale (a well, its place in its bell kept) and bent by a magnetic
 *        field -- the velocity turned by the field's angle each block (exactly: the turn never adds
 *        energy), the well's pull and the damping, and a kick of white noise, so that the orbits keep
 *        about half the radius (a damped oscillator's mean square, sigma^2 / (2 g k)). Fast small
 *        circles ride a slow circling of the well.
 * @param F        the field (orbitWell, orbitField, orbitDamp; radius: cents to a full side)
 * @param prev     the last block's (offset, side, their velocities) per partial, cents and cents a second
 * @param next     this block's
 * @param me       the partial's index in the tables
 * @param logHome  log2 of the partial's own place
 * @param keep     log2 offset kept from the well (its place in its cloud's bell)
 * @param now      the offset it has now (taken over when the orbit is fresh)
 * @param seed     the note's dice
 * @param up       the partial
 * @param block    the block's index (the kicks' dice)
 * @param restart  a new note: at home, at rest
 * @param dt       the block's length, seconds
 * @param side     receives its shift across the field, -0.6 .. 0.6
 * @return         the new offset, cents
 */
__device__ float orbitStep(const SpectralField& F, const float4* prev, float4* next, size_t me, float logHome, float keep, float now,
                           std::uint32_t seed, std::uint32_t up, std::uint32_t block, bool restart, float dt, float& side)
{
    float4 s = restart ? make_float4(0.0f, 0.0f, 0.0f, 0.0f) : (F.orbitFresh ? make_float4(now, 0.0f, 0.0f, 0.0f) : prev[me]);
    float well = 0.0f;
    if (F.count > 0) {
        const float at = logHome + s.x * (1.0f / 1200.0f);
        int lo = 0, hi = F.count - 1;
        while (hi - lo > 1) {
            const int mid = (lo + hi) >> 1;
            if (F.pos[mid] <= at) lo = mid; else hi = mid;
        }
        const float q = fabsf(at - F.pos[lo]) <= fabsf(F.pos[hi] - at) ? F.pos[lo] : F.pos[hi];
        well = 1200.0f * (q + keep - logHome);
    }
    const float R = fmaxf(F.radius, 1.0f);
    const float k = F.orbitWell * F.orbitWell, g = fmaxf(F.orbitDamp, 0.0f);
    float rs, rc;
    __sincosf(F.orbitField * dt, &rs, &rc);
    float vx = s.z * rc + s.w * rs, vy = s.w * rc - s.z * rs;   // the field turns the velocity
    const float sigma = 0.5f * R * sqrtf(2.0f * fmaxf(g, 0.01f) * k) * sqrtf(dt);
    const float kx = (hash01(seed ^ (block * 0x9E3779B9U), up, 20) - 0.5f) * 3.4641016f;   // unit variance
    const float ky = (hash01(seed ^ (block * 0x85EBCA6BU), up, 21) - 0.5f) * 3.4641016f;
    vx += dt * (-k * (s.x - well) - g * vx) + sigma * kx;
    vy += dt * (-k * s.y - g * vy) + sigma * ky;
    const float x = fminf(fmaxf(s.x + vx * dt, -1200.0f), 1200.0f);
    const float y = fminf(fmaxf(s.y + vy * dt, -2.0f * R), 2.0f * R);
    next[me] = make_float4(x, y, vx, vy);
    side = 0.6f * fminf(fmaxf(y / R, -1.0f), 1.0f);
    return x;
}

/**
 * @brief The hall's ear and memory, one CUDA block: the partials' amplitudes gathered over the octaves
 *        this block (whole numbers, cleared here) as a share of the whole, a little blurred; the spectrum's
 *        density followed over a third of a second (the partials' own clocks read it); and, while
 *        anything sounds and Grow is not nothing, the hall's memory a step of Memory towards it.
 *        Summed in a fixed order.
 * @param hdr      the header
 * @param hist     per band the amplitude gathered this block (fixed point, 1e-12)
 * @param memory   per band the hall's memory, 1 the average
 * @param density  per band the spectrum's density, 1 the average
 */
__global__ void __launch_bounds__(kHallBins) hallKernel(const BlockHeader* hdr, unsigned long long* hist, float* memory, float* density)
{
    const ModalHall& mh = hdr->modal;
    if (!mh.listen) return;
    __shared__ float e[kHallBins];
    __shared__ float red[kHallBins];
    const int b = static_cast<int>(threadIdx.x);
    const float h = static_cast<float>(hist[b]) * 1.0e-12f;
    hist[b] = 0ULL;
    red[b] = h;
    __syncthreads();
    for (int s = kHallBins / 2; s > 0; s >>= 1) {
        if (b < s) red[b] += red[b + s];
        __syncthreads();
    }
    const float total = red[0];
    e[b] = total > 1.0e-12f ? h / total * static_cast<float>(kHallBins) : 0.0f;
    __syncthreads();
    const float here = 0.25f * e[max(b - 1, 0)] + 0.5f * e[b] + 0.25f * e[min(b + 1, kHallBins - 1)];
    if (mh.densityCoef > 0.0f) density[b] += (here - density[b]) * mh.densityCoef;
    if (mh.memoryCoef > 0.0f && total > 1.0e-9f) memory[b] += (here - memory[b]) * mh.memoryCoef;
}

/**
 * @brief One partial's complex gain at each ear: the head's shadow, the pinna and the time round
 *        the head (Brown and Duda 1998; Woodworth), the pair normalised to the pan law's power.
 * @param B      the binaural settings
 * @param place  the partial's place, -1 .. 1 (its azimuth over the span, plus the orbit)
 * @param u      its own die, 0 .. 1 (its elevation)
 * @param f      its frequency, Hz
 * @param gl     receives the left ear's gain (real, imaginary)
 * @param gr     receives the right ear's gain
 */
__device__ void earGains(const Binaural& B, float place, float u, float f, float* gl, float* gr)
{
    const float az = place * 1.5707963f * (1.0f + fminf(fmaxf(B.span, 0.0f), 1.0f)) + B.angle;
    const float el = fminf(fmaxf(B.height, 0.0f), 1.0f) * 1.0471976f * (2.0f * u - 1.0f);
    float sa, ca, se, ce;
    sincosf(az, &sa, &ca);
    sincosf(el, &se, &ce);
    // The pinna, the same at both ears here: five reflections whose times follow azimuth and
    // elevation (samples at 44.1 kHz), normalised by its mean power (1 + sum of rho^2 = 2.625).
    const float rho[5] = { 0.5f, -1.0f, 0.5f, -0.25f, 0.25f };
    const float A[5] = { 1.0f, 5.0f, 5.0f, 5.0f, 5.0f };
    const float Bn[5] = { 2.0f, 4.0f, 7.0f, 11.0f, 13.0f };
    const float D[5] = { 1.0f, 0.5f, 0.5f, 0.5f, 0.5f };
    const float half = fabsf(cosf(0.5f * az));
    float pr = 1.0f, pim = 0.0f;
    for (int k = 0; k < 5; ++k) {
        const float tk = A[k] * half * sinf(D[k] * (1.5707963f - el)) + Bn[k];
        float s, c;
        sincosf(-6.2831853f * f * tk * (1.0f / 44100.0f), &s, &c);
        pr += rho[k] * c;
        pim += rho[k] * s;
    }
    pr *= 0.6172134f;
    pim *= 0.6172134f;
    const float x = sa * ce;   // towards the right ear
    float g[2][2];
    for (int e = 0; e < 2; ++e) {   // 0 left, 1 right
        const float beta = acosf(fminf(fmaxf(e == 1 ? x : -x, -1.0f), 1.0f));   // 0 facing this ear .. pi behind the head from it
        const float alpha = 1.05f + 0.95f * cosf(beta * (180.0f / 150.0f));
        const float w = f * 8.0143e-4f;   // pi f a / c, a = 8.75 cm, c = 343 m/s
        const float den = 1.0f / (1.0f + w * w);
        const float hr = (1.0f + alpha * w * w) * den, hi = (alpha - 1.0f) * w * den;
        const float tau = beta < 1.5707963f ? -2.551e-4f * cosf(beta) : 2.551e-4f * (beta - 1.5707963f);
        float s, c;
        sincosf(-6.2831853f * f * tau, &s, &c);
        // head shadow x pinna x delay
        const float r1 = hr * pr - hi * pim, i1 = hr * pim + hi * pr;
        g[e][0] = r1 * c - i1 * s;
        g[e][1] = r1 * s + i1 * c;
    }
    const float norm = rsqrtf(g[0][0] * g[0][0] + g[0][1] * g[0][1] + g[1][0] * g[1][0] + g[1][1] * g[1][1] + 1.0e-20f);
    gl[0] = g[0][0] * norm; gl[1] = g[0][1] * norm;
    gr[0] = g[1][0] * norm; gr[1] = g[1][1] * norm;
}

/**
 * @brief What a partial reads from a cell of the living field, 0 .. 1: Lenia's value, or the
 *        Turing field's second chemical (V, which lives under 0.4) scaled up.
 * @param mode  the field's mode
 * @param c     the cell (Lenia: x; Turing: U, V)
 * @return      0 .. 1
 */
__device__ __forceinline__ float lifeValue(LifeMode mode, float2 c)
{
    if (mode == LifeMode::Turing) return fminf(c.y * 3.0f, 1.0f);
    if (mode == LifeMode::Rain) return 0.5f + 0.5f * tanhf(3.0f * c.x);   // the water's height about its rest
    return c.x;
}

// ---- Animated pictures (KaleidoscopeEnhanced's Scene2D/Chain*.frag, its continuous transforms) ----

/** @brief Smooth value noise, 0 .. 1 (corners hashed). @param x across @param y along @return the noise */
__device__ float pictureNoise(float x, float y)
{
    const float ix = floorf(x), iy = floorf(y), fx = x - ix, fy = y - iy;
    const float sx = fx * fx * (3.0f - 2.0f * fx), sy = fy * fy * (3.0f - 2.0f * fy);
    auto h = [](float a, float b) {
        return static_cast<float>(hash32(static_cast<std::uint32_t>(static_cast<int>(a)) * 0x8DA6B343U ^ static_cast<std::uint32_t>(static_cast<int>(b)) * 0xD8163841U) >> 8)
             * (1.0f / 16777216.0f);
    };
    const float a = h(ix, iy), b = h(ix + 1.0f, iy), c = h(ix, iy + 1.0f), d = h(ix + 1.0f, iy + 1.0f);
    return a + (b - a) * sx + (c - a) * sy + (a - b - c + d) * sx * sy;
}

/** @brief Three octaves of noise, 0 .. 1 (the shaders' fbm3). @param x across @param y along @return the noise */
__device__ float pictureFbm(float x, float y)
{
    float v = 0.0f, a = 0.5f;
    for (int i = 0; i < 3; ++i) {
        v += a * pictureNoise(x, y);
        const float nx = 1.6f * x + 1.2f * y + 2.3f, ny = -1.2f * x + 1.6f * y + 2.3f;
        x = nx;
        y = ny;
        a *= 0.5f;
    }
    return v / 0.875f;
}

/** @brief The shaders' mirror repeat: identity on [0, 1], mirrored beyond, so every fold is seamless. @param u a coordinate @return 0 .. 1 */
__device__ __forceinline__ float pictureMirror(float u)
{
    const float f = u * 0.5f - floorf(u * 0.5f);
    return 1.0f - fabsf(f * 2.0f - 1.0f);
}

/**
 * @brief The landscape as a picture at (u, v), mirror-repeated: u its time (left to right), v its
 *        frequency (low at the bottom); its brightness on the landscapes' 40 dB scale.
 * @param hdr   the header (the landscape's size)
 * @param land  the landscape
 * @param u     across
 * @param v     up
 * @return      0 .. 1
 */
__device__ float pictureAt(const BlockHeader* hdr, const float* land, float u, float v)
{
    const int cols = hdr->landColumns, rows = hdr->landRows;
    if (cols < 2 || rows < 2) return 0.5f;
    const float fr = pictureMirror(u) * static_cast<float>(rows - 1), fc = pictureMirror(v) * static_cast<float>(cols - 1);
    const int r0 = min(static_cast<int>(fr), rows - 2), c0 = min(static_cast<int>(fc), cols - 2);
    const float ar = fr - static_cast<float>(r0), ac = fc - static_cast<float>(c0);
    const float* p = land + static_cast<size_t>(r0) * cols + c0;
    const float top = p[0] + (p[1] - p[0]) * ac, bottom = p[cols] + (p[cols + 1] - p[cols]) * ac;
    const float a = top + (bottom - top) * ar;
    return fminf(fmaxf(1.0f + 20.0f * log10f(fmaxf(a, 1.0e-6f)) / 40.0f, 0.0f), 1.0f);
}

/**
 * @brief One cell of an animated picture. The field is the screen (twice as wide as high); the
 *        transforms are the shaders' (tKaleido, tTunnel, tSpiral, tWarp), the feedback and the
 *        rain read the field before this step.
 * @param hdr   the header (its life: time, angle, sides, character)
 * @param land  the landscape
 * @param src   the field before this step
 * @param x     the cell across
 * @param y     the cell along
 * @return      the cell (x: the value, or Rain's height; y: Rain's height before)
 */
__device__ float2 pictureCell(const BlockHeader* hdr, const float* land, const float2* src, int x, int y)
{
    const LifeField& L = hdr->life;
    const float px = (static_cast<float>(x) + 0.5f) / static_cast<float>(kLifeW), py = (static_cast<float>(y) + 0.5f) / static_cast<float>(kLifeH);
    const float dx = (px - 0.5f) * 2.0f, dy = py - 0.5f;   // centred, the aspect kept
    const float r = fmaxf(sqrtf(dx * dx + dy * dy), 1.0e-4f), a = atan2f(dy, dx);
    const float T = L.time, ch = L.character, S = fmaxf(L.sides, 2.0f);
    const float kPi = 3.14159265f;
    switch (L.mode) {
    case LifeMode::Kaleidoscope: {
        // tKaleido round a centre that wanders slowly through the picture, the picture flowing under it.
        const float cx = 0.5f + 0.2f * __sinf(0.0107f * L.seconds), cy = 0.5f + 0.2f * __cosf(0.0131f * L.seconds);
        const float sec = 2.0f * kPi / S;
        const float w = fabsf(a - sec * floorf(a / sec) - 0.5f * sec) + L.angle;
        float u = cx + 0.6f * r * __cosf(w) + 0.05f * T, v = cy + 0.6f * r * __sinf(w);
        u += ch * 0.15f * (pictureFbm(3.0f * u + T, 3.0f * v) - 0.5f);
        v += ch * 0.15f * (pictureFbm(3.0f * u + 5.2f, 3.0f * v - T) - 0.5f);
        return make_float2(pictureAt(hdr, land, u, v), 0.0f);
    }
    case LifeMode::Tunnel: {
        // tTunnel: the angle round (folded into Sides mirror wedges), the depth 1/r along, flying in.
        const float sec = 2.0f * kPi / S;
        const float w = fabsf(a + L.angle - sec * floorf((a + L.angle) / sec) - 0.5f * sec) / sec;
        const float depth = 0.25f / r + 0.3f * T;
        return make_float2(pictureAt(hdr, land, w + ch * 0.3f * depth, depth), 0.0f);
    }
    case LifeMode::Spiral: {
        // tSpiral: log-polar, the angle sheared into the radius -- an endless zoom along Sides arms.
        const float arms = 2.0f * floorf(S * 0.5f);
        const float u = logf(r) * (0.5f + 0.5f * ch) - 0.3f * T + (a + L.angle) / kPi;
        const float v = (a + L.angle) / kPi * arms * 0.5f;
        return make_float2(pictureAt(hdr, land, u, v), 0.0f);
    }
    case LifeMode::Flow: {
        // tWarp, twice: the picture carried on a current of noise that itself drifts.
        float u = px + 0.05f * T * __cosf(L.angle), v = py + 0.05f * T * __sinf(L.angle);
        const float s = 0.15f + 0.45f * ch;
        for (int k = 0; k < 2; ++k) {
            const float wu = pictureFbm(2.5f * u + 0.3f * T, 2.5f * v) - 0.5f, wv = pictureFbm(2.5f * u + 5.2f, 2.5f * v - 0.3f * T) - 0.5f;
            u += s * wu;
            v += s * wv;
        }
        return make_float2(pictureAt(hdr, land, u, v), 0.0f);
    }
    case LifeMode::Feedback: {
        // Video feedback: this cell takes the field from a point turned and drawn towards the middle
        // (so everything spirals outward), fading a little, and the picture seeps in.
        // A step every block (375 a second): it zooms out by 0.15 .. 0.6 a second and forgets in about three.
        const float step = 0.0004f * (1.0f + 3.0f * ch), rot = L.turnStep;
        const float cs = __cosf(rot), sn = __sinf(rot);
        const float qx = (dx * cs - dy * sn) / (1.0f + step), qy = (dx * sn + dy * cs) / (1.0f + step);
        const float fx = (qx * 0.5f + 0.5f) * static_cast<float>(kLifeW) - 0.5f, fy = (qy + 0.5f) * static_cast<float>(kLifeH) - 0.5f;
        const int x0 = static_cast<int>(floorf(fx)), y0 = static_cast<int>(floorf(fy));
        const float ax = fx - static_cast<float>(x0), ay = fy - static_cast<float>(y0);
        auto at = [&](int xx, int yy) { return src[(yy & (kLifeH - 1)) * kLifeW + (xx & (kLifeW - 1))].x; };
        const float back = (at(x0, y0) * (1.0f - ax) + at(x0 + 1, y0) * ax) * (1.0f - ay) + (at(x0, y0 + 1) * (1.0f - ax) + at(x0 + 1, y0 + 1) * ax) * ay;
        const float seep = pictureAt(hdr, land, px + 0.02f * T, py + 0.5f * __sinf(0.01f * L.seconds));
        return make_float2(fminf(fmaxf(0.997f * back + 0.003f * seep, 0.0f), 1.0f), 0.0f);
    }
    case LifeMode::Rain: {
        // A damped 2D wave (leapfrog, the five-point Laplacian), struck at chance cells.
        const float2 c = src[y * kLifeW + x];
        const float l = src[y * kLifeW + ((x - 1) & (kLifeW - 1))].x, rr = src[y * kLifeW + ((x + 1) & (kLifeW - 1))].x;
        const float u = src[((y - 1) & (kLifeH - 1)) * kLifeW + x].x, d = src[((y + 1) & (kLifeH - 1)) * kLifeW + x].x;
        // A step every block: the rings cross the field in about three seconds; Character keeps them ringing.
        const float damp = 0.0005f + 0.004f * (1.0f - ch);
        float next = (2.0f * c.x - (1.0f - damp) * c.y + 0.05f * (l + rr + u + d - 4.0f * c.x)) / (1.0f + damp);
        const float roll = static_cast<float>(hash32(static_cast<std::uint32_t>(y * kLifeW + x) * 0x9E3779B9U ^ hdr->blockIndex * 0x85EBCA6BU) >> 8)
                         * (1.0f / 16777216.0f);
        if (roll < L.drops) next += 1.5f;
        return make_float2(fminf(fmaxf(next, -4.0f), 4.0f), c.x);
    }
    case LifeMode::Starfield: {
        // Stars on a log-polar grid streaming outwards; three layers of different density.
        float v = 0.0f;
        for (int k = 0; k < 3; ++k) {
            const float density = 6.0f * S * (1.0f + static_cast<float>(k));
            const float gu = (a + L.angle * (1.0f + 0.3f * k)) / (2.0f * kPi) * density, gv = (logf(r) - T * (0.6f + 0.4f * k)) * density * 0.25f;
            const float cu = floorf(gu), cv = floorf(gv);
            const std::uint32_t hs = hash32(static_cast<std::uint32_t>(static_cast<int>(cu)) * 0x27D4EB2DU ^ static_cast<std::uint32_t>(static_cast<int>(cv)) * 0x165667B1U ^ (k * 0x9E3779B9U));
            if ((hs & 255U) < static_cast<std::uint32_t>(40 + 60 * ch)) {
                const float su = gu - cu - (0.2f + 0.6f * static_cast<float>((hs >> 8) & 255U) / 255.0f), sv = gv - cv - (0.2f + 0.6f * static_cast<float>((hs >> 16) & 255U) / 255.0f);
                v = fmaxf(v, __expf(-(su * su + sv * sv) * 25.0f) * fminf(r * 3.0f, 1.0f));
            }
        }
        // A faint glow under the stars: a sky of nothing but points is so sparse that no make-up can
        // keep its level (measured 2 dB under), and the partials between them would all but vanish.
        return make_float2(0.12f + 0.88f * v, 0.0f);
    }
    case LifeMode::Moire: {
        // Two line gratings, nearly alike and turning nearly together: where they agree and where they
        // cancel makes broad bands that wander slowly (the moire), far wider than the lines.
        const float f1 = 2.0f * S, f2 = f1 * (1.0f + 0.02f + 0.08f * ch);
        const float a1 = L.angle, a2 = 0.9f * L.angle + 0.08f;
        const float g1 = 0.5f + 0.5f * __cosf(2.0f * kPi * f1 * (dx * __cosf(a1) + dy * __sinf(a1)) + T);
        const float g2 = 0.5f + 0.5f * __cosf(2.0f * kPi * f2 * (dx * __cosf(a2) + dy * __sinf(a2)) - 0.7f * T);
        return make_float2(g1 * g2, 0.0f);
    }
    default:
        return make_float2(0.0f, 0.0f);
    }
}

/** @brief The field buffer that holds the field after this block's steps. @param hdr the header @param a buffer A @param b buffer B @return the current one */
__device__ __forceinline__ const float2* lifeCurrent(const BlockHeader* hdr, const float2* a, const float2* b)
{
    return ((hdr->lifeParity + hdr->life.steps) & 1) ? b : a;
}

/**
 * @brief Seeds the living field when asked (hdr->life.reseed): Lenia a few patches of noise, the
 *        Turing field U = 1, V = 0 with a few squares of V -- from a hash of the cell and the salt.
 * @param hdr  the header
 * @param a    buffer A
 * @param b    buffer B (the one before the steps is seeded)
 */
__global__ void lifeSeedKernel(const BlockHeader* hdr, float2* a, float2* b)
{
    const LifeField& L = hdr->life;
    if (!L.reseed || L.mode == LifeMode::Off) return;
    const int i = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= kLifeCells) return;
    float2* dst = (hdr->lifeParity & 1) ? b : a;
    const int x = i & (kLifeW - 1), y = i >> 8;
    const std::uint32_t salt = static_cast<std::uint32_t>(L.salt) * 0x9E3779B9U;
    float2 c = L.mode == LifeMode::Turing ? make_float2(1.0f, 0.0f) : make_float2(0.0f, 0.0f);
    if (lifeIsPicture(L.mode)) { dst[i] = c; return; }   // the pictures are computed, Rain starts from still water
    // Turing dense (eighty small squares: a sparse field is uneven where the partials read it --
    // measured 3 dB over while it grew from fourteen), Lenia a few large patches of noise.
    const int patches = L.mode == LifeMode::Turing ? 80 : 8;
    for (int k = 0; k < patches; ++k) {
        const std::uint32_t h = hash32(salt + 7919U * static_cast<std::uint32_t>(k));
        const int cx = static_cast<int>(h & (kLifeW - 1)), cy = static_cast<int>((h >> 8) & (kLifeH - 1));
        const int r = L.mode == LifeMode::Turing ? 3 + static_cast<int>((h >> 16) % 5U) : 10 + static_cast<int>((h >> 16) % 9U);
        int dx = abs(x - cx), dy = abs(y - cy);
        dx = min(dx, kLifeW - dx);
        dy = min(dy, kLifeH - dy);
        if (dx > r || dy > r) continue;
        const float noise = static_cast<float>(hash32(salt ^ static_cast<std::uint32_t>(i) * 2654435761U) >> 8) * (1.0f / 16777216.0f);
        if (L.mode == LifeMode::Turing) c = make_float2(0.5f, 0.25f + 0.05f * noise);
        else c.x = fmaxf(c.x, noise);
    }
    dst[i] = c;
}

/**
 * @brief One step of the living field, cell by cell, from one buffer into the other (which one by
 *        the parity and the step's number); a step this block does not have does nothing.
 *
 * Lenia (Chan 2019): the ring kernel's weighted sum u of the neighbourhood, the growth
 * 2 exp(-(u - mu)^2 / 2 sigma^2) - 1, a tenth of it added. Gray-Scott (Pearson 1993):
 * dU = Du lap U - U V^2 + F (1 - U), dV = Dv lap V + U V^2 - (F + k) V, Du 0.2097, Dv 0.105, one
 * unit of time a step, the five-point Laplacian.
 * @param hdr   the header
 * @param node  this step's number in the block, 0 or 1
 * @param a     buffer A
 * @param b     buffer B
 * @param land  the landscape (the animated pictures read it)
 */
__global__ void lifeStepKernel(const BlockHeader* hdr, int node, float2* a, float2* b, const float* land)
{
    const LifeField& L = hdr->life;
    if (L.mode == LifeMode::Off || L.mode == LifeMode::Fluid || L.mode == LifeMode::Islands || node >= L.steps) return;   // those two: their own kernels
    const int i = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= kLifeCells) return;
    const bool fromB = ((hdr->lifeParity + node) & 1) != 0;
    const float2* src = fromB ? b : a;
    float2* dst = fromB ? a : b;
    const int x = i & (kLifeW - 1), y = i >> 8;
    if (lifeIsPicture(L.mode)) {
        dst[i] = pictureCell(hdr, land, src, x, y);
    } else if (L.mode == LifeMode::Lenia) {
        constexpr int R = kLifeRadius, K = 2 * kLifeRadius + 1;
        float u = 0.0f;
        for (int dy = -R; dy <= R; ++dy) {
            const float2* row = src + static_cast<size_t>((y + dy) & (kLifeH - 1)) * kLifeW;
            for (int dx = -R; dx <= R; ++dx) {
                const float k = cLeniaKernel[(dy + R) * K + (dx + R)];
                if (k != 0.0f) u += k * row[(x + dx) & (kLifeW - 1)].x;
            }
        }
        const float d = (u - L.lenMu) / fmaxf(L.lenSigma, 1.0e-4f);
        const float g = 2.0f * __expf(-0.5f * d * d) - 1.0f;
        dst[i] = make_float2(fminf(fmaxf(src[i].x + 0.1f * g, 0.0f), 1.0f), 0.0f);
    } else {
        const float2 c = src[i];
        const float2 l = src[y * kLifeW + ((x - 1) & (kLifeW - 1))], r = src[y * kLifeW + ((x + 1) & (kLifeW - 1))];
        const float2 u = src[((y - 1) & (kLifeH - 1)) * kLifeW + x], d = src[((y + 1) & (kLifeH - 1)) * kLifeW + x];
        const float lapU = l.x + r.x + u.x + d.x - 4.0f * c.x, lapV = l.y + r.y + u.y + d.y - 4.0f * c.y;
        const float uvv = c.x * c.y * c.y;
        dst[i] = make_float2(fminf(fmaxf(c.x + 0.2097f * lapU - uvv + L.feed * (1.0f - c.x), 0.0f), 1.0f),
                             fminf(fmaxf(c.y + 0.105f * lapV + uvv - (L.feed + L.kill) * c.y, 0.0f), 1.0f));
    }
}

// ---- The Fluid: a lattice Boltzmann fluid (D2Q9, BGK) on the field's torus, with ink ----

__constant__ int cLbmX[9] = { 0, 1, 0, -1, 0, 1, -1, -1, 1 };                    ///< the lattice's directions, across
__constant__ int cLbmY[9] = { 0, 0, 1, 0, -1, 1, 1, -1, -1 };                    ///< and along
__constant__ float cLbmW[9] = { 4.0f / 9.0f, 1.0f / 9.0f, 1.0f / 9.0f, 1.0f / 9.0f, 1.0f / 9.0f,
                                1.0f / 36.0f, 1.0f / 36.0f, 1.0f / 36.0f, 1.0f / 36.0f };   ///< their weights

/** @brief One direction's equilibrium. @param k the direction @param rho the density @param ux the velocity across @param uy along @return f_eq */
__device__ __forceinline__ float lbmEq(int k, float rho, float ux, float uy)
{
    const float eu = static_cast<float>(cLbmX[k]) * ux + static_cast<float>(cLbmY[k]) * uy;
    return cLbmW[k] * rho * (1.0f + 3.0f * eu + 4.5f * eu * eu - 1.5f * (ux * ux + uy * uy));
}

/** @brief The shortest way round the torus. @param d a difference @param n the size @return it within -n/2 .. n/2 */
__device__ __forceinline__ float torusDelta(float d, float n)
{
    return d - n * rintf(d / n);
}

/**
 * @brief Seeds the Fluid when asked: at rest but for six eddies of either turn (from the salt).
 * @param hdr  the header
 * @param fa   the lattice, buffer A
 * @param fb   buffer B
 */
__global__ void fluidSeedKernel(const BlockHeader* hdr, float* fa, float* fb)
{
    const LifeField& L = hdr->life;
    if (!L.reseed || L.mode != LifeMode::Fluid) return;
    const int i = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= kLifeCells) return;
    float* dst = (hdr->lifeParity & 1) ? fb : fa;
    const int x = i & (kLifeW - 1), y = i >> 8;
    const std::uint32_t salt = static_cast<std::uint32_t>(L.salt) * 0x9E3779B9U;
    float ux = 0.0f, uy = 0.0f;
    for (int k = 0; k < 6; ++k) {
        const std::uint32_t h = hash32(salt + 104729U * static_cast<std::uint32_t>(k + 1));
        const float cx = static_cast<float>(h & (kLifeW - 1)), cy = static_cast<float>((h >> 8) & (kLifeH - 1));
        const float turn = (h >> 20) & 1U ? 1.0f : -1.0f;
        const float dx = torusDelta(static_cast<float>(x) - cx, static_cast<float>(kLifeW));
        const float dy = torusDelta(static_cast<float>(y) - cy, static_cast<float>(kLifeH));
        const float g = 0.006f * turn * __expf(-(dx * dx + dy * dy) * (1.0f / 128.0f));   // an eddy eight cells across
        ux -= g * dy;
        uy += g * dx;
    }
    for (int k = 0; k < 9; ++k) dst[static_cast<size_t>(k) * kLifeCells + i] = lbmEq(k, 1.0f, ux, uy);
}

/**
 * @brief One step of the Fluid, cell by cell: the lattice streamed (pulled from the neighbours),
 *        pushed by the jets (the velocity shifted by tau F / rho), relaxed to its equilibrium; the
 *        ink carried back along the flow, the jets' ink added, a little of it fading. Reads one
 *        buffer, writes the other, as the field's steps do.
 * @param hdr    the header
 * @param node   this step's number in the block, 0 or 1
 * @param fa     the lattice, buffer A
 * @param fb     buffer B
 * @param lifeA  the field (the ink), buffer A
 * @param lifeB  buffer B
 */
__global__ void fluidStepKernel(const BlockHeader* hdr, int node, float* fa, float* fb, float2* lifeA, float2* lifeB)
{
    const LifeField& L = hdr->life;
    if (L.mode != LifeMode::Fluid || node >= L.steps) return;
    const int i = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= kLifeCells) return;
    const bool fromB = ((hdr->lifeParity + node) & 1) != 0;
    const float* src = fromB ? fb : fa;
    float* dst = fromB ? fa : fb;
    const float2* ink = fromB ? lifeB : lifeA;
    float2* inkOut = fromB ? lifeA : lifeB;
    const int x = i & (kLifeW - 1), y = i >> 8;
    float f[9], rho = 0.0f, ux = 0.0f, uy = 0.0f;
    for (int k = 0; k < 9; ++k) {
        const int sx = (x - cLbmX[k]) & (kLifeW - 1), sy = (y - cLbmY[k]) & (kLifeH - 1);
        f[k] = src[static_cast<size_t>(k) * kLifeCells + sy * kLifeW + sx];
        rho += f[k];
        ux += f[k] * static_cast<float>(cLbmX[k]);
        uy += f[k] * static_cast<float>(cLbmY[k]);
    }
    if (!(rho > 0.05f && rho < 20.0f)) {   // a lattice that left the world: at rest again
        rho = 1.0f;
        ux = uy = 0.0f;
        for (int k = 0; k < 9; ++k) f[k] = lbmEq(k, 1.0f, 0.0f, 0.0f);
    } else {
        ux /= rho;
        uy /= rho;
    }
    // The jets: a little above the bottom, each where its note lies across, blowing up with its lean.
    float fx = 0.0f, fy = 0.0f, inkIn = 0.0f;
    for (int j = 0; j < kLifeJets; ++j) {
        const float amt = L.jetAmt[j];
        if (amt <= 0.0f) continue;
        const float dx = torusDelta(static_cast<float>(x) - L.jetX[j], static_cast<float>(kLifeW));
        const float dy = torusDelta(static_cast<float>(y) - 6.0f, static_cast<float>(kLifeH));
        const float g = amt * __expf(-(dx * dx + dy * dy) * (1.0f / 18.0f));   // three cells wide
        fx += g * L.jetSway[j];
        fy += g;
        inkIn += g;
    }
    const float tau = fmaxf(L.tau, 0.502f);
    float vx = ux + tau * L.push * fx / rho, vy = uy + tau * L.push * fy / rho;
    const float speed = sqrtf(vx * vx + vy * vy);
    if (speed > 0.12f) { vx *= 0.12f / speed; vy *= 0.12f / speed; }   // under a tenth of the lattice's sound: it stays a fluid
    for (int k = 0; k < 9; ++k) dst[static_cast<size_t>(k) * kLifeCells + i] = f[k] - (f[k] - lbmEq(k, rho, vx, vy)) / tau;
    // The ink, from where the flow brought it (bilinear on the torus).
    const float px = static_cast<float>(x) - ux, py = static_cast<float>(y) - uy;
    const int x0 = static_cast<int>(floorf(px)), y0 = static_cast<int>(floorf(py));
    const float ax = px - static_cast<float>(x0), ay = py - static_cast<float>(y0);
    auto at = [&](int cx, int cy) { return ink[(cy & (kLifeH - 1)) * kLifeW + (cx & (kLifeW - 1))].x; };
    const float top = at(x0, y0) + (at(x0 + 1, y0) - at(x0, y0)) * ax;
    const float bottom = at(x0, y0 + 1) + (at(x0 + 1, y0 + 1) - at(x0, y0 + 1)) * ax;
    const float d = fminf((top + (bottom - top) * ay) * L.fade + 0.03f * inkIn, 1.0f);
    inkOut[i] = make_float2(d, 10.0f * sqrtf(ux * ux + uy * uy));
}

// ---- Islands: Cahn-Hilliard separation with Oono and Puri's long-range term, on a coarse grid ----

/**
 * @brief The Islands, in one CUDA block: its 64 x 32 grid in shared memory, `substeps` explicit
 *        steps of dc = lap(c^3 - c - g lap c) - eps (c - mean) (g 0.6, a fiftieth of a unit each, a
 *        grain of noise so it never quite freezes), then the field drawn from it: each of its cells
 *        read bilinearly, drifted across, as 0.5 + 0.5 tanh(2.5 c). Seeded with a little noise about
 *        the mean when asked.
 * @param hdr    the header
 * @param isles  the grid, kept between blocks
 * @param lifeA  the field, buffer A
 * @param lifeB  buffer B
 */
__global__ void __launch_bounds__(1024) islandsKernel(const BlockHeader* hdr, float* isles, float2* lifeA, float2* lifeB)
{
    const LifeField& L = hdr->life;
    if (L.mode != LifeMode::Islands || L.steps < 1) return;
    constexpr int N = kIsleW * kIsleH;
    __shared__ float c[N], mu[N];
    const int t = static_cast<int>(threadIdx.x);
    const std::uint32_t salt = static_cast<std::uint32_t>(L.salt) * 0x9E3779B9U;
    for (int k = t; k < N; k += blockDim.x)
        c[k] = L.reseed ? L.mean + 0.2f * (static_cast<float>(hash32(salt ^ (static_cast<std::uint32_t>(k) * 2654435761U)) >> 8) * (1.0f / 16777216.0f) - 0.5f)
                        : isles[k];
    __syncthreads();
    constexpr float g = 0.6f, dt = 0.02f;
    for (int s = 0; s < L.substeps; ++s) {
        for (int k = t; k < N; k += blockDim.x) {
            const int x = k & (kIsleW - 1), y = k / kIsleW;
            const float lap = c[y * kIsleW + ((x + 1) & (kIsleW - 1))] + c[y * kIsleW + ((x - 1) & (kIsleW - 1))]
                            + c[((y + 1) & (kIsleH - 1)) * kIsleW + x] + c[((y - 1) & (kIsleH - 1)) * kIsleW + x] - 4.0f * c[k];
            mu[k] = c[k] * c[k] * c[k] - c[k] - g * lap;
        }
        __syncthreads();
        for (int k = t; k < N; k += blockDim.x) {
            const int x = k & (kIsleW - 1), y = k / kIsleW;
            const float lap = mu[y * kIsleW + ((x + 1) & (kIsleW - 1))] + mu[y * kIsleW + ((x - 1) & (kIsleW - 1))]
                            + mu[((y + 1) & (kIsleH - 1)) * kIsleW + x] + mu[((y - 1) & (kIsleH - 1)) * kIsleW + x] - 4.0f * mu[k];
            const std::uint32_t h = hash32((hdr->blockIndex * 16U + static_cast<std::uint32_t>(s)) * 2654435761U ^ static_cast<std::uint32_t>(k) ^ salt);
            const float noise = 0.004f * (static_cast<float>(h >> 8) * (1.0f / 16777216.0f) - 0.5f);
            c[k] = fminf(fmaxf(c[k] + dt * (lap - L.eps * (c[k] - L.mean)) + noise, -1.5f), 1.5f);
        }
        __syncthreads();
    }
    for (int k = t; k < N; k += blockDim.x) isles[k] = c[k];
    // The field (the buffer this block's one step leads to), drawn from the grid.
    float2* dst = (hdr->lifeParity & 1) ? lifeA : lifeB;
    constexpr float sx = static_cast<float>(kIsleW) / static_cast<float>(kLifeW), sy = static_cast<float>(kIsleH) / static_cast<float>(kLifeH);
    for (int i = t; i < kLifeCells; i += blockDim.x) {
        const float px = (static_cast<float>(i & (kLifeW - 1)) + 0.5f) * sx - 0.5f + L.drift, py = (static_cast<float>(i >> 8) + 0.5f) * sy - 0.5f;
        const int x0 = static_cast<int>(floorf(px)), y0 = static_cast<int>(floorf(py));
        const float ax = px - static_cast<float>(x0), ay = py - static_cast<float>(y0);
        auto at = [&](int cx, int cy) { return c[(cy & (kIsleH - 1)) * kIsleW + (cx & (kIsleW - 1))]; };
        const float top = at(x0, y0) + (at(x0 + 1, y0) - at(x0, y0)) * ax;
        const float bottom = at(x0, y0 + 1) + (at(x0 + 1, y0 + 1) - at(x0, y0 + 1)) * ax;
        const float v = top + (bottom - top) * ay;
        dst[i] = make_float2(0.5f + 0.5f * tanhf(2.5f * v), v);
    }
}

/**
 * @brief The living field's mean and mean square of what the partials read, in one CUDA block, in
 *        a fixed order (bit-identical runs).
 * @param hdr    the header
 * @param a      buffer A
 * @param b      buffer B
 * @param stats  receives mean, mean square
 */
__global__ void lifeStatsKernel(const BlockHeader* hdr, const float2* a, const float2* b, float* stats)
{
    __shared__ float s1[256], s2[256];
    const LifeField& L = hdr->life;
    const int t = static_cast<int>(threadIdx.x);
    float m = 0.0f, q = 0.0f;
    if (L.mode != LifeMode::Off) {
        const float2* cur = lifeCurrent(hdr, a, b);
        for (int i = t; i < kLifeCells; i += 256) {
            const float v = lifeValue(L.mode, cur[i]);
            m += v;
            q += v * v;
        }
    }
    s1[t] = m;
    s2[t] = q;
    __syncthreads();
    for (int s = 128; s > 0; s >>= 1) {
        if (t < s) { s1[t] += s1[t + s]; s2[t] += s2[t + s]; }
        __syncthreads();
    }
    if (t == 0) { stats[0] = s1[0] / kLifeCells; stats[1] = s2[0] / kLifeCells; }
}

/**
 * @brief The living field at a place, read bilinearly on the torus.
 * @param hdr  the header
 * @param cur  the current buffer
 * @param fx   across, cells (wraps)
 * @param fy   along, cells (wraps)
 * @return     what the partial reads, 0 .. 1
 */
__device__ float lifeAt(const BlockHeader* hdr, const float2* cur, float fx, float fy)
{
    const int x0 = static_cast<int>(floorf(fx)), y0 = static_cast<int>(floorf(fy));
    const float ax = fx - static_cast<float>(x0), ay = fy - static_cast<float>(y0);
    const LifeMode mode = hdr->life.mode;
    auto at = [&](int x, int y) { return lifeValue(mode, cur[(y & (kLifeH - 1)) * kLifeW + (x & (kLifeW - 1))]); };
    const float top = at(x0, y0) + (at(x0 + 1, y0) - at(x0, y0)) * ax;
    const float bottom = at(x0, y0 + 1) + (at(x0 + 1, y0 + 1) - at(x0, y0 + 1)) * ax;
    return top + (bottom - top) * ay;
}

/**
 * @brief The partial sync's mean fields: for every harmonic of every voice (and its air band, as one
 *        more group) the means of its members' phasors from the last block, summed in member order.
 *        Global: e^(iθ). Chimera: e^(iθ), e^(i(θ - x)), e^(i(θ + x)) with x the member's place round
 *        the ring. Swarm: e^(i(x + θ)), e^(i(x - θ)) with x its place on its circle.
 * @param hdr    the header
 * @param voices the voices
 * @param pcap   partials per voice in the tables
 * @param theta  per partial its phase
 * @param sx     per partial its place on its circle (Swarm)
 * @param order  per voice and group the mean phasors, three planes `plane` apart
 * @param plane  the planes' distance (all the partials)
 */
__global__ void syncOrderKernel(const BlockHeader* hdr, const CloudVoice* voices, int pcap, const float* theta, const float* sx, float2* order,
                                size_t plane)
{
    if (hdr->sync.depth <= 0.0f) return;
    const int gid = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int v = gid / pcap, g = gid - v * pcap;
    if (v >= hdr->voices) return;
    const CloudVoice& cv = voices[v];
    if (!cv.active) return;
    const int H = cv.harmonics, K = cv.members, HP = cv.harmonicPartials;
    int first, count;
    if (g < H) { first = g * K; count = K; }
    else if (g == H && cv.partials > HP) { first = HP; count = cv.partials - HP; }
    else return;
    const SyncMode mode = hdr->sync.mode;
    float2 z0 = make_float2(0.0f, 0.0f), z1 = z0, z2 = z0;
    const float* th = theta + static_cast<size_t>(v) * pcap + first;
    const float* xs = sx + static_cast<size_t>(v) * pcap + first;
    const float ring = 6.2831853f / static_cast<float>(max(count, 1));
    for (int j = 0; j < count; ++j) {
        float s, c;
        if (mode == SyncMode::Swarm) {
            __sincosf(xs[j] + th[j], &s, &c);
            z1.x += c; z1.y += s;
            __sincosf(xs[j] - th[j], &s, &c);
            z2.x += c; z2.y += s;
            continue;
        }
        __sincosf(th[j], &s, &c);
        z0.x += c; z0.y += s;
        if (mode == SyncMode::Chimera) {
            const float x = ring * (static_cast<float>(j) + 0.5f);
            __sincosf(th[j] - x, &s, &c);
            z1.x += c; z1.y += s;
            __sincosf(th[j] + x, &s, &c);
            z2.x += c; z2.y += s;
        }
    }
    const float inv = 1.0f / static_cast<float>(max(count, 1));
    const size_t at = static_cast<size_t>(v) * pcap + g;
    order[at] = make_float2(z0.x * inv, z0.y * inv);
    order[at + plane] = make_float2(z1.x * inv, z1.y * inv);
    order[at + 2 * plane] = make_float2(z2.x * inv, z2.y * inv);
}

/**
 * @brief A member's place in its cloud in cents (its slice of the bell, jittered as the update
 *        kernel places it).
 * @param seed     the note's dice
 * @param first    the cloud's first partial in the voice
 * @param j        the member
 * @param count    the cloud's members
 * @param sigma    the cloud's width, cents per unit of the bell
 * @return         cents from the cloud's centre
 */
__device__ __forceinline__ float memberCents(std::uint32_t seed, int first, int j, int count, float sigma)
{
    if (count <= 1) return 0.0f;
    const std::uint32_t up = static_cast<std::uint32_t>(first + j);
    return normcdfinvf((static_cast<float>(j) + 0.5f + 0.9f * (hash01(seed, up, 1) - 0.5f)) / static_cast<float>(count)) * sigma;
}

/**
 * @brief One block of the flock for one partial (Reynolds 1987, in frequency): its offset and
 *        velocity steered by the three members on either side in its cloud -- towards their mean
 *        offset (cohesion), towards their mean velocity (alignment), away from any that has come closer
 *        than half their distance at home (separation: the cloud keeps its shape; a fixed distance of
 *        six cents blew a cloud of 7 cents up to 20, its members lie a cent apart) -- plus its cloud's
 *        wind, a spring home and drag.
 *        Read from the last block's state, written into the other buffer: bit-identical.
 * @param F        the field (spring; strength: the wind)
 * @param prev     the last block's (offset, velocity) per partial
 * @param next     this block's
 * @param vbase    the voice's first partial in the tables
 * @param first    the cloud's first partial in the voice
 * @param count    its members
 * @param k        this partial's member
 * @param seed     the note's dice
 * @param sigma    the cloud's width, cents per unit of the bell
 * @param wind     its cloud's wind, cents per second squared
 * @param now      the offset it has now (taken over when the flock is fresh)
 * @param restart  a new note: offset and velocity 0
 * @param dt       the block's length, seconds
 * @return         the new offset, cents
 */
__device__ float flockStep(const SpectralField& F, const float2* prev, float2* next, size_t vbase, int first, int count, int k,
                           std::uint32_t seed, float sigma, float wind, float now, bool restart, float dt)
{
    const size_t me = vbase + static_cast<size_t>(first + k);
    if (restart) { next[me] = make_float2(0.0f, 0.0f); return 0.0f; }
    if (F.flockFresh) { next[me] = make_float2(now, 0.0f); return now; }
    const float2 s = prev[me];
    const float myHome = memberCents(seed, first, k, count, sigma);
    const float myX = myHome + s.x;
    float sumPos = 0.0f, sumVel = 0.0f, push = 0.0f;
    int nb = 0;
    for (int d = -3; d <= 3; ++d) {
        const int j = k + d;
        if (d == 0 || j < 0 || j >= count) continue;
        const float2 o = prev[vbase + static_cast<size_t>(first + j)];
        sumPos += o.x;
        sumVel += o.y;
        ++nb;
        const float home = memberCents(seed, first, j, count, sigma);
        const float sep = 0.5f * fabsf(myHome - home);
        const float dx = myX - (home + o.x);
        if (fabsf(dx) < sep) push += (j < k ? 1.0f : -1.0f) * (sep - fabsf(dx));   // the order kept: a lower member stays below
    }
    float acc = wind + 40.0f * push - 1.5f * s.y - (0.5f + 4.0f * F.spring) * s.x;
    if (nb > 0) acc += 4.0f * (sumPos / static_cast<float>(nb) - s.x) + 3.0f * (sumVel / static_cast<float>(nb) - s.y);
    const float vel = fminf(fmaxf(s.y + acc * dt, -200.0f), 200.0f);
    const float pos = fminf(fmaxf(s.x + vel * dt, -600.0f), 600.0f);
    next[me] = make_float2(pos, vel);
    return pos;
}

/**
 * @brief The ring of the voices' levels: this block's level of every voice written where the
 *        block's index points, a voice's whole ring cleared when it starts a note (so a delayed
 *        partial of the new note never hears the old one's level).
 * @param hdr     the block header
 * @param voices  the voices
 * @param ring    per voice kGainRing levels
 */
__global__ void gainRingKernel(const BlockHeader* hdr, const CloudVoice* voices, float* ring)
{
    const int v = static_cast<int>(blockIdx.x);
    if (v >= hdr->voices) return;
    const CloudVoice& cv = voices[v];
    float* row = ring + static_cast<size_t>(v) * kGainRing;
    if (cv.restart)
        for (int i = threadIdx.x; i < kGainRing; i += blockDim.x) row[i] = 0.0f;
    __syncthreads();
    if (threadIdx.x == 0) row[hdr->blockIndex % kGainRing] = cv.active ? cv.gain : 0.0f;
}

/**
 * @brief A voice's level `blocks` blocks ago, read from the ring and interpolated.
 * @param row     the voice's ring
 * @param index   this block's index
 * @param blocks  how far back, in blocks (fractional)
 * @return        the level; 0 beyond the ring
 */
__device__ float gainAgo(const float* row, std::uint32_t index, float blocks)
{
    if (blocks <= 0.0f) return row[index % kGainRing];
    if (blocks >= static_cast<float>(kGainRing - 2)) return 0.0f;
    const int b0 = static_cast<int>(blocks);
    const float fr = blocks - static_cast<float>(b0);
    const float g0 = row[(index - static_cast<std::uint32_t>(b0)) % kGainRing];
    const float g1 = row[(index - static_cast<std::uint32_t>(b0 + 1)) % kGainRing];
    return g0 + (g1 - g0) * fr;
}

/**
 * @brief The voice filters transported into each other (Morph.h), one CUDA block a voice that carries
 *        two or more: each filter's power over the octaves (kFilterBins bands, floored at -60 dB) read
 *        as a mass, its distribution function by a running sum in a fixed order; the Wasserstein
 *        barycentre's quantile function the weighted mean of theirs (in one dimension the barycentre
 *        is exactly that); its distribution function at the bands' edges found back in it; a band's
 *        power the mean mass times its share, its gain the root. A single filter comes out as itself.
 * @param hdr    the header
 * @param voices the voices
 * @param table  per voice kFilterBins gains
 */
__global__ void __launch_bounds__(256) filterTransportKernel(const BlockHeader* hdr, const CloudVoice* voices, float* table)
{
    const int v = static_cast<int>(blockIdx.x);
    if (v >= hdr->voices) return;
    const CloudVoice& cv = voices[v];
    if (!cv.active || cv.filterCount < 2) return;
    const int K = min(cv.filterCount, kVoiceFilters);
    __shared__ float cdf[kVoiceFilters][kFilterBins + 1];
    __shared__ float qs[kFilterBins + 1];
    __shared__ float mass[kVoiceFilters];
    const int t = static_cast<int>(threadIdx.x);
    for (int k = 0; k < K; ++k)
        for (int b = t; b < kFilterBins; b += blockDim.x) {
            const float f = 20.0f * exp2f((static_cast<float>(b) + 0.5f) * (9.96578428f / static_cast<float>(kFilterBins)));
            const float g = filterGain(static_cast<int>(cv.filterModels[k]), f, cv.filterCutoffs[k], cv.filterQs[k], cv.filterVowels[k]);
            cdf[k][b + 1] = fmaxf(g * g, 1.0e-6f);
        }
    __syncthreads();
    if (t < K) {   // the running sums, one filter a thread, in band order
        float s = 0.0f;
        cdf[t][0] = 0.0f;
        for (int b = 1; b <= kFilterBins; ++b) { s += cdf[t][b]; cdf[t][b] = s; }
        mass[t] = s;
    }
    __syncthreads();
    for (int k = 0; k < K; ++k)
        for (int b = t; b <= kFilterBins; b += blockDim.x) cdf[k][b] /= mass[k];
    __syncthreads();
    // The barycentre's quantile function at every level j / kFilterBins, in bands.
    for (int j = t; j <= kFilterBins; j += blockDim.x) {
        const float s = static_cast<float>(j) / static_cast<float>(kFilterBins);
        float x = 0.0f;
        for (int k = 0; k < K; ++k) {
            int lo = 0, hi = kFilterBins;
            while (hi - lo > 1) {
                const int mid = (lo + hi) >> 1;
                if (cdf[k][mid] <= s) lo = mid; else hi = mid;
            }
            const float c0 = cdf[k][lo], c1 = cdf[k][lo + 1];
            const float fr = c1 > c0 ? fminf(fmaxf((s - c0) / (c1 - c0), 0.0f), 1.0f) : 0.0f;
            x += cv.filterWeight[k] * (static_cast<float>(lo) + fr);
        }
        qs[j] = x;
    }
    __syncthreads();
    float M = 0.0f;
    for (int k = 0; k < K; ++k) M += cv.filterWeight[k] * mass[k];
    // Its distribution function at an edge (in bands): where the quantile function crosses it.
    auto F = [&](float edge) {
        if (edge <= qs[0]) return 0.0f;
        if (edge >= qs[kFilterBins]) return 1.0f;
        int lo = 0, hi = kFilterBins;
        while (hi - lo > 1) {
            const int mid = (lo + hi) >> 1;
            if (qs[mid] <= edge) lo = mid; else hi = mid;
        }
        const float q0 = qs[lo], q1 = qs[lo + 1];
        const float fr = q1 > q0 ? fminf(fmaxf((edge - q0) / (q1 - q0), 0.0f), 1.0f) : 0.0f;
        return (static_cast<float>(lo) + fr) / static_cast<float>(kFilterBins);
    };
    float* row = table + static_cast<size_t>(v) * kFilterBins;
    for (int b = t; b < kFilterBins; b += blockDim.x)
        row[b] = sqrtf(fmaxf(M * (F(static_cast<float>(b + 1)) - F(static_cast<float>(b))), 0.0f));
}

/**
 * @brief The energy cascade (Turbulence, Cloud.h), one thread a harmonic of a voice: its power p
 *        (against harmonic 1's at rest) loses drive p^2 and gains seven tenths of what the one below
 *        lost and three tenths of what its half lost, and settles back to the spectrum's own over
 *        settle / (1 + h / 8) seconds. Kept as the power against the spectrum's at rest, q = p / p0,
 *        read by the update kernel as sqrt(q) on the harmonic's amplitude. Reads one buffer, writes
 *        the other (by the block's parity); a new note starts at rest.
 * @param hdr     the header
 * @param voices  the voices
 * @param pcap    partials per voice in the tables (room for every harmonic)
 * @param qa      per voice and harmonic its q, buffer A
 * @param qb      buffer B
 */
__global__ void cascadeKernel(const BlockHeader* hdr, const CloudVoice* voices, int pcap, float* qa, float* qb)
{
    if (hdr->turb.on <= 0.0f) return;
    const int gid = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int v = gid / pcap, k = gid - v * pcap;   // harmonic k + 1
    if (v >= hdr->voices) return;
    const CloudVoice& cv = voices[v];
    const int H = cv.harmonics;
    if (!cv.active || k >= H) return;
    const bool odd = (hdr->blockIndex & 1U) != 0;
    const float* src = (odd ? qb : qa) + static_cast<size_t>(v) * pcap;
    float* dst = (odd ? qa : qb) + static_cast<size_t>(v) * pcap;
    // Kept as q - 1, so a table cleared to nothing is at rest.
    if (cv.restart) { dst[k] = 0.0f; return; }
    const float s1 = fmaxf(harmonicShape(1, cv.tilt, cv.oddEven, cv.brightness), 1.0e-9f);
    auto rest = [&](int h) { const float s = harmonicShape(h, cv.tilt, cv.oddEven, cv.brightness) / s1; return fmaxf(s * s, 1.0e-12f); };
    auto power = [&](int h) { return (1.0f + src[h - 1]) * rest(h); };
    const int h = k + 1;
    const float drive = cv.turbulence * cv.gain * cv.gain;
    const float dt = static_cast<float>(hdr->n) / hdr->sampleRate;
    const float p0 = rest(h), p = power(h);
    float in = 0.0f;
    if (h > 1) { const float pb = power(h - 1); in += 0.7f * drive * pb * pb; }
    if ((h & 1) == 0) { const float ph = power(h / 2); in += 0.3f * drive * ph * ph; }
    const float settle = fminf(1.0f, dt * (1.0f + static_cast<float>(h) * 0.125f) / fmaxf(hdr->turb.settle, 0.1f));
    const float next = fmaxf(p + dt * (in - drive * p * p) + (p0 - p) * settle, 0.0f);
    dst[k] = fminf(next / p0, 1000.0f) - 1.0f;
}

/**
 * @brief The update kernel: every partial's frequency glide, amplitude ramps and start phase for this block.
 * @param hdr     the block header (device copy)
 * @param voices  the voices (device copy)
 * @param pcap    partials per voice in the tables (a multiple of kChunk)
 * @param shapes  the envelope bank
 * @param land    the landscape
 * @param run     per partial: start phase, phase step, its change per sample, the field's offset in cents (as bits)
 * @param amp     per partial: the near left, near right, far left and far right amplitudes the last block ended on
 * @param gains   per partial three entries: near (left start, left step, right start, right step), far, and
 *                the near plane's cosine parts (binaural: the ears' phases)
 * @param ring    per voice kGainRing levels (gainRingKernel), for the dispersion
 * @param ampQ    per partial the near plane's cosine parts the last block ended on
 * @param snap    every kSnapStride-th partial's frequency and amplitude, for the picture
 * @param lifeA   the living field, buffer A
 * @param lifeB   buffer B
 * @param theta   per partial the sync's phase
 * @param sx      per partial its place on its circle (Swarm)
 * @param order   per voice and group the sync's mean phasors (syncOrderKernel), three planes `plane` apart
 * @param plane   the planes' distance
 * @param flockA  the flock's state, buffer A (even blocks read it)
 * @param flockB  buffer B
 * @param qa      the energy cascade, buffer A (cascadeKernel)
 * @param qb      buffer B
 * @param filterTable  per voice the transported filter (filterTransportKernel)
 * @param hallHist     per band the amplitude gathered this block, for the hall's ear (hallKernel)
 * @param density      per band the spectrum's density the block before (hallKernel)
 * @param tau          per partial its own clock (Time Bend)
 * @param orbitA       the Orbit field's state, buffer A (even blocks read it)
 * @param orbitB       buffer B
 */
__global__ void updateKernel(const BlockHeader* hdr, const CloudVoice* voices, int pcap, const EnvShape* shapes,
                             const float* land, uint4* run, float4* amp, float4* gains, const float* ring, float2* ampQ,
                             float2* snap, const float2* lifeA, const float2* lifeB, float* theta, float* sx, const float2* order,
                             size_t plane, float2* flockA, float2* flockB, const float* qa, const float* qb, const float* filterTable,
                             unsigned long long* hallHist, const float* density, float* tau, float4* orbitA, float4* orbitB)
{
    const int gid = blockIdx.x * blockDim.x + threadIdx.x;
    const int v = gid / pcap;
    const int p = gid - v * pcap;
    if (v >= hdr->voices) return;
    const CloudVoice& cv = voices[v];
    const bool snapped = p % kSnapStride == 0;
    float2* dot = snapped ? snap + static_cast<size_t>(v) * (pcap / kSnapStride) + p / kSnapStride : nullptr;
    if (!cv.active || p >= cv.partials) {
        if (dot) *dot = make_float2(0.0f, 0.0f);
        // Beyond the voice's partials (or a silent voice): the render kernel skips whole chunks, the
        // tail of the last chunk it covers must be silent.
        if (cv.active) {
            gains[3 * gid] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            gains[3 * gid + 1] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            gains[3 * gid + 2] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            amp[gid] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            ampQ[gid] = make_float2(0.0f, 0.0f);
        }
        return;
    }
    const float fs = hdr->sampleRate;
    const int n = hdr->n;
    const std::uint32_t seed = cv.seed;
    const std::uint32_t up = static_cast<std::uint32_t>(p);
    const float t = cv.time + static_cast<float>(n) / fs;   // every target holds at the block's end

    float target = 0.0f;
    float f = 0.0f;
    float fieldOff = cv.restart ? 0.0f : __uint_as_float(run[gid].w);   // the field's offset, cents
    float place = cv.stereo * (2.0f * hash01(seed, up, 10) - 1.0f);   // -1 .. 1 across the field
    const int H = cv.harmonics, K = cv.members;
    const int HP = cv.harmonicPartials;
    const bool isAir = p >= HP;
    if (isAir || p < H * K) {
        // Who the partial is: a member of harmonic h's cloud, or one of the air band's partials.
        int h = 1;
        float z = 0.0f, centre, amp0, sh, drift;
        float slot = 0.5f;   // the partial's place among its cloud's members, evenly 0 .. 1 (the living field's rows)
        int groupFirst = 0, groupCount = 1, member = 0;   // its cloud in the voice, for the flock
        float sigmaCents = 0.0f;                            // the cloud's width, cents per unit of the bell
        // Time Bend: the partial's own clock (its drift, its shimmer, its sync), slower -- or faster --
        // where the spectrum is dense, read at its harmonic from the density of the block before. At
        // nothing it is the voice's clock, kept in step, so turning it up never jumps.
        float tp = t, clockRate = 1.0f;
        if (hdr->field.timeBend != 0.0f) {
            const float fh = isAir ? cv.hz * cv.airCenter : cv.hz * partialRatio(p / max(K, 1) + 1, cv.inharmonic, cv.stretch);
            const float dense = fminf(fmaxf(log2f(1.0f + density[hallBin(fh)]) - 1.0f, 0.0f), 4.0f);
            const float bend = hdr->field.timeBend;
            clockRate = bend > 0.0f ? 1.0f / (1.0f + bend * dense) : 1.0f - bend * dense;
            tp = (cv.restart ? 0.0f : tau[gid]) + clockRate * static_cast<float>(n) / fs;
        }
        tau[gid] = tp;
        if (!isAir) {
            h = p / K + 1;
            const int k = p - (h - 1) * K;
            slot = (static_cast<float>(k) + 0.5f) / static_cast<float>(K);
            // The member's place in its harmonic's cloud: stratified over the bell curve, so K members
            // always cover it evenly and only jitter within their own slice.
            if (K > 1) z = normcdfinvf((static_cast<float>(k) + 0.5f + 0.9f * (hash01(seed, up, 1) - 0.5f)) / static_cast<float>(K));
            const float sigma = 0.5f * cv.width * __powf(static_cast<float>(h), cv.widthSlope);
            groupFirst = (h - 1) * K;
            groupCount = K;
            member = k;
            sigmaCents = sigma;
            centre = cv.hz * partialRatio(h, cv.inharmonic, cv.stretch) * exp2f(z * sigma * (1.0f / 1200.0f)) + cv.shift;
            drift = cv.drift * (0.65f * slowSine(cv.driftRate * (0.5f + hash01(seed, up, 2)), tp, hash01(seed, up, 3))
                              + 0.35f * slowSine(cv.driftRate * (1.2f + hash01(seed, up, 4)), tp, hash01(seed, up, 5)));
            sh = fmaxf(0.0f, 1.0f + cv.shimmer * (0.7f * slowSine(cv.shimmerRate * (0.5f + hash01(seed, up, 6)), tp, hash01(seed, up, 7))
                                                 + 0.3f * slowSine(cv.shimmerRate * (1.3f + hash01(seed, up, 8)), tp, hash01(seed, up, 9))));
            amp0 = cv.scale * harmonicShape(h, cv.tilt, cv.oddEven, cv.brightness);
            if (hdr->turb.on > 0.0f && !cv.restart) {   // the cascade's power against the spectrum's (this block's, cascadeKernel)
                const float* q = ((hdr->blockIndex & 1U) ? qa : qb) + static_cast<size_t>(v) * pcap;
                amp0 *= sqrtf(fmaxf(1.0f + q[h - 1], 0.0f));
            }
        } else {
            // Air (Noctuary's band-passed noise around a multiple of the fundamental): a wide cloud
            // whose partials flicker in pitch and level at audio-near rates -- dense random sines in
            // a band are heard as its noise.
            const int A = cv.partials - HP;
            const int a = p - HP;
            slot = (static_cast<float>(a) + 0.5f) / static_cast<float>(max(A, 1));
            if (A > 1) z = normcdfinvf((static_cast<float>(a) + 0.5f + 0.9f * (hash01(seed, up, 1) - 0.5f)) / static_cast<float>(A));
            groupFirst = HP;
            groupCount = A;
            member = a;
            sigmaCents = 0.5f * cv.airWidth;
            centre = cv.hz * cv.airCenter * exp2f(z * 0.5f * cv.airWidth * (1.0f / 1200.0f)) + cv.shift;
            const float j = cv.airJitter;
            drift = 40.0f * (0.6f * slowSine(j * (0.5f + hash01(seed, up, 2)), tp, hash01(seed, up, 3))
                           + 0.4f * slowSine(j * (1.3f + hash01(seed, up, 4)), tp, hash01(seed, up, 5)));
            sh = fmaxf(0.0f, 1.0f + 0.8f * (0.7f * slowSine(j * (0.7f + hash01(seed, up, 6)), tp, hash01(seed, up, 7))
                                          + 0.3f * slowSine(j * (1.6f + hash01(seed, up, 8)), tp, hash01(seed, up, 9))));
            amp0 = cv.airScale;
        }

        // The envelope lanes, read against this partial's sources.
        LaneContext c;
        centre = fmaxf(centre, 1.0f);   // a partial the shift took below nothing (it is faded out)
        const float pitchPos = log2f(fmaxf(centre, kLandLowHz) * (1.0f / kLandLowHz)) * (1.0f / 9.96578428f);   // log2(1000)
        c.src[0] = 0.0f;
        c.src[1] = fminf(fmaxf(pitchPos, 0.0f), 1.0f);
        c.src[2] = isAir ? 0.5f : (H > 1 ? static_cast<float>(h - 1) / static_cast<float>(H - 1) : 0.0f);
        c.src[3] = fminf(fmaxf((z + 3.0f) * (1.0f / 6.0f), 0.0f), 1.0f);
        c.src[4] = 0.0f;   // Random: laneValue throws its own die per parameter
        c.src[5] = (isAir || (h & 1)) ? 0.0f : 1.0f;
        c.seed = seed;
        c.p = up;
        // The dispersion: this partial's clock late by its share of the dispersion, by its pitch.
        const float D = hdr->field.dispersion;
        const float lateBy = D >= 0.0f ? D * (1.0f - c.src[1]) : -D * c.src[1];
        c.t = t - lateBy;
        c.releaseAt = cv.releaseAt;
        c.shapes = shapes;
        c.shapeCount = hdr->shapeCount;
        c.land = land;
        c.landColumns = hdr->landColumns;
        c.landRows = hdr->landRows;
        c.landRowSeconds = fmaxf(hdr->landRowSeconds, 1.0e-4f);
        c.landColumn = c.src[1] * static_cast<float>(max(hdr->landColumns - 1, 0));
        float depth = 0.0f;
        float env = 1.0f;
        if (cv.amp.amount != 0.0f) {
            const float e = laneValue(cv.amp, c, 30, false, depth);
            depth = fminf(fmaxf(depth, 0.0f), 1.0f);
            env = 1.0f - depth + depth * e;
        }
        // Partial sync: the phase turns at its own rate and is pulled to its cloud (Global), to its
        // neighbours round the ring (Chimera), or with its place on its circle (Swarm).
        float swarmCents = 0.0f, swarmSide = 0.0f;
        if (hdr->sync.depth > 0.0f) {
            const PartialSync& S = hdr->sync;
            const float dtb = static_cast<float>(n) / fs;
            const float ringX = 6.2831853f * (static_cast<float>(member) + 0.5f) / static_cast<float>(max(groupCount, 1));
            float th;
            if (!cv.restart) {
                th = theta[gid];
            } else if (S.mode == SyncMode::Chimera) {
                // Abrams and Strogatz's start: ordered round the middle of the ring, scattered at its ends.
                const float xc = ringX - 3.1415927f;
                th = 6.0f * (hash01(seed, up, 13) - 0.5f) * __expf(-0.76f * xc * xc);
            } else {
                th = 6.2831853f * hash01(seed, up, 13);
            }
            const float u = 2.0f * hash01(seed, up, 12) - 1.0f;
            const size_t at = static_cast<size_t>(v) * pcap + (isAir ? H : h - 1);
            float s, co;
            __sincosf(th, &s, &co);
            if (S.mode == SyncMode::Swarm) {
                // In the frame turning at the rate: what is left of each rate, and the place's own drift.
                float x = cv.restart ? 6.2831853f * hash01(seed, up, 14) : sx[gid];
                float dth = 6.2831853f * S.rate * S.spread * u, dx = 3.1415927f * S.rate * S.spread * (2.0f * hash01(seed, up, 15) - 1.0f);
                if (!cv.restart) {
                    const float2 wp = order[at + plane], wm = order[at + 2 * plane];
                    float sp, cp, sm, cm;
                    __sincosf(x + th, &sp, &cp);
                    __sincosf(x - th, &sm, &cm);
                    const float ap = wp.y * cp - wp.x * sp;   // S+ sin(Φ+ - (x + θ))
                    const float am = wm.y * cm - wm.x * sm;   // S- sin(Φ- - (x - θ))
                    dx += 0.5f * S.swarm * (ap + am);
                    dth += 0.5f * S.coupling * (ap - am);
                }
                x += dx * dtb * clockRate;
                x -= 6.2831853f * floorf(x * 0.15915494f);
                sx[gid] = x;
                th += dth * dtb * clockRate;
                th -= 6.2831853f * floorf(th * 0.15915494f);
                theta[gid] = th;
                float sxs, sxc;
                __sincosf(x, &sxs, &sxc);
                swarmCents = S.reach * sxc;
                swarmSide = 0.5f * fminf(S.reach * (1.0f / 30.0f), 1.0f) * sxs;
                // The heard phase turns on at the rate.
                const float heard = th + 6.2831853f * S.rate * tp;
                env *= fmaxf(0.0f, 1.0f + S.depth * __sinf(heard)) * rsqrtf(1.0f + 0.5f * S.depth * S.depth);
            } else {
                const float w = 6.2831853f * S.rate * (1.0f + S.spread * u);
                float pull = 0.0f;
                if (!cv.restart) {
                    const float2 z = order[at];
                    if (S.mode == SyncMode::Chimera) {
                        // Σ G(x_i - x_j) e^(iθ_j) / N with G = 1 + A cos: the mean, and the two harmonics of the ring.
                        const float2 zm = order[at + plane], zp = order[at + 2 * plane];
                        float rs, rc;
                        __sincosf(ringX, &rs, &rc);
                        const float A = 0.4975f;   // A / 2, A = 0.995
                        const float cx = z.x + A * ((rc * zm.x - rs * zm.y) + (rc * zp.x + rs * zp.y));
                        const float cy = z.y + A * ((rc * zm.y + rs * zm.x) + (rc * zp.y - rs * zp.x));
                        float ls, lc;
                        __sincosf(th + S.lag, &ls, &lc);
                        pull = S.coupling * (cy * lc - cx * ls);   // K Im(C e^(-i(θ + α)))
                    } else {
                        pull = S.coupling * (z.y * co - z.x * s);   // K R sin(ψ - θ)
                    }
                }
                th += (w + pull) * dtb * clockRate;
                th -= 6.2831853f * floorf(th * 0.15915494f);
                theta[gid] = th;
                env *= fmaxf(0.0f, 1.0f + S.depth * __sinf(th)) * rsqrtf(1.0f + 0.5f * S.depth * S.depth);
            }
        }
        // The ripple: the wave along the spectrum at this partial's frequency.
        if (hdr->ripple.depth > 0.0f) {
            const float x = c.src[1] * static_cast<float>(kRippleCells - 1);
            const int x0 = min(static_cast<int>(x), kRippleCells - 2);
            const float u = hdr->ripple.u[x0] + (hdr->ripple.u[x0 + 1] - hdr->ripple.u[x0]) * (x - static_cast<float>(x0));
            env *= fmaxf(0.0f, 1.0f + hdr->ripple.depth * tanhf(u)) * hdr->ripple.makeup;
        }
        // The living landscape: the field at this partial's frequency and its place in its cloud.
        if (hdr->life.mode != LifeMode::Off && hdr->life.depth > 0.0f) {
            const LifeField& L = hdr->life;
            // Along: the member's slot, evenly over the rows -- what the field's measure assumes (the
            // bell-shaped member position read the middle rows only: 2.6 dB off while a Turing field grew).
            const float v = lifeAt(hdr, lifeCurrent(hdr, lifeA, lifeB), c.src[1] * L.zoom * static_cast<float>(kLifeW), slot * static_cast<float>(kLifeH));
            env *= 1.0f - L.depth + L.depth * v * L.makeup;
        }
        float cents = drift + swarmCents;
        if (swarmSide != 0.0f) place = fminf(fmaxf(place + swarmSide, -1.0f), 1.0f);
        if (cv.pitch.amount != 0.0f) cents += laneValue(cv.pitch, c, 40, true, depth) * depth;
        if (cv.pan.amount != 0.0f) place = fminf(fmaxf(place + laneValue(cv.pan, c, 50, true, depth) * depth, -1.0f), 1.0f);

        // The field: the partial's offset moves towards what pulls it, from where it was.
        const float home = centre * exp2f(cents * (1.0f / 1200.0f));
        if (hdr->field.mode == FieldMode::Flock) {
            // The flock: every cloud a flock of its own, blown by the wind of its own attractor.
            const bool odd = (hdr->blockIndex & 1U) != 0;
            const float wind = hdr->field.pos[(isAir ? 0 : h - 1) & (kFieldAttractors - 1)] * 0.2f * hdr->field.strength;
            fieldOff = flockStep(hdr->field, odd ? flockB : flockA, odd ? flockA : flockB, static_cast<size_t>(v) * pcap, groupFirst, groupCount,
                                 member, seed, sigmaCents, wind, fieldOff, cv.restart != 0, static_cast<float>(n) / fs);
        } else if (hdr->field.mode == FieldMode::Orbit) {
            // The orbit: drawn to the scale's pitches, bent by the field into circles of pitch and side.
            const bool odd = (hdr->blockIndex & 1U) != 0;
            float side = 0.0f;
            fieldOff = orbitStep(hdr->field, odd ? orbitB : orbitA, odd ? orbitA : orbitB, static_cast<size_t>(gid), log2f(fmaxf(home, 1.0f)),
                                 z * (2.0f / 1200.0f), fieldOff, seed, up, hdr->blockIndex, cv.restart != 0, static_cast<float>(n) / fs, side);
            place = fminf(fmaxf(place + side, -1.0f), 1.0f);
        } else {
            fieldOff = fieldStep(hdr->field, log2f(fmaxf(home, 1.0f)) + fieldOff * (1.0f / 1200.0f), fieldOff, static_cast<float>(n) / fs,
                                 z * (2.0f / 1200.0f));
        }
        f = home * exp2f(fieldOff * (1.0f / 1200.0f));
        // The top fades out between 0.43 and 0.47 of the sample rate: nothing reaches Nyquist; a
        // partial the shift took down fades out between 20 and 5 Hz (and never runs below 1).
        const float fade = fminf(fmaxf((0.47f * fs - f) / (0.04f * fs), 0.0f), 1.0f) * fminf(fmaxf((f - 5.0f) * (1.0f / 15.0f), 0.0f), 1.0f);
        f = fmaxf(f, 1.0f);
        // Distance (Noctuary's planes): 6 dB per unit, and the air's low pass -- a first-order
        // corner falling from 16 kHz by up to five octaves per unit -- read at this partial's
        // frequency, where in a filter it would have been a whole filter. Taken relative to the
        // corner at the ear, so a partial at distance 0 keeps exactly its level (as in Noctuary).
        const float dist = fminf(fmaxf(cv.distance, 0.0f), 1.0f);
        const float x0 = f * (1.0f / 16000.0f);
        const float x = x0 / exp2f(-5.0f * dist * cv.airAbsorb);
        const float absorb = sqrtf((1.0f + x0 * x0) / (1.0f + x * x));
        // The voice's colour and filter, each read at this partial's frequency: the filter as it is, or
        // during a morph between different ones their transport (filterTransportKernel's table).
        float colour;
        if (cv.filterCount >= 2) {
            const float* row = filterTable + static_cast<size_t>(v) * kFilterBins;
            const float pos = fminf(fmaxf(log2f(f * (1.0f / 20.0f)) * (static_cast<float>(kFilterBins) / 9.96578428f) - 0.5f, 0.0f),
                                    static_cast<float>(kFilterBins - 1) - 1.0e-3f);
            const int i0 = static_cast<int>(pos);
            colour = row[i0] + (row[min(i0 + 1, kFilterBins - 1)] - row[i0]) * (pos - static_cast<float>(i0));
        } else {
            colour = filterGain(cv.filterModel, f, cv.cutoff, cv.filterQ, cv.vowel);
        }
        if (hdr->comb.depth > 0.0f) {   // the spectral comb: a raised cosine per notch, on the log axis
            const SpectralComb& K = hdr->comb;
            const float x = log2f(f) * K.density + K.phase + 0.25f * K.stereo * place;
            colour *= (1.0f - K.depth * (0.5f - 0.5f * __cosf(6.2831853f * x))) * K.makeup;
        }
        if (!isAir) colour *= presenceGain(f, cv.presenceDb, dist) * lowCutGain(f, cv.lowCut);
        // The voice's level, as late as this partial's clock (the dispersion), from the ring.
        const float level = lateBy > 0.0f ? gainAgo(ring + static_cast<size_t>(v) * kGainRing, hdr->blockIndex, lateBy * fs / static_cast<float>(n))
                                          : cv.gain;
        target = amp0 * sh * fade * env * level * exp2f(-dist) * absorb * colour;
        // The hall's ear: this partial's amplitude in its band, in whole numbers (the sum is the same in
        // any order). Amplitudes, not powers: a chord's upper harmonics count as heard too, not only the
        // few loudest bands.
        if (hdr->modal.listen && target > 0.0f)
            atomicAdd(hallHist + hallBin(f), static_cast<unsigned long long>(target * 1.0e12f));
    }
    const std::uint32_t incTarget = static_cast<std::uint32_t>(fminf(f / fs, 0.49f) * 4294967296.0f);

    // Phase and glide: the step moves linearly from where the last block left it to this block's
    // target, so the phase is quadratic in the sample and pitch envelopes never step.
    std::uint32_t ph, inc0, dinc = 0;
    float4 a0;
    float2 q0 = make_float2(0.0f, 0.0f);
    int onset = 0;
    if (cv.restart) {
        // A random phase. (Never aligned: a harmonic's hundreds of members lie within cents of each
        // other, so in phase they add in amplitude -- measured 19 dB louder for half a second, and
        // a plain sine rather than a pulse. The cloud's level assumes they add in power.)
        ph = hash32(seed ^ hash32(up + 0x51ED270BU));
        inc0 = incTarget;
        a0 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        onset = min(max(cv.onset, 0), n - 1);           // the ramp starts where the note does
    } else {
        const uint4 r = run[gid];
        const std::uint32_t np = static_cast<std::uint32_t>(hdr->nPrev);
        ph = r.x + np * r.y + r.z * (np * (np - 1u) / 2u);
        inc0 = r.y + r.z * np;
        const long long diff = static_cast<long long>(incTarget) - static_cast<long long>(inc0);
        if (diff > -n && diff < n) inc0 = incTarget;   // close enough: land exactly (under a millihertz)
        else dinc = static_cast<std::uint32_t>(static_cast<std::int32_t>(diff / n));
        a0 = amp[gid];
        q0 = ampQ[gid];
    }

    // Equal-power pan: the partial's place in the field; equal-power planes: near cos, far sin.
    float gl, gr, gn, gf;
    __sincosf(0.78539816f * (1.0f + place), &gr, &gl);
    __sincosf(1.57079633f * fminf(fmaxf(cv.distance, 0.0f), 1.0f), &gf, &gn);
    float4 t4 = make_float4(target * gl * gn, target * gr * gn, target * gl * gf, target * gr * gf);
    float2 tq = make_float2(0.0f, 0.0f);
    if (hdr->binaural.on && target != 0.0f) {
        // Binaural: the near plane's two ears as complex gains, a sine part and a cosine part each.
        float el[2], er[2];
        earGains(hdr->binaural, place, hash01(seed, up, 11), f, el, er);
        t4.x = target * gn * el[0];
        t4.y = target * gn * er[0];
        tq = make_float2(target * gn * el[1], target * gn * er[1]);
    }
    const float inv = 1.0f / static_cast<float>(n - onset);
    gains[3 * gid] = make_float4(a0.x, (t4.x - a0.x) * inv, a0.y, (t4.y - a0.y) * inv);
    gains[3 * gid + 1] = make_float4(a0.z, (t4.z - a0.z) * inv, a0.w, (t4.w - a0.w) * inv);
    gains[3 * gid + 2] = make_float4(q0.x, (tq.x - q0.x) * inv, q0.y, (tq.y - q0.y) * inv);
    amp[gid] = t4;
    ampQ[gid] = tq;
    if (dot) *dot = make_float2(f, target);
    run[gid] = make_uint4(ph, inc0, dinc, __float_as_uint(fieldOff));
}

/**
 * @brief The render kernel: one chunk of partials summed into its own four-channel block of scratch.
 * @param hdr      the block header
 * @param voices   the voices
 * @param pcap     partials per voice in the tables
 * @param run      per partial: start phase, phase step, its change per sample
 * @param gains    per partial three entries: near (left start, left step, right start, right step), far, near cosine parts
 * @param scratch  per chunk: maxN samples of near left, near right, far left, far right
 * @param maxN     the engine's block size (the scratch stride)
 */
__global__ void __launch_bounds__(kThreads)
renderKernel(const BlockHeader* hdr, const CloudVoice* voices, int pcap,
             const uint4* run, const float4* gains, float* scratch, int maxN)
{
    __shared__ float4 sgn[kChunk];        // the chunk's near amplitude ramps
    __shared__ float4 sgf[kChunk];        // and its far ones
    __shared__ float4 sgq[kChunk];        // binaural: the near plane's cosine parts
    __shared__ uint4  sp[kChunk];         // its start phases, steps and glides
    __shared__ float  red[kCloudChannels][kThreads];   // the groups' sums when the block is shorter than kThreads

    const int chunk = blockIdx.x;
    const int chunksPerVoice = pcap / kChunk;
    const int v = chunk / chunksPerVoice;
    const int first = (chunk - v * chunksPerVoice) * kChunk;
    const int n = hdr->n;
    float* out = scratch + static_cast<size_t>(chunk) * kCloudChannels * maxN;

    const CloudVoice& cv = voices[v];
    if (v >= hdr->voices || !cv.active || first >= cv.partials) {
        for (int s = threadIdx.x; s < n; s += kThreads)
            for (int c = 0; c < kCloudChannels; ++c) out[c * maxN + s] = 0.0f;
        return;
    }
    const int count = min(kChunk, cv.partials - first);
    const int base = v * pcap + first;
    const bool binaural = hdr->binaural.on != 0;
    for (int j = threadIdx.x; j < count; j += kThreads) {
        sgn[j] = gains[3 * (base + j)];
        sgf[j] = gains[3 * (base + j) + 1];
        if (binaural) sgq[j] = gains[3 * (base + j) + 2];
        sp[j] = run[base + j];
    }
    // A note that begins in this block begins at its onset: before it every ramp holds at its start.
    const int onset = cv.restart ? min(max(cv.onset, 0), n - 1) : 0;
    __syncthreads();

    // One thread per sample; a block shorter than kThreads splits the partials among groups.
    const int lanes = min(n, kThreads);
    const int groups = kThreads / lanes;
    const int lane = threadIdx.x % lanes;
    const int g = threadIdx.x / lanes;
    for (int s0 = 0; s0 < n; s0 += lanes) {
        const int s = s0 + lane;
        const std::uint32_t su = static_cast<std::uint32_t>(s);
        const std::uint32_t tri = su * (su - 1u) / 2u;   // the glide's share: sum of 0 .. s-1
        const float sa = static_cast<float>(max(s - onset, 0));
        float nl = 0.0f, nr = 0.0f, fl = 0.0f, fr = 0.0f;
        if (!binaural) {
            for (int j = g; j < count; j += groups) {
                const uint4 pi = sp[j];
                const float4 a = sgn[j];
                const float4 b = sgf[j];
                // The phase as a signed fraction of a turn: -pi .. pi, where __sinf is most accurate.
                const std::uint32_t ph = pi.x + su * pi.y + tri * pi.z;
                const float x = __sinf(static_cast<float>(static_cast<std::int32_t>(ph)) * 1.46291808e-9f);
                nl = fmaf(fmaf(a.y, sa, a.x), x, nl);
                nr = fmaf(fmaf(a.w, sa, a.z), x, nr);
                fl = fmaf(fmaf(b.y, sa, b.x), x, fl);
                fr = fmaf(fmaf(b.w, sa, b.z), x, fr);
            }
        } else {
            // Binaural: each ear hears the partial at its own phase -- a sine part and a cosine part.
            for (int j = g; j < count; j += groups) {
                const uint4 pi = sp[j];
                const float4 a = sgn[j];
                const float4 b = sgf[j];
                const float4 q = sgq[j];
                const std::uint32_t ph = pi.x + su * pi.y + tri * pi.z;
                float x, y;
                __sincosf(static_cast<float>(static_cast<std::int32_t>(ph)) * 1.46291808e-9f, &x, &y);
                nl = fmaf(fmaf(a.y, sa, a.x), x, fmaf(fmaf(q.y, sa, q.x), y, nl));
                nr = fmaf(fmaf(a.w, sa, a.z), x, fmaf(fmaf(q.w, sa, q.z), y, nr));
                fl = fmaf(fmaf(b.y, sa, b.x), x, fl);
                fr = fmaf(fmaf(b.w, sa, b.z), x, fr);
            }
        }
        if (groups > 1) {
            red[0][threadIdx.x] = nl;
            red[1][threadIdx.x] = nr;
            red[2][threadIdx.x] = fl;
            red[3][threadIdx.x] = fr;
            __syncthreads();
            if (g == 0)
                for (int q = 1; q < groups; ++q) {
                    nl += red[0][lane + q * lanes];
                    nr += red[1][lane + q * lanes];
                    fl += red[2][lane + q * lanes];
                    fr += red[3][lane + q * lanes];
                }
            __syncthreads();
        }
        if (g == 0) { out[s] = nl; out[maxN + s] = nr; out[2 * maxN + s] = fl; out[3 * maxN + s] = fr; }
    }
}

/**
 * @brief The reduce kernel: the chunks' sums added sample by sample into the first kCloudChannels
 *        (near L, near R, far L, far R) of interleaved frames of kCloudOutChannels.
 * @param hdr      the block header
 * @param scratch  the render kernel's per-chunk blocks
 * @param chunks   how many chunks
 * @param maxN     the scratch stride
 * @param out      n frames of kCloudOutChannels
 */
__global__ void reduceKernel(const BlockHeader* hdr, const float* scratch, int chunks, int maxN, float* out)
{
    __shared__ float part[kReduceGroups][kReduceLanes + 1];
    const int n = hdr->n;
    const int t = blockIdx.x * kReduceLanes + threadIdx.x;   // channel-major, as the scratch is laid out
    const int c = t / maxN;
    const int s = t - c * maxN;
    const bool live = c < kCloudChannels && s < n;
    float acc = 0.0f;
    if (live)
        for (int k = threadIdx.y; k < chunks; k += kReduceGroups) acc += scratch[static_cast<size_t>(k) * kCloudChannels * maxN + t];
    part[threadIdx.y][threadIdx.x] = acc;
    __syncthreads();
    if (threadIdx.y == 0 && live) {
        for (int q = 1; q < kReduceGroups; ++q) acc += part[q][threadIdx.x];
        out[s * kCloudOutChannels + c] = acc;
    }
}

/**
 * @brief A uniform number in [0, 1) for one mode of the modal hall.
 * @param mode  the mode
 * @param salt  which of its numbers
 * @return      the number
 */
__device__ __forceinline__ float modalUni(int mode, std::uint32_t salt)
{
    return static_cast<float>(hash32(static_cast<std::uint32_t>(mode) * 0x9E3779B9U ^ (salt * 0x85EBCA6BU)) >> 8) * (1.0f / 16777216.0f);
}

/**
 * @brief The modal hall: one thread per mode, a complex one-pole driven by the block the voices
 *        just made, summed per sample over the CUDA block (warp shuffles, then the warps in order).
 *
 * Each mode works out where it stands and how fast it dies from the header, every block -- a
 * hash gives it its place, its pitch if it is tuned, its drift and its two output phases -- so a
 * change of key, decay or tuning reaches it at the next block, and a ringing mode retunes without
 * a click (its state is kept; only the rotation changes). The input is the near and the far plane
 * together: the hall is a space both stand in.
 * @param hdr      the block header (its `modal` settings)
 * @param in       the block's frames, kCloudOutChannels each; the four voice channels are read
 * @param state    per mode its complex state, carried from block to block
 * @param scratch  per CUDA block two channels of maxN sums
 * @param maxN     the scratch stride
 * @param land     the landscape (the painted hall reads it)
 */
__global__ void __launch_bounds__(kModalThreads)
modalKernel(const BlockHeader* hdr, const float* in, float2* state, float* scratch, int maxN, const float* land, const float* memory)
{
    __shared__ float sIn[2][kModalTile];
    __shared__ float sAcc[kModalThreads / 32][2][kModalTile];
    const ModalHall& mh = hdr->modal;
    const int n = hdr->n;
    const int mode = static_cast<int>(blockIdx.x) * kModalThreads + static_cast<int>(threadIdx.x);
    float* out = scratch + static_cast<size_t>(blockIdx.x) * 2 * maxN;
    if (!mh.on) {   // resting: silent, and cleared so that it starts from silence when it is opened again
        state[mode] = make_float2(0.0f, 0.0f);
        for (int s = threadIdx.x; s < n; s += kModalThreads) { out[s] = 0.0f; out[maxN + s] = 0.0f; }
        return;
    }
    const float fs = hdr->sampleRate;
    const float lo = fmaxf(mh.low, 20.0f);
    const float hi = fminf(16000.0f, 0.45f * fs);
    // Where it stands: on one of the scale's pitches (the share `tune` of the modes), or anywhere
    // between the lowest mode and 16 kHz, evenly in hertz (so the hall is as dense at the top as
    // at the bottom, where a decaying mode is as wide).
    float f = lo + (hi - lo) * modalUni(mode, 4);
    float weight = 1.0f;
    if (mh.tunedCount > 0 && modalUni(mode, 1) < mh.tune) {
        const int k = static_cast<int>(hash32(static_cast<std::uint32_t>(mode) ^ 0xA511E9B3U) % static_cast<std::uint32_t>(mh.tunedCount));
        const float p = mh.tuned[k];
        f = p * exp2f(mh.spread * (modalUni(mode, 2) + modalUni(mode, 3) - 1.0f) * (1.0f / 1200.0f));
        // Tuned modes stand closer together than spread ones -- all of them within a few cents of a
        // few dozen pitches -- and a sound on those pitches would wake them all: weighted down by
        // that density (the pitches' band against the whole range), the tuned share colours the
        // hall instead of making it louder.
        const float cents = fmaxf(mh.spread + mh.drift, 1.0f);
        const float bandHz = static_cast<float>(mh.tunedCount) * p * cents * (2.0f * 0.000577623f);   // ln 2 / 1200
        weight = sqrtf(fminf(4.0f, bandHz / (hi - lo)));
    }
    // The drift: a slow sine of its own; every rate a whole number of cycles in 1000 s, the clock's wrap.
    const float rate = (20.0f + static_cast<float>(hash32(static_cast<std::uint32_t>(mode) ^ 0x5BD1E995U) % 100U)) * 0.001f;
    float ph = rate * mh.time + modalUni(mode, 6);
    ph -= floorf(ph);
    f *= exp2f(mh.drift * (1.0f / 1200.0f) * __sinf(6.2831853f * ph));
    const bool silent = f < 10.0f || f >= 0.49f * fs;
    // How fast it dies: the decay at and below 250 Hz, each octave above 2^damp times faster; with
    // breath, every mode's decay wanders by up to an octave on a slow sine of its own (again a whole
    // number of cycles in 1000 s), so the hall's colour moves while it rings.
    float breathe = 0.0f;
    if (mh.breath > 0.0f) {
        const float br = (10.0f + static_cast<float>(hash32(static_cast<std::uint32_t>(mode) ^ 0x27D4EB2FU) % 40U)) * 0.001f;
        float bp = br * mh.time + modalUni(mode, 10);
        bp -= floorf(bp);
        breathe = mh.breath * __sinf(6.2831853f * bp);
    }
    // The hall's memory (Grow): where much was heard, the mode rings longer (growth) or shorter
    // (erosion) -- up to 2^1.5 at seven times the average, half an octave the other way where nothing was.
    float grown = 0.0f;
    if (mh.grow != 0.0f) grown = mh.grow * fminf(fmaxf(log2f(1.0f + memory[hallBin(f)]) - 1.0f, -0.5f), 1.5f);
    const float T = fmaxf(0.05f, mh.decay * exp2f(breathe + grown - mh.damp * fmaxf(0.0f, log2f(f * (1.0f / 250.0f)))));
    const float r = mh.freeze ? 0.99999988f : __expf(-6.9077553f / (T * fs));
    float sw, cw;
    sincosf(6.2831853f * f / fs, &sw, &cw);
    const float ar = r * cw, ai = r * sw;
    // Driven with sqrt(2 (1 - r^2) / N): white noise in, white noise of the same power out, whatever the decay.
    const float g = (silent || mh.freeze) ? 0.0f : sqrtf(2.0f * fmaxf(1.0f - r * r, 0.0f) / static_cast<float>(kModalModes)) * weight;
    float inR, inL;
    __sincosf(1.5707963f * modalUni(mode, 7), &inR, &inL);   // its share of the left and the right input
    const float phL = 6.2831853f * modalUni(mode, 8);
    const float phR = phL + mh.width * 3.1415927f * (2.0f * modalUni(mode, 9) - 1.0f);
    float oLs, oLc, oRs, oRc;
    __sincosf(phL, &oLs, &oLc);
    __sincosf(phR, &oRs, &oRc);
    if (silent) { oLs = oLc = oRs = oRc = 0.0f; }
    // The painted hall: the mode heard through the landscape at its frequency, as the landscape
    // moves on -- the tail draws the aurora, the rain, the tides. Made up by the landscape's RMS.
    if (mh.paint > 0.0f && hdr->landColumns > 0 && hdr->landRows > 0) {
        const float pos = fminf(fmaxf(log2f(fmaxf(f, kLandLowHz) * (1.0f / kLandLowHz)) * (1.0f / 9.96578428f), 0.0f), 1.0f);
        const float col = pos * static_cast<float>(hdr->landColumns - 1);
        const float rows = static_cast<float>(hdr->landRows);
        float row = mh.paintTime / fmaxf(hdr->landRowSeconds, 1.0e-4f);
        row -= floorf(row / rows) * rows;
        const int r0 = min(static_cast<int>(row), hdr->landRows - 1), r1 = r0 + 1 == hdr->landRows ? 0 : r0 + 1;
        const int c0 = min(static_cast<int>(col), hdr->landColumns - 1), c1 = min(c0 + 1, hdr->landColumns - 1);
        const float fr = row - static_cast<float>(r0), fc = col - static_cast<float>(c0);
        const float* a = land + static_cast<size_t>(r0) * hdr->landColumns;
        const float* b = land + static_cast<size_t>(r1) * hdr->landColumns;
        const float top = a[c0] + (a[c1] - a[c0]) * fc, bottom = b[c0] + (b[c1] - b[c0]) * fc;
        const float k = 1.0f - mh.paint + mh.paint * (top + (bottom - top) * fr) * mh.paintNorm;
        oLs *= k; oLc *= k; oRs *= k; oRc *= k;
    }
    float2 y = state[mode];
    const int lane = static_cast<int>(threadIdx.x) & 31, warp = static_cast<int>(threadIdx.x) >> 5;
    for (int t0 = 0; t0 < n; t0 += kModalTile) {
        const int tn = min(kModalTile, n - t0);
        for (int s = threadIdx.x; s < tn; s += kModalThreads) {
            const float* fr = in + static_cast<size_t>(t0 + s) * kCloudOutChannels;
            sIn[0][s] = fr[0] + fr[2];
            sIn[1][s] = fr[1] + fr[3];
        }
        __syncthreads();
        for (int i = 0; i < tn; ++i) {
            const float x = g * (inL * sIn[0][i] + inR * sIn[1][i]);
            const float yr = fmaf(ar, y.x, fmaf(-ai, y.y, x));
            const float yi = fmaf(ar, y.y, ai * y.x);
            y.x = yr;
            y.y = yi;
            float cL = yr * oLc - yi * oLs;
            float cR = yr * oRc - yi * oRs;
            for (int off = 16; off > 0; off >>= 1) {
                cL += __shfl_down_sync(0xFFFFFFFFU, cL, off);
                cR += __shfl_down_sync(0xFFFFFFFFU, cR, off);
            }
            if (lane == 0) { sAcc[warp][0][i] = cL; sAcc[warp][1][i] = cR; }
        }
        __syncthreads();
        for (int s = threadIdx.x; s < 2 * tn; s += kModalThreads) {
            const int c = s / tn, i = s - c * tn;
            float a = 0.0f;
            for (int w = 0; w < kModalThreads / 32; ++w) a += sAcc[w][c][i];
            out[c * maxN + t0 + i] = a;
        }
        __syncthreads();
    }
    state[mode] = y;
}

/**
 * @brief The modal hall's reduce: the CUDA blocks' sums added sample by sample, in a fixed order,
 *        into the frames' last two channels.
 * @param hdr      the block header
 * @param scratch  the modal kernel's per-block sums
 * @param maxN     the scratch stride
 * @param out      n frames of kCloudOutChannels; channels 4 and 5 are written
 */
__global__ void modalReduceKernel(const BlockHeader* hdr, const float* scratch, int maxN, float* out)
{
    __shared__ float part[kReduceGroups][kReduceLanes + 1];
    const int n = hdr->n;
    const int t = static_cast<int>(blockIdx.x) * kReduceLanes + static_cast<int>(threadIdx.x);   // channel-major, as the scratch is laid out
    const int c = t / maxN;
    const int s = t - c * maxN;
    const bool live = c < 2 && s < n;
    float acc = 0.0f;
    if (live)
        for (int k = threadIdx.y; k < kModalBlocks; k += kReduceGroups) acc += scratch[static_cast<size_t>(k) * 2 * maxN + t];
    part[threadIdx.y][threadIdx.x] = acc;
    __syncthreads();
    if (threadIdx.y == 0 && live) {
        for (int q = 1; q < kReduceGroups; ++q) acc += part[q][threadIdx.x];
        out[s * kCloudOutChannels + kCloudChannels + c] = acc;
    }
}

/**
 * @brief A 1024-point complex FFT in shared memory, by every thread of a kEchoThreads block:
 *        bit-reversed order, then ten stages of one butterfly per thread (unnormalised).
 * @param a        the 1024 values, transformed in place
 * @param inverse  true: the inverse transform (times 1024)
 */
__device__ void fft1024(float2* a, bool inverse)
{
    for (int i = threadIdx.x; i < kEchoN; i += kEchoThreads) {
        const int j = static_cast<int>(__brev(static_cast<unsigned>(i)) >> 22);
        if (j > i) { const float2 tmp = a[i]; a[i] = a[j]; a[j] = tmp; }
    }
    __syncthreads();
    const int t = static_cast<int>(threadIdx.x);
    for (int len = 2; len <= kEchoN; len <<= 1) {
        const int half = len >> 1;
        const int pos = t % half;
        const int i0 = (t / half) * len + pos, i1 = i0 + half;
        float sn, cs;
        sincospif((inverse ? 2.0f : -2.0f) * static_cast<float>(pos) / static_cast<float>(len), &sn, &cs);
        const float2 u = a[i0], v = a[i1];
        const float2 w = make_float2(v.x * cs - v.y * sn, v.x * sn + v.y * cs);
        a[i0] = make_float2(u.x + w.x, u.y + w.y);
        a[i1] = make_float2(u.x - w.x, u.y - w.y);
        __syncthreads();
    }
}

/**
 * @brief The spectral echo, one CUDA block: the block into the history, a Hann-windowed frame
 *        through the FFT (left as the real part, right as the imaginary), every band of each ear
 *        through its own delay line with feedback, back through the inverse FFT, overlapped into
 *        the output; the oldest block of the overlap is finished and written to the frames' last
 *        two channels.
 * @param hdr   the block header (its `echo`)
 * @param io    the block's frames, kCloudOutChannels each: the voices read, channels 6 and 7 written
 * @param hist  the input history, two rings of kEchoN
 * @param acc   the overlap, two rings of kEchoN on the history's indices
 * @param ring  per ear and band kEchoSlots spectra: the delay lines
 * @param st    the history's write index and the frame count
 */
__global__ void __launch_bounds__(kEchoThreads)
spectralEchoKernel(const BlockHeader* hdr, float* io, float* hist, float* acc, float2* ring, int* st)
{
    __shared__ float2 z[kEchoN];
    const SpectralEcho& E = hdr->echo;
    const int n = hdr->n;
    const int t = static_cast<int>(threadIdx.x);
    if (E.clear) {
        for (int i = t; i < 2 * kEchoBins * kEchoSlots; i += kEchoThreads) ring[i] = make_float2(0.0f, 0.0f);
        for (int i = t; i < 2 * kEchoN; i += kEchoThreads) { hist[i] = 0.0f; acc[i] = 0.0f; }
        if (t == 0) { st[0] = 0; st[1] = 0; }
        __syncthreads();
    }
    if (!E.on || n > kEchoN / 4) {   // off, or blocks too long for the window's overlap: silent
        for (int s = t; s < n; s += kEchoThreads) { io[s * kCloudOutChannels + 6] = 0.0f; io[s * kCloudOutChannels + 7] = 0.0f; }
        return;
    }
    const int w0 = st[0], frame = st[1];
    for (int s = t; s < n; s += kEchoThreads) {   // the block over the history's oldest samples
        const float* f = io + static_cast<size_t>(s) * kCloudOutChannels;
        const int i = (w0 + s) & (kEchoN - 1);
        hist[i] = f[0] + f[2];
        hist[kEchoN + i] = f[1] + f[3];
    }
    __syncthreads();
    const int start = (w0 + n) & (kEchoN - 1);   // the frame's oldest sample
    for (int k = t; k < kEchoN; k += kEchoThreads) {
        const int i = (start + k) & (kEchoN - 1);
        const float win = 0.5f - 0.5f * cospif(2.0f * static_cast<float>(k) / static_cast<float>(kEchoN));
        z[k] = make_float2(win * hist[i], win * hist[kEchoN + i]);
    }
    __syncthreads();
    fft1024(z, false);
    // Every band: the two ears' spectra apart, each through its line, the delayed ones kept.
    const float fs = hdr->sampleRate;
    float2 yl[2], yr[2];
    const int bins = t == 0 ? 2 : 1;
    for (int b = 0; b < bins; ++b) {
        const int k = b == 0 ? t : kEchoN / 2;
        const float2 Zk = z[k], Zm = z[(kEchoN - k) & (kEchoN - 1)];
        const float2 xl = make_float2(0.5f * (Zk.x + Zm.x), 0.5f * (Zk.y - Zm.y));
        const float2 xr = make_float2(0.5f * (Zk.y + Zm.y), -0.5f * (Zk.x - Zm.x));
        const float fk = static_cast<float>(k) * fs / static_cast<float>(kEchoN);
        const float place = fminf(fmaxf(log2f(fmaxf(fk, 20.0f) * (1.0f / 20.0f)) * (1.0f / 9.96578428f), 0.0f), 1.0f);
        const float scatter = 1.0f + 0.75f * E.scatter * (2.0f * static_cast<float>(hash32(static_cast<std::uint32_t>(k) ^ 0x3C6EF372U) >> 8) * (1.0f / 16777216.0f) - 1.0f);
        const float d = E.frames * exp2f(2.0f * E.tilt * (place - 0.5f)) * scatter;
        const int di = min(max(static_cast<int>(lrintf(d)), 1), kEchoSlots - 1);
        const float fb = E.feedback * (1.0f - E.damp * place);
        const int slot = frame & (kEchoSlots - 1), back = (frame - di) & (kEchoSlots - 1);
        float2* rl = ring + static_cast<size_t>(k) * kEchoSlots;
        float2* rr = ring + static_cast<size_t>(kEchoBins + k) * kEchoSlots;
        yl[b] = rl[back];
        yr[b] = rr[back];
        rl[slot] = make_float2(xl.x + fb * yl[b].x, xl.y + fb * yl[b].y);
        rr[slot] = make_float2(xr.x + fb * yr[b].x, xr.y + fb * yr[b].y);
    }
    __syncthreads();
    for (int b = 0; b < bins; ++b) {   // the two ears together again: left real, right imaginary
        const int k = b == 0 ? t : kEchoN / 2;
        z[k] = make_float2(yl[b].x - yr[b].y, yl[b].y + yr[b].x);
        if (k > 0 && k < kEchoN / 2) z[kEchoN - k] = make_float2(yl[b].x + yr[b].y, -yl[b].y + yr[b].x);
    }
    __syncthreads();
    fft1024(z, true);
    // Overlap-add under the same window: Hann times Hann over N / n frames sums to 3/8 N / n.
    const float norm = 1.0f / (static_cast<float>(kEchoN) * 0.375f * static_cast<float>(kEchoN) / static_cast<float>(n));
    for (int k = t; k < kEchoN; k += kEchoThreads) {
        const int i = (start + k) & (kEchoN - 1);
        const float win = (0.5f - 0.5f * cospif(2.0f * static_cast<float>(k) / static_cast<float>(kEchoN))) * norm;
        acc[i] += win * z[k].x;
        acc[kEchoN + i] += win * z[k].y;
    }
    __syncthreads();
    for (int s = t; s < n; s += kEchoThreads) {   // the oldest block is finished: out, and cleared for the next
        const int i = (start + s) & (kEchoN - 1);
        io[s * kCloudOutChannels + 6] = acc[i];
        io[s * kCloudOutChannels + 7] = acc[kEchoN + i];
        acc[i] = 0.0f;
        acc[kEchoN + i] = 0.0f;
    }
    if (t == 0) { st[0] = (w0 + n) & (kEchoN - 1); st[1] = frame + 1; }
}

/**
 * @brief A 4096-point complex FFT in shared memory, by every thread of a kMeltThreads block:
 *        bit-reversed order, then twelve stages of two butterflies per thread (unnormalised).
 * @param a        the 4096 values, transformed in place
 * @param inverse  true: the inverse transform (times 4096)
 */
__device__ void fft4096(float2* a, bool inverse)
{
    for (int i = threadIdx.x; i < kMeltN; i += kMeltThreads) {
        const int j = static_cast<int>(__brev(static_cast<unsigned>(i)) >> 20);
        if (j > i) { const float2 tmp = a[i]; a[i] = a[j]; a[j] = tmp; }
    }
    __syncthreads();
    for (int len = 2; len <= kMeltN; len <<= 1) {
        const int half = len >> 1;
        for (int b = threadIdx.x; b < kMeltN / 2; b += kMeltThreads) {
            const int pos = b & (half - 1);
            const int i0 = (b / half) * len + pos, i1 = i0 + half;
            float sn, cs;
            sincospif((inverse ? 2.0f : -2.0f) * static_cast<float>(pos) / static_cast<float>(len), &sn, &cs);
            const float2 u = a[i0], v = a[i1];
            const float2 w = make_float2(v.x * cs - v.y * sn, v.x * sn + v.y * cs);
            a[i0] = make_float2(u.x + w.x, u.y + w.y);
            a[i1] = make_float2(u.x - w.x, u.y - w.y);
        }
        __syncthreads();
    }
}

/**
 * @brief A sum of two double-floats (a value as an unevaluated sum hi + lo of two floats, about 48
 *        bits): Knuth's two-sum, then renormalised. Only additions -- nothing to contract or reorder.
 * @param a  one (x the high part, y the low)
 * @param b  the other
 * @return   a + b
 */
__device__ float2 dfAdd(float2 a, float2 b)
{
    const float s = a.x + b.x;
    const float bb = s - a.x;
    const float e = (a.x - (s - bb)) + (b.x - bb) + a.y + b.y;
    const float hi = s + e;
    return make_float2(hi, e - (hi - s));
}

/** @brief The difference of two double-floats. @param a  one @param b  the other @return a - b */
__device__ float2 dfSub(float2 a, float2 b) { return dfAdd(a, make_float2(-b.x, -b.y)); }

/**
 * @brief A cumulative sum below a point on the band axis, band b covering [b, b + 1): the
 *        inclusive sums at whole bands, linear within one.
 * @param c  the inclusive sums as double-floats, kMeltBands (c[b] = the sum over bands 0 .. b)
 * @param x  the point, 0 .. kMeltBands
 * @return   the sum below it
 */
__device__ float2 meltBelow(const float2* c, float x)
{
    const int j = min(static_cast<int>(x), kMeltBands);
    const float2 lo = j > 0 ? c[j - 1] : make_float2(0.0f, 0.0f);
    if (j >= kMeltBands) return lo;
    const float2 d = dfSub(c[j], lo);
    return dfAdd(lo, make_float2((x - static_cast<float>(j)) * (d.x + d.y), 0.0f));
}

/**
 * @brief The melt, one CUDA block: the block into the history; every kMeltHop samples a frame --
 *        Hann window, FFT (left real, right imaginary), each band's energy held and its heard
 *        frequency tracked, the held energy blurred across the bands (each band scatters it evenly
 *        over 1 + 2 kappa x bands around its centre x, gathered through a cumulative sum in double-float,
 *        so nothing is gained or lost), sung again
 *        at the tracked frequencies, inverse FFT, overlapped into the output; the oldest block of the
 *        overlap finished and written to channels 8 and 9.
 * @param hdr     the header (its `melt`)
 * @param io      the block's frames: the voices read, channels 8 and 9 written
 * @param hist    the input history, two rings of kMeltN
 * @param acc     the overlap, two rings of kMeltN on the history's indices
 * @param energy  the held energy per band, left then right
 * @param omega   each band's sung frequency, radians a sample
 * @param phase   each band's sung phase, left then right
 * @param heard   each band's last heard phase, left then right
 * @param st      the history's write index, the samples since the last frame
 */
__global__ void __launch_bounds__(kMeltThreads)
meltKernel(const BlockHeader* hdr, float* io, float* hist, float* acc, float* energy, float* omega, float* phase, float* heard, int* st)
{
    __shared__ float2 z[kMeltN];          // the frame; between the transforms the cumulative sums
    __shared__ float2 warpSum[2][32];     // the scan's warp totals, per ear
    const Melt& M = hdr->melt;
    const int n = hdr->n;
    const int t = static_cast<int>(threadIdx.x);
    constexpr float kTwoPi = 6.28318531f;
    if (M.clear) {
        for (int i = t; i < 2 * kMeltN; i += kMeltThreads) { hist[i] = 0.0f; acc[i] = 0.0f; }
        for (int b = t; b < kMeltBands; b += kMeltThreads) {
            energy[b] = energy[kMeltBands + b] = 0.0f;
            omega[b] = kTwoPi * static_cast<float>(b) / static_cast<float>(kMeltN);
            phase[b] = kTwoPi * static_cast<float>(hash32(static_cast<std::uint32_t>(b) ^ 0x4D454C54U) >> 8) * (1.0f / 16777216.0f);
            phase[kMeltBands + b] = kTwoPi * static_cast<float>(hash32(static_cast<std::uint32_t>(b) ^ 0x4D454C55U) >> 8) * (1.0f / 16777216.0f);
            heard[b] = heard[kMeltBands + b] = 0.0f;
        }
        if (t == 0) { st[0] = 0; st[1] = 0; }
        __syncthreads();
    }
    if (!M.on || n > kMeltHop) {   // off, or blocks longer than the hop: silent
        for (int s = t; s < n; s += kMeltThreads) { io[s * kCloudOutChannels + 8] = 0.0f; io[s * kCloudOutChannels + 9] = 0.0f; }
        return;
    }
    const int w0 = st[0], count = st[1] + n;
    for (int s = t; s < n; s += kMeltThreads) {
        const float* f = io + static_cast<size_t>(s) * kCloudOutChannels;
        const int i = (w0 + s) & (kMeltN - 1);
        hist[i] = f[0] + f[2];
        hist[kMeltN + i] = f[1] + f[3];
    }
    __syncthreads();
    const int start = (w0 + n) & (kMeltN - 1);   // the window's oldest sample
    if (count >= kMeltHop) {
        for (int k = t; k < kMeltN; k += kMeltThreads) {
            const int i = (start + k) & (kMeltN - 1);
            const float win = 0.5f - 0.5f * cospif(2.0f * static_cast<float>(k) / static_cast<float>(kMeltN));
            z[k] = make_float2(win * hist[i], win * hist[kMeltN + i]);
        }
        __syncthreads();
        fft4096(z, false);
        // Each of the thread's two bands (2t, 2t + 1): heard, held, tracked.
        const float hop = static_cast<float>(kMeltHop);
        float held[2][2];   // [band][ear]
        for (int q = 0; q < 2; ++q) {
            const int b = 2 * t + q;
            const float2 Zk = z[b], Zm = z[(kMeltN - b) & (kMeltN - 1)];
            const float2 xl = make_float2(0.5f * (Zk.x + Zm.x), 0.5f * (Zk.y - Zm.y));
            const float2 xr = make_float2(0.5f * (Zk.y + Zm.y), -0.5f * (Zk.x - Zm.x));
            const float eL = b > 0 ? xl.x * xl.x + xl.y * xl.y : 0.0f;
            const float eR = b > 0 ? xr.x * xr.x + xr.y * xr.y : 0.0f;
            // The heard frequency: the phase's advance over the hop, beyond the band centre's, wrapped.
            const float centre = kTwoPi * static_cast<float>(b) / static_cast<float>(kMeltN);
            const float pL = atan2f(xl.y, xl.x), pR = atan2f(xr.y, xr.x);
            float dL = pL - heard[b] - centre * hop, dR = pR - heard[kMeltBands + b] - centre * hop;
            dL -= kTwoPi * rintf(dL / kTwoPi);
            dR -= kTwoPi * rintf(dR / kTwoPi);
            heard[b] = pL;
            heard[kMeltBands + b] = pR;
            const float wIn = eL + eR > 1.0e-20f ? centre + (eL * dL + eR * dR) / ((eL + eR) * hop) : centre;
            held[q][0] = M.keep * energy[b] + (1.0f - M.keep) * eL;
            held[q][1] = M.keep * energy[kMeltBands + b] + (1.0f - M.keep) * eR;
            const float fresh = fminf(1.0f, (1.0f - M.keep) * (eL + eR) / (held[q][0] + held[q][1] + 1.0e-20f));
            float w = omega[b];
            w += fresh * (wIn - w);
            w += M.pull * (centre - w);
            omega[b] = w;
        }
        __syncthreads();   // every band read: z becomes the cumulative sums
        // The blur scatters band x's energy evenly over [x - h, x + h], h = 0.5 + kappa x (x its
        // centre): a density held / (1 + 2 kappa x). Band b then gathers the densities of the bands
        // whose centres lie in [(x - 0.5) / (1 + kappa), (x + 0.5) / (1 - kappa)] -- the cumulative
        // densities' difference. kappa 0 is the identity; the variance a frame is about (kappa x)^2 / 3.
        const float kappa = 1.7320508f * M.widen;
        float dens[2][2];
        for (int q = 0; q < 2; ++q)
            for (int ear = 0; ear < 2; ++ear)
                dens[q][ear] = held[q][ear] / (1.0f + 2.0f * kappa * (static_cast<float>(2 * t + q) + 0.5f));
        // Inclusive sums over the bands, per ear, in double-float: the thread's pair, a warp's scan by
        // shuffles, the warps' totals by the first warp -- all in a fixed order.
        float2* c = z;   // [ear * kMeltBands + b]
        const int lane = t & 31, warp = t >> 5;
        float2 excl[2], pair[2];
        for (int ear = 0; ear < 2; ++ear) {
            pair[ear] = dfAdd(make_float2(dens[0][ear], 0.0f), make_float2(dens[1][ear], 0.0f));
            float2 s = pair[ear];
            for (int off = 1; off < 32; off <<= 1) {
                const float2 o = make_float2(__shfl_up_sync(0xFFFFFFFFU, s.x, off), __shfl_up_sync(0xFFFFFFFFU, s.y, off));
                if (lane >= off) s = dfAdd(o, s);
            }
            if (lane == 31) warpSum[ear][warp] = s;
            excl[ear] = dfSub(s, pair[ear]);
        }
        __syncthreads();
        if (warp == 0) {
            for (int ear = 0; ear < 2; ++ear) {
                const float2 own = warpSum[ear][lane];
                float2 s = own;
                for (int off = 1; off < 32; off <<= 1) {
                    const float2 o = make_float2(__shfl_up_sync(0xFFFFFFFFU, s.x, off), __shfl_up_sync(0xFFFFFFFFU, s.y, off));
                    if (lane >= off) s = dfAdd(o, s);
                }
                warpSum[ear][lane] = dfSub(s, own);   // exclusive
            }
        }
        __syncthreads();
        for (int ear = 0; ear < 2; ++ear) {
            const float2 base = dfAdd(warpSum[ear][warp], excl[ear]);
            c[ear * kMeltBands + 2 * t] = dfAdd(base, make_float2(dens[0][ear], 0.0f));
            c[ear * kMeltBands + 2 * t + 1] = dfAdd(base, pair[ear]);
        }
        __syncthreads();
        float out[2][2];
        for (int q = 0; q < 2; ++q) {
            const int b = 2 * t + q;
            const float x = static_cast<float>(b) + 0.5f;
            const float lo = fmaxf(1.0f, (x - 0.5f) / (1.0f + kappa)), hi = fminf(static_cast<float>(kMeltBands), (x + 0.5f) / (1.0f - kappa));
            for (int ear = 0; ear < 2; ++ear) {
                const float2* ce = c + ear * kMeltBands;
                float e = 0.0f;
                if (b > 0 && hi > lo) {
                    const float2 d = dfSub(meltBelow(ce, hi), meltBelow(ce, lo));
                    e = d.x + d.y;
                }
                out[q][ear] = fmaxf(e, 0.0f);
            }
        }
        __syncthreads();   // the sums read: z becomes the sung spectrum
        for (int q = 0; q < 2; ++q) {
            const int b = 2 * t + q;
            energy[b] = out[q][0];
            energy[kMeltBands + b] = out[q][1];
            float2 y[2];
            for (int ear = 0; ear < 2; ++ear) {
                const float det = M.detune * (2.0f * static_cast<float>(hash32(static_cast<std::uint32_t>(b) ^ (ear ? 0x44455452U : 0x4445544CU)) >> 8) * (1.0f / 16777216.0f) - 1.0f);
                float ph = phase[ear * kMeltBands + b] + (omega[b] + det) * hop;
                ph -= kTwoPi * rintf(ph / kTwoPi);
                phase[ear * kMeltBands + b] = ph;
                float sn, cs;
                __sincosf(ph, &sn, &cs);
                const float mag = b > 0 ? sqrtf(out[q][ear]) : 0.0f;
                y[ear] = make_float2(mag * cs, mag * sn);
            }
            z[b] = make_float2(y[0].x - y[1].y, y[0].y + y[1].x);
            if (b > 0) z[kMeltN - b] = make_float2(y[0].x + y[1].y, -y[0].y + y[1].x);
        }
        if (t == 0) z[kMeltN / 2] = make_float2(0.0f, 0.0f);
        __syncthreads();
        fft4096(z, true);
        const float norm = 1.0f / (static_cast<float>(kMeltN) * 0.375f * static_cast<float>(kMeltN) / hop);
        for (int k = t; k < kMeltN; k += kMeltThreads) {
            const int i = (start + k) & (kMeltN - 1);
            const float win = (0.5f - 0.5f * cospif(2.0f * static_cast<float>(k) / static_cast<float>(kMeltN))) * norm;
            acc[i] += win * z[k].x;
            acc[kMeltN + i] += win * z[k].y;
        }
        __syncthreads();
    }
    for (int s = t; s < n; s += kMeltThreads) {   // the oldest block is finished: out, and cleared for the next
        const int i = (start + s) & (kMeltN - 1);
        io[s * kCloudOutChannels + 8] = acc[i];
        io[s * kCloudOutChannels + 9] = acc[kMeltN + i];
        acc[i] = 0.0f;
        acc[kMeltN + i] = 0.0f;
    }
    if (t == 0) { st[0] = (w0 + n) & (kMeltN - 1); st[1] = count >= kMeltHop ? count - kMeltHop : count; }
}

/**
 * @brief The instrument listening to itself, one CUDA block: the voices' block into a history, a
 *        Hann-windowed frame through the FFT, and three numbers -- the level, the energy's centre
 *        on the log axis, the change (the magnitudes' rise against the last frame over their sum).
 *        Summed in a fixed order.
 * @param hdr    the header
 * @param io     the block's frames (the voices' four channels are read)
 * @param hist   the history, two rings of kEchoN
 * @param prev   the last frame's magnitudes, kEchoBins
 * @param st     the history's write index
 * @param stats  receives level (energy), centroid, flux at [2], [3], [4]
 */
__global__ void __launch_bounds__(kEchoThreads)
listenKernel(const BlockHeader* hdr, const float* io, float* hist, float* prev, int* st, float* stats)
{
    __shared__ float2 z[kEchoN];
    __shared__ float red[3][kEchoThreads];
    const int n = hdr->n;
    const int t = static_cast<int>(threadIdx.x);
    const int w0 = st[0];
    for (int s = t; s < n; s += kEchoThreads) {
        const float* f = io + static_cast<size_t>(s) * kCloudOutChannels;
        const int i = (w0 + s) & (kEchoN - 1);
        hist[i] = f[0] + f[2];
        hist[kEchoN + i] = f[1] + f[3];
    }
    __syncthreads();
    const int start = (w0 + n) & (kEchoN - 1);
    for (int k = t; k < kEchoN; k += kEchoThreads) {
        const int i = (start + k) & (kEchoN - 1);
        const float win = 0.5f - 0.5f * cospif(2.0f * static_cast<float>(k) / static_cast<float>(kEchoN));
        z[k] = make_float2(win * hist[i], win * hist[kEchoN + i]);
    }
    __syncthreads();
    fft1024(z, false);
    const float fs = hdr->sampleRate;
    float e = 0.0f, c = 0.0f, rise = 0.0f, sum = 0.0f;
    const int bins = t == 0 ? 2 : 1;
    for (int b = 0; b < bins; ++b) {
        const int k = b == 0 ? t : kEchoN / 2;
        const float2 Zk = z[k], Zm = z[(kEchoN - k) & (kEchoN - 1)];
        const float xr = 0.5f * (Zk.x + Zm.x), xi = 0.5f * (Zk.y - Zm.y);
        const float yr = 0.5f * (Zk.y + Zm.y), yi = -0.5f * (Zk.x - Zm.x);
        const float en = xr * xr + xi * xi + yr * yr + yi * yi;
        const float mag = sqrtf(en);
        const float fk = static_cast<float>(k) * fs / static_cast<float>(kEchoN);
        const float place = fminf(fmaxf(log2f(fmaxf(fk, 20.0f) * (1.0f / 20.0f)) * (1.0f / 9.96578428f), 0.0f), 1.0f);
        e += en;
        c += en * place;
        rise += fmaxf(0.0f, mag - prev[k]);
        sum += mag;
        prev[k] = mag;
    }
    red[0][t] = e;
    red[1][t] = c;
    red[2][t] = rise;
    __syncthreads();
    for (int s = kEchoThreads / 2; s > 0; s >>= 1) {
        if (t < s) { red[0][t] += red[0][t + s]; red[1][t] += red[1][t + s]; red[2][t] += red[2][t + s]; }
        __syncthreads();
    }
    // The magnitudes' sum for the flux, in a second pass over the energy's shared row (its total kept first).
    __shared__ float total[2];
    if (t == 0) { total[0] = red[0][0]; total[1] = red[1][0]; }
    __syncthreads();
    red[1][t] = sum;
    __syncthreads();
    for (int s = kEchoThreads / 2; s > 0; s >>= 1) {
        if (t < s) red[1][t] += red[1][t + s];
        __syncthreads();
    }
    if (t == 0) {
        // Energy to a level: a full-scale sine in one ear gives (N/2)^2 (3/8) per bin pair under Hann.
        const float ref = 0.375f * 0.25f * static_cast<float>(kEchoN) * static_cast<float>(kEchoN);
        stats[2] = 10.0f * log10f(total[0] / ref + 1.0e-12f);
        stats[3] = total[0] > 1.0e-12f ? total[1] / total[0] : 0.0f;
        stats[4] = red[1][0] > 1.0e-9f ? red[2][0] / red[1][0] : 0.0f;
        st[0] = (w0 + n) & (kEchoN - 1);
    }
}

/**
 * @brief Fills the reservoir once: the coupling uniform in +-sqrt(3/N) (unit spectral radius, scaled
 *        by `chaos` when it runs), the inputs' and the readouts' weights in +-1, a small start.
 * @param w      the coupling, N x N
 * @param win    the input weights, N x kReservoirInputs
 * @param rout   the readouts, kReservoirOutputs x N
 * @param x      the state, both buffers
 */
__global__ void reservoirInitKernel(float* w, float* win, float* rout, float* x)
{
    const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const float a = sqrtf(3.0f / static_cast<float>(kReservoirN));
    auto uni = [](size_t k, std::uint32_t salt) { return static_cast<float>(hash32(static_cast<std::uint32_t>(k) * 0x9E3779B9U ^ salt) >> 8) * (1.0f / 16777216.0f); };
    if (i < static_cast<size_t>(kReservoirN) * kReservoirN) w[i] = a * (2.0f * uni(i, 0x52455330U) - 1.0f);
    if (i < static_cast<size_t>(kReservoirN) * kReservoirInputs) win[i] = 2.0f * uni(i, 0x52455331U) - 1.0f;
    if (i < static_cast<size_t>(kReservoirOutputs) * kReservoirN) rout[i] = 2.0f * uni(i, 0x52455332U) - 1.0f;
    if (i < 2 * static_cast<size_t>(kReservoirN)) x[i] = 0.1f * (2.0f * uni(i % kReservoirN, 0x52455333U) - 1.0f);
}

/**
 * @brief One block of the reservoir: a warp per neuron sums its row of the coupling against the last
 *        state (lanes strided, then shuffled in a fixed order), adds its inputs, and leaks towards
 *        tanh of it. Reads one buffer, writes the other (by the block's parity).
 * @param hdr   the header
 * @param w     the coupling
 * @param win   the input weights
 * @param x     the state, two buffers of N
 */
__global__ void reservoirStepKernel(const BlockHeader* hdr, const float* w, const float* win, float* x)
{
    const Reservoir& R = hdr->reservoir;
    if (!R.on) return;
    const int warp = static_cast<int>((blockIdx.x * blockDim.x + threadIdx.x) >> 5), lane = static_cast<int>(threadIdx.x & 31);
    if (warp >= kReservoirN) return;
    const bool odd = (hdr->blockIndex & 1U) != 0;
    const float* xp = x + (odd ? kReservoirN : 0);
    float* xn = x + (odd ? 0 : kReservoirN);
    const float* row = w + static_cast<size_t>(warp) * kReservoirN;
    float s = 0.0f;
    for (int j = lane; j < kReservoirN; j += 32) s = fmaf(row[j], xp[j], s);
    for (int off = 16; off > 0; off >>= 1) s += __shfl_down_sync(0xFFFFFFFFU, s, off);
    if (lane == 0) {
        float in = 0.0f;
        for (int k = 0; k < kReservoirInputs; ++k) in += win[warp * kReservoirInputs + k] * R.u[k];
        const float bias = 0.2f * (static_cast<float>(hash32(static_cast<std::uint32_t>(warp) ^ 0x42494153U) >> 8) * (1.0f / 16777216.0f) - 0.5f);
        xn[warp] = (1.0f - R.leak) * xp[warp] + R.leak * tanhf(R.chaos * s + R.input * in + bias);
    }
}

/**
 * @brief The reservoir's readouts, one CUDA block: tanh of each fixed projection of the new state, summed in a fixed order.
 * @param hdr    the header
 * @param rout   the readouts
 * @param x      the state, two buffers
 * @param stats  receives the outputs at [5] .. [8]
 */
__global__ void reservoirReadKernel(const BlockHeader* hdr, const float* rout, const float* x, float* stats)
{
    __shared__ float red[256];
    const Reservoir& R = hdr->reservoir;
    if (!R.on) return;
    const bool odd = (hdr->blockIndex & 1U) != 0;
    const float* xn = x + (odd ? 0 : kReservoirN);
    const int t = static_cast<int>(threadIdx.x);
    for (int k = 0; k < kReservoirOutputs; ++k) {
        float s = 0.0f;
        for (int i = t; i < kReservoirN; i += 256) s += rout[k * kReservoirN + i] * xn[i];
        red[t] = s;
        __syncthreads();
        for (int h = 128; h > 0; h >>= 1) {
            if (t < h) red[t] += red[t + h];
            __syncthreads();
        }
        if (t == 0) stats[5 + k] = tanhf(red[0] * 2.0f * rsqrtf(static_cast<float>(kReservoirN)));
        __syncthreads();
    }
}

/**
 * @brief Writes a failed CUDA call into `error`.
 * @param e      the call's result
 * @param what   the call, for the message
 * @param error  receives the message (may be null)
 * @return       true when the call succeeded
 */
bool ok(cudaError_t e, const char* what, std::string* error)
{
    if (e == cudaSuccess) return true;
    if (error) *error = std::string(what) + ": " + cudaGetErrorString(e);
    return false;
}

} // namespace

/** @brief The CUDA side of CloudEngine. */
struct CloudEngine::Impl {
    CloudConfig cfg;                  ///< the layout
    int pcap = 0;                     ///< partials per voice in the tables, a multiple of kChunk
    int chunks = 0;                   ///< render CUDA blocks: voices * pcap / kChunk
    std::vector<CloudVoice> voices;   ///< the host's voices, edited by the caller
    /** @brief What a voice's power sum was computed for: when none of it changed, the sum is reused. */
    struct ShapeKey {
        float hz = -1.0f;          ///< fundamental
        int   harmonics = 0;       ///< harmonics summed
        float tilt = 0.0f;         ///< spectral tilt
        float oddEven = 0.0f;      ///< odd/even weight
        float brightness = 0.0f;   ///< brightness window
        float inharmonic = 0.0f;   ///< stretch
        float presenceDb = 0.0f;   ///< presence bell
        float distance = 0.0f;     ///< the presence's distance
        float lowCut = 0.0f;       ///< pad low cut
        float stretch = 1.0f;      ///< the harmonics' stretch exponent
        float shift = 0.0f;        ///< the partials' shift, Hz
        /** @brief Equal in every field. @param o the other key @return true when the sum can be reused */
        bool operator==(const ShapeKey& o) const
        {
            return hz == o.hz && harmonics == o.harmonics && tilt == o.tilt && oddEven == o.oddEven && brightness == o.brightness
                && inharmonic == o.inharmonic && presenceDb == o.presenceDb && distance == o.distance && lowCut == o.lowCut
                && stretch == o.stretch && shift == o.shift;
        }
    };
    std::vector<ShapeKey> shapeKey;   ///< per voice: what powerSum was made for
    std::vector<double> powerSum;     ///< per voice: the sum over its harmonics of the squared shape (tilt, odd/even, brightness, presence, low cut)
    int nPrev = 0;                    ///< the previous block's length, sent with the next
    int shapeCount = 0;               ///< shapes in the bank on the GPU
    int landColumns = 0;              ///< the landscape on the GPU: columns (0 = none)
    int landRows = 0;                 ///< its rows
    float landRowSeconds = 0.1f;      ///< seconds per row
    int device = 0;                   ///< the CUDA device
    cudaDeviceProp props{};           ///< its properties
    cudaStream_t stream = nullptr;    ///< the stream the graph runs on (highest priority)
    cudaEvent_t evStart = nullptr;    ///< recorded before the graph
    cudaEvent_t evStop = nullptr;     ///< recorded after it: wait() waits for this one
    cudaGraph_t graph = nullptr;      ///< the captured block
    cudaGraphExec_t exec = nullptr;   ///< its executable form
    unsigned char* hUpload = nullptr; ///< pinned host: header + voices as the GPU will read them
    unsigned char* dUpload = nullptr; ///< device: the same
    float* hOut = nullptr;            ///< pinned host: the finished block
    float* dOut = nullptr;            ///< device: the finished block
    float* dScratch = nullptr;        ///< device: per chunk a stereo block
    uint4* dRun = nullptr;            ///< device: per partial start phase, step, glide
    float4* dAmp = nullptr;           ///< device: per partial the four amplitudes (near L/R, far L/R) the last block ended on
    float4* dGains = nullptr;         ///< device: per partial three entries, the near, the far and the near cosine ramps of this block
    float2* dAmpQ = nullptr;          ///< device: per partial the near plane's cosine parts the last block ended on (binaural)
    float* dEchoHist = nullptr;       ///< device: the spectral echo's input history, two rings of kEchoN
    float* dEchoAcc = nullptr;        ///< device: its overlap, two rings of kEchoN
    float2* dEchoRing = nullptr;      ///< device: its delay lines, per ear and band kEchoSlots spectra
    int* dEchoState = nullptr;        ///< device: its history index and frame count
    float2* dFlockA = nullptr;        ///< device: the flock's state (offset, velocity), buffer A
    float2* dFlockB = nullptr;        ///< device: buffer B
    float* dSyncTheta = nullptr;      ///< device: per partial the sync's phase
    float* dSyncX = nullptr;          ///< device: per partial its place on its circle (Swarm)
    float2* dSyncOrder = nullptr;     ///< device: per voice and group the sync's mean phasors, three planes
    PartialSync sync;                 ///< the host's copy of the partial sync's settings
    Ripple ripple;                    ///< the host's copy of the ripple
    Reservoir reservoir;              ///< the host's copy of the reservoir's settings
    Melt melt;                        ///< the host's copy of the melt's settings
    SpectralComb comb;                ///< the host's copy of the spectral comb
    Turbulence turb;                  ///< the host's copy of the energy cascade's settings
    float* dCascadeA = nullptr;       ///< device: per voice and harmonic its power against the spectrum's, buffer A (even blocks read it)
    float* dCascadeB = nullptr;       ///< device: buffer B
    float* dFilterTable = nullptr;    ///< device: per voice the transported filter, kFilterBins gains
    unsigned long long* dHallHist = nullptr;   ///< device: per band the partials' power this block (fixed point)
    float* dHallMemory = nullptr;     ///< device: the hall's memory, per band
    float* dDensity = nullptr;        ///< device: the spectrum's density, per band
    float* dTau = nullptr;            ///< device: per partial its own clock
    float4* dOrbitA = nullptr;        ///< device: the Orbit field's state, buffer A
    float4* dOrbitB = nullptr;        ///< device: buffer B
    float* hHallMemory = nullptr;     ///< pinned host: the memory as the last snapshot brought it down
    float* hHallUp = nullptr;         ///< pinned host: a memory to send up
    bool hallUp = false;              ///< the next launch sends hHallUp up first
    bool hallDown = false;            ///< hHallMemory holds a memory
    float* dMeltHist = nullptr;       ///< device: the melt's input history, two rings of kMeltN
    float* dMeltAcc = nullptr;        ///< device: its overlap, two rings of kMeltN
    float* dMeltEnergy = nullptr;     ///< device: its held energy, per ear and band
    float* dMeltOmega = nullptr;      ///< device: its bands' sung frequencies
    float* dMeltPhase = nullptr;      ///< device: their sung phases, per ear
    float* dMeltHeard = nullptr;      ///< device: their last heard phases, per ear
    int* dMeltState = nullptr;        ///< device: its history index and the samples since its last frame
    float* dResW = nullptr;           ///< device: the reservoir's coupling, N x N
    float* dResIn = nullptr;          ///< device: its input weights
    float* dResOut = nullptr;         ///< device: its readouts
    float* dResX = nullptr;           ///< device: its state, two buffers
    float2* dLifeA = nullptr;         ///< device: the living field, buffer A
    float2* dLifeB = nullptr;         ///< device: buffer B
    float* dFluidA = nullptr;         ///< device: the Fluid's lattice, nine directions a cell, buffer A
    float* dFluidB = nullptr;         ///< device: buffer B
    float* dIsles = nullptr;          ///< device: the Islands' grid
    float* dLifeStats = nullptr;      ///< device: its mean and mean square, then what was heard (level, centroid, flux)
    float* dListenHist = nullptr;     ///< device: the listening's history, two rings of kEchoN
    float* dListenPrev = nullptr;     ///< device: its last frame's magnitudes
    int* dListenState = nullptr;      ///< device: its history index
    float* hLifeStats = nullptr;      ///< pinned host: the same, brought down every block
    float2* hLife = nullptr;          ///< pinned host: the field brought down with a snapshot
    std::vector<float> lifeCells;     ///< the field as the partials read it, for the picture
    bool lifeShown = false;           ///< lifeCells holds a field
    LifeField life;                   ///< the host's copy of the living landscape's settings
    int lifeParity = 0;               ///< which buffer holds the field before the next block's steps
    float2* dSnap = nullptr;          ///< device: the picture's snapshot, every kSnapStride-th partial (Hz, amplitude)
    float2* hSnap = nullptr;          ///< pinned host: the snapshot brought down
    bool snapAsked = false;           ///< the next launch brings it down
    bool snapInFlight = false;        ///< the launched block brings it down
    bool snapReady = false;           ///< a snapshot came down and was not taken yet
    Binaural binaural;                ///< the host's copy of the binaural placement
    SpectralEcho echo;                ///< the host's copy of the spectral echo's settings
    EnvShape* dShapes = nullptr;      ///< device: the envelope bank, kEnvShapes
    float* dLand = nullptr;           ///< device: the landscape, room for kLandMaxColumns * kLandMaxRows
    float2* dModal = nullptr;         ///< device: the modal hall's states, kModalModes
    float* dGainRing = nullptr;       ///< device: per voice kGainRing levels, for the dispersion
    SpectralField field;              ///< the host's copy of the field and the dispersion
    std::uint32_t blockIndex = 0;     ///< blocks launched since open()
    float* dModalScratch = nullptr;   ///< device: per modal CUDA block two channels of sums
    ModalHall modal;                  ///< the host's copy of the modal hall's settings
    float gpuMs = 0.0f;               ///< GPU time of the last finished block
    bool launched = false;            ///< a block is in flight

    /** @brief Bytes of the upload: the header and the voices. @return bytes */
    size_t uploadBytes() const { return sizeof(BlockHeader) + sizeof(CloudVoice) * static_cast<size_t>(cfg.voices); }

    /** @brief Frees whatever was allocated, in any state. */
    void release()
    {
        if (stream) cudaStreamSynchronize(stream);
        if (exec) cudaGraphExecDestroy(exec);
        if (graph) cudaGraphDestroy(graph);
        if (evStart) cudaEventDestroy(evStart);
        if (evStop) cudaEventDestroy(evStop);
        if (stream) cudaStreamDestroy(stream);
        if (hUpload) cudaFreeHost(hUpload);
        if (hOut) cudaFreeHost(hOut);
        if (hSnap) cudaFreeHost(hSnap);
        if (hLifeStats) cudaFreeHost(hLifeStats);
        if (hLife) cudaFreeHost(hLife);
        for (void* p : { static_cast<void*>(dUpload), static_cast<void*>(dOut), static_cast<void*>(dScratch), static_cast<void*>(dRun),
                         static_cast<void*>(dAmp), static_cast<void*>(dGains), static_cast<void*>(dShapes), static_cast<void*>(dLand),
                         static_cast<void*>(dModal), static_cast<void*>(dModalScratch), static_cast<void*>(dGainRing),
                         static_cast<void*>(dAmpQ), static_cast<void*>(dEchoHist), static_cast<void*>(dEchoAcc),
                         static_cast<void*>(dEchoRing), static_cast<void*>(dEchoState), static_cast<void*>(dSnap),
                         static_cast<void*>(dLifeA), static_cast<void*>(dLifeB), static_cast<void*>(dLifeStats),
                         static_cast<void*>(dSyncTheta), static_cast<void*>(dSyncOrder), static_cast<void*>(dFlockA), static_cast<void*>(dFlockB),
                         static_cast<void*>(dListenHist), static_cast<void*>(dListenPrev), static_cast<void*>(dListenState),
                         static_cast<void*>(dResW), static_cast<void*>(dResIn), static_cast<void*>(dResOut), static_cast<void*>(dResX),
                         static_cast<void*>(dMeltHist), static_cast<void*>(dMeltAcc), static_cast<void*>(dMeltEnergy),
                         static_cast<void*>(dMeltOmega), static_cast<void*>(dMeltPhase), static_cast<void*>(dMeltHeard), static_cast<void*>(dMeltState),
                         static_cast<void*>(dSyncX), static_cast<void*>(dFluidA), static_cast<void*>(dFluidB), static_cast<void*>(dIsles),
                         static_cast<void*>(dCascadeA), static_cast<void*>(dCascadeB), static_cast<void*>(dFilterTable),
                         static_cast<void*>(dHallHist), static_cast<void*>(dHallMemory), static_cast<void*>(dDensity), static_cast<void*>(dTau),
                         static_cast<void*>(dOrbitA), static_cast<void*>(dOrbitB) })
            if (p) cudaFree(p);
        exec = nullptr; graph = nullptr; evStart = evStop = nullptr; stream = nullptr;
        hUpload = dUpload = nullptr; hOut = dOut = dScratch = dLand = nullptr;
        dRun = nullptr; dAmp = nullptr; dGains = nullptr; dShapes = nullptr;
        dModal = nullptr; dModalScratch = nullptr; dGainRing = nullptr;
        dAmpQ = nullptr; dEchoHist = nullptr; dEchoAcc = nullptr; dEchoRing = nullptr; dEchoState = nullptr;
        dSnap = nullptr; hSnap = nullptr;
        dLifeA = dLifeB = nullptr; dLifeStats = nullptr; hLifeStats = nullptr; hLife = nullptr;
        dSyncTheta = nullptr; dSyncOrder = nullptr; dFlockA = nullptr; dFlockB = nullptr;
        dListenHist = nullptr; dListenPrev = nullptr; dListenState = nullptr;
        dResW = nullptr; dResIn = nullptr; dResOut = nullptr; dResX = nullptr;
        dMeltHist = dMeltAcc = dMeltEnergy = dMeltOmega = dMeltPhase = dMeltHeard = nullptr; dMeltState = nullptr;
        dSyncX = nullptr; dFluidA = dFluidB = nullptr; dIsles = nullptr; dCascadeA = dCascadeB = nullptr; dFilterTable = nullptr;
        if (hHallMemory) cudaFreeHost(hHallMemory);
        if (hHallUp) cudaFreeHost(hHallUp);
        dHallHist = nullptr; dHallMemory = dDensity = dTau = nullptr; dOrbitA = dOrbitB = nullptr; hHallMemory = hHallUp = nullptr;
    }

    /** @brief Waits for a block in flight, so the tables may be replaced. */
    void settle()
    {
        if (stream) cudaStreamSynchronize(stream);
        if (launched) { cudaEventElapsedTime(&gpuMs, evStart, evStop); launched = false; }
    }

    /**
     * @brief Fills a voice's derived fields: how its partials divide between the harmonic clouds and
     *        the air band, the harmonics below the cutoff (with the stiff string's stretch), members
     *        per harmonic, and the scales that make the voice's RMS its level -- normalised after the
     *        whole spectral shape, as Noctuary does, so brightness, odd/even, presence and the low cut
     *        colour the voice without changing its level (the filter, the distance and the motions do).
     * @param i  the voice
     * @param v  its upload copy, completed here
     */
    void derive(int i, CloudVoice& v)
    {
        const float fs = static_cast<float>(cfg.sampleRate);
        const float cutoff = std::min(20000.0f, 0.43f * fs);
        const int partials = std::clamp(v.partials, 1, cfg.partialsPerVoice);
        v.partials = partials;
        const int airPartials = (v.air > 0.0f && partials >= 2) ? std::clamp(static_cast<int>(std::lround(partials * kAirShare)), 1, partials - 1) : 0;
        const int HP = partials - airPartials;
        const float hz = std::max(v.hz, 1.0f);
        // Harmonics below the cutoff: f0 h^s sqrt(1 + B h^2) + shift < cutoff -- rising in h, so the
        // last one below is found by halving.
        const float s = std::clamp(v.stretch, 0.5f, 2.0f);
        int lo = 1, hi = HP;
        auto below = [&](int h) { return hz * partialRatio(h, std::max(0.0f, v.inharmonic), s) + v.shift < cutoff; };
        if (below(hi)) lo = hi;
        else
            while (hi - lo > 1) {
                const int mid = (lo + hi) / 2;
                (below(mid) ? lo : hi) = mid;
            }
        const int H = std::clamp(lo, 1, HP);
        v.harmonics = H;
        v.members = HP / H;
        v.harmonicPartials = HP;
        const ShapeKey key{ hz, H, v.tilt, v.oddEven, v.brightness, v.inharmonic, v.presenceDb, std::clamp(v.distance, 0.0f, 1.0f), v.lowCut, s, v.shift };
        if (!(shapeKey[i] == key)) {
            double sum = 0.0;
            for (int h = 1; h <= H; ++h) {
                const float fh = std::max(hz * partialRatio(h, v.inharmonic, s) + v.shift, 1.0f);
                const double a = static_cast<double>(harmonicShape(h, v.tilt, v.oddEven, v.brightness))
                               * presenceGain(fh, v.presenceDb, key.distance) * lowCutGain(fh, v.lowCut);
                sum += a * a;
            }
            powerSum[i] = std::max(sum, 1.0e-30);
            shapeKey[i] = key;
        }
        // Power of the voice: every member a sine (mean square 1/2), the shimmer's two sines add
        // their own mean square (0.7^2 + 0.3^2) / 2 times its depth squared. The envelopes are
        // the sound's own motion and are not compensated.
        const float shimmerMs = 1.0f + 0.29f * v.shimmer * v.shimmer;
        v.scale = static_cast<float>(v.level / std::sqrt(0.5 * v.members * powerSum[i] * shimmerMs));
        // The air band at `air` times the voice's level; its flicker's mean square is 1 + 0.29 * 0.8^2.
        v.airScale = airPartials > 0 ? static_cast<float>(v.air * v.level / std::sqrt(0.5 * airPartials * (1.0 + 0.29 * 0.64))) : 0.0f;
    }
};

CloudEngine::CloudEngine() = default;
CloudEngine::~CloudEngine() { close(); }

bool CloudEngine::open(const CloudConfig& config, std::string* error)
{
    close();
    const int n = config.blockSize;
    if (n < 32 || n > 4096 || (n & (n - 1)) != 0) { if (error) *error = "blockSize must be a power of two, 32 .. 4096"; return false; }
    if (config.voices < 1 || config.partialsPerVoice < 1 || config.sampleRate < 8000.0) { if (error) *error = "voices, partials and sample rate must be positive"; return false; }

    auto im = std::make_unique<Impl>();
    im->cfg = config;
    im->pcap = (config.partialsPerVoice + kChunk - 1) / kChunk * kChunk;
    im->chunks = config.voices * im->pcap / kChunk;
    im->voices.assign(config.voices, CloudVoice{});
    im->shapeKey.assign(config.voices, Impl::ShapeKey{});
    im->powerSum.assign(config.voices, 1.0);

    int count = 0;
    if (!ok(cudaGetDeviceCount(&count), "cudaGetDeviceCount", error)) return false;
    if (count < 1) { if (error) *error = "no CUDA device"; return false; }
    // The flags only take before the context exists; a second engine in the process keeps the first's.
    cudaSetDeviceFlags(config.spinWait ? cudaDeviceScheduleSpin : cudaDeviceScheduleBlockingSync);
    cudaGetLastError();
    Impl& m = *im;
    bool good = ok(cudaSetDevice(m.device), "cudaSetDevice", error)
             && ok(cudaGetDeviceProperties(&m.props, m.device), "cudaGetDeviceProperties", error);
    // A card this build has no code for (CMAKE_CUDA_ARCHITECTURES) says so here, by name, not at the
    // first block's launch.
    if (cudaFuncAttributes attr{}; good && cudaFuncGetAttributes(&attr, updateKernel) != cudaSuccess) {
        cudaGetLastError();
        if (error) *error = std::string(m.props.name) + " (compute " + std::to_string(m.props.major) + "." + std::to_string(m.props.minor)
                          + ") is not among the cards this build is made for";
        return false;
    }
    int lo = 0, hi = 0;
    good = good && ok(cudaDeviceGetStreamPriorityRange(&lo, &hi), "cudaDeviceGetStreamPriorityRange", error)
                && ok(cudaStreamCreateWithPriority(&m.stream, cudaStreamNonBlocking, hi), "cudaStreamCreateWithPriority", error)
                && ok(cudaEventCreate(&m.evStart), "cudaEventCreate", error)
                && ok(cudaEventCreateWithFlags(&m.evStop, config.spinWait ? cudaEventDefault : cudaEventBlockingSync), "cudaEventCreate", error);
    const size_t partialsTotal = static_cast<size_t>(config.voices) * m.pcap;
    const size_t landFloats = static_cast<size_t>(kLandMaxColumns) * kLandMaxRows;
    good = good && ok(cudaHostAlloc(&m.hUpload, m.uploadBytes(), cudaHostAllocDefault), "cudaHostAlloc upload", error)
                && ok(cudaHostAlloc(&m.hOut, sizeof(float) * kCloudOutChannels * n, cudaHostAllocDefault), "cudaHostAlloc out", error)
                && ok(cudaMalloc(&m.dUpload, m.uploadBytes()), "cudaMalloc upload", error)
                && ok(cudaMalloc(&m.dOut, sizeof(float) * kCloudOutChannels * n), "cudaMalloc out", error)
                && ok(cudaMalloc(&m.dScratch, sizeof(float) * kCloudChannels * n * static_cast<size_t>(m.chunks)), "cudaMalloc scratch", error)
                && ok(cudaMalloc(&m.dRun, sizeof(uint4) * partialsTotal), "cudaMalloc run", error)
                && ok(cudaMalloc(&m.dAmp, sizeof(float4) * partialsTotal), "cudaMalloc amp", error)
                && ok(cudaMalloc(&m.dGains, sizeof(float4) * 3 * partialsTotal), "cudaMalloc gains", error)
                && ok(cudaMalloc(&m.dAmpQ, sizeof(float2) * partialsTotal), "cudaMalloc ampQ", error)
                && ok(cudaMalloc(&m.dEchoHist, sizeof(float) * 2 * kEchoN), "cudaMalloc echo history", error)
                && ok(cudaMalloc(&m.dEchoAcc, sizeof(float) * 2 * kEchoN), "cudaMalloc echo overlap", error)
                && ok(cudaMalloc(&m.dEchoRing, sizeof(float2) * 2 * kEchoBins * static_cast<size_t>(kEchoSlots)), "cudaMalloc echo lines", error)
                && ok(cudaMalloc(&m.dEchoState, sizeof(int) * 2), "cudaMalloc echo state", error)
                && ok(cudaMalloc(&m.dSnap, sizeof(float2) * (partialsTotal / kSnapStride)), "cudaMalloc snapshot", error)
                && ok(cudaHostAlloc(&m.hSnap, sizeof(float2) * (partialsTotal / kSnapStride), cudaHostAllocDefault), "cudaHostAlloc snapshot", error)
                && ok(cudaMalloc(&m.dLifeA, sizeof(float2) * kLifeCells), "cudaMalloc life", error)
                && ok(cudaMalloc(&m.dLifeB, sizeof(float2) * kLifeCells), "cudaMalloc life", error)
                && ok(cudaMalloc(&m.dFluidA, sizeof(float) * 9 * kLifeCells), "cudaMalloc fluid", error)
                && ok(cudaMalloc(&m.dFluidB, sizeof(float) * 9 * kLifeCells), "cudaMalloc fluid", error)
                && ok(cudaMalloc(&m.dIsles, sizeof(float) * kIsleW * kIsleH), "cudaMalloc islands", error)
                && ok(cudaMalloc(&m.dCascadeA, sizeof(float) * partialsTotal), "cudaMalloc cascade", error)
                && ok(cudaMalloc(&m.dCascadeB, sizeof(float) * partialsTotal), "cudaMalloc cascade", error)
                && ok(cudaMalloc(&m.dFilterTable, sizeof(float) * kFilterBins * static_cast<size_t>(config.voices)), "cudaMalloc filter table", error)
                && ok(cudaMalloc(&m.dHallHist, sizeof(unsigned long long) * kHallBins), "cudaMalloc hall", error)
                && ok(cudaMalloc(&m.dHallMemory, sizeof(float) * kHallBins), "cudaMalloc hall", error)
                && ok(cudaMalloc(&m.dDensity, sizeof(float) * kHallBins), "cudaMalloc density", error)
                && ok(cudaMalloc(&m.dTau, sizeof(float) * partialsTotal), "cudaMalloc clocks", error)
                && ok(cudaMalloc(&m.dOrbitA, sizeof(float4) * partialsTotal), "cudaMalloc orbit", error)
                && ok(cudaMalloc(&m.dOrbitB, sizeof(float4) * partialsTotal), "cudaMalloc orbit", error)
                && ok(cudaHostAlloc(&m.hHallMemory, sizeof(float) * kHallBins, cudaHostAllocDefault), "cudaHostAlloc hall", error)
                && ok(cudaHostAlloc(&m.hHallUp, sizeof(float) * kHallBins, cudaHostAllocDefault), "cudaHostAlloc hall", error)
                && ok(cudaMalloc(&m.dLifeStats, sizeof(float) * 12), "cudaMalloc life stats", error)
                && ok(cudaMalloc(&m.dResW, sizeof(float) * kReservoirN * kReservoirN), "cudaMalloc reservoir", error)
                && ok(cudaMalloc(&m.dResIn, sizeof(float) * kReservoirN * kReservoirInputs), "cudaMalloc reservoir", error)
                && ok(cudaMalloc(&m.dResOut, sizeof(float) * kReservoirN * kReservoirOutputs), "cudaMalloc reservoir", error)
                && ok(cudaMalloc(&m.dResX, sizeof(float) * 2 * kReservoirN), "cudaMalloc reservoir", error)
                && ok(cudaMalloc(&m.dMeltHist, sizeof(float) * 2 * kMeltN), "cudaMalloc melt", error)
                && ok(cudaMalloc(&m.dMeltAcc, sizeof(float) * 2 * kMeltN), "cudaMalloc melt", error)
                && ok(cudaMalloc(&m.dMeltEnergy, sizeof(float) * 2 * kMeltBands), "cudaMalloc melt", error)
                && ok(cudaMalloc(&m.dMeltOmega, sizeof(float) * kMeltBands), "cudaMalloc melt", error)
                && ok(cudaMalloc(&m.dMeltPhase, sizeof(float) * 2 * kMeltBands), "cudaMalloc melt", error)
                && ok(cudaMalloc(&m.dMeltHeard, sizeof(float) * 2 * kMeltBands), "cudaMalloc melt", error)
                && ok(cudaMalloc(&m.dMeltState, sizeof(int) * 2), "cudaMalloc melt", error)
                && ok(cudaMalloc(&m.dListenHist, sizeof(float) * 2 * kEchoN), "cudaMalloc listen", error)
                && ok(cudaMalloc(&m.dListenPrev, sizeof(float) * kEchoBins), "cudaMalloc listen", error)
                && ok(cudaMalloc(&m.dListenState, sizeof(int) * 2), "cudaMalloc listen", error)
                && ok(cudaMalloc(&m.dSyncTheta, sizeof(float) * partialsTotal), "cudaMalloc sync", error)
                && ok(cudaMalloc(&m.dFlockA, sizeof(float2) * partialsTotal), "cudaMalloc flock", error)
                && ok(cudaMalloc(&m.dFlockB, sizeof(float2) * partialsTotal), "cudaMalloc flock", error)
                && ok(cudaMalloc(&m.dSyncOrder, sizeof(float2) * 3 * partialsTotal), "cudaMalloc sync order", error)
                && ok(cudaMalloc(&m.dSyncX, sizeof(float) * partialsTotal), "cudaMalloc sync places", error)
                && ok(cudaHostAlloc(&m.hLifeStats, sizeof(float) * 12, cudaHostAllocDefault), "cudaHostAlloc life stats", error)
                && ok(cudaHostAlloc(&m.hLife, sizeof(float2) * kLifeCells, cudaHostAllocDefault), "cudaHostAlloc life", error)
                && ok(cudaMalloc(&m.dShapes, sizeof(EnvShape) * kEnvShapes), "cudaMalloc shapes", error)
                && ok(cudaMalloc(&m.dLand, sizeof(float) * landFloats), "cudaMalloc landscape", error)
                && ok(cudaMalloc(&m.dModal, sizeof(float2) * kModalModes), "cudaMalloc modal", error)
                && ok(cudaMalloc(&m.dModalScratch, sizeof(float) * 2 * n * static_cast<size_t>(kModalBlocks)), "cudaMalloc modal scratch", error)
                && ok(cudaMalloc(&m.dGainRing, sizeof(float) * kGainRing * static_cast<size_t>(config.voices)), "cudaMalloc gain ring", error);
    // Cleared on our own stream and waited for: the stream does not synchronise with the default
    // stream, so a plain cudaMemset could still be running when the first block's update kernel
    // writes the same tables (it did: one render in three differed).
    good = good && ok(cudaMemsetAsync(m.dRun, 0, sizeof(uint4) * partialsTotal, m.stream), "cudaMemsetAsync", error)
                && ok(cudaMemsetAsync(m.dAmp, 0, sizeof(float4) * partialsTotal, m.stream), "cudaMemsetAsync", error)
                && ok(cudaMemsetAsync(m.dGains, 0, sizeof(float4) * 3 * partialsTotal, m.stream), "cudaMemsetAsync", error)
                && ok(cudaMemsetAsync(m.dAmpQ, 0, sizeof(float2) * partialsTotal, m.stream), "cudaMemsetAsync", error)
                && ok(cudaMemsetAsync(m.dEchoHist, 0, sizeof(float) * 2 * kEchoN, m.stream), "cudaMemsetAsync", error)
                && ok(cudaMemsetAsync(m.dEchoAcc, 0, sizeof(float) * 2 * kEchoN, m.stream), "cudaMemsetAsync", error)
                && ok(cudaMemsetAsync(m.dEchoRing, 0, sizeof(float2) * 2 * kEchoBins * static_cast<size_t>(kEchoSlots), m.stream), "cudaMemsetAsync", error)
                && ok(cudaMemsetAsync(m.dEchoState, 0, sizeof(int) * 2, m.stream), "cudaMemsetAsync", error)
                && ok(cudaMemsetAsync(m.dSnap, 0, sizeof(float2) * (partialsTotal / kSnapStride), m.stream), "cudaMemsetAsync", error)
                && ok(cudaMemsetAsync(m.dLifeA, 0, sizeof(float2) * kLifeCells, m.stream), "cudaMemsetAsync", error)
                && ok(cudaMemsetAsync(m.dLifeB, 0, sizeof(float2) * kLifeCells, m.stream), "cudaMemsetAsync", error)
                && ok(cudaMemsetAsync(m.dFluidA, 0, sizeof(float) * 9 * kLifeCells, m.stream), "cudaMemsetAsync", error)
                && ok(cudaMemsetAsync(m.dFluidB, 0, sizeof(float) * 9 * kLifeCells, m.stream), "cudaMemsetAsync", error)
                && ok(cudaMemsetAsync(m.dIsles, 0, sizeof(float) * kIsleW * kIsleH, m.stream), "cudaMemsetAsync", error)
                && ok(cudaMemsetAsync(m.dCascadeA, 0, sizeof(float) * partialsTotal, m.stream), "cudaMemsetAsync", error)
                && ok(cudaMemsetAsync(m.dCascadeB, 0, sizeof(float) * partialsTotal, m.stream), "cudaMemsetAsync", error)
                && ok(cudaMemsetAsync(m.dFilterTable, 0, sizeof(float) * kFilterBins * static_cast<size_t>(config.voices), m.stream), "cudaMemsetAsync", error)
                && ok(cudaMemsetAsync(m.dHallHist, 0, sizeof(unsigned long long) * kHallBins, m.stream), "cudaMemsetAsync", error)
                && ok(cudaMemsetAsync(m.dDensity, 0, sizeof(float) * kHallBins, m.stream), "cudaMemsetAsync", error)
                && ok(cudaMemsetAsync(m.dTau, 0, sizeof(float) * partialsTotal, m.stream), "cudaMemsetAsync", error)
                && ok(cudaMemsetAsync(m.dOrbitA, 0, sizeof(float4) * partialsTotal, m.stream), "cudaMemsetAsync", error)
                && ok(cudaMemsetAsync(m.dOrbitB, 0, sizeof(float4) * partialsTotal, m.stream), "cudaMemsetAsync", error)
                && ok(cudaMemsetAsync(m.dLifeStats, 0, sizeof(float) * 12, m.stream), "cudaMemsetAsync", error)
                && ok(cudaMemsetAsync(m.dListenHist, 0, sizeof(float) * 2 * kEchoN, m.stream), "cudaMemsetAsync", error)
                && ok(cudaMemsetAsync(m.dListenPrev, 0, sizeof(float) * kEchoBins, m.stream), "cudaMemsetAsync", error)
                && ok(cudaMemsetAsync(m.dListenState, 0, sizeof(int) * 2, m.stream), "cudaMemsetAsync", error)
                && ok(cudaMemsetAsync(m.dSyncTheta, 0, sizeof(float) * partialsTotal, m.stream), "cudaMemsetAsync", error)
                && ok(cudaMemsetAsync(m.dFlockA, 0, sizeof(float2) * partialsTotal, m.stream), "cudaMemsetAsync", error)
                && ok(cudaMemsetAsync(m.dFlockB, 0, sizeof(float2) * partialsTotal, m.stream), "cudaMemsetAsync", error)
                && ok(cudaMemsetAsync(m.dSyncOrder, 0, sizeof(float2) * 3 * partialsTotal, m.stream), "cudaMemsetAsync", error)
                && ok(cudaMemsetAsync(m.dSyncX, 0, sizeof(float) * partialsTotal, m.stream), "cudaMemsetAsync", error)
                && ok(cudaMemsetAsync(m.dShapes, 0, sizeof(EnvShape) * kEnvShapes, m.stream), "cudaMemsetAsync", error)
                && ok(cudaMemsetAsync(m.dLand, 0, sizeof(float) * landFloats, m.stream), "cudaMemsetAsync", error)
                && ok(cudaMemsetAsync(m.dModal, 0, sizeof(float2) * kModalModes, m.stream), "cudaMemsetAsync", error)
                && ok(cudaMemsetAsync(m.dGainRing, 0, sizeof(float) * kGainRing * static_cast<size_t>(config.voices), m.stream), "cudaMemsetAsync", error)
                && ok(cudaStreamSynchronize(m.stream), "cudaStreamSynchronize", error);
    if (good) {
        // Lenia's ring kernel: a bell around half the radius, normalised (Noctuary's, on a larger torus).
        constexpr int R = kLifeRadius, K = 2 * kLifeRadius + 1;
        float kernel[K * K];
        float sum = 0.0f;
        for (int dy = -R; dy <= R; ++dy)
            for (int dx = -R; dx <= R; ++dx) {
                const float r = std::sqrt(static_cast<float>(dx * dx + dy * dy)) / static_cast<float>(R);
                const float d = (r - 0.5f) / 0.15f;
                const float k = r <= 1.0f ? std::exp(-0.5f * d * d) : 0.0f;
                kernel[(dy + R) * K + (dx + R)] = k;
                sum += k;
            }
        for (float& k : kernel) k /= sum;
        good = ok(cudaMemcpyToSymbol(cLeniaKernel, kernel, sizeof(kernel)), "cudaMemcpyToSymbol lenia", error);
        m.lifeCells.assign(static_cast<size_t>(kLifeCells), 0.0f);
        reservoirInitKernel<<<static_cast<unsigned>((static_cast<size_t>(kReservoirN) * kReservoirN + 255) / 256), 256, 0, m.stream>>>(m.dResW, m.dResIn, m.dResOut, m.dResX);
        good = ok(cudaStreamSynchronize(m.stream), "reservoir init", error);
        std::memset(m.hLifeStats, 0, sizeof(float) * 12);
        m.melt.clear = 1;   // the first block fills the melt's tables
        for (int b = 0; b < kHallBins; ++b) m.hHallUp[b] = 1.0f;   // the hall new: every band at the average
        good = good && ok(cudaMemcpy(m.dHallMemory, m.hHallUp, sizeof(float) * kHallBins, cudaMemcpyHostToDevice), "cudaMemcpy hall", error);
    }
    if (good) {
        std::memset(m.hUpload, 0, m.uploadBytes());
        std::memset(m.hOut, 0, sizeof(float) * kCloudOutChannels * n);
        // The block as one graph: voices up, the levels' ring, three kernels for the voices, two for
        // the modal hall and one for the spectral echo (which hear the voices' block on the card),
        // the block (eight channels) down.
        const auto* hdr = reinterpret_cast<const BlockHeader*>(m.dUpload);
        const auto* voices = reinterpret_cast<const CloudVoice*>(m.dUpload + sizeof(BlockHeader));
        good = ok(cudaStreamBeginCapture(m.stream, cudaStreamCaptureModeThreadLocal), "cudaStreamBeginCapture", error);
        if (good) {
            cudaMemcpyAsync(m.dUpload, m.hUpload, m.uploadBytes(), cudaMemcpyHostToDevice, m.stream);
            constexpr unsigned lifeGrid = (kLifeCells + 255) / 256;
            lifeSeedKernel<<<lifeGrid, 256, 0, m.stream>>>(hdr, m.dLifeA, m.dLifeB);
            lifeStepKernel<<<lifeGrid, 256, 0, m.stream>>>(hdr, 0, m.dLifeA, m.dLifeB, m.dLand);
            lifeStepKernel<<<lifeGrid, 256, 0, m.stream>>>(hdr, 1, m.dLifeA, m.dLifeB, m.dLand);
            fluidSeedKernel<<<lifeGrid, 256, 0, m.stream>>>(hdr, m.dFluidA, m.dFluidB);
            fluidStepKernel<<<lifeGrid, 256, 0, m.stream>>>(hdr, 0, m.dFluidA, m.dFluidB, m.dLifeA, m.dLifeB);
            fluidStepKernel<<<lifeGrid, 256, 0, m.stream>>>(hdr, 1, m.dFluidA, m.dFluidB, m.dLifeA, m.dLifeB);
            islandsKernel<<<1, 1024, 0, m.stream>>>(hdr, m.dIsles, m.dLifeA, m.dLifeB);
            lifeStatsKernel<<<1, 256, 0, m.stream>>>(hdr, m.dLifeA, m.dLifeB, m.dLifeStats);
            gainRingKernel<<<static_cast<unsigned>(config.voices), 256, 0, m.stream>>>(hdr, voices, m.dGainRing);
            cascadeKernel<<<static_cast<unsigned>((partialsTotal + kThreads - 1) / kThreads), kThreads, 0, m.stream>>>(hdr, voices, m.pcap, m.dCascadeA, m.dCascadeB);
            filterTransportKernel<<<static_cast<unsigned>(config.voices), 256, 0, m.stream>>>(hdr, voices, m.dFilterTable);
            syncOrderKernel<<<static_cast<unsigned>((partialsTotal + kThreads - 1) / kThreads), kThreads, 0, m.stream>>>(
                hdr, voices, m.pcap, m.dSyncTheta, m.dSyncX, m.dSyncOrder, partialsTotal);
            updateKernel<<<static_cast<unsigned>((partialsTotal + kThreads - 1) / kThreads), kThreads, 0, m.stream>>>(
                hdr, voices, m.pcap, m.dShapes, m.dLand, m.dRun, m.dAmp, m.dGains, m.dGainRing, m.dAmpQ, m.dSnap, m.dLifeA, m.dLifeB, m.dSyncTheta,
                m.dSyncX, m.dSyncOrder, partialsTotal, m.dFlockA, m.dFlockB, m.dCascadeA, m.dCascadeB, m.dFilterTable,
                m.dHallHist, m.dDensity, m.dTau, m.dOrbitA, m.dOrbitB);
            hallKernel<<<1, kHallBins, 0, m.stream>>>(hdr, m.dHallHist, m.dHallMemory, m.dDensity);
            renderKernel<<<static_cast<unsigned>(m.chunks), kThreads, 0, m.stream>>>(
                hdr, voices, m.pcap, m.dRun, m.dGains, m.dScratch, n);
            reduceKernel<<<static_cast<unsigned>(kCloudChannels * n / kReduceLanes), dim3(kReduceLanes, kReduceGroups), 0, m.stream>>>(
                hdr, m.dScratch, m.chunks, n, m.dOut);
            modalKernel<<<static_cast<unsigned>(kModalBlocks), kModalThreads, 0, m.stream>>>(hdr, m.dOut, m.dModal, m.dModalScratch, n, m.dLand, m.dHallMemory);
            modalReduceKernel<<<static_cast<unsigned>(2 * n / kReduceLanes), dim3(kReduceLanes, kReduceGroups), 0, m.stream>>>(
                hdr, m.dModalScratch, n, m.dOut);
            spectralEchoKernel<<<1, kEchoThreads, 0, m.stream>>>(hdr, m.dOut, m.dEchoHist, m.dEchoAcc, m.dEchoRing, m.dEchoState);
            meltKernel<<<1, kMeltThreads, 0, m.stream>>>(hdr, m.dOut, m.dMeltHist, m.dMeltAcc, m.dMeltEnergy, m.dMeltOmega, m.dMeltPhase,
                                                          m.dMeltHeard, m.dMeltState);
            listenKernel<<<1, kEchoThreads, 0, m.stream>>>(hdr, m.dOut, m.dListenHist, m.dListenPrev, m.dListenState, m.dLifeStats);
            reservoirStepKernel<<<static_cast<unsigned>(kReservoirN * 32 / 256), 256, 0, m.stream>>>(hdr, m.dResW, m.dResIn, m.dResX);
            reservoirReadKernel<<<1, 256, 0, m.stream>>>(hdr, m.dResOut, m.dResX, m.dLifeStats);
            cudaMemcpyAsync(m.hLifeStats, m.dLifeStats, sizeof(float) * 12, cudaMemcpyDeviceToHost, m.stream);
            cudaMemcpyAsync(m.hOut, m.dOut, sizeof(float) * kCloudOutChannels * n, cudaMemcpyDeviceToHost, m.stream);
            good = ok(cudaStreamEndCapture(m.stream, &m.graph), "cudaStreamEndCapture", error)
                && ok(cudaGraphInstantiate(&m.exec, m.graph, 0), "cudaGraphInstantiate", error);
        }
    }
    if (!good) { m.release(); return false; }
    impl_ = std::move(im);
    return true;
}

void CloudEngine::close()
{
    if (!impl_) return;
    impl_->release();
    impl_.reset();
}

const CloudConfig& CloudEngine::config() const { return impl_->cfg; }

bool CloudEngine::setShapes(const EnvShape* shapes, int count)
{
    Impl& m = *impl_;
    count = std::clamp(count, 0, kEnvShapes);
    m.settle();
    if (count > 0 && cudaMemcpyAsync(m.dShapes, shapes, sizeof(EnvShape) * count, cudaMemcpyHostToDevice, m.stream) != cudaSuccess) return false;
    if (cudaStreamSynchronize(m.stream) != cudaSuccess) return false;
    m.shapeCount = count;
    return true;
}

bool CloudEngine::setLandscape(const float* amps, int columns, int rows, float rowSeconds)
{
    Impl& m = *impl_;
    if (columns < 2 || columns > kLandMaxColumns || rows < 1 || rows > kLandMaxRows || !(rowSeconds > 0.0f) || !amps) return false;
    m.settle();
    if (cudaMemcpyAsync(m.dLand, amps, sizeof(float) * static_cast<size_t>(columns) * rows, cudaMemcpyHostToDevice, m.stream) != cudaSuccess) return false;
    if (cudaStreamSynchronize(m.stream) != cudaSuccess) return false;
    m.landColumns = columns;
    m.landRows = rows;
    m.landRowSeconds = rowSeconds;
    return true;
}

CloudVoice& CloudEngine::voice(int i) { return impl_->voices[std::clamp(i, 0, impl_->cfg.voices - 1)]; }

ModalHall& CloudEngine::modal() { return impl_->modal; }

SpectralField& CloudEngine::field() { return impl_->field; }

Binaural& CloudEngine::binaural() { return impl_->binaural; }

SpectralEcho& CloudEngine::spectralEcho() { return impl_->echo; }

void CloudEngine::requestSnapshot() { impl_->snapAsked = true; }

LifeField& CloudEngine::life() { return impl_->life; }

PartialSync& CloudEngine::sync() { return impl_->sync; }

Ripple& CloudEngine::ripple() { return impl_->ripple; }

Reservoir& CloudEngine::reservoir() { return impl_->reservoir; }

Melt& CloudEngine::melt() { return impl_->melt; }

Turbulence& CloudEngine::turbulence() { return impl_->turb; }

bool CloudEngine::hallMemory(float* out) const
{
    if (!impl_ || !impl_->hallDown) return false;
    std::memcpy(out, impl_->hHallMemory, sizeof(float) * kHallBins);
    return true;
}

void CloudEngine::setHallMemory(const float* in)
{
    if (!impl_) return;
    std::memcpy(impl_->hHallUp, in, sizeof(float) * kHallBins);
    impl_->hallUp = true;
}

SpectralComb& CloudEngine::comb() { return impl_->comb; }

void CloudEngine::reservoirOut(float* out) const
{
    for (int k = 0; k < kReservoirOutputs; ++k) out[k] = impl_->hLifeStats[5 + k];
}

void CloudEngine::lifeStats(float& mean, float& meanSq) const
{
    mean = impl_->hLifeStats[0];
    meanSq = impl_->hLifeStats[1];
}

void CloudEngine::heard(float& levelDb, float& centroid, float& flux) const
{
    levelDb = impl_->hLifeStats[2];
    centroid = impl_->hLifeStats[3];
    flux = impl_->hLifeStats[4];
}

const float* CloudEngine::lifePicture() const { return impl_->lifeShown ? impl_->lifeCells.data() : nullptr; }

bool CloudEngine::takeSnapshot(const float*& dots, int& perVoice)
{
    Impl& m = *impl_;
    if (!m.snapReady) return false;
    m.snapReady = false;
    dots = reinterpret_cast<const float*>(m.hSnap);
    perVoice = m.pcap / kSnapStride;
    return true;
}

void CloudEngine::launch()
{
    Impl& m = *impl_;
    const int n = m.cfg.blockSize;
    auto* hdr = reinterpret_cast<BlockHeader*>(m.hUpload);
    hdr->n = n;
    hdr->nPrev = m.nPrev;
    hdr->sampleRate = static_cast<float>(m.cfg.sampleRate);
    hdr->voices = m.cfg.voices;
    hdr->shapeCount = m.shapeCount;
    hdr->landColumns = m.landColumns;
    hdr->landRows = m.landRows;
    hdr->landRowSeconds = m.landRowSeconds;
    hdr->modal = m.modal;
    hdr->modal.tunedCount = std::clamp(m.modal.tunedCount, 0, kModalTuned);
    hdr->field = m.field;
    const bool pitches = m.field.mode == FieldMode::Scale || m.field.mode == FieldMode::Orbit;   // the scale's pitches, not attractors
    hdr->field.count = std::clamp(m.field.count, 0, pitches ? kFieldPoints : kFieldAttractors);
    hdr->field.dispersion = std::clamp(m.field.dispersion, -2.0f, 2.0f);
    hdr->blockIndex = m.blockIndex++;
    hdr->binaural = m.binaural;
    hdr->echo = m.echo;
    m.echo.clear = 0;   // asked for once
    hdr->sync = m.sync;
    hdr->ripple = m.ripple;
    hdr->reservoir = m.reservoir;
    hdr->melt = m.melt;
    hdr->comb = m.comb;
    hdr->turb = m.turb;
    m.melt.clear = 0;   // asked for once
    hdr->life = m.life;
    hdr->life.steps = m.life.mode == LifeMode::Off ? 0 : std::clamp(m.life.steps, 0, 2);
    hdr->lifeParity = m.lifeParity;
    m.lifeParity = (m.lifeParity + hdr->life.steps) & 1;   // where the field is after this block
    m.life.reseed = 0;   // asked for once
    auto* up = reinterpret_cast<CloudVoice*>(m.hUpload + sizeof(BlockHeader));
    const float dt = static_cast<float>(n / m.cfg.sampleRate);
    for (int i = 0; i < m.cfg.voices; ++i) {
        CloudVoice& v = m.voices[i];
        if (v.restart) v.time = 0.0f;
        up[i] = v;
        if (v.active) {
            m.derive(i, up[i]);
            v.harmonics = up[i].harmonics;   // back to the host's copy, for logs and pictures
            v.members = up[i].members;
            v.scale = up[i].scale;
        }
        v.restart = 0;
        if (v.active) v.time += dt;
    }
    if (m.hallUp) {   // a memory loaded (or a new hall): up before the block
        cudaMemcpyAsync(m.dHallMemory, m.hHallUp, sizeof(float) * kHallBins, cudaMemcpyHostToDevice, m.stream);
        m.hallUp = false;
    }
    cudaEventRecord(m.evStart, m.stream);
    cudaGraphLaunch(m.exec, m.stream);
    m.snapInFlight = m.snapAsked;
    if (m.snapAsked) {   // the picture's snapshot behind the block, before the event wait() waits for
        cudaMemcpyAsync(m.hSnap, m.dSnap, sizeof(float2) * (static_cast<size_t>(m.cfg.voices) * m.pcap / kSnapStride), cudaMemcpyDeviceToHost, m.stream);
        cudaMemcpyAsync(m.hHallMemory, m.dHallMemory, sizeof(float) * kHallBins, cudaMemcpyDeviceToHost, m.stream);   // and the hall's memory
        if (m.life.mode != LifeMode::Off)   // and the living field, as it stands after this block's steps
            cudaMemcpyAsync(m.hLife, m.lifeParity ? m.dLifeB : m.dLifeA, sizeof(float2) * kLifeCells, cudaMemcpyDeviceToHost, m.stream);
        m.snapAsked = false;
    }
    cudaEventRecord(m.evStop, m.stream);
    m.nPrev = n;
    m.launched = true;
}

bool CloudEngine::ready() const
{
    return !impl_->launched || cudaEventQuery(impl_->evStop) != cudaErrorNotReady;
}

const float* CloudEngine::wait()
{
    Impl& m = *impl_;
    if (m.launched) {
        if (m.cfg.spinWait) {
            while (cudaEventQuery(m.evStop) == cudaErrorNotReady) _mm_pause();
        } else {
            cudaEventSynchronize(m.evStop);
        }
        cudaEventElapsedTime(&m.gpuMs, m.evStart, m.evStop);
        m.launched = false;
        if (m.snapInFlight) {
            m.snapReady = true;
            m.hallDown = true;
            m.snapInFlight = false;
            m.lifeShown = m.life.mode != LifeMode::Off;
            if (m.lifeShown)
                for (int i = 0; i < kLifeCells; ++i)
                    m.lifeCells[static_cast<size_t>(i)] = m.life.mode == LifeMode::Turing ? std::min(m.hLife[i].y * 3.0f, 1.0f) : m.hLife[i].x;
        }
    }
    return m.hOut;
}

float CloudEngine::lastGpuMs() const { return impl_->gpuMs; }

std::string CloudEngine::deviceName() const
{
    const cudaDeviceProp& p = impl_->props;
    return std::string(p.name) + " (sm_" + std::to_string(p.major) + std::to_string(p.minor) + ")";
}

} // namespace additiv
