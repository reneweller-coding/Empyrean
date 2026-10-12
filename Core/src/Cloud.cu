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
 * @brief The update kernel: every partial's frequency glide, amplitude ramps and start phase for this block.
 * @param hdr     the block header (device copy)
 * @param voices  the voices (device copy)
 * @param pcap    partials per voice in the tables (a multiple of kChunk)
 * @param shapes  the envelope bank
 * @param land    the landscape
 * @param run     per partial: start phase, phase step, its change per sample, the field's offset in cents (as bits)
 * @param amp     per partial: the near left, near right, far left and far right amplitudes the last block ended on
 * @param gains   per partial two entries: near (left start, left step, right start, right step), then far
 * @param ring    per voice kGainRing levels (gainRingKernel), for the dispersion
 */
__global__ void updateKernel(const BlockHeader* hdr, const CloudVoice* voices, int pcap, const EnvShape* shapes,
                             const float* land, uint4* run, float4* amp, float4* gains, const float* ring)
{
    const int gid = blockIdx.x * blockDim.x + threadIdx.x;
    const int v = gid / pcap;
    const int p = gid - v * pcap;
    if (v >= hdr->voices) return;
    const CloudVoice& cv = voices[v];
    if (!cv.active || p >= cv.partials) {
        // Beyond the voice's partials (or a silent voice): the render kernel skips whole chunks, the
        // tail of the last chunk it covers must be silent.
        if (cv.active) {
            gains[2 * gid] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            gains[2 * gid + 1] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            amp[gid] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
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
        if (!isAir) {
            h = p / K + 1;
            const int k = p - (h - 1) * K;
            // The member's place in its harmonic's cloud: stratified over the bell curve, so K members
            // always cover it evenly and only jitter within their own slice.
            if (K > 1) z = normcdfinvf((static_cast<float>(k) + 0.5f + 0.9f * (hash01(seed, up, 1) - 0.5f)) / static_cast<float>(K));
            const float sigma = 0.5f * cv.width * __powf(static_cast<float>(h), cv.widthSlope);
            const float stretch = stretchOf(h, cv.inharmonic);
            centre = cv.hz * static_cast<float>(h) * stretch * exp2f(z * sigma * (1.0f / 1200.0f));
            drift = cv.drift * (0.65f * slowSine(cv.driftRate * (0.5f + hash01(seed, up, 2)), t, hash01(seed, up, 3))
                              + 0.35f * slowSine(cv.driftRate * (1.2f + hash01(seed, up, 4)), t, hash01(seed, up, 5)));
            sh = fmaxf(0.0f, 1.0f + cv.shimmer * (0.7f * slowSine(cv.shimmerRate * (0.5f + hash01(seed, up, 6)), t, hash01(seed, up, 7))
                                                 + 0.3f * slowSine(cv.shimmerRate * (1.3f + hash01(seed, up, 8)), t, hash01(seed, up, 9))));
            amp0 = cv.scale * harmonicShape(h, cv.tilt, cv.oddEven, cv.brightness);
        } else {
            // Air (Noctuary's band-passed noise around a multiple of the fundamental): a wide cloud
            // whose partials flicker in pitch and level at audio-near rates -- dense random sines in
            // a band are heard as its noise.
            const int A = cv.partials - HP;
            const int a = p - HP;
            if (A > 1) z = normcdfinvf((static_cast<float>(a) + 0.5f + 0.9f * (hash01(seed, up, 1) - 0.5f)) / static_cast<float>(A));
            centre = cv.hz * cv.airCenter * exp2f(z * 0.5f * cv.airWidth * (1.0f / 1200.0f));
            const float j = cv.airJitter;
            drift = 40.0f * (0.6f * slowSine(j * (0.5f + hash01(seed, up, 2)), t, hash01(seed, up, 3))
                           + 0.4f * slowSine(j * (1.3f + hash01(seed, up, 4)), t, hash01(seed, up, 5)));
            sh = fmaxf(0.0f, 1.0f + 0.8f * (0.7f * slowSine(j * (0.7f + hash01(seed, up, 6)), t, hash01(seed, up, 7))
                                          + 0.3f * slowSine(j * (1.6f + hash01(seed, up, 8)), t, hash01(seed, up, 9))));
            amp0 = cv.airScale;
        }

        // The envelope lanes, read against this partial's sources.
        LaneContext c;
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
        float cents = drift;
        if (cv.pitch.amount != 0.0f) cents += laneValue(cv.pitch, c, 40, true, depth) * depth;
        if (cv.pan.amount != 0.0f) place = fminf(fmaxf(place + laneValue(cv.pan, c, 50, true, depth) * depth, -1.0f), 1.0f);

        // The field: the partial's offset moves towards what pulls it, from where it was.
        const float home = centre * exp2f(cents * (1.0f / 1200.0f));
        fieldOff = fieldStep(hdr->field, log2f(fmaxf(home, 1.0f)) + fieldOff * (1.0f / 1200.0f), fieldOff, static_cast<float>(n) / fs,
                             z * (2.0f / 1200.0f));
        f = home * exp2f(fieldOff * (1.0f / 1200.0f));
        // The top fades out between 0.43 and 0.47 of the sample rate: nothing reaches Nyquist.
        const float fade = fminf(fmaxf((0.47f * fs - f) / (0.04f * fs), 0.0f), 1.0f);
        // Distance (Noctuary's planes): 6 dB per unit, and the air's low pass -- a first-order
        // corner falling from 16 kHz by up to five octaves per unit -- read at this partial's
        // frequency, where in a filter it would have been a whole filter. Taken relative to the
        // corner at the ear, so a partial at distance 0 keeps exactly its level (as in Noctuary).
        const float dist = fminf(fmaxf(cv.distance, 0.0f), 1.0f);
        const float x0 = f * (1.0f / 16000.0f);
        const float x = x0 / exp2f(-5.0f * dist * cv.airAbsorb);
        const float absorb = sqrtf((1.0f + x0 * x0) / (1.0f + x * x));
        // The voice's colour and filter, each read at this partial's frequency.
        float colour = filterGain(cv.filterModel, f, cv.cutoff, cv.filterQ, cv.vowel);
        if (!isAir) colour *= presenceGain(f, cv.presenceDb, dist) * lowCutGain(f, cv.lowCut);
        // The voice's level, as late as this partial's clock (the dispersion), from the ring.
        const float level = lateBy > 0.0f ? gainAgo(ring + static_cast<size_t>(v) * kGainRing, hdr->blockIndex, lateBy * fs / static_cast<float>(n))
                                          : cv.gain;
        target = amp0 * sh * fade * env * level * exp2f(-dist) * absorb * colour;
    }
    const std::uint32_t incTarget = static_cast<std::uint32_t>(fminf(f / fs, 0.49f) * 4294967296.0f);

    // Phase and glide: the step moves linearly from where the last block left it to this block's
    // target, so the phase is quadratic in the sample and pitch envelopes never step.
    std::uint32_t ph, inc0, dinc = 0;
    float4 a0;
    int onset = 0;
    if (cv.restart) {
        ph = hash32(seed ^ hash32(up + 0x51ED270BU));   // a random phase
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
    }

    // Equal-power pan: the partial's place in the field; equal-power planes: near cos, far sin.
    float gl, gr, gn, gf;
    __sincosf(0.78539816f * (1.0f + place), &gr, &gl);
    __sincosf(1.57079633f * fminf(fmaxf(cv.distance, 0.0f), 1.0f), &gf, &gn);
    const float4 t4 = make_float4(target * gl * gn, target * gr * gn, target * gl * gf, target * gr * gf);
    const float inv = 1.0f / static_cast<float>(n - onset);
    gains[2 * gid] = make_float4(a0.x, (t4.x - a0.x) * inv, a0.y, (t4.y - a0.y) * inv);
    gains[2 * gid + 1] = make_float4(a0.z, (t4.z - a0.z) * inv, a0.w, (t4.w - a0.w) * inv);
    amp[gid] = t4;
    run[gid] = make_uint4(ph, inc0, dinc, __float_as_uint(fieldOff));
}

/**
 * @brief The render kernel: one chunk of partials summed into its own four-channel block of scratch.
 * @param hdr      the block header
 * @param voices   the voices
 * @param pcap     partials per voice in the tables
 * @param run      per partial: start phase, phase step, its change per sample
 * @param gains    per partial two entries: near (left start, left step, right start, right step), then far
 * @param scratch  per chunk: maxN samples of near left, near right, far left, far right
 * @param maxN     the engine's block size (the scratch stride)
 */
__global__ void __launch_bounds__(kThreads)
renderKernel(const BlockHeader* hdr, const CloudVoice* voices, int pcap,
             const uint4* run, const float4* gains, float* scratch, int maxN)
{
    __shared__ float4 sgn[kChunk];        // the chunk's near amplitude ramps
    __shared__ float4 sgf[kChunk];        // and its far ones
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
    for (int j = threadIdx.x; j < count; j += kThreads) {
        sgn[j] = gains[2 * (base + j)];
        sgf[j] = gains[2 * (base + j) + 1];
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
 */
__global__ void __launch_bounds__(kModalThreads)
modalKernel(const BlockHeader* hdr, const float* in, float2* state, float* scratch, int maxN)
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
    const float T = fmaxf(0.05f, mh.decay * exp2f(breathe - mh.damp * fmaxf(0.0f, log2f(f * (1.0f / 250.0f)))));
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
        /** @brief Equal in every field. @param o the other key @return true when the sum can be reused */
        bool operator==(const ShapeKey& o) const
        {
            return hz == o.hz && harmonics == o.harmonics && tilt == o.tilt && oddEven == o.oddEven && brightness == o.brightness
                && inharmonic == o.inharmonic && presenceDb == o.presenceDb && distance == o.distance && lowCut == o.lowCut;
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
    float4* dGains = nullptr;         ///< device: per partial two entries, the near and the far amplitude ramps of this block
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
        for (void* p : { static_cast<void*>(dUpload), static_cast<void*>(dOut), static_cast<void*>(dScratch), static_cast<void*>(dRun),
                         static_cast<void*>(dAmp), static_cast<void*>(dGains), static_cast<void*>(dShapes), static_cast<void*>(dLand),
                         static_cast<void*>(dModal), static_cast<void*>(dModalScratch), static_cast<void*>(dGainRing) })
            if (p) cudaFree(p);
        exec = nullptr; graph = nullptr; evStart = evStop = nullptr; stream = nullptr;
        hUpload = dUpload = nullptr; hOut = dOut = dScratch = dLand = nullptr;
        dRun = nullptr; dAmp = nullptr; dGains = nullptr; dShapes = nullptr;
        dModal = nullptr; dModalScratch = nullptr; dGainRing = nullptr;
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
        // Harmonics below the cutoff: h f0 sqrt(1 + B h^2) < cutoff, solved for h.
        const double c = cutoff / hz;
        const double B = std::max(0.0f, v.inharmonic);
        const double hmax = B > 0.0 ? std::sqrt((-1.0 + std::sqrt(1.0 + 4.0 * B * c * c)) / (2.0 * B)) : c;
        const int H = std::clamp(static_cast<int>(hmax), 1, HP);
        v.harmonics = H;
        v.members = HP / H;
        v.harmonicPartials = HP;
        const ShapeKey key{ hz, H, v.tilt, v.oddEven, v.brightness, v.inharmonic, v.presenceDb, std::clamp(v.distance, 0.0f, 1.0f), v.lowCut };
        if (!(shapeKey[i] == key)) {
            double sum = 0.0;
            for (int h = 1; h <= H; ++h) {
                const float fh = hz * static_cast<float>(h) * stretchOf(h, v.inharmonic);
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
                && ok(cudaMalloc(&m.dGains, sizeof(float4) * 2 * partialsTotal), "cudaMalloc gains", error)
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
                && ok(cudaMemsetAsync(m.dGains, 0, sizeof(float4) * 2 * partialsTotal, m.stream), "cudaMemsetAsync", error)
                && ok(cudaMemsetAsync(m.dShapes, 0, sizeof(EnvShape) * kEnvShapes, m.stream), "cudaMemsetAsync", error)
                && ok(cudaMemsetAsync(m.dLand, 0, sizeof(float) * landFloats, m.stream), "cudaMemsetAsync", error)
                && ok(cudaMemsetAsync(m.dModal, 0, sizeof(float2) * kModalModes, m.stream), "cudaMemsetAsync", error)
                && ok(cudaMemsetAsync(m.dGainRing, 0, sizeof(float) * kGainRing * static_cast<size_t>(config.voices), m.stream), "cudaMemsetAsync", error)
                && ok(cudaStreamSynchronize(m.stream), "cudaStreamSynchronize", error);
    if (good) {
        std::memset(m.hUpload, 0, m.uploadBytes());
        std::memset(m.hOut, 0, sizeof(float) * kCloudOutChannels * n);
        // The block as one graph: voices up, three kernels for the voices, two for the modal hall
        // (which hears the voices' block on the card), the block (six channels) down.
        const auto* hdr = reinterpret_cast<const BlockHeader*>(m.dUpload);
        const auto* voices = reinterpret_cast<const CloudVoice*>(m.dUpload + sizeof(BlockHeader));
        good = ok(cudaStreamBeginCapture(m.stream, cudaStreamCaptureModeThreadLocal), "cudaStreamBeginCapture", error);
        if (good) {
            cudaMemcpyAsync(m.dUpload, m.hUpload, m.uploadBytes(), cudaMemcpyHostToDevice, m.stream);
            gainRingKernel<<<static_cast<unsigned>(config.voices), 256, 0, m.stream>>>(hdr, voices, m.dGainRing);
            updateKernel<<<static_cast<unsigned>((partialsTotal + kThreads - 1) / kThreads), kThreads, 0, m.stream>>>(
                hdr, voices, m.pcap, m.dShapes, m.dLand, m.dRun, m.dAmp, m.dGains, m.dGainRing);
            renderKernel<<<static_cast<unsigned>(m.chunks), kThreads, 0, m.stream>>>(
                hdr, voices, m.pcap, m.dRun, m.dGains, m.dScratch, n);
            reduceKernel<<<static_cast<unsigned>(kCloudChannels * n / kReduceLanes), dim3(kReduceLanes, kReduceGroups), 0, m.stream>>>(
                hdr, m.dScratch, m.chunks, n, m.dOut);
            modalKernel<<<static_cast<unsigned>(kModalBlocks), kModalThreads, 0, m.stream>>>(hdr, m.dOut, m.dModal, m.dModalScratch, n);
            modalReduceKernel<<<static_cast<unsigned>(2 * n / kReduceLanes), dim3(kReduceLanes, kReduceGroups), 0, m.stream>>>(
                hdr, m.dModalScratch, n, m.dOut);
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
    hdr->field.count = std::clamp(m.field.count, 0, m.field.mode == FieldMode::Scale ? kFieldPoints : kFieldAttractors);
    hdr->field.dispersion = std::clamp(m.field.dispersion, -2.0f, 2.0f);
    hdr->blockIndex = m.blockIndex++;
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
    cudaEventRecord(m.evStart, m.stream);
    cudaGraphLaunch(m.exec, m.stream);
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
