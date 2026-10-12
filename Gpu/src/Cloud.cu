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

/**
 * @brief The update kernel: every partial's frequency glide, amplitude ramps and start phase for this block.
 * @param hdr     the block header (device copy)
 * @param voices  the voices (device copy)
 * @param pcap    partials per voice in the tables (a multiple of kChunk)
 * @param shapes  the envelope bank
 * @param land    the landscape
 * @param run     per partial: start phase, phase step, its change per sample, unused (state and render input)
 * @param amp     per partial: the left and right amplitudes the last block ended on
 * @param gains   per partial: left start, left step, right start, right step for the render kernel
 */
__global__ void updateKernel(const BlockHeader* hdr, const CloudVoice* voices, int pcap, const EnvShape* shapes,
                             const float* land, uint4* run, float2* amp, float4* gains)
{
    const int gid = blockIdx.x * blockDim.x + threadIdx.x;
    const int v = gid / pcap;
    const int p = gid - v * pcap;
    if (v >= hdr->voices) return;
    const CloudVoice& cv = voices[v];
    if (!cv.active || p >= cv.partials) {
        // Beyond the voice's partials (or a silent voice): the render kernel skips whole chunks, the
        // tail of the last chunk it covers must be silent.
        if (cv.active) { gains[gid] = make_float4(0.0f, 0.0f, 0.0f, 0.0f); amp[gid] = make_float2(0.0f, 0.0f); }
        return;
    }
    const float fs = hdr->sampleRate;
    const int n = hdr->n;
    const std::uint32_t seed = cv.seed;
    const std::uint32_t up = static_cast<std::uint32_t>(p);
    const float t = cv.time + static_cast<float>(n) / fs;   // every target holds at the block's end

    float target = 0.0f;
    float f = 0.0f;
    float place = cv.stereo * (2.0f * hash01(seed, up, 10) - 1.0f);   // -1 .. 1 across the field
    const int H = cv.harmonics, K = cv.members;
    if (p < H * K) {
        const int h = p / K + 1;
        const int k = p - (h - 1) * K;
        // The member's place in its harmonic's cloud: stratified over the bell curve, so K members
        // always cover it evenly and only jitter within their own slice.
        float z = 0.0f;
        if (K > 1) z = normcdfinvf((static_cast<float>(k) + 0.5f + 0.9f * (hash01(seed, up, 1) - 0.5f)) / static_cast<float>(K));
        const float sigma = 0.5f * cv.width * __powf(static_cast<float>(h), cv.widthSlope);
        const float centre = cv.hz * static_cast<float>(h) * exp2f(z * sigma * (1.0f / 1200.0f));
        const float d = cv.drift * (0.65f * slowSine(cv.driftRate * (0.5f + hash01(seed, up, 2)), t, hash01(seed, up, 3))
                                  + 0.35f * slowSine(cv.driftRate * (1.2f + hash01(seed, up, 4)), t, hash01(seed, up, 5)));
        const float sh = fmaxf(0.0f, 1.0f + cv.shimmer * (0.7f * slowSine(cv.shimmerRate * (0.5f + hash01(seed, up, 6)), t, hash01(seed, up, 7))
                                                         + 0.3f * slowSine(cv.shimmerRate * (1.3f + hash01(seed, up, 8)), t, hash01(seed, up, 9))));

        // The envelope lanes, read against this partial's sources.
        LaneContext c;
        const float pitchPos = log2f(fmaxf(centre, kLandLowHz) * (1.0f / kLandLowHz)) * (1.0f / 9.96578428f);   // log2(1000)
        c.src[0] = 0.0f;
        c.src[1] = fminf(fmaxf(pitchPos, 0.0f), 1.0f);
        c.src[2] = H > 1 ? static_cast<float>(h - 1) / static_cast<float>(H - 1) : 0.0f;
        c.src[3] = fminf(fmaxf((z + 3.0f) * (1.0f / 6.0f), 0.0f), 1.0f);
        c.src[4] = 0.0f;   // Random: laneValue throws its own die per parameter
        c.src[5] = (h & 1) ? 0.0f : 1.0f;
        c.seed = seed;
        c.p = up;
        c.t = t;
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
        float cents = z * sigma + d;
        if (cv.pitch.amount != 0.0f) cents += laneValue(cv.pitch, c, 40, true, depth) * depth;
        if (cv.pan.amount != 0.0f) place = fminf(fmaxf(place + laneValue(cv.pan, c, 50, true, depth) * depth, -1.0f), 1.0f);

        f = cv.hz * static_cast<float>(h) * exp2f(cents * (1.0f / 1200.0f));
        // The top fades out between 0.43 and 0.47 of the sample rate: nothing reaches Nyquist.
        const float fade = fminf(fmaxf((0.47f * fs - f) / (0.04f * fs), 0.0f), 1.0f);
        target = cv.scale * __powf(static_cast<float>(h), -cv.tilt) * sh * fade * env * cv.gain;
    }
    const std::uint32_t incTarget = static_cast<std::uint32_t>(fminf(f / fs, 0.49f) * 4294967296.0f);

    // Phase and glide: the step moves linearly from where the last block left it to this block's
    // target, so the phase is quadratic in the sample and pitch envelopes never step.
    std::uint32_t ph, inc0, dinc = 0;
    float2 a0;
    if (cv.restart) {
        ph = hash32(seed ^ hash32(up + 0x51ED270BU));   // a random phase
        inc0 = incTarget;
        a0 = make_float2(0.0f, 0.0f);
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

    // Equal-power pan: the partial's place in the field.
    float gl, gr;
    __sincosf(0.78539816f * (1.0f + place), &gr, &gl);
    const float tl = target * gl, tr = target * gr;
    const float inv = 1.0f / static_cast<float>(n);
    gains[gid] = make_float4(a0.x, (tl - a0.x) * inv, a0.y, (tr - a0.y) * inv);
    amp[gid] = make_float2(tl, tr);
    run[gid] = make_uint4(ph, inc0, dinc, 0u);
}

/**
 * @brief The render kernel: one chunk of partials summed into its own stereo block of scratch.
 * @param hdr      the block header
 * @param voices   the voices
 * @param pcap     partials per voice in the tables
 * @param run      per partial: start phase, phase step, its change per sample
 * @param gains    per partial: left start, left step, right start, right step
 * @param scratch  per chunk: maxN left samples, then maxN right samples
 * @param maxN     the engine's block size (the scratch stride)
 */
__global__ void __launch_bounds__(kThreads)
renderKernel(const BlockHeader* hdr, const CloudVoice* voices, int pcap,
             const uint4* run, const float4* gains, float* scratch, int maxN)
{
    __shared__ float4 sg[kChunk];         // the chunk's amplitude ramps
    __shared__ uint4  sp[kChunk];         // the chunk's start phases, steps and glides
    __shared__ float  red[2][kThreads];   // the groups' sums when the block is shorter than kThreads

    const int chunk = blockIdx.x;
    const int chunksPerVoice = pcap / kChunk;
    const int v = chunk / chunksPerVoice;
    const int first = (chunk - v * chunksPerVoice) * kChunk;
    const int n = hdr->n;
    float* outL = scratch + static_cast<size_t>(chunk) * 2 * maxN;
    float* outR = outL + maxN;

    const CloudVoice& cv = voices[v];
    if (v >= hdr->voices || !cv.active || first >= cv.partials) {
        for (int s = threadIdx.x; s < n; s += kThreads) { outL[s] = 0.0f; outR[s] = 0.0f; }
        return;
    }
    const int count = min(kChunk, cv.partials - first);
    const int base = v * pcap + first;
    for (int j = threadIdx.x; j < count; j += kThreads) {
        sg[j] = gains[base + j];
        sp[j] = run[base + j];
    }
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
        const float sf = static_cast<float>(s);
        float L = 0.0f, R = 0.0f;
        for (int j = g; j < count; j += groups) {
            const uint4 pi = sp[j];
            const float4 a = sg[j];
            // The phase as a signed fraction of a turn: -pi .. pi, where __sinf is most accurate.
            const std::uint32_t ph = pi.x + su * pi.y + tri * pi.z;
            const float x = __sinf(static_cast<float>(static_cast<std::int32_t>(ph)) * 1.46291808e-9f);
            L = fmaf(fmaf(a.y, sf, a.x), x, L);
            R = fmaf(fmaf(a.w, sf, a.z), x, R);
        }
        if (groups > 1) {
            red[0][threadIdx.x] = L;
            red[1][threadIdx.x] = R;
            __syncthreads();
            if (g == 0)
                for (int q = 1; q < groups; ++q) { L += red[0][lane + q * lanes]; R += red[1][lane + q * lanes]; }
            __syncthreads();
        }
        if (g == 0) { outL[s] = L; outR[s] = R; }
    }
}

/**
 * @brief The reduce kernel: the chunks' sums added sample by sample into interleaved stereo.
 * @param hdr      the block header
 * @param scratch  the render kernel's per-chunk blocks
 * @param chunks   how many chunks
 * @param maxN     the scratch stride
 * @param out      n interleaved stereo frames
 */
__global__ void reduceKernel(const BlockHeader* hdr, const float* scratch, int chunks, int maxN, float* out)
{
    __shared__ float part[kReduceGroups][kReduceLanes + 1];
    const int n = hdr->n;
    const int t = blockIdx.x * kReduceLanes + threadIdx.x;   // channel-major, as the scratch is laid out
    const int c = t / maxN;
    const int s = t - c * maxN;
    const bool live = c < 2 && s < n;
    float acc = 0.0f;
    if (live)
        for (int k = threadIdx.y; k < chunks; k += kReduceGroups) acc += scratch[static_cast<size_t>(k) * 2 * maxN + t];
    part[threadIdx.y][threadIdx.x] = acc;
    __syncthreads();
    if (threadIdx.y == 0 && live) {
        for (int q = 1; q < kReduceGroups; ++q) acc += part[q][threadIdx.x];
        out[s * 2 + c] = acc;
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
    std::vector<float> powerSum;      ///< per voice: sum of h^-2tilt over its harmonics, cached
    std::vector<int>   powerH;        ///< the harmonic count that sum was made for
    std::vector<float> powerTilt;     ///< and the tilt
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
    float2* dAmp = nullptr;           ///< device: per partial the left/right amplitudes the last block ended on
    float4* dGains = nullptr;         ///< device: per partial the amplitude ramps of this block
    EnvShape* dShapes = nullptr;      ///< device: the envelope bank, kEnvShapes
    float* dLand = nullptr;           ///< device: the landscape, room for kLandMaxColumns * kLandMaxRows
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
                         static_cast<void*>(dAmp), static_cast<void*>(dGains), static_cast<void*>(dShapes), static_cast<void*>(dLand) })
            if (p) cudaFree(p);
        exec = nullptr; graph = nullptr; evStart = evStop = nullptr; stream = nullptr;
        hUpload = dUpload = nullptr; hOut = dOut = dScratch = dLand = nullptr;
        dRun = nullptr; dAmp = nullptr; dGains = nullptr; dShapes = nullptr;
    }

    /** @brief Waits for a block in flight, so the tables may be replaced. */
    void settle()
    {
        if (stream) cudaStreamSynchronize(stream);
        if (launched) { cudaEventElapsedTime(&gpuMs, evStart, evStop); launched = false; }
    }

    /**
     * @brief Fills a voice's derived fields: harmonics below the cutoff, members per harmonic, and
     *        the scale that makes the voice's RMS its level.
     * @param i  the voice
     * @param v  its upload copy, completed here
     */
    void derive(int i, CloudVoice& v)
    {
        const float fs = static_cast<float>(cfg.sampleRate);
        const float cutoff = std::min(20000.0f, 0.43f * fs);
        const int partials = std::clamp(v.partials, 1, cfg.partialsPerVoice);
        v.partials = partials;
        const float hz = std::max(v.hz, 1.0f);
        const int H = std::clamp(static_cast<int>(cutoff / hz), 1, partials);
        v.harmonics = H;
        v.members = partials / H;
        if (powerH[i] != H || powerTilt[i] != v.tilt) {
            double sum = 0.0;
            for (int h = 1; h <= H; ++h) sum += std::pow(static_cast<double>(h), -2.0 * v.tilt);
            powerSum[i] = static_cast<float>(sum);
            powerH[i] = H;
            powerTilt[i] = v.tilt;
        }
        // Power of the voice: every member a sine (mean square 1/2), the shimmer's two sines add
        // their own mean square (0.7^2 + 0.3^2) / 2 times its depth squared. The envelopes are
        // the sound's own motion and are not compensated.
        const float shimmerMs = 1.0f + 0.29f * v.shimmer * v.shimmer;
        v.scale = v.level / std::sqrt(0.5f * static_cast<float>(v.members) * powerSum[i] * shimmerMs);
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
    im->powerSum.assign(config.voices, 0.0f);
    im->powerH.assign(config.voices, -1);
    im->powerTilt.assign(config.voices, 0.0f);

    int count = 0;
    if (!ok(cudaGetDeviceCount(&count), "cudaGetDeviceCount", error)) return false;
    if (count < 1) { if (error) *error = "no CUDA device"; return false; }
    // The flags only take before the context exists; a second engine in the process keeps the first's.
    cudaSetDeviceFlags(config.spinWait ? cudaDeviceScheduleSpin : cudaDeviceScheduleBlockingSync);
    cudaGetLastError();
    Impl& m = *im;
    bool good = ok(cudaSetDevice(m.device), "cudaSetDevice", error)
             && ok(cudaGetDeviceProperties(&m.props, m.device), "cudaGetDeviceProperties", error);
    int lo = 0, hi = 0;
    good = good && ok(cudaDeviceGetStreamPriorityRange(&lo, &hi), "cudaDeviceGetStreamPriorityRange", error)
                && ok(cudaStreamCreateWithPriority(&m.stream, cudaStreamNonBlocking, hi), "cudaStreamCreateWithPriority", error)
                && ok(cudaEventCreate(&m.evStart), "cudaEventCreate", error)
                && ok(cudaEventCreateWithFlags(&m.evStop, config.spinWait ? cudaEventDefault : cudaEventBlockingSync), "cudaEventCreate", error);
    const size_t partialsTotal = static_cast<size_t>(config.voices) * m.pcap;
    const size_t landFloats = static_cast<size_t>(kLandMaxColumns) * kLandMaxRows;
    good = good && ok(cudaHostAlloc(&m.hUpload, m.uploadBytes(), cudaHostAllocDefault), "cudaHostAlloc upload", error)
                && ok(cudaHostAlloc(&m.hOut, sizeof(float) * 2 * n, cudaHostAllocDefault), "cudaHostAlloc out", error)
                && ok(cudaMalloc(&m.dUpload, m.uploadBytes()), "cudaMalloc upload", error)
                && ok(cudaMalloc(&m.dOut, sizeof(float) * 2 * n), "cudaMalloc out", error)
                && ok(cudaMalloc(&m.dScratch, sizeof(float) * 2 * n * static_cast<size_t>(m.chunks)), "cudaMalloc scratch", error)
                && ok(cudaMalloc(&m.dRun, sizeof(uint4) * partialsTotal), "cudaMalloc run", error)
                && ok(cudaMalloc(&m.dAmp, sizeof(float2) * partialsTotal), "cudaMalloc amp", error)
                && ok(cudaMalloc(&m.dGains, sizeof(float4) * partialsTotal), "cudaMalloc gains", error)
                && ok(cudaMalloc(&m.dShapes, sizeof(EnvShape) * kEnvShapes), "cudaMalloc shapes", error)
                && ok(cudaMalloc(&m.dLand, sizeof(float) * landFloats), "cudaMalloc landscape", error);
    // Cleared on our own stream and waited for: the stream does not synchronise with the default
    // stream, so a plain cudaMemset could still be running when the first block's update kernel
    // writes the same tables (it did: one render in three differed).
    good = good && ok(cudaMemsetAsync(m.dRun, 0, sizeof(uint4) * partialsTotal, m.stream), "cudaMemsetAsync", error)
                && ok(cudaMemsetAsync(m.dAmp, 0, sizeof(float2) * partialsTotal, m.stream), "cudaMemsetAsync", error)
                && ok(cudaMemsetAsync(m.dGains, 0, sizeof(float4) * partialsTotal, m.stream), "cudaMemsetAsync", error)
                && ok(cudaMemsetAsync(m.dShapes, 0, sizeof(EnvShape) * kEnvShapes, m.stream), "cudaMemsetAsync", error)
                && ok(cudaMemsetAsync(m.dLand, 0, sizeof(float) * landFloats, m.stream), "cudaMemsetAsync", error)
                && ok(cudaStreamSynchronize(m.stream), "cudaStreamSynchronize", error);
    if (good) {
        std::memset(m.hUpload, 0, m.uploadBytes());
        std::memset(m.hOut, 0, sizeof(float) * 2 * n);
        // The block as one graph: voices up, three kernels, the block down.
        const auto* hdr = reinterpret_cast<const BlockHeader*>(m.dUpload);
        const auto* voices = reinterpret_cast<const CloudVoice*>(m.dUpload + sizeof(BlockHeader));
        good = ok(cudaStreamBeginCapture(m.stream, cudaStreamCaptureModeThreadLocal), "cudaStreamBeginCapture", error);
        if (good) {
            cudaMemcpyAsync(m.dUpload, m.hUpload, m.uploadBytes(), cudaMemcpyHostToDevice, m.stream);
            updateKernel<<<static_cast<unsigned>((partialsTotal + kThreads - 1) / kThreads), kThreads, 0, m.stream>>>(
                hdr, voices, m.pcap, m.dShapes, m.dLand, m.dRun, m.dAmp, m.dGains);
            renderKernel<<<static_cast<unsigned>(m.chunks), kThreads, 0, m.stream>>>(
                hdr, voices, m.pcap, m.dRun, m.dGains, m.dScratch, n);
            reduceKernel<<<static_cast<unsigned>(2 * n / kReduceLanes), dim3(kReduceLanes, kReduceGroups), 0, m.stream>>>(
                hdr, m.dScratch, m.chunks, n, m.dOut);
            cudaMemcpyAsync(m.hOut, m.dOut, sizeof(float) * 2 * n, cudaMemcpyDeviceToHost, m.stream);
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
