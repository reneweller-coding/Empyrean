/**
 * @file Engine.cpp
 * @brief The instrument (Engine.h): voices, conductors, tuning, the GPU cloud, far reverb, master,
 *        and the worker that renders ahead of the audio callback.
 */
#include "additiv/Engine.h"

#include "additiv/Landscape.h"
#include "additiv/Motion.h"
#include "additiv/Presets.h"
#include "additiv/Shapes.h"
#include "additiv/Spectrum.h"

#define NOMINMAX
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <avrt.h>
#include <xmmintrin.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>

namespace additiv {
namespace {

constexpr std::uint32_t kQueueSize = 4096;   ///< note events in flight from the audio thread (a power of two)
constexpr float kVoiceLevel = 0.1f;          ///< RMS of one voice at full gain, all of its partials together (set so the autoplay sits near -24 dBFS, Noctuary's range)
constexpr double kGlideSeconds = 0.3;        ///< how fast a sounding voice follows a change of tuning
constexpr float kPurityNarrowing = 0.6f;     ///< at Purity 1 the clouds are this much narrower, so pure intervals can lock (PLAN.md)
constexpr float kMetresPerUnit = 20.0f;      ///< one unit of distance, for the doppler (Noctuary: about twenty metres)
constexpr float kSpeedOfSound = 343.0f;      ///< m/s

/**
 * @brief Denormals off while it lives (flush to zero, denormals are zero), the old mode back after.
 *
 * The effects' tails -- reverbs, echoes, the far hall -- decay into numbers the CPU takes a
 * hundred times longer over. JUCE's ScopedNoDenormals covers only the audio callback, and the
 * blocks are rendered on the worker (or, offline, wherever process() is called). Measured before:
 * the slowest block of a second at 2.5 ms of a block's 2.7 (and an underrun at a standalone's start).
 */
struct NoDenormals {
    unsigned int csr = _mm_getcsr();                       ///< the mode it found
    NoDenormals() { _mm_setcsr(csr | 0x8040u); }           ///< FTZ (bit 15) and DAZ (bit 6) on
    ~NoDenormals() { _mm_setcsr(csr); }                    ///< the mode back
    NoDenormals(const NoDenormals&) = delete;              ///< one per scope
    NoDenormals& operator=(const NoDenormals&) = delete;   ///< one per scope
};

/**
 * @brief Raises this process's GPU scheduling class to REALTIME (PLAN.md, Phase 0: the only class
 *        that kept other processes' GPU work from delaying our blocks).
 * @return the NTSTATUS, 0 when it took
 */
long raiseGpuClass()
{
    using SetClass = long (APIENTRY*)(HANDLE, int);
    HMODULE gdi = GetModuleHandleW(L"gdi32.dll");
    if (!gdi) gdi = LoadLibraryW(L"gdi32.dll");
    const auto fn = gdi ? reinterpret_cast<SetClass>(reinterpret_cast<void*>(GetProcAddress(gdi, "D3DKMTSetProcessSchedulingPriorityClass"))) : nullptr;
    return fn ? fn(GetCurrentProcess(), 5) : -1;
}

/** @brief Registers the calling thread with MMCSS as "Pro Audio" while it lives. */
struct ProAudioThread {
    HANDLE task = nullptr;   ///< the registration, null when it failed

    /** @brief Registers the thread at high priority within Pro Audio. */
    ProAudioThread()
    {
        DWORD index = 0;
        task = AvSetMmThreadCharacteristicsW(L"Pro Audio", &index);
        if (task) AvSetMmThreadPriority(task, AVRT_PRIORITY_HIGH);
    }
    ~ProAudioThread() { if (task) AvRevertMmThreadCharacteristics(task); }
    ProAudioThread(const ProAudioThread&) = delete;              ///< one registration per thread
    ProAudioThread& operator=(const ProAudioThread&) = delete;   ///< one registration per thread
};

/** @brief The smallest power of two at least x. @param x a positive number @return the power */
std::int64_t nextPow2(std::int64_t x)
{
    std::int64_t p = 1;
    while (p < x) p <<= 1;
    return p;
}

/** @brief The generated landscapes and their RMS, made once (generating one takes up to a tenth of a second). */
struct FactoryLandscapes {
    std::vector<Landscape> land;   ///< kNumFactoryLandscapes of them
    std::vector<float> rms;        ///< and their RMS

    FactoryLandscapes()
    {
        for (int i = 0; i < kNumFactoryLandscapes; ++i) {
            land.push_back(factoryLandscape(i));
            rms.push_back(landscapeRms(land.back()));
        }
    }
};

/** @brief The one set, built on first use. @return the landscapes */
const FactoryLandscapes& factoryLandscapes()
{
    static const FactoryLandscapes f;
    return f;
}

} // namespace

Engine::Engine()
{
    resetParams();
    for (auto& r : modRows_) r.store(packModRow(ModRow{}), std::memory_order_relaxed);
    for (auto& p : played_) p.store(std::numeric_limits<float>::quiet_NaN(), std::memory_order_relaxed);
}

void Engine::setModRow(int i, const ModRow& row)
{
    if (i < 0 || i >= kModRows) return;
    ModRow r = row;
    if (r.target >= 0 && !isModTarget(static_cast<ParamId>(r.target))) r.target = -1;
    r.depth = std::clamp(r.depth, -1.0f, 1.0f);
    modRows_[static_cast<size_t>(i)].store(packModRow(r), std::memory_order_relaxed);
}

ModRow Engine::modRow(int i) const
{
    if (i < 0 || i >= kModRows) return {};
    return unpackModRow(modRows_[static_cast<size_t>(i)].load(std::memory_order_relaxed));
}

std::string Engine::modText() const
{
    ModRow rows[kModRows];
    for (int i = 0; i < kModRows; ++i) rows[i] = modRow(i);
    return modRowsToText(rows, kModRows);
}

void Engine::setModText(const std::string& text)
{
    ModRow rows[kModRows];
    modRowsFromText(text, rows, kModRows);
    for (int i = 0; i < kModRows; ++i) setModRow(i, rows[i]);
}

Engine::~Engine() { release(); }

void Engine::resetParams()
{
    for (const ParamDesc& d : paramTable()) params_[static_cast<size_t>(d.id)].store(d.def, std::memory_order_relaxed);
}

void Engine::setParam(ParamId id, float value)
{
    const ParamDesc& d = paramDesc(id);
    params_[static_cast<size_t>(id)].store(std::clamp(value, d.min, d.max), std::memory_order_relaxed);
}

std::string Engine::deviceName() const { return prepared_ ? cloud_.deviceName() : std::string(); }

bool Engine::prepare(const EngineConfig& config, std::string* error)
{
    release();
    cfg_ = config;
    const int G = cfg_.gpuBlock;
    if (G < 32 || G > 4096 || (G & (G - 1)) != 0) { if (error) *error = "gpuBlock must be a power of two, 32 .. 4096"; return false; }
    cfg_.maxHostBlock = std::max(cfg_.maxHostBlock, 1);

    CloudConfig cc;
    cc.sampleRate = cfg_.sampleRate;
    cc.blockSize = G;
    cc.voices = cfg_.voices + std::max(0, cfg_.ghosts);
    cc.partialsPerVoice = cfg_.partialsPerVoice;
    cc.spinWait = true;
    if (!cloud_.open(cc, error)) return false;
    const std::vector<EnvShape> bank = factoryShapes();
    cloud_.setShapes(bank.data(), static_cast<int>(bank.size()));
    factoryLandscapes();   // made now, not in the middle of a block
    landChoice_ = -1;
    if (cfg_.realtimeGpuClass) raiseGpuClass();

    voices_.assign(static_cast<size_t>(cfg_.voices), Voice{});
    for (Voice& v : voices_) v.env.setSampleRate(cfg_.sampleRate / G);   // stepped once per GPU block
    rng_.seed(cfg_.seed);
    side_.seed(cfg_.seed ^ 0xA5A5A5A55A5A5A5Aull);
    tide_.init(side_);
    std::fill(std::begin(specKey_), std::end(specKey_), -1.0f);
    scaleChoice_ = -1;
    mods_.reset(cfg_.seed ^ 0x6D6F647363ull);
    for (OwnUnit& o : own_) o.live = false;
    soundValid_ = false;
    morphT_ = 1.0f;
    morphSeen_ = morphAsked_.load(std::memory_order_acquire);
    abWas_ = false;
    mapWas_ = false;
    phraseStart_ = false;
    prevSounding_ = 0;
    readParams();
    refreshSpectrum();
    refreshScale();
    refreshLandscape();
    brain_.reset(rng_.fork(), 48 + rootPc_);
    brain2_.reset(side_.fork(), 36 + rootPc_);   // its own stream: switching it on never moves the first
    stepWas_ = param(ParamId::AutoStep) >= 0.5f;
    far_.prepare(cfg_.sampleRate);
    diffuser_.prepare(cfg_.sampleRate);
    unmask_.prepare(cfg_.sampleRate);
    delay_.prepare(cfg_.sampleRate);
    delay2_.prepare(cfg_.sampleRate);
    early_.prepare(cfg_.sampleRate);
    nearRev_.prepare(cfg_.sampleRate);
    midSide_.prepare(cfg_.sampleRate);
    for (std::vector<float>* b : { &nearL_, &nearR_, &farL_, &farR_, &dryL_, &dryR_, &wetL_, &wetR_, &modalL_, &modalR_, &specL_, &specR_, &meltL_, &meltR_ })
        b->assign(static_cast<size_t>(G), 0.0f);
    ghosts_.assign(static_cast<size_t>(std::max(0, cfg_.ghosts)), Ghost{});
    ghostRng_ = Rng(cfg_.seed ^ 0x47686F7374ull);
    noteRng_ = Rng(cfg_.seed ^ 0x4E6F7465ull);
    randomNote_ = 0.0f;
    roughness_ = 0.0f;
    roughTick_ = 0;
    prevSpecMix_ = prevSpecToFar_ = 0.0f;
    specWasOn_ = false;
    prevMeltMix_ = prevMeltToFar_ = 0.0f;
    meltWasOn_ = false;
    binAngle_ = 0.0f;
    paintTime_ = 0.0f;
    prevDelayMix_ = prevDelayToFar_ = prevDelay2Mix_ = prevDelay2ToFar_ = prevModalLevel_ = 0.0f;
    modalTime_ = 0.0f;
    srcDistance_ = 0.3f;
    {   // The attractors' Lorenz systems: each started near the butterfly from the seed, apart from the others.
        Rng r(cfg_.seed ^ 0x4C6F72656Eull);
        for (auto& s : lorenz_) { s[0] = 8.0 * r.bipolar(); s[1] = 8.0 * r.bipolar(); s[2] = 20.0 + 8.0 * r.bipolar(); }
    }

    queue_.assign(kQueueSize, Event{});
    qHead_.store(0);
    qTail_.store(0);
    pending_.clear();
    pending_.reserve(kQueueSize);

    // The timeline runs two host blocks plus the lookahead ahead of what the callback has taken,
    // rounded up to whole GPU blocks; the ring holds that and room to spare. Two, because a driver
    // may deliver two callbacks back to back (WASAPI did, in the standalone: the ring stood at 448
    // samples for a callback of 480 while the slowest block took 2 ms), and the worker cannot
    // render between them.
    const int ahead = 2 * cfg_.maxHostBlock + static_cast<int>(std::ceil(cfg_.lookaheadMs * 0.001 * cfg_.sampleRate));
    latency_ = (ahead + G - 1) / G * G;
    // The start reserve: as much again rendered before the first callback and not replaced while it
    // is used up, so the first callbacks may come in a burst (a device starting fills its buffer at
    // once: the standalone's ring fell to 224 of 1536 in its first second, with an underrun) while
    // the first GPU blocks are still slow. The latency the host is told stays; an event in the
    // reserve's first stretch comes up to the reserve later (stampFor), never late.
    startReserve_ = realtime_ ? latency_ : 0;   // offline, process() renders what it takes: nothing runs ahead
    const std::int64_t frames = nextPow2(static_cast<std::int64_t>(latency_) + startReserve_ + cfg_.maxHostBlock + 2 * G);
    ring_.assign(static_cast<size_t>(frames) * 2, 0.0f);
    ringMask_ = frames - 1;
    renderPos_.store(0);
    outPos_.store(0);
    underruns_.store(0);
    late_.store(0);
    sounding_.store(0);
    prepared_ = true;
    if (realtime_) {
        startWorker();
        // The first callback must find the ring full: the first blocks are the slowest (the GPU's
        // module loads on its first launch), and the host may call at once.
        for (int waited = 0; waited < 500 && renderPos_.load(std::memory_order_acquire) + G <= latency_ + startReserve_; ++waited) Sleep(1);
    }
    return true;
}

void Engine::release()
{
    stopWorker();
    if (prepared_) cloud_.close();
    prepared_ = false;
}

void Engine::setRealtime(bool realtime)
{
    if (realtime == realtime_) return;
    if (!realtime) stopWorker();
    realtime_ = realtime;
    if (realtime_ && prepared_) startWorker();
}

void Engine::startWorker()
{
    if (worker_.joinable()) return;
    running_.store(true);
    worker_ = std::thread([this] { workerLoop(); });
}

void Engine::stopWorker()
{
    if (!worker_.joinable()) return;
    running_.store(false);
    worker_.join();
}

void Engine::workerLoop()
{
    ProAudioThread pro;
    // A high-resolution timer for the naps: a plain Sleep(1) can last a whole 15.6 ms timer tick,
    // more than the lookahead.
    HANDLE timer = CreateWaitableTimerExW(nullptr, nullptr, CREATE_WAITABLE_TIMER_HIGH_RESOLUTION, TIMER_ALL_ACCESS);
    const int G = cfg_.gpuBlock;
    while (running_.load(std::memory_order_relaxed)) {
        const std::int64_t out = outPos_.load(std::memory_order_acquire);
        std::int64_t r = renderPos_.load(std::memory_order_relaxed);
        if (r < out) { r = out; renderPos_.store(r, std::memory_order_release); }   // after an underrun: catch up
        // Ahead by the latency, and at the start by the reserve not used up yet (out + its rest
        // stays at its end until the callbacks have taken it).
        const std::int64_t reserve = std::max<std::int64_t>(0, startReserve_ - out);
        if (r + G <= out + latency_ + reserve) { renderBlock(); continue; }
        if (timer) {
            LARGE_INTEGER due;
            due.QuadPart = -5000;   // 0.5 ms
            if (SetWaitableTimerEx(timer, &due, 0, nullptr, nullptr, nullptr, 0)) { WaitForSingleObject(timer, 5); continue; }
        }
        Sleep(1);
    }
    if (timer) CloseHandle(timer);
}

void Engine::noteOn(int note, float velocity, int offset)
{
    const std::uint32_t head = qHead_.load(std::memory_order_relaxed);
    if (!prepared_ || head - qTail_.load(std::memory_order_acquire) >= kQueueSize) return;
    queue_[head & (kQueueSize - 1)] = { 0, note, std::clamp(velocity, 0.0f, 1.0f), stampFor(offset) };
    qHead_.store(head + 1, std::memory_order_release);
}

void Engine::noteOff(int note, int offset)
{
    const std::uint32_t head = qHead_.load(std::memory_order_relaxed);
    if (!prepared_ || head - qTail_.load(std::memory_order_acquire) >= kQueueSize) return;
    queue_[head & (kQueueSize - 1)] = { 1, note, 0.0f, stampFor(offset) };
    qHead_.store(head + 1, std::memory_order_release);
}

void Engine::allNotesOff()
{
    const std::uint32_t head = qHead_.load(std::memory_order_relaxed);
    if (!prepared_ || head - qTail_.load(std::memory_order_acquire) >= kQueueSize) return;
    queue_[head & (kQueueSize - 1)] = { 2, 0, 0.0f, stampFor(0) };
    qHead_.store(head + 1, std::memory_order_release);
}

void Engine::process(float* L, float* R, int n)
{
    if (!prepared_) {
        std::fill(L, L + n, 0.0f);
        std::fill(R, R + n, 0.0f);
        return;
    }
    for (int done = 0; done < n;) {
        const int len = std::min(n - done, cfg_.maxHostBlock);
        const std::int64_t out = outPos_.load(std::memory_order_relaxed);
        if (!realtime_)
            while (renderPos_.load(std::memory_order_relaxed) < out + len) renderBlock();
        const std::int64_t have = renderPos_.load(std::memory_order_acquire) - out;
        const int take = static_cast<int>(std::clamp<std::int64_t>(have, 0, len));
        const int fill = static_cast<int>(std::clamp<std::int64_t>(have, 0, 1 << 30));
        int seen = minFill_.load(std::memory_order_relaxed);
        while ((seen < 0 || fill < seen) && !minFill_.compare_exchange_weak(seen, fill, std::memory_order_relaxed)) {}
        int big = maxCallback_.load(std::memory_order_relaxed);
        while (n > big && !maxCallback_.compare_exchange_weak(big, n, std::memory_order_relaxed)) {}
        for (int i = 0; i < take; ++i) {
            const size_t at = static_cast<size_t>((out + i) & ringMask_) * 2;
            L[done + i] = ring_[at];
            R[done + i] = ring_[at + 1];
        }
        if (take < len) {
            std::fill(L + done + take, L + done + len, 0.0f);
            std::fill(R + done + take, R + done + len, 0.0f);
            underruns_.fetch_add(1, std::memory_order_relaxed);
        }
        outPos_.store(out + len, std::memory_order_release);
        done += len;
    }
}

void Engine::drainEvents()
{
    std::uint32_t tail = qTail_.load(std::memory_order_relaxed);
    const std::uint32_t head = qHead_.load(std::memory_order_acquire);
    while (tail != head && pending_.size() < kQueueSize) pending_.push_back(queue_[tail++ & (kQueueSize - 1)]);
    qTail_.store(tail, std::memory_order_release);
}

void Engine::beginMorph(double seconds)
{
    morphSeconds_.store(static_cast<float>(std::max(seconds, 0.05)), std::memory_order_relaxed);
    morphAsked_.fetch_add(1, std::memory_order_release);
}

void Engine::setMorphSlot(int slot, const float* values, const std::string& modText)
{
    ModRow rows[kModRows];
    const int n = modRowsFromText(modText, rows, kModRows);
    {
        const std::lock_guard<std::mutex> lock(slotLock_);
        pendingSlots_[slot & 1].set(values, rows, n);
    }
    slotSet_[static_cast<size_t>(slot & 1)].store(true, std::memory_order_relaxed);
    abVersion_.fetch_add(1, std::memory_order_release);
    slotVersion_.fetch_add(1, std::memory_order_release);
}

void Engine::setMapTarget(const MorphState& s)
{
    {
        const std::lock_guard<std::mutex> lock(slotLock_);
        pendingSlots_[2] = s;
    }
    slotSet_[2].store(true, std::memory_order_relaxed);
    slotVersion_.fetch_add(1, std::memory_order_release);
}

void Engine::loadPreset(int index, double glideSeconds)
{
    if (index < 0 || index >= numPresets()) return;
    float values[kNumParams];
    std::string mod;
    presetValues(index, values, &mod);
    if (glideSeconds > 0.0) beginMorph(glideSeconds);
    for (int i = 0; i < kNumParams; ++i)
        if (!keptByPresets(static_cast<ParamId>(i))) setParam(static_cast<ParamId>(i), values[i]);
    setModText(mod);
}

void Engine::updateSound()
{
    const double dt = cfg_.gpuBlock / cfg_.sampleRate;
    // The slots, when the editor chose one (taken only if the lock is free: the next block tries again).
    bool slotsChanged = false;
    const int version = slotVersion_.load(std::memory_order_acquire);
    if (version != slotSeen_ && slotLock_.try_lock()) {
        const int abVersion = abVersion_.load(std::memory_order_acquire);
        if (abVersion != abSeen_) {
            slots_[0] = pendingSlots_[0];
            slots_[1] = pendingSlots_[1];
            abSeen_ = abVersion;
            slotsChanged = true;
        }
        slots_[2] = pendingSlots_[2];
        slotLock_.unlock();
        slotSeen_ = version;
    }
    // Where the sound goes: the map's blend round its cursor, the A/B blend, or the knobs and the matrix.
    const bool map = param(ParamId::MapActive) >= 0.5f && mapTargetSet();
    const bool ab = !map && param(ParamId::MorphActive) >= 0.5f && morphSlotSet(0) && morphSlotSet(1);
    if (map) {
        target_ = slots_[2];
    } else if (ab) {
        blendMorph(slots_[0], slots_[1], std::clamp(param(ParamId::MorphPos), 0.0f, 1.0f), target_);
    } else {
        float live[kNumParams];
        for (int i = 0; i < kNumParams; ++i) live[i] = param(static_cast<ParamId>(i));
        ModRow rows[kModRows];
        for (int r = 0; r < kModRows; ++r) rows[r] = unpackModRow(modRows_[static_cast<size_t>(r)].load(std::memory_order_relaxed));
        target_.set(live, rows, kModRows);
    }
    for (int i = 0; i < kNumParams; ++i)   // the listener's and the player's own stay as they are set
        if (keptByPresets(static_cast<ParamId>(i)) || static_cast<ParamId>(i) == ParamId::AutoStep)
            target_.values[i] = param(static_cast<ParamId>(i));
    // A glide asked for, or the A/B morph switched or re-chosen: from what played the last block.
    float len = -1.0f;
    const int asked = morphAsked_.load(std::memory_order_acquire);
    if (asked != morphSeen_) { morphSeen_ = asked; len = morphSeconds_.load(std::memory_order_relaxed); }
    if (ab != abWas_ || (ab && slotsChanged)) {
        if (len < 0.0f) len = std::max(param(ParamId::PresetGlide), 1.0f);
        abWas_ = ab;
    }
    if (map != mapWas_) {   // the map taking over or letting go glides too
        if (len < 0.0f) len = std::max(param(ParamId::MapGlide), 1.0f);
        mapWas_ = map;
    }
    if (len > 0.0f && soundValid_) {
        from_ = sound_;
        morphT_ = 0.0f;
        morphLen_ = len;
    }
    if (morphT_ < 1.0f) {
        morphT_ = std::min(1.0f, morphT_ + static_cast<float>(dt) / morphLen_);
        if (morphT_ < 1.0f) blendMorph(from_, target_, morphCurve(morphT_), sound_);
        else sound_ = target_;
    } else if (map) {
        // The map's cursor moves (a hand, a route, automation): the sound follows it, about 95 %
        // of the way in Map Glide.
        const float coef = 1.0f - std::exp(-static_cast<float>(dt) / std::max(param(ParamId::MapGlide) / 3.0f, 0.01f));
        approachMorph(sound_, target_, coef, from_);   // from_ is free outside a glide
        sound_ = from_;
    } else {
        sound_ = target_;
    }
    soundValid_ = true;
    morphProgress_.store(morphT_, std::memory_order_relaxed);
    for (int i = 0; i < kNumParams; ++i) soundLive_[static_cast<size_t>(i)].store(sound_.values[i], std::memory_order_relaxed);
    soundMix_.store(sound_.filterMix, std::memory_order_relaxed);
}

void Engine::runOwnModulators(float dt)
{
    bool read[kMorphSetups] = {};
    float weight[kMorphSetups] = {};
    for (int r = 0; r < sound_.rowCount; ++r) {
        const ModRow& row = sound_.rows[r];
        if ((isOwnSource(row.source) || isOwnSource(row.via)) && row.setup >= 0 && row.setup < sound_.setupCount) {
            read[row.setup] = true;
            weight[row.setup] += std::fabs(row.depth);
        }
    }
    bool wanted[kOwnUnits] = {};
    int unmatched[kMorphSetups];
    int pending = 0, strongest = -1;
    float strongestWeight = -1.0f;
    for (int k = 0; k < sound_.setupCount; ++k) {
        ownOf_[k] = -1;
        if (!read[k]) continue;
        for (int u = 0; u < kOwnUnits; ++u)
            if (own_[static_cast<size_t>(u)].live && !wanted[u] && own_[static_cast<size_t>(u)].setup == sound_.setups[k]) {
                ownOf_[k] = u;
                wanted[u] = true;
                break;
            }
        if (ownOf_[k] < 0) unmatched[pending++] = k;
        else if (weight[k] > strongestWeight) { strongestWeight = weight[k]; strongest = ownOf_[k]; }
    }
    for (int j = 0; j < pending; ++j) {
        const int k = unmatched[j];
        int u = -1;
        for (int v = 0; v < kOwnUnits && u < 0; ++v)   // one no row reads any more runs on with the new settings
            if (own_[static_cast<size_t>(v)].live && !wanted[v]) u = v;
        if (u < 0) {
            for (int v = 0; v < kOwnUnits && u < 0; ++v)
                if (!own_[static_cast<size_t>(v)].live) u = v;
            if (u < 0) continue;   // all running: its rows read the panel's (rare: a blend of blends of many sounds)
            own_[static_cast<size_t>(u)].mods = strongest >= 0 ? own_[static_cast<size_t>(strongest)].mods : mods_.own();
            own_[static_cast<size_t>(u)].live = true;
        }
        own_[static_cast<size_t>(u)].setup = sound_.setups[k];
        ownOf_[k] = u;
        wanted[u] = true;
    }
    for (int u = 0; u < kOwnUnits; ++u) {
        OwnUnit& o = own_[static_cast<size_t>(u)];
        if (!wanted[u]) { o.live = false; continue; }
        o.mods.advance(dt, mods_.envTime(), o.setup);
    }
}

void Engine::readParams()
{
    // What plays before the matrix: the knobs, the A/B morph, or a glide between (Morph.h).
    updateSound();
    // The modulation: the sources advance by a block, then every row adds its depth times its
    // source to its target in the target's normalised domain (Modulation.h).
    {
        ModInputs in;
        for (int i = 0; i < 4; ++i) {
            in.lfoShape[i] = static_cast<int>(std::lround(base(static_cast<ParamId>(static_cast<int>(ParamId::Lfo1Shape) + 2 * i))));
            in.lfoRate[i] = base(static_cast<ParamId>(static_cast<int>(ParamId::Lfo1Rate) + 2 * i));
        }
        for (int e = 0; e < 2; ++e) {
            in.envShape[e] = static_cast<int>(std::lround(base(static_cast<ParamId>(static_cast<int>(ParamId::Env1Shape) + 2 * e))));
            in.envRate[e] = base(static_cast<ParamId>(static_cast<int>(ParamId::Env1Rate) + 2 * e));
        }
        for (int m = 0; m < 4; ++m) in.macro[m] = base(static_cast<ParamId>(static_cast<int>(ParamId::Macro1) + m));
        in.wheel = wheel_.load(std::memory_order_relaxed);
        in.pressure = pressure_.load(std::memory_order_relaxed);
        in.velocity = lastVelocity_;
        in.phraseStart = phraseStart_;
        in.coherence = base(ParamId::Coherence);
        in.coherenceRate = base(ParamId::CoherenceRate);
        in.leniaRate = base(ParamId::LeniaRate);
        in.leniaGrowth = base(ParamId::LeniaGrowth);
        in.chaosPeriod = base(ParamId::ChaosPeriod);
        in.excitation = static_cast<float>(brain_.excitation());
        in.rootAge = static_cast<float>(brain_.rootAgeSeconds());
        in.randomNote = randomNote_;
        {
            // The Beat (Noctuary's): the two lowest distinct pitches, the simplest ratio near their
            // interval, and the difference of the harmonics that would meet if it were just.
            double f1 = 0.0, f2 = 0.0;
            for (const Voice& v : voices_)
                if (v.note >= 0 && v.lastGain > 0.0f && (f1 <= 0.0 || v.hz < f1)) f1 = v.hz;
            if (f1 > 0.0)
                for (const Voice& v : voices_)
                    if (v.note >= 0 && v.lastGain > 0.0f && v.hz / f1 > 1.003 && (f2 <= 0.0 || v.hz < f2)) f2 = v.hz;
            float want = 0.0f;
            if (f1 > 0.0 && f2 > 0.0) {
                static const int kRatios[][2] = { {1,1}, {6,5}, {5,4}, {4,3}, {3,2}, {8,5}, {5,3}, {7,4}, {2,1}, {5,2}, {3,1}, {4,1} };
                const double r = f2 / f1;
                double bestErr = 1e30;
                int bp = 1, bq = 1;
                for (const auto& pq : kRatios) {
                    const double err = std::fabs(std::log(r / (static_cast<double>(pq[0]) / pq[1])));
                    if (err < bestErr) { bestErr = err; bp = pq[0]; bq = pq[1]; }
                }
                want = static_cast<float>(std::fabs(static_cast<double>(bq) * f2 - static_cast<double>(bp) * f1));
                if (!(want == want) || want > 30.0f) want = 30.0f;
            }
            in.beatHz = want;
            // Note and Distance: the loudest voice's, as Noctuary reads its per-voice sources for the matrix.
            const Voice* loud = nullptr;
            for (const Voice& v : voices_)
                if (v.note >= 0 && (loud == nullptr || v.lastGain > loud->lastGain)) loud = &v;
            in.note = loud ? (static_cast<float>(loud->note) - 24.0f) / 84.0f : 0.5f;
            in.distance = loud ? loud->distance : 0.5f;
            // What was heard in the last block (the GPU's), and the chord's roughness: Sethares'
            // dissonance curve (after Plomp and Levelt) over the first eight harmonics of every pair of
            // sounding voices, weighted by their loudness and 1/h, over its own weight -- every eighth
            // block (up to 15,000 exponentials; the source is smoothed over a second anyway).
            if (prepared_) cloud_.heard(in.heardDb, in.heardCentroid, in.heardFlux);
            double rough = 0.0, weight = 0.0;
            const bool roughNow = --roughTick_ <= 0;
            if (roughNow) roughTick_ = 8;
            for (size_t a = 0; roughNow && a < voices_.size(); ++a) {
                const Voice& va = voices_[a];
                if (va.note < 0 || va.lastGain <= 0.0f) continue;
                for (size_t b = a + 1; b < voices_.size(); ++b) {
                    const Voice& vb = voices_[b];
                    if (vb.note < 0 || vb.lastGain <= 0.0f) continue;
                    for (int ha = 1; ha <= 8; ++ha)
                        for (int hb = 1; hb <= 8; ++hb) {
                            const double fa = va.hz * partialRatio(ha, blk_.inharmonicB, blk_.stretch) + blk_.shift;
                            const double fb = vb.hz * partialRatio(hb, blk_.inharmonicB, blk_.stretch) + blk_.shift;
                            const double s = 0.24 / (0.021 * std::min(fa, fb) + 19.0), d = std::fabs(fb - fa) * s;
                            const double w = static_cast<double>(va.lastGain) * vb.lastGain / (ha * hb);
                            rough += w * (std::exp(-3.5 * d) - std::exp(-5.75 * d));
                            weight += w;
                        }
                }
            }
            // Scaled so that a minor second, the roughest common interval, reads about 0.55 and a pure
            // fifth under 0.1 (most of the 64 pairs lie far apart and add nothing to the mean).
            if (roughNow) roughness_ = weight > 1.0e-12 ? static_cast<float>(16.0 * rough / weight) : 0.0f;
            in.roughness = roughness_;
            if (prepared_) cloud_.reservoirOut(in.reservoir);
        }
        phraseStart_ = false;
        mods_.advance(static_cast<float>(cfg_.gpuBlock / cfg_.sampleRate), in);
        runOwnModulators(static_cast<float>(cfg_.gpuBlock / cfg_.sampleRate));
        for (int s = 1; s < kNumModSources; ++s)
            sourceLive_[static_cast<size_t>(s)].store(mods_.value(static_cast<ModSource>(s)), std::memory_order_relaxed);
        // A source as a row reads it: an LFO or envelope from the modulators of the sound the row came with.
        auto source = [this](ModSource s, int setup) {
            if (isOwnSource(s) && setup >= 0 && setup < sound_.setupCount && ownOf_[setup] >= 0)
                return own_[static_cast<size_t>(ownOf_[setup])].mods.value(s);
            return mods_.value(s);
        };
        float offset[kNumParams] = {};
        bool modded[kNumParams] = {};
        for (int r = 0; r < sound_.rowCount; ++r) {   // the matrix where the sound stands (a glide weighs two)
            const ModRow& row = sound_.rows[r];
            if (row.source == ModSource::Off || row.target < 0 || row.target >= kNumParams || row.depth == 0.0f) continue;
            const float amount = row.via == ModSource::Off ? 1.0f : viaAmount(row.via, source(row.via, row.setup));
            offset[row.target] += row.depth * amount * source(row.source, row.setup);
            modded[row.target] = true;
        }
        // A gated amount is heard by its gate, the matrix's push too: nothing where its kind changes.
        for (int g = 0; g < kMorphGates; ++g) offset[static_cast<int>(kGatedAmounts[g])] *= sound_.gate[g];
        for (int i = 0; i < kNumParams; ++i) {
            const ParamDesc& d = paramTable()[static_cast<size_t>(i)];
            const float base = this->base(d.id);
            if (!modded[i]) {
                // Whole numbers glide unrounded while the map's cursor is followed: rounded here.
                cur_[static_cast<size_t>(i)] = d.kind == ParamKind::Int ? std::round(base) : base;
                played_[static_cast<size_t>(i)].store(std::numeric_limits<float>::quiet_NaN(), std::memory_order_relaxed);
                continue;
            }
            const float n = std::clamp(toNorm(d, base) + offset[i], 0.0f, 1.0f);
            float v = fromNorm(d, n);
            if (d.kind == ParamKind::Int) v = std::round(v);
            cur_[static_cast<size_t>(i)] = v;
            played_[static_cast<size_t>(i)].store(n, std::memory_order_relaxed);
        }
    }
    // The family jam's energy: a follower opens and closes with the leader's section, by up to an
    // octave of cutoff and 40 % of brightness (at 0.5, nothing).
    {
        const float e = std::clamp(jamEnergy_.load(std::memory_order_relaxed), 0.0f, 1.0f);
        if (e != 0.5f) {
            cur_[static_cast<size_t>(ParamId::Brightness)] = std::clamp(cur_[static_cast<size_t>(ParamId::Brightness)] * (0.6f + 0.8f * e), 0.0f, 1.0f);
            cur_[static_cast<size_t>(ParamId::Cutoff)] = std::clamp(cur_[static_cast<size_t>(ParamId::Cutoff)] * std::exp2(2.0f * (e - 0.5f)), 40.0f, 18000.0f);
        }
    }
    auto g = [this](ParamId id) { return cur(id); };
    blk_.master = dbToGain(g(ParamId::MasterGain));
    blk_.partials = static_cast<int>(std::lround(g(ParamId::Partials)));
    blk_.tilt = g(ParamId::Tilt);
    blk_.brightness = g(ParamId::Brightness);
    blk_.oddEven = g(ParamId::OddEven);
    blk_.inharmonicB = inharmonicB(g(ParamId::Inharmonic));
    blk_.stretch = g(ParamId::Stretch) / 1200.0f;
    blk_.shift = g(ParamId::Shift);
    blk_.purity = g(ParamId::Purity);
    blk_.width = g(ParamId::CloudWidth) * (1.0f - kPurityNarrowing * blk_.purity);
    blk_.widthSlope = g(ParamId::WidthSlope);
    blk_.drift = g(ParamId::Drift);
    blk_.driftRate = g(ParamId::DriftRate);
    blk_.shimmer = g(ParamId::Shimmer);
    blk_.shimmerRate = g(ParamId::ShimmerRate);
    blk_.stereo = g(ParamId::Stereo);
    blk_.air = g(ParamId::Air);
    blk_.airCenter = g(ParamId::AirCenter);
    blk_.airWidth = g(ParamId::AirWidth);
    blk_.airJitter = g(ParamId::AirJitter);
    blk_.filterModel = static_cast<int>(std::lround(g(ParamId::FilterModel)));
    blk_.cutoff = g(ParamId::Cutoff);
    blk_.resonance = g(ParamId::Resonance);
    blk_.filterEnv = g(ParamId::FilterEnv);
    blk_.keyTrack = g(ParamId::KeyTrack);
    blk_.filterDrift = g(ParamId::FilterDrift);
    blk_.motion = static_cast<int>(std::lround(g(ParamId::Motion)));
    blk_.motionRate = g(ParamId::MotionRate);
    blk_.motionDepth = g(ParamId::MotionDepth);
    blk_.attack = g(ParamId::Attack);
    blk_.decay = g(ParamId::Decay);
    blk_.sustain = g(ParamId::Sustain);
    blk_.release = g(ParamId::Release);
    blk_.refPitch = g(ParamId::RefPitch);
    blk_.tide = g(ParamId::Tide);
    blk_.tidePeriod = g(ParamId::TidePeriod);
    blk_.depth = g(ParamId::Depth);
    blk_.keysDepth = g(ParamId::KeysDepth);
    blk_.farLevel = g(ParamId::FarLevel);
    blk_.presence = g(ParamId::Presence);
    blk_.padLowCut = g(ParamId::PadLowCut);
    blk_.breath = g(ParamId::Breath);
    blk_.breathRate = g(ParamId::BreathRate);
    blk_.doppler = g(ParamId::Doppler);
    blk_.brain2 = g(ParamId::Brain2On) >= 0.5f;
    blk_.brain2Interval = static_cast<int>(std::lround(g(ParamId::Brain2Interval)));
    blk_.brain2Depth = g(ParamId::Brain2Depth);

    bp_.on = g(ParamId::BrainOn) >= 0.5f;
    bp_.mode = static_cast<BrainMode>(std::clamp(static_cast<int>(std::lround(g(ParamId::AutoMode))), 0, kNumBrainModes - 1));
    bp_.density = static_cast<int>(std::lround(g(ParamId::BrainDensity)));
    bp_.rateSeconds = bp_.mode == BrainMode::Chords ? g(ParamId::AutoRate) : g(ParamId::BrainRate);
    bp_.holdMin = g(ParamId::BrainHoldMin);
    bp_.cascade = g(ParamId::BrainCascade);
    bp_.holdMax = g(ParamId::BrainHoldMax);
    bp_.low = static_cast<int>(std::lround(g(ParamId::BrainLow)));
    bp_.high = static_cast<int>(std::lround(g(ParamId::BrainHigh)));
    bp_.consonance = g(ParamId::BrainConsonance);
    bp_.wander = g(ParamId::BrainWander);
    bp_.voiceLead = g(ParamId::AutoLead);
    bp_.chordTension = g(ParamId::AutoTension);
    bp_.rootMove = g(ParamId::AutoRootMove);
    bp_.timbre = g(ParamId::BrainTimbre);
    bp_.spectrum = &brainSpec_;

    // The second conductor: the first one's knobs, its own pace, density and register (its root
    // follows the first one's plus the interval), always in Free -- a slow background.
    bp2_ = bp_;
    bp2_.on = blk_.brain2;
    bp2_.mode = BrainMode::Free;
    bp2_.density = static_cast<int>(std::lround(g(ParamId::Brain2Density)));
    bp2_.rateSeconds = g(ParamId::Brain2Rate);
    bp2_.holdMin = 2.0f * bp_.holdMin;
    bp2_.holdMax = 2.0f * bp_.holdMax;
    bp2_.low = std::clamp(bp_.low + blk_.brain2Interval, 12, 115);
    bp2_.high = std::clamp(bp_.high + blk_.brain2Interval, bp2_.low, 120);
}

void Engine::refreshSpectrum()
{
    const float key[5] = { blk_.tilt, blk_.oddEven, blk_.brightness, blk_.inharmonicB, blk_.stretch };
    if (std::equal(std::begin(key), std::end(key), std::begin(specKey_))) return;
    std::copy(std::begin(key), std::end(key), std::begin(specKey_));
    // The conductors judge timbre against the voice's own harmonic spectrum, both stretches included
    // (so the Timbre scale of a stretched spectrum is the stretched scale that makes it consonant).
    brainSpec_.count = BrainSpectrum::kMax;
    for (int h = 1; h <= BrainSpectrum::kMax; ++h) {
        brainSpec_.ratio[h - 1] = static_cast<double>(partialRatio(h, blk_.inharmonicB, blk_.stretch));
        brainSpec_.amp[h - 1] = harmonicShape(h, blk_.tilt, blk_.oddEven, blk_.brightness);
    }
    if (scaleChoice_ == kTimbreScaleIndex) scaleChoice_ = -1;   // the Timbre scale follows the spectrum
}

void Engine::refreshScale()
{
    const int choice = static_cast<int>(std::lround(base(ParamId::Scale)));
    if (choice != scaleChoice_) {
        FixedScale s;
        // The Timbre scale (Sethares) is the one a timbre asks for: here the cloud's own spectrum.
        const bool built = choice == kTimbreScaleIndex ? makeTimbreScale(brainSpec_, s) : makeBuiltinScale(choice, s);
        if (built) scale_ = s;
        else if (scaleChoice_ < 0) makeBuiltinScale(0, scale_);   // the Scala slot comes with the editor: 12-TET until then
        scaleChoice_ = choice;
    }
    rootPc_ = std::clamp(static_cast<int>(std::lround(base(ParamId::RootNote))), 0, 11);
    snapKeys_ = std::lround(base(ParamId::KeyMap)) == 0;
}

void Engine::setUserLandscape(const Landscape& l)
{
    if (l.columns < 2 || l.columns > kLandMaxColumns || l.rows < 1 || l.amps.size() < static_cast<size_t>(l.columns) * l.rows) return;
    Landscape fit = l;
    if (fit.rows > kLandMaxRows) {   // a long recording: rows averaged in groups, the loop kept
        const int k = (fit.rows + kLandMaxRows - 1) / kLandMaxRows, rows = fit.rows / k;
        std::vector<float> a(static_cast<size_t>(rows) * fit.columns, 0.0f);
        for (int r = 0; r < rows; ++r)
            for (int j = 0; j < k; ++j)
                for (int c = 0; c < fit.columns; ++c) a[static_cast<size_t>(r) * fit.columns + c] += fit.amps[static_cast<size_t>(r * k + j) * fit.columns + c] / k;
        fit.amps.swap(a);
        fit.rowSeconds *= static_cast<float>(k);
        fit.rows = rows;
    }
    const float rms = landscapeRms(fit);
    {
        const std::lock_guard<std::mutex> lock(userLandLock_);
        userLand_ = std::move(fit);
        userLandRms_ = rms;
    }
    userLandVersion_.fetch_add(1, std::memory_order_release);
}

void Engine::refreshLandscape()
{
    int choice = std::clamp(static_cast<int>(std::lround(base(ParamId::Landscape))), 0, kNumLandscapeChoices - 1);
    const int version = userLandVersion_.load(std::memory_order_acquire);
    if (choice == kUserLandscape && version == 0) choice = 0;   // none of the player's yet: Aurora
    if (choice == kUserLandscape) {
        if (landChoice_ == choice && version == userLandSeen_) return;
        if (!userLandLock_.try_lock()) return;   // the editor is handing one over: the next block
        cloud_.setLandscape(userLand_.amps.data(), userLand_.columns, userLand_.rows, userLand_.rowSeconds);
        landRms_ = userLandRms_;
        landLoop_ = static_cast<float>(userLand_.rows) * userLand_.rowSeconds;
        userLandLock_.unlock();
        landChoice_ = choice;
        userLandSeen_ = version;
        return;
    }
    if (choice == landChoice_) return;
    const FactoryLandscapes& f = factoryLandscapes();
    const Landscape& l = f.land[static_cast<size_t>(choice)];
    cloud_.setLandscape(l.amps.data(), l.columns, l.rows, l.rowSeconds);
    landRms_ = f.rms[static_cast<size_t>(choice)];
    landLoop_ = static_cast<float>(l.rows) * l.rowSeconds;
    landChoice_ = choice;
}

double Engine::frequencyOf(int note) const
{
    // Purity blends 12-TET (0) with the scale (1) in the log domain, as in Noctuary.
    const double ref = blk_.refPitch;
    const double pure = scaleFrequency(scale_, note, 60 + rootPc_, ref, snapKeys_);
    const double et = ref * std::pow(2.0, (note - 69) / 12.0);
    const double p = std::clamp(static_cast<double>(blk_.purity), 0.0, 1.0);
    return std::exp(std::log(et) + (std::log(pure) - std::log(et)) * p);
}

float Engine::brainDistance()
{
    // Noctuary's spatial model: the conductor's notes are bimodal, 40 % close, 60 % deep.
    const float u = rng_.uniform();
    return rng_.uniform() < 0.4f ? blk_.depth * 0.15f * u : blk_.depth * (0.55f + 0.45f * u);
}

void Engine::beginOn(int i, int note, float velocity, float distance, int source, int onset)
{
    Voice& v = voices_[static_cast<size_t>(i)];
    lastVelocity_ = velocity;
    v.note = note;
    v.source = source;
    v.velocity = velocity;
    v.place = std::clamp(distance, 0.0f, 1.0f);
    v.distance = v.place;
    v.prevDistance = v.place;
    v.breath.init(side_);
    v.cutoffDrift.init(side_);
    v.env.kill();
    v.env.noteOn();
    v.seed = static_cast<std::uint32_t>(rng_.next()) | 1u;
    v.restart = true;
    v.onset = onset;
    v.released = false;
    v.tailBlocks = 1;
    v.tailArmed = false;
    v.tuneCents = 0.0f;
    v.lastGain = 0.0f;
    randomNote_ = noteRng_.bipolar();
    v.age = 0.0;
    v.stealing = false;
    v.stealBlocks = 0;
    cloud_.voice(i).releaseAt = -1.0f;
}

void Engine::startNote(int note, float velocity, float distance, int source, int onset)
{
    const int count = static_cast<int>(voices_.size());
    for (int i = 0; i < count; ++i)
        if (voices_[static_cast<size_t>(i)].note < 0 && !voices_[static_cast<size_t>(i)].stealing) { beginOn(i, note, velocity, distance, source, onset); return; }
    // Every voice sounds: take the quietest released one, else the oldest. It fades to silence
    // over one block (a restart from full level would click) and the new note begins after it.
    int best = -1;
    float bestLevel = 2.0f;
    for (int i = 0; i < count; ++i) {
        const Voice& v = voices_[static_cast<size_t>(i)];
        if (!v.stealing && v.released && v.env.level() < bestLevel) { bestLevel = v.env.level(); best = i; }
    }
    if (best < 0) {
        double oldest = -1.0;
        for (int i = 0; i < count; ++i) {
            const Voice& v = voices_[static_cast<size_t>(i)];
            if (!v.stealing && v.age > oldest) { oldest = v.age; best = i; }
        }
    }
    if (best < 0) return;
    Voice& v = voices_[static_cast<size_t>(best)];
    v.stealing = true;
    v.stealBlocks = 1;
    v.next = { 0, note, velocity, 0 };
    v.nextSource = source;
    v.nextDistance = distance;
}

void Engine::stopNote(int note, int source)
{
    for (size_t i = 0; i < voices_.size(); ++i) {
        Voice& v = voices_[i];
        if (v.stealing && v.next.type == 0 && v.next.note == note && v.nextSource == source) { v.next.type = -1; return; }
        if (v.note == note && v.source == source && !v.released && !v.stealing) {
            v.released = true;
            v.env.noteOff();
            CloudVoice& cv = cloud_.voice(static_cast<int>(i));
            cv.releaseAt = cv.time;   // the motion's release tails start here
            if (cur(ParamId::MemoryLevel) > 0.0f && !ghosts_.empty()) captureGhost(static_cast<int>(i));
            return;
        }
    }
}

void Engine::lifeSnapshot(std::vector<float>& cells)
{
    const std::lock_guard<std::mutex> lock(snapLock_);
    cells = snapLife_;
}

bool Engine::partialSnapshot(std::vector<float>& dots, int& perVoice)
{
    const std::lock_guard<std::mutex> lock(snapLock_);
    if (snapSeq_ == snapSeen_ || snap_.empty()) return false;
    snapSeen_ = snapSeq_;
    dots = snap_;
    perVoice = snapPer_;
    return true;
}

void Engine::stepRipple(float dt)
{
    Ripple& R = cloud_.ripple();
    R.depth = cur(ParamId::RippleDepth);
    const int strikes = rippleStrikeCount_;
    rippleStrikeCount_ = 0;
    if (R.depth <= 0.0f) {   // at rest: the string is let go, so it starts quiet when it is turned up
        std::fill(std::begin(rippleNow_), std::end(rippleNow_), 0.0f);
        std::fill(std::begin(ripplePrev_), std::end(ripplePrev_), 0.0f);
        std::fill(std::begin(R.u), std::end(R.u), 0.0f);
        R.makeup = 1.0f;
        return;
    }
    constexpr float kCellsPerOctave = static_cast<float>(kRippleCells - 1) / 9.96578428f;
    // Strikes: every note that started this block, at its first eight harmonics, the stronger the
    // lower the harmonic -- a pebble into the spectrum where the note lands. The shape is a Ricker
    // wavelet (a "Mexican hat", zero in sum) about a quarter octave wide, so a passing wave makes a
    // partial louder and then softer rather than louder on the whole; and it goes into both the now
    // and the step before, a displacement at rest, which parts into two halves running up and down
    // (a struck velocity would leave a standing plateau behind in one dimension: measured, 3 dB
    // louder where the notes are and no motion).
    for (int s = 0; s < strikes; ++s) {
        for (int h = 1; h <= 8; ++h) {
            const float pos = std::log2(std::max(rippleStrikes_[s] * h, 20.0f) / 20.0f) * kCellsPerOctave;
            for (int c = std::max(0, static_cast<int>(pos) - 24); c <= std::min(kRippleCells - 1, static_cast<int>(pos) + 24); ++c) {
                const float d = (static_cast<float>(c) - pos) / 5.0f;
                // 6/h: a passing half-wave of the fundamental's strike takes the loudness from near silence
                // to twice (tanh saturates); at 2/h it moved a harmonic by a dB or two, too tame to hear.
                const float bump = (6.0f / static_cast<float>(h)) * (1.0f - d * d) * std::exp(-0.5f * d * d);
                rippleNow_[c] += bump;
                ripplePrev_[c] += bump;
            }
        }
    }
    // The damped wave, u_tt = c^2 u_xx - g u_t, by leapfrog; the ends reflect.
    const float c = std::max(cur(ParamId::RippleSpeed), 0.01f) * kCellsPerOctave;   // cells a second
    const float g = 2.0f / std::max(cur(ParamId::RippleDecay), 0.1f);
    const int steps = std::max(1, static_cast<int>(std::ceil(c * dt / 0.9f)));
    const float h = dt / static_cast<float>(steps), r2 = (c * h) * (c * h), damp = g * h * 0.5f;
    float next[kRippleCells];
    for (int s = 0; s < steps; ++s) {
        for (int i = 0; i < kRippleCells; ++i) {
            const float l = rippleNow_[i > 0 ? i - 1 : 1], rr = rippleNow_[i < kRippleCells - 1 ? i + 1 : kRippleCells - 2];
            next[i] = (2.0f * rippleNow_[i] - (1.0f - damp) * ripplePrev_[i] + r2 * (l - 2.0f * rippleNow_[i] + rr)) / (1.0f + damp);
        }
        std::copy(std::begin(rippleNow_), std::end(rippleNow_), std::begin(ripplePrev_));
        std::copy(std::begin(next), std::end(next), std::begin(rippleNow_));
    }
    double mean2 = 0.0;
    for (int i = 0; i < kRippleCells; ++i) {
        if (!std::isfinite(rippleNow_[i])) rippleNow_[i] = 0.0f;
        R.u[i] = rippleNow_[i];
        const double t = std::tanh(static_cast<double>(rippleNow_[i]));
        mean2 += t * t;
    }
    R.makeup = static_cast<float>(1.0 / std::sqrt(1.0 + static_cast<double>(R.depth) * R.depth * mean2 / kRippleCells));
}

void Engine::captureGhost(int i)
{
    size_t slot = 0;
    for (size_t g = 0; g < ghosts_.size(); ++g) {
        if (!ghosts_[g].on) { slot = g; break; }
        if (ghosts_[g].env < ghosts_[slot].env) slot = g;
    }
    Ghost& g = ghosts_[slot];
    g.base = cloud_.voice(i);
    g.base.releaseAt = -1.0f;   // a ghost is held: its motion loops on
    g.on = true;
    g.restart = true;
    g.seed = static_cast<std::uint32_t>(ghostRng_.next()) | 1u;
    g.age = 0.0f;
    g.life = std::max(1.0f, cur(ParamId::MemoryTime));
    g.bloom = std::max(0.05f, cur(ParamId::MemoryBloom));
    g.env = 0.0f;
}

void Engine::renderBlock()
{
    const NoDenormals noDenormals;
    LARGE_INTEGER qpcStart;
    QueryPerformanceCounter(&qpcStart);
    const int G = cfg_.gpuBlock;
    const double dt = G / cfg_.sampleRate;
    const float dtf = static_cast<float>(dt);
    const std::int64_t pos = renderPos_.load(std::memory_order_relaxed);
    readParams();
    refreshSpectrum();
    refreshScale();
    refreshLandscape();

    // 1. The played notes that are due, in the order they came, each on its own sample.
    drainEvents();
    for (size_t i = 0; i < pending_.size();) {
        const Event e = pending_[i];
        if (e.stamp >= pos + G) { ++i; continue; }
        pending_.erase(pending_.begin() + static_cast<std::ptrdiff_t>(i));
        if (e.stamp < pos) late_.fetch_add(1, std::memory_order_relaxed);
        const int onset = static_cast<int>(std::clamp<std::int64_t>(e.stamp - pos, 0, G - 1));
        if (e.type == 0) startNote(e.note, e.velocity, blk_.keysDepth, 0, onset);
        else if (e.type == 1) stopNote(e.note, 0);
        else
            for (const Voice& v : voices_)
                if (v.note >= 0 && v.source == 0 && !v.released) stopNote(v.note, 0);
    }

    // 2. The conductors. Step fires on the rising edge of its switch, or from autoplayStep().
    const bool step = cur(ParamId::AutoStep) >= 0.5f;
    if ((step && !stepWas_) | stepAsked_.exchange(false, std::memory_order_relaxed)) brain_.requestStep();
    stepWas_ = step;
    if (fillAsked_.exchange(false, std::memory_order_relaxed)) {
        if (bp_.on) brain_.requestFill();
        if (blk_.brain2) brain2_.requestFill();
    }
    auto freqOf = [this](int note) { return frequencyOf(note); };
    brain_.update(dt, bp_, -1, freqOf, [this](const BrainEvent& e) {
        if (e.type == BrainEvent::Type::NoteOn) startNote(e.note, e.velocity, brainDistance(), 1, 0);
        else stopNote(e.note, 1);
    });
    // The family jam: a follower's conductor keeps its root on the leader's pitch class, the nearest
    // note of it (Noctuary's way: its tuning stays its own).
    if (const int jr = jamRoot_.load(std::memory_order_relaxed); jr >= 0 && ((brain_.root() % 12) + 12) % 12 != jr) {
        const int r = brain_.root();
        int best = r, bestDist = 99;
        for (int d = -6; d <= 6; ++d)
            if ((((r + d) % 12) + 12) % 12 == jr && std::abs(d) < bestDist) { bestDist = std::abs(d); best = r + d; }
        brain_.setRoot(std::clamp(best, 0, 127));
    }
    brain2_.setRoot(std::clamp(brain_.root() + blk_.brain2Interval, 0, 127));
    brain2_.update(dt, bp2_, -1, freqOf, [this](const BrainEvent& e) {
        if (e.type == BrainEvent::Type::NoteOn) startNote(e.note, e.velocity, blk_.brain2Depth * (0.85f + 0.15f * rng_.uniform()), 2, 0);
        else stopNote(e.note, 2);
    });
    brainRoot_.store(brain_.root(), std::memory_order_relaxed);

    // 3. The voices: envelope, place, frequency, colour, filter, air, motion.
    // While the sound glides, the tuning follows slowly: a new scale is reached by the partials
    // sliding into it over seconds (a tenth of the glide, at most six).
    const double glideSeconds = morphT_ < 1.0f ? std::clamp(0.1 * morphLen_, kGlideSeconds, 6.0) : kGlideSeconds;
    const double follow = 1.0 - std::exp(-dt / glideSeconds);
    const double tideMul = std::pow(2.0, blk_.tide * tide_.update(dtf, 1.0f / (blk_.tidePeriod * 60.0f), side_) / 1200.0);
    // Self-tuning: where a harmonic of one voice nearly meets a harmonic of another (within Reach),
    // the two pull together -- weighted by their loudness, by a bell over the distance and by how low
    // the harmonics are (the beats of low ones are what is heard). Every voice moves half the way
    // its weighted mismatch asks for, at the speed Self-Tune sets (the other moves the other half),
    // and the anchor draws it back to its pitch in the scale. A chord in equal temperament settles
    // towards just intervals and gives way again at the next change; a note starts on the scale.
    {
        const float grav = cur(ParamId::TuneGravity), reach = std::max(1.0f, cur(ParamId::TuneReach)), anchor = cur(ParamId::TuneAnchor);
        const int nv = static_cast<int>(voices_.size());
        if (grav > 0.0f) {
            float kLogH[9] = { 0.0f };   // log2 of where harmonics 1 .. 8 lie (both stretches)
            for (int h = 1; h <= 8; ++h) kLogH[h] = std::log2(partialRatio(h, blk_.inharmonicB, blk_.stretch));
            std::array<float, 64> lb{}, moved{};
            for (int a = 0; a < nv && a < 64; ++a) {
                const Voice& v = voices_[static_cast<size_t>(a)];
                lb[static_cast<size_t>(a)] = v.note >= 0 ? static_cast<float>(std::log2(frequencyOf(v.note) * tideMul)) + v.tuneCents / 1200.0f : 0.0f;
            }
            const float rate = 1.0f - std::exp(-2.0f * grav * dtf);
            for (int a = 0; a < nv && a < 64; ++a) {
                const Voice& va = voices_[static_cast<size_t>(a)];
                if (va.note < 0 || va.stealing || va.lastGain <= 0.0f) continue;
                double force = 0.0, wsum = 0.0;
                for (int b = 0; b < nv && b < 64; ++b) {
                    const Voice& vb = voices_[static_cast<size_t>(b)];
                    if (b == a || vb.note < 0 || vb.stealing || vb.lastGain <= 0.0f) continue;
                    for (int ha = 1; ha <= 8; ++ha)
                        for (int hb = 1; hb <= 8; ++hb) {
                            const float d = 1200.0f * (lb[static_cast<size_t>(b)] + kLogH[hb] - lb[static_cast<size_t>(a)] - kLogH[ha]);
                            if (std::fabs(d) >= reach) continue;
                            const double w = static_cast<double>(va.lastGain) * vb.lastGain * std::exp(-(d / reach) * (d / reach)) / (ha * hb);
                            force += w * d;
                            wsum += w;
                        }
                }
                if (wsum > 1.0e-12) moved[static_cast<size_t>(a)] = static_cast<float>(0.5 * force / wsum) * rate;
            }
            const float back = std::exp(-2.0f * anchor * dtf);
            for (int a = 0; a < nv && a < 64; ++a) {
                Voice& v = voices_[static_cast<size_t>(a)];
                v.tuneCents = std::clamp((v.tuneCents + moved[static_cast<size_t>(a)]) * back, -50.0f, 50.0f);
            }
        } else {
            for (Voice& v : voices_) v.tuneCents *= std::exp(-2.0f * dtf);   // switched off: back to the scale
        }
    }
    const MotionKind motion = static_cast<MotionKind>(std::clamp(blk_.motion, 0, kNumMotions - 1));
    // A voice that reads the landscape is as much quieter as the landscape is dark: made up here
    // (at most 18 dB), so a dim landscape changes the motion, not the level.
    float landMakeup = 1.0f;
    if (motionUsesLandscape(motion)) landMakeup = 1.0f / std::max(0.125f, 1.0f - blk_.motionDepth + blk_.motionDepth * landRms_);
    int sounding = 0;
    int by[3] = { 0, 0, 0 };
    float srcW = 0.0f, srcD = 0.0f;   // the near room's source: the voices' distance, weighted by loudness
    for (int i = 0; i < static_cast<int>(voices_.size()); ++i) {
        Voice& v = voices_[static_cast<size_t>(i)];
        CloudVoice& cv = cloud_.voice(i);
        if (v.stealing) {
            if (v.stealBlocks > 0) { --v.stealBlocks; cv.gain = 0.0f; v.env.kill(); ++sounding; continue; }
            v.stealing = false;
            if (v.next.type == 0) beginOn(i, v.next.note, v.next.velocity, v.nextDistance, v.nextSource, 0);
            else v.note = -1;
        }
        if (v.note < 0) { cv.active = 0; continue; }
        v.env.setTimes(blk_.attack, blk_.decay, blk_.sustain, blk_.release);
        const float gain = v.env.process();
        v.lastGain = gain;
        if (!v.env.isActive()) {
            // With dispersion the late partials sound on after the envelope: the voice stays until
            // the latest of them has heard the end of it.
            if (!v.tailArmed) {
                v.tailArmed = true;
                v.tailBlocks += static_cast<int>(std::ceil(std::fabs(cur(ParamId::Dispersion)) * cfg_.sampleRate / G));
            }
            if (v.tailBlocks > 0) { --v.tailBlocks; cv.gain = 0.0f; ++sounding; continue; }
            v.note = -1;
            cv.active = 0;
            continue;
        }
        // Breath: the voice's plane wanders (Noctuary's "the room breathes"); Doppler: coming closer
        // raises it, receding lowers it.
        v.distance = std::clamp(v.place + 0.35f * blk_.breath * v.breath.update(dtf, blk_.breathRate, side_), 0.0f, 1.0f);
        const float velocity = (v.distance - v.prevDistance) / dtf * kMetresPerUnit;
        const double doppler = v.restart ? 1.0 : 1.0 - static_cast<double>(blk_.doppler * velocity / kSpeedOfSound);
        v.prevDistance = v.distance;
        const double target = frequencyOf(v.note) * tideMul * std::exp2(static_cast<double>(v.tuneCents) / 1200.0);
        v.hz = v.restart ? target : v.hz * std::pow(target / v.hz, follow);
        // The filter's cutoff: key tracking, the envelope, and its own drift of up to an octave.
        const float oct = blk_.keyTrack * static_cast<float>(v.note - 60) / 12.0f + blk_.filterEnv * gain
                        + blk_.filterDrift * v.cutoffDrift.update(dtf, 0.05f, side_);
        const float fc = std::clamp(blk_.cutoff * std::exp2(oct), 20.0f, 20000.0f);
        const FilterModel model = static_cast<FilterModel>(std::clamp(blk_.filterModel, 0, kNumFilterModels - 1));
        float q = 0.707f + blk_.resonance * blk_.resonance * 11.3f;   // up to Q 12
        if (model == FilterModel::Peak) q = 1.0f + 3.0f * blk_.resonance;   // the bell's gain, up to +12 dB
        if (model == FilterModel::Formant) q = 4.0f + 16.0f * blk_.resonance;

        float level = kVoiceLevel * (v.source != 0 ? 0.6f + 0.4f * v.velocity : 0.4f + 0.6f * v.velocity);
        cv.hz = static_cast<float>(v.hz * doppler);
        cv.level = level * landMakeup;
        cv.gain = gain;
        cv.tilt = blk_.tilt;
        cv.brightness = blk_.brightness;
        cv.oddEven = blk_.oddEven;
        cv.inharmonic = blk_.inharmonicB;
        cv.stretch = blk_.stretch;
        cv.shift = blk_.shift;
        cv.presenceDb = blk_.presence;
        cv.lowCut = blk_.padLowCut;
        cv.filterModel = static_cast<int>(model);
        cv.filterMix = sound_.filterMix;
        cv.cutoff = fc;
        cv.filterQ = q;
        cv.vowel = 4.0f * std::clamp(std::log(fc / 40.0f) / std::log(18000.0f / 40.0f), 0.0f, 1.0f);
        cv.air = blk_.air;
        cv.airCenter = blk_.airCenter;
        cv.airWidth = blk_.airWidth;
        cv.airJitter = blk_.airJitter;
        cv.width = blk_.width;
        cv.widthSlope = blk_.widthSlope;
        cv.drift = blk_.drift;
        cv.driftRate = blk_.driftRate;
        cv.shimmer = blk_.shimmer;
        cv.shimmerRate = blk_.shimmerRate;
        cv.stereo = blk_.stereo;
        cv.partials = blk_.partials;
        cv.distance = v.distance;
        srcW += gain * level;
        srcD += gain * level * v.distance;
        cv.airAbsorb = 1.0f;
        cv.seed = v.seed;
        cv.active = 1;
        cv.restart = v.restart ? 1 : 0;
        if (v.restart && rippleStrikeCount_ < 64) rippleStrikes_[rippleStrikeCount_++] = static_cast<float>(v.hz);
        cv.onset = v.restart ? v.onset : 0;
        applyMotion(motion, cv, blk_.motionRate, blk_.motionDepth);
        v.restart = false;
        v.onset = 0;
        v.age += dt;
        ++sounding;
        ++by[std::clamp(v.source, 0, 2)];
    }
    sounding_.store(sounding, std::memory_order_relaxed);
    if (srcW > 1.0e-6f) srcDistance_ = srcD / srcW;
    if (prevSounding_ == 0 && sounding > 0) phraseStart_ = true;   // a note into silence: the envelopes' phrase begins
    prevSounding_ = sounding;
    for (int s = 0; s < 3; ++s) soundingBy_[s].store(by[s], std::memory_order_relaxed);
    // The memory's ghosts: each a released voice sounding on from where it was let go, rising over
    // Bloom, fading by 60 dB over Memory Time, and as it fades widening, drifting, darkening and
    // receding (Blur, Sink). New phases of its own, so it never adds to the voice it came from.
    {
        const float level = cur(ParamId::MemoryLevel), blur = cur(ParamId::MemoryBlur), sink = cur(ParamId::MemorySink);
        const int first = static_cast<int>(voices_.size());
        for (size_t gi = 0; gi < ghosts_.size(); ++gi) {
            Ghost& g = ghosts_[gi];
            CloudVoice& cv = cloud_.voice(first + static_cast<int>(gi));
            if (!g.on) { cv.active = 0; continue; }
            g.age += dtf;
            g.env = std::min(1.0f, g.age / g.bloom) * std::exp(-6.9077553f * std::max(0.0f, g.age - g.bloom) / g.life);
            if (g.age > g.bloom && g.env < 1.0e-4f) { g.on = false; cv.active = 0; continue; }
            const float x = std::min(1.0f, g.age / g.life);
            const float keepTime = cv.time;
            cv = g.base;
            cv.time = g.restart ? 0.0f : keepTime;
            cv.restart = g.restart ? 1 : 0;
            cv.onset = 0;
            cv.seed = g.seed;
            cv.active = 1;
            cv.gain = g.env * level;
            cv.width = g.base.width * (1.0f + 3.0f * blur * x);
            cv.drift = g.base.drift + 12.0f * blur * x;
            cv.brightness = g.base.brightness * (1.0f - 0.7f * blur * x);
            cv.distance = g.base.distance + (1.0f - g.base.distance) * sink * std::min(1.0f, 2.0f * x);
            g.restart = false;
        }
    }

    for (int i = 0; i < static_cast<int>(voices_.size()) && i < kMaxViewVoices; ++i) {
        const Voice& v = voices_[static_cast<size_t>(i)];
        const CloudVoice& cv = cloud_.voice(i);
        const bool on = v.note >= 0 && cv.active;
        viewHz_[static_cast<size_t>(i)].store(on ? cv.hz : 0.0f, std::memory_order_relaxed);
        viewGain_[static_cast<size_t>(i)].store(on ? cv.gain : 0.0f, std::memory_order_relaxed);
        viewDistance_[static_cast<size_t>(i)].store(v.distance, std::memory_order_relaxed);
        viewTime_[static_cast<size_t>(i)].store(cv.time, std::memory_order_relaxed);
        viewSource_[static_cast<size_t>(i)].store(v.source, std::memory_order_relaxed);
        viewNote_[static_cast<size_t>(i)].store(on ? v.note : -1, std::memory_order_relaxed);
    }
    for (size_t gi = 0; gi < ghosts_.size(); ++gi) {   // the ghosts after the voices, as source 3
        const size_t i = voices_.size() + gi;
        if (i >= static_cast<size_t>(kMaxViewVoices)) break;
        const CloudVoice& cv = cloud_.voice(static_cast<int>(i));
        const bool on = ghosts_[gi].on && cv.active;
        viewHz_[i].store(on ? cv.hz : 0.0f, std::memory_order_relaxed);
        viewGain_[i].store(on ? cv.gain : 0.0f, std::memory_order_relaxed);
        viewDistance_[i].store(cv.distance, std::memory_order_relaxed);
        viewTime_[i].store(cv.time, std::memory_order_relaxed);
        viewSource_[i].store(3, std::memory_order_relaxed);
        viewNote_[i].store(-1, std::memory_order_relaxed);
    }

    // 4. The modal hall's settings: it runs on the GPU in the same graph, hearing the voices' block
    //    there. The level is applied here, so the hall rests (on = 0) only once the level has been
    //    at 0 for a whole block; its pitches are the scale's, from its lowest mode to 16 kHz.
    {
        ModalHall& mh = cloud_.modal();
        const float level = cur(ParamId::ModalLevel);
        mh.on = (level > 0.0f || prevModalLevel_ > 0.0f) ? 1 : 0;
        mh.freeze = cur(ParamId::ModalFreeze) >= 0.5f ? 1 : 0;
        mh.decay = cur(ParamId::ModalDecay);
        mh.damp = cur(ParamId::ModalDamp);
        mh.tune = cur(ParamId::ModalTune);
        mh.spread = cur(ParamId::ModalSpread);
        mh.drift = cur(ParamId::ModalDrift);
        mh.width = cur(ParamId::ModalWidth);
        mh.low = cur(ParamId::ModalLow);
        mh.time = modalTime_;
        mh.breath = cur(ParamId::ModalBreath);
        // The painted hall reads the landscape at its own clock, wrapped at the landscape's loop.
        mh.paint = cur(ParamId::ModalPaint);
        mh.paintNorm = 1.0f / std::max(0.125f, landRms_);
        paintTime_ += dtf;
        if (landLoop_ > 0.0f && paintTime_ >= landLoop_) paintTime_ -= landLoop_;
        mh.paintTime = paintTime_;
        int count = 0;
        if (mh.on)
            for (int note = 0; note < 128 && count < kModalTuned; ++note) {
                const double f = frequencyOf(note) * tideMul;
                if (f >= mh.low && f <= 16000.0) mh.tuned[count++] = static_cast<float>(f);
            }
        mh.tunedCount = count;
        modalTime_ += dtf;
        if (modalTime_ >= 1000.0f) modalTime_ -= 1000.0f;
    }

    // The field on the partials and the dispersion of their clocks. Scale: the scale's pitches,
    // ascending; Swarm and Scatter: eight Lorenz systems always wander (so the count can change
    // without a jump), the first `count` are the attractors -- x across seven octaves from 60 Hz,
    // z the weight, negative for Scatter.
    {
        SpectralField& F = cloud_.field();
        const int mode = std::clamp(static_cast<int>(std::lround(cur(ParamId::FieldMode))), 0, kNumFieldModes - 1);
        F.mode = static_cast<FieldMode>(mode);
        const float pull = cur(ParamId::FieldStrength), spring = cur(ParamId::FieldSpring);
        F.strength = 600.0f * pull * pull;
        F.radius = cur(ParamId::FieldRadius);
        F.spring = 4.0f * spring * spring;
        F.dispersion = cur(ParamId::Dispersion);
        F.flockFresh = (F.mode == FieldMode::Flock && !flockWas_) ? 1 : 0;
        flockWas_ = F.mode == FieldMode::Flock;
        const double span = static_cast<double>(cur(ParamId::FieldRate)) * dt;
        const int steps = std::max(1, static_cast<int>(std::ceil(span / 0.002)));
        const double h = span / steps;
        for (auto& s : lorenz_)
            for (int k = 0; k < steps; ++k) {   // the midpoint rule, small steps
                auto d = [](const double* p, double* o) { o[0] = 10.0 * (p[1] - p[0]); o[1] = p[0] * (28.0 - p[2]) - p[1]; o[2] = p[0] * p[1] - (8.0 / 3.0) * p[2]; };
                double k1[3], mid[3], k2[3];
                d(s, k1);
                for (int j = 0; j < 3; ++j) mid[j] = s[j] + 0.5 * h * k1[j];
                d(mid, k2);
                for (int j = 0; j < 3; ++j) s[j] += h * k2[j];
            }
        if (F.mode == FieldMode::Scale) {
            int count = 0;
            for (int note = 0; note < 128 && count < kFieldPoints; ++note) {
                const float lf = static_cast<float>(std::log2(frequencyOf(note) * tideMul));
                if (count == 0 || lf > F.pos[count - 1] + 1.0e-4f) F.pos[count++] = lf;
            }
            F.count = count;
        } else if (F.mode == FieldMode::Flock) {
            // The flock's winds: each cloud blown by one of the eight Lorenz systems' x (cloud h by h mod 8).
            for (int k = 0; k < kFieldAttractors; ++k) F.pos[k] = std::clamp(static_cast<float>(lorenz_[k][0] / 22.0), -1.0f, 1.0f);
            F.count = kFieldAttractors;
        } else if (F.mode != FieldMode::Off) {
            const int count = std::clamp(static_cast<int>(std::lround(cur(ParamId::FieldCount))), 1, kFieldAttractors);
            const float lo = std::log2(60.0f), octaves = std::log2(8000.0f / 60.0f);
            for (int k = 0; k < count; ++k) {
                const float u = std::clamp(static_cast<float>((lorenz_[k][0] + 22.0) / 44.0), 0.0f, 1.0f);
                const float w = 0.4f + 0.8f * std::clamp(static_cast<float>(lorenz_[k][2] / 50.0), 0.0f, 1.0f);
                F.pos[k] = lo + u * octaves;
                F.mass[k] = F.mode == FieldMode::Scatter ? -w : w;
            }
            F.count = count;
        } else {
            F.count = 0;
        }
    }

    // The living landscape: its mode (a change seeds it), its steps (up to two a block: Lenia 2 to 40
    // a second, the Turing field 20 to 750, by Life Rate), its character (Lenia's growth; the
    // Turing field along spots, mitosis, coral, worms, mazes in Pearson's map), the make-up from
    // the field's measured power, and a new seeding when it has died or filled up.
    {
        LifeField& L = cloud_.life();
        const int mode = std::clamp(static_cast<int>(std::lround(cur(ParamId::LifeMode))), 0, kNumLifeModes - 1);
        L.mode = static_cast<LifeMode>(mode);
        lifeQuiet_ += dtf;
        if (mode != lifeModeWas_ && mode != 0) { L.reseed = 1; ++L.salt; lifeAcc_ = 0.0f; lifeQuiet_ = 0.0f; lifeMakeup_ = 1.0f; }
        lifeModeWas_ = mode;
        if (mode != 0) {
            const float rate = cur(ParamId::LifeRate);
            if (lifeIsPicture(L.mode)) {
                // The animated pictures: a step every block; the flow at Life Rate (squared: slow at
                // the bottom), the turn at Turn, Rain's drops by Life Rate too.
                L.steps = 1;
                pictureTime_ += (0.01 + 0.5 * static_cast<double>(rate) * rate) * dt;
                pictureAngle_ += 2.0 * 3.14159265358979 * static_cast<double>(cur(ParamId::LifeTurn)) / 60.0 * dt;
                pictureSeconds_ += dt;
                pictureAngle_ = std::remainder(pictureAngle_, 2.0 * 3.14159265358979 * 60.0);   // whole turns away, kept small
                L.time = static_cast<float>(std::fmod(pictureTime_, 4096.0));
                L.angle = static_cast<float>(pictureAngle_);
                L.seconds = static_cast<float>(std::fmod(pictureSeconds_, 600000.0));
                L.sides = std::round(cur(ParamId::LifeSides));
                L.character = std::clamp(cur(ParamId::LifeCharacter), 0.0f, 1.0f);
                L.drops = static_cast<float>((0.5 + 30.0 * static_cast<double>(rate) * rate) * dt / (kLifeW * kLifeH));   // drops a second, over every cell
                L.turnStep = static_cast<float>(2.0 * 3.14159265358979 * static_cast<double>(cur(ParamId::LifeTurn)) / 60.0 * dt);
            } else {
                lifeAcc_ += (L.mode == LifeMode::Lenia ? 2.0f + 38.0f * rate : 20.0f + 730.0f * rate) * dtf;
                const int steps = std::min(2, static_cast<int>(lifeAcc_));
                lifeAcc_ = std::min(lifeAcc_ - static_cast<float>(steps), 2.0f);
                L.steps = steps;
            }
            const float ch = std::clamp(cur(ParamId::LifeCharacter), 0.0f, 1.0f);
            L.lenMu = 0.12f + 0.08f * ch;
            L.lenSigma = 0.012f + 0.012f * ch;
            static const float kF[5] = { 0.035f, 0.0367f, 0.0545f, 0.078f, 0.029f }, kK[5] = { 0.065f, 0.0649f, 0.062f, 0.061f, 0.057f };
            const float at = ch * 4.0f;
            const int seg = std::min(3, static_cast<int>(at));
            const float fr = at - static_cast<float>(seg);
            L.feed = kF[seg] + (kF[seg + 1] - kF[seg]) * fr;
            L.kill = kK[seg] + (kK[seg + 1] - kK[seg]) * fr;
            L.depth = cur(ParamId::LifeDepth);
            L.zoom = cur(ParamId::LifeZoom);
            float mean = 0.0f, meanSq = 0.0f;
            cloud_.lifeStats(mean, meanSq);
            // The make-up m that keeps the partials' power: E[(1 - d + d v m)^2] = 1, a quadratic in m
            // (one over the RMS is right only at full depth; below it the steady share is mixed in).
            {
                const double d = std::clamp(static_cast<double>(L.depth), 0.0, 1.0), e = mean, q = std::max(static_cast<double>(meanSq), 1.0 / 4096.0);
                const double A = d * d * q, B = 2.0 * (1.0 - d) * d * e, C = (1.0 - d) * (1.0 - d) - 1.0;
                const double m = A > 1.0e-12 ? (-B + std::sqrt(std::max(B * B - 4.0 * A * C, 0.0))) / (2.0 * A) : 1.0;
                lifeMakeup_ += (1.0f - std::exp(-dtf / 0.5f)) * (static_cast<float>(std::clamp(m, 0.0, 8.0)) - lifeMakeup_);
                L.makeup = lifeMakeup_;
            }
            const bool dead = !lifeIsPicture(L.mode) && mean < 0.002f, full = L.mode == LifeMode::Lenia && mean > 0.85f;
            if ((dead || full) && lifeQuiet_ > 2.0f && L.reseed == 0) { L.reseed = 1; ++L.salt; lifeQuiet_ = 0.0f; }
        } else {
            L.steps = 0;
        }
    }

    stepRipple(dtf);

    // The spectral comb: its phase runs on at the sweep (kept within a notch), and the make-up is
    // exact -- the raised cosine's mean square is (1 - d/2)^2 + d^2/8 over the notches.
    {
        SpectralComb& K = cloud_.comb();
        K.depth = cur(ParamId::CombDepth);
        K.density = cur(ParamId::CombDensity);
        K.stereo = cur(ParamId::CombStereo);
        combPhase_ += static_cast<double>(cur(ParamId::CombSweep)) * dt;
        combPhase_ -= std::floor(combPhase_);
        K.phase = static_cast<float>(combPhase_);
        const float d = K.depth;
        K.makeup = 1.0f / std::sqrt(std::max((1.0f - 0.5f * d) * (1.0f - 0.5f * d) + 0.125f * d * d, 1.0e-6f));
    }

    // The reservoir: stepped only while a row reads it; fed with what the instrument hears and
    // plays (the sources as the last block left them), each standardised against its last half
    // minute -- the reservoir hears how the music departs from its recent self.
    {
        Reservoir& R = cloud_.reservoir();
        R.on = 0;
        for (int r = 0; r < kModRows; ++r) {
            const ModRow row = unpackModRow(modRows_[static_cast<size_t>(r)].load(std::memory_order_relaxed));
            auto isRes = [](ModSource s) { return s >= ModSource::Reservoir1 && s <= ModSource::Reservoir4; };
            if (row.target >= 0 && row.depth != 0.0f && (isRes(row.source) || isRes(row.via))) R.on = 1;
        }
        if (sourcesWatched_.load(std::memory_order_relaxed)) R.on = 1;
        R.chaos = cur(ParamId::ReservoirChaos);
        R.leak = 1.0f - std::exp(-dtf / std::max(cur(ParamId::ReservoirMemory), 0.05f));
        R.input = cur(ParamId::ReservoirInput);
        const ModSource feed[kReservoirInputs] = { ModSource::Loudness, ModSource::Centroid, ModSource::Flux, ModSource::Roughness,
                                                   ModSource::Beat, ModSource::Note, ModSource::Cascade, ModSource::Lenia1 };
        const float a = 1.0f - std::exp(-dtf / 30.0f);
        for (int k = 0; k < kReservoirInputs; ++k) {
            const float v = mods_.value(feed[k]);
            if (!resStarted_) { resMean_[k] = v; resVar_[k] = 0.0f; }
            const float d = v - resMean_[k];
            resMean_[k] += a * d;
            resVar_[k] = (1.0f - a) * (resVar_[k] + a * d * d);
            R.u[k] = std::clamp(0.5f * d / (std::sqrt(resVar_[k]) + 0.05f), -2.0f, 2.0f);
        }
        resStarted_ = true;
    }

    // Partial sync: the coupling in units of the oscillators' own spread of rates -- 2.5 times it at
    // full Coupling, well past the critical strength (for a uniform spread about 1.3 times it).
    {
        PartialSync& S = cloud_.sync();
        S.depth = cur(ParamId::SyncDepth);
        S.rate = cur(ParamId::SyncRate);
        S.spread = cur(ParamId::SyncSpread);
        S.mode = static_cast<SyncMode>(std::clamp(static_cast<int>(std::lround(cur(ParamId::SyncMode))), 0, kNumSyncModes - 1));
        S.coupling = cur(ParamId::SyncCoupling) * 2.5f * kTwoPi * S.rate * (S.spread + 0.05f);
        // Chimera and Swarm live on their own time, a couple of the rate's turns: in units of it.
        if (S.mode != SyncMode::Global) S.coupling = cur(ParamId::SyncCoupling) * 2.0f * kTwoPi * std::max(S.rate, 0.05f);
        S.swarm = cur(ParamId::SyncSwarm) * 2.0f * kTwoPi * std::max(S.rate, 0.05f);
        S.reach = cur(ParamId::SyncReach);
        S.lag = 1.25f + 0.3f * std::clamp(cur(ParamId::SyncLag), 0.0f, 1.0f);   // 1.39 (Abrams and Strogatz) at about 0.47
    }

    // The binaural cloud (its orbit turns the whole of it) and the spectral echo (its lines
    // cleared when it is switched on; its delay less the 1024 - G samples its window adds).
    {
        Binaural& b = cloud_.binaural();
        b.on = cur(ParamId::Binaural) >= 0.5f ? 1 : 0;
        b.span = cur(ParamId::BinauralSpan);
        b.height = cur(ParamId::BinauralHeight);
        binAngle_ += 6.2831853f * cur(ParamId::BinauralOrbit) * dtf;
        if (binAngle_ > 3.14159265f) binAngle_ -= 6.2831853f;
        if (binAngle_ < -3.14159265f) binAngle_ += 6.2831853f;
        b.angle = binAngle_;
        SpectralEcho& se = cloud_.spectralEcho();
        const bool on = cur(ParamId::SpecMix) > 0.0f || cur(ParamId::SpecToFar) * cur(ParamId::SpecMix) > 0.0f || prevSpecMix_ > 0.0f;
        if (on && !specWasOn_) se.clear = 1;
        specWasOn_ = on;
        se.on = on ? 1 : 0;
        se.frames = std::max(1.0f, (cur(ParamId::SpecTime) * static_cast<float>(cfg_.sampleRate) - static_cast<float>(1024 - G)) / static_cast<float>(G));
        se.tilt = cur(ParamId::SpecTilt);
        se.scatter = cur(ParamId::SpecScatter);
        se.feedback = cur(ParamId::SpecFeedback);
        se.damp = cur(ParamId::SpecDamp);

        // The melt: its time to -60 dB as the held energy's share kept a frame, the spread as a
        // diffusion (after a second the standard deviation is Spread octaves), the bands' pull to
        // their centres and their detuning growing with it.
        Melt& me = cloud_.melt();
        const bool meltOn = cur(ParamId::MeltMix) > 0.0f || prevMeltMix_ > 0.0f;
        if (meltOn && !meltWasOn_) me.clear = 1;
        meltWasOn_ = meltOn;
        me.on = meltOn ? 1 : 0;
        const float hopS = static_cast<float>(kMeltHop) / static_cast<float>(cfg_.sampleRate);
        const float spread = cur(ParamId::MeltSpread);
        me.keep = std::pow(10.0f, -6.0f * hopS / std::max(cur(ParamId::MeltTime), 0.1f));
        me.widen = spread * 0.693147f * std::sqrt(hopS);
        me.pull = 1.0f - std::exp(-2.0f * spread * hopS);
        me.detune = (0.1f + 0.4f * std::min(spread, 1.0f)) * 6.2831853f / static_cast<float>(kMeltN);
    }

    // 5. The GPU: near and far, left and right, the modal hall and the spectral echo.
    // The picture's snapshot about thirty times a second, brought down with the block.
    if (--snapCountdown_ <= 0) {
        snapCountdown_ = std::max(1, static_cast<int>(std::lround(0.033 * cfg_.sampleRate / G)));
        cloud_.requestSnapshot();
    }
    LARGE_INTEGER qpcLaunch, qpcBack;
    QueryPerformanceCounter(&qpcLaunch);
    cloud_.launch();
    const float* q = cloud_.wait();
    QueryPerformanceCounter(&qpcBack);
    {
        const float* dots = nullptr;
        int per = 0;
        if (cloud_.takeSnapshot(dots, per)) {
            std::unique_lock<std::mutex> lock(snapLock_, std::try_to_lock);   // the picture is reading: this one is skipped
            if (lock.owns_lock()) {
                const size_t count = static_cast<size_t>(cloud_.config().voices) * static_cast<size_t>(per) * 2;
                if (snap_.size() != count) snap_.resize(count);   // once: the size stays
                std::memcpy(snap_.data(), dots, count * sizeof(float));
                snapPer_ = per;
                if (const float* cells = cloud_.lifePicture()) snapLife_.assign(cells, cells + kLifeW * kLifeH);
                else snapLife_.clear();
                ++snapSeq_;
            }
        }
    }
    gpuMs_.store(cloud_.lastGpuMs(), std::memory_order_relaxed);
    for (int s = 0; s < G; ++s) {
        const float* f = q + static_cast<size_t>(kCloudOutChannels) * s;
        const size_t k = static_cast<size_t>(s);
        nearL_[k] = f[0];
        nearR_[k] = f[1];
        farL_[k] = f[2] * blk_.farLevel;
        farR_[k] = f[3] * blk_.farLevel;
        modalL_[k] = f[4];
        modalR_[k] = f[5];
        specL_[k] = f[6];
        specR_[k] = f[7];
        meltL_[k] = f[8];
        meltR_[k] = f[9];
    }
    // A level from the last block's to this one's, across the block (no zipper when it is modulated).
    auto ramp = [G](float from, float to, int s) { return from + (to - from) * static_cast<float>(s + 1) / static_cast<float>(G); };

    // 6. The near plane, as Noctuary's: two echoes, the second in series after the first, receding
    //    partly into the far plane; a small reverb; the early reflections of a room, from the dry voices.
    std::copy(nearL_.begin(), nearL_.end(), dryL_.begin());
    std::copy(nearR_.begin(), nearR_.end(), dryR_.begin());
    delay_.set(cur(ParamId::DelayTimeL), cur(ParamId::DelayTimeR), cur(ParamId::DelayFeedback), cur(ParamId::DelayCross),
               cur(ParamId::DelayDamp), cur(ParamId::DelayAbsorb));
    delay_.setDuck(cur(ParamId::DelayDuck));
    delay_.process(nearL_.data(), nearR_.data(), wetL_.data(), wetR_.data(), G);
    {
        const float m1 = cur(ParamId::DelayMix), t1 = cur(ParamId::DelayToFar);
        for (int s = 0; s < G; ++s) {
            const size_t k = static_cast<size_t>(s);
            const float m = ramp(prevDelayMix_, m1, s), tf = ramp(prevDelayToFar_, t1, s);
            nearL_[k] += wetL_[k] * m;  nearR_[k] += wetR_[k] * m;
            farL_[k] += wetL_[k] * tf;  farR_[k] += wetR_[k] * tf;
        }
        prevDelayMix_ = m1;
        prevDelayToFar_ = t1;
    }
    delay2_.set(cur(ParamId::Delay2TimeL), cur(ParamId::Delay2TimeR), cur(ParamId::Delay2Feedback), cur(ParamId::DelayCross),
                cur(ParamId::Delay2Damp), cur(ParamId::DelayAbsorb));
    delay2_.setDuck(cur(ParamId::DelayDuck));
    delay2_.process(nearL_.data(), nearR_.data(), wetL_.data(), wetR_.data(), G);
    {
        const float m2 = cur(ParamId::Delay2Mix), t2 = cur(ParamId::Delay2ToFar);
        for (int s = 0; s < G; ++s) {
            const size_t k = static_cast<size_t>(s);
            const float m = ramp(prevDelay2Mix_, m2, s), tf = ramp(prevDelay2ToFar_, t2, s);
            nearL_[k] += wetL_[k] * m;  nearR_[k] += wetR_[k] * m;
            farL_[k] += wetL_[k] * tf;  farR_[k] += wetR_[k] * tf;
        }
        prevDelay2Mix_ = m2;
        prevDelay2ToFar_ = t2;
    }
    {   // the spectral echo, from the GPU: onto the near plane and partly into the far
        const float m3 = cur(ParamId::SpecMix), t3 = cur(ParamId::SpecToFar) * m3;
        for (int s = 0; s < G; ++s) {
            const size_t k = static_cast<size_t>(s);
            const float m = ramp(prevSpecMix_, m3, s), tf = ramp(prevSpecToFar_, t3, s);
            nearL_[k] += specL_[k] * m;  nearR_[k] += specR_[k] * m;
            farL_[k] += specL_[k] * tf;  farR_[k] += specR_[k] * tf;
        }
        prevSpecMix_ = m3;
        prevSpecToFar_ = t3;
    }
    {   // the melt, from the GPU: onto the near plane and partly into the far
        const float m4 = cur(ParamId::MeltMix), t4 = cur(ParamId::MeltToFar) * m4;
        for (int s = 0; s < G; ++s) {
            const size_t k = static_cast<size_t>(s);
            const float m = ramp(prevMeltMix_, m4, s), tf = ramp(prevMeltToFar_, t4, s);
            nearL_[k] += meltL_[k] * m;  nearR_[k] += meltR_[k] * m;
            farL_[k] += meltL_[k] * tf;  farR_[k] += meltR_[k] * tf;
        }
        prevMeltMix_ = m4;
        prevMeltToFar_ = t4;
    }
    nearRev_.setSpace(0.3f, 20000.0f, 80.0f);
    nearRev_.set(0.6f, cur(ParamId::NearDecay), cur(ParamId::NearDamp), 5.0f, false, cur(ParamId::NearMix));
    nearRev_.process(nearL_.data(), nearR_.data(), G);
    early_.setRoom(cur(ParamId::EarlySize), cur(ParamId::EarlyAbsorb), 1.0f);
    early_.setSource(0.0f, srcDistance_);
    early_.setLevel(cur(ParamId::EarlyLevel));
    if (early_.active()) early_.process(dryL_.data(), dryR_.data(), nearL_.data(), nearR_.data(), G);

    // 7. The far plane is heard only through the far reverb: the swell of the diffusion first, the
    //    hall stepping aside for the near plane band by band after (Unmask).
    diffuser_.set(cur(ParamId::FarDiffuse));
    diffuser_.process(farL_.data(), farR_.data(), G);
    far_.setMode(std::clamp(static_cast<int>(std::lround(cur(ParamId::FarMode))), 0, kNumFarModes - 1));
    far_.set(cur(ParamId::FarSize), cur(ParamId::FarDecay), cur(ParamId::FarDamp), cur(ParamId::FarPreDelay),
             cur(ParamId::FarFreeze) >= 0.5f, 1.0f);
    far_.setSpace(0.5f, 9000.0f, 60.0f);
    far_.process(farL_.data(), farR_.data(), G);
    unmask_.set(cur(ParamId::FarUnmask), 0.3f, 0.5f);
    unmask_.process(nearL_.data(), nearR_.data(), farL_.data(), farR_.data(), G);

    // 8. Together: the near plane, the far hall and the modal hall.
    {
        const float ml = cur(ParamId::ModalLevel);
        for (int s = 0; s < G; ++s) {
            const size_t k = static_cast<size_t>(s);
            const float m = ramp(prevModalLevel_, ml, s);
            nearL_[k] += farL_[k] + modalL_[k] * m;
            nearR_[k] += farR_[k] + modalR_[k] * m;
        }
        prevModalLevel_ = ml;
    }
    // Binaural: the master leaves the two ears alone (no mono bass, no width): both would undo the head.
    if (cur(ParamId::Binaural) >= 0.5f) midSide_.set(20.0f, 0.0f, 1.0f);
    else midSide_.set(cur(ParamId::BassMono), 0.0f, cur(ParamId::StereoWidth));
    midSide_.process(nearL_.data(), nearR_.data(), G);
    for (int s = 0; s < G; ++s) {
        const size_t at = static_cast<size_t>((pos + s) & ringMask_) * 2;
        ring_[at] = softClip(nearL_[static_cast<size_t>(s)] * blk_.master);
        ring_[at + 1] = softClip(nearR_[static_cast<size_t>(s)] * blk_.master);
    }
    renderPos_.store(pos + G, std::memory_order_release);
    LARGE_INTEGER qpcEnd, qpcFreq;
    QueryPerformanceCounter(&qpcEnd);
    QueryPerformanceFrequency(&qpcFreq);
    auto keepWorst = [&](std::atomic<int>& at, LONGLONG from, LONGLONG to) {
        const int us = static_cast<int>((to - from) * 1000000 / qpcFreq.QuadPart);
        int worst = at.load(std::memory_order_relaxed);
        while (us > worst && !at.compare_exchange_weak(worst, us, std::memory_order_relaxed)) {}
    };
    keepWorst(worstBlockUs_, qpcStart.QuadPart, qpcEnd.QuadPart);
    keepWorst(worstPrepUs_, qpcStart.QuadPart, qpcLaunch.QuadPart);
    keepWorst(worstWaitUs_, qpcLaunch.QuadPart, qpcBack.QuadPart);
    keepWorst(worstPostUs_, qpcBack.QuadPart, qpcEnd.QuadPart);
}

} // namespace additiv
