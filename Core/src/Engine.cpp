/**
 * @file Engine.cpp
 * @brief The instrument (Engine.h): voices, conductors, tuning, the GPU cloud, far reverb, master,
 *        and the worker that renders ahead of the audio callback.
 * @author Rene Weller
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

constexpr std::uint32_t QueueSize = 4096;   ///< note events in flight from the audio thread (a power of two)
constexpr float VoiceLevel = 0.1f;          ///< RMS of one voice at full gain, all of its partials together (set so the autoplay sits near -24 dBFS, Noctuary's range)
constexpr double GlideSeconds = 0.3;        ///< how fast a sounding voice follows a change of tuning
constexpr float PurityNarrowing = 0.6f;     ///< at Purity 1 the clouds are this much narrower, so pure intervals can lock (PLAN.md)
constexpr float MetresPerUnit = 20.0f;      ///< one unit of distance, for the doppler (Noctuary: about twenty metres)
constexpr float SpeedOfSound = 343.0f;      ///< m/s

/**
 * @brief Denormals off while it lives (flush to zero, denormals are zero), the old mode back after.
 *
 * The effects' tails -- reverbs, echoes, the far hall -- decay into numbers the CPU takes a
 * hundred times longer over. JUCE's ScopedNoDenormals covers only the audio callback, and the
 * blocks are rendered on the worker (or, offline, wherever process() is called). Measured before:
 * the slowest block of a second at 2.5 ms of a block's 2.7 (and an underrun at a standalone's start).
 */
struct NoDenormals {
    unsigned int m_csr = _mm_getcsr();                       ///< the mode it found
    NoDenormals() { _mm_setcsr(m_csr | 0x8040u); }           ///< FTZ (bit 15) and DAZ (bit 6) on
    ~NoDenormals() { _mm_setcsr(m_csr); }                    ///< the mode back
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
    using SetClassT = long (APIENTRY*)(HANDLE, int);
    HMODULE gdi = GetModuleHandleW(L"gdi32.dll");
    if (!gdi) gdi = LoadLibraryW(L"gdi32.dll");
    const auto fn = gdi ? reinterpret_cast<SetClassT>(reinterpret_cast<void*>(GetProcAddress(gdi, "D3DKMTSetProcessSchedulingPriorityClass"))) : nullptr;
    return fn ? fn(GetCurrentProcess(), 5) : -1;
}

/** @brief Registers the calling thread with MMCSS as "Pro Audio" while it lives. */
struct ProAudioThread {
    HANDLE m_task = nullptr;   ///< the registration, null when it failed

    /** @brief Registers the thread at high priority within Pro Audio. */
    ProAudioThread()
    {
        DWORD index = 0;
        m_task = AvSetMmThreadCharacteristicsW(L"Pro Audio", &index);
        if (m_task) AvSetMmThreadPriority(m_task, AVRT_PRIORITY_HIGH);
    }
    ~ProAudioThread() { if (m_task) AvRevertMmThreadCharacteristics(m_task); }
    ProAudioThread(const ProAudioThread&) = delete;              ///< one registration per thread
    ProAudioThread& operator=(const ProAudioThread&) = delete;   ///< one registration per thread
};

/**
 * @brief The smallest power of two at least x.
 * @param[in] x  a positive number
 * @return the power
 */
std::int64_t nextPow2(std::int64_t x)
{
    std::int64_t p = 1;
    while (p < x) p <<= 1;
    return p;
}

/** @brief The generated landscapes and their RMS, made once (generating one takes up to a tenth of a second). */
struct FactoryLandscapes {
    std::vector<Landscape> m_land;   ///< NumFactoryLandscapes of them
    std::vector<float> m_rms;        ///< and their RMS

    FactoryLandscapes()
    {
        for (int i = 0; i < NumFactoryLandscapes; ++i) {
            m_land.push_back(factoryLandscape(i));
            m_rms.push_back(landscapeRms(m_land.back()));
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
    for (auto& r : m_modRows) r.store(packModRow(ModRow{}), std::memory_order_relaxed);
    for (auto& p : m_played) p.store(std::numeric_limits<float>::quiet_NaN(), std::memory_order_relaxed);
    for (auto& s : m_chanSlide) s.store(0.5f, std::memory_order_relaxed);   // MPE: a slide at rest
}

void Engine::setModRow(int i, const ModRow& row)
{
    if (i < 0 || i >= ModRows) return;
    ModRow r = row;
    if (r.m_target >= 0 && !isModTarget(static_cast<ParamIdE>(r.m_target))) r.m_target = -1;
    r.m_depth = std::clamp(r.m_depth, -1.0f, 1.0f);
    m_modRows[static_cast<size_t>(i)].store(packModRow(r), std::memory_order_relaxed);
}

ModRow Engine::modRow(int i) const
{
    if (i < 0 || i >= ModRows) return {};
    return unpackModRow(m_modRows[static_cast<size_t>(i)].load(std::memory_order_relaxed));
}

std::string Engine::modText() const
{
    ModRow rows[ModRows];
    for (int i = 0; i < ModRows; ++i) rows[i] = modRow(i);
    return modRowsToText(rows, ModRows);
}

void Engine::setModText(const std::string& text)
{
    ModRow rows[ModRows];
    modRowsFromText(text, rows, ModRows);
    for (int i = 0; i < ModRows; ++i) setModRow(i, rows[i]);
}

Engine::~Engine() { release(); }

void Engine::resetParams()
{
    for (const ParamDesc& d : paramTable()) m_params[static_cast<size_t>(d.m_id)].store(d.m_def, std::memory_order_relaxed);
}

void Engine::setParam(ParamIdE id, float value)
{
    const ParamDesc& d = paramDesc(id);
    m_params[static_cast<size_t>(id)].store(std::clamp(value, d.m_min, d.m_max), std::memory_order_relaxed);
}

std::string Engine::deviceName() const { return m_prepared ? m_cloud.deviceName() : std::string(); }

bool Engine::prepare(const EngineConfig& config, std::string* error)
{
    release();
    m_cfg = config;
    const int G = m_cfg.m_gpuBlock;
    if (G < 32 || G > 4096 || (G & (G - 1)) != 0) { if (error) *error = "gpuBlock must be a power of two, 32 .. 4096"; return false; }
    m_cfg.m_maxHostBlock = std::max(m_cfg.m_maxHostBlock, 1);

    CloudConfig cc;
    cc.m_sampleRate = m_cfg.m_sampleRate;
    cc.m_blockSize = G;
    cc.m_voices = m_cfg.m_voices + std::max(0, m_cfg.m_ghosts);
    cc.m_partialsPerVoice = m_cfg.m_partialsPerVoice;
    cc.m_spinWait = true;
    if (!m_cloud.open(cc, error)) return false;
    {   // the hall remembers across a new start (a new sample rate, a new block): its memory goes up again
        const std::lock_guard<std::mutex> lock(m_snapLock);
        if (!m_hallPendingSet && m_snapHallValid) {
            m_hallPending = m_snapHall;
            m_hallPendingAge = m_snapHallAge;
            m_hallPendingSet = true;
        }
    }
    const std::vector<EnvShape> bank = factoryShapes();
    m_cloud.setShapes(bank.data(), static_cast<int>(bank.size()));
    factoryLandscapes();   // made now, not in the middle of a block
    m_landChoice = -1;
    if (m_cfg.m_realtimeGpuClass) raiseGpuClass();

    m_voices.assign(static_cast<size_t>(m_cfg.m_voices), Voice{});
    m_foundationNote = -1;
    for (Voice& v : m_voices) v.m_env.setSampleRate(m_cfg.m_sampleRate / G);   // stepped once per GPU block
    m_rng.seed(m_cfg.m_seed);
    m_side.seed(m_cfg.m_seed ^ 0xA5A5A5A55A5A5A5Aull);
    m_tide.init(m_side);
    std::fill(std::begin(m_specKey), std::end(m_specKey), -1.0f);
    m_scaleChoice = -1;
    m_mods.reset(m_cfg.m_seed ^ 0x6D6F647363ull);
    for (OwnUnit& o : m_own) o.m_live = false;
    m_soundValid = false;
    m_morphT = 1.0f;
    m_morphSeen = m_morphAsked.load(std::memory_order_acquire);
    m_abWas = false;
    m_mapWas = false;
    m_phraseStart = false;
    m_prevSounding = 0;
    readParams();
    refreshSpectrum();
    refreshScale();
    refreshLandscape();
    m_brain.reset(m_rng.fork(), 48 + m_rootPc);
    m_brain2.reset(m_side.fork(), 36 + m_rootPc);   // its own stream: switching it on never moves the first
    m_stepWas = param(ParamIdE::AUTO_STEP) >= 0.5f;
    m_far.prepare(m_cfg.m_sampleRate);
    m_diffuser.prepare(m_cfg.m_sampleRate);
    m_unmask.prepare(m_cfg.m_sampleRate);
    m_delay.prepare(m_cfg.m_sampleRate);
    m_delay2.prepare(m_cfg.m_sampleRate);
    m_early.prepare(m_cfg.m_sampleRate);
    m_nearRev.prepare(m_cfg.m_sampleRate);
    m_midSide.prepare(m_cfg.m_sampleRate);
    for (std::vector<float>* b : { &m_nearL, &m_nearR, &m_farL, &m_farR, &m_dryL, &m_dryR, &m_wetL, &m_wetR, &m_modalL, &m_modalR, &m_specL, &m_specR, &m_meltL, &m_meltR })
        b->assign(static_cast<size_t>(G), 0.0f);
    m_ghosts.assign(static_cast<size_t>(std::max(0, m_cfg.m_ghosts)), Ghost{});
    m_ghostRng = Rng(m_cfg.m_seed ^ 0x47686F7374ull);
    m_noteRng = Rng(m_cfg.m_seed ^ 0x4E6F7465ull);
    m_randomNote = 0.0f;
    m_roughness = 0.0f;
    m_roughTick = 0;
    m_prevSpecMix = m_prevSpecToFar = 0.0f;
    m_specWasOn = false;
    m_prevMeltMix = m_prevMeltToFar = 0.0f;
    m_meltWasOn = false;
    m_binAngle = 0.0f;
    m_paintTime = 0.0f;
    m_prevDelayMix = m_prevDelayToFar = m_prevDelay2Mix = m_prevDelay2ToFar = m_prevModalLevel = 0.0f;
    m_modalTime = 0.0f;
    m_srcDistance = 0.3f;
    {   // The attractors' Lorenz systems: each started near the butterfly from the seed, apart from the others.
        Rng r(m_cfg.m_seed ^ 0x4C6F72656Eull);
        for (auto& s : m_lorenz) { s[0] = 8.0 * r.bipolar(); s[1] = 8.0 * r.bipolar(); s[2] = 20.0 + 8.0 * r.bipolar(); }
    }

    m_queue.assign(QueueSize, Event{});
    m_qHead.store(0);
    m_qTail.store(0);
    m_pending.clear();
    m_pending.reserve(QueueSize);

    // The timeline runs two host blocks plus the lookahead ahead of what the callback has taken,
    // rounded up to whole GPU blocks; the ring holds that and room to spare. Two, because a driver
    // may deliver two callbacks back to back (WASAPI did, in the standalone: the ring stood at 448
    // samples for a callback of 480 while the slowest block took 2 ms), and the worker cannot
    // render between them.
    const int ahead = 2 * m_cfg.m_maxHostBlock + static_cast<int>(std::ceil(m_cfg.m_lookaheadMs * 0.001 * m_cfg.m_sampleRate));
    m_latency = (ahead + G - 1) / G * G;
    // The start reserve: as much again rendered before the first callback and not replaced while it
    // is used up, so the first callbacks may come in a burst (a device starting fills its buffer at
    // once: the standalone's ring fell to 224 of 1536 in its first second, with an underrun) while
    // the first GPU blocks are still slow. The latency the host is told stays; an event in the
    // reserve's first stretch comes up to the reserve later (stampFor), never late.
    m_startReserve = m_realtime ? m_latency : 0;   // offline, process() renders what it takes: nothing runs ahead
    const std::int64_t frames = nextPow2(static_cast<std::int64_t>(m_latency) + m_startReserve + m_cfg.m_maxHostBlock + 2 * G);
    m_ring.assign(static_cast<size_t>(frames) * 2, 0.0f);
    m_ringMask = frames - 1;
    m_renderPos.store(0);
    m_outPos.store(0);
    m_underruns.store(0);
    m_late.store(0);
    m_sounding.store(0);
    m_prepared = true;
    if (m_realtime) {
        startWorker();
        // The first callback must find the ring full: the first blocks are the slowest (the GPU's
        // module loads on its first launch), and the host may call at once.
        for (int waited = 0; waited < 500 && m_renderPos.load(std::memory_order_acquire) + G <= m_latency + m_startReserve; ++waited) Sleep(1);
    }
    return true;
}

void Engine::release()
{
    stopWorker();
    if (m_prepared) m_cloud.close();
    m_prepared = false;
}

void Engine::setRealtime(bool realtime)
{
    if (realtime == m_realtime) return;
    if (!realtime) stopWorker();
    m_realtime = realtime;
    if (m_realtime && m_prepared) startWorker();
}

void Engine::startWorker()
{
    if (m_worker.joinable()) return;
    m_running.store(true);
    m_worker = std::thread([this] { workerLoop(); });
}

void Engine::stopWorker()
{
    if (!m_worker.joinable()) return;
    m_running.store(false);
    m_worker.join();
}

void Engine::workerLoop()
{
    ProAudioThread pro;
    // A high-resolution timer for the naps: a plain Sleep(1) can last a whole 15.6 ms timer tick,
    // more than the lookahead.
    HANDLE timer = CreateWaitableTimerExW(nullptr, nullptr, CREATE_WAITABLE_TIMER_HIGH_RESOLUTION, TIMER_ALL_ACCESS);
    const int G = m_cfg.m_gpuBlock;
    while (m_running.load(std::memory_order_relaxed)) {
        const std::int64_t out = m_outPos.load(std::memory_order_acquire);
        std::int64_t r = m_renderPos.load(std::memory_order_relaxed);
        if (r < out) { r = out; m_renderPos.store(r, std::memory_order_release); }   // after an underrun: catch up
        // Ahead by the latency, and at the start by the reserve not used up yet (out + its rest
        // stays at its end until the callbacks have taken it).
        const std::int64_t reserve = std::max<std::int64_t>(0, m_startReserve - out);
        if (r + G <= out + m_latency + reserve) { renderBlock(); continue; }
        if (timer) {
            LARGE_INTEGER due;
            due.QuadPart = -5000;   // 0.5 ms
            if (SetWaitableTimerEx(timer, &due, 0, nullptr, nullptr, nullptr, 0)) { WaitForSingleObject(timer, 5); continue; }
        }
        Sleep(1);
    }
    if (timer) CloseHandle(timer);
}

void Engine::noteOn(int note, float velocity, int offset, int channel)
{
    const std::uint32_t head = m_qHead.load(std::memory_order_relaxed);
    if (!m_prepared || head - m_qTail.load(std::memory_order_acquire) >= QueueSize) return;
    m_queue[head & (QueueSize - 1)] = { 0, note, std::clamp(velocity, 0.0f, 1.0f), stampFor(offset), std::clamp(channel, -1, 15) };
    m_qHead.store(head + 1, std::memory_order_release);
}

void Engine::noteOff(int note, int offset, int channel)
{
    const std::uint32_t head = m_qHead.load(std::memory_order_relaxed);
    if (!m_prepared || head - m_qTail.load(std::memory_order_acquire) >= QueueSize) return;
    m_queue[head & (QueueSize - 1)] = { 1, note, 0.0f, stampFor(offset), std::clamp(channel, -1, 15) };
    m_qHead.store(head + 1, std::memory_order_release);
}

void Engine::allNotesOff()
{
    const std::uint32_t head = m_qHead.load(std::memory_order_relaxed);
    if (!m_prepared || head - m_qTail.load(std::memory_order_acquire) >= QueueSize) return;
    m_queue[head & (QueueSize - 1)] = { 2, 0, 0.0f, stampFor(0) };
    m_qHead.store(head + 1, std::memory_order_release);
}

void Engine::process(float* L, float* R, int n)
{
    if (!m_prepared) {
        std::fill(L, L + n, 0.0f);
        std::fill(R, R + n, 0.0f);
        return;
    }
    for (int done = 0; done < n;) {
        const int len = std::min(n - done, m_cfg.m_maxHostBlock);
        const std::int64_t out = m_outPos.load(std::memory_order_relaxed);
        if (!m_realtime)
            while (m_renderPos.load(std::memory_order_relaxed) < out + len) renderBlock();
        const std::int64_t have = m_renderPos.load(std::memory_order_acquire) - out;
        const int take = static_cast<int>(std::clamp<std::int64_t>(have, 0, len));
        const int fill = static_cast<int>(std::clamp<std::int64_t>(have, 0, 1 << 30));
        int seen = m_minFill.load(std::memory_order_relaxed);
        while ((seen < 0 || fill < seen) && !m_minFill.compare_exchange_weak(seen, fill, std::memory_order_relaxed)) {}
        int big = m_maxCallback.load(std::memory_order_relaxed);
        while (n > big && !m_maxCallback.compare_exchange_weak(big, n, std::memory_order_relaxed)) {}
        for (int i = 0; i < take; ++i) {
            const size_t at = static_cast<size_t>((out + i) & m_ringMask) * 2;
            L[done + i] = m_ring[at];
            R[done + i] = m_ring[at + 1];
        }
        if (take < len) {
            std::fill(L + done + take, L + done + len, 0.0f);
            std::fill(R + done + take, R + done + len, 0.0f);
            m_underruns.fetch_add(1, std::memory_order_relaxed);
        }
        m_outPos.store(out + len, std::memory_order_release);
        done += len;
    }
}

void Engine::drainEvents()
{
    std::uint32_t tail = m_qTail.load(std::memory_order_relaxed);
    const std::uint32_t head = m_qHead.load(std::memory_order_acquire);
    while (tail != head && m_pending.size() < QueueSize) m_pending.push_back(m_queue[tail++ & (QueueSize - 1)]);
    m_qTail.store(tail, std::memory_order_release);
}

void Engine::beginMorph(double seconds)
{
    m_morphSeconds.store(static_cast<float>(std::max(seconds, 0.05)), std::memory_order_relaxed);
    m_morphAsked.fetch_add(1, std::memory_order_release);
}

void Engine::setMorphSlot(int slot, const float* values, const std::string& modText)
{
    ModRow rows[ModRows];
    const int n = modRowsFromText(modText, rows, ModRows);
    {
        const std::lock_guard<std::mutex> lock(m_slotLock);
        m_pendingSlots[slot & 1].set(values, rows, n);
    }
    m_slotSet[static_cast<size_t>(slot & 1)].store(true, std::memory_order_relaxed);
    m_abVersion.fetch_add(1, std::memory_order_release);
    m_slotVersion.fetch_add(1, std::memory_order_release);
}

void Engine::setMapTarget(const MorphState& s)
{
    {
        const std::lock_guard<std::mutex> lock(m_slotLock);
        m_pendingSlots[2] = s;
    }
    m_slotSet[2].store(true, std::memory_order_relaxed);
    m_slotVersion.fetch_add(1, std::memory_order_release);
}

void Engine::loadPreset(int index, double glideSeconds)
{
    if (index < 0 || index >= numPresets()) return;
    float values[NumParams];
    std::string mod;
    presetValues(index, values, &mod);
    if (glideSeconds > 0.0) beginMorph(glideSeconds);
    for (int i = 0; i < NumParams; ++i)
        if (!keptByPresets(static_cast<ParamIdE>(i))) setParam(static_cast<ParamIdE>(i), values[i]);
    setModText(mod);
}

void Engine::updateSound()
{
    const double dt = m_cfg.m_gpuBlock / m_cfg.m_sampleRate;
    // The slots, when the editor chose one (taken only if the lock is free: the next block tries again).
    bool slotsChanged = false;
    const int version = m_slotVersion.load(std::memory_order_acquire);
    if (version != m_slotSeen && m_slotLock.try_lock()) {
        const int abVersion = m_abVersion.load(std::memory_order_acquire);
        if (abVersion != m_abSeen) {
            m_slots[0] = m_pendingSlots[0];
            m_slots[1] = m_pendingSlots[1];
            m_abSeen = abVersion;
            slotsChanged = true;
        }
        m_slots[2] = m_pendingSlots[2];
        m_slotLock.unlock();
        m_slotSeen = version;
    }
    // Where the sound goes: the map's blend round its cursor, the A/B blend, or the knobs and the matrix.
    const bool map = param(ParamIdE::MAP_ACTIVE) >= 0.5f && mapTargetSet();
    const bool ab = !map && param(ParamIdE::MORPH_ACTIVE) >= 0.5f && morphSlotSet(0) && morphSlotSet(1);
    if (map) {
        m_target = m_slots[2];
    } else if (ab) {
        blendMorph(m_slots[0], m_slots[1], std::clamp(param(ParamIdE::MORPH_POS), 0.0f, 1.0f), m_target);
    } else {
        float live[NumParams];
        for (int i = 0; i < NumParams; ++i) live[i] = param(static_cast<ParamIdE>(i));
        ModRow rows[ModRows];
        for (int r = 0; r < ModRows; ++r) rows[r] = unpackModRow(m_modRows[static_cast<size_t>(r)].load(std::memory_order_relaxed));
        m_target.set(live, rows, ModRows);
    }
    for (int i = 0; i < NumParams; ++i)   // the listener's and the player's own stay as they are set
        if (keptByPresets(static_cast<ParamIdE>(i)) || static_cast<ParamIdE>(i) == ParamIdE::AUTO_STEP)
            m_target.m_values[i] = param(static_cast<ParamIdE>(i));
    // On the map the conductor plays or rests as the player set it: a preset made for the hands among
    // the neighbours would otherwise stop it where its weight passes the middle.
    if (map) m_target.m_values[static_cast<size_t>(ParamIdE::BRAIN_ON)] = param(ParamIdE::BRAIN_ON);
    // A glide asked for, or the A/B morph switched or re-chosen: from what played the last block.
    float len = -1.0f;
    const int asked = m_morphAsked.load(std::memory_order_acquire);
    if (asked != m_morphSeen) { m_morphSeen = asked; len = m_morphSeconds.load(std::memory_order_relaxed); }
    if (ab != m_abWas || (ab && slotsChanged)) {
        if (len < 0.0f) len = std::max(param(ParamIdE::PRESET_GLIDE), 1.0f);
        m_abWas = ab;
    }
    if (map != m_mapWas) {   // the map taking over or letting go glides too
        if (len < 0.0f) len = std::max(param(ParamIdE::MAP_GLIDE), 1.0f);
        m_mapWas = map;
    }
    if (len > 0.0f && m_soundValid) {
        m_from = m_sound;
        m_morphT = 0.0f;
        m_morphLen = len;
    }
    if (m_morphT < 1.0f) {
        m_morphT = std::min(1.0f, m_morphT + static_cast<float>(dt) / m_morphLen);
        if (m_morphT < 1.0f) blendMorph(m_from, m_target, morphCurve(m_morphT), m_sound);
        else m_sound = m_target;
    } else if (map) {
        // The map's cursor moves (a hand, a route, automation): the sound follows it, about 95 %
        // of the way in Map Glide.
        const float coef = 1.0f - std::exp(-static_cast<float>(dt) / std::max(param(ParamIdE::MAP_GLIDE) / 3.0f, 0.01f));
        approachMorph(m_sound, m_target, coef, m_from);   // m_from is free outside a glide
        m_sound = m_from;
    } else {
        m_sound = m_target;
    }
    m_soundValid = true;
    m_morphProgress.store(m_morphT, std::memory_order_relaxed);
    for (int i = 0; i < NumParams; ++i) m_soundLive[static_cast<size_t>(i)].store(m_sound.m_values[i], std::memory_order_relaxed);
    m_soundMix.store(m_sound.m_filterMix, std::memory_order_relaxed);
}

void Engine::runOwnModulators(float dt)
{
    bool read[MorphSetups] = {};
    float weight[MorphSetups] = {};
    for (int r = 0; r < m_sound.m_rowCount; ++r) {
        const ModRow& row = m_sound.m_rows[r];
        if ((isOwnSource(row.m_source) || isOwnSource(row.m_via)) && row.m_setup >= 0 && row.m_setup < m_sound.m_setupCount) {
            read[row.m_setup] = true;
            weight[row.m_setup] += std::fabs(row.m_depth);
        }
    }
    bool wanted[OwnUnits] = {};
    int unmatched[MorphSetups];
    int pending = 0, strongest = -1;
    float strongestWeight = -1.0f;
    for (int k = 0; k < m_sound.m_setupCount; ++k) {
        m_ownOf[k] = -1;
        if (!read[k]) continue;
        for (int u = 0; u < OwnUnits; ++u)
            if (m_own[static_cast<size_t>(u)].m_live && !wanted[u] && m_own[static_cast<size_t>(u)].m_setup == m_sound.m_setups[k]) {
                m_ownOf[k] = u;
                wanted[u] = true;
                break;
            }
        if (m_ownOf[k] < 0) unmatched[pending++] = k;
        else if (weight[k] > strongestWeight) { strongestWeight = weight[k]; strongest = m_ownOf[k]; }
    }
    for (int j = 0; j < pending; ++j) {
        const int k = unmatched[j];
        int u = -1;
        for (int v = 0; v < OwnUnits && u < 0; ++v)   // one no row reads any more runs on with the new settings
            if (m_own[static_cast<size_t>(v)].m_live && !wanted[v]) u = v;
        if (u < 0) {
            for (int v = 0; v < OwnUnits && u < 0; ++v)
                if (!m_own[static_cast<size_t>(v)].m_live) u = v;
            if (u < 0) continue;   // all running: its rows read the panel's (rare: a blend of blends of many sounds)
            m_own[static_cast<size_t>(u)].m_mods = strongest >= 0 ? m_own[static_cast<size_t>(strongest)].m_mods : m_mods.own();
            m_own[static_cast<size_t>(u)].m_live = true;
        }
        m_own[static_cast<size_t>(u)].m_setup = m_sound.m_setups[k];
        m_ownOf[k] = u;
        wanted[u] = true;
    }
    for (int u = 0; u < OwnUnits; ++u) {
        OwnUnit& o = m_own[static_cast<size_t>(u)];
        if (!wanted[u]) { o.m_live = false; continue; }
        o.m_mods.advance(dt, m_mods.envTime(), o.m_setup);
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
            in.m_lfoShape[i] = static_cast<int>(std::lround(base(static_cast<ParamIdE>(static_cast<int>(ParamIdE::LFO1_SHAPE) + 2 * i))));
            in.m_lfoRate[i] = base(static_cast<ParamIdE>(static_cast<int>(ParamIdE::LFO1_RATE) + 2 * i));
        }
        for (int e = 0; e < 2; ++e) {
            in.m_envShape[e] = static_cast<int>(std::lround(base(static_cast<ParamIdE>(static_cast<int>(ParamIdE::ENV1_SHAPE) + 2 * e))));
            in.m_envRate[e] = base(static_cast<ParamIdE>(static_cast<int>(ParamIdE::ENV1_RATE) + 2 * e));
        }
        for (int m = 0; m < 4; ++m) in.m_macro[m] = base(static_cast<ParamIdE>(static_cast<int>(ParamIdE::MACRO1) + m));
        in.m_wheel = m_wheel.load(std::memory_order_relaxed);
        in.m_pressure = m_pressure.load(std::memory_order_relaxed);
        in.m_velocity = m_lastVelocity;
        in.m_phraseStart = m_phraseStart;
        in.m_coherence = base(ParamIdE::COHERENCE);
        in.m_coherenceRate = base(ParamIdE::COHERENCE_RATE);
        in.m_leniaRate = base(ParamIdE::LENIA_RATE);
        in.m_leniaGrowth = base(ParamIdE::LENIA_GROWTH);
        in.m_chaosPeriod = base(ParamIdE::CHAOS_PERIOD);
        in.m_excitation = static_cast<float>(m_brain.excitation());
        in.m_rootAge = static_cast<float>(m_brain.rootAgeSeconds());
        in.m_randomNote = m_randomNote;
        {
            // The Beat (Noctuary's): the two lowest distinct pitches, the simplest ratio near their
            // interval, and the difference of the harmonics that would meet if it were just.
            double f1 = 0.0, f2 = 0.0;
            for (const Voice& v : m_voices)
                if (v.m_note >= 0 && v.m_lastGain > 0.0f && (f1 <= 0.0 || v.m_hz < f1)) f1 = v.m_hz;
            if (f1 > 0.0)
                for (const Voice& v : m_voices)
                    if (v.m_note >= 0 && v.m_lastGain > 0.0f && v.m_hz / f1 > 1.003 && (f2 <= 0.0 || v.m_hz < f2)) f2 = v.m_hz;
            float want = 0.0f;
            if (f1 > 0.0 && f2 > 0.0) {
                static const int Ratios[][2] = { {1,1}, {6,5}, {5,4}, {4,3}, {3,2}, {8,5}, {5,3}, {7,4}, {2,1}, {5,2}, {3,1}, {4,1} };
                const double r = f2 / f1;
                double bestErr = 1e30;
                int bp = 1, bq = 1;
                for (const auto& pq : Ratios) {
                    const double err = std::fabs(std::log(r / (static_cast<double>(pq[0]) / pq[1])));
                    if (err < bestErr) { bestErr = err; bp = pq[0]; bq = pq[1]; }
                }
                want = static_cast<float>(std::fabs(static_cast<double>(bq) * f2 - static_cast<double>(bp) * f1));
                if (!(want == want) || want > 30.0f) want = 30.0f;
            }
            in.m_beatHz = want;
            // Note and Distance: the loudest voice's, as Noctuary reads its per-voice sources for the matrix.
            const Voice* loud = nullptr;
            for (const Voice& v : m_voices)
                if (v.m_note >= 0 && (loud == nullptr || v.m_lastGain > loud->m_lastGain)) loud = &v;
            in.m_note = loud ? (static_cast<float>(loud->m_note) - 24.0f) / 84.0f : 0.5f;
            in.m_distance = loud ? loud->m_distance : 0.5f;
            // What was heard in the last block (the GPU's), and the chord's roughness: Sethares'
            // dissonance curve (after Plomp and Levelt) over the first eight harmonics of every pair of
            // sounding voices, weighted by their loudness and 1/h, over its own weight -- every eighth
            // block (up to 15,000 exponentials; the source is smoothed over a second anyway).
            if (m_prepared) m_cloud.heard(in.m_heardDb, in.m_heardCentroid, in.m_heardFlux);
            double rough = 0.0, weight = 0.0;
            const bool roughNow = --m_roughTick <= 0;
            if (roughNow) m_roughTick = 8;
            for (size_t a = 0; roughNow && a < m_voices.size(); ++a) {
                const Voice& va = m_voices[a];
                if (va.m_note < 0 || va.m_lastGain <= 0.0f) continue;
                for (size_t b = a + 1; b < m_voices.size(); ++b) {
                    const Voice& vb = m_voices[b];
                    if (vb.m_note < 0 || vb.m_lastGain <= 0.0f) continue;
                    for (int ha = 1; ha <= 8; ++ha)
                        for (int hb = 1; hb <= 8; ++hb) {
                            const double fa = va.m_hz * partialRatio(ha, m_blk.m_inharmonicB, m_blk.m_stretch) + m_blk.m_shift;
                            const double fb = vb.m_hz * partialRatio(hb, m_blk.m_inharmonicB, m_blk.m_stretch) + m_blk.m_shift;
                            const double s = 0.24 / (0.021 * std::min(fa, fb) + 19.0), d = std::fabs(fb - fa) * s;
                            const double w = static_cast<double>(va.m_lastGain) * vb.m_lastGain / (ha * hb);
                            rough += w * (std::exp(-3.5 * d) - std::exp(-5.75 * d));
                            weight += w;
                        }
                }
            }
            // Scaled so that a minor second, the roughest common interval, reads about 0.55 and a pure
            // fifth under 0.1 (most of the 64 pairs lie far apart and add nothing to the mean).
            if (roughNow) m_roughness = weight > 1.0e-12 ? static_cast<float>(16.0 * rough / weight) : 0.0f;
            in.m_roughness = m_roughness;
            if (m_prepared) m_cloud.reservoirOut(in.m_reservoir);
        }
        m_phraseStart = false;
        m_mods.advance(static_cast<float>(m_cfg.m_gpuBlock / m_cfg.m_sampleRate), in);
        runOwnModulators(static_cast<float>(m_cfg.m_gpuBlock / m_cfg.m_sampleRate));
        for (int s = 1; s < NumModSources; ++s)
            m_sourceLive[static_cast<size_t>(s)].store(m_mods.value(static_cast<ModSourceE>(s)), std::memory_order_relaxed);
        // A source as a row reads it: an LFO or envelope from the modulators of the sound the row came with.
        auto source = [this](ModSourceE s, int setup) {
            if (isOwnSource(s) && setup >= 0 && setup < m_sound.m_setupCount && m_ownOf[setup] >= 0)
                return m_own[static_cast<size_t>(m_ownOf[setup])].m_mods.value(s);
            return m_mods.value(s);
        };
        float offset[NumParams] = {};
        bool modded[NumParams] = {};
        for (int r = 0; r < m_sound.m_rowCount; ++r) {   // the matrix where the sound stands (a glide weighs two)
            const ModRow& row = m_sound.m_rows[r];
            if (row.m_source == ModSourceE::OFF || row.m_target < 0 || row.m_target >= NumParams || row.m_depth == 0.0f) continue;
            const float amount = row.m_via == ModSourceE::OFF ? 1.0f : viaAmount(row.m_via, source(row.m_via, row.m_setup));
            offset[row.m_target] += row.m_depth * amount * source(row.m_source, row.m_setup);
            modded[row.m_target] = true;
        }
        // The macros taught by example (Focus.h), each at its macro's value -- like rows, before the gates.
        {
            std::unique_lock<std::mutex> lock(m_lessonLock, std::try_to_lock);
            if (lock.owns_lock())
                for (size_t m = 0; m < 4; ++m)
                    if (m_lessonPendingSet[m]) {
                        m_lessons[m] = m_lessonsPending[m];
                        m_lessonPendingSet[m] = false;
                    }
        }
        for (size_t m = 0; m < 4; ++m)
            if (m_lessons[m].m_count >= 2) m_lessons[m].apply(in.m_macro[m], offset, modded);
        // A gated amount is heard by its gate, the matrix's push too: nothing where its kind changes.
        for (int g = 0; g < MorphGates; ++g) offset[static_cast<int>(GatedAmounts[g])] *= m_sound.m_gate[g];
        m_cutoffPush = offset[static_cast<int>(ParamIdE::CUTOFF)];
        m_resonancePush = offset[static_cast<int>(ParamIdE::RESONANCE)];
        for (int i = 0; i < NumParams; ++i) {
            const ParamDesc& d = paramTable()[static_cast<size_t>(i)];
            const float base = this->base(d.m_id);
            if (!modded[i]) {
                // Whole numbers glide unrounded while the map's cursor is followed: rounded here.
                m_cur[static_cast<size_t>(i)] = d.m_kind == ParamKindE::INT ? std::round(base) : base;
                m_played[static_cast<size_t>(i)].store(std::numeric_limits<float>::quiet_NaN(), std::memory_order_relaxed);
                continue;
            }
            const float n = std::clamp(toNorm(d, base) + offset[i], 0.0f, 1.0f);
            float v = fromNorm(d, n);
            if (d.m_kind == ParamKindE::INT) v = std::round(v);
            m_cur[static_cast<size_t>(i)] = v;
            m_played[static_cast<size_t>(i)].store(n, std::memory_order_relaxed);
        }
    }
    // The family jam's energy: a follower opens and closes with the leader's section, by up to an
    // octave of cutoff and 40 % of brightness (at 0.5, nothing).
    {
        const float e = std::clamp(m_jamEnergy.load(std::memory_order_relaxed), 0.0f, 1.0f);
        m_jamCutoff = e != 0.5f ? std::exp2(2.0f * (e - 0.5f)) : 1.0f;
        if (e != 0.5f) {
            m_cur[static_cast<size_t>(ParamIdE::BRIGHTNESS)] = std::clamp(m_cur[static_cast<size_t>(ParamIdE::BRIGHTNESS)] * (0.6f + 0.8f * e), 0.0f, 1.0f);
            m_cur[static_cast<size_t>(ParamIdE::CUTOFF)] = std::clamp(m_cur[static_cast<size_t>(ParamIdE::CUTOFF)] * std::exp2(2.0f * (e - 0.5f)), 40.0f, 18000.0f);
        }
    }
    auto g = [this](ParamIdE id) { return cur(id); };
    m_blk.m_master = dbToGain(g(ParamIdE::MASTER_GAIN));
    m_blk.m_partials = static_cast<int>(std::lround(g(ParamIdE::PARTIALS)));
    m_blk.m_tilt = g(ParamIdE::TILT);
    m_blk.m_brightness = g(ParamIdE::BRIGHTNESS);
    m_blk.m_oddEven = g(ParamIdE::ODD_EVEN);
    m_blk.m_inharmonicB = inharmonicB(g(ParamIdE::INHARMONIC));
    m_blk.m_stretch = g(ParamIdE::STRETCH) / 1200.0f;
    m_blk.m_shift = g(ParamIdE::SHIFT);
    m_blk.m_purity = g(ParamIdE::PURITY);
    m_blk.m_width = g(ParamIdE::CLOUD_WIDTH) * (1.0f - PurityNarrowing * m_blk.m_purity);
    m_blk.m_widthSlope = g(ParamIdE::WIDTH_SLOPE);
    m_blk.m_drift = g(ParamIdE::DRIFT);
    m_blk.m_driftRate = g(ParamIdE::DRIFT_RATE);
    m_blk.m_shimmer = g(ParamIdE::SHIMMER);
    m_blk.m_shimmerRate = g(ParamIdE::SHIMMER_RATE);
    m_blk.m_stereo = g(ParamIdE::STEREO);
    m_blk.m_air = g(ParamIdE::AIR);
    m_blk.m_airCenter = g(ParamIdE::AIR_CENTER);
    m_blk.m_airWidth = g(ParamIdE::AIR_WIDTH);
    m_blk.m_airJitter = g(ParamIdE::AIR_JITTER);
    m_blk.m_filterModel = static_cast<int>(std::lround(g(ParamIdE::FILTER_MODEL)));
    m_blk.m_cutoff = g(ParamIdE::CUTOFF);
    m_blk.m_resonance = g(ParamIdE::RESONANCE);
    m_blk.m_filterEnv = g(ParamIdE::FILTER_ENV);
    m_blk.m_keyTrack = g(ParamIdE::KEY_TRACK);
    m_blk.m_filterDrift = g(ParamIdE::FILTER_DRIFT);
    m_blk.m_motion = static_cast<int>(std::lround(g(ParamIdE::MOTION)));
    m_blk.m_motionRate = g(ParamIdE::MOTION_RATE);
    m_blk.m_motionDepth = g(ParamIdE::MOTION_DEPTH);
    m_blk.m_attack = g(ParamIdE::ATTACK);
    m_blk.m_decay = g(ParamIdE::DECAY);
    m_blk.m_sustain = g(ParamIdE::SUSTAIN);
    m_blk.m_release = g(ParamIdE::RELEASE);
    m_blk.m_refPitch = g(ParamIdE::REF_PITCH);
    m_blk.m_tide = g(ParamIdE::TIDE);
    m_blk.m_tidePeriod = g(ParamIdE::TIDE_PERIOD);
    m_blk.m_depth = g(ParamIdE::DEPTH);
    m_blk.m_keysDepth = g(ParamIdE::KEYS_DEPTH);
    m_blk.m_farLevel = g(ParamIdE::FAR_LEVEL);
    m_blk.m_presence = g(ParamIdE::PRESENCE);
    m_blk.m_padLowCut = g(ParamIdE::PAD_LOW_CUT);
    m_blk.m_breath = g(ParamIdE::BREATH);
    m_blk.m_breathRate = g(ParamIdE::BREATH_RATE);
    m_blk.m_doppler = g(ParamIdE::DOPPLER);
    m_blk.m_brain2 = g(ParamIdE::BRAIN2_ON) >= 0.5f;
    m_blk.m_brain2Interval = static_cast<int>(std::lround(g(ParamIdE::BRAIN2_INTERVAL)));
    m_blk.m_brain2Depth = g(ParamIdE::BRAIN2_DEPTH);

    m_bp.m_on = g(ParamIdE::BRAIN_ON) >= 0.5f;
    m_bp.m_mode = static_cast<BrainModeE>(std::clamp(static_cast<int>(std::lround(g(ParamIdE::AUTO_MODE))), 0, NumBrainModes - 1));
    m_bp.m_density = static_cast<int>(std::lround(g(ParamIdE::BRAIN_DENSITY)));
    m_bp.m_rateSeconds = m_bp.m_mode == BrainModeE::CHORDS ? g(ParamIdE::AUTO_RATE) : g(ParamIdE::BRAIN_RATE);
    m_bp.m_holdMin = g(ParamIdE::BRAIN_HOLD_MIN);
    m_bp.m_cascade = g(ParamIdE::BRAIN_CASCADE);
    m_bp.m_holdMax = g(ParamIdE::BRAIN_HOLD_MAX);
    m_bp.m_low = static_cast<int>(std::lround(g(ParamIdE::BRAIN_LOW)));
    m_bp.m_high = static_cast<int>(std::lround(g(ParamIdE::BRAIN_HIGH)));
    m_bp.m_consonance = g(ParamIdE::BRAIN_CONSONANCE);
    m_bp.m_wander = g(ParamIdE::BRAIN_WANDER);
    m_bp.m_voiceLead = g(ParamIdE::AUTO_LEAD);
    m_bp.m_chordTension = g(ParamIdE::AUTO_TENSION);
    m_bp.m_rootMove = g(ParamIdE::AUTO_ROOT_MOVE);
    m_bp.m_timbre = g(ParamIdE::BRAIN_TIMBRE);
    m_bp.m_spectrum = &m_brainSpec;

    // The second conductor: the first one's knobs, its own pace, density and register (its root
    // follows the first one's plus the interval), always in Free -- a slow background.
    m_bp2 = m_bp;
    m_bp2.m_on = m_blk.m_brain2;
    m_bp2.m_mode = BrainModeE::FREE;
    m_bp2.m_density = static_cast<int>(std::lround(g(ParamIdE::BRAIN2_DENSITY)));
    m_bp2.m_rateSeconds = g(ParamIdE::BRAIN2_RATE);
    m_bp2.m_holdMin = 2.0f * m_bp.m_holdMin;
    m_bp2.m_holdMax = 2.0f * m_bp.m_holdMax;
    m_bp2.m_low = std::clamp(m_bp.m_low + m_blk.m_brain2Interval, 12, 115);
    m_bp2.m_high = std::clamp(m_bp.m_high + m_blk.m_brain2Interval, m_bp2.m_low, 120);

    // The foundation: a voice on the root in the octave from C2 up (from Lowest when the register
    // begins lower); the chord is drawn from a fifth above it (an octave of room at the least), so
    // nothing crowds or doubles it there.
    m_foundation = std::clamp(g(ParamIdE::BRAIN_FOUNDATION), 0.0f, 1.0f);
    m_foundationLow = std::min({ m_bp.m_low, m_bp.m_high, 36 });
    if (m_foundation > 0.0f && m_bp.m_on) {
        const int hi = std::max(m_bp.m_low, m_bp.m_high);
        const int f = foundationNote(m_brain.root(), m_foundationLow);
        m_bp.m_low = std::max(std::min(m_bp.m_low, hi), std::min(f + 7, hi - 12));
        m_bp.m_high = hi;
    }
}

void Engine::updateFoundation()
{
    const int want = m_bp.m_on && m_foundation > 0.0f ? foundationNote(m_brain.root(), m_foundationLow) : -1;
    if (m_foundationNote >= 0 && want == m_foundationNote) {
        // Still sounding? Its voice may have been taken when every voice was busy.
        bool held = false;
        for (const Voice& v : m_voices)
            if (v.m_source == FoundationSource && v.m_note == m_foundationNote && !v.m_released) held = true;
        if (held) return;
        for (const Voice& v : m_voices)
            if (v.m_stealing && v.m_nextSource == FoundationSource && v.m_next.m_note == m_foundationNote) held = true;
        if (held) return;
        m_foundationNote = -1;
    }
    if (want == m_foundationNote) return;
    if (m_foundationNote >= 0) stopNote(m_foundationNote, FoundationSource);
    if (want >= 0) startNote(want, m_foundation, m_blk.m_depth * 0.15f, FoundationSource, 0);   // close, as the conductor's near notes: the floor is under you
    m_foundationNote = want;
}

void Engine::refreshSpectrum()
{
    const float key[5] = { m_blk.m_tilt, m_blk.m_oddEven, m_blk.m_brightness, m_blk.m_inharmonicB, m_blk.m_stretch };
    if (std::equal(std::begin(key), std::end(key), std::begin(m_specKey))) return;
    std::copy(std::begin(key), std::end(key), std::begin(m_specKey));
    // The conductors judge timbre against the voice's own harmonic spectrum, both stretches included
    // (so the Timbre scale of a stretched spectrum is the stretched scale that makes it consonant).
    m_brainSpec.m_count = BrainSpectrum::Max;
    for (int h = 1; h <= BrainSpectrum::Max; ++h) {
        m_brainSpec.m_ratio[h - 1] = static_cast<double>(partialRatio(h, m_blk.m_inharmonicB, m_blk.m_stretch));
        m_brainSpec.m_amp[h - 1] = harmonicShape(h, m_blk.m_tilt, m_blk.m_oddEven, m_blk.m_brightness);
    }
    if (m_scaleChoice == TimbreScaleIndex) m_scaleChoice = -1;   // the Timbre scale follows the spectrum
}

void Engine::refreshScale()
{
    const int choice = static_cast<int>(std::lround(base(ParamIdE::SCALE)));
    if (choice != m_scaleChoice) {
        FixedScale s;
        // The Timbre scale (Sethares) is the one a timbre asks for: here the cloud's own spectrum.
        const bool built = choice == TimbreScaleIndex ? makeTimbreScale(m_brainSpec, s) : makeBuiltinScale(choice, s);
        if (built) m_scale = s;
        else if (m_scaleChoice < 0) makeBuiltinScale(0, m_scale);   // the Scala slot comes with the editor: 12-TET until then
        m_scaleChoice = choice;
    }
    m_rootPc = std::clamp(static_cast<int>(std::lround(base(ParamIdE::ROOT_NOTE))), 0, 11);
    m_snapKeys = std::lround(base(ParamIdE::KEY_MAP)) == 0;
}

void Engine::setUserLandscape(const Landscape& l)
{
    if (l.m_columns < 2 || l.m_columns > LandMaxColumns || l.m_rows < 1 || l.m_amps.size() < static_cast<size_t>(l.m_columns) * l.m_rows) return;
    Landscape fit = l;
    if (fit.m_rows > LandMaxRows) {   // a long recording: rows averaged in groups, the loop kept
        const int k = (fit.m_rows + LandMaxRows - 1) / LandMaxRows, rows = fit.m_rows / k;
        std::vector<float> a(static_cast<size_t>(rows) * fit.m_columns, 0.0f);
        for (int r = 0; r < rows; ++r)
            for (int j = 0; j < k; ++j)
                for (int c = 0; c < fit.m_columns; ++c) a[static_cast<size_t>(r) * fit.m_columns + c] += fit.m_amps[static_cast<size_t>(r * k + j) * fit.m_columns + c] / k;
        fit.m_amps.swap(a);
        fit.m_rowSeconds *= static_cast<float>(k);
        fit.m_rows = rows;
    }
    const float rms = landscapeRms(fit);
    {
        const std::lock_guard<std::mutex> lock(m_userLandLock);
        m_userLand = std::move(fit);
        m_userLandRms = rms;
    }
    m_userLandVersion.fetch_add(1, std::memory_order_release);
}

void Engine::refreshLandscape()
{
    int choice = std::clamp(static_cast<int>(std::lround(base(ParamIdE::LANDSCAPE))), 0, NumLandscapeChoices - 1);
    const int version = m_userLandVersion.load(std::memory_order_acquire);
    if (choice == UserLandscape && version == 0) choice = 0;   // none of the player's yet: Aurora
    if (choice == UserLandscape) {
        if (m_landChoice == choice && version == m_userLandSeen) return;
        if (!m_userLandLock.try_lock()) return;   // the editor is handing one over: the next block
        m_cloud.setLandscape(m_userLand.m_amps.data(), m_userLand.m_columns, m_userLand.m_rows, m_userLand.m_rowSeconds);
        m_landRms = m_userLandRms;
        m_landLoop = static_cast<float>(m_userLand.m_rows) * m_userLand.m_rowSeconds;
        m_userLandLock.unlock();
        m_landChoice = choice;
        m_userLandSeen = version;
        return;
    }
    if (choice == m_landChoice) return;
    const FactoryLandscapes& f = factoryLandscapes();
    const Landscape& l = f.m_land[static_cast<size_t>(choice)];
    m_cloud.setLandscape(l.m_amps.data(), l.m_columns, l.m_rows, l.m_rowSeconds);
    m_landRms = f.m_rms[static_cast<size_t>(choice)];
    m_landLoop = static_cast<float>(l.m_rows) * l.m_rowSeconds;
    m_landChoice = choice;
}

void Engine::updateLight()
{
    const float amount = cur(ParamIdE::BRAIN_LIGHT);
    if (amount <= 0.0f) {
        m_lightOn = false;
        return;
    }
    if (m_lightOn && --m_lightCountdown > 0) return;
    m_lightCountdown = 8;
    // The light over the spectrum's axis (20 Hz .. 20 kHz, log, as the partials read it), 128 points.
    constexpr int Points = 128;
    float profile[Points];
    bool have = false;
    const int lifeMode = std::clamp(static_cast<int>(std::lround(cur(ParamIdE::LIFE_MODE))), 0, NumLifeModes - 1);
    const float lifeDepth = cur(ParamIdE::LIFE_DEPTH);
    if (lifeMode != 0 && lifeDepth > 0.0f) {
        if (const float* cells = m_cloud.lifePicture()) {   // the living field: its columns, averaged along
            float column[LifeW];
            for (int x = 0; x < LifeW; ++x) {
                float s = 0.0f;
                for (int y = 0; y < LifeH; ++y) s += cells[static_cast<size_t>(y) * LifeW + x];
                column[x] = s / static_cast<float>(LifeH);
            }
            const float zoom = cur(ParamIdE::LIFE_ZOOM);
            for (int i = 0; i < Points; ++i) {
                float x = (static_cast<float>(i) + 0.5f) / Points * zoom * static_cast<float>(LifeW);
                x -= std::floor(x / static_cast<float>(LifeW)) * static_cast<float>(LifeW);
                profile[i] = 1.0f - lifeDepth + lifeDepth * column[std::min(static_cast<int>(x), LifeW - 1)];
            }
            have = true;
        }
    }
    const float motionDepth = cur(ParamIdE::MOTION_DEPTH);
    if (!have && motionUsesLandscape(static_cast<MotionKindE>(std::lround(cur(ParamIdE::MOTION)))) && motionDepth > 0.0f && m_landChoice >= 0) {
        // The landscape as a new note reads it: its first eight seconds at the motion's rate.
        auto readLand = [&](const Landscape& l) {
            if (l.m_columns < 2 || l.m_rows < 1) return;
            const int rows = std::clamp(static_cast<int>(std::lround(8.0f * cur(ParamIdE::MOTION_RATE) / std::max(l.m_rowSeconds, 0.001f))), 1, l.m_rows);
            for (int i = 0; i < Points; ++i) {
                const int c = std::min(static_cast<int>((static_cast<float>(i) + 0.5f) / Points * static_cast<float>(l.m_columns)), l.m_columns - 1);
                float s = 0.0f;
                for (int r = 0; r < rows; ++r) s += l.m_amps[static_cast<size_t>(r) * l.m_columns + c];
                profile[i] = 1.0f - motionDepth + motionDepth * s / static_cast<float>(rows);
            }
            have = true;
        };
        if (m_landChoice == UserLandscape) {
            if (m_userLandLock.try_lock()) {   // the editor handing one over: the light as it was
                readLand(m_userLand);
                m_userLandLock.unlock();
            } else if (m_lightOn) {
                return;
            }
        } else {
            readLand(factoryLandscapes().m_land[static_cast<size_t>(m_landChoice)]);
        }
    }
    if (!have) {
        m_lightOn = false;
        return;
    }
    // Per note: its harmonics' weight in the light (the voice's own spectrum), against the register's mean.
    float score[128];
    double mean = 0.0;
    int counted = 0;
    for (int n = 0; n < 128; ++n) {
        const double f0 = frequencyOf(n);
        double s = 0.0, w = 0.0;
        for (int h = 0; h < std::max(m_brainSpec.m_count, 1) && h < BrainSpectrum::Max; ++h) {
            const double fh = f0 * (m_brainSpec.m_count > 0 ? m_brainSpec.m_ratio[h] : static_cast<double>(h + 1));
            if (fh < LandLowHz || fh > 20000.0) continue;
            const double a = m_brainSpec.m_count > 0 ? m_brainSpec.m_amp[h] : 1.0 / (h + 1);
            const int i = std::clamp(static_cast<int>(std::log2(fh / LandLowHz) / 9.96578428 * Points), 0, Points - 1);
            s += a * profile[i];
            w += a;
        }
        score[n] = w > 0.0 ? static_cast<float>(s / w) : 0.0f;
        if (n >= m_bp.m_low && n <= m_bp.m_high) { mean += score[n]; ++counted; }
    }
    mean = counted > 0 ? mean / counted : 0.5;
    // The factor: the light's share against the register's, cubed at full Follow Light -- twice the
    // light, eight times as likely.
    for (int n = 0; n < 128; ++n)
        m_light[n] = std::clamp(static_cast<float>(std::pow(std::max(score[n], 0.01f) / std::max(mean, 0.01), 3.0 * amount)), 0.02f, 50.0f);
    m_lightOn = true;
}

double Engine::frequencyOf(int note) const
{
    // Purity blends 12-TET (0) with the scale (1) in the log domain, as in Noctuary.
    const double ref = m_blk.m_refPitch;
    const double pure = scaleFrequency(m_scale, note, 60 + m_rootPc, ref, m_snapKeys);
    const double et = ref * std::pow(2.0, (note - 69) / 12.0);
    const double p = std::clamp(static_cast<double>(m_blk.m_purity), 0.0, 1.0);
    return std::exp(std::log(et) + (std::log(pure) - std::log(et)) * p);
}

float Engine::brainDistance()
{
    // Noctuary's spatial model: the conductor's notes are bimodal, 40 % close, 60 % deep.
    const float u = m_rng.uniform();
    return m_rng.uniform() < 0.4f ? m_blk.m_depth * 0.15f * u : m_blk.m_depth * (0.55f + 0.45f * u);
}

void Engine::beginOn(int i, int note, float velocity, float distance, int source, int onset, int channel)
{
    Voice& v = m_voices[static_cast<size_t>(i)];
    m_lastVelocity = velocity;
    v.m_note = note;
    v.m_source = source;
    v.m_channel = source == 0 ? channel : -1;
    v.m_bend = v.m_channel >= 0 ? m_chanBend[static_cast<size_t>(v.m_channel)].load(std::memory_order_relaxed) : 0.0f;
    v.m_press = v.m_channel >= 0 ? m_chanPress[static_cast<size_t>(v.m_channel)].load(std::memory_order_relaxed) : 0.0f;
    v.m_slide = v.m_channel >= 0 ? m_chanSlide[static_cast<size_t>(v.m_channel)].load(std::memory_order_relaxed) : 0.5f;
    v.m_velocity = velocity;
    v.m_place = std::clamp(distance, 0.0f, 1.0f);
    v.m_distance = v.m_place;
    v.m_prevDistance = v.m_place;
    v.m_breath.init(m_side);
    v.m_cutoffDrift.init(m_side);
    v.m_env.kill();
    v.m_env.noteOn();
    v.m_seed = static_cast<std::uint32_t>(m_rng.next()) | 1u;
    v.m_restart = true;
    v.m_onset = onset;
    v.m_released = false;
    v.m_tailBlocks = 1;
    v.m_tailArmed = false;
    v.m_tuneCents = 0.0f;
    v.m_lastGain = 0.0f;
    m_randomNote = m_noteRng.bipolar();
    v.m_age = 0.0;
    v.m_stealing = false;
    v.m_stealBlocks = 0;
    m_cloud.voice(i).m_releaseAt = -1.0f;
}

void Engine::startNote(int note, float velocity, float distance, int source, int onset, int channel)
{
    const int count = static_cast<int>(m_voices.size());
    for (int i = 0; i < count; ++i)
        if (m_voices[static_cast<size_t>(i)].m_note < 0 && !m_voices[static_cast<size_t>(i)].m_stealing) { beginOn(i, note, velocity, distance, source, onset, channel); return; }
    // Every voice sounds: take the quietest released one, else the oldest. It fades to silence
    // over one block (a restart from full level would click) and the new note begins after it.
    int best = -1;
    float bestLevel = 2.0f;
    for (int i = 0; i < count; ++i) {
        const Voice& v = m_voices[static_cast<size_t>(i)];
        if (!v.m_stealing && v.m_released && v.m_env.level() < bestLevel) { bestLevel = v.m_env.level(); best = i; }
    }
    if (best < 0) {   // the oldest -- not the foundation, which is the oldest by design
        double oldest = -1.0;
        for (int i = 0; i < count; ++i) {
            const Voice& v = m_voices[static_cast<size_t>(i)];
            if (!v.m_stealing && v.m_source != FoundationSource && v.m_age > oldest) { oldest = v.m_age; best = i; }
        }
    }
    if (best < 0) {
        double oldest = -1.0;
        for (int i = 0; i < count; ++i) {
            const Voice& v = m_voices[static_cast<size_t>(i)];
            if (!v.m_stealing && v.m_age > oldest) { oldest = v.m_age; best = i; }
        }
    }
    if (best < 0) return;
    Voice& v = m_voices[static_cast<size_t>(best)];
    v.m_stealing = true;
    v.m_stealBlocks = 1;
    v.m_next = { 0, note, velocity, 0, channel };
    v.m_nextSource = source;
    v.m_nextDistance = distance;
}

void Engine::stopNote(int note, int source, int channel)
{
    for (size_t i = 0; i < m_voices.size(); ++i) {
        Voice& v = m_voices[i];
        const bool ours = channel < 0 || v.m_channel < 0 || v.m_channel == channel;
        if (v.m_stealing && v.m_next.m_type == 0 && v.m_next.m_note == note && v.m_nextSource == source && (channel < 0 || v.m_next.m_channel == channel)) { v.m_next.m_type = -1; return; }
        if (v.m_note == note && v.m_source == source && ours && !v.m_released && !v.m_stealing) {
            v.m_released = true;
            v.m_env.noteOff();
            CloudVoice& cv = m_cloud.voice(static_cast<int>(i));
            cv.m_releaseAt = cv.m_time;   // the motion's release tails start here
            if (cur(ParamIdE::MEMORY_LEVEL) > 0.0f && !m_ghosts.empty()) captureGhost(static_cast<int>(i));
            return;
        }
    }
}

void Engine::lifeSnapshot(std::vector<float>& cells)
{
    const std::lock_guard<std::mutex> lock(m_snapLock);
    cells = m_snapLife;
}

void Engine::setMacroLesson(int macro, const MacroLesson& lesson)
{
    if (macro < 0 || macro >= 4) return;
    const std::lock_guard<std::mutex> lock(m_lessonLock);
    m_lessonsPending[static_cast<size_t>(macro)] = lesson;
    m_lessonPendingSet[static_cast<size_t>(macro)] = true;
}

void Engine::rippleSnapshot(std::vector<float>& cells)
{
    const std::lock_guard<std::mutex> lock(m_snapLock);
    cells = m_snapRipple;
}

bool Engine::partialSnapshot(std::vector<float>& dots, int& perVoice)
{
    const std::lock_guard<std::mutex> lock(m_snapLock);
    if (m_snapSeq == m_snapSeen || m_snap.empty()) return false;
    m_snapSeen = m_snapSeq;
    dots = m_snap;
    perVoice = m_snapPer;
    return true;
}

bool Engine::hallMemory(std::vector<float>& out, double& age)
{
    const std::lock_guard<std::mutex> lock(m_snapLock);
    out.resize(static_cast<size_t>(HallBins));
    if (m_hallPendingSet) {
        std::copy(m_hallPending.begin(), m_hallPending.end(), out.begin());
        age = m_hallPendingAge;
    } else if (m_snapHallValid) {
        std::copy(m_snapHall.begin(), m_snapHall.end(), out.begin());
        age = m_snapHallAge;
    } else {
        std::fill(out.begin(), out.end(), 1.0f);
        age = 0.0;
    }
    return age > 0.0;
}

void Engine::setHallMemory(const std::vector<float>& in, double age)
{
    const std::lock_guard<std::mutex> lock(m_snapLock);
    const bool whole = in.size() == static_cast<size_t>(HallBins);
    for (int b = 0; b < HallBins; ++b) {
        const float m = whole ? in[static_cast<size_t>(b)] : 1.0f;
        m_hallPending[static_cast<size_t>(b)] = std::isfinite(m) ? std::clamp(m, 0.0f, 1024.0f) : 1.0f;
    }
    m_hallPendingAge = whole ? std::max(0.0, age) : 0.0;
    m_hallPendingSet = true;
}

void Engine::stepRipple(float dt)
{
    Ripple& R = m_cloud.ripple();
    R.m_depth = cur(ParamIdE::RIPPLE_DEPTH);
    const int strikes = m_rippleStrikeCount;
    m_rippleStrikeCount = 0;
    const int sounding = m_rippleVoices;
    m_rippleVoices = 0;
    if (R.m_depth <= 0.0f) {   // at rest: the string is let go, so it starts quiet when it is turned up
        std::fill(std::begin(m_rippleNow), std::end(m_rippleNow), 0.0f);
        std::fill(std::begin(m_ripplePrev), std::end(m_ripplePrev), 0.0f);
        std::fill(std::begin(R.m_u), std::end(R.m_u), 0.0f);
        R.m_makeup = 1.0f;
        return;
    }
    constexpr float CellsPerOctave = static_cast<float>(RippleCells - 1) / 9.96578428f;
    // Strikes: every note that started this block, at its first eight harmonics, the stronger the
    // lower the harmonic -- a pebble into the spectrum where the note lands. The shape is a Ricker
    // wavelet (a "Mexican hat", zero in sum) about a quarter octave wide, so a passing wave makes a
    // partial louder and then softer rather than louder on the whole; and it goes into both the now
    // and the step before, a displacement at rest, which parts into two halves running up and down
    // (a struck velocity would leave a standing plateau behind in one dimension: measured, 3 dB
    // louder where the notes are and no motion).
    for (int s = 0; s < strikes; ++s) {
        for (int h = 1; h <= 8; ++h) {
            const float pos = std::log2(std::max(m_rippleStrikes[s] * h, 20.0f) / 20.0f) * CellsPerOctave;
            for (int c = std::max(0, static_cast<int>(pos) - 24); c <= std::min(RippleCells - 1, static_cast<int>(pos) + 24); ++c) {
                const float d = (static_cast<float>(c) - pos) / 5.0f;
                // 6/h: a passing half-wave of the fundamental's strike takes the loudness from near silence
                // to twice (tanh saturates); at 2/h it moved a harmonic by a dB or two, too tame to hear.
                const float bump = (6.0f / static_cast<float>(h)) * (1.0f - d * d) * std::exp(-0.5f * d * d);
                m_rippleNow[c] += bump;
                m_ripplePrev[c] += bump;
            }
        }
    }
    // The damped wave, u_tt = (c^2 u_x)_x - g u_t, by leapfrog; the ends reflect.
    const float c = std::max(cur(ParamIdE::RIPPLE_SPEED), 0.01f) * CellsPerOctave;   // cells a second
    const float g = 2.0f / std::max(cur(ParamIdE::RIPPLE_DECAY), 0.1f);
    const int steps = std::max(1, static_cast<int>(std::ceil(c * dt / 0.9f)));
    const float h = dt / static_cast<float>(steps), r2 = (c * h) * (c * h), damp = g * h * 0.5f;
    // Bend: the wave slower where the spectrum is loud -- the sounding voices' first sixteen harmonics,
    // their amplitudes against the spectrum's mean, each over a sixth of an octave (as wide as a strike:
    // a narrower slow zone lets the wave leak out at once) -- up to a ninth of its speed, so a wave
    // lingers at the notes, gathers there and is partly thrown back, as light in a lens -- a fifth of its
    // speed where a lone note's fundamental is, up to a seventeenth. (A slower cell is as stable.)
    float r2c[RippleCells];
    std::fill(std::begin(r2c), std::end(r2c), r2);
    const float bend = cur(ParamIdE::RIPPLE_BEND);
    if (bend > 0.0f && sounding > 0) {
        float power[RippleCells] = {};
        float shape[16];
        for (int k = 1; k <= 16; ++k) shape[k - 1] = harmonicShape(k, m_blk.m_tilt, m_blk.m_oddEven, m_blk.m_brightness);
        double total = 0.0;
        for (int v = 0; v < sounding; ++v)
            for (int k = 1; k <= 16; ++k) {
                const float fk = m_rippleHz[v] * partialRatio(k, m_blk.m_inharmonicB, m_blk.m_stretch) + m_blk.m_shift;
                if (fk < 20.0f || fk > 20000.0f) continue;
                const float w = m_rippleAmp[v] * shape[k - 1];
                const float pos = std::log2(fk / 20.0f) * CellsPerOctave;
                for (int cc = std::max(0, static_cast<int>(pos) - 12); cc <= std::min(RippleCells - 1, static_cast<int>(pos) + 13); ++cc) {
                    const float d = (static_cast<float>(cc) - pos) / 4.0f;
                    power[cc] += w * std::exp(-0.5f * d * d);
                }
                total += w;
            }
        if (total > 1.0e-20) {
            const float mean = static_cast<float>(total * 10.0265 / RippleCells);   // a bump's sum, 4 sqrt(2 pi)
            for (int i = 0; i < RippleCells; ++i) {
                const float dense = std::clamp(std::log2(1.0f + power[i] / mean) - 1.0f, 0.0f, 4.0f);
                const float ci = 1.0f / (1.0f + 4.0f * bend * dense);
                r2c[i] = r2 * ci * ci;
            }
        }
    }
    // In the conservative form, u_tt = (c^2 u_x)_x: a wave entering a slow zone piles up there, as a
    // wave on a shelving beach, and leaves it no louder than it came (in c^2 u_xx it would leave louder).
    float next[RippleCells];
    for (int s = 0; s < steps; ++s) {
        for (int i = 0; i < RippleCells; ++i) {
            const int il = i > 0 ? i - 1 : 1, ir = i < RippleCells - 1 ? i + 1 : RippleCells - 2;
            const float kl = 0.5f * (r2c[i] + r2c[il]), kr = 0.5f * (r2c[i] + r2c[ir]);
            const float flow = kr * (m_rippleNow[ir] - m_rippleNow[i]) - kl * (m_rippleNow[i] - m_rippleNow[il]);
            next[i] = (2.0f * m_rippleNow[i] - (1.0f - damp) * m_ripplePrev[i] + flow) / (1.0f + damp);
        }
        std::copy(std::begin(m_rippleNow), std::end(m_rippleNow), std::begin(m_ripplePrev));
        std::copy(std::begin(next), std::end(next), std::begin(m_rippleNow));
    }
    double mean2 = 0.0;
    for (int i = 0; i < RippleCells; ++i) {
        if (!std::isfinite(m_rippleNow[i])) m_rippleNow[i] = 0.0f;
        R.m_u[i] = m_rippleNow[i];
        const double t = std::tanh(static_cast<double>(m_rippleNow[i]));
        mean2 += t * t;
    }
    R.m_makeup = static_cast<float>(1.0 / std::sqrt(1.0 + static_cast<double>(R.m_depth) * R.m_depth * mean2 / RippleCells));
}

void Engine::captureGhost(int i)
{
    size_t slot = 0;
    for (size_t g = 0; g < m_ghosts.size(); ++g) {
        if (!m_ghosts[g].m_on) { slot = g; break; }
        if (m_ghosts[g].m_env < m_ghosts[slot].m_env) slot = g;
    }
    Ghost& g = m_ghosts[slot];
    g.m_base = m_cloud.voice(i);
    g.m_base.m_releaseAt = -1.0f;   // a ghost is held: its motion loops on
    g.m_on = true;
    g.m_restart = true;
    g.m_seed = static_cast<std::uint32_t>(m_ghostRng.next()) | 1u;
    g.m_age = 0.0f;
    g.m_life = std::max(1.0f, cur(ParamIdE::MEMORY_TIME));
    g.m_bloom = std::max(0.05f, cur(ParamIdE::MEMORY_BLOOM));
    g.m_env = 0.0f;
}

void Engine::renderBlock()
{
    const NoDenormals noDenormals;
    LARGE_INTEGER qpcStart;
    QueryPerformanceCounter(&qpcStart);
    const int G = m_cfg.m_gpuBlock;
    const double dt = G / m_cfg.m_sampleRate;
    const float dtf = static_cast<float>(dt);
    const std::int64_t pos = m_renderPos.load(std::memory_order_relaxed);
    readParams();
    refreshSpectrum();
    refreshScale();
    refreshLandscape();
    {   // the scale for the panel: four octaves of keys folded into one over the root, the same pitch once
        const double ref = m_blk.m_refPitch;
        const int rootNote = 60 + m_rootPc;
        const double f0 = ref * std::pow(2.0, (rootNote - 69) / 12.0);
        float cents[48];
        int n = 0;
        for (int k = 0; k < 48 && f0 > 0.0; ++k) {
            const double hz = scaleFrequency(m_scale, rootNote - 12 + k, rootNote, ref, m_snapKeys);
            if (!(hz > 0.0) || !std::isfinite(hz)) continue;   // a scale that has no pitch there: nothing to draw (and no NaN for the sort)
            double c = std::fmod(1200.0 * std::log2(hz / f0), 1200.0);
            if (!std::isfinite(c)) continue;
            if (c < 0.0) c += 1200.0;
            if (c > 1199.5) c = 0.0;
            bool seen = false;
            for (int j = 0; j < n && !seen; ++j) seen = std::fabs(cents[j] - c) < 1.0;
            if (!seen) cents[n++] = static_cast<float>(c);
        }
        std::sort(cents, cents + n);
        for (int i = 0; i < n; ++i) m_viewScale[static_cast<size_t>(i)].store(cents[i], std::memory_order_relaxed);
        m_viewScaleCount.store(n, std::memory_order_relaxed);
        m_viewRootHz.store(static_cast<float>(f0), std::memory_order_relaxed);
        // Every key's pitch, anew only when what tunes it changed.
        const double key[5] = { static_cast<double>(m_scaleChoice), static_cast<double>(m_rootPc), ref, static_cast<double>(m_blk.m_purity), m_snapKeys ? 1.0 : 0.0 };
        if (!std::equal(std::begin(key), std::end(key), std::begin(m_noteHzKey))) {
            std::copy(std::begin(key), std::end(key), std::begin(m_noteHzKey));
            for (int k = 0; k < 128; ++k) m_viewNoteHz[static_cast<size_t>(k)].store(static_cast<float>(frequencyOf(k)), std::memory_order_relaxed);
        }
    }

    // 1. The played notes that are due, in the order they came, each on its own sample.
    drainEvents();
    for (size_t i = 0; i < m_pending.size();) {
        const Event e = m_pending[i];
        if (e.m_stamp >= pos + G) { ++i; continue; }
        m_pending.erase(m_pending.begin() + static_cast<std::ptrdiff_t>(i));
        if (e.m_stamp < pos) m_late.fetch_add(1, std::memory_order_relaxed);
        const int onset = static_cast<int>(std::clamp<std::int64_t>(e.m_stamp - pos, 0, G - 1));
        if (e.m_type == 0) startNote(e.m_note, e.m_velocity, m_blk.m_keysDepth, 0, onset, e.m_channel);
        else if (e.m_type == 1) stopNote(e.m_note, 0, e.m_channel);
        else
            for (const Voice& v : m_voices)
                if (v.m_note >= 0 && v.m_source == 0 && !v.m_released) stopNote(v.m_note, 0);
    }

    // 2. The conductors. Step fires on the rising edge of its switch, or from autoplayStep().
    const bool step = cur(ParamIdE::AUTO_STEP) >= 0.5f;
    if ((step && !m_stepWas) | m_stepAsked.exchange(false, std::memory_order_relaxed)) m_brain.requestStep();
    m_stepWas = step;
    if (m_fillAsked.exchange(false, std::memory_order_relaxed)) {
        if (m_bp.m_on) m_brain.requestFill();
        if (m_blk.m_brain2) m_brain2.requestFill();
    }
    updateLight();
    m_bp.m_light = m_lightOn ? m_light : nullptr;
    m_bp2.m_light = m_bp.m_light;
    auto freqOf = [this](int note) { return frequencyOf(note); };
    m_brain.update(dt, m_bp, -1, freqOf, [this](const BrainEvent& e) {
        if (e.m_type == BrainEvent::TypeE::NOTE_ON) startNote(e.m_note, e.m_velocity, brainDistance(), 1, 0);
        else stopNote(e.m_note, 1);
    });
    // The family jam: a follower's conductor keeps its root on the leader's pitch class, the nearest
    // note of it (Noctuary's way: its tuning stays its own).
    if (const int jr = m_jamRoot.load(std::memory_order_relaxed); jr >= 0 && ((m_brain.root() % 12) + 12) % 12 != jr) {
        const int r = m_brain.root();
        int best = r, bestDist = 99;
        for (int d = -6; d <= 6; ++d)
            if ((((r + d) % 12) + 12) % 12 == jr && std::abs(d) < bestDist) { bestDist = std::abs(d); best = r + d; }
        m_brain.setRoot(std::clamp(best, 0, 127));
    }
    updateFoundation();
    m_brain2.setRoot(std::clamp(m_brain.root() + m_blk.m_brain2Interval, 0, 127));
    m_brain2.update(dt, m_bp2, -1, freqOf, [this](const BrainEvent& e) {
        if (e.m_type == BrainEvent::TypeE::NOTE_ON) startNote(e.m_note, e.m_velocity, m_blk.m_brain2Depth * (0.85f + 0.15f * m_rng.uniform()), 2, 0);
        else stopNote(e.m_note, 2);
    });
    m_brainRoot.store(m_brain.root(), std::memory_order_relaxed);
    {   // the conductor's countdown and the foundation, for the panel
        const double next = m_brain.untilNext();
        if (next > m_conductorWas + 0.5) m_conductorSpan.store(static_cast<float>(next), std::memory_order_relaxed);
        m_conductorWas = next;
        m_conductorNext.store(static_cast<float>(next), std::memory_order_relaxed);
        m_foundationView.store(m_foundationNote, std::memory_order_relaxed);
    }

    // 3. The voices: envelope, place, frequency, colour, filter, air, motion.
    // While the sound glides, the tuning follows slowly: a new scale is reached by the partials
    // sliding into it over seconds (a tenth of the glide, at most six).
    const double glideSeconds = m_morphT < 1.0f ? std::clamp(0.1 * m_morphLen, GlideSeconds, 6.0) : GlideSeconds;
    const double follow = 1.0 - std::exp(-dt / glideSeconds);
    const double tideMul = std::pow(2.0, m_blk.m_tide * m_tide.update(dtf, 1.0f / (m_blk.m_tidePeriod * 60.0f), m_side) / 1200.0);
    // Self-tuning: where a harmonic of one voice nearly meets a harmonic of another (within Reach),
    // the two pull together -- weighted by their loudness, by a bell over the distance and by how low
    // the harmonics are (the beats of low ones are what is heard). Every voice moves half the way
    // its weighted mismatch asks for, at the speed Self-Tune sets (the other moves the other half),
    // and the anchor draws it back to its pitch in the scale. A chord in equal temperament settles
    // towards just intervals and gives way again at the next change; a note starts on the scale.
    {
        const float grav = cur(ParamIdE::TUNE_GRAVITY), reach = std::max(1.0f, cur(ParamIdE::TUNE_REACH)), anchor = cur(ParamIdE::TUNE_ANCHOR);
        const int nv = static_cast<int>(m_voices.size());
        if (grav > 0.0f) {
            float kLogH[9] = { 0.0f };   // log2 of where harmonics 1 .. 8 lie (both stretches)
            for (int h = 1; h <= 8; ++h) kLogH[h] = std::log2(partialRatio(h, m_blk.m_inharmonicB, m_blk.m_stretch));
            std::array<float, 64> lb{}, moved{};
            for (int a = 0; a < nv && a < 64; ++a) {
                const Voice& v = m_voices[static_cast<size_t>(a)];
                lb[static_cast<size_t>(a)] = v.m_note >= 0 ? static_cast<float>(std::log2(frequencyOf(v.m_note) * tideMul)) + v.m_tuneCents / 1200.0f : 0.0f;
            }
            const float rate = 1.0f - std::exp(-2.0f * grav * dtf);
            for (int a = 0; a < nv && a < 64; ++a) {
                const Voice& va = m_voices[static_cast<size_t>(a)];
                if (va.m_note < 0 || va.m_stealing || va.m_lastGain <= 0.0f) continue;
                double force = 0.0, wsum = 0.0;
                for (int b = 0; b < nv && b < 64; ++b) {
                    const Voice& vb = m_voices[static_cast<size_t>(b)];
                    if (b == a || vb.m_note < 0 || vb.m_stealing || vb.m_lastGain <= 0.0f) continue;
                    for (int ha = 1; ha <= 8; ++ha)
                        for (int hb = 1; hb <= 8; ++hb) {
                            const float d = 1200.0f * (lb[static_cast<size_t>(b)] + kLogH[hb] - lb[static_cast<size_t>(a)] - kLogH[ha]);
                            if (std::fabs(d) >= reach) continue;
                            const double w = static_cast<double>(va.m_lastGain) * vb.m_lastGain * std::exp(-(d / reach) * (d / reach)) / (ha * hb);
                            force += w * d;
                            wsum += w;
                        }
                }
                if (wsum > 1.0e-12) moved[static_cast<size_t>(a)] = static_cast<float>(0.5 * force / wsum) * rate;
            }
            const float back = std::exp(-2.0f * anchor * dtf);
            for (int a = 0; a < nv && a < 64; ++a) {
                Voice& v = m_voices[static_cast<size_t>(a)];
                v.m_tuneCents = std::clamp((v.m_tuneCents + moved[static_cast<size_t>(a)]) * back, -50.0f, 50.0f);
            }
        } else {
            for (Voice& v : m_voices) v.m_tuneCents *= std::exp(-2.0f * dtf);   // switched off: back to the scale
        }
    }
    const MotionKindE motion = static_cast<MotionKindE>(std::clamp(m_blk.m_motion, 0, NumMotions - 1));
    // A voice that reads the landscape is as much quieter as the landscape is dark: made up here
    // (at most 18 dB), so a dim landscape changes the motion, not the level.
    float landMakeup = 1.0f;
    if (motionUsesLandscape(motion)) landMakeup = 1.0f / std::max(0.125f, 1.0f - m_blk.m_motionDepth + m_blk.m_motionDepth * m_landRms);
    int sounding = 0;
    float chordGain = 0.0f;   // the conductor's voices' envelopes (not the foundation's), for the foundation's level
    int chordVoices = 0;
    int by[3] = { 0, 0, 0 };
    float srcW = 0.0f, srcD = 0.0f;   // the near room's source: the voices' distance, weighted by loudness
    for (int i = 0; i < static_cast<int>(m_voices.size()); ++i) {
        Voice& v = m_voices[static_cast<size_t>(i)];
        CloudVoice& cv = m_cloud.voice(i);
        if (v.m_stealing) {
            if (v.m_stealBlocks > 0) { --v.m_stealBlocks; cv.m_gain = 0.0f; v.m_env.kill(); ++sounding; continue; }
            v.m_stealing = false;
            if (v.m_next.m_type == 0) beginOn(i, v.m_next.m_note, v.m_next.m_velocity, v.m_nextDistance, v.m_nextSource, 0, v.m_next.m_channel);
            else v.m_note = -1;
        }
        if (v.m_note < 0) { cv.m_active = 0; continue; }
        v.m_env.setTimes(m_blk.m_attack, m_blk.m_decay, m_blk.m_sustain, m_blk.m_release);
        const float gain = v.m_env.process();
        v.m_lastGain = gain;
        if (v.m_source == 1 && !v.m_released) { chordGain += gain * gain; ++chordVoices; }   // the foundation matches these
        if (!v.m_env.isActive()) {
            // With dispersion the late partials sound on after the envelope: the voice stays until
            // the latest of them has heard the end of it.
            if (!v.m_tailArmed) {
                v.m_tailArmed = true;
                v.m_tailBlocks += static_cast<int>(std::ceil(std::fabs(cur(ParamIdE::DISPERSION)) * m_cfg.m_sampleRate / G));
            }
            if (v.m_tailBlocks > 0) { --v.m_tailBlocks; cv.m_gain = 0.0f; ++sounding; continue; }
            v.m_note = -1;
            cv.m_active = 0;
            continue;
        }
        // Breath: the voice's plane wanders (Noctuary's "the room breathes"); Doppler: coming closer
        // raises it, receding lowers it.
        v.m_distance = std::clamp(v.m_place + 0.35f * m_blk.m_breath * v.m_breath.update(dtf, m_blk.m_breathRate, m_side), 0.0f, 1.0f);
        const float velocity = (v.m_distance - v.m_prevDistance) / dtf * MetresPerUnit;
        const double doppler = v.m_restart ? 1.0 : 1.0 - static_cast<double>(m_blk.m_doppler * velocity / SpeedOfSound);
        v.m_prevDistance = v.m_distance;
        const double target = frequencyOf(v.m_note) * tideMul * std::exp2(static_cast<double>(v.m_tuneCents) / 1200.0);
        v.m_hz = v.m_restart ? target : v.m_hz * std::pow(target / v.m_hz, follow);
        // MPE: a played note's own bend, pressure and slide while it is held (kept as they were once it
        // is let go, so a channel used again moves only its new note), and the master's bend. Each of
        // pressure and slide moves its target: pressure from nothing up, slide either way about its rest.
        float mpeBend = 0.0f;
        float mpe[NumMpeTargets] = {};
        if (v.m_source == 0) {
            if (!v.m_released && v.m_channel >= 0) {
                v.m_bend = m_chanBend[static_cast<size_t>(v.m_channel)].load(std::memory_order_relaxed);
                v.m_press = m_chanPress[static_cast<size_t>(v.m_channel)].load(std::memory_order_relaxed);
                v.m_slide = m_chanSlide[static_cast<size_t>(v.m_channel)].load(std::memory_order_relaxed);
            }
            mpeBend = m_chanBend[16].load(std::memory_order_relaxed) + v.m_bend;
            if (v.m_channel >= 0) {
                mpe[std::clamp(static_cast<int>(std::lround(cur(ParamIdE::MPE_PRESS))), 0, NumMpeTargets - 1)] += cur(ParamIdE::MPE_PRESS_DEPTH) * v.m_press;
                mpe[std::clamp(static_cast<int>(std::lround(cur(ParamIdE::MPE_SLIDE))), 0, NumMpeTargets - 1)] += cur(ParamIdE::MPE_SLIDE_DEPTH) * (2.0f * v.m_slide - 1.0f);
            }
        }
        auto expr = [&mpe](MpeTargetE t) { return mpe[static_cast<int>(t)]; };
        // The filter's cutoff: key tracking, the envelope, its own drift of up to an octave, and MPE's up to three.
        const float oct = m_blk.m_keyTrack * static_cast<float>(v.m_note - 60) / 12.0f + m_blk.m_filterEnv * gain
                        + m_blk.m_filterDrift * v.m_cutoffDrift.update(dtf, 0.05f, m_side) + 3.0f * expr(MpeTargetE::FILTER);
        const float fc = std::clamp(m_blk.m_cutoff * std::exp2(oct), 20.0f, 20000.0f);
        const FilterModelE model = static_cast<FilterModelE>(std::clamp(m_blk.m_filterModel, 0, NumFilterModels - 1));
        const float q = filterQ(static_cast<int>(model), m_blk.m_resonance);   // Spectrum.h: the picture's curve reads the same

        // The foundation: as loud as the chord is (m_foundationMatch) -- a held voice beside voices that
        // spend half their lives fading in, or under a chord still arriving, would stand far over it --
        // and its knob on top.
        float level = VoiceLevel * (v.m_source == FoundationSource ? (0.6f + 0.4f * m_foundation) * m_foundationMatch
                                     : v.m_source != 0 ? 0.6f + 0.4f * v.m_velocity : 0.4f + 0.6f * v.m_velocity);
        cv.m_hz = static_cast<float>(v.m_hz * doppler * std::exp2(static_cast<double>(mpeBend) / 12.0));
        cv.m_level = level * landMakeup * std::exp2(expr(MpeTargetE::LEVEL));   // up to 6 dB
        cv.m_gain = gain;
        // MPE's Brightness: the tilt flattened (or steepened) by up to 0.6 and the window opened by up
        // to a half -- the window alone moves little under a falling spectrum.
        cv.m_tilt = std::max(0.0f, m_blk.m_tilt - 0.6f * expr(MpeTargetE::BRIGHTNESS));
        cv.m_brightness = std::clamp(m_blk.m_brightness + 0.5f * expr(MpeTargetE::BRIGHTNESS), 0.0f, 1.0f);
        cv.m_oddEven = m_blk.m_oddEven;
        cv.m_inharmonic = m_blk.m_inharmonicB;
        cv.m_stretch = m_blk.m_stretch;
        cv.m_shift = m_blk.m_shift;
        // Turbulence: the cascade's drive, by the cube of the note's strength (with how loud the
        // instrument is, below): a soft chord stays pure.
        const float strength = 0.4f + 0.6f * v.m_velocity;
        cv.m_turbulence = m_turbDrive * strength * strength * strength;
        cv.m_presenceDb = m_blk.m_presence;
        cv.m_lowCut = m_blk.m_padLowCut;
        cv.m_filterModel = static_cast<int>(model);
        cv.m_filterMix = m_sound.m_filterMix;   // for the panel; the card reads the filter list below
        cv.m_cutoff = fc;
        cv.m_filterQ = q;
        cv.m_vowel = 4.0f * std::clamp(std::log(fc / 40.0f) / std::log(18000.0f / 40.0f), 0.0f, 1.0f);
        // During a morph between different filters (Morph.h): the heaviest four of the sound's, each
        // with this voice's key tracking, envelope and drift and the matrix's push, transported into
        // each other on the card. One filter alone is the filter above.
        cv.m_filterCount = 0;
        if (m_sound.m_filterCount > 1) {
            int order[MorphFilters];
            const int n = m_sound.m_filterCount;
            for (int k = 0; k < n; ++k) order[k] = k;
            std::stable_sort(order, order + n, [this](int a, int b) { return m_sound.m_filters[a].m_weight > m_sound.m_filters[b].m_weight; });
            const int count = std::min(n, VoiceFilters);
            float sum = 0.0f;
            for (int k = 0; k < count; ++k) sum += m_sound.m_filters[order[k]].m_weight;
            const ParamDesc& cd = paramDesc(ParamIdE::CUTOFF);
            const ParamDesc& rd = paramDesc(ParamIdE::RESONANCE);
            for (int k = 0; k < count; ++k) {
                const MorphFilter& mf = m_sound.m_filters[order[k]];
                const FilterModelE m = static_cast<FilterModelE>(std::clamp(mf.m_model, 0, NumFilterModels - 1));
                const float base = fromNorm(cd, std::clamp(toNorm(cd, mf.m_cutoff) + m_cutoffPush, 0.0f, 1.0f)) * m_jamCutoff;
                const float fck = std::clamp(base * std::exp2(oct), 20.0f, 20000.0f);
                const float res = fromNorm(rd, std::clamp(toNorm(rd, mf.m_resonance) + m_resonancePush, 0.0f, 1.0f));
                const float qk = filterQ(static_cast<int>(m), res);
                cv.m_filterWeight[k] = mf.m_weight / std::max(sum, 1.0e-9f);
                cv.m_filterModels[k] = static_cast<float>(m);
                cv.m_filterCutoffs[k] = fck;
                cv.m_filterQs[k] = qk;
                cv.m_filterVowels[k] = 4.0f * std::clamp(std::log(fck / 40.0f) / std::log(18000.0f / 40.0f), 0.0f, 1.0f);
            }
            cv.m_filterCount = count;
        }
        cv.m_air = m_blk.m_air;
        cv.m_airCenter = m_blk.m_airCenter;
        cv.m_airWidth = m_blk.m_airWidth;
        cv.m_airJitter = m_blk.m_airJitter;
        cv.m_width = m_blk.m_width * std::exp2(1.5f * expr(MpeTargetE::WIDTH));
        cv.m_widthSlope = m_blk.m_widthSlope;
        cv.m_drift = m_blk.m_drift;
        cv.m_driftRate = m_blk.m_driftRate;
        cv.m_shimmer = std::clamp(m_blk.m_shimmer + 0.5f * expr(MpeTargetE::SHIMMER), 0.0f, 1.0f);
        cv.m_shimmerRate = m_blk.m_shimmerRate;
        cv.m_stereo = m_blk.m_stereo;
        cv.m_partials = m_blk.m_partials;
        cv.m_distance = std::clamp(v.m_distance - 0.5f * expr(MpeTargetE::NEARER), 0.0f, 1.0f);
        srcW += gain * level;
        srcD += gain * level * v.m_distance;
        if (m_rippleVoices < 64) {
            m_rippleHz[m_rippleVoices] = static_cast<float>(v.m_hz);
            m_rippleAmp[m_rippleVoices++] = gain * level;
        }
        cv.m_airAbsorb = 1.0f;
        cv.m_seed = v.m_seed;
        cv.m_active = 1;
        cv.m_restart = v.m_restart ? 1 : 0;
        if (v.m_restart && m_rippleStrikeCount < 64) m_rippleStrikes[m_rippleStrikeCount++] = static_cast<float>(v.m_hz);
        cv.m_onset = v.m_restart ? v.m_onset : 0;
        applyMotion(motion, cv, m_blk.m_motionRate, std::clamp(m_blk.m_motionDepth * (1.0f + expr(MpeTargetE::MOTION)), 0.0f, 1.0f));
        v.m_restart = false;
        v.m_onset = 0;
        v.m_age += dt;
        ++sounding;
        ++by[v.m_source == FoundationSource ? 1 : std::clamp(v.m_source, 0, 2)];   // the foundation is the conductor's
    }
    m_sounding.store(sounding, std::memory_order_relaxed);
    {   // the chord's mean envelope, followed over a few seconds (the foundation's own sustain divided out)
        // As loud as half the chord's voices together (their power), seen over the foundation's own
        // sustain: half a voice under a lone note, a little over one voice under five.
        const float target = chordVoices > 0 ? 0.5f * std::sqrt(chordGain) / std::max(m_blk.m_sustain, 0.2f) : 0.0f;
        m_foundationMatch += (std::min(target, 1.5f) - m_foundationMatch) * (1.0f - std::exp(-dtf / 4.0f));
    }
    if (srcW > 1.0e-6f) m_srcDistance = srcD / srcW;
    if (m_prevSounding == 0 && sounding > 0) m_phraseStart = true;   // a note into silence: the envelopes' phrase begins
    m_prevSounding = sounding;
    for (int s = 0; s < 3; ++s) m_soundingBy[s].store(by[s], std::memory_order_relaxed);
    // The memory's ghosts: each a released voice sounding on from where it was let go, rising over
    // Bloom, fading by 60 dB over Memory Time, and as it fades widening, drifting, darkening and
    // receding (Blur, Sink). New phases of its own, so it never adds to the voice it came from.
    {
        const float level = cur(ParamIdE::MEMORY_LEVEL), blur = cur(ParamIdE::MEMORY_BLUR), sink = cur(ParamIdE::MEMORY_SINK);
        const int first = static_cast<int>(m_voices.size());
        for (size_t gi = 0; gi < m_ghosts.size(); ++gi) {
            Ghost& g = m_ghosts[gi];
            CloudVoice& cv = m_cloud.voice(first + static_cast<int>(gi));
            if (!g.m_on) { cv.m_active = 0; continue; }
            g.m_age += dtf;
            g.m_env = std::min(1.0f, g.m_age / g.m_bloom) * std::exp(-6.9077553f * std::max(0.0f, g.m_age - g.m_bloom) / g.m_life);
            if (g.m_age > g.m_bloom && g.m_env < 1.0e-4f) { g.m_on = false; cv.m_active = 0; continue; }
            const float x = std::min(1.0f, g.m_age / g.m_life);
            const float keepTime = cv.m_time;
            cv = g.m_base;
            cv.m_time = g.m_restart ? 0.0f : keepTime;
            cv.m_restart = g.m_restart ? 1 : 0;
            cv.m_onset = 0;
            cv.m_seed = g.m_seed;
            cv.m_active = 1;
            cv.m_gain = g.m_env * level;
            cv.m_width = g.m_base.m_width * (1.0f + 3.0f * blur * x);
            cv.m_drift = g.m_base.m_drift + 12.0f * blur * x;
            cv.m_brightness = g.m_base.m_brightness * (1.0f - 0.7f * blur * x);
            cv.m_distance = g.m_base.m_distance + (1.0f - g.m_base.m_distance) * sink * std::min(1.0f, 2.0f * x);
            g.m_restart = false;
        }
    }

    for (int i = 0; i < static_cast<int>(m_voices.size()) && i < MaxViewVoices; ++i) {
        const Voice& v = m_voices[static_cast<size_t>(i)];
        const CloudVoice& cv = m_cloud.voice(i);
        const bool on = v.m_note >= 0 && cv.m_active;
        m_viewHz[static_cast<size_t>(i)].store(on ? cv.m_hz : 0.0f, std::memory_order_relaxed);
        m_viewGain[static_cast<size_t>(i)].store(on ? cv.m_gain : 0.0f, std::memory_order_relaxed);
        m_viewDistance[static_cast<size_t>(i)].store(v.m_distance, std::memory_order_relaxed);
        m_viewTime[static_cast<size_t>(i)].store(cv.m_time, std::memory_order_relaxed);
        m_viewSource[static_cast<size_t>(i)].store(v.m_source == FoundationSource ? 1 : v.m_source, std::memory_order_relaxed);
        m_viewNote[static_cast<size_t>(i)].store(on ? v.m_note : -1, std::memory_order_relaxed);
        m_viewPress[static_cast<size_t>(i)].store(v.m_press, std::memory_order_relaxed);
        m_viewSlide[static_cast<size_t>(i)].store(v.m_slide, std::memory_order_relaxed);
        m_viewBend[static_cast<size_t>(i)].store(v.m_bend, std::memory_order_relaxed);
        m_viewTune[static_cast<size_t>(i)].store(v.m_tuneCents, std::memory_order_relaxed);
    }
    for (size_t gi = 0; gi < m_ghosts.size(); ++gi) {   // the ghosts after the voices, as source 3
        const size_t i = m_voices.size() + gi;
        if (i >= static_cast<size_t>(MaxViewVoices)) break;
        const CloudVoice& cv = m_cloud.voice(static_cast<int>(i));
        const bool on = m_ghosts[gi].m_on && cv.m_active;
        m_viewHz[i].store(on ? cv.m_hz : 0.0f, std::memory_order_relaxed);
        m_viewGain[i].store(on ? cv.m_gain : 0.0f, std::memory_order_relaxed);
        m_viewDistance[i].store(cv.m_distance, std::memory_order_relaxed);
        m_viewTime[i].store(cv.m_time, std::memory_order_relaxed);
        m_viewSource[i].store(3, std::memory_order_relaxed);
        m_viewNote[i].store(-1, std::memory_order_relaxed);
    }

    // 4. The modal hall's settings: it runs on the GPU in the same graph, hearing the voices' block
    //    there. The level is applied here, so the hall rests (on = 0) only once the level has been
    //    at 0 for a whole block; its pitches are the scale's, from its lowest mode to 16 kHz.
    {
        ModalHall& mh = m_cloud.modal();
        const float level = cur(ParamIdE::MODAL_LEVEL);
        mh.m_on = (level > 0.0f || m_prevModalLevel > 0.0f) ? 1 : 0;
        mh.m_freeze = cur(ParamIdE::MODAL_FREEZE) >= 0.5f ? 1 : 0;
        mh.m_decay = cur(ParamIdE::MODAL_DECAY);
        mh.m_damp = cur(ParamIdE::MODAL_DAMP);
        mh.m_tune = cur(ParamIdE::MODAL_TUNE);
        mh.m_spread = cur(ParamIdE::MODAL_SPREAD);
        mh.m_drift = cur(ParamIdE::MODAL_DRIFT);
        mh.m_width = cur(ParamIdE::MODAL_WIDTH);
        mh.m_low = cur(ParamIdE::MODAL_LOW);
        mh.m_time = m_modalTime;
        mh.m_breath = cur(ParamIdE::MODAL_BREATH);
        // The painted hall reads the landscape at its own clock, wrapped at the landscape's loop.
        mh.m_paint = cur(ParamIdE::MODAL_PAINT);
        mh.m_paintNorm = 1.0f / std::max(0.125f, m_landRms);
        m_paintTime += dtf;
        if (m_landLoop > 0.0f && m_paintTime >= m_landLoop) m_paintTime -= m_landLoop;
        mh.m_paintTime = m_paintTime;
        int count = 0;
        if (mh.m_on)
            for (int note = 0; note < 128 && count < ModalTuned; ++note) {
                const double f = frequencyOf(note) * tideMul;
                if (f >= mh.m_low && f <= 16000.0) mh.m_tuned[count++] = static_cast<float>(f);
            }
        mh.m_tunedCount = count;
        // Its memory (Grow): it listens while Grow or the partials' time asks for it -- the partials'
        // amplitudes gathered over the octaves on the card, the spectrum's density followed over a third of
        // a second, and the memory a step of Memory towards it while anything sounds. Grow at nothing:
        // it neither learns nor changes. It learns with the hall's level at nothing too.
        mh.m_grow = cur(ParamIdE::HALL_GROW);
        const bool bent = cur(ParamIdE::TIME_BEND) != 0.0f;
        mh.m_listen = (mh.m_grow != 0.0f || bent) ? 1 : 0;
        mh.m_memoryCoef = mh.m_grow != 0.0f ? 1.0f - std::exp(-dtf / std::max(cur(ParamIdE::HALL_MEMORY), 1.0f)) : 0.0f;
        mh.m_densityCoef = mh.m_listen ? 1.0f - std::exp(-dtf / 0.3f) : 0.0f;
        if (mh.m_memoryCoef > 0.0f && m_rippleVoices > 0) m_hallAge += dt;
        m_modalTime += dtf;
        if (m_modalTime >= 1000.0f) m_modalTime -= 1000.0f;
    }

    // The field on the partials and the dispersion of their clocks. Scale: the scale's pitches,
    // ascending; Swarm and Scatter: eight Lorenz systems always wander (so the count can change
    // without a jump), the first `count` are the attractors -- x across seven octaves from 60 Hz,
    // z the weight, negative for Scatter.
    {
        SpectralField& F = m_cloud.field();
        const int mode = std::clamp(static_cast<int>(std::lround(cur(ParamIdE::FIELD_MODE))), 0, NumFieldModes - 1);
        F.m_mode = static_cast<FieldModeE>(mode);
        const float pull = cur(ParamIdE::FIELD_STRENGTH), spring = cur(ParamIdE::FIELD_SPRING);
        F.m_strength = 600.0f * pull * pull;
        F.m_radius = cur(ParamIdE::FIELD_RADIUS);
        F.m_spring = 4.0f * spring * spring;
        F.m_dispersion = cur(ParamIdE::DISPERSION);
        F.m_flockFresh = (F.m_mode == FieldModeE::FLOCK && !m_flockWas) ? 1 : 0;
        m_flockWas = F.m_mode == FieldModeE::FLOCK;
        F.m_orbitFresh = (F.m_mode == FieldModeE::ORBIT && !m_orbitWas) ? 1 : 0;
        m_orbitWas = F.m_mode == FieldModeE::ORBIT;
        // The Orbit: Pull the wells' depth (0.5 .. 8.5 rad/s), Wander the magnetic field (the circling,
        // 0.5 .. 30 rad/s), Spring the damping -- light, the circles keep their course; heavy, they
        // wander; Radius the cents to a full side, and the orbits' size about half of it.
        F.m_orbitWell = 0.5f + 8.0f * pull;
        F.m_orbitField = 0.5f + 30.0f * cur(ParamIdE::FIELD_RATE);
        F.m_orbitDamp = 0.05f + 1.5f * spring * spring;
        F.m_timeBend = cur(ParamIdE::TIME_BEND);
        const double span = static_cast<double>(cur(ParamIdE::FIELD_RATE)) * dt;
        const int steps = std::max(1, static_cast<int>(std::ceil(span / 0.002)));
        const double h = span / steps;
        for (auto& s : m_lorenz)
            for (int k = 0; k < steps; ++k) {   // the midpoint rule, small steps
                auto d = [](const double* p, double* o) { o[0] = 10.0 * (p[1] - p[0]); o[1] = p[0] * (28.0 - p[2]) - p[1]; o[2] = p[0] * p[1] - (8.0 / 3.0) * p[2]; };
                double k1[3], mid[3], k2[3];
                d(s, k1);
                for (int j = 0; j < 3; ++j) mid[j] = s[j] + 0.5 * h * k1[j];
                d(mid, k2);
                for (int j = 0; j < 3; ++j) s[j] += h * k2[j];
            }
        if (F.m_mode == FieldModeE::SCALE || F.m_mode == FieldModeE::ORBIT) {
            int count = 0;
            for (int note = 0; note < 128 && count < FieldPoints; ++note) {
                const float lf = static_cast<float>(std::log2(frequencyOf(note) * tideMul));
                if (count == 0 || lf > F.m_pos[count - 1] + 1.0e-4f) F.m_pos[count++] = lf;
            }
            F.m_count = count;
        } else if (F.m_mode == FieldModeE::FLOCK) {
            // The flock's winds: each cloud blown by one of the eight Lorenz systems' x (cloud h by h mod 8).
            for (int k = 0; k < FieldAttractors; ++k) F.m_pos[k] = std::clamp(static_cast<float>(m_lorenz[k][0] / 22.0), -1.0f, 1.0f);
            F.m_count = FieldAttractors;
        } else if (F.m_mode != FieldModeE::OFF) {
            const int count = std::clamp(static_cast<int>(std::lround(cur(ParamIdE::FIELD_COUNT))), 1, FieldAttractors);
            const float lo = std::log2(60.0f), octaves = std::log2(8000.0f / 60.0f);
            for (int k = 0; k < count; ++k) {
                const float u = std::clamp(static_cast<float>((m_lorenz[k][0] + 22.0) / 44.0), 0.0f, 1.0f);
                const float w = 0.4f + 0.8f * std::clamp(static_cast<float>(m_lorenz[k][2] / 50.0), 0.0f, 1.0f);
                F.m_pos[k] = lo + u * octaves;
                F.m_mass[k] = F.m_mode == FieldModeE::SCATTER ? -w : w;
            }
            F.m_count = count;
        } else {
            F.m_count = 0;
        }
    }

    // The living landscape: its mode (a change seeds it), its steps (up to two a block: Lenia 2 to 40
    // a second, the Turing field 20 to 750, by Life Rate), its character (Lenia's growth; the
    // Turing field along spots, mitosis, coral, worms, mazes in Pearson's map), the make-up from
    // the field's measured power, and a new seeding when it has died or filled up.
    {
        LifeField& L = m_cloud.life();
        const int mode = std::clamp(static_cast<int>(std::lround(cur(ParamIdE::LIFE_MODE))), 0, NumLifeModes - 1);
        L.m_mode = static_cast<LifeModeE>(mode);
        m_lifeQuiet += dtf;
        if (mode != m_lifeModeWas && mode != 0) { L.m_reseed = 1; ++L.m_salt; m_lifeAcc = 0.0f; m_lifeQuiet = 0.0f; m_lifeMakeup = 1.0f; }
        m_lifeModeWas = mode;
        if (mode != 0) {
            const float rate = cur(ParamIdE::LIFE_RATE);
            if (lifeIsPicture(L.m_mode)) {
                // The animated pictures: a step every block; the flow at Life Rate (squared: slow at
                // the bottom), the turn at Turn, Rain's drops by Life Rate too.
                L.m_steps = 1;
                m_pictureTime += (0.01 + 0.5 * static_cast<double>(rate) * rate) * dt;
                m_pictureAngle += 2.0 * 3.14159265358979 * static_cast<double>(cur(ParamIdE::LIFE_TURN)) / 60.0 * dt;
                m_pictureSeconds += dt;
                m_pictureAngle = std::remainder(m_pictureAngle, 2.0 * 3.14159265358979 * 60.0);   // whole turns away, kept small
                L.m_time = static_cast<float>(std::fmod(m_pictureTime, 4096.0));
                L.m_angle = static_cast<float>(m_pictureAngle);
                L.m_seconds = static_cast<float>(std::fmod(m_pictureSeconds, 600000.0));
                L.m_sides = std::round(cur(ParamIdE::LIFE_SIDES));
                L.m_character = std::clamp(cur(ParamIdE::LIFE_CHARACTER), 0.0f, 1.0f);
                L.m_drops = static_cast<float>((0.5 + 30.0 * static_cast<double>(rate) * rate) * dt / (LifeW * LifeH));   // drops a second, over every cell
                L.m_turnStep = static_cast<float>(2.0 * 3.14159265358979 * static_cast<double>(cur(ParamIdE::LIFE_TURN)) / 60.0 * dt);
            } else if (L.m_mode == LifeModeE::ISLANDS) {
                // The separation: the field drawn every block, Life Rate's substeps of it (squared: slow
                // at the bottom), drifting across.
                L.m_steps = 1;
                L.m_substeps = std::clamp(1 + static_cast<int>(std::lround(15.0f * rate * rate)), 1, 16);
                m_islandDrift = std::fmod(m_islandDrift + (0.05 + 0.6 * static_cast<double>(rate)) * dt, static_cast<double>(IsleW));
                L.m_drift = static_cast<float>(m_islandDrift);
            } else {
                // Lenia 2 to 40 steps a second, the Fluid 30 to 430, the Turing field 20 to 750.
                const float perSecond = L.m_mode == LifeModeE::LENIA ? 2.0f + 38.0f * rate : (L.m_mode == LifeModeE::FLUID ? 30.0f + 400.0f * rate : 20.0f + 730.0f * rate);
                m_lifeAcc += perSecond * dtf;
                const int steps = std::min(2, static_cast<int>(m_lifeAcc));
                m_lifeAcc = std::min(m_lifeAcc - static_cast<float>(steps), 2.0f);
                L.m_steps = steps;
                L.m_fade = std::exp(-1.0f / (25.0f * perSecond));   // the Fluid's ink lasts about 25 s
            }
            m_lifeSeconds += dt;
            const float ch = std::clamp(cur(ParamIdE::LIFE_CHARACTER), 0.0f, 1.0f);
            L.m_lenMu = 0.12f + 0.08f * ch;
            L.m_lenSigma = 0.012f + 0.012f * ch;
            static const float F[5] = { 0.035f, 0.0367f, 0.0545f, 0.078f, 0.029f }, K[5] = { 0.065f, 0.0649f, 0.062f, 0.061f, 0.057f };
            const float at = ch * 4.0f;
            const int seg = std::min(3, static_cast<int>(at));
            const float fr = at - static_cast<float>(seg);
            L.m_feed = F[seg] + (F[seg + 1] - F[seg]) * fr;
            L.m_kill = K[seg] + (K[seg + 1] - K[seg]) * fr;
            // The Fluid: Character from syrup to turbulence. The Islands: from a fine labyrinth to a
            // few large islands of the minority (the mean moved off the middle, Oono and Puri's term
            // weakened so the domains keep growing).
            L.m_tau = 0.56f - 0.055f * ch;
            L.m_mean = -0.4f * ch;
            L.m_eps = 0.0015f * (1.0f - ch) + 0.00005f;
            // The Fluid's jets: the eight loudest notes, each blowing at its frequency and at its
            // octave, leaning slowly to and fro.
            std::fill(std::begin(L.m_jetAmt), std::end(L.m_jetAmt), 0.0f);
            if (L.m_mode == LifeModeE::FLUID) {
                std::array<int, 64> loud{};
                int blowing = 0;
                for (int i = 0; i < static_cast<int>(m_voices.size()) && blowing < static_cast<int>(loud.size()); ++i) {
                    const Voice& v = m_voices[static_cast<size_t>(i)];
                    if (v.m_note >= 0 && v.m_lastGain > 0.001f && v.m_hz > 0.0) loud[static_cast<size_t>(blowing++)] = i;
                }
                std::stable_sort(loud.begin(), loud.begin() + blowing, [this](int a, int b) {
                    return m_voices[static_cast<size_t>(a)].m_lastGain > m_voices[static_cast<size_t>(b)].m_lastGain;
                });
                const int count = std::min(blowing, LifeJets / 2);
                for (int j = 0; j < count; ++j) {
                    const Voice& v = m_voices[static_cast<size_t>(loud[j])];
                    const float across = L.m_zoom * static_cast<float>(LifeW) / std::log2(1000.0f);
                    const float x = std::log2(std::max(static_cast<float>(v.m_hz), 20.0f) / 20.0f) * across;
                    const float amt = std::min(1.0f, 1.5f * v.m_lastGain);
                    const float sway = 0.35f * static_cast<float>(std::sin(0.13 * m_lifeSeconds + 1.7 * v.m_note));
                    L.m_jetX[2 * j] = x;
                    L.m_jetAmt[2 * j] = amt;
                    L.m_jetSway[2 * j] = sway;
                    L.m_jetX[2 * j + 1] = x + across;   // the octave
                    L.m_jetAmt[2 * j + 1] = 0.6f * amt;
                    L.m_jetSway[2 * j + 1] = -sway;
                }
            }
            L.m_depth = cur(ParamIdE::LIFE_DEPTH);
            L.m_zoom = cur(ParamIdE::LIFE_ZOOM);
            float mean = 0.0f, meanSq = 0.0f;
            m_cloud.lifeStats(mean, meanSq);
            // The make-up m that keeps the partials' power: E[(1 - d + d v m)^2] = 1, a quadratic in m
            // (one over the RMS is right only at full depth; below it the steady share is mixed in).
            {
                const double d = std::clamp(static_cast<double>(L.m_depth), 0.0, 1.0), e = mean, q = std::max(static_cast<double>(meanSq), 1.0 / 4096.0);
                const double A = d * d * q, B = 2.0 * (1.0 - d) * d * e, C = (1.0 - d) * (1.0 - d) - 1.0;
                const double m = A > 1.0e-12 ? (-B + std::sqrt(std::max(B * B - 4.0 * A * C, 0.0))) / (2.0 * A) : 1.0;
                m_lifeMakeup += (1.0f - std::exp(-dtf / 0.5f)) * (static_cast<float>(std::clamp(m, 0.0, 8.0)) - m_lifeMakeup);
                L.m_makeup = m_lifeMakeup;
            }
            const bool dead = !lifeIsPicture(L.m_mode) && mean < 0.002f, full = L.m_mode == LifeModeE::LENIA && mean > 0.85f;
            if ((dead || full) && m_lifeQuiet > 2.0f && L.m_reseed == 0) { L.m_reseed = 1; ++L.m_salt; m_lifeQuiet = 0.0f; }
        } else {
            L.m_steps = 0;
        }
    }

    stepRipple(dtf);

    // Turbulence: drive 7 T^2 a second at full strength, by the cube of how loud the instrument sounds; the
    // cascade keeps running for twice its settling time after it is turned off, so the harmonics
    // come back to rest before it stops.
    {
        const float T = cur(ParamIdE::TURBULENCE);
        Turbulence& tb = m_cloud.turbulence();
        tb.m_settle = cur(ParamIdE::TURBULENCE_TIME);
        const float loud = 0.25f + 0.75f * m_mods.value(ModSourceE::LOUDNESS);
        m_turbDrive = 7.0f * T * T * loud * loud * loud;
        m_turbAfter = T > 0.0f ? 2.0f * tb.m_settle : std::max(0.0f, m_turbAfter - dtf);
        tb.m_on = (T > 0.0f || m_turbAfter > 0.0f) ? 1.0f : 0.0f;
    }

    // The spectral comb: its phase runs on at the sweep (kept within a notch), and the make-up is
    // exact -- the raised cosine's mean square is (1 - d/2)^2 + d^2/8 over the notches.
    {
        SpectralComb& K = m_cloud.comb();
        K.m_depth = cur(ParamIdE::COMB_DEPTH);
        K.m_density = cur(ParamIdE::COMB_DENSITY);
        K.m_stereo = cur(ParamIdE::COMB_STEREO);
        m_combPhase += static_cast<double>(cur(ParamIdE::COMB_SWEEP)) * dt;
        m_combPhase -= std::floor(m_combPhase);
        K.m_phase = static_cast<float>(m_combPhase);
        m_combPhaseView.store(K.m_phase, std::memory_order_relaxed);
        const float d = K.m_depth;
        K.m_makeup = 1.0f / std::sqrt(std::max((1.0f - 0.5f * d) * (1.0f - 0.5f * d) + 0.125f * d * d, 1.0e-6f));
    }

    // The reservoir: stepped only while a row reads it; fed with what the instrument hears and
    // plays (the sources as the last block left them), each standardised against its last half
    // minute -- the reservoir hears how the music departs from its recent self.
    {
        Reservoir& R = m_cloud.reservoir();
        R.m_on = 0;
        for (int r = 0; r < ModRows; ++r) {
            const ModRow row = unpackModRow(m_modRows[static_cast<size_t>(r)].load(std::memory_order_relaxed));
            auto isRes = [](ModSourceE s) { return s >= ModSourceE::RESERVOIR1 && s <= ModSourceE::RESERVOIR4; };
            if (row.m_target >= 0 && row.m_depth != 0.0f && (isRes(row.m_source) || isRes(row.m_via))) R.m_on = 1;
        }
        if (m_sourcesWatched.load(std::memory_order_relaxed)) R.m_on = 1;
        R.m_chaos = cur(ParamIdE::RESERVOIR_CHAOS);
        R.m_leak = 1.0f - std::exp(-dtf / std::max(cur(ParamIdE::RESERVOIR_MEMORY), 0.05f));
        R.m_input = cur(ParamIdE::RESERVOIR_INPUT);
        const ModSourceE feed[ReservoirInputs] = { ModSourceE::LOUDNESS, ModSourceE::CENTROID, ModSourceE::FLUX, ModSourceE::ROUGHNESS,
                                                   ModSourceE::BEAT, ModSourceE::NOTE, ModSourceE::CASCADE, ModSourceE::LENIA1 };
        const float a = 1.0f - std::exp(-dtf / 30.0f);
        for (int k = 0; k < ReservoirInputs; ++k) {
            const float v = m_mods.value(feed[k]);
            if (!m_resStarted) { m_resMean[k] = v; m_resVar[k] = 0.0f; }
            const float d = v - m_resMean[k];
            m_resMean[k] += a * d;
            m_resVar[k] = (1.0f - a) * (m_resVar[k] + a * d * d);
            R.m_u[k] = std::clamp(0.5f * d / (std::sqrt(m_resVar[k]) + 0.05f), -2.0f, 2.0f);
        }
        m_resStarted = true;
    }

    // Partial sync: the coupling in units of the oscillators' own spread of rates -- 2.5 times it at
    // full Coupling, well past the critical strength (for a uniform spread about 1.3 times it).
    {
        PartialSync& S = m_cloud.sync();
        S.m_depth = cur(ParamIdE::SYNC_DEPTH);
        S.m_rate = cur(ParamIdE::SYNC_RATE);
        S.m_spread = cur(ParamIdE::SYNC_SPREAD);
        S.m_mode = static_cast<SyncModeE>(std::clamp(static_cast<int>(std::lround(cur(ParamIdE::SYNC_MODE))), 0, NumSyncModes - 1));
        S.m_coupling = cur(ParamIdE::SYNC_COUPLING) * 2.5f * TwoPi * S.m_rate * (S.m_spread + 0.05f);
        // Chimera and Swarm live on their own time, a couple of the rate's turns: in units of it.
        if (S.m_mode != SyncModeE::GLOBAL) S.m_coupling = cur(ParamIdE::SYNC_COUPLING) * 2.0f * TwoPi * std::max(S.m_rate, 0.05f);
        S.m_swarm = cur(ParamIdE::SYNC_SWARM) * 2.0f * TwoPi * std::max(S.m_rate, 0.05f);
        S.m_reach = cur(ParamIdE::SYNC_REACH);
        S.m_lag = 1.25f + 0.3f * std::clamp(cur(ParamIdE::SYNC_LAG), 0.0f, 1.0f);   // 1.39 (Abrams and Strogatz) at about 0.47
    }

    // The binaural cloud (its orbit turns the whole of it) and the spectral echo (its lines
    // cleared when it is switched on; its delay less the 1024 - G samples its window adds).
    {
        Binaural& b = m_cloud.binaural();
        b.m_on = cur(ParamIdE::BINAURAL) >= 0.5f ? 1 : 0;
        b.m_span = cur(ParamIdE::BINAURAL_SPAN);
        b.m_height = cur(ParamIdE::BINAURAL_HEIGHT);
        m_binAngle += 6.2831853f * cur(ParamIdE::BINAURAL_ORBIT) * dtf;
        if (m_binAngle > 3.14159265f) m_binAngle -= 6.2831853f;
        if (m_binAngle < -3.14159265f) m_binAngle += 6.2831853f;
        b.m_angle = m_binAngle;
        SpectralEcho& se = m_cloud.spectralEcho();
        const bool on = cur(ParamIdE::SPEC_MIX) > 0.0f || cur(ParamIdE::SPEC_TO_FAR) * cur(ParamIdE::SPEC_MIX) > 0.0f || m_prevSpecMix > 0.0f;
        if (on && !m_specWasOn) se.m_clear = 1;
        m_specWasOn = on;
        se.m_on = on ? 1 : 0;
        se.m_frames = std::max(1.0f, (cur(ParamIdE::SPEC_TIME) * static_cast<float>(m_cfg.m_sampleRate) - static_cast<float>(1024 - G)) / static_cast<float>(G));
        se.m_tilt = cur(ParamIdE::SPEC_TILT);
        se.m_scatter = cur(ParamIdE::SPEC_SCATTER);
        se.m_feedback = cur(ParamIdE::SPEC_FEEDBACK);
        se.m_damp = cur(ParamIdE::SPEC_DAMP);

        // The melt: its time to -60 dB as the held energy's share kept a frame, the spread as a
        // diffusion (after a second the standard deviation is Spread octaves), the bands' pull to
        // their centres and their detuning growing with it.
        Melt& me = m_cloud.melt();
        const bool meltOn = cur(ParamIdE::MELT_MIX) > 0.0f || m_prevMeltMix > 0.0f;
        if (meltOn && !m_meltWasOn) me.m_clear = 1;
        m_meltWasOn = meltOn;
        me.m_on = meltOn ? 1 : 0;
        const float hopS = static_cast<float>(MeltHop) / static_cast<float>(m_cfg.m_sampleRate);
        const float spread = cur(ParamIdE::MELT_SPREAD);
        me.m_keep = std::pow(10.0f, -6.0f * hopS / std::max(cur(ParamIdE::MELT_TIME), 0.1f));
        me.m_widen = spread * 0.693147f * std::sqrt(hopS);
        me.m_pull = 1.0f - std::exp(-2.0f * spread * hopS);
        me.m_detune = (0.1f + 0.4f * std::min(spread, 1.0f)) * 6.2831853f / static_cast<float>(MeltN);
    }

    // 5. The GPU: near and far, left and right, the modal hall and the spectral echo.
    // The picture's snapshot about thirty times a second, brought down with the block.
    if (--m_snapCountdown <= 0) {
        m_snapCountdown = std::max(1, static_cast<int>(std::lround(0.033 * m_cfg.m_sampleRate / G)));
        m_cloud.requestSnapshot();
    }
    {   // a hall loaded (or a new one): up with this block, when the message thread is not just writing it
        std::unique_lock<std::mutex> lock(m_snapLock, std::try_to_lock);
        if (lock.owns_lock() && m_hallPendingSet) {
            m_cloud.setHallMemory(m_hallPending.data());
            m_hallAge = m_hallPendingAge;
            m_snapHall = m_hallPending;
            m_snapHallAge = m_hallAge;
            m_snapHallValid = true;
            m_hallPendingSet = false;
        }
    }
    LARGE_INTEGER qpcLaunch, qpcBack;
    QueryPerformanceCounter(&qpcLaunch);
    m_cloud.launch();
    const float* q = m_cloud.wait();
    QueryPerformanceCounter(&qpcBack);
    {
        const float* dots = nullptr;
        int per = 0;
        if (m_cloud.takeSnapshot(dots, per)) {
            std::unique_lock<std::mutex> lock(m_snapLock, std::try_to_lock);   // the picture is reading: this one is skipped
            if (lock.owns_lock()) {
                const size_t count = static_cast<size_t>(m_cloud.config().m_voices) * static_cast<size_t>(per) * 2;
                if (m_snap.size() != count) m_snap.resize(count);   // once: the size stays
                std::memcpy(m_snap.data(), dots, count * sizeof(float));
                m_snapPer = per;
                if (const float* cells = m_cloud.lifePicture()) m_snapLife.assign(cells, cells + LifeW * LifeH);
                else m_snapLife.clear();
                if (m_cloud.ripple().m_depth > 0.0f) {   // the ripple's string (once sized: no allocation after)
                    m_snapRipple.resize(static_cast<size_t>(RippleCells));
                    std::copy(std::begin(m_rippleNow), std::end(m_rippleNow), m_snapRipple.begin());
                } else {
                    m_snapRipple.clear();
                }
                if (m_cloud.hallMemory(m_snapHall.data())) {   // and the hall's memory
                    m_snapHallValid = true;
                    m_snapHallAge = m_hallAge;
                }
                ++m_snapSeq;
            }
        }
    }
    m_gpuMs.store(m_cloud.lastGpuMs(), std::memory_order_relaxed);
    for (int s = 0; s < G; ++s) {
        const float* f = q + static_cast<size_t>(CloudOutChannels) * s;
        const size_t k = static_cast<size_t>(s);
        m_nearL[k] = f[0];
        m_nearR[k] = f[1];
        m_farL[k] = f[2] * m_blk.m_farLevel;
        m_farR[k] = f[3] * m_blk.m_farLevel;
        m_modalL[k] = f[4];
        m_modalR[k] = f[5];
        m_specL[k] = f[6];
        m_specR[k] = f[7];
        m_meltL[k] = f[8];
        m_meltR[k] = f[9];
    }
    // A level from the last block's to this one's, across the block (no zipper when it is modulated).
    auto ramp = [G](float from, float to, int s) { return from + (to - from) * static_cast<float>(s + 1) / static_cast<float>(G); };

    // 6. The near plane, as Noctuary's: two echoes, the second in series after the first, receding
    //    partly into the far plane; a small reverb; the early reflections of a room, from the dry voices.
    std::copy(m_nearL.begin(), m_nearL.end(), m_dryL.begin());
    std::copy(m_nearR.begin(), m_nearR.end(), m_dryR.begin());
    m_delay.set(cur(ParamIdE::DELAY_TIME_L), cur(ParamIdE::DELAY_TIME_R), cur(ParamIdE::DELAY_FEEDBACK), cur(ParamIdE::DELAY_CROSS),
               cur(ParamIdE::DELAY_DAMP), cur(ParamIdE::DELAY_ABSORB));
    m_delay.setDuck(cur(ParamIdE::DELAY_DUCK));
    m_delay.process(m_nearL.data(), m_nearR.data(), m_wetL.data(), m_wetR.data(), G);
    {
        const float m1 = cur(ParamIdE::DELAY_MIX), t1 = cur(ParamIdE::DELAY_TO_FAR);
        for (int s = 0; s < G; ++s) {
            const size_t k = static_cast<size_t>(s);
            const float m = ramp(m_prevDelayMix, m1, s), tf = ramp(m_prevDelayToFar, t1, s);
            m_nearL[k] += m_wetL[k] * m;  m_nearR[k] += m_wetR[k] * m;
            m_farL[k] += m_wetL[k] * tf;  m_farR[k] += m_wetR[k] * tf;
        }
        m_prevDelayMix = m1;
        m_prevDelayToFar = t1;
    }
    m_delay2.set(cur(ParamIdE::DELAY2_TIME_L), cur(ParamIdE::DELAY2_TIME_R), cur(ParamIdE::DELAY2_FEEDBACK), cur(ParamIdE::DELAY_CROSS),
                cur(ParamIdE::DELAY2_DAMP), cur(ParamIdE::DELAY_ABSORB));
    m_delay2.setDuck(cur(ParamIdE::DELAY_DUCK));
    m_delay2.process(m_nearL.data(), m_nearR.data(), m_wetL.data(), m_wetR.data(), G);
    {
        const float m2 = cur(ParamIdE::DELAY2_MIX), t2 = cur(ParamIdE::DELAY2_TO_FAR);
        for (int s = 0; s < G; ++s) {
            const size_t k = static_cast<size_t>(s);
            const float m = ramp(m_prevDelay2Mix, m2, s), tf = ramp(m_prevDelay2ToFar, t2, s);
            m_nearL[k] += m_wetL[k] * m;  m_nearR[k] += m_wetR[k] * m;
            m_farL[k] += m_wetL[k] * tf;  m_farR[k] += m_wetR[k] * tf;
        }
        m_prevDelay2Mix = m2;
        m_prevDelay2ToFar = t2;
    }
    {   // the spectral echo, from the GPU: onto the near plane and partly into the far
        const float m3 = cur(ParamIdE::SPEC_MIX), t3 = cur(ParamIdE::SPEC_TO_FAR) * m3;
        for (int s = 0; s < G; ++s) {
            const size_t k = static_cast<size_t>(s);
            const float m = ramp(m_prevSpecMix, m3, s), tf = ramp(m_prevSpecToFar, t3, s);
            m_nearL[k] += m_specL[k] * m;  m_nearR[k] += m_specR[k] * m;
            m_farL[k] += m_specL[k] * tf;  m_farR[k] += m_specR[k] * tf;
        }
        m_prevSpecMix = m3;
        m_prevSpecToFar = t3;
    }
    {   // the melt, from the GPU: onto the near plane and partly into the far
        const float m4 = cur(ParamIdE::MELT_MIX), t4 = cur(ParamIdE::MELT_TO_FAR) * m4;
        for (int s = 0; s < G; ++s) {
            const size_t k = static_cast<size_t>(s);
            const float m = ramp(m_prevMeltMix, m4, s), tf = ramp(m_prevMeltToFar, t4, s);
            m_nearL[k] += m_meltL[k] * m;  m_nearR[k] += m_meltR[k] * m;
            m_farL[k] += m_meltL[k] * tf;  m_farR[k] += m_meltR[k] * tf;
        }
        m_prevMeltMix = m4;
        m_prevMeltToFar = t4;
    }
    m_nearRev.setSpace(0.3f, 20000.0f, 80.0f);
    m_nearRev.set(0.6f, cur(ParamIdE::NEAR_DECAY), cur(ParamIdE::NEAR_DAMP), 5.0f, false, cur(ParamIdE::NEAR_MIX));
    m_nearRev.process(m_nearL.data(), m_nearR.data(), G);
    m_early.setRoom(cur(ParamIdE::EARLY_SIZE), cur(ParamIdE::EARLY_ABSORB), 1.0f);
    m_early.setSource(0.0f, m_srcDistance);
    m_early.setLevel(cur(ParamIdE::EARLY_LEVEL));
    if (m_early.active()) m_early.process(m_dryL.data(), m_dryR.data(), m_nearL.data(), m_nearR.data(), G);

    // 7. The far plane is heard only through the far reverb: the swell of the diffusion first, the
    //    hall stepping aside for the near plane band by band after (Unmask).
    m_diffuser.set(cur(ParamIdE::FAR_DIFFUSE));
    m_diffuser.process(m_farL.data(), m_farR.data(), G);
    m_far.setMode(std::clamp(static_cast<int>(std::lround(cur(ParamIdE::FAR_MODE))), 0, NumFarModes - 1));
    m_far.set(cur(ParamIdE::FAR_SIZE), cur(ParamIdE::FAR_DECAY), cur(ParamIdE::FAR_DAMP), cur(ParamIdE::FAR_PRE_DELAY),
             cur(ParamIdE::FAR_FREEZE) >= 0.5f, 1.0f);
    m_far.setSpace(0.5f, 9000.0f, 60.0f);
    m_far.process(m_farL.data(), m_farR.data(), G);
    m_unmask.set(cur(ParamIdE::FAR_UNMASK), 0.3f, 0.5f);
    m_unmask.process(m_nearL.data(), m_nearR.data(), m_farL.data(), m_farR.data(), G);

    // 8. Together: the near plane, the far hall and the modal hall.
    {
        const float ml = cur(ParamIdE::MODAL_LEVEL);
        for (int s = 0; s < G; ++s) {
            const size_t k = static_cast<size_t>(s);
            const float m = ramp(m_prevModalLevel, ml, s);
            m_nearL[k] += m_farL[k] + m_modalL[k] * m;
            m_nearR[k] += m_farR[k] + m_modalR[k] * m;
        }
        m_prevModalLevel = ml;
    }
    // Binaural: the master leaves the two ears alone (no mono bass, no width): both would undo the head.
    if (cur(ParamIdE::BINAURAL) >= 0.5f) m_midSide.set(20.0f, 0.0f, 1.0f);
    else m_midSide.set(cur(ParamIdE::BASS_MONO), 0.0f, cur(ParamIdE::STEREO_WIDTH));
    m_midSide.process(m_nearL.data(), m_nearR.data(), G);
    for (int s = 0; s < G; ++s) {
        const size_t at = static_cast<size_t>((pos + s) & m_ringMask) * 2;
        m_ring[at] = softClip(m_nearL[static_cast<size_t>(s)] * m_blk.m_master);
        m_ring[at + 1] = softClip(m_nearR[static_cast<size_t>(s)] * m_blk.m_master);
    }
    m_renderPos.store(pos + G, std::memory_order_release);
    LARGE_INTEGER qpcEnd, qpcFreq;
    QueryPerformanceCounter(&qpcEnd);
    QueryPerformanceFrequency(&qpcFreq);
    auto keepWorst = [&](std::atomic<int>& at, LONGLONG from, LONGLONG to) {
        const int us = static_cast<int>((to - from) * 1000000 / qpcFreq.QuadPart);
        int worst = at.load(std::memory_order_relaxed);
        while (us > worst && !at.compare_exchange_weak(worst, us, std::memory_order_relaxed)) {}
    };
    keepWorst(m_worstBlockUs, qpcStart.QuadPart, qpcEnd.QuadPart);
    keepWorst(m_worstPrepUs, qpcStart.QuadPart, qpcLaunch.QuadPart);
    keepWorst(m_worstWaitUs, qpcLaunch.QuadPart, qpcBack.QuadPart);
    keepWorst(m_worstPostUs, qpcBack.QuadPart, qpcEnd.QuadPart);
}

} // namespace additiv
