/**
 * @file Engine.cpp
 * @brief The instrument (Engine.h): voices, conductor, tuning, the GPU cloud, far reverb, master,
 *        and the worker that renders ahead of the audio callback.
 */
#include "additiv/Engine.h"

#include "additiv/Landscape.h"
#include "additiv/Motion.h"
#include "additiv/Shapes.h"

#define NOMINMAX
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <avrt.h>

#include <algorithm>
#include <cmath>
#include <cstring>

namespace additiv {
namespace {

constexpr std::uint32_t kQueueSize = 4096;   ///< note events in flight from the audio thread (a power of two)
constexpr float kVoiceLevel = 0.1f;          ///< RMS of one voice at full gain, all of its partials together (set so the autoplay sits near -24 dBFS, Noctuary's range)
constexpr double kGlideSeconds = 0.3;        ///< how fast a sounding voice follows a change of tuning
constexpr float kPurityNarrowing = 0.6f;     ///< at Purity 1 the clouds are this much narrower, so pure intervals can lock (PLAN.md)

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

} // namespace

Engine::Engine() { resetParams(); }

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
    cc.voices = cfg_.voices;
    cc.partialsPerVoice = cfg_.partialsPerVoice;
    cc.spinWait = true;
    if (!cloud_.open(cc, error)) return false;
    const std::vector<EnvShape> bank = factoryShapes();
    cloud_.setShapes(bank.data(), static_cast<int>(bank.size()));
    const Landscape aurora = landscapeAurora(30.0f);
    cloud_.setLandscape(aurora.amps.data(), aurora.columns, aurora.rows, aurora.rowSeconds);
    if (cfg_.realtimeGpuClass) raiseGpuClass();

    voices_.assign(static_cast<size_t>(cfg_.voices), Voice{});
    for (Voice& v : voices_) v.env.setSampleRate(cfg_.sampleRate / G);   // stepped once per GPU block
    rng_.seed(cfg_.seed);
    scaleChoice_ = -1;
    refreshScale();
    brain_.reset(rng_.fork(), 48 + rootPc_);
    far_.prepare(cfg_.sampleRate);
    midSide_.prepare(cfg_.sampleRate);
    nearL_.assign(static_cast<size_t>(G), 0.0f);
    nearR_.assign(static_cast<size_t>(G), 0.0f);
    farL_.assign(static_cast<size_t>(G), 0.0f);
    farR_.assign(static_cast<size_t>(G), 0.0f);

    queue_.assign(kQueueSize, Event{});
    qHead_.store(0);
    qTail_.store(0);
    pending_.clear();
    pending_.reserve(kQueueSize);

    // The timeline runs a host block plus the lookahead ahead of what the callback has taken,
    // rounded up to whole GPU blocks; the ring holds that and room to spare.
    const int ahead = cfg_.maxHostBlock + static_cast<int>(std::ceil(cfg_.lookaheadMs * 0.001 * cfg_.sampleRate));
    latency_ = (ahead + G - 1) / G * G;
    const std::int64_t frames = nextPow2(static_cast<std::int64_t>(latency_) + cfg_.maxHostBlock + 2 * G);
    ring_.assign(static_cast<size_t>(frames) * 2, 0.0f);
    ringMask_ = frames - 1;
    renderPos_.store(0);
    outPos_.store(0);
    underruns_.store(0);
    late_.store(0);
    sounding_.store(0);
    prepared_ = true;
    if (realtime_) startWorker();
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
        if (r + G <= out + latency_) { renderBlock(); continue; }
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
    queue_[head & (kQueueSize - 1)] = { 0, note, std::clamp(velocity, 0.0f, 1.0f), outPos_.load(std::memory_order_relaxed) + std::max(offset, 0) + latency_ };
    qHead_.store(head + 1, std::memory_order_release);
}

void Engine::noteOff(int note, int offset)
{
    const std::uint32_t head = qHead_.load(std::memory_order_relaxed);
    if (!prepared_ || head - qTail_.load(std::memory_order_acquire) >= kQueueSize) return;
    queue_[head & (kQueueSize - 1)] = { 1, note, 0.0f, outPos_.load(std::memory_order_relaxed) + std::max(offset, 0) + latency_ };
    qHead_.store(head + 1, std::memory_order_release);
}

void Engine::allNotesOff()
{
    const std::uint32_t head = qHead_.load(std::memory_order_relaxed);
    if (!prepared_ || head - qTail_.load(std::memory_order_acquire) >= kQueueSize) return;
    queue_[head & (kQueueSize - 1)] = { 2, 0, 0.0f, outPos_.load(std::memory_order_relaxed) + latency_ };
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

void Engine::readParams()
{
    auto g = [this](ParamId id) { return param(id); };
    blk_.master = dbToGain(g(ParamId::MasterGain));
    blk_.partials = static_cast<int>(std::lround(g(ParamId::Partials)));
    blk_.tilt = g(ParamId::Tilt);
    blk_.purity = g(ParamId::Purity);
    blk_.width = g(ParamId::CloudWidth) * (1.0f - kPurityNarrowing * blk_.purity);
    blk_.widthSlope = g(ParamId::WidthSlope);
    blk_.drift = g(ParamId::Drift);
    blk_.driftRate = g(ParamId::DriftRate);
    blk_.shimmer = g(ParamId::Shimmer);
    blk_.shimmerRate = g(ParamId::ShimmerRate);
    blk_.stereo = g(ParamId::Stereo);
    blk_.motion = static_cast<int>(std::lround(g(ParamId::Motion)));
    blk_.motionRate = g(ParamId::MotionRate);
    blk_.motionDepth = g(ParamId::MotionDepth);
    blk_.attack = g(ParamId::Attack);
    blk_.decay = g(ParamId::Decay);
    blk_.sustain = g(ParamId::Sustain);
    blk_.release = g(ParamId::Release);
    blk_.refPitch = g(ParamId::RefPitch);
    blk_.depth = g(ParamId::Depth);
    blk_.keysDepth = g(ParamId::KeysDepth);
    blk_.farLevel = g(ParamId::FarLevel);

    bp_.on = g(ParamId::BrainOn) >= 0.5f;
    bp_.mode = static_cast<BrainMode>(std::clamp(static_cast<int>(std::lround(g(ParamId::AutoMode))), 0, kNumBrainModes - 1));
    bp_.density = static_cast<int>(std::lround(g(ParamId::BrainDensity)));
    bp_.rateSeconds = bp_.mode == BrainMode::Chords ? g(ParamId::AutoRate) : g(ParamId::BrainRate);
    bp_.holdMin = g(ParamId::BrainHoldMin);
    bp_.holdMax = g(ParamId::BrainHoldMax);
    bp_.low = static_cast<int>(std::lround(g(ParamId::BrainLow)));
    bp_.high = static_cast<int>(std::lround(g(ParamId::BrainHigh)));
    bp_.consonance = g(ParamId::BrainConsonance);
    bp_.wander = g(ParamId::BrainWander);
    bp_.voiceLead = g(ParamId::AutoLead);
    bp_.chordTension = g(ParamId::AutoTension);
    bp_.rootMove = g(ParamId::AutoRootMove);
    bp_.timbre = g(ParamId::BrainTimbre);
    // The conductor judges timbre against the cloud's own harmonic spectrum.
    brainSpec_.count = BrainSpectrum::kMax;
    for (int h = 1; h <= BrainSpectrum::kMax; ++h) {
        brainSpec_.ratio[h - 1] = h;
        brainSpec_.amp[h - 1] = std::pow(static_cast<double>(h), -static_cast<double>(blk_.tilt));
    }
    bp_.spectrum = &brainSpec_;
}

void Engine::refreshScale()
{
    const int choice = static_cast<int>(std::lround(param(ParamId::Scale)));
    if (choice != scaleChoice_) {
        FixedScale s;
        if (makeBuiltinScale(choice, s)) scale_ = s;
        else if (scaleChoice_ < 0) makeBuiltinScale(0, scale_);   // the Scala and timbre slots come later: 12-TET until then
        scaleChoice_ = choice;
    }
    rootPc_ = std::clamp(static_cast<int>(std::lround(param(ParamId::RootNote))), 0, 11);
    snapKeys_ = std::lround(param(ParamId::KeyMap)) == 0;
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

void Engine::beginOn(int i, int note, float velocity, float distance, bool fromBrain, int onset)
{
    Voice& v = voices_[static_cast<size_t>(i)];
    v.note = note;
    v.fromBrain = fromBrain;
    v.velocity = velocity;
    v.distance = std::clamp(distance, 0.0f, 1.0f);
    v.env.kill();
    v.env.noteOn();
    v.seed = static_cast<std::uint32_t>(rng_.next()) | 1u;
    v.restart = true;
    v.onset = onset;
    v.released = false;
    v.tailBlocks = 1;
    v.age = 0.0;
    v.stealing = false;
    v.stealBlocks = 0;
    cloud_.voice(i).releaseAt = -1.0f;
}

void Engine::startNote(int note, float velocity, float distance, bool fromBrain, int onset)
{
    const int count = static_cast<int>(voices_.size());
    for (int i = 0; i < count; ++i)
        if (voices_[static_cast<size_t>(i)].note < 0 && !voices_[static_cast<size_t>(i)].stealing) { beginOn(i, note, velocity, distance, fromBrain, onset); return; }
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
    v.nextFromBrain = fromBrain;
    v.nextDistance = distance;
}

void Engine::stopNote(int note, bool fromBrain)
{
    for (size_t i = 0; i < voices_.size(); ++i) {
        Voice& v = voices_[i];
        if (v.stealing && v.next.type == 0 && v.next.note == note && v.nextFromBrain == fromBrain) { v.next.type = -1; return; }
        if (v.note == note && v.fromBrain == fromBrain && !v.released && !v.stealing) {
            v.released = true;
            v.env.noteOff();
            CloudVoice& cv = cloud_.voice(static_cast<int>(i));
            cv.releaseAt = cv.time;   // the motion's release tails start here
            return;
        }
    }
}

void Engine::renderBlock()
{
    const int G = cfg_.gpuBlock;
    const double dt = G / cfg_.sampleRate;
    const std::int64_t pos = renderPos_.load(std::memory_order_relaxed);
    readParams();
    refreshScale();

    // 1. The played notes that are due, in the order they came, each on its own sample.
    drainEvents();
    for (size_t i = 0; i < pending_.size();) {
        const Event e = pending_[i];
        if (e.stamp >= pos + G) { ++i; continue; }
        pending_.erase(pending_.begin() + static_cast<std::ptrdiff_t>(i));
        if (e.stamp < pos) late_.fetch_add(1, std::memory_order_relaxed);
        const int onset = static_cast<int>(std::clamp<std::int64_t>(e.stamp - pos, 0, G - 1));
        if (e.type == 0) startNote(e.note, e.velocity, blk_.keysDepth, false, onset);
        else if (e.type == 1) stopNote(e.note, false);
        else
            for (const Voice& v : voices_)
                if (v.note >= 0 && !v.fromBrain && !v.released) stopNote(v.note, false);
    }

    // 2. The conductor.
    if (stepAsked_.exchange(false, std::memory_order_relaxed)) brain_.requestStep();
    brain_.update(dt, bp_, -1,
                  [this](int note) { return frequencyOf(note); },
                  [this](const BrainEvent& e) {
                      if (e.type == BrainEvent::Type::NoteOn) startNote(e.note, e.velocity, brainDistance(), true, 0);
                      else stopNote(e.note, true);
                  });
    brainRoot_.store(brain_.root(), std::memory_order_relaxed);

    // 3. The voices: envelope, frequency, the cloud and its motion.
    const double follow = 1.0 - std::exp(-dt / kGlideSeconds);
    int sounding = 0;
    for (int i = 0; i < static_cast<int>(voices_.size()); ++i) {
        Voice& v = voices_[static_cast<size_t>(i)];
        CloudVoice& cv = cloud_.voice(i);
        if (v.stealing) {
            if (v.stealBlocks > 0) { --v.stealBlocks; cv.gain = 0.0f; v.env.kill(); ++sounding; continue; }
            v.stealing = false;
            if (v.next.type == 0) beginOn(i, v.next.note, v.next.velocity, v.nextDistance, v.nextFromBrain, 0);
            else v.note = -1;
        }
        if (v.note < 0) { cv.active = 0; continue; }
        v.env.setTimes(blk_.attack, blk_.decay, blk_.sustain, blk_.release);
        const float gain = v.env.process();
        if (!v.env.isActive()) {
            if (v.tailBlocks > 0) { --v.tailBlocks; cv.gain = 0.0f; ++sounding; continue; }
            v.note = -1;
            cv.active = 0;
            continue;
        }
        const double target = frequencyOf(v.note);
        v.hz = v.restart ? target : v.hz * std::pow(target / v.hz, follow);
        const float level = kVoiceLevel * (v.fromBrain ? 0.6f + 0.4f * v.velocity : 0.4f + 0.6f * v.velocity);
        cv.hz = static_cast<float>(v.hz);
        cv.level = level;
        cv.gain = gain;
        cv.tilt = blk_.tilt;
        cv.width = blk_.width;
        cv.widthSlope = blk_.widthSlope;
        cv.drift = blk_.drift;
        cv.driftRate = blk_.driftRate;
        cv.shimmer = blk_.shimmer;
        cv.shimmerRate = blk_.shimmerRate;
        cv.stereo = blk_.stereo;
        cv.partials = blk_.partials;
        cv.distance = v.distance;
        cv.airAbsorb = 1.0f;
        cv.seed = v.seed;
        cv.active = 1;
        cv.restart = v.restart ? 1 : 0;
        cv.onset = v.restart ? v.onset : 0;
        applyMotion(static_cast<MotionKind>(std::clamp(blk_.motion, 0, kNumMotions - 1)), cv, blk_.motionRate, blk_.motionDepth);
        v.restart = false;
        v.onset = 0;
        v.age += dt;
        ++sounding;
    }
    sounding_.store(sounding, std::memory_order_relaxed);

    // 4. The GPU: near and far, left and right.
    cloud_.launch();
    const float* q = cloud_.wait();
    gpuMs_.store(cloud_.lastGpuMs(), std::memory_order_relaxed);
    for (int s = 0; s < G; ++s) {
        nearL_[static_cast<size_t>(s)] = q[kCloudChannels * s];
        nearR_[static_cast<size_t>(s)] = q[kCloudChannels * s + 1];
        farL_[static_cast<size_t>(s)] = q[kCloudChannels * s + 2] * blk_.farLevel;
        farR_[static_cast<size_t>(s)] = q[kCloudChannels * s + 3] * blk_.farLevel;
    }

    // 5. The far plane is heard only through the far reverb; the near plane stays dry.
    far_.set(param(ParamId::FarSize), param(ParamId::FarDecay), param(ParamId::FarDamp), 20.0f, false, 1.0f);
    far_.setSpace(0.5f, 9000.0f, 60.0f);
    far_.process(farL_.data(), farR_.data(), G);
    for (int s = 0; s < G; ++s) {
        nearL_[static_cast<size_t>(s)] += farL_[static_cast<size_t>(s)];
        nearR_[static_cast<size_t>(s)] += farR_[static_cast<size_t>(s)];
    }
    midSide_.set(param(ParamId::BassMono), 0.0f, param(ParamId::StereoWidth));
    midSide_.process(nearL_.data(), nearR_.data(), G);
    for (int s = 0; s < G; ++s) {
        const size_t at = static_cast<size_t>((pos + s) & ringMask_) * 2;
        ring_[at] = softClip(nearL_[static_cast<size_t>(s)] * blk_.master);
        ring_[at + 1] = softClip(nearR_[static_cast<size_t>(s)] * blk_.master);
    }
    renderPos_.store(pos + G, std::memory_order_release);
}

} // namespace additiv
