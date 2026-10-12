/**
 * @file main.cpp
 * @brief probe: does the GPU partial cloud keep real time on this machine, and what do its
 *        partials and their envelopes sound like?
 *
 *     probe info               the device
 *     probe selftest           partials against exact sines, levels, envelopes, glides, pan, landscape, determinism
 *     probe bench   [opts]     blocks back to back: GPU time, round trip, how far ahead of real time
 *     probe latency [opts]     blocks paced at real time from a Pro Audio thread, each waited for at
 *                              once: what an audio callback would see if it rendered on the GPU itself
 *     probe ahead   [opts]     the planned design: a worker renders ahead into a ring, a paced Pro
 *                              Audio consumer takes one block per period and counts underruns
 *     probe render  [opts]     one chord at several partial counts (and a --scene), as WAVs
 *     probe scenes  [opts]     every envelope scene on the same chord, as WAVs
 *
 * Options: --block N (128)  --voices N (16)  --partials N[,N..] (20000)  --seconds S
 *          --lookahead MS (10)  --wait spin|block  --csv FILE  --out DIR
 *          --width CENTS (25)  --slope X (0.3)  --drift CENTS (3)  --shimmer X (0.4)
 *          --gpu-priority idle|below|normal|above|high|realtime  (the process's GPU scheduling class)
 *          --scene NAME (none)  --landscape FILE (picture or WAV/FLAC)  --loop S (30, a picture's length)
 *          --range DB (40)  --whiten X (0.7)  --seamless X (0.1)  --smooth S (0.15)
 *          --reservoir  (the reservoir stepped every block too)  --melt  (the melt running too)
 */
#define NOMINMAX
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <avrt.h>
#include <timeapi.h>

#include "additiv/Cloud.h"
#include "additiv/Landscape.h"
#include "additiv/Motion.h"
#include "additiv/Shapes.h"

#include <algorithm>
#include <atomic>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <thread>
#include <vector>

namespace fs = std::filesystem;
using namespace additiv;

namespace {

constexpr double kSampleRate = 48000.0;   ///< every probe runs at 48 kHz

/** @brief The command line; every subcommand reads what it needs. */
struct Options {
    int block = 128;                       ///< samples per block
    int voices = 16;                       ///< voices sounding
    std::vector<int> partials{ 20000 };    ///< partials per voice (render and bench take a list)
    double seconds = 0.0;                  ///< run length; 0 = the subcommand's default
    double lookaheadMs = 10.0;             ///< ahead: how far the worker renders ahead
    bool spin = true;                      ///< wait for the GPU by spinning (else sleep on the event)
    std::string csv;                       ///< latency: every block's times into this file
    std::string out;                       ///< render, scenes: the folder for the WAVs
    float width = 25.0f;                   ///< cloud width at harmonic 1, cents
    float slope = 0.3f;                    ///< how the width grows with the harmonic number
    float drift = 3.0f;                    ///< each partial's own pitch wander, cents
    float shimmer = 0.4f;                  ///< each partial's own amplitude wander, 0 .. 1
    int gpuPriority = -1;                  ///< the process's GPU scheduling class (D3DKMT, 0 idle .. 5 realtime); -1 = leave it
    std::string scene = "none";            ///< which envelope scene the voices play
    std::string landscape;                 ///< a picture or a recording for the landscape scenes
    float loopSeconds = 30.0f;             ///< how long a picture (or the generated landscape) lasts
    float rangeDb = 40.0f;                 ///< the decibel range of a landscape's brightness
    float whiten = 0.7f;                   ///< how far a recording's columns lose their own colour
    float seamless = 0.1f;                 ///< the share of a recording crossfaded into its beginning
    float smooth = 0.15f;                  ///< seconds a recording's landscape is smoothed over
    bool reservoir = false;                ///< the reservoir stepped every block too
    bool melt = false;                     ///< the melt running too
};

/**
 * @brief Sets this process's GPU scheduling priority class, which decides how the GPU's scheduler
 *        weighs our work against other processes' (D3DKMTSetProcessSchedulingPriorityClass, from
 *        gdi32 at run time so nothing links against the kernel-mode thunks).
 * @param cls  0 idle, 1 below normal, 2 normal, 3 above normal, 4 high, 5 realtime
 * @return     the NTSTATUS, 0 when it took
 */
long setGpuPriority(int cls)
{
    using SetClass = long (APIENTRY*)(HANDLE, int);
    HMODULE gdi = GetModuleHandleW(L"gdi32.dll");
    if (!gdi) gdi = LoadLibraryW(L"gdi32.dll");
    const auto fn = gdi ? reinterpret_cast<SetClass>(reinterpret_cast<void*>(GetProcAddress(gdi, "D3DKMTSetProcessSchedulingPriorityClass"))) : nullptr;
    return fn ? fn(GetCurrentProcess(), cls) : -1;
}

/** @brief QueryPerformanceCounter's ticks per second. @return ticks per second */
double qpcFreq()
{
    static const double f = [] { LARGE_INTEGER q; QueryPerformanceFrequency(&q); return static_cast<double>(q.QuadPart); }();
    return f;
}

/** @brief Now, in QueryPerformanceCounter ticks. @return ticks */
std::int64_t qpc()
{
    LARGE_INTEGER q;
    QueryPerformanceCounter(&q);
    return q.QuadPart;
}

/** @brief Ticks to microseconds. @param ticks QueryPerformanceCounter ticks @return microseconds */
double toUs(std::int64_t ticks) { return static_cast<double>(ticks) * 1.0e6 / qpcFreq(); }

/** @brief Registers the calling thread with MMCSS as "Pro Audio", as a real audio thread is, while it lives. */
struct ProAudio {
    HANDLE task = nullptr;   ///< the MMCSS registration, null when it failed

    /** @brief Registers the thread. @param priority the MMCSS priority within Pro Audio */
    explicit ProAudio(AVRT_PRIORITY priority)
    {
        DWORD index = 0;
        task = AvSetMmThreadCharacteristicsW(L"Pro Audio", &index);
        if (task) AvSetMmThreadPriority(task, priority);
    }
    ~ProAudio() { if (task) AvRevertMmThreadCharacteristics(task); }
    ProAudio(const ProAudio&) = delete;              ///< one registration per thread
    ProAudio& operator=(const ProAudio&) = delete;   ///< one registration per thread
};

/** @brief Sleeps until a moment with a high-resolution waitable timer and spins the last 300 µs. */
class Pacer {
public:
    Pacer() : timer_(CreateWaitableTimerExW(nullptr, nullptr, CREATE_WAITABLE_TIMER_HIGH_RESOLUTION, TIMER_ALL_ACCESS)) {}
    ~Pacer() { if (timer_) CloseHandle(timer_); }
    Pacer(const Pacer&) = delete;              ///< owns the timer
    Pacer& operator=(const Pacer&) = delete;   ///< owns the timer

    /** @brief Returns at `target` (or at once when it has passed). @param target QueryPerformanceCounter ticks */
    void until(std::int64_t target)
    {
        const std::int64_t spinTicks = static_cast<std::int64_t>(qpcFreq() * 300.0e-6);
        for (;;) {
            const std::int64_t left = target - qpc();
            if (left <= 0) return;
            if (left > spinTicks && timer_) {
                LARGE_INTEGER due;
                due.QuadPart = -static_cast<LONGLONG>(static_cast<double>(left - spinTicks) * 1.0e7 / qpcFreq());
                if (due.QuadPart < 0 && SetWaitableTimerEx(timer_, &due, 0, nullptr, nullptr, nullptr, 0)) {
                    WaitForSingleObject(timer_, INFINITE);
                    continue;
                }
            }
            YieldProcessor();
        }
    }

private:
    HANDLE timer_;   ///< the high-resolution timer, null on systems without one (then it spins)
};

/** @brief Mean, percentiles and maximum of a series. */
struct Summary {
    double mean = 0.0;   ///< the mean
    double p50 = 0.0;    ///< the median
    double p99 = 0.0;    ///< 99th percentile
    double p999 = 0.0;   ///< 99.9th percentile
    double max = 0.0;    ///< the worst
};

/** @brief Summarises a series. @param v the values (copied, then sorted) @return the summary */
Summary summarize(std::vector<double> v)
{
    Summary s;
    if (v.empty()) return s;
    std::sort(v.begin(), v.end());
    double sum = 0.0;
    for (double x : v) sum += x;
    s.mean = sum / static_cast<double>(v.size());
    auto at = [&](double q) { return v[std::min(v.size() - 1, static_cast<size_t>(q * static_cast<double>(v.size())))]; };
    s.p50 = at(0.5);
    s.p99 = at(0.99);
    s.p999 = at(0.999);
    s.max = v.back();
    return s;
}

/** @brief Prints one summary line in microseconds. @param label what was measured @param s the summary */
void printSummary(const char* label, const Summary& s)
{
    std::printf("  %-12s mean %8.1f   p50 %8.1f   p99 %8.1f   p99.9 %8.1f   max %9.1f  us\n",
                label, s.mean, s.p50, s.p99, s.p999, s.max);
}

/** @brief Prints how many values exceed each threshold. @param v microseconds @param label what they are */
void printOver(const std::vector<double>& v, const char* label)
{
    static const double kMs[] = { 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0 };
    std::printf("  %s over:", label);
    for (double ms : kMs) {
        const auto count = std::count_if(v.begin(), v.end(), [&](double x) { return x > ms * 1000.0; });
        std::printf("  %g ms: %lld", ms, static_cast<long long>(count));
    }
    std::printf("\n");
}

/** @brief The repository root: the first folder above the program that holds CMakeLists.txt and Core. @return the root (the working directory as a fallback) */
fs::path repoRoot()
{
    wchar_t buf[MAX_PATH] = {};
    GetModuleFileNameW(nullptr, buf, MAX_PATH);
    fs::path p = fs::path(buf).parent_path();
    for (int up = 0; up < 5 && !p.empty(); ++up) {
        if (fs::exists(p / "CMakeLists.txt") && fs::exists(p / "Core")) return p;
        p = p.parent_path();
    }
    return fs::current_path();
}

/** @brief MIDI note to Hz. @param note MIDI note number @return Hz */
float noteHz(int note) { return 440.0f * std::exp2((static_cast<float>(note) - 69.0f) / 12.0f); }

/**
 * @brief The GPU's four channels (near L, near R, far L, far R) as stereo: near and far added.
 * @param q   n frames of kCloudOutChannels (the modal hall's two are left out: the probe keeps it resting)
 * @param n   frames
 * @param lr  receives n interleaved stereo frames
 */
void mixToStereo(const float* q, int n, float* lr)
{
    for (int i = 0; i < n; ++i) {
        lr[2 * i] = q[kCloudOutChannels * i] + q[kCloudOutChannels * i + 2];
        lr[2 * i + 1] = q[kCloudOutChannels * i + 1] + q[kCloudOutChannels * i + 3];
    }
}

/**
 * @brief Appends a GPU block to an interleaved stereo buffer, near and far added.
 * @param lr  the buffer
 * @param q   n frames of kCloudOutChannels (the modal hall's two are left out: the probe keeps it resting)
 * @param n   frames
 */
void appendStereo(std::vector<float>& lr, const float* q, int n)
{
    const size_t at = lr.size();
    lr.resize(at + static_cast<size_t>(n) * 2);
    mixToStereo(q, n, lr.data() + at);
}

// ---------------------------------------------------------------------------------------------
// Scenes: the core's motions (Motion.h), plus the recording as a landscape.

/** @brief A scene: its name, one line about it, its motion and where its landscape comes from. */
struct Scene {
    const char* name;      ///< the --scene name
    const char* what;      ///< one line for the log
    int landscape;         ///< 0 none, 1 the generated aurora (or --landscape), 2 a recording (or --landscape)
    MotionKind motion;     ///< what it sets on a voice's lanes
};

/** @brief Every scene, in the order `probe scenes` renders them. */
const Scene kScenes[] = {
    { "none", "the cloud alone, no envelopes", 0, MotionKind::None },
    { "random", "every partial for itself: random shape, rate and phase", 0, MotionKind::Random },
    { "wave", "a wave climbing through the spectrum", 0, MotionKind::Wave },
    { "pendulum", "a pendulum wave across the harmonics, over a bass that stays", 0, MotionKind::Pendulum },
    { "breathe", "loudness, place and pitch travelling through each cloud", 0, MotionKind::Breathe },
    { "deepnote", "every partial glides into the chord from up to 15 semitones away", 0, MotionKind::Glissando },
    { "aurora", "the amplitude lane reads a painted landscape", 1, MotionKind::Aurora },
    { "recording", "the amplitude lane reads a recording's spectrum (sea)", 2, MotionKind::Aurora },
};

/** @brief A scene by name. @param name the --scene name @return the scene, null when unknown */
const Scene* findScene(const std::string& name)
{
    for (const Scene& s : kScenes)
        if (name == s.name) return &s;
    return nullptr;
}

/** @brief The default recording for the recording scene: the first sea in Noctuary's field recordings. @return the file, empty when none is there */
fs::path defaultRecording()
{
    const fs::path dir = repoRoot().parent_path() / "AmbientSynth" / "Library" / "FieldRecordings";
    std::error_code ec;
    std::vector<fs::path> seas;
    for (const auto& entry : fs::directory_iterator(dir, ec))
        if (entry.path().filename().string().rfind("fr-sea-", 0) == 0) seas.push_back(entry.path());
    std::sort(seas.begin(), seas.end());
    return seas.empty() ? fs::path() : seas.front();
}

/**
 * @brief Gives an engine the factory shapes and, when the scene reads one, its landscape; draws the
 *        landscape into work/landscapes so it can be looked at (and painted over).
 * @param e  the engine
 * @param o  the options (scene, landscape file, loop, range, whiten, seamless)
 * @return   true when the engine is ready for the scene
 */
bool prepareEngine(CloudEngine& e, const Options& o)
{
    const std::vector<EnvShape> bank = factoryShapes();
    if (!e.setShapes(bank.data(), static_cast<int>(bank.size()))) { std::fprintf(stderr, "probe: the shapes did not upload\n"); return false; }
    const Scene* scene = findScene(o.scene);
    if (!scene || scene->landscape == 0) return true;

    Landscape land;
    std::string error, name;
    fs::path file = o.landscape;
    if (file.empty() && scene->landscape == 2) {
        file = defaultRecording();
        if (file.empty()) { std::fprintf(stderr, "probe: no recording found (Noctuary's FieldRecordings); pass --landscape FILE\n"); return false; }
    }
    if (file.empty()) {
        land = landscapeAurora(o.loopSeconds, o.rangeDb);
        name = "aurora";
    } else {
        std::string ext = file.extension().string();
        std::transform(ext.begin(), ext.end(), ext.begin(), [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
        const bool audio = ext == ".wav" || ext == ".flac";
        const bool good = audio ? landscapeFromAudio(file, o.rangeDb, o.whiten, o.seamless, o.smooth, land, &error)
                                : landscapeFromImage(file, o.loopSeconds, o.rangeDb, land, &error);
        if (!good) { std::fprintf(stderr, "probe: %s\n", error.c_str()); return false; }
        name = file.stem().string();
    }
    if (!e.setLandscape(land.amps.data(), land.columns, land.rows, land.rowSeconds)) { std::fprintf(stderr, "probe: the landscape did not upload\n"); return false; }
    const fs::path pictures = repoRoot() / "work" / "landscapes";
    fs::create_directories(pictures);
    const fs::path picture = pictures / (name + ".bmp");
    writeLandscapeBmp(land, picture, o.rangeDb, nullptr);
    std::printf("landscape %s: %d columns x %d rows, %.1f s loop -> %s\n", name.c_str(), land.columns, land.rows,
                land.rows * land.rowSeconds, picture.string().c_str());
    return true;
}

/**
 * @brief One voice set up as a note: the cloud parameters from the options, the scene's lanes, full gain, a fresh start.
 * @param v         the voice
 * @param hz        its pitch
 * @param partials  partials per voice
 * @param seed      the note's dice
 * @param o         the options (width, slope, drift, shimmer, scene)
 */
void setNote(CloudVoice& v, float hz, int partials, std::uint32_t seed, const Options& o)
{
    v = CloudVoice{};
    v.hz = hz;
    v.partials = partials;
    v.seed = seed;
    v.width = o.width;
    v.widthSlope = o.slope;
    v.drift = o.drift;
    v.shimmer = o.shimmer;
    v.level = 0.05f;
    v.gain = 1.0f;
    v.active = 1;
    v.restart = 1;
    if (const Scene* s = findScene(o.scene)) applyMotion(s->motion, v, 1.0f, 1.0f);
}

/**
 * @brief The load of the timing probes: every voice sounding, spread over four octaves from A1.
 * @param e         the engine
 * @param partials  partials per voice
 * @param o         the options
 */
void setLoad(CloudEngine& e, int partials, const Options& o)
{
    for (int i = 0; i < e.config().voices; ++i)
        setNote(e.voice(i), noteHz(33 + (i * 7) % 48), partials, 100u + static_cast<std::uint32_t>(i), o);
    e.reservoir().on = o.reservoir ? 1 : 0;
    e.melt().on = o.melt ? 1 : 0;
    e.melt().widen = o.melt ? 0.05f : 0.0f;
}

/**
 * @brief Opens an engine or says why not.
 * @param e  the engine
 * @param c  the layout
 * @return   true when it is open
 */
bool openEngine(CloudEngine& e, const CloudConfig& c)
{
    std::string error;
    if (e.open(c, &error)) return true;
    std::fprintf(stderr, "probe: the GPU engine did not open: %s\n", error.c_str());
    return false;
}

/** @brief The layout the options describe. @param o the options @param partials partials per voice @return the layout */
CloudConfig layout(const Options& o, int partials)
{
    CloudConfig c;
    c.sampleRate = kSampleRate;
    c.blockSize = o.block;
    c.voices = o.voices;
    c.partialsPerVoice = partials;
    c.spinWait = o.spin;
    return c;
}

/** @brief probe info. @param o the options @return the exit code */
int cmdInfo(const Options& o)
{
    CloudEngine e;
    if (!openEngine(e, layout(o, o.partials[0]))) return 1;
    std::printf("%s\n", e.deviceName().c_str());
    return 0;
}

/** @brief probe bench: blocks back to back. @param o the options @return the exit code */
int cmdBench(const Options& o)
{
    const int blocks = o.seconds > 0.0 ? static_cast<int>(o.seconds * kSampleRate / o.block) : 3000;
    const double blockUs = o.block / kSampleRate * 1.0e6;
    bool first = true;
    for (int partials : o.partials) {
        CloudEngine e;
        if (!openEngine(e, layout(o, partials)) || !prepareEngine(e, o)) return 1;
        if (first) { std::printf("%s, block %d (%.0f us), %d voices, %s wait, scene %s\n", e.deviceName().c_str(), o.block, blockUs, o.voices, o.spin ? "spin" : "block", o.scene.c_str()); first = false; }
        setLoad(e, partials, o);
        for (int b = 0; b < 100; ++b) { e.launch(); e.wait(); }
        std::vector<double> rtt, gpu;
        rtt.reserve(blocks);
        gpu.reserve(blocks);
        for (int b = 0; b < blocks; ++b) {
            const std::int64_t t0 = qpc();
            e.launch();
            e.wait();
            rtt.push_back(toUs(qpc() - t0));
            gpu.push_back(e.lastGpuMs() * 1000.0);
        }
        const Summary r = summarize(rtt), g = summarize(gpu);
        std::printf("%7d partials x %d = %9lld: GPU %7.1f us, round trip %7.1f us (p99 %7.1f, max %8.1f), %6.1fx real time\n",
                    partials, o.voices, static_cast<long long>(partials) * o.voices, g.mean, r.mean, r.p99, r.max, r.mean > 0.0 ? blockUs / r.mean : 0.0);
    }
    return 0;
}

/** @brief probe latency: paced blocks, each waited for at once. @param o the options @return the exit code */
int cmdLatency(const Options& o)
{
    const int partials = o.partials[0];
    CloudEngine e;
    if (!openEngine(e, layout(o, partials)) || !prepareEngine(e, o)) return 1;
    setLoad(e, partials, o);
    for (int b = 0; b < 200; ++b) { e.launch(); e.wait(); }

    const double seconds = o.seconds > 0.0 ? o.seconds : 60.0;
    const int blocks = static_cast<int>(seconds * kSampleRate / o.block);
    const double periodUs = o.block / kSampleRate * 1.0e6;
    const std::int64_t periodTicks = static_cast<std::int64_t>(qpcFreq() * o.block / kSampleRate);
    std::printf("%s: %d voices x %d partials, scene %s, block %d = %.0f us, %.0f s, %s wait, Pro Audio thread\n",
                e.deviceName().c_str(), o.voices, partials, o.scene.c_str(), o.block, periodUs, seconds, o.spin ? "spin" : "block");

    std::vector<double> rtt, gpu, wake, when;
    rtt.reserve(blocks); gpu.reserve(blocks); wake.reserve(blocks); when.reserve(blocks);
    int resyncs = 0;
    timeBeginPeriod(1);
    {
        ProAudio pa(AVRT_PRIORITY_CRITICAL);
        if (!pa.task) std::printf("  (MMCSS Pro Audio not available: running at normal priority)\n");
        Pacer pacer;
        const std::int64_t start = qpc();
        std::int64_t next = start + periodTicks;
        double worstSoFar = 0.0;
        int reportAt = static_cast<int>(10.0 * kSampleRate / o.block);
        for (int b = 0; b < blocks; ++b) {
            pacer.until(next);
            const std::int64_t t0 = qpc();
            e.launch();
            e.wait();
            const std::int64_t t1 = qpc();
            rtt.push_back(toUs(t1 - t0));
            gpu.push_back(e.lastGpuMs() * 1000.0);
            wake.push_back(toUs(t0 - next));
            when.push_back(toUs(t0 - start) * 1.0e-6);
            worstSoFar = std::max(worstSoFar, rtt.back());
            next += periodTicks;
            if (qpc() > next + periodTicks) { next = qpc() + periodTicks; ++resyncs; }
            if (b + 1 == reportAt) {
                std::printf("  %4.0f s: worst round trip so far %.1f us\n", when.back(), worstSoFar);
                std::fflush(stdout);
                reportAt += static_cast<int>(10.0 * kSampleRate / o.block);
            }
        }
    }
    timeEndPeriod(1);

    printSummary("round trip", summarize(rtt));
    printSummary("GPU", summarize(gpu));
    printSummary("wake-up late", summarize(wake));
    printOver(rtt, "round trips");
    const auto missed = std::count_if(rtt.begin(), rtt.end(), [&](double x) { return x > periodUs; });
    const auto half = std::count_if(rtt.begin(), rtt.end(), [&](double x) { return x > 0.5 * periodUs; });
    std::printf("  longer than the block (a certain dropout in the callback): %lld, longer than half of it: %lld, fell behind: %d\n",
                static_cast<long long>(missed), static_cast<long long>(half), resyncs);
    std::vector<size_t> order(rtt.size());
    for (size_t i = 0; i < order.size(); ++i) order[i] = i;
    const size_t worst = std::min<size_t>(10, order.size());
    std::partial_sort(order.begin(), order.begin() + static_cast<std::ptrdiff_t>(worst), order.end(), [&](size_t a, size_t b) { return rtt[a] > rtt[b]; });
    std::printf("  worst blocks:");
    for (size_t i = 0; i < worst; ++i) std::printf("  %.1f us @ %.1f s", rtt[order[i]], when[order[i]]);
    std::printf("\n");

    if (!o.csv.empty()) {
        std::ofstream f(o.csv);
        f << "t_s,round_trip_us,gpu_us,wake_late_us\n";
        for (size_t i = 0; i < rtt.size(); ++i) f << when[i] << ',' << rtt[i] << ',' << gpu[i] << ',' << wake[i] << '\n';
        std::printf("  every block: %s\n", o.csv.c_str());
    }
    return 0;
}

/** @brief probe ahead: a worker renders ahead into a ring, a paced consumer takes from it. @param o the options @return the exit code */
int cmdAhead(const Options& o)
{
    const int partials = o.partials[0];
    CloudEngine e;
    if (!openEngine(e, layout(o, partials)) || !prepareEngine(e, o)) return 1;
    setLoad(e, partials, o);
    for (int b = 0; b < 200; ++b) { e.launch(); e.wait(); }

    const double seconds = o.seconds > 0.0 ? o.seconds : 60.0;
    const int blocks = static_cast<int>(seconds * kSampleRate / o.block);
    const double periodMs = o.block / kSampleRate * 1000.0;
    const int ring = std::max(2, static_cast<int>(std::ceil(o.lookaheadMs / periodMs)));
    const std::int64_t periodTicks = static_cast<std::int64_t>(qpcFreq() * o.block / kSampleRate);
    std::printf("%s: %d voices x %d partials, scene %s, block %d (%.2f ms), ring %d blocks = %.1f ms ahead, %.0f s\n",
                e.deviceName().c_str(), o.voices, partials, o.scene.c_str(), o.block, periodMs, ring, ring * periodMs, seconds);

    const size_t frames = static_cast<size_t>(o.block) * 2;
    std::vector<float> slots(frames * ring);
    std::atomic<std::uint64_t> written{ 0 }, read{ 0 };
    std::atomic<bool> running{ true };
    HANDLE consumed = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    std::vector<double> workerUs;
    workerUs.reserve(static_cast<size_t>(blocks) + ring + 16);

    std::thread worker([&] {
        ProAudio pa(AVRT_PRIORITY_HIGH);
        while (running.load(std::memory_order_relaxed)) {
            const std::uint64_t w = written.load(std::memory_order_relaxed);
            if (w - read.load(std::memory_order_acquire) < static_cast<std::uint64_t>(ring)) {
                const std::int64_t t0 = qpc();
                e.launch();
                const float* block = e.wait();
                mixToStereo(block, o.block, &slots[(w % ring) * frames]);
                workerUs.push_back(toUs(qpc() - t0));
                written.store(w + 1, std::memory_order_release);
            } else {
                WaitForSingleObject(consumed, 2);
            }
        }
    });
    while (written.load(std::memory_order_acquire) < static_cast<std::uint64_t>(ring)) Sleep(1);

    int underruns = 0;
    std::uint64_t minFill = ring;
    std::vector<double> underrunAt;
    volatile float sink = 0.0f;
    timeBeginPeriod(1);
    {
        ProAudio pa(AVRT_PRIORITY_CRITICAL);
        Pacer pacer;
        const std::int64_t start = qpc();
        std::int64_t next = start + periodTicks;
        for (int b = 0; b < blocks; ++b) {
            pacer.until(next);
            const std::uint64_t r = read.load(std::memory_order_relaxed);
            const std::uint64_t fill = written.load(std::memory_order_acquire) - r;
            minFill = std::min(minFill, fill);
            if (fill == 0) {
                ++underruns;
                underrunAt.push_back(toUs(qpc() - start) * 1.0e-6);
            } else {
                sink = sink + slots[(r % ring) * frames];
                read.store(r + 1, std::memory_order_release);
            }
            SetEvent(consumed);
            next += periodTicks;
            if (qpc() > next + periodTicks) next = qpc() + periodTicks;
        }
    }
    timeEndPeriod(1);
    running = false;
    SetEvent(consumed);
    worker.join();
    CloseHandle(consumed);

    std::printf("  underruns: %d of %d blocks; fullest emergency: %llu of %d blocks left (%.2f ms)\n",
                underruns, blocks, static_cast<unsigned long long>(minFill), ring, static_cast<double>(minFill) * periodMs);
    if (!underrunAt.empty()) {
        std::printf("  underruns at:");
        for (size_t i = 0; i < std::min<size_t>(10, underrunAt.size()); ++i) std::printf(" %.1f s", underrunAt[i]);
        std::printf("\n");
    }
    printSummary("worker block", summarize(workerUs));
    printOver(workerUs, "worker blocks");
    return underruns == 0 ? 0 : 2;
}

/**
 * @brief Writes interleaved stereo as a 24-bit WAV.
 * @param path        the file
 * @param lr          interleaved stereo samples
 * @param sampleRate  Hz
 * @param gain        applied before quantising
 * @return            true when written
 */
bool writeWav24(const fs::path& path, const std::vector<float>& lr, int sampleRate, float gain)
{
    const std::uint32_t dataBytes = static_cast<std::uint32_t>(lr.size() * 3);
    std::vector<unsigned char> bytes;
    bytes.reserve(44 + dataBytes);
    auto tag = [&](const char* t) { bytes.insert(bytes.end(), t, t + 4); };
    auto u32 = [&](std::uint32_t x) { for (int i = 0; i < 4; ++i) bytes.push_back(static_cast<unsigned char>(x >> (8 * i))); };
    auto u16 = [&](std::uint16_t x) { bytes.push_back(static_cast<unsigned char>(x)); bytes.push_back(static_cast<unsigned char>(x >> 8)); };
    tag("RIFF"); u32(36 + dataBytes); tag("WAVE");
    tag("fmt "); u32(16); u16(1); u16(2); u32(static_cast<std::uint32_t>(sampleRate)); u32(static_cast<std::uint32_t>(sampleRate) * 6); u16(6); u16(24);
    tag("data"); u32(dataBytes);
    for (float x : lr) {
        const float y = std::clamp(x * gain, -1.0f, 1.0f);
        const std::int32_t q = static_cast<std::int32_t>(std::lround(y * 8388607.0f));
        bytes.push_back(static_cast<unsigned char>(q));
        bytes.push_back(static_cast<unsigned char>(q >> 8));
        bytes.push_back(static_cast<unsigned char>(q >> 16));
    }
    std::ofstream f(path, std::ios::binary);
    f.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    return static_cast<bool>(f);
}

constexpr int kChord[] = { 45, 52, 59, 60, 64 };   ///< A2 E3 B3 C4 E4: A minor add 9, the chord of every render
constexpr int kChordVoices = static_cast<int>(sizeof(kChord) / sizeof(kChord[0]));   ///< voices of the chord

/**
 * @brief Renders the chord on an engine with kChordVoices voices: the scene's lanes, a raised-cosine
 *        attack, the release (note-off for the lanes, a fade for the voices) over the last stretch,
 *        normalised to -20 dBFS RMS (peaks kept below -1 dBFS), as a WAV.
 * @param e         the engine, prepared for the scene
 * @param partials  partials per voice
 * @param o         the options (cloud parameters, scene)
 * @param seconds   length
 * @param file      the WAV
 * @return          true when written
 */
bool renderChord(CloudEngine& e, int partials, const Options& o, double seconds, const fs::path& file)
{
    const int n = e.config().blockSize;
    const double attack = std::min(4.0, seconds / 4.0), release = std::min(8.0, seconds / 4.0);
    for (int i = 0; i < kChordVoices; ++i) setNote(e.voice(i), noteHz(kChord[i]), partials, 1000u + static_cast<std::uint32_t>(i), o);
    const int blocks = static_cast<int>(seconds * kSampleRate / n);
    std::vector<float> lr;
    lr.reserve(static_cast<size_t>(blocks) * n * 2);
    const std::int64_t t0 = qpc();
    for (int b = 0; b < blocks; ++b) {
        const double t = (b + 1) * n / kSampleRate;
        double env = 1.0;
        if (t < attack) env = 0.5 - 0.5 * std::cos(3.14159265358979 * t / attack);
        else if (t > seconds - release) env = 0.5 + 0.5 * std::cos(3.14159265358979 * std::min(1.0, (t - (seconds - release)) / release));
        for (int i = 0; i < kChordVoices; ++i) {
            CloudVoice& v = e.voice(i);
            v.gain = static_cast<float>(env);
            if (t > seconds - release && v.releaseAt < 0.0f) v.releaseAt = v.time;
        }
        e.launch();
        const float* block = e.wait();
        appendStereo(lr, block, n);
    }
    const double renderS = toUs(qpc() - t0) * 1.0e-6;
    double sq = 0.0;
    float peak = 0.0f;
    for (float x : lr) { sq += static_cast<double>(x) * x; peak = std::max(peak, std::fabs(x)); }
    const double rms = std::sqrt(sq / static_cast<double>(std::max<size_t>(lr.size(), 1)));
    float gain = static_cast<float>(0.1 / std::max(rms, 1.0e-9));
    gain = std::min(gain, 0.891f / std::max(peak, 1.0e-9f));
    if (!writeWav24(file, lr, static_cast<int>(kSampleRate), gain)) { std::fprintf(stderr, "probe: cannot write %s\n", file.string().c_str()); return false; }
    std::printf("         raw %.1f dBFS RMS, peak %.1f dBFS, gain %+.1f dB, %.0f s rendered in %.2f s -> %s\n",
                20.0 * std::log10(std::max(rms, 1.0e-9)), 20.0 * std::log10(std::max(peak, 1.0e-9f)), 20.0 * std::log10(gain), seconds, renderS, file.string().c_str());
    return true;
}

/** @brief The chord's engine: kChordVoices voices, 256-sample blocks. @param partials the most partials per voice @return the layout */
CloudConfig chordLayout(int partials)
{
    CloudConfig c;
    c.sampleRate = kSampleRate;
    c.blockSize = 256;
    c.voices = kChordVoices;
    c.partialsPerVoice = partials;
    return c;
}

/** @brief probe render: the chord at every partial count of --partials, in the --scene, each a WAV. @param o the options @return the exit code */
int cmdRender(const Options& o)
{
    const double seconds = o.seconds > 0.0 ? o.seconds : 30.0;
    const fs::path out = o.out.empty() ? repoRoot() / "work" / "renders" / "cloud" : fs::path(o.out);
    fs::create_directories(out);
    CloudEngine e;
    if (!openEngine(e, chordLayout(*std::max_element(o.partials.begin(), o.partials.end()))) || !prepareEngine(e, o)) return 1;
    std::printf("%s: chord A2 E3 B3 C4 E4, scene %s, %.0f s, width %.0f ct (slope %.2f), drift %.1f ct, shimmer %.2f\n",
                e.deviceName().c_str(), o.scene.c_str(), seconds, o.width, o.slope, o.drift, o.shimmer);
    for (int partials : o.partials) {
        const std::string stem = (o.scene == "none" ? std::string("cloud") : o.scene) + "-" + std::to_string(partials);
        std::printf("%6d partials per voice:", partials);
        for (int i = 0; i < kChordVoices; ++i) setNote(e.voice(i), noteHz(kChord[i]), partials, 1u, o);
        e.launch(); e.wait();   // one block to learn the voices' harmonics and members
        for (int i = 0; i < kChordVoices; ++i) std::printf("  %d x %d", e.voice(i).harmonics, e.voice(i).members);
        std::printf("  (harmonics x members)\n");
        if (!renderChord(e, partials, o, seconds, out / (stem + ".wav"))) return 1;
    }
    return 0;
}

/** @brief probe scenes: every scene on the chord with the first --partials, each a WAV. @param o the options @return the exit code */
int cmdScenes(const Options& o)
{
    const double seconds = o.seconds > 0.0 ? o.seconds : 40.0;
    const fs::path out = o.out.empty() ? repoRoot() / "work" / "renders" / "scenes" : fs::path(o.out);
    fs::create_directories(out);
    const int partials = o.partials[0];
    for (const Scene& s : kScenes) {
        Options so = o;
        so.scene = s.name;
        CloudEngine e;
        if (!openEngine(e, chordLayout(partials))) return 1;
        std::printf("%-10s %s\n", s.name, s.what);
        if (!prepareEngine(e, so)) { std::printf("           skipped\n"); continue; }
        if (!renderChord(e, partials, so, seconds, out / (std::string(s.name) + ".wav"))) return 1;
    }
    return 0;
}

/**
 * @brief Renders `blocks` blocks of an engine's voices into one interleaved buffer.
 * @param e       the engine
 * @param blocks  how many
 * @return        the samples
 */
std::vector<float> renderBlocks(CloudEngine& e, int blocks)
{
    const int n = e.config().blockSize;
    std::vector<float> lr;
    lr.reserve(static_cast<size_t>(blocks) * n * 2);
    for (int b = 0; b < blocks; ++b) {
        e.launch();
        const float* block = e.wait();
        appendStereo(lr, block, n);
    }
    return lr;
}

/**
 * @brief One partial at 1500 Hz -- four whole cycles per 128-sample block, so a block's RMS is
 *        exact -- centred, without drift or shimmer, at level 0.1 (amplitude 0.1 per channel).
 * @param bank    the shapes to upload
 * @param land    a landscape to upload (may be null)
 * @param blocks  how many blocks
 * @param setup   sets the voice's lanes (and release)
 * @return        the samples, interleaved stereo
 */
template <class Setup>
std::vector<float> onePartial(const std::vector<EnvShape>& bank, const Landscape* land, int blocks, Setup&& setup)
{
    CloudConfig c;
    c.sampleRate = kSampleRate;
    c.blockSize = 128;
    c.voices = 1;
    c.partialsPerVoice = 1;
    CloudEngine e;
    if (!openEngine(e, c)) return {};
    e.setShapes(bank.data(), static_cast<int>(bank.size()));
    if (land) e.setLandscape(land->amps.data(), land->columns, land->rows, land->rowSeconds);
    CloudVoice& v = e.voice(0);
    v = CloudVoice{};
    v.hz = 1500.0f;
    v.partials = 1;
    v.seed = 7u;
    v.drift = 0.0f;
    v.shimmer = 0.0f;
    v.stereo = 0.0f;
    v.level = 0.1f;
    v.gain = 1.0f;
    v.active = 1;
    v.restart = 1;
    setup(v);
    return renderBlocks(e, blocks);
}

/**
 * @brief A few partials, one per harmonic (no cloud: width 0), centred, without drift or shimmer,
 *        at level 0.1 -- for the spectral shape's checks.
 * @param partials  harmonics, and the engine's capacity
 * @param hz        the fundamental
 * @param blocks    how many 128-sample blocks
 * @param setup     sets the shape
 * @return          the samples, interleaved stereo
 */
template <class Setup>
std::vector<float> fewPartials(int partials, float hz, int blocks, Setup&& setup)
{
    CloudConfig c;
    c.sampleRate = kSampleRate;
    c.blockSize = 128;
    c.voices = 1;
    c.partialsPerVoice = partials;
    CloudEngine e;
    if (!openEngine(e, c)) return {};
    CloudVoice& v = e.voice(0);
    v = CloudVoice{};
    v.hz = hz;
    v.partials = partials;
    v.seed = 9u;
    v.width = 0.0f;
    v.drift = 0.0f;
    v.shimmer = 0.0f;
    v.stereo = 0.0f;
    v.level = 0.1f;
    v.gain = 1.0f;
    v.active = 1;
    v.restart = 1;
    setup(v);
    return renderBlocks(e, blocks);
}

/**
 * @brief The amplitude of one sine in the left channel, by projection on the very phase step the
 *        GPU computes for that frequency (float, as the kernel does).
 * @param lr    interleaved stereo
 * @param from  first frame
 * @param f     the sine's frequency, Hz
 * @return      its amplitude
 */
double toneAmp(const std::vector<float>& lr, size_t from, float f)
{
    const std::uint32_t step = static_cast<std::uint32_t>(std::min(f / static_cast<float>(kSampleRate), 0.49f) * 4294967296.0f);
    double re = 0.0, im = 0.0;
    const size_t frames = lr.size() / 2;
    for (size_t s = from; s < frames; ++s) {
        const double ph = static_cast<double>(static_cast<std::uint32_t>(s * step)) * (6.283185307179586 / 4294967296.0);
        re += lr[s * 2] * std::cos(ph);
        im += lr[s * 2] * std::sin(ph);
    }
    return 2.0 * std::sqrt(re * re + im * im) / static_cast<double>(std::max<size_t>(frames - from, 1));
}

/** @brief A block's amplitude in one channel: sqrt(2 * mean square). @param lr interleaved stereo @param b the block (128 frames) @param ch 0 left, 1 right @return the amplitude */
double blockAmp(const std::vector<float>& lr, size_t b, int ch)
{
    double sq = 0.0;
    for (size_t s = b * 128; s < (b + 1) * 128 && s * 2 + 1 < lr.size(); ++s) sq += static_cast<double>(lr[s * 2 + ch]) * lr[s * 2 + ch];
    return std::sqrt(2.0 * sq / 128.0);
}

/**
 * @brief Fits a sine of a known phase step to a stretch of one channel and returns the worst residual.
 * @param lr     interleaved stereo
 * @param from   first frame of the stretch
 * @param step   the phase step per sample, as a fraction of 2^32
 * @param amp    receives the fitted amplitude
 * @return       the worst absolute difference between the channel and the fitted sine
 */
double fitSine(const std::vector<float>& lr, size_t from, std::uint32_t step, double& amp)
{
    const size_t frames = lr.size() / 2;
    auto phase = [&](size_t s) { return static_cast<double>(static_cast<std::uint32_t>(s * step)) * (6.283185307179586 / 4294967296.0); };
    double ss = 0, sc = 0, cc = 0, ys = 0, yc = 0;
    for (size_t s = from; s < frames; ++s) {
        const double sn = std::sin(phase(s)), cs = std::cos(phase(s)), y = lr[s * 2];
        ss += sn * sn; sc += sn * cs; cc += cs * cs; ys += y * sn; yc += y * cs;
    }
    const double det = std::max(ss * cc - sc * sc, 1.0e-30);   // (0 only if nothing was rendered)
    const double a = (ys * cc - yc * sc) / det, b = (yc * ss - ys * sc) / det;
    double worst = 0.0;
    for (size_t s = from; s < frames; ++s) worst = std::max(worst, std::fabs(lr[s * 2] - (a * std::sin(phase(s)) + b * std::cos(phase(s)))));
    amp = std::sqrt(a * a + b * b);
    return worst;
}

/** @brief probe selftest. @param o the options @return the exit code: 0 when every check passed */
int cmdSelftest(const Options& o)
{
    Options base = o;
    base.scene = "none";
    int failures = 0;
    auto check = [&](bool good, const char* what, const std::string& detail) {
        std::printf("  %s  %-46s %s\n", good ? "ok  " : "FAIL", what, detail.c_str());
        if (!good) ++failures;
    };
    char buf[256];

    // 1. One partial against its exact sine: the phase step the GPU must have used, the fitted
    //    amplitude and phase, and what is left after the fit.
    {
        CloudConfig c;
        c.sampleRate = kSampleRate;
        c.blockSize = 128;
        c.voices = 1;
        c.partialsPerVoice = 1;
        CloudEngine e;
        if (!openEngine(e, c)) return 1;
        std::printf("%s\n", e.deviceName().c_str());
        CloudVoice& v = e.voice(0);
        setNote(v, 997.0f, 1, 7u, base);
        v.drift = 0.0f; v.shimmer = 0.0f; v.stereo = 0.0f; v.level = 0.1f;
        const std::vector<float> lr = renderBlocks(e, 400);
        const std::uint32_t step = static_cast<std::uint32_t>(std::min(997.0f / static_cast<float>(kSampleRate), 0.49f) * 4294967296.0f);
        double amp = 0.0;
        const double worst = fitSine(lr, static_cast<size_t>(c.blockSize), step, amp);
        double lrDiff = 0.0;
        for (size_t s = static_cast<size_t>(c.blockSize); s < lr.size() / 2; ++s) lrDiff = std::max(lrDiff, static_cast<double>(std::fabs(lr[s * 2] - lr[s * 2 + 1])));
        std::snprintf(buf, sizeof buf, "amplitude %.6f (0.1 expected)", amp);
        check(std::fabs(amp - 0.1) < 1.0e-4, "one partial: amplitude", buf);
        std::snprintf(buf, sizeof buf, "worst %.2e = %.1f dB below the partial", worst, 20.0 * std::log10(std::max(worst, 1.0e-12) / 0.1));
        check(worst < 1.0e-5, "one partial: residual after the exact sine", buf);
        std::snprintf(buf, sizeof buf, "worst |L - R| %.2e", lrDiff);
        check(lrDiff < 1.0e-7, "one partial: centred is the same left and right", buf);
    }

    // 2. A full voice meets its level: 20,000 partials at random phases, RMS over both channels.
    {
        CloudConfig c = chordLayout(20000);
        c.voices = 1;
        CloudEngine e;
        if (!openEngine(e, c)) return 1;
        setNote(e.voice(0), 110.0f, 20000, 11u, base);
        const std::vector<float> lr = renderBlocks(e, static_cast<int>(3.0 * kSampleRate / c.blockSize));
        double sq = 0.0;
        const size_t skip = static_cast<size_t>(0.5 * kSampleRate) * 2;
        for (size_t i = skip; i < lr.size(); ++i) sq += static_cast<double>(lr[i]) * lr[i];
        const double rms = std::sqrt(sq / static_cast<double>((lr.size() - skip) / 2));
        const double db = 20.0 * std::log10(rms / 0.05);
        std::snprintf(buf, sizeof buf, "RMS %.4f, %+.2f dB from 0.05 (%d x %d)", rms, db, e.voice(0).harmonics, e.voice(0).members);
        check(std::fabs(db) < 1.0, "full voice: level", buf);
    }

    // 3 and 4. The same notes render the same bits, without envelopes and with all three lanes:
    //    the reduction always adds in the same order and the envelopes are functions of time.
    for (const char* scene : { "none", "breathe" }) {
        Options load = base;
        load.block = 128;
        load.voices = 16;
        load.scene = scene;
        std::vector<float> first, second;
        for (std::vector<float>* dst : { &first, &second }) {
            CloudEngine e;
            if (!openEngine(e, layout(load, 20000)) || !prepareEngine(e, load)) return 1;
            setLoad(e, 20000, load);
            *dst = renderBlocks(e, 200);
        }
        const bool same = first.size() == second.size() && std::memcmp(first.data(), second.data(), sizeof(float) * first.size()) == 0;
        std::string detail = "bit for bit";
        if (!same) {
            size_t at = 0;
            while (at < std::min(first.size(), second.size()) && std::memcmp(&first[at], &second[at], sizeof(float)) == 0) ++at;
            std::snprintf(buf, sizeof buf, "differs from frame %zu (block %zu) on", at / 2, at / 2 / static_cast<size_t>(load.block));
            detail = buf;
        }
        std::snprintf(buf, sizeof buf, "16 x 20,000 partials, scene %s: deterministic", scene);
        check(same, buf, detail);
    }

    // 5. The amplitude lane follows its shape: a triangle looping every 2 s, block by block.
    {
        const std::vector<EnvShape> bank{ makeShape({ { 0, 0 }, { 1, 1 }, { 2, 0 } }, 0, 2) };
        const int blocks = static_cast<int>(4.0 * kSampleRate / 128);
        const std::vector<float> lr = onePartial(bank, nullptr, blocks, [](CloudVoice& v) { v.amp.amount = 1.0f; });
        double worst = 0.0;
        for (int b = 1; b < blocks; ++b) {
            double mid = std::fmod((b + 0.5) * 128.0 / kSampleRate, 2.0);
            mid = mid < 1.0 ? mid : 2.0 - mid;
            worst = std::max(worst, std::fabs(blockAmp(lr, static_cast<size_t>(b), 0) - 0.1 * mid));
        }
        std::snprintf(buf, sizeof buf, "worst block %.5f from the triangle (of 0.1)", worst);
        check(worst < 0.002, "amplitude lane: a looping triangle", buf);
    }

    // 6. The pitch lane glides an octave up into 1500 Hz over a second -- without a phase jump --
    //    and lands exactly on the sine of 1500 Hz.
    {
        const std::vector<EnvShape> bank{ makeShape({ { 0, -1 }, { 1, 0 } }, -1, -1) };
        const int blocks = static_cast<int>(2.0 * kSampleRate / 128);
        const std::vector<float> lr = onePartial(bank, nullptr, blocks, [](CloudVoice& v) { v.pitch.amount = 1200.0f; });
        double jump = 0.0;
        for (size_t s = 1; s < lr.size() / 2; ++s) jump = std::max(jump, static_cast<double>(std::fabs(lr[s * 2] - lr[(s - 1) * 2])));
        const double limit = 2.0 * 3.14159265358979 * 1500.0 / kSampleRate * 0.1 * 1.05;
        std::snprintf(buf, sizeof buf, "largest step %.4f (a sine of 1500 Hz: %.4f)", jump, limit / 1.05);
        check(jump < limit, "pitch lane: the glide never jumps", buf);
        double amp = 0.0;
        const double worst = fitSine(lr, static_cast<size_t>(1.2 * kSampleRate), 1u << 27, amp);
        std::snprintf(buf, sizeof buf, "residual %.2e, amplitude %.5f", worst, amp);
        check(worst < 1.0e-5 && std::fabs(amp - 0.1) < 1.0e-4, "pitch lane: lands exactly on 1500 Hz", buf);
    }

    // 7. The pan lane swings the partial across the field at constant power.
    {
        const std::vector<EnvShape> bank{ makeShape({ { 0, 0 }, { 1, 1 }, { 3, -1 }, { 4, 0 } }, 0, 3) };
        const int blocks = static_cast<int>(4.0 * kSampleRate / 128);
        const std::vector<float> lr = onePartial(bank, nullptr, blocks, [](CloudVoice& v) { v.pan.amount = 1.0f; });
        double worstPower = 0.0, minL = 1.0, maxL = 0.0;
        for (int b = 2; b < blocks; ++b) {
            const double l = blockAmp(lr, static_cast<size_t>(b), 0), r = blockAmp(lr, static_cast<size_t>(b), 1);
            worstPower = std::max(worstPower, std::fabs((l * l + r * r) / 0.02 - 1.0));
            minL = std::min(minL, l);
            maxL = std::max(maxL, l);
        }
        std::snprintf(buf, sizeof buf, "power within %.2f %%, left from %.4f to %.4f", 100.0 * worstPower, minL, maxL);
        check(worstPower < 0.01 && minL < 0.005 && maxL > 0.135, "pan lane: across the field at constant power", buf);
    }

    // 8. The landscape sets the amplitude: everywhere 0.25 gives a quarter.
    {
        Landscape land;
        land.columns = 2;
        land.rows = 1;
        land.rowSeconds = 1.0f;
        land.amps.assign(2, 0.25f);
        const std::vector<EnvShape> bank{ makeShape({ { 0, 1 } }, -1, -1) };
        const std::vector<float> lr = onePartial(bank, &land, 100, [](CloudVoice& v) { v.amp.amount = 1.0f; v.amp.landscape = 1; });
        const double a = blockAmp(lr, 50, 0);
        std::snprintf(buf, sizeof buf, "amplitude %.5f (0.025 expected)", a);
        check(std::fabs(a - 0.025) < 0.0005, "landscape: reads its amplitude", buf);
    }

    // 9. A release tail: held at 1 until the release at 1 s, then down to 0 over 2 s.
    {
        const std::vector<EnvShape> bank{ makeShape({ { 0, 1 }, { 1, 1 }, { 3, 0 } }, 0, 1) };
        const int blocks = static_cast<int>(4.0 * kSampleRate / 128);
        const std::vector<float> lr = onePartial(bank, nullptr, blocks, [](CloudVoice& v) { v.amp.amount = 1.0f; v.releaseAt = 1.0f; });
        const double held = blockAmp(lr, static_cast<size_t>(0.8 * kSampleRate / 128), 0);
        const double halfway = blockAmp(lr, static_cast<size_t>(2.0 * kSampleRate / 128) - 1, 0);
        const double gone = blockAmp(lr, static_cast<size_t>(3.5 * kSampleRate / 128), 0);
        std::snprintf(buf, sizeof buf, "held %.4f, at 2 s %.4f, at 3.5 s %.5f", held, halfway, gone);
        check(std::fabs(held - 0.1) < 0.001 && std::fabs(halfway - 0.05) < 0.002 && gone < 0.0005, "release: the tail from the loop to silence", buf);
    }

    // 10. The voice filter as a frequency response: LP12 an octave above its corner, and its peak.
    {
        const int blocks = static_cast<int>(1.0 * kSampleRate / 128);
        const std::vector<EnvShape> none;
        const std::vector<float> below = onePartial(none, nullptr, blocks, [](CloudVoice& v) {
            v.filterModel = static_cast<int>(FilterModel::LP12); v.cutoff = 750.0f; v.filterQ = 0.70710678f; });
        const std::vector<float> peak = onePartial(none, nullptr, blocks, [](CloudVoice& v) {
            v.filterModel = static_cast<int>(FilterModel::LP12); v.cutoff = 1500.0f; v.filterQ = 4.0f; });
        const double a = blockAmp(below, 100, 0), b = blockAmp(peak, 100, 0);
        const double expect = 0.1 / std::sqrt(9.0 + 8.0);   // r = 2: (1 - r^2)^2 + (r/Q)^2
        std::snprintf(buf, sizeof buf, "an octave above: %.5f (%.5f), at the corner Q 4: %.4f (0.4)", a, expect, b);
        check(std::fabs(a / expect - 1.0) < 0.01 && std::fabs(b / 0.4 - 1.0) < 0.01, "filter: LP12 read at the partial", buf);
    }

    // 11. Noctuary's spectral formula, partial by partial, after the voice's normalisation:
    //     presence lifts 3.2 kHz by 6 dB at the ear, the low cut drops an octave below it by 12 dB.
    {
        const int blocks = static_cast<int>(1.0 * kSampleRate / 128);
        const size_t from = 4800;
        const std::vector<float> pres = fewPartials(2, 1600.0f, blocks, [](CloudVoice& v) { v.presenceDb = 6.0f; });
        const double ratioP = toneAmp(pres, from, 3200.0f) / toneAmp(pres, from, 1600.0f);
        const double expectP = std::pow(2.0, -1.2) * std::pow(10.0, 6.0 / 20.0);
        const std::vector<float> cut = fewPartials(2, 100.0f, blocks, [](CloudVoice& v) { v.lowCut = 200.0f; });
        const double ratioC = toneAmp(cut, from, 200.0f) / toneAmp(cut, from, 100.0f);
        const double expectC = std::pow(2.0, -1.2) / 0.25;
        std::snprintf(buf, sizeof buf, "presence %.4f (%.4f), low cut %.4f (%.4f)", ratioP, expectP, ratioC, expectC);
        check(std::fabs(ratioP / expectP - 1.0) < 0.01 && std::fabs(ratioC / expectC - 1.0) < 0.01, "spectrum: presence and low cut", buf);
    }

    // 12. Brightness closes the window above harmonic 9; odd/even +1 silences the even harmonics.
    {
        const int blocks = static_cast<int>(1.0 * kSampleRate / 128);
        const size_t from = 4800;
        const std::vector<float> dark = fewPartials(200, 100.0f, blocks, [](CloudVoice& v) { v.brightness = 0.2f; });
        const double h5 = toneAmp(dark, from, 500.0f), h20 = toneAmp(dark, from, 2000.0f);
        const std::vector<float> odd = fewPartials(4, 200.0f, blocks, [](CloudVoice& v) { v.oddEven = 1.0f; });
        const double o2 = toneAmp(odd, from, 400.0f), o3 = toneAmp(odd, from, 600.0f);
        std::snprintf(buf, sizeof buf, "h20/h5 %.1e, h2/h3 %.1e", h20 / h5, o2 / o3);
        check(h5 > 1e-4 && h20 / h5 < 1e-4 && o3 > 1e-4 && o2 / o3 < 1e-4, "spectrum: brightness window and odd/even", buf);
    }

    // 13. Inharmonic: the stiff string moves even the fundamental to f0 sqrt(1 + B).
    {
        const std::vector<float> lr = fewPartials(1, 100.0f, static_cast<int>(1.0 * kSampleRate / 128), [](CloudVoice& v) { v.inharmonic = 0.02f; });
        const float f = 100.0f * std::sqrt(1.0f + 0.02f);
        const double at = toneAmp(lr, 4800, f), off = toneAmp(lr, 4800, 100.0f);
        // A second's projection one hertz off still leaks sinc(pi * 0.995 * 0.9) = 11 %: the full
        // amplitude at the stretched frequency is the proof, the unstretched one only leakage.
        std::snprintf(buf, sizeof buf, "at %.3f Hz %.4f, at 100 Hz %.4f (leakage)", f, at, off);
        check(std::fabs(at - 0.1) < 0.0002 && off < 0.02, "spectrum: the stiff string's stretch", buf);
    }

    // 14. Air: a band of partials at the voice's own level beside it, so the whole is 3 dB up.
    {
        auto rms = [](const std::vector<float>& lr) {
            double sq = 0.0;
            for (size_t i = lr.size() / 4; i < lr.size(); ++i) sq += static_cast<double>(lr[i]) * lr[i];
            return std::sqrt(sq / static_cast<double>(lr.size() - lr.size() / 4));
        };
        const int blocks = static_cast<int>(2.0 * kSampleRate / 128);
        const std::vector<float> dry = fewPartials(2000, 110.0f, blocks, [](CloudVoice& v) { v.width = 25.0f; v.stereo = 1.0f; });
        const std::vector<float> wet = fewPartials(2000, 110.0f, blocks, [](CloudVoice& v) { v.width = 25.0f; v.stereo = 1.0f; v.air = 1.0f; });
        const double db = 20.0 * std::log10(rms(wet) / rms(dry));
        std::snprintf(buf, sizeof buf, "%+.2f dB with the band (+3.01 expected)", db);
        check(std::fabs(db - 3.01) < 0.6, "air: a band of partials at the voice's level", buf);
    }

    // 15. Distance 1: nothing near, half the level, and the air's low pass read at 1.5 kHz.
    {
        const std::vector<EnvShape> none;
        const std::vector<float> lr = onePartial(none, nullptr, 100, [](CloudVoice& v) { v.distance = 1.0f; });
        const double x0 = 1500.0 / 16000.0, x = x0 * 32.0;
        const double expect = 0.1 * 0.5 * std::sqrt((1.0 + x0 * x0) / (1.0 + x * x));
        const double a = blockAmp(lr, 50, 0);
        std::snprintf(buf, sizeof buf, "amplitude %.5f (%.5f)", a, expect);
        check(std::fabs(a / expect - 1.0) < 0.01, "distance: the far plane, darker and quieter", buf);
    }

    std::printf(failures == 0 ? "selftest passed\n" : "selftest: %d FAILED\n", failures);
    return failures == 0 ? 0 : 1;
}

/** @brief Splits "200,2000,20000" into numbers. @param s the list @return the numbers (empty on a bad entry) */
std::vector<int> parseList(const std::string& s)
{
    std::vector<int> out;
    size_t pos = 0;
    while (pos <= s.size()) {
        const size_t comma = s.find(',', pos);
        const std::string item = s.substr(pos, comma == std::string::npos ? std::string::npos : comma - pos);
        const int x = std::atoi(item.c_str());
        if (x <= 0) return {};
        out.push_back(x);
        if (comma == std::string::npos) break;
        pos = comma + 1;
    }
    return out;
}

/** @brief Prints the usage. */
void usage()
{
    std::printf("probe info | selftest | bench | latency | ahead | render | scenes\n"
                "  --block N (128)  --voices N (16)  --partials N[,N..] (20000)  --seconds S\n"
                "  --lookahead MS (10)  --wait spin|block  --csv FILE  --out DIR\n"
                "  --width CENTS (25)  --slope X (0.3)  --drift CENTS (3)  --shimmer X (0.4)\n"
                "  --gpu-priority idle|below|normal|above|high|realtime\n"
                "  --scene NAME  --landscape FILE  --loop S (30)  --range DB (40)  --whiten X (0.7)  --seamless X (0.1)  --smooth S (0.15)\n"
                "scenes:");
    for (const Scene& s : kScenes) std::printf(" %s", s.name);
    std::printf("\n");
}

} // namespace

/**
 * @brief Reads the subcommand and its options and runs it.
 * @param argc  argument count
 * @param argv  arguments
 * @return      0 when the run succeeded (ahead: 2 when it underran)
 */
int main(int argc, char** argv)
{
    if (argc < 2) { usage(); return 1; }
    const std::string cmd = argv[1];
    Options o;
    for (int i = 2; i < argc; ++i) {
        const std::string a = argv[i];
        const bool hasValue = i + 1 < argc;
        const std::string val = hasValue ? argv[i + 1] : "";
        auto number = [&] { return static_cast<float>(std::atof(val.c_str())); };
        if (a == "--block" && hasValue) { o.block = std::atoi(val.c_str()); ++i; }
        else if (a == "--voices" && hasValue) { o.voices = std::atoi(val.c_str()); ++i; }
        else if (a == "--partials" && hasValue) { o.partials = parseList(val); ++i; }
        else if (a == "--seconds" && hasValue) { o.seconds = std::atof(val.c_str()); ++i; }
        else if (a == "--lookahead" && hasValue) { o.lookaheadMs = std::atof(val.c_str()); ++i; }
        else if (a == "--wait" && hasValue) { o.spin = val != "block"; ++i; }
        else if (a == "--csv" && hasValue) { o.csv = val; ++i; }
        else if (a == "--out" && hasValue) { o.out = val; ++i; }
        else if (a == "--width" && hasValue) { o.width = number(); ++i; }
        else if (a == "--slope" && hasValue) { o.slope = number(); ++i; }
        else if (a == "--drift" && hasValue) { o.drift = number(); ++i; }
        else if (a == "--shimmer" && hasValue) { o.shimmer = number(); ++i; }
        else if (a == "--scene" && hasValue) { o.scene = val; ++i; }
        else if (a == "--landscape" && hasValue) { o.landscape = val; ++i; }
        else if (a == "--loop" && hasValue) { o.loopSeconds = number(); ++i; }
        else if (a == "--range" && hasValue) { o.rangeDb = number(); ++i; }
        else if (a == "--whiten" && hasValue) { o.whiten = number(); ++i; }
        else if (a == "--seamless" && hasValue) { o.seamless = number(); ++i; }
        else if (a == "--smooth" && hasValue) { o.smooth = number(); ++i; }
        else if (a == "--reservoir") o.reservoir = true;
        else if (a == "--melt") o.melt = true;
        else if (a == "--gpu-priority" && hasValue) {
            static const char* const kNames[] = { "idle", "below", "normal", "above", "high", "realtime" };
            for (int k = 0; k < 6; ++k) if (val == kNames[k]) o.gpuPriority = k;
            if (o.gpuPriority < 0) { std::fprintf(stderr, "probe: --gpu-priority idle|below|normal|above|high|realtime\n"); return 1; }
            ++i;
        }
        else { std::fprintf(stderr, "probe: unknown option %s\n", a.c_str()); usage(); return 1; }
    }
    if (o.partials.empty() || o.block < 32 || o.voices < 1) { std::fprintf(stderr, "probe: bad --partials, --block or --voices\n"); return 1; }
    if (!findScene(o.scene)) { std::fprintf(stderr, "probe: unknown scene %s\n", o.scene.c_str()); usage(); return 1; }
    if (o.gpuPriority >= 0) {
        const long status = setGpuPriority(o.gpuPriority);
        std::printf("GPU scheduling priority class %d: %s (NTSTATUS 0x%08lX)\n", o.gpuPriority, status == 0 ? "set" : "REFUSED", static_cast<unsigned long>(status));
    }

    if (cmd == "info") return cmdInfo(o);
    if (cmd == "selftest") return cmdSelftest(o);
    if (cmd == "bench") return cmdBench(o);
    if (cmd == "latency") return cmdLatency(o);
    if (cmd == "ahead") return cmdAhead(o);
    if (cmd == "render") return cmdRender(o);
    if (cmd == "scenes") return cmdScenes(o);
    usage();
    return 1;
}
