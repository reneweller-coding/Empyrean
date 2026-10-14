/**
 * @file main.cpp
 * @brief empy_render: the instrument offline, without a window -- the conductor alone for minutes
 *        or hours, a chord played and held, or a journey for a whole night -- into 24-bit WAVs,
 *        with a line every ten seconds.
 *
 *     empy_render [--seconds S (120)] [--out FILE] [--seed N] [--preset NAME|N] [--set key=value]... [--mod ROWS]
 *                    [--chord 50,57,62,66] [--hold S] [--no-brain] [--journey NAME|FILE] [--journey-seed N]
 *                    [--route NAME|FILE] [--route-speed X] [--fill]
 *     empy_render --dump-params        (the parameter table as JSON, for Tools/manual/make_manual.py)
 *
 * --preset loads a factory preset (its name, or its number 0 .. 2047) with its matrix; --set takes any
 * key of the parameter table (Core/include/additiv/Params.h) after it; a choice takes its index or its
 * name ("--set auto_mode=Free", "--set motion=Aurora"). --journey plays a journey (Journey.h): one that
 * comes with the instrument by its name ("Whole Night - Deep Space"), or a .journey file; each step
 * glides into its preset as the instrument does. --route walks a route over the preset map (Route.h):
 * one that comes with the instrument by its name ("Dawn") or a .route file, the sound the blend of
 * the presets round the cursor, followed over Map Glide. --fill lets the conductor's chord arrive within
 * seconds, as a preset chosen on the panel does, instead of entering one voice at a time. The WAV is written as it is rendered; beyond an hour
 * it is cut into files of an hour (render-01.wav, render-02.wav, ...).
 */
#include "additiv/Engine.h"
#include "additiv/Journey.h"
#include "additiv/PresetMap.h"
#include "additiv/Route.h"
#include "additiv/Presets.h"

#define NOMINMAX
#define WIN32_LEAN_AND_MEAN
#include <windows.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using namespace additiv;

namespace {

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

/** @brief A 24-bit stereo WAV written as it comes: the header first with no length, the length set at close(). */
class WavStream {
public:
    /**
     * @brief Opens the file and writes a header.
     * @param path        the file
     * @param sampleRate  Hz
     * @return            true when it is open
     */
    bool open(const fs::path& path, int sampleRate)
    {
        f_.open(path, std::ios::binary | std::ios::trunc);
        frames_ = 0;
        const std::uint32_t sr = static_cast<std::uint32_t>(sampleRate);
        unsigned char h[44] = {};
        auto put = [&](int at, const char* s) { std::memcpy(h + at, s, 4); };
        auto u32 = [&](int at, std::uint32_t x) { for (int i = 0; i < 4; ++i) h[at + i] = static_cast<unsigned char>(x >> (8 * i)); };
        auto u16 = [&](int at, std::uint16_t x) { h[at] = static_cast<unsigned char>(x); h[at + 1] = static_cast<unsigned char>(x >> 8); };
        put(0, "RIFF"); put(8, "WAVE"); put(12, "fmt "); u32(16, 16); u16(20, 1); u16(22, 2); u32(24, sr); u32(28, sr * 6); u16(32, 6); u16(34, 24);
        put(36, "data");
        f_.write(reinterpret_cast<const char*>(h), 44);
        return static_cast<bool>(f_);
    }
    /** @brief Appends frames. @param L left @param R right @param n frames */
    void write(const float* L, const float* R, int n)
    {
        buf_.resize(static_cast<size_t>(n) * 6);
        for (int i = 0; i < n; ++i)
            for (int c = 0; c < 2; ++c) {
                const float x = c == 0 ? L[i] : R[i];
                const std::int32_t q = static_cast<std::int32_t>(std::lround(std::clamp(x, -1.0f, 1.0f) * 8388607.0f));
                unsigned char* o = &buf_[static_cast<size_t>(i) * 6 + static_cast<size_t>(c) * 3];
                o[0] = static_cast<unsigned char>(q); o[1] = static_cast<unsigned char>(q >> 8); o[2] = static_cast<unsigned char>(q >> 16);
            }
        f_.write(reinterpret_cast<const char*>(buf_.data()), static_cast<std::streamsize>(buf_.size()));
        frames_ += n;
    }
    /** @brief Sets the lengths in the header and closes. @return true when everything was written */
    bool close()
    {
        const std::uint32_t data = static_cast<std::uint32_t>(frames_ * 6);
        auto u32 = [&](std::streamoff at, std::uint32_t x) {
            unsigned char b[4];
            for (int i = 0; i < 4; ++i) b[i] = static_cast<unsigned char>(x >> (8 * i));
            f_.seekp(at);
            f_.write(reinterpret_cast<const char*>(b), 4);
        };
        u32(4, 36 + data);
        u32(40, data);
        const bool ok = static_cast<bool>(f_);
        f_.close();
        return ok;
    }
    /** @brief Frames written so far. @return the count */
    std::int64_t frames() const { return frames_; }

private:
    std::ofstream f_;                       ///< the file
    std::int64_t frames_ = 0;               ///< frames written
    std::vector<unsigned char> buf_;        ///< a block's bytes
};

/**
 * @brief Applies one --set key=value: a number, or for a choice also its name.
 * @param e    the engine
 * @param arg  "key=value"
 * @return     true when the key exists and the value was read
 */
bool applySet(Engine& e, const std::string& arg)
{
    const size_t eq = arg.find('=');
    if (eq == std::string::npos) return false;
    const std::string key = arg.substr(0, eq), val = arg.substr(eq + 1);
    const ParamDesc* d = findParam(key.c_str());
    if (!d) return false;
    if (d->kind == ParamKind::Choice)
        for (int i = 0; i < d->numChoices; ++i)
            if (val == d->choices[i]) { e.setParam(d->id, static_cast<float>(i)); return true; }
    if (d->kind == ParamKind::Bool && (val == "on" || val == "off")) { e.setParam(d->id, val == "on" ? 1.0f : 0.0f); return true; }
    char* end = nullptr;
    const float x = std::strtof(val.c_str(), &end);
    if (end == val.c_str()) return false;
    e.setParam(d->id, x);
    return true;
}

/**
 * @brief A string as a JSON literal.
 * @param s  the text
 * @return   it quoted, with quotes and backslashes escaped
 */
std::string jsonString(const char* s)
{
    std::string r = "\"";
    for (const char* c = s ? s : ""; *c; ++c) {
        if (*c == '"' || *c == '\\') r += '\\';
        r += *c;
    }
    return r + "\"";
}

/** @brief Writes every row of the parameter table as JSON to stdout (key, name, section, kind, range, default, skew, unit, choices). */
void dumpParams()
{
    static const char* const kKinds[] = { "float", "int", "bool", "choice" };
    std::printf("[\n");
    const auto& table = paramTable();
    for (size_t i = 0; i < table.size(); ++i) {
        const ParamDesc& d = table[i];
        std::printf("  {\"key\": %s, \"name\": %s, \"section\": %s, \"kind\": \"%s\", \"min\": %g, \"max\": %g, \"default\": %g, \"skew\": %g, \"unit\": %s, \"choices\": [",
                    jsonString(d.key).c_str(), jsonString(d.name).c_str(), jsonString(d.section).c_str(), kKinds[static_cast<int>(d.kind)],
                    static_cast<double>(d.min), static_cast<double>(d.max), static_cast<double>(d.def), static_cast<double>(d.skew), jsonString(d.unit).c_str());
        for (int c = 0; c < d.numChoices; ++c) std::printf("%s%s", c ? ", " : "", jsonString(d.choices[c]).c_str());
        std::printf("]}%s\n", i + 1 < table.size() ? "," : "");
    }
    std::printf("]\n");
}

/**
 * @brief Loads a factory preset into the engine: every parameter and the matrix.
 * @param e     the engine
 * @param what  the preset's name, or its number
 * @return      the preset's index, -1 when there is none of that name or number
 */
int applyPreset(Engine& e, const std::string& what)
{
    int index = findPreset(what.c_str());
    if (index < 0 && !what.empty() && what.find_first_not_of("0123456789") == std::string::npos) index = std::atoi(what.c_str());
    if (index < 0 || index >= numPresets()) return -1;
    e.loadPreset(index);
    return index;
}

/**
 * @brief A journey by name (one that comes with the instrument) or from a file.
 * @param what  the name or the path
 * @param j     receives it
 * @return      true when it was found and read
 */
bool findJourney(const std::string& what, Journey& j)
{
    for (int i = 0; i < numBuiltinJourneys(); ++i) {
        Journey b;
        if (b.parse(builtinJourneyText(i)) && b.name == what) { j = b; return true; }
    }
    return j.load(what.c_str()) && !j.steps.empty();
}

} // namespace

/**
 * @brief Reads the options, renders, writes the WAV.
 * @param argc  argument count
 * @param argv  arguments
 * @return      0 on success
 */
int main(int argc, char** argv)
{
    double seconds = 120.0, hold = -1.0;
    fs::path out = repoRoot() / "work" / "renders" / "render.wav";
    std::uint64_t seed = 0x5EEDu;
    std::vector<std::string> sets;
    std::string modRows;   // --mod
    std::vector<int> chord;
    std::string presetName, journeyName, routeName;
    double routeSpeed = 1.0;
    std::uint64_t journeySeed = 1;
    bool brain = true, fill = false;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        const bool has = i + 1 < argc;
        if (a == "--seconds" && has) seconds = std::atof(argv[++i]);
        else if (a == "--out" && has) out = argv[++i];
        else if (a == "--seed" && has) seed = std::strtoull(argv[++i], nullptr, 10);
        else if (a == "--set" && has) sets.push_back(argv[++i]);
        else if (a == "--mod" && has) modRows = argv[++i];   // matrix rows in the text form, after the preset's (Modulation.h)
        else if (a == "--hold" && has) hold = std::atof(argv[++i]);
        else if (a == "--no-brain") brain = false;
        else if (a == "--fill") fill = true;
        else if (a == "--preset" && has) presetName = argv[++i];
        else if (a == "--journey" && has) journeyName = argv[++i];
        else if (a == "--journey-seed" && has) journeySeed = std::strtoull(argv[++i], nullptr, 10);
        else if (a == "--route" && has) routeName = argv[++i];
        else if (a == "--route-speed" && has) routeSpeed = std::atof(argv[++i]);
        else if (a == "--dump-params") { dumpParams(); return 0; }
        else if (a == "--chord" && has) {
            std::string s = argv[++i];
            for (size_t p = 0; p <= s.size();) {
                const size_t c = s.find(',', p);
                chord.push_back(std::atoi(s.substr(p, c == std::string::npos ? std::string::npos : c - p).c_str()));
                if (c == std::string::npos) break;
                p = c + 1;
            }
        } else {
            std::fprintf(stderr, "empy_render [--seconds S] [--out FILE] [--seed N] [--preset NAME|N] [--set key=value]... [--mod ROWS] [--chord 50,57,62] [--hold S] [--no-brain]\n"
                                 "            [--journey NAME|FILE] [--journey-seed N] [--route NAME|FILE] [--route-speed X]\n"
                                 "empy_render --dump-params\n");
            return 1;
        }
    }

    Engine e;
    e.setRealtime(false);
    if (!presetName.empty()) {
        const int index = applyPreset(e, presetName);
        if (index < 0) { std::fprintf(stderr, "empy_render: no preset %s\n", presetName.c_str()); return 1; }
        std::printf("preset %d: %s (%s)\n", index, preset(index).name, kPresetGroupNames[preset(index).group]);
    }
    Journey journey;
    JourneyPlayer player;
    if (!journeyName.empty()) {
        if (!findJourney(journeyName, journey)) { std::fprintf(stderr, "empy_render: no journey %s\n", journeyName.c_str()); return 1; }
        player.start(journey, journeySeed);
        JourneyStep st;
        double glide = 0.0;
        if (player.advance(0.0, st, glide) && findPreset(st.preset.c_str()) >= 0) {   // the first at once
            e.loadPreset(findPreset(st.preset.c_str()));
            e.setParam(ParamId::BrainOn, 1.0f);   // a journey keeps the conductor playing (as the instrument does)
        }
        std::printf("journey %s: %zu steps, a pass about %s\n", journey.name.c_str(), journey.steps.size(), Journey::timeText(journey.meanLength()).c_str());
    }
    Route route;
    MapBlender blender;
    float mapX = -1.0f, mapY = -1.0f, mapR = -1.0f;
    if (!routeName.empty()) {
        std::string text;
        for (int i = 0; i < numRoutePresets() && text.empty(); ++i)
            if (routeName == routePreset(i).name) text = routePreset(i).points;
        if (text.empty()) {
            std::ifstream f(routeName, std::ios::binary);
            std::stringstream ss;
            ss << f.rdbuf();
            text = ss.str();
        }
        if (!route.parse(text.c_str())) { std::fprintf(stderr, "empy_render: no route %s\n", routeName.c_str()); return 1; }
        // Starting on the first waypoint's preset, the map on.
        const Waypoint& w = route.point(0);
        if (w.preset >= 0) e.loadPreset(w.preset);
        e.setParam(ParamId::BrainOn, 1.0f);   // a route keeps the conductor playing (as the instrument does)
        route.start(w.x, w.y, w.radius, true);
        e.setParam(ParamId::MapActive, 1.0f);
        std::printf("route %s: %d waypoints\n", routeName.c_str(), route.count());
    }
    for (const std::string& s : sets)
        if (!applySet(e, s)) { std::fprintf(stderr, "empy_render: cannot apply --set %s\n", s.c_str()); return 1; }
    if (!modRows.empty()) e.setModText(modRows);   // replaces the preset's matrix
    if (!brain) e.setParam(ParamId::BrainOn, 0.0f);
    EngineConfig c;
    c.sampleRate = 48000.0;
    c.maxHostBlock = 512;
    c.realtimeGpuClass = false;
    c.seed = seed;
    std::string error;
    if (!e.prepare(c, &error)) { std::fprintf(stderr, "empy_render: %s\n", error.c_str()); return 1; }
    if (fill) e.requestFill();
    std::printf("%s, latency %d samples, scale %s, motion %s, autoplay %s, %.0f s\n", e.deviceName().c_str(), e.latencySamples(),
                kScaleNames[static_cast<int>(e.param(ParamId::Scale))], kMotionNames[static_cast<int>(e.param(ParamId::Motion))],
                e.param(ParamId::BrainOn) >= 0.5f ? kBrainModeNames[static_cast<int>(e.param(ParamId::AutoMode))] : "off", seconds);
    for (int note : chord) e.noteOn(note, 0.8f, 0);

    const int n = 512;
    const std::int64_t total = static_cast<std::int64_t>(seconds * c.sampleRate);
    // Beyond an hour, a file an hour: name-01.wav, name-02.wav, ...
    const std::int64_t perFile = static_cast<std::int64_t>(3600.0 * c.sampleRate);
    const bool parts = total > perFile;
    fs::create_directories(out.parent_path());
    auto partPath = [&](int k) {
        if (!parts) return out;
        char suffix[16];
        std::snprintf(suffix, sizeof(suffix), "-%02d", k);
        return out.parent_path() / (out.stem().string() + suffix + out.extension().string());
    };
    WavStream wav;
    int part = 1;
    if (!wav.open(partPath(part), static_cast<int>(c.sampleRate))) { std::fprintf(stderr, "empy_render: cannot write %s\n", partPath(part).string().c_str()); return 1; }
    std::vector<float> L(n), R(n);
    double sq = 0.0, peak = 0.0, gpu = 0.0;
    int blocks = 0;
    bool released = chord.empty();
    const auto t0 = std::chrono::steady_clock::now();
    for (std::int64_t done = 0; done < total; done += n) {
        if (!released && hold >= 0.0 && done >= static_cast<std::int64_t>(hold * c.sampleRate)) {
            for (int note : chord) e.noteOff(note, 0);
            released = true;
        }
        if (route.running()) {   // the route moves the cursor; a moved cursor is blended anew
            const int seg = route.segment();
            float x = 0.0f, y = 0.0f, r = 0.0f;
            route.update(static_cast<float>(n / c.sampleRate * routeSpeed), x, y, r);
            if (x != mapX || y != mapY || r != mapR) {
                MorphState s;
                blender.blend(x, y, r, s);
                e.setMapTarget(s);
                mapX = x;
                mapY = y;
                mapR = r;
            }
            if (route.segment() != seg) {
                const Waypoint& w = route.point(route.segment());
                std::printf("  %5.0f s: towards %d, %s\n", done / c.sampleRate, route.segment() + 1, w.preset >= 0 ? preset(w.preset).name : "a place");
            }
        }
        if (player.running()) {   // the journey: a step glides into its preset
            JourneyStep st;
            double glide = 0.0;
            if (player.advance(n / c.sampleRate, st, glide)) {
                const int index = findPreset(st.preset.c_str());
                if (index >= 0) { e.loadPreset(index, std::max(glide, 0.5)); e.setParam(ParamId::BrainOn, 1.0f); }
                std::printf("  %5.0f s: step %d, %s, glide %.0f s\n", done / c.sampleRate, player.step() + 1, st.preset.c_str(), glide);
            }
        }
        e.process(L.data(), R.data(), n);
        if (wav.frames() >= perFile) {
            wav.close();
            wav.open(partPath(++part), static_cast<int>(c.sampleRate));
        }
        wav.write(L.data(), R.data(), n);
        for (int i = 0; i < n; ++i) {
            sq += 0.5 * (static_cast<double>(L[i]) * L[i] + static_cast<double>(R[i]) * R[i]);
            peak = std::max(peak, static_cast<double>(std::max(std::fabs(L[i]), std::fabs(R[i]))));
        }
        gpu += e.lastGpuMs();
        ++blocks;
        const std::int64_t now = done + n;
        if (now % (static_cast<std::int64_t>(c.sampleRate) * 10) < n) {
            const double rms = std::sqrt(sq / (blocks * static_cast<double>(n)));
            std::printf("  %5.0f s: %6.1f dBFS RMS, peak %6.1f dBFS, %2d voices, root %3d, GPU %.3f ms/block\n",
                        now / c.sampleRate, 20.0 * std::log10(std::max(rms, 1e-9)), 20.0 * std::log10(std::max(peak, 1e-9)),
                        e.soundingVoices(), e.brainRoot(), gpu / blocks);
            sq = 0.0; peak = 0.0; gpu = 0.0; blocks = 0;
        }
    }
    const double took = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    if (!wav.close()) { std::fprintf(stderr, "empy_render: cannot write %s\n", partPath(part).string().c_str()); return 1; }
    std::printf("%.0f s in %.1f s (%.0fx real time), late notes %llu -> %s%s\n", seconds, took, seconds / std::max(took, 1e-9),
                static_cast<unsigned long long>(e.lateEvents()), out.string().c_str(), parts ? " (in parts of an hour)" : "");
    return 0;
}
