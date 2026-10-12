/**
 * @file main.cpp
 * @brief additiv_render: the instrument offline, without a window -- the conductor alone for minutes
 *        or hours, or a chord played and held -- into a 24-bit WAV, with a line every ten seconds.
 *
 *     additiv_render [--seconds S (120)] [--out FILE] [--seed N] [--preset NAME|N] [--set key=value]...
 *                    [--chord 50,57,62,66] [--hold S] [--no-brain]
 *     additiv_render --dump-params        (the parameter table as JSON, for Tools/manual/make_manual.py)
 *
 * --preset loads a factory preset (its name, or its number 0 .. 1023) with its matrix; --set takes any
 * key of the parameter table (Core/include/additiv/Params.h) after it; a choice takes its index or its
 * name ("--set auto_mode=Free", "--set motion=Aurora").
 */
#include "additiv/Engine.h"
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

/**
 * @brief Writes interleaved stereo as a 24-bit WAV.
 * @param path        the file
 * @param lr          interleaved stereo samples
 * @param sampleRate  Hz
 * @return            true when written
 */
bool writeWav24(const fs::path& path, const std::vector<float>& lr, int sampleRate)
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
        const std::int32_t q = static_cast<std::int32_t>(std::lround(std::clamp(x, -1.0f, 1.0f) * 8388607.0f));
        bytes.push_back(static_cast<unsigned char>(q));
        bytes.push_back(static_cast<unsigned char>(q >> 8));
        bytes.push_back(static_cast<unsigned char>(q >> 16));
    }
    std::ofstream f(path, std::ios::binary);
    f.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    return static_cast<bool>(f);
}

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
    int index = -1;
    for (int i = 0; i < numPresets() && index < 0; ++i)
        if (what == preset(i).name) index = i;
    if (index < 0 && !what.empty() && what.find_first_not_of("0123456789") == std::string::npos) index = std::atoi(what.c_str());
    if (index < 0 || index >= numPresets()) return -1;
    float values[kNumParams] = {};
    std::string mod;
    presetValues(index, values, &mod);
    for (int i = 0; i < kNumParams; ++i) e.setParam(static_cast<ParamId>(i), values[i]);
    e.setModText(mod);
    return index;
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
    std::vector<int> chord;
    std::string presetName;
    bool brain = true;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        const bool has = i + 1 < argc;
        if (a == "--seconds" && has) seconds = std::atof(argv[++i]);
        else if (a == "--out" && has) out = argv[++i];
        else if (a == "--seed" && has) seed = std::strtoull(argv[++i], nullptr, 10);
        else if (a == "--set" && has) sets.push_back(argv[++i]);
        else if (a == "--hold" && has) hold = std::atof(argv[++i]);
        else if (a == "--no-brain") brain = false;
        else if (a == "--preset" && has) presetName = argv[++i];
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
            std::fprintf(stderr, "additiv_render [--seconds S] [--out FILE] [--seed N] [--preset NAME|N] [--set key=value]... [--chord 50,57,62] [--hold S] [--no-brain]\n"
                                 "additiv_render --dump-params\n");
            return 1;
        }
    }

    Engine e;
    e.setRealtime(false);
    if (!presetName.empty()) {
        const int index = applyPreset(e, presetName);
        if (index < 0) { std::fprintf(stderr, "additiv_render: no preset %s\n", presetName.c_str()); return 1; }
        std::printf("preset %d: %s (%s)\n", index, preset(index).name, kPresetGroupNames[preset(index).group]);
    }
    for (const std::string& s : sets)
        if (!applySet(e, s)) { std::fprintf(stderr, "additiv_render: cannot apply --set %s\n", s.c_str()); return 1; }
    if (!brain) e.setParam(ParamId::BrainOn, 0.0f);
    EngineConfig c;
    c.sampleRate = 48000.0;
    c.maxHostBlock = 512;
    c.realtimeGpuClass = false;
    c.seed = seed;
    std::string error;
    if (!e.prepare(c, &error)) { std::fprintf(stderr, "additiv_render: %s\n", error.c_str()); return 1; }
    std::printf("%s, latency %d samples, scale %s, motion %s, autoplay %s, %.0f s\n", e.deviceName().c_str(), e.latencySamples(),
                kScaleNames[static_cast<int>(e.param(ParamId::Scale))], kMotionNames[static_cast<int>(e.param(ParamId::Motion))],
                e.param(ParamId::BrainOn) >= 0.5f ? kBrainModeNames[static_cast<int>(e.param(ParamId::AutoMode))] : "off", seconds);
    for (int note : chord) e.noteOn(note, 0.8f, 0);

    const int n = 512;
    const std::int64_t total = static_cast<std::int64_t>(seconds * c.sampleRate);
    std::vector<float> lr;
    lr.reserve(static_cast<size_t>(total) * 2);
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
        e.process(L.data(), R.data(), n);
        for (int i = 0; i < n; ++i) {
            lr.push_back(L[i]);
            lr.push_back(R[i]);
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
    fs::create_directories(out.parent_path());
    if (!writeWav24(out, lr, static_cast<int>(c.sampleRate))) { std::fprintf(stderr, "additiv_render: cannot write %s\n", out.string().c_str()); return 1; }
    std::printf("%.0f s in %.1f s (%.0fx real time), late notes %llu -> %s\n", seconds, took, seconds / std::max(took, 1e-9),
                static_cast<unsigned long long>(e.lateEvents()), out.string().c_str());
    return 0;
}
