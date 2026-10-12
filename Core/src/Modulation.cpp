/**
 * @file Modulation.cpp
 * @brief The modulation system (Modulation.h).
 */
#include "additiv/Modulation.h"

#include "additiv/Shapes.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <sstream>
#include <vector>

namespace additiv {

const char* const kModSourceNames[kNumModSources] = { "Off", "LFO 1", "LFO 2", "LFO 3", "LFO 4", "Env 1", "Env 2", "Macro 1", "Macro 2",
                                                      "Macro 3", "Macro 4", "Mod Wheel", "Pressure", "Velocity", "Random" };
const char* const kModSourceTokens[kNumModSources] = { "off", "lfo1", "lfo2", "lfo3", "lfo4", "env1", "env2", "macro1", "macro2",
                                                       "macro3", "macro4", "wheel", "pressure", "velocity", "random" };

namespace {

/**
 * @brief A factory shape's held course at time tau (the host's twin of the GPU's shapeHeld): its
 *        start, then its loop for ever, or its last level.
 * @param s    the shape
 * @param tau  seconds of shape time
 * @return     the level
 */
float shapeAt(const EnvShape& s, double tau)
{
    const int n = s.count;
    if (n <= 0) return 0.0f;
    const bool loop = s.loopStart >= 0 && s.loopEnd > s.loopStart && s.loopEnd < n;
    if (loop) {
        const double t0 = s.p[s.loopStart].time, len = s.p[s.loopEnd].time - t0;
        if (len > 0.0 && tau >= t0) tau = t0 + std::fmod(tau - t0, len);
    }
    if (tau <= s.p[0].time) return s.p[0].level;
    const int last = loop ? s.loopEnd : n - 1;
    if (tau >= s.p[last].time) return s.p[last].level;
    int i = 0;
    while (i + 1 < last && s.p[i + 1].time <= tau) ++i;
    const double span = s.p[i + 1].time - s.p[i].time;
    float x = span > 0.0 ? static_cast<float>((tau - s.p[i].time) / span) : 1.0f;
    const float c = s.p[i + 1].curve;
    if (c > 0.001f) x = std::pow(x, 1.0f + 5.0f * c);
    else if (c < -0.001f) x = 1.0f - std::pow(1.0f - x, 1.0f - 5.0f * c);
    return s.p[i].level + (s.p[i + 1].level - s.p[i].level) * x;
}

/** @brief The factory bank, made once. @return the shapes */
const std::vector<EnvShape>& bank()
{
    static const std::vector<EnvShape> b = factoryShapes();
    return b;
}

} // namespace

bool isModTarget(ParamId id)
{
    const int i = static_cast<int>(id);
    if (i < 0 || i >= kNumParams) return false;
    const ParamKind k = paramDesc(id).kind;
    return k == ParamKind::Float || k == ParamKind::Int;
}

std::uint64_t packModRow(const ModRow& r)
{
    std::uint32_t depth = 0;
    std::memcpy(&depth, &r.depth, sizeof depth);
    const std::uint64_t src = static_cast<std::uint64_t>(static_cast<std::uint8_t>(r.source));
    const std::uint64_t target = static_cast<std::uint64_t>(static_cast<std::uint16_t>(r.target < 0 ? 0xFFFF : r.target));
    return (src << 56) | (target << 32) | depth;
}

ModRow unpackModRow(std::uint64_t bits)
{
    ModRow r;
    const int src = static_cast<int>(bits >> 56);
    r.source = src < kNumModSources ? static_cast<ModSource>(src) : ModSource::Off;
    const int target = static_cast<int>((bits >> 32) & 0xFFFF);
    r.target = target == 0xFFFF ? -1 : target;
    const std::uint32_t depth = static_cast<std::uint32_t>(bits & 0xFFFFFFFFu);
    std::memcpy(&r.depth, &depth, sizeof depth);
    return r;
}

std::string modRowsToText(const ModRow* rows, int n)
{
    std::ostringstream s;
    bool first = true;
    for (int i = 0; i < n; ++i) {
        const ModRow& r = rows[i];
        if (r.source == ModSource::Off || r.target < 0 || r.target >= kNumParams) continue;
        if (!first) s << ';';
        s << kModSourceTokens[static_cast<int>(r.source)] << '>' << paramDesc(static_cast<ParamId>(r.target)).key << ':' << r.depth;
        first = false;
    }
    return s.str();
}

int modRowsFromText(const std::string& text, ModRow* rows, int n)
{
    for (int i = 0; i < n; ++i) rows[i] = ModRow{};
    int count = 0;
    size_t pos = 0;
    while (pos < text.size() && count < n) {
        size_t end = text.find(';', pos);
        if (end == std::string::npos) end = text.size();
        const std::string item = text.substr(pos, end - pos);
        pos = end + 1;
        const size_t gt = item.find('>'), colon = item.find(':');
        if (gt == std::string::npos || colon == std::string::npos || colon < gt) continue;
        const std::string src = item.substr(0, gt), key = item.substr(gt + 1, colon - gt - 1);
        int s = -1;
        for (int k = 1; k < kNumModSources; ++k)
            if (src == kModSourceTokens[k]) s = k;
        const ParamDesc* d = findParam(key.c_str());
        if (s < 0 || d == nullptr || !isModTarget(d->id)) continue;
        rows[count].source = static_cast<ModSource>(s);
        rows[count].target = static_cast<int>(d->id);
        rows[count].depth = std::clamp(static_cast<float>(std::atof(item.substr(colon + 1).c_str())), -1.0f, 1.0f);
        ++count;
    }
    return count;
}

void Modulators::reset(std::uint64_t seed)
{
    rng_.seed(seed);
    for (int i = 0; i < 4; ++i) {
        phase_[i] = 0.0;
        random_[i].init(rng_);
        stepFrom_[i] = stepTo_[i] = rng_.bipolar();
    }
    slow_.init(rng_);
    envTime_ = 0.0;
    std::fill(std::begin(values_), std::end(values_), 0.0f);
}

float Modulators::lfoValue(int i, int shape) const
{
    const float p = static_cast<float>(phase_[i]);
    switch (shape) {
    case 1:   // Triangle
        return 1.0f - 4.0f * std::fabs(p - 0.5f);
    case 2:   // Ramp up, falling back over the last 7 % (never a step)
        return p < 0.93f ? -1.0f + 2.0f * p / 0.93f : 1.0f - 2.0f * (p - 0.93f) / 0.07f;
    case 3:   // Ramp down
        return -(p < 0.93f ? -1.0f + 2.0f * p / 0.93f : 1.0f - 2.0f * (p - 0.93f) / 0.07f);
    case 4:   // Square with its edges ramped over about 7 % of a cycle
        return std::clamp(std::sin(kTwoPi * p) / std::sin(kTwoPi * 0.035f), -1.0f, 1.0f);
    case 5:   // Random: a smooth random curve at the rate
        return random_[i].value();
    case 6: { // Steps: a new random level every cycle, glided into over a tenth of it
        const float x = std::min(p / 0.1f, 1.0f);
        return stepFrom_[i] + (stepTo_[i] - stepFrom_[i]) * x * x * (3.0f - 2.0f * x);
    }
    default:  // Sine
        return std::sin(kTwoPi * p);
    }
}

void Modulators::advance(float dt, const ModInputs& in)
{
    for (int i = 0; i < 4; ++i) {
        const float rate = std::max(in.lfoRate[i], 0.0f);
        phase_[i] += static_cast<double>(rate) * dt;
        if (phase_[i] >= 1.0) {
            phase_[i] -= std::floor(phase_[i]);
            stepFrom_[i] = stepTo_[i];
            stepTo_[i] = rng_.bipolar();
        }
        random_[i].update(dt, rate, rng_);
        values_[static_cast<int>(ModSource::Lfo1) + i] = lfoValue(i, in.lfoShape[i]);
    }
    if (in.phraseStart) envTime_ = 0.0;
    envTime_ += dt;
    const std::vector<EnvShape>& b = bank();
    for (int e = 0; e < 2; ++e) {
        const int shape = std::clamp(in.envShape[e], 0, static_cast<int>(b.size()) - 1);
        values_[static_cast<int>(ModSource::Env1) + e] = shapeAt(b[static_cast<size_t>(shape)], envTime_ * std::max(in.envRate[e], 0.0f));
    }
    for (int m = 0; m < 4; ++m) values_[static_cast<int>(ModSource::Macro1) + m] = in.macro[m];
    values_[static_cast<int>(ModSource::ModWheel)] = in.wheel;
    values_[static_cast<int>(ModSource::Pressure)] = in.pressure;
    values_[static_cast<int>(ModSource::Velocity)] = in.velocity;
    values_[static_cast<int>(ModSource::Random)] = slow_.update(dt, 0.2f, rng_);
}

} // namespace additiv
