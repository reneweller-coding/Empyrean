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
                                                      "Macro 3", "Macro 4", "Mod Wheel", "Pressure", "Velocity", "Random",
                                                      "Lenia 1", "Lenia 2", "Lenia 3", "Lenia 4", "Lorenz X", "Lorenz Y", "Lorenz Z",
                                                      "Roessler X", "Roessler Y", "Roessler Z", "Kura 1", "Kura 2", "Kura 3", "Kura 4",
                                                      "Beat", "Cascade", "Root Age", "Note", "Distance", "Random per Note",
                                                      "Loudness", "Heard Brightness", "Change", "Roughness",
                                                      "Reservoir 1", "Reservoir 2", "Reservoir 3", "Reservoir 4" };
const char* const kModSourceTokens[kNumModSources] = { "off", "lfo1", "lfo2", "lfo3", "lfo4", "env1", "env2", "macro1", "macro2",
                                                       "macro3", "macro4", "wheel", "pressure", "velocity", "random",
                                                       "lenia1", "lenia2", "lenia3", "lenia4", "lorenz_x", "lorenz_y", "lorenz_z",
                                                       "rossler_x", "rossler_y", "rossler_z", "kura1", "kura2", "kura3", "kura4",
                                                       "beat", "cascade", "root_age", "note", "distance", "random_note",
                                                       "loudness", "centroid", "flux", "roughness",
                                                       "reservoir1", "reservoir2", "reservoir3", "reservoir4" };

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

bool isUnipolarSource(ModSource s)
{
    switch (s) {
    case ModSource::Env1: case ModSource::Env2:
    case ModSource::Macro1: case ModSource::Macro2: case ModSource::Macro3: case ModSource::Macro4:
    case ModSource::ModWheel: case ModSource::Pressure: case ModSource::Velocity:
    case ModSource::Lenia1: case ModSource::Lenia2: case ModSource::Lenia3: case ModSource::Lenia4:
    case ModSource::Cascade: case ModSource::RootAge: case ModSource::Note: case ModSource::Distance:
    case ModSource::Loudness: case ModSource::Centroid: case ModSource::Flux: case ModSource::Roughness:
        return true;
    default:
        return false;
    }
}

bool isModTarget(ParamId id)
{
    const int i = static_cast<int>(id);
    if (i < 0 || i >= kNumParams) return false;
    // The morph's position and the glide are where the base values come from, before the matrix.
    if (id == ParamId::MorphPos || id == ParamId::PresetGlide || id == ParamId::MapX || id == ParamId::MapY || id == ParamId::MapRadius
        || id == ParamId::MapGlide || id == ParamId::RouteSpeed) return false;
    const ParamKind k = paramDesc(id).kind;
    return k == ParamKind::Float || k == ParamKind::Int;
}

std::uint64_t packModRow(const ModRow& r)
{
    std::uint32_t depth = 0;
    std::memcpy(&depth, &r.depth, sizeof depth);
    const std::uint64_t src = static_cast<std::uint64_t>(static_cast<std::uint8_t>(r.source));
    const std::uint64_t target = static_cast<std::uint64_t>(static_cast<std::uint16_t>(r.target < 0 ? 0xFFFF : r.target));
    const std::uint64_t via = static_cast<std::uint64_t>(static_cast<std::uint8_t>(r.via));
    return (src << 56) | (via << 48) | (target << 32) | depth;
}

ModRow unpackModRow(std::uint64_t bits)
{
    ModRow r;
    const int src = static_cast<int>(bits >> 56);
    r.source = src < kNumModSources ? static_cast<ModSource>(src) : ModSource::Off;
    const int via = static_cast<int>((bits >> 48) & 0xFF);
    r.via = via < kNumModSources ? static_cast<ModSource>(via) : ModSource::Off;
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
        if (r.via != ModSource::Off) s << ':' << kModSourceTokens[static_cast<int>(r.via)];
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
        const size_t viaAt = item.find(':', colon + 1);   // an optional via source after the depth
        if (viaAt != std::string::npos) {
            const std::string via = item.substr(viaAt + 1);
            for (int k = 1; k < kNumModSources; ++k)
                if (via == kModSourceTokens[k]) rows[count].via = static_cast<ModSource>(k);
        }
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
    // Noctuary's fields and chaos, each from its own start: the same every time, as in Noctuary.
    leniaRng_.seed(0x1E51Aull);
    {
        constexpr int R = kLeniaRadius, K = 2 * R + 1;
        float sum = 0.0f;
        for (int dy = -R; dy <= R; ++dy)
            for (int dx = -R; dx <= R; ++dx) {
                const float r = std::sqrt(static_cast<float>(dx * dx + dy * dy)) / static_cast<float>(R);
                const float d = (r - 0.5f) / 0.15f;
                const float k = r <= 1.0f ? std::exp(-0.5f * d * d) : 0.0f;
                leniaKernel_[(dy + R) * K + (dx + R)] = k;
                sum += k;
            }
        for (float& k : leniaKernel_) k /= sum;
    }
    seedLenia();
    leniaRowAcc_ = 0.0f;
    for (int k = 0; k < 4; ++k) leniaTarget_[k] = leniaOut_[k] = 0.0f;
    lorenz_[0] = 0.1; lorenz_[1] = 0.0; lorenz_[2] = 20.0;
    rossler_[0] = 1.0; rossler_[1] = 1.0; rossler_[2] = 1.0;
    // Warmed up, so that a row reading them moves from the first second: the attractors run 25 of
    // Lorenz's time units (and 100 of Roessler's) onto their attractors, Lenia twenty generations at
    // the default growth -- the same every time.
    stepChaos(25.0f / 1.5f, 1.0f);
    for (int g = 0; g < 20; ++g) stepLenia(1.0f, 1.0f, 0.15f);
    for (int k = 0; k < 4; ++k) leniaOut_[k] = leniaTarget_[k];
    for (int k = 0; k < 4; ++k) kuraPhase_[k] = kTwoPi * static_cast<float>(k) * 0.25f;
    beatHz_ = 0.0f;
    beatPhase_ = 0.0f;
    cascade_ = 0.0f;
    for (float& h : heard_) h = 0.0f;
}

void Modulators::seedLenia()
{
    constexpr int S = kLeniaSize;
    for (float& v : lenia_) v = 0.0f;
    for (int b = 0; b < 3; ++b) {
        const float cx = leniaRng_.uniform() * S, cy = leniaRng_.uniform() * S;
        const float rad = 2.5f + 2.5f * leniaRng_.uniform();
        const float peak = 0.6f + 0.4f * leniaRng_.uniform();
        for (int y = 0; y < S; ++y)
            for (int x = 0; x < S; ++x) {
                float dx = std::fabs(static_cast<float>(x) - cx), dy = std::fabs(static_cast<float>(y) - cy);
                dx = std::min(dx, static_cast<float>(S) - dx);
                dy = std::min(dy, static_cast<float>(S) - dy);
                lenia_[y * S + x] = std::min(1.0f, lenia_[y * S + x] + peak * std::exp(-(dx * dx + dy * dy) / (2.0f * rad * rad)));
            }
    }
    std::memcpy(leniaNext_, lenia_, sizeof(lenia_));
    leniaRow_ = 0;
}

void Modulators::stepLenia(float dt, float rate, float mu)
{
    // Lenia (Chan 2019), as Noctuary runs it: a ring-shaped neighbourhood, a smooth growth bell,
    // a tenth of the way per step, rows stepped at the rate and the remainder carried over.
    constexpr int S = kLeniaSize, R = kLeniaRadius, K = 2 * R + 1;
    leniaRowAcc_ += static_cast<float>(S) * dt * std::max(rate, 0.01f);
    int rows = static_cast<int>(leniaRowAcc_);
    leniaRowAcc_ -= static_cast<float>(rows);
    rows = std::clamp(rows, 0, S);
    const float sigma = 0.02f;
    for (int pass = 0; pass < rows; ++pass) {
        const int y = leniaRow_;
        for (int x = 0; x < S; ++x) {
            float u = 0.0f;
            for (int dy = -R; dy <= R; ++dy) {
                const float* row = &lenia_[((y + dy + S) % S) * S];
                const float* krow = &leniaKernel_[(dy + R) * K];
                for (int dx = -R; dx <= R; ++dx) u += krow[dx + R] * row[(x + dx + S) % S];
            }
            const float d = (u - mu) / sigma;
            const float g = 2.0f * std::exp(-0.5f * d * d) - 1.0f;
            leniaNext_[y * S + x] = std::clamp(lenia_[y * S + x] + 0.1f * g, 0.0f, 1.0f);
        }
        if (++leniaRow_ >= S) {
            leniaRow_ = 0;
            std::memcpy(lenia_, leniaNext_, sizeof(lenia_));
            float mass = 0.0f;
            for (float v : lenia_) mass += v;
            if (mass < 3.0f || mass > 0.85f * static_cast<float>(S * S)) seedLenia();
            static const int px[4] = { 8, 24, 8, 24 }, py[4] = { 8, 8, 24, 24 };
            for (int k = 0; k < 4; ++k) {
                float s = 0.0f;
                for (int dy = -1; dy <= 1; ++dy)
                    for (int dx = -1; dx <= 1; ++dx) s += lenia_[((py[k] + dy) % S) * S + (px[k] + dx) % S];
                leniaTarget_[k] = s / 9.0f;
            }
        }
    }
    const float c = 1.0f - std::exp(-dt * std::max(rate, 0.01f));
    for (int k = 0; k < 4; ++k) leniaOut_[k] += c * (leniaTarget_[k] - leniaOut_[k]);
}

void Modulators::stepChaos(float dt, float period)
{
    // Noctuary's: Lorenz (10, 28, 8/3) and Roessler (0.2, 0.2, 5.7) by fourth-order Runge-Kutta in
    // their own time, substepped below a hundredth; a state that left the finite world is seeded again.
    auto rk4 = [](double* s, double h, auto&& f) {
        double k1[3], k2[3], k3[3], k4[3], t[3];
        f(s, k1);
        for (int i = 0; i < 3; ++i) t[i] = s[i] + 0.5 * h * k1[i];
        f(t, k2);
        for (int i = 0; i < 3; ++i) t[i] = s[i] + 0.5 * h * k2[i];
        f(t, k3);
        for (int i = 0; i < 3; ++i) t[i] = s[i] + h * k3[i];
        f(t, k4);
        for (int i = 0; i < 3; ++i) s[i] += h / 6.0 * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]);
    };
    auto lorenz = [](const double* s, double* d) { d[0] = 10.0 * (s[1] - s[0]); d[1] = s[0] * (28.0 - s[2]) - s[1]; d[2] = s[0] * s[1] - (8.0 / 3.0) * s[2]; };
    auto rossler = [](const double* s, double* d) { d[0] = -s[1] - s[2]; d[1] = s[0] + 0.2 * s[1]; d[2] = 0.2 + s[2] * (s[0] - 5.7); };
    const double p = std::max(static_cast<double>(period), 1.0);
    const double hL = static_cast<double>(dt) * 1.5 / p, hR = static_cast<double>(dt) * 6.0 / p;
    for (double left = hL; left > 0.0; left -= 0.01) rk4(lorenz_, std::min(left, 0.01), lorenz);
    for (double left = hR; left > 0.0; left -= 0.01) rk4(rossler_, std::min(left, 0.01), rossler);
    auto finite3 = [](const double* v) { return std::isfinite(v[0]) && std::isfinite(v[1]) && std::isfinite(v[2]); };
    if (!finite3(lorenz_)) { lorenz_[0] = 0.1; lorenz_[1] = 0.0; lorenz_[2] = 20.0; }
    if (!finite3(rossler_)) { rossler_[0] = 1.0; rossler_[1] = 1.0; rossler_[2] = 1.0; }
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

    // Lenia, the attractors.
    stepLenia(dt, in.leniaRate, in.leniaGrowth);
    for (int k = 0; k < 4; ++k) values_[static_cast<int>(ModSource::Lenia1) + k] = std::clamp(leniaOut_[k], 0.0f, 1.0f);
    stepChaos(dt, in.chaosPeriod);
    values_[static_cast<int>(ModSource::LorenzX)] = std::clamp(static_cast<float>(lorenz_[0] / 20.0), -1.0f, 1.0f);
    values_[static_cast<int>(ModSource::LorenzY)] = std::clamp(static_cast<float>(lorenz_[1] / 27.0), -1.0f, 1.0f);
    values_[static_cast<int>(ModSource::LorenzZ)] = std::clamp(static_cast<float>((lorenz_[2] - 25.0) / 22.0), -1.0f, 1.0f);
    values_[static_cast<int>(ModSource::RosslerX)] = std::clamp(static_cast<float>(rossler_[0] / 12.0), -1.0f, 1.0f);
    values_[static_cast<int>(ModSource::RosslerY)] = std::clamp(static_cast<float>(rossler_[1] / 12.0), -1.0f, 1.0f);
    values_[static_cast<int>(ModSource::RosslerZ)] = std::clamp(static_cast<float>((rossler_[2] - 8.0) / 12.0), -1.0f, 1.0f);

    // The Kuramoto bank (Noctuary's): natural periods 23/31/41/53 s over the rate, coupled with
    // slightly lopsided weights round the ring so that it locks in frequency but never into one
    // phase -- four oscillators behaving as one would be the opposite of what it is for.
    {
        const float rate = std::max(in.coherenceRate, 0.05f);
        const float K = 0.6f * std::clamp(in.coherence, 0.0f, 1.0f) * rate;
        static const float kPeriods[4] = { 23.0f, 31.0f, 41.0f, 53.0f };
        static const float kW[4] = { 1.0f, 1.22f, 1.0f, 0.78f };
        float dth[4];
        for (int i = 0; i < 4; ++i) {
            float coupling = 0.0f;
            if (K != 0.0f)
                for (int j = 0; j < 4; ++j) coupling += kW[(j - i) & 3] * std::sin(kuraPhase_[j] - kuraPhase_[i]);
            dth[i] = kTwoPi * rate / kPeriods[i] + K * coupling * 0.25f;
        }
        for (int i = 0; i < 4; ++i) {
            kuraPhase_[i] += dth[i] * dt;
            if (kuraPhase_[i] > kTwoPi) kuraPhase_[i] -= kTwoPi;
            if (kuraPhase_[i] < 0.0f) kuraPhase_[i] += kTwoPi;
            values_[static_cast<int>(ModSource::Kura1) + i] = std::sin(kuraPhase_[i]);
        }
    }

    // The Beat (Noctuary's): the interval's beat rate followed over two seconds, standing still
    // below a fiftieth of a hertz -- a chord in tune leaves what it drives where it is.
    beatHz_ += (1.0f - std::exp(-dt / 2.0f)) * (std::clamp(in.beatHz, 0.0f, 30.0f) - beatHz_);
    if (beatHz_ > 0.02f) {
        beatPhase_ += kTwoPi * beatHz_ * dt;
        while (beatPhase_ > kTwoPi) beatPhase_ -= kTwoPi;
    }
    values_[static_cast<int>(ModSource::Beat)] = std::sin(beatPhase_);

    // The conductor's cascade, as exc / (1 + exc) smoothed over two seconds (a step on a send is a click).
    const float exc = std::max(in.excitation, 0.0f);
    cascade_ += (1.0f - std::exp(-dt / 2.0f)) * (exc / (1.0f + exc) - cascade_);
    values_[static_cast<int>(ModSource::Cascade)] = cascade_;
    values_[static_cast<int>(ModSource::RootAge)] = std::clamp(in.rootAge / 600.0f, 0.0f, 1.0f);
    values_[static_cast<int>(ModSource::Note)] = std::clamp(in.note, 0.0f, 1.0f);
    values_[static_cast<int>(ModSource::Distance)] = std::clamp(in.distance, 0.0f, 1.0f);
    values_[static_cast<int>(ModSource::RandomNote)] = std::clamp(in.randomNote, -1.0f, 1.0f);

    // What it hears of itself, each followed at its own pace (a level jumps, a judgement settles).
    const float target[4] = { std::clamp((in.heardDb + 70.0f) / 60.0f, 0.0f, 1.0f), std::clamp(in.heardCentroid, 0.0f, 1.0f),
                              std::clamp(in.heardFlux * 4.0f, 0.0f, 1.0f), std::clamp(in.roughness, 0.0f, 1.0f) };
    const float tau[4] = { 0.33f, 0.5f, 0.2f, 1.0f };
    for (int k = 0; k < 4; ++k) {
        heard_[k] += (1.0f - std::exp(-dt / tau[k])) * (target[k] - heard_[k]);
        values_[static_cast<int>(ModSource::Loudness) + k] = heard_[k];
    }
    for (int k = 0; k < 4; ++k) values_[static_cast<int>(ModSource::Reservoir1) + k] = std::clamp(in.reservoir[k], -1.0f, 1.0f);
}

} // namespace additiv
