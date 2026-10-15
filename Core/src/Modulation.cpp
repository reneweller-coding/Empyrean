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

const char* const ModSourceNames[NumModSources] = { "Off", "LFO 1", "LFO 2", "LFO 3", "LFO 4", "Env 1", "Env 2", "Macro 1", "Macro 2",
                                                      "Macro 3", "Macro 4", "Mod Wheel", "Pressure", "Velocity", "Random",
                                                      "Lenia 1", "Lenia 2", "Lenia 3", "Lenia 4", "Lorenz X", "Lorenz Y", "Lorenz Z",
                                                      "Roessler X", "Roessler Y", "Roessler Z", "Kura 1", "Kura 2", "Kura 3", "Kura 4",
                                                      "Beat", "Cascade", "Root Age", "Note", "Distance", "Random per Note",
                                                      "Loudness", "Heard Brightness", "Change", "Roughness",
                                                      "Reservoir 1", "Reservoir 2", "Reservoir 3", "Reservoir 4" };
const char* const ModSourceTokens[NumModSources] = { "off", "lfo1", "lfo2", "lfo3", "lfo4", "env1", "env2", "macro1", "macro2",
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
    const int n = s.m_count;
    if (n <= 0) return 0.0f;
    const bool loop = s.m_loopStart >= 0 && s.m_loopEnd > s.m_loopStart && s.m_loopEnd < n;
    if (loop) {
        const double t0 = s.m_p[s.m_loopStart].m_time, len = s.m_p[s.m_loopEnd].m_time - t0;
        if (len > 0.0 && tau >= t0) tau = t0 + std::fmod(tau - t0, len);
    }
    if (tau <= s.m_p[0].m_time) return s.m_p[0].m_level;
    const int last = loop ? s.m_loopEnd : n - 1;
    if (tau >= s.m_p[last].m_time) return s.m_p[last].m_level;
    int i = 0;
    while (i + 1 < last && s.m_p[i + 1].m_time <= tau) ++i;
    const double span = s.m_p[i + 1].m_time - s.m_p[i].m_time;
    float x = span > 0.0 ? static_cast<float>((tau - s.m_p[i].m_time) / span) : 1.0f;
    const float c = s.m_p[i + 1].m_curve;
    if (c > 0.001f) x = std::pow(x, 1.0f + 5.0f * c);
    else if (c < -0.001f) x = 1.0f - std::pow(1.0f - x, 1.0f - 5.0f * c);
    return s.m_p[i].m_level + (s.m_p[i + 1].m_level - s.m_p[i].m_level) * x;
}

/** @brief The factory bank, made once. @return the shapes */
const std::vector<EnvShape>& bank()
{
    static const std::vector<EnvShape> b = factoryShapes();
    return b;
}

} // namespace

bool isUnipolarSource(ModSourceE s)
{
    switch (s) {
    case ModSourceE::ENV1: case ModSourceE::ENV2:
    case ModSourceE::MACRO1: case ModSourceE::MACRO2: case ModSourceE::MACRO3: case ModSourceE::MACRO4:
    case ModSourceE::MOD_WHEEL: case ModSourceE::PRESSURE: case ModSourceE::VELOCITY:
    case ModSourceE::LENIA1: case ModSourceE::LENIA2: case ModSourceE::LENIA3: case ModSourceE::LENIA4:
    case ModSourceE::CASCADE: case ModSourceE::ROOT_AGE: case ModSourceE::NOTE: case ModSourceE::DISTANCE:
    case ModSourceE::LOUDNESS: case ModSourceE::CENTROID: case ModSourceE::FLUX: case ModSourceE::ROUGHNESS:
        return true;
    default:
        return false;
    }
}

bool isModTarget(ParamIdE id)
{
    const int i = static_cast<int>(id);
    if (i < 0 || i >= NumParams) return false;
    // The morph's position and the glide are where the base values come from, before the matrix.
    if (id == ParamIdE::MORPH_POS || id == ParamIdE::PRESET_GLIDE || id == ParamIdE::MAP_X || id == ParamIdE::MAP_Y || id == ParamIdE::MAP_RADIUS
        || id == ParamIdE::MAP_GLIDE || id == ParamIdE::ROUTE_SPEED || id == ParamIdE::MPE_BEND) return false;
    const ParamKindE k = paramDesc(id).m_kind;
    return k == ParamKindE::FLOAT || k == ParamKindE::INT;
}

std::uint64_t packModRow(const ModRow& r)
{
    std::uint32_t depth = 0;
    std::memcpy(&depth, &r.m_depth, sizeof depth);
    const std::uint64_t src = static_cast<std::uint64_t>(static_cast<std::uint8_t>(r.m_source));
    const std::uint64_t target = static_cast<std::uint64_t>(static_cast<std::uint16_t>(r.m_target < 0 ? 0xFFFF : r.m_target));
    const std::uint64_t via = static_cast<std::uint64_t>(static_cast<std::uint8_t>(r.m_via));
    return (src << 56) | (via << 48) | (target << 32) | depth;
}

ModRow unpackModRow(std::uint64_t bits)
{
    ModRow r;
    const int src = static_cast<int>(bits >> 56);
    r.m_source = src < NumModSources ? static_cast<ModSourceE>(src) : ModSourceE::OFF;
    const int via = static_cast<int>((bits >> 48) & 0xFF);
    r.m_via = via < NumModSources ? static_cast<ModSourceE>(via) : ModSourceE::OFF;
    const int target = static_cast<int>((bits >> 32) & 0xFFFF);
    r.m_target = target == 0xFFFF ? -1 : target;
    const std::uint32_t depth = static_cast<std::uint32_t>(bits & 0xFFFFFFFFu);
    std::memcpy(&r.m_depth, &depth, sizeof depth);
    return r;
}

std::string modRowsToText(const ModRow* rows, int n)
{
    std::ostringstream s;
    bool first = true;
    for (int i = 0; i < n; ++i) {
        const ModRow& r = rows[i];
        if (r.m_source == ModSourceE::OFF || r.m_target < 0 || r.m_target >= NumParams) continue;
        if (!first) s << ';';
        s << ModSourceTokens[static_cast<int>(r.m_source)] << '>' << paramDesc(static_cast<ParamIdE>(r.m_target)).m_key << ':' << r.m_depth;
        if (r.m_via != ModSourceE::OFF) s << ':' << ModSourceTokens[static_cast<int>(r.m_via)];
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
        for (int k = 1; k < NumModSources; ++k)
            if (src == ModSourceTokens[k]) s = k;
        const ParamDesc* d = findParam(key.c_str());
        if (s < 0 || d == nullptr || !isModTarget(d->m_id)) continue;
        rows[count].m_source = static_cast<ModSourceE>(s);
        rows[count].m_target = static_cast<int>(d->m_id);
        rows[count].m_depth = std::clamp(static_cast<float>(std::atof(item.substr(colon + 1).c_str())), -1.0f, 1.0f);
        const size_t viaAt = item.find(':', colon + 1);   // an optional via source after the depth
        if (viaAt != std::string::npos) {
            const std::string via = item.substr(viaAt + 1);
            for (int k = 1; k < NumModSources; ++k)
                if (via == ModSourceTokens[k]) rows[count].m_via = static_cast<ModSourceE>(k);
        }
        ++count;
    }
    return count;
}

bool ModSetup::operator==(const ModSetup& o) const
{
    for (int i = 0; i < 4; ++i)
        if (m_lfoShape[i] != o.m_lfoShape[i] || m_lfoRate[i] != o.m_lfoRate[i]) return false;
    for (int e = 0; e < 2; ++e)
        if (m_envShape[e] != o.m_envShape[e] || m_envRate[e] != o.m_envRate[e]) return false;
    return true;
}

ModSetup modSetupFrom(const float* values)
{
    ModSetup s;
    for (int i = 0; i < 4; ++i) {
        s.m_lfoShape[i] = static_cast<int>(std::lround(values[static_cast<int>(ParamIdE::LFO1_SHAPE) + 2 * i]));
        s.m_lfoRate[i] = values[static_cast<int>(ParamIdE::LFO1_RATE) + 2 * i];
    }
    for (int e = 0; e < 2; ++e) {
        s.m_envShape[e] = static_cast<int>(std::lround(values[static_cast<int>(ParamIdE::ENV1_SHAPE) + 2 * e]));
        s.m_envRate[e] = values[static_cast<int>(ParamIdE::ENV1_RATE) + 2 * e];
    }
    return s;
}

void OwnModulators::reset(std::uint64_t seed)
{
    m_rng.seed(seed);
    for (int i = 0; i < 4; ++i) {
        m_phase[i] = 0.0;
        m_random[i].init(m_rng);
        m_stepFrom[i] = m_stepTo[i] = m_rng.bipolar();
    }
    std::fill(std::begin(m_values), std::end(m_values), 0.0f);
}

void OwnModulators::advance(float dt, double envTime, const ModSetup& s)
{
    for (int i = 0; i < 4; ++i) {
        const float rate = std::max(s.m_lfoRate[i], 0.0f);
        m_phase[i] += static_cast<double>(rate) * dt;
        if (m_phase[i] >= 1.0) {
            m_phase[i] -= std::floor(m_phase[i]);
            m_stepFrom[i] = m_stepTo[i];
            m_stepTo[i] = m_rng.bipolar();
        }
        m_random[i].update(dt, rate, m_rng);
        m_values[i] = lfoValue(i, s.m_lfoShape[i]);
    }
    const std::vector<EnvShape>& b = bank();
    for (int e = 0; e < 2; ++e) {
        const int shape = std::clamp(s.m_envShape[e], 0, static_cast<int>(b.size()) - 1);
        m_values[4 + e] = shapeAt(b[static_cast<size_t>(shape)], envTime * std::max(s.m_envRate[e], 0.0f));
    }
}

float OwnModulators::value(ModSourceE s) const
{
    return isOwnSource(s) ? m_values[static_cast<int>(s) - static_cast<int>(ModSourceE::LFO1)] : 0.0f;
}

void Modulators::reset(std::uint64_t seed)
{
    m_rng.seed(seed);
    m_own.reset(seed ^ 0x4F574E4D4F44ull);
    m_slow.init(m_rng);
    m_envTime = 0.0;
    std::fill(std::begin(m_values), std::end(m_values), 0.0f);
    // Noctuary's fields and chaos, each from its own start: the same every time, as in Noctuary.
    m_leniaRng.seed(0x1E51Aull);
    {
        constexpr int R = LeniaRadius, K = 2 * R + 1;
        float sum = 0.0f;
        for (int dy = -R; dy <= R; ++dy)
            for (int dx = -R; dx <= R; ++dx) {
                const float r = std::sqrt(static_cast<float>(dx * dx + dy * dy)) / static_cast<float>(R);
                const float d = (r - 0.5f) / 0.15f;
                const float k = r <= 1.0f ? std::exp(-0.5f * d * d) : 0.0f;
                m_leniaKernel[(dy + R) * K + (dx + R)] = k;
                sum += k;
            }
        for (float& k : m_leniaKernel) k /= sum;
    }
    seedLenia();
    m_leniaRowAcc = 0.0f;
    for (int k = 0; k < 4; ++k) m_leniaTarget[k] = m_leniaOut[k] = 0.0f;
    m_lorenz[0] = 0.1; m_lorenz[1] = 0.0; m_lorenz[2] = 20.0;
    m_rossler[0] = 1.0; m_rossler[1] = 1.0; m_rossler[2] = 1.0;
    // Warmed up, so that a row reading them moves from the first second: the attractors run 25 of
    // Lorenz's time units (and 100 of Roessler's) onto their attractors, Lenia twenty generations at
    // the default growth -- the same every time.
    stepChaos(25.0f / 1.5f, 1.0f);
    for (int g = 0; g < 20; ++g) stepLenia(1.0f, 1.0f, 0.15f);
    for (int k = 0; k < 4; ++k) m_leniaOut[k] = m_leniaTarget[k];
    for (int k = 0; k < 4; ++k) m_kuraPhase[k] = TwoPi * static_cast<float>(k) * 0.25f;
    m_beatHz = 0.0f;
    m_beatPhase = 0.0f;
    m_cascade = 0.0f;
    for (float& h : m_heard) h = 0.0f;
}

void Modulators::seedLenia()
{
    constexpr int S = LeniaSize;
    for (float& v : m_lenia) v = 0.0f;
    for (int b = 0; b < 3; ++b) {
        const float cx = m_leniaRng.uniform() * S, cy = m_leniaRng.uniform() * S;
        const float rad = 2.5f + 2.5f * m_leniaRng.uniform();
        const float peak = 0.6f + 0.4f * m_leniaRng.uniform();
        for (int y = 0; y < S; ++y)
            for (int x = 0; x < S; ++x) {
                float dx = std::fabs(static_cast<float>(x) - cx), dy = std::fabs(static_cast<float>(y) - cy);
                dx = std::min(dx, static_cast<float>(S) - dx);
                dy = std::min(dy, static_cast<float>(S) - dy);
                m_lenia[y * S + x] = std::min(1.0f, m_lenia[y * S + x] + peak * std::exp(-(dx * dx + dy * dy) / (2.0f * rad * rad)));
            }
    }
    std::memcpy(m_leniaNext, m_lenia, sizeof(m_lenia));
    m_leniaRow = 0;
}

void Modulators::stepLenia(float dt, float rate, float mu)
{
    // Lenia (Chan 2019), as Noctuary runs it: a ring-shaped neighbourhood, a smooth growth bell,
    // a tenth of the way per step, rows stepped at the rate and the remainder carried over.
    constexpr int S = LeniaSize, R = LeniaRadius, K = 2 * R + 1;
    m_leniaRowAcc += static_cast<float>(S) * dt * std::max(rate, 0.01f);
    int rows = static_cast<int>(m_leniaRowAcc);
    m_leniaRowAcc -= static_cast<float>(rows);
    rows = std::clamp(rows, 0, S);
    const float sigma = 0.02f;
    for (int pass = 0; pass < rows; ++pass) {
        const int y = m_leniaRow;
        for (int x = 0; x < S; ++x) {
            float u = 0.0f;
            for (int dy = -R; dy <= R; ++dy) {
                const float* row = &m_lenia[((y + dy + S) % S) * S];
                const float* krow = &m_leniaKernel[(dy + R) * K];
                for (int dx = -R; dx <= R; ++dx) u += krow[dx + R] * row[(x + dx + S) % S];
            }
            const float d = (u - mu) / sigma;
            const float g = 2.0f * std::exp(-0.5f * d * d) - 1.0f;
            m_leniaNext[y * S + x] = std::clamp(m_lenia[y * S + x] + 0.1f * g, 0.0f, 1.0f);
        }
        if (++m_leniaRow >= S) {
            m_leniaRow = 0;
            std::memcpy(m_lenia, m_leniaNext, sizeof(m_lenia));
            float mass = 0.0f;
            for (float v : m_lenia) mass += v;
            if (mass < 3.0f || mass > 0.85f * static_cast<float>(S * S)) seedLenia();
            static const int px[4] = { 8, 24, 8, 24 }, py[4] = { 8, 8, 24, 24 };
            for (int k = 0; k < 4; ++k) {
                float s = 0.0f;
                for (int dy = -1; dy <= 1; ++dy)
                    for (int dx = -1; dx <= 1; ++dx) s += m_lenia[((py[k] + dy) % S) * S + (px[k] + dx) % S];
                m_leniaTarget[k] = s / 9.0f;
            }
        }
    }
    const float c = 1.0f - std::exp(-dt * std::max(rate, 0.01f));
    for (int k = 0; k < 4; ++k) m_leniaOut[k] += c * (m_leniaTarget[k] - m_leniaOut[k]);
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
    for (double left = hL; left > 0.0; left -= 0.01) rk4(m_lorenz, std::min(left, 0.01), lorenz);
    for (double left = hR; left > 0.0; left -= 0.01) rk4(m_rossler, std::min(left, 0.01), rossler);
    auto finite3 = [](const double* v) { return std::isfinite(v[0]) && std::isfinite(v[1]) && std::isfinite(v[2]); };
    if (!finite3(m_lorenz)) { m_lorenz[0] = 0.1; m_lorenz[1] = 0.0; m_lorenz[2] = 20.0; }
    if (!finite3(m_rossler)) { m_rossler[0] = 1.0; m_rossler[1] = 1.0; m_rossler[2] = 1.0; }
}

float OwnModulators::lfoValue(int i, int shape) const
{
    const float p = static_cast<float>(m_phase[i]);
    switch (shape) {
    case 1:   // Triangle
        return 1.0f - 4.0f * std::fabs(p - 0.5f);
    case 2:   // Ramp up, falling back over the last 7 % (never a step)
        return p < 0.93f ? -1.0f + 2.0f * p / 0.93f : 1.0f - 2.0f * (p - 0.93f) / 0.07f;
    case 3:   // Ramp down
        return -(p < 0.93f ? -1.0f + 2.0f * p / 0.93f : 1.0f - 2.0f * (p - 0.93f) / 0.07f);
    case 4:   // Square with its edges ramped over about 7 % of a cycle
        return std::clamp(std::sin(TwoPi * p) / std::sin(TwoPi * 0.035f), -1.0f, 1.0f);
    case 5:   // Random: a smooth random curve at the rate
        return m_random[i].value();
    case 6: { // Steps: a new random level every cycle, glided into over a tenth of it
        const float x = std::min(p / 0.1f, 1.0f);
        return m_stepFrom[i] + (m_stepTo[i] - m_stepFrom[i]) * x * x * (3.0f - 2.0f * x);
    }
    default:  // Sine
        return std::sin(TwoPi * p);
    }
}

void Modulators::advance(float dt, const ModInputs& in)
{
    if (in.m_phraseStart) m_envTime = 0.0;
    m_envTime += dt;
    {
        ModSetup s;
        for (int i = 0; i < 4; ++i) { s.m_lfoShape[i] = in.m_lfoShape[i]; s.m_lfoRate[i] = in.m_lfoRate[i]; }
        for (int e = 0; e < 2; ++e) { s.m_envShape[e] = in.m_envShape[e]; s.m_envRate[e] = in.m_envRate[e]; }
        m_own.advance(dt, m_envTime, s);
        for (int k = static_cast<int>(ModSourceE::LFO1); k <= static_cast<int>(ModSourceE::ENV2); ++k)
            m_values[k] = m_own.value(static_cast<ModSourceE>(k));
    }
    for (int m = 0; m < 4; ++m) m_values[static_cast<int>(ModSourceE::MACRO1) + m] = in.m_macro[m];
    m_values[static_cast<int>(ModSourceE::MOD_WHEEL)] = in.m_wheel;
    m_values[static_cast<int>(ModSourceE::PRESSURE)] = in.m_pressure;
    m_values[static_cast<int>(ModSourceE::VELOCITY)] = in.m_velocity;
    m_values[static_cast<int>(ModSourceE::RANDOM)] = m_slow.update(dt, 0.2f, m_rng);

    // Lenia, the attractors.
    stepLenia(dt, in.m_leniaRate, in.m_leniaGrowth);
    for (int k = 0; k < 4; ++k) m_values[static_cast<int>(ModSourceE::LENIA1) + k] = std::clamp(m_leniaOut[k], 0.0f, 1.0f);
    stepChaos(dt, in.m_chaosPeriod);
    m_values[static_cast<int>(ModSourceE::LORENZ_X)] = std::clamp(static_cast<float>(m_lorenz[0] / 20.0), -1.0f, 1.0f);
    m_values[static_cast<int>(ModSourceE::LORENZ_Y)] = std::clamp(static_cast<float>(m_lorenz[1] / 27.0), -1.0f, 1.0f);
    m_values[static_cast<int>(ModSourceE::LORENZ_Z)] = std::clamp(static_cast<float>((m_lorenz[2] - 25.0) / 22.0), -1.0f, 1.0f);
    m_values[static_cast<int>(ModSourceE::ROSSLER_X)] = std::clamp(static_cast<float>(m_rossler[0] / 12.0), -1.0f, 1.0f);
    m_values[static_cast<int>(ModSourceE::ROSSLER_Y)] = std::clamp(static_cast<float>(m_rossler[1] / 12.0), -1.0f, 1.0f);
    m_values[static_cast<int>(ModSourceE::ROSSLER_Z)] = std::clamp(static_cast<float>((m_rossler[2] - 8.0) / 12.0), -1.0f, 1.0f);

    // The Kuramoto bank (Noctuary's): natural periods 23/31/41/53 s over the rate, coupled with
    // slightly lopsided weights round the ring so that it locks in frequency but never into one
    // phase -- four oscillators behaving as one would be the opposite of what it is for.
    {
        const float rate = std::max(in.m_coherenceRate, 0.05f);
        const float K = 0.6f * std::clamp(in.m_coherence, 0.0f, 1.0f) * rate;
        static const float Periods[4] = { 23.0f, 31.0f, 41.0f, 53.0f };
        static const float W[4] = { 1.0f, 1.22f, 1.0f, 0.78f };
        float dth[4];
        for (int i = 0; i < 4; ++i) {
            float coupling = 0.0f;
            if (K != 0.0f)
                for (int j = 0; j < 4; ++j) coupling += W[(j - i) & 3] * std::sin(m_kuraPhase[j] - m_kuraPhase[i]);
            dth[i] = TwoPi * rate / Periods[i] + K * coupling * 0.25f;
        }
        for (int i = 0; i < 4; ++i) {
            m_kuraPhase[i] += dth[i] * dt;
            if (m_kuraPhase[i] > TwoPi) m_kuraPhase[i] -= TwoPi;
            if (m_kuraPhase[i] < 0.0f) m_kuraPhase[i] += TwoPi;
            m_values[static_cast<int>(ModSourceE::KURA1) + i] = std::sin(m_kuraPhase[i]);
        }
    }

    // The Beat (Noctuary's): the interval's beat rate followed over two seconds, standing still
    // below a fiftieth of a hertz -- a chord in tune leaves what it drives where it is.
    m_beatHz += (1.0f - std::exp(-dt / 2.0f)) * (std::clamp(in.m_beatHz, 0.0f, 30.0f) - m_beatHz);
    if (m_beatHz > 0.02f) {
        m_beatPhase += TwoPi * m_beatHz * dt;
        while (m_beatPhase > TwoPi) m_beatPhase -= TwoPi;
    }
    m_values[static_cast<int>(ModSourceE::BEAT)] = std::sin(m_beatPhase);

    // The conductor's cascade, as exc / (1 + exc) smoothed over two seconds (a step on a send is a click).
    const float exc = std::max(in.m_excitation, 0.0f);
    m_cascade += (1.0f - std::exp(-dt / 2.0f)) * (exc / (1.0f + exc) - m_cascade);
    m_values[static_cast<int>(ModSourceE::CASCADE)] = m_cascade;
    m_values[static_cast<int>(ModSourceE::ROOT_AGE)] = std::clamp(in.m_rootAge / 600.0f, 0.0f, 1.0f);
    m_values[static_cast<int>(ModSourceE::NOTE)] = std::clamp(in.m_note, 0.0f, 1.0f);
    m_values[static_cast<int>(ModSourceE::DISTANCE)] = std::clamp(in.m_distance, 0.0f, 1.0f);
    m_values[static_cast<int>(ModSourceE::RANDOM_NOTE)] = std::clamp(in.m_randomNote, -1.0f, 1.0f);

    // What it hears of itself, each followed at its own pace (a level jumps, a judgement settles).
    const float target[4] = { std::clamp((in.m_heardDb + 70.0f) / 60.0f, 0.0f, 1.0f), std::clamp(in.m_heardCentroid, 0.0f, 1.0f),
                              std::clamp(in.m_heardFlux * 4.0f, 0.0f, 1.0f), std::clamp(in.m_roughness, 0.0f, 1.0f) };
    const float tau[4] = { 0.33f, 0.5f, 0.2f, 1.0f };
    for (int k = 0; k < 4; ++k) {
        m_heard[k] += (1.0f - std::exp(-dt / tau[k])) * (target[k] - m_heard[k]);
        m_values[static_cast<int>(ModSourceE::LOUDNESS) + k] = m_heard[k];
    }
    for (int k = 0; k < 4; ++k) m_values[static_cast<int>(ModSourceE::RESERVOIR1) + k] = std::clamp(in.m_reservoir[k], -1.0f, 1.0f);
}

} // namespace additiv
