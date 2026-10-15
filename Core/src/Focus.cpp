/**
 * @file Focus.cpp
 * @brief Nudges, the focus, variations and taught macros (Focus.h).
 */
#include "additiv/Focus.h"

#include "additiv/Modulation.h"
#include "additiv/Presets.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <map>
#include <string>
#include <vector>

namespace additiv {

const char* const NudgeLow[NudgeAxes] = { "Darker", "Calmer", "Narrower", "Purer" };
const char* const NudgeHigh[NudgeAxes] = { "Brighter", "Livelier", "Wider", "Airier" };

namespace {

/** @brief A shaping knob as the measure left it (Nudges.inc). */
struct NudgeKnob {
    const char* m_key;               ///< its key
    const char* m_gate[2];           ///< its gates' keys (null: fewer)
    float m_d[NumMeasures];         ///< its sensitivity per measure, for its whole travel
    int m_heard;                     ///< in how many measured presets it was heard
};

#include "Nudges.inc"

/** @brief A shaping knob, resolved. */
struct Knob {
    ParamIdE m_id = ParamIdE::COUNT;   ///< the knob
    ParamIdE m_gate[2] = { ParamIdE::COUNT, ParamIdE::COUNT };   ///< its gates (Count: none)
    float m_d[NumMeasures] = {};    ///< its sensitivity
};

/** @brief The table, resolved once. */
struct Table {
    std::vector<Knob> m_knobs;                   ///< the shaping knobs
    std::vector<ParamIdE> m_ids;                  ///< their ids
    int m_index[NumParams];                     ///< per ParamId its knob, -1 none
    /** @brief Resolves Nudges.inc against the parameter table. */
    Table()
    {
        std::fill(std::begin(m_index), std::end(m_index), -1);
        for (const NudgeKnob& n : NudgeKnobs) {
            const ParamDesc* d = findParam(n.m_key);
            if (d == nullptr) continue;
            Knob k;
            k.m_id = d->m_id;
            for (int g = 0; g < 2; ++g)
                if (n.m_gate[g] != nullptr)
                    if (const ParamDesc* gd = findParam(n.m_gate[g])) k.m_gate[g] = gd->m_id;
            std::copy(std::begin(n.m_d), std::end(n.m_d), std::begin(k.m_d));
            m_index[static_cast<int>(k.m_id)] = static_cast<int>(m_knobs.size());
            m_knobs.push_back(k);
            m_ids.push_back(k.m_id);
        }
    }
};

/** @brief The table. @return it */
const Table& table()
{
    static const Table t;
    return t;
}

/** @brief A knob's value moved by @p dn of its normalised travel, within it. @param id the knob @param v its value @param dn the move @return the new value */
float moved(ParamIdE id, float v, float dn)
{
    const ParamDesc& d = paramDesc(id);
    float x = fromNorm(d, std::clamp(toNorm(d, v) + dn, 0.0f, 1.0f));
    if (d.m_kind == ParamKindE::INT) x = std::round(x);
    return x;
}

/**
 * @brief Whether a nudge or a variation may move a knob: heard, and not an amount at nothing -- what is
 *        off stays off. Turning a layer on (the air, the melt, the hall) is no small step: measured, a
 *        nudge that lifted Air from 0 to 0.07 made a choir 5.9 dB louder and livelier where it asked for calmer.
 * @param id      the knob
 * @param values  the sound's knobs
 * @return        true when it may move
 */
bool movable(ParamIdE id, const float* values)
{
    const ParamDesc& d = paramDesc(id);
    if (d.m_min == 0.0f && values[static_cast<int>(id)] == 0.0f) return false;
    return paramHeard(id, values);
}

/** @brief A small generator for the variations (splitmix64). */
struct Dice {
    std::uint64_t m_s;   ///< the state
    /** @brief The next 64 bits. @return them */
    std::uint64_t next()
    {
        std::uint64_t z = (m_s += 0x9E3779B97F4A7C15ull);
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
        return z ^ (z >> 31);
    }
    /** @brief Uniform in [0, 1). @return it */
    double uniform() { return static_cast<double>(next() >> 11) * (1.0 / 9007199254740992.0); }
    /** @brief A standard normal (Box-Muller). @return it */
    double normal() { return std::sqrt(-2.0 * std::log(std::max(uniform(), 1e-12))) * std::cos(6.283185307179586 * uniform()); }
};

} // namespace

const ParamIdE* shapingParams(int& count)
{
    count = static_cast<int>(table().m_ids.size());
    return table().m_ids.data();
}

bool paramHeard(ParamIdE id, const float* values)
{
    const int i = static_cast<int>(id);
    if (i < 0 || i >= NumParams || table().m_index[i] < 0) return true;
    const Knob& k = table().m_knobs[static_cast<size_t>(table().m_index[i])];
    if (k.m_gate[0] == ParamIdE::COUNT) return true;
    for (ParamIdE g : k.m_gate)
        if (g != ParamIdE::COUNT && std::fabs(values[static_cast<int>(g)]) > 1.0e-6f) return true;
    return false;
}

bool sensitivity(ParamIdE id, float* out)
{
    const int i = static_cast<int>(id);
    if (i < 0 || i >= NumParams || table().m_index[i] < 0) {
        std::fill(out, out + NumMeasures, 0.0f);
        return false;
    }
    const Knob& k = table().m_knobs[static_cast<size_t>(table().m_index[i])];
    std::copy(std::begin(k.m_d), std::end(k.m_d), out);
    return true;
}

float measureSpread(MeasureE m) { return std::max(MeasureSpreads[static_cast<int>(m)], 1.0e-6f); }

float nudge(const float* values, int axis, float sign, float size, float* out)
{
    std::copy(values, values + NumParams, out);
    if (axis < 0 || axis >= NudgeAxes) return 0.0f;
    if (axis == static_cast<int>(MeasureE::MOTION)) {
        // Calmer and livelier by a rule, not by the measure: the motion measured over eight seconds is
        // too much at the mercy of slow cycles (a landscape's loop, LFOs of minutes) to steer by -- the
        // nudge check went the wrong way in six of sixteen. Calmer is slower: every rate that moves
        // something by a factor of 1.23 a quarter-spread step, drift and shimmer a little less; the
        // motion's own depth stays (on a landscape it moves the level too).
        static const ParamIdE Rates[] = { ParamIdE::MOTION_RATE, ParamIdE::DRIFT_RATE, ParamIdE::SHIMMER_RATE, ParamIdE::SYNC_RATE, ParamIdE::LIFE_RATE,
                                          ParamIdE::FIELD_RATE, ParamIdE::BREATH_RATE, ParamIdE::COMB_SWEEP, ParamIdE::RIPPLE_SPEED };
        static const ParamIdE Depths[] = { ParamIdE::DRIFT, ParamIdE::SHIMMER };
        const float factor = std::exp2(sign * size * 1.2f);
        double got = 0.0;
        for (ParamIdE id : Rates) {
            const int p = static_cast<int>(id);
            if (!movable(id, values) || values[p] == 0.0f) continue;
            const ParamDesc& d = paramDesc(id);
            out[p] = std::clamp(values[p] * factor, d.m_min, d.m_max);
            float s[NumMeasures];
            if (sensitivity(id, s)) got += s[axis] / measureSpread(MeasureE::MOTION) * (toNorm(d, out[p]) - toNorm(d, values[p]));
        }
        for (ParamIdE id : Depths) {
            const int p = static_cast<int>(id);
            if (!movable(id, values)) continue;
            const ParamDesc& d = paramDesc(id);
            out[p] = moved(id, values[p], sign * size * 0.16f);
            float s[NumMeasures];
            if (sensitivity(id, s)) got += s[axis] / measureSpread(MeasureE::MOTION) * (toNorm(d, out[p]) - toNorm(d, values[p]));
        }
        return static_cast<float>(got);
    }
    // The few knobs that move this measure most specifically: much of it, little of the others and of
    // the level (each in its spread). Measured (--nudgecheck, 8 presets): moving every heard knob a
    // little, with the other measures held by opposing moves, went the wrong way in a quarter of the
    // cases and jumped up to 4 dB; a handful of specific knobs is steadier, and can be told in words.
    struct Pick {
        const Knob* m_knob;   ///< the knob
        double m_g;           ///< what it does for the measure
        double m_l;           ///< and for the level
        double m_rank;        ///< how specifically
    };
    std::vector<Pick> picks;
    const double levelSpread = measureSpread(MeasureE::LEVEL);
    for (const Knob& k : table().m_knobs) {
        if (!movable(k.m_id, values)) continue;
        const double g = k.m_d[axis] / measureSpread(static_cast<MeasureE>(axis));
        if (std::fabs(g) < 0.05) continue;   // does little for it
        double side = std::fabs(k.m_d[static_cast<int>(MeasureE::LEVEL)]) / levelSpread;
        for (int m = 0; m < NudgeAxes; ++m)
            if (m != axis) side += std::fabs(k.m_d[m]) / measureSpread(static_cast<MeasureE>(m));
        picks.push_back({ &k, g, k.m_d[static_cast<int>(MeasureE::LEVEL)] / levelSpread, std::fabs(g) / (std::fabs(g) + side) * std::sqrt(std::fabs(g)) });
    }
    if (picks.empty()) return 0.0f;
    std::sort(picks.begin(), picks.end(), [](const Pick& a, const Pick& b) { return a.m_rank > b.m_rank; });
    if (picks.size() > 4) picks.resize(4);
    const size_t n = picks.size();
    // The shortest step over them; the level held when two or more leave room for it.
    std::vector<double> want(n);
    double gg = 0.0, gl = 0.0, ll = 0.0;
    for (size_t i = 0; i < n; ++i) {
        gg += picks[i].m_g * picks[i].m_g;
        gl += picks[i].m_g * picks[i].m_l;
        ll += picks[i].m_l * picks[i].m_l;
    }
    double ww = 0.0;
    for (size_t i = 0; i < n; ++i) {
        want[i] = picks[i].m_g - (n >= 2 && ll > 1.0e-12 ? gl / ll * picks[i].m_l : 0.0);
        ww += want[i] * want[i];
    }
    if (ww < gg / 4.0) {   // holding the level would leave too little: the plain step
        ww = gg;
        for (size_t i = 0; i < n; ++i) want[i] = picks[i].m_g;
    }
    if (ww < 1.0e-12) return 0.0f;
    double gw = 0.0;
    for (size_t i = 0; i < n; ++i) gw += picks[i].m_g * want[i];
    std::vector<double> step(n);
    double largest = 0.0;
    for (size_t i = 0; i < n; ++i) {
        step[i] = static_cast<double>(sign * size) * want[i] / std::max(gw, 1.0e-12);
        largest = std::max(largest, std::fabs(step[i]));
    }
    const double scale = largest > 0.2 ? 0.2 / largest : 1.0;   // no knob further than a fifth of its travel at once
    double got = 0.0;
    for (size_t i = 0; i < n; ++i) {
        const ParamIdE id = picks[i].m_knob->m_id;
        const int p = static_cast<int>(id);
        const ParamDesc& d = paramDesc(id);
        const float before = toNorm(d, values[p]);
        out[p] = moved(id, values[p], static_cast<float>(step[i] * scale));
        got += picks[i].m_g * static_cast<double>(toNorm(d, out[p]) - before);
    }
    return static_cast<float>(got);
}

int focusParams(const float* values, ParamIdE* out, int n)
{
    std::vector<std::pair<float, ParamIdE>> score;
    for (const Knob& k : table().m_knobs) {
        if (!paramHeard(k.m_id, values)) continue;
        float s = 0.0f;
        for (int m = 0; m < NudgeAxes; ++m) s += std::fabs(k.m_d[m]) / measureSpread(static_cast<MeasureE>(m));
        score.emplace_back(s, k.m_id);
    }
    std::sort(score.begin(), score.end(), [](const auto& a, const auto& b) { return a.first > b.first; });
    std::map<std::string, int> perSection;
    int m = 0;
    for (const auto& [s, id] : score) {
        if (m >= n) break;
        int& c = perSection[paramDesc(id).m_section];
        if (c >= 3) continue;
        ++c;
        out[m++] = id;
    }
    return m;
}

void vary(const float* values, float far, std::uint64_t seed, float* out)
{
    std::copy(values, values + NumParams, out);
    Dice dice{ seed * 0x2545F4914F6CDD1Dull + 1 };
    // The heard shaping knobs, weighted by what they do.
    std::vector<std::pair<float, const Knob*>> pool;
    float most = 0.0f;
    for (const Knob& k : table().m_knobs) {
        if (!movable(k.m_id, values)) continue;
        float w = 0.0f;
        for (int m = 0; m < NudgeAxes; ++m) w += std::fabs(k.m_d[m]) / measureSpread(static_cast<MeasureE>(m));
        if (w <= 1.0e-6f) continue;
        pool.emplace_back(w, &k);
        most = std::max(most, w);
    }
    const int count = std::min(static_cast<int>(pool.size()), 4 + static_cast<int>(dice.next() % 5));
    double level = 0.0;
    for (int c = 0; c < count; ++c) {
        float total = 0.0f;
        for (const auto& p : pool) total += p.first;
        float r = static_cast<float>(dice.uniform()) * total;
        size_t pick = 0;
        for (; pick + 1 < pool.size(); ++pick) {
            r -= pool[pick].first;
            if (r <= 0.0f) break;
        }
        const Knob& k = *pool[pick].second;
        // A larger move on a knob that does more; at Far 1 about a third of the travel.
        const float dn = far * 0.35f * static_cast<float>(dice.normal()) * (0.5f + 0.5f * pool[pick].first / std::max(most, 1.0e-6f));
        const ParamDesc& d = paramDesc(k.m_id);
        const int p = static_cast<int>(k.m_id);
        const float before = toNorm(d, values[p]);
        out[p] = moved(k.m_id, values[p], dn);
        level += static_cast<double>(k.m_d[static_cast<int>(MeasureE::LEVEL)]) * static_cast<double>(toNorm(d, out[p]) - before);
        pool.erase(pool.begin() + static_cast<std::ptrdiff_t>(pick));
        if (pool.empty()) break;
    }
    // The level held by the master, by what the knobs were measured to do to it.
    const ParamDesc& master = paramDesc(ParamIdE::MASTER_GAIN);
    out[static_cast<int>(ParamIdE::MASTER_GAIN)] = std::clamp(values[static_cast<int>(ParamIdE::MASTER_GAIN)] - static_cast<float>(std::clamp(level, -6.0, 6.0)),
                                                             master.m_min, master.m_max);
}

void MacroLesson::apply(float at, float* offset, bool* touched) const
{
    if (m_count < 2) return;
    int k = 0;
    float u = 0.0f;
    if (at <= m_pos[0]) {
        k = 0;
    } else if (at >= m_pos[m_count - 1]) {
        k = m_count - 2;
        u = 1.0f;
    } else {
        while (k + 2 < m_count && at > m_pos[k + 1]) ++k;
        u = (at - m_pos[k]) / std::max(m_pos[k + 1] - m_pos[k], 1.0e-6f);
    }
    for (int p = 0; p < NumParams; ++p) {
        const float d = m_delta[k][static_cast<size_t>(p)] + (m_delta[k + 1][static_cast<size_t>(p)] - m_delta[k][static_cast<size_t>(p)]) * u;
        if (d == 0.0f) continue;
        offset[p] += d;
        if (touched != nullptr) touched[p] = true;
    }
}

void MacroLesson::set(float at, const float* d)
{
    int slot = -1;
    for (int i = 0; i < m_count; ++i)
        if (std::fabs(m_pos[i] - at) < 0.02f) slot = i;
    if (slot < 0 && m_count < LessonPlaces) slot = m_count++;
    if (slot < 0) {   // all places taken: the nearest gives way
        slot = 0;
        for (int i = 1; i < m_count; ++i)
            if (std::fabs(m_pos[i] - at) < std::fabs(m_pos[slot] - at)) slot = i;
    }
    m_pos[slot] = at;
    std::copy(d, d + NumParams, m_delta[slot].begin());
    // Ascending by place.
    for (int i = 1; i < m_count; ++i)
        for (int j = i; j > 0 && m_pos[j] < m_pos[j - 1]; --j) {
            std::swap(m_pos[j], m_pos[j - 1]);
            std::swap(m_delta[j], m_delta[j - 1]);
        }
}

int MacroLesson::knobs() const
{
    int n = 0;
    for (int p = 0; p < NumParams; ++p) {
        bool any = false;
        for (int i = 0; i < m_count && !any; ++i) any = m_delta[i][static_cast<size_t>(p)] != 0.0f;
        n += any ? 1 : 0;
    }
    return n;
}

bool teachable(ParamIdE id)
{
    switch (id) {
    case ParamIdE::MACRO1: case ParamIdE::MACRO2: case ParamIdE::MACRO3: case ParamIdE::MACRO4:
    case ParamIdE::MAP_X: case ParamIdE::MAP_Y: case ParamIdE::MAP_RADIUS: case ParamIdE::MAP_ACTIVE: case ParamIdE::MAP_GLIDE:
    case ParamIdE::MORPH_ACTIVE: case ParamIdE::MORPH_POS: case ParamIdE::PRESET_GLIDE: case ParamIdE::ROUTE_SPEED:
        return false;
    default:
        return isModTarget(id) && !keptByPresets(id);
    }
}

} // namespace additiv
