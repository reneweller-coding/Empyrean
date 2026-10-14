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

const char* const kNudgeLow[kNudgeAxes] = { "Darker", "Calmer", "Narrower", "Purer" };
const char* const kNudgeHigh[kNudgeAxes] = { "Brighter", "Livelier", "Wider", "Airier" };

namespace {

/** @brief A shaping knob as the measure left it (Nudges.inc). */
struct NudgeKnob {
    const char* key;               ///< its key
    const char* gate[2];           ///< its gates' keys (null: fewer)
    float d[kNumMeasures];         ///< its sensitivity per measure, for its whole travel
    int heard;                     ///< in how many measured presets it was heard
};

#include "Nudges.inc"

/** @brief A shaping knob, resolved. */
struct Knob {
    ParamId id = ParamId::Count;   ///< the knob
    ParamId gate[2] = { ParamId::Count, ParamId::Count };   ///< its gates (Count: none)
    float d[kNumMeasures] = {};    ///< its sensitivity
};

/** @brief The table, resolved once. */
struct Table {
    std::vector<Knob> knobs;                   ///< the shaping knobs
    std::vector<ParamId> ids;                  ///< their ids
    int index[kNumParams];                     ///< per ParamId its knob, -1 none
    /** @brief Resolves Nudges.inc against the parameter table. */
    Table()
    {
        std::fill(std::begin(index), std::end(index), -1);
        for (const NudgeKnob& n : kNudgeKnobs) {
            const ParamDesc* d = findParam(n.key);
            if (d == nullptr) continue;
            Knob k;
            k.id = d->id;
            for (int g = 0; g < 2; ++g)
                if (n.gate[g] != nullptr)
                    if (const ParamDesc* gd = findParam(n.gate[g])) k.gate[g] = gd->id;
            std::copy(std::begin(n.d), std::end(n.d), std::begin(k.d));
            index[static_cast<int>(k.id)] = static_cast<int>(knobs.size());
            knobs.push_back(k);
            ids.push_back(k.id);
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
float moved(ParamId id, float v, float dn)
{
    const ParamDesc& d = paramDesc(id);
    float x = fromNorm(d, std::clamp(toNorm(d, v) + dn, 0.0f, 1.0f));
    if (d.kind == ParamKind::Int) x = std::round(x);
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
bool movable(ParamId id, const float* values)
{
    const ParamDesc& d = paramDesc(id);
    if (d.min == 0.0f && values[static_cast<int>(id)] == 0.0f) return false;
    return paramHeard(id, values);
}

/** @brief A small generator for the variations (splitmix64). */
struct Dice {
    std::uint64_t s;   ///< the state
    /** @brief The next 64 bits. @return them */
    std::uint64_t next()
    {
        std::uint64_t z = (s += 0x9E3779B97F4A7C15ull);
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

const ParamId* shapingParams(int& count)
{
    count = static_cast<int>(table().ids.size());
    return table().ids.data();
}

bool paramHeard(ParamId id, const float* values)
{
    const int i = static_cast<int>(id);
    if (i < 0 || i >= kNumParams || table().index[i] < 0) return true;
    const Knob& k = table().knobs[static_cast<size_t>(table().index[i])];
    if (k.gate[0] == ParamId::Count) return true;
    for (ParamId g : k.gate)
        if (g != ParamId::Count && std::fabs(values[static_cast<int>(g)]) > 1.0e-6f) return true;
    return false;
}

bool sensitivity(ParamId id, float* out)
{
    const int i = static_cast<int>(id);
    if (i < 0 || i >= kNumParams || table().index[i] < 0) {
        std::fill(out, out + kNumMeasures, 0.0f);
        return false;
    }
    const Knob& k = table().knobs[static_cast<size_t>(table().index[i])];
    std::copy(std::begin(k.d), std::end(k.d), out);
    return true;
}

float measureSpread(Measure m) { return std::max(kMeasureSpreads[static_cast<int>(m)], 1.0e-6f); }

float nudge(const float* values, int axis, float sign, float size, float* out)
{
    std::copy(values, values + kNumParams, out);
    if (axis < 0 || axis >= kNudgeAxes) return 0.0f;
    if (axis == static_cast<int>(Measure::Motion)) {
        // Calmer and livelier by a rule, not by the measure: the motion measured over eight seconds is
        // too much at the mercy of slow cycles (a landscape's loop, LFOs of minutes) to steer by -- the
        // nudge check went the wrong way in six of sixteen. Calmer is slower: every rate that moves
        // something by a factor of 1.23 a quarter-spread step, drift and shimmer a little less; the
        // motion's own depth stays (on a landscape it moves the level too).
        static const ParamId kRates[] = { ParamId::MotionRate, ParamId::DriftRate, ParamId::ShimmerRate, ParamId::SyncRate, ParamId::LifeRate,
                                          ParamId::FieldRate, ParamId::BreathRate, ParamId::CombSweep, ParamId::RippleSpeed };
        static const ParamId kDepths[] = { ParamId::Drift, ParamId::Shimmer };
        const float factor = std::exp2(sign * size * 1.2f);
        double got = 0.0;
        for (ParamId id : kRates) {
            const int p = static_cast<int>(id);
            if (!movable(id, values) || values[p] == 0.0f) continue;
            const ParamDesc& d = paramDesc(id);
            out[p] = std::clamp(values[p] * factor, d.min, d.max);
            float s[kNumMeasures];
            if (sensitivity(id, s)) got += s[axis] / measureSpread(Measure::Motion) * (toNorm(d, out[p]) - toNorm(d, values[p]));
        }
        for (ParamId id : kDepths) {
            const int p = static_cast<int>(id);
            if (!movable(id, values)) continue;
            const ParamDesc& d = paramDesc(id);
            out[p] = moved(id, values[p], sign * size * 0.16f);
            float s[kNumMeasures];
            if (sensitivity(id, s)) got += s[axis] / measureSpread(Measure::Motion) * (toNorm(d, out[p]) - toNorm(d, values[p]));
        }
        return static_cast<float>(got);
    }
    // The few knobs that move this measure most specifically: much of it, little of the others and of
    // the level (each in its spread). Measured (--nudgecheck, 8 presets): moving every heard knob a
    // little, with the other measures held by opposing moves, went the wrong way in a quarter of the
    // cases and jumped up to 4 dB; a handful of specific knobs is steadier, and can be told in words.
    struct Pick {
        const Knob* knob;   ///< the knob
        double g;           ///< what it does for the measure
        double l;           ///< and for the level
        double rank;        ///< how specifically
    };
    std::vector<Pick> picks;
    const double levelSpread = measureSpread(Measure::Level);
    for (const Knob& k : table().knobs) {
        if (!movable(k.id, values)) continue;
        const double g = k.d[axis] / measureSpread(static_cast<Measure>(axis));
        if (std::fabs(g) < 0.05) continue;   // does little for it
        double side = std::fabs(k.d[static_cast<int>(Measure::Level)]) / levelSpread;
        for (int m = 0; m < kNudgeAxes; ++m)
            if (m != axis) side += std::fabs(k.d[m]) / measureSpread(static_cast<Measure>(m));
        picks.push_back({ &k, g, k.d[static_cast<int>(Measure::Level)] / levelSpread, std::fabs(g) / (std::fabs(g) + side) * std::sqrt(std::fabs(g)) });
    }
    if (picks.empty()) return 0.0f;
    std::sort(picks.begin(), picks.end(), [](const Pick& a, const Pick& b) { return a.rank > b.rank; });
    if (picks.size() > 4) picks.resize(4);
    const size_t n = picks.size();
    // The shortest step over them; the level held when two or more leave room for it.
    std::vector<double> want(n);
    double gg = 0.0, gl = 0.0, ll = 0.0;
    for (size_t i = 0; i < n; ++i) {
        gg += picks[i].g * picks[i].g;
        gl += picks[i].g * picks[i].l;
        ll += picks[i].l * picks[i].l;
    }
    double ww = 0.0;
    for (size_t i = 0; i < n; ++i) {
        want[i] = picks[i].g - (n >= 2 && ll > 1.0e-12 ? gl / ll * picks[i].l : 0.0);
        ww += want[i] * want[i];
    }
    if (ww < gg / 4.0) {   // holding the level would leave too little: the plain step
        ww = gg;
        for (size_t i = 0; i < n; ++i) want[i] = picks[i].g;
    }
    if (ww < 1.0e-12) return 0.0f;
    double gw = 0.0;
    for (size_t i = 0; i < n; ++i) gw += picks[i].g * want[i];
    std::vector<double> step(n);
    double largest = 0.0;
    for (size_t i = 0; i < n; ++i) {
        step[i] = static_cast<double>(sign * size) * want[i] / std::max(gw, 1.0e-12);
        largest = std::max(largest, std::fabs(step[i]));
    }
    const double scale = largest > 0.2 ? 0.2 / largest : 1.0;   // no knob further than a fifth of its travel at once
    double got = 0.0;
    for (size_t i = 0; i < n; ++i) {
        const ParamId id = picks[i].knob->id;
        const int p = static_cast<int>(id);
        const ParamDesc& d = paramDesc(id);
        const float before = toNorm(d, values[p]);
        out[p] = moved(id, values[p], static_cast<float>(step[i] * scale));
        got += picks[i].g * static_cast<double>(toNorm(d, out[p]) - before);
    }
    return static_cast<float>(got);
}

int focusParams(const float* values, ParamId* out, int n)
{
    std::vector<std::pair<float, ParamId>> score;
    for (const Knob& k : table().knobs) {
        if (!paramHeard(k.id, values)) continue;
        float s = 0.0f;
        for (int m = 0; m < kNudgeAxes; ++m) s += std::fabs(k.d[m]) / measureSpread(static_cast<Measure>(m));
        score.emplace_back(s, k.id);
    }
    std::sort(score.begin(), score.end(), [](const auto& a, const auto& b) { return a.first > b.first; });
    std::map<std::string, int> perSection;
    int m = 0;
    for (const auto& [s, id] : score) {
        if (m >= n) break;
        int& c = perSection[paramDesc(id).section];
        if (c >= 3) continue;
        ++c;
        out[m++] = id;
    }
    return m;
}

void vary(const float* values, float far, std::uint64_t seed, float* out)
{
    std::copy(values, values + kNumParams, out);
    Dice dice{ seed * 0x2545F4914F6CDD1Dull + 1 };
    // The heard shaping knobs, weighted by what they do.
    std::vector<std::pair<float, const Knob*>> pool;
    float most = 0.0f;
    for (const Knob& k : table().knobs) {
        if (!movable(k.id, values)) continue;
        float w = 0.0f;
        for (int m = 0; m < kNudgeAxes; ++m) w += std::fabs(k.d[m]) / measureSpread(static_cast<Measure>(m));
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
        const ParamDesc& d = paramDesc(k.id);
        const int p = static_cast<int>(k.id);
        const float before = toNorm(d, values[p]);
        out[p] = moved(k.id, values[p], dn);
        level += static_cast<double>(k.d[static_cast<int>(Measure::Level)]) * static_cast<double>(toNorm(d, out[p]) - before);
        pool.erase(pool.begin() + static_cast<std::ptrdiff_t>(pick));
        if (pool.empty()) break;
    }
    // The level held by the master, by what the knobs were measured to do to it.
    const ParamDesc& master = paramDesc(ParamId::MasterGain);
    out[static_cast<int>(ParamId::MasterGain)] = std::clamp(values[static_cast<int>(ParamId::MasterGain)] - static_cast<float>(std::clamp(level, -6.0, 6.0)),
                                                             master.min, master.max);
}

void MacroLesson::apply(float at, float* offset, bool* touched) const
{
    if (count < 2) return;
    int k = 0;
    float u = 0.0f;
    if (at <= pos[0]) {
        k = 0;
    } else if (at >= pos[count - 1]) {
        k = count - 2;
        u = 1.0f;
    } else {
        while (k + 2 < count && at > pos[k + 1]) ++k;
        u = (at - pos[k]) / std::max(pos[k + 1] - pos[k], 1.0e-6f);
    }
    for (int p = 0; p < kNumParams; ++p) {
        const float d = delta[k][static_cast<size_t>(p)] + (delta[k + 1][static_cast<size_t>(p)] - delta[k][static_cast<size_t>(p)]) * u;
        if (d == 0.0f) continue;
        offset[p] += d;
        if (touched != nullptr) touched[p] = true;
    }
}

void MacroLesson::set(float at, const float* d)
{
    int slot = -1;
    for (int i = 0; i < count; ++i)
        if (std::fabs(pos[i] - at) < 0.02f) slot = i;
    if (slot < 0 && count < kLessonPlaces) slot = count++;
    if (slot < 0) {   // all places taken: the nearest gives way
        slot = 0;
        for (int i = 1; i < count; ++i)
            if (std::fabs(pos[i] - at) < std::fabs(pos[slot] - at)) slot = i;
    }
    pos[slot] = at;
    std::copy(d, d + kNumParams, delta[slot].begin());
    // Ascending by place.
    for (int i = 1; i < count; ++i)
        for (int j = i; j > 0 && pos[j] < pos[j - 1]; --j) {
            std::swap(pos[j], pos[j - 1]);
            std::swap(delta[j], delta[j - 1]);
        }
}

int MacroLesson::knobs() const
{
    int n = 0;
    for (int p = 0; p < kNumParams; ++p) {
        bool any = false;
        for (int i = 0; i < count && !any; ++i) any = delta[i][static_cast<size_t>(p)] != 0.0f;
        n += any ? 1 : 0;
    }
    return n;
}

bool teachable(ParamId id)
{
    switch (id) {
    case ParamId::Macro1: case ParamId::Macro2: case ParamId::Macro3: case ParamId::Macro4:
    case ParamId::MapX: case ParamId::MapY: case ParamId::MapRadius: case ParamId::MapActive: case ParamId::MapGlide:
    case ParamId::MorphActive: case ParamId::MorphPos: case ParamId::PresetGlide: case ParamId::RouteSpeed:
        return false;
    default:
        return isModTarget(id) && !keptByPresets(id);
    }
}

} // namespace additiv
