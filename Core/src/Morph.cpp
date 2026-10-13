/**
 * @file Morph.cpp
 * @brief One sound becoming another (Morph.h).
 */
#include "additiv/Morph.h"

#include "additiv/Cloud.h"
#include "additiv/Motion.h"

#include <algorithm>
#include <cmath>

namespace additiv {

void MorphState::set(const float* v, const ModRow* r, int count)
{
    std::copy(v, v + kNumParams, values);
    filterMix = 1.0f;
    for (float& g : gate) g = 1.0f;
    setupCount = 1;
    setups[0] = modSetupFrom(v);
    rowCount = 0;
    for (int i = 0; i < count && rowCount < kMorphRows; ++i)
        if (r[i].source != ModSource::Off && r[i].target >= 0 && r[i].depth != 0.0f) {
            rows[rowCount] = r[i];
            rows[rowCount].setup = 0;
            ++rowCount;
        }
}

namespace {

/** @brief A parameter's value at a point between two, in its normalised (skewed) domain. @param d the parameter @param a at 0 @param b at 1 @param w 0 .. 1 @return the value */
float lerpParam(const ParamDesc& d, float a, float b, float w)
{
    if (a == b || w <= 0.0f) return a;   // the ends exactly: the skew's round trip is not
    if (w >= 1.0f) return b;
    const float n = toNorm(d, a) + (toNorm(d, b) - toNorm(d, a)) * w;
    const float v = fromNorm(d, n);
    return d.kind == ParamKind::Int ? std::round(v) : v;
}

/** @brief The side a choice is taken from while it changes, its amount (normalised) there and its gate. */
struct Side {
    const MorphState* from;   ///< whose choice and companions; null: the choice is the same on both
    float amount;             ///< the amount, 0 .. 1 of its range
    float gate;               ///< the share the amount, modulation included, is heard at
};

/**
 * @brief Where a choice with an amount stands between two sounds (the file comment of Morph.h).
 * @param ca    the choice at 0
 * @param cb    the choice at 1
 * @param na    the amount at 0, normalised (taken as 0 where the choice is off)
 * @param nb    the amount at 1, normalised
 * @param ga    the gate at 0
 * @param gb    the gate at 1
 * @param off   the choice that means off, -1 when none does
 * @param same    the two count as the same (then the amount simply glides)
 * @param middle  change at the middle even from or to off (the landscape it reads changes there)
 * @param w     0 .. 1
 * @param a     the sound at 0
 * @param b     the sound at 1
 * @return      the side, the amount and the gate
 */
Side side(int ca, int cb, float na, float nb, float ga, float gb, int off, bool same, bool middle, float w, const MorphState& a, const MorphState& b)
{
    if (ca == off) { na = 0.0f; ga = 0.0f; }
    if (cb == off) { nb = 0.0f; gb = 0.0f; }
    if (same) return { nullptr, na + (nb - na) * w, ga + (gb - ga) * w };
    if (!middle && ca == off) return { &b, w * nb, w * gb };
    if (!middle && cb == off) return { &a, (1.0f - w) * na, (1.0f - w) * ga };
    return w < 0.5f ? Side{ &a, na * (1.0f - 2.0f * w), ga * (1.0f - 2.0f * w) } : Side{ &b, nb * (2.0f * w - 1.0f), gb * (2.0f * w - 1.0f) };
}

/** @brief A parameter's value as an index. @param s the sound @param id the parameter @return it rounded */
int choiceOf(const MorphState& s, ParamId id) { return static_cast<int>(std::lround(s.values[static_cast<size_t>(id)])); }

/** @brief A parameter normalised. @param s the sound @param id the parameter @return 0 .. 1 */
float normOf(const MorphState& s, ParamId id) { return toNorm(paramDesc(id), s.values[static_cast<size_t>(id)]); }

/** @brief One of the choices with an amount: what it is made of. */
struct Pair {
    ParamId choice;             ///< the choice
    ParamId amount;             ///< its amount (the filter's is filterMix: ParamId::Count)
    int gate;                   ///< its gate in MorphState::gate, -1 for the filter (filterMix is its own gate)
    ParamId companions[2];      ///< what goes with the choice's side
    int companionCount;         ///< how many
};

/** @brief The choices with an amount, in the order blendMorph and approachMorph take them. */
const Pair kPairs[] = {
    { ParamId::Motion, ParamId::MotionDepth, 0, { ParamId::Motion, ParamId::Motion }, 0 },
    { ParamId::LifeMode, ParamId::LifeDepth, 1, { ParamId::LifeMode, ParamId::LifeMode }, 0 },
    { ParamId::FieldMode, ParamId::FieldStrength, 2, { ParamId::FieldMode, ParamId::FieldMode }, 0 },
    { ParamId::FilterModel, ParamId::Count, -1, { ParamId::Cutoff, ParamId::Resonance }, 2 },
};

/** @brief Whether a pair's kind reads the landscape (Aurora, the animated pictures). @param p the pair @param kind its choice @return true then */
bool readsLandscape(const Pair& p, int kind)
{
    if (p.choice == ParamId::Motion) return motionUsesLandscape(static_cast<MotionKind>(kind));
    if (p.choice == ParamId::LifeMode) return lifeIsPicture(static_cast<LifeMode>(kind));
    return false;
}

/** @brief Whether the landscape changes between two sounds for a pair that reads it on either side. @param p the pair @param a one @param b the other @return true then */
bool landChanges(const Pair& p, const MorphState& a, const MorphState& b)
{
    return choiceOf(a, ParamId::Landscape) != choiceOf(b, ParamId::Landscape)
        && (readsLandscape(p, choiceOf(a, p.choice)) || readsLandscape(p, choiceOf(b, p.choice)));
}

/** @brief Whether two sounds have the same kind for a pair (and the same landscape, if it reads one). @param p the pair @param a one @param b the other @return true then */
bool sameKind(const Pair& p, const MorphState& a, const MorphState& b)
{
    return choiceOf(a, p.choice) == choiceOf(b, p.choice) && !landChanges(p, a, b);
}

/** @brief A pair's amount, normalised. @param p the pair @param s the sound @return 0 .. 1 */
float amountOf(const Pair& p, const MorphState& s) { return p.gate < 0 ? s.filterMix : normOf(s, p.amount); }

/** @brief A pair's gate. @param p the pair @param s the sound @return 0 .. 1 */
float gateOf(const Pair& p, const MorphState& s) { return p.gate < 0 ? s.filterMix : s.gate[p.gate]; }

/** @brief Sets a pair's amount (normalised) and gate. @param p the pair @param s the sound @param amount 0 .. 1 @param gate 0 .. 1 */
void setAmount(const Pair& p, MorphState& s, float amount, float gate)
{
    if (p.gate < 0) { s.filterMix = amount; return; }
    s.values[static_cast<size_t>(p.amount)] = fromNorm(paramDesc(p.amount), amount);
    s.gate[p.gate] = gate;
}

/** @brief Takes a pair's choice and companions from a sound. @param p the pair @param from where from @param out where to */
void takeKind(const Pair& p, const MorphState& from, MorphState& out)
{
    out.values[static_cast<size_t>(p.choice)] = from.values[static_cast<size_t>(p.choice)];
    for (int k = 0; k < p.companionCount; ++k)
        out.values[static_cast<size_t>(p.companions[k])] = from.values[static_cast<size_t>(p.companions[k])];
}

/** @brief Whether a row reads a sound's own modulators (its source or its via). @param r the row @return true then */
bool readsOwn(const ModRow& r) { return isOwnSource(r.source) || isOwnSource(r.via); }

/**
 * @brief A setup's place in a sound: the one that is equal, or a new one (the first when full).
 * @param out  the sound
 * @param s    the setup
 * @return     its index
 */
int addSetup(MorphState& out, const ModSetup& s)
{
    for (int i = 0; i < out.setupCount; ++i)
        if (out.setups[i] == s) return i;
    if (out.setupCount < kMorphSetups) {
        out.setups[out.setupCount] = s;
        return out.setupCount++;
    }
    return 0;
}

/** @brief Whether two rows move the same thing the same way (their depths add): on the same modulators, if they read a sound's own. @param x one @param y the other @return true then */
bool sameRow(const ModRow& x, const ModRow& y)
{
    return x.source == y.source && x.target == y.target && x.via == y.via && (!readsOwn(x) || x.setup == y.setup);
}

/**
 * @brief Adds a row of a sound with its depth weighted, its setup carried over: merged with one that
 *        moves the same, or appended, or (when full) put in place of the weakest if it is stronger.
 * @param out   the sound being made
 * @param from  the sound the row is from
 * @param r     the row
 * @param k     its weight
 */
void addRow(MorphState& out, const MorphState& from, const ModRow& r, float k)
{
    const float depth = r.depth * k;
    if (depth == 0.0f) return;
    ModRow s = r;
    s.depth = depth;
    s.setup = readsOwn(r) && r.setup >= 0 && r.setup < from.setupCount ? addSetup(out, from.setups[r.setup]) : 0;
    for (int i = 0; i < out.rowCount; ++i)
        if (sameRow(out.rows[i], s)) { out.rows[i].depth += depth; return; }
    if (out.rowCount < kMorphRows) { out.rows[out.rowCount++] = s; return; }
    int weakest = 0;
    for (int i = 1; i < out.rowCount; ++i)
        if (std::fabs(out.rows[i].depth) < std::fabs(out.rows[weakest].depth)) weakest = i;
    if (std::fabs(depth) > std::fabs(out.rows[weakest].depth)) out.rows[weakest] = s;
}

/** @brief Drops the setups no row reads any more, the rows' indices following. @param s the sound */
void compactSetups(MorphState& s)
{
    int map[kMorphSetups];
    bool used[kMorphSetups] = {};
    for (int i = 0; i < s.rowCount; ++i)
        if (readsOwn(s.rows[i]) && s.rows[i].setup < s.setupCount) used[s.rows[i].setup] = true;
    int n = 0;
    for (int k = 0; k < s.setupCount; ++k) {
        map[k] = n;
        if (used[k]) s.setups[n++] = s.setups[k];
    }
    for (int i = 0; i < s.rowCount; ++i)
        s.rows[i].setup = readsOwn(s.rows[i]) && s.rows[i].setup < s.setupCount ? map[s.rows[i].setup] : 0;
    s.setupCount = n;
}

/** @brief A number a share of the way, unrounded, arriving exactly when close. @param d the parameter @param a from @param b to @param k the share @return the value */
float follow(const ParamDesc& d, float a, float b, float k)
{
    if (a == b) return a;
    const float na = toNorm(d, a), nb = toNorm(d, b);
    if (std::fabs(nb - na) < 1.0e-5f) return b;
    return fromNorm(d, na + (nb - na) * k);
}

} // namespace

void approachMorph(const MorphState& cur, const MorphState& target, float coef, MorphState& out)
{
    coef = std::clamp(coef, 0.0f, 1.0f);
    for (const ParamDesc& d : paramTable()) {
        const size_t i = static_cast<size_t>(d.id);
        out.values[i] = d.kind == ParamKind::Float || d.kind == ParamKind::Int ? follow(d, cur.values[i], target.values[i], coef) : target.values[i];
    }
    out.filterMix = cur.filterMix + (target.filterMix - cur.filterMix) * coef;
    for (int g = 0; g < kMorphGates; ++g) out.gate[g] = cur.gate[g] + (target.gate[g] - cur.gate[g]) * coef;
    // A kind that differs: its amount and gate fade out (twice the pace), its companions stay; it
    // changes when nothing of it is heard. From off it changes at once, with nothing of it yet.
    // The landscape changes when nothing that reads it is heard: until then those that read it
    // stay at nothing.
    const float fade = std::max(0.0f, 1.0f - 2.0f * coef);
    out.values[static_cast<size_t>(ParamId::Landscape)] = cur.values[static_cast<size_t>(ParamId::Landscape)];
    for (const Pair& p : kPairs) {
        if (sameKind(p, cur, target)) continue;
        if (choiceOf(cur, p.choice) == 0) {
            takeKind(p, target, out);
            setAmount(p, out, 0.0f, 0.0f);
            continue;
        }
        const float gate = gateOf(p, cur) * fade;
        if (gate < 0.005f) {
            takeKind(p, target, out);
            setAmount(p, out, 0.0f, choiceOf(target, p.choice) == 0 ? gateOf(p, target) : 0.0f);
        } else {
            takeKind(p, cur, out);
            setAmount(p, out, amountOf(p, cur) * fade, gate);
        }
    }
    bool heard = false;
    for (const Pair& p : kPairs)
        heard = heard || (readsLandscape(p, choiceOf(out, p.choice)) && gateOf(p, out) >= 0.005f);
    if (!heard) out.values[static_cast<size_t>(ParamId::Landscape)] = target.values[static_cast<size_t>(ParamId::Landscape)];
    out.rowCount = 0;
    out.setupCount = 0;
    for (int i = 0; i < cur.rowCount; ++i) addRow(out, cur, cur.rows[i], 1.0f - coef);
    for (int i = 0; i < target.rowCount; ++i) addRow(out, target, target.rows[i], coef);
    // Rows the target no longer has fade geometrically: drop them once they are nothing.
    int keep = 0;
    for (int i = 0; i < out.rowCount; ++i)
        if (std::fabs(out.rows[i].depth) > 1.0e-4f) out.rows[keep++] = out.rows[i];
    out.rowCount = keep;
    compactSetups(out);
}

void blendMorph(const MorphState& a, const MorphState& b, float w, MorphState& out)
{
    w = std::clamp(w, 0.0f, 1.0f);
    // Every parameter: numbers glide, choices and switches change at the middle.
    for (const ParamDesc& d : paramTable()) {
        const size_t i = static_cast<size_t>(d.id);
        out.values[i] = d.kind == ParamKind::Float || d.kind == ParamKind::Int ? lerpParam(d, a.values[i], b.values[i], w)
                                                                                : (w < 0.5f ? a.values[i] : b.values[i]);
    }
    // The choices with an amount (the motion, the living landscape, the field, the voice filter with
    // its cutoff and resonance -- an Off filter's are not heard and should not sweep): the amount
    // and the gate by the rule, the kind from the side in use. Where the landscape changes (at the
    // middle, a plain choice), those that read it pass through nothing there, even from or to off.
    for (const Pair& p : kPairs) {
        const Side s = side(choiceOf(a, p.choice), choiceOf(b, p.choice), amountOf(p, a), amountOf(p, b), gateOf(p, a), gateOf(p, b), 0,
                            sameKind(p, a, b), landChanges(p, a, b), w, a, b);
        setAmount(p, out, s.amount, s.gate);
        if (s.from != nullptr) takeKind(p, *s.from, out);
    }
    // The matrix: both, weighted, each row with the modulators it came with.
    out.rowCount = 0;
    out.setupCount = 0;
    for (int i = 0; i < a.rowCount; ++i) addRow(out, a, a.rows[i], 1.0f - w);
    for (int i = 0; i < b.rowCount; ++i) addRow(out, b, b.rows[i], w);
}

} // namespace additiv
