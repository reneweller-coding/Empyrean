/**
 * @file Morph.cpp
 * @brief One sound becoming another (Morph.h).
 */
#include "additiv/Morph.h"

#include "additiv/Motion.h"

#include <algorithm>
#include <cmath>

namespace additiv {

void MorphState::set(const float* v, const ModRow* r, int count)
{
    std::copy(v, v + kNumParams, values);
    filterMix = 1.0f;
    rowCount = 0;
    for (int i = 0; i < count && rowCount < kMorphRows; ++i)
        if (r[i].source != ModSource::Off && r[i].target >= 0 && r[i].depth != 0.0f) rows[rowCount++] = r[i];
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

/** @brief The side a choice is taken from while it changes, and its amount (normalised) there. */
struct Side {
    const MorphState* from;   ///< whose choice and companions; null: the choice is the same on both
    float amount;             ///< the amount, 0 .. 1 of its range
};

/**
 * @brief Where a choice with an amount stands between two sounds (the file comment of Morph.h).
 * @param ca    the choice at 0
 * @param cb    the choice at 1
 * @param na    the amount at 0, normalised (taken as 0 where the choice is off)
 * @param nb    the amount at 1, normalised
 * @param off   the choice that means off, -1 when none does
 * @param same  the two count as the same (then the amount simply glides)
 * @param w     0 .. 1
 * @param a     the sound at 0
 * @param b     the sound at 1
 * @return      the side and the amount
 */
Side side(int ca, int cb, float na, float nb, int off, bool same, float w, const MorphState& a, const MorphState& b)
{
    if (ca == off) na = 0.0f;
    if (cb == off) nb = 0.0f;
    if (same) return { nullptr, na + (nb - na) * w };
    if (ca == off) return { &b, w * nb };
    if (cb == off) return { &a, (1.0f - w) * na };
    return w < 0.5f ? Side{ &a, na * (1.0f - 2.0f * w) } : Side{ &b, nb * (2.0f * w - 1.0f) };
}

/** @brief A parameter's value as an index. @param s the sound @param id the parameter @return it rounded */
int choiceOf(const MorphState& s, ParamId id) { return static_cast<int>(std::lround(s.values[static_cast<size_t>(id)])); }

/** @brief A parameter normalised. @param s the sound @param id the parameter @return 0 .. 1 */
float normOf(const MorphState& s, ParamId id) { return toNorm(paramDesc(id), s.values[static_cast<size_t>(id)]); }

/**
 * @brief Applies a choice with an amount: the choice (and its companions) from the side in use,
 *        the amount from the rule.
 * @param a           the sound at 0
 * @param b           the sound at 1
 * @param w           0 .. 1
 * @param out         the blend, its plain glide already done
 * @param choice      the choice
 * @param amount      its amount
 * @param off         the choice that means off, -1 none
 * @param same        whether the two count as the same
 * @param companions  parameters that go with the choice's side
 * @param count       how many
 */
void choiceWithAmount(const MorphState& a, const MorphState& b, float w, MorphState& out, ParamId choice, ParamId amount, int off, bool same,
                      const ParamId* companions, int count)
{
    const Side s = side(choiceOf(a, choice), choiceOf(b, choice), normOf(a, amount), normOf(b, amount), off, same, w, a, b);
    out.values[static_cast<size_t>(amount)] = fromNorm(paramDesc(amount), s.amount);
    if (s.from == nullptr) return;
    out.values[static_cast<size_t>(choice)] = s.from->values[static_cast<size_t>(choice)];
    for (int k = 0; k < count; ++k) out.values[static_cast<size_t>(companions[k])] = s.from->values[static_cast<size_t>(companions[k])];
}

/** @brief Whether two rows move the same thing the same way (their depths add). @param x one @param y the other @return true then */
bool sameRow(const ModRow& x, const ModRow& y) { return x.source == y.source && x.target == y.target && x.via == y.via; }

/**
 * @brief Adds a row with its depth weighted: merged with one that moves the same, or appended, or
 *        (when full) put in place of the weakest if it is stronger.
 * @param out  the sound
 * @param r    the row
 * @param k    its weight
 */
void addRow(MorphState& out, const ModRow& r, float k)
{
    const float depth = r.depth * k;
    if (depth == 0.0f) return;
    for (int i = 0; i < out.rowCount; ++i)
        if (sameRow(out.rows[i], r)) { out.rows[i].depth += depth; return; }
    ModRow s = r;
    s.depth = depth;
    if (out.rowCount < kMorphRows) { out.rows[out.rowCount++] = s; return; }
    int weakest = 0;
    for (int i = 1; i < out.rowCount; ++i)
        if (std::fabs(out.rows[i].depth) < std::fabs(out.rows[weakest].depth)) weakest = i;
    if (std::fabs(depth) > std::fabs(out.rows[weakest].depth)) out.rows[weakest] = s;
}

} // namespace

void blendMorph(const MorphState& a, const MorphState& b, float w, MorphState& out)
{
    w = std::clamp(w, 0.0f, 1.0f);
    // Every parameter: numbers glide, choices and switches change at the middle.
    for (const ParamDesc& d : paramTable()) {
        const size_t i = static_cast<size_t>(d.id);
        out.values[i] = d.kind == ParamKind::Float || d.kind == ParamKind::Int ? lerpParam(d, a.values[i], b.values[i], w)
                                                                                : (w < 0.5f ? a.values[i] : b.values[i]);
    }
    // The motion and the landscape Aurora reads, through the motion's depth.
    {
        const int ma = choiceOf(a, ParamId::Motion), mb = choiceOf(b, ParamId::Motion);
        const bool reads = motionUsesLandscape(static_cast<MotionKind>(ma));
        const bool same = ma == mb && (!reads || choiceOf(a, ParamId::Landscape) == choiceOf(b, ParamId::Landscape));
        const ParamId companions[] = { ParamId::Landscape };
        choiceWithAmount(a, b, w, out, ParamId::Motion, ParamId::MotionDepth, 0, same, companions, 1);
    }
    // The living landscape through its depth, the field through its pull.
    choiceWithAmount(a, b, w, out, ParamId::LifeMode, ParamId::LifeDepth, 0, choiceOf(a, ParamId::LifeMode) == choiceOf(b, ParamId::LifeMode), nullptr, 0);
    choiceWithAmount(a, b, w, out, ParamId::FieldMode, ParamId::FieldStrength, 0, choiceOf(a, ParamId::FieldMode) == choiceOf(b, ParamId::FieldMode), nullptr, 0);
    // The voice filter through its share: a filter that is off has none, and the side in use
    // brings its own cutoff and resonance (an Off filter's are not heard and should not sweep).
    {
        const int fa = choiceOf(a, ParamId::FilterModel), fb = choiceOf(b, ParamId::FilterModel);
        const Side s = side(fa, fb, a.filterMix, b.filterMix, 0, fa == fb, w, a, b);
        out.filterMix = s.amount;
        if (s.from != nullptr)
            for (ParamId id : { ParamId::FilterModel, ParamId::Cutoff, ParamId::Resonance })
                out.values[static_cast<size_t>(id)] = s.from->values[static_cast<size_t>(id)];
    }
    // The matrix: both, weighted.
    out.rowCount = 0;
    for (int i = 0; i < a.rowCount; ++i) addRow(out, a.rows[i], 1.0f - w);
    for (int i = 0; i < b.rowCount; ++i) addRow(out, b.rows[i], w);
}

} // namespace additiv
