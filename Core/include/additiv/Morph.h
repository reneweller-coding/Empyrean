/**
 * @file Morph.h
 * @brief One sound becoming another: every parameter, the voice filter's share and the matrix of
 *        two sounds blended at a point between them.
 *
 * Noctuary crossfades two whole engines when a preset changes, because its sounds -- samples,
 * wavetables, rooms -- cannot be interpolated. Here every voice is a spectrum written afresh every
 * block, so the sound itself can travel: the numbers glide in each parameter's own (skewed)
 * domain, the tuning glides the partials into the new scale, and the voice filter, read as a
 * gain at every partial, is raised to a power between 0 (no filter) and 1, so one filter fades
 * out and another in without a step. The matrix is blended as two weighted sums of rows.
 *
 * What cannot glide switches where it is not heard: a choice that has an amount (the motion and
 * its depth, the living landscape, the field, the filter) fades its amount to nothing on the
 * side it leaves and up on the side it reaches -- from Off it switches at once with its amount
 * at nothing, to Off it switches at the end, and between two others at the middle, where both
 * are faded out. Every other choice and switch changes at the middle.
 */
#pragma once

#include "additiv/Modulation.h"
#include "additiv/Params.h"

namespace additiv {

constexpr int kMorphRows = 64;   ///< weighted matrix rows a sound can hold (a blend of blends merges and, past this, drops the weakest)

/** @brief A sound as the engine plays it before the matrix: every parameter, the filter's share and the weighted rows. */
struct MorphState {
    float values[kNumParams] = {};   ///< every parameter, in its own units
    float filterMix = 1.0f;          ///< 0 .. 1: the voice filter's gain is raised to this (0: no filter)
    int rowCount = 0;                ///< rows in use
    ModRow rows[kMorphRows];         ///< the matrix, each depth already weighted

    /**
     * @brief Takes parameters and plain rows as they stand: the filter at full share.
     * @param v      kNumParams values
     * @param r      the rows (rows with no source, no target or no depth are left out)
     * @param count  how many
     */
    void set(const float* v, const ModRow* r, int count);
};

/**
 * @brief The sound at a point between two: the rules of the file comment.
 * @param a    the sound at 0
 * @param b    the sound at 1
 * @param w    0 .. 1 (clamped)
 * @param out  receives the blend; must not be a or b
 */
void blendMorph(const MorphState& a, const MorphState& b, float w, MorphState& out);

/**
 * @brief One step of following a sound that moves (the map's cursor): every real number a share `coef`
 *        of the way in its own domain, the filter's share and the matrix likewise, every choice and
 *        switch and whole number the target's at once (the map's blend has already faded an amount through nothing
 *        where its neighbours' choices meet).
 * @param cur     where the sound is
 * @param target  where it goes
 * @param coef    0 .. 1: the share of the way this step
 * @param out     receives the step; must not be cur or target
 */
void approachMorph(const MorphState& cur, const MorphState& target, float coef, MorphState& out);

/**
 * @brief The glide's shape over time: smoothstep, so a glide leaves and arrives gently.
 * @param t  0 .. 1 of the glide's time
 * @return   0 .. 1 of the way
 */
inline float morphCurve(float t)
{
    const float x = t < 0.0f ? 0.0f : (t > 1.0f ? 1.0f : t);
    return x * x * (3.0f - 2.0f * x);
}

} // namespace additiv
