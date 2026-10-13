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
 * are faded out. The fade is also kept as a gate, which the engine applies to the amount after
 * the matrix: a row that pushes the depth would otherwise keep it from nothing where the kind
 * changes. The landscape is read by Aurora and by the animated pictures: where it changes, every
 * one of them that reads it fades through nothing at the middle, and it changes there. Every
 * other choice and switch changes at the middle.
 *
 * Each sound brings its own LFOs and envelopes (ModSetup): a blend keeps the setups of the sounds
 * in it, and every row the setup it came with, so the engine runs one set of modulators per setup
 * and a row fades out on the modulators it was made for.
 */
#pragma once

#include "additiv/Modulation.h"
#include "additiv/Params.h"

namespace additiv {

constexpr int kMorphRows = 64;     ///< weighted matrix rows a sound can hold (a blend of blends merges and, past this, drops the weakest)
constexpr int kMorphSetups = 16;   ///< sounds' own modulators a blend can hold (a seventh sound's rows share the first's)
constexpr int kMorphGates = 3;     ///< the choices with an amount that have a gate: the motion, the living landscape, the field

/** @brief The amounts the gates multiply, in gate order (the filter's is filterMix). */
constexpr ParamId kGatedAmounts[kMorphGates] = { ParamId::MotionDepth, ParamId::LifeDepth, ParamId::FieldStrength };

/** @brief A sound as the engine plays it before the matrix: every parameter, the filter's share and the weighted rows. */
struct MorphState {
    float values[kNumParams] = {};   ///< every parameter, in its own units
    float filterMix = 1.0f;          ///< 0 .. 1: the voice filter's gain is raised to this (0: no filter)
    float gate[kMorphGates] = { 1.0f, 1.0f, 1.0f };   ///< 0 .. 1: each gated amount, the matrix's push included, is multiplied by this (0 where its kind changes)
    int setupCount = 0;              ///< setups in use
    ModSetup setups[kMorphSetups];   ///< the LFOs and envelopes of the sounds in it
    int rowCount = 0;                ///< rows in use
    ModRow rows[kMorphRows];         ///< the matrix, each depth already weighted, each with its setup

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
 * @brief One step of following a sound that moves (the map's cursor): every number a share `coef`
 *        of the way in its own domain (whole numbers too, unrounded: the engine rounds them where it
 *        uses them), the filter's share, the gates and the matrix likewise. A choice with an amount
 *        that differs first fades its amount and gate out at twice that pace, its companions held,
 *        and changes when nothing of it is heard; every other choice and switch is the target's at once.
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
