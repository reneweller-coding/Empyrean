/**
 * @file Focus.h
 * @brief The simple ways in (the Focus page): what the knobs do to what is heard, measured, and what
 *        follows from it.
 *
 *        The measure: around 48 presets each knob that shapes the sound was moved by 0.15 of its
 *        travel and the sound measured again (additiv_presets --sensitivity) -- its brightness,
 *        motion, width, noisiness and level (Describe.h). Per knob the mean change over the presets
 *        in which it is heard (its group awake) is its sensitivity (Tools/presets/make_nudges.py
 *        writes Nudges.inc).
 *
 *        What follows: a nudge ("brighter", "calmer", "wider", "airier") moves the few heard knobs
 *        (up to four) that move that measure most specifically -- much of it, little of the others and
 *        of the level -- the shortest way there over them, the level held where they leave room, an
 *        amount at nothing left at nothing (what is off stays off: a layer turned on is no small step).
 *        Calmer and livelier go by a rule instead -- every rate slower or faster, drift and shimmer a
 *        little less or more -- the motion measured over eight seconds being too unsteady to steer by;
 *        the focus
 *        is the knobs that do the most for this sound; variations are small random steps over those.
 *        And a macro can be taught by example: at each of its places how far every knob stands from
 *        where it stood (MacroLesson).
 * @author Rene Weller
 */
#pragma once

#include "additiv/Params.h"

#include <array>
#include <cstdint>

namespace additiv {

/** @brief The measures a nudge moves (and the level, which it holds). */
enum class MeasureE : int {
    BRIGHTNESS = 0,   ///< the spectral centroid, octaves
    MOTION,           ///< how much the spectrum changes
    WIDTH,            ///< 1 - the correlation of left and right
    NOISE,            ///< the spectral flatness
    LEVEL,            ///< dBFS of the loudest second
    COUNT             ///< number of measures
};
constexpr int NumMeasures = static_cast<int>(MeasureE::COUNT);   ///< entries of Measure
constexpr int NudgeAxes = 4;                                    ///< the measures a nudge moves (all but the level)
extern const char* const NudgeLow[NudgeAxes];                  ///< "Darker", "Calmer", "Narrower", "Purer"
extern const char* const NudgeHigh[NudgeAxes];                 ///< "Brighter", "Livelier", "Wider", "Airier"

/**
 * @brief The knobs that shape the sound: those measured, those a variation moves.
 * @param[out] count  receives how many
 * @return them
 */
const ParamIdE* shapingParams(int& count);

/**
 * @brief Whether a knob is heard in a sound: its group's gate is open (the air's level for the air's
 *        centre, the filter's model for the cutoff, the hall's level for its decay ...); a knob
 *        without a gate is always heard.
 * @param[in] id      the knob
 * @param[in] values  the sound's knobs, NumParams values
 * @return true when moving it can be heard
 */
bool paramHeard(ParamIdE id, const float* values);

/**
 * @brief A knob's measured sensitivity: per measure the change for its whole travel (Measure order),
 *        the mean over the presets in which it is heard.
 * @param[in]  id   the knob
 * @param[out] out  receives NumMeasures values; all nought for a knob not measured
 * @return false for a knob not measured
 */
bool sensitivity(ParamIdE id, float* out);

/**
 * @brief A measure's spread over the bank (one standard deviation): how much is "much".
 * @param[in] m  the measure
 * @return it
 */
float measureSpread(MeasureE m);

/**
 * @brief A nudge: the few heard knobs that move the measure most specifically, a little (see the file comment).
 * @param[in]  values  the sound's knobs, NumParams values
 * @param[in]  axis    0 .. NudgeAxes-1 (Measure order)
 * @param[in]  sign    +1 towards NudgeHigh, -1 towards NudgeLow
 * @param[in]  size    how far, in spreads of the measure (0.25: a small step)
 * @param[out] out     receives the knobs after it (NumParams values; untouched knobs as they were)
 * @return the predicted change of the measure, in spreads (less than asked when knobs hit their ends)
 */
float nudge(const float* values, int axis, float sign, float size, float* out);

/**
 * @brief The focus: the heard knobs that do the most for this sound, most first -- by their summed
 *        sensitivity over the four measures, each in its spread; at most three from one group.
 * @param[in]  values  the sound's knobs
 * @param[out] out     receives ParamIds
 * @param[in]  n       how many at most
 * @return how many it wrote
 */
int focusParams(const float* values, ParamIdE* out, int n);

/**
 * @brief A variation: a few heard shaping knobs (four to eight; none that is an amount at nothing) moved at random, the larger moves on
 *        the knobs that do more, the level held by the master by the measured sensitivity.
 * @param[in]  values  the sound's knobs
 * @param[in]  far     how far, 0 .. 1 (0.3: near)
 * @param[in]  seed    the dice
 * @param[out] out     receives the knobs after it
 */
void vary(const float* values, float far, std::uint64_t seed, float* out);

constexpr int LessonPlaces = 4;   ///< places a taught macro may hold

/**
 * @brief A macro taught by example (the Focus page): at each of its places how far every knob stands
 *        from where it stood when the teaching began, in the knob's normalised travel. Between places
 *        the macro glides; before the first and after the last it holds. The engine adds it to the knobs
 *        like a matrix row (Engine::setMacroLesson) -- relative, so a lesson carries over to the next
 *        preset as a gesture: "this much brighter and slower".
 */
struct MacroLesson {
    int m_count = 0;                                            ///< places, 0 .. LessonPlaces (fewer than two: it does nothing)
    float m_pos[LessonPlaces] = {};                            ///< the macro's value at each place, ascending
    std::array<float, NumParams> m_delta[LessonPlaces] = {};  ///< per place, per knob the offset in its normalised travel
    /**
     * @brief The offsets at a macro value, added.
     * @param[in]     at       the macro's value
     * @param[in,out] offset   NumParams offsets, added to
     * @param[in,out] touched  NumParams flags, set where it adds (may be null)
     */
    void apply(float at, float* offset, bool* touched) const;
    /**
     * @brief A place set (one within 0.02 of it replaced; with all places taken, the nearest).
     * @param[in] at     the macro's value
     * @param[in] delta  NumParams offsets
     */
    void set(float at, const float* delta);
    /** @brief The knobs it moves anywhere. @return how many */
    int knobs() const;
};

/**
 * @brief Whether a knob can be taught to a macro: a matrix target, not the listener's, not the macros', the map's or
 *        the morph's own.
 * @param[in] id  the knob
 * @return true then
 */
bool teachable(ParamIdE id);

} // namespace additiv
