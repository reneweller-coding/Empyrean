/**
 * @file TuningView.h
 * @brief The Tuning page's picture: one octave over the root as a ruler, equal temperament's twelve steps on it
 *        as faint ticks with their names, the scale's pitches as lines with how far each stands from the nearest
 *        step (in cents), and the voices where they sound -- a light per voice, its octave up the picture, its
 *        colour its source's, a tail behind it as long as Self-Tune pulls it. So one sees where a just or a timbre
 *        scale departs from the keyboard, how far Purity takes it, and the chord being drawn into pure intervals.
 */
#pragma once

#include "ParamPage.h"
#include "PluginProcessor.h"

/** @brief The tuning's picture (see the file comment). */
class TuningView final : public juce::Component, private juce::Timer {
public:
    /** @brief The tuning of @p proc's engine. @param proc the processor */
    explicit TuningView(AdditivProcessor& proc);
    /** @brief The ruler, the scale, the voices. @param g where */
    void paint(juce::Graphics& g) override;
    /** @brief The picture in words, for the help. @return the text */
    juce::String describe() const;

private:
    /** @brief Fifteen times a second while it is seen. */
    void timerCallback() override;

    AdditivProcessor& proc_;   ///< whose engine
};

/** @brief The Tuning page: its knobs at the top, the tuning's picture below. */
class TuningPage final : public juce::Component {
public:
    /**
     * @brief Builds the page.
     * @param proc     the processor
     * @param actions  the controls' right-click menu
     * @param live     the live rings
     */
    TuningPage(AdditivProcessor& proc, const frame::ControlActions& actions, frame::LiveRings* live);
    /** @brief The height the page needs at a width. @param w pixels @return pixels */
    int heightFor(int w) const;
    /** @brief The knobs at the top, the picture below in the rest. */
    void resized() override;
    /** @brief The page in words. @return the text */
    juce::String describe() const;

private:
    ParamPage params_;    ///< Tuning, Tide, Self-Tune
    TuningView view_;     ///< the picture
};
