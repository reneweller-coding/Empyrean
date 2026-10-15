/**
 * @file TuningView.h
 * @brief The Tuning page's picture: one octave over the root as a ruler, equal temperament's twelve steps on it
 *        as faint ticks with their names, the scale's pitches as lines with how far each stands from the nearest
 *        step (in cents), and the voices where they sound -- a light per voice, its octave up the picture, its
 *        colour its source's, a tail behind it as long as Self-Tune pulls it. So one sees where a just or a timbre
 *        scale departs from the keyboard, how far Purity takes it, and the chord being drawn into pure intervals.
 * @author Rene Weller
 */
#pragma once

#include "ParamPage.h"
#include "PluginProcessor.h"

#include <vector>

/** @brief The tuning's picture (see the file comment). */
class TuningView final : public juce::Component, private juce::Timer
{
public:
    /**
     * @brief The tuning of @p proc's engine.
     * @param[in,out] proc  the processor
     */
    explicit TuningView( AdditivProcessor &proc );
    /**
     * @brief The ruler, the scale, the voices.
     * @param[in,out] g  where
     */
    void paint( juce::Graphics &g ) override;
    /** @brief The picture in words, for the help. @return the text */
    juce::String describe() const;
    /**
     * @brief A click on a line of the scale plays it (in the octave clicked); with Shift it is held, with the others
     *        held.
     * @param[in] e  the press
     */
    void mouseDown( const juce::MouseEvent &e ) override;
    /**
     * @brief The note let go -- unless held.
     * @param[in] e  the release
     */
    void mouseUp( const juce::MouseEvent &e ) override;
    /** @brief Every held note let go when the page is left. */
    void visibilityChanged() override;

private:
    /** @brief The ruler's area. @return pixels */
    juce::Rectangle<float> area() const;
    /** @brief The root's equal-tempered pitch, never nothing. @return hertz */
    float rootHz0() const;
    /** @brief Lets every note go. */
    void releaseAll();
    int              m_playing = -1;   ///< the note the mouse plays, -1 none
    std::vector<int> m_held;           ///< the notes held with Shift
    /** @brief Fifteen times a second while it is seen. */
    void timerCallback() override;

    AdditivProcessor &m_proc;   ///< whose engine
};

/** @brief The Tuning page: its knobs at the top, the tuning's picture below. */
class TuningPage final : public juce::Component
{
public:
    /**
     * @brief Builds the page.
     * @param[in,out] proc     the processor
     * @param[in]     actions  the controls' right-click menu
     * @param[in,out] live     the live rings
     */
    TuningPage( AdditivProcessor &proc, const frame::ControlActions &actions, frame::LiveRings *live );
    /**
     * @brief The height the page needs at a width.
     * @param[in] w  pixels
     * @return pixels
     */
    int heightFor( int w ) const;
    /** @brief The knobs at the top, the picture below in the rest. */
    void resized() override;
    /** @brief The page in words. @return the text */
    juce::String describe() const;

private:
    ParamPage  m_params;   ///< Tuning, Tide, Self-Tune
    TuningView m_view;     ///< the picture
};
