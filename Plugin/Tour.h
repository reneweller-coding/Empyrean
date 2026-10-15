/**
 * @file Tour.h
 * @brief The first steps: a short tour on the first start (and whenever asked for), one control at a
 *        time lit in a dimmed window with a sentence on it -- Play, the presets, the picture, Focus, the
 *        search, the info line, Save. Next goes on, Skip ends it.
 */
#pragma once

#include <juce_gui_basics/juce_gui_basics.h>

#include <functional>
#include <vector>

/** @brief The tour (see the file comment). */
class Tour final : public juce::Component {
public:
    /** @brief One step: what is lit, and what is said about it. */
    struct Step {
        juce::Component::SafePointer<juce::Component> m_target;   ///< the control lit
        juce::String m_title;                                       ///< its name
        juce::String m_text;                                        ///< what it does
    };
    /** @brief Called when the tour ends (done or skipped). */
    std::function<void()> m_onDone;
    /** @brief A tour, not yet shown. */
    Tour();
    /** @brief Begins it. @param steps its steps */
    void start(std::vector<Step> steps);
    /** @brief The dimmed window with the lit control and the bubble. */
    void paint(juce::Graphics& g) override;
    /** @brief The bubble's buttons. */
    void resized() override;

private:
    /** @brief The lit control's rectangle in this component. @return pixels (empty when it is gone) */
    juce::Rectangle<int> hole() const;
    /** @brief The bubble's rectangle. @return pixels */
    juce::Rectangle<int> bubble() const;
    /** @brief The next step, or the end. */
    void next();
    /** @brief Ends it. */
    void finish();

    std::vector<Step> m_steps;                 ///< the steps
    size_t m_at = 0;                           ///< the one shown
    juce::TextButton m_next{ "Next" };         ///< on
    juce::TextButton m_skip{ "Skip" };         ///< ends it
};
