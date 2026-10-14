/**
 * @file InfoLine.h
 * @brief The info line at the bottom of the window (as Ableton's Info View): what the control under the
 *        mouse does, in a sentence (ParamInfo.h), its value now and what in the matrix moves it -- or,
 *        for any other control, its tooltip -- without waiting for a tooltip or opening the manual.
 */
#pragma once

#include "PluginProcessor.h"

/** @brief The info line (see the file comment). */
class InfoLine final : public juce::Component, private juce::Timer {
public:
    /**
     * @brief A line that follows the mouse over @p watched and everything in it.
     * @param proc     the processor (values, the matrix)
     * @param watched  the window it follows the mouse over
     */
    InfoLine(AdditivProcessor& proc, juce::Component& watched);
    /** @brief Stops following the mouse. */
    ~InfoLine() override;
    /** @brief The name in the accent's colour, the rest dim. */
    void paint(juce::Graphics& g) override;
    /** @brief Tells about a parameter without the mouse on it (the palette found it). @param id the parameter */
    void showParam(int id);

private:
    /** @brief Follows the mouse over every component of the window (a listener of its own: the line is no parent of them). */
    struct Tracker final : public juce::MouseListener {
        InfoLine* line = nullptr;   ///< whose
        /** @brief Over a component. @param e the move */
        void mouseMove(const juce::MouseEvent& e) override { line->over(e.eventComponent); }
        /** @brief Into a component. @param e the entry */
        void mouseEnter(const juce::MouseEvent& e) override { line->over(e.eventComponent); }
        /** @brief Turning a knob: its value follows. @param e the drag */
        void mouseDrag(const juce::MouseEvent& e) override { line->over(e.eventComponent); }
    };
    /** @brief The line for a component: a control's parameter, or its tooltip. @param c what the mouse is over */
    void over(juce::Component* c);
    /** @brief The value and the matrix anew while the mouse rests on a control. */
    void timerCallback() override;
    /** @brief The text for a parameter. @param id the parameter @param name receives its name @return the rest */
    juce::String describe(int id, juce::String& name) const;

    AdditivProcessor& proc_;        ///< the processor
    juce::Component& watched_;      ///< the window
    Tracker tracker_;               ///< the mouse's follower
    int id_ = -1;                   ///< the parameter shown, -1 none
    juce::String name_;             ///< its name (or nothing for a tooltip)
    juce::String text_;             ///< the rest of the line
};
