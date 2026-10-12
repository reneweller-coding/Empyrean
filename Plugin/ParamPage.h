/**
 * @file ParamPage.h
 * @brief A page of the panel: group boxes in their family's colour, each holding the frame's knobs,
 *        choosers and switches for a list of parameters, bound to the host's parameters. The boxes
 *        flow left to right and wrap; every control has the right-click menu (MIDI learn, forget,
 *        default).
 */
#pragma once

#include "Frame.h"
#include "PluginProcessor.h"

#include <memory>
#include <vector>

/** @brief One page of parameter groups. */
class ParamPage final : public juce::Component {
public:
    /** @brief A group: its title, its family (the colour) and its parameters in order. */
    struct Group {
        juce::String title;                      ///< the box's title
        frame::Family family;                    ///< its colour
        std::vector<additiv::ParamId> ids;       ///< its parameters
    };

    /**
     * @brief Builds the page.
     * @param proc     the processor whose parameters the controls edit
     * @param actions  what the controls' right-click menu does
     * @param groups   the boxes, in order
     */
    ParamPage(AdditivProcessor& proc, const frame::ControlActions& actions, std::vector<Group> groups);

    /** @brief Flows the boxes and lays out their cells. */
    void resized() override;
    /** @brief Draws the boxes and their titles. */
    void paint(juce::Graphics& g) override;
    /** @brief The page's parameters in words, for the help: name, range, unit, key. @return the text */
    juce::String describe() const;
    /** @brief The height the page needs at a width. @param width pixels @return pixels */
    int heightFor(int width) const;

private:
    /** @brief One control in a box. */
    struct Cell {
        juce::Component* control = nullptr;   ///< the knob, chooser or switch
        juce::Label* label = nullptr;         ///< its name
        int width = 0;                        ///< its cell's width
    };
    /** @brief One group box. */
    struct Box {
        juce::String title;                   ///< its title
        juce::Colour colour;                  ///< its family's colour
        std::vector<Cell> cells;              ///< its controls
        juce::Rectangle<int> bounds;          ///< where it was laid out
        int width = 0;                        ///< the width it needs
    };
    /**
     * @brief Where every box goes at a width.
     * @param width  pixels
     * @return       the boxes' rectangles, in order
     */
    std::vector<juce::Rectangle<int>> flow(int width) const;

    AdditivProcessor& proc_;                                                      ///< whose parameters
    std::vector<Box> boxes_;                                                      ///< the boxes
    std::vector<additiv::ParamId> ids_;                                           ///< every parameter on the page
    juce::OwnedArray<juce::Component> controls_;                                  ///< the controls
    juce::OwnedArray<juce::Label> labels_;                                        ///< their names
    std::vector<std::unique_ptr<juce::SliderParameterAttachment>> sliders_;      ///< knobs to parameters
    std::vector<std::unique_ptr<juce::ComboBoxParameterAttachment>> combos_;     ///< choosers to parameters
    std::vector<std::unique_ptr<juce::ButtonParameterAttachment>> buttons_;      ///< switches to parameters
};
