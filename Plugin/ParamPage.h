/**
 * @file ParamPage.h
 * @brief A page of the panel: group boxes in their family's colour, each holding the frame's knobs,
 *        choosers and switches for a list of parameters, bound to the host's parameters. The boxes
 *        flow left to right and wrap, in as few rows as they allow; every control has the right-click
 *        menu (MIDI learn, forget, default).
 */
#pragma once

#include "Frame.h"
#include "PluginProcessor.h"

#include <memory>
#include <vector>

/** @brief One page of parameter groups. */
class ParamPage final : public juce::Component, private juce::Timer {
public:
    /**
     * @brief A group: its title, its family (the colour), its parameters in order, and what wakes it --
     *        the parameters that, all at nothing, leave the group resting (its other controls dimmed,
     *        "off" by its title); they are drawn as its main controls.
     */
    struct Group {
        juce::String title;                      ///< the box's title
        frame::Family family;                    ///< its colour
        std::vector<additiv::ParamId> ids;       ///< its parameters
        std::vector<additiv::ParamId> gate = {}; ///< what wakes it (none: always awake)
    };

    /**
     * @brief Builds the page.
     * @param proc     the processor whose parameters the controls edit
     * @param actions  what the controls' right-click menu does
     * @param groups   the boxes, in order
     * @param live     the live rings every knob joins (may be null)
     */
    ParamPage(AdditivProcessor& proc, const frame::ControlActions& actions, std::vector<Group> groups, frame::LiveRings* live = nullptr);

    /** @brief Flows the boxes and lays out their cells. */
    void resized() override;
    /** @brief Draws the boxes and their titles, and a soft halo behind each group's main controls. */
    void paint(juce::Graphics& g) override;
    /** @brief The matrix rows on the knobs, as arcs round them (KnobMod.h). */
    void paintOverChildren(juce::Graphics& g) override;
    /** @brief The page's parameters in words, for the help: name, range, unit, key. @return the text */
    juce::String describe() const;
    /** @brief The height the page needs at a width. @param width pixels @return pixels */
    int heightFor(int width) const;
    /** @brief A parameter's control on this page. @param id the parameter @return it, null when it is not here */
    juce::Component* controlFor(int id) const;
    /** @brief The title of the box a parameter is in. @param id the parameter @return it, "" when it is not here */
    juce::String groupOf(int id) const;
    /** @brief Lets a parameter's control light up for a moment (the palette found it). @param id the parameter */
    void flash(int id);

private:
    /** @brief Wakes or rests the groups as their gates say; repaints the knobs the matrix moves. */
    void timerCallback() override;

    /** @brief One control in a box. */
    struct Cell {
        juce::Component* control = nullptr;   ///< the knob, chooser or switch
        juce::Label* label = nullptr;         ///< its name
        int width = 0;                        ///< its cell's width
        int id = -1;                          ///< its parameter
        bool gate = false;                    ///< one of what wakes its group (a main control)
        bool moved = false;                   ///< a matrix row was on it at the last look
        bool away = false;                    ///< it stood away from its preset at the last look
    };
    /** @brief One group box. */
    struct Box {
        juce::String title;                   ///< its title
        juce::Colour colour;                  ///< its family's colour
        std::vector<Cell> cells;              ///< its controls
        juce::Rectangle<int> bounds;          ///< where it was laid out
        int width = 0;                        ///< the width it needs
        bool resting = false;                 ///< every gate at nothing: its other controls dimmed
    };
    /**
     * @brief Where every box goes at a width: in rows, in order -- or, where a later box taken up into an
     *        earlier row's gap saves a row, in the fewest rows.
     * @param width  pixels
     * @return       the boxes' rectangles, in the boxes' order
     */
    std::vector<juce::Rectangle<int>> flow(int width) const;

    AdditivProcessor& proc_;                                                      ///< whose parameters
    std::vector<Box> boxes_;                                                      ///< the boxes
    int flashId_ = -1;                                                            ///< the control lighting up, -1 none
    double flashUntil_ = 0.0;                                                     ///< until when (ms)
    std::vector<additiv::ParamId> ids_;                                           ///< every parameter on the page
    juce::OwnedArray<juce::Component> controls_;                                  ///< the controls
    juce::OwnedArray<juce::Label> labels_;                                        ///< their names
    std::vector<std::unique_ptr<juce::SliderParameterAttachment>> sliders_;      ///< knobs to parameters
    std::vector<std::unique_ptr<juce::ComboBoxParameterAttachment>> combos_;     ///< choosers to parameters
    std::vector<std::unique_ptr<juce::ButtonParameterAttachment>> buttons_;      ///< switches to parameters
};
