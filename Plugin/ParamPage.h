/**
 * @file ParamPage.h
 * @brief A page of the panel: group boxes in their family's colour, each holding the frame's knobs,
 *        choosers and switches for a list of parameters, bound to the host's parameters. The boxes
 *        flow left to right and wrap, in as few rows as they allow; every control has the right-click
 *        menu (MIDI learn, forget, default). Two parameters that make a span (Lowest and Highest) are one
 *        bar with two ends; two knobs that go together (an echo's Time L and R) can be linked by a small
 *        chain between them, so that one turned turns the other in proportion.
 */
#pragma once

#include "Frame.h"
#include "PluginProcessor.h"

#include <functional>
#include <memory>
#include <vector>

/** @brief One page of parameter groups. */
class ParamPage final : public juce::Component, public juce::DragAndDropTarget, private juce::Timer {
public:
    /** @brief A group's state, for the page's marks in the tabs. */
    struct GroupState {
        juce::String m_title;     ///< the group's title
        juce::Colour m_colour;    ///< its family's colour
        bool m_gated = false;     ///< it has main controls that wake it
        bool m_awake = true;      ///< a main control is up (or a row moves it)
    };
    /**
     * @brief A group: its title, its family (the colour), its parameters in order, and what wakes it --
     *        the parameters that, all at nothing, leave the group resting (its other controls dimmed,
     *        "off" by its title); they are drawn as its main controls.
     */
    struct Group {
        juce::String m_title;                      ///< the box's title
        frame::FamilyE m_family;                    ///< its colour
        std::vector<additiv::ParamIdE> m_ids;       ///< its parameters
        std::vector<additiv::ParamIdE> m_gate = {}; ///< what wakes it (none: always awake)
        std::vector<std::pair<additiv::ParamIdE, additiv::ParamIdE>> m_spans = {};   ///< low and high of a span, shown as one bar
        std::vector<std::pair<additiv::ParamIdE, additiv::ParamIdE>> m_links = {};   ///< two knobs that may be linked
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
    /** @brief Every group's state now (also while the page is hidden). @return them, in order */
    std::vector<GroupState> groupStates() const;

    /** @brief Every page shows the MIDI controller learned on each knob as a small mark (the header's MIDI button). */
    static inline bool M_ShowMidi = false;
    /** @brief Who hears that a source was dropped on a knob: the editor, found up the page's parents (every page, also those made later). */
    struct DropListener {
        virtual ~DropListener() = default;
        /** @brief A row was made (or not: row -1). @param s the source @param id the parameter @param row the row */
        virtual void modDropped(additiv::ModSourceE s, int id, int row) = 0;
    };
    /** @brief Only a source's chip (ModDock). @param d the drag @return true for one */
    bool isInterestedInDragSource(const SourceDetails& d) override;
    /** @brief The knob under the source lights up. @param d the drag */
    void itemDragMove(const SourceDetails& d) override;
    /** @brief Nothing lights up any more. @param d the drag */
    void itemDragExit(const SourceDetails& d) override;
    /** @brief The source modulates the knob under it: a new row. @param d the drop */
    void itemDropped(const SourceDetails& d) override;

private:
    /** @brief Wakes or rests the groups as their gates say; repaints the knobs the matrix moves. */
    void timerCallback() override;
    /** @brief The knob a source would modulate at a point. @param p the point @return its parameter, -1 none */
    int targetAt(juce::Point<int> p) const;

    /** @brief One control in a box. */
    struct Cell {
        juce::Component* m_control = nullptr;   ///< the knob, chooser or switch
        juce::Label* m_label = nullptr;         ///< its name
        int m_width = 0;                        ///< its cell's width
        int m_id = -1;                          ///< its parameter
        int m_id2 = -1;                         ///< a span's high end, -1 none
        bool m_gate = false;                    ///< one of what wakes its group (a main control)
        bool m_moved = false;                   ///< a matrix row was on it at the last look
        bool m_away = false;                    ///< it stood away from its preset at the last look
    };
    /** @brief One group box. */
    struct Box {
        juce::String m_title;                   ///< its title
        juce::Colour m_colour;                  ///< its family's colour
        std::vector<Cell> m_cells;              ///< its controls
        juce::Rectangle<int> m_bounds;          ///< where it was laid out
        int m_width = 0;                        ///< the width it needs
        bool m_resting = false;                 ///< every gate at nothing: its other controls dimmed
    };
    /** @brief Whether a group has main controls. @param b the group @return true then */
    static bool gated(const Box& b);
    /** @brief Whether a group is awake: a main control up, or a row moving one. @param b the group @return true then */
    bool awake(const Box& b) const;
    /**
     * @brief Where every box goes at a width: in rows, in order -- or, where a later box taken up into an
     *        earlier row's gap saves a row, in the fewest rows.
     * @param width  pixels
     * @return       the boxes' rectangles, in the boxes' order
     */
    std::vector<juce::Rectangle<int>> flow(int width) const;

    AdditivProcessor& m_proc;                                                      ///< whose parameters
    std::vector<Box> m_boxes;                                                      ///< the boxes
    /** @brief Two linked knobs and their chain. */
    struct Link {
        juce::Slider* m_a = nullptr;                    ///< the first knob
        juce::Slider* m_b = nullptr;                    ///< the second
        std::unique_ptr<juce::Button> m_chain;          ///< links and unlinks them
        double m_ratio = 1.0;                           ///< b over a, taken when a drag begins
        bool m_busy = false;                            ///< one is being moved by the other
    };
    std::vector<std::unique_ptr<Link>> m_links;                                    ///< the linked pairs
    int m_flashId = -1;                                                            ///< the control lighting up, -1 none
    int m_dropId = -1;                                                             ///< the knob under a dragged source, -1 none
    additiv::ModSourceE m_dropSource = additiv::ModSourceE::OFF;                     ///< the source dragged
    double m_flashUntil = 0.0;                                                     ///< until when (ms)
    std::vector<additiv::ParamIdE> m_ids;                                           ///< every parameter on the page
    juce::OwnedArray<juce::Component> m_controls;                                  ///< the controls
    juce::OwnedArray<juce::Label> m_labels;                                        ///< their names
    std::vector<std::unique_ptr<juce::SliderParameterAttachment>> m_sliders;      ///< knobs to parameters
    std::vector<std::unique_ptr<juce::ComboBoxParameterAttachment>> m_combos;     ///< choosers to parameters
    std::vector<std::unique_ptr<juce::ButtonParameterAttachment>> m_buttons;      ///< switches to parameters
};
