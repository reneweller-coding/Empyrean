/**
 * @file JourneyView.h
 * @brief The Journey page: the glide between presets, the A/B morph and its two slots, and the
 *        journeys -- presets in a row, each glided into the next, for a whole night (Journey.h).
 */
#pragma once

#include "Frame.h"
#include "ParamPage.h"
#include "PluginProcessor.h"

#include <vector>

/** @brief The slots and the journeys (below the page's knobs). */
class JourneyView final : public juce::Component, private juce::Timer {
public:
    /** @brief The slots and journeys of @p proc. @param proc the processor */
    explicit JourneyView(AdditivProcessor& proc);
    /** @brief The rows. */
    void resized() override;
    /** @brief The boxes and their headings. */
    void paint(juce::Graphics& g) override;
    /** @brief The height it needs. @return pixels */
    static int idealHeight() { return 2 * 34 + 30 + 2 * 34 + 30 + 20; }
    /** @brief What it shows, in words, for the help. @return the text */
    juce::String describe() const;

private:
    /** @brief One slot's row. */
    struct SlotRow {
        juce::Label name;                    ///< what the slot holds
        juce::TextButton fromSound{ "This sound" };   ///< the slot takes what plays
        juce::TextButton fromPreset{ "Preset ..." };  ///< the slot takes a preset from the bank
    };
    /** @brief The status, the slots' names and the journey list now and then. */
    void timerCallback() override;
    /** @brief Fills the journey box: the ones that come with the instrument, then the player's files. */
    void refreshJourneys();
    /** @brief The bank as a menu; the chosen preset goes to a slot. @param slot 0 or 1 */
    void choosePreset(int slot);
    /** @brief Starts the journey chosen in the box. */
    void startChosen();
    /** @brief Asks for a name and writes the journey being written into the journeys folder. */
    void saveDraft();

    AdditivProcessor& proc_;                                    ///< whose slots and journeys
    SlotRow slots_[2];                                          ///< A and B
    juce::ComboBox journeys_;                                   ///< the journeys to choose from
    std::vector<AdditivProcessor::JourneyEntry> entries_;       ///< what the box's items are (item id = index + 1)
    /** @brief Starts the journey chosen in the box. */
    frame::IconButton start_{ frame::IconButton::Icon::Play, "Starts the journey chosen beside it: each preset held for its time, then glided into the next" };
    /** @brief Stops the journey under way. */
    frame::IconButton stop_{ frame::IconButton::Icon::Stop, "Stops the journey; the preset playing stays" };
    juce::Label status_;                                        ///< the journey under way
    juce::TextButton add_{ "+ This preset" };                   ///< appends the preset playing to the journey being written
    /** @brief Saves the journey being written. */
    frame::IconButton save_{ frame::IconButton::Icon::Save, "Writes the journey being written into Documents/Empyrean/Journeys under a name, and offers it in the box" };
    /** @brief Opens the journeys folder. */
    frame::IconButton folder_{ frame::IconButton::Icon::Open, "Opens the journeys folder (Documents/Empyrean/Journeys): journeys are plain text, one preset a line" };
    juce::Label draft_;                                         ///< how many presets the journey being written has
    std::unique_ptr<juce::FileChooser> chooser_;                ///< the save dialog while it is open
    int tick_ = 0;                                              ///< timer ticks, for the slower refreshes
};

/** @brief The Journey page: its knobs (the glide, the morph) above, the slots and journeys below. */
class JourneyPage final : public juce::Component {
public:
    /**
     * @brief Builds the page.
     * @param proc     the processor
     * @param actions  the controls' right-click menu
     * @param live     the live rings
     */
    JourneyPage(AdditivProcessor& proc, const frame::ControlActions& actions, frame::LiveRings* live);
    /** @brief The height the page needs at a width. @param w pixels @return pixels */
    int heightFor(int w) const { return params_.heightFor(w) + JourneyView::idealHeight() + 10; }
    /** @brief The knobs above, the rest below. */
    void resized() override;
    /** @brief The page in words. @return the text */
    juce::String describe() const { return params_.describe() + view_.describe(); }

private:
    ParamPage params_;    ///< the glide and the morph's knobs
    JourneyView view_;    ///< the slots and the journeys
};
