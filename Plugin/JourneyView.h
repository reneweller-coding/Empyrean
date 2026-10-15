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
    /** @brief The journey chosen in the box, read (the timeline's preview). @param out receives it @return false when none is chosen */
    bool chosen(additiv::Journey& out) const
    {
        const int id = journeys_.getSelectedId();
        return id > 0 && id <= static_cast<int>(entries_.size()) && AdditivProcessor::readJourney(entries_[static_cast<size_t>(id - 1)], out);
    }

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

/**
 * @brief The journey's timeline: its steps as a band over time, each as wide as it lasts (its glide a ramp into it),
 *        the preset's name on it; the step under way lit and filling as its time runs, and a click on another goes
 *        there. Before a journey runs, the one chosen in the box, faint. Under it, once one is being written, the
 *        player's own journey to edit by the hand: a block's right edge dragged sets how long it stays, its ramp how
 *        long it glides in, the block itself dragged moves it, a right-click removes it. "Edit a copy" takes the
 *        journey above into it.
 */
class JourneyTimeline final : public juce::Component, private juce::Timer {
public:
    /** @brief The timeline of @p proc's journeys. @param proc the processor @param view the box it previews */
    JourneyTimeline(AdditivProcessor& proc, const JourneyView& view);
    /** @brief The bands, the steps, where the journey is. @param g where */
    void paint(juce::Graphics& g) override;
    /** @brief The copy button. */
    void resized() override;
    /** @brief A step clicked (skip there), or a stroke on the one being written begins. @param e the press */
    void mouseDown(const juce::MouseEvent& e) override;
    /** @brief The edge, the ramp or the block follows. @param e the drag */
    void mouseDrag(const juce::MouseEvent& e) override;
    /** @brief A moved block lands where it was let go. @param e the release */
    void mouseUp(const juce::MouseEvent& e) override;
    /** @brief The cursor says what a drag would do. @param e the move */
    void mouseMove(const juce::MouseEvent& e) override;

private:
    /** @brief One step's place in a band: where its glide begins, where it stands, where it ends. */
    struct Block {
        float x0 = 0.0f;   ///< the glide's start
        float xg = 0.0f;   ///< the step's own start
        float x1 = 0.0f;   ///< its end
    };
    /** @brief What a drag on the journey being written does. */
    enum class Drag { None, Dwell, Glide, Move };
    /** @brief Twice a second while it is seen (ten while a drag is on). */
    void timerCallback() override;
    /** @brief The journey above: the one under way, else the one chosen. @param out receives it @return false when there is none */
    bool upper(additiv::Journey& out) const;
    /** @brief A band's rectangle. @param which 0 the journey above, 1 the one being written @return pixels */
    juce::Rectangle<float> band(int which) const;
    /** @brief The blocks of a journey in a band, on its own time scale. @param j the journey @param r the band @return them */
    static std::vector<Block> blocks(const additiv::Journey& j, juce::Rectangle<float> r);
    /** @brief Draws a journey in a band. @param g where @param j the journey @param r the band @param lit the step lit, -1 none @param done how far it has gone, 0 .. 1 @param faint drawn faint */
    void drawBand(juce::Graphics& g, const additiv::Journey& j, juce::Rectangle<float> r, int lit, double done, bool faint) const;
    /** @brief The block of the one being written under a point. @param p the point @param drag receives what a drag there does @return its index, -1 none */
    int draftAt(juce::Point<float> p, Drag& drag) const;

    AdditivProcessor& proc_;            ///< whose journeys
    const JourneyView& view_;           ///< the box
    juce::TextButton copy_{ "Edit a copy" };   ///< the journey above into the one being written
    Drag drag_ = Drag::None;            ///< the drag under way
    int dragStep_ = -1;                 ///< its step
    float dragX_ = 0.0f;                ///< where it began
    double perPixel_ = 1.0;             ///< seconds a pixel when it began (the scale held while dragging)
    additiv::JourneyStep before_;       ///< the step as it was when it began
    int dropAt_ = -1;                   ///< where a moved block would land, -1 nowhere
};

/** @brief The Journey page: its knobs (the glide, the morph) above, the slots and journeys below, the timeline under them. */
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
    int heightFor(int w) const { return params_.heightFor(w) + JourneyView::idealHeight() + 10 + kTimelineMin + 10; }
    /** @brief The knobs above, the rest below. */
    void resized() override;
    /** @brief The page in words. @return the text */
    juce::String describe() const { return params_.describe() + view_.describe(); }

private:
    static constexpr int kTimelineMin = 110;   ///< the timeline's least height (it takes what the page leaves; two bands once a journey is being written)
    ParamPage params_;          ///< the glide and the morph's knobs
    JourneyView view_;          ///< the slots and the journeys
    JourneyTimeline timeline_;  ///< the journey over time
};
