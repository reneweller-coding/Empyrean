/**
 * @file HallView.h
 * @brief The Hall page: the modal hall's knobs, its growth (Grow, Memory) and its memory drawn band by
 *        band -- what it has heard over the hours, where it rings longer (growth) or shorter (erosion)
 *        -- with saving and loading a hall (Documents/Empyrean/Halls) and starting a new one.
 */
#pragma once

#include "Frame.h"
#include "ParamPage.h"
#include "PluginProcessor.h"

#include <memory>
#include <vector>

/** @brief The hall's memory drawn, its age, and the buttons that keep it (see the file comment). */
class HallView final : public juce::Component, private juce::Timer {
public:
    /** @brief The memory of @p proc's engine. @param proc the processor */
    explicit HallView(AdditivProcessor& proc);
    /** @brief The buttons and the status. */
    void resized() override;
    /** @brief The box, the curve, the octaves. */
    void paint(juce::Graphics& g) override;
    /** @brief What it shows, in words. @return the text */
    juce::String describe() const;

private:
    /** @brief The memory fetched anew, a few times a second. */
    void timerCallback() override;
    /** @brief The curve's area in the box. @return pixels */
    juce::Rectangle<float> plot() const;
    /** @brief Asks for a name and writes the hall into the halls folder. */
    void save();
    /** @brief Asks for a hall file and hears it in. */
    void load();
    /** @brief The status line: name, age, what Grow does with it. @return the text */
    juce::String statusText() const;

    AdditivProcessor& proc_;                       ///< the processor
    std::vector<float> memory_;                    ///< the memory as last fetched, kHallBins bands, 1 the average
    double age_ = 0.0;                             ///< its seconds learnt
    /** @brief Saves the hall. */
    frame::IconButton save_{ frame::IconButton::Icon::Save, "Writes the hall's memory into Documents/Empyrean/Halls under a name" };
    /** @brief Loads a hall. */
    frame::IconButton load_{ frame::IconButton::Icon::Open, "Loads a hall's memory from a file: the hall rings as that one had grown" };
    juce::TextButton new_{ "New hall" };           ///< forgets everything: every band at the average
    juce::Label status_;                           ///< name, age, what Grow does
    std::unique_ptr<juce::FileChooser> chooser_;   ///< the dialog while it is open
};

/** @brief The Hall page: the modal hall's knobs and its growth above, the memory below. */
class HallPage final : public juce::Component {
public:
    /**
     * @brief Builds the page.
     * @param proc     the processor
     * @param actions  the controls' right-click menu
     * @param live     the live rings
     */
    HallPage(AdditivProcessor& proc, const frame::ControlActions& actions, frame::LiveRings* live);
    /** @brief The height the page needs at a width. @param w pixels @return pixels */
    int heightFor(int w) const;
    /** @brief The knobs at the top, the memory below. */
    void resized() override;
    /** @brief The page in words. @return the text */
    juce::String describe() const;

private:
    ParamPage params_;    ///< Modal Hall; Growth
    HallView view_;       ///< the memory
};
