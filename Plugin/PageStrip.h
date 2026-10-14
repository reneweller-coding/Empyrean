/**
 * @file PageStrip.h
 * @brief The pages' strip, in place of plain tabs: the pages in sections under small headings (Sound,
 *        Harmony, Space, Modulation, Explore), and each page with what is on in it -- a dot for every
 *        group with main controls, lit in the group's colour while it is awake, faint while it rests;
 *        a count where a page counts something (the matrix's rows); the page's name dimmed while all
 *        of its groups rest. Hovering a page tells which groups are on and which are off.
 */
#pragma once

#include <juce_gui_basics/juce_gui_basics.h>

#include <functional>
#include <vector>

/** @brief The pages' strip (see the file comment). */
class PageStrip final : public juce::Component, public juce::SettableTooltipClient {
public:
    static constexpr int kHeight = 38;   ///< the strip's height: the headings and the pages

    /** @brief One group's mark. */
    struct Dot {
        juce::Colour colour;   ///< its group's colour
        bool on = false;       ///< awake
    };
    /** @brief A page's state. */
    struct Status {
        std::vector<Dot> dots;   ///< its groups with main controls
        juce::String badge;      ///< a count, "" none
        bool quiet = false;      ///< everything on it rests
        juce::String tip;        ///< which groups are on, which off
    };
    /** @brief A section: its heading ("" none) and its pages, by their index. */
    struct Section {
        juce::String name;       ///< the heading
        std::vector<int> pages;  ///< the pages
    };

    /** @brief The pages and their sections. @param names every page's name, by index @param sections the sections, in order */
    void setPages(juce::StringArray names, std::vector<Section> sections);
    /** @brief A page's state, asked for on refresh(). */
    std::function<Status(int)> statusOf;
    /** @brief A page was clicked. */
    std::function<void(int)> onSelect;
    /** @brief The page in front. @param page its index */
    void setCurrent(int page);
    /** @brief Asks every page's state anew; repaints when one changed. */
    void refresh();
    /** @brief The width the strip needs. @return pixels */
    int idealWidth() const;

    /** @brief The headings, the pages, their marks. @param g where */
    void paint(juce::Graphics& g) override;
    /** @brief The pages' places. */
    void resized() override;
    /** @brief Shows the page clicked. @param e the click */
    void mouseDown(const juce::MouseEvent& e) override;
    /** @brief The page under the mouse lights up. @param e the move */
    void mouseMove(const juce::MouseEvent& e) override;
    /** @brief Nothing lights up. @param e the exit */
    void mouseExit(const juce::MouseEvent& e) override;
    /** @brief The page under the mouse: its groups on and off (the info line shows it). @return the text */
    juce::String getTooltip() override;

private:
    /** @brief A page's width: its name, its dots, its count. @param page its index @return pixels */
    int widthOf(int page) const;
    /** @brief The page at a point. @param p the point @return its index, -1 none */
    int pageAt(juce::Point<int> p) const;

    juce::StringArray names_;                       ///< the pages' names
    std::vector<Section> sections_;                 ///< the sections
    std::vector<Status> status_;                    ///< every page's state as last asked
    std::vector<juce::Rectangle<int>> bounds_;      ///< every page's place
    std::vector<juce::Rectangle<int>> heads_;       ///< every section's heading's place
    int current_ = 0;                               ///< the page in front
    int hover_ = -1;                                ///< the page under the mouse
};
