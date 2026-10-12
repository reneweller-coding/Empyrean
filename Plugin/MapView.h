/**
 * @file MapView.h
 * @brief The Map page: every preset a point on the map (across dark to bright, up still to moving),
 *        the cursor that blends the presets round it, and the routes over it -- waypoints with the
 *        time to travel to each (how long the sound takes to turn into the next) and to stay.
 */
#pragma once

#include "Frame.h"
#include "ParamPage.h"
#include "PluginProcessor.h"

#include <memory>
#include <vector>

/** @brief The map itself (see the file comment). */
class MapView final : public juce::Component, private juce::Timer {
public:
    /** @brief The map of @p proc's bank. @param proc the processor */
    explicit MapView(AdditivProcessor& proc);
    /** @brief The points (cached), the axes, the route, the neighbours, the cursor, the name under the mouse. */
    void paint(juce::Graphics& g) override;
    /** @brief The cached points are drawn again at the new size. */
    void resized() override;
    /** @brief A click: on a preset the cursor goes there, elsewhere to that place; with Shift a waypoint is added. @param e the click */
    void mouseDown(const juce::MouseEvent& e) override;
    /** @brief The cursor follows the mouse. @param e the drag */
    void mouseDrag(const juce::MouseEvent& e) override;
    /** @brief The preset under the mouse, for its name. @param e the move */
    void mouseMove(const juce::MouseEvent& e) override;
    /** @brief No name when the mouse has left. @param e the exit */
    void mouseExit(const juce::MouseEvent& e) override;

private:
    /** @brief Repaints while something moves. */
    void timerCallback() override;
    /** @brief The plane's square in the component. @return it */
    juce::Rectangle<float> plane() const;
    /** @brief A place on the map on the screen. @param x across @param y up @return the point */
    juce::Point<float> toScreen(float x, float y) const;
    /** @brief The place under a screen point, clamped. @param p the point @return (x, y) on the map */
    juce::Point<float> toMap(juce::Point<float> p) const;
    /** @brief The preset whose point is within reach of a screen point. @param p the point @param reach pixels @return its index, -1 none */
    int presetAt(juce::Point<float> p, float reach) const;

    AdditivProcessor& proc_;            ///< whose bank and cursor
    juce::Image points_;                ///< every preset's point, drawn once per size
    int hover_ = -1;                    ///< the preset under the mouse, -1 none
};

/** @brief The route editor beside the map. */
class RouteView final : public juce::Component, private juce::Timer {
public:
    /** @brief The routes of @p proc. @param proc the processor */
    explicit RouteView(AdditivProcessor& proc);
    /** @brief The rows. */
    void resized() override;
    /** @brief The box and its headings. */
    void paint(juce::Graphics& g) override;
    /** @brief The height it needs with its waypoints. @return pixels */
    int idealHeight() const;
    /** @brief What it shows, in words. @return the text */
    juce::String describe() const;

private:
    /** @brief One waypoint's row. */
    struct Row {
        juce::Label name;              ///< its number and preset (or place)
        juce::TextEditor travel;       ///< the time to travel there, m:ss
        juce::TextEditor hold;         ///< the time to stay, m:ss
        juce::TextEditor radius;       ///< its blend radius
        juce::TextButton remove{ "x" };   ///< removes it
    };
    /** @brief Status, and the rows rebuilt when the route changed. */
    void timerCallback() override;
    /** @brief The rows anew from the route. */
    void rebuild();
    /** @brief Fills the route box. */
    void refreshRoutes();
    /** @brief A row's fields written into its waypoint. @param i the row */
    void commit(int i);
    /** @brief Asks for a name and writes the route into the routes folder. */
    void save();

    AdditivProcessor& proc_;                                   ///< whose route
    juce::ComboBox routes_;                                    ///< the routes to choose
    std::vector<AdditivProcessor::RouteEntry> entries_;        ///< what the box's items are (id = index + 1)
    /** @brief Starts the route. */
    frame::IconButton play_{ frame::IconButton::Icon::Play, "Walks the route from where the cursor is: each waypoint reached over its travel time, held for its hold time" };
    /** @brief Stops it. */
    frame::IconButton stop_{ frame::IconButton::Icon::Stop, "Stops the route; the cursor stays" };
    juce::ToggleButton loop_{ "Loop" };                        ///< starts over after the last point
    juce::Label status_;                                       ///< where the route is
    juce::Label heads_;                                        ///< the columns' names
    std::vector<std::unique_ptr<Row>> rows_;                   ///< one a waypoint
    juce::TextButton addCursor_{ "+ Cursor" };                 ///< the cursor's place as a waypoint
    juce::TextButton addPreset_{ "+ Preset" };                 ///< the preset playing as a waypoint
    juce::TextButton clear_{ "Clear" };                        ///< empties the route
    /** @brief Saves the route. */
    frame::IconButton save_{ frame::IconButton::Icon::Save, "Writes the route into Documents/Empyrean/Routes under a name" };
    /** @brief Opens the routes folder. */
    frame::IconButton folder_{ frame::IconButton::Icon::Open, "Opens the routes folder: routes are plain text, a waypoint a line (preset or x,y | travel s | hold s | radius)" };
    std::unique_ptr<juce::FileChooser> chooser_;               ///< the save dialog while it is open
    juce::String shown_;                                       ///< the route the rows show (its text)
    int tick_ = 0;                                             ///< timer ticks
};

/** @brief The Map page: its knobs above, the map and the route editor below. */
class MapPage final : public juce::Component {
public:
    /**
     * @brief Builds the page.
     * @param proc     the processor
     * @param actions  the controls' right-click menu
     * @param live     the live rings
     */
    MapPage(AdditivProcessor& proc, const frame::ControlActions& actions, frame::LiveRings* live);
    /** @brief The height the page needs at a width. @param w pixels @return pixels */
    int heightFor(int w) const;
    /** @brief The knobs above; the map left, the routes right. */
    void resized() override;
    /** @brief The page in words. @return the text */
    juce::String describe() const;

private:
    ParamPage params_;    ///< Map, Radius, Map Glide, Route Speed
    MapView map_;         ///< the map
    RouteView route_;     ///< the routes
};
