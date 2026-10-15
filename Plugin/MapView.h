/**
 * @file MapView.h
 * @brief The Map page: every preset a point on the map (across dark to bright, up still to moving),
 *        the cursor that blends the presets round it, and the routes over it -- waypoints with the
 *        time to travel to each (how long the sound takes to turn into the next) and to stay. The map zooms
 *        (the wheel, round the mouse) and moves (a right- or middle-drag), with an overview in its corner.
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
    /** @brief The points (cached), the axes, the route, the neighbours, the cursor, the overview, the name under the mouse. @param g where */
    void paint(juce::Graphics& g) override;
    /** @brief The cached points and the overview are drawn again at the new size. */
    void resized() override;
    /** @brief A click: on a preset the cursor goes there, elsewhere to that place; with Shift a waypoint is added. @param e the click */
    void mouseDown(const juce::MouseEvent& e) override;
    /** @brief The cursor follows the mouse -- or the waypoint taken. @param e the drag */
    void mouseDrag(const juce::MouseEvent& e) override;
    /** @brief A waypoint let go. @param e the release */
    void mouseUp(const juce::MouseEvent& e) override;
    /** @brief The preset under the mouse, for its name. @param e the move */
    void mouseMove(const juce::MouseEvent& e) override;
    /** @brief No name when the mouse has left. @param e the exit */
    void mouseExit(const juce::MouseEvent& e) override;
    /** @brief The wheel zooms round the mouse; sideways it moves the view. @param e the event @param wheel how far */
    void mouseWheelMove(const juce::MouseEvent& e, const juce::MouseWheelDetails& wheel) override;
    /** @brief A pinch zooms round the mouse. @param e the event @param scaleFactor how much nearer */
    void mouseMagnify(const juce::MouseEvent& e, float scaleFactor) override;

private:
    /** @brief Repaints while something moves; the view after the cursor (or the preset playing), the names once it rests. */
    void timerCallback() override;
    /** @brief The overview's square, in the plane's corner. @return it */
    juce::Rectangle<float> mini() const;
    /** @brief Whether the view shows less than the whole map. @return true when zoomed in */
    bool zoomed() const;
    /** @brief What the view follows: the cursor with the map on (or no preset chosen), else the preset playing. @return its place on the map */
    juce::Point<float> focus() const;
    /** @brief Whether a place on the map is in the view. @param x across @param y up @return true when it is */
    bool inView(float x, float y) const;
    /** @brief A view kept on the map, between the whole of it and the closest. @param v the view @return it, kept */
    static juce::Rectangle<float> clampView(juce::Rectangle<float> v);
    /** @brief Shows a part of the map. @param v the part (x, y up, its size) @param stopGlide whether a glide after the cursor ends */
    void setView(juce::Rectangle<float> v, bool stopGlide = true);
    /** @brief Zooms with the place under a screen point kept there. @param s the point @param factor nearer above 1 */
    void zoomAt(juce::Point<float> s, float factor);
    /** @brief The view centred on a place, its size kept. @param x across @param y up */
    void centreOn(float x, float y);
    /** @brief The points of the presets in the view drawn into the cache. @param names whether their names are written where there is room */
    void drawPoints(bool names);
    /** @brief The whole map, small, for the overview. */
    void drawOverview();
    /** @brief The plane's square in the component. @return it */
    juce::Rectangle<float> plane() const;
    /** @brief A place on the map on the screen. @param x across @param y up @return the point */
    juce::Point<float> toScreen(float x, float y) const;
    /** @brief The place under a screen point, clamped. @param p the point @return (x, y) on the map */
    juce::Point<float> toMap(juce::Point<float> p) const;
    /** @brief The preset whose point is within reach of a screen point. @param p the point @param reach pixels @return its index, -1 none */
    int presetAt(juce::Point<float> p, float reach) const;

    AdditivProcessor& proc_;            ///< whose bank and cursor
    juce::Image points_;                ///< the points in the view (and their names), drawn when the view or the size changes
    juce::Image overview_;              ///< every preset's point, small, for the overview
    juce::Rectangle<float> view_{ 0.0f, 0.0f, 1.0f, 1.0f };   ///< the part of the map shown (x, y up, its size)
    juce::Rectangle<float> panFrom_;    ///< the view when the right button took it
    juce::Rectangle<float> target_;     ///< where the view glides after the cursor
    juce::Point<float> lastFocus_{ -1.0f, -1.0f };    ///< what the view follows, where it was a tick ago
    juce::uint32 viewChanged_ = 0;      ///< when the view last moved (ms)
    bool panning_ = false;              ///< the right or middle button moves the view
    bool overviewDrag_ = false;         ///< the mouse moves the view on the overview
    bool gliding_ = false;              ///< the view glides after the cursor
    bool focusInside_ = true;           ///< what the view follows was in the view a tick ago
    bool named_ = true;                 ///< the cache has the names of the view at rest
    juce::TextButton whole_{ "Whole map" };   ///< back to the whole map (shown while zoomed in)
    int hover_ = -1;                    ///< the preset under the mouse, -1 none
    int dragPoint_ = -1;                ///< the waypoint being moved, -1 none
    /** @brief The waypoint at a point. @param p pixels @return its index, -1 none */
    int waypointAt(juce::Point<float> p) const;
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

/** @brief The Map page: the map on the left, its knobs and the route editor beside it. */
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
    /** @brief The map left from the top, the knobs and the routes right. */
    void resized() override;
    /** @brief The page in words. @return the text */
    juce::String describe() const;

private:
    ParamPage params_;    ///< Map, Radius, Map Glide, Route Speed
    MapView map_;         ///< the map
    RouteView route_;     ///< the routes
};
