/**
 * @file MapView.cpp
 * @brief The Map page (MapView.h).
 */
#include "MapView.h"

#include "Theme.h"

#include "additiv/Journey.h"
#include "additiv/Presets.h"

using additiv::ParamId;

namespace {

/** @brief A section's colour: the sounds, the living systems, the sleep concerts, the visions. @param group the preset's group @return the colour */
juce::Colour sectionColour(int group)
{
    const frame::Skin& skin = additivui::skin();
    switch (group / 16) {
    case 0: return skin.family(frame::Family::Source);
    case 1: return additivui::colour::violet;
    case 2: return skin.family(frame::Family::Space);
    default: return skin.family(frame::Family::Envelope);
    }
}

} // namespace

// ---- the map

MapView::MapView(AdditivProcessor& proc) : proc_(proc) { startTimerHz(15); }

void MapView::timerCallback() { repaint(); }

juce::Rectangle<float> MapView::plane() const
{
    const auto b = getLocalBounds().toFloat().reduced(26.0f, 22.0f);
    const float s = std::min(b.getWidth(), b.getHeight());
    return { b.getX(), b.getY(), s, s };
}

juce::Point<float> MapView::toScreen(float x, float y) const
{
    const auto p = plane();
    return { p.getX() + x * p.getWidth(), p.getBottom() - y * p.getHeight() };
}

juce::Point<float> MapView::toMap(juce::Point<float> s) const
{
    const auto p = plane();
    return { juce::jlimit(0.0f, 1.0f, (s.x - p.getX()) / p.getWidth()), juce::jlimit(0.0f, 1.0f, (p.getBottom() - s.y) / p.getHeight()) };
}

int MapView::presetAt(juce::Point<float> s, float reach) const
{
    const auto m = toMap(s);
    const int i = proc_.nearestPreset(m.x, m.y);
    float x = 0.0f, y = 0.0f;
    additiv::presetMapPosition(i, x, y);
    return toScreen(x, y).getDistanceFrom(s) <= reach ? i : -1;
}

void MapView::resized()
{
    const auto p = plane();
    if (p.getWidth() < 10.0f) return;
    points_ = juce::Image(juce::Image::ARGB, getWidth(), getHeight(), true);
    juce::Graphics g(points_);
    for (int i = 0; i < additiv::numPresets(); ++i) {
        float x = 0.0f, y = 0.0f;
        additiv::presetMapPosition(i, x, y);
        const auto s = toScreen(x, y);
        g.setColour(sectionColour(additiv::preset(i).group).withAlpha(0.75f));
        g.fillEllipse(s.x - 1.6f, s.y - 1.6f, 3.2f, 3.2f);
    }
}

void MapView::paint(juce::Graphics& g)
{
    const frame::Skin& skin = additivui::skin();
    g.setColour(skin.group.withAlpha(0.92f));
    g.fillRoundedRectangle(getLocalBounds().toFloat(), skin.radius);
    g.setColour(skin.edge);
    g.drawRoundedRectangle(getLocalBounds().toFloat().reduced(0.5f), skin.radius, 1.0f);
    const auto p = plane();
    g.setColour(skin.bg.withAlpha(0.6f));
    g.fillRect(p);
    if (points_.isValid()) g.drawImageAt(points_, 0, 0);
    g.setColour(skin.dim);
    g.setFont(juce::FontOptions(11.0f));
    g.drawText("darker", juce::Rectangle<float>(p.getX(), p.getBottom() + 3.0f, 80.0f, 14.0f), juce::Justification::centredLeft);
    g.drawText("brighter", juce::Rectangle<float>(p.getRight() - 80.0f, p.getBottom() + 3.0f, 80.0f, 14.0f), juce::Justification::centredRight);
    g.saveState();
    g.addTransform(juce::AffineTransform::rotation(-juce::MathConstants<float>::halfPi, p.getX() - 8.0f, p.getBottom()));
    g.drawText("still", juce::Rectangle<float>(p.getX() - 8.0f, p.getBottom() - 14.0f, 60.0f, 14.0f), juce::Justification::centredLeft);
    g.drawText("moving", juce::Rectangle<float>(p.getX() - 8.0f + p.getHeight() - 60.0f, p.getBottom() - 14.0f, 60.0f, 14.0f), juce::Justification::centredRight);
    g.restoreState();

    // The route: its line, its numbered points, the one under way.
    const additiv::Route& r = proc_.route();
    if (r.count() > 0) {
        juce::Path line;
        for (int i = 0; i < r.count(); ++i) {
            const auto s = toScreen(r.point(i).x, r.point(i).y);
            if (i == 0) line.startNewSubPath(s);
            else line.lineTo(s);
        }
        if (proc_.routeLoop() && r.count() > 2) line.closeSubPath();
        g.setColour(skin.family(frame::Family::Motion).withAlpha(0.7f));
        g.strokePath(line, juce::PathStrokeType(1.5f));
        g.setFont(juce::FontOptions(11.0f, juce::Font::bold));
        for (int i = 0; i < r.count(); ++i) {
            const auto s = toScreen(r.point(i).x, r.point(i).y);
            const bool now = r.running() && r.segment() == i;
            g.setColour(now ? skin.ink : skin.family(frame::Family::Motion));
            g.fillEllipse(s.x - 7.0f, s.y - 7.0f, 14.0f, 14.0f);
            g.setColour(skin.bg);
            g.drawText(juce::String(i + 1), juce::Rectangle<float>(s.x - 7.0f, s.y - 7.0f, 14.0f, 14.0f), juce::Justification::centred);
        }
    }

    // The cursor, its radius and the presets it blends.
    const bool on = proc_.value(static_cast<int>(ParamId::MapActive)) >= 0.5f;
    const float cx = proc_.value(static_cast<int>(ParamId::MapX)), cy = proc_.value(static_cast<int>(ParamId::MapY));
    const float cr = proc_.value(static_cast<int>(ParamId::MapRadius));
    const auto c = toScreen(cx, cy);
    if (on) {
        const additiv::MapNeighbours b = additiv::mapNeighbours(cx, cy, cr);
        for (int k = 0; k < b.count; ++k) {
            if (b.weight[k] < 0.02f) continue;
            float x = 0.0f, y = 0.0f;
            additiv::presetMapPosition(b.index[k], x, y);
            const auto s = toScreen(x, y);
            g.setColour(skin.ink.withAlpha(0.25f + 0.75f * b.weight[k]));
            const float rad = 3.0f + 5.0f * b.weight[k];
            g.drawEllipse(s.x - rad, s.y - rad, 2.0f * rad, 2.0f * rad, 1.2f);
        }
        const float rr = cr * p.getWidth();
        g.setColour(additivui::colour::accent.withAlpha(0.12f));
        g.fillEllipse(c.x - rr, c.y - rr, 2.0f * rr, 2.0f * rr);
        g.setColour(additivui::colour::accent);
        g.drawEllipse(c.x - rr, c.y - rr, 2.0f * rr, 2.0f * rr, 1.0f);
        g.fillEllipse(c.x - 4.0f, c.y - 4.0f, 8.0f, 8.0f);
    } else {
        g.setColour(skin.dim.withAlpha(0.6f));
        g.drawEllipse(c.x - 5.0f, c.y - 5.0f, 10.0f, 10.0f, 1.0f);
    }

    // The preset playing (without the map), and the one under the mouse with its name.
    const int cp = proc_.currentPreset();
    if (!on && cp >= 0) {
        float x = 0.0f, y = 0.0f;
        additiv::presetMapPosition(cp, x, y);
        const auto s = toScreen(x, y);
        g.setColour(skin.ink);
        g.drawEllipse(s.x - 5.0f, s.y - 5.0f, 10.0f, 10.0f, 1.5f);
    }
    if (hover_ >= 0) {
        float x = 0.0f, y = 0.0f;
        additiv::presetMapPosition(hover_, x, y);
        const auto s = toScreen(x, y);
        g.setColour(skin.ink);
        g.drawEllipse(s.x - 5.0f, s.y - 5.0f, 10.0f, 10.0f, 1.2f);
        const juce::String name = juce::String(additiv::kPresetGroupNames[additiv::preset(hover_).group]) + ": " + additiv::preset(hover_).name;
        g.setFont(juce::FontOptions(12.0f));
        const float w = static_cast<float>(juce::GlyphArrangement::getStringWidthInt(g.getCurrentFont(), name)) + 12.0f;
        juce::Rectangle<float> box(s.x + 8.0f, s.y - 22.0f, w, 18.0f);
        if (box.getRight() > p.getRight()) box.setX(s.x - 8.0f - w);
        g.setColour(skin.raised.withAlpha(0.95f));
        g.fillRoundedRectangle(box, 4.0f);
        g.setColour(skin.ink);
        g.drawText(name, box, juce::Justification::centred);
    }
}

void MapView::mouseDown(const juce::MouseEvent& e)
{
    if (!plane().expanded(4.0f).contains(e.position)) return;
    const int hit = presetAt(e.position, 7.0f);
    const auto m = toMap(e.position);
    if (e.mods.isShiftDown()) {   // a waypoint: the preset under the mouse, or this place
        proc_.addWaypoint(m.x, m.y, hit);
        return;
    }
    if (hit >= 0) {
        float x = 0.0f, y = 0.0f;
        additiv::presetMapPosition(hit, x, y);
        proc_.moveCursor(x, y);
    } else {
        proc_.moveCursor(m.x, m.y);
    }
}

void MapView::mouseDrag(const juce::MouseEvent& e)
{
    if (e.mods.isShiftDown()) return;
    const auto m = toMap(e.position);
    proc_.moveCursor(m.x, m.y);
}

void MapView::mouseMove(const juce::MouseEvent& e) { hover_ = plane().expanded(4.0f).contains(e.position) ? presetAt(e.position, 7.0f) : -1; }

void MapView::mouseExit(const juce::MouseEvent&) { hover_ = -1; }

// ---- the routes

RouteView::RouteView(AdditivProcessor& proc) : proc_(proc)
{
    routes_.setTextWhenNothingSelected("Choose a route");
    routes_.setTooltip("Routes over the map: the ones that come with the instrument, then yours (Documents/Empyrean/Routes)");
    routes_.onChange = [this] {
        const int id = routes_.getSelectedId();
        if (id > 0 && id <= static_cast<int>(entries_.size())) proc_.loadRoute(entries_[static_cast<size_t>(id - 1)]);
    };
    play_.onClick = [this] { proc_.startRoute(); };
    stop_.onClick = [this] { proc_.stopRoute(); };
    loop_.setToggleState(proc_.routeLoop(), juce::dontSendNotification);
    loop_.onClick = [this] { proc_.setRouteLoop(loop_.getToggleState()); };
    addCursor_.setTooltip("Appends the cursor's place as a waypoint (Shift-click on the map does it too)");
    addCursor_.onClick = [this] {
        const float x = proc_.value(static_cast<int>(ParamId::MapX)), y = proc_.value(static_cast<int>(ParamId::MapY));
        proc_.addWaypoint(x, y, -1);
    };
    addPreset_.setTooltip("Appends the preset playing as a waypoint");
    addPreset_.onClick = [this] {
        if (proc_.currentPreset() >= 0) proc_.addWaypoint(0.0f, 0.0f, proc_.currentPreset());
    };
    clear_.onClick = [this] { proc_.clearRoute(); };
    save_.onClick = [this] { save(); };
    folder_.onClick = [] {
        const juce::File dir = AdditivProcessor::routeFolder();
        dir.createDirectory();
        dir.startAsProcess();
    };
    for (juce::Label* l : { &status_, &heads_ }) {
        l->setColour(juce::Label::textColourId, additivui::skin().dim);
        l->setFont(juce::FontOptions(12.0f));
        addAndMakeVisible(l);
    }
    heads_.setText("waypoint                                   travel     hold     radius", juce::dontSendNotification);
    for (juce::Component* c : { static_cast<juce::Component*>(&routes_), static_cast<juce::Component*>(&play_), static_cast<juce::Component*>(&stop_),
                                static_cast<juce::Component*>(&loop_), static_cast<juce::Component*>(&addCursor_), static_cast<juce::Component*>(&addPreset_),
                                static_cast<juce::Component*>(&clear_), static_cast<juce::Component*>(&save_), static_cast<juce::Component*>(&folder_) })
        addAndMakeVisible(c);
    refreshRoutes();
    rebuild();
    startTimerHz(4);
}

void RouteView::refreshRoutes()
{
    entries_ = proc_.routes();
    routes_.clear(juce::dontSendNotification);
    bool yours = false;
    routes_.addSectionHeading("Routes");
    for (size_t i = 0; i < entries_.size(); ++i) {
        if (entries_[i].builtin < 0 && !yours) { routes_.addSectionHeading("Yours"); yours = true; }
        routes_.addItem(entries_[i].name, static_cast<int>(i) + 1);
    }
}

void RouteView::rebuild()
{
    rows_.clear();
    const additiv::Route& r = proc_.route();
    for (int i = 0; i < r.count(); ++i) {
        auto row = std::make_unique<Row>();
        const additiv::Waypoint& w = r.point(i);
        row->name.setText(juce::String(i + 1) + "  " + (w.preset >= 0 ? juce::String(additiv::preset(w.preset).name)
                                                                        : juce::String(w.x, 2) + ", " + juce::String(w.y, 2)),
                          juce::dontSendNotification);
        row->name.setColour(juce::Label::textColourId, additivui::skin().ink);
        row->travel.setText(additiv::Journey::timeText(w.travel), false);
        row->hold.setText(additiv::Journey::timeText(w.hold), false);
        row->radius.setText(juce::String(w.radius, 3), false);
        row->travel.setTooltip("How long the sound takes to glide here from the point before (m:ss)");
        row->hold.setTooltip("How long it stays here (m:ss)");
        row->radius.setTooltip("How wide the blend is here (0.01 .. 0.3)");
        for (juce::TextEditor* t : { &row->travel, &row->hold, &row->radius }) {
            t->setJustification(juce::Justification::centred);
            t->onReturnKey = [this, i] { commit(i); };
            t->onFocusLost = [this, i] { commit(i); };
            addAndMakeVisible(t);
        }
        row->remove.onClick = [this, i] { proc_.removeWaypoint(i); };
        addAndMakeVisible(row->name);
        addAndMakeVisible(row->remove);
        rows_.push_back(std::move(row));
    }
    std::vector<char> buf(static_cast<size_t>(r.count()) * 200 + 64, 0);
    shown_ = r.count() > 0 && r.write(buf.data(), buf.size()) > 0 ? juce::String(buf.data()) : juce::String();
    resized();
    if (auto* p = getParentComponent()) p->resized();   // the page grows with the rows
}

void RouteView::commit(int i)
{
    if (i < 0 || i >= static_cast<int>(rows_.size()) || i >= proc_.route().count()) return;
    const Row& row = *rows_[static_cast<size_t>(i)];
    additiv::Waypoint w = proc_.route().point(i);
    double t = 0.0;
    if (additiv::Journey::parseTime(row.travel.getText().trim().toStdString(), t)) w.travel = static_cast<float>(std::max(0.0, t));
    if (additiv::Journey::parseTime(row.hold.getText().trim().toStdString(), t)) w.hold = static_cast<float>(std::max(0.0, t));
    w.radius = juce::jlimit(0.01f, 0.3f, row.radius.getText().getFloatValue() > 0.0f ? row.radius.getText().getFloatValue() : w.radius);
    proc_.setWaypoint(i, w);
}

void RouteView::timerCallback()
{
    // The box shows the route that is loaded (chosen here, from the state, or by a shot).
    for (size_t i = 0; i < entries_.size(); ++i)
        if (entries_[i].name == proc_.routeName() && routes_.getSelectedId() != static_cast<int>(i) + 1) {
            routes_.setSelectedId(static_cast<int>(i) + 1, juce::dontSendNotification);
            break;
        }
    const juce::String st = proc_.routeStatus();
    status_.setText(st.isEmpty() ? (proc_.route().count() > 0 ? juce::String("Not walking.") : juce::String("No route: choose one, or Shift-click the map."))
                                 : st,
                    juce::dontSendNotification);
    stop_.setEnabled(proc_.routeRunning());
    play_.setEnabled(proc_.route().count() > 0);
    // The rows anew when the route changed (a waypoint added, removed, a route chosen) and no field is being typed in.
    const additiv::Route& r = proc_.route();
    std::vector<char> buf(static_cast<size_t>(r.count()) * 200 + 64, 0);
    const juce::String now = r.count() > 0 && r.write(buf.data(), buf.size()) > 0 ? juce::String(buf.data()) : juce::String();
    bool typing = false;
    for (const auto& row : rows_) typing = typing || row->travel.hasKeyboardFocus(true) || row->hold.hasKeyboardFocus(true) || row->radius.hasKeyboardFocus(true);
    if (now != shown_ && !typing) rebuild();
    if (++tick_ % 20 == 0) {   // every five seconds: a route written by hand in the folder appears
        size_t files = 0;
        for (const auto& en : entries_) if (en.builtin < 0) ++files;
        if (AdditivProcessor::routeFolder().findChildFiles(juce::File::findFiles, false, "*.route").size() != static_cast<int>(files)) refreshRoutes();
    }
}

void RouteView::save()
{
    const juce::File dir = AdditivProcessor::routeFolder();
    dir.createDirectory();
    chooser_ = std::make_unique<juce::FileChooser>("Save the route", dir.getChildFile("My Route.route"), "*.route");
    chooser_->launchAsync(juce::FileBrowserComponent::saveMode | juce::FileBrowserComponent::warnAboutOverwriting, [this](const juce::FileChooser& fc) {
        const juce::File f = fc.getResult();
        if (f == juce::File()) return;
        if (proc_.saveRoute(f.withFileExtension("route"))) refreshRoutes();
        else juce::AlertWindow::showMessageBoxAsync(juce::MessageBoxIconType::WarningIcon, "Route", "The route could not be written there.");
    });
}

int RouteView::idealHeight() const { return 28 + 30 + 22 + static_cast<int>(rows_.size()) * 28 + 40 + 10; }

void RouteView::resized()
{
    const int w = getWidth();
    routes_.setBounds(12, 28, std::max(120, w - 12 - 140), 26);
    play_.setBounds(w - 124, 28, 28, 26);
    stop_.setBounds(w - 92, 28, 28, 26);
    loop_.setBounds(w - 60, 28, 56, 26);
    status_.setBounds(12, 58, w - 24, 20);
    heads_.setBounds(12, 80, w - 24, 18);
    int y = 100;
    const int nameW = std::max(80, w - 12 - 3 * 64 - 30 - 20);
    for (const auto& row : rows_) {
        row->name.setBounds(12, y, nameW, 24);
        row->travel.setBounds(12 + nameW + 4, y, 60, 24);
        row->hold.setBounds(12 + nameW + 68, y, 60, 24);
        row->radius.setBounds(12 + nameW + 132, y, 60, 24);
        row->remove.setBounds(12 + nameW + 196, y, 26, 24);
        y += 28;
    }
    y += 6;
    addCursor_.setBounds(12, y, 90, 26);
    addPreset_.setBounds(106, y, 90, 26);
    clear_.setBounds(200, y, 64, 26);
    save_.setBounds(270, y, 28, 26);
    folder_.setBounds(302, y, 28, 26);
}

void RouteView::paint(juce::Graphics& g)
{
    const frame::Skin& skin = additivui::skin();
    g.setColour(skin.group.withAlpha(0.92f));
    g.fillRoundedRectangle(getLocalBounds().toFloat(), skin.radius);
    g.setColour(skin.edge);
    g.drawRoundedRectangle(getLocalBounds().toFloat().reduced(0.5f), skin.radius, 1.0f);
    g.setColour(skin.family(frame::Family::Motion));
    g.setFont(juce::FontOptions(13.0f, juce::Font::bold));
    g.drawText("Route: waypoints, the time to glide to each and to stay", 16, 4, getWidth() - 24, 20, juce::Justification::centredLeft);
}

juce::String RouteView::describe() const
{
    juce::String s("Route: ");
    s << (proc_.routeName().isNotEmpty() ? proc_.routeName() : juce::String("(unnamed)")) << ", " << proc_.route().count() << " waypoints"
      << (proc_.routeLoop() ? ", looping" : "") << "\n";
    for (int i = 0; i < proc_.route().count(); ++i) {
        const additiv::Waypoint& w = proc_.route().point(i);
        s << "    " << (i + 1) << ". " << (w.preset >= 0 ? juce::String(additiv::preset(w.preset).name) : juce::String(w.x, 2) + ", " + juce::String(w.y, 2))
          << "  travel " << additiv::Journey::timeText(w.travel) << "  hold " << additiv::Journey::timeText(w.hold) << "  radius " << juce::String(w.radius, 3) << "\n";
    }
    return s;
}

// ---- the page

MapPage::MapPage(AdditivProcessor& proc, const frame::ControlActions& actions, frame::LiveRings* live)
    : params_(proc, actions,
              { { "Map", frame::Family::Motion, { ParamId::MapActive, ParamId::MapRadius, ParamId::MapGlide } },
                { "Route", frame::Family::Space, { ParamId::RouteSpeed } } },
              live),
      map_(proc), route_(proc)
{
    addAndMakeVisible(params_);
    addAndMakeVisible(map_);
    addAndMakeVisible(route_);
}

/** @brief The smallest side the map is given before the page scrolls. */
constexpr int kMapMin = 360;

/** @brief The map's largest side at a page width: a square, at most 600 pixels. @param w the page's width @return pixels */
static int mapSide(int w) { return std::min(600, (w - 30) * 55 / 100); }

/** @brief The right column's width beside a map. @param w the page's width @param side the map's side @return pixels */
static int rightWidth(int w, int side) { return w - 20 - side; }

int MapPage::heightFor(int w) const
{
    // Only the least it needs: the viewport makes the page as high as the window shows, and the
    // map takes that height (resized).
    const int right = params_.heightFor(rightWidth(w, mapSide(w))) + route_.idealHeight();
    return std::max(10 + std::min(kMapMin, mapSide(w)), right) + 10;
}

void MapPage::resized()
{
    // The map from the top and as large as the page's height allows, so that the whole square is
    // seen without scrolling; the knobs and the routes beside it.
    const int side = juce::jlimit(std::min(kMapMin, mapSide(getWidth())), mapSide(getWidth()), getHeight() - 20);
    map_.setBounds(10, 10, side, side);
    const int x = 10 + side, w = rightWidth(getWidth(), side);
    const int top = params_.heightFor(w);
    params_.setBounds(x, 0, w, top);
    route_.setBounds(x + 10, top, w - 20, std::max(route_.idealHeight(), getHeight() - 10 - top));
}

juce::String MapPage::describe() const
{
    return params_.describe() + route_.describe()
         + "\nThe map lays every preset on a square: across from darker to brighter, up from still to moving. With Map on, the "
           "sound is the blend of the presets round the cursor (Radius: how wide), and it follows the cursor over Map Glide. Click "
           "a preset to go there, drag to wander; Shift-click adds a waypoint. A route walks the cursor from waypoint to waypoint: "
           "each reached over its travel time -- the time the sound takes to turn into the next -- and held for its hold time, "
           "Route Speed faster or slower, round and round with Loop.\n";
}
