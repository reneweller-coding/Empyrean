/**
 * @file MapView.cpp
 * @brief The Map page (MapView.h).
 */
#include "MapView.h"

#include "Theme.h"

#include "additiv/Journey.h"
#include "additiv/Presets.h"

#include <algorithm>
#include <cmath>

using additiv::ParamId;

namespace {

/** @brief A section's colour: the sounds, the living systems, the sleep concerts, the visions, the swarms and currents, the halls and orbits. @param group the preset's group @return the colour */
juce::Colour sectionColour(int group)
{
    const frame::Skin& skin = additivui::skin();
    switch (group / 16) {
    case 0: return skin.family(frame::Family::Source);
    case 1: return additivui::colour::violet;
    case 2: return skin.family(frame::Family::Space);
    case 3: return skin.family(frame::Family::Envelope);
    case 4: return skin.family(frame::Family::Filter);
    default: return additivui::colour::root;
    }
}

} // namespace

// ---- the map

namespace {

constexpr float kMostZoom = 32.0f;    ///< the closest view: a thirty-second of the map across
constexpr size_t kMostSeen = 400;     ///< the presets' names stand at their points when no more than this many are in view
constexpr size_t kMostNames = 60;     ///< and no more names than this, the middle's first (the rest by the mouse)

} // namespace

MapView::MapView(AdditivProcessor& proc) : proc_(proc)
{
    whole_.setTooltip("Shows the whole map again");
    whole_.onClick = [this] { setView({ 0.0f, 0.0f, 1.0f, 1.0f }); };
    addChildComponent(whole_);
    startTimerHz(15);
}

void MapView::timerCallback()
{
    // The cursor (or the preset playing) walking out of the view by itself -- a route, a preset chosen: the view glides after it.
    const auto f = focus();
    const bool moved = f != lastFocus_;
    lastFocus_ = f;
    if (zoomed() && moved && focusInside_ && !inView(f.x, f.y) && !panning_ && dragPoint_ < 0) {
        const float w = view_.getWidth();
        target_ = clampView({ f.x - 0.5f * w, f.y - 0.5f * w, w, w });
        gliding_ = true;
    }
    if (gliding_) {   // a third of the way each tick
        const float dx = target_.getX() - view_.getX(), dy = target_.getY() - view_.getY();
        const bool there = std::abs(dx) + std::abs(dy) < 0.004f * view_.getWidth();
        setView(there ? target_ : view_.translated(0.35f * dx, 0.35f * dy), false);
        gliding_ = !there;
    }
    focusInside_ = inView(f.x, f.y);
    // The names written once the view has come to rest.
    if (!named_ && !panning_ && !gliding_ && juce::Time::getMillisecondCounter() - viewChanged_ > 180u) {
        drawPoints(true);
        named_ = true;
    }
    repaint();
}

juce::Rectangle<float> MapView::plane() const
{
    const auto b = getLocalBounds().toFloat().reduced(26.0f, 22.0f);
    const float s = std::min(b.getWidth(), b.getHeight());
    return { b.getX(), b.getY(), s, s };
}

juce::Rectangle<float> MapView::mini() const
{
    const auto p = plane();
    const float s = juce::jlimit(60.0f, 130.0f, p.getWidth() * 0.24f);
    return { p.getRight() - s - 8.0f, p.getY() + 8.0f, s, s };
}

bool MapView::zoomed() const { return view_.getWidth() < 0.999f; }

juce::Point<float> MapView::focus() const
{
    // With the map on (or no preset chosen), the cursor; else the preset playing.
    const int cp = proc_.currentPreset();
    if (proc_.value(static_cast<int>(ParamId::MapActive)) >= 0.5f || cp < 0)
        return { proc_.value(static_cast<int>(ParamId::MapX)), proc_.value(static_cast<int>(ParamId::MapY)) };
    float x = 0.0f, y = 0.0f;
    additiv::presetMapPosition(cp, x, y);
    return { x, y };
}

bool MapView::inView(float x, float y) const
{
    return x >= view_.getX() && x <= view_.getRight() && y >= view_.getY() && y <= view_.getBottom();
}

juce::Rectangle<float> MapView::clampView(juce::Rectangle<float> v)
{
    const float w = juce::jlimit(1.0f / kMostZoom, 1.0f, v.getWidth());
    return { juce::jlimit(0.0f, 1.0f - w, v.getX()), juce::jlimit(0.0f, 1.0f - w, v.getY()), w, w };
}

void MapView::setView(juce::Rectangle<float> v, bool stopGlide)
{
    if (stopGlide) gliding_ = false;
    v = clampView(v);
    if (v == view_) return;
    view_ = v;
    viewChanged_ = juce::Time::getMillisecondCounter();
    named_ = false;
    whole_.setVisible(zoomed());
    drawPoints(false);   // the names follow when the view rests (timerCallback)
    repaint();
}

void MapView::zoomAt(juce::Point<float> s, float factor)
{
    const auto p = plane();
    const float w = juce::jlimit(1.0f / kMostZoom, 1.0f, view_.getWidth() / factor);
    const float u = (s.x - p.getX()) / p.getWidth(), v = (p.getBottom() - s.y) / p.getHeight();
    const float mx = view_.getX() + u * view_.getWidth(), my = view_.getY() + v * view_.getHeight();
    setView({ mx - u * w, my - v * w, w, w });   // the place under the mouse stays under it
}

void MapView::centreOn(float x, float y)
{
    const float w = view_.getWidth();
    setView({ x - 0.5f * w, y - 0.5f * w, w, w });
}

juce::Point<float> MapView::toScreen(float x, float y) const
{
    const auto p = plane();
    const float u = (x - view_.getX()) / view_.getWidth(), v = (y - view_.getY()) / view_.getHeight();
    return { p.getX() + u * p.getWidth(), p.getBottom() - v * p.getHeight() };
}

juce::Point<float> MapView::toMap(juce::Point<float> s) const
{
    const auto p = plane();
    const float u = juce::jlimit(0.0f, 1.0f, (s.x - p.getX()) / p.getWidth()), v = juce::jlimit(0.0f, 1.0f, (p.getBottom() - s.y) / p.getHeight());
    return { view_.getX() + u * view_.getWidth(), view_.getY() + v * view_.getHeight() };
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
    const auto m = mini();
    whole_.setBounds(juce::Rectangle<float>(m.getX() - 3.0f, m.getBottom() + 8.0f, m.getWidth() + 6.0f, 22.0f).toNearestInt());
    drawOverview();
    drawPoints(true);
    named_ = true;
}

void MapView::drawPoints(bool names)
{
    const auto p = plane();
    if (p.getWidth() < 10.0f || getWidth() <= 0 || getHeight() <= 0) {
        points_ = {};
        return;
    }
    if (!points_.isValid() || points_.getWidth() != getWidth() || points_.getHeight() != getHeight())
        points_ = juce::Image(juce::Image::ARGB, getWidth(), getHeight(), true);
    else
        points_.clear(points_.getBounds());
    juce::Graphics g(points_);
    g.reduceClipRegion(p.getSmallestIntegerContainer());
    const float zoom = 1.0f / view_.getWidth();
    const float r = std::min(4.0f, 1.6f * std::pow(zoom, 0.35f));   // nearer, a little larger
    const auto seen = view_.expanded(8.0f * view_.getWidth() / p.getWidth());
    const auto mid = p.getCentre();
    std::vector<std::pair<float, int>> shown;   // (distance from the middle, preset), for the names
    for (int i = 0; i < additiv::numPresets(); ++i) {
        float x = 0.0f, y = 0.0f;
        additiv::presetMapPosition(i, x, y);
        if (!seen.contains(x, y)) continue;
        const auto s = toScreen(x, y);
        g.setColour(sectionColour(additiv::preset(i).group).withAlpha(0.75f));
        g.fillEllipse(s.x - r, s.y - r, 2.0f * r, 2.0f * r);
        if (names && p.contains(s)) shown.emplace_back(s.getDistanceSquaredFrom(mid), i);
    }
    if (shown.empty() || shown.size() > kMostSeen) return;   // too many for names: they would hide the points
    // The names where there is room for them, the middle's first.
    std::sort(shown.begin(), shown.end());
    const juce::Font font(juce::FontOptions(10.5f));
    g.setFont(font);
    g.setColour(additivui::skin().dim);
    std::vector<juce::Rectangle<float>> taken;
    for (const auto& [distance, i] : shown) {
        float x = 0.0f, y = 0.0f;
        additiv::presetMapPosition(i, x, y);
        const auto s = toScreen(x, y);
        const juce::String name(additiv::preset(i).name);
        const float w = static_cast<float>(juce::GlyphArrangement::getStringWidthInt(font, name)) + 4.0f;
        juce::Rectangle<float> box(s.x + r + 3.0f, s.y - 7.0f, w, 14.0f);
        if (box.getRight() > p.getRight()) box.setX(s.x - r - 3.0f - w);
        if (std::any_of(taken.begin(), taken.end(), [&box](const juce::Rectangle<float>& t) { return t.intersects(box.expanded(3.0f, 2.0f)); })) continue;
        taken.push_back(box);
        g.drawText(name, box, juce::Justification::centredLeft);
        if (taken.size() >= kMostNames) break;
    }
}

void MapView::drawOverview()
{
    const auto m = mini();
    overview_ = {};
    if (m.getWidth() < 20.0f) return;
    const int n = juce::roundToInt(m.getWidth());
    overview_ = juce::Image(juce::Image::ARGB, n, n, true);
    juce::Graphics g(overview_);
    for (int i = 0; i < additiv::numPresets(); ++i) {
        float x = 0.0f, y = 0.0f;
        additiv::presetMapPosition(i, x, y);
        g.setColour(sectionColour(additiv::preset(i).group).withAlpha(0.55f));
        g.fillRect(x * static_cast<float>(n - 1), (1.0f - y) * static_cast<float>(n - 1), 1.0f, 1.0f);
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
    // How near the view is, and how to move it.
    g.drawText(zoomed() ? juce::String(1.0f / view_.getWidth(), 1) + "x  -  right-drag moves" : juce::String("wheel zooms, right-drag moves"),
               juce::Rectangle<float>(p.getX() + 70.0f, p.getBottom() + 3.0f, p.getWidth() - 140.0f, 14.0f), juce::Justification::centred);
    g.saveState();
    g.addTransform(juce::AffineTransform::rotation(-juce::MathConstants<float>::halfPi, p.getX() - 8.0f, p.getBottom()));
    g.drawText("still", juce::Rectangle<float>(p.getX() - 8.0f, p.getBottom() - 14.0f, 60.0f, 14.0f), juce::Justification::centredLeft);
    g.drawText("moving", juce::Rectangle<float>(p.getX() - 8.0f + p.getHeight() - 60.0f, p.getBottom() - 14.0f, 60.0f, 14.0f), juce::Justification::centredRight);
    g.restoreState();

    const bool on = proc_.value(static_cast<int>(ParamId::MapActive)) >= 0.5f;
    const float cx = proc_.value(static_cast<int>(ParamId::MapX)), cy = proc_.value(static_cast<int>(ParamId::MapY));
    const float cr = proc_.value(static_cast<int>(ParamId::MapRadius));
    const auto c = toScreen(cx, cy);

    g.saveState();
    g.reduceClipRegion(p.getSmallestIntegerContainer());   // what lies beyond the view is not drawn

    // The presets the last search by example found (Focus page), ringed, numbered nearest first.
    {
        const auto& found = proc_.found();
        g.setFont(juce::FontOptions(10.0f, juce::Font::bold));
        for (size_t i = 0; i < found.size(); ++i) {
            float x = 0.0f, y = 0.0f;
            additiv::presetMapPosition(found[i], x, y);
            const auto s = toScreen(x, y);
            g.setColour(skin.accent.withAlpha(i == 0 ? 1.0f : 0.7f));
            g.drawEllipse(s.x - 6.0f, s.y - 6.0f, 12.0f, 12.0f, i == 0 ? 2.0f : 1.2f);
            g.drawText(juce::String(static_cast<int>(i) + 1), juce::Rectangle<float>(s.x + 6.0f, s.y - 14.0f, 14.0f, 12.0f), juce::Justification::centredLeft);
        }
    }

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
        const float rr = std::max(b.sigma * p.getWidth() / view_.getWidth(), 6.0f);   // the blend's width there (the radius in the local spacing)
        g.setColour(additivui::colour::accent.withAlpha(0.12f));
        g.fillEllipse(c.x - rr, c.y - rr, 2.0f * rr, 2.0f * rr);
        g.setColour(additivui::colour::accent);
        g.drawEllipse(c.x - rr, c.y - rr, 2.0f * rr, 2.0f * rr, 1.0f);
        g.fillEllipse(c.x - 4.0f, c.y - 4.0f, 8.0f, 8.0f);
    } else {
        g.setColour(skin.dim.withAlpha(0.6f));
        g.drawEllipse(c.x - 5.0f, c.y - 5.0f, 10.0f, 10.0f, 1.0f);
    }

    // The preset playing (without the map).
    const int cp = proc_.currentPreset();
    if (!on && cp >= 0) {
        float x = 0.0f, y = 0.0f;
        additiv::presetMapPosition(cp, x, y);
        const auto s = toScreen(x, y);
        g.setColour(skin.ink);
        g.drawEllipse(s.x - 5.0f, s.y - 5.0f, 10.0f, 10.0f, 1.5f);
    }
    g.restoreState();

    if (zoomed()) {
        // The cursor (or the preset playing) beyond the view: a ring at the edge, a line pointing to it.
        const auto f = focus();
        if (!inView(f.x, f.y)) {
            const auto in = p.reduced(7.0f);
            const auto fs = toScreen(f.x, f.y);
            const juce::Point<float> e(juce::jlimit(in.getX(), in.getRight(), fs.x), juce::jlimit(in.getY(), in.getBottom(), fs.y));
            const auto towards = fs - e;
            const float len = towards.getDistanceFromOrigin();
            g.setColour(on ? additivui::colour::accent : skin.ink);
            g.drawEllipse(e.x - 5.0f, e.y - 5.0f, 10.0f, 10.0f, 1.5f);
            if (len > 0.0f) g.drawLine(juce::Line<float>(e, e + towards * (11.0f / len)), 1.5f);
        }
        // The overview: the whole map small, the view on it as a frame, the cursor.
        const auto m = mini();
        g.setColour(skin.bg.withAlpha(0.92f));
        g.fillRoundedRectangle(m.expanded(3.0f), 4.0f);
        g.setColour(skin.edge);
        g.drawRoundedRectangle(m.expanded(3.0f), 4.0f, 1.0f);
        if (overview_.isValid()) g.drawImageAt(overview_, juce::roundToInt(m.getX()), juce::roundToInt(m.getY()));
        const juce::Rectangle<float> v(m.getX() + view_.getX() * m.getWidth(), m.getBottom() - view_.getBottom() * m.getHeight(),
                                       view_.getWidth() * m.getWidth(), view_.getHeight() * m.getHeight());
        g.setColour(skin.ink);
        g.drawRect(v.withSizeKeepingCentre(std::max(v.getWidth(), 4.0f), std::max(v.getHeight(), 4.0f)), 1.0f);
        g.setColour(on ? additivui::colour::accent : skin.ink);
        g.fillEllipse(m.getX() + f.x * m.getWidth() - 2.0f, m.getBottom() - f.y * m.getHeight() - 2.0f, 4.0f, 4.0f);
    }

    // The preset under the mouse, with its name.
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

int MapView::waypointAt(juce::Point<float> p) const
{
    const additiv::Route& r = proc_.route();
    for (int i = r.count() - 1; i >= 0; --i)
        if (toScreen(r.point(i).x, r.point(i).y).getDistanceFrom(p) <= 9.0f) return i;
    return -1;
}

void MapView::mouseDown(const juce::MouseEvent& e)
{
    gliding_ = false;
    if (zoomed() && mini().expanded(3.0f).contains(e.position)) {   // the overview: the view goes where it is clicked
        overviewDrag_ = true;
        mouseDrag(e);
        return;
    }
    if (!plane().expanded(4.0f).contains(e.position)) return;
    const int w = waypointAt(e.position);
    // A waypoint's right-click: removed.
    if (w >= 0 && e.mods.isPopupMenu()) {
        juce::PopupMenu m;
        m.addItem("Remove waypoint " + juce::String(w + 1), [this, w] {
            proc_.removeWaypoint(w);
            repaint();
        });
        m.showMenuAsync(juce::PopupMenu::Options().withTargetComponent(this).withTargetScreenArea(juce::Rectangle<int>(e.getScreenX(), e.getScreenY(), 1, 1)));
        return;
    }
    // The right or the middle button moves the view.
    if (e.mods.isPopupMenu() || e.mods.isMiddleButtonDown()) {
        panning_ = true;
        panFrom_ = view_;
        return;
    }
    // A waypoint: dragged where it should be.
    if (w >= 0 && !e.mods.isShiftDown()) {
        dragPoint_ = w;
        return;
    }
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
    if (overviewDrag_) {   // the view centred where the mouse is on the overview
        const auto m = mini();
        centreOn(juce::jlimit(0.0f, 1.0f, (e.position.x - m.getX()) / m.getWidth()), juce::jlimit(0.0f, 1.0f, (m.getBottom() - e.position.y) / m.getHeight()));
        return;
    }
    if (panning_) {   // the map held under the mouse
        const auto p = plane();
        const auto d = e.position - e.mouseDownPosition;
        setView(panFrom_.translated(-d.x / p.getWidth() * panFrom_.getWidth(), d.y / p.getHeight() * panFrom_.getHeight()));
        return;
    }
    if (dragPoint_ >= 0 && dragPoint_ < proc_.route().count()) {   // the waypoint follows; over a preset it snaps to it
        additiv::Waypoint w = proc_.route().point(dragPoint_);
        const int hit = presetAt(e.position, 7.0f);
        float x = 0.0f, y = 0.0f;
        if (hit >= 0) additiv::presetMapPosition(hit, x, y);
        else {
            const auto m = toMap(e.position);
            x = m.x;
            y = m.y;
        }
        w.x = std::clamp(x, 0.0f, 1.0f);
        w.y = std::clamp(y, 0.0f, 1.0f);
        w.preset = hit;
        proc_.setWaypoint(dragPoint_, w);
        repaint();
        return;
    }
    if (e.mods.isShiftDown()) return;
    const auto m = toMap(e.position);
    proc_.moveCursor(m.x, m.y);
}

void MapView::mouseUp(const juce::MouseEvent&)
{
    dragPoint_ = -1;
    panning_ = false;
    overviewDrag_ = false;
}

void MapView::mouseMove(const juce::MouseEvent& e)
{
    if (zoomed() && mini().expanded(3.0f).contains(e.position)) hover_ = -1;
    else hover_ = plane().expanded(4.0f).contains(e.position) ? presetAt(e.position, 7.0f) : -1;
}

void MapView::mouseExit(const juce::MouseEvent&) { hover_ = -1; }

void MapView::mouseWheelMove(const juce::MouseEvent& e, const juce::MouseWheelDetails& wheel)
{
    if (!plane().contains(e.position)) {
        juce::Component::mouseWheelMove(e, wheel);   // the page's, should it scroll
        return;
    }
    const float sign = wheel.isReversed ? -1.0f : 1.0f;
    if (wheel.deltaY != 0.0f) zoomAt(e.position, std::exp2(2.0f * juce::jlimit(-1.0f, 1.0f, sign * wheel.deltaY)));   // a notch: about 1.4 times
    else if (wheel.deltaX != 0.0f) setView(view_.translated(-0.5f * sign * wheel.deltaX * view_.getWidth(), 0.0f));
}

void MapView::mouseMagnify(const juce::MouseEvent& e, float scaleFactor)
{
    if (plane().contains(e.position) && scaleFactor > 0.0f) zoomAt(e.position, scaleFactor);
}

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
        row->radius.setTooltip("How far the blend reaches here, in the spacing of the presets round it (0.1 .. 4; under 0.5 a preset's own point plays it alone)");
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
    w.radius = juce::jlimit(0.1f, 4.0f, row.radius.getText().getFloatValue() > 0.0f ? row.radius.getText().getFloatValue() : w.radius);
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
              { { "Map", frame::Family::Motion, { ParamId::MapActive, ParamId::MapRadius, ParamId::MapGlide }, { ParamId::MapActive } },
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
           "sound is the blend of the presets round the cursor (Radius: how far, in the spacing of the presets there), and it follows the cursor over Map Glide. Click "
           "a preset to go there, drag to wander; Shift-click adds a waypoint, a waypoint dragged moves (over a preset it snaps to it), "
           "a right-click removes it. The wheel zooms round the mouse (up to 32 times), a right- or middle-drag moves the view; "
           "zoomed in, the names stand at the points, the overview in the corner shows where the view is (click it to go there), "
           "Whole map goes back, and the view follows the cursor (with the map off: the preset playing) when a route or a preset takes it out. A route walks the cursor from waypoint to waypoint: "
           "each reached over its travel time -- the time the sound takes to turn into the next -- and held for its hold time, "
           "Route Speed faster or slower, round and round with Loop.\n";
}
