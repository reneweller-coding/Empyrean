/**
 * @file PageStrip.cpp
 * @brief The pages' strip (PageStrip.h).
 */
#include "PageStrip.h"

#include "Theme.h"

namespace {

constexpr int kHead = 12;      ///< the headings' row
constexpr int kGapIn = 2;      ///< between the pages of a section
constexpr int kGapOut = 22;    ///< between the sections
constexpr int kDot = 5;        ///< a dot's size
constexpr int kDotStep = 7;    ///< a dot's place

/** @brief The pages' names' font. @return it */
juce::Font nameFont() { return juce::Font(juce::FontOptions(14.0f)); }

} // namespace

void PageStrip::setPages(juce::StringArray names, std::vector<Section> sections)
{
    names_ = std::move(names);
    sections_ = std::move(sections);
    status_.assign(static_cast<size_t>(names_.size()), Status{});
    bounds_.assign(static_cast<size_t>(names_.size()), {});
    resized();
}

void PageStrip::setCurrent(int page)
{
    if (page == current_) return;
    current_ = page;
    repaint();
}

void PageStrip::refresh()
{
    if (!statusOf) return;
    bool changed = false;
    for (int p = 0; p < names_.size(); ++p) {
        Status s = statusOf(p);
        Status& was = status_[static_cast<size_t>(p)];
        bool same = s.badge == was.badge && s.quiet == was.quiet && s.dots.size() == was.dots.size();
        for (size_t k = 0; same && k < s.dots.size(); ++k) same = s.dots[k].on == was.dots[k].on;
        if (!same) changed = true;
        was = std::move(s);
    }
    if (changed) {
        resized();
        repaint();
    }
}

int PageStrip::widthOf(int page) const
{
    const Status& s = status_[static_cast<size_t>(page)];
    int w = juce::GlyphArrangement::getStringWidthInt(nameFont(), names_[page]) + 20;
    if (!s.dots.empty()) w += 4 + static_cast<int>(s.dots.size()) * kDotStep;
    if (s.badge.isNotEmpty()) w += 6 + juce::GlyphArrangement::getStringWidthInt(juce::Font(juce::FontOptions(10.5f, juce::Font::bold)), s.badge) + 8;
    return w;
}

int PageStrip::idealWidth() const
{
    int w = 0;
    for (const Section& sec : sections_) {
        for (int p : sec.pages) w += widthOf(p) + kGapIn;
        w += kGapOut;
    }
    return w;
}

void PageStrip::resized()
{
    heads_.clear();
    int x = 0;
    for (const Section& sec : sections_) {
        const int x0 = x;
        for (int p : sec.pages) {
            const int w = widthOf(p);
            bounds_[static_cast<size_t>(p)] = { x, kHead, w, getHeight() - kHead };
            x += w + kGapIn;
        }
        heads_.emplace_back(x0, 0, x - kGapIn - x0, kHead);
        x += kGapOut - kGapIn;
    }
}

void PageStrip::paint(juce::Graphics& g)
{
    const frame::Skin& skin = additivui::skin();
    g.setColour(skin.edge);
    g.fillRect(0, getHeight() - 1, getWidth(), 1);
    // The headings: small capitals over their pages, a hairline from the name to the section's end.
    for (size_t i = 0; i < sections_.size() && i < heads_.size(); ++i) {
        if (sections_[i].name.isEmpty()) continue;
        const auto h = heads_[i];
        const juce::Font f(juce::FontOptions(10.0f, juce::Font::bold));
        const juce::String text = sections_[i].name.toUpperCase();
        const int tw = juce::GlyphArrangement::getStringWidthInt(f, text);
        g.setColour(skin.dim.withAlpha(0.8f));
        g.setFont(f);
        g.drawText(text, h.getX() + 8, h.getY(), tw + 2, h.getHeight(), juce::Justification::centredLeft);
        g.setColour(skin.edge);
        g.fillRect(h.getX() + 8 + tw + 6, h.getCentreY(), std::max(0, h.getRight() - 8 - (h.getX() + 8 + tw + 6)), 1);
    }
    for (int p = 0; p < names_.size(); ++p) {
        const auto r = bounds_[static_cast<size_t>(p)];
        const Status& s = status_[static_cast<size_t>(p)];
        const bool front = p == current_, hover = p == hover_;
        if (front) {
            g.setColour(skin.panel);
            g.fillRect(r);
            g.setColour(skin.accent);
            g.fillRect(static_cast<float>(r.getX()) + 6.0f, static_cast<float>(r.getBottom()) - 2.5f, static_cast<float>(r.getWidth()) - 12.0f, 2.5f);
        } else if (hover) {
            g.setColour(skin.group.withAlpha(0.8f));
            g.fillRect(r);
        }
        juce::Colour text = front ? skin.accent : (hover ? skin.ink : skin.dim);
        if (s.quiet && !front) text = text.withAlpha(0.5f);
        g.setColour(text);
        g.setFont(nameFont());
        const int nameW = juce::GlyphArrangement::getStringWidthInt(nameFont(), names_[p]);
        int x = r.getX() + 10;
        g.drawText(names_[p], x, r.getY(), nameW + 2, r.getHeight() - 3, juce::Justification::centredLeft);
        x += nameW + 4;
        // The groups: lit while awake, a faint ring while resting.
        const float cy = static_cast<float>(r.getY()) + static_cast<float>(r.getHeight() - 3) * 0.5f;
        for (const Dot& d : s.dots) {
            const juce::Rectangle<float> dot(static_cast<float>(x) + 1.0f, cy - kDot * 0.5f, kDot, kDot);
            if (d.on) {
                g.setColour(d.colour);
                g.fillEllipse(dot);
            } else {
                g.setColour(skin.faint);
                g.drawEllipse(dot.reduced(0.5f), 1.0f);
            }
            x += kDotStep;
        }
        if (s.badge.isNotEmpty()) {
            const juce::Font f(juce::FontOptions(10.5f, juce::Font::bold));
            const int bw = juce::GlyphArrangement::getStringWidthInt(f, s.badge) + 8;
            const juce::Rectangle<float> b(static_cast<float>(x) + 4.0f, cy - 7.0f, static_cast<float>(bw), 14.0f);
            g.setColour(skin.accent.withAlpha(0.25f));
            g.fillRoundedRectangle(b, 7.0f);
            g.setColour(skin.ink);
            g.setFont(f);
            g.drawText(s.badge, b, juce::Justification::centred);
        }
    }
}

int PageStrip::pageAt(juce::Point<int> p) const
{
    for (size_t i = 0; i < bounds_.size(); ++i)
        if (bounds_[i].contains(p)) return static_cast<int>(i);
    return -1;
}

void PageStrip::mouseDown(const juce::MouseEvent& e)
{
    const int p = pageAt(e.getPosition());
    if (p >= 0 && onSelect) onSelect(p);
}

void PageStrip::mouseMove(const juce::MouseEvent& e)
{
    const int p = pageAt(e.getPosition());
    if (p != hover_) {
        hover_ = p;
        repaint();
    }
}

void PageStrip::mouseExit(const juce::MouseEvent&)
{
    hover_ = -1;
    repaint();
}

juce::String PageStrip::getTooltip()
{
    if (hover_ < 0 || static_cast<size_t>(hover_) >= status_.size()) return {};
    const Status& s = status_[static_cast<size_t>(hover_)];
    return names_[hover_] + (s.tip.isNotEmpty() ? ": " + s.tip : juce::String());
}
