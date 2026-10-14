/**
 * @file Tour.cpp
 * @brief The first steps (Tour.h).
 */
#include "Tour.h"

#include "Theme.h"

Tour::Tour()
{
    next_.onClick = [this] { next(); };
    skip_.onClick = [this] { finish(); };
    addAndMakeVisible(next_);
    addAndMakeVisible(skip_);
    setInterceptsMouseClicks(true, true);
}

void Tour::start(std::vector<Step> steps)
{
    steps_ = std::move(steps);
    at_ = 0;
    if (steps_.empty()) return;
    setVisible(true);
    toFront(true);
    resized();
    repaint();
}

void Tour::next()
{
    if (++at_ >= steps_.size()) { finish(); return; }
    resized();
    repaint();
}

void Tour::finish()
{
    setVisible(false);
    steps_.clear();
    if (onDone) onDone();
}

juce::Rectangle<int> Tour::hole() const
{
    if (at_ >= steps_.size() || steps_[at_].target == nullptr) return {};
    return getLocalArea(steps_[at_].target.getComponent(), steps_[at_].target->getLocalBounds()).expanded(6);
}

juce::Rectangle<int> Tour::bubble() const
{
    const auto h = hole();
    const int w = 420, ht = 150;
    int x = juce::jlimit(12, std::max(12, getWidth() - w - 12), h.getCentreX() - w / 2);
    int y = h.getBottom() + 14;
    if (y + ht > getHeight() - 12) y = std::max(12, h.getY() - ht - 14);   // above it when there is no room below
    return { x, y, w, ht };
}

void Tour::resized()
{
    const auto b = bubble();
    next_.setButtonText(at_ + 1 >= steps_.size() ? "Done" : "Next");
    next_.setBounds(b.getRight() - 92, b.getBottom() - 36, 80, 26);
    skip_.setBounds(b.getRight() - 180, b.getBottom() - 36, 80, 26);
    skip_.setVisible(at_ + 1 < steps_.size());
}

void Tour::paint(juce::Graphics& g)
{
    if (at_ >= steps_.size()) return;
    const frame::Skin& skin = additivui::skin();
    const auto h = hole();
    // The window dimmed, the control left bright.
    juce::Path dim;
    dim.addRectangle(getLocalBounds().toFloat());
    if (!h.isEmpty()) dim.addRoundedRectangle(h.toFloat(), 8.0f);
    dim.setUsingNonZeroWinding(false);
    g.setColour(skin.bg.withAlpha(0.72f));
    g.fillPath(dim);
    if (!h.isEmpty()) {
        g.setColour(skin.accent);
        g.drawRoundedRectangle(h.toFloat(), 8.0f, 2.0f);
    }
    const auto b = bubble().toFloat();
    g.setColour(skin.group);
    g.fillRoundedRectangle(b, 8.0f);
    g.setColour(skin.accent.withAlpha(0.7f));
    g.drawRoundedRectangle(b.reduced(0.5f), 8.0f, 1.2f);
    const Step& s = steps_[at_];
    g.setColour(skin.accent);
    g.setFont(juce::FontOptions(15.0f, juce::Font::bold));
    g.drawText(s.title, b.reduced(14.0f, 10.0f).withHeight(22.0f), juce::Justification::centredLeft);
    g.setColour(skin.dim);
    g.setFont(juce::FontOptions(11.0f));
    g.drawText(juce::String(static_cast<int>(at_) + 1) + " / " + juce::String(static_cast<int>(steps_.size())), b.reduced(14.0f, 10.0f).withHeight(22.0f),
               juce::Justification::centredRight);
    g.setColour(skin.ink);
    g.setFont(juce::FontOptions(13.0f));
    g.drawFittedText(s.text, b.reduced(14.0f, 10.0f).withTrimmedTop(26.0f).withTrimmedBottom(36.0f).toNearestInt(), juce::Justification::topLeft, 4);
}
