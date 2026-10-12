/**
 * @file SourcesView.cpp
 * @brief The modulation sources, live (SourcesView.h).
 */
#include "SourcesView.h"

#include "Theme.h"

using additiv::ModSource;

namespace {

/** @brief A source's colour: by its kind, as the matrix's menu sections them. @param s the source @return the colour */
juce::Colour sourceColour(ModSource s)
{
    const frame::Skin& skin = additivui::skin();
    if (s >= ModSource::Reservoir1) return additivui::colour::violet;
    if (s >= ModSource::Loudness) return skin.family(frame::Family::Space);
    if (s >= ModSource::Beat) return skin.family(frame::Family::Envelope);
    if (s >= ModSource::Lenia1) return skin.family(frame::Family::Motion);
    return skin.family(frame::Family::Source);
}

} // namespace

SourcesView::SourcesView(AdditivProcessor& proc) : proc_(proc)
{
    for (auto& h : history_) h.fill(0.0f);
    proc_.engine().setSourcesWatched(true);
    startTimerHz(30);
}

SourcesView::~SourcesView()
{
    proc_.engine().setSourcesWatched(false);
}

int SourcesView::columns(int w) { return std::max(1, (w - 20 + kGap) / (kCellW + kGap)); }

int SourcesView::heightFor(int w)
{
    const int count = additiv::kNumModSources - 1, cols = columns(w);
    const int rows = (count + cols - 1) / cols;
    return kHead + rows * (kCellH + kGap) + 6;
}

void SourcesView::timerCallback()
{
    for (int s = 1; s < additiv::kNumModSources; ++s)
        history_[static_cast<size_t>(s)][static_cast<size_t>(head_)] = proc_.engine().sourceValue(static_cast<ModSource>(s));
    head_ = (head_ + 1) % kHistory;
    if (isShowing()) repaint();
}

void SourcesView::paint(juce::Graphics& g)
{
    const frame::Skin& skin = additivui::skin();
    g.setColour(skin.group.withAlpha(0.92f));
    g.fillRoundedRectangle(getLocalBounds().toFloat(), skin.radius);
    g.setColour(skin.edge);
    g.drawRoundedRectangle(getLocalBounds().toFloat().reduced(0.5f), skin.radius, 1.0f);
    g.setColour(skin.family(frame::Family::Motion));
    g.setFont(juce::FontOptions(13.0f, juce::Font::bold));
    g.drawText("Sources, live: the last four seconds of each", 18, 4, getWidth() - 30, 20, juce::Justification::centredLeft);

    const int cols = columns(getWidth());
    const float cellW = static_cast<float>(getWidth() - 20 - (cols - 1) * kGap) / static_cast<float>(cols);
    for (int s = 1; s < additiv::kNumModSources; ++s) {
        const int i = s - 1;
        const juce::Rectangle<float> cell(10.0f + static_cast<float>(i % cols) * (cellW + kGap), static_cast<float>(kHead + (i / cols) * (kCellH + kGap)),
                                          cellW, static_cast<float>(kCellH));
        g.setColour(skin.raised.withAlpha(0.55f));
        g.fillRoundedRectangle(cell, 4.0f);
        const ModSource src = static_cast<ModSource>(s);
        const bool uni = additiv::isUnipolarSource(src);
        const auto& h = history_[static_cast<size_t>(s)];
        const float now = h[static_cast<size_t>((head_ + kHistory - 1) % kHistory)];
        g.setFont(juce::FontOptions(10.5f));
        g.setColour(skin.dim);
        g.drawText(additiv::kModSourceNames[s], cell.reduced(6.0f, 2.0f).withHeight(14.0f), juce::Justification::centredLeft);
        g.drawText(juce::String(now, 2), cell.reduced(6.0f, 2.0f).withHeight(14.0f), juce::Justification::centredRight);
        // The trace: bipolar about the middle, unipolar from the floor.
        const juce::Rectangle<float> plot = cell.reduced(6.0f, 3.0f).withTrimmedTop(14.0f);
        auto yOf = [&](float v) {
            const float u = uni ? std::clamp(v, 0.0f, 1.0f) : 0.5f + 0.5f * std::clamp(v, -1.0f, 1.0f);
            return plot.getBottom() - u * plot.getHeight();
        };
        g.setColour(skin.faint);
        g.drawHorizontalLine(static_cast<int>(yOf(0.0f)), plot.getX(), plot.getRight());
        juce::Path p;
        for (int k = 0; k < kHistory; ++k) {
            const float v = h[static_cast<size_t>((head_ + k) % kHistory)];
            const float x = plot.getX() + plot.getWidth() * static_cast<float>(k) / static_cast<float>(kHistory - 1);
            if (k == 0) p.startNewSubPath(x, yOf(v));
            else p.lineTo(x, yOf(v));
        }
        g.setColour(sourceColour(src));
        g.strokePath(p, juce::PathStrokeType(1.4f));
    }
}
