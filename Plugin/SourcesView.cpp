/**
 * @file SourcesView.cpp
 * @brief The modulation sources, live (SourcesView.h).
 * @author Rene Weller
 */
#include "SourcesView.h"

#include "KnobMod.h"
#include "Theme.h"

using additiv::ModSourceE;
using knobmod::sourceColour;

SourcesView::SourcesView(AdditivProcessor& proc, std::vector<ModSourceE> shown) : m_proc(proc), m_shown(std::move(shown))
{
    if (m_shown.empty())
        for (int s = 1; s < additiv::NumModSources; ++s) m_shown.push_back(static_cast<ModSourceE>(s));
    for (auto& h : m_history) h.fill(0.0f);
    m_proc.engine().setSourcesWatched(true);
    startTimerHz(30);
}

SourcesView::~SourcesView()
{
    m_proc.engine().setSourcesWatched(false);
}

int SourcesView::columns(int w) { return std::max(1, (w - 20 + Gap) / (CellW + Gap)); }

int SourcesView::mostColumns(int w) { return std::max(columns(w), (w - 20 + Gap) / (CellMinW + Gap)); }

int SourcesView::heightFor(int w, int count)
{
    const int cols = mostColumns(w), rows = (count + cols - 1) / cols;
    return Head + rows * (CellMinH + Gap) + 6;
}

std::pair<int, int> SourcesView::grid(int w, int h, int count)
{
    const int most = mostColumns(w);
    int cols = columns(w), cellH = CellH;
    for (;; ++cols) {
        const int rows = (count + cols - 1) / cols;
        cellH = (h - Head - 6) / rows - Gap;
        if (cellH >= 36 || cols >= most) break;
    }
    return { cols, std::clamp(cellH, CellMinH, CellMaxH) };
}

void SourcesView::mouseDown(const juce::MouseEvent& e)
{
    const auto [cols, cellH] = grid(getWidth(), getHeight(), static_cast<int>(m_shown.size()));
    const float cellW = static_cast<float>(getWidth() - 20 - (cols - 1) * Gap) / static_cast<float>(cols);
    const float x = static_cast<float>(e.x) - 10.0f, y = static_cast<float>(e.y - Head);
    if (x < 0.0f || y < 0.0f) return;
    const int col = static_cast<int>(x / (cellW + Gap)), row = y >= 0.0f ? static_cast<int>(y / static_cast<float>(cellH + Gap)) : -1;
    const int i = row * cols + col;
    if (col < cols && row >= 0 && i < static_cast<int>(m_shown.size()) && m_onChoose) m_onChoose(m_shown[static_cast<size_t>(i)]);
}

void SourcesView::timerCallback()
{
    for (int s = 1; s < additiv::NumModSources; ++s)
        m_history[static_cast<size_t>(s)][static_cast<size_t>(m_head)] = m_proc.engine().sourceValue(static_cast<ModSourceE>(s));
    m_head = (m_head + 1) % History;
    if (isShowing()) repaint();
}

void SourcesView::paint(juce::Graphics& g)
{
    const frame::Skin& skin = additivui::skin();
    g.setColour(skin.m_group.withAlpha(0.92f));
    g.fillRoundedRectangle(getLocalBounds().toFloat(), skin.m_radius);
    g.setColour(skin.m_edge);
    g.drawRoundedRectangle(getLocalBounds().toFloat().reduced(0.5f), skin.m_radius, 1.0f);
    g.setColour(skin.family(frame::FamilyE::MOTION));
    g.setFont(juce::FontOptions(13.0f, juce::Font::bold));
    g.drawText("Sources, live: the last four seconds of each", 18, 4, getWidth() - 30, 20, juce::Justification::centredLeft);

    const auto [cols, cellH] = grid(getWidth(), getHeight(), static_cast<int>(m_shown.size()));
    const float cellW = static_cast<float>(getWidth() - 20 - (cols - 1) * Gap) / static_cast<float>(cols);
    for (size_t n = 0; n < m_shown.size(); ++n) {
        const int i = static_cast<int>(n), s = static_cast<int>(m_shown[n]);
        const juce::Rectangle<float> cell(10.0f + static_cast<float>(i % cols) * (cellW + Gap), static_cast<float>(Head + (i / cols) * (cellH + Gap)),
                                          cellW, static_cast<float>(cellH));
        g.setColour(skin.m_raised.withAlpha(0.55f));
        g.fillRoundedRectangle(cell, 4.0f);
        if (m_shown[n] == m_chosen) {
            g.setColour(skin.m_accent);
            g.drawRoundedRectangle(cell.reduced(0.5f), 4.0f, 1.4f);
        }
        const ModSourceE src = static_cast<ModSourceE>(s);
        const bool uni = additiv::isUnipolarSource(src);
        const auto& h = m_history[static_cast<size_t>(s)];
        const float now = h[static_cast<size_t>((m_head + History - 1) % History)];
        g.setFont(juce::FontOptions(10.5f));
        g.setColour(skin.m_dim);
        g.drawText(additiv::ModSourceNames[s], cell.reduced(6.0f, 2.0f).withHeight(14.0f), juce::Justification::centredLeft);
        g.drawText(juce::String(now, 2), cell.reduced(6.0f, 2.0f).withHeight(14.0f), juce::Justification::centredRight);
        // The trace: bipolar about the middle, unipolar from the floor.
        const juce::Rectangle<float> plot = cell.reduced(6.0f, 3.0f).withTrimmedTop(14.0f);
        auto yOf = [&](float v) {
            const float u = uni ? std::clamp(v, 0.0f, 1.0f) : 0.5f + 0.5f * std::clamp(v, -1.0f, 1.0f);
            return plot.getBottom() - u * plot.getHeight();
        };
        g.setColour(skin.m_faint);
        g.drawHorizontalLine(static_cast<int>(yOf(0.0f)), plot.getX(), plot.getRight());
        juce::Path p;
        for (int k = 0; k < History; ++k) {
            const float v = h[static_cast<size_t>((m_head + k) % History)];
            const float x = plot.getX() + plot.getWidth() * static_cast<float>(k) / static_cast<float>(History - 1);
            if (k == 0) p.startNewSubPath(x, yOf(v));
            else p.lineTo(x, yOf(v));
        }
        g.setColour(sourceColour(src));
        g.strokePath(p, juce::PathStrokeType(1.4f));
    }
}
