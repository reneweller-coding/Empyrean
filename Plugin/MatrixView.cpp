/**
 * @file MatrixView.cpp
 * @brief The modulation matrix on the panel (MatrixView.h).
 */
#include "MatrixView.h"

#include "Theme.h"

#include "additiv/Modulation.h"

using additiv::ModRow;
using additiv::ModSource;
using additiv::ParamId;

MatrixView::MatrixView(AdditivProcessor& proc) : proc_(proc)
{
    for (int i = 0; i < additiv::kNumParams; ++i)
        if (additiv::isModTarget(static_cast<ParamId>(i))) targets_.push_back(i);
    for (int i = 0; i < additiv::kModRows; ++i) {
        auto row = std::make_unique<Row>();
        for (int s = 0; s < additiv::kNumModSources; ++s) row->source.addItem(additiv::kModSourceNames[s], s + 1);
        row->target.addItem("-", 1);
        juce::String section;
        for (size_t k = 0; k < targets_.size(); ++k) {
            const additiv::ParamDesc& d = additiv::paramDesc(static_cast<ParamId>(targets_[k]));
            if (section != d.section) { section = d.section; row->target.addSectionHeading(section); }
            row->target.addItem(d.name, static_cast<int>(k) + 2);
        }
        row->depth.setSliderStyle(juce::Slider::LinearHorizontal);
        row->depth.setTextBoxStyle(juce::Slider::TextBoxRight, false, 52, 20);
        row->depth.setRange(-1.0, 1.0, 0.01);
        row->depth.setDoubleClickReturnValue(true, 0.0);
        row->depth.setColour(juce::Slider::trackColourId, additivui::skin().family(frame::Family::Motion));
        row->source.onChange = [this, i] { push(i); };
        row->target.onChange = [this, i] { push(i); };
        row->depth.onValueChange = [this, i] { push(i); };
        addAndMakeVisible(row->source);
        addAndMakeVisible(row->target);
        addAndMakeVisible(row->depth);
        rows_.push_back(std::move(row));
    }
    timerCallback();
    startTimerHz(4);
}

void MatrixView::push(int i)
{
    if (updating_) return;
    const Row& r = *rows_[static_cast<size_t>(i)];
    ModRow m;
    m.source = static_cast<ModSource>(std::max(r.source.getSelectedId() - 1, 0));
    const int t = r.target.getSelectedId() - 2;
    m.target = t >= 0 && t < static_cast<int>(targets_.size()) ? targets_[static_cast<size_t>(t)] : -1;
    m.depth = static_cast<float>(r.depth.getValue());
    proc_.engine().setModRow(i, m);
}

void MatrixView::timerCallback()
{
    updating_ = true;
    for (int i = 0; i < additiv::kModRows; ++i) {
        const ModRow m = proc_.engine().modRow(i);
        Row& r = *rows_[static_cast<size_t>(i)];
        r.source.setSelectedId(static_cast<int>(m.source) + 1, juce::dontSendNotification);
        int item = 1;
        for (size_t k = 0; k < targets_.size(); ++k)
            if (targets_[k] == m.target) item = static_cast<int>(k) + 2;
        r.target.setSelectedId(item, juce::dontSendNotification);
        if (!r.depth.isMouseButtonDown()) r.depth.setValue(m.depth, juce::dontSendNotification);
    }
    updating_ = false;
}

void MatrixView::resized()
{
    const int colW = (getWidth() - 30) / 2;
    for (int i = 0; i < additiv::kModRows; ++i) {
        Row& r = *rows_[static_cast<size_t>(i)];
        const int x = 10 + (i / 8) * (colW + 10) + 26, y = 28 + (i % 8) * 30;
        const int w = colW - 26;
        r.source.setBounds(x, y, w * 26 / 100, 24);
        r.target.setBounds(x + w * 27 / 100, y, w * 37 / 100, 24);
        r.depth.setBounds(x + w * 65 / 100, y, w * 35 / 100, 24);
    }
}

void MatrixView::paint(juce::Graphics& g)
{
    const frame::Skin& skin = additivui::skin();
    g.setColour(skin.group.withAlpha(0.92f));
    g.fillRoundedRectangle(getLocalBounds().toFloat(), skin.radius);
    g.setColour(skin.edge);
    g.drawRoundedRectangle(getLocalBounds().toFloat().reduced(0.5f), skin.radius, 1.0f);
    g.setColour(skin.family(frame::Family::Motion));
    g.setFont(juce::FontOptions(13.0f, juce::Font::bold));
    g.drawText("Matrix: source -> target x depth (of the target's range)", 18, 4, getWidth() - 30, 20, juce::Justification::centredLeft);
    g.setColour(skin.dim);
    g.setFont(juce::FontOptions(11.0f));
    const int colW = (getWidth() - 30) / 2;
    for (int i = 0; i < additiv::kModRows; ++i)
        g.drawText(juce::String(i + 1), 10 + (i / 8) * (colW + 10), 28 + (i % 8) * 30, 22, 24, juce::Justification::centredRight);
}

juce::String MatrixView::describe() const
{
    juce::String s("Matrix rows:\n");
    int n = 0;
    for (int i = 0; i < additiv::kModRows; ++i) {
        const ModRow m = proc_.engine().modRow(i);
        if (m.source == ModSource::Off || m.target < 0) continue;
        s << "    " << additiv::kModSourceNames[static_cast<int>(m.source)] << " -> " << additiv::paramDesc(static_cast<ParamId>(m.target)).name
          << " x " << juce::String(m.depth, 2) << "\n";
        ++n;
    }
    if (n == 0) s << "    (none)\n";
    s << "\nA source moves its target by depth times its value, in the target's own range: LFOs and the random curve swing "
         "both ways, envelopes follow their shapes, macros, the mod wheel, pressure and velocity go from 0 to 1. The envelopes "
         "start over when a note arrives into silence.\n";
    return s;
}
