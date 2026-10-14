/**
 * @file InfoLine.cpp
 * @brief The info line (InfoLine.h).
 */
#include "InfoLine.h"

#include "Frame.h"
#include "ParamInfo.h"
#include "Theme.h"

#include "additiv/Modulation.h"

InfoLine::InfoLine(AdditivProcessor& proc, juce::Component& watched) : proc_(proc), watched_(watched)
{
    tracker_.line = this;
    watched_.addMouseListener(&tracker_, true);
    setInterceptsMouseClicks(false, false);
    text_ = "Point at a control: what it does is told here.";
    startTimerHz(5);
}

InfoLine::~InfoLine() { watched_.removeMouseListener(&tracker_); }

juce::String InfoLine::describe(int id, juce::String& name) const
{
    const additiv::ParamDesc& d = additiv::paramDesc(static_cast<additiv::ParamId>(id));
    name = d.name;
    juce::String s = paramInfo(d.key);
    // Its value now.
    const float v = proc_.value(id);
    s << "    now ";
    if (d.kind == additiv::ParamKind::Choice) s << d.choices[juce::jlimit(0, d.numChoices - 1, juce::roundToInt(v))];
    else if (d.kind == additiv::ParamKind::Bool) s << (v >= 0.5f ? "on" : "off");
    else s << juce::String(v, std::abs(v) >= 100.0f ? 0 : std::abs(v) >= 10.0f ? 1 : 3) << (d.unit[0] != 0 ? juce::String(" ") + d.unit : juce::String());
    // What moves it.
    juce::String by;
    for (int i = 0; i < additiv::kModRows; ++i) {
        const additiv::ModRow r = proc_.engine().modRow(i);
        if (r.source == additiv::ModSource::Off || r.target != id) continue;
        by << (by.isEmpty() ? "" : ", ") << additiv::kModSourceNames[static_cast<int>(r.source)] << " (" << (r.depth >= 0.0f ? "+" : "") << juce::String(r.depth, 2) << ")";
    }
    if (by.isNotEmpty()) s << "    moved by " << by;
    return s;
}

void InfoLine::over(juce::Component* c)
{
    // The control under the mouse, or the nearest of its parents that is one.
    for (juce::Component* p = c; p != nullptr && p != &watched_; p = p->getParentComponent()) {
        int id = -1;
        if (auto* k = dynamic_cast<frame::Knob*>(p)) id = k->id();
        else if (auto* ch = dynamic_cast<frame::Choice*>(p)) id = ch->id();
        else if (auto* sw = dynamic_cast<frame::Switch*>(p)) id = sw->id();
        if (id >= 0 && id < additiv::kNumParams) {
            id_ = id;
            text_ = describe(id, name_);
            repaint();
            return;
        }
        if (auto* tip = dynamic_cast<juce::SettableTooltipClient*>(p)) {
            const juce::String t = tip->getTooltip();
            if (t.isNotEmpty()) {
                id_ = -1;
                name_.clear();
                text_ = t;
                repaint();
                return;
            }
        }
    }
}

void InfoLine::showParam(int id)
{
    if (id < 0 || id >= additiv::kNumParams) return;
    id_ = id;
    text_ = describe(id, name_);
    repaint();
}

void InfoLine::timerCallback()
{
    if (id_ < 0) return;
    juce::String name;
    const juce::String t = describe(id_, name);
    if (t != text_) {
        text_ = t;
        repaint();
    }
}

void InfoLine::paint(juce::Graphics& g)
{
    const frame::Skin& skin = additivui::skin();
    g.setColour(skin.group.withAlpha(0.85f));
    g.fillRect(getLocalBounds());
    g.setColour(skin.edge);
    g.drawHorizontalLine(0, 0.0f, static_cast<float>(getWidth()));
    juce::AttributedString a;
    if (name_.isNotEmpty()) a.append(name_ + "    ", juce::FontOptions(12.5f, juce::Font::bold), skin.accent);
    a.append(text_, juce::FontOptions(12.5f), skin.dim);
    a.setJustification(juce::Justification::centredLeft);
    a.setWordWrap(juce::AttributedString::none);
    a.draw(g, getLocalBounds().reduced(12, 0).toFloat());
}
