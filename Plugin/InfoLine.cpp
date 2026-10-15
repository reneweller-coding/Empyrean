/**
 * @file InfoLine.cpp
 * @brief The info line (InfoLine.h).
 * @author Rene Weller
 */
#include "InfoLine.h"

#include "Frame.h"
#include "ParamInfo.h"
#include "Theme.h"

#include "additiv/Modulation.h"

InfoLine::InfoLine(AdditivProcessor& proc, juce::Component& watched) : m_proc(proc), m_watched(watched)
{
    m_tracker.m_line = this;
    m_watched.addMouseListener(&m_tracker, true);
    setInterceptsMouseClicks(false, false);
    m_text = "Point at a control: what it does is told here.";
    startTimerHz(5);
}

InfoLine::~InfoLine() { m_watched.removeMouseListener(&m_tracker); }

juce::String InfoLine::describe(int id, juce::String& name) const
{
    const additiv::ParamDesc& d = additiv::paramDesc(static_cast<additiv::ParamIdE>(id));
    name = d.m_name;
    juce::String s = paramInfo(d.m_key);
    // Its value now.
    const float v = m_proc.value(id);
    s << "    now ";
    if (d.m_kind == additiv::ParamKindE::CHOICE) s << d.m_choices[juce::jlimit(0, d.m_numChoices - 1, juce::roundToInt(v))];
    else if (d.m_kind == additiv::ParamKindE::BOOL) s << (v >= 0.5f ? "on" : "off");
    else s << juce::String(v, std::abs(v) >= 100.0f ? 0 : std::abs(v) >= 10.0f ? 1 : 3) << (d.m_unit[0] != 0 ? juce::String(" ") + d.m_unit : juce::String());
    // What moves it.
    juce::String by;
    for (int i = 0; i < additiv::ModRows; ++i) {
        const additiv::ModRow r = m_proc.engine().modRow(i);
        if (r.m_source == additiv::ModSourceE::OFF || r.m_target != id) continue;
        by << (by.isEmpty() ? "" : ", ") << additiv::ModSourceNames[static_cast<int>(r.m_source)] << " (" << (r.m_depth >= 0.0f ? "+" : "") << juce::String(r.m_depth, 2) << ")";
    }
    if (by.isNotEmpty()) s << "    moved by " << by;
    return s;
}

void InfoLine::over(juce::Component* c)
{
    // The control under the mouse, or the nearest of its parents that is one.
    for (juce::Component* p = c; p != nullptr && p != &m_watched; p = p->getParentComponent()) {
        int id = -1;
        if (auto* k = dynamic_cast<frame::Knob*>(p)) id = k->id();
        else if (auto* ch = dynamic_cast<frame::Choice*>(p)) id = ch->id();
        else if (auto* sw = dynamic_cast<frame::Switch*>(p)) id = sw->id();
        if (id >= 0 && id < additiv::NumParams) {
            m_id = id;
            m_text = describe(id, m_name);
            repaint();
            return;
        }
        if (auto* tip = dynamic_cast<juce::SettableTooltipClient*>(p)) {
            const juce::String t = tip->getTooltip();
            if (t.isNotEmpty()) {
                m_id = -1;
                m_name.clear();
                m_text = t;
                repaint();
                return;
            }
        }
    }
}

void InfoLine::showParam(int id)
{
    if (id < 0 || id >= additiv::NumParams) return;
    m_id = id;
    m_text = describe(id, m_name);
    repaint();
}

void InfoLine::timerCallback()
{
    if (m_id < 0) return;
    juce::String name;
    const juce::String t = describe(m_id, name);
    if (t != m_text) {
        m_text = t;
        repaint();
    }
}

void InfoLine::paint(juce::Graphics& g)
{
    const frame::Skin& skin = additivui::skin();
    g.setColour(skin.m_group.withAlpha(0.85f));
    g.fillRect(getLocalBounds());
    g.setColour(skin.m_edge);
    g.drawHorizontalLine(0, 0.0f, static_cast<float>(getWidth()));
    juce::AttributedString a;
    if (m_name.isNotEmpty()) a.append(m_name + "    ", juce::FontOptions(12.5f, juce::Font::bold), skin.m_accent);
    a.append(m_text, juce::FontOptions(12.5f), skin.m_dim);
    a.setJustification(juce::Justification::centredLeft);
    a.setWordWrap(juce::AttributedString::none);
    a.draw(g, getLocalBounds().reduced(12, 0).toFloat());
}
