/**
 * @file KnobMod.cpp
 * @brief Modulation at the knob (KnobMod.h).
 */
#include "KnobMod.h"

#include "Theme.h"

#include <cmath>

namespace knobmod {

using additiv::ModRow;
using additiv::ModSource;

juce::Colour sourceColour(ModSource s)
{
    const frame::Skin& skin = additivui::skin();
    if (s >= ModSource::Reservoir1) return additivui::colour::violet;
    if (s >= ModSource::Loudness) return skin.family(frame::Family::Space);
    if (s >= ModSource::Beat) return skin.family(frame::Family::Envelope);
    if (s >= ModSource::Lenia1) return skin.family(frame::Family::Motion);
    return skin.family(frame::Family::Source);
}

namespace {

/** @brief A section of the sources, as the matrix's menu has them. */
struct Section {
    const char* name;    ///< its heading
    ModSource first;     ///< its first source
    ModSource last;      ///< its last
};

/** @brief The sources by kind. */
constexpr Section kSections[] = {
    { "LFOs and envelopes", ModSource::Lfo1, ModSource::Env2 },
    { "Macros and the hands", ModSource::Macro1, ModSource::Random },
    { "Fields and chaos", ModSource::Lenia1, ModSource::Kura4 },
    { "The piece", ModSource::Beat, ModSource::RandomNote },
    { "What it hears", ModSource::Loudness, ModSource::Roughness },
    { "The reservoir", ModSource::Reservoir1, ModSource::Reservoir4 },
};

/** @brief A row's name: its source, and its via source when it has one. @param r the row @return the name */
juce::String rowName(const ModRow& r)
{
    juce::String s(additiv::kModSourceNames[static_cast<int>(r.source)]);
    if (r.via != ModSource::Off) s << " x " << additiv::kModSourceNames[static_cast<int>(r.via)];
    return s;
}

/** @brief One row on the knob in its menu: its source, its depth to drag, a button that removes it. */
class RowItem final : public juce::PopupMenu::CustomComponent {
public:
    /** @brief Row @p row of @p proc's matrix. @param proc the processor @param row 0 .. kModRows-1 */
    RowItem(AdditivProcessor& proc, int row) : juce::PopupMenu::CustomComponent(false), proc_(proc), row_(row)
    {
        const ModRow r = proc_.engine().modRow(row_);
        name_.setText(rowName(r), juce::dontSendNotification);
        name_.setColour(juce::Label::textColourId, sourceColour(r.source));
        name_.setFont(juce::FontOptions(13.0f));
        name_.setInterceptsMouseClicks(false, false);
        depth_.setSliderStyle(juce::Slider::LinearHorizontal);
        depth_.setTextBoxStyle(juce::Slider::TextBoxRight, false, 52, 18);
        depth_.setRange(-1.0, 1.0, 0.001);
        depth_.setValue(r.depth, juce::dontSendNotification);
        depth_.setDoubleClickReturnValue(true, 0.0);
        depth_.setColour(juce::Slider::trackColourId, sourceColour(r.source));
        depth_.setTooltip("How far it moves the knob, of the knob's whole range (negative: the other way)");
        depth_.onValueChange = [this] {
            ModRow now = proc_.engine().modRow(row_);
            now.depth = static_cast<float>(depth_.getValue());
            proc_.engine().setModRow(row_, now);
        };
        remove_.setTooltip("Removes this row from the matrix");
        remove_.onClick = [this] {
            proc_.engine().setModRow(row_, ModRow{});
            triggerMenuItem();   // the menu closes: the row is gone
        };
        for (juce::Component* c : { static_cast<juce::Component*>(&name_), static_cast<juce::Component*>(&depth_), static_cast<juce::Component*>(&remove_) })
            addAndMakeVisible(c);
    }
    /** @brief A row's size in the menu. @param w width @param h height */
    void getIdealSize(int& w, int& h) override
    {
        w = 340;
        h = 30;
    }
    /** @brief Name left, depth, the button right. */
    void resized() override
    {
        auto r = getLocalBounds().reduced(6, 3);
        remove_.setBounds(r.removeFromRight(24));
        name_.setBounds(r.removeFromLeft(120));
        depth_.setBounds(r.reduced(4, 0));
    }

private:
    AdditivProcessor& proc_;            ///< whose matrix
    int row_;                           ///< which row
    juce::Label name_;                  ///< its source
    juce::Slider depth_;                ///< its depth
    juce::TextButton remove_{ "x" };    ///< removes it
};

} // namespace

void addMenuItems(AdditivProcessor& proc, juce::PopupMenu& m, int id)
{
    if (id < 0 || id >= additiv::kNumParams || !additiv::isModTarget(static_cast<additiv::ParamId>(id))) return;
    m.addSeparator();
    m.addSectionHeader("Modulation");
    int free = -1;
    for (int i = 0; i < additiv::kModRows; ++i) {
        const ModRow r = proc.engine().modRow(i);
        if (r.source == ModSource::Off) {
            if (free < 0) free = i;
            continue;
        }
        if (r.target == id) m.addCustomItem(1000 + i, std::make_unique<RowItem>(proc, i), nullptr, rowName(r));
    }
    juce::PopupMenu with;
    for (const Section& s : kSections) {
        with.addSectionHeader(s.name);
        for (int k = static_cast<int>(s.first); k <= static_cast<int>(s.last); ++k) {
            const auto src = static_cast<ModSource>(k);
            with.addItem(additiv::kModSourceNames[k], free >= 0, false, [&proc, free, id, src] {
                ModRow r;
                r.source = src;
                r.target = id;
                r.depth = 0.2f;   // a fifth of the knob's range: heard, and easy to change in the menu or on the page
                proc.engine().setModRow(free, r);
            });
        }
    }
    m.addSubMenu(free >= 0 ? juce::String("Modulate with") : juce::String("Modulate with (all 16 rows taken)"), with, free >= 0);
}

bool paintArcs(juce::Graphics& g, AdditivProcessor& proc, juce::Slider& knob, int id)
{
    ModRow rows[additiv::kModRows];
    int n = 0;
    for (int i = 0; i < additiv::kModRows; ++i) {
        const ModRow r = proc.engine().modRow(i);
        if (r.source != ModSource::Off && r.target == id) rows[n++] = r;
    }
    if (n == 0) return false;
    // The knob's own geometry (Frame's drawRotarySlider): its ring, and outside it the live ring.
    const auto area = knob.getLookAndFeel().getSliderLayout(knob).sliderBounds.toFloat() + knob.getPosition().toFloat();
    const float size = std::min(area.getWidth(), area.getHeight());
    const auto c = area.getCentre();
    const float ring = size * 0.5f - 3.0f;
    const auto rp = knob.getRotaryParameters();
    const float a0 = rp.startAngleRadians, a1 = rp.endAngleRadians;
    const float pos = static_cast<float>(knob.valueToProportionOfLength(knob.getValue()));
    const float step = n > 3 ? 1.8f : 2.6f;
    for (int k = 0; k < n; ++k) {
        const ModRow& r = rows[k];
        const float d = r.depth;
        const bool uni = additiv::isUnipolarSource(r.source);
        const float lo = juce::jlimit(0.0f, 1.0f, uni ? std::min(pos, pos + d) : pos - std::abs(d));
        const float hi = juce::jlimit(0.0f, 1.0f, uni ? std::max(pos, pos + d) : pos + std::abs(d));
        const float via = r.via != ModSource::Off ? additiv::viaAmount(r.via, proc.engine().sourceValue(r.via)) : 1.0f;
        const float now = juce::jlimit(0.0f, 1.0f, pos + d * proc.engine().sourceValue(r.source) * via);
        const float rad = ring + 4.5f + step * static_cast<float>(k);
        const juce::Colour col = sourceColour(r.source);
        juce::Path span;
        span.addCentredArc(c.x, c.y, rad, rad, 0.0f, a0 + lo * (a1 - a0), a0 + std::max(hi, lo + 0.002f) * (a1 - a0), true);
        g.setColour(col.withAlpha(0.45f));
        g.strokePath(span, juce::PathStrokeType(1.4f, juce::PathStrokeType::curved, juce::PathStrokeType::rounded));
        const float an = a0 + now * (a1 - a0);
        g.setColour(col);
        g.fillEllipse(c.x + std::sin(an) * rad - 2.0f, c.y - std::cos(an) * rad - 2.0f, 4.0f, 4.0f);
    }
    return true;
}

} // namespace knobmod
