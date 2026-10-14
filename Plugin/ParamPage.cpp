/**
 * @file ParamPage.cpp
 * @brief A page of the panel (ParamPage.h).
 */
#include "ParamPage.h"

#include "KnobMod.h"
#include "Theme.h"

#include "additiv/Presets.h"

#include <algorithm>

namespace {

constexpr int kKnobCell = 82;     ///< a knob's cell
constexpr int kChoiceCell = 132;  ///< a chooser's cell
constexpr int kTitle = 26;        ///< a box's title row
constexpr int kCellHeight = 100;  ///< a cell: name, control, value
constexpr int kGap = 10;          ///< between boxes
constexpr int kPad = 8;           ///< inside a box

} // namespace

ParamPage::ParamPage(AdditivProcessor& proc, const frame::ControlActions& actions, std::vector<Group> groups, frame::LiveRings* live)
    : proc_(proc)
{
    const frame::Skin& skin = additivui::skin();
    for (const Group& g : groups) {
        Box box;
        box.title = g.title;
        box.colour = skin.family(g.family);
        for (additiv::ParamId pid : g.ids) {
            const int id = static_cast<int>(pid);
            juce::RangedAudioParameter* param = proc_.parameter(id);
            if (param == nullptr) continue;
            ids_.push_back(pid);
            const additiv::ParamDesc& d = additiv::paramDesc(pid);
            auto* label = labels_.add(new juce::Label({}, d.name));
            label->setJustificationType(juce::Justification::centred);
            label->setColour(juce::Label::textColourId, skin.dim);
            label->setFont(juce::FontOptions(12.0f));
            label->setMinimumHorizontalScale(0.7f);
            addAndMakeVisible(label);
            Cell cell;
            cell.label = label;
            cell.id = id;
            cell.gate = std::find(g.gate.begin(), g.gate.end(), pid) != g.gate.end();
            if (cell.gate) {   // a main control: its name in the group's colour
                label->setColour(juce::Label::textColourId, box.colour);
                label->setFont(juce::FontOptions(12.5f, juce::Font::bold));
            }
            if (d.kind == additiv::ParamKind::Choice) {
                auto* c = new frame::Choice(&actions, id);
                for (int i = 0; i < d.numChoices; ++i) c->addItem(d.choices[i], i + 1);
                c->setColour(juce::ComboBox::arrowColourId, box.colour);
                controls_.add(c);
                combos_.push_back(std::make_unique<juce::ComboBoxParameterAttachment>(*param, *c));
                cell.width = kChoiceCell;
            } else if (d.kind == additiv::ParamKind::Bool) {
                auto* b = new frame::Switch(&actions, id);
                b->setColour(juce::ToggleButton::tickColourId, box.colour);
                controls_.add(b);
                buttons_.push_back(std::make_unique<juce::ButtonParameterAttachment>(*param, *b));
                cell.width = kKnobCell;
            } else {
                auto* k = new frame::Knob(juce::Slider::RotaryHorizontalVerticalDrag, juce::Slider::TextBoxBelow, &actions, id);
                k->setTextBoxStyle(juce::Slider::TextBoxBelow, false, kKnobCell - 6, 16);
                k->setColour(juce::Slider::rotarySliderFillColourId, box.colour);
                k->setTextValueSuffix(d.unit[0] != 0 ? juce::String(" ") + d.unit : juce::String());
                k->setTooltip(juce::String(d.key) + ": " + d.name + (d.unit[0] != 0 ? juce::String(" (") + d.unit + ")" : juce::String()));
                controls_.add(k);
                sliders_.push_back(std::make_unique<juce::SliderParameterAttachment>(*param, *k));
                k->setDoubleClickReturnValue(true, d.def);
                if (live != nullptr) live->add(*k, id);   // where the matrix makes it play
                cell.width = kKnobCell;
            }
            addAndMakeVisible(controls_.getLast());
            cell.control = controls_.getLast();
            box.cells.push_back(cell);
        }
        if (box.cells.empty()) continue;
        box.width = 2 * kPad;
        for (const Cell& c : box.cells) box.width += c.width;
        boxes_.push_back(std::move(box));
    }
    startTimerHz(15);
}

void ParamPage::timerCallback()
{
    if (!isShowing()) return;
    for (Box& b : boxes_) {
        // Awake while a gate is up, or a matrix row moves one (it lifts it from nothing).
        bool any = false, awake = false;
        for (const Cell& c : b.cells)
            if (c.gate) {
                any = true;
                awake = awake || std::abs(proc_.value(c.id)) > 1.0e-6f;
                for (int i = 0; i < additiv::kModRows && !awake; ++i) {
                    const additiv::ModRow r = proc_.engine().modRow(i);
                    awake = r.source != additiv::ModSource::Off && r.target == c.id && r.depth != 0.0f;
                }
            }
        const bool resting = any && !awake;
        if (resting != b.resting) {
            b.resting = resting;
            for (Cell& c : b.cells)
                if (!c.gate) {
                    c.control->setAlpha(resting ? 0.4f : 1.0f);
                    c.label->setAlpha(resting ? 0.5f : 1.0f);
                }
            repaint(b.bounds);
        }
        // The knobs the matrix moves (and those it moved a moment ago, to clear their arcs); the
        // preset's marks with them (the timer repaints every knob that has moved from its preset).
        for (Cell& c : b.cells) {
            auto* k = dynamic_cast<juce::Slider*>(c.control);
            if (k == nullptr) continue;
            bool on = false;
            for (int i = 0; i < additiv::kModRows && !on; ++i) {
                const additiv::ModRow r = proc_.engine().modRow(i);
                on = r.source != additiv::ModSource::Off && r.target == c.id;
            }
            const auto& ref = proc_.reference();
            const bool away = ref.size() == static_cast<size_t>(additiv::kNumParams) && ref[static_cast<size_t>(c.id)] != static_cast<float>(k->getValue());
            if (on || c.moved || away != c.away) repaint(k->getBounds().expanded(12));
            c.moved = on;
            c.away = away;
        }
    }
}

void ParamPage::paintOverChildren(juce::Graphics& g)
{
    const auto& ref = proc_.reference();
    const bool marks = ref.size() == static_cast<size_t>(additiv::kNumParams);
    for (const Box& b : boxes_)
        for (const Cell& c : b.cells)
            if (auto* k = dynamic_cast<juce::Slider*>(c.control)) {
                knobmod::paintArcs(g, proc_, *k, c.id);
                // Where the preset had it, when it has moved: a small notch outside the ring.
                if (!marks || additiv::keptByPresets(static_cast<additiv::ParamId>(c.id))) continue;
                const float was = ref[static_cast<size_t>(c.id)];
                if (std::fabs(was - static_cast<float>(k->getValue())) <= 1.0e-5f * static_cast<float>(std::max(1.0, k->getMaximum() - k->getMinimum()))) continue;
                const auto area = k->getLookAndFeel().getSliderLayout(*k).sliderBounds.toFloat() + k->getPosition().toFloat();
                const float size = std::min(area.getWidth(), area.getHeight()), ring = size * 0.5f - 3.0f;
                const auto rp = k->getRotaryParameters();
                const float a = rp.startAngleRadians + static_cast<float>(k->valueToProportionOfLength(was)) * (rp.endAngleRadians - rp.startAngleRadians);
                const auto c0 = area.getCentre();
                const float r0 = ring + 1.0f, r1 = ring + 6.0f;
                g.setColour(additivui::skin().ink.withAlpha(0.75f));
                g.drawLine(c0.x + std::sin(a) * r0, c0.y - std::cos(a) * r0, c0.x + std::sin(a) * r1, c0.y - std::cos(a) * r1, 1.6f);
            }
}

std::vector<juce::Rectangle<int>> ParamPage::flow(int width) const
{
    std::vector<juce::Rectangle<int>> out;
    int x = kGap, y = kGap;
    const int h = kTitle + kCellHeight + kPad;
    for (const Box& b : boxes_) {
        const int w = std::min(b.width, std::max(width - 2 * kGap, kKnobCell + 2 * kPad));
        if (x > kGap && x + w > width - kGap) { x = kGap; y += h + kGap; }
        out.emplace_back(x, y, w, h);
        x += w + kGap;
    }
    return out;
}

int ParamPage::heightFor(int width) const
{
    const auto r = flow(width);
    return r.empty() ? 0 : r.back().getBottom() + kGap;
}

void ParamPage::resized()
{
    const auto r = flow(getWidth());
    for (size_t i = 0; i < boxes_.size(); ++i) {
        Box& b = boxes_[i];
        b.bounds = r[i];
        int x = b.bounds.getX() + kPad;
        const int top = b.bounds.getY() + kTitle;
        for (Cell& c : b.cells) {
            c.label->setBounds(x, top, c.width, 16);
            if (dynamic_cast<juce::ComboBox*>(c.control) != nullptr) c.control->setBounds(x + 6, top + 34, c.width - 12, 26);
            else if (dynamic_cast<juce::ToggleButton*>(c.control) != nullptr) c.control->setBounds(x + 14, top + 34, c.width - 28, 26);
            else c.control->setBounds(x, top + 16, c.width, kCellHeight - 18);
            x += c.width;
        }
    }
}

void ParamPage::paint(juce::Graphics& g)
{
    const frame::Skin& skin = additivui::skin();
    for (const Box& b : boxes_) {
        const auto r = b.bounds.toFloat();
        g.setColour(skin.group.withAlpha(0.92f));
        g.fillRoundedRectangle(r, skin.radius);
        g.setColour(skin.edge);
        g.drawRoundedRectangle(r.reduced(0.5f), skin.radius, 1.0f);
        g.setColour(b.colour.withAlpha(b.resting ? 0.5f : 1.0f));
        g.fillRoundedRectangle(r.getX() + kPad, r.getY() + 9.0f, 3.0f, 10.0f, 1.5f);
        g.setFont(juce::FontOptions(13.0f, juce::Font::bold));
        g.drawText(b.resting ? b.title + juce::String("  -  off") : b.title, b.bounds.withHeight(kTitle).withTrimmedLeft(kPad + 9),
                   juce::Justification::centredLeft);
        // The main controls: a soft halo of the group's colour behind them.
        for (const Cell& c : b.cells) {
            if (!c.gate || c.control == nullptr) continue;
            const bool knob = dynamic_cast<juce::Slider*>(c.control) != nullptr;
            const auto cb = c.control->getBounds().toFloat();
            const float d = std::min(cb.getWidth(), cb.getHeight() - (knob ? 16.0f : 0.0f));
            const auto centre = knob ? juce::Point<float>(cb.getCentreX(), cb.getY() + d * 0.5f) : cb.getCentre();
            g.setGradientFill(juce::ColourGradient(b.colour.withAlpha(0.16f), centre.x, centre.y, b.colour.withAlpha(0.0f), centre.x + d * 0.62f,
                                                   centre.y, true));
            g.fillEllipse(centre.x - d * 0.62f, centre.y - d * 0.62f, d * 1.24f, d * 1.24f);
        }
    }
}

juce::String ParamPage::describe() const
{
    juce::String s;
    for (const Box& b : boxes_) {
        s << b.title << "\n";
        for (const Cell& c : b.cells) {
            const int id = c.control != nullptr ? (dynamic_cast<frame::Knob*>(c.control) != nullptr ? dynamic_cast<frame::Knob*>(c.control)->id()
                                                  : dynamic_cast<frame::Choice*>(c.control) != nullptr ? dynamic_cast<frame::Choice*>(c.control)->id()
                                                  : dynamic_cast<frame::Switch*>(c.control) != nullptr ? dynamic_cast<frame::Switch*>(c.control)->id() : -1) : -1;
            if (id < 0) continue;
            const additiv::ParamDesc& d = additiv::paramDesc(static_cast<additiv::ParamId>(id));
            s << "    " << d.name << " (" << d.key << "): ";
            if (d.kind == additiv::ParamKind::Choice) {
                for (int i = 0; i < d.numChoices; ++i) s << (i > 0 ? ", " : "") << d.choices[i];
            } else if (d.kind == additiv::ParamKind::Bool) {
                s << "on or off";
            } else {
                s << juce::String(d.min) << " .. " << juce::String(d.max) << (d.unit[0] != 0 ? juce::String(" ") + d.unit : juce::String());
            }
            s << ", now " << juce::String(proc_.value(id), 2) << "\n";
        }
        s << "\n";
    }
    return s;
}
