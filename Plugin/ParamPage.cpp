/**
 * @file ParamPage.cpp
 * @brief A page of the panel (ParamPage.h).
 */
#include "ParamPage.h"

#include "Theme.h"

namespace {

constexpr int kKnobCell = 82;     ///< a knob's cell
constexpr int kChoiceCell = 132;  ///< a chooser's cell
constexpr int kTitle = 26;        ///< a box's title row
constexpr int kCellHeight = 100;  ///< a cell: name, control, value
constexpr int kGap = 10;          ///< between boxes
constexpr int kPad = 8;           ///< inside a box

} // namespace

ParamPage::ParamPage(AdditivProcessor& proc, const frame::ControlActions& actions, std::vector<Group> groups) : proc_(proc)
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
        g.setColour(b.colour);
        g.fillRoundedRectangle(r.getX() + kPad, r.getY() + 9.0f, 3.0f, 10.0f, 1.5f);
        g.setFont(juce::FontOptions(13.0f, juce::Font::bold));
        g.drawText(b.title, b.bounds.withHeight(kTitle).withTrimmedLeft(kPad + 9), juce::Justification::centredLeft);
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
