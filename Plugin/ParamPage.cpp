/**
 * @file ParamPage.cpp
 * @brief A page of the panel (ParamPage.h).
 */
#include "ParamPage.h"

#include "KnobMod.h"
#include "Theme.h"

#include "additiv/Presets.h"

#include <algorithm>
#include <functional>

namespace {

constexpr int kKnobCell = 82;     ///< a knob's cell
constexpr int kChoiceCell = 132;  ///< a chooser's cell
constexpr int kTitle = 26;        ///< a box's title row
constexpr int kCellHeight = 100;  ///< a cell: name, control, value
constexpr int kGap = 10;          ///< between boxes
constexpr int kPad = 8;           ///< inside a box

/**
 * @brief A span: two parameters, its low and its high end, as one bar with two ends to drag. Bound to the host's
 *        parameters with gestures (an undo step each drag); a right-click offers the nearer end's menu.
 */
class SpanControl final : public juce::Component, public juce::SettableTooltipClient {
public:
    /**
     * @brief The span of @p low and @p high.
     * @param proc     the processor
     * @param actions  the right-click menu
     * @param low      the low end's parameter
     * @param high     the high end's
     * @param colour   its group's colour
     */
    SpanControl(AdditivProcessor& proc, const frame::ControlActions& actions, int low, int high, juce::Colour colour)
        : proc_(proc), actions_(actions), id_{ low, high }, colour_(colour)
    {
        for (int e = 0; e < 2; ++e)
            if (auto* p = proc_.parameter(id_[e]))
                attach_[e] = std::make_unique<juce::ParameterAttachment>(*p, [this](float) { repaint(); });
        for (auto& a : attach_)
            if (a != nullptr) a->sendInitialUpdate();
    }
    /** @brief The track, the span lit, the two ends, their values. @param g where */
    void paint(juce::Graphics& g) override
    {
        const frame::Skin& skin = additivui::skin();
        const auto t = track();
        g.setColour(skin.raised);
        g.fillRoundedRectangle(t.withSizeKeepingCentre(t.getWidth(), 4.0f), 2.0f);
        const float x0 = t.getX() + t.getWidth() * norm(0), x1 = t.getX() + t.getWidth() * norm(1);
        g.setColour(colour_);
        g.fillRoundedRectangle(juce::Rectangle<float>::leftTopRightBottom(std::min(x0, x1), t.getCentreY() - 2.0f, std::max(x0, x1), t.getCentreY() + 2.0f), 2.0f);
        for (int e = 0; e < 2; ++e) {
            const float x = e == 0 ? x0 : x1;
            g.setColour(e == held_ ? skin.ink : colour_.brighter(0.3f));
            g.fillEllipse(x - 6.0f, t.getCentreY() - 6.0f, 12.0f, 12.0f);
            g.setColour(skin.bg);
            g.fillEllipse(x - 2.5f, t.getCentreY() - 2.5f, 5.0f, 5.0f);
        }
        g.setColour(skin.dim);
        g.setFont(juce::FontOptions(11.5f));
        g.drawText(text(0) + "  -  " + text(1), getLocalBounds().removeFromBottom(18), juce::Justification::centred);
    }
    /** @brief The nearer end is taken (a right-click: its menu). @param e the press */
    void mouseDown(const juce::MouseEvent& e) override
    {
        const auto t = track();
        const float x0 = t.getX() + t.getWidth() * norm(0), x1 = t.getX() + t.getWidth() * norm(1);
        const int near = std::fabs(static_cast<float>(e.x) - x0) <= std::fabs(static_cast<float>(e.x) - x1) ? 0 : 1;
        if (e.mods.isPopupMenu()) {
            actions_.showMenu(*this, id_[near]);
            return;
        }
        held_ = near;
        if (attach_[held_] != nullptr) attach_[held_]->beginGesture();
        mouseDrag(e);
    }
    /** @brief The end held follows, never past the other. @param e the drag */
    void mouseDrag(const juce::MouseEvent& e) override
    {
        if (held_ < 0 || attach_[held_] == nullptr) return;
        const auto t = track();
        float n = std::clamp((static_cast<float>(e.x) - t.getX()) / t.getWidth(), 0.0f, 1.0f);
        n = held_ == 0 ? std::min(n, norm(1)) : std::max(n, norm(0));
        auto* p = proc_.parameter(id_[held_]);
        attach_[held_]->setValueAsPartOfGesture(p->convertFrom0to1(n));
        repaint();
    }
    /** @brief Let go. @param e the release */
    void mouseUp(const juce::MouseEvent&) override
    {
        if (held_ >= 0 && attach_[held_] != nullptr) attach_[held_]->endGesture();
        held_ = -1;
        repaint();
    }
    /** @brief Both ends, as the info line shows them. @return the text */
    juce::String getTooltip() override
    {
        return juce::String(additiv::paramDesc(static_cast<additiv::ParamId>(id_[0])).name) + " and " + additiv::paramDesc(static_cast<additiv::ParamId>(id_[1])).name
               + ": " + text(0) + " to " + text(1) + " -- drag either end (right-click: its menu)";
    }

private:
    /** @brief The track's line. @return pixels */
    juce::Rectangle<float> track() const { return getLocalBounds().toFloat().reduced(10.0f, 0.0f).withTrimmedBottom(20.0f); }
    /** @brief An end where it stands, 0 .. 1 along the track (its parameter's own skew). @param e 0 low, 1 high @return 0 .. 1 */
    float norm(int e) const
    {
        auto* p = proc_.parameter(id_[e]);
        return p != nullptr ? p->getValue() : 0.0f;
    }
    /** @brief An end's value in words: a key's name for a note, else the value and its unit. @param e 0 low, 1 high @return the text */
    juce::String text(int e) const
    {
        const additiv::ParamDesc& d = additiv::paramDesc(static_cast<additiv::ParamId>(id_[e]));
        const float v = proc_.value(id_[e]);
        if (d.id == additiv::ParamId::BrainLow || d.id == additiv::ParamId::BrainHigh) return juce::MidiMessage::getMidiNoteName(juce::roundToInt(v), true, true, 4);
        return juce::String(v, v >= 100.0f ? 0 : 1) + (d.unit[0] != 0 ? juce::String(" ") + d.unit : juce::String());
    }

    AdditivProcessor& proc_;                                ///< whose parameters
    const frame::ControlActions& actions_;                  ///< the right-click menu
    int id_[2];                                             ///< low and high
    juce::Colour colour_;                                   ///< its group's
    std::unique_ptr<juce::ParameterAttachment> attach_[2];  ///< to the parameters
    int held_ = -1;                                         ///< the end being dragged, -1 none
};

/** @brief A small chain between two knobs: lit, they turn together in proportion. */
class ChainButton final : public juce::Button {
public:
    /** @brief A chain, the two unlinked. */
    ChainButton() : juce::Button("link")
    {
        setClickingTogglesState(true);
        setTooltip("Link the two: turning one turns the other in proportion");
    }
    /** @brief Two rings of a chain, lit when linked. @param g where @param over under the mouse @param down pressed */
    void paintButton(juce::Graphics& g, bool over, bool down) override
    {
        (void)down;
        const frame::Skin& skin = additivui::skin();
        const auto r = getLocalBounds().toFloat().reduced(2.0f);
        g.setColour(getToggleState() ? skin.accent : (over ? skin.ink : skin.faint));
        const float w = r.getWidth() * 0.62f, h = r.getHeight() * 0.42f;
        g.drawRoundedRectangle(r.getX(), r.getCentreY() - h * 0.5f, w, h, h * 0.5f, 1.6f);
        g.drawRoundedRectangle(r.getRight() - w, r.getCentreY() - h * 0.5f, w, h, h * 0.5f, 1.6f);
    }
};

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
            if (std::any_of(g.spans.begin(), g.spans.end(), [pid](const auto& s) { return s.second == pid; })) continue;   // in its low end's bar
            if (auto span = std::find_if(g.spans.begin(), g.spans.end(), [pid](const auto& s) { return s.first == pid; }); span != g.spans.end()) {
                const additiv::ParamDesc& lo = additiv::paramDesc(span->first), &hi = additiv::paramDesc(span->second);
                auto* label = labels_.add(new juce::Label({}, juce::String(lo.name) + " - " + hi.name));
                label->setJustificationType(juce::Justification::centred);
                label->setColour(juce::Label::textColourId, skin.dim);
                label->setFont(juce::FontOptions(12.0f));
                addAndMakeVisible(label);
                auto* c = new SpanControl(proc_, actions, id, static_cast<int>(span->second), box.colour);
                controls_.add(c);
                addAndMakeVisible(c);
                Cell cell;
                cell.label = label;
                cell.id = id;
                cell.id2 = static_cast<int>(span->second);
                cell.control = c;
                cell.width = 2 * kKnobCell;
                box.cells.push_back(cell);
                continue;
            }
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
        // Linked pairs: a chain between their knobs.
        for (const auto& [ia, ib] : g.links) {
            juce::Slider* a = nullptr;
            juce::Slider* b = nullptr;
            for (const Cell& c : box.cells) {
                if (c.id == static_cast<int>(ia)) a = dynamic_cast<juce::Slider*>(c.control);
                if (c.id == static_cast<int>(ib)) b = dynamic_cast<juce::Slider*>(c.control);
            }
            if (a == nullptr || b == nullptr) continue;
            auto link = std::make_unique<Link>();
            link->a = a;
            link->b = b;
            link->chain = std::make_unique<ChainButton>();
            addAndMakeVisible(*link->chain);
            Link* l = link.get();
            auto follow = [l](juce::Slider* moved, juce::Slider* other, bool forward) {
                if (!l->chain->getToggleState() || l->busy) return;
                l->busy = true;
                const double v = forward ? moved->getValue() * l->ratio : moved->getValue() / std::max(l->ratio, 1.0e-6);
                other->setValue(juce::jlimit(other->getMinimum(), other->getMaximum(), v), juce::sendNotificationSync);
                l->busy = false;
            };
            auto take = [l] { l->ratio = l->a->getValue() > 1.0e-9 ? l->b->getValue() / l->a->getValue() : 1.0; };
            a->onDragStart = take;
            b->onDragStart = take;
            a->onValueChange = [l, follow] { follow(l->a, l->b, true); };
            b->onValueChange = [l, follow] { follow(l->b, l->a, false); };
            l->chain->onClick = take;
            links_.push_back(std::move(link));
        }
        boxes_.push_back(std::move(box));
    }
    startTimerHz(15);
}

void ParamPage::timerCallback()
{
    if (!isShowing()) return;
    if (flashId_ >= 0) {   // the found control's ring fades
        if (juce::Component* c = controlFor(flashId_)) repaint(c->getBounds().expanded(8));
        if (juce::Time::getMillisecondCounterHiRes() > flashUntil_) flashId_ = -1;
    }
    for (Box& b : boxes_) {
        const bool resting = gated(b) && !awake(b);
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

bool ParamPage::gated(const Box& b)
{
    for (const Cell& c : b.cells)
        if (c.gate) return true;
    return false;
}

bool ParamPage::awake(const Box& b) const
{
    // Awake while a gate is up, or a matrix row moves one (it lifts it from nothing).
    for (const Cell& c : b.cells) {
        if (!c.gate) continue;
        if (std::abs(proc_.value(c.id)) > 1.0e-6f) return true;
        for (int i = 0; i < additiv::kModRows; ++i) {
            const additiv::ModRow r = proc_.engine().modRow(i);
            if (r.source != additiv::ModSource::Off && r.target == c.id && r.depth != 0.0f) return true;
        }
    }
    return !gated(b);
}

std::vector<ParamPage::GroupState> ParamPage::groupStates() const
{
    std::vector<GroupState> out;
    for (const Box& b : boxes_) out.push_back({ b.title, b.colour, gated(b), awake(b) });
    return out;
}

int ParamPage::targetAt(juce::Point<int> p) const
{
    for (const Box& b : boxes_)
        for (const Cell& c : b.cells)
            if (dynamic_cast<juce::Slider*>(c.control) != nullptr && c.control->getBounds().expanded(4).contains(p) && knobmod::canTarget(c.id)) return c.id;
    return -1;
}

bool ParamPage::isInterestedInDragSource(const SourceDetails& d) { return knobmod::sourceOfDrag(d.description) != additiv::ModSource::Off; }

void ParamPage::itemDragMove(const SourceDetails& d)
{
    const int id = targetAt(d.localPosition);
    if (id == dropId_) return;
    dropId_ = id;
    dropSource_ = knobmod::sourceOfDrag(d.description);
    repaint();
}

void ParamPage::itemDragExit(const SourceDetails&)
{
    dropId_ = -1;
    repaint();
}

void ParamPage::itemDropped(const SourceDetails& d)
{
    const int id = targetAt(d.localPosition);
    const additiv::ModSource s = knobmod::sourceOfDrag(d.description);
    dropId_ = -1;
    repaint();
    if (id < 0) return;
    const int row = knobmod::addRow(proc_, s, id);
    if (row >= 0) flash(id);
    if (auto* listener = findParentComponentOfClass<DropListener>()) listener->modDropped(s, id, row);
}

juce::Component* ParamPage::controlFor(int id) const
{
    for (const Box& b : boxes_)
        for (const Cell& c : b.cells)
            if (c.id == id || c.id2 == id) return c.control;
    return nullptr;
}

juce::String ParamPage::groupOf(int id) const
{
    for (const Box& b : boxes_)
        for (const Cell& c : b.cells)
            if (c.id == id || c.id2 == id) return b.title;
    return {};
}

void ParamPage::flash(int id)
{
    flashId_ = id;
    flashUntil_ = juce::Time::getMillisecondCounterHiRes() + 1800.0;
    repaint();
}

void ParamPage::paintOverChildren(juce::Graphics& g)
{
    // The knob a dragged source would modulate: a ring in the source's colour, and what would be made.
    if (dropId_ >= 0)
        if (juce::Component* c = controlFor(dropId_)) {
            const juce::Colour col = knobmod::sourceColour(dropSource_);
            g.setColour(col.withAlpha(0.18f));
            g.fillRoundedRectangle(c->getBounds().toFloat().expanded(4.0f), 8.0f);
            g.setColour(col);
            g.drawRoundedRectangle(c->getBounds().toFloat().expanded(4.0f), 8.0f, 2.0f);
            g.setFont(juce::FontOptions(11.0f, juce::Font::bold));
            g.drawText(juce::String(additiv::kModSourceNames[static_cast<int>(dropSource_)]) + " -> " + additiv::paramDesc(static_cast<additiv::ParamId>(dropId_)).name,
                       c->getBounds().withY(c->getY() - 18).withHeight(14).expanded(40, 0), juce::Justification::centred);
        }
    // A control the palette found: a ring that fades over two seconds.
    if (flashId_ >= 0)
        if (juce::Component* c = controlFor(flashId_)) {
            const double left = flashUntil_ - juce::Time::getMillisecondCounterHiRes();
            if (left > 0.0) {
                const float a = static_cast<float>(std::min(1.0, left / 900.0));
                g.setColour(additivui::skin().accent.withAlpha(a));
                g.drawRoundedRectangle(c->getBounds().toFloat().expanded(4.0f), 8.0f, 2.5f);
            }
        }
    // The MIDI controllers learned, as small marks over the controls (the header's MIDI button).
    if (showMidi)
        for (const Box& b : boxes_)
            for (const Cell& c : b.cells) {
                const int cc = proc_.controllerFor(c.id);
                if (cc < 0 || c.control == nullptr) continue;
                const juce::String text = "CC " + juce::String(cc);
                const auto at = juce::Rectangle<float>(static_cast<float>(c.control->getRight()) - 34.0f, static_cast<float>(c.control->getY()) - 2.0f, 34.0f, 14.0f);
                g.setColour(additivui::skin().accent.withAlpha(0.9f));
                g.fillRoundedRectangle(at, 4.0f);
                g.setColour(additivui::skin().bg);
                g.setFont(juce::FontOptions(10.0f, juce::Font::bold));
                g.drawText(text, at, juce::Justification::centred);
            }
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
    // The rows: in order, a new one whenever the next box does not fit -- unless taking a later box up
    // into an earlier row's gap saves a row. Then the fewest rows, found by trying every box in every
    // row so far before a new one: the first way found keeps the earlier boxes in the earlier rows.
    const int n = static_cast<int>(boxes_.size());
    const int room = std::max(width - 2 * kGap, kKnobCell + 2 * kPad);
    std::vector<int> w(static_cast<size_t>(n)), row(static_cast<size_t>(n)), best(static_cast<size_t>(n));
    for (int i = 0; i < n; ++i) w[static_cast<size_t>(i)] = std::min(boxes_[static_cast<size_t>(i)].width, room);
    int rows = 0, sum = 0;
    for (int i = 0, used = 0; i < n; ++i) {
        const int wi = w[static_cast<size_t>(i)];
        if (used > 0 && used + kGap + wi > room) { ++rows; used = 0; }
        best[static_cast<size_t>(i)] = rows;
        used += (used > 0 ? kGap : 0) + wi;
        sum += wi + kGap;
    }
    if (n > 0) ++rows;
    std::vector<int> used;
    int limit = 0;
    std::function<bool(int)> place = [&](int i) {
        if (i == n) return true;
        const int wi = w[static_cast<size_t>(i)];
        for (size_t r = 0; r < used.size(); ++r)
            if (used[r] + kGap + wi <= room) {
                used[r] += kGap + wi;
                row[static_cast<size_t>(i)] = static_cast<int>(r);
                if (place(i + 1)) return true;
                used[r] -= kGap + wi;
            }
        if (static_cast<int>(used.size()) >= limit) return false;
        used.push_back(wi);
        row[static_cast<size_t>(i)] = static_cast<int>(used.size()) - 1;
        if (place(i + 1)) return true;
        used.pop_back();
        return false;
    };
    for (limit = std::max(1, (sum - kGap + room) / (room + kGap)); limit < rows; ++limit) {
        used.clear();
        if (place(0)) {
            best = row;
            rows = limit;
            break;
        }
    }
    std::vector<juce::Rectangle<int>> out(static_cast<size_t>(n));
    std::vector<int> x(static_cast<size_t>(rows), kGap);
    const int h = kTitle + kCellHeight + kPad;
    for (int i = 0; i < n; ++i) {
        const int r = best[static_cast<size_t>(i)], wi = w[static_cast<size_t>(i)];
        out[static_cast<size_t>(i)] = { x[static_cast<size_t>(r)], kGap + r * (h + kGap), wi, h };
        x[static_cast<size_t>(r)] += wi + kGap;
    }
    return out;
}

int ParamPage::heightFor(int width) const
{
    int bottom = 0;   // the lowest box (a later box may sit in an earlier row)
    for (const auto& b : flow(width)) bottom = std::max(bottom, b.getBottom());
    return bottom > 0 ? bottom + kGap : 0;
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
            if (c.id2 >= 0) c.control->setBounds(x + 4, top + 26, c.width - 8, 46);
            else if (dynamic_cast<juce::ComboBox*>(c.control) != nullptr) c.control->setBounds(x + 6, top + 34, c.width - 12, 26);
            else if (dynamic_cast<juce::ToggleButton*>(c.control) != nullptr) c.control->setBounds(x + 14, top + 34, c.width - 28, 26);
            else c.control->setBounds(x, top + 16, c.width, kCellHeight - 18);
            x += c.width;
        }
    }
    for (const auto& l : links_) {   // the chain between the two knobs, at their rings' height
        const auto ra = l->a->getBounds(), rb = l->b->getBounds();
        const int cx = (ra.getRight() + rb.getX()) / 2, cy = ra.getY() + (ra.getHeight() - 16) / 2;
        l->chain->setBounds(cx - 9, cy - 7, 18, 14);
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
            if (c.id2 >= 0) {   // a span: both ends
                for (int end : { c.id, c.id2 }) {
                    const additiv::ParamDesc& d = additiv::paramDesc(static_cast<additiv::ParamId>(end));
                    s << "    " << d.name << " (" << d.key << "): " << juce::String(d.min) << " .. " << juce::String(d.max)
                      << (d.unit[0] != 0 ? juce::String(" ") + d.unit : juce::String()) << ", now " << juce::String(proc_.value(end), 2) << "\n";
                }
                continue;
            }
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
