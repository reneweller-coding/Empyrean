/**
 * @file ParamPage.cpp
 * @brief A page of the panel (ParamPage.h).
 * @author Rene Weller
 */
#include "ParamPage.h"

#include "KnobMod.h"
#include "Theme.h"

#include "additiv/Presets.h"

#include <algorithm>
#include <functional>

namespace {

constexpr int KnobCell = 82;     ///< a knob's cell
constexpr int ChoiceCell = 132;  ///< a chooser's cell
constexpr int Title = 26;        ///< a box's title row
constexpr int CellHeight = 100;  ///< a cell: name, control, value
constexpr int Gap = 10;          ///< between boxes
constexpr int Pad = 8;           ///< inside a box

/**
 * @brief A span: two parameters, its low and its high end, as one bar with two ends to drag. Bound to the host's
 *        parameters with gestures (an undo step each drag); a right-click offers the nearer end's menu.
 */
class SpanControl final : public juce::Component, public juce::SettableTooltipClient {
public:
    /**
     * @brief The span of @p low and @p high.
     * @param[in,out] proc     the processor
     * @param[in]     actions  the right-click menu
     * @param[in]     low      the low end's parameter
     * @param[in]     high     the high end's
     * @param[in]     colour   its group's colour
     */
    SpanControl(AdditivProcessor& proc, const frame::ControlActions& actions, int low, int high, juce::Colour colour)
        : m_proc(proc), m_actions(actions), m_id{ low, high }, m_colour(colour)
    {
        for (int e = 0; e < 2; ++e)
            if (auto* p = m_proc.parameter(m_id[e]))
                m_attach[e] = std::make_unique<juce::ParameterAttachment>(*p, [this](float) { repaint(); });
        for (auto& a : m_attach)
            if (a != nullptr) a->sendInitialUpdate();
    }
    /**
     * @brief The track, the span lit, the two ends, their values.
     * @param[in,out] g  where
     */
    void paint(juce::Graphics& g) override
    {
        const frame::Skin& skin = additivui::skin();
        const auto t = track();
        g.setColour(skin.m_raised);
        g.fillRoundedRectangle(t.withSizeKeepingCentre(t.getWidth(), 4.0f), 2.0f);
        const float x0 = t.getX() + t.getWidth() * norm(0), x1 = t.getX() + t.getWidth() * norm(1);
        g.setColour(m_colour);
        g.fillRoundedRectangle(juce::Rectangle<float>::leftTopRightBottom(std::min(x0, x1), t.getCentreY() - 2.0f, std::max(x0, x1), t.getCentreY() + 2.0f), 2.0f);
        for (int e = 0; e < 2; ++e) {
            const float x = e == 0 ? x0 : x1;
            g.setColour(e == m_held ? skin.m_ink : m_colour.brighter(0.3f));
            g.fillEllipse(x - 6.0f, t.getCentreY() - 6.0f, 12.0f, 12.0f);
            g.setColour(skin.m_bg);
            g.fillEllipse(x - 2.5f, t.getCentreY() - 2.5f, 5.0f, 5.0f);
        }
        g.setColour(skin.m_dim);
        g.setFont(juce::FontOptions(11.5f));
        g.drawText(text(0) + "  -  " + text(1), getLocalBounds().removeFromBottom(18), juce::Justification::centred);
    }
    /**
     * @brief The nearer end is taken (a right-click: its menu).
     * @param[in] e  the press
     */
    void mouseDown(const juce::MouseEvent& e) override
    {
        const auto t = track();
        const float x0 = t.getX() + t.getWidth() * norm(0), x1 = t.getX() + t.getWidth() * norm(1);
        const int near = std::fabs(static_cast<float>(e.x) - x0) <= std::fabs(static_cast<float>(e.x) - x1) ? 0 : 1;
        if (e.mods.isPopupMenu()) {
            m_actions.showMenu(*this, m_id[near]);
            return;
        }
        m_held = near;
        if (m_attach[m_held] != nullptr) m_attach[m_held]->beginGesture();
        mouseDrag(e);
    }
    /**
     * @brief The end held follows, never past the other.
     * @param[in] e  the drag
     */
    void mouseDrag(const juce::MouseEvent& e) override
    {
        if (m_held < 0 || m_attach[m_held] == nullptr) return;
        const auto t = track();
        float n = std::clamp((static_cast<float>(e.x) - t.getX()) / t.getWidth(), 0.0f, 1.0f);
        n = m_held == 0 ? std::min(n, norm(1)) : std::max(n, norm(0));
        auto* p = m_proc.parameter(m_id[m_held]);
        m_attach[m_held]->setValueAsPartOfGesture(p->convertFrom0to1(n));
        repaint();
    }
    /**
     * @brief Let go.
     */
    void mouseUp(const juce::MouseEvent&) override
    {
        if (m_held >= 0 && m_attach[m_held] != nullptr) m_attach[m_held]->endGesture();
        m_held = -1;
        repaint();
    }
    /** @brief Both ends, as the info line shows them. @return the text */
    juce::String getTooltip() override
    {
        return juce::String(additiv::paramDesc(static_cast<additiv::ParamIdE>(m_id[0])).m_name) + " and " + additiv::paramDesc(static_cast<additiv::ParamIdE>(m_id[1])).m_name
               + ": " + text(0) + " to " + text(1) + " -- drag either end (right-click: its menu)";
    }

private:
    /** @brief The track's line. @return pixels */
    juce::Rectangle<float> track() const { return getLocalBounds().toFloat().reduced(10.0f, 0.0f).withTrimmedBottom(20.0f); }
    /**
     * @brief An end where it stands, 0 .. 1 along the track (its parameter's own skew).
     * @param[in] e  0 low, 1 high
     * @return 0 .. 1
     */
    float norm(int e) const
    {
        auto* p = m_proc.parameter(m_id[e]);
        return p != nullptr ? p->getValue() : 0.0f;
    }
    /**
     * @brief An end's value in words: a key's name for a note, else the value and its unit.
     * @param[in] e  0 low, 1 high
     * @return the text
     */
    juce::String text(int e) const
    {
        const additiv::ParamDesc& d = additiv::paramDesc(static_cast<additiv::ParamIdE>(m_id[e]));
        const float v = m_proc.value(m_id[e]);
        if (d.m_id == additiv::ParamIdE::BRAIN_LOW || d.m_id == additiv::ParamIdE::BRAIN_HIGH) return juce::MidiMessage::getMidiNoteName(juce::roundToInt(v), true, true, 4);
        return juce::String(v, v >= 100.0f ? 0 : 1) + (d.m_unit[0] != 0 ? juce::String(" ") + d.m_unit : juce::String());
    }

    AdditivProcessor& m_proc;                                ///< whose parameters
    const frame::ControlActions& m_actions;                  ///< the right-click menu
    int m_id[2];                                             ///< low and high
    juce::Colour m_colour;                                   ///< its group's
    std::unique_ptr<juce::ParameterAttachment> m_attach[2];  ///< to the parameters
    int m_held = -1;                                         ///< the end being dragged, -1 none
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
    /**
     * @brief Two rings of a chain, lit when linked.
     * @param[in,out] g     where
     * @param[in]     over  under the mouse
     * @param[in]     down  pressed
     */
    void paintButton(juce::Graphics& g, bool over, bool down) override
    {
        (void)down;
        const frame::Skin& skin = additivui::skin();
        const auto r = getLocalBounds().toFloat().reduced(2.0f);
        g.setColour(getToggleState() ? skin.m_accent : (over ? skin.m_ink : skin.m_faint));
        const float w = r.getWidth() * 0.62f, h = r.getHeight() * 0.42f;
        g.drawRoundedRectangle(r.getX(), r.getCentreY() - h * 0.5f, w, h, h * 0.5f, 1.6f);
        g.drawRoundedRectangle(r.getRight() - w, r.getCentreY() - h * 0.5f, w, h, h * 0.5f, 1.6f);
    }
};

} // namespace

ParamPage::ParamPage(AdditivProcessor& proc, const frame::ControlActions& actions, std::vector<Group> groups, frame::LiveRings* live)
    : m_proc(proc)
{
    const frame::Skin& skin = additivui::skin();
    for (const Group& g : groups) {
        Box box;
        box.m_title = g.m_title;
        box.m_colour = skin.family(g.m_family);
        for (additiv::ParamIdE pid : g.m_ids) {
            const int id = static_cast<int>(pid);
            juce::RangedAudioParameter* param = m_proc.parameter(id);
            if (param == nullptr) continue;
            m_ids.push_back(pid);
            if (std::any_of(g.m_spans.begin(), g.m_spans.end(), [pid](const auto& s) { return s.second == pid; })) continue;   // in its low end's bar
            if (auto span = std::find_if(g.m_spans.begin(), g.m_spans.end(), [pid](const auto& s) { return s.first == pid; }); span != g.m_spans.end()) {
                const additiv::ParamDesc& lo = additiv::paramDesc(span->first), &hi = additiv::paramDesc(span->second);
                auto* label = m_labels.add(new juce::Label({}, juce::String(lo.m_name) + " - " + hi.m_name));
                label->setJustificationType(juce::Justification::centred);
                label->setColour(juce::Label::textColourId, skin.m_dim);
                label->setFont(juce::FontOptions(12.0f));
                addAndMakeVisible(label);
                auto* c = new SpanControl(m_proc, actions, id, static_cast<int>(span->second), box.m_colour);
                m_controls.add(c);
                addAndMakeVisible(c);
                Cell cell;
                cell.m_label = label;
                cell.m_id = id;
                cell.m_id2 = static_cast<int>(span->second);
                cell.m_control = c;
                cell.m_width = 2 * KnobCell;
                box.m_cells.push_back(cell);
                continue;
            }
            const additiv::ParamDesc& d = additiv::paramDesc(pid);
            auto* label = m_labels.add(new juce::Label({}, d.m_name));
            label->setJustificationType(juce::Justification::centred);
            label->setColour(juce::Label::textColourId, skin.m_dim);
            label->setFont(juce::FontOptions(12.0f));
            label->setMinimumHorizontalScale(0.7f);
            addAndMakeVisible(label);
            Cell cell;
            cell.m_label = label;
            cell.m_id = id;
            cell.m_gate = std::find(g.m_gate.begin(), g.m_gate.end(), pid) != g.m_gate.end();
            if (cell.m_gate) {   // a main control: its name in the group's colour
                label->setColour(juce::Label::textColourId, box.m_colour);
                label->setFont(juce::FontOptions(12.5f, juce::Font::bold));
            }
            if (d.m_kind == additiv::ParamKindE::CHOICE) {
                auto* c = new frame::Choice(&actions, id);
                for (int i = 0; i < d.m_numChoices; ++i) c->addItem(d.m_choices[i], i + 1);
                c->setColour(juce::ComboBox::arrowColourId, box.m_colour);
                m_controls.add(c);
                m_combos.push_back(std::make_unique<juce::ComboBoxParameterAttachment>(*param, *c));
                cell.m_width = ChoiceCell;
            } else if (d.m_kind == additiv::ParamKindE::BOOL) {
                auto* b = new frame::Switch(&actions, id);
                b->setColour(juce::ToggleButton::tickColourId, box.m_colour);
                m_controls.add(b);
                m_buttons.push_back(std::make_unique<juce::ButtonParameterAttachment>(*param, *b));
                cell.m_width = KnobCell;
            } else {
                auto* k = new frame::Knob(juce::Slider::RotaryHorizontalVerticalDrag, juce::Slider::TextBoxBelow, &actions, id);
                k->setTextBoxStyle(juce::Slider::TextBoxBelow, false, KnobCell - 6, 16);
                k->setColour(juce::Slider::rotarySliderFillColourId, box.m_colour);
                k->setTextValueSuffix(d.m_unit[0] != 0 ? juce::String(" ") + d.m_unit : juce::String());
                k->setTooltip(juce::String(d.m_key) + ": " + d.m_name + (d.m_unit[0] != 0 ? juce::String(" (") + d.m_unit + ")" : juce::String()));
                m_controls.add(k);
                m_sliders.push_back(std::make_unique<juce::SliderParameterAttachment>(*param, *k));
                k->setDoubleClickReturnValue(true, d.m_def);
                if (live != nullptr) live->add(*k, id);   // where the matrix makes it play
                cell.m_width = KnobCell;
            }
            addAndMakeVisible(m_controls.getLast());
            cell.m_control = m_controls.getLast();
            box.m_cells.push_back(cell);
        }
        if (box.m_cells.empty()) continue;
        box.m_width = 2 * Pad;
        for (const Cell& c : box.m_cells) box.m_width += c.m_width;
        // Linked pairs: a chain between their knobs.
        for (const auto& [ia, ib] : g.m_links) {
            juce::Slider* a = nullptr;
            juce::Slider* b = nullptr;
            for (const Cell& c : box.m_cells) {
                if (c.m_id == static_cast<int>(ia)) a = dynamic_cast<juce::Slider*>(c.m_control);
                if (c.m_id == static_cast<int>(ib)) b = dynamic_cast<juce::Slider*>(c.m_control);
            }
            if (a == nullptr || b == nullptr) continue;
            auto link = std::make_unique<Link>();
            link->m_a = a;
            link->m_b = b;
            link->m_chain = std::make_unique<ChainButton>();
            addAndMakeVisible(*link->m_chain);
            Link* l = link.get();
            auto follow = [l](juce::Slider* moved, juce::Slider* other, bool forward) {
                if (!l->m_chain->getToggleState() || l->m_busy) return;
                l->m_busy = true;
                const double v = forward ? moved->getValue() * l->m_ratio : moved->getValue() / std::max(l->m_ratio, 1.0e-6);
                other->setValue(juce::jlimit(other->getMinimum(), other->getMaximum(), v), juce::sendNotificationSync);
                l->m_busy = false;
            };
            auto take = [l] { l->m_ratio = l->m_a->getValue() > 1.0e-9 ? l->m_b->getValue() / l->m_a->getValue() : 1.0; };
            a->onDragStart = take;
            b->onDragStart = take;
            a->onValueChange = [l, follow] { follow(l->m_a, l->m_b, true); };
            b->onValueChange = [l, follow] { follow(l->m_b, l->m_a, false); };
            l->m_chain->onClick = take;
            m_links.push_back(std::move(link));
        }
        m_boxes.push_back(std::move(box));
    }
    startTimerHz(15);
}

void ParamPage::timerCallback()
{
    if (!isShowing()) return;
    if (m_flashId >= 0) {   // the found control's ring fades
        if (juce::Component* c = controlFor(m_flashId)) repaint(c->getBounds().expanded(8));
        if (juce::Time::getMillisecondCounterHiRes() > m_flashUntil) m_flashId = -1;
    }
    for (Box& b : m_boxes) {
        const bool resting = gated(b) && !awake(b);
        if (resting != b.m_resting) {
            b.m_resting = resting;
            for (Cell& c : b.m_cells)
                if (!c.m_gate) {
                    c.m_control->setAlpha(resting ? 0.4f : 1.0f);
                    c.m_label->setAlpha(resting ? 0.5f : 1.0f);
                }
            repaint(b.m_bounds);
        }
        // The knobs the matrix moves (and those it moved a moment ago, to clear their arcs); the
        // preset's marks with them (the timer repaints every knob that has moved from its preset).
        for (Cell& c : b.m_cells) {
            auto* k = dynamic_cast<juce::Slider*>(c.m_control);
            if (k == nullptr) continue;
            bool on = false;
            for (int i = 0; i < additiv::ModRows && !on; ++i) {
                const additiv::ModRow r = m_proc.engine().modRow(i);
                on = r.m_source != additiv::ModSourceE::OFF && r.m_target == c.m_id;
            }
            const auto& ref = m_proc.reference();
            const bool away = ref.size() == static_cast<size_t>(additiv::NumParams) && ref[static_cast<size_t>(c.m_id)] != static_cast<float>(k->getValue());
            if (on || c.m_moved || away != c.m_away) repaint(k->getBounds().expanded(12));
            c.m_moved = on;
            c.m_away = away;
        }
    }
}

bool ParamPage::gated(const Box& b)
{
    for (const Cell& c : b.m_cells)
        if (c.m_gate) return true;
    return false;
}

bool ParamPage::awake(const Box& b) const
{
    // Awake while a gate is up, or a matrix row moves one (it lifts it from nothing).
    for (const Cell& c : b.m_cells) {
        if (!c.m_gate) continue;
        if (std::abs(m_proc.value(c.m_id)) > 1.0e-6f) return true;
        for (int i = 0; i < additiv::ModRows; ++i) {
            const additiv::ModRow r = m_proc.engine().modRow(i);
            if (r.m_source != additiv::ModSourceE::OFF && r.m_target == c.m_id && r.m_depth != 0.0f) return true;
        }
    }
    return !gated(b);
}

std::vector<ParamPage::GroupState> ParamPage::groupStates() const
{
    std::vector<GroupState> out;
    for (const Box& b : m_boxes) out.push_back({ b.m_title, b.m_colour, gated(b), awake(b) });
    return out;
}

int ParamPage::targetAt(juce::Point<int> p) const
{
    for (const Box& b : m_boxes)
        for (const Cell& c : b.m_cells)
            if (dynamic_cast<juce::Slider*>(c.m_control) != nullptr && c.m_control->getBounds().expanded(4).contains(p) && knobmod::canTarget(c.m_id)) return c.m_id;
    return -1;
}

bool ParamPage::isInterestedInDragSource(const SourceDetails& d) { return knobmod::sourceOfDrag(d.description) != additiv::ModSourceE::OFF; }

void ParamPage::itemDragMove(const SourceDetails& d)
{
    const int id = targetAt(d.localPosition);
    if (id == m_dropId) return;
    m_dropId = id;
    m_dropSource = knobmod::sourceOfDrag(d.description);
    repaint();
}

void ParamPage::itemDragExit(const SourceDetails&)
{
    m_dropId = -1;
    repaint();
}

void ParamPage::itemDropped(const SourceDetails& d)
{
    const int id = targetAt(d.localPosition);
    const additiv::ModSourceE s = knobmod::sourceOfDrag(d.description);
    m_dropId = -1;
    repaint();
    if (id < 0) return;
    const int row = knobmod::addRow(m_proc, s, id);
    if (row >= 0) flash(id);
    if (auto* listener = findParentComponentOfClass<DropListener>()) listener->modDropped(s, id, row);
}

juce::Component* ParamPage::controlFor(int id) const
{
    for (const Box& b : m_boxes)
        for (const Cell& c : b.m_cells)
            if (c.m_id == id || c.m_id2 == id) return c.m_control;
    return nullptr;
}

juce::String ParamPage::groupOf(int id) const
{
    for (const Box& b : m_boxes)
        for (const Cell& c : b.m_cells)
            if (c.m_id == id || c.m_id2 == id) return b.m_title;
    return {};
}

void ParamPage::flash(int id)
{
    m_flashId = id;
    m_flashUntil = juce::Time::getMillisecondCounterHiRes() + 1800.0;
    repaint();
}

void ParamPage::paintOverChildren(juce::Graphics& g)
{
    // The knob a dragged source would modulate: a ring in the source's colour, and what would be made.
    if (m_dropId >= 0)
        if (juce::Component* c = controlFor(m_dropId)) {
            const juce::Colour col = knobmod::sourceColour(m_dropSource);
            g.setColour(col.withAlpha(0.18f));
            g.fillRoundedRectangle(c->getBounds().toFloat().expanded(4.0f), 8.0f);
            g.setColour(col);
            g.drawRoundedRectangle(c->getBounds().toFloat().expanded(4.0f), 8.0f, 2.0f);
            g.setFont(juce::FontOptions(11.0f, juce::Font::bold));
            g.drawText(juce::String(additiv::ModSourceNames[static_cast<int>(m_dropSource)]) + " -> " + additiv::paramDesc(static_cast<additiv::ParamIdE>(m_dropId)).m_name,
                       c->getBounds().withY(c->getY() - 18).withHeight(14).expanded(40, 0), juce::Justification::centred);
        }
    // A control the palette found: a ring that fades over two seconds.
    if (m_flashId >= 0)
        if (juce::Component* c = controlFor(m_flashId)) {
            const double left = m_flashUntil - juce::Time::getMillisecondCounterHiRes();
            if (left > 0.0) {
                const float a = static_cast<float>(std::min(1.0, left / 900.0));
                g.setColour(additivui::skin().m_accent.withAlpha(a));
                g.drawRoundedRectangle(c->getBounds().toFloat().expanded(4.0f), 8.0f, 2.5f);
            }
        }
    // The MIDI controllers learned, as small marks over the controls (the header's MIDI button).
    if (M_ShowMidi)
        for (const Box& b : m_boxes)
            for (const Cell& c : b.m_cells) {
                const int cc = m_proc.controllerFor(c.m_id);
                if (cc < 0 || c.m_control == nullptr) continue;
                const juce::String text = "CC " + juce::String(cc);
                const auto at = juce::Rectangle<float>(static_cast<float>(c.m_control->getRight()) - 34.0f, static_cast<float>(c.m_control->getY()) - 2.0f, 34.0f, 14.0f);
                g.setColour(additivui::skin().m_accent.withAlpha(0.9f));
                g.fillRoundedRectangle(at, 4.0f);
                g.setColour(additivui::skin().m_bg);
                g.setFont(juce::FontOptions(10.0f, juce::Font::bold));
                g.drawText(text, at, juce::Justification::centred);
            }
    const auto& ref = m_proc.reference();
    const bool marks = ref.size() == static_cast<size_t>(additiv::NumParams);
    for (const Box& b : m_boxes)
        for (const Cell& c : b.m_cells)
            if (auto* k = dynamic_cast<juce::Slider*>(c.m_control)) {
                knobmod::paintArcs(g, m_proc, *k, c.m_id);
                // Where the preset had it, when it has moved: a small notch outside the ring.
                if (!marks || additiv::keptByPresets(static_cast<additiv::ParamIdE>(c.m_id))) continue;
                const float was = ref[static_cast<size_t>(c.m_id)];
                if (std::fabs(was - static_cast<float>(k->getValue())) <= 1.0e-5f * static_cast<float>(std::max(1.0, k->getMaximum() - k->getMinimum()))) continue;
                const auto area = k->getLookAndFeel().getSliderLayout(*k).sliderBounds.toFloat() + k->getPosition().toFloat();
                const float size = std::min(area.getWidth(), area.getHeight()), ring = size * 0.5f - 3.0f;
                const auto rp = k->getRotaryParameters();
                const float a = rp.startAngleRadians + static_cast<float>(k->valueToProportionOfLength(was)) * (rp.endAngleRadians - rp.startAngleRadians);
                const auto c0 = area.getCentre();
                const float r0 = ring + 1.0f, r1 = ring + 6.0f;
                g.setColour(additivui::skin().m_ink.withAlpha(0.75f));
                g.drawLine(c0.x + std::sin(a) * r0, c0.y - std::cos(a) * r0, c0.x + std::sin(a) * r1, c0.y - std::cos(a) * r1, 1.6f);
            }
}

std::vector<juce::Rectangle<int>> ParamPage::flow(int width) const
{
    // The rows: in order, a new one whenever the next box does not fit -- unless taking a later box up
    // into an earlier row's gap saves a row. Then the fewest rows, found by trying every box in every
    // row so far before a new one: the first way found keeps the earlier boxes in the earlier rows.
    const int n = static_cast<int>(m_boxes.size());
    const int room = std::max(width - 2 * Gap, KnobCell + 2 * Pad);
    std::vector<int> w(static_cast<size_t>(n)), row(static_cast<size_t>(n)), best(static_cast<size_t>(n));
    for (int i = 0; i < n; ++i) w[static_cast<size_t>(i)] = std::min(m_boxes[static_cast<size_t>(i)].m_width, room);
    int rows = 0, sum = 0;
    for (int i = 0, used = 0; i < n; ++i) {
        const int wi = w[static_cast<size_t>(i)];
        if (used > 0 && used + Gap + wi > room) { ++rows; used = 0; }
        best[static_cast<size_t>(i)] = rows;
        used += (used > 0 ? Gap : 0) + wi;
        sum += wi + Gap;
    }
    if (n > 0) ++rows;
    std::vector<int> used;
    int limit = 0;
    std::function<bool(int)> place = [&](int i) {
        if (i == n) return true;
        const int wi = w[static_cast<size_t>(i)];
        for (size_t r = 0; r < used.size(); ++r)
            if (used[r] + Gap + wi <= room) {
                used[r] += Gap + wi;
                row[static_cast<size_t>(i)] = static_cast<int>(r);
                if (place(i + 1)) return true;
                used[r] -= Gap + wi;
            }
        if (static_cast<int>(used.size()) >= limit) return false;
        used.push_back(wi);
        row[static_cast<size_t>(i)] = static_cast<int>(used.size()) - 1;
        if (place(i + 1)) return true;
        used.pop_back();
        return false;
    };
    for (limit = std::max(1, (sum - Gap + room) / (room + Gap)); limit < rows; ++limit) {
        used.clear();
        if (place(0)) {
            best = row;
            rows = limit;
            break;
        }
    }
    std::vector<juce::Rectangle<int>> out(static_cast<size_t>(n));
    std::vector<int> x(static_cast<size_t>(rows), Gap);
    const int h = Title + CellHeight + Pad;
    for (int i = 0; i < n; ++i) {
        const int r = best[static_cast<size_t>(i)], wi = w[static_cast<size_t>(i)];
        out[static_cast<size_t>(i)] = { x[static_cast<size_t>(r)], Gap + r * (h + Gap), wi, h };
        x[static_cast<size_t>(r)] += wi + Gap;
    }
    return out;
}

int ParamPage::heightFor(int width) const
{
    int bottom = 0;   // the lowest box (a later box may sit in an earlier row)
    for (const auto& b : flow(width)) bottom = std::max(bottom, b.getBottom());
    return bottom > 0 ? bottom + Gap : 0;
}

void ParamPage::resized()
{
    const auto r = flow(getWidth());
    for (size_t i = 0; i < m_boxes.size(); ++i) {
        Box& b = m_boxes[i];
        b.m_bounds = r[i];
        int x = b.m_bounds.getX() + Pad;
        const int top = b.m_bounds.getY() + Title;
        for (Cell& c : b.m_cells) {
            c.m_label->setBounds(x, top, c.m_width, 16);
            if (c.m_id2 >= 0) c.m_control->setBounds(x + 4, top + 26, c.m_width - 8, 46);
            else if (dynamic_cast<juce::ComboBox*>(c.m_control) != nullptr) c.m_control->setBounds(x + 6, top + 34, c.m_width - 12, 26);
            else if (dynamic_cast<juce::ToggleButton*>(c.m_control) != nullptr) c.m_control->setBounds(x + 14, top + 34, c.m_width - 28, 26);
            else c.m_control->setBounds(x, top + 16, c.m_width, CellHeight - 18);
            x += c.m_width;
        }
    }
    for (const auto& l : m_links) {   // the chain between the two knobs, at their rings' height
        const auto ra = l->m_a->getBounds(), rb = l->m_b->getBounds();
        const int cx = (ra.getRight() + rb.getX()) / 2, cy = ra.getY() + (ra.getHeight() - 16) / 2;
        l->m_chain->setBounds(cx - 9, cy - 7, 18, 14);
    }
}

void ParamPage::paint(juce::Graphics& g)
{
    const frame::Skin& skin = additivui::skin();
    for (const Box& b : m_boxes) {
        const auto r = b.m_bounds.toFloat();
        g.setColour(skin.m_group.withAlpha(0.92f));
        g.fillRoundedRectangle(r, skin.m_radius);
        g.setColour(skin.m_edge);
        g.drawRoundedRectangle(r.reduced(0.5f), skin.m_radius, 1.0f);
        g.setColour(b.m_colour.withAlpha(b.m_resting ? 0.5f : 1.0f));
        g.fillRoundedRectangle(r.getX() + Pad, r.getY() + 9.0f, 3.0f, 10.0f, 1.5f);
        g.setFont(juce::FontOptions(13.0f, juce::Font::bold));
        g.drawText(b.m_resting ? b.m_title + juce::String("  -  off") : b.m_title, b.m_bounds.withHeight(Title).withTrimmedLeft(Pad + 9),
                   juce::Justification::centredLeft);
        // The main controls: a soft halo of the group's colour behind them.
        for (const Cell& c : b.m_cells) {
            if (!c.m_gate || c.m_control == nullptr) continue;
            const bool knob = dynamic_cast<juce::Slider*>(c.m_control) != nullptr;
            const auto cb = c.m_control->getBounds().toFloat();
            const float d = std::min(cb.getWidth(), cb.getHeight() - (knob ? 16.0f : 0.0f));
            const auto centre = knob ? juce::Point<float>(cb.getCentreX(), cb.getY() + d * 0.5f) : cb.getCentre();
            g.setGradientFill(juce::ColourGradient(b.m_colour.withAlpha(0.16f), centre.x, centre.y, b.m_colour.withAlpha(0.0f), centre.x + d * 0.62f,
                                                   centre.y, true));
            g.fillEllipse(centre.x - d * 0.62f, centre.y - d * 0.62f, d * 1.24f, d * 1.24f);
        }
    }
}

juce::String ParamPage::describe() const
{
    juce::String s;
    for (const Box& b : m_boxes) {
        s << b.m_title << "\n";
        for (const Cell& c : b.m_cells) {
            if (c.m_id2 >= 0) {   // a span: both ends
                for (int end : { c.m_id, c.m_id2 }) {
                    const additiv::ParamDesc& d = additiv::paramDesc(static_cast<additiv::ParamIdE>(end));
                    s << "    " << d.m_name << " (" << d.m_key << "): " << juce::String(d.m_min) << " .. " << juce::String(d.m_max)
                      << (d.m_unit[0] != 0 ? juce::String(" ") + d.m_unit : juce::String()) << ", now " << juce::String(m_proc.value(end), 2) << "\n";
                }
                continue;
            }
            const int id = c.m_control != nullptr ? (dynamic_cast<frame::Knob*>(c.m_control) != nullptr ? dynamic_cast<frame::Knob*>(c.m_control)->id()
                                                  : dynamic_cast<frame::Choice*>(c.m_control) != nullptr ? dynamic_cast<frame::Choice*>(c.m_control)->id()
                                                  : dynamic_cast<frame::Switch*>(c.m_control) != nullptr ? dynamic_cast<frame::Switch*>(c.m_control)->id() : -1) : -1;
            if (id < 0) continue;
            const additiv::ParamDesc& d = additiv::paramDesc(static_cast<additiv::ParamIdE>(id));
            s << "    " << d.m_name << " (" << d.m_key << "): ";
            if (d.m_kind == additiv::ParamKindE::CHOICE) {
                for (int i = 0; i < d.m_numChoices; ++i) s << (i > 0 ? ", " : "") << d.m_choices[i];
            } else if (d.m_kind == additiv::ParamKindE::BOOL) {
                s << "on or off";
            } else {
                s << juce::String(d.m_min) << " .. " << juce::String(d.m_max) << (d.m_unit[0] != 0 ? juce::String(" ") + d.m_unit : juce::String());
            }
            s << ", now " << juce::String(m_proc.value(id), 2) << "\n";
        }
        s << "\n";
    }
    return s;
}
