/**
 * @file ModDock.cpp
 * @brief The modulation dock and the source inspector (ModDock.h).
 */
#include "ModDock.h"

#include "KnobMod.h"
#include "Theme.h"

#include <algorithm>
#include <cmath>

using additiv::ModRow;
using additiv::ModSource;
using additiv::ParamId;
using knobmod::sourceColour;

namespace {

/** @brief The sources that always have a chip: the classic ones and the hands'. */
constexpr ModSource kClassic[] = { ModSource::Lfo1,   ModSource::Lfo2,   ModSource::Lfo3,   ModSource::Lfo4,     ModSource::Env1,     ModSource::Env2,
                                   ModSource::Macro1, ModSource::Macro2, ModSource::Macro3, ModSource::Macro4,   ModSource::ModWheel, ModSource::Pressure,
                                   ModSource::Velocity, ModSource::Random };

/** @brief Whether a source always has a chip. @param s the source @return true then */
bool classic(ModSource s) { return std::find(std::begin(kClassic), std::end(kClassic), s) != std::end(kClassic); }

constexpr int kToggleW = 112;   ///< the dock's button
constexpr int kPlusW = 30;      ///< the "+" chip
constexpr int kChipMin = 58;    ///< a chip's least width
constexpr int kChipMax = 86;    ///< its greatest

/**
 * @brief Draws a trace: bipolar about the middle, unipolar from the floor.
 * @param g      where
 * @param r      the area
 * @param dock   whose traces
 * @param s      the source
 * @param width  the line's width
 */
void drawTrace(juce::Graphics& g, juce::Rectangle<float> r, const ModDock& dock, ModSource s, float width)
{
    const bool uni = additiv::isUnipolarSource(s);
    auto yOf = [&](float v) {
        const float u = uni ? std::clamp(v, 0.0f, 1.0f) : 0.5f + 0.5f * std::clamp(v, -1.0f, 1.0f);
        return r.getBottom() - u * r.getHeight();
    };
    g.setColour(additivui::skin().faint);
    g.drawHorizontalLine(juce::roundToInt(yOf(0.0f)), r.getX(), r.getRight());
    juce::Path p;
    for (int k = 0; k < ModDock::kHistory; ++k) {
        const float x = r.getX() + r.getWidth() * static_cast<float>(k) / static_cast<float>(ModDock::kHistory - 1);
        const float y = yOf(dock.traceAt(s, k));
        if (k == 0) p.startNewSubPath(x, y);
        else p.lineTo(x, y);
    }
    g.setColour(sourceColour(s));
    g.strokePath(p, juce::PathStrokeType(width));
}

} // namespace

// ============================================================================================================ dock

ModDock::ModDock(AdditivProcessor& proc) : proc_(proc)
{
    for (auto& h : history_) h.fill(0.0f);
    toggle_.setButtonText("Inspector");
    toggle_.setTooltip("The sources' inspector: a source's trace, its knobs and where it goes");
    toggle_.onClick = [this] { if (onToggle) onToggle(); };
    addAndMakeVisible(toggle_);
    startTimerHz(30);
}

void ModDock::setChosen(ModSource s)
{
    chosen_ = s;
    repaint();
}

void ModDock::setOpen(bool open) { toggle_.setToggleState(open, juce::dontSendNotification); }

float ModDock::traceAt(ModSource s, int k) const
{
    const int i = static_cast<int>(s);
    if (i <= 0 || i >= additiv::kNumModSources) return 0.0f;
    return history_[static_cast<size_t>(i)][static_cast<size_t>((head_ + k) % kHistory)];
}

void ModDock::timerCallback()
{
    for (int s = 1; s < additiv::kNumModSources; ++s)
        history_[static_cast<size_t>(s)][static_cast<size_t>(head_)] = proc_.engine().sourceValue(static_cast<ModSource>(s));
    head_ = (head_ + 1) % kHistory;
    if (isShowing()) repaint();
}

int ModDock::rowsOf(ModSource s) const
{
    int n = 0;
    for (int i = 0; i < additiv::kModRows; ++i) {
        const ModRow r = proc_.engine().modRow(i);
        if (r.source == s && r.target >= 0) ++n;
    }
    return n;
}

std::vector<ModSource> ModDock::wanted() const
{
    std::vector<ModSource> out(std::begin(kClassic), std::end(kClassic));
    for (int k = 1; k < additiv::kNumModSources; ++k) {   // the others the matrix reads, in order
        const auto s = static_cast<ModSource>(k);
        if (classic(s)) continue;
        bool read = false;
        for (int i = 0; i < additiv::kModRows && !read; ++i) {
            const ModRow r = proc_.engine().modRow(i);
            read = (r.source == s || r.via == s) && r.target >= 0;
        }
        if (read) out.push_back(s);
    }
    if (pinned_ != ModSource::Off && std::find(out.begin(), out.end(), pinned_) == out.end()) out.push_back(pinned_);
    if (chosen_ != ModSource::Off && std::find(out.begin(), out.end(), chosen_) == out.end()) out.push_back(chosen_);
    return out;
}

juce::Rectangle<int> ModDock::chipArea() const { return getLocalBounds().reduced(0, 3).withTrimmedLeft(kToggleW + 8).withTrimmedRight(kPlusW + 6); }

juce::Rectangle<int> ModDock::chipBounds(size_t k, size_t n) const
{
    const auto a = chipArea();
    const int w = std::clamp(a.getWidth() / std::max<int>(1, static_cast<int>(n)), kChipMin, kChipMax);
    return { a.getX() + static_cast<int>(k) * w, a.getY(), w - 4, a.getHeight() };
}

juce::Rectangle<int> ModDock::plusBounds() const
{
    const size_t n = shown_.size();
    const auto last = n > 0 ? chipBounds(n - 1, n) : chipArea().withWidth(0);
    return { last.getRight() + 4, last.getY(), kPlusW, chipArea().getHeight() };
}

int ModDock::chipAt(juce::Point<int> p) const
{
    for (size_t k = 0; k < shown_.size(); ++k)
        if (chipBounds(k, shown_.size()).contains(p)) return static_cast<int>(k);
    return plusBounds().contains(p) ? -2 : -1;
}

void ModDock::resized() { toggle_.setBounds(getLocalBounds().removeFromLeft(kToggleW).reduced(0, 4)); }

void ModDock::paint(juce::Graphics& g)
{
    const frame::Skin& skin = additivui::skin();
    // As many chips as fit at their least width.
    std::vector<ModSource> want = wanted();
    const size_t fit = static_cast<size_t>(std::max(1, chipArea().getWidth() / kChipMin));
    if (want.size() > fit) want.resize(fit);
    shown_ = want;
    for (size_t k = 0; k < shown_.size(); ++k) {
        const ModSource s = shown_[k];
        const auto r = chipBounds(k, shown_.size()).toFloat();
        const juce::Colour c = sourceColour(s);
        const bool chosen = s == chosen_, hover = static_cast<int>(k) == hover_;
        g.setColour(skin.raised.withAlpha(hover ? 1.0f : 0.8f));
        g.fillRoundedRectangle(r, 5.0f);
        g.setColour(chosen ? skin.accent : skin.edge);
        g.drawRoundedRectangle(r.reduced(0.5f), 5.0f, chosen ? 1.6f : 1.0f);
        g.setColour(c);
        g.fillRoundedRectangle(r.getX() + 3.0f, r.getY() + 4.0f, 2.5f, r.getHeight() - 8.0f, 1.2f);
        g.setFont(juce::FontOptions(10.5f, rowsOf(s) > 0 ? juce::Font::bold : juce::Font::plain));
        g.setColour(rowsOf(s) > 0 ? skin.ink : skin.dim);
        g.drawFittedText(additiv::kModSourceNames[static_cast<int>(s)], r.reduced(9.0f, 2.0f).withHeight(13.0f).toNearestInt(), juce::Justification::centredLeft, 1, 0.8f);
        drawTrace(g, r.reduced(9.0f, 3.0f).withTrimmedTop(13.0f), *this, s, 1.1f);
    }
    const auto p = plusBounds().toFloat();
    g.setColour(skin.raised.withAlpha(hover_ == -2 ? 1.0f : 0.8f));
    g.fillRoundedRectangle(p, 5.0f);
    g.setColour(skin.edge);
    g.drawRoundedRectangle(p.reduced(0.5f), 5.0f, 1.0f);
    g.setColour(skin.ink);
    g.setFont(juce::FontOptions(16.0f));
    g.drawText("+", p, juce::Justification::centred);
}

void ModDock::mouseDown(const juce::MouseEvent& e)
{
    pressed_ = chipAt(e.getPosition());
    dragging_ = false;
}

void ModDock::mouseDrag(const juce::MouseEvent& e)
{
    if (dragging_ || pressed_ < 0 || static_cast<size_t>(pressed_) >= shown_.size() || e.getDistanceFromDragStart() < 5) return;
    auto* container = juce::DragAndDropContainer::findParentDragContainerFor(this);
    if (container == nullptr) return;
    dragging_ = true;
    const auto r = chipBounds(static_cast<size_t>(pressed_), shown_.size());
    container->startDragging(knobmod::dragOf(shown_[static_cast<size_t>(pressed_)]), this, juce::ScaledImage(createComponentSnapshot(r)), false);
}

void ModDock::mouseUp(const juce::MouseEvent& e)
{
    const int at = chipAt(e.getPosition());
    if (!dragging_ && at == pressed_) {
        if (at == -2) showMore();
        else if (at >= 0 && static_cast<size_t>(at) < shown_.size() && onChoose) onChoose(shown_[static_cast<size_t>(at)]);
    }
    pressed_ = -1;
    dragging_ = false;
}

void ModDock::mouseMove(const juce::MouseEvent& e)
{
    const int at = chipAt(e.getPosition());
    if (at != hover_) {
        hover_ = at;
        repaint();
    }
}

void ModDock::mouseExit(const juce::MouseEvent&)
{
    hover_ = -1;
    repaint();
}

juce::String ModDock::getTooltip()
{
    if (hover_ == -2) return "Every source: the one chosen gets a chip here";
    if (hover_ < 0 || static_cast<size_t>(hover_) >= shown_.size()) return {};
    const ModSource s = shown_[static_cast<size_t>(hover_)];
    const int n = rowsOf(s);
    return juce::String(additiv::kModSourceNames[static_cast<int>(s)]) + ", now " + juce::String(proc_.engine().sourceValue(s), 2)
           + (n > 0 ? ", moves " + juce::String(n) + (n == 1 ? " knob" : " knobs") : juce::String(", moves nothing yet"))
           + ". Drag it onto a knob to modulate it; click: its knobs and where it goes";
}

void ModDock::showMore()
{
    struct Section {
        const char* name;   ///< its heading
        ModSource first;    ///< its first source
        ModSource last;     ///< its last
    };
    static constexpr Section kSections[] = { { "Fields and chaos", ModSource::Lenia1, ModSource::Kura4 },
                                             { "The piece", ModSource::Beat, ModSource::RandomNote },
                                             { "What it hears", ModSource::Loudness, ModSource::Roughness },
                                             { "The reservoir", ModSource::Reservoir1, ModSource::Reservoir4 } };
    juce::PopupMenu m;
    for (const Section& sec : kSections) {
        m.addSectionHeader(sec.name);
        for (int k = static_cast<int>(sec.first); k <= static_cast<int>(sec.last); ++k)
            m.addItem(additiv::kModSourceNames[k], [this, k] {
                pinned_ = static_cast<ModSource>(k);
                if (onChoose) onChoose(pinned_);
                repaint();
            });
    }
    m.showMenuAsync(juce::PopupMenu::Options().withTargetComponent(this).withTargetScreenArea(localAreaToGlobal(plusBounds())));
}

// ======================================================================================================= inspector

SourceInspector::SourceInspector(AdditivProcessor& proc, const frame::ControlActions& actions, const ModDock& dock)
    : proc_(proc), actions_(actions), dock_(dock)
{
    hint_.setText("No knob listens to it yet: drag its chip onto a knob, or right-click a knob, Modulate with.", juce::dontSendNotification);
    hint_.setColour(juce::Label::textColourId, additivui::skin().dim);
    hint_.setFont(juce::FontOptions(12.0f));
    hint_.setJustificationType(juce::Justification::centredLeft);
    addChildComponent(hint_);
    close_.setTooltip("Shut the inspector");
    close_.onClick = [this] { if (onClose) onClose(); };
    addChildComponent(close_);
    matrix_.setTooltip("The matrix page: every row");
    matrix_.onClick = [this] { if (onMatrix) onMatrix(); };
    addAndMakeVisible(matrix_);
    show(ModSource::Lfo1);
    startTimerHz(30);
}

void SourceInspector::setOverlay(bool on)
{
    overlay_ = on;
    close_.setVisible(on);
    resized();
}

std::vector<ParamId> SourceInspector::ownParams(ModSource s)
{
    switch (s) {
        case ModSource::Lfo1: return { ParamId::Lfo1Shape, ParamId::Lfo1Rate };
        case ModSource::Lfo2: return { ParamId::Lfo2Shape, ParamId::Lfo2Rate };
        case ModSource::Lfo3: return { ParamId::Lfo3Shape, ParamId::Lfo3Rate };
        case ModSource::Lfo4: return { ParamId::Lfo4Shape, ParamId::Lfo4Rate };
        case ModSource::Env1: return { ParamId::Env1Shape, ParamId::Env1Rate };
        case ModSource::Env2: return { ParamId::Env2Shape, ParamId::Env2Rate };
        case ModSource::Macro1: return { ParamId::Macro1 };
        case ModSource::Macro2: return { ParamId::Macro2 };
        case ModSource::Macro3: return { ParamId::Macro3 };
        case ModSource::Macro4: return { ParamId::Macro4 };
        case ModSource::Lenia1: case ModSource::Lenia2: case ModSource::Lenia3: case ModSource::Lenia4:
            return { ParamId::LeniaRate, ParamId::LeniaGrowth };
        case ModSource::LorenzX: case ModSource::LorenzY: case ModSource::LorenzZ:
        case ModSource::RosslerX: case ModSource::RosslerY: case ModSource::RosslerZ:
            return { ParamId::ChaosPeriod };
        case ModSource::Kura1: case ModSource::Kura2: case ModSource::Kura3: case ModSource::Kura4:
            return { ParamId::Coherence, ParamId::CoherenceRate };
        case ModSource::Reservoir1: case ModSource::Reservoir2: case ModSource::Reservoir3: case ModSource::Reservoir4:
            return { ParamId::ReservoirChaos, ParamId::ReservoirMemory, ParamId::ReservoirInput };
        default: return {};
    }
}

ModSource SourceInspector::sourceOf(int id)
{
    for (int k = 1; k < additiv::kNumModSources; ++k)
        for (ParamId p : ownParams(static_cast<ModSource>(k)))
            if (static_cast<int>(p) == id) return static_cast<ModSource>(k);
    return ModSource::Off;
}

void SourceInspector::show(ModSource s)
{
    if (s == ModSource::Off) return;
    source_ = s;
    rebuildControls();
    rebuildRows();
    repaint();
}

void SourceInspector::rebuildControls()
{
    sliders_.clear();
    combos_.clear();
    controls_.clear();
    labels_.clear();
    const frame::Skin& skin = additivui::skin();
    for (ParamId pid : ownParams(source_)) {
        const int id = static_cast<int>(pid);
        juce::RangedAudioParameter* param = proc_.parameter(id);
        if (param == nullptr) continue;
        const additiv::ParamDesc& d = additiv::paramDesc(pid);
        auto label = std::make_unique<juce::Label>(juce::String(), d.name);
        label->setJustificationType(juce::Justification::centred);
        label->setColour(juce::Label::textColourId, skin.dim);
        label->setFont(juce::FontOptions(11.5f));
        label->setMinimumHorizontalScale(0.7f);
        addAndMakeVisible(*label);
        labels_.push_back(std::move(label));
        if (d.kind == additiv::ParamKind::Choice) {
            auto c = std::make_unique<frame::Choice>(&actions_, id);
            for (int i = 0; i < d.numChoices; ++i) c->addItem(d.choices[i], i + 1);
            c->setColour(juce::ComboBox::arrowColourId, sourceColour(source_));
            combos_.push_back(std::make_unique<juce::ComboBoxParameterAttachment>(*param, *c));
            addAndMakeVisible(*c);
            c->sendLookAndFeelChange();   // made before it had the frame's look
            controls_.push_back(std::move(c));
        } else {
            auto k = std::make_unique<frame::Knob>(juce::Slider::RotaryHorizontalVerticalDrag, juce::Slider::TextBoxBelow, &actions_, id);
            k->setTextBoxStyle(juce::Slider::TextBoxBelow, false, 70, 15);
            k->setColour(juce::Slider::rotarySliderFillColourId, sourceColour(source_));
            k->setTextValueSuffix(d.unit[0] != 0 ? juce::String(" ") + d.unit : juce::String());
            k->setDoubleClickReturnValue(true, d.def);
            sliders_.push_back(std::make_unique<juce::SliderParameterAttachment>(*param, *k));
            addAndMakeVisible(*k);
            k->sendLookAndFeelChange();   // its value box made under the frame's look
            controls_.push_back(std::move(k));
        }
    }
    resized();
}

juce::String SourceInspector::rowsKey() const
{
    juce::String k;
    for (int i = 0; i < additiv::kModRows; ++i) {
        const ModRow r = proc_.engine().modRow(i);
        if (r.source == source_ && r.target >= 0) k << i << ":" << r.target << ":" << static_cast<int>(r.via) << ";";
    }
    return k;
}

void SourceInspector::rebuildRows()
{
    rows_.clear();
    key_ = rowsKey();
    const juce::Colour c = sourceColour(source_);
    for (int i = 0; i < additiv::kModRows; ++i) {
        const ModRow r = proc_.engine().modRow(i);
        if (r.source != source_ || r.target < 0) continue;
        auto ui = std::make_unique<RowUi>();
        ui->row = i;
        juce::String name(additiv::paramDesc(static_cast<ParamId>(r.target)).name);
        if (r.via != ModSource::Off) name << " (x " << additiv::kModSourceNames[static_cast<int>(r.via)] << ")";
        ui->target.setButtonText(name);
        ui->target.setTooltip("Row " + juce::String(i + 1) + ": click to see the knob it moves");
        ui->target.onClick = [this, t = r.target] { if (onReveal) onReveal(t); };
        ui->depth.setSliderStyle(juce::Slider::LinearHorizontal);
        ui->depth.setTextBoxStyle(juce::Slider::TextBoxRight, false, 44, 18);
        ui->depth.setRange(-1.0, 1.0, 0.001);
        ui->depth.setValue(r.depth, juce::dontSendNotification);
        ui->depth.setDoubleClickReturnValue(true, 0.0);
        ui->depth.setColour(juce::Slider::trackColourId, c);
        ui->depth.setTooltip("How far it moves the knob, of the knob's whole range (negative: the other way)");
        ui->depth.onValueChange = [this, i, s = &ui->depth] {
            ModRow now = proc_.engine().modRow(i);
            now.depth = static_cast<float>(s->getValue());
            proc_.engine().setModRow(i, now);
        };
        ui->remove.setTooltip("Removes this row from the matrix");
        ui->remove.onClick = [this, i] {
            proc_.engine().setModRow(i, ModRow{});
            juce::MessageManager::callAsync([safe = juce::Component::SafePointer<SourceInspector>(this)] { if (safe != nullptr) safe->rebuildRows(); });
        };
        addAndMakeVisible(ui->target);
        addAndMakeVisible(ui->depth);
        addAndMakeVisible(ui->remove);
        ui->depth.sendLookAndFeelChange();
        rows_.push_back(std::move(ui));
    }
    hint_.setVisible(rows_.empty());
    resized();
}

juce::Rectangle<int> SourceInspector::traceArea() const { return { 14, 30, std::min(260, getWidth() / 4), getHeight() - 40 }; }

void SourceInspector::resized()
{
    const auto trace = traceArea();
    close_.setBounds(getWidth() - 30, 4, 24, 20);
    matrix_.setBounds((overlay_ ? close_.getX() : getWidth()) - 92, 4, 84, 20);
    // The source's own knobs and choosers, after the trace.
    int x = trace.getRight() + 16;
    const int top = 26;
    for (size_t i = 0; i < controls_.size(); ++i) {
        const bool choice = dynamic_cast<juce::ComboBox*>(controls_[i].get()) != nullptr;
        const int w = choice ? 124 : 76;
        labels_[i]->setBounds(x, top, w, 14);
        if (choice) controls_[i]->setBounds(x + 4, top + 30, w - 8, 24);
        else controls_[i]->setBounds(x, top + 14, w, getHeight() - top - 18);
        x += w;
    }
    // Its rows, in columns of three.
    x += controls_.empty() ? 0 : 16;
    const int colW = 330, rowH = 26;
    hint_.setBounds(x, top + 10, std::max(0, getWidth() - x - 12), 40);
    for (size_t i = 0; i < rows_.size(); ++i) {
        const int cx = x + static_cast<int>(i / 3) * (colW + 10), cy = top + 4 + static_cast<int>(i % 3) * rowH;
        RowUi& r = *rows_[i];
        const bool fits = cx + colW <= getWidth() - 8;
        r.target.setVisible(fits);
        r.depth.setVisible(fits);
        r.remove.setVisible(fits);
        r.target.setBounds(cx, cy, 140, rowH - 4);
        r.depth.setBounds(cx + 144, cy, colW - 144 - 28, rowH - 4);
        r.remove.setBounds(cx + colW - 24, cy, 22, rowH - 4);
    }
}

void SourceInspector::paint(juce::Graphics& g)
{
    const frame::Skin& skin = additivui::skin();
    const auto b = getLocalBounds().toFloat();
    g.setColour(overlay_ ? skin.group : skin.group.withAlpha(0.92f));
    g.fillRoundedRectangle(b, skin.radius);
    g.setColour(overlay_ ? skin.accent.withAlpha(0.6f) : skin.edge);
    g.drawRoundedRectangle(b.reduced(0.5f), skin.radius, 1.0f);
    const juce::Colour c = sourceColour(source_);
    g.setColour(c);
    g.fillRoundedRectangle(8.0f, 9.0f, 3.0f, 10.0f, 1.5f);
    g.setFont(juce::FontOptions(13.0f, juce::Font::bold));
    juce::String title(additiv::kModSourceNames[static_cast<int>(source_)]);
    title << "   " << juce::String(proc_.engine().sourceValue(source_), 2);
    const int n = static_cast<int>(rows_.size());
    title << "   " << (n == 0 ? juce::String("moves nothing") : "moves " + juce::String(n) + (n == 1 ? " knob" : " knobs"));
    g.drawText(title, 17, 4, getWidth() - 200, 20, juce::Justification::centredLeft);
    const auto t = traceArea().toFloat();
    g.setColour(skin.raised.withAlpha(0.55f));
    g.fillRoundedRectangle(t, 4.0f);
    drawTrace(g, t.reduced(6.0f, 6.0f), dock_, source_, 1.6f);
}

void SourceInspector::timerCallback()
{
    if (!isShowing()) return;
    if (rowsKey() != key_) rebuildRows();
    for (auto& r : rows_)   // depths changed elsewhere (the matrix page, a knob's menu)
        if (!r->depth.isMouseButtonDown()) r->depth.setValue(proc_.engine().modRow(r->row).depth, juce::dontSendNotification);
    repaint(getLocalBounds().withHeight(26));
    repaint(traceArea());
}
