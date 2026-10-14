/**
 * @file FocusView.cpp
 * @brief The Focus page (FocusView.h).
 */
#include "FocusView.h"

#include "Theme.h"

#include "additiv/Focus.h"
#include "additiv/Presets.h"

#include <algorithm>
#include <cmath>

using additiv::ParamId;

namespace {

constexpr int kTitle = 26;      ///< a box's title row
constexpr int kGap = 10;        ///< between the boxes
constexpr int kNudgeH = 92;     ///< the nudges' box
constexpr int kVaryH = 176;     ///< the variations' box
constexpr int kTeachH = 160;    ///< the teaching's box
constexpr int kFindH = 132;     ///< the search's box
const char* const kAxis[additiv::kNudgeAxes] = { "brightness", "motion", "width", "air" };   ///< what each pair moves

} // namespace

void VariationTile::paint(juce::Graphics& g)
{
    const frame::Skin& skin = additivui::skin();
    const bool listening = proc_.previewing() == k_;
    const auto r = getLocalBounds().toFloat().reduced(1.0f);
    g.setColour(listening ? skin.accent.withAlpha(0.22f) : skin.raised);
    g.fillRoundedRectangle(r, skin.radius);
    g.setColour(listening ? skin.accent : skin.edge);
    g.drawRoundedRectangle(r, skin.radius, 1.0f);
    if (k_ >= proc_.variationCount()) return;
    g.setColour(skin.accent);
    g.setFont(juce::FontOptions(18.0f, juce::Font::bold));
    g.drawText(juce::String(k_ + 1), r.withTrimmedLeft(8.0f).withHeight(24.0f), juce::Justification::centredLeft);
    g.setColour(skin.dim);
    g.setFont(juce::FontOptions(11.0f));
    g.drawFittedText(proc_.variationSummary(k_), r.reduced(8.0f, 4.0f).withTrimmedTop(20.0f).toNearestInt(), juce::Justification::topLeft, 3);
}

void VariationTile::mouseDown(const juce::MouseEvent&)
{
    proc_.previewVariation(k_);
    repaint();
}

void VariationTile::mouseUp(const juce::MouseEvent&)
{
    proc_.previewVariation(-1);
    repaint();
}

FocusPage::FocusPage(AdditivProcessor& proc, const frame::ControlActions& actions, frame::LiveRings* live, Undoable undoable)
    : proc_(proc), actions_(actions), live_(live), undoable_(std::move(undoable))
{
    const frame::Skin& skin = additivui::skin();
    for (int a = 0; a < additiv::kNudgeAxes; ++a) {
        juce::TextButton& lo = lower_[static_cast<size_t>(a)];
        juce::TextButton& hi = higher_[static_cast<size_t>(a)];
        lo.setButtonText(juce::String("< ") + additiv::kNudgeLow[a]);
        hi.setButtonText(juce::String(additiv::kNudgeHigh[a]) + " >");
        lo.setTooltip(juce::String("A little ") + juce::String(additiv::kNudgeLow[a]).toLowerCase() + ": every knob that is heard moves the way it was measured to make it so, the level held");
        hi.setTooltip(juce::String("A little ") + juce::String(additiv::kNudgeHigh[a]).toLowerCase() + ": every knob that is heard moves the way it was measured to make it so, the level held");
        for (int s = 0; s < 2; ++s) {
            juce::TextButton& b = s == 0 ? lo : hi;
            const float sign = s == 0 ? -1.0f : 1.0f;
            const juce::String name = s == 0 ? additiv::kNudgeLow[a] : additiv::kNudgeHigh[a];
            b.onClick = [this, a, sign, name] {
                float got = 0.0f;
                undoable_("Nudge: " + name, [this, a, sign, &got] { got = proc_.nudgeSound(a, sign); });
                if (a == 1)   // calmer and livelier go by a rule (Focus.h)
                    nudgeStatus_.setText(name + juce::String(sign < 0.0f ? ": every rate that moves something 1.23 times slower, drift and shimmer a little less"
                                                                       : ": every rate that moves something 1.23 times faster, drift and shimmer a little more"),
                                         juce::dontSendNotification);
                else
                    nudgeStatus_.setText(std::fabs(got) < 0.02f ? name + ": nothing that is heard here moves that way"
                                                                  : name + ": " + juce::String(std::fabs(got), 2) + " of the bank's spread in " + kAxis[a] + " (as measured)",
                                         juce::dontSendNotification);
            };
            addAndMakeVisible(b);
        }
    }
    nudgeStatus_.setColour(juce::Label::textColourId, skin.dim);
    nudgeStatus_.setFont(juce::FontOptions(12.0f));
    nudgeStatus_.setText("Each press moves the sound a little; undo takes it back.", juce::dontSendNotification);
    addAndMakeVisible(nudgeStatus_);

    make_.setTooltip("Six variations near the sound now: hold a tile to listen, Keep to take it (the next six are then made around it)");
    make_.onClick = [this] {
        proc_.makeVariations(static_cast<float>(far_.getValue()));
        for (auto& t : tiles_) t->repaint();
    };
    back_.setTooltip("Back to the sound before the last variation kept");
    back_.onClick = [this] {
        undoable_("Variation back", [this] { proc_.variationBack(); });
        for (auto& t : tiles_) t->repaint();
    };
    far_.setSliderStyle(juce::Slider::LinearHorizontal);
    far_.setTextBoxStyle(juce::Slider::NoTextBox, true, 0, 0);
    far_.setRange(0.05, 1.0, 0.01);
    far_.setValue(0.3, juce::dontSendNotification);
    far_.setTooltip("How far the variations go from the sound now");
    farName_.setText("near ... far", juce::dontSendNotification);
    farName_.setColour(juce::Label::textColourId, skin.dim);
    farName_.setFont(juce::FontOptions(12.0f));
    for (juce::Component* c : { static_cast<juce::Component*>(&make_), static_cast<juce::Component*>(&back_), static_cast<juce::Component*>(&far_),
                                static_cast<juce::Component*>(&farName_) })
        addAndMakeVisible(c);
    for (int k = 0; k < 6; ++k) {
        tiles_.push_back(std::make_unique<VariationTile>(proc_, k));
        addAndMakeVisible(*tiles_.back());
        juce::TextButton& keep = keep_[static_cast<size_t>(k)];
        keep.setButtonText("Keep");
        keep.setTooltip("This variation becomes the sound; six new ones are made around it");
        keep.onClick = [this, k] {
            undoable_("Variation " + juce::String(k + 1), [this, k] { proc_.takeVariation(k); });
            for (auto& t : tiles_) t->repaint();
        };
        addAndMakeVisible(keep);
    }

    for (int m = 0; m < 4; ++m) {
        const size_t i = static_cast<size_t>(m);
        macroName_[i].setText("Macro " + juce::String(m + 1), juce::dontSendNotification);
        macroName_[i].setColour(juce::Label::textColourId, skin.ink);
        macroName_[i].setFont(juce::FontOptions(13.0f, juce::Font::bold));
        macro_[i].setSliderStyle(juce::Slider::LinearHorizontal);
        macro_[i].setTextBoxStyle(juce::Slider::TextBoxRight, false, 44, 18);
        if (auto* p = proc_.parameter(static_cast<int>(ParamId::Macro1) + m)) macroLinks_.push_back(std::make_unique<juce::SliderParameterAttachment>(*p, macro_[i]));
        teach_[i].setButtonText("Teach");
        teach_[i].setTooltip("Begin: the sound now is this macro's place at its value now. Then move the knobs, set the macro, and press Here");
        teach_[i].onClick = [this, m] { proc_.teachBegin(m); };
        here_[i].setButtonText("Here");
        here_[i].setTooltip("The sound now becomes this macro's place at its value now; the knobs go back, the macro carries it");
        here_[i].onClick = [this, m] { proc_.teachHere(m); };
        forget_[i].setButtonText("Forget");
        forget_[i].setTooltip("Forgets what this macro was taught (its matrix rows stay)");
        forget_[i].onClick = [this, m] { proc_.teachForget(m); };
        teachStatus_[i].setColour(juce::Label::textColourId, skin.dim);
        teachStatus_[i].setFont(juce::FontOptions(12.0f));
        for (juce::Component* c : { static_cast<juce::Component*>(&macroName_[i]), static_cast<juce::Component*>(&macro_[i]), static_cast<juce::Component*>(&teach_[i]),
                                    static_cast<juce::Component*>(&here_[i]), static_cast<juce::Component*>(&forget_[i]), static_cast<juce::Component*>(&teachStatus_[i]) })
            addAndMakeVisible(c);
    }

    open_.setTooltip("A recording (WAV or FLAC): the presets that sound most like it -- or drop one on this page");
    open_.onClick = [this] {
        chooser_ = std::make_unique<juce::FileChooser>("Find the presets that sound like a recording", juce::File::getSpecialLocation(juce::File::userMusicDirectory),
                                                       "*.wav;*.flac");
        chooser_->launchAsync(juce::FileBrowserComponent::openMode | juce::FileBrowserComponent::canSelectFiles, [this](const juce::FileChooser& fc) {
            if (fc.getResult() != juce::File()) find(fc.getResult());
        });
    };
    findStatus_.setColour(juce::Label::textColourId, skin.dim);
    findStatus_.setFont(juce::FontOptions(12.0f));
    findStatus_.setText("Open or drop a recording: its brightness, motion, width and noise are measured as the presets' were.", juce::dontSendNotification);
    addAndMakeVisible(open_);
    addAndMakeVisible(findStatus_);
    for (size_t i = 0; i < results_.size(); ++i) {
        results_[i].onClick = [this, i] {
            if (i >= proc_.found().size()) return;
            const int index = proc_.found()[i];
            undoable_("Preset", [this, index] { proc_.choosePreset(index); });
        };
        addChildComponent(results_[i]);
    }

    rebuildFocus({});
    timerCallback();
    startTimerHz(4);
}

void FocusPage::rebuildFocus(const std::vector<ParamId>& ids)
{
    focusIds_ = ids;
    if (focus_ != nullptr) removeChildComponent(focus_.get());
    std::vector<ParamPage::Group> groups;
    groups.push_back({ "Focus: the knobs that do the most for this sound", frame::Family::Source, ids });
    focus_ = std::make_unique<ParamPage>(proc_, actions_, std::move(groups), live_);
    addAndMakeVisible(*focus_);
    focus_->sendLookAndFeelChange();   // made after the page joined the editor: the frame's look for its knobs too
    resized();
}

void FocusPage::timerCallback()
{
    if (!isShowing() && focus_ != nullptr && !focusIds_.empty()) return;
    // The focus anew when what matters changed (a preset, a gate opened or closed).
    if (tick_++ % 4 == 0) {
        const std::vector<float> v = proc_.knobValues();
        ParamId ids[8];
        const int n = additiv::focusParams(v.data(), ids, 8);
        std::vector<ParamId> now(ids, ids + n);
        std::vector<ParamId> a = now, b = focusIds_;
        std::sort(a.begin(), a.end());
        std::sort(b.begin(), b.end());
        if (a != b) rebuildFocus(now);
    }
    back_.setEnabled(true);
    for (size_t k = 0; k < keep_.size(); ++k) keep_[k].setEnabled(static_cast<int>(k) < proc_.variationCount());
    for (int m = 0; m < 4; ++m) {
        const juce::String s = proc_.teachStatus(m);
        if (teachStatus_[static_cast<size_t>(m)].getText() != s) teachStatus_[static_cast<size_t>(m)].setText(s, juce::dontSendNotification);
        here_[static_cast<size_t>(m)].setEnabled(proc_.teaching(m));
    }
}

void FocusPage::find(const juce::File& f)
{
    juce::String why;
    if (!proc_.findLike(f, why)) {
        findStatus_.setText(f.getFileName() + ": " + why, juce::dontSendNotification);
        return;
    }
    findStatus_.setText(f.getFileName() + ": the presets nearest it, nearest first (ringed on the map)", juce::dontSendNotification);
    showFound();
}

void FocusPage::showFound()
{
    const auto& found = proc_.found();
    for (size_t i = 0; i < results_.size(); ++i) {
        const bool has = i < found.size();
        results_[i].setVisible(has);
        if (has) {
            const additiv::Preset& p = additiv::preset(found[i]);
            results_[i].setButtonText(juce::String(p.name) + "  (" + additiv::kPresetGroupNames[p.group] + ")");
            results_[i].setTooltip("Distance " + juce::String(proc_.foundDistance()[i], 2) + " (in the bank's spreads): click to play it");
        }
    }
}

bool FocusPage::isInterestedInFileDrag(const juce::StringArray& files)
{
    return files.size() == 1 && (files[0].endsWithIgnoreCase(".wav") || files[0].endsWithIgnoreCase(".flac"));
}

void FocusPage::filesDropped(const juce::StringArray& files, int, int)
{
    if (files.size() == 1) find(juce::File(files[0]));
}

std::array<juce::Rectangle<int>, 5> FocusPage::boxes(int w) const
{
    std::array<juce::Rectangle<int>, 5> r;
    int y = kGap;
    r[0] = { kGap, y, w - 2 * kGap, kNudgeH };
    y += kNudgeH;
    const int focusH = focus_ != nullptr ? focus_->heightFor(w) : 0;
    r[1] = { 0, y, w, focusH };
    y += focusH;
    r[2] = { kGap, y, w - 2 * kGap, kVaryH };
    y += kVaryH + kGap;
    r[3] = { kGap, y, w - 2 * kGap, kTeachH };
    y += kTeachH + kGap;
    r[4] = { kGap, y, w - 2 * kGap, kFindH };
    return r;
}

int FocusPage::heightFor(int w) const { return boxes(w)[4].getBottom() + kGap; }

void FocusPage::resized()
{
    const auto r = boxes(getWidth());
    {   // the nudges: four pairs in a row
        auto row = r[0].reduced(12, 0).withTrimmedTop(kTitle).withHeight(30);
        const int pair = row.getWidth() / additiv::kNudgeAxes;
        for (int a = 0; a < additiv::kNudgeAxes; ++a) {
            auto cell = row.removeFromLeft(pair).reduced(4, 0);
            lower_[static_cast<size_t>(a)].setBounds(cell.removeFromLeft(cell.getWidth() / 2).reduced(2, 1));
            higher_[static_cast<size_t>(a)].setBounds(cell.reduced(2, 1));
        }
        nudgeStatus_.setBounds(r[0].getX() + 12, r[0].getY() + kTitle + 34, r[0].getWidth() - 24, 20);
    }
    if (focus_ != nullptr) focus_->setBounds(r[1]);
    {   // the variations
        auto area = r[2].reduced(12, 0).withTrimmedTop(kTitle);
        auto top = area.removeFromTop(28);
        make_.setBounds(top.removeFromLeft(110).reduced(2));
        back_.setBounds(top.removeFromLeft(80).reduced(2));
        top.removeFromLeft(12);
        farName_.setBounds(top.removeFromLeft(90));
        far_.setBounds(top.removeFromLeft(220).reduced(0, 4));
        area.removeFromTop(6);
        const int w = area.getWidth() / 6;
        for (int k = 0; k < 6; ++k) {
            auto cell = area.withX(area.getX() + k * w).withWidth(w).reduced(4, 0);
            tiles_[static_cast<size_t>(k)]->setBounds(cell.withHeight(76));
            keep_[static_cast<size_t>(k)].setBounds(cell.withTrimmedTop(80).withHeight(24));
        }
    }
    {   // the teaching
        auto area = r[3].reduced(12, 0).withTrimmedTop(kTitle);
        for (int m = 0; m < 4; ++m) {
            const size_t i = static_cast<size_t>(m);
            auto row = area.removeFromTop(30);
            macroName_[i].setBounds(row.removeFromLeft(80));
            macro_[i].setBounds(row.removeFromLeft(220).reduced(0, 4));
            row.removeFromLeft(8);
            teach_[i].setBounds(row.removeFromLeft(70).reduced(2));
            here_[i].setBounds(row.removeFromLeft(70).reduced(2));
            forget_[i].setBounds(row.removeFromLeft(70).reduced(2));
            row.removeFromLeft(8);
            teachStatus_[i].setBounds(row);
        }
    }
    {   // the search
        auto area = r[4].reduced(12, 0).withTrimmedTop(kTitle);
        auto top = area.removeFromTop(28);
        open_.setBounds(top.removeFromLeft(170).reduced(2));
        top.removeFromLeft(8);
        findStatus_.setBounds(top);
        area.removeFromTop(6);
        const int w = area.getWidth() / 4;
        for (size_t i = 0; i < results_.size(); ++i)
            results_[i].setBounds(area.getX() + static_cast<int>(i % 4) * w + 2, area.getY() + static_cast<int>(i / 4) * 30, w - 4, 26);
    }
}

void FocusPage::paint(juce::Graphics& g)
{
    const frame::Skin& skin = additivui::skin();
    const auto r = boxes(getWidth());
    const char* const titles[5] = { "Nudge: a little more of one thing, the level held", nullptr,
                                    "Variations: hold a tile to listen, Keep to take it", "Teach the macros by example: Teach, move the knobs, set the macro, Here",
                                    "Find by example: the presets that sound like a recording" };
    const frame::Family families[5] = { frame::Family::Source, frame::Family::Source, frame::Family::Motion, frame::Family::Envelope, frame::Family::Space };
    for (size_t i = 0; i < r.size(); ++i) {
        if (titles[i] == nullptr) continue;
        const auto b = r[i].toFloat();
        g.setColour(skin.group.withAlpha(0.92f));
        g.fillRoundedRectangle(b, skin.radius);
        g.setColour(skin.edge);
        g.drawRoundedRectangle(b.reduced(0.5f), skin.radius, 1.0f);
        g.setColour(skin.family(families[i]));
        g.fillRoundedRectangle(b.getX() + 8.0f, b.getY() + 9.0f, 3.0f, 10.0f, 1.5f);
        g.setFont(juce::FontOptions(13.0f, juce::Font::bold));
        g.drawText(titles[i], r[i].withHeight(kTitle).withTrimmedLeft(17), juce::Justification::centredLeft);
    }
}

juce::String FocusPage::describe() const
{
    juce::String s("Focus\n");
    s << "    Nudge: " << nudgeStatus_.getText() << "\n";
    s << "    Focus: ";
    for (size_t i = 0; i < focusIds_.size(); ++i) s << (i > 0 ? ", " : "") << additiv::paramDesc(focusIds_[i]).name;
    s << "\n    Variations: " << proc_.variationCount() << "\n";
    for (int k = 0; k < proc_.variationCount(); ++k) s << "        " << (k + 1) << ". " << proc_.variationSummary(k) << "\n";
    for (int m = 0; m < 4; ++m) s << "    Macro " << (m + 1) << ": " << proc_.teachStatus(m) << "\n";
    s << "    Find: " << findStatus_.getText() << "\n";
    for (size_t i = 0; i < proc_.found().size(); ++i) s << "        " << additiv::preset(proc_.found()[i]).name << "\n";
    return s + "\nThe simple ways in. Nudge moves every knob that is heard a little, each the way it was measured to make the sound "
               "brighter, livelier, wider or airier (or the other way), the level held. Focus shows the few knobs that do the most "
               "for this sound. Variations are six near the sound: hold a tile to listen, Keep to take it, and the next six are made "
               "around it. Teach lets a macro learn a gesture by example. Find measures a recording and finds the presets that sound "
               "most like it.\n";
}
