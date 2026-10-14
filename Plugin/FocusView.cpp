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
constexpr int kSnapH = 104;     ///< the snapshots' box
constexpr int kVaryH = 176;     ///< the variations' box
constexpr int kTeachH = 214;    ///< the teaching's box: four columns of a name, a large knob, three buttons, a status
constexpr int kFindH = 132;     ///< the search's box
// Two columns, where the page is wide enough: what changes the sound on the left (nudge, focus, teach),
// what keeps and finds sounds on the right (snapshots, variations, find).
constexpr int kLeftMin = 680;   ///< the left column's least width: the focus box's eight knobs
constexpr int kRightMin = 440;  ///< the right column's least width
constexpr int kSnapTwoH = 108;  ///< the snapshots' box in the right column: the tiles in two rows of four
constexpr int kVaryTwoH = 206;  ///< the variations' box there: the tiles in two rows of three
constexpr int kFindTwoH = 158;  ///< the search's box there: the presets found in three columns

/** @brief The page is wide enough for two columns. @param w its width @return true for two */
bool twoColumns(int w) { return w - 2 * kGap >= kLeftMin + kGap + kRightMin; }
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
    const bool low = r.getHeight() < 60.0f;   // a low tile: the number beside what it changes, not above it
    g.drawText(juce::String(k_ + 1), r.withTrimmedLeft(8.0f).withHeight(24.0f), juce::Justification::centredLeft);
    g.setColour(skin.dim);
    g.setFont(juce::FontOptions(11.0f));
    g.drawFittedText(proc_.variationSummary(k_), (low ? r.reduced(4.0f, 3.0f).withTrimmedLeft(24.0f) : r.reduced(8.0f, 4.0f).withTrimmedTop(20.0f)).toNearestInt(),
                     juce::Justification::topLeft, low ? 2 : 3);
}

void VariationTile::mouseDown(const juce::MouseEvent&)
{
    if (k_ >= proc_.variationCount()) return;
    if (onHeard) onHeard(k_);
    proc_.previewVariation(k_);
    repaint();
}

void VariationTile::mouseUp(const juce::MouseEvent&)
{
    proc_.previewVariation(-1);
    repaint();
}

/** @brief A snapshot's tile (see FocusPage). */
class FocusPage::SnapshotTile final : public juce::Component, public juce::SettableTooltipClient {
public:
    /** @brief Tile @p k of @p page. @param page the page @param k 0 .. 7 */
    SnapshotTile(FocusPage& page, int k) : page_(page), k_(k)
    {
        setTooltip("A snapshot: click to come back to it (the sound glides there); right-click: into the morph's slot A or B, or forget it");
    }
    /** @brief What the tile does now: keep the sound (the +), come back to one, or nothing yet. @return the text */
    juce::String getTooltip() override
    {
        const int count = static_cast<int>(page_.proc_.snapshots().size());
        if (k_ == count) return "Keep the sound now -- every knob and the matrix, with its picture";
        return k_ < count ? juce::SettableTooltipClient::getTooltip() : juce::String();
    }
    /** @brief The + lights up under the mouse. @param e ignored */
    void mouseEnter(const juce::MouseEvent&) override { repaint(); }
    /** @brief And goes dim again. @param e ignored */
    void mouseExit(const juce::MouseEvent&) override { repaint(); }
    /** @brief Its picture and number, or an empty frame. */
    void paint(juce::Graphics& g) override
    {
        const frame::Skin& skin = additivui::skin();
        const auto r = getLocalBounds().toFloat().reduced(1.0f);
        const auto& shots = page_.proc_.snapshots();
        const int count = static_cast<int>(shots.size());
        const bool has = k_ < count;
        if (k_ == count) {   // the first free one: a + that keeps the sound
            g.setColour(skin.accent.withAlpha(isMouseOver() ? 0.16f : 0.07f));
            g.fillRoundedRectangle(r, skin.radius);
            g.setColour(skin.accent.withAlpha(0.7f));
            const float dash[2] = { 4.0f, 3.0f };
            juce::Path outline;
            outline.addRoundedRectangle(r, skin.radius);
            juce::Path dashed;
            juce::PathStrokeType(1.0f).createDashedStroke(dashed, outline, dash, 2);
            g.fillPath(dashed);
            g.setFont(juce::FontOptions(r.getHeight() < 40.0f ? 12.0f : 14.0f, juce::Font::bold));
            g.drawText(r.getWidth() > 90.0f ? "+ Snapshot" : "+", r, juce::Justification::centred);
            return;
        }
        if (!has) {   // further free ones: barely there until they are filled
            g.setColour(skin.faint.withAlpha(0.35f));
            g.drawRoundedRectangle(r, skin.radius, 1.0f);
            return;
        }
        g.setColour(skin.raised);
        g.fillRoundedRectangle(r, skin.radius);
        if (shots[static_cast<size_t>(k_)].picture.isValid())
            g.drawImage(shots[static_cast<size_t>(k_)].picture, r.reduced(3.0f), juce::RectanglePlacement::stretchToFit);
        g.setColour(skin.accent);
        g.drawRoundedRectangle(r, skin.radius, 1.2f);
        g.setFont(juce::FontOptions(12.0f, juce::Font::bold));
        g.drawText(juce::String(k_ + 1), r.reduced(6.0f, 3.0f), juce::Justification::topLeft);
    }
    /** @brief A click recalls it; a right-click offers the morph's slots and forgetting. @param e the click */
    void mouseUp(const juce::MouseEvent& e) override
    {
        const int count = static_cast<int>(page_.proc_.snapshots().size());
        if (k_ == count && !e.mods.isPopupMenu()) {   // the +
            page_.proc_.takeSnapshot(page_.picture_ ? page_.picture_() : juce::Image());
            for (auto& t : page_.snapTiles_) t->repaint();
            return;
        }
        if (k_ >= count) return;
        if (e.mods.isPopupMenu()) {
            juce::PopupMenu m;
            m.addItem("Into the morph's slot A", [this] { page_.proc_.snapshotToSlot(k_, 0); });
            m.addItem("Into the morph's slot B", [this] { page_.proc_.snapshotToSlot(k_, 1); });
            m.addSeparator();
            m.addItem("Forget it", [this] {
                page_.proc_.forgetSnapshot(k_);
                for (auto& t : page_.snapTiles_) t->repaint();
            });
            m.showMenuAsync(juce::PopupMenu::Options().withTargetComponent(this));
            return;
        }
        page_.undoable_("Snapshot " + juce::String(k_ + 1), [this] { page_.proc_.recallSnapshot(k_); });
    }

private:
    FocusPage& page_;   ///< whose
    int k_;             ///< which
};

FocusPage::FocusPage(AdditivProcessor& proc, const frame::ControlActions& actions, frame::LiveRings* live, Undoable undoable,
                     std::function<juce::Image()> picture)
    : proc_(proc), actions_(actions), live_(live), undoable_(std::move(undoable)), picture_(std::move(picture))
{
    for (int k = 0; k < 8; ++k) {
        snapTiles_.push_back(std::make_unique<SnapshotTile>(*this, k));
        addAndMakeVisible(*snapTiles_.back());
    }
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

    make_.setButtonText("New six");
    make_.setTooltip("Six other variations near the sound now (they are made on their own whenever the sound moves): hold a tile to listen, Keep to take it");
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
        tiles_.back()->onHeard = [this](int which) {
            heard_[static_cast<size_t>(which)] = true;
            keep_[static_cast<size_t>(which)].setVisible(true);
        };
        addAndMakeVisible(*tiles_.back());
        juce::TextButton& keep = keep_[static_cast<size_t>(k)];
        keep.setButtonText("Keep");
        keep.setTooltip("This variation becomes the sound; six new ones are made around it");
        keep.onClick = [this, k] {
            undoable_("Variation " + juce::String(k + 1), [this, k] { proc_.takeVariation(k); });
            for (auto& t : tiles_) t->repaint();
        };
        addChildComponent(keep);   // shown once its tile was heard
    }

    for (int m = 0; m < 4; ++m) {
        const size_t i = static_cast<size_t>(m);
        macroName_[i].setText(proc_.macroName(m), juce::dontSendNotification);
        macroName_[i].setColour(juce::Label::textColourId, skin.family(frame::Family::Envelope));
        macroName_[i].setFont(juce::FontOptions(14.0f, juce::Font::bold));
        macroName_[i].setJustificationType(juce::Justification::centred);
        macroName_[i].setMinimumHorizontalScale(0.7f);
        const int id = static_cast<int>(ParamId::Macro1) + m;
        auto knob = std::make_unique<frame::Knob>(juce::Slider::RotaryHorizontalVerticalDrag, juce::Slider::TextBoxBelow, &actions_, id);
        knob->setTextBoxStyle(juce::Slider::TextBoxBelow, false, 70, 16);
        knob->setColour(juce::Slider::rotarySliderFillColourId, skin.family(frame::Family::Envelope));
        knob->setTooltip("Macro " + juce::String(m + 1) + ": what the matrix and its lesson let it move");
        if (auto* p = proc_.parameter(id)) macroLinks_.push_back(std::make_unique<juce::SliderParameterAttachment>(*p, *knob));
        if (live_ != nullptr) live_->add(*knob, id);
        macro_.push_back(std::move(knob));
        juce::Component* macroKnob = macro_.back().get();
        teach_[i].setButtonText("Teach");
        teach_[i].setTooltip("Begin: the sound now is this macro's place at its value now. Then move the knobs, set the macro, and press Set here");
        teach_[i].onClick = [this, m] {
            proc_.teachBegin(m);
            timerCallback();
        };
        here_[i].setButtonText("Set here");
        here_[i].setTooltip("The sound now becomes this macro's place at its value now; the knobs go back, the macro carries it");
        here_[i].onClick = [this, m] {
            proc_.teachHere(m);
            timerCallback();
        };
        cancel_[i].setButtonText("Cancel");
        cancel_[i].setTooltip("Ends the teaching: the knobs go back where it began, the macro keeps what it had");
        cancel_[i].onClick = [this, m] {
            proc_.teachCancel(m);
            timerCallback();
        };
        forget_[i].setButtonText("x");
        forget_[i].setTooltip("Forgets what this macro was taught (its matrix rows stay)");
        forget_[i].onClick = [this, m] {
            proc_.teachForget(m);
            timerCallback();
        };
        teachStatus_[i].setColour(juce::Label::textColourId, skin.dim);
        teachStatus_[i].setFont(juce::FontOptions(12.0f));
        teachStatus_[i].setJustificationType(juce::Justification::centredTop);
        teachStatus_[i].setFont(juce::FontOptions(11.0f));
        for (juce::Component* c : { static_cast<juce::Component*>(&macroName_[i]), macroKnob, static_cast<juce::Component*>(&teach_[i]),
                                    static_cast<juce::Component*>(&teachStatus_[i]) })
            addAndMakeVisible(c);
        for (juce::Component* c : { static_cast<juce::Component*>(&here_[i]), static_cast<juce::Component*>(&cancel_[i]), static_cast<juce::Component*>(&forget_[i]) })
            addChildComponent(c);
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
    startTimerHz(15);
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
    const bool slow = tick_ % 4 == 0;   // four times a second; the teaching's pulse at fifteen
    // The focus anew when what matters changed (a preset, a gate opened or closed).
    if (tick_++ % 16 == 0) {
        const std::vector<float> v = proc_.knobValues();
        ParamId ids[8];
        const int n = additiv::focusParams(v.data(), ids, 8);
        std::vector<ParamId> now(ids, ids + n);
        std::vector<ParamId> a = now, b = focusIds_;
        std::sort(a.begin(), a.end());
        std::sort(b.begin(), b.end());
        if (a != b) rebuildFocus(now);
    }
    // The teaching's pulse.
    bool anyTeaching = false;
    for (int m = 0; m < 4; ++m) anyTeaching = anyTeaching || proc_.teaching(m);
    if (anyTeaching) {
        pulse_ += 1.0 / 15.0;
        repaint(boxes(getWidth())[4]);
    }
    if (!slow) return;
    // The variations: made on their own around the sound as it is -- when there are none, or the sound has
    // moved since -- while nothing else plays it (no blend, nothing listened to).
    if (isShowing() && proc_.variationsFree()) {
        const std::vector<float> now = proc_.knobValues();
        if (proc_.variationCount() == 0 || proc_.variationParent() != now) proc_.makeVariations(static_cast<float>(far_.getValue()));
    }
    if (proc_.variationGeneration() != generation_) {   // new ones: none heard yet
        generation_ = proc_.variationGeneration();
        heard_.fill(false);
        for (auto& k : keep_) k.setVisible(false);
        for (auto& t : tiles_) t->repaint();
    }
    back_.setEnabled(true);
    // Teaching in steps: Teach; while teaching, Set here and Cancel; once taught, a small x to forget.
    bool relayout = false;
    for (int m = 0; m < 4; ++m) {
        const size_t i = static_cast<size_t>(m);
        const bool teaching = proc_.teaching(m), taught = proc_.taught(m);
        const juce::String s = teaching ? juce::String("move the knobs, turn the macro, then Set here") : taught ? proc_.teachStatus(m) : juce::String();
        const juce::String name = proc_.macroName(m);
        if (teachStatus_[i].getText() != s) teachStatus_[i].setText(s, juce::dontSendNotification);
        if (macroName_[i].getText() != name) macroName_[i].setText(name, juce::dontSendNotification);
        const bool showTeach = !teaching, showForget = taught && !teaching;
        if (teach_[i].isVisible() != showTeach || here_[i].isVisible() != teaching || forget_[i].isVisible() != showForget) relayout = true;
        teach_[i].setVisible(showTeach);
        here_[i].setVisible(teaching);
        cancel_[i].setVisible(teaching);
        forget_[i].setVisible(showForget);
    }
    if (relayout) {
        resized();
        repaint();
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

std::array<juce::Rectangle<int>, 6> FocusPage::boxes(int w) const
{
    std::array<juce::Rectangle<int>, 6> r;
    if (twoColumns(w)) {
        const int inner = w - 2 * kGap, left = std::max(kLeftMin, (inner - kGap) / 2), right = inner - kGap - left, x1 = 2 * kGap + left;
        int y = kGap;
        r[0] = { kGap, y, left, kNudgeH };
        y += kNudgeH;
        const int focusH = focus_ != nullptr ? focus_->heightFor(left + 2 * kGap) : 0;
        r[2] = { 0, y, left + 2 * kGap, focusH };
        y += focusH > 0 ? focusH : kGap;
        r[4] = { kGap, y, left, kTeachH };
        y = kGap;
        r[1] = { x1, y, right, kSnapTwoH };
        y += kSnapTwoH + kGap;
        r[3] = { x1, y, right, kVaryTwoH };
        y += kVaryTwoH + kGap;
        r[5] = { x1, y, right, kFindTwoH };
        return r;
    }
    int y = kGap;
    r[0] = { kGap, y, w - 2 * kGap, kNudgeH };
    y += kNudgeH + kGap;
    r[1] = { kGap, y, w - 2 * kGap, kSnapH };
    y += kSnapH;
    const int focusH = focus_ != nullptr ? focus_->heightFor(w) : 0;
    r[2] = { 0, y, w, focusH };
    y += focusH > 0 ? focusH : kGap;
    r[3] = { kGap, y, w - 2 * kGap, kVaryH };
    y += kVaryH + kGap;
    r[4] = { kGap, y, w - 2 * kGap, kTeachH };
    y += kTeachH + kGap;
    r[5] = { kGap, y, w - 2 * kGap, kFindH };
    return r;
}

int FocusPage::heightFor(int w) const
{
    const auto r = boxes(w);
    return std::max(r[4].getBottom(), r[5].getBottom()) + kGap;
}

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
    {   // the snapshots: the button, then eight tiles -- in a row, or in two rows of four where the box is narrow
        auto area = r[1].reduced(12, 0).withTrimmedTop(kTitle).withTrimmedBottom(10);
        const int across = area.getWidth() / 8 >= 90 ? 8 : 4, down = 8 / across;
        const int w = area.getWidth() / across, h = (area.getHeight() - (down - 1) * 4) / down;
        for (int k = 0; k < 8; ++k)
            snapTiles_[static_cast<size_t>(k)]->setBounds(area.getX() + (k % across) * w + 3, area.getY() + (k / across) * (h + 4), w - 6, h);
    }
    if (focus_ != nullptr) focus_->setBounds(r[2]);
    {   // the variations
        auto area = r[3].reduced(12, 0).withTrimmedTop(kTitle);
        auto top = area.removeFromTop(28);
        make_.setBounds(top.removeFromLeft(110).reduced(2));
        back_.setBounds(top.removeFromLeft(80).reduced(2));
        top.removeFromLeft(12);
        farName_.setBounds(top.removeFromLeft(90));
        far_.setBounds(top.removeFromLeft(220).reduced(0, 4));
        area.removeFromTop(6);
        // Six tiles in a row, or two rows of three where the box is narrow (the tiles lower then).
        const int across = area.getWidth() / 6 >= 120 ? 6 : 3, w = area.getWidth() / across;
        const int tileH = across == 6 ? 76 : 40, keepH = across == 6 ? 24 : 22, rowH = tileH + 4 + keepH + 8;
        for (int k = 0; k < 6; ++k) {
            const auto cell = juce::Rectangle<int>(area.getX() + (k % across) * w, area.getY() + (k / across) * rowH, w, rowH).reduced(4, 0);
            tiles_[static_cast<size_t>(k)]->setBounds(cell.withHeight(tileH));
            keep_[static_cast<size_t>(k)].setBounds(cell.withTrimmedTop(tileH + 4).withHeight(keepH));
        }
    }
    {   // the teaching: four columns, each a name, a large knob, Teach / Here / Forget, its status
        auto area = r[4].reduced(12, 0).withTrimmedTop(kTitle);
        const int w = area.getWidth() / 4;
        for (int m = 0; m < 4; ++m) {
            const size_t i = static_cast<size_t>(m);
            auto col = area.withX(area.getX() + m * w).withWidth(w).reduced(6, 0);
            macroName_[i].setBounds(col.removeFromTop(18));
            macro_[i]->setBounds(col.removeFromTop(96).withSizeKeepingCentre(96, 96));
            auto buttons = col.removeFromTop(26);
            if (proc_.teaching(m)) {   // Set here | Cancel
                auto pair = buttons.withSizeKeepingCentre(std::min(buttons.getWidth(), 176), 26);
                here_[i].setBounds(pair.removeFromLeft(pair.getWidth() / 2).reduced(2, 0));
                cancel_[i].setBounds(pair.reduced(2, 0));
            } else if (proc_.taught(m)) {   // Teach  x
                auto pair = buttons.withSizeKeepingCentre(std::min(buttons.getWidth(), 116), 26);
                forget_[i].setBounds(pair.removeFromRight(26).reduced(1, 0));
                teach_[i].setBounds(pair.reduced(2, 0));
            } else {
                teach_[i].setBounds(buttons.withSizeKeepingCentre(std::min(buttons.getWidth(), 90), 26));
            }
            teachStatus_[i].setBounds(col.withHeight(34));
        }
    }
    {   // the search
        auto area = r[5].reduced(12, 0).withTrimmedTop(kTitle);
        auto top = area.removeFromTop(28);
        open_.setBounds(top.removeFromLeft(170).reduced(2));
        top.removeFromLeft(8);
        findStatus_.setBounds(top);
        area.removeFromTop(6);
        const int across = area.getWidth() >= 900 ? 4 : area.getWidth() >= 480 ? 3 : 2, w = area.getWidth() / across;
        for (size_t i = 0; i < results_.size(); ++i)
            results_[i].setBounds(area.getX() + static_cast<int>(i % static_cast<size_t>(across)) * w + 2,
                                  area.getY() + static_cast<int>(i / static_cast<size_t>(across)) * 30, w - 4, 26);
    }
}

void FocusPage::paint(juce::Graphics& g)
{
    const frame::Skin& skin = additivui::skin();
    const auto r = boxes(getWidth());
    const char* const titles[6] = { "Nudge: a little more of one thing, the level held", "Snapshots: keep the sound now, come back with a click",
                                    nullptr, "Variations: hold a tile to listen, Keep to take it",
                                    "Teach the macros by example: Teach, move the knobs, turn the macro, Set here",
                                    "Find by example: the presets that sound like a recording" };
    const frame::Family families[6] = { frame::Family::Source, frame::Family::Filter, frame::Family::Source, frame::Family::Motion, frame::Family::Envelope,
                                        frame::Family::Space };
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
    // A ring pulsing round each macro being taught.
    const int w = r[4].reduced(12, 0).getWidth() / 4;
    const float a = 0.45f + 0.35f * static_cast<float>(std::sin(pulse_ * 4.0));
    for (int m = 0; m < 4; ++m) {
        if (!proc_.teaching(m)) continue;
        const auto col = juce::Rectangle<int>(r[4].getX() + 12 + m * w, r[4].getY() + kTitle - 4, w, r[4].getHeight() - kTitle).reduced(4, 0).toFloat();
        g.setColour(skin.accent.withAlpha(a));
        g.drawRoundedRectangle(col, 8.0f, 2.0f);
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
