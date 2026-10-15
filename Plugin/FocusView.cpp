/**
 * @file FocusView.cpp
 * @brief The Focus page (FocusView.h).
 * @author Rene Weller
 */
#include "FocusView.h"

#include "Theme.h"

#include "additiv/Focus.h"
#include "additiv/Presets.h"

#include <algorithm>
#include <cmath>

using additiv::ParamIdE;

namespace {

constexpr int Title = 26;      ///< a box's title row
constexpr int Gap = 10;        ///< between the boxes
constexpr int NudgeH = 92;     ///< the nudges' box
constexpr int SnapH = 104;     ///< the snapshots' box
constexpr int VaryH = 176;     ///< the variations' box
constexpr int TeachH = 214;    ///< the teaching's box: four columns of a name, a large knob, three buttons, a status
constexpr int FindH = 132;     ///< the search's box
// Two columns, where the page is wide enough: what changes the sound on the left (nudge, focus, teach),
// what keeps and finds sounds on the right (snapshots, variations, find).
constexpr int LeftMin = 680;   ///< the left column's least width: the focus box's eight knobs
constexpr int RightMin = 440;  ///< the right column's least width
constexpr int SnapTwoH = 108;  ///< the snapshots' box in the right column: the tiles in two rows of four
constexpr int VaryTwoH = 206;  ///< the variations' box there: the tiles in two rows of three
constexpr int FindTwoH = 158;  ///< the search's box there: the presets found in three columns

/**
 * @brief The page is wide enough for two columns.
 * @param[in] w  its width
 * @return true for two
 */
bool twoColumns(int w) { return w - 2 * Gap >= LeftMin + Gap + RightMin; }
const char* const Axis[additiv::NudgeAxes] = { "brightness", "motion", "width", "air" };   ///< what each pair moves

} // namespace

void VariationTile::paint(juce::Graphics& g)
{
    const frame::Skin& skin = additivui::skin();
    const bool listening = m_proc.previewing() == m_k;
    const auto r = getLocalBounds().toFloat().reduced(1.0f);
    g.setColour(listening ? skin.m_accent.withAlpha(0.22f) : skin.m_raised);
    g.fillRoundedRectangle(r, skin.m_radius);
    g.setColour(listening ? skin.m_accent : skin.m_edge);
    g.drawRoundedRectangle(r, skin.m_radius, 1.0f);
    if (m_k >= m_proc.variationCount()) return;
    g.setColour(skin.m_accent);
    g.setFont(juce::FontOptions(18.0f, juce::Font::bold));
    const bool low = r.getHeight() < 60.0f;   // a low tile: the number beside what it changes, not above it
    g.drawText(juce::String(m_k + 1), r.withTrimmedLeft(8.0f).withHeight(24.0f), juce::Justification::centredLeft);
    g.setColour(skin.m_dim);
    g.setFont(juce::FontOptions(11.0f));
    g.drawFittedText(m_proc.variationSummary(m_k), (low ? r.reduced(4.0f, 3.0f).withTrimmedLeft(24.0f) : r.reduced(8.0f, 4.0f).withTrimmedTop(20.0f)).toNearestInt(),
                     juce::Justification::topLeft, low ? 2 : 3);
}

void VariationTile::mouseDown(const juce::MouseEvent&)
{
    if (m_k >= m_proc.variationCount()) return;
    if (m_onHeard) m_onHeard(m_k);
    m_proc.previewVariation(m_k);
    repaint();
}

void VariationTile::mouseUp(const juce::MouseEvent&)
{
    m_proc.previewVariation(-1);
    repaint();
}

/** @brief A snapshot's tile (see FocusPage). */
class FocusPage::SnapshotTile final : public juce::Component, public juce::SettableTooltipClient {
public:
    /**
     * @brief Tile @p k of @p page.
     * @param[in,out] page  the page
     * @param[in]     k     0 .. 7
     */
    SnapshotTile(FocusPage& page, int k) : m_page(page), m_k(k)
    {
        setTooltip("A snapshot: click to come back to it (the sound glides there); right-click: into the morph's slot A or B, or forget it");
    }
    /** @brief What the tile does now: keep the sound (the +), come back to one, or nothing yet. @return the text */
    juce::String getTooltip() override
    {
        const int count = static_cast<int>(m_page.m_proc.snapshots().size());
        if (m_k == count) return "Keep the sound now -- every knob and the matrix, with its picture";
        return m_k < count ? juce::SettableTooltipClient::getTooltip() : juce::String();
    }
    /**
     * @brief The + lights up under the mouse.
     */
    void mouseEnter(const juce::MouseEvent&) override { repaint(); }
    /**
     * @brief And goes dim again.
     */
    void mouseExit(const juce::MouseEvent&) override { repaint(); }
    /**
     * @brief Its picture and number, or an empty frame.
     * @param[in,out] g  the graphics context
     */
    void paint(juce::Graphics& g) override
    {
        const frame::Skin& skin = additivui::skin();
        const auto r = getLocalBounds().toFloat().reduced(1.0f);
        const auto& shots = m_page.m_proc.snapshots();
        const int count = static_cast<int>(shots.size());
        const bool has = m_k < count;
        if (m_k == count) {   // the first free one: a + that keeps the sound
            g.setColour(skin.m_accent.withAlpha(isMouseOver() ? 0.16f : 0.07f));
            g.fillRoundedRectangle(r, skin.m_radius);
            g.setColour(skin.m_accent.withAlpha(0.7f));
            const float dash[2] = { 4.0f, 3.0f };
            juce::Path outline;
            outline.addRoundedRectangle(r, skin.m_radius);
            juce::Path dashed;
            juce::PathStrokeType(1.0f).createDashedStroke(dashed, outline, dash, 2);
            g.fillPath(dashed);
            g.setFont(juce::FontOptions(r.getHeight() < 40.0f ? 12.0f : 14.0f, juce::Font::bold));
            g.drawText(r.getWidth() > 90.0f ? "+ Snapshot" : "+", r, juce::Justification::centred);
            return;
        }
        if (!has) {   // further free ones: barely there until they are filled
            g.setColour(skin.m_faint.withAlpha(0.35f));
            g.drawRoundedRectangle(r, skin.m_radius, 1.0f);
            return;
        }
        g.setColour(skin.m_raised);
        g.fillRoundedRectangle(r, skin.m_radius);
        if (shots[static_cast<size_t>(m_k)].m_picture.isValid())
            g.drawImage(shots[static_cast<size_t>(m_k)].m_picture, r.reduced(3.0f), juce::RectanglePlacement::stretchToFit);
        g.setColour(skin.m_accent);
        g.drawRoundedRectangle(r, skin.m_radius, 1.2f);
        g.setFont(juce::FontOptions(12.0f, juce::Font::bold));
        g.drawText(juce::String(m_k + 1), r.reduced(6.0f, 3.0f), juce::Justification::topLeft);
    }
    /**
     * @brief A click recalls it; a right-click offers the morph's slots and forgetting.
     * @param[in] e  the click
     */
    void mouseUp(const juce::MouseEvent& e) override
    {
        const int count = static_cast<int>(m_page.m_proc.snapshots().size());
        if (m_k == count && !e.mods.isPopupMenu()) {   // the +
            m_page.m_proc.takeSnapshot(m_page.m_picture ? m_page.m_picture() : juce::Image());
            for (auto& t : m_page.m_snapTiles) t->repaint();
            return;
        }
        if (m_k >= count) return;
        if (e.mods.isPopupMenu()) {
            juce::PopupMenu m;
            m.addItem("Into the morph's slot A", [this] { m_page.m_proc.snapshotToSlot(m_k, 0); });
            m.addItem("Into the morph's slot B", [this] { m_page.m_proc.snapshotToSlot(m_k, 1); });
            m.addSeparator();
            m.addItem("Forget it", [this] {
                m_page.m_proc.forgetSnapshot(m_k);
                for (auto& t : m_page.m_snapTiles) t->repaint();
            });
            m.showMenuAsync(juce::PopupMenu::Options().withTargetComponent(this));
            return;
        }
        m_page.m_undoable("Snapshot " + juce::String(m_k + 1), [this] { m_page.m_proc.recallSnapshot(m_k); });
    }

private:
    FocusPage& m_page;   ///< whose
    int m_k;             ///< which
};

FocusPage::FocusPage(AdditivProcessor& proc, const frame::ControlActions& actions, frame::LiveRings* live, UndoableT undoable,
                     std::function<juce::Image()> picture)
    : m_proc(proc), m_actions(actions), m_live(live), m_undoable(std::move(undoable)), m_picture(std::move(picture))
{
    for (int k = 0; k < 8; ++k) {
        m_snapTiles.push_back(std::make_unique<SnapshotTile>(*this, k));
        addAndMakeVisible(*m_snapTiles.back());
    }
    const frame::Skin& skin = additivui::skin();
    for (int a = 0; a < additiv::NudgeAxes; ++a) {
        juce::TextButton& lo = m_lower[static_cast<size_t>(a)];
        juce::TextButton& hi = m_higher[static_cast<size_t>(a)];
        lo.setButtonText(juce::String("< ") + additiv::NudgeLow[a]);
        hi.setButtonText(juce::String(additiv::NudgeHigh[a]) + " >");
        lo.setTooltip(juce::String("A little ") + juce::String(additiv::NudgeLow[a]).toLowerCase() + ": every knob that is heard moves the way it was measured to make it so, the level held");
        hi.setTooltip(juce::String("A little ") + juce::String(additiv::NudgeHigh[a]).toLowerCase() + ": every knob that is heard moves the way it was measured to make it so, the level held");
        for (int s = 0; s < 2; ++s) {
            juce::TextButton& b = s == 0 ? lo : hi;
            const float sign = s == 0 ? -1.0f : 1.0f;
            const juce::String name = s == 0 ? additiv::NudgeLow[a] : additiv::NudgeHigh[a];
            b.onClick = [this, a, sign, name] {
                float got = 0.0f;
                m_undoable("Nudge: " + name, [this, a, sign, &got] { got = m_proc.nudgeSound(a, sign); });
                if (a == 1)   // calmer and livelier go by a rule (Focus.h)
                    m_nudgeStatus.setText(name + juce::String(sign < 0.0f ? ": every rate that moves something 1.23 times slower, drift and shimmer a little less"
                                                                       : ": every rate that moves something 1.23 times faster, drift and shimmer a little more"),
                                         juce::dontSendNotification);
                else
                    m_nudgeStatus.setText(std::fabs(got) < 0.02f ? name + ": nothing that is heard here moves that way"
                                                                  : name + ": " + juce::String(std::fabs(got), 2) + " of the bank's spread in " + Axis[a] + " (as measured)",
                                         juce::dontSendNotification);
            };
            addAndMakeVisible(b);
        }
    }
    m_nudgeStatus.setColour(juce::Label::textColourId, skin.m_dim);
    m_nudgeStatus.setFont(juce::FontOptions(12.0f));
    m_nudgeStatus.setText("Each press moves the sound a little; undo takes it back.", juce::dontSendNotification);
    addAndMakeVisible(m_nudgeStatus);

    m_make.setButtonText("New six");
    m_make.setTooltip("Six other variations near the sound now (they are made on their own whenever the sound moves): hold a tile to listen, Keep to take it");
    m_make.onClick = [this] {
        m_proc.makeVariations(static_cast<float>(m_far.getValue()));
        for (auto& t : m_tiles) t->repaint();
    };
    m_back.setTooltip("Back to the sound before the last variation kept");
    m_back.onClick = [this] {
        m_undoable("Variation back", [this] { m_proc.variationBack(); });
        for (auto& t : m_tiles) t->repaint();
    };
    m_far.setSliderStyle(juce::Slider::LinearHorizontal);
    m_far.setTextBoxStyle(juce::Slider::NoTextBox, true, 0, 0);
    m_far.setRange(0.05, 1.0, 0.01);
    m_far.setValue(0.3, juce::dontSendNotification);
    m_far.setTooltip("How far the variations go from the sound now");
    m_farName.setText("near ... far", juce::dontSendNotification);
    m_farName.setColour(juce::Label::textColourId, skin.m_dim);
    m_farName.setFont(juce::FontOptions(12.0f));
    for (juce::Component* c : { static_cast<juce::Component*>(&m_make), static_cast<juce::Component*>(&m_back), static_cast<juce::Component*>(&m_far),
                                static_cast<juce::Component*>(&m_farName) })
        addAndMakeVisible(c);
    for (int k = 0; k < 6; ++k) {
        m_tiles.push_back(std::make_unique<VariationTile>(m_proc, k));
        m_tiles.back()->m_onHeard = [this](int which) {
            m_heard[static_cast<size_t>(which)] = true;
            m_keep[static_cast<size_t>(which)].setVisible(true);
        };
        addAndMakeVisible(*m_tiles.back());
        juce::TextButton& keep = m_keep[static_cast<size_t>(k)];
        keep.setButtonText("Keep");
        keep.setTooltip("This variation becomes the sound; six new ones are made around it");
        keep.onClick = [this, k] {
            m_undoable("Variation " + juce::String(k + 1), [this, k] { m_proc.takeVariation(k); });
            for (auto& t : m_tiles) t->repaint();
        };
        addChildComponent(keep);   // shown once its tile was heard
    }

    for (int m = 0; m < 4; ++m) {
        const size_t i = static_cast<size_t>(m);
        m_macroName[i].setText(m_proc.macroName(m), juce::dontSendNotification);
        m_macroName[i].setColour(juce::Label::textColourId, skin.family(frame::FamilyE::ENVELOPE));
        m_macroName[i].setFont(juce::FontOptions(14.0f, juce::Font::bold));
        m_macroName[i].setJustificationType(juce::Justification::centred);
        m_macroName[i].setMinimumHorizontalScale(0.7f);
        const int id = static_cast<int>(ParamIdE::MACRO1) + m;
        auto knob = std::make_unique<frame::Knob>(juce::Slider::RotaryHorizontalVerticalDrag, juce::Slider::TextBoxBelow, &m_actions, id);
        knob->setTextBoxStyle(juce::Slider::TextBoxBelow, false, 70, 16);
        knob->setColour(juce::Slider::rotarySliderFillColourId, skin.family(frame::FamilyE::ENVELOPE));
        knob->setTooltip("Macro " + juce::String(m + 1) + ": what the matrix and its lesson let it move");
        if (auto* p = m_proc.parameter(id)) m_macroLinks.push_back(std::make_unique<juce::SliderParameterAttachment>(*p, *knob));
        if (m_live != nullptr) m_live->add(*knob, id);
        m_macro.push_back(std::move(knob));
        juce::Component* macroKnob = m_macro.back().get();
        m_teach[i].setButtonText("Teach");
        m_teach[i].setTooltip("Begin: the sound now is this macro's place at its value now. Then move the knobs, set the macro, and press Set here");
        m_teach[i].onClick = [this, m] {
            m_proc.teachBegin(m);
            timerCallback();
        };
        m_here[i].setButtonText("Set here");
        m_here[i].setTooltip("The sound now becomes this macro's place at its value now; the knobs go back, the macro carries it");
        m_here[i].onClick = [this, m] {
            m_proc.teachHere(m);
            timerCallback();
        };
        m_cancel[i].setButtonText("Cancel");
        m_cancel[i].setTooltip("Ends the teaching: the knobs go back where it began, the macro keeps what it had");
        m_cancel[i].onClick = [this, m] {
            m_proc.teachCancel(m);
            timerCallback();
        };
        m_forget[i].setButtonText("x");
        m_forget[i].setTooltip("Forgets what this macro was taught (its matrix rows stay)");
        m_forget[i].onClick = [this, m] {
            m_proc.teachForget(m);
            timerCallback();
        };
        m_teachStatus[i].setColour(juce::Label::textColourId, skin.m_dim);
        m_teachStatus[i].setFont(juce::FontOptions(12.0f));
        m_teachStatus[i].setJustificationType(juce::Justification::centredTop);
        m_teachStatus[i].setFont(juce::FontOptions(11.0f));
        for (juce::Component* c : { static_cast<juce::Component*>(&m_macroName[i]), macroKnob, static_cast<juce::Component*>(&m_teach[i]),
                                    static_cast<juce::Component*>(&m_teachStatus[i]) })
            addAndMakeVisible(c);
        for (juce::Component* c : { static_cast<juce::Component*>(&m_here[i]), static_cast<juce::Component*>(&m_cancel[i]), static_cast<juce::Component*>(&m_forget[i]) })
            addChildComponent(c);
    }

    m_open.setTooltip("A recording (WAV or FLAC): the presets that sound most like it -- or drop one on this page");
    m_open.onClick = [this] {
        m_chooser = std::make_unique<juce::FileChooser>("Find the presets that sound like a recording", juce::File::getSpecialLocation(juce::File::userMusicDirectory),
                                                       "*.wav;*.flac");
        m_chooser->launchAsync(juce::FileBrowserComponent::openMode | juce::FileBrowserComponent::canSelectFiles, [this](const juce::FileChooser& fc) {
            if (fc.getResult() != juce::File()) find(fc.getResult());
        });
    };
    m_findStatus.setColour(juce::Label::textColourId, skin.m_dim);
    m_findStatus.setFont(juce::FontOptions(12.0f));
    m_findStatus.setText("Open or drop a recording: its brightness, motion, width and noise are measured as the presets' were.", juce::dontSendNotification);
    addAndMakeVisible(m_open);
    addAndMakeVisible(m_findStatus);
    for (size_t i = 0; i < m_results.size(); ++i) {
        m_results[i].onClick = [this, i] {
            if (i >= m_proc.found().size()) return;
            const int index = m_proc.found()[i];
            m_undoable("Preset", [this, index] { m_proc.choosePreset(index); });
        };
        addChildComponent(m_results[i]);
    }

    rebuildFocus({});
    timerCallback();
    startTimerHz(15);
}

void FocusPage::rebuildFocus(const std::vector<ParamIdE>& ids)
{
    m_focusIds = ids;
    if (m_focus != nullptr) removeChildComponent(m_focus.get());
    std::vector<ParamPage::Group> groups;
    groups.push_back({ "Focus: the knobs that do the most for this sound", frame::FamilyE::SOURCE, ids });
    m_focus = std::make_unique<ParamPage>(m_proc, m_actions, std::move(groups), m_live);
    addAndMakeVisible(*m_focus);
    m_focus->sendLookAndFeelChange();   // made after the page joined the editor: the frame's look for its knobs too
    resized();
}

void FocusPage::timerCallback()
{
    if (!isShowing() && m_focus != nullptr && !m_focusIds.empty()) return;
    const bool slow = m_tick % 4 == 0;   // four times a second; the teaching's pulse at fifteen
    // The focus anew when what matters changed (a preset, a gate opened or closed).
    if (m_tick++ % 16 == 0) {
        const std::vector<float> v = m_proc.knobValues();
        ParamIdE ids[8];
        const int n = additiv::focusParams(v.data(), ids, 8);
        std::vector<ParamIdE> now(ids, ids + n);
        std::vector<ParamIdE> a = now, b = m_focusIds;
        std::sort(a.begin(), a.end());
        std::sort(b.begin(), b.end());
        if (a != b) rebuildFocus(now);
    }
    // The teaching's pulse.
    bool anyTeaching = false;
    for (int m = 0; m < 4; ++m) anyTeaching = anyTeaching || m_proc.teaching(m);
    if (anyTeaching) {
        m_pulse += 1.0 / 15.0;
        repaint(boxes(getWidth())[4]);
    }
    if (!slow) return;
    // The variations: made on their own around the sound as it is -- when there are none, or the sound has
    // moved since -- while nothing else plays it (no blend, nothing listened to).
    if (isShowing() && m_proc.variationsFree()) {
        const std::vector<float> now = m_proc.knobValues();
        if (m_proc.variationCount() == 0 || m_proc.variationParent() != now) m_proc.makeVariations(static_cast<float>(m_far.getValue()));
    }
    if (m_proc.variationGeneration() != m_generation) {   // new ones: none heard yet
        m_generation = m_proc.variationGeneration();
        m_heard.fill(false);
        for (auto& k : m_keep) k.setVisible(false);
        for (auto& t : m_tiles) t->repaint();
    }
    m_back.setEnabled(true);
    // Teaching in steps: Teach; while teaching, Set here and Cancel; once taught, a small x to forget.
    bool relayout = false;
    for (int m = 0; m < 4; ++m) {
        const size_t i = static_cast<size_t>(m);
        const bool teaching = m_proc.teaching(m), taught = m_proc.taught(m);
        const juce::String s = teaching ? juce::String("move the knobs, turn the macro, then Set here") : taught ? m_proc.teachStatus(m) : juce::String();
        const juce::String name = m_proc.macroName(m);
        if (m_teachStatus[i].getText() != s) m_teachStatus[i].setText(s, juce::dontSendNotification);
        if (m_macroName[i].getText() != name) m_macroName[i].setText(name, juce::dontSendNotification);
        const bool showTeach = !teaching, showForget = taught && !teaching;
        if (m_teach[i].isVisible() != showTeach || m_here[i].isVisible() != teaching || m_forget[i].isVisible() != showForget) relayout = true;
        m_teach[i].setVisible(showTeach);
        m_here[i].setVisible(teaching);
        m_cancel[i].setVisible(teaching);
        m_forget[i].setVisible(showForget);
    }
    if (relayout) {
        resized();
        repaint();
    }
}

void FocusPage::find(const juce::File& f)
{
    juce::String why;
    if (!m_proc.findLike(f, why)) {
        m_findStatus.setText(f.getFileName() + ": " + why, juce::dontSendNotification);
        return;
    }
    m_findStatus.setText(f.getFileName() + ": the presets nearest it, nearest first (ringed on the map)", juce::dontSendNotification);
    showFound();
}

void FocusPage::showFound()
{
    const auto& found = m_proc.found();
    for (size_t i = 0; i < m_results.size(); ++i) {
        const bool has = i < found.size();
        m_results[i].setVisible(has);
        if (has) {
            const additiv::Preset& p = additiv::preset(found[i]);
            m_results[i].setButtonText(juce::String(p.m_name) + "  (" + additiv::PresetGroupNames[p.m_group] + ")");
            m_results[i].setTooltip("Distance " + juce::String(m_proc.foundDistance()[i], 2) + " (in the bank's spreads): click to play it");
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
        const int inner = w - 2 * Gap, left = std::max(LeftMin, (inner - Gap) / 2), right = inner - Gap - left, x1 = 2 * Gap + left;
        int y = Gap;
        r[0] = { Gap, y, left, NudgeH };
        y += NudgeH;
        const int focusH = m_focus != nullptr ? m_focus->heightFor(left + 2 * Gap) : 0;
        r[2] = { 0, y, left + 2 * Gap, focusH };
        y += focusH > 0 ? focusH : Gap;
        r[4] = { Gap, y, left, TeachH };
        y = Gap;
        r[1] = { x1, y, right, SnapTwoH };
        y += SnapTwoH + Gap;
        r[3] = { x1, y, right, VaryTwoH };
        y += VaryTwoH + Gap;
        r[5] = { x1, y, right, FindTwoH };
        return r;
    }
    int y = Gap;
    r[0] = { Gap, y, w - 2 * Gap, NudgeH };
    y += NudgeH + Gap;
    r[1] = { Gap, y, w - 2 * Gap, SnapH };
    y += SnapH;
    const int focusH = m_focus != nullptr ? m_focus->heightFor(w) : 0;
    r[2] = { 0, y, w, focusH };
    y += focusH > 0 ? focusH : Gap;
    r[3] = { Gap, y, w - 2 * Gap, VaryH };
    y += VaryH + Gap;
    r[4] = { Gap, y, w - 2 * Gap, TeachH };
    y += TeachH + Gap;
    r[5] = { Gap, y, w - 2 * Gap, FindH };
    return r;
}

int FocusPage::heightFor(int w) const
{
    const auto r = boxes(w);
    return std::max(r[4].getBottom(), r[5].getBottom()) + Gap;
}

void FocusPage::resized()
{
    const auto r = boxes(getWidth());
    {   // the nudges: four pairs in a row
        auto row = r[0].reduced(12, 0).withTrimmedTop(Title).withHeight(30);
        const int pair = row.getWidth() / additiv::NudgeAxes;
        for (int a = 0; a < additiv::NudgeAxes; ++a) {
            auto cell = row.removeFromLeft(pair).reduced(4, 0);
            m_lower[static_cast<size_t>(a)].setBounds(cell.removeFromLeft(cell.getWidth() / 2).reduced(2, 1));
            m_higher[static_cast<size_t>(a)].setBounds(cell.reduced(2, 1));
        }
        m_nudgeStatus.setBounds(r[0].getX() + 12, r[0].getY() + Title + 34, r[0].getWidth() - 24, 20);
    }
    {   // the snapshots: the button, then eight tiles -- in a row, or in two rows of four where the box is narrow
        auto area = r[1].reduced(12, 0).withTrimmedTop(Title).withTrimmedBottom(10);
        const int across = area.getWidth() / 8 >= 90 ? 8 : 4, down = 8 / across;
        const int w = area.getWidth() / across, h = (area.getHeight() - (down - 1) * 4) / down;
        for (int k = 0; k < 8; ++k)
            m_snapTiles[static_cast<size_t>(k)]->setBounds(area.getX() + (k % across) * w + 3, area.getY() + (k / across) * (h + 4), w - 6, h);
    }
    if (m_focus != nullptr) m_focus->setBounds(r[2]);
    {   // the variations
        auto area = r[3].reduced(12, 0).withTrimmedTop(Title);
        auto top = area.removeFromTop(28);
        m_make.setBounds(top.removeFromLeft(110).reduced(2));
        m_back.setBounds(top.removeFromLeft(80).reduced(2));
        top.removeFromLeft(12);
        m_farName.setBounds(top.removeFromLeft(90));
        m_far.setBounds(top.removeFromLeft(220).reduced(0, 4));
        area.removeFromTop(6);
        // Six tiles in a row, or two rows of three where the box is narrow (the tiles lower then).
        const int across = area.getWidth() / 6 >= 120 ? 6 : 3, w = area.getWidth() / across;
        const int tileH = across == 6 ? 76 : 40, keepH = across == 6 ? 24 : 22, rowH = tileH + 4 + keepH + 8;
        for (int k = 0; k < 6; ++k) {
            const auto cell = juce::Rectangle<int>(area.getX() + (k % across) * w, area.getY() + (k / across) * rowH, w, rowH).reduced(4, 0);
            m_tiles[static_cast<size_t>(k)]->setBounds(cell.withHeight(tileH));
            m_keep[static_cast<size_t>(k)].setBounds(cell.withTrimmedTop(tileH + 4).withHeight(keepH));
        }
    }
    {   // the teaching: four columns, each a name, a large knob, Teach / Here / Forget, its status
        auto area = r[4].reduced(12, 0).withTrimmedTop(Title);
        const int w = area.getWidth() / 4;
        for (int m = 0; m < 4; ++m) {
            const size_t i = static_cast<size_t>(m);
            auto col = area.withX(area.getX() + m * w).withWidth(w).reduced(6, 0);
            m_macroName[i].setBounds(col.removeFromTop(18));
            m_macro[i]->setBounds(col.removeFromTop(96).withSizeKeepingCentre(96, 96));
            auto buttons = col.removeFromTop(26);
            if (m_proc.teaching(m)) {   // Set here | Cancel
                auto pair = buttons.withSizeKeepingCentre(std::min(buttons.getWidth(), 176), 26);
                m_here[i].setBounds(pair.removeFromLeft(pair.getWidth() / 2).reduced(2, 0));
                m_cancel[i].setBounds(pair.reduced(2, 0));
            } else if (m_proc.taught(m)) {   // Teach  x
                auto pair = buttons.withSizeKeepingCentre(std::min(buttons.getWidth(), 116), 26);
                m_forget[i].setBounds(pair.removeFromRight(26).reduced(1, 0));
                m_teach[i].setBounds(pair.reduced(2, 0));
            } else {
                m_teach[i].setBounds(buttons.withSizeKeepingCentre(std::min(buttons.getWidth(), 90), 26));
            }
            m_teachStatus[i].setBounds(col.withHeight(34));
        }
    }
    {   // the search
        auto area = r[5].reduced(12, 0).withTrimmedTop(Title);
        auto top = area.removeFromTop(28);
        m_open.setBounds(top.removeFromLeft(170).reduced(2));
        top.removeFromLeft(8);
        m_findStatus.setBounds(top);
        area.removeFromTop(6);
        const int across = area.getWidth() >= 900 ? 4 : area.getWidth() >= 480 ? 3 : 2, w = area.getWidth() / across;
        for (size_t i = 0; i < m_results.size(); ++i)
            m_results[i].setBounds(area.getX() + static_cast<int>(i % static_cast<size_t>(across)) * w + 2,
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
    const frame::FamilyE families[6] = { frame::FamilyE::SOURCE, frame::FamilyE::FILTER, frame::FamilyE::SOURCE, frame::FamilyE::MOTION, frame::FamilyE::ENVELOPE,
                                        frame::FamilyE::SPACE };
    for (size_t i = 0; i < r.size(); ++i) {
        if (titles[i] == nullptr) continue;
        const auto b = r[i].toFloat();
        g.setColour(skin.m_group.withAlpha(0.92f));
        g.fillRoundedRectangle(b, skin.m_radius);
        g.setColour(skin.m_edge);
        g.drawRoundedRectangle(b.reduced(0.5f), skin.m_radius, 1.0f);
        g.setColour(skin.family(families[i]));
        g.fillRoundedRectangle(b.getX() + 8.0f, b.getY() + 9.0f, 3.0f, 10.0f, 1.5f);
        g.setFont(juce::FontOptions(13.0f, juce::Font::bold));
        g.drawText(titles[i], r[i].withHeight(Title).withTrimmedLeft(17), juce::Justification::centredLeft);
    }
    // A ring pulsing round each macro being taught.
    const int w = r[4].reduced(12, 0).getWidth() / 4;
    const float a = 0.45f + 0.35f * static_cast<float>(std::sin(m_pulse * 4.0));
    for (int m = 0; m < 4; ++m) {
        if (!m_proc.teaching(m)) continue;
        const auto col = juce::Rectangle<int>(r[4].getX() + 12 + m * w, r[4].getY() + Title - 4, w, r[4].getHeight() - Title).reduced(4, 0).toFloat();
        g.setColour(skin.m_accent.withAlpha(a));
        g.drawRoundedRectangle(col, 8.0f, 2.0f);
    }
}

juce::String FocusPage::describe() const
{
    juce::String s("Focus\n");
    s << "    Nudge: " << m_nudgeStatus.getText() << "\n";
    s << "    Focus: ";
    for (size_t i = 0; i < m_focusIds.size(); ++i) s << (i > 0 ? ", " : "") << additiv::paramDesc(m_focusIds[i]).m_name;
    s << "\n    Variations: " << m_proc.variationCount() << "\n";
    for (int k = 0; k < m_proc.variationCount(); ++k) s << "        " << (k + 1) << ". " << m_proc.variationSummary(k) << "\n";
    for (int m = 0; m < 4; ++m) s << "    Macro " << (m + 1) << ": " << m_proc.teachStatus(m) << "\n";
    s << "    Find: " << m_findStatus.getText() << "\n";
    for (size_t i = 0; i < m_proc.found().size(); ++i) s << "        " << additiv::preset(m_proc.found()[i]).m_name << "\n";
    return s + "\nThe simple ways in. Nudge moves every knob that is heard a little, each the way it was measured to make the sound "
               "brighter, livelier, wider or airier (or the other way), the level held. Focus shows the few knobs that do the most "
               "for this sound. Variations are six near the sound: hold a tile to listen, Keep to take it, and the next six are made "
               "around it. Teach lets a macro learn a gesture by example. Find measures a recording and finds the presets that sound "
               "most like it.\n";
}
