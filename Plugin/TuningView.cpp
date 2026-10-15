/**
 * @file TuningView.cpp
 * @brief The Tuning page and its picture (TuningView.h).
 */
#include "TuningView.h"

#include "Theme.h"

#include <algorithm>
#include <cmath>

using additiv::ParamId;

namespace {

constexpr int kViewMin = 200;   ///< the picture's least height
const char* const kNames[12] = { "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B" };   ///< the steps' names

} // namespace

TuningView::TuningView(AdditivProcessor& proc) : proc_(proc) { startTimerHz(15); }

void TuningView::timerCallback()
{
    if (isShowing()) {
        repaint();
        return;
    }
    // The page left (its tab hides the page, not this view): nothing it plays may hang.
    if (playing_ >= 0) proc_.tapNote(playing_, 0.0f);
    playing_ = -1;
    if (!held_.empty()) releaseAll();
}

void TuningView::paint(juce::Graphics& g)
{
    const frame::Skin& skin = additivui::skin();
    const auto b = getLocalBounds().toFloat();
    g.setColour(skin.group.withAlpha(0.92f));
    g.fillRoundedRectangle(b, skin.radius);
    g.setColour(skin.edge);
    g.drawRoundedRectangle(b.reduced(0.5f), skin.radius, 1.0f);
    g.setColour(skin.family(frame::Family::Motion));
    g.setFont(juce::FontOptions(13.0f, juce::Font::bold));
    g.drawText("The octave over the root: equal temperament's steps, the scale's pitches (cents off the step), the voices where they sound",
               b.reduced(18.0f, 4.0f).withHeight(20.0f), juce::Justification::centredLeft);

    additiv::Engine& e = proc_.engine();
    const int rootPc = juce::jlimit(0, 11, juce::roundToInt(proc_.value(static_cast<int>(ParamId::RootNote))));
    const float purity = proc_.value(static_cast<int>(ParamId::Purity));
    const auto area = this->area();
    auto xOf = [&](float cents) { return area.getX() + area.getWidth() * cents / 1200.0f; };
    const float ruler = area.getBottom();

    // Equal temperament: a faint line every hundred cents, the step's name under it.
    g.setFont(juce::FontOptions(11.0f));
    for (int k = 0; k <= 12; ++k) {
        const float x = xOf(100.0f * static_cast<float>(k));
        g.setColour(skin.faint.withAlpha(k % 12 == 0 ? 0.9f : 0.5f));
        g.drawVerticalLine(juce::roundToInt(x), area.getY(), ruler);
        g.setColour(k % 12 == 0 ? skin.ink : skin.dim);
        g.drawText(kNames[(rootPc + k) % 12], juce::Rectangle<float>(x - 20.0f, ruler + 3.0f, 40.0f, 14.0f), juce::Justification::centred);
    }
    g.setColour(skin.edge);
    g.drawHorizontalLine(juce::roundToInt(ruler), area.getX(), area.getRight());

    // The scale: a line at each pitch, how far it stands from the nearest step above it.
    float cents[48];
    const int n = e.scaleCents(cents, 48);
    const juce::Colour scaleColour = skin.family(frame::Family::Motion);
    // The pitches the mouse plays or Shift holds: their lines lit.
    std::vector<float> sounding;
    for (int note : held_) sounding.push_back(static_cast<float>(std::fmod(1200.0 * std::log2(std::max(e.noteHz(note), 1.0f) / rootHz0()) + 12000.0, 1200.0)));
    if (playing_ >= 0) sounding.push_back(static_cast<float>(std::fmod(1200.0 * std::log2(std::max(e.noteHz(playing_), 1.0f) / rootHz0()) + 12000.0, 1200.0)));
    for (int i = 0; i < n; ++i) {
        const float x = xOf(cents[i]);
        bool lit = false;
        for (float c : sounding) lit = lit || std::fabs(c - cents[i]) < 3.0f;
        g.setColour(lit ? skin.ink : scaleColour.withAlpha(0.75f));
        g.fillRect(x - (lit ? 1.5f : 1.0f), area.getY() + 14.0f, lit ? 3.0f : 2.0f, ruler - area.getY() - 14.0f);
        const float off = cents[i] - 100.0f * std::round(cents[i] / 100.0f);
        if (std::fabs(off) >= 1.0f) {
            g.setColour(scaleColour);
            g.drawText((off > 0.0f ? "+" : "") + juce::String(juce::roundToInt(off)), juce::Rectangle<float>(x - 18.0f, area.getY(), 36.0f, 13.0f),
                       juce::Justification::centred);
        }
    }

    // The voices: across where they sound in the octave, up by their octave, a tail as long as Self-Tune pulls.
    const float rootHz = std::max(e.rootHz(), 1.0f);
    const juce::Colour bySource[4] = { skin.accent, additivui::colour::violet, skin.family(frame::Family::Space), skin.dim };
    const float lane = (ruler - area.getY() - 22.0f) / 7.0f;   // seven octaves, from two below the root's
    for (int i = 0; i < additiv::kMaxViewVoices; ++i) {
        const additiv::VoiceView v = e.voiceView(i);
        if (v.hz <= 0.0f || v.gain < 0.03f) continue;
        const double c = 1200.0 * std::log2(static_cast<double>(v.hz) / rootHz);
        const double octave = std::floor(c / 1200.0);
        const float within = static_cast<float>(c - 1200.0 * octave);
        const float y = ruler - 8.0f - lane * static_cast<float>(juce::jlimit(0.0, 6.0, octave + 2.0)) - lane * 0.5f;
        const float x = xOf(within), r = 3.0f + 4.0f * v.gain;
        const juce::Colour col = bySource[juce::jlimit(0, 3, v.source)];
        if (std::fabs(v.tune) > 0.3f) {
            g.setColour(col.withAlpha(0.5f));
            g.drawLine(xOf(within - v.tune), y, x, y, 2.0f);
        }
        g.setColour(col.withAlpha(0.35f + 0.65f * v.gain));
        g.fillEllipse(x - r, y - r, 2.0f * r, 2.0f * r);
    }

    g.setColour(skin.dim);
    g.setFont(juce::FontOptions(11.5f));
    g.drawText("Click a line to hear it (Shift holds it, with others: an interval locking)   Purity " + juce::String(purity, 2) + ": the voices stand " + juce::String(juce::roundToInt(100.0f * purity)) + " % of the way from equal temperament to the scale",
               b.reduced(18.0f, 4.0f).removeFromBottom(16.0f), juce::Justification::centredLeft);
}

juce::Rectangle<float> TuningView::area() const { return getLocalBounds().toFloat().reduced(30.0f, 0.0f).withTrimmedTop(46.0f).withTrimmedBottom(42.0f); }

float TuningView::rootHz0() const { return std::max(proc_.engine().rootHz(), 1.0f); }

void TuningView::mouseDown(const juce::MouseEvent& e)
{
    additiv::Engine& eng = proc_.engine();
    const auto a = area();
    if (!a.expanded(0.0f, 10.0f).contains(e.position)) return;
    // The nearest line of the scale (within ten pixels), in the octave the height says.
    float cents[48];
    const int n = eng.scaleCents(cents, 48);
    int best = -1;
    float bestD = 10.0f;
    for (int i = 0; i < n; ++i) {
        const float d = std::fabs(a.getX() + a.getWidth() * cents[i] / 1200.0f - e.position.x);
        if (d < bestD) { bestD = d; best = i; }
    }
    if (best < 0) return;
    const float lane = (a.getBottom() - a.getY() - 22.0f) / 7.0f;
    const int octave = juce::jlimit(0, 6, static_cast<int>((a.getBottom() - 8.0f - e.position.y) / std::max(lane, 1.0f))) - 2;
    const double want = rootHz0() * std::exp2(octave + cents[best] / 1200.0);
    int note = -1;
    double nearest = 1.0e9;
    for (int k = 12; k < 120; ++k) {
        const double hz = eng.noteHz(k);
        if (hz <= 0.0) continue;
        const double d = std::fabs(std::log2(hz / want));
        if (d < nearest) { nearest = d; note = k; }
    }
    if (note < 0 || nearest > 0.05) return;   // no key sounds that pitch (a scale with fewer steps than keys)
    if (e.mods.isShiftDown()) {   // held: added, or let go when it is held already
        if (auto it = std::find(held_.begin(), held_.end(), note); it != held_.end()) {
            proc_.tapNote(note, 0.0f);
            held_.erase(it);
        } else {
            proc_.tapNote(note, 0.7f);
            held_.push_back(note);
        }
        repaint();
        return;
    }
    releaseAll();
    playing_ = note;
    proc_.tapNote(note, 0.7f);
    repaint();
}

void TuningView::mouseUp(const juce::MouseEvent&)
{
    if (playing_ >= 0) proc_.tapNote(playing_, 0.0f);
    playing_ = -1;
    repaint();
}

void TuningView::visibilityChanged()
{
    if (!isShowing()) releaseAll();
}

void TuningView::releaseAll()
{
    for (int note : held_) proc_.tapNote(note, 0.0f);
    held_.clear();
}

juce::String TuningView::describe() const
{
    return "\nBelow the knobs, the octave over the root as a ruler: equal temperament's twelve steps as faint lines with their names, the "
           "scale's pitches as lines with how far each stands from its step (in cents), and every voice as a light where it sounds in the "
           "octave -- higher up for a higher octave, green for yours, violet for the conductor's -- with a tail as long as Self-Tune is "
           "pulling it towards a pure interval. A click on a line plays it, in the octave the height says; Shift holds it (and the others "
           "Shift held), so an interval can be heard locking as Purity and Self-Tune move.\n";
}

// ============================================================================================================== page

TuningPage::TuningPage(AdditivProcessor& proc, const frame::ControlActions& actions, frame::LiveRings* live)
    : params_(proc, actions,
              { { "Tuning", frame::Family::Motion, { ParamId::Scale, ParamId::RootNote, ParamId::KeyMap, ParamId::RefPitch, ParamId::Purity } },
                { "Tide", frame::Family::Motion, { ParamId::Tide, ParamId::TidePeriod }, { ParamId::Tide } },
                { "Self-Tune", frame::Family::Motion, { ParamId::TuneGravity, ParamId::TuneReach, ParamId::TuneAnchor }, { ParamId::TuneGravity } } },
              live),
      view_(proc)
{
    addAndMakeVisible(params_);
    addAndMakeVisible(view_);
}

int TuningPage::heightFor(int w) const { return params_.heightFor(w) + kViewMin + 10; }

void TuningPage::resized()
{
    const int top = params_.heightFor(getWidth());
    params_.setBounds(0, 0, getWidth(), top);
    view_.setBounds(10, top, getWidth() - 20, std::max(kViewMin, getHeight() - 10 - top));
}

juce::String TuningPage::describe() const { return params_.describe() + view_.describe(); }
