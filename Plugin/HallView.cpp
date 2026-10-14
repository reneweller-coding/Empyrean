/**
 * @file HallView.cpp
 * @brief The Hall page (see HallView.h).
 */
#include "HallView.h"
#include "Theme.h"

#include <algorithm>
#include <cmath>

using additiv::ParamId;

/** @brief A band's effect on its modes' decay, octaves at Grow 1: the card's own curve (modalKernel). @param m the memory, 1 the average @return -0.5 .. 1.5 */
static float grownOctaves(float m) { return std::clamp(std::log2(1.0f + std::max(m, 0.0f)) - 1.0f, -0.5f, 1.5f); }

/** @brief Seconds as "3 h 12 min", "12 min", "40 s". @param s seconds @return the text */
static juce::String ageText(double s)
{
    if (s < 60.0) return juce::String(static_cast<int>(s)) + " s";
    const int minutes = static_cast<int>(s / 60.0);
    if (minutes < 60) return juce::String(minutes) + " min";
    return juce::String(minutes / 60) + " h " + juce::String(minutes % 60) + " min";
}

HallView::HallView(AdditivProcessor& proc) : proc_(proc)
{
    memory_.assign(static_cast<size_t>(additiv::kHallBins), 1.0f);
    save_.onClick = [this] { save(); };
    load_.onClick = [this] { load(); };
    new_.setTooltip("A new hall: everything it has heard forgotten, every band at the average");
    new_.onClick = [this] { proc_.newHall(); };
    status_.setColour(juce::Label::textColourId, additivui::skin().dim);
    status_.setFont(juce::FontOptions(12.0f));
    for (juce::Component* c : { static_cast<juce::Component*>(&save_), static_cast<juce::Component*>(&load_), static_cast<juce::Component*>(&new_),
                                static_cast<juce::Component*>(&status_) })
        addAndMakeVisible(c);
    timerCallback();
    startTimerHz(4);
}

void HallView::timerCallback()
{
    proc_.engine().hallMemory(memory_, age_);
    const juce::String s = statusText();
    if (status_.getText() != s) status_.setText(s, juce::dontSendNotification);
    repaint(plot().getSmallestIntegerContainer().expanded(2));
}

juce::String HallView::statusText() const
{
    const float grow = proc_.value(static_cast<int>(ParamId::HallGrow));
    juce::String s = proc_.hallName().isNotEmpty() ? proc_.hallName() : juce::String("This hall");
    s << ": " << (age_ > 0.0 ? "heard for " + ageText(age_) : juce::String("new, nothing heard yet")) << ". ";
    if (grow > 0.0f)
        s << "Growth: where it heard much, its modes ring up to " << juce::String(std::exp2(1.5f * grow), 1) << " times longer.";
    else if (grow < 0.0f)
        s << "Erosion: where it heard much, its modes ring up to " << juce::String(std::exp2(-1.5f * grow), 1) << " times shorter.";
    else
        s << "Grow at nothing: it neither learns nor changes.";
    return s;
}

juce::Rectangle<float> HallView::plot() const { return juce::Rectangle<float>(16.0f, 34.0f, static_cast<float>(getWidth()) - 32.0f, static_cast<float>(getHeight()) - 34.0f - 66.0f); }

void HallView::resized()
{
    const int w = getWidth(), h = getHeight();
    status_.setBounds(12, h - 58, w - 24, 20);
    save_.setBounds(16, h - 34, 28, 26);
    load_.setBounds(48, h - 34, 28, 26);
    new_.setBounds(84, h - 34, 90, 26);
}

void HallView::paint(juce::Graphics& g)
{
    const frame::Skin& skin = additivui::skin();
    g.setColour(skin.group.withAlpha(0.92f));
    g.fillRoundedRectangle(getLocalBounds().toFloat(), skin.radius);
    g.setColour(skin.edge);
    g.drawRoundedRectangle(getLocalBounds().toFloat().reduced(0.5f), skin.radius, 1.0f);
    const juce::Colour colour = skin.family(frame::Family::Motion);
    g.setColour(colour);
    g.setFont(juce::FontOptions(13.0f, juce::Font::bold));
    g.drawText("Memory: what the hall has heard, band by band (up: its modes ring longer as it grows)", 16, 4, getWidth() - 24, 20,
               juce::Justification::centredLeft);

    const juce::Rectangle<float> r = plot();
    if (r.getWidth() < 20.0f || r.getHeight() < 20.0f) return;
    // -0.5 .. 1.5 octaves of decay at Grow 1: the average (0) a quarter up from the bottom.
    auto yOf = [&r](float oct) { return r.getBottom() - (oct + 0.5f) / 2.0f * r.getHeight(); };
    auto xOf = [&r](float hz) { return r.getX() + std::log2(hz / 20.0f) / 9.96578428f * r.getWidth(); };
    g.setFont(juce::FontOptions(11.0f));
    for (const float hz : { 50.0f, 100.0f, 200.0f, 500.0f, 1000.0f, 2000.0f, 5000.0f, 10000.0f }) {
        const float x = xOf(hz);
        g.setColour(skin.faint.withAlpha(0.5f));
        g.drawVerticalLine(static_cast<int>(x), r.getY(), r.getBottom());
        g.setColour(skin.dim);
        g.drawText(hz >= 1000.0f ? juce::String(static_cast<int>(hz / 1000.0f)) + "k" : juce::String(static_cast<int>(hz)), static_cast<int>(x) - 20,
                   static_cast<int>(r.getBottom()) + 1, 40, 14, juce::Justification::centred);
    }
    g.setColour(skin.faint);
    g.drawHorizontalLine(static_cast<int>(yOf(0.0f)), r.getX(), r.getRight());
    if (memory_.size() != static_cast<size_t>(additiv::kHallBins)) return;
    juce::Path curve, fill;
    const float base = yOf(0.0f);
    fill.startNewSubPath(r.getX(), base);
    for (int b = 0; b < additiv::kHallBins; ++b) {
        const float x = r.getX() + (static_cast<float>(b) + 0.5f) / static_cast<float>(additiv::kHallBins) * r.getWidth();
        const float y = yOf(grownOctaves(memory_[static_cast<size_t>(b)]));
        if (b == 0) curve.startNewSubPath(x, y);
        else curve.lineTo(x, y);
        fill.lineTo(x, y);
    }
    fill.lineTo(r.getRight(), base);
    fill.closeSubPath();
    const bool learning = proc_.value(static_cast<int>(ParamId::HallGrow)) != 0.0f;
    g.setColour(colour.withAlpha(learning ? 0.25f : 0.12f));
    g.fillPath(fill);
    g.setColour(learning ? colour : colour.withAlpha(0.5f));
    g.strokePath(curve, juce::PathStrokeType(1.5f));
}

void HallView::save()
{
    const juce::File dir = AdditivProcessor::hallFolder();
    dir.createDirectory();
    const juce::String name = proc_.hallName().isNotEmpty() ? proc_.hallName() : juce::String("My Hall");
    chooser_ = std::make_unique<juce::FileChooser>("Save the hall", dir.getChildFile(name + ".hall"), "*.hall");
    chooser_->launchAsync(juce::FileBrowserComponent::saveMode | juce::FileBrowserComponent::warnAboutOverwriting, [this](const juce::FileChooser& fc) {
        const juce::File f = fc.getResult();
        if (f == juce::File()) return;
        if (!proc_.saveHall(f.withFileExtension("hall")))
            juce::AlertWindow::showMessageBoxAsync(juce::MessageBoxIconType::WarningIcon, "Hall", "The hall could not be written there.");
    });
}

void HallView::load()
{
    const juce::File dir = AdditivProcessor::hallFolder();
    dir.createDirectory();
    chooser_ = std::make_unique<juce::FileChooser>("Load a hall", dir, "*.hall");
    chooser_->launchAsync(juce::FileBrowserComponent::openMode | juce::FileBrowserComponent::canSelectFiles, [this](const juce::FileChooser& fc) {
        const juce::File f = fc.getResult();
        if (f == juce::File()) return;
        if (!proc_.loadHall(f))
            juce::AlertWindow::showMessageBoxAsync(juce::MessageBoxIconType::WarningIcon, "Hall", "That file holds no hall.");
    });
}

juce::String HallView::describe() const
{
    juce::String s("Hall memory: ");
    s << statusText() << "\n";
    // The memory in words: its brightest and darkest regions.
    if (memory_.size() == static_cast<size_t>(additiv::kHallBins) && age_ > 0.0) {
        int hi = 0, lo = 0;
        for (int b = 1; b < additiv::kHallBins; ++b) {
            if (memory_[static_cast<size_t>(b)] > memory_[static_cast<size_t>(hi)]) hi = b;
            if (memory_[static_cast<size_t>(b)] < memory_[static_cast<size_t>(lo)]) lo = b;
        }
        auto hz = [](int b) { return 20.0 * std::exp2((b + 0.5) / additiv::kHallBins * 9.96578428); };
        s << "    most heard near " << juce::String(hz(hi), 0) << " Hz (" << juce::String(memory_[static_cast<size_t>(hi)], 1) << " times the average), least near "
          << juce::String(hz(lo), 0) << " Hz\n";
    }
    return s;
}

// ---- the page

HallPage::HallPage(AdditivProcessor& proc, const frame::ControlActions& actions, frame::LiveRings* live)
    : params_(proc, actions,
              { { "Modal Hall", frame::Family::Motion, { ParamId::ModalLevel, ParamId::ModalDecay, ParamId::ModalDamp, ParamId::ModalTune, ParamId::ModalSpread,
                                                         ParamId::ModalDrift, ParamId::ModalWidth, ParamId::ModalLow, ParamId::ModalBreath,
                                                         ParamId::ModalPaint, ParamId::ModalFreeze }, { ParamId::ModalLevel } },
                { "Growth", frame::Family::Envelope, { ParamId::HallGrow, ParamId::HallMemory }, { ParamId::HallGrow } } },
              live),
      view_(proc)
{
    addAndMakeVisible(params_);
    addAndMakeVisible(view_);
}

/** @brief The memory's box: its height. */
constexpr int kHallViewHeight = 300;

int HallPage::heightFor(int w) const { return params_.heightFor(w) + kHallViewHeight + 10; }

void HallPage::resized()
{
    const int top = params_.heightFor(getWidth());
    params_.setBounds(0, 0, getWidth(), top);
    view_.setBounds(10, top, getWidth() - 20, std::max(kHallViewHeight, getHeight() - 10 - top));
}

juce::String HallPage::describe() const
{
    return params_.describe() + view_.describe()
         + "\nThe modal hall is a room of thousands of resonances the voices excite. With Grow it remembers what it hears: "
           "band by band over the octaves, a step of Memory (a minute to four hours) towards the spectrum while anything sounds. "
           "Growth lets the bands it heard much ring longer -- a room that grows into the music played in it; erosion lets them "
           "ring shorter, so it wears away where it was used and the rest rings on. The memory stays through every preset, is "
           "kept with the session, and a hall can be saved, loaded, or begun anew.\n";
}
