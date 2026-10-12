/**
 * @file PluginEditor.cpp
 * @brief The panel (PluginEditor.h).
 */
#include "PluginEditor.h"

#include "AdditivData.h"

#include "additiv/Landscape.h"

using additiv::ParamId;

namespace {

constexpr const char* kApp = "AdditivAmbientSynth";   ///< the settings' name (application data/<name>)

/** @brief A page in a viewport: it scrolls only where the window is smaller than the page. */
class ScrollPage final : public juce::Viewport {
public:
    /** @brief Holds @p page. @param page the page (owned) */
    explicit ScrollPage(ParamPage* page) : page_(page)
    {
        setViewedComponent(page, true);
        setScrollBarsShown(true, false);
    }
    /** @brief The page as wide as the viewport and as high as it needs. */
    void resized() override
    {
        juce::Viewport::resized();
        const int w = getMaximumVisibleWidth();
        page_->setSize(w, std::max(page_->heightFor(w), getMaximumVisibleHeight()));
    }

private:
    ParamPage* page_;   ///< the page (owned by the viewport)
};

/** @brief A short way to list ParamIds. */
using Ids = std::vector<ParamId>;

} // namespace

AdditivEditor::AdditivEditor(AdditivProcessor& proc) : juce::AudioProcessorEditor(proc), proc_(proc)
{
    setLookAndFeel(&lnf_);
    actions_.learn = [this](int id) { proc_.learn(id); };
    actions_.controllerFor = [this](int id) { return proc_.controllerFor(id); };
    actions_.forget = [this](int id) { proc_.forget(id); };
    actions_.reset = [this](int id) {
        history_.begin("Default", values(), {}, true);
        proc_.resetToDefault(id);
        history_.end(values(), {});
    };
    actions_.describe = [](int id) {
        const additiv::ParamDesc& d = additiv::paramDesc(static_cast<ParamId>(id));
        return juce::String(d.name) + "  (" + d.key + ")";
    };

    addAndMakeVisible(title_);
    addAndMakeVisible(play_);
    play_.onClick = [this] {
        history_.begin("Autoplay", values(), {}, true);
        proc_.setValue(static_cast<int>(ParamId::BrainOn), proc_.value(static_cast<int>(ParamId::BrainOn)) >= 0.5f ? 0.0f : 1.0f);
        history_.end(values(), {});
    };
    addAndMakeVisible(step_);
    step_.setTooltip("Autoplay: exchange one voice of the chord now");
    step_.onClick = [this] { proc_.engine().autoplayStep(); };
    bindChoice(mode_, ParamId::AutoMode);
    bindChoice(scale_, ParamId::Scale);
    bindChoice(root_, ParamId::RootNote);
    bindChoice(motion_, ParamId::Motion);
    bindChoice(landscape_, ParamId::Landscape);
    mode_.setTooltip("Autoplay: Free lets notes come and go; Chords keeps the chord full and exchanges one voice at a time");
    motion_.setTooltip("How every partial's envelope moves");
    landscape_.setTooltip("The landscape the Aurora motion reads");
    status_.setColour(juce::Label::textColourId, additivui::skin().dim);
    status_.setFont(juce::FontOptions(12.0f));
    addAndMakeVisible(status_);
    for (auto* b : { &undo_, &redo_, &help_, &settings_ }) addAndMakeVisible(b);
    undo_.onClick = [this] { apply(history_.undo(), false); };
    redo_.onClick = [this] { apply(history_.redo(), true); };
    help_.onClick = [this] { showHelp(helpView_ == nullptr || !helpView_->isVisible()); };
    settings_.onClick = [this] { showSettings(); };
    addAndMakeVisible(cloud_);

    // The pages: what belongs together sits in one box, coloured by its family. The tabs join the
    // editor first, so every page's controls are made under the frame's look and feel.
    addAndMakeVisible(tabs_);
    using F = frame::Family;
    auto page = [this](const juce::String& name, std::vector<ParamPage::Group> groups) {
        auto* p = new ParamPage(proc_, actions_, std::move(groups));
        pages_.push_back(p);
        tabs_.addTab(name, additivui::skin().panel, new ScrollPage(p), true);
    };
    page("Cloud", { { "Spectrum", F::Source, Ids{ ParamId::Partials, ParamId::Tilt, ParamId::Brightness, ParamId::OddEven, ParamId::Inharmonic } },
                    { "Cloud", F::Source, Ids{ ParamId::CloudWidth, ParamId::WidthSlope, ParamId::Drift, ParamId::DriftRate, ParamId::Shimmer,
                                               ParamId::ShimmerRate, ParamId::Stereo } },
                    { "Air", F::Source, Ids{ ParamId::Air, ParamId::AirCenter, ParamId::AirWidth, ParamId::AirJitter } } });
    page("Motion", { { "Motion", F::Motion, Ids{ ParamId::Motion, ParamId::MotionRate, ParamId::MotionDepth, ParamId::Landscape } } });
    page("Filter", { { "Filter", F::Filter, Ids{ ParamId::FilterModel, ParamId::Cutoff, ParamId::Resonance, ParamId::FilterEnv, ParamId::KeyTrack,
                                                 ParamId::FilterDrift } },
                     { "Envelope", F::Envelope, Ids{ ParamId::Attack, ParamId::Decay, ParamId::Sustain, ParamId::Release } } });
    page("Tuning", { { "Tuning", F::Motion, Ids{ ParamId::Scale, ParamId::RootNote, ParamId::KeyMap, ParamId::RefPitch, ParamId::Purity } },
                     { "Tide", F::Motion, Ids{ ParamId::Tide, ParamId::TidePeriod } } });
    page("Conductor", { { "Conductor", F::Motion, Ids{ ParamId::BrainOn, ParamId::AutoMode, ParamId::BrainRate, ParamId::AutoRate, ParamId::BrainDensity,
                                                       ParamId::BrainHoldMin, ParamId::BrainHoldMax } },
                        { "Register", F::Motion, Ids{ ParamId::BrainLow, ParamId::BrainHigh } },
                        { "Harmony", F::Motion, Ids{ ParamId::BrainConsonance, ParamId::BrainWander, ParamId::AutoLead, ParamId::AutoTension,
                                                     ParamId::AutoRootMove, ParamId::BrainTimbre, ParamId::AutoStep } },
                        { "Brain 2", F::Envelope, Ids{ ParamId::Brain2On, ParamId::Brain2Interval, ParamId::Brain2Density, ParamId::Brain2Rate,
                                                       ParamId::Brain2Depth } } });
    page("Space", { { "Planes", F::Space, Ids{ ParamId::Depth, ParamId::KeysDepth, ParamId::Breath, ParamId::BreathRate, ParamId::Doppler } },
                    { "Near", F::Space, Ids{ ParamId::Presence, ParamId::PadLowCut } },
                    { "Far Reverb", F::Space, Ids{ ParamId::FarLevel, ParamId::FarSize, ParamId::FarDecay, ParamId::FarDamp } },
                    { "Master", F::Space, Ids{ ParamId::MasterGain, ParamId::BassMono, ParamId::StereoWidth } } });
    tabs_.setTabBarDepth(32);
    tabs_.setOutline(0);
    sendLookAndFeelChange();

    for (auto* p : proc_.getParameters()) p->addListener(this);
    frame::keepKeysForEditor(*this);   // the keys stay the editor's (Space plays, not the switch last clicked)
    setWantsKeyboardFocus(true);
    setResizable(true, true);
    setResizeLimits(800, 520, 4800, 3100);
    setSize(1280, 860);   // the family's window size
    shotPrefix_ = juce::SystemStats::getEnvironmentVariable("ADDITIV_SHOT", {});
    startTimerHz(15);
    timerCallback();
}

AdditivEditor::~AdditivEditor()
{
    for (auto* p : proc_.getParameters()) p->removeListener(this);
    stopTimer();
    setLookAndFeel(nullptr);
}

void AdditivEditor::bindChoice(frame::Choice& box, ParamId id)
{
    const additiv::ParamDesc& d = additiv::paramDesc(id);
    for (int i = 0; i < d.numChoices; ++i) box.addItem(d.choices[i], i + 1);
    box.setColour(juce::ComboBox::arrowColourId, additivui::skin().accent);
    headerAttach_.push_back(std::make_unique<juce::ComboBoxParameterAttachment>(*proc_.parameter(static_cast<int>(id)), box));
    addAndMakeVisible(box);
}

std::vector<float> AdditivEditor::values() const
{
    std::vector<float> v(static_cast<size_t>(additiv::kNumParams));
    for (int i = 0; i < additiv::kNumParams; ++i) v[static_cast<size_t>(i)] = proc_.value(i);
    return v;
}

void AdditivEditor::parameterGestureChanged(int index, bool starting)
{
    // Gestures from the panel come on the message thread; a host's from anywhere -- those are its own to undo.
    if (applying_ || !juce::MessageManager::getInstance()->isThisTheMessageThread()) return;
    if (starting) {
        if (!history_.open()) history_.begin(additiv::paramDesc(static_cast<ParamId>(index)).name, values(), {}, false);
        history_.touch(index);
    } else if (history_.open()) {
        history_.end(values(), {});
    }
}

void AdditivEditor::apply(const frame::UndoStep* step, bool after)
{
    if (step == nullptr) return;
    applying_ = true;
    for (const auto& [id, before, now] : step->values) proc_.setValue(id, after ? now : before);
    applying_ = false;
}

void AdditivEditor::timerCallback()
{
    additiv::Engine& e = proc_.engine();
    const bool on = proc_.value(static_cast<int>(ParamId::BrainOn)) >= 0.5f;
    play_.show(on ? frame::IconButton::Icon::Stop : frame::IconButton::Icon::Play, on ? "Autoplay: stop the conductor" : "Autoplay: start the conductor");
    undo_.setEnabled(history_.canUndo());
    redo_.setEnabled(history_.canRedo());
    juce::String s;
    if (proc_.learning() >= 0) {
        s << "MIDI learn: move a controller for " << additiv::paramDesc(static_cast<ParamId>(proc_.learning())).name;
    } else if (!e.prepared()) {
        s << "The GPU engine is not running (no CUDA device, or the audio is off)";
    } else {
        static const char* const kNames[12] = { "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B" };
        const int root = e.brainRoot();
        s << e.soundingVoices() << " voices (" << e.soundingFrom(1) << " conductor, " << e.soundingFrom(2) << " brain 2, " << e.soundingFrom(0)
          << " played)   root " << kNames[((root % 12) + 12) % 12] << (root / 12 - 1) << "   GPU " << juce::String(e.lastGpuMs(), 3) << " ms   latency "
          << juce::String(e.latencySamples() * 1000.0 / std::max(e.config().sampleRate, 1.0), 1) << " ms   " << e.deviceName();
        if (e.underruns() > 0) s << "   underruns " << static_cast<int>(e.underruns());
    }
    status_.setText(s, juce::dontSendNotification);
    cloud_.refresh();
    shootStep();
    // ADDITIV_LOG: a line a second on the real-time path (the slowest block, the emptiest ring, the
    // largest callback, the underruns so far), for finding what makes it stutter.
    if (++logTick_ % 15 == 0) {
        const juce::String path = juce::SystemStats::getEnvironmentVariable("ADDITIV_LOG", {});
        if (path.isNotEmpty() && e.prepared())
            juce::File(path).appendText(juce::String(logTick_ / 15) + " s: worst block " + juce::String(e.takeWorstBlockMs(), 3) + " ms, min fill "
                                        + juce::String(e.takeMinFill()) + " of " + juce::String(e.latencySamples()) + ", callback " + juce::String(e.takeMaxCallback())
                                        + ", underruns " + juce::String(static_cast<int>(e.underruns())) + "\n");
    }
}

void AdditivEditor::shootStep()
{
    // ADDITIV_SHOT_AFTER seconds (at least 3) for the engine to sound, then every page: shown on one tick,
    // photographed three ticks later.
    const int start = 15 * std::max(3, juce::SystemStats::getEnvironmentVariable("ADDITIV_SHOT_AFTER", "3").getIntValue());
    if (shotPrefix_.isEmpty() || ++shotTick_ < start || (shotTick_ - start) % 3 != 0) return;
    const int t = (shotTick_ - start) / 3;
    if (t > 0) {
        const juce::Image img = createComponentSnapshot(getLocalBounds());
        const juce::File f(shotPrefix_ + "-" + juce::String(t - 1) + "-" + tabs_.getCurrentTabName() + ".png");
        f.deleteFile();
        juce::FileOutputStream out(f);
        if (out.openedOk()) juce::PNGImageFormat().writeImageToStream(img, out);
    }
    if (t < tabs_.getNumTabs()) {
        tabs_.setCurrentTabIndex(t);
    } else {
        shotPrefix_ = {};
        if (standalone()) juce::JUCEApplicationBase::quit();
    }
}

void AdditivEditor::paint(juce::Graphics& g)
{
    backdrop_.paint(g, getLocalBounds(), headerBottom_, additivui::skin(), frame::Settings::of(kApp).backdrop());
    if (additivui::skin().logo) additivui::skin().logo(g, logo_);
}

void AdditivEditor::resized()
{
    auto area = getLocalBounds().reduced(12, 8);
    frame::Header h;
    h.logo = &logo_;
    h.title = &title_;
    h.titleWidth = static_cast<int>(frame::titleWidth(additivui::skin(), frame::kHeaderRow * 0.62f)) + 8;
    h.choices = { { &mode_, 96 }, { &scale_, 168 }, { &root_, 64 }, { &motion_, 112 }, { &landscape_, 108 } };
    h.actions = { { &step_, 64 } };
    h.play = &play_;
    h.status = &status_;
    h.tools = { &undo_, &redo_, nullptr, &help_, &settings_ };
    frame::layoutHeader(area, h);
    headerBottom_ = area.getY();
    cloud_.setBounds(area.removeFromTop(juce::jlimit(120, 260, area.getHeight() / 4)));
    area.removeFromTop(8);
    tabs_.setBounds(area);
    if (helpView_ != nullptr) helpView_->setBounds(area);
}

bool AdditivEditor::keyPressed(const juce::KeyPress& key)
{
    frame::Keys k;
    if (standalone()) {   // in a host, Space and F11 are the host's
        k.playStop = [this] { play_.triggerClick(); };
        k.fullScreen = [this] { toggleFullScreen(); };
    }
    k.undo = [this] { apply(history_.undo(), false); };
    k.redo = [this] { apply(history_.redo(), true); };
    k.help = [this] { showHelp(helpView_ == nullptr || !helpView_->isVisible()); };
    k.escape = [this] {
        if (helpView_ != nullptr && helpView_->isVisible()) showHelp(false);
        else if (juce::Desktop::getInstance().getKioskModeComponent() != nullptr) toggleFullScreen();
    };
    return frame::handleKey(key, k);
}

void AdditivEditor::toggleFullScreen()
{
    if (!standalone()) return;
    auto& desk = juce::Desktop::getInstance();
    desk.setKioskModeComponent(desk.getKioskModeComponent() == nullptr ? getTopLevelComponent() : nullptr, false);
}

void AdditivEditor::showHelp(bool on)
{
    if (on && helpView_ == nullptr) {
        helpView_ = std::make_unique<frame::HelpView>(juce::String::fromUTF8(AdditivData::chapters_txt, AdditivData::chapters_txtSize), additivui::skin());
        helpView_->onClose = [this] { showHelp(false); };
        helpView_->extraTopics = [this] {
            std::vector<std::pair<juce::String, juce::String>> t;
            const int i = tabs_.getCurrentTabIndex();
            if (i >= 0 && i < static_cast<int>(pages_.size())) t.emplace_back("This tab: " + tabs_.getTabNames()[i], pages_[static_cast<size_t>(i)]->describe());
            t.emplace_back("Keys", frame::keysText());
            return t;
        };
        addChildComponent(*helpView_);
    }
    if (helpView_ == nullptr) return;
    help_.setToggleState(on, juce::dontSendNotification);
    if (on) helpView_->refresh();
    helpView_->setVisible(on);
    resized();
    if (on) helpView_->grabKeyboardFocus();
    else grabKeyboardFocus();
}

void AdditivEditor::showSettings()
{
    frame::SettingsMenu m;
    m.app = kApp;
    m.version = JucePlugin_VersionString;
    m.canFullScreen = [this] { return standalone(); };
    m.isFullScreen = [] { return juce::Desktop::getInstance().getKioskModeComponent() != nullptr; };
    m.toggleFullScreen = [this] { toggleFullScreen(); };
    m.setWindowScale = [this](float k) {
        if (juce::Desktop::getInstance().getKioskModeComponent() == nullptr) setSize(juce::roundToInt(1280.0f * k), juce::roundToInt(860.0f * k));
    };
    m.about = [this] {
        return juce::String("The additive ambient synth: 20,000 partials per voice on the GPU, after Noctuary.\n")
             + (proc_.engine().prepared() ? proc_.engine().deviceName() : juce::String("GPU engine not running"));
    };
    m.show(settings_);
}
