/**
 * @file PluginEditor.cpp
 * @brief The panel (PluginEditor.h).
 */
#include "PluginEditor.h"

#include "AdditivData.h"

#include "additiv/Landscape.h"
#include "additiv/Presets.h"

using additiv::ParamId;

namespace {

constexpr const char* kApp = "Empyrean";   ///< the settings' name (application data/<name>)

/** @brief A page in a viewport: it scrolls only where the window is smaller than the page. */
class ScrollPage final : public juce::Viewport {
public:
    /** @brief Holds @p page, which needs @p heightFor at a width. @param page the page (owned) @param heightFor its height at a width */
    ScrollPage(juce::Component* page, std::function<int(int)> heightFor) : page_(page), heightFor_(std::move(heightFor))
    {
        setViewedComponent(page, true);
        setScrollBarsShown(true, false);
    }
    /** @brief The page as wide as the viewport and as high as it needs. */
    void resized() override
    {
        juce::Viewport::resized();
        const int w = getMaximumVisibleWidth();
        page_->setSize(w, std::max(heightFor_(w), getMaximumVisibleHeight()));
    }
    /** @brief The height the page needs at the viewport's width. @return pixels */
    int pageHeight() const { return heightFor_(getMaximumVisibleWidth()); }

private:
    juce::Component* page_;                 ///< the page (owned by the viewport)
    std::function<int(int)> heightFor_;     ///< its height at a width
};

/** @brief The Modulation page: the sources' knobs above, the matrix below. */
class ModPage final : public juce::Component {
public:
    /**
     * @brief Builds the page.
     * @param proc     the processor
     * @param actions  the controls' right-click menu
     * @param live     the live rings
     */
    ModPage(AdditivProcessor& proc, const frame::ControlActions& actions, frame::LiveRings* live)
        : params_(proc, actions,
                  { { "LFOs", frame::Family::Motion, { ParamId::Lfo1Shape, ParamId::Lfo1Rate, ParamId::Lfo2Shape, ParamId::Lfo2Rate,
                                                      ParamId::Lfo3Shape, ParamId::Lfo3Rate, ParamId::Lfo4Shape, ParamId::Lfo4Rate } },
                    { "Envelopes", frame::Family::Envelope, { ParamId::Env1Shape, ParamId::Env1Rate, ParamId::Env2Shape, ParamId::Env2Rate } },
                    { "Macros", frame::Family::Source, { ParamId::Macro1, ParamId::Macro2, ParamId::Macro3, ParamId::Macro4 } },
                    { "Fields and Chaos", frame::Family::Motion, { ParamId::Coherence, ParamId::CoherenceRate, ParamId::LeniaRate,
                                                                   ParamId::LeniaGrowth, ParamId::ChaosPeriod } },
                    { "Reservoir", frame::Family::Envelope, { ParamId::ReservoirChaos, ParamId::ReservoirMemory, ParamId::ReservoirInput } } },
                  live),
          matrix_(proc)
    {
        addAndMakeVisible(params_);
        addAndMakeVisible(matrix_);
    }
    /** @brief The height the page needs at a width. @param w pixels @return pixels */
    int heightFor(int w) const { return params_.heightFor(w) + MatrixView::idealHeight() + 10; }
    /** @brief The knobs above, the matrix below. */
    void resized() override
    {
        const int top = params_.heightFor(getWidth());
        params_.setBounds(0, 0, getWidth(), top);
        matrix_.setBounds(10, top, getWidth() - 20, MatrixView::idealHeight());
    }
    /** @brief The page in words. @return the text */
    juce::String describe() const { return params_.describe() + matrix_.describe(); }

private:
    ParamPage params_;     ///< the sources' knobs
    MatrixView matrix_;    ///< the matrix
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
    for (auto* b : { &prevPreset_, &presetName_, &nextPreset_ }) addAndMakeVisible(b);
    prevPreset_.setTooltip("The preset before");
    nextPreset_.setTooltip("The preset after");
    presetName_.setTooltip("The preset playing: click for the bank (16 groups of 64)");
    prevPreset_.onClick = [this] { loadPreset((std::max(proc_.currentPreset(), 0) + additiv::numPresets() - 1) % additiv::numPresets()); };
    nextPreset_.onClick = [this] { loadPreset((proc_.currentPreset() + 1) % additiv::numPresets()); };
    presetName_.onClick = [this] { showPresets(); };
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
        auto* p = new ParamPage(proc_, actions_, std::move(groups), &live_);
        describe_.push_back([p] { return p->describe(); });
        tabs_.addTab(name, additivui::skin().panel, new ScrollPage(p, [p](int w) { return p->heightFor(w); }), true);
    };
    page("Cloud", { { "Spectrum", F::Source, Ids{ ParamId::Partials, ParamId::Tilt, ParamId::Brightness, ParamId::OddEven, ParamId::Inharmonic } },
                    { "Cloud", F::Source, Ids{ ParamId::CloudWidth, ParamId::WidthSlope, ParamId::Drift, ParamId::DriftRate, ParamId::Shimmer,
                                               ParamId::ShimmerRate, ParamId::Stereo } },
                    { "Air", F::Source, Ids{ ParamId::Air, ParamId::AirCenter, ParamId::AirWidth, ParamId::AirJitter } } });
    page("Motion", { { "Motion", F::Motion, Ids{ ParamId::Motion, ParamId::MotionRate, ParamId::MotionDepth, ParamId::Landscape } },
                     { "Field", F::Motion, Ids{ ParamId::FieldMode, ParamId::FieldStrength, ParamId::FieldRadius, ParamId::FieldSpring,
                                                ParamId::FieldRate, ParamId::FieldCount } },
                     { "Dispersion", F::Envelope, Ids{ ParamId::Dispersion } },
                     { "Living Landscape", F::Source, Ids{ ParamId::LifeMode, ParamId::LifeDepth, ParamId::LifeRate, ParamId::LifeCharacter,
                                                           ParamId::LifeZoom } },
                     { "Partial Sync", F::Envelope, Ids{ ParamId::SyncDepth, ParamId::SyncCoupling, ParamId::SyncRate, ParamId::SyncSpread } },
                     { "Ripple", F::Space, Ids{ ParamId::RippleDepth, ParamId::RippleSpeed, ParamId::RippleDecay } } });
    page("Filter", { { "Filter", F::Filter, Ids{ ParamId::FilterModel, ParamId::Cutoff, ParamId::Resonance, ParamId::FilterEnv, ParamId::KeyTrack,
                                                 ParamId::FilterDrift } },
                     { "Envelope", F::Envelope, Ids{ ParamId::Attack, ParamId::Decay, ParamId::Sustain, ParamId::Release } } });
    page("Tuning", { { "Tuning", F::Motion, Ids{ ParamId::Scale, ParamId::RootNote, ParamId::KeyMap, ParamId::RefPitch, ParamId::Purity } },
                     { "Tide", F::Motion, Ids{ ParamId::Tide, ParamId::TidePeriod } },
                     { "Self-Tune", F::Motion, Ids{ ParamId::TuneGravity, ParamId::TuneReach, ParamId::TuneAnchor } } });
    page("Conductor", { { "Conductor", F::Motion, Ids{ ParamId::BrainOn, ParamId::AutoMode, ParamId::BrainRate, ParamId::AutoRate, ParamId::BrainDensity,
                                                       ParamId::BrainHoldMin, ParamId::BrainHoldMax } },
                        { "Register", F::Motion, Ids{ ParamId::BrainLow, ParamId::BrainHigh } },
                        { "Harmony", F::Motion, Ids{ ParamId::BrainConsonance, ParamId::BrainWander, ParamId::AutoLead, ParamId::AutoTension,
                                                     ParamId::AutoRootMove, ParamId::BrainTimbre, ParamId::BrainCascade, ParamId::AutoStep } },
                        { "Brain 2", F::Envelope, Ids{ ParamId::Brain2On, ParamId::Brain2Interval, ParamId::Brain2Density, ParamId::Brain2Rate,
                                                       ParamId::Brain2Depth } } });
    page("Space", { { "Planes", F::Space, Ids{ ParamId::Depth, ParamId::KeysDepth, ParamId::Breath, ParamId::BreathRate, ParamId::Doppler } },
                    { "Near", F::Space, Ids{ ParamId::Presence, ParamId::PadLowCut } },
                    { "Far Reverb", F::Space, Ids{ ParamId::FarLevel, ParamId::FarSize, ParamId::FarDecay, ParamId::FarDamp, ParamId::FarDiffuse,
                                                   ParamId::FarPreDelay, ParamId::FarUnmask, ParamId::FarMode, ParamId::FarFreeze } },
                    { "Master", F::Space, Ids{ ParamId::MasterGain, ParamId::BassMono, ParamId::StereoWidth } },
                    { "Binaural", F::Space, Ids{ ParamId::Binaural, ParamId::BinauralSpan, ParamId::BinauralHeight, ParamId::BinauralOrbit } } });
    page("Effects", { { "Echo", F::Space, Ids{ ParamId::DelayMix, ParamId::DelayTimeL, ParamId::DelayTimeR, ParamId::DelayFeedback, ParamId::DelayCross,
                                               ParamId::DelayDamp, ParamId::DelayAbsorb, ParamId::DelayDuck, ParamId::DelayToFar } },
                      { "Echo 2", F::Space, Ids{ ParamId::Delay2Mix, ParamId::Delay2TimeL, ParamId::Delay2TimeR, ParamId::Delay2Feedback,
                                                 ParamId::Delay2Damp, ParamId::Delay2ToFar } },
                      { "Room", F::Envelope, Ids{ ParamId::EarlyLevel, ParamId::EarlySize, ParamId::EarlyAbsorb, ParamId::NearMix, ParamId::NearDecay,
                                                  ParamId::NearDamp } },
                      { "Modal Hall", F::Motion, Ids{ ParamId::ModalLevel, ParamId::ModalDecay, ParamId::ModalDamp, ParamId::ModalTune, ParamId::ModalSpread,
                                                      ParamId::ModalDrift, ParamId::ModalWidth, ParamId::ModalLow, ParamId::ModalBreath,
                                                      ParamId::ModalPaint, ParamId::ModalFreeze } },
                      { "Spectral Echo", F::Space, Ids{ ParamId::SpecMix, ParamId::SpecTime, ParamId::SpecTilt, ParamId::SpecScatter,
                                                        ParamId::SpecFeedback, ParamId::SpecDamp, ParamId::SpecToFar } },
                      { "Melt", F::Motion, Ids{ ParamId::MeltMix, ParamId::MeltTime, ParamId::MeltSpread, ParamId::MeltToFar } },
                      { "Memory", F::Envelope, Ids{ ParamId::MemoryLevel, ParamId::MemoryTime, ParamId::MemoryBloom, ParamId::MemoryBlur,
                                                    ParamId::MemorySink } } });
    {
        auto* m = new ModPage(proc_, actions_, &live_);
        describe_.push_back([m] { return m->describe(); });
        tabs_.addTab("Modulation", additivui::skin().panel, new ScrollPage(m, [m](int w) { return m->heightFor(w); }), true);
    }
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
    const int cp = proc_.currentPreset();
    const juce::String name = cp >= 0 ? juce::String(additiv::kPresetGroupNames[additiv::preset(cp).group]) + ": " + additiv::preset(cp).name : juce::String("Presets ...");
    if (presetName_.getButtonText() != name) presetName_.setButtonText(name);
    undo_.setEnabled(history_.canUndo());
    redo_.setEnabled(history_.canRedo());
    juce::String s;
    if (proc_.learning() >= 0) {
        s << "MIDI learn: move a controller for " << additiv::paramDesc(static_cast<ParamId>(proc_.learning())).name;
    } else if (!e.prepared()) {
        const juce::String why = proc_.engineError();
        s << "The GPU engine is not running: " << (why.isEmpty() ? juce::String("the audio is off") : why)
          << "  (an NVIDIA card, GeForce RTX 20 or newer, with a current driver)";
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
    // largest callback, the underruns so far), for finding what makes it stutter; the slowest block
    // is split into its preparation, its wait for the GPU and its effects (each its own slowest).
    if (++logTick_ % 15 == 0) {
        const juce::String path = juce::SystemStats::getEnvironmentVariable("ADDITIV_LOG", {});
        if (path.isNotEmpty() && e.prepared()) {
            float prep = 0.0f, gpu = 0.0f, fx = 0.0f;
            e.takeWorstParts(prep, gpu, fx);
            juce::File(path).appendText(juce::String(logTick_ / 15) + " s: worst block " + juce::String(e.takeWorstBlockMs(), 3) + " ms (prep "
                                        + juce::String(prep, 3) + ", gpu " + juce::String(gpu, 3) + ", effects " + juce::String(fx, 3) + "), min fill "
                                        + juce::String(e.takeMinFill()) + " of " + juce::String(e.latencySamples()) + ", callback " + juce::String(e.takeMaxCallback())
                                        + ", underruns " + juce::String(static_cast<int>(e.underruns())) + "\n");
        }
    }
}

void AdditivEditor::shootStep()
{
    // ADDITIV_SHOT_AFTER seconds (at least 3) for the engine to sound, then every page: shown on one tick,
    // photographed three ticks later.
    const int start = 15 * std::max(3, juce::SystemStats::getEnvironmentVariable("ADDITIV_SHOT_AFTER", "3").getIntValue());
    // ADDITIV_PRESET (a name or a number): the preset the pictures show, loaded before the engine is heard.
    if (shotPrefix_.isNotEmpty() && shotTick_ == 0) {
        const juce::String want = juce::SystemStats::getEnvironmentVariable("ADDITIV_PRESET", {});
        int index = -1;
        for (int i = 0; i < additiv::numPresets() && index < 0 && want.isNotEmpty(); ++i)
            if (want == additiv::preset(i).name) index = i;
        if (index < 0 && want.containsOnly("0123456789") && want.isNotEmpty()) index = want.getIntValue();
        if (index >= 0) proc_.applyPreset(index);
        // ADDITIV_SET ("key=value;key=value"): parameters after the preset, a choice by its name or its index.
        for (const juce::String& item : juce::StringArray::fromTokens(juce::SystemStats::getEnvironmentVariable("ADDITIV_SET", {}), ";", ""))
            if (const additiv::ParamDesc* d = additiv::findParam(item.upToFirstOccurrenceOf("=", false, false).trim().toRawUTF8())) {
                const juce::String val = item.fromFirstOccurrenceOf("=", false, false).trim();
                float v = val.getFloatValue();
                for (int c = 0; c < d->numChoices; ++c)
                    if (val == d->choices[c]) v = static_cast<float>(c);
                proc_.setValue(static_cast<int>(d->id), v);
            }
    }
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
        // ADDITIV_SHOT_FULL: the window as tall as the page, so that nothing of it scrolls out of the picture.
        if (juce::SystemStats::getEnvironmentVariable("ADDITIV_SHOT_FULL", {}).isNotEmpty())
            if (auto* sp = dynamic_cast<ScrollPage*>(tabs_.getTabContentComponent(t)))
                setSize(getWidth(), std::clamp(getHeight() - sp->getMaximumVisibleHeight() + sp->pageHeight() + 16, 860, 3100));
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
    h.choices = { { &prevPreset_, 28 }, { &presetName_, 250 }, { &nextPreset_, 28 }, { &mode_, 92 }, { &scale_, 150 }, { &root_, 58 }, { &motion_, 104 } };
    landscape_.setVisible(false);   // on the Motion page; the header has room for the bank
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
            if (i >= 0 && i < static_cast<int>(describe_.size())) t.emplace_back("This tab: " + tabs_.getTabNames()[i], describe_[static_cast<size_t>(i)]());
            t.emplace_back("Keys", frame::keysText());
            t.emplace_back("Headset (Meta Quest)", frame::headsetGrammar("Step", "Brain 2 in or out") + "\n\nThe left hand's height is the brightness (Macro 2), the right hand's the motion (Macro 1); both hands pinched: the next preset.\n\n" + proc_.headset().statusText());
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

void AdditivEditor::loadPreset(int index)
{
    history_.begin("Preset", values(), {}, true);
    proc_.applyPreset(index);
    history_.end(values(), {});
}

void AdditivEditor::showPresets()
{
    juce::PopupMenu bank;
    const int current = proc_.currentPreset();
    for (int g = 0; g < additiv::kNumPresetGroups; ++g) {
        juce::PopupMenu group;
        for (int i = 0; i < 64; ++i) {
            const int index = additiv::presetAt(g, i);
            if (index < 0) break;
            group.addItem(index + 1, additiv::preset(index).name, true, index == current);
        }
        bank.addSubMenu(additiv::kPresetGroupNames[g], group, true, nullptr, current >= 0 && additiv::preset(current).group == g);
    }
    bank.showMenuAsync(juce::PopupMenu::Options().withTargetComponent(&presetName_), [this](int id) {
        if (id > 0) loadPreset(id - 1);
    });
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
    m.headsetStatus = [this] { return proc_.headset().statusText(); };
    m.jamStatus = [this] { return proc_.jamStatus(); };   // the family jam: Off, Lead, Follow
    m.about = [this] {
        return juce::String("The additive ambient synth: 20,000 partials per voice on the GPU, after Noctuary.\n")
             + (proc_.engine().prepared() ? proc_.engine().deviceName() : juce::String("GPU engine not running"));
    };
    m.show(settings_);
}
