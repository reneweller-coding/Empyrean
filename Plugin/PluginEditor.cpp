/**
 * @file PluginEditor.cpp
 * @brief The panel (PluginEditor.h).
 */
#include "PluginEditor.h"

#include "AdditivData.h"
#include "JourneyView.h"
#include "FocusView.h"
#include "HallView.h"
#include "KnobMod.h"
#include "MapView.h"
#include "SourcesView.h"
#include "SpaceView.h"
#include "TuningView.h"

#include "additiv/Find.h"
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
    /** @brief The height the page needs at width @p w. @param w pixels @return pixels */
    int heightAt(int w) const { return heightFor_(w); }

private:
    juce::Component* page_;                 ///< the page (owned by the viewport)
    std::function<int(int)> heightFor_;     ///< its height at a width
};

/**
 * @brief The Modulation page: the sixteen rows in a column, every source live beside them (below two columns of rows
 *        where the page is narrow). A click on a source opens it in the inspector -- its own knobs (an LFO's shape and
 *        rate, a macro, the fields' and the reservoir's) live there.
 */
class MatrixPage final : public juce::Component {
public:
    /** @brief Builds the page. @param proc the processor @param choose a source clicked */
    MatrixPage(AdditivProcessor& proc, std::function<void(additiv::ModSource)> choose) : proc_(proc), sources_(proc), matrix_(proc)
    {
        sources_.onChoose = std::move(choose);
        addAndMakeVisible(matrix_);
        addAndMakeVisible(sources_);
    }
    /** @brief The source the inspector shows, marked among the traces. @param s the source */
    void setChosen(additiv::ModSource s) { sources_.setChosen(s); }
    /** @brief The height the page needs at a width. @param w pixels @return pixels */
    int heightFor(int w) const
    {
        if (beside(w)) return 10 + std::max(MatrixView::heightFor(1), SourcesView::heightFor(w - 30 - kMatrixW)) + 10;
        return 10 + MatrixView::heightFor(2) + 10 + SourcesView::heightFor(w - 20) + 10;
    }
    /** @brief The rows, and the sources beside or below them. */
    void resized() override
    {
        const int w = getWidth(), h = getHeight();
        if (beside(w)) {
            matrix_.setBounds(10, 10, kMatrixW, h - 20);
            sources_.setBounds(20 + kMatrixW, 10, w - 30 - kMatrixW, h - 20);
        } else {
            matrix_.setBounds(10, 10, w - 20, MatrixView::heightFor(2));
            sources_.setBounds(10, matrix_.getBottom() + 10, w - 20, std::max(SourcesView::heightFor(w - 20), h - matrix_.getBottom() - 20));
        }
    }
    /** @brief The page in words: the rows, and the modulators' knobs. @return the text */
    juce::String describe() const
    {
        juce::String s = matrix_.describe() + "\nThe modulators (their knobs are in the inspector: click a source):\n";
        for (const additiv::ParamDesc& d : additiv::paramTable()) {
            if (juce::String(d.section) != "Modulation") continue;
            s << "    " << d.name << " (" << d.key << "): now " << juce::String(proc_.value(static_cast<int>(d.id)), 3) << "\n";
        }
        return s;
    }

private:
    static constexpr int kMatrixW = 600;   ///< the rows' column, beside the sources
    AdditivProcessor& proc_;               ///< whose
    /** @brief The sources fit beside the rows' column. @param w the page's width @return true then */
    static bool beside(int w) { return w - 30 - kMatrixW >= 440; }
    SourcesView sources_;  ///< every source, live
    MatrixView matrix_;    ///< the rows
};

/** @brief A short way to list ParamIds. */
using Ids = std::vector<ParamId>;

} // namespace

static void findParamPages(juce::Component* c, std::vector<ParamPage*>& out);

AdditivEditor::AdditivEditor(AdditivProcessor& proc) : juce::AudioProcessorEditor(proc), proc_(proc)
{
    setLookAndFeel(&lnf_);
    actions_.learn = [this](int id) { proc_.learn(id); };
    actions_.controllerFor = [this](int id) { return proc_.controllerFor(id); };
    actions_.forget = [this](int id) { proc_.forget(id); };
    actions_.reset = [this](int id) {
        history_.begin("Default", values(), matrixText(), true);
        proc_.resetToDefault(id);
        endStep();
    };
    actions_.more = [this](juce::PopupMenu& m, int id) {
        // Back to where the preset had it, when it has moved.
        const auto& ref = proc_.reference();
        if (id >= 0 && id < additiv::kNumParams && ref.size() == static_cast<size_t>(additiv::kNumParams) && !additiv::keptByPresets(static_cast<ParamId>(id))
            && proc_.value(id) != ref[static_cast<size_t>(id)]) {
            const additiv::ParamDesc& d = additiv::paramDesc(static_cast<ParamId>(id));
            m.addItem("Back to the preset's value (" + juce::String(ref[static_cast<size_t>(id)], 3) + (d.unit[0] != 0 ? juce::String(" ") + d.unit : juce::String()) + ")",
                      [this, id] {
                          history_.begin("Back to the preset", values(), matrixText(), true);
                          proc_.revertKnob(id);
                          endStep();
                      });
        }
        knobmod::addMenuItems(proc_, m, id);
    };
    // Alt-drag on a knob: the source the inspector shows modulates it, its depth set by the drag (up: more).
    actions_.altDrag = [this](int id, const juce::MouseEvent& e, int phase) {
        const additiv::ModSource s = inspector_ != nullptr ? inspector_->source() : additiv::ModSource::Off;
        if (phase == 0) {
            if (s == additiv::ModSource::Off || !knobmod::canTarget(id)) return false;
            altRow_ = knobmod::addRow(proc_, s, id, 0.0f);
            if (altRow_ < 0) return false;
            altDepth_ = proc_.engine().modRow(altRow_).depth;
            return true;
        }
        if (altRow_ < 0) return false;
        additiv::ModRow r = proc_.engine().modRow(altRow_);
        r.depth = juce::jlimit(-1.0f, 1.0f, altDepth_ - static_cast<float>(e.getDistanceFromDragStartY()) / 200.0f);
        proc_.engine().setModRow(altRow_, r);
        if (info_ != nullptr)
            info_->showText(juce::String(additiv::kModSourceNames[static_cast<int>(s)]) + " -> " + additiv::paramDesc(static_cast<ParamId>(id)).name
                            + "   depth " + (r.depth >= 0.0f ? "+" : "") + juce::String(r.depth, 2) + "   (Alt-drag: up more, down less or the other way)");
        if (phase == 2) {
            if (std::fabs(r.depth) < 0.005f) proc_.engine().setModRow(altRow_, additiv::ModRow{});   // dragged back to nothing: no row
            chooseSource(s, false);
            altRow_ = -1;
        }
        return true;
    };
    actions_.describe = [](int id) {
        const additiv::ParamDesc& d = additiv::paramDesc(static_cast<ParamId>(id));
        return juce::String(d.name) + "  (" + d.key + ")";
    };

    addAndMakeVisible(title_);
    addAndMakeVisible(play_);
    play_.onClick = [this] {
        history_.begin("Autoplay", values(), matrixText(), true);
        proc_.setValue(static_cast<int>(ParamId::BrainOn), proc_.value(static_cast<int>(ParamId::BrainOn)) >= 0.5f ? 0.0f : 1.0f);
        endStep();
    };
    for (auto* b : { &prevPreset_, &presetName_, &nextPreset_ }) addAndMakeVisible(b);
    prevPreset_.setTooltip("The preset before");
    nextPreset_.setTooltip("The preset after");
    presetName_.setTooltip("The preset playing: click to browse the bank (96 groups of 64: the sounds, the living systems, the sleep concerts, the visions, the swarms and currents, the halls and orbits)");
    prevPreset_.onClick = [this] { loadPreset((std::max(proc_.currentPreset(), 0) + additiv::numPresets() - 1) % additiv::numPresets()); };
    nextPreset_.onClick = [this] { loadPreset((proc_.currentPreset() + 1) % additiv::numPresets()); };
    presetName_.onClick = [this] { browser_.open(); };
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
    addAndMakeVisible(save_);
    save_.onClick = [this] { showSaveMenu(); };
    undo_.onClick = [this] {
        flushMatrix();
        apply(history_.undo(), false);
    };
    redo_.onClick = [this] {
        flushMatrix();
        apply(history_.redo(), true);
    };
    help_.onClick = [this] { showHelp(helpView_ == nullptr || !helpView_->isVisible()); };
    settings_.onClick = [this] { showSettings(); };
    addAndMakeVisible(cloud_);
    addAndMakeVisible(canvasButton_);
    addAndMakeVisible(searchButton_);
    searchButton_.onClick = [this] { openPalette(); };
    addAndMakeVisible(compare_);
    compare_.onStateChange = [this] { proc_.compareWithPreset(compare_.isDown() && compare_.isEnabled()); };
    addAndMakeVisible(midi_);
    midi_.setClickingTogglesState(true);
    midi_.onClick = [this] {
        ParamPage::showMidi = midi_.getToggleState();
        tabs_.repaint();
        for (int t = 0; t < tabs_.getNumTabs(); ++t)
            if (auto* view = dynamic_cast<juce::Viewport*>(tabs_.getTabContentComponent(t)); view != nullptr && view->getViewedComponent() != nullptr)
                view->getViewedComponent()->repaint();
    };
    addAndMakeVisible(meter_);
    addChildComponent(palette_);
    addChildComponent(browser_);
    browser_.onChoose = [this](int factory, int user) {
        if (factory >= 0) loadPreset(factory);
        else if (user >= 0) {
            history_.begin("Preset", values(), matrixText(), true);
            proc_.chooseUserPreset(user);
            endStep();
        }
    };
    info_ = std::make_unique<InfoLine>(proc_, *this);
    addAndMakeVisible(*info_);
    addAndMakeVisible(nightButton_);
    nightButton_.setClickingTogglesState(false);
    nightButton_.onClick = [this] { setNight(!night_); };
    activity_.last = &lastActivity_;
    lastActivity_ = juce::Time::getMillisecondCounterHiRes();
    addMouseListener(&activity_, true);
    night_ = proc_.userSettings().getBoolValue("night", false);
    nightButton_.setToggleState(night_, juce::dontSendNotification);
    addChildComponent(tour_);
    veil_.setInterceptsMouseClicks(false, false);
    addChildComponent(veil_);   // over the window, then the overlays over it
    palette_.toFront(false);
    browser_.toFront(false);
    tour_.toFront(false);
    tour_.onDone = [this] {
        proc_.userSettings().setValue("tourDone", true);
        proc_.userSettings().saveIfNeeded();
    };
    canvasButton_.onClick = [this] { setCanvas(!canvas_); };
    cloud_.onDoubleClick = [this] { setCanvas(!canvas_); };

    // The pages: what belongs together sits in one box, coloured by its family. The tabs join the
    // editor first, so every page's controls are made under the frame's look and feel.
    addAndMakeVisible(tabs_);
    using F = frame::Family;
    auto page = [this](const juce::String& name, std::vector<ParamPage::Group> groups) {
        auto* p = new ParamPage(proc_, actions_, std::move(groups), &live_);
        describe_.push_back([p] { return p->describe(); });
        tabs_.addTab(name, additivui::skin().panel, new ScrollPage(p, [p](int w) { return p->heightFor(w); }), true);
    };
    {   // the simple ways in, first
        auto* f = new FocusPage(proc_, actions_, &live_, [this](const juce::String& name, std::function<void()> change) {
            history_.begin(name, values(), matrixText(), true);
            change();
            endStep();
        }, [this] { return cloud_.thumbnail(160, 60); });
        describe_.push_back([f] { return f->describe(); });
        tabs_.addTab("Focus", additivui::skin().panel, new ScrollPage(f, [f](int w) { return f->heightFor(w); }), true);
    }
    page("Cloud & Filter", { { "Spectrum", F::Source, Ids{ ParamId::Partials, ParamId::Tilt, ParamId::Brightness, ParamId::OddEven, ParamId::Inharmonic,
                                                         ParamId::Stretch, ParamId::Shift } },
                    { "Cloud", F::Source, Ids{ ParamId::CloudWidth, ParamId::WidthSlope, ParamId::Drift, ParamId::DriftRate, ParamId::Shimmer,
                                               ParamId::ShimmerRate, ParamId::Stereo } },
                    { "Air", F::Source, Ids{ ParamId::Air, ParamId::AirCenter, ParamId::AirWidth, ParamId::AirJitter }, Ids{ ParamId::Air } },
                    { "Turbulence", F::Source, Ids{ ParamId::Turbulence, ParamId::TurbulenceTime }, Ids{ ParamId::Turbulence } },
                    { "Filter", F::Filter, Ids{ ParamId::FilterModel, ParamId::Cutoff, ParamId::Resonance, ParamId::FilterEnv, ParamId::KeyTrack,
                                                ParamId::FilterDrift }, Ids{ ParamId::FilterModel } },
                    { "Spectral Comb", F::Filter, Ids{ ParamId::CombDepth, ParamId::CombDensity, ParamId::CombSweep, ParamId::CombStereo }, Ids{ ParamId::CombDepth } },
                    { "Envelope", F::Envelope, Ids{ ParamId::Attack, ParamId::Decay, ParamId::Sustain, ParamId::Release } } });
    page("Motion", { { "Motion", F::Motion, Ids{ ParamId::Motion, ParamId::MotionRate, ParamId::MotionDepth, ParamId::Landscape }, Ids{ ParamId::Motion } },
                     { "Field", F::Motion, Ids{ ParamId::FieldMode, ParamId::FieldStrength, ParamId::FieldRadius, ParamId::FieldSpring,
                                                ParamId::FieldRate, ParamId::FieldCount }, Ids{ ParamId::FieldMode } },
                     { "Time", F::Envelope, Ids{ ParamId::Dispersion, ParamId::TimeBend }, Ids{ ParamId::Dispersion, ParamId::TimeBend } },
                     { "Living Landscape", F::Source, Ids{ ParamId::LifeMode, ParamId::LifeDepth, ParamId::LifeRate, ParamId::LifeCharacter,
                                                           ParamId::LifeZoom, ParamId::LifeSides, ParamId::LifeTurn }, Ids{ ParamId::LifeMode } },
                     { "Partial Sync", F::Envelope, Ids{ ParamId::SyncDepth, ParamId::SyncMode, ParamId::SyncCoupling, ParamId::SyncRate, ParamId::SyncSpread,
                                                         ParamId::SyncSwarm, ParamId::SyncReach, ParamId::SyncLag }, Ids{ ParamId::SyncDepth } },
                     { "Ripple", F::Space, Ids{ ParamId::RippleDepth, ParamId::RippleSpeed, ParamId::RippleDecay, ParamId::RippleBend }, Ids{ ParamId::RippleDepth } } });
    {   // the tuning, with its picture: the octave over the root, the scale, the voices and Self-Tune's pull
        auto* tp = new TuningPage(proc_, actions_, &live_);
        describe_.push_back([tp] { return tp->describe(); });
        tabs_.addTab("Tuning", additivui::skin().panel, new ScrollPage(tp, [tp](int w) { return tp->heightFor(w); }), true);
    }
    page("Conductor", { { "Conductor", F::Motion, Ids{ ParamId::BrainOn, ParamId::AutoMode, ParamId::BrainRate, ParamId::AutoRate, ParamId::BrainDensity,
                                                       ParamId::BrainHoldMin, ParamId::BrainHoldMax }, Ids{ ParamId::BrainOn },
                          { { ParamId::BrainHoldMin, ParamId::BrainHoldMax } } },
                        { "Register", F::Motion, Ids{ ParamId::BrainLow, ParamId::BrainHigh, ParamId::BrainFoundation }, {},
                          { { ParamId::BrainLow, ParamId::BrainHigh } } },
                        { "Harmony", F::Motion, Ids{ ParamId::BrainConsonance, ParamId::BrainWander, ParamId::AutoLead, ParamId::AutoTension,
                                                     ParamId::AutoRootMove, ParamId::BrainTimbre, ParamId::BrainCascade, ParamId::BrainLight, ParamId::AutoStep } },
                        { "Brain 2", F::Envelope, Ids{ ParamId::Brain2On, ParamId::Brain2Interval, ParamId::Brain2Density, ParamId::Brain2Rate,
                                                       ParamId::Brain2Depth }, Ids{ ParamId::Brain2On } },
                        { "MPE", F::Filter, Ids{ ParamId::MpeOn, ParamId::MpeBend, ParamId::MpePress, ParamId::MpePressDepth, ParamId::MpeSlide,
                                                 ParamId::MpeSlideDepth }, Ids{ ParamId::MpeOn } } });
    {
        auto* s = new SpacePage(proc_, actions_, &live_);
        describe_.push_back([s] { return s->describe(); });
        tabs_.addTab("Space", additivui::skin().panel, new ScrollPage(s, [s](int w) { return s->heightFor(w); }), true);
    }
    page("Effects", { { "Echo", F::Space, Ids{ ParamId::DelayMix, ParamId::DelayTimeL, ParamId::DelayTimeR, ParamId::DelayFeedback, ParamId::DelayCross,
                                               ParamId::DelayDamp, ParamId::DelayAbsorb, ParamId::DelayDuck, ParamId::DelayToFar }, Ids{ ParamId::DelayMix }, {},
                        { { ParamId::DelayTimeL, ParamId::DelayTimeR } } },
                      { "Echo 2", F::Space, Ids{ ParamId::Delay2Mix, ParamId::Delay2TimeL, ParamId::Delay2TimeR, ParamId::Delay2Feedback,
                                                 ParamId::Delay2Damp, ParamId::Delay2ToFar }, Ids{ ParamId::Delay2Mix }, {},
                        { { ParamId::Delay2TimeL, ParamId::Delay2TimeR } } },
                      { "Room", F::Envelope, Ids{ ParamId::EarlyLevel, ParamId::EarlySize, ParamId::EarlyAbsorb, ParamId::NearMix, ParamId::NearDecay,
                                                  ParamId::NearDamp }, Ids{ ParamId::EarlyLevel, ParamId::NearMix } },
                      { "Spectral Echo", F::Space, Ids{ ParamId::SpecMix, ParamId::SpecTime, ParamId::SpecTilt, ParamId::SpecScatter,
                                                        ParamId::SpecFeedback, ParamId::SpecDamp, ParamId::SpecToFar }, Ids{ ParamId::SpecMix } },
                      { "Melt", F::Motion, Ids{ ParamId::MeltMix, ParamId::MeltTime, ParamId::MeltSpread, ParamId::MeltToFar }, Ids{ ParamId::MeltMix } },
                      { "Memory", F::Envelope, Ids{ ParamId::MemoryLevel, ParamId::MemoryTime, ParamId::MemoryBloom, ParamId::MemoryBlur,
                                                    ParamId::MemorySink }, Ids{ ParamId::MemoryLevel } } });
    {
        auto* h = new HallPage(proc_, actions_, &live_);
        describe_.push_back([h] { return h->describe(); });
        tabs_.addTab("Hall", additivui::skin().panel, new ScrollPage(h, [h](int w) { return h->heightFor(w); }), true);
    }
    {
        auto* x = new MatrixPage(proc_, [this](additiv::ModSource s) { chooseSource(s, false); });
        matrixPage_ = x;
        describe_.push_back([x] { return x->describe(); });
        tabs_.addTab("Modulation", additivui::skin().panel, new ScrollPage(x, [x](int w) { return x->heightFor(w); }), true);
    }
    {
        auto* j = new JourneyPage(proc_, actions_, &live_);
        describe_.push_back([j] { return j->describe(); });
        tabs_.addTab("Journey", additivui::skin().panel, new ScrollPage(j, [j](int w) { return j->heightFor(w); }), true);
    }
    {
        auto* m = new MapPage(proc_, actions_, &live_);
        describe_.push_back([m] { return m->describe(); });
        tabs_.addTab("Map", additivui::skin().panel, new ScrollPage(m, [m](int w) { return m->heightFor(w); }), true);
    }
    tabs_.setTabBarDepth(0);   // the strip shows the pages
    tabs_.setOutline(0);
    sendLookAndFeelChange();

    // The pages' strip: the pages in their sections, each with what is on in it.
    {
        const juce::StringArray names = tabs_.getTabNames();
        auto at = [&names](const char* n) { return names.indexOf(n); };
        std::vector<PageStrip::Section> sections = { { "", { at("Focus") } },
                                                     { "Sound", { at("Cloud & Filter"), at("Motion") } },
                                                     { "Harmony", { at("Tuning"), at("Conductor") } },
                                                     { "Space", { at("Space"), at("Effects"), at("Hall") } },
                                                     { "Modulation", { at("Modulation") } },
                                                     { "Explore", { at("Journey"), at("Map") } } };
        for (auto& s : sections) s.pages.erase(std::remove(s.pages.begin(), s.pages.end(), -1), s.pages.end());
        strip_.setPages(names, std::move(sections));
        strip_.statusOf = [this](int t) { return pageStatus(t); };
        strip_.onSelect = [this](int t) {
            if (canvas_) setCanvas(false);
            if (helpView_ != nullptr && helpView_->isVisible()) showHelp(false);
            tabs_.setCurrentTabIndex(t);
            strip_.setCurrent(t);
            if (rememberSize_) proc_.userSettings().setValue("page", tabs_.getTabNames()[t]);
        };
        addAndMakeVisible(strip_);
        strip_.refresh();
        const int last = tabs_.getTabNames().indexOf(proc_.userSettings().getValue("page", "Focus"));
        if (last > 0 && juce::SystemStats::getEnvironmentVariable("ADDITIV_SHOT", {}).isEmpty()) tabs_.setCurrentTabIndex(last);
    }
    // The modulation dock and its inspector.
    dock_ = std::make_unique<ModDock>(proc_);
    addAndMakeVisible(*dock_);
    inspector_ = std::make_unique<SourceInspector>(proc_, actions_, *dock_);
    addChildComponent(*inspector_);
    dock_->sendLookAndFeelChange();   // both made before they had the frame's look
    inspector_->sendLookAndFeelChange();
    inspectorOpen_ = proc_.userSettings().getBoolValue("inspector", true);
    dock_->onChoose = [this](additiv::ModSource s) { chooseSource(s, true); };
    dock_->onToggle = [this] { setInspector(!(inspectorDocked_ ? inspectorOpen_ : overlayOpen_)); };
    dock_->setChosen(inspector_->source());
    if (matrixPage_ != nullptr) static_cast<MatrixPage*>(matrixPage_)->setChosen(inspector_->source());
    inspector_->onClose = [this] { setInspector(false); };
    inspector_->onReveal = [this](int id) { revealParam(id); };
    inspector_->onMatrix = [this] {
        const int t = tabs_.getTabNames().indexOf("Modulation");
        if (t >= 0) strip_.onSelect(t);
    };
    // The night's veil over all of it, the overlays over the veil.
    veil_.toFront(false);
    palette_.toFront(false);
    browser_.toFront(false);
    tour_.toFront(false);

    for (auto* p : proc_.getParameters()) p->addListener(this);
    frame::keepKeysForEditor(*this);   // the keys stay the editor's (Space plays, not the switch last clicked)
    setWantsKeyboardFocus(true);
    setResizable(true, true);
    setResizeLimits(800, 520, 4800, 3100);
    zoom_ = std::clamp(static_cast<float>(proc_.userSettings().getDoubleValue("zoom", 1.0)), 1.0f, 1.5f);
    if (const juce::String z = juce::SystemStats::getEnvironmentVariable("ADDITIV_SHOT_ZOOM", {}); z.isNotEmpty()) zoom_ = std::clamp(z.getFloatValue(), 1.0f, 1.5f);
    const auto size = startSize();
    setSize(size.x, size.y);
    if (zoom_ != 1.0f) juce::AudioProcessorEditor::setScaleFactor(hostScale_ * zoom_);
    shotPrefix_ = juce::SystemStats::getEnvironmentVariable("ADDITIV_SHOT", {});
    rememberSize_ = shotPrefix_.isEmpty() && juce::SystemStats::getEnvironmentVariable("ADDITIV_LAYOUT", {}).isEmpty();
    if (!rememberSize_) tourAsked_ = true;   // no tour in the screenshot and layout modes
    startTimerHz(15);
    timerCallback();
}

std::vector<ParamPage*> AdditivEditor::pagesOn(int tab)
{
    // Looked up each time: a page that rebuilds its knobs (Focus) makes new ones.
    std::vector<ParamPage*> pages;
    if (auto* view = dynamic_cast<juce::Viewport*>(tabs_.getTabContentComponent(tab))) findParamPages(view->getViewedComponent(), pages);
    return pages;
}

void AdditivEditor::modDropped(additiv::ModSource s, int, int row)
{
    if (row < 0)
        juce::AlertWindow::showMessageBoxAsync(juce::MessageBoxIconType::NoIcon, "Empyrean", "All sixteen rows of the matrix are taken: free one on the Modulation page.",
                                               "Close");
    else
        chooseSource(s, false);
}

AdditivEditor::~AdditivEditor()
{
    if (windowKeeper_.window != nullptr) windowKeeper_.window->removeComponentListener(&windowKeeper_);
    proc_.userSettings().saveIfNeeded();   // the window's size and place
    removeMouseListener(&activity_);
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
        if (!history_.open()) history_.begin(additiv::paramDesc(static_cast<ParamId>(index)).name, values(), matrixText(), false);
        history_.touch(index);
    } else if (history_.open()) {
        endStep();
    }
}

void AdditivEditor::apply(const frame::UndoStep* step, bool after)
{
    if (step == nullptr) return;
    applying_ = true;
    for (const auto& [id, before, now] : step->values) proc_.setValue(id, after ? now : before);
    if (step->extraBefore != step->extraAfter) proc_.engine().setModText((after ? step->extraAfter : step->extraBefore).toStdString());
    applying_ = false;
    lastMatrix_ = matrixText();
    pendingBefore_.clear();
}

void AdditivEditor::endStep()
{
    history_.end(values(), matrixText());
    lastMatrix_ = matrixText();
    pendingBefore_.clear();
}

void AdditivEditor::watchMatrix()
{
    const juce::String now = matrixText();
    const juce::String key = juce::String(proc_.currentPreset()) + "|" + proc_.currentUserPreset();
    // What a preset, a journey, the map, a morph or a preview does to the matrix is theirs, not an edit.
    const bool blended = proc_.value(static_cast<int>(ParamId::MapActive)) >= 0.5f || proc_.value(static_cast<int>(ParamId::MorphActive)) >= 0.5f
                         || proc_.journeyStatus().isNotEmpty() || proc_.comparing() || proc_.previewing() >= 0;
    if (key != presetKey_ || blended || applying_ || history_.open()) {
        presetKey_ = key;
        lastMatrix_ = now;
        pendingBefore_.clear();
        return;
    }
    if (now != lastMatrix_) {
        if (pendingBefore_.isEmpty()) pendingBefore_ = lastMatrix_;
        pendingAt_ = juce::Time::getMillisecondCounterHiRes();
        lastMatrix_ = now;
    }
    if (pendingBefore_.isNotEmpty() && juce::Time::getMillisecondCounterHiRes() - pendingAt_ > 700.0) flushMatrix();
}

void AdditivEditor::flushMatrix()
{
    if (pendingBefore_.isEmpty()) return;
    const juce::String before = pendingBefore_;
    pendingBefore_.clear();
    if (before == matrixText()) return;
    history_.begin("Modulation", values(), before, true);
    history_.end(values(), matrixText());
    lastMatrix_ = matrixText();
}

void AdditivEditor::timerCallback()
{
    additiv::Engine& e = proc_.engine();
    const bool on = proc_.value(static_cast<int>(ParamId::BrainOn)) >= 0.5f;
    play_.show(on ? frame::IconButton::Icon::Stop : frame::IconButton::Icon::Play, on ? "Autoplay: stop the conductor" : "Autoplay: start the conductor");
    const int cp = proc_.currentPreset();
    juce::String name = cp >= 0 ? juce::String(additiv::kPresetGroupNames[additiv::preset(cp).group]) + ": " + additiv::preset(cp).name
                      : proc_.currentUserPreset().isNotEmpty() ? "Yours: " + proc_.currentUserPreset() : juce::String("Presets ...");
    if ((cp >= 0 || proc_.currentUserPreset().isNotEmpty()) && proc_.modified()) name << " *";   // moved from its preset
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
          << " played)   root " << kNames[((root % 12) + 12) % 12] << (root / 12 - 1);
        // The machine only when it matters: a block that took more than half its time, or audio that broke.
        const double blockMs = 1000.0 * e.config().gpuBlock / std::max(e.config().sampleRate, 1.0);
        if (e.lastGpuMs() > 0.5 * blockMs || e.underruns() > 0) s << "   GPU " << juce::String(e.lastGpuMs(), 2) << " ms";
        if (e.underruns() > 0) s << "   underruns " << static_cast<int>(e.underruns());
        status_.setTooltip("GPU " + juce::String(e.lastGpuMs(), 3) + " ms a block of " + juce::String(blockMs, 2) + " ms   latency "
                           + juce::String(e.latencySamples() * 1000.0 / std::max(e.config().sampleRate, 1.0), 1) + " ms   " + e.deviceName());
        const juce::String journey = proc_.journeyStatus();
        if (journey.isNotEmpty()) s << "      Journey: " << journey;
        else if (e.morphProgress() < 1.0f) s << "      gliding " << juce::roundToInt(100.0f * e.morphProgress()) << " %";
    }
    status_.setText(s, juce::dontSendNotification);
    {   // the meter, the comparison, the Step's countdown, the matrix's undo
        float pl = 0.0f, pr = 0.0f;
        proc_.takePeaks(pl, pr);
        meter_.push(pl, pr);
        compare_.setEnabled(proc_.modified() || proc_.comparing());
        const float next = e.conductorNext(), span = std::max(e.conductorSpan(), 0.001f);
        stepLeft_ = on && e.prepared() ? std::clamp(next / span, 0.0f, 1.0f) : -1.0f;
        step_.setTooltip(stepLeft_ >= 0.0f ? "Autoplay: exchange one voice of the chord now (the next in " + juce::String(juce::roundToInt(next)) + " s)"
                                           : juce::String("Autoplay: exchange one voice of the chord now"));
        repaint(step_.getBounds().expanded(2, 4));
        watchMatrix();
    }
    cloud_.setPage(canvas_ ? juce::String() : tabs_.getCurrentTabName());   // the picture shows what the page in front edits
    strip_.setCurrent(tabs_.getCurrentTabIndex());
    if (logTick_ % 4 == 0) strip_.refresh();   // what is on in every page, four times a second
    // The night: the picture calmer (half as often), the veil -- deeper in a canvas left alone for a minute and a half.
    if (!night_ || (++refreshTick_ & 1) == 0) cloud_.refresh();
    float veil = 0.0f;
    if (night_) {
        veil = 0.5f;
        const double idle = (juce::Time::getMillisecondCounterHiRes() - lastActivity_) * 0.001;
        if (canvas_ && idle > 90.0) veil = static_cast<float>(std::min(0.9, 0.5 + (idle - 90.0) / 20.0 * 0.4));
    }
    if (std::abs(veil - veil_.alpha) > 0.004f) {
        veil_.alpha = veil;
        veil_.setVisible(veil > 0.0f);
        veil_.repaint();
    }
    // ADDITIV_LAYOUT: the pages measured, then the standalone quits.
    if (const juce::String layout = juce::SystemStats::getEnvironmentVariable("ADDITIV_LAYOUT", {}); layout.isNotEmpty() && !layoutChecked_ && isShowing()) {
        layoutChecked_ = true;
        layoutCheck(juce::File(layout));
        return;
    }
    // The first steps, once, on the first start (not in the shot mode).
    if (!tourAsked_ && shotPrefix_.isEmpty() && !layoutChecked_) {
        tourAsked_ = true;
        if (!proc_.userSettings().getBoolValue("tourDone", false))
            juce::Timer::callAfterDelay(1200, [safe = juce::Component::SafePointer<AdditivEditor>(this)] { if (safe != nullptr) safe->startTour(); });
    }
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
        // From the defaults, whatever the standalone's last session left (the map on, a morph, a
        // radius): the pictures are the same every time. What it left is given back before quitting.
        proc_.getStateInformation(shotSaved_);
        for (int i = 0; i < additiv::kNumParams; ++i) proc_.resetToDefault(i);
        const juce::String want = juce::SystemStats::getEnvironmentVariable("ADDITIV_PRESET", {});
        int index = -1;
        for (int i = 0; i < additiv::numPresets() && index < 0 && want.isNotEmpty(); ++i)
            if (want == additiv::preset(i).name) index = i;
        if (index < 0 && want.containsOnly("0123456789") && want.isNotEmpty()) index = want.getIntValue();
        if (index >= 0) proc_.applyPreset(index);
        // ADDITIV_ROUTE: a route by name, walked from the start.
        const juce::String routeName = juce::SystemStats::getEnvironmentVariable("ADDITIV_ROUTE", {});
        for (const auto& r : proc_.routes())
            if (routeName.isNotEmpty() && r.name == routeName && proc_.loadRoute(r)) proc_.startRoute();
        // ADDITIV_LANDSCAPE: a picture or a recording as the player's own landscape.
        const juce::String land = juce::SystemStats::getEnvironmentVariable("ADDITIV_LANDSCAPE", {});
        if (land.isNotEmpty()) {
            juce::String why;
            proc_.loadLandscape(juce::File(land), true, &why);
        }
        // ADDITIV_SET ("key=value;key=value"): parameters after the preset, a choice by its name or its index.
        for (const juce::String& item : juce::StringArray::fromTokens(juce::SystemStats::getEnvironmentVariable("ADDITIV_SET", {}), ";", ""))
            if (const additiv::ParamDesc* d = additiv::findParam(item.upToFirstOccurrenceOf("=", false, false).trim().toRawUTF8())) {
                const juce::String val = item.fromFirstOccurrenceOf("=", false, false).trim();
                float v = val.getFloatValue();
                for (int c = 0; c < d->numChoices; ++c)
                    if (val == d->choices[c]) v = static_cast<float>(c);
                proc_.setValue(static_cast<int>(d->id), v);
            }
        // ADDITIV_SHOT_SAVE: the sound saved as the player's preset of that name, then Tilt moved (the star, the marks).
        const juce::String saveName = juce::SystemStats::getEnvironmentVariable("ADDITIV_SHOT_SAVE", {});
        if (saveName.isNotEmpty()) {
            juce::String why;
            proc_.saveUserPreset(saveName, why);
            proc_.setValue(static_cast<int>(ParamId::Tilt), proc_.value(static_cast<int>(ParamId::Tilt)) + 0.4f);
        }
        // ADDITIV_SHOT_SHOW: a knob shown as the palette shows it (its page, lit, told about in the info line).
        if (const additiv::ParamDesc* shown = additiv::findParam(juce::SystemStats::getEnvironmentVariable("ADDITIV_SHOT_SHOW", {}).toRawUTF8())) {
            const int id = static_cast<int>(shown->id);
            juce::Timer::callAfterDelay(3000, [safe = juce::Component::SafePointer<AdditivEditor>(this), id] { if (safe != nullptr) safe->revealParam(id); });
        }
        // ADDITIV_SHOT_NIGHT: the night mode, for this run only; ADDITIV_SHOT_SNAP: two snapshots a few seconds in;
        // ADDITIV_SHOT_TOUR: the tour's first step.
        if (juce::SystemStats::getEnvironmentVariable("ADDITIV_SHOT_NIGHT", {}).isNotEmpty()) night_ = true;
        if (juce::SystemStats::getEnvironmentVariable("ADDITIV_SHOT_SNAP", {}).isNotEmpty())
            for (int ms : { 2000, 2600 })
                juce::Timer::callAfterDelay(ms, [safe = juce::Component::SafePointer<AdditivEditor>(this)] {
                    if (safe != nullptr) safe->proc_.takeSnapshot(safe->cloud_.thumbnail(160, 60));
                });
        if (juce::SystemStats::getEnvironmentVariable("ADDITIV_SHOT_TOUR", {}).isNotEmpty())
            juce::Timer::callAfterDelay(1500, [safe = juce::Component::SafePointer<AdditivEditor>(this)] { if (safe != nullptr) safe->startTour(); });
        // ADDITIV_SHOT_SOURCE: a source in the inspector (by its name), opened.
        const juce::String sourceName = juce::SystemStats::getEnvironmentVariable("ADDITIV_SHOT_SOURCE", {});
        for (int k = 1; k < additiv::kNumModSources && sourceName.isNotEmpty(); ++k)
            if (sourceName == additiv::kModSourceNames[k])
                juce::Timer::callAfterDelay(1000, [safe = juce::Component::SafePointer<AdditivEditor>(this), k] {
                    if (safe != nullptr) safe->chooseSource(static_cast<additiv::ModSource>(k), false);
                });
        // ADDITIV_SHOT_DROP ("LFO 2>cutoff"): a source dropped on a knob, as the dock's chip drops it.
        const juce::String drop = juce::SystemStats::getEnvironmentVariable("ADDITIV_SHOT_DROP", {});
        if (drop.contains(">"))
            juce::Timer::callAfterDelay(1500, [safe = juce::Component::SafePointer<AdditivEditor>(this), drop] {
                if (safe == nullptr) return;
                const additiv::ParamDesc* d = additiv::findParam(drop.fromFirstOccurrenceOf(">", false, false).trim().toRawUTF8());
                int src = 0;
                for (int k = 1; k < additiv::kNumModSources; ++k)
                    if (drop.upToFirstOccurrenceOf(">", false, false).trim() == additiv::kModSourceNames[k]) src = k;
                if (d == nullptr || src == 0 || !safe->revealParam(static_cast<int>(d->id))) return;
                for (ParamPage* p : safe->pagesOn(safe->tabs_.getCurrentTabIndex()))   // the page revealParam showed
                    if (juce::Component* c = p->controlFor(static_cast<int>(d->id))) {
                        p->itemDropped({ knobmod::dragOf(static_cast<additiv::ModSource>(src)), safe->dock_.get(), c->getBounds().getCentre() });
                        break;
                    }
            });
        // ADDITIV_SHOT_JOURNEY: a journey under way, by its name (the timeline's picture).
        const juce::String journeyName = juce::SystemStats::getEnvironmentVariable("ADDITIV_SHOT_JOURNEY", {});
        for (const auto& entry : proc_.journeys())
            if (journeyName.isNotEmpty() && entry.name == journeyName) {
                proc_.startJourney(entry);
                // ADDITIV_SHOT_DRAFT: and a copy of it as the journey being written (the timeline's second band).
                if (juce::SystemStats::getEnvironmentVariable("ADDITIV_SHOT_DRAFT", {}).isNotEmpty()) {
                    additiv::Journey copy = proc_.journeyPlayer().journey();
                    copy.name += " (mine)";
                    proc_.setJourneyDraft(copy);
                }
                break;
            }
        // ADDITIV_SHOT_UNDO: after a drop (ADDITIV_SHOT_DROP), Undo at four seconds -- the matrix as it was.
        if (juce::SystemStats::getEnvironmentVariable("ADDITIV_SHOT_UNDO", {}).isNotEmpty())
            juce::Timer::callAfterDelay(4000, [safe = juce::Component::SafePointer<AdditivEditor>(this)] {
                if (safe == nullptr) return;
                safe->flushMatrix();
                safe->apply(safe->history_.undo(), false);
            });
        // ADDITIV_SHOT_BROWSER: the preset browser open.
        if (juce::SystemStats::getEnvironmentVariable("ADDITIV_SHOT_BROWSER", {}).isNotEmpty())
            juce::MessageManager::callAsync([this] { browser_.open(); });
        // ADDITIV_SHOT_PALETTE: the command palette open with that typed into it.
        const juce::String query = juce::SystemStats::getEnvironmentVariable("ADDITIV_SHOT_PALETTE", {});
        if (query.isNotEmpty()) juce::MessageManager::callAsync([this, query] {
            openPalette();
            palette_.type(query);
        });
        // ADDITIV_SHOT_VARY: the Focus page's six variations made, for their picture.
        if (juce::SystemStats::getEnvironmentVariable("ADDITIV_SHOT_VARY", {}).isNotEmpty()) proc_.makeVariations(0.3f);
    }
    if (shotPrefix_.isEmpty() || ++shotTick_ < start || (shotTick_ - start) % 3 != 0) return;
    const int t = (shotTick_ - start) / 3;
    if (t > 0) {
        const juce::Image img = createComponentSnapshot(getLocalBounds());
        const juce::File f(shotPrefix_ + "-" + juce::String(t - 1) + "-" + (canvas_ ? juce::String("Canvas") : tabs_.getCurrentTabName()) + ".png");
        f.deleteFile();
        juce::FileOutputStream out(f);
        if (out.openedOk()) juce::PNGImageFormat().writeImageToStream(img, out);
    }
    if (t == tabs_.getNumTabs() && !canvas_ && juce::SystemStats::getEnvironmentVariable("ADDITIV_SHOT_CANVAS", {}).isNotEmpty()) {
        setCanvas(true);   // ADDITIV_SHOT_CANVAS: the canvas last, photographed on the next round
        shotTick_ -= 3;
        return;
    }
    if (t < tabs_.getNumTabs()) {
        tabs_.setCurrentTabIndex(t);
        // ADDITIV_SHOT_FULL: the window as tall as the page, so that nothing of it scrolls out of the picture.
        if (juce::SystemStats::getEnvironmentVariable("ADDITIV_SHOT_FULL", {}).isNotEmpty())
            if (auto* sp = dynamic_cast<ScrollPage*>(tabs_.getTabContentComponent(t)))
                setSize(getWidth(), std::clamp(getHeight() - sp->getMaximumVisibleHeight() + sp->pageHeight() + 16, 860, 3100));
    } else {
        shotPrefix_ = {};
        if (shotSaved_.getSize() > 0) proc_.setStateInformation(shotSaved_.getData(), static_cast<int>(shotSaved_.getSize()));
        if (standalone()) juce::JUCEApplicationBase::quit();
    }
}

void AdditivEditor::layoutCheck(const juce::File& out)
{
    const auto before = getBounds();
    juce::String report;
    int scrolling = 0;
    const int sizes[][2] = { { 1280, 860 }, { 1280, 800 }, { 1280, 760 }, { 1440, 900 }, { 1920, 1000 } };
    for (const auto& s : sizes) {
        setSize(s[0], s[1]);
        report << s[0] << " x " << s[1] << "\n";
        for (int t = 0; t < tabs_.getNumTabs(); ++t)
            if (auto* sp = dynamic_cast<ScrollPage*>(tabs_.getTabContentComponent(t))) {
                const int need = sp->heightAt(sp->getWidth()), room = sp->getHeight();
                report << "    " << tabs_.getTabNames()[t].paddedRight(' ', 12) << " needs " << need << ", has " << room
                       << (need > room ? "   SCROLLS by " + juce::String(need - room) : juce::String()) << "\n";
                if (need > room && s[0] == 1280 && s[1] == 860) ++scrolling;
            }
    }
    {   // and the kinds of audio device the standalone offers (ASIO among them on Windows)
        juce::AudioDeviceManager devices;
        juce::OwnedArray<juce::AudioIODeviceType> types;
        devices.createAudioDeviceTypes(types);
        report << "on screen " << getScreenBounds().getWidth() << " x " << getScreenBounds().getHeight() << " at zoom " << zoom_ << "\n";
        report << "audio device types:";
        for (auto* type : types) report << " " << type->getTypeName() << ";";
        report << "\n";
    }
    out.replaceWithText(report);
    setBounds(before);
    if (standalone()) {
        juce::JUCEApplicationBase::getInstance()->setApplicationReturnValue(scrolling);
        juce::JUCEApplicationBase::quit();
    }
}

void AdditivEditor::paintOverChildren(juce::Graphics& g)
{
    if (stepLeft_ < 0.0f || canvas_ || !step_.isShowing() || palette_.isVisible() || browser_.isVisible() || tour_.isVisible()) return;
    const auto b = step_.getBounds().toFloat();
    const float w = (b.getWidth() - 8.0f) * stepLeft_;
    g.setColour(additivui::skin().accent.withAlpha(0.8f));
    g.fillRoundedRectangle(b.getX() + 4.0f, b.getBottom() + 1.0f, w, 2.0f, 1.0f);
}

void AdditivEditor::setNight(bool on)
{
    night_ = on;
    nightButton_.setToggleState(on, juce::dontSendNotification);
    proc_.userSettings().setValue("night", on);
    proc_.userSettings().saveIfNeeded();
    lastActivity_ = juce::Time::getMillisecondCounterHiRes();
}

void AdditivEditor::startTour()
{
    if (canvas_) setCanvas(false);
    tabs_.setCurrentTabIndex(0);
    std::vector<Tour::Step> s;
    s.push_back({ &play_, "Play", "The conductor plays by itself: chords in just intonation that change one voice at a time. Space starts and stops it in the standalone; play the keys over it." });
    s.push_back({ &presetName_, "6144 presets", "Click the name to browse them: by name and group, by how they sound (darker, brighter, stiller, more moving ...), by your stars and the ones played lately. Hold a row's play mark to listen." });
    s.push_back({ &cloud_, "The partials themselves", "Twenty thousand sines a voice, as the graphics card has them. The page in front draws what it edits here -- the filter's curve, the register, the air -- and you can drag it. Double-click: the canvas." });
    s.push_back({ &strip_, "Focus", "The simple ways in: nudge the sound brighter, calmer, wider or airier; the few knobs that matter for it; snapshots; variations to evolve it by ear; macros taught by example; presets found by a recording." });
    s.push_back({ &searchButton_, "Search (Ctrl+K)", "Type a few letters to find any knob (its page opens, it lights up; \"cutoff 800\" sets it), any preset or action." });
    if (info_ != nullptr) s.push_back({ info_.get(), "The info line", "Point at any control: what it does, its value and what moves it are told here." });
    s.push_back({ &save_, "Your own presets", "Save the sound as your own (Ctrl+S); a star by the name tells that it has moved from its preset, a notch on a knob where the preset had it." });
    tour_.start(std::move(s));
}

void AdditivEditor::paint(juce::Graphics& g)
{
    backdrop_.paint(g, getLocalBounds(), headerBottom_, additivui::skin(), frame::Settings::of(kApp).backdrop());
    if (additivui::skin().logo) additivui::skin().logo(g, logo_);
}

void AdditivEditor::resized()
{
    if (dock_ == nullptr) return;   // not built yet
    auto full = getLocalBounds();
    if (info_ != nullptr) {   // the info line along the bottom (not in the canvas)
        info_->setVisible(!canvas_);
        if (!canvas_) info_->setBounds(full.removeFromBottom(24));
    }
    auto area = full.reduced(12, 8);
    frame::Header h;
    h.logo = &logo_;
    h.title = &title_;
    h.titleWidth = static_cast<int>(frame::titleWidth(additivui::skin(), frame::kHeaderRow * 0.62f)) + 8;
    h.choices = { { &prevPreset_, 28 }, { &presetName_, 250 }, { &nextPreset_, 28 }, { &compare_, 32 }, { &mode_, 92 }, { &scale_, 150 }, { &root_, 58 }, { &motion_, 104 } };
    landscape_.setVisible(false);   // on the Motion page; the header has room for the bank
    h.actions = { { &step_, 64 } };
    h.play = &play_;
    h.status = &status_;
    h.tools = { &undo_, &redo_, nullptr, &searchButton_, &canvasButton_, &nightButton_, &midi_, nullptr, &save_, &help_, &settings_ };
    frame::layoutHeader(area, h);
    {   // the meter at the status line's right end
        auto sb = status_.getBounds();
        meter_.setBounds(sb.removeFromRight(110).reduced(4, 5));
        status_.setBounds(sb);
    }
    headerBottom_ = area.getY();
    palette_.setBounds(getLocalBounds());
    veil_.setBounds(getLocalBounds());
    browser_.setBounds(getLocalBounds());
    strip_.setVisible(!canvas_);
    dock_->setVisible(!canvas_);
    if (canvas_) inspector_->setVisible(false);
    if (rememberSize_ && juce::Desktop::getInstance().getKioskModeComponent() == nullptr) {
        // The size it is left at -- not a maximised one: that is kept in the window's state (restoreWindow).
        const auto* window = findParentComponentOfClass<juce::ResizableWindow>();
        if (window == nullptr || !window->isFullScreen()) {
            proc_.userSettings().setValue("windowWidth", getWidth());
            proc_.userSettings().setValue("windowHeight", getHeight());
        }
    }
    if (canvas_) {   // the picture is the window
        cloud_.setBounds(area);
        tour_.setBounds(getLocalBounds());
        return;
    }
    // The dock along the bottom. The picture a third of the room (at most 420 high), but never what a
    // page needs: the tallest page at this width fits under it while the picture can be 100 high. The inspector docked
    // over the dock where that leaves the picture 150 high, else it opens over the pages when wanted.
    const auto dockRow = area.removeFromBottom(ModDock::kHeight);
    dock_->setBounds(dockRow);
    area.removeFromBottom(4);
    int need = 0;
    for (int t = 0; t < tabs_.getNumTabs(); ++t)
        if (auto* sp = dynamic_cast<ScrollPage*>(tabs_.getTabContentComponent(t))) need = std::max(need, sp->heightAt(area.getWidth()));
    const int under = 8 + PageStrip::kHeight + need;
    inspectorDocked_ = area.getHeight() - under - (SourceInspector::kHeight + 6) >= 150;
    const bool inspect = inspectorDocked_ ? inspectorOpen_ : overlayOpen_;
    if (inspectorDocked_ && inspect) {
        inspector_->setBounds(area.removeFromBottom(SourceInspector::kHeight));
        area.removeFromBottom(6);
    } else if (!inspectorDocked_) {
        inspector_->setBounds(dockRow.getX(), dockRow.getY() - 4 - SourceInspector::kHeight, dockRow.getWidth(), SourceInspector::kHeight);
    }
    inspector_->setOverlay(!inspectorDocked_);
    inspector_->setVisible(inspect);
    dock_->setOpen(inspect);
    const int rest = area.getHeight() - under;
    cloud_.setBounds(area.removeFromTop(juce::jlimit(100, 420, std::min(area.getHeight() / 3, rest))));
    tour_.setBounds(getLocalBounds());
    area.removeFromTop(8);
    const auto pagesArea = area;
    strip_.setBounds(area.removeFromTop(PageStrip::kHeight));
    tabs_.setBounds(area);
    if (helpView_ != nullptr) helpView_->setBounds(pagesArea);
}

juce::Point<int> AdditivEditor::startSize()
{
    if (juce::SystemStats::getEnvironmentVariable("ADDITIV_SHOT", {}).isNotEmpty() || juce::SystemStats::getEnvironmentVariable("ADDITIV_LAYOUT", {}).isNotEmpty()) {
        // ADDITIV_SHOT_SIZE ("1920x1200"): the pictures at another size.
        const juce::String size = juce::SystemStats::getEnvironmentVariable("ADDITIV_SHOT_SIZE", {});
        const int w = size.upToFirstOccurrenceOf("x", false, false).getIntValue(), h = size.fromFirstOccurrenceOf("x", false, false).getIntValue();
        return w >= 800 && h >= 520 ? juce::Point<int>(w, h) : juce::Point<int>(1280, 860);
    }
    const juce::Displays::Display* d = juce::Desktop::getInstance().getDisplays().getPrimaryDisplay();
    const auto screen = d != nullptr ? d->userBounds.toNearestInt() : juce::Rectangle<int>(0, 0, 1920, 1040);
    const auto user = juce::Rectangle<int>(juce::roundToInt(static_cast<float>(screen.getWidth()) / zoom_), juce::roundToInt(static_cast<float>(screen.getHeight()) / zoom_));
    const int maxW = std::max(800, user.getWidth() - 40), maxH = std::max(520, user.getHeight() - 80);   // room for the host's frame
    const int lastW = proc_.userSettings().getIntValue("windowWidth", 0), lastH = proc_.userSettings().getIntValue("windowHeight", 0);
    // The size it was left at, where it fits a screen: any of them, as it may have been on a larger one than the main one.
    int roomW = user.getWidth(), roomH = user.getHeight();
    for (const auto& other : juce::Desktop::getInstance().getDisplays().displays) {
        roomW = std::max(roomW, juce::roundToInt(static_cast<float>(other.userBounds.getWidth()) / zoom_));
        roomH = std::max(roomH, juce::roundToInt(static_cast<float>(other.userBounds.getHeight()) / zoom_));
    }
    if (lastW >= 800 && lastH >= 520 && lastW <= roomW && lastH <= roomH) return { lastW, lastH };
    const int w = std::clamp(juce::roundToInt(static_cast<float>(user.getWidth()) * 0.75f), 1280, 2200);
    const int h = std::max(860, juce::roundToInt(static_cast<float>(w) * 0.625f));
    return { std::max(800, std::min(w, maxW)), std::max(520, std::min(h, maxH)) };
}

PageStrip::Status AdditivEditor::pageStatus(int tab)
{
    PageStrip::Status s;
    if (tab < 0 || tab >= tabs_.getNumTabs()) return s;
    juce::StringArray on, off;
    bool free = false;   // a group that never rests
    for (ParamPage* p : pagesOn(tab))
        for (const ParamPage::GroupState& g : p->groupStates()) {
            if (!g.gated) {
                free = true;
                continue;
            }
            s.dots.push_back({ g.colour, g.awake });
            (g.awake ? on : off).add(g.title);
        }
    const juce::String name = tabs_.getTabNames()[tab];
    if (name == "Modulation") {
        int rows = 0;
        for (int i = 0; i < additiv::kModRows; ++i) {
            const additiv::ModRow r = proc_.engine().modRow(i);
            if (r.source != additiv::ModSource::Off && r.target >= 0) ++rows;
        }
        if (rows > 0) s.badge = juce::String(rows);
        s.tip = rows == 0 ? juce::String("no row in use") : juce::String(rows) + (rows == 1 ? " row" : " rows") + " of 16 in use";
        return s;
    }
    if (name == "Journey" && proc_.journeyStatus().isNotEmpty()) {
        s.dots.push_back({ additivui::skin().accent, true });
        on.add("a journey (" + proc_.journeyStatus() + ")");
    }
    if (name == "Map" && proc_.routeRunning()) {
        s.dots.push_back({ additivui::skin().accent, true });
        on.add("a route");
    }
    s.quiet = !free && !s.dots.empty() && on.isEmpty();
    if (!on.isEmpty()) s.tip << "on: " << on.joinIntoString(", ");
    if (!off.isEmpty()) s.tip << (s.tip.isEmpty() ? "" : "; ") << "off: " << off.joinIntoString(", ");
    return s;
}

void AdditivEditor::chooseSource(additiv::ModSource s, bool toggle)
{
    const bool open = inspectorDocked_ ? inspectorOpen_ : overlayOpen_;
    if (toggle && open && inspector_->source() == s) {
        setInspector(false);
        return;
    }
    inspector_->show(s);
    dock_->setChosen(s);
    if (matrixPage_ != nullptr) static_cast<MatrixPage*>(matrixPage_)->setChosen(s);
    setInspector(true);
}

void AdditivEditor::setInspector(bool open)
{
    if (inspectorDocked_) {
        inspectorOpen_ = open;
        if (rememberSize_) proc_.userSettings().setValue("inspector", open);   // not from the screenshot mode
    } else {
        overlayOpen_ = open;
    }
    resized();
}

bool AdditivEditor::keyPressed(const juce::KeyPress& key)
{
    if (key == juce::KeyPress('k', juce::ModifierKeys::ctrlModifier, 0) || key == juce::KeyPress('K', juce::ModifierKeys::ctrlModifier, 0)) {
        openPalette();
        return true;
    }
    frame::Keys k;
    if (standalone()) {   // in a host, Space and F11 are the host's
        k.playStop = [this] { play_.triggerClick(); };
        k.fullScreen = [this] { toggleFullScreen(); };
    }
    k.save = [this] { quickSave(); };
    k.undo = [this] {
        flushMatrix();
        apply(history_.undo(), false);
    };
    k.redo = [this] {
        flushMatrix();
        apply(history_.redo(), true);
    };
    k.help = [this] { showHelp(helpView_ == nullptr || !helpView_->isVisible()); };
    k.escape = [this] {
        if (overlayOpen_ && !inspectorDocked_) setInspector(false);
        else if (canvas_) setCanvas(false);
        else if (helpView_ != nullptr && helpView_->isVisible()) showHelp(false);
        else if (juce::Desktop::getInstance().getKioskModeComponent() != nullptr) toggleFullScreen();
    };
    return frame::handleKey(key, k);
}

void AdditivEditor::setScaleFactor(float newScale)
{
    hostScale_ = newScale;
    juce::AudioProcessorEditor::setScaleFactor(hostScale_ * zoom_);
}

void AdditivEditor::setZoom(float z)
{
    zoom_ = std::clamp(z, 1.0f, 1.5f);
    if (rememberSize_) proc_.userSettings().setValue("zoom", static_cast<double>(zoom_));
    juce::AudioProcessorEditor::setScaleFactor(hostScale_ * zoom_);
}

void AdditivEditor::parentHierarchyChanged()
{
    // On the next turn of the message loop: the standalone's window is still putting its content in
    // when this is called. In a host the plug-in lives in the host's window and this finds nothing.
    juce::MessageManager::callAsync([safe = juce::Component::SafePointer<AdditivEditor>(this)] {
        if (safe == nullptr) return;
        if (auto* window = safe->findParentComponentOfClass<juce::DocumentWindow>()) {
            window->setTitleBarButtonsRequired(juce::DocumentWindow::minimiseButton | juce::DocumentWindow::maximiseButton
                                                   | juce::DocumentWindow::closeButton, false);
            safe->restoreWindow(*window);
        }
    });
}

void AdditivEditor::restoreWindow(juce::ResizableWindow& window)
{
    // Only the standalone's own window, once; never in the screenshot and layout modes.
    if (!standalone() || !rememberSize_ || windowKeeper_.window != nullptr) return;
    // JUCE's window has put itself on one screen at the size the editor asked for; the state kept says exactly
    // where it was -- across two screens, maximised -- and is brought back onto a screen should that one be gone.
    const juce::String state = proc_.userSettings().getValue("windowState");
    if (state.isNotEmpty()) window.restoreWindowStateFromString(state);
    windowKeeper_.owner = this;
    windowKeeper_.window = &window;
    window.addComponentListener(&windowKeeper_);
}

void AdditivEditor::WindowKeeper::componentMovedOrResized(juce::Component&, bool, bool)
{
    // Not while minimised or in full screen (the state would be the screen's); maximised is kept as such.
    if (owner == nullptr || window == nullptr || window->isMinimised() || juce::Desktop::getInstance().getKioskModeComponent() != nullptr) return;
    owner->proc_.userSettings().setValue("windowState", window->getWindowStateAsString());
}

void AdditivEditor::setCanvas(bool on)
{
    canvas_ = on;
    if (on && helpView_ != nullptr && helpView_->isVisible()) showHelp(false);
    tabs_.setVisible(!on);
    canvasButton_.setToggleState(on, juce::dontSendNotification);
    cloud_.setCanvas(on);
    resized();
}

/** @brief Every ParamPage under a component, depth first. @param c the component @param out receives them */
static void findParamPages(juce::Component* c, std::vector<ParamPage*>& out)
{
    if (c == nullptr) return;
    if (auto* p = dynamic_cast<ParamPage*>(c)) out.push_back(p);
    for (juce::Component* child : c->getChildren()) findParamPages(child, out);
}

bool AdditivEditor::revealParam(int id)
{
    if (canvas_) setCanvas(false);
    for (int t = 0; t < tabs_.getNumTabs(); ++t) {
        auto* view = dynamic_cast<juce::Viewport*>(tabs_.getTabContentComponent(t));
        if (view == nullptr) continue;
        std::vector<ParamPage*> pages;
        findParamPages(view->getViewedComponent(), pages);
        for (ParamPage* page : pages)
            if (juce::Component* control = page->controlFor(id)) {
                tabs_.setCurrentTabIndex(t);
                const auto at = view->getViewedComponent()->getLocalPoint(control, juce::Point<int>(0, 0));
                view->setViewPosition(0, std::max(0, at.y - 80));
                page->flash(id);
                if (info_ != nullptr) info_->showParam(id);
                return true;
            }
    }
    // A modulator's knob: its source in the inspector, on the Modulation page.
    if (const additiv::ModSource s = SourceInspector::sourceOf(id); s != additiv::ModSource::Off) {
        const int t = tabs_.getTabNames().indexOf("Modulation");
        if (t >= 0) tabs_.setCurrentTabIndex(t);
        chooseSource(s, false);
        if (info_ != nullptr) info_->showParam(id);
        return true;
    }
    return false;
}

void AdditivEditor::openPalette()
{
    if (helpView_ != nullptr && helpView_->isVisible()) showHelp(false);
    // Words for how a preset sounds -- "dark calm wide", "hell luftig" -- find the presets measured so.
    palette_.fromWords = [this](const juce::StringArray& words) {
        struct Word {
            const char* text;   ///< the word
            int trait;          ///< 0 brightness, 1 motion, 2 width, 3 noise
            float way;          ///< -1 less, +1 more
        };
        static const Word kWords[] = {
            { "dark", 0, -1 }, { "darker", 0, -1 }, { "dunkel", 0, -1 }, { "dusky", 0, -1 }, { "warm", 0, -1 },
            { "bright", 0, 1 }, { "brighter", 0, 1 }, { "hell", 0, 1 }, { "glassy", 0, 1 }, { "shiny", 0, 1 },
            { "calm", 1, -1 }, { "still", 1, -1 }, { "static", 1, -1 }, { "quiet", 1, -1 }, { "ruhig", 1, -1 }, { "slow", 1, -1 },
            { "moving", 1, 1 }, { "lively", 1, 1 }, { "busy", 1, 1 }, { "restless", 1, 1 }, { "bewegt", 1, 1 }, { "lebendig", 1, 1 },
            { "narrow", 2, -1 }, { "mono", 2, -1 }, { "schmal", 2, -1 }, { "eng", 2, -1 },
            { "wide", 2, 1 }, { "broad", 2, 1 }, { "spacious", 2, 1 }, { "breit", 2, 1 }, { "weit", 2, 1 },
            { "pure", 3, -1 }, { "clean", 3, -1 }, { "clear", 3, -1 }, { "rein", 3, -1 }, { "klar", 3, -1 },
            { "airy", 3, 1 }, { "noisy", 3, 1 }, { "breathy", 3, 1 }, { "luftig", 3, 1 }, { "rauschig", 3, 1 },
        };
        static const char* const kSaid[4][2] = { { "dark", "bright" }, { "calm", "moving" }, { "narrow", "wide" }, { "pure", "airy" } };
        float want[4] = { 0, 0, 0, 0 };
        bool asked = false;
        for (const juce::String& w : words)
            for (const Word& k : kWords)
                if (w == k.text) {
                    want[k.trait] = k.way;
                    asked = true;
                }
        std::vector<CommandPalette::Command> out;
        if (!asked) return out;
        juce::String said;
        for (int t = 0; t < 4; ++t)
            if (want[t] != 0.0f) said << (said.isEmpty() ? "" : ", ") << kSaid[t][want[t] > 0.0f ? 1 : 0];
        // The nearest to a spread and a fifth that way in each trait asked for; the others free.
        std::vector<std::pair<float, int>> near;
        for (int i = 0; i < additiv::numPresets(); ++i) {
            float f[4];
            if (!additiv::presetFeatures(i, f)) continue;
            float d = 0.0f;
            for (int t = 0; t < 4; ++t)
                if (want[t] != 0.0f) d += (f[t] - 1.2f * want[t]) * (f[t] - 1.2f * want[t]);
            near.emplace_back(d, i);
        }
        const size_t n = std::min<size_t>(10, near.size());
        std::partial_sort(near.begin(), near.begin() + static_cast<std::ptrdiff_t>(n), near.end());
        for (size_t k = 0; k < n; ++k) {
            const int index = near[k].second;
            const additiv::Preset& p = additiv::preset(index);
            out.push_back({ p.name, "sounds " + said + "  (" + additiv::kPresetGroupNames[p.group] + ")", "Preset", false,
                            [this, index](const juce::String&) { loadPreset(index); } });
        }
        return out;
    };
    palette_.open(paletteCommands());
}

std::vector<CommandPalette::Command> AdditivEditor::paletteCommands()
{
    std::vector<CommandPalette::Command> c;
    auto action = [&c](const juce::String& title, const juce::String& detail, std::function<void()> f) {
        c.push_back({ title, detail, "Action", false, [f](const juce::String&) { f(); } });
    };
    action("Canvas", "the picture alone, the whole window", [this] { setCanvas(!canvas_); });
    action("Play / Stop", "the conductor", [this] { play_.triggerClick(); });
    action("Step", "exchange one voice of the chord now", [this] { proc_.engine().autoplayStep(); });
    action("Save", "your own preset (Ctrl+S)", [this] { quickSave(); });
    action("Save as ...", "your own preset under a new name", [this] { saveAs(); });
    action("Back to the preset", "every knob where the preset had it", [this] {
        history_.begin("Back to the preset", values(), matrixText(), true);
        proc_.revert();
        endStep();
    });
    action("Variations", "six near the sound (Focus page)", [this] {
        proc_.makeVariations(0.3f);
        tabs_.setCurrentTabIndex(0);
    });
    action("Help", "the manual (F1)", [this] { showHelp(true); });
    action("Night mode", "the window dim and warm for a dark room", [this] { setNight(!night_); });
    action("Tour", "the first steps again", [this] { startTour(); });
    action("Snapshot", "keep the sound now (Focus page)", [this] { proc_.takeSnapshot(cloud_.thumbnail(160, 60)); });
    action("Browse presets", "by name, group, star, recent, and how they sound", [this] { browser_.open(); });
    action("Presets menu", "the bank's groups as a menu", [this] { showPresets(); });
    for (int t = 0; t < tabs_.getNumTabs(); ++t)
        action("Page: " + tabs_.getTabNames()[t], "show it", [this, t] {
            if (canvas_) setCanvas(false);
            tabs_.setCurrentTabIndex(t);
        });
    // Every knob, where it is: its page and its group.
    std::vector<juce::String> where(static_cast<size_t>(additiv::kNumParams));
    for (int t = tabs_.getNumTabs() - 1; t >= 0; --t) {   // the first page that shows it wins
        auto* view = dynamic_cast<juce::Viewport*>(tabs_.getTabContentComponent(t));
        if (view == nullptr) continue;
        std::vector<ParamPage*> pages;
        findParamPages(view->getViewedComponent(), pages);
        for (ParamPage* page : pages)
            for (int id = 0; id < additiv::kNumParams; ++id)
                if (page->controlFor(id) != nullptr) where[static_cast<size_t>(id)] = tabs_.getTabNames()[t] + " / " + page->groupOf(id);
    }
    for (int id = 0; id < additiv::kNumParams; ++id) {
        if (const additiv::ModSource s = SourceInspector::sourceOf(id); where[static_cast<size_t>(id)].isEmpty() && s != additiv::ModSource::Off)
            where[static_cast<size_t>(id)] = juce::String("Modulation / ") + additiv::kModSourceNames[static_cast<int>(s)];
        if (where[static_cast<size_t>(id)].isEmpty()) continue;
        const additiv::ParamDesc& d = additiv::paramDesc(static_cast<ParamId>(id));
        const bool number = d.kind == additiv::ParamKind::Float || d.kind == additiv::ParamKind::Int;
        c.push_back({ juce::String(d.name), where[static_cast<size_t>(id)], "Knob", number, [this, id](const juce::String& value) {
                         if (value.isNotEmpty()) {
                             history_.begin(additiv::paramDesc(static_cast<ParamId>(id)).name, values(), matrixText(), true);
                             proc_.setValue(id, value.getFloatValue());
                             endStep();
                         }
                         revealParam(id);
                     } });
    }
    // The player's presets, then the bank's.
    for (size_t i = 0; i < proc_.userPresets().size(); ++i)
        c.push_back({ proc_.userPresets()[i].name, "your preset", "Yours", false, [this, i](const juce::String&) {
                         history_.begin("Preset", values(), matrixText(), true);
                         proc_.chooseUserPreset(static_cast<int>(i));
                         endStep();
                     } });
    for (int i = 0; i < additiv::numPresets(); ++i)
        c.push_back({ juce::String(additiv::preset(i).name), juce::String(additiv::kPresetGroupNames[additiv::preset(i).group]), "Preset", false,
                      [this, i](const juce::String&) { loadPreset(i); } });
    return c;
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
    history_.begin("Preset", values(), matrixText(), true);
    proc_.choosePreset(index);
    endStep();
}

void AdditivEditor::showSaveMenu()
{
    juce::PopupMenu m;
    const juce::String own = proc_.currentUserPreset();
    m.addItem(own.isNotEmpty() ? "Save \"" + own + "\"  (Ctrl+S)" : juce::String("Save  (Ctrl+S)"), [this] { quickSave(); });
    m.addItem("Save as ...", [this] { saveAs(); });
    m.addSeparator();
    m.addItem("Back to the preset", proc_.modified(), false, [this] {
        history_.begin("Back to the preset", values(), matrixText(), true);
        proc_.revert();
        endStep();
    });
    m.addItem("Open your presets' folder", [] {
        AdditivProcessor::presetFolder().createDirectory();
        AdditivProcessor::presetFolder().startAsProcess();
    });
    m.showMenuAsync(juce::PopupMenu::Options().withTargetComponent(&save_));
}

void AdditivEditor::quickSave()
{
    const juce::String own = proc_.currentUserPreset();
    if (own.isEmpty()) { saveAs(); return; }
    juce::String why;
    if (!proc_.saveUserPreset(own, why)) juce::AlertWindow::showMessageBoxAsync(juce::MessageBoxIconType::WarningIcon, "Save", "Not saved: " + why);
}

void AdditivEditor::saveAs()
{
    const int cp = proc_.currentPreset();
    const juce::String suggestion = proc_.currentUserPreset().isNotEmpty() ? proc_.currentUserPreset() + " 2"
                                    : cp >= 0 ? juce::String("My ") + additiv::preset(cp).name : juce::String("My Sound");
    askName_ = std::make_unique<juce::AlertWindow>("Save as your own preset", "Its name (in Documents/Empyrean/Presets; journeys find it by it):",
                                                   juce::MessageBoxIconType::NoIcon, this);
    askName_->addTextEditor("name", suggestion);
    askName_->addButton("Save", 1, juce::KeyPress(juce::KeyPress::returnKey));
    askName_->addButton("Cancel", 0, juce::KeyPress(juce::KeyPress::escapeKey));
    askName_->enterModalState(true, juce::ModalCallbackFunction::create([this](int result) {
        const juce::String name = askName_ != nullptr ? askName_->getTextEditorContents("name") : juce::String();
        askName_.reset();
        if (result != 1) return;
        juce::String why;
        if (!proc_.saveUserPreset(name, why)) juce::AlertWindow::showMessageBoxAsync(juce::MessageBoxIconType::WarningIcon, "Save", "Not saved: " + why);
    }), false);
}

void AdditivEditor::showPresets()
{
    juce::PopupMenu bank;
    const int current = proc_.currentPreset();
    proc_.refreshUserPresets();
    if (!proc_.userPresets().empty()) {   // the player's own first
        bank.addSectionHeader("Yours");
        juce::PopupMenu yours;
        for (size_t i = 0; i < proc_.userPresets().size(); ++i)
            yours.addItem(100000 + static_cast<int>(i) + 1, proc_.userPresets()[i].name, true, proc_.userPresets()[i].name == proc_.currentUserPreset());
        bank.addSubMenu("Your presets", yours, true, nullptr, proc_.currentUserPreset().isNotEmpty());
    }
    for (int g = 0; g < additiv::kNumPresetGroups; ++g) {
        if (g == 0) bank.addSectionHeader("Sounds");
        if (g == additiv::kNumSoundGroups) bank.addSectionHeader("Living Systems");
        if (g == additiv::kFirstSleepGroup) bank.addSectionHeader("Sleep Concerts");
        if (g == additiv::kFirstVisionGroup) bank.addSectionHeader("Visions");
        if (g == additiv::kFirstSwarmGroup) bank.addSectionHeader("Swarms & Currents");
        if (g == additiv::kFirstHallGroup) bank.addSectionHeader("Halls & Orbits");
        juce::PopupMenu group;
        for (int i = 0; i < 64; ++i) {
            const int index = additiv::presetAt(g, i);
            if (index < 0) break;
            group.addItem(index + 1, additiv::preset(index).name, true, index == current);
        }
        bank.addSubMenu(additiv::kPresetGroupNames[g], group, true, nullptr, current >= 0 && additiv::preset(current).group == g);
    }
    bank.showMenuAsync(juce::PopupMenu::Options().withTargetComponent(&presetName_), [this](int id) {
        if (id > 100000) {
            history_.begin("Preset", values(), matrixText(), true);
            proc_.chooseUserPreset(id - 100001);
            endStep();
        } else if (id > 0) {
            loadPreset(id - 1);
        }
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
    m.moreItems = [this](juce::PopupMenu& menu) {
        juce::PopupMenu zoom;   // the controls larger, beside the window's size
        for (int pct : { 100, 115, 125, 150 })
            zoom.addItem(juce::String(pct) + " %", true, juce::roundToInt(zoom_ * 100.0f) == pct, [this, pct] { setZoom(static_cast<float>(pct) / 100.0f); });
        menu.addSubMenu("Zoom", zoom);
        menu.addItem("Night mode", true, night_, [this] { setNight(!night_); });
        menu.addItem("The first steps (tour)", [this] { startTour(); });
    };
    m.about = [this] {
        return juce::String("The additive ambient synth: 20,000 partials per voice on the GPU, after Noctuary.\n")
             + (proc_.engine().prepared() ? proc_.engine().deviceName() : juce::String("GPU engine not running"));
    };
    m.show(settings_);
}
