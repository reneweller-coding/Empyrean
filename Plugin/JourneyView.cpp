/**
 * @file JourneyView.cpp
 * @brief The Journey page (JourneyView.h).
 */
#include "JourneyView.h"

#include "Theme.h"

#include "additiv/Presets.h"

using additiv::ParamId;

JourneyView::JourneyView(AdditivProcessor& proc) : proc_(proc)
{
    for (int k = 0; k < 2; ++k) {
        SlotRow& s = slots_[k];
        s.name.setColour(juce::Label::textColourId, additivui::skin().ink);
        s.name.setFont(juce::FontOptions(13.0f));
        s.fromSound.setTooltip(juce::String("Slot ") + (k == 0 ? "A" : "B") + " takes the sound as it plays now: every knob and the matrix");
        s.fromPreset.setTooltip(juce::String("Slot ") + (k == 0 ? "A" : "B") + " takes a preset from the bank");
        s.fromSound.onClick = [this, k] { proc_.setSlotFromSound(k); };
        s.fromPreset.onClick = [this, k] { choosePreset(k); };
        addAndMakeVisible(s.name);
        addAndMakeVisible(s.fromSound);
        addAndMakeVisible(s.fromPreset);
    }
    journeys_.setTextWhenNothingSelected("Choose a journey");
    journeys_.setTooltip("The journeys: Whole Night plays about six and a half hours before it comes round again; a group's Night is its "
                         "twelve stillest presets, its Journey twelve of its different faces. Yours are the files in Documents/Empyrean/Journeys.");
    addAndMakeVisible(journeys_);
    start_.onClick = [this] { startChosen(); };
    stop_.onClick = [this] { proc_.stopJourney(); };
    add_.setTooltip("Appends the preset playing to the journey being written (held 8-15 minutes, glided into over 1:30-3:00; edit the file for other times)");
    add_.onClick = [this] { proc_.journeyAddCurrent(); };
    save_.onClick = [this] { saveDraft(); };
    folder_.onClick = [] {
        const juce::File dir = AdditivProcessor::journeyFolder();
        dir.createDirectory();
        dir.startAsProcess();
    };
    for (juce::Label* l : { &status_, &draft_ }) {
        l->setColour(juce::Label::textColourId, additivui::skin().dim);
        l->setFont(juce::FontOptions(12.5f));
        addAndMakeVisible(l);
    }
    for (juce::Component* c : { static_cast<juce::Component*>(&start_), static_cast<juce::Component*>(&stop_), static_cast<juce::Component*>(&add_),
                                static_cast<juce::Component*>(&save_), static_cast<juce::Component*>(&folder_) })
        addAndMakeVisible(c);
    refreshJourneys();
    timerCallback();
    startTimerHz(2);
}

void JourneyView::refreshJourneys()
{
    entries_ = proc_.journeys();
    const int keep = journeys_.getSelectedId();
    journeys_.clear(juce::dontSendNotification);
    const char* const kHeads[] = { "Whole nights", "Nights", "Journeys", "Yours" };
    for (int kind = 0; kind < 4; ++kind) {
        bool headed = false;
        for (size_t i = 0; i < entries_.size(); ++i) {
            const AdditivProcessor::JourneyEntry& j = entries_[i];
            const int k = j.builtin < 0 ? 3 : j.name.startsWith("Whole Night") ? 0 : j.name.endsWith(" - Night") ? 1 : 2;
            if (k != kind) continue;
            if (!headed) { journeys_.addSectionHeading(kHeads[kind]); headed = true; }
            journeys_.addItem(j.name, static_cast<int>(i) + 1);
        }
    }
    if (keep > 0 && keep <= static_cast<int>(entries_.size())) journeys_.setSelectedId(keep, juce::dontSendNotification);
}

void JourneyView::timerCallback()
{
    for (int k = 0; k < 2; ++k) {
        const juce::String n = juce::String(k == 0 ? "A:  " : "B:  ") + proc_.slotName(k);
        if (slots_[k].name.getText() != n) slots_[k].name.setText(n, juce::dontSendNotification);
    }
    const juce::String st = proc_.journeyStatus();
    status_.setText(st.isEmpty() ? juce::String("No journey under way.") : st, juce::dontSendNotification);
    stop_.setEnabled(st.isNotEmpty());
    const int n = proc_.journeyDraftSteps();
    draft_.setText(n == 0 ? juce::String("Writing a journey: + adds the preset playing, the disk saves it.")
                          : juce::String("The journey being written: ") + juce::String(n) + (n == 1 ? " preset" : " presets"),
                   juce::dontSendNotification);
    save_.setEnabled(n > 0);
    if (++tick_ % 10 == 0) {   // every five seconds: a journey written by hand in the folder appears
        size_t files = 0;
        for (const auto& j : entries_) if (j.builtin < 0) ++files;
        if (AdditivProcessor::journeyFolder().findChildFiles(juce::File::findFiles, false, "*.journey").size() != static_cast<int>(files))
            refreshJourneys();
    }
}

void JourneyView::choosePreset(int slot)
{
    juce::PopupMenu bank;
    for (int g = 0; g < additiv::kNumPresetGroups; ++g) {
        if (g == 0) bank.addSectionHeader("Sounds");
        if (g == additiv::kNumSoundGroups) bank.addSectionHeader("Living Systems");
        if (g == additiv::kFirstSleepGroup) bank.addSectionHeader("Sleep Concerts");
        if (g == additiv::kFirstVisionGroup) bank.addSectionHeader("Visions");
        juce::PopupMenu group;
        for (int i = 0; i < 64; ++i) {
            const int index = additiv::presetAt(g, i);
            if (index < 0) break;
            group.addItem(index + 1, additiv::preset(index).name);
        }
        bank.addSubMenu(additiv::kPresetGroupNames[g], group);
    }
    juce::Component* target = &slots_[slot & 1].fromPreset;
    bank.showMenuAsync(juce::PopupMenu::Options().withTargetComponent(target), [this, slot](int id) {
        if (id > 0) proc_.setSlotFromPreset(slot, id - 1);
    });
}

void JourneyView::startChosen()
{
    const int id = journeys_.getSelectedId();
    if (id <= 0 || id > static_cast<int>(entries_.size())) {
        journeys_.showPopup();
        return;
    }
    if (!proc_.startJourney(entries_[static_cast<size_t>(id - 1)]))
        juce::AlertWindow::showMessageBoxAsync(juce::MessageBoxIconType::WarningIcon, "Journey",
                                               "This journey could not be read: a malformed line, no steps, or no preset the bank knows.");
}

void JourneyView::saveDraft()
{
    const juce::File dir = AdditivProcessor::journeyFolder();
    dir.createDirectory();
    chooser_ = std::make_unique<juce::FileChooser>("Save the journey", dir.getChildFile("My Journey.journey"), "*.journey");
    chooser_->launchAsync(juce::FileBrowserComponent::saveMode | juce::FileBrowserComponent::warnAboutOverwriting, [this](const juce::FileChooser& fc) {
        const juce::File f = fc.getResult();
        if (f == juce::File()) return;
        if (proc_.saveJourneyDraft(f.withFileExtension("journey"))) refreshJourneys();
        else juce::AlertWindow::showMessageBoxAsync(juce::MessageBoxIconType::WarningIcon, "Journey", "The journey could not be written there.");
    });
}

void JourneyView::resized()
{
    const int w = getWidth();
    for (int k = 0; k < 2; ++k) {
        const int y = 28 + k * 34;
        slots_[k].name.setBounds(14, y, w - 14 - 250 - 20, 26);
        slots_[k].fromSound.setBounds(w - 250 - 10, y, 120, 26);
        slots_[k].fromPreset.setBounds(w - 125 - 10, y, 120, 26);
    }
    const int y1 = 28 + 2 * 34 + 30, y2 = y1 + 34;
    const int boxW = std::min(360, w / 3);
    journeys_.setBounds(14, y1, boxW, 26);
    start_.setBounds(14 + boxW + 8, y1, 28, 26);
    stop_.setBounds(14 + boxW + 40, y1, 28, 26);
    status_.setBounds(14 + boxW + 78, y1, w - (14 + boxW + 78) - 10, 26);
    add_.setBounds(14, y2, 130, 26);
    save_.setBounds(152, y2, 28, 26);
    folder_.setBounds(184, y2, 28, 26);
    draft_.setBounds(220, y2, w - 230, 26);
}

void JourneyView::paint(juce::Graphics& g)
{
    const frame::Skin& skin = additivui::skin();
    g.setColour(skin.group.withAlpha(0.92f));
    g.fillRoundedRectangle(getLocalBounds().toFloat(), skin.radius);
    g.setColour(skin.edge);
    g.drawRoundedRectangle(getLocalBounds().toFloat().reduced(0.5f), skin.radius, 1.0f);
    g.setFont(juce::FontOptions(13.0f, juce::Font::bold));
    g.setColour(skin.family(frame::Family::Motion));
    g.drawText("Morph slots: what A - B blends between, the matrix included", 18, 4, getWidth() - 30, 20, juce::Justification::centredLeft);
    g.setColour(skin.family(frame::Family::Space));
    g.drawText("Journeys: presets in a row, each held a while and glided into the next -- for a whole night", 18, 28 + 2 * 34 + 6, getWidth() - 30, 20,
               juce::Justification::centredLeft);
}

juce::String JourneyView::describe() const
{
    juce::String s;
    s << "Morph slots:\n    A: " << proc_.slotName(0) << "\n    B: " << proc_.slotName(1) << "\n";
    const juce::String st = proc_.journeyStatus();
    s << "Journey: " << (st.isEmpty() ? juce::String("none under way") : st) << "\n";
    s << "\nA preset chosen glides from the sound before it over Glide (0: at once). With Morph on, the sound is the blend of the two "
         "slots at A - B: every number on its way, the filter fading from one into the other, a motion or a living landscape fading "
         "out where it changes and in again. A journey holds each of its presets for a time drawn from its range and glides into the "
         "next; the journeys that come with the instrument are made from the bank (Whole Night: about six and a half hours round).\n";
    return s;
}

JourneyPage::JourneyPage(AdditivProcessor& proc, const frame::ControlActions& actions, frame::LiveRings* live)
    : params_(proc, actions,
              { { "Glide", frame::Family::Space, { ParamId::PresetGlide } },
                { "Morph", frame::Family::Motion, { ParamId::MorphActive, ParamId::MorphPos } } },
              live),
      view_(proc)
{
    addAndMakeVisible(params_);
    addAndMakeVisible(view_);
}

void JourneyPage::resized()
{
    const int top = params_.heightFor(getWidth());
    params_.setBounds(0, 0, getWidth(), top);
    view_.setBounds(10, top, getWidth() - 20, JourneyView::idealHeight());
}
