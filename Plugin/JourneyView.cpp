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
        s.fromPreset.setTooltip(juce::String("Slot ") + (k == 0 ? "A" : "B") + " takes a preset from the bank, or one of your snapshots");
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
    const auto& shots = proc_.snapshots();   // the Focus page's snapshots first: a morph between two of one's own
    if (!shots.empty()) {
        bank.addSectionHeader("Snapshots (Focus page)");
        for (size_t k = 0; k < shots.size(); ++k) bank.addItem(100000 + static_cast<int>(k), "Snapshot " + juce::String(static_cast<int>(k) + 1));
        bank.addSeparator();
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
            group.addItem(index + 1, additiv::preset(index).name);
        }
        bank.addSubMenu(additiv::kPresetGroupNames[g], group);
    }
    juce::Component* target = &slots_[slot & 1].fromPreset;
    bank.showMenuAsync(juce::PopupMenu::Options().withTargetComponent(target), [this, slot](int id) {
        if (id >= 100000) proc_.snapshotToSlot(id - 100000, slot);
        else if (id > 0) proc_.setSlotFromPreset(slot, id - 1);
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
         "next; the journeys that come with the instrument are made from the bank (Whole Night: about six and a half hours round). The timeline shows the journey chosen: a click on a step goes there. Below it, the one being written: drag a step's edge (how long it stays), its ramp (how long it glides in), the step itself (its place); right-click removes it; Edit a copy takes the journey above to change. A morph slot takes a snapshot of the Focus page too (Preset ..., first in the menu).\n";
    return s;
}

JourneyPage::JourneyPage(AdditivProcessor& proc, const frame::ControlActions& actions, frame::LiveRings* live)
    : params_(proc, actions,
              { { "Glide", frame::Family::Space, { ParamId::PresetGlide } },
                { "Morph", frame::Family::Motion, { ParamId::MorphActive, ParamId::MorphPos }, { ParamId::MorphActive } } },
              live),
      view_(proc), timeline_(proc, view_)
{
    addAndMakeVisible(params_);
    addAndMakeVisible(view_);
    addAndMakeVisible(timeline_);
}

void JourneyPage::resized()
{
    const int top = params_.heightFor(getWidth());
    params_.setBounds(0, 0, getWidth(), top);
    view_.setBounds(10, top, getWidth() - 20, JourneyView::idealHeight());
    timeline_.setBounds(10, view_.getBottom() + 10, getWidth() - 20, std::max(kTimelineMin, getHeight() - view_.getBottom() - 20));
}

// ========================================================================================================= timeline

JourneyTimeline::JourneyTimeline(AdditivProcessor& proc, const JourneyView& view) : proc_(proc), view_(view)
{
    copy_.setTooltip("Takes the journey above into the one being written, to edit here and save with the disk above");
    copy_.onClick = [this] {
        additiv::Journey j;
        if (!upper(j)) return;
        j.name = j.name.empty() ? std::string("My Journey") : j.name + " (mine)";
        proc_.setJourneyDraft(j);
        repaint();
    };
    addAndMakeVisible(copy_);
    startTimerHz(2);
}

void JourneyTimeline::timerCallback()
{
    if (isShowing()) repaint();
}

bool JourneyTimeline::upper(additiv::Journey& out) const
{
    if (proc_.journeyPlayer().running()) {
        out = proc_.journeyPlayer().journey();
        return true;
    }
    return view_.chosen(out);
}

juce::Rectangle<float> JourneyTimeline::band(int which) const
{
    auto r = getLocalBounds().toFloat().reduced(18.0f, 0.0f).withTrimmedTop(28.0f).withTrimmedBottom(6.0f);
    if (proc_.journeyDraftSteps() == 0) return which == 0 ? r.withTrimmedBottom(20.0f) : juce::Rectangle<float>();
    const float h = (r.getHeight() - 20.0f - 22.0f) / 2.0f;   // two bands, the hours under the first, a title over the second
    return which == 0 ? r.withHeight(h) : r.withTrimmedTop(h + 20.0f + 22.0f);
}

std::vector<JourneyTimeline::Block> JourneyTimeline::blocks(const additiv::Journey& j, juce::Rectangle<float> r)
{
    std::vector<Block> out;
    double total = 0.0;
    for (const additiv::JourneyStep& s : j.steps) total += 0.5 * (s.fadeLo + s.fadeHi) + 0.5 * (s.dwellLo + s.dwellHi);
    if (total <= 0.0) return out;
    double t = 0.0;
    for (const additiv::JourneyStep& s : j.steps) {
        Block b;
        b.x0 = r.getX() + r.getWidth() * static_cast<float>(t / total);
        t += 0.5 * (s.fadeLo + s.fadeHi);
        b.xg = r.getX() + r.getWidth() * static_cast<float>(t / total);
        t += 0.5 * (s.dwellLo + s.dwellHi);
        b.x1 = r.getX() + r.getWidth() * static_cast<float>(t / total);
        out.push_back(b);
    }
    return out;
}

void JourneyTimeline::drawBand(juce::Graphics& g, const additiv::Journey& j, juce::Rectangle<float> r, int lit, double done, bool faint) const
{
    const frame::Skin& skin = additivui::skin();
    const juce::Colour hue = skin.family(frame::Family::Motion);
    const auto bs = blocks(j, r);
    g.setFont(juce::FontOptions(11.0f));
    for (size_t i = 0; i < bs.size(); ++i) {
        const Block& b = bs[i];
        const bool now = static_cast<int>(i) == lit;
        const juce::Colour c = now ? hue : (faint ? skin.raised : skin.raised.brighter(0.15f));
        juce::Path ramp;   // the glide: rising into the step
        ramp.startNewSubPath(b.x0, r.getBottom());
        ramp.lineTo(b.xg, r.getY());
        ramp.lineTo(b.xg, r.getBottom());
        ramp.closeSubPath();
        g.setColour(c.withAlpha(now ? 0.35f : 0.6f));
        g.fillPath(ramp);
        g.setColour(c.withAlpha(now ? 0.35f : 0.8f));
        g.fillRect(juce::Rectangle<float>::leftTopRightBottom(b.xg, r.getY(), b.x1 - 1.0f, r.getBottom()));
        if (now) {
            g.setColour(hue.withAlpha(0.85f));
            g.fillRect(juce::Rectangle<float>::leftTopRightBottom(b.xg, r.getY(), b.xg + (b.x1 - 1.0f - b.xg) * static_cast<float>(done), r.getBottom()));
        }
        if (static_cast<int>(i) == dropAt_ && r == band(1)) {   // where a moved block would land
            g.setColour(skin.accent);
            g.fillRect(b.x0 - 1.5f, r.getY() - 2.0f, 3.0f, r.getHeight() + 4.0f);
        }
        g.setColour(now ? skin.bg : skin.ink.withAlpha(faint ? 0.55f : 0.85f));
        g.drawFittedText(j.steps[i].preset, juce::Rectangle<float>(b.xg + 4.0f, r.getY() + 2.0f, std::max(0.0f, b.x1 - b.xg - 8.0f), r.getHeight() - 4.0f).toNearestInt(),
                         juce::Justification::centredLeft, 2, 0.7f);
    }
}

void JourneyTimeline::paint(juce::Graphics& g)
{
    const frame::Skin& skin = additivui::skin();
    const auto b = getLocalBounds().toFloat();
    g.setColour(skin.group.withAlpha(0.92f));
    g.fillRoundedRectangle(b, skin.radius);
    g.setColour(skin.edge);
    g.drawRoundedRectangle(b.reduced(0.5f), skin.radius, 1.0f);
    const juce::Colour hue = skin.family(frame::Family::Motion);
    const additiv::JourneyPlayer& player = proc_.journeyPlayer();
    const bool running = player.running();
    additiv::Journey j;
    const bool any = upper(j);
    double total = 0.0;
    for (const additiv::JourneyStep& s : j.steps) total += 0.5 * (s.fadeLo + s.fadeHi) + 0.5 * (s.dwellLo + s.dwellHi);
    g.setFont(juce::FontOptions(13.0f, juce::Font::bold));
    g.setColour(running ? hue : skin.dim);
    if (!any) {
        g.drawText("Choose a journey in the box above: its steps are drawn here over time, the one under way lit while it runs.",
                   b.reduced(18.0f, 4.0f).withHeight(20.0f), juce::Justification::centredLeft);
    } else {
        juce::String head = juce::String(j.name) + "   " + juce::String(static_cast<int>(j.steps.size())) + " steps, about "
                            + juce::String(additiv::Journey::timeText(total)) + (j.cyclic ? ", then again" : "");
        if (running) head << "   -- step " << (player.step() + 1) << ", " << juce::String(additiv::Journey::timeText(player.remaining())) << " left in it (click a step: there)";
        else head << "   (not under way: a click on a step starts it there)";
        g.drawText(head, b.reduced(18.0f, 4.0f).withHeight(20.0f).withTrimmedRight(120.0f), juce::Justification::centredLeft);
        const auto r0 = band(0);
        double done = 0.0;
        if (running && player.step() >= 0 && player.step() < static_cast<int>(j.steps.size())) {
            const additiv::JourneyStep& s = j.steps[static_cast<size_t>(player.step())];
            done = juce::jlimit(0.0, 1.0, 1.0 - player.remaining() / std::max(0.5 * (s.dwellLo + s.dwellHi), 1.0));
        }
        drawBand(g, j, r0, running ? player.step() : -1, done, !running);
        g.setColour(skin.dim);   // the hours under it
        g.setFont(juce::FontOptions(11.0f));
        const double every = total > 4.0 * 3600.0 ? 3600.0 : (total > 3600.0 ? 1800.0 : 600.0);
        for (double h = 0.0; total > 0.0 && h <= total; h += every) {
            const float x = r0.getX() + r0.getWidth() * static_cast<float>(h / total);
            g.drawVerticalLine(juce::roundToInt(x), r0.getBottom(), r0.getBottom() + 4.0f);
            g.drawText(juce::String(additiv::Journey::timeText(h)), juce::Rectangle<float>(x - 30.0f, r0.getBottom() + 4.0f, 60.0f, 14.0f), juce::Justification::centred);
        }
    }
    // The one being written, to edit.
    const additiv::Journey& d = proc_.journeyDraft();
    if (!d.steps.empty()) {
        const auto r1 = band(1);
        double dt = 0.0;
        for (const additiv::JourneyStep& s : d.steps) dt += 0.5 * (s.fadeLo + s.fadeHi) + 0.5 * (s.dwellLo + s.dwellHi);
        g.setColour(skin.accent);
        g.setFont(juce::FontOptions(12.5f, juce::Font::bold));
        g.drawText("Being written: " + juce::String(d.name.empty() ? "your journey" : d.name) + ", " + juce::String(static_cast<int>(d.steps.size()))
                       + " steps, about " + juce::String(additiv::Journey::timeText(dt))
                       + "   -- drag an edge: how long it stays; the ramp: how long it glides; the block: where it goes; right-click: remove",
                   juce::Rectangle<float>(r1.getX(), r1.getY() - 21.0f, r1.getWidth(), 18.0f), juce::Justification::centredLeft);
        drawBand(g, d, r1, -1, 0.0, false);
    }
}

void JourneyTimeline::resized() { copy_.setBounds(getWidth() - 118, 5, 104, 20); }

int JourneyTimeline::draftAt(juce::Point<float> p, Drag& drag) const
{
    drag = Drag::None;
    const auto r = band(1);
    if (r.isEmpty() || !r.expanded(4.0f).contains(p)) return -1;
    const auto bs = blocks(proc_.journeyDraft(), r);
    for (size_t i = 0; i < bs.size(); ++i) {
        const Block& b = bs[i];
        if (p.x < b.x0 || p.x > b.x1 + 4.0f) continue;
        if (p.x >= b.x1 - 6.0f) drag = Drag::Dwell;
        else if (p.x < b.xg) drag = Drag::Glide;
        else drag = Drag::Move;
        return static_cast<int>(i);
    }
    return -1;
}

void JourneyTimeline::mouseMove(const juce::MouseEvent& e)
{
    Drag d = Drag::None;
    draftAt(e.position, d);
    setMouseCursor(d == Drag::Dwell || d == Drag::Glide ? juce::MouseCursor::LeftRightResizeCursor
                   : d == Drag::Move                     ? juce::MouseCursor::DraggingHandCursor
                                                         : juce::MouseCursor::NormalCursor);
}

void JourneyTimeline::mouseDown(const juce::MouseEvent& e)
{
    // The journey above: a click on a step goes there (starting the one chosen there when none runs).
    const auto r0 = band(0);
    additiv::Journey j;
    if (r0.contains(e.position) && upper(j)) {
        const auto bs = blocks(j, r0);
        for (size_t i = 0; i < bs.size(); ++i)
            if (e.position.x >= bs[i].x0 && e.position.x < bs[i].x1) {
                if (!proc_.journeyPlayer().running()) {
                    for (const auto& entry : proc_.journeys())
                        if (entry.name.toStdString() == j.name || (entry.builtin >= 0 && juce::String(j.name) == entry.name)) {
                            proc_.startJourney(entry);
                            break;
                        }
                }
                proc_.skipJourneyTo(static_cast<int>(i));
                repaint();
                return;
            }
        return;
    }
    // The one being written.
    Drag d = Drag::None;
    const int i = draftAt(e.position, d);
    if (i < 0) return;
    if (e.mods.isPopupMenu()) {
        juce::PopupMenu m;
        m.addItem("Remove " + juce::String(proc_.journeyDraft().steps[static_cast<size_t>(i)].preset), [this, i] {
            additiv::Journey dj = proc_.journeyDraft();
            dj.steps.erase(dj.steps.begin() + i);
            proc_.setJourneyDraft(dj);
            repaint();
        });
        m.showMenuAsync(juce::PopupMenu::Options().withTargetComponent(this).withTargetScreenArea(juce::Rectangle<int>(e.getScreenX(), e.getScreenY(), 1, 1)));
        return;
    }
    drag_ = d;
    dragStep_ = i;
    dragX_ = e.position.x;
    before_ = proc_.journeyDraft().steps[static_cast<size_t>(i)];
    double total = 0.0;
    for (const additiv::JourneyStep& s : proc_.journeyDraft().steps) total += 0.5 * (s.fadeLo + s.fadeHi) + 0.5 * (s.dwellLo + s.dwellHi);
    perPixel_ = total / std::max(1.0f, band(1).getWidth());
    startTimerHz(10);
}

void JourneyTimeline::mouseDrag(const juce::MouseEvent& e)
{
    if (drag_ == Drag::None || dragStep_ < 0 || dragStep_ >= proc_.journeyDraftSteps()) return;
    additiv::Journey dj = proc_.journeyDraft();
    additiv::JourneyStep& s = dj.steps[static_cast<size_t>(dragStep_)];
    const double moved = (e.position.x - dragX_) * perPixel_;   // seconds, on the scale the drag began with
    if (drag_ == Drag::Dwell || drag_ == Drag::Glide) {
        const double lo = drag_ == Drag::Dwell ? before_.dwellLo : before_.fadeLo, hi = drag_ == Drag::Dwell ? before_.dwellHi : before_.fadeHi;
        const double mean = 0.5 * (lo + hi), least = drag_ == Drag::Dwell ? 30.0 : 0.0;
        const double now = std::max(least, mean + (drag_ == Drag::Glide ? -moved : moved));   // the ramp: dragged left, a longer glide
        const double k = mean > 0.0 ? now / mean : 1.0;   // the span kept in proportion
        const double newLo = mean > 0.0 ? lo * k : now, newHi = mean > 0.0 ? hi * k : now;
        if (drag_ == Drag::Dwell) { s.dwellLo = newLo; s.dwellHi = newHi; }
        else { s.fadeLo = newLo; s.fadeHi = newHi; }
        proc_.setJourneyDraft(dj);
    } else {   // the block: where it would land
        Drag ignored = Drag::None;
        dropAt_ = draftAt({ e.position.x, band(1).getCentreY() }, ignored);
    }
    repaint();
}

void JourneyTimeline::mouseUp(const juce::MouseEvent&)
{
    if (drag_ == Drag::Move && dropAt_ >= 0 && dropAt_ != dragStep_ && dragStep_ < proc_.journeyDraftSteps()) {
        additiv::Journey dj = proc_.journeyDraft();
        const additiv::JourneyStep s = dj.steps[static_cast<size_t>(dragStep_)];
        dj.steps.erase(dj.steps.begin() + dragStep_);
        dj.steps.insert(dj.steps.begin() + std::min(dropAt_, static_cast<int>(dj.steps.size())), s);
        proc_.setJourneyDraft(dj);
    }
    drag_ = Drag::None;
    dragStep_ = -1;
    dropAt_ = -1;
    startTimerHz(2);
    repaint();
}
