/**
 * @file Browser.cpp
 * @brief The preset browser (Browser.h).
 */
#include "Browser.h"

#include "Theme.h"

#include "additiv/Find.h"
#include "additiv/Presets.h"

#include <cmath>

namespace {

/** @brief The bank's sections, as the menu sections them. */
const char* const kSections[6] = { "Sounds", "Living Systems", "Sleep Concerts", "Visions", "Swarms & Currents", "Halls & Orbits" };
/** @brief The traits' ends. */
const char* const kTraitLow[4] = { "darker", "stiller", "narrower", "purer" };
const char* const kTraitHigh[4] = { "brighter", "more moving", "wider", "airier" };   ///< and their other ends

} // namespace

/** @brief A row of the list (see PresetBrowser). */
class PresetBrowser::Row final : public juce::Component {
public:
    /** @brief A row of @p owner. @param owner the browser */
    explicit Row(PresetBrowser& owner) : owner_(owner) {}
    /** @brief Shows entry @p e at @p row. @param row the list's row @param e the entry, -1 none @param selected chosen */
    void show(int row, int e, bool selected)
    {
        row_ = row;
        entry_ = e;
        selected_ = selected;
        repaint();
    }
    /** @brief The play mark, the star, the name, the group, the traits. */
    void paint(juce::Graphics& g) override
    {
        if (entry_ < 0) return;
        const frame::Skin& skin = additivui::skin();
        const Entry& e = owner_.all_[static_cast<size_t>(entry_)];
        if (selected_) {
            g.setColour(skin.accent.withAlpha(0.16f));
            g.fillRoundedRectangle(getLocalBounds().toFloat().reduced(1.0f), skin.radius);
        }
        const bool listening = listening_;
        // The play mark: a triangle, lit while held.
        juce::Path tri;
        tri.addTriangle(10.0f, 8.0f, 10.0f, getHeight() - 8.0f, 22.0f, getHeight() * 0.5f);
        g.setColour(listening ? skin.accent : skin.dim);
        g.fillPath(tri);
        // The star.
        const bool fav = owner_.proc_.favorite(e.name);
        juce::Path star;
        star.addStar({ 40.0f, getHeight() * 0.5f }, 5, 3.0f, 7.0f);
        g.setColour(fav ? additivui::colour::root : skin.faint);
        if (fav) g.fillPath(star);
        else g.strokePath(star, juce::PathStrokeType(1.2f));
        g.setColour(e.user >= 0 ? additivui::colour::root : skin.ink);
        g.setFont(juce::FontOptions(14.0f));
        g.drawText(e.name, 56, 0, getWidth() / 2 - 56, getHeight(), juce::Justification::centredLeft);
        g.setColour(skin.dim);
        g.setFont(juce::FontOptions(12.0f));
        g.drawText(e.group + (e.hands ? "   (for the hands)" : ""), getWidth() / 2, 0, getWidth() / 2 - 150, getHeight(), juce::Justification::centredLeft);
        // The traits as four small bars about their middles.
        if (e.measured)
            for (int k = 0; k < 4; ++k) {
                const float x = static_cast<float>(getWidth() - 140 + k * 34), mid = getHeight() * 0.5f;
                const float v = juce::jlimit(-2.5f, 2.5f, e.traits[k]) / 2.5f * (getHeight() * 0.38f);
                g.setColour(skin.faint);
                g.drawHorizontalLine(static_cast<int>(mid), x, x + 26.0f);
                g.setColour(skin.family(frame::Family::Source).withAlpha(0.8f));
                g.fillRect(juce::Rectangle<float>(x + 8.0f, std::min(mid, mid - v), 10.0f, std::max(1.0f, std::abs(v))));
            }
    }
    /** @brief Listens (the play mark), stars, or chooses the row. @param e the press */
    void mouseDown(const juce::MouseEvent& e) override
    {
        if (entry_ < 0) return;
        owner_.list_.selectRow(row_);
        const Entry& en = owner_.all_[static_cast<size_t>(entry_)];
        if (e.x < 30) {
            listening_ = true;
            if (en.factory >= 0) owner_.proc_.previewPreset(en.factory);
            else owner_.proc_.previewUserPreset(en.user);
            repaint();
        } else if (e.x < 52) {
            owner_.proc_.setFavorite(en.name, !owner_.proc_.favorite(en.name));
            repaint();
            if (owner_.categories_[static_cast<size_t>(owner_.category_)].kind == 1) owner_.filter();
        }
    }
    /** @brief Back to the sound before the listening. @param e the release */
    void mouseUp(const juce::MouseEvent&) override
    {
        if (!listening_) return;
        listening_ = false;
        owner_.proc_.endPreview();
        repaint();
    }
    /** @brief Takes it. @param e the double click */
    void mouseDoubleClick(const juce::MouseEvent& e) override
    {
        if (e.x >= 52) owner_.choose(row_);
    }

private:
    PresetBrowser& owner_;     ///< whose
    int row_ = -1;             ///< the list's row
    int entry_ = -1;           ///< the entry shown
    bool selected_ = false;    ///< chosen
    bool listening_ = false;   ///< its play mark is held
};

/** @brief A range over a trait (see PresetBrowser): -3 .. 3 spreads, its two ends dragged, a double click opens it wide. */
class PresetBrowser::RangeBar final : public juce::Component, public juce::SettableTooltipClient {
public:
    std::function<void()> onChange;   ///< called while an end moves
    float lo = -3.0f;                 ///< the range's lower end
    float hi = 3.0f;                  ///< and its upper end
    /** @brief Whether it lets everything through. @return true when wide open */
    bool open() const { return lo <= -3.0f && hi >= 3.0f; }
    /** @brief The track, the range lit, the two ends. */
    void paint(juce::Graphics& g) override
    {
        const frame::Skin& skin = additivui::skin();
        const auto r = getLocalBounds().toFloat().reduced(6.0f, 0.0f);
        const float y = r.getCentreY();
        g.setColour(skin.faint);
        g.fillRoundedRectangle(r.getX(), y - 2.0f, r.getWidth(), 4.0f, 2.0f);
        const float x0 = xOf(lo), x1 = xOf(hi);
        g.setColour(open() ? skin.dim : skin.accent);
        g.fillRoundedRectangle(x0, y - 2.0f, x1 - x0, 4.0f, 2.0f);
        for (float x : { x0, x1 }) g.fillEllipse(x - 5.0f, y - 5.0f, 10.0f, 10.0f);
    }
    /** @brief Takes the nearer end. @param e the press */
    void mouseDown(const juce::MouseEvent& e) override
    {
        held_ = std::abs(e.position.x - xOf(lo)) <= std::abs(e.position.x - xOf(hi)) ? 0 : 1;
        mouseDrag(e);
    }
    /** @brief Moves it. @param e the drag */
    void mouseDrag(const juce::MouseEvent& e) override
    {
        const float v = valueAt(e.position.x);
        if (held_ == 0) lo = std::min(v, hi - 0.1f);
        else hi = std::max(v, lo + 0.1f);
        repaint();
        if (onChange) onChange();
    }
    /** @brief Wide open again. @param e the double click */
    void mouseDoubleClick(const juce::MouseEvent&) override
    {
        lo = -3.0f;
        hi = 3.0f;
        repaint();
        if (onChange) onChange();
    }

private:
    /** @brief A value's place. @param v spreads @return x */
    float xOf(float v) const { return 6.0f + (v + 3.0f) / 6.0f * (static_cast<float>(getWidth()) - 12.0f); }
    /** @brief The value at a place. @param x pixels @return spreads */
    float valueAt(float x) const { return juce::jlimit(-3.0f, 3.0f, (x - 6.0f) / std::max(1.0f, static_cast<float>(getWidth()) - 12.0f) * 6.0f - 3.0f); }
    int held_ = 0;   ///< the end being dragged
};

/** @brief The categories' list (see PresetBrowser). */
class PresetBrowser::Categories final : public juce::ListBoxModel {
public:
    /** @brief The categories of @p owner. @param owner the browser */
    explicit Categories(PresetBrowser& owner) : owner_(owner) {}
    /** @brief How many. @return rows */
    int getNumRows() override { return static_cast<int>(owner_.categories_.size()); }
    /** @brief One: a heading for a section, indented for a group. */
    void paintListBoxItem(int row, juce::Graphics& g, int width, int height, bool selected) override
    {
        if (row < 0 || row >= getNumRows()) return;
        const frame::Skin& skin = additivui::skin();
        const Category& c = owner_.categories_[static_cast<size_t>(row)];
        if (selected) {
            g.setColour(skin.accent.withAlpha(0.18f));
            g.fillRoundedRectangle(juce::Rectangle<float>(0.0f, 0.0f, static_cast<float>(width), static_cast<float>(height)), skin.radius);
        }
        const bool heading = c.kind == 4 || c.kind <= 3;
        g.setColour(heading ? skin.ink : skin.dim);
        g.setFont(juce::FontOptions(heading ? 13.0f : 12.0f, heading ? juce::Font::bold : juce::Font::plain));
        g.drawText(c.title, c.kind == 5 ? 18 : 6, 0, width - 20, height, juce::Justification::centredLeft);
    }
    /** @brief Chooses it. */
    void selectedRowsChanged(int row) override
    {
        if (row < 0) return;
        owner_.category_ = row;
        owner_.filter();
    }

private:
    PresetBrowser& owner_;   ///< whose
};

PresetBrowser::~PresetBrowser() = default;

PresetBrowser::PresetBrowser(AdditivProcessor& proc) : proc_(proc), list_({}, this)
{
    const frame::Skin& skin = additivui::skin();
    search_.setFont(juce::FontOptions(16.0f));
    search_.setTextToShowWhenEmpty("A name or a group ...", skin.dim);
    search_.setColour(juce::TextEditor::backgroundColourId, skin.raised);
    search_.setColour(juce::TextEditor::outlineColourId, skin.edge);
    search_.addListener(this);
    addAndMakeVisible(search_);
    categoryModel_ = std::make_unique<Categories>(*this);
    categoryList_ = std::make_unique<juce::ListBox>("categories", categoryModel_.get());
    categoryList_->setRowHeight(22);
    categoryList_->setColour(juce::ListBox::backgroundColourId, juce::Colours::transparentBlack);
    addAndMakeVisible(*categoryList_);
    for (int k = 0; k < 4; ++k) {
        traits_[static_cast<size_t>(k)] = std::make_unique<RangeBar>();
        RangeBar& s = *traits_[static_cast<size_t>(k)];
        s.setTooltip(juce::String("Only the presets measured between these ends: ") + kTraitLow[k] + " to " + kTraitHigh[k]
                     + " (in spreads of the bank; double-click: all)");
        s.onChange = [this] { filter(); };
        addAndMakeVisible(s);
        juce::Label& l = traitNames_[static_cast<size_t>(k)];
        l.setText(juce::String(kTraitLow[k]) + " ... " + kTraitHigh[k], juce::dontSendNotification);
        l.setColour(juce::Label::textColourId, skin.dim);
        l.setFont(juce::FontOptions(12.0f));
        addAndMakeVisible(l);
    }
    who_.addItem("Anyone plays", 1);
    who_.addItem("The conductor plays", 2);
    who_.addItem("For the hands", 3);
    who_.setSelectedId(1, juce::dontSendNotification);
    who_.onChange = [this] { filter(); };
    addAndMakeVisible(who_);
    take_.onClick = [this] { choose(list_.getSelectedRow()); };
    close_.onClick = [this] { close(); };
    addAndMakeVisible(take_);
    addAndMakeVisible(close_);
    count_.setColour(juce::Label::textColourId, skin.dim);
    count_.setFont(juce::FontOptions(12.0f));
    addAndMakeVisible(count_);
    list_.setRowHeight(28);
    list_.setColour(juce::ListBox::backgroundColourId, juce::Colours::transparentBlack);
    addAndMakeVisible(list_);
    setWantsKeyboardFocus(true);
}

void PresetBrowser::open()
{
    // Every preset: the player's, then the bank's with their measured traits.
    all_.clear();
    proc_.refreshUserPresets();
    for (size_t i = 0; i < proc_.userPresets().size(); ++i) {
        Entry e;
        e.user = static_cast<int>(i);
        e.name = proc_.userPresets()[i].name;
        e.group = "Yours";
        e.section = 6;
        e.hands = proc_.userPresets()[i].values[static_cast<size_t>(additiv::ParamId::BrainOn)] < 0.5f;
        all_.push_back(e);
    }
    for (int i = 0; i < additiv::numPresets(); ++i) {
        Entry e;
        e.factory = i;
        e.name = additiv::preset(i).name;
        e.group = additiv::kPresetGroupNames[additiv::preset(i).group];
        e.section = additiv::preset(i).group / 16;
        e.measured = additiv::presetFeatures(i, e.traits);
        e.hands = juce::String(additiv::preset(i).params).contains("brain_on=0");
        all_.push_back(e);
    }
    categories_.clear();
    categories_.push_back({ "All", 0, 0 });
    categories_.push_back({ "Favourites", 1, 0 });
    categories_.push_back({ "Recently played", 2, 0 });
    categories_.push_back({ "Yours", 3, 0 });
    for (int s = 0; s < 6; ++s) {
        categories_.push_back({ kSections[s], 4, s });
        for (int g = s * 16; g < (s + 1) * 16 && g < additiv::kNumPresetGroups; ++g) categories_.push_back({ additiv::kPresetGroupNames[g], 5, g });
    }
    categoryList_->updateContent();
    if (category_ >= static_cast<int>(categories_.size())) category_ = 0;
    categoryList_->selectRow(category_, true, true);
    filter();
    setVisible(true);
    toFront(true);
    search_.grabKeyboardFocus();
}

void PresetBrowser::close()
{
    proc_.endPreview();
    setVisible(false);
    if (auto* p = getParentComponent()) p->grabKeyboardFocus();
}

void PresetBrowser::filter()
{
    const juce::String q = search_.getText().trim().toLowerCase();
    const Category& c = categories_.empty() ? Category{} : categories_[static_cast<size_t>(category_)];
    const juce::StringArray recent = c.kind == 2 ? proc_.recentPresets() : juce::StringArray();
    shown_.clear();
    std::vector<int> order;
    for (size_t i = 0; i < all_.size(); ++i) {
        const Entry& e = all_[i];
        if (c.kind == 1 && !proc_.favorite(e.name)) continue;
        if (c.kind == 2 && !recent.contains(e.name)) continue;
        if (c.kind == 3 && e.user < 0) continue;
        if (c.kind == 4 && (e.factory < 0 || e.section != c.which)) continue;
        if (c.kind == 5 && (e.factory < 0 || additiv::preset(e.factory).group != c.which)) continue;
        if (q.isNotEmpty() && !e.name.toLowerCase().contains(q) && !e.group.toLowerCase().contains(q)) continue;
        const int who = who_.getSelectedId();
        if ((who == 2 && e.hands) || (who == 3 && !e.hands)) continue;
        bool inside = true;
        for (int k = 0; k < 4 && inside; ++k) {
            const RangeBar& s = *traits_[static_cast<size_t>(k)];
            if (!s.open() && (!e.measured || e.traits[k] < s.lo || e.traits[k] > s.hi)) inside = false;
        }
        if (inside) shown_.push_back(static_cast<int>(i));
    }
    if (c.kind == 2)   // the recent ones newest first
        std::stable_sort(shown_.begin(), shown_.end(), [&](int a, int b) { return recent.indexOf(all_[static_cast<size_t>(a)].name) < recent.indexOf(all_[static_cast<size_t>(b)].name); });
    list_.updateContent();
    list_.selectRow(shown_.empty() ? -1 : 0);
    list_.repaint();
    count_.setText(juce::String(static_cast<int>(shown_.size())) + " presets", juce::dontSendNotification);
}

void PresetBrowser::choose(int row)
{
    if (row < 0 || row >= static_cast<int>(shown_.size())) return;
    const Entry e = all_[static_cast<size_t>(shown_[static_cast<size_t>(row)])];
    if (onChoose) onChoose(e.factory, e.user);
    close();
}

juce::Rectangle<int> PresetBrowser::panel() const
{
    const int w = std::min(1000, getWidth() - 40), h = std::min(640, getHeight() - 80);
    return { (getWidth() - w) / 2, 50, w, h };
}

void PresetBrowser::resized()
{
    auto r = panel().reduced(12);
    auto top = r.removeFromTop(34);
    close_.setBounds(top.removeFromRight(80).reduced(2));
    take_.setBounds(top.removeFromRight(80).reduced(2));
    top.removeFromRight(8);
    search_.setBounds(top);
    r.removeFromTop(8);
    categoryList_->setBounds(r.removeFromLeft(210));
    r.removeFromLeft(10);
    auto filters = r.removeFromTop(64);
    const int w = filters.getWidth() / 5;
    for (int k = 0; k < 4; ++k) {
        auto cell = filters.withX(filters.getX() + k * w).withWidth(w).reduced(4, 0);
        traitNames_[static_cast<size_t>(k)].setBounds(cell.removeFromTop(18));
        traits_[static_cast<size_t>(k)]->setBounds(cell.removeFromTop(24));
    }
    auto last = filters.withX(filters.getX() + 4 * w).withWidth(w).reduced(4, 0);
    count_.setBounds(last.removeFromTop(18));
    who_.setBounds(last.removeFromTop(24));
    r.removeFromTop(6);
    list_.setBounds(r);
}

void PresetBrowser::paint(juce::Graphics& g)
{
    const frame::Skin& skin = additivui::skin();
    g.fillAll(skin.bg.withAlpha(0.55f));
    const auto p = panel().toFloat();
    g.setColour(skin.group);
    g.fillRoundedRectangle(p, skin.radius * 2.0f);
    g.setColour(skin.accent.withAlpha(0.5f));
    g.drawRoundedRectangle(p.reduced(0.5f), skin.radius * 2.0f, 1.2f);
}

void PresetBrowser::mouseDown(const juce::MouseEvent& e)
{
    if (!panel().contains(e.getPosition())) close();
}

bool PresetBrowser::keyPressed(const juce::KeyPress& key)
{
    if (key.isKeyCode(juce::KeyPress::escapeKey)) { close(); return true; }
    if (key.isKeyCode(juce::KeyPress::returnKey)) { choose(list_.getSelectedRow()); return true; }
    return false;
}

int PresetBrowser::getNumRows() { return static_cast<int>(shown_.size()); }

juce::Component* PresetBrowser::refreshComponentForRow(int row, bool selected, juce::Component* existing)
{
    auto* r = dynamic_cast<Row*>(existing);
    if (r == nullptr) {
        delete existing;
        r = new Row(*this);
    }
    r->show(row, row >= 0 && row < static_cast<int>(shown_.size()) ? shown_[static_cast<size_t>(row)] : -1, selected);
    return r;
}
