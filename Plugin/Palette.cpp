/**
 * @file Palette.cpp
 * @brief The command palette (Palette.h).
 */
#include "Palette.h"

#include "Theme.h"

#include <algorithm>

namespace {

/**
 * @brief How well one word of the search fits a text: its start best, a word's start next, anywhere
 *        in it, then its letters in order (scattered).
 * @param text  the text, lower case
 * @param word  the word, lower case
 * @return      0 when it does not fit, else the higher the better
 */
float fit(const juce::String& text, const juce::String& word)
{
    if (text.startsWith(word)) return 4.0f;
    const int at = text.indexOf(word);
    if (at > 0) return text[at - 1] == ' ' ? 3.0f : 2.0f;
    int i = 0;
    for (const juce::juce_wchar c : text) {
        if (i < word.length() && c == word[i]) ++i;
    }
    return i == word.length() ? 0.6f : 0.0f;
}

} // namespace

CommandPalette::CommandPalette() : list_({}, this)
{
    const frame::Skin& skin = additivui::skin();
    search_.setFont(juce::FontOptions(17.0f));
    search_.setTextToShowWhenEmpty("A knob, a preset or an action -- \"hall decay\", \"cutoff 800\", \"canvas\" ...", skin.dim);
    search_.setColour(juce::TextEditor::backgroundColourId, skin.raised);
    search_.setColour(juce::TextEditor::outlineColourId, skin.edge);
    search_.setColour(juce::TextEditor::focusedOutlineColourId, skin.accent);
    search_.addListener(this);
    search_.addKeyListener(this);
    list_.setRowHeight(30);
    list_.setColour(juce::ListBox::backgroundColourId, juce::Colours::transparentBlack);
    addAndMakeVisible(search_);
    addAndMakeVisible(list_);
    setInterceptsMouseClicks(true, true);
}

void CommandPalette::open(std::vector<Command> commands)
{
    all_ = std::move(commands);
    fixed_ = all_.size();
    search_.clear();
    filter();
    setVisible(true);
    toFront(true);
    search_.grabKeyboardFocus();
}

void CommandPalette::close()
{
    setVisible(false);
    all_.clear();
    shown_.clear();
    if (auto* p = getParentComponent()) p->grabKeyboardFocus();
}

juce::Rectangle<int> CommandPalette::panel() const
{
    const int w = std::min(680, getWidth() - 40), h = std::min(480, getHeight() - 120);
    return { (getWidth() - w) / 2, 70, w, h };
}

void CommandPalette::resized()
{
    auto r = panel().reduced(12);
    search_.setBounds(r.removeFromTop(36));
    r.removeFromTop(8);
    list_.setBounds(r);
}

void CommandPalette::paint(juce::Graphics& g)
{
    const frame::Skin& skin = additivui::skin();
    g.fillAll(skin.bg.withAlpha(0.55f));
    const auto p = panel().toFloat();
    g.setColour(skin.group);
    g.fillRoundedRectangle(p, skin.radius * 2.0f);
    g.setColour(skin.accent.withAlpha(0.6f));
    g.drawRoundedRectangle(p.reduced(0.5f), skin.radius * 2.0f, 1.2f);
}

void CommandPalette::mouseDown(const juce::MouseEvent& e)
{
    if (!panel().contains(e.getPosition())) close();
}

void CommandPalette::filter()
{
    // The words, and a number after them (a knob's value).
    juce::StringArray words = juce::StringArray::fromTokens(search_.getText().toLowerCase().trim(), " ", "");
    words.removeEmptyStrings();
    number_.clear();
    if (!words.isEmpty() && words[words.size() - 1].containsOnly("0123456789.-") && words[words.size() - 1].containsAnyOf("0123456789")) {
        number_ = words[words.size() - 1];
        words.remove(words.size() - 1);
    }
    std::vector<std::pair<float, int>> scored;
    // What the words make (presets by how they sound): first, in their own order.
    all_.resize(fixed_);
    if (fromWords && !words.isEmpty())
        for (Command& c : fromWords(words)) {
            scored.emplace_back(1000.0f - static_cast<float>(scored.size()), static_cast<int>(all_.size()));
            all_.push_back(std::move(c));
        }
    for (size_t i = 0; i < fixed_; ++i) {
        const Command& c = all_[i];
        if (words.isEmpty()) {   // nothing typed: the actions first, then the knobs
            if (c.kind == "Action") scored.emplace_back(1.0f, static_cast<int>(i));
            continue;
        }
        const juce::String title = c.title.toLowerCase(), detail = c.detail.toLowerCase();
        float s = 0.0f;
        bool all = true;
        for (const juce::String& w : words) {
            const float t = fit(title, w), d = fit(detail, w) * 0.4f;
            if (t <= 0.0f && d <= 0.0f) { all = false; break; }
            s += std::max(t, d);
        }
        if (!all) continue;
        if (!number_.isEmpty() && !c.takesNumber) s *= 0.5f;   // a number asks for a knob
        s += c.kind == "Action" ? 0.3f : c.kind == "Knob" ? 0.2f : c.kind == "Yours" ? 0.1f : 0.0f;
        s -= 0.002f * static_cast<float>(c.title.length());    // the shorter of two equal fits
        scored.emplace_back(s, static_cast<int>(i));
    }
    std::stable_sort(scored.begin(), scored.end(), [](const auto& a, const auto& b) { return a.first > b.first; });
    shown_.clear();
    for (size_t i = 0; i < scored.size() && i < 80; ++i) shown_.push_back(scored[i].second);
    list_.updateContent();
    list_.selectRow(shown_.empty() ? -1 : 0);
    list_.repaint();
}

void CommandPalette::run(int row)
{
    if (row < 0 || row >= static_cast<int>(shown_.size())) return;
    const Command c = all_[static_cast<size_t>(shown_[static_cast<size_t>(row)])];
    const juce::String number = c.takesNumber ? number_ : juce::String();
    close();
    if (c.run) c.run(number);
}

int CommandPalette::getNumRows() { return static_cast<int>(shown_.size()); }

void CommandPalette::paintListBoxItem(int row, juce::Graphics& g, int width, int height, bool selected)
{
    if (row < 0 || row >= static_cast<int>(shown_.size())) return;
    const frame::Skin& skin = additivui::skin();
    const Command& c = all_[static_cast<size_t>(shown_[static_cast<size_t>(row)])];
    if (selected) {
        g.setColour(skin.accent.withAlpha(0.18f));
        g.fillRoundedRectangle(juce::Rectangle<float>(0.0f, 1.0f, static_cast<float>(width), static_cast<float>(height - 2)), skin.radius);
    }
    const juce::Colour kindColour = c.kind == "Knob" ? skin.family(frame::Family::Source)
                                  : c.kind == "Action" ? skin.accent
                                  : c.kind == "Yours" ? additivui::colour::root
                                                      : skin.family(frame::Family::Space);
    g.setColour(kindColour);
    g.setFont(juce::FontOptions(11.0f, juce::Font::bold));
    g.drawText(c.kind.toUpperCase(), 8, 0, 64, height, juce::Justification::centredLeft);
    g.setColour(skin.ink);
    g.setFont(juce::FontOptions(14.0f));
    const juce::String title = c.takesNumber && number_.isNotEmpty() ? c.title + "  = " + number_ : c.title;
    g.drawText(title, 76, 0, width / 2, height, juce::Justification::centredLeft);
    g.setColour(skin.dim);
    g.setFont(juce::FontOptions(12.0f));
    g.drawText(c.detail, 76 + width / 2, 0, width / 2 - 84, height, juce::Justification::centredRight);
}

void CommandPalette::listBoxItemClicked(int row, const juce::MouseEvent&) { run(row); }

void CommandPalette::returnKeyPressed(int row) { run(row); }

void CommandPalette::textEditorTextChanged(juce::TextEditor&) { filter(); }

void CommandPalette::textEditorReturnKeyPressed(juce::TextEditor&) { run(list_.getSelectedRow()); }

void CommandPalette::textEditorEscapeKeyPressed(juce::TextEditor&) { close(); }

bool CommandPalette::keyPressed(const juce::KeyPress& key, juce::Component*)
{
    const int n = static_cast<int>(shown_.size());
    if (n == 0) return false;
    if (key.isKeyCode(juce::KeyPress::downKey)) { list_.selectRow(std::min(list_.getSelectedRow() + 1, n - 1)); return true; }
    if (key.isKeyCode(juce::KeyPress::upKey)) { list_.selectRow(std::max(list_.getSelectedRow() - 1, 0)); return true; }
    if (key.isKeyCode(juce::KeyPress::pageDownKey)) { list_.selectRow(std::min(list_.getSelectedRow() + 10, n - 1)); return true; }
    if (key.isKeyCode(juce::KeyPress::pageUpKey)) { list_.selectRow(std::max(list_.getSelectedRow() - 10, 0)); return true; }
    return false;
}
