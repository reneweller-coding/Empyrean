/**
 * @file Palette.cpp
 * @brief The command palette (Palette.h).
 * @author Rene Weller
 */
#include "Palette.h"

#include "Theme.h"

#include <algorithm>

namespace {

/**
 * @brief How well one word of the search fits a text: its start best, a word's start next, anywhere
 *        in it, then its letters in order (scattered).
 * @param[in] text  the text, lower case
 * @param[in] word  the word, lower case
 * @return 0 when it does not fit, else the higher the better
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

CommandPalette::CommandPalette() : m_list({}, this)
{
    const frame::Skin& skin = additivui::skin();
    m_search.setFont(juce::FontOptions(17.0f));
    m_search.setTextToShowWhenEmpty("A knob, a preset or an action -- \"hall decay\", \"cutoff 800\", \"canvas\" ...", skin.m_dim);
    m_search.setColour(juce::TextEditor::backgroundColourId, skin.m_raised);
    m_search.setColour(juce::TextEditor::outlineColourId, skin.m_edge);
    m_search.setColour(juce::TextEditor::focusedOutlineColourId, skin.m_accent);
    m_search.addListener(this);
    m_search.addKeyListener(this);
    m_list.setRowHeight(30);
    m_list.setColour(juce::ListBox::backgroundColourId, juce::Colours::transparentBlack);
    addAndMakeVisible(m_search);
    addAndMakeVisible(m_list);
    setInterceptsMouseClicks(true, true);
}

void CommandPalette::open(std::vector<Command> commands)
{
    m_all = std::move(commands);
    m_fixed = m_all.size();
    m_search.clear();
    filter();
    setVisible(true);
    toFront(true);
    m_search.grabKeyboardFocus();
}

void CommandPalette::close()
{
    setVisible(false);
    m_all.clear();
    m_shown.clear();
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
    m_search.setBounds(r.removeFromTop(36));
    r.removeFromTop(8);
    m_list.setBounds(r);
}

void CommandPalette::paint(juce::Graphics& g)
{
    const frame::Skin& skin = additivui::skin();
    g.fillAll(skin.m_bg.withAlpha(0.55f));
    const auto p = panel().toFloat();
    g.setColour(skin.m_group);
    g.fillRoundedRectangle(p, skin.m_radius * 2.0f);
    g.setColour(skin.m_accent.withAlpha(0.6f));
    g.drawRoundedRectangle(p.reduced(0.5f), skin.m_radius * 2.0f, 1.2f);
}

void CommandPalette::mouseDown(const juce::MouseEvent& e)
{
    if (!panel().contains(e.getPosition())) close();
}

void CommandPalette::filter()
{
    // The words, and a number after them (a knob's value).
    juce::StringArray words = juce::StringArray::fromTokens(m_search.getText().toLowerCase().trim(), " ", "");
    words.removeEmptyStrings();
    m_number.clear();
    if (!words.isEmpty() && words[words.size() - 1].containsOnly("0123456789.-") && words[words.size() - 1].containsAnyOf("0123456789")) {
        m_number = words[words.size() - 1];
        words.remove(words.size() - 1);
    }
    std::vector<std::pair<float, int>> scored;
    // What the words make (presets by how they sound): first, in their own order.
    m_all.resize(m_fixed);
    if (m_fromWords && !words.isEmpty())
        for (Command& c : m_fromWords(words)) {
            scored.emplace_back(1000.0f - static_cast<float>(scored.size()), static_cast<int>(m_all.size()));
            m_all.push_back(std::move(c));
        }
    for (size_t i = 0; i < m_fixed; ++i) {
        const Command& c = m_all[i];
        if (words.isEmpty()) {   // nothing typed: the actions first, then the knobs
            if (c.m_kind == "Action") scored.emplace_back(1.0f, static_cast<int>(i));
            continue;
        }
        const juce::String title = c.m_title.toLowerCase(), detail = c.m_detail.toLowerCase();
        float s = 0.0f;
        bool all = true;
        for (const juce::String& w : words) {
            const float t = fit(title, w), d = fit(detail, w) * 0.4f;
            if (t <= 0.0f && d <= 0.0f) { all = false; break; }
            s += std::max(t, d);
        }
        if (!all) continue;
        if (!m_number.isEmpty() && !c.m_takesNumber) s *= 0.5f;   // a number asks for a knob
        s += c.m_kind == "Action" ? 0.3f : c.m_kind == "Knob" ? 0.2f : c.m_kind == "Yours" ? 0.1f : 0.0f;
        s -= 0.002f * static_cast<float>(c.m_title.length());    // the shorter of two equal fits
        scored.emplace_back(s, static_cast<int>(i));
    }
    std::stable_sort(scored.begin(), scored.end(), [](const auto& a, const auto& b) { return a.first > b.first; });
    m_shown.clear();
    for (size_t i = 0; i < scored.size() && i < 80; ++i) m_shown.push_back(scored[i].second);
    m_list.updateContent();
    m_list.selectRow(m_shown.empty() ? -1 : 0);
    m_list.repaint();
}

void CommandPalette::run(int row)
{
    if (row < 0 || row >= static_cast<int>(m_shown.size())) return;
    const Command c = m_all[static_cast<size_t>(m_shown[static_cast<size_t>(row)])];
    const juce::String number = c.m_takesNumber ? m_number : juce::String();
    close();
    if (c.m_run) c.m_run(number);
}

int CommandPalette::getNumRows() { return static_cast<int>(m_shown.size()); }

void CommandPalette::paintListBoxItem(int row, juce::Graphics& g, int width, int height, bool selected)
{
    if (row < 0 || row >= static_cast<int>(m_shown.size())) return;
    const frame::Skin& skin = additivui::skin();
    const Command& c = m_all[static_cast<size_t>(m_shown[static_cast<size_t>(row)])];
    if (selected) {
        g.setColour(skin.m_accent.withAlpha(0.18f));
        g.fillRoundedRectangle(juce::Rectangle<float>(0.0f, 1.0f, static_cast<float>(width), static_cast<float>(height - 2)), skin.m_radius);
    }
    const juce::Colour kindColour = c.m_kind == "Knob" ? skin.family(frame::FamilyE::SOURCE)
                                  : c.m_kind == "Action" ? skin.m_accent
                                  : c.m_kind == "Yours" ? additivui::colour::Root
                                                      : skin.family(frame::FamilyE::SPACE);
    g.setColour(kindColour);
    g.setFont(juce::FontOptions(11.0f, juce::Font::bold));
    g.drawText(c.m_kind.toUpperCase(), 8, 0, 64, height, juce::Justification::centredLeft);
    g.setColour(skin.m_ink);
    g.setFont(juce::FontOptions(14.0f));
    const juce::String title = c.m_takesNumber && m_number.isNotEmpty() ? c.m_title + "  = " + m_number : c.m_title;
    g.drawText(title, 76, 0, width / 2, height, juce::Justification::centredLeft);
    g.setColour(skin.m_dim);
    g.setFont(juce::FontOptions(12.0f));
    g.drawText(c.m_detail, 76 + width / 2, 0, width / 2 - 84, height, juce::Justification::centredRight);
}

void CommandPalette::listBoxItemClicked(int row, const juce::MouseEvent&) { run(row); }

void CommandPalette::returnKeyPressed(int row) { run(row); }

void CommandPalette::textEditorTextChanged(juce::TextEditor&) { filter(); }

void CommandPalette::textEditorReturnKeyPressed(juce::TextEditor&) { run(m_list.getSelectedRow()); }

void CommandPalette::textEditorEscapeKeyPressed(juce::TextEditor&) { close(); }

bool CommandPalette::keyPressed(const juce::KeyPress& key, juce::Component*)
{
    const int n = static_cast<int>(m_shown.size());
    if (n == 0) return false;
    if (key.isKeyCode(juce::KeyPress::downKey)) { m_list.selectRow(std::min(m_list.getSelectedRow() + 1, n - 1)); return true; }
    if (key.isKeyCode(juce::KeyPress::upKey)) { m_list.selectRow(std::max(m_list.getSelectedRow() - 1, 0)); return true; }
    if (key.isKeyCode(juce::KeyPress::pageDownKey)) { m_list.selectRow(std::min(m_list.getSelectedRow() + 10, n - 1)); return true; }
    if (key.isKeyCode(juce::KeyPress::pageUpKey)) { m_list.selectRow(std::max(m_list.getSelectedRow() - 10, 0)); return true; }
    return false;
}
