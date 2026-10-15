/**
 * @file Palette.h
 * @brief The command palette (Ctrl+K), as in a code editor or Bitwig's: type a few letters and find
 *        any knob (the page opens and the knob lights up; "cutoff 800" sets it), any preset, the
 *        player's own, or an action (Canvas, Save, Variations, the night mode ...). The letters may
 *        be scattered: "mhd" finds Modal Hall Decay. Up and down choose, Enter runs, Esc closes.
 * @author Rene Weller
 */
#pragma once

#include <juce_gui_basics/juce_gui_basics.h>

#include <functional>
#include <vector>

/** @brief The command palette (see the file comment). */
class CommandPalette final : public juce::Component, private juce::ListBoxModel, private juce::TextEditor::Listener, private juce::KeyListener {
public:
    /** @brief One thing it can do. */
    struct Command {
        juce::String m_title;     ///< what it is called
        juce::String m_detail;    ///< where it is, or what it does
        juce::String m_kind;      ///< "Knob", "Preset", "Yours", "Action"
        bool m_takesNumber = false;   ///< a number after its name is its value ("cutoff 800")
        std::function<void(const juce::String& number)> m_run;   ///< what it does (the number, "" when none was typed)
    };
    /** @brief An empty palette; open() fills it. */
    CommandPalette();
    /**
     * @brief Opens it with these commands, the search empty and focused.
     * @param[in] commands  what it can do
     */
    void open(std::vector<Command> commands);
    /** @brief Commands made from the words typed (presets by how they sound), shown first; empty when the words ask for none. */
    std::function<std::vector<Command>(const juce::StringArray&)> m_fromWords;
    /** @brief Closes it. */
    void close();
    /**
     * @brief Types into the search (the shot mode's way of showing it).
     * @param[in] text  what
     */
    void type(const juce::String& text) { m_search.setText(text, true); }
    /**
     * @brief The dimmed window and the panel.
     * @param[in,out] g  the graphics context
     */
    void paint(juce::Graphics& g) override;
    /** @brief The search on top, the list under it. */
    void resized() override;
    /**
     * @brief A click beside the panel closes it.
     * @param[in] e  the click
     */
    void mouseDown(const juce::MouseEvent& e) override;

private:
    /** @brief The commands that match the search, best first. */
    void filter();
    /**
     * @brief Runs a shown command and closes.
     * @param[in] row  its row
     */
    void run(int row);
    /** @brief The panel's rectangle. @return pixels */
    juce::Rectangle<int> panel() const;
    /** @brief How many match. @return rows */
    int getNumRows() override;
    /**
     * @brief One row: the kind, the title, the detail.
     * @param[in]     row       the row
     * @param[in,out] g         the graphics context
     * @param[in]     width     the width in pixels
     * @param[in]     height    the height in pixels
     * @param[in]     selected  true for the chosen row
     */
    void paintListBoxItem(int row, juce::Graphics& g, int width, int height, bool selected) override;
    /**
     * @brief A click runs it.
     * @param[in] row  the row
     * @param[in] e    the mouse event
     */
    void listBoxItemClicked(int row, const juce::MouseEvent& e) override;
    /**
     * @brief Return runs the chosen one.
     * @param[in] row  the chosen row
     */
    void returnKeyPressed(int row) override;
    /** @brief Typing filters. */
    void textEditorTextChanged(juce::TextEditor&) override;
    /** @brief Return in the search runs the chosen one. */
    void textEditorReturnKeyPressed(juce::TextEditor&) override;
    /** @brief Esc closes. */
    void textEditorEscapeKeyPressed(juce::TextEditor&) override;
    /**
     * @brief Up and down in the search move the choice.
     * @param[in]     key     the key
     * @param[in,out] origin  the component it went to
     * @return true when the key was used
     */
    bool keyPressed(const juce::KeyPress& key, juce::Component* origin) override;

    std::vector<Command> m_all;          ///< what it can do
    size_t m_fixed = 0;                  ///< how many of them were given to open() (the rest came from the words)
    std::vector<int> m_shown;            ///< what matches, best first
    juce::String m_number;               ///< a number typed after the words
    juce::TextEditor m_search;           ///< the search
    juce::ListBox m_list;                ///< the matches
};
