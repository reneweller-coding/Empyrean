/**
 * @file Palette.h
 * @brief The command palette (Ctrl+K), as in a code editor or Bitwig's: type a few letters and find
 *        any knob (the page opens and the knob lights up; "cutoff 800" sets it), any preset, the
 *        player's own, or an action (Canvas, Save, Variations, the night mode ...). The letters may
 *        be scattered: "mhd" finds Modal Hall Decay. Up and down choose, Enter runs, Esc closes.
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
        juce::String title;     ///< what it is called
        juce::String detail;    ///< where it is, or what it does
        juce::String kind;      ///< "Knob", "Preset", "Yours", "Action"
        bool takesNumber = false;   ///< a number after its name is its value ("cutoff 800")
        std::function<void(const juce::String& number)> run;   ///< what it does (the number, "" when none was typed)
    };
    /** @brief An empty palette; open() fills it. */
    CommandPalette();
    /**
     * @brief Opens it with these commands, the search empty and focused.
     * @param commands  what it can do
     */
    void open(std::vector<Command> commands);
    /** @brief Commands made from the words typed (presets by how they sound), shown first; empty when the words ask for none. */
    std::function<std::vector<Command>(const juce::StringArray&)> fromWords;
    /** @brief Closes it. */
    void close();
    /** @brief Types into the search (the shot mode's way of showing it). @param text what */
    void type(const juce::String& text) { search_.setText(text, true); }
    /** @brief The dimmed window and the panel. */
    void paint(juce::Graphics& g) override;
    /** @brief The search on top, the list under it. */
    void resized() override;
    /** @brief A click beside the panel closes it. @param e the click */
    void mouseDown(const juce::MouseEvent& e) override;

private:
    /** @brief The commands that match the search, best first. */
    void filter();
    /** @brief Runs a shown command and closes. @param row its row */
    void run(int row);
    /** @brief The panel's rectangle. @return pixels */
    juce::Rectangle<int> panel() const;
    /** @brief How many match. @return rows */
    int getNumRows() override;
    /** @brief One row: the kind, the title, the detail. */
    void paintListBoxItem(int row, juce::Graphics& g, int width, int height, bool selected) override;
    /** @brief A click runs it. */
    void listBoxItemClicked(int row, const juce::MouseEvent& e) override;
    /** @brief Return runs the chosen one. */
    void returnKeyPressed(int row) override;
    /** @brief Typing filters. */
    void textEditorTextChanged(juce::TextEditor&) override;
    /** @brief Return in the search runs the chosen one. */
    void textEditorReturnKeyPressed(juce::TextEditor&) override;
    /** @brief Esc closes. */
    void textEditorEscapeKeyPressed(juce::TextEditor&) override;
    /** @brief Up and down in the search move the choice. */
    bool keyPressed(const juce::KeyPress& key, juce::Component* origin) override;

    std::vector<Command> all_;          ///< what it can do
    size_t fixed_ = 0;                  ///< how many of them were given to open() (the rest came from the words)
    std::vector<int> shown_;            ///< what matches, best first
    juce::String number_;               ///< a number typed after the words
    juce::TextEditor search_;           ///< the search
    juce::ListBox list_;                ///< the matches
};
