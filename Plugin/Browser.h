/**
 * @file Browser.h
 * @brief The preset browser: every preset, the player's own and the bank's, found by name, by its
 *        section and group, by the favourites' star and the recent list, and by how it was measured to
 *        sound -- darker or brighter, stiller or more moving, narrower or wider, purer or airier -- and
 *        by who plays it (the conductor, or the hands). Hold a row's play mark to listen (the sound
 *        glides there and back), double-click or Enter to take it.
 */
#pragma once

#include "PluginProcessor.h"

#include <functional>
#include <vector>

/** @brief The preset browser (see the file comment). */
class PresetBrowser final : public juce::Component, private juce::ListBoxModel, private juce::TextEditor::Listener {
public:
    /** @brief A preset taken from the browser: its factory index, or the player's (index into userPresets()). */
    std::function<void(int factory, int user)> onChoose;
    /** @brief The browser of @p proc's presets. @param proc the processor */
    explicit PresetBrowser(AdditivProcessor& proc);
    /** @brief Lets go of the lists. */
    ~PresetBrowser() override;
    /** @brief Opens it: the lists anew, the search focused. */
    void open();
    /** @brief Closes it (a preset listened to is let go). */
    void close();
    /** @brief The dimmed window and the panel. */
    void paint(juce::Graphics& g) override;
    /** @brief The search, the categories, the traits, the list. */
    void resized() override;
    /** @brief A click beside the panel closes it. @param e the click */
    void mouseDown(const juce::MouseEvent& e) override;
    /** @brief Esc closes, Enter takes the chosen one. @param key the key @return true when used */
    bool keyPressed(const juce::KeyPress& key) override;

private:
    /** @brief One preset as the browser lists it. */
    struct Entry {
        int factory = -1;          ///< its index in the bank, or -1
        int user = -1;             ///< its index in the player's, or -1
        juce::String name;         ///< its name
        juce::String group;        ///< its group ("Yours" for the player's)
        int section = 0;           ///< 0 .. 5 the bank's sections, 6 the player's
        float traits[4] = {};      ///< brightness, motion, width, noise in spreads (the player's: unknown, nought)
        bool measured = false;     ///< the traits are measured
        bool hands = false;        ///< made for the hands (the conductor rests)
    };
    /** @brief One category on the left: All, Favourites, Recent, Yours, a section or a group. */
    struct Category {
        juce::String title;        ///< as listed
        int kind = 0;              ///< 0 all, 1 favourites, 2 recent, 3 yours, 4 a section, 5 a group
        int which = 0;             ///< the section's or group's number
    };
    /** @brief A row of the list: the play mark (held: listen), the star, the name, the group, the traits. */
    class Row;
    /** @brief The categories' list. */
    class Categories;
    /** @brief A range over a trait, its two ends to drag. */
    class RangeBar;

    /** @brief The entries that pass every filter, in order. */
    void filter();
    /** @brief Takes a shown entry. @param row its row */
    void choose(int row);
    /** @brief The panel's rectangle. @return pixels */
    juce::Rectangle<int> panel() const;
    /** @brief How many pass. @return rows */
    int getNumRows() override;
    /** @brief Unused: the rows are components (Row). */
    void paintListBoxItem(int, juce::Graphics&, int, int, bool) override {}
    /** @brief A row's component. */
    juce::Component* refreshComponentForRow(int row, bool selected, juce::Component* existing) override;
    /** @brief Double-click takes it. */
    void listBoxItemDoubleClicked(int row, const juce::MouseEvent&) override { choose(row); }
    /** @brief Return takes it. */
    void returnKeyPressed(int row) override { choose(row); }
    /** @brief Typing filters. */
    void textEditorTextChanged(juce::TextEditor&) override { filter(); }
    /** @brief Return in the search takes the chosen one. */
    void textEditorReturnKeyPressed(juce::TextEditor&) override { choose(list_.getSelectedRow()); }
    /** @brief Esc closes. */
    void textEditorEscapeKeyPressed(juce::TextEditor&) override { close(); }

    AdditivProcessor& proc_;                           ///< whose presets
    std::vector<Entry> all_;                           ///< every preset
    std::vector<int> shown_;                           ///< those that pass
    std::vector<Category> categories_;                 ///< the categories
    int category_ = 0;                                 ///< the one chosen
    juce::TextEditor search_;                          ///< by name or group
    std::unique_ptr<Categories> categoryModel_;        ///< the categories' list's model (made before the list, gone after it)
    std::unique_ptr<juce::ListBox> categoryList_;      ///< the categories' list
    std::array<std::unique_ptr<RangeBar>, 4> traits_;  ///< the traits' ranges
    std::array<juce::Label, 4> traitNames_;            ///< their ends' names
    juce::ComboBox who_;                               ///< who plays: anyone, the conductor, the hands
    juce::TextButton take_{ "Take" };                  ///< takes the chosen one
    juce::TextButton close_{ "Close" };                ///< closes
    juce::Label count_;                                ///< how many pass
    juce::ListBox list_;                               ///< the presets that pass
};
