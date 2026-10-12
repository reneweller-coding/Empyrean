/**
 * @file PluginProcessor.h
 * @brief The JUCE shell around the engine: parameters from the one table, MIDI into stamped notes
 *        (with the family's controllers: 64 holds the notes, 74 moves the filter, and MIDI learn for
 *        the rest), the engine's latency reported to the host, the state as key=value pairs.
 *
 * The GPU opens in prepareToPlay, never in the constructor: the VST3 build step loads the plugin to
 * read its description, and a scan must not wake the GPU.
 */
#pragma once

#include "Jam.h"

#include "additiv/Engine.h"

#include <juce_audio_utils/juce_audio_utils.h>

#include <array>
#include <atomic>
#include <bitset>
#include <mutex>
#include <string>
#include <vector>

/** @brief The plugin and the standalone: the engine behind JUCE's processor interface. */
class AdditivProcessor : public juce::AudioProcessor, private juce::AsyncUpdater, private juce::Timer {
public:
    AdditivProcessor();
    ~AdditivProcessor() override;

    /** @brief Opens the engine at the host's rate and block size. @param sampleRate Hz @param samplesPerBlock the block the host announced */
    void prepareToPlay(double sampleRate, int samplesPerBlock) override;
    /** @brief Stops the engine's worker and frees the GPU. */
    void releaseResources() override;
    /** @brief Stereo out, nothing in. @param layouts the layout asked for @return true for stereo out */
    bool isBusesLayoutSupported(const BusesLayout& layouts) const override;
    /** @brief Parameters, controllers and notes into the engine, its samples out. @param buffer the output @param midi the notes of this block */
    void processBlock(juce::AudioBuffer<float>& buffer, juce::MidiBuffer& midi) override;

    /** @brief The family's panel (PluginEditor.h). @return a new editor */
    juce::AudioProcessorEditor* createEditor() override;
    /** @brief There is an editor. @return true */
    bool hasEditor() const override { return true; }

    /** @brief The plugin's name. @return the product name */
    const juce::String getName() const override { return JucePlugin_Name; }
    /** @brief It plays notes. @return true */
    bool acceptsMidi() const override { return true; }
    /** @brief It sends none. @return false */
    bool producesMidi() const override { return false; }
    /** @brief The far reverb rings on. @return seconds */
    double getTailLengthSeconds() const override { return 30.0; }

    /** @brief The factory presets are the programs. @return their count */
    int getNumPrograms() override;
    /** @brief The preset last applied. @return its index (0 when none was) */
    int getCurrentProgram() override { return std::max(current_.load(std::memory_order_relaxed), 0); }
    /** @brief Applies a factory preset (message thread). @param index the preset */
    void setCurrentProgram(int index) override { applyPreset(index); }
    /** @brief A preset's name with its group. @param index the preset @return "Group: Name" */
    const juce::String getProgramName(int index) override;
    /** @brief Nothing to rename. @param index ignored @param name ignored */
    void changeProgramName(int index, const juce::String& name) override { (void)index; (void)name; }

    /** @brief Every parameter as key=value, and the learned controllers. @param dest receives the XML */
    void getStateInformation(juce::MemoryBlock& dest) override;
    /** @brief Reads key=value; a parameter the state does not name goes to its default. @param data the XML @param size its bytes */
    void setStateInformation(const void* data, int size) override;

    /** @brief The engine, for the editor's status and pictures. @return the engine */
    additiv::Engine& engine() { return engine_; }
    /** @brief A parameter by its ParamId. @param id the parameter's index @return the host parameter, null outside the table */
    juce::RangedAudioParameter* parameter(int id) const
    {
        return id >= 0 && id < static_cast<int>(params_.size()) ? params_[static_cast<size_t>(id)] : nullptr;
    }
    /** @brief A parameter's value in its unit. @param id ParamId @return the value */
    float value(int id) const;
    /** @brief Sets a parameter in its unit, telling the host. @param id ParamId @param v the value */
    void setValue(int id, float v);

    /** @brief MIDI learn: the next controller that moves is bound to parameter @p id. @param id ParamId */
    void learn(int id) { learning_.store(id, std::memory_order_relaxed); }
    /** @brief The controller bound to a parameter. @param id ParamId @return 0 .. 127, -1 none */
    int controllerFor(int id) const;
    /** @brief Unbinds a parameter's controller. @param id ParamId */
    void forget(int id);
    /** @brief A parameter back to its default. @param id ParamId */
    void resetToDefault(int id);
    /** @brief Whether a learn is waiting for its controller. @return the parameter waiting, -1 none */
    int learning() const { return learning_.load(std::memory_order_relaxed); }

    /**
     * @brief Applies a factory preset: every parameter to the preset's value (its default where the
     *        preset says nothing), the whole matrix replaced. Message thread.
     * @param index  the preset
     */
    void applyPreset(int index);
    /** @brief The preset last applied. @return its index, -1 none */
    int currentPreset() const { return current_.load(std::memory_order_relaxed); }

    /** @brief The family jam's line for the settings menu. @return who leads, who follows, what is heard */
    juce::String jamStatus() const { return jam_.status(); }
    /** @brief Why the GPU engine did not open, for the status line. @return the reason, empty when it runs */
    juce::String engineError() const { const std::lock_guard<std::mutex> lock(errorLock_); return juce::String(engineError_); }
    /** @brief The hands arriving from a Quest app in bridge mode. @return the headset */
    frame::Headset& headset() { return headset_; }
    static constexpr const char* kApp = "Empyrean";   ///< the settings' and the jam's name

private:
    additiv::Engine engine_;                               ///< the instrument
    std::vector<juce::RangedAudioParameter*> params_;      ///< the host's parameters, index = ParamId
    std::vector<float> pushed_;                            ///< the value last handed to the engine, per parameter
    juce::AudioBuffer<float> spare_;                       ///< a right channel when the host gives only one
    std::array<std::atomic<int>, 128> ccMap_;              ///< per controller the parameter it moves, -1 none
    std::atomic<int> learning_{ -1 };                      ///< the parameter waiting for a controller, -1 none
    /** @brief Applies a program change that came from MIDI (message thread). */
    void handleAsyncUpdate() override;
    /** @brief The jam's role from the settings, and the headset's hands, thirty times a second (message thread). */
    void timerCallback() override;
    /** @brief The hands' moves in the family's grammar, on the macros every preset shares. */
    void pollHeadset();

    frame::JamBus jam_{ kApp };                            ///< the family jam's bus
    frame::Headset headset_;                               ///< the hands from the Quest app (bridge mode)
    int jamPostedRoot_ = -1;                               ///< leading: the root posted last
    int jamPostedScale_ = -1;                              ///< leading: the scale posted last

    std::atomic<int> current_{ -1 };                       ///< the preset last applied, -1 none
    mutable std::mutex errorLock_;                         ///< guards engineError_ (set by prepareToPlay, read by the editor)
    std::string engineError_;                              ///< why the engine did not open, empty when it runs
    const bool muted_ = juce::SystemStats::getEnvironmentVariable("ADDITIV_MUTE", {}).isNotEmpty();   ///< ADDITIV_MUTE: silent output
    std::atomic<int> pendingProgram_{ -1 };                ///< a MIDI program change waiting for the message thread
    int bankGroup_ = 0;                                    ///< the group bank select (CC 0) chose
    bool sustain_ = false;                                 ///< the hold pedal (CC 64) is down
    std::bitset<128> held_;                                ///< notes let go while the pedal was down, released with it

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(AdditivProcessor)
};
