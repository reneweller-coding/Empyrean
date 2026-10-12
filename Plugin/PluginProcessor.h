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

#include "additiv/Engine.h"

#include <juce_audio_utils/juce_audio_utils.h>

#include <array>
#include <atomic>
#include <bitset>
#include <vector>

/** @brief The plugin and the standalone: the engine behind JUCE's processor interface. */
class AdditivProcessor : public juce::AudioProcessor {
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

    /** @brief One program, the current state. @return 1 */
    int getNumPrograms() override { return 1; }
    /** @brief The one program. @return 0 */
    int getCurrentProgram() override { return 0; }
    /** @brief Nothing to switch. @param index ignored */
    void setCurrentProgram(int index) override { (void)index; }
    /** @brief No program names. @param index ignored @return "" */
    const juce::String getProgramName(int index) override { (void)index; return {}; }
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

private:
    additiv::Engine engine_;                               ///< the instrument
    std::vector<juce::RangedAudioParameter*> params_;      ///< the host's parameters, index = ParamId
    std::vector<float> pushed_;                            ///< the value last handed to the engine, per parameter
    juce::AudioBuffer<float> spare_;                       ///< a right channel when the host gives only one
    std::array<std::atomic<int>, 128> ccMap_;              ///< per controller the parameter it moves, -1 none
    std::atomic<int> learning_{ -1 };                      ///< the parameter waiting for a controller, -1 none
    bool sustain_ = false;                                 ///< the hold pedal (CC 64) is down
    std::bitset<128> held_;                                ///< notes let go while the pedal was down, released with it

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(AdditivProcessor)
};
