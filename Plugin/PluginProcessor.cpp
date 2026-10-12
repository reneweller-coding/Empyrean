/**
 * @file PluginProcessor.cpp
 * @brief The JUCE shell around the engine (PluginProcessor.h).
 */
#include "PluginProcessor.h"

#include "PluginEditor.h"

#include <algorithm>
#include <cmath>
#include <map>
#include <memory>
#include <string>

using additiv::ParamDesc;
using additiv::ParamId;
using additiv::ParamKind;

namespace {

/**
 * @brief The host's parameter for one row of the table.
 * @param d  the row
 * @return   a float, int, switch or choice parameter with the row's range, default and unit
 */
std::unique_ptr<juce::RangedAudioParameter> makeParameter(const ParamDesc& d)
{
    const juce::ParameterID id{ d.key, 1 };
    switch (d.kind) {
    case ParamKind::Int:
        return std::make_unique<juce::AudioParameterInt>(id, d.name, static_cast<int>(d.min), static_cast<int>(d.max), static_cast<int>(d.def),
                                                         juce::AudioParameterIntAttributes().withLabel(d.unit));
    case ParamKind::Bool:
        return std::make_unique<juce::AudioParameterBool>(id, d.name, d.def >= 0.5f);
    case ParamKind::Choice: {
        juce::StringArray names;
        for (int i = 0; i < d.numChoices; ++i) names.add(d.choices[i]);
        return std::make_unique<juce::AudioParameterChoice>(id, d.name, names, static_cast<int>(d.def));
    }
    case ParamKind::Float:
    default:
        // Shown with the precision its size deserves: 440 Hz, 25.0 ct, 1.20, 0.150.
        return std::make_unique<juce::AudioParameterFloat>(id, d.name, juce::NormalisableRange<float>(d.min, d.max, 0.0f, d.skew), d.def,
                                                           juce::AudioParameterFloatAttributes().withLabel(d.unit).withStringFromValueFunction(
                                                               [](float v, int) {
                                                                   const float a = std::fabs(v);
                                                                   return juce::String(v, a >= 100.0f ? 0 : a >= 10.0f ? 1 : a >= 1.0f ? 2 : 3);
                                                               }));
    }
}

} // namespace

AdditivProcessor::AdditivProcessor()
    : AudioProcessor(BusesProperties().withOutput("Output", juce::AudioChannelSet::stereo(), true))
{
    // The table's sections become the hosts' groups, in the order they first appear.
    params_.assign(static_cast<size_t>(additiv::kNumParams), nullptr);
    std::vector<std::unique_ptr<juce::AudioProcessorParameterGroup>> groups;
    std::map<std::string, juce::AudioProcessorParameterGroup*> bySection;
    for (const ParamDesc& d : additiv::paramTable()) {
        auto it = bySection.find(d.section);
        if (it == bySection.end()) {
            groups.push_back(std::make_unique<juce::AudioProcessorParameterGroup>(d.section, d.section, "|"));
            it = bySection.emplace(d.section, groups.back().get()).first;
        }
        auto p = makeParameter(d);
        params_[static_cast<size_t>(d.id)] = p.get();
        it->second->addChild(std::move(p));
    }
    for (auto& g : groups) addParameterGroup(std::move(g));
    pushed_.assign(params_.size(), -1.0e30f);
    for (auto& c : ccMap_) c.store(-1, std::memory_order_relaxed);
}

AdditivProcessor::~AdditivProcessor() { engine_.release(); }

void AdditivProcessor::prepareToPlay(double sampleRate, int samplesPerBlock)
{
    additiv::EngineConfig c;
    c.sampleRate = sampleRate;
    c.maxHostBlock = std::max(samplesPerBlock, 256);
    // The REALTIME GPU class is a property of the whole process (PLAN.md): the standalone takes it,
    // a plugin leaves its host alone until that is decided.
    c.realtimeGpuClass = wrapperType == wrapperType_Standalone;
    engine_.setRealtime(!isNonRealtime());
    std::string error;
    if (!engine_.prepare(c, &error)) { DBG("AdditivAmbientSynth: the engine did not open: " << error); }
    std::fill(pushed_.begin(), pushed_.end(), -1.0e30f);
    setLatencySamples(engine_.latencySamples());
}

void AdditivProcessor::releaseResources() { engine_.release(); }

bool AdditivProcessor::isBusesLayoutSupported(const BusesLayout& layouts) const
{
    return layouts.getMainOutputChannelSet() == juce::AudioChannelSet::stereo() && layouts.getMainInputChannelSet().isDisabled();
}

float AdditivProcessor::value(int id) const
{
    const auto* p = parameter(id);
    return p != nullptr ? p->convertFrom0to1(p->getValue()) : 0.0f;
}

void AdditivProcessor::setValue(int id, float v)
{
    auto* p = parameter(id);
    if (p == nullptr) return;
    const ParamDesc& d = additiv::paramDesc(static_cast<ParamId>(id));
    p->setValueNotifyingHost(p->convertTo0to1(std::clamp(v, d.min, d.max)));
}

int AdditivProcessor::controllerFor(int id) const
{
    for (int cc = 0; cc < 128; ++cc)
        if (ccMap_[static_cast<size_t>(cc)].load(std::memory_order_relaxed) == id) return cc;
    return -1;
}

void AdditivProcessor::forget(int id)
{
    for (auto& c : ccMap_)
        if (c.load(std::memory_order_relaxed) == id) c.store(-1, std::memory_order_relaxed);
}

void AdditivProcessor::resetToDefault(int id)
{
    if (id < 0 || id >= additiv::kNumParams) return;
    setValue(id, additiv::paramDesc(static_cast<ParamId>(id)).def);
}

void AdditivProcessor::processBlock(juce::AudioBuffer<float>& buffer, juce::MidiBuffer& midi)
{
    juce::ScopedNoDenormals noDenormals;
    for (const auto meta : midi) {
        const juce::MidiMessage m = meta.getMessage();
        const int pos = meta.samplePosition;
        if (m.isNoteOn()) {
            held_.reset(static_cast<size_t>(m.getNoteNumber()));
            engine_.noteOn(m.getNoteNumber(), m.getFloatVelocity(), pos);
        } else if (m.isNoteOff()) {
            // The hold pedal keeps a note sounding until it comes up (the family's CC 64).
            if (sustain_) held_.set(static_cast<size_t>(m.getNoteNumber()));
            else engine_.noteOff(m.getNoteNumber(), pos);
        } else if (m.isAllNotesOff() || m.isAllSoundOff()) {
            held_.reset();
            engine_.allNotesOff();
        } else if (m.isController()) {
            const int cc = m.getControllerNumber();
            const float x = static_cast<float>(m.getControllerValue()) / 127.0f;
            const int waiting = learning_.exchange(-1, std::memory_order_relaxed);
            if (waiting >= 0 && cc != 64) {
                forget(waiting);
                ccMap_[static_cast<size_t>(cc)].store(waiting, std::memory_order_relaxed);
            } else if (waiting >= 0) {
                learning_.store(waiting, std::memory_order_relaxed);   // the pedal is never learned
            }
            const int target = ccMap_[static_cast<size_t>(cc)].load(std::memory_order_relaxed);
            if (target >= 0) {
                if (auto* p = parameter(target)) p->setValueNotifyingHost(x);
            } else if (cc == 64) {
                const bool down = m.getControllerValue() >= 64;
                if (sustain_ && !down)
                    for (int n = 0; n < 128; ++n)
                        if (held_.test(static_cast<size_t>(n))) engine_.noteOff(n, pos);
                if (!down) held_.reset();
                sustain_ = down;
            } else if (cc == 74) {
                if (auto* p = parameter(static_cast<int>(ParamId::Cutoff))) p->setValueNotifyingHost(x);   // the family's filter controller
            }
        }
    }
    for (size_t i = 0; i < params_.size(); ++i) {
        const float v = params_[i]->convertFrom0to1(params_[i]->getValue());
        if (v != pushed_[i]) { engine_.setParam(static_cast<ParamId>(i), v); pushed_[i] = v; }
    }
    const int n = buffer.getNumSamples();
    float* L = buffer.getWritePointer(0);
    float* R = nullptr;
    if (buffer.getNumChannels() > 1) R = buffer.getWritePointer(1);
    else {
        spare_.setSize(1, n, false, false, true);
        R = spare_.getWritePointer(0);
    }
    engine_.process(L, R, n);
    for (int c = 2; c < buffer.getNumChannels(); ++c) buffer.clear(c, 0, n);
}

juce::AudioProcessorEditor* AdditivProcessor::createEditor() { return new AdditivEditor(*this); }

void AdditivProcessor::getStateInformation(juce::MemoryBlock& dest)
{
    juce::XmlElement xml("AdditivState");
    xml.setAttribute("version", ADDITIV_VERSION);
    for (const ParamDesc& d : additiv::paramTable()) {
        const auto* p = params_[static_cast<size_t>(d.id)];
        xml.setAttribute(d.key, static_cast<double>(p->convertFrom0to1(p->getValue())));
    }
    for (int cc = 0; cc < 128; ++cc) {
        const int id = ccMap_[static_cast<size_t>(cc)].load(std::memory_order_relaxed);
        if (id < 0) continue;
        auto* e = xml.createNewChildElement("Learn");
        e->setAttribute("cc", cc);
        e->setAttribute("key", additiv::paramDesc(static_cast<ParamId>(id)).key);
    }
    copyXmlToBinary(xml, dest);
}

void AdditivProcessor::setStateInformation(const void* data, int size)
{
    const std::unique_ptr<juce::XmlElement> xml = getXmlFromBinary(data, size);
    if (!xml || !xml->hasTagName("AdditivState")) return;
    // A parameter the state does not name goes to its default, never stays where it was: otherwise
    // a state would sound different depending on what was loaded before it (the family's lesson).
    for (const ParamDesc& d : additiv::paramTable()) {
        auto* p = params_[static_cast<size_t>(d.id)];
        const float v = xml->hasAttribute(d.key) ? static_cast<float>(xml->getDoubleAttribute(d.key)) : d.def;
        p->setValueNotifyingHost(p->convertTo0to1(std::clamp(v, d.min, d.max)));
    }
    for (auto& c : ccMap_) c.store(-1, std::memory_order_relaxed);
    for (auto* e : xml->getChildWithTagNameIterator("Learn")) {
        const int cc = e->getIntAttribute("cc", -1);
        const ParamDesc* d = additiv::findParam(e->getStringAttribute("key").toRawUTF8());
        if (cc >= 0 && cc < 128 && d != nullptr) ccMap_[static_cast<size_t>(cc)].store(static_cast<int>(d->id), std::memory_order_relaxed);
    }
}

/** @brief JUCE's factory for the plugin and the standalone. @return a new processor */
juce::AudioProcessor* JUCE_CALLTYPE createPluginFilter() { return new AdditivProcessor(); }
