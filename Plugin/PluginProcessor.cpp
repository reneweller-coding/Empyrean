/**
 * @file PluginProcessor.cpp
 * @brief The JUCE shell around the engine (PluginProcessor.h).
 */
#include "PluginProcessor.h"

#include "PluginEditor.h"

#include "additiv/Motion.h"
#include "additiv/Find.h"
#include "additiv/Focus.h"
#include "additiv/Presets.h"

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
    refreshUserPresets();   // the player's own presets, for the bank and the journeys
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
    startTimerHz(30);
}

void AdditivProcessor::timerCallback()
{
    jam_.setRole(frame::Settings::of(kApp).jamRole());   // the family jam (Jam.h): Off, Lead or Follow, from the settings
    pollHeadset();
    journeyTick();
    mapTick();
}

void AdditivProcessor::pollHeadset()
{
    frame::Settings& st = frame::Settings::of(kApp);
    headset_.listen(st.headset() == frame::Settings::HeadsetMode::Off ? 0 : st.headsetPort());
    const frame::HeadsetEvents e = headset_.poll();
    // The family's grammar on what every preset means the same by: the left hand the brightness
    // (Macro 2), the right the motion (Macro 1); pinches play and stop the conductor, step it,
    // bring Brain 2 in and out, and both hands move to the next preset.
    if (e.playStop) setValue(static_cast<int>(ParamId::BrainOn), value(static_cast<int>(ParamId::BrainOn)) >= 0.5f ? 0.0f : 1.0f);
    if (e.action) engine_.autoplayStep();
    if (e.hold) setValue(static_cast<int>(ParamId::Brain2On), value(static_cast<int>(ParamId::Brain2On)) >= 0.5f ? 0.0f : 1.0f);
    if (e.filterMoved) setValue(static_cast<int>(ParamId::Macro2), 0.5f * (e.filter + 1.0f));
    if (e.throwMoved) setValue(static_cast<int>(ParamId::Macro1), e.throwAmount);
    if (e.next) choosePreset((currentPreset() + 1) % additiv::numPresets());
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
    if (!engine_.prepare(c, &error)) { DBG("Empyrean: the engine did not open: " << error); }
    {
        const std::lock_guard<std::mutex> lock(errorLock_);
        engineError_ = engine_.prepared() ? std::string() : error;
    }
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

int AdditivProcessor::getNumPrograms() { return additiv::numPresets(); }

const juce::String AdditivProcessor::getProgramName(int index)
{
    const additiv::Preset& p = additiv::preset(index);
    return juce::String(additiv::kPresetGroupNames[p.group]) + ": " + p.name;
}

void AdditivProcessor::applyPreset(int index, double glide)
{
    if (index < 0 || index >= additiv::numPresets()) return;
    float values[additiv::kNumParams];
    std::string mod;
    additiv::presetValues(index, values, &mod);
    // The glide is asked for first: the engine starts it from what it played the block before,
    // whatever of the new preset has reached it by then (Engine::beginMorph).
    const double g = glide >= 0.0 ? glide : static_cast<double>(value(static_cast<int>(ParamId::PresetGlide)));
    // With the map on, a preset chosen is a place: the cursor goes there and the sound follows it.
    if (value(static_cast<int>(ParamId::MapActive)) >= 0.5f && !route_.running()) {
        float x = 0.0f, y = 0.0f;
        additiv::presetMapPosition(index, x, y);
        setValue(static_cast<int>(ParamId::MapX), x);
        setValue(static_cast<int>(ParamId::MapY), y);
    } else if (g > 0.0) {
        engine_.beginMorph(g);
    }
    for (int i = 0; i < additiv::kNumParams; ++i)
        if (!additiv::keptByPresets(static_cast<additiv::ParamId>(i))) setValue(i, values[i]);
    engine_.setModText(mod);
    if (g <= 0.0) engine_.requestFill();   // the arriving preset's chord within seconds (Noctuary's way); a glide keeps the chord
    current_.store(index, std::memory_order_relaxed);
    currentUser_.clear();
    setReference();
}

// ---- the player's own presets

/**
 * @brief A matrix's text as the engine would write it back: each row through its packing (what the
 *        audio side holds), so that "changed" compares like with like.
 * @param text  the text form
 * @return      the engine's spelling of it
 */
static std::string modTextAsEngine(const std::string& text)
{
    additiv::ModRow rows[additiv::kModRows];
    additiv::modRowsFromText(text, rows, additiv::kModRows);
    for (auto& r : rows) r = additiv::unpackModRow(additiv::packModRow(r));
    return additiv::modRowsToText(rows, additiv::kModRows);
}

juce::File AdditivProcessor::presetFolder()
{
    return juce::File::getSpecialLocation(juce::File::userDocumentsDirectory).getChildFile("Empyrean").getChildFile("Presets");
}

void AdditivProcessor::refreshUserPresets()
{
    userPresets_.clear();
    for (const juce::File& f : presetFolder().findChildFiles(juce::File::findFiles, false, "*.empyrean")) {
        const std::unique_ptr<juce::XmlElement> xml = juce::XmlDocument::parse(f);
        if (!xml || !xml->hasTagName("EmpyreanPreset")) continue;
        UserPreset u;
        u.name = f.getFileNameWithoutExtension();
        u.file = f;
        u.values.resize(static_cast<size_t>(additiv::kNumParams));
        for (const ParamDesc& d : additiv::paramTable()) u.values[static_cast<size_t>(d.id)] = d.def;
        for (const juce::String& kv : juce::StringArray::fromTokens(xml->getStringAttribute("values"), " ", ""))
            if (const ParamDesc* d = additiv::findParam(kv.upToFirstOccurrenceOf("=", false, false).toRawUTF8()))
                u.values[static_cast<size_t>(d->id)] = std::clamp(kv.fromFirstOccurrenceOf("=", false, false).getFloatValue(), d->min, d->max);
        u.mod = xml->getStringAttribute("mod").toStdString();
        userPresets_.push_back(std::move(u));
    }
    std::sort(userPresets_.begin(), userPresets_.end(), [](const UserPreset& a, const UserPreset& b) { return a.name.compareNatural(b.name) < 0; });
}

int AdditivProcessor::findUserPreset(const juce::String& name) const
{
    for (size_t i = 0; i < userPresets_.size(); ++i)
        if (userPresets_[i].name == name) return static_cast<int>(i);
    return -1;
}

bool AdditivProcessor::saveUserPreset(const juce::String& name, juce::String& why)
{
    const juce::String legal = juce::File::createLegalFileName(name.trim());
    if (legal.isEmpty()) { why = "a preset needs a name"; return false; }
    if (additiv::findPreset(legal.toRawUTF8()) >= 0) { why = "a factory preset has that name (journeys find presets by name)"; return false; }
    juce::String values;
    for (const ParamDesc& d : additiv::paramTable())
        if (!additiv::keptByPresets(d.id)) values << d.key << "=" << juce::String(value(static_cast<int>(d.id)), 6) << " ";
    juce::XmlElement xml("EmpyreanPreset");
    xml.setAttribute("version", ADDITIV_VERSION);
    xml.setAttribute("values", values.trim());
    xml.setAttribute("mod", juce::String(engine_.modText()));
    presetFolder().createDirectory();
    if (!xml.writeTo(presetFolder().getChildFile(legal + ".empyrean"))) { why = "it could not be written into " + presetFolder().getFullPathName(); return false; }
    refreshUserPresets();
    current_.store(-1, std::memory_order_relaxed);
    currentUser_ = legal;
    setReference();
    return true;
}

void AdditivProcessor::applyUserPreset(int i, double glide)
{
    if (i < 0 || i >= static_cast<int>(userPresets_.size())) return;
    const UserPreset& u = userPresets_[static_cast<size_t>(i)];
    const double g = glide >= 0.0 ? glide : static_cast<double>(value(static_cast<int>(ParamId::PresetGlide)));
    if (g > 0.0) engine_.beginMorph(g);
    for (int k = 0; k < additiv::kNumParams; ++k)
        if (!additiv::keptByPresets(static_cast<ParamId>(k))) setValue(k, u.values[static_cast<size_t>(k)]);
    engine_.setModText(u.mod);
    if (g <= 0.0) engine_.requestFill();
    current_.store(-1, std::memory_order_relaxed);
    currentUser_ = u.name;
    setReference();
}

void AdditivProcessor::chooseUserPreset(int i)
{
    if (i < 0 || i >= static_cast<int>(userPresets_.size())) return;
    browsing_ = false;
    noteRecent(userPresets_[static_cast<size_t>(i)].name);
    journeyPlayer_.stop();
    setValue(static_cast<int>(ParamId::MorphActive), 0.0f);
    leaveMap(false);
    applyUserPreset(i, -1.0);
}

juce::PropertiesFile& AdditivProcessor::userSettings()
{
    if (userSettings_ == nullptr) {
        juce::PropertiesFile::Options o;
        o.applicationName = "player";
        o.folderName = "Empyrean";
        o.filenameSuffix = ".settings";
        o.osxLibrarySubFolder = "Application Support";
        userSettings_ = std::make_unique<juce::PropertiesFile>(o);
    }
    return *userSettings_;
}

bool AdditivProcessor::favorite(const juce::String& name) const
{
    auto* self = const_cast<AdditivProcessor*>(this);   // the settings file is read on first use
    return juce::StringArray::fromLines(self->userSettings().getValue("favorites")).contains(name);
}

void AdditivProcessor::setFavorite(const juce::String& name, bool on)
{
    juce::StringArray f = juce::StringArray::fromLines(userSettings().getValue("favorites"));
    f.removeEmptyStrings();
    f.removeString(name);
    if (on) f.add(name);
    userSettings().setValue("favorites", f.joinIntoString("\n"));
    userSettings().saveIfNeeded();
}

juce::StringArray AdditivProcessor::recentPresets() const
{
    auto* self = const_cast<AdditivProcessor*>(this);
    juce::StringArray r = juce::StringArray::fromLines(self->userSettings().getValue("recent"));
    r.removeEmptyStrings();
    return r;
}

void AdditivProcessor::noteRecent(const juce::String& name)
{
    juce::StringArray r = recentPresets();
    r.removeString(name);
    r.insert(0, name);
    while (r.size() > 40) r.remove(r.size() - 1);
    userSettings().setValue("recent", r.joinIntoString("\n"));
    userSettings().saveIfNeeded();
}

void AdditivProcessor::previewValues(const std::vector<float>& v, const std::string& mod)
{
    if (!browsing_) {
        stepOutOfBlends();
        browseFrom_ = knobValues();
        browseFromMod_ = engine_.modText();
        browsing_ = true;
    }
    engine_.beginMorph(0.4);
    for (int k = 0; k < additiv::kNumParams; ++k)
        if (!additiv::keptByPresets(static_cast<ParamId>(k))) setValue(k, v[static_cast<size_t>(k)]);
    engine_.setModText(mod);
}

void AdditivProcessor::previewPreset(int index)
{
    if (index < 0 || index >= additiv::numPresets()) return;
    float values[additiv::kNumParams];
    std::string mod;
    additiv::presetValues(index, values, &mod);
    previewValues(std::vector<float>(values, values + additiv::kNumParams), mod);
}

void AdditivProcessor::previewUserPreset(int i)
{
    if (i >= 0 && i < static_cast<int>(userPresets_.size())) previewValues(userPresets_[static_cast<size_t>(i)].values, userPresets_[static_cast<size_t>(i)].mod);
}

void AdditivProcessor::endPreview()
{
    if (!browsing_) return;
    browsing_ = false;
    engine_.beginMorph(0.4);
    for (int k = 0; k < additiv::kNumParams; ++k)
        if (!additiv::keptByPresets(static_cast<ParamId>(k))) setValue(k, browseFrom_[static_cast<size_t>(k)]);
    engine_.setModText(browseFromMod_);
}

void AdditivProcessor::setReference()
{
    reference_ = knobValues();
    referenceMod_ = engine_.modText();
}

bool AdditivProcessor::modified() const
{
    if (reference_.size() != static_cast<size_t>(additiv::kNumParams)) return false;
    for (const ParamDesc& d : additiv::paramTable()) {
        if (additiv::keptByPresets(d.id)) continue;
        if (std::fabs(value(static_cast<int>(d.id)) - reference_[static_cast<size_t>(d.id)]) > 1.0e-5f * std::max(1.0f, d.max - d.min)) return true;
    }
    return engine_.modText() != referenceMod_;
}

void AdditivProcessor::revert()
{
    if (reference_.size() != static_cast<size_t>(additiv::kNumParams)) return;
    engine_.beginMorph(1.0);
    for (int k = 0; k < additiv::kNumParams; ++k)
        if (!additiv::keptByPresets(static_cast<ParamId>(k))) setValue(k, reference_[static_cast<size_t>(k)]);
    engine_.setModText(referenceMod_);
}

void AdditivProcessor::revertKnob(int id)
{
    if (id >= 0 && id < additiv::kNumParams && reference_.size() == static_cast<size_t>(additiv::kNumParams)) setValue(id, reference_[static_cast<size_t>(id)]);
}

void AdditivProcessor::choosePreset(int index)
{
    if (index < 0 || index >= additiv::numPresets()) return;
    browsing_ = false;   // chosen while listened to: it stays
    noteRecent(additiv::preset(index).name);
    journeyPlayer_.stop();
    setValue(static_cast<int>(ParamId::MorphActive), 0.0f);
    if (value(static_cast<int>(ParamId::MapActive)) >= 0.5f) {
        leaveMap(false);   // the route stops with it; the engine glides from the map's blend (Map Glide)
        float x = 0.0f, y = 0.0f;
        additiv::presetMapPosition(index, x, y);
        setValue(static_cast<int>(ParamId::MapX), x);
        setValue(static_cast<int>(ParamId::MapY), y);
    } else {
        route_.stop();
    }
    applyPreset(index);
}

void AdditivProcessor::setSlotFromSound(int slot)
{
    slot &= 1;
    slotValues_[slot].assign(static_cast<size_t>(additiv::kNumParams), 0.0f);
    for (int i = 0; i < additiv::kNumParams; ++i) slotValues_[slot][static_cast<size_t>(i)] = value(i);
    slotMods_[slot] = engine_.modText();
    engine_.setMorphSlot(slot, slotValues_[slot].data(), slotMods_[slot]);
    const int cp = currentPreset();
    slotNames_[slot] = cp >= 0 ? juce::String("the sound of ") + additiv::preset(cp).name : juce::String("the sound");
}

void AdditivProcessor::setSlotFromPreset(int slot, int index)
{
    if (index < 0 || index >= additiv::numPresets()) return;
    slot &= 1;
    slotValues_[slot].assign(static_cast<size_t>(additiv::kNumParams), 0.0f);
    additiv::presetValues(index, slotValues_[slot].data(), &slotMods_[slot]);
    engine_.setMorphSlot(slot, slotValues_[slot].data(), slotMods_[slot]);
    slotNames_[slot] = juce::String(additiv::kPresetGroupNames[additiv::preset(index).group]) + ": " + additiv::preset(index).name;
}

bool AdditivProcessor::loadLandscape(const juce::File& file, bool letItSound, juce::String* error)
{
    additiv::Landscape l;
    std::string why;
    const juce::String ext = file.getFileExtension().toLowerCase();
    const bool audio = ext == ".wav" || ext == ".flac";
    // A picture lasts a minute from left to right (Motion Rate speeds it); a recording keeps its own
    // time, its columns whitened so it carries its motion rather than its colour (the probe's settings).
    const bool read = audio ? additiv::landscapeFromAudio(file.getFullPathName().toStdString(), 40.0f, 0.7f, 0.1f, 0.15f, l, &why)
                            : additiv::landscapeFromImage(file.getFullPathName().toStdString(), 60.0f, 40.0f, l, &why);
    if (!read) {
        if (error != nullptr) *error = juce::String(why.empty() ? std::string("it could not be read") : why);
        return false;
    }
    engine_.setUserLandscape(l);
    userLand_ = std::move(l);
    userLandFile_ = file;
    ++userLandVersion_;
    setValue(static_cast<int>(ParamId::Landscape), static_cast<float>(additiv::kUserLandscape));
    // Heard: the Aurora motion reads the landscape, the living landscape's picture modes too.
    const bool read2 = additiv::motionUsesLandscape(static_cast<additiv::MotionKind>(juce::roundToInt(value(static_cast<int>(ParamId::Motion)))));
    const bool pictured = additiv::lifeIsPicture(static_cast<additiv::LifeMode>(juce::roundToInt(value(static_cast<int>(ParamId::LifeMode)))));
    if (letItSound && !read2 && !pictured) setValue(static_cast<int>(ParamId::Motion), static_cast<float>(additiv::MotionKind::Aurora));
    return true;
}

void AdditivProcessor::moveCursor(float x, float y)
{
    route_.stop();
    setValue(static_cast<int>(ParamId::MapActive), 1.0f);
    setValue(static_cast<int>(ParamId::MapX), juce::jlimit(0.0f, 1.0f, x));
    setValue(static_cast<int>(ParamId::MapY), juce::jlimit(0.0f, 1.0f, y));
}

int AdditivProcessor::nearestPreset(float x, float y) const
{
    const additiv::MapNeighbours b = additiv::mapNeighbours(x, y, 0.01f);
    return b.count > 0 ? b.index[0] : 0;
}

void AdditivProcessor::leaveMap(bool keepSound)
{
    route_.stop();
    if (value(static_cast<int>(ParamId::MapActive)) < 0.5f) { mapOnWas_ = false; return; }
    setValue(static_cast<int>(ParamId::MapActive), 0.0f);
    mapOnWas_ = false;
    if (keepSound)
        applyPreset(nearestPreset(value(static_cast<int>(ParamId::MapX)), value(static_cast<int>(ParamId::MapY))),
                    std::max(1.0, static_cast<double>(value(static_cast<int>(ParamId::MapGlide)))));
}

void AdditivProcessor::mapTick()
{
    const double now = juce::Time::getMillisecondCounterHiRes() * 0.001;
    const double dt = mapLastTick_ > 0.0 ? juce::jlimit(0.0, 1.0, now - mapLastTick_) : 0.0;
    mapLastTick_ = now;
    // The route walks the cursor (its times divided by Route Speed); the map plays while it does.
    if (route_.running()) {
        float x = 0.0f, y = 0.0f, r = 0.0f;
        route_.update(static_cast<float>(dt) * value(static_cast<int>(ParamId::RouteSpeed)), x, y, r);
        if (value(static_cast<int>(ParamId::MapActive)) < 0.5f) setValue(static_cast<int>(ParamId::MapActive), 1.0f);
        setValue(static_cast<int>(ParamId::MapX), x);
        setValue(static_cast<int>(ParamId::MapY), y);
        setValue(static_cast<int>(ParamId::MapRadius), r);
    }
    const bool on = value(static_cast<int>(ParamId::MapActive)) >= 0.5f;
    if (!on) {
        if (mapOnWas_) leaveMap(true);   // switched off by hand: the sound stays near where it was
        return;
    }
    mapOnWas_ = true;
    const float x = value(static_cast<int>(ParamId::MapX)), y = value(static_cast<int>(ParamId::MapY)), r = value(static_cast<int>(ParamId::MapRadius));
    if (x == mapX_ && y == mapY_ && r == mapR_ && engine_.mapTargetSet()) return;
    mapX_ = x;
    mapY_ = y;
    mapR_ = r;
    additiv::MorphState s;
    blender_.blend(x, y, r, s);
    engine_.setMapTarget(s);
}

/** @brief A hall's memory as text: its kHallBins floats, base64. @param m the memory @return the text */
static juce::String hallText(const std::vector<float>& m)
{
    const juce::MemoryBlock b(m.data(), m.size() * sizeof(float));
    return b.toBase64Encoding();
}

/** @brief A hall's memory from its text. @param s the text @param m receives the memory @return false when it is no memory */
static bool hallFromText(const juce::String& s, std::vector<float>& m)
{
    juce::MemoryBlock b;
    if (!b.fromBase64Encoding(s) || b.getSize() != sizeof(float) * static_cast<size_t>(additiv::kHallBins)) return false;
    m.resize(static_cast<size_t>(additiv::kHallBins));
    std::memcpy(m.data(), b.getData(), b.getSize());
    return true;
}

// ---- the Focus page

std::vector<float> AdditivProcessor::knobValues() const
{
    std::vector<float> v(static_cast<size_t>(additiv::kNumParams));
    for (int i = 0; i < additiv::kNumParams; ++i) v[static_cast<size_t>(i)] = value(i);
    return v;
}

void AdditivProcessor::applyValues(const std::vector<float>& v, double glide)
{
    if (v.size() != static_cast<size_t>(additiv::kNumParams)) return;
    if (glide > 0.0) engine_.beginMorph(glide);
    for (int i = 0; i < additiv::kNumParams; ++i)
        if (!additiv::keptByPresets(static_cast<ParamId>(i)) && v[static_cast<size_t>(i)] != value(i)) setValue(i, v[static_cast<size_t>(i)]);
}

void AdditivProcessor::stepOutOfBlends()
{
    if (value(static_cast<int>(ParamId::MapActive)) >= 0.5f) leaveMap(true);
    if (value(static_cast<int>(ParamId::MorphActive)) >= 0.5f) setValue(static_cast<int>(ParamId::MorphActive), 0.0f);
}

float AdditivProcessor::nudgeSound(int axis, float sign)
{
    stepOutOfBlends();
    const std::vector<float> now = knobValues();
    std::vector<float> to(now.size());
    const float got = additiv::nudge(now.data(), axis, sign, 0.25f, to.data());
    applyValues(to, 0.6);
    return got;
}

void AdditivProcessor::makeVariations(float far)
{
    stepOutOfBlends();
    varFar_ = std::clamp(far, 0.05f, 1.0f);
    parent_ = knobValues();
    variations_.assign(6, std::vector<float>(parent_.size()));
    for (size_t k = 0; k < variations_.size(); ++k) additiv::vary(parent_.data(), varFar_, varSeed_++, variations_[k].data());
}

void AdditivProcessor::previewVariation(int k)
{
    if (k >= 0 && k < variationCount()) {
        if (previewing_ < 0) previewFrom_ = knobValues();
        previewing_ = k;
        applyValues(variations_[static_cast<size_t>(k)], 0.4);
    } else if (previewing_ >= 0) {
        previewing_ = -1;
        applyValues(previewFrom_, 0.4);
    }
}

void AdditivProcessor::takeVariation(int k)
{
    if (k < 0 || k >= variationCount()) return;
    if (previewing_ >= 0) {   // the sound before the listening is where it is kept from
        previewing_ = -1;
        applyValues(previewFrom_, 0.0);
    }
    parents_.push_back(parent_);
    if (parents_.size() > 32) parents_.erase(parents_.begin());
    applyValues(variations_[static_cast<size_t>(k)], 0.5);
    makeVariations(varFar_);
}

bool AdditivProcessor::variationBack()
{
    if (parents_.empty()) return false;
    const std::vector<float> back = parents_.back();
    parents_.pop_back();
    applyValues(back, 0.5);
    makeVariations(varFar_);
    return true;
}

juce::String AdditivProcessor::variationSummary(int k) const
{
    if (k < 0 || k >= variationCount() || parent_.size() != static_cast<size_t>(additiv::kNumParams)) return {};
    std::vector<std::pair<float, int>> moved;
    for (int i = 0; i < additiv::kNumParams; ++i) {
        const additiv::ParamDesc& d = additiv::paramTable()[static_cast<size_t>(i)];
        if (d.id == ParamId::MasterGain) continue;
        const float dn = additiv::toNorm(d, variations_[static_cast<size_t>(k)][static_cast<size_t>(i)]) - additiv::toNorm(d, parent_[static_cast<size_t>(i)]);
        if (std::fabs(dn) > 0.01f) moved.emplace_back(dn, i);
    }
    std::sort(moved.begin(), moved.end(), [](const auto& a, const auto& b) { return std::fabs(a.first) > std::fabs(b.first); });
    juce::String s;
    for (size_t j = 0; j < moved.size() && j < 3; ++j)
        s << (j > 0 ? ", " : "") << additiv::paramTable()[static_cast<size_t>(moved[j].second)].name << (moved[j].first > 0.0f ? " up" : " down");
    if (moved.size() > 3) s << " ...";
    return s;
}

void AdditivProcessor::teachBegin(int m)
{
    if (m < 0 || m >= 4) return;
    stepOutOfBlends();
    teachBase_[static_cast<size_t>(m)] = knobValues();
    additiv::MacroLesson& l = lessons_[static_cast<size_t>(m)];
    std::array<float, additiv::kNumParams> zero{};
    if (l.count < 2) {   // a new lesson: its first place is the sound now
        l = additiv::MacroLesson{};
        l.set(value(static_cast<int>(ParamId::Macro1) + m), zero.data());
    }
    engine_.setMacroLesson(m, l);
}

bool AdditivProcessor::teachHere(int m)
{
    if (!teaching(m)) return false;
    const std::vector<float>& from = teachBase_[static_cast<size_t>(m)];
    additiv::MacroLesson& l = lessons_[static_cast<size_t>(m)];
    const float at = value(static_cast<int>(ParamId::Macro1) + m);
    // What the macro already carries here, and what the knobs moved since: the new place.
    std::array<float, additiv::kNumParams> carried{}, delta{};
    l.apply(at, carried.data(), nullptr);
    for (int i = 0; i < additiv::kNumParams; ++i) {
        const additiv::ParamDesc& d = additiv::paramTable()[static_cast<size_t>(i)];
        if (!additiv::teachable(d.id)) continue;
        const float dn = additiv::toNorm(d, value(i)) - additiv::toNorm(d, from[static_cast<size_t>(i)]) + carried[static_cast<size_t>(i)];
        delta[static_cast<size_t>(i)] = std::fabs(dn) > 0.002f ? dn : 0.0f;
    }
    l.set(at, delta.data());
    // The macro carries the change, the knobs go back where the teaching began -- in the same moment, no
    // glide: a glide of the knobs under a lesson already in force would count the change twice for its length.
    engine_.setMacroLesson(m, l);
    for (int i = 0; i < additiv::kNumParams; ++i)
        if (additiv::teachable(static_cast<ParamId>(i)) && value(i) != from[static_cast<size_t>(i)]) setValue(i, from[static_cast<size_t>(i)]);
    return true;
}

void AdditivProcessor::teachForget(int m)
{
    if (m < 0 || m >= 4) return;
    lessons_[static_cast<size_t>(m)] = additiv::MacroLesson{};
    teachBase_[static_cast<size_t>(m)].clear();
    engine_.setMacroLesson(m, lessons_[static_cast<size_t>(m)]);
}

juce::String AdditivProcessor::macroName(int m) const
{
    if (m < 0 || m >= 4) return {};
    std::vector<std::pair<float, int>> moved;   // per knob how far the macro moves it, the larger first
    const additiv::MacroLesson& l = lessons_[static_cast<size_t>(m)];
    if (l.count >= 2)
        for (int p = 0; p < additiv::kNumParams; ++p) {
            float most = 0.0f;
            for (int i = 0; i < l.count; ++i) most = std::max(most, std::fabs(l.delta[i][static_cast<size_t>(p)]));
            if (most > 0.0f) moved.emplace_back(most, p);
        }
    const auto source = static_cast<additiv::ModSource>(static_cast<int>(additiv::ModSource::Macro1) + m);
    for (int i = 0; i < additiv::kModRows; ++i) {
        const additiv::ModRow r = engine_.modRow(i);
        if (r.source == source && r.target >= 0 && r.target < additiv::kNumParams && r.depth != 0.0f) moved.emplace_back(std::fabs(r.depth), r.target);
    }
    if (moved.empty()) return "Macro " + juce::String(m + 1);
    std::sort(moved.begin(), moved.end(), [](const auto& a, const auto& b) { return a.first > b.first; });
    std::vector<int> knobs;
    for (const auto& x : moved)
        if (std::find(knobs.begin(), knobs.end(), x.second) == knobs.end()) knobs.push_back(x.second);
    // A name two knobs share ("Depth": the motion's and the space's) carries its section.
    const ParamDesc& first = additiv::paramTable()[static_cast<size_t>(knobs[0])];
    bool shared = false;
    for (const ParamDesc& d : additiv::paramTable()) shared = shared || (d.id != first.id && juce::String(d.name) == first.name);
    juce::String s = shared ? juce::String(first.section) + " " + first.name : juce::String(first.name);
    if (knobs.size() > 1) s << " & " << static_cast<int>(knobs.size() - 1) << " more";
    return s;
}

juce::String AdditivProcessor::teachStatus(int m) const
{
    if (m < 0 || m >= 4) return {};
    const additiv::MacroLesson& l = lessons_[static_cast<size_t>(m)];
    juce::String s;
    if (l.count >= 2) {
        s << l.count << " places (";
        for (int i = 0; i < l.count; ++i) s << (i > 0 ? ", " : "") << juce::String(l.pos[i], 2);
        s << "), " << l.knobs() << " knobs";
    } else {
        s << "nothing taught";
    }
    if (teaching(m)) s << " -- teaching: move the knobs, set the macro, press Here";
    return s;
}

bool AdditivProcessor::findLike(const juce::File& file, juce::String& why)
{
    additiv::Description d;
    std::string error;
    if (!additiv::describeFile(file.getFullPathName().toStdString(), d, nullptr, &error)) {
        why = juce::String(error);
        return false;
    }
    int idx[12];
    float dist[12];
    const int n = additiv::findLike(d, idx, 12, dist);
    found_.assign(idx, idx + n);
    foundDistance_.assign(dist, dist + n);
    return true;
}

juce::File AdditivProcessor::hallFolder()
{
    return juce::File::getSpecialLocation(juce::File::userDocumentsDirectory).getChildFile("Empyrean").getChildFile("Halls");
}

bool AdditivProcessor::saveHall(const juce::File& file)
{
    std::vector<float> m;
    double age = 0.0;
    engine_.hallMemory(m, age);
    juce::XmlElement xml("EmpyreanHall");
    xml.setAttribute("version", ADDITIV_VERSION);
    xml.setAttribute("age", age);
    xml.setAttribute("memory", hallText(m));
    if (!xml.writeTo(file)) return false;
    hallName_ = file.getFileNameWithoutExtension();
    return true;
}

bool AdditivProcessor::loadHall(const juce::File& file)
{
    const std::unique_ptr<juce::XmlElement> xml = juce::XmlDocument::parse(file);
    std::vector<float> m;
    if (!xml || !xml->hasTagName("EmpyreanHall") || !hallFromText(xml->getStringAttribute("memory"), m)) return false;
    engine_.setHallMemory(m, xml->getDoubleAttribute("age"));
    hallName_ = file.getFileNameWithoutExtension();
    return true;
}

void AdditivProcessor::newHall()
{
    engine_.setHallMemory({}, 0.0);
    hallName_.clear();
}

juce::File AdditivProcessor::routeFolder()
{
    return juce::File::getSpecialLocation(juce::File::userDocumentsDirectory).getChildFile("Empyrean").getChildFile("Routes");
}

std::vector<AdditivProcessor::RouteEntry> AdditivProcessor::routes() const
{
    std::vector<RouteEntry> out;
    for (int i = 0; i < additiv::numRoutePresets(); ++i) {
        RouteEntry e;
        e.name = additiv::routePreset(i).name;
        e.builtin = i;
        out.push_back(e);
    }
    juce::Array<juce::File> files = routeFolder().findChildFiles(juce::File::findFiles, false, "*.route");
    files.sort();
    for (const juce::File& f : files) {
        RouteEntry e;
        e.name = f.getFileNameWithoutExtension();
        e.file = f;
        out.push_back(e);
    }
    return out;
}

bool AdditivProcessor::loadRoute(const RouteEntry& entry)
{
    const juce::String text = entry.builtin >= 0 ? juce::String(additiv::routePreset(entry.builtin).points) : entry.file.loadFileAsString();
    additiv::Route r;
    if (!r.parse(text.toRawUTF8())) return false;
    route_ = r;
    routeName_ = entry.name;
    return true;
}

bool AdditivProcessor::startRoute()
{
    if (route_.count() == 0) return false;
    journeyPlayer_.stop();
    setValue(static_cast<int>(ParamId::MorphActive), 0.0f);
    setValue(static_cast<int>(ParamId::MapActive), 1.0f);
    setValue(static_cast<int>(ParamId::BrainOn), 1.0f);   // a route is listened to: the conductor plays (the map keeps it so)
    route_.start(value(static_cast<int>(ParamId::MapX)), value(static_cast<int>(ParamId::MapY)), value(static_cast<int>(ParamId::MapRadius)), routeLoop_);
    return true;
}

void AdditivProcessor::addWaypoint(float x, float y, int preset)
{
    additiv::Waypoint w;
    w.x = juce::jlimit(0.0f, 1.0f, x);
    w.y = juce::jlimit(0.0f, 1.0f, y);
    w.preset = preset;
    if (preset >= 0) additiv::presetMapPosition(preset, w.x, w.y);
    w.radius = value(static_cast<int>(ParamId::MapRadius));
    w.travel = route_.count() == 0 ? 20.0f : 120.0f;
    w.hold = 60.0f;
    route_.stop();
    route_.add(w);
}

bool AdditivProcessor::saveRoute(const juce::File& file)
{
    if (route_.count() == 0) return false;
    std::vector<char> buf(static_cast<size_t>(route_.count()) * 200 + 64);
    if (route_.write(buf.data(), buf.size()) <= 0) return false;
    // One waypoint a line, under a line that says what the fields are.
    const juce::String text = juce::String("# Empyrean route: <preset or x,y> | travel s | hold s | radius\n")
                            + juce::String(buf.data()).replace(";", "\n") + "\n";
    file.getParentDirectory().createDirectory();
    if (!file.replaceWithText(text)) return false;
    routeName_ = file.getFileNameWithoutExtension();
    return true;
}

juce::String AdditivProcessor::routeStatus() const
{
    if (!route_.running()) return {};
    const int i = route_.segment();
    const additiv::Waypoint& w = route_.point(i);
    juce::String s = routeName_.isNotEmpty() ? routeName_ + "   " : juce::String();
    s << (i + 1) << "/" << route_.count() << "   " << (w.preset >= 0 ? juce::String(additiv::preset(w.preset).name) : juce::String("a place")) << "   "
      << juce::roundToInt(100.0f * route_.progress()) << " %";
    return s;
}

juce::File AdditivProcessor::journeyFolder()
{
    return juce::File::getSpecialLocation(juce::File::userDocumentsDirectory).getChildFile("Empyrean").getChildFile("Journeys");
}

std::vector<AdditivProcessor::JourneyEntry> AdditivProcessor::journeys() const
{
    std::vector<JourneyEntry> out;
    for (int i = 0; i < additiv::numBuiltinJourneys(); ++i) {
        additiv::Journey j;
        if (!j.parse(additiv::builtinJourneyText(i))) continue;
        JourneyEntry e;
        e.name = j.name;
        e.builtin = i;
        out.push_back(e);
    }
    juce::Array<juce::File> files = journeyFolder().findChildFiles(juce::File::findFiles, false, "*.journey");
    files.sort();
    for (const juce::File& f : files) {
        JourneyEntry e;
        e.name = f.getFileNameWithoutExtension();
        e.file = f;
        out.push_back(e);
    }
    return out;
}

bool AdditivProcessor::startJourney(const JourneyEntry& entry)
{
    additiv::Journey j;
    const bool read = entry.builtin >= 0 ? j.parse(additiv::builtinJourneyText(entry.builtin)) : j.load(entry.file.getFullPathName().toRawUTF8());
    if (!read || j.steps.empty()) return false;
    bool known = false;
    for (const additiv::JourneyStep& st : j.steps)
        known = known || additiv::findPreset(st.preset.c_str()) >= 0 || findUserPreset(juce::String(st.preset)) >= 0;
    if (!known) return false;
    if (j.name.empty()) j.name = entry.name.toStdString();
    journey_ = j;
    // The A/B morph holds the sound between its slots, the map plays the blend at its cursor: a
    // journey that loads presets under either would seem to do nothing (Noctuary's lesson), so both let go.
    setValue(static_cast<int>(ParamId::MorphActive), 0.0f);
    leaveMap(false);
    // Seeded from the clock: the same journey runs differently every night.
    journeyPlayer_.start(journey_, static_cast<std::uint64_t>(juce::Time::currentTimeMillis()) | 1ull, 0);
    journeyLastTick_ = 0.0;
    journeyTick();   // the first step begins now
    return true;
}

void AdditivProcessor::journeyTick()
{
    const double now = juce::Time::getMillisecondCounterHiRes() * 0.001;
    if (journeyLastTick_ <= 0.0) journeyLastTick_ = now;
    const double dt = juce::jlimit(0.0, 5.0, now - journeyLastTick_);
    journeyLastTick_ = now;
    if (!journeyPlayer_.running()) return;
    additiv::JourneyStep st;
    double glide = 60.0;
    if (!journeyPlayer_.advance(dt, st, glide)) return;
    const int index = additiv::findPreset(st.preset.c_str());
    const int own = index < 0 ? findUserPreset(juce::String(st.preset)) : -1;   // one of the player's presets by that name
    if (index >= 0) applyPreset(index, std::max(glide, 0.5));   // a name the bank no longer has: nothing changes until the next step
    else if (own >= 0) applyUserPreset(own, std::max(glide, 0.5));
    // A journey is listened to, not played: the conductor plays on through a preset made for the
    // hands (it would stop for the whole of that step).
    if (index >= 0 || own >= 0) setValue(static_cast<int>(ParamId::BrainOn), 1.0f);
}

juce::String AdditivProcessor::journeyStatus() const
{
    if (!journeyPlayer_.running()) return {};
    const int n = static_cast<int>(journey_.steps.size());
    const int step = journeyPlayer_.step();
    juce::String s = juce::String(journey_.name) + "   " + juce::String(step + 1) + "/" + juce::String(n);
    if (step >= 0 && step < n) s << "   " << journey_.steps[static_cast<size_t>(step)].preset;
    const float p = engine_.morphProgress();
    if (p < 1.0f) s << "   gliding " << juce::roundToInt(100.0f * p) << " %";
    s << "   next in " << additiv::Journey::timeText(std::max(0.0, journeyPlayer_.remaining()));
    return s;
}

void AdditivProcessor::journeyAddCurrent()
{
    const int cp = currentPreset();
    if (cp < 0 && currentUser_.isEmpty()) return;
    additiv::JourneyStep st;
    st.preset = cp >= 0 ? std::string(additiv::preset(cp).name) : currentUser_.toStdString();
    st.dwellLo = 480.0; st.dwellHi = 900.0; st.fadeLo = 90.0; st.fadeHi = 180.0;
    draft_.steps.push_back(st);
}

bool AdditivProcessor::saveJourneyDraft(const juce::File& file)
{
    if (draft_.steps.empty()) return false;
    additiv::Journey j = draft_;
    j.name = file.getFileNameWithoutExtension().toStdString();
    file.getParentDirectory().createDirectory();
    if (!j.save(file.getFullPathName().toRawUTF8())) return false;
    draft_ = additiv::Journey();
    return true;
}

void AdditivProcessor::handleAsyncUpdate()
{
    const int index = pendingProgram_.exchange(-1, std::memory_order_relaxed);
    if (index >= 0) choosePreset(index);
}

bool AdditivProcessor::mpeController(int ch, int cc, int value)
{
    if (cc == 99 || cc == 98) { rpn_[ch] = 0x3FFF; return false; }   // an NRPN: what follows is not ours
    if (cc == 101) { rpn_[ch] = (rpn_[ch] & 0x7F) | (value << 7); return true; }
    if (cc == 100) { rpn_[ch] = (rpn_[ch] & 0x3F80) | value; return true; }
    if (cc != 6 || rpn_[ch] == 0x3FFF) return false;
    if (rpn_[ch] == 0) {   // the bend range
        if (ch == 0) masterBend_ = static_cast<float>(value);
        else setValue(static_cast<int>(ParamId::MpeBend), static_cast<float>(std::max(value, 1)));
    } else if (rpn_[ch] == 6 && ch == 0) {   // MPE's configuration message: the lower zone with `value` member channels
        setValue(static_cast<int>(ParamId::MpeOn), value > 0 ? 1.0f : 0.0f);
    }
    return true;
}

void AdditivProcessor::processBlock(juce::AudioBuffer<float>& buffer, juce::MidiBuffer& midi)
{
    juce::ScopedNoDenormals noDenormals;
    taps_.processNextMidiBuffer(midi, 0, buffer.getNumSamples(), true);   // the keys the panel plays
    // MPE (the lower zone): with it on, a note on channels 2 .. 16 has that channel's bend, pressure
    // and slide (CC 74) for its own; channel 1 is the master, its bend moves every played note.
    const bool mpe = value(static_cast<int>(ParamId::MpeOn)) >= 0.5f;
    const float memberBend = value(static_cast<int>(ParamId::MpeBend));
    for (const auto meta : midi) {
        const juce::MidiMessage m = meta.getMessage();
        const int pos = meta.samplePosition;
        const int ch = std::clamp(m.getChannel() - 1, 0, 15);
        const bool member = mpe && ch >= 1;
        if (m.isNoteOn()) {
            held_.reset(static_cast<size_t>(m.getNoteNumber()));
            engine_.noteOn(m.getNoteNumber(), m.getFloatVelocity(), pos, member ? ch : -1);
        } else if (m.isNoteOff()) {
            // The hold pedal keeps a note sounding until it comes up (the family's CC 64).
            if (sustain_) held_.set(static_cast<size_t>(m.getNoteNumber()));
            else engine_.noteOff(m.getNoteNumber(), pos, member ? ch : -1);
        } else if (m.isAllNotesOff() || m.isAllSoundOff()) {
            held_.reset();
            engine_.allNotesOff();
        } else if (m.isPitchWheel()) {
            const float x = static_cast<float>(m.getPitchWheelValue() - 8192) / 8192.0f;
            if (member) engine_.setChannelBend(ch, x * memberBend);
            else engine_.setChannelBend(-1, x * masterBend_);
        } else if (m.isController() && mpeController(ch, m.getControllerNumber(), m.getControllerValue())) {
            // an RPN: taken
        } else if (m.isController() && member && m.getControllerNumber() == 74) {
            engine_.setChannelSlide(ch, static_cast<float>(m.getControllerValue()) / 127.0f);   // a note's slide (never learned)
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
            if (cc == 0 && target < 0) {
                bankGroup_ = m.getControllerValue() % additiv::kNumPresetGroups;   // bank select: a group of the bank
            } else if (target >= 0) {
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
            } else if (cc == 1) {
                engine_.setModWheel(x);   // a source of the matrix (the family leaves the wheel free)
            }
        } else if (m.isProgramChange()) {
            // Bank select (CC 0) chooses the group, the program its preset: applied on the message thread.
            const int index = additiv::presetAt(bankGroup_, m.getProgramChangeNumber());
            if (index >= 0) { pendingProgram_.store(index, std::memory_order_relaxed); triggerAsyncUpdate(); }
        } else if (m.isChannelPressure() && member) {
            engine_.setChannelPressure(ch, static_cast<float>(m.getChannelPressureValue()) / 127.0f);   // a note's pressure
        } else if (m.isChannelPressure()) {
            engine_.setPressure(static_cast<float>(m.getChannelPressureValue()) / 127.0f);
        } else if (m.isAftertouch()) {
            engine_.setPressure(static_cast<float>(m.getAfterTouchValue()) / 127.0f);
        }
    }
    for (size_t i = 0; i < params_.size(); ++i) {
        const float v = params_[i]->convertFrom0to1(params_[i]->getValue());
        if (v != pushed_[i]) { engine_.setParam(static_cast<ParamId>(i), v); pushed_[i] = v; }
    }
    // The family jam: following, the leader's root and energy; leading, this conductor's root and scale.
    if (jam_.role() == frame::Settings::JamRole::Follow) {
        const frame::JamState js = jam_.stateAt(-1.0);
        engine_.setJamRoot(js.root);
        engine_.setJamEnergy(js.root >= 0 ? js.energy : 0.5f);
    } else {
        engine_.setJamRoot(-1);
        engine_.setJamEnergy(0.5f);
        if (jam_.role() == frame::Settings::JamRole::Lead) {
            const int root = ((engine_.brainRoot() % 12) + 12) % 12;
            const int scale = juce::roundToInt(value(static_cast<int>(ParamId::Scale)));
            if (root != jamPostedRoot_ || scale != jamPostedScale_) {
                jam_.postKey(root, additiv::kScaleNames[std::clamp(scale, 0, additiv::kNumScaleChoices - 1)]);
                jamPostedRoot_ = root;
                jamPostedScale_ = scale;
            }
        }
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
    if (muted_) buffer.clear();   // ADDITIV_MUTE: the engine runs and is seen, nothing is heard (the manual's pictures)
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
    xml.createNewChildElement("Mod")->setAttribute("rows", juce::String(engine_.modText()));
    xml.setAttribute("preset", current_.load(std::memory_order_relaxed));
    if (currentUser_.isNotEmpty()) xml.setAttribute("userPreset", currentUser_);
    if (userLandFile_ != juce::File()) xml.setAttribute("landscapeFile", userLandFile_.getFullPathName());
    if (route_.count() > 0) {   // the route being played or written (not whether it runs)
        std::vector<char> buf(static_cast<size_t>(route_.count()) * 200 + 64);
        if (route_.write(buf.data(), buf.size()) > 0) {
            auto* r = xml.createNewChildElement("Route");
            r->setAttribute("points", juce::String(buf.data()));
            r->setAttribute("name", routeName_);
            r->setAttribute("loop", routeLoop_);
        }
    }
    for (int k = 0; k < 2; ++k) {   // the morph's slots, as their values and their matrix
        if (slotValues_[k].size() != static_cast<size_t>(additiv::kNumParams)) continue;
        auto* s = xml.createNewChildElement("Slot");
        s->setAttribute("index", k);
        s->setAttribute("name", slotNames_[k]);
        s->setAttribute("rows", juce::String(slotMods_[k]));
        for (const ParamDesc& d : additiv::paramTable()) s->setAttribute(juce::String("p_") + d.key, static_cast<double>(slotValues_[k][static_cast<size_t>(d.id)]));
    }
    for (int m = 0; m < 4; ++m) {   // the macros taught by example: per place its value and the knobs it moves
        const additiv::MacroLesson& l = lessons_[static_cast<size_t>(m)];
        if (l.count < 2) continue;
        auto* e = xml.createNewChildElement("Lesson");
        e->setAttribute("macro", m);
        for (int i = 0; i < l.count; ++i) {
            juce::String knobs;
            for (int p = 0; p < additiv::kNumParams; ++p)
                if (l.delta[i][static_cast<size_t>(p)] != 0.0f)
                    knobs << additiv::paramTable()[static_cast<size_t>(p)].key << "=" << juce::String(l.delta[i][static_cast<size_t>(p)], 5) << ";";
            auto* place = e->createNewChildElement("Place");
            place->setAttribute("at", static_cast<double>(l.pos[i]));
            place->setAttribute("knobs", knobs);
        }
    }
    {   // the hall's memory, once it has heard anything (the instrument's, not a preset's)
        std::vector<float> m;
        double age = 0.0;
        if (engine_.hallMemory(m, age)) {
            auto* h = xml.createNewChildElement("Hall");
            h->setAttribute("age", age);
            h->setAttribute("name", hallName_);
            h->setAttribute("memory", hallText(m));
        }
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
    current_.store(xml->getIntAttribute("preset", -1), std::memory_order_relaxed);
    currentUser_ = xml->getStringAttribute("userPreset");
    {   // the player's own landscape, read again from its file (a moved file: Aurora plays instead)
        const juce::File f(xml->getStringAttribute("landscapeFile"));
        const float choice = value(static_cast<int>(ParamId::Landscape));
        if (f.existsAsFile() && f != userLandFile_) {
            juce::String why;
            loadLandscape(f, false, &why);
            setValue(static_cast<int>(ParamId::Landscape), choice);   // as the state chose, not forced to Your Own
        }
    }
    const juce::XmlElement* mod = xml->getChildByName("Mod");
    engine_.setModText(mod != nullptr ? mod->getStringAttribute("rows").toStdString() : std::string());   // no rows named: an empty matrix
    if (const juce::XmlElement* r = xml->getChildByName("Route")) {
        additiv::Route rt;
        if (rt.parse(r->getStringAttribute("points").toRawUTF8())) {
            route_ = rt;
            routeName_ = r->getStringAttribute("name");
            routeLoop_ = r->getBoolAttribute("loop", true);
        }
    }
    mapX_ = -1.0f;   // the cursor is blended anew
    for (auto* s : xml->getChildWithTagNameIterator("Slot")) {
        const int k = s->getIntAttribute("index", -1);
        if (k < 0 || k > 1) continue;
        slotValues_[k].assign(static_cast<size_t>(additiv::kNumParams), 0.0f);
        for (const ParamDesc& d : additiv::paramTable())
            slotValues_[k][static_cast<size_t>(d.id)] = std::clamp(static_cast<float>(s->getDoubleAttribute(juce::String("p_") + d.key, d.def)), d.min, d.max);
        slotMods_[k] = s->getStringAttribute("rows").toStdString();
        slotNames_[k] = s->getStringAttribute("name");
        engine_.setMorphSlot(k, slotValues_[k].data(), slotMods_[k]);
    }
    for (int m = 0; m < 4; ++m) {   // the taught macros as the state kept them; none named: none
        lessons_[static_cast<size_t>(m)] = additiv::MacroLesson{};
        teachBase_[static_cast<size_t>(m)].clear();
    }
    for (auto* e : xml->getChildWithTagNameIterator("Lesson")) {
        const int m = e->getIntAttribute("macro", -1);
        if (m < 0 || m >= 4) continue;
        additiv::MacroLesson& l = lessons_[static_cast<size_t>(m)];
        for (auto* place : e->getChildWithTagNameIterator("Place")) {
            std::array<float, additiv::kNumParams> delta{};
            for (const juce::String& kv : juce::StringArray::fromTokens(place->getStringAttribute("knobs"), ";", ""))
                if (const additiv::ParamDesc* d = additiv::findParam(kv.upToFirstOccurrenceOf("=", false, false).toRawUTF8()))
                    if (additiv::teachable(d->id))
                        delta[static_cast<size_t>(d->id)] = std::clamp(kv.fromFirstOccurrenceOf("=", false, false).getFloatValue(), -1.0f, 1.0f);
            l.set(std::clamp(static_cast<float>(place->getDoubleAttribute("at")), 0.0f, 1.0f), delta.data());
        }
    }
    for (int m = 0; m < 4; ++m) engine_.setMacroLesson(m, lessons_[static_cast<size_t>(m)]);
    {   // the hall's memory as the state kept it; a state without one: a new hall
        std::vector<float> m;
        const juce::XmlElement* h = xml->getChildByName("Hall");
        if (h != nullptr && hallFromText(h->getStringAttribute("memory"), m)) {
            engine_.setHallMemory(m, h->getDoubleAttribute("age"));
            hallName_ = h->getStringAttribute("name");
        } else {
            newHall();
        }
    }
    for (auto& c : ccMap_) c.store(-1, std::memory_order_relaxed);
    for (auto* e : xml->getChildWithTagNameIterator("Learn")) {
        const int cc = e->getIntAttribute("cc", -1);
        const ParamDesc* d = additiv::findParam(e->getStringAttribute("key").toRawUTF8());
        if (cc >= 0 && cc < 128 && d != nullptr) ccMap_[static_cast<size_t>(cc)].store(static_cast<int>(d->id), std::memory_order_relaxed);
    }
    // What "changed" is measured against: the preset the state's sound came from, when it is known.
    setReference();
    const int cp = current_.load(std::memory_order_relaxed);
    if (cp >= 0 && cp < additiv::numPresets()) {
        float values[additiv::kNumParams];
        std::string rows;
        additiv::presetValues(cp, values, &rows);
        reference_.assign(values, values + additiv::kNumParams);
        for (int k = 0; k < additiv::kNumParams; ++k)   // the listener's own are as the state has them
            if (additiv::keptByPresets(static_cast<ParamId>(k))) reference_[static_cast<size_t>(k)] = value(k);
        referenceMod_ = modTextAsEngine(rows);
    } else if (const int own = findUserPreset(currentUser_); own >= 0) {
        reference_ = userPresets_[static_cast<size_t>(own)].values;
        for (int k = 0; k < additiv::kNumParams; ++k)
            if (additiv::keptByPresets(static_cast<ParamId>(k))) reference_[static_cast<size_t>(k)] = value(k);
        referenceMod_ = modTextAsEngine(userPresets_[static_cast<size_t>(own)].mod);
    }
}

/** @brief JUCE's factory for the plugin and the standalone. @return a new processor */
juce::AudioProcessor* JUCE_CALLTYPE createPluginFilter() { return new AdditivProcessor(); }
