# The instrument as a VST3 and a standalone, on the family's panel (Frame.h, copied unchanged from Totality's
# Plugin/, which is canonical). After Totality's Plugin/CMakeLists.txt. Working name AdditivAmbientSynth until the
# instrument has its name.
juce_add_plugin(AdditivAmbientSynth
    COMPANY_NAME                "Rene Weller"
    BUNDLE_ID                   "com.reneweller.additivambientsynth"
    IS_SYNTH                    TRUE
    NEEDS_MIDI_INPUT            TRUE
    NEEDS_MIDI_OUTPUT           FALSE
    IS_MIDI_EFFECT              FALSE
    EDITOR_WANTS_KEYBOARD_FOCUS FALSE
    COPY_PLUGIN_AFTER_BUILD     FALSE
    PLUGIN_MANUFACTURER_CODE    Rnwl
    PLUGIN_CODE                 Adtv
    FORMATS                     VST3 Standalone   # Windows only for now (no macOS path, 11.10.2026)
    VST3_CATEGORIES             Instrument Synth
    VERSION                     ${PROJECT_VERSION}
    PRODUCT_NAME                "AdditivAmbientSynth")
target_sources(AdditivAmbientSynth PRIVATE PluginProcessor.cpp PluginEditor.cpp ParamPage.cpp CloudView.cpp Theme.cpp Frame.cpp)
# The help's prose (frame::HelpView).
juce_add_binary_data(AdditivData
    HEADER_NAME AdditivData.h
    NAMESPACE AdditivData
    SOURCES Resources/chapters.txt)
target_compile_definitions(AdditivAmbientSynth PUBLIC
    JUCE_WEB_BROWSER=0
    JUCE_USE_CURL=0
    JUCE_VST3_CAN_REPLACE_VST2=0
    JUCE_REPORT_APP_USAGE=0
    _CRT_SECURE_NO_WARNINGS)
target_link_libraries(AdditivAmbientSynth
    PRIVATE
        AdditivCore
        AdditivData
        juce::juce_audio_utils
        juce::juce_osc
    PUBLIC
        juce::juce_recommended_config_flags
        juce::juce_recommended_warning_flags)

# Into bin/<preset> after every build (cmake/Family.cmake): the standalone and the VST3 bundle.
family_bin(AdditivAmbientSynth_Standalone)
family_bin(AdditivAmbientSynth_VST3 VST3)
