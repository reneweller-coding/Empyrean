# The instrument as a VST3 and a standalone, on the family's panel (Frame.h, copied unchanged from Totality's
# Plugin/, which is canonical). After Totality's Plugin/CMakeLists.txt. Empyrean since 12.10.2026 (the working name
# was AdditivAmbientSynth).
juce_add_plugin(Empyrean
    COMPANY_NAME                "Rene Weller"
    BUNDLE_ID                   "com.reneweller.empyrean"
    IS_SYNTH                    TRUE
    NEEDS_MIDI_INPUT            TRUE
    NEEDS_MIDI_OUTPUT           FALSE
    IS_MIDI_EFFECT              FALSE
    EDITOR_WANTS_KEYBOARD_FOCUS FALSE
    COPY_PLUGIN_AFTER_BUILD     FALSE
    PLUGIN_MANUFACTURER_CODE    Rnwl
    PLUGIN_CODE                 Empy
    FORMATS                     VST3 Standalone   # Windows only for now (no macOS path, 11.10.2026)
    VST3_CATEGORIES             Instrument Synth
    VERSION                     ${PROJECT_VERSION}
    PRODUCT_NAME                "Empyrean"
    ICON_BIG                    "${CMAKE_SOURCE_DIR}/Deploy/empyrean.png")
target_sources(Empyrean PRIVATE PluginProcessor.cpp PluginEditor.cpp ParamPage.cpp CloudView.cpp MatrixView.cpp SourcesView.cpp JourneyView.cpp MapView.cpp HallView.cpp KnobMod.cpp ModDock.cpp PageStrip.cpp SpaceView.cpp FocusView.cpp Palette.cpp Browser.cpp ParamInfo.cpp InfoLine.cpp Tour.cpp Theme.cpp Frame.cpp Jam.cpp)
# The help's prose (frame::HelpView).
juce_add_binary_data(AdditivData
    HEADER_NAME AdditivData.h
    NAMESPACE AdditivData
    SOURCES Resources/chapters.txt)
target_compile_definitions(Empyrean PUBLIC
    JUCE_WEB_BROWSER=0
    JUCE_USE_CURL=0
    JUCE_VST3_CAN_REPLACE_VST2=0
    JUCE_REPORT_APP_USAGE=0
    _CRT_SECURE_NO_WARNINGS)
target_link_libraries(Empyrean
    PRIVATE
        AdditivCore
        AdditivData
        juce::juce_audio_utils
        juce::juce_osc
    PUBLIC
        juce::juce_recommended_config_flags
        juce::juce_recommended_warning_flags)

# Into bin/<preset> after every build (cmake/Family.cmake): the standalone and the VST3 bundle.
family_bin(Empyrean_Standalone)
family_bin(Empyrean_VST3 VST3)
