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
# The program's icon is Deploy/empyrean.ico, drawn per size by Deploy/make_icon.py (the small ones with fewer, broader
# stalks), not JUCE's: juceaide scales ICON_BIG down, and only when CMake configures -- a new PNG alone left the old
# icon in the program (0.15.0). configure_file lays ours over it and configures again whenever the .ico changes.
get_target_property(_empyrean_code Empyrean JUCE_GENERATED_SOURCES_DIRECTORY)
configure_file("${CMAKE_SOURCE_DIR}/Deploy/empyrean.ico" "${_empyrean_code}/icon.ico" COPYONLY)
target_sources(Empyrean PRIVATE PluginProcessor.cpp PluginEditor.cpp HeaderParts.cpp ParamPage.cpp CloudView.cpp MatrixView.cpp SourcesView.cpp JourneyView.cpp MapView.cpp HallView.cpp KnobMod.cpp ModDock.cpp PageStrip.cpp SpaceView.cpp TuningView.cpp FocusView.cpp Palette.cpp Browser.cpp ParamInfo.cpp InfoLine.cpp Tour.cpp Theme.cpp Frame.cpp Jam.cpp)
# The picture behind the panel (frame::Backdrop) and the help's prose (frame::HelpView).
juce_add_binary_data(AdditivData
    HEADER_NAME AdditivData.h
    NAMESPACE AdditivData
    SOURCES Resources/backdrop.jpg Resources/chapters.txt)
target_compile_definitions(Empyrean PUBLIC
    JUCE_WEB_BROWSER=0
    JUCE_USE_CURL=0
    JUCE_VST3_CAN_REPLACE_VST2=0
    JUCE_REPORT_APP_USAGE=0
    _CRT_SECURE_NO_WARNINGS)
# ASIO in the standalone's audio dialog (Windows): the Steinberg ASIO SDK under its GPLv3 option (since 29.10.2025),
# which goes with this program's AGPLv3. The SDK Steinberg publishes, where it lies beside the repositories
# (_Sources/ASIO-SDK/ASIOSDK, from https://www.steinberg.net/asiosdk); else the same headers JUCE carries.
if(WIN32)
    set(ADDITIV_ASIO_SDK "${CMAKE_SOURCE_DIR}/../_Sources/ASIO-SDK/ASIOSDK/common" CACHE PATH "the ASIO SDK's common folder")
    target_compile_definitions(Empyrean PUBLIC JUCE_ASIO=1)
    if(EXISTS "${ADDITIV_ASIO_SDK}/iasiodrv.h")
        target_compile_definitions(Empyrean PUBLIC JUCE_ASIO_USE_EXTERNAL_SDK=1)
        target_include_directories(Empyrean PRIVATE "${ADDITIV_ASIO_SDK}")
        message(STATUS "Empyrean: ASIO from ${ADDITIV_ASIO_SDK}")
    else()
        message(STATUS "Empyrean: ASIO from JUCE's copy of the SDK's headers")
    endif()
endif()
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
