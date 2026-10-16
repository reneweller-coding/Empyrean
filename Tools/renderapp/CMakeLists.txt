# Empyrean Journeys (16.10.2026): the family's render app (RenderKit.h, the same in every instrument) with Empyrean's
# source -- its journeys rendered into WAV, FLAC, Ogg Vorbis or MP3, faster than real time (the engine offline), each
# shown with its length.
set(_app EmpyreanJourneys)
juce_add_gui_app(${_app}
    COMPANY_NAME     "Rene Weller"
    BUNDLE_ID        "com.reneweller.empyrean.journeys"
    PRODUCT_NAME     "${_app}"
    VERSION          ${PROJECT_VERSION}
    ICON_BIG         "${CMAKE_SOURCE_DIR}/Deploy/empyrean.png")
# The icon drawn per size (Deploy/make_icon.py) over the one juceaide scales down (Plugin/CMakeLists.txt says why).
get_target_property(_app_code ${_app} JUCE_GENERATED_SOURCES_DIRECTORY)
configure_file("${CMAKE_SOURCE_DIR}/Deploy/empyrean.ico" "${_app_code}/icon.ico" COPYONLY)
target_sources(${_app} PRIVATE RenderKit.cpp RenderWindow.cpp Mp3Writer.cpp Source.cpp)
target_compile_definitions(${_app} PRIVATE
    JUCE_WEB_BROWSER=0
    JUCE_USE_CURL=0
    JUCE_REPORT_APP_USAGE=0
    JUCE_APPLICATION_NAME_STRING="$<TARGET_PROPERTY:${_app},JUCE_PRODUCT_NAME>"
    JUCE_APPLICATION_VERSION_STRING="$<TARGET_PROPERTY:${_app},JUCE_VERSION>"
    _CRT_SECURE_NO_WARNINGS)
target_link_libraries(${_app}
    PRIVATE
        AdditivCore
        juce::juce_gui_basics
        juce::juce_data_structures
        juce::juce_audio_formats
    PUBLIC
        juce::juce_recommended_config_flags
        juce::juce_recommended_warning_flags)
if(WIN32)
    target_link_libraries(${_app} PRIVATE mfplat mfreadwrite mfuuid ole32)   # MP3 through Media Foundation
endif()
# Into bin/<preset> after every build (cmake/Family.cmake).
family_bin(${_app})
# Every format written, read back and measured (a few seconds of the first journey each), and the queue.
if(ADDITIV_BUILD_TOOLS)
    add_test(NAME renderapp COMMAND ${_app} --selftest 6)
    set_tests_properties(renderapp PROPERTIES LABELS "audio" TIMEOUT 900)
endif()
