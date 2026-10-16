# Empyrean Journeys (16.10.2026): a small program beside the instrument that renders its journeys into WAV, FLAC,
# Ogg Vorbis or MP3, faster than real time (the engine offline), each journey shown with its length.
juce_add_gui_app(EmpyreanJourneys
    COMPANY_NAME     "Rene Weller"
    BUNDLE_ID        "com.reneweller.empyrean.journeys"
    PRODUCT_NAME     "EmpyreanJourneys"
    VERSION          ${PROJECT_VERSION}
    ICON_BIG         "${CMAKE_SOURCE_DIR}/Deploy/empyrean.png")
get_target_property(_journeys_code EmpyreanJourneys JUCE_GENERATED_SOURCES_DIRECTORY)
configure_file("${CMAKE_SOURCE_DIR}/Deploy/empyrean.ico" "${_journeys_code}/icon.ico" COPYONLY)
target_sources(EmpyreanJourneys PRIVATE Main.cpp JourneysComponent.cpp JourneyRender.cpp Mp3Writer.cpp)
target_compile_definitions(EmpyreanJourneys PRIVATE
    JUCE_WEB_BROWSER=0
    JUCE_USE_CURL=0
    JUCE_REPORT_APP_USAGE=0
    JUCE_APPLICATION_NAME_STRING="$<TARGET_PROPERTY:EmpyreanJourneys,JUCE_PRODUCT_NAME>"
    JUCE_APPLICATION_VERSION_STRING="$<TARGET_PROPERTY:EmpyreanJourneys,JUCE_VERSION>"
    _CRT_SECURE_NO_WARNINGS)
target_link_libraries(EmpyreanJourneys
    PRIVATE
        AdditivCore
        juce::juce_gui_basics
        juce::juce_data_structures
        juce::juce_audio_formats
        mfplat mfreadwrite mfuuid ole32
    PUBLIC
        juce::juce_recommended_config_flags
        juce::juce_recommended_warning_flags)
# Into bin/<preset> after every build (cmake/Family.cmake).
family_bin(EmpyreanJourneys)
# Every format written, read back and measured (a few seconds of the first journey each).
if(ADDITIV_BUILD_TOOLS)
    add_test(NAME journeys COMMAND EmpyreanJourneys --selftest 6)
    set_tests_properties(journeys PROPERTIES LABELS "audio" TIMEOUT 600)
endif()
