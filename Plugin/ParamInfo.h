/**
 * @file ParamInfo.h
 * @brief One sentence per parameter, for the info line (InfoLine.h): what it does, in the manual's words.
 * @author Rene Weller
 */
#pragma once

#include <juce_core/juce_core.h>

/**
 * @brief What a parameter does, in a sentence.
 * @param[in] key  its key (ParamDesc::key)
 * @return the sentence, "" for a key it does not know
 */
juce::String paramInfo( const char *key );
