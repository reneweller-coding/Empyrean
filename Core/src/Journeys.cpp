/**
 * @file Journeys.cpp
 * @brief The journeys the instrument comes with (Journey.h); the table itself is generated (Journeys.inc).
 */
#include "additiv/Journey.h"

#include <algorithm>

namespace additiv {

namespace {
#include "Journeys.inc"
constexpr int Count = static_cast<int>(sizeof(Journeys) / sizeof(Journeys[0]));   ///< journeys in the table
} // namespace

int numBuiltinJourneys() { return Count; }

const char* builtinJourneyText(int i) { return Journeys[std::clamp(i, 0, Count - 1)]; }

} // namespace additiv
