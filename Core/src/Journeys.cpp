/**
 * @file Journeys.cpp
 * @brief The journeys the instrument comes with (Journey.h); the table itself is generated (Journeys.inc).
 */
#include "additiv/Journey.h"

#include <algorithm>

namespace additiv {

namespace {
#include "Journeys.inc"
constexpr int kCount = static_cast<int>(sizeof(kJourneys) / sizeof(kJourneys[0]));   ///< journeys in the table
} // namespace

int numBuiltinJourneys() { return kCount; }

const char* builtinJourneyText(int i) { return kJourneys[std::clamp(i, 0, kCount - 1)]; }

} // namespace additiv
