/**
 * @file Rate.h
 * @brief The sample rate, written once (11.11.2026).
 *
 * The instrument runs at the host's rate: prepare() hands it to every stage. This is the rate they stand at until
 * then, and the rate of everything that has no host -- the render tools, the probes, the tests, the bank's
 * measurements. Nothing else in the repository writes the number.
 * @author Rene Weller
 */
#pragma once

namespace additiv
{

/** @brief The sample rate before prepare() names the host's, and the tools' and tests' own, Hz. */
constexpr double DefaultSampleRate = 48000.0;

}   // namespace additiv
