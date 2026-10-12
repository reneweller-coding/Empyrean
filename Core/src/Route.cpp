/**
 * @file Route.cpp
 * @brief The route over the preset map: the built-in route presets, the text form, and the walk.
 *
 * Noctuary's. Route.h says what a route is; this file holds the routes that ship (Routes.inc,
 * made by Tools/presets/make_map.py), the parser and writer of the text form, and Route::update(),
 * the walk itself. parse() resolves a waypoint given by preset name to that preset's place on the
 * map at parse time, which is why the built-in routes are written as names: when the map is laid
 * out again they follow the presets. A waypoint given as `x,y` is a fixed place on the map.
 *
 * The walk runs at control rate: update() is given the time that passed
 * (already multiplied by the route's speed) and moves the cursor -- position and blend radius --
 * along a smoothstep from where it started to the next waypoint over that waypoint's travel time,
 * then holds it there for the hold time, then goes on to the next point, looping or stopping at
 * the end. Nothing here allocates; parse() works in a fixed buffer of Route::kMaxPoints waypoints.
 */
#include "additiv/Route.h"
#include "additiv/PresetMap.h"
#include "additiv/Presets.h"

#include <algorithm>
#include <cstring>
#include <cstdlib>
#include <cstdio>

namespace additiv {

namespace {
#include "Routes.inc"
}

int numRoutePresets() { return static_cast<int>(sizeof(kRoutes) / sizeof(kRoutes[0])); }
const RoutePreset& routePreset(int index) { return kRoutes[std::clamp(index, 0, numRoutePresets() - 1)]; }

bool Route::parse(const char* text)
{
    Waypoint pts[kMaxPoints];
    int n = 0;
    const char* s = text;
    if (s == nullptr) return false;
    while (*s) {
        while (*s == ' ' || *s == '\n' || *s == '\r' || *s == '\t') ++s;
        if (!*s) break;
        const char* e = s; while (*e && *e != ';' && *e != '\n') ++e;
        // split by '|'
        char buf[192];
        const int len = static_cast<int>(e - s);
        if (len <= 0 || len >= static_cast<int>(sizeof(buf))) return false;
        std::memcpy(buf, s, static_cast<size_t>(len)); buf[len] = 0;
        char* fields[4] = { buf, nullptr, nullptr, nullptr };
        int nf = 1;
        for (char* p = buf; *p && nf < 4; ++p) if (*p == '|') { *p = 0; fields[nf++] = p + 1; }
        if (nf < 3) return false;
        // trim the position field
        char* pos = fields[0]; while (*pos == ' ') ++pos;
        for (char* q = pos + std::strlen(pos); q > pos && q[-1] == ' '; --q) q[-1] = 0;
        Waypoint w;
        char* endp = nullptr;
        const float fx = std::strtof(pos, &endp);
        if (endp != pos && *endp == ',') {   // x,y
            w.x = std::clamp(fx, 0.0f, 1.0f);
            w.y = std::clamp(std::strtof(endp + 1, nullptr), 0.0f, 1.0f);
        } else {                              // preset name
            const int found = findPreset(pos);
            if (found < 0) return false;
            presetMapPosition(found, w.x, w.y);
            w.preset = found;
        }
        w.travel = std::max(0.0f, std::strtof(fields[1], nullptr));
        w.hold   = std::max(0.0f, std::strtof(fields[2], nullptr));
        if (nf >= 4) w.radius = std::clamp(std::strtof(fields[3], nullptr), 0.01f, 0.3f);
        if (n >= kMaxPoints) return false;
        pts[n++] = w;
        s = (*e == ';' || *e == '\n') ? e + 1 : e;
    }
    if (n == 0) return false;
    for (int i = 0; i < n; ++i) points_[i] = pts[i];
    count_ = n;
    stop();
    return true;
}

int Route::write(char* buf, size_t cap) const
{
    size_t used = 0;
    for (int i = 0; i < count_; ++i) {
        const Waypoint& w = points_[i];
        char item[160];
        int len;
        if (w.preset >= 0 && w.preset < numPresets())
            len = std::snprintf(item, sizeof(item), "%s|%g|%g|%g", preset(w.preset).name, w.travel, w.hold, w.radius);
        else
            len = std::snprintf(item, sizeof(item), "%.3f,%.3f|%g|%g|%g", w.x, w.y, w.travel, w.hold, w.radius);
        if (len <= 0 || len >= static_cast<int>(sizeof(item))) return 0;   // a name too long for the item is a name the text cannot hold
        const size_t need = static_cast<size_t>(len) + (i > 0 ? 1 : 0);
        if (used + need + 1 > cap) return 0;
        if (i > 0) buf[used++] = ';';
        std::memcpy(buf + used, item, static_cast<size_t>(len)); used += static_cast<size_t>(len);
    }
    buf[used] = 0;
    return static_cast<int>(used);
}

void Route::start(float fromX, float fromY, float fromRadius, bool loop)
{
    if (count_ == 0) return;
    fromX_ = curX_ = fromX; fromY_ = curY_ = fromY; fromR_ = curR_ = fromRadius;
    loop_ = loop; running_ = true; holding_ = false; seg_ = 0; prog_ = 0.0f; elapsed_ = 0.0f;
}

bool Route::update(float dt, float& x, float& y, float& radius)
{
    if (!running_ || count_ == 0) { x = curX_; y = curY_; radius = curR_; return false; }
    elapsed_ += std::max(dt, 0.0f);
    const Waypoint& w = points_[seg_];
    if (!holding_) {
        const float t = w.travel > 1e-3f ? std::clamp(elapsed_ / w.travel, 0.0f, 1.0f) : 1.0f;
        const float c = t * t * (3.0f - 2.0f * t);
        curX_ = fromX_ + (w.x - fromX_) * c;
        curY_ = fromY_ + (w.y - fromY_) * c;
        curR_ = fromR_ + (w.radius - fromR_) * c;
        prog_ = t;
        if (t >= 1.0f) { holding_ = true; elapsed_ = 0.0f; prog_ = 0.0f; }
    } else {
        curX_ = w.x; curY_ = w.y; curR_ = w.radius;
        prog_ = w.hold > 1e-3f ? std::clamp(elapsed_ / w.hold, 0.0f, 1.0f) : 1.0f;
        if (elapsed_ >= w.hold) {
            fromX_ = w.x; fromY_ = w.y; fromR_ = w.radius;
            holding_ = false; elapsed_ = 0.0f; prog_ = 0.0f;
            if (seg_ + 1 < count_) ++seg_;
            else if (loop_) seg_ = 0;
            else { running_ = false; }
        }
    }
    x = curX_; y = curY_; radius = curR_;
    return running_;
}

} // namespace additiv
