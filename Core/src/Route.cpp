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
 * the end. Nothing here allocates; parse() works in a fixed buffer of Route::MaxPoints waypoints.
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

int numRoutePresets() { return static_cast<int>(sizeof(Routes) / sizeof(Routes[0])); }
const RoutePreset& routePreset(int index) { return Routes[std::clamp(index, 0, numRoutePresets() - 1)]; }

bool Route::parse(const char* text)
{
    Waypoint pts[MaxPoints];
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
            w.m_x = std::clamp(fx, 0.0f, 1.0f);
            w.m_y = std::clamp(std::strtof(endp + 1, nullptr), 0.0f, 1.0f);
        } else {                              // preset name
            const int found = findPreset(pos);
            if (found < 0) return false;
            presetMapPosition(found, w.m_x, w.m_y);
            w.m_preset = found;
        }
        w.m_travel = std::max(0.0f, std::strtof(fields[1], nullptr));
        w.m_hold   = std::max(0.0f, std::strtof(fields[2], nullptr));
        if (nf >= 4) w.m_radius = std::clamp(std::strtof(fields[3], nullptr), 0.1f, 4.0f);
        if (n >= MaxPoints) return false;
        pts[n++] = w;
        s = (*e == ';' || *e == '\n') ? e + 1 : e;
    }
    if (n == 0) return false;
    for (int i = 0; i < n; ++i) m_points[i] = pts[i];
    m_count = n;
    stop();
    return true;
}

int Route::write(char* buf, size_t cap) const
{
    size_t used = 0;
    for (int i = 0; i < m_count; ++i) {
        const Waypoint& w = m_points[i];
        char item[160];
        int len;
        if (w.m_preset >= 0 && w.m_preset < numPresets())
            len = std::snprintf(item, sizeof(item), "%s|%g|%g|%g", preset(w.m_preset).m_name, w.m_travel, w.m_hold, w.m_radius);
        else
            len = std::snprintf(item, sizeof(item), "%.3f,%.3f|%g|%g|%g", w.m_x, w.m_y, w.m_travel, w.m_hold, w.m_radius);
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
    if (m_count == 0) return;
    m_fromX = m_curX = fromX; m_fromY = m_curY = fromY; m_fromR = m_curR = fromRadius;
    m_loop = loop; m_running = true; m_holding = false; m_seg = 0; m_prog = 0.0f; m_elapsed = 0.0f;
}

bool Route::update(float dt, float& x, float& y, float& radius)
{
    if (!m_running || m_count == 0) { x = m_curX; y = m_curY; radius = m_curR; return false; }
    m_elapsed += std::max(dt, 0.0f);
    const Waypoint& w = m_points[m_seg];
    if (!m_holding) {
        const float t = w.m_travel > 1e-3f ? std::clamp(m_elapsed / w.m_travel, 0.0f, 1.0f) : 1.0f;
        const float c = t * t * (3.0f - 2.0f * t);
        m_curX = m_fromX + (w.m_x - m_fromX) * c;
        m_curY = m_fromY + (w.m_y - m_fromY) * c;
        m_curR = m_fromR + (w.m_radius - m_fromR) * c;
        m_prog = t;
        if (t >= 1.0f) { m_holding = true; m_elapsed = 0.0f; m_prog = 0.0f; }
    } else {
        m_curX = w.m_x; m_curY = w.m_y; m_curR = w.m_radius;
        m_prog = w.m_hold > 1e-3f ? std::clamp(m_elapsed / w.m_hold, 0.0f, 1.0f) : 1.0f;
        if (m_elapsed >= w.m_hold) {
            m_fromX = w.m_x; m_fromY = w.m_y; m_fromR = w.m_radius;
            m_holding = false; m_elapsed = 0.0f; m_prog = 0.0f;
            if (m_seg + 1 < m_count) ++m_seg;
            else if (m_loop) m_seg = 0;
            else { m_running = false; }
        }
    }
    x = m_curX; y = m_curY; radius = m_curR;
    return m_running;
}

} // namespace additiv
