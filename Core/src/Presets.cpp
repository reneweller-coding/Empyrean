/**
 * @file Presets.cpp
 * @brief The factory presets (Presets.h); the table itself is generated (Presets.inc).
 * @author Rene Weller
 */
#include "additiv/Presets.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>

namespace additiv
{

const char *const PresetGroupNames[NumPresetGroups] = {
    "Sleep Concert",
    "Choirs",
    "Glass",
    "Bells",
    "Strings",
    "Organs",
    "Air & Noise",
    "Deep Space",
    "Glissandi",
    "Pendulums",
    "Waves",
    "Shepard & Tides",
    "Shimmer",
    "Dark Drones",
    "Microtonal",
    "Keys",
    "Living Fields",
    "Turing Gardens",
    "Synchrony",
    "Flocks",
    "Ripples",
    "Melting Spectra",
    "Spectral Cascades",
    "Resonant Halls",
    "Ghosts & Memory",
    "Strange Attractors",
    "Reservoir Minds",
    "Self-Listening",
    "Gravity Wells",
    "Smear & Dispersion",
    "Lattices & Beats",
    "Night Gardens",
    "Lullabies",
    "Slow Breath",
    "Deep Sleep",
    "Dream Drift",
    "Moonlight",
    "Night Rain",
    "Ocean Sleep",
    "Cradle",
    "Starfield",
    "Hypnagogia",
    "Twilight Choirs",
    "Warm Drones",
    "Snowfall",
    "Forest Night",
    "Slow Tides",
    "Afterglow",
    "Kaleidoscopes",
    "Tunnels",
    "Spirals",
    "Ink & Flow",
    "Feedback Loops",
    "Raindrops",
    "Star Streams",
    "Moire Veils",
    "Stretched Worlds",
    "Shifted Metals",
    "Comb Phasers",
    "Barber Poles",
    "Painted Light",
    "Prisms",
    "Alien Harmony",
    "Dream Machines",
    "Swarms",
    "Splinters",
    "Chimeras",
    "Twin Swarms",
    "Currents",
    "Vortex Streets",
    "Ink Plumes",
    "Islands",
    "Labyrinths",
    "Archipelagos",
    "Gong Blooms",
    "Bronze Swells",
    "Shimmer Cascades",
    "Pressure Pads",
    "Expressive Voices",
    "Living Storms",
    "Growing Halls",
    "Eroded Halls",
    "Cathedral Years",
    "Worn Stone",
    "Slow Clocks",
    "Quickened Air",
    "Gravity Lenses",
    "Lingering Waves",
    "Orbits",
    "Cyclotrons",
    "Planetaria",
    "Magnetic Choirs",
    "Event Horizons",
    "Memory Palaces",
    "Celestial Mechanics",
    "Living Architecture",
    "Overblown Flutes",
    "Breath Registers",
    "Four Seasons",
    "Turning Years",
    "Avalanches",
    "Glacier Calving",
    "Subterranea",
    "Period Doubling",
    "Halo Horizons",
    "Aurora Curtains",
    "Rainfall Grains",
    "Breathing Grains",
    "Breather Chains",
    "Harmonic Ecology",
    "Night Ecology",
    "Wild Harmonics",
    "Phantom Choirs",
    "Inner Voices",
    "Talbot Carpets",
    "Bloch Tides",
    "Glass Forests",
    "Bamboo Groves",
    "Wind Tubes",
    "Temple Winds",
    "Storm Gardens",
    "Dreaming Rooms",
    "Night Memories",
    "Premonitions",
    "Breathing In",
    "Phantom Gardens",
    "Lucid Dreams",
    "Elsewhere",
    "Rain on Water",
    "Tin Roofs",
    "Forest Rain",
    "Stone Gardens",
    "Rustling Woods",
    "Pine Winds",
    "Seashores",
    "Pebble Beaches",
    "Storm Coasts",
    "Tam-Tam Halls",
    "Thunder Sheets",
    "Swarm Rings",
    "Firefly Swarms",
    "Slime Veins",
    "Mycelium Nights",
    "Weather Gardens",
    "Chapel Air",
    "Stone Halls",
    "Curtained Rooms",
    "Walking Rooms",
    "Standing Waves",
    "Halo Rings",
    "Counter Halos",
    "Edge States",
    "Still Halos",
    "Halo Chapels",
    "Quantum Harmony",
    "Interference Nights",
    "Tonnetz Walks",
    "Quantum Halos",
    "Resonant Spaces",
    "Topologies",
    "Aeolian Harps",
    "Harp Gardens",
    "Distant Storms",
    "Summer Storms",
    "Rolling Thunder",
    "Singing Lakes",
    "Frozen Nights",
    "Thin Ice",
    "Twin Streams",
    "Fusion Choirs",
    "Parting Clouds",
    "Endless Ritardando",
    "Endless Accelerando",
    "Night Arcs",
    "Inside the Cloud",
        "Night Weather",
    "Cricket Meadows",
    "Katydid Treetops",
    "Frog Ponds",
    "Summer Nights",
    "Warm Evenings",
    "Cool Nights",
    "Answering Choirs",
    "Tuned Choruses",
    "Time Mirrors",
    "Reverse Halls",
    "Mirror Chapels",
    "Converging Halos",
    "Storm Choruses",
    "Lake Nights",
    "Night Concerts",
    "Nocturnes",
};

namespace
{
#include "Presets.inc"
constexpr int Count = static_cast<int>( sizeof( Presets ) / sizeof( Presets[0] ) );   ///< presets in the table
}   // namespace

bool keptByPresets( ParamIdE id )
{
    return id == ParamIdE::BINAURAL || id == ParamIdE::MORPH_ACTIVE || id == ParamIdE::MORPH_POS ||
           id == ParamIdE::PRESET_GLIDE || id == ParamIdE::MAP_ACTIVE || id == ParamIdE::MAP_X ||
           id == ParamIdE::MAP_Y || id == ParamIdE::MAP_RADIUS || id == ParamIdE::MAP_GLIDE ||
           id == ParamIdE::ROUTE_SPEED || id == ParamIdE::MPE_ON ||
           id == ParamIdE::MPE_BEND;   // the player's controller
}

int numPresets()
{
    return Count;
}

const Preset &preset( int i )
{
    return Presets[std::clamp( i, 0, Count - 1 )];
}

int presetValues( int i, float *values, std::string *modText )
{
    for ( const ParamDesc &d : paramTable() ) values[static_cast<size_t>( d.m_id )] = d.m_def;
    const Preset &p = preset( i );
    int           known = 0;
    const char   *s = p.m_params;
    while ( *s != 0 )
    {
        while ( *s == ' ' ) ++s;
        const char *eq = std::strchr( s, '=' );
        if ( eq == nullptr ) break;
        const char *end = std::strchr( eq, ' ' );
        if ( end == nullptr ) end = eq + std::strlen( eq );
        char key[64] = {};
        std::memcpy( key, s, static_cast<size_t>( std::min<std::ptrdiff_t>( eq - s, 63 ) ) );
        if ( const ParamDesc *d = findParam( key ) )
        {
            values[static_cast<size_t>( d->m_id )] =
                std::clamp( static_cast<float>( std::atof( eq + 1 ) ), d->m_min, d->m_max );
            ++known;
        }
        s = end;
    }
    if ( modText != nullptr ) *modText = p.m_mod;
    return known;
}

int presetIndexInGroup( int i )
{
    const int g = preset( i ).m_group;
    int       index = 0;
    for ( int k = 0; k < std::clamp( i, 0, Count - 1 ); ++k )
        if ( Presets[k].m_group == g ) ++index;
    return index;
}

int findPreset( const char *name )
{
    for ( int k = 0; k < Count && name != nullptr; ++k )
        if ( std::strcmp( Presets[k].m_name, name ) == 0 ) return k;
    return -1;
}

int presetAt( int group, int index )
{
    int seen = 0;
    for ( int k = 0; k < Count; ++k )
        if ( Presets[k].m_group == group && seen++ == index ) return k;
    return -1;
}

}   // namespace additiv
