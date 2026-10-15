/**
 * @file Morph.cpp
 * @brief One sound becoming another (Morph.h).
 * @author Rene Weller
 */
#include "additiv/Morph.h"

#include "additiv/Cloud.h"
#include "additiv/Motion.h"

#include <algorithm>
#include <cmath>

namespace additiv
{

void MorphState::set( const float *v, const ModRow *r, int count )
{
    std::copy( v, v + NumParams, m_values );
    m_filterMix = 1.0f;
    for ( float &g : m_gate ) g = 1.0f;
    m_filterCount = 1;
    m_filters[0].m_model = static_cast<int>( std::lround( v[static_cast<size_t>( ParamIdE::FILTER_MODEL )] ) );
    m_filters[0].m_cutoff = v[static_cast<size_t>( ParamIdE::CUTOFF )];
    m_filters[0].m_resonance = v[static_cast<size_t>( ParamIdE::RESONANCE )];
    m_filters[0].m_weight = 1.0f;
    m_filterMix = m_filters[0].m_model == 0 ? 0.0f : 1.0f;
    m_setupCount = 1;
    m_setups[0] = modSetupFrom( v );
    m_rowCount = 0;
    for ( int i = 0; i < count && m_rowCount < MorphRows; ++i )
        if ( r[i].m_source != ModSourceE::OFF && r[i].m_target >= 0 && r[i].m_depth != 0.0f )
        {
            m_rows[m_rowCount] = r[i];
            m_rows[m_rowCount].m_setup = 0;
            ++m_rowCount;
        }
}

namespace
{

/**
 * @brief A parameter's value at a point between two, in its normalised (skewed) domain.
 * @param[in] d  the parameter
 * @param[in] a  at 0
 * @param[in] b  at 1
 * @param[in] w  0 .. 1
 * @return the value
 */
float lerpParam( const ParamDesc &d, float a, float b, float w )
{
    if ( a == b || w <= 0.0f ) return a;   // the ends exactly: the skew's round trip is not
    if ( w >= 1.0f ) return b;
    const float n = toNorm( d, a ) + ( toNorm( d, b ) - toNorm( d, a ) ) * w;
    const float v = fromNorm( d, n );
    return d.m_kind == ParamKindE::INT ? std::round( v ) : v;
}

/** @brief The side a choice is taken from while it changes, its amount (normalised) there and its gate. */
struct Side
{
    const MorphState *m_from;     ///< whose choice and companions; null: the choice is the same on both
    float             m_amount;   ///< the amount, 0 .. 1 of its range
    float             m_gate;     ///< the share the amount, modulation included, is heard at
};

/**
 * @brief Where a choice with an amount stands between two sounds (the file comment of Morph.h).
 * @param[in] ca      the choice at 0
 * @param[in] cb      the choice at 1
 * @param[in] na      the amount at 0, normalised (taken as 0 where the choice is off)
 * @param[in] nb      the amount at 1, normalised
 * @param[in] ga      the gate at 0
 * @param[in] gb      the gate at 1
 * @param[in] off     the choice that means off, -1 when none does
 * @param[in] same    the two count as the same (then the amount simply glides)
 * @param[in] middle  change at the middle even from or to off (the landscape it reads changes there)
 * @param[in] w       0 .. 1
 * @param[in] a       the sound at 0
 * @param[in] b       the sound at 1
 * @return the side, the amount and the gate
 */
Side side( int ca, int cb, float na, float nb, float ga, float gb, int off, bool same, bool middle, float w,
           const MorphState &a, const MorphState &b )
{
    if ( ca == off )
    {
        na = 0.0f;
        ga = 0.0f;
    }
    if ( cb == off )
    {
        nb = 0.0f;
        gb = 0.0f;
    }
    if ( same ) return { nullptr, na + ( nb - na ) * w, ga + ( gb - ga ) * w };
    if ( !middle && ca == off ) return { &b, w * nb, w * gb };
    if ( !middle && cb == off ) return { &a, ( 1.0f - w ) * na, ( 1.0f - w ) * ga };
    return w < 0.5f ? Side{ &a, na * ( 1.0f - 2.0f * w ), ga * ( 1.0f - 2.0f * w ) }
                    : Side{ &b, nb * ( 2.0f * w - 1.0f ), gb * ( 2.0f * w - 1.0f ) };
}

/**
 * @brief A parameter's value as an index.
 * @param[in] s   the sound
 * @param[in] id  the parameter
 * @return it rounded
 */
int choiceOf( const MorphState &s, ParamIdE id )
{
    return static_cast<int>( std::lround( s.m_values[static_cast<size_t>( id )] ) );
}

/**
 * @brief A parameter normalised.
 * @param[in] s   the sound
 * @param[in] id  the parameter
 * @return 0 .. 1
 */
float normOf( const MorphState &s, ParamIdE id )
{
    return toNorm( paramDesc( id ), s.m_values[static_cast<size_t>( id )] );
}

/** @brief One of the choices with an amount: what it is made of. */
struct Pair
{
    ParamIdE m_choice;           ///< the choice
    ParamIdE m_amount;           ///< its amount (the filter's is filterMix: ParamId::Count)
    int      m_gate;             ///< its gate in MorphState::gate, -1 for the filter (filterMix is its own gate)
    ParamIdE m_companions[2];    ///< what goes with the choice's side
    int      m_companionCount;   ///< how many
};

/** @brief The choices with an amount, in the order blendMorph and approachMorph take them. */
const Pair Pairs[] = {
    { ParamIdE::MOTION, ParamIdE::MOTION_DEPTH, 0, { ParamIdE::MOTION, ParamIdE::MOTION }, 0 },
    { ParamIdE::LIFE_MODE, ParamIdE::LIFE_DEPTH, 1, { ParamIdE::LIFE_MODE, ParamIdE::LIFE_MODE }, 0 },
    { ParamIdE::FIELD_MODE, ParamIdE::FIELD_STRENGTH, 2, { ParamIdE::FIELD_MODE, ParamIdE::FIELD_MODE }, 0 },
};

/**
 * @brief Whether a pair's kind reads the landscape (Aurora, the animated pictures).
 * @param[in] p     the pair
 * @param[in] kind  its choice
 * @return true then
 */
bool readsLandscape( const Pair &p, int kind )
{
    if ( p.m_choice == ParamIdE::MOTION ) return motionUsesLandscape( static_cast<MotionKindE>( kind ) );
    if ( p.m_choice == ParamIdE::LIFE_MODE ) return lifeIsPicture( static_cast<LifeModeE>( kind ) );
    return false;
}

/**
 * @brief Whether the landscape changes between two sounds for a pair that reads it on either side.
 * @param[in] p  the pair
 * @param[in] a  one
 * @param[in] b  the other
 * @return true then
 */
bool landChanges( const Pair &p, const MorphState &a, const MorphState &b )
{
    return choiceOf( a, ParamIdE::LANDSCAPE ) != choiceOf( b, ParamIdE::LANDSCAPE ) &&
           ( readsLandscape( p, choiceOf( a, p.m_choice ) ) || readsLandscape( p, choiceOf( b, p.m_choice ) ) );
}

/**
 * @brief Whether two sounds have the same kind for a pair (and the same landscape, if it reads one).
 * @param[in] p  the pair
 * @param[in] a  one
 * @param[in] b  the other
 * @return true then
 */
bool sameKind( const Pair &p, const MorphState &a, const MorphState &b )
{
    return choiceOf( a, p.m_choice ) == choiceOf( b, p.m_choice ) && !landChanges( p, a, b );
}

/**
 * @brief A pair's amount, normalised.
 * @param[in] p  the pair
 * @param[in] s  the sound
 * @return 0 .. 1
 */
float amountOf( const Pair &p, const MorphState &s )
{
    return p.m_gate < 0 ? s.m_filterMix : normOf( s, p.m_amount );
}

/**
 * @brief A pair's gate.
 * @param[in] p  the pair
 * @param[in] s  the sound
 * @return 0 .. 1
 */
float gateOf( const Pair &p, const MorphState &s )
{
    return p.m_gate < 0 ? s.m_filterMix : s.m_gate[p.m_gate];
}

/**
 * @brief Sets a pair's amount (normalised) and gate.
 * @param[in]     p       the pair
 * @param[in,out] s       the sound
 * @param[in]     amount  0 .. 1
 * @param[in]     gate    0 .. 1
 */
void setAmount( const Pair &p, MorphState &s, float amount, float gate )
{
    if ( p.m_gate < 0 )
    {
        s.m_filterMix = amount;
        return;
    }
    s.m_values[static_cast<size_t>( p.m_amount )] = fromNorm( paramDesc( p.m_amount ), amount );
    s.m_gate[p.m_gate] = gate;
}

/**
 * @brief Takes a pair's choice and companions from a sound.
 * @param[in]     p     the pair
 * @param[in]     from  where from
 * @param[in,out] out   where to
 */
void takeKind( const Pair &p, const MorphState &from, MorphState &out )
{
    out.m_values[static_cast<size_t>( p.m_choice )] = from.m_values[static_cast<size_t>( p.m_choice )];
    for ( int k = 0; k < p.m_companionCount; ++k )
        out.m_values[static_cast<size_t>( p.m_companions[k] )] =
            from.m_values[static_cast<size_t>( p.m_companions[k] )];
}

/**
 * @brief Whether a row reads a sound's own modulators (its source or its via).
 * @param[in] r  the row
 * @return true then
 */
bool readsOwn( const ModRow &r )
{
    return isOwnSource( r.m_source ) || isOwnSource( r.m_via );
}

/**
 * @brief A setup's place in a sound: the one that is equal, or a new one (the first when full).
 * @param[in,out] out  the sound
 * @param[in]     s    the setup
 * @return its index
 */
int addSetup( MorphState &out, const ModSetup &s )
{
    for ( int i = 0; i < out.m_setupCount; ++i )
        if ( out.m_setups[i] == s ) return i;
    if ( out.m_setupCount < MorphSetups )
    {
        out.m_setups[out.m_setupCount] = s;
        return out.m_setupCount++;
    }
    return 0;
}

/**
 * @brief Whether two rows move the same thing the same way (their depths add): on the same modulators, if they read a
 *        sound's own.
 * @param[in] x  one
 * @param[in] y  the other
 * @return true then
 */
bool sameRow( const ModRow &x, const ModRow &y )
{
    return x.m_source == y.m_source && x.m_target == y.m_target && x.m_via == y.m_via &&
           ( !readsOwn( x ) || x.m_setup == y.m_setup );
}

/**
 * @brief Adds a row of a sound with its depth weighted, its setup carried over: merged with one that
 *        moves the same, or appended, or (when full) put in place of the weakest if it is stronger.
 * @param[in,out] out   the sound being made
 * @param[in]     from  the sound the row is from
 * @param[in]     r     the row
 * @param[in]     k     its weight
 */
void addRow( MorphState &out, const MorphState &from, const ModRow &r, float k )
{
    const float depth = r.m_depth * k;
    if ( depth == 0.0f ) return;
    ModRow s = r;
    s.m_depth = depth;
    s.m_setup = readsOwn( r ) && r.m_setup >= 0 && r.m_setup < from.m_setupCount
                    ? addSetup( out, from.m_setups[r.m_setup] )
                    : 0;
    for ( int i = 0; i < out.m_rowCount; ++i )
        if ( sameRow( out.m_rows[i], s ) )
        {
            out.m_rows[i].m_depth += depth;
            return;
        }
    if ( out.m_rowCount < MorphRows )
    {
        out.m_rows[out.m_rowCount++] = s;
        return;
    }
    int weakest = 0;
    for ( int i = 1; i < out.m_rowCount; ++i )
        if ( std::fabs( out.m_rows[i].m_depth ) < std::fabs( out.m_rows[weakest].m_depth ) ) weakest = i;
    if ( std::fabs( depth ) > std::fabs( out.m_rows[weakest].m_depth ) ) out.m_rows[weakest] = s;
}

/**
 * @brief Drops the setups no row reads any more, the rows' indices following.
 * @param[in,out] s  the sound
 */
void compactSetups( MorphState &s )
{
    int  map[MorphSetups];
    bool used[MorphSetups] = {};
    for ( int i = 0; i < s.m_rowCount; ++i )
        if ( readsOwn( s.m_rows[i] ) && s.m_rows[i].m_setup < s.m_setupCount ) used[s.m_rows[i].m_setup] = true;
    int n = 0;
    for ( int k = 0; k < s.m_setupCount; ++k )
    {
        map[k] = n;
        if ( used[k] ) s.m_setups[n++] = s.m_setups[k];
    }
    for ( int i = 0; i < s.m_rowCount; ++i )
        s.m_rows[i].m_setup =
            readsOwn( s.m_rows[i] ) && s.m_rows[i].m_setup < s.m_setupCount ? map[s.m_rows[i].m_setup] : 0;
    s.m_setupCount = n;
}

/**
 * @brief Adds a sound's filter with its weight scaled: merged with the entry of its model (cutoff
 *        and resonance as the weighted mean, the cutoff in its knob's domain), or appended.
 * @param[in,out] out  the sound being made
 * @param[in]     f    the filter
 * @param[in]     k    its weight's scale
 */
void addFilter( MorphState &out, const MorphFilter &f, float k )
{
    const float w = f.m_weight * k;
    if ( w <= 0.0f ) return;
    for ( int i = 0; i < out.m_filterCount; ++i )
    {
        MorphFilter &e = out.m_filters[i];
        if ( e.m_model != f.m_model ) continue;
        const float      sum = e.m_weight + w;
        const ParamDesc &cd = paramDesc( ParamIdE::CUTOFF );
        e.m_cutoff = fromNorm( cd, ( e.m_weight * toNorm( cd, e.m_cutoff ) + w * toNorm( cd, f.m_cutoff ) ) / sum );
        e.m_resonance = ( e.m_weight * e.m_resonance + w * f.m_resonance ) / sum;
        e.m_weight = sum;
        return;
    }
    if ( out.m_filterCount < MorphFilters )
    {
        out.m_filters[out.m_filterCount] = f;
        out.m_filters[out.m_filterCount].m_weight = w;
        ++out.m_filterCount;
    }
}

/**
 * @brief The filter list made whole: the faint ones dropped, the weights brought to 1, the share
 *        that is not Off, and the knobs on the heaviest that is not Off (Off when there is none).
 * @param[in,out] s  the sound
 */
void finishFilters( MorphState &s )
{
    int   keep = 0;
    float sum = 0.0f;
    for ( int i = 0; i < s.m_filterCount; ++i )
        if ( s.m_filters[i].m_weight > 1.0e-5f )
        {
            s.m_filters[keep++] = s.m_filters[i];
            sum += s.m_filters[i].m_weight;
        }
    s.m_filterCount = keep;
    if ( keep == 0 )
    {
        s.m_filterCount = 1;
        s.m_filters[0] = MorphFilter{};
        s.m_filters[0].m_weight = 1.0f;
        sum = 1.0f;
    }
    int heaviest = -1;
    s.m_filterMix = 0.0f;
    for ( int i = 0; i < s.m_filterCount; ++i )
    {
        s.m_filters[i].m_weight /= sum;
        if ( s.m_filters[i].m_model == 0 ) continue;
        s.m_filterMix += s.m_filters[i].m_weight;
        if ( heaviest < 0 || s.m_filters[i].m_weight > s.m_filters[heaviest].m_weight ) heaviest = i;
    }
    s.m_values[static_cast<size_t>( ParamIdE::FILTER_MODEL )] =
        heaviest >= 0 ? static_cast<float>( s.m_filters[heaviest].m_model ) : 0.0f;
    if ( heaviest >= 0 )
    {
        s.m_values[static_cast<size_t>( ParamIdE::CUTOFF )] = s.m_filters[heaviest].m_cutoff;
        s.m_values[static_cast<size_t>( ParamIdE::RESONANCE )] = s.m_filters[heaviest].m_resonance;
    }
}

/**
 * @brief A number a share of the way, unrounded, arriving exactly when close.
 * @param[in] d  the parameter
 * @param[in] a  from
 * @param[in] b  to
 * @param[in] k  the share
 * @return the value
 */
float follow( const ParamDesc &d, float a, float b, float k )
{
    if ( a == b ) return a;
    const float na = toNorm( d, a ), nb = toNorm( d, b );
    if ( std::fabs( nb - na ) < 1.0e-5f ) return b;
    return fromNorm( d, na + ( nb - na ) * k );
}

}   // namespace

void approachMorph( const MorphState &cur, const MorphState &target, float coef, MorphState &out )
{
    coef = std::clamp( coef, 0.0f, 1.0f );
    for ( const ParamDesc &d : paramTable() )
    {
        const size_t i = static_cast<size_t>( d.m_id );
        out.m_values[i] = d.m_kind == ParamKindE::FLOAT || d.m_kind == ParamKindE::INT
                              ? follow( d, cur.m_values[i], target.m_values[i], coef )
                              : target.m_values[i];
    }
    for ( int g = 0; g < MorphGates; ++g ) out.m_gate[g] = cur.m_gate[g] + ( target.m_gate[g] - cur.m_gate[g] ) * coef;
    // The filters: both lists, weighted (the card transports one into the other).
    out.m_filterCount = 0;
    for ( int i = 0; i < cur.m_filterCount; ++i ) addFilter( out, cur.m_filters[i], 1.0f - coef );
    for ( int i = 0; i < target.m_filterCount; ++i ) addFilter( out, target.m_filters[i], coef );
    finishFilters( out );
    // A kind that differs: its amount and gate fade out (twice the pace), its companions stay; it
    // changes when nothing of it is heard. From off it changes at once, with nothing of it yet.
    // The landscape changes when nothing that reads it is heard: until then those that read it
    // stay at nothing.
    const float fade = std::max( 0.0f, 1.0f - 2.0f * coef );
    out.m_values[static_cast<size_t>( ParamIdE::LANDSCAPE )] = cur.m_values[static_cast<size_t>( ParamIdE::LANDSCAPE )];
    for ( const Pair &p : Pairs )
    {
        if ( sameKind( p, cur, target ) ) continue;
        if ( choiceOf( cur, p.m_choice ) == 0 )
        {
            takeKind( p, target, out );
            setAmount( p, out, 0.0f, 0.0f );
            continue;
        }
        const float gate = gateOf( p, cur ) * fade;
        if ( gate < 0.005f )
        {
            takeKind( p, target, out );
            setAmount( p, out, 0.0f, choiceOf( target, p.m_choice ) == 0 ? gateOf( p, target ) : 0.0f );
        }
        else
        {
            takeKind( p, cur, out );
            setAmount( p, out, amountOf( p, cur ) * fade, gate );
        }
    }
    bool heard = false;
    for ( const Pair &p : Pairs )
        heard = heard || ( readsLandscape( p, choiceOf( out, p.m_choice ) ) && gateOf( p, out ) >= 0.005f );
    if ( !heard )
        out.m_values[static_cast<size_t>( ParamIdE::LANDSCAPE )] =
            target.m_values[static_cast<size_t>( ParamIdE::LANDSCAPE )];
    out.m_rowCount = 0;
    out.m_setupCount = 0;
    for ( int i = 0; i < cur.m_rowCount; ++i ) addRow( out, cur, cur.m_rows[i], 1.0f - coef );
    for ( int i = 0; i < target.m_rowCount; ++i ) addRow( out, target, target.m_rows[i], coef );
    // Rows the target no longer has fade geometrically: drop them once they are nothing.
    int keep = 0;
    for ( int i = 0; i < out.m_rowCount; ++i )
        if ( std::fabs( out.m_rows[i].m_depth ) > 1.0e-4f ) out.m_rows[keep++] = out.m_rows[i];
    out.m_rowCount = keep;
    compactSetups( out );
}

void blendMorph( const MorphState &a, const MorphState &b, float w, MorphState &out )
{
    w = std::clamp( w, 0.0f, 1.0f );
    // Every parameter: numbers glide, choices and switches change at the middle.
    for ( const ParamDesc &d : paramTable() )
    {
        const size_t i = static_cast<size_t>( d.m_id );
        out.m_values[i] = d.m_kind == ParamKindE::FLOAT || d.m_kind == ParamKindE::INT
                              ? lerpParam( d, a.m_values[i], b.m_values[i], w )
                              : ( w < 0.5f ? a.m_values[i] : b.m_values[i] );
    }
    // The choices with an amount (the motion, the living landscape, the field): the amount and the
    // gate by the rule, the kind from the side in use. Where the landscape changes (at the middle, a
    // plain choice), those that read it pass through nothing there, even from or to off.
    for ( const Pair &p : Pairs )
    {
        const Side s = side( choiceOf( a, p.m_choice ), choiceOf( b, p.m_choice ), amountOf( p, a ), amountOf( p, b ),
                             gateOf( p, a ), gateOf( p, b ), 0, sameKind( p, a, b ), landChanges( p, a, b ), w, a, b );
        setAmount( p, out, s.m_amount, s.m_gate );
        if ( s.m_from != nullptr ) takeKind( p, *s.m_from, out );
    }
    // The voice filters: both lists, weighted; the same model blends its numbers, different ones
    // stand side by side (the card transports one into the other).
    out.m_filterCount = 0;
    for ( int i = 0; i < a.m_filterCount; ++i ) addFilter( out, a.m_filters[i], 1.0f - w );
    for ( int i = 0; i < b.m_filterCount; ++i ) addFilter( out, b.m_filters[i], w );
    finishFilters( out );
    // The matrix: both, weighted, each row with the modulators it came with.
    out.m_rowCount = 0;
    out.m_setupCount = 0;
    for ( int i = 0; i < a.m_rowCount; ++i ) addRow( out, a, a.m_rows[i], 1.0f - w );
    for ( int i = 0; i < b.m_rowCount; ++i ) addRow( out, b, b.m_rows[i], w );
}

}   // namespace additiv
