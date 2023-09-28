/*
Copyright 2007-2023. Algoryx Simulation AB.

All AGX source code, intellectual property, documentation, sample code,
tutorials, scene files and technical white papers, are copyrighted, proprietary
and confidential material of Algoryx Simulation AB. You may not download, read,
store, distribute, publish, copy or otherwise disseminate, use or expose this
material unless having a written signed agreement with Algoryx Simulation AB, or having been
advised so by Algoryx Simulation AB for a time limited evaluation, or having purchased a
valid commercial license from Algoryx Simulation AB.

Algoryx Simulation AB disclaims all responsibilities for loss or damage caused
from using this software, unless otherwise stated in written agreements with
Algoryx Simulation AB.
*/

#ifndef AGX_VEC6_H
#define AGX_VEC6_H

#include <iostream>
#include <cstring>

#include <agx/agx.h>
#include <agx/Math.h>
#include <agx/Vec3.h>

namespace agx
{

  class Vec6
  {
    public:
      inline Vec6()
      {
        memset( m_data, 0, 6 * sizeof( Real ) );
      }

      Vec6( const Vec6& v ) = default;

      inline Vec6( const Real& x1, const Real& x2, const Real& x3,
            const Real& x4, const Real& x5, const Real& x6 )
      {
        m_data[ 0 ] = x1, m_data[ 1 ] = x2, m_data[ 2 ] = x3;
        m_data[ 3 ] = x4, m_data[ 4 ] = x5, m_data[ 5 ] = x6;
      }

      inline Vec6( const Vec3& v1, const Vec3& v2 )
      {
        m_data[ 0 ] = v1[ 0 ], m_data[ 1 ] = v1[ 1 ], m_data[ 2 ] = v1[ 2 ];
        m_data[ 3 ] = v2[ 0 ], m_data[ 4 ] = v2[ 1 ], m_data[ 5 ] = v2[ 2 ];
      }


      inline Vec3 v1() const
      {
        return Vec3( m_data[ 0 ], m_data[ 1 ], m_data[ 2 ] );
      }

      inline Vec3 v2() const
      {
        return Vec3( m_data[ 3 ], m_data[ 4 ], m_data[ 5 ] );
      }

      Vec6& operator=( const Vec6& other ) = default;

      inline const Real& operator [] ( agx::UInt i ) const
      {
        return m_data[ i ];
      }

      inline Real& operator [] ( agx::UInt i )
      {
        return m_data[ i ];
      }

      inline Real operator * ( const Vec6& v ) const
      {
        return  v[ 0 ] * m_data[ 0 ] +
                v[ 1 ] * m_data[ 1 ] +
                v[ 2 ] * m_data[ 2 ] +
                v[ 3 ] * m_data[ 3 ] +
                v[ 4 ] * m_data[ 4 ] +
                v[ 5 ] * m_data[ 5 ];
      }

      inline Vec6 operator * ( const Real& r ) const
      {
        return Vec6(  r * m_data[ 0 ],
                      r * m_data[ 1 ],
                      r * m_data[ 2 ],
                      r * m_data[ 3 ],
                      r * m_data[ 4 ],
                      r * m_data[ 5 ] );
      }

      inline Vec6 operator / ( const Real& r ) const
      {
        Real d = agx::Real(1.0)/r;
        return Vec6(  m_data[ 0 ] *d,
                      m_data[ 1 ] *d,
                      m_data[ 2 ] *d,
                      m_data[ 3 ] *d,
                      m_data[ 4 ] *d,
                      m_data[ 5 ] *d );
      }

      inline Vec6 operator + ( const Vec6& v ) const
      {
        return Vec6(  m_data[ 0 ] + v[ 0 ], m_data[ 1 ] + v[ 1 ], m_data[ 2 ] + v[ 2 ],
                       m_data[ 3 ] + v[ 3 ], m_data[ 4 ] + v[ 4 ], m_data[ 5 ] + v[ 5 ] );
      }

      inline Vec6 operator - ( const Vec6& v ) const
      {
        return Vec6(  m_data[ 0 ] - v[ 0 ], m_data[ 1 ] - v[ 1 ], m_data[ 2 ] - v[ 2 ],
                       m_data[ 3 ] - v[ 3 ], m_data[ 4 ] - v[ 4 ], m_data[ 5 ] - v[ 5 ] );
      }

      inline Vec6 operator - () const
      {
        return Vec6( -m_data[ 0 ], -m_data[ 1 ], -m_data[ 2 ], -m_data[ 3 ], -m_data[ 4 ], -m_data[ 5 ] );
      }

      inline const Vec6& operator += ( const Vec6& v )
      {
        m_data[ 0 ] += v[ 0 ];
        m_data[ 1 ] += v[ 1 ];
        m_data[ 2 ] += v[ 2 ];
        m_data[ 3 ] += v[ 3 ];
        m_data[ 4 ] += v[ 4 ];
        m_data[ 5 ] += v[ 5 ];

        return *this;
      }

      inline const Vec6& operator -= ( const Vec6& v )
      {
        m_data[ 0 ] -= v[ 0 ];
        m_data[ 1 ] -= v[ 1 ];
        m_data[ 2 ] -= v[ 2 ];
        m_data[ 3 ] -= v[ 3 ];
        m_data[ 4 ] -= v[ 4 ];
        m_data[ 5 ] -= v[ 5 ];

        return *this;
      }

      inline const Vec6& operator *= ( const Real& r )
      {
        m_data[ 0 ] *= r;
        m_data[ 1 ] *= r;
        m_data[ 2 ] *= r;
        m_data[ 3 ] *= r;
        m_data[ 4 ] *= r;
        m_data[ 5 ] *= r;

        return *this;
      }

      inline const Vec6& operator /= ( const Real& r )
      {
        Real d = agx::Real(1.0)/r;
        m_data[ 0 ] *= d;
        m_data[ 1 ] *= d;
        m_data[ 2 ] *= d;
        m_data[ 3 ] *= d;
        m_data[ 4 ] *= d;
        m_data[ 5 ] *= d;

        return *this;
      }

      inline bool operator == ( const Vec6& v ) const
      {
        if      ( m_data[ 0 ] != v[ 0 ] ) return false;
        else if ( m_data[ 1 ] != v[ 1 ] ) return false;
        else if ( m_data[ 2 ] != v[ 2 ] ) return false;
        else if ( m_data[ 3 ] != v[ 3 ] ) return false;
        else if ( m_data[ 4 ] != v[ 4 ] ) return false;
        else if ( m_data[ 5 ] != v[ 5 ] ) return false;
        else return true;
      }

      inline bool operator != ( const Vec6& v ) const
      {
        return !( *this == v );
      }

      inline const Real* ptr() const
      {
        return m_data;
      }

      inline Real* ptr()
      {
        return m_data;
      }

      inline Real length() const
      {
        return static_cast< Real >( sqrt(  m_data[ 0 ] * m_data[ 0 ] +
                                           m_data[ 1 ] * m_data[ 1 ] +
                                           m_data[ 2 ] * m_data[ 2 ] +
                                           m_data[ 3 ] * m_data[ 3 ] +
                                           m_data[ 4 ] * m_data[ 4 ] +
                                           m_data[ 5 ] * m_data[ 5 ]  ) );
      }

      inline Real length2() const
      {
        return  m_data[ 0 ] * m_data[ 0 ] +
                m_data[ 1 ] * m_data[ 1 ] +
                m_data[ 2 ] * m_data[ 2 ] +
                m_data[ 3 ] * m_data[ 3 ] +
                m_data[ 4 ] * m_data[ 4 ] +
                m_data[ 5 ] * m_data[ 5 ];
      }

    private:
      Real m_data[ 6 ];
  };

  inline agx::Vec6 operator * ( const agx::Real& r, const agx::Vec6& v )
  {
    return v * r;
  }

} // namespace agx

inline std::ostream& operator << ( std::ostream& o, const agx::Vec6& v )
{
  o << "["  << v[ 0 ] << ", " << v[ 1 ] << ", " << v[ 2 ]
    << ", " << v[ 3 ] << ", " << v[ 4 ] << ", " << v[ 5 ] << "]";

  return o;
}

#endif
