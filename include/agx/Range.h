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

#pragma once

#include <agx/agxCore_export.h>

#include <agx/agx.h>
#include <agx/Vec2.h>

namespace agx
{

  /**
  This class will store a computed violation of a range.

  sign is +1 when violating the upper limit, -1 when violating the lower limit, and 0
  otherwise.
  The excess is either 0 if the argument is in range, or
  the (positive) distance to the nearest bound otherwise.
  */
  template <class T>
  class RangeViolation
  {
    public:

      /// Default constructor. sign=0, v=0
      RangeViolation();

      RangeViolation(int sign, const T& v);

      /***
      \return a reference to the sign which is +1 when violating the upper limit, -1 when violating the lower limit, and 0 otherwise.      
      */
      int& sign();

      /***
      \return the sign which is +1 when violating the upper limit, -1 when violating the lower limit, and 0 otherwise.      
      */
      int sign() const;

      /**
      \return The reference to the excess which is either 0 if the argument is in range, or the (positive) distance to the nearest bound otherwise.
      */
      T& excess();

      /**
      \return The excess which is either 0 if the argument is in range, or the (positive) distance to the nearest bound otherwise.
      */
      T  excess() const;

    private:
      T m_v;
      int m_sign;

  };

  /**
  A range object has a min and max value of a given type and provides
  services to tell whether a value is in that range or not.

  A range object is setup with upper and lower limits, either or both of
  which may be infinite as defined in numeric_limits<Real>::infinity()
  */
  template <class T>
  class Range
  {
    public:
      /// Standard creation: set min and max.
      Range( const T& Min, const T& Max );

      /// Cast operator from Vec2
      explicit Range( const Vec2& vec );

      /// Default constructor. Set min/max to -/+ infinity
      Range();

      /// Set min=-x and max=x
      explicit Range( const T& x );

      /// Copy constructor
      Range( const Range& rhs );

      // Copy constructor with one modified element, for min and max setters.
      Range( const Range& rhs, const T& value, size_t i );

      /**
      \return a reference to the lower value
      */
      T& lower();

      /**
      \return the lower value
      */
      T   lower() const;

      /**
      \return a reference to the upper value
      */
      T& upper();

      /**
      \return the upper value
      */
      T   upper() const;

      /**
      Set the lower bound for the range
      \param lower - the new lower bound
      */
      void setLower(T lower);

      /**
      Set the upper bound for the range
      \param upper - the new upper bound
      */
      void setUpper(T upper);

      /**
      \return true if the range spans an infinite range
      */
      bool isInfinite() const;

      /**
      \return the span of this range (i.e., max - min)
      */
      T span() const;

      /// Assignment operator
      Range& operator=( const Range& rhs );

      template< typename T2 >
      Range operator * ( const T2& rhs ) const;

      /**
      Compute all the details of range violation: parity is +1 when
      violating the upper limit, -1 when violating the lower limit, and 0
      otherwise.  The excess is either 0 if the argument is in range, or
      the (positive) distance to the nearest bound otherwise.
      */
      RangeViolation<T> excess(const T& x) const;

      /**
      \return -1 if the argument x is less than the lower limit, +1 if it
      is greater than the upper limit, and 0 otherwise.
      */
      int outside(const T& x)  const;

      /// \return true if the value x is within range, including boundary.
      bool inside(const T& x ) const;

      /**
      \return true if the two interval overlap, including the case where
      they  touch at the boundary.
      */
      bool overlap(const Range<T>& r ) const;

    private:
      void sync();

    private:
      template <typename T2>
      friend inline std::ostream& operator << ( std::ostream& stream, const Range<T>& range );

      T mMin;
      T mMax;
  };

  template <typename T>
  inline std::ostream& operator << ( std::ostream& stream, const Range<T>& range )
  {
    stream << range.lower() << " " << range.upper();
    return stream;
  }

  /** A short hand for ranges of floating point numbers.
  */
  typedef Range<Real> RangeReal;
  typedef RangeViolation<Real> RangeViolationReal;


  /** A short hand for ranges of unsigned integer numbers.
   */
  typedef Range<UInt> RangeUInt;
  typedef RangeViolation<UInt> RangeViolationUInt;


  // Implementation
  template <class T>
  inline RangeViolation<T>::RangeViolation() : m_sign(0), m_v(0)
  {
  }

  template <class T>
  inline RangeViolation<T>::RangeViolation(int sign, const T& v) : m_v(v), m_sign(sign)
  {

  }

  template <class T>
  AGX_FORCE_INLINE int& RangeViolation<T>::sign()
  {
    return m_sign;
  }

  template <class T>
  AGX_FORCE_INLINE int RangeViolation<T>::sign() const
  {
    return m_sign;
  }

  template <class T>
  AGX_FORCE_INLINE T& RangeViolation<T>::excess()
  {
    return m_v;
  }

  template <class T>
  AGX_FORCE_INLINE T RangeViolation<T>::excess() const
  {
    return m_v;
  }

  template <class T>
  Range<T>::Range( const T& Min, const T& Max ) : mMin( Min ) , mMax( Max )
  {
    sync();
  }

  template <class T>
  inline Range<T>::Range( const Vec2& vec ) : mMin(vec[0]), mMax(vec[1])
  {
    sync();
  }

  template <class T>
  inline Range<T>::Range() : mMin( -std::numeric_limits<T>::infinity() ),
    mMax( std::numeric_limits<T>::infinity() ) {}

  template <class T>
  inline Range<T>::Range( const T& x ) : mMin( -x ),
    mMax( x )
  {
    sync();
  }

  template <class T>
  inline Range<T>::Range( const Range& rhs )
  {
    *this = rhs;
  }

  template <class T>
  inline Range<T>::Range( const Range& rhs, const T& value, size_t i )
  {
    if (i == 0) {
      mMin = value;
      mMax = rhs.mMax;
    } else {
      mMin = rhs.mMin;
      mMax = value;
    }

    sync();
  }

  template <class T>
  AGX_FORCE_INLINE T& Range<T>::lower()
  {
    return mMin;
  }

  template <class T>
  AGX_FORCE_INLINE T Range<T>::lower() const
  {
    return mMin;
  }

  template <class T>
  AGX_FORCE_INLINE T& Range<T>::upper()
  {
    return mMax;
  }

  template <class T>
  AGX_FORCE_INLINE T   Range<T>::upper() const
  {
    return mMax;
  }

  template <class T>
  AGX_FORCE_INLINE void Range<T>::setLower(T lower)
  {
    mMin = lower;
    sync();
  }

  template <class T>
  AGX_FORCE_INLINE void Range<T>::setUpper(T upper)
  {
    mMax = upper;
    sync();
  }

  template <class T>
  AGX_FORCE_INLINE bool Range<T>::isInfinite() const
  {
    return mMin == -std::numeric_limits<T>::infinity() && mMax == std::numeric_limits<T>::infinity();
  }

  template <class T>
  AGX_FORCE_INLINE T Range<T>::span() const
  {
    return mMax - mMin;
  }

  template <class T>
  inline Range<T>& Range<T>::operator=( const Range& rhs )
  {
    this->mMax = rhs.mMax;

    this->mMin = rhs.mMin;

    sync();

    return *this;
  }

  template <class T>
  template< typename T2 >
  AGX_FORCE_INLINE Range<T> Range<T>::operator * ( const T2& rhs ) const
  {
    return Range< T >( mMin * (T)rhs, mMax * (T)rhs );
  }

  template <class T>
  inline RangeViolation<T> Range<T>::excess(const T& x) const
  {
    RangeViolation<T> ret(0, static_cast<T>(0));
    Range<T> lhs = *this;
    if ( x <= lhs.mMin ) {
      ret.sign() = -1;
      ret.excess() = lhs.mMin - x;
    } else if ( x >= lhs.mMax ) {
      ret.sign() =  1;
      ret.excess() = x - lhs.mMax;
    }
    return ret;
  }

  template <class T>
  inline int Range<T>::outside(const T& x)  const
  {
    int ret  = 0;
    Range<T> lhs = *this;
    if ( x < lhs.mMin ) {
      ret = -1;
    } else if ( x > lhs.mMax  ) {
      ret = 1;
    }
    return ret;
  }

  template <class T>
  AGX_FORCE_INLINE bool Range<T>::inside(const T& x ) const
  {
    Range<T> lhs = *this;
    return (x >= lhs.mMin) && (x <= lhs.mMax) ;
  }

  template <class T>
  AGX_FORCE_INLINE bool Range<T>::overlap(const Range<T>& r ) const
  {
    return  ( inside(r.lower())  || inside(r.upper())
              || r.inside(mMin) || r.inside(mMax) )  ;
  }

  template <class T>
  inline void Range<T>::sync()
  {
    if (mMax < mMin ) {
      std::swap(mMax, mMin);
    }
  }
} // namespace agx
