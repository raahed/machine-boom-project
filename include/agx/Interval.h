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

#ifndef AGX_INTERVAL_H
#define AGX_INTERVAL_H

DOXYGEN_START_INTERNAL_BLOCK()


#include <agx/agxPhysics_export.h>

#ifndef _WIN32
#include <float.h>
#endif

#include <agx/agx.h>

namespace agx
{
  /**
  * A class describing a closed one-dimensional mathematical interval.
  * The interval is considered as empty if its min is larger than its max.
  */
  class AGXPHYSICS_EXPORT Interval {
  public:
    /// Creates an interval with min and max values.
    Interval( Real newMinValue, Real newMaxValue );

    /// Creates an interval with min and max values. Sorts them first if sorted is set to false.
    Interval( Real valueA, Real valueB, bool sorted );

    /// Creates an interval -inf as min and inf as max.
    Interval();

    /// Intersects with other interval.
    void intersectWith( const Interval& other );

    /// Intersects with an interval which is unbounded to the left.
    void intersectWithRightUnbounded( Real leftBound );

    /// Intersects with an interval which is unbounded to the right.
    void intersectWithLeftUnbounded( Real rightBound );

    /**
    * Computes an overlap with another interval, leaving both unchanged.
    * \param other The other interval.
    * \retval The overlapping interval.
    */
    Interval computeOverlapWith(const Interval& other) const;

    /// Does it have any overlap with this other interval?
    bool hasOverlapWith(const Interval& other) const;

    /// Is this value contained in the interval?
    bool containsValue(const Real value) const;

    /// Gives minimum value.
    Real& minValue();

    /// Gives maximum value.
    Real& maxValue();

      /// Gives minimum value.
    Real minValue() const;

    /// Gives maximum value.
    Real maxValue() const;

    /// Is the interval empty? (Is its min > max?). Opposite of hasElements.
    bool isEmpty() const;

    /// Does the interval have any elements? (Is its min <= max?). Opposite of isEmpty.
    bool hasElements() const;

    /*
    * Computes length of interval.
    * Will give 0 for an interval with 1 element and a negative number for empty intervals.
    */
    Real getLength() const;

  protected:
    Real m_values[2];
  };



  /// Implementations

  AGX_FORCE_INLINE Interval::Interval()
  {
    minValue() = -std::numeric_limits<Real>::infinity();
    maxValue() = std::numeric_limits<Real>::infinity();
  }


  AGX_FORCE_INLINE void Interval::intersectWith( const Interval& other )
  {
    minValue() = std::max( minValue(), other.minValue() );
    maxValue() = std::min( maxValue(), other.maxValue() );
  }


  AGX_FORCE_INLINE void Interval::intersectWithRightUnbounded( Real leftBound )
  {
    minValue() = std::max( minValue(), leftBound );
  }


  AGX_FORCE_INLINE void Interval::intersectWithLeftUnbounded( Real rightBound )
  {
    maxValue() = std::min( maxValue(), rightBound );
  }


  AGX_FORCE_INLINE Real& Interval::minValue()
  {
    return m_values[0];
  }


  AGX_FORCE_INLINE Real& Interval::maxValue()
  {
    return m_values[1];
  }


  AGX_FORCE_INLINE Real Interval::minValue() const
  {
    return m_values[0];
  }


  AGX_FORCE_INLINE Real Interval::maxValue() const
  {
    return m_values[1];
  }


  AGX_FORCE_INLINE bool Interval::isEmpty() const
  {
    return minValue() > maxValue();
  }


  AGX_FORCE_INLINE bool Interval::hasElements() const
  {
    return minValue() <= maxValue();
  }


  AGX_FORCE_INLINE Interval::Interval( Real newMinValue, Real newMaxValue )
  {
    minValue() = newMinValue;
    maxValue() = newMaxValue;
  }


  AGX_FORCE_INLINE Interval::Interval( Real valueA, Real valueB, bool sorted )
  {
    if (sorted) {
      minValue() = valueA;
      maxValue() = valueB;
    }
    else {
      minValue() = std::min(valueA, valueB);
      maxValue() = std::max(valueA, valueB);
    }
  }

  AGX_FORCE_INLINE Interval Interval::computeOverlapWith(const Interval& other) const
  {
    return Interval(std::max( minValue(), other.minValue() ),
               std::min( maxValue(), other.maxValue() ));
  }


  AGX_FORCE_INLINE bool Interval::hasOverlapWith(const Interval& other) const
  {
    return !isEmpty() && !other.isEmpty() &&
      (containsValue(other.minValue()) || containsValue(other.maxValue()) ||
      other.containsValue(minValue()) || other.containsValue(maxValue()));
  }


  AGX_FORCE_INLINE bool Interval::containsValue(const Real value) const
  {
    return (value >= minValue() && value <= maxValue());
  }


  AGX_FORCE_INLINE Real Interval::getLength() const
  {
    return maxValue() - minValue();
  }


}

DOXYGEN_END_INTERNAL_BLOCK()
#endif
