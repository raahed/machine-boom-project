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


#ifndef AGX_RANGE_6_H
#define AGX_RANGE_6_H

#include <agx/Range.h>
#include <agxData/Type.h>

namespace agx
{

  template <typename T>
  class Range6T
  {
  public:
    enum Direction {
      X_DIRECTION, Y_DIRECTION, Z_DIRECTION,
      X_ROTATION,  Y_ROTATION,  Z_ROTATION,
      NUM_DIRECTIONS
    };

  public:
    Range<T>& operator[](Direction direction);
    const Range<T>& operator[](Direction direction) const;

    Range<T>& operator[](UInt index);
    const Range<T>& operator[](UInt index) const;

    Range<T> get(Direction direction);
    void set(Direction direction, Range<T> newRange);
  private:
    Range<T> m_ranges[6];
  };


  template <typename T>
  Range<T>& Range6T<T>::operator[](Direction direction)
  {
    agxAssert(direction < 6);
    return m_ranges[direction];
  }


  template <typename T>
  const Range<T>& Range6T<T>::operator[](Direction direction) const
  {
    agxAssert(direction < 6);
    return m_ranges[direction];
  }


  template <typename T>
  Range<T>& Range6T<T>::operator[](UInt index)
  {
    agxAssert(index < 6);
    return m_ranges[index];
  }


  template <typename T>
  const Range<T>& Range6T<T>::operator[](UInt index) const
  {
    agxAssert(index < 6);
    return m_ranges[index];
  }


  template <typename T>
  Range<T> Range6T<T>::get(Direction direction)
  {
    agxAssert(direction < 6);
    return m_ranges[direction];
  }


  template <typename T>
  void Range6T<T>::set(Direction direction, Range<T> newRange)
  {
    agxAssert(direction < 6);
    m_ranges[direction] = newRange;
  }




  typedef Range6T<Real> Range6;
  typedef Range6T<Real32> Range6f;
  typedef Range6T<Real64> Range6d;

}

#endif
