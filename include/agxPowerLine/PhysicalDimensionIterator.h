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


#ifndef AGXPOWERLINE_PHYSICAL_DIMENSION_ITERATOR_H
#define AGXPOWERLINE_PHYSICAL_DIMENSION_ITERATOR_H

#include <agxModel/export.h>

#include <agxPowerLine/Sides.h>
#include <agxPowerLine/detail/PhysicalDimensionIteratorTraits.h>

namespace agxPowerLine
{
  /**
  Iterator that iterates over the non-null PhysicalDimensions of a Unit.

  The templating is that so we can iterate over both const and non-const Units.
  Iterating over a const Unit will return const PhysicalDimensions.
  */
  template<typename UnitPtr>
  class PhysicalDimensionIterator
  {
    public:
      typedef typename detail::PhysicalDimensionIteratorTypes<UnitPtr>::PhysicalDimensionPtr PhysicalDimensionPtr;

    public:
      PhysicalDimensionIterator(UnitPtr unit);

      /**
      \return A pointer to the current PhysicalDimension, or nullptr if the
      iterator is invalid.
      */
      PhysicalDimensionPtr operator*();

      PhysicalDimensionPtr operator->();

      /**
      Move the iterator to the next PhysicalDimension. The iterator will become
      invalid if there is no next PhysicalDimension.
      */
      PhysicalDimensionIterator& operator++();

      bool isValid() const;

  protected:


    private:
      void scanForNonNull();
      void operator=(const PhysicalDimensionIterator&) {}

    private:
      UnitPtr m_unit;
      size_t m_index;
      const agxPowerLine::PhysicalDimensionRefVector& m_dimensions;
  };

  template <typename UnitPtr>
  PhysicalDimensionIterator<UnitPtr> iterateDimensions(UnitPtr unit);
}

#include <agxPowerLine/detail/PhysicalDimensionIteratorImpl.h>

#endif
