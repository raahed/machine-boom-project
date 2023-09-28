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

#ifndef AGXMODEL_PHYSICAL_DIMENSION_ITERATOR_IMPL_H
#define AGXMODEL_PHYSICAL_DIMENSION_ITERATOR_IMPL_H

#include <agxPowerLine/Unit.h>


// Only included to help the IDE. Will be removed by the preprocessor.
#include <agxPowerLine/PhysicalDimensionIterator.h>


template<typename UnitPtr>
agxPowerLine::PhysicalDimensionIterator<UnitPtr>::PhysicalDimensionIterator(UnitPtr unit) :
  m_unit(unit), m_index(0), m_dimensions(unit->getDimensions())
{
  this->scanForNonNull();
}



template<typename UnitPtr>
typename agxPowerLine::PhysicalDimensionIterator<UnitPtr>::PhysicalDimensionPtr agxPowerLine::PhysicalDimensionIterator<UnitPtr>::operator*()
{
  if (this->isValid())
    return m_dimensions[m_index];
  else
    return nullptr;
}



template<typename UnitPtr>
typename agxPowerLine::PhysicalDimensionIterator<UnitPtr>::PhysicalDimensionPtr agxPowerLine::PhysicalDimensionIterator<UnitPtr>::operator->()
{
  if (this->isValid())
    return m_dimensions[m_index];
  else
    return nullptr;
}



template<typename UnitPtr>
agxPowerLine::PhysicalDimensionIterator<UnitPtr>& agxPowerLine::PhysicalDimensionIterator<UnitPtr>::operator++()
{
  if (m_index == m_dimensions.size())
  {
    return *this;
  }

  ++m_index;
  this->scanForNonNull();

  return *this;
}



template<typename UnitPtr>
bool agxPowerLine::PhysicalDimensionIterator<UnitPtr>::isValid() const
{
  return m_index < m_dimensions.size() && m_dimensions[m_index] != nullptr;
}



template<typename UnitPtr>
void agxPowerLine::PhysicalDimensionIterator<UnitPtr>::scanForNonNull()
{
  for (; m_index < m_dimensions.size(); ++m_index)
  {
    if (this->isValid())
    {
      break;
    }
  }
}



template <typename UnitPtr>
agxPowerLine::PhysicalDimensionIterator<UnitPtr> agxPowerLine::iterateDimensions(UnitPtr unit)
{
  return PhysicalDimensionIterator<UnitPtr>(unit);
}


#endif
