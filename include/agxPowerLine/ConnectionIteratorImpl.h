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


#ifndef AGXPOWERLINE_CONNECTION_ITERATOR_IMPL_H
#define AGXPOWERLINE_CONNECTION_ITERATOR_IMPL_H

#include <agxPowerLine/ConnectionIterator.h>
#include <agxPowerLine/Unit.h>


extern AGXMODEL_EXPORT const agxPowerLine::Side s_firstSearchSide;


template<typename T>
agxPowerLine::ConnectionIterator<T>::ConnectionIterator(SourcePtr dimension) :
  m_source(dimension), m_side(s_firstSearchSide), m_index(0)
{
  this->scanForNonNull();
}


template<typename T>
typename agxPowerLine::ConnectionIterator<T>::ConnectionPtr agxPowerLine::ConnectionIterator<T>::operator*()
{
  if (!this->isValid())
    return 0;

  return this->getConnection();
}


template<typename T>
agxPowerLine::ConnectionIterator<T>& agxPowerLine::ConnectionIterator<T>::operator++()
{
  if (!this->isValid())
    return *this;

  ++m_index;
  this->scanForNonNull();

  return *this;
}


template<typename T>
bool agxPowerLine::ConnectionIterator<T>::isValid() const
{
  return m_source != 0 &&
         m_side != agxPowerLine::Side::NO_SIDE &&
         m_index != agx::InvalidIndex;
}


template<typename T>
const agxPowerLine::ConnectionRefVector& agxPowerLine::ConnectionIterator<T>::getConnections()
{
  return agxPowerLine::detail::ConnectionIteratorOperations<T>::getConnections( m_source, m_side );
}


template<typename T>
typename agxPowerLine::ConnectionIterator<T>::ConnectionPtr agxPowerLine::ConnectionIterator<T>::getConnection()
{
  return this->getConnections()[m_index];
}


template<typename T>
bool agxPowerLine::ConnectionIterator<T>::isConnectionValid()
{
  ConnectionPtr connection = this->getConnection();
  if (connection == 0)
    return false;

  return detail::ConnectionIteratorTypes<T>::getTarget(connection) != 0;
}


template<typename T>
size_t agxPowerLine::ConnectionIterator<T>::getNumConnections()
{
  return this->getConnections().size();
}


template<typename T>
void agxPowerLine::ConnectionIterator<T>::scanForNonNull()
{
  this->scanConnectionsForNonNull();
  if (m_index == agx::InvalidIndex && m_side == s_firstSearchSide) {
    m_side = agxPowerLine::opposite(m_side);
    m_index = 0;
    this->scanConnectionsForNonNull();
  }
  if (m_index == agx::InvalidIndex) {
    this->invalidate();
  }
}


template<typename T>
void agxPowerLine::ConnectionIterator<T>::scanConnectionsForNonNull()
{
  for (; m_index < this->getNumConnections(); ++m_index) {
    if (this->isConnectionValid()) {
      return;
    }
  }
  m_index = agx::InvalidIndex;
}


template<typename T>
void agxPowerLine::ConnectionIterator<T>::invalidate()
{
  m_source = 0;
  m_side = agxPowerLine::Side::NO_SIDE;
  m_index = agx::InvalidIndex;
}



#endif
