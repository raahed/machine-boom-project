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


#ifndef AGXPOWERLINE_CONNECTOR_ITERATOR_H
#define AGXPOWERLINE_CONNECTOR_ITERATOR_H

#include <agxModel/export.h>

#include <agxPowerLine/Sides.h>
#include <agxPowerLine/PhysicalDimension.h>
#include <agxPowerLine/Connector.h>

#include <agxPowerLine/ConnectionIteratorTraits.h>

namespace agxPowerLine
{
  /**
   * Iterator that iterates over all, both input and output, connections that
   * have a non-null Connector.
   *
   * The template parameter T is the type of the object that contains the Connections
   * to iterate over. Currently supported types are const and non-const PhysicalDimension
   * and Connector.
   */
  template<typename T>
  class ConnectionIterator
  {
    public:
      typedef typename detail::ConnectionIteratorTypes<T>::SourcePtr  SourcePtr;
      typedef typename detail::ConnectionIteratorTypes<T>::ConnectionPtr ConnectionPtr;

    public:
      ConnectionIterator(SourcePtr dimension);

      ConnectionPtr operator*();
      ConnectionIterator& operator++();

      bool isValid() const;

    private:
      const ConnectionRefVector& getConnections();
      ConnectionPtr getConnection();
      bool isConnectionValid();
      size_t getNumConnections();
      void scanForNonNull();
      void scanConnectionsForNonNull();
      void invalidate();

    private:
      SourcePtr m_source;
      Side m_side;
      size_t m_index;
  };
}



#include <agxPowerLine/ConnectionIteratorImpl.h>

#endif
