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


#ifndef AGXPOWERLINE_CONNECTION_ITERATOR_TRAITS_H
#define AGXPOWERLINE_CONNECTION_ITERATOR_TRAITS_H

#include <agxPowerLine/Sides.h>
#include <agxPowerLine/PhysicalDimension.h>
#include <agxPowerLine/Connector.h>

namespace agxPowerLine
{

  namespace detail
  {

    template<typename T>
    class ConnectionIteratorTypes
    {
      public:
    };

    template<>
    class ConnectionIteratorTypes<PhysicalDimension*>
    {
      public:
        typedef PhysicalDimension* SourcePtr;
        typedef Connection* ConnectionPtr;
        typedef Connector* TargetPtr;
        typedef Side SideType;

        static TargetPtr getTarget(ConnectionPtr connection)
        {
          return connection->getConnector();
        }
    };


    template<>
    class ConnectionIteratorTypes<const PhysicalDimension*>
    {
      public:
        typedef const PhysicalDimension* SourcePtr;
        typedef const Connection* ConnectionPtr;
        typedef const Connector* TargetPtr;
        typedef Side SideType;

        static TargetPtr getTarget(ConnectionPtr connection)
        {
          return connection->getConnector();
        }
    };

    template<>
    class ConnectionIteratorTypes<Connector*>
    {
      public:
        typedef Connector* SourcePtr;
        typedef Connection* ConnectionPtr;
        typedef PhysicalDimension* TargetPtr;
        typedef Side SideType;

        static TargetPtr getTarget(ConnectionPtr connection)
        {
          return connection->getDimension();
        }
    };


    template<>
    class ConnectionIteratorTypes<const Connector*>
    {
      public:
        typedef const Connector* SourcePtr;
        typedef const Connection* ConnectionPtr;
        typedef const PhysicalDimension* TargetPtr;
        typedef Side SideType;

        static TargetPtr getTarget(ConnectionPtr connection)
        {
          return connection->getDimension();
        }
    };




    template<typename T>
    class ConnectionIteratorOperations
    {
      public:
        typedef typename ConnectionIteratorTypes<T>::SourcePtr SourcePtr;
        typedef typename ConnectionIteratorTypes<T>::ConnectionPtr ConnectionPtr;
        typedef typename ConnectionIteratorTypes<T>::TargetPtr TargetPtr;
        typedef typename ConnectionIteratorTypes<T>::SideType SideType;

        static const ConnectionRefVector& getConnections(SourcePtr source, Side side)
        {
          return source->getConnections((SideType)side);
        }

        static ConnectionPtr getConnection(SourcePtr source, Side side, size_t index)
        {
          return getConnections(source, side)[index];
        }

        static TargetPtr getTarget(ConnectionPtr connection)
        {
          return ConnectionIteratorTypes<T>::getTarget(connection);
        }

        static bool isConnectionValid(SourcePtr source, Side side, size_t index)
        {
          ConnectionPtr connection = getConnection(source, side, index);
          return connection != 0 && getTarget(connection) != 0;
        }
    };

  }
}


#endif
