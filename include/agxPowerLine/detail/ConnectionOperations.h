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


#ifndef AGXPOWERLINE_CONNECTION_OPERATIONS_H
#define AGXPOWERLINE_CONNECTION_OPERATIONS_H


#include <agxPowerLine/Sides.h>


// Required for PhysicalDimension::Type.
#include <agxPowerLine/PhysicalDimension.h>

namespace agxPowerLine
{
  class Unit;
  class CompositeUnit;

  class Connector;
  class CompositeConnector;

  namespace detail
  {
    class AGXMODEL_EXPORT ConnectionOperations
    {
      public:

        /* Unit -> rest */

        /**
        Connect Unit->Unit.
        */
        static bool connect(
            agxPowerLine::Unit* unit1,
            agxPowerLine::Side side1,
            agxPowerLine::Side side2,
            agxPowerLine::Unit* unit2);

        /**
        Connect Unit->CompositeUnit
        */
        static bool connect(
            agxPowerLine::Unit* unit1,
            agxPowerLine::Side side1,
            agxPowerLine::Side side2,
            agxPowerLine::CompositeUnit* unit2);

        /**
        Connect Unit->Connector.
        */
        static bool connect(
            agxPowerLine::Unit* unit,
            agxPowerLine::Side unitSide,
            agxPowerLine::Side connectorSide,
            agxPowerLine::Connector* connector);

        /**
        Connect Unit->CompositeConnector.
        */
        static bool connect(
            agxPowerLine::Unit* unit,
            agxPowerLine::Side unitSide,
            agxPowerLine::Side connectorSide,
            agxPowerLine::CompositeConnector* connector);


        /* CompositeUnit -> rest */

        /**
        Connect CompositeUnit->CompositeUnit
        */
        static bool connect(
            agxPowerLine::CompositeUnit* unit1,
            agxPowerLine::Side side1,
            agxPowerLine::Side side2,
            agxPowerLine::CompositeUnit& unit2);

        /**
        Connect CompositeUnit->Connector
        */
        static bool connect(
            agxPowerLine::CompositeUnit* unit,
            agxPowerLine::Side unitSide,
            agxPowerLine::Side connectorSide,
            agxPowerLine::Connector* connector);

        /**
        Connect CompositeUnit->CompositeConnector
        */
        static bool connect(
            agxPowerLine::CompositeUnit* unit,
            agxPowerLine::Side unitSide,
            agxPowerLine::Side connectorSide,
            agxPowerLine::CompositeConnector* connector);


        /* Connector -> rest */

        /**
        Connect Connector->Connector
        */
        static bool connect(
            agxPowerLine::Connector* connector1,
            agxPowerLine::Side* side1,
            agxPowerLine::Side* side2,
            agxPowerLine::Connector* connector2);

        /**
        Connect Connector->CompositeConnector
        */
        static bool connect(
            agxPowerLine::Connector* connector1,
            agxPowerLine::Side side1,
            agxPowerLine::Side side2,
            agxPowerLine::CompositeConnector* connector2);


        /* CompositeConnector -> rest */

        /**
        Connect CompositeConnector->CompositeConnector
        */
        static bool connect(
            agxPowerLine::CompositeConnector* connector1,
            agxPowerLine::Side side1,
            agxPowerLine::Side side2,
            agxPowerLine::CompositeConnector* connector2);



        /* Low-level helpers for connect. */

        static bool lowLevelConnect(
            agxPowerLine::PhysicalDimension* dimension1,
            agxPowerLine::Side side1,
            agxPowerLine::Side side2,
            agxPowerLine::PhysicalDimension* dimension2);

        static void lowLevelConnect(
            agxPowerLine::PhysicalDimension* dimension,
            agxPowerLine::Side dimensionSide,
            agxPowerLine::Side connectorSide,
            agxPowerLine::Connector* connector);






        /*
        Disconnect operations.
        */


        static bool disconnect(
            agxPowerLine::Unit* unit,
            agxPowerLine::Connector* connector);

        static bool lowLevelDisconnect(
            agxPowerLine::PhysicalDimension* dimension,
            agxPowerLine::Connector* connector);

        static bool lowLevelDisconnect(
            agxPowerLine::PhysicalDimension* dimension,
            agxPowerLine::Side dimensionSide,
            agxPowerLine::Side connectorSide,
            agxPowerLine::Connector* connector);



        /*
        Transfer operations.
        */
        static bool transferConnections(
            agxPowerLine::Connector* from,
            agxPowerLine::Connector* to);

        static bool transferConnections(
            agxPowerLine::Connector* from,
            agxPowerLine::Side fromSide,
            agxPowerLine::Side toSide,
            agxPowerLine::Connector* to);


        /* Low-level helpers for transfer. */

        static void lowLevelTransferConnections(
            agxPowerLine::Connector* from,
            agxPowerLine::Side fromSide,
            agxPowerLine::Side toSide,
            agxPowerLine::Connector* to);



        /*
        Root and power line updating.
        */

        static void spreadRootAndPowerLine(
            agxPowerLine::Unit* unit,
            agxPowerLine::Connector* connector);

        static void spreadRootAndPowerLine(
            agxPowerLine::Connector* connector);


      private:
        static bool mayConnect(
            agxPowerLine::Unit* unit1,
            agxPowerLine::Side side1,
            agxPowerLine::Side side2,
            agxPowerLine::Unit* unit2);

        static bool mayConnect(
            agxPowerLine::Unit* unit,
            agxPowerLine::Side unitSide,
            agxPowerLine::Side connectorSide,
            agxPowerLine::Connector* connector);

        static bool mayConnect(
            agxPowerLine::PhysicalDimension* dimension,
            agxPowerLine::Side dimensionSide,
            agxPowerLine::Side connectorSide,
            agxPowerLine::Connector* connector);


        static bool mayTransfer(
            agxPowerLine::Connector* from,
            agxPowerLine::Side fromSide,
            agxPowerLine::Side toSide,
            agxPowerLine::Connector* to);


        static agxPowerLine::PhysicalDimension::Type findCommonType(
            agxPowerLine::Unit* unit1,
            agxPowerLine::Side side1,
            agxPowerLine::Side side2,
            agxPowerLine::Unit* unit2);

        static agxPowerLine::PhysicalDimension::Type findCommonType(
            PhysicalDimension::TypeVector& types1,
            PhysicalDimension::TypeVector& types2);

      private:

        static PhysicalDimension* searchForConnector(ConnectionRefVector& connections, Connector* connector);

        static PhysicalDimension* getUnitDimensionLocal(agxPowerLine::Unit* unit, Connector* connector);

        static PhysicalDimension* getUnitDimension(agxPowerLine::Unit* unit, agxPowerLine::Connector* connector);

        ConnectionOperations() {}
        ~ConnectionOperations() {}
    };
  }
}

#endif
