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

#ifndef AGXPOWERLINE_POWER_LINE_TRAVERSER_H
#define AGXPOWERLINE_POWER_LINE_TRAVERSER_H


#include <agxModel/export.h>
#include <agxPowerLine/Unit.h>
#include <agxPowerLine/PhysicalDimension.h>
#include <agxPowerLine/Connector.h>

namespace agxPowerLine
{
  /**
  Enumeration listing the types of nodes that the PowerLineTraversers can visit.
  Represents a bit-field, so values may be or:ed together.
  */
  enum GraphNodeTypes
  {
    UNITS = 1,
    DIMENSIONS = 2,
    CONNECTORS = 4,
    ALL_GRAPH_NODES = UNITS | DIMENSIONS | CONNECTORS
  };

  inline GraphNodeTypes operator|(GraphNodeTypes lhs, GraphNodeTypes rhs)
  {
    return static_cast<GraphNodeTypes>((int)lhs | (int)rhs);
  }



  /**
  Base class for all power line traversers. Templated on the node type pointers
  so that both const and non-const versions are possible.

  For each node in the graph the traverser may call the pure virtual 'visit'
  method on itself, depending on the current value of the GraphNodeTypes variable.

  Subclasses can control the extent of the traversal by overriding
  'shouldQueueChildren'. By returning 'false' when passed a Unit the traversal
  will not descend into the PhysicalDimensions of that Unit, and by returning
  'false' when passed a PhysicalDimension or a Connector the traversal will not
  follow any Connections from that PhysicalDimension or Connector. The default
  implementations always returns true.

  There are two types of relationships in a power line graph: connections and
  compounds. Connections link PhysicalDimensions and Connectors, while compounds
  nest Connectors and Units inside each other. The traverser can work in two modes.
  It either only follows the PhysicalDimension/Connector connections or both the
  connections and the nesting.

  */
  template<typename UnitPtr, typename DimensionPtr, typename ConnectorPtr, typename SubGraphPtr>
  class AbstractPowerLineTraverser
  {
    public:
      /**
      Enumeration that controls how the traversal should behave when encountering
      compound GraphNodes.
      */
      enum TraversalMode
      {
        CONNECTIONS_ONLY,
        CONNECTIONS_AND_COMPOUNDS
      };

    public:
      AbstractPowerLineTraverser();

      /**
      Start a traversal that only visits Units.
      */
      bool traverseUnits(UnitPtr source, TraversalMode mode);

      /**
      Start a traversal that only visits PhysicalDimensions.
      */
      bool traverseDimensions(UnitPtr source, TraversalMode mode);

      /**
      Start a traversal that only visits Connectors.
      */
      bool traverseConnectors(UnitPtr source, TraversalMode mode);

      /**
      Start a traversal that visits the types specified by the given GraphNodeTypes.
      */
      bool traverse(GraphNodeTypes typesToVisit, UnitPtr source, TraversalMode mode);

      /**
      Start a traversal from all the units in the powerline that visits the types specified by the given GraphNodeTypes.
      */
      bool traverse(GraphNodeTypes typesToVisit, PowerLine* source, TraversalMode mode);

    protected:
      virtual ~AbstractPowerLineTraverser() {}

      /**
      Called when a Unit is encountered in the graph, if the GraphNodeTypes
      variable includes UNITS.
      */
      virtual bool visit(UnitPtr unit) = 0;

      /**
      Called when a PhysicalDimension is encountered in the graph, if the
      GraphNodeTypes variable includes DIMENSIONS.
      */
      virtual bool visit(DimensionPtr dimension) = 0;

      /**
      Called when a Connector is encountered in the graph, if the GraphNodeTypes
      variable includes CONNECTORS.
      */
      virtual bool visit(ConnectorPtr connector) = 0;

      virtual bool shouldQueueChildren(UnitPtr unit);
      virtual bool shouldQueueChildren(DimensionPtr dimension);
      virtual bool shouldQueueChildren(ConnectorPtr connector);

    private:
      void approachDimension(DimensionPtr dimension, TraversalMode mode);
      void approachConnector(ConnectorPtr connector, TraversalMode mode);
      void approachUnit(UnitPtr unit, TraversalMode mode);
      void traverseToConnector(ConnectorPtr connector, TraversalMode mode);
      void traverseToUnit(UnitPtr unit, TraversalMode mode);
      void traverseInternal(SubGraphPtr root);
      void queueDimensionsInUnit(UnitPtr unit);
      void queueDimensionsInConnector(ConnectorPtr connector, TraversalMode mode);
      void collectCompoundNode(SubGraphPtr root);

    private:
      agxPowerLine::GraphNodeTypes m_typesToVisit;
      agx::SetVector<UnitPtr> m_seenUnits;
      agx::SetVector<DimensionPtr> m_seenDimensions;
      agx::SetVector<ConnectorPtr> m_seenConnectors;
      agx::Vector<DimensionPtr> m_queue;
      bool m_allVisitedSuccessfully;
  };



  /// Template specialization for non-const traversal.
  class AGXMODEL_EXPORT PowerLineTraverser :
      public AbstractPowerLineTraverser<
        agxPowerLine::Unit*,
        agxPowerLine::PhysicalDimension*,
        agxPowerLine::Connector*,
        agxPowerLine::SubGraph*>
  {
  };

  /// Template specialization for const traversal.
  class AGXMODEL_EXPORT PowerLineConstTraverser :
      public AbstractPowerLineTraverser<
        const agxPowerLine::Unit*,
        const agxPowerLine::PhysicalDimension*,
        const agxPowerLine::Connector*,
        const agxPowerLine::SubGraph*>
  {
  };


  /**
  PowerLineTraverser that stores the nodes it visits. It can be seen as taking
  a snapshot of the graph contents so that the same nodes can be processed later
  even if the graph itself has been reconfigured.

  The collector keeps references to the collected nodes so that they stay valid
  even if they are removed from the graph itself.
  */
  class AGXMODEL_EXPORT PowerLineCollector : public agxPowerLine::PowerLineTraverser
  {
    public:
      PowerLineCollector();
      PowerLineCollector(agxPowerLine::PowerLine* source, TraversalMode mode);
      PowerLineCollector(agxPowerLine::Unit* source, TraversalMode mode);
      PowerLineCollector(agxPowerLine::GraphNodeTypes typesToCollect, agxPowerLine::Unit* source, TraversalMode mode);

      const agxPowerLine::UnitRefVector& getUnits();
      const agxPowerLine::PhysicalDimensionRefVector& getDimensions();
      const agxPowerLine::ConnectorRefVector& getConnectors();

    protected:
      virtual bool visit(agxPowerLine::Unit* unit) override;
      virtual bool visit(agxPowerLine::PhysicalDimension* dimension) override;
      virtual bool visit(agxPowerLine::Connector* connector) override;

    private:
      agxPowerLine::UnitRefVector m_units;
      agxPowerLine::PhysicalDimensionRefVector m_dimensions;
      agxPowerLine::ConnectorRefVector m_connectors;
  };


  /**
  Traverser that calls a user supplied callback in its 'visit' methods. Can be
  used if creating a proper subclass is considered to cumbersome and only simple
  operations are to be performed.
  */
  class AGXMODEL_EXPORT PowerLineCallbackTraverser : public PowerLineTraverser
  {
    public:
      DOXYGEN_START_INTERNAL_BLOCK()
      /// Moving from raw function pointers to std::function as the need arises.
      typedef bool(*ShouldExecuteUnit)(agxPowerLine::Unit*);
      typedef bool(*ShouldQueueUnitChildren)(agxPowerLine::Unit*);
      typedef std::function<bool(agxPowerLine::Unit*)> ExecuteUnit;

      typedef bool(*ShouldExecuteDimension)(agxPowerLine::PhysicalDimension*);
      typedef bool(*ShouldQueueDimensionChildren)(agxPowerLine::PhysicalDimension*);
      typedef bool(*ExecuteDimension)(agxPowerLine::PhysicalDimension*);

      typedef bool(*ShouldExecuteConnector)(agxPowerLine::Connector*);
      typedef bool(*ShouldQueueConnectorChildren)(agxPowerLine::Connector*);
      typedef std::function<bool(agxPowerLine::Connector*)> ExecuteConnector;
      DOXYGEN_END_INTERNAL_BLOCK()

    public:
      PowerLineCallbackTraverser();

      bool traverseUnits(Unit* source, ShouldExecuteUnit shouldFunc, ShouldQueueUnitChildren shouldQueueFunc, ExecuteUnit executeFunc, TraversalMode mode);
      bool traverseDimensions(Unit* source, ShouldExecuteDimension shouldFunc, ShouldQueueDimensionChildren shouldQueueFunc, ExecuteDimension executeFunc, TraversalMode mode);
      bool traverseConnectors(Unit* source, ShouldExecuteConnector shouldFunc, ShouldQueueConnectorChildren shouldQueueFunc, ExecuteConnector executeFunc, TraversalMode mode);

      void setShouldExecuteUnit(ShouldExecuteUnit func);
      void setShouldQueueUnitChildren(ShouldQueueUnitChildren func);
      void setExecuteUnit(ExecuteUnit func);

      void setShouldExecuteDimension(ShouldExecuteDimension func);
      void setShouldQueueDimensionChildren(ShouldQueueDimensionChildren func);
      void setExecuteDimension(ExecuteDimension func);

      void setShouldExecuteConnector(ShouldExecuteConnector func);
      void setShouldQueueConnectorChildren(ShouldQueueConnectorChildren func);
      void setExecuteConnector(ExecuteConnector func);

      using PowerLineTraverser::traverseUnits;
      using PowerLineTraverser::traverseDimensions;
      using PowerLineTraverser::traverseConnectors;

    protected:
      virtual bool visit(agxPowerLine::Unit* unit) override;
      virtual bool visit(agxPowerLine::PhysicalDimension* dimension) override;
      virtual bool visit(agxPowerLine::Connector* connector) override;

      virtual bool shouldQueueChildren(Unit* unit) override;
      virtual bool shouldQueueChildren(PhysicalDimension* dimension) override;
      virtual bool shouldQueueChildren(Connector* connector) override;


    private:
      ShouldExecuteUnit m_shouldExecuteUnitFunc;
      ShouldQueueUnitChildren m_shouldQueueUnitChildrenFunc;
      ExecuteUnit m_executeUnitFunc;

      ShouldExecuteDimension m_shouldExecuteDimensionFunc;
      ShouldQueueDimensionChildren m_shouldQueueDimensionChildrenFunc;
      ExecuteDimension m_executeDimensionFunc;

      ShouldExecuteConnector m_shouldExecuteConnectorFunc;
      ShouldQueueConnectorChildren m_shouldQueueConnectorChildrenFunc;
      ExecuteConnector m_executeConnectorFunc;
  };


}

// Implementations of templated methods moved to separate file for readability purposes.
#include <agxPowerLine/PowerLineTraverserImpl.h>


#endif
