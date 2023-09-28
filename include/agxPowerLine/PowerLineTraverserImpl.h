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


#ifndef AGXPOWERLINE_POWER_LINE_TRAVERSER_IMPL_H
#define AGXPOWERLINE_POWER_LINE_TRAVERSER_IMPL_H

#include <agxPowerLine/PowerLineTraverser.h>
#include <agxPowerLine/PhysicalDimension.h>
#include <agxPowerLine/ConnectionIterator.h>
#include <agxPowerLine/PowerLine.h>

#include <agx/Vector.h>
#include "PhysicalDimensionIterator.h"



namespace agxPowerLine
{
  template<typename UnitPtr, typename DimensionPtr, typename ConnectorPtr, typename SubGraphPtr>
  AbstractPowerLineTraverser<UnitPtr, DimensionPtr, ConnectorPtr, SubGraphPtr>::AbstractPowerLineTraverser()
    : m_typesToVisit(ALL_GRAPH_NODES)
    , m_allVisitedSuccessfully(true)
  {
  }



  template<typename UnitPtr, typename DimensionPtr, typename ConnectorPtr, typename SubGraphPtr>
  bool AbstractPowerLineTraverser<UnitPtr, DimensionPtr, ConnectorPtr, SubGraphPtr>::traverseUnits(
      UnitPtr source,
      TraversalMode mode)
  {
    return this->traverse(agxPowerLine::UNITS, source, mode);
  }



  template<typename UnitPtr, typename DimensionPtr, typename ConnectorPtr, typename SubGraphPtr>
  bool AbstractPowerLineTraverser<UnitPtr, DimensionPtr, ConnectorPtr, SubGraphPtr>::traverseDimensions(
      UnitPtr source,
      TraversalMode mode)
  {
    return this->traverse(agxPowerLine::DIMENSIONS, source, mode);
  }



  template<typename UnitPtr, typename DimensionPtr, typename ConnectorPtr, typename SubGraphPtr>
  bool AbstractPowerLineTraverser<UnitPtr, DimensionPtr, ConnectorPtr, SubGraphPtr>::traverseConnectors(
      UnitPtr source,
      TraversalMode mode)
  {
    return this->traverse(agxPowerLine::CONNECTORS, source, mode);
  }





  template<typename UnitPtr, typename DimensionPtr, typename ConnectorPtr, typename SubGraphPtr>
  void AbstractPowerLineTraverser<UnitPtr, DimensionPtr, ConnectorPtr, SubGraphPtr>::approachDimension(
      DimensionPtr dimension,
      TraversalMode mode)
  {
    // std::cout << "Visiting dimension " << dimension->getBody()->getStaticTypeName() << std::endl;
    if (m_typesToVisit & agxPowerLine::DIMENSIONS)
    {
      m_allVisitedSuccessfully &= this->visit(dimension);
    }

    for (auto it = ConnectionIterator<DimensionPtr>(dimension); it.isValid(); ++it)
    {
      ConnectorPtr connector = (*it)->getConnector();
      this->approachConnector(connector, mode);
    }
  }



  template<typename UnitPtr, typename DimensionPtr, typename ConnectorPtr, typename SubGraphPtr>
  void AbstractPowerLineTraverser<UnitPtr, DimensionPtr, ConnectorPtr, SubGraphPtr>::approachConnector(
      ConnectorPtr connector,
      TraversalMode mode)
  {
    if (m_seenConnectors.contains(connector))
    {
      return;
    }

    switch (mode)
    {
      case CONNECTIONS_AND_COMPOUNDS:
      {
        SubGraphPtr root = connector->getRootEnclosingGraph();
        if (root != connector)
        {
          this->collectCompoundNode(root);
        }
        else
        {
          this->traverseToConnector(connector, CONNECTIONS_AND_COMPOUNDS);
        }
        break;
      }
      case CONNECTIONS_ONLY:
        this->traverseToConnector(connector, CONNECTIONS_ONLY);
        break;
    }
  }



  template<typename UnitPtr, typename DimensionPtr, typename ConnectorPtr, typename SubGraphPtr>
  void AbstractPowerLineTraverser<UnitPtr, DimensionPtr, ConnectorPtr, SubGraphPtr>::approachUnit(
      UnitPtr unit,
      TraversalMode mode)
  {
    if (m_seenUnits.contains(unit))
    {
      return;
    }

    switch (mode)
    {
      case CONNECTIONS_AND_COMPOUNDS:
      {
        SubGraphPtr root = unit->getRootEnclosingGraph();
        if (root != unit)
        {
          this->collectCompoundNode(root);
        }
        else
        {
          this->traverseToUnit(unit, CONNECTIONS_AND_COMPOUNDS);
        }
        break;
      }
      case CONNECTIONS_ONLY:
        this->traverseToUnit(unit, CONNECTIONS_ONLY);
        break;
    }
  }



  template<typename UnitPtr, typename DimensionPtr, typename ConnectorPtr, typename SubGraphPtr>
  void AbstractPowerLineTraverser<UnitPtr, DimensionPtr, ConnectorPtr, SubGraphPtr>::traverseToConnector(
      ConnectorPtr connector,
      TraversalMode mode)
  {
    m_seenConnectors.insert(connector);

    if (m_typesToVisit & agxPowerLine::CONNECTORS)
    {
      m_allVisitedSuccessfully &= this->visit(connector);
    }

    if (this->shouldQueueChildren(connector))
    {
      this->queueDimensionsInConnector(connector, mode);
    }

    if (mode == CONNECTIONS_AND_COMPOUNDS)
    {
      this->traverseInternal(connector);
    }
  }



  template<typename UnitPtr, typename DimensionPtr, typename ConnectorPtr, typename SubGraphPtr>
  void AbstractPowerLineTraverser<UnitPtr, DimensionPtr, ConnectorPtr, SubGraphPtr>::traverseToUnit(
      UnitPtr unit,
      TraversalMode mode)
  {
    m_seenUnits.insert(unit);

    if (m_typesToVisit & agxPowerLine::UNITS)
    {
      m_allVisitedSuccessfully &= this->visit(unit);
    }

    if (this->shouldQueueChildren(unit))
    {
      this->queueDimensionsInUnit(unit);
    }

    if (mode == CONNECTIONS_AND_COMPOUNDS)
    {
      this->traverseInternal(unit);
    }
  }



  template<typename UnitPtr, typename DimensionPtr, typename ConnectorPtr, typename SubGraphPtr>
  void AbstractPowerLineTraverser<UnitPtr, DimensionPtr, ConnectorPtr, SubGraphPtr>::traverseInternal(SubGraphPtr root)
  {
    // Can never have seen an internal Connector or Unit before because internals
    // are always visited top-to-bottom whenever a compound SubGraph is found,
    // regardless of where in that three the compound SubGraph is first encountered.

    for (size_t i = 0; i < root->getInternalConnectors().size(); ++i)
    {
      ConnectorPtr connector = root->getInternalConnectors()[i];
      agxVerify(!m_seenConnectors.contains(connector));
      this->traverseToConnector(connector, CONNECTIONS_AND_COMPOUNDS);
    }

    for (size_t i = 0; i < root->getInternalUnits().size(); ++i)
    {
      UnitPtr unit = root->getInternalUnits()[i];
      agxVerify(!m_seenUnits.contains(unit));
      this->traverseToUnit(unit, CONNECTIONS_AND_COMPOUNDS);
    }
  }




  template<typename UnitPtr, typename DimensionPtr, typename ConnectorPtr, typename SubGraphPtr>
  void AbstractPowerLineTraverser<UnitPtr, DimensionPtr, ConnectorPtr, SubGraphPtr>::queueDimensionsInConnector(
      ConnectorPtr connector,
      TraversalMode mode)
  {
    for (auto it = ConnectionIterator<ConnectorPtr>(connector); it.isValid(); ++it)
    {
      DimensionPtr dimension = (*it)->getDimension();
      if (m_seenDimensions.contains(dimension))
        continue;

      UnitPtr unit = dimension->getUnit();
      if (m_seenUnits.contains(unit))
        continue;

      this->approachUnit(unit, mode);
    }
  }


  template<typename UnitPtr, typename DimensionPtr, typename ConnectorPtr, typename SubGraphPtr>
  void AbstractPowerLineTraverser<UnitPtr, DimensionPtr, ConnectorPtr, SubGraphPtr>::queueDimensionsInUnit(UnitPtr unit)
  {
    for (auto it = PhysicalDimensionIterator<UnitPtr>(unit); it.isValid(); ++it)
    {
      DimensionPtr dimension = *it;
      if (m_seenDimensions.contains(dimension))
        continue;
      m_seenDimensions.insert(dimension);
      m_queue.push_back(dimension);
    }
  }



  template<typename UnitPtr, typename DimensionPtr, typename ConnectorPtr, typename SubGraphPtr>
  void AbstractPowerLineTraverser<UnitPtr, DimensionPtr, ConnectorPtr, SubGraphPtr>::collectCompoundNode(SubGraphPtr root)
  {
    if (UnitPtr unit = dynamic_cast<UnitPtr>(root))
    {
      if (!m_seenUnits.contains(unit))
        this->traverseToUnit(unit, CONNECTIONS_AND_COMPOUNDS);
    }
    else if (ConnectorPtr connector = dynamic_cast<ConnectorPtr>(root))
    {
      if (!m_seenConnectors.contains(connector))
        this->traverseToConnector(connector, CONNECTIONS_AND_COMPOUNDS);
    }
  }






  template<typename UnitPtr, typename DimensionPtr, typename ConnectorPtr, typename SubGraphPtr>
  bool AbstractPowerLineTraverser<UnitPtr, DimensionPtr, ConnectorPtr, SubGraphPtr>::traverse(
      GraphNodeTypes typesToVisit,
      UnitPtr source,
      TraversalMode mode)
  {
    agxVerify(source != nullptr);

    // Make sure we didn't leak state from the last traversal.
    agxVerify(m_allVisitedSuccessfully);
    agxVerify(m_seenConnectors.empty());
    agxVerify(m_seenDimensions.empty());
    agxVerify(m_seenUnits.empty());
    agxVerify(m_queue.empty());


    m_typesToVisit = typesToVisit;

    // std::cout << "\n  Starting new traverse." << std::endl;

    this->approachUnit(source, mode);

    while (!m_queue.empty())
    {
      DimensionPtr dimension = m_queue.back();
      m_queue.pop_back();

      this->approachDimension(dimension, mode);
    }

    // std::cout << "  Traverse complete" << std::endl;

    // Don't leak state into the next traversal.
    bool result = m_allVisitedSuccessfully;
    m_allVisitedSuccessfully = true;
    m_seenConnectors.clear();
    m_seenDimensions.clear();
    m_seenUnits.clear();

    return result;
  }


  template<typename UnitPtr, typename DimensionPtr, typename ConnectorPtr, typename SubGraphPtr>
  bool AbstractPowerLineTraverser<UnitPtr, DimensionPtr, ConnectorPtr, SubGraphPtr>::traverse(
      GraphNodeTypes typesToVisit,
      PowerLine* source,
      TraversalMode mode)
  {
    agxVerify(source != nullptr);

    // Make sure we didn't leak state from the last traversal.
    agxVerify(m_allVisitedSuccessfully);
    agxVerify(m_seenConnectors.empty());
    agxVerify(m_seenDimensions.empty());
    agxVerify(m_seenUnits.empty());
    agxVerify(m_queue.empty());

    m_typesToVisit = typesToVisit;

    for (size_t i = 0; i < source->getUnits().size(); ++i) {
      this->approachUnit(source->getUnits()[i], mode);

      while (!m_queue.empty())
      {
        DimensionPtr dimension = m_queue.back();
        m_queue.pop_back();

        this->approachDimension(dimension, mode);
      }
    }

    // Don't leak state into the next traversal.
    bool result = m_allVisitedSuccessfully;
    m_allVisitedSuccessfully = true;
    m_seenConnectors.clear();
    m_seenDimensions.clear();
    m_seenUnits.clear();

    return result;
  }







  template<typename UnitPtr, typename DimensionPtr, typename ConnectorPtr, typename SubGraphPtr>
  bool AbstractPowerLineTraverser<UnitPtr, DimensionPtr, ConnectorPtr, SubGraphPtr>::shouldQueueChildren(UnitPtr /*unit*/)
  {
    return true;
  }

  template<typename UnitPtr, typename DimensionPtr, typename ConnectorPtr, typename SubGraphPtr>
  bool AbstractPowerLineTraverser<UnitPtr, DimensionPtr, ConnectorPtr, SubGraphPtr>::shouldQueueChildren(DimensionPtr /*dimension*/)
  {
    return true;
  }

  template<typename UnitPtr, typename DimensionPtr, typename ConnectorPtr, typename SubGraphPtr>
  bool AbstractPowerLineTraverser<UnitPtr, DimensionPtr, ConnectorPtr, SubGraphPtr>::shouldQueueChildren(ConnectorPtr /*connector*/) {
    return true;
  }
}
#endif
