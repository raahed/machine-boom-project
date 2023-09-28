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


#ifndef AGXPOWERLINE_SUB_GRAPH_ROUTER_H
#define AGXPOWERLINE_SUB_GRAPH_ROUTER_H

#include <agxModel/export.h>
#include <agx/Constraint.h>

namespace agx
{
  class RigidBody;
}

namespace agxPowerLine
{
  class PowerLine;
  class PhysicalDimension;
  class Connector;
  class Unit;

  typedef agx::Vector<Connector*> ConnectorPtrVector;
  typedef agx::Vector<Unit*> UnitPtrVector;

  typedef agx::ref_ptr<Connector> ConnectorRef;
  typedef agx::Vector<ConnectorRef> ConnectorRefVector;
  typedef agx::ref_ptr<Unit> UnitRef;
  typedef agx::Vector<UnitRef> UnitRefVector;
}

namespace agxUtil
{
  class ConstraintHolder;
}

namespace agxPowerLine
{

  AGX_DECLARE_POINTER_TYPES(SubGraph);
  AGX_DECLARE_VECTOR_TYPES(SubGraph);


  /**
   *
   */
  class AGXMODEL_EXPORT SubGraph : public agx::Referenced, public agxStream::Serializable
  {
    public:

      SubGraph();

      /**
      Set the name of this power line component.
      \param name - The new name.
      */
      void setName(const agx::Name& name);

      /**
      \return The name of this power line component.
      */
      const agx::Name& getName() const;

      SubGraph* getEnclosingGraph();
      const SubGraph* getEnclosingGraph() const;

      /**
       * \return The top-most Enclosing Graph. May return 'this'.
       */
      SubGraph* getRootEnclosingGraph();

      /**
       * \return The top-most Enclosing Graph. May return 'this'.
       */
      const SubGraph* getRootEnclosingGraph() const;

      Unit* getEnclosingUnit();
      const Unit* getEnclosingUnit() const;

      Connector* getEnclosingConnector();
      const Connector* getEnclosingConnector() const;

      /**
      \return True if this component is enclosed within the given component.
      */
      bool isEnclosedWithin(const SubGraph* component) const;

      /**
      Fill the given vector with the Connectors that are held by this SubGraph.
      The collection is recursive, so internal Connectors will be collected.
      */
      virtual void getConnectorsRecursive(ConnectorPtrVector& connectors);

      /**
      Fill the given vector with the Units that are held by this SubGraph. The
      collection is recursive, so internal Units will be collected.
      */
      virtual void getUnitsRecursive(UnitPtrVector& units);


    // Methods called by the rest of the PowerLine frame work.
    public:

      virtual bool addNotification(agxSDK::Simulation* simulation) = 0;

      virtual void prepareForStore();
      virtual void prepareForRestore();


      const agxPowerLine::UnitRefVector& getInternalUnits() const;
      const agxPowerLine::ConnectorRefVector& getInternalConnectors() const;
      virtual bool removeNotification(agxUtil::ConstraintHolder* holder, agxSDK::Simulation* simulation) = 0;


      void spreadPowerLineToGraph();

      /**
      \internal
      Called every time step before the solve.

      While it is allowed to do some restructuring of the PowerLine graph in the
      callback, such as adding or removing nodes, it is illegal to change the
      inter-graph relationships. That is, one may not connect two graph nodes
      that have separate roots that both were part of the power line before the
      call to PowerLine::pre for the current time step.
      */
      virtual bool preUpdate(agx::Real timeStep);
      virtual bool postUpdate(agx::Real timeStep);

      /**
      Stores internal data into stream.
      */
      virtual bool store(agxStream::StorageStream& str) const;

      /**
      Restores internal data from stream.
      */
      virtual bool restore(agxStream::StorageStream& str);

      AGXSTREAM_DECLARE_ABSTRACT_SERIALIZABLE(agxPowerLine::SubGraph);
      virtual void restore(agxStream::InputArchive& in) override;
      virtual void store(agxStream::OutputArchive& out) const override;

    protected:
      virtual ~SubGraph();

      void addInternalUnit(agxPowerLine::Unit* unit);
      void addInternalConnector(agxPowerLine::Connector* connector);

      void removeInternalUnit(agxPowerLine::Unit* unit);
      void removeInternalConnector(agxPowerLine::Connector* connector);

      agx::Bool replaceInternalUnit(agxPowerLine::Unit* oldUnit, agxPowerLine::Unit* newUnit);

    protected:
      SubGraphObserver m_enclosingGraph;
      UnitRefVector m_internalUnits;
      ConnectorRefVector m_internalConnectors;
      PowerLine* m_powerLine;
      agx::Name m_name;
  };


}

#endif
