/*
Copyright 2007-2023. Algoryx Simulation AB.

All AGX source code, intellectual property, documentation, sample code,
tutorials, scene files and technical white papers, are copyrighted, proprietary
and confidential material of Algoryx Simulation AB. You may not download, read,
store, distribute, publish, copy or otherwise disseminate, use or expose this
material unless having a written signed agreement with Algoryx Simulation AB, or
having been advised so by Algoryx Simulation AB for a time limited evaluation,
or having purchased a valid commercial license from Algoryx Simulation AB.

Algoryx Simulation AB disclaims all responsibilities for loss or damage caused
from using this software, unless otherwise stated in written agreements with
Algoryx Simulation AB.
*/

#ifndef AGXPOWERLINE_POWER_LINE_H
#define AGXPOWERLINE_POWER_LINE_H




#include <agxModel/export.h>
#include <agxPowerLine/Unit.h>

#include <agxSDK/StepEventListener.h>
#include <agxStream/Serializable.h>
#include <agxStream/StorageStream.h>

#include <agxUtil/agxUtil.h>

#include <agx/RigidBody.h>
#include <agx/Singleton.h>
#include <agx/ElementaryConstraint.h>

#include <agxPowerLine/SlotMapper.h>
#include <agxPowerLine/RotationalDimension.h>
#include <agxPowerLine/TranslationalDimension.h>

#include <agxSDK/Assembly.h>


#include <agx/HashVector.h>




#define POWER_LINE_DEBUG 0


namespace agxPowerLine
{

  class Actuator;
  class PowerLinesHandler;
  class PowerLineCollector;

  typedef agx::HashVector< agxPowerLine::Connector*, bool > ConnectorBoolHashVector;
  typedef agx::HashVector< agxPowerLine::SubGraph*, bool > SubGraphRouterBoolHashVector;
  typedef agx::HashVector< agxPowerLine::Unit*, agxPowerLine::PowerLine*> UnitPtrPowerLinePtrHashVector;

  typedef agx::observer_ptr< PowerLine > PowerLineObsPtr;
  typedef agx::ref_ptr< PowerLine > PowerLineRef;
  typedef agx::VectorPOD< PowerLine* > PowerLinePtrVector;
  typedef agx::Vector< PowerLineRef > PowerLineRefVector;

  /**
  PowerLine is the base class for a system of power transfer integrated into the rigid body simulation.
  */
  class AGXMODEL_EXPORT PowerLine : public agxSDK::Assembly
  {
  public:
    public:

      PowerLine();

    public:
      /**
      \return the indexed value for a named physical dimension. Will index a non existing name by a call to bool addPhysicalDimension(std::string name);
         \returns -1 if the name is not indexed and the name fails to be indexed(how?).
      */
      int getDimensionType(std::string name);

      /**
      Creates a constraint holder for all elementary constraints between the physical dimensions.
      */
      void createConstraintHolder();

      /**
      \returns pointer to the constraint holder.
      */
      agxUtil::ConstraintHolder* getConstraintHolder() const;

      /**
      Decide if: when units added to this powerLine
      are to be connected to units not belonging to any power line
      will be added to this power line, or not.
      */
      void setAddUnitsOnConnect(agx::Bool addOnConnect);
      agx::Bool getAddUnitsOnConnect() const;

      /**
      Get pointers to all roots, each connected to a sub graph.
      the root set will be increased in size for each new root. (Existing roots are to remain in set).
      */
      void getRoots(agxPowerLine::UnitPtrPowerLinePtrHashVector& units) const;



      /**
      ---------------------------------------------------------------------------------------
      ---------------------------------------------------------------------------------------
      ---------------------------------------------------------------------------------------
      OBSOLETE SINCE ROOT STRUCTURE
      */
      /**
      Set the unit the traverser shall start traversing from.
      */
      bool setSource(agxPowerLine::Unit* source);
      bool setSource(agxPowerLine::SubGraph* source);

      /**
      \returns pointer to the source unit.
      */
      const agxPowerLine::Unit* getSource() const;
      agxPowerLine::Unit* getSource();
      /**
      OBSOLETE SINCE ROOT STRUCTURE
      ---------------------------------------------------------------------------------------
      ---------------------------------------------------------------------------------------
      ---------------------------------------------------------------------------------------
      */


      bool add(agxPowerLine::Unit* component);
      bool add(agxPowerLine::SubGraph* component);
      bool remove(agxPowerLine::SubGraph* component);

      size_t getNumUnits() const;
      size_t getNumConnectors() const;

      /**
      Find and return the first unit with the given name, if any exist. Will do
      a linear search through all units.
      \param name - The name of the unit to get.
      \return A unit with the given name, or nullptr.
      */
      Unit* getUnit(const agx::Name& name);

      /**
      Find and return the first unit with the given name, if any exist. Will do
      a linear search through all units.
      \param name - The name of the unit to get.
      \return A unit with the given name, or nullptr.
      */
      const Unit* getUnit(const agx::Name& name) const;

      /**
      Set the current solve type
      DIRECT_AND_ITERATIVE is used by default
      */
      void setSolveType( agx::Constraint::SolveType solveType );

      /**
      \returns the current solve type
      */
      agx::Constraint::SolveType getSolveType( ) const;


      void insertSubGraphRouterTemporary(SubGraph* router);

      void insertConnectorTemporary( Connector* connector );

      void collectConstraintsToHolder();

      /**
      Write the current graph state of the powerline to a .dot file (path)
      \return true if successful.
      */
      bool writeDimensionsGraph(const agx::String& path) const;
      bool writeDimensionsGraph(const char* path) const;


      const agxPowerLine::UnitRefSetVector& getUnits() const;
      const agxPowerLine::ConnectorRefSet& getConnectors() const;

      virtual void addNotification(agxSDK::Simulation* simulation) override;
      virtual void removeNotification(agxSDK::Simulation* simulation) override;

      /**
      Called when this assembly is added to another assembly
      */
      virtual void addNotification(agxSDK::Assembly*) override {}
      using agxSDK::Assembly::addNotification;
      using agxSDK::Assembly::removeNotification;
      /**
      Called when this assembly is removed from another assembly
      */
      virtual void removeNotification(agxSDK::Assembly*) override {}
      agxPowerLine::SlotMapper* getSlotMapper();

      /**
      Stores internal data for this PowerLine into stream.

      The StorageStream may be left in an inconsistent state if an error occurs.

      \param str - Stream to write into. Must be in STORE mode.
      \return True if the store was successful. False otherwise.
      */
      virtual bool store(agxStream::StorageStream& str) const;

      /**
      Restores internal data from stream.

      Both the PowerLine and the StorgeStream may be left in an inconsistent
      state if an error occurs during restore.

      \param str - The stream to read from. Must be in RESTORE mode.
      \return True if the restore was successfull. False otherwise.
      */
      virtual bool restore(agxStream::StorageStream& str);

      AGXSTREAM_DECLARE_SERIALIZABLE(agxPowerLine::PowerLine);

      static bool renderDebug();

      /**
      Search the given simulation for a power line with the given name.
      \param simulation - The simulation to search in.
      \param name - The name of the power line to search for.
      \return A power line with name \p name in simulation \p simulation, or nullptr.
      */
      static agxPowerLine::PowerLine* find(agxSDK::Simulation* simulation, const agx::Name& name);

      /**
      Search the given assembly for a power line with the given name.
      \param assembly - The assembly to search in.
      \param name - The name of the power line to search for.
      \return A power line with name \p name in assembly \p assembly, or nullptr.
      */
      static agxPowerLine::PowerLine* find(agxSDK::Assembly* assembly, const agx::Name& name);

      /**
      Find all powerlines with given name.
      \param simulation - simulation the terrain is part of
      \param name - name of the powerlines
      \return vector of powerlines
      */
      static PowerLinePtrVector findAll( const agxSDK::Simulation* simulation, const agx::Name& name );

      /**
      Finds all powerlines in the given simulation.
      \param simulation - simulation with powerlines.
      \return vector of powerlines
      */
      static PowerLinePtrVector findAll( const agxSDK::Simulation* simulation );

    public:
      bool removeSubGraph(agxPowerLine::SubGraph* component);
      bool removeUnit(agxPowerLine::Unit* unit);
      bool removeConnector(agxPowerLine::Connector* connector);

      bool pointAddConnector(agxPowerLine::Connector* connector);
      bool pointAddUnit(agxPowerLine::Unit* unit);
      bool pointReplace( agxPowerLine::Unit* oldUnit, agxPowerLine::Unit* newUnit );

      bool cacheAdd(agxPowerLine::SubGraph* component);
      bool cacheRemove(agxPowerLine::SubGraph* component);
      bool hasCachedOperations() const;
      void clearAddRemoveCache(ConnectorPtrVector& addedConnectors, UnitPtrVector& addedUnits);

    protected:

      friend class PowerLinesHandler;

      virtual ~PowerLine();

      bool traverseUnits(bool (*callback)(agxPowerLine::Unit*));
      bool traverseDimensions(bool (*callback)(agxPowerLine::PhysicalDimension*));
      bool traverseConnectors(bool (*callback)(agxPowerLine::Connector*));

      virtual void initialize();

      void pre(agx::Real time);
      void post(agx::Real time);

      void prepareConstraintDimensions();

      bool preUpdateGraph();

    protected:
      void restorePre46(agxStream::InputArchive& in);

      /**
      Set the given Unit's PowerLine to \p this.
      Used by unit tests to circumvent a full PowerLine::add(Unit*).
      */
      void setUnitPowerLine(agxPowerLine::Unit* unit);



    private:
      bool shouldRenderDebug();

    protected:
      agxPowerLine::UnitRefSetVector m_units;
      agxPowerLine::ConnectorRefSet m_connectors;

      agxPowerLine::SlotMapperRef m_slotMapper;

      agx::observer_ptr<agx::RigidBody> m_parentBody;

      agxUtil::ConstraintHolderRef m_constraintHolder;

      bool m_addOnConnect;

      agx::Constraint::SolveType m_solveType;

      SubGraphRouterBoolHashVector m_tempRouterHashVector;
      ConnectorBoolHashVector m_tempConnectorHashVector;

      agxPowerLine::SubGraphRefVector m_cachedAdds;
      agxPowerLine::SubGraphRefVector m_cachedRemoves;
      bool m_cacheAddsAndRemoves;
  };
}


// AGXMODEL_POWER_LINE_H
#endif
