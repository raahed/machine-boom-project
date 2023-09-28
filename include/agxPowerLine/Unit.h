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

#ifndef AGXPOWERLINE_UNIT_H
#define AGXPOWERLINE_UNIT_H

#include <agxModel/export.h>
#include <agxPowerLine/PhysicalDimension.h>
#include <agxPowerLine/PowerLineUtils.h>
#include <agxPowerLine/Connector.h>
#include <agxPowerLine/SubGraph.h>

#include <agxSDK/StepEventListener.h>
#include <agxStream/Serializable.h>

#include <agxUtil/agxUtil.h>

#include <agx/RigidBody.h>
#include <agx/ElementaryConstraint.h>


namespace agxPowerLine
{
  /************************************************************************/
  /*
  The agxPowerLine::PowerLine implementation is the base for energy transportation
  in a 1D physical system.

  The energy is stored in units connected by connectors. Each unit can have a set of both
  in and out connections, representing different physical dimensions.

  The physical dimensions can either be dynamically solved or be represented as
  a kinematic calculator that uses a lookup table to calculate its current state
  given the surrounding units. It could affect the power produced in its dynamically
  solved parent, or the state of gradient for that parent.(For example a look up table
  working as a gear box for a drive train will affect the torque and RPM of the
  incoming shaft. Since it has no shaft of its own it will convert the parent shaft )
  */
  /************************************************************************/
  class Unit;
  class Traverser;
  typedef agx::ref_ptr<Unit> UnitRef;
  typedef agx::Vector< UnitRef > UnitRefVector;
  typedef agx::Vector< agx::observer_ptr< Unit > > UnitObsVector;
  typedef agx::Vector< ConnectorRef > ConnectorRefVector;
  typedef agx::Vector< Connector* > ConnectorPtrVector;
  typedef agx::HashSet< UnitRef > UnitRefSet;
  typedef agx::SetVector< UnitRef > UnitRefSetVector;
  typedef agx::SetVector< Unit* > UnitPtrSetVector;


  typedef agx::Vector<std::pair<agxPowerLine::PhysicalDimension*, agxPowerLine::Side> > DimensionSideVector;


  /**
  Pure virtual class.

  */
  class AGXMODEL_EXPORT Unit : public agxPowerLine::SubGraph
  {
  public:
    /**
    Constructor.
    */
    Unit();

    /**
    \return True if the unit and its connections and constraints are valid.
    */
    virtual bool isValid() const;

    /**
    \returns true if this is connected the other through a connector
    */
    virtual bool isConnected(agxPowerLine::Unit* otherUnit) const;

    void getConnectedUnits(UnitPtrSetVector& result, agxPowerLine::Side side);

    /**
    Define which body in the simulation this Unit is attached to. If the body
    is nullptr the Unit is attached to world. This will override any parent
    body set to the power line the unit is a part of.
    */
    void attach(agx::RigidBody* body);

    /**
    Connect the unit to a connector
    \param connector - a connector
    \returns true and connects if there are physical dimensions that match
    */
    virtual bool connect(agxPowerLine::Connector* connector);

    /**
     * Connect the unit to a connector, specifying the side of each.
     * \param mySide - The side on this unit to connect.
     * \param connectorSide - The side on the connector to connect.
     * \param connector - The connector to connect.
     * \return True if the connection was made, false otherwise.
     */
    virtual bool connect(
      agxPowerLine::Side mySide,
      agxPowerLine::Side connectorSide,
      agxPowerLine::Connector* connector);

    /**
    Connect the unit to another unit.
    \param otherUnit - another unit.
    */
    virtual bool connect(agxPowerLine::Unit* otherUnit);

    virtual bool connect(
        agxPowerLine::Side mySide,
        agxPowerLine::Side otherSide,
        agxPowerLine::Unit* other);



    virtual bool disconnect(agxPowerLine::Unit* unit);

    void disconnectAll();



  // Methods called by the rest of the PowerLine/Hydraulics frame work.
  public:
    /**
    \return The dimension of the given types that accepts connections on the
            given side. Can be a PhysicalDimension owned by an internal Unit.
    */
    virtual agxPowerLine::DimensionAndSide getConnectableDimension(
        agxPowerLine::PhysicalDimension::Type type, agxPowerLine::Side side);

    /**
    \return The dimension of the given types that accepts connections. Can be a PhysicalDimension owned by an internal Unit.
    */
    agxPowerLine::PhysicalDimension* getConnectableInputDimension(
      agxPowerLine::PhysicalDimension::Type type);

    agxPowerLine::PhysicalDimension* getConnectableOutputDimension(
      agxPowerLine::PhysicalDimension::Type type);

    /**
    \internal
    \todo Why does this take a Vector instead of a Set? We need the Set
          behavior. Is is because of determinism? Will we always test
          dimension types in the same order every time if a Set is used
          here? How about a sorted set? How about a bit set indexed on the
          dimension type IDs?
    */
    virtual void getConnectableDimensionTypes(
        agxPowerLine::PhysicalDimension::TypeVector& types, agxPowerLine::Side side) const;


    /**
    \return The number of connectable PhysicalDimensions in this Unit, including internal.
    */
    virtual size_t getNumConnectableDimensions(Side side) const;


    /**
    Constructs the set of Units connected to this Unit's own dimensions.
    Does not consider internal units.
    */
    void getOwnConnectedUnits(UnitPtrSetVector& unitSet, bool startFromCleanUnitSet = true) const;
    void getOwnConnectedUnits(UnitPtrSetVector& unitSet, agxPowerLine::Side side, bool startFromCleanUnitSet = true) const;

    /**
    \return A vector containing the PhysicalDimensions in this Unit, not
           including internal Units.
    */
    const agxPowerLine::PhysicalDimensionRefVector& getDimensions() const;
    agxPowerLine::PhysicalDimensionRefVector& getDimensions();


    /**
    \return The PhysicalDimension with the given type, or nullptr. Does not
            search internal Units.
    */
    agxPowerLine::PhysicalDimension* getDimension(agxPowerLine::PhysicalDimension::Type type);
    const agxPowerLine::PhysicalDimension* getDimension(agxPowerLine::PhysicalDimension::Type type) const;


    /**
    \return The index:th non-null PhysicalDimension in this Unit, not including
            internal Units, or nullptr.
    */
    agxPowerLine::PhysicalDimension* getActiveDimension(size_t index);
    const agxPowerLine::PhysicalDimension* getActiveDimension(size_t index) const;

    /**
    The number of non-null PhysicalDimension in this Unit, not including
    internal Units.
    */
    size_t getNumActiveDimensions() const;

    /**
    \return A concatenation of the names of the bodies of the dimensions in
    this Unit.
    */
    agx::String buildDimensionsName() const;

    /*
    Functions called by static functions. Do NOT use them.
    */


    void addNotification(agxPowerLine::PowerLine* powerLine);
    void removeNotification(agxPowerLine::PowerLine* powerLine);

    DOXYGEN_START_INTERNAL_BLOCK()
    /**
    \internal
    May be called multiple times without any removeNotification in between.
    Happens when some other part of the circuit is reconfigured.
    */
    virtual bool addNotification(agxSDK::Simulation* simulation) override;
    virtual bool removeNotification(agxUtil::ConstraintHolder* constraintHolder, agxSDK::Simulation* simulation) override;
    DOXYGEN_END_INTERNAL_BLOCK()


    agxPowerLine::PowerLine* getPowerLine() const;

    /// \todo Shold this search internally?
    bool hasConnector( const Connector* connector ) const;


    /// \todo Include external connections made on internal Units?
    ///       Does not consider internal units.
    size_t getNumConnections() const;

    /**
    Append all connectors connecting this Unit to the given Unit to the
    'result' Vector. Returns the index of the first Connector added. Will be
    one-past-end (result.size()) if no connectors are added.

    \return The index of the first appended Connector.

    \todo Should this incldue connectors found on internal boundary Units?
     */
    size_t getConnectors(const agxPowerLine::Unit* unit, agxPowerLine::ConnectorPtrVector& result);
    size_t getInputConnectors(const agxPowerLine::Unit* unit, agxPowerLine::ConnectorPtrVector& result);
    size_t getOutputConnectors(const agxPowerLine::Unit* unit, agxPowerLine::ConnectorPtrVector& result);

    size_t getConnectors(agxPowerLine::ConnectorPtrVector& result) const;
    size_t getInputConnectors(agxPowerLine::ConnectorPtrVector& result) const;
    size_t getOutputConnectors(agxPowerLine::ConnectorPtrVector& result) const;


    Connector* getFirstInputConnector(const agxPowerLine::Unit* unit) const;
    Connector* getFirstOutputConnector(const agxPowerLine::Unit* unit) const;

    agxPowerLine::Side getSide(const Connector* connector) const;


    bool getIgnoreForStoreRestoreStream() const;


    virtual bool postStore(agxStream::StorageStream& str) const;

    virtual bool postRestore(agxStream::StorageStream& str);

    /**
    Stores internal data into stream.
    */
    virtual bool store(agxStream::StorageStream& str) const override;

    /**
    Restores internal data from stream.
    */
    virtual bool restore(agxStream::StorageStream& str) override;

    void spreadRootToGraph();

    agxPowerLine::Unit* getRoot() const;

    void setRoot(agxPowerLine::Unit* root);

    virtual bool vetoConnect(agxPowerLine::Connector* connector) const;

    AGXSTREAM_DECLARE_ABSTRACT_SERIALIZABLE(agxPowerLine::Unit);
    virtual void store(agxStream::OutputArchive& out) const override;
    virtual void restore(agxStream::InputArchive& in) override;

  protected:
    virtual ~Unit();

    /**
    Traverser helper functions
    */
    friend class Traverser;
    friend class GraphTraverser;
    friend class PowerLine;
    friend class Actuator;
    friend class Connector;

    /**
    returns if it is possible to connect to this unit
    */
    virtual bool hasValidInputConnection() const;

    /**
    Point to the power line the unit is a part of.
    */
    void setPowerLine(PowerLine* powerLine);

    bool activateDimension( PhysicalDimension* dimension );

    void restorePre46(agxStream::InputArchive& in);

  protected:
    PhysicalDimensionRefVector m_dimensions;

    agx::observer_ptr<agx::RigidBody> m_attachBody;

    bool m_ignoreMeForStoreRestoreStream;

    agxPowerLine::Unit* m_rootUnit;
  };

  typedef agx::ref_ptr< Unit > UnitRef;
}

#endif // AGXMODEL_UNIT_H
