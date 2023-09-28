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

#ifndef AGXPOWERLINE_CONNECTOR_H
#define AGXPOWERLINE_CONNECTOR_H

#include <agxModel/export.h>

#include <agxPowerLine/PowerLineUtils.h>
#include <agxStream/Serializable.h>
#include <agxStream/StorageStream.h>
#include <agx/List.h>
#include <agx/Vector.h>
#include <agx/SetVector.h>
#include <agxUtil/agxUtil.h>
#include <agxPowerLine/PhysicalDimension.h>
#include <agxPowerLine/Sides.h>
#include <agxPowerLine/SubGraph.h>
#include <agx/LookupTables.h>

namespace agxPowerLine
{
  class Unit;
  class PowerLine;
  class PhysicalDimensionMultiBodyConstraintImplementation;
  class ElementaryPhysicalDimensionConstraint;

  typedef agx::SetVector<Unit*> UnitPtrSetVector;
  typedef agx::SetVector<const Unit*> UnitConstPtrSetVector;
  typedef agx::Vector< PhysicalDimension::Type > DimensionTypeVector;

  // Local typdef to avoid include loop.
  typedef agx::ref_ptr<Unit> UnitRef;
  typedef agx::Vector<UnitRef> UnitRefVector;

  /**
  Links two connections together.
  Also defines the constraint between the two connected units.
  */
  class AGXMODEL_EXPORT Connector : public agxPowerLine::SubGraph
  {
    public:

      /**
      create a connector
      */
      Connector();

      virtual void initConstraint();

      virtual bool addNotification(agxSDK::Simulation* simulation) override;

      /**
      remove connector and constraint from simulation
      */
      virtual bool removeNotification(agxUtil::ConstraintHolder* holder, agxSDK::Simulation* simulation) override;

      /**
      Connect a unit to this Connector.
      */
      virtual bool connect(agxPowerLine::Unit* unit);

      virtual bool connect(
          agxPowerLine::Side mySide,
          agxPowerLine::Side unitSide,
          agxPowerLine::Unit* unit);

      /**
       * Disconnect the given unit from this Connector. Will reduce the refCount
       * of the Connector, so the instance may have been deleted when this method
       * returns. Keep an extra ConnectorRef, at the call site of elsewhere, to
       * avoid this.
       */
      virtual bool disconnect(agxPowerLine::Unit* unit);

      virtual bool disconnect();


      virtual agxPowerLine::Side getSide(const agxPowerLine::Unit* unit) const;


      agx::Real getPowerOut() const;

      agx::Real getPowerIn() const;

      /**
      calculate simulated efficiency
      */
      agx::Real calculateEfficiency() const;

      /**
      calculate violation for constraint
      */
      virtual agx::Real calculateViolation( ) const;

      bool compareWithInputTypes( PhysicalDimensionRefVector& dimensions ) const;

      bool compareWithOutputTypes( PhysicalDimensionRefVector& dimensions ) const;

      /// \todo Take Vector& argument and collect types from m_inputConnectors/m_outputConnectors.
      const DimensionTypeVector& getInputTypes() const;
      const DimensionTypeVector& getOutputTypes() const;
      const DimensionTypeVector& getTypes(agxPowerLine::Side side) const;

      // Get a hash vector with pointers to all units connected to this connector.
      /// \todo Loop over all Connectors on boundary.
      bool getInputUnits(UnitPtrSetVector& units, const bool startFromCleanUnitSet = true);
      bool getOutputUnits(UnitPtrSetVector& units, const bool startFromCleanUnitSet = true);
      bool getUnits(UnitPtrSetVector& units, agxPowerLine::Side side, const bool startFromCleanUnitSet = true);
      bool getUnits(UnitPtrSetVector& units, const bool startFromCleanUnitSet = true);


      // Get a hash vector with pointers to all units connected to this connector.
      bool getInputUnits(UnitConstPtrSetVector& units, const bool startFromCleanUnitSet = true) const;
      bool getOutputUnits(UnitConstPtrSetVector& units, const bool startFromCleanUnitSet = true) const;
      bool getUnits(UnitConstPtrSetVector& units, agxPowerLine::Side side, const bool startFromCleanUnitSet = true) const;
      bool getUnits(UnitConstPtrSetVector& units, const bool startFromCleanUnitSet = true) const;

      static PhysicalDimension* getDimension(const ConnectionRefVector& connections, const PhysicalDimension::Type dimensionType);
      PhysicalDimension* getInputDimension(const PhysicalDimension::Type dimensionType) const;
      PhysicalDimension* getOutputDimension(const PhysicalDimension::Type dimensionType) const;

      /**
      Define number of input dimensions the constraint should handle.
      */
      void setNumPossibleInputConnections( int num );

      /**
      Define number of output dimensions the constraint should handle
      */
      void setNumPossibleOutputConnections( int num );

      /**
      \returns number of input dimensions the constraint can handle.
      */
      int getNumPossibleInputConnections() const;

      /**
      \return number of output dimensions the constraint can handle.
      */
      int getNumPossibleOutputConnections() const;


      int getNumInputConnections() const;
      int getNumOutputConnections() const;
      int getNumConnections(agxPowerLine::Side side) const;
      int getNumConnections() const;

      // All of these return -1 when the any number of connections is allowed.
      int getNumFreeInputConnections() const;
      int getNumFreeOutputConnections() const;
      int getNumFreeConnections(agxPowerLine::Side side) const;
      int getNumFreeConnections() const;

      bool hasFreeInputConnection() const;
      bool hasFreeOutputConnection() const;
      bool hasFreeConnection(agxPowerLine::Side side) const;
      bool hasFreeConnection() const;

      /**
      Find out if the connection belongs to this connector.
      */
      bool hasInputConnection(Connection* connection);

      /**
      \return true if if the connection belongs to this connector.
      */
      bool hasOutputConnection(Connection* connection);


      /**
      return a specific output connection
      */
      const agxPowerLine::Connection* getInputConnection(size_t index) const;

      /**
      return a specific output connection
      */
      const agxPowerLine::Connection* getOutputConnection(size_t index) const;

      /**
      return a specific input connection
      */
      agxPowerLine::Connection* getInputConnection(size_t index);

      /**
      return a specific input connection
      */
      agxPowerLine::Connection* getOutputConnection(size_t index);

      agxPowerLine::Connection* findInputConnection(const agxPowerLine::Unit* unit);
      agxPowerLine::Connection* findOutputConnection(const agxPowerLine::Unit* unit);
      agxPowerLine::Connection* findConnection(const agxPowerLine::Unit* unit, agxPowerLine::Side side);


      const agxPowerLine::Connection* findInputConnection(const agxPowerLine::Unit* unit) const;
      const agxPowerLine::Connection* findOutputConnection(const agxPowerLine::Unit* unit) const;
      const agxPowerLine::Connection* findConnection(const agxPowerLine::Unit* unit, agxPowerLine::Side side) const;

      /**
      return vector to all output connections from this connector
      */
      const agxPowerLine::ConnectionRefVector& getOutputConnections() const;

      /**
      return vector to all input connections from this connector
      */
      const agxPowerLine::ConnectionRefVector& getInputConnections() const;

      /**
      return vector to all output connections from this connector
      */
      agxPowerLine::ConnectionRefVector& getOutputConnections();

      /**
      return vector to all input connections from this connector
      */
      agxPowerLine::ConnectionRefVector& getInputConnections();

      const agxPowerLine::ConnectionRefVector& getConnections(agxPowerLine::Side side) const;

      DOXYGEN_START_INTERNAL_BLOCK()
      /**

      This is public only because one cannot add the const qualifier to
      typedef'd pointers. Make private again when ConnectionIteratorOperations
      has been rewritten to not access the vector directly.
      */
      agxPowerLine::ConnectionRefVector& getConnections(agxPowerLine::Side side);
      DOXYGEN_END_INTERNAL_BLOCK()

      /**
      set the constraint that represents this connector in the solver.
      */
      void setConstraint(agxPowerLine::PhysicalDimensionMultiBodyConstraintImplementation* constraint);

      /**
      Create the constraint (implement this to create whatever constraint you need)
      \return the created constraint
      */
      virtual agxPowerLine::PhysicalDimensionMultiBodyConstraintImplementation* createConstraint() = 0;

      /**
      \return the constraint representing this connector in the solver.
      */
      agxPowerLine::PhysicalDimensionMultiBodyConstraintImplementation* getConstraint() const;

      /**
      \return The first elementary constraint in the constraint returned by getConstraint(),
              or nullptr if there is no constraint.
      */
      agxPowerLine::ElementaryPhysicalDimensionConstraint* getElementaryConstraint() const;

      /**
      Find out if a connection is output from this connector
      */
      virtual bool isOutput( Connection* connection ) const;

      /**
      Find out if the given dimension is part of any of the output connections.
      */
      virtual bool isOutput(const agxPowerLine::PhysicalDimension* dimension ) const;

      /**
      Determine if the given dimension is part of any of the input connections of this Connector.
      \param dimension - The dimension to look for.
      \return True if the dimension was found on the input side of this Connector. False otherwise.
      */
      virtual bool isInput(const agxPowerLine::PhysicalDimension* dimension) const;

      /**
      * Set the damping of the constraint.
      */
      void setDamping(agx::Real damping);
      agx::Real getDamping() const;


      agx::Real getRatio() const;
      agx::Real getRatioDerivative() const;

      virtual agx::Real getRatio(agx::Real x) const;
      virtual agx::Real getRatioDerivative(agx::Real x) const;

      /**
      Calculate the compliance and damping according to the connector properties (efficiency, ratio, ..)
      */
      virtual agx::RegularizationParameters::VariableType calculateComplianceAndDamping(const agx::Real /*timeStep*/, agx::Real& /*compliance*/, agx::Real& /*damping*/);


      /**
      Set values of constraints. Must be called after the constraints are created.
      */
      virtual bool postStore( agxStream::StorageStream& str ) const;
      virtual bool postRestore(agxStream::StorageStream& str);

      /**
      Returns the constraint violation for connectors with holonomic constraints.
      */
      virtual agx::Real getViolation() const;

      void setPowerLine(agxPowerLine::PowerLine* powerLine);

      agxPowerLine::PowerLine* getPowerLine() const;

      bool connectDimensions(Unit* unit, agxPowerLine::Side mySide, agxPowerLine::Side unitSide);

      /**
      Connect on the physical dimension level only. Will not update the parent
      Unit's root node and will not spread the power line over the graph.
      */
      bool connect(agxPowerLine::PhysicalDimension* dimension,
                   agxPowerLine::Side connectorSide,
                   agxPowerLine::Side dimensionSide);

      virtual bool connect(agxPowerLine::PhysicalDimension* inputDimension,
                   agxPowerLine::Side inputSide,
                   agxPowerLine::Side outputSide,
                   agxPowerLine::PhysicalDimension* outputDimension);

      bool disconnect(agxPowerLine::PhysicalDimension* dimension,
                      agxPowerLine::Side connectorSide,
                      agxPowerLine::Side dimensionSide);


      bool isConstraintValid() const;

      virtual bool vetoConnect(agxPowerLine::Unit* unit) const;

      /**
      Stores internal data for this segment into stream.
      */
      virtual bool store(agxStream::StorageStream& str) const override;

      /**
      Restores internal data from stream.
      */
      virtual bool restore(agxStream::StorageStream& str) override;

      bool getIgnoreForStoreRestoreStream() const;
      void setIgnoreForStoreRestoreStream(bool ignore);

      AGXSTREAM_DECLARE_ABSTRACT_SERIALIZABLE(agxPowerLine::Connector);
      virtual void store(agxStream::OutputArchive& out) const override;
      virtual void restore(agxStream::InputArchive& in) override;

    protected:

      /**
      set/get the ratio (scaling of load through the constraint).
      */
      void setRatio(agx::Real ratio);
      void setRatioDerivative(agx::Real ratioDerivative);

      bool getUnits(UnitPtrSetVector& units, ConnectionRefVector& connections, const bool startFromCleanUnitSet = true);
      bool getUnits(UnitConstPtrSetVector& units, const ConnectionRefVector& connections, const bool startFromCleanUnitSet = true) const;

      bool hasNonValidInputConnection() const;
      bool hasNonValidOutputConnection() const;

// Swig tries to generate bindings to this protected connect member function
// which leads to compiler errors since the function is protected.
#ifndef SWIG
      friend class agxPowerLine::detail::ConnectionOperations;
      virtual bool connect(
        agxPowerLine::Unit* inputUnit, agxPowerLine::Side inputUnitSide,
        agxPowerLine::Side outputUnitSide, agxPowerLine::Unit* outputUnit);
#endif

      void uninitialize();
      virtual ~Connector();

    protected:
      int m_numPossibleInputs;
      int m_numPossibleOutputs;
      ConnectionRefVector m_inputConnections;
      ConnectionRefVector m_outputConnections;
      DimensionTypeVector m_inputTypes;
      DimensionTypeVector m_outputTypes;

      agx::Real m_damping;

      agx::Real m_ratio;
      agx::Real m_ratioDerivative;
      PhysicalDimensionMultiBodyConstraintImplementation* m_constraint;

      bool m_ignoreMeForStoreRestoreStream;
  };

  typedef agx::Vector<Connector*> ConnectorPtrVector;
  typedef agx::SetVector<Connector*> ConnectorPtrSetVector;

  typedef agx::ref_ptr<Connector> ConnectorRef;
  typedef agx::Vector<ConnectorRef> ConnectorRefVector;
  typedef agx::SetVector<ConnectorRef> ConnectorRefSet;
  typedef agx::Vector<ConnectorObserver> ConnectorObserverVector;

}

#endif // AGXMODEL_POWER_LINE_UTILS_H
