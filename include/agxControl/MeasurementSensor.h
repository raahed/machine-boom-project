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

#ifndef AGXCONTROL_MEASUREMENTSENSOR_H
#define AGXCONTROL_MEASUREMENTSENSOR_H

#include <agx/agx.h>
#include <agx/BitState.h>
#include <agxControl/EventSensor.h>
#include <agxControl/MeasurementOperations.h>

#include <agx/QuadraticProbingHashTable.h>

namespace agxControl
{
  typedef agx::QuadraticProbingHashTable<agx::Name, agxControl::MeasurementSensorOperation*> OperationMap;
}

namespace agxControl
{
  /**
  * This class extends the EventSensor class by providing extra functionality for extracting different data from the bodies inside
  * the sensor geometry. Has internal operations that calculate properties such as total mass and average mass flow inside the sensor volume.
  **/
  AGX_DECLARE_POINTER_TYPES( MeasurementSensor );
  AGX_DECLARE_VECTOR_TYPES( MeasurementSensor );
  class CALLABLE AGXPHYSICS_EXPORT MeasurementSensor : public EventSensor
  {
  public:

    /**
    Enumeration specifying what should be recorded in a measurement sensor, expressed as a bit field.
    */
    enum MeasurementState
    {
      MASS     = 1,
      MASSFLOW = 2,
      VOLUME   = 4,
      ALL = MASS | MASSFLOW | VOLUME
    };

    typedef agx::BitState<MeasurementState, agx::Int32> MeasurementBitState;

    /// Default constructor
    MeasurementSensor();

    /**
    Constructor that takes the priority of the internal contact sensors.
    \param priority - Priority of the internal contact listeners.
    */
    MeasurementSensor(int priority);

    /**
    Constructor that takes the geometry to couple the sensor too.
    \param geom The geometry that the sensor will register all the contacts from.
    */
    MeasurementSensor(agxCollide::Geometry * geom);

    /**
    Constructor that takes the geometry to couple the sensor too. All collision on the geometry will be registered by the sensor.
    \param geom The geometry that the sensor will register all the contacts from.
    \param priority - Priority of the internal contact listeners.
    */
    MeasurementSensor(agxCollide::Geometry * geom, int priority);

    /**
    Sets the MeasurementState which determines what should be recorded in the MeasurementSensor.
    \param state - MeasurementState specifying what the sensor should record.
    */
    void setMeasurementState(MeasurementState state);

    /**
    \param simulation - simulation the MeasurementSensor is part of.
    \param name - name of the MeasurementSensors.
    \return the MeasurementSensor with the given name.
    */
    static MeasurementSensor* find( const agxSDK::Simulation* simulation, const agx::Name& name );

    /**
    Find all MeasurementSensors with given name.
    \param simulation - simulation the MeasurementSensor is part of.
    \param name - name of the MeasurementSensor.
    \return vector of MeasurementSensors.
    */
    static MeasurementSensorPtrVector findAll( const agxSDK::Simulation* simulation, const agx::Name& name );

    /**
    Finds all MeasurementSensors in the given simulation.
    \param simulation - simulation with MeasurementSensors.
    \return vector of MeasurementSensors.
    */
    static MeasurementSensorPtrVector findAll( const agxSDK::Simulation* simulation );

    /**
    Returns the MeasurementState that determines what should be recorded in the MeasurementSensor.
    */
    MeasurementState getMeasurementState() const;

    /// Returns the total particle mass flow inside the sensor volume
    agx::Real getTotalParticleMassFlow() const;

    /// Returns the total particle mass inside the sensor volume
    agx::Real getTotalParticleMass() const;

    /// Returns the total particle volume inside the sensor volume
    agx::Real getTotalParticleVolume() const;

    /// Returns the total particle volume inside the sensor volume
    agx::Real getParticlePackingFraction() const;

    /// Returns the total RigidBody mass inside the sensor volume
    agx::Real getTotalRigidBodyMass() const;

    /// Returns the total RigidBody mass flow inside the sensor volume
    agx::Real getTotalRigidBodyMassFlow() const;

    /// Returns the total RigidBody volume inside the sensor volume
    agx::Real getTotalRigidBodyVolume() const;

    /// Returns the total RigidBody volume inside the sensor volume
    agx::Real getTotalRigidBodyPackingFraction() const;

    /// Set if the measurement sensor should use exponential filters to smooth data from the measurements.
    void setEnableDataFilter(bool enable);

    /// Get if the measurement sensor should use exponential filters to smooth data from the measurements.
    bool getEnableDataFilter() const;

    /// Set the period time for the internal exponential filters.
    void setFilterPeriodTime(agx::Real periodTime);

    /// Returns the period time for the internal exponential filters.
    agx::Real getFilterPeriodTime() const;

    /// Set the discrete sampling time for the internal exponential filters.
    void setFilterSamplingTime(agx::Real samplingTime);

    /// Get the discrete sampling time for the internal exponential filters.
    agx::Real getFilterSamplingTime() const;

    AGXSTREAM_DECLARE_SERIALIZABLE(agxControl::MeasurementSensor);

  protected:
    /// Default destructor
    virtual ~MeasurementSensor();

    /// Initializes and adds the internal operations to the simulation that the sensor is added to.
    void initOperations();

    /// Initializes the value filters in the measurement operations.
    void initFilters();

    void updateMeasurementOperationsFromBitState();

    /// Returns a value from a measurement operations with specified name identifier. Will return a filtered version if useFiltered is specified.
    agx::Real getValue(const agx::Name& operationIdentifier, bool useFiltered) const;

  private:
    /// Declare copy constructor and equals operator private since we don't implement them.
    MeasurementSensor(const MeasurementSensor&);
    MeasurementSensor& operator=(const MeasurementSensor&);

    //////////////////////////////////////////////////////////////////////////
    // Variables
    //////////////////////////////////////////////////////////////////////////
  protected:

    MeasurementBitState m_measurementState;

    // Direct reference to the measurement operations
    agxControl::CalculateTotalMassRef                 m_totalMassOperation;
    agxControl::CalculateTotalMassFlowRef             m_totalMassFlowOperation;
    agxControl::VolumeCalculatorRef                   m_totalVolumeOperation;
    agxControl::CalculateTotalRigidBodyDEMMassRef     m_totalRigidBodyMass;
    agxControl::CalculateTotalRigidBodyDEMMassFlowRef m_totalRigidBodyMassFlow;
    agxControl::RigidBodyDEMVolumeCalculatorRef       m_totalRigidBodyVolume;

    // MeasurementSensor operation map
    OperationMap m_operationMap;
    bool         m_useDataFilters;
    agx::Real    m_filterPeriodTime;
    agx::Real    m_filterSamplingTime;
  };

  AGX_FORCE_INLINE bool MeasurementSensor::getEnableDataFilter() const { return m_useDataFilters; }

  AGX_FORCE_INLINE void MeasurementSensor::setEnableDataFilter(bool enable) { m_useDataFilters = enable; }

  inline MeasurementSensor::MeasurementState operator|(MeasurementSensor::MeasurementState lhs, MeasurementSensor::MeasurementState rhs)
  {
    return static_cast<MeasurementSensor::MeasurementState>((agx::Int32)lhs | (agx::Int32)rhs);
  }
}

#endif

