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

#ifndef AGXCONTROL_MEASUREMENT_OPERATIONS_H
#define AGXCONTROL_MEASUREMENT_OPERATIONS_H

#include <agxControl/SensorEvent.h>
#include <agxControl/EventSensor.h>

namespace agxControl
{

  AGX_DECLARE_POINTER_TYPES(DataFilter);
  /**
  Virtual base class of a data filter for data smoothing. Child classes implement different types of filters.
  */
  class AGXPHYSICS_EXPORT DataFilter : public agx::Referenced
  {
  public:
    /// Default constructor
    DataFilter(agx::Real samplingTime);

    /// Returns the a filtered signal given a value supplied to the filter
    virtual void filterValue(agx::Real inValue) = 0;

    /// Returns the current filtered value
    agx::Real getFilteredValue() const;

    /// Set the discrete sampling time of the filter
    virtual bool setSamplingTime(agx::Real samplingTime);

  protected:
    /// Default destructor
    virtual ~DataFilter();

    //////////////////////////////////////////////////////////////////////////
    // Variables
    //////////////////////////////////////////////////////////////////////////
   protected:
     agx::Real m_filteredValue;               // Current filtered value
     agx::Real m_samplingTime;                // Discrete sampling time
  };

  AGX_FORCE_INLINE agx::Real agxControl::DataFilter::getFilteredValue() const { return m_filteredValue; }

  AGX_DECLARE_POINTER_TYPES(ExponentialFilter);
  /**
  Exponential filter that smoothes values in the form: x_filter(t) = a * x_filter(t-1) + ( 1 - a ) * x_raw(t) where
  a = exp( - dt / T ). dt is the discrete sampling rate and T is the period time. https://en.wikipedia.org/wiki/Exponential_smoothing
  */
  class AGXPHYSICS_EXPORT ExponentialFilter : public agxControl::DataFilter
  {
  public:
    /// Constructor for the Exponential filter
    ExponentialFilter( agx::Real samplingTime, agx::Real periodTime );

    /// Filters the value from a given thing
    void filterValue(agx::Real inValue) override;

    /// Set sampling time
    bool setSamplingTime(agx::Real samplingTime) override;

    /// Set the period time
    bool setPeriodTime(agx::Real periodTime);

  protected:
    /// Default Destructor
    virtual ~ExponentialFilter();

    void calculateConstant();

    //////////////////////////////////////////////////////////////////////////
    // Variables
    //////////////////////////////////////////////////////////////////////////
  protected:
    agx::Real m_periodTime;               // Period Time
    agx::Real m_a;                        // Time constant
  };

  /*
  * Base class of operations used by the measurement sensor. Handles filters for data smoothing.
  */
  AGX_DECLARE_POINTER_TYPES(MeasurementSensorOperation);
  class AGXPHYSICS_EXPORT MeasurementSensorOperation : public agxControl::SensorOperation
  {
  public:
    /// Default constructor
    MeasurementSensorOperation(const agx::Name& name = agx::Name());

    virtual void postContactHandling(agxSDK::Simulation *simulation, EventSensor * sensor) override = 0;

    /// Returns the latest calculated value from the operation. Returns the raw value if no filter is attached
    agx::Real getFilteredValue() const;

    /// Returns the latest calculated value from the operation
    agx::Real getRawValue() const;

    /// Set a filter on the operation that handles the values given
    void setFilter(DataFilter * filter);

    /// Gets the current active filter in the operation
    DataFilter* getFilter();

   protected:
     /// Default destructor
     virtual ~MeasurementSensorOperation();

    /// Gets a raw value from the subclass
    virtual agx::Real getValue() const = 0;

    /// Updates the internal filter
    void updateFilter();

    //////////////////////////////////////////////////////////////////////////
    // Variables
    //////////////////////////////////////////////////////////////////////////
  protected:
    agxControl::DataFilterRef m_filter;
  };

  /*
  * This operations calculates the total mass of the bodies _inside_ the sensor geometry.
  */
  AGX_DECLARE_POINTER_TYPES(CalculateTotalMass);
  class AGXPHYSICS_EXPORT CalculateTotalMass : public agxControl::MeasurementSensorOperation
  {
  public:
    /// Default constructor
    CalculateTotalMass(const agx::Name& name = agx::Name());

    void postContactHandling(agxSDK::Simulation *simulation, EventSensor * sensor) override;

    /// Returns the latest calculated mass from the operation
    agx::Real getTotalMass() const;

  protected:
    /// Default destructor
    virtual ~CalculateTotalMass() {}

    virtual agx::Real getValue() const override;

    //////////////////////////////////////////////////////////////////////////
    // Variables
    //////////////////////////////////////////////////////////////////////////
  protected:
    agx::Real m_totalMass;
  };

  /*
  * This operations calculates the total volume of the bodies _inside_ the sensor geometry.
  */
  AGX_DECLARE_POINTER_TYPES(VolumeCalculator);
  class AGXPHYSICS_EXPORT VolumeCalculator : public agxControl::MeasurementSensorOperation
  {
  public:
    /// Default constructor
    VolumeCalculator(const agx::Name& name = agx::Name());

    void postContactHandling(agxSDK::Simulation *simulation, EventSensor * sensor) override;

    /// Returns the latest calculated volume from the operation
    agx::Real getTotalVolume() const;

    /// Returns the latest calculated packing fraction from the operation
    agx::Real getPackingFractionInSensor() const;

  protected:
    /// Default destructor
    virtual ~VolumeCalculator() {}

    virtual agx::Real getValue() const override;

    //////////////////////////////////////////////////////////////////////////
    // Variables
    //////////////////////////////////////////////////////////////////////////
  protected:
    agx::Real m_totalVolume;
    agx::Real m_packingFraction;
    agx::Real m_sensorVolume;
  };

  /*
  * Operation that calculates the average mass flow _inside_ the sensor volume
  */
  AGX_DECLARE_POINTER_TYPES(CalculateTotalMassFlow);
  class AGXPHYSICS_EXPORT CalculateTotalMassFlow : public agxControl::MeasurementSensorOperation
  {
  public:
    /// Default constructor
    CalculateTotalMassFlow(const agx::Name& name = agx::Name());

    void postContactHandling(agxSDK::Simulation *simulation, EventSensor * sensor) override;

    /// Returns the latest calculated mass flow from the operation (kg/s).
    agx::Real getTotalMassFlow() const;

  protected:
    /// Default destructor
    virtual ~CalculateTotalMassFlow() {}

    virtual agx::Real getValue() const override;

    //////////////////////////////////////////////////////////////////////////
    // Variables
    //////////////////////////////////////////////////////////////////////////
  protected:
    agx::Real m_totalMassFlow;
  };

  /*
  * This operations calculates the total mass of the bodies _inside_ the sensor geometry.
  */
  AGX_DECLARE_POINTER_TYPES(CalculateTotalRigidBodyDEMMass);
  class AGXPHYSICS_EXPORT CalculateTotalRigidBodyDEMMass : public agxControl::MeasurementSensorOperation
  {
  public:
    /// Default constructor
    CalculateTotalRigidBodyDEMMass(const agx::Name& name = agx::Name());

    void postContactHandling(agxSDK::Simulation *simulation, EventSensor * sensor) override;

    /// Returns the latest calculated mass from the operation
    agx::Real getTotalMass() const;

  protected:
    /// Default destructor
    virtual ~CalculateTotalRigidBodyDEMMass() {}

    virtual agx::Real getValue() const override;

    //////////////////////////////////////////////////////////////////////////
    // Variables
    //////////////////////////////////////////////////////////////////////////
  protected:
    agx::Real m_totalMass;
  };

  /*
  * Operation that calculates the average mass flow _inside_ the sensor volume
  */
  AGX_DECLARE_POINTER_TYPES(CalculateTotalRigidBodyDEMMassFlow);
  class AGXPHYSICS_EXPORT CalculateTotalRigidBodyDEMMassFlow : public agxControl::MeasurementSensorOperation
  {
  public:
    /// Default constructor
    CalculateTotalRigidBodyDEMMassFlow(const agx::Name& name = agx::Name());

    void postContactHandling(agxSDK::Simulation *simulation, EventSensor * sensor) override;

    /// Returns the latest calculated mass flow from the operation (kg/s).
    agx::Real getTotalMassFlow() const;

  protected:
    /// Default destructor
    virtual ~CalculateTotalRigidBodyDEMMassFlow() {}

    virtual agx::Real getValue() const override;

    //////////////////////////////////////////////////////////////////////////
    // Variables
    //////////////////////////////////////////////////////////////////////////
  protected:
    agx::Real m_totalMassFlow;
  };


  /*
  * This operations calculates the total volume of the bodies _inside_ the sensor geometry.
  */
  AGX_DECLARE_POINTER_TYPES(RigidBodyDEMVolumeCalculator);
  class AGXPHYSICS_EXPORT RigidBodyDEMVolumeCalculator : public agxControl::MeasurementSensorOperation
  {
  public:
    /// Default constructor
    RigidBodyDEMVolumeCalculator(const agx::Name& name = agx::Name());

    void postContactHandling(agxSDK::Simulation *simulation, EventSensor * sensor) override;

    /// Returns the latest calculated volume from the operation
    agx::Real getTotalVolume() const;

    /// Returns the latest calculated packing fraction from the operation
    agx::Real getPackingFractionInSensor() const;

  protected:
    /// Default destructor
    virtual ~RigidBodyDEMVolumeCalculator() {}

    virtual agx::Real getValue() const override;

    //////////////////////////////////////////////////////////////////////////
    // Variables
    //////////////////////////////////////////////////////////////////////////
  protected:
    agx::Real m_totalVolume;
    agx::Real m_packingFraction;
    agx::Real m_sensorVolume;
  };
}

#endif
