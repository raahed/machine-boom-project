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

#ifndef AGXPOWERLINE_POWER_LINE_UTILS_H
#define AGXPOWERLINE_POWER_LINE_UTILS_H

#include <agxModel/export.h>

#include <agxStream/Serializable.h>
#include <agx/List.h>
#include <agx/Vector.h>
#include <agxUtil/agxUtil.h>
#include <agxPowerLine/PhysicalDimension.h>
#include <agx/LookupTables.h>


namespace agxPowerLine
{
  namespace detail
  {
    class ConnectionOperations;
  }

}


namespace agxPowerLine
{
  class Unit;
  class PowerLine;
  class PhysicalDimensionMultiBodyConstraintImplementation;

  typedef agx::ref_ptr<Unit> UnitRef;
  typedef agx::observer_ptr<Unit> UnitObserver;
  typedef agx::HashVector<UnitRef, bool> UnitRefBoolHashVector;

  AGXMODEL_EXPORT agxPowerLine::PhysicalDimension::Type getDimensionType(
      const std::string& dimensionName);

  /**
  Return a value of y given any value of x

  The variable look up function can be set to calculate any wanted x value.

  The look up table return tabulated (and interpolated) values that
  can be modified by defining a result scaler function.
  Scaling the result is convenient when for example:
  1. You have an torqueScaling look up table given an variable \f$ \nu \f$.
  2. You have a torque that is found directly from an rpm look up table.
  3. The result torque is automatically scaled with your result scaling function depending on \f$ \nu \f$.

  Default variable look up function is just the gradient of the 1D physical dimension.
  Default result scaler function is not scaling at all.
  */
  class AGXMODEL_EXPORT LookupTable : virtual public agx::LookupTable // virtual inheritance due to diamond
  {
    public:
      /**
      Create a lookup table
      \param dimension - the dimension which will be used to calculate the lookup variable, to find the value.
      */
      LookupTable(agxPowerLine::PhysicalDimension* dimension);

      /**
      change dependent physical dimension
      */
      void setDimension( agxPowerLine::PhysicalDimension* dimension );

      /**
      Lookup the current value (given the dimension set.)
      */
      agx::Real lookupCurrent() const;

      /**
      Implement this in any child class to define the lookup variable.
      Default look up variable is the gradient of the dimension.
      */
      virtual agx::Real variableLookupFunction() const;

      /**
      Stores internal data into stream.
      */
      virtual bool store(agxStream::StorageStream& str) const;

      /**
      Restores internal data from stream.
      */
      virtual bool restore(agxStream::StorageStream& str);


      virtual void store( agxStream::OutputArchive& out ) const override;
      virtual void restore( agxStream::InputArchive& in ) override;

      AGXSTREAM_DECLARE_ABSTRACT_SERIALIZABLE(agxPowerLine::LookupTable);
    protected:
      LookupTable();
      virtual ~LookupTable();
      agx::observer_ptr<PhysicalDimension> m_dimension;
  };


  /**
  Linear interpolation between the inserted variable/value pairs.
  Values looked up outside the set range will be interpolated using the tangent at that end of the table.
  */
  class AGXMODEL_EXPORT LinearLookupTable : public LookupTable, public agx::LinearLookupTable
  {
    public:
      /**
      Constructor.
      */
      LinearLookupTable(agxPowerLine::PhysicalDimension* dimension);

      /**
      Lookup a specific value
      */
      virtual agx::Real lookup( const agx::Real x ) const override;

      /**
      Stores internal data into stream.
      */
      virtual bool store(agxStream::StorageStream& str) const override;

      /**
      Restores internal data from stream.
      */
      virtual bool restore(agxStream::StorageStream& str) override;

      AGXSTREAM_DECLARE_SERIALIZABLE(agxPowerLine::LinearLookupTable);
    protected:
      LinearLookupTable();
      virtual ~LinearLookupTable();
    };
    typedef agx::ref_ptr<LinearLookupTable> LinearLookupTableRef;


    /**
    A lookup table used for power time integral (torque for rotational power, force for translational power)
    */
    class AGXMODEL_EXPORT PowerTimeIntegralLookupTable : public agxPowerLine::LinearLookupTable
    {
    public:
      /**
      The time integral is the one of the power lookup table in the power generator.
      \param dimension - the physical dimension that will be exposed to the power generation
      \param generator - the power generator that produces power, and will be the time derivative of this time integral.
      */
      PowerTimeIntegralLookupTable( agxPowerLine::PhysicalDimension* dimension, agxPowerLine::PowerGenerator* generator );

      /**
      Stores internal data into stream.
      */
      virtual bool store(agxStream::StorageStream& str) const override;

      /**
      Restores internal data from stream.
      */
      virtual bool restore(agxStream::StorageStream& str) override;

      AGXSTREAM_DECLARE_SERIALIZABLE(agxPowerLine::PowerTimeIntegralLookupTable);

    protected:
      virtual ~PowerTimeIntegralLookupTable();
      PowerTimeIntegralLookupTable();
      PowerGeneratorObserver m_generator;

  };
  typedef agx::ref_ptr<PowerTimeIntegralLookupTable> PowerTimeIntegralLookupTableRef;


  /**
  Will generate power by adding a load to a physical dimension
  */
  class AGXMODEL_EXPORT PowerGenerator : public agx::Referenced, public agxStream::Serializable
  {
    public:
      /**
      Create a power generator
      \param dimension - the physical dimension that will be exposed for the generated load.
      */
      PowerGenerator(agxPowerLine::PhysicalDimension* dimension);

      /**
      Set the time integral table.
      */
      void setPowerTimeIntegralLookupTable( agxPowerLine::PowerTimeIntegralLookupTable* table );

      /**
      \return The time integral table.
      */
      agxPowerLine::PowerTimeIntegralLookupTable* getPowerTimeIntegralLookupTable() const;

      /**
      return how big load that is added to the simulation through the physical dimension at the moment
      */
      virtual agx::Real getCurrentPowerTimeIntegral() const;

      /**
      Implement this to apply power each timestep. Do not call this.
      Will be called from the power line pre() function
      */
      virtual void applyPower() = 0;

      /**
      Implement this to scale lookups accordingly
      */
      virtual agx::Real variableLookupFunction( ) const = 0;

      /**
      Implement this to scale lookups accordingly
      */
      virtual agx::Real resultScalerFunction( ) const = 0;

      /**
      clear all reference pointers
      */
      void clear();

      /**
      Stores internal data into stream.
      */
      virtual bool store(agxStream::StorageStream& str) const;

      /**
      Restores internal data from stream.
      */
      virtual bool restore(agxStream::StorageStream& str);

      virtual void store( agxStream::OutputArchive& out ) const override;
      virtual void restore( agxStream::InputArchive& in ) override;

      AGXSTREAM_DECLARE_ABSTRACT_SERIALIZABLE(agxPowerLine::PowerGenerator);

    protected:
      PowerGenerator();
      virtual ~PowerGenerator();

      PowerTimeIntegralLookupTableRef m_timeIntegralLookupTable;
  };


  /*
  A Connection can link one physical dimension to another.
  */
  class AGXMODEL_EXPORT Connection : public agx::Referenced, public agxStream::Serializable
  {
    public:
      /**
      Create a connection in or out of a physical dimension.
      \param dimension - the dimension that is connecting
      */
      Connection(agxPowerLine::PhysicalDimension* dimension, agxPowerLine::Connector* connector);

      /**
      \return True if the connection has both a connector and a dimension.
      */
      bool isValid() const;

      /**
      \return a pointer to its connector.
      */
      const agxPowerLine::Connector* getConnector() const;

      /**
      \return a pointer to its connector.
      */
      agxPowerLine::Connector* getConnector();

      /**
      \deprecated
      set pointer to a connector
      */
      bool setConnector( agxPowerLine::Connector* connector );

      /**
      \return pointer to its dimension
      */
      const agxPowerLine::PhysicalDimension* getDimension() const;

      /**
      \return pointer to its dimension
      */
      agxPowerLine::PhysicalDimension* getDimension();

      /**
      clear all reference pointers
      */
      virtual void clear();

      /**
      Stores internal data into stream.
      */
      virtual bool store(agxStream::StorageStream& str) const;

      /**
      Restores internal data from stream.
      */
      virtual bool restore(agxStream::StorageStream& str);

      //Will use the efficiency of the output connection
      AGXSTREAM_DECLARE_SERIALIZABLE(agxPowerLine::Connection);

    protected:
      Connection();
      virtual ~Connection();

    protected:
      Connector* m_connector;
      PhysicalDimension* m_dimension;
  };

}

#endif // AGXMODEL_POWER_LINE_UTILS_H
