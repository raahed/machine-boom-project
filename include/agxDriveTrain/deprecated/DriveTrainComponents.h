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

#pragma once

#include <agxPowerLine/PowerLine.h>
#include <agxPowerLine/RotationalUnit.h>
#include <agxDriveTrain/deprecated/TorqueConverter.h>
#include <agxDriveTrain/Shaft.h>
#include <agxDriveTrain/Gear.h>
#include <agxDriveTrain/GearBox.h>

#include <agx/Hinge.h>
#include <agx/Constraint.h>

namespace agxDriveTrain
{
  class Clutch;

  namespace deprecated
  {

    /**
     * A DriveTrainComponent is a Shaft that comes with a Connector and an input shaft.
     * \deprecated Use the individual components instead of these generic pre-packaged ones.
     */
    class AGXMODEL_EXPORT DriveTrainComponent : public agxDriveTrain::Shaft
    {
    public:
      DriveTrainComponent( agxPowerLine::Connector* internalConnector );

      /**
      \return The dimension of the given types that accepts connections on the
      given side. Can be a PhysicalDimension owned by an internal Unit.
      */
      virtual agxPowerLine::DimensionAndSide getConnectableDimension(
        agxPowerLine::PhysicalDimension::Type type, agxPowerLine::Side side ) override;

      virtual void getConnectableDimensionTypes(
        agxPowerLine::PhysicalDimension::TypeVector& types, agxPowerLine::Side side ) const override;


      const agxDriveTrain::Shaft* getOutputShaft( ) const;
      const agxDriveTrain::Shaft* getInputShaft( ) const;

      virtual void setInertia( agx::Real inertia ) override;
      virtual agx::Real getInertia( ) const override;

      virtual agxPowerLine::Connector* getConnector( );

      agxPowerLine::RotationalDimension* getRotationalDimension( agxPowerLine::Side side );
      const agxPowerLine::RotationalDimension* getRotationalDimension( agxPowerLine::Side side ) const;

      using RotationalUnit::getRotationalDimension;


      virtual const agxPowerLine::Connector* getConnector( ) const;

      /**
      Stores internal data into stream.
      */
      virtual bool store( agxStream::StorageStream& str ) const override;

      /**
      Restores internal data from stream.
      */
      virtual bool restore( agxStream::StorageStream& str ) override;

      virtual void getConnectorsRecursive( agxPowerLine::ConnectorPtrVector& connectors ) override;
      virtual void getUnitsRecursive( agxPowerLine::UnitPtrVector& units ) override;

      void store( agxStream::OutputArchive& out ) const override;
      void restore( agxStream::InputArchive& in ) override;

      AGXSTREAM_DECLARE_ABSTRACT_SERIALIZABLE( agxDriveTrain::deprecated::DriveTrainComponent );

    protected:
      void connectInternalConnector( );
      bool replaceConnector( agxPowerLine::Connector* connector );

      virtual ~DriveTrainComponent( );
      agxDriveTrain::ShaftRef m_inputShaft;
      agxPowerLine::ConnectorRef m_connector;

    };

    /**
    Base Torque converter for an automatic transmission
    */
    class AGXMODEL_EXPORT TorqueConverterTable : public agxDriveTrain::deprecated::DriveTrainComponent
    {
    public:
      /**
      Create the converter.
      */
      TorqueConverterTable( );

      agx::Real getWantedEfficiency( ) const;

      agx::Real getCurrentEfficiency( ) const;

      /**
      When the efficiency goes to zero, the torque multiplication goes to infinity
      set/get the largest possible multiplication.
      */
      void setMaxMultiplication( agx::Real maxMultiplication );

      /**
      \returns the maximum multiplication of the torque converter.
      */
      agx::Real getMaxMultiplication( ) const;

      /**
      Set the pump torque table.
      At a given ratio between pump and turbine rpms (nu) the pump load will be  possible to calculate from tabulated value at a fixed pump rpm (usually 1000rpm).
      */
      void setPumpTorqueTable( agx::Vector<std::pair<agx::Real, agx::Real> > const& nu_torques );

      /**
      Set table mapping velocity ratio to torque multiplication.
      */
      void setMuTable( agx::Vector<std::pair<agx::Real, agx::Real> > const& nu_to_mu );

      /**
      Return the torque multiplication given the current state
      */
      agx::Real getCurrentMultiplication( ) const;

      /**
      \returns pointer to the pump torque table.
      */
      const agxPowerLine::LinearLookupTable* getPumpTorqueTable( ) const;
      agxPowerLine::LinearLookupTable* getPumpTorqueTable( );

      /**
      Return pointer to table mapping angular velocity ratio to torque multiplication
      */
      const agxPowerLine::LinearLookupTable* getMuTable( ) const;

      /**
      Clear the table mapping angular velocity ratio to torque multiplication
      */
      void clearMuTable( );

      /**
      Insert value in the table mapping angular velocity ratio to torque multiplication
      */
      void insertNuMuLookupValue( agx::Real nu, agx::Real mu );

      /**
      Set the pump torque reference rpm for the table.
      The pump torque values are set given a reference rpm. For other rpm's the pump load torque is found from the following formula:
      T_rpm(nu) = T_referenceRpm(nu) * (rpm/referenceRPm)^2
      */
      void setPumpTorqueReferenceRPM( agx::Real rpm );

      /**
      \returns the reference RPM used for the pump torque table.
      */
      agx::Real getPumpTorqueReferenceRPM( ) const;

      //    virtual void createConnector();


      /**
      Stores internal data into stream.
      */
      virtual bool store( agxStream::StorageStream& str ) const override;

      /**
      Restores internal data from stream.
      */
      virtual bool restore( agxStream::StorageStream& str ) override;

      AGXSTREAM_DECLARE_SERIALIZABLE( agxDriveTrain::deprecated::TorqueConverterTable );
    protected:

      virtual ~TorqueConverterTable( );

      agxDriveTrain::deprecated::TorqueConverterRef m_torqueConverterConnector;

    };
    typedef agx::ref_ptr<TorqueConverterTable> TorqueConverterTableRef;


    /**
    The gear box can have an arbitrary number of gears.
    A gear is added by adding a gear ratio to the gear box.
    The gear will be sorted among the already existing gears.

    A negative gear ratio is considered to be a reverse gear.
    By setting gear to no. -1, will give the reverse gear with the lowest gear ratio.
    If there is no reverse gear, the setGear function will return false.

    The gear box will always have a neutral gear by default.
    */
    class AGXMODEL_EXPORT FixedGear : public agxDriveTrain::deprecated::DriveTrainComponent
    {
    public:
      /**
      Create a gear box.
      */
      FixedGear( );

      explicit FixedGear( agx::Real ratio );

      /**
      Set the fixed gear ratio
      */
      void setGearRatio( agx::Real ratio );



      agx::Real getGearRatio( ) const;

      // Removed because we're now passing the correct connector to the DriveTrain-
      // Component constructor.
      //      void createConnector() override;

      virtual bool preUpdate( agx::Real timeStep ) override;

      /**
      Stores internal data into stream.
      */
      virtual bool store( agxStream::StorageStream& str ) const override;

      /**
      Restores internal data from stream.
      */
      virtual bool restore( agxStream::StorageStream& str ) override;

      AGXSTREAM_DECLARE_SERIALIZABLE( agxDriveTrain::deprecated::FixedGear );
    protected:
      virtual ~FixedGear( );
      agxDriveTrain::GearRef m_gearConnector;

    };
    typedef agx::ref_ptr<FixedGear> FixedGearRef;


    /**
    The gear box can have an arbitrary number of gears.
    A gear is added by adding a gear ratio to the gear box.
    The gear will be sorted among the already existing gears.

    A negative gear ratio is considered to be a reverse gear.
    By setting gear to no. -1, will give the reverse gear with the lowest gear ratio.
    If there is no reverse gear, the setGear function will return false.

    The gear box will always have a neutral gear by default.
    */


    class AGXMODEL_EXPORT GearBox : public agxDriveTrain::deprecated::DriveTrainComponent
    {
    public:
      /**
      Create a gear box.
      */
      GearBox( );

      /**
      Create a gear box.
      \param gears - a vector with gear ratios
      */
      GearBox( const agx::RealVector& gears );

      /**
      Set the gear ratio for gear indexed like -2, -1, 0, 1, 2, 3, 4 (negative is reverse gears, 0 is neutral, positive if forward gears)
      */
      void setGearRatios( const agx::RealVector& gear_ratios );

      /**
      change gears
      */
      bool gearUp( );
      bool gearDown( );
      bool setGear( int gearIndex );

      /**
      \returns the index of the current gear.
      */
      int getGear( ) const;

      /**
      \returns the ratio of a gear with index gearindex
      */
      agx::Real getGearRatio( int gearIndex ) const;

      agx::Real getCurrentGearRatio( ) const;

      const agxDriveTrain::GearBox* getGearBox( ) const;

      /**
      Functions called by a static function. Do NOT use.
      */
      virtual bool preUpdate( agx::Real timeStep ) override;

      /**
      Stores internal data into stream.
      */
      virtual bool store( agxStream::StorageStream& str ) const override;

      /**
      Restores internal data from stream.
      */
      virtual bool restore( agxStream::StorageStream& str ) override;

      AGXSTREAM_DECLARE_SERIALIZABLE( agxDriveTrain::deprecated::GearBox );
    protected:
      virtual ~GearBox( );
      void createBox( const agx::RealVector& gears );
      agxDriveTrain::GearBoxRef m_gearBoxConnector;
    };
    typedef agx::ref_ptr<GearBox> GearBoxRef;



    class AGXMODEL_EXPORT Clutch : public agxDriveTrain::deprecated::DriveTrainComponent
    {
    public:
      /**
      Create the clutch.
      */
      Clutch( );

      /**
      Specify the efficiency of the clutch:
      0 is freely coupled
      1 is tightly coupled
      */
      void setEfficiency( agx::Real efficiency );

      /// \return the current efficiency of the clutch
      agx::Real getEfficiency( ) const;


      /**
      The torque transfer constant (ttc) determines how fast the possible torque transfer goes to infinity.
      It will given the equation:
      LimitTorque = ttc / (1 - eff) - ttc

      ttc have the unit of torque / radian/s (input velocity), and will also determine the exact torque limit at 50% efficiency (eff).

      It will also satisfy the wanted efficiency (eff).

      */
      void setTorqueTransferConstant( agx::Real constant );
      agx::Real getTorqueTransferConstant( ) const;

      /**
      Setting the forced compliance to a non negative overrides the Clutch internal compliance and torque limit calculations.
      the forced compliance is set to negative valu by default - disabled.
      */
      void setForcedCompliance( agx::Real compliance );
      agx::Real getForcedCompliance( ) const;


      agxDriveTrain::Clutch* getClutchConnector( );
      const agxDriveTrain::Clutch* getClutchConnector( ) const;

      /*
      Function that have to be public since it is called by static functions. Do NOT use.
      */

      /**
      Stores internal data into stream.
      */
      virtual bool store( agxStream::StorageStream& str ) const override;

      /**
      Restores internal data from stream.
      */
      virtual bool restore( agxStream::StorageStream& str ) override;

      AGXSTREAM_DECLARE_SERIALIZABLE( agxDriveTrain::deprecated::Clutch );

    protected:
      virtual ~Clutch( );

    protected:
      // The reference is owned by DriveTrainComponent, a base class of Clutch.
      agxDriveTrain::Clutch* m_clutchConnector;
    };

    typedef agx::ref_ptr<Clutch> ClutchRef;
  }

}


