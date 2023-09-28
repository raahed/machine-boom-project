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

#ifndef AGXMODEL_HIGH_LEVEL_DRIVE_TRAIN_H
#define AGXMODEL_HIGH_LEVEL_DRIVE_TRAIN_H

#include <agxDriveTrain/deprecated/DriveTrainComponents.h>
#include <agxDriveTrain/Differential.h>
#include <list>
#include <vector>

namespace agxDriveTrain
{

  namespace deprecated
  {

    /**
    Implementation of a torque converter (TQ)
    The characteristics of a real TQ is captured in two look up tables.

    Lookup table 1:
    The nu /efficiency table - a lookup for TQ efficiency given rotational speed ratio between turbine and pump (nu).

    To set the efficiency table use:
    void setEfficiencyTable(agx::Vector<std::pair<agx::Real, agx::Real> > const& nu_efficiency);

    The efficiency table is directly correlated to the torque multiplier of TQ.
    To set the torque multiplication use:
    void setMuTable(agx::Vector<std::pair<agx::Real, agx::Real> > const& nu_mus);


    Lookup table 2:
    The maximum possible torque delivery given nu from and to the unit connected to the TQ pump (usually the engine).

    To set the table use:
    void setPumpTorqueTable(agx::Vector<std::pair<agx::Real, agx::Real> > const& nu_torques);

    The torque delivery limit is dependent of RPM of the pump. The torque is scaled using T = T_ref*(rpm/rpm_ref)^2;
    The reference rpm is set using:
    void setPumpTorqueReferenceRPM( agx::Real rpm );

    Both functions 'setPumpTorqueTable' and 'setPumpTorqueReferenceRPM' are implemented in the parent class "agxModel::TorqueConverterTable".

    */
    class AGXMODEL_EXPORT HighLevelTorqueConverter : public agxDriveTrain::deprecated::TorqueConverterTable
    {
    public:
      /**
      The only constructor.
      */
      HighLevelTorqueConverter( );

      /**
      \returns the torque delivered by the pump
      */
      agx::Real getPumpTorque( ) const;

      /**
      \returns the rpm of the pump
      */
      agx::Real getPumpRPM( ) const;

      /**
      \returns the torque delivered by the turbine.
      */
      agx::Real getTurbineTorque( ) const;

      /**
      \returns the rpm of the turbine.
      */
      agx::Real getTurbineRPM( ) const;

      /**
      \returns the current torque multiplication.
      */
      agx::Real getMu( ) const;

      /**
      \returns the current speed ratio between the turbine and the pump.
      */
      agx::Real getNu( ) const;

      /**
      Stores internal data into stream.
      */
      virtual bool store( agxStream::StorageStream& str ) const override;

      /**
      Restores internal data from stream.
      */
      virtual bool restore( agxStream::StorageStream& str ) override;

      AGXSTREAM_DECLARE_SERIALIZABLE( agxDriveTrain::deprecated::HighLevelTorqueConverter );
    protected:
      virtual ~HighLevelTorqueConverter( );
    };
    typedef agx::ref_ptr< HighLevelTorqueConverter > HighLevelTorqueConverterRef;


    /**
    The gear box can have an arbitrary number of gears.
    A gear is added by adding a gear ratio to the gear box.
    The gear will be sorted among the already existing gears.

    A negative gear ratio is considered to be a reverse gear.
    For example: by setting the gear index to no. -1 will give the reverse gear with the lowest gear ratio.
    If there is no reverse gear, the setGear function will return false.

    The gear box will always have a neutral gear by default. (The neutral always have gear index 0 (zero) )
    */
    class AGXMODEL_EXPORT HighLevelGearBox : public agxDriveTrain::deprecated::GearBox
    {
    public:
      /**
      The only constructor
      */
      HighLevelGearBox( );

      /**
      \returns current rpm of the output shaft.
      */
      agx::Real getRPM( ) const;

      /**
      \returns current rpm for the incoming shaft.
      */
      agx::Real getIncomingRPM( ) const;

      /**
      Get current load torque.
      */
      agx::Real getTorque( ) const;

      /**
      Stores internal data into stream.
      */
      virtual bool store( agxStream::StorageStream& str ) const override;

      /**
      Restores internal data from stream.
      */
      virtual bool restore( agxStream::StorageStream& str ) override;

      AGXSTREAM_DECLARE_SERIALIZABLE( agxDriveTrain::deprecated::HighLevelGearBox );
    protected:
      virtual ~HighLevelGearBox( );

    };
    typedef agx::ref_ptr<HighLevelGearBox> HighLevelGearBoxRef;

    /**
    Implementation of a differential.
    The input torque will be distributed evenly on all output shafts.
    */
    class AGXMODEL_EXPORT HighLevelDifferential : public agxDriveTrain::Differential
    {
    public:
      /**
      The only constructor.
      */
      HighLevelDifferential( );

      /**
      \returns input RPM.
      */
      agx::Real getDriveShaftRPM( ) const;

      /**
      \param outputIndex - index of output shaft
      \returns RPM of connected output
      */
      agx::Real getOutputShaftRPM( size_t outputIndex ) const;

      const agxPowerLine::PhysicalDimension* getInputDimension( ) const;

      const agxPowerLine::PhysicalDimension* getOutputDimension( size_t index ) const;


      /**
      \returns input torque.
      */
      agx::Real getDriveShaftTorque( ) const;

      /**
      \param outputIndex - index of output shaft.
      \returns output torque for a given output shaft.
      */
      agx::Real getOutputShaftTorque( size_t outputIndex ) const;

      /**
      \returns torque of first connected output.
      */
      agx::Real getLeftTorque( ) const;

      /**
      \returns torque of second connected output.
      */
      agx::Real getRightTorque( ) const;

      /**
      Stores internal data into stream.
      */
      virtual bool store( agxStream::StorageStream& str ) const override;

      /**
      Restores internal data from stream.
      */
      virtual bool restore( agxStream::StorageStream& str ) override;

      AGXSTREAM_DECLARE_SERIALIZABLE( agxDriveTrain::deprecated::HighLevelDifferential );
    protected:
      virtual ~HighLevelDifferential( );
    };
    typedef agx::ref_ptr<HighLevelDifferential> HighLevelDifferentialRef;
  }
}

#endif // AGXMODEL_HIGH_LEVEL_DRIVE_TRAIN_H
