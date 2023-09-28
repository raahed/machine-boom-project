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
#include <agxPowerLine/PowerLineConstraints.h>
#include <agxDriveTrain/Gear.h>

namespace agxDriveTrain
{

  namespace deprecated
  {

    /**
    Lookup table where the lookup variable depend on the velocity ratio between in and output.
    */
    class AGXMODEL_EXPORT TorqueConverterSpeedRatioLookup : public agxPowerLine::LinearLookupTable
    {
    public:
      /**
      Create a lookup table where the lookup  variable is speed ratio between input and output.
      */
      TorqueConverterSpeedRatioLookup();

      /**
      This virtual function is called to find the current value of the lookup variable.
      */
      virtual agx::Real variableLookupFunction() const override;

#ifndef SWIG
      /**
      Stores internal data into stream.
      */
      virtual bool store(agxStream::StorageStream& str) const override;

      /**
      Restores internal data from stream.
      */
      virtual bool restore(agxStream::StorageStream& str) override;

      AGXSTREAM_DECLARE_SERIALIZABLE(agxDriveTrain::deprecated::TorqueConverterSpeedRatioLookup);
#endif

    protected:
      virtual ~TorqueConverterSpeedRatioLookup();
    };

    typedef agx::ref_ptr<TorqueConverterSpeedRatioLookup> TorqueConverterSpeedRatioLookupRef;

    /**
    Base Torque converter for an automatic transmission
    */
    class AGXMODEL_EXPORT TorqueConverter : public agxDriveTrain::SlipGear
    {
    public:
      /**
      Create the converter.
      */
      TorqueConverter();

      /**
      \returns the current speed ratio between the turbine and the pump.
      */
      agx::Real getNu() const;

      /**
      When the efficiency goes to zero, the torque multiplication goes to infinity
      set/get the largest possible multiplication.
      */
      void setMaxMultiplication(agx::Real maxMultiplication);

      /**
      \returns the maximum multiplication of the torque converter.
      */
      agx::Real getMaxMultiplication() const;

      /**
      Set the pump torque table.

      At a given ratio between pump and turbine rpm (nu) the pump load will be
      possible to calculate from tabulated value at a fixed pump rpm (usually
      1000rpm).
      */
      void setPumpTorqueTable(const agx::RealPairVector& nu_torques);

      /**
      Set table mapping velocity ratio to torque multiplication.
      */
      void setMuTable(const agx::RealPairVector& nu_to_mu);

      /**
      Return the torque multiplication iven the current state
      */
      agx::Real getCurrentMultiplication() const;

      /**
      \returns pointer to the pump torque table.
      */
      const agxPowerLine::LinearLookupTable* getPumpTorqueTable() const;

      /**
      \returns pointer to the pump torque table.
      */
      agxPowerLine::LinearLookupTable* getPumpTorqueTable();

      /**
      Return pointer to table mapping angular velocity ratio to torque multiplication
      */
      const agxPowerLine::LinearLookupTable* getMuTable() const;

      /**
      Clear the table mapping angular velocity ratio to torque multiplication
      */
      void clearMuTable();

      /**
      Insert value in the table mapping angular velocity ratio to torque multiplication
      */
      void insertNuMuLookupValue(agx::Real nu, agx::Real mu);

      /**
      Set the pump torque reference rpm for the table.
      The pump torque values are set given a reference rpm. For other rpm's the pump load torque is found from the following formula:
      T_rpm(nu) = T_referenceRpm(nu) * (rpm/referenceRPm)^2
      */
      void setPumpTorqueReferenceRPM(agx::Real rpm);

      /**
      \returns the reference RPM used for the pump torque table.
      */
      agx::Real getPumpTorqueReferenceRPM() const;

      DOXYGEN_START_INTERNAL_BLOCK()
        virtual void initConstraint() override;
      agx::Real getWantedEfficiency() const;
      DOXYGEN_END_INTERNAL_BLOCK()

        /**
        Calculates compliance according to a mathematical torque converter.
        */
        virtual agx::RegularizationParameters::VariableType calculateComplianceAndDamping(const agx::Real timeStep, agx::Real& compliance, agx::Real& damping) override;

#ifndef SWIG
      /**
      Stores internal data into stream.
      */
      virtual bool store(agxStream::StorageStream& str) const override;

      /**
      Restores internal data from stream.
      */
      virtual bool restore(agxStream::StorageStream& str) override;

      AGXSTREAM_DECLARE_SERIALIZABLE(agxDriveTrain::deprecated::TorqueConverter);
#endif

    protected:

      virtual ~TorqueConverter();
      agx::Real m_maxMultiplication;
      agx::Real m_pumpTorqueReferenceRPM;

      TorqueConverterSpeedRatioLookupRef m_pumpTorqueTable;
      TorqueConverterSpeedRatioLookupRef m_muTable;

    };
    typedef agx::ref_ptr<TorqueConverter> TorqueConverterRef;
  }
}
