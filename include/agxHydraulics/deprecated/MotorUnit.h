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


#ifndef AGXHYDRAULICS_MOTOR_UNIT_H
#define AGXHYDRAULICS_MOTOR_UNIT_H

#include <agxHydraulics/FlowUnit.h>

namespace agxPowerLine
{
  class RotationalDimension;
  AGX_DECLARE_POINTER_TYPES(RotationalDimension);
}

namespace agxHydraulics
{
  class Motor;
  AGX_DECLARE_POINTER_TYPES(Motor);

  namespace deprecated
  {
    AGX_DECLARE_POINTER_TYPES(MotorUnit);

    /**
    \deprecated
    The MotorUnit is a FlowUnit that also contains a RotationalDimension and a
    MotorConnector. The FlowUnit part represents the fluid chamber inside the
    motor and the MotorConnector represents the attachment point to the rotating
    shaft, represented by the RotationalDimension.
    */
    class AGXHYDRAULICS_EXPORT MotorUnit : public agxHydraulics::FlowUnit
    {
      public:
        /**
        Create a new MotorUnit with the given fluid chamber properties.
        */
        MotorUnit(agx::Real pipeLength, agx::Real pipeArea, agx::Real fluidDensity);

        /**
        Provides access to the MotorConnector.
        \return The MotorConnector.
        */
        Motor* getMotorConnector();

        /**
        Provides access to the RotationalDimension.
        \return The RotationalDimension.
        */
        agxPowerLine::RotationalDimension* getRotationalDimension();
        const agxPowerLine::RotationalDimension* getRotationalDimension() const;


      // Methods called by the rest of the PowerLine/Hydraulics frame work.
      public:
        virtual agxPowerLine::DimensionAndSide getConnectableDimension(
            agxPowerLine::PhysicalDimension::Type type, agxPowerLine::Side side) override;

        virtual void getConnectableDimensionTypes(
            agxPowerLine::PhysicalDimension::TypeVector& types, agxPowerLine::Side side) const override;

        /// \cond INTERNAL_DOCUMENTATION
        /**
        Called by the PowerLine during stream serialization. Stores internal
        data into the given stream.
        \param str - The StorageStream to write internal data into.
        */
        virtual bool store(agxStream::StorageStream& str) const override;

        /**
        Called by the PowerLine during stream deserialization. Restores internal
        data from the given stream.
        \param str - The StorageStream to read internal data from.
        */
        virtual bool restore(agxStream::StorageStream& str) override;
        /// \endcond

        AGXSTREAM_DECLARE_SERIALIZABLE(agxHydraulics::deprecated::MotorUnit);

      protected:
        MotorUnit();
        virtual ~MotorUnit() {}

      private:
        agxHydraulics::MotorRef m_motorConnector;
        agxPowerLine::RotationalDimensionRef m_rotationalDimension;
    };
  }
}

#endif
