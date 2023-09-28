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

#include <agxPowerLine/Connector.h>

#include <agxDriveTrain/Gear.h>
#include <agxDriveTrain/Shaft.h>

#include <agx/Vector.h>

#include <agxModel/export.h>


namespace agxDriveTrain
{
  AGX_DECLARE_POINTER_TYPES(MultiGear);

  /**
  A MultiGear is a type of gear connector that can have multiple connected
  shafts, each with its own gear ratio. Rotating a shaft connected with gear
  ratio \f$\alpha\f$ one radian will result in an internal rotation of the MultiGear
  by that gear ratio. Other connected shafts will, in response, rotate by \f$\alpha\f$
  divided by each connected shaft's gear ratio.
  */
  class AGXMODEL_EXPORT MultiGear : public agxPowerLine::Connector
  {
    public:
      MultiGear();

      /**
      Connect the given unit to this MultiGear. The unit must have a rotational
      dimension for the connection to be successful.

      \param unit - The unit to connect.
      \param gearRatio - The gear ratio that the given unit is to be connected with.
      \return True if the connection is made successfully. False otherwise.
      */
      bool connect(agxPowerLine::Unit* unit, agx::Real gearRatio);

      /**
      Connect the given unit with a gear ratio of 1. The gear ratio can be
      specified later using \p setGearRatio.

      \param unit - The unit to connect.
      \return True if the connection is made successfully. False otherwise.
      */
      virtual bool connect(agxPowerLine::Unit* unit) override;

      virtual bool connect(agxPowerLine::Side mySide, agxPowerLine::Side otherSide, agxPowerLine::Unit* otherUnit) override;

      /**
      Disconnect the give unit from this MultiGear.

      \param unit - The unit to disconnect.
      \return True if the disconnect was successful. False otherwise.
      */
      virtual bool disconnect(agxPowerLine::Unit* unit) override;


      /**
      Change the gear ratio for the given unit. The unit must be connected to
      this MultiGear at the time of the call.

      \param unit - The unit to change gear ratio for.
      \param gearRatio - The new gear ratio.
      \return True if the gear ratio was changed. False otherwise.
      */
      bool setGearRatio(agxPowerLine::Unit* unit, agx::Real gearRatio);

      /**
      Returns the gear ratio with which the given unit is connected to this
      MultiGear. Returns 0.0 if the given unit is not connected to this
      MultiGear.

      \param unit - The unit to get the gear ratio for.
      \return The gear ratio of the gear connecting the given unit, or 0.0 the unit is not connected.
      */
      agx::Real getGearRatio(const agxPowerLine::Unit* unit) const;


      /**
      \return The internal shaft that the connected external shafts connect to.
      */
      agxDriveTrain::Shaft* getInternalShaft();
      const agxDriveTrain::Shaft* getInternalShaft() const;


    public: // Methods called by the rest of the PowerLine framework.
      DOXYGEN_START_INTERNAL_BLOCK()

      using agxPowerLine::Connector::connect;
      using agxPowerLine::Connector::disconnect;

      /**
      Sided connect. The sides are ignored. A MultiGear can only have output
      connections, and will only connect to the input side of the unit.
      */
      bool connect(
          agxPowerLine::Side mySide,
          agxPowerLine::Side unitSide,
          agxPowerLine::Unit* unit,
          agx::Real gearRatio);

#ifndef SWIG
      virtual agxPowerLine::PhysicalDimensionMultiBodyConstraintImplementation* createConstraint() override;
#endif

      AGXSTREAM_DECLARE_SERIALIZABLE(agxDriveTrain::MultiGear);
      virtual bool store(agxStream::StorageStream& out) const override;
      virtual bool restore(agxStream::StorageStream& in) override;

      DOXYGEN_END_INTERNAL_BLOCK()
    private:
      agx::Vector<agxDriveTrain::GearRef> m_connections;
      agxDriveTrain::ShaftRef m_internalShaft;
  };
}
