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

#ifndef AGXPOWERLINE_ROTATIONAL_WIRE_WINCH_H
#define AGXPOWERLINE_ROTATIONAL_WIRE_WINCH_H

#include <agxPowerLine/Actuator1DOF.h>
#include <agxPowerLine/detail/WireWinchConnector.h>

#include <agxPowerLine/TranslationalUnit.h>

#include <agxWire/WireWinchController.h>

namespace agxPowerLine
{
  AGX_DECLARE_POINTER_TYPES(WireWinchActuator);

  /**
  This actuator can be used to connect a Winch (which is a linear constraint) to a powerline for example a rotating shaft.
  This allows for driving a winch using powerline components such as motors, hydraulics etc.
  */
  class AGXMODEL_EXPORT WireWinchActuator : public agxPowerLine::TranslationalActuator
  {
    public:
      /**
      Create a wire winch actuator, with same arguments as needed for a WireWinchController.
      */
      WireWinchActuator(agxWire::WireWinchController* winch);

      void setWinch(agxWire::WireWinchController* winch);

      agxWire::WireWinchController* getWinch() const;

      void setEnable(bool enabled);
      bool getEnable() const;

      agx::Bool getBounceSpoolDirectionBeforeOutOfWire() const;

      void setBounceSpoolDirectionBeforeOutOfWire(agx::Bool bounce);

      agxPowerLine::TranslationalUnit* getInputRod();

      virtual bool preUpdate(agx::Real timeStamp) override;

      virtual bool postUpdate(agx::Real timeStamp) override;


      virtual void prepareForStore() override;
      virtual void prepareForRestore() override;


      /**
      Stores internal data into stream.
      */
      virtual bool store(agxStream::StorageStream& str) const override;

      /**
      Restores internal data from stream.
      */
      virtual bool restore(agxStream::StorageStream& str) override;


      AGXSTREAM_DECLARE_SERIALIZABLE(agxPowerLine::WireWinchActuator);


    public:
      // Methods called by the rest of the PowerLine framework.
      DOXYGEN_START_INTERNAL_BLOCK()

      virtual agxPowerLine::DimensionAndSide getConnectableDimension(
        agxPowerLine::PhysicalDimension::Type type,
        agxPowerLine::Side side) override;

      DOXYGEN_END_INTERNAL_BLOCK()


    private:
      void takeWinchControlParameters();
      void returnWinchControlParameters();
      void clearTakenWinchControlParameters();

    protected:

      WireWinchActuator();
      virtual ~WireWinchActuator();

      void detachActuatorUnitsFromEachOther();

      void connectWireWinchConnector();
      void disconnectWireWinchConnector();
      void reconnectWireWinchConnector();

    protected:
      agx::observer_ptr<agxWire::WireWinchController> m_winchController;

      agxPowerLine::TranslationalUnitRef m_inputRod;
      agxPowerLine::detail::WireWinchConnectorRef m_wireWinchConnector;

      agx::RangeReal m_tmpForceRange;
      agx::RangeReal m_tmpBrakeRange;
      agx::Bool      m_bounceSpoolDirectionBeforeOutOfWire;
  };


}

#endif //
