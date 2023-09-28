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


#ifndef AGXHYDRAULICS_CONSTANT_FLOW_VALVE_H
#define AGXHYDRAULICS_CONSTANT_FLOW_VALVE_H

#include <agxHydraulics/FlowUnit.h>
#include <agxHydraulics/detail/FlowRateConstraint.h>

namespace agxHydraulics
{
  AGX_DECLARE_POINTER_TYPES(ConstantFlowValve);

  /**
  A constant flow valve is a valve that acts to limit the flow rate through a
  pipe to some maximum value. In the default mode it behaves exactly like a
  regular pipe as long as the flow rate is less than the target flow rate. When
  the target flow rate has been reached the valve will prevent any further
  acceleration of the fluid.

  In the pumping mode the constant flow valve will add pressure to the system
  in order to reach the target flow rate. It acts like a pump. This mode is
  useful when the complexity of an engine/pump combination is unnecessary.
  */
  class AGXHYDRAULICS_EXPORT ConstantFlowValve : public agxHydraulics::FlowUnit
  {
    public:
      /**
      Create a new constant flow valve with the given geometry and target flow rate.

      \param length - The length of the pipe.
      \param area - The area of the pipe.
      \param fluidDensity - The density of the flud passing through the pipe.
      \param targetFlowRate - The target flow rate of the constant flow valve.
      \param allowPumping - True if the valve should be able to accelerate the fluid.
      */
      ConstantFlowValve(
           agx::Real length, agx::Real area, agx::Real fluidDensity,
           agx::Real targetFlowRate, bool allowPumping = false);

      /**
      \return The target flow rate of the constant flow valve.
      */
      agx::Real getTargetFlowRate() const;

      /**
      Set the target flow rate of the constant flow valve.
      \param targetFlowRate - The new target flow rate.
      */
      void setTargetFlowRate(agx::Real targetFlowRate);

      /**
      \return True if the constant flow valve is allowed to accelerate the fluid.
      */
      bool getAllowPumping() const;

      /**
      Set to true to allow the constant flow valve to accelerate the fluid.
      \param allowPumping - True if pumping is to be allowed. False otherwise.
      */
      void setAllowPumping(bool allowPumping);


      /**
      Enable or disable the constant flow valve. When disabled the valve acts
      just like a regular pipe.
      \param enable - True if the constant flow valve is to be enabled. False otherwise.
      */
      void setEnable(bool enable);

      /**
      \return True if the constant flow valve is enabled. False otherwise.
      */
      bool getEnable() const;


      /**
      Provides access to the constraint that regulates the flow rate.
      */
      agx::Constraint* getConstraint();
      const agx::Constraint* getConstraint() const;



    // Methods called by the rest of the PowerLine/Hydraulics frame work.
    public:
      DOXYGEN_START_INTERNAL_BLOCK()

      /**
      Called at the beginning of every time step.
      */
      virtual bool preUpdate(agx::Real timeStep) override;

      /**
      Called when the constant flow valve is added to a simulation. Will add
      the flow rate constraint to the simulation.
      */
      virtual bool addNotification(agxSDK::Simulation* simulation) override;

#ifndef SWIG
      /**
      Called when the constant flow valve is removed from a simulation. Will
      remove the flow rate constraint from the simulation.
      */
      virtual bool removeNotification(agxUtil::ConstraintHolder* constraintHolder, agxSDK::Simulation* simulation) override;


      /**
      Stores internal data into stream.
      */
      virtual bool store(agxStream::StorageStream& str) const override;

      /**
      Store flow rate constraint data to stream.
      */
      virtual bool postStore(agxStream::StorageStream& str) const override;

      /**
      Restores internal data from stream.
      */
      virtual bool restore(agxStream::StorageStream& str) override;

      /**
      Store flow rate constraint data to stream.
      */
      virtual bool postRestore(agxStream::StorageStream& str) override;

      AGXSTREAM_DECLARE_SERIALIZABLE(agxHydraulics::ConstantFlowValve);
#endif

    DOXYGEN_END_INTERNAL_BLOCK()

    protected:
      ConstantFlowValve();
      virtual ~ConstantFlowValve();

      void createFlowRateConstraint();
      void destroyFlowRateConstraint();

    private:
      agxHydraulics::detail::FlowRateConstraintData m_constraintData;
      agxHydraulics::detail::FlowRateConstraintRef m_flowRateConstraint;
  };
}


#endif
