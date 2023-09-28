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


#ifndef AGXHYDRAULIC_CHECK_VALVE_H
#define AGXHYDRAULIC_CHECK_VALVE_H

#include <agxHydraulics/detail/FlowBlockConstraint.h>

namespace agxHydraulics
{
  AGX_DECLARE_POINTER_TYPES(CheckValve);

  /**
  A check valve is a FlowUnit that only allows flow in one direction. It can,
  for example, be used to prevent a hydraulic cylinder from collapsing due to
  heavy load when the pressure source isn't strong enough to hold the weight.

  The allowed flow direction is configurable and can be changed during runtime.
  */
  class AGXHYDRAULICS_EXPORT CheckValve : public agxHydraulics::FlowUnit
  {
    public:
      enum AllowedDirection
      {
        FORWARD = AGXHYDRAULICS_FLOWDIRECTION_FORWARD, ///< Flow from the input to the output is allowed.
        BACKWARD = AGXHYDRAULICS_FLOWDIRECTION_BACKWARD ///< Flow from the output to the input is allowed.
      };

      /**
      Create a new check valve with the given dimensions and allowed flow
      direction.
      \param length - The length of the check valve.
      \param area - The area of the check valve.
      \param density - The density of the fluid passing through the check valve.
      \param direction - The direction which flow through the check valve is allowed.
      */
      CheckValve(agx::Real length, agx::Real area, agx::Real density, AllowedDirection direction = FORWARD);

      /**
      Used to change the direction of flow that the relief valve will allow.
      Beware that changing the allowed direction while there is still flow
      along the old allowed direction may cause large and sudden pressure
      spikes. Consider using other means to reduce the flow rate before
      reconfiguring the check valve.

      \param direction - The new allowed flow direction.
      */
      void setAllowedDirection(AllowedDirection direction);

      /**
      \return The current allowed flow direction.
      */
      AllowedDirection getAllowedDirection() const;

      /**
      Provides access to the constraint that blocks flow in the wrong
      direction. Leakage through the check valve can be controlled by setting
      the compliance on this constraint.

      \return The constraint that prevents flow in the blocked direction.
      */
      agxHydraulics::FlowBlockConstraint* getFlowBlockConstraint();
      const agxHydraulics::FlowBlockConstraint* getFlowBlockConstraint() const;


    // Methods called by the rest of the PowerLine/Hydraulics frame work.
    public:
      DOXYGEN_START_INTERNAL_BLOCK()
      /**
      Creates, if necessary, the flow block constraint and adds it to the given
      simulation.
      */
      virtual bool addNotification(agxSDK::Simulation* simulation) override;

#ifndef SWIG
      /**
      Removes the flow block constraint from the given simulation.
      */
      virtual bool removeNotification(agxUtil::ConstraintHolder* constraintHolder, agxSDK::Simulation* simulation) override;

      virtual bool store(agxStream::StorageStream& str) const override;
      virtual bool postStore(agxStream::StorageStream& str) const override;

      virtual bool restore(agxStream::StorageStream& str) override;
      virtual bool postRestore(agxStream::StorageStream& str) override;
#endif
      DOXYGEN_END_INTERNAL_BLOCK()
      AGXSTREAM_DECLARE_SERIALIZABLE(agxHydraulics::CheckValve);

    protected:
      CheckValve();
      virtual ~CheckValve() {}

      /**
      Apply the locally stored allowed direction to the flow block constraint,
      if possible.
      */
      void updateFlowDirection();

    private:
      /// Constraint that prevents flow in the wrong direction.
      agxHydraulics::FlowBlockConstraintRef m_flowBlockConstraint;

      /// The direction in which flow is allowed. Flow in the other direction
      /// will be blocked.
      AllowedDirection m_allowedDirection;
  };

}


#endif
