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


#ifndef AGXHYDRAULIC_STOP_VALVE_H
#define AGXHYDRAULIC_STOP_VALVE_H


#include <agxHydraulics/detail/FlowBlockConstraint.h>

namespace agxHydraulics
{

  AGX_DECLARE_POINTER_TYPES(StopValve);

  /**
  A stop valve is either completely open or completely closed. The state is
  completely controlled by the user. Opening and closing is instantaneous.
  Closing a StopValve that has a large flow rate will cause a large pressure
  spike.
  */
  class AGXHYDRAULICS_EXPORT StopValve : public agxHydraulics::FlowUnit
  {
    public:
      /**
      The two states that a StopValve can be in.
      */
      enum State
      {
        OPEN = AGXHYDRAULICS_FLOWDIRECTION_BOTH,
        CLOSED = AGXHYDRAULICS_FLOWDIRECTION_NONE,

        /// \deprecated Use OPEN instead.
        BOTH = AGXHYDRAULICS_FLOWDIRECTION_BOTH,
        /// \deprecated Use CLOSED instead.
        NONE = AGXHYDRAULICS_FLOWDIRECTION_NONE
      };

      /**
      Create a new StopValve with the given geometry and fluid density.
      \param length - The length of the StopValve.
      \param area - The area of the StopVAlve.
      \param density - The density of the fluid flowing through the StopValve.
      \param state - The initial state of the StopValve. Either OPEN or CLOSED.
      */
      StopValve(agx::Real length, agx::Real area, agx::Real density, State state = CLOSED);

      /**
      Open or close the StopValve.
      \param state - The new state of the StopValve. Either OPEN or CLOSED.
      */
      void setState(State state);

      /**
      \return The current state of the StopValve. Either OPEN or CLOSED.
      */
      State getState() const;



      DOXYGEN_START_INTERNAL_BLOCK()
      // Methods called by the rest of the PowerLine framework.
    public:
#ifndef SWIG
      /**  */
      virtual bool addNotification(agxSDK::Simulation* simulation) override;

      /**  */
      virtual bool removeNotification(agxUtil::ConstraintHolder* constraintHolder, agxSDK::Simulation* simulation) override;

      /**  */
      virtual bool preUpdate(agx::Real timeStep) override;

      /**  */
      virtual bool store(agxStream::StorageStream& str) const override;
      /**  */
      virtual bool postStore(agxStream::StorageStream& str) const override;

      /**  */
      virtual bool restore(agxStream::StorageStream& str) override;
      /**  */
      virtual bool postRestore(agxStream::StorageStream& str) override;

      AGXSTREAM_DECLARE_SERIALIZABLE(agxHydraulics::StopValve);
#endif
      DOXYGEN_END_INTERNAL_BLOCK()

      /// \deprecated Use setState instead.
      void setAllowedDirection(State state);

      /// \deprecated Use getState instead.
      State getAllowedDirection() const;

      /// \deprecated This constraint does nothing.
      const agxHydraulics::FlowBlockConstraint* getFlowBlockConstraint() const;

    protected:
      StopValve();
      virtual ~StopValve() {}
      void updateFlowDirection();

    private:
      agxHydraulics::FlowBlockConstraintRef m_flowBlockConstraint;
      State m_state;
  };
}



#endif
