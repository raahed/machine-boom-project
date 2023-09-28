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

#ifndef AGXPOWERLINE_WIRE_WINCH_CONSTRAINTS_H
#define AGXPOWERLINE_WIRE_WINCH_CONSTRAINTS_H


#include <agxPowerLine/detail/ActuatorConstraints.h>


namespace agxPowerLine
{
  class WireWinchActuator;

  namespace detail
  {
    /*
     * A collection of classes used internally by the WireWinchConnector.
     */

    /**
     * Elementary constraint that connects a RotationalDimension in a PowerLine to the two PhysicalDimensions of a RigidBodyUnit.
     */
    class AGXMODEL_EXPORT ElementaryWireWinchConnectorConstraint :
        public agxPowerLine::ElementaryActuatorConstraint
    {
      public:
        ElementaryWireWinchConnectorConstraint();
        AGXSTREAM_DECLARE_SERIALIZABLE(agxPowerLine::detail::ElementaryWireWinchConnectorConstraint);

      protected:
        virtual ~ElementaryWireWinchConnectorConstraint() {}
    };


    class AGXMODEL_EXPORT WireWinchConnectorConstraintImplementation :
        public agxPowerLine::ActuatorConstraintImplementation
    {
      public:
        WireWinchConnectorConstraintImplementation(WireWinchActuator* owningActuator);
        AGXSTREAM_DECLARE_SERIALIZABLE(agxPowerLine::detail::WireWinchConnectorConstraintImplementation);

      protected:
        WireWinchConnectorConstraintImplementation();
        virtual ~WireWinchConnectorConstraintImplementation() {}
    };
  }
}

#endif
