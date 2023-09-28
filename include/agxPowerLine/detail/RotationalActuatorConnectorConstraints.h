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

#ifndef AGXPOWERLINE_ROTATIONAL_ACTUATOR_CONSTRAINTS_H
#define AGXPOWERLINE_ROTATIONAL_ACTUATOR_CONSTRAINTS_H

#include <agxPowerLine/detail/ActuatorConstraints.h>

/// \cond INTERNAL_DOCUMENTATION

namespace agxPowerLine
{
  class RotationalActuator;

  namespace detail
  {
    /*
     A collection of constraint-related classes used internally by the RotationalActuator.
    */


    /**
    Elementary constraint that connects the input shaft of a RotationalActuator
    to the ActuatorBodyUnits that the actuator contains.
    */
    class AGXMODEL_EXPORT ElementaryRotationalActuatorConstraint :
        public agxPowerLine::ElementaryActuatorConstraint
    {
      public:
        ElementaryRotationalActuatorConstraint();
        AGXSTREAM_DECLARE_SERIALIZABLE(agxPowerLine::detail::ElementaryRotationalActuatorConstraint);

      protected:
        virtual ~ElementaryRotationalActuatorConstraint() {}
    };


    /**
    Constraint implementation that connects the input shaft of a
    RotationalActuator to the ActuatorBodyUnits that the actuator contains.
    */
    class AGXMODEL_EXPORT RotationalActuatorConstraintImplementation :
        public agxPowerLine::ActuatorConstraintImplementation
    {
      public:
        RotationalActuatorConstraintImplementation(RotationalActuator* owningActuator);
        AGXSTREAM_DECLARE_SERIALIZABLE(agxPowerLine::detail::RotationalActuatorConstraintImplementation);

      protected:
        RotationalActuatorConstraintImplementation();
        virtual ~RotationalActuatorConstraintImplementation() {}
    };
  }
}

/// \endcond

#endif
