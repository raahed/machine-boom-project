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

#ifndef AGXPOWERLINE_TRANSLATIONAL_ACTUATOR_CONNECTOR_CONSTRAINTS
#define AGXPOWERLINE_TRANSLATIONAL_ACTUATOR_CONNECTOR_CONSTRAINTS

#include <agxPowerLine/detail/ActuatorConstraints.h>

/// \cond INTERNAL_DOCUMENTATION

namespace agxPowerLine
{
  class TranslationalActuator;

  namespace detail
  {
    /**
    Elementary constraint used by the TranslationalActuator. It sits between the
    input shaft of the translational actuator and the two bodies that the
    actuator's constraint is attached to.
    */
    class AGXMODEL_EXPORT ElementaryTranslationalActuatorConstraint :
        public agxPowerLine::ElementaryActuatorConstraint
    {
      public:
        ElementaryTranslationalActuatorConstraint();
        AGXSTREAM_DECLARE_SERIALIZABLE(agxPowerLine::detail::ElementaryTranslationalActuatorConstraint);

      protected:
        virtual ~ElementaryTranslationalActuatorConstraint() {}
    };



    /**
    A constraint implementation that contains a single ElementaryTranslationalActuatorConstraint.
    */
    class AGXMODEL_EXPORT TranslationalActuatorConstraintImplementation :
        public agxPowerLine::ActuatorConstraintImplementation
    {
      public:
        TranslationalActuatorConstraintImplementation(TranslationalActuator* owningActuator);
        AGXSTREAM_DECLARE_SERIALIZABLE(agxPowerLine::detail::TranslationalActuatorConstraintImplementation);

      protected:
        TranslationalActuatorConstraintImplementation();
        ~TranslationalActuatorConstraintImplementation() {}
    };
  }
}


/// \endcond

#endif
