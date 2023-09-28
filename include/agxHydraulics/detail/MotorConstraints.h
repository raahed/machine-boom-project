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

#include <agxPowerLine/PowerLineConstraints.h>
#include <agxHydraulics/export.h>

namespace agxHydraulics
{
  /// \cond INTERNAL_DOCUMENTATION
  namespace detail
  {
    /**
    \todo Can these two classes be removed and let the motor use the base
          classes directly? Will move some responsibility to the Motor class.
    */

    /**
    Basic power line constraint used by the hydraulic motor. Its only purpose is
    to provide a name and an ID string for the store/restore framework.

    Most of the configuration and management are done by the motor and its base
    classes.
    */
    class AGXHYDRAULICS_EXPORT ElementaryMotorConstraint : public agxPowerLine::ElementaryPhysicalDimensionMultiBodyConstraint
    {
      public:
        ElementaryMotorConstraint();
        AGXSTREAM_DECLARE_SERIALIZABLE(agxHydraulics::detail::ElementaryMotorConstraint);

      protected:
        virtual ~ElementaryMotorConstraint() {}
    };

    typedef agx::ref_ptr<ElementaryMotorConstraint> ElementaryMotorConstraintRef;


    /**
    Basic power line constraint used by the hydraulic motor. Its only purpose is
    to provide an ElementaryMotorConstraint and an ID string for the
    store/restore framework.
    */
    class AGXHYDRAULICS_EXPORT MotorConstraintImplementation : public agxPowerLine::PhysicalDimensionMultiBodyConstraintImplementation
    {
      public:
        MotorConstraintImplementation();
        AGXSTREAM_DECLARE_SERIALIZABLE(agxHydraulics::detail::MotorConstraintImplementation);

      protected:
        virtual ~MotorConstraintImplementation() {}
    };
  }
  /// \endcond
}
