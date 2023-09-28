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
    Basic power line constraint used by the hydraulic pump. Its only purpose is
    to provide a name and an ID string for the store/restore framework.

    Most of the configuration and management are done by the pump and its base
    classes.
    */
    class AGXHYDRAULICS_EXPORT ElementaryPumpConstraint : public agxPowerLine::ElementaryPhysicalDimensionMultiBodyConstraint
    {
      public:
        ElementaryPumpConstraint();
        AGXSTREAM_DECLARE_SERIALIZABLE(agxHydraulics::detail::ElementaryPumpConstraint);

      protected:
        virtual ~ElementaryPumpConstraint() {}
    };

    typedef agx::ref_ptr<ElementaryPumpConstraint> ElementaryPumpConstraintRef;



    /**
    Basic power line constraint used by the hydraulic pump. Its only purpose is
    to provide an ElementaryMotorConstraint and an ID string for the
    store/restore framework.
    */
    class AGXHYDRAULICS_EXPORT PumpConstraintImplementation : public agxPowerLine::PhysicalDimensionMultiBodyConstraintImplementation
    {
      public:
        PumpConstraintImplementation();
        AGXSTREAM_DECLARE_SERIALIZABLE(agxHydraulics::detail::PumpConstraintImplementation);

      protected:
        virtual ~PumpConstraintImplementation() {}
    };
  }
  /// \endcond

}


