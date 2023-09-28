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


#ifndef AGXHYDRAULICS_DETAIL_PISTON_CONSTRAINT_H
#define AGXHYDRAULICS_DETAIL_PISTON_CONSTRAINT_H


#include <agxPowerLine/PowerLineConstraints.h>
#include <agxPowerLine/ActuatorConnector.h>
#include <agxHydraulics/export.h>

namespace agxHydraulics
{
  class PistonActuator;
  class PistonActuatorConnector;

  /// \cond INTERNAL_DOCUMENTATION
  namespace detail
  {
    /**
    Constraint that connects a fluid chamber in a hydraulic cylinder to the two
    6-DoF bodies.
    */
    class AGXHYDRAULICS_EXPORT ElementaryPistonActuatorConstraint : public agxPowerLine::ElementaryActuatorConstraint
    {
      public:
        ElementaryPistonActuatorConstraint();

        virtual size_t getJacobian(
          agx::Jacobian6DOFElement* G,
          const agxPowerLine::RigidBodyPtrIntHashVector& bodyToIndexTable,
          const agx::ConstraintRigidBodyContainer& /*bodies*/,
          const agxPowerLine::PhysicalDimensionPtrIntTable& dimensions,
          const agx::Real /*timeStep*/) override;

        using agxPowerLine::ElementaryPhysicalDimensionConstraint::getJacobian;

        /*
        No getViolation because in the PowerLine it is the Connector that is
        responsible for this.
        */

        PistonActuatorConnector* getPistonChamberConnector();

        AGXSTREAM_DECLARE_SERIALIZABLE(agxHydraulics::detail::ElementaryPistonActuatorConstraint);

      protected:
        virtual ~ElementaryPistonActuatorConstraint() {}
    };


    /**
    Constraint that connects a fluid chamber in a hydraulic cylinder to the two
    6-DoF bodies.
    */
    class AGXHYDRAULICS_EXPORT PistonActuatorConstraintImplementation : public agxPowerLine::ActuatorConstraintImplementation
    {
      public:
        PistonActuatorConstraintImplementation(PistonActuator* owningActuator);

        virtual bool updateValid() override;

        AGXSTREAM_DECLARE_SERIALIZABLE(agxHydraulics::detail::PistonActuatorConstraintImplementation);

      protected:
        PistonActuatorConstraintImplementation();
        virtual ~PistonActuatorConstraintImplementation() {}

      private:
        void initialize();

      private:
        ElementaryPistonActuatorConstraint* m_pistonConstraint;
    };


  }
  /// \endcond
}

#endif

