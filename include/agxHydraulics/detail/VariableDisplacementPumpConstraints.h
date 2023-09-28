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

#ifndef AGXHYDRAULICS_DETAIL_VARIABLE_DISPLACEMENT_PUMP_CONSTRAINTS_H
#define AGXHYDRAULICS_DETAIL_VARIABLE_DISPLACEMENT_PUMP_CONSTRAINTS_H

#include <agxPowerLine/PowerLineConstraints.h>
#include <agxHydraulics/export.h>



/*
This file contains a bunch of constraints used by the VariableDisplacementPump.
Some of the classes are used to relate the rotation of the pump shaft to the
rest of the pump, which other constraints dictate the allowed motions of the
VariableDisplacementPump's poppet. These are called spring constraints.
*/



namespace agxPowerLine
{
  class TranslationalDimension;
}



namespace agxHydraulics
{

  class VariableDisplacementPump;

  /// \cond INTERNAL_DOCUMENTATION
  namespace detail
  {
    /**
    Main flow constraint for the pump. It relates rotation of the input shaft to
    flow through the connected pipes and pushes on the regulator poppet.
    */
    class AGXHYDRAULICS_EXPORT ElementaryVariablePumpConstraint : public agxPowerLine::ElementaryPhysicalDimensionMultiBodyConstraint
    {
      public:
        /**
        \param deadheadPRessure - Currently not used. Could be used as a force range. \todo Consider removing this parameter.
        \param internalLeakage - The compliance of the constraints.
        */
        ElementaryVariablePumpConstraint(agx::Real deadheadPressure, agx::Real internalLeakage);

        virtual size_t getJacobian(
            agx::Jacobian6DOFElement* G,
            const agxPowerLine::RigidBodyPtrIntHashVector& bodyToIndexTable,
            const agx::ConstraintRigidBodyContainer& bodies,
            const agxPowerLine::PhysicalDimensionPtrIntTable& dimensions,
            const agx::Real timeStep) override;

        VariableDisplacementPump* getPumpConnector();

        AGXSTREAM_DECLARE_SERIALIZABLE(agxHydraulics::detail::ElementaryVariablePumpConstraint);

      protected:
        ElementaryVariablePumpConstraint();
        virtual ~ElementaryVariablePumpConstraint() {}

        using ElementaryPhysicalDimensionMultiBodyConstraint::getJacobian;
    };



    /**
    Constraint implementation for the variable displacement pump constraint.
    The VariableDisplacementPump connector creates instances of this
    constraint.

    \see ElementaryVariablePumpConstraint
    */
    class AGXHYDRAULICS_EXPORT VariablePumpConstraintImplementation : public agxPowerLine::PhysicalDimensionMultiBodyConstraintImplementation
    {
      public:
        /**
        \param deadheadPressure - Currently not used. Could be used as a force range. \todo Consider removing this parameter.
        \param internalLeakage - The compliance of the constraint.
        */
        VariablePumpConstraintImplementation(agx::Real deadheadPressure, agx::Real internalLeakage);
        virtual void addNotification() override;

        AGXSTREAM_DECLARE_SERIALIZABLE(agxHydraulics::detail::VariablePumpConstraintImplementation);

      protected:
        VariablePumpConstraintImplementation();
        virtual ~VariablePumpConstraintImplementation() {}

      private:
        agxHydraulics::detail::ElementaryVariablePumpConstraint* m_pumpConstraint;
    };





    /**

    Communication interface between the spring constraints and the
    SpringImplementation. It provides access to the poppets direction of motion
    and current displacement.
    */
    class SpringData : public agx::ElementaryConstraintData
    {
      public:
        /**
        \param
        Create a SpringData object that holds pointers to the given direction
        and displacement arguments.
        */
        SpringData(const agx::Vec3& direction, const agx::Real& displacement);
        SpringData(const SpringData& other);
        SpringData();

        void reconfigure(const agx::Vec3& direction, const agx::Real& displacement);

        agx::Vec3 getDirection() const;
        agx::Real getDisplacement() const;


    protected:
      SpringData& operator=(const SpringData&)
      {
        agxAbort1("Should never get here");
        return *this;
      }
    private:
        const agx::Vec3* m_direction;
        const agx::Real* m_displacement;
    };


    /**
    Constraint that keeps the poppet inside it's [0, 1] range. Used twice, one
    for each edge of the bound.
    */
    class ElementarySpringStopper : public agx::ElementaryConstraintNData<1, SpringData>
    {
      public:
        /**
        \param data - Contains pointers to data owned by the SpringImplementation that the constraint needs.
        \param limit - The bound that this ElementarySpringStopper enforces.
        \param allowedSide - When  1, then the constraint enforces displacement > limit.
                             When -1, then the constraint enforces displacement < limit.
                             No other values should be passed.
         */
        ElementarySpringStopper(const SpringData& data, agx::Real limit, agx::Real allowedSide);

        /**
        \return The point at which the stopper will stop the poppet.
        */
        agx::Real getLimit() const;

        /**
        \return The distance that the poppet is from the bound.
        */
        virtual agx::UInt getViolation(agx::Real* g, agx::UInt row) override;

        virtual agx::UInt getJacobian(
            agx::Jacobian6DOFElement* G, agx::UInt numBlocks,
            agx::UInt row, agx::GWriteState::Enum writeState) override;


        AGXSTREAM_DECLARE_SERIALIZABLE(agxHydraulics::detail::ElementarySpringStopper);

      protected:
        ElementarySpringStopper();
        virtual ~ElementarySpringStopper() {}

      private:
        agx::Real m_limit;
        agx::Real m_allowedSide;
    };


    /**
    Constraint that represents the spring holding the poppet in its seat.
    */
    class ElementarySpringForce : public agx::ElementaryConstraintNData<1, SpringData>
    {
      public:
        /**

        The zero point and the spring constant are calculated (by the
        VariableDisplacementPumpParameters helper class) so that the
        VariablePumpConstraint is just able to lift the poppet from its lower
        bound, i.e., overcoming the spring, when the pressure in the pump
        reaches the cutoff pressure.

        \param data - contains pointers to data owned by the SpringImplementation that the constraint needs.
        \param zeroPoint - The rest position of the spring.
        \param springConstant - The spring constant of the spring, measured in units of force / unt length.
        */
        ElementarySpringForce(const SpringData& data, agx::Real zeroPoint, agx::Real springConstant);

        virtual agx::UInt getJacobian(
            agx::Jacobian6DOFElement* G, agx::UInt numBlocks,
            agx::UInt row, agx::GWriteState::Enum writeState) override;

        virtual agx::UInt getViolation(agx::Real* g, agx::UInt row) override;

        AGXSTREAM_DECLARE_SERIALIZABLE(agxHydraulics::detail::ElementarySpringForce);

      protected:
        ElementarySpringForce();
        virtual ~ElementarySpringForce() {}

      private:
        agx::Real m_zeroPoint;
    };


    /**
    Constraint implementation containing elementary spring constraints; one for
    the actual spring and one each for the two bounds.
    */
    class SpringImplementation : public agx::ConstraintImplementation, public agxStream::Serializable
    {
      public:
        /**
        \param dimension - The TranslationalDimension of the poppets TranslationalUnit.
        \param zeroPoint - The rest position of the spring holding the poppet in place.
        \param springConstant - The spring constant of the spring holding the poppet in place.
        */
        SpringImplementation(agxPowerLine::TranslationalDimension* dimension, agx::Real zeroPoint, agx::Real springConstant);

        /**
        Make the current poppet state available to the elementary constraints.
        */
        virtual void prepare() override;

        AGXSTREAM_DECLARE_SERIALIZABLE(agxHydraulics::detail::SpringImplementation);

      protected:
        SpringImplementation();
        virtual ~SpringImplementation() {}

      private:
        agxPowerLine::TranslationalDimension* m_dimension;
        ElementarySpringForce* m_spring;
        ElementarySpringStopper* m_upperStopper;
        ElementarySpringStopper* m_lowerStopper;

        agx::Vec3 m_direction;    // The elementary constraints have pointers
        agx::Real m_displacement; // to these two members, which hold the state
    };                            // of the poppet.




    /**

    Constraint that holds the poppet in place. Contains two elementary
    constraints that keeps the poppet within its allowed bound and one
    constraint that acts as a spring in order to make the poppet position
    depend on the pressure in the pump.
    */
    class SpringConstraint : public agx::Constraint
    {
      public:
        /**
        \param dimension - The TranslationalDimension of the poppets TranslationalUnit.
        \param zeroPoint - The rest position of the spring holding the poppet in place.
        \param springConstant - The spring constant of the spring holding the poppet in place.
        */
        SpringConstraint(agxPowerLine::TranslationalDimension* dimension, agx::Real zeroPoint, agx::Real springConstant);
        virtual void render(agxRender::RenderManager* /*canvas*/, float /*scale*/) const override {}

        AGXSTREAM_DECLARE_SERIALIZABLE(agxHydraulics::detail::SpringConstraint);

      protected:
        SpringConstraint();
        virtual ~SpringConstraint();
        virtual int getNumDOF() const override;

      private:
        SpringImplementation* m_springImplementation;
    };
  }
  /// \endcond
}

#endif
