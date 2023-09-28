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



#ifndef AGXHYDRAULICS_SPOOL_VALVE_CONSTRAINTS_H
#define AGXHYDRAULICS_SPOOL_VALVE_CONSTRAINTS_H


#include <agxPowerLine/PowerLineConstraints.h>
#include <agxHydraulics/export.h>


namespace agxHydraulics
{
  namespace detail
  {
    /**
    \internal
    A constraint that has no effect on the simulation. Used by the GraphJoiningConnector.

    \see GraphJoiningConnector
    */
    class AGXHYDRAULICS_EXPORT ElementaryGraphJoiningConstraint : public agxPowerLine::ElementaryPhysicalDimensionMultiBodyConstraint
    {
      public:
        /**
        Set all Jacobians to zero.
        */
        virtual size_t getJacobian(
            agx::Jacobian6DOFElement* G,
            const agxPowerLine::RigidBodyPtrIntHashVector& bodyToIndexTable,
            const agx::ConstraintRigidBodyContainer& bodies,
            const agxPowerLine::PhysicalDimensionPtrIntTable& dimensions,
            const agx::Real timeStep) override;

        /**
        Set the Jacobian to zero.
        */
        virtual void getJacobian(const agxPowerLine::PhysicalDimension* dimension, agx::Jacobian6DOFElement& G) const override;

        AGXSTREAM_DECLARE_SERIALIZABLE(agxHydraulics::detail::ElementaryGraphJoiningConstraint);

      protected:
        virtual ~ElementaryGraphJoiningConstraint() {}
        using agxPowerLine::ElementaryPhysicalDimensionMultiBodyConstraint::getJacobian;
    };



    /**
    \internal
    ConstraintImplementation containing an ElementaryGraphJoiningConstraint.
    */
    class AGXHYDRAULICS_EXPORT GraphJoiningConstraintImplementation : public agxPowerLine::PhysicalDimensionMultiBodyConstraintImplementation
    {
      public:
        GraphJoiningConstraintImplementation();

        AGXSTREAM_DECLARE_SERIALIZABLE(agxHydraulics::detail::GraphJoiningConstraintImplementation);

      protected:
        virtual ~GraphJoiningConstraintImplementation() {}
    };





    /**
    \internal
    A Connector that joins two power line sub-graphs without having an effect
    on the simulation. Used by the spool valve to keep the whole surrounding
    system a single graph, even when the linking suggests otherwise, in order
    to facilitate graph traversal over the spool valve.
    */
    class AGXHYDRAULICS_EXPORT GraphJoiningConnector : public agxPowerLine::Connector
    {
      public:
        GraphJoiningConnector();
        virtual agxPowerLine::PhysicalDimensionMultiBodyConstraintImplementation* createConstraint() override;

        /**
        Stores internal data into stream.
        */
        virtual bool store(agxStream::StorageStream& str) const override;

        /**
        Restores internal data from stream.
        */
        virtual bool restore(agxStream::StorageStream& str) override;

        AGXSTREAM_DECLARE_SERIALIZABLE(agxHydraulics::detail::GraphJoiningConnector);

      protected:
        virtual ~GraphJoiningConnector() {}
    };


  }
}

#endif
