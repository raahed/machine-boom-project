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

#ifndef AGXPOWERLINE_POWER_LINE_CONSTRAINTS_H
#define AGXPOWERLINE_POWER_LINE_CONSTRAINTS_H

#include <agxModel/export.h>
#include <agxPowerLine/PowerLine.h>

#include <agxSDK/StepEventListener.h>

#include <agx/RigidBody.h>
#include <agx/ElementaryConstraint.h>
#include <agx/ConstraintImplementation.h>

namespace agxPowerLine
{
  typedef agx::HashVector<PhysicalDimension*, int> PhysicalDimensionPtrIntTable;

  /**
  Pure virtual function.
  Elementary constraint for constraining two physical dimensions
  Inherit from this to make a constraint between two physical dimensions.
  */
  class AGXMODEL_EXPORT ElementaryPhysicalDimensionConstraint : public agx::ElementaryConstraintN< 1 >
  {
    public:

      /**
      Constructor for pure virtual function.
      */
      ElementaryPhysicalDimensionConstraint();

      /**
      getJacobian must be implemented for all classes that inherits from this.
      */
      virtual size_t getJacobian( agx::Jacobian6DOFElement* /*G*/,
        const RigidBodyPtrIntHashVector& /*bodyToIndexTable*/,
        const agx::ConstraintRigidBodyContainer& /*bodies*/,
        const PhysicalDimensionPtrIntTable& /*dimensions*/,
        const agx::Real /*dt*/ ) = 0;


      virtual void getJacobian(const agxPowerLine::PhysicalDimension* dimension, agx::Jacobian6DOFElement& G) const = 0;

      /**
      \returns the constraint violation. (Default zero. Also zero for non-holonomic constraints.)
      */
      virtual agx::UInt getViolation( agx::Real* /*g*/, agx::UInt /*row*/ ) override;

      /**
      Set a pointer to the connector that will define this constraint.
      */
      void setConnector(Connector* connector);
      Connector* getConnector();
      const Connector* getConnector() const;

      /**
      Check if the dimensions of the constraint are legit.
      */
      virtual bool isValid() const;

      /**
      Store this object to stream.
      */
      virtual void store( agxStream::OutputArchive& out ) const override;

      /**
      Restore this object from stream.
      */
      virtual void restore( agxStream::InputArchive& in ) override;


      AGXSTREAM_DECLARE_ABSTRACT_SERIALIZABLE(agxPowerLine::ElementaryPhysicalDimensionConstraint);


    protected:
      /**
      destructor.
      */
      virtual ~ElementaryPhysicalDimensionConstraint();

    protected:
      Connector*                     m_connector;

      using agx::ElementaryConstraintN<1>::getJacobian;

      /**
      Forced inheritance from agx::ElementaryConstraint.
      */
      virtual agx::UInt getJacobian( agx::Jacobian6DOFElement* /*G*/,
        agx::UInt /*numBlocks*/,
        agx::UInt /*row*/,
        agx::GWriteState::Enum /*writeState*/ ) override;
  };


  /**
  Elementary constraint for constraining an arbitrary number of ONE specific PhysicalDimension::Type
  */
  class AGXMODEL_EXPORT ElementaryPhysicalDimensionMultiBodyConstraint : public agxPowerLine::ElementaryPhysicalDimensionConstraint
  {
    public:
      /**
      Constructor.
      */
      ElementaryPhysicalDimensionMultiBodyConstraint();

      /**
      Implementation of getJacobian for a multi body constraint for the power line, where all dimensions have the same type.
      */
      virtual size_t getJacobian(
          agx::Jacobian6DOFElement* G,
          const RigidBodyPtrIntHashVector& bodyToIndexTable,
          const agx::ConstraintRigidBodyContainer& bodies,
          const PhysicalDimensionPtrIntTable& dimensions,
          const agx::Real dt ) override;


      virtual void getJacobian(
          const agxPowerLine::PhysicalDimension* dimension,
          agx::Jacobian6DOFElement& G) const override;

    protected:
      virtual ~ElementaryPhysicalDimensionMultiBodyConstraint();

      using agxPowerLine::ElementaryPhysicalDimensionConstraint::getJacobian;

    protected:

  };


  /**
  Constraint implementation for constraining a number of physical dimensions of the same type.
  */
  class AGXMODEL_EXPORT PhysicalDimensionMultiBodyConstraintImplementation : public agx::ConstraintImplementation , public agxStream::Serializable
  {
    public:
      /**
      Constructor.
      */
      PhysicalDimensionMultiBodyConstraintImplementation();

      ~PhysicalDimensionMultiBodyConstraintImplementation();

      /**
      Test if the constraint is still valid.
      */
      virtual bool updateValid() override;

      virtual void prepareDimensions();

      /**
      Update the jacobians of the included dimensions.
      */
      virtual size_t updateJacobian( agx::Jacobian6DOFElement* jacobians ) override;

      void setConnector(agxPowerLine::Connector* connector);

      /**
      Prepare system before solve.
      */
      virtual void preSystemCallback( agx::DynamicsSystem* system ) override;

      /**
      Prepare system after solve.
      */
      virtual void postSystemCallback( agx::DynamicsSystem* system ) override;

      /**
      \returns pointer to the elementary constraint of the implementation.
      */
      ElementaryPhysicalDimensionConstraint* getElementaryConstraint() const;
      using agx::ConstraintImplementation::getElementaryConstraint;

      /**
      \returns the compliance of the constraint.
      */
      agx::Real getCompliance() const;

      /**
      Set the compliance of the constraint.
      */
      void setCompliance( agx::Real compliance );

      /**
      \returns the current load of the constraint.
      */
      agx::Real getCurrentForce() const;


      /**
      Add a physical dimension to the constraint.
      */
      void add(PhysicalDimension* dimension);


      virtual void storeLightData(agxStream::StorageStream& str) const override;
      virtual void restoreLightData(agxStream::StorageStream& str) override;

      AGXSTREAM_DECLARE_SERIALIZABLE(agxPowerLine::PhysicalDimensionMultiBodyConstraintImplementation);

    protected:

      PhysicalDimensionMultiBodyConstraintImplementation( ElementaryPhysicalDimensionConstraint* elementaryConstraint );

      void initialize();

    protected:
      ElementaryPhysicalDimensionConstraint*   m_elementaryConstraint;
      RigidBodyPtrIntHashVector      m_bodyToIndexTable;
      agx::Vector<agx::observer_ptr<agx::RigidBody> > m_observedBodies;
      PhysicalDimensionPtrIntTable             m_dimensionToIndexTable;
      size_t                                   m_indexCounter;
  };



  /**
  An elementary gear constraint constrains the velocity of the rotational dimensions of a number bodies.
  */
  class AGXMODEL_EXPORT ElementaryGearConstraint : public agxPowerLine::ElementaryPhysicalDimensionMultiBodyConstraint
  {
    public:
      ElementaryGearConstraint();
      AGXSTREAM_DECLARE_SERIALIZABLE(agxPowerLine::ElementaryGearConstraint);

     protected:
      virtual ~ElementaryGearConstraint() {}
  };

  typedef agx::ref_ptr<ElementaryGearConstraint> GearConstraintRef;



  /**
  Constraint implementation that constrains the velocity of the rotational dimensions of a number bodies.
  */
  class AGXMODEL_EXPORT GearConstraintImplementation : public PhysicalDimensionMultiBodyConstraintImplementation
  {
    public:
      GearConstraintImplementation();
      AGXSTREAM_DECLARE_SERIALIZABLE(agxPowerLine::GearConstraintImplementation);

    protected:
      virtual ~GearConstraintImplementation() {}
  };



  /**
  An elementary gear constraint constrains the velocity of the rotational dimensions of a number bodies.
  */
  class AGXMODEL_EXPORT ElementaryHolonomicGearConstraint : public agxPowerLine::ElementaryGearConstraint
  {
    public:
      ElementaryHolonomicGearConstraint();

      /**
      Check if the dimensions of the constraint are legit.
      */
      virtual bool isValid() const override;
  };

  typedef agx::ref_ptr<ElementaryHolonomicGearConstraint> HolonomicGearConstraintRef;


  /**
  Constraint implementation that constrains the velocity of the rotational dimensions of a number bodies.
  */
  class AGXMODEL_EXPORT HolonomicGearConstraintImplementation : public GearConstraintImplementation
  {
    public:
      HolonomicGearConstraintImplementation();
  };


  class AGXMODEL_EXPORT ElementaryTranslationalConstraint :
      public agxPowerLine::ElementaryPhysicalDimensionMultiBodyConstraint
  {
  public:
    ElementaryTranslationalConstraint();
    AGXSTREAM_DECLARE_SERIALIZABLE(agxPowerLine::ElementaryTranslationalConstraint);
  protected:
    virtual ~ElementaryTranslationalConstraint() {}
  };



  class AGXMODEL_EXPORT TranslationalConstraintImplementation : public PhysicalDimensionMultiBodyConstraintImplementation
  {
    public:
      TranslationalConstraintImplementation();
      AGXSTREAM_DECLARE_SERIALIZABLE(agxPowerLine::TranslationalConstraintImplementation);

    protected:
      virtual ~TranslationalConstraintImplementation() {}
  };


}

#endif // AGXMODEL_POWER_LINE_CONSTRAINTS_H
