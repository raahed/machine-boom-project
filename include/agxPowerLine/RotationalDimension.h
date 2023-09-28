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

#ifndef AGXPOWERLINE_ROTATIONAL_DIMENSION_H
#define AGXPOWERLINE_ROTATIONAL_DIMENSION_H

#include <agxModel/export.h>
#include <agxPowerLine/PhysicalDimension.h>
#include <agxPowerLine/Connector.h>
#include <agxPowerLine/DirectionReference.h>

#include <agxSDK/StepEventListener.h>

#include <agxUtil/agxUtil.h>

#include <agx/RigidBody.h>
#include <agx/ElementaryConstraint.h>
#include <agx/Attachment.h>

namespace agxPowerLine
{
  #define DEFAULT_ROTATIONAL_DIRECTION agx::Vec3(0,1,0)

  class RotationalUnit;
  /**
  The rotational has one rotational degree of freedom.
  It has an angle as value,
  angular velocity as gradient and
  angular acceleration as second gradient.

  */
  class AGXMODEL_EXPORT RotationalDimension : public agxPowerLine::PhysicalDimension
  {
    public:
      /**
      Create a RotationalDimension with a direction relative to the world coordinate frame.
      */
      RotationalDimension(Unit *unit);

      /**
      Virtual function so that an instance can find its own name and not the name of the base class.
      */
      RotationalDimension(
          agxPowerLine::Unit* unit,
          agx::RigidBody* body,
          agx::UInt8 slot);

      /**
      Create a RotationalDimension whose direction can be relative to either the world or the body coordinate frame.
      */
      RotationalDimension(
          agxPowerLine::Unit* unit,
          agx::RigidBody* body,
          const agx::Vec3& direction,
          agxPowerLine::DirectionReference directionReference,
          bool externalBody = false);


      /**
      Render the axis of the dimension.
      */
      virtual void renderDebug() override;

      /**
      \returns input/output torque
      */
      virtual agx::Real getOutputLoad() const override;
      agx::Real getOutputTorque() const;
      agx::Real getInputTorque() const;


      /**
      \returns power in/out of the rotational dof.
      */
      virtual agx::Real getPowerIn() const override;
      virtual agx::Real getPowerOut() const override;

      /**
      \returns the inertia of the rotational dimension.
      */
      virtual agx::Real getMassProperty() const override;
      agx::Real getInertia() const;

      /**
      set the inertia of the rotational dimension.
      \retval Was it successful?
      */
      virtual bool setInertia( agx::Real value );

      /**
      set the angular velocity damping.
      */
      virtual void setAngularVelocityDamping( agx::Real damping );

      /**
      Stores internal data into stream.
      */
      virtual bool store(agxStream::StorageStream& out) const override;

      /**
      Restores internal data from stream.
      */
      virtual bool restore(agxStream::StorageStream& in) override;

      AGXSTREAM_DECLARE_SERIALIZABLE(agxPowerLine::RotationalDimension);

      /**
      Virtual function so that an instance can find its own name and not the name of the base class.
      */
      virtual std::string getTypeName() const override;

      /**
      Each class inheriting from agxPowerLine::PhysicalDimension has to have a
      static function returning a unique name. The rotational dimension has the
      name "ROTATIONAL".
      */
      static std::string getStaticTypeName();

      static agxPowerLine::PhysicalDimension::Type getStaticType();



    protected:
      friend class RotationalUnit;
      void postUpdate(agx::Real timeStep);

    protected:
      RotationalDimension();
      virtual ~RotationalDimension();
  };

  typedef agx::ref_ptr<RotationalDimension> RotationalDimensionRef;



  /**
  Defines a stiff gear constraint between two rotational dimensions
  */
  class AGXMODEL_EXPORT RotationalConnector : public agxPowerLine::Connector
  {
  public:
    /**
    Create a connector between two rotational dimensions.
    */
    RotationalConnector();

    /**
    Create a gear constraint.
    */
    virtual PhysicalDimensionMultiBodyConstraintImplementation* createConstraint() override;

    /**
    Calculate a compliance that will result in the wanted efficiency.
    */
    virtual agx::RegularizationParameters::VariableType calculateComplianceAndDamping( const agx::Real timeStep, agx::Real& compliance, agx::Real& damping) override;

    /**
    Stores internal data into stream.
    */
    virtual bool store(agxStream::StorageStream& str) const override;

    /**
    Restores internal data from stream.
    */
    virtual bool restore(agxStream::StorageStream& str) override;

    AGXSTREAM_DECLARE_SERIALIZABLE(agxPowerLine::RotationalConnector);

  protected:
    virtual ~RotationalConnector();

  };

  typedef agx::ref_ptr<RotationalConnector> RotationalConnectorRef;




  /**
  A torque generator adds torque to the body of a rotational dimension.
  */
  class AGXMODEL_EXPORT TorqueGenerator : public agxPowerLine::PowerGenerator
  {
  public:
    /**
    Create a torque generator.
    \param dimension - a dimension which has a body that will get torque added according to a lookup table.
    */
    TorqueGenerator(agxPowerLine::RotationalDimension* dimension);

    /**
    Finds the current RPM (and not angular velocity)
    */
    virtual agx::Real variableLookupFunction( ) const override;

    /**
    Implement this to scale lookups accordingly
    */
    virtual agx::Real resultScalerFunction( ) const override;

    /**
    Stores internal data into stream.
    */
    virtual bool store(agxStream::StorageStream& str) const override;

    /**
    Restores internal data from stream.
    */
    virtual bool restore(agxStream::StorageStream& str) override;

    AGXSTREAM_DECLARE_SERIALIZABLE(agxPowerLine::TorqueGenerator);

  protected:
    TorqueGenerator();
    virtual ~TorqueGenerator();
    virtual void applyPower() override;
    agx::observer_ptr<RotationalDimension> m_rotationalDimension;

  };

}

#endif // AGXMODEL_ROTATIONAL_DIMENSION_H
