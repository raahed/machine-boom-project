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

#ifndef AGXPOWERLINE_TRANSLATIONAL_DIMENSION_H
#define AGXPOWERLINE_TRANSLATIONAL_DIMENSION_H

#include <agxModel/export.h>
#include <agxPowerLine/PhysicalDimension.h>
#include <agxPowerLine/Connector.h>
#include <agxPowerLine/DirectionReference.h>

#include <agxSDK/StepEventListener.h>

#include <agxUtil/agxUtil.h>

#include <agx/RigidBody.h>
#include <agx/ElementaryConstraint.h>

namespace agxPowerLine
{

  DOXYGEN_START_INTERNAL_BLOCK()

  #define DEFAULT_TRANSLATIONAL_DIRECTION agx::Vec3(0, 1, 0)

  class TranslationalUnit;

  /**
  The translational has one linear degree of freedom.
  It has a scalar position as value,
  linear velocity as gradient and
  linear acceleration as second gradient.
  */
  class AGXMODEL_EXPORT TranslationalDimension : public agxPowerLine::PhysicalDimension
  {
    public:
      /**
      Create a TranslationalDimension that is backed by a body created and owned
      by  the PowerLine. Such a body will be strictly one-dimensional and a single
      body can be used by multiple PhysicalDimensions.
      */
      TranslationalDimension(agxPowerLine::Unit* unit);

      /**
      Create a TranslationalDimension that is backed by a body created and owned
      by something other than the PowerLine itself. This can, for example, be
      a body part of a constraint for which an \p Actuator has been attached.
      */
      TranslationalDimension(
          agxPowerLine::Unit* unit,
          agx::RigidBody* body,
          const agx::Vec3& direction,
          agxPowerLine::DirectionReference directionReference,
          bool externalBody = false);

      /**
      Each class inheriting from agxPowerLine::PhysicalDimension has to have a static function returning a unique name.
      The translational dimension has the name "TRANSLATIONAL"
      */
      static std::string getStaticTypeName();

      /**
      Virtual function so that an instance can find its own name and not the name of the base class.
      */
      virtual std::string getTypeName() const override;

      /**
      Render the axles of the dimensions
      */
      virtual void renderDebug() override;

      /**
      \returns input/output force
      */
      virtual agx::Real getOutputLoad() const override;
      virtual agx::Real getInputLoad() const override;
      agx::Real getOutputForce() const;
      agx::Real getInputForce() const;


      /**
      \returns power in/out of the translational dof.
      */
      virtual agx::Real getPowerIn() const override;
      virtual agx::Real getPowerOut() const override;

      /**
      \returns the direction of the rotational axis in world coordinates
      */
      virtual agx::Vec3 getWorldDirection() const;

      /**
      Returns the ID of the TranslationalDimension. This ID is used to find
      TranslationalDimensions in the PhysicalDimension vector held by all
      Units.

      \return The PhysicalDimension ID of the TranslationalDimension.
      */
      static agxPowerLine::PhysicalDimension::Type getStaticType();

      /**
      Stores internal data into stream.
      */
      virtual bool store(agxStream::StorageStream& out) const override;

      /**
      Restores internal data from stream.
      */
      virtual bool restore(agxStream::StorageStream& in) override;

      AGXSTREAM_DECLARE_SERIALIZABLE(agxPowerLine::TranslationalDimension);

    protected:
      TranslationalDimension();
      virtual ~TranslationalDimension();
  };


  typedef agx::ref_ptr<TranslationalDimension> TranslationalDimensionRef;

  /**
  UNDER DEVELOPMENT, DOES NOT FUNCTION VERY WELL.
  */
  class AGXMODEL_EXPORT TranslationalConnector : public agxPowerLine::Connector
  {
  public:
    TranslationalConnector();

    virtual PhysicalDimensionMultiBodyConstraintImplementation* createConstraint() override;
    virtual agx::RegularizationParameters::VariableType calculateComplianceAndDamping( const agx::Real timeStep, agx::Real& compliance, agx::Real& damping) override;

    /**
    Stores internal data into stream.
    */
    virtual bool store(agxStream::StorageStream& out) const override;

    /**
    Restores internal data from stream.
    */
    virtual bool restore(agxStream::StorageStream& in) override;

    AGXSTREAM_DECLARE_SERIALIZABLE(agxPowerLine::TranslationalConnector);

  protected:
    virtual ~TranslationalConnector() {}
  };

  typedef agx::ref_ptr<TranslationalConnector> TranslationalConnectorRef;

  DOXYGEN_END_INTERNAL_BLOCK()
}

#endif // AGXMODEL_TRANSLATIONAL_DIMENSION_H
