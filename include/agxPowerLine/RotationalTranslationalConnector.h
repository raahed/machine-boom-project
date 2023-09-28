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

#ifndef AGXPOWERLINE_ROTATIONAL_TRANSLATIONAL_CONNECTOR_H
#define AGXPOWERLINE_ROTATIONAL_TRANSLATIONAL_CONNECTOR_H

#include <agxPowerLine/PowerLine.h>
#include <agxPowerLine/Connector.h>
#include <agxPowerLine/RotationalTranslationalConstraints.h>

namespace agxPowerLine
{

  /**
  Constrains a rotational degree of freedom with a translational degree of freedom
  with a non-holonomic constraint.

  The gearing depends on the shaft radius.
  If the shaft radius is set to 1,
  the angular velocity of the rotational degree of freedom will be constrained to be equal
  to the linear velocity of the translational degree of freedom.
  */
  class AGXMODEL_EXPORT RotationalTranslationalConnector : public agxPowerLine::Connector
  {
    public:
      /**
      Create a connector connecting a RotationalDimension with a TranslationalDimension.
      */
      RotationalTranslationalConnector();

      virtual bool postUpdate(agx::Real timeStep) override;

      virtual agx::RegularizationParameters::VariableType calculateComplianceAndDamping(
        const agx::Real timeStep, agx::Real& compliance, agx::Real& damping) override;

      using Connector::connect;

      /**
      Create the constraint.
      \return the created constraint
      */
      virtual agxPowerLine::PhysicalDimensionMultiBodyConstraintImplementation* createConstraint() override;

      /**
      Set the radius of the shaft.
      */
      void setShaftRadius(agx::Real shaftRadius);
      virtual agx::Real getShaftRadius() const;

      agx::Real getShaftAngle() const;
      agx::Real getRodPosition() const;

      /**
      Stores internal data into stream.
      */
      virtual bool store(agxStream::StorageStream& str) const override;

      /**
      Restores internal data from stream.
      */
      virtual bool restore(agxStream::StorageStream& str) override;

      AGXSTREAM_DECLARE_SERIALIZABLE( agxPowerLine::RotationalTranslationalConnector );

    protected:
      virtual ~RotationalTranslationalConnector() {}
#ifndef SWIGPYTHON
      virtual bool connect( PhysicalDimension* inputDimension, Side inputSide, Side outputSide, PhysicalDimension* outputDimension ) override;
#endif
    protected:
      agx::Real m_shaftRadius;
      agx::Real m_shaftAngle;
      agx::Real m_rodPosition;
  };

  typedef agx::ref_ptr< RotationalTranslationalConnector > RotationalTranslationalConnectorRef;



  /**
  This class can be used to connect the position of a rotational and a positional unit.
  Typically between a rotating shaft and a linear winch/prismatic constraint.
  */
  class AGXMODEL_EXPORT RotationalTranslationalHolonomicConnector : public RotationalTranslationalConnector
  {
  public:
    /**
    Create a connector connecting a RotationalDimension with a TranslationalDimension.
    */
    RotationalTranslationalHolonomicConnector( );

    virtual agx::RegularizationParameters::VariableType calculateComplianceAndDamping( const agx::Real timeStep, agx::Real& compliance, agx::Real& damping ) override;

    virtual agx::Real calculateViolation( ) const override;

    /**
    Stores internal data into stream.
    */
    virtual bool store( agxStream::StorageStream& str ) const override;

    /**
    Restores internal data from stream.
    */
    virtual bool restore( agxStream::StorageStream& str ) override;

    AGXSTREAM_DECLARE_SERIALIZABLE( agxPowerLine::RotationalTranslationalHolonomicConnector );
  };

  typedef agx::ref_ptr< RotationalTranslationalHolonomicConnector > RotationalTranslationalHolonomicConnectorRef;

}

#endif
