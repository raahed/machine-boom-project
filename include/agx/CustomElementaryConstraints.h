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

#ifndef AGX_CUSTOMELEMENTARYCONSTRAINTS_H
#define AGX_CUSTOMELEMENTARYCONSTRAINTS_H

#include <agx/ElementaryConstraint.h>

namespace agx
{
  class AGXPHYSICS_EXPORT ContactNormal : public ElementaryConstraintN< 1 >
  {
    public:
      ContactNormal();

      agx::ContactNormal* set( const agx::Vec3f& n, const agx::Vec3f& v1, const agx::Vec3f& v2, agx::Real c, agx::Real impactSpeed, agx::Real restitution, agx::Real impactCondition = agx::Real( -0.5 ), agx::Real adhesiveOverlap = agx::Real( 0 ) );

      agx::Real getImpactSpeed() const;

      void setImpactSpeed( Real impactSpeed );

      agx::Real getDepth() const;

      void setDepth( agx::Real c );

      agx::Real getAdhesiveOverlap() const;

      void setAdhesiveOverlap( agx::Real adhesiveOverlap );

      const agx::Vec3f& getV1() const;

      void setV1( const agx::Vec3f& v1 );

      const agx::Vec3f& getV2() const;

      void setV2( const agx::Vec3f& v2 );

      const agx::Vec3f& getN()  const;

      void setN( const agx::Vec3f& n );

      AGXSTREAM_DECLARE_SERIALIZABLE(agx::ContactNormal);

    protected:
      virtual ~ContactNormal();

      virtual agx::UInt getJacobian( agx::Jacobian6DOFElement* G, agx::UInt numBlocks, agx::UInt row, agx::GWriteState::Enum writeState ) override;
      virtual agx::UInt getViolation( agx::Real* g, agx::UInt row ) override;

    protected:
      agx::Vec3f m_n;
      agx::Vec3f m_v1;
      agx::Vec3f m_v2;
      agx::Real  m_c;
      agx::Real  m_impactSpeed;
      agx::Real  m_adhesiveOverlap;
  };

  typedef agx::ref_ptr< ContactNormal > ContactNormalRef;

  class AGXPHYSICS_EXPORT ContactTangent : public ElementaryConstraintN< 1 >
  {
    public:
      ContactTangent();

      agx::ContactTangent* set( const agx::Vec3f& dir, const agx::Vec3f& v1, const agx::Vec3f& v2, const agx::Vec3f& velocity );

      const agx::Vec3f& getDirection() const;

      const agx::Vec3f& getV1() const;

      const agx::Vec3f& getV2() const;

      AGXSTREAM_DECLARE_SERIALIZABLE(agx::ContactTangent);

    protected:
      virtual ~ContactTangent();

      virtual agx::UInt getJacobian( agx::Jacobian6DOFElement* G, agx::UInt numBlocks, agx::UInt row, agx::GWriteState::Enum writeState ) override;
      virtual agx::UInt getViolation( agx::Real* g, agx::UInt row ) override;
      virtual agx::UInt getVelocity( agx::Real* v, agx::UInt row ) const override;

    protected:
      agx::Vec3f m_dir;
      agx::Vec3f m_v1;
      agx::Vec3f m_v2;
      agx::Vec3f m_velocity;
  };

  typedef agx::ref_ptr< ContactTangent > ContactTangentRef;
}

#endif
