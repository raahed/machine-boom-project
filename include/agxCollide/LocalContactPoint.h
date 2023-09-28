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

#include <agxCollide/Geometry.h>

#include <agx/Vec3.h>
#include <agx/Vector.h>

#include <agx/Physics/ParticleEntity.h>

#ifdef _MSC_VER
# pragma warning(push)
# pragma warning(disable: 6385) // Disable warning C6385: Reading invalid data
# pragma warning(disable: 6386) // Disable warning C6386: Buffer overrun while writing
#endif

namespace agx
{
  class RigidBody;
  class ContactMaterial;
}

namespace agxCollide
{
  class LocalContactPoint
  {
    public:
      enum ContactPointState : agx::UInt8
      {
        IMPACTING = 1 << 0,  /**< This contact point is impacting - i.e.,
                                 relative speed along the normal is above
                                 some threshold. This state is calculated
                                 by the solver and is updated in the contact
                                 point before the post/last callbacks. */

        NONHOLONOMIC = 1 << 1  /**< This contact point is treated as a non-holonomic
                                    constraint in the normal direction - i.e. a velocity
                                    constraint with target velocity 0 ( similar to friction )
                                    where the contact depth is disregarded. This removes
                                    the right-hand side in the normal equation for the
                                    contact point. */
      };

      /**
      Creates an enabled copy of given \p pointType (agxCollide::ContactPoint).
      \param pointType - source contact point
      */
      template<typename T>
      static LocalContactPoint create( const T& pointType );

    public:
      AGX_FORCE_INLINE LocalContactPoint()
        : m_point()
        , m_normal()
        , m_velocity()
        , m_depth(0)
        , m_maxNormalForce(agx::Infinity)
        , m_area(1)
        , m_elasticRestLengthShape1(1)
        , m_elasticRestLengthShape2(1)
        , m_shape1()
        , m_shape2()
        , m_material(nullptr)
        , m_faceIndex1(0)
        , m_faceIndex2(0)
        , m_faceFeature1(0)
        , m_faceFeature2(0)
        , m_enabled(true)
        , m_state( 0 )
        , m_padding( 0 )
      {
        m_padding = 0; // remove unused variable warning
      }

      AGX_FORCE_INLINE LocalContactPoint(const agx::Vec3& p, const agx::Vec3f& n, agx::Real d)
        : m_point(p)
        , m_normal(n)
        , m_velocity()
        , m_depth(d)
        , m_maxNormalForce(agx::Infinity)
        , m_area(1)
        , m_elasticRestLengthShape1(1)
        , m_elasticRestLengthShape2(1)
        , m_shape1()
        , m_shape2()
        , m_material(nullptr)
        , m_faceIndex1(0)
        , m_faceIndex2(0)
        , m_faceFeature1(0)
        , m_faceFeature2(0)
        , m_enabled(true)
        , m_state( 0 )
        , m_padding( 0 )
      {
        m_padding = 0; // remove unused variable warning
      }

      AGX_FORCE_INLINE LocalContactPoint(const agx::Vec3& p, const agx::Vec3& n, agx::Real d)
        : LocalContactPoint( p, (agx::Vec3f)n, d )
      {
      }
#ifndef SWIG
      AGX_FORCE_INLINE agx::Vec3& point() { return m_point; }
      AGX_FORCE_INLINE agx::Vec3f& normal() { return m_normal; }
      AGX_FORCE_INLINE agx::Vec3f& velocity() { return m_velocity; }
      AGX_FORCE_INLINE agx::Real& depth() { return m_depth; }
      AGX_FORCE_INLINE agx::Real& maxNormalForce() { return m_maxNormalForce; }
      AGX_FORCE_INLINE agx::Real& area() { return m_area; }
      AGX_FORCE_INLINE agx::Real& elasticRestLengthShape1() { return m_elasticRestLengthShape1; }
      AGX_FORCE_INLINE agx::Real& elasticRestLengthShape2() { return m_elasticRestLengthShape2; }
      AGX_FORCE_INLINE agx::UInt32& faceIndex1() { return m_faceIndex1; }
      AGX_FORCE_INLINE agx::UInt32& faceIndex2() { return m_faceIndex2; }
      AGX_FORCE_INLINE agx::UInt8& faceFeature1() { return m_faceFeature1; }
      AGX_FORCE_INLINE agx::UInt8& faceFeature2() { return m_faceFeature2; }
      AGX_FORCE_INLINE agx::Physics::Geometry::ShapePtr& shape1() {return m_shape1;}
      AGX_FORCE_INLINE agx::Physics::Geometry::ShapePtr& shape2() {return m_shape2;}
      AGX_FORCE_INLINE bool& enabled() { return m_enabled; }
      AGX_FORCE_INLINE const agx::ContactMaterial*& material() { return m_material; }
      AGX_FORCE_INLINE agx::UInt8& state() { return m_state; }
#endif
      AGX_FORCE_INLINE void setIsHolonomic( bool isHolonomic )
      {
        if ( isHolonomic )
        {
          m_state = ( agx::UInt8 )( m_state & ~NONHOLONOMIC );
        }
        else
        {
          m_state |= ( agx::UInt8 ) NONHOLONOMIC;
        }
      }

      AGX_FORCE_INLINE agx::Vec3 point() const { return m_point; }
      AGX_FORCE_INLINE agx::Vec3f normal() const { return m_normal; }
      AGX_FORCE_INLINE agx::Vec3f velocity() const { return m_velocity; }
      AGX_FORCE_INLINE agx::Real depth() const { return m_depth; }
      AGX_FORCE_INLINE agx::Real maxNormalForce() const { return m_maxNormalForce; }
      AGX_FORCE_INLINE agx::Real area() const { return m_area; }
      AGX_FORCE_INLINE agx::Real elasticRestLengthShape1() const { return m_elasticRestLengthShape1; }
      AGX_FORCE_INLINE agx::Real elasticRestLengthShape2() const { return m_elasticRestLengthShape2; }
      AGX_FORCE_INLINE agx::UInt32 faceIndex1() const { return m_faceIndex1; }
      AGX_FORCE_INLINE agx::UInt32 faceIndex2() const { return m_faceIndex2; }
      AGX_FORCE_INLINE agx::UInt8 faceFeature1() const { return m_faceFeature1; }
      AGX_FORCE_INLINE agx::UInt8 faceFeature2() const { return m_faceFeature2; }
      AGX_FORCE_INLINE agx::Physics::Geometry::ShapePtr shape1() const {return m_shape1;}
      AGX_FORCE_INLINE agx::Physics::Geometry::ShapePtr shape2() const {return m_shape2;}
      AGX_FORCE_INLINE bool enabled() const { return m_enabled; }
      AGX_FORCE_INLINE const agx::ContactMaterial* material() const { return m_material; }
      AGX_FORCE_INLINE agx::UInt8 state() const { return m_state; }
      AGX_FORCE_INLINE bool isHolonomic() const { return ( m_state & NONHOLONOMIC ) == 0; }

    private:
      agx::Vec3                        m_point;
      agx::Vec3f                       m_normal;
      agx::Vec3f                       m_velocity;
      agx::Real                        m_depth;
      agx::Real                        m_maxNormalForce;
      agx::Real                        m_area;
      agx::Real                        m_elasticRestLengthShape1;
      agx::Real                        m_elasticRestLengthShape2;
      agx::Physics::Geometry::ShapePtr m_shape1;
      agx::Physics::Geometry::ShapePtr m_shape2;
      const agx::ContactMaterial*      m_material;
      agx::UInt32                      m_faceIndex1;
      agx::UInt32                      m_faceIndex2;
      agx::UInt8                       m_faceFeature1;
      agx::UInt8                       m_faceFeature2;
      bool                             m_enabled;
      agx::UInt8                       m_state;
      agx::UInt8                       m_padding;
  };

  template<typename T>
  LocalContactPoint LocalContactPoint::create( const T& pointType )
  {
    LocalContactPoint local( pointType.point(), pointType.normal(), pointType.depth() );
    local.m_maxNormalForce          = pointType.maxNormalForce();
    local.m_velocity                = pointType.velocity();
    local.m_area                    = pointType.area();
    local.m_elasticRestLengthShape1 = pointType.elasticRestLengthShape1();
    local.m_elasticRestLengthShape2 = pointType.elasticRestLengthShape2();
    local.m_shape1                  = pointType.shape1();
    local.m_shape2                  = pointType.shape2();
    local.m_faceIndex1              = pointType.faceIndex1();
    local.m_faceIndex2              = pointType.faceIndex2();
    local.m_faceFeature1            = pointType.faceFeature1();
    local.m_faceFeature2            = pointType.faceFeature2();
    local.m_state                   = pointType.state();
    return local;
  }

  typedef agx::VectorPOD<LocalContactPoint> LocalContactPointVector;

  class LocalGeometryContact
  {
    public:
      typedef LocalContactPoint PointType;

    public:
      AGX_FORCE_INLINE LocalGeometryContact() {
        m_geometries[0]       = nullptr;
        m_geometries[1]       = nullptr;
        m_bodies[0]           = nullptr;
        m_bodies[1]           = nullptr;
        m_contactMaterial     = nullptr;
        m_hasInternalMaterial = false;
        m_hasSurfaceVelocity  = false;
      }

      AGX_FORCE_INLINE LocalGeometryContact(Geometry *g1, Geometry *g2) {
        m_geometries[0]       = g1;
        m_geometries[1]       = g2;
        if (m_geometries[0])
          m_bodies[0]         = m_geometries[0]->getRigidBody();
        if (m_geometries[1])
          m_bodies[1]         = m_geometries[1]->getRigidBody();
        m_contactMaterial     = nullptr;
        m_hasInternalMaterial = false;
        m_hasSurfaceVelocity  = false;
      }

      AGX_FORCE_INLINE LocalGeometryContact( Geometry *g1, Geometry *g2, agx::RigidBody* b1, agx::RigidBody* b2 ) {
        m_geometries[0]       = g1;
        m_geometries[1]       = g2;
        m_bodies[0]           = b1;
        m_bodies[1]           = b2;
        m_contactMaterial     = nullptr;
        m_hasInternalMaterial = false;
        m_hasSurfaceVelocity  = false;
      }

      AGX_FORCE_INLINE void init(Geometry *g1, Geometry *g2) {
        m_geometries[0] = g1;
        m_geometries[1] = g2;
        if (m_geometries[0])
          m_bodies[0]   = m_geometries[0]->getRigidBody();
        if (m_geometries[1])
          m_bodies[1]   = m_geometries[1]->getRigidBody();
        m_points.clear();
      }

      AGX_FORCE_INLINE void init(Geometry *g1, Geometry *g2, agx::RigidBody* b1, agx::RigidBody* b2) {
        m_geometries[0] = g1;
        m_geometries[1] = g2;
        m_bodies[0]     = b1;
        m_bodies[1]     = b2;
        m_points.clear();
      }

      AGX_FORCE_INLINE void setGeometry( size_t ith, Geometry *geom ) { agxAssert(ith <2 ); m_geometries[ith] = geom; }
      AGX_FORCE_INLINE void setRigidBody( size_t ith, agx::RigidBody* body ) { agxAssert(ith <2 ); m_bodies[ith] = body; }
      AGX_FORCE_INLINE void setContactMaterial(agx::ContactMaterial *material) { m_contactMaterial = material; }
      AGX_FORCE_INLINE void setHasInternalMaterial(const agx::Bool hasInternalMaterial) { m_hasInternalMaterial = hasInternalMaterial; }
      AGX_FORCE_INLINE void setHasSurfaceVelocity(const agx::Bool hasSurfaceVelocity) { m_hasSurfaceVelocity = hasSurfaceVelocity; }

      AGX_FORCE_INLINE void addPoint(const LocalContactPoint& point) { m_points.push_back(point); }
      AGX_FORCE_INLINE LocalContactPointVector& points() { return m_points; }
      AGX_FORCE_INLINE const LocalContactPointVector& points() const { return m_points; }

      AGX_FORCE_INLINE agxCollide::Geometry* geometry( size_t ith ) { agxAssert(ith <2 ); return m_geometries[ith]; }
      AGX_FORCE_INLINE const agxCollide::Geometry* geometry( size_t ith ) const { agxAssert(ith <2 ); return m_geometries[ith]; }

      AGX_FORCE_INLINE agx::RigidBody* rigidBody( size_t ith ) { agxAssert(ith <2 ); return m_bodies[ith]; }
      AGX_FORCE_INLINE const agx::RigidBody* rigidBody( size_t ith ) const { agxAssert(ith <2 ); return m_bodies[ith]; }

      AGX_FORCE_INLINE agx::ContactMaterial *material() { return m_contactMaterial; }
      AGX_FORCE_INLINE const agx::ContactMaterial *material() const { return m_contactMaterial; }

      AGX_FORCE_INLINE agx::Bool& hasInternalMaterial() { return m_hasInternalMaterial; }
      AGX_FORCE_INLINE const agx::Bool& hasInternalMaterial() const { return m_hasInternalMaterial; }

      AGX_FORCE_INLINE agx::Bool& getHasSurfaceVelocity() { return m_hasSurfaceVelocity; }
      AGX_FORCE_INLINE const agx::Bool& getHasSurfaceVelocity() const { return m_hasSurfaceVelocity; }


    private:
      Geometry*               m_geometries[2];
      agx::RigidBody*         m_bodies[2];
      LocalContactPointVector m_points;
      agx::ContactMaterial*   m_contactMaterial;
      agx::Bool               m_hasInternalMaterial;
      agx::Bool               m_hasSurfaceVelocity;
  };

  typedef agx::Vector<LocalGeometryContact> LocalGeometryContactVector;

  class LocalParticleGeometryContact : public LocalContactPoint
  {
  public:
    AGX_FORCE_INLINE LocalParticleGeometryContact() : LocalContactPoint()
    {
    }

    AGX_FORCE_INLINE LocalParticleGeometryContact(agx::Physics::ParticlePtr particle, agx::Physics::GeometryPtr geometry) : LocalContactPoint(), m_particle(particle), m_geometry(geometry)
    {
    }

    AGX_FORCE_INLINE const agx::Physics::GeometryPtr& geometry() const { return m_geometry; }
    AGX_FORCE_INLINE agx::Physics::GeometryPtr& geometry() { return m_geometry; }

    AGX_FORCE_INLINE const agx::Physics::ParticlePtr& particle() const { return m_particle; }
    AGX_FORCE_INLINE agx::Physics::ParticlePtr& particle() { return m_particle; }

  private:
    agx::Physics::ParticlePtr m_particle;
    agx::Physics::GeometryPtr m_geometry;
  };

  typedef agx::Vector<LocalParticleGeometryContact> LocalParticleGeometryContactVector;

  class LocalParticlePairContact : public LocalContactPoint
  {
  public:
    AGX_FORCE_INLINE LocalParticlePairContact() : LocalContactPoint()
    {
    }

    AGX_FORCE_INLINE LocalParticlePairContact(agx::Physics::ParticlePtr p1, agx::Physics::ParticlePtr p2) : LocalContactPoint(), m_particle1(p1), m_particle2(p2)
    {
    }

    AGX_FORCE_INLINE const agx::Physics::ParticlePtr& particle1() const { return m_particle1; }
    AGX_FORCE_INLINE agx::Physics::ParticlePtr& particle1() { return m_particle1; }


    AGX_FORCE_INLINE const agx::Physics::ParticlePtr& particle2() const { return m_particle2; }
    AGX_FORCE_INLINE agx::Physics::ParticlePtr& particle2() { return m_particle2; }

  private:
    agx::Physics::ParticlePtr m_particle1;
    agx::Physics::ParticlePtr m_particle2;
  };

  typedef agx::Vector<LocalParticlePairContact> LocalParticlePairContactVector;
}

#ifdef _MSC_VER
# pragma warning(pop)
#endif
