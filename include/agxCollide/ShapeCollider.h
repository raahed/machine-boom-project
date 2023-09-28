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

#ifndef AGXCOLLIDE_SHAPECOLLIDER_H
#define AGXCOLLIDE_SHAPECOLLIDER_H

#include <agx/macros.h>

DOXYGEN_START_INTERNAL_BLOCK()

#define SHAPE_COLLIDER_VELOCITIES 0

#include <agx/agx.h>
#include <agx/agxPhysics_export.h>
#include <agxCollide/PluginManager.h>

#include <agx/AffineMatrix4x4.h>
#include <agx/SymmetricPair.h>

#include <agxCollide/LocalContactPoint.h>
#include <agxCollide/Shape.h>

#include <agx/Thread.h>
#include <agxData/LocalVector.h>
#include <agx/Physics/Geometry/ShapeEntity.h>

#ifdef _MSC_VER
# pragma warning(push)
# pragma warning( disable : 4251 ) // class X needs to have dll-interface to be used by clients of class Y
#endif


namespace agxCollide
{
  /**
  Interface that must be implemented by all colliders
  */
  class AGXPHYSICS_EXPORT ShapeCollider
  {
  public:
    typedef agxData::LocalVector<LocalContactPoint> LocalContactPointVector;

  public:
    virtual ~ShapeCollider() {}

    typedef agx::SymmetricPair<agx::UInt8> TypePair;
    TypePair getDescriptor() const;


    void calculateContacts(
        agx::Physics::Geometry::ShapeInstance shape1, agx::Physics::Geometry::ShapeInstance shape2,
        const agx::AffineMatrix4x4& transform1, const agx::AffineMatrix4x4& transform2,
        const agx::Vec3& linearVelocity1, const agx::Vec3& linearVelocity2,
        const agx::Vec3& angularVelocity1, const agx::Vec3& angularVelocity2,
        bool earlyOut, LocalContactPointVector& result);

    void calculateContacts(
      agx::Physics::Geometry::ShapeInstance shape1, agx::Physics::Geometry::ShapeInstance shape2,
      const agx::AffineMatrix4x4& transform1, const agx::AffineMatrix4x4& transform2,
      bool earlyOut, LocalContactPointVector& result);


    /* Convenience wrappers */
    void calculateContacts(agxCollide::Shape *shape1, agxCollide::Shape *shape2,
    const agx::AffineMatrix4x4& transform1, const agx::AffineMatrix4x4& transform2,
    const agx::Vec3& linearVelocity1, const agx::Vec3& linearVelocity2,
    const agx::Vec3& angularVelocity1, const agx::Vec3& angularVelocity2,
    LocalContactPointVector& result);

    void calculateContacts(agxCollide::Shape *shape1, agxCollide::Shape *shape2,
    const agx::AffineMatrix4x4& transform1, const agx::AffineMatrix4x4& transform2,
    LocalContactPointVector& result);

    static void addContactPoint(LocalContactPointVector& result, const agx::Vec3& point,
      const agx::Vec3& normal, agx::Real depth);

    /// Add a disabled contact point with zero point, normal and depth. Used to
    /// mark that a collider did an early out.
    static void addEarlyOutMarkerPoint(LocalContactPointVector& result);


  protected:
    ShapeCollider(agx::UInt8 shapeType1, agx::UInt8 shapeType2) : m_descriptor(shapeType1, shapeType2) {}

    virtual void _calculateContacts(
        agx::Physics::Geometry::ShapeInstance shape1, agx::Physics::Geometry::ShapeInstance shape2,
        const agx::AffineMatrix4x4& transform1, const agx::AffineMatrix4x4& transform2,
        const agx::Vec3& linearVelocity1, const agx::Vec3& linearVelocity2,
        const agx::Vec3& angularVelocity1, const agx::Vec3& angularVelocity2,
        bool earlyOut, LocalContactPointVector& result) = 0;


  private:
    TypePair m_descriptor;
  };

  class AGXPHYSICS_EXPORT ShapeColliderPlugin
  {
  public:
    virtual agxCollide::ShapeCollider* create() = 0;
    virtual void destroy(agxCollide::ShapeCollider *collider) = 0;
  };

  template <typename T>
  class ShapeColliderPluginT : public ShapeColliderPlugin
  {
  public:
    ShapeColliderPluginT()
    {
      PluginManager::instance()->registerPlugin(this);
    }

    virtual agxCollide::ShapeCollider* create() override
    {
      return new T();
    }

    virtual void destroy(agxCollide::ShapeCollider *collider) override
    {
      agxVerify(dynamic_cast<T *>(collider));
      delete collider;
    }

  private:
  };

  #define AGX_SHAPE_COLLIDER(T) static agxCollide::ShapeColliderPluginT<T> s_colliderPlugin;



  /* Implementation */
  AGX_FORCE_INLINE ShapeCollider::TypePair ShapeCollider::getDescriptor() const
  {
    return m_descriptor;
  }



  AGX_FORCE_INLINE void ShapeCollider::calculateContacts(
      agx::Physics::Geometry::ShapeInstance shape1, agx::Physics::Geometry::ShapeInstance shape2,
      const agx::AffineMatrix4x4& transform1, const agx::AffineMatrix4x4& transform2,
      bool earlyOut, LocalContactPointVector& result)
  {
    agx::Vec3 foo;
    this->calculateContacts(shape1, shape2, transform1, transform2, foo, foo, foo, foo, earlyOut, result);
  }



  /// \todo Consider adding earlyOut to these convenience wrappers for calculateContacts.


  AGX_FORCE_INLINE void ShapeCollider::calculateContacts(
      agxCollide::Shape* shape1, agxCollide::Shape* shape2,
      const agx::AffineMatrix4x4& transform1, const agx::AffineMatrix4x4& transform2,
      LocalContactPointVector& result)
  {
    this->calculateContacts(shape1->getEntity(), shape2->getEntity(), transform1, transform2, false, result);
  }



  AGX_FORCE_INLINE void ShapeCollider::calculateContacts(
      agxCollide::Shape* shape1, agxCollide::Shape* shape2,
      const agx::AffineMatrix4x4& transform1, const agx::AffineMatrix4x4& transform2,
      const agx::Vec3& linearVelocity1, const agx::Vec3& linearVelocity2,
      const agx::Vec3& angularVelocity1, const agx::Vec3& angularVelocity2,
      LocalContactPointVector& result)
  {
    this->calculateContacts(
        shape1->getEntity(), shape2->getEntity(), transform1, transform2, linearVelocity1, linearVelocity2,
        angularVelocity1, angularVelocity2, false, result);
  }



  AGX_FORCE_INLINE void ShapeCollider::addContactPoint(
      LocalContactPointVector& result, const agx::Vec3& point,
      const agx::Vec3& normal, agx::Real depth)
  {
    agxAssert(agx::equivalent(normal.length2(), agx::Real(1), agx::Real(1.0e-4)));
    agxAssert(depth >= agx::Real(0) && depth != agx::Infinity);
    result.push_back(LocalContactPoint(point, (agx::Vec3f) normal, depth ) );
  }



  inline void ShapeCollider::addEarlyOutMarkerPoint(LocalContactPointVector& result)
  {
    result.push_back(LocalContactPoint(agx::Vec3(agx::Real(0.0)), agx::Vec3(agx::Real(0.0)), agx::Real(0.0)));
    result.back().enabled() = false;
  }
}

#ifdef _MSC_VER
# pragma warning(pop)
#endif

DOXYGEN_END_INTERNAL_BLOCK()
#endif

