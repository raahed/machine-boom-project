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

#ifdef _MSC_VER
# pragma warning(push)
# pragma warning( disable : 4275 ) //  warning C4275: non dll-interface class
#endif

#include <agx/agxPhysics_export.h>

#include <agxCollide/agxCollide.h>
#include <agx/agx.h>
#include <agx/Referenced.h>
#include <agx/RigidBody.h>
#include <agx/Vec3.h>
#include <agxCollide/GeometryPair.h>

#include <agx/Material.h>
#include <agx/Vector.h>
#include <agx/Physics/GeometryEntity.h>
#include <agx/Physics/ContactPointEntity.h>
#include <agx/Physics/GeometryContactEntity.h>
#include <agx/Physics/BroadPhasePairEntity.h>
#include <agxCollide/LocalContactPoint.h>


namespace agxSDK
{
  class MaterialManager;
}

namespace agxCollide
{
  class Geometry;


  /**
  A contact point in a contact data.
  */
  class AGXPHYSICS_EXPORT ContactPoint : public agx::Physics::ContactPointPtr
  {
    public:
      enum ContactForceComponents
      {
        NORMAL_FORCE = 0,     /**< Normal force. */
        TANGENTIAL_FORCE_U,   /**< Tangential force U, orthogonal to the normal force and V. */
        TANGENTIAL_FORCE_V    /**< Tangential force V, orthogonal to the normal force and U. */
      };

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

    public:
      /**
      Default constructor.
      */
      ContactPoint();

      /**
      \internal

      Construct given entity pointer (this base class).
      */
      ContactPoint(agx::Physics::ContactPointPtr contact);

      /**
      Construct given point, normal and depth.
      \param thePoint - point in world
      \param theNormal - normalized normal in world
      \param theDepth - contact point depth
      */
      ContactPoint(agx::Vec3 thePoint, agx::Vec3f theNormal, agx::Real theDepth);

      /// Clear all values in this contact
      void clear();

      using agx::Physics::ContactPointPtr::localForce;

      /**
      \param idx - Specifies which direction (see enum ContactForceComponents)
      \return a reference to the (signed) local force (in coordinate system n, u, v)
      for the selected component.
      */
      agx::Real& localForce( size_t idx );

      /**
      \param idx - Specifies which direction (see enum ContactForceComponents)
      \return the (signed) local force (in coordinate system n, u, v)
      for the selected component.
      */
      agx::Real localForce( size_t idx ) const;

      /**
      \return The complete (signed) contact force, including both normal and tangential (friction),
      in the world coordinate system.
      */
      agx::Vec3 getForce() const;

      /**
      \return The complete (unsigned) contact force magnitude, including both normal and tangential (friction).
      */
      agx::Real getForceMagnitude() const;

      /**
      \return The (signed) normal force in world coordinates.
      */
      agx::Vec3 getNormalForce() const;

      /**
      \return the (unsigned) magnitude of the normal force
      */
      agx::Real getNormalForceMagnitude() const;

      /**
      \return the (unsigned) magnitude of the tangential force in U-direction (friction force )
      */
      agx::Real getTangentialForceUMagnitude() const;

      /**
      \return the (unsigned) magnitude of the tangential force in U-direction (friction force )
      */
      agx::Real getTangentialForceVMagnitude() const;

      /**
      \return the (unsigned) magnitude of the tangential force (friction force in both U and V)
      */
      agx::Real getTangentialForceMagnitude() const;

      /**
      \return The (signed) tangential force (friction force) in world coordinates.
      */
      agx::Vec3 getTangentialForce() const;

      /**
      \param ith - 0 for first, 1 for second, index > 1 will return reference to second
      \return reference to the ith face index.
      */
      agx::UInt32& faceIndex(size_t ith);

      /**
      \param ith - 0 for first, 1 for second, index > 1 will return reference to second
      \return const reference to the ith face index.
      */
      agx::UInt32 const& faceIndex(size_t ith) const;

      /**
      \param ith - 0 for first, 1 for second, index > 1 will return reference to second
      \return reference to the ith face feature.
      */
      agx::UInt8& faceFeature(size_t ith);

      /**
      \param ith - 0 for first, 1 for second, index > 1 will return reference to second
      \return const reference to the ith face feature.
      */
      agx::UInt8 const& faceFeature(size_t ith) const;

      /**
      \param ith - 0 for first, 1 for second, index > 1 will return reference to second
      \return  reference to the ith shape.
      */
      agx::Physics::Geometry::ShapePtr& shape(size_t ith);

      /**
      \param ith - 0 for first, 1 for second, index > 1 will return reference to second
      \return const reference to the ith shape.
      */
      agx::Physics::Geometry::ShapePtr const& shape(size_t ith) const;

      /**
      Get witness point for ith geometry.
      The point is in world coordinates, and at time of collision detection.
      So it is valid in a PRE-StepEventListener or a ContactEventListener, but e.g. in a
      POST-StepEventListener the geometries will most likely have moved from the contact.
      \param ith - 0 for first, 1 for second, index > 1 will return value for second
      \return witness point for ith geometry, in world coordinates at collision detection time.
      */
      agx::Vec3 getWitnessPoint(size_t ith) const;

      /**
      \return true if this contact point has given state
      */
      agx::Bool hasState( ContactPointState contactPointState ) const;

      /**
      Set if contact point should be treated as a holonomic constraint ( regular contact constraint with depth )
      or a non-holonomic constraint ( strictly dependent on velocity ).
      \param isHolonomic - true if contact should be treated as a holonomic constraint, false if it should be treated as
                           a non-holonomic constraint. ( Default: true )
      */
      void setIsHolonomic( agx::Bool isHolonomic );

      /**
      \return true if this contact point is treated as a holonomic constraint,
              non-holonomic ( velocity only, no depth/violation ) if false.
      */
      agx::Bool isHolonomic() const;
  };


  /**
  Copy between the different kinds of contact points.
  Note that only the common members between all 3 classes
  are copied (so not e.g. material or localForce).
  */
  template<typename T1, typename T2>
  void copyContactPoint(const T1& from, T2& to)
  {
    to.point() = from.point();
    to.normal() = from.normal();
    to.depth() = from.depth();
    to.area() = from.area();

    to.velocity() = from.velocity();
    to.enabled() = from.enabled();

    to.shape1() = from.shape1();
    to.shape2() = from.shape2();

    to.faceIndex1() = from.faceIndex1();
    to.faceIndex2() = from.faceIndex2();
    to.faceFeature1() = from.faceFeature1();
    to.faceFeature2() = from.faceFeature2();
    to.state() = from.state();
  }

#ifndef SWIG
  template void copyContactPoint(const ContactPoint&, LocalContactPoint&);
  template void copyContactPoint(const LocalContactPoint&, ContactPoint&);
  template void copyContactPoint(const LocalContactPoint&, agx::Physics::ContactPointPtr&);
  template void copyContactPoint(const agx::Physics::ContactPointPtr&, LocalContactPoint&);


  AGX_STATIC_ASSERT(sizeof(ContactPoint) == sizeof(agx::Physics::ContactPointPtr));
#endif
  typedef agxData::Array<ContactPoint> ContactPointVector;

  /**
  A contact between two geometries.
  */
  class AGXPHYSICS_EXPORT GeometryContact : public agx::Physics::GeometryContactPtr
  {
    public:
      typedef ContactPoint PointType;

    public:
      /// Specify the state of the contact
      enum ContactState {
        NO_CONTACT       = 0, ///<! No contact yet
        IMPACT_STATE     = 1, ///<! First contact
        CONTACT_STATE    = 2, ///<! Continuous contact
        SEPARATION_STATE = 3  ///<! Contact in last time step, now separated
      };

    public:
      /**
      Default constructor.
      */
      GeometryContact();

      /**
      Construct given geometry contact entity data.
      \param contact - other geometry (to copy from)
      */
      GeometryContact( agx::Physics::GeometryContactPtr contact );

      /**
      Clear contact points.
      */
      void clear();

      /**
      Calculate the relative velocity in a contact point.
      \param pointIndex - The contact point index in the points array containing
      all contact points for this pair of geometries.

      \return relative velocity in world coordinate system.
      */
      agx::Vec3f calculateRelativeVelocity( size_t pointIndex = 0 ) const;

      /**
      Calculate the relative velocity in a contact point between \p geometry1 and \p geometry2
      Put result in \p p

      \return false if none of the geometries have surface velocity enabled. Returns false if the
      resulting velocity.equalsZero()
      */
      static bool calculateSurfaceVelocity( const agxCollide::Geometry* geometry1, const agxCollide::Geometry* geometry2, LocalContactPoint& p );

      /**
      Calculate the relative velocity in a contact point between \p geometry1 and \p geometry2
      Put result in \p p

      \return false if none of the geometries have surface velocity enabled. Returns false if the
      resulting velocity.equalsZero()
      */
      static bool calculateSurfaceVelocity( const agxCollide::Geometry* geometry1, const agxCollide::Geometry* geometry2, ContactPoint& p );

      /**
      \param ith - index of geometry, 0 or 1.
      \return pointer to the ith geometry
      */
      agxCollide::Geometry* geometry( size_t ith );

      /**
      \param ith - index of geometry, 0 or 1.
      \return const pointer to the ith geometry
      */
      const agxCollide::Geometry* geometry( size_t ith ) const;

      /**
      Returns access to rigid body data.
      \param ith - index of RigidBody, 0 or 1.
      \return pointer to the ith RigidBody high level data
      */
      agx::RigidBody* rigidBody( size_t ith );

      /**
      Returns access to rigid body data.
      \param ith - index of RigidBody, 0 or 1.
      \return const pointer to the ith RigidBody high level data
      */
      const agx::RigidBody* rigidBody( size_t ith ) const;

      /// \return a reference to the contact points
      ContactPointVector& points();

      /// \return a const reference to the contact points
      const ContactPointVector& points() const;

      ///\return pointer to custom data (with one data element for each contact point)
      void* getCustomData();

      ///\return const pointer to custom data (with one data element for each contact point)
      const void* getCustomData() const;

      /**
      Store custom data for this Contact
      \param customData - data to be stored
      */
      void setCustomData( void* customData );

      /**
      If the contact material for this geometry contact is explicit it can be accessed here. If the contact material
      in this geometry contact is implicit, this method will return 0, and you have to create an explicit contact material
      with help of agxSDK::MaterialManager::getOrCreateContactMaterial.
      \return the explicit contact material if present
      */
      agx::ContactMaterial* getMaterial();

      /**
      If the contact material for this geometry contact is explicit it can be accessed here. If the contact material
      in this geometry contact is implicit, this method will return 0, and you have to create an explicit contact material
      with help of agxSDK::MaterialManager::getOrCreateContactMaterial.
      \return the explicit contact material if present
      */
      const agx::ContactMaterial* getMaterial() const;

      /**
      It's not possible to change parameters in an implicit contact material. This method returns any type of contact material
      but const. It's recommended to use the getMaterial() method, this method is used internally.
      \return the explicit or implicit contact material if present
      */
      const agx::ContactMaterial* getMaterial( bool ) const;

      /// \return the current  of this geometry contact (CONTACT, IMPACT)
      agx::UInt8 state() const;

      /// remove the specified ContactPoint from the set of contact points.
      void removeContactPoint( ContactPoint& point );

      /**
      Set to false to disable the contact.
      */
      void setEnable(bool flag);

      /**
      Will return -1 if body does not exist in this GeometryContact, 0 if it is the first (index 0) and
      1 if it is the second (index 1).
      This index (if > -1) can be used to get the rigid body with a call to rigidBody(idx)

      \param body - the rigid body that you want to look for in this geometry contact
      \returns -1 if \p body does not exist in this GeometryContact, 0 if it is the first and 1 if it is the second
      */
      int contains(agx::RigidBody * body) const;

      /**
      Will return -1 if geometry does not exist in this GeometryContact, 0 if it is the first (index 0) and
      1 if it is the second (index 1).
      This index (if > -1) can be used to get the geometry with a call to geometry(idx)

      \param geometry - The geometry that you want to look for in this geometry contact
      \returns -1 if \p geometry does not exist in this GeometryContact, 0 if it is the first and 1 if it is the second
      */
      int contains(agxCollide::Geometry * geometry) const;

      /**
      \return True if the contact is enabled.
      */
      bool isEnabled() const;

      /**
      \note This state is calculated for the points during solve so the result
            is only reliable after solve and before next collision detection.
      \return true if this geometry contact contains at least one contact point
              with state IMPACTING - otherwise false
      */
      agx::Bool hasImpactingPoints() const;


      DOXYGEN_START_INTERNAL_BLOCK()


      /**
      Set the ith geometry of this contact
      \param ith - 0/1
      \param geom - The geometry that will be stored
      */
      void setGeometry( size_t ith, Geometry* geom );

      /***
    When called, this contact should be removed before entering the solver
      NOTE: Contacts are never removed, only marked as disabled
    \param immediately - Should the contact be removed immediately
    */
      void markForRemoval( bool immediately );

      /**
      If markForRemoval( true ), is called, it means that this contact was immediately removed, and should not be used in
      contact listeners anymore.
      \return true if this contact should be triggered by listeners.
      */
      bool shouldBeCalled() const;

      /**
      Only for internal use. Assign contact material to this geometry contact.
      */
      void setMaterial( agx::ContactMaterial* material );

      void setHasSurfaceVelocity( bool flag );

      GeometryPair getGeometryPair(); /// \todo: API change during low-level mapping here. Used to return a GeometryPair&.

      void setHasInternalMaterial( bool flag );
      bool getHasInternalMaterial( ) const;

      void setState( agx::UInt8 state );
      DOXYGEN_END_INTERNAL_BLOCK()
  };



  inline int GeometryContact::contains(agx::RigidBody * body) const
  {
    if (this->rigidBody(0) == body) return 0;
    if (this->rigidBody(1) == body) return 1;
    return -1;
  }

  inline int GeometryContact::contains(agxCollide::Geometry * geometry) const
  {
    if (this->geometry(0) == geometry) return 0;
    if (this->geometry(1) == geometry) return 1;
    return -1;
  }



#ifndef SWIG
  //AGX_STATIC_ASSERT(sizeof(GeometryContact) == sizeof(agx::Physics::GeometryContactPtr));
#endif

  /* Implementation */
  inline ContactPoint::ContactPoint() : agx::Physics::ContactPointPtr( agx::Physics::ContactPointModel::createInstance() )
  {
  }

  inline ContactPoint::ContactPoint(agx::Physics::ContactPointPtr contact) : agx::Physics::ContactPointPtr(contact)
  {
  }


  AGX_FORCE_INLINE void ContactPoint::clear()
  {
    depth() = agx::Real( 0 );
    enabled() = true;
    point().set(0, 0, 0);
    normal().set(0, 0, 0);
    state() = agx::UInt8( 0 );
    tangentU().set(0, 0, 0);
    tangentV().set(0, 0, 0);
    localForce().set( 0, 0, 0 );
    velocity().set(0, 0, 0);
    faceIndex1() = 0;
    faceIndex2() = 0;
    faceFeature1() = 0;
    faceFeature2() = 0;
    area() = agx::Real( 0 );
    shape1() = agx::Physics::Geometry::ShapePtr();
    shape2() = agx::Physics::Geometry::ShapePtr();
  }


  AGX_FORCE_INLINE agx::Real& ContactPoint::localForce( size_t idx )
  {
    agxAssert(idx < 3);
    return localForce()[idx];
  }


  AGX_FORCE_INLINE agx::Real ContactPoint::localForce( size_t idx ) const
  {
    agxAssert(idx < 3);
    return localForce()[idx];
  }



  AGX_FORCE_INLINE agx::Vec3 ContactPoint::getForce() const
  {
    return getNormalForce() + getTangentialForce();
  }


  AGX_FORCE_INLINE agx::Real ContactPoint::getForceMagnitude() const
  {
    agx::Real sqr =
      localForce()[ NORMAL_FORCE ] * localForce()[ NORMAL_FORCE ] +
      localForce()[ TANGENTIAL_FORCE_U ] * localForce()[ TANGENTIAL_FORCE_U ] +
      localForce()[ TANGENTIAL_FORCE_V ] * localForce()[ TANGENTIAL_FORCE_V ];
    return sqr > 0 ? std::sqrt( sqr ) : 0;
  }


  AGX_FORCE_INLINE agx::Vec3 ContactPoint::getNormalForce() const
  {
    return agx::Vec3(normal()) * localForce()[ NORMAL_FORCE ];
  }


  AGX_FORCE_INLINE agx::Real ContactPoint::getNormalForceMagnitude() const
  {
    return std::abs(localForce()[ NORMAL_FORCE ]);
  }


  AGX_FORCE_INLINE agx::Vec3 ContactPoint::getTangentialForce() const
  {
    const agx::Vec3 forceU = agx::Vec3(tangentU()) * localForce()[ TANGENTIAL_FORCE_U ];
    const agx::Vec3 forceV = agx::Vec3(tangentV()) * localForce()[ TANGENTIAL_FORCE_V ];
    return forceU + forceV;
  }


  AGX_FORCE_INLINE agx::Real ContactPoint::getTangentialForceUMagnitude() const
  {
    return std::abs(localForce()[ TANGENTIAL_FORCE_U ]);
  }


  AGX_FORCE_INLINE agx::Real ContactPoint::getTangentialForceVMagnitude() const
  {
    return std::abs(localForce()[ TANGENTIAL_FORCE_V ]);
  }


  AGX_FORCE_INLINE agx::Real ContactPoint::getTangentialForceMagnitude() const
  {
    agx::Real sqr = localForce()[ TANGENTIAL_FORCE_U ] * localForce()[ TANGENTIAL_FORCE_U ] + localForce()[ TANGENTIAL_FORCE_V ] * localForce()[ TANGENTIAL_FORCE_V ];
    return sqr > 0 ? std::sqrt( sqr ) : 0;
  }

  inline agx::Bool ContactPoint::hasState( ContactPoint::ContactPointState contactPointState ) const
  {
    return ( state() & contactPointState ) != 0;
  }

  AGX_FORCE_INLINE agx::Bool ContactPoint::isHolonomic() const
  {
    return !hasState( NONHOLONOMIC );
  }

  AGX_FORCE_INLINE void ContactPoint::setIsHolonomic( agx::Bool isHolonomic )
  {
    if ( isHolonomic )
    {
      // remove non-holonomic bit
      state() = (agx::UInt8)( state() & ~NONHOLONOMIC );
    }
    else
    {
      // add non-holonomic bit
      state() |= ( agx::UInt8 )( NONHOLONOMIC );
    }
  }

  AGX_FORCE_INLINE  GeometryContact::GeometryContact()
  {

  }

  AGX_FORCE_INLINE  GeometryContact::GeometryContact(agx::Physics::GeometryContactPtr contact) :
    agx::Physics::GeometryContactPtr(contact)

  {
  }

  AGX_FORCE_INLINE agxCollide::Geometry* GeometryContact::geometry( size_t ith )
  {
    agxAssert(ith < 2);
    if(ith == 0)
      return geometry1().model();
    else
      return geometry2().model();
  }

  AGX_FORCE_INLINE const agxCollide::Geometry* GeometryContact::geometry( size_t ith ) const
  {
    agxAssert(ith < 2);
    if(ith == 0)
      return geometry1().model();
    else
      return geometry2().model();
  }

  AGX_FORCE_INLINE GeometryPair GeometryContact::getGeometryPair()
  {
    return GeometryPair(geometry(0), geometry(1));
  }

  AGX_FORCE_INLINE void GeometryContact::setGeometry( size_t ith, Geometry* geom )
  {
    agxAssert(ith < 2 );
    if(ith == 0)
      geometry1() = geom->getEntity();
    else
      geometry2() = geom->getEntity();
  }

  AGX_FORCE_INLINE agx::Vec3f GeometryContact::calculateRelativeVelocity(size_t pointIndex) const
  {
    agx::Vec3 v1, v2;
    if (pointIndex > points().size() - 1)
      return agx::Vec3f();

    const agx::RigidBody* rb1 = geometry(0)->getRigidBody();
    const agx::RigidBody* rb2 = geometry(1)->getRigidBody();

    if (rb1)
      v1 = rb1->getVelocity() + (rb1->getAngularVelocity() ^ (points()[pointIndex].point() - rb1->getCmPosition()));
    if (rb2)
      v2 = rb2->getVelocity() + (rb2->getAngularVelocity() ^ (points()[pointIndex].point() - rb2->getCmPosition()));

    return static_cast<agx::Vec3f>(v1 - v2);
  }


  inline bool GeometryContact::calculateSurfaceVelocity(const agxCollide::Geometry* geometry1, const agxCollide::Geometry* geometry2, ContactPoint& p)
  {
    LocalContactPoint lcp;
    lcp.point() = p.point();
    bool hasSurfaceVelocity = calculateSurfaceVelocity(geometry1, geometry2, lcp);
    p.velocity() = lcp.velocity();
    return hasSurfaceVelocity;
  }


  AGX_FORCE_INLINE agx::RigidBody* GeometryContact::rigidBody( size_t ith )
  {
    agxAssert(ith < 2 );
    return geometry(ith)->getRigidBody();
  }

  AGX_FORCE_INLINE const agx::RigidBody* GeometryContact::rigidBody( size_t ith ) const
  {
    agxAssert(ith < 2 );
    return geometry(ith)->getRigidBody();
  }


  AGX_FORCE_INLINE ContactPointVector& GeometryContact::points()
  {
    return reinterpret_cast<ContactPointVector&>(agx::Physics::GeometryContactPtr::points());
  }

  AGX_FORCE_INLINE const ContactPointVector& GeometryContact::points() const
  {
    return reinterpret_cast<const ContactPointVector&>(agx::Physics::GeometryContactPtr::points());
  }

  AGX_FORCE_INLINE void* GeometryContact::getCustomData()
  {
    return customData();
  }

  AGX_FORCE_INLINE const void* GeometryContact::getCustomData() const
  {
    return customData();
  }

  inline void GeometryContact::setCustomData( void* customData )
  {
    this->customData() = customData;
  }


  AGX_FORCE_INLINE void GeometryContact::markForRemoval( bool immediately )
  {
    this->immediately() = immediately;
    this->enabled() = false;
  }

  AGX_FORCE_INLINE void GeometryContact::setEnable(bool flag)
  {
    this->enabled() = flag;
  }

  AGX_FORCE_INLINE bool GeometryContact::isEnabled() const
  {
    return this->enabled();
  }


  AGX_FORCE_INLINE bool GeometryContact::shouldBeCalled() const
  {
    return !immediately();
  }

  AGX_FORCE_INLINE agx::ContactMaterial* GeometryContact::getMaterial()
  {
    if ( material() && material().isExplicit() )
      return material().model();
    return nullptr;
  }

  AGX_FORCE_INLINE const agx::ContactMaterial* GeometryContact::getMaterial() const
  {
    if ( material() && material().isExplicit() )
      return material().model();
    return nullptr;
  }

  AGX_FORCE_INLINE const agx::ContactMaterial* GeometryContact::getMaterial( bool ) const
  {
    return material() ? material().model() : nullptr;
  }

  AGX_FORCE_INLINE void GeometryContact::setMaterial( agx::ContactMaterial* material )
  {
    agxAssert( material != nullptr );
    if (material)
      this->material() = material->getEntity();
  }

  AGX_FORCE_INLINE void GeometryContact::setHasSurfaceVelocity( bool flag )
  {
    hasSurfaceVelocity() = flag;
  }


  AGX_FORCE_INLINE void GeometryContact::setHasInternalMaterial( bool flag )
  {
    hasInternalMaterial() = flag;
  }

  AGX_FORCE_INLINE bool GeometryContact::getHasInternalMaterial( ) const
  {
    return hasInternalMaterial();
  }

  AGX_FORCE_INLINE agx::UInt8 GeometryContact::state() const
  {
    if ( broadPhasePair() )
      return broadPhasePair().state();

    return GeometryContact::NO_CONTACT;
  }

  AGX_FORCE_INLINE void GeometryContact::setState( agx::UInt8 state )
  {
    agxAssert(broadPhasePair());
    broadPhasePair().state() = state;
  }

  AGX_FORCE_INLINE agx::UInt32& ContactPoint::faceIndex(size_t ith)
  {
    if (ith == 0) return faceIndex1(); else return faceIndex2();
  }


  AGX_FORCE_INLINE agx::UInt32 const& ContactPoint::faceIndex(size_t ith) const
  {
    if (ith == 0) return faceIndex1(); else return faceIndex2();
  }


  AGX_FORCE_INLINE agx::UInt8& ContactPoint::faceFeature(size_t ith)
  {
    if (ith == 0) return faceFeature1(); else return faceFeature2();
  }


  AGX_FORCE_INLINE agx::UInt8 const& ContactPoint::faceFeature(size_t ith) const
  {
    if (ith == 0) return faceFeature1(); else return faceFeature2();
  }


  AGX_FORCE_INLINE agx::Physics::Geometry::ShapePtr& ContactPoint::shape(size_t ith)
  {
    if (ith == 0) return shape1(); else return shape2();
  }


  AGX_FORCE_INLINE agx::Physics::Geometry::ShapePtr const& ContactPoint::shape(size_t ith) const
  {
    if (ith == 0) return shape1(); else return shape2();
  }


  AGX_FORCE_INLINE agx::Vec3 ContactPoint::getWitnessPoint(size_t ith) const
  {
    if (ith == 0)
      return point() - agx::Vec3(normal()) * (depth() * agx::Real(0.5));
    else
      return point() + agx::Vec3(normal()) * (depth() * agx::Real(0.5));
  }
}

#ifdef _MSC_VER
# pragma warning(pop)
#endif

