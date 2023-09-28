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

#ifndef AGX_CONTACT_FORCE_READER_H
#define AGX_CONTACT_FORCE_READER_H

#include <agx/agx.h>
#include <agx/Referenced.h>
#include <agx/agxPhysics_export.h>
#include <agxSDK/Simulation.h>

namespace agx
{
  /**
  Class for reading contact forces between rigid bodies and geometries.
  If this class is used several times per time step, it is usually faster to do this "by hand"
  (with a smart for-loop over the contacts from Space::getGeometryContacts()), since this
  class will loop over these contacts separately for each call.
  */
  class CALLABLE AGXPHYSICS_EXPORT ContactForceReader : public agx::Referenced
  {
    public:
      enum ContactType
      {
        IMPACT = (1 << 0),
        RESTING = (1 << 1),
        IMPACT_AND_RESTING = (IMPACT & RESTING)
      };

      ContactForceReader(const agxSDK::Simulation* sim);

      /**
      Get sum of all contact (Normal and Friction) forces in world coordinate system between two rigid bodies.
      \param body1 - First rigid body
      \param body2 - Second rigid body (null matches any rigid body)
      \param type - Specifies which type of contact force to be returned
      \returns Sum of contact forces in world coordinate system
      */
      Vec3 getContactForces(
        const agx::RigidBody* body1, const agx::RigidBody* body2 = nullptr, ContactType type = IMPACT_AND_RESTING);

      /**
      Get sum of all contact (Normal and Friction) forces in world coordinate system between a rigid body and a geometry.
      \param body - Matching rigid body
      \param geometry - Matching geometry
      \param type - Specifies which type of contact force to be returned
      \returns Sum of contact forces in world coordinate system
      */
      Vec3 getContactForces(
        const agx::RigidBody* body, const agxCollide::Geometry* geometry, ContactType type = IMPACT_AND_RESTING);

      /**
      Get sum of all contact (Normal and Friction) forces in world coordinate system between two geometries.
      \param geometry1 - First matching geometry
      \param geometry2 - Second matching geometry (null matches any geometry)
      \param type - Specifies which type of contact force to be returned
      \returns Sum of contact forces in world coordinate system
      */
      Vec3 getContactForces(
        const agxCollide::Geometry* geometry1, const agxCollide::Geometry* geometry2 = nullptr,
        ContactType type = IMPACT_AND_RESTING);

      /**
      Get sum of all Normal contact forces in world coordinate system between two rigid bodies.
      \param body1 - First matching rigid body
      \param body2 - Second matching rigid body (null matches any rigid body)
      \param type - Specifies which type of contact force to be returned
      \returns Sum of normal contact forces in world coordinate system
      */
      Vec3 getNormalContactForces(
        const agx::RigidBody* body1, const agx::RigidBody* body2 = nullptr, ContactType type = IMPACT_AND_RESTING);

      /**
      Get sum of all Normal contact forces in world coordinate system between a rigid body and a geometry
      \param body - matching rigid body
      \param geometry - matching geometry
      \param type - Specifies which type of contact force to be returned
      \returns Sum of normal contact forces in world coordinate system
      */
      Vec3 getNormalContactForces(
        const agx::RigidBody* body, const agxCollide::Geometry* geometry, ContactType type = IMPACT_AND_RESTING);

      /**
      Get sum of all Normal contact forces in world coordinate system between two geometries.
      \param geometry1 - first matching geometry
      \param geometry2 - second matching geometry (null matches any geometry)
      \param type - Specifies which type of contact force to be returned
      \returns Sum of normal contact forces in world coordinate system
      */
      Vec3 getNormalContactForces(
        const agxCollide::Geometry* geometry1, const agxCollide::Geometry* geometry2 = nullptr,
        ContactType type = IMPACT_AND_RESTING);

      /**
      Get sum of all Friction contact forces in world coordinate system between two rigid bodies.
      \param body1 - first matching rigid body
      \param body2 - second matching rigid body (null matches any rigid body)
      \param type - Specifies which type of contact force to be returned
      \returns Sum of friction contact forces in world coordinate system
      */
      Vec3 getTangentialContactForces(
        const agx::RigidBody* body1, const agx::RigidBody* body2 = nullptr, ContactType type = IMPACT_AND_RESTING);

      /**
      Get sum of all Friction contact forces in world coordinate system between a rigid body and a geometry
      \param body - matching rigid body
      \param geometry - matching geometry
      \param type - Specifies which type of contact force to be returned
      \returns Sum of friction contact forces in world coordinate system
      */
      Vec3 getTangentialContactForces(
        const agx::RigidBody* body, const agxCollide::Geometry* geometry, ContactType type = IMPACT_AND_RESTING);

      /**
      Get sum of all Friction contact forces in world coordinate system between two geometries.
      \param geometry1 - First matching geometry
      \param geometry2 - Second matching geometry (null matches any geometry)
      \param type - Specifies which type of contact force to be returned
      \returns Sum of friction contact forces in world coordinate system
      */
      Vec3 getTangentialContactForces(
        const agxCollide::Geometry* geometry1, const agxCollide::Geometry* geometry2 = nullptr,
        ContactType type = IMPACT_AND_RESTING);

      /**
      Get sum of all absolute contact forces between two rigid bodies.
      \param body1 - First matching rigid body
      \param body2 - Second matching rigid body (null matches any rigid body)
      \param type - Specifies which type of contact force to be returned
      \returns Sum of absolute contact forces
      */
      Vec3 getSumAbsoluteContactForces(
        const agx::RigidBody* body1, const agx::RigidBody* body2 = nullptr, ContactType type = IMPACT_AND_RESTING);

      /**
      Get sum of all absolute contact forces between a rigid body and a geometry
      \param body - matching rigid body
      \param geometry - matching geometry
      \param type - Specifies which type of contact force to be returned
      \returns Sum of absolute contact forces
      */
      Vec3 getSumAbsoluteContactForces(
        const agx::RigidBody* body, const agxCollide::Geometry* geometry, ContactType type = IMPACT_AND_RESTING);

      /**
      Get sum of all absolute contact forces between two geometries
      \param geometry1 - First matching geometry
      \param geometry2 - Second matching geometry (null matches any geometry)
      \param type - Specifies which type of contact force to be returned
      \returns Sum of absolute contact forces
      */
      Vec3 getSumAbsoluteContactForces(
        const agxCollide::Geometry* geometry1, const agxCollide::Geometry* geometry2 = nullptr,
        ContactType type = IMPACT_AND_RESTING);

      /**
      Get sum of all absolute normal contact forces between two rigid bodies
      \param body1 - First matching rigid body
      \param body2 - Second matching rigid body (null matches any rigid body)
      \param type - Specifies which type of contact force to be returned
      \returns Sum of absolute normal contact forces
      */
      Vec3 getSumAbsoluteNormalContactForces(
        const agx::RigidBody* body1, const agx::RigidBody* body2 = nullptr, ContactType type = IMPACT_AND_RESTING);

      /**
      Get sum of all absolute normal contact forces between two rigid bodies
      \param body - matching rigid body
      \param geometry - matching geometry
      \param type - Specifies which type of contact force to be returned
      \returns Sum of absolute normal contact forces
      */
      Vec3 getSumAbsoluteNormalContactForces(
        const agx::RigidBody* body, const agxCollide::Geometry* geometry, ContactType type = IMPACT_AND_RESTING);

      /**
      Get sum of all absolute normal contact forces between two geometries
      \param geometry1 - First matching geometry
      \param geometry2 - Second matching geometry (null matches any geometry)
      \param type - Specifies which type of contact force to be returned
      \returns Sum of absolute normal contact forces
      */
      Vec3 getSumAbsoluteNormalContactForces(
        const agxCollide::Geometry* geometry1, const agxCollide::Geometry* geometry2 = nullptr,
        ContactType type = IMPACT_AND_RESTING);

      /**
      Get sum of all absolute friction contact forces between two rigid bodies
      \param body1 - First matching rigid body
      \param body2 - Second matching rigid body (null matches any rigid body)
      \param type - Specifies which type of contact force to be returned
      \returns Sum of absolute friction contact forces
      */
      Vec3 getSumAbsoluteTangentialContactForces(
        const agx::RigidBody* body1, const agx::RigidBody* body2 = nullptr, ContactType type = IMPACT_AND_RESTING);

      /**
      Get sum of all absolute friction contact forces between a rigid body and a geometry
      \param body - matching rigid body
      \param geometry - matching geometry
      \param type - Specifies which type of contact force to be returned
      \returns Sum of absolute friction contact forces
      */
      Vec3 getSumAbsoluteTangentialContactForces(
        const agx::RigidBody* body, const agxCollide::Geometry* geometry, ContactType type = IMPACT_AND_RESTING);

      /**
      Get sum of all absolute friction contact forces between two geometries
      \param geometry1 - First matching geometries
      \param geometry2 - Second matching geometry (null matches any geometry)
      \param type - Specifies which type of contact force to be returned
      \returns Sum of absolute friction contact forces
      */
      Vec3 getSumAbsoluteTangentialContactForces(
        const agxCollide::Geometry* geometry1, const agxCollide::Geometry* geometry2 = nullptr,
        ContactType type = IMPACT_AND_RESTING);

      /// Update reference to Simulation
      void setSimulation(agxSDK::Simulation* simulation);

    private:
      agx::observer_ptr<const agxSDK::Simulation> m_sim;
  };

  typedef ref_ptr< ContactForceReader > ContactForceReaderRef;

} // namespace agx

#endif
