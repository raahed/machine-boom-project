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

#ifndef AGX_GRANLAR_CONTACT_FORCE_READER_H
#define AGX_GRANLAR_CONTACT_FORCE_READER_H

#include <agx/agx.h>
#include <agx/Referenced.h>
#include <agx/agxPhysics_export.h>
#include <agxSDK/Simulation.h>

namespace agx
{
  /**
  Class for reading contact forces between granular bodies and rigid bodies and geometries.
  If this class is used several times per time step, it is usually faster to do this "by hand"
  (with a smart for-loop over the contacts from Space::getGeometryContacts()), since this
  class will loop over these contacts separately for each call.
  */
  class CALLABLE AGXPHYSICS_EXPORT GranularContactForceReader : public agx::Referenced
  {
    public:
      GranularContactForceReader(const agxSDK::Simulation* sim);

      /// Get all contact forces on a single rigid body
      agx::Vec3 getContactForces(const agx::RigidBody* rb);
      /// Get all contact forces on a single geometry
      agx::Vec3 getContactForces(const agxCollide::Geometry* geo);

      /// Get all normal forces on a single rigid body
      agx::Vec3 getNormalContactForces(const agx::RigidBody* rb);
      /// Get all normal forces on a single geometry
      agx::Vec3 getNormalContactForces(const agxCollide::Geometry* geo);

      /// Get all tangential forces on a single rigid body.
      agx::Vec3 getFrictionContactForces(const agx::RigidBody* rb);
      /// Get all tangential forces on a single geometry.
      agx::Vec3 getFrictionContactForces(const agxCollide::Geometry* geo);

      /// Get the sum of all contact forces on a single rigid body, summing the absolute forces per contact point.
      agx::Vec3 getSumAbsoluteContactForces(const agx::RigidBody* rb);
      /// Get the sum of all contact forces on a single geometry, summing the absolute forces per contact point.
      agx::Vec3 getSumAbsoluteContactForces(const agxCollide::Geometry* geo);

      /// Get the sum of all normal forces on a single rigid body, summing the absolute forces per contact point.
      agx::Vec3 getSumAbsoluteNormalContactForces(const agx::RigidBody* rb);
      /// Get the sum of all normal forces on a single geometry, summing the absolute forces per contact point.
      agx::Vec3 getSumAbsoluteNormalContactForces(const agxCollide::Geometry* geo);

      /// Get the sum of all tangential forces on a single rigid body, summing the absolute forces per contact point.
      agx::Vec3 getSumAbsoluteFrictionContactForces(const agx::RigidBody* rb);
      /// Get the sum of all tangential forces on a single geometry, summing the absolute forces per contact point.
      agx::Vec3 getSumAbsoluteFrictionContactForces(const agxCollide::Geometry* geo);

      /// Update reference to Simulation
      void setSimulation(agxSDK::Simulation* sim);

    private:
      agx::observer_ptr<const agxSDK::Simulation> m_sim;
  };

  typedef ref_ptr< GranularContactForceReader > GranularContactForceReaderRef;

} // namespace agx

#endif
