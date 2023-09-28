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

#include <agxTerrain/PrimaryActiveZone.h>
#include <agxTerrain/SoilSimulationInterface.h>
#include <agx/RigidBody.h>
#include <agx/LockJoint.h>
#include <agx/Plane.h>

namespace agxTerrain
{
  class Shovel;
  class TerrainToolCollection;
  class ActiveZone;
  class TerrainMaterial;

  typedef agx::Physics::GranularBodyPtr SoilParticlePtr;
  typedef agx::Physics::GranularBodyPtrVector SoilParticlePtrVector;

  AGX_DECLARE_POINTER_TYPES( SoilParticleAggregate );
  AGX_DECLARE_VECTOR_TYPES(SoilParticleAggregate);

  class AGXTERRAIN_EXPORT SoilParticleAggregate : public agx::Referenced
  {
    public:
      /**
      Construct given shovel.
      */
      SoilParticleAggregate();

      /**
      \note It's undefined to make changes to this body since all its
            properties are calculated given the soil particles it
            represent.
      \return inner rigid body used as soil particle aggregate
      */
      const agx::RigidBody* getInnerBody() const;

      /**
      \note It's undefined to make changes to this body since all its
            properties are calculated given the soil particles it
            represent.
      \return wedge rigid body used as soil particle aggregate
      */
      const agx::RigidBody* getWedgeBody() const;

      /**
      \note It's undefined to make changes to this geometry since all its
            properties are calculated given the soil particles it
            represent.
      \return geometry used as soil particle aggregate
      */
      const agxCollide::Geometry* getInnerGeometry() const;

      /**
      \note It's undefined to make changes to this geometry since all its
            properties are calculated given the soil particles it
            represent.
      \return geometry used as soil particle aggregate
      */
      const agxCollide::Geometry* getWedgeGeometry() const;

      /**
      \note It's undefined to make changes to this lock since all its
            properties are calculated during the aggregate update step.
            ALSO NOTE, this lock is only relevant if the body aggregate
            is part of a primary active zone.
      \return lock between the inner body and the wedge shape, IF used by primary active zone.
      */
      const agx::LockJoint* getLockJoint() const;

      /**
      Assign material used for the contact between the shovel and this
      soil particle aggregate (the soil particles implicitly).
      \param material - new material
      */
      void setMaterial( agx::Material* material );

      /**
      \return the material used between the shovel and this soil particle aggregate
      */
      agx::Material* getMaterial() const;

      /**
      \return the mass of this aggregate, representing both soil particles and fluid mass
      */
      agx::Real getMass() const;

      /**
      \return the soil particles that the inner body of the soil particle aggregate represents
      */
      const SoilParticlePtrVector& getInnerBodyParticles() const;

      /**
      \return the soil particles that the wedge body of the soil particle aggregate represents
      */
      const SoilParticlePtrVector& getWedgeBodyParticles() const;

      /**
      Set the compliance of the lock joint in the soil aggregate in translational degrees of freedom.
      \param compliance - Set the joint compliance in the translation degrees of freedom.
      */
      void setLockTranslationalCompliance(agx::Real compliance);

      /**
      Set the compliance of the lock joint in the soil aggregate in rotational degrees of freedom.
      \param compliance - Set the joint compliance in the rotational degrees of freedom.
      */
      void setLockRotationalCompliance(agx::Real compliance);

      /**
      \return the mass of all the rigid bodies present in the aggregate.
      */
      agx::Real getRigidBodyMassSum() const;

    public:
      /**
      \internal

      On addNotification of Terrain and Shovel.
      */
      void addNotification( agxSDK::Simulation* simulation, Terrain* terrain );

      /**
      \internal

      On removeNotification of Terrain and Shovel.
      */
      void removeNotification( agxSDK::Simulation* simulation, Terrain* terrain );

      /**
      \internal

      On pre-collide step.
      */
      void onPreCollide( TerrainToolCollection* collection );

      /**
      \internal

      On pre step of Terrain and Shovel.
      */
      void onPre( TerrainToolCollection* collection, ActiveZone* activeZone );

      /**
      \internal

      On post step of Terrain and Shovel.
      */
      void onPost( TerrainToolCollection* collection );

      /**
      \internal

      Called when Shovel enable is changed.
      */
      void onEnableChange( bool enable );

      static agx::Plane transformPlaneToWorld(const agx::Frame* localFrame, const agx::Plane& localPlane);

    protected:
      /**
      Reference counted object - protected destructor.
      */
      virtual ~SoilParticleAggregate();

      /**
      Internal mutable geometry accessor.
      */
      agxCollide::Geometry* _getInnerGeometry() const;

      /**
      Internal mutable geometry accessor.
      */
      agxCollide::Geometry* _getWedgeGeometry() const;

    private:
      agx::HashSet<agx::Vec3i> calculateWedgeVoxels(TerrainToolCollection* collection, ActiveZone* activeZone);

      agx::HashSet<agx::Vec3i> filterWedgeVoxelsSplittingPlane(TerrainToolCollection* collection,
                                                               ActiveZone* activeZone,
                                                               const agx::HashSet<agx::Vec3i>& wedgeVoxels);

      void updateDynamicProperties(const SoilParticlePtrVector& soilParticles,
                                   agx::RigidBody* rb,
                                   agx::HashSet<agx::Vec3i>& voxels,
                                   TerrainToolCollection* collection);

      void synchronizeLockJoint(TerrainToolCollection* collection);

      void computePointsInsideActiveZone(TerrainToolCollection* collection, ActiveZone* activeZone);

      void sortParticlesBySeparatingPlane(Terrain* terrain,
                                          ActiveZone* activeZone,
                                          SoilParticlePtrVector& innerParticles,
                                          SoilParticlePtrVector& wedgeParticles);

      void computeParticlesInInnerShape(PrimaryActiveZone* activeZone, SoilParticlePtrVector& innerParticles);

      SoilParticlePtrVector computeParticlesInWedgeShape(ActiveZone* activeZone, SoilParticlePtrVector& wedgeParticleCandidates);

      void computeParticlesAboveShovel(TerrainToolCollection* collection, agx::Vec3 up, SoilParticlePtrVector& wedgeParticles);

      bool isPointAbovePlane(const agx::Plane& plane, const agx::Vec3& point) const;

      bool isSphereIntersectingPlane(const agx::Plane& plane, const agx::Vec3& point, agx::Real radius) const;

      void updateLockJointComplianceFromBulkMaterial(const TerrainToolCollection* collection,
                                                     agx::LockJoint* lockJoint);

    private:
      agx::RigidBodyRef m_innerBody;
      agx::RigidBodyRef m_wedgeBody;
      agx::Vec3         m_innerBodyMomentum;
      agx::Vec3         m_wedgeBodyMomentum;
      agx::Vec3         m_innerBodyAngularMomentum;
      agx::Vec3         m_wedgeBodyAngularMomentum;

      SoilParticlePtrVector m_innerParticles;
      SoilParticlePtrVector m_wedgeParticles;

      agx::LockJointRef m_lock;
  };



  inline bool SoilParticleAggregate::isPointAbovePlane(const agx::Plane& plane, const agx::Vec3& point) const
  {
    return plane.signedDistanceToPoint(point) > 0;
  }



  inline bool SoilParticleAggregate::isSphereIntersectingPlane(const agx::Plane& plane, const agx::Vec3& point, agx::Real radius) const
  {
    return std::abs(plane.signedDistanceToPoint(point)) < radius;
  }




}
