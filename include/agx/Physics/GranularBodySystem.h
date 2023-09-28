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

#ifndef AGX_PHYSICS_GRANULARBODYSYSTEM_H
#define AGX_PHYSICS_GRANULARBODYSYSTEM_H

#include <agx/ParticleSystem.h>
#include <agx/Physics/GranularBodyEntity.h>
#include <agx/Physics/GranularBody/CachedContact.h>

namespace agx
{
  namespace Physics
  {
    /**
    Derived class from ParticleSystem that implements 6-DOF particles with Hertzian contacts with Coulomb friction and rolling resistance.
    */
    AGX_DECLARE_POINTER_TYPES(GranularBodySystem);
    class AGXPHYSICS_EXPORT GranularBodySystem : public agx::ParticleSystem
    {
    public:
      /**
      The MotionControl enumeration indicates what makes a GranularBody in the GranularBodySystem move.

      There are two forms of motion allowed:
      -  KINEMATICS means that motion is scripted;
      -  DYNAMICS means that motion results from forces;
      */
      enum MotionControl
      {
        KINEMATICS = 2, /**< This Granular Body's motion is scripted. (Position/velocity set by the user)*/
        DYNAMICS = 3  /**< This Granular Body moves from the influence of forces. (Position/velocity updated by the system) */
      };

    public:
      GranularBodySystem(Device *device = CpuDevice::instance());

      /**
      Create a new particle in the particle system.
      */
      Physics::GranularBodyPtr createParticle();

      /**
      * Set the task that will be executed whenever it is time to step the simulation.
      */
      virtual bool setUpdateTask(Task* updateTask) override;

#if (!defined SWIG ) || ( defined SWIGCSHARP || defined SWIGJAVA)
      /**
      \return All active particles in the particle system (only valid until storage is modified, e.g. by deleting a particle).
      */
      const agx::Physics::GranularBodyPtrArray getParticles() const;
      agx::Physics::GranularBodyPtrArray getParticles();

      agx::Physics::GranularBodyPtrVector getGranularBodyPtrVector() const;
      agx::Physics::GranularBodyPtrVector getGranularBodyPtrVector();
#endif

      /// Set motion control of the granular body
      void setMotionControl( agx::Physics::GranularBodyPtr gptr, MotionControl motionControl );

      /// Get the motion control of a granular body
      MotionControl getMotionControl( agx::Physics::GranularBodyPtr gptr ) const;

      /// Copy data from GranularBody source to target body
      static void copyGranularBodyData( agx::Physics::GranularBodyPtr target, agx::Physics::GranularBodyPtr source, bool copyMaterial=true );

      /// Set single particle radius given a specified id
      virtual void setSingleParticleRadius(agx::Index particleId, agx::Real radius) override;

      /// Set angular velocity for a single particle
      void setSingleParticleAngularVelocity(agx::Index particleId, const agx::Vec3& angularVelocity );

      /// Get angular velocity for a single particle
      agx::Vec3 getSingleParticleAngularVelocity(agx::Index particleId);

      /// Convert a granular body to a rigid body
      static agx::RigidBody* convertGranularBodyToRigidBody(agx::Physics::GranularBodyPtr gptr);

      /// Apply a 4x4 transform on a particle
      static void applyTransform( agx::Physics::GranularBodyPtr gptr, const agx::AffineMatrix4x4& transform, bool includeVelocity = false );

      /// Checks if a particle system is a granular body system by either class or by internal models
      static bool isGranularBodySystem(agx::ParticleSystem* system);

      /**
      Set granular body mass and inertia. Note that the default mass is normally derived from the default material of the
      granular system when granular bodies are created. This function is used to set the default value of the mass and
      inertia buffers and also to manually set the mass and inertia on all granular bodies.
      \note - Setting a material on either the granular system or an individual granular bodies will override this mass value.
      \param mass - the specified mass to set as default and to all existing granular bodies if that option is used.
      \param updateExistingParticles - true if the mass and inertia of all existing bodies should be updated, false otherwise. ( Default: false )
      */
      virtual void setParticleMass( Real mass, bool updateExistingParticles = false ) override;

      /**
      Set the contact mode of the GranularBodySystem. (Default is 4)
      2 - Only friction and normal  forces are calculated in the granular contacts.
      3 - Normal, friction and rolling resistance forces are calculated in the granular contacts.
      4 - Normal, friction, rolling resistance and twisting resistance forces are calculated in the granular contacts.
      */
      void setContactMode(agx::UInt contactMode);

      /**
      Pre-allocate memory for elements via the `reserve` function in the storages related to
      particle-particle contact and constraint data with specified number of elements.
      \param numElements - the number of elements to pre-allocate in the storage.
      */
      void preAllocateParticlePairContactStorage( size_t numElements );

      /**
      Pre-allocate memory for elements via the `reserve` function in the storages related to
      particle-geometry contact and constraint data with specified number of elements.
      \param numElements - the number of elements to pre-allocate in the storage.
      */
      void preAllocateParticleGeometryContactStorage( size_t numElements );

      /**
      Pre-allocate memory for elements via the `reserve` function in the particle data storage
      with specified number of elements.
      \param numElements - the number of elements to pre-allocate in the particle storage.
      */
      void preAllocateParticleStorage( size_t numElements );

      AGXSTREAM_DECLARE_SERIALIZABLE( agx::Physics::GranularBodySystem );

    protected:
      void invalidLicenseCallback();
      void init();

      virtual ~GranularBodySystem();

    private:
    };


    AGX_FORCE_INLINE Physics::GranularBodyPtr GranularBodySystem::createParticle() { return ParticleSystem::createParticle(); }
    AGX_FORCE_INLINE agx::Physics::GranularBodyPtrArray GranularBodySystem::getParticles() { return getParticleStorage()->getInstances<Physics::GranularBodyPtr>(); }
    AGX_FORCE_INLINE const agx::Physics::GranularBodyPtrArray GranularBodySystem::getParticles() const { return const_cast<GranularBodySystem *>(this)->getParticles(); }
  }
}


#endif /* AGX_PHYSICS_GRANULARBODYSYSTEM_H */
