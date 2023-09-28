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

#include <agxSDK/MergeSplitAlgorithm.h>
#include <agx/Physics/GranularBodySystem.h>

namespace agxSDK
{
  AGX_DECLARE_POINTER_TYPES( GranularContactMergeSplitAlgorithm );

  class AGXPHYSICS_EXPORT GranularContactMergeSplitAlgorithm : public MergeSplitAlgorithm
  {
    public:
      enum EContactState
      {
        VALID     = 1 << 0,
        IMPACTING = 1 << 1,
        SLIDING   = 1 << 2,
        ROLLING   = 1 << 3,
        RESTING   = 1 << 4
      };
      typedef agx::BitState<EContactState, agx::Int32> ContactState;

    public:
      /**
      Default constructor.
      */
      GranularContactMergeSplitAlgorithm();

      AGXSTREAM_DECLARE_SERIALIZABLE( agxSDK::GranularContactMergeSplitAlgorithm );

      static agx::Physics::GranularBodyPtr ConvertRigidBodyGranulateToGranularBody(agxSDK::Simulation * simulation, agx::RigidBody * rb);

    protected:

      GranularContactMergeSplitAlgorithm::ContactState findContactState(const agx::Vec3& point,
        const agx::Vec3f& normal, const agx::Vec3& cmPos1, const agx::Vec3& cmPos2, const agx::Vec3& v1,
        const agx::Vec3& v2, const agx::Vec3& a1, const agx::Vec3& a2, const agx::Vec3f& u, const agx::Vec3f& v);

      GranularContactMergeSplitAlgorithm::ContactState findContactState(const agx::Physics::ParticleGeometryContactPtr& contact);

      GranularContactMergeSplitAlgorithm::ContactState findContactState(const agx::Physics::ParticlePairContactPtr& contact);

      void postSolveParticlePairContact(const agx::Physics::ParticlePairContactPtr& contact, const agxSDK::MergedState& mergedState, agxSDK::MergeSplitActionContainer& actions);

      void postSolveParticleGeometryContact(const agx::Physics::ParticleGeometryContactPtr& contact, const agxSDK::MergedState& mergedState, agxSDK::MergeSplitActionContainer& actions);

      agx::RigidBody * createRigidBodyFromGranulate(agx::Physics::GranularBodyPtr gptr);

      agxSDK::Simulation * getSimulation();
      const agxSDK::Simulation * getSimulation() const;

      agx::Physics::GranularBodySystem * getGranularBodySystem();

      void destroyMergedParticles();

      /**
      Reference counted object, protected destructor.
      */
      virtual ~GranularContactMergeSplitAlgorithm();

    private:
      friend class agxSDK::MergeSplitHandler;

      //////////////////////////////////////////////////////////////////////////
      // Variables
      //////////////////////////////////////////////////////////////////////////
    protected:
      agx::HashSet<agx::Index> m_particlesToDestroy;    // Contains particles that are marked for merge and should be removed replaced with rigid bodies
  };
}
