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

#ifndef AGX_HASH_TYPES_H
#define AGX_HASH_TYPES_H

#include <agx/agx.h>


#include <agx/HashTable.h>
#include <agx/HashSet.h>

#include <agx/agx_vector_types.h>

namespace agx
{

  // Types using HashTable
  class Interaction;
  class RigidBody;
  class Constraint;
  class ParticleSystem;
  class ParticleEmitter;
  class ParticleSink;
  class ConstraintImplementation;

  class Attachment;

  //typedef HashTable< Interaction *, ref_ptr< Interaction> > InteractionTable;
  typedef HashTable< RigidBody*, ref_ptr<RigidBody> > RigidBodyTable;
  //typedef HashTable< Constraint*, ref_ptr<Constraint> > ConstraintTable;
  typedef HashTable< ParticleSystem*, ref_ptr<ParticleSystem> > ParticleSystemTable;
  typedef HashTable< ParticleEmitter*, ref_ptr<ParticleEmitter> > ParticleEmitterTable;
  typedef HashTable< ParticleSink*, ref_ptr<ParticleSink> > ParticleSinkTable;
  typedef HashTable< ConstraintImplementation*, int > ConstraintImplementationPtrIntHashTable;
  typedef HashTable< RigidBody*, int > RigidBodyPtrIntHash;
  typedef HashTable<RigidBody *, Real> RigidBodyPtrRealHash;
  typedef HashTable<RigidBody *, Vec3> RigidBodyPtrVec3Hash;


  typedef agx::HashTable<int, ConstraintImplementation*> ConstraintImplementationHashTable;
  typedef agx::HashTable<RigidBody*, Attachment*> RigidBodyPtrAttachmentPtrHashTable;
  class FluidHasher;
  typedef HashTable<RigidBody *, FluidContactVector *, FluidHasher> FluidParticleHash;

  typedef agx::HashSet<agx::Index> IndexHashSet;
}

#endif
