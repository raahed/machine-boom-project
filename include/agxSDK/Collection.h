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

#include <agxSDK/Assembly.h>

namespace agxSDK
{
  /**
  A Collection is a collection of basic simulation objects, such as rigid bodies,
  constraints, geometries. The assembly may also contain other assemblies,
  which enables a hierarchical structuring of the simulation, allowing
  manipulation at different conceptual levels.

  The difference from Assembly, is that Collection does NOT build a Transformation hierarchy.
  It is merely for collecting objects without any Frame structure.

  The assembly tree can be traversed using an AssemblyVisitor.
  */
  class AGXPHYSICS_EXPORT Collection : public Assembly
  {
    public:

      /// Default constructor
      Collection();

      using agxSDK::Assembly::add;
      using agxSDK::Assembly::remove;

      /**
      Add a particle system to this assembly.
      ParticleSystems should only be added to Collections.
      */
      virtual void add( agx::ParticleSystem* particleSystem );

      /**
      Remove a particle system from this assembly
      */
      virtual void remove(agx::ParticleSystem* particleSystem);

    protected:
      AGXSTREAM_DECLARE_SERIALIZABLE(agxSDK::Collection);

      virtual ~Collection();
  };

  typedef agx::ref_ptr<Collection> CollectionRef;

}
