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

#ifndef AGX_PHYSICS_GRANULARBODY_CACHEDCONTACT_H
#define AGX_PHYSICS_GRANULARBODY_CACHEDCONTACT_H

#include <agxData/Type.h>
#include <agx/Real.h>
#include <agx/Integer.h>
// #include <agx/Physics/ParticleEntity.h>
#include <agx/Physics/GranularBodyEntity.h>
#include <agx/Physics/GeometryEntity.h>
#include <agx/SymmetricPair.h>

namespace agx
{
  namespace Physics
  {
    namespace GranularBody
    {
      struct ContactLambda
      {
        ContactLambda()
        {
          for (size_t i = 0; i < 6; ++i)
            lambda[i] = 0;
        }

        agx::Real32 lambda[6];
      };
      
      typedef SymmetricPair<GranularBodyPtr> GranularGranularPair;
      typedef std::pair<GranularBodyPtr, GeometryPtr> GranularGeometryPair;
    }
  }
}

AGX_TYPE_BINDING(agx::Physics::GranularBody::ContactLambda, "Physics.GranularBody.ContactLambda")
AGX_TYPE_BINDING(agx::Physics::GranularBody::GranularGranularPair, "Physics.GranularBody.GranularGranularPair")
AGX_TYPE_BINDING(agx::Physics::GranularBody::GranularGeometryPair, "Physics.GranularBody.GranularGeometryPair")


#endif /* AGX_PHYSICS_GRANULARBODY_CACHEDCONTACT_H */
