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

#ifndef AGXSDK_SIMULATIONPROXY_H
#define AGXSDK_SIMULATIONPROXY_H

#include <agx/Referenced.h>

#include <agxCollide/LocalContactPoint.h>

namespace agxCollide
{
  class Geometry;
}

namespace agx
{
  class RigidBody;
  class Constraint;
  class StrongInteraction;
  class GravityField;
}

namespace agxSDK
{
  // Forward declarations.
  class Simulation;
  class MaterialManager;
  class Assembly;
  class EventListener;

  /**
  Interface class used by internal objects running in parallel.
  For documentation, check agxWire::SimulationProxy.
  */
  class AGXPHYSICS_EXPORT SimulationProxy : public agx::Referenced
  {
    public:
      virtual bool add( agxCollide::Geometry* geometry ) = 0;
      virtual bool remove( agxCollide::Geometry* geometry ) = 0;
      virtual void setEnable( agxCollide::Geometry* geometry, bool enable ) = 0;
      virtual void addGroup( agxCollide::Geometry* geometry, agx::UInt32 id ) = 0;
      virtual void addGroup( agxCollide::Geometry* geometry, const agx::Name& name ) = 0;
      virtual void removeGroup( agxCollide::Geometry* geometry, agx::UInt32 id ) = 0;
      virtual void removeGroup( agxCollide::Geometry* geometry, const agx::Name& name ) = 0;

      virtual bool add( agx::RigidBody* body, bool addGeometries = true ) = 0;
      virtual bool remove( agx::RigidBody* body, bool removeGeometries = true ) = 0;
      virtual void setEnable( agx::RigidBody* body, bool enable ) = 0;

      virtual bool add( agx::Constraint* c ) = 0;
      virtual bool remove( agx::Constraint* c ) = 0;

      virtual bool add( agx::StrongInteraction* si ) = 0;
      virtual bool remove( agx::StrongInteraction* si ) = 0;

      virtual bool add( agxSDK::Assembly* assembly ) = 0;
      virtual bool remove( agxSDK::Assembly* assembly ) = 0;

      virtual bool add( agxSDK::EventListener* listener ) = 0;
      virtual bool remove( agxSDK::EventListener* listener) = 0;

      virtual void addGeometryContacts( const agxCollide::LocalGeometryContactVector& contacts ) = 0;

      virtual void transferToDefaultStorage( agx::ContactMaterial* material ) = 0;
      virtual void transferToSimulationStorage( agx::ContactMaterial* material ) = 0;

      virtual bool empty() const = 0;

      virtual bool getRunningParallel() const = 0;

      virtual void preParallelRun() = 0;
      virtual void postParallelRun() = 0;
      virtual void commit( agxSDK::Simulation* simulation ) = 0;

      virtual agx::Real getTimeStep() const = 0;

      virtual const agx::GravityField* getGravityField() const = 0;

      virtual const agxSDK::MaterialManager* getMaterialManager() const = 0;

      virtual const agxSDK::Simulation* getSimulation() const = 0;

      virtual void setSimulation( agxSDK::Simulation* simulation ) = 0;
      virtual agxSDK::Simulation* getSimulationUnsafe() const = 0;
  };

  typedef agx::ref_ptr< SimulationProxy > SimulationProxyRef;
}

#endif
