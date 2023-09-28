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

#include <agx/config/AGX_USE_WEBSOCKETS.h>
#include <agxSDK/agxSDK.h>
#include <agxSDK/SimulationProxy.h>
#include <agxSDK/SimulationSerializer.h>
#include <agxSDK/EventManager.h>
#include <agxSDK/MaterialManager.h>
#include <agxSDK/Assembly.h>
#include <agxSDK/StatisticsEntries.h>
#include <agxSDK/SimulationStatisticsListener.h>

#include <agx/DynamicsSystem.h>
#include <agx/TimeStamp.h>
#include <agx/HashTable.h>
#include <agx/Component.h>
#include <agx/GravityField.h>
#include <agx/Clock.h>
#include <agx/ParticleContactSensor.h>
#include <agxData/Frame.h>

#include <agxControl/ActionManager.h>

#include <agxCollide/Space.h>

#include <agxSDK/SimulationParameter.h>

#ifdef _MSC_VER
# pragma warning(push)
# pragma warning( disable : 4290 ) // C++ exception specification ignored except to indicate a function is not __declspec(nothrow)
#endif

namespace agx
{
  class Interaction;
  class ParticleSystem;
  class RigidBody;
  class MergeSplit;
  class MergedBody;
  class ParticleEmitter;
  class RigidBodyEmitter;
  class Journal;
  class RemoteSolver;
  class Solver;
  class ObserverFrame;
  class StabilityReport;

  class Hinge;
  class Prismatic;
  class DistanceJoint;
  class LockJoint;
  class BallJoint;
  class PlaneJoint;
  class AngularLockJoint;
  class CylindricalJoint;
  class UniversalJoint;
  class PrismaticUniversalJoint;
}

namespace agxCollide
{
  class Geometry;
}

namespace agxNet
{
  class RemoteDebugClient;
}

namespace agxRender
{
  class RenderManager;
}

namespace agxWire
{
  class Wire;
}

namespace agxModel
{
  class SurfaceVelocityConveyorBelt;
}

namespace agxPlot
{
  class System;
  class DataSeries;
}

namespace agxSDK
{
  class SimulationFrameReader;
  class SimulationFrameWriter;
  class EventListener;
  class SimulationFrameWriter;
  class GuiEventAdapterImplementation;
  class PickHandler;
  class MergeSplitHandler;
  class TerrainInstance;
  class EnergyManager;

  typedef agx::HashVector<agxSDK::Assembly*, agxSDK::AssemblyRef> AssemblyHash;

  AGX_DECLARE_POINTER_TYPES(Simulation);
  /**
  Simulation is a class that bridges the collision space agxCollide::Space and the dynamic simulation system
  agx::DynamicsSystem. All add/remove of objects must go through a the interface for Simulation.
  */
  class CALLABLE AGXPHYSICS_EXPORT Simulation : public agx::Component
  {
    public:
      typedef agx::Event2<Simulation*, agxCollide::Geometry*> GeometryEvent;
      typedef agx::Event2<Simulation*, agx::RigidBody*> RigidBodyEvent;
      typedef agx::Event2<Simulation*, agx::Constraint*> ConstraintEvent;
      typedef agx::Event2<Simulation*, agx::Material*> MaterialEvent;
      typedef agx::Event2<Simulation*, agx::ContactMaterial*> ContactMaterialEvent;
      typedef agx::Event2<Simulation*, agx::Journal*> JournalAttachEvent;

      GeometryEvent addGeometryEvent;
      GeometryEvent removeGeometryEvent;

      RigidBodyEvent addRigidBodyEvent;
      RigidBodyEvent removeRigidBodyEvent;

      ConstraintEvent addConstraintEvent;
      ConstraintEvent removeConstraintEvent;

      MaterialEvent addMaterialEvent;
      MaterialEvent removeMaterialEvent;

      ContactMaterialEvent addContactMaterialEvent;
      ContactMaterialEvent removeContactMaterialEvent;

      JournalAttachEvent journalAttachedEvent;

    public:
      /**
      Creates new simulation proxy of default type.
      \return new simulation proxy of default type
      */
      static agxSDK::SimulationProxyRef createDefaultProxy( agxSDK::Simulation* simulation );

    public:
      /**
      Default constructor
      This constructor CAN Throw a std::runtime_exception if configuration file is broken/invalid.
      */
      Simulation();

      /**
      Add a material to the simulation.
      If a material with the same name already exists, it will return false.
      \return true if adding the material was successfully added to the simulation.
      */
      bool add( agx::Material* material );

      /**
      Add an explicit contact material to the simulation
      \return true if the contact material was successfully added to the simulation
      */
      bool add( agx::ContactMaterial* material );

      /**
      Remove an explicit contact material from the set of existing ContactMaterials
      \return true if the contact material existed and was successfully removed
      */
      bool remove( agx::ContactMaterial* material );

      /**
      \return a material given a name, null if not found.
      */
      const agx::Material* getMaterial(const agx::Name& materialName) const;

      /**
      \return a material given a name, null if not found.
      */
      agx::Material* getMaterial(const agx::Name& materialName);

      /**
      \return a material given a name, null if not found.
      */
      agx::Material* getMaterial(const agx::Uuid& uuid);

      /**
      \return the material manager.
      */
      MaterialManager* getMaterialManager();

      /**
      \return the material manager.
      */
      const MaterialManager* getMaterialManager() const;

      /**
      Remove a specified material from the simulation
      \return true if material was successfully removed
      */
      bool remove( agx::Material* material );

      /*!
      Add a geometry to the Simulation
      Side effect: Will add the material associated to the geometry to the material manager too.
      \return true if adding was successful (and not already added)
      */
      bool add( agxCollide::Geometry* geometry );

      /**
      Remove \p geometry from collision space.
      Any subsequent call to Space::testGeometryOverlap will not contain this geometry.

      Space will keep one reference to the geometry until next call to Space::update() where it will be released.
      Also The Geometry will also possibly be present in the list of contacts and BroadPhase overlaps. This will also
      be cleaned up after next call to Space::update()
      \return true if removal was successful.
      */
      bool remove( agxCollide::Geometry* geometry );

      /*!
      Add a body to the Simulation.
      Side effect: the associated Geometries will also be added to Collision space (if \p addGeometries is == true)
      \param body - The body that will be added to the simulation.
      \param addGeometries - If true, then all geometries in body will also be added to simulation.
      \return true if adding was successful (and not already added)
      */
      bool add( agx::RigidBody* body, bool addGeometries = true );

      /**
      Remove a body from the Simulation, the body will be disabled!
      \param body - The body that should be removed
      \param removeGeometries - If true, any geometries associated with this body should
      also silently be removed. If the removal of any of these geometries fails, it is not reported
      in the return value of this method.
      \return true if the body was successfully removed
      */
      bool remove( agx::RigidBody* body, bool removeGeometries = true );

      /**
      Add a merged body to this simulation.
      \param mergedBody - merged body to add
      \return true if added, otherwise false
      */
      bool add( agx::MergedBody* mergedBody );

      /**
      Removes the merged body from this simulation.
      \param mergedBody - merged body to remove
      \return true if removed, otherwise false
      */
      bool remove( agx::MergedBody* mergedBody );

      /**
      Add a constraint to the simulation.
      \param constraint - The constraint to be added.
      \return true if the constraint was successfully added (and not already added)
      */
      bool add( agx::Constraint* constraint );

      /**
      Remove a constraint from the simulation.
      \param constraint - The constraint to be removed.
      \return true if removal was successful (was part of the simulation)
      */
      bool remove( agx::Constraint* constraint );

      /**
      Add an Interaction to the Simulation.
      \param interaction
      \return true if adding was successful (and not already added)
      */
      bool add( agx::Interaction* interaction );

      /**
      Remove an Interaction from the Simulation
      \param interaction - The object to be removed
      \return true if the interaction was successfully removed
      */
      bool remove( agx::Interaction* interaction );

      /**
      Add an assembly to the simulation including all its parts.
      \param assembly - The assembly to be added
      \param addAllEntries - If true (default) all its contained entries will also be added.
      \return false if one of the elements in the assembly failed to be added
      */
      bool add( agxSDK::Assembly* assembly, bool addAllEntries = true );

      /**
      Remove an assembly from the simulation.
      \param removeAllEntries - If true (default) it will also remove all its contained
      entries, such as geometries, bodies, EventListeners, sub-assemblies and constraints
      \return true if assembly and all its contained elements (if removeAllEntries==true are) successfully removed
      */
      bool remove( agxSDK::Assembly* assembly, bool removeAllEntries = true );

      /**
      Add a sub component to the Simulation
      \param subComponent - The component to be added as a sub component to this Simulation
      \return true if adding the component was successful.
      */
      bool add( agx::Component* subComponent );

      /**
      Remove a sub component from Simulation
      \param subComponent - The component to be removed
      \return true if the removal was successful.
      */
      bool remove( agx::Component* subComponent );

      /**
      Add a ObserverFrame to the Simulation
      \param observerFrame the frame to add
      \return true if adding the frame was successful
      */
      bool add(agx::ObserverFrame* observerFrame);

      /**
      Remove a ObserverFrame to the Simulation
      \param observerFrame the frame to remove
      \return true if removal of the frame was successful
      */
      bool remove(agx::ObserverFrame* observerFrame);

      /**
      Is the object contained in the simulation?
      \param object - The object to test
      \return true if the object is contained in the simulation.
      \note Might be linear complexity in all the objects of the same type in the simulation.
      */
      bool contains( const agx::Component* object ) const;

      /**
      Register an action to this simulation.
      \param action - action to add
      \return true if added (and not already added)
      */
      bool add( agxControl::Action* action );

      /**
      Remove an action from this simulation.
      \param action - action to remove
      \return true if removed
      */
      bool remove( agxControl::Action* action );

      /**
      Register an operation to this simulation.
      \param operation - operation to add
      \return true if added (and not already added)
      */
      bool add( agxControl::Operation* operation );

      /**
      Remove an operation from this simulation.
      \param operation - operation to remove
      \return true if removed (and not already added)
      */
      bool remove( agxControl::Operation* operation );

      /**
      Add a particle system to the simulation.
      \param particleSystem - The particle system to be added.
      \return true if the particle system was successfully added.
      */
      bool add( agx::ParticleSystem* particleSystem );

      /**
      Remove a particle system from the simulation.
      \param particleSystem - The particle system to be removed.
      \return true if the particle system was successfully removed.
      */
      bool remove( agx::ParticleSystem* particleSystem);

      /**
      Add an emitter to the Simulation.
      \param emitter - The emitter to be added.
      \param addGeometries - true if the emitter geometry should be
                             added to the simulation, false otherwise. (Default: true)
      \return true if the emitter was successfully added.
      */
      bool add( agx::Emitter* emitter, bool addGeometries=true);

      /**
      Remove an emitter from the system.
      \param emitter - The emitter to be removed.
      \param removeGeometries - true if the emitter geometry should be
                               removed from the simulation, false otherwise. (Default: true)
      \return true if the emitter was successfully removed.
      */
      bool remove(agx::Emitter* emitter, bool removeGeometries=true);

      /**
      Add a particle contact sensor to the Simulation.
      \param sensor - The particle contact sensor to be added.
      \return true if the particle contact sensor was successfully added.
      */
      bool add( agx::ParticleContactSensor* sensor );

      /**
      Remove a particle contact sensor from the system.
      \param sensor - The particle contact sensor to be removed.
      \return true if the particle contact sensor was successfully removed.
      */
      bool remove( agx::ParticleContactSensor* sensor );

      /**
      * \return a vector of all particle emitters in the simulation
      */
      agx::ParticleEmitterRefVector getParticleEmitters();

      /**
      * \return a vector of all emitters in the simulation, including ParticleEmitters and RigidBodyEmitters
      */
      agx::EmitterRefVector getEmitters();

      /**
      \return a vector of all particle systems which is added to this Simulation
      */
      agx::ParticleSystemRefVector& getParticleSystems();

      /**
      \return a vector of all particle systems which is added to this Simulation
      */
      const agx::ParticleSystemRefVector& getParticleSystems() const;

      /**
      \return The active particle system in the simulation.
      */
      agx::ParticleSystem* getParticleSystem() const;

      /**
      Add terrain (agxTerrain::Terrain) instance to this simulation.
      \param terrain - terrain to add
      \return true if added, false if nullptr or already present in this or another simulation
      */
      bool add( agxSDK::TerrainInstance* terrain );

      /**
      Remove terrain (agxTerrain::Terrain) instance from this simulation.
      \param terrain - terrain to remove
      \return true if removed, false if nullptr or not part of this simulation
      */
      bool remove( agxSDK::TerrainInstance* terrain );

      /**
      Set the uniform gravity.
      If a non-uniform gravity field is currently used (for example PointGravityField), this method will return false and will not change anything.
      If this is the case, and you want a uniform field, you need to create a uniform gravity field: sim->setGravityField(new agx::UniformGravityField(g))
      \param g - Gravity vector
      \return True if the uniform gravity could be set. False if a non-uniform gravity field is currently being used in the DynamicsSystem
      */
      bool setUniformGravity( const agx::Vec3& g );

      /**
      If the current gravity field is not a uniform one (for example PointGravityField), 0,0,0 will be returned.
      \return the uniform gravity
      */
      agx::Vec3 getUniformGravity() const;

      /**
      Set the GravityField model used in gravity calculations. Default is UniformGravityField.
      \param gravityField - The model used.
      */
      void setGravityField( agx::GravityField* gravityField );

      /// \return the current GravityField
      agx::GravityField* getGravityField();

      /// \return the current GravityField
      const agx::GravityField* getGravityField() const;

      /**
      Specify which is the main working thread for the Simulation.
      All calls from event listeners (StepEventListener::pre() etc.) will come from this thread.
      \param thread - A pointer to the thread that should be the main thread.
      */
      void setMainWorkThread(agx::Thread* thread);

      /**
      Find (linear search) and return named RigidBody
      \param name - name of the body to find
      \return pointer to the found RigidBody body, null if not found
      */
      agx::RigidBody* getRigidBody( const agx::Name& name );

      /**
      Find (linear search) and return named RigidBody
      \param name - name of the body to find
      \return pointer to the found RigidBody body, null if not found
      */
      const agx::RigidBody* getRigidBody( const agx::Name& name ) const;

      /**
      Find (linear search) and return a pointer to a RigidBody with the given uuid
      \param uuid - uuid of the requested rigid body
      \return pointer to the found RigidBody, null if not found
      */
      agx::RigidBody* getRigidBody( const agx::Uuid& uuid ) ;

      /**
      Find (linear search) and return a pointer to a RigidBody with the given uuid
      \param uuid - uuid of the requested rigid body
      \return pointer to the found RigidBody, null if not found
      */
      const agx::RigidBody* getRigidBody( const agx::Uuid& uuid ) const;

      /**
      Find (linear search) and return named MergedBody
      \param name - name of the body to find
      \return pointer to the found MergedBody body, null if not found
      */
      agx::MergedBody* getMergedBody(const agx::Name& name);

      /**
      Find (linear search) and return named MergedBody
      \param name - name of the body to find
      \return pointer to the found MergedBody body, null if not found
      */
      const agx::MergedBody* getMergedBody(const agx::Name& name) const;

      /**
      Find (linear search) and return a pointer to a MergedBody with the given uuid
      \param uuid - uuid of the requested rigid body
      \return pointer to the found MergedBody, null if not found
      */
      agx::MergedBody* getMergedBody(const agx::Uuid& uuid);

      /**
      Find (linear search) and return a pointer to a MergedBody with the given uuid
      \param uuid - uuid of the requested rigid body
      \return pointer to the found MergedBody, null if not found
      */
      const agx::MergedBody* getMergedBody(const agx::Uuid& uuid) const;

      /**
      Find (linear search) and return a pointer to a Constraint with the given uuid
      \param uuid - uuid of the requested constraint
      \return const pointer to the found constraint, null if not found
      */
      const agx::Constraint* getConstraint( const agx::Uuid& uuid ) const;

      /**
      Find and return a pointer to a Constraint with the given uuid
      \param uuid - uuid of the requested constraint
      \return pointer to the found Constraint, null if not found
      */
      agx::Constraint* getConstraint( const agx::Uuid& uuid );

      /**
      Find (linear search) and return named Constraint
      \param name - name of the Constraint to find
      \return pointer to the found Constraint, null if not found
      */
      const agx::Constraint* getConstraint(const agx::Name& name) const;

      /**
      Find (linear search) and return named Constraint
      \param name - name of the Constraint to find
      \return pointer to the found Constraint, null if not found
      */
      agx::Constraint* getConstraint(const agx::Name& name);

      /**
      Find (linear search) and return named Assembly
      \param name - name of the Assembly to find
      \return pointer to the found Assembly, null if not found
      */
      const agxSDK::Assembly* getAssembly(const agx::Name& name) const;

      /**
      Find (linear search) and return named Assembly
      \param name - name of the Assembly to find
      \return pointer to the found Assembly, null if not found
      */
      agxSDK::Assembly* getAssembly(const agx::Name& name);

      /**
      Find (linear search) and return an Assembly matching the given uuid
      \param uuid - uuid of the Assembly to find
      \return pointer to the found Assembly, null if not found
      */
      const agxSDK::Assembly* getAssembly(const agx::Uuid& uuid) const;

      /**
      Find (linear search) and return an Assembly matching the given uuid
      \param uuid - uuid of the Assembly to find
      \return pointer to the found Assembly, null if not found
      */
      agxSDK::Assembly* getAssembly(const agx::Uuid& uuid);

      /**
      Find (linear search) and return an emitter matching the given uuid
      \param uuid - uuid of the emitter to find
      \return pointer to the found emitter, null if not found
      */
      const agx::Emitter* getEmitter(const agx::Uuid& uuid) const;

      /**
      Find (linear search) and return an emitter matching the given uuid
      \param uuid - uuid of the emitter to find
      \return pointer to the found emitter, null if not found
      */
      agx::Emitter* getEmitter(const agx::Uuid& uuid);

      /**
      Find (linear search) and return an emitter matching the given name
      \param name - name of the emitter to find
      \return pointer to the found emitter, null if not found
      */
      agx::Emitter* getEmitter(const agx::Name& name) const;

      /**
      Find (linear search) and return an emitter matching the given name
      \param name - name of the emitter to find
      \return pointer to the found emitter, null if not found
      */
      agx::Emitter* getEmitter(const agx::Name& name);

      /**
      Find and return named Constraint of a templated type, for example Hinge:

      agx::Hinge *hinge = simulation->getConstraint<agx::Hinge>(uuid);

      \param uuid - uuid of the Constraint to find
      \return typed pointer to the found Constraint, null if not found
      */
      template < typename T >
      T* getConstraint(const agx::Uuid& uuid);

      /**
      Find and return named Constraint of a templated type, for example Hinge:

      const agx::Hinge* hinge = simulation->getConstraint<agx::Hinge>(uuid);

      \param uuid - uuid of the Constraint to find
      \return typed pointer to the found Constraint, null if not found
      */
      template < typename T >
      const T* getConstraint(const agx::Uuid& uuid) const;

      /**
      Find and return named Constraint of a templated type, for example Hinge:

      agx::Hinge *hinge = simulation->getConstraint<agx::Hinge>("FrontWheelHinge");

      \param name - name of the Constraint to find
      \return typed pointer to the found Constraint, null if not found
      */
      template < typename T >
      T* getConstraint( const agx::Name& name );

      /**
      Find and return named Constraint of a templated type, for example Hinge:

      const agx::Hinge* hinge = simulation->getConstraint<agx::Hinge>("FrontWheelHinge");

      \param name - name of the Constraint to find
      \return typed pointer to the found Constraint, null if not found
      */
      template < typename T >
      const T* getConstraint( const agx::Name& name ) const;

#define GETCONSTRAINT_TYPE(CLASS_NAME, TYPENAME)    \
      CLASS_NAME *get ## TYPENAME(const agx::Name& name); \
      CLASS_NAME *get ##TYPENAME(const agx::Uuid& uuid);

      GETCONSTRAINT_TYPE(agx::AngularLockJoint, AngularLockJoint)
      GETCONSTRAINT_TYPE(agx::BallJoint, BallJoint)
      GETCONSTRAINT_TYPE(agx::CylindricalJoint, CylindricalJoint)
      GETCONSTRAINT_TYPE(agx::DistanceJoint, DistanceJoint)
      GETCONSTRAINT_TYPE(agx::Hinge, Hinge)
      GETCONSTRAINT_TYPE(agx::LockJoint, LockJoint)
      GETCONSTRAINT_TYPE(agx::PlaneJoint, PlaneJoint)
      GETCONSTRAINT_TYPE(agx::Prismatic, Prismatic)
      GETCONSTRAINT_TYPE(agx::PrismaticUniversalJoint, PrismaticUniversalJoint)
      GETCONSTRAINT_TYPE(agx::UniversalJoint, UniversalJoint)
#undef GETCONSTRAINT_TYPE

      /**
      Find (linear search) and return named collision Geometry
      \param name - name of the collision Geometry to find
      \return pointer to the found collision Geometry, null if not found
      */
      const agxCollide::Geometry* getGeometry(const agx::Name& name ) const;

      /**
      Find (linear search) and return named collision Geometry
      \param name - name of the collision Geometry to find
      \return pointer to the found collision Geometry, null if not found
      */
      agxCollide::Geometry* getGeometry(const agx::Name& name );

      /**
      Find (linear search) and return a geometry matching the given uuid
      \param uuid - uuid of the collision Geometry to find
      \return pointer to the found collision Geometry, null if not found
      */
      const agxCollide::Geometry* getGeometry( const agx::Uuid& uuid ) const;

      /**
      Find (linear search) and return a geometry matching the given uuid
      \param uuid - uuid of the collision Geometry to find
      \return pointer to the found collision Geometry, null if not found
      */
      agxCollide::Geometry* getGeometry( const agx::Uuid& uuid );

      /**
      Find and return the first matching named event listener
      \param name - name of the event listener to find
      \return pointer to the found event listener, null if not found
      */
      const agxSDK::EventListener* getEventListener(const agx::Name& name ) const;

      /**
      Find and return the first matching named event listener
      \param name - name of the event listener to find
      \return pointer to the found event listener, null if not found
      */
      agxSDK::EventListener* getEventListener(const agx::Name& name );

      /**
      Find (linear search) and return the first event listener matching the uuid
      \param uuid - uuid of the event listener to find
      \return pointer to the found event listener, null if not found
      */
      const agxSDK::EventListener* getEventListener(const agx::Uuid& uuid) const;

      /**
      Find (linear search) and return the first event listener matching the uuid
      \param uuid - uuid of the event listener to find
      \return pointer to the found event listener, null if not found
      */
      agxSDK::EventListener* getEventListener(const agx::Uuid& uuid);


      /**
      Find and return the first matching named wire
      \param name - name of the wire to find
      \return pointer to the found wire, null if not found
      */
      const agxWire::Wire* getWire(const agx::Name& name) const;

      /**
      Find and return the first matching named wire
      \param name - name of the wire to find
      \return pointer to the found wire, null if not found
      */
      agxWire::Wire* getWire(const agx::Name& name);

      /**
      Find (linear search) and return the first wire matching the uuid
      \param uuid - uuid of the wire to find
      \return pointer to the found wire, null if not found
      */
      const agxWire::Wire* getWire(const agx::Uuid& uuid) const;

      /**
      Find (linear search) and return the first wire matching the uuid
      \param uuid - uuid of the wire to find
      \return pointer to the found wire, null if not found
      */
      agxWire::Wire* getWire(const agx::Uuid& uuid);

      /**
      Find (linear search) and return named ObserverFrame
      \param name - name of the observer to find
      \return pointer to the found ObserverFrame, null if not found
      */
      agx::ObserverFrame* getObserver(const agx::Name& name);

      /**
      Find (linear search) and return named ObserverFrame
      \param name - name of the observer to find
      \return pointer to the found ObserverFrame, null if not found
      */
      const agx::ObserverFrame* getObserver(const agx::Name& name) const;

      /**
      Find (linear search) and return a pointer to a ObserverFrame with the given uuid
      \param uuid - uuid of the requested observer frame
      \return pointer to the found ObserverFrame, null if not found
      */
      agx::ObserverFrame* getObserver(const agx::Uuid& uuid);

      /**
      Find (linear search) and return a pointer to a ObserverFrame with the given uuid
      \param uuid - uuid of the requested observer frame
      \return pointer to the found ObserverFrame, null if not found
      */
      const agx::ObserverFrame* getObserver(const agx::Uuid& uuid) const;

      /**
      Step the simulation forward 1 or more time steps in time until we get to the time t.
      The length of time step is determined by the TimeGovernor in the DynamicsSystem.
      \return the current timestamp of the system. This can be now+t, but it can also be less due to a call
      to the breakStepTo() method from a method in an EventListener.
      */
      agx::TimeStamp stepTo( agx::TimeStamp t );

      /// \return the time step determined by the TimeGovernor in the DynamicsSystem.
      agx::Real getTimeStep() const;

      /**
      Sets the time step determined by the TimeGovernor in the DynamicsSystem.
      \param timeStep - The new time step.
      \return true if the timestep was set successfully, false otherwise.
      */
      bool setTimeStep( agx::Real timeStep );

      /**
      \return The clock.
      */
      agx::Clock* getClock();

      /**
      \return The clock.
      */
      const agx::Clock* getClock() const;

      /**
      \return The attached simulation journal.
      */
      agx::Journal* getJournal();

      /**
      \return The attached simulation journal.
      */
      const agx::Journal* getJournal() const;

  #if AGX_USE_WEBSOCKETS()
      /**
      \return The attached remote solver.
      */
      agx::RemoteSolver* getRemoteSolver();

      /**
      \return The attached remote solver.
      */
      const agx::RemoteSolver* getRemoteSolver() const;
  #endif

      /**
      Break an initiated stepTo() after the current simulation step.
      When stepTo() is called, a number if iterations calling stepForward() will be done.
      If you want to cancel these iterations from within an EventListener you can call breakStepTo().
      \return true if a started stepTo() is stopped, otherwise it returns false and does nothing.
      */
      bool breakStepTo();

      /**
      \return The current accumulated simulation time.
      */
      agx::Real getTimeStamp() const;

      /**
      Set the timestamp for the simulation (and the DynamicsSystem)
      \param t - New timestamp
      */
      void setTimeStamp( agx::TimeStamp t );

      /**
      Take one step forward in the simulation.
      Length of time step is determined by the TimeGovernor in the DynamicsSystem
      */
      void stepForward();

      /**
      \return A pointer to the Collision space used in the Simulation
      */
      agxCollide::Space* getSpace();

      /**
      \return A pointer to the Collision space used in the Simulation
      */
      const agxCollide::Space* getSpace() const;

      /**
      \return A pointer to the DynamicsSystem used in the Simulation
      */
      agx::DynamicsSystem* getDynamicsSystem();

      /**
      \return A pointer to the DynamicsSystem used in the Simulation
      */
      const agx::DynamicsSystem* getDynamicsSystem() const;

      /**
      \return A pointer to the PlotSystem used in the Simulation
       */
      agxPlot::System* getPlotSystem() const;

      /**
      Replace the current PlotSystem with \p plotSystem
      */
      void setPlotSystem(agxPlot::System* plotSystem);

      /**
      Replace the current Collision space with \p space
      */
      void setSpace( agxCollide::Space* space );

      /**
      Replace the current DynamicsSystem with \p system
      */
      void setDynamicsSystem( agx::DynamicsSystem* system );


      /**
      Remove the specified listener. The listener will get a call to its removeNotification method
      if removal was successful.
      \return true if the listener was successfully removed.
      */
      bool removeEventListener( agxSDK::EventListener* listener );

      /**
      Add an event listener
      The listener will get a call to its addNotification method if adding was successful.

      \param listener - Pointer to the listener that will be activated during events.

      \param priority - An execution priority for this listener. Higher executed earlier in the list of added listeners.
      Range of priority is [EventManager::LOWEST_PRIORITY, EventManager::HIGHEST_PRIORITY]
      \return false if listener was already added, or if priority is outside of the valid range, otherwise true.
      */
      bool addEventListener( agxSDK::EventListener* listener, int priority = EventManager::DEFAULT_PRIORITY );

      /**
      Add an event listener
      The listener will get a call to its addNotification method if adding was successful.

      \param listener - Pointer to the listener that will be activated during events.

      \param priority - An execution priority for this listener. Higher executed earlier in the list of added listeners.
      Range of priority is [EventManager::LOWEST_PRIORITY, EventManager::HIGHEST_PRIORITY]
      \return false if listener was already added, or if priority is outside of the valid range, otherwise true.
      */
      bool add( agxSDK::EventListener* listener, int priority = EventManager::DEFAULT_PRIORITY );

      /**
      Remove a Event listener.
      \param listener The listener to remove.
      \retval true if successfully removed (and part of Simulation before).
      */
      bool remove( agxSDK::EventListener* listener );

      /**
      Execute a GuiEvent
      \return true if event is listened to and handled.
      */
      bool triggerEvent( const GuiEvent& e );

      /**
      A data series that contains all time stamps of the simulation.
      */
      agxPlot::DataSeries* getTimeDataSeries() const;

#ifdef SWIG
      enum CleanupSelectionMask {
        STEP_LISTENERS = 0x1,                                 //!< Remove all step contact listeners
        CONTACT_LISTENERS = 0x2,                              //!< Remove all contact contact listeners
        GUI_LISTENERS = 0x4,                                  //!< Remove all gui event listeners
        LISTENERS = STEP_LISTENERS | CONTACT_LISTENERS | GUI_LISTENERS,  //!< Remove all listeners
        SYSTEM = 0x200,                                       //!< Remove the dynamic system
        SPACE = 0x400,                                        //!< Remove the collision space
        MATERIALS = 0x800,                                    //!< Remove all materials
        ASSEMBLIES = 0x1000,                                  //!< Remove all assemblies
        PARTICLE_SYSTEMS = 0x2000,                            //!< Remove all particle systems
        CONTACT_DATA = 0x4000,                                //!< Remove all contact data
        CLEANUP_ALL = LISTENERS | SYSTEM | SPACE | MATERIALS | ASSEMBLIES | PARTICLE_SYSTEMS | CONTACT_DATA
      };
#else
      /**
      Specification of what should be cleaned up from a simulation during a call to the cleanup() method.
      These flags can be bit masked/or:ed together.
      */
      enum CleanupSelectionMask {
        STEP_LISTENERS = EventManager::STEP_LISTENERS,        //!< Remove all step contact listeners
        CONTACT_LISTENERS = EventManager::CONTACT_LISTENERS,  //!< Remove all contact contact listeners
        GUI_LISTENERS = EventManager::GUI_LISTENERS,          //!< Remove all gui event listeners
        LISTENERS = EventManager::LISTENERS,                  //!< Remove all listeners
        SYSTEM = 0x200,                                       //!< Remove the dynamic system
        SPACE = 0x400,                                        //!< Remove the collision space
        MATERIALS = 0x800,                                    //!< Remove all materials
        ASSEMBLIES = 0x1000,                                  //!< Remove all assemblies
        PARTICLE_SYSTEMS = 0x2000,                            //!< Remove all particle systems
        CONTACT_DATA = 0x4000,                                //!< Remove all contact data
        PLOTSYSTEM = 0x8000,                                  //!< Remove plotsystem
        CLEANUP_ALL = LISTENERS | SYSTEM | SPACE | MATERIALS | ASSEMBLIES | PARTICLE_SYSTEMS | CONTACT_DATA | PLOTSYSTEM
      };
#endif

      /**
      Enum for specifying some additional, part of a simulation that can be restored from
      a serialization.
      */
      enum ReadSelectionMask {
        READ_NONE = 0x0,             ///<! Select to read none of the items below
        READ_TIMESTEP = 0x1,         ///<! Select to read and restore the TimeStep of the Simulation.
        READ_TIMESTAMP = 0x2,        ///<! Select to read the TimeStamp (current time of the simulation)
        READ_GRAVITY = 0x4,          ///<! Select to read and restore the Gravity model of the simulation.
        READ_SOLVER = 0x8,           ///<! Select to read solver parameters.
        READ_ALL = READ_TIMESTEP + READ_TIMESTAMP + READ_GRAVITY + READ_SOLVER,
        READ_DEFAULT = READ_TIMESTEP /* + READ_TIMESTAMP */ + READ_GRAVITY + READ_SOLVER // Not using READ_TIMESTAMP
      };

      /**
      This method will remove/cleanup selected parts of the simulation.
      By using the CleanupSelectionMask one can select what should be removed, for example all contact listeners.
      \param selection - Specify from enum CleanupSelectionMask what should be cleaned up
      \param fast - If true, we will try to remove bodies, geometries, listeners one by one instead of creating new DynamicsSystem,
      Space etc. which does have a lot of overhead.
      */
      void cleanup( agx::UInt selection = CLEANUP_ALL, bool fast=false );

      /**
      Truncating an AGX simulation does not reset the timestamp in the action
      manager. This function allows for resetting the action manager
      timestamp from Simulation.h
      */
      void setActionManagerTimeStamp(const agx::TimeStamp& time);

      /**
      Gets the action manager.
      */
      agxControl::ActionManager* getActionManager();

      /**
      Clears contact data in the simulation
      */
      void clearContactData();

      /**
      \return the solver.
      */
      agx::Solver* getSolver();

      /**
      \return the solver.
      */
      const agx::Solver* getSolver() const;

      /**
      Set true to integrate positions at the start of the timestep rather than at the end.
      */
      void setPreIntegratePositions( bool flag );

      /**
      \return True if the positions are integrated at the start of the timestep, false if integrated at the end.
      */
      bool getPreIntegratePositions() const;

      /**
      Add an object as a serializable object.
      Might be any of the above classes having an add method: Geometry, RigidBody etc.
      \retval: successfully added (and not already added)?
      */
      bool add( agxStream::Serializable* object );

      /**
      Remove an object as a serializable object.
      Might be any of the above classes having an add method: Geometry, RigidBody etc.
      \retval: successfully removed?
      */
      bool remove( agxStream::Serializable* object );


      /**
      \return the interface to the pick handler associated to this simulation object
      \note If no pick handler has been set, these methods will return zero.
      */
      agxSDK::PickHandler* getPickHandler();

      /**
      \return the interface to the pick handler associated to this simulation object
      \note If no pick handler has been set, these methods will return zero.
      */
      const agxSDK::PickHandler* getPickHandler() const;

      /**
      Assign interface for pick handler.
      \param pickHandler - pick handler interface
      */
      void setPickHandler( agxSDK::PickHandler* pickHandler );

      /**
      Specify the path to a file where the statistics information will be written when enabled.
      \param path - The path to a filename into which statistics information will be written.
      */
      void setStatisticsPath( const agx::String& path );

      /**
      \param interval - specifies the interval in seconds for writing statistics data down to disk
      */
      void setStatisticsInterval( agx::Real interval );

      /**
      Enable/disable statistics logging to disk.
      This will also toggle the agx::Statistics::instance()->setEnable() which will
      control whether statistics data is logged at all.
      */
      void setEnableStatistics( bool enable );

      /// \return true if statistics logging is enabled
      bool getEnableStatistics() const;

      /// \param enable - if true, the rendering of statistics will be dispatched to the agxRender::RenderManager
      void setEnableStatisticsRendering( bool enable );

      /// \return true if rendering of statistics is enabled
      bool getEnableStatisticsRendering() const;

      /**
      Specify the frequency for performance profiling.
      */
      void setProfilingFrequency(agx::Real frequency);

      /**
      \return The profiling frequency
      */
      agx::Real getProfilingFrequency() const;

      /**
      \return all assemblies in the added to the Simulation.
      */
      const AssemblyHash& getAssemblies() const;


      /**
      Get a vector containing all the registered event listeners.
      */
      void getEventListeners( EventListenerPtrVector& eventListeners ) const;

      /**
      \return vector of geometries which has been added to this space
      */
      const agxCollide::GeometryRefVector& getGeometries() const;

      /**
      \return a vector of all bodies in the DynamicsSystem, both enabled and disabled
      */
      const agx::RigidBodyRefVector& getRigidBodies() const;

      /**
      \return a vector of all bodies in the DynamicsSystem, both enabled and disabled
      */
      agx::RigidBodyRefVector& getRigidBodies();


      /**
      \return All constraints in the system.
      */
      const agx::ConstraintRefSetVector& getConstraints() const;

      /**
      \return All constraints in the system.
      */
      agx::ConstraintRefSetVector& getConstraints();

      /**
      \return All observer frames in the system.
      */
      const agx::ObserverFrameRefSetVector& getObserverFrames() const;

      /**
      \return All observer frames in the system.
      */
      agx::ObserverFrameRefSetVector& getObserverFrames();

      /**
      \return a set of all enabled bodies in this system
      */
      agx::RigidBodyPtrSetVector& getEnabledRigidBodies();

      /**
      \return a set of all enabled bodies in this system
      */
      const agx::RigidBodyPtrSetVector& getEnabledRigidBodies() const;

      /**
      Find (linear search) and return a pointer to a ObserverFrame with the given uuid
      \param uuid - uuid of the requested observer frame
      \return const pointer to the found observer frame, null if not found
      */
      const agx::ObserverFrame* getObserverFrame(const agx::Uuid& uuid) const;

      /**
      Find and return a pointer to a ObserverFrame with the given uuid
      \param uuid - uuid of the requested observer frame
      \return pointer to the found observer frame, null if not found
      */
      agx::ObserverFrame* getObserverFrame(const agx::Uuid& uuid);

      /**
      Find (linear search) the first ObserverFrame object in the system that matches the name
      \return a pointer to the found observer frame, nullptr if none found.
      */
      agx::ObserverFrame* getObserverFrame(const agx::Name& name);

      /**
      Find (linear search) the first ObserverFrame object in the system that matches the name
      \return a pointer to the found observer frame, nullptr if none found.
      */
      const agx::ObserverFrame* getObserverFrame(const agx::Name& name) const;


      /**
      Serialize the simulation to the specified file
      \param filename - Name of file to which the simulation will be written
      \returns Number of objects serialized to disk
      */
      size_t write( const agx::String& filename ) const;

      /**
      Serialize the simulation to the specified file
      \param stream - The stream to which the simulation object will be written
      \param binary - if true, it's a binary dataset.
      \returns Number of objects serialized to disk
      */
      size_t write( std::ostream& stream, bool binary ) const;

      /**
      Read serialized simulation from the specified file and append to this simulation
      \param parent - If != nullptr, all read objects will be added to this parent too.
      \param selection - Options for reading in. See ReadSelectionMask.
      \return number of serialized objects read from file
      */
      size_t read( const agx::String& filename, agxSDK::Assembly* parent = nullptr, agx::UInt selection = READ_DEFAULT);

      /**
      Read serialized simulation from the specified stream.
      \param stream - Stream from which serialized data is to be read.
      \param isBinary - if true, it's a binary dataset.
      \param parent - If != nullptr, all read objects will be added to this parent too.
      \param selection - Options for reading in. See ReadSelectionMask.
      \return number of serialized objects read from file
      */
      size_t read( std::istream& stream, bool isBinary, agxSDK::Assembly* parent = nullptr, agx::UInt selection = READ_DEFAULT);

      /**
      Restore state from a serialized scene file. Matching object UUIDs, so any object present
      in both current simulation and the serialized scene will retain the current object.
      \return true if successful
      */
      bool restore(const agx::String& filename, agx::UInt selection = READ_DEFAULT);
      bool restore(std::istream& stream, bool isBinary, agx::UInt selection = READ_DEFAULT);


      /**
      Add a restore listener to the Simulation. Whenever a simulation is restored from a serialization, these
      listeners will be called upon.

      \param listener - A pointer to a RestoreListener
      \return true if listener did not previously existed, and hence is added
      */
      bool addRestoreListener( agxStream::RestoreListener *listener );

      /**
      Remove an existing listener in a simulation.
      \param listener The listener to be removed
      \return true if listener existed and hence was removed
      */
      bool removeRestoreListener(  agxStream::RestoreListener *listener );

      /**
      Save this simulation to the output archive
      \param archive - Archive associated to a stream to which the data will be written.
      */
      void store( agxStream::OutputArchive& archive ) const;

      /**
      Read data from the input archive and add to the simulation.
      \param in - Archive to read from
      \param parent - if != nullptr, all items will also be added to this assembly
      \param selection - Options for reading in. See ReadSelectionMask.
      \param replaceUsingCache - Replace current state by archive, but reuse matching objects that exist in both representations
      */
      void restore( agxStream::InputArchive& in, agxSDK::Assembly* parent = nullptr, agx::UInt selection = READ_DEFAULT, bool replaceUsingCache = false );

      /**
      Enable/Disable remote debugging.
      \param flag - If true, a server for remote debugging will be created
      \param port - Socket port. If -1, the default port will be used.
      \param useCompression - If true the data will be compressed before sent to the remote listener
      \param waitForSynchronization - If true the server will wait for the client to ask for more data before it sends another frame
      */
      void setEnableRemoteDebugging( bool flag, agx::Int16 port = -1, bool useCompression = false, bool waitForSynchronization = true );

      /**
      \return true if remote debugging is enabled
      */
      bool getEnableRemoteDebugging() const;

      /**
      Sets the time step for remote debugging. Should be a multiple of the simulation time step.
      If \p timeStep=0, then the simulation time step will be used.
      */
      void setRemoteDebuggingTimeStep( agx::Real timeStep );

      /**
      \return the time step for remote debugging. Should be a multiple of the simulation time step.
      */
      agx::Real getRemoteDebuggingTimeStep() const;

      /**
      Specify the bin resolution used when evaluating contacts for reduction between body-body contacts.
      (Body-body contact reduction is not enabled by default, see settings in ContactMaterial to enable it).
      A high value will keep more contacts, lower will result in more aggressive reduction.
      Commonly a value of 2-3 will give good results. Default is 3.
      Values from 1 to 10 are valid.
      \param binResolution - parameter to ContactReducer.
      */
      void setContactReductionBinResolution( agx::UInt8 binResolution );

      /**
      \return the bin resolution used in body-body contact reduction for this Simulation.
      */
      agx::UInt8 getContactReductionBinResolution() const;

      /**
      Specify the minimum number of contacts between two bodies required to execute contact reduction.
      Default value is 3 contacts.
      \param threshold - Set the lower bound threshold for number of contact points in a body-body rigid body contact in order run contact reduction on it.
      */
      void setContactReductionThreshold(agx::UInt threshold);

      /**
      \return The minimum number of contacts between two bodies required to execute the contact reduction.
      */
      agx::UInt getContactReductionThreshold() const;

      /**
      \return a pointer to the mergeSplit object. nullptr if mergeSplit is not enabled.
      */
      agx::MergeSplit* getMergeSplit();

      /**
      \return a const pointer to the mergeSplit object. nullptr if mergeSplit is not enabled.
      */
      const agx::MergeSplit* getMergeSplit() const;

      /**
      Merge split handler is an object that enables performance boosts by merging rigid bodies
      together which results in fewer rigid bodies for the solver to solve for.
      \return the merge split handler for this simulation
      */
      agxSDK::MergeSplitHandler* getMergeSplitHandler() const;

      /**
      \return a pointer to the RenderManager for this simulation
      */
      agxRender::RenderManager* getRenderManager();

      /**
      \return a pointer to the RenderManager for this simulation
      */
      const agxRender::RenderManager* getRenderManager() const;

      /**
      Make an explicit call to RenderManager::update() to update the debug rendering information.
      */
      void updateRenderManager();

      /**
      Make an explicit call to RenderManager::update() to update the statistics rendering information
      */
      void updateStatisticsRenderData();

      /**
      Use with caution!

      Collect garbage of internal states (e.g., removed geometries and rigid bodies). The garbage will
      default be collected automatically during next simulation step.
      \note After calling this method e.g., the geometry contact list could be invalid. Other internal
            structures may as well become invalid. So in general it is not recommended to call this
            method at all.
      */
      void garbageCollect();

      /**
      Add a parameter to the simulation configuration interface.
      */
      bool add(agxSDK::SimulationParameter *parameter);

      /**
      Remove a parameter from the simulation configuration interface.
      */
      bool remove(agxSDK::SimulationParameter *parameter);

      /**
      \return The set of published simulation parameters.
      */
      const agxSDK::SimulationParameterRefVector& getParameters() const;

      /**
      \return The parameter with a specified name.
      */
      agxSDK::SimulationParameter *getParameter(const agx::Name& name);

      /**
      return the energy manager of the system.
      */
      agxSDK::EnergyManager* getEnergyManager() const;

  public:
      // Internal, but public methods

      /// Send the simulation to the remote debugger (if enabled)
      void sendSimulation();

      /**
      \return a pointer to the SimulationSerializer which is a StepEventListener that will be periodically called
      from the Simulation's stepForward method to store the simulation state to disk (if enabled)
      */
      SimulationSerializer* getSerializer();

      /**
      \return a pointer to the SimulationSerializer which is a StepEventListener that will be periodically called
      from the Simulation's stepForward method to store the simulation state to disk (if enabled)
      */
      const SimulationSerializer* getSerializer() const;

      /**
      Replace the current SimulationSerializer with a new instance
      */
      void setSerializer( SimulationSerializer* serializer);


      /**
      Internal method.

      Update the internal active ActionManager in the simulation
      */
      void updateActionManager();

      /**
      Internal method.

      Proxy of this object used by objects running in parallel and still
      needs to add/remove things from this simulation.
      */
      agxSDK::SimulationProxy* getSimulationProxy() const;

      /**
      Internal method.

      Add content from a simulation proxy to this simulation.
      \param simulationProxy - simulation proxy to commit
      */
      void commit( agxSDK::SimulationProxy* simulationProxy );

      /**
      Internal method.

      Handles stability report sent from the solver.
      */
      void handleStabiltyReport( const agx::StabilityReport* stabilityReport );

      agx::TaskGroup* getUpdateTask();

      void renderComponents();
      void createRenderTask();
      agx::SolveModel* getSolveModel(const agx::Name& name);

      const agx::SolveModelRefVector& getSolveModels() const;

      #if AGX_USE_WEBSOCKETS()
        void setRemoteSolver(agx::RemoteSolver* remoteSolver);
      #endif

      /// Special constructor which will associate the Simulation to a specified device.
      Simulation( agx::Device* device );

      static Simulation* load(agx::TiXmlElement* eSimulation, agx::Device* device);

      /// Internal method
      static agx::Model* ClassModel();

      /**
      Enable/disable the listener that collects all statistics data from the simulation. Disabling will remove/delete
      the previous listener, hence no data will be stored.
      Make sure you update any external pointers to the SimulationStatisticsListener retrieved through the getSimulationStatisticsListener() method.
      */
      void setEnableSimulationStatisticsListener( bool flag );

      SimulationStatisticsListener* getSimulationStatisticsListener();
      const SimulationStatisticsListener* getSimulationStatisticsListener() const;

      void addSolveModel(agx::SolveModel* model);
      void removeSolveModel(agx::SolveModel* model);
      void loadSolveModels(const agx::String& path, bool clear = false);

      bool getUseComplexImpactStage() const;
      void setUseComplexImpactStage(bool flag);

      // For LUA only
      agx::Constraint1DOF* getConstraint1DOF(const agx::Name& name) { return this->getConstraint<agx::Constraint1DOF>(name); }
      agx::Constraint2DOF* getConstraint2DOF(const agx::Name& name) { return this->getConstraint<agx::Constraint2DOF>(name); }

      /**
      \return the number of merged bodies
      */
      agx::UInt getNumMergedBodies() const;

      /**
      \sa getNumMergedBodies
      \return merged body with index i
      */
      agx::MergedBody* getMergedBody( agx::UInt i ) const;

      /**
      \return the number of current terrains in the simulation
      */
      agx::UInt getNumTerrains() const;

      /**
      \param i - index (i < getNumTerrains())
      \return terrain instance at given index \p i
      */
      agxSDK::TerrainInstance* getTerrain( agx::UInt i ) const;

      /// Reset profiling timers
      void resetTaskTimers();

      /**
      Print current task performance timers to a file.
      */
      void dumpTaskTimers(const agx::String& filePath = "BufferAllocations.xml");

      /**
      Print current agxData buffer allocations to a file.
      */
      void dumpMemoryUsage(const agx::String& filePath = "TaskTimers.xml");


      /// \return true if particles are allowed to be created at this point in the simulaion step
      bool allowCreateParticles() const;

      agx::Component *getHeader();
      void readStoragePermutations(agxStream::InputArchive& in, bool clearStorages = false);
      void writeStoragePermutations(agxStream::OutputArchive& archive) const;

      void writeStatisticsFiles(const agx::String& directoryPath = "Statistics", bool taskProfile = true, bool threadTimeline = true, const agx::Vector<agx::String>& timelineFormats = agx::Vector<agx::String>(), bool bufferAllocations = true);

      agxData::EntityStorage* getEmitterStorage();

    protected:
      friend class agxNet::RemoteDebugClient;
      friend class GuiEventAdapterImplementation;
      friend class agx::DynamicsSystem;
      friend class agxSDK::SimulationFrameReader;
      friend class agxRender::RenderManager;

      /// Destructor
      virtual ~Simulation();

    private:
      void addListener(EventListener *listener);
      void removeListener(EventListener *listener);
      bool hasListener(EventListener *listener);

    private:
      ///  Internal method: Init the whole simulation, called from the constructors.
      void init();

      /// Part of the init process, including reading settings.cfg etc.
      void initConfiguration();

      /// Internal method
      void updateStatisticsEnable();

      void setEnableComputeAllConstraintForces(bool flag);
      bool getEnableComputeAllConstraintForces() const;

      /// Internal method: make sure that Graph renderer will be re initialized.
      void resetGraphStatistics();

      /// Internal method
      agx::Vec4 getSubsystemColor(agx::RigidBody* body);

      /// Internal method
      void setupStatisticsLogging();

      void updateConstraintForceData();

      void createUpdateTask();

      void generateContactMaterials();
      void preCollide();
      void prepareContacts();
      void triggerParticleContactSensors();
      void triggerPreStepEvents();
      void triggerPostStepEvents();
      void triggerLastStepEvents();
      void terrainPostUpdate();
      void stepSystem();
      void resetDirtyMaterials();
      void updateRemoteDebugger();
      void recordStatistics();
      void recordPlaybackStatistics();
      void recordParticleCount();
      void recordWarmStartingData();
      void printStatistics();
      void updateClock();


      void pruneContacts();
      void contactPointReduction();
      void updateSleepState();
      void triggerPreCollideStepEvent();
      void triggerSeparationContactEventListeners();
      void triggerContactEventListeners();
      void assignContactMaterials();
      void dumpThreadLogs(agx::Task *);

      void applyPreIntegrationSettings();

    private:
      friend class agx::Solver;
      void registerParticleSystemPointers();
      void unregisterParticleSystemPointers();
      void removeParticleSolverObjects( agx::ParticleSystem* particleSystem );

    private:

      class StatisticsWriter;

      friend class agx::Journal;
      friend class agxSDK::SimulationFrameWriter;
      #if AGX_USE_WEBSOCKETS()
        friend class agx::RemoteSolver;
      #endif

      void setJournal(agx::Journal* journal);
      void setHasJournal(bool flag);
      void preStoreSynchronization();

      StatisticsEntries m_statisticsEntry;
      StatisticsWriter* m_statWriter;

      typedef agx::HashTable<agxCollide::Geometry*, int> GeometryTable;

      AssemblyHash m_assemblies;

      agxCollide::SpaceRef m_space;
      agx::DynamicsSystemRef m_system;

      agx::ref_ptr<agx::Referenced> m_plotSystem;
      agx::ref_ptr<agx::Referenced> m_timeDataSeries;

      agx::ComponentRef m_header;
      bool m_stepToStarted;

      EventManagerRef m_eventManager;
      MaterialManagerRef m_materialManager;

      agxControl::ActionManagerRef m_actionManager;
      agx::TaskRef m_actionManagerTask;

      agx::observer_ptr< Referenced > m_pickHandler;

      SimulationSerializerRef m_serializationListener;

      agx::ParticleSystemRefVector m_particleSystems;
      agx::TaskGroupRef m_updateEmitters;
      agxData::EntityStorageRef m_emitterStorage;

      agx::ref_ptr<agx::Referenced> m_controlledServer;
      agx::ref_ptr<agx::Referenced> m_debugServer;
      agx::UInt8 m_contactReductionBinResolution;
      agx::UInt m_contactReductionThreshold;
      agx::ComponentRefVector m_subComponents;
      agx::Real m_remoteDebuggerTimeStep;
      agx::TimeStamp m_remoteDebuggerTimeStamp;
      agx::TaskGroupRef m_updateTask;
      agx::TaskRefVector m_preIntegrateTasks;
      bool m_preIntegrate;
      bool m_fastCleanup;
      bool m_allowCreateParticles;

      agx::Timer m_stepTimer;
      agx::Timer m_updateRenderManagerTimer;
      agx::ClockRef m_clock;
      agx::TaskRef m_renderTask;
      agx::GravityFieldRef m_gravityField;

      agx::SolveModelTable m_solveModelTable;
      agx::SolveModelRefVector m_solveModels;

      agx::ref_ptr<agx::Referenced> m_renderManager;
      agx::Thread* m_mainWorkThread;
      agxData::FrameRef m_journalLastWrittenFrame;
      agx::UInt32 m_journalAgxArchiveIdCounter;

      void* m_journal;
      bool m_hasJournal;
      void* m_remoteSolver;

      agxData::EntityStorageRef m_constraintForcesStorage;
      bool m_enableComputeAllConstraintForces;
      bool m_inDestructor;
      SimulationProxyRef m_simulationProxy;  /**< Used by wires when running in parallel. */
      agx::ParticleContactSensorRefVector m_particleContactSensors;

      typedef agx::HashTable<agxCollide::Geometry*, agx::ParticleContactSensorPtrVector*> ParticleContactSensorTable;
      ParticleContactSensorTable m_particleContactSensorTable;

      typedef agx::HashTable<agxStream::RestoreListener*, agxStream::RestoreListenerRef> RestoreListenerHash;
      RestoreListenerHash m_restoreListeners;
      agxSDK::SimulationStatisticsListenerRef m_simulationStatisticsListener;
      agx::Real m_profilingFrequency;
      agxData::ValueRefT<bool> m_useComplexImpactStage;

      typedef agx::Vector<agx::ref_ptr< const agx::Referenced>> StabilityReportContainer;
      StabilityReportContainer m_stabilityReports;

      typedef agx::ref_ptr<agx::Referenced> MergedBodyContainerType;
      typedef agx::Vector<MergedBodyContainerType> MergedBodiesContainer;
      MergedBodiesContainer m_mergedBodies;

      typedef agx::ref_ptr<agx::Referenced> MergeSplitHandlerPointer;
      MergeSplitHandlerPointer m_mergeSplitHandler;

      using LinkedStructureContainerType = agx::ref_ptr<agx::Referenced>;
      using LinkedStructureContainer = agx::Vector<LinkedStructureContainerType>;
      LinkedStructureContainer m_linkedStructures;

      using TerrainInstanceContainerType = agx::ref_ptr<agx::Referenced>;
      using TerrainInstanceContainer = agx::Vector<TerrainInstanceContainerType>;
      TerrainInstanceContainer m_terrains;

      bool m_disableGeometryAutoAdd;
      agxSDK::SimulationParameterRefVector m_simulationParameters;


      agx::RangeJobVector    m_postSolveConstraintForceJobs;
      agx::ParallelTaskRef   m_postSolveConstraintForceTask;

      agx::ref_ptr<agx::Referenced> m_energyManager;
  };

  /* Implementation */
  AGX_FORCE_INLINE agx::Real Simulation::getTimeStamp() const
  {
    return m_clock->getTime();
  }

  AGX_FORCE_INLINE agxCollide::Space* Simulation::getSpace()
  {
    return m_space.get();
  }

  AGX_FORCE_INLINE const agxCollide::Space* Simulation::getSpace() const
  {
    return m_space.get();
  }

  AGX_FORCE_INLINE const agx::DynamicsSystem* Simulation::getDynamicsSystem() const
  {
    return m_system.get();
  }

  AGX_FORCE_INLINE agx::DynamicsSystem* Simulation::getDynamicsSystem()
  {
    return m_system.get();
  }

  AGX_FORCE_INLINE agx::Real Simulation::getTimeStep() const
  {
    if (m_system) return m_system->getTimeGovernor()->getTimeStep();
    else return 0;
  }

  AGX_FORCE_INLINE agx::Clock* Simulation::getClock()
  {
    return m_clock;
  }

  AGX_FORCE_INLINE const agx::Clock* Simulation::getClock() const
  {
    return m_clock;
  }

  AGX_FORCE_INLINE bool Simulation::allowCreateParticles() const
  {
    return m_allowCreateParticles;
  }

  AGX_FORCE_INLINE const agx::SolveModelRefVector& Simulation::getSolveModels() const
  {
    return m_solveModels;
  }

  template< typename T >
  T* Simulation::getConstraint( const agx::Uuid& uuid )
  {
    return dynamic_cast< T* >( this->getConstraint( uuid ) );
  }

  template< typename T >
  const T* Simulation::getConstraint( const agx::Uuid& uuid ) const
  {
    return dynamic_cast< const T* >( this->getConstraint( uuid ) );
  }

  template< typename T >
  T* Simulation::getConstraint( const agx::Name& name )
  {
    return dynamic_cast< T* >( this->getConstraint( name ) );
  }

  template< typename T >
  const T* Simulation::getConstraint( const agx::Name& name ) const
  {
    return dynamic_cast< const T* >( this->getConstraint( name ) );
  }
} // namespace agxSDK

#ifdef _MSC_VER
# pragma warning(pop) // C++ exception specification ignored except to indicate a function is not __declspec(nothrow). Restored.
#endif
