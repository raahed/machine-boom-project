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

#include <agx/agxPhysics_export.h>

#include <agx/agx.h>

#include <agx/Vec3.h>
#include <agx/Component.h>
#include <agx/Constraint.h>
#include <agx/TimeGovernor.h>
#include <agx/SymmetricPair.h>
#include <agx/agx_hash_types.h>
#include <agx/MergeSplit.h>
#include <agx/AutoSleep.h>
#include <agx/Task.h>
#include <agx/Solver.h>
#include <agx/ObserverFrame.h>

namespace agxCollide
{
  class Space;
}

namespace agxSDK
{
  class Simulation;
}

namespace agx
{
  typedef agx::SetVector<ConstraintRef> ConstraintRefSetVector;
  typedef agx::SetVector<InteractionRef> InteractionRefSetVector;
  typedef agx::SetVector<InteractionRef> StrongInteractionRefSetVector;
  typedef agx::SetVector<RigidBody*> RigidBodyPtrSetVector;


  // forward declarations
  class ConstraintImplementation;
  class DynamicsSystem;
  class IndexLambdaKernel;

  /**
     The complete physical system with bodies, interactions, data
     layout, time stepper, and solver.

     The DynamicsSystem class holds all the data necessary to perform
     time integration.
  */
  class AGXPHYSICS_EXPORT DynamicsSystem : public Component
  {
    public:
      static agx::Model* ClassModel();

    public:
      /// Default constructor
      DynamicsSystem();

      /// Create a DynamicsSystem with a specific device
      DynamicsSystem( agx::Device* device );

      /**
      Add a RigidBody to the system. If the body was previously scheduled for removal this body will now be enabled.
      \return false if body already existed and therefore could not be added
      */
      bool add( agx::RigidBody* rigidBody );

      /**
      Schedule a RigidBody for removal. As the body can be part of contacts etc. we cannot really completely remove it
      It will not show up in search for bodies.
      until the beginning of next time step.

      Side effect: Will disable the body
      \return true if removal was successful
      */
      bool remove( agx::RigidBody* rigidBody );

      /**
      Add a ObserverFrame to the system. If the observer frame was previously scheduled for removal this observer frame will now be enabled.
      \return false if the observer frame already existed and therefore could not be added
      */
      bool add( agx::ObserverFrame* observerFrame );

      /**
      Schedule a ObserverFrame for removal.

      \return true if removal was successful
      */
      bool remove( agx::ObserverFrame* observerFrame );

      /**
      Add an Interaction to the system
      \return false if Interaction already existed and therefore could not be added
      */
      bool add( agx::Interaction* interaction );

      /**
      Remove the specified Interaction from the system
      \return true if removal was successful
      */
      bool remove( agx::Interaction* interaction );

      /**
      Add a constraint from the simulation.
      \return true if the constraint was added successfully, false if constraint was already added.
      */
      bool add( agx::Constraint* constraint );

      /**
      Remove a constraint from the simulation.
      \return true if constraint was removed.
      */
      bool remove( agx::Constraint* constraint );

      /**
      Find (linear search) the first Physical in the system that matches the name
      \return a pointer to the found RigidBody, 0 if none found.
      */
      RigidBody* getRigidBody( const Name& name );

      /**
      Find (linear search) the first Physical in the system that matches the name
      \return a pointer to the found RigidBody, 0 if none found.
      */
      const RigidBody* getRigidBody(const Name& name) const;

      /**
      Find (linear search) the first Constraint object in the system that matches the name
      \return a pointer to the found Constraint, nullptr if none found.
      */
      Constraint* getConstraint( const Name& name);

      /**
      Find (linear search) the first Constraint object in the system that matches the name
      \return a pointer to the found Constraint, nullptr if none found.
      */
      const Constraint* getConstraint( const Name& name ) const;

      /**
      Find (linear search) the first ObserverFrame object in the system that matches the name
      \return a pointer to the found observer frame, nullptr if none found.
      */
      ObserverFrame* getObserverFrame(const Name& name);

      /**
      Find (linear search) the first ObserverFrame object in the system that matches the name
      \return a pointer to the found observer frame, nullptr if none found.
      */
      const ObserverFrame* getObserverFrame(const Name& name) const;

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
      Replace the current iTimeGovernor with the specified one
      \param tg - pointer to an TimeGovernor
      */
      void setTimeGovernor( agx::TimeGovernor* tg );

      /**
      \return a pointer to the current TimeGovernor
      */
      agx::TimeGovernor* getTimeGovernor() ;

      /**
      \return a pointer to the current TimeGovernor
      */
      const agx::TimeGovernor* getTimeGovernor() const;

      /**
      Set the geometry contacts to resolve.
      */
      void setGeometryContacts( const agxCollide::GeometryContactPtrVector* contacts );

      /**
      Get the geometry contacts to be resolved.
      */
      const agxCollide::GeometryContactPtrVector* getGeometryContacts() const;

      /**
      \deprecated Instead we recommend using AMOR/MergeSplitHandler
      \return a pointer to the MergeSplit system
      */
      agx::MergeSplit* getMergeSplit();

      /**
      \deprecated Instead we recommend using AMOR/MergeSplitHandler
      \return a const pointer to the MergeSplit system
      */
      const agx::MergeSplit* getMergeSplit() const;

      /**
      Step the Dynamics system forward using dt calculated from the current TimeGovernor
      */
      void stepForward();

      /**
      Return the total linear momentum in the system.
      */
      agx::Vec3 getLinearMomentum() const;

      /**
      Return the total angular momentum in the system.
      */
      agx::Vec3 getAngularMomentum() const;

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
      const ObserverFrameRefSetVector& getObserverFrames() const;

      /**
      \return All observer frames in the system.
      */
      ObserverFrameRefSetVector& getObserverFrames();

      /**
      \return a set of all enabled bodies in this system
      */
      agx::RigidBodyPtrSetVector& getEnabledRigidBodies();

      /**
      \return a set of all enabled bodies in this system
      */
      const agx::RigidBodyPtrSetVector& getEnabledRigidBodies() const;

      /**
      \return a set of all sleeping bodies in this system
      */
      agx::RigidBodyPtrSetVector& getSleepingRigidBodies();

      /**
      \return a set of all sleeping bodies in this system
      */
      const agx::RigidBodyPtrSetVector& getSleepingRigidBodies() const;

      /**
      \return a set of all added interactions (ForceFields etc) in this system
      */
      const agx::InteractionRefSetVector& getInteractions() const;

      /**
      \return a set of all added strong interactions in this system
      */
      const agx::StrongInteractionRefSetVector& getStrongInteractions() const;

      /**
      \return pointer to the AutoSleep controller
      */
      agx::AutoSleep* getAutoSleep();

      /**
      \return pointer to the AutoSleep controller
      */
      const agx::AutoSleep* getAutoSleep() const;

      /**
      \return the solver.
      */
      agx::Solver* getSolver();

      /**
      \return the solver.
      */
      const agx::Solver* getSolver() const;


      /**
      Enable solver data extraction. Extracted data will be stored to disk in a
      HDF5 file with the given name. Extracted data will be collected in a group
      named after the date and time at the time of the call.

      May be called when solver data extraction is already enabled. In that case
      future data will be stored in a new group in (possibly) a new HDF5 archive.
      A counter will be added to the group name in order to avoid overwriting
      previous extractions.

      \param filename The filename of the HDF5 archive to write into.
      \param solverDataGroup Name of the group in the HDF5 archive where solver data for this run will be stored.
      */
      void enableSolverDataExtraction(const agx::String& filename, const agx::String& solverDataGroup);

      /**
      Disable solver data extraction.
      \see enableSolverDataExtraction
      */
      void disableSolverDataExtraction();

      /**
      \return true if solver data extraction is currently enabled, false otherwise.
      */
      bool getEnableSolverDataExtraction() const;


      DOXYGEN_START_INTERNAL_BLOCK()

      /// Internal: return the underlying storage of RigidBodies
      agxData::EntityStorage* getRigidBodyStorage();

      /// Internal:
      void _setGravityTask( agx::Task* task );

      /// Internal
      void updateSleepState( const agxCollide::SeparationPairVector& separations, const agxCollide::GeometryContactPtrVector& contacts );

      /// Internal
      agx::TaskGroup* getUpdateTask();

      /**
      Internal method

      Find the rigid body with the specified uuid and return its instance id
      in the journal. returns -1 on failure
      */
      agx::Index getInstanceId(const agx::Uuid& uuid) const;

      DOXYGEN_END_INTERNAL_BLOCK()


      /**
      Contact warmstarting adds an extra step to the solver where the previous timesteps contact
      data is matched to the current timesteps data so the previous forces can be used by the
      solver to find a solution faster when direct friction is used.

      If using auto-generated ContactMaterials and default settings, enabling this is not recommended
      and will just add some overhead. But if a FrictionModel with DIRECT solvetype is used, enabling
      this is recommended.

      The state data used to match contacts for warmstarting is not serialized. This means that
      storing and then restoring the simulation can give slightly different trajectories compared to
      running the simulation non-stop.

      \sa agx::FrictionModel
      */
      void setEnableContactWarmstarting( bool enable );

      /**
      \return True if contact warmstarting is enabled.
      */
      bool getEnableContactWarmstarting() const;

      DOXYGEN_START_INTERNAL_BLOCK()
      /**
      \return utility kernel to execute local functions through the task system
      */
      agx::IndexLambdaKernel* getIndexLambdaKernel() const;
      DOXYGEN_END_INTERNAL_BLOCK()

    protected:

      /// Destructor
      virtual ~DynamicsSystem();

    private:

      void updateFrames();

      void createUpdateTask();

      void setSimulation(agxSDK::Simulation* simulation);

      class SolveTask;
      friend class RigidBody;
      friend class agxSDK::Simulation;

      Task* createIntegratePositionsTask();
      Task* createUpdateFramesTask();
      Task* createSynchronizeGeometriesTask();
      Task* createMergedBodyPostIntegrateTask();

      virtual void setContext(Object* context) override;
      void updateGravity();
      void init();
      void commitRemovedBodies();
      void reset();
      void disableBody( RigidBody* body );
      void enableBody( RigidBody* body );

      void sleepBody( RigidBody* body );
      bool wakeBody( RigidBody* body );


      /// \deprecated Instead we recommend using AMOR / MergeSplitHandler
      void mergeSplitPre();
      /// \deprecated Instead we recommend using AMOR / MergeSplitHandler
      void mergeSplitPost();
      void updateInteractionForces();
      void updateSleepState();

      void startUpdateTimer(agx::Task* task);
      void stopUpdateTimerAndReport(agx::Task* task);
      void recordPlaybackStatistics();

      agxData::EntityStorageRef m_rigidBodyStorage;
      TaskRef m_updateWorldMassAndInertiaTask;
      TaskRef m_integratePositionsTask;
      TaskRef m_gravityTask;

    private:
      RigidBodyPtrSetVector m_enabledBodies;
      RigidBodyPtrSetVector m_sleepingBodies;
      RigidBodyRefSetVector m_removedBodies;

      RigidBodyRefVector m_rigidBodies;


      ConstraintRefSetVector m_constraints;
      InteractionRefSetVector m_interactions;
      StrongInteractionRefSetVector m_strongInteractions;

      ObserverFrameRefSetVector m_observerFrames;

      ref_ptr<TimeGovernor> m_timeGovernor;

      const agxCollide::GeometryContactPtrVector* m_geometryContacts;

      int m_garbageCollectionCounter;

      ref_ptr<MergeSplit> m_mergeSplit;
      ref_ptr<AutoSleep> m_autoSleep;

      TaskGroupRef m_updateTask;
      agxSDK::Simulation* m_simulation;
      agx::Timer m_updateTimer;
      agx::Task::ExecutionEvent::CallbackType m_startUpdateTimerCallback;
      agx::Task::ExecutionEvent::CallbackType m_stopUpdateTimerAndReportCallback;

      SolverRef m_solver;
      bool m_useSSE;

      LambdaKernelRef m_indexLambdaKernel;
  };

  typedef ref_ptr<DynamicsSystem> DynamicsSystemRef;


  //-------------------------------------------------------------------------
  // inline code

  inline RigidBodyPtrSetVector& DynamicsSystem::getEnabledRigidBodies()
  {
    return m_enabledBodies;
  }

  inline const RigidBodyPtrSetVector& DynamicsSystem::getEnabledRigidBodies() const
  {
    return m_enabledBodies;
  }

  inline RigidBodyPtrSetVector& DynamicsSystem::getSleepingRigidBodies()
  {
    return m_sleepingBodies;
  }
  inline const RigidBodyPtrSetVector& DynamicsSystem::getSleepingRigidBodies() const
  {
    return m_sleepingBodies;
  }

  inline void DynamicsSystem::setGeometryContacts( const agxCollide::GeometryContactPtrVector* contacts )
  {
    m_geometryContacts = contacts;
  }

  inline const agxCollide::GeometryContactPtrVector* DynamicsSystem::getGeometryContacts() const
  {
    return m_geometryContacts;
  }

  inline Solver* DynamicsSystem::getSolver() { return m_solver; }
  inline const Solver* DynamicsSystem::getSolver() const { return m_solver; }


} //namespace agx

