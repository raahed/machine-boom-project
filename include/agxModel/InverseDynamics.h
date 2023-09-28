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

#include <agxModel/export.h>

#include <agx/ref_ptr.h>
#include <agx/Real.h>
#include <agx/AffineMatrix4x4.h>
#include <agx/Vector.h>
#include <agx/HashSet.h>
#include <agx/RigidBody.h>

namespace agx {
  using RigidBodyConstRef     = agx::ref_ptr<const agx::RigidBody>;
  using RigidBodyPtrHashSet   = agx::HashSet<agx::RigidBody* >;

  template<typename T1, typename T2>
  class SetVector;

  class Constraint;
  using ConstraintRef          = agx::ref_ptr<agx::Constraint>;
  using ConstraintConstRef     = agx::ref_ptr<const agx::Constraint>;
  using ConstraintRefVector    = agx::Vector<ConstraintRef>;
  using ConstraintConstPtrSet  = agx::HashSet<const agx::Constraint*>;
  using ConstraintRefSetVector = agx::SetVector<agx::ConstraintRef, agx::HashFn<agx::ConstraintRef> >;

  class LockJoint;
  using LockJointRef          = agx::ref_ptr<agx::LockJoint>;

  using AffineMatrix4x4Vector = agx::VectorPOD<agx::AffineMatrix4x4T<agx::Real> >;
  using RealVector            = agx::VectorPOD<agx::Real>;
}

namespace agxSDK {
  class Simulation;
  using SimulationRef    = agx::ref_ptr<agxSDK::Simulation>;

  class Assembly;
  using AssemblyRef      = agx::ref_ptr<agxSDK::Assembly>;
  using AssemblyConstRef = agx::ref_ptr<const agxSDK::Assembly>;

  class Collection;
  using CollectionRef    = agx::ref_ptr<agxSDK::Collection>;
}

namespace agxPowerLine {
  class Unit;
  using UnitRef            = agx::ref_ptr<Unit>;
  using UnitRefVector      = agx::Vector<UnitRef>;

  class PowerLine;
  using PowerLineRef       = agx::ref_ptr<PowerLine>;
  using PowerLineRefVector = agx::Vector<PowerLineRef>;
}

namespace agxDriveTrain {
  class VelocityConstraint;
  using VelocityConstraintRef       = agx::ref_ptr<VelocityConstraint>;
  using VelocityConstraintRefVector = agx::Vector<VelocityConstraintRef>;
}


namespace agxModel
{
  /**
  Struct holding information about a joint.
  Instances of this struct are used to store
  resulting forces from inverse dynamics computations.

  The jointIndex value in the struct corresponds to the constraint index in the input data.
  If all input constraints are acting on input bodies, then indices will match 1-to-1.
  If some constraint is considered invalid, the indices will be different compared to
  the structure given to the InverseDynamics constructor.

  To get the matching constraint for the data, the safest way to do so is:
  \code
    inverseDynamics->getSourceData()->getConstraints()[ jointIndex ]
  \endcode
  */
  typedef struct IDJointValue {
    agx::UInt32 jointIndex;
    agx::UInt32 angleIndex;
    agx::Real   value;
  } IDJointValue;


  /**
  Struct holding information about an agxPowerLine::Unit.

  Output from inverse dynamics and the solve calls involve
  generic "value" and "gradient" for specified units.

  The same struct is reused for gravity compensation results,
  but then "value" will contain force/torque needed to hold 0 velocity
  and gradient will be set to 0.
  */
  typedef struct IDUnitValue {
    agxPowerLine::Unit* unit;
    agx::UInt32         dimensionIndex;
    agx::Real           value;
    agx::Real           gradient;
  } IDUnitValue;


  /**
  A read-only view into data computed by InverseDynamics.

  References to the data should not be stored over time.
  If persistance is needed, make copies.
  */
  template<typename T>
  class IDSpan
  {
    public:
      /// Fetch value at index. Input index must be less than size.
      const T& at( size_t index ) const;

      const T& operator[]( size_t index ) const;

      /// Number of elements accessible via this view into the data
      size_t size() const;

    private:
      friend class InverseDynamics;

      IDSpan(const T* ptr, size_t size);

      const T* m_ptr;
      size_t   m_size;
  };

  using IDJointValueView = IDSpan<IDJointValue>;
  using IDUnitValueView  = IDSpan<IDUnitValue>;


  /**
  Class for computing forces/torques which can be used to reach a desired configuration.
  It can also be used to compensate for gravity by computing forces/torques needed to keep
  current pose and have zero velocity.

  This class is intended for use with constrained systems. There is also partial powerline support:
  - Gravity compensation can provide which force/torque that should be applied at a PowerLine Unit.
  - Trajectory computations via solve only provide value/gradient for specified powerline units
    and these units are not used to drive the motion.

  \note Controlling a powerline unit to control a constraint to reach a certain
        pose is likely to have worse precision compared to working on the contraints
        unless all the powerline connections are stiff.

        The InverseDynamics class is agnostic to what the powerline units actually
        represents and how the value and gradient should be used depends on the
        actual unit.

  \warning The API for this class is not finalized and breaking changes can occur.
  */
  class AGXMODEL_EXPORT InverseDynamics: public agx::Referenced
  {
    public:
      /**
      Enumeration with status values returned from InverseDynamics.
      */
      enum InverseDynamicsStatus {
        ID_SUCCESS,               ///< Return value for successful operation.
        ID_INVALID_TIME,          ///< Invalid timestamp provided as argument.
        ID_BAD_ARGUMENTS,         ///< Problem with timestamp and/or transform input values.
        ID_BODY_NOT_FOUND,        ///< Provided body argument not found.
        ID_VALID_POSE_NOT_FOUND,  ///< A valid pose could not be found with provided thresholds.
        ID_COLLISION_OCCURRED     ///< A body part of the jointed structure collided with something.
      };

      enum SyncLevel {
        INTERNAL = 0,                                           ///< Update Motion body transforms from Dynamic
        MOTION = 1 << 0,                                        ///< Update Motion body transforms with the source
        DYNAMIC = 1 << 1,                                       ///< Update Dynamic body transforms with the source
        FULL = MOTION + DYNAMIC                                 ///< Sync everything given the source
      };

      /**
      Struct for defining a move operation during one call to
      one of the solve functions of InverseDynamics.
      */
      struct MoveOperation
      {
        MoveOperation(agx::RigidBody* body,
          const agx::AffineMatrix4x4& transform,
          agx::Real translationalCompliance,
          agx::Real rotationalCompliance,
          agx::Real translationalThreshold,
          agx::Real rotationThreshold,
          bool useRotation = true         // If set use entire target transform, if false just move to the position

          )
          : m_body(body), m_transform(transform), m_translationalCompliance(translationalCompliance),
          m_rotationalCompliance(rotationalCompliance), m_translationalThreshold(translationalThreshold),
          m_rotationThreshold(rotationThreshold), m_useRotation(useRotation)
        {}
        MoveOperation() :
          m_body(nullptr), m_transform(agx::AffineMatrix4x4()), m_translationalCompliance(agx::Real(1)),
          m_rotationalCompliance(agx::Real(1)), m_translationalThreshold(agx::Real(1)),
          m_rotationThreshold(agx::Real(1)), m_useRotation(true)
        {}
        agx::RigidBodyObserver m_body;
        agx::AffineMatrix4x4 m_transform;
        agx::Real m_translationalCompliance;
        agx::Real m_rotationalCompliance;
        agx::Real m_translationalThreshold;
        agx::Real m_rotationThreshold;
        bool m_useRotation;
      };

      typedef agx::Vector<InverseDynamics::MoveOperation> MoveOperationVector;

      /**
      Create an InverseDynamics instance holding structures matching the input bodies and
      constraints. Some settings will also be copied from the simulation to be
      able to have a matching configuration (timestep, collision groups from space).

      The source data will not be changed by InverseDynamics when computations are performed.

      \param sim The simulation from which settings should be read.
      \param bodies A container with bodies
      \param constraints A container with constraints for which InverseDynamics will find force/torque values

      The following is required for later computations to be valid:
      - The constraints must act on RigidBodies present in the bodies container

      To detect possible problems, check output from the logger or compare
      the sizes of the containers for bodies and constraints in getSourceData.
      */
      InverseDynamics(const agxSDK::Simulation* sim,
                      const agx::RigidBodyRefVector& bodies,
                      const agx::ConstraintRefVector& constraints);

      /**
      This constructor is similar to the three argument constructor but also
      includes an additional vector with constraints.

      The first vector with constraints are "controlled" by InverseDynamics and force/torque
      values are to be found for these constraints.

      The second vector with constraints are included in the internal calculations but the
      controllers for these constraints are left as is.

      This can for example be used for a robot where the end effector is attached to a tool that
      can perform grasping and the inverse dynamics calculations should not change the fingers.

      \param sim The simulation from which settings should be read.
      \param bodies A container with bodies
      \param constraints A container with constraints for which InverseDynamics will find force/torque values
      \param otherConstraints A container with constraints that InverseDynamics will include in the
                              computations, but not adjust the values for the controllers.
      */
      InverseDynamics(const agxSDK::Simulation* sim,
                      const agx::RigidBodyRefVector& bodies,
                      const agx::ConstraintRefVector& constraints,
                      const agx::ConstraintRefVector& otherConstraints);

      /**
      Create an InverseDynamics instance holding structures matching the input bodies,
      constraints and powerlines. Some settings will also be copied from the simulation to be
      able to have a matching configuration (timestep, collision groups from space).

      The source data will not be changed by InverseDynamics when computations are performed.

      \note solve and solveLinear do include powerlines in the simulations, but the constraints
            are currently used for driving the motion and not the actual powerlines.
            Accessing the force- and torque data for the constraints can be done with getJointForces.
            Data for specified powerline units can be accessed via getPowerLineValue to see value and gradient.

      \param sim The simulation from which settings should be read.
      \param bodies A container with bodies.
      \param constraints A container with constraints.
      \param powerlines A container with powerlines.
      \param units A container with powerline units.

      The following is required for later computations to be valid:
      - The constraints must act on RigidBodies present in the bodies container
      - PowerLine Actuators must act on constraints available in the constraint container
      - PowerLine Units must belong to powerlines present in the powerline container
      */
      InverseDynamics(const agxSDK::Simulation* sim,
                      const agx::RigidBodyRefVector& bodies,
                      const agx::ConstraintRefVector& constraints,
                      const agxPowerLine::PowerLineRefVector& powerlines,
                      const agxPowerLine::UnitRefVector& units);

      /**
      Create an InverseDynamics instance holding structures matching the input bodies,
      constraints and powerlines. Some settings will also be copied from the simulation to be
      able to have a matching configuration (timestep, collision groups from space).

      The source data will not be changed by InverseDynamics when computations are performed.

      \note solve and solveLinear do include powerlines in the simulations, but the constraints
            are currently used for driving the motion and not the actual powerlines.
            Accessing the force- and torque data for the constraints can be done with getJointForces.
            Data for specified powerline units can be accessed via getPowerLineValue to see value and gradient.

      \param sim The simulation from which settings should be read.
      \param bodies A container with bodies.
      \param constraints A container with constraints.
      \param otherConstraints A container with constraints that InverseDynamics will include in the
                              computations, but not adjust the values for the controllers.
      \param powerlines A container with powerlines.
      \param units A container with powerline units.

      The following is required for later computations to be valid:
      - The constraints must act on RigidBodies present in the bodies container
      - PowerLine Actuators must act on constraints available in the constraint container
      - PowerLine Units must belong to powerlines present in the powerline container
      */
      InverseDynamics(const agxSDK::Simulation* sim,
                      const agx::RigidBodyRefVector& bodies,
                      const agx::ConstraintRefVector& constraints,
                      const agx::ConstraintRefVector& otherConstraints,
                      const agxPowerLine::PowerLineRefVector& powerlines,
                      const agxPowerLine::UnitRefVector& units);

      /**
      Update the motion control to specified value for the internal bodies that are
      mirroring the specified body.

      \param body A rigidbody that should have be part of the input bodies to InverseDynamics.
      \param mc The motion control value to set.
      \return True if mirroring bodies were found and motioncontrol could be set
      */
      bool setMotionControl(const agx::RigidBody* body, agx::RigidBody::MotionControl mc);

      /**
      This method will synchronize transforms and velocities so that solve calls
      operate on data that is in a known state.

      Using this method between calls to solve is strongly advised to get expected
      results.

      See InverseDynamics::SyncLevel for the different options.

      \note This method should not be called to do a InverseDynamics::FULL sync while the source
            simulation is being stepped to avoid threadding issues.

            Only body and powerline data will be synced.
            Other changes to the source simulation such as for example:
            - changed compliance/damping for constraints
            - changes to materials/contact materials
            - changes to collision groups
            will not be synced.

      \param level define which InverseDynamics::SyncLevel to sync
      \return True if sync could be performed successfully.
      */

      bool sync(agxModel::InverseDynamics::SyncLevel level = agxModel::InverseDynamics::FULL);

      /**
      Compute joint forces needed to compensate for gravity.

      This computation will use the internal dynamics bodies and compute
      forces/torques needed for the robot to behave as if there is zero gravity.

      If PowerLine(s) are present and the following conditions are fulfilled, then
      matching constraints will not be used to keep the speed at 0, but instead the PowerLine:
      - PowerLine P is present among the powerlines given to the InverseDynamics constructor
      - PowerLine P has a Unit U given to the InverseDynamics constructor
      - PowerLine P has an Actuator A
      - Actuator A is connected to a Constraint C in the jointed structure.
      - Constraint C will then be controlled via PowerLine instead of SecondaryConstraints
      - Output will be the torque/force needed at Unit U to maintain zero velocity.

      If needed, use sync to have the inverse dynamics system and source simulation in matching states.

      On success, getNumFrames() will have 1 frame worth of data and the results are available
      via getJointForces(0) and getPowerLineValues(0).

      \note The source- and internal simulation must match if the computed force/torque
            should be valid and able to hold the robot at zero velocity.

            If incompatible changes are made to the robot after the InverseDynamics instance
            has been created, then the computed results are likely not usable. See the \sa sync
            method for a list of items that are not synced.

      \return The return value from gravityCompensation can currently be ignored,
              but will be revised in a future release.
      */
      InverseDynamicsStatus gravityCompensation();

      /**
      Try to find a pose for the structure so the specified body has the requested transform
      after the specified delta time has elapsed.

      This solve method will monotonically change the joint angles from the ones in the initial
      pose until they reach the values in the end pose. The computed trajectory aims for
      landing on the target transform with zero velocity.

      The computations are performed in two steps, first a kinematic configuration is found
      and then forces/torques are computed that can be applied to reach specified transform.

      The state of the internal bodies that are used in the computations should be considered
      unknown after solve.

      \param body A body from one of the input source bodies.
      \param transform The desired transform for the body.
      \param time Delta time from now, amount of simulated time it should take for the body to reach
             the desired transform.
      \param positionThreshold Value used to determine if final kinematic configuration in the motion
             trajectory is close enough to requested transform. Value in meters.
      \param rotationThreshold Similar to positionThreshold, but for rotation. Value in radians.
      \param stopAtCollision set true will exit the inverse dynamics solve at a collision, default false, no effect if onlyMotionSimulation is true.
      \param onlyMotionSimulation set true will not generate torques as a result, default false

      \return A status value indicating the result of the computation.
      */
      InverseDynamicsStatus solve(const agx::RigidBody* body, agx::AffineMatrix4x4 transform,
          agx::Real time, agx::Real positionThreshold, agx::Real rotationThreshold, const bool stopAtCollision = false, const bool onlyMotionSimulation = false);

      /**
      Similar to solve, but with a series of transforms and delta timestamps.
      The deltas are relative when the solve method is called, not the previous timestamp.
      So if transform X1 is desired after 1s and transform X2 after 1.5s, then the vectors should
      contain transforms: [X1, X2] and timestamps [1, 1.5].

      This method can be seen as using the solve-method that takes one delta time and one transform
      multiple times after each other.

      \param body A body from one of the input source bodies.
      \param transforms A vector with desired transforms for the body.
      \param timestamps A vector with delta times from now when the matching desired transforms should be reached.
      \param positionThreshold Value used to determine if final kinematic configuration in the motion
             trajectory is close enough to requested transform. Value in meters.
      \param rotationThreshold Similar to positionThreshold, but for rotation. Value in radians.
      \param stopAtCollision set true will exit the inverse dynamics solve at a collision, default false, no effect if onlyMotionSimulation is true.
      \param onlyMotionSimulation set true will not generate torques as a result, default false

      \return A status value indicating the result of the computation.
      */
      InverseDynamicsStatus solve(const agx::RigidBody* body, const agx::AffineMatrix4x4Vector& transforms,
          const agx::RealVector& timestamps, agx::Real positionThreshold, agx::Real rotationThreshold, const bool stopAtCollision = false, const bool onlyMotionSimulation = false);

      /**
      Try to find a pose for the structure so the specified body has the requested transform
      after the specified delta time has elapsed.

      This solve method will try to move the specified body along a linear path, where feasible,
      between the initial pose and the end pose. The computed trajectory aims for
      landing on the target transform with zero velocity.

      \param body A body from one of the input source bodies.
      \param transform The desired transform for the body.
      \param time Delta time from now, amount of simulated time it should take for the body to reach
             the desired transform.
      \param positionThreshold Value used to determine if final kinematic configuration in the motion
             trajectory is close enough to requested transform. Value in meters.
      \param rotationThreshold Similar to positionThreshold, but for rotation. Value in radians.
      \param stopAtCollision set true will stop the inverse dynamics solve at a collision, default false, no effect if onlyMotionSimulation is true.
      \param onlyMotionSimulation set true will not generate torques as a result, default false

      \return A status value indicating the result of the computation.
      */
      InverseDynamicsStatus solveLinear(const agx::RigidBody* body,
                                        agx::AffineMatrix4x4 transform,
                                        agx::Real time, agx::Real positionThreshold, agx::Real rotationThreshold,
                                        const bool stopAtCollision = false,
                                        const bool onlyMotionSimulation = false);
      /**
      This solve method will try to move the specified bodies along individual linear paths, where feasible,
      between the initial pose and the end pose.
      Another difference with the solveLinear method is that each move can have individual
      thresholds and compliance of the trajectory constraints.
      Initialize a solve for N bodies, with desired end transforms.
      \param moves             - All wanted move operations
      \param time              - The time in seconds to solve the system

      */
      InverseDynamics::InverseDynamicsStatus solveLinearN(const InverseDynamics::MoveOperationVector& moves,
                                                          agx::Real time,
                                                          const bool stopAtCollision = false,
                                                          const bool onlyMotionSimulation = false);

      /**
      Compute the translational difference between body and the matching
      internal clone that is used in dynamics computations.

      \note This is an optional computation of an error that can be used after gravity compensation.

            If this method is used directly after solve or solveLinear, the return value
            is likely a large difference due to one is comparing the body's current transform
            with the end transform after the solve computations.

            The other computeTranslationalDifference method that has two arguments
            is suitable for use after solve.

      \param body A rigidbody whose transform should be compared with the matching internal dynamics body.

      \return Magnitude of difference in position. Negative value if body was not found.
      */
      agx::Real computeTranslationalDifference(const agx::RigidBody* body) const;

      /**
      Compute the translational difference between input transform and the internal body
      used for dynamics computation matching the input body.

      \note This is an optional computation of an error measurement that can be used after solve or solveLinear.

      \param body A rigidbody whose transform should be compared with the input transform
      \param transform The transform to use in the difference computation.

      \return Magnitude of difference in position. Negative value if body was not found.
      */
      agx::Real computeTranslationalDifference(const agx::RigidBody* body, const agx::AffineMatrix4x4& transform) const;

      /**
      Compute the rotational difference between body and the matching
      internal clone that is used in dynamics computations.

      \note This is an optional computation of an error that can be used after gravity compensation.

            If this method is used directly after solve or solveLinear, the return value
            is likely a large difference due to one is comparing the body's current transform
            with the end transform after the solve computations.

            The other computeRotationalDifference method that has two arguments
            is suitable for use after solve.

      \param body A rigidbody whose transform should be compared with the matching internal dynamics body.

      \return Difference in rotation in radians. Range is [-PI,PI]. Negative 2*PI if body was not found.
      */
      agx::Real computeRotationalDifference(const agx::RigidBody* body) const;

      /**
      Compute the rotational difference between input transform and the internal body
      used for dynamics computation matching the input body.

      \note This is an optional computation of an error measurement that can be used after solve or solveLinear.

      \param body A rigidbody whose transform should be compared with the input transform
      \param transform The transform to use in the difference computation.

      \return Difference in rotation in radians. Range is [-PI,PI]. Negative 2*PI if body was not found.
      */
      agx::Real computeRotationalDifference(const agx::RigidBody* body, const agx::AffineMatrix4x4& transform) const;

      /**
      Return the number of frames worth of force data that is available.
      If gravityCompensation has been called sucessfully, 1 is returned.
      If a solve-method has been used, the amount of output depends on the given delta time.

      The value used when requesting data from \sa getJointForces, \sa getJointAngles and
      \sa getPowerLineValues should be less than the return value from this method.

      \return Number of frames worth of data that is available.
      */
      size_t getNumFrames() const;

      /**
      Returns the last timestamp used during trajectory computation.
      The timestamp is a delta value and starts from zero when a solve call is performed.

      On a successful solve this timestamp will be n*dt >= input_time
      where the integer n is the least amounts of steps.

      On failure, e.g. collision, this timestamps can indicate when the
      problem occurred.

      \return Current timestamp in internal trajectory simulation.
      */
      agx::Real getLastTrajectoryTimeStamp() const;

      /**
      Return the last transform the solve method reached during the trajectory computation.
      On successful solve, this transform should be similar to the solve input transform

      \return The last transform for specified body in the trajectorysimulation used by solve
      */
      agx::AffineMatrix4x4 getLastTrajectoryTransform();

      /**
      Returns the last timestamp reached during the force computations.
      The timestamp is a delta value and starts from zero when a solve call is performed.

      On a successful solve this timestamp will be n*dt >= input_time
      where the integer n is the least amounts of steps.

      On failure, e.g. collision, this timestamps can indicate when the
      problem occurred.

      \return Current timestamp in internal dynamics simulation.
      */
      agx::Real getLastDynamicsTimeStamp() const;

      /**
      Return the last transform the solve method reached during the force computations.
      On successful solve, this transform should be similar to the solve input transform.

      \return The last transform for specified body in the dynamics simulation used by solve
      */
      agx::AffineMatrix4x4 getLastDynamicsTransform();

      /**
      Fetch joint forces produced by solve call or gravityCompensation.

      The resulting structure, IDJointValue, contains [joint index, angle index, value]
      where the joint index refers to the index among the constraints in the container
      given to the constructor for this class.

      \param frameIndex Which data to access.
      \return A view with data. Valid if size() on returned instance is >0.
      */
      IDJointValueView getJointForces(size_t frameIndex);

      /**
      If solve saved angles, fetch the values for specified frame.

      \param frameIndex Which data to access.
      \return A view with data. Valid if size() on returned instance is >0.
      */
      IDJointValueView getJointAngles(size_t frameIndex);

      /**
      Fetch possible values for agxPowerLine Units.

      - Output after a solve call has value and gradient from the unit stored in the corresponding variables.
      - Output after gravityCompensation has the load stored as value and gradient set to 0.

      \param frameIndex Which data to access.
      \return A view with data. Valid if size() on returned instance is >0.
      */
      IDUnitValueView getPowerLineValues(size_t frameIndex);

      /**
      Fetch the internal Collection holding the input bodies and constraints.

      \return Collection holding references to input data.
      */
      const agxSDK::Assembly* getSourceData() const;

      /**
      Fetch the internal Collection holding the cloned bodies and constraints
      that are used for motion/trajectory computations.

      \return Collection of internal clones used for motion/trajectory computations.
      */
      const agxSDK::Assembly* getMotionData() const;

      /**
      Fetch the internal Collection holding the cloned bodies and constraints
      that are used for dynamics/force/torque computations.

      \return Collection of internal clones used for dynamics computations.
      */
      const agxSDK::Assembly* getDynamicsData() const;

      /**
      Initialize a solve for N bodies, with desired end transforms.
      \param moves             - All wanted move operations
      \param time              - The time in seconds to solve the system
      */
      InverseDynamics::InverseDynamicsStatus initializeAdaptiveSolveN(const InverseDynamics::MoveOperationVector& moves, agx::Real time);

      /**
      This update will define new target transforms for the moves already assigned
      at intitializeAdaptiveSolveN.
      Assume the moves come in the same order as the adaptive solve was initialized with
      regarding the bodies in the MoveOperation's. If not, you first need to finalize the
      adaptive solve and then initialize it again with the new order of moves.
      \param moves             - All wanted move operations
      \param time              - The time in seconds to solve the system
      */
      InverseDynamics::InverseDynamicsStatus updateAdaptiveSolveN(const InverseDynamics::MoveOperationVector& moves, agx::Real time);

      /**
      Step the adaptive solve one timestep.
      */
      InverseDynamics::InverseDynamicsStatus stepAdaptiveSolveN();

      /**
      End the adaptive solve.
      */
      InverseDynamics::InverseDynamicsStatus finalizeAdaptiveSolveN();

      /**
      Return the internal frame count.
      */
      agx::UInt getCurrentFrameIndex() const;

      /**
      Will step the motion simulation one time step 
      */
      void forceStepMotionSimulation();

      /**
      Will step the dynamics simulation one time step 
      */
      void forceStepDynamicsSimulation();

    private:
      using PowerLineRefPair  = std::pair<agxPowerLine::PowerLineRef, agxPowerLine::PowerLineRef>;
      using PowerLineMapping  = agx::Vector<PowerLineRefPair>;

      using PowerLineUnitPair    = std::pair<agxPowerLine::Unit*, agxPowerLine::Unit*>;
      using PowerLineUnitMapping = agx::Vector<PowerLineUnitPair>;

      void reserveData(const size_t numReserveFrames);

      /**
      Create a kinematic trajectory body for the motion simulation.
      \return the body created and added to the motion simulation.
      */
      agx::RigidBody* createKinematicTrajectoryBody(agxSDK::SimulationRef& motionSimulation);

      void createKinematicTrajectoryBodies(agxSDK::SimulationRef& motionSimulation, agx::RigidBodyRefVector& kinematicTrajectoryBodies, const size_t numBodies);

      InverseDynamics::InverseDynamicsStatus initializeSolveN(const InverseDynamics::MoveOperationVector& moves, agx::Real time);

      // Return true if did brake on impact
      bool solveMotionSimulation(const agx::Vec3Vector& endPositions,
                                agx::Real time,
                                const bool stopAtCollision = false,
                                const bool onlyMotionSimulation = false);

      void interpolateVelocity(const agx::Vec3Vector& endPositions, agx::Real t);

      // Return true if did brake on impact
      bool stepInternalSimulations(const agx::ConstraintRefSetVector& motionJoints,
                                const agx::ConstraintRefSetVector& dynamicsJoints,
                                const bool stopAtCollision = false,
                                const bool onlyMotionSimulation = false);

      InverseDynamics::InverseDynamicsStatus finalizeSolveN(const InverseDynamics::MoveOperationVector& moves);

      void removeKinematicObjects();

      /**
      Create container with reference to source items.
      */
      static agxSDK::Collection* createSourceContainer(const agx::RigidBodyRefVector& bodies,
                                                       const agx::ConstraintRefVector& constraints,
                                                       const agx::ConstraintRefVector* otherConstraint = nullptr,
                                                       agx::UInt64* numControlledConstraints = nullptr,
                                                       const agxPowerLine::PowerLineRefVector* powerlines = nullptr );

      /**
      */
      InverseDynamics(const agxSDK::Simulation* sim,
                      const agxSDK::Assembly* assembly,
                      const agxPowerLine::PowerLineRefVector* powerlines);

      /**
      Setup and validate powerline related inputs.
      */
      void setupInternalPowerlines(
        const agxPowerLine::PowerLineRefVector& powerlines,
        const agxPowerLine::UnitRefVector& units);

      /**
      Check if powerline have some Actuator that has a constraint in the constraints container.
      */
      bool haveConstraints(const agxPowerLine::PowerLine* powerline,
                           const agx::ConstraintRefSetVector& constraints);

      /**
      All all constraints found in powerline via Actuators that are in
      the constraint container to hashset for fast lookup.
      */
      void markConstraints(const agxPowerLine::PowerLine* powerline,
                           const agx::ConstraintRefSetVector& constraints,
                           agx::ConstraintConstPtrSet& hashset);

      /**
      Create a collection with copies of the items available in source.
      */
      agxSDK::Collection* buildCopy(const agxSDK::Assembly* source,
                                    const agxPowerLine::PowerLineRefVector& sourcePowerLines,
                                    PowerLineMapping& mapping);

      /**
      Solves using startpose, endpose and interpolation of states inbetween.
      */
      InverseDynamicsStatus doSolve(const agx::RigidBody* body,
                                    const agx::AffineMatrix4x4* transforms,
                                    const agx::Real* timestamps,
                                    size_t N,
                                    agx::Real positionThreshold,
                                    agx::Real rotationThreshold,
                                    const bool stopAtCollision = false,
                                    const bool onlyMotionSimulation = false);

      // Output data
      agx::VectorPOD<IDJointValue>     m_jointForces;
      agx::VectorPOD<IDJointValue>     m_jointAngles;
      agx::VectorPOD<IDUnitValue>      m_unitValues;

      // Source objects
      agxSDK::AssemblyConstRef         m_sourceData;
      agxPowerLine::PowerLineRefVector m_sourcePowerLines;
      agx::UInt64                      m_sourceNumControlledConstraints;

      // Dynamics objects
      agxSDK::SimulationRef            m_dynamicsSimulation;
      agxSDK::AssemblyRef              m_dynamicsData;
      agx::RigidBodyPtrHashSet         m_dynamicsJointedBodies;

      PowerLineMapping                 m_dynamicsPowerLines;
      PowerLineUnitMapping             m_dynamicsPowerLineUnits;
      agx::ConstraintConstPtrSet       m_dynamicsPowerLineControlledConstraints;

      agxDriveTrain::VelocityConstraintRefVector m_dynamicsVelocityConstraints;

      // Motion objects
      agxSDK::SimulationRef    m_motionSimulation;
      agxSDK::AssemblyRef      m_motionData;
      agx::RigidBodyPtrHashSet m_motionJointedBodies;
      PowerLineMapping         m_motionPowerLines;

      agx::RigidBodyRefVector         m_kinematicTrajectoryBodies;
      agx::Vector<agx::ConstraintRef> m_trajectoryConstraints;
      agx::Vec3Vector                 m_startPositions;
      agx::Vec3Vector                 m_endPositions;
      agx::RigidBodyRefVector         m_trajectoryBodies;

      // Misc
      agx::AffineMatrix4x4     m_solveTransform1;
      agx::AffineMatrix4x4     m_solveTransform2;

      agx::Real                m_largestCompliance;
      agx::Real                m_adaptiveTime;
      agx::UInt                m_numJointValuesPerTimestep;
      agx::UInt                m_numPowerLineValuesPerTimestep;
      agx::UInt                m_currentFrameIndex;
  };


  template<typename T>
  inline IDSpan<T>::IDSpan(const T* ptr, size_t size) :
    m_ptr(ptr), m_size(size)
  {
  }


  template<typename T>
  inline const T& IDSpan<T>::at( size_t index ) const
  {
    return *(m_ptr + index);
  }


  template<typename T>
  inline const T& IDSpan<T>::operator[]( size_t index ) const
  {
    return *(m_ptr + index);
  }

  template<typename T>
  inline size_t IDSpan<T>::size() const
  {
    return m_size;
  }

}
