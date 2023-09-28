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

#ifndef AGX_CONSTRAINT_IMPLEMENTATION_H
#define AGX_CONSTRAINT_IMPLEMENTATION_H

#include <agxSDK/SimulationProxy.h>

#include <agx/FrictionModel.h>
#include <agx/RigidBody.h>
#include <agx/SVec.h>
#include <agxStream/Serializable.h>
#include <agx/agx_hash_types.h>
#include <agx/Constraint.h>
#include <agx/Jacobian.h>
#include <agx/ElementaryConstraint.h>
#include <agx/Physics/ConstraintForcesEntity.h>
#include <agx/BitState.h>
#include <agx/agx_valarray_types.h>

#ifdef _MSC_VER
# pragma warning(push)
# pragma warning( disable : 4251 ) // class X needs to have dll-interface to be used by clients of class Y
#endif

#define DEFAULT_COMPLIANCE 1.0E-8

#define CONSTRAINT_SUCCESS( success ) \
  if ( !(success) ) { \
    LOGGER_WARNING() << "Unable to create constraint. Constraint cleared." << std::endl << LOGGER_END(); \
    m_bodies.clear(); \
    m_localState.remove( VALID ); \
    return; \
  }

#define CONSTRAINT_SUCCESS_RETURN_BOOL( success ) \
  if ( !(success) ) { \
    LOGGER_WARNING() << "Unable to create constraint. Constraint cleared." << std::endl << LOGGER_END(); \
    m_bodies.clear(); \
    m_localState.remove( VALID ); \
    return false; \
  }

namespace agx
{
  // Forward declaration
  class ElementaryConstraint;
  class Constraint;
  class RegularizationParameters;
  class SparseRangeReal;
  class LSquareComplianceMatrix;
  struct NlmcpCallbackSolverData;

  typedef Vector< RigidBody* > ConstraintRigidBodyContainer;
  typedef std::pair< Real, Real > RealPair;

  class BlockStructure
  {
    public:
      typedef UInt16 BlockType;
      BlockStructure()
        : m_numBlocks( 0 ), m_numRows( 0 ), m_numDynBlocks( 0 ) {}

      AGX_FORCE_INLINE void setNumBlocks( BlockType numBlocks ) { m_numBlocks = numBlocks; }
      AGX_FORCE_INLINE void setNumRows( BlockType numRows ) { m_numRows = numRows; }

      AGX_FORCE_INLINE BlockType getNumBlocks() const { return m_numBlocks; }
      AGX_FORCE_INLINE BlockType getNumRows() const { return m_numRows; }

      AGX_FORCE_INLINE BlockType& blocks() { return m_numBlocks; }
      AGX_FORCE_INLINE BlockType& rows() { return m_numRows; }
      AGX_FORCE_INLINE BlockType  blocks() const { return m_numBlocks; }
      AGX_FORCE_INLINE BlockType  rows() const { return m_numRows; }

      AGX_FORCE_INLINE BlockType& dynBlocks() { return m_numDynBlocks; }
      AGX_FORCE_INLINE BlockType  dynBlocks() const { return m_numDynBlocks; }
      AGX_FORCE_INLINE BlockType  kinBlocks() const { return (BlockType)(m_numBlocks - m_numDynBlocks); }

      AGX_FORCE_INLINE void reset() { m_numBlocks = m_numRows = m_numDynBlocks = 0; }

    private:
      BlockType m_numBlocks;
      BlockType m_numRows;
      BlockType m_numDynBlocks;
  };

  class CALLABLE AGXPHYSICS_EXPORT ConstraintImplementation
  {
    public:
      enum Tags
      {
        NONE                  = 0,
        MANY_BODY             = (1<<0), /**< Default is binary, i.e., ~MANY_BODY. */
        DEFAULT_BINARY_TAG    = NONE,
        DEFAULT_MANY_BODY_TAG = MANY_BODY
      };

      /**
      Object that holds the state seen from the user and from the solver.
      */
      class AGXPHYSICS_EXPORT ConstrainedBodiesState
      {
        public:
          /**
          Info with number of dynamic and non-static body counters.
          */
          class AGXPHYSICS_EXPORT BodiesInfo
          {
            public:
              BodiesInfo();

              /**
              \return the dynamic bodies counter
              */
              agx::UInt& numDynamicBodies();

              /**
              \return the dynamic bodies counter
              */
              agx::UInt  numDynamicBodies() const;

              /**
              \return the non-static bodies counter
              */
              agx::UInt& numNonStaticBodies();

              /**
              \return the non-static bodies counter
              */
              agx::UInt  numNonStaticBodies() const;

            private:
              agx::UInt m_numDynamicBodies;
              agx::UInt m_numNonStaticBodies;
          };

        public:
          /**
          Default constructor.
          */
          ConstrainedBodiesState();

          /**
          Resets all data and allocates for a maximum of \p numBodies.
          \param numBodies - maximum amount of bodies (i.e., the number of bodies in the original setup)
          */
          void reset( agx::UInt numBodies );

          /**
          Increments data given a rigid body.
          \note This body should not be a root body.
          */
          agx::Bool increment( agx::RigidBody* rb );

          /**
          Finalize given state of constraint. If valid == false the counters will be set to 0.
          \return true if the constraint should be active, otherwise false
          */
          agx::Bool finalize( agx::Bool valid );

          /**
          \return body info as seen from the user
          */
          const BodiesInfo& getOriginalInfo() const;

          /**
          \return body info as seen from the solver (includes info about merged bodies)
          */
          const BodiesInfo& getSolverInfo() const;

          /**
          \return the permutation vector (includes merged bodies)
          */
          const agx::Int32Vector& getPermutation() const;

          /**
          \return the permutation vector (includes merged bodies)
          */
          agx::Int32Vector& getPermutation();

          /**
          \return the body indices for the solver (includes merged bodies)
          */
          const agx::Int32Vector& getBodyIndices() const;

          /**
          \return the body indices for the solver (includes merged bodies)
          */
          agx::Int32Vector& getBodyIndices();

          /**
          \return the bodies to use by the solver
          */
          const agx::ConstraintRigidBodyContainer& getSolverBodies() const;

          /**
          Counter that increments each time a merged body already has been
          registered and comes back. Note that it still can be several
          different merged bodies in the constraint! (E.g., 4 bodies,
          2 merged to mb1 and two to mb2. Recurrent merged bodies will
          be 2 in that case.)
          */
          agx::UInt getNumRecurrentMergedBodies() const;

        private:
          BodiesInfo m_originalInfo;
          BodiesInfo m_solverInfo;
          ConstraintRigidBodyContainer m_bodies;
          Int32Vector m_permutation;
          Int32Vector m_bodyIndices;
          UInt m_numRecurrentMergedBodies;
      };

      /**
      Interface to the bodies, either user or solver, since these
      bodies can differ when e.g., a body is merged.
      */
      #if !defined(SWIGPYTHON)
      class AGXPHYSICS_EXPORT BodyView
      {
        public:
          /**
          Internal, use ConstriantImplementation::getUserView().
          Construct given info and bodies container.
          */
          BodyView( const ConstrainedBodiesState::BodiesInfo& info,  const ConstraintRigidBodyContainer& bodies );

          /**
          Copy constructor.
          */
          BodyView( const BodyView& other );

          /**
          \return the number of dynamic bodies from this view
          */
          agx::UInt getNumDynamicBodies() const;

          /**
          \return the number of non-static bodies from this view
          */
          agx::UInt getNumNonStaticBodies() const;

          /**
          \return the number of bodies from this view
          */
          agx::UInt getNumBodies() const;

          /**
          \return rigid body at index
          */
          agx::RigidBody* getBodyAt( agx::UInt i );

          /**
          \return rigid body at index
          */
          const agx::RigidBody* getBodyAt( agx::UInt i ) const;

          /**
          \return the rigid bodies
          */
          const ConstraintRigidBodyContainer& getBodies() const;

          /**
          Hidden, unsupported.
          */
          BodyView& operator = ( const BodyView& ) = delete;

        private:
          const ConstrainedBodiesState::BodiesInfo& m_info;
          const ConstraintRigidBodyContainer& m_bodies;
      };

      class AGXPHYSICS_EXPORT SolverBodyView : public BodyView
      {
        public:
          /**
          Internal, use ConstriantImplementation::getSolverView().
          Construct given current state.
          */
          SolverBodyView( ConstrainedBodiesState& state );

          /**
          Copy constructor.
          */
          SolverBodyView( const SolverBodyView& other );

          /**
          \return current state
          */
          const ConstrainedBodiesState& getState() const;

          /**
          \return current state
          */
          ConstrainedBodiesState& getState();

          /**
          \return rigid body at index given current permutation
          */
          agx::RigidBody* getPermutedBodyAt( agx::UInt i );

          /**
          \return rigid body at index given current permutation
          */
          const agx::RigidBody* getPermutedBodyAt( agx::UInt i ) const;

        private:
          /**
          Hidden, unsupported.
          */
          inline SolverBodyView& operator = ( const SolverBodyView& ) { agxAbort(); return *this; }

        private:
          ConstrainedBodiesState& m_state;
      };
      #endif

    public:
      /**
      Default constructor. Defaults to binary (one or two body) constraint, solve type DIRECT,
      enabled and valid.
      */
      ConstraintImplementation();

      /**
      Destructor. Doesn't delete any user-added, non-ref-counted objects.
      */
      virtual ~ConstraintImplementation();

      /**
      Bodies as viewed from the user. This is the default way to look at the bodies.
      */
      #if !defined(SWIGPYTHON)
      BodyView getUserView() const;

      /**
      Bodies as viewed from the solver. It's only valid to use this interface if you're a solver.
      */
      SolverBodyView getSolverView() const;
      #endif

      // ****************************************************************************
      // Callbacks from solvers *****************************************************
      // ****************************************************************************

      /**
      Updates the valid conditions for this constraint. This constraint is not valid
      if for example; one or more bodies are removed (either from simulation or deallocated),
      one or more bodies are disabled or the number of dynamic bodies in this constraint
      is zero.
      \return m_valid, true if valid, otherwise false
      */
      virtual bool updateValid();

      /**
      Prepares this constraint. Update attachment transforms given the current transforms
      of the bodies (in most cases equivalent to calculate the Jacobian), compute the
      joint angles (if this constraint supports that) and calculates the amount of data
      needed.
      */
      virtual void prepare();

      /**
      Updated storing of Jacobian matrices. Row based, each Jacobian6DOFElement is 1x6 separated into
      spatial and rotational part. The number of used rows should be:
        number_of_non_static_bodies * number_of_active_rows
      Start the indexing at 0.
      */
      virtual size_t updateJacobian( agx::Jacobian6DOFElement* jacobians );

      /**
      Calculates and pushes the constraint values into array given current row.
      \param g - array of constraint values, must at least be of size row + numRowsThisConstraintUses
      \param row - row to start push values (g[row] = firstConstraintValue)
      \return the number of rows this constraint uses (could differ from time step to time step)
      */
      virtual int getViolation( agx::Real* g, int row );

      /**
      Gets the average size of the elements in getViolation() by calculating the
      magnitude and dividing by the number of elements.
      \return the average size of the elements in getViolation()
      */
      agx::Real calculateViolationVectorAverageSize();

      /**
      Writes regularization parameters for all active rows into \p epsilon.
      \param epsilon - array to write into (number of active rows allocated)
      \param h - current time step
      */
      virtual void getEpsilon( agx::Real* epsilon, agx::Real h );

      /**
      Writes solver parameters such as damping and holonomic. This should be deprecated.
      */
      virtual void getParameters( agx::Real* violation,
                                  agx::Real* epsilon,
                                  agx::Real* damping,
                                  bool* holonomic,
                                  agx::Real* velocity,
                                  agx::RangeReal* bounds,
                                  int8_t* indexSetState,
                                  agx::Real h,
                                  agx::Real minPerturbation = agx::Real( 1E-10 ) );

      /**
      Calculates and pushes the constraint velocities into array given current row. Constraint
      velocity is mainly used for non-holonomic constraint such as hinge- and prismatic motors.
      \param v - array of constraint velocity values, must at least be of size row + numRowsThisConstraintUses
      \param row - row to start push values (v[row] = firstConstraintValue)
      \return the number of rows this constraint uses (could differ from time step to time step)
      */
      virtual int getConstraintVelocity( agx::Real* v, int row );

      /**
      Calculates and pushes the bounds into a sparse range data structure. Every equation in
      this constraint that has a bound must during this call push its bounds at its current
      global row (row + localRow). The bounds is not a force, rather an impulse (dimension force*time).
      \param bounds - all bounds, use bounds.push_back( IndexedRangeReal( std::make_pair( row + localRow, RangeReal( myLowerBound, myUpperBound ) ) ) )
      \param dt - time step, often used to scale forces to impulses
      */
      virtual void getBounds( agx::RangeReal* bounds, agx::Real dt ) const;

      /**
      Callback with read/write access to the compliance matrix of this constraint
      in the direct solver. The lower triangle is accessible and will be used, all
      entries must be less than or equal to zero. The diagonal has already been
      written with (negative) regularization parameters of each row of this constraint.
      \param matrix - compliance matrix
      */
      CALLABLE_IGNORE
      virtual void updateComplianceMatrix( LSquareComplianceMatrix matrix ) const;

      /**
      Internal callback from the direct solver with perturbation matrix data.
      */
      void onDirectComplianceMatrix( Real* data, unsigned int ld ) const;

      /**
      Peak at the solution after the solver is done.
      \param jacobians - Jacobian rows for this constraint (index 0 is first dynamic body first row)
      \param solution - solution (impulse - divide by dt to get force)
      \param dt - time step size used to get solution
      */
      virtual void postSolveCallback( const agx::Jacobian6DOFElement* jacobians, const agx::Real* solution, agx::Real dt );

      /**
      Callback after the solver is done about the state of each equation in the constraint.
      The information is if an equation was at a upper or lower bound or free.
      */
      void postSolveIndexSetCallback( const int8_t* state );

      /**
      Before prepare, and before the system writes to the data buffers, this constraint gets a call with
      the main system it's in. During this call it's possible to add or remove bodies from the system,
      bodies this constraint are responsible of.
      \param system - the main dynamics system
      */
      virtual void preSystemCallback( DynamicsSystem* system );

      /**
      Last in the dynamics system step forward loop, this constraint gets a call with the main system. This
      call is made after the data buffers have been written back to the bodies.
      \param system - the main dynamics system
      */
      virtual void postSystemCallback( DynamicsSystem* system );

      /**
      This method will sort this constraint by global indexing of the bodies. It
      creates a permutation vector (i.e., it doesn't reorder the bodies nor the
      indices) so that one can index m_bodies and m_indices via the permutation.
      */
      void sort();

      /**
      Reset the solution vectors of the primary and secondary elementary constraints in the implementation.
      */
      void reset();

      /**
      Interface for non-linear callback from solver.
      */
      virtual void addNlCallbacks( agx::SparseRangeReal& sparseRangeReal, const NlmcpCallbackSolverData& solverData ) const;

      // ****************************************************************************

      // ****************************************************************************
      // Methods used by the solvers ************************************************
      // ****************************************************************************

      /**
      \return the number of rows this constraint uses (equivalent to m_jacobian[0].getNumRows())
      */
      AGX_FORCE_INLINE agx::UInt getJacobianRowUsage() const { return m_numRows; }

      /**
      \return the block structure for this constraint (number of rows, number of Jacobian blocks etc.)
      */
      AGX_FORCE_INLINE const BlockStructure& getBlockStructure() const { return m_blockStructure; }

      /**
      Sets the global start index for this constraint in the global, non-permuted matrix
      \param blockRowIndex - global start row for this constraint in the global, non-permuted matrix
      */
      AGX_FORCE_INLINE void setBlockRowIndex( agx::UInt blockRowIndex ) { m_blockRowIndex = blockRowIndex; }

      /**
      \return the global start row for this constraint in the global, non-permuted matrix
      */
      AGX_FORCE_INLINE agx::UInt getBlockRowIndex() const { return m_blockRowIndex; }

      /**
      \return true if this constraint will be solved iteratively by a hybrid solver - otherwise false
      \note Should be removed, still here for old hybrid solvers.
      */
      AGX_FORCE_INLINE bool getSolveIterative() const { return m_solveType == Constraint::ITERATIVE; }

      /**
      \return true if this constraint is solved both with the direct solve and with the iterative solver
      */
      AGX_FORCE_INLINE bool getSolveBoth() const { return m_solveType == Constraint::DIRECT_AND_ITERATIVE; }

      /**
      Set solve type, either DIRECT, ITERATIVE or DIRECT_AND_ITERATIVE.
      */
      AGX_FORCE_INLINE void setSolveType( Constraint::SolveType type ) { m_solveType = type; }

      /**
      \return the solve type for this constraint
      */
      AGX_FORCE_INLINE Constraint::SolveType getSolveType() const { return m_solveType; }

      /**
      \return true if one or more equations in this constraint is impacting
      */
      AGX_FORCE_INLINE bool isImpacting() const { return m_localState.Is( IMPACTING ); }

      /**
      Increment the number of constraints connection to \p rb.
      \param rb - rigid body to increment the constraint count
      */
      AGX_FORCE_INLINE void incrementConstraintCount( RigidBody* rb ) const { agxAssert( rb ); if (rb) rb->incrementNumConstraints(); }

      /**
      \return the current tag of this constraint
      */
      AGX_FORCE_INLINE int getTag() const { return m_tag; }

#ifndef SWIG
      void* customData;  /**< Custom data for the solvers to store data into */
#endif

      // ****************************************************************************

      // ****************************************************************************
      // Callbacks from the simulation during add and remove of this constraint *****
      // ****************************************************************************

      /**
      Callback when this constraint is added to Simulation.
      */
      virtual void addNotification() {}

      /**
      Callback when this constraint is removed from Simulation.
      */
      virtual void removeNotification() {}

      // ****************************************************************************

      // ****************************************************************************
      // Methods used by non-implementation constraints but not the 'end user' ******
      // ****************************************************************************

      /**
      \return a vector of the elementary constraints
      */
      AGX_FORCE_INLINE agx::ElementaryConstraint::RefContainer& getElementaryConstraints() { return m_ec; }
      AGX_FORCE_INLINE const agx::ElementaryConstraint::RefContainer& getElementaryConstraints() const { return m_ec; }

      /**
      \return a vector of secondary constraints
      */
      AGX_FORCE_INLINE agx::ElementaryConstraint::RefContainer& getSecondaryConstraints() { return m_sc; }
      AGX_FORCE_INLINE const agx::ElementaryConstraint::RefContainer& getSecondaryConstraints() const { return m_sc; }

      /**
      \return a vector of the sub-constraints
      */
      AGX_FORCE_INLINE ConstraintImplPtrVector& getSubConstraints() { return m_subConstraints; }
      AGX_FORCE_INLINE const ConstraintImplPtrVector& getSubConstraints() const { return m_subConstraints; }

      // ****************************************************************************
      // Methods used by 'the user' *************************************************
      // ****************************************************************************

      /**
      The general version of rebind takes the first attachment frame and moves the other attachment
      frame so they have the exact same world transform. This means that the first rigid body is the
      reference (other is another rigid body or world). E.g., Prismatic and Hinge axis will be preserved
      seen from the reference body.
      \return true if successful - otherwise false
      */
      virtual bool rebind() { return false; }

      /**
      \return true if this constraint is enabled, otherwise false
      \note The constraint can't disable itself, i.e., enable == false only if the user has disabled it
      */
      virtual AGX_FORCE_INLINE bool getEnable() const { return m_enable; }

      /**
      Sets the enable flag. If this flags sets to false the solver will ignore this constraint.
      \param enable - set to true if this constraint should be enabled, false to disable it
      */
      virtual void setEnable(bool enable);

      /**
      \return true if this constraint is enabled and valid, otherwise false
      */
      AGX_FORCE_INLINE bool getActive() const { return m_enable && m_localState.Is( VALID ); }

      /**
      \return true if this constraint is valid (no disabled bodies, removed bodies, has dynamic bodies etc.), otherwise false
      */
      AGX_FORCE_INLINE bool isValid() const { return m_localState.Is( VALID ); }

      // ****************************************************************************

      // ****************************************************************************
      // Methods for user API, these should not be used internally ******************
      // ****************************************************************************

      /**
      Enable (or disable) computation of the forces applied to the dynamic bodies in this constraint.
      Matrix-vector operation to compute the forces after solve.
      \sa getLastForce
      \param enable - true to enable, false to disable
      */
      virtual void setEnableComputeForces( agx::Bool enable );

      /**
      \return true if this constraint has been enabled to compute the forces applied to its bodies - otherwise false
      */
      virtual agx::Bool getEnableComputeForces() const;

      /**
      If 'compute forces' is enabled, returns the last force and torque applied by this constraint on
      the body with \p bodyIndex. The force is given in world coordinates and is the one applied
      at the anchor position of this constraint.

      The result includes force and torque from this constraint including all enabled
      controllers (e.g., motors, locks, ranges).
      \sa setEnableComputeForces
      \param bodyIndex      - index of body, if number of bodies = 1 and bodyIndex = 1, the force and
                              torque applied to "the world body" is returned
      \param retForce       - the force applied by this constraint on body at \p bodyIndex last solve
      \param retTorque      - the torque applied by this constraint on body at \p bodyIndex last solve
      \param giveForceAtCm  - this parameter affects the resulting torque. The default behavior (giveForceAtCm = false)
                              calculates the force applied by this constraint at the anchor position.
                              Letting giveForceAtCm = true, the force will be applied at center of mass,
                              affecting the torque as \f$T_{new} = T - r \times F\f$ where r is the vector
                              from the anchor point to the center of mass of the body.
      \return true if resulting force and torque was written to \p retForce and \p retTorque - otherwise false
      */
      virtual agx::Bool getLastForce( agx::UInt bodyIndex, agx::Vec3& retForce, agx::Vec3& retTorque, agx::Bool giveForceAtCm = false ) const;

      /**
      If 'compute forces' is enabled, returns the last force and torque applied by this constraint on
      the body \p rb. The force is given in world coordinates and is the one applied at the anchor
      position of this constraint.

      The result includes force and torque from this constraint including all enabled
      controllers (e.g., motors, locks, ranges).
      \sa setEnableComputeForces
      \param rb             - body in this constraint. If this constraint is attached in world, nullptr can be used.
      \param retForce       - the force applied by this constraint on body \p rb at the last solve
      \param retTorque      - the torque applied by this constraint on body \p rb at the last solve
      \param giveForceAtCm  - this parameter affects the resulting torque. The default behavior (giveForceAtCm = false)
                              calculates the force applied by this constraint at the anchor position.
                              Letting giveForceAtCm = true, the force will be applied at center of mass,
                              affecting the torque as \f$T_{new} = T - r \times F\f$ where r is the vector
                              from the anchor point to the center of mass of the body.
      \return true if resulting force and torque was written to \p retForce and \p retTorque - otherwise false
      */
      virtual agx::Bool getLastForce( const agx::RigidBody* rb, agx::Vec3& retForce, agx::Vec3& retTorque, agx::Bool giveForceAtCm = false ) const;

      /**
      If 'compute forces' is enabled, returns the last force and torque applied by this constraint on
      the body with \p bodyIndex. The force is given in the frame of the constraint.

      The result includes force and torque from this constraint including all enabled
      controllers (e.g., motors, locks, ranges).
      \sa setEnableComputeForces
      \param bodyIndex      - index of body, if number of bodies = 1 and bodyIndex = 1, the force and
                              torque applied to "the world body" is returned
      \param retForce       - the force applied by this constraint on body at \p bodyIndex last solve
      \param retTorque      - the torque applied by this constraint on body at \p bodyIndex last solve
      \param giveForceAtCm  - this parameter affects the resulting torque. The default behavior (giveForceAtCm = false)
                              calculates the force applied by this constraint at the anchor position.
                              Letting giveForceAtCm = true, the force will be applied at center of mass,
                              affecting the torque as \f$T_{new} = T - r \times F\f$ where r is the vector
                              from the anchor point to the center of mass of the body.
      \return true if resulting force and torque was written to \p retForce and \p retTorque - otherwise false
      */
      virtual agx::Bool getLastLocalForce( agx::UInt bodyIndex, agx::Vec3& retForce, agx::Vec3& retTorque, agx::Bool giveForceAtCm = false ) const;

      /**
      If 'compute forces' is enabled, returns the last force and torque applied by this constraint on
      the body \p rb. The force is given in the frame of the constraint.

      The result includes force and torque from this constraint including all enabled
      controllers (e.g., motors, locks, ranges).
      \sa setEnableComputeForces
      \param rb             - body in this constraint. If this constraint is attached in world, nullptr can be used.
      \param retForce       - the force applied by this constraint on body \p rb at the last solve
      \param retTorque      - the torque applied by this constraint on body \p rb at the last solve
      \param giveForceAtCm  - this parameter affects the resulting torque. The default behavior (giveForceAtCm = false)
                              calculates the force applied by this constraint at the anchor position.
                              Letting giveForceAtCm = true, the force will be applied at center of mass,
                              affecting the torque as \f$T_{new} = T - r \times F\f$ where r is the vector
                              from the anchor point to the center of mass of the body.
      \return true if resulting force and torque was written to \p retForce and \p retTorque - otherwise false
      */
      virtual agx::Bool getLastLocalForce( const agx::RigidBody* rb, agx::Vec3& retForce, agx::Vec3& retTorque, agx::Bool giveForceAtCm = false ) const;

      /**
       * Control whether or not linearization should be performed on the
       * ElementaryConstraints that require it, such as QuatLock.
       *
       * \see agx::Constraint::setEnableLinearization
       */
      void setEnableLinearization(bool enable);

      /**
       * \return true if linearization is enabled for this constraint.
       */
      bool getEnableLinearization() const;

      /**
      \return the constraint attachment given a rigid body (returns 0 if invalid)
      */
      virtual agx::Attachment* userAPIgetAttachment( const agx::RigidBody* /*rb*/ ) const { return nullptr; }

      /**
      \return the constraint attachment given index
      */
      virtual agx::Attachment* userAPIgetAttachment( agx::UInt /*index*/ ) const { return nullptr; }

      /**
      \return the constraint attachment pair
      */
      virtual agx::AttachmentPair* getAttachmentPair() const { return nullptr; }

      /**
      Add elementary constraint (like Spherical, Dot1, Dot2 etc) given name.
      General for elementary constraints is that their input is based on
      body relative data (far from general).
      \return true if this constraint supports the elementary constraint and the elementary constraint will be used - otherwise false
      */
      virtual bool addElementaryConstraint( const agx::String& name, agx::ElementaryConstraint* elementaryConstraint );

      /**
      Add secondary constraint (like motor, range and/or lock etc) given name.
      General for secondary constraints is that their input is based on the
      current constraint angle.
      \return true if this constraint supports the secondary constraint and the secondary constraint will be used - otherwise false
      */
      virtual bool addSecondaryConstraint( const agx::String& name, agx::ElementaryConstraint* secondaryConstraint );

      /**
      Remove elementary constraint.
      \param elementaryConstraint - elementary constraint to remove
      \return true if removed
      */
      virtual bool removeElementaryConstraint( agx::ElementaryConstraint* elementaryConstraint );

      /**
      Remove secondary constraint.
      \param secondaryConstraint - secondary constraint to remove
      \return true if removed
      */
      virtual bool removeSecondaryConstraint( agx::ElementaryConstraint* secondaryConstraint );

      /**
      Remove elementary constraint.
      \param name - name of the elementary constraint to remove
      \return true if removed
      */
      virtual bool removeElementaryConstraint( const agx::String& name );

      /**
      Remove secondary constraint.
      \param name - name of the secondary constraint to remove
      \return true if removed
      */
      virtual bool removeSecondaryConstraint( const agx::String& name );

      /**
      \param index - index of the elementary constraint
      \return elementary constraint given index
      */
      virtual agx::ElementaryConstraint* getElementaryConstraint( const agx::UInt index ) const;

      /**
      \param name - name of the elementary constraint
      \return elementary constraint given name
      */
      virtual agx::ElementaryConstraint* getElementaryConstraint( const agx::String& name ) const;

      /**
      \param index - index of the secondary constraint
      \return secondary constraint given index
      */
      virtual agx::ElementaryConstraint* getSecondaryConstraint( const agx::UInt index ) const;

      /**
      \param name - name of the elementary constraint
      \return elementary constraint given name
      */
      virtual agx::ElementaryConstraint* getSecondaryConstraint( const agx::String& name ) const;

      /**
      \return the secondary constraint of specified type after \p count occurrences (of that type)
      */
      template< typename T >
      T* findSecondaryConstraintGivenType( const agx::UInt count = 0 ) const;

      /**
      \return a motor if a motor is present for this constraint, otherwise nullptr
      */
      agx::Motor1D* getMotor1D( agx::UInt number = 0 ) const;

      /**
      \return a lock if a lock is present for this constraint, otherwise nullptr
      */
      agx::Lock1D* getLock1D( agx::UInt number = 0 ) const;

      /**
      \return a range if a range is present for this constraint, otherwise nullptr
      */
      agx::Range1D* getRange1D( agx::UInt number = 0 ) const;

      /**
      \return a friction controller if present for this constraint, otherwise nullptr
      */
      agx::FrictionController* getFrictionController( agx::UInt number = 0 ) const;

      /**
      If the constraint supports angle calculations and 'dof' is in range, this method returns the angle of the given degree of freedom.
      \return the angle given degree of freedom - if 'dof' is out of range or if this constraint doesn't support angles this method returns 0
      */
      virtual agx::Real getAngle( agx::UInt /*dof*/ = 0 ) const { return 0; }

      /**
      If this method is implemented, a call to this method will return the current speed of the angle.
      This is only valid for 1DOF and 2DOF constraints (such as hinge, prismatic, distance and cylindrical).
      \return the current angle speed, returns 0 for constraints that are non-compatible to this call
      */
      virtual agx::Real getCurrentSpeed( agx::UInt /*dof*/ = 0 ) const { return 0; }

      /**
      Get the regularization parameter i, i.e., the regularization parameter for equation i
      which is constraint dependent.
      \param i - corresponds to the i'th constraint row of this constraint
      \return regularization parameter for equation i
      */
      agx::RegularizationParameters* userAPIgetRegularizationParameters( agx::UInt i );
      const agx::RegularizationParameters* userAPIgetRegularizationParameters( agx::UInt i ) const;

      /**
      \return the compliance of the i'th constraint row of this constraint
      */
      agx::Real userAPIgetCompliance( agx::UInt dof ) const;

      /**
      Set the compliance for constraint row i of this constraint.
      \param compliance - compliance
      \param dof - corresponds to the i'th constraint row of this constraint
      */
      void userAPIsetCompliance( agx::Real compliance, int dof );

      /**
      \return the damping of the i'th constraint row of this constraint
      */
      agx::Real userAPIgetDamping( agx::UInt dof ) const;

      /**
      Set the damping for constraint row i of this constraint.
      \param damping - damping
      \param dof - corresponds to the i'th constraint row of this constraint
      */
      void userAPIsetDamping( agx::Real damping, int dof );

      /**
      Assign force range, of an elementary constraint, for a given DOF.
      \note For controllers (secondary constraints), use their specific methods, e.g., constraint->getMotor1D()->setForceRange( fRange ).
      \param forceRange - value of force range for the given DOF
      \param dof - index of the DOF (-1 indicates all)
      */
      void userAPIsetForceRange( agx::RangeReal forceRange, agx::Int dof );

      /**
      Get the force range for DOF \p dof.
      \note Indices > number of DOF for this constraint is not defined, an infinite force range will be returned.
      \note For controllers (secondary constraints), use their specific method, e.g., constraint->getMotor1D()->getForceRange().
      \param dof - index of the requested DOF
      */
      agx::RangeReal userAPIgetForceRange( agx::UInt dof ) const;

      /**
      Consider using \p getLastForce instead. This method returns the magnitude of the force in
      a given degree of freedom (DOF). Enabled controllers in the given DOF will NOT be included,
      agx::ElementaryConstraint::getCurrentForce has to be used for that.
      \return the current force magnitude (i.e., the force magnitude last time step) in given DOF >= 0 && <= NUM_DOFS
      \sa setEnableComputeForces and getLastForce
      */
      agx::Real userAPIgetCurrentForce( agx::UInt dof ) const;

      /**
      Calculates the current number of active rows, including both elementary and secondary
      constraints. The result can be used to allocate temporary buffers for additional calls
      to solver related methods, e.g., getViolation.
      \return the (current) total number of active rows in this constraint
      */
      agx::UInt calculateNumActiveRows() const;


      // **************************************************************************

      DOXYGEN_START_INTERNAL_BLOCK()

      /**
      \note For the result of calling this method to make sense, the constraint
            has to be prepared and in the solver.
      \return the compliance for active dof
      */
      agx::RealValarray getCompliance() const;

      /**
      \return internal data for merge split
      */
      agx::Referenced* getInternalData() const;

      /**
      Assign merge split data for this body.
      */
      void setInternalData( agx::Referenced* data );

      /**
      \return the Simulation this constraint belongs to
      */
      AGX_FORCE_INLINE agxSDK::SimulationProxy* getSimulationProxy() { return m_simulationProxy; }
      AGX_FORCE_INLINE const agxSDK::SimulationProxy* getSimulationProxy() const { return m_simulationProxy; }

      agx::Physics::ConstraintForcesPtr getForceData();
      void setForceData(agx::Physics::ConstraintForcesPtr forceData);

      virtual void storeLightData( agxStream::StorageStream& ) const {}
      virtual void restoreLightData( agxStream::StorageStream& ) {}

      DOXYGEN_END_INTERNAL_BLOCK()

    protected:
      enum State : Int32
      {
        VALID = 1 << 0,                     // Replaces m_valid.
        IMPACTING = 1 << 1,                 // Replaces m_impactState.
        SUPPORTS_COMPLIANCE_MATRIX = 1 << 2 // Set if this constraint supports write access to the compliance matrix
                                            // allocated in the system matrix.
      };

      using LocalState = agx::BitState<State, Int32>;

    protected:
      friend class Constraint;
      /**
      Set the Simulation this constraint belongs to
      \param simulation - the Simulation this constraint belongs to
      */
      void setSimulation( agxSDK::Simulation* simulation );

      /**
      Enable/disable compliance matrix callbacks from the direct solver
      where read/write access to the compliance matrix will be available
      in an implementation of updateComplianceMatrix.
      \param supportsComplianceMatrix - true to enable access, false to disable (Default: Disabled).
      */
      void setSupportsComplianceMatrix( Bool supportsComplianceMatrix );

    protected:
      ElementaryConstraint::RefContainer  m_ec;                         /**< Vector of elementary constraints */
      ElementaryConstraint::RefContainer  m_sc;                         /**< Vector of secondary constraints */

      ConstraintRigidBodyContainer        m_bodies;                     /**< Vector with observer pointers to bodies */
      ConstrainedBodiesState              m_constrainedBodiesState;     /**< State of the included bodies, updated in updateValid. */
      mutable ConstraintImplPtrVector     m_subConstraints;             /**< Vector with sub-constraints */
      UInt                                m_numRows;                    /**< Number of rows used by this constraint */
      BlockStructure                      m_blockStructure;             /**< The block structure; number of block, rows, dynamic blocks etc. */
      UInt                                m_blockRowIndex;              /**< Start block row for this constraint in the global matrix */
      int                                 m_tag;                        /**< Tag for this constraint. Default tag is either BINARY or MANY_BODY. */

    private:
      agxSDK::SimulationProxyRef          m_simulationProxy;            /**< Pointer to the simulation this constraint belongs to */
      Physics::ConstraintForcesPtr        m_forceData;
      ref_ptr<Referenced>                 m_internalData;

    protected:
      Constraint::SolveType               m_solveType;                  /**< Only for supported solvers. Default: DIRECT. agxData::Values: DIRECT, ITERATIVE or DIRECT_AND_ITERATIVE. */
      bool                                m_enable;                     /**< User controlled enable flag, constraint ignored by solver if false */
      bool                                m_enableLinearization;        /**< When necessary, tweak the values sent to the solver to make violation depend linearly on force or torque. */
      LocalState                          m_localState;                 /**< Current local state, updated all the time, not serialized. */
  };

  class AGXPHYSICS_EXPORT HighLevelConstraintImplementation : public ConstraintImplementation, public virtual agxStream::Serializable
  {
    public:
      HighLevelConstraintImplementation();
      virtual ~HighLevelConstraintImplementation();

      /**
      Updates the valid conditions for this constraint. This constraint is not valid
      if for example; one or more bodies are removed (either from simulation or deallocated),
      one or more bodies are disabled or the number of dynamic bodies in this constraint
      is zero.
      \return m_valid, true if valid, otherwise false
      */
      virtual bool updateValid() override;

      /**
      Prepares this constraint. Update attachment transforms given the current transforms
      of the bodies (in most cases equivalent to calculate the Jacobian), compute the
      joint angles (if this constraint supports that) and calculates the amount of data
      needed.
      */
      virtual void prepare() override;

      /**
      Updates the Jacobian matrices. The elementary constraints pushes data into this constraints
      Jacobian matrices. For some solvers this is equivalent to push the data into the global matrix.
      \return the number of rows this constraint uses (could differ from time step to time step)
      */
      virtual size_t updateJacobian( agx::Jacobian6DOFElement* jacobians ) override;

      /**
      Peak at the solution after the solver is done.
      \param jacobians - Jacobian rows for this constraint (index 0 is first dynamic body first row)
      \param solution - solution (impulse - divide by dt to get force)
      \param dt - time step size used to get solution
      */
      virtual void postSolveCallback( const agx::Jacobian6DOFElement* jacobians, const agx::Real* solution, agx::Real dt ) override;

      /**
      The general version of rebind takes the first attachment frame and moves the other attachment
      frame so they have the exact same world transform. This means that the first rigid body is the
      reference (other is another rigid body or world). E.g., Prismatic and Hinge axis will be preserved
      seen from the reference body.
      \return true if successful - otherwise false
      */
      virtual bool rebind() override;

      /**
      Computes the forces, if valid and enabled.
      \param jacobians - jacobians for this constraint
      */
      virtual void computeForces( const agx::Jacobian6DOFElement* jacobians );

      /**
      Computes forces on rb1 and rb2 given this constraint has been solved and all the
      elementary constraints have their current force updated. The vector \p result will
      be of size 3 * numBodies, where the entries are:
      [forceOnRb1 torqueOnRb1 addedTorqueDueToAnchorPosRb1 forceOnRb2 torqueOnRb2 addedTorqueDueToAnchorPosRb2]
      Note that the size is 3 if rb2 == nullptr.
      \param jacobians - jacobians for this constraint
      \param result - resulting vector with forces
      \return true if data was written to \p result - otherwise false
      */
      virtual agx::Bool computeForces( const agx::Jacobian6DOFElement* jacobians, agx::Vec3Vector& result ) const;

      /**
      Enable (or disable) computation of the forces applied to the dynamic bodies in this constraint.
      Matrix-vector operation to compute the forces after solve.
      \sa getLastForce
      \param enable - true to enable, false to disable
      */
      virtual void setEnableComputeForces( agx::Bool enable ) override;

      /**
      \return true if this constraint has been enabled to compute the forces applied to its bodies - otherwise false
      */
      virtual agx::Bool getEnableComputeForces() const override;

      /**
      If 'compute forces' is enabled, returns the last force and torque applied by this constraint on
      the body with \p bodyIndex. The force is given in world coordinates and is the one applied
      at the anchor position of this constraint.

      The result includes force and torque from this constraint including all enabled
      controllers (e.g., motors, locks, ranges).
      \sa setEnableComputeForces
      \param bodyIndex      - index of body, if number of bodies = 1 and bodyIndex = 1, the force and
                              torque applied to "the world body" is returned
      \param retForce       - the force applied by this constraint on body at \p bodyIndex last solve
      \param retTorque      - the torque applied by this constraint on body at \p bodyIndex last solve
      \param giveForceAtCm  - this parameter affects the resulting torque. The default behavior (giveForceAtCm = false)
                              calculates the force applied by this constraint at the anchor position.
                              Letting giveForceAtCm = true, the force will be applied at center of mass,
                              affecting the torque as \f$T_{new} = T - r \times F\f$ where r is the vector
                              from the anchor point to the center of mass of the body.
      \return true if resulting force and torque was written to \p retForce and \p retTorque - otherwise false
      */
      virtual agx::Bool getLastForce( agx::UInt bodyIndex, agx::Vec3& retForce, agx::Vec3& retTorque, agx::Bool giveForceAtCm = false ) const override;

      /**
      If 'compute forces' is enabled, returns the last force and torque applied by this constraint on
      the body \p rb. The force is given in world coordinates and is the one applied at the anchor
      position of this constraint.

      The result includes force and torque from this constraint including all enabled
      controllers (e.g., motors, locks, ranges).
      \sa setEnableComputeForces
      \param rb             - body in this constraint. If this constraint is attached in world, nullptr can be used.
      \param retForce       - the force applied by this constraint on body \p rb at the last solve
      \param retTorque      - the torque applied by this constraint on body \p rb at the last solve
      \param giveForceAtCm  - this parameter affects the resulting torque. The default behavior (giveForceAtCm = false)
                              calculates the force applied by this constraint at the anchor position.
                              Letting giveForceAtCm = true, the force will be applied at center of mass,
                              affecting the torque as \f$T_{new} = T - r \times F\f$ where r is the vector
                              from the anchor point to the center of mass of the body.
      \return true if resulting force and torque was written to \p retForce and \p retTorque - otherwise false
      */
      virtual agx::Bool getLastForce( const RigidBody* rb, Vec3& retForce, Vec3& retTorque, agx::Bool giveForceAtCm = false ) const override;

      /**
      If 'compute forces' is enabled, returns the last force and torque applied by this constraint on
      the body with \p bodyIndex. The force is given in the frame of the constraint.

      The result includes force and torque from this constraint including all enabled
      controllers (e.g., motors, locks, ranges).
      \sa setEnableComputeForces
      \param bodyIndex      - index of body, if number of bodies = 1 and bodyIndex = 1, the force and
                              torque applied to "the world body" is returned
      \param retForce       - the force applied by this constraint on body at \p bodyIndex last solve
      \param retTorque      - the torque applied by this constraint on body at \p bodyIndex last solve
      \param giveForceAtCm  - this parameter affects the resulting torque. The default behavior (giveForceAtCm = false)
                              calculates the force applied by this constraint at the anchor position.
                              Letting giveForceAtCm = true, the force will be applied at center of mass,
                              affecting the torque as \f$ T_{new} = T - r \times F\f$ where r is the vector
                              from the anchor point to the center of mass of the body.
      \return true if resulting force and torque was written to \p retForce and \p retTorque - otherwise false
      */
      virtual agx::Bool getLastLocalForce( agx::UInt bodyIndex, agx::Vec3& retForce, agx::Vec3& retTorque, agx::Bool giveForceAtCm = false ) const override;

      /**
      If 'compute forces' is enabled, returns the last force and torque applied by this constraint on
      the body \p rb. The force is given in the frame of the constraint.

      The result includes force and torque from this constraint including all enabled
      controllers (e.g., motors, locks, ranges).
      \sa setEnableComputeForces
      \param rb             - body in this constraint. If this constraint is attached in world, nullptr can be used.
      \param retForce       - the force applied by this constraint on body \p rb at the last solve
      \param retTorque      - the torque applied by this constraint on body \p rb at the last solve
      \param giveForceAtCm  - this parameter affects the resulting torque. The default behavior (giveForceAtCm = false)
                              calculates the force applied by this constraint at the anchor position.
                              Letting giveForceAtCm = true, the force will be applied at center of mass,
                              affecting the torque as \f$ T_{new} = T - r \times F\f$ where r is the vector
                              from the anchor point to the center of mass of the body.
      \return true if resulting force and torque was written to \p retForce and \p retTorque - otherwise false
      */
      virtual agx::Bool getLastLocalForce( const agx::RigidBody* rb, agx::Vec3& retForce, agx::Vec3& retTorque, agx::Bool giveForceAtCm = false ) const override;

      // **************************************************************************
      // Internal utility methods for one or two body constraints *****************
      // **************************************************************************

      typedef agx::Bool (*CreateElementaryConstraintsFunction)( HighLevelConstraintImplementation* );

      bool construct( RigidBody* rb1, Frame* rb1AttachmentFrame, RigidBody* rb2, Frame* rb2AttachmentFrame, CreateElementaryConstraintsFunction = nullptr );
      bool setupOneOrTwoBodySystem( RigidBodyAttachment* a1, RigidBodyAttachment* a2 );
      bool validateRigidBodiesAndAttachments( RigidBody* rb1, Frame* rb1AttachmentFrame, RigidBody* rb2, Frame* rb2AttachmentFrame );

      /**
      \return the rigid body attachment given a rigid body (returns 0 if invalid)
      */
      virtual agx::Attachment* userAPIgetAttachment( const agx::RigidBody* rb ) const override;

      /**
      \return the constraint attachment at index
      */
      virtual agx::Attachment* userAPIgetAttachment( agx::UInt index ) const override;

      /**
      \return the constraint attachment pair
      */
      virtual agx::AttachmentPair* getAttachmentPair() const override;

      /**
      If the constraint supports angle calculations and 'dof' is in range, this method returns the angle of the given degree of freedom.
      \return the angle given degree of freedom - if 'dof' is out of range or if this constraint doesn't support angles this method returns 0
      */
      virtual agx::Real getAngle( agx::UInt dof = 0 ) const override;

      /**
      \return true if one or more bodies are deleted
      */
      bool bodiesValid() const;

      /**
      \return the constraint attachment of given type (should always be valid to call after init of a constraint)
      */
      template< typename T >
      T* getAttachment( agx::UInt index );

      /**
      \return the constraint attachment of given type (should always be valid to call after init of a constraint)
      */
      template< typename T >
      const T* getAttachment( agx::UInt index ) const;

      AGXSTREAM_DECLARE_ABSTRACT_SERIALIZABLE( agx::HighLevelConstraintImplementation );

      virtual void restore( agxStream::InputArchive& in ) override;
      virtual void store( agxStream::OutputArchive& out ) const override;

      virtual void storeLightData( agxStream::StorageStream& str ) const override;
      virtual void restoreLightData( agxStream::StorageStream& str ) override;

    protected:
      AttachmentPair m_attachmentPair;
      Vec3Vector     m_lastForces;
  };

  AGX_FORCE_INLINE UInt& ConstraintImplementation::ConstrainedBodiesState::BodiesInfo::numDynamicBodies()
  {
    return m_numDynamicBodies;
  }

  AGX_FORCE_INLINE UInt ConstraintImplementation::ConstrainedBodiesState::BodiesInfo::numDynamicBodies() const
  {
    return m_numDynamicBodies;
  }

  AGX_FORCE_INLINE UInt& ConstraintImplementation::ConstrainedBodiesState::BodiesInfo::numNonStaticBodies()
  {
    return m_numNonStaticBodies;
  }

  AGX_FORCE_INLINE UInt ConstraintImplementation::ConstrainedBodiesState::BodiesInfo::numNonStaticBodies() const
  {
    return m_numNonStaticBodies;
  }

  inline const ConstraintImplementation::ConstrainedBodiesState::BodiesInfo& ConstraintImplementation::ConstrainedBodiesState::getOriginalInfo() const
  {
    return m_originalInfo;
  }

  inline const ConstraintImplementation::ConstrainedBodiesState::BodiesInfo& ConstraintImplementation::ConstrainedBodiesState::getSolverInfo() const
  {
    return m_solverInfo;
  }

  inline const Int32Vector& ConstraintImplementation::ConstrainedBodiesState::getPermutation() const
  {
    return m_permutation;
  }

  inline Int32Vector& ConstraintImplementation::ConstrainedBodiesState::getPermutation()
  {
    return m_permutation;
  }

  inline const Int32Vector& ConstraintImplementation::ConstrainedBodiesState::getBodyIndices() const
  {
    return m_bodyIndices;
  }

  inline Int32Vector& ConstraintImplementation::ConstrainedBodiesState::getBodyIndices()
  {
    return m_bodyIndices;
  }

  inline const ConstraintRigidBodyContainer& ConstraintImplementation::ConstrainedBodiesState::getSolverBodies() const
  {
    return m_bodies;
  }
#if !defined(SWIGPYTHON)
  AGX_FORCE_INLINE ConstraintImplementation::BodyView::BodyView(const ConstrainedBodiesState::BodiesInfo &info, const ConstraintRigidBodyContainer &bodies) : m_info(info), m_bodies(bodies)
  {
  }

  AGX_FORCE_INLINE ConstraintImplementation::BodyView::BodyView( const BodyView& other ) : m_info( other.m_info ), m_bodies( other.m_bodies )
  {
  }

  AGX_FORCE_INLINE UInt ConstraintImplementation::BodyView::getNumDynamicBodies() const
  {
    return m_info.numDynamicBodies();
  }

  AGX_FORCE_INLINE UInt ConstraintImplementation::BodyView::getNumNonStaticBodies() const
  {
    return m_info.numNonStaticBodies();
  }

  AGX_FORCE_INLINE UInt ConstraintImplementation::BodyView::getNumBodies() const
  {
    return m_bodies.size();
  }
  AGX_FORCE_INLINE RigidBody* ConstraintImplementation::BodyView::getBodyAt( agx::UInt i )
  {
    agxAssert( i < getNumBodies() ); return m_bodies[ i ];
  }

  AGX_FORCE_INLINE const RigidBody* ConstraintImplementation::BodyView::getBodyAt( agx::UInt i ) const
  {
    agxAssert( i < getNumBodies() ); return m_bodies[ i ];
  }

  AGX_FORCE_INLINE const ConstraintRigidBodyContainer& ConstraintImplementation::BodyView::getBodies() const
  {
    return m_bodies;
  }

  AGX_FORCE_INLINE ConstraintImplementation::SolverBodyView::SolverBodyView( ConstrainedBodiesState& state )
    : BodyView( state.getSolverInfo(), state.getSolverBodies() ), m_state( state )
  {
  }

  AGX_FORCE_INLINE ConstraintImplementation::SolverBodyView::SolverBodyView( const SolverBodyView& other )
    : BodyView( other.m_state.getSolverInfo(), other.m_state.getSolverBodies() ), m_state( other.m_state )
  {
  }

  AGX_FORCE_INLINE const ConstraintImplementation::ConstrainedBodiesState& ConstraintImplementation::SolverBodyView::getState() const
  {
    return m_state;
  }

  AGX_FORCE_INLINE ConstraintImplementation::ConstrainedBodiesState& ConstraintImplementation::SolverBodyView::getState()
  {
    return m_state;
  }

  AGX_FORCE_INLINE const RigidBody* ConstraintImplementation::SolverBodyView::getPermutedBodyAt( UInt i ) const
  {
    agxAssert( i < m_state.getPermutation().size() );
    UInt permutedIndex = (UInt)m_state.getPermutation()[ i ];
    agxAssert( permutedIndex < getNumBodies() );
    return getBodyAt( permutedIndex );
  }

  AGX_FORCE_INLINE RigidBody* ConstraintImplementation::SolverBodyView::getPermutedBodyAt( UInt i )
  {
    agxAssert( i < m_state.getPermutation().size() );
    UInt permutedIndex = (UInt)m_state.getPermutation()[ i ];
    agxAssert( permutedIndex < getNumBodies() );
    return getBodyAt( permutedIndex );
  }
#endif
  template< typename T >
  inline T* ConstraintImplementation::findSecondaryConstraintGivenType( UInt count /*= 0*/ ) const
  {
    UInt currentNumber = 0;
    for ( UInt i = 0, num = m_sc.size(); i < num; ++i ) {
      T* ec = dynamic_cast< T* >( m_sc[ i ].get() );
      if ( ec != nullptr && currentNumber == count )
        return ec;
      currentNumber += ec != nullptr;
    }

    return nullptr;
  }

  inline void ConstraintImplementation::preSystemCallback( DynamicsSystem* /*system*/ ) {}

  inline void ConstraintImplementation::postSystemCallback( DynamicsSystem* /*system*/ ) {}

  inline AttachmentPair* HighLevelConstraintImplementation::getAttachmentPair() const
  {
    return const_cast< AttachmentPair* >( &m_attachmentPair );
  }

  template< typename T >
  AGX_FORCE_INLINE T* HighLevelConstraintImplementation::getAttachment( UInt index )
  {
    return m_attachmentPair[ index ] != nullptr ? m_attachmentPair[ index ]->as< T >() : nullptr;
  }

  template< typename T >
  AGX_FORCE_INLINE const T* HighLevelConstraintImplementation::getAttachment( UInt index ) const
  {
    return m_attachmentPair[ index ] != nullptr ? m_attachmentPair[ index ]->as< T >() : nullptr;
  }

  AGX_FORCE_INLINE bool HighLevelConstraintImplementation::bodiesValid() const
  {
    return m_attachmentPair[ 0 ] != nullptr && !m_attachmentPair[ 0 ]->objectDeleted() &&
           m_attachmentPair[ 1 ] != nullptr && !m_attachmentPair[ 1 ]->objectDeleted();
  }

  /**
  a and b should be orthogonal
  */
  AGX_FORCE_INLINE Real getAngle( const Vec3& a, const Vec3& b, const Vec3& c )
  {
    return std::atan2( c * b, c * a );
  }

  /**
  Add secondary controllers with default name to the constraint.
  */
  AGXPHYSICS_EXPORT void addSecondaryConstraints1DOF( agx::HighLevelConstraintImplementation* constraint, agx::Angle* angle );

  /**
  Add secondary controllers with default name to the constraint.
  */
  AGXPHYSICS_EXPORT void addSecondaryConstraints2DOF( agx::HighLevelConstraintImplementation* constraint, agx::SeparationAngle* sepAngle, agx::RotationalAngle* rotAngle );
} // namespace agx

#ifdef _MSC_VER
# pragma warning(pop)
#endif


#endif /* _CONSTRAINTIMPLEMENTATION_H_ */
