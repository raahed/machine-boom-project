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

#include <agx/Vector.h>
#include <agx/Vec3.h>
#include <agx/Range.h>
#include <agx/agx_valarray_types.h>

namespace agx
{
  class FrictionModel;
  class INlSolveDataH5;
  class SparseRangeReal;
  class ConstraintImplementation;
  class DirectSolverData;
  class RigidBody;

  namespace Physics
  {
    class RigidBodyData;
    class RigidBodyPtr;
  }

  /**
  Interface class for solver callbacks.
  */
  class AGXPHYSICS_EXPORT NlmcpCallback : public agx::Referenced
  {
    public:
      struct Args
      {
        Args( agx::RealValarray& q,
              const agx::RealValarray& z,
              const agx::SparseRangeReal& bounds,
              const agx::Real* iterativeSolutionContacts,
              const agx::Real* iterativeSolutionBinaryConstraints,
              const agx::Real* iterativeSolutionManyBodyConstraints,
              agx::INlSolveDataH5* solveDataH5 )
          : q( q ),
            z( z ),
            bounds( bounds ),
            iterativeSolutionContacts( iterativeSolutionContacts ),
            iterativeSolutionBinaryConstraints( iterativeSolutionBinaryConstraints ),
            iterativeSolutionManyBodyConstraints( iterativeSolutionManyBodyConstraints ),
            solveDataH5( solveDataH5 )
        {
        }

        Args( const Args& ) = delete;
        Args& operator = ( const Args& ) = delete;

        agx::RealValarray& q;
        const agx::RealValarray& z;
        const agx::SparseRangeReal& bounds;
        const agx::Real* iterativeSolutionContacts;
        const agx::Real* iterativeSolutionBinaryConstraints;
        const agx::Real* iterativeSolutionManyBodyConstraints;
        agx::INlSolveDataH5* solveDataH5;
      };

    public:
      /**
      This method is a test for the non-linear solver if this callback is active for non-linear solve.
      \return true if active and should receive update calls from inner loop of the non-linear solver,
              false if this callback isn't interested in updates
      */
      virtual agx::Bool initialize( const agx::NlmcpCallback::Args& args ) = 0;

      /**
      Calculate residual given current solution.
      */
      virtual agx::Real calculateResidual( const agx::NlmcpCallback::Args& args ) const = 0;

      /**
      Update call when bounds and/or q may be updated given the current solution z.
      */
      virtual void update( const agx::NlmcpCallback::Args& args ) const = 0;

      /**
      Before direct solve, after iterative solve.
      */
      virtual void postIterativeSolve( const agx::NlmcpCallback::Args& args ) const = 0;

      /**
      \return the permuted start row in global matrix
      */
      agx::UInt getDirectRow() const;

      /**
      \internal

      Assign direct row of permuted matrix.
      */
      void setDirectRow( agx::UInt directRow );

      /**
      \return the row used in iterative buffers
      */
      agx::UInt getIterativeRow() const;

      /**
      \internal

      Assign iterative row for iterative buffers.
      */
      void setIterativeRow( agx::UInt iterativeRow );

      /**
      \return the index in sparse bounds (argument in update method)
      */
      agx::UInt getSparseBoundsIndex() const;

      /**
      \internal

      Assign new sparse bounds index.
      */
      void setSparseBoundsIndex( agx::UInt sparseBoundsIndex );

      /**
      \return the number of rows this callback handles
      */
      agx::UInt getNumRows() const;

      /**
      \internal

      Assign the number of rows this callback handles.
      */
      void setNumRows( agx::UInt numRows );

    protected:
      /**
      Default constructor.
      */
      NlmcpCallback();

      /**
      Protected destructor, reference counted object.
      */
      virtual ~NlmcpCallback();

    private:
      agx::UInt m_directRow;
      agx::UInt m_iterativeRow;
      agx::UInt m_sparseBoundsIndex;
      agx::UInt m_numRows;
  };

  using NlmcpCallbackRef = ref_ptr<NlmcpCallback>;

  AGX_FORCE_INLINE agx::UInt NlmcpCallback::getDirectRow() const
  {
    return m_directRow;
  }

  AGX_FORCE_INLINE agx::UInt NlmcpCallback::getIterativeRow() const
  {
    return m_iterativeRow;
  }

  AGX_FORCE_INLINE agx::UInt NlmcpCallback::getSparseBoundsIndex() const
  {
    return m_sparseBoundsIndex;
  }

  AGX_FORCE_INLINE agx::UInt NlmcpCallback::getNumRows() const
  {
    return m_numRows;
  }

  /**
  Data from the solver the ConstraintNlmcpCallback is part of.
  Use this data in onSetContext.
  */
  struct AGXPHYSICS_EXPORT NlmcpCallbackSolverData
  {
    const Physics::RigidBodyData& rigidBody; /**< Global rigid body data. */
    const DirectSolverData& directSolverData; /**< Direct solver data the ConstraintNlmcpCallback constraint is part of. */

    /**
    \param rb - rigid body instance in the direct solver
    \return permuted start row index in the direct solver (e.g., q and z)
    */
    UInt32 getPermutedDirectStartRow( const RigidBody* rb ) const;

    /**
    \param entity - rigid body entity in the direct solver
    \return permuted start row index in the direct solver (e.g., q and z)
    */
    UInt32 getPermutedDirectStartRow( const Physics::RigidBodyPtr& entity ) const;

    /**
    \param globalIndex - rigid body entity index
    \return permuted start row index in the direct solver (e.g., q and z)
    */
    UInt32 getRigidBodyPermutedDirectStartRow( Index globalIndex ) const;

    /**
    Finds number of rows the given rigid body has allocated in the solver.
    6 for rigid bodies and 3 for particles.
    \param rb - rigid body
    \return number of rows allocated in the solver for the given rigid body
    */
    UInt32 getNumRows( const RigidBody* rb ) const;

    /**
    Finds number of rows the given rigid body entity has allocated in the solver.
    6 for rigid bodies and 3 for particles.
    \param entity - rigid body entity
    \return number of rows allocated in the solver for the given rigid body
    */
    UInt32 getNumRows( const Physics::RigidBodyPtr& entity ) const;

    /**
    Finds number of rows the given rigid body index has allocated in the solver.
    6 for rigid bodies and 3 for particles.
    \param globalIndex - global index of the rigid body
    \return number of rows allocated in the solver for the given rigid body index
    */
    UInt32 getRigidBodyNumRows( Index globalIndex ) const;

    NlmcpCallbackSolverData( const NlmcpCallbackSolverData& ) = delete;
    NlmcpCallbackSolverData& operator = ( const NlmcpCallbackSolverData& ) = delete;
  };

  /**
  Constraint solver callback interface with context and start rows for
  context.
  */
  class AGXPHYSICS_EXPORT ConstraintNlmcpCallback : public agx::NlmcpCallback
  {
    public:
      /**
      \return constraint where this solver callback is used
      */
      const agx::ConstraintImplementation* getContext() const;

      /**
      \return direct row of context constraint
      */
      agx::UInt getContextDirectRow() const;

      /**
      \return iterative row of context constraint
      */
      agx::UInt getContextIterativeRow() const;

    protected:
      ConstraintNlmcpCallback();
      virtual ~ConstraintNlmcpCallback();

      /**
      Called when context constraint has been set and all data is valid.
      \param solverData - solver related data our context is part of
      */
      virtual void onSetContext( const NlmcpCallbackSolverData& solverData );

    private:
      friend class ConstraintImplementation;
      void setContext( const ConstraintImplementation* context,
                       UInt contextDirectRow,
                       UInt contextIterativeRow,
                       const NlmcpCallbackSolverData& solverData );

    private:
      const ConstraintImplementation* m_context;
      agx::UInt m_contextDirectRow;
      agx::UInt m_contextIterativeRow;
  };

  using ConstraintNlmcpCallbackRef = ref_ptr<ConstraintNlmcpCallback>;
} // namespace agx
