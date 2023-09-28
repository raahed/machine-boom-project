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

#ifndef AGX_BOXEDKELLERALGORITHM_H
#define AGX_BOXEDKELLERALGORITHM_H

#include <agx/MixedLCP.h>
#include <agx/Logger.h>


namespace agx
{
  class AGXPHYSICS_EXPORT BoxedKellerAlgorithm : public agx::NlMixedCp::McpAlgorithm
  {
    public:
      /**
      Construct given global tolerance and the maximum iterations.
      */
      BoxedKellerAlgorithm( agx::Real globalTolerance, agx::UInt maxNumIterations, bool gatherStatistics );

      /**
      Solves MCP using Keller's Principal Pivot Method.
      */
      virtual agx::NlMixedCp::McpAlgorithm::SolveReport solve( agx::NlMixedCp::McpAlgorithm::Args& args ) override;

      /**
      Searches solution assuming z is feasible and the index set is up to date.
      */
      agx::NlMixedCp::McpAlgorithm::SolveReport searchSolution( agx::NlMixedCp::McpAlgorithm::Args& args );

      /**
      Calculates a feasible solution z and switches indices. The slack is also updated.
      \param args - arguments
      \param maxNumIterations - maximum number of iterations used
      \return true if done before max number of iterations reached
      */
      agx::Bool makeSolutionFeasible( Args& args, agx::UInt maxNumIterations = agx::UInt( 64 ) );

    protected:

      /**
      Data for pivot index with sign and current range of the most violated index.
      */
      struct PivotIndex
      {
        PivotIndex() : index( agx::InvalidIndex ), sign( agx::Real( 0 ) ), done( false ) {}

        agx::UInt index;
        agx::Real sign;
        agx::Bool done;
        agx::RangeReal range;
      };

      /**
      Data for step length with the different thetas and indices (r2 and r3) of other
      bounded equations that may change state.
      */
      struct StepLength
      {
        agx::Real mss;
        agx::Real t;
        agx::Real t0;
        agx::Real t1;
        agx::Real t2;
        agx::Real t3;
        agx::UInt r2;
        agx::UInt r3;
      };

#if 0
// List of solver statistics we want to extract. Some of these are not specific
// to Keller and should therefore be included in McpAlgorithmStats.
      struct Stats
      {
        std::vector<int> inner_iterations; /// one entry for each outer iteration
        size_t rank_update;                /// number of rank updates (should be same as sum of inner_iterations )
        std::vector<int> updates;          /// for each bounded variable, this keeps track of the number of times it was changed.
        int  projections;                  /// number of projection iterations
        int  projection_deletions;         /// total number of variables deleted
        int  variables;                    /// total number of variables
        int  active;                       /// total number of free variables at solution
        Stats( size_t n ) : inner_iterations(), updates(n, -1), rank_update(0), projections(0),
                            projection_deletions(0), variables(n), active(0){}

        Stats() : Stats(0){}
      };
#endif

    protected:



      /**
      Reference counted object - protected destructor.
      */
      virtual ~BoxedKellerAlgorithm();

      /**
      Custom implementation of switch indices where slack is ignored and we only pivot on infeasible elements z.
      */
      virtual agx::UInt switchIndices( const agx::RealValarray& z, const agx::RealValarray& w, const agx::SparseRangeReal& bounds, agx::IndexSet& indexSet, agx::Real tolerance = agx::Real( 1E-8 ) ) const override;


    protected:
      /**
      Initializes the index set.
      \param args - arguments
      */
      void initializeIndexSet( Args& args ) const;

      /**
      Finds the initial starting point. Performs solvePp and updates z and w in args.
      \param args - arguments
      */
      void getInitialPoint( Args& args );

      /**
      Finds and returns the most violated index.
      \param args - arguments
      \return index data
      */
      PivotIndex getPivotIndex( Args& args ) const;

      /**
      Finds the search direction given most violated index.
      \param args - arguments
      \param pivotCol - column data
      \return the search direction
      */
      agx::RealValarray getSearchDirection( Args& args, const agx::RealValarray& pivotCol );

      /**
      Calculates the step length given search direction.
      \param args - arguments
      \param searchDirection - the search direction
      \param pi - pivot index data
      \param pivotRow - pivot row data
      \return the step length
      */
      StepLength getStepLength( Args& args, const agx::RealValarray& searchDirection, const PivotIndex& pi, const agx::RealValarray& pivotRow ) const;

      /**
      Perform step, i.e., update z and w given search direction and step length.
      \param args - arguments
      \param sl - step length
      \param searchDirection - the search direction
      \param pi - pivot index data
      \param pivotCol - pivot column data
      */
      void doStep( Args& args, const StepLength& sl, agx::RealValarray& searchDirection, const PivotIndex& pi, const agx::RealValarray& pivotCol ) const;

      /**
      Given step length, updates the index set.
      \param args - arguments
      \param sl - step length
      \param pi - pivot index data
      \return true if done
      */
      agx::Bool doPivot( Args& args, const StepLength& sl, const PivotIndex& pi ) const;


    protected:
      agx::Real m_localTolerance;
  };

  typedef agx::ref_ptr< BoxedKellerAlgorithm > BoxedKellerAlgorithmRef;
}

#endif
