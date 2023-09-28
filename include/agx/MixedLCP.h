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

#include <agx/Referenced.h>
#include <agx/SparseMatrix.h>
#include <agx/SparseInverse.h>
#include <agx/IndexSet.h>
#include <agx/SparseTypes.h>
#include <agx/Solver.h>

#include <agxSabre/agxSabre.h>

#if defined(_MSC_VER)
#pragma warning( push )
#pragma warning( disable : 4275 ) // C4275: non dll-interface class 'x' used as base for dll-interface class 'y'
#endif


namespace agx
{
  class StabilityReport;
  class SparseInverse;
  class ColumnStorage;

  typedef agx::ref_ptr< class NlMixedCp > NlMixedCpRef;

  /**
  Storage location for solver statistics.

  Each McpAlgorithm holds one of these and records events of interest throughout
  the solve. It is important that the there is one of these per algorithm, per
  solve island, and that it is re-created every new step forward.
  */
  class McpAlgorithmStats
  {
    public:
      void setNumEquations(size_t numEquations);

      void reportAddedAndRemovedEquations(const agxSabre::UInt32Vector& add, const agxSabre::UInt32Vector& del);
      void reportNumNlMixedCpIterations(UInt numIterations);
      void reportMurtyIterations(UInt numIterations);
      void reportKellerMakeFeasibleIterations(UInt numIterations);
      void reportKellerSearchIterations(UInt numIterations);
      void reportNumFreeEquationsAtEnd(const agx::IndexSet& indexSet);

      void addStats(const McpAlgorithmStats& other);

      void clear(size_t numEquations);

      void setEnableGatherStatistics(bool enable);
      bool getEnableGatherStatistics() const;




      const std::vector<int>& timesAdded() const;
      const std::vector<int>& timesRemoved() const;
      UInt numNlMixedCpIterations() const;
      const std::vector<UInt>& numMurtyIterations() const;
      const std::vector<UInt>& numKellerMakeFeasibleIterations() const;
      const std::vector<UInt>& numKellerSearchIterations() const;
      const std::vector<UInt>& numFreeEquationsAtEnd() const;

    private:
      /**
      The number of times a particular equation has been added to the active set
      in \p solvePp. The container contains one counter per equation.
      */
      std::vector<int> m_timesAdded;

      /**
      The number of times a particular equation has been remvoed from the active
      set in \p solvePp. The container contains one counter per equation.
      */
      std::vector<int> m_timesRemoved;

      UInt m_numNlMixedCpIterations = 0;

      std::vector<UInt> m_numMurtyIterations;
      std::vector<UInt> m_numKellerMakeFeasibleIterations;
      std::vector<UInt> m_numKellerSearchIterations;

      std::vector<UInt> m_numFreeEquationsAtEnd;

      bool m_gatherStatistics = false;
  };

  /**
  Class that implements a method to solve linear- or nonlinear
  complementary problems.
  */
  class AGXPHYSICS_EXPORT NlMixedCp : public agx::Referenced
  {
    public:
      /**
      Interface class for algorithm to solve general mixed complementarity
      problem. After call to solve the solution is stored in the \p z array.
      */
      class AGXPHYSICS_EXPORT McpAlgorithm : public agx::Referenced
      {
        public:
          class AGXPHYSICS_EXPORT SolveReport
          {
            public:
              /**
              Construct given solve result.
              \param numIterations - number of iterations
              \param residual - residual of used solution
              \param successful - true if solution is within tolerance
              */
              SolveReport( agx::UInt numIterations, agx::Real residual, agx::Bool successful );

              /**
              \return the number of iterations used
              */
              agx::UInt getNumIterations() const;

              /**
              \return the residual of the solution used
              */
              agx::Real getResidual() const;

              /**
              \return true if solve was successful - otherwise false
              */
              agx::Bool isSuccessful() const;

            private:
              agx::UInt m_numIterations;
              agx::Real m_residual;
              agx::Bool m_successful;
          };

          class AGXPHYSICS_EXPORT Exception : public std::exception
          {
            public:
              Exception() : std::exception() {}
          };

          /**
          Structure for all the arguments passed around.
          */
          struct Args
          {
            Args( agx::SparseMatrix& m,
                  agx::RealValarray& q,
                  agx::RealValarray& z,
                  agx::RealValarray& z0,
                  const agx::SparseRangeReal& bounds,
                  agx::IndexSet& indexSet,
                  const agx::RealValarray& signs,
                  agx::SparseInverse& sInv,
                  agx::Bool warm = false,
                  agx::StabilityReport* stabilityReport = nullptr );

            ~Args();

            agx::SparseMatrix& m;
            agx::RealValarray& q;
            agx::RealValarray& z;
            agx::RealValarray& z0;
            agx::RealValarray  w;
            agx::RealValarray  rhs0;
            const agx::SparseRangeReal& bounds;
            agx::IndexSet& indexSet;
            const agx::RealValarray& signs;
            agx::Bool warm;
            agx::StabilityReport* stabilityReport;
            agx::ColumnStorage* columnStorage;
            agx::SparseInverse& inv;

            private:
              Args& operator = ( const Args& ) = delete;
          };

        public:
          /**
          Computes the complementarity error of a current solution.
          \param indexSet - index set
          \param bounds - bounded variables
          \param z - current solution
          \param w - current slack
          \return normalized error
          */
          static agx::Real getComplementarityError( const agx::IndexSet& indexSet, const agx::SparseRangeReal& bounds, const agx::RealValarray& z, const agx::RealValarray& w );

          /**
          Checks if the solution z is feasible, i.e., within bounds.
          \param bounds - bounded equations in z
          \param z - current solution
          \param threshold - feasible threshold
          \return true if z is feasible, otherwise false
          */
          static agx::Bool isFeasible( const SparseRangeReal& bounds, const RealValarray& z, const Real threshold );

          /**
          Save matrix, solution, right hand side, sign convention and bounds to a unique problem in \p filename.
          \param filename - name of file to save the problem to
          \param m - matrix
          \param z - solution
          \param q - rhs
          \param signs - sign convention (-1 of constraints 1 for bodies)
          \param bounds - bounds
          \return true if successful, otherwise false
          */
          static agx::Bool saveAsHdf5( const agx::String& filename, const SparseMatrix& m, RealValarray& z, RealValarray& q, RealValarray& signs, const agx::SparseRangeReal& bounds );

        public:
          /**
          Construct given minimum tolerance and maximum number of iterations.
          */
          McpAlgorithm( agx::Real globalTolerance, agx::UInt maxNumIterations, bool gatherStatistics );

          /**
          \return the tolerance (accuracy) of the solution
          */
          agx::Real getGlobalTolerance() const;

          /**
          Assign the tolerance (accuracy) of the solution.
          */
          void setGlobalTolerance( agx::Real globalTolerance );

          /**
          \return the maximum number of iterations this algorithm may perform to reach the given tolerance of the solution
          */
          agx::UInt getMaxNumIterations() const;

          /**
          Assign the maximum number of iterations this algorithm may perform to
          reach the given tolerance of the solution.
          */
          void setMaxNumIterations( agx::UInt maxNumIterations );

          /**
          Ensure that the solver statistics can report data for the given number
          of equations. Any new allocations will be initialized to zero.
          \param numEquations The number of equations to gather statistics for.
          */
          void resizeStats(size_t numEquations);

          /**
          Ensure that the solver statistics can report data for the given number
          of equations. All previously reported statistics are set to zero.
          \param numEquations The number of equations to gather statistics for.
          */
          void clearStats(size_t numEquations);

          /**
          \return The solver statistics for the current solve algorithm.
          */
          const McpAlgorithmStats& getStats() const;

          /**
          \return The solver statistics for the current solve algorithm.
          */
          McpAlgorithmStats& getStats();

          /**
          Solve given current state of the index set, matrix, q, bounds etc.
          \return the solve report
          */
          virtual agx::NlMixedCp::McpAlgorithm::SolveReport solve( agx::NlMixedCp::McpAlgorithm::Args& args ) = 0;


        protected:
          /**
          Switch indices in the index set given current solution and slack.
          \param z - solution
          \param w - slack
          \param bounds - bounded equations
          \param indexSet - Output variable: with index set
          \param tolerance - tolerance of switch condition
          \return number of switched indices
          */
          virtual agx::UInt switchIndices( const agx::RealValarray& z, const agx::RealValarray& w, const agx::SparseRangeReal& bounds, agx::IndexSet& indexSet, agx::Real tolerance = agx::Real( 1E-8 ) ) const;

          /**
          Solve principle problem given sparse matrix, initial solution, bounds and current index set.
          \param m - sparse matrix (support solve and for unit solve)
          \param sparseInverse - sparse inverse with storage of columns of the inverse
          \param z0 - initial, linear solution
          \param bounds - bounded variables
          \param indexSet - current state of the index set
          \param signs - sign convention (-1 for constraints 1 for bodies)
          \param[out] z - solution
          */
          void solvePp( agx::SparseMatrix& m, agx::SparseInverse& sparseInverse, const RealValarray& z0, const agx::SparseRangeReal& bounds, const IndexSet& indexSet, const RealValarray& signs, RealValarray& z ) const;


          /**
          Solve principle problem given sparse matrix, initial solution, bounds and current index set.
          \param m - sparse matrix (support solve and for unit solve)
          \param bounds - bounded variables
          \param indexSet - current state of the index set
          \param signs - sign convention (-1 for constraints 1 for bodies)
          \param q - right hand side
          \param[out] z - solution
          \param workspace Temporary vector used for storage and to avoid reallocations
          */
          void solvePp( agx::SparseMatrix& m, const agx::SparseRangeReal& bounds, const IndexSet& indexSet, const RealValarray& signs,
                        const RealValarray& q, RealValarray& z, RealValarray& workspace  );

          /**
          Computes slack w = M z + q.
          \param m - sparse matrix
          \param z - current solution (NOTE: multiplied with signs before matrix multiply, but and later restored, hence not const)
          \param q - right hand side
          \param signs - sign convention (-1 for constraints 1 for bodies)
          \param[out] slack - output slack
          */
          void getSlack( agx::SparseMatrix& m, agx::RealValarray& z, const agx::RealValarray& q, const agx::RealValarray& signs, agx::RealValarray& slack ) const;

          /**
          Solves: M z = -q for z.
          \param m - sparse matrix
          \param[out] z - solution
          \param q - right hand side
          \param signs - sign convention (-1 for constraints 1 for bodies)
          */
          void linearSolve( agx::SparseMatrix& m, agx::RealValarray& z, const agx::RealValarray& q, const agx::RealValarray& signs ) const;

          /**
          Performs linear solve and stores the solution into args.z0 AND args.z.
          */
          void solveInitial( agx::NlMixedCp::McpAlgorithm::Args& args ) const;

          /**
          Write the elements v[i] for which i is at a bound.
          */
          void forceToBound( RealValarray& v, const agx::SparseRangeReal& bounds, const agx::IndexSet& indexSet ) const;

          /**
          Asserts if z has invalid values.
          */
          void debugCheckSolution( const RealValarray& z, const SparseRangeReal& /*bounds*/ ) const;

        protected:
          agx::Real m_globalTolerance;
          agx::UInt m_maxNumIterations;

          McpAlgorithmStats m_stats;
      };

      typedef agx::ref_ptr< McpAlgorithm > McpAlgorithmRef;

    public:
      /**
      Creates default solver assuming symmetric-, sparse- and lower triangle matrix.
      */
      static NlMixedCpRef create( agx::Solver::NlMcpConfig config, bool gatherStatistics );

    public:
      NlMixedCp( agx::NlMixedCp::McpAlgorithm* mcpAlgorithm );

      /**
      \return the matrix
      */
      agx::SparseMatrix& getMatrix();

      /**
      \return the matrix
      */
      const agx::SparseMatrix& getMatrix() const;

      /**
      \return the negative right hand side q (Hx + q = w)
      */
      agx::RealValarray& getQ();

      /**
      \return the negative right hand side q (Hx + q = w)
      */
      const agx::RealValarray& getQ() const;

      /**
      \return the current solution
      */
      agx::RealValarray& getSolution();

      /**
      \return the current solution
      */
      const agx::RealValarray& getSolution() const;

      /**
      \return sparse bounds object used by the NLMCP
      */
      agx::SparseRangeReal& getSparseBounds();

      /**
      \return sparse bounds object used by the NLMCP
      */
      const agx::SparseRangeReal& getSparseBounds() const;

      /**
      \return the signs, should be -1 for constraints +1 for bodies
      */
      agx::RealValarray& getSigns();

      /**
      \return the signs, should be -1 for constraints +1 for bodies
      */
      const agx::RealValarray& getSigns() const;


      const agx::NlMixedCp::McpAlgorithm* getAlgorithm() const;


      /**
      Set the size of the system. Calling this method will clear all data.
      */
      virtual void setSize( agx::UInt numEquations );

      /**
      Solves the system, returns the total number of iterations made
      in the inner (mcp) loop.
      */
      virtual agx::UInt solve( agx::Bool warm = false,
                               agx::StabilityReport* stabilityReport = nullptr,
                               agx::INlSolveDataH5* nlSolveDataH5 = nullptr );

    protected:
      virtual ~NlMixedCp();

    protected:
      friend class DirectSolverData;
      Solver::NlMcpConfig m_config;
      agx::NlMixedCp::McpAlgorithmRef m_mcp;
      agx::SparseMatrix m_matrix;
      agx::RealValarray m_q;
      agx::RealValarray m_z;
      agx::RealValarray m_z0;
      agx::SparseRangeReal m_sparseBounds;
      agx::RealValarray m_signs;
      agx::IndexSet m_indexSet;
      agx::SparseInverse m_inv;
      agx::UInt m_maxNumOuterIterations;
      agx::Real m_outerGlobalTolerance;
  };

  /**
  Definition of a mixed LCP problem.
  */
  class AGXPHYSICS_EXPORT MixedLCPOld
  {
    public:

      /**
      Construction copies references to the internal objects. If the number
      of maximum iterations is not specified, default values are used.
      */
      MixedLCPOld(SparseMatrix &M, RealValarray &q,
          SparseRangeReal& bounds,  RealValarray& z, const RealValarray& signs, int max_iterations=-1) ;

      ~MixedLCPOld();

      /** Solve the problem . */
      void solve(bool warm = false);

      /** Returns the number of iterations.*/
      int getIterations() const;

      Real getLastResidual() const;

      /** Returns true if the solver computed a correct answer, false * otherwise*/
      bool isSuccess() const;

      SparseRangeReal & bounds() ;
      const SparseRangeReal & bounds() const ;
      RealValarray &  z();
      RealValarray &  q();
      const RealValarray &  z() const;
      const RealValarray &  q() const;

      std::ostream & diag(std::ostream &os, const agx::String & msg = "" ) const ;

    private:
      class MixedLCPRep * m_rep;          // hide the representation.
  };
}

#if defined(_MSC_VER)
#pragma warning( pop ) // restoring: warning( disable : 4127 )
#endif
