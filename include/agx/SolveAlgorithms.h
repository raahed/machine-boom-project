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

#ifndef AGX_SOLVEALGORITHMS_H
#define AGX_SOLVEALGORITHMS_H

#include <agx/Vector.h>

#include <agx/Physics/SolveGroupEntity.h>

#include <agxData/Array.h>


#ifdef _MSC_VER
# pragma warning( push )
# pragma warning( disable : 4251 ) // class X needs to have dll-interface to be used by clients of class Y
#endif

namespace agx
{
  // Forward declarations agx.
  class Solver;
  class SolveKernel;
  class SolveJob;
  class JobGroup;

  // Forward declarations agx::Physics.
  namespace Physics
  {
    class SolveGroupPtr;
    class SolveIslandPtr;
    class SolveGroupData;
  }

  // Typedefs.
  typedef agx::VectorPOD< agx::JobGroup* > JobGroupPtrVector;

  /**
  Utility class used to generate jobs for ALL islands.
  */
  class AGXPHYSICS_EXPORT JobGenerator
  {
    public:
      /**
      Construct generator given allocated arrays of group- and solve jobs.
      \param wrapperJobs - group jobs
      \param solveJobs - solve jobs for kernels
      */
      JobGenerator( agxData::Array< agx::JobGroup >& wrapperJobs, agxData::Array< agx::SolveJob >& solveJobs );

      /**
      Initializes, increments internal counter and returns a new group/wrapper job.
      \param solver - general solver (this group is task for the solver)
      */
      agx::JobGroup* getNewWrapperJob( agx::Solver* solver );

      /**
      Increments internal counter and returns a new solve job.
      */
      agx::SolveJob* getNewSolveJob();

    protected:
      agxData::Array< agx::JobGroup > m_wrapperJobs;
      agxData::Array< agx::SolveJob > m_solveJobs;
      agx::UInt m_wrapperJobCounter;
      agx::UInt m_solveJobCounter;
  };

  /**
  Parent class for solve algorithms compatible with the constraint framework.
  The general solve algorithm is a tree structure of groups and solve jobs,
  where all children to the root are groups and dependent (serial) on each
  other.
  */
  class AGXPHYSICS_EXPORT SolveAlgorithm : public agx::Referenced
  {
    public:
      /**
      Solve algorithm type "guessed" by findSolveAlgorithm.
      */
      enum Type {
                  EMPTY,                    /**< Solver for islands with zero models. */
                  PURE_DIRECT,              /**< Only direct constraints. */
                  PURE_ITERATIVE,           /**< Only iterative constraints. */
                  HYBRID                    /**< Combination of direct and iterative constraints. */
                };

    public:
      /**
      Algorithm task either empty, single or group type. The type of
      this task determines which callback (getSolveKernel) is called.
      EMPTY: Includes no jobs but has children. No callback.
      EXIT: Includes no jobs and any children and/or dependency will be ignored. This is an exit node. No callback.
      SINGLE: Includes one task (given agx::Solver) and receives getSolveKernel( agx::Solver* ) callback.
      GROUP: Includes several tasks (given SolveGroupPtr) and receives getSolveKernel( agx::Physics::SolveGroupPtr ) callback.
      */
      class AGXPHYSICS_EXPORT Task : public agx::Referenced
      {
        public:
          /**
          Task type either EMPTY branch (no callback), EXIT branch (no callback, traversal of this branch will end),
          SINGLE (getSolveKernel given agx::Solver callback) or GROUP (getSolveKernel given SolveGroupPtr callback).
          */
          enum Type { EMPTY, EXIT, SINGLE, GROUP, CUSTOM };

        public:
          /**
          The type of tasks may depend on the data in this structure.
          */
          struct ConditionData
          {
            agx::Bool impacting;
            agx::Bool dryFrictionIterations;
            agx::Bool useParallelPgs;
            agx::Bool calculateResiduals;
          };

        public:
          /**
          Default constructor, type set to EMPTY.
          */
          Task();

          /**
          Add child task.
          \param child - child task to add
          */
          void addChild( agx::SolveAlgorithm::Task* child );

          /**
          \param i - index of child
          \return child with index \p i
          */
          agx::SolveAlgorithm::Task* getChild( agx::UInt i ) const;

          /**
          \return the number of children this parent class has
          */
          agx::UInt getNumChildren() const;

          /**
          \return parent task this this task
          */
          agx::SolveAlgorithm::Task* getParent() const;

          /**
          \return the type of this solve algorithm task
          */
          virtual agx::SolveAlgorithm::Task::Type getType( const agx::SolveAlgorithm::Task::ConditionData& conditionData ) const;

          /**
          Callback for solve algorithms task type SINGLE.
          \param solver - solver with tasks
          \return the solve kernel for this task
          */
          virtual agx::SolveKernel* getSolveKernel( agx::Solver* solver ) const;

          /**
          Callback for solve algorithms task type GROUP. One call for each solve group.
          \param group - current solve group
          \return the solve kernel for this group
          */
          virtual agx::SolveKernel* getSolveKernel( agx::Physics::SolveGroupPtr group ) const;

          virtual void customSetup(Solver *solver, JobGroup *entry, JobGroup *exit);

        protected:
          /**
          Construct given type.
          \param type - type of this task
          */
          Task( agx::SolveAlgorithm::Task::Type type );

          /**
          Reference counted object - protected virtual destructor.
          */
          virtual ~Task();

        protected:
          typedef agx::Vector< agx::ref_ptr< SolveAlgorithm::Task > > SolveAlgorithmTaskContainer;

        protected:
          agx::SolveAlgorithm::Task::SolveAlgorithmTaskContainer m_children;
          agx::SolveAlgorithm::Task*                       m_parent;
          agx::SolveAlgorithm::Task::Type                  m_type;
      };

      typedef agx::ref_ptr< SolveAlgorithm::Task > TaskRef;

    public:
      /**
      Static method to find the optimal solve algorithm for an island.
      \param island - island to find solve algorithm for
      \param numZones - Number of zones
      \return suggested, and compatible, type of the solve algorithm to use
      */
      static agx::SolveAlgorithm::Type findSolveAlgorithm( agx::Physics::SolveIslandPtr island, UInt numZones );

    public:
      /**
      Initializes a solve algorithm and/or returns the root task.
      \note This method has to be called for a solve algorithm to be created
      \return the root task
      */
      virtual agx::SolveAlgorithm::Task* instantiate();

      /**
      After create method, this recursive method can be called to
      pre-calculate the number of types in the tree for a given island.
      \param task - current parent task
      \param conditionData - branch condition data
      \param numEmptyTasks - increment the number of empty tasks \p task contains
      \param numSingleTasks - increment the number of single tasks \p task contains
      \param numGroupTasks - increment the number of group tasks \p task contains
      \param numCustomTasks - increment the number of custom tasks \p task contains
      */
      void calculateNumTasks( const agx::SolveAlgorithm::Task* task, const agx::SolveAlgorithm::Task::ConditionData& conditionData, agx::UInt& numEmptyTasks, agx::UInt& numSingleTasks, agx::UInt& numGroupTasks, agx::UInt& numCustomTasks ) const;

      /**
      Generate jobs given the tree structure built by the implemented solve algorithm.
      \param generator - pool of groups and jobs
      \param conditionData - specific condition data for current island
      \param island - the island
      \param solveGroup - solve group data
      */
      virtual void generateJobs( agx::JobGenerator& generator, const SolveAlgorithm::Task::ConditionData& conditionData,  agx::Physics::SolveIslandPtr island, const agx::Physics::SolveGroupData& solveGroup );

    protected:
      /**
      Protected constructor for pure virtual object.
      */
      SolveAlgorithm(Solver *solver);

      /**
      Reference counted object, protected destructor.
      */
      virtual ~SolveAlgorithm();

      /**
      Create method called one time during first call to
      getOrCreateRoot. During this call the solve algorithm
      should build the task tree.
      \return the root task
      */
      virtual agx::SolveAlgorithm::Task* create(bool useComplexImpactStage) = 0;

      /**
      Recursive part of generateJobs.
      \param task - current parent task
      \param conditionData - branch condition data
      \param parentJobGroup - the job group of the parent of to \p task
      \param dependentJobs - list of jobs that are dependent (only children to root)
      \param generator - job pool
      \param island - pointer to the current island
      \param solveGroup - solve group data
      */
      virtual void generateJobs( agx::SolveAlgorithm::Task* task, const agx::SolveAlgorithm::Task::ConditionData& conditionData, agx::JobGroup* parentJobGroup, agx::JobGroupPtrVector& dependentJobs, agx::JobGenerator& generator, agx::Physics::SolveIslandPtr island, const agx::Physics::SolveGroupData& solveGroup );

      /**
      \return the solver
      */
      agx::Solver* getSolver() { return m_solver; }

    protected:
      agx::SolveAlgorithm::TaskRef m_root;
      agx::Solver* m_solver;
  };

  typedef agx::ref_ptr< SolveAlgorithm > SolveAlgorithmRef;

  /**
  Algorithm for an island without solve models.
  */
  class AGXPHYSICS_EXPORT EmptySolver : public SolveAlgorithm
  {
    public:
      /**
      Construct empty solver.
      */
      EmptySolver(Solver *solver);

    protected:
      /**
      Reference counted. Protected destructor.
      */
      virtual ~EmptySolver();

      /**
      Creates tree structure for an empty solver.
      */
      virtual SolveAlgorithm::Task* create(bool useComplexImpactStage) override;
  };

  /**
  Algorithm for pure iterative solve. I.e., it is not possible to couple
  other types of solvers with this solver.
  */
  class AGXPHYSICS_EXPORT PureIterativeSolver : public SolveAlgorithm
  {
    public:
      /**
      Construct pure iterative solver.
      */
      PureIterativeSolver( Solver *solver );

    protected:
      /**
      Reference counted. Protected destructor.
      */
      virtual ~PureIterativeSolver();

      /**
      Creates tree structure of this type of solver.
      \return the root task of a pure iterative solver
      */
      virtual SolveAlgorithm::Task* create(bool useComplexImpactStage) override;

    protected:
      agx::UInt m_numImpactIterations;
      agx::UInt m_numRestingIterations;
      agx::UInt m_numPPGSIterations;
  };

  /**
  Algorithm for pure direct solve. I.e., it is not possible to couple
  other types of solvers with this solver.
  */
  class AGXPHYSICS_EXPORT PureDirectSolver : public SolveAlgorithm
  {
    public:
      /**
      Construct pure direct solver.
      */
      PureDirectSolver(Solver *solver);

    protected:
      /**
      Reference counted. Protected destructor.
      */
      virtual ~PureDirectSolver();

      /**
      Creates tree structure of this type of solver.
      \return the root task of a pure direct solver
      */
      virtual SolveAlgorithm::Task* create(bool useComplexImpactStage) override;
  };

  /**
  Algorithm for hybrid solve. I.e., a combination of direct and
  iterative solves etc..
  */
  class AGXPHYSICS_EXPORT HybridSolver : public SolveAlgorithm
  {
    public:
      /**
      Construct a hybrid solve algorithm based on a previous solver (use same settings)
      */
      HybridSolver( agx::Solver *solver );

    protected:
      /**
      Reference counted. Protected destructor.
      */
      virtual ~HybridSolver();

      /**
      Creates tree structure of this type of solver.
      \return the root task of a hybrid solver
      */
      virtual SolveAlgorithm::Task* create(bool useComplexImpactStage) override;

    protected:
      agx::UInt m_numImpactIterations;
      agx::UInt m_numRestingIterations;
      agx::UInt m_numDryFrictionIterations;
      agx::UInt m_numPPGSIterations;
  };
}

#ifdef _MSC_VER
# pragma warning( pop )
#endif

#endif
