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

#ifndef AGX_SOLVER_H
#define AGX_SOLVER_H

#include <agx/Task.h>
#include <agx/Kernel.h>
// #include <agx/DirectSolverData.h>

#include <agx/Physics/SolveGroupEntity.h>
#include <agx/Physics/SolveIslandEntity.h>
#include <agx/Physics/HierarchicalGrid/ContactZoneEntity.h>
#include <agx/Physics/HierarchicalGrid/ContactZoneDependencyEntity.h>
#include <agx/Physics/GeometryContactEntity.h>
#include <agx/Physics/ParticleGeometryContactEntity.h>
#include <agx/Physics/BinaryConstraintEntity.h>
#include <agx/Physics/ManyBodyConstraintEntity.h>
#include <agx/agx_vector_types.h>
#include <agxStream/Serializable.h>

namespace agxSDK
{
  class Simulation;
}

#ifdef _MSC_VER
# pragma warning(push)
# pragma warning(disable:4355) // Disable warnings about passing the 'this' pointer to base class constructors.
# pragma warning( disable : 4251 ) // class X needs to have dll-interface to be used by clients of class Y
#endif


namespace agx
{

  AGX_DECLARE_POINTER_TYPES(SolveKernel);
  AGX_DECLARE_VECTOR_TYPES(SolveKernel);
  AGX_DECLARE_POINTER_TYPES(Solver);
  AGX_DECLARE_POINTER_TYPES(MobilitySolver);
  AGX_DECLARE_VECTOR_TYPES(MobilitySolver);

  class SolveJob;
  class DirectSolverData;
  class INlSolveDataH5;

  /**
  Solver interface.
  */
  class CALLABLE AGXPHYSICS_EXPORT Solver : public TaskGroup, public virtual agxStream::Serializable
  {
    public:
      static agx::Model* ClassModel();

      enum JobTag
      {
        ITERATION = 0x1,
        STORE_CONTACT_FORCES = 0x2,
        RHS_UPDATE = 0x3,
        IMPACT_ITERATION = 0x4,
        DIRECT_SOLVE = 0x5,
        SYNCHRONIZE_ITERATIVE = 0x6,

        JOB_TYPE_MASK = 0xFF,
        ITERATION_INDEX_SHIFT = 8
      };

      DOXYGEN_START_INTERNAL_BLOCK()

      static UInt32 getInteractionTag(const String& name);
      static const String& getTagName(UInt32 id);
      DOXYGEN_END_INTERNAL_BLOCK()

    public:
      enum McpAlgoritmType
      {
        BLOCK_PIVOT,
        KELLER,
        HYBRID_PIVOT,
        NUM_MCP_ALGORITHMS
      };

      /**
      Configuration structure for the solver.
      */
      struct AGXPHYSICS_EXPORT NlMcpConfig
      {
        /**
        Create default configuration settings for the solver.
        */
        static agx::Solver::NlMcpConfig createDefault();

        agx::Solver::McpAlgoritmType mcpAlgorithmType; /**< Solve type algorithm. */
        agx::UInt numMcpIterations;                    /**< Mixed-CP iterations, relevant for block pivot (i.e., not Keller's). */
        agx::UInt numOuterIterations;                  /**< Non-linear loop max iterations. */
        agx::Real mcpTolerance;                        /**< Tolerance of the Mixed-CP section, definition could differ between algorithms. */
        agx::Real outerTolerance;                      /**< Tolerance of the non-linear loop. */

        bool indexSetWarmStarting;                     /**< If information from previous timestep should be used in LCP solver */

        private:
          // Use createDefault() instead.
          NlMcpConfig();
      };

      /**
      \return short description of the MCP algorithm
      */
      static agx::String getDescription( agx::Solver::McpAlgoritmType mcpAlgorithm );

    public:
      Solver();

      /**
      \return Number of resting iterations.
      */
      UInt getNumRestingIterations() const;

      /**
      Set number of resting iterations.
      */
      void setNumRestingIterations(UInt numIterations);

      /**
      \return number of Parallel-Projected-Gauss-Seidel (PPGS) resting iterations, if it is used in the solver.
      */
      UInt getNumPPGSRestingIterations() const;

      /**
      Set number of Parallel-Projected-Gauss-Seidel (PPGS) resting iterations, if it is used in the solver.
      */
      void setNumPPGSRestingIterations(UInt numPPGSIterations);

      /**
      \return Number of dry friction iterations.
      */
      UInt getNumDryFrictionIterations() const;

      /**
      Set number of dry friction iterations.
      */
      void setNumDryFrictionIterations(UInt numIterations);

      /**
      \note this is only used when using the complex impact stage:  agxSDK::Simulation::setUseComplexImpactStage( bool);
      \returns the number of iterations used for solving impact stage.
      */
      UInt getNumImpactIterations() const;

      /**
      Specifies the number of iterations used for solving impact stage.
      \note this is only used when using the complex impact stage:  agxSDK::Simulation::setUseComplexImpactStage( bool);
      */
      void setNumImpactIterations(UInt numIterations);

      /**
      \return true if force iterative is enabled
      */
      bool isForceIterative() const;

      /**
      Set to true to force iterative solve, bypassing the direct solver.
      */
      void setForceIterative(bool flag);

      /**
      \return the configuration of the direct solver
      */
      agx::Solver::NlMcpConfig& getNlMcpConfig();

      /**
      \return the configuration of the direct solver
      */
      agx::Solver::NlMcpConfig getNlMcpConfig() const;

      /**
      Assign Mixed Complementarity Problem (MCP) algorithm type for this solver.
      \param mcpType - new type
      */
      void setMcpAlgorithmType( agx::Solver::McpAlgoritmType mcpType );

      /**
      \return the MCP algorithm type used for this solver
      */
      agx::Solver::McpAlgoritmType getMcpAlgorithmType() const;

      /**
      Specify the threshold for which a speed (during overlap) lower than this value will be treated as a resting contact.
      */
      void setImpactSpeedThreshold(agx::Real threshold);

      /**
      \return the current speed threshold when a contact should be treated as a resting contact.
      */
      Real getImpactSpeedThreshold() const;

      /**
      \return true if stability report is enabled
      */
      agx::Bool isStabilityReportEnable() const;

      /**
      Enable or disable stability reporting.
      */
      void setStabilityReportEnable( agx::Bool enable );


      void setSolverStatiticsEnable(bool enable);
      bool getSolverStatisticsEnable() const;

      /**
      Enable/disable the use of Parallel GS solver for GranularBodies.
      This will speed up simulation of large systems with the number of threads that AGX get to use.
      \param flag - If true, a parallel version of GS will be used for Granular bodies.
      */
      void setUseParallelPgs(bool flag);

      /**
      \return true if Parallel GS solver is being used.
      */
      bool useParallelPgs() const;

      /**
      Enable/disable warm starting of lambdas in granular body pipeline
      */
      void setUseGranularWarmStarting(bool flag);

      /**
      \return true if warm starting of lambdas is used in the granular body pipeline
      */
      bool useGranularWarmStarting() const;

      /**
      \return true if the GranularBody solver run in 32bit floating point precision.
      */
      bool use32bitGranularBodySolver() const;


      /**
      Specify if granular bodies should be solved using 32bit floating point precision.
      NOTE: It is only a valid setting when the agx library is built using 64bit floating
      point precision.
      \param flag - If true, 32bit solver will be used
      */
      void setUse32bitGranularBodySolver(bool flag);


      /**
      Enable/disable if iteration residuals should be calculated.
      \param flag - If true enable calculation
      */
      void setCalculateIterationResiduals(bool flag);

      /**
      \return true if iteration residuals are calculated.
      */
      bool getCalculateIterationResiduals() const;

      /**
      \return The iteration residuals, one element per iteration.
      */
      const agxData::Buffer* getIterationResiduals() const;

      /**
      \return if residual calculation is enabled, the iteration residuals, one element per iteration.
      */
      agx::RealVector getIterationResidualsVector() const;

      /**
      \return if residual calculation is enabled, return the global residual for the PPGS solver
      */
      agx::Real getGlobalRelativeResidual() const;

      /**
      Enable non-linear solver HDF5 dumping given filename. If the file exists,
      the data will be appended.
      \param filename - HDF5 filename
      */
      void enableNlSolveDataH5( const agx::String& filename );

      /**
      Disable HDF5 dump in the non-linear solver.
      */
      void disableNlSolveDataH5();

      /**
      \return true if HDF5 dump of non-linear solver data is enabled - otherwise false
      */
      agx::Bool getEnableNlSolveDataH5() const;

      DOXYGEN_START_INTERNAL_BLOCK()
      agx::INlSolveDataH5* getNlSolveDataH5() const;

      MobilitySolver* getMobilitySolver();
      MobilitySolver* getMobilityReferenceSolver();
      const SolveKernelRefVector& getKernels() const;

      SolveKernel* getStoreContactForcesKernel();
      SolveKernel* getCommitDirectVelocitiesKernel();

      SolveKernel* getPrepareDirectSolverKernel();
      SolveKernel* getDirectRhsKernel();

      SolveKernel* getDirectSolverKernel();
      SolveKernel* getSynchronizeIterativeSolverKernel();
      SolveKernel* getUpdateBoundsSolverKernel();
      SolveKernel* getSynchronizeDirectSolverKernel();

      SolveKernel* getUpdateIterationCountKernel();

      void addKernel(SolveKernel* kernel);
      void removeKernel(SolveKernel* kernel);


      void generateSolveJobs(
        agx::Physics::SolveIslandData& solveIsland,
        agx::Physics::SolveGroupData& solveGroup,
        agxSDK::Simulation* simulation,
        agx::Physics::HierarchicalGrid::ContactZoneData& zone,
        agx::Physics::HierarchicalGrid::ContactZoneDependencyData& zoneDependency,
        agx::Physics::HierarchicalGrid::ContactZonePtr& iterationStartZone,
        agx::Physics::HierarchicalGrid::ContactZonePtr& iterationEndZone,
        UInt numTiers
      );

      agx::Physics::SolveIslandData& getIslands();
      bool hasIslands() const;

      DirectSolverData* getSharedSolverData(Physics::SolveIslandPtr island);
      void registerDirectSolverData(Physics::SolveIslandPtr island, DirectSolverData* data);
      void clearDirectSolverData();
      DirectSolverData* getDirectSolverData(Physics::SolveIslandPtr island);

      AGXSTREAM_DECLARE_SERIALIZABLE(agx::Solver);

      void setSimulation(agxSDK::Simulation* simulation);
      agxSDK::Simulation* getSimulation();

      Task* getGranularVelocityIntegrationKernel();

      void reset();

      void setEnableMobilitySolverDataSharing(bool flag);
      bool getEnableMobilitySolverDataSharing() const;

      DOXYGEN_END_INTERNAL_BLOCK()

    protected:
      virtual ~Solver();

    private:
      friend class agxSDK::Simulation;
      void applyCustomParticlePipeline();

    private:
      void prepareKernels();
      void commitKernels();
      void updateImpactSpeedThreshold();

    public:
      DOXYGEN_START_INTERNAL_BLOCK()

      struct ParallelPGS
      {
        /////////////////////////////////////
        class IterationStart : public Job
        {
        public:
          IterationStart();
          virtual ~IterationStart();

          void reset();
          Timer& getTimer();

          virtual void implementation() override;

        private:
          UInt m_currentIteration;
          Timer m_timer;
        };

        /////////////////////////////////////
        class IterationEnd : public Job
        {
        public:
          IterationEnd(Solver* solver, Physics::SolveIslandPtr island, IterationStart* iterationStart);
          virtual ~IterationEnd();

          void setNumIterations(UInt numIterations);
          void setExit(Job* exitJob);

          virtual void implementation() override;

          void printResidual();
          void verifyIteration();
          void verifyGranularBodyConstraints(const Path& path);
          Real calculateComputeCost();

        private:
          Solver* m_solver;
          Physics::SolveIslandPtr m_island;
          UInt m_numIterations;
          UInt m_currentIteration;
          // UInt64 m_totNumTicks;
          IterationStart* m_iterationStart;
          Job* m_exitJob;
        };

        /////////////////////////////////////
        class JobGraphInstaller : public Job
        {
        public:
          JobGraphInstaller(Job* startJob, Job* endJob);
          virtual ~JobGraphInstaller();
          virtual void implementation();

          Job exit;

        private:
          Job* m_startJob;
          Job* m_endJob;
        };

        class IterationLoopInstaller : public Job
        {
        public:
          IterationLoopInstaller(Solver* solver, IterationStart* startJob, IterationEnd* endJob, UInt numIterations);
          virtual ~IterationLoopInstaller();
          virtual void implementation();

          Job exit;

        private:
          void initIterationCounters(const Path& path);
          void initIterationCounters();

        private:
          Solver* m_solver;
          IterationStart* m_startJob;
          IterationEnd* m_endJob;
          UInt m_numIterations;
        };

        /////////////////////////////////////
        AGX_DECLARE_POINTER_TYPES(JobGraph);
        class JobGraph : public Referenced
        {
        public:
          Job* rhsStart;
          Job* rhsEnd;

          Job* storeContactForcesStart;
          Job* storeContactForcesEnd;

          IterationStart* iterationStart;
          IterationEnd* iterationEnd;
          JobGroup* serialIterationJob;

          Job* residualStart;
          Job* residualEnd;

          Solver* solver;

        public:

          JobGraph(Solver* solver_, agx::Physics::SolveIslandPtr island);
          void init(agx::Physics::SolveIslandPtr island);


          void createZoneIndependentJobs(
            agx::Physics::HierarchicalGrid::ContactZoneData& zone,
            agx::Physics::SolveIslandPtr solveIsland,
            JobGroup*& groupIt,
            SolveJob*& jobIt,
            SolveKernel* particleParticleKernel,
            SolveKernel* particleGeometryKernel,
            SolveKernel* geometryGeometryKernel,
            Job* preJob,
            Job* postJob
          );

          void createZoneRhsJobs(
            agx::Physics::HierarchicalGrid::ContactZoneData& zone,
            agx::Physics::SolveIslandPtr solveIsland,
            JobGroup*& groupIt,
            SolveJob*& jobIt,
            SolveKernel* particleParticleKernel,
            SolveKernel* particleGeometryKernel,
            SolveKernel* geometryGeometryKernel
          );

          void createZoneStoreContactForcesJobs(
            agx::Physics::HierarchicalGrid::ContactZoneData& zone,
            agx::Physics::SolveIslandPtr solveIsland,
            JobGroup*& groupIt,
            SolveJob*& jobIt,
            SolveKernel* particleParticleKernel,
            SolveKernel* particleGeometryKernel,
            SolveKernel* geometryGeometryKernel
          );

          void createZoneResidualJobs(
            agx::Physics::HierarchicalGrid::ContactZoneData& zone,
            agx::Physics::SolveIslandPtr solveIsland,
            JobGroup*& groupIt,
            SolveJob*& jobIt,
            SolveKernel* particleParticleKernel,
            SolveKernel* particleGeometryKernel,
            SolveKernel* geometryGeometryKernel
          );


          void createIslandRhsJobs(
            agx::Physics::SolveIslandPtr solveIsland,
            SolveJob*& jobIt
          );


          void createIslandStoreContactForcesJobs(
            agx::Physics::SolveIslandPtr solveIsland,
            SolveJob*& jobIt
          );

          void createZoneIterationJobs(
            agx::Physics::HierarchicalGrid::ContactZoneData& zone,
            agx::Physics::HierarchicalGrid::ContactZoneDependencyData& zoneDependency,
            agx::Physics::SolveIslandPtr solveIsland,
            JobGroup*& groupIt,
            SolveJob*& jobIt,
            UInt numTiers,
            SolveKernel* particleParticleKernel,
            SolveKernel* particleGeometryKernel,
            SolveKernel* geometryGeometryKernel
          );

          void createIslandIterationJobs(
            agx::Physics::SolveIslandPtr solveIsland,
            SolveJob *& jobIt
          );


        protected:
          virtual ~JobGraph();

        private:
          agxData::IndexArray m_geometryGeometryContactConstraints;
          agxData::IndexArray m_particleGeometryContactConstraints;
          agxData::IndexArray m_binaryConstraints;
          agxData::IndexArray m_manyBodyConstraints;
          agxData::IndexArray m_directBodies;
          agxData::BufferRef m_indexListBuffer;
          Physics::SolveGroupPtr m_solveGroupPtrs[4];
        };

      };

      /////////////////////////////////////




      ParallelPGS::JobGraph* generateParallelPgsJobGraph(
        agx::Physics::SolveIslandPtr solveIsland,
        agxSDK::Simulation* simulation,
        agx::Physics::HierarchicalGrid::ContactZoneData& zone,
        agx::Physics::HierarchicalGrid::ContactZoneDependencyData& zoneDependency,
        agx::Physics::HierarchicalGrid::ContactZonePtr& iterationStartZone,
        agx::Physics::HierarchicalGrid::ContactZonePtr& iterationEndZone,
        agx::UInt numTiers
      );

      ParallelPGS::JobGraph* getParallelPgsJobGraph();
      void addJobGraphInstaller(Job* loopInstaller);


    private:

      struct SolveJobContext
      {
        agx::Physics::SolveIslandData* solveIsland;
        agx::Physics::SolveGroupData* solveGroup;
        agxSDK::Simulation* simulation;
        agx::Physics::HierarchicalGrid::ContactZoneData* zone;
        agx::Physics::HierarchicalGrid::ContactZoneDependencyData* zoneDependency;
        agx::Physics::HierarchicalGrid::ContactZonePtr* iterationStartZone;
        agx::Physics::HierarchicalGrid::ContactZonePtr* iterationEndZone;
        agx::UInt numTiers;
      };

      SolveJobContext m_solveJobContext;

    private:
      SolveKernelRefVector m_kernels;

      agxData::ValueRefT<UInt> m_numImpactIterations;
      agxData::ValueRefT<UInt> m_numRestingIterations;
      agxData::ValueRefT<UInt> m_numPPGSRestingIterations;
      agxData::ValueRefT<UInt> m_numDryFrictionIterations;
      agxData::ValueRefT<Bool> m_forceIterative;
      agxData::ValueRefT<Real> m_impactSpeedThreshold;
      agxData::ValueRefT<Bool> m_stabilityReportEnable;
      bool m_impactSpeedExplicitlySet;

      NlMcpConfig m_nlMcpConfig;

      agxData::BufferRef m_iterationResiduals;
      bool m_calculateIterationResiduals;

      SolveKernelRef m_storeContactForcesKernel;
      SolveKernelRef m_commitDirectVelocitiesKernel;

      SolveKernelRef m_directRhsKernel;

      SolveKernelRef m_prepareDirectSolverKernel;
      SolveKernelRef m_directSolverKernel;
      SolveKernelRef m_synchronizeIterativeSolverKernel;
      SolveKernelRef m_updateDirectBounds;
      SolveKernelRef m_synchronizeDirectSolverKernel;
      SolveKernelRef m_iterationCountKernel;

      HashSet<Job* > m_visitedJobs;
      bool m_useParallelPGS;
      bool m_use32bitGranularSolver;
      bool m_useGranularWarmStarting;
      bool m_gatherStatistics;
      bool m_enableMobilitySolverDataSharing;

      ParallelPGS::JobGraphRef m_parallelPgsJobGraph;
      agx::Vector<Job* > m_jobGraphInstallers;

      MobilitySolverRefVector m_mobilitySolvers;
      UInt m_numActiveMobilitySolvers;

      agxData::BufferRef m_solveJobBuffer;
      agxData::BufferRef m_solveWrapperJobBuffer;

      agxData::BufferRef m_pgsSolveJobBuffer;
      agxData::BufferRef m_pgsSolveWrapperJobBuffer;

      typedef agx::Vector<ref_ptr<Referenced> > DirectSolverDataTable;
      DirectSolverDataTable m_directSolverDataTable;

      agxSDK::Simulation* m_simulation;

      agx::TaskRef m_granularIntegrateVelocitiesKernel;
      JobPtrVector m_jobStack;
      agx::Physics::SolveIslandData* m_islands;

      ref_ptr<Referenced> m_nlSolveDataH5;
  };


  //---------------------------------------------------------------

  class AGXPHYSICS_EXPORT MobilitySolver : public Solver
  {
  public:
    MobilitySolver(Solver* masterSolver, bool isReference);

    Solver* getMasterSolver();
    bool isReferenceSolver() const;

    void addKernelCopy(SolveKernel* kernel);
    void removeKernelCopy(SolveKernel* kernel);

    SolveKernel* getMobilityKernel(SolveKernel* kernel);

    void prepare();
    void verify();
    void copyToMaster();
    void copyFromMaster();

    agxData::EntityStorage* getRigidBodyStorage();

  protected:
    virtual ~MobilitySolver();

  private:
    void copy(bool import);

    void preSolve(Task* task);
    void postSolve(Task* task);
    agxData::EntityStorage* setupRigidBodyStorage();
    agxData::EntityStorage* setupConstraintRowStorage(const Path& storagePath);

    void setupSynchronizedBuffer(agxData::EntityStorage* masterStorage, agxData::EntityStorage* localStorage, agxData::Attribute* attribute);

    struct BufferSynchronization
    {
      agxData::EntityStorageRef masterStorage;
      agxData::EntityStorageRef localStorage;
      agxData::AttributeRef attribute;
    };

  private:
    Solver* m_masterSolver;
    bool m_isReference;
    agxData::EntityStorageRef m_rigidBodyStorage;
    agxData::EntityStorageRef m_contactConstraintRowStorage;
    agxData::EntityStorageRef m_binaryConstraintRowStorage;
    agxData::EntityStorageRef m_manyBodyConstraintRowStorage;
    TaskRef m_preSolveTask;
    TaskRef m_postSolveTask;
    Task::ExecutionEvent::CallbackType m_preSolveCallback;
    Task::ExecutionEvent::CallbackType m_postSolveCallback;

    typedef HashTable<SolveKernel *, SolveKernel *> KernelTable;
    KernelTable m_kernelTable;

    Vector<BufferSynchronization> m_bufferSynchronizations;
  };


  //---------------------------------------------------------------

  AGX_DECLARE_POINTER_TYPES(SolveModel);
  AGX_DECLARE_VECTOR_TYPES(SolveModel);
  typedef HashTable<String, SolveModel* > SolveModelTable;

  class SolveModel : public Object
  {
    public:
      enum SolveType
      {
        INVALID_SOLVE_TYPE = -1,
        ITERATIVE = (1<<0),
        DIRECT_AND_ITERATIVE = (1<<1),
        DIRECT = (1<<2),
        SPLIT = (1<<3)
      };

      enum ConstraintType
      {
        INVALID_CONSTRAINT_TYPE = -1,
        CONTACT_CONSTRAINT,
        BINARY_CONSTRAINT,
        MANY_BODY_CONSTRAINT,
        STRONG_INTERACTION
      };

      static agx::Model* ClassModel();
      static SolveModel::SolveType toSolveType( const agx::String& str );
      static SolveModel::ConstraintType toConstraintType( const agx::String& str );
      static SolveModel* load(TiXmlElement* eModel, agx::Device* device);
      void configure(TiXmlElement* eModel);

    public:
      SolveModel(const Name& name, SolveType solveType, ConstraintType constraintType);

      bool isDirect() const;
      SolveType getSolveType() const;
      ConstraintType getConstraintType() const;
      agx::UInt32 getId() const;

      SolveKernel* getPrepareDirectSolveKernel();
      SolveKernel* getIterativeSolveKernel();
      SolveKernel* getInternalZoneBeginKernel();
      SolveKernel* getInternalZoneEndKernel();
      SolveKernel* getIntegrateVelocitiesKernel();
      SolveKernel* getUpdateRhsKernel();
      SolveKernel* getResidualKernel();
      SolveKernel* getSynchronizeIterativeKernel();
      SolveKernel* getSynchronizeDirectKernel();
      SolveKernel* getStoreContactForcesKernel();

      void setPrepareDirectSolveKernel(SolveKernel* kernel);
      void setIterativeSolveKernel(SolveKernel* kernel);
      void setIntegrateVelocitiesKernel(SolveKernel* kernel);
      void setUpdateRhsKernel(SolveKernel* kernel);
      void setResidualKernel(SolveKernel* kernel);
      void setSynchronizeIterativeKernel(SolveKernel* kernel);
      void setSynchronizeDirectKernel(SolveKernel* kernel);


      const SolveKernelRefVector& getKernels() const;

      void setId(agx::UInt32 id);
    protected:
      virtual ~SolveModel();

    private:
      UInt32 m_id;
      SolveType m_solveType;
      ConstraintType m_constraintType;
      SolveKernelRef m_prepareDirectSolverKernel;
      SolveKernelRef m_iterativeSolveKernel;
      SolveKernelRef m_internalZoneBeginKernel;
      SolveKernelRef m_internalZoneEndKernel;
      SolveKernelRef m_integrateVelocitiesKernel;
      SolveKernelRef m_updateRhsKernel;
      SolveKernelRef m_residualKernel;
      SolveKernelRef m_storeContactForcesKernel;
      SolveKernelRef m_synchronizeIterativeKernel;
      SolveKernelRef m_synchronizeDirectKernel;
      SolveKernelRefVector m_kernels;
  };

  AGXPHYSICS_EXPORT Physics::SolveGroupPtr getSolveGroup(Physics::SolveIslandPtr island, const String& name);

  //---------------------------------------------------------------

  struct AGXPHYSICS_EXPORT SolveContext
  {
    typedef agx::Physics::HierarchicalGrid::ContactZoneInstance ContactZoneInstance;

    SolveContext();
    SolveContext(
      agxData::IndexArray interactions_, Physics::SolveIslandPtr island_, Physics::SolveGroupPtr group_,
      ContactZoneInstance contactZone_ = ContactZoneInstance());


    agxData::IndexArray interactions;
    Physics::SolveIslandPtr island;
    Physics::SolveGroupPtr group;
    ContactZoneInstance contactZone;
  };

  //---------------------------------------------------------------

  class AGXPHYSICS_EXPORT SolveJob : public Job
  {
  public:
    typedef Callback1<SolveContext> SolveCallback;

  public:
    SolveJob();
    SolveJob(const SolveCallback& callback, SolveContext context, agx::UInt32 cost);
    virtual ~SolveJob();

    void init(const SolveCallback& callback, SolveContext context, agx::UInt32 cost);

    SolveContext& getContext();

  protected:
    virtual void implementation() override;

  private:
    // void dispatch();

  protected:
    SolveCallback m_solveCallback;
    SolveContext m_context;
  };

  //---------------------------------------------------------------

  template <class T>
  class SolveJobT : public SolveJob
  {
  public:
    SolveJobT();
    SolveJobT(SolveContext context, agx::UInt32 cost);
    virtual ~SolveJobT();

  protected:
    virtual void implementation() override;

  };

  //---------------------------------------------------------------

  class AGXPHYSICS_EXPORT SolveKernel : public CpuKernel
  {
  public:
    typedef SolveJob::SolveCallback SolveDispatch;

  public:
    SolveKernel(const Name& name, SolveDispatch dispatch);

    const SolveDispatch& getDispatch() const;

    virtual void constructJob(SolveJob* job) = 0;
  protected:
    virtual ~SolveKernel();

  private:
    SolveDispatch m_dispatch;
  };

  DOXYGEN_END_INTERNAL_BLOCK()

  //---------------------------------------------------------------

  /* Implementation */
  AGX_FORCE_INLINE UInt Solver::getNumImpactIterations() const { return m_numImpactIterations->get(); }
  AGX_FORCE_INLINE UInt Solver::getNumRestingIterations() const { return m_numRestingIterations->get(); }
  AGX_FORCE_INLINE UInt Solver::getNumPPGSRestingIterations() const { return m_numPPGSRestingIterations->get(); }
  AGX_FORCE_INLINE UInt Solver::getNumDryFrictionIterations() const { return m_numDryFrictionIterations->get(); }
  AGX_FORCE_INLINE bool Solver::isForceIterative() const { return m_forceIterative->get(); }
  AGX_FORCE_INLINE const SolveKernelRefVector& Solver::getKernels() const { return m_kernels; }

  AGX_FORCE_INLINE SolveKernel* Solver::getStoreContactForcesKernel() { return m_storeContactForcesKernel; }
  AGX_FORCE_INLINE SolveKernel* Solver::getCommitDirectVelocitiesKernel() { return m_commitDirectVelocitiesKernel; }

  AGX_FORCE_INLINE SolveKernel* Solver::getDirectRhsKernel() { return m_directRhsKernel; }


  AGX_FORCE_INLINE SolveKernel* Solver::getPrepareDirectSolverKernel() { return m_prepareDirectSolverKernel; }
  AGX_FORCE_INLINE SolveKernel* Solver::getDirectSolverKernel() { return m_directSolverKernel; }
  AGX_FORCE_INLINE SolveKernel* Solver::getSynchronizeIterativeSolverKernel() { return m_synchronizeIterativeSolverKernel; }
  AGX_FORCE_INLINE SolveKernel* Solver::getUpdateBoundsSolverKernel() { return m_updateDirectBounds; }
  AGX_FORCE_INLINE SolveKernel* Solver::getSynchronizeDirectSolverKernel() { return m_synchronizeDirectSolverKernel; }
  AGX_FORCE_INLINE SolveKernel* Solver::getUpdateIterationCountKernel() { return m_iterationCountKernel; }

  AGX_FORCE_INLINE DirectSolverData* Solver::getDirectSolverData(Physics::SolveIslandPtr island)
  {
    return (DirectSolverData *)m_directSolverDataTable[island.getId()].get();
  }


  //-----------------------------------------------------------------------------------------------------

  AGX_FORCE_INLINE bool SolveModel::isDirect() const { return m_solveType > ITERATIVE; }
  AGX_FORCE_INLINE SolveModel::SolveType SolveModel::getSolveType() const { return m_solveType; }
  AGX_FORCE_INLINE SolveModel::ConstraintType SolveModel::getConstraintType() const { return m_constraintType; }
  AGX_FORCE_INLINE agx::UInt32 SolveModel::getId() const { return m_id; }
  AGX_FORCE_INLINE SolveKernel* SolveModel::getPrepareDirectSolveKernel() { return m_prepareDirectSolverKernel; }
  AGX_FORCE_INLINE SolveKernel* SolveModel::getIterativeSolveKernel() { return m_iterativeSolveKernel; }
  AGX_FORCE_INLINE SolveKernel* SolveModel::getInternalZoneBeginKernel() { return m_internalZoneBeginKernel; }
  AGX_FORCE_INLINE SolveKernel* SolveModel::getInternalZoneEndKernel() { return m_internalZoneEndKernel; }
  AGX_FORCE_INLINE SolveKernel* SolveModel::getIntegrateVelocitiesKernel() { return m_integrateVelocitiesKernel; }
  AGX_FORCE_INLINE SolveKernel* SolveModel::getUpdateRhsKernel() { return m_updateRhsKernel; }
  AGX_FORCE_INLINE SolveKernel* SolveModel::getResidualKernel() { return m_residualKernel; }
  AGX_FORCE_INLINE SolveKernel* SolveModel::getStoreContactForcesKernel() { return m_storeContactForcesKernel; }
  AGX_FORCE_INLINE SolveKernel* SolveModel::getSynchronizeIterativeKernel() { return m_synchronizeIterativeKernel; }
  AGX_FORCE_INLINE SolveKernel* SolveModel::getSynchronizeDirectKernel() { return m_synchronizeDirectKernel; }

  AGX_FORCE_INLINE const SolveKernelRefVector& SolveModel::getKernels() const { return m_kernels; }

  //-----------------------------------------------------------------------------------------------------

  AGX_FORCE_INLINE SolveContext::SolveContext() {}
  AGX_FORCE_INLINE SolveContext::SolveContext(
    agxData::IndexArray interactions_,
    Physics::SolveIslandPtr island_,
    Physics::SolveGroupPtr group_,
    ContactZoneInstance contactZone_)
    : interactions(interactions_)
    , island(island_)
    , group(group_)
    , contactZone(contactZone_)
  {}

  //-----------------------------------------------------------------------------------------------------

  AGX_FORCE_INLINE SolveJob::SolveJob() // : Job(Callback(&SolveJob::dispatch, this))
  {
  }

  AGX_FORCE_INLINE SolveJob::SolveJob(const SolveCallback& callback, SolveContext context, agx::UInt32 cost) // : Job(Callback(&SolveJob::dispatch, this))
  {
    this->init(callback, context, cost);
  }

  AGX_FORCE_INLINE void SolveJob::init(const SolveCallback& callback, SolveContext context, agx::UInt32 cost)
  {
    m_context = context;
    m_solveCallback = callback;

    // Job::init(Callback(&SolveJob::dispatch, this), cost);
    Job::init(Job::Callback(), cost);
  }


  AGX_FORCE_INLINE SolveContext& SolveJob::getContext() { return m_context; }

  //-----------------------------------------------------------------------------------------------------

  template <class T>
  SolveJobT<T>::SolveJobT() : SolveJob()
  {
  }

  template <class T>
  SolveJobT<T>::SolveJobT(SolveContext context, agx::UInt32 cost) : SolveJob(SolveCallback(), context, cost)
  {
  }

  template <class T>
  SolveJobT<T>::~SolveJobT()
  {
  }

  template <class T>
  void SolveJobT<T>::implementation()
  {
    agxAssert(dynamic_cast<T*>(m_task));
    static_cast<T*>(m_task)->dispatch(m_context);
  }


  AGX_FORCE_INLINE const SolveKernel::SolveDispatch& SolveKernel::getDispatch() const { return m_dispatch; }
}

AGX_TYPE_BINDING(agx::SolveJob, "SolveJob")

#ifdef _MSC_VER
# pragma warning(pop)
#endif

#endif /* AGX_SOLVER_H */
