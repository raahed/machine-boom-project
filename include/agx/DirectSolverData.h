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

#ifndef AGX_DIRECTSOLVERDATA_H
#define AGX_DIRECTSOLVERDATA_H

#include <agx/MixedLCP.h>
#include <agx/Solver.h>
#include <agx/RigidBodyState.h>
#include <agx/StabilityReport.h>
#include <agx/agx_vector_types.h>

#include <agx/Physics/SolveIslandEntity.h>
#include <agx/Physics/StrongInteractionEntity.h>

#include <agxData/Array.h>

#define to_permuted_matrix_index( lId ) matrix.getBlockPermutedStartRowIndex( lId )

namespace agx
{
  class LogicalStructure
  {
    public:
      typedef std::valarray< unsigned int > UIntContainer;
    public:
      LogicalStructure() : m_riCounter( 0 ), m_riIndex( 0 ), m_ciIndex( 0 ), m_dimIndex( 0 ) {}

      void resize( size_t riSize, size_t ciSize, size_t dimSize )
      {
        m_ri.resize( riSize, UIntContainer::value_type(-1) );
        m_ci.resize( ciSize, UIntContainer::value_type(-1) );
        m_dim.resize( dimSize, UIntContainer::value_type(-1) );
        m_riCounter = 0;
        m_riIndex = 0;
        m_ciIndex = 0;
        m_dimIndex = 0;
      }

      inline void newRow( size_t dim )
      {
        agxAssert( m_riIndex < m_ri.size() );
        m_ri[ m_riIndex ] = (UIntContainer::value_type)m_riCounter;
        m_dim[ m_dimIndex ] = (UIntContainer::value_type)dim;
        ++m_riIndex;
        ++m_dimIndex;
      }

      inline void insertBlock( size_t colIndex )
      {
        agxAssert( m_ciIndex < m_ci.size() );
        m_ci[ m_ciIndex ] = (UIntContainer::value_type)colIndex;
        ++m_ciIndex;
        ++m_riCounter;
      }

      void finalize()
      {
        agxAssert( m_riIndex == m_ri.size() - 1 );
        agxAssert( m_ciIndex == m_ci.size() );
        m_ri[ m_riIndex ] = (UIntContainer::value_type)m_riCounter;
      }

      unsigned int* riPtr()  { return &( m_ri[0]); }
      unsigned int* ciPtr()  { return &( m_ci[0]); }
      unsigned int* dimPtr() { return &(m_dim[0]); }

      bool operator==(const LogicalStructure& rhs) const;
    private:
      UIntContainer m_ri;         /**< Block row index in packed format */
      UIntContainer m_ci;         /**< Block column index in packed format */
      UIntContainer m_dim;        /**< Dimension of each block */
      size_t        m_riCounter;
      size_t        m_riIndex;
      size_t        m_ciIndex;
      size_t        m_dimIndex;
  };

  class AGXPHYSICS_EXPORT DirectSolverData : public Referenced
  {
    public:
      typedef agx::HashTable< agx::UInt, agx::UIntVector > StrongInteractionTable;

    public:
      /**
      Class that handles the grouping.
      */
      class AGXPHYSICS_EXPORT ContactGroups : public agx::Referenced
      {
        public:
          typedef agx::VectorPOD< agx::Index > GroupsContainer;
          enum Buffers { SPLIT, DIRECT, DIRECT_AND_ITERATIVE, NUM_BUFFERS };

          ContactGroups() {}

          /**
          \return true if there are contacts going to the direct solver - otherwise false
          */
          bool initialize( Physics::SolveIslandPtr& island );

          AGX_FORCE_INLINE const GroupsContainer& getBuffer( Buffers buffer ) const { return m_buffers[ buffer ]; }

        protected:
          virtual ~ContactGroups() {}

        private:
          ContactGroups( const ContactGroups& ) {}

        private:
          GroupsContainer m_buffers[ NUM_BUFFERS ];
      };

      typedef agx::ref_ptr< ContactGroups > ContactGroupsRef;

    public:
      /**
      Construct given total number of equations in the direct system, number of block rows
      and the total number of blocks in the matrix.
      \param numEquations - total number of equations
      \param numBlockRows - total number of block rows in matrix
      \param numBlocks - total number of blocks in matrix
      \param nlMcpConfig - direct solver configuration
      */
      DirectSolverData( agx::UInt numEquations,
                        agx::UInt numBlockRows,
                        agx::UInt numBlocks,
                        agx::Solver::NlMcpConfig nlMcpConfig,
                        agx::Bool gatherStatistics,
                        INlSolveDataH5* nlSolveDataH5 );

      /**
      Initialize for a new solve. This method can be called many times, e.g., if this object is stored
      between time steps.
      */
      void initialize();

      /**
      Clones solver data.
      */
      DirectSolverData* clone();

      /**
      Builds table that maps a rigid body to the other bodies it interacts with.
      \param strongInteractions - strong interactions
      \param globalToLocalIndexArray - index array that maps global body indices to island indices
      */
      void buildStrongInteractionTable( agx::Physics::InteractionGroupPtr islandGroup,
                                        agx::Physics::StrongInteractionData& strongInteractions,
                                        const agxData::IndexArray& globalToLocalIndexArray );

      /**
      \return the logical structure of the matrix (used to permute the matrix)
      */
      agx::LogicalStructure& getLogicalStructure();
      const agx::LogicalStructure& getLogicalStructure() const;

      /**
      \return the matrix
      */
      agx::SparseMatrix& getMatrix();
      const agx::SparseMatrix& getMatrix() const;

      /**
      \return the total number of block rows
      */
      agx::UInt getNumBlockRows() const;

      /**
      \return the signs, should be -1 for constraints +1 for bodies
      */
      agx::RealValarray& getSigns();
      const agx::RealValarray& getSigns() const;

      /**
      \return sparse bounds object used by the NLMCP
      */
      agx::SparseRangeReal& getSparseBounds();
      const agx::SparseRangeReal& getSparseBounds() const;

      /**
      Solve the current configuration of the matrix, right hand side and bounds. The
      solution is stored and is accessible through \p getSolution().
      \return the solution of the solve
      */
      const agx::RealValarray& solve();

      /**
      \return the negative right hand side q (Hx + q = w)
      */
      agx::RealValarray& getQ();
      const agx::RealValarray& getQ() const;

      /**
      \return the current solution
      */
      agx::RealValarray& getSolution();
      const agx::RealValarray& getSolution() const;

      /**
      \return the current IndexSet
      */
      agx::IndexSet& getIndexSet();

      /**
      \return the contact grouping, if present (assigned externally)
      */
      agx::DirectSolverData::ContactGroups* getContactGroups() const;

      /**
      \return The McpAlgorithm used by this solver.
      */
      const agx::NlMixedCp::McpAlgorithm* getAlgorithm() const;

      /**
      \return the stability report
      */
      agx::StabilityReport* getStabilityReport() const;

      /**
      \return the strong interaction table that maps body to other strong interaction bodies (on the same row)
      */
      const agx::DirectSolverData::StrongInteractionTable& getStrongInteractionTable() const;

      /**
      Assign new contact groups object.
      */
      void setContactGroups( agx::DirectSolverData::ContactGroups* contactGroups );

      /**
      Copy solution to velocity buffers.
      \param indices - direct body local to global indices
      \param linearVelocities - linear velocity buffer
      \param angularVelocities - angular velocity buffer
      \param states - states
      */
      void commitVelocities( const agxData::IndexArray& indices,
                             agxData::Array< agx::Vec3 >& linearVelocities,
                             agxData::Array< agx::Vec3 >& angularVelocities,
                             const agxData::Array< agx::RigidBodyState >& states ) const;

    protected:
      DirectSolverData() = delete;
      virtual ~DirectSolverData();


    protected:
      UInt                   m_numEquations;
      UInt                   m_numBlockRows;
      UInt                   m_numBlocks;
      NlMixedCpRef           m_nlmcp;
      LogicalStructure       m_logicalStructure;
      ContactGroupsRef       m_contactGroups;
      UInt                   m_numSolves;
      StabilityReportRef     m_stabilityReport;
      StrongInteractionTable m_strongInteractionTable;
      INlSolveDataH5*        m_nlSolveDataH5;
  };

  AGX_DECLARE_POINTER_TYPES( DirectSolverData );
}

#endif
