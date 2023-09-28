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

#ifndef AGX_STABILITYREPORT_H
#define AGX_STABILITYREPORT_H

#include <agx/SparseTypes.h>
#include <agx/agx_valarray_types.h>

#include <agx/Physics/BinaryConstraintEntity.h>
#include <agx/Physics/ManyBodyConstraintEntity.h>
#include <agx/Physics/ContactConstraintEntity.h>
#include <agx/Physics/SolveIslandEntity.h>

namespace agx
{
  class Solver;
  class DirectSolverData;
  class ElementaryConstraint;

  AGX_DECLARE_POINTER_TYPES( StabilityReport );

  class AGXPHYSICS_EXPORT StabilityReport : public Referenced
  {
    public:
      class AGXPHYSICS_EXPORT Diff
      {
        public:
          typedef agx::VectorPOD< Diff > Container;

        public:
          Diff();
          Diff( const agx::RealValarray& solution,
                agx::RangeReal range,
                agx::UInt permutedRowIndex );

          agx::Bool isValid() const { return m_permutedRowIndex != agx::InvalidIndex; }
          agx::Real getSolution() const { return m_solution; }
          agx::RangeReal getBound() const { return m_bound; }
          agx::Real getDiff() const { return m_diff; }
          agx::UInt getPermutedRowIndex() const { return m_permutedRowIndex; }

          bool operator < ( const Diff& other ) const { return m_diff < other.m_diff; }

        private:
          Real      m_solution;
          RangeReal m_bound;
          Real      m_diff;
          UInt      m_permutedRowIndex;
      };

      class AGXPHYSICS_EXPORT Frame
      {
        public:
          typedef agx::Vector< Frame > Container;

        public:
          Frame( const agx::RealValarray& solution, const agx::SparseRangeReal& bounds, const agx::String& description );

          const Diff::Container& getDiffs() const;

        private:
          void initialize( const agx::RealValarray& solution, const agx::SparseRangeReal& bounds, const agx::String& description );

        private:
          Diff::Container m_diffs;
          agx::String m_description;
      };

      class AGXPHYSICS_EXPORT PotentialContactIssue
      {
        public:
          typedef agx::Vector<PotentialContactIssue> Container;

        public:
          PotentialContactIssue( const agx::StabilityReport::Diff& diff,
                                 const agx::Physics::ContactConstraintPtr contact,
                                 const agx::UInt permutedStartRow )
            : diff( diff )
            , contact( contact )
            , permutedStartRow( permutedStartRow )
          {
          }

          const agx::StabilityReport::Diff diff;
          const agx::Physics::ContactConstraintPtr contact;
          const agx::UInt permutedStartRow;
      };

      class AGXPHYSICS_EXPORT PotentialConstraintIssue
      {
        public:
          typedef agx::Vector<PotentialConstraintIssue> Container;

        public:
          PotentialConstraintIssue( const agx::StabilityReport::Diff& diff,
                                    const agx::ConstraintImplementation* constraint,
                                    const agx::Physics::ConstraintBasePtr constraintEntity,
                                    const agx::ElementaryConstraint* elementaryConstraint,
                                    const agx::UInt permutedStartRow )
            : diff( diff )
            , constraint( constraint )
            , constraintEntity( constraintEntity )
            , elementaryConstraint( elementaryConstraint )
            , permutedStartRow( permutedStartRow )
          {
          }

          const agx::StabilityReport::Diff diff;
          const agx::ConstraintImplementation* constraint;
          const agx::Physics::ConstraintBasePtr constraintEntity;
          const agx::ElementaryConstraint* elementaryConstraint;
          const agx::UInt permutedStartRow;
      };

    public:
      StabilityReport();

      void setEnable( agx::Bool enable );
      agx::Bool getEnable() const;

      agx::Bool empty() const;

      const agx::StabilityReport::PotentialConstraintIssue::Container& getPotentialConstraintIssues() const;
      const agx::StabilityReport::PotentialContactIssue::Container& getPotentialContactIssues() const;

      virtual void report( const agx::RealValarray& z,
                           const agx::SparseRangeReal& bounds,
                           const agx::String& description );
      virtual void finalize( agx::Solver *solver,
                             const agx::Physics::SolveIslandPtr island,
                             const agx::Physics::ContactConstraintData& contactConstraint,
                             const agx::Physics::BinaryConstraintData& binaryConstraint,
                             const agx::Physics::ManyBodyConstraintData& manyBodyConstraint );

    protected:
      virtual ~StabilityReport();

    protected:
      agx::Bool m_enable;
      Frame::Container m_frames;
      PotentialConstraintIssue::Container m_potentialConstraintIssues;
      PotentialContactIssue::Container m_potentialContactIssues;
  };
}

#endif
