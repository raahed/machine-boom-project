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


#include <agxWire/Node.h>

#include <agx/ElementaryConstraint.h>

#include <agx/LinearProbingHashTable.h>
#include <agx/QuadraticProbingHashTable.h>

DOXYGEN_START_INTERNAL_BLOCK()

namespace agxWire
{
  struct BendJacobian
  {
    static agx::Quat calculate( const agx::Vec3& p1, const agx::Vec3& p2, const agx::Vec3& p3, agx::Vec3* G1, agx::Vec3* G2 );
  };

  class ElementaryWireDistanceConstraint : public agx::ElementaryConstraintN< 1 >
  {
    public:
#include <HashImplementationSwitcher.h>
#if HASH_FOR_WIRE == HASH_NEW
      using IndexTable = agx::LinearProbingHashTable<const agx::RigidBody*, agx::UInt>;
#elif HASH_FOR_WIRE == HASH_OLD
      typedef agx::QuadraticProbingHashTable< const agx::RigidBody*, agx::UInt > IndexTable;
#else
      #error
#endif

    public:
      ElementaryWireDistanceConstraint();

      /**
      Not used.
      */
      virtual agx::UInt getJacobian( agx::Jacobian6DOFElement* /*G*/, agx::UInt /*numBlocks*/, agx::UInt /*row*/, agx::GWriteState::Enum /*writeState*/ ) override { return 0; }

      /**
      \return number of jacobian rows (Jacobian6DOFElement) used
      */
      agx::UInt getJacobian( agx::Jacobian6DOFElement* G, const IndexTable& bodyToIndexTable, NodeConstIterator begin, NodeConstIterator end );

      /**
      Not used.
      */
      virtual agx::UInt getViolation( agx::Real* /*g*/, agx::UInt /*row*/ ) override { return 1; }

      /**
      The current length is calculated during getJacobin call. Have that in mind before using this method.
      \return the current length of this distance constraint
      */
      inline agx::Real getCurrentLength() const { return m_currentLength; }

    protected:
      virtual ~ElementaryWireDistanceConstraint() {}

    private:
      agx::Real m_currentLength;
  };

  typedef agx::ref_ptr< ElementaryWireDistanceConstraint > ElementaryWireDistanceConstraintRef;

  class ElementaryWireBendConstraint : public agx::ElementaryConstraint
  {
    public:
#include <HashImplementationSwitcher.h>
#if HASH_FOR_WIRE == HASH_NEW
      using IndexTable = agx::LinearProbingHashTable<const agx::RigidBody*, agx::UInt>;
#elif HASH_FOR_WIRE == HASH_OLD
      typedef agx::QuadraticProbingHashTable< const agx::RigidBody*, agx::UInt > IndexTable;
#else
      #error
#endif

    public:
      ElementaryWireBendConstraint( agx::UInt numRows );

      /**
      Push violation to g.
      \param g - constraint value
      \param row - row to write to
      \return the number of rows used by this constraint
      */
      virtual agx::UInt getViolation( agx::Real* g, agx::UInt row ) override = 0;

      /**
      \return the number of jacobian rows used
      */
      virtual agx::UInt getJacobian( agx::Jacobian6DOFElement* G, const Node* first, const Node* second, const Node* third, const IndexTable& bodyToIndexTable ) = 0;

      /**
      Get jacobian method for binary constraints. This is a many body constraint so we can ignore that call.
      */
      virtual agx::UInt getJacobian( agx::Jacobian6DOFElement* /*G*/, agx::UInt /*numBlocks*/, agx::UInt /*row*/, agx::GWriteState::Enum /*writeState*/ ) override { return 0; }

      /**
      Set compliance and damping for this elementary bend constraint.
      \param compliance - compliance
      \param damping - damping
      */
      inline void setComplianceAndDamping( agx::Real compliance, agx::Real damping );

    protected:
      virtual ~ElementaryWireBendConstraint() {}

    protected:
      agx::RegularizationParameters m_regParams[2];
      agx::Real                     m_forces[2];
      agx::RangeReal                m_bounds[2];
  };

  inline void ElementaryWireBendConstraint::setComplianceAndDamping( agx::Real compliance, agx::Real damping )
  {
    getRegularizationParameters()->setCompliance( compliance );
    getRegularizationParameters()->setDamping( damping );
    if ( getNumRows() > 1 ) {
      getRegularizationParameters( 1 )->setCompliance( compliance );
      getRegularizationParameters( 1 )->setDamping( damping );
    }
  }

  /**
  Bend constraints that works on the angle between the three nodes.
  */
  class ElementaryWireBendConstraintAngle : public ElementaryWireBendConstraint
  {
    public:
      ElementaryWireBendConstraintAngle();

      virtual agx::UInt getJacobian( agx::Jacobian6DOFElement* G, const Node* first, const Node* second, const Node* third, const IndexTable& bodyToIndexTable ) override;
      virtual agx::UInt getViolation( agx::Real* g, agx::UInt row ) override;

    protected:
      virtual ~ElementaryWireBendConstraintAngle() {}

    private:
      using ElementaryWireBendConstraint::getJacobian;

    protected:
      agx::Real m_angle;
  };

  typedef agx::ref_ptr< ElementaryWireBendConstraintAngle > ElementaryWireBendConstraintAngleRef;

  class ElementaryQuaternionBendConstraint : public agxWire::ElementaryWireBendConstraint
  {
    public:
      ElementaryQuaternionBendConstraint();

      /**
      Push violation to g.
      \param g - constraint value
      \param row - row to write to
      \return the number of rows used by this constraint
      */
      virtual agx::UInt getViolation( agx::Real* g, agx::UInt row ) override;

      /**
      Used by the wires. Computes the jacobian given three wire nodes.
      \param G - local jacobian matrix
      \param first - first node in this constraint
      \param second - second, middle, node in this constraint
      \param third - third, last, node in this constraint
      \param bodyToIndexTable - used to find the correct element in G
      \return the number of rows used by this constraint
      */
      virtual agx::UInt getJacobian( agx::Jacobian6DOFElement* G, const Node* first, const Node* second, const Node* third, const IndexTable& bodyToIndexTable ) override;

    protected:
      virtual ~ElementaryQuaternionBendConstraint() {}

    private:
      using agxWire::ElementaryWireBendConstraint::getJacobian;

    protected:
      agx::Real m_violation[2];
  };

  typedef agx::ref_ptr< ElementaryQuaternionBendConstraint > ElementaryQuaternionBendConstraintRef;
}

DOXYGEN_END_INTERNAL_BLOCK()