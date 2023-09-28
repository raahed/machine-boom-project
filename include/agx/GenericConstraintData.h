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

#include <agx/agx_vector_types.h>
#include <agx/Logger.h>
#include <agx/Range.h>

namespace agx
{
  /**
  Constraint data used with GenericElementaryConstraint containing
  data such as; constraint violation, constraint velocity and bounds.
  */
  class AGXPHYSICS_EXPORT GenericConstraintData
  {
    public:
      using BoundsVector = agx::VectorPOD<agx::RangeReal>;

    public:
      /**
      Construct given number of rows in the elementary constraint.
      \param numRows - number of rows in the elementary constraint
      */
      GenericConstraintData( agx::UInt numRows );

      /**
      Assign violation for given row in the elementary constraint.
      \param row - elementary constraint row
      \param violation - elementary constraint violation
      */
      void setViolation( agx::UInt row, agx::Real violation );

      /**
      \return the violations
      */
      const RealVector& getViolation() const;

      /**
      Assign velocity for given row in the elementary constraint.
      \param row - elementary constraint row
      \param velocity - elementary constraint velocity
      */
      void setVelocity( agx::UInt row, agx::Real velocity );

      /**
      \return the velocities
      */
      const agx::RealVector& getVelocity() const;

      /**
      Assign bound for given row in the elementary constraint.
      \note The bound given by the user has already been assigned
            when the GenericElementaryConstraint receives its callback.
      \param row - elementary constraint row
      \param range - elementary constraint bound
      */
      void setBound( agx::UInt row, agx::RangeReal range );

      /**
      \return the bounds
      */
      const BoundsVector& getBounds() const;

      /**
      Assign impacting state of the elementary constraint.
      Default: false.
      \param isImpacting - true if the elementary constraint is impacting, otherwise false
      */
      void setIsImpacting( agx::Bool isImpacting );

      /**
      \return true if the elementary constraint is impacting, otherwise false
      */
      agx::Bool getIsImpacting() const;

      /**
      Resets all values to default.
      */
      void reset();

    private:
      agx::RealVector m_violation;
      agx::RealVector m_velocity;
      BoundsVector m_bounds;
      agx::Bool m_isImpacting;
  };

  /**
  Constraint body data used with GenericElementaryConstraint containing
  the per body Jacobian data.
  */
  class AGXPHYSICS_EXPORT GenericConstraintBodyData
  {
    public:
      /**
      Construct given number of rows in the elementary constraint.
      \param numRows - number of rows in the elementary constraint
      */
      GenericConstraintBodyData( agx::UInt numRows );

      /**
      Assign Jacobian for given row as spatial and rotation vectors.
      \param row - row in the Jacobian
      \param spatial - spatial part of the Jacobian
      \param rotational - rotational part of the Jacobain (ignored if particle)
      */
      void setJacobian( agx::UInt row, const agx::Vec3& spatial, const agx::Vec3& rotational );

      /**
      \return the Jacobian data where index 2 * i is spatial part and 2 * i + 1 the rotational
      */
      const agx::Vec3Vector& getJacobian() const;

      /**
      Resets all vectors to (0, 0, 0).
      */
      void reset();

    private:
      agx::Vec3Vector m_jacobian;
  };
}
