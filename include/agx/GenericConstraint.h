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

#include <agx/Constraint.h>
#include <agx/GenericConstraintData.h>

namespace agx
{
  class GenericConstraintComplianceMatrix;

  /**
  Generic constraint which by default is empty of elementary constraints.
  This enables the possibility to build custom/generic constraints with
  additional GenericElementaryConstraint instances or to use the already
  implemented in the toolkit.

  This constraint receives; onAddNotification, onRemoveNotification, onPreSolve and onPostSolve.

  Python example:
    class MyHinge(agx.GenericConstraint):
        def __init__(self, rb1: agx.RigidBody, rb1Frame: agx.Frame, rb2: agx.RigidBody, rb2Frame: agx.Frame):
            super().__init__( rb1, rb1Frame, rb2, rb2Frame )

            self.addElementaryConstraint( "SP", agx.SphericalRel( agx.ElementaryConstraintData( self.getAttachmentPair() ) ) )
            self.addElementaryConstraint( "D1VN", MyDot1( agx.Attachment.V, agx.Attachment.N ) )
            self.addElementaryConstraint( "D1UN", MyDot1( agx.Attachment.U, agx.Attachment.N ) )

        def onAddNotification(self):
            print( 'add notification!' )

        def onRemoveNotification(self):
            print( 'remove notification!' )

        def onPreSolve(self):
            return True

  */
  class AGXPHYSICS_EXPORT GenericConstraint : public agx::Constraint
  {
    public:
      /**
      Construct given one rigid body (second is null).
      \note Two frames will be created and added to the AttachmentPair of this constraint.
            The transform of the constraint frames is identity.
      \param rb1 - first rigid body
      */
      GenericConstraint( agx::RigidBody* rb1 );

      /**
      Construct given two rigid bodies (first has to be != null).
      \note Two frames will be created and added to the AttachmentPair of this constraint.
            The transform of the constraint frames is identity.
      \param rb1 - first rigid body
      \param rb2 - second rigid body
      */
      GenericConstraint( agx::RigidBody* rb1, agx::RigidBody* rb2 );

      /**
      Construct given two bodies and frames.
      \param rb1 - first rigid body
      \param f1 - constraint frame relative to rb1
      \param rb2 - second rigid body
      \param f2 - constraint frame relative to rb2
      */
      GenericConstraint( agx::RigidBody* rb1, agx::Frame* f1, agx::RigidBody* rb2, agx::Frame* f2 );

      /**
      Enable/disable updateComplianceMatrix callback with write access
      to the elements in the compliance matrix of this constraint. If
      \p isStaticSize is true, meaning the number of active rows in this
      constraint is constant, the matrix is accessible and wont reset
      to zero each time step.
      \note This is only supported by the direct solver.
      \param enable - true to enable, false to disable
      \param isStaticSize - true if this constraints active rows is constant, e.g., LockJoint,
                            enabling access to the compliance matrix in runtime and the
                            matrix isn't reset to zero each time step.
      */
      void setSupportComplianceMatrix( agx::Bool enable, agx::Bool isStaticSize = false );

      /**
      If this constraint active rows doesn't vary and isStaticSize = true has
      been flagged when enabling support for compliance matrix, this method
      will return the matrix at any time. Otherwise nullptr is returned.
      \return compliance matrix if accessible, otherwise nullptr
      */
      GenericConstraintComplianceMatrix* getComplianceMatrix() const;

      /**
      Assign/update rigid bodies of this constraint. Note that \p rb1 must be
      valid for this constraint to be valid.
      \param rb1 - first rigid body
      \param rb2 - second rigid body
      \return true if this constraint is valid after the bodies has been assigned, otherwise false
      */
      agx::Bool setRigidBodies( agx::RigidBody* rb1, agx::RigidBody* rb2 );

    public:
      /**
      Callback when the constraint has been added to a simulation.
      */
      virtual void onAddNotification();

      /**
      Callback when the constraint has been removed from a simulation.
      */
      virtual void onRemoveNotification();

      /**
      Callback before this constraint reaches the solver (actually preSystemCallback). If the
      return value is true and this constraint is valid, this constraint will continue into
      the solver.
      \return true if this constraint is active, false if inactive
      */
      virtual agx::Bool onPreSolve();

      /**
      Callback from after the solve and the solutions been pushed to the elementary constraints.
      */
      virtual void onPostSolve();

      /**
      Callback to write data. All values must be greater than or equal to zero. Data is
      only stored for the lower triangle since the matrix must be symmetric and positive
      definite.
      */
      virtual void updateComplianceMatrix( GenericConstraintComplianceMatrix* matrix );

    public:
#ifndef SWIG
      static void onPreSolveCallback( const agxSDK::Simulation* simulation );
      static void onPostSolveCallback( const agxSDK::Simulation* simulation );
      static void onCleanup( const agxSDK::Simulation* simulation );
#endif

    public:
      int getNumDOF() const override;
      void render( class agxRender::RenderManager* manager, float scale ) const override;

    protected:
      virtual ~GenericConstraint();

#ifndef SWIG
      virtual void addNotification() override;
      virtual void removeNotification() override;
#endif
  };

  /**
  Compliance matrix data containing the lower triangle (including the diagonal)
  of the compliance matrix of the constraint. This matrix can be written to by
  calling self.setSupportComplianceMatrix( True ) and implement:

  def updateComplianceMatrix(self, matrix: agx.GenericConstraintComplianceMatrix):
      matrix.set(1, 0, 1.0E-4)
  */
  class AGXPHYSICS_EXPORT GenericConstraintComplianceMatrix : public agx::Referenced
  {
    public:
      /**
      Construct given maximum size, i.e., when all elementary constraint
      rows are active and enabled, and if the size is constant, i.e., all
      elementary constraints are always active and enabled.
      */
      GenericConstraintComplianceMatrix( UInt32 maxSize, Bool isStaticSize );

      /**
      \param row - matrix row number
      \param col - matrix column number
      \return value at given row and column
      */
      Real get( UInt32 row, UInt32 col ) const;

      /**
      Assign value to given row and column in the matrix.
      \param row - matrix row number
      \param col - matrix column number
      */
      void set( UInt32 row, UInt32 col, Real value );

      /**
      \return the size, i.e., N in this N x N matrix
      */
      UInt32 size() const;

      /**
      Reset this matrix to zero and resize. If this matrix is of static
      size and \p size != current size, an exception is thrown.
      \param size - new size of this matrix
      */
      void reset( UInt32 size );

      /**
      \return vector of diagonal values that has been assigned
      */
      const UInt32Vector& getDiagonalUpdates() const;

      /**
      \return true if this matrix has a static size, otherwise false
      */
      Bool isStaticSize() const;

    protected:
      /**
      Reference counted object, protected destructor.
      */
      virtual ~GenericConstraintComplianceMatrix();

    private:
      /**
      Calculates index in m_data given matrix row and column.
      */
      UInt32 calculateIndex( UInt32 row, UInt32 col ) const;

      /**
      \return array storage size
      */
      UInt32 arraySize() const;

    private:
      RealVector m_data;
      UInt32 m_size;
      UInt32 m_maxSize;
      Bool m_isStaticSize;
      UInt32Vector m_diagonalUpdates;
  };

  using GenericConstraintComplianceMatrixRef = agx::ref_ptr<GenericConstraintComplianceMatrix>;
}
