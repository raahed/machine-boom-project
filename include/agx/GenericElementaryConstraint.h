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

#include <agx/ElementaryConstraint.h>
#include <agx/GenericConstraintData.h>

namespace agx
{
  /**
  Generic elementary constraint with minimal amount of user callbacks
  to reduce development/prototype time and trivially exported using SWIG
  to e.g., Python or C#. This elementary constraint is only valid for
  one or two body constraints.

  Python example for elementary Dot1:
    class MyDot1(agx.GenericElementaryConstraint):
      def __init__(self, dir1, dir2):
          super().__init__( 1 )
          self.dir1 = dir1
          self.dir2 = dir2

      def onWriteData(self,
                      rb1Data: agx.GenericConstraintBodyData,
                      rb2Data: agx.GenericConstraintBodyData,
                      constraintData: agx.GenericConstraintData):
          a1 = self.getAttachmentPair().getAttachment1() # type: agx.Attachment
          a2 = self.getAttachmentPair().getAttachment2() # type: agx.Attachment
          v1 = a1.get( self.dir1 ) # type: agx.Vec3
          v2 = a2.get( self.dir2 ) # type: agx.Vec3

          rb1Data.setJacobian( 0, agx.Vec3(), v1.cross( v2 ) )
          rb2Data.setJacobian( 0, agx.Vec3(), v2.cross( v1 ) )
          constraintData.setViolation( 0, v1 * v2 )

    myHinge.addElementaryConstraint( "D1_VN", MyDot1( agx.Attachment.V, agx.Attachment.N ) )
    myHinge.addElementaryConstraint( "D1_UN", MyDot1( agx.Attachment.U, agx.Attachment.N ) )
  */
  class AGXPHYSICS_EXPORT GenericElementaryConstraint : public agx::ElementaryConstraint
  {
    public:
      /**
      Construct given the number of rows in this elementary constraint.
      \note Using this constructor the enable flag is set to true.
      \param numRows - number of rows in this elementary constraint
      */
      GenericElementaryConstraint( agx::UInt numRows );

      /**
      Construct given the number of rows in this elementary constraint and default enable flag.
      \param numRows - number of rows in this elementary constraint
      \param enable - true to be enabled by default, false disabled
      */
      GenericElementaryConstraint( agx::UInt numRows, agx::Bool enable );

      /**
      \return the attachment pair in the constraint
      */
      const agx::AttachmentPair* getAttachmentPair() const;

      /**
      \internal

      Enable (or disable) non-linear callback if this is an elementary
      contact constraint where the normal is at row 0 and the tangents
      occupies the rest of the rows.

      Note that this has to be called each time the contact point or
      bodies has changed.
      
      \param enable - true to enable, false to disable (disabled by default)
      \param primaryFrictionCoefficient - primary friction coefficient
      \param secondaryFrictionCoefficient - secondary friction coefficient
      \param contactPointPosition - position of the contact point in world frame
      */
      void setEnableMultiTangentNlCallback( agx::Bool enable,
                                            agx::Real primaryFrictionCoefficient,
                                            agx::Real secondaryFrictionCoefficient,
                                            const agx::Vec3& contactPointPosition );

      /**
      \internal

      \return true if enabled, otherwise false
      */
      agx::Bool getEnableMultiTangentNlCallback() const;

    public:
      /**
      Callback when the elementary constraints in the constraint receives the
      prepare call. The return value defines if this elementary constraint is
      active or not (regardless of the enable flag). When this elementary
      constraint isn't active, onWriteData won't be called.
      \return true if this elementary constraint is active, false if inactive
      */
      virtual agx::Bool onPrepare();

      /**
      Callback when to write Jacobian and constraint data if this elementary
      constraint is active.
      \param rb1Data - constraint body data (Jacobian) for the first rigid body in the constraint
      \param rb2Data - constraint body data (Jacobian) for the second rigid body in the constraint
      \param constraintData - constraint data for this elementary constraint
      */
      virtual void onWriteData( agx::GenericConstraintBodyData* rb1Data,
                                agx::GenericConstraintBodyData* rb2Data,
                                agx::GenericConstraintData* constraintData );

#ifndef SWIG
    public:
      /**
      Performing callbacks and collecting data in serial.
      */
      void fetchData();
#endif

    public:
      void prepare() override;
      agx::Bool isActive() const override;
      agx::UInt getJacobian( agx::Jacobian6DOFElement* G, agx::UInt numBlocks, agx::UInt row, agx::GWriteState::Enum writeState ) override;
      agx::UInt getViolation( agx::Real* g, agx::UInt row ) override;
      agx::UInt getVelocity( agx::Real* v, agx::UInt row ) const override;
      agx::UInt getBounds( agx::RangeReal* bounds, agx::UInt row, agx::Real h ) const override;
      agx::Bool isImpacting() const override;
#ifndef SWIG
      agx::ConstraintNlmcpCallback* getNlCallback() const override;
#endif

      void setAttachmentPair( const agx::AttachmentPair* ap ) override;

    protected:
      virtual ~GenericElementaryConstraint();

    private:
      GenericConstraintBodyData m_rb1Data;
      GenericConstraintBodyData m_rb2Data;
      GenericConstraintData m_constraintData;
      const AttachmentPair* m_attachmentPair;
      Bool m_active;
      ref_ptr<Referenced> m_multiTangentNlCallback;
  };
}
