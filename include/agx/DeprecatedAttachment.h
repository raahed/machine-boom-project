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

#ifndef AGX_DEPRECATEDATTACHMENT_H
#define AGX_DEPRECATEDATTACHMENT_H

#include <agxStream/Serializable.h>

#include <agx/Vec3.h>

#ifdef _MSC_VER
# pragma warning( push )
# pragma warning( disable : 4251 ) // class X needs to have dll-interface to be used by clients of class Y
#endif

namespace agx
{
  DOXYGEN_START_INTERNAL_BLOCK()
  /**
  This class is only present to support files stored before version 2.4.1.1.
  */
  class AGXPHYSICS_EXPORT OldConstraintAttachmentSkeleton : public virtual agxStream::Serializable
  {
    public:
      OldConstraintAttachmentSkeleton() : m_data( 0 ), m_size(0) {}
      OldConstraintAttachmentSkeleton( Vec3* data, size_t n ) : m_data( data ), m_size( n ) {}
      virtual ~OldConstraintAttachmentSkeleton() {}

      AGXSTREAM_DECLARE_SERIALIZABLE(agx::OldConstraintAttachmentSkeleton);

    protected:
      Vec3*  m_data;
      size_t m_size;
  };

  class AGXPHYSICS_EXPORT HingeAttachment : public OldConstraintAttachmentSkeleton
  {
    public:
      HingeAttachment() : OldConstraintAttachmentSkeleton( m_data, 9 ) {}

    protected:
      AGXSTREAM_DECLARE_SERIALIZABLE(agx::HingeAttachment);

      Vec3 m_data[ 9 ];
      OldConstraintAttachmentSkeleton m_xformed[ 1 ];
  };

  class AGXPHYSICS_EXPORT PrismaticAttachment : public HingeAttachment { public: AGXSTREAM_DECLARE_SERIALIZABLE(agx::PrismaticAttachment); };
  class AGXPHYSICS_EXPORT CylindricalJointAttachment : public HingeAttachment { public: AGXSTREAM_DECLARE_SERIALIZABLE(agx::CylindricalJointAttachment); };
  class AGXPHYSICS_EXPORT LockJointAttachment : public HingeAttachment { public: AGXSTREAM_DECLARE_SERIALIZABLE(agx::LockJointAttachment); };
  class AGXPHYSICS_EXPORT BallJointAttachment : public OldConstraintAttachmentSkeleton
  {
    public:
      BallJointAttachment() : OldConstraintAttachmentSkeleton( m_data, 3 ) {}
      AGXSTREAM_DECLARE_SERIALIZABLE(agx::BallJointAttachment);
    protected:
      Vec3 m_data[ 3 ];
      OldConstraintAttachmentSkeleton m_xformed[ 1 ];
  };

  class AGXPHYSICS_EXPORT DistanceJointAttachment : public OldConstraintAttachmentSkeleton
  {
    public:
      DistanceJointAttachment() : OldConstraintAttachmentSkeleton( m_data, 3 ) {}
      AGXSTREAM_DECLARE_SERIALIZABLE(agx::DistanceJointAttachment);
    protected:
      Vec3 m_data[ 3 ];
      OldConstraintAttachmentSkeleton m_xformed[ 1 ];
  };

  class AGXPHYSICS_EXPORT AngularLockJointAttachment : public OldConstraintAttachmentSkeleton
  {
    public:
      AngularLockJointAttachment() : OldConstraintAttachmentSkeleton( m_data, 6 ) {}
      AGXSTREAM_DECLARE_SERIALIZABLE(agx::AngularLockJointAttachment);
    protected:
      Vec3 m_data[ 6 ];
      OldConstraintAttachmentSkeleton m_xformed[ 1 ];
  };

  class AGXPHYSICS_EXPORT Dot1JointAttachment : public AngularLockJointAttachment
  {
    public:
      AGXSTREAM_DECLARE_SERIALIZABLE(agx::Dot1JointAttachment);
  };
  DOXYGEN_END_INTERNAL_BLOCK()
} // namespace agx

#ifdef _MSC_VER
#pragma warning( pop )
#endif

#endif
