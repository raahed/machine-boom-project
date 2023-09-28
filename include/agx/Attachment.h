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

#ifndef AGX_ATTACHMENT_H
#define AGX_ATTACHMENT_H

#include <agx/ConstraintAngle.h>
#include <agx/RigidBody.h>

#ifdef _MSC_VER
# pragma warning( push )
# pragma warning( disable : 4251 ) // class X needs to have dll-interface to be used by clients of class Y
# pragma warning(disable: 6385) // Disable warning C6385: Reading invalid data
#endif


namespace agx
{
  /**
  Constraint attachment base class for any type of frame. Subclass for
  moving frames (e.g., rigid bodies).
  */
  class AGXPHYSICS_EXPORT Attachment : public agx::Referenced, public agxStream::Serializable
  {
    public:
      /**
      Data access enum.
      */
      enum Transformed
      {
        U,                  /**< Constraint attachment x-axis. */
        V,                  /**< Constraint attachment y-axis. */
        N,                  /**< Constraint attachment z-axis. */
        ANCHOR_POS,         /**< Constraint attachment world position. */
        CM_TO_ANCHOR_POS,   /**< Vector from center of mass (if moving) to anchor point given in world frame. */
        NUM_ELEMENTS        /**< Number of data elements. */
      };

    public:
      /**
      Given vector \p N, create orthonormal base.
      \param N - in vector (a.k.a. z axis)
      \param U - out vector (a.k.a. x axis)
      \param V - out vector (a.k.a. y axis)
      */
      static void createAttachmentBase( const agx::Vec3d& N, agx::Vec3d& U, agx::Vec3d& V );
      static void createAttachmentBase( const agx::Vec3f& N, agx::Vec3f& U, agx::Vec3f& V );

    public:
      /**
      Construct given frame.
      \param relFrame - that transforms from center to anchor point.
      */
      Attachment( agx::Frame* relFrame );

      /**
      Subclass - implement this method!

      Default this method will copy data from relative frame to
      transformed data.
      */
      virtual void transform();

      /**
      \return true if the parent object has been deleted
      */
      virtual agx::Bool objectDeleted() const;

      /**
      \return true if valid
      */
      virtual agx::Bool valid() const;

      /**
      \return the transformed data
      */
      const agx::Vec3& get( agx::Attachment::Transformed entry ) const;

      /**
      \return the local data
      */
      agx::Vec3 getLocal( agx::Attachment::Transformed entry ) const;

      /**
      Assign a new frame - transformed data will be updated.
      \param frame - new frame
      */
      void setFrame( agx::Frame* frame );

      /**
      \return the frame
      */
      agx::Frame* getFrame() const;

      /**
      \param other - The other attachment
      \return the distance between this and the other attachment
      */
      virtual agx::Real calculateDistance( const agx::Attachment* other ) const;

      /**
      \return the projected (along this attachments world z-axis) distance between this and the other attachment
      */
      virtual agx::Real calculateLinearDistance( const agx::Attachment* other ) const;

      /**
      \return relative speed along world z-axis of this attachment
      */
      virtual agx::Real calculateLinearSpeed( const agx::Attachment* other ) const;

      /**
      \return relative speed about world z-axis of this attachment
      */
      virtual agx::Real calculateAngularSpeed( const agx::Attachment* other ) const;

      /**
      \return linear velocity of this attachment given in world frame
      */
      virtual agx::Vec3 getLinearVelocity() const;

      /**
      \return linear velocity of this attachment given in world frame
      */
      virtual agx::Vec3 getAngularVelocity() const;

      /**
      \return true if this object is of type \p T
      */
      template< typename T >
      agx::Bool is() const;

      /**
      Unsafe version, static cast of this to \p T.
      \see safeAs
      \return this attachment as subclass using static cast
      */
      template< typename T >
      T* as();

      /**
      Dynamic cast of this to \p T.
      \return this attachment as subclass using dynamic cast
      */
      template< typename T >
      T* safeAs();

      /**
      Unsafe version, static cast of this to \p T.
      \sa safeAs
      \return this attachment as subclass using static cast
      */
      template< typename T >
      const T* as() const;

      /**
      Dynamic cast of this to \p T.
      \return this attachment as subclass using dynamic cast
      */
      template< typename T >
      const T* safeAs() const;

      AGXSTREAM_DECLARE_SERIALIZABLE(agx::Attachment);

    protected:
      /**
      Constructor to use during restore.
      */
      Attachment();

      /**
      Reference counted object, protected destructor.
      */
      virtual ~Attachment();

      /**
      Fill \p Transformed data given center of mass transform and offset from
      model frame to center of mass frame (assumes not rotated relative each other).
      \param cmTransform - world transform of the center of mass frame
      \param cmToModel - in local coordinates, vector from center of mass frame to model frame
      \param localConstraintTransform - transform of the local constraint frame
      */
      void transform( const agx::AffineMatrix4x4& cmTransform, const agx::Vec3& cmToModel, const agx::AffineMatrix4x4& localConstraintTransform );

    protected:
      agx::Vec3     m_transformed[ NUM_ELEMENTS ];
      agx::FrameRef m_relFrame;
  };

  typedef agx::ref_ptr< Attachment > AttachmentRef;

  /**
  Constraint attachment class for agx::RigidBody.
  */
  class AGXPHYSICS_EXPORT RigidBodyAttachment : public agx::Attachment
  {
    public:
      /**
      Construct given rigid body and relative frame.
      \param rb - rigid body (0 if e.g., world frame or other type of static frame)
      \param relFrame - constraint attachment frame relative \p rb model frame
      */
      RigidBodyAttachment( agx::RigidBody* rb, agx::Frame* relFrame );

      /**
      \return true if the rigid body or the frame has been deleted
      */
      virtual agx::Bool objectDeleted() const override;

      /**
      \return true if the current configuration is valid
      */
      virtual agx::Bool valid() const override;

      /**
      \return the distance between this and the other attachment
      \sa calculateLinearDistance
      */
      virtual agx::Real calculateDistance( const agx::Attachment* other ) const override;

      /**
      \return the projected (along this attachments world z-axis) distance between this and the other attachment
      */
      virtual agx::Real calculateLinearDistance( const agx::Attachment* other ) const override;

      /**
      \return relative speed along world z-axis of this attachment
      */
      virtual agx::Real calculateLinearSpeed( const agx::Attachment* other ) const override;

      /**
      \return relative speed about world z-axis of this attachment
      */
      virtual agx::Real calculateAngularSpeed( const agx::Attachment* other ) const override;

      /**
      \return linear velocity of this attachment given in world frame
      */
      virtual agx::Vec3 getLinearVelocity() const override;

      /**
      \return linear velocity of this attachment given in world frame
      */
      virtual agx::Vec3 getAngularVelocity() const override;

      /**
      Transform given rigid body and frame.
      */
      virtual void transform() override;

      /**
      Assign rigid body to this constraint frame.
      \note The actual constraint will not know about this change unless it's handled explicitly.
      \param rb - new rigid body
      */
      void setRigidBody( agx::RigidBody* rb );

      /**
      \return the rigid body
      */
      agx::RigidBody* getRigidBody() const;

      AGXSTREAM_DECLARE_SERIALIZABLE(agx::RigidBodyAttachment);

    protected:
      /**
      Protected default constructor used by restore.
      */
      RigidBodyAttachment();

      /**
      Reference counted object, protected destructor.
      */
      virtual ~RigidBodyAttachment();

    protected:
      agx::RigidBodyObserver m_rb;
      agx::Bool              m_hadValidRb;
  };

  typedef agx::ref_ptr< RigidBodyAttachment > RigidBodyAttachmentRef;

  /**
  Constraint attachment pair class. Data holder for a pair of constraint
  attachments (i.e., what defines the difference between the frames that
  is used in constraints).
  */
  class AGXPHYSICS_EXPORT AttachmentPair
  {
    public:
      /**
      Data access enum.
      */
      enum Transformed
      {
        SEPARATION,  /**< Separation vector, i.e., the vector from frame 2 to frame 1. */
        NUM_ELEMENTS /**< Number of data elements. */
      };

    public:
      /**
      Custom data for separation pair. During transform the subclass will get
      transform-call AFTER Attachment::transform and BEFORE
      Angle::transform.
      */
      class AGXPHYSICS_EXPORT CustomData : public agx::Referenced
      {
        public:
          /**
          Called after the other attachments have been transformed.
          */
          virtual void transform( const agx::AttachmentPair& attachmentPair ) = 0;

        protected:
          /**
          Reference counted object, protected destructor.
          */
          virtual ~CustomData() {}
      };

      typedef agx::ref_ptr< CustomData > CustomDataRef;

    public:
      /**
      Default constructor, attachment frame 1 = attachment frame 2 = 0.
      */
      AttachmentPair();

      /**
      Destructor.
      */
      virtual ~AttachmentPair();

      /**
      Assign constraint attachment frames.
      \param a1 - first attachment frame
      \param a2 - second attachment frame
      */
      void set( agx::Attachment* a1, agx::Attachment* a2 );

      /**
      Add constraint angle.
      \param angle - constraint angle to add
      */
      void add( agx::Angle* angle );

      /**
      Access constraint angle given index.
      \param index - index of angle
      \return constraint angle at index (0 if index is out of bounds)
      */
      agx::Angle* getAngle( agx::UInt index ) const;

      /**
      \return constraint angle container
      */
      const agx::AngleContainer& getAngles() const;

      /**
      \return true if angle exist in this attachment pair
      */
      agx::Bool hasAngle( const agx::Angle* angle ) const;

      /**
      Calculates the rotational/angular speed between the two attachments, about
      the constraint axis.
      \return rotational speed about the constraint axis
      */
      agx::Real calculateAngularSpeed() const;

      /**
      Calculates the linear speed between the two attachments, along
      the constraint axis.
      \return linear speed along the constraint axis
      */
      agx::Real calculateLinearSpeed() const;

      /**
      Assign custom data to receive transform calls.
      \param customData - user defined custom data
      */
      void setCustomData( agx::AttachmentPair::CustomData* customData );

      /**
      \return custom data if assigned
      */
      agx::AttachmentPair::CustomData* getCustomData() const;

      /**
      \return the attachment of index \p i (0 or 1!)
      */
      agx::Attachment* operator[] ( agx::UInt i ) const;

      /**
      \return first attachment in this attachment pair
      */
      agx::Attachment* getAttachment1() const;

      /**
      \return second attachment in this attachment pair
      */
      agx::Attachment* getAttachment2() const;

      /**
      \return true if the current state is valid
      */
      agx::Bool valid() const;

      /**
      Transform attachment frames and updates data.
      */
      virtual void transform();

      /**
      Rebind given first frame.
      */
      virtual agx::Bool rebind();

      /**
      \return transformed data
      */
      const agx::Vec3& get( agx::AttachmentPair::Transformed entry ) const;

      /**
      Store given stream.
      */
      virtual void store( agxStream::OutputArchive& out ) const;

      /**
      Restore given stream.
      */
      virtual void restore( agxStream::InputArchive& in );

      /**
      Store structural independent data to stream.
      */
      virtual void storeLightData( agxStream::StorageStream& str ) const;

      /**
      Restore structural independent data from stream.
      */
      virtual void restoreLightData( agxStream::StorageStream& str );

    protected:
      AttachmentRef  m_attachments[ 2 ];
      agx::Vec3      m_transformed[ NUM_ELEMENTS ];
      CustomDataRef  m_customData;
      AngleContainer m_angles;
  };

  AGX_FORCE_INLINE const Vec3& Attachment::get( Transformed entry ) const
  {
    return m_transformed[ entry ];
  }

  AGX_FORCE_INLINE Frame* Attachment::getFrame() const
  {
    return m_relFrame;
  }

  template< typename T >
  AGX_FORCE_INLINE Bool Attachment::is() const
  {
    return dynamic_cast< const T* >( this ) != nullptr;
  }

  template<>
  AGX_FORCE_INLINE Bool Attachment::is<Attachment>() const
  {
    return true;
  }

  template< typename T >
  AGX_FORCE_INLINE T* Attachment::as()
  {
    // Compiler warns on 'this == nullptr' because it is illegal to call member
    // functions on nullptr. However, if it helps us find a nullptr dereference
    // earlier, and with an assert instead of segmentation fault, then I think
    // it's ok to leave the check.
    #include <agx/PushDisableWarnings.h>
    agxAssert( (void *)this == nullptr || this->is<T>() );
    return static_cast< T* >( this );
    #include <agx/PopDisableWarnings.h>
  }

  template< typename T >
  AGX_FORCE_INLINE T* Attachment::safeAs()
  {
    return dynamic_cast< T* >( this );
  }

  template< typename T >
  AGX_FORCE_INLINE const T* Attachment::as() const
  {
    // Compiler warns on 'this == nullptr' because it is illegal to call member
    // functions on nullptr. However, if it helps us find a nullptr dereference
    // earlier, and with an assert instead of segmentation fault, then I think
    // it's ok to leave the check.
    #include <agx/PushDisableWarnings.h>
    agxAssert( (void *)this == nullptr || this->is< T >() ); return static_cast< const T* >( this );
    #include <agx/PopDisableWarnings.h>
  }

  template< typename T >
  AGX_FORCE_INLINE const T* Attachment::safeAs() const
  {
    return dynamic_cast< const T* >( this );
  }

  AGX_FORCE_INLINE RigidBody* RigidBodyAttachment::getRigidBody() const
  {
    return m_rb;
  }

  AGX_FORCE_INLINE Attachment* AttachmentPair::operator[]( UInt i ) const
  {
    agxAssert( i < 2 );
    return m_attachments[ i ].get();
  }

  inline Attachment* AttachmentPair::getAttachment1() const
  {
    return (*this)[ 0 ];
  }

  inline Attachment* AttachmentPair::getAttachment2() const
  {
    return (*this)[ 1 ];
  }

  AGX_FORCE_INLINE const Vec3& AttachmentPair::get( Transformed entry ) const
  {
    return m_transformed[ entry ];
  }
} // namespace agx

#ifdef _MSC_VER
#pragma warning( pop )
#endif

#endif
