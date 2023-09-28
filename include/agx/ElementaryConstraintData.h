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

#ifndef AGX_ELEMENTARYCONSTRAINTDATA_H
#define AGX_ELEMENTARYCONSTRAINTDATA_H

#include <agx/Attachment.h>

#ifdef _MSC_VER
# pragma warning(push)
# pragma warning(disable: 6011) // Disable warningC6011: dereferencing nullptr pointer
#endif


namespace agx
{
  /**
     Class to hold data for elementary constraints.
  */
  class AGXPHYSICS_EXPORT ElementaryConstraintData
  {
  public:
    /**
       Used during serialization restore.
    */
    ElementaryConstraintData();

    /**
       Construct given attachment pair.
       \param ap - attachment pair
    */
    ElementaryConstraintData( const agx::AttachmentPair* ap );

    /**
       Destructor.
    */
    virtual ~ElementaryConstraintData();

    /**
       Assign new attachment pair.
       \param ap - attachment pair
    */
    virtual void set( const agx::AttachmentPair* ap );

    /**
       \return the attachment pair
    */
    AGX_FORCE_INLINE const agx::AttachmentPair* getAttachmentPair() const { agxAssert( m_ap != nullptr && m_ap->valid() ); return m_ap; }

    /**
       \param index - index of the attachment (0 or 1)
       \return the constraint attachment for attachment with index \p index
    */
    AGX_FORCE_INLINE const agx::Attachment* getAttachment( agx::UInt index ) const { agxAssert( m_ap != nullptr && m_ap->valid() ); return (*m_ap)[ index ]; }

    /**
       \return the separation vector between the two attachments (att1->pos - att2->pos)
    */
    AGX_FORCE_INLINE const agx::Vec3& getSeparation() const { agxAssert( m_ap != nullptr && m_ap->valid() ); return m_ap->get( AttachmentPair::SEPARATION ); }

    /**
       \return the vector going from center of mass to the constraint anchor point given index (0 or 1)
    */
    AGX_FORCE_INLINE const agx::Vec3& getCmToAnchorPosition( agx::UInt index ) const { agxAssert( m_ap != nullptr && m_ap->valid() ); return getAttachment( index )->get( Attachment::CM_TO_ANCHOR_POS ); }

    /**
       \return the constraint anchor point in world coordinates given index (0 or 1)
    */
    AGX_FORCE_INLINE const agx::Vec3& getWorldToAnchorPosition( agx::UInt index ) const { agxAssert( m_ap != nullptr && m_ap->valid() ); return getAttachment( index )->get( Attachment::ANCHOR_POS ); }

    /**
       Store to stream.
    */
    virtual void store( agxStream::OutputArchive& out ) const;

    /**
       Restore from stream.
    */
    virtual void restore( agxStream::InputArchive& in );

  protected:
    const agx::AttachmentPair* m_ap;
  };

  /**
     Data holder class for elementary constraint Dot1.
  */
  class AGXPHYSICS_EXPORT Dot1Data : public ElementaryConstraintData
  {
  public:
    /**
       Default constructor used during restore of this object.
    */
    Dot1Data();

    /**
       Construct given attachment pair and the two directions that should be orthogonal.
       \param ap - constraint attachment pair
       \param dir1 - direction reference attachment 1
       \param dir2 - direction reference attachment 2
    */
    Dot1Data( const agx::AttachmentPair* ap, agx::Attachment::Transformed dir1, agx::Attachment::Transformed dir2 );

    /**
       Destructor.
    */
    virtual ~Dot1Data();

    /**
       \return direction reference attachment 1
    */
    AGX_FORCE_INLINE const agx::Vec3& getDir1() const { return getAttachment( 0 )->get( m_dir1 ); }

    /**
       \return direction reference attachment 2
    */
    AGX_FORCE_INLINE const agx::Vec3& getDir2() const { return getAttachment( 1 )->get( m_dir2 ); }

    /**
       Store data to stream.
    */
    virtual void store( agxStream::OutputArchive& out ) const override;

    /**
       Restore data from stream.
    */
    virtual void restore( agxStream::InputArchive& in ) override;

  protected:
    agx::Attachment::Transformed m_dir1;
    agx::Attachment::Transformed m_dir2;
  };


  /**
  Extended Dot1Data with additional slack parameter
  */
  class AGXPHYSICS_EXPORT Dot1SlackData : public Dot1Data
  {
    public:
    /**
    Default constructor
    */
    Dot1SlackData();

    /**
    */
    Dot1SlackData( const agx::AttachmentPair* ap, agx::Attachment::Transformed dir1, agx::Attachment::Transformed dir2 );

    virtual ~Dot1SlackData();

    /**
    Set amount of slack in constraint
    */
    AGX_FORCE_INLINE void setSlack( agx::Real slack )
    {
      m_slack = slack;
    }

    /**
    \return Get amount of slack in constraint
    */
    AGX_FORCE_INLINE agx::Real getSlack() const
    {
      return m_slack;
    }

    /**
    Store data to stream.
    */
    virtual void store( agxStream::OutputArchive& out ) const override;

    /**
    Restore data from stream.
    */
    virtual void restore( agxStream::InputArchive& in ) override;


  protected:
    agx::Real m_slack;
  };

  /**
     Data holder class for elementary constraint Dot2.
  */
  class AGXPHYSICS_EXPORT Dot2Data : public ElementaryConstraintData
  {
  public:
    /**
       Default constructor used during restore of this object.
    */
    Dot2Data();

    /**
       Construct given constraint attachment pair and direction to be
       orthogonal to the separation vector.
       \param ap - attachment pair
       \param dir - direction reference attachment 1 to be orthogonal to the separation vector
       \param body - the first body of the constraint
    */
    Dot2Data(
      const agx::AttachmentPair* ap,
      agx::Attachment::Transformed dir,
      agx::Int body=0
    );

    /**
       Destructor.
    */
    virtual ~Dot2Data();

    virtual agx::Int  getReferenceBody() const { return m_body; }

    /**
       \return direction reference attachment for "first body" to be orthogonal to the separation vector
    */
    AGX_FORCE_INLINE const agx::Vec3& getDir() const { return getAttachment( m_body )->get( m_dir ); }

    /**
       Store data to stream.
    */
    virtual void store( agxStream::OutputArchive& out ) const override;

    /**
       Restore data from stream.
    */
    virtual void restore( agxStream::InputArchive& in ) override;

  protected:
    agx::Attachment::Transformed m_dir;
    agx::Int m_body;
  };



  /**
     Data holder class for elementary constraint Dot2Slack.
  */
  class AGXPHYSICS_EXPORT Dot2SlackData : public Dot2Data
  {
  public:
    /**
       Default constructor used during restore of this object.
    */
    Dot2SlackData();

    /**
       Construct given constraint attachment pair and direction to be
       orthogonal to the separation vector.
       \param ap - attachment pair
       \param dir - direction reference attachment 1 to be orthogonal to the separation vector
    */
    Dot2SlackData( const agx::AttachmentPair* ap, agx::Attachment::Transformed dir );

    /**
       Destructor.
    */
    virtual ~Dot2SlackData();


    /**
       Store data to stream.
    */
    virtual void store( agxStream::OutputArchive& out ) const override;

    /**
       Restore data from stream.
    */
    virtual void restore( agxStream::InputArchive& in ) override;

    AGX_FORCE_INLINE agx::Real getSlack() const
    {
      return m_slack;
    }

    AGX_FORCE_INLINE void setSlack( agx::Real slack )
    {
      m_slack = slack;
    }

  protected:
    agx::Real m_slack;
  };




  /**
  Generic class for holding constraint data including 3 slack parameters
  */
  class AGXPHYSICS_EXPORT Slack3ConstraintData : public ElementaryConstraintData
  {
  public:
    Slack3ConstraintData();

    Slack3ConstraintData( const agx::AttachmentPair* ap );

    virtual ~Slack3ConstraintData();

    virtual void store( agxStream::OutputArchive& out ) const override;

    virtual void restore( agxStream::InputArchive& in ) override;

    AGX_FORCE_INLINE const agx::Vec3 getSlack() const
    {
      return m_slack;
    }

    AGX_FORCE_INLINE void setSlack( agx::Vec3 slack )
    {
      m_slack = slack;
    }

  protected:
    agx::Vec3 m_slack;
  };



  /**
     Data holder class for elementary constraint QuatLock.
  */
  class AGXPHYSICS_EXPORT QuatLockData : public ElementaryConstraintData
  {
  public:
    /**
       The class holding the actual data and that transforms the data.
    */
    class AGXPHYSICS_EXPORT QuatTransformer : public AttachmentPair::CustomData
    {
    public:
      /**
         Default constructor.
      */
      QuatTransformer();

      /**
         Transform data given the attachments.
      */
      virtual void transform( const agx::AttachmentPair& attachmentPair ) override;

      /**
         \return the quaternion with index \p i < 3
      */
      AGX_FORCE_INLINE const agx::Quat& operator[] ( agx::UInt i ) const { return m_q[ i ]; }

      /**
         \return the quaternion with index \p i < 3
      */
      AGX_FORCE_INLINE       agx::Quat& operator[] ( agx::UInt i )       { return m_q[ i ]; }

    protected:
      /**
         Reference counted object, protected destructor.
      */
      virtual ~QuatTransformer() {}

    protected:
      agx::Quat m_q[ 3 ]; /**< [0] = attachment 1 quaternion, [1] = attachment 2 quaternion, [2] = defined relative quaternion (default identity) */
    };

    typedef agx::ref_ptr< QuatTransformer > QuatTransformerRef;

  public:
    /**
       Default constructor used during restore of this object.
    */
    QuatLockData();

    /**
       Construct given attachment pair.
       \param ap - constraint attachment pair
    */
    QuatLockData( const agx::AttachmentPair* ap );

    /**
       Destructor.
    */
    virtual ~QuatLockData();

    /**
       Associate attachment pair to this object, for this object
       to register QuatTransformer as attachment pair custom data.
       \param ap - the constraint attachment pair
    */
    void set( const agx::AttachmentPair* ap ) override;

    /**
       \return the quaternion with index \p i < 3
    */
    AGX_FORCE_INLINE const agx::Quat& operator[] ( agx::UInt i ) const { agxAssert( m_qTransformer != nullptr ); return (*m_qTransformer)[ i ]; }

    /**
       \return the quaternion with index \p i < 3
    */
    AGX_FORCE_INLINE       agx::Quat& operator[] ( agx::UInt i )       { agxAssert( m_qTransformer != nullptr ); return (*m_qTransformer)[ i ]; }

    /**
       Store data to stream.
    */
    virtual void store( agxStream::OutputArchive& out ) const override;

    /**
       Restore data from stream.
    */
    virtual void restore( agxStream::InputArchive& in ) override;

  protected:
    QuatTransformerRef m_qTransformer;
  };

  /**
     Basic data holder class for "angle based" (secondary) constraints.
  */
  class AGXPHYSICS_EXPORT ConstraintAngleBasedData : public ElementaryConstraintData
  {
  public:
    /**
       Default constructor used during restore of this object.
    */
    ConstraintAngleBasedData();

    /**
       Construct given constraint attachment pair and the input angle for
       the elementary constraint to work on.
       \param ap - constraint attachment pair
       \param angle - input angle
    */
    ConstraintAngleBasedData( const agx::AttachmentPair* ap, const agx::Angle* angle );

    /**
       Destructor.
    */
    virtual ~ConstraintAngleBasedData();

    /**
       \return the angle used by the elementary constraint
    */
    const agx::Angle* getAngle() const;

    /**
       \return the world direction the angle is defined along or about for first attachment
    */
    const agx::Vec3& getDir1() const;

    /**
       \return the world direction the angle is defined along or about for second attachment
    */
    const agx::Vec3& getDir2() const;

    /**
       Store data to stream.
    */
    virtual void store( agxStream::OutputArchive& out ) const override;

    /**
       Restore data from stream.
    */
    virtual void restore( agxStream::InputArchive& in ) override;

  protected:
    typedef agx::ref_ptr< const agx::Angle > AngleRefType;

  protected:
    AngleRefType m_angle;
  };

  /**
     Data holder class for elementary swing constraint.
  */
  class AGXPHYSICS_EXPORT SwingData : public ElementaryConstraintData {
    public:
      /**
         Default constructor used during restore of this object.
      */
      SwingData();

      /**
         Construct given attachment pair and the two directions that should be orthogonal.
         \param ap - constraint attachment pair
         \param dir1U - direction reference attachment 1
         \param dir1V - direction reference attachment 1
         \param dir1N - direction reference attachment 1
         \param dir2N - direction reference attachment 2
      */
      SwingData(const agx::AttachmentPair* ap, agx::Attachment::Transformed dir1U, agx::Attachment::Transformed dir1V,
                agx::Attachment::Transformed dir1N, agx::Attachment::Transformed dir2N);

      /**
         Destructor.
      */
      virtual ~SwingData();

      /**
         \return direction reference attachment U of frame 1.
      */
      AGX_FORCE_INLINE const agx::Vec3& getDir1U() const
      {
        return getAttachment( 0 )->get( m_dir1U );
      }

      /**
         \return direction reference attachment V of frame 1.
      */
      AGX_FORCE_INLINE const agx::Vec3& getDir1V() const
      {
        return getAttachment( 0 )->get( m_dir1V );
      }

      /**
         \return direction reference attachment N of frame 1.
      */
      AGX_FORCE_INLINE const agx::Vec3& getDir1N() const
      {
        return getAttachment( 0 )->get( m_dir1N );
      }

      /**
         \return direction reference attachment N of frame 2.
      */
      AGX_FORCE_INLINE const agx::Vec3& getDir2N() const
      {
        return getAttachment( 1 )->get( m_dir2N );
      }

      /**
         \return swing axis
      */
      AGX_FORCE_INLINE agx::Vec3 getSwingAxis() const
      {
        return getDir1N().getPerpendicularUnitVector(getDir2N());
      }

      /**
         \return swing angle theta
      */
      AGX_FORCE_INLINE agx::Real getSwingAngle() const
      {
        return std::acos(agx::clamp(getDir1N() * getDir2N(), agx::Real(-1.0), agx::Real(1.0)));
      }

      /**
         Store data to stream.
      */
      virtual void store(agxStream::OutputArchive& out) const override;

      /**
         Restore data from stream.
      */
      virtual void restore(agxStream::InputArchive& in) override;

    protected:
      agx::Attachment::Transformed m_dir1U;
      agx::Attachment::Transformed m_dir1V;
      agx::Attachment::Transformed m_dir1N;
      agx::Attachment::Transformed m_dir2N;
  };

  AGX_FORCE_INLINE const Angle* ConstraintAngleBasedData::getAngle() const
  {
    return m_angle;
  }

  AGX_FORCE_INLINE const Vec3& ConstraintAngleBasedData::getDir1() const
  {
    agxAssert( m_angle != nullptr );
    return m_angle->getDir( *m_ap, 0 );
  }

  AGX_FORCE_INLINE const Vec3& ConstraintAngleBasedData::getDir2() const
  {
    agxAssert( m_angle != nullptr );
    return m_angle->getDir( *m_ap, 1 );
  }



  /**
     Data holder class for elementary constraint Twist.
  */
  class AGXPHYSICS_EXPORT TwistData : public QuatLockData {

    public:
      /**
         Default constructor used during restore of this object.
      */
      TwistData();

      /**
         Construct given attachment pair.
         \param ap - constraint attachment pair
      */
      TwistData(const agx::AttachmentPair* ap, agx::Attachment::Transformed dir1, agx::Attachment::Transformed dir2);

      /**
         Destructor.
      */
      virtual ~TwistData();

      /**
         Store data to stream.
      */
      virtual void store(agxStream::OutputArchive& out) const override;

      /**
         Restore data from stream.
      */
      virtual void restore(agxStream::InputArchive& in) override;


      /**
         \return direction reference attachment 1
      */
      AGX_FORCE_INLINE const agx::Vec3& getDir1() const
      {
        return getAttachment( 0 )->get( m_dir1 );
      }

      /**
         \return direction reference attachment 2
      */
      AGX_FORCE_INLINE const agx::Vec3& getDir2() const
      {
        return getAttachment( 1 )->get( m_dir2 );
      }

    protected:
      agx::Attachment::Transformed m_dir1;
      agx::Attachment::Transformed m_dir2;

  };

}

#ifdef _MSC_VER
# pragma warning(pop)
#endif


#endif
