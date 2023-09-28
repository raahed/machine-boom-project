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

#ifndef AGX_CONSTRAINTANGLE_H
#define AGX_CONSTRAINTANGLE_H

#include <agx/Vec3.h>

#include <agxStream/Serializable.h>

namespace agx
{
  class AttachmentPair;

  /**
  Constraint angle class.
  */
  class AGXPHYSICS_EXPORT Angle : public agx::Referenced, public agxStream::Serializable
  {
    public:
      /**
      Axis this constraint angle is defined about or along.
      */
      enum Axis
      {
        U, /**< Constraint x axis. */
        V, /**< Constraint y axis. */
        N  /**< Constraint z axis. */
      };

      /**
      Type, i.e., defines \p Axis as 'along' or 'about'.
      */
      enum Type
      {
        TRANSLATIONAL, /**< \p Axis is along U, V or N. */
        ROTATIONAL     /**< \p Axis is about U, V or N. */
      };

    public:
      /**
      Transform method to update the value of this angle.
      */
      virtual void transform( const agx::AttachmentPair& );

      /**
      \return the axis this constraint angle is defined about or along
      */
      agx::Angle::Axis getAxis() const;

      /**
      \return the type of this constraint angle (i.e., if \p Axis is 'along' or 'about')
      */
      agx::Angle::Type getType() const;

      /**
      \return const reference to value (so it may be used by elementary constraints that works on this angle)
      */
      const agx::Real& getValue() const;

      /**
      \param ap    - attachment pair
      \param index - which attachment (0,1)
      \return the world direction of the axis given the attachment pair
      */
      virtual const agx::Vec3& getDir( const agx::AttachmentPair& ap, agx::UInt index ) const;

      /**
      Store given stream.
      */
      virtual void store( agxStream::OutputArchive& out ) const override;

      /**
      Restore given stream.
      */
      virtual void restore( agxStream::InputArchive& in ) override;

      AGXSTREAM_DECLARE_ABSTRACT_SERIALIZABLE( agx::Angle );

      /**
      Store structural independent data to stream.
      */
      virtual void storeLightData( agxStream::StorageStream& str ) const override;

      /**
      Restore structural independent data from stream.
      */
      virtual void restoreLightData( agxStream::StorageStream& str ) override;

    protected:
      /**
      Construct given axis and initial angle.
      */
      Angle( agx::Angle::Axis axis, agx::Angle::Type type, agx::Real initialAngle = agx::Real( 0 ) );

      /**
      Reference counted object, protected destructor.
      */
      virtual ~Angle();

    protected:
      Axis      m_axis;
      Type      m_type;
      agx::Real m_value;
  };

  typedef agx::ref_ptr< Angle > AngleRef;
  typedef agx::Vector< AngleRef > AngleContainer;

  /**
  Implementation of constraint angle ABOUT an axis.
  */
  class AGXPHYSICS_EXPORT RotationalAngle : public agx::Angle
  {
    public:
      /**
      Utility cast method.
      */
      static agx::RotationalAngle* safeCast( const agx::Angle* angle );

    public:
      /**
      Construct given axis and initial angle.
      */
      RotationalAngle( agx::Angle::Axis axis, agx::Real initialAngle = agx::Real( 0 ) );

      /**
      Update value given constraint pair.
      */
      virtual void transform( const agx::AttachmentPair& ap ) override;

      /**
      \return the winding number
      */
      agx::Int getWindingNumber() const;

      /**
      Use with caution since each values of \p windingNumber represent plus or minus
      360 degrees of the joint angle (e.g, affecting locks and ranges).
      \param windingNumber - new windingNumber
      */
      void setWindingNumber( agx::Int windingNumber );

      /**
      \return the previous angle (-pi, +pi) (winding number NOT included)
      */
      agx::Real getLastAngle() const;

      /**
      Set the last angle bewteen (-pi, +pi) (winding number NOT included)
      \param lastAngle - will be clamped in range (-pi, +pi) if outside
      */
      void setLastAngle(agx::Real lastAngle);

      /**
      \param ap    - attachment pair
      \return the calulated angle given the constraint attachment frames of an attachment pair (winding number NOT included)
      */
      agx::Real calculateCurrentAngle(const AttachmentPair& ap) const;

      AGXSTREAM_DECLARE_SERIALIZABLE(agx::RotationalAngle);

      /**
      Store structural independent data to stream.
      */
      virtual void storeLightData( agxStream::StorageStream& str ) const override;

      /**
      Restore structural independent data from stream.
      */
      virtual void restoreLightData( agxStream::StorageStream& str ) override;

    protected:
      /**
      Default constructor used during restore.
      */
      RotationalAngle();

      /**
      Reference counted object, protected destructor.
      */
      virtual ~RotationalAngle();

    protected:
      agx::Int  m_windingNumber;
      agx::Real m_lastAngle;
  };

  typedef agx::ref_ptr< RotationalAngle > RotationalAngleRef;

  /**
  Implementation of constraint angle ALONG an axis.
  */
  class AGXPHYSICS_EXPORT SeparationAngle : public agx::Angle
  {
    public:
      /**
      Utility cast method.
      */
      static agx::SeparationAngle* safeCast( const agx::Angle* angle );

    public:
      /**
      Construct given axis and initial angle.
      */
      SeparationAngle( agx::Angle::Axis axis, agx::Real initialAngle = agx::Real( 0 ) );

      /**
      Update value given constraint pair.
      */
      virtual void transform( const agx::AttachmentPair& ap ) override;

      AGXSTREAM_DECLARE_SERIALIZABLE(agx::SeparationAngle);

    protected:
      /**
      Default constructor used during restore.
      */
      SeparationAngle();

      /**
      Reference counted object, protected destructor.
      */
      virtual ~SeparationAngle();
  };

  typedef agx::ref_ptr< SeparationAngle > SeparationAngleRef;

  AGX_FORCE_INLINE const Real& Angle::getValue() const
  {
    return m_value;
  }
} //namespace agx

#endif
