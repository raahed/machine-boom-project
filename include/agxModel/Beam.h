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

#include <agxModel/BeamModel.h>
#include <agxModel/BeamSegment.h>
#include <agxModel/BeamSegmentIterator.h>

#include <agxSDK/LinkedStructure.h>

namespace agxModel
{
  AGX_DECLARE_POINTER_TYPES( Beam );
  AGX_DECLARE_VECTOR_TYPES( Beam );

  /**
  Beam is a segmented, lumped element structure of a given length and resolution.
  The properties of a beam, e.g., geometry, parameters, moment of inertia (area and
  polar) are defined in the model of the beam (agxModel::BeamModel). The beam models
  may be shared between several beam instances.

  An initialized beam with default transform will be created along the z axis with
  start at the origin. This means that the end of the last beam segment is at
  (0, 0, length). The segments of the beam is defined to have x up, y right and
  z forward (long the beam).
  */
  class AGXMODEL_EXPORT Beam : public agxSDK::LinkedStructure
  {
    public:
      /**
      Creates an initialized beam between the given start and end positions, beam model,
      beam resolution and world up axis (default z). The created beam up axis (x)
      will align with the given \p worldUpAxis. E.g., given two points and an agxModel::IBeam
      model with world up = (0, 0, 1) would result in I, but world up = (0, 1, 0) would result
      in |-|.

      Beam frame:
          x
          |  z
          | /
          |/____ y

      The beam segments will be initialized along the z-axis in the beam frame.
      \param startPosition - start position of the beam, given in world coordinate frame
      \param endPosition - end position of the beam, given in world coordinate frame
      \param model - beam model of the beam
      \param resolution - resolution, number of segments, of the beam
      \param worldUpAxis - world up axis the beam up axis should align with
      \return beam instance if all inputs are valid, otherwise nullptr (with logged warnings)
      */
      static BeamRef create( agx::Vec3 startPosition,
                             agx::Vec3 endPosition,
                             BeamModel* model,
                             agx::UInt resolution,
                             agx::Vec3 worldUpAxis = agx::Vec3::Z_AXIS() );

      /**
      Calculates the transform of a beam given start position, forward direction and
      world up axis the beam up axis should align with.
      \param startPosition - start position of the beam in world coordinate frame
      \param forward - forward direction of the beam in world coordinate frame, e.g., |end position - start position|
      \param worldUpAxis - world up axis the beam should align with
      \return transform of the beam if all inputs are valid, otherwise an identity matrix (with logged warnings)
      */
      static agx::AffineMatrix4x4 calculateTransform( agx::Vec3 startPosition,
                                                      agx::Vec3 forward,
                                                      agx::Vec3 worldUpAxis = agx::Vec3::Z_AXIS() );

      /**
      Calculates the rotation of a beam frame given forward direction and world up axis
      the beam frame up axis should align with.
      \param forward - forward direction of the beam in world coordinate frame, e.g., |end position - start position|
      \param worldUpAxis - world up axis the beam should align with
      \return rotation of the beam if all inputs are valid, otherwise and identity quaternion (with logged warnings)
      */
      static agx::Quat calculateRotation( agx::Vec3 forward,
                                          agx::Vec3 worldUpAxis = agx::Vec3::Z_AXIS() );

      /**
      Creates a new beam given \p model and restores all data from the given storage
      stream. When resolution and length are stored in the stream, only the beam model
      is needed. The parameters (Young's modulus, Poisson's ratio, etc.) will be written
      to the \p model properties. This method calls restoreLightData on the created beam.
      \param model - beam model of the stored beam (invalid if nullptr and undefined if
                     the beam model type differs from when the data was stored).
      \param str - storage stream with mode RESTORE
      \return restored beam instance if everything is valid, otherwise nullptr
      */
      static BeamRef restore( agxModel::BeamModel* model, agxStream::StorageStream& str );

      /**
      \param rb - rigid body
      \return beam instance \p rb belongs to, given \p rb is a BeamSegment body
      */
      static Beam* get( agx::RigidBody* rb );

      /**
      \param rb - rigid body
      \return beam instance \p rb belongs to, given \p rb is a BeamSegment body
      */
      static const Beam* get( const agx::RigidBody* rb );

      /**
      Find beam instance given simulation and name.
      \param simulation - simulation
      \param name - name of the beam instance to find
      \return beam instance with given name if found - otherwise nullptr
      */
      static Beam* find( agxSDK::Simulation* simulation, const agx::Name& name );

      /**
      Find beam instance given simulation and unique id.
      \param simulation - simulation
      \param uuid - UUID of the beam instance to find
      \return beam instance with given UUID if found - otherwise nullptr
      */
      static Beam* find( agxSDK::Simulation* simulation, const agx::Uuid& uuid );

      /**
      Finds all agxModel::Beam instances in the given simulation.
      \param simulation - simulation
      \return beam instances found in the given simulation
      */
      static agxModel::BeamPtrVector findAll( const agxSDK::Simulation* simulation );

      /**
      Finds all beam instances given simulation and name.
      \param simulation - simulation
      \param name - name of beam instances to find
      \return beam instances with given name
      */
      static agxModel::BeamPtrVector findAll( const agxSDK::Simulation* simulation, const agx::Name& name );

    public:
      /**
      Construct a beam given model, resolution (number of elements) and total length.
      \param model - beam model of this beam (invalid if nullptr)
      \param resolution - the total number of beam segments in this beam when initialized, i.e.,
                          the number of rigid bodies and resolution - 1 constraints (invalid if 0)
      \param length - the total length of this beam (invalid if 0.0)
      */
      Beam( BeamModel* model, agx::UInt resolution, agx::Real length );

      /**
      \return the model of this beam
      */
      BeamModel* getModel() const;

      /**
      Current begin position, in world coordinate frame, of this (initialized) beam. Shortcut for:
          beam->getSegments().front()->getBeginPosition()
      \return the begin position of this initialized beam, in world coordinate frame. If this
              beam hasn't been initialized, a warning is issued and (0, 0, 0) returned.
      */
      agx::Vec3 getBeginPosition() const;

      /**
      Current end position, in world coordinate frame, of this (initialized) beam. Shortcut for:
          beam->getSegments().back()->getEndPosition()
      \return the end position of this initialized beam, in world coordinate frame. If this
              beam hasn't been initialized, a warning is issued and (0, 0, 0) returned.
      */
      agx::Vec3 getEndPosition() const;

      /**
      \return the segments of this beam when initialized, otherwise an empty range is returned
      */
      BeamSegmentRange getSegments() const;

      /**
      \return the number of segments in this beam, zero if not initialized
      */
      agx::UInt getNumSegments() const;

      /**
      Finds the segment overlapping the given rest length from start. If \p restLengthFromStart
      is negative, the first segment is returned and if \p restLengthFromStarts is larger than
      the total rest length of this beam, the last segment is returned. If this beam hasn't been
      initialized, nullptr is returned.
      \param restLengthFromStart - rest length along this beam, from start, to find the segment
      \return segment at the given rest length from start, nullptr if this beam hasn't been initialized
      */
      BeamSegment* getSegment( agx::Real restLengthFromStart ) const;

      /**
      \return segment range where each segment is constrained with the previous segment,
              (segment->getConstraint() != nullptr), i.e., this range contains all segments
              of the beam except the first one
      */
      BeamSegmentRange getConstrainedSegments() const;

      /**
      \return true if this beam is initialized, otherwise false
      */
      agx::Bool isInitialized() const;

      /**
      Initialize this beam, creating segments, constraints and assigning properties related to the
      model of this beam. This method will be called when this beam is added to a simulation, if
      not explicitly called before. This method returns true when this beam wasn't previously
      initialized and is now successfully initialized. If this beam already is initialized or
      is il-configured, false is returned.
      \return true if successfully initialized, otherwise false
      */
      agx::Bool initialize();

      /**
      Propagate model specific properties, such as stiffness in the constraints. This method is
      always called before the system is solved and when this beam has been successfully
      initialized.
      */
      void updateProperties();

      /**
      Utility method to attach a given segment (of this beam) at a local segment position to
      another rigid body and optionally adopt the material properties of the model of this beam.
      \note The returned lock joint won't be added to the simulation or to this beam and this beam
            has to be initialized.
      \param segment - segment of this beam to attach
      \param localSegmentPosition - beam segment attachment position given in local segment frame
      \param otherRb - other rigid body to attach \p segment to - nullptr is world
      \param adoptModelProperties - calculate compliance and damping of the attachment given the
                                    beam model of this beam
      \return lock joint if valid, otherwise nullptr (e.g., \p segment is null or not part of this beam)
      */
      agx::LockJointRef attach( BeamSegment* segment,
                                agx::Vec3 localSegmentPosition,
                                agx::RigidBody* otherRb = nullptr,
                                agx::Bool adoptModelProperties = false ) const;

      /**
      Utility method to attach a given segment (of this beam) at begin or end (of the segment) to
      another rigid body and optionally adopt the material properties of the model of this beam.
      \note The returned lock joint won't be added to the simulation or to this beam and this beam
            has to be initialized.
      \param segment - segment of this beam to attach
      \param atSegmentBegin - attached at local (0, 0, 0) if true, otherwise (0, 0, length) if false
      \param otherRb - other rigid body to attach \p segment to - nullptr is world
      \param adoptModelProperties - calculate compliance and damping of the attachment given the
                                    beam model of this beam
      \return lock joint if valid, otherwise nullptr (e.g., \p segment is null or not part of this beam)
      */
      agx::LockJointRef attach( BeamSegment* segment,
                                agx::Bool atSegmentBegin,
                                agx::RigidBody* otherRb = nullptr,
                                agx::Bool adoptModelProperties = false ) const;

      /**
      Utility method to attach the first segment of this beam at begin (of the segment) to
      another rigid body and optionally adopt the material properties of the model of this beam.
      \note The returned lock joint won't be added to the simulation or to this beam and this beam
            has to be initialized.
      \param otherRb - other rigid body to attach first segment to - nullptr is world
      \param adoptModelProperties - calculate compliance and damping of the attachment given the
                                    beam model of this beam
      \return lock joint if valid, otherwise nullptr (e.g., \p segment is null or not part of this beam)
      */
      agx::LockJointRef attachBegin( agx::RigidBody* otherRb = nullptr, agx::Bool adoptModelProperties = false ) const;

      /**
      Utility method to attach the last segment of this beam at end (of the segment) to
      another rigid body and optionally adopt the material properties of the model of this beam.
      \note The returned lock joint won't be added to the simulation or to this beam and this beam
            has to be initialized.
      \param otherRb - other rigid body to attach first segment to - nullptr is world
      \param adoptModelProperties - calculate compliance and damping of the attachment given the
                                    beam model of this beam
      \return lock joint if valid, otherwise nullptr (e.g., \p segment is null or not part of this beam)
      */
      agx::LockJointRef attachEnd( agx::RigidBody* otherRb = nullptr, agx::Bool adoptModelProperties = false ) const;

      /**
      Calculates per segment stiffness dependent on beam model moment of inertia, cross section area,
      beam segment length and number of segments in this beam.
      \return the calculated translational and rotational stiffness and damping times
      */
      BeamStiffnessDamping calculateSegmentStiffnessDamping() const;

    public:
      /**
      \return the rest length of this beam
      */
      virtual agx::Real getRestLength() const override;

    public:
      DOXYGEN_START_INTERNAL_BLOCK()

      AGXSTREAM_DECLARE_SERIALIZABLE( agxModel::Beam );

      virtual void storeLightData( agxStream::StorageStream& str ) const override;
      virtual void restoreLightData( agxStream::StorageStream& str ) override;

    protected:
      Beam();
      virtual ~Beam();

      virtual void onAddNotification( agxSDK::Simulation* simulation ) override;
      virtual void onRemoveNotification( agxSDK::Simulation* simulation ) override;

      virtual void onPreStep() override;

    private:
      enum StateEnum : agx::UInt32
      {
        INITIALIZED = 1 << 0,
      };
      using State = agx::BitState<StateEnum, agx::UInt32>;

    private:
      agx::Bool validateMembersAndWarn( agx::String context ) const;
      void restoreLightData( agxStream::StorageStream& str, agx::Bool& valid );

      DOXYGEN_END_INTERNAL_BLOCK()

    private:
      BeamModelRef m_model;
      agx::UInt m_resolution;
      agx::Real m_length;
      State m_state;
  };

  inline Beam* Beam::get( agx::RigidBody* rb )
  {
    return LinkedStructure::getT<Beam>( rb );
  }

  inline const Beam* Beam::get( const agx::RigidBody* rb )
  {
    return LinkedStructure::getT<Beam>( rb );
  }
}
