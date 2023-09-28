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

#include <agxModel/export.h>

#include <agxSDK/LinkedSegment.h>

namespace agxModel
{
  AGX_DECLARE_POINTER_TYPES( BeamSegment );

  /**
  Beam segment in an agxModel::Beam containing a rigid body, a constraint and is of fixed length.
  Note that the first segment in the beam doesn't have a constraint, the constraints of a beam
  is defined with previous segment.

  The geometry of a beam segment starts at local (0, 0, 0) and ends at (0, 0, length). Width is
  between (0, -width/2, 0) to (0, width/2) and height is between (-height/2, 0, 0) and (height/2, 0, 0).
  I.e., local x is height, y is width and z is length.
  */
  class AGXMODEL_EXPORT BeamSegment : public agxSDK::LinkedSegment
  {
    public:
      /**
      \param rb - rigid body instance
      \return beam segment for \p rb if \p rb belongs to a beam
      */
      static BeamSegment* get( agx::RigidBody* rb );

      /**
      \param rb - rigid body instance
      \return beam segment for \p rb if \p rb belongs to a beam
      */
      static const BeamSegment* get( const agx::RigidBody* rb );

    public:
      /**
      Construct given transform of the rigid body and half extents. An empty
      rigid body will be created with the given transform.
      \param transform - initial transform of this segment
      \param halfExtents - half extents of this segment, defining length, width
                           and thickness of the linked structure
      */
      using agxSDK::LinkedSegment::LinkedSegment;

      /**
      \return the beam instance this segment belongs to
      */
      class Beam* getBeam() const;

    public:
      DOXYGEN_START_INTERNAL_BLOCK()

      AGXSTREAM_DECLARE_SERIALIZABLE( agxModel::BeamSegment );

    protected:
      virtual ~BeamSegment();

      DOXYGEN_END_INTERNAL_BLOCK()
  };

  inline BeamSegment* BeamSegment::get( agx::RigidBody* rb )
  {
    return agxSDK::LinkedSegment::getT<BeamSegment>( rb );
  }

  inline const BeamSegment* BeamSegment::get( const agx::RigidBody* rb )
  {
    return agxSDK::LinkedSegment::getT<BeamSegment>( rb );
  }
}
