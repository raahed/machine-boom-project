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

#include <agxVehicle/Wheel.h>
#include <agxVehicle/TrackNode.h>

#include <agx/MergedBody.h>

namespace agxVehicle
{
  AGX_DECLARE_POINTER_TYPES( TrackWheel );

  /**
  Wheel used in tracked vehicles.
  */
  class AGXVEHICLE_EXPORT TrackWheel : public agxVehicle::Wheel
  {
    public:
      enum Model
      {
        SPROCKET, /**< Sprocket - geared driving wheel with default property MERGE_NODES. */
        IDLER,    /**< Idler - geared non-powered wheel with default property MERGE_NODES. */
        ROLLER    /**< Roller - track return or road wheel. */
      };

      enum Property : agx::UInt32
      {
        MERGE_NODES    = 1 << 0, /**< Wheels with this property and nodes will merge to
                                      them, simulating geared, non-slip, type of properties. */
        SPLIT_SEGMENTS = 1 << 2, /**< Wheels with this property will split segments, i.e.,
                                      intermediate nodes that are merged. Use this property
                                      on ROLLER models where the track changes angle. */
        MOVE_NODES_TO_ROTATION_PLANE = 1 << 3, /**< If enabled - when a node is merged to the wheel, move the node
                                                    into the plane defined by the wheel center position and rotation
                                                    axis. This will prevent the tracks from sliding of its path but
                                                    all wheels with MERGE_NODES must be aligned. Default: Disabled */
        MOVE_NODES_TO_WHEEL = 1 << 4 /**< Similar to MOVE_NODE_TO_ROTATION_PLANE but this property will
                                          also make sure the node is moved to the surface of the wheel.
                                          Default: Disabled */
      };

    public:
#ifndef SWIG
      using ControlPointContainer = agx::Vec3Vector;
#else
      typedef agx::Vec3Vector ControlPointContainer;
#endif
      using Properties            = agx::BitState<Property, agx::UInt32>;

    public:
      /**
      Construct given property (wheel type), radius, rigid body and frame defining wheel center,
      up and rotation axes in the rigid body frame.

      Rotation axis is by definition the y axis in \p rbRelFrame (if given, otherwise
      y axis of \p rb).

      Up axis is by definition the z axis in \p rbRelFrame (if given, otherwise z axis
      of \p rb).
      \param model - wheel model
      \param radius - radius of the wheel
      \param rb - wheel rigid body
      \param rbRelFrame - frame defining center position, up and rotation axes in \p rb frame
      */
      TrackWheel( Model model,
                  agx::Real radius,
                  agx::RigidBody* rb,
                  agx::FrameRef rbRelFrame = nullptr );

      /**
      Construct given property (wheel type), radius, rigid body and transform defining wheel center,
      up and rotation axes in the rigid body frame.

      Rotation axis is by definition the y axis in \p rbRelFrame (if given, otherwise
      y axis of \p rb).

      Up axis is by definition the z axis in \p rbRelFrame (if given, otherwise z axis
      of \p rb).
      \param model - wheel model
      \param radius - radius of the wheel
      \param rb - wheel rigid body
      \param rbRelTransform - transform defining center position, up and rotation axes in \p rb frame
      */
      TrackWheel( Model model,
                  agx::Real radius,
                  agx::RigidBody* rb,
                  agx::AffineMatrix4x4 rbRelTransform );

      /**
      Creates a set of points defining the wheel circle, each length \p circleSegmentLength apart.
      The points are given in the world coordinate frame.
      \param circleSegmentLength - desired circle segment length between each point
      \return points defining the wheel circle, given in world coordinate frame
      */
      ControlPointContainer getControlPoints( agx::Real circleSegmentLength ) const;

      /**
      Creates a set of points defining the wheel circle, each length \p circleSegmentLength apart.
      The points are given in the world coordinate frame and will be added to \p container.
      \param circleSegmentLength - desired circle segment length between each point
      \param container - control points will be added to this container
      \return true if points were added - otherwise false
      */
      agx::Bool getControlPoints( agx::Real circleSegmentLength, ControlPointContainer& container ) const;

      /**
      \return true if the given point can be assumed to be a control point on this wheel
      */
      agx::Bool hasControlPoint( const agx::Vec3& point, const agx::Real epsilon = agx::REAL_SQRT_EPSILON ) const;

      /**
      \return the model of this track wheel
      */
      Model getModel() const;

      /**
      \return true if this track wheel model matches any of the arguments
      */
      template<typename... ModelTypes>
      agx::Bool isModel( ModelTypes... modelTypes ) const;

      /**
      \return the properties of this wheel
      */
      const Properties getProperties() const;

      /**
      Enable/disable a property for this wheel.
      \param property - property to enable/disable
      \param enable - true to enable the property, false to disable
      */
      void setEnableProperty( Property property, agx::Bool enable );

      /**
      \param property - wheel property
      \return true if the property is enabled - otherwise false
      */
      agx::Bool getEnableProperty( Property property ) const;

      /**
      \return merged body of this wheel - if something is merged to it
      */
      agx::MergedBody* getMergedBody() const;

      /**
      \return an already created instance or creates a new instance of a merged body
      */
      agx::MergedBody* getOrCreateMergedBody( agxSDK::Simulation* simulation );

      /**
      List of node indices merged to this wheel. The resulting list is sorted with
      lowest index first.
      */
      agx::UIntVector findMergedNodesIndices() const;

      /**
      Finds merged nodes indices and splits the list into ranges where the
      indices aren't consecutive. Note that if the first and the last node
      both are in the list the last node will be included in the first range.
      E.g., instead of [[0, 1, 2, 3] [78, 79]] the resulting range could be
      (given it's 80 nodes in the track) [78, 79, 0, 1, 2, 3],
      */
      IndexRangeContainer findMergedNodesIndexRanges() const;

      /**
      \return track instance this track wheel belongs to
      */
      agxVehicle::Track* getTrack() const;

    public:
      DOXYGEN_START_INTERNAL_BLOCK()

      AGXSTREAM_DECLARE_SERIALIZABLE( agxVehicle::TrackWheel );

      virtual void onAdd( Track* track );
      virtual void onRemove( Track* track );

      DOXYGEN_END_INTERNAL_BLOCK()

    protected:
      TrackWheel();
      virtual ~TrackWheel();

    private:
      Model m_model;
      Properties m_properties;
      agx::MergedBodyRef m_mergedBody;
      Track* m_track;
  };

  inline agx::MergedBody* TrackWheel::getMergedBody() const
  {
    return m_mergedBody;
  }

  template<typename... ModelTypes>
  inline agx::Bool TrackWheel::isModel( ModelTypes... modelTypes ) const
  {
    std::array<Model, sizeof... (ModelTypes)> args = { {modelTypes...} };
    for ( const auto arg : args )
      if ( arg == m_model )
        return true;
    return false;
  }

  inline const TrackWheel::Properties TrackWheel::getProperties() const
  {
    return m_properties;
  }

  inline agx::Bool TrackWheel::getEnableProperty( TrackWheel::Property property ) const
  {
    return m_properties.Is( property );
  }

  inline Track* TrackWheel::getTrack() const
  {
    return m_track;
  }
}
