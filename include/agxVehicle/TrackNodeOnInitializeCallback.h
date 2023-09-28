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

#include <agxVehicle/TrackNode.h>
#include <agxVehicle/TrackWheel.h>

namespace agxVehicle
{
  AGX_DECLARE_POINTER_TYPES( TrackNodeOnInitializeCallback );

  /**
  Track node initialize callback, called when a track node has been created
  and is about to be added to a track. Default behavior of onInitialize is
  to add a box given the size of the node. Override onInitialize to assign
  different geometry/geometries to the node.

  To enable receiving callbacks from the track, the Track::initialize method
  has to be called explicitly.
  */
  class AGXVEHICLE_EXPORT TrackNodeOnInitializeCallback : public agx::Referenced
  {
    public:
      /**
      Callback from the track when a node has been created and is about
      to be added to the track. Default behavior of this method is to
      create a box given the size of the node.
      */
      virtual void onInitialize( const TrackNode& node );
  };

  struct AGXVEHICLE_EXPORT TrackDesc
  {
    TrackDesc( agx::UInt numberOfNodes,
               agx::Real width,
               agx::Real thickness,
               agx::Real initialTensionDistance = agx::Real( 0 ) );

    agx::UInt numberOfNodes;
    agx::Real width;
    agx::Real thickness;
    agx::Real initialTensionDistance;
  };

  struct AGXVEHICLE_EXPORT TrackWheelDesc
  {
    TrackWheelDesc( agxVehicle::TrackWheel::Model model,
                    agx::Real radius,
                    agx::AffineMatrix4x4 rbTransform,
                    agx::AffineMatrix4x4 rbRelTransform );

    agxVehicle::TrackWheel::Model model;
    agx::Real radius;
    agx::AffineMatrix4x4 rbTransform;
    agx::AffineMatrix4x4 rbRelTransform;
  };

  struct AGXVEHICLE_EXPORT TrackNodeDesc
  {
    agx::Vec3 halfExtents;
    agx::AffineMatrix4x4 transform;
  };

  using TrackNodeDescVector = agx::Vector<TrackNodeDesc>;
  using TrackWheelDescVector = agx::Vector<TrackWheelDesc>;

  AGXVEHICLE_EXPORT TrackNodeDescVector findTrackNodeConfiguration( TrackDesc trackDesc, TrackWheelDescVector wheelsDesc );
}

