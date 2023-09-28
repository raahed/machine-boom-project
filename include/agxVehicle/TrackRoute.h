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

#include <agxUtil/ConvexHull2D.h>

namespace agxVehicle
{
  class Track;
  class TrackWheel;

  AGX_DECLARE_POINTER_TYPES( TrackRoute );

  /**
  Track route object to initialize tracks.
  */
  class AGXVEHICLE_EXPORT TrackRoute : public agx::Referenced, public agxStream::Serializable
  {
    public:
      /**
      Wheel <-> node interactions after initialize.
      */
      struct WheelInteraction
      {
        TrackWheel* wheel;
        TrackNode* node;
      };
      using WheelInteractionContainer = agx::Vector<WheelInteraction>;

      using Curve        = agxUtil::ConvexHull2D::Curve;
      using SegmentPoint = agxUtil::ConvexHull2D::Curve::SegmentPoint;

      /**
      Resulting curve and resolution.
      */
      struct SegmentationResult
      {
        SegmentationResult( agx::UInt numberOfSegments )
          : curve(), solution( numberOfSegments ),
            successful( false ), error( agx::Infinity ),
            tolerance( agx::Real( 0 ) )
        {
        }

        Curve curve;
        Curve::SegmentationResult solution;
        agx::Bool successful;
        agx::Real error;
        agx::Real tolerance;
      };

    public:
      /**
      Construct given number of nodes, size and initial tension distance.
      \param numberOfNodes - number of nodes
      \param width - width of the track (this is an implicit definition, the geometry may be any size)
      \param thickness - thickness of the track (this is an implicit definition, the geometry may be any size)
      \param initialDistanceTension - value (distance) of how much shorter each node should be which causes tension in the
                                      system of tracks and wheels. Ideal case
                                        track_tension = initialDistanceTension * track_constraint_compliance.
                                      Since contacts and other factors are included it's not possible to know
                                      the exact tension after the system has been created.
      */
      TrackRoute( agx::UInt numberOfNodes, agx::Real width, agx::Real thickness, agx::Real initialDistanceTension = agx::Real( 0 ) );

      /**
      \return copy/clone of this object
      */
      TrackRouteRef clone() const;

      /**
      \return the number of nodes in the track
      */
      agx::UInt getNumberOfNodes() const;

      /**
      \return the width of the track
      */
      agx::Real getNodeWidth() const;

      /**
      \return the thickness of the track
      */
      agx::Real getNodeThickness() const;

      /**
      \note This value is not given, it's calculated during initialize.
      \return the length of each node
      */
      agx::Real getNodeLength() const;

      /**
      \return the initial distance tension of the track at creation time.
      */
      agx::Real getInitialDistanceTension() const;

      /**
      \return wheel <-> node interactions found during initialize
      */
      const WheelInteractionContainer& getWheelInteractions() const;

      DOXYGEN_START_INTERNAL_BLOCK()
      /**
      */
      SegmentationResult solve( const Track& track,
                                agx::UInt numNodes,
                                agx::Real nodeThickness,
                                agx::Real perSegmentTolerance = agx::Real( 1.0E-6 ),
                                agx::Real curveSegmentationTolerance = agx::Real( 1.0E-6 ) ) const;
      DOXYGEN_END_INTERNAL_BLOCK()

    public:
      /**
      Calculates the initial node configuration given the track wheels.
      \param track - reference track
      \return list of nodes with nodes.size == getNumberOfNodes() if successful
      */
      virtual const agxSDK::LinkedSegmentContainer& initialize( const Track& track );

      /**
      Post initialize callback to clear some internal data.
      */
      virtual void postInitialize();

    public:
      AGXSTREAM_DECLARE_SERIALIZABLE( agxVehicle::TrackRoute );

    protected:
      TrackRoute();
      ~TrackRoute();

    private:
      void add( TrackNodeRef node );

    private:
      agx::UInt m_numberOfNodes;
      agx::Vec3 m_nodeHalfExtents;
      agx::Real m_initialDistanceTension;
      agxSDK::LinkedSegmentContainer m_nodes;
      WheelInteractionContainer m_wheelInteractions;
  };
}
