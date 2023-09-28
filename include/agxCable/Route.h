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

#ifndef AGXCABLE_ROUTE_H
#define AGXCABLE_ROUTE_H

#include <agx/Referenced.h>

#include <agxStream/Serializable.h>

#include <agxCable/Node.h>
#include <agxCable/CableSegment.h>
#include <agxCable/RouteInitializationReport.h>


// Deformable1D classes are used solely for reading old serializations.
#include <agxModel/Deformable1DRoute.h>

namespace agxCable
{
  AGX_DECLARE_POINTER_TYPES(HomogeneousSegmentsRoute);

  /**
  An ordered collection of agxCable::Node objects that has been added to a
  particular cable. Base class for segmentation algorithm implementations.
  */
  class AGXCABLE_EXPORT HomogeneousSegmentsRoute : public agx::Referenced, public agxStream::Serializable
  {
    public:
      bool add(agxCable::RoutingNode* node);

      void setMaxAllowedError(agx::Real error);
      agx::Real getMaxAllowedError() const;

      agx::Real getResolution() const;

      const agxCable::RoutingNodeRefVector& getRouteNodes() const;
      const agxCable::CableSegmentRefVector& getSegments() const;

      virtual agxCable::RouteInitializationReport initialize() = 0;
      virtual agxCable::RouteInitializationReport tryInitialize() = 0;

      AGXSTREAM_DECLARE_ABSTRACT_SERIALIZABLE(agxCable::Route);
      virtual void store(agxStream::OutputArchive& out) const override;
      virtual void restore(agxStream::InputArchive& in) override;

    protected:
      HomogeneousSegmentsRoute();
      HomogeneousSegmentsRoute(agx::Real resolution, agx::Real maxAllowedError);
      HomogeneousSegmentsRoute(agx::Real resolution, agx::Real originalResolution, agx::Real maxAllowedError,
                               agxCable::RoutingNodeRefVector&& routingNodes,
                               agxCable::CableSegmentRefVector&& segments);

      virtual ~HomogeneousSegmentsRoute();

    protected:
      agxCable::RoutingNodeRefVector m_routingNodes;
      agxCable::CableSegmentRefVector m_segments;

      agx::Real m_resolution; ///< The current resolution, updated by running the segmentation algorithm.
      agx::Real m_originalResolution; ///< The resolution supplied to the constructor. Resolution search always start here.
      agx::Real m_maxAllowedError;
  };


  AGX_DECLARE_POINTER_TYPES(IdentityRoute);

  /**
  A route that creates segments directly from the routing nodes. The routing
  nodes become the segments.
  */
  class AGXCABLE_EXPORT IdentityRoute : public HomogeneousSegmentsRoute
  {
    public:
      explicit IdentityRoute(agx::Real resolution);
      virtual agxCable::RouteInitializationReport initialize() override;
      virtual agxCable::RouteInitializationReport tryInitialize() override;

      AGXSTREAM_DECLARE_SERIALIZABLE(agxCable::IdentityRoute);

    protected:
      IdentityRoute();
      ~IdentityRoute();
  };


  AGX_DECLARE_POINTER_TYPES(SegmentingRoute);

  /**
  A route that finds a segment length that divides the route legs as evenly as
  possible. Routing nodes become a subset of the cable segments, and their
  positions are strictly adhered to.
  */
  class AGXCABLE_EXPORT SegmentingRoute : public HomogeneousSegmentsRoute
  {
    public:
      /**
      \param startingResolution - The starting point for cable resolution search.
      */
      explicit SegmentingRoute(agx::Real startingResolution);

      virtual agxCable::RouteInitializationReport initialize() override;
      virtual agxCable::RouteInitializationReport tryInitialize() override;

    public:
      SegmentingRoute(agx::Real resolution, agx::Real originalResolution, agx::Real maxAllowedError,
                      agxCable::RoutingNodeRefVector&& routingNodes,
                      agxCable::CableSegmentRefVector&& segments);
      AGXSTREAM_DECLARE_SERIALIZABLE(agxCable::SegmentingRoute);
      using HomogeneousSegmentsRoute::restore;

    protected:
      SegmentingRoute();
      ~SegmentingRoute();
  };


  /**
  A route that creates a path roughly following the routing nodes. Routing node
  positions are not strictly adhered to, and routing nodes are not included in
  the final cable.
  */
  class AGXCABLE_EXPORT PathRoute : public HomogeneousSegmentsRoute
  {
    public:
      explicit PathRoute(agx::Real targetResolution);
      virtual agxCable::RouteInitializationReport initialize() override;
      virtual agxCable::RouteInitializationReport tryInitialize() override;

    public:
      AGXSTREAM_DECLARE_SERIALIZABLE(agxCable::PathRoute);
      using HomogeneousSegmentsRoute::restore;

    protected:
      PathRoute();
      ~PathRoute();

    private:
      agx::Real m_targetResolution;
  };

  /**
  Legacy route type that exist solely for the purpose of restoring old serializations.
  */
  class AGXCABLE_EXPORT Route : public agxModel::Deformable1DPointRoute
  {
    public:
      /**
      Create a new empty cable route with the given peferred resolution.
      \param resolutionPerUnitLength The starting point for cable resolution search.
      */
      explicit Route(agx::Real resolutionPerUnitLength);

    public:
      AGXSTREAM_DECLARE_SERIALIZABLE(agxCable::Route);

    protected:
      Route();
      virtual ~Route();
  };
}

#endif
