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

#ifndef AGXMODEL_DEFORMABLE1DROUTE_H
#define AGXMODEL_DEFORMABLE1DROUTE_H

#include <agxModel/Deformable1DNode.h>

namespace agxModel
{
  AGX_DECLARE_POINTER_TYPES(Deformable1DRoute);
  AGX_DECLARE_POINTER_TYPES(Deformable1DPointRoute);
  AGX_DECLARE_POINTER_TYPES(Deformable1DMinErrorPointRoute);

  /**
  Initialization report kept in Deformable1D object for the user
  to check validity of the initial state.
  */
  class AGXMODEL_EXPORT Deformable1DInitializationReport
  {
    public:
      /**
      Create a report with values indication that no initialization has happened
      yet, or that the initialization failed.

      \p Deformable1DInitializationReport::successful will return false for this
      initialization report.
      */
      Deformable1DInitializationReport();

      /**
      Construct given target (minimum) error and the actual (maximum) error from route.
      */
      Deformable1DInitializationReport(agx::Real targetError, agx::Real maxError, agx::Real resolution, size_t numNodes);

      /**
      \return true if maximum error is less than or equal to the target error
      */
      agx::Bool successful() const;

      /**
      \return the target error
      */
      agx::Real getTargetError() const;

      /**
      \return the maximum error after routing
      */
      agx::Real getMaximumError() const;

      /**
      \return The number of nodes in the initialized route. This is equal to the number of segments.
      */
      size_t getNumNodes() const;

      /**
      \return The resolution of the routed path.
      */
      agx::Real getResolution() const;

      /**
      \return The total length of the created segments. May be different from
              the total distances between the routing node positions due to
              divisibility.
      */
      agx::Real getLength() const;

      /**
      Store to output archive.
      */
      void store(agxStream::OutputArchive& out) const;

      /**
      Restore from input archive.
      */
      void restore(agxStream::InputArchive& in);

    private:
      agx::Real m_targetError;
      agx::Real m_maxError;
      agx::Real m_resolution;
      size_t m_numNodes;
  };

  /**
  Base class for routing algorithms holding all nodes and resolution.
  */
  class AGXMODEL_EXPORT Deformable1DRoute : public agx::Referenced, public agxStream::Serializable
  {
    public:
      /**
      Initialize call from the Deformable1D implementation expecting a report
      how the routing went.
      \return route report from routing algorithm
      */
      agxModel::Deformable1DInitializationReport initialize();

      agxModel::Deformable1DInitializationReport tryInitialize();

      /**
      Called after the Deformable1D has been successfully initialized.
      */
      virtual void postInitialize();

      /**
      All nodes from routing - these nodes are only kept during initialization stage.
      \return all the nodes created during initialize
      */
      const agxModel::Deformable1DNodeRefContainer& getNodes() const;

      /**
      \return the resolution per unit length to be used during routing
      */
      agx::Real getResolutionPerUnitLength() const;

      AGXSTREAM_DECLARE_ABSTRACT_SERIALIZABLE( agxModel::Deformable1DRoute );

    protected:
      /**
      Construct given resolution per unit length.
      \param resolutionPerUnitLength - resolution per unit length
      */
      Deformable1DRoute( agx::Real resolutionPerUnitLength );

      /**
      Reference counted object, protected destructor.
      */
      virtual ~Deformable1DRoute();


      virtual agxModel::Deformable1DInitializationReport tryInitializeAllNodes() = 0;

      /**
      Pure virtual method where the implementation should execute its routing
      algorithm and add the nodes for final creation.
      */
      virtual agxModel::Deformable1DInitializationReport initializeAndAddAllNodes() = 0;

      /**
      Add-node-method for implementations. The \p node has to be unique and will be
      used in the final Deformable1D.
      \param node - node to add
      \param length - length of the node
      \return true if the node was added (false if null or already added)
      */
      agx::Bool add( agxModel::Deformable1DNodeRef node, agx::Real length );

      /**
      Possibility for implementations to change the resolution.
      \param resolutionPerUnitLength - new resolution per unit length
      */
      void setResolutionPerUnitLength( agx::Real resolutionPerUnitLength );

      /**
      Store given output archive.
      */
      void store( agxStream::OutputArchive& out ) const override;

      /**
      Restore given input archive.
      */
      void restore( agxStream::InputArchive& in ) override;

    private:
      Deformable1DNodeRefContainer m_allNodes;
      agx::Real m_resolutionPerUnitLength;
  };

  /**
  Routing implementation where the user add 'points', i.e., orientation is neglected.
  */
  class AGXMODEL_EXPORT Deformable1DPointRoute : public agxModel::Deformable1DRoute
  {
    public:
      /**
      Construct given wanted resolution per unit length and minimum wanted error from routing.

      The resolution will be changed to fit the length of the route and if \p minError is finite,
      an iterative process will try to find the optimal resolution constrained by this given error.
      \param resolutionPerUnitLength - wanted resolution per unit length
      \param minError - minimum error tolerance after routing. Pass a negative value for automatic. (default: negative)
      \note Be careful with \p minError. For complex routes (with many points) and with minimum error
            set low, there's no limitation of how high the resolution may be.
      */
      Deformable1DPointRoute( agx::Real resolutionPerUnitLength, agx::Real minError = agx::Real(-1) );

      void setMinError(agx::Real minError);
      agx::Real getMinError() const;

      /**
      Add node to the route.
      \param node - node to add to the route
      \return true if added, false if invalid or already in this route
      */
      agx::Bool add( agxModel::Deformable1DNodeRef node );

      /**
      Add free world point to the route. The Deformable1D will pass through this point after initialize.
      \param point - point in world where the Deformable1D should pass through after initialize
      \return route node if valid, otherwise null
      */
      agxModel::Deformable1DNodeRef add( const agx::Vec3& point );

      /**
      Add attachment to the route. The Deformable1D will pass through this attachment after initialize.
      \param attachment - attachment to add
      \return route node with \p attachment if valid, otherwise null
      */
      agxModel::Deformable1DNodeRef add( agxModel::NodeAttachmentRef attachment );

      /**
      \return the added route nodes
      */
      const agxModel::Deformable1DNodeRefContainer& getRouteNodes() const;


      /**
      \return The resolution with which this route was created.
      */
      agx::Real getOriginalResolutionPerUnitLength() const;


      AGXSTREAM_DECLARE_SERIALIZABLE( agxModel::Deformable1DPointRoute );

    protected:
      /**
      Default constructor for serialization.
      */
      Deformable1DPointRoute();

      /**
      Reference counted object, protected destructor.
      */
      virtual ~Deformable1DPointRoute();


      virtual agxModel::Deformable1DInitializationReport tryInitializeAllNodes() override;

      /**
      Initialize call from base.
      */
      virtual agxModel::Deformable1DInitializationReport initializeAndAddAllNodes() override;

    protected:
      agx::Real m_minError;
      agx::Real m_originalResolutionPerUnitLength;
      Deformable1DNodeRefContainer m_routeNodes;
  };

  /**
  Route the initial setup given nodes. This route assumes
  that the route algorithm is handled elsewhere, i.e.,
  the added nodes and the final Deformable1D nodes is
  one to one (identical, even the pointers).
  */
  class AGXMODEL_EXPORT Deformable1DNodeRoute : public agxModel::Deformable1DRoute
  {
    public:
      struct AGXMODEL_EXPORT RouteData
      {
        typedef agx::Vector< RouteData > Container;

        RouteData( agxModel::Deformable1DNodeRef node, agx::Real length ) : node( node ), length( length ) {}
        agxModel::Deformable1DNodeRef node;
        agx::Real length;
      };

    public:
      /**
      Default constructor.
      */
      Deformable1DNodeRoute();

      /**
      Add node to route.
      \param node - node to add
      \param length - the length of the node (segment)
      \return same node as the one added
      */
      agxModel::Deformable1DNodeRef add( agxModel::Deformable1DNodeRef node, agx::Real length );

      /**
      \return the route data container containing the added nodes
      */
      const RouteData::Container& getRouteDataContainer() const;

      AGXSTREAM_DECLARE_SERIALIZABLE( agxModel::Deformable1DNodeRoute );

    protected:
      /**
      Default constructor for serialization.
      */
      virtual ~Deformable1DNodeRoute();

      /**
      Initializes.
      */
      virtual agxModel::Deformable1DInitializationReport initializeAndAddAllNodes() override;

      virtual agxModel::Deformable1DInitializationReport tryInitializeAllNodes() override;

    protected:
      RouteData::Container m_routeNodes;
  };
}

#endif // AGXMODEL_DEFORMABLE1DROUTE_H
