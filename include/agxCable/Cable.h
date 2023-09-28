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

#include <agxSDK/LinkedStructure.h>

#include <agxCable/export.h>
#include <agxCable/CableProperties.h>
#include <agxCable/CableIterator.h>
#include <agxCable/Route.h>

namespace agxSDK
{
  class Simulation;
}

namespace agxCable
{
  class RoutingNode;


  AGX_DECLARE_POINTER_TYPES(Cable);
  AGX_DECLARE_VECTOR_TYPES(Cable);

  /**
  A model of a cable using a lumped elements model. A number of RigidBodies,
  called segments, are created and constrained together to represent the cable.
  */
  class AGXCABLE_EXPORT Cable : public agxSDK::LinkedStructure
  {
    public:

      /**
      Create a new cable with the given radius and resolution. The given
      resolution is used as a starting point, but particulars of the routing
      and initialization may cause the final resolution to differ from the
      given one. See \p tryInitialize.

      At creation the cable will have zero length and no position in space.
      Route the cable by adding nodes to it.

      \param radius - The radius of the cable.
      \param resolutionPerUnitLength - The number of lumped elements to create per unit length.
      */
      Cable(agx::Real radius, agx::Real resolutionPerUnitLength);


      /**
      Create a new cable using the given routing algorithm. The routing
      algorithm determines final segment placement when the cable is
      initialized.

      The cable is routed using \p Cable::add(RoutingNode*) as usual. Different
      routing algorithm may impose various limitations on how how the cable may
      be routed.

      \param radius - The radius of the cable.
      \param routingAlgorithm - The routing algorithm to use when creating cable segments from the routing nodes.
      */
      Cable(agx::Real radius, agxCable::HomogeneousSegmentsRoute* routingAlgorithm);

      /**
      Add the given cable node to the end of the cable. The cable is extended
      from the most recently added node to the new node. The cable, by
      definition, starts at the first node added and ends at the last node.

      All modes must be added before the cable initialization is finalized. This
      happens when the cable is added to the simulation.

      \param node - The node to add to the cable route.
      \return True if the node was added. Will return false if the cable has already been initialized.
      */
      agx::Bool add(agxCable::RoutingNode* node);

      /**
      Make a dry-run of the cable initialization. This will produce a report
      detailing the configuration of the cable that would be produced if added
      to the simulation right now. From this report one can read the routing
      error and the number of segments created.

      The resolution of the cable is updated to match the resolution found by
      the test-initialization and can be queried using \p getResolution().

      \return Report detailing the initialization results.
      */
      agxCable::RouteInitializationReport tryInitialize();

      /**
      \return A report describing the result of the initialization.
      */
      agxCable::RouteInitializationReport getInitializationReport() const;


      /**
      Make a dry-run with a specified maximum allowed routing error. The error
      is the amount of overlap or gap between consecutive segments at
      initialization. The initialization process tries to find a resolution
      that produces an error that is below the given limit. This is not always
      possible.

      If the cable is initialized without specifying a maximum error then an
      error limit is estimated from the route.
      */
      agxCable::RouteInitializationReport tryInitialize(agx::Real maxError);

      /**
      \return A range spanning over all the segments in this cable.
      */
      agxCable::SegmentRange getSegments() const;

      /**
      Returns the number of segments in the cable. Only valid after the cable
      has been fully initialized, i.e., added to a agxSDK::Simulation.

      Results from the dry-runs performed by tryInitialize is not seen here.
      Inspect the report returned from \p tryInitialize instead.

      \return The number of segments in the cable.
      */
      agx::UInt getNumSegments() const;

      /**
      Returns the resolution of the cable, measured in units of length per
      segment. This value is updated by both the cable initialization and the
      dry-runs performed by \p tryInitialize.
      */
      agx::Real getResolution() const;


      /**
      \return The radius of this cable.
      */
      agx::Real getRadius() const;

      /**
      Get the rest length of the cable. For an initialized cable this is the sum
      of the segment lengths. For an uninitialized cable it is the length of the
      path created by the routing nodes added so far.

      \return The rest length of the cable.
      */
      virtual agx::Real getRestLength() const override;

      /**
      Get the current length of the cable, including any stretching. The current
      length of an uninitialized cable is the length of the path created by the
      routing nodes added so far.

      \return The current length of the cable.
      */
      virtual agx::Real getCurrentLength() const override;

      /**
      Returns an iterator that walks over the segments of the cable. From the
      iterator one can read information such as segment length and position,
      cable tension and get access to the underlying RigidBody.
      */
      agxCable::CableIterator begin() const;
      agxCable::CableIterator begin();

      /**
      Returns an iterator pointing one-past the last segment. This iterator may
      not be dereferenced.
      */
      agxCable::CableIterator end() const;
      agxCable::CableIterator end();

      /**
      Returns an iterator pointing to the last segment.
      */
      agxCable::CableIterator findLast() const;

      /**
      Return the cable properties for this cable. CableProperties may be shared
      among several cables.
      */
      agxCable::CableProperties* getCableProperties() const;

      /**
      Set the cable properties object used by this cable. This will cause this
      cable to share properties with the cable from which the cable properties
      object was fetched.
      */
      void setCableProperties(agxCable::CableProperties* cableProperties);

      /**
      Attach the given segment to the given rigid body using a BallJoint that
      lets the cable rotate freely around the point. The relative translate is
      relative to the body's model coordinate frame.

      \param segment - CableIterator pointing to the segment that is to be attached.
      \param body - RigidBody to attach the segment to.
      \param relativeTranslate - Point on the body that the cable is to be attached to.
      \return True if the segment was successfully attached. False otherwise.
      */
      bool attach(agxCable::CableIterator segment, agx::RigidBody* body, const agx::Vec3& relativeTranslate);

      /**
      Attach the given segment to the given rigid body using a LockJoint. The
      cable will be fixed relative to the body. The relative transform describe
      the transformation from the body's model coordinate frame to the point
      that the cable should pass through. The cable will pass through this
      point along the transformed Z axis.

      \param segment - CableIterator pointing to the segment that is to be attached.
      \param body - RigidBody to attach the segment to.
      \param relativeTransform - Transformation describing the orientation of the cable relative to the body.
      \return True if the segment was successfully attached. False otherwise.
      */
      bool attach(agxCable::CableIterator segment, agx::RigidBody* body, const agx::AffineMatrix4x4& relativeTransform);


      static agxCable::Cable* asCable(agxSDK::Assembly* assembly);


      /**
      \return The cable that the given geometry is part of, or nullptr if there is no such cable.
      */
      static agxCable::Cable* getCableForGeometry(agxCollide::Geometry* geometry);

      static const agxCable::Cable* getCableForGeometry(const agxCollide::Geometry* geometry);

      /**
       * \return The cable that the given body is part of, or nullptr if there is no such cable.
       */
      static agxCable::Cable* getCableForBody(agx::RigidBody* body);
      static const agxCable::Cable* getCableForBody(const agx::RigidBody* body);

      /**
      This method will do a linear search in Simulation and return a pointer to the specified Cable.

      \param simulation - The instance of the simulation in which we are searching
      \param name - Name of the Cable we are searching for
      \return Pointer to the found Cable or null if not found.
      */
      static Cable* find(agxSDK::Simulation* simulation, const agx::Name& name);

      /**
      This method will do a linear search in Simulation and return a pointer to the specified Cable.

      \param simulation - The instance of the simulation in which we are searching
      \param uuid - uuid of the Cable we are searching for
      \returns pointer to the found Cable or null if not found.
      */
      static Cable* find(agxSDK::Simulation* simulation, const agx::Uuid& uuid);

      /**
      This method will do a linear search in Simulation and return a vector with pointers to all matching Cables.

      \param simulation - The instance of the simulation in which we are searching
      \param name - Name of the Cable's we are searching for
      \returns Vector containing pointers to all matching Cables.
      */
      static agxCable::CablePtrVector findAll(agxSDK::Simulation* simulation, const agx::Name& name);

      /**
       * Collect all Cables in the given simulation.
       * \param simulation - The simulation to get cables from.
       * \return Vector containing pointes to all cables in the simulation.
       */
      static agxCable::CablePtrVector getAll(agxSDK::Simulation* simulation);

      /**
      Load named CableProperties from the Material Library for the current Cable.
      \returns True if properties could be loaded
      */
      bool loadLibraryProperties(const agx::String& name);

      /**
      Returns a vector with all the named CableProperties that were found in the Material Library
      \returns A vector with the names of all found CableProperties in the MaterialLibrary
      */
      static agx::StringVector getAvailableLibraryProperties();


    DOXYGEN_START_INTERNAL_BLOCK()
    public:
      AGXSTREAM_DECLARE_SERIALIZABLE(agxCable::Cable);
    DOXYGEN_END_INTERNAL_BLOCK()

    protected:
      /// Used for serialization only.
      Cable();

      virtual ~Cable();

      /// \return The route defining the initial path of the cable.
      agxCable::HomogeneousSegmentsRoute* getRoute() const;

    protected:
      virtual void onAddNotification(agxSDK::Simulation* simulation) override;
      virtual void onPreStep() override;
      virtual void onPostStep() override;

    private:
      friend class CableSegment;
      void onAttachmentAdded(agxCable::CableSegment* segment, agxCable::SegmentAttachment* attachment);

    private:
      /// Restore from an archive that contains a LinkedStructure based cable.
      void restoreFromLinkedCable(agxStream::InputArchive& in);

      /// Restore from an archive that contains a Deformable1D based cable.
      void restoreFromDeformable1D(agxStream::InputArchive& in);

    private:
      enum StateEnum : agx::UInt32
      {
        INITIALIZED = 1 << 0, // After add notification and route successfully initialized.
      };
      using State = agx::BitState<StateEnum, agx::UInt32>;


    private:
      agx::Real m_radius;

      agxCable::HomogeneousSegmentsRouteRef m_initializationRoute;
      agxCable::RouteInitializationReport m_initializationReport;

      agxCable::CablePropertiesRef m_properties;

      State m_state;
  };


  // Needed by for-each loops.
  AGXCABLE_EXPORT agxCable::CableIterator begin(const agxCable::Cable& cable);
  AGXCABLE_EXPORT agxCable::CableIterator end(const agxCable::Cable& cable);
}
