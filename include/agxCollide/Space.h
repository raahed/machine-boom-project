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

#include <agxCollide/agxCollide.h>
#include <agx/HashVector.h>

#include <agx/SetVector.h>
#include <agx/Vector.h>
#include <agx/HashSet.h>
#include <string>

#include <agxCollide/Geometry.h>
#include <agxCollide/HierarchicalGrid.h>
#include <agxCollide/Contacts.h>
#include <agxCollide/DisabledCollisionsState.h>

#include <agxCollide/SweepAndPrune.h>
#include <agxCollide/GeometryPair.h>
#include <agxCollide/CollisionGroupManager.h>
#include <agxCollide/Sphere.h>
#include <agx/Task.h>
#include <agx/Timer.h>
#include <agx/Component.h>
#include <agx/HashTableComponent.h>

#include <agx/Physics/GeometryContactEntity.h>
#include <agx/Physics/ContactPointEntity.h>
#include <agx/Physics/ParticleGeometryContactEntity.h>
#include <agx/Physics/ParticlePairContactEntity.h>
#include <agxCollide/SpaceListener.h>


namespace agxIO
{
  class SceneExporter;
}

namespace agx
{
  class ParticleSystem;
}

namespace agxSDK
{
  class Simulation;
}

namespace agxCollide
{
  typedef agx::Vector<agx::Physics::BroadPhasePairPtr, agxData::BufferProxyAllocator> BroadPhasePairVector;
  typedef agx::Vector<agx::Physics::GeometryPairPtr, agxData::BufferProxyAllocator> SeparationPairVector;
  typedef agx::Vector<agx::Physics::GeometryPairPtr> LocalSeparationPairVector;
  typedef std::pair<agx::Physics::ParticlePtr, agx::Physics::GeometryPtr> ParticleGeometryPair;
  typedef agx::Vector<ParticleGeometryPair> ParticleGeometryPairVector;

  typedef agx::Vector<agx::Physics::ParticleGeometryContactPtr> ParticleGeometryContactVector;
  typedef agx::Vector<agx::Physics::ParticlePairContactPtr>     ParticlePairContactVector;

  AGX_DECLARE_POINTER_TYPES(Space);

  /**
  This class contains all Geometries and performs Broad Phase and Narrow Phase collision detection
  to calculate overlapping Geometry pairs.
  It also contains methods for explicit overlapping tests with rays etc.
  */
  class AGXPHYSICS_EXPORT Space : public agx::Component, public virtual agxStream::Serializable
  {
    public:

      /// Enum for specifying which BroadPhase algorithm space should use
      enum BroadPhaseAlgorithm {
        INVALID_BROAD_PHASE_ALGORITHM = -1,
        SWEEP_AND_PRUNE, /**< SAP good for scenes which are semi stationary. Good for temporal coherency. */
        HIERARCHICAL_GRID /**< Good for large scale scenes, particle systems where one can expect large motions between time steps. */
      };

      /**
      Default constructor
      */
      Space();

      /// Instantiate Space with a specific device
      Space(agx::Device* device);

      /**
      Cleanup the internal state of the object. All Geometries will be removed one by one.
      The state after calling this method should be the same as creating a new object.
      */
      void cleanup();

      /**
      Specify a new broad phase algorithm implementation
      \return false if the algorithm can't be changed, eg if using ParallelPGS solver, or particle system, which both require the HIERARCHICAL_GRID
      */
      bool setBroadPhaseAlgorithm(BroadPhaseAlgorithm algorithm);

      /**
      \return The active broad phase algorithm.
      */
      BroadPhaseAlgorithm getBroadPhaseAlgorithm() const;

      /**
      Insert a new geometry into the Space so it can collide with other geometries.
      \return true if geometry is added successfully. False if geometry was already added.
      */
      bool add( Geometry* geometry );

      /**
      Remove a geometry from Space.
      \param geometry - Pointer to a Geometry that will be removed
      \return true if geometry existed in space and was successfully removed
      */
      bool remove( Geometry* geometry );

      /**
      Find (linear search) the first Geometry in space with a matching name
      \param name - name of geometry to search for
      \return a pointer to the found Geometry, null if not found.
      */
      const Geometry* getGeometry( const agx::Name& name ) const;

      /**
      Find (linear search) the first Geometry in space with a matching name
      \param name - name of geometry to search for

      \return a pointer to the found Geometry, null if not found.
      */
      Geometry* getGeometry( const agx::Name& name );

      /**
      Find (linear search) the first Geometry in space matching the specified uuid
      \param uuid - uuid of geometry to search for

      \return a pointer to the found Geometry, null if not found.
      */
      const Geometry* getGeometry( const agx::Uuid& uuid ) const;

      /**
      Find (linear search) the first Geometry in space matching the specified uuid
      \param uuid - name of geometry to search for

      \return a pointer to the found Geometry, null if not found.
      */
      Geometry* getGeometry( const agx::Uuid& uuid );

      /**
      Find (linear search) the first Geometry in space matching the specified id
      \param id - id of geometry to search for

      \return a pointer to the found Geometry, null if not found.
      */
      const Geometry* getGeometry(agx::Index id) const;

      /**
      Find (linear search) the first Geometry in space matching the specified id
      \param id - name of geometry to search for

      \return a pointer to the found Geometry, null if not found.
      */
      Geometry* getGeometry(agx::Index id);

      /**
      \return vector of geometries which has been added to this space
      */
      const GeometryRefVector& getGeometries() const;

      /**
      Update the broad-phase state of the space.
      */
      void updateBroadPhase();

      /**
      Update the contact state of the entire space including Broad Phase and narrow phase.
      */
      void update();

      /**
      Update the narrow phase state, calculating contact data for all overlapping geometry pairs.
      i.e. check all current bounding box overlaps for possible contacts
      */
      void updateNarrowPhase();

      /**
      Access to the vector containing all narrow phase contacts.
      \return A reference to a vector with reference pointers to GeometryContact
      This vector contain pointers to GeometryContact which are valid during this simulation step only, so
      never store a pointer to them and use them over several time steps.
      */
      const GeometryContactPtrVector& getGeometryContacts() const;

      /**
      Extract a vector with all matching geometry contacts involving the two specified geometries.
      This method will perform a linear search through all contact points.

      \param geometry1 - The first specified geometry to search for among geometry contacts
      \param geometry2 - The second specified geometry to search for among geometry contacts (if null only match geometry1)
      \param matches - a vector with all matching geometry contact points
      This vector contain pointers to GeometryContact which are valid during this simulation step only, so
      never store a pointer to them and use them over several time steps.
      \return number of matching geometry contacts
      */
      size_t getGeometryContacts(GeometryContactPtrVector& matches, const agxCollide::Geometry *geometry1, const agxCollide::Geometry *geometry2 = nullptr) const;

      /**
      Extract a vector with all matching geometry contacts involving the two specified rigid bodies
      This method will perform a linear search through all contact points.

      \param body1 - The first specified rigid body to search for among geometry contacts
      \param body2 - The second specified rigid body to search for among geometry contacts (if null only match body1)
      \param matches - a vector with all matching geometry contact points
      This vector contain pointers to GeometryContact which are valid during this simulation step only, so
      never store a pointer to them and use them over several time steps.
      \return number of matching geometry contacts
      */
      size_t getGeometryContacts(GeometryContactPtrVector& matches, const agx::RigidBody *body1, const agx::RigidBody *body2 = nullptr) const;

      /**
      Remove a specified GeometryContact from the list of contacts.
      It is important that this iterator is valid, i.e. accessed during the same time step and that the list of contacts
      is not modified since it is acquired.
      \param contact - iterator taken from the getGeometryContacts() at the same time step.
      */
      GeometryContactPtrVector::const_iterator removeContact(GeometryContactPtrVector::const_iterator contact);

      /**
      Explicitly add geometry contacts.
      This must be done _after_ Space::update(), and _not_ from within a ContactEventListener;
      preferable via agxSDK::StepEventListener::pre(t).
      Contact materials are _not_ automatically calculated;
      they must be explicitly set for each contact.
      Contacts without a contact material will be ignored.
      Impact state for each LocalGeometryContact has to be set
        manually by using the 'solveImpact'-member (default: on).
      */
      void addGeometryContacts(const LocalGeometryContactVector& localContacts);

      /**
      Add a listener to space. The listener will be signaled at add/remove geometry events.
      */
      void addListener(agxCollide::SpaceListener* listener);

      /**
      Remove a listener from Space.
      \returns true if removal was successful
      */
      bool removeListener(agxCollide::SpaceListener* listener);

      /**
      Tests a number of geometries for bounding volume overlaps without changing the
      geometries already in space besides from updating bounding volumes.

    \param testGeometries - The vector with geometries that should be tested for overlaps.
      These may be rearranged to speed up testing.
      \param geometryPairs - The geometry pairs which have overlapping bounding volumes
      \param particleGeometryPairs - The particle-geometry pairs which have overlapping bounding volumes
      \param skip - If true, skip collisions between geometries that both are in
                  the testGeometries vector. Useful when some geometries are
                  already added to space.
      \return true if any overlaps were found.
      */
      bool testBoundingVolumeOverlap(
        GeometryPtrVector& testGeometries,
        GeometryPairVector& geometryPairs,
        ParticleGeometryPairVector& particleGeometryPairs,
        bool skip = true );

      /**
      Test if a number of geometries collides with something already present in
      space without changing space, besides from updating some bounding volumes.
      \param testGeometries The vector with geometries that should be tested for overlaps.
      These may be rearranged to speed up testing.

    \param geometryContacts - The geometry contacts that were found
      \param particleGeometryContacts - The particle-geometry contacts that were found
      \param skip - If true, skip collisions between geometries that both are in
                  the testGeometries vector. Useful when some geometries are
                  already added to space.
      \return True if any overlaps were found.
      */
      bool testGeometryOverlap(
        GeometryPtrVector& testGeometries,
        LocalGeometryContactVector& geometryContacts,
        LocalParticleGeometryContactVector& particleGeometryContacts,
        bool skip = true
      );

      /**
      Returns geometry contact between two geometries, if existing.
      \param g1 The first geometry.
      \param g2 The second geometry.
      \return Geometry contact between two geometries.
      If no matching geometry contact is found,
      an empty GeometryContact (GeometryContact()) will get returned.
      Thus, the return value should be checked for isValid() before being used.
      \note Works only for geometry contacts added by the
      general collision detection pipeline, not manually added ones;
      an alternative in that case is to use getGeometryContacts()).
      */
      GeometryContact getContact(Geometry* g1, Geometry* g2) const;

      /**
      Sends a ray into the collision space and returns information about
      the objects the ray hits. The ray does not have to hit anything and
      the returned contact data can be empty.
      \param point start point of the ray
      \param direction Direction of the ray
      \param geometryContacts A GeometryContactPtrVector where contact data will be stored for the objects the ray intersected
      the objects are sorted closest to furthest away
      \param particleGeometryContacts A GeometryContactPtrVector where contact data will be stored for the objects the ray intersected
      the objects are sorted closest to furthest away
      \param length Max length of ray, objects further away than length will not
      be hit even if they are in the path of the ray
      \return true if at least one object was intersected, otherwise false
      */
      bool intersect(
        const agx::Vec3& point,
        const agx::Vec3& direction,
        LocalGeometryContactVector& geometryContacts,
        LocalParticleGeometryContactVector& particleGeometryContacts,
        const agx::Real length = FLT_MAX / 4
      );

      /**
      Enable/Disable contact reduction between pairs of geometries.
      \param flag - if true, contact reduction is enabled.
      */
      void setEnableContactReduction( bool flag );

      /**
      Specify the default (for this Simulation) resolution used when evaluating contacts for reduction between geometry-geometry contacts.
      A high value will keep more contacts, lower will result in more aggressive reduction the default.
      If no resolution is specified per material this value will be used.
      Commonly a value of 2-3 will give good results.
      Values from 1 to 10 are valid.
      This value will be used for all geometry contacts where either no contact material is set,
      or where the ContactReductionBinResolution of the contact material is 0.
      \param binResolution - parameter to ContactReducer.
      */
      void setContactReductionBinResolution(agx::UInt binResolution);

      /// \return Is contact reduction for geometry pairs enabled?
      bool getEnableContactReduction() const;

      /// \retval How many bins per dimension for contact reduction between geometry pairs.
      agx::UInt getContactReductionBinResolution() const;

      /**
      Specify the lower threshold for the number of contact points needed per geometry contact
      in order to do contact reduction per geometry contact.
      \param threshold - Lower threshold for number of contact points in a geometry contact in order run contact reduction on it.
      */
      void setContactReductionThreshold(agx::UInt threshold);

      /**
      \return Lower threshold number of contact points in a geometry contact in order run contact reduction on it.
      */
      agx::UInt getContactReductionThreshold() const;

      /**
      \return a unique group ID starting from 2^31
      */
      agx::UInt32 getUniqueGroupID();

      /**
      Enable/disable collisions between two geometry group ID:s.
      This can be used to group Geometries into separate collision groups, where
      collisions can be enabled/disabled between the different GeometryID groups.

      The largest number for a group ID is defined to be 2^31 - 1 (2 147 483 647),
      the rest are reserved by space and can be accessed via getUniqueGroupID().
      \param group1 - First group id
      \param group2 - Second group id
      \param flag - If true, the geometries in groupID1 and groupID2 can collide, otherwise not.
      */
      void setEnablePair( agx::UInt32 group1, agx::UInt32 group2, bool flag );
      void setEnablePair( const agx::Name& group1, const agx::Name& group2, bool flag );

      /**
      Get information if \p group1 and \p group2 are allowed to collide.
      \param group1 - First id
      \param group2 - Second id
      \return true if the two groupID is allowed to collide, otherwise false.
      */
      bool getEnablePair( agx::UInt32 group1, agx::UInt32 group2 ) const;

      /**
      Get information if \p group1 and \p group2 are allowed to collide.
      \param group1 - First id
      \param group2 - Second id
      \return true if the two groupID is allowed to collide, otherwise false.
      */
      bool getEnablePair(const agx::Name& group1, const agx::Name& group2) const;

      /**
      If for some reason a geometry or several geometries collides with more geometries now than previously,
      we need to make sure that current overlaps are reported, as they might be ignored otherwise.
      \param relevantCollision A class that has the function
                               const bool operator()(const agxCollide::Geometry*, const agxCollide::Geometry*)
                               that returns whether or not the collision between the two geometries should be
                               reevaluated
      */
      template< typename F >
      void enableForContacts(F relevantCollision)
      {
        // For all COMBINATIONS of geometries, check if the collision is possibly relevant
        // and add them to broad phase if their bounding volumes are overlapping.
        for ( agx::UInt i = 0, numGeometries = m_geometries.size(); !m_geometries.empty() && i < numGeometries - 1; ++i ) {
          for ( agx::UInt j = i + 1; j < numGeometries; ++j ) {
            if ( i == j )
              continue;

            // Relevant collision:
            // Report overlap.
            if ( relevantCollision(m_geometries[ i ], m_geometries[ j ] ) &&
              m_sweepAndPrune && // Only need to report overlaps to SAP, the grid will find them by itself on its next search.
              canCollide( m_geometries[ i ], m_geometries[ j ] ) &&
              m_geometries[ i ]->getBoundingVolume()->hasOverlap( *m_geometries[ j ]->getBoundingVolume() ) ) {
                reportOverlap( m_geometries[ i ], m_geometries[ j ] );
            }
          }
        }
      }

      /**
      If for some reason a geometry or several geometries collides with less geometries now than previously,
      we need to make sure that previous overlaps are removed, as they might remain otherwise.
      \param relevantCollision A class that has the function
                               const bool operator()(const agxCollide::Geometry*, const agxCollide::Geometry*)
                               that returns whether or not the collision between the two geometries should be
                               reevaluated
      */
      template< typename F >
      void disableForContacts(F relevantCollision)
      {
        // Among the current broad phase pairs, check if we have
        // a relevant match of geometries. If so, remove (will send
        // separation event if someone is listening to that).
        GeometryPairVector pairsToRemove;
        pairsToRemove.reserve( 16 );

        const BroadPhasePairVector& broadPhasePairs = getBroadPhasePairs();
        for ( agx::UInt i = 0, numBroadPhasePairs = broadPhasePairs.size(); i < numBroadPhasePairs; ++i ) {
          const BroadPhasePairVector::value_type& bpPair = broadPhasePairs[ i ];
          if ( bpPair.geometry1() && bpPair.geometry2() && relevantCollision(bpPair.geometry1().model(), bpPair.geometry2().model() ) )
            pairsToRemove.push_back( GeometryPair( bpPair.geometry1().model(), bpPair.geometry2().model() ) );
        }

        while ( !pairsToRemove.empty() ) {
          if (m_sweepAndPrune)
            m_sweepAndPrune->removePair( pairsToRemove.back().first, pairsToRemove.back().second );
          pairsToRemove.pop_back();
        }
      }

      /**
      Consider using setEnablePair( group1, group2, false ) instead.

      Similar to setEnablePair( group1, group2, false ) but this method doesn't
      look for (and removes) broad phase pairs matching the groups. I.e., this
      method isn't safe to use if you are unsure there are geometries overlapping
      matching the groups.
      \param group1 - first group id
      \param group2 - second group id
      \sa setEnablePair
      */
      void insertDisabledGroupPair( agx::UInt32 group1, agx::UInt32 group2 );

      /**
      Consider using setEnablePair( group1, group2, true ) instead.

      Similar to setEnablePair( group1, group2, true ) but this method doesn't
      look for (and adds) broad phase pairs matching the groups. I.e., this
      method isn't safe to use if you are unsure there are geometries overlapping
      matching the groups.
      \param group1 - first group id
      \param group2 - second group id
      \sa setEnablePair
      */
      void eraseDisabledGroupPair( agx::UInt32 group1, agx::UInt32 group2 );

      /**
      Check if two geometries can collide.

      Following has to be true to make this method return true.

      * Both geometries have to be enabled
      * if they belong to a group pair that is not disabled
      * g1 != g2.
      * collisions for g1 are not disabled against g2 (g1->setEnableCollisions(g2, false)
      * if they both have a body, that it is not the same body.

      \return true if the two geometries can collide.
      */
      bool canCollide( const Geometry* g1, const Geometry* g2 ) const;

      /** Explicitly disable/enable a geometry pair */
      void setEnableCollisions(const Geometry* g1, const Geometry* g2, bool flag);

      // To be called for geometries that are enabled/disabled
      void setEnableGeometry(Geometry* geometry, bool flag);

      /**
       * Returns the current threshold for where oriented bounds are created for Geometries.
       */
      agx::Real getOrientedBoundsThreshold() const;

      /**
       * Sets the threshold where oriented bounds are created for Geometries. Changing the value will
       * create or destroy oriented bounds as necessary for already created Geometries.
       */
      void setOrientedBoundsThreshold( agx::Real threshold );

      /**
      Schedule a removal of a specified Narrow phase contact, the contact will not be removed
      until the method commitRemovedContacts() is called (during the next update())
      \param contact - Remove the specified contact.
      */
      void removeContact(agx::Physics::GeometryContactPtr contact);

      /**
      Go through all the contact and contact points scheduled for removal and remove them from the list of contacts
      */
      void commitRemovedContacts();

      /**
      \return a vector of the current overlapping bounding volumes.
      */
      const BroadPhasePairVector& getBroadPhasePairs() const;

      /**
      \return a vector of the geometries which are no longer overlapping (could still have overlapping bounding volumes)
      */
      const SeparationPairVector& getSeparationPairs() const;


      /**
      Extract a new vector including geometries which just recently separated
      \param geometry1 - Specifies the first geometry to search for among the separation pairs
      \param geometry2 - Specifies the second geometry to search for among the separation pairs, if nullptr then only match geometry1
      \param matches - a vector of the geometries which are no longer overlapping (could still have overlapping bounding volumes)
      \return size of matching vector
      */
      size_t getSeparationPairs(agx::Physics::GeometryPairPtrVector &matches, const agxCollide::Geometry *geometry1, const agxCollide::Geometry *geometry2=nullptr) const;

      /**
      Extract a new vector including rigid bodies which just recently separated
      \param body1 - Specifies the first rigid body to search for among the separation pairs
      \param body2 - Specifies the second rigid body to search for among the separation pairs, if nullptr then only match geometry1
      \param matches - a vector of the geometries which are no longer overlapping (could still have overlapping bounding volumes)
      \return size of matching vector
      */
      size_t getSeparationPairs(agx::Physics::GeometryPairPtrVector& matches, const agx::RigidBody *body1, const agx::RigidBody *body2=nullptr) const;


      /**
      Add an explicit bounding volume overlap between the two specified geometries.
      \param geometry1, geometry2 - The two geometries in the new Broad phase overlap.
      */
      void reportOverlap( agxCollide::Geometry* geometry1, agxCollide::Geometry* geometry2 );

      /**
      Remove a (Sweep and Prune) bounding overlap, if it exists, between \p geometry1 and \p geoemtry2.
      It doesn't matter in which order the two geometries are passed to this method.
      \param geometry1 - first geometry
      \param geometry2 - second geometry
      */
      void removeOverlap( agxCollide::Geometry* geometry1, agxCollide::Geometry* geometry2 );

      /**
      \return a pointer to the Sweep and prune implementation.
      */
      agxCollide::SweepAndPrune* getSweepAndPrune();

      typedef agx::SetVector<GeometryPair> GeometryPairHash;

      /// \return an hashtable containing all pairs of disabled geometries
      const GeometryPairHash& getDisabledPairs() const;

      void resetParticleCellAssignments();
      void resetGeometryCellAssignments(agxCollide::Geometry* geometry);


      /**
      \return The hierarchical grid.
      */
      HierarchicalGrid *getHierarchicalGrid();

      /// \return the current ParticleGeometryContacts in the simulation
      const ParticleGeometryContactVector& getParticleGeometryContacts();

      /// \return the current ParticleParticle contacts in the simulation
      const ParticlePairContactVector& getParticleParticleContacts();

      /// \return the computed number of enabled geometry contact points in the simulation.
      size_t computeNumberOfEnabledContactPoints() const;

      /**
      Find the complete state of disabled collisions, including name <-> name pairs,
      id <-> id pairs and geometry <-> geometry pairs.
      \return current disabled collisions state
      */
      agxCollide::DisabledCollisionsState findDisabledCollisionsState() const;

    public:

      DOXYGEN_START_INTERNAL_BLOCK()


      /**
      Internal method: Synchronize geometry transformations to corresponding rigid bodies and update bounding volumes.
      */
      void synchronizeTransforms();

      /**
      Internal method: calculate the bounding volumes for each geometry
      */
      void synchronizeBounds();
      /**
      Print a string containing a formated output of all the contacts in the list to std::cout
      Used for debugging mainly.
      */
      void printContacts(const agx::String& title);

      CollisionGroupManager* getCollisionGroupManager();
      const CollisionGroupManager* getCollisionGroupManager() const;

      agxData::EntityStorage* getShapeStorage(agxData::EntityModel* shapeModel);

      agxData::EntityStorage *getGeometryStorage();

      agxData::EntityStorage* getGeometryContactStorage();
      agxData::EntityStorage* getContactPointStorage();
      agxData::EntityStorage* getOrientedGeometryBoundStorage();

      void setEnableParticleGridTasks(bool flag);
      void optimizeGridCellFitting();

      /// \return The group of tasks that is run when the simulation is stepped.
      agx::TaskGroup* getUpdateTask();


      SpaceListener::GeometryEvent addGeometryEvent;
      SpaceListener::GeometryEvent removeGeometryEvent;

      SpaceListener::ShapeEvent addShapeEvent;
      SpaceListener::ShapeEvent removeShapeEvent;

      static agx::Model* ClassModel();

      AGXSTREAM_DECLARE_SERIALIZABLE( agxCollide::Space );

      typedef agx::SymmetricPair<unsigned> GeometryIDPair;

      /// Internal class for storing/restoring Space.
      class SpaceSerializer : public agxStream::Serializable
      {
        public:
          SpaceSerializer( const agxCollide::Space* space ) : m_space(space) {}
          AGXSTREAM_DECLARE_SERIALIZABLE(agxCollide::Space::SpaceSerializer);

          const agxCollide::Space* getSpace() const {
            return m_space;
          }
          void setSpace( agxCollide::Space* space ) {
            m_space = space;
          }

          typedef agx::Vector<GeometryIDPair> DisablePairVector;
          DisablePairVector m_disabledGeometryIDPairs;
          agx::UInt32 m_uniqueGroupID;

        protected:
          SpaceSerializer( ) : m_space(nullptr) {}
          const Space* m_space;

      };

      friend class SpaceSerializer;
      friend class agxSDK::Simulation;

      DOXYGEN_END_INTERNAL_BLOCK()

    protected:

      /// Destructor
      virtual ~Space();

      DOXYGEN_START_INTERNAL_BLOCK()

      void clearContactData();

      /// Perform an internal clean up of removed geometries.
      void garbageCollect(bool clearGcList = true);

      void restore( const SpaceSerializer& ss );

      void commitNewOverlaps();

      friend class Geometry;

      /// Initialize space tasks etc. Called from the constructor.
      void init();


      /// Do the narrow phase overlap tests
      template <typename T1, typename T2>
      void calculateNarrowPhase(BroadPhasePair* pair, T1& contactContainer, T2& separationContainer);

      void createGeometryContacts();
      void startUpdateTimer(agx::Task* task);
      void stopUpdateTimerAndReport(agx::Task* task);
      void createShapeStorage(const agx::Path& entityPath);
      void updateContactState();
      void createMissingContactMaterials();
      void updateSweepAndPrune();

      void loadTasks(BroadPhaseAlgorithm algorithm);
      agx::Task* createSynchronizeTransformsTask();
      agx::Task* createSynchronizeBoundsTask();
      agx::Task* createUpdateBroadPhaseTask(BroadPhaseAlgorithm algorithm);
      agx::Task* createUpdateNarrowPhaseTask( const agx::String& implementation = "");
      agx::Task* createUpdateContactStateTask();

      void createParticlePairContacts();
      void createParticleGeometryContacts();


      DOXYGEN_END_INTERNAL_BLOCK()

    private:
      void validate();

      bool removeSweepAndPrune();
      bool removeGrid();

      friend class agxSDK::SimulationFrameWriter;
      void setCollisionGroupManager(CollisionGroupManager* m);

      BroadPhaseAlgorithm m_broadPhaseAlgorithm;

      HierarchicalGridRef m_grid;
      agxCollide::SweepAndPruneRef m_sweepAndPrune;

      agx::TaskGroupRef m_updateTask;
      agx::TaskRef m_synchronizeTransformsTask;
      agx::TaskRef m_synchronizeBoundsTask;
      agx::TaskRef m_updateBroadPhaseTask;
      agx::TaskRef m_updateNarrowPhaseTask;
      agx::TaskRef m_updateContactStateTask;
      agx::TaskRef m_createMissingMaterialsTask;

      agx::TaskRef m_synchronizeTransformsTask2;
      agx::TaskRef m_synchronizeBoundsTask2;
      agx::TaskRef m_updateBroadPhaseTask2;
      agx::TaskRef m_updateNarrowPhaseTask2;
      agx::TaskRef m_updateContactStateTask2;
      agx::TaskRef m_createMissingMaterialsTask2;

      GeometryRefVector m_geometries;
      agx::Vector<GeometryObserver> m_potentialProblemGeometries;

      agxData::EntityStorageRef m_geometryStorage;
      agxData::EntityStorageRef m_geometryContactStorage;
      agxData::EntityStorageRef m_contactPointStorage;
      agxData::EntityStorageRef m_broadPhasePairStorage;
      agxData::EntityStorageRef m_geometryContactSeparations;
      agxData::EntityStorageRef m_orientedGeometryBoundStorage;
      agxData::BufferRef m_materialLessGeometryContacts;
      agxData::BufferRef m_sortedGeometryContacts;
      agxData::BufferRef m_sortedSeparationPairs;
      agxData::BufferRef m_filteredBroadPhasePairs;

      BroadPhasePairVector m_broadPhasePairVector;
      SeparationPairVector m_separationPairVector;

      GeometryPairHash m_disabledGeometryPairs;

      friend class agxIO::SceneExporter;
      friend class agxCollide::SweepAndPrune;

      agxCollide::SweepAndPrune::ExternallyReportedPairsContainer m_externallyReportedPairs;

      typedef agx::HashVector< agx::Physics::GeometryPtr, GeometryRef > GeometryRemoveTable;
      GeometryRemoveTable m_geometriesToRemove;

      agx::Timer m_updateTimer;
      agx::Task::ExecutionEvent::CallbackType m_startUpdateTimerCallback;
      agx::Task::ExecutionEvent::CallbackType m_stopUpdateTimerAndReportCallback;

      agxData::ValueRefT<agx::Bool> m_contactReductionEnable;
      agxData::ValueRefT<agx::UInt> m_contactReductionBinResolution;
      agxData::ValueRefT<agx::UInt> m_contactReductionThreshold;
      agxData::ValueRefT<agx::UInt> m_narrowPhaseMinJobSize;

      agxData::ValueRefT<agx::Real> m_orientedBoundsThreshold;

      GeometryRef m_ray;
      SphereRef m_proxyParticle;

      GeometryContactPtrVector m_geometryContacts;
      agxData::BufferRef m_geometryContactInstanceBuffer;
      agxData::BufferRef m_contactPointInstanceBuffer;

      ParticleGeometryContactVector m_particleGeometryContacts;
      ParticlePairContactVector     m_particlePairContacts;

      typedef agx::HashTable<agxData::EntityModel*, agxData::EntityStorage*> ShapeStorageTable;
      ShapeStorageTable m_shapeStorageTable;

      typedef agx::Vector< SpaceListenerRef > SpaceListenerRefVector;
      SpaceListenerRefVector m_listeners;
      agxData::Buffer::Event::CallbackType m_shapeTransformReallocationCallback;


      agxData::EntityStorageRef m_collisionObjectStorage;
      agxData::BufferRef m_collisionObjectBuffer_sourceIndex;
      agxData::BufferRef m_collisionObjectBuffer_subsystem;
      agxData::BufferRef m_collisionObjectBuffer_position;
      agxData::BufferRef m_collisionObjectBuffer_radius;
      agxData::BufferRef m_collisionObjectBuffer_obbIndex;

      agxData::BufferRef m_gridOvelapBuffer_collisionObject1;
      agxData::BufferRef m_gridOvelapBuffer_collisionObject2;

      agx::TaskRef m_overlapTestTask;

      agx::HashSet<Geometry*> m_skipTable;

      typedef agx::HashTableComponent<agxData::GeometryPair, agx::Physics::GeometryContactPtr> GridContactTableComponent;
      typedef agx::QuadraticProbingHashTable<agxData::GeometryPair, agx::Physics::GeometryContactPtr, agx::HashFn<agxData::GeometryPair>, agxData::BufferProxyAllocator> GridContactTable;


      CollisionGroupManagerRef m_collisionGroupManager;

      agxData::Value::Event::CallbackType m_gridCellSizeAlignmentCallback;
      void gridCellSizeAlignmentCallback(agxData::Value* sizeAlignment);

      GridContactTable* m_gridContactTable;
  };

  /* Implementation */
  AGX_FORCE_INLINE  bool Space::canCollide( const Geometry* g1, const Geometry* g2 ) const
  {
    if (!(g1 && g2) )
      return false;

    // We cannot collide with ourself. TODO Not necessarily true in general case, e.g., if a deformable is defined as one geometry
    if (g1 == g2)
      return false;

    // One of the geometries are disabled for collision
    if (!(g1->isEnabled() && g2->isEnabled()))
      return false;

    if (!(g1->getEnableCollisions() && g2->getEnableCollisions()))
      return false;

    const agx::RigidBody* rb1 = g1->getRigidBody();
    const agx::RigidBody* rb2 = g2->getRigidBody();

    // If one is disabled, ignore overlap.
    if ( (rb1 && !rb1->getEnable()) || (rb2 && !rb2->getEnable()) )
      return false;

    // not same (non-null) body
    if (rb1 && (rb1 == rb2))
      return false;

    // Is this geometry pair disabled?
    if (m_disabledGeometryPairs.contains( GeometryPair(const_cast<Geometry*>(g1), const_cast<Geometry*>(g2)) ))
      return false;

    // Can these two groups collide?
    const agx::Physics::CollisionGroupSetPtr& groupSet1 = g1->getGroupSet();
    const agx::Physics::CollisionGroupSetPtr& groupSet2 = g2->getGroupSet();
    if (groupSet1 && groupSet2 && g1->getSpace() == this && g2->getSpace() == this && !m_collisionGroupManager->canSetsCollide(groupSet1, groupSet2))
      return false;

    return true;
  }

  DOXYGEN_START_INTERNAL_BLOCK()

  AGX_FORCE_INLINE agxData::EntityStorage* Space::getOrientedGeometryBoundStorage()
  {
    return m_orientedGeometryBoundStorage;
  }

  AGX_FORCE_INLINE agxData::EntityStorage* Space::getGeometryContactStorage()
  {
    return m_geometryContactStorage;
  }

  AGX_FORCE_INLINE agxData::EntityStorage* Space::getContactPointStorage()
  {
    return m_contactPointStorage;
  }


  AGX_FORCE_INLINE CollisionGroupManager* Space::getCollisionGroupManager()
  {
    return m_collisionGroupManager;
  }

  AGX_FORCE_INLINE const CollisionGroupManager* Space::getCollisionGroupManager() const
  {
    return m_collisionGroupManager;
  }

  AGX_FORCE_INLINE agxData::EntityStorage *Space::getGeometryStorage() { return m_geometryStorage; }

  DOXYGEN_END_INTERNAL_BLOCK()

  AGX_FORCE_INLINE const GeometryRefVector& Space::getGeometries() const
  {
    return m_geometries;
  }

  AGX_FORCE_INLINE bool Space::getEnablePair( const agx::Name& group1, const agx::Name& group2 ) const
  {
    return this->getEnablePair(
      m_collisionGroupManager->getGroupId(group1),
      m_collisionGroupManager->getGroupId(group2));
  }

  AGX_FORCE_INLINE bool Space::getEnablePair( agx::UInt32 group1, agx::UInt32 group2 ) const
  {
    return m_collisionGroupManager->canGroupsCollide(group1, group2);
  }

  AGX_FORCE_INLINE const BroadPhasePairVector& Space::getBroadPhasePairs() const
  {
    m_broadPhasePairVector.allocator().update();
    return m_broadPhasePairVector;
  }

  AGX_FORCE_INLINE const SeparationPairVector& Space::getSeparationPairs() const
  {
    m_separationPairVector.allocator().update();
    return m_separationPairVector;
  }

  AGX_FORCE_INLINE void Space::removeContact(agx::Physics::GeometryContactPtr contact)
  {
    if (contact)
      contact.enabled() = false;
  }

  AGX_FORCE_INLINE void Space::reportOverlap( Geometry* geometry1, Geometry* geometry2 )
  {
    if ( geometry1 == 0L || geometry2 == 0L || !geometry1->isEnabled() || !geometry2->isEnabled() )
      return;

    m_externallyReportedPairs.insert( SweepAndPrune::SymmetricGeometryPtrPair( geometry1, geometry2 ), SweepAndPrune::SymmetricGeometryRefPair( geometry1, geometry2 ) );
  }

  AGX_FORCE_INLINE void Space::removeOverlap( Geometry* geometry1, Geometry* geometry2 )
  {
    if ( m_sweepAndPrune == 0L || geometry1 == 0L || geometry2 == 0L )
      return;

    auto it = m_externallyReportedPairs.find( SweepAndPrune::SymmetricGeometryPtrPair( geometry1, geometry2 ) );
    if ( it != m_externallyReportedPairs.end() )
      m_externallyReportedPairs.erase( it );
    else
      m_sweepAndPrune->removePair( geometry1, geometry2 );
  }

  AGX_FORCE_INLINE bool Space::getEnableContactReduction() const
  {
    return m_contactReductionEnable->get();
  }

  AGX_FORCE_INLINE agx::UInt Space::getContactReductionBinResolution() const
  {
    return m_contactReductionBinResolution->get();
  }

  AGX_FORCE_INLINE agx::Real Space::getOrientedBoundsThreshold() const
  {
    return m_orientedBoundsThreshold->get();
  }

  AGX_FORCE_INLINE HierarchicalGrid *Space::getHierarchicalGrid() { return m_grid; }

}
