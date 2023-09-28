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

#ifndef AGX_MERGESPLIT_H
#define AGX_MERGESPLIT_H

#include <agx/TimeStamp.h>
#include <agxCollide/Space.h>
#include <agx/agxPhysics_export.h>
#include <agx/SleepThreshold.h>
#include <agx/MergeSplitBodyStorage.h>
#include <agxCollide/agxcollide_hash_types.h>

namespace agx
{
  /**
  Class which will go through contacting/separating bodies and determine if the bodies
  can be merged (geometries from bodyA can go into bodyB and bodyB will now move as one rigid body) or split
  (the original geometries of bodyA can be transferred back so BodyA can move as a separate body).

  Default is splitting bodies due to contacts in a PRE step, that is after contacts are generated and before solver.
  \deprecated Instead we recommend using AMOR/MergeSplitHandler
  */
  class AGXPHYSICS_EXPORT MergeSplit : public agx::Referenced, public agxStream::Serializable
  {
    public:

      /// Default constructor
      MergeSplit( agxCollide::Space *space=nullptr );

      /**
      \return a pointer to a MergeSplitThreshold object which holds the acceleration/velocity threshold settings
      */
      MergeSplitThreshold *getThreshold();

      /**
      \return a const pointer to a MergeSplitThreshold object which holds the acceleration/velocity threshold settings
      */
      const MergeSplitThreshold *getThreshold() const;

      /**
      Enable/disable MergeSplit.
      \param flag - If true, MergeSplit will be enabled, if false, all merged bodies will be split,
      */
      void setEnable( bool flag );

      /**
      Specifies whether the split should occur in the pre solver step, or in the post step.
      A system where split occurs in pre will appear more dynamic and realistic, however, it will
      probably result in a more overall dynamic system with somewhat lower performance.
      Splitting in pre might also cause jitter in large piles, due to that when splitting in pre, no contacts are
      created until next time step. Meaning that gravity will pull the split body, let it fall during one time step, and at
      the next contacts will be created and it will be stabilized. So for single objects on merged ground split in pre will
      work just fine, for larger piles, some jitter might be noticed.
      \param flag if true, the split will occur in post, otherwise in pre (default)
      */
      void setSplitInPost( bool flag );

      /// \return true if split is enabled to occur in post solver step.
      bool getSplitInPost() const;

      ///\return true if MergeSplit is enabled
      bool getEnable( ) const;

      ///\return true if MergeSplit is enabled
      bool isEnabled( ) const;

      /**
      Split all merged bodies, clear all data structures.
      */
      void clear();

      /**
      Explicit call to split, will go and ask space for contacts and separations and see
      if something can be split based on those contacts. Requires that Space::update() has been called prior to this.
      */
      void split();

      /**
      Take a specified geometry \p geometry and locate its ORIGINAL body (which was the original body
      that this geometry belonged to) and split that. This will also mean that additional geometry (belonging
      to this Body) will be taken out of a merged body.

      \param geometry - The geometry which has an original owner body that will be split
      \param shouldUpdateContacts - If true, and IF the split succeeded, all contacts will be investigated and updated to get correct body/geometry pointers.
      \param forceSplit - if true, body will be split, even if it has properties stating it should not. Geometry will be split
      even if it has property stating it should not.
      \param splitImmediately - If true the geometry will be split now and not later in a batch with other splitted geometries
      \return true if body was successfully split, false if body was not merged, hence split failed.
      */
      bool split( agxCollide::Geometry* geometry, bool shouldUpdateContacts = true, bool forceSplit = false, bool splitImmediately=true );

      /**
      Explicit call to merge, will go and ask space for contacts and separations and see
      if something can be merged based on those contacts. Requires that Space::update() has been called prior to this.
      */
      void merge();

      /// Tell MergeSplit which space to use for accessing contacts
      void setSpace( agxCollide::Space *space );

      /**
      Calculate the relative velocity/acceleration of two rigidbodies center of mass velocities
      */
      static void calculateSeparatingVelocityAcceleration( agx::RigidBody *body1, agx::RigidBody* body2, agx::Real& dVel, agx::Real& dOmega, agx::Real& dAcc, agx::Real& dAngularAcc );

      AGXSTREAM_DECLARE_SERIALIZABLE( agx::MergeSplit );

      void pre( const agx::TimeStamp& t );
      void post( const agx::TimeStamp& t );

      /**
      Take the specified body \p bodyToBeSplit and if it is merged, then split it from
      any parents, make it a free non-merged body.

      \param bodyToBeSplit - The body that should be split from any merged bodies
      \param forceSplit - if true, body will be split, even if it has properties stating it should not.
      \param splitImmediately - Split now not later in a batch with other bodies that are going to be split
      \return true if body was successfully split, false if body was not merged, hence split failed
      */
      bool split( agx::RigidBody *bodyToBeSplit, bool forceSplit = false, bool splitImmediately=true );


      /**
      Explicitly merge two bodies. \p parent will be parent if the following is true:

      * parent is static, child is not
      * parent has a constraint attached to itself, child has not

      If the above rules fails, the two bodies will be swapped and child will be parent.
      If merge cannot proceed (if both bodies are static or both have constraints, the method will return false).

      There is currently no test if any of the two bodies are already merged. If so, the result will be unpredicted.
      \param child - The body that will be the child unless above rule fails
      \param parent - The body that will be the parent unless above rule fails
      \return true if merge was done, false if something went wrong.
      */
      bool merge( agx::RigidBody *child, agx::RigidBody *parent );

      /**
      Class for intercepting merge/split events.

      the merge()/split() methods can be used to enable/disable mergeSplit operations.
      \deprecated Instead we recommend using AMOR/MergeSplitHandler
      */
      class Listener : public agx::Referenced
      {
      public:
        Listener() {}

        /**
        If this method returns true, the merge operation will proceed, otherwise it will be canceled.
        This method is called after all other other test confirm that the two bodies should be merged, so
        this is the last chance to cancel the operation.

        \param parent - pointer to the RigidBody that will BECOME parent if merge is proceeded.
        \param child - pointer to the RigidBody that will BECOME a child of \p parent if merge is proceeded.
        \return if true, the merge operation will proceed as planned.
        */
        virtual bool merge( agx::RigidBody *parent, agx::RigidBody *child ) = 0;

        /**
        If this method returns true, the split operation will proceed, otherwise it will be canceled.
        This method is called after all other tests confirm that the child body should be split,
        this is the last chance to cancel the operation.

        \param parent - pointer to the body which currently is parent of the \p child. This pointer CAN be null if
        child does not have a parent.
        \param child - pointer to the body that will be split from its parent.
        \param contact - pointer to the GeometryContact resulting in the split. This pointer CAN be nullptr if its an
        explicit split, or if its a separation resulting in this split.
        \return if true, the split operation will proceed as planned, otherwise it will be canceled.
        */
        virtual bool split( agx::RigidBody *parent, agx::RigidBody *child, agxCollide::GeometryContact *contact ) = 0;


      protected:
        virtual ~Listener() {}
      };


      typedef agx::ref_ptr<Listener> ListenerRef;

      /**
      Specify a point to a listener that will be called upon merge/split events
      */
      void setListener( Listener *listener );

      /**
      \return pointer to a registered listener
      */
      Listener *getListener( );

      /**
      \return a const pointer to a registered listener
      */
      const Listener *getListener( ) const;

      bool validate ();

    protected:
      friend class agxSDK::Simulation;
      void registerGeometriesBodies( agxSDK::Simulation *sim );

      bool removeFromBodyStorage( agx::RigidBody *body );


      bool shouldMerge( agx::RigidBody *body1, agx::RigidBody *body2, agxCollide::GeometryContact *contact ) const;
      void updateContacts( const agxCollide::GeometryContactPtrVector& contacts, bool removeContacts = false ) const;

      /**
      Take the specified body \p bodyToBeSplit and if it is merged, then split it from
      any parents, make it a free non-merged body.

      \param bodyToBeSplit - The body that should be split from any merged bodies
      \param contact - set to the GeometryContact if it was a contact/impacting resulting in the split, otherwise nullptr.
      \param forceSplit - if true, body will be split, even if it has properties stating it should not.
      \param splitImmediately - Split now not later in a batch with other bodies that are going to be split
      \return true if body was successfully split, false if body was not merged, hence split failed
      */
      bool _split( agx::RigidBody *bodyToBeSplit, agxCollide::GeometryContact *contact, bool forceSplit = false, bool splitImmediately=true );


      bool sleep( agx::RigidBody *body1, agx::RigidBody *body2 );
      bool wake( agx::RigidBody *body1, agx::RigidBody *body2 );


      bool split( const agxCollide::GeometryContactPtrVector& contacts );

      bool split( agxCollide::Geometry *geom1, agxCollide::Geometry *geom2, agxCollide::GeometryContact *contact=nullptr );
      bool split( const agxCollide::SeparationPairVector& separations );

      bool merge( const agxCollide::GeometryContactPtrVector& contacts );
      bool performMerge( agx::RigidBody *child, agx::RigidBody *parent );

      //void transferGeometries( agx::RigidBody *child, const  agxCollide::GeometryPtrVector& geometries, agx::RigidBody *parent );
      void transferGeometries( agx::RigidBody *child, agxCollide::GeometryRefVector geometries, agx::RigidBody *parent );
      void prepareForMerge( agx::RigidBody *body );
      //bool verifyStorage();

      void addNotification();
      void removeNotification();

      virtual ~MergeSplit();

      MergeSplitBodyStorage *getBodyStorage();
      void returnBodyStorage( MergeSplitBodyStorage *bodyStorage );

      void clearStats();

      typedef agx::HashVector<agxCollide::Geometry *, agx::ref_ptr<MergeSplitBodyStorage> > GeometryBodyHashVector;
      GeometryBodyHashVector m_geometryBodyHash;

      size_t m_numMerge;
      size_t m_numSplit;

      agx::ref_ptr<SleepThreshold> m_threshold;

      typedef agx::Vector<agx::ref_ptr<MergeSplitBodyStorage> > MergeSplitBodyStorageRefVector;
      typedef agx::Vector<MergeSplitBodyStorage * > MergeSplitBodyStoragePtrVector;

      MergeSplitBodyStoragePtrVector m_freeBodyStorage;
      MergeSplitBodyStorageRefVector m_bodyStoragePool;
      agx::observer_ptr<agxCollide::Space> m_space;

      bool m_enabled;
      ListenerRef m_listener;

      agx::RigidBodyObserverVector m_bodiesToBeSplit;
      bool m_splitInPost;
  };

  typedef agx::ref_ptr<MergeSplit> MergeSplitRef;

}

#endif /* _AGX_MERGESPLIT_H_ */
