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

#ifndef AGX_MERGESPLITBODYSTORAGE_H
#define AGX_MERGESPLITBODYSTORAGE_H

#include <agx/agxPhysics_export.h>
#include <agx/Referenced.h>
#include <agx/RigidBody.h>
#include <agxCollide/Geometry.h>
#include <agxStream/Serializable.h>

namespace agx
{

  class MergeSplit;

  /**
  Book keeping of a body which can be merged with other bodies.
  Keep track of its children, parent bodies, original geometries belonging to this body etc.
  */
  class AGXPHYSICS_EXPORT MergeSplitBodyStorage : public agx::Referenced, public virtual agxStream::Serializable
  {

    public:

      typedef agx::Vector<agx::observer_ptr<agxCollide::Geometry> > GeometryObserverVector;

      /// Default constructor
      MergeSplitBodyStorage();

      /**
      Constructor which associates a body to this body storage
      */
      MergeSplitBodyStorage( agx::RigidBody *body );

      /// \return a const reference to the vector of geometries associated to this body storage.
      const agxCollide::GeometryRefVector& getGeometries() const;

      AGXSTREAM_DECLARE_SERIALIZABLE( agx::MergeSplitBodyStorage );

      static void addMassProperties( MergeSplitBodyStorage* targetStorage, const MergeSplitBodyStorage* sourceStorage, Real sign);
      static void removeMassProperties( MergeSplitBodyStorage* targetStorage, const MergeSplitBodyStorage* sourceStorage, Real sign);


      /**
      Set the threshold when the inertiatensor of a parent should be completely rebuilt.
      \param numChildren - If a parent has less children than \p numChildren, rebuild the whole tensor
      \param numUpdates - If a parent has been incrementally recalculated more than \p numUpdates, rebuild the whole tensor
      */
      static void setRebuildThreshold( size_t numChildren, size_t numUpdates );

      bool validate() const;

      /// \return a pointer to the associated body
      agx::RigidBody *getRigidBody();
        /// \return a pointer to the associated body
      const agx::RigidBody *getRigidBody() const;


      /// \return a pointer to a parent MergeSplitBodyStorage
      MergeSplitBodyStorage* getParent( );

      /// \return a pointer to a parent MergeSplitBodyStorage
      const MergeSplitBodyStorage* getParent( ) const;

      AffineMatrix4x4 getTransformRelativeParent() const;

      /**
      \return true if the specified storage is a child of this
      */
      bool hasChild( MergeSplitBodyStorage* child ) const;

      /**
      \return true if this storage has children
      */
      bool hasChildren() const;

      /**
      \return the number of children this body storage has
      */
      size_t getNumChildren() const;

      /**
      \return a pointer to the nth child, nullptr if there are no children of this index.
      */
      MergeSplitBodyStorage *getChild( size_t nth );

      /**
      \return a pointer to the nth child, nullptr if there are no children of this index.
      */
      const MergeSplitBodyStorage *getChild( size_t nth ) const;


    protected:

      virtual ~MergeSplitBodyStorage();

      /// Reset this storage so its not associated to any body
      void clear();

      /**
      Associate this storage with the specified body. Stores information about motioncontrol, associated geometries etc.
      */
      void setRigidBody( agx::RigidBody* body );

      /// \return true if this body is merged into a superbody
      bool isMerged() const;

      /// Set the pointer to the parent body of this body
      void setParent( MergeSplitBodyStorage* parent );

      /// Calculate and store the transform relative the parent body
      void calculateRelativeTransform( const MergeSplitBodyStorage* parentStorage );

      /**
      Add a storage as a child to this storage
      */
      void addChild( MergeSplitBodyStorage* child );

      /**
      Remove a storage so its no longer a child of this storage
      */
      bool removeChild( MergeSplitBodyStorage* child );

      /**
      \return a pointer to this body storages frame
      */
      agx::Frame *getFrame();

      const agx::Frame *getFrame() const;

      agx::RigidBody::MotionControl getMotionControl() const;

      void setMotionControl( agx::RigidBody::MotionControl m );

      void addGeometry( agxCollide::Geometry *geom );

      /**
      Restore the associated body in that it will get back all its original geometries, motioncontrol etc.
      */
      void restore( bool beingErased=false );

      /**
      Make the body follow the first geometry in the list of geometries
      */
      void followGeometry();

      void restoreMassProperties();

      inline void incrementNumUpdates();

      inline void resetNumUpdates();

      inline size_t getNumUpdates();

    private:

      friend class MergeSplit;

      agx::observer_ptr<agx::RigidBody> m_body;
      agx::observer_ptr<agx::Frame> m_parentFrame;

      agx::RigidBody::MotionControl m_motionControl;
      agx::AffineMatrix4x4 m_cmTransform;
      agx::SPDMatrix3x3 m_tensor;
      agx::UInt32 m_autoGenerateTensorMask;

      agx::Real m_mass;

      agx::FrameRef m_frame;

      typedef agx::SetVector<MergeSplitBodyStorage *> RigidBodyPtrHashVector;
      RigidBodyPtrHashVector m_children;
      agx::observer_ptr<MergeSplitBodyStorage> m_parentStorage;

      agxCollide::GeometryRefVector m_geometries;
      agx::Vector<AffineMatrix4x4>  m_geometryTransforms;
      agx::AffineMatrix4x4 m_transformRelParent;

      size_t m_numUpdates;

      static size_t m_rebuildNumChildren;
      static size_t m_rebuildNumUpdates;

  };

  inline bool MergeSplitBodyStorage::isMerged() const { return m_parentStorage.isValid(); }

  inline void MergeSplitBodyStorage::incrementNumUpdates() { m_numUpdates++; }

  inline void MergeSplitBodyStorage::resetNumUpdates() { m_numUpdates = 0; }

  inline size_t MergeSplitBodyStorage::getNumUpdates() { return m_numUpdates; }

  inline agx::RigidBody* MergeSplitBodyStorage::getRigidBody() { return m_body; }
  inline const agx::RigidBody* MergeSplitBodyStorage::getRigidBody() const { return m_body; }

  inline MergeSplitBodyStorage* MergeSplitBodyStorage::getParent( ) { return m_parentStorage; }
  inline const MergeSplitBodyStorage* MergeSplitBodyStorage::getParent( ) const { return m_parentStorage; }

  inline AffineMatrix4x4 MergeSplitBodyStorage::getTransformRelativeParent() const { return m_transformRelParent; }

  inline size_t MergeSplitBodyStorage::getNumChildren() const
  {
    return m_children.size();
  }

  inline agx::Frame *MergeSplitBodyStorage::getFrame()
  {
    return m_frame;
  }

  inline const agx::Frame *MergeSplitBodyStorage::getFrame() const
  {
    return m_frame;
  }

  inline agx::RigidBody::MotionControl MergeSplitBodyStorage::getMotionControl() const
  {
    return m_motionControl;
  }

  inline bool MergeSplitBodyStorage::hasChildren() const
  {
    return m_children.size() > 0;
  }


}

#endif /* AGX_MERGESPLITBODYSTORAGE_H */
