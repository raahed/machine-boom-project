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
#include <agx/agx.h>
#include <agx/observer_ptr.h>
#include <agxCollide/BoundingAABB.h>
#include <agxCollide/Shape.h>
#include <agx/Physics/GeometryEntity.h>
#include <agx/Material.h>
#include <agx/Event.h>

#include <agxStream/Serializable.h>

#include <agx/Property.h>

#include <agxCollide/agxcollide_vector_types.h>
#include <agx/RigidBody.h>
#include <agx/HashSet.h>
#include <agx/AffineMatrix4x4.h>
#include <agx/BitArray.h>
#include <agx/SymmetricPair.h>
#include <agxCollide/CollisionGroupManager.h>
#include <agxCollide/GeometryState.h>
#include <agxCollide/GroupIdCollection.h>
#include <agx/SetVector.h>


namespace agx
{
  class MergeSplitBodyStorage;
  class Frame;
  class InternalData;
  class Emitter;
}

namespace agxCollide
{
  class SweepAndPrune;
  class LocalContactPoint;

}

namespace agxTerrain
{
  class Terrain;
}

namespace agxData
{
  typedef agx::SymmetricPair<agx::Physics::GeometryPtr> GeometryPair;
}

AGX_TYPE_BINDING(agxData::GeometryPair, "GeometryPair")


namespace agxCollide
{
  class Space;
  class ContactPoint;


  AGX_DECLARE_POINTER_TYPES( Geometry );
  AGX_DECLARE_VECTOR_TYPES( Geometry );
  typedef agx::HashVector<const Geometry *,  agx::observer_ptr<const Geometry> > GeometryHashVector;
  typedef agx::HashSet<const Geometry * > GeometryHashSet;
  typedef agx::SetVector< agx::ref_ptr<agxCollide::Geometry> > GeometryRefSetVector;

  /**
  The geometry representation used by the collision detection engine. A geometry
  may be a composite of multiple shapes with local transformations.

  Internal to the physics engine, the only geometry considered is that
  used for interference detection.  Because of the namespace, there
  should be no collision with any other name.
  */
  class AGXPHYSICS_EXPORT Geometry : public agx::Referenced, public virtual agxStream::Serializable
  {
    public:
      typedef agx::Event1<Geometry *> ShapeChangeEvent;
      ShapeChangeEvent shapeChangeEvent;

    public:

      /**
      Constructor
      \param name - name of the geometry
      */
      Geometry( const agx::Name& name = "" );


      /**
      Constructor
      \param shape - pointer to a shape that will be part of this Geometry
      */
      Geometry( Shape* shape );

      /**
      Constructor
      \param shape - pointer to a shape that will be part of this Geometry
      \param localShapeTransform - Transformation of the \p shape relative to the Geometry
      */
      Geometry( Shape* shape, const agx::AffineMatrix4x4& localShapeTransform );

      /**
      This method will make a deep copy of the geometry, excluding membership in any RigidBody.
      \returns a deep copy of the geometry
      */
      Geometry *clone( bool shallowCopyRenderData=false ) const;

      /**
      \return The geometry id.
      */
      agx::Int64 getId() const;

      /**
      Get the name.
      \return the name string of this Geometry.
      */
      const agx::Name& getName() const;

      /// Set the name of this Geometry
      void setName( const agx::Name& name );

      /**
      \return The body which the geometry is part of.
      */
      agx::RigidBody *getRigidBody();

      /**
      \return The body which the geometry is part of.
      */
      const agx::RigidBody *getRigidBody() const;

      /**
      Set the material of the geometry.
      */
      void setMaterial( agx::Material *material );

      /**
      \return the material of the geometry.
      */
      agx::Material *getMaterial();

      /**
      \return the material of the geometry.
      */
      const agx::Material *getMaterial() const;

      /**
      Remove the material from the geometry.
      */
      void removeMaterial();

      /**
      \note: This will give wrong results if the
      geometry contains overlapping shapes.
      \return The mass of the geometry.
      */
      agx::Real calculateMass() const;

      /**
      \note: This will give wrong results if the
      geometry contains overlapping shapes.
      \return The volume of the geometry.
      */
      agx::Real calculateVolume() const;

      /**
      \note: This will give wrong results if the
      geometry contains overlapping shapes.
      \return The center of mass offset
      */
      agx::Vec3 calculateCenterOfMassOffset() const;

      /**
      \note: This will give wrong results if the
      geometry contains overlapping shapes.
      \return The inertia tensor locally in center of mass.
      */
      agx::SPDMatrix3x3 calculateInertia( agx::Real mass ) const;

      /**
      Enable/disable collision testing for geometry.
      A disabled geometry does not render in debug rendering or osg.
      */
      void setEnable( bool flag );

      /**
      Alias for getEnable
      Return whether the geometry should be used in intersection tests.
      This can be set using the setEnable() method.
      \return true if the geometry is used in intersection tests, false if not.
      */
      bool isEnabled() const;

      /**
      Return whether the geometry should be used in intersection tests.
      This can be set using the setEnable() method.
      \return true if the geometry is used in intersection tests, false if not.
      */
      bool getEnable() const;

      /**
      Sets if the geometry should be a sensor. Sensors are included in collision
      detection but the resulting contacts are not passed to the solver.

      Contact data generation for sensors is disabled by passing \p false into
      \p generateContactData. GeometryContacts are still created, but they will
      contain no contact points. This flag has no effect while \p isSensor
      returns \p false. Support for disabling contact data generation is
      currently only supported for some shape type pairs. Other shape type pairs
      will continue to get full contact data.

      \param flag - True to make this Geometry a sensor.
      \param generateContactData - True if contact data is to be generated, false otherwise. False only supported for some shape type pairs.
      */
      void setSensor(bool flag, bool generateContactData = true);

      /**
      Return true if geometry is a sensor.
      */
      bool isSensor() const;

      /**
      \return True if collision data will be omitted from contacts while this geometry is a sensor.
      \see setSensor
      */
      bool isSensorGeneratingContactData() const;

      /**
      If set to true, this geometry will be included in mass property computation when part of a rigid body. (default==true)
      \note This does not affect the returned data from calculateInertia, calculateMass etc.
      \param enableMassProperties - If true this geometry will contribute to mass properties for a rigid body
      */
      void setEnableMassProperties(bool enableMassProperties);
      
      /**
      \return true if this geometry should contribute to the massproperties of a rigidbody.
      */
      bool getEnableMassProperties() const;

      /**
      Add the specified \p id to a vector of group id, this will make a geometry part
      of the group \p id. By default a geometry is not part of any group.

      This can be used to partition the simulation where some groups cannot collide with other groups.
      Which groups can collide or not is determined by Space and can be set using the method
      agxCollide::Space::setEnablePair(id1,id2);

      This does not remove current contacts. To do this, \p forceContactUpdate should be set to
      true. This can be an expensive operation, so if repeated many times and issues might occur,
      it could be better to do this manually using Space::enableForContacts and Space::disableForContacts.
      */
      void addGroup( agx::UInt32 id, bool forceContactUpdate = false );

      /**
      Add the specified \p name to a vector of named group id, this will make a geometry part
      of the group \p name. By default a geometry is not part of any group.

      This can be used to partition the simulation where some groups cannot collide with other groups.
      Which groups can collide or not is determined by Space and can be set using the method
      agxCollide::Space::setEnablePair(name1,name2);

      This does not remove current contacts. To do this, \p forceContactUpdate should be set to
      true. This can be an expensive operation, so if repeated many times and issues might occur,
      it could be better to do this manually using Space::enableForContacts and Space::disableForContacts.
      */
      void addGroup( const agx::Name& name, bool forceContactUpdate = false);

      /**
      Remove a collision group id from the vector of group ids.
      \param id - The group id to be removed
      \param forceContactUpdate - Should current overlaps between disabled groups be found and create impacts.
      */
      void removeGroup( agx::UInt32 id, bool forceContactUpdate = false );

      /**
      Remove a named collision group from the vector of named group ids.
      \param name - The group id to be removed
      \param forceContactUpdate - Should current overlaps between disabled groups be found and create impacts.
      */
      void removeGroup( const agx::Name& name, bool forceContactUpdate = false);

      /**
      Remove all collision group ids.
      \param forceContactUpdate - Should current overlaps between disabled groups be found and create impacts.
      */
      void removeAllGroups(bool forceContactUpdate = false);

      /**
      This is performing a linear search among the collision group id for this Geometry.

      \param id - The group id we are looking for.
      \return true if the Geometry is part of the group \p id
      */
      bool hasGroup( agx::UInt32 id ) const;

      /**
      Searches for the named group name for this geometry and return true if it can be found.

      \param name - The group id we are looking for.
      \return true if the Geometry is part of the named group \p name
      */
      bool hasGroup( const agx::Name& name ) const;

      /**
      Set the collision group set of the geometry
      \param set - the collision group set to set on the geometry.
      */
      void setGroupSet( agx::Physics::CollisionGroupSetPtr set );

      /**
      \return The current group set.
      */
      const agx::Physics::CollisionGroupSetPtr& getGroupSet() const;

      /// Deprecated: Only for backward compatibility
      typedef CollisionGroupManager::GroupIdHash GroupSet;
      const GroupSet& getGroupIDs() const;

      /**
      Specify whether \p otherGeometry is allowed to collide with this geometry
      */
      void setEnableCollisions( const Geometry* otherGeometry, bool flag );

      /**
      This method is only based on what is set using setEnableCollisions(). Any collision settings set 
      by collision groups will not be taken into account. If you want to know if two geometries can collide at all, 
      use canCollide().
      \return true if this geometry is allowed to collide with \p geometry according to geometry pairs.
      */
      bool getEnableCollisions( const Geometry* otherGeometry ) const;

      /**
      Should this geometry collide with any other geometries.
      Disabling this still means that it will be part of MassProperty calculations if it belongs to a rigid body.
      \param collide - If false this geometry will not generate any contacts with any other geometry
       */
      void setEnableCollisions(bool collide);

      /**
      \return true if this geometry is allowed to collide with any other geometry
      */
      bool getEnableCollisions() const;


      /**
      \return true if \p this geometry CAN collide with \p otherGeometry.

      Following has to be true to make this method return true.

      * if they belong to a group pair that is not disabled
      * g1 != g2.
      * collisions for g1 is not disabled against g2 (g1->setEnableCollisions(g2, false)
      * if they both have a body, that it is not the same body.
      */
      bool canCollide( const Geometry *otherGeometry ) const;


      /**
      Implement this method to get access to different surface velocities on the geometry:s surface.
      One example would be to have a texture specifying a velocity field of the geometry
      \param point - ContactPoint in world coordinates that can be used to calculate the surface velocity.
      \param index - Which geometry in the contact is this one? Valid values: 0 or 1.
         The normal in 'point' points in the direction that 0 has to move to leave the contact with 0.
      \return the calculated velocity.
      */
      virtual agx::Vec3f calculateSurfaceVelocity( const agxCollide::LocalContactPoint& point, size_t index ) const;

      /**
      The bounding volume for the geometry is
      made large enough to fit all shape bounds.
      */
      virtual const BoundingAABB* getBoundingVolume() const;

      /// \return a pointer to the PropertyContainer
      agx::PropertyContainer* getPropertyContainer();

      /// \return a pointer to the PropertyContainer
      const agx::PropertyContainer* getPropertyContainer() const;

      /// \return true if it has an initialized PropertyContainer
      bool hasPropertyContainer() const;

      /**
      Give this a new PropertyContainer.
      \param container - The new PropertyContainer of this instance.
      */
      void setPropertyContainer( agx::PropertyContainer *container );

      /**
      \return The space where the geometry is placed.
      */
      Space* getSpace();

      /**
      \return The space where the geometry is placed.
      */
      const Space* getSpace() const;


      AGXSTREAM_DECLARE_SERIALIZABLE( agxCollide::Geometry );


      /**
      \return a pointer to a user defined data inherited from agx::Referenced class
      */
      agx::Referenced* getCustomData();

      /**
      \return a const pointer to a user defined data inherited from agx::Referenced class
      */
      const agx::Referenced* getCustomData() const;

      /**
      Store a ref_ptr to a user custom data class, which must be inherited from Referenced.
      */
      void setCustomData( agx::Referenced* data );

      /**
      This method is guaranteed to always return the same value as getRigidBody() if MergeSplit is not in use.
      \returns a pointer to the RigidBody this geometry was assigned to before that body was merged into another body.
      */
      agx::RigidBody* getOriginalBody();

      /**
      This method is guaranteed to always return the same value as getRigidBody() if MergeSplit is not in use.
      \returns a const pointer to the RigidBody this geometry was assigned to before that body was merged with another body.
      */
      const agx::RigidBody* getOriginalBody() const;

      /**
      Set the velocity of this geometry's surface in the geometry's local coordinate frame. If this velocity is
      non-zero the constraints will try to achieve the relative velocity between the objects (in
      the friction plane) to be the wanted surface velocity.
      \param surfaceVelocity - velocity given in geometry's local coordinate frame
      */
      void setSurfaceVelocity( const agx::Vec3f& surfaceVelocity );

      /**
      \return the surface velocity of this geometry in geometry's local coordinate frame as stored
      by the method setSurfaceVelocity()
      */
      const agx::Vec3f& getSurfaceVelocity() const;

      agx::UInt32 getSweepAndPruneIndex() const;

      // Normally updated automatically
      virtual void updateBoundingVolume();

      agx::Physics::GeometryPtr getEntity() const;

      // Normally handled automatically
      virtual void shapeUpdated();

      /**
      \return collection of group names and id's that has been added to this geometry
      */
      agxCollide::GroupIdCollection findGroupIdCollection() const;

    public:
      /**
      \return the frame of the geometry.
      */
      agx::Frame *getFrame();

      /**
      \return the frame of the geometry.
      */
      const agx::Frame *getFrame() const;

      /**
      Set the transform of the geometry. Its frame will move to the specified transform,
      which is given in world coordintaes.
      \param matrix - desired transform for the frame in world coordinates.
      */
      void setTransform( const agx::AffineMatrix4x4& matrix );

      /**
      \return - the local transformation matrix of the geometry's frame, relative to the parent's frame.
      */
      const agx::AffineMatrix4x4& getLocalTransform() const;

      /**
      Assign the local transformation matrix for this geometry, ignoring any eventual parent transformation.
      \param matrix - transformation matrix relative to parent transform for the geometry's frame
      */
      void setLocalTransform( const agx::AffineMatrix4x4& matrix );

      /**
      \return - the relative translate to the parent frame of the geometry's frame.
      */
      agx::Vec3 getLocalPosition() const;

      /**
      Set the position of the geometry relative to its frame's parent frame.
      \param p - local translate of geometry as a 3D vector
      */
      void setLocalPosition( const agx::Vec3& p );

      /**
      Set the position of the geometry relative to its frame's parent frame.
      \param x - local x translate
      \param y - local y translate
      \param z - local z translate
      */
      void setLocalPosition( agx::Real x, agx::Real y, agx::Real z );

      /**
      Set the rotation of the geometry relative to world frame.
      \param q - rotation given as a quaternion
      */
      void setRotation( const agx::Quat& q );

      /**
      Set the rotation of the geometry relative to world frame.
      \param e - rotation given as Euler angles
      */
      void setRotation( const agx::EulerAngles& e);

      /**
      Set the rotation of the geometry relative to world frame.
      \param m - rotation given as an orthogonal transformation matrix
      */
      void setRotation( const agx::OrthoMatrix3x3& m );

      /**
      \return - the geometry's rotation relative to its frame's parent frame.
      */
      agx::Quat getLocalRotation() const;

      /**
      Set the rotation of the geometry relative to its frame's parent frame.
      \param q - rotation given as a quaternion
      */
      void setLocalRotation( const agx::Quat& q );

      /**
      Set the rotation of the geometry relative to its frame's parent frame.
      \param e - rotation given as Euler angles
      */
      void setLocalRotation( const agx::EulerAngles& e );

      /**
      \return - the parent frame of the geometry's frame, or 0 if this frame has no parent
      */
      agx::Frame* getParentFrame();

      /**
      \return - the parent frame of the geometry's frame, or 0 if this frame has no parent
      */
      const agx::Frame* getParentFrame() const;

      /**
      Set the parent frame of this geometry's frame.

      This means that getLocalTranslate, getLocalRotate, getLocalTransform will be given
      in the parents coordinate frame. I.e., this frame's transformation
      will be concatenated with the parents.
      \param frame - new parent frame, 0 to remove parent
      \return true if parent is changed (not the same as before) - otherwise false
      */
      bool setParentFrame( agx::Frame* frame );

      /**
      Set the position of the frame in world coordinates.
      \param p - desired position in world coordinates.
      */
      void setPosition( const agx::Vec3& p );

      /**
      Set the position of the frame in world coordinates.
      \param x - desired x-coordinate in world frame
      \param y - desired y-coordinate in world frame
      \param z - desired z-coordinate in world frame
      */
      void setPosition( agx::Real x, agx::Real y, agx::Real z );

      /**
      \return the current transformation of the geometry in world coordinate space
      */
      agx::AffineMatrix4x4 getTransform() const;

      /**
      \return the current position/translation of the geometry in world coordinate space
      */
      agx::Vec3 getPosition() const;

      /**
      \return the current rotation of the geometry in world coordinate space
      */
      agx::Quat getRotation() const;


    public:
      /* Old interface, use ShapeGroup instead */

      /**
      Add a shape to the geometry. Will create an implicit shape group if needed.

      \param shape The shape to add.
      \param localTransform The shape transformation in local geometry coordinates.
      \return true if the shape was added. False if shape==null.
      */
      bool add( Shape *shape, const agx::AffineMatrix4x4& localTransform = agx::AffineMatrix4x4() );

      /**
      Remove a shape from the geometry.
      \return true if shape was removed. False if shape == null or Geometry does not contain the shape
      */
      bool remove( Shape *shape );

      DOXYGEN_START_INTERNAL_BLOCK()
      /**
      This method return a pointer to the Shape in the Geometry.
      Notice that the instance of this shape might change if you add more than one Shape to the Geometry.
      First, if you add one shape, this method will return for example a agxCollide::Box, but when you add another Shape,
      this instance will change to a agxCollide::ShapeGroup. This means that the UUID of getShape()->getUuid() will change as
      it is a new instance of Shape.

      We recommend that you instead use getShapes().

      \return The shape of the geometry.
      */
      Shape *getShape();

      /**
      This method return a pointer to the Shape in the Geometry.
      Notice that the instance of this shape might change if you add more than one Shape to the Geometry.
      First, if you add one shape, this method will return for example a agxCollide::Box, but when you add another Shape,
      this instance will change to a agxCollide::ShapeGroup. This means that the UUID of getShape()->getUuid() will change as
      it is a new instance of Shape.

      We recommend that you instead use getShapes().

      \return The shape of the geometry.
      */
      const Shape *getShape() const;

      /**
      Set the shape of the geometry.
      \param shape The shape.
      \retval Did setting the shape succeed?
      */
      bool setShape(Shape *shape);
      DOXYGEN_END_INTERNAL_BLOCK()


      /**
      Remove the i:th shape from the Geometry.
      \return true if shape was removed. False if shape == null or Geometry does not contain the shape
      */
      bool remove( size_t index );

      /**
      \return the shapes of the geometry.
      */
      const ShapeRefVector& getShapes() const;

      /**
      Replace a shape in the geometry.
      */
      bool replace( size_t shapeIndex, Shape *newShape, const agx::AffineMatrix4x4& localTransform = agx::AffineMatrix4x4());
      bool replace( Shape *oldShape, Shape *newShape, const agx::AffineMatrix4x4& localTransform = agx::AffineMatrix4x4());

      /**
      Replace the shape transform for the specified shapeIndex.
      \param shapeIndex - Index of the transform that should be replaced. Valid values are 0..numShapes-1
      \param newTransform - The new transform
      \return true if \p shapeIndex is valid.
      */
      bool replaceShapeTransform( size_t shapeIndex, const agx::AffineMatrix4x4& newTransform );

      /**
      Replace the shape transform for the specified shape.
      \param shape - The shape for which the transform should be replaced.
      \param newTransform - The new transform
      \return true if \p shape is valid.
      */
      bool replaceShapeTransform( Shape *shape, const agx::AffineMatrix4x4& newTransform );

      /**
      Clone the shape render data from the geometry shape onto target shape.

      \note - Used internally in the ::clone() method.
      */
      void cloneShapeRenderData(agxCollide::Shape* target, bool shallowCopy) const;

    public:
      /// Specify MergeSplit properties for a Geometry
      class AGXPHYSICS_EXPORT MergeSplitProperties
      {
        public:
          /**
          Constructor.
          ENABLE_SPLIT: true
          ENABLE_MERGE: true
          FORCE_SPLIT_IN_PRE: false
          */
          MergeSplitProperties();

          /// If \p flag is true, this geometry can split merged bodies at contacts/separations
          void setEnableSplit( bool flag );

          /// \return true when this geometry can split merged bodies at contacts/separations, otherwise false.
          bool getEnableSplit( ) const;

          /**
          if \p flag is true, this geometry can result in a merge when it is in resting contact with another geometry.
          */
          void setEnableMerge( bool flag );
          /**
          \return true if this geometry can result in a merge when it is in resting contact with another geometry.
          */
          bool getEnableMerge( ) const;

          /**
          If \p flag is true, this geometry will split merged bodies directly in the pre solve step.
          Default is false, split in the post solve step (to avoid small jerks due to the fact that bodies will fall for one time step
          before a contact constraint is created).
          */
          void setForceSplitInPreStep( bool flag );
          /**
          \return true if this geometry should force split in a pre solve step, instead of default post solve step.
          */
          bool getForceSplitInPreStep( ) const;

        private:
          enum FLAGS {
            ENABLE_SPLIT,
            ENABLE_MERGE,
            FORCE_SPLIT_IN_PRE,
            NUM_ELEMENTS
          };

          typedef agx::BitArray<NUM_ELEMENTS> FlagVector;
          FlagVector m_flags;
      };

      /// \return a reference to the MergeSplit properties
      MergeSplitProperties& getMergeSplitProperties();

      /// \return a reference to the MergeSplit properties
      const MergeSplitProperties& getMergeSplitProperties() const;

    protected:
      /// Destructor
      virtual ~Geometry();

    private:
      friend class agx::MergeSplitBodyStorage;
      friend class agx::Emitter;
      friend class agx::RigidBody;
      friend class agx::Frame;
      friend class GeometryGroup;
      friend class agxCollide::Space;
      friend class agxCollide::SweepAndPrune;
      friend class agxCollide::ShapeGroup;
      friend class agxTerrain::Terrain;

      void propagateTransform();
      void updateFrameStorage();
      ShapeGroup *createWrapperGroup();

      void init(const agx::Name& name);

      mutable agx::PropertyContainerRef m_propertyContainer;

      void setBody( agx::RigidBody *body, const agx::AffineMatrix4x4& invBodyTransform );
      void removeBody();

      /**
      Sets the space for the geometry.
      \param space The space.
      \param id The id.
      \retval Did setting the space succeed?
      */
      bool setSpace( Space* space, agx::Int64 id );

      void setSweepAndPruneIndex( agx::UInt32 index );
      void setEntity( agx::Physics::GeometryPtr entity );
      // void transfer(Space *space);
      void transfer( agxData::EntityStorage *storage );

      //void synchronizeShapeLocalTransformPointers();

      /// Store which body this geometry belonged to before that body was merged with another body.
      void setOriginalBody( agx::RigidBody * );

      size_t findShapeIndex(Shape *shape);

      void enableCollisionsWithGroup(agx::UInt32 groupId);
      void disableCollisionsWithGroup(agx::UInt32 groupId);

      DOXYGEN_START_INTERNAL_BLOCK()
      friend class agx::InternalData;
      /**
      \return internal data
      */
      agx::Referenced* getInternalData() const;

      /**
      Assign internal data.
      */
      void setInternalData( agx::Referenced* data );
      DOXYGEN_END_INTERNAL_BLOCK()

    private:
      agx::Int64                m_id;
      ShapeRef                  m_shape;
      agx::Physics::GeometryRef m_entity;
      agx::RigidBody *          m_rigidBody;
      Space *                   m_space;

      agx::MaterialRef m_material;

      mutable GeometryHashVector m_disabledCollisionsHash;

      agx::FrameRef                     m_frame;
      agx::ref_ptr<agx::Referenced>     m_customData;
      agx::ref_ptr<agx::Referenced>     m_internalData;
      agx::observer_ptr<agx::RigidBody> m_originalBody;
      MergeSplitProperties              m_msProperties;

      // Backward compatibility
      ShapeRefVector                m_singleShapeVector;

      static ShapeRefVector s_defaultChildVector;
      static GroupSet s_emptySet;
  };


  /* Implementation */

  AGX_FORCE_INLINE Shape *Geometry::getShape()
  {
    return m_shape;
  }

  AGX_FORCE_INLINE const Shape *Geometry::getShape() const
  {
    return const_cast<Geometry *>( this )->getShape();
  }

  inline agx::Physics::GeometryPtr Geometry::getEntity() const
  {
    return m_entity;
  }
  AGX_FORCE_INLINE agx::RigidBody *Geometry::getRigidBody()
  {
    return m_rigidBody;
  }

  AGX_FORCE_INLINE const agx::RigidBody *Geometry::getRigidBody() const
  {
    return m_rigidBody;
  }

  AGX_FORCE_INLINE agx::Int64 Geometry::getId() const
  {
    return m_id;
  }

  AGX_FORCE_INLINE bool Geometry::isEnabled() const
  {
    return m_entity.state().enabled();
  }

  AGX_FORCE_INLINE bool Geometry::isSensor() const
  {
    return m_entity.state().sensor();
  }

  AGX_FORCE_INLINE bool Geometry::getEnableMassProperties() const
  {
    return m_entity.state().enableMassProperties();
  }


  inline bool Geometry::isSensorGeneratingContactData() const
  {
    return !m_entity.state().booleanSensor();
  }

  AGX_FORCE_INLINE agx::UInt32 Geometry::getSweepAndPruneIndex() const
  {
    return m_entity.sweepAndPruneIndex();
  }

  AGX_FORCE_INLINE bool Geometry::hasGroup( agx::UInt32 id ) const
  {
    return CollisionGroupManager::hasGroup(m_entity.collisionGroupSet(), id);
  }

  AGX_FORCE_INLINE bool Geometry::hasGroup( const agx::Name& name ) const
  {
    return CollisionGroupManager::hasGroup(m_entity.collisionGroupSet(), name);
  }

  AGX_FORCE_INLINE const agx::Physics::CollisionGroupSetPtr& Geometry::getGroupSet() const
  {
    return m_entity.collisionGroupSet();
  }

  AGX_FORCE_INLINE void Geometry::setGroupSet( agx::Physics::CollisionGroupSetPtr set )
  {
    m_entity.collisionGroupSet() = set;
  }

  AGX_FORCE_INLINE const agx::Material *Geometry::getMaterial() const
  {
    return m_material;
  }

  AGX_FORCE_INLINE agx::Material *Geometry::getMaterial()
  {
    return m_material;
  }

  AGX_FORCE_INLINE void Geometry::setSurfaceVelocity( const agx::Vec3f& surfaceVelocity )
  {
    m_entity.surfaceVelocity() = surfaceVelocity;
  }

  AGX_FORCE_INLINE const agx::Vec3f& Geometry::getSurfaceVelocity( ) const
  {
    return m_entity.surfaceVelocity();
  }

  AGX_FORCE_INLINE void Geometry::setOriginalBody( agx::RigidBody *body )
  {
    m_originalBody = body;
  }

  AGX_FORCE_INLINE agx::RigidBody* Geometry::getOriginalBody( )
  {
    return m_originalBody.get() ? m_originalBody.get() : m_rigidBody;
  }

  AGX_FORCE_INLINE const agx::RigidBody* Geometry::getOriginalBody( ) const
  {
    return m_originalBody.get() ? m_originalBody.get() : m_rigidBody;
  }

  AGX_FORCE_INLINE bool Geometry::getEnable() const
  {
    return m_entity.state().enabled();
  }

  AGX_FORCE_INLINE Space* Geometry::getSpace()
  {
    return m_space;
  }
  AGX_FORCE_INLINE const Space* Geometry::getSpace() const
  {
    return m_space;
  }

  AGX_FORCE_INLINE agx::Frame* Geometry::getFrame()
  {
    return m_frame.get();
  }

  AGX_FORCE_INLINE const agx::Frame* Geometry::getFrame() const
  {
    return m_frame.get();
  }

  AGX_FORCE_INLINE const BoundingAABB* Geometry::getBoundingVolume() const
  {
    return &m_entity.boundingAABB();
  }


  AGX_FORCE_INLINE agx::PropertyContainer* Geometry::getPropertyContainer()
  {
    if ( !m_propertyContainer )
      m_propertyContainer = new agx::PropertyContainer;
    return m_propertyContainer;
  }

  AGX_FORCE_INLINE bool Geometry::hasPropertyContainer() const
  {
    return ( m_propertyContainer.isValid() );
  }

  AGX_FORCE_INLINE const agx::PropertyContainer* Geometry::getPropertyContainer() const
  {
    if ( !m_propertyContainer )
      m_propertyContainer = new agx::PropertyContainer;

    return m_propertyContainer;
  }

  AGX_FORCE_INLINE agx::Vec3f Geometry::calculateSurfaceVelocity( const agxCollide::LocalContactPoint& /* point */, size_t /* index */) const
  {
    return getSurfaceVelocity();
  }

  AGX_FORCE_INLINE Geometry::MergeSplitProperties& Geometry::getMergeSplitProperties()
  {
    return m_msProperties;
  }
  AGX_FORCE_INLINE const Geometry::MergeSplitProperties& Geometry::getMergeSplitProperties() const
  {
    return m_msProperties;
  }

  AGX_FORCE_INLINE void Geometry::MergeSplitProperties::setEnableSplit( bool flag )
  {
    m_flags[ENABLE_SPLIT] = flag;
  }
  AGX_FORCE_INLINE bool Geometry::MergeSplitProperties::getEnableSplit( ) const
  {
    return m_flags[ENABLE_SPLIT];
  }

  AGX_FORCE_INLINE void Geometry::MergeSplitProperties::setEnableMerge( bool flag )
  {
    m_flags[ENABLE_MERGE] = flag;
  }

  AGX_FORCE_INLINE bool Geometry::MergeSplitProperties::getEnableMerge( ) const
  {
    return m_flags[ENABLE_MERGE];
  }

  AGX_FORCE_INLINE void Geometry::MergeSplitProperties::setForceSplitInPreStep( bool flag )
  {
    m_flags[FORCE_SPLIT_IN_PRE] = flag;
  }

  AGX_FORCE_INLINE bool Geometry::MergeSplitProperties::getForceSplitInPreStep( ) const
  {
    return m_flags[FORCE_SPLIT_IN_PRE];
  }


  AGX_FORCE_INLINE agx::AffineMatrix4x4 Geometry::getTransform() const
  {
    if (m_rigidBody)
      return m_entity.transform();
    else
      return m_frame->getMatrix();

  }

  AGX_FORCE_INLINE agx::Vec3 Geometry::getPosition() const
  {
    if (m_rigidBody)
      return m_entity.transform().getTranslate();
    else
      return m_frame->getTranslate();
  }

  AGX_FORCE_INLINE agx::Quat Geometry::getRotation() const
  {
    if (m_rigidBody)
      return m_entity.transform().getRotate();
    return m_frame->getRotate();
  }

  AGX_FORCE_INLINE void Geometry::updateFrameStorage()
  {
    m_frame->setMatrixPointers(agxData::AttributePtr<agx::AffineMatrix4x4>(getEntity(), agx::Physics::GeometryModel::localTransformAttribute), agxData::AttributePtr<agx::AffineMatrix4x4>(getEntity(), agx::Physics::GeometryModel::transformAttribute));
  }

  inline agx::Referenced* Geometry::getInternalData() const
  {
    return m_internalData;
  }
} // namespace agxCollide
