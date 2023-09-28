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

#ifndef AGXCOLLIDE_SHAPE_H
#define AGXCOLLIDE_SHAPE_H


#include <agx/agxPhysics_export.h>

#include <agx/Vec3.h>
#include <agx/SPDMatrix3x3.h>
#include <agx/AffineMatrix4x4.h>

#include <agxCollide/agxCollide.h>
#include <agxCollide/BoundingAABB.h>
#include <agxStream/Serializable.h>

#include <agxCollide/RenderData.h>

#include <agx/Physics/Geometry/ShapeEntity.h>
#include <agx/Physics/Geometry/ShapeGroupEntity.h>

namespace agxTerrain
{
  class Terrain;
}

namespace agxCollide
{
  class Geometry;
  class Space;

  AGX_DECLARE_POINTER_TYPES(Shape);
  AGX_DECLARE_VECTOR_TYPES(Shape);
  AGX_DECLARE_POINTER_TYPES(ShapeGroup);

  /**
  Base class for a shape.
  */
  class AGXPHYSICS_EXPORT Shape : public agx::Referenced, public agxStream::Serializable
  {
    public:

      /// Define the type of a Shape
      enum Type {
        GROUP,
        BOX,
        CAPSULE,
        CYLINDER,
        LINE,
        PLANE,
        SPHERE,
        TRIMESH,
        HEIGHT_FIELD,
        CONVEX,
        WIRE_SHAPE,
        CONE,
        HOLLOW_CONE,
        HOLLOW_CYLINDER,
        NUM_TYPES
      };

      /// \return a string name for the given Type \p type
      static const char* getTypeName(agx::UInt8 type);

    public:

      /**
      Create a clone
      */
      virtual Shape *clone() const;

      /**
      \return The shape type name.
      */
      const char* getTypeName() const;

      /**
      \return The shape type.
      */
      agx::UInt8 getType() const;

      /**
      \return True if the shape is a group.
      */
      bool isGroup() const;

      /**
      Auto-cast shape to group.
      */
      ShapeGroup* asGroup();

      /**
      Auto-cast shape to group.
      */
      const ShapeGroup* asGroup() const;

      /**
      \return the bounding volume of the shape, in global coordinates.
      */
      const BoundingAABB& getBoundingVolume() const;

      /**
      \return true if the shape implements its own support function, used by Minkowski
      difference based collision algorithms.
      */
      virtual bool hasSupportFunction() const;

      /**
      Returns the support point on face of the shape.
      \param supportDirection - Direction in shape coords where to find a support point
      */
      virtual agx::Vec3 getSupportPoint( const agx::Vec3&  supportDirection) const;

      /**
      \return the center (local, relative to the Shape) of the shape. This is in most cases 0,0,0 - but can
      have other positions, e.g. in Convex shapes.
      */
      virtual agx::Vec3 getCenter() const;

      /**
      Return the volume of the shape
      */
      virtual agx::Real getVolume() const = 0;

      /**
      Calculate the inertia for the shape given a mass.
      \return Matrix with inertia in shape coordinate system
      */
      virtual agx::SPDMatrix3x3 calculateInertia(agx::Real mass) const = 0;

      /**
      */
      virtual void propagateTransform(const agx::AffineMatrix4x4& transform);

      /**
      Calculate the bounding volume based on the current Shape parameters including transformation.
      \return new calculated bounding volume
      */
      virtual const BoundingAABB& updateBoundingVolume() = 0;

      /**
      Calculate the bounding volume in the local coordinate system.
      \return the calculated bounding volume.
      */
      virtual BoundingAABB calculateLocalBound() const = 0;

      /**
      Calculate the bounding volume (AABB) into \p localAABB and calculate the bounding radius based
      on the new AABB size.
      \return the radius of the bounding sphere surrounding the AABB.
      */
      virtual agx::Real calculateBoundingRadius( BoundingAABB& localAABB ) const;

      agx::Physics::Geometry::ShapePtr getEntity() const;

      /**
      \return pointer to the associated Geometry, null if none.
      */
      Geometry* getGeometry();

      /**
      \return pointer to the associated Geometry, null if none.
      */
      const Geometry* getGeometry() const;

      /// \return the transform of this Shape
      agx::AffineMatrix4x4 getTransform() const;

      /// \return the transform relative to its parent (Geometry) of this Shape
      agx::AffineMatrix4x4 getLocalTransform() const;


      /**
      \return a bounding box calculated from a transformed (\p transform) AABB (\p localBound)
      */
      static BoundingAABB calculateBound(const agx::AffineMatrix4x4& transform, const BoundingAABB& localBound);

      /**
      \return number of times the shape have been modified. Increments at every set method.
      */
      agx::UInt32 getModifiedCount() const;

      DOXYGEN_START_INTERNAL_BLOCK()


      AGXSTREAM_DECLARE_ABSTRACT_SERIALIZABLE( agxCollide::Shape );
      void store( agxStream::OutputArchive& out ) const override;
      void restore( agxStream::InputArchive& in ) override;
      DOXYGEN_END_INTERNAL_BLOCK()


      /**
      Store render data with the Shape.
      If the shape is a Group, the render data will be stored in all the children and NOT in the actual ShapeGroup.
      It is better to actually access the actual primitives and store the render data there instead of the group.
      */
      virtual void setRenderData( agxCollide::RenderData *renderData );

      /**
      If this is a ShapeGroup, then this method will return nullptr as the render data is stored in the actual children (Box, Mesh etc. shapes)
      You need to go through each child and access the render data from those.
      \return the render data stored with the shape.
      */
      const agxCollide::RenderData* getRenderData( ) const;
      agxCollide::RenderData* getRenderData();


    protected:

      /**
      Constructor
      \param type - Type of the Shape
      \param entity - pointer to an entity. By default a new instance is created.
      */
      Shape(Type type, agx::Physics::Geometry::ShapePtr entity = agx::Physics::Geometry::ShapeModel::createInstance());

      /**
      Increment the number of times the shape have been modified.
      \return the incremented value.
      */
      agx::UInt32 incrementModifiedCount();

      /// Destructor
      virtual ~Shape();

      DOXYGEN_START_INTERNAL_BLOCK()


      Shape& operator=(const Shape&) = delete;

      /// Internal: Make sure to call this from the child class when the shape configuration is changed
      void synchronize();
      DOXYGEN_END_INTERNAL_BLOCK()

    private:
      friend class ShapeGroup;
      friend class Geometry;
      friend class Space;
      friend class agxTerrain::Terrain;

      Shape();
      virtual void setGeometry(Geometry* geometry);
      virtual void transfer(Space* space);

      virtual void updateGeometryPtr();


    protected:

      void syncronizeShapeIndex();

      agx::Physics::Geometry::ShapeRef m_entity;
      Geometry* m_geometry;
      agxCollide::RenderDataRef m_renderData;
      agx::Index m_shapeIndex;

      ShapeGroup* m_parent;
  };


  DOXYGEN_START_INTERNAL_BLOCK()


  AGX_DECLARE_POINTER_TYPES(ShapeGroup);
  class AGXPHYSICS_EXPORT ShapeGroup : public Shape
  {
    public:
      ShapeGroup();
      virtual Shape *clone() const override;

      void addChild(Shape* shape, const agx::AffineMatrix4x4& transform = agx::AffineMatrix4x4());
      void removeChild(Shape* shape);
      void removeChild(size_t index);

      /**
      Set the render data for all this shapegroup's children.
      The ShapeGroup it self will not store the render data.
      */
      virtual void setRenderData( agxCollide::RenderData *renderData ) override;


      agx::Physics::Geometry::ShapeGroupPtr getEntity() const;

      size_t getNumChildren() const;
      Shape* getChild(size_t index);
      const Shape* getChild(size_t index) const;
      const agx::AffineMatrix4x4& getChildTransform(size_t index) const;

      /**
      Set the transform for the \p ith shape
      \param ith - index of shape. Valid range is 0..numShapes-1..
      \return true if \p ith is valid.
      */
      bool setChildTransform(size_t childIndex, const agx::AffineMatrix4x4& transform);


      const ShapeRefVector& getChildren() const;
      const agx::AffineMatrix4x4Vector& getChildTransforms() const;

      bool replaceChild(size_t childIndex, Shape* newChild, const agx::AffineMatrix4x4& transform = agx::AffineMatrix4x4());

      /* Common shape methods */

      /// Calculates inertia. Will be wrong if child shapes overlap.
      virtual agx::SPDMatrix3x3 calculateInertia(agx::Real mass) const override;

      /// Calculates volume. Will be wrong if child shapes overlap.
      virtual agx::Real getVolume() const override;

      /// Calculates center. Will be wrong if child shapes overlap.
      virtual agx::Vec3 getCenter() const override;

      // Computes bounding volume in world coordinate system.
      virtual const BoundingAABB& updateBoundingVolume() override;

      // Computes bounding volume in local coordinate system.
      virtual BoundingAABB calculateLocalBound() const override;

      virtual void propagateTransform(const agx::AffineMatrix4x4& transform) override;

      AGXSTREAM_DECLARE_SERIALIZABLE(agxCollide::ShapeGroup);

    protected:
      virtual ~ShapeGroup();

    private:
      friend class Geometry;

      // void childUpdated(Shape *child);
      virtual void updateGeometryPtr() override;
      virtual void setGeometry(Geometry* geometry) override;
      virtual void transfer(Space* space) override;

    private:
      ShapeRefVector m_children;
  };

  class AGXPHYSICS_EXPORT ShapeIterator
  {
    public:
      ShapeIterator(Geometry* geometry);

      Shape* getShape();
      const Shape* getShape() const;
      agx::AffineMatrix4x4 getLocalTransform() const;

      bool isValid() const;
      void next();

    private:
      void push(Shape* node, const agx::AffineMatrix4x4& transform);

    private:
      agx::Vector<Shape*> m_shapes;
      agx::Vector<agx::AffineMatrix4x4> m_transforms;
      size_t m_index;
  };

  DOXYGEN_END_INTERNAL_BLOCK()


  /* Implementation */

  AGX_FORCE_INLINE agx::UInt8 Shape::getType() const
  {
    return m_entity.type();
  }
  AGX_FORCE_INLINE bool Shape::isGroup() const
  {
    return getType() == GROUP;
  }
  AGX_FORCE_INLINE ShapeGroup* Shape::asGroup()
  {
    agxAssert(this->isGroup());
    return static_cast<ShapeGroup*>(this);
  }
  AGX_FORCE_INLINE const ShapeGroup* Shape::asGroup() const
  {
    return const_cast<Shape*>(this)->asGroup();
  }

  inline agx::Physics::Geometry::ShapePtr Shape::getEntity() const
  {
    return m_entity;
  }
  AGX_FORCE_INLINE Geometry* Shape::getGeometry()
  {
    return m_geometry;
  }
  AGX_FORCE_INLINE const Geometry* Shape::getGeometry() const
  {
    return m_geometry;
  }
  AGX_FORCE_INLINE agx::AffineMatrix4x4 Shape::getTransform() const
  {
    return m_entity.transform();
  }

  AGX_FORCE_INLINE const BoundingAABB& Shape::getBoundingVolume() const
  {
    return m_entity.boundingAABB();
  }

  AGX_FORCE_INLINE BoundingAABB Shape::calculateBound(const agx::AffineMatrix4x4& transform, const BoundingAABB& localBound)
  {
    return BoundingAABB(localBound, transform);
  }

  AGX_FORCE_INLINE size_t ShapeGroup::getNumChildren() const
  {
    return m_children.size();
  }
  AGX_FORCE_INLINE Shape* ShapeGroup::getChild(size_t index)
  {
    return m_children[index];
  }
  AGX_FORCE_INLINE const Shape* ShapeGroup::getChild(size_t index) const
  {
    return m_children[index];
  }
  AGX_FORCE_INLINE const agx::AffineMatrix4x4& ShapeGroup::getChildTransform(size_t index) const
  {
    return getEntity().childTransforms()[index];
  }

  inline agx::Physics::Geometry::ShapeGroupPtr ShapeGroup::getEntity() const
  {
    return m_entity;
  }

  AGX_FORCE_INLINE const ShapeRefVector& ShapeGroup::getChildren() const
  {
    return m_children;
  }
  AGX_FORCE_INLINE const agx::AffineMatrix4x4Vector& ShapeGroup::getChildTransforms() const
  {
    return reinterpret_cast<agx::AffineMatrix4x4Vector&>(getEntity().childTransforms());
  }

  /////////////////////////////////////////////////////////////////////////////////////////////
  AGX_FORCE_INLINE   agx::UInt32 Shape::getModifiedCount() const
  {
    return m_entity.modifiedCount();
  }

  AGX_FORCE_INLINE   agx::UInt32 Shape::incrementModifiedCount()
  {
    return ++m_entity.modifiedCount();
  }


  AGX_FORCE_INLINE Shape* ShapeIterator::getShape()
  {
    agxAssert(this->isValid());
    return m_shapes[m_index];
  }

  AGX_FORCE_INLINE const Shape* ShapeIterator::getShape() const
  {
    return const_cast<ShapeIterator*>(this)->getShape();
  }

  inline agx::AffineMatrix4x4 ShapeIterator::getLocalTransform() const
  {
    agxAssert(this->isValid());
    return m_transforms[m_index];
  }

  AGX_FORCE_INLINE bool ShapeIterator::isValid() const
  {
    return m_index < m_shapes.size();
  }

  AGX_FORCE_INLINE void ShapeIterator::next()
  {
    agxAssert(this->isValid());
    m_index++;
  }

  inline bool Shape::hasSupportFunction() const
  {
    return false;
  }

  inline agx::Vec3 Shape::getSupportPoint( const agx::Vec3&  ) const
  {
    return agx::Vec3(0, 0, 0);
  }

  inline agx::Vec3 Shape::getCenter() const
  {
    return agx::Vec3(0, 0, 0);
  }

}
#endif
