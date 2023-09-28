/*
Copyright 2007-2023. Algoryx Simulation AB.

All AGX source code, intellectual property, documentation, sample code,
tutorials, scene files and technical white papers, are copyrighted, proprietary
and confidential material of Algoryx Simulation AB. You may not download, read,
store, distribute, publish, copy or otherwise disseminate, use or expose this
material unless having a written signed agreement with Algoryx Simulation AB, or
having been advised so by Algoryx Simulation AB for a time limited evaluation,
or having purchased a valid commercial license from Algoryx Simulation AB.

Algoryx Simulation AB disclaims all responsibilities for loss or damage caused
from using this software, unless otherwise stated in written agreements with
Algoryx Simulation AB.
*/

//////////////////////////////////////////////////
// AUTOMATICALLY GENERATED ENTITY, DO NOT EDIT! //
//////////////////////////////////////////////////

#ifndef GENERATED_AGX_PHYSICS_GEOMETRY_SHAPE_H_PLUGIN
#define GENERATED_AGX_PHYSICS_GEOMETRY_SHAPE_H_PLUGIN

#define AGX_ENTITY_WRAPPER 1


#ifdef _MSC_VER
# pragma warning(push)
// warning C4505: 'agxData::VectorAttributeT<T>::print' : unreferenced local function has been removed
# pragma warning( disable : 4505 )
//  warning C4251:  'X' : class 'Y' needs to have dll-interface to be used by clients of class 'Z'
# pragma warning( disable : 4251 )
//  warning C4355: 'this' : used in base member initializer list
# pragma warning( disable : 4355 )
//  marked as __forceinline not inlined
# pragma warning( disable: 4714 )
#endif

#include <agxData/EntityModel.h>
#include <agxData/EntityStorage.h>
#include <agxData/EntityRef.h>
#include <agxData/EntityPtr.h>
#include <agxData/EntityInstance.h>
#include <agx/Integer.h>
#include <agx/Real.h>
#include <agx/macros.h>
#include <agx/ReferencedEntity.h>
#include <agx/Integer.h>
#include <agxCollide/BoundingAABB.h>
#include <agx/Physics/GeometryEntity.h>
#include <agx/AffineMatrix4x4.h>
namespace agxCollide { class Shape; }

namespace agx { namespace Physics { class GeometryPtr; }}

namespace agx
{
  namespace Physics
  {
    namespace Geometry
    {

      class ShapeModel;
      class ShapeData;
      class ShapePtr;
      class ShapeInstance;
      class ShapeSemantics;


      AGX_DECLARE_POINTER_TYPES(ShapeModel);

      /** 
      Abstract description of the data attributes for the Physics.Geometry.Shape entity.
      */ 
      class AGXPHYSICS_EXPORT ShapeModel : public agx::ReferencedModel
      {
      public:
        typedef ShapePtr PtrT;

        ShapeModel(const agx::String& name = "Shape");

        /// \return The entity model singleton.
        static ShapeModel* instance();

        /// Create and return a pointer to a new instance in the default storage for this entity model.
        static ShapePtr createInstance();

        /// \return The default storage for this entity model.
        static agxData::EntityStorage* defaultStorage();

        /// This is part of internal cleanup and should not be called by users
        virtual void shutdownCleanup() override;



        /* Attributes */
        static agxData::ScalarAttributeT< agx::UInt8 >* typeAttribute;
        static agxData::ScalarAttributeT< agx::Bool >* inGroupAttribute;
        static agxData::ScalarAttributeT< agxCollide::BoundingAABB >* boundingAABBAttribute;
        static agxData::ScalarAttributeT< agx::Physics::GeometryPtr >* geometryAttribute;
        static agxData::ScalarAttributeT< agx::AffineMatrix4x4 >* transformAttribute;
        static agxData::ScalarAttributeT< agx::UInt32 >* modifiedCountAttribute;
        static agxData::PointerAttributeT< agxCollide::Shape*>* modelAttribute;

      protected:
        virtual ~ShapeModel();
        virtual agxData::EntityData* createData(agxData::EntityStorage* storage) override;
        virtual void configure(agx::TiXmlElement* eEntity) override;
        virtual void initAttributeAccessors() override;
        virtual void construct(agxData::EntityPtr instance) override;
        void construct(agx::Physics::Geometry::ShapePtr shape);
      };


      DOXYGEN_START_INTERNAL_BLOCK()
      #ifndef AGX_PHYSICS_GEOMETRY_SHAPE_DATA_SET_OVERRIDE
      #define AGX_PHYSICS_GEOMETRY_SHAPE_DATA_SET
      class AGXPHYSICS_EXPORT ShapeData : public agx::ReferencedData
      {
      public:
        ShapeInstance operator[] (size_t index);

      public:
        agxData::Array< ShapePtr >& instance;
        agxData::Array< agx::UInt8 > type;
        agxData::Array< agx::Bool > inGroup;
        agxData::Array< agxCollide::BoundingAABB > boundingAABB;
        agxData::Array< agx::Physics::GeometryPtr > geometry;
        agxData::Array< agx::AffineMatrix4x4 > transform;
        agxData::Array< agx::UInt32 > modifiedCount;
        agxData::Array< agxCollide::Shape* > model;

      public:
        typedef agx::UInt8 typeType;
        typedef agx::Bool inGroupType;
        typedef agxCollide::BoundingAABB boundingAABBType;
        typedef agx::Physics::GeometryPtr geometryType;
        typedef agx::AffineMatrix4x4 transformType;
        typedef agx::UInt32 modifiedCountType;
        typedef agxCollide::Shape* modelType;

      public:
        ShapeData(agxData::EntityStorage* storage);
        ShapeData();

      protected:
        virtual ~ShapeData() {}
        virtual void setNumElements(agx::Index numElements) override;

      private:
        ShapeData& operator= (const ShapeData&) { return *this; }

      };
      #endif
      DOXYGEN_END_INTERNAL_BLOCK()


      DOXYGEN_START_INTERNAL_BLOCK()
      class AGXPHYSICS_EXPORT ShapeSemantics : public agx::ReferencedSemantics
      {
      public:

        // Automatic getters
        agx::UInt8 const& getType() const;
        agx::Bool const& getInGroup() const;
        agxCollide::BoundingAABB const& getBoundingAABB() const;
        agx::Physics::GeometryPtr const& getGeometry() const;
        agx::AffineMatrix4x4 const& getTransform() const;
        agx::UInt32 const& getModifiedCount() const;
        agxCollide::Shape* const& getModel() const;

        // Semantics defined by explicit kernels

        // Automatic setters
        void setType(agx::UInt8 const& value);
        void setInGroup(agx::Bool const& value);
        void setBoundingAABB(agxCollide::BoundingAABB const& value);
        void setGeometry(agx::Physics::GeometryPtr const& value);
        void setTransform(agx::AffineMatrix4x4 const& value);
        void setModifiedCount(agx::UInt32 const& value);
        void setModel(agxCollide::Shape* const& value);


      protected:
        friend class ShapePtr;
        friend class ShapeInstance;
        ShapeSemantics();
      };
      DOXYGEN_END_INTERNAL_BLOCK()


      /**
      Pointer to a entity instance of type Physics.Geometry.Shape
      */
      class CALLABLE ShapePtr : public agx::ReferencedPtr
      {
      public:
        typedef ShapeModel ModelType;
        typedef ShapeData DataType;
        typedef ShapeInstance InstanceType;

      public:
        AGXPHYSICS_EXPORT ShapePtr();
        AGXPHYSICS_EXPORT ShapePtr(agxData::EntityStorage* storage, agx::Index id);
        AGXPHYSICS_EXPORT ShapePtr(const agxData::EntityPtr& ptr);
        AGXPHYSICS_EXPORT ShapePtr(const agxData::EntityInstance& instance);
        AGXPHYSICS_EXPORT ShapePtr& operator= (const agxData::EntityPtr& ptr);
        AGXPHYSICS_EXPORT ShapePtr& operator= (const agxData::EntityInstance& instance);
        AGXPHYSICS_EXPORT ShapeInstance instance();
        AGXPHYSICS_EXPORT const ShapeInstance instance() const;

        AGXPHYSICS_EXPORT ShapeSemantics* operator->();
        AGXPHYSICS_EXPORT const ShapeSemantics* operator->() const;

        ShapeData* getData();
        const ShapeData* getData() const;


        /// \return reference to the type attribute
        AGXPHYSICS_EXPORT agx::UInt8& type();
        /// \return const reference to the type attribute
        AGXPHYSICS_EXPORT agx::UInt8 const& type() const;

        /// \return reference to the inGroup attribute
        AGXPHYSICS_EXPORT agx::Bool& inGroup();
        /// \return const reference to the inGroup attribute
        AGXPHYSICS_EXPORT agx::Bool const& inGroup() const;

        /// \return reference to the boundingAABB attribute
        AGXPHYSICS_EXPORT agxCollide::BoundingAABB& boundingAABB();
        /// \return const reference to the boundingAABB attribute
        AGXPHYSICS_EXPORT agxCollide::BoundingAABB const& boundingAABB() const;

        /// \return reference to the geometry attribute
        AGXPHYSICS_EXPORT agx::Physics::GeometryPtr& geometry();
        /// \return const reference to the geometry attribute
        AGXPHYSICS_EXPORT agx::Physics::GeometryPtr const& geometry() const;

        /// \return reference to the transform attribute
        AGXPHYSICS_EXPORT agx::AffineMatrix4x4& transform();
        /// \return const reference to the transform attribute
        AGXPHYSICS_EXPORT agx::AffineMatrix4x4 const& transform() const;

        /// \return reference to the modifiedCount attribute
        AGXPHYSICS_EXPORT agx::UInt32& modifiedCount();
        /// \return const reference to the modifiedCount attribute
        AGXPHYSICS_EXPORT agx::UInt32 const& modifiedCount() const;

        /// \return reference to the model attribute
        AGXPHYSICS_EXPORT agxCollide::Shape*& model();
        /// \return const reference to the model attribute
        AGXPHYSICS_EXPORT agxCollide::Shape* const& model() const;

        DOXYGEN_START_INTERNAL_BLOCK()
        // Only used by tolua!
        #ifdef AGX_GENERATED_TOLUA_WRAPPER
        inline agx::UInt8 const& getType() const { return (*this)->getType(); }
        inline agx::Bool const& getInGroup() const { return (*this)->getInGroup(); }
        inline agxCollide::BoundingAABB const& getBoundingAABB() const { return (*this)->getBoundingAABB(); }
        inline agx::Physics::GeometryPtr const& getGeometry() const { return (*this)->getGeometry(); }
        inline agx::AffineMatrix4x4 const& getTransform() const { return (*this)->getTransform(); }
        inline agx::UInt32 const& getModifiedCount() const { return (*this)->getModifiedCount(); }
        inline agxCollide::Shape* const& getModel() const { return (*this)->getModel(); }


        inline void setType(agx::UInt8 const& value) { (*this)->setType(value); }
        inline void setInGroup(agx::Bool const& value) { (*this)->setInGroup(value); }
        inline void setBoundingAABB(agxCollide::BoundingAABB const& value) { (*this)->setBoundingAABB(value); }
        inline void setGeometry(agx::Physics::GeometryPtr const& value) { (*this)->setGeometry(value); }
        inline void setTransform(agx::AffineMatrix4x4 const& value) { (*this)->setTransform(value); }
        inline void setModifiedCount(agx::UInt32 const& value) { (*this)->setModifiedCount(value); }
        inline void setModel(agxCollide::Shape* const& value) { (*this)->setModel(value); }
        #endif
        DOXYGEN_END_INTERNAL_BLOCK()

      };

      // Entity is Referenced
      typedef agxData::EntityRef< ShapePtr > ShapeRef;


      DOXYGEN_START_INTERNAL_BLOCK()
      class AGXPHYSICS_EXPORT ShapeInstance : public agx::ReferencedInstance
      {
      public:
        ShapeInstance();
        ShapeInstance(ShapeData* data, agx::Index index);
        ShapeInstance(agxData::EntityStorage *storage, agx::Index index);
        ShapeInstance(const agxData::EntityInstance& other);
        ShapeInstance(const agxData::EntityPtr& ptr);

        ShapeData* getData();
        const ShapeData* getData() const;

      public:
        /// \return reference to the type attribute
        agx::UInt8& type();
        /// \return const reference to the type attribute
        agx::UInt8 const& type() const;

        /// \return reference to the inGroup attribute
        agx::Bool& inGroup();
        /// \return const reference to the inGroup attribute
        agx::Bool const& inGroup() const;

        /// \return reference to the boundingAABB attribute
        agxCollide::BoundingAABB& boundingAABB();
        /// \return const reference to the boundingAABB attribute
        agxCollide::BoundingAABB const& boundingAABB() const;

        /// \return reference to the geometry attribute
        agx::Physics::GeometryPtr& geometry();
        /// \return const reference to the geometry attribute
        agx::Physics::GeometryPtr const& geometry() const;

        /// \return reference to the transform attribute
        agx::AffineMatrix4x4& transform();
        /// \return const reference to the transform attribute
        agx::AffineMatrix4x4 const& transform() const;

        /// \return reference to the modifiedCount attribute
        agx::UInt32& modifiedCount();
        /// \return const reference to the modifiedCount attribute
        agx::UInt32 const& modifiedCount() const;

        /// \return reference to the model attribute
        agxCollide::Shape*& model();
        /// \return const reference to the model attribute
        agxCollide::Shape* const& model() const;

      };
      DOXYGEN_END_INTERNAL_BLOCK()



      typedef agx::VectorPOD<ShapePtr> ShapePtrVector;
      typedef agxData::Array<ShapePtr> ShapePtrArray;



      DOXYGEN_START_INTERNAL_BLOCK()
      /* Implementation */
      //-----------------------------------------------------------------------------------------------------
      //-----------------------------------------------------------------------------------------------------
      inline ShapeInstance agx::Physics::Geometry::ShapeData::operator[] (size_t index) { return ShapeInstance(this, (agx::Index)index); }
      //-----------------------------------------------------------------------------------------------------
      AGX_FORCE_INLINE ShapePtr::ShapePtr() {}
      AGX_FORCE_INLINE ShapePtr::ShapePtr(agxData::EntityStorage* storage, agx::Index id) : agx::ReferencedPtr(storage, id) {}
      AGX_FORCE_INLINE ShapePtr::ShapePtr(const agxData::EntityPtr& ptr) : agx::ReferencedPtr(ptr)
      {
        agxAssertN(!ptr || ptr.isInstanceOf(ShapeModel::instance()),
          "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
          EntityPtr::getModel()->fullPath().c_str(), ShapeModel::instance()->fullPath().c_str());
      }

      AGX_FORCE_INLINE ShapePtr::ShapePtr(const agxData::EntityInstance& instance) : agx::ReferencedPtr(instance)
      {
        agxAssertN(!instance || instance.isInstanceOf(ShapeModel::instance()),
          "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
          EntityPtr::getModel()->fullPath().c_str(), ShapeModel::instance()->fullPath().c_str());
      }

      AGX_FORCE_INLINE ShapePtr& ShapePtr::operator= (const agxData::EntityPtr& ptr)
      {
        agxData::EntityPtr::operator= (ptr);
        agxAssertN(!ptr || ptr.isInstanceOf(ShapeModel::instance()),
          "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
          EntityPtr::getModel()->fullPath().c_str(), ShapeModel::instance()->fullPath().c_str());
        return *this;
      }

      AGX_FORCE_INLINE ShapePtr& ShapePtr::operator= (const agxData::EntityInstance& instance)
      {
        agxData::EntityPtr::operator= (instance);
        agxAssertN(!instance || instance.isInstanceOf(ShapeModel::instance()),
          "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
          EntityPtr::getModel()->fullPath().c_str(), ShapeModel::instance()->fullPath().c_str());
        return *this;
      }

      inline ShapeInstance ShapePtr::instance() { return agxData::EntityPtr::instance(); }
      inline const ShapeInstance ShapePtr::instance() const { return agxData::EntityPtr::instance(); }
      AGX_FORCE_INLINE ShapeSemantics* ShapePtr::operator->() { return (ShapeSemantics* )this; }
      AGX_FORCE_INLINE const ShapeSemantics* ShapePtr::operator->() const { return (const ShapeSemantics* )this; }
      AGX_FORCE_INLINE ShapeData* ShapePtr::getData() { return static_cast<ShapeData* >(agxData::EntityPtr::getData()); }
      AGX_FORCE_INLINE const ShapeData* ShapePtr::getData() const { return static_cast<const ShapeData* >(agxData::EntityPtr::getData()); }

      AGX_FORCE_INLINE agx::UInt8& ShapePtr::type() { verifyIndex(); return getData()->type[calculateIndex()]; }
      AGX_FORCE_INLINE agx::UInt8 const& ShapePtr::type() const { verifyIndex(); return getData()->type[calculateIndex()]; }

      AGX_FORCE_INLINE agx::Bool& ShapePtr::inGroup() { verifyIndex(); return getData()->inGroup[calculateIndex()]; }
      AGX_FORCE_INLINE agx::Bool const& ShapePtr::inGroup() const { verifyIndex(); return getData()->inGroup[calculateIndex()]; }

      AGX_FORCE_INLINE agxCollide::BoundingAABB& ShapePtr::boundingAABB() { verifyIndex(); return getData()->boundingAABB[calculateIndex()]; }
      AGX_FORCE_INLINE agxCollide::BoundingAABB const& ShapePtr::boundingAABB() const { verifyIndex(); return getData()->boundingAABB[calculateIndex()]; }

      AGX_FORCE_INLINE agx::Physics::GeometryPtr& ShapePtr::geometry() { verifyIndex(); return getData()->geometry[calculateIndex()]; }
      AGX_FORCE_INLINE agx::Physics::GeometryPtr const& ShapePtr::geometry() const { verifyIndex(); return getData()->geometry[calculateIndex()]; }

      AGX_FORCE_INLINE agx::AffineMatrix4x4& ShapePtr::transform() { verifyIndex(); return getData()->transform[calculateIndex()]; }
      AGX_FORCE_INLINE agx::AffineMatrix4x4 const& ShapePtr::transform() const { verifyIndex(); return getData()->transform[calculateIndex()]; }

      AGX_FORCE_INLINE agx::UInt32& ShapePtr::modifiedCount() { verifyIndex(); return getData()->modifiedCount[calculateIndex()]; }
      AGX_FORCE_INLINE agx::UInt32 const& ShapePtr::modifiedCount() const { verifyIndex(); return getData()->modifiedCount[calculateIndex()]; }

      AGX_FORCE_INLINE agxCollide::Shape*& ShapePtr::model() { verifyIndex(); return getData()->model[calculateIndex()]; }
      AGX_FORCE_INLINE agxCollide::Shape* const& ShapePtr::model() const { verifyIndex(); return getData()->model[calculateIndex()]; }

      //-----------------------------------------------------------------------------------------------------
      AGX_FORCE_INLINE ShapeInstance::ShapeInstance() {}
      AGX_FORCE_INLINE ShapeInstance::ShapeInstance(ShapeData* data, agx::Index index) : agx::ReferencedInstance(data, index) {}
      AGX_FORCE_INLINE ShapeInstance::ShapeInstance(agxData::EntityStorage* storage, agx::Index index) : agx::ReferencedInstance(storage, index) {}
      AGX_FORCE_INLINE ShapeInstance::ShapeInstance(const agxData::EntityInstance& other) : agx::ReferencedInstance(other)
      {
        agxAssertN(!other || other.isInstanceOf(ShapeModel::instance()),
          "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
          EntityInstance::getModel()->fullPath().c_str(), ShapeModel::instance()->fullPath().c_str());
      }

      AGX_FORCE_INLINE ShapeInstance::ShapeInstance(const agxData::EntityPtr& ptr) : agx::ReferencedInstance(ptr)
      {
        agxAssertN(!ptr || ptr.isInstanceOf(ShapeModel::instance()),
          "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
          EntityInstance::getModel()->fullPath().c_str(), ShapeModel::instance()->fullPath().c_str());
      }


      AGX_FORCE_INLINE ShapeData* ShapeInstance::getData() { return static_cast<ShapeData* >(agxData::EntityInstance::getData()); }
      AGX_FORCE_INLINE const ShapeData* ShapeInstance::getData() const { return static_cast<const ShapeData* >(agxData::EntityInstance::getData()); }

      AGX_FORCE_INLINE agx::UInt8& ShapeInstance::type() { verifyIndex(); return getData()->type[getIndex()]; }
      AGX_FORCE_INLINE agx::UInt8 const& ShapeInstance::type() const { verifyIndex(); return getData()->type[getIndex()]; }

      AGX_FORCE_INLINE agx::Bool& ShapeInstance::inGroup() { verifyIndex(); return getData()->inGroup[getIndex()]; }
      AGX_FORCE_INLINE agx::Bool const& ShapeInstance::inGroup() const { verifyIndex(); return getData()->inGroup[getIndex()]; }

      AGX_FORCE_INLINE agxCollide::BoundingAABB& ShapeInstance::boundingAABB() { verifyIndex(); return getData()->boundingAABB[getIndex()]; }
      AGX_FORCE_INLINE agxCollide::BoundingAABB const& ShapeInstance::boundingAABB() const { verifyIndex(); return getData()->boundingAABB[getIndex()]; }

      AGX_FORCE_INLINE agx::Physics::GeometryPtr& ShapeInstance::geometry() { verifyIndex(); return getData()->geometry[getIndex()]; }
      AGX_FORCE_INLINE agx::Physics::GeometryPtr const& ShapeInstance::geometry() const { verifyIndex(); return getData()->geometry[getIndex()]; }

      AGX_FORCE_INLINE agx::AffineMatrix4x4& ShapeInstance::transform() { verifyIndex(); return getData()->transform[getIndex()]; }
      AGX_FORCE_INLINE agx::AffineMatrix4x4 const& ShapeInstance::transform() const { verifyIndex(); return getData()->transform[getIndex()]; }

      AGX_FORCE_INLINE agx::UInt32& ShapeInstance::modifiedCount() { verifyIndex(); return getData()->modifiedCount[getIndex()]; }
      AGX_FORCE_INLINE agx::UInt32 const& ShapeInstance::modifiedCount() const { verifyIndex(); return getData()->modifiedCount[getIndex()]; }

      AGX_FORCE_INLINE agxCollide::Shape*& ShapeInstance::model() { verifyIndex(); return getData()->model[getIndex()]; }
      AGX_FORCE_INLINE agxCollide::Shape* const& ShapeInstance::model() const { verifyIndex(); return getData()->model[getIndex()]; }

      //-----------------------------------------------------------------------------------------------------
      AGX_FORCE_INLINE ShapeSemantics::ShapeSemantics() {}
      //-----------------------------------------------------------------------------------------------------
      DOXYGEN_END_INTERNAL_BLOCK()
    }
  }
}

AGX_TYPE_BINDING(agx::Physics::Geometry::ShapePtr, "Physics.Geometry.ShapePtr")
AGX_TYPE_BINDING(agx::Physics::Geometry::ShapeInstance, "Physics.Geometry.ShapeInstance")

#ifdef _MSC_VER
# pragma warning(pop)
#endif

#undef AGX_ENTITY_WRAPPER
#undef AGX_ENTITY_NAMESPACE
#endif

