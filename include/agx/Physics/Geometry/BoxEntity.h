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

#ifndef GENERATED_AGX_PHYSICS_GEOMETRY_BOX_H_PLUGIN
#define GENERATED_AGX_PHYSICS_GEOMETRY_BOX_H_PLUGIN

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
#include <agx/Physics/Geometry/ShapeEntity.h>
#include <agx/Vec3.h>


namespace agx
{
  namespace Physics
  {
    namespace Geometry
    {

      class BoxModel;
      class BoxData;
      class BoxPtr;
      class BoxInstance;
      class BoxSemantics;


      AGX_DECLARE_POINTER_TYPES(BoxModel);

      /** 
      Abstract description of the data attributes for the Physics.Geometry.Box entity.
      */ 
      class AGXPHYSICS_EXPORT BoxModel : public agx::Physics::Geometry::ShapeModel
      {
      public:
        typedef BoxPtr PtrT;

        BoxModel(const agx::String& name = "Box");

        /// \return The entity model singleton.
        static BoxModel* instance();

        /// Create and return a pointer to a new instance in the default storage for this entity model.
        static BoxPtr createInstance();

        /// \return The default storage for this entity model.
        static agxData::EntityStorage* defaultStorage();

        /// This is part of internal cleanup and should not be called by users
        virtual void shutdownCleanup() override;



        /* Attributes */
        static agxData::ScalarAttributeT< agx::Vec3 >* halfExtentsAttribute;

      protected:
        virtual ~BoxModel();
        virtual agxData::EntityData* createData(agxData::EntityStorage* storage) override;
        virtual void configure(agx::TiXmlElement* eEntity) override;
        virtual void initAttributeAccessors() override;
        virtual void construct(agxData::EntityPtr instance) override;
        void construct(agx::Physics::Geometry::BoxPtr box);
      };


      DOXYGEN_START_INTERNAL_BLOCK()
      #ifndef AGX_PHYSICS_GEOMETRY_BOX_DATA_SET_OVERRIDE
      #define AGX_PHYSICS_GEOMETRY_BOX_DATA_SET
      class AGXPHYSICS_EXPORT BoxData : public agx::Physics::Geometry::ShapeData
      {
      public:
        BoxInstance operator[] (size_t index);

      public:
        agxData::Array< BoxPtr >& instance;
        agxData::Array< agx::Vec3 > halfExtents;

      public:
        typedef agx::Vec3 halfExtentsType;

      public:
        BoxData(agxData::EntityStorage* storage);
        BoxData();

      protected:
        virtual ~BoxData() {}
        virtual void setNumElements(agx::Index numElements) override;

      private:
        BoxData& operator= (const BoxData&) { return *this; }

      };
      #endif
      DOXYGEN_END_INTERNAL_BLOCK()


      DOXYGEN_START_INTERNAL_BLOCK()
      class AGXPHYSICS_EXPORT BoxSemantics : public agx::Physics::Geometry::ShapeSemantics
      {
      public:

        // Automatic getters
        agx::Vec3 const& getHalfExtents() const;

        // Semantics defined by explicit kernels

        // Automatic setters
        void setHalfExtents(agx::Vec3 const& value);


      protected:
        friend class BoxPtr;
        friend class BoxInstance;
        BoxSemantics();
      };
      DOXYGEN_END_INTERNAL_BLOCK()


      /**
      Pointer to a entity instance of type Physics.Geometry.Box
      */
      class CALLABLE BoxPtr : public agx::Physics::Geometry::ShapePtr
      {
      public:
        typedef BoxModel ModelType;
        typedef BoxData DataType;
        typedef BoxInstance InstanceType;

      public:
        AGXPHYSICS_EXPORT BoxPtr();
        AGXPHYSICS_EXPORT BoxPtr(agxData::EntityStorage* storage, agx::Index id);
        AGXPHYSICS_EXPORT BoxPtr(const agxData::EntityPtr& ptr);
        AGXPHYSICS_EXPORT BoxPtr(const agxData::EntityInstance& instance);
        AGXPHYSICS_EXPORT BoxPtr& operator= (const agxData::EntityPtr& ptr);
        AGXPHYSICS_EXPORT BoxPtr& operator= (const agxData::EntityInstance& instance);
        AGXPHYSICS_EXPORT BoxInstance instance();
        AGXPHYSICS_EXPORT const BoxInstance instance() const;

        AGXPHYSICS_EXPORT BoxSemantics* operator->();
        AGXPHYSICS_EXPORT const BoxSemantics* operator->() const;

        BoxData* getData();
        const BoxData* getData() const;


        /// \return reference to the halfExtents attribute
        AGXPHYSICS_EXPORT agx::Vec3& halfExtents();
        /// \return const reference to the halfExtents attribute
        AGXPHYSICS_EXPORT agx::Vec3 const& halfExtents() const;

        DOXYGEN_START_INTERNAL_BLOCK()
        // Only used by tolua!
        #ifdef AGX_GENERATED_TOLUA_WRAPPER
        inline agx::Vec3 const& getHalfExtents() const { return (*this)->getHalfExtents(); }


        inline void setHalfExtents(agx::Vec3 const& value) { (*this)->setHalfExtents(value); }
        #endif
        DOXYGEN_END_INTERNAL_BLOCK()

      };

      // Entity is Referenced
      typedef agxData::EntityRef< BoxPtr > BoxRef;


      DOXYGEN_START_INTERNAL_BLOCK()
      class AGXPHYSICS_EXPORT BoxInstance : public agx::Physics::Geometry::ShapeInstance
      {
      public:
        BoxInstance();
        BoxInstance(BoxData* data, agx::Index index);
        BoxInstance(agxData::EntityStorage *storage, agx::Index index);
        BoxInstance(const agxData::EntityInstance& other);
        BoxInstance(const agxData::EntityPtr& ptr);

        BoxData* getData();
        const BoxData* getData() const;

      public:
        /// \return reference to the halfExtents attribute
        agx::Vec3& halfExtents();
        /// \return const reference to the halfExtents attribute
        agx::Vec3 const& halfExtents() const;

      };
      DOXYGEN_END_INTERNAL_BLOCK()



      typedef agx::VectorPOD<BoxPtr> BoxPtrVector;
      typedef agxData::Array<BoxPtr> BoxPtrArray;



      DOXYGEN_START_INTERNAL_BLOCK()
      /* Implementation */
      //-----------------------------------------------------------------------------------------------------
      //-----------------------------------------------------------------------------------------------------
      inline BoxInstance agx::Physics::Geometry::BoxData::operator[] (size_t index) { return BoxInstance(this, (agx::Index)index); }
      //-----------------------------------------------------------------------------------------------------
      AGX_FORCE_INLINE BoxPtr::BoxPtr() {}
      AGX_FORCE_INLINE BoxPtr::BoxPtr(agxData::EntityStorage* storage, agx::Index id) : agx::Physics::Geometry::ShapePtr(storage, id) {}
      AGX_FORCE_INLINE BoxPtr::BoxPtr(const agxData::EntityPtr& ptr) : agx::Physics::Geometry::ShapePtr(ptr)
      {
        agxAssertN(!ptr || ptr.isInstanceOf(BoxModel::instance()),
          "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
          EntityPtr::getModel()->fullPath().c_str(), BoxModel::instance()->fullPath().c_str());
      }

      AGX_FORCE_INLINE BoxPtr::BoxPtr(const agxData::EntityInstance& instance) : agx::Physics::Geometry::ShapePtr(instance)
      {
        agxAssertN(!instance || instance.isInstanceOf(BoxModel::instance()),
          "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
          EntityPtr::getModel()->fullPath().c_str(), BoxModel::instance()->fullPath().c_str());
      }

      AGX_FORCE_INLINE BoxPtr& BoxPtr::operator= (const agxData::EntityPtr& ptr)
      {
        agxData::EntityPtr::operator= (ptr);
        agxAssertN(!ptr || ptr.isInstanceOf(BoxModel::instance()),
          "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
          EntityPtr::getModel()->fullPath().c_str(), BoxModel::instance()->fullPath().c_str());
        return *this;
      }

      AGX_FORCE_INLINE BoxPtr& BoxPtr::operator= (const agxData::EntityInstance& instance)
      {
        agxData::EntityPtr::operator= (instance);
        agxAssertN(!instance || instance.isInstanceOf(BoxModel::instance()),
          "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
          EntityPtr::getModel()->fullPath().c_str(), BoxModel::instance()->fullPath().c_str());
        return *this;
      }

      inline BoxInstance BoxPtr::instance() { return agxData::EntityPtr::instance(); }
      inline const BoxInstance BoxPtr::instance() const { return agxData::EntityPtr::instance(); }
      AGX_FORCE_INLINE BoxSemantics* BoxPtr::operator->() { return (BoxSemantics* )this; }
      AGX_FORCE_INLINE const BoxSemantics* BoxPtr::operator->() const { return (const BoxSemantics* )this; }
      AGX_FORCE_INLINE BoxData* BoxPtr::getData() { return static_cast<BoxData* >(agxData::EntityPtr::getData()); }
      AGX_FORCE_INLINE const BoxData* BoxPtr::getData() const { return static_cast<const BoxData* >(agxData::EntityPtr::getData()); }

      AGX_FORCE_INLINE agx::Vec3& BoxPtr::halfExtents() { verifyIndex(); return getData()->halfExtents[calculateIndex()]; }
      AGX_FORCE_INLINE agx::Vec3 const& BoxPtr::halfExtents() const { verifyIndex(); return getData()->halfExtents[calculateIndex()]; }

      //-----------------------------------------------------------------------------------------------------
      AGX_FORCE_INLINE BoxInstance::BoxInstance() {}
      AGX_FORCE_INLINE BoxInstance::BoxInstance(BoxData* data, agx::Index index) : agx::Physics::Geometry::ShapeInstance(data, index) {}
      AGX_FORCE_INLINE BoxInstance::BoxInstance(agxData::EntityStorage* storage, agx::Index index) : agx::Physics::Geometry::ShapeInstance(storage, index) {}
      AGX_FORCE_INLINE BoxInstance::BoxInstance(const agxData::EntityInstance& other) : agx::Physics::Geometry::ShapeInstance(other)
      {
        agxAssertN(!other || other.isInstanceOf(BoxModel::instance()),
          "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
          EntityInstance::getModel()->fullPath().c_str(), BoxModel::instance()->fullPath().c_str());
      }

      AGX_FORCE_INLINE BoxInstance::BoxInstance(const agxData::EntityPtr& ptr) : agx::Physics::Geometry::ShapeInstance(ptr)
      {
        agxAssertN(!ptr || ptr.isInstanceOf(BoxModel::instance()),
          "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
          EntityInstance::getModel()->fullPath().c_str(), BoxModel::instance()->fullPath().c_str());
      }


      AGX_FORCE_INLINE BoxData* BoxInstance::getData() { return static_cast<BoxData* >(agxData::EntityInstance::getData()); }
      AGX_FORCE_INLINE const BoxData* BoxInstance::getData() const { return static_cast<const BoxData* >(agxData::EntityInstance::getData()); }

      AGX_FORCE_INLINE agx::Vec3& BoxInstance::halfExtents() { verifyIndex(); return getData()->halfExtents[getIndex()]; }
      AGX_FORCE_INLINE agx::Vec3 const& BoxInstance::halfExtents() const { verifyIndex(); return getData()->halfExtents[getIndex()]; }

      //-----------------------------------------------------------------------------------------------------
      AGX_FORCE_INLINE BoxSemantics::BoxSemantics() {}
      //-----------------------------------------------------------------------------------------------------
      DOXYGEN_END_INTERNAL_BLOCK()
    }
  }
}

AGX_TYPE_BINDING(agx::Physics::Geometry::BoxPtr, "Physics.Geometry.BoxPtr")
AGX_TYPE_BINDING(agx::Physics::Geometry::BoxInstance, "Physics.Geometry.BoxInstance")

#ifdef _MSC_VER
# pragma warning(pop)
#endif

#undef AGX_ENTITY_WRAPPER
#undef AGX_ENTITY_NAMESPACE
#endif

