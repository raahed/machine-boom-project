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

#ifndef GENERATED_AGX_PHYSICS_GEOMETRY_PLANE_H_PLUGIN
#define GENERATED_AGX_PHYSICS_GEOMETRY_PLANE_H_PLUGIN

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
#include <agx/Plane.h>


namespace agx
{
  namespace Physics
  {
    namespace Geometry
    {

      class PlaneModel;
      class PlaneData;
      class PlanePtr;
      class PlaneInstance;
      class PlaneSemantics;


      AGX_DECLARE_POINTER_TYPES(PlaneModel);

      /** 
      Abstract description of the data attributes for the Physics.Geometry.Plane entity.
      */ 
      class AGXPHYSICS_EXPORT PlaneModel : public agx::Physics::Geometry::ShapeModel
      {
      public:
        typedef PlanePtr PtrT;

        PlaneModel(const agx::String& name = "Plane");

        /// \return The entity model singleton.
        static PlaneModel* instance();

        /// Create and return a pointer to a new instance in the default storage for this entity model.
        static PlanePtr createInstance();

        /// \return The default storage for this entity model.
        static agxData::EntityStorage* defaultStorage();

        /// This is part of internal cleanup and should not be called by users
        virtual void shutdownCleanup() override;



        /* Attributes */
        static agxData::ScalarAttributeT< agx::Plane >* planeAttribute;

      protected:
        virtual ~PlaneModel();
        virtual agxData::EntityData* createData(agxData::EntityStorage* storage) override;
        virtual void configure(agx::TiXmlElement* eEntity) override;
        virtual void initAttributeAccessors() override;
        virtual void construct(agxData::EntityPtr instance) override;
        void construct(agx::Physics::Geometry::PlanePtr plane);
      };


      DOXYGEN_START_INTERNAL_BLOCK()
      #ifndef AGX_PHYSICS_GEOMETRY_PLANE_DATA_SET_OVERRIDE
      #define AGX_PHYSICS_GEOMETRY_PLANE_DATA_SET
      class AGXPHYSICS_EXPORT PlaneData : public agx::Physics::Geometry::ShapeData
      {
      public:
        PlaneInstance operator[] (size_t index);

      public:
        agxData::Array< PlanePtr >& instance;
        agxData::Array< agx::Plane > plane;

      public:
        typedef agx::Plane planeType;

      public:
        PlaneData(agxData::EntityStorage* storage);
        PlaneData();

      protected:
        virtual ~PlaneData() {}
        virtual void setNumElements(agx::Index numElements) override;

      private:
        PlaneData& operator= (const PlaneData&) { return *this; }

      };
      #endif
      DOXYGEN_END_INTERNAL_BLOCK()


      DOXYGEN_START_INTERNAL_BLOCK()
      class AGXPHYSICS_EXPORT PlaneSemantics : public agx::Physics::Geometry::ShapeSemantics
      {
      public:

        // Automatic getters
        agx::Plane const& getPlane() const;

        // Semantics defined by explicit kernels

        // Automatic setters
        void setPlane(agx::Plane const& value);


      protected:
        friend class PlanePtr;
        friend class PlaneInstance;
        PlaneSemantics();
      };
      DOXYGEN_END_INTERNAL_BLOCK()


      /**
      Pointer to a entity instance of type Physics.Geometry.Plane
      */
      class CALLABLE PlanePtr : public agx::Physics::Geometry::ShapePtr
      {
      public:
        typedef PlaneModel ModelType;
        typedef PlaneData DataType;
        typedef PlaneInstance InstanceType;

      public:
        AGXPHYSICS_EXPORT PlanePtr();
        AGXPHYSICS_EXPORT PlanePtr(agxData::EntityStorage* storage, agx::Index id);
        AGXPHYSICS_EXPORT PlanePtr(const agxData::EntityPtr& ptr);
        AGXPHYSICS_EXPORT PlanePtr(const agxData::EntityInstance& instance);
        AGXPHYSICS_EXPORT PlanePtr& operator= (const agxData::EntityPtr& ptr);
        AGXPHYSICS_EXPORT PlanePtr& operator= (const agxData::EntityInstance& instance);
        AGXPHYSICS_EXPORT PlaneInstance instance();
        AGXPHYSICS_EXPORT const PlaneInstance instance() const;

        AGXPHYSICS_EXPORT PlaneSemantics* operator->();
        AGXPHYSICS_EXPORT const PlaneSemantics* operator->() const;

        PlaneData* getData();
        const PlaneData* getData() const;


        /// \return reference to the plane attribute
        AGXPHYSICS_EXPORT agx::Plane& plane();
        /// \return const reference to the plane attribute
        AGXPHYSICS_EXPORT agx::Plane const& plane() const;

        DOXYGEN_START_INTERNAL_BLOCK()
        // Only used by tolua!
        #ifdef AGX_GENERATED_TOLUA_WRAPPER
        inline agx::Plane const& getPlane() const { return (*this)->getPlane(); }


        inline void setPlane(agx::Plane const& value) { (*this)->setPlane(value); }
        #endif
        DOXYGEN_END_INTERNAL_BLOCK()

      };

      // Entity is Referenced
      typedef agxData::EntityRef< PlanePtr > PlaneRef;


      DOXYGEN_START_INTERNAL_BLOCK()
      class AGXPHYSICS_EXPORT PlaneInstance : public agx::Physics::Geometry::ShapeInstance
      {
      public:
        PlaneInstance();
        PlaneInstance(PlaneData* data, agx::Index index);
        PlaneInstance(agxData::EntityStorage *storage, agx::Index index);
        PlaneInstance(const agxData::EntityInstance& other);
        PlaneInstance(const agxData::EntityPtr& ptr);

        PlaneData* getData();
        const PlaneData* getData() const;

      public:
        /// \return reference to the plane attribute
        agx::Plane& plane();
        /// \return const reference to the plane attribute
        agx::Plane const& plane() const;

      };
      DOXYGEN_END_INTERNAL_BLOCK()



      typedef agx::VectorPOD<PlanePtr> PlanePtrVector;
      typedef agxData::Array<PlanePtr> PlanePtrArray;



      DOXYGEN_START_INTERNAL_BLOCK()
      /* Implementation */
      //-----------------------------------------------------------------------------------------------------
      //-----------------------------------------------------------------------------------------------------
      inline PlaneInstance agx::Physics::Geometry::PlaneData::operator[] (size_t index) { return PlaneInstance(this, (agx::Index)index); }
      //-----------------------------------------------------------------------------------------------------
      AGX_FORCE_INLINE PlanePtr::PlanePtr() {}
      AGX_FORCE_INLINE PlanePtr::PlanePtr(agxData::EntityStorage* storage, agx::Index id) : agx::Physics::Geometry::ShapePtr(storage, id) {}
      AGX_FORCE_INLINE PlanePtr::PlanePtr(const agxData::EntityPtr& ptr) : agx::Physics::Geometry::ShapePtr(ptr)
      {
        agxAssertN(!ptr || ptr.isInstanceOf(PlaneModel::instance()),
          "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
          EntityPtr::getModel()->fullPath().c_str(), PlaneModel::instance()->fullPath().c_str());
      }

      AGX_FORCE_INLINE PlanePtr::PlanePtr(const agxData::EntityInstance& instance) : agx::Physics::Geometry::ShapePtr(instance)
      {
        agxAssertN(!instance || instance.isInstanceOf(PlaneModel::instance()),
          "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
          EntityPtr::getModel()->fullPath().c_str(), PlaneModel::instance()->fullPath().c_str());
      }

      AGX_FORCE_INLINE PlanePtr& PlanePtr::operator= (const agxData::EntityPtr& ptr)
      {
        agxData::EntityPtr::operator= (ptr);
        agxAssertN(!ptr || ptr.isInstanceOf(PlaneModel::instance()),
          "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
          EntityPtr::getModel()->fullPath().c_str(), PlaneModel::instance()->fullPath().c_str());
        return *this;
      }

      AGX_FORCE_INLINE PlanePtr& PlanePtr::operator= (const agxData::EntityInstance& instance)
      {
        agxData::EntityPtr::operator= (instance);
        agxAssertN(!instance || instance.isInstanceOf(PlaneModel::instance()),
          "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
          EntityPtr::getModel()->fullPath().c_str(), PlaneModel::instance()->fullPath().c_str());
        return *this;
      }

      inline PlaneInstance PlanePtr::instance() { return agxData::EntityPtr::instance(); }
      inline const PlaneInstance PlanePtr::instance() const { return agxData::EntityPtr::instance(); }
      AGX_FORCE_INLINE PlaneSemantics* PlanePtr::operator->() { return (PlaneSemantics* )this; }
      AGX_FORCE_INLINE const PlaneSemantics* PlanePtr::operator->() const { return (const PlaneSemantics* )this; }
      AGX_FORCE_INLINE PlaneData* PlanePtr::getData() { return static_cast<PlaneData* >(agxData::EntityPtr::getData()); }
      AGX_FORCE_INLINE const PlaneData* PlanePtr::getData() const { return static_cast<const PlaneData* >(agxData::EntityPtr::getData()); }

      AGX_FORCE_INLINE agx::Plane& PlanePtr::plane() { verifyIndex(); return getData()->plane[calculateIndex()]; }
      AGX_FORCE_INLINE agx::Plane const& PlanePtr::plane() const { verifyIndex(); return getData()->plane[calculateIndex()]; }

      //-----------------------------------------------------------------------------------------------------
      AGX_FORCE_INLINE PlaneInstance::PlaneInstance() {}
      AGX_FORCE_INLINE PlaneInstance::PlaneInstance(PlaneData* data, agx::Index index) : agx::Physics::Geometry::ShapeInstance(data, index) {}
      AGX_FORCE_INLINE PlaneInstance::PlaneInstance(agxData::EntityStorage* storage, agx::Index index) : agx::Physics::Geometry::ShapeInstance(storage, index) {}
      AGX_FORCE_INLINE PlaneInstance::PlaneInstance(const agxData::EntityInstance& other) : agx::Physics::Geometry::ShapeInstance(other)
      {
        agxAssertN(!other || other.isInstanceOf(PlaneModel::instance()),
          "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
          EntityInstance::getModel()->fullPath().c_str(), PlaneModel::instance()->fullPath().c_str());
      }

      AGX_FORCE_INLINE PlaneInstance::PlaneInstance(const agxData::EntityPtr& ptr) : agx::Physics::Geometry::ShapeInstance(ptr)
      {
        agxAssertN(!ptr || ptr.isInstanceOf(PlaneModel::instance()),
          "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
          EntityInstance::getModel()->fullPath().c_str(), PlaneModel::instance()->fullPath().c_str());
      }


      AGX_FORCE_INLINE PlaneData* PlaneInstance::getData() { return static_cast<PlaneData* >(agxData::EntityInstance::getData()); }
      AGX_FORCE_INLINE const PlaneData* PlaneInstance::getData() const { return static_cast<const PlaneData* >(agxData::EntityInstance::getData()); }

      AGX_FORCE_INLINE agx::Plane& PlaneInstance::plane() { verifyIndex(); return getData()->plane[getIndex()]; }
      AGX_FORCE_INLINE agx::Plane const& PlaneInstance::plane() const { verifyIndex(); return getData()->plane[getIndex()]; }

      //-----------------------------------------------------------------------------------------------------
      AGX_FORCE_INLINE PlaneSemantics::PlaneSemantics() {}
      //-----------------------------------------------------------------------------------------------------
      DOXYGEN_END_INTERNAL_BLOCK()
    }
  }
}

AGX_TYPE_BINDING(agx::Physics::Geometry::PlanePtr, "Physics.Geometry.PlanePtr")
AGX_TYPE_BINDING(agx::Physics::Geometry::PlaneInstance, "Physics.Geometry.PlaneInstance")

#ifdef _MSC_VER
# pragma warning(pop)
#endif

#undef AGX_ENTITY_WRAPPER
#undef AGX_ENTITY_NAMESPACE
#endif

