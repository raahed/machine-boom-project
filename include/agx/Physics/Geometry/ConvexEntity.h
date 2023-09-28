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

#ifndef GENERATED_AGX_PHYSICS_GEOMETRY_CONVEX_H_PLUGIN
#define GENERATED_AGX_PHYSICS_GEOMETRY_CONVEX_H_PLUGIN

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
#include <agx/Physics/Geometry/TrimeshEntity.h>


namespace agx
{
  namespace Physics
  {
    namespace Geometry
    {

      class ConvexModel;
      class ConvexData;
      class ConvexPtr;
      class ConvexInstance;
      class ConvexSemantics;


      AGX_DECLARE_POINTER_TYPES(ConvexModel);

      /** 
      Abstract description of the data attributes for the Physics.Geometry.Convex entity.
      */ 
      class AGXPHYSICS_EXPORT ConvexModel : public agx::Physics::Geometry::TrimeshModel
      {
      public:
        typedef ConvexPtr PtrT;

        ConvexModel(const agx::String& name = "Convex");

        /// \return The entity model singleton.
        static ConvexModel* instance();

        /// Create and return a pointer to a new instance in the default storage for this entity model.
        static ConvexPtr createInstance();

        /// \return The default storage for this entity model.
        static agxData::EntityStorage* defaultStorage();

        /// This is part of internal cleanup and should not be called by users
        virtual void shutdownCleanup() override;



        /* Attributes */

      protected:
        virtual ~ConvexModel();
        virtual agxData::EntityData* createData(agxData::EntityStorage* storage) override;
        virtual void configure(agx::TiXmlElement* eEntity) override;
        virtual void initAttributeAccessors() override;
        virtual void construct(agxData::EntityPtr instance) override;
        void construct(agx::Physics::Geometry::ConvexPtr convex);
      };


      DOXYGEN_START_INTERNAL_BLOCK()
      #ifndef AGX_PHYSICS_GEOMETRY_CONVEX_DATA_SET_OVERRIDE
      #define AGX_PHYSICS_GEOMETRY_CONVEX_DATA_SET
      class AGXPHYSICS_EXPORT ConvexData : public agx::Physics::Geometry::TrimeshData
      {
      public:
        ConvexInstance operator[] (size_t index);

      public:
        agxData::Array< ConvexPtr >& instance;

      public:

      public:
        ConvexData(agxData::EntityStorage* storage);
        ConvexData();

      protected:
        virtual ~ConvexData() {}
        virtual void setNumElements(agx::Index numElements) override;

      private:
        ConvexData& operator= (const ConvexData&) { return *this; }

      };
      #endif
      DOXYGEN_END_INTERNAL_BLOCK()


      DOXYGEN_START_INTERNAL_BLOCK()
      class AGXPHYSICS_EXPORT ConvexSemantics : public agx::Physics::Geometry::TrimeshSemantics
      {
      public:

        // Automatic getters

        // Semantics defined by explicit kernels

        // Automatic setters


      protected:
        friend class ConvexPtr;
        friend class ConvexInstance;
        ConvexSemantics();
      };
      DOXYGEN_END_INTERNAL_BLOCK()


      /**
      Pointer to a entity instance of type Physics.Geometry.Convex
      */
      class CALLABLE ConvexPtr : public agx::Physics::Geometry::TrimeshPtr
      {
      public:
        typedef ConvexModel ModelType;
        typedef ConvexData DataType;
        typedef ConvexInstance InstanceType;

      public:
        AGXPHYSICS_EXPORT ConvexPtr();
        AGXPHYSICS_EXPORT ConvexPtr(agxData::EntityStorage* storage, agx::Index id);
        AGXPHYSICS_EXPORT ConvexPtr(const agxData::EntityPtr& ptr);
        AGXPHYSICS_EXPORT ConvexPtr(const agxData::EntityInstance& instance);
        AGXPHYSICS_EXPORT ConvexPtr& operator= (const agxData::EntityPtr& ptr);
        AGXPHYSICS_EXPORT ConvexPtr& operator= (const agxData::EntityInstance& instance);
        AGXPHYSICS_EXPORT ConvexInstance instance();
        AGXPHYSICS_EXPORT const ConvexInstance instance() const;

        AGXPHYSICS_EXPORT ConvexSemantics* operator->();
        AGXPHYSICS_EXPORT const ConvexSemantics* operator->() const;

        ConvexData* getData();
        const ConvexData* getData() const;


        DOXYGEN_START_INTERNAL_BLOCK()
        // Only used by tolua!
        #ifdef AGX_GENERATED_TOLUA_WRAPPER


        #endif
        DOXYGEN_END_INTERNAL_BLOCK()

      };

      // Entity is Referenced
      typedef agxData::EntityRef< ConvexPtr > ConvexRef;


      DOXYGEN_START_INTERNAL_BLOCK()
      class AGXPHYSICS_EXPORT ConvexInstance : public agx::Physics::Geometry::TrimeshInstance
      {
      public:
        ConvexInstance();
        ConvexInstance(ConvexData* data, agx::Index index);
        ConvexInstance(agxData::EntityStorage *storage, agx::Index index);
        ConvexInstance(const agxData::EntityInstance& other);
        ConvexInstance(const agxData::EntityPtr& ptr);

        ConvexData* getData();
        const ConvexData* getData() const;

      public:
      };
      DOXYGEN_END_INTERNAL_BLOCK()



      typedef agx::VectorPOD<ConvexPtr> ConvexPtrVector;
      typedef agxData::Array<ConvexPtr> ConvexPtrArray;



      DOXYGEN_START_INTERNAL_BLOCK()
      /* Implementation */
      //-----------------------------------------------------------------------------------------------------
      //-----------------------------------------------------------------------------------------------------
      inline ConvexInstance agx::Physics::Geometry::ConvexData::operator[] (size_t index) { return ConvexInstance(this, (agx::Index)index); }
      //-----------------------------------------------------------------------------------------------------
      AGX_FORCE_INLINE ConvexPtr::ConvexPtr() {}
      AGX_FORCE_INLINE ConvexPtr::ConvexPtr(agxData::EntityStorage* storage, agx::Index id) : agx::Physics::Geometry::TrimeshPtr(storage, id) {}
      AGX_FORCE_INLINE ConvexPtr::ConvexPtr(const agxData::EntityPtr& ptr) : agx::Physics::Geometry::TrimeshPtr(ptr)
      {
        agxAssertN(!ptr || ptr.isInstanceOf(ConvexModel::instance()),
          "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
          EntityPtr::getModel()->fullPath().c_str(), ConvexModel::instance()->fullPath().c_str());
      }

      AGX_FORCE_INLINE ConvexPtr::ConvexPtr(const agxData::EntityInstance& instance) : agx::Physics::Geometry::TrimeshPtr(instance)
      {
        agxAssertN(!instance || instance.isInstanceOf(ConvexModel::instance()),
          "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
          EntityPtr::getModel()->fullPath().c_str(), ConvexModel::instance()->fullPath().c_str());
      }

      AGX_FORCE_INLINE ConvexPtr& ConvexPtr::operator= (const agxData::EntityPtr& ptr)
      {
        agxData::EntityPtr::operator= (ptr);
        agxAssertN(!ptr || ptr.isInstanceOf(ConvexModel::instance()),
          "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
          EntityPtr::getModel()->fullPath().c_str(), ConvexModel::instance()->fullPath().c_str());
        return *this;
      }

      AGX_FORCE_INLINE ConvexPtr& ConvexPtr::operator= (const agxData::EntityInstance& instance)
      {
        agxData::EntityPtr::operator= (instance);
        agxAssertN(!instance || instance.isInstanceOf(ConvexModel::instance()),
          "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
          EntityPtr::getModel()->fullPath().c_str(), ConvexModel::instance()->fullPath().c_str());
        return *this;
      }

      inline ConvexInstance ConvexPtr::instance() { return agxData::EntityPtr::instance(); }
      inline const ConvexInstance ConvexPtr::instance() const { return agxData::EntityPtr::instance(); }
      AGX_FORCE_INLINE ConvexSemantics* ConvexPtr::operator->() { return (ConvexSemantics* )this; }
      AGX_FORCE_INLINE const ConvexSemantics* ConvexPtr::operator->() const { return (const ConvexSemantics* )this; }
      AGX_FORCE_INLINE ConvexData* ConvexPtr::getData() { return static_cast<ConvexData* >(agxData::EntityPtr::getData()); }
      AGX_FORCE_INLINE const ConvexData* ConvexPtr::getData() const { return static_cast<const ConvexData* >(agxData::EntityPtr::getData()); }

      //-----------------------------------------------------------------------------------------------------
      AGX_FORCE_INLINE ConvexInstance::ConvexInstance() {}
      AGX_FORCE_INLINE ConvexInstance::ConvexInstance(ConvexData* data, agx::Index index) : agx::Physics::Geometry::TrimeshInstance(data, index) {}
      AGX_FORCE_INLINE ConvexInstance::ConvexInstance(agxData::EntityStorage* storage, agx::Index index) : agx::Physics::Geometry::TrimeshInstance(storage, index) {}
      AGX_FORCE_INLINE ConvexInstance::ConvexInstance(const agxData::EntityInstance& other) : agx::Physics::Geometry::TrimeshInstance(other)
      {
        agxAssertN(!other || other.isInstanceOf(ConvexModel::instance()),
          "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
          EntityInstance::getModel()->fullPath().c_str(), ConvexModel::instance()->fullPath().c_str());
      }

      AGX_FORCE_INLINE ConvexInstance::ConvexInstance(const agxData::EntityPtr& ptr) : agx::Physics::Geometry::TrimeshInstance(ptr)
      {
        agxAssertN(!ptr || ptr.isInstanceOf(ConvexModel::instance()),
          "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
          EntityInstance::getModel()->fullPath().c_str(), ConvexModel::instance()->fullPath().c_str());
      }


      AGX_FORCE_INLINE ConvexData* ConvexInstance::getData() { return static_cast<ConvexData* >(agxData::EntityInstance::getData()); }
      AGX_FORCE_INLINE const ConvexData* ConvexInstance::getData() const { return static_cast<const ConvexData* >(agxData::EntityInstance::getData()); }

      //-----------------------------------------------------------------------------------------------------
      AGX_FORCE_INLINE ConvexSemantics::ConvexSemantics() {}
      //-----------------------------------------------------------------------------------------------------
      DOXYGEN_END_INTERNAL_BLOCK()
    }
  }
}

AGX_TYPE_BINDING(agx::Physics::Geometry::ConvexPtr, "Physics.Geometry.ConvexPtr")
AGX_TYPE_BINDING(agx::Physics::Geometry::ConvexInstance, "Physics.Geometry.ConvexInstance")

#ifdef _MSC_VER
# pragma warning(pop)
#endif

#undef AGX_ENTITY_WRAPPER
#undef AGX_ENTITY_NAMESPACE
#endif

