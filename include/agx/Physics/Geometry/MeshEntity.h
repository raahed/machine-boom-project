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

#ifndef GENERATED_AGX_PHYSICS_GEOMETRY_MESH_H_PLUGIN
#define GENERATED_AGX_PHYSICS_GEOMETRY_MESH_H_PLUGIN

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


namespace agx
{
  namespace Physics
  {
    namespace Geometry
    {

      class MeshModel;
      class MeshData;
      class MeshPtr;
      class MeshInstance;
      class MeshSemantics;


      AGX_DECLARE_POINTER_TYPES(MeshModel);

      /** 
      Abstract description of the data attributes for the Physics.Geometry.Mesh entity.
      */ 
      class AGXPHYSICS_EXPORT MeshModel : public agx::Physics::Geometry::ShapeModel
      {
      public:
        typedef MeshPtr PtrT;

        MeshModel(const agx::String& name = "Mesh");

        /// \return The entity model singleton.
        static MeshModel* instance();

        /// Create and return a pointer to a new instance in the default storage for this entity model.
        static MeshPtr createInstance();

        /// \return The default storage for this entity model.
        static agxData::EntityStorage* defaultStorage();

        /// This is part of internal cleanup and should not be called by users
        virtual void shutdownCleanup() override;



        /* Attributes */

      protected:
        virtual ~MeshModel();
        virtual agxData::EntityData* createData(agxData::EntityStorage* storage) override;
        virtual void configure(agx::TiXmlElement* eEntity) override;
        virtual void initAttributeAccessors() override;
        virtual void construct(agxData::EntityPtr instance) override;
        void construct(agx::Physics::Geometry::MeshPtr mesh);
      };


      DOXYGEN_START_INTERNAL_BLOCK()
      #ifndef AGX_PHYSICS_GEOMETRY_MESH_DATA_SET_OVERRIDE
      #define AGX_PHYSICS_GEOMETRY_MESH_DATA_SET
      class AGXPHYSICS_EXPORT MeshData : public agx::Physics::Geometry::ShapeData
      {
      public:
        MeshInstance operator[] (size_t index);

      public:
        agxData::Array< MeshPtr >& instance;

      public:

      public:
        MeshData(agxData::EntityStorage* storage);
        MeshData();

      protected:
        virtual ~MeshData() {}
        virtual void setNumElements(agx::Index numElements) override;

      private:
        MeshData& operator= (const MeshData&) { return *this; }

      };
      #endif
      DOXYGEN_END_INTERNAL_BLOCK()


      DOXYGEN_START_INTERNAL_BLOCK()
      class AGXPHYSICS_EXPORT MeshSemantics : public agx::Physics::Geometry::ShapeSemantics
      {
      public:

        // Automatic getters

        // Semantics defined by explicit kernels

        // Automatic setters


      protected:
        friend class MeshPtr;
        friend class MeshInstance;
        MeshSemantics();
      };
      DOXYGEN_END_INTERNAL_BLOCK()


      /**
      Pointer to a entity instance of type Physics.Geometry.Mesh
      */
      class CALLABLE MeshPtr : public agx::Physics::Geometry::ShapePtr
      {
      public:
        typedef MeshModel ModelType;
        typedef MeshData DataType;
        typedef MeshInstance InstanceType;

      public:
        AGXPHYSICS_EXPORT MeshPtr();
        AGXPHYSICS_EXPORT MeshPtr(agxData::EntityStorage* storage, agx::Index id);
        AGXPHYSICS_EXPORT MeshPtr(const agxData::EntityPtr& ptr);
        AGXPHYSICS_EXPORT MeshPtr(const agxData::EntityInstance& instance);
        AGXPHYSICS_EXPORT MeshPtr& operator= (const agxData::EntityPtr& ptr);
        AGXPHYSICS_EXPORT MeshPtr& operator= (const agxData::EntityInstance& instance);
        AGXPHYSICS_EXPORT MeshInstance instance();
        AGXPHYSICS_EXPORT const MeshInstance instance() const;

        AGXPHYSICS_EXPORT MeshSemantics* operator->();
        AGXPHYSICS_EXPORT const MeshSemantics* operator->() const;

        MeshData* getData();
        const MeshData* getData() const;


        DOXYGEN_START_INTERNAL_BLOCK()
        // Only used by tolua!
        #ifdef AGX_GENERATED_TOLUA_WRAPPER


        #endif
        DOXYGEN_END_INTERNAL_BLOCK()

      };

      // Entity is Referenced
      typedef agxData::EntityRef< MeshPtr > MeshRef;


      DOXYGEN_START_INTERNAL_BLOCK()
      class AGXPHYSICS_EXPORT MeshInstance : public agx::Physics::Geometry::ShapeInstance
      {
      public:
        MeshInstance();
        MeshInstance(MeshData* data, agx::Index index);
        MeshInstance(agxData::EntityStorage *storage, agx::Index index);
        MeshInstance(const agxData::EntityInstance& other);
        MeshInstance(const agxData::EntityPtr& ptr);

        MeshData* getData();
        const MeshData* getData() const;

      public:
      };
      DOXYGEN_END_INTERNAL_BLOCK()



      typedef agx::VectorPOD<MeshPtr> MeshPtrVector;
      typedef agxData::Array<MeshPtr> MeshPtrArray;



      DOXYGEN_START_INTERNAL_BLOCK()
      /* Implementation */
      //-----------------------------------------------------------------------------------------------------
      //-----------------------------------------------------------------------------------------------------
      inline MeshInstance agx::Physics::Geometry::MeshData::operator[] (size_t index) { return MeshInstance(this, (agx::Index)index); }
      //-----------------------------------------------------------------------------------------------------
      AGX_FORCE_INLINE MeshPtr::MeshPtr() {}
      AGX_FORCE_INLINE MeshPtr::MeshPtr(agxData::EntityStorage* storage, agx::Index id) : agx::Physics::Geometry::ShapePtr(storage, id) {}
      AGX_FORCE_INLINE MeshPtr::MeshPtr(const agxData::EntityPtr& ptr) : agx::Physics::Geometry::ShapePtr(ptr)
      {
        agxAssertN(!ptr || ptr.isInstanceOf(MeshModel::instance()),
          "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
          EntityPtr::getModel()->fullPath().c_str(), MeshModel::instance()->fullPath().c_str());
      }

      AGX_FORCE_INLINE MeshPtr::MeshPtr(const agxData::EntityInstance& instance) : agx::Physics::Geometry::ShapePtr(instance)
      {
        agxAssertN(!instance || instance.isInstanceOf(MeshModel::instance()),
          "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
          EntityPtr::getModel()->fullPath().c_str(), MeshModel::instance()->fullPath().c_str());
      }

      AGX_FORCE_INLINE MeshPtr& MeshPtr::operator= (const agxData::EntityPtr& ptr)
      {
        agxData::EntityPtr::operator= (ptr);
        agxAssertN(!ptr || ptr.isInstanceOf(MeshModel::instance()),
          "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
          EntityPtr::getModel()->fullPath().c_str(), MeshModel::instance()->fullPath().c_str());
        return *this;
      }

      AGX_FORCE_INLINE MeshPtr& MeshPtr::operator= (const agxData::EntityInstance& instance)
      {
        agxData::EntityPtr::operator= (instance);
        agxAssertN(!instance || instance.isInstanceOf(MeshModel::instance()),
          "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
          EntityPtr::getModel()->fullPath().c_str(), MeshModel::instance()->fullPath().c_str());
        return *this;
      }

      inline MeshInstance MeshPtr::instance() { return agxData::EntityPtr::instance(); }
      inline const MeshInstance MeshPtr::instance() const { return agxData::EntityPtr::instance(); }
      AGX_FORCE_INLINE MeshSemantics* MeshPtr::operator->() { return (MeshSemantics* )this; }
      AGX_FORCE_INLINE const MeshSemantics* MeshPtr::operator->() const { return (const MeshSemantics* )this; }
      AGX_FORCE_INLINE MeshData* MeshPtr::getData() { return static_cast<MeshData* >(agxData::EntityPtr::getData()); }
      AGX_FORCE_INLINE const MeshData* MeshPtr::getData() const { return static_cast<const MeshData* >(agxData::EntityPtr::getData()); }

      //-----------------------------------------------------------------------------------------------------
      AGX_FORCE_INLINE MeshInstance::MeshInstance() {}
      AGX_FORCE_INLINE MeshInstance::MeshInstance(MeshData* data, agx::Index index) : agx::Physics::Geometry::ShapeInstance(data, index) {}
      AGX_FORCE_INLINE MeshInstance::MeshInstance(agxData::EntityStorage* storage, agx::Index index) : agx::Physics::Geometry::ShapeInstance(storage, index) {}
      AGX_FORCE_INLINE MeshInstance::MeshInstance(const agxData::EntityInstance& other) : agx::Physics::Geometry::ShapeInstance(other)
      {
        agxAssertN(!other || other.isInstanceOf(MeshModel::instance()),
          "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
          EntityInstance::getModel()->fullPath().c_str(), MeshModel::instance()->fullPath().c_str());
      }

      AGX_FORCE_INLINE MeshInstance::MeshInstance(const agxData::EntityPtr& ptr) : agx::Physics::Geometry::ShapeInstance(ptr)
      {
        agxAssertN(!ptr || ptr.isInstanceOf(MeshModel::instance()),
          "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
          EntityInstance::getModel()->fullPath().c_str(), MeshModel::instance()->fullPath().c_str());
      }


      AGX_FORCE_INLINE MeshData* MeshInstance::getData() { return static_cast<MeshData* >(agxData::EntityInstance::getData()); }
      AGX_FORCE_INLINE const MeshData* MeshInstance::getData() const { return static_cast<const MeshData* >(agxData::EntityInstance::getData()); }

      //-----------------------------------------------------------------------------------------------------
      AGX_FORCE_INLINE MeshSemantics::MeshSemantics() {}
      //-----------------------------------------------------------------------------------------------------
      DOXYGEN_END_INTERNAL_BLOCK()
    }
  }
}

AGX_TYPE_BINDING(agx::Physics::Geometry::MeshPtr, "Physics.Geometry.MeshPtr")
AGX_TYPE_BINDING(agx::Physics::Geometry::MeshInstance, "Physics.Geometry.MeshInstance")

#ifdef _MSC_VER
# pragma warning(pop)
#endif

#undef AGX_ENTITY_WRAPPER
#undef AGX_ENTITY_NAMESPACE
#endif

