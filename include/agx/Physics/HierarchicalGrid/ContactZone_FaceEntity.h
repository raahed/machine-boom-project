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

#ifndef GENERATED_AGX_PHYSICS_HIERARCHICALGRID_CONTACTZONE_FACE_H_PLUGIN
#define GENERATED_AGX_PHYSICS_HIERARCHICALGRID_CONTACTZONE_FACE_H_PLUGIN

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
#include <agx/Physics/HierarchicalGrid/ContactZoneEntity.h>


namespace agx
{
  namespace Physics
  {
    namespace HierarchicalGrid
    {

      class ContactZone_FaceModel;
      class ContactZone_FaceData;
      class ContactZone_FacePtr;
      class ContactZone_FaceInstance;
      class ContactZone_FaceSemantics;


      AGX_DECLARE_POINTER_TYPES(ContactZone_FaceModel);

      /** 
      Abstract description of the data attributes for the Physics.HierarchicalGrid.ContactZone_Face entity.
      */ 
      class AGXPHYSICS_EXPORT ContactZone_FaceModel : public agx::Physics::HierarchicalGrid::ContactZoneModel
      {
      public:
        typedef ContactZone_FacePtr PtrT;

        ContactZone_FaceModel(const agx::String& name = "ContactZone_Face");

        /// \return The entity model singleton.
        static ContactZone_FaceModel* instance();

        /// Create and return a pointer to a new instance in the default storage for this entity model.
        static ContactZone_FacePtr createInstance();

        /// \return The default storage for this entity model.
        static agxData::EntityStorage* defaultStorage();

        /// This is part of internal cleanup and should not be called by users
        virtual void shutdownCleanup() override;



        /* Attributes */

      protected:
        virtual ~ContactZone_FaceModel();
        virtual agxData::EntityData* createData(agxData::EntityStorage* storage) override;
        virtual void configure(agx::TiXmlElement* eEntity) override;
        virtual void initAttributeAccessors() override;
        virtual void construct(agxData::EntityPtr instance) override;
        void construct(agx::Physics::HierarchicalGrid::ContactZone_FacePtr contactZone_Face);
      };


      DOXYGEN_START_INTERNAL_BLOCK()
      #ifndef AGX_PHYSICS_HIERARCHICALGRID_CONTACTZONE_FACE_DATA_SET_OVERRIDE
      #define AGX_PHYSICS_HIERARCHICALGRID_CONTACTZONE_FACE_DATA_SET
      class AGXPHYSICS_EXPORT ContactZone_FaceData : public agx::Physics::HierarchicalGrid::ContactZoneData
      {
      public:
        ContactZone_FaceInstance operator[] (size_t index);

      public:
        agxData::Array< ContactZone_FacePtr >& instance;

      public:

      public:
        ContactZone_FaceData(agxData::EntityStorage* storage);
        ContactZone_FaceData();

      protected:
        virtual ~ContactZone_FaceData() {}
        virtual void setNumElements(agx::Index numElements) override;

      private:
        ContactZone_FaceData& operator= (const ContactZone_FaceData&) { return *this; }

      };
      #endif
      DOXYGEN_END_INTERNAL_BLOCK()


      DOXYGEN_START_INTERNAL_BLOCK()
      class AGXPHYSICS_EXPORT ContactZone_FaceSemantics : public agx::Physics::HierarchicalGrid::ContactZoneSemantics
      {
      public:

        // Automatic getters

        // Semantics defined by explicit kernels

        // Automatic setters


      protected:
        friend class ContactZone_FacePtr;
        friend class ContactZone_FaceInstance;
        ContactZone_FaceSemantics();
      };
      DOXYGEN_END_INTERNAL_BLOCK()


      /**
      Pointer to a entity instance of type Physics.HierarchicalGrid.ContactZone_Face
      */
      class CALLABLE ContactZone_FacePtr : public agx::Physics::HierarchicalGrid::ContactZonePtr
      {
      public:
        typedef ContactZone_FaceModel ModelType;
        typedef ContactZone_FaceData DataType;
        typedef ContactZone_FaceInstance InstanceType;

      public:
        AGXPHYSICS_EXPORT ContactZone_FacePtr();
        AGXPHYSICS_EXPORT ContactZone_FacePtr(agxData::EntityStorage* storage, agx::Index id);
        AGXPHYSICS_EXPORT ContactZone_FacePtr(const agxData::EntityPtr& ptr);
        AGXPHYSICS_EXPORT ContactZone_FacePtr(const agxData::EntityInstance& instance);
        AGXPHYSICS_EXPORT ContactZone_FacePtr& operator= (const agxData::EntityPtr& ptr);
        AGXPHYSICS_EXPORT ContactZone_FacePtr& operator= (const agxData::EntityInstance& instance);
        AGXPHYSICS_EXPORT ContactZone_FaceInstance instance();
        AGXPHYSICS_EXPORT const ContactZone_FaceInstance instance() const;

        AGXPHYSICS_EXPORT ContactZone_FaceSemantics* operator->();
        AGXPHYSICS_EXPORT const ContactZone_FaceSemantics* operator->() const;

        ContactZone_FaceData* getData();
        const ContactZone_FaceData* getData() const;


        DOXYGEN_START_INTERNAL_BLOCK()
        // Only used by tolua!
        #ifdef AGX_GENERATED_TOLUA_WRAPPER


        #endif
        DOXYGEN_END_INTERNAL_BLOCK()

      };


      DOXYGEN_START_INTERNAL_BLOCK()
      class AGXPHYSICS_EXPORT ContactZone_FaceInstance : public agx::Physics::HierarchicalGrid::ContactZoneInstance
      {
      public:
        ContactZone_FaceInstance();
        ContactZone_FaceInstance(ContactZone_FaceData* data, agx::Index index);
        ContactZone_FaceInstance(agxData::EntityStorage *storage, agx::Index index);
        ContactZone_FaceInstance(const agxData::EntityInstance& other);
        ContactZone_FaceInstance(const agxData::EntityPtr& ptr);

        ContactZone_FaceData* getData();
        const ContactZone_FaceData* getData() const;

      public:
      };
      DOXYGEN_END_INTERNAL_BLOCK()



      typedef agx::VectorPOD<ContactZone_FacePtr> ContactZone_FacePtrVector;
      typedef agxData::Array<ContactZone_FacePtr> ContactZone_FacePtrArray;



      DOXYGEN_START_INTERNAL_BLOCK()
      /* Implementation */
      //-----------------------------------------------------------------------------------------------------
      //-----------------------------------------------------------------------------------------------------
      inline ContactZone_FaceInstance agx::Physics::HierarchicalGrid::ContactZone_FaceData::operator[] (size_t index) { return ContactZone_FaceInstance(this, (agx::Index)index); }
      //-----------------------------------------------------------------------------------------------------
      AGX_FORCE_INLINE ContactZone_FacePtr::ContactZone_FacePtr() {}
      AGX_FORCE_INLINE ContactZone_FacePtr::ContactZone_FacePtr(agxData::EntityStorage* storage, agx::Index id) : agx::Physics::HierarchicalGrid::ContactZonePtr(storage, id) {}
      AGX_FORCE_INLINE ContactZone_FacePtr::ContactZone_FacePtr(const agxData::EntityPtr& ptr) : agx::Physics::HierarchicalGrid::ContactZonePtr(ptr)
      {
        agxAssertN(!ptr || ptr.isInstanceOf(ContactZone_FaceModel::instance()),
          "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
          EntityPtr::getModel()->fullPath().c_str(), ContactZone_FaceModel::instance()->fullPath().c_str());
      }

      AGX_FORCE_INLINE ContactZone_FacePtr::ContactZone_FacePtr(const agxData::EntityInstance& instance) : agx::Physics::HierarchicalGrid::ContactZonePtr(instance)
      {
        agxAssertN(!instance || instance.isInstanceOf(ContactZone_FaceModel::instance()),
          "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
          EntityPtr::getModel()->fullPath().c_str(), ContactZone_FaceModel::instance()->fullPath().c_str());
      }

      AGX_FORCE_INLINE ContactZone_FacePtr& ContactZone_FacePtr::operator= (const agxData::EntityPtr& ptr)
      {
        agxData::EntityPtr::operator= (ptr);
        agxAssertN(!ptr || ptr.isInstanceOf(ContactZone_FaceModel::instance()),
          "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
          EntityPtr::getModel()->fullPath().c_str(), ContactZone_FaceModel::instance()->fullPath().c_str());
        return *this;
      }

      AGX_FORCE_INLINE ContactZone_FacePtr& ContactZone_FacePtr::operator= (const agxData::EntityInstance& instance)
      {
        agxData::EntityPtr::operator= (instance);
        agxAssertN(!instance || instance.isInstanceOf(ContactZone_FaceModel::instance()),
          "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
          EntityPtr::getModel()->fullPath().c_str(), ContactZone_FaceModel::instance()->fullPath().c_str());
        return *this;
      }

      inline ContactZone_FaceInstance ContactZone_FacePtr::instance() { return agxData::EntityPtr::instance(); }
      inline const ContactZone_FaceInstance ContactZone_FacePtr::instance() const { return agxData::EntityPtr::instance(); }
      AGX_FORCE_INLINE ContactZone_FaceSemantics* ContactZone_FacePtr::operator->() { return (ContactZone_FaceSemantics* )this; }
      AGX_FORCE_INLINE const ContactZone_FaceSemantics* ContactZone_FacePtr::operator->() const { return (const ContactZone_FaceSemantics* )this; }
      AGX_FORCE_INLINE ContactZone_FaceData* ContactZone_FacePtr::getData() { return static_cast<ContactZone_FaceData* >(agxData::EntityPtr::getData()); }
      AGX_FORCE_INLINE const ContactZone_FaceData* ContactZone_FacePtr::getData() const { return static_cast<const ContactZone_FaceData* >(agxData::EntityPtr::getData()); }

      //-----------------------------------------------------------------------------------------------------
      AGX_FORCE_INLINE ContactZone_FaceInstance::ContactZone_FaceInstance() {}
      AGX_FORCE_INLINE ContactZone_FaceInstance::ContactZone_FaceInstance(ContactZone_FaceData* data, agx::Index index) : agx::Physics::HierarchicalGrid::ContactZoneInstance(data, index) {}
      AGX_FORCE_INLINE ContactZone_FaceInstance::ContactZone_FaceInstance(agxData::EntityStorage* storage, agx::Index index) : agx::Physics::HierarchicalGrid::ContactZoneInstance(storage, index) {}
      AGX_FORCE_INLINE ContactZone_FaceInstance::ContactZone_FaceInstance(const agxData::EntityInstance& other) : agx::Physics::HierarchicalGrid::ContactZoneInstance(other)
      {
        agxAssertN(!other || other.isInstanceOf(ContactZone_FaceModel::instance()),
          "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
          EntityInstance::getModel()->fullPath().c_str(), ContactZone_FaceModel::instance()->fullPath().c_str());
      }

      AGX_FORCE_INLINE ContactZone_FaceInstance::ContactZone_FaceInstance(const agxData::EntityPtr& ptr) : agx::Physics::HierarchicalGrid::ContactZoneInstance(ptr)
      {
        agxAssertN(!ptr || ptr.isInstanceOf(ContactZone_FaceModel::instance()),
          "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
          EntityInstance::getModel()->fullPath().c_str(), ContactZone_FaceModel::instance()->fullPath().c_str());
      }


      AGX_FORCE_INLINE ContactZone_FaceData* ContactZone_FaceInstance::getData() { return static_cast<ContactZone_FaceData* >(agxData::EntityInstance::getData()); }
      AGX_FORCE_INLINE const ContactZone_FaceData* ContactZone_FaceInstance::getData() const { return static_cast<const ContactZone_FaceData* >(agxData::EntityInstance::getData()); }

      //-----------------------------------------------------------------------------------------------------
      AGX_FORCE_INLINE ContactZone_FaceSemantics::ContactZone_FaceSemantics() {}
      //-----------------------------------------------------------------------------------------------------
      DOXYGEN_END_INTERNAL_BLOCK()
    }
  }
}

AGX_TYPE_BINDING(agx::Physics::HierarchicalGrid::ContactZone_FacePtr, "Physics.HierarchicalGrid.ContactZone_FacePtr")
AGX_TYPE_BINDING(agx::Physics::HierarchicalGrid::ContactZone_FaceInstance, "Physics.HierarchicalGrid.ContactZone_FaceInstance")

#ifdef _MSC_VER
# pragma warning(pop)
#endif

#undef AGX_ENTITY_WRAPPER
#undef AGX_ENTITY_NAMESPACE
#endif

