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

#ifndef GENERATED_AGX_PHYSICS_HIERARCHICALGRID_CONTACTZONE_INTERNAL_H_PLUGIN
#define GENERATED_AGX_PHYSICS_HIERARCHICALGRID_CONTACTZONE_INTERNAL_H_PLUGIN

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

      class ContactZone_InternalModel;
      class ContactZone_InternalData;
      class ContactZone_InternalPtr;
      class ContactZone_InternalInstance;
      class ContactZone_InternalSemantics;


      AGX_DECLARE_POINTER_TYPES(ContactZone_InternalModel);

      /** 
      Abstract description of the data attributes for the Physics.HierarchicalGrid.ContactZone_Internal entity.
      */ 
      class AGXPHYSICS_EXPORT ContactZone_InternalModel : public agx::Physics::HierarchicalGrid::ContactZoneModel
      {
      public:
        typedef ContactZone_InternalPtr PtrT;

        ContactZone_InternalModel(const agx::String& name = "ContactZone_Internal");

        /// \return The entity model singleton.
        static ContactZone_InternalModel* instance();

        /// Create and return a pointer to a new instance in the default storage for this entity model.
        static ContactZone_InternalPtr createInstance();

        /// \return The default storage for this entity model.
        static agxData::EntityStorage* defaultStorage();

        /// This is part of internal cleanup and should not be called by users
        virtual void shutdownCleanup() override;



        /* Attributes */

      protected:
        virtual ~ContactZone_InternalModel();
        virtual agxData::EntityData* createData(agxData::EntityStorage* storage) override;
        virtual void configure(agx::TiXmlElement* eEntity) override;
        virtual void initAttributeAccessors() override;
        virtual void construct(agxData::EntityPtr instance) override;
        void construct(agx::Physics::HierarchicalGrid::ContactZone_InternalPtr contactZone_Internal);
      };


      DOXYGEN_START_INTERNAL_BLOCK()
      #ifndef AGX_PHYSICS_HIERARCHICALGRID_CONTACTZONE_INTERNAL_DATA_SET_OVERRIDE
      #define AGX_PHYSICS_HIERARCHICALGRID_CONTACTZONE_INTERNAL_DATA_SET
      class AGXPHYSICS_EXPORT ContactZone_InternalData : public agx::Physics::HierarchicalGrid::ContactZoneData
      {
      public:
        ContactZone_InternalInstance operator[] (size_t index);

      public:
        agxData::Array< ContactZone_InternalPtr >& instance;

      public:

      public:
        ContactZone_InternalData(agxData::EntityStorage* storage);
        ContactZone_InternalData();

      protected:
        virtual ~ContactZone_InternalData() {}
        virtual void setNumElements(agx::Index numElements) override;

      private:
        ContactZone_InternalData& operator= (const ContactZone_InternalData&) { return *this; }

      };
      #endif
      DOXYGEN_END_INTERNAL_BLOCK()


      DOXYGEN_START_INTERNAL_BLOCK()
      class AGXPHYSICS_EXPORT ContactZone_InternalSemantics : public agx::Physics::HierarchicalGrid::ContactZoneSemantics
      {
      public:

        // Automatic getters

        // Semantics defined by explicit kernels

        // Automatic setters


      protected:
        friend class ContactZone_InternalPtr;
        friend class ContactZone_InternalInstance;
        ContactZone_InternalSemantics();
      };
      DOXYGEN_END_INTERNAL_BLOCK()


      /**
      Pointer to a entity instance of type Physics.HierarchicalGrid.ContactZone_Internal
      */
      class CALLABLE ContactZone_InternalPtr : public agx::Physics::HierarchicalGrid::ContactZonePtr
      {
      public:
        typedef ContactZone_InternalModel ModelType;
        typedef ContactZone_InternalData DataType;
        typedef ContactZone_InternalInstance InstanceType;

      public:
        AGXPHYSICS_EXPORT ContactZone_InternalPtr();
        AGXPHYSICS_EXPORT ContactZone_InternalPtr(agxData::EntityStorage* storage, agx::Index id);
        AGXPHYSICS_EXPORT ContactZone_InternalPtr(const agxData::EntityPtr& ptr);
        AGXPHYSICS_EXPORT ContactZone_InternalPtr(const agxData::EntityInstance& instance);
        AGXPHYSICS_EXPORT ContactZone_InternalPtr& operator= (const agxData::EntityPtr& ptr);
        AGXPHYSICS_EXPORT ContactZone_InternalPtr& operator= (const agxData::EntityInstance& instance);
        AGXPHYSICS_EXPORT ContactZone_InternalInstance instance();
        AGXPHYSICS_EXPORT const ContactZone_InternalInstance instance() const;

        AGXPHYSICS_EXPORT ContactZone_InternalSemantics* operator->();
        AGXPHYSICS_EXPORT const ContactZone_InternalSemantics* operator->() const;

        ContactZone_InternalData* getData();
        const ContactZone_InternalData* getData() const;


        DOXYGEN_START_INTERNAL_BLOCK()
        // Only used by tolua!
        #ifdef AGX_GENERATED_TOLUA_WRAPPER


        #endif
        DOXYGEN_END_INTERNAL_BLOCK()

      };


      DOXYGEN_START_INTERNAL_BLOCK()
      class AGXPHYSICS_EXPORT ContactZone_InternalInstance : public agx::Physics::HierarchicalGrid::ContactZoneInstance
      {
      public:
        ContactZone_InternalInstance();
        ContactZone_InternalInstance(ContactZone_InternalData* data, agx::Index index);
        ContactZone_InternalInstance(agxData::EntityStorage *storage, agx::Index index);
        ContactZone_InternalInstance(const agxData::EntityInstance& other);
        ContactZone_InternalInstance(const agxData::EntityPtr& ptr);

        ContactZone_InternalData* getData();
        const ContactZone_InternalData* getData() const;

      public:
      };
      DOXYGEN_END_INTERNAL_BLOCK()



      typedef agx::VectorPOD<ContactZone_InternalPtr> ContactZone_InternalPtrVector;
      typedef agxData::Array<ContactZone_InternalPtr> ContactZone_InternalPtrArray;



      DOXYGEN_START_INTERNAL_BLOCK()
      /* Implementation */
      //-----------------------------------------------------------------------------------------------------
      //-----------------------------------------------------------------------------------------------------
      inline ContactZone_InternalInstance agx::Physics::HierarchicalGrid::ContactZone_InternalData::operator[] (size_t index) { return ContactZone_InternalInstance(this, (agx::Index)index); }
      //-----------------------------------------------------------------------------------------------------
      AGX_FORCE_INLINE ContactZone_InternalPtr::ContactZone_InternalPtr() {}
      AGX_FORCE_INLINE ContactZone_InternalPtr::ContactZone_InternalPtr(agxData::EntityStorage* storage, agx::Index id) : agx::Physics::HierarchicalGrid::ContactZonePtr(storage, id) {}
      AGX_FORCE_INLINE ContactZone_InternalPtr::ContactZone_InternalPtr(const agxData::EntityPtr& ptr) : agx::Physics::HierarchicalGrid::ContactZonePtr(ptr)
      {
        agxAssertN(!ptr || ptr.isInstanceOf(ContactZone_InternalModel::instance()),
          "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
          EntityPtr::getModel()->fullPath().c_str(), ContactZone_InternalModel::instance()->fullPath().c_str());
      }

      AGX_FORCE_INLINE ContactZone_InternalPtr::ContactZone_InternalPtr(const agxData::EntityInstance& instance) : agx::Physics::HierarchicalGrid::ContactZonePtr(instance)
      {
        agxAssertN(!instance || instance.isInstanceOf(ContactZone_InternalModel::instance()),
          "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
          EntityPtr::getModel()->fullPath().c_str(), ContactZone_InternalModel::instance()->fullPath().c_str());
      }

      AGX_FORCE_INLINE ContactZone_InternalPtr& ContactZone_InternalPtr::operator= (const agxData::EntityPtr& ptr)
      {
        agxData::EntityPtr::operator= (ptr);
        agxAssertN(!ptr || ptr.isInstanceOf(ContactZone_InternalModel::instance()),
          "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
          EntityPtr::getModel()->fullPath().c_str(), ContactZone_InternalModel::instance()->fullPath().c_str());
        return *this;
      }

      AGX_FORCE_INLINE ContactZone_InternalPtr& ContactZone_InternalPtr::operator= (const agxData::EntityInstance& instance)
      {
        agxData::EntityPtr::operator= (instance);
        agxAssertN(!instance || instance.isInstanceOf(ContactZone_InternalModel::instance()),
          "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
          EntityPtr::getModel()->fullPath().c_str(), ContactZone_InternalModel::instance()->fullPath().c_str());
        return *this;
      }

      inline ContactZone_InternalInstance ContactZone_InternalPtr::instance() { return agxData::EntityPtr::instance(); }
      inline const ContactZone_InternalInstance ContactZone_InternalPtr::instance() const { return agxData::EntityPtr::instance(); }
      AGX_FORCE_INLINE ContactZone_InternalSemantics* ContactZone_InternalPtr::operator->() { return (ContactZone_InternalSemantics* )this; }
      AGX_FORCE_INLINE const ContactZone_InternalSemantics* ContactZone_InternalPtr::operator->() const { return (const ContactZone_InternalSemantics* )this; }
      AGX_FORCE_INLINE ContactZone_InternalData* ContactZone_InternalPtr::getData() { return static_cast<ContactZone_InternalData* >(agxData::EntityPtr::getData()); }
      AGX_FORCE_INLINE const ContactZone_InternalData* ContactZone_InternalPtr::getData() const { return static_cast<const ContactZone_InternalData* >(agxData::EntityPtr::getData()); }

      //-----------------------------------------------------------------------------------------------------
      AGX_FORCE_INLINE ContactZone_InternalInstance::ContactZone_InternalInstance() {}
      AGX_FORCE_INLINE ContactZone_InternalInstance::ContactZone_InternalInstance(ContactZone_InternalData* data, agx::Index index) : agx::Physics::HierarchicalGrid::ContactZoneInstance(data, index) {}
      AGX_FORCE_INLINE ContactZone_InternalInstance::ContactZone_InternalInstance(agxData::EntityStorage* storage, agx::Index index) : agx::Physics::HierarchicalGrid::ContactZoneInstance(storage, index) {}
      AGX_FORCE_INLINE ContactZone_InternalInstance::ContactZone_InternalInstance(const agxData::EntityInstance& other) : agx::Physics::HierarchicalGrid::ContactZoneInstance(other)
      {
        agxAssertN(!other || other.isInstanceOf(ContactZone_InternalModel::instance()),
          "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
          EntityInstance::getModel()->fullPath().c_str(), ContactZone_InternalModel::instance()->fullPath().c_str());
      }

      AGX_FORCE_INLINE ContactZone_InternalInstance::ContactZone_InternalInstance(const agxData::EntityPtr& ptr) : agx::Physics::HierarchicalGrid::ContactZoneInstance(ptr)
      {
        agxAssertN(!ptr || ptr.isInstanceOf(ContactZone_InternalModel::instance()),
          "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
          EntityInstance::getModel()->fullPath().c_str(), ContactZone_InternalModel::instance()->fullPath().c_str());
      }


      AGX_FORCE_INLINE ContactZone_InternalData* ContactZone_InternalInstance::getData() { return static_cast<ContactZone_InternalData* >(agxData::EntityInstance::getData()); }
      AGX_FORCE_INLINE const ContactZone_InternalData* ContactZone_InternalInstance::getData() const { return static_cast<const ContactZone_InternalData* >(agxData::EntityInstance::getData()); }

      //-----------------------------------------------------------------------------------------------------
      AGX_FORCE_INLINE ContactZone_InternalSemantics::ContactZone_InternalSemantics() {}
      //-----------------------------------------------------------------------------------------------------
      DOXYGEN_END_INTERNAL_BLOCK()
    }
  }
}

AGX_TYPE_BINDING(agx::Physics::HierarchicalGrid::ContactZone_InternalPtr, "Physics.HierarchicalGrid.ContactZone_InternalPtr")
AGX_TYPE_BINDING(agx::Physics::HierarchicalGrid::ContactZone_InternalInstance, "Physics.HierarchicalGrid.ContactZone_InternalInstance")

#ifdef _MSC_VER
# pragma warning(pop)
#endif

#undef AGX_ENTITY_WRAPPER
#undef AGX_ENTITY_NAMESPACE
#endif

