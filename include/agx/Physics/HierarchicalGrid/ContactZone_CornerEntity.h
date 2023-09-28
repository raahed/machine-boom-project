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

#ifndef GENERATED_AGX_PHYSICS_HIERARCHICALGRID_CONTACTZONE_CORNER_H_PLUGIN
#define GENERATED_AGX_PHYSICS_HIERARCHICALGRID_CONTACTZONE_CORNER_H_PLUGIN

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
#include <agx/Physics/HierarchicalGrid/ContactZoneEntity.h>

namespace agx { namespace Physics { namespace HierarchicalGrid { class ContactZonePtr; }}}
namespace agx { namespace Physics { namespace HierarchicalGrid { class ContactZonePtr; }}}

namespace agx
{
  namespace Physics
  {
    namespace HierarchicalGrid
    {

      class ContactZone_CornerModel;
      class ContactZone_CornerData;
      class ContactZone_CornerPtr;
      class ContactZone_CornerInstance;
      class ContactZone_CornerSemantics;


      AGX_DECLARE_POINTER_TYPES(ContactZone_CornerModel);

      /** 
      Abstract description of the data attributes for the Physics.HierarchicalGrid.ContactZone_Corner entity.
      */ 
      class AGXPHYSICS_EXPORT ContactZone_CornerModel : public agx::Physics::HierarchicalGrid::ContactZoneModel
      {
      public:
        typedef ContactZone_CornerPtr PtrT;

        ContactZone_CornerModel(const agx::String& name = "ContactZone_Corner");

        /// \return The entity model singleton.
        static ContactZone_CornerModel* instance();

        /// Create and return a pointer to a new instance in the default storage for this entity model.
        static ContactZone_CornerPtr createInstance();

        /// \return The default storage for this entity model.
        static agxData::EntityStorage* defaultStorage();

        /// This is part of internal cleanup and should not be called by users
        virtual void shutdownCleanup() override;



        /* Attributes */
        static agxData::ScalarAttributeT< agx::Physics::HierarchicalGrid::ContactZonePtr >* edgeConnectionsAttribute;
        static const size_t edgeConnectionsArraySize = 4;
        static agxData::ScalarAttributeT< agx::Physics::HierarchicalGrid::ContactZonePtr >* cornerConnectionsAttribute;
        static const size_t cornerConnectionsArraySize = 4;

      protected:
        virtual ~ContactZone_CornerModel();
        virtual agxData::EntityData* createData(agxData::EntityStorage* storage) override;
        virtual void configure(agx::TiXmlElement* eEntity) override;
        virtual void initAttributeAccessors() override;
        virtual void construct(agxData::EntityPtr instance) override;
        void construct(agx::Physics::HierarchicalGrid::ContactZone_CornerPtr contactZone_Corner);
      };


      DOXYGEN_START_INTERNAL_BLOCK()
      #ifndef AGX_PHYSICS_HIERARCHICALGRID_CONTACTZONE_CORNER_DATA_SET_OVERRIDE
      #define AGX_PHYSICS_HIERARCHICALGRID_CONTACTZONE_CORNER_DATA_SET
      class AGXPHYSICS_EXPORT ContactZone_CornerData : public agx::Physics::HierarchicalGrid::ContactZoneData
      {
      public:
        ContactZone_CornerInstance operator[] (size_t index);

      public:
        agxData::Array< ContactZone_CornerPtr >& instance;
        agxData::Array< agx::Physics::HierarchicalGrid::ContactZonePtr > edgeConnections;
        static const size_t edgeConnectionsArraySize = 4;
        agxData::Array< agx::Physics::HierarchicalGrid::ContactZonePtr > cornerConnections;
        static const size_t cornerConnectionsArraySize = 4;

      public:
        typedef agx::Physics::HierarchicalGrid::ContactZonePtr edgeConnectionsType;
        typedef agx::Physics::HierarchicalGrid::ContactZonePtr cornerConnectionsType;

      public:
        ContactZone_CornerData(agxData::EntityStorage* storage);
        ContactZone_CornerData();

      protected:
        virtual ~ContactZone_CornerData() {}
        virtual void setNumElements(agx::Index numElements) override;

      private:
        ContactZone_CornerData& operator= (const ContactZone_CornerData&) { return *this; }

      };
      #endif
      DOXYGEN_END_INTERNAL_BLOCK()


      DOXYGEN_START_INTERNAL_BLOCK()
      class AGXPHYSICS_EXPORT ContactZone_CornerSemantics : public agx::Physics::HierarchicalGrid::ContactZoneSemantics
      {
      public:

        // Automatic getters

        // Semantics defined by explicit kernels

        // Automatic setters


      protected:
        friend class ContactZone_CornerPtr;
        friend class ContactZone_CornerInstance;
        ContactZone_CornerSemantics();
      };
      DOXYGEN_END_INTERNAL_BLOCK()


      /**
      Pointer to a entity instance of type Physics.HierarchicalGrid.ContactZone_Corner
      */
      class CALLABLE ContactZone_CornerPtr : public agx::Physics::HierarchicalGrid::ContactZonePtr
      {
      public:
        typedef ContactZone_CornerModel ModelType;
        typedef ContactZone_CornerData DataType;
        typedef ContactZone_CornerInstance InstanceType;

      public:
        AGXPHYSICS_EXPORT ContactZone_CornerPtr();
        AGXPHYSICS_EXPORT ContactZone_CornerPtr(agxData::EntityStorage* storage, agx::Index id);
        AGXPHYSICS_EXPORT ContactZone_CornerPtr(const agxData::EntityPtr& ptr);
        AGXPHYSICS_EXPORT ContactZone_CornerPtr(const agxData::EntityInstance& instance);
        AGXPHYSICS_EXPORT ContactZone_CornerPtr& operator= (const agxData::EntityPtr& ptr);
        AGXPHYSICS_EXPORT ContactZone_CornerPtr& operator= (const agxData::EntityInstance& instance);
        AGXPHYSICS_EXPORT ContactZone_CornerInstance instance();
        AGXPHYSICS_EXPORT const ContactZone_CornerInstance instance() const;

        AGXPHYSICS_EXPORT ContactZone_CornerSemantics* operator->();
        AGXPHYSICS_EXPORT const ContactZone_CornerSemantics* operator->() const;

        ContactZone_CornerData* getData();
        const ContactZone_CornerData* getData() const;


        AGXPHYSICS_EXPORT agxData::Array< agx::Physics::HierarchicalGrid::ContactZonePtr > edgeConnections();
        AGXPHYSICS_EXPORT agxData::Array< agx::Physics::HierarchicalGrid::ContactZonePtr > const edgeConnections() const;
        AGXPHYSICS_EXPORT agx::Physics::HierarchicalGrid::ContactZonePtr& edgeConnections(size_t index);
        AGXPHYSICS_EXPORT agx::Physics::HierarchicalGrid::ContactZonePtr const& edgeConnections(size_t index) const;

        AGXPHYSICS_EXPORT agxData::Array< agx::Physics::HierarchicalGrid::ContactZonePtr > cornerConnections();
        AGXPHYSICS_EXPORT agxData::Array< agx::Physics::HierarchicalGrid::ContactZonePtr > const cornerConnections() const;
        AGXPHYSICS_EXPORT agx::Physics::HierarchicalGrid::ContactZonePtr& cornerConnections(size_t index);
        AGXPHYSICS_EXPORT agx::Physics::HierarchicalGrid::ContactZonePtr const& cornerConnections(size_t index) const;

        DOXYGEN_START_INTERNAL_BLOCK()
        // Only used by tolua!
        #ifdef AGX_GENERATED_TOLUA_WRAPPER


        #endif
        DOXYGEN_END_INTERNAL_BLOCK()

      };


      DOXYGEN_START_INTERNAL_BLOCK()
      class AGXPHYSICS_EXPORT ContactZone_CornerInstance : public agx::Physics::HierarchicalGrid::ContactZoneInstance
      {
      public:
        ContactZone_CornerInstance();
        ContactZone_CornerInstance(ContactZone_CornerData* data, agx::Index index);
        ContactZone_CornerInstance(agxData::EntityStorage *storage, agx::Index index);
        ContactZone_CornerInstance(const agxData::EntityInstance& other);
        ContactZone_CornerInstance(const agxData::EntityPtr& ptr);

        ContactZone_CornerData* getData();
        const ContactZone_CornerData* getData() const;

      public:
        agxData::Array< agx::Physics::HierarchicalGrid::ContactZonePtr > edgeConnections();
        agxData::Array< agx::Physics::HierarchicalGrid::ContactZonePtr > const edgeConnections() const;
        agx::Physics::HierarchicalGrid::ContactZonePtr& edgeConnections(size_t index);
        agx::Physics::HierarchicalGrid::ContactZonePtr const& edgeConnections(size_t index) const;

        agxData::Array< agx::Physics::HierarchicalGrid::ContactZonePtr > cornerConnections();
        agxData::Array< agx::Physics::HierarchicalGrid::ContactZonePtr > const cornerConnections() const;
        agx::Physics::HierarchicalGrid::ContactZonePtr& cornerConnections(size_t index);
        agx::Physics::HierarchicalGrid::ContactZonePtr const& cornerConnections(size_t index) const;

      };
      DOXYGEN_END_INTERNAL_BLOCK()



      typedef agx::VectorPOD<ContactZone_CornerPtr> ContactZone_CornerPtrVector;
      typedef agxData::Array<ContactZone_CornerPtr> ContactZone_CornerPtrArray;



      DOXYGEN_START_INTERNAL_BLOCK()
      /* Implementation */
      //-----------------------------------------------------------------------------------------------------
      //-----------------------------------------------------------------------------------------------------
      inline ContactZone_CornerInstance agx::Physics::HierarchicalGrid::ContactZone_CornerData::operator[] (size_t index) { return ContactZone_CornerInstance(this, (agx::Index)index); }
      //-----------------------------------------------------------------------------------------------------
      AGX_FORCE_INLINE ContactZone_CornerPtr::ContactZone_CornerPtr() {}
      AGX_FORCE_INLINE ContactZone_CornerPtr::ContactZone_CornerPtr(agxData::EntityStorage* storage, agx::Index id) : agx::Physics::HierarchicalGrid::ContactZonePtr(storage, id) {}
      AGX_FORCE_INLINE ContactZone_CornerPtr::ContactZone_CornerPtr(const agxData::EntityPtr& ptr) : agx::Physics::HierarchicalGrid::ContactZonePtr(ptr)
      {
        agxAssertN(!ptr || ptr.isInstanceOf(ContactZone_CornerModel::instance()),
          "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
          EntityPtr::getModel()->fullPath().c_str(), ContactZone_CornerModel::instance()->fullPath().c_str());
      }

      AGX_FORCE_INLINE ContactZone_CornerPtr::ContactZone_CornerPtr(const agxData::EntityInstance& instance) : agx::Physics::HierarchicalGrid::ContactZonePtr(instance)
      {
        agxAssertN(!instance || instance.isInstanceOf(ContactZone_CornerModel::instance()),
          "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
          EntityPtr::getModel()->fullPath().c_str(), ContactZone_CornerModel::instance()->fullPath().c_str());
      }

      AGX_FORCE_INLINE ContactZone_CornerPtr& ContactZone_CornerPtr::operator= (const agxData::EntityPtr& ptr)
      {
        agxData::EntityPtr::operator= (ptr);
        agxAssertN(!ptr || ptr.isInstanceOf(ContactZone_CornerModel::instance()),
          "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
          EntityPtr::getModel()->fullPath().c_str(), ContactZone_CornerModel::instance()->fullPath().c_str());
        return *this;
      }

      AGX_FORCE_INLINE ContactZone_CornerPtr& ContactZone_CornerPtr::operator= (const agxData::EntityInstance& instance)
      {
        agxData::EntityPtr::operator= (instance);
        agxAssertN(!instance || instance.isInstanceOf(ContactZone_CornerModel::instance()),
          "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
          EntityPtr::getModel()->fullPath().c_str(), ContactZone_CornerModel::instance()->fullPath().c_str());
        return *this;
      }

      inline ContactZone_CornerInstance ContactZone_CornerPtr::instance() { return agxData::EntityPtr::instance(); }
      inline const ContactZone_CornerInstance ContactZone_CornerPtr::instance() const { return agxData::EntityPtr::instance(); }
      AGX_FORCE_INLINE ContactZone_CornerSemantics* ContactZone_CornerPtr::operator->() { return (ContactZone_CornerSemantics* )this; }
      AGX_FORCE_INLINE const ContactZone_CornerSemantics* ContactZone_CornerPtr::operator->() const { return (const ContactZone_CornerSemantics* )this; }
      AGX_FORCE_INLINE ContactZone_CornerData* ContactZone_CornerPtr::getData() { return static_cast<ContactZone_CornerData* >(agxData::EntityPtr::getData()); }
      AGX_FORCE_INLINE const ContactZone_CornerData* ContactZone_CornerPtr::getData() const { return static_cast<const ContactZone_CornerData* >(agxData::EntityPtr::getData()); }

      AGX_FORCE_INLINE agxData::Array< agx::Physics::HierarchicalGrid::ContactZonePtr > ContactZone_CornerPtr::edgeConnections() { verifyIndex(); return getData()->edgeConnections.slice(agx::IndexRange32(calculateIndex() * (agx::Index)ContactZone_CornerData::edgeConnectionsArraySize, (calculateIndex()+1) * (agx::Index)ContactZone_CornerData::edgeConnectionsArraySize)); }
      AGX_FORCE_INLINE const agxData::Array< agx::Physics::HierarchicalGrid::ContactZonePtr > ContactZone_CornerPtr::edgeConnections() const { verifyIndex(); return getData()->edgeConnections.slice(agx::IndexRange32(calculateIndex() * (agx::Index)ContactZone_CornerData::edgeConnectionsArraySize, (calculateIndex()+1) * (agx::Index)ContactZone_CornerData::edgeConnectionsArraySize)); }
      AGX_FORCE_INLINE agx::Physics::HierarchicalGrid::ContactZonePtr& ContactZone_CornerPtr::edgeConnections(size_t index) { return this->edgeConnections()[index]; }
      AGX_FORCE_INLINE agx::Physics::HierarchicalGrid::ContactZonePtr const& ContactZone_CornerPtr::edgeConnections(size_t index) const { return this->edgeConnections()[index]; }

      AGX_FORCE_INLINE agxData::Array< agx::Physics::HierarchicalGrid::ContactZonePtr > ContactZone_CornerPtr::cornerConnections() { verifyIndex(); return getData()->cornerConnections.slice(agx::IndexRange32(calculateIndex() * (agx::Index)ContactZone_CornerData::cornerConnectionsArraySize, (calculateIndex()+1) * (agx::Index)ContactZone_CornerData::cornerConnectionsArraySize)); }
      AGX_FORCE_INLINE const agxData::Array< agx::Physics::HierarchicalGrid::ContactZonePtr > ContactZone_CornerPtr::cornerConnections() const { verifyIndex(); return getData()->cornerConnections.slice(agx::IndexRange32(calculateIndex() * (agx::Index)ContactZone_CornerData::cornerConnectionsArraySize, (calculateIndex()+1) * (agx::Index)ContactZone_CornerData::cornerConnectionsArraySize)); }
      AGX_FORCE_INLINE agx::Physics::HierarchicalGrid::ContactZonePtr& ContactZone_CornerPtr::cornerConnections(size_t index) { return this->cornerConnections()[index]; }
      AGX_FORCE_INLINE agx::Physics::HierarchicalGrid::ContactZonePtr const& ContactZone_CornerPtr::cornerConnections(size_t index) const { return this->cornerConnections()[index]; }

      //-----------------------------------------------------------------------------------------------------
      AGX_FORCE_INLINE ContactZone_CornerInstance::ContactZone_CornerInstance() {}
      AGX_FORCE_INLINE ContactZone_CornerInstance::ContactZone_CornerInstance(ContactZone_CornerData* data, agx::Index index) : agx::Physics::HierarchicalGrid::ContactZoneInstance(data, index) {}
      AGX_FORCE_INLINE ContactZone_CornerInstance::ContactZone_CornerInstance(agxData::EntityStorage* storage, agx::Index index) : agx::Physics::HierarchicalGrid::ContactZoneInstance(storage, index) {}
      AGX_FORCE_INLINE ContactZone_CornerInstance::ContactZone_CornerInstance(const agxData::EntityInstance& other) : agx::Physics::HierarchicalGrid::ContactZoneInstance(other)
      {
        agxAssertN(!other || other.isInstanceOf(ContactZone_CornerModel::instance()),
          "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
          EntityInstance::getModel()->fullPath().c_str(), ContactZone_CornerModel::instance()->fullPath().c_str());
      }

      AGX_FORCE_INLINE ContactZone_CornerInstance::ContactZone_CornerInstance(const agxData::EntityPtr& ptr) : agx::Physics::HierarchicalGrid::ContactZoneInstance(ptr)
      {
        agxAssertN(!ptr || ptr.isInstanceOf(ContactZone_CornerModel::instance()),
          "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
          EntityInstance::getModel()->fullPath().c_str(), ContactZone_CornerModel::instance()->fullPath().c_str());
      }


      AGX_FORCE_INLINE ContactZone_CornerData* ContactZone_CornerInstance::getData() { return static_cast<ContactZone_CornerData* >(agxData::EntityInstance::getData()); }
      AGX_FORCE_INLINE const ContactZone_CornerData* ContactZone_CornerInstance::getData() const { return static_cast<const ContactZone_CornerData* >(agxData::EntityInstance::getData()); }

      AGX_FORCE_INLINE agxData::Array< agx::Physics::HierarchicalGrid::ContactZonePtr > ContactZone_CornerInstance::edgeConnections() { verifyIndex(); return getData()->edgeConnections.slice(agx::IndexRange32(getIndex() * (agx::Index)ContactZone_CornerData::edgeConnectionsArraySize, (getIndex()+1) * (agx::Index)ContactZone_CornerData::edgeConnectionsArraySize)); }
      AGX_FORCE_INLINE const agxData::Array< agx::Physics::HierarchicalGrid::ContactZonePtr > ContactZone_CornerInstance::edgeConnections() const { verifyIndex(); return getData()->edgeConnections.slice(agx::IndexRange32(getIndex() * (agx::Index)ContactZone_CornerData::edgeConnectionsArraySize, (getIndex()+1) * (agx::Index)ContactZone_CornerData::edgeConnectionsArraySize)); }
      AGX_FORCE_INLINE agx::Physics::HierarchicalGrid::ContactZonePtr& ContactZone_CornerInstance::edgeConnections(size_t index) { return this->edgeConnections()[index]; }
      AGX_FORCE_INLINE agx::Physics::HierarchicalGrid::ContactZonePtr const& ContactZone_CornerInstance::edgeConnections(size_t index) const { return this->edgeConnections()[index]; }

      AGX_FORCE_INLINE agxData::Array< agx::Physics::HierarchicalGrid::ContactZonePtr > ContactZone_CornerInstance::cornerConnections() { verifyIndex(); return getData()->cornerConnections.slice(agx::IndexRange32(getIndex() * (agx::Index)ContactZone_CornerData::cornerConnectionsArraySize, (getIndex()+1) * (agx::Index)ContactZone_CornerData::cornerConnectionsArraySize)); }
      AGX_FORCE_INLINE const agxData::Array< agx::Physics::HierarchicalGrid::ContactZonePtr > ContactZone_CornerInstance::cornerConnections() const { verifyIndex(); return getData()->cornerConnections.slice(agx::IndexRange32(getIndex() * (agx::Index)ContactZone_CornerData::cornerConnectionsArraySize, (getIndex()+1) * (agx::Index)ContactZone_CornerData::cornerConnectionsArraySize)); }
      AGX_FORCE_INLINE agx::Physics::HierarchicalGrid::ContactZonePtr& ContactZone_CornerInstance::cornerConnections(size_t index) { return this->cornerConnections()[index]; }
      AGX_FORCE_INLINE agx::Physics::HierarchicalGrid::ContactZonePtr const& ContactZone_CornerInstance::cornerConnections(size_t index) const { return this->cornerConnections()[index]; }

      //-----------------------------------------------------------------------------------------------------
      AGX_FORCE_INLINE ContactZone_CornerSemantics::ContactZone_CornerSemantics() {}
      //-----------------------------------------------------------------------------------------------------
      DOXYGEN_END_INTERNAL_BLOCK()
    }
  }
}

AGX_TYPE_BINDING(agx::Physics::HierarchicalGrid::ContactZone_CornerPtr, "Physics.HierarchicalGrid.ContactZone_CornerPtr")
AGX_TYPE_BINDING(agx::Physics::HierarchicalGrid::ContactZone_CornerInstance, "Physics.HierarchicalGrid.ContactZone_CornerInstance")

#ifdef _MSC_VER
# pragma warning(pop)
#endif

#undef AGX_ENTITY_WRAPPER
#undef AGX_ENTITY_NAMESPACE
#endif

