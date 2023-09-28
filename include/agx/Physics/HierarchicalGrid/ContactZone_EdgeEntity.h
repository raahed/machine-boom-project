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

#ifndef GENERATED_AGX_PHYSICS_HIERARCHICALGRID_CONTACTZONE_EDGE_H_PLUGIN
#define GENERATED_AGX_PHYSICS_HIERARCHICALGRID_CONTACTZONE_EDGE_H_PLUGIN

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

      class ContactZone_EdgeModel;
      class ContactZone_EdgeData;
      class ContactZone_EdgePtr;
      class ContactZone_EdgeInstance;
      class ContactZone_EdgeSemantics;


      AGX_DECLARE_POINTER_TYPES(ContactZone_EdgeModel);

      /** 
      Abstract description of the data attributes for the Physics.HierarchicalGrid.ContactZone_Edge entity.
      */ 
      class AGXPHYSICS_EXPORT ContactZone_EdgeModel : public agx::Physics::HierarchicalGrid::ContactZoneModel
      {
      public:
        typedef ContactZone_EdgePtr PtrT;

        ContactZone_EdgeModel(const agx::String& name = "ContactZone_Edge");

        /// \return The entity model singleton.
        static ContactZone_EdgeModel* instance();

        /// Create and return a pointer to a new instance in the default storage for this entity model.
        static ContactZone_EdgePtr createInstance();

        /// \return The default storage for this entity model.
        static agxData::EntityStorage* defaultStorage();

        /// This is part of internal cleanup and should not be called by users
        virtual void shutdownCleanup() override;



        /* Attributes */
        static agxData::ScalarAttributeT< agx::Physics::HierarchicalGrid::ContactZonePtr >* internalConnectionsAttribute;
        static const size_t internalConnectionsArraySize = 2;
        static agxData::ScalarAttributeT< agx::Physics::HierarchicalGrid::ContactZonePtr >* cornerConnectionsAttribute;
        static const size_t cornerConnectionsArraySize = 2;

      protected:
        virtual ~ContactZone_EdgeModel();
        virtual agxData::EntityData* createData(agxData::EntityStorage* storage) override;
        virtual void configure(agx::TiXmlElement* eEntity) override;
        virtual void initAttributeAccessors() override;
        virtual void construct(agxData::EntityPtr instance) override;
        void construct(agx::Physics::HierarchicalGrid::ContactZone_EdgePtr contactZone_Edge);
      };


      DOXYGEN_START_INTERNAL_BLOCK()
      #ifndef AGX_PHYSICS_HIERARCHICALGRID_CONTACTZONE_EDGE_DATA_SET_OVERRIDE
      #define AGX_PHYSICS_HIERARCHICALGRID_CONTACTZONE_EDGE_DATA_SET
      class AGXPHYSICS_EXPORT ContactZone_EdgeData : public agx::Physics::HierarchicalGrid::ContactZoneData
      {
      public:
        ContactZone_EdgeInstance operator[] (size_t index);

      public:
        agxData::Array< ContactZone_EdgePtr >& instance;
        agxData::Array< agx::Physics::HierarchicalGrid::ContactZonePtr > internalConnections;
        static const size_t internalConnectionsArraySize = 2;
        agxData::Array< agx::Physics::HierarchicalGrid::ContactZonePtr > cornerConnections;
        static const size_t cornerConnectionsArraySize = 2;

      public:
        typedef agx::Physics::HierarchicalGrid::ContactZonePtr internalConnectionsType;
        typedef agx::Physics::HierarchicalGrid::ContactZonePtr cornerConnectionsType;

      public:
        ContactZone_EdgeData(agxData::EntityStorage* storage);
        ContactZone_EdgeData();

      protected:
        virtual ~ContactZone_EdgeData() {}
        virtual void setNumElements(agx::Index numElements) override;

      private:
        ContactZone_EdgeData& operator= (const ContactZone_EdgeData&) { return *this; }

      };
      #endif
      DOXYGEN_END_INTERNAL_BLOCK()


      DOXYGEN_START_INTERNAL_BLOCK()
      class AGXPHYSICS_EXPORT ContactZone_EdgeSemantics : public agx::Physics::HierarchicalGrid::ContactZoneSemantics
      {
      public:

        // Automatic getters

        // Semantics defined by explicit kernels

        // Automatic setters


      protected:
        friend class ContactZone_EdgePtr;
        friend class ContactZone_EdgeInstance;
        ContactZone_EdgeSemantics();
      };
      DOXYGEN_END_INTERNAL_BLOCK()


      /**
      Pointer to a entity instance of type Physics.HierarchicalGrid.ContactZone_Edge
      */
      class CALLABLE ContactZone_EdgePtr : public agx::Physics::HierarchicalGrid::ContactZonePtr
      {
      public:
        typedef ContactZone_EdgeModel ModelType;
        typedef ContactZone_EdgeData DataType;
        typedef ContactZone_EdgeInstance InstanceType;

      public:
        AGXPHYSICS_EXPORT ContactZone_EdgePtr();
        AGXPHYSICS_EXPORT ContactZone_EdgePtr(agxData::EntityStorage* storage, agx::Index id);
        AGXPHYSICS_EXPORT ContactZone_EdgePtr(const agxData::EntityPtr& ptr);
        AGXPHYSICS_EXPORT ContactZone_EdgePtr(const agxData::EntityInstance& instance);
        AGXPHYSICS_EXPORT ContactZone_EdgePtr& operator= (const agxData::EntityPtr& ptr);
        AGXPHYSICS_EXPORT ContactZone_EdgePtr& operator= (const agxData::EntityInstance& instance);
        AGXPHYSICS_EXPORT ContactZone_EdgeInstance instance();
        AGXPHYSICS_EXPORT const ContactZone_EdgeInstance instance() const;

        AGXPHYSICS_EXPORT ContactZone_EdgeSemantics* operator->();
        AGXPHYSICS_EXPORT const ContactZone_EdgeSemantics* operator->() const;

        ContactZone_EdgeData* getData();
        const ContactZone_EdgeData* getData() const;


        AGXPHYSICS_EXPORT agxData::Array< agx::Physics::HierarchicalGrid::ContactZonePtr > internalConnections();
        AGXPHYSICS_EXPORT agxData::Array< agx::Physics::HierarchicalGrid::ContactZonePtr > const internalConnections() const;
        AGXPHYSICS_EXPORT agx::Physics::HierarchicalGrid::ContactZonePtr& internalConnections(size_t index);
        AGXPHYSICS_EXPORT agx::Physics::HierarchicalGrid::ContactZonePtr const& internalConnections(size_t index) const;

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
      class AGXPHYSICS_EXPORT ContactZone_EdgeInstance : public agx::Physics::HierarchicalGrid::ContactZoneInstance
      {
      public:
        ContactZone_EdgeInstance();
        ContactZone_EdgeInstance(ContactZone_EdgeData* data, agx::Index index);
        ContactZone_EdgeInstance(agxData::EntityStorage *storage, agx::Index index);
        ContactZone_EdgeInstance(const agxData::EntityInstance& other);
        ContactZone_EdgeInstance(const agxData::EntityPtr& ptr);

        ContactZone_EdgeData* getData();
        const ContactZone_EdgeData* getData() const;

      public:
        agxData::Array< agx::Physics::HierarchicalGrid::ContactZonePtr > internalConnections();
        agxData::Array< agx::Physics::HierarchicalGrid::ContactZonePtr > const internalConnections() const;
        agx::Physics::HierarchicalGrid::ContactZonePtr& internalConnections(size_t index);
        agx::Physics::HierarchicalGrid::ContactZonePtr const& internalConnections(size_t index) const;

        agxData::Array< agx::Physics::HierarchicalGrid::ContactZonePtr > cornerConnections();
        agxData::Array< agx::Physics::HierarchicalGrid::ContactZonePtr > const cornerConnections() const;
        agx::Physics::HierarchicalGrid::ContactZonePtr& cornerConnections(size_t index);
        agx::Physics::HierarchicalGrid::ContactZonePtr const& cornerConnections(size_t index) const;

      };
      DOXYGEN_END_INTERNAL_BLOCK()



      typedef agx::VectorPOD<ContactZone_EdgePtr> ContactZone_EdgePtrVector;
      typedef agxData::Array<ContactZone_EdgePtr> ContactZone_EdgePtrArray;



      DOXYGEN_START_INTERNAL_BLOCK()
      /* Implementation */
      //-----------------------------------------------------------------------------------------------------
      //-----------------------------------------------------------------------------------------------------
      inline ContactZone_EdgeInstance agx::Physics::HierarchicalGrid::ContactZone_EdgeData::operator[] (size_t index) { return ContactZone_EdgeInstance(this, (agx::Index)index); }
      //-----------------------------------------------------------------------------------------------------
      AGX_FORCE_INLINE ContactZone_EdgePtr::ContactZone_EdgePtr() {}
      AGX_FORCE_INLINE ContactZone_EdgePtr::ContactZone_EdgePtr(agxData::EntityStorage* storage, agx::Index id) : agx::Physics::HierarchicalGrid::ContactZonePtr(storage, id) {}
      AGX_FORCE_INLINE ContactZone_EdgePtr::ContactZone_EdgePtr(const agxData::EntityPtr& ptr) : agx::Physics::HierarchicalGrid::ContactZonePtr(ptr)
      {
        agxAssertN(!ptr || ptr.isInstanceOf(ContactZone_EdgeModel::instance()),
          "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
          EntityPtr::getModel()->fullPath().c_str(), ContactZone_EdgeModel::instance()->fullPath().c_str());
      }

      AGX_FORCE_INLINE ContactZone_EdgePtr::ContactZone_EdgePtr(const agxData::EntityInstance& instance) : agx::Physics::HierarchicalGrid::ContactZonePtr(instance)
      {
        agxAssertN(!instance || instance.isInstanceOf(ContactZone_EdgeModel::instance()),
          "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
          EntityPtr::getModel()->fullPath().c_str(), ContactZone_EdgeModel::instance()->fullPath().c_str());
      }

      AGX_FORCE_INLINE ContactZone_EdgePtr& ContactZone_EdgePtr::operator= (const agxData::EntityPtr& ptr)
      {
        agxData::EntityPtr::operator= (ptr);
        agxAssertN(!ptr || ptr.isInstanceOf(ContactZone_EdgeModel::instance()),
          "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
          EntityPtr::getModel()->fullPath().c_str(), ContactZone_EdgeModel::instance()->fullPath().c_str());
        return *this;
      }

      AGX_FORCE_INLINE ContactZone_EdgePtr& ContactZone_EdgePtr::operator= (const agxData::EntityInstance& instance)
      {
        agxData::EntityPtr::operator= (instance);
        agxAssertN(!instance || instance.isInstanceOf(ContactZone_EdgeModel::instance()),
          "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
          EntityPtr::getModel()->fullPath().c_str(), ContactZone_EdgeModel::instance()->fullPath().c_str());
        return *this;
      }

      inline ContactZone_EdgeInstance ContactZone_EdgePtr::instance() { return agxData::EntityPtr::instance(); }
      inline const ContactZone_EdgeInstance ContactZone_EdgePtr::instance() const { return agxData::EntityPtr::instance(); }
      AGX_FORCE_INLINE ContactZone_EdgeSemantics* ContactZone_EdgePtr::operator->() { return (ContactZone_EdgeSemantics* )this; }
      AGX_FORCE_INLINE const ContactZone_EdgeSemantics* ContactZone_EdgePtr::operator->() const { return (const ContactZone_EdgeSemantics* )this; }
      AGX_FORCE_INLINE ContactZone_EdgeData* ContactZone_EdgePtr::getData() { return static_cast<ContactZone_EdgeData* >(agxData::EntityPtr::getData()); }
      AGX_FORCE_INLINE const ContactZone_EdgeData* ContactZone_EdgePtr::getData() const { return static_cast<const ContactZone_EdgeData* >(agxData::EntityPtr::getData()); }

      AGX_FORCE_INLINE agxData::Array< agx::Physics::HierarchicalGrid::ContactZonePtr > ContactZone_EdgePtr::internalConnections() { verifyIndex(); return getData()->internalConnections.slice(agx::IndexRange32(calculateIndex() * (agx::Index)ContactZone_EdgeData::internalConnectionsArraySize, (calculateIndex()+1) * (agx::Index)ContactZone_EdgeData::internalConnectionsArraySize)); }
      AGX_FORCE_INLINE const agxData::Array< agx::Physics::HierarchicalGrid::ContactZonePtr > ContactZone_EdgePtr::internalConnections() const { verifyIndex(); return getData()->internalConnections.slice(agx::IndexRange32(calculateIndex() * (agx::Index)ContactZone_EdgeData::internalConnectionsArraySize, (calculateIndex()+1) * (agx::Index)ContactZone_EdgeData::internalConnectionsArraySize)); }
      AGX_FORCE_INLINE agx::Physics::HierarchicalGrid::ContactZonePtr& ContactZone_EdgePtr::internalConnections(size_t index) { return this->internalConnections()[index]; }
      AGX_FORCE_INLINE agx::Physics::HierarchicalGrid::ContactZonePtr const& ContactZone_EdgePtr::internalConnections(size_t index) const { return this->internalConnections()[index]; }

      AGX_FORCE_INLINE agxData::Array< agx::Physics::HierarchicalGrid::ContactZonePtr > ContactZone_EdgePtr::cornerConnections() { verifyIndex(); return getData()->cornerConnections.slice(agx::IndexRange32(calculateIndex() * (agx::Index)ContactZone_EdgeData::cornerConnectionsArraySize, (calculateIndex()+1) * (agx::Index)ContactZone_EdgeData::cornerConnectionsArraySize)); }
      AGX_FORCE_INLINE const agxData::Array< agx::Physics::HierarchicalGrid::ContactZonePtr > ContactZone_EdgePtr::cornerConnections() const { verifyIndex(); return getData()->cornerConnections.slice(agx::IndexRange32(calculateIndex() * (agx::Index)ContactZone_EdgeData::cornerConnectionsArraySize, (calculateIndex()+1) * (agx::Index)ContactZone_EdgeData::cornerConnectionsArraySize)); }
      AGX_FORCE_INLINE agx::Physics::HierarchicalGrid::ContactZonePtr& ContactZone_EdgePtr::cornerConnections(size_t index) { return this->cornerConnections()[index]; }
      AGX_FORCE_INLINE agx::Physics::HierarchicalGrid::ContactZonePtr const& ContactZone_EdgePtr::cornerConnections(size_t index) const { return this->cornerConnections()[index]; }

      //-----------------------------------------------------------------------------------------------------
      AGX_FORCE_INLINE ContactZone_EdgeInstance::ContactZone_EdgeInstance() {}
      AGX_FORCE_INLINE ContactZone_EdgeInstance::ContactZone_EdgeInstance(ContactZone_EdgeData* data, agx::Index index) : agx::Physics::HierarchicalGrid::ContactZoneInstance(data, index) {}
      AGX_FORCE_INLINE ContactZone_EdgeInstance::ContactZone_EdgeInstance(agxData::EntityStorage* storage, agx::Index index) : agx::Physics::HierarchicalGrid::ContactZoneInstance(storage, index) {}
      AGX_FORCE_INLINE ContactZone_EdgeInstance::ContactZone_EdgeInstance(const agxData::EntityInstance& other) : agx::Physics::HierarchicalGrid::ContactZoneInstance(other)
      {
        agxAssertN(!other || other.isInstanceOf(ContactZone_EdgeModel::instance()),
          "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
          EntityInstance::getModel()->fullPath().c_str(), ContactZone_EdgeModel::instance()->fullPath().c_str());
      }

      AGX_FORCE_INLINE ContactZone_EdgeInstance::ContactZone_EdgeInstance(const agxData::EntityPtr& ptr) : agx::Physics::HierarchicalGrid::ContactZoneInstance(ptr)
      {
        agxAssertN(!ptr || ptr.isInstanceOf(ContactZone_EdgeModel::instance()),
          "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
          EntityInstance::getModel()->fullPath().c_str(), ContactZone_EdgeModel::instance()->fullPath().c_str());
      }


      AGX_FORCE_INLINE ContactZone_EdgeData* ContactZone_EdgeInstance::getData() { return static_cast<ContactZone_EdgeData* >(agxData::EntityInstance::getData()); }
      AGX_FORCE_INLINE const ContactZone_EdgeData* ContactZone_EdgeInstance::getData() const { return static_cast<const ContactZone_EdgeData* >(agxData::EntityInstance::getData()); }

      AGX_FORCE_INLINE agxData::Array< agx::Physics::HierarchicalGrid::ContactZonePtr > ContactZone_EdgeInstance::internalConnections() { verifyIndex(); return getData()->internalConnections.slice(agx::IndexRange32(getIndex() * (agx::Index)ContactZone_EdgeData::internalConnectionsArraySize, (getIndex()+1) * (agx::Index)ContactZone_EdgeData::internalConnectionsArraySize)); }
      AGX_FORCE_INLINE const agxData::Array< agx::Physics::HierarchicalGrid::ContactZonePtr > ContactZone_EdgeInstance::internalConnections() const { verifyIndex(); return getData()->internalConnections.slice(agx::IndexRange32(getIndex() * (agx::Index)ContactZone_EdgeData::internalConnectionsArraySize, (getIndex()+1) * (agx::Index)ContactZone_EdgeData::internalConnectionsArraySize)); }
      AGX_FORCE_INLINE agx::Physics::HierarchicalGrid::ContactZonePtr& ContactZone_EdgeInstance::internalConnections(size_t index) { return this->internalConnections()[index]; }
      AGX_FORCE_INLINE agx::Physics::HierarchicalGrid::ContactZonePtr const& ContactZone_EdgeInstance::internalConnections(size_t index) const { return this->internalConnections()[index]; }

      AGX_FORCE_INLINE agxData::Array< agx::Physics::HierarchicalGrid::ContactZonePtr > ContactZone_EdgeInstance::cornerConnections() { verifyIndex(); return getData()->cornerConnections.slice(agx::IndexRange32(getIndex() * (agx::Index)ContactZone_EdgeData::cornerConnectionsArraySize, (getIndex()+1) * (agx::Index)ContactZone_EdgeData::cornerConnectionsArraySize)); }
      AGX_FORCE_INLINE const agxData::Array< agx::Physics::HierarchicalGrid::ContactZonePtr > ContactZone_EdgeInstance::cornerConnections() const { verifyIndex(); return getData()->cornerConnections.slice(agx::IndexRange32(getIndex() * (agx::Index)ContactZone_EdgeData::cornerConnectionsArraySize, (getIndex()+1) * (agx::Index)ContactZone_EdgeData::cornerConnectionsArraySize)); }
      AGX_FORCE_INLINE agx::Physics::HierarchicalGrid::ContactZonePtr& ContactZone_EdgeInstance::cornerConnections(size_t index) { return this->cornerConnections()[index]; }
      AGX_FORCE_INLINE agx::Physics::HierarchicalGrid::ContactZonePtr const& ContactZone_EdgeInstance::cornerConnections(size_t index) const { return this->cornerConnections()[index]; }

      //-----------------------------------------------------------------------------------------------------
      AGX_FORCE_INLINE ContactZone_EdgeSemantics::ContactZone_EdgeSemantics() {}
      //-----------------------------------------------------------------------------------------------------
      DOXYGEN_END_INTERNAL_BLOCK()
    }
  }
}

AGX_TYPE_BINDING(agx::Physics::HierarchicalGrid::ContactZone_EdgePtr, "Physics.HierarchicalGrid.ContactZone_EdgePtr")
AGX_TYPE_BINDING(agx::Physics::HierarchicalGrid::ContactZone_EdgeInstance, "Physics.HierarchicalGrid.ContactZone_EdgeInstance")

#ifdef _MSC_VER
# pragma warning(pop)
#endif

#undef AGX_ENTITY_WRAPPER
#undef AGX_ENTITY_NAMESPACE
#endif

