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

#ifndef GENERATED_AGX_PHYSICS_HIERARCHICALGRID_CONTACTZONEDEPENDENCY_H_PLUGIN
#define GENERATED_AGX_PHYSICS_HIERARCHICALGRID_CONTACTZONEDEPENDENCY_H_PLUGIN

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

namespace agx { namespace Physics { namespace HierarchicalGrid { class ContactZoneInstance; }}}
namespace agx { namespace Physics { namespace HierarchicalGrid { class ContactZoneInstance; }}}
namespace agx { namespace Physics { namespace HierarchicalGrid { class ContactZoneDependencyPtr; }}}

namespace agx
{
  namespace Physics
  {
    namespace HierarchicalGrid
    {

      class ContactZoneDependencyModel;
      class ContactZoneDependencyData;
      class ContactZoneDependencyPtr;
      class ContactZoneDependencyInstance;
      class ContactZoneDependencySemantics;


      AGX_DECLARE_POINTER_TYPES(ContactZoneDependencyModel);

      /** 
      Abstract description of the data attributes for the Physics.HierarchicalGrid.ContactZoneDependency entity.
      */ 
      class AGXPHYSICS_EXPORT ContactZoneDependencyModel : public agxData::EntityModel
      {
      public:
        typedef ContactZoneDependencyPtr PtrT;

        ContactZoneDependencyModel(const agx::String& name = "ContactZoneDependency");

        /// \return The entity model singleton.
        static ContactZoneDependencyModel* instance();

        /// Create and return a pointer to a new instance in the default storage for this entity model.
        static ContactZoneDependencyPtr createInstance();

        /// \return The default storage for this entity model.
        static agxData::EntityStorage* defaultStorage();

        /// This is part of internal cleanup and should not be called by users
        virtual void shutdownCleanup() override;



        /* Attributes */
        static agxData::ScalarAttributeT< agx::Physics::HierarchicalGrid::ContactZoneInstance >* parentAttribute;
        static agxData::ScalarAttributeT< agx::Physics::HierarchicalGrid::ContactZoneInstance >* childAttribute;
        static agxData::ScalarAttributeT< agx::Physics::HierarchicalGrid::ContactZoneDependencyPtr >* nextAttribute;

      protected:
        virtual ~ContactZoneDependencyModel();
        virtual agxData::EntityData* createData(agxData::EntityStorage* storage) override;
        virtual void configure(agx::TiXmlElement* eEntity) override;
        virtual void initAttributeAccessors() override;
        virtual void construct(agxData::EntityPtr instance) override;
        void construct(agx::Physics::HierarchicalGrid::ContactZoneDependencyPtr contactZoneDependency);
      };


      DOXYGEN_START_INTERNAL_BLOCK()
      #ifndef AGX_PHYSICS_HIERARCHICALGRID_CONTACTZONEDEPENDENCY_DATA_SET_OVERRIDE
      #define AGX_PHYSICS_HIERARCHICALGRID_CONTACTZONEDEPENDENCY_DATA_SET
      class AGXPHYSICS_EXPORT ContactZoneDependencyData : public agxData::EntityData
      {
      public:
        ContactZoneDependencyInstance operator[] (size_t index);

      public:
        agxData::Array< ContactZoneDependencyPtr >& instance;
        agxData::Array< agx::Physics::HierarchicalGrid::ContactZoneInstance > parent;
        agxData::Array< agx::Physics::HierarchicalGrid::ContactZoneInstance > child;
        agxData::Array< agx::Physics::HierarchicalGrid::ContactZoneDependencyPtr > next;

      public:
        typedef agx::Physics::HierarchicalGrid::ContactZoneInstance parentType;
        typedef agx::Physics::HierarchicalGrid::ContactZoneInstance childType;
        typedef agx::Physics::HierarchicalGrid::ContactZoneDependencyPtr nextType;

      public:
        ContactZoneDependencyData(agxData::EntityStorage* storage);
        ContactZoneDependencyData();

      protected:
        virtual ~ContactZoneDependencyData() {}
        virtual void setNumElements(agx::Index numElements) override;

      private:
        ContactZoneDependencyData& operator= (const ContactZoneDependencyData&) { return *this; }

      };
      #endif
      DOXYGEN_END_INTERNAL_BLOCK()


      DOXYGEN_START_INTERNAL_BLOCK()
      class AGXPHYSICS_EXPORT ContactZoneDependencySemantics : protected agxData::EntityPtr
      {
      public:

        // Automatic getters
        agx::Physics::HierarchicalGrid::ContactZoneInstance const& getParent() const;
        agx::Physics::HierarchicalGrid::ContactZoneInstance const& getChild() const;
        agx::Physics::HierarchicalGrid::ContactZoneDependencyPtr const& getNext() const;

        // Semantics defined by explicit kernels

        // Automatic setters
        void setParent(agx::Physics::HierarchicalGrid::ContactZoneInstance const& value);
        void setChild(agx::Physics::HierarchicalGrid::ContactZoneInstance const& value);
        void setNext(agx::Physics::HierarchicalGrid::ContactZoneDependencyPtr const& value);


      protected:
        friend class ContactZoneDependencyPtr;
        friend class ContactZoneDependencyInstance;
        ContactZoneDependencySemantics();
      };
      DOXYGEN_END_INTERNAL_BLOCK()


      /**
      Pointer to a entity instance of type Physics.HierarchicalGrid.ContactZoneDependency
      */
      class CALLABLE ContactZoneDependencyPtr : public agxData::EntityPtr
      {
      public:
        typedef ContactZoneDependencyModel ModelType;
        typedef ContactZoneDependencyData DataType;
        typedef ContactZoneDependencyInstance InstanceType;

      public:
        AGXPHYSICS_EXPORT ContactZoneDependencyPtr();
        AGXPHYSICS_EXPORT ContactZoneDependencyPtr(agxData::EntityStorage* storage, agx::Index id);
        AGXPHYSICS_EXPORT ContactZoneDependencyPtr(const agxData::EntityPtr& ptr);
        AGXPHYSICS_EXPORT ContactZoneDependencyPtr(const agxData::EntityInstance& instance);
        AGXPHYSICS_EXPORT ContactZoneDependencyPtr& operator= (const agxData::EntityPtr& ptr);
        AGXPHYSICS_EXPORT ContactZoneDependencyPtr& operator= (const agxData::EntityInstance& instance);
        AGXPHYSICS_EXPORT ContactZoneDependencyInstance instance();
        AGXPHYSICS_EXPORT const ContactZoneDependencyInstance instance() const;

        AGXPHYSICS_EXPORT ContactZoneDependencySemantics* operator->();
        AGXPHYSICS_EXPORT const ContactZoneDependencySemantics* operator->() const;

        ContactZoneDependencyData* getData();
        const ContactZoneDependencyData* getData() const;


        /// \return reference to the parent attribute
        AGXPHYSICS_EXPORT agx::Physics::HierarchicalGrid::ContactZoneInstance& parent();
        /// \return const reference to the parent attribute
        AGXPHYSICS_EXPORT agx::Physics::HierarchicalGrid::ContactZoneInstance const& parent() const;

        /// \return reference to the child attribute
        AGXPHYSICS_EXPORT agx::Physics::HierarchicalGrid::ContactZoneInstance& child();
        /// \return const reference to the child attribute
        AGXPHYSICS_EXPORT agx::Physics::HierarchicalGrid::ContactZoneInstance const& child() const;

        /// \return reference to the next attribute
        AGXPHYSICS_EXPORT agx::Physics::HierarchicalGrid::ContactZoneDependencyPtr& next();
        /// \return const reference to the next attribute
        AGXPHYSICS_EXPORT agx::Physics::HierarchicalGrid::ContactZoneDependencyPtr const& next() const;

        DOXYGEN_START_INTERNAL_BLOCK()
        // Only used by tolua!
        #ifdef AGX_GENERATED_TOLUA_WRAPPER
        inline agx::Physics::HierarchicalGrid::ContactZoneInstance const& getParent() const { return (*this)->getParent(); }
        inline agx::Physics::HierarchicalGrid::ContactZoneInstance const& getChild() const { return (*this)->getChild(); }
        inline agx::Physics::HierarchicalGrid::ContactZoneDependencyPtr const& getNext() const { return (*this)->getNext(); }


        inline void setParent(agx::Physics::HierarchicalGrid::ContactZoneInstance const& value) { (*this)->setParent(value); }
        inline void setChild(agx::Physics::HierarchicalGrid::ContactZoneInstance const& value) { (*this)->setChild(value); }
        inline void setNext(agx::Physics::HierarchicalGrid::ContactZoneDependencyPtr const& value) { (*this)->setNext(value); }
        #endif
        DOXYGEN_END_INTERNAL_BLOCK()

      };


      DOXYGEN_START_INTERNAL_BLOCK()
      class AGXPHYSICS_EXPORT ContactZoneDependencyInstance : public agxData::EntityInstance
      {
      public:
        ContactZoneDependencyInstance();
        ContactZoneDependencyInstance(ContactZoneDependencyData* data, agx::Index index);
        ContactZoneDependencyInstance(agxData::EntityStorage *storage, agx::Index index);
        ContactZoneDependencyInstance(const agxData::EntityInstance& other);
        ContactZoneDependencyInstance(const agxData::EntityPtr& ptr);

        ContactZoneDependencyData* getData();
        const ContactZoneDependencyData* getData() const;

      public:
        /// \return reference to the parent attribute
        agx::Physics::HierarchicalGrid::ContactZoneInstance& parent();
        /// \return const reference to the parent attribute
        agx::Physics::HierarchicalGrid::ContactZoneInstance const& parent() const;

        /// \return reference to the child attribute
        agx::Physics::HierarchicalGrid::ContactZoneInstance& child();
        /// \return const reference to the child attribute
        agx::Physics::HierarchicalGrid::ContactZoneInstance const& child() const;

        /// \return reference to the next attribute
        agx::Physics::HierarchicalGrid::ContactZoneDependencyPtr& next();
        /// \return const reference to the next attribute
        agx::Physics::HierarchicalGrid::ContactZoneDependencyPtr const& next() const;

      };
      DOXYGEN_END_INTERNAL_BLOCK()



      typedef agx::VectorPOD<ContactZoneDependencyPtr> ContactZoneDependencyPtrVector;
      typedef agxData::Array<ContactZoneDependencyPtr> ContactZoneDependencyPtrArray;



      DOXYGEN_START_INTERNAL_BLOCK()
      /* Implementation */
      //-----------------------------------------------------------------------------------------------------
      //-----------------------------------------------------------------------------------------------------
      inline ContactZoneDependencyInstance agx::Physics::HierarchicalGrid::ContactZoneDependencyData::operator[] (size_t index) { return ContactZoneDependencyInstance(this, (agx::Index)index); }
      //-----------------------------------------------------------------------------------------------------
      AGX_FORCE_INLINE ContactZoneDependencyPtr::ContactZoneDependencyPtr() {}
      AGX_FORCE_INLINE ContactZoneDependencyPtr::ContactZoneDependencyPtr(agxData::EntityStorage* storage, agx::Index id) : agxData::EntityPtr(storage, id) {}
      AGX_FORCE_INLINE ContactZoneDependencyPtr::ContactZoneDependencyPtr(const agxData::EntityPtr& ptr) : agxData::EntityPtr(ptr)
      {
        agxAssertN(!ptr || ptr.isInstanceOf(ContactZoneDependencyModel::instance()),
          "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
          EntityPtr::getModel()->fullPath().c_str(), ContactZoneDependencyModel::instance()->fullPath().c_str());
      }

      AGX_FORCE_INLINE ContactZoneDependencyPtr::ContactZoneDependencyPtr(const agxData::EntityInstance& instance) : agxData::EntityPtr(instance)
      {
        agxAssertN(!instance || instance.isInstanceOf(ContactZoneDependencyModel::instance()),
          "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
          EntityPtr::getModel()->fullPath().c_str(), ContactZoneDependencyModel::instance()->fullPath().c_str());
      }

      AGX_FORCE_INLINE ContactZoneDependencyPtr& ContactZoneDependencyPtr::operator= (const agxData::EntityPtr& ptr)
      {
        agxData::EntityPtr::operator= (ptr);
        agxAssertN(!ptr || ptr.isInstanceOf(ContactZoneDependencyModel::instance()),
          "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
          EntityPtr::getModel()->fullPath().c_str(), ContactZoneDependencyModel::instance()->fullPath().c_str());
        return *this;
      }

      AGX_FORCE_INLINE ContactZoneDependencyPtr& ContactZoneDependencyPtr::operator= (const agxData::EntityInstance& instance)
      {
        agxData::EntityPtr::operator= (instance);
        agxAssertN(!instance || instance.isInstanceOf(ContactZoneDependencyModel::instance()),
          "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
          EntityPtr::getModel()->fullPath().c_str(), ContactZoneDependencyModel::instance()->fullPath().c_str());
        return *this;
      }

      inline ContactZoneDependencyInstance ContactZoneDependencyPtr::instance() { return agxData::EntityPtr::instance(); }
      inline const ContactZoneDependencyInstance ContactZoneDependencyPtr::instance() const { return agxData::EntityPtr::instance(); }
      AGX_FORCE_INLINE ContactZoneDependencySemantics* ContactZoneDependencyPtr::operator->() { return (ContactZoneDependencySemantics* )this; }
      AGX_FORCE_INLINE const ContactZoneDependencySemantics* ContactZoneDependencyPtr::operator->() const { return (const ContactZoneDependencySemantics* )this; }
      AGX_FORCE_INLINE ContactZoneDependencyData* ContactZoneDependencyPtr::getData() { return static_cast<ContactZoneDependencyData* >(agxData::EntityPtr::getData()); }
      AGX_FORCE_INLINE const ContactZoneDependencyData* ContactZoneDependencyPtr::getData() const { return static_cast<const ContactZoneDependencyData* >(agxData::EntityPtr::getData()); }

      AGX_FORCE_INLINE agx::Physics::HierarchicalGrid::ContactZoneInstance& ContactZoneDependencyPtr::parent() { verifyIndex(); return getData()->parent[calculateIndex()]; }
      AGX_FORCE_INLINE agx::Physics::HierarchicalGrid::ContactZoneInstance const& ContactZoneDependencyPtr::parent() const { verifyIndex(); return getData()->parent[calculateIndex()]; }

      AGX_FORCE_INLINE agx::Physics::HierarchicalGrid::ContactZoneInstance& ContactZoneDependencyPtr::child() { verifyIndex(); return getData()->child[calculateIndex()]; }
      AGX_FORCE_INLINE agx::Physics::HierarchicalGrid::ContactZoneInstance const& ContactZoneDependencyPtr::child() const { verifyIndex(); return getData()->child[calculateIndex()]; }

      AGX_FORCE_INLINE agx::Physics::HierarchicalGrid::ContactZoneDependencyPtr& ContactZoneDependencyPtr::next() { verifyIndex(); return getData()->next[calculateIndex()]; }
      AGX_FORCE_INLINE agx::Physics::HierarchicalGrid::ContactZoneDependencyPtr const& ContactZoneDependencyPtr::next() const { verifyIndex(); return getData()->next[calculateIndex()]; }

      //-----------------------------------------------------------------------------------------------------
      AGX_FORCE_INLINE ContactZoneDependencyInstance::ContactZoneDependencyInstance() {}
      AGX_FORCE_INLINE ContactZoneDependencyInstance::ContactZoneDependencyInstance(ContactZoneDependencyData* data, agx::Index index) : agxData::EntityInstance(data, index) {}
      AGX_FORCE_INLINE ContactZoneDependencyInstance::ContactZoneDependencyInstance(agxData::EntityStorage* storage, agx::Index index) : agxData::EntityInstance(storage, index) {}
      AGX_FORCE_INLINE ContactZoneDependencyInstance::ContactZoneDependencyInstance(const agxData::EntityInstance& other) : agxData::EntityInstance(other)
      {
        agxAssertN(!other || other.isInstanceOf(ContactZoneDependencyModel::instance()),
          "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
          EntityInstance::getModel()->fullPath().c_str(), ContactZoneDependencyModel::instance()->fullPath().c_str());
      }

      AGX_FORCE_INLINE ContactZoneDependencyInstance::ContactZoneDependencyInstance(const agxData::EntityPtr& ptr) : agxData::EntityInstance(ptr)
      {
        agxAssertN(!ptr || ptr.isInstanceOf(ContactZoneDependencyModel::instance()),
          "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
          EntityInstance::getModel()->fullPath().c_str(), ContactZoneDependencyModel::instance()->fullPath().c_str());
      }


      AGX_FORCE_INLINE ContactZoneDependencyData* ContactZoneDependencyInstance::getData() { return static_cast<ContactZoneDependencyData* >(agxData::EntityInstance::getData()); }
      AGX_FORCE_INLINE const ContactZoneDependencyData* ContactZoneDependencyInstance::getData() const { return static_cast<const ContactZoneDependencyData* >(agxData::EntityInstance::getData()); }

      AGX_FORCE_INLINE agx::Physics::HierarchicalGrid::ContactZoneInstance& ContactZoneDependencyInstance::parent() { verifyIndex(); return getData()->parent[getIndex()]; }
      AGX_FORCE_INLINE agx::Physics::HierarchicalGrid::ContactZoneInstance const& ContactZoneDependencyInstance::parent() const { verifyIndex(); return getData()->parent[getIndex()]; }

      AGX_FORCE_INLINE agx::Physics::HierarchicalGrid::ContactZoneInstance& ContactZoneDependencyInstance::child() { verifyIndex(); return getData()->child[getIndex()]; }
      AGX_FORCE_INLINE agx::Physics::HierarchicalGrid::ContactZoneInstance const& ContactZoneDependencyInstance::child() const { verifyIndex(); return getData()->child[getIndex()]; }

      AGX_FORCE_INLINE agx::Physics::HierarchicalGrid::ContactZoneDependencyPtr& ContactZoneDependencyInstance::next() { verifyIndex(); return getData()->next[getIndex()]; }
      AGX_FORCE_INLINE agx::Physics::HierarchicalGrid::ContactZoneDependencyPtr const& ContactZoneDependencyInstance::next() const { verifyIndex(); return getData()->next[getIndex()]; }

      //-----------------------------------------------------------------------------------------------------
      AGX_FORCE_INLINE ContactZoneDependencySemantics::ContactZoneDependencySemantics() {}
      //-----------------------------------------------------------------------------------------------------
      DOXYGEN_END_INTERNAL_BLOCK()
    }
  }
}

AGX_TYPE_BINDING(agx::Physics::HierarchicalGrid::ContactZoneDependencyPtr, "Physics.HierarchicalGrid.ContactZoneDependencyPtr")
AGX_TYPE_BINDING(agx::Physics::HierarchicalGrid::ContactZoneDependencyInstance, "Physics.HierarchicalGrid.ContactZoneDependencyInstance")

#ifdef _MSC_VER
# pragma warning(pop)
#endif

#undef AGX_ENTITY_WRAPPER
#undef AGX_ENTITY_NAMESPACE
#endif

