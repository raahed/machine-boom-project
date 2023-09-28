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

#ifndef GENERATED_AGX_PHYSICS_CONTACT_H_PLUGIN
#define GENERATED_AGX_PHYSICS_CONTACT_H_PLUGIN

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
#include <agx/Physics/InteractionEntity.h>
#include <agx/Integer.h>
#include <agx/Physics/ContactMaterialEntity.h>

namespace agx { namespace Physics { class ContactMaterialPtr; }}

namespace agx
{
  namespace Physics
  {

    class ContactModel;
    class ContactData;
    class ContactPtr;
    class ContactInstance;
    class ContactSemantics;


    AGX_DECLARE_POINTER_TYPES(ContactModel);

    /** 
    Abstract description of the data attributes for the Physics.Contact entity.
    */ 
    class AGXPHYSICS_EXPORT ContactModel : public agx::Physics::InteractionModel
    {
    public:
      typedef ContactPtr PtrT;

      ContactModel(const agx::String& name = "Contact");

      /// \return The entity model singleton.
      static ContactModel* instance();

      /// Create and return a pointer to a new instance in the default storage for this entity model.
      static ContactPtr createInstance();

      /// \return The default storage for this entity model.
      static agxData::EntityStorage* defaultStorage();

      /// This is part of internal cleanup and should not be called by users
      virtual void shutdownCleanup() override;



      /* Attributes */
      static agxData::ScalarAttributeT< agx::UInt32 >* constraintIndexAttribute;
      static agxData::ScalarAttributeT< agx::Physics::ContactMaterialPtr >* materialAttribute;
      static agxData::ScalarAttributeT< agx::UInt32 >* contactZoneIndexAttribute;
      static agxData::ScalarAttributeT< agx::UInt16 >* contactZoneSlotAttribute;

    protected:
      virtual ~ContactModel();
      virtual agxData::EntityData* createData(agxData::EntityStorage* storage) override;
      virtual void configure(agx::TiXmlElement* eEntity) override;
      virtual void initAttributeAccessors() override;
      virtual void construct(agxData::EntityPtr instance) override;
      void construct(agx::Physics::ContactPtr contact);
    };


    DOXYGEN_START_INTERNAL_BLOCK()
    #ifndef AGX_PHYSICS_CONTACT_DATA_SET_OVERRIDE
    #define AGX_PHYSICS_CONTACT_DATA_SET
    class AGXPHYSICS_EXPORT ContactData : public agx::Physics::InteractionData
    {
    public:
      ContactInstance operator[] (size_t index);

    public:
      agxData::Array< ContactPtr >& instance;
      agxData::Array< agx::UInt32 > constraintIndex;
      agxData::Array< agx::Physics::ContactMaterialPtr > material;
      agxData::Array< agx::UInt32 > contactZoneIndex;
      agxData::Array< agx::UInt16 > contactZoneSlot;

    public:
      typedef agx::UInt32 constraintIndexType;
      typedef agx::Physics::ContactMaterialPtr materialType;
      typedef agx::UInt32 contactZoneIndexType;
      typedef agx::UInt16 contactZoneSlotType;

    public:
      ContactData(agxData::EntityStorage* storage);
      ContactData();

    protected:
      virtual ~ContactData() {}
      virtual void setNumElements(agx::Index numElements) override;

    private:
      ContactData& operator= (const ContactData&) { return *this; }

    };
    #endif
    DOXYGEN_END_INTERNAL_BLOCK()


    DOXYGEN_START_INTERNAL_BLOCK()
    class AGXPHYSICS_EXPORT ContactSemantics : public agx::Physics::InteractionSemantics
    {
    public:

      // Automatic getters
      agx::UInt32 const& getConstraintIndex() const;
      agx::Physics::ContactMaterialPtr const& getMaterial() const;
      agx::UInt32 const& getContactZoneIndex() const;
      agx::UInt16 const& getContactZoneSlot() const;

      // Semantics defined by explicit kernels

      // Automatic setters
      void setConstraintIndex(agx::UInt32 const& value);
      void setMaterial(agx::Physics::ContactMaterialPtr const& value);
      void setContactZoneIndex(agx::UInt32 const& value);
      void setContactZoneSlot(agx::UInt16 const& value);


    protected:
      friend class ContactPtr;
      friend class ContactInstance;
      ContactSemantics();
    };
    DOXYGEN_END_INTERNAL_BLOCK()


    /**
    Pointer to a entity instance of type Physics.Contact
    */
    class CALLABLE ContactPtr : public agx::Physics::InteractionPtr
    {
    public:
      typedef ContactModel ModelType;
      typedef ContactData DataType;
      typedef ContactInstance InstanceType;

    public:
      AGXPHYSICS_EXPORT ContactPtr();
      AGXPHYSICS_EXPORT ContactPtr(agxData::EntityStorage* storage, agx::Index id);
      AGXPHYSICS_EXPORT ContactPtr(const agxData::EntityPtr& ptr);
      AGXPHYSICS_EXPORT ContactPtr(const agxData::EntityInstance& instance);
      AGXPHYSICS_EXPORT ContactPtr& operator= (const agxData::EntityPtr& ptr);
      AGXPHYSICS_EXPORT ContactPtr& operator= (const agxData::EntityInstance& instance);
      AGXPHYSICS_EXPORT ContactInstance instance();
      AGXPHYSICS_EXPORT const ContactInstance instance() const;

      AGXPHYSICS_EXPORT ContactSemantics* operator->();
      AGXPHYSICS_EXPORT const ContactSemantics* operator->() const;

      ContactData* getData();
      const ContactData* getData() const;


      /// \return reference to the constraintIndex attribute
      AGXPHYSICS_EXPORT agx::UInt32& constraintIndex();
      /// \return const reference to the constraintIndex attribute
      AGXPHYSICS_EXPORT agx::UInt32 const& constraintIndex() const;

      /// \return reference to the material attribute
      AGXPHYSICS_EXPORT agx::Physics::ContactMaterialPtr& material();
      /// \return const reference to the material attribute
      AGXPHYSICS_EXPORT agx::Physics::ContactMaterialPtr const& material() const;

      /// \return reference to the contactZoneIndex attribute
      AGXPHYSICS_EXPORT agx::UInt32& contactZoneIndex();
      /// \return const reference to the contactZoneIndex attribute
      AGXPHYSICS_EXPORT agx::UInt32 const& contactZoneIndex() const;

      /// \return reference to the contactZoneSlot attribute
      AGXPHYSICS_EXPORT agx::UInt16& contactZoneSlot();
      /// \return const reference to the contactZoneSlot attribute
      AGXPHYSICS_EXPORT agx::UInt16 const& contactZoneSlot() const;

      DOXYGEN_START_INTERNAL_BLOCK()
      // Only used by tolua!
      #ifdef AGX_GENERATED_TOLUA_WRAPPER
      inline agx::UInt32 const& getConstraintIndex() const { return (*this)->getConstraintIndex(); }
      inline agx::Physics::ContactMaterialPtr const& getMaterial() const { return (*this)->getMaterial(); }
      inline agx::UInt32 const& getContactZoneIndex() const { return (*this)->getContactZoneIndex(); }
      inline agx::UInt16 const& getContactZoneSlot() const { return (*this)->getContactZoneSlot(); }


      inline void setConstraintIndex(agx::UInt32 const& value) { (*this)->setConstraintIndex(value); }
      inline void setMaterial(agx::Physics::ContactMaterialPtr const& value) { (*this)->setMaterial(value); }
      inline void setContactZoneIndex(agx::UInt32 const& value) { (*this)->setContactZoneIndex(value); }
      inline void setContactZoneSlot(agx::UInt16 const& value) { (*this)->setContactZoneSlot(value); }
      #endif
      DOXYGEN_END_INTERNAL_BLOCK()

    };


    DOXYGEN_START_INTERNAL_BLOCK()
    class AGXPHYSICS_EXPORT ContactInstance : public agx::Physics::InteractionInstance
    {
    public:
      ContactInstance();
      ContactInstance(ContactData* data, agx::Index index);
      ContactInstance(agxData::EntityStorage *storage, agx::Index index);
      ContactInstance(const agxData::EntityInstance& other);
      ContactInstance(const agxData::EntityPtr& ptr);

      ContactData* getData();
      const ContactData* getData() const;

    public:
      /// \return reference to the constraintIndex attribute
      agx::UInt32& constraintIndex();
      /// \return const reference to the constraintIndex attribute
      agx::UInt32 const& constraintIndex() const;

      /// \return reference to the material attribute
      agx::Physics::ContactMaterialPtr& material();
      /// \return const reference to the material attribute
      agx::Physics::ContactMaterialPtr const& material() const;

      /// \return reference to the contactZoneIndex attribute
      agx::UInt32& contactZoneIndex();
      /// \return const reference to the contactZoneIndex attribute
      agx::UInt32 const& contactZoneIndex() const;

      /// \return reference to the contactZoneSlot attribute
      agx::UInt16& contactZoneSlot();
      /// \return const reference to the contactZoneSlot attribute
      agx::UInt16 const& contactZoneSlot() const;

    };
    DOXYGEN_END_INTERNAL_BLOCK()



    typedef agx::VectorPOD<ContactPtr> ContactPtrVector;
    typedef agxData::Array<ContactPtr> ContactPtrArray;



    DOXYGEN_START_INTERNAL_BLOCK()
    /* Implementation */
    //-----------------------------------------------------------------------------------------------------
    //-----------------------------------------------------------------------------------------------------
    inline ContactInstance agx::Physics::ContactData::operator[] (size_t index) { return ContactInstance(this, (agx::Index)index); }
    //-----------------------------------------------------------------------------------------------------
    AGX_FORCE_INLINE ContactPtr::ContactPtr() {}
    AGX_FORCE_INLINE ContactPtr::ContactPtr(agxData::EntityStorage* storage, agx::Index id) : agx::Physics::InteractionPtr(storage, id) {}
    AGX_FORCE_INLINE ContactPtr::ContactPtr(const agxData::EntityPtr& ptr) : agx::Physics::InteractionPtr(ptr)
    {
      agxAssertN(!ptr || ptr.isInstanceOf(ContactModel::instance()),
        "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
        EntityPtr::getModel()->fullPath().c_str(), ContactModel::instance()->fullPath().c_str());
    }

    AGX_FORCE_INLINE ContactPtr::ContactPtr(const agxData::EntityInstance& instance) : agx::Physics::InteractionPtr(instance)
    {
      agxAssertN(!instance || instance.isInstanceOf(ContactModel::instance()),
        "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
        EntityPtr::getModel()->fullPath().c_str(), ContactModel::instance()->fullPath().c_str());
    }

    AGX_FORCE_INLINE ContactPtr& ContactPtr::operator= (const agxData::EntityPtr& ptr)
    {
      agxData::EntityPtr::operator= (ptr);
      agxAssertN(!ptr || ptr.isInstanceOf(ContactModel::instance()),
        "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
        EntityPtr::getModel()->fullPath().c_str(), ContactModel::instance()->fullPath().c_str());
      return *this;
    }

    AGX_FORCE_INLINE ContactPtr& ContactPtr::operator= (const agxData::EntityInstance& instance)
    {
      agxData::EntityPtr::operator= (instance);
      agxAssertN(!instance || instance.isInstanceOf(ContactModel::instance()),
        "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
        EntityPtr::getModel()->fullPath().c_str(), ContactModel::instance()->fullPath().c_str());
      return *this;
    }

    inline ContactInstance ContactPtr::instance() { return agxData::EntityPtr::instance(); }
    inline const ContactInstance ContactPtr::instance() const { return agxData::EntityPtr::instance(); }
    AGX_FORCE_INLINE ContactSemantics* ContactPtr::operator->() { return (ContactSemantics* )this; }
    AGX_FORCE_INLINE const ContactSemantics* ContactPtr::operator->() const { return (const ContactSemantics* )this; }
    AGX_FORCE_INLINE ContactData* ContactPtr::getData() { return static_cast<ContactData* >(agxData::EntityPtr::getData()); }
    AGX_FORCE_INLINE const ContactData* ContactPtr::getData() const { return static_cast<const ContactData* >(agxData::EntityPtr::getData()); }

    AGX_FORCE_INLINE agx::UInt32& ContactPtr::constraintIndex() { verifyIndex(); return getData()->constraintIndex[calculateIndex()]; }
    AGX_FORCE_INLINE agx::UInt32 const& ContactPtr::constraintIndex() const { verifyIndex(); return getData()->constraintIndex[calculateIndex()]; }

    AGX_FORCE_INLINE agx::Physics::ContactMaterialPtr& ContactPtr::material() { verifyIndex(); return getData()->material[calculateIndex()]; }
    AGX_FORCE_INLINE agx::Physics::ContactMaterialPtr const& ContactPtr::material() const { verifyIndex(); return getData()->material[calculateIndex()]; }

    AGX_FORCE_INLINE agx::UInt32& ContactPtr::contactZoneIndex() { verifyIndex(); return getData()->contactZoneIndex[calculateIndex()]; }
    AGX_FORCE_INLINE agx::UInt32 const& ContactPtr::contactZoneIndex() const { verifyIndex(); return getData()->contactZoneIndex[calculateIndex()]; }

    AGX_FORCE_INLINE agx::UInt16& ContactPtr::contactZoneSlot() { verifyIndex(); return getData()->contactZoneSlot[calculateIndex()]; }
    AGX_FORCE_INLINE agx::UInt16 const& ContactPtr::contactZoneSlot() const { verifyIndex(); return getData()->contactZoneSlot[calculateIndex()]; }

    //-----------------------------------------------------------------------------------------------------
    AGX_FORCE_INLINE ContactInstance::ContactInstance() {}
    AGX_FORCE_INLINE ContactInstance::ContactInstance(ContactData* data, agx::Index index) : agx::Physics::InteractionInstance(data, index) {}
    AGX_FORCE_INLINE ContactInstance::ContactInstance(agxData::EntityStorage* storage, agx::Index index) : agx::Physics::InteractionInstance(storage, index) {}
    AGX_FORCE_INLINE ContactInstance::ContactInstance(const agxData::EntityInstance& other) : agx::Physics::InteractionInstance(other)
    {
      agxAssertN(!other || other.isInstanceOf(ContactModel::instance()),
        "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
        EntityInstance::getModel()->fullPath().c_str(), ContactModel::instance()->fullPath().c_str());
    }

    AGX_FORCE_INLINE ContactInstance::ContactInstance(const agxData::EntityPtr& ptr) : agx::Physics::InteractionInstance(ptr)
    {
      agxAssertN(!ptr || ptr.isInstanceOf(ContactModel::instance()),
        "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
        EntityInstance::getModel()->fullPath().c_str(), ContactModel::instance()->fullPath().c_str());
    }


    AGX_FORCE_INLINE ContactData* ContactInstance::getData() { return static_cast<ContactData* >(agxData::EntityInstance::getData()); }
    AGX_FORCE_INLINE const ContactData* ContactInstance::getData() const { return static_cast<const ContactData* >(agxData::EntityInstance::getData()); }

    AGX_FORCE_INLINE agx::UInt32& ContactInstance::constraintIndex() { verifyIndex(); return getData()->constraintIndex[getIndex()]; }
    AGX_FORCE_INLINE agx::UInt32 const& ContactInstance::constraintIndex() const { verifyIndex(); return getData()->constraintIndex[getIndex()]; }

    AGX_FORCE_INLINE agx::Physics::ContactMaterialPtr& ContactInstance::material() { verifyIndex(); return getData()->material[getIndex()]; }
    AGX_FORCE_INLINE agx::Physics::ContactMaterialPtr const& ContactInstance::material() const { verifyIndex(); return getData()->material[getIndex()]; }

    AGX_FORCE_INLINE agx::UInt32& ContactInstance::contactZoneIndex() { verifyIndex(); return getData()->contactZoneIndex[getIndex()]; }
    AGX_FORCE_INLINE agx::UInt32 const& ContactInstance::contactZoneIndex() const { verifyIndex(); return getData()->contactZoneIndex[getIndex()]; }

    AGX_FORCE_INLINE agx::UInt16& ContactInstance::contactZoneSlot() { verifyIndex(); return getData()->contactZoneSlot[getIndex()]; }
    AGX_FORCE_INLINE agx::UInt16 const& ContactInstance::contactZoneSlot() const { verifyIndex(); return getData()->contactZoneSlot[getIndex()]; }

    //-----------------------------------------------------------------------------------------------------
    AGX_FORCE_INLINE ContactSemantics::ContactSemantics() {}
    //-----------------------------------------------------------------------------------------------------
    DOXYGEN_END_INTERNAL_BLOCK()
  }
}

AGX_TYPE_BINDING(agx::Physics::ContactPtr, "Physics.ContactPtr")
AGX_TYPE_BINDING(agx::Physics::ContactInstance, "Physics.ContactInstance")

#ifdef _MSC_VER
# pragma warning(pop)
#endif

#undef AGX_ENTITY_WRAPPER
#undef AGX_ENTITY_NAMESPACE
#endif

