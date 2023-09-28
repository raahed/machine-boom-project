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

#ifndef GENERATED_AGX_PHYSICS_CONTACTCONSTRAINT_H_PLUGIN
#define GENERATED_AGX_PHYSICS_CONTACTCONSTRAINT_H_PLUGIN

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
#include <agx/Physics/BinaryConstraintEntity.h>
#include <agx/Integer.h>
#include <agx/IndexRange.h>


namespace agx
{
  namespace Physics
  {

    class ContactConstraintModel;
    class ContactConstraintData;
    class ContactConstraintPtr;
    class ContactConstraintInstance;
    class ContactConstraintSemantics;


    AGX_DECLARE_POINTER_TYPES(ContactConstraintModel);

    /** 
    Abstract description of the data attributes for the Physics.ContactConstraint entity.
    */ 
    class AGXPHYSICS_EXPORT ContactConstraintModel : public agx::Physics::BinaryConstraintModel
    {
    public:
      typedef ContactConstraintPtr PtrT;

      ContactConstraintModel(const agx::String& name = "ContactConstraint");

      /// \return The entity model singleton.
      static ContactConstraintModel* instance();

      /// Create and return a pointer to a new instance in the default storage for this entity model.
      static ContactConstraintPtr createInstance();

      /// \return The default storage for this entity model.
      static agxData::EntityStorage* defaultStorage();

      /// This is part of internal cleanup and should not be called by users
      virtual void shutdownCleanup() override;



      /* Attributes */
      static agxData::ScalarAttributeT< agx::UInt32 >* contactIndexAttribute;
      static agxData::ScalarAttributeT< agx::UInt32 >* materialIndexAttribute;
      static agxData::ScalarAttributeT< agx::IndexRange >* contactPointRangeAttribute;

    protected:
      virtual ~ContactConstraintModel();
      virtual agxData::EntityData* createData(agxData::EntityStorage* storage) override;
      virtual void configure(agx::TiXmlElement* eEntity) override;
      virtual void initAttributeAccessors() override;
      virtual void construct(agxData::EntityPtr instance) override;
      void construct(agx::Physics::ContactConstraintPtr contactConstraint);
    };


    DOXYGEN_START_INTERNAL_BLOCK()
    #ifndef AGX_PHYSICS_CONTACTCONSTRAINT_DATA_SET_OVERRIDE
    #define AGX_PHYSICS_CONTACTCONSTRAINT_DATA_SET
    class AGXPHYSICS_EXPORT ContactConstraintData : public agx::Physics::BinaryConstraintData
    {
    public:
      ContactConstraintInstance operator[] (size_t index);

    public:
      agxData::Array< ContactConstraintPtr >& instance;
      agxData::Array< agx::UInt32 > contactIndex;
      agxData::Array< agx::UInt32 > materialIndex;
      agxData::Array< agx::IndexRange > contactPointRange;

    public:
      typedef agx::UInt32 contactIndexType;
      typedef agx::UInt32 materialIndexType;
      typedef agx::IndexRange contactPointRangeType;

    public:
      ContactConstraintData(agxData::EntityStorage* storage);
      ContactConstraintData();

    protected:
      virtual ~ContactConstraintData() {}
      virtual void setNumElements(agx::Index numElements) override;

    private:
      ContactConstraintData& operator= (const ContactConstraintData&) { return *this; }

    };
    #endif
    DOXYGEN_END_INTERNAL_BLOCK()


    DOXYGEN_START_INTERNAL_BLOCK()
    class AGXPHYSICS_EXPORT ContactConstraintSemantics : public agx::Physics::BinaryConstraintSemantics
    {
    public:

      // Automatic getters
      agx::UInt32 const& getContactIndex() const;
      agx::UInt32 const& getMaterialIndex() const;
      agx::IndexRange const& getContactPointRange() const;

      // Semantics defined by explicit kernels

      // Automatic setters
      void setContactIndex(agx::UInt32 const& value);
      void setMaterialIndex(agx::UInt32 const& value);
      void setContactPointRange(agx::IndexRange const& value);


    protected:
      friend class ContactConstraintPtr;
      friend class ContactConstraintInstance;
      ContactConstraintSemantics();
    };
    DOXYGEN_END_INTERNAL_BLOCK()


    /**
    Pointer to a entity instance of type Physics.ContactConstraint
    */
    class CALLABLE ContactConstraintPtr : public agx::Physics::BinaryConstraintPtr
    {
    public:
      typedef ContactConstraintModel ModelType;
      typedef ContactConstraintData DataType;
      typedef ContactConstraintInstance InstanceType;

    public:
      AGXPHYSICS_EXPORT ContactConstraintPtr();
      AGXPHYSICS_EXPORT ContactConstraintPtr(agxData::EntityStorage* storage, agx::Index id);
      AGXPHYSICS_EXPORT ContactConstraintPtr(const agxData::EntityPtr& ptr);
      AGXPHYSICS_EXPORT ContactConstraintPtr(const agxData::EntityInstance& instance);
      AGXPHYSICS_EXPORT ContactConstraintPtr& operator= (const agxData::EntityPtr& ptr);
      AGXPHYSICS_EXPORT ContactConstraintPtr& operator= (const agxData::EntityInstance& instance);
      AGXPHYSICS_EXPORT ContactConstraintInstance instance();
      AGXPHYSICS_EXPORT const ContactConstraintInstance instance() const;

      AGXPHYSICS_EXPORT ContactConstraintSemantics* operator->();
      AGXPHYSICS_EXPORT const ContactConstraintSemantics* operator->() const;

      ContactConstraintData* getData();
      const ContactConstraintData* getData() const;


      /// \return reference to the contactIndex attribute
      AGXPHYSICS_EXPORT agx::UInt32& contactIndex();
      /// \return const reference to the contactIndex attribute
      AGXPHYSICS_EXPORT agx::UInt32 const& contactIndex() const;

      /// \return reference to the materialIndex attribute
      AGXPHYSICS_EXPORT agx::UInt32& materialIndex();
      /// \return const reference to the materialIndex attribute
      AGXPHYSICS_EXPORT agx::UInt32 const& materialIndex() const;

      /// \return reference to the contactPointRange attribute
      AGXPHYSICS_EXPORT agx::IndexRange& contactPointRange();
      /// \return const reference to the contactPointRange attribute
      AGXPHYSICS_EXPORT agx::IndexRange const& contactPointRange() const;

      DOXYGEN_START_INTERNAL_BLOCK()
      // Only used by tolua!
      #ifdef AGX_GENERATED_TOLUA_WRAPPER
      inline agx::UInt32 const& getContactIndex() const { return (*this)->getContactIndex(); }
      inline agx::UInt32 const& getMaterialIndex() const { return (*this)->getMaterialIndex(); }
      inline agx::IndexRange const& getContactPointRange() const { return (*this)->getContactPointRange(); }


      inline void setContactIndex(agx::UInt32 const& value) { (*this)->setContactIndex(value); }
      inline void setMaterialIndex(agx::UInt32 const& value) { (*this)->setMaterialIndex(value); }
      inline void setContactPointRange(agx::IndexRange const& value) { (*this)->setContactPointRange(value); }
      #endif
      DOXYGEN_END_INTERNAL_BLOCK()

    };


    DOXYGEN_START_INTERNAL_BLOCK()
    class AGXPHYSICS_EXPORT ContactConstraintInstance : public agx::Physics::BinaryConstraintInstance
    {
    public:
      ContactConstraintInstance();
      ContactConstraintInstance(ContactConstraintData* data, agx::Index index);
      ContactConstraintInstance(agxData::EntityStorage *storage, agx::Index index);
      ContactConstraintInstance(const agxData::EntityInstance& other);
      ContactConstraintInstance(const agxData::EntityPtr& ptr);

      ContactConstraintData* getData();
      const ContactConstraintData* getData() const;

    public:
      /// \return reference to the contactIndex attribute
      agx::UInt32& contactIndex();
      /// \return const reference to the contactIndex attribute
      agx::UInt32 const& contactIndex() const;

      /// \return reference to the materialIndex attribute
      agx::UInt32& materialIndex();
      /// \return const reference to the materialIndex attribute
      agx::UInt32 const& materialIndex() const;

      /// \return reference to the contactPointRange attribute
      agx::IndexRange& contactPointRange();
      /// \return const reference to the contactPointRange attribute
      agx::IndexRange const& contactPointRange() const;

    };
    DOXYGEN_END_INTERNAL_BLOCK()



    typedef agx::VectorPOD<ContactConstraintPtr> ContactConstraintPtrVector;
    typedef agxData::Array<ContactConstraintPtr> ContactConstraintPtrArray;



    DOXYGEN_START_INTERNAL_BLOCK()
    /* Implementation */
    //-----------------------------------------------------------------------------------------------------
    //-----------------------------------------------------------------------------------------------------
    inline ContactConstraintInstance agx::Physics::ContactConstraintData::operator[] (size_t index) { return ContactConstraintInstance(this, (agx::Index)index); }
    //-----------------------------------------------------------------------------------------------------
    AGX_FORCE_INLINE ContactConstraintPtr::ContactConstraintPtr() {}
    AGX_FORCE_INLINE ContactConstraintPtr::ContactConstraintPtr(agxData::EntityStorage* storage, agx::Index id) : agx::Physics::BinaryConstraintPtr(storage, id) {}
    AGX_FORCE_INLINE ContactConstraintPtr::ContactConstraintPtr(const agxData::EntityPtr& ptr) : agx::Physics::BinaryConstraintPtr(ptr)
    {
      agxAssertN(!ptr || ptr.isInstanceOf(ContactConstraintModel::instance()),
        "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
        EntityPtr::getModel()->fullPath().c_str(), ContactConstraintModel::instance()->fullPath().c_str());
    }

    AGX_FORCE_INLINE ContactConstraintPtr::ContactConstraintPtr(const agxData::EntityInstance& instance) : agx::Physics::BinaryConstraintPtr(instance)
    {
      agxAssertN(!instance || instance.isInstanceOf(ContactConstraintModel::instance()),
        "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
        EntityPtr::getModel()->fullPath().c_str(), ContactConstraintModel::instance()->fullPath().c_str());
    }

    AGX_FORCE_INLINE ContactConstraintPtr& ContactConstraintPtr::operator= (const agxData::EntityPtr& ptr)
    {
      agxData::EntityPtr::operator= (ptr);
      agxAssertN(!ptr || ptr.isInstanceOf(ContactConstraintModel::instance()),
        "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
        EntityPtr::getModel()->fullPath().c_str(), ContactConstraintModel::instance()->fullPath().c_str());
      return *this;
    }

    AGX_FORCE_INLINE ContactConstraintPtr& ContactConstraintPtr::operator= (const agxData::EntityInstance& instance)
    {
      agxData::EntityPtr::operator= (instance);
      agxAssertN(!instance || instance.isInstanceOf(ContactConstraintModel::instance()),
        "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
        EntityPtr::getModel()->fullPath().c_str(), ContactConstraintModel::instance()->fullPath().c_str());
      return *this;
    }

    inline ContactConstraintInstance ContactConstraintPtr::instance() { return agxData::EntityPtr::instance(); }
    inline const ContactConstraintInstance ContactConstraintPtr::instance() const { return agxData::EntityPtr::instance(); }
    AGX_FORCE_INLINE ContactConstraintSemantics* ContactConstraintPtr::operator->() { return (ContactConstraintSemantics* )this; }
    AGX_FORCE_INLINE const ContactConstraintSemantics* ContactConstraintPtr::operator->() const { return (const ContactConstraintSemantics* )this; }
    AGX_FORCE_INLINE ContactConstraintData* ContactConstraintPtr::getData() { return static_cast<ContactConstraintData* >(agxData::EntityPtr::getData()); }
    AGX_FORCE_INLINE const ContactConstraintData* ContactConstraintPtr::getData() const { return static_cast<const ContactConstraintData* >(agxData::EntityPtr::getData()); }

    AGX_FORCE_INLINE agx::UInt32& ContactConstraintPtr::contactIndex() { verifyIndex(); return getData()->contactIndex[calculateIndex()]; }
    AGX_FORCE_INLINE agx::UInt32 const& ContactConstraintPtr::contactIndex() const { verifyIndex(); return getData()->contactIndex[calculateIndex()]; }

    AGX_FORCE_INLINE agx::UInt32& ContactConstraintPtr::materialIndex() { verifyIndex(); return getData()->materialIndex[calculateIndex()]; }
    AGX_FORCE_INLINE agx::UInt32 const& ContactConstraintPtr::materialIndex() const { verifyIndex(); return getData()->materialIndex[calculateIndex()]; }

    AGX_FORCE_INLINE agx::IndexRange& ContactConstraintPtr::contactPointRange() { verifyIndex(); return getData()->contactPointRange[calculateIndex()]; }
    AGX_FORCE_INLINE agx::IndexRange const& ContactConstraintPtr::contactPointRange() const { verifyIndex(); return getData()->contactPointRange[calculateIndex()]; }

    //-----------------------------------------------------------------------------------------------------
    AGX_FORCE_INLINE ContactConstraintInstance::ContactConstraintInstance() {}
    AGX_FORCE_INLINE ContactConstraintInstance::ContactConstraintInstance(ContactConstraintData* data, agx::Index index) : agx::Physics::BinaryConstraintInstance(data, index) {}
    AGX_FORCE_INLINE ContactConstraintInstance::ContactConstraintInstance(agxData::EntityStorage* storage, agx::Index index) : agx::Physics::BinaryConstraintInstance(storage, index) {}
    AGX_FORCE_INLINE ContactConstraintInstance::ContactConstraintInstance(const agxData::EntityInstance& other) : agx::Physics::BinaryConstraintInstance(other)
    {
      agxAssertN(!other || other.isInstanceOf(ContactConstraintModel::instance()),
        "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
        EntityInstance::getModel()->fullPath().c_str(), ContactConstraintModel::instance()->fullPath().c_str());
    }

    AGX_FORCE_INLINE ContactConstraintInstance::ContactConstraintInstance(const agxData::EntityPtr& ptr) : agx::Physics::BinaryConstraintInstance(ptr)
    {
      agxAssertN(!ptr || ptr.isInstanceOf(ContactConstraintModel::instance()),
        "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
        EntityInstance::getModel()->fullPath().c_str(), ContactConstraintModel::instance()->fullPath().c_str());
    }


    AGX_FORCE_INLINE ContactConstraintData* ContactConstraintInstance::getData() { return static_cast<ContactConstraintData* >(agxData::EntityInstance::getData()); }
    AGX_FORCE_INLINE const ContactConstraintData* ContactConstraintInstance::getData() const { return static_cast<const ContactConstraintData* >(agxData::EntityInstance::getData()); }

    AGX_FORCE_INLINE agx::UInt32& ContactConstraintInstance::contactIndex() { verifyIndex(); return getData()->contactIndex[getIndex()]; }
    AGX_FORCE_INLINE agx::UInt32 const& ContactConstraintInstance::contactIndex() const { verifyIndex(); return getData()->contactIndex[getIndex()]; }

    AGX_FORCE_INLINE agx::UInt32& ContactConstraintInstance::materialIndex() { verifyIndex(); return getData()->materialIndex[getIndex()]; }
    AGX_FORCE_INLINE agx::UInt32 const& ContactConstraintInstance::materialIndex() const { verifyIndex(); return getData()->materialIndex[getIndex()]; }

    AGX_FORCE_INLINE agx::IndexRange& ContactConstraintInstance::contactPointRange() { verifyIndex(); return getData()->contactPointRange[getIndex()]; }
    AGX_FORCE_INLINE agx::IndexRange const& ContactConstraintInstance::contactPointRange() const { verifyIndex(); return getData()->contactPointRange[getIndex()]; }

    //-----------------------------------------------------------------------------------------------------
    AGX_FORCE_INLINE ContactConstraintSemantics::ContactConstraintSemantics() {}
    //-----------------------------------------------------------------------------------------------------
    DOXYGEN_END_INTERNAL_BLOCK()
  }
}

AGX_TYPE_BINDING(agx::Physics::ContactConstraintPtr, "Physics.ContactConstraintPtr")
AGX_TYPE_BINDING(agx::Physics::ContactConstraintInstance, "Physics.ContactConstraintInstance")

#ifdef _MSC_VER
# pragma warning(pop)
#endif

#undef AGX_ENTITY_WRAPPER
#undef AGX_ENTITY_NAMESPACE
#endif

