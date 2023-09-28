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

#ifndef GENERATED_AGX_PHYSICS_BODYCONTACT_H_PLUGIN
#define GENERATED_AGX_PHYSICS_BODYCONTACT_H_PLUGIN

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
#include <agx/Integer.h>


namespace agx
{
  namespace Physics
  {

    class BodyContactModel;
    class BodyContactData;
    class BodyContactPtr;
    class BodyContactInstance;
    class BodyContactSemantics;


    AGX_DECLARE_POINTER_TYPES(BodyContactModel);

    /** 
    Abstract description of the data attributes for the Physics.BodyContact entity.
    */ 
    class AGXPHYSICS_EXPORT BodyContactModel : public agxData::EntityModel
    {
    public:
      typedef BodyContactPtr PtrT;

      BodyContactModel(const agx::String& name = "BodyContact");

      /// \return The entity model singleton.
      static BodyContactModel* instance();

      /// Create and return a pointer to a new instance in the default storage for this entity model.
      static BodyContactPtr createInstance();

      /// \return The default storage for this entity model.
      static agxData::EntityStorage* defaultStorage();

      /// This is part of internal cleanup and should not be called by users
      virtual void shutdownCleanup() override;



      /* Attributes */
      static agxData::ScalarAttributeT< agx::Int >* body1Attribute;
      static agxData::ScalarAttributeT< agx::Int >* body2Attribute;
      static agxData::ScalarAttributeT< agx::Int >* contactListOffsetAttribute;

    protected:
      virtual ~BodyContactModel();
      virtual agxData::EntityData* createData(agxData::EntityStorage* storage) override;
      virtual void configure(agx::TiXmlElement* eEntity) override;
      virtual void initAttributeAccessors() override;
      virtual void construct(agxData::EntityPtr instance) override;
      void construct(agx::Physics::BodyContactPtr bodyContact);
    };


    DOXYGEN_START_INTERNAL_BLOCK()
    #ifndef AGX_PHYSICS_BODYCONTACT_DATA_SET_OVERRIDE
    #define AGX_PHYSICS_BODYCONTACT_DATA_SET
    class AGXPHYSICS_EXPORT BodyContactData : public agxData::EntityData
    {
    public:
      BodyContactInstance operator[] (size_t index);

    public:
      agxData::Array< BodyContactPtr >& instance;
      agxData::Array< agx::Int > body1;
      agxData::Array< agx::Int > body2;
      agxData::Array< agx::Int > contactListOffset;

    public:
      typedef agx::Int body1Type;
      typedef agx::Int body2Type;
      typedef agx::Int contactListOffsetType;

    public:
      BodyContactData(agxData::EntityStorage* storage);
      BodyContactData();

    protected:
      virtual ~BodyContactData() {}
      virtual void setNumElements(agx::Index numElements) override;

    private:
      BodyContactData& operator= (const BodyContactData&) { return *this; }

    };
    #endif
    DOXYGEN_END_INTERNAL_BLOCK()


    DOXYGEN_START_INTERNAL_BLOCK()
    class AGXPHYSICS_EXPORT BodyContactSemantics : protected agxData::EntityPtr
    {
    public:

      // Automatic getters
      agx::Int const& getBody1() const;
      agx::Int const& getBody2() const;
      agx::Int const& getContactListOffset() const;

      // Semantics defined by explicit kernels

      // Automatic setters
      void setBody1(agx::Int const& value);
      void setBody2(agx::Int const& value);
      void setContactListOffset(agx::Int const& value);


    protected:
      friend class BodyContactPtr;
      friend class BodyContactInstance;
      BodyContactSemantics();
    };
    DOXYGEN_END_INTERNAL_BLOCK()


    /**
    Pointer to a entity instance of type Physics.BodyContact
    */
    class CALLABLE BodyContactPtr : public agxData::EntityPtr
    {
    public:
      typedef BodyContactModel ModelType;
      typedef BodyContactData DataType;
      typedef BodyContactInstance InstanceType;

    public:
      AGXPHYSICS_EXPORT BodyContactPtr();
      AGXPHYSICS_EXPORT BodyContactPtr(agxData::EntityStorage* storage, agx::Index id);
      AGXPHYSICS_EXPORT BodyContactPtr(const agxData::EntityPtr& ptr);
      AGXPHYSICS_EXPORT BodyContactPtr(const agxData::EntityInstance& instance);
      AGXPHYSICS_EXPORT BodyContactPtr& operator= (const agxData::EntityPtr& ptr);
      AGXPHYSICS_EXPORT BodyContactPtr& operator= (const agxData::EntityInstance& instance);
      AGXPHYSICS_EXPORT BodyContactInstance instance();
      AGXPHYSICS_EXPORT const BodyContactInstance instance() const;

      AGXPHYSICS_EXPORT BodyContactSemantics* operator->();
      AGXPHYSICS_EXPORT const BodyContactSemantics* operator->() const;

      BodyContactData* getData();
      const BodyContactData* getData() const;


      /// \return reference to the body1 attribute
      AGXPHYSICS_EXPORT agx::Int& body1();
      /// \return const reference to the body1 attribute
      AGXPHYSICS_EXPORT agx::Int const& body1() const;

      /// \return reference to the body2 attribute
      AGXPHYSICS_EXPORT agx::Int& body2();
      /// \return const reference to the body2 attribute
      AGXPHYSICS_EXPORT agx::Int const& body2() const;

      /// \return reference to the contactListOffset attribute
      AGXPHYSICS_EXPORT agx::Int& contactListOffset();
      /// \return const reference to the contactListOffset attribute
      AGXPHYSICS_EXPORT agx::Int const& contactListOffset() const;

      DOXYGEN_START_INTERNAL_BLOCK()
      // Only used by tolua!
      #ifdef AGX_GENERATED_TOLUA_WRAPPER
      inline agx::Int const& getBody1() const { return (*this)->getBody1(); }
      inline agx::Int const& getBody2() const { return (*this)->getBody2(); }
      inline agx::Int const& getContactListOffset() const { return (*this)->getContactListOffset(); }


      inline void setBody1(agx::Int const& value) { (*this)->setBody1(value); }
      inline void setBody2(agx::Int const& value) { (*this)->setBody2(value); }
      inline void setContactListOffset(agx::Int const& value) { (*this)->setContactListOffset(value); }
      #endif
      DOXYGEN_END_INTERNAL_BLOCK()

    };


    DOXYGEN_START_INTERNAL_BLOCK()
    class AGXPHYSICS_EXPORT BodyContactInstance : public agxData::EntityInstance
    {
    public:
      BodyContactInstance();
      BodyContactInstance(BodyContactData* data, agx::Index index);
      BodyContactInstance(agxData::EntityStorage *storage, agx::Index index);
      BodyContactInstance(const agxData::EntityInstance& other);
      BodyContactInstance(const agxData::EntityPtr& ptr);

      BodyContactData* getData();
      const BodyContactData* getData() const;

    public:
      /// \return reference to the body1 attribute
      agx::Int& body1();
      /// \return const reference to the body1 attribute
      agx::Int const& body1() const;

      /// \return reference to the body2 attribute
      agx::Int& body2();
      /// \return const reference to the body2 attribute
      agx::Int const& body2() const;

      /// \return reference to the contactListOffset attribute
      agx::Int& contactListOffset();
      /// \return const reference to the contactListOffset attribute
      agx::Int const& contactListOffset() const;

    };
    DOXYGEN_END_INTERNAL_BLOCK()



    typedef agx::VectorPOD<BodyContactPtr> BodyContactPtrVector;
    typedef agxData::Array<BodyContactPtr> BodyContactPtrArray;



    DOXYGEN_START_INTERNAL_BLOCK()
    /* Implementation */
    //-----------------------------------------------------------------------------------------------------
    //-----------------------------------------------------------------------------------------------------
    inline BodyContactInstance agx::Physics::BodyContactData::operator[] (size_t index) { return BodyContactInstance(this, (agx::Index)index); }
    //-----------------------------------------------------------------------------------------------------
    AGX_FORCE_INLINE BodyContactPtr::BodyContactPtr() {}
    AGX_FORCE_INLINE BodyContactPtr::BodyContactPtr(agxData::EntityStorage* storage, agx::Index id) : agxData::EntityPtr(storage, id) {}
    AGX_FORCE_INLINE BodyContactPtr::BodyContactPtr(const agxData::EntityPtr& ptr) : agxData::EntityPtr(ptr)
    {
      agxAssertN(!ptr || ptr.isInstanceOf(BodyContactModel::instance()),
        "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
        EntityPtr::getModel()->fullPath().c_str(), BodyContactModel::instance()->fullPath().c_str());
    }

    AGX_FORCE_INLINE BodyContactPtr::BodyContactPtr(const agxData::EntityInstance& instance) : agxData::EntityPtr(instance)
    {
      agxAssertN(!instance || instance.isInstanceOf(BodyContactModel::instance()),
        "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
        EntityPtr::getModel()->fullPath().c_str(), BodyContactModel::instance()->fullPath().c_str());
    }

    AGX_FORCE_INLINE BodyContactPtr& BodyContactPtr::operator= (const agxData::EntityPtr& ptr)
    {
      agxData::EntityPtr::operator= (ptr);
      agxAssertN(!ptr || ptr.isInstanceOf(BodyContactModel::instance()),
        "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
        EntityPtr::getModel()->fullPath().c_str(), BodyContactModel::instance()->fullPath().c_str());
      return *this;
    }

    AGX_FORCE_INLINE BodyContactPtr& BodyContactPtr::operator= (const agxData::EntityInstance& instance)
    {
      agxData::EntityPtr::operator= (instance);
      agxAssertN(!instance || instance.isInstanceOf(BodyContactModel::instance()),
        "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
        EntityPtr::getModel()->fullPath().c_str(), BodyContactModel::instance()->fullPath().c_str());
      return *this;
    }

    inline BodyContactInstance BodyContactPtr::instance() { return agxData::EntityPtr::instance(); }
    inline const BodyContactInstance BodyContactPtr::instance() const { return agxData::EntityPtr::instance(); }
    AGX_FORCE_INLINE BodyContactSemantics* BodyContactPtr::operator->() { return (BodyContactSemantics* )this; }
    AGX_FORCE_INLINE const BodyContactSemantics* BodyContactPtr::operator->() const { return (const BodyContactSemantics* )this; }
    AGX_FORCE_INLINE BodyContactData* BodyContactPtr::getData() { return static_cast<BodyContactData* >(agxData::EntityPtr::getData()); }
    AGX_FORCE_INLINE const BodyContactData* BodyContactPtr::getData() const { return static_cast<const BodyContactData* >(agxData::EntityPtr::getData()); }

    AGX_FORCE_INLINE agx::Int& BodyContactPtr::body1() { verifyIndex(); return getData()->body1[calculateIndex()]; }
    AGX_FORCE_INLINE agx::Int const& BodyContactPtr::body1() const { verifyIndex(); return getData()->body1[calculateIndex()]; }

    AGX_FORCE_INLINE agx::Int& BodyContactPtr::body2() { verifyIndex(); return getData()->body2[calculateIndex()]; }
    AGX_FORCE_INLINE agx::Int const& BodyContactPtr::body2() const { verifyIndex(); return getData()->body2[calculateIndex()]; }

    AGX_FORCE_INLINE agx::Int& BodyContactPtr::contactListOffset() { verifyIndex(); return getData()->contactListOffset[calculateIndex()]; }
    AGX_FORCE_INLINE agx::Int const& BodyContactPtr::contactListOffset() const { verifyIndex(); return getData()->contactListOffset[calculateIndex()]; }

    //-----------------------------------------------------------------------------------------------------
    AGX_FORCE_INLINE BodyContactInstance::BodyContactInstance() {}
    AGX_FORCE_INLINE BodyContactInstance::BodyContactInstance(BodyContactData* data, agx::Index index) : agxData::EntityInstance(data, index) {}
    AGX_FORCE_INLINE BodyContactInstance::BodyContactInstance(agxData::EntityStorage* storage, agx::Index index) : agxData::EntityInstance(storage, index) {}
    AGX_FORCE_INLINE BodyContactInstance::BodyContactInstance(const agxData::EntityInstance& other) : agxData::EntityInstance(other)
    {
      agxAssertN(!other || other.isInstanceOf(BodyContactModel::instance()),
        "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
        EntityInstance::getModel()->fullPath().c_str(), BodyContactModel::instance()->fullPath().c_str());
    }

    AGX_FORCE_INLINE BodyContactInstance::BodyContactInstance(const agxData::EntityPtr& ptr) : agxData::EntityInstance(ptr)
    {
      agxAssertN(!ptr || ptr.isInstanceOf(BodyContactModel::instance()),
        "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
        EntityInstance::getModel()->fullPath().c_str(), BodyContactModel::instance()->fullPath().c_str());
    }


    AGX_FORCE_INLINE BodyContactData* BodyContactInstance::getData() { return static_cast<BodyContactData* >(agxData::EntityInstance::getData()); }
    AGX_FORCE_INLINE const BodyContactData* BodyContactInstance::getData() const { return static_cast<const BodyContactData* >(agxData::EntityInstance::getData()); }

    AGX_FORCE_INLINE agx::Int& BodyContactInstance::body1() { verifyIndex(); return getData()->body1[getIndex()]; }
    AGX_FORCE_INLINE agx::Int const& BodyContactInstance::body1() const { verifyIndex(); return getData()->body1[getIndex()]; }

    AGX_FORCE_INLINE agx::Int& BodyContactInstance::body2() { verifyIndex(); return getData()->body2[getIndex()]; }
    AGX_FORCE_INLINE agx::Int const& BodyContactInstance::body2() const { verifyIndex(); return getData()->body2[getIndex()]; }

    AGX_FORCE_INLINE agx::Int& BodyContactInstance::contactListOffset() { verifyIndex(); return getData()->contactListOffset[getIndex()]; }
    AGX_FORCE_INLINE agx::Int const& BodyContactInstance::contactListOffset() const { verifyIndex(); return getData()->contactListOffset[getIndex()]; }

    //-----------------------------------------------------------------------------------------------------
    AGX_FORCE_INLINE BodyContactSemantics::BodyContactSemantics() {}
    //-----------------------------------------------------------------------------------------------------
    DOXYGEN_END_INTERNAL_BLOCK()
  }
}

AGX_TYPE_BINDING(agx::Physics::BodyContactPtr, "Physics.BodyContactPtr")
AGX_TYPE_BINDING(agx::Physics::BodyContactInstance, "Physics.BodyContactInstance")

#ifdef _MSC_VER
# pragma warning(pop)
#endif

#undef AGX_ENTITY_WRAPPER
#undef AGX_ENTITY_NAMESPACE
#endif

