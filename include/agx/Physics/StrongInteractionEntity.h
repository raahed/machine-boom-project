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

#ifndef GENERATED_AGX_PHYSICS_STRONGINTERACTION_H_PLUGIN
#define GENERATED_AGX_PHYSICS_STRONGINTERACTION_H_PLUGIN

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
namespace agx { class StrongInteraction; }


namespace agx
{
  namespace Physics
  {

    class StrongInteractionModel;
    class StrongInteractionData;
    class StrongInteractionPtr;
    class StrongInteractionInstance;
    class StrongInteractionSemantics;


    AGX_DECLARE_POINTER_TYPES(StrongInteractionModel);

    /** 
    Abstract description of the data attributes for the Physics.StrongInteraction entity.
    */ 
    class AGXPHYSICS_EXPORT StrongInteractionModel : public agx::Physics::InteractionModel
    {
    public:
      typedef StrongInteractionPtr PtrT;

      StrongInteractionModel(const agx::String& name = "StrongInteraction");

      /// \return The entity model singleton.
      static StrongInteractionModel* instance();

      /// Create and return a pointer to a new instance in the default storage for this entity model.
      static StrongInteractionPtr createInstance();

      /// \return The default storage for this entity model.
      static agxData::EntityStorage* defaultStorage();

      /// This is part of internal cleanup and should not be called by users
      virtual void shutdownCleanup() override;



      /* Attributes */
      static agxData::PointerAttributeT< agx::StrongInteraction*>* modelAttribute;
      static agxData::ScalarAttributeT< agx::UInt >* body1Attribute;
      static agxData::ScalarAttributeT< agx::UInt >* body2Attribute;

    protected:
      virtual ~StrongInteractionModel();
      virtual agxData::EntityData* createData(agxData::EntityStorage* storage) override;
      virtual void configure(agx::TiXmlElement* eEntity) override;
      virtual void initAttributeAccessors() override;
      virtual void construct(agxData::EntityPtr instance) override;
      void construct(agx::Physics::StrongInteractionPtr strongInteraction);
    };


    DOXYGEN_START_INTERNAL_BLOCK()
    #ifndef AGX_PHYSICS_STRONGINTERACTION_DATA_SET_OVERRIDE
    #define AGX_PHYSICS_STRONGINTERACTION_DATA_SET
    class AGXPHYSICS_EXPORT StrongInteractionData : public agx::Physics::InteractionData
    {
    public:
      StrongInteractionInstance operator[] (size_t index);

    public:
      agxData::Array< StrongInteractionPtr >& instance;
      agxData::Array< agx::StrongInteraction* > model;
      agxData::Array< agx::UInt > body1;
      agxData::Array< agx::UInt > body2;

    public:
      typedef agx::StrongInteraction* modelType;
      typedef agx::UInt body1Type;
      typedef agx::UInt body2Type;

    public:
      StrongInteractionData(agxData::EntityStorage* storage);
      StrongInteractionData();

    protected:
      virtual ~StrongInteractionData() {}
      virtual void setNumElements(agx::Index numElements) override;

    private:
      StrongInteractionData& operator= (const StrongInteractionData&) { return *this; }

    };
    #endif
    DOXYGEN_END_INTERNAL_BLOCK()


    DOXYGEN_START_INTERNAL_BLOCK()
    class AGXPHYSICS_EXPORT StrongInteractionSemantics : public agx::Physics::InteractionSemantics
    {
    public:

      // Automatic getters
      agx::StrongInteraction* const& getModel() const;
      agx::UInt const& getBody1() const;
      agx::UInt const& getBody2() const;

      // Semantics defined by explicit kernels

      // Automatic setters
      void setModel(agx::StrongInteraction* const& value);
      void setBody1(agx::UInt const& value);
      void setBody2(agx::UInt const& value);


    protected:
      friend class StrongInteractionPtr;
      friend class StrongInteractionInstance;
      StrongInteractionSemantics();
    };
    DOXYGEN_END_INTERNAL_BLOCK()


    /**
    Pointer to a entity instance of type Physics.StrongInteraction
    */
    class CALLABLE StrongInteractionPtr : public agx::Physics::InteractionPtr
    {
    public:
      typedef StrongInteractionModel ModelType;
      typedef StrongInteractionData DataType;
      typedef StrongInteractionInstance InstanceType;

    public:
      AGXPHYSICS_EXPORT StrongInteractionPtr();
      AGXPHYSICS_EXPORT StrongInteractionPtr(agxData::EntityStorage* storage, agx::Index id);
      AGXPHYSICS_EXPORT StrongInteractionPtr(const agxData::EntityPtr& ptr);
      AGXPHYSICS_EXPORT StrongInteractionPtr(const agxData::EntityInstance& instance);
      AGXPHYSICS_EXPORT StrongInteractionPtr& operator= (const agxData::EntityPtr& ptr);
      AGXPHYSICS_EXPORT StrongInteractionPtr& operator= (const agxData::EntityInstance& instance);
      AGXPHYSICS_EXPORT StrongInteractionInstance instance();
      AGXPHYSICS_EXPORT const StrongInteractionInstance instance() const;

      AGXPHYSICS_EXPORT StrongInteractionSemantics* operator->();
      AGXPHYSICS_EXPORT const StrongInteractionSemantics* operator->() const;

      StrongInteractionData* getData();
      const StrongInteractionData* getData() const;


      /// \return reference to the model attribute
      AGXPHYSICS_EXPORT agx::StrongInteraction*& model();
      /// \return const reference to the model attribute
      AGXPHYSICS_EXPORT agx::StrongInteraction* const& model() const;

      /// \return reference to the body1 attribute
      AGXPHYSICS_EXPORT agx::UInt& body1();
      /// \return const reference to the body1 attribute
      AGXPHYSICS_EXPORT agx::UInt const& body1() const;

      /// \return reference to the body2 attribute
      AGXPHYSICS_EXPORT agx::UInt& body2();
      /// \return const reference to the body2 attribute
      AGXPHYSICS_EXPORT agx::UInt const& body2() const;

      DOXYGEN_START_INTERNAL_BLOCK()
      // Only used by tolua!
      #ifdef AGX_GENERATED_TOLUA_WRAPPER
      inline agx::StrongInteraction* const& getModel() const { return (*this)->getModel(); }
      inline agx::UInt const& getBody1() const { return (*this)->getBody1(); }
      inline agx::UInt const& getBody2() const { return (*this)->getBody2(); }


      inline void setModel(agx::StrongInteraction* const& value) { (*this)->setModel(value); }
      inline void setBody1(agx::UInt const& value) { (*this)->setBody1(value); }
      inline void setBody2(agx::UInt const& value) { (*this)->setBody2(value); }
      #endif
      DOXYGEN_END_INTERNAL_BLOCK()

    };


    DOXYGEN_START_INTERNAL_BLOCK()
    class AGXPHYSICS_EXPORT StrongInteractionInstance : public agx::Physics::InteractionInstance
    {
    public:
      StrongInteractionInstance();
      StrongInteractionInstance(StrongInteractionData* data, agx::Index index);
      StrongInteractionInstance(agxData::EntityStorage *storage, agx::Index index);
      StrongInteractionInstance(const agxData::EntityInstance& other);
      StrongInteractionInstance(const agxData::EntityPtr& ptr);

      StrongInteractionData* getData();
      const StrongInteractionData* getData() const;

    public:
      /// \return reference to the model attribute
      agx::StrongInteraction*& model();
      /// \return const reference to the model attribute
      agx::StrongInteraction* const& model() const;

      /// \return reference to the body1 attribute
      agx::UInt& body1();
      /// \return const reference to the body1 attribute
      agx::UInt const& body1() const;

      /// \return reference to the body2 attribute
      agx::UInt& body2();
      /// \return const reference to the body2 attribute
      agx::UInt const& body2() const;

    };
    DOXYGEN_END_INTERNAL_BLOCK()



    typedef agx::VectorPOD<StrongInteractionPtr> StrongInteractionPtrVector;
    typedef agxData::Array<StrongInteractionPtr> StrongInteractionPtrArray;



    DOXYGEN_START_INTERNAL_BLOCK()
    /* Implementation */
    //-----------------------------------------------------------------------------------------------------
    //-----------------------------------------------------------------------------------------------------
    inline StrongInteractionInstance agx::Physics::StrongInteractionData::operator[] (size_t index) { return StrongInteractionInstance(this, (agx::Index)index); }
    //-----------------------------------------------------------------------------------------------------
    AGX_FORCE_INLINE StrongInteractionPtr::StrongInteractionPtr() {}
    AGX_FORCE_INLINE StrongInteractionPtr::StrongInteractionPtr(agxData::EntityStorage* storage, agx::Index id) : agx::Physics::InteractionPtr(storage, id) {}
    AGX_FORCE_INLINE StrongInteractionPtr::StrongInteractionPtr(const agxData::EntityPtr& ptr) : agx::Physics::InteractionPtr(ptr)
    {
      agxAssertN(!ptr || ptr.isInstanceOf(StrongInteractionModel::instance()),
        "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
        EntityPtr::getModel()->fullPath().c_str(), StrongInteractionModel::instance()->fullPath().c_str());
    }

    AGX_FORCE_INLINE StrongInteractionPtr::StrongInteractionPtr(const agxData::EntityInstance& instance) : agx::Physics::InteractionPtr(instance)
    {
      agxAssertN(!instance || instance.isInstanceOf(StrongInteractionModel::instance()),
        "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
        EntityPtr::getModel()->fullPath().c_str(), StrongInteractionModel::instance()->fullPath().c_str());
    }

    AGX_FORCE_INLINE StrongInteractionPtr& StrongInteractionPtr::operator= (const agxData::EntityPtr& ptr)
    {
      agxData::EntityPtr::operator= (ptr);
      agxAssertN(!ptr || ptr.isInstanceOf(StrongInteractionModel::instance()),
        "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
        EntityPtr::getModel()->fullPath().c_str(), StrongInteractionModel::instance()->fullPath().c_str());
      return *this;
    }

    AGX_FORCE_INLINE StrongInteractionPtr& StrongInteractionPtr::operator= (const agxData::EntityInstance& instance)
    {
      agxData::EntityPtr::operator= (instance);
      agxAssertN(!instance || instance.isInstanceOf(StrongInteractionModel::instance()),
        "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
        EntityPtr::getModel()->fullPath().c_str(), StrongInteractionModel::instance()->fullPath().c_str());
      return *this;
    }

    inline StrongInteractionInstance StrongInteractionPtr::instance() { return agxData::EntityPtr::instance(); }
    inline const StrongInteractionInstance StrongInteractionPtr::instance() const { return agxData::EntityPtr::instance(); }
    AGX_FORCE_INLINE StrongInteractionSemantics* StrongInteractionPtr::operator->() { return (StrongInteractionSemantics* )this; }
    AGX_FORCE_INLINE const StrongInteractionSemantics* StrongInteractionPtr::operator->() const { return (const StrongInteractionSemantics* )this; }
    AGX_FORCE_INLINE StrongInteractionData* StrongInteractionPtr::getData() { return static_cast<StrongInteractionData* >(agxData::EntityPtr::getData()); }
    AGX_FORCE_INLINE const StrongInteractionData* StrongInteractionPtr::getData() const { return static_cast<const StrongInteractionData* >(agxData::EntityPtr::getData()); }

    AGX_FORCE_INLINE agx::StrongInteraction*& StrongInteractionPtr::model() { verifyIndex(); return getData()->model[calculateIndex()]; }
    AGX_FORCE_INLINE agx::StrongInteraction* const& StrongInteractionPtr::model() const { verifyIndex(); return getData()->model[calculateIndex()]; }

    AGX_FORCE_INLINE agx::UInt& StrongInteractionPtr::body1() { verifyIndex(); return getData()->body1[calculateIndex()]; }
    AGX_FORCE_INLINE agx::UInt const& StrongInteractionPtr::body1() const { verifyIndex(); return getData()->body1[calculateIndex()]; }

    AGX_FORCE_INLINE agx::UInt& StrongInteractionPtr::body2() { verifyIndex(); return getData()->body2[calculateIndex()]; }
    AGX_FORCE_INLINE agx::UInt const& StrongInteractionPtr::body2() const { verifyIndex(); return getData()->body2[calculateIndex()]; }

    //-----------------------------------------------------------------------------------------------------
    AGX_FORCE_INLINE StrongInteractionInstance::StrongInteractionInstance() {}
    AGX_FORCE_INLINE StrongInteractionInstance::StrongInteractionInstance(StrongInteractionData* data, agx::Index index) : agx::Physics::InteractionInstance(data, index) {}
    AGX_FORCE_INLINE StrongInteractionInstance::StrongInteractionInstance(agxData::EntityStorage* storage, agx::Index index) : agx::Physics::InteractionInstance(storage, index) {}
    AGX_FORCE_INLINE StrongInteractionInstance::StrongInteractionInstance(const agxData::EntityInstance& other) : agx::Physics::InteractionInstance(other)
    {
      agxAssertN(!other || other.isInstanceOf(StrongInteractionModel::instance()),
        "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
        EntityInstance::getModel()->fullPath().c_str(), StrongInteractionModel::instance()->fullPath().c_str());
    }

    AGX_FORCE_INLINE StrongInteractionInstance::StrongInteractionInstance(const agxData::EntityPtr& ptr) : agx::Physics::InteractionInstance(ptr)
    {
      agxAssertN(!ptr || ptr.isInstanceOf(StrongInteractionModel::instance()),
        "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
        EntityInstance::getModel()->fullPath().c_str(), StrongInteractionModel::instance()->fullPath().c_str());
    }


    AGX_FORCE_INLINE StrongInteractionData* StrongInteractionInstance::getData() { return static_cast<StrongInteractionData* >(agxData::EntityInstance::getData()); }
    AGX_FORCE_INLINE const StrongInteractionData* StrongInteractionInstance::getData() const { return static_cast<const StrongInteractionData* >(agxData::EntityInstance::getData()); }

    AGX_FORCE_INLINE agx::StrongInteraction*& StrongInteractionInstance::model() { verifyIndex(); return getData()->model[getIndex()]; }
    AGX_FORCE_INLINE agx::StrongInteraction* const& StrongInteractionInstance::model() const { verifyIndex(); return getData()->model[getIndex()]; }

    AGX_FORCE_INLINE agx::UInt& StrongInteractionInstance::body1() { verifyIndex(); return getData()->body1[getIndex()]; }
    AGX_FORCE_INLINE agx::UInt const& StrongInteractionInstance::body1() const { verifyIndex(); return getData()->body1[getIndex()]; }

    AGX_FORCE_INLINE agx::UInt& StrongInteractionInstance::body2() { verifyIndex(); return getData()->body2[getIndex()]; }
    AGX_FORCE_INLINE agx::UInt const& StrongInteractionInstance::body2() const { verifyIndex(); return getData()->body2[getIndex()]; }

    //-----------------------------------------------------------------------------------------------------
    AGX_FORCE_INLINE StrongInteractionSemantics::StrongInteractionSemantics() {}
    //-----------------------------------------------------------------------------------------------------
    DOXYGEN_END_INTERNAL_BLOCK()
  }
}

AGX_TYPE_BINDING(agx::Physics::StrongInteractionPtr, "Physics.StrongInteractionPtr")
AGX_TYPE_BINDING(agx::Physics::StrongInteractionInstance, "Physics.StrongInteractionInstance")

#ifdef _MSC_VER
# pragma warning(pop)
#endif

#undef AGX_ENTITY_WRAPPER
#undef AGX_ENTITY_NAMESPACE
#endif

