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

#ifndef GENERATED_AGX_PHYSICS_BROADPHASEPAIR_H_PLUGIN
#define GENERATED_AGX_PHYSICS_BROADPHASEPAIR_H_PLUGIN

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
#include <agx/Physics/GeometryPairEntity.h>
#include <agx/Integer.h>
#include <agx/Physics/GeometryContactEntity.h>
#include <agx/Physics/WarmStartingDataEntity.h>

namespace agx { namespace Physics { class GeometryContactPtr; }}
namespace agx { namespace Physics { class WarmStartingDataPtr; }}

namespace agx
{
  namespace Physics
  {

    class BroadPhasePairModel;
    class BroadPhasePairData;
    class BroadPhasePairPtr;
    class BroadPhasePairInstance;
    class BroadPhasePairSemantics;


    AGX_DECLARE_POINTER_TYPES(BroadPhasePairModel);

    /** 
    Abstract description of the data attributes for the Physics.BroadPhasePair entity.
    */ 
    class AGXPHYSICS_EXPORT BroadPhasePairModel : public agx::Physics::GeometryPairModel
    {
    public:
      typedef BroadPhasePairPtr PtrT;

      BroadPhasePairModel(const agx::String& name = "BroadPhasePair");

      /// \return The entity model singleton.
      static BroadPhasePairModel* instance();

      /// Create and return a pointer to a new instance in the default storage for this entity model.
      static BroadPhasePairPtr createInstance();

      /// \return The default storage for this entity model.
      static agxData::EntityStorage* defaultStorage();

      /// This is part of internal cleanup and should not be called by users
      virtual void shutdownCleanup() override;



      /* Attributes */
      static agxData::ScalarAttributeT< agx::UInt8 >* stateAttribute;
      static agxData::ScalarAttributeT< agx::Physics::GeometryContactPtr >* contactAttribute;
      static agxData::ArrayAttributeT< agx::Physics::WarmStartingDataPtr >* warmStartingDataAttribute;

    protected:
      virtual ~BroadPhasePairModel();
      virtual agxData::EntityData* createData(agxData::EntityStorage* storage) override;
      virtual void configure(agx::TiXmlElement* eEntity) override;
      virtual void initAttributeAccessors() override;
      virtual void construct(agxData::EntityPtr instance) override;
      void construct(agx::Physics::BroadPhasePairPtr broadPhasePair);
    };


    DOXYGEN_START_INTERNAL_BLOCK()
    #ifndef AGX_PHYSICS_BROADPHASEPAIR_DATA_SET_OVERRIDE
    #define AGX_PHYSICS_BROADPHASEPAIR_DATA_SET
    class AGXPHYSICS_EXPORT BroadPhasePairData : public agx::Physics::GeometryPairData
    {
    public:
      BroadPhasePairInstance operator[] (size_t index);

    public:
      agxData::Array< BroadPhasePairPtr >& instance;
      agxData::Array< agx::UInt8 > state;
      agxData::Array< agx::Physics::GeometryContactPtr > contact;
      agxData::Array< agxData::Array< agx::Physics::WarmStartingDataPtr > > warmStartingData;

    public:
      typedef agx::UInt8 stateType;
      typedef agx::Physics::GeometryContactPtr contactType;
      typedef agxData::Array< agx::Physics::WarmStartingDataPtr > warmStartingDataType;

    public:
      BroadPhasePairData(agxData::EntityStorage* storage);
      BroadPhasePairData();

    protected:
      virtual ~BroadPhasePairData() {}
      virtual void setNumElements(agx::Index numElements) override;

    private:
      BroadPhasePairData& operator= (const BroadPhasePairData&) { return *this; }

    };
    #endif
    DOXYGEN_END_INTERNAL_BLOCK()


    DOXYGEN_START_INTERNAL_BLOCK()
    class AGXPHYSICS_EXPORT BroadPhasePairSemantics : public agx::Physics::GeometryPairSemantics
    {
    public:

      // Automatic getters
      agx::UInt8 const& getState() const;
      agx::Physics::GeometryContactPtr const& getContact() const;
      agxData::Array< agx::Physics::WarmStartingDataPtr > const& getWarmStartingData() const;

      // Semantics defined by explicit kernels

      // Automatic setters
      void setState(agx::UInt8 const& value);
      void setContact(agx::Physics::GeometryContactPtr const& value);
      void setWarmStartingData(agxData::Array< agx::Physics::WarmStartingDataPtr > const& value);


    protected:
      friend class BroadPhasePairPtr;
      friend class BroadPhasePairInstance;
      BroadPhasePairSemantics();
    };
    DOXYGEN_END_INTERNAL_BLOCK()


    /**
    Pointer to a entity instance of type Physics.BroadPhasePair
    */
    class CALLABLE BroadPhasePairPtr : public agx::Physics::GeometryPairPtr
    {
    public:
      typedef BroadPhasePairModel ModelType;
      typedef BroadPhasePairData DataType;
      typedef BroadPhasePairInstance InstanceType;

    public:
      AGXPHYSICS_EXPORT BroadPhasePairPtr();
      AGXPHYSICS_EXPORT BroadPhasePairPtr(agxData::EntityStorage* storage, agx::Index id);
      AGXPHYSICS_EXPORT BroadPhasePairPtr(const agxData::EntityPtr& ptr);
      AGXPHYSICS_EXPORT BroadPhasePairPtr(const agxData::EntityInstance& instance);
      AGXPHYSICS_EXPORT BroadPhasePairPtr& operator= (const agxData::EntityPtr& ptr);
      AGXPHYSICS_EXPORT BroadPhasePairPtr& operator= (const agxData::EntityInstance& instance);
      AGXPHYSICS_EXPORT BroadPhasePairInstance instance();
      AGXPHYSICS_EXPORT const BroadPhasePairInstance instance() const;

      AGXPHYSICS_EXPORT BroadPhasePairSemantics* operator->();
      AGXPHYSICS_EXPORT const BroadPhasePairSemantics* operator->() const;

      BroadPhasePairData* getData();
      const BroadPhasePairData* getData() const;


      /// \return reference to the state attribute
      AGXPHYSICS_EXPORT agx::UInt8& state();
      /// \return const reference to the state attribute
      AGXPHYSICS_EXPORT agx::UInt8 const& state() const;

      /// \return reference to the contact attribute
      AGXPHYSICS_EXPORT agx::Physics::GeometryContactPtr& contact();
      /// \return const reference to the contact attribute
      AGXPHYSICS_EXPORT agx::Physics::GeometryContactPtr const& contact() const;

      /// \return reference to the warmStartingData attribute
      AGXPHYSICS_EXPORT agxData::Array< agx::Physics::WarmStartingDataPtr >& warmStartingData();
      /// \return const reference to the warmStartingData attribute
      AGXPHYSICS_EXPORT agxData::Array< agx::Physics::WarmStartingDataPtr > const& warmStartingData() const;

      DOXYGEN_START_INTERNAL_BLOCK()
      // Only used by tolua!
      #ifdef AGX_GENERATED_TOLUA_WRAPPER
      inline agx::UInt8 const& getState() const { return (*this)->getState(); }
      inline agx::Physics::GeometryContactPtr const& getContact() const { return (*this)->getContact(); }
      inline agxData::Array< agx::Physics::WarmStartingDataPtr > const& getWarmStartingData() const { return (*this)->getWarmStartingData(); }


      inline void setState(agx::UInt8 const& value) { (*this)->setState(value); }
      inline void setContact(agx::Physics::GeometryContactPtr const& value) { (*this)->setContact(value); }
      inline void setWarmStartingData(agxData::Array< agx::Physics::WarmStartingDataPtr > const& value) { (*this)->setWarmStartingData(value); }
      #endif
      DOXYGEN_END_INTERNAL_BLOCK()

    };


    DOXYGEN_START_INTERNAL_BLOCK()
    class AGXPHYSICS_EXPORT BroadPhasePairInstance : public agx::Physics::GeometryPairInstance
    {
    public:
      BroadPhasePairInstance();
      BroadPhasePairInstance(BroadPhasePairData* data, agx::Index index);
      BroadPhasePairInstance(agxData::EntityStorage *storage, agx::Index index);
      BroadPhasePairInstance(const agxData::EntityInstance& other);
      BroadPhasePairInstance(const agxData::EntityPtr& ptr);

      BroadPhasePairData* getData();
      const BroadPhasePairData* getData() const;

    public:
      /// \return reference to the state attribute
      agx::UInt8& state();
      /// \return const reference to the state attribute
      agx::UInt8 const& state() const;

      /// \return reference to the contact attribute
      agx::Physics::GeometryContactPtr& contact();
      /// \return const reference to the contact attribute
      agx::Physics::GeometryContactPtr const& contact() const;

      /// \return reference to the warmStartingData attribute
      agxData::Array< agx::Physics::WarmStartingDataPtr >& warmStartingData();
      /// \return const reference to the warmStartingData attribute
      agxData::Array< agx::Physics::WarmStartingDataPtr > const& warmStartingData() const;

    };
    DOXYGEN_END_INTERNAL_BLOCK()



    typedef agx::VectorPOD<BroadPhasePairPtr> BroadPhasePairPtrVector;
    typedef agxData::Array<BroadPhasePairPtr> BroadPhasePairPtrArray;



    DOXYGEN_START_INTERNAL_BLOCK()
    /* Implementation */
    //-----------------------------------------------------------------------------------------------------
    //-----------------------------------------------------------------------------------------------------
    inline BroadPhasePairInstance agx::Physics::BroadPhasePairData::operator[] (size_t index) { return BroadPhasePairInstance(this, (agx::Index)index); }
    //-----------------------------------------------------------------------------------------------------
    AGX_FORCE_INLINE BroadPhasePairPtr::BroadPhasePairPtr() {}
    AGX_FORCE_INLINE BroadPhasePairPtr::BroadPhasePairPtr(agxData::EntityStorage* storage, agx::Index id) : agx::Physics::GeometryPairPtr(storage, id) {}
    AGX_FORCE_INLINE BroadPhasePairPtr::BroadPhasePairPtr(const agxData::EntityPtr& ptr) : agx::Physics::GeometryPairPtr(ptr)
    {
      agxAssertN(!ptr || ptr.isInstanceOf(BroadPhasePairModel::instance()),
        "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
        EntityPtr::getModel()->fullPath().c_str(), BroadPhasePairModel::instance()->fullPath().c_str());
    }

    AGX_FORCE_INLINE BroadPhasePairPtr::BroadPhasePairPtr(const agxData::EntityInstance& instance) : agx::Physics::GeometryPairPtr(instance)
    {
      agxAssertN(!instance || instance.isInstanceOf(BroadPhasePairModel::instance()),
        "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
        EntityPtr::getModel()->fullPath().c_str(), BroadPhasePairModel::instance()->fullPath().c_str());
    }

    AGX_FORCE_INLINE BroadPhasePairPtr& BroadPhasePairPtr::operator= (const agxData::EntityPtr& ptr)
    {
      agxData::EntityPtr::operator= (ptr);
      agxAssertN(!ptr || ptr.isInstanceOf(BroadPhasePairModel::instance()),
        "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
        EntityPtr::getModel()->fullPath().c_str(), BroadPhasePairModel::instance()->fullPath().c_str());
      return *this;
    }

    AGX_FORCE_INLINE BroadPhasePairPtr& BroadPhasePairPtr::operator= (const agxData::EntityInstance& instance)
    {
      agxData::EntityPtr::operator= (instance);
      agxAssertN(!instance || instance.isInstanceOf(BroadPhasePairModel::instance()),
        "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
        EntityPtr::getModel()->fullPath().c_str(), BroadPhasePairModel::instance()->fullPath().c_str());
      return *this;
    }

    inline BroadPhasePairInstance BroadPhasePairPtr::instance() { return agxData::EntityPtr::instance(); }
    inline const BroadPhasePairInstance BroadPhasePairPtr::instance() const { return agxData::EntityPtr::instance(); }
    AGX_FORCE_INLINE BroadPhasePairSemantics* BroadPhasePairPtr::operator->() { return (BroadPhasePairSemantics* )this; }
    AGX_FORCE_INLINE const BroadPhasePairSemantics* BroadPhasePairPtr::operator->() const { return (const BroadPhasePairSemantics* )this; }
    AGX_FORCE_INLINE BroadPhasePairData* BroadPhasePairPtr::getData() { return static_cast<BroadPhasePairData* >(agxData::EntityPtr::getData()); }
    AGX_FORCE_INLINE const BroadPhasePairData* BroadPhasePairPtr::getData() const { return static_cast<const BroadPhasePairData* >(agxData::EntityPtr::getData()); }

    AGX_FORCE_INLINE agx::UInt8& BroadPhasePairPtr::state() { verifyIndex(); return getData()->state[calculateIndex()]; }
    AGX_FORCE_INLINE agx::UInt8 const& BroadPhasePairPtr::state() const { verifyIndex(); return getData()->state[calculateIndex()]; }

    AGX_FORCE_INLINE agx::Physics::GeometryContactPtr& BroadPhasePairPtr::contact() { verifyIndex(); return getData()->contact[calculateIndex()]; }
    AGX_FORCE_INLINE agx::Physics::GeometryContactPtr const& BroadPhasePairPtr::contact() const { verifyIndex(); return getData()->contact[calculateIndex()]; }

    AGX_FORCE_INLINE agxData::Array< agx::Physics::WarmStartingDataPtr >& BroadPhasePairPtr::warmStartingData() { verifyIndex(); return getData()->warmStartingData[calculateIndex()]; }
    AGX_FORCE_INLINE agxData::Array< agx::Physics::WarmStartingDataPtr > const& BroadPhasePairPtr::warmStartingData() const { verifyIndex(); return getData()->warmStartingData[calculateIndex()]; }

    //-----------------------------------------------------------------------------------------------------
    AGX_FORCE_INLINE BroadPhasePairInstance::BroadPhasePairInstance() {}
    AGX_FORCE_INLINE BroadPhasePairInstance::BroadPhasePairInstance(BroadPhasePairData* data, agx::Index index) : agx::Physics::GeometryPairInstance(data, index) {}
    AGX_FORCE_INLINE BroadPhasePairInstance::BroadPhasePairInstance(agxData::EntityStorage* storage, agx::Index index) : agx::Physics::GeometryPairInstance(storage, index) {}
    AGX_FORCE_INLINE BroadPhasePairInstance::BroadPhasePairInstance(const agxData::EntityInstance& other) : agx::Physics::GeometryPairInstance(other)
    {
      agxAssertN(!other || other.isInstanceOf(BroadPhasePairModel::instance()),
        "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
        EntityInstance::getModel()->fullPath().c_str(), BroadPhasePairModel::instance()->fullPath().c_str());
    }

    AGX_FORCE_INLINE BroadPhasePairInstance::BroadPhasePairInstance(const agxData::EntityPtr& ptr) : agx::Physics::GeometryPairInstance(ptr)
    {
      agxAssertN(!ptr || ptr.isInstanceOf(BroadPhasePairModel::instance()),
        "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
        EntityInstance::getModel()->fullPath().c_str(), BroadPhasePairModel::instance()->fullPath().c_str());
    }


    AGX_FORCE_INLINE BroadPhasePairData* BroadPhasePairInstance::getData() { return static_cast<BroadPhasePairData* >(agxData::EntityInstance::getData()); }
    AGX_FORCE_INLINE const BroadPhasePairData* BroadPhasePairInstance::getData() const { return static_cast<const BroadPhasePairData* >(agxData::EntityInstance::getData()); }

    AGX_FORCE_INLINE agx::UInt8& BroadPhasePairInstance::state() { verifyIndex(); return getData()->state[getIndex()]; }
    AGX_FORCE_INLINE agx::UInt8 const& BroadPhasePairInstance::state() const { verifyIndex(); return getData()->state[getIndex()]; }

    AGX_FORCE_INLINE agx::Physics::GeometryContactPtr& BroadPhasePairInstance::contact() { verifyIndex(); return getData()->contact[getIndex()]; }
    AGX_FORCE_INLINE agx::Physics::GeometryContactPtr const& BroadPhasePairInstance::contact() const { verifyIndex(); return getData()->contact[getIndex()]; }

    AGX_FORCE_INLINE agxData::Array< agx::Physics::WarmStartingDataPtr >& BroadPhasePairInstance::warmStartingData() { verifyIndex(); return getData()->warmStartingData[getIndex()]; }
    AGX_FORCE_INLINE agxData::Array< agx::Physics::WarmStartingDataPtr > const& BroadPhasePairInstance::warmStartingData() const { verifyIndex(); return getData()->warmStartingData[getIndex()]; }

    //-----------------------------------------------------------------------------------------------------
    AGX_FORCE_INLINE BroadPhasePairSemantics::BroadPhasePairSemantics() {}
    //-----------------------------------------------------------------------------------------------------
    DOXYGEN_END_INTERNAL_BLOCK()
  }
}

AGX_TYPE_BINDING(agx::Physics::BroadPhasePairPtr, "Physics.BroadPhasePairPtr")
AGX_TYPE_BINDING(agx::Physics::BroadPhasePairInstance, "Physics.BroadPhasePairInstance")

#ifdef _MSC_VER
# pragma warning(pop)
#endif

#undef AGX_ENTITY_WRAPPER
#undef AGX_ENTITY_NAMESPACE
#endif

