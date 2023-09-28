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

#ifndef GENERATED_AGX_PHYSICS_EMITTER_H_PLUGIN
#define GENERATED_AGX_PHYSICS_EMITTER_H_PLUGIN

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
#include <agx/ReferencedEntity.h>
#include <agx/Integer.h>
#include <agx/Name.h>
#include <agx/Physics/GeometryEntity.h>
#include <agx/Physics/CollisionGroupSetEntity.h>
#include <agx/Vec3.h>
#include <agx/Real.h>
namespace agx { class Emitter; }

namespace agx { namespace Physics { class GeometryPtr; }}
namespace agx { namespace Physics { class CollisionGroupSetPtr; }}

namespace agx
{
  namespace Physics
  {

    class EmitterModel;
    class EmitterData;
    class EmitterPtr;
    class EmitterInstance;
    class EmitterSemantics;


    AGX_DECLARE_POINTER_TYPES(EmitterModel);

    /** 
    Abstract description of the data attributes for the Physics.Emitter entity.
    */ 
    class AGXPHYSICS_EXPORT EmitterModel : public agx::ReferencedModel
    {
    public:
      typedef EmitterPtr PtrT;

      EmitterModel(const agx::String& name = "Emitter");

      /// \return The entity model singleton.
      static EmitterModel* instance();

      /// Create and return a pointer to a new instance in the default storage for this entity model.
      static EmitterPtr createInstance();

      /// \return The default storage for this entity model.
      static agxData::EntityStorage* defaultStorage();

      /// This is part of internal cleanup and should not be called by users
      virtual void shutdownCleanup() override;



      /* Attributes */
      static agxData::ScalarAttributeT< agx::UInt32 >* idAttribute;
      static agxData::ScalarAttributeT< agx::Name >* nameAttribute;
      static agxData::ScalarAttributeT< agx::Physics::GeometryPtr >* geometryAttribute;
      static agxData::ScalarAttributeT< agx::Physics::CollisionGroupSetPtr >* collisionGroupSetAttribute;
      static agxData::ScalarAttributeT< agx::Vec3 >* velocityAttribute;
      static agxData::ScalarAttributeT< agx::Real >* rateAttribute;
      static agxData::ScalarAttributeT< agx::Real >* emittedQuantityAttribute;
      static agxData::ScalarAttributeT< agx::Real >* emittedQuantityMaximumAttribute;
      static agxData::ScalarAttributeT< agx::UInt >* emittedCountAttribute;
      static agxData::ScalarAttributeT< agx::UInt32 >* seedAttribute;
      static agxData::PointerAttributeT< agx::Emitter*>* modelAttribute;

    protected:
      virtual ~EmitterModel();
      virtual agxData::EntityData* createData(agxData::EntityStorage* storage) override;
      virtual void configure(agx::TiXmlElement* eEntity) override;
      virtual void initAttributeAccessors() override;
      virtual void construct(agxData::EntityPtr instance) override;
      void construct(agx::Physics::EmitterPtr emitter);
    };


    DOXYGEN_START_INTERNAL_BLOCK()
    #ifndef AGX_PHYSICS_EMITTER_DATA_SET_OVERRIDE
    #define AGX_PHYSICS_EMITTER_DATA_SET
    class AGXPHYSICS_EXPORT EmitterData : public agx::ReferencedData
    {
    public:
      EmitterInstance operator[] (size_t index);

    public:
      agxData::Array< EmitterPtr >& instance;
      agxData::Array< agx::UInt32 > id;
      agxData::Array< agx::Name > name;
      agxData::Array< agx::Physics::GeometryPtr > geometry;
      agxData::Array< agx::Physics::CollisionGroupSetPtr > collisionGroupSet;
      agxData::Array< agx::Vec3 > velocity;
      agxData::Array< agx::Real > rate;
      agxData::Array< agx::Real > emittedQuantity;
      agxData::Array< agx::Real > emittedQuantityMaximum;
      agxData::Array< agx::UInt > emittedCount;
      agxData::Array< agx::UInt32 > seed;
      agxData::Array< agx::Emitter* > model;

    public:
      typedef agx::UInt32 idType;
      typedef agx::Name nameType;
      typedef agx::Physics::GeometryPtr geometryType;
      typedef agx::Physics::CollisionGroupSetPtr collisionGroupSetType;
      typedef agx::Vec3 velocityType;
      typedef agx::Real rateType;
      typedef agx::Real emittedQuantityType;
      typedef agx::Real emittedQuantityMaximumType;
      typedef agx::UInt emittedCountType;
      typedef agx::UInt32 seedType;
      typedef agx::Emitter* modelType;

    public:
      EmitterData(agxData::EntityStorage* storage);
      EmitterData();

    protected:
      virtual ~EmitterData() {}
      virtual void setNumElements(agx::Index numElements) override;

    private:
      EmitterData& operator= (const EmitterData&) { return *this; }

    };
    #endif
    DOXYGEN_END_INTERNAL_BLOCK()


    DOXYGEN_START_INTERNAL_BLOCK()
    class AGXPHYSICS_EXPORT EmitterSemantics : public agx::ReferencedSemantics
    {
    public:

      // Automatic getters
      agx::UInt32 const& getId() const;
      agx::Name const& getName() const;
      agx::Physics::GeometryPtr const& getGeometry() const;
      agx::Physics::CollisionGroupSetPtr const& getCollisionGroupSet() const;
      agx::Vec3 const& getVelocity() const;
      agx::Real const& getRate() const;
      agx::Real const& getEmittedQuantity() const;
      agx::Real const& getEmittedQuantityMaximum() const;
      agx::UInt const& getEmittedCount() const;
      agx::UInt32 const& getSeed() const;
      agx::Emitter* const& getModel() const;

      // Semantics defined by explicit kernels

      // Automatic setters
      void setId(agx::UInt32 const& value);
      void setName(agx::Name const& value);
      void setGeometry(agx::Physics::GeometryPtr const& value);
      void setCollisionGroupSet(agx::Physics::CollisionGroupSetPtr const& value);
      void setVelocity(agx::Vec3 const& value);
      void setRate(agx::Real const& value);
      void setEmittedQuantity(agx::Real const& value);
      void setEmittedQuantityMaximum(agx::Real const& value);
      void setEmittedCount(agx::UInt const& value);
      void setSeed(agx::UInt32 const& value);
      void setModel(agx::Emitter* const& value);


    protected:
      friend class EmitterPtr;
      friend class EmitterInstance;
      EmitterSemantics();
    };
    DOXYGEN_END_INTERNAL_BLOCK()


    /**
    Pointer to a entity instance of type Physics.Emitter
    */
    class CALLABLE EmitterPtr : public agx::ReferencedPtr
    {
    public:
      typedef EmitterModel ModelType;
      typedef EmitterData DataType;
      typedef EmitterInstance InstanceType;

    public:
      AGXPHYSICS_EXPORT EmitterPtr();
      AGXPHYSICS_EXPORT EmitterPtr(agxData::EntityStorage* storage, agx::Index id);
      AGXPHYSICS_EXPORT EmitterPtr(const agxData::EntityPtr& ptr);
      AGXPHYSICS_EXPORT EmitterPtr(const agxData::EntityInstance& instance);
      AGXPHYSICS_EXPORT EmitterPtr& operator= (const agxData::EntityPtr& ptr);
      AGXPHYSICS_EXPORT EmitterPtr& operator= (const agxData::EntityInstance& instance);
      AGXPHYSICS_EXPORT EmitterInstance instance();
      AGXPHYSICS_EXPORT const EmitterInstance instance() const;

      AGXPHYSICS_EXPORT EmitterSemantics* operator->();
      AGXPHYSICS_EXPORT const EmitterSemantics* operator->() const;

      EmitterData* getData();
      const EmitterData* getData() const;


      /// \return reference to the id attribute
      AGXPHYSICS_EXPORT agx::UInt32& id();
      /// \return const reference to the id attribute
      AGXPHYSICS_EXPORT agx::UInt32 const& id() const;

      /// \return reference to the name attribute
      AGXPHYSICS_EXPORT agx::Name& name();
      /// \return const reference to the name attribute
      AGXPHYSICS_EXPORT agx::Name const& name() const;

      /// \return reference to the geometry attribute
      AGXPHYSICS_EXPORT agx::Physics::GeometryPtr& geometry();
      /// \return const reference to the geometry attribute
      AGXPHYSICS_EXPORT agx::Physics::GeometryPtr const& geometry() const;

      /// \return reference to the collisionGroupSet attribute
      AGXPHYSICS_EXPORT agx::Physics::CollisionGroupSetPtr& collisionGroupSet();
      /// \return const reference to the collisionGroupSet attribute
      AGXPHYSICS_EXPORT agx::Physics::CollisionGroupSetPtr const& collisionGroupSet() const;

      /// \return reference to the velocity attribute
      AGXPHYSICS_EXPORT agx::Vec3& velocity();
      /// \return const reference to the velocity attribute
      AGXPHYSICS_EXPORT agx::Vec3 const& velocity() const;

      /// \return reference to the rate attribute
      AGXPHYSICS_EXPORT agx::Real& rate();
      /// \return const reference to the rate attribute
      AGXPHYSICS_EXPORT agx::Real const& rate() const;

      /// \return reference to the emittedQuantity attribute
      AGXPHYSICS_EXPORT agx::Real& emittedQuantity();
      /// \return const reference to the emittedQuantity attribute
      AGXPHYSICS_EXPORT agx::Real const& emittedQuantity() const;

      /// \return reference to the emittedQuantityMaximum attribute
      AGXPHYSICS_EXPORT agx::Real& emittedQuantityMaximum();
      /// \return const reference to the emittedQuantityMaximum attribute
      AGXPHYSICS_EXPORT agx::Real const& emittedQuantityMaximum() const;

      /// \return reference to the emittedCount attribute
      AGXPHYSICS_EXPORT agx::UInt& emittedCount();
      /// \return const reference to the emittedCount attribute
      AGXPHYSICS_EXPORT agx::UInt const& emittedCount() const;

      /// \return reference to the seed attribute
      AGXPHYSICS_EXPORT agx::UInt32& seed();
      /// \return const reference to the seed attribute
      AGXPHYSICS_EXPORT agx::UInt32 const& seed() const;

      /// \return reference to the model attribute
      AGXPHYSICS_EXPORT agx::Emitter*& model();
      /// \return const reference to the model attribute
      AGXPHYSICS_EXPORT agx::Emitter* const& model() const;

      DOXYGEN_START_INTERNAL_BLOCK()
      // Only used by tolua!
      #ifdef AGX_GENERATED_TOLUA_WRAPPER
      inline agx::UInt32 const& getId() const { return (*this)->getId(); }
      inline agx::Name const& getName() const { return (*this)->getName(); }
      inline agx::Physics::GeometryPtr const& getGeometry() const { return (*this)->getGeometry(); }
      inline agx::Physics::CollisionGroupSetPtr const& getCollisionGroupSet() const { return (*this)->getCollisionGroupSet(); }
      inline agx::Vec3 const& getVelocity() const { return (*this)->getVelocity(); }
      inline agx::Real const& getRate() const { return (*this)->getRate(); }
      inline agx::Real const& getEmittedQuantity() const { return (*this)->getEmittedQuantity(); }
      inline agx::Real const& getEmittedQuantityMaximum() const { return (*this)->getEmittedQuantityMaximum(); }
      inline agx::UInt const& getEmittedCount() const { return (*this)->getEmittedCount(); }
      inline agx::UInt32 const& getSeed() const { return (*this)->getSeed(); }
      inline agx::Emitter* const& getModel() const { return (*this)->getModel(); }


      inline void setId(agx::UInt32 const& value) { (*this)->setId(value); }
      inline void setName(agx::Name const& value) { (*this)->setName(value); }
      inline void setGeometry(agx::Physics::GeometryPtr const& value) { (*this)->setGeometry(value); }
      inline void setCollisionGroupSet(agx::Physics::CollisionGroupSetPtr const& value) { (*this)->setCollisionGroupSet(value); }
      inline void setVelocity(agx::Vec3 const& value) { (*this)->setVelocity(value); }
      inline void setRate(agx::Real const& value) { (*this)->setRate(value); }
      inline void setEmittedQuantity(agx::Real const& value) { (*this)->setEmittedQuantity(value); }
      inline void setEmittedQuantityMaximum(agx::Real const& value) { (*this)->setEmittedQuantityMaximum(value); }
      inline void setEmittedCount(agx::UInt const& value) { (*this)->setEmittedCount(value); }
      inline void setSeed(agx::UInt32 const& value) { (*this)->setSeed(value); }
      inline void setModel(agx::Emitter* const& value) { (*this)->setModel(value); }
      #endif
      DOXYGEN_END_INTERNAL_BLOCK()

    };

    // Entity is Referenced
    typedef agxData::EntityRef< EmitterPtr > EmitterRef;


    DOXYGEN_START_INTERNAL_BLOCK()
    class AGXPHYSICS_EXPORT EmitterInstance : public agx::ReferencedInstance
    {
    public:
      EmitterInstance();
      EmitterInstance(EmitterData* data, agx::Index index);
      EmitterInstance(agxData::EntityStorage *storage, agx::Index index);
      EmitterInstance(const agxData::EntityInstance& other);
      EmitterInstance(const agxData::EntityPtr& ptr);

      EmitterData* getData();
      const EmitterData* getData() const;

    public:
      /// \return reference to the id attribute
      agx::UInt32& id();
      /// \return const reference to the id attribute
      agx::UInt32 const& id() const;

      /// \return reference to the name attribute
      agx::Name& name();
      /// \return const reference to the name attribute
      agx::Name const& name() const;

      /// \return reference to the geometry attribute
      agx::Physics::GeometryPtr& geometry();
      /// \return const reference to the geometry attribute
      agx::Physics::GeometryPtr const& geometry() const;

      /// \return reference to the collisionGroupSet attribute
      agx::Physics::CollisionGroupSetPtr& collisionGroupSet();
      /// \return const reference to the collisionGroupSet attribute
      agx::Physics::CollisionGroupSetPtr const& collisionGroupSet() const;

      /// \return reference to the velocity attribute
      agx::Vec3& velocity();
      /// \return const reference to the velocity attribute
      agx::Vec3 const& velocity() const;

      /// \return reference to the rate attribute
      agx::Real& rate();
      /// \return const reference to the rate attribute
      agx::Real const& rate() const;

      /// \return reference to the emittedQuantity attribute
      agx::Real& emittedQuantity();
      /// \return const reference to the emittedQuantity attribute
      agx::Real const& emittedQuantity() const;

      /// \return reference to the emittedQuantityMaximum attribute
      agx::Real& emittedQuantityMaximum();
      /// \return const reference to the emittedQuantityMaximum attribute
      agx::Real const& emittedQuantityMaximum() const;

      /// \return reference to the emittedCount attribute
      agx::UInt& emittedCount();
      /// \return const reference to the emittedCount attribute
      agx::UInt const& emittedCount() const;

      /// \return reference to the seed attribute
      agx::UInt32& seed();
      /// \return const reference to the seed attribute
      agx::UInt32 const& seed() const;

      /// \return reference to the model attribute
      agx::Emitter*& model();
      /// \return const reference to the model attribute
      agx::Emitter* const& model() const;

    };
    DOXYGEN_END_INTERNAL_BLOCK()



    typedef agx::VectorPOD<EmitterPtr> EmitterPtrVector;
    typedef agxData::Array<EmitterPtr> EmitterPtrArray;



    DOXYGEN_START_INTERNAL_BLOCK()
    /* Implementation */
    //-----------------------------------------------------------------------------------------------------
    //-----------------------------------------------------------------------------------------------------
    inline EmitterInstance agx::Physics::EmitterData::operator[] (size_t index) { return EmitterInstance(this, (agx::Index)index); }
    //-----------------------------------------------------------------------------------------------------
    AGX_FORCE_INLINE EmitterPtr::EmitterPtr() {}
    AGX_FORCE_INLINE EmitterPtr::EmitterPtr(agxData::EntityStorage* storage, agx::Index id) : agx::ReferencedPtr(storage, id) {}
    AGX_FORCE_INLINE EmitterPtr::EmitterPtr(const agxData::EntityPtr& ptr) : agx::ReferencedPtr(ptr)
    {
      agxAssertN(!ptr || ptr.isInstanceOf(EmitterModel::instance()),
        "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
        EntityPtr::getModel()->fullPath().c_str(), EmitterModel::instance()->fullPath().c_str());
    }

    AGX_FORCE_INLINE EmitterPtr::EmitterPtr(const agxData::EntityInstance& instance) : agx::ReferencedPtr(instance)
    {
      agxAssertN(!instance || instance.isInstanceOf(EmitterModel::instance()),
        "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
        EntityPtr::getModel()->fullPath().c_str(), EmitterModel::instance()->fullPath().c_str());
    }

    AGX_FORCE_INLINE EmitterPtr& EmitterPtr::operator= (const agxData::EntityPtr& ptr)
    {
      agxData::EntityPtr::operator= (ptr);
      agxAssertN(!ptr || ptr.isInstanceOf(EmitterModel::instance()),
        "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
        EntityPtr::getModel()->fullPath().c_str(), EmitterModel::instance()->fullPath().c_str());
      return *this;
    }

    AGX_FORCE_INLINE EmitterPtr& EmitterPtr::operator= (const agxData::EntityInstance& instance)
    {
      agxData::EntityPtr::operator= (instance);
      agxAssertN(!instance || instance.isInstanceOf(EmitterModel::instance()),
        "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
        EntityPtr::getModel()->fullPath().c_str(), EmitterModel::instance()->fullPath().c_str());
      return *this;
    }

    inline EmitterInstance EmitterPtr::instance() { return agxData::EntityPtr::instance(); }
    inline const EmitterInstance EmitterPtr::instance() const { return agxData::EntityPtr::instance(); }
    AGX_FORCE_INLINE EmitterSemantics* EmitterPtr::operator->() { return (EmitterSemantics* )this; }
    AGX_FORCE_INLINE const EmitterSemantics* EmitterPtr::operator->() const { return (const EmitterSemantics* )this; }
    AGX_FORCE_INLINE EmitterData* EmitterPtr::getData() { return static_cast<EmitterData* >(agxData::EntityPtr::getData()); }
    AGX_FORCE_INLINE const EmitterData* EmitterPtr::getData() const { return static_cast<const EmitterData* >(agxData::EntityPtr::getData()); }

    AGX_FORCE_INLINE agx::UInt32& EmitterPtr::id() { verifyIndex(); return getData()->id[calculateIndex()]; }
    AGX_FORCE_INLINE agx::UInt32 const& EmitterPtr::id() const { verifyIndex(); return getData()->id[calculateIndex()]; }

    AGX_FORCE_INLINE agx::Name& EmitterPtr::name() { verifyIndex(); return getData()->name[calculateIndex()]; }
    AGX_FORCE_INLINE agx::Name const& EmitterPtr::name() const { verifyIndex(); return getData()->name[calculateIndex()]; }

    AGX_FORCE_INLINE agx::Physics::GeometryPtr& EmitterPtr::geometry() { verifyIndex(); return getData()->geometry[calculateIndex()]; }
    AGX_FORCE_INLINE agx::Physics::GeometryPtr const& EmitterPtr::geometry() const { verifyIndex(); return getData()->geometry[calculateIndex()]; }

    AGX_FORCE_INLINE agx::Physics::CollisionGroupSetPtr& EmitterPtr::collisionGroupSet() { verifyIndex(); return getData()->collisionGroupSet[calculateIndex()]; }
    AGX_FORCE_INLINE agx::Physics::CollisionGroupSetPtr const& EmitterPtr::collisionGroupSet() const { verifyIndex(); return getData()->collisionGroupSet[calculateIndex()]; }

    AGX_FORCE_INLINE agx::Vec3& EmitterPtr::velocity() { verifyIndex(); return getData()->velocity[calculateIndex()]; }
    AGX_FORCE_INLINE agx::Vec3 const& EmitterPtr::velocity() const { verifyIndex(); return getData()->velocity[calculateIndex()]; }

    AGX_FORCE_INLINE agx::Real& EmitterPtr::rate() { verifyIndex(); return getData()->rate[calculateIndex()]; }
    AGX_FORCE_INLINE agx::Real const& EmitterPtr::rate() const { verifyIndex(); return getData()->rate[calculateIndex()]; }

    AGX_FORCE_INLINE agx::Real& EmitterPtr::emittedQuantity() { verifyIndex(); return getData()->emittedQuantity[calculateIndex()]; }
    AGX_FORCE_INLINE agx::Real const& EmitterPtr::emittedQuantity() const { verifyIndex(); return getData()->emittedQuantity[calculateIndex()]; }

    AGX_FORCE_INLINE agx::Real& EmitterPtr::emittedQuantityMaximum() { verifyIndex(); return getData()->emittedQuantityMaximum[calculateIndex()]; }
    AGX_FORCE_INLINE agx::Real const& EmitterPtr::emittedQuantityMaximum() const { verifyIndex(); return getData()->emittedQuantityMaximum[calculateIndex()]; }

    AGX_FORCE_INLINE agx::UInt& EmitterPtr::emittedCount() { verifyIndex(); return getData()->emittedCount[calculateIndex()]; }
    AGX_FORCE_INLINE agx::UInt const& EmitterPtr::emittedCount() const { verifyIndex(); return getData()->emittedCount[calculateIndex()]; }

    AGX_FORCE_INLINE agx::UInt32& EmitterPtr::seed() { verifyIndex(); return getData()->seed[calculateIndex()]; }
    AGX_FORCE_INLINE agx::UInt32 const& EmitterPtr::seed() const { verifyIndex(); return getData()->seed[calculateIndex()]; }

    AGX_FORCE_INLINE agx::Emitter*& EmitterPtr::model() { verifyIndex(); return getData()->model[calculateIndex()]; }
    AGX_FORCE_INLINE agx::Emitter* const& EmitterPtr::model() const { verifyIndex(); return getData()->model[calculateIndex()]; }

    //-----------------------------------------------------------------------------------------------------
    AGX_FORCE_INLINE EmitterInstance::EmitterInstance() {}
    AGX_FORCE_INLINE EmitterInstance::EmitterInstance(EmitterData* data, agx::Index index) : agx::ReferencedInstance(data, index) {}
    AGX_FORCE_INLINE EmitterInstance::EmitterInstance(agxData::EntityStorage* storage, agx::Index index) : agx::ReferencedInstance(storage, index) {}
    AGX_FORCE_INLINE EmitterInstance::EmitterInstance(const agxData::EntityInstance& other) : agx::ReferencedInstance(other)
    {
      agxAssertN(!other || other.isInstanceOf(EmitterModel::instance()),
        "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
        EntityInstance::getModel()->fullPath().c_str(), EmitterModel::instance()->fullPath().c_str());
    }

    AGX_FORCE_INLINE EmitterInstance::EmitterInstance(const agxData::EntityPtr& ptr) : agx::ReferencedInstance(ptr)
    {
      agxAssertN(!ptr || ptr.isInstanceOf(EmitterModel::instance()),
        "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
        EntityInstance::getModel()->fullPath().c_str(), EmitterModel::instance()->fullPath().c_str());
    }


    AGX_FORCE_INLINE EmitterData* EmitterInstance::getData() { return static_cast<EmitterData* >(agxData::EntityInstance::getData()); }
    AGX_FORCE_INLINE const EmitterData* EmitterInstance::getData() const { return static_cast<const EmitterData* >(agxData::EntityInstance::getData()); }

    AGX_FORCE_INLINE agx::UInt32& EmitterInstance::id() { verifyIndex(); return getData()->id[getIndex()]; }
    AGX_FORCE_INLINE agx::UInt32 const& EmitterInstance::id() const { verifyIndex(); return getData()->id[getIndex()]; }

    AGX_FORCE_INLINE agx::Name& EmitterInstance::name() { verifyIndex(); return getData()->name[getIndex()]; }
    AGX_FORCE_INLINE agx::Name const& EmitterInstance::name() const { verifyIndex(); return getData()->name[getIndex()]; }

    AGX_FORCE_INLINE agx::Physics::GeometryPtr& EmitterInstance::geometry() { verifyIndex(); return getData()->geometry[getIndex()]; }
    AGX_FORCE_INLINE agx::Physics::GeometryPtr const& EmitterInstance::geometry() const { verifyIndex(); return getData()->geometry[getIndex()]; }

    AGX_FORCE_INLINE agx::Physics::CollisionGroupSetPtr& EmitterInstance::collisionGroupSet() { verifyIndex(); return getData()->collisionGroupSet[getIndex()]; }
    AGX_FORCE_INLINE agx::Physics::CollisionGroupSetPtr const& EmitterInstance::collisionGroupSet() const { verifyIndex(); return getData()->collisionGroupSet[getIndex()]; }

    AGX_FORCE_INLINE agx::Vec3& EmitterInstance::velocity() { verifyIndex(); return getData()->velocity[getIndex()]; }
    AGX_FORCE_INLINE agx::Vec3 const& EmitterInstance::velocity() const { verifyIndex(); return getData()->velocity[getIndex()]; }

    AGX_FORCE_INLINE agx::Real& EmitterInstance::rate() { verifyIndex(); return getData()->rate[getIndex()]; }
    AGX_FORCE_INLINE agx::Real const& EmitterInstance::rate() const { verifyIndex(); return getData()->rate[getIndex()]; }

    AGX_FORCE_INLINE agx::Real& EmitterInstance::emittedQuantity() { verifyIndex(); return getData()->emittedQuantity[getIndex()]; }
    AGX_FORCE_INLINE agx::Real const& EmitterInstance::emittedQuantity() const { verifyIndex(); return getData()->emittedQuantity[getIndex()]; }

    AGX_FORCE_INLINE agx::Real& EmitterInstance::emittedQuantityMaximum() { verifyIndex(); return getData()->emittedQuantityMaximum[getIndex()]; }
    AGX_FORCE_INLINE agx::Real const& EmitterInstance::emittedQuantityMaximum() const { verifyIndex(); return getData()->emittedQuantityMaximum[getIndex()]; }

    AGX_FORCE_INLINE agx::UInt& EmitterInstance::emittedCount() { verifyIndex(); return getData()->emittedCount[getIndex()]; }
    AGX_FORCE_INLINE agx::UInt const& EmitterInstance::emittedCount() const { verifyIndex(); return getData()->emittedCount[getIndex()]; }

    AGX_FORCE_INLINE agx::UInt32& EmitterInstance::seed() { verifyIndex(); return getData()->seed[getIndex()]; }
    AGX_FORCE_INLINE agx::UInt32 const& EmitterInstance::seed() const { verifyIndex(); return getData()->seed[getIndex()]; }

    AGX_FORCE_INLINE agx::Emitter*& EmitterInstance::model() { verifyIndex(); return getData()->model[getIndex()]; }
    AGX_FORCE_INLINE agx::Emitter* const& EmitterInstance::model() const { verifyIndex(); return getData()->model[getIndex()]; }

    //-----------------------------------------------------------------------------------------------------
    AGX_FORCE_INLINE EmitterSemantics::EmitterSemantics() {}
    //-----------------------------------------------------------------------------------------------------
    DOXYGEN_END_INTERNAL_BLOCK()
  }
}

AGX_TYPE_BINDING(agx::Physics::EmitterPtr, "Physics.EmitterPtr")
AGX_TYPE_BINDING(agx::Physics::EmitterInstance, "Physics.EmitterInstance")

#ifdef _MSC_VER
# pragma warning(pop)
#endif

#undef AGX_ENTITY_WRAPPER
#undef AGX_ENTITY_NAMESPACE
#endif

