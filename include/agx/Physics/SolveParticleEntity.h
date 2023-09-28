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

#ifndef GENERATED_AGX_PHYSICS_SOLVEPARTICLE_H_PLUGIN
#define GENERATED_AGX_PHYSICS_SOLVEPARTICLE_H_PLUGIN

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
#include <agx/Vec3.h>
#include <agx/Real.h>
#include <agx/Integer.h>


namespace agx
{
  namespace Physics
  {

    class SolveParticleModel;
    class SolveParticleData;
    class SolveParticlePtr;
    class SolveParticleInstance;
    class SolveParticleSemantics;


    AGX_DECLARE_POINTER_TYPES(SolveParticleModel);

    /** 
    Abstract description of the data attributes for the Physics.SolveParticle entity.
    */ 
    class AGXPHYSICS_EXPORT SolveParticleModel : public agxData::EntityModel
    {
    public:
      typedef SolveParticlePtr PtrT;

      SolveParticleModel(const agx::String& name = "SolveParticle");

      /// \return The entity model singleton.
      static SolveParticleModel* instance();

      /// Create and return a pointer to a new instance in the default storage for this entity model.
      static SolveParticlePtr createInstance();

      /// \return The default storage for this entity model.
      static agxData::EntityStorage* defaultStorage();

      /// This is part of internal cleanup and should not be called by users
      virtual void shutdownCleanup() override;



      /* Attributes */
      static agxData::ScalarAttributeT< agx::Vec3 >* positionAttribute;
      static agxData::ScalarAttributeT< agx::Vec3 >* velocityAttribute;
      static agxData::ScalarAttributeT< agx::Vec3 >* forceAttribute;
      static agxData::ScalarAttributeT< agx::Real >* densityAttribute;
      static agxData::ScalarAttributeT< agx::UInt32 >* sourceIndexAttribute;
      static agxData::ScalarAttributeT< agx::UInt32 >* solveIndexAttribute;
      static agxData::ArrayAttributeT< agx::UInt32 >* contactListAttribute;

    protected:
      virtual ~SolveParticleModel();
      virtual agxData::EntityData* createData(agxData::EntityStorage* storage) override;
      virtual void configure(agx::TiXmlElement* eEntity) override;
      virtual void initAttributeAccessors() override;
      virtual void construct(agxData::EntityPtr instance) override;
      void construct(agx::Physics::SolveParticlePtr solveParticle);
    };


    DOXYGEN_START_INTERNAL_BLOCK()
    #ifndef AGX_PHYSICS_SOLVEPARTICLE_DATA_SET_OVERRIDE
    #define AGX_PHYSICS_SOLVEPARTICLE_DATA_SET
    class AGXPHYSICS_EXPORT SolveParticleData : public agxData::EntityData
    {
    public:
      SolveParticleInstance operator[] (size_t index);

    public:
      agxData::Array< SolveParticlePtr >& instance;
      agxData::Array< agx::Vec3 > position;
      agxData::Array< agx::Vec3 > velocity;
      agxData::Array< agx::Vec3 > force;
      agxData::Array< agx::Real > density;
      agxData::Array< agx::UInt32 > sourceIndex;
      agxData::Array< agx::UInt32 > solveIndex;
      agxData::Array< agxData::Array< agx::UInt32 > > contactList;

    public:
      typedef agx::Vec3 positionType;
      typedef agx::Vec3 velocityType;
      typedef agx::Vec3 forceType;
      typedef agx::Real densityType;
      typedef agx::UInt32 sourceIndexType;
      typedef agx::UInt32 solveIndexType;
      typedef agxData::Array< agx::UInt32 > contactListType;

    public:
      SolveParticleData(agxData::EntityStorage* storage);
      SolveParticleData();

    protected:
      virtual ~SolveParticleData() {}
      virtual void setNumElements(agx::Index numElements) override;

    private:
      SolveParticleData& operator= (const SolveParticleData&) { return *this; }

    };
    #endif
    DOXYGEN_END_INTERNAL_BLOCK()


    DOXYGEN_START_INTERNAL_BLOCK()
    class AGXPHYSICS_EXPORT SolveParticleSemantics : protected agxData::EntityPtr
    {
    public:

      // Automatic getters
      agx::Vec3 const& getPosition() const;
      agx::Vec3 const& getVelocity() const;
      agx::Vec3 const& getForce() const;
      agx::Real const& getDensity() const;
      agx::UInt32 const& getSourceIndex() const;
      agx::UInt32 const& getSolveIndex() const;
      agxData::Array< agx::UInt32 > const& getContactList() const;

      // Semantics defined by explicit kernels

      // Automatic setters
      void setPosition(agx::Vec3 const& value);
      void setVelocity(agx::Vec3 const& value);
      void setForce(agx::Vec3 const& value);
      void setDensity(agx::Real const& value);
      void setSourceIndex(agx::UInt32 const& value);
      void setSolveIndex(agx::UInt32 const& value);
      void setContactList(agxData::Array< agx::UInt32 > const& value);


    protected:
      friend class SolveParticlePtr;
      friend class SolveParticleInstance;
      SolveParticleSemantics();
    };
    DOXYGEN_END_INTERNAL_BLOCK()


    /**
    Pointer to a entity instance of type Physics.SolveParticle
    */
    class CALLABLE SolveParticlePtr : public agxData::EntityPtr
    {
    public:
      typedef SolveParticleModel ModelType;
      typedef SolveParticleData DataType;
      typedef SolveParticleInstance InstanceType;

    public:
      AGXPHYSICS_EXPORT SolveParticlePtr();
      AGXPHYSICS_EXPORT SolveParticlePtr(agxData::EntityStorage* storage, agx::Index id);
      AGXPHYSICS_EXPORT SolveParticlePtr(const agxData::EntityPtr& ptr);
      AGXPHYSICS_EXPORT SolveParticlePtr(const agxData::EntityInstance& instance);
      AGXPHYSICS_EXPORT SolveParticlePtr& operator= (const agxData::EntityPtr& ptr);
      AGXPHYSICS_EXPORT SolveParticlePtr& operator= (const agxData::EntityInstance& instance);
      AGXPHYSICS_EXPORT SolveParticleInstance instance();
      AGXPHYSICS_EXPORT const SolveParticleInstance instance() const;

      AGXPHYSICS_EXPORT SolveParticleSemantics* operator->();
      AGXPHYSICS_EXPORT const SolveParticleSemantics* operator->() const;

      SolveParticleData* getData();
      const SolveParticleData* getData() const;


      /// \return reference to the position attribute
      AGXPHYSICS_EXPORT agx::Vec3& position();
      /// \return const reference to the position attribute
      AGXPHYSICS_EXPORT agx::Vec3 const& position() const;

      /// \return reference to the velocity attribute
      AGXPHYSICS_EXPORT agx::Vec3& velocity();
      /// \return const reference to the velocity attribute
      AGXPHYSICS_EXPORT agx::Vec3 const& velocity() const;

      /// \return reference to the force attribute
      AGXPHYSICS_EXPORT agx::Vec3& force();
      /// \return const reference to the force attribute
      AGXPHYSICS_EXPORT agx::Vec3 const& force() const;

      /// \return reference to the density attribute
      AGXPHYSICS_EXPORT agx::Real& density();
      /// \return const reference to the density attribute
      AGXPHYSICS_EXPORT agx::Real const& density() const;

      /// \return reference to the sourceIndex attribute
      AGXPHYSICS_EXPORT agx::UInt32& sourceIndex();
      /// \return const reference to the sourceIndex attribute
      AGXPHYSICS_EXPORT agx::UInt32 const& sourceIndex() const;

      /// \return reference to the solveIndex attribute
      AGXPHYSICS_EXPORT agx::UInt32& solveIndex();
      /// \return const reference to the solveIndex attribute
      AGXPHYSICS_EXPORT agx::UInt32 const& solveIndex() const;

      /// \return reference to the contactList attribute
      AGXPHYSICS_EXPORT agxData::Array< agx::UInt32 >& contactList();
      /// \return const reference to the contactList attribute
      AGXPHYSICS_EXPORT agxData::Array< agx::UInt32 > const& contactList() const;

      DOXYGEN_START_INTERNAL_BLOCK()
      // Only used by tolua!
      #ifdef AGX_GENERATED_TOLUA_WRAPPER
      inline agx::Vec3 const& getPosition() const { return (*this)->getPosition(); }
      inline agx::Vec3 const& getVelocity() const { return (*this)->getVelocity(); }
      inline agx::Vec3 const& getForce() const { return (*this)->getForce(); }
      inline agx::Real const& getDensity() const { return (*this)->getDensity(); }
      inline agx::UInt32 const& getSourceIndex() const { return (*this)->getSourceIndex(); }
      inline agx::UInt32 const& getSolveIndex() const { return (*this)->getSolveIndex(); }
      inline agxData::Array< agx::UInt32 > const& getContactList() const { return (*this)->getContactList(); }


      inline void setPosition(agx::Vec3 const& value) { (*this)->setPosition(value); }
      inline void setVelocity(agx::Vec3 const& value) { (*this)->setVelocity(value); }
      inline void setForce(agx::Vec3 const& value) { (*this)->setForce(value); }
      inline void setDensity(agx::Real const& value) { (*this)->setDensity(value); }
      inline void setSourceIndex(agx::UInt32 const& value) { (*this)->setSourceIndex(value); }
      inline void setSolveIndex(agx::UInt32 const& value) { (*this)->setSolveIndex(value); }
      inline void setContactList(agxData::Array< agx::UInt32 > const& value) { (*this)->setContactList(value); }
      #endif
      DOXYGEN_END_INTERNAL_BLOCK()

    };


    DOXYGEN_START_INTERNAL_BLOCK()
    class AGXPHYSICS_EXPORT SolveParticleInstance : public agxData::EntityInstance
    {
    public:
      SolveParticleInstance();
      SolveParticleInstance(SolveParticleData* data, agx::Index index);
      SolveParticleInstance(agxData::EntityStorage *storage, agx::Index index);
      SolveParticleInstance(const agxData::EntityInstance& other);
      SolveParticleInstance(const agxData::EntityPtr& ptr);

      SolveParticleData* getData();
      const SolveParticleData* getData() const;

    public:
      /// \return reference to the position attribute
      agx::Vec3& position();
      /// \return const reference to the position attribute
      agx::Vec3 const& position() const;

      /// \return reference to the velocity attribute
      agx::Vec3& velocity();
      /// \return const reference to the velocity attribute
      agx::Vec3 const& velocity() const;

      /// \return reference to the force attribute
      agx::Vec3& force();
      /// \return const reference to the force attribute
      agx::Vec3 const& force() const;

      /// \return reference to the density attribute
      agx::Real& density();
      /// \return const reference to the density attribute
      agx::Real const& density() const;

      /// \return reference to the sourceIndex attribute
      agx::UInt32& sourceIndex();
      /// \return const reference to the sourceIndex attribute
      agx::UInt32 const& sourceIndex() const;

      /// \return reference to the solveIndex attribute
      agx::UInt32& solveIndex();
      /// \return const reference to the solveIndex attribute
      agx::UInt32 const& solveIndex() const;

      /// \return reference to the contactList attribute
      agxData::Array< agx::UInt32 >& contactList();
      /// \return const reference to the contactList attribute
      agxData::Array< agx::UInt32 > const& contactList() const;

    };
    DOXYGEN_END_INTERNAL_BLOCK()



    typedef agx::VectorPOD<SolveParticlePtr> SolveParticlePtrVector;
    typedef agxData::Array<SolveParticlePtr> SolveParticlePtrArray;



    DOXYGEN_START_INTERNAL_BLOCK()
    /* Implementation */
    //-----------------------------------------------------------------------------------------------------
    //-----------------------------------------------------------------------------------------------------
    inline SolveParticleInstance agx::Physics::SolveParticleData::operator[] (size_t index) { return SolveParticleInstance(this, (agx::Index)index); }
    //-----------------------------------------------------------------------------------------------------
    AGX_FORCE_INLINE SolveParticlePtr::SolveParticlePtr() {}
    AGX_FORCE_INLINE SolveParticlePtr::SolveParticlePtr(agxData::EntityStorage* storage, agx::Index id) : agxData::EntityPtr(storage, id) {}
    AGX_FORCE_INLINE SolveParticlePtr::SolveParticlePtr(const agxData::EntityPtr& ptr) : agxData::EntityPtr(ptr)
    {
      agxAssertN(!ptr || ptr.isInstanceOf(SolveParticleModel::instance()),
        "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
        EntityPtr::getModel()->fullPath().c_str(), SolveParticleModel::instance()->fullPath().c_str());
    }

    AGX_FORCE_INLINE SolveParticlePtr::SolveParticlePtr(const agxData::EntityInstance& instance) : agxData::EntityPtr(instance)
    {
      agxAssertN(!instance || instance.isInstanceOf(SolveParticleModel::instance()),
        "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
        EntityPtr::getModel()->fullPath().c_str(), SolveParticleModel::instance()->fullPath().c_str());
    }

    AGX_FORCE_INLINE SolveParticlePtr& SolveParticlePtr::operator= (const agxData::EntityPtr& ptr)
    {
      agxData::EntityPtr::operator= (ptr);
      agxAssertN(!ptr || ptr.isInstanceOf(SolveParticleModel::instance()),
        "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
        EntityPtr::getModel()->fullPath().c_str(), SolveParticleModel::instance()->fullPath().c_str());
      return *this;
    }

    AGX_FORCE_INLINE SolveParticlePtr& SolveParticlePtr::operator= (const agxData::EntityInstance& instance)
    {
      agxData::EntityPtr::operator= (instance);
      agxAssertN(!instance || instance.isInstanceOf(SolveParticleModel::instance()),
        "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
        EntityPtr::getModel()->fullPath().c_str(), SolveParticleModel::instance()->fullPath().c_str());
      return *this;
    }

    inline SolveParticleInstance SolveParticlePtr::instance() { return agxData::EntityPtr::instance(); }
    inline const SolveParticleInstance SolveParticlePtr::instance() const { return agxData::EntityPtr::instance(); }
    AGX_FORCE_INLINE SolveParticleSemantics* SolveParticlePtr::operator->() { return (SolveParticleSemantics* )this; }
    AGX_FORCE_INLINE const SolveParticleSemantics* SolveParticlePtr::operator->() const { return (const SolveParticleSemantics* )this; }
    AGX_FORCE_INLINE SolveParticleData* SolveParticlePtr::getData() { return static_cast<SolveParticleData* >(agxData::EntityPtr::getData()); }
    AGX_FORCE_INLINE const SolveParticleData* SolveParticlePtr::getData() const { return static_cast<const SolveParticleData* >(agxData::EntityPtr::getData()); }

    AGX_FORCE_INLINE agx::Vec3& SolveParticlePtr::position() { verifyIndex(); return getData()->position[calculateIndex()]; }
    AGX_FORCE_INLINE agx::Vec3 const& SolveParticlePtr::position() const { verifyIndex(); return getData()->position[calculateIndex()]; }

    AGX_FORCE_INLINE agx::Vec3& SolveParticlePtr::velocity() { verifyIndex(); return getData()->velocity[calculateIndex()]; }
    AGX_FORCE_INLINE agx::Vec3 const& SolveParticlePtr::velocity() const { verifyIndex(); return getData()->velocity[calculateIndex()]; }

    AGX_FORCE_INLINE agx::Vec3& SolveParticlePtr::force() { verifyIndex(); return getData()->force[calculateIndex()]; }
    AGX_FORCE_INLINE agx::Vec3 const& SolveParticlePtr::force() const { verifyIndex(); return getData()->force[calculateIndex()]; }

    AGX_FORCE_INLINE agx::Real& SolveParticlePtr::density() { verifyIndex(); return getData()->density[calculateIndex()]; }
    AGX_FORCE_INLINE agx::Real const& SolveParticlePtr::density() const { verifyIndex(); return getData()->density[calculateIndex()]; }

    AGX_FORCE_INLINE agx::UInt32& SolveParticlePtr::sourceIndex() { verifyIndex(); return getData()->sourceIndex[calculateIndex()]; }
    AGX_FORCE_INLINE agx::UInt32 const& SolveParticlePtr::sourceIndex() const { verifyIndex(); return getData()->sourceIndex[calculateIndex()]; }

    AGX_FORCE_INLINE agx::UInt32& SolveParticlePtr::solveIndex() { verifyIndex(); return getData()->solveIndex[calculateIndex()]; }
    AGX_FORCE_INLINE agx::UInt32 const& SolveParticlePtr::solveIndex() const { verifyIndex(); return getData()->solveIndex[calculateIndex()]; }

    AGX_FORCE_INLINE agxData::Array< agx::UInt32 >& SolveParticlePtr::contactList() { verifyIndex(); return getData()->contactList[calculateIndex()]; }
    AGX_FORCE_INLINE agxData::Array< agx::UInt32 > const& SolveParticlePtr::contactList() const { verifyIndex(); return getData()->contactList[calculateIndex()]; }

    //-----------------------------------------------------------------------------------------------------
    AGX_FORCE_INLINE SolveParticleInstance::SolveParticleInstance() {}
    AGX_FORCE_INLINE SolveParticleInstance::SolveParticleInstance(SolveParticleData* data, agx::Index index) : agxData::EntityInstance(data, index) {}
    AGX_FORCE_INLINE SolveParticleInstance::SolveParticleInstance(agxData::EntityStorage* storage, agx::Index index) : agxData::EntityInstance(storage, index) {}
    AGX_FORCE_INLINE SolveParticleInstance::SolveParticleInstance(const agxData::EntityInstance& other) : agxData::EntityInstance(other)
    {
      agxAssertN(!other || other.isInstanceOf(SolveParticleModel::instance()),
        "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
        EntityInstance::getModel()->fullPath().c_str(), SolveParticleModel::instance()->fullPath().c_str());
    }

    AGX_FORCE_INLINE SolveParticleInstance::SolveParticleInstance(const agxData::EntityPtr& ptr) : agxData::EntityInstance(ptr)
    {
      agxAssertN(!ptr || ptr.isInstanceOf(SolveParticleModel::instance()),
        "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
        EntityInstance::getModel()->fullPath().c_str(), SolveParticleModel::instance()->fullPath().c_str());
    }


    AGX_FORCE_INLINE SolveParticleData* SolveParticleInstance::getData() { return static_cast<SolveParticleData* >(agxData::EntityInstance::getData()); }
    AGX_FORCE_INLINE const SolveParticleData* SolveParticleInstance::getData() const { return static_cast<const SolveParticleData* >(agxData::EntityInstance::getData()); }

    AGX_FORCE_INLINE agx::Vec3& SolveParticleInstance::position() { verifyIndex(); return getData()->position[getIndex()]; }
    AGX_FORCE_INLINE agx::Vec3 const& SolveParticleInstance::position() const { verifyIndex(); return getData()->position[getIndex()]; }

    AGX_FORCE_INLINE agx::Vec3& SolveParticleInstance::velocity() { verifyIndex(); return getData()->velocity[getIndex()]; }
    AGX_FORCE_INLINE agx::Vec3 const& SolveParticleInstance::velocity() const { verifyIndex(); return getData()->velocity[getIndex()]; }

    AGX_FORCE_INLINE agx::Vec3& SolveParticleInstance::force() { verifyIndex(); return getData()->force[getIndex()]; }
    AGX_FORCE_INLINE agx::Vec3 const& SolveParticleInstance::force() const { verifyIndex(); return getData()->force[getIndex()]; }

    AGX_FORCE_INLINE agx::Real& SolveParticleInstance::density() { verifyIndex(); return getData()->density[getIndex()]; }
    AGX_FORCE_INLINE agx::Real const& SolveParticleInstance::density() const { verifyIndex(); return getData()->density[getIndex()]; }

    AGX_FORCE_INLINE agx::UInt32& SolveParticleInstance::sourceIndex() { verifyIndex(); return getData()->sourceIndex[getIndex()]; }
    AGX_FORCE_INLINE agx::UInt32 const& SolveParticleInstance::sourceIndex() const { verifyIndex(); return getData()->sourceIndex[getIndex()]; }

    AGX_FORCE_INLINE agx::UInt32& SolveParticleInstance::solveIndex() { verifyIndex(); return getData()->solveIndex[getIndex()]; }
    AGX_FORCE_INLINE agx::UInt32 const& SolveParticleInstance::solveIndex() const { verifyIndex(); return getData()->solveIndex[getIndex()]; }

    AGX_FORCE_INLINE agxData::Array< agx::UInt32 >& SolveParticleInstance::contactList() { verifyIndex(); return getData()->contactList[getIndex()]; }
    AGX_FORCE_INLINE agxData::Array< agx::UInt32 > const& SolveParticleInstance::contactList() const { verifyIndex(); return getData()->contactList[getIndex()]; }

    //-----------------------------------------------------------------------------------------------------
    AGX_FORCE_INLINE SolveParticleSemantics::SolveParticleSemantics() {}
    //-----------------------------------------------------------------------------------------------------
    DOXYGEN_END_INTERNAL_BLOCK()
  }
}

AGX_TYPE_BINDING(agx::Physics::SolveParticlePtr, "Physics.SolveParticlePtr")
AGX_TYPE_BINDING(agx::Physics::SolveParticleInstance, "Physics.SolveParticleInstance")

#ifdef _MSC_VER
# pragma warning(pop)
#endif

#undef AGX_ENTITY_WRAPPER
#undef AGX_ENTITY_NAMESPACE
#endif

