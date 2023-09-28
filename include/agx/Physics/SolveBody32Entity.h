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

#ifndef GENERATED_AGX_PHYSICS_SOLVEBODY32_H_PLUGIN
#define GENERATED_AGX_PHYSICS_SOLVEBODY32_H_PLUGIN

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
#include <agx/SpinMutex.h>


namespace agx
{
  namespace Physics
  {

    class SolveBody32Model;
    class SolveBody32Data;
    class SolveBody32Ptr;
    class SolveBody32Instance;
    class SolveBody32Semantics;


    AGX_DECLARE_POINTER_TYPES(SolveBody32Model);

    /** 
    Abstract description of the data attributes for the Physics.SolveBody32 entity.
    */ 
    class AGXPHYSICS_EXPORT SolveBody32Model : public agxData::EntityModel
    {
    public:
      typedef SolveBody32Ptr PtrT;

      SolveBody32Model(const agx::String& name = "SolveBody32");

      /// \return The entity model singleton.
      static SolveBody32Model* instance();

      /// Create and return a pointer to a new instance in the default storage for this entity model.
      static SolveBody32Ptr createInstance();

      /// \return The default storage for this entity model.
      static agxData::EntityStorage* defaultStorage();

      /// This is part of internal cleanup and should not be called by users
      virtual void shutdownCleanup() override;



      /* Attributes */
      static agxData::ScalarAttributeT< agx::Vec3f >* velocityAttribute;
      static agxData::ScalarAttributeT< agx::Vec3f >* angularVelocityAttribute;
      static agxData::ScalarAttributeT< agx::Vec3f >* oldVelocityAttribute;
      static agxData::ScalarAttributeT< agx::Vec3f >* oldAngularVelocityAttribute;
      static agxData::ScalarAttributeT< agx::Real32 >* invMassAttribute;
      static agxData::ScalarAttributeT< agx::Real32 >* invInertiaAttribute;
      static agxData::ScalarAttributeT< agx::SpinMutex >* solveMutexAttribute;

    protected:
      virtual ~SolveBody32Model();
      virtual agxData::EntityData* createData(agxData::EntityStorage* storage) override;
      virtual void configure(agx::TiXmlElement* eEntity) override;
      virtual void initAttributeAccessors() override;
      virtual void construct(agxData::EntityPtr instance) override;
      void construct(agx::Physics::SolveBody32Ptr solveBody32);
    };


    DOXYGEN_START_INTERNAL_BLOCK()
    #ifndef AGX_PHYSICS_SOLVEBODY32_DATA_SET_OVERRIDE
    #define AGX_PHYSICS_SOLVEBODY32_DATA_SET
    class AGXPHYSICS_EXPORT SolveBody32Data : public agxData::EntityData
    {
    public:
      SolveBody32Instance operator[] (size_t index);

    public:
      agxData::Array< SolveBody32Ptr >& instance;
      agxData::Array< agx::Vec3f > velocity;
      agxData::Array< agx::Vec3f > angularVelocity;
      agxData::Array< agx::Vec3f > oldVelocity;
      agxData::Array< agx::Vec3f > oldAngularVelocity;
      agxData::Array< agx::Real32 > invMass;
      agxData::Array< agx::Real32 > invInertia;
      agxData::Array< agx::SpinMutex > solveMutex;

    public:
      typedef agx::Vec3f velocityType;
      typedef agx::Vec3f angularVelocityType;
      typedef agx::Vec3f oldVelocityType;
      typedef agx::Vec3f oldAngularVelocityType;
      typedef agx::Real32 invMassType;
      typedef agx::Real32 invInertiaType;
      typedef agx::SpinMutex solveMutexType;

    public:
      SolveBody32Data(agxData::EntityStorage* storage);
      SolveBody32Data();

    protected:
      virtual ~SolveBody32Data() {}
      virtual void setNumElements(agx::Index numElements) override;

    private:
      SolveBody32Data& operator= (const SolveBody32Data&) { return *this; }

    };
    #endif
    DOXYGEN_END_INTERNAL_BLOCK()


    DOXYGEN_START_INTERNAL_BLOCK()
    class AGXPHYSICS_EXPORT SolveBody32Semantics : protected agxData::EntityPtr
    {
    public:

      // Automatic getters
      agx::Vec3f const& getVelocity() const;
      agx::Vec3f const& getAngularVelocity() const;
      agx::Vec3f const& getOldVelocity() const;
      agx::Vec3f const& getOldAngularVelocity() const;
      agx::Real32 const& getInvMass() const;
      agx::Real32 const& getInvInertia() const;
      agx::SpinMutex const& getSolveMutex() const;

      // Semantics defined by explicit kernels

      // Automatic setters
      void setVelocity(agx::Vec3f const& value);
      void setAngularVelocity(agx::Vec3f const& value);
      void setOldVelocity(agx::Vec3f const& value);
      void setOldAngularVelocity(agx::Vec3f const& value);
      void setInvMass(agx::Real32 const& value);
      void setInvInertia(agx::Real32 const& value);
      void setSolveMutex(agx::SpinMutex const& value);


    protected:
      friend class SolveBody32Ptr;
      friend class SolveBody32Instance;
      SolveBody32Semantics();
    };
    DOXYGEN_END_INTERNAL_BLOCK()


    /**
    Pointer to a entity instance of type Physics.SolveBody32
    */
    class CALLABLE SolveBody32Ptr : public agxData::EntityPtr
    {
    public:
      typedef SolveBody32Model ModelType;
      typedef SolveBody32Data DataType;
      typedef SolveBody32Instance InstanceType;

    public:
      AGXPHYSICS_EXPORT SolveBody32Ptr();
      AGXPHYSICS_EXPORT SolveBody32Ptr(agxData::EntityStorage* storage, agx::Index id);
      AGXPHYSICS_EXPORT SolveBody32Ptr(const agxData::EntityPtr& ptr);
      AGXPHYSICS_EXPORT SolveBody32Ptr(const agxData::EntityInstance& instance);
      AGXPHYSICS_EXPORT SolveBody32Ptr& operator= (const agxData::EntityPtr& ptr);
      AGXPHYSICS_EXPORT SolveBody32Ptr& operator= (const agxData::EntityInstance& instance);
      AGXPHYSICS_EXPORT SolveBody32Instance instance();
      AGXPHYSICS_EXPORT const SolveBody32Instance instance() const;

      AGXPHYSICS_EXPORT SolveBody32Semantics* operator->();
      AGXPHYSICS_EXPORT const SolveBody32Semantics* operator->() const;

      SolveBody32Data* getData();
      const SolveBody32Data* getData() const;


      /// \return reference to the velocity attribute
      AGXPHYSICS_EXPORT agx::Vec3f& velocity();
      /// \return const reference to the velocity attribute
      AGXPHYSICS_EXPORT agx::Vec3f const& velocity() const;

      /// \return reference to the angularVelocity attribute
      AGXPHYSICS_EXPORT agx::Vec3f& angularVelocity();
      /// \return const reference to the angularVelocity attribute
      AGXPHYSICS_EXPORT agx::Vec3f const& angularVelocity() const;

      /// \return reference to the oldVelocity attribute
      AGXPHYSICS_EXPORT agx::Vec3f& oldVelocity();
      /// \return const reference to the oldVelocity attribute
      AGXPHYSICS_EXPORT agx::Vec3f const& oldVelocity() const;

      /// \return reference to the oldAngularVelocity attribute
      AGXPHYSICS_EXPORT agx::Vec3f& oldAngularVelocity();
      /// \return const reference to the oldAngularVelocity attribute
      AGXPHYSICS_EXPORT agx::Vec3f const& oldAngularVelocity() const;

      /// \return reference to the invMass attribute
      AGXPHYSICS_EXPORT agx::Real32& invMass();
      /// \return const reference to the invMass attribute
      AGXPHYSICS_EXPORT agx::Real32 const& invMass() const;

      /// \return reference to the invInertia attribute
      AGXPHYSICS_EXPORT agx::Real32& invInertia();
      /// \return const reference to the invInertia attribute
      AGXPHYSICS_EXPORT agx::Real32 const& invInertia() const;

      /// \return reference to the solveMutex attribute
      AGXPHYSICS_EXPORT agx::SpinMutex& solveMutex();
      /// \return const reference to the solveMutex attribute
      AGXPHYSICS_EXPORT agx::SpinMutex const& solveMutex() const;

      DOXYGEN_START_INTERNAL_BLOCK()
      // Only used by tolua!
      #ifdef AGX_GENERATED_TOLUA_WRAPPER
      inline agx::Vec3f const& getVelocity() const { return (*this)->getVelocity(); }
      inline agx::Vec3f const& getAngularVelocity() const { return (*this)->getAngularVelocity(); }
      inline agx::Vec3f const& getOldVelocity() const { return (*this)->getOldVelocity(); }
      inline agx::Vec3f const& getOldAngularVelocity() const { return (*this)->getOldAngularVelocity(); }
      inline agx::Real32 const& getInvMass() const { return (*this)->getInvMass(); }
      inline agx::Real32 const& getInvInertia() const { return (*this)->getInvInertia(); }
      inline agx::SpinMutex const& getSolveMutex() const { return (*this)->getSolveMutex(); }


      inline void setVelocity(agx::Vec3f const& value) { (*this)->setVelocity(value); }
      inline void setAngularVelocity(agx::Vec3f const& value) { (*this)->setAngularVelocity(value); }
      inline void setOldVelocity(agx::Vec3f const& value) { (*this)->setOldVelocity(value); }
      inline void setOldAngularVelocity(agx::Vec3f const& value) { (*this)->setOldAngularVelocity(value); }
      inline void setInvMass(agx::Real32 const& value) { (*this)->setInvMass(value); }
      inline void setInvInertia(agx::Real32 const& value) { (*this)->setInvInertia(value); }
      inline void setSolveMutex(agx::SpinMutex const& value) { (*this)->setSolveMutex(value); }
      #endif
      DOXYGEN_END_INTERNAL_BLOCK()

    };


    DOXYGEN_START_INTERNAL_BLOCK()
    class AGXPHYSICS_EXPORT SolveBody32Instance : public agxData::EntityInstance
    {
    public:
      SolveBody32Instance();
      SolveBody32Instance(SolveBody32Data* data, agx::Index index);
      SolveBody32Instance(agxData::EntityStorage *storage, agx::Index index);
      SolveBody32Instance(const agxData::EntityInstance& other);
      SolveBody32Instance(const agxData::EntityPtr& ptr);

      SolveBody32Data* getData();
      const SolveBody32Data* getData() const;

    public:
      /// \return reference to the velocity attribute
      agx::Vec3f& velocity();
      /// \return const reference to the velocity attribute
      agx::Vec3f const& velocity() const;

      /// \return reference to the angularVelocity attribute
      agx::Vec3f& angularVelocity();
      /// \return const reference to the angularVelocity attribute
      agx::Vec3f const& angularVelocity() const;

      /// \return reference to the oldVelocity attribute
      agx::Vec3f& oldVelocity();
      /// \return const reference to the oldVelocity attribute
      agx::Vec3f const& oldVelocity() const;

      /// \return reference to the oldAngularVelocity attribute
      agx::Vec3f& oldAngularVelocity();
      /// \return const reference to the oldAngularVelocity attribute
      agx::Vec3f const& oldAngularVelocity() const;

      /// \return reference to the invMass attribute
      agx::Real32& invMass();
      /// \return const reference to the invMass attribute
      agx::Real32 const& invMass() const;

      /// \return reference to the invInertia attribute
      agx::Real32& invInertia();
      /// \return const reference to the invInertia attribute
      agx::Real32 const& invInertia() const;

      /// \return reference to the solveMutex attribute
      agx::SpinMutex& solveMutex();
      /// \return const reference to the solveMutex attribute
      agx::SpinMutex const& solveMutex() const;

    };
    DOXYGEN_END_INTERNAL_BLOCK()



    typedef agx::VectorPOD<SolveBody32Ptr> SolveBody32PtrVector;
    typedef agxData::Array<SolveBody32Ptr> SolveBody32PtrArray;



    DOXYGEN_START_INTERNAL_BLOCK()
    /* Implementation */
    //-----------------------------------------------------------------------------------------------------
    //-----------------------------------------------------------------------------------------------------
    inline SolveBody32Instance agx::Physics::SolveBody32Data::operator[] (size_t index) { return SolveBody32Instance(this, (agx::Index)index); }
    //-----------------------------------------------------------------------------------------------------
    AGX_FORCE_INLINE SolveBody32Ptr::SolveBody32Ptr() {}
    AGX_FORCE_INLINE SolveBody32Ptr::SolveBody32Ptr(agxData::EntityStorage* storage, agx::Index id) : agxData::EntityPtr(storage, id) {}
    AGX_FORCE_INLINE SolveBody32Ptr::SolveBody32Ptr(const agxData::EntityPtr& ptr) : agxData::EntityPtr(ptr)
    {
      agxAssertN(!ptr || ptr.isInstanceOf(SolveBody32Model::instance()),
        "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
        EntityPtr::getModel()->fullPath().c_str(), SolveBody32Model::instance()->fullPath().c_str());
    }

    AGX_FORCE_INLINE SolveBody32Ptr::SolveBody32Ptr(const agxData::EntityInstance& instance) : agxData::EntityPtr(instance)
    {
      agxAssertN(!instance || instance.isInstanceOf(SolveBody32Model::instance()),
        "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
        EntityPtr::getModel()->fullPath().c_str(), SolveBody32Model::instance()->fullPath().c_str());
    }

    AGX_FORCE_INLINE SolveBody32Ptr& SolveBody32Ptr::operator= (const agxData::EntityPtr& ptr)
    {
      agxData::EntityPtr::operator= (ptr);
      agxAssertN(!ptr || ptr.isInstanceOf(SolveBody32Model::instance()),
        "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
        EntityPtr::getModel()->fullPath().c_str(), SolveBody32Model::instance()->fullPath().c_str());
      return *this;
    }

    AGX_FORCE_INLINE SolveBody32Ptr& SolveBody32Ptr::operator= (const agxData::EntityInstance& instance)
    {
      agxData::EntityPtr::operator= (instance);
      agxAssertN(!instance || instance.isInstanceOf(SolveBody32Model::instance()),
        "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
        EntityPtr::getModel()->fullPath().c_str(), SolveBody32Model::instance()->fullPath().c_str());
      return *this;
    }

    inline SolveBody32Instance SolveBody32Ptr::instance() { return agxData::EntityPtr::instance(); }
    inline const SolveBody32Instance SolveBody32Ptr::instance() const { return agxData::EntityPtr::instance(); }
    AGX_FORCE_INLINE SolveBody32Semantics* SolveBody32Ptr::operator->() { return (SolveBody32Semantics* )this; }
    AGX_FORCE_INLINE const SolveBody32Semantics* SolveBody32Ptr::operator->() const { return (const SolveBody32Semantics* )this; }
    AGX_FORCE_INLINE SolveBody32Data* SolveBody32Ptr::getData() { return static_cast<SolveBody32Data* >(agxData::EntityPtr::getData()); }
    AGX_FORCE_INLINE const SolveBody32Data* SolveBody32Ptr::getData() const { return static_cast<const SolveBody32Data* >(agxData::EntityPtr::getData()); }

    AGX_FORCE_INLINE agx::Vec3f& SolveBody32Ptr::velocity() { verifyIndex(); return getData()->velocity[calculateIndex()]; }
    AGX_FORCE_INLINE agx::Vec3f const& SolveBody32Ptr::velocity() const { verifyIndex(); return getData()->velocity[calculateIndex()]; }

    AGX_FORCE_INLINE agx::Vec3f& SolveBody32Ptr::angularVelocity() { verifyIndex(); return getData()->angularVelocity[calculateIndex()]; }
    AGX_FORCE_INLINE agx::Vec3f const& SolveBody32Ptr::angularVelocity() const { verifyIndex(); return getData()->angularVelocity[calculateIndex()]; }

    AGX_FORCE_INLINE agx::Vec3f& SolveBody32Ptr::oldVelocity() { verifyIndex(); return getData()->oldVelocity[calculateIndex()]; }
    AGX_FORCE_INLINE agx::Vec3f const& SolveBody32Ptr::oldVelocity() const { verifyIndex(); return getData()->oldVelocity[calculateIndex()]; }

    AGX_FORCE_INLINE agx::Vec3f& SolveBody32Ptr::oldAngularVelocity() { verifyIndex(); return getData()->oldAngularVelocity[calculateIndex()]; }
    AGX_FORCE_INLINE agx::Vec3f const& SolveBody32Ptr::oldAngularVelocity() const { verifyIndex(); return getData()->oldAngularVelocity[calculateIndex()]; }

    AGX_FORCE_INLINE agx::Real32& SolveBody32Ptr::invMass() { verifyIndex(); return getData()->invMass[calculateIndex()]; }
    AGX_FORCE_INLINE agx::Real32 const& SolveBody32Ptr::invMass() const { verifyIndex(); return getData()->invMass[calculateIndex()]; }

    AGX_FORCE_INLINE agx::Real32& SolveBody32Ptr::invInertia() { verifyIndex(); return getData()->invInertia[calculateIndex()]; }
    AGX_FORCE_INLINE agx::Real32 const& SolveBody32Ptr::invInertia() const { verifyIndex(); return getData()->invInertia[calculateIndex()]; }

    AGX_FORCE_INLINE agx::SpinMutex& SolveBody32Ptr::solveMutex() { verifyIndex(); return getData()->solveMutex[calculateIndex()]; }
    AGX_FORCE_INLINE agx::SpinMutex const& SolveBody32Ptr::solveMutex() const { verifyIndex(); return getData()->solveMutex[calculateIndex()]; }

    //-----------------------------------------------------------------------------------------------------
    AGX_FORCE_INLINE SolveBody32Instance::SolveBody32Instance() {}
    AGX_FORCE_INLINE SolveBody32Instance::SolveBody32Instance(SolveBody32Data* data, agx::Index index) : agxData::EntityInstance(data, index) {}
    AGX_FORCE_INLINE SolveBody32Instance::SolveBody32Instance(agxData::EntityStorage* storage, agx::Index index) : agxData::EntityInstance(storage, index) {}
    AGX_FORCE_INLINE SolveBody32Instance::SolveBody32Instance(const agxData::EntityInstance& other) : agxData::EntityInstance(other)
    {
      agxAssertN(!other || other.isInstanceOf(SolveBody32Model::instance()),
        "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
        EntityInstance::getModel()->fullPath().c_str(), SolveBody32Model::instance()->fullPath().c_str());
    }

    AGX_FORCE_INLINE SolveBody32Instance::SolveBody32Instance(const agxData::EntityPtr& ptr) : agxData::EntityInstance(ptr)
    {
      agxAssertN(!ptr || ptr.isInstanceOf(SolveBody32Model::instance()),
        "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
        EntityInstance::getModel()->fullPath().c_str(), SolveBody32Model::instance()->fullPath().c_str());
    }


    AGX_FORCE_INLINE SolveBody32Data* SolveBody32Instance::getData() { return static_cast<SolveBody32Data* >(agxData::EntityInstance::getData()); }
    AGX_FORCE_INLINE const SolveBody32Data* SolveBody32Instance::getData() const { return static_cast<const SolveBody32Data* >(agxData::EntityInstance::getData()); }

    AGX_FORCE_INLINE agx::Vec3f& SolveBody32Instance::velocity() { verifyIndex(); return getData()->velocity[getIndex()]; }
    AGX_FORCE_INLINE agx::Vec3f const& SolveBody32Instance::velocity() const { verifyIndex(); return getData()->velocity[getIndex()]; }

    AGX_FORCE_INLINE agx::Vec3f& SolveBody32Instance::angularVelocity() { verifyIndex(); return getData()->angularVelocity[getIndex()]; }
    AGX_FORCE_INLINE agx::Vec3f const& SolveBody32Instance::angularVelocity() const { verifyIndex(); return getData()->angularVelocity[getIndex()]; }

    AGX_FORCE_INLINE agx::Vec3f& SolveBody32Instance::oldVelocity() { verifyIndex(); return getData()->oldVelocity[getIndex()]; }
    AGX_FORCE_INLINE agx::Vec3f const& SolveBody32Instance::oldVelocity() const { verifyIndex(); return getData()->oldVelocity[getIndex()]; }

    AGX_FORCE_INLINE agx::Vec3f& SolveBody32Instance::oldAngularVelocity() { verifyIndex(); return getData()->oldAngularVelocity[getIndex()]; }
    AGX_FORCE_INLINE agx::Vec3f const& SolveBody32Instance::oldAngularVelocity() const { verifyIndex(); return getData()->oldAngularVelocity[getIndex()]; }

    AGX_FORCE_INLINE agx::Real32& SolveBody32Instance::invMass() { verifyIndex(); return getData()->invMass[getIndex()]; }
    AGX_FORCE_INLINE agx::Real32 const& SolveBody32Instance::invMass() const { verifyIndex(); return getData()->invMass[getIndex()]; }

    AGX_FORCE_INLINE agx::Real32& SolveBody32Instance::invInertia() { verifyIndex(); return getData()->invInertia[getIndex()]; }
    AGX_FORCE_INLINE agx::Real32 const& SolveBody32Instance::invInertia() const { verifyIndex(); return getData()->invInertia[getIndex()]; }

    AGX_FORCE_INLINE agx::SpinMutex& SolveBody32Instance::solveMutex() { verifyIndex(); return getData()->solveMutex[getIndex()]; }
    AGX_FORCE_INLINE agx::SpinMutex const& SolveBody32Instance::solveMutex() const { verifyIndex(); return getData()->solveMutex[getIndex()]; }

    //-----------------------------------------------------------------------------------------------------
    AGX_FORCE_INLINE SolveBody32Semantics::SolveBody32Semantics() {}
    //-----------------------------------------------------------------------------------------------------
    DOXYGEN_END_INTERNAL_BLOCK()
  }
}

AGX_TYPE_BINDING(agx::Physics::SolveBody32Ptr, "Physics.SolveBody32Ptr")
AGX_TYPE_BINDING(agx::Physics::SolveBody32Instance, "Physics.SolveBody32Instance")

#ifdef _MSC_VER
# pragma warning(pop)
#endif

#undef AGX_ENTITY_WRAPPER
#undef AGX_ENTITY_NAMESPACE
#endif

