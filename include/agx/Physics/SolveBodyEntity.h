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

#ifndef GENERATED_AGX_PHYSICS_SOLVEBODY_H_PLUGIN
#define GENERATED_AGX_PHYSICS_SOLVEBODY_H_PLUGIN

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

    class SolveBodyModel;
    class SolveBodyData;
    class SolveBodyPtr;
    class SolveBodyInstance;
    class SolveBodySemantics;


    AGX_DECLARE_POINTER_TYPES(SolveBodyModel);

    /** 
    Abstract description of the data attributes for the Physics.SolveBody entity.
    */ 
    class AGXPHYSICS_EXPORT SolveBodyModel : public agxData::EntityModel
    {
    public:
      typedef SolveBodyPtr PtrT;

      SolveBodyModel(const agx::String& name = "SolveBody");

      /// \return The entity model singleton.
      static SolveBodyModel* instance();

      /// Create and return a pointer to a new instance in the default storage for this entity model.
      static SolveBodyPtr createInstance();

      /// \return The default storage for this entity model.
      static agxData::EntityStorage* defaultStorage();

      /// This is part of internal cleanup and should not be called by users
      virtual void shutdownCleanup() override;



      /* Attributes */
      static agxData::ScalarAttributeT< agx::Vec3 >* velocityAttribute;
      static agxData::ScalarAttributeT< agx::Vec3 >* angularVelocityAttribute;
      static agxData::ScalarAttributeT< agx::Vec3 >* oldVelocityAttribute;
      static agxData::ScalarAttributeT< agx::Vec3 >* oldAngularVelocityAttribute;
      static agxData::ScalarAttributeT< agx::Real >* invMassAttribute;
      static agxData::ScalarAttributeT< agx::Real >* invInertiaAttribute;
      static agxData::ScalarAttributeT< agx::SpinMutex >* solveMutexAttribute;

    protected:
      virtual ~SolveBodyModel();
      virtual agxData::EntityData* createData(agxData::EntityStorage* storage) override;
      virtual void configure(agx::TiXmlElement* eEntity) override;
      virtual void initAttributeAccessors() override;
      virtual void construct(agxData::EntityPtr instance) override;
      void construct(agx::Physics::SolveBodyPtr solveBody);
    };


    DOXYGEN_START_INTERNAL_BLOCK()
    #ifndef AGX_PHYSICS_SOLVEBODY_DATA_SET_OVERRIDE
    #define AGX_PHYSICS_SOLVEBODY_DATA_SET
    class AGXPHYSICS_EXPORT SolveBodyData : public agxData::EntityData
    {
    public:
      SolveBodyInstance operator[] (size_t index);

    public:
      agxData::Array< SolveBodyPtr >& instance;
      agxData::Array< agx::Vec3 > velocity;
      agxData::Array< agx::Vec3 > angularVelocity;
      agxData::Array< agx::Vec3 > oldVelocity;
      agxData::Array< agx::Vec3 > oldAngularVelocity;
      agxData::Array< agx::Real > invMass;
      agxData::Array< agx::Real > invInertia;
      agxData::Array< agx::SpinMutex > solveMutex;

    public:
      typedef agx::Vec3 velocityType;
      typedef agx::Vec3 angularVelocityType;
      typedef agx::Vec3 oldVelocityType;
      typedef agx::Vec3 oldAngularVelocityType;
      typedef agx::Real invMassType;
      typedef agx::Real invInertiaType;
      typedef agx::SpinMutex solveMutexType;

    public:
      SolveBodyData(agxData::EntityStorage* storage);
      SolveBodyData();

    protected:
      virtual ~SolveBodyData() {}
      virtual void setNumElements(agx::Index numElements) override;

    private:
      SolveBodyData& operator= (const SolveBodyData&) { return *this; }

    };
    #endif
    DOXYGEN_END_INTERNAL_BLOCK()


    DOXYGEN_START_INTERNAL_BLOCK()
    class AGXPHYSICS_EXPORT SolveBodySemantics : protected agxData::EntityPtr
    {
    public:

      // Automatic getters
      agx::Vec3 const& getVelocity() const;
      agx::Vec3 const& getAngularVelocity() const;
      agx::Vec3 const& getOldVelocity() const;
      agx::Vec3 const& getOldAngularVelocity() const;
      agx::Real const& getInvMass() const;
      agx::Real const& getInvInertia() const;
      agx::SpinMutex const& getSolveMutex() const;

      // Semantics defined by explicit kernels

      // Automatic setters
      void setVelocity(agx::Vec3 const& value);
      void setAngularVelocity(agx::Vec3 const& value);
      void setOldVelocity(agx::Vec3 const& value);
      void setOldAngularVelocity(agx::Vec3 const& value);
      void setInvMass(agx::Real const& value);
      void setInvInertia(agx::Real const& value);
      void setSolveMutex(agx::SpinMutex const& value);


    protected:
      friend class SolveBodyPtr;
      friend class SolveBodyInstance;
      SolveBodySemantics();
    };
    DOXYGEN_END_INTERNAL_BLOCK()


    /**
    Pointer to a entity instance of type Physics.SolveBody
    */
    class CALLABLE SolveBodyPtr : public agxData::EntityPtr
    {
    public:
      typedef SolveBodyModel ModelType;
      typedef SolveBodyData DataType;
      typedef SolveBodyInstance InstanceType;

    public:
      AGXPHYSICS_EXPORT SolveBodyPtr();
      AGXPHYSICS_EXPORT SolveBodyPtr(agxData::EntityStorage* storage, agx::Index id);
      AGXPHYSICS_EXPORT SolveBodyPtr(const agxData::EntityPtr& ptr);
      AGXPHYSICS_EXPORT SolveBodyPtr(const agxData::EntityInstance& instance);
      AGXPHYSICS_EXPORT SolveBodyPtr& operator= (const agxData::EntityPtr& ptr);
      AGXPHYSICS_EXPORT SolveBodyPtr& operator= (const agxData::EntityInstance& instance);
      AGXPHYSICS_EXPORT SolveBodyInstance instance();
      AGXPHYSICS_EXPORT const SolveBodyInstance instance() const;

      AGXPHYSICS_EXPORT SolveBodySemantics* operator->();
      AGXPHYSICS_EXPORT const SolveBodySemantics* operator->() const;

      SolveBodyData* getData();
      const SolveBodyData* getData() const;


      /// \return reference to the velocity attribute
      AGXPHYSICS_EXPORT agx::Vec3& velocity();
      /// \return const reference to the velocity attribute
      AGXPHYSICS_EXPORT agx::Vec3 const& velocity() const;

      /// \return reference to the angularVelocity attribute
      AGXPHYSICS_EXPORT agx::Vec3& angularVelocity();
      /// \return const reference to the angularVelocity attribute
      AGXPHYSICS_EXPORT agx::Vec3 const& angularVelocity() const;

      /// \return reference to the oldVelocity attribute
      AGXPHYSICS_EXPORT agx::Vec3& oldVelocity();
      /// \return const reference to the oldVelocity attribute
      AGXPHYSICS_EXPORT agx::Vec3 const& oldVelocity() const;

      /// \return reference to the oldAngularVelocity attribute
      AGXPHYSICS_EXPORT agx::Vec3& oldAngularVelocity();
      /// \return const reference to the oldAngularVelocity attribute
      AGXPHYSICS_EXPORT agx::Vec3 const& oldAngularVelocity() const;

      /// \return reference to the invMass attribute
      AGXPHYSICS_EXPORT agx::Real& invMass();
      /// \return const reference to the invMass attribute
      AGXPHYSICS_EXPORT agx::Real const& invMass() const;

      /// \return reference to the invInertia attribute
      AGXPHYSICS_EXPORT agx::Real& invInertia();
      /// \return const reference to the invInertia attribute
      AGXPHYSICS_EXPORT agx::Real const& invInertia() const;

      /// \return reference to the solveMutex attribute
      AGXPHYSICS_EXPORT agx::SpinMutex& solveMutex();
      /// \return const reference to the solveMutex attribute
      AGXPHYSICS_EXPORT agx::SpinMutex const& solveMutex() const;

      DOXYGEN_START_INTERNAL_BLOCK()
      // Only used by tolua!
      #ifdef AGX_GENERATED_TOLUA_WRAPPER
      inline agx::Vec3 const& getVelocity() const { return (*this)->getVelocity(); }
      inline agx::Vec3 const& getAngularVelocity() const { return (*this)->getAngularVelocity(); }
      inline agx::Vec3 const& getOldVelocity() const { return (*this)->getOldVelocity(); }
      inline agx::Vec3 const& getOldAngularVelocity() const { return (*this)->getOldAngularVelocity(); }
      inline agx::Real const& getInvMass() const { return (*this)->getInvMass(); }
      inline agx::Real const& getInvInertia() const { return (*this)->getInvInertia(); }
      inline agx::SpinMutex const& getSolveMutex() const { return (*this)->getSolveMutex(); }


      inline void setVelocity(agx::Vec3 const& value) { (*this)->setVelocity(value); }
      inline void setAngularVelocity(agx::Vec3 const& value) { (*this)->setAngularVelocity(value); }
      inline void setOldVelocity(agx::Vec3 const& value) { (*this)->setOldVelocity(value); }
      inline void setOldAngularVelocity(agx::Vec3 const& value) { (*this)->setOldAngularVelocity(value); }
      inline void setInvMass(agx::Real const& value) { (*this)->setInvMass(value); }
      inline void setInvInertia(agx::Real const& value) { (*this)->setInvInertia(value); }
      inline void setSolveMutex(agx::SpinMutex const& value) { (*this)->setSolveMutex(value); }
      #endif
      DOXYGEN_END_INTERNAL_BLOCK()

    };


    DOXYGEN_START_INTERNAL_BLOCK()
    class AGXPHYSICS_EXPORT SolveBodyInstance : public agxData::EntityInstance
    {
    public:
      SolveBodyInstance();
      SolveBodyInstance(SolveBodyData* data, agx::Index index);
      SolveBodyInstance(agxData::EntityStorage *storage, agx::Index index);
      SolveBodyInstance(const agxData::EntityInstance& other);
      SolveBodyInstance(const agxData::EntityPtr& ptr);

      SolveBodyData* getData();
      const SolveBodyData* getData() const;

    public:
      /// \return reference to the velocity attribute
      agx::Vec3& velocity();
      /// \return const reference to the velocity attribute
      agx::Vec3 const& velocity() const;

      /// \return reference to the angularVelocity attribute
      agx::Vec3& angularVelocity();
      /// \return const reference to the angularVelocity attribute
      agx::Vec3 const& angularVelocity() const;

      /// \return reference to the oldVelocity attribute
      agx::Vec3& oldVelocity();
      /// \return const reference to the oldVelocity attribute
      agx::Vec3 const& oldVelocity() const;

      /// \return reference to the oldAngularVelocity attribute
      agx::Vec3& oldAngularVelocity();
      /// \return const reference to the oldAngularVelocity attribute
      agx::Vec3 const& oldAngularVelocity() const;

      /// \return reference to the invMass attribute
      agx::Real& invMass();
      /// \return const reference to the invMass attribute
      agx::Real const& invMass() const;

      /// \return reference to the invInertia attribute
      agx::Real& invInertia();
      /// \return const reference to the invInertia attribute
      agx::Real const& invInertia() const;

      /// \return reference to the solveMutex attribute
      agx::SpinMutex& solveMutex();
      /// \return const reference to the solveMutex attribute
      agx::SpinMutex const& solveMutex() const;

    };
    DOXYGEN_END_INTERNAL_BLOCK()



    typedef agx::VectorPOD<SolveBodyPtr> SolveBodyPtrVector;
    typedef agxData::Array<SolveBodyPtr> SolveBodyPtrArray;



    DOXYGEN_START_INTERNAL_BLOCK()
    /* Implementation */
    //-----------------------------------------------------------------------------------------------------
    //-----------------------------------------------------------------------------------------------------
    inline SolveBodyInstance agx::Physics::SolveBodyData::operator[] (size_t index) { return SolveBodyInstance(this, (agx::Index)index); }
    //-----------------------------------------------------------------------------------------------------
    AGX_FORCE_INLINE SolveBodyPtr::SolveBodyPtr() {}
    AGX_FORCE_INLINE SolveBodyPtr::SolveBodyPtr(agxData::EntityStorage* storage, agx::Index id) : agxData::EntityPtr(storage, id) {}
    AGX_FORCE_INLINE SolveBodyPtr::SolveBodyPtr(const agxData::EntityPtr& ptr) : agxData::EntityPtr(ptr)
    {
      agxAssertN(!ptr || ptr.isInstanceOf(SolveBodyModel::instance()),
        "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
        EntityPtr::getModel()->fullPath().c_str(), SolveBodyModel::instance()->fullPath().c_str());
    }

    AGX_FORCE_INLINE SolveBodyPtr::SolveBodyPtr(const agxData::EntityInstance& instance) : agxData::EntityPtr(instance)
    {
      agxAssertN(!instance || instance.isInstanceOf(SolveBodyModel::instance()),
        "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
        EntityPtr::getModel()->fullPath().c_str(), SolveBodyModel::instance()->fullPath().c_str());
    }

    AGX_FORCE_INLINE SolveBodyPtr& SolveBodyPtr::operator= (const agxData::EntityPtr& ptr)
    {
      agxData::EntityPtr::operator= (ptr);
      agxAssertN(!ptr || ptr.isInstanceOf(SolveBodyModel::instance()),
        "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
        EntityPtr::getModel()->fullPath().c_str(), SolveBodyModel::instance()->fullPath().c_str());
      return *this;
    }

    AGX_FORCE_INLINE SolveBodyPtr& SolveBodyPtr::operator= (const agxData::EntityInstance& instance)
    {
      agxData::EntityPtr::operator= (instance);
      agxAssertN(!instance || instance.isInstanceOf(SolveBodyModel::instance()),
        "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
        EntityPtr::getModel()->fullPath().c_str(), SolveBodyModel::instance()->fullPath().c_str());
      return *this;
    }

    inline SolveBodyInstance SolveBodyPtr::instance() { return agxData::EntityPtr::instance(); }
    inline const SolveBodyInstance SolveBodyPtr::instance() const { return agxData::EntityPtr::instance(); }
    AGX_FORCE_INLINE SolveBodySemantics* SolveBodyPtr::operator->() { return (SolveBodySemantics* )this; }
    AGX_FORCE_INLINE const SolveBodySemantics* SolveBodyPtr::operator->() const { return (const SolveBodySemantics* )this; }
    AGX_FORCE_INLINE SolveBodyData* SolveBodyPtr::getData() { return static_cast<SolveBodyData* >(agxData::EntityPtr::getData()); }
    AGX_FORCE_INLINE const SolveBodyData* SolveBodyPtr::getData() const { return static_cast<const SolveBodyData* >(agxData::EntityPtr::getData()); }

    AGX_FORCE_INLINE agx::Vec3& SolveBodyPtr::velocity() { verifyIndex(); return getData()->velocity[calculateIndex()]; }
    AGX_FORCE_INLINE agx::Vec3 const& SolveBodyPtr::velocity() const { verifyIndex(); return getData()->velocity[calculateIndex()]; }

    AGX_FORCE_INLINE agx::Vec3& SolveBodyPtr::angularVelocity() { verifyIndex(); return getData()->angularVelocity[calculateIndex()]; }
    AGX_FORCE_INLINE agx::Vec3 const& SolveBodyPtr::angularVelocity() const { verifyIndex(); return getData()->angularVelocity[calculateIndex()]; }

    AGX_FORCE_INLINE agx::Vec3& SolveBodyPtr::oldVelocity() { verifyIndex(); return getData()->oldVelocity[calculateIndex()]; }
    AGX_FORCE_INLINE agx::Vec3 const& SolveBodyPtr::oldVelocity() const { verifyIndex(); return getData()->oldVelocity[calculateIndex()]; }

    AGX_FORCE_INLINE agx::Vec3& SolveBodyPtr::oldAngularVelocity() { verifyIndex(); return getData()->oldAngularVelocity[calculateIndex()]; }
    AGX_FORCE_INLINE agx::Vec3 const& SolveBodyPtr::oldAngularVelocity() const { verifyIndex(); return getData()->oldAngularVelocity[calculateIndex()]; }

    AGX_FORCE_INLINE agx::Real& SolveBodyPtr::invMass() { verifyIndex(); return getData()->invMass[calculateIndex()]; }
    AGX_FORCE_INLINE agx::Real const& SolveBodyPtr::invMass() const { verifyIndex(); return getData()->invMass[calculateIndex()]; }

    AGX_FORCE_INLINE agx::Real& SolveBodyPtr::invInertia() { verifyIndex(); return getData()->invInertia[calculateIndex()]; }
    AGX_FORCE_INLINE agx::Real const& SolveBodyPtr::invInertia() const { verifyIndex(); return getData()->invInertia[calculateIndex()]; }

    AGX_FORCE_INLINE agx::SpinMutex& SolveBodyPtr::solveMutex() { verifyIndex(); return getData()->solveMutex[calculateIndex()]; }
    AGX_FORCE_INLINE agx::SpinMutex const& SolveBodyPtr::solveMutex() const { verifyIndex(); return getData()->solveMutex[calculateIndex()]; }

    //-----------------------------------------------------------------------------------------------------
    AGX_FORCE_INLINE SolveBodyInstance::SolveBodyInstance() {}
    AGX_FORCE_INLINE SolveBodyInstance::SolveBodyInstance(SolveBodyData* data, agx::Index index) : agxData::EntityInstance(data, index) {}
    AGX_FORCE_INLINE SolveBodyInstance::SolveBodyInstance(agxData::EntityStorage* storage, agx::Index index) : agxData::EntityInstance(storage, index) {}
    AGX_FORCE_INLINE SolveBodyInstance::SolveBodyInstance(const agxData::EntityInstance& other) : agxData::EntityInstance(other)
    {
      agxAssertN(!other || other.isInstanceOf(SolveBodyModel::instance()),
        "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
        EntityInstance::getModel()->fullPath().c_str(), SolveBodyModel::instance()->fullPath().c_str());
    }

    AGX_FORCE_INLINE SolveBodyInstance::SolveBodyInstance(const agxData::EntityPtr& ptr) : agxData::EntityInstance(ptr)
    {
      agxAssertN(!ptr || ptr.isInstanceOf(SolveBodyModel::instance()),
        "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
        EntityInstance::getModel()->fullPath().c_str(), SolveBodyModel::instance()->fullPath().c_str());
    }


    AGX_FORCE_INLINE SolveBodyData* SolveBodyInstance::getData() { return static_cast<SolveBodyData* >(agxData::EntityInstance::getData()); }
    AGX_FORCE_INLINE const SolveBodyData* SolveBodyInstance::getData() const { return static_cast<const SolveBodyData* >(agxData::EntityInstance::getData()); }

    AGX_FORCE_INLINE agx::Vec3& SolveBodyInstance::velocity() { verifyIndex(); return getData()->velocity[getIndex()]; }
    AGX_FORCE_INLINE agx::Vec3 const& SolveBodyInstance::velocity() const { verifyIndex(); return getData()->velocity[getIndex()]; }

    AGX_FORCE_INLINE agx::Vec3& SolveBodyInstance::angularVelocity() { verifyIndex(); return getData()->angularVelocity[getIndex()]; }
    AGX_FORCE_INLINE agx::Vec3 const& SolveBodyInstance::angularVelocity() const { verifyIndex(); return getData()->angularVelocity[getIndex()]; }

    AGX_FORCE_INLINE agx::Vec3& SolveBodyInstance::oldVelocity() { verifyIndex(); return getData()->oldVelocity[getIndex()]; }
    AGX_FORCE_INLINE agx::Vec3 const& SolveBodyInstance::oldVelocity() const { verifyIndex(); return getData()->oldVelocity[getIndex()]; }

    AGX_FORCE_INLINE agx::Vec3& SolveBodyInstance::oldAngularVelocity() { verifyIndex(); return getData()->oldAngularVelocity[getIndex()]; }
    AGX_FORCE_INLINE agx::Vec3 const& SolveBodyInstance::oldAngularVelocity() const { verifyIndex(); return getData()->oldAngularVelocity[getIndex()]; }

    AGX_FORCE_INLINE agx::Real& SolveBodyInstance::invMass() { verifyIndex(); return getData()->invMass[getIndex()]; }
    AGX_FORCE_INLINE agx::Real const& SolveBodyInstance::invMass() const { verifyIndex(); return getData()->invMass[getIndex()]; }

    AGX_FORCE_INLINE agx::Real& SolveBodyInstance::invInertia() { verifyIndex(); return getData()->invInertia[getIndex()]; }
    AGX_FORCE_INLINE agx::Real const& SolveBodyInstance::invInertia() const { verifyIndex(); return getData()->invInertia[getIndex()]; }

    AGX_FORCE_INLINE agx::SpinMutex& SolveBodyInstance::solveMutex() { verifyIndex(); return getData()->solveMutex[getIndex()]; }
    AGX_FORCE_INLINE agx::SpinMutex const& SolveBodyInstance::solveMutex() const { verifyIndex(); return getData()->solveMutex[getIndex()]; }

    //-----------------------------------------------------------------------------------------------------
    AGX_FORCE_INLINE SolveBodySemantics::SolveBodySemantics() {}
    //-----------------------------------------------------------------------------------------------------
    DOXYGEN_END_INTERNAL_BLOCK()
  }
}

AGX_TYPE_BINDING(agx::Physics::SolveBodyPtr, "Physics.SolveBodyPtr")
AGX_TYPE_BINDING(agx::Physics::SolveBodyInstance, "Physics.SolveBodyInstance")

#ifdef _MSC_VER
# pragma warning(pop)
#endif

#undef AGX_ENTITY_WRAPPER
#undef AGX_ENTITY_NAMESPACE
#endif

