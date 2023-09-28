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

#ifndef GENERATED_AGX_PHYSICS_GRANULARBODY_H_PLUGIN
#define GENERATED_AGX_PHYSICS_GRANULARBODY_H_PLUGIN

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
#include <agx/Physics/ParticleEntity.h>
#include <agx/Vec3.h>
#include <agx/Quat.h>
#include <agx/Real.h>


namespace agx
{
  namespace Physics
  {

    class GranularBodyModel;
    class GranularBodyData;
    class GranularBodyPtr;
    class GranularBodyInstance;
    class GranularBodySemantics;


    AGX_DECLARE_POINTER_TYPES(GranularBodyModel);

    /** 
    Abstract description of the data attributes for the Physics.GranularBody entity.
    */ 
    class AGXPHYSICS_EXPORT GranularBodyModel : public agx::Physics::ParticleModel
    {
    public:
      typedef GranularBodyPtr PtrT;

      GranularBodyModel(const agx::String& name = "GranularBody");

      /// \return The entity model singleton.
      static GranularBodyModel* instance();

      /// Create and return a pointer to a new instance in the default storage for this entity model.
      static GranularBodyPtr createInstance();

      /// \return The default storage for this entity model.
      static agxData::EntityStorage* defaultStorage();

      /// This is part of internal cleanup and should not be called by users
      virtual void shutdownCleanup() override;



      /* Attributes */
      static agxData::ScalarAttributeT< agx::Vec3 >* angularVelocityAttribute;
      static agxData::ScalarAttributeT< agx::Vec3 >* oldAngularVelocityAttribute;
      static agxData::ScalarAttributeT< agx::Quat >* rotationAttribute;
      static agxData::ScalarAttributeT< agx::Vec3 >* torqueAttribute;
      static agxData::ScalarAttributeT< agx::Real >* inertiaAttribute;
      static agxData::ScalarAttributeT< agx::Real >* invInertiaAttribute;

    protected:
      virtual ~GranularBodyModel();
      virtual agxData::EntityData* createData(agxData::EntityStorage* storage) override;
      virtual void configure(agx::TiXmlElement* eEntity) override;
      virtual void initAttributeAccessors() override;
      virtual void construct(agxData::EntityPtr instance) override;
      void construct(agx::Physics::GranularBodyPtr granularBody);
    };


    DOXYGEN_START_INTERNAL_BLOCK()
    #ifndef AGX_PHYSICS_GRANULARBODY_DATA_SET_OVERRIDE
    #define AGX_PHYSICS_GRANULARBODY_DATA_SET
    class AGXPHYSICS_EXPORT GranularBodyData : public agx::Physics::ParticleData
    {
    public:
      GranularBodyInstance operator[] (size_t index);

    public:
      agxData::Array< GranularBodyPtr >& instance;
      agxData::Array< agx::Vec3 > angularVelocity;
      agxData::Array< agx::Vec3 > oldAngularVelocity;
      agxData::Array< agx::Quat > rotation;
      agxData::Array< agx::Vec3 > torque;
      agxData::Array< agx::Real > inertia;
      agxData::Array< agx::Real > invInertia;

    public:
      typedef agx::Vec3 angularVelocityType;
      typedef agx::Vec3 oldAngularVelocityType;
      typedef agx::Quat rotationType;
      typedef agx::Vec3 torqueType;
      typedef agx::Real inertiaType;
      typedef agx::Real invInertiaType;

    public:
      GranularBodyData(agxData::EntityStorage* storage);
      GranularBodyData();

    protected:
      virtual ~GranularBodyData() {}
      virtual void setNumElements(agx::Index numElements) override;

    private:
      GranularBodyData& operator= (const GranularBodyData&) { return *this; }

    };
    #endif
    DOXYGEN_END_INTERNAL_BLOCK()


    DOXYGEN_START_INTERNAL_BLOCK()
    class AGXPHYSICS_EXPORT GranularBodySemantics : public agx::Physics::ParticleSemantics
    {
    public:

      // Automatic getters
      agx::Vec3 const& getAngularVelocity() const;
      agx::Vec3 const& getOldAngularVelocity() const;
      agx::Quat const& getRotation() const;
      agx::Vec3 const& getTorque() const;
      agx::Real const& getInertia() const;
      agx::Real const& getInvInertia() const;

      // Semantics defined by explicit kernels
      void integratePosition(const agx::Real& clock_timeStep);
      void integrateVelocity(const agx::Real& clock_timeStep);
      void setAngularVelocity(const agx::Vec3& angularVelocity);
      void setMass(const agx::Real& mass);
      void setMaterial(const agx::Physics::MaterialPtr& material);
      void setRadius(const agx::Real& radius);

      // Automatic setters
      void setOldAngularVelocity(agx::Vec3 const& value);
      void setRotation(agx::Quat const& value);
      void setTorque(agx::Vec3 const& value);
      void setInertia(agx::Real const& value);
      void setInvInertia(agx::Real const& value);


    protected:
      friend class GranularBodyPtr;
      friend class GranularBodyInstance;
      GranularBodySemantics();
    };
    DOXYGEN_END_INTERNAL_BLOCK()


    /**
    Pointer to a entity instance of type Physics.GranularBody
    */
    class CALLABLE GranularBodyPtr : public agx::Physics::ParticlePtr
    {
    public:
      typedef GranularBodyModel ModelType;
      typedef GranularBodyData DataType;
      typedef GranularBodyInstance InstanceType;

    public:
      AGXPHYSICS_EXPORT GranularBodyPtr();
      AGXPHYSICS_EXPORT GranularBodyPtr(agxData::EntityStorage* storage, agx::Index id);
      AGXPHYSICS_EXPORT GranularBodyPtr(const agxData::EntityPtr& ptr);
      AGXPHYSICS_EXPORT GranularBodyPtr(const agxData::EntityInstance& instance);
      AGXPHYSICS_EXPORT GranularBodyPtr& operator= (const agxData::EntityPtr& ptr);
      AGXPHYSICS_EXPORT GranularBodyPtr& operator= (const agxData::EntityInstance& instance);
      AGXPHYSICS_EXPORT GranularBodyInstance instance();
      AGXPHYSICS_EXPORT const GranularBodyInstance instance() const;

      AGXPHYSICS_EXPORT GranularBodySemantics* operator->();
      AGXPHYSICS_EXPORT const GranularBodySemantics* operator->() const;

      GranularBodyData* getData();
      const GranularBodyData* getData() const;


      /// \return reference to the angularVelocity attribute
      AGXPHYSICS_EXPORT agx::Vec3& angularVelocity();
      /// \return const reference to the angularVelocity attribute
      AGXPHYSICS_EXPORT agx::Vec3 const& angularVelocity() const;

      /// \return reference to the oldAngularVelocity attribute
      AGXPHYSICS_EXPORT agx::Vec3& oldAngularVelocity();
      /// \return const reference to the oldAngularVelocity attribute
      AGXPHYSICS_EXPORT agx::Vec3 const& oldAngularVelocity() const;

      /// \return reference to the rotation attribute
      AGXPHYSICS_EXPORT agx::Quat& rotation();
      /// \return const reference to the rotation attribute
      AGXPHYSICS_EXPORT agx::Quat const& rotation() const;

      /// \return reference to the torque attribute
      AGXPHYSICS_EXPORT agx::Vec3& torque();
      /// \return const reference to the torque attribute
      AGXPHYSICS_EXPORT agx::Vec3 const& torque() const;

      /// \return reference to the inertia attribute
      AGXPHYSICS_EXPORT agx::Real& inertia();
      /// \return const reference to the inertia attribute
      AGXPHYSICS_EXPORT agx::Real const& inertia() const;

      /// \return reference to the invInertia attribute
      AGXPHYSICS_EXPORT agx::Real& invInertia();
      /// \return const reference to the invInertia attribute
      AGXPHYSICS_EXPORT agx::Real const& invInertia() const;

      DOXYGEN_START_INTERNAL_BLOCK()
      // Only used by tolua!
      #ifdef AGX_GENERATED_TOLUA_WRAPPER
      inline agx::Vec3 const& getAngularVelocity() const { return (*this)->getAngularVelocity(); }
      inline agx::Vec3 const& getOldAngularVelocity() const { return (*this)->getOldAngularVelocity(); }
      inline agx::Quat const& getRotation() const { return (*this)->getRotation(); }
      inline agx::Vec3 const& getTorque() const { return (*this)->getTorque(); }
      inline agx::Real const& getInertia() const { return (*this)->getInertia(); }
      inline agx::Real const& getInvInertia() const { return (*this)->getInvInertia(); }

      inline void integratePosition(const agx::Real& clock_timeStep) { (*this)->integratePosition(clock_timeStep); }
      inline void integrateVelocity(const agx::Real& clock_timeStep) { (*this)->integrateVelocity(clock_timeStep); }
      inline void setAngularVelocity(const agx::Vec3& angularVelocity) { (*this)->setAngularVelocity(angularVelocity); }
      inline void setMass(const agx::Real& mass) { (*this)->setMass(mass); }
      inline void setMaterial(const agx::Physics::MaterialPtr& material) { (*this)->setMaterial(material); }
      inline void setRadius(const agx::Real& radius) { (*this)->setRadius(radius); }

      inline void setOldAngularVelocity(agx::Vec3 const& value) { (*this)->setOldAngularVelocity(value); }
      inline void setRotation(agx::Quat const& value) { (*this)->setRotation(value); }
      inline void setTorque(agx::Vec3 const& value) { (*this)->setTorque(value); }
      inline void setInertia(agx::Real const& value) { (*this)->setInertia(value); }
      inline void setInvInertia(agx::Real const& value) { (*this)->setInvInertia(value); }
      #endif
      DOXYGEN_END_INTERNAL_BLOCK()

    };


    DOXYGEN_START_INTERNAL_BLOCK()
    class AGXPHYSICS_EXPORT GranularBodyInstance : public agx::Physics::ParticleInstance
    {
    public:
      GranularBodyInstance();
      GranularBodyInstance(GranularBodyData* data, agx::Index index);
      GranularBodyInstance(agxData::EntityStorage *storage, agx::Index index);
      GranularBodyInstance(const agxData::EntityInstance& other);
      GranularBodyInstance(const agxData::EntityPtr& ptr);

      GranularBodyData* getData();
      const GranularBodyData* getData() const;

    public:
      /// \return reference to the angularVelocity attribute
      agx::Vec3& angularVelocity();
      /// \return const reference to the angularVelocity attribute
      agx::Vec3 const& angularVelocity() const;

      /// \return reference to the oldAngularVelocity attribute
      agx::Vec3& oldAngularVelocity();
      /// \return const reference to the oldAngularVelocity attribute
      agx::Vec3 const& oldAngularVelocity() const;

      /// \return reference to the rotation attribute
      agx::Quat& rotation();
      /// \return const reference to the rotation attribute
      agx::Quat const& rotation() const;

      /// \return reference to the torque attribute
      agx::Vec3& torque();
      /// \return const reference to the torque attribute
      agx::Vec3 const& torque() const;

      /// \return reference to the inertia attribute
      agx::Real& inertia();
      /// \return const reference to the inertia attribute
      agx::Real const& inertia() const;

      /// \return reference to the invInertia attribute
      agx::Real& invInertia();
      /// \return const reference to the invInertia attribute
      agx::Real const& invInertia() const;

    };
    DOXYGEN_END_INTERNAL_BLOCK()



    typedef agx::VectorPOD<GranularBodyPtr> GranularBodyPtrVector;
    typedef agxData::Array<GranularBodyPtr> GranularBodyPtrArray;



    DOXYGEN_START_INTERNAL_BLOCK()
    /* Implementation */
    //-----------------------------------------------------------------------------------------------------
    //-----------------------------------------------------------------------------------------------------
    inline GranularBodyInstance agx::Physics::GranularBodyData::operator[] (size_t index) { return GranularBodyInstance(this, (agx::Index)index); }
    //-----------------------------------------------------------------------------------------------------
    AGX_FORCE_INLINE GranularBodyPtr::GranularBodyPtr() {}
    AGX_FORCE_INLINE GranularBodyPtr::GranularBodyPtr(agxData::EntityStorage* storage, agx::Index id) : agx::Physics::ParticlePtr(storage, id) {}
    AGX_FORCE_INLINE GranularBodyPtr::GranularBodyPtr(const agxData::EntityPtr& ptr) : agx::Physics::ParticlePtr(ptr)
    {
      agxAssertN(!ptr || ptr.isInstanceOf(GranularBodyModel::instance()),
        "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
        EntityPtr::getModel()->fullPath().c_str(), GranularBodyModel::instance()->fullPath().c_str());
    }

    AGX_FORCE_INLINE GranularBodyPtr::GranularBodyPtr(const agxData::EntityInstance& instance) : agx::Physics::ParticlePtr(instance)
    {
      agxAssertN(!instance || instance.isInstanceOf(GranularBodyModel::instance()),
        "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
        EntityPtr::getModel()->fullPath().c_str(), GranularBodyModel::instance()->fullPath().c_str());
    }

    AGX_FORCE_INLINE GranularBodyPtr& GranularBodyPtr::operator= (const agxData::EntityPtr& ptr)
    {
      agxData::EntityPtr::operator= (ptr);
      agxAssertN(!ptr || ptr.isInstanceOf(GranularBodyModel::instance()),
        "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
        EntityPtr::getModel()->fullPath().c_str(), GranularBodyModel::instance()->fullPath().c_str());
      return *this;
    }

    AGX_FORCE_INLINE GranularBodyPtr& GranularBodyPtr::operator= (const agxData::EntityInstance& instance)
    {
      agxData::EntityPtr::operator= (instance);
      agxAssertN(!instance || instance.isInstanceOf(GranularBodyModel::instance()),
        "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
        EntityPtr::getModel()->fullPath().c_str(), GranularBodyModel::instance()->fullPath().c_str());
      return *this;
    }

    inline GranularBodyInstance GranularBodyPtr::instance() { return agxData::EntityPtr::instance(); }
    inline const GranularBodyInstance GranularBodyPtr::instance() const { return agxData::EntityPtr::instance(); }
    AGX_FORCE_INLINE GranularBodySemantics* GranularBodyPtr::operator->() { return (GranularBodySemantics* )this; }
    AGX_FORCE_INLINE const GranularBodySemantics* GranularBodyPtr::operator->() const { return (const GranularBodySemantics* )this; }
    AGX_FORCE_INLINE GranularBodyData* GranularBodyPtr::getData() { return static_cast<GranularBodyData* >(agxData::EntityPtr::getData()); }
    AGX_FORCE_INLINE const GranularBodyData* GranularBodyPtr::getData() const { return static_cast<const GranularBodyData* >(agxData::EntityPtr::getData()); }

    AGX_FORCE_INLINE agx::Vec3& GranularBodyPtr::angularVelocity() { verifyIndex(); return getData()->angularVelocity[calculateIndex()]; }
    AGX_FORCE_INLINE agx::Vec3 const& GranularBodyPtr::angularVelocity() const { verifyIndex(); return getData()->angularVelocity[calculateIndex()]; }

    AGX_FORCE_INLINE agx::Vec3& GranularBodyPtr::oldAngularVelocity() { verifyIndex(); return getData()->oldAngularVelocity[calculateIndex()]; }
    AGX_FORCE_INLINE agx::Vec3 const& GranularBodyPtr::oldAngularVelocity() const { verifyIndex(); return getData()->oldAngularVelocity[calculateIndex()]; }

    AGX_FORCE_INLINE agx::Quat& GranularBodyPtr::rotation() { verifyIndex(); return getData()->rotation[calculateIndex()]; }
    AGX_FORCE_INLINE agx::Quat const& GranularBodyPtr::rotation() const { verifyIndex(); return getData()->rotation[calculateIndex()]; }

    AGX_FORCE_INLINE agx::Vec3& GranularBodyPtr::torque() { verifyIndex(); return getData()->torque[calculateIndex()]; }
    AGX_FORCE_INLINE agx::Vec3 const& GranularBodyPtr::torque() const { verifyIndex(); return getData()->torque[calculateIndex()]; }

    AGX_FORCE_INLINE agx::Real& GranularBodyPtr::inertia() { verifyIndex(); return getData()->inertia[calculateIndex()]; }
    AGX_FORCE_INLINE agx::Real const& GranularBodyPtr::inertia() const { verifyIndex(); return getData()->inertia[calculateIndex()]; }

    AGX_FORCE_INLINE agx::Real& GranularBodyPtr::invInertia() { verifyIndex(); return getData()->invInertia[calculateIndex()]; }
    AGX_FORCE_INLINE agx::Real const& GranularBodyPtr::invInertia() const { verifyIndex(); return getData()->invInertia[calculateIndex()]; }

    //-----------------------------------------------------------------------------------------------------
    AGX_FORCE_INLINE GranularBodyInstance::GranularBodyInstance() {}
    AGX_FORCE_INLINE GranularBodyInstance::GranularBodyInstance(GranularBodyData* data, agx::Index index) : agx::Physics::ParticleInstance(data, index) {}
    AGX_FORCE_INLINE GranularBodyInstance::GranularBodyInstance(agxData::EntityStorage* storage, agx::Index index) : agx::Physics::ParticleInstance(storage, index) {}
    AGX_FORCE_INLINE GranularBodyInstance::GranularBodyInstance(const agxData::EntityInstance& other) : agx::Physics::ParticleInstance(other)
    {
      agxAssertN(!other || other.isInstanceOf(GranularBodyModel::instance()),
        "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
        EntityInstance::getModel()->fullPath().c_str(), GranularBodyModel::instance()->fullPath().c_str());
    }

    AGX_FORCE_INLINE GranularBodyInstance::GranularBodyInstance(const agxData::EntityPtr& ptr) : agx::Physics::ParticleInstance(ptr)
    {
      agxAssertN(!ptr || ptr.isInstanceOf(GranularBodyModel::instance()),
        "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
        EntityInstance::getModel()->fullPath().c_str(), GranularBodyModel::instance()->fullPath().c_str());
    }


    AGX_FORCE_INLINE GranularBodyData* GranularBodyInstance::getData() { return static_cast<GranularBodyData* >(agxData::EntityInstance::getData()); }
    AGX_FORCE_INLINE const GranularBodyData* GranularBodyInstance::getData() const { return static_cast<const GranularBodyData* >(agxData::EntityInstance::getData()); }

    AGX_FORCE_INLINE agx::Vec3& GranularBodyInstance::angularVelocity() { verifyIndex(); return getData()->angularVelocity[getIndex()]; }
    AGX_FORCE_INLINE agx::Vec3 const& GranularBodyInstance::angularVelocity() const { verifyIndex(); return getData()->angularVelocity[getIndex()]; }

    AGX_FORCE_INLINE agx::Vec3& GranularBodyInstance::oldAngularVelocity() { verifyIndex(); return getData()->oldAngularVelocity[getIndex()]; }
    AGX_FORCE_INLINE agx::Vec3 const& GranularBodyInstance::oldAngularVelocity() const { verifyIndex(); return getData()->oldAngularVelocity[getIndex()]; }

    AGX_FORCE_INLINE agx::Quat& GranularBodyInstance::rotation() { verifyIndex(); return getData()->rotation[getIndex()]; }
    AGX_FORCE_INLINE agx::Quat const& GranularBodyInstance::rotation() const { verifyIndex(); return getData()->rotation[getIndex()]; }

    AGX_FORCE_INLINE agx::Vec3& GranularBodyInstance::torque() { verifyIndex(); return getData()->torque[getIndex()]; }
    AGX_FORCE_INLINE agx::Vec3 const& GranularBodyInstance::torque() const { verifyIndex(); return getData()->torque[getIndex()]; }

    AGX_FORCE_INLINE agx::Real& GranularBodyInstance::inertia() { verifyIndex(); return getData()->inertia[getIndex()]; }
    AGX_FORCE_INLINE agx::Real const& GranularBodyInstance::inertia() const { verifyIndex(); return getData()->inertia[getIndex()]; }

    AGX_FORCE_INLINE agx::Real& GranularBodyInstance::invInertia() { verifyIndex(); return getData()->invInertia[getIndex()]; }
    AGX_FORCE_INLINE agx::Real const& GranularBodyInstance::invInertia() const { verifyIndex(); return getData()->invInertia[getIndex()]; }

    //-----------------------------------------------------------------------------------------------------
    AGX_FORCE_INLINE GranularBodySemantics::GranularBodySemantics() {}
    //-----------------------------------------------------------------------------------------------------
    DOXYGEN_END_INTERNAL_BLOCK()
  }
}

AGX_TYPE_BINDING(agx::Physics::GranularBodyPtr, "Physics.GranularBodyPtr")
AGX_TYPE_BINDING(agx::Physics::GranularBodyInstance, "Physics.GranularBodyInstance")

#ifdef _MSC_VER
# pragma warning(pop)
#endif

#undef AGX_ENTITY_WRAPPER
#undef AGX_ENTITY_NAMESPACE
#endif

