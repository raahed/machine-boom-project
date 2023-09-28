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

#ifndef GENERATED_AGX_PHYSICS_PARTICLEPAIRCONTACT_H_PLUGIN
#define GENERATED_AGX_PHYSICS_PARTICLEPAIRCONTACT_H_PLUGIN

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
#include <agx/Physics/ContactEntity.h>
#include <agx/Integer.h>
#include <agx/Vec3.h>
#include <agx/Real.h>


namespace agx
{
  namespace Physics
  {

    class ParticlePairContactModel;
    class ParticlePairContactData;
    class ParticlePairContactPtr;
    class ParticlePairContactInstance;
    class ParticlePairContactSemantics;


    AGX_DECLARE_POINTER_TYPES(ParticlePairContactModel);

    /** 
    Abstract description of the data attributes for the Physics.ParticlePairContact entity.
    */ 
    class AGXPHYSICS_EXPORT ParticlePairContactModel : public agx::Physics::ContactModel
    {
    public:
      typedef ParticlePairContactPtr PtrT;

      ParticlePairContactModel(const agx::String& name = "ParticlePairContact");

      /// \return The entity model singleton.
      static ParticlePairContactModel* instance();

      /// Create and return a pointer to a new instance in the default storage for this entity model.
      static ParticlePairContactPtr createInstance();

      /// \return The default storage for this entity model.
      static agxData::EntityStorage* defaultStorage();

      /// This is part of internal cleanup and should not be called by users
      virtual void shutdownCleanup() override;



      /* Attributes */
      static agxData::ScalarAttributeT< agx::UInt32 >* particle1Attribute;
      static agxData::ScalarAttributeT< agx::UInt32 >* particle2Attribute;
      static agxData::ScalarAttributeT< agx::UInt32 >* particleId1Attribute;
      static agxData::ScalarAttributeT< agx::UInt32 >* particleId2Attribute;
      static agxData::ScalarAttributeT< agx::Bool >* enabledAttribute;
      static agxData::ScalarAttributeT< agx::Vec3 >* pointAttribute;
      static agxData::ScalarAttributeT< agx::Vec3f >* normalAttribute;
      static agxData::ScalarAttributeT< agx::Vec3f >* tangentUAttribute;
      static agxData::ScalarAttributeT< agx::Vec3f >* tangentVAttribute;
      static agxData::ScalarAttributeT< agx::Real >* depthAttribute;
      static agxData::ScalarAttributeT< agx::Vec3f >* velocityAttribute;
      static agxData::ScalarAttributeT< agx::Vec3 >* localForceAttribute;
      static agxData::ScalarAttributeT< agx::Real >* impactNormalForceAttribute;
      static agxData::ScalarAttributeT< agx::Real >* contactEnergyAttribute;
      static agxData::ScalarAttributeT< agx::Bool >* isImpactingAttribute;
      static agxData::ScalarAttributeT< agx::Real32 >* timeStampAttribute;
      static agxData::ScalarAttributeT< agx::Real32 >* charContactTimeAttribute;

    protected:
      virtual ~ParticlePairContactModel();
      virtual agxData::EntityData* createData(agxData::EntityStorage* storage) override;
      virtual void configure(agx::TiXmlElement* eEntity) override;
      virtual void initAttributeAccessors() override;
      virtual void construct(agxData::EntityPtr instance) override;
      void construct(agx::Physics::ParticlePairContactPtr particlePairContact);
    };


    DOXYGEN_START_INTERNAL_BLOCK()
    #ifndef AGX_PHYSICS_PARTICLEPAIRCONTACT_DATA_SET_OVERRIDE
    #define AGX_PHYSICS_PARTICLEPAIRCONTACT_DATA_SET
    class AGXPHYSICS_EXPORT ParticlePairContactData : public agx::Physics::ContactData
    {
    public:
      ParticlePairContactInstance operator[] (size_t index);

    public:
      agxData::Array< ParticlePairContactPtr >& instance;
      agxData::Array< agx::UInt32 > particle1;
      agxData::Array< agx::UInt32 > particle2;
      agxData::Array< agx::UInt32 > particleId1;
      agxData::Array< agx::UInt32 > particleId2;
      agxData::Array< agx::Bool > enabled;
      agxData::Array< agx::Vec3 > point;
      agxData::Array< agx::Vec3f > normal;
      agxData::Array< agx::Vec3f > tangentU;
      agxData::Array< agx::Vec3f > tangentV;
      agxData::Array< agx::Real > depth;
      agxData::Array< agx::Vec3f > velocity;
      agxData::Array< agx::Vec3 > localForce;
      agxData::Array< agx::Real > impactNormalForce;
      agxData::Array< agx::Real > contactEnergy;
      agxData::Array< agx::Bool > isImpacting;
      agxData::Array< agx::Real32 > timeStamp;
      agxData::Array< agx::Real32 > charContactTime;

    public:
      typedef agx::UInt32 particle1Type;
      typedef agx::UInt32 particle2Type;
      typedef agx::UInt32 particleId1Type;
      typedef agx::UInt32 particleId2Type;
      typedef agx::Bool enabledType;
      typedef agx::Vec3 pointType;
      typedef agx::Vec3f normalType;
      typedef agx::Vec3f tangentUType;
      typedef agx::Vec3f tangentVType;
      typedef agx::Real depthType;
      typedef agx::Vec3f velocityType;
      typedef agx::Vec3 localForceType;
      typedef agx::Real impactNormalForceType;
      typedef agx::Real contactEnergyType;
      typedef agx::Bool isImpactingType;
      typedef agx::Real32 timeStampType;
      typedef agx::Real32 charContactTimeType;

    public:
      ParticlePairContactData(agxData::EntityStorage* storage);
      ParticlePairContactData();

    protected:
      virtual ~ParticlePairContactData() {}
      virtual void setNumElements(agx::Index numElements) override;

    private:
      ParticlePairContactData& operator= (const ParticlePairContactData&) { return *this; }

    };
    #endif
    DOXYGEN_END_INTERNAL_BLOCK()


    DOXYGEN_START_INTERNAL_BLOCK()
    class AGXPHYSICS_EXPORT ParticlePairContactSemantics : public agx::Physics::ContactSemantics
    {
    public:

      // Automatic getters
      agx::UInt32 const& getParticle1() const;
      agx::UInt32 const& getParticle2() const;
      agx::UInt32 const& getParticleId1() const;
      agx::UInt32 const& getParticleId2() const;
      agx::Bool const& getEnabled() const;
      agx::Vec3 const& getPoint() const;
      agx::Vec3f const& getNormal() const;
      agx::Vec3f const& getTangentU() const;
      agx::Vec3f const& getTangentV() const;
      agx::Real const& getDepth() const;
      agx::Vec3f const& getVelocity() const;
      agx::Vec3 const& getLocalForce() const;
      agx::Real const& getImpactNormalForce() const;
      agx::Real const& getContactEnergy() const;
      agx::Bool const& getIsImpacting() const;
      agx::Real32 const& getTimeStamp() const;
      agx::Real32 const& getCharContactTime() const;

      // Semantics defined by explicit kernels

      // Automatic setters
      void setParticle1(agx::UInt32 const& value);
      void setParticle2(agx::UInt32 const& value);
      void setParticleId1(agx::UInt32 const& value);
      void setParticleId2(agx::UInt32 const& value);
      void setEnabled(agx::Bool const& value);
      void setPoint(agx::Vec3 const& value);
      void setNormal(agx::Vec3f const& value);
      void setTangentU(agx::Vec3f const& value);
      void setTangentV(agx::Vec3f const& value);
      void setDepth(agx::Real const& value);
      void setVelocity(agx::Vec3f const& value);
      void setLocalForce(agx::Vec3 const& value);
      void setImpactNormalForce(agx::Real const& value);
      void setContactEnergy(agx::Real const& value);
      void setIsImpacting(agx::Bool const& value);
      void setTimeStamp(agx::Real32 const& value);
      void setCharContactTime(agx::Real32 const& value);


    protected:
      friend class ParticlePairContactPtr;
      friend class ParticlePairContactInstance;
      ParticlePairContactSemantics();
    };
    DOXYGEN_END_INTERNAL_BLOCK()


    /**
    Pointer to a entity instance of type Physics.ParticlePairContact
    */
    class CALLABLE ParticlePairContactPtr : public agx::Physics::ContactPtr
    {
    public:
      typedef ParticlePairContactModel ModelType;
      typedef ParticlePairContactData DataType;
      typedef ParticlePairContactInstance InstanceType;

    public:
      AGXPHYSICS_EXPORT ParticlePairContactPtr();
      AGXPHYSICS_EXPORT ParticlePairContactPtr(agxData::EntityStorage* storage, agx::Index id);
      AGXPHYSICS_EXPORT ParticlePairContactPtr(const agxData::EntityPtr& ptr);
      AGXPHYSICS_EXPORT ParticlePairContactPtr(const agxData::EntityInstance& instance);
      AGXPHYSICS_EXPORT ParticlePairContactPtr& operator= (const agxData::EntityPtr& ptr);
      AGXPHYSICS_EXPORT ParticlePairContactPtr& operator= (const agxData::EntityInstance& instance);
      AGXPHYSICS_EXPORT ParticlePairContactInstance instance();
      AGXPHYSICS_EXPORT const ParticlePairContactInstance instance() const;

      AGXPHYSICS_EXPORT ParticlePairContactSemantics* operator->();
      AGXPHYSICS_EXPORT const ParticlePairContactSemantics* operator->() const;

      ParticlePairContactData* getData();
      const ParticlePairContactData* getData() const;


      /// \return reference to the particle1 attribute
      AGXPHYSICS_EXPORT agx::UInt32& particle1();
      /// \return const reference to the particle1 attribute
      AGXPHYSICS_EXPORT agx::UInt32 const& particle1() const;

      /// \return reference to the particle2 attribute
      AGXPHYSICS_EXPORT agx::UInt32& particle2();
      /// \return const reference to the particle2 attribute
      AGXPHYSICS_EXPORT agx::UInt32 const& particle2() const;

      /// \return reference to the particleId1 attribute
      AGXPHYSICS_EXPORT agx::UInt32& particleId1();
      /// \return const reference to the particleId1 attribute
      AGXPHYSICS_EXPORT agx::UInt32 const& particleId1() const;

      /// \return reference to the particleId2 attribute
      AGXPHYSICS_EXPORT agx::UInt32& particleId2();
      /// \return const reference to the particleId2 attribute
      AGXPHYSICS_EXPORT agx::UInt32 const& particleId2() const;

      /// \return reference to the enabled attribute
      AGXPHYSICS_EXPORT agx::Bool& enabled();
      /// \return const reference to the enabled attribute
      AGXPHYSICS_EXPORT agx::Bool const& enabled() const;

      /// \return reference to the point attribute
      AGXPHYSICS_EXPORT agx::Vec3& point();
      /// \return const reference to the point attribute
      AGXPHYSICS_EXPORT agx::Vec3 const& point() const;

      /// \return reference to the normal attribute
      AGXPHYSICS_EXPORT agx::Vec3f& normal();
      /// \return const reference to the normal attribute
      AGXPHYSICS_EXPORT agx::Vec3f const& normal() const;

      /// \return reference to the tangentU attribute
      AGXPHYSICS_EXPORT agx::Vec3f& tangentU();
      /// \return const reference to the tangentU attribute
      AGXPHYSICS_EXPORT agx::Vec3f const& tangentU() const;

      /// \return reference to the tangentV attribute
      AGXPHYSICS_EXPORT agx::Vec3f& tangentV();
      /// \return const reference to the tangentV attribute
      AGXPHYSICS_EXPORT agx::Vec3f const& tangentV() const;

      /// \return reference to the depth attribute
      AGXPHYSICS_EXPORT agx::Real& depth();
      /// \return const reference to the depth attribute
      AGXPHYSICS_EXPORT agx::Real const& depth() const;

      /// \return reference to the velocity attribute
      AGXPHYSICS_EXPORT agx::Vec3f& velocity();
      /// \return const reference to the velocity attribute
      AGXPHYSICS_EXPORT agx::Vec3f const& velocity() const;

      /// \return reference to the localForce attribute
      AGXPHYSICS_EXPORT agx::Vec3& localForce();
      /// \return const reference to the localForce attribute
      AGXPHYSICS_EXPORT agx::Vec3 const& localForce() const;

      /// \return reference to the impactNormalForce attribute
      AGXPHYSICS_EXPORT agx::Real& impactNormalForce();
      /// \return const reference to the impactNormalForce attribute
      AGXPHYSICS_EXPORT agx::Real const& impactNormalForce() const;

      /// \return reference to the contactEnergy attribute
      AGXPHYSICS_EXPORT agx::Real& contactEnergy();
      /// \return const reference to the contactEnergy attribute
      AGXPHYSICS_EXPORT agx::Real const& contactEnergy() const;

      /// \return reference to the isImpacting attribute
      AGXPHYSICS_EXPORT agx::Bool& isImpacting();
      /// \return const reference to the isImpacting attribute
      AGXPHYSICS_EXPORT agx::Bool const& isImpacting() const;

      /// \return reference to the timeStamp attribute
      AGXPHYSICS_EXPORT agx::Real32& timeStamp();
      /// \return const reference to the timeStamp attribute
      AGXPHYSICS_EXPORT agx::Real32 const& timeStamp() const;

      /// \return reference to the charContactTime attribute
      AGXPHYSICS_EXPORT agx::Real32& charContactTime();
      /// \return const reference to the charContactTime attribute
      AGXPHYSICS_EXPORT agx::Real32 const& charContactTime() const;

      DOXYGEN_START_INTERNAL_BLOCK()
      // Only used by tolua!
      #ifdef AGX_GENERATED_TOLUA_WRAPPER
      inline agx::UInt32 const& getParticle1() const { return (*this)->getParticle1(); }
      inline agx::UInt32 const& getParticle2() const { return (*this)->getParticle2(); }
      inline agx::UInt32 const& getParticleId1() const { return (*this)->getParticleId1(); }
      inline agx::UInt32 const& getParticleId2() const { return (*this)->getParticleId2(); }
      inline agx::Bool const& getEnabled() const { return (*this)->getEnabled(); }
      inline agx::Vec3 const& getPoint() const { return (*this)->getPoint(); }
      inline agx::Vec3f const& getNormal() const { return (*this)->getNormal(); }
      inline agx::Vec3f const& getTangentU() const { return (*this)->getTangentU(); }
      inline agx::Vec3f const& getTangentV() const { return (*this)->getTangentV(); }
      inline agx::Real const& getDepth() const { return (*this)->getDepth(); }
      inline agx::Vec3f const& getVelocity() const { return (*this)->getVelocity(); }
      inline agx::Vec3 const& getLocalForce() const { return (*this)->getLocalForce(); }
      inline agx::Real const& getImpactNormalForce() const { return (*this)->getImpactNormalForce(); }
      inline agx::Real const& getContactEnergy() const { return (*this)->getContactEnergy(); }
      inline agx::Bool const& getIsImpacting() const { return (*this)->getIsImpacting(); }
      inline agx::Real32 const& getTimeStamp() const { return (*this)->getTimeStamp(); }
      inline agx::Real32 const& getCharContactTime() const { return (*this)->getCharContactTime(); }


      inline void setParticle1(agx::UInt32 const& value) { (*this)->setParticle1(value); }
      inline void setParticle2(agx::UInt32 const& value) { (*this)->setParticle2(value); }
      inline void setParticleId1(agx::UInt32 const& value) { (*this)->setParticleId1(value); }
      inline void setParticleId2(agx::UInt32 const& value) { (*this)->setParticleId2(value); }
      inline void setEnabled(agx::Bool const& value) { (*this)->setEnabled(value); }
      inline void setPoint(agx::Vec3 const& value) { (*this)->setPoint(value); }
      inline void setNormal(agx::Vec3f const& value) { (*this)->setNormal(value); }
      inline void setTangentU(agx::Vec3f const& value) { (*this)->setTangentU(value); }
      inline void setTangentV(agx::Vec3f const& value) { (*this)->setTangentV(value); }
      inline void setDepth(agx::Real const& value) { (*this)->setDepth(value); }
      inline void setVelocity(agx::Vec3f const& value) { (*this)->setVelocity(value); }
      inline void setLocalForce(agx::Vec3 const& value) { (*this)->setLocalForce(value); }
      inline void setImpactNormalForce(agx::Real const& value) { (*this)->setImpactNormalForce(value); }
      inline void setContactEnergy(agx::Real const& value) { (*this)->setContactEnergy(value); }
      inline void setIsImpacting(agx::Bool const& value) { (*this)->setIsImpacting(value); }
      inline void setTimeStamp(agx::Real32 const& value) { (*this)->setTimeStamp(value); }
      inline void setCharContactTime(agx::Real32 const& value) { (*this)->setCharContactTime(value); }
      #endif
      DOXYGEN_END_INTERNAL_BLOCK()

    };


    DOXYGEN_START_INTERNAL_BLOCK()
    class AGXPHYSICS_EXPORT ParticlePairContactInstance : public agx::Physics::ContactInstance
    {
    public:
      ParticlePairContactInstance();
      ParticlePairContactInstance(ParticlePairContactData* data, agx::Index index);
      ParticlePairContactInstance(agxData::EntityStorage *storage, agx::Index index);
      ParticlePairContactInstance(const agxData::EntityInstance& other);
      ParticlePairContactInstance(const agxData::EntityPtr& ptr);

      ParticlePairContactData* getData();
      const ParticlePairContactData* getData() const;

    public:
      /// \return reference to the particle1 attribute
      agx::UInt32& particle1();
      /// \return const reference to the particle1 attribute
      agx::UInt32 const& particle1() const;

      /// \return reference to the particle2 attribute
      agx::UInt32& particle2();
      /// \return const reference to the particle2 attribute
      agx::UInt32 const& particle2() const;

      /// \return reference to the particleId1 attribute
      agx::UInt32& particleId1();
      /// \return const reference to the particleId1 attribute
      agx::UInt32 const& particleId1() const;

      /// \return reference to the particleId2 attribute
      agx::UInt32& particleId2();
      /// \return const reference to the particleId2 attribute
      agx::UInt32 const& particleId2() const;

      /// \return reference to the enabled attribute
      agx::Bool& enabled();
      /// \return const reference to the enabled attribute
      agx::Bool const& enabled() const;

      /// \return reference to the point attribute
      agx::Vec3& point();
      /// \return const reference to the point attribute
      agx::Vec3 const& point() const;

      /// \return reference to the normal attribute
      agx::Vec3f& normal();
      /// \return const reference to the normal attribute
      agx::Vec3f const& normal() const;

      /// \return reference to the tangentU attribute
      agx::Vec3f& tangentU();
      /// \return const reference to the tangentU attribute
      agx::Vec3f const& tangentU() const;

      /// \return reference to the tangentV attribute
      agx::Vec3f& tangentV();
      /// \return const reference to the tangentV attribute
      agx::Vec3f const& tangentV() const;

      /// \return reference to the depth attribute
      agx::Real& depth();
      /// \return const reference to the depth attribute
      agx::Real const& depth() const;

      /// \return reference to the velocity attribute
      agx::Vec3f& velocity();
      /// \return const reference to the velocity attribute
      agx::Vec3f const& velocity() const;

      /// \return reference to the localForce attribute
      agx::Vec3& localForce();
      /// \return const reference to the localForce attribute
      agx::Vec3 const& localForce() const;

      /// \return reference to the impactNormalForce attribute
      agx::Real& impactNormalForce();
      /// \return const reference to the impactNormalForce attribute
      agx::Real const& impactNormalForce() const;

      /// \return reference to the contactEnergy attribute
      agx::Real& contactEnergy();
      /// \return const reference to the contactEnergy attribute
      agx::Real const& contactEnergy() const;

      /// \return reference to the isImpacting attribute
      agx::Bool& isImpacting();
      /// \return const reference to the isImpacting attribute
      agx::Bool const& isImpacting() const;

      /// \return reference to the timeStamp attribute
      agx::Real32& timeStamp();
      /// \return const reference to the timeStamp attribute
      agx::Real32 const& timeStamp() const;

      /// \return reference to the charContactTime attribute
      agx::Real32& charContactTime();
      /// \return const reference to the charContactTime attribute
      agx::Real32 const& charContactTime() const;

    };
    DOXYGEN_END_INTERNAL_BLOCK()



    typedef agx::VectorPOD<ParticlePairContactPtr> ParticlePairContactPtrVector;
    typedef agxData::Array<ParticlePairContactPtr> ParticlePairContactPtrArray;



    DOXYGEN_START_INTERNAL_BLOCK()
    /* Implementation */
    //-----------------------------------------------------------------------------------------------------
    //-----------------------------------------------------------------------------------------------------
    inline ParticlePairContactInstance agx::Physics::ParticlePairContactData::operator[] (size_t index) { return ParticlePairContactInstance(this, (agx::Index)index); }
    //-----------------------------------------------------------------------------------------------------
    AGX_FORCE_INLINE ParticlePairContactPtr::ParticlePairContactPtr() {}
    AGX_FORCE_INLINE ParticlePairContactPtr::ParticlePairContactPtr(agxData::EntityStorage* storage, agx::Index id) : agx::Physics::ContactPtr(storage, id) {}
    AGX_FORCE_INLINE ParticlePairContactPtr::ParticlePairContactPtr(const agxData::EntityPtr& ptr) : agx::Physics::ContactPtr(ptr)
    {
      agxAssertN(!ptr || ptr.isInstanceOf(ParticlePairContactModel::instance()),
        "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
        EntityPtr::getModel()->fullPath().c_str(), ParticlePairContactModel::instance()->fullPath().c_str());
    }

    AGX_FORCE_INLINE ParticlePairContactPtr::ParticlePairContactPtr(const agxData::EntityInstance& instance) : agx::Physics::ContactPtr(instance)
    {
      agxAssertN(!instance || instance.isInstanceOf(ParticlePairContactModel::instance()),
        "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
        EntityPtr::getModel()->fullPath().c_str(), ParticlePairContactModel::instance()->fullPath().c_str());
    }

    AGX_FORCE_INLINE ParticlePairContactPtr& ParticlePairContactPtr::operator= (const agxData::EntityPtr& ptr)
    {
      agxData::EntityPtr::operator= (ptr);
      agxAssertN(!ptr || ptr.isInstanceOf(ParticlePairContactModel::instance()),
        "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
        EntityPtr::getModel()->fullPath().c_str(), ParticlePairContactModel::instance()->fullPath().c_str());
      return *this;
    }

    AGX_FORCE_INLINE ParticlePairContactPtr& ParticlePairContactPtr::operator= (const agxData::EntityInstance& instance)
    {
      agxData::EntityPtr::operator= (instance);
      agxAssertN(!instance || instance.isInstanceOf(ParticlePairContactModel::instance()),
        "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
        EntityPtr::getModel()->fullPath().c_str(), ParticlePairContactModel::instance()->fullPath().c_str());
      return *this;
    }

    inline ParticlePairContactInstance ParticlePairContactPtr::instance() { return agxData::EntityPtr::instance(); }
    inline const ParticlePairContactInstance ParticlePairContactPtr::instance() const { return agxData::EntityPtr::instance(); }
    AGX_FORCE_INLINE ParticlePairContactSemantics* ParticlePairContactPtr::operator->() { return (ParticlePairContactSemantics* )this; }
    AGX_FORCE_INLINE const ParticlePairContactSemantics* ParticlePairContactPtr::operator->() const { return (const ParticlePairContactSemantics* )this; }
    AGX_FORCE_INLINE ParticlePairContactData* ParticlePairContactPtr::getData() { return static_cast<ParticlePairContactData* >(agxData::EntityPtr::getData()); }
    AGX_FORCE_INLINE const ParticlePairContactData* ParticlePairContactPtr::getData() const { return static_cast<const ParticlePairContactData* >(agxData::EntityPtr::getData()); }

    AGX_FORCE_INLINE agx::UInt32& ParticlePairContactPtr::particle1() { verifyIndex(); return getData()->particle1[calculateIndex()]; }
    AGX_FORCE_INLINE agx::UInt32 const& ParticlePairContactPtr::particle1() const { verifyIndex(); return getData()->particle1[calculateIndex()]; }

    AGX_FORCE_INLINE agx::UInt32& ParticlePairContactPtr::particle2() { verifyIndex(); return getData()->particle2[calculateIndex()]; }
    AGX_FORCE_INLINE agx::UInt32 const& ParticlePairContactPtr::particle2() const { verifyIndex(); return getData()->particle2[calculateIndex()]; }

    AGX_FORCE_INLINE agx::UInt32& ParticlePairContactPtr::particleId1() { verifyIndex(); return getData()->particleId1[calculateIndex()]; }
    AGX_FORCE_INLINE agx::UInt32 const& ParticlePairContactPtr::particleId1() const { verifyIndex(); return getData()->particleId1[calculateIndex()]; }

    AGX_FORCE_INLINE agx::UInt32& ParticlePairContactPtr::particleId2() { verifyIndex(); return getData()->particleId2[calculateIndex()]; }
    AGX_FORCE_INLINE agx::UInt32 const& ParticlePairContactPtr::particleId2() const { verifyIndex(); return getData()->particleId2[calculateIndex()]; }

    AGX_FORCE_INLINE agx::Bool& ParticlePairContactPtr::enabled() { verifyIndex(); return getData()->enabled[calculateIndex()]; }
    AGX_FORCE_INLINE agx::Bool const& ParticlePairContactPtr::enabled() const { verifyIndex(); return getData()->enabled[calculateIndex()]; }

    AGX_FORCE_INLINE agx::Vec3& ParticlePairContactPtr::point() { verifyIndex(); return getData()->point[calculateIndex()]; }
    AGX_FORCE_INLINE agx::Vec3 const& ParticlePairContactPtr::point() const { verifyIndex(); return getData()->point[calculateIndex()]; }

    AGX_FORCE_INLINE agx::Vec3f& ParticlePairContactPtr::normal() { verifyIndex(); return getData()->normal[calculateIndex()]; }
    AGX_FORCE_INLINE agx::Vec3f const& ParticlePairContactPtr::normal() const { verifyIndex(); return getData()->normal[calculateIndex()]; }

    AGX_FORCE_INLINE agx::Vec3f& ParticlePairContactPtr::tangentU() { verifyIndex(); return getData()->tangentU[calculateIndex()]; }
    AGX_FORCE_INLINE agx::Vec3f const& ParticlePairContactPtr::tangentU() const { verifyIndex(); return getData()->tangentU[calculateIndex()]; }

    AGX_FORCE_INLINE agx::Vec3f& ParticlePairContactPtr::tangentV() { verifyIndex(); return getData()->tangentV[calculateIndex()]; }
    AGX_FORCE_INLINE agx::Vec3f const& ParticlePairContactPtr::tangentV() const { verifyIndex(); return getData()->tangentV[calculateIndex()]; }

    AGX_FORCE_INLINE agx::Real& ParticlePairContactPtr::depth() { verifyIndex(); return getData()->depth[calculateIndex()]; }
    AGX_FORCE_INLINE agx::Real const& ParticlePairContactPtr::depth() const { verifyIndex(); return getData()->depth[calculateIndex()]; }

    AGX_FORCE_INLINE agx::Vec3f& ParticlePairContactPtr::velocity() { verifyIndex(); return getData()->velocity[calculateIndex()]; }
    AGX_FORCE_INLINE agx::Vec3f const& ParticlePairContactPtr::velocity() const { verifyIndex(); return getData()->velocity[calculateIndex()]; }

    AGX_FORCE_INLINE agx::Vec3& ParticlePairContactPtr::localForce() { verifyIndex(); return getData()->localForce[calculateIndex()]; }
    AGX_FORCE_INLINE agx::Vec3 const& ParticlePairContactPtr::localForce() const { verifyIndex(); return getData()->localForce[calculateIndex()]; }

    AGX_FORCE_INLINE agx::Real& ParticlePairContactPtr::impactNormalForce() { verifyIndex(); return getData()->impactNormalForce[calculateIndex()]; }
    AGX_FORCE_INLINE agx::Real const& ParticlePairContactPtr::impactNormalForce() const { verifyIndex(); return getData()->impactNormalForce[calculateIndex()]; }

    AGX_FORCE_INLINE agx::Real& ParticlePairContactPtr::contactEnergy() { verifyIndex(); return getData()->contactEnergy[calculateIndex()]; }
    AGX_FORCE_INLINE agx::Real const& ParticlePairContactPtr::contactEnergy() const { verifyIndex(); return getData()->contactEnergy[calculateIndex()]; }

    AGX_FORCE_INLINE agx::Bool& ParticlePairContactPtr::isImpacting() { verifyIndex(); return getData()->isImpacting[calculateIndex()]; }
    AGX_FORCE_INLINE agx::Bool const& ParticlePairContactPtr::isImpacting() const { verifyIndex(); return getData()->isImpacting[calculateIndex()]; }

    AGX_FORCE_INLINE agx::Real32& ParticlePairContactPtr::timeStamp() { verifyIndex(); return getData()->timeStamp[calculateIndex()]; }
    AGX_FORCE_INLINE agx::Real32 const& ParticlePairContactPtr::timeStamp() const { verifyIndex(); return getData()->timeStamp[calculateIndex()]; }

    AGX_FORCE_INLINE agx::Real32& ParticlePairContactPtr::charContactTime() { verifyIndex(); return getData()->charContactTime[calculateIndex()]; }
    AGX_FORCE_INLINE agx::Real32 const& ParticlePairContactPtr::charContactTime() const { verifyIndex(); return getData()->charContactTime[calculateIndex()]; }

    //-----------------------------------------------------------------------------------------------------
    AGX_FORCE_INLINE ParticlePairContactInstance::ParticlePairContactInstance() {}
    AGX_FORCE_INLINE ParticlePairContactInstance::ParticlePairContactInstance(ParticlePairContactData* data, agx::Index index) : agx::Physics::ContactInstance(data, index) {}
    AGX_FORCE_INLINE ParticlePairContactInstance::ParticlePairContactInstance(agxData::EntityStorage* storage, agx::Index index) : agx::Physics::ContactInstance(storage, index) {}
    AGX_FORCE_INLINE ParticlePairContactInstance::ParticlePairContactInstance(const agxData::EntityInstance& other) : agx::Physics::ContactInstance(other)
    {
      agxAssertN(!other || other.isInstanceOf(ParticlePairContactModel::instance()),
        "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
        EntityInstance::getModel()->fullPath().c_str(), ParticlePairContactModel::instance()->fullPath().c_str());
    }

    AGX_FORCE_INLINE ParticlePairContactInstance::ParticlePairContactInstance(const agxData::EntityPtr& ptr) : agx::Physics::ContactInstance(ptr)
    {
      agxAssertN(!ptr || ptr.isInstanceOf(ParticlePairContactModel::instance()),
        "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
        EntityInstance::getModel()->fullPath().c_str(), ParticlePairContactModel::instance()->fullPath().c_str());
    }


    AGX_FORCE_INLINE ParticlePairContactData* ParticlePairContactInstance::getData() { return static_cast<ParticlePairContactData* >(agxData::EntityInstance::getData()); }
    AGX_FORCE_INLINE const ParticlePairContactData* ParticlePairContactInstance::getData() const { return static_cast<const ParticlePairContactData* >(agxData::EntityInstance::getData()); }

    AGX_FORCE_INLINE agx::UInt32& ParticlePairContactInstance::particle1() { verifyIndex(); return getData()->particle1[getIndex()]; }
    AGX_FORCE_INLINE agx::UInt32 const& ParticlePairContactInstance::particle1() const { verifyIndex(); return getData()->particle1[getIndex()]; }

    AGX_FORCE_INLINE agx::UInt32& ParticlePairContactInstance::particle2() { verifyIndex(); return getData()->particle2[getIndex()]; }
    AGX_FORCE_INLINE agx::UInt32 const& ParticlePairContactInstance::particle2() const { verifyIndex(); return getData()->particle2[getIndex()]; }

    AGX_FORCE_INLINE agx::UInt32& ParticlePairContactInstance::particleId1() { verifyIndex(); return getData()->particleId1[getIndex()]; }
    AGX_FORCE_INLINE agx::UInt32 const& ParticlePairContactInstance::particleId1() const { verifyIndex(); return getData()->particleId1[getIndex()]; }

    AGX_FORCE_INLINE agx::UInt32& ParticlePairContactInstance::particleId2() { verifyIndex(); return getData()->particleId2[getIndex()]; }
    AGX_FORCE_INLINE agx::UInt32 const& ParticlePairContactInstance::particleId2() const { verifyIndex(); return getData()->particleId2[getIndex()]; }

    AGX_FORCE_INLINE agx::Bool& ParticlePairContactInstance::enabled() { verifyIndex(); return getData()->enabled[getIndex()]; }
    AGX_FORCE_INLINE agx::Bool const& ParticlePairContactInstance::enabled() const { verifyIndex(); return getData()->enabled[getIndex()]; }

    AGX_FORCE_INLINE agx::Vec3& ParticlePairContactInstance::point() { verifyIndex(); return getData()->point[getIndex()]; }
    AGX_FORCE_INLINE agx::Vec3 const& ParticlePairContactInstance::point() const { verifyIndex(); return getData()->point[getIndex()]; }

    AGX_FORCE_INLINE agx::Vec3f& ParticlePairContactInstance::normal() { verifyIndex(); return getData()->normal[getIndex()]; }
    AGX_FORCE_INLINE agx::Vec3f const& ParticlePairContactInstance::normal() const { verifyIndex(); return getData()->normal[getIndex()]; }

    AGX_FORCE_INLINE agx::Vec3f& ParticlePairContactInstance::tangentU() { verifyIndex(); return getData()->tangentU[getIndex()]; }
    AGX_FORCE_INLINE agx::Vec3f const& ParticlePairContactInstance::tangentU() const { verifyIndex(); return getData()->tangentU[getIndex()]; }

    AGX_FORCE_INLINE agx::Vec3f& ParticlePairContactInstance::tangentV() { verifyIndex(); return getData()->tangentV[getIndex()]; }
    AGX_FORCE_INLINE agx::Vec3f const& ParticlePairContactInstance::tangentV() const { verifyIndex(); return getData()->tangentV[getIndex()]; }

    AGX_FORCE_INLINE agx::Real& ParticlePairContactInstance::depth() { verifyIndex(); return getData()->depth[getIndex()]; }
    AGX_FORCE_INLINE agx::Real const& ParticlePairContactInstance::depth() const { verifyIndex(); return getData()->depth[getIndex()]; }

    AGX_FORCE_INLINE agx::Vec3f& ParticlePairContactInstance::velocity() { verifyIndex(); return getData()->velocity[getIndex()]; }
    AGX_FORCE_INLINE agx::Vec3f const& ParticlePairContactInstance::velocity() const { verifyIndex(); return getData()->velocity[getIndex()]; }

    AGX_FORCE_INLINE agx::Vec3& ParticlePairContactInstance::localForce() { verifyIndex(); return getData()->localForce[getIndex()]; }
    AGX_FORCE_INLINE agx::Vec3 const& ParticlePairContactInstance::localForce() const { verifyIndex(); return getData()->localForce[getIndex()]; }

    AGX_FORCE_INLINE agx::Real& ParticlePairContactInstance::impactNormalForce() { verifyIndex(); return getData()->impactNormalForce[getIndex()]; }
    AGX_FORCE_INLINE agx::Real const& ParticlePairContactInstance::impactNormalForce() const { verifyIndex(); return getData()->impactNormalForce[getIndex()]; }

    AGX_FORCE_INLINE agx::Real& ParticlePairContactInstance::contactEnergy() { verifyIndex(); return getData()->contactEnergy[getIndex()]; }
    AGX_FORCE_INLINE agx::Real const& ParticlePairContactInstance::contactEnergy() const { verifyIndex(); return getData()->contactEnergy[getIndex()]; }

    AGX_FORCE_INLINE agx::Bool& ParticlePairContactInstance::isImpacting() { verifyIndex(); return getData()->isImpacting[getIndex()]; }
    AGX_FORCE_INLINE agx::Bool const& ParticlePairContactInstance::isImpacting() const { verifyIndex(); return getData()->isImpacting[getIndex()]; }

    AGX_FORCE_INLINE agx::Real32& ParticlePairContactInstance::timeStamp() { verifyIndex(); return getData()->timeStamp[getIndex()]; }
    AGX_FORCE_INLINE agx::Real32 const& ParticlePairContactInstance::timeStamp() const { verifyIndex(); return getData()->timeStamp[getIndex()]; }

    AGX_FORCE_INLINE agx::Real32& ParticlePairContactInstance::charContactTime() { verifyIndex(); return getData()->charContactTime[getIndex()]; }
    AGX_FORCE_INLINE agx::Real32 const& ParticlePairContactInstance::charContactTime() const { verifyIndex(); return getData()->charContactTime[getIndex()]; }

    //-----------------------------------------------------------------------------------------------------
    AGX_FORCE_INLINE ParticlePairContactSemantics::ParticlePairContactSemantics() {}
    //-----------------------------------------------------------------------------------------------------
    DOXYGEN_END_INTERNAL_BLOCK()
  }
}

AGX_TYPE_BINDING(agx::Physics::ParticlePairContactPtr, "Physics.ParticlePairContactPtr")
AGX_TYPE_BINDING(agx::Physics::ParticlePairContactInstance, "Physics.ParticlePairContactInstance")

#ifdef _MSC_VER
# pragma warning(pop)
#endif

#undef AGX_ENTITY_WRAPPER
#undef AGX_ENTITY_NAMESPACE
#endif

