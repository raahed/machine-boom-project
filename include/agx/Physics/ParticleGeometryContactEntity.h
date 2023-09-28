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

#ifndef GENERATED_AGX_PHYSICS_PARTICLEGEOMETRYCONTACT_H_PLUGIN
#define GENERATED_AGX_PHYSICS_PARTICLEGEOMETRYCONTACT_H_PLUGIN

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
#include <agx/Physics/GeometryEntity.h>
#include <agx/Vec3.h>
#include <agx/Real.h>

namespace agx { namespace Physics { class GeometryPtr; }}

namespace agx
{
  namespace Physics
  {

    class ParticleGeometryContactModel;
    class ParticleGeometryContactData;
    class ParticleGeometryContactPtr;
    class ParticleGeometryContactInstance;
    class ParticleGeometryContactSemantics;


    AGX_DECLARE_POINTER_TYPES(ParticleGeometryContactModel);

    /** 
    Abstract description of the data attributes for the Physics.ParticleGeometryContact entity.
    */ 
    class AGXPHYSICS_EXPORT ParticleGeometryContactModel : public agx::Physics::ContactModel
    {
    public:
      typedef ParticleGeometryContactPtr PtrT;

      ParticleGeometryContactModel(const agx::String& name = "ParticleGeometryContact");

      /// \return The entity model singleton.
      static ParticleGeometryContactModel* instance();

      /// Create and return a pointer to a new instance in the default storage for this entity model.
      static ParticleGeometryContactPtr createInstance();

      /// \return The default storage for this entity model.
      static agxData::EntityStorage* defaultStorage();

      /// This is part of internal cleanup and should not be called by users
      virtual void shutdownCleanup() override;



      /* Attributes */
      static agxData::ScalarAttributeT< agx::UInt32 >* particleAttribute;
      static agxData::ScalarAttributeT< agx::UInt32 >* geometryAttribute;
      static agxData::ScalarAttributeT< agx::UInt32 >* bodyAttribute;
      static agxData::ScalarAttributeT< agx::UInt32 >* particleIdAttribute;
      static agxData::ScalarAttributeT< agx::UInt32 >* geometryIdAttribute;
      static agxData::ScalarAttributeT< agx::Physics::GeometryPtr >* geometryPtrAttribute;
      static agxData::ScalarAttributeT< agx::UInt32 >* faceIndexAttribute;
      static agxData::ScalarAttributeT< agx::UInt8 >* faceFeatureAttribute;
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
      static agxData::ScalarAttributeT< agx::Int >* nextAttribute;

    protected:
      virtual ~ParticleGeometryContactModel();
      virtual agxData::EntityData* createData(agxData::EntityStorage* storage) override;
      virtual void configure(agx::TiXmlElement* eEntity) override;
      virtual void initAttributeAccessors() override;
      virtual void construct(agxData::EntityPtr instance) override;
      void construct(agx::Physics::ParticleGeometryContactPtr particleGeometryContact);
    };


    DOXYGEN_START_INTERNAL_BLOCK()
    #ifndef AGX_PHYSICS_PARTICLEGEOMETRYCONTACT_DATA_SET_OVERRIDE
    #define AGX_PHYSICS_PARTICLEGEOMETRYCONTACT_DATA_SET
    class AGXPHYSICS_EXPORT ParticleGeometryContactData : public agx::Physics::ContactData
    {
    public:
      ParticleGeometryContactInstance operator[] (size_t index);

    public:
      agxData::Array< ParticleGeometryContactPtr >& instance;
      agxData::Array< agx::UInt32 > particle;
      agxData::Array< agx::UInt32 > geometry;
      agxData::Array< agx::UInt32 > body;
      agxData::Array< agx::UInt32 > particleId;
      agxData::Array< agx::UInt32 > geometryId;
      agxData::Array< agx::Physics::GeometryPtr > geometryPtr;
      agxData::Array< agx::UInt32 > faceIndex;
      agxData::Array< agx::UInt8 > faceFeature;
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
      agxData::Array< agx::Int > next;

    public:
      typedef agx::UInt32 particleType;
      typedef agx::UInt32 geometryType;
      typedef agx::UInt32 bodyType;
      typedef agx::UInt32 particleIdType;
      typedef agx::UInt32 geometryIdType;
      typedef agx::Physics::GeometryPtr geometryPtrType;
      typedef agx::UInt32 faceIndexType;
      typedef agx::UInt8 faceFeatureType;
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
      typedef agx::Int nextType;

    public:
      ParticleGeometryContactData(agxData::EntityStorage* storage);
      ParticleGeometryContactData();

    protected:
      virtual ~ParticleGeometryContactData() {}
      virtual void setNumElements(agx::Index numElements) override;

    private:
      ParticleGeometryContactData& operator= (const ParticleGeometryContactData&) { return *this; }

    };
    #endif
    DOXYGEN_END_INTERNAL_BLOCK()


    DOXYGEN_START_INTERNAL_BLOCK()
    class AGXPHYSICS_EXPORT ParticleGeometryContactSemantics : public agx::Physics::ContactSemantics
    {
    public:

      // Automatic getters
      agx::UInt32 const& getParticle() const;
      agx::UInt32 const& getGeometry() const;
      agx::UInt32 const& getBody() const;
      agx::UInt32 const& getParticleId() const;
      agx::UInt32 const& getGeometryId() const;
      agx::Physics::GeometryPtr const& getGeometryPtr() const;
      agx::UInt32 const& getFaceIndex() const;
      agx::UInt8 const& getFaceFeature() const;
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
      agx::Int const& getNext() const;

      // Semantics defined by explicit kernels

      // Automatic setters
      void setParticle(agx::UInt32 const& value);
      void setGeometry(agx::UInt32 const& value);
      void setBody(agx::UInt32 const& value);
      void setParticleId(agx::UInt32 const& value);
      void setGeometryId(agx::UInt32 const& value);
      void setGeometryPtr(agx::Physics::GeometryPtr const& value);
      void setFaceIndex(agx::UInt32 const& value);
      void setFaceFeature(agx::UInt8 const& value);
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
      void setNext(agx::Int const& value);


    protected:
      friend class ParticleGeometryContactPtr;
      friend class ParticleGeometryContactInstance;
      ParticleGeometryContactSemantics();
    };
    DOXYGEN_END_INTERNAL_BLOCK()


    /**
    Pointer to a entity instance of type Physics.ParticleGeometryContact
    */
    class CALLABLE ParticleGeometryContactPtr : public agx::Physics::ContactPtr
    {
    public:
      typedef ParticleGeometryContactModel ModelType;
      typedef ParticleGeometryContactData DataType;
      typedef ParticleGeometryContactInstance InstanceType;

    public:
      AGXPHYSICS_EXPORT ParticleGeometryContactPtr();
      AGXPHYSICS_EXPORT ParticleGeometryContactPtr(agxData::EntityStorage* storage, agx::Index id);
      AGXPHYSICS_EXPORT ParticleGeometryContactPtr(const agxData::EntityPtr& ptr);
      AGXPHYSICS_EXPORT ParticleGeometryContactPtr(const agxData::EntityInstance& instance);
      AGXPHYSICS_EXPORT ParticleGeometryContactPtr& operator= (const agxData::EntityPtr& ptr);
      AGXPHYSICS_EXPORT ParticleGeometryContactPtr& operator= (const agxData::EntityInstance& instance);
      AGXPHYSICS_EXPORT ParticleGeometryContactInstance instance();
      AGXPHYSICS_EXPORT const ParticleGeometryContactInstance instance() const;

      AGXPHYSICS_EXPORT ParticleGeometryContactSemantics* operator->();
      AGXPHYSICS_EXPORT const ParticleGeometryContactSemantics* operator->() const;

      ParticleGeometryContactData* getData();
      const ParticleGeometryContactData* getData() const;


      /// \return reference to the particle attribute
      AGXPHYSICS_EXPORT agx::UInt32& particle();
      /// \return const reference to the particle attribute
      AGXPHYSICS_EXPORT agx::UInt32 const& particle() const;

      /// \return reference to the geometry attribute
      AGXPHYSICS_EXPORT agx::UInt32& geometry();
      /// \return const reference to the geometry attribute
      AGXPHYSICS_EXPORT agx::UInt32 const& geometry() const;

      /// \return reference to the body attribute
      AGXPHYSICS_EXPORT agx::UInt32& body();
      /// \return const reference to the body attribute
      AGXPHYSICS_EXPORT agx::UInt32 const& body() const;

      /// \return reference to the particleId attribute
      AGXPHYSICS_EXPORT agx::UInt32& particleId();
      /// \return const reference to the particleId attribute
      AGXPHYSICS_EXPORT agx::UInt32 const& particleId() const;

      /// \return reference to the geometryId attribute
      AGXPHYSICS_EXPORT agx::UInt32& geometryId();
      /// \return const reference to the geometryId attribute
      AGXPHYSICS_EXPORT agx::UInt32 const& geometryId() const;

      /// \return reference to the geometryPtr attribute
      AGXPHYSICS_EXPORT agx::Physics::GeometryPtr& geometryPtr();
      /// \return const reference to the geometryPtr attribute
      AGXPHYSICS_EXPORT agx::Physics::GeometryPtr const& geometryPtr() const;

      /// The face index of shape 2 in contact (e.g. index of triangle for trimesh).
      AGXPHYSICS_EXPORT agx::UInt32& faceIndex();
      /// The face index of shape 2 in contact (e.g. index of triangle for trimesh).
      AGXPHYSICS_EXPORT agx::UInt32 const& faceIndex() const;

      /// The face feature of shape 2 in contact (e.g. voronoi region on triangle for trimesh).
      AGXPHYSICS_EXPORT agx::UInt8& faceFeature();
      /// The face feature of shape 2 in contact (e.g. voronoi region on triangle for trimesh).
      AGXPHYSICS_EXPORT agx::UInt8 const& faceFeature() const;

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

      /// \return reference to the next attribute
      AGXPHYSICS_EXPORT agx::Int& next();
      /// \return const reference to the next attribute
      AGXPHYSICS_EXPORT agx::Int const& next() const;

      DOXYGEN_START_INTERNAL_BLOCK()
      // Only used by tolua!
      #ifdef AGX_GENERATED_TOLUA_WRAPPER
      inline agx::UInt32 const& getParticle() const { return (*this)->getParticle(); }
      inline agx::UInt32 const& getGeometry() const { return (*this)->getGeometry(); }
      inline agx::UInt32 const& getBody() const { return (*this)->getBody(); }
      inline agx::UInt32 const& getParticleId() const { return (*this)->getParticleId(); }
      inline agx::UInt32 const& getGeometryId() const { return (*this)->getGeometryId(); }
      inline agx::Physics::GeometryPtr const& getGeometryPtr() const { return (*this)->getGeometryPtr(); }
      inline agx::UInt32 const& getFaceIndex() const { return (*this)->getFaceIndex(); }
      inline agx::UInt8 const& getFaceFeature() const { return (*this)->getFaceFeature(); }
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
      inline agx::Int const& getNext() const { return (*this)->getNext(); }


      inline void setParticle(agx::UInt32 const& value) { (*this)->setParticle(value); }
      inline void setGeometry(agx::UInt32 const& value) { (*this)->setGeometry(value); }
      inline void setBody(agx::UInt32 const& value) { (*this)->setBody(value); }
      inline void setParticleId(agx::UInt32 const& value) { (*this)->setParticleId(value); }
      inline void setGeometryId(agx::UInt32 const& value) { (*this)->setGeometryId(value); }
      inline void setGeometryPtr(agx::Physics::GeometryPtr const& value) { (*this)->setGeometryPtr(value); }
      inline void setFaceIndex(agx::UInt32 const& value) { (*this)->setFaceIndex(value); }
      inline void setFaceFeature(agx::UInt8 const& value) { (*this)->setFaceFeature(value); }
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
      inline void setNext(agx::Int const& value) { (*this)->setNext(value); }
      #endif
      DOXYGEN_END_INTERNAL_BLOCK()

    };


    DOXYGEN_START_INTERNAL_BLOCK()
    class AGXPHYSICS_EXPORT ParticleGeometryContactInstance : public agx::Physics::ContactInstance
    {
    public:
      ParticleGeometryContactInstance();
      ParticleGeometryContactInstance(ParticleGeometryContactData* data, agx::Index index);
      ParticleGeometryContactInstance(agxData::EntityStorage *storage, agx::Index index);
      ParticleGeometryContactInstance(const agxData::EntityInstance& other);
      ParticleGeometryContactInstance(const agxData::EntityPtr& ptr);

      ParticleGeometryContactData* getData();
      const ParticleGeometryContactData* getData() const;

    public:
      /// \return reference to the particle attribute
      agx::UInt32& particle();
      /// \return const reference to the particle attribute
      agx::UInt32 const& particle() const;

      /// \return reference to the geometry attribute
      agx::UInt32& geometry();
      /// \return const reference to the geometry attribute
      agx::UInt32 const& geometry() const;

      /// \return reference to the body attribute
      agx::UInt32& body();
      /// \return const reference to the body attribute
      agx::UInt32 const& body() const;

      /// \return reference to the particleId attribute
      agx::UInt32& particleId();
      /// \return const reference to the particleId attribute
      agx::UInt32 const& particleId() const;

      /// \return reference to the geometryId attribute
      agx::UInt32& geometryId();
      /// \return const reference to the geometryId attribute
      agx::UInt32 const& geometryId() const;

      /// \return reference to the geometryPtr attribute
      agx::Physics::GeometryPtr& geometryPtr();
      /// \return const reference to the geometryPtr attribute
      agx::Physics::GeometryPtr const& geometryPtr() const;

      /// The face index of shape 2 in contact (e.g. index of triangle for trimesh).
      agx::UInt32& faceIndex();
      /// The face index of shape 2 in contact (e.g. index of triangle for trimesh).
      agx::UInt32 const& faceIndex() const;

      /// The face feature of shape 2 in contact (e.g. voronoi region on triangle for trimesh).
      agx::UInt8& faceFeature();
      /// The face feature of shape 2 in contact (e.g. voronoi region on triangle for trimesh).
      agx::UInt8 const& faceFeature() const;

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

      /// \return reference to the next attribute
      agx::Int& next();
      /// \return const reference to the next attribute
      agx::Int const& next() const;

    };
    DOXYGEN_END_INTERNAL_BLOCK()



    typedef agx::VectorPOD<ParticleGeometryContactPtr> ParticleGeometryContactPtrVector;
    typedef agxData::Array<ParticleGeometryContactPtr> ParticleGeometryContactPtrArray;



    DOXYGEN_START_INTERNAL_BLOCK()
    /* Implementation */
    //-----------------------------------------------------------------------------------------------------
    //-----------------------------------------------------------------------------------------------------
    inline ParticleGeometryContactInstance agx::Physics::ParticleGeometryContactData::operator[] (size_t index) { return ParticleGeometryContactInstance(this, (agx::Index)index); }
    //-----------------------------------------------------------------------------------------------------
    AGX_FORCE_INLINE ParticleGeometryContactPtr::ParticleGeometryContactPtr() {}
    AGX_FORCE_INLINE ParticleGeometryContactPtr::ParticleGeometryContactPtr(agxData::EntityStorage* storage, agx::Index id) : agx::Physics::ContactPtr(storage, id) {}
    AGX_FORCE_INLINE ParticleGeometryContactPtr::ParticleGeometryContactPtr(const agxData::EntityPtr& ptr) : agx::Physics::ContactPtr(ptr)
    {
      agxAssertN(!ptr || ptr.isInstanceOf(ParticleGeometryContactModel::instance()),
        "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
        EntityPtr::getModel()->fullPath().c_str(), ParticleGeometryContactModel::instance()->fullPath().c_str());
    }

    AGX_FORCE_INLINE ParticleGeometryContactPtr::ParticleGeometryContactPtr(const agxData::EntityInstance& instance) : agx::Physics::ContactPtr(instance)
    {
      agxAssertN(!instance || instance.isInstanceOf(ParticleGeometryContactModel::instance()),
        "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
        EntityPtr::getModel()->fullPath().c_str(), ParticleGeometryContactModel::instance()->fullPath().c_str());
    }

    AGX_FORCE_INLINE ParticleGeometryContactPtr& ParticleGeometryContactPtr::operator= (const agxData::EntityPtr& ptr)
    {
      agxData::EntityPtr::operator= (ptr);
      agxAssertN(!ptr || ptr.isInstanceOf(ParticleGeometryContactModel::instance()),
        "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
        EntityPtr::getModel()->fullPath().c_str(), ParticleGeometryContactModel::instance()->fullPath().c_str());
      return *this;
    }

    AGX_FORCE_INLINE ParticleGeometryContactPtr& ParticleGeometryContactPtr::operator= (const agxData::EntityInstance& instance)
    {
      agxData::EntityPtr::operator= (instance);
      agxAssertN(!instance || instance.isInstanceOf(ParticleGeometryContactModel::instance()),
        "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
        EntityPtr::getModel()->fullPath().c_str(), ParticleGeometryContactModel::instance()->fullPath().c_str());
      return *this;
    }

    inline ParticleGeometryContactInstance ParticleGeometryContactPtr::instance() { return agxData::EntityPtr::instance(); }
    inline const ParticleGeometryContactInstance ParticleGeometryContactPtr::instance() const { return agxData::EntityPtr::instance(); }
    AGX_FORCE_INLINE ParticleGeometryContactSemantics* ParticleGeometryContactPtr::operator->() { return (ParticleGeometryContactSemantics* )this; }
    AGX_FORCE_INLINE const ParticleGeometryContactSemantics* ParticleGeometryContactPtr::operator->() const { return (const ParticleGeometryContactSemantics* )this; }
    AGX_FORCE_INLINE ParticleGeometryContactData* ParticleGeometryContactPtr::getData() { return static_cast<ParticleGeometryContactData* >(agxData::EntityPtr::getData()); }
    AGX_FORCE_INLINE const ParticleGeometryContactData* ParticleGeometryContactPtr::getData() const { return static_cast<const ParticleGeometryContactData* >(agxData::EntityPtr::getData()); }

    AGX_FORCE_INLINE agx::UInt32& ParticleGeometryContactPtr::particle() { verifyIndex(); return getData()->particle[calculateIndex()]; }
    AGX_FORCE_INLINE agx::UInt32 const& ParticleGeometryContactPtr::particle() const { verifyIndex(); return getData()->particle[calculateIndex()]; }

    AGX_FORCE_INLINE agx::UInt32& ParticleGeometryContactPtr::geometry() { verifyIndex(); return getData()->geometry[calculateIndex()]; }
    AGX_FORCE_INLINE agx::UInt32 const& ParticleGeometryContactPtr::geometry() const { verifyIndex(); return getData()->geometry[calculateIndex()]; }

    AGX_FORCE_INLINE agx::UInt32& ParticleGeometryContactPtr::body() { verifyIndex(); return getData()->body[calculateIndex()]; }
    AGX_FORCE_INLINE agx::UInt32 const& ParticleGeometryContactPtr::body() const { verifyIndex(); return getData()->body[calculateIndex()]; }

    AGX_FORCE_INLINE agx::UInt32& ParticleGeometryContactPtr::particleId() { verifyIndex(); return getData()->particleId[calculateIndex()]; }
    AGX_FORCE_INLINE agx::UInt32 const& ParticleGeometryContactPtr::particleId() const { verifyIndex(); return getData()->particleId[calculateIndex()]; }

    AGX_FORCE_INLINE agx::UInt32& ParticleGeometryContactPtr::geometryId() { verifyIndex(); return getData()->geometryId[calculateIndex()]; }
    AGX_FORCE_INLINE agx::UInt32 const& ParticleGeometryContactPtr::geometryId() const { verifyIndex(); return getData()->geometryId[calculateIndex()]; }

    AGX_FORCE_INLINE agx::Physics::GeometryPtr& ParticleGeometryContactPtr::geometryPtr() { verifyIndex(); return getData()->geometryPtr[calculateIndex()]; }
    AGX_FORCE_INLINE agx::Physics::GeometryPtr const& ParticleGeometryContactPtr::geometryPtr() const { verifyIndex(); return getData()->geometryPtr[calculateIndex()]; }

    AGX_FORCE_INLINE agx::UInt32& ParticleGeometryContactPtr::faceIndex() { verifyIndex(); return getData()->faceIndex[calculateIndex()]; }
    AGX_FORCE_INLINE agx::UInt32 const& ParticleGeometryContactPtr::faceIndex() const { verifyIndex(); return getData()->faceIndex[calculateIndex()]; }

    AGX_FORCE_INLINE agx::UInt8& ParticleGeometryContactPtr::faceFeature() { verifyIndex(); return getData()->faceFeature[calculateIndex()]; }
    AGX_FORCE_INLINE agx::UInt8 const& ParticleGeometryContactPtr::faceFeature() const { verifyIndex(); return getData()->faceFeature[calculateIndex()]; }

    AGX_FORCE_INLINE agx::Bool& ParticleGeometryContactPtr::enabled() { verifyIndex(); return getData()->enabled[calculateIndex()]; }
    AGX_FORCE_INLINE agx::Bool const& ParticleGeometryContactPtr::enabled() const { verifyIndex(); return getData()->enabled[calculateIndex()]; }

    AGX_FORCE_INLINE agx::Vec3& ParticleGeometryContactPtr::point() { verifyIndex(); return getData()->point[calculateIndex()]; }
    AGX_FORCE_INLINE agx::Vec3 const& ParticleGeometryContactPtr::point() const { verifyIndex(); return getData()->point[calculateIndex()]; }

    AGX_FORCE_INLINE agx::Vec3f& ParticleGeometryContactPtr::normal() { verifyIndex(); return getData()->normal[calculateIndex()]; }
    AGX_FORCE_INLINE agx::Vec3f const& ParticleGeometryContactPtr::normal() const { verifyIndex(); return getData()->normal[calculateIndex()]; }

    AGX_FORCE_INLINE agx::Vec3f& ParticleGeometryContactPtr::tangentU() { verifyIndex(); return getData()->tangentU[calculateIndex()]; }
    AGX_FORCE_INLINE agx::Vec3f const& ParticleGeometryContactPtr::tangentU() const { verifyIndex(); return getData()->tangentU[calculateIndex()]; }

    AGX_FORCE_INLINE agx::Vec3f& ParticleGeometryContactPtr::tangentV() { verifyIndex(); return getData()->tangentV[calculateIndex()]; }
    AGX_FORCE_INLINE agx::Vec3f const& ParticleGeometryContactPtr::tangentV() const { verifyIndex(); return getData()->tangentV[calculateIndex()]; }

    AGX_FORCE_INLINE agx::Real& ParticleGeometryContactPtr::depth() { verifyIndex(); return getData()->depth[calculateIndex()]; }
    AGX_FORCE_INLINE agx::Real const& ParticleGeometryContactPtr::depth() const { verifyIndex(); return getData()->depth[calculateIndex()]; }

    AGX_FORCE_INLINE agx::Vec3f& ParticleGeometryContactPtr::velocity() { verifyIndex(); return getData()->velocity[calculateIndex()]; }
    AGX_FORCE_INLINE agx::Vec3f const& ParticleGeometryContactPtr::velocity() const { verifyIndex(); return getData()->velocity[calculateIndex()]; }

    AGX_FORCE_INLINE agx::Vec3& ParticleGeometryContactPtr::localForce() { verifyIndex(); return getData()->localForce[calculateIndex()]; }
    AGX_FORCE_INLINE agx::Vec3 const& ParticleGeometryContactPtr::localForce() const { verifyIndex(); return getData()->localForce[calculateIndex()]; }

    AGX_FORCE_INLINE agx::Real& ParticleGeometryContactPtr::impactNormalForce() { verifyIndex(); return getData()->impactNormalForce[calculateIndex()]; }
    AGX_FORCE_INLINE agx::Real const& ParticleGeometryContactPtr::impactNormalForce() const { verifyIndex(); return getData()->impactNormalForce[calculateIndex()]; }

    AGX_FORCE_INLINE agx::Real& ParticleGeometryContactPtr::contactEnergy() { verifyIndex(); return getData()->contactEnergy[calculateIndex()]; }
    AGX_FORCE_INLINE agx::Real const& ParticleGeometryContactPtr::contactEnergy() const { verifyIndex(); return getData()->contactEnergy[calculateIndex()]; }

    AGX_FORCE_INLINE agx::Bool& ParticleGeometryContactPtr::isImpacting() { verifyIndex(); return getData()->isImpacting[calculateIndex()]; }
    AGX_FORCE_INLINE agx::Bool const& ParticleGeometryContactPtr::isImpacting() const { verifyIndex(); return getData()->isImpacting[calculateIndex()]; }

    AGX_FORCE_INLINE agx::Real32& ParticleGeometryContactPtr::timeStamp() { verifyIndex(); return getData()->timeStamp[calculateIndex()]; }
    AGX_FORCE_INLINE agx::Real32 const& ParticleGeometryContactPtr::timeStamp() const { verifyIndex(); return getData()->timeStamp[calculateIndex()]; }

    AGX_FORCE_INLINE agx::Real32& ParticleGeometryContactPtr::charContactTime() { verifyIndex(); return getData()->charContactTime[calculateIndex()]; }
    AGX_FORCE_INLINE agx::Real32 const& ParticleGeometryContactPtr::charContactTime() const { verifyIndex(); return getData()->charContactTime[calculateIndex()]; }

    AGX_FORCE_INLINE agx::Int& ParticleGeometryContactPtr::next() { verifyIndex(); return getData()->next[calculateIndex()]; }
    AGX_FORCE_INLINE agx::Int const& ParticleGeometryContactPtr::next() const { verifyIndex(); return getData()->next[calculateIndex()]; }

    //-----------------------------------------------------------------------------------------------------
    AGX_FORCE_INLINE ParticleGeometryContactInstance::ParticleGeometryContactInstance() {}
    AGX_FORCE_INLINE ParticleGeometryContactInstance::ParticleGeometryContactInstance(ParticleGeometryContactData* data, agx::Index index) : agx::Physics::ContactInstance(data, index) {}
    AGX_FORCE_INLINE ParticleGeometryContactInstance::ParticleGeometryContactInstance(agxData::EntityStorage* storage, agx::Index index) : agx::Physics::ContactInstance(storage, index) {}
    AGX_FORCE_INLINE ParticleGeometryContactInstance::ParticleGeometryContactInstance(const agxData::EntityInstance& other) : agx::Physics::ContactInstance(other)
    {
      agxAssertN(!other || other.isInstanceOf(ParticleGeometryContactModel::instance()),
        "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
        EntityInstance::getModel()->fullPath().c_str(), ParticleGeometryContactModel::instance()->fullPath().c_str());
    }

    AGX_FORCE_INLINE ParticleGeometryContactInstance::ParticleGeometryContactInstance(const agxData::EntityPtr& ptr) : agx::Physics::ContactInstance(ptr)
    {
      agxAssertN(!ptr || ptr.isInstanceOf(ParticleGeometryContactModel::instance()),
        "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
        EntityInstance::getModel()->fullPath().c_str(), ParticleGeometryContactModel::instance()->fullPath().c_str());
    }


    AGX_FORCE_INLINE ParticleGeometryContactData* ParticleGeometryContactInstance::getData() { return static_cast<ParticleGeometryContactData* >(agxData::EntityInstance::getData()); }
    AGX_FORCE_INLINE const ParticleGeometryContactData* ParticleGeometryContactInstance::getData() const { return static_cast<const ParticleGeometryContactData* >(agxData::EntityInstance::getData()); }

    AGX_FORCE_INLINE agx::UInt32& ParticleGeometryContactInstance::particle() { verifyIndex(); return getData()->particle[getIndex()]; }
    AGX_FORCE_INLINE agx::UInt32 const& ParticleGeometryContactInstance::particle() const { verifyIndex(); return getData()->particle[getIndex()]; }

    AGX_FORCE_INLINE agx::UInt32& ParticleGeometryContactInstance::geometry() { verifyIndex(); return getData()->geometry[getIndex()]; }
    AGX_FORCE_INLINE agx::UInt32 const& ParticleGeometryContactInstance::geometry() const { verifyIndex(); return getData()->geometry[getIndex()]; }

    AGX_FORCE_INLINE agx::UInt32& ParticleGeometryContactInstance::body() { verifyIndex(); return getData()->body[getIndex()]; }
    AGX_FORCE_INLINE agx::UInt32 const& ParticleGeometryContactInstance::body() const { verifyIndex(); return getData()->body[getIndex()]; }

    AGX_FORCE_INLINE agx::UInt32& ParticleGeometryContactInstance::particleId() { verifyIndex(); return getData()->particleId[getIndex()]; }
    AGX_FORCE_INLINE agx::UInt32 const& ParticleGeometryContactInstance::particleId() const { verifyIndex(); return getData()->particleId[getIndex()]; }

    AGX_FORCE_INLINE agx::UInt32& ParticleGeometryContactInstance::geometryId() { verifyIndex(); return getData()->geometryId[getIndex()]; }
    AGX_FORCE_INLINE agx::UInt32 const& ParticleGeometryContactInstance::geometryId() const { verifyIndex(); return getData()->geometryId[getIndex()]; }

    AGX_FORCE_INLINE agx::Physics::GeometryPtr& ParticleGeometryContactInstance::geometryPtr() { verifyIndex(); return getData()->geometryPtr[getIndex()]; }
    AGX_FORCE_INLINE agx::Physics::GeometryPtr const& ParticleGeometryContactInstance::geometryPtr() const { verifyIndex(); return getData()->geometryPtr[getIndex()]; }

    AGX_FORCE_INLINE agx::UInt32& ParticleGeometryContactInstance::faceIndex() { verifyIndex(); return getData()->faceIndex[getIndex()]; }
    AGX_FORCE_INLINE agx::UInt32 const& ParticleGeometryContactInstance::faceIndex() const { verifyIndex(); return getData()->faceIndex[getIndex()]; }

    AGX_FORCE_INLINE agx::UInt8& ParticleGeometryContactInstance::faceFeature() { verifyIndex(); return getData()->faceFeature[getIndex()]; }
    AGX_FORCE_INLINE agx::UInt8 const& ParticleGeometryContactInstance::faceFeature() const { verifyIndex(); return getData()->faceFeature[getIndex()]; }

    AGX_FORCE_INLINE agx::Bool& ParticleGeometryContactInstance::enabled() { verifyIndex(); return getData()->enabled[getIndex()]; }
    AGX_FORCE_INLINE agx::Bool const& ParticleGeometryContactInstance::enabled() const { verifyIndex(); return getData()->enabled[getIndex()]; }

    AGX_FORCE_INLINE agx::Vec3& ParticleGeometryContactInstance::point() { verifyIndex(); return getData()->point[getIndex()]; }
    AGX_FORCE_INLINE agx::Vec3 const& ParticleGeometryContactInstance::point() const { verifyIndex(); return getData()->point[getIndex()]; }

    AGX_FORCE_INLINE agx::Vec3f& ParticleGeometryContactInstance::normal() { verifyIndex(); return getData()->normal[getIndex()]; }
    AGX_FORCE_INLINE agx::Vec3f const& ParticleGeometryContactInstance::normal() const { verifyIndex(); return getData()->normal[getIndex()]; }

    AGX_FORCE_INLINE agx::Vec3f& ParticleGeometryContactInstance::tangentU() { verifyIndex(); return getData()->tangentU[getIndex()]; }
    AGX_FORCE_INLINE agx::Vec3f const& ParticleGeometryContactInstance::tangentU() const { verifyIndex(); return getData()->tangentU[getIndex()]; }

    AGX_FORCE_INLINE agx::Vec3f& ParticleGeometryContactInstance::tangentV() { verifyIndex(); return getData()->tangentV[getIndex()]; }
    AGX_FORCE_INLINE agx::Vec3f const& ParticleGeometryContactInstance::tangentV() const { verifyIndex(); return getData()->tangentV[getIndex()]; }

    AGX_FORCE_INLINE agx::Real& ParticleGeometryContactInstance::depth() { verifyIndex(); return getData()->depth[getIndex()]; }
    AGX_FORCE_INLINE agx::Real const& ParticleGeometryContactInstance::depth() const { verifyIndex(); return getData()->depth[getIndex()]; }

    AGX_FORCE_INLINE agx::Vec3f& ParticleGeometryContactInstance::velocity() { verifyIndex(); return getData()->velocity[getIndex()]; }
    AGX_FORCE_INLINE agx::Vec3f const& ParticleGeometryContactInstance::velocity() const { verifyIndex(); return getData()->velocity[getIndex()]; }

    AGX_FORCE_INLINE agx::Vec3& ParticleGeometryContactInstance::localForce() { verifyIndex(); return getData()->localForce[getIndex()]; }
    AGX_FORCE_INLINE agx::Vec3 const& ParticleGeometryContactInstance::localForce() const { verifyIndex(); return getData()->localForce[getIndex()]; }

    AGX_FORCE_INLINE agx::Real& ParticleGeometryContactInstance::impactNormalForce() { verifyIndex(); return getData()->impactNormalForce[getIndex()]; }
    AGX_FORCE_INLINE agx::Real const& ParticleGeometryContactInstance::impactNormalForce() const { verifyIndex(); return getData()->impactNormalForce[getIndex()]; }

    AGX_FORCE_INLINE agx::Real& ParticleGeometryContactInstance::contactEnergy() { verifyIndex(); return getData()->contactEnergy[getIndex()]; }
    AGX_FORCE_INLINE agx::Real const& ParticleGeometryContactInstance::contactEnergy() const { verifyIndex(); return getData()->contactEnergy[getIndex()]; }

    AGX_FORCE_INLINE agx::Bool& ParticleGeometryContactInstance::isImpacting() { verifyIndex(); return getData()->isImpacting[getIndex()]; }
    AGX_FORCE_INLINE agx::Bool const& ParticleGeometryContactInstance::isImpacting() const { verifyIndex(); return getData()->isImpacting[getIndex()]; }

    AGX_FORCE_INLINE agx::Real32& ParticleGeometryContactInstance::timeStamp() { verifyIndex(); return getData()->timeStamp[getIndex()]; }
    AGX_FORCE_INLINE agx::Real32 const& ParticleGeometryContactInstance::timeStamp() const { verifyIndex(); return getData()->timeStamp[getIndex()]; }

    AGX_FORCE_INLINE agx::Real32& ParticleGeometryContactInstance::charContactTime() { verifyIndex(); return getData()->charContactTime[getIndex()]; }
    AGX_FORCE_INLINE agx::Real32 const& ParticleGeometryContactInstance::charContactTime() const { verifyIndex(); return getData()->charContactTime[getIndex()]; }

    AGX_FORCE_INLINE agx::Int& ParticleGeometryContactInstance::next() { verifyIndex(); return getData()->next[getIndex()]; }
    AGX_FORCE_INLINE agx::Int const& ParticleGeometryContactInstance::next() const { verifyIndex(); return getData()->next[getIndex()]; }

    //-----------------------------------------------------------------------------------------------------
    AGX_FORCE_INLINE ParticleGeometryContactSemantics::ParticleGeometryContactSemantics() {}
    //-----------------------------------------------------------------------------------------------------
    DOXYGEN_END_INTERNAL_BLOCK()
  }
}

AGX_TYPE_BINDING(agx::Physics::ParticleGeometryContactPtr, "Physics.ParticleGeometryContactPtr")
AGX_TYPE_BINDING(agx::Physics::ParticleGeometryContactInstance, "Physics.ParticleGeometryContactInstance")

#ifdef _MSC_VER
# pragma warning(pop)
#endif

#undef AGX_ENTITY_WRAPPER
#undef AGX_ENTITY_NAMESPACE
#endif

