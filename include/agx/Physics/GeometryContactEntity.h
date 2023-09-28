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

#ifndef GENERATED_AGX_PHYSICS_GEOMETRYCONTACT_H_PLUGIN
#define GENERATED_AGX_PHYSICS_GEOMETRYCONTACT_H_PLUGIN

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
#include <agx/Physics/GeometryEntity.h>
#include <agx/Physics/RigidBodyEntity.h>
#include <agx/Physics/ContactPointEntity.h>
#include <agx/Physics/BroadPhasePairEntity.h>
#include <agx/Integer.h>

namespace agx { namespace Physics { class GeometryPtr; }}
namespace agx { namespace Physics { class GeometryPtr; }}
namespace agx { namespace Physics { class RigidBodyPtr; }}
namespace agx { namespace Physics { class RigidBodyPtr; }}
namespace agx { namespace Physics { class BroadPhasePairPtr; }}
namespace agx { namespace Physics { class ContactPointPtr; }}

namespace agx
{
  namespace Physics
  {

    class GeometryContactModel;
    class GeometryContactData;
    class GeometryContactPtr;
    class GeometryContactInstance;
    class GeometryContactSemantics;


    AGX_DECLARE_POINTER_TYPES(GeometryContactModel);

    /** 
    Abstract description of the data attributes for the Physics.GeometryContact entity.
    A geometry contact hold contact information between two geometries.
    */ 
    class AGXPHYSICS_EXPORT GeometryContactModel : public agx::Physics::ContactModel
    {
    public:
      typedef GeometryContactPtr PtrT;

      GeometryContactModel(const agx::String& name = "GeometryContact");

      /// \return The entity model singleton.
      static GeometryContactModel* instance();

      /// Create and return a pointer to a new instance in the default storage for this entity model.
      static GeometryContactPtr createInstance();

      /// \return The default storage for this entity model.
      static agxData::EntityStorage* defaultStorage();

      /// This is part of internal cleanup and should not be called by users
      virtual void shutdownCleanup() override;



      /* Attributes */
      static agxData::ScalarAttributeT< agx::Physics::GeometryPtr >* geometry1Attribute;
      static agxData::ScalarAttributeT< agx::Physics::GeometryPtr >* geometry2Attribute;
      static agxData::ScalarAttributeT< agx::Physics::RigidBodyPtr >* body1Attribute;
      static agxData::ScalarAttributeT< agx::Physics::RigidBodyPtr >* body2Attribute;
      static agxData::ArrayAttributeT< agx::Physics::ContactPointPtr >* pointsAttribute;
      static agxData::ScalarAttributeT< agx::Physics::BroadPhasePairPtr >* broadPhasePairAttribute;
      static agxData::ScalarAttributeT< agx::Bool >* enabledAttribute;
      static agxData::ScalarAttributeT< agx::Bool >* immediatelyAttribute;
      static agxData::ScalarAttributeT< agx::Bool >* hasSurfaceVelocityAttribute;
      static agxData::ScalarAttributeT< agx::Bool >* hasInternalMaterialAttribute;
      static agxData::ScalarAttributeT< agx::Bool >* usingPPGSAttribute;
      static agxData::PointerAttributeT< void*>* customDataAttribute;

    protected:
      virtual ~GeometryContactModel();
      virtual agxData::EntityData* createData(agxData::EntityStorage* storage) override;
      virtual void configure(agx::TiXmlElement* eEntity) override;
      virtual void initAttributeAccessors() override;
      virtual void construct(agxData::EntityPtr instance) override;
      void construct(agx::Physics::GeometryContactPtr geometryContact);
    };


    DOXYGEN_START_INTERNAL_BLOCK()
    #ifndef AGX_PHYSICS_GEOMETRYCONTACT_DATA_SET_OVERRIDE
    #define AGX_PHYSICS_GEOMETRYCONTACT_DATA_SET
    class AGXPHYSICS_EXPORT GeometryContactData : public agx::Physics::ContactData
    {
    public:
      GeometryContactInstance operator[] (size_t index);

    public:
      agxData::Array< GeometryContactPtr >& instance;
      agxData::Array< agx::Physics::GeometryPtr > geometry1;
      agxData::Array< agx::Physics::GeometryPtr > geometry2;
      agxData::Array< agx::Physics::RigidBodyPtr > body1;
      agxData::Array< agx::Physics::RigidBodyPtr > body2;
      agxData::Array< agxData::Array< agx::Physics::ContactPointPtr > > points;
      agxData::Array< agx::Physics::BroadPhasePairPtr > broadPhasePair;
      agxData::Array< agx::Bool > enabled;
      agxData::Array< agx::Bool > immediately;
      agxData::Array< agx::Bool > hasSurfaceVelocity;
      agxData::Array< agx::Bool > hasInternalMaterial;
      agxData::Array< agx::Bool > usingPPGS;
      agxData::Array< void* > customData;

    public:
      typedef agx::Physics::GeometryPtr geometry1Type;
      typedef agx::Physics::GeometryPtr geometry2Type;
      typedef agx::Physics::RigidBodyPtr body1Type;
      typedef agx::Physics::RigidBodyPtr body2Type;
      typedef agxData::Array< agx::Physics::ContactPointPtr > pointsType;
      typedef agx::Physics::BroadPhasePairPtr broadPhasePairType;
      typedef agx::Bool enabledType;
      typedef agx::Bool immediatelyType;
      typedef agx::Bool hasSurfaceVelocityType;
      typedef agx::Bool hasInternalMaterialType;
      typedef agx::Bool usingPPGSType;
      typedef void* customDataType;

    public:
      GeometryContactData(agxData::EntityStorage* storage);
      GeometryContactData();

    protected:
      virtual ~GeometryContactData() {}
      virtual void setNumElements(agx::Index numElements) override;

    private:
      GeometryContactData& operator= (const GeometryContactData&) { return *this; }

    };
    #endif
    DOXYGEN_END_INTERNAL_BLOCK()


    DOXYGEN_START_INTERNAL_BLOCK()
    class AGXPHYSICS_EXPORT GeometryContactSemantics : public agx::Physics::ContactSemantics
    {
    public:

      // Automatic getters
      agx::Physics::GeometryPtr const& getGeometry1() const;
      agx::Physics::GeometryPtr const& getGeometry2() const;
      agx::Physics::RigidBodyPtr const& getBody1() const;
      agx::Physics::RigidBodyPtr const& getBody2() const;
      agxData::Array< agx::Physics::ContactPointPtr > const& getPoints() const;
      agx::Physics::BroadPhasePairPtr const& getBroadPhasePair() const;
      agx::Bool const& getEnabled() const;
      agx::Bool const& getImmediately() const;
      agx::Bool const& getHasSurfaceVelocity() const;
      agx::Bool const& getHasInternalMaterial() const;
      agx::Bool const& getUsingPPGS() const;
      void* const& getCustomData() const;

      // Semantics defined by explicit kernels

      // Automatic setters
      void setGeometry1(agx::Physics::GeometryPtr const& value);
      void setGeometry2(agx::Physics::GeometryPtr const& value);
      void setBody1(agx::Physics::RigidBodyPtr const& value);
      void setBody2(agx::Physics::RigidBodyPtr const& value);
      void setPoints(agxData::Array< agx::Physics::ContactPointPtr > const& value);
      void setBroadPhasePair(agx::Physics::BroadPhasePairPtr const& value);
      void setEnabled(agx::Bool const& value);
      void setImmediately(agx::Bool const& value);
      void setHasSurfaceVelocity(agx::Bool const& value);
      void setHasInternalMaterial(agx::Bool const& value);
      void setUsingPPGS(agx::Bool const& value);
      void setCustomData(void* const& value);


    protected:
      friend class GeometryContactPtr;
      friend class GeometryContactInstance;
      GeometryContactSemantics();
    };
    DOXYGEN_END_INTERNAL_BLOCK()


    /**
    A geometry contact hold contact information between two geometries.
    */
    class CALLABLE GeometryContactPtr : public agx::Physics::ContactPtr
    {
    public:
      typedef GeometryContactModel ModelType;
      typedef GeometryContactData DataType;
      typedef GeometryContactInstance InstanceType;

    public:
      AGXPHYSICS_EXPORT GeometryContactPtr();
      AGXPHYSICS_EXPORT GeometryContactPtr(agxData::EntityStorage* storage, agx::Index id);
      AGXPHYSICS_EXPORT GeometryContactPtr(const agxData::EntityPtr& ptr);
      AGXPHYSICS_EXPORT GeometryContactPtr(const agxData::EntityInstance& instance);
      AGXPHYSICS_EXPORT GeometryContactPtr& operator= (const agxData::EntityPtr& ptr);
      AGXPHYSICS_EXPORT GeometryContactPtr& operator= (const agxData::EntityInstance& instance);
      AGXPHYSICS_EXPORT GeometryContactInstance instance();
      AGXPHYSICS_EXPORT const GeometryContactInstance instance() const;

      AGXPHYSICS_EXPORT GeometryContactSemantics* operator->();
      AGXPHYSICS_EXPORT const GeometryContactSemantics* operator->() const;

      GeometryContactData* getData();
      const GeometryContactData* getData() const;


      /// \return reference to the geometry1 attribute
      AGXPHYSICS_EXPORT agx::Physics::GeometryPtr& geometry1();
      /// \return const reference to the geometry1 attribute
      AGXPHYSICS_EXPORT agx::Physics::GeometryPtr const& geometry1() const;

      /// \return reference to the geometry2 attribute
      AGXPHYSICS_EXPORT agx::Physics::GeometryPtr& geometry2();
      /// \return const reference to the geometry2 attribute
      AGXPHYSICS_EXPORT agx::Physics::GeometryPtr const& geometry2() const;

      /// body1 is the rigid body of geometry1, can be assumed to be a valid rigid body.
      AGXPHYSICS_EXPORT agx::Physics::RigidBodyPtr& body1();
      /// body1 is the rigid body of geometry1, can be assumed to be a valid rigid body.
      AGXPHYSICS_EXPORT agx::Physics::RigidBodyPtr const& body1() const;

      /// body2 is the rigid body of geometry2, can be NULL.
      AGXPHYSICS_EXPORT agx::Physics::RigidBodyPtr& body2();
      /// body2 is the rigid body of geometry2, can be NULL.
      AGXPHYSICS_EXPORT agx::Physics::RigidBodyPtr const& body2() const;

      /// The list of contact points.
      AGXPHYSICS_EXPORT agxData::Array< agx::Physics::ContactPointPtr >& points();
      /// The list of contact points.
      AGXPHYSICS_EXPORT agxData::Array< agx::Physics::ContactPointPtr > const& points() const;

      /// \return reference to the broadPhasePair attribute
      AGXPHYSICS_EXPORT agx::Physics::BroadPhasePairPtr& broadPhasePair();
      /// \return const reference to the broadPhasePair attribute
      AGXPHYSICS_EXPORT agx::Physics::BroadPhasePairPtr const& broadPhasePair() const;

      /// \return reference to the enabled attribute
      AGXPHYSICS_EXPORT agx::Bool& enabled();
      /// \return const reference to the enabled attribute
      AGXPHYSICS_EXPORT agx::Bool const& enabled() const;

      /// \return reference to the immediately attribute
      AGXPHYSICS_EXPORT agx::Bool& immediately();
      /// \return const reference to the immediately attribute
      AGXPHYSICS_EXPORT agx::Bool const& immediately() const;

      /// \return reference to the hasSurfaceVelocity attribute
      AGXPHYSICS_EXPORT agx::Bool& hasSurfaceVelocity();
      /// \return const reference to the hasSurfaceVelocity attribute
      AGXPHYSICS_EXPORT agx::Bool const& hasSurfaceVelocity() const;

      /// \return reference to the hasInternalMaterial attribute
      AGXPHYSICS_EXPORT agx::Bool& hasInternalMaterial();
      /// \return const reference to the hasInternalMaterial attribute
      AGXPHYSICS_EXPORT agx::Bool const& hasInternalMaterial() const;

      /// \return reference to the usingPPGS attribute
      AGXPHYSICS_EXPORT agx::Bool& usingPPGS();
      /// \return const reference to the usingPPGS attribute
      AGXPHYSICS_EXPORT agx::Bool const& usingPPGS() const;

      /// \return reference to the customData attribute
      AGXPHYSICS_EXPORT void*& customData();
      /// \return const reference to the customData attribute
      AGXPHYSICS_EXPORT void* const& customData() const;

      DOXYGEN_START_INTERNAL_BLOCK()
      // Only used by tolua!
      #ifdef AGX_GENERATED_TOLUA_WRAPPER
      inline agx::Physics::GeometryPtr const& getGeometry1() const { return (*this)->getGeometry1(); }
      inline agx::Physics::GeometryPtr const& getGeometry2() const { return (*this)->getGeometry2(); }
      inline agx::Physics::RigidBodyPtr const& getBody1() const { return (*this)->getBody1(); }
      inline agx::Physics::RigidBodyPtr const& getBody2() const { return (*this)->getBody2(); }
      inline agxData::Array< agx::Physics::ContactPointPtr > const& getPoints() const { return (*this)->getPoints(); }
      inline agx::Physics::BroadPhasePairPtr const& getBroadPhasePair() const { return (*this)->getBroadPhasePair(); }
      inline agx::Bool const& getEnabled() const { return (*this)->getEnabled(); }
      inline agx::Bool const& getImmediately() const { return (*this)->getImmediately(); }
      inline agx::Bool const& getHasSurfaceVelocity() const { return (*this)->getHasSurfaceVelocity(); }
      inline agx::Bool const& getHasInternalMaterial() const { return (*this)->getHasInternalMaterial(); }
      inline agx::Bool const& getUsingPPGS() const { return (*this)->getUsingPPGS(); }
      inline void* const& getCustomData() const { return (*this)->getCustomData(); }


      inline void setGeometry1(agx::Physics::GeometryPtr const& value) { (*this)->setGeometry1(value); }
      inline void setGeometry2(agx::Physics::GeometryPtr const& value) { (*this)->setGeometry2(value); }
      inline void setBody1(agx::Physics::RigidBodyPtr const& value) { (*this)->setBody1(value); }
      inline void setBody2(agx::Physics::RigidBodyPtr const& value) { (*this)->setBody2(value); }
      inline void setPoints(agxData::Array< agx::Physics::ContactPointPtr > const& value) { (*this)->setPoints(value); }
      inline void setBroadPhasePair(agx::Physics::BroadPhasePairPtr const& value) { (*this)->setBroadPhasePair(value); }
      inline void setEnabled(agx::Bool const& value) { (*this)->setEnabled(value); }
      inline void setImmediately(agx::Bool const& value) { (*this)->setImmediately(value); }
      inline void setHasSurfaceVelocity(agx::Bool const& value) { (*this)->setHasSurfaceVelocity(value); }
      inline void setHasInternalMaterial(agx::Bool const& value) { (*this)->setHasInternalMaterial(value); }
      inline void setUsingPPGS(agx::Bool const& value) { (*this)->setUsingPPGS(value); }
      inline void setCustomData(void* const& value) { (*this)->setCustomData(value); }
      #endif
      DOXYGEN_END_INTERNAL_BLOCK()

    };


    DOXYGEN_START_INTERNAL_BLOCK()
    class AGXPHYSICS_EXPORT GeometryContactInstance : public agx::Physics::ContactInstance
    {
    public:
      GeometryContactInstance();
      GeometryContactInstance(GeometryContactData* data, agx::Index index);
      GeometryContactInstance(agxData::EntityStorage *storage, agx::Index index);
      GeometryContactInstance(const agxData::EntityInstance& other);
      GeometryContactInstance(const agxData::EntityPtr& ptr);

      GeometryContactData* getData();
      const GeometryContactData* getData() const;

    public:
      /// \return reference to the geometry1 attribute
      agx::Physics::GeometryPtr& geometry1();
      /// \return const reference to the geometry1 attribute
      agx::Physics::GeometryPtr const& geometry1() const;

      /// \return reference to the geometry2 attribute
      agx::Physics::GeometryPtr& geometry2();
      /// \return const reference to the geometry2 attribute
      agx::Physics::GeometryPtr const& geometry2() const;

      /// body1 is the rigid body of geometry1, can be assumed to be a valid rigid body.
      agx::Physics::RigidBodyPtr& body1();
      /// body1 is the rigid body of geometry1, can be assumed to be a valid rigid body.
      agx::Physics::RigidBodyPtr const& body1() const;

      /// body2 is the rigid body of geometry2, can be NULL.
      agx::Physics::RigidBodyPtr& body2();
      /// body2 is the rigid body of geometry2, can be NULL.
      agx::Physics::RigidBodyPtr const& body2() const;

      /// The list of contact points.
      agxData::Array< agx::Physics::ContactPointPtr >& points();
      /// The list of contact points.
      agxData::Array< agx::Physics::ContactPointPtr > const& points() const;

      /// \return reference to the broadPhasePair attribute
      agx::Physics::BroadPhasePairPtr& broadPhasePair();
      /// \return const reference to the broadPhasePair attribute
      agx::Physics::BroadPhasePairPtr const& broadPhasePair() const;

      /// \return reference to the enabled attribute
      agx::Bool& enabled();
      /// \return const reference to the enabled attribute
      agx::Bool const& enabled() const;

      /// \return reference to the immediately attribute
      agx::Bool& immediately();
      /// \return const reference to the immediately attribute
      agx::Bool const& immediately() const;

      /// \return reference to the hasSurfaceVelocity attribute
      agx::Bool& hasSurfaceVelocity();
      /// \return const reference to the hasSurfaceVelocity attribute
      agx::Bool const& hasSurfaceVelocity() const;

      /// \return reference to the hasInternalMaterial attribute
      agx::Bool& hasInternalMaterial();
      /// \return const reference to the hasInternalMaterial attribute
      agx::Bool const& hasInternalMaterial() const;

      /// \return reference to the usingPPGS attribute
      agx::Bool& usingPPGS();
      /// \return const reference to the usingPPGS attribute
      agx::Bool const& usingPPGS() const;

      /// \return reference to the customData attribute
      void*& customData();
      /// \return const reference to the customData attribute
      void* const& customData() const;

    };
    DOXYGEN_END_INTERNAL_BLOCK()



    typedef agx::VectorPOD<GeometryContactPtr> GeometryContactPtrVector;
    typedef agxData::Array<GeometryContactPtr> GeometryContactPtrArray;



    DOXYGEN_START_INTERNAL_BLOCK()
    /* Implementation */
    //-----------------------------------------------------------------------------------------------------
    //-----------------------------------------------------------------------------------------------------
    inline GeometryContactInstance agx::Physics::GeometryContactData::operator[] (size_t index) { return GeometryContactInstance(this, (agx::Index)index); }
    //-----------------------------------------------------------------------------------------------------
    AGX_FORCE_INLINE GeometryContactPtr::GeometryContactPtr() {}
    AGX_FORCE_INLINE GeometryContactPtr::GeometryContactPtr(agxData::EntityStorage* storage, agx::Index id) : agx::Physics::ContactPtr(storage, id) {}
    AGX_FORCE_INLINE GeometryContactPtr::GeometryContactPtr(const agxData::EntityPtr& ptr) : agx::Physics::ContactPtr(ptr)
    {
      agxAssertN(!ptr || ptr.isInstanceOf(GeometryContactModel::instance()),
        "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
        EntityPtr::getModel()->fullPath().c_str(), GeometryContactModel::instance()->fullPath().c_str());
    }

    AGX_FORCE_INLINE GeometryContactPtr::GeometryContactPtr(const agxData::EntityInstance& instance) : agx::Physics::ContactPtr(instance)
    {
      agxAssertN(!instance || instance.isInstanceOf(GeometryContactModel::instance()),
        "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
        EntityPtr::getModel()->fullPath().c_str(), GeometryContactModel::instance()->fullPath().c_str());
    }

    AGX_FORCE_INLINE GeometryContactPtr& GeometryContactPtr::operator= (const agxData::EntityPtr& ptr)
    {
      agxData::EntityPtr::operator= (ptr);
      agxAssertN(!ptr || ptr.isInstanceOf(GeometryContactModel::instance()),
        "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
        EntityPtr::getModel()->fullPath().c_str(), GeometryContactModel::instance()->fullPath().c_str());
      return *this;
    }

    AGX_FORCE_INLINE GeometryContactPtr& GeometryContactPtr::operator= (const agxData::EntityInstance& instance)
    {
      agxData::EntityPtr::operator= (instance);
      agxAssertN(!instance || instance.isInstanceOf(GeometryContactModel::instance()),
        "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
        EntityPtr::getModel()->fullPath().c_str(), GeometryContactModel::instance()->fullPath().c_str());
      return *this;
    }

    inline GeometryContactInstance GeometryContactPtr::instance() { return agxData::EntityPtr::instance(); }
    inline const GeometryContactInstance GeometryContactPtr::instance() const { return agxData::EntityPtr::instance(); }
    AGX_FORCE_INLINE GeometryContactSemantics* GeometryContactPtr::operator->() { return (GeometryContactSemantics* )this; }
    AGX_FORCE_INLINE const GeometryContactSemantics* GeometryContactPtr::operator->() const { return (const GeometryContactSemantics* )this; }
    AGX_FORCE_INLINE GeometryContactData* GeometryContactPtr::getData() { return static_cast<GeometryContactData* >(agxData::EntityPtr::getData()); }
    AGX_FORCE_INLINE const GeometryContactData* GeometryContactPtr::getData() const { return static_cast<const GeometryContactData* >(agxData::EntityPtr::getData()); }

    AGX_FORCE_INLINE agx::Physics::GeometryPtr& GeometryContactPtr::geometry1() { verifyIndex(); return getData()->geometry1[calculateIndex()]; }
    AGX_FORCE_INLINE agx::Physics::GeometryPtr const& GeometryContactPtr::geometry1() const { verifyIndex(); return getData()->geometry1[calculateIndex()]; }

    AGX_FORCE_INLINE agx::Physics::GeometryPtr& GeometryContactPtr::geometry2() { verifyIndex(); return getData()->geometry2[calculateIndex()]; }
    AGX_FORCE_INLINE agx::Physics::GeometryPtr const& GeometryContactPtr::geometry2() const { verifyIndex(); return getData()->geometry2[calculateIndex()]; }

    AGX_FORCE_INLINE agx::Physics::RigidBodyPtr& GeometryContactPtr::body1() { verifyIndex(); return getData()->body1[calculateIndex()]; }
    AGX_FORCE_INLINE agx::Physics::RigidBodyPtr const& GeometryContactPtr::body1() const { verifyIndex(); return getData()->body1[calculateIndex()]; }

    AGX_FORCE_INLINE agx::Physics::RigidBodyPtr& GeometryContactPtr::body2() { verifyIndex(); return getData()->body2[calculateIndex()]; }
    AGX_FORCE_INLINE agx::Physics::RigidBodyPtr const& GeometryContactPtr::body2() const { verifyIndex(); return getData()->body2[calculateIndex()]; }

    AGX_FORCE_INLINE agxData::Array< agx::Physics::ContactPointPtr >& GeometryContactPtr::points() { verifyIndex(); return getData()->points[calculateIndex()]; }
    AGX_FORCE_INLINE agxData::Array< agx::Physics::ContactPointPtr > const& GeometryContactPtr::points() const { verifyIndex(); return getData()->points[calculateIndex()]; }

    AGX_FORCE_INLINE agx::Physics::BroadPhasePairPtr& GeometryContactPtr::broadPhasePair() { verifyIndex(); return getData()->broadPhasePair[calculateIndex()]; }
    AGX_FORCE_INLINE agx::Physics::BroadPhasePairPtr const& GeometryContactPtr::broadPhasePair() const { verifyIndex(); return getData()->broadPhasePair[calculateIndex()]; }

    AGX_FORCE_INLINE agx::Bool& GeometryContactPtr::enabled() { verifyIndex(); return getData()->enabled[calculateIndex()]; }
    AGX_FORCE_INLINE agx::Bool const& GeometryContactPtr::enabled() const { verifyIndex(); return getData()->enabled[calculateIndex()]; }

    AGX_FORCE_INLINE agx::Bool& GeometryContactPtr::immediately() { verifyIndex(); return getData()->immediately[calculateIndex()]; }
    AGX_FORCE_INLINE agx::Bool const& GeometryContactPtr::immediately() const { verifyIndex(); return getData()->immediately[calculateIndex()]; }

    AGX_FORCE_INLINE agx::Bool& GeometryContactPtr::hasSurfaceVelocity() { verifyIndex(); return getData()->hasSurfaceVelocity[calculateIndex()]; }
    AGX_FORCE_INLINE agx::Bool const& GeometryContactPtr::hasSurfaceVelocity() const { verifyIndex(); return getData()->hasSurfaceVelocity[calculateIndex()]; }

    AGX_FORCE_INLINE agx::Bool& GeometryContactPtr::hasInternalMaterial() { verifyIndex(); return getData()->hasInternalMaterial[calculateIndex()]; }
    AGX_FORCE_INLINE agx::Bool const& GeometryContactPtr::hasInternalMaterial() const { verifyIndex(); return getData()->hasInternalMaterial[calculateIndex()]; }

    AGX_FORCE_INLINE agx::Bool& GeometryContactPtr::usingPPGS() { verifyIndex(); return getData()->usingPPGS[calculateIndex()]; }
    AGX_FORCE_INLINE agx::Bool const& GeometryContactPtr::usingPPGS() const { verifyIndex(); return getData()->usingPPGS[calculateIndex()]; }

    AGX_FORCE_INLINE void*& GeometryContactPtr::customData() { verifyIndex(); return getData()->customData[calculateIndex()]; }
    AGX_FORCE_INLINE void* const& GeometryContactPtr::customData() const { verifyIndex(); return getData()->customData[calculateIndex()]; }

    //-----------------------------------------------------------------------------------------------------
    AGX_FORCE_INLINE GeometryContactInstance::GeometryContactInstance() {}
    AGX_FORCE_INLINE GeometryContactInstance::GeometryContactInstance(GeometryContactData* data, agx::Index index) : agx::Physics::ContactInstance(data, index) {}
    AGX_FORCE_INLINE GeometryContactInstance::GeometryContactInstance(agxData::EntityStorage* storage, agx::Index index) : agx::Physics::ContactInstance(storage, index) {}
    AGX_FORCE_INLINE GeometryContactInstance::GeometryContactInstance(const agxData::EntityInstance& other) : agx::Physics::ContactInstance(other)
    {
      agxAssertN(!other || other.isInstanceOf(GeometryContactModel::instance()),
        "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
        EntityInstance::getModel()->fullPath().c_str(), GeometryContactModel::instance()->fullPath().c_str());
    }

    AGX_FORCE_INLINE GeometryContactInstance::GeometryContactInstance(const agxData::EntityPtr& ptr) : agx::Physics::ContactInstance(ptr)
    {
      agxAssertN(!ptr || ptr.isInstanceOf(GeometryContactModel::instance()),
        "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
        EntityInstance::getModel()->fullPath().c_str(), GeometryContactModel::instance()->fullPath().c_str());
    }


    AGX_FORCE_INLINE GeometryContactData* GeometryContactInstance::getData() { return static_cast<GeometryContactData* >(agxData::EntityInstance::getData()); }
    AGX_FORCE_INLINE const GeometryContactData* GeometryContactInstance::getData() const { return static_cast<const GeometryContactData* >(agxData::EntityInstance::getData()); }

    AGX_FORCE_INLINE agx::Physics::GeometryPtr& GeometryContactInstance::geometry1() { verifyIndex(); return getData()->geometry1[getIndex()]; }
    AGX_FORCE_INLINE agx::Physics::GeometryPtr const& GeometryContactInstance::geometry1() const { verifyIndex(); return getData()->geometry1[getIndex()]; }

    AGX_FORCE_INLINE agx::Physics::GeometryPtr& GeometryContactInstance::geometry2() { verifyIndex(); return getData()->geometry2[getIndex()]; }
    AGX_FORCE_INLINE agx::Physics::GeometryPtr const& GeometryContactInstance::geometry2() const { verifyIndex(); return getData()->geometry2[getIndex()]; }

    AGX_FORCE_INLINE agx::Physics::RigidBodyPtr& GeometryContactInstance::body1() { verifyIndex(); return getData()->body1[getIndex()]; }
    AGX_FORCE_INLINE agx::Physics::RigidBodyPtr const& GeometryContactInstance::body1() const { verifyIndex(); return getData()->body1[getIndex()]; }

    AGX_FORCE_INLINE agx::Physics::RigidBodyPtr& GeometryContactInstance::body2() { verifyIndex(); return getData()->body2[getIndex()]; }
    AGX_FORCE_INLINE agx::Physics::RigidBodyPtr const& GeometryContactInstance::body2() const { verifyIndex(); return getData()->body2[getIndex()]; }

    AGX_FORCE_INLINE agxData::Array< agx::Physics::ContactPointPtr >& GeometryContactInstance::points() { verifyIndex(); return getData()->points[getIndex()]; }
    AGX_FORCE_INLINE agxData::Array< agx::Physics::ContactPointPtr > const& GeometryContactInstance::points() const { verifyIndex(); return getData()->points[getIndex()]; }

    AGX_FORCE_INLINE agx::Physics::BroadPhasePairPtr& GeometryContactInstance::broadPhasePair() { verifyIndex(); return getData()->broadPhasePair[getIndex()]; }
    AGX_FORCE_INLINE agx::Physics::BroadPhasePairPtr const& GeometryContactInstance::broadPhasePair() const { verifyIndex(); return getData()->broadPhasePair[getIndex()]; }

    AGX_FORCE_INLINE agx::Bool& GeometryContactInstance::enabled() { verifyIndex(); return getData()->enabled[getIndex()]; }
    AGX_FORCE_INLINE agx::Bool const& GeometryContactInstance::enabled() const { verifyIndex(); return getData()->enabled[getIndex()]; }

    AGX_FORCE_INLINE agx::Bool& GeometryContactInstance::immediately() { verifyIndex(); return getData()->immediately[getIndex()]; }
    AGX_FORCE_INLINE agx::Bool const& GeometryContactInstance::immediately() const { verifyIndex(); return getData()->immediately[getIndex()]; }

    AGX_FORCE_INLINE agx::Bool& GeometryContactInstance::hasSurfaceVelocity() { verifyIndex(); return getData()->hasSurfaceVelocity[getIndex()]; }
    AGX_FORCE_INLINE agx::Bool const& GeometryContactInstance::hasSurfaceVelocity() const { verifyIndex(); return getData()->hasSurfaceVelocity[getIndex()]; }

    AGX_FORCE_INLINE agx::Bool& GeometryContactInstance::hasInternalMaterial() { verifyIndex(); return getData()->hasInternalMaterial[getIndex()]; }
    AGX_FORCE_INLINE agx::Bool const& GeometryContactInstance::hasInternalMaterial() const { verifyIndex(); return getData()->hasInternalMaterial[getIndex()]; }

    AGX_FORCE_INLINE agx::Bool& GeometryContactInstance::usingPPGS() { verifyIndex(); return getData()->usingPPGS[getIndex()]; }
    AGX_FORCE_INLINE agx::Bool const& GeometryContactInstance::usingPPGS() const { verifyIndex(); return getData()->usingPPGS[getIndex()]; }

    AGX_FORCE_INLINE void*& GeometryContactInstance::customData() { verifyIndex(); return getData()->customData[getIndex()]; }
    AGX_FORCE_INLINE void* const& GeometryContactInstance::customData() const { verifyIndex(); return getData()->customData[getIndex()]; }

    //-----------------------------------------------------------------------------------------------------
    AGX_FORCE_INLINE GeometryContactSemantics::GeometryContactSemantics() {}
    //-----------------------------------------------------------------------------------------------------
    DOXYGEN_END_INTERNAL_BLOCK()
  }
}

AGX_TYPE_BINDING(agx::Physics::GeometryContactPtr, "Physics.GeometryContactPtr")
AGX_TYPE_BINDING(agx::Physics::GeometryContactInstance, "Physics.GeometryContactInstance")

#ifdef _MSC_VER
# pragma warning(pop)
#endif

#undef AGX_ENTITY_WRAPPER
#undef AGX_ENTITY_NAMESPACE
#endif

