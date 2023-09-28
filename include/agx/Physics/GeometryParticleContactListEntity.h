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

#ifndef GENERATED_AGX_PHYSICS_GEOMETRYPARTICLECONTACTLIST_H_PLUGIN
#define GENERATED_AGX_PHYSICS_GEOMETRYPARTICLECONTACTLIST_H_PLUGIN

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
#include <agx/Physics/GeometryEntity.h>
#include <agx/Physics/RigidBodyEntity.h>
#include <agx/Vec3.h>
#include <agx/Integer.h>
#include <agx/Physics/ParticleGeometryContactEntity.h>

namespace agx { namespace Physics { class GeometryPtr; }}
namespace agx { namespace Physics { class RigidBodyPtr; }}
namespace agx { namespace Physics { class ParticleGeometryContactPtr; }}

namespace agx
{
  namespace Physics
  {

    class GeometryParticleContactListModel;
    class GeometryParticleContactListData;
    class GeometryParticleContactListPtr;
    class GeometryParticleContactListInstance;
    class GeometryParticleContactListSemantics;


    AGX_DECLARE_POINTER_TYPES(GeometryParticleContactListModel);

    /** 
    Abstract description of the data attributes for the Physics.GeometryParticleContactList entity.
    */ 
    class AGXPHYSICS_EXPORT GeometryParticleContactListModel : public agx::Physics::InteractionModel
    {
    public:
      typedef GeometryParticleContactListPtr PtrT;

      GeometryParticleContactListModel(const agx::String& name = "GeometryParticleContactList");

      /// \return The entity model singleton.
      static GeometryParticleContactListModel* instance();

      /// Create and return a pointer to a new instance in the default storage for this entity model.
      static GeometryParticleContactListPtr createInstance();

      /// \return The default storage for this entity model.
      static agxData::EntityStorage* defaultStorage();

      /// This is part of internal cleanup and should not be called by users
      virtual void shutdownCleanup() override;



      /* Attributes */
      static agxData::ScalarAttributeT< agx::Physics::GeometryPtr >* geometryAttribute;
      static agxData::ScalarAttributeT< agx::Physics::RigidBodyPtr >* bodyAttribute;
      static agxData::ScalarAttributeT< agx::Vec3 >* bodyForceAttribute;
      static agxData::ScalarAttributeT< agx::Vec3 >* bodyTorqueAttribute;
      static agxData::ScalarAttributeT< agx::Bool >* isImpactingAttribute;
      static agxData::ArrayAttributeT< agx::Physics::ParticleGeometryContactPtr >* contactsAttribute;

    protected:
      virtual ~GeometryParticleContactListModel();
      virtual agxData::EntityData* createData(agxData::EntityStorage* storage) override;
      virtual void configure(agx::TiXmlElement* eEntity) override;
      virtual void initAttributeAccessors() override;
      virtual void construct(agxData::EntityPtr instance) override;
      void construct(agx::Physics::GeometryParticleContactListPtr geometryParticleContactList);
    };


    DOXYGEN_START_INTERNAL_BLOCK()
    #ifndef AGX_PHYSICS_GEOMETRYPARTICLECONTACTLIST_DATA_SET_OVERRIDE
    #define AGX_PHYSICS_GEOMETRYPARTICLECONTACTLIST_DATA_SET
    class AGXPHYSICS_EXPORT GeometryParticleContactListData : public agx::Physics::InteractionData
    {
    public:
      GeometryParticleContactListInstance operator[] (size_t index);

    public:
      agxData::Array< GeometryParticleContactListPtr >& instance;
      agxData::Array< agx::Physics::GeometryPtr > geometry;
      agxData::Array< agx::Physics::RigidBodyPtr > body;
      agxData::Array< agx::Vec3 > bodyForce;
      agxData::Array< agx::Vec3 > bodyTorque;
      agxData::Array< agx::Bool > isImpacting;
      agxData::Array< agxData::Array< agx::Physics::ParticleGeometryContactPtr > > contacts;

    public:
      typedef agx::Physics::GeometryPtr geometryType;
      typedef agx::Physics::RigidBodyPtr bodyType;
      typedef agx::Vec3 bodyForceType;
      typedef agx::Vec3 bodyTorqueType;
      typedef agx::Bool isImpactingType;
      typedef agxData::Array< agx::Physics::ParticleGeometryContactPtr > contactsType;

    public:
      GeometryParticleContactListData(agxData::EntityStorage* storage);
      GeometryParticleContactListData();

    protected:
      virtual ~GeometryParticleContactListData() {}
      virtual void setNumElements(agx::Index numElements) override;

    private:
      GeometryParticleContactListData& operator= (const GeometryParticleContactListData&) { return *this; }

    };
    #endif
    DOXYGEN_END_INTERNAL_BLOCK()


    DOXYGEN_START_INTERNAL_BLOCK()
    class AGXPHYSICS_EXPORT GeometryParticleContactListSemantics : public agx::Physics::InteractionSemantics
    {
    public:

      // Automatic getters
      agx::Physics::GeometryPtr const& getGeometry() const;
      agx::Physics::RigidBodyPtr const& getBody() const;
      agx::Vec3 const& getBodyForce() const;
      agx::Vec3 const& getBodyTorque() const;
      agx::Bool const& getIsImpacting() const;
      agxData::Array< agx::Physics::ParticleGeometryContactPtr > const& getContacts() const;

      // Semantics defined by explicit kernels

      // Automatic setters
      void setGeometry(agx::Physics::GeometryPtr const& value);
      void setBody(agx::Physics::RigidBodyPtr const& value);
      void setBodyForce(agx::Vec3 const& value);
      void setBodyTorque(agx::Vec3 const& value);
      void setIsImpacting(agx::Bool const& value);
      void setContacts(agxData::Array< agx::Physics::ParticleGeometryContactPtr > const& value);


    protected:
      friend class GeometryParticleContactListPtr;
      friend class GeometryParticleContactListInstance;
      GeometryParticleContactListSemantics();
    };
    DOXYGEN_END_INTERNAL_BLOCK()


    /**
    Pointer to a entity instance of type Physics.GeometryParticleContactList
    */
    class CALLABLE GeometryParticleContactListPtr : public agx::Physics::InteractionPtr
    {
    public:
      typedef GeometryParticleContactListModel ModelType;
      typedef GeometryParticleContactListData DataType;
      typedef GeometryParticleContactListInstance InstanceType;

    public:
      AGXPHYSICS_EXPORT GeometryParticleContactListPtr();
      AGXPHYSICS_EXPORT GeometryParticleContactListPtr(agxData::EntityStorage* storage, agx::Index id);
      AGXPHYSICS_EXPORT GeometryParticleContactListPtr(const agxData::EntityPtr& ptr);
      AGXPHYSICS_EXPORT GeometryParticleContactListPtr(const agxData::EntityInstance& instance);
      AGXPHYSICS_EXPORT GeometryParticleContactListPtr& operator= (const agxData::EntityPtr& ptr);
      AGXPHYSICS_EXPORT GeometryParticleContactListPtr& operator= (const agxData::EntityInstance& instance);
      AGXPHYSICS_EXPORT GeometryParticleContactListInstance instance();
      AGXPHYSICS_EXPORT const GeometryParticleContactListInstance instance() const;

      AGXPHYSICS_EXPORT GeometryParticleContactListSemantics* operator->();
      AGXPHYSICS_EXPORT const GeometryParticleContactListSemantics* operator->() const;

      GeometryParticleContactListData* getData();
      const GeometryParticleContactListData* getData() const;


      /// \return reference to the geometry attribute
      AGXPHYSICS_EXPORT agx::Physics::GeometryPtr& geometry();
      /// \return const reference to the geometry attribute
      AGXPHYSICS_EXPORT agx::Physics::GeometryPtr const& geometry() const;

      /// \return reference to the body attribute
      AGXPHYSICS_EXPORT agx::Physics::RigidBodyPtr& body();
      /// \return const reference to the body attribute
      AGXPHYSICS_EXPORT agx::Physics::RigidBodyPtr const& body() const;

      /// \return reference to the bodyForce attribute
      AGXPHYSICS_EXPORT agx::Vec3& bodyForce();
      /// \return const reference to the bodyForce attribute
      AGXPHYSICS_EXPORT agx::Vec3 const& bodyForce() const;

      /// \return reference to the bodyTorque attribute
      AGXPHYSICS_EXPORT agx::Vec3& bodyTorque();
      /// \return const reference to the bodyTorque attribute
      AGXPHYSICS_EXPORT agx::Vec3 const& bodyTorque() const;

      /// \return reference to the isImpacting attribute
      AGXPHYSICS_EXPORT agx::Bool& isImpacting();
      /// \return const reference to the isImpacting attribute
      AGXPHYSICS_EXPORT agx::Bool const& isImpacting() const;

      /// \return reference to the contacts attribute
      AGXPHYSICS_EXPORT agxData::Array< agx::Physics::ParticleGeometryContactPtr >& contacts();
      /// \return const reference to the contacts attribute
      AGXPHYSICS_EXPORT agxData::Array< agx::Physics::ParticleGeometryContactPtr > const& contacts() const;

      DOXYGEN_START_INTERNAL_BLOCK()
      // Only used by tolua!
      #ifdef AGX_GENERATED_TOLUA_WRAPPER
      inline agx::Physics::GeometryPtr const& getGeometry() const { return (*this)->getGeometry(); }
      inline agx::Physics::RigidBodyPtr const& getBody() const { return (*this)->getBody(); }
      inline agx::Vec3 const& getBodyForce() const { return (*this)->getBodyForce(); }
      inline agx::Vec3 const& getBodyTorque() const { return (*this)->getBodyTorque(); }
      inline agx::Bool const& getIsImpacting() const { return (*this)->getIsImpacting(); }
      inline agxData::Array< agx::Physics::ParticleGeometryContactPtr > const& getContacts() const { return (*this)->getContacts(); }


      inline void setGeometry(agx::Physics::GeometryPtr const& value) { (*this)->setGeometry(value); }
      inline void setBody(agx::Physics::RigidBodyPtr const& value) { (*this)->setBody(value); }
      inline void setBodyForce(agx::Vec3 const& value) { (*this)->setBodyForce(value); }
      inline void setBodyTorque(agx::Vec3 const& value) { (*this)->setBodyTorque(value); }
      inline void setIsImpacting(agx::Bool const& value) { (*this)->setIsImpacting(value); }
      inline void setContacts(agxData::Array< agx::Physics::ParticleGeometryContactPtr > const& value) { (*this)->setContacts(value); }
      #endif
      DOXYGEN_END_INTERNAL_BLOCK()

    };


    DOXYGEN_START_INTERNAL_BLOCK()
    class AGXPHYSICS_EXPORT GeometryParticleContactListInstance : public agx::Physics::InteractionInstance
    {
    public:
      GeometryParticleContactListInstance();
      GeometryParticleContactListInstance(GeometryParticleContactListData* data, agx::Index index);
      GeometryParticleContactListInstance(agxData::EntityStorage *storage, agx::Index index);
      GeometryParticleContactListInstance(const agxData::EntityInstance& other);
      GeometryParticleContactListInstance(const agxData::EntityPtr& ptr);

      GeometryParticleContactListData* getData();
      const GeometryParticleContactListData* getData() const;

    public:
      /// \return reference to the geometry attribute
      agx::Physics::GeometryPtr& geometry();
      /// \return const reference to the geometry attribute
      agx::Physics::GeometryPtr const& geometry() const;

      /// \return reference to the body attribute
      agx::Physics::RigidBodyPtr& body();
      /// \return const reference to the body attribute
      agx::Physics::RigidBodyPtr const& body() const;

      /// \return reference to the bodyForce attribute
      agx::Vec3& bodyForce();
      /// \return const reference to the bodyForce attribute
      agx::Vec3 const& bodyForce() const;

      /// \return reference to the bodyTorque attribute
      agx::Vec3& bodyTorque();
      /// \return const reference to the bodyTorque attribute
      agx::Vec3 const& bodyTorque() const;

      /// \return reference to the isImpacting attribute
      agx::Bool& isImpacting();
      /// \return const reference to the isImpacting attribute
      agx::Bool const& isImpacting() const;

      /// \return reference to the contacts attribute
      agxData::Array< agx::Physics::ParticleGeometryContactPtr >& contacts();
      /// \return const reference to the contacts attribute
      agxData::Array< agx::Physics::ParticleGeometryContactPtr > const& contacts() const;

    };
    DOXYGEN_END_INTERNAL_BLOCK()



    typedef agx::VectorPOD<GeometryParticleContactListPtr> GeometryParticleContactListPtrVector;
    typedef agxData::Array<GeometryParticleContactListPtr> GeometryParticleContactListPtrArray;



    DOXYGEN_START_INTERNAL_BLOCK()
    /* Implementation */
    //-----------------------------------------------------------------------------------------------------
    //-----------------------------------------------------------------------------------------------------
    inline GeometryParticleContactListInstance agx::Physics::GeometryParticleContactListData::operator[] (size_t index) { return GeometryParticleContactListInstance(this, (agx::Index)index); }
    //-----------------------------------------------------------------------------------------------------
    AGX_FORCE_INLINE GeometryParticleContactListPtr::GeometryParticleContactListPtr() {}
    AGX_FORCE_INLINE GeometryParticleContactListPtr::GeometryParticleContactListPtr(agxData::EntityStorage* storage, agx::Index id) : agx::Physics::InteractionPtr(storage, id) {}
    AGX_FORCE_INLINE GeometryParticleContactListPtr::GeometryParticleContactListPtr(const agxData::EntityPtr& ptr) : agx::Physics::InteractionPtr(ptr)
    {
      agxAssertN(!ptr || ptr.isInstanceOf(GeometryParticleContactListModel::instance()),
        "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
        EntityPtr::getModel()->fullPath().c_str(), GeometryParticleContactListModel::instance()->fullPath().c_str());
    }

    AGX_FORCE_INLINE GeometryParticleContactListPtr::GeometryParticleContactListPtr(const agxData::EntityInstance& instance) : agx::Physics::InteractionPtr(instance)
    {
      agxAssertN(!instance || instance.isInstanceOf(GeometryParticleContactListModel::instance()),
        "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
        EntityPtr::getModel()->fullPath().c_str(), GeometryParticleContactListModel::instance()->fullPath().c_str());
    }

    AGX_FORCE_INLINE GeometryParticleContactListPtr& GeometryParticleContactListPtr::operator= (const agxData::EntityPtr& ptr)
    {
      agxData::EntityPtr::operator= (ptr);
      agxAssertN(!ptr || ptr.isInstanceOf(GeometryParticleContactListModel::instance()),
        "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
        EntityPtr::getModel()->fullPath().c_str(), GeometryParticleContactListModel::instance()->fullPath().c_str());
      return *this;
    }

    AGX_FORCE_INLINE GeometryParticleContactListPtr& GeometryParticleContactListPtr::operator= (const agxData::EntityInstance& instance)
    {
      agxData::EntityPtr::operator= (instance);
      agxAssertN(!instance || instance.isInstanceOf(GeometryParticleContactListModel::instance()),
        "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
        EntityPtr::getModel()->fullPath().c_str(), GeometryParticleContactListModel::instance()->fullPath().c_str());
      return *this;
    }

    inline GeometryParticleContactListInstance GeometryParticleContactListPtr::instance() { return agxData::EntityPtr::instance(); }
    inline const GeometryParticleContactListInstance GeometryParticleContactListPtr::instance() const { return agxData::EntityPtr::instance(); }
    AGX_FORCE_INLINE GeometryParticleContactListSemantics* GeometryParticleContactListPtr::operator->() { return (GeometryParticleContactListSemantics* )this; }
    AGX_FORCE_INLINE const GeometryParticleContactListSemantics* GeometryParticleContactListPtr::operator->() const { return (const GeometryParticleContactListSemantics* )this; }
    AGX_FORCE_INLINE GeometryParticleContactListData* GeometryParticleContactListPtr::getData() { return static_cast<GeometryParticleContactListData* >(agxData::EntityPtr::getData()); }
    AGX_FORCE_INLINE const GeometryParticleContactListData* GeometryParticleContactListPtr::getData() const { return static_cast<const GeometryParticleContactListData* >(agxData::EntityPtr::getData()); }

    AGX_FORCE_INLINE agx::Physics::GeometryPtr& GeometryParticleContactListPtr::geometry() { verifyIndex(); return getData()->geometry[calculateIndex()]; }
    AGX_FORCE_INLINE agx::Physics::GeometryPtr const& GeometryParticleContactListPtr::geometry() const { verifyIndex(); return getData()->geometry[calculateIndex()]; }

    AGX_FORCE_INLINE agx::Physics::RigidBodyPtr& GeometryParticleContactListPtr::body() { verifyIndex(); return getData()->body[calculateIndex()]; }
    AGX_FORCE_INLINE agx::Physics::RigidBodyPtr const& GeometryParticleContactListPtr::body() const { verifyIndex(); return getData()->body[calculateIndex()]; }

    AGX_FORCE_INLINE agx::Vec3& GeometryParticleContactListPtr::bodyForce() { verifyIndex(); return getData()->bodyForce[calculateIndex()]; }
    AGX_FORCE_INLINE agx::Vec3 const& GeometryParticleContactListPtr::bodyForce() const { verifyIndex(); return getData()->bodyForce[calculateIndex()]; }

    AGX_FORCE_INLINE agx::Vec3& GeometryParticleContactListPtr::bodyTorque() { verifyIndex(); return getData()->bodyTorque[calculateIndex()]; }
    AGX_FORCE_INLINE agx::Vec3 const& GeometryParticleContactListPtr::bodyTorque() const { verifyIndex(); return getData()->bodyTorque[calculateIndex()]; }

    AGX_FORCE_INLINE agx::Bool& GeometryParticleContactListPtr::isImpacting() { verifyIndex(); return getData()->isImpacting[calculateIndex()]; }
    AGX_FORCE_INLINE agx::Bool const& GeometryParticleContactListPtr::isImpacting() const { verifyIndex(); return getData()->isImpacting[calculateIndex()]; }

    AGX_FORCE_INLINE agxData::Array< agx::Physics::ParticleGeometryContactPtr >& GeometryParticleContactListPtr::contacts() { verifyIndex(); return getData()->contacts[calculateIndex()]; }
    AGX_FORCE_INLINE agxData::Array< agx::Physics::ParticleGeometryContactPtr > const& GeometryParticleContactListPtr::contacts() const { verifyIndex(); return getData()->contacts[calculateIndex()]; }

    //-----------------------------------------------------------------------------------------------------
    AGX_FORCE_INLINE GeometryParticleContactListInstance::GeometryParticleContactListInstance() {}
    AGX_FORCE_INLINE GeometryParticleContactListInstance::GeometryParticleContactListInstance(GeometryParticleContactListData* data, agx::Index index) : agx::Physics::InteractionInstance(data, index) {}
    AGX_FORCE_INLINE GeometryParticleContactListInstance::GeometryParticleContactListInstance(agxData::EntityStorage* storage, agx::Index index) : agx::Physics::InteractionInstance(storage, index) {}
    AGX_FORCE_INLINE GeometryParticleContactListInstance::GeometryParticleContactListInstance(const agxData::EntityInstance& other) : agx::Physics::InteractionInstance(other)
    {
      agxAssertN(!other || other.isInstanceOf(GeometryParticleContactListModel::instance()),
        "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
        EntityInstance::getModel()->fullPath().c_str(), GeometryParticleContactListModel::instance()->fullPath().c_str());
    }

    AGX_FORCE_INLINE GeometryParticleContactListInstance::GeometryParticleContactListInstance(const agxData::EntityPtr& ptr) : agx::Physics::InteractionInstance(ptr)
    {
      agxAssertN(!ptr || ptr.isInstanceOf(GeometryParticleContactListModel::instance()),
        "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
        EntityInstance::getModel()->fullPath().c_str(), GeometryParticleContactListModel::instance()->fullPath().c_str());
    }


    AGX_FORCE_INLINE GeometryParticleContactListData* GeometryParticleContactListInstance::getData() { return static_cast<GeometryParticleContactListData* >(agxData::EntityInstance::getData()); }
    AGX_FORCE_INLINE const GeometryParticleContactListData* GeometryParticleContactListInstance::getData() const { return static_cast<const GeometryParticleContactListData* >(agxData::EntityInstance::getData()); }

    AGX_FORCE_INLINE agx::Physics::GeometryPtr& GeometryParticleContactListInstance::geometry() { verifyIndex(); return getData()->geometry[getIndex()]; }
    AGX_FORCE_INLINE agx::Physics::GeometryPtr const& GeometryParticleContactListInstance::geometry() const { verifyIndex(); return getData()->geometry[getIndex()]; }

    AGX_FORCE_INLINE agx::Physics::RigidBodyPtr& GeometryParticleContactListInstance::body() { verifyIndex(); return getData()->body[getIndex()]; }
    AGX_FORCE_INLINE agx::Physics::RigidBodyPtr const& GeometryParticleContactListInstance::body() const { verifyIndex(); return getData()->body[getIndex()]; }

    AGX_FORCE_INLINE agx::Vec3& GeometryParticleContactListInstance::bodyForce() { verifyIndex(); return getData()->bodyForce[getIndex()]; }
    AGX_FORCE_INLINE agx::Vec3 const& GeometryParticleContactListInstance::bodyForce() const { verifyIndex(); return getData()->bodyForce[getIndex()]; }

    AGX_FORCE_INLINE agx::Vec3& GeometryParticleContactListInstance::bodyTorque() { verifyIndex(); return getData()->bodyTorque[getIndex()]; }
    AGX_FORCE_INLINE agx::Vec3 const& GeometryParticleContactListInstance::bodyTorque() const { verifyIndex(); return getData()->bodyTorque[getIndex()]; }

    AGX_FORCE_INLINE agx::Bool& GeometryParticleContactListInstance::isImpacting() { verifyIndex(); return getData()->isImpacting[getIndex()]; }
    AGX_FORCE_INLINE agx::Bool const& GeometryParticleContactListInstance::isImpacting() const { verifyIndex(); return getData()->isImpacting[getIndex()]; }

    AGX_FORCE_INLINE agxData::Array< agx::Physics::ParticleGeometryContactPtr >& GeometryParticleContactListInstance::contacts() { verifyIndex(); return getData()->contacts[getIndex()]; }
    AGX_FORCE_INLINE agxData::Array< agx::Physics::ParticleGeometryContactPtr > const& GeometryParticleContactListInstance::contacts() const { verifyIndex(); return getData()->contacts[getIndex()]; }

    //-----------------------------------------------------------------------------------------------------
    AGX_FORCE_INLINE GeometryParticleContactListSemantics::GeometryParticleContactListSemantics() {}
    //-----------------------------------------------------------------------------------------------------
    DOXYGEN_END_INTERNAL_BLOCK()
  }
}

AGX_TYPE_BINDING(agx::Physics::GeometryParticleContactListPtr, "Physics.GeometryParticleContactListPtr")
AGX_TYPE_BINDING(agx::Physics::GeometryParticleContactListInstance, "Physics.GeometryParticleContactListInstance")

#ifdef _MSC_VER
# pragma warning(pop)
#endif

#undef AGX_ENTITY_WRAPPER
#undef AGX_ENTITY_NAMESPACE
#endif

