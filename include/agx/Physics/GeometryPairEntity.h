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

#ifndef GENERATED_AGX_PHYSICS_GEOMETRYPAIR_H_PLUGIN
#define GENERATED_AGX_PHYSICS_GEOMETRYPAIR_H_PLUGIN

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
#include <agx/Physics/GeometryEntity.h>

namespace agx { namespace Physics { class GeometryPtr; }}
namespace agx { namespace Physics { class GeometryPtr; }}

namespace agx
{
  namespace Physics
  {

    class GeometryPairModel;
    class GeometryPairData;
    class GeometryPairPtr;
    class GeometryPairInstance;
    class GeometryPairSemantics;


    AGX_DECLARE_POINTER_TYPES(GeometryPairModel);

    /** 
    Abstract description of the data attributes for the Physics.GeometryPair entity.
    */ 
    class AGXPHYSICS_EXPORT GeometryPairModel : public agxData::EntityModel
    {
    public:
      typedef GeometryPairPtr PtrT;

      GeometryPairModel(const agx::String& name = "GeometryPair");

      /// \return The entity model singleton.
      static GeometryPairModel* instance();

      /// Create and return a pointer to a new instance in the default storage for this entity model.
      static GeometryPairPtr createInstance();

      /// \return The default storage for this entity model.
      static agxData::EntityStorage* defaultStorage();

      /// This is part of internal cleanup and should not be called by users
      virtual void shutdownCleanup() override;



      /* Attributes */
      static agxData::ScalarAttributeT< agx::Physics::GeometryPtr >* geometry1Attribute;
      static agxData::ScalarAttributeT< agx::Physics::GeometryPtr >* geometry2Attribute;

    protected:
      virtual ~GeometryPairModel();
      virtual agxData::EntityData* createData(agxData::EntityStorage* storage) override;
      virtual void configure(agx::TiXmlElement* eEntity) override;
      virtual void initAttributeAccessors() override;
      virtual void construct(agxData::EntityPtr instance) override;
      void construct(agx::Physics::GeometryPairPtr geometryPair);
    };


    DOXYGEN_START_INTERNAL_BLOCK()
    #ifndef AGX_PHYSICS_GEOMETRYPAIR_DATA_SET_OVERRIDE
    #define AGX_PHYSICS_GEOMETRYPAIR_DATA_SET
    class AGXPHYSICS_EXPORT GeometryPairData : public agxData::EntityData
    {
    public:
      GeometryPairInstance operator[] (size_t index);

    public:
      agxData::Array< GeometryPairPtr >& instance;
      agxData::Array< agx::Physics::GeometryPtr > geometry1;
      agxData::Array< agx::Physics::GeometryPtr > geometry2;

    public:
      typedef agx::Physics::GeometryPtr geometry1Type;
      typedef agx::Physics::GeometryPtr geometry2Type;

    public:
      GeometryPairData(agxData::EntityStorage* storage);
      GeometryPairData();

    protected:
      virtual ~GeometryPairData() {}
      virtual void setNumElements(agx::Index numElements) override;

    private:
      GeometryPairData& operator= (const GeometryPairData&) { return *this; }

    };
    #endif
    DOXYGEN_END_INTERNAL_BLOCK()


    DOXYGEN_START_INTERNAL_BLOCK()
    class AGXPHYSICS_EXPORT GeometryPairSemantics : protected agxData::EntityPtr
    {
    public:

      // Automatic getters
      agx::Physics::GeometryPtr const& getGeometry1() const;
      agx::Physics::GeometryPtr const& getGeometry2() const;

      // Semantics defined by explicit kernels

      // Automatic setters
      void setGeometry1(agx::Physics::GeometryPtr const& value);
      void setGeometry2(agx::Physics::GeometryPtr const& value);


    protected:
      friend class GeometryPairPtr;
      friend class GeometryPairInstance;
      GeometryPairSemantics();
    };
    DOXYGEN_END_INTERNAL_BLOCK()


    /**
    Pointer to a entity instance of type Physics.GeometryPair
    */
    class CALLABLE GeometryPairPtr : public agxData::EntityPtr
    {
    public:
      typedef GeometryPairModel ModelType;
      typedef GeometryPairData DataType;
      typedef GeometryPairInstance InstanceType;

    public:
      AGXPHYSICS_EXPORT GeometryPairPtr();
      AGXPHYSICS_EXPORT GeometryPairPtr(agxData::EntityStorage* storage, agx::Index id);
      AGXPHYSICS_EXPORT GeometryPairPtr(const agxData::EntityPtr& ptr);
      AGXPHYSICS_EXPORT GeometryPairPtr(const agxData::EntityInstance& instance);
      AGXPHYSICS_EXPORT GeometryPairPtr& operator= (const agxData::EntityPtr& ptr);
      AGXPHYSICS_EXPORT GeometryPairPtr& operator= (const agxData::EntityInstance& instance);
      AGXPHYSICS_EXPORT GeometryPairInstance instance();
      AGXPHYSICS_EXPORT const GeometryPairInstance instance() const;

      AGXPHYSICS_EXPORT GeometryPairSemantics* operator->();
      AGXPHYSICS_EXPORT const GeometryPairSemantics* operator->() const;

      GeometryPairData* getData();
      const GeometryPairData* getData() const;


      /// \return reference to the geometry1 attribute
      AGXPHYSICS_EXPORT agx::Physics::GeometryPtr& geometry1();
      /// \return const reference to the geometry1 attribute
      AGXPHYSICS_EXPORT agx::Physics::GeometryPtr const& geometry1() const;

      /// \return reference to the geometry2 attribute
      AGXPHYSICS_EXPORT agx::Physics::GeometryPtr& geometry2();
      /// \return const reference to the geometry2 attribute
      AGXPHYSICS_EXPORT agx::Physics::GeometryPtr const& geometry2() const;

      DOXYGEN_START_INTERNAL_BLOCK()
      // Only used by tolua!
      #ifdef AGX_GENERATED_TOLUA_WRAPPER
      inline agx::Physics::GeometryPtr const& getGeometry1() const { return (*this)->getGeometry1(); }
      inline agx::Physics::GeometryPtr const& getGeometry2() const { return (*this)->getGeometry2(); }


      inline void setGeometry1(agx::Physics::GeometryPtr const& value) { (*this)->setGeometry1(value); }
      inline void setGeometry2(agx::Physics::GeometryPtr const& value) { (*this)->setGeometry2(value); }
      #endif
      DOXYGEN_END_INTERNAL_BLOCK()

    };


    DOXYGEN_START_INTERNAL_BLOCK()
    class AGXPHYSICS_EXPORT GeometryPairInstance : public agxData::EntityInstance
    {
    public:
      GeometryPairInstance();
      GeometryPairInstance(GeometryPairData* data, agx::Index index);
      GeometryPairInstance(agxData::EntityStorage *storage, agx::Index index);
      GeometryPairInstance(const agxData::EntityInstance& other);
      GeometryPairInstance(const agxData::EntityPtr& ptr);

      GeometryPairData* getData();
      const GeometryPairData* getData() const;

    public:
      /// \return reference to the geometry1 attribute
      agx::Physics::GeometryPtr& geometry1();
      /// \return const reference to the geometry1 attribute
      agx::Physics::GeometryPtr const& geometry1() const;

      /// \return reference to the geometry2 attribute
      agx::Physics::GeometryPtr& geometry2();
      /// \return const reference to the geometry2 attribute
      agx::Physics::GeometryPtr const& geometry2() const;

    };
    DOXYGEN_END_INTERNAL_BLOCK()



    typedef agx::VectorPOD<GeometryPairPtr> GeometryPairPtrVector;
    typedef agxData::Array<GeometryPairPtr> GeometryPairPtrArray;



    DOXYGEN_START_INTERNAL_BLOCK()
    /* Implementation */
    //-----------------------------------------------------------------------------------------------------
    //-----------------------------------------------------------------------------------------------------
    inline GeometryPairInstance agx::Physics::GeometryPairData::operator[] (size_t index) { return GeometryPairInstance(this, (agx::Index)index); }
    //-----------------------------------------------------------------------------------------------------
    AGX_FORCE_INLINE GeometryPairPtr::GeometryPairPtr() {}
    AGX_FORCE_INLINE GeometryPairPtr::GeometryPairPtr(agxData::EntityStorage* storage, agx::Index id) : agxData::EntityPtr(storage, id) {}
    AGX_FORCE_INLINE GeometryPairPtr::GeometryPairPtr(const agxData::EntityPtr& ptr) : agxData::EntityPtr(ptr)
    {
      agxAssertN(!ptr || ptr.isInstanceOf(GeometryPairModel::instance()),
        "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
        EntityPtr::getModel()->fullPath().c_str(), GeometryPairModel::instance()->fullPath().c_str());
    }

    AGX_FORCE_INLINE GeometryPairPtr::GeometryPairPtr(const agxData::EntityInstance& instance) : agxData::EntityPtr(instance)
    {
      agxAssertN(!instance || instance.isInstanceOf(GeometryPairModel::instance()),
        "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
        EntityPtr::getModel()->fullPath().c_str(), GeometryPairModel::instance()->fullPath().c_str());
    }

    AGX_FORCE_INLINE GeometryPairPtr& GeometryPairPtr::operator= (const agxData::EntityPtr& ptr)
    {
      agxData::EntityPtr::operator= (ptr);
      agxAssertN(!ptr || ptr.isInstanceOf(GeometryPairModel::instance()),
        "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
        EntityPtr::getModel()->fullPath().c_str(), GeometryPairModel::instance()->fullPath().c_str());
      return *this;
    }

    AGX_FORCE_INLINE GeometryPairPtr& GeometryPairPtr::operator= (const agxData::EntityInstance& instance)
    {
      agxData::EntityPtr::operator= (instance);
      agxAssertN(!instance || instance.isInstanceOf(GeometryPairModel::instance()),
        "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
        EntityPtr::getModel()->fullPath().c_str(), GeometryPairModel::instance()->fullPath().c_str());
      return *this;
    }

    inline GeometryPairInstance GeometryPairPtr::instance() { return agxData::EntityPtr::instance(); }
    inline const GeometryPairInstance GeometryPairPtr::instance() const { return agxData::EntityPtr::instance(); }
    AGX_FORCE_INLINE GeometryPairSemantics* GeometryPairPtr::operator->() { return (GeometryPairSemantics* )this; }
    AGX_FORCE_INLINE const GeometryPairSemantics* GeometryPairPtr::operator->() const { return (const GeometryPairSemantics* )this; }
    AGX_FORCE_INLINE GeometryPairData* GeometryPairPtr::getData() { return static_cast<GeometryPairData* >(agxData::EntityPtr::getData()); }
    AGX_FORCE_INLINE const GeometryPairData* GeometryPairPtr::getData() const { return static_cast<const GeometryPairData* >(agxData::EntityPtr::getData()); }

    AGX_FORCE_INLINE agx::Physics::GeometryPtr& GeometryPairPtr::geometry1() { verifyIndex(); return getData()->geometry1[calculateIndex()]; }
    AGX_FORCE_INLINE agx::Physics::GeometryPtr const& GeometryPairPtr::geometry1() const { verifyIndex(); return getData()->geometry1[calculateIndex()]; }

    AGX_FORCE_INLINE agx::Physics::GeometryPtr& GeometryPairPtr::geometry2() { verifyIndex(); return getData()->geometry2[calculateIndex()]; }
    AGX_FORCE_INLINE agx::Physics::GeometryPtr const& GeometryPairPtr::geometry2() const { verifyIndex(); return getData()->geometry2[calculateIndex()]; }

    //-----------------------------------------------------------------------------------------------------
    AGX_FORCE_INLINE GeometryPairInstance::GeometryPairInstance() {}
    AGX_FORCE_INLINE GeometryPairInstance::GeometryPairInstance(GeometryPairData* data, agx::Index index) : agxData::EntityInstance(data, index) {}
    AGX_FORCE_INLINE GeometryPairInstance::GeometryPairInstance(agxData::EntityStorage* storage, agx::Index index) : agxData::EntityInstance(storage, index) {}
    AGX_FORCE_INLINE GeometryPairInstance::GeometryPairInstance(const agxData::EntityInstance& other) : agxData::EntityInstance(other)
    {
      agxAssertN(!other || other.isInstanceOf(GeometryPairModel::instance()),
        "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
        EntityInstance::getModel()->fullPath().c_str(), GeometryPairModel::instance()->fullPath().c_str());
    }

    AGX_FORCE_INLINE GeometryPairInstance::GeometryPairInstance(const agxData::EntityPtr& ptr) : agxData::EntityInstance(ptr)
    {
      agxAssertN(!ptr || ptr.isInstanceOf(GeometryPairModel::instance()),
        "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
        EntityInstance::getModel()->fullPath().c_str(), GeometryPairModel::instance()->fullPath().c_str());
    }


    AGX_FORCE_INLINE GeometryPairData* GeometryPairInstance::getData() { return static_cast<GeometryPairData* >(agxData::EntityInstance::getData()); }
    AGX_FORCE_INLINE const GeometryPairData* GeometryPairInstance::getData() const { return static_cast<const GeometryPairData* >(agxData::EntityInstance::getData()); }

    AGX_FORCE_INLINE agx::Physics::GeometryPtr& GeometryPairInstance::geometry1() { verifyIndex(); return getData()->geometry1[getIndex()]; }
    AGX_FORCE_INLINE agx::Physics::GeometryPtr const& GeometryPairInstance::geometry1() const { verifyIndex(); return getData()->geometry1[getIndex()]; }

    AGX_FORCE_INLINE agx::Physics::GeometryPtr& GeometryPairInstance::geometry2() { verifyIndex(); return getData()->geometry2[getIndex()]; }
    AGX_FORCE_INLINE agx::Physics::GeometryPtr const& GeometryPairInstance::geometry2() const { verifyIndex(); return getData()->geometry2[getIndex()]; }

    //-----------------------------------------------------------------------------------------------------
    AGX_FORCE_INLINE GeometryPairSemantics::GeometryPairSemantics() {}
    //-----------------------------------------------------------------------------------------------------
    DOXYGEN_END_INTERNAL_BLOCK()
  }
}

AGX_TYPE_BINDING(agx::Physics::GeometryPairPtr, "Physics.GeometryPairPtr")
AGX_TYPE_BINDING(agx::Physics::GeometryPairInstance, "Physics.GeometryPairInstance")

#ifdef _MSC_VER
# pragma warning(pop)
#endif

#undef AGX_ENTITY_WRAPPER
#undef AGX_ENTITY_NAMESPACE
#endif

