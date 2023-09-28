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

#ifndef GENERATED_AGX_PHYSICS_WARMSTARTINGDATA_H_PLUGIN
#define GENERATED_AGX_PHYSICS_WARMSTARTINGDATA_H_PLUGIN

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
#include <agx/Vec4.h>


namespace agx
{
  namespace Physics
  {

    class WarmStartingDataModel;
    class WarmStartingDataData;
    class WarmStartingDataPtr;
    class WarmStartingDataInstance;
    class WarmStartingDataSemantics;


    AGX_DECLARE_POINTER_TYPES(WarmStartingDataModel);

    /** 
    Abstract description of the data attributes for the Physics.WarmStartingData entity.
    */ 
    class AGXPHYSICS_EXPORT WarmStartingDataModel : public agxData::EntityModel
    {
    public:
      typedef WarmStartingDataPtr PtrT;

      WarmStartingDataModel(const agx::String& name = "WarmStartingData");

      /// \return The entity model singleton.
      static WarmStartingDataModel* instance();

      /// Create and return a pointer to a new instance in the default storage for this entity model.
      static WarmStartingDataPtr createInstance();

      /// \return The default storage for this entity model.
      static agxData::EntityStorage* defaultStorage();

      /// This is part of internal cleanup and should not be called by users
      virtual void shutdownCleanup() override;



      /* Attributes */
      static agxData::ScalarAttributeT< agx::Vec3 >* localForceAttribute;
      static agxData::ScalarAttributeT< agx::Vec4i8 >* indexSetAttribute;
      static agxData::ScalarAttributeT< agx::Vec3 >* shape1posAttribute;
      static agxData::ScalarAttributeT< agx::Vec3 >* shape2posAttribute;
      static agxData::ScalarAttributeT< agx::Vec3 >* shape1normalAttribute;

    protected:
      virtual ~WarmStartingDataModel();
      virtual agxData::EntityData* createData(agxData::EntityStorage* storage) override;
      virtual void configure(agx::TiXmlElement* eEntity) override;
      virtual void initAttributeAccessors() override;
      virtual void construct(agxData::EntityPtr instance) override;
      void construct(agx::Physics::WarmStartingDataPtr warmStartingData);
    };


    DOXYGEN_START_INTERNAL_BLOCK()
    #ifndef AGX_PHYSICS_WARMSTARTINGDATA_DATA_SET_OVERRIDE
    #define AGX_PHYSICS_WARMSTARTINGDATA_DATA_SET
    class AGXPHYSICS_EXPORT WarmStartingDataData : public agxData::EntityData
    {
    public:
      WarmStartingDataInstance operator[] (size_t index);

    public:
      agxData::Array< WarmStartingDataPtr >& instance;
      agxData::Array< agx::Vec3 > localForce;
      agxData::Array< agx::Vec4i8 > indexSet;
      agxData::Array< agx::Vec3 > shape1pos;
      agxData::Array< agx::Vec3 > shape2pos;
      agxData::Array< agx::Vec3 > shape1normal;

    public:
      typedef agx::Vec3 localForceType;
      typedef agx::Vec4i8 indexSetType;
      typedef agx::Vec3 shape1posType;
      typedef agx::Vec3 shape2posType;
      typedef agx::Vec3 shape1normalType;

    public:
      WarmStartingDataData(agxData::EntityStorage* storage);
      WarmStartingDataData();

    protected:
      virtual ~WarmStartingDataData() {}
      virtual void setNumElements(agx::Index numElements) override;

    private:
      WarmStartingDataData& operator= (const WarmStartingDataData&) { return *this; }

    };
    #endif
    DOXYGEN_END_INTERNAL_BLOCK()


    DOXYGEN_START_INTERNAL_BLOCK()
    class AGXPHYSICS_EXPORT WarmStartingDataSemantics : protected agxData::EntityPtr
    {
    public:

      // Automatic getters
      agx::Vec3 const& getLocalForce() const;
      agx::Vec4i8 const& getIndexSet() const;
      agx::Vec3 const& getShape1pos() const;
      agx::Vec3 const& getShape2pos() const;
      agx::Vec3 const& getShape1normal() const;

      // Semantics defined by explicit kernels

      // Automatic setters
      void setLocalForce(agx::Vec3 const& value);
      void setIndexSet(agx::Vec4i8 const& value);
      void setShape1pos(agx::Vec3 const& value);
      void setShape2pos(agx::Vec3 const& value);
      void setShape1normal(agx::Vec3 const& value);


    protected:
      friend class WarmStartingDataPtr;
      friend class WarmStartingDataInstance;
      WarmStartingDataSemantics();
    };
    DOXYGEN_END_INTERNAL_BLOCK()


    /**
    Pointer to a entity instance of type Physics.WarmStartingData
    */
    class CALLABLE WarmStartingDataPtr : public agxData::EntityPtr
    {
    public:
      typedef WarmStartingDataModel ModelType;
      typedef WarmStartingDataData DataType;
      typedef WarmStartingDataInstance InstanceType;

    public:
      AGXPHYSICS_EXPORT WarmStartingDataPtr();
      AGXPHYSICS_EXPORT WarmStartingDataPtr(agxData::EntityStorage* storage, agx::Index id);
      AGXPHYSICS_EXPORT WarmStartingDataPtr(const agxData::EntityPtr& ptr);
      AGXPHYSICS_EXPORT WarmStartingDataPtr(const agxData::EntityInstance& instance);
      AGXPHYSICS_EXPORT WarmStartingDataPtr& operator= (const agxData::EntityPtr& ptr);
      AGXPHYSICS_EXPORT WarmStartingDataPtr& operator= (const agxData::EntityInstance& instance);
      AGXPHYSICS_EXPORT WarmStartingDataInstance instance();
      AGXPHYSICS_EXPORT const WarmStartingDataInstance instance() const;

      AGXPHYSICS_EXPORT WarmStartingDataSemantics* operator->();
      AGXPHYSICS_EXPORT const WarmStartingDataSemantics* operator->() const;

      WarmStartingDataData* getData();
      const WarmStartingDataData* getData() const;


      /// \return reference to the localForce attribute
      AGXPHYSICS_EXPORT agx::Vec3& localForce();
      /// \return const reference to the localForce attribute
      AGXPHYSICS_EXPORT agx::Vec3 const& localForce() const;

      /// \return reference to the indexSet attribute
      AGXPHYSICS_EXPORT agx::Vec4i8& indexSet();
      /// \return const reference to the indexSet attribute
      AGXPHYSICS_EXPORT agx::Vec4i8 const& indexSet() const;

      /// \return reference to the shape1pos attribute
      AGXPHYSICS_EXPORT agx::Vec3& shape1pos();
      /// \return const reference to the shape1pos attribute
      AGXPHYSICS_EXPORT agx::Vec3 const& shape1pos() const;

      /// \return reference to the shape2pos attribute
      AGXPHYSICS_EXPORT agx::Vec3& shape2pos();
      /// \return const reference to the shape2pos attribute
      AGXPHYSICS_EXPORT agx::Vec3 const& shape2pos() const;

      /// \return reference to the shape1normal attribute
      AGXPHYSICS_EXPORT agx::Vec3& shape1normal();
      /// \return const reference to the shape1normal attribute
      AGXPHYSICS_EXPORT agx::Vec3 const& shape1normal() const;

      DOXYGEN_START_INTERNAL_BLOCK()
      // Only used by tolua!
      #ifdef AGX_GENERATED_TOLUA_WRAPPER
      inline agx::Vec3 const& getLocalForce() const { return (*this)->getLocalForce(); }
      inline agx::Vec4i8 const& getIndexSet() const { return (*this)->getIndexSet(); }
      inline agx::Vec3 const& getShape1pos() const { return (*this)->getShape1pos(); }
      inline agx::Vec3 const& getShape2pos() const { return (*this)->getShape2pos(); }
      inline agx::Vec3 const& getShape1normal() const { return (*this)->getShape1normal(); }


      inline void setLocalForce(agx::Vec3 const& value) { (*this)->setLocalForce(value); }
      inline void setIndexSet(agx::Vec4i8 const& value) { (*this)->setIndexSet(value); }
      inline void setShape1pos(agx::Vec3 const& value) { (*this)->setShape1pos(value); }
      inline void setShape2pos(agx::Vec3 const& value) { (*this)->setShape2pos(value); }
      inline void setShape1normal(agx::Vec3 const& value) { (*this)->setShape1normal(value); }
      #endif
      DOXYGEN_END_INTERNAL_BLOCK()

    };


    DOXYGEN_START_INTERNAL_BLOCK()
    class AGXPHYSICS_EXPORT WarmStartingDataInstance : public agxData::EntityInstance
    {
    public:
      WarmStartingDataInstance();
      WarmStartingDataInstance(WarmStartingDataData* data, agx::Index index);
      WarmStartingDataInstance(agxData::EntityStorage *storage, agx::Index index);
      WarmStartingDataInstance(const agxData::EntityInstance& other);
      WarmStartingDataInstance(const agxData::EntityPtr& ptr);

      WarmStartingDataData* getData();
      const WarmStartingDataData* getData() const;

    public:
      /// \return reference to the localForce attribute
      agx::Vec3& localForce();
      /// \return const reference to the localForce attribute
      agx::Vec3 const& localForce() const;

      /// \return reference to the indexSet attribute
      agx::Vec4i8& indexSet();
      /// \return const reference to the indexSet attribute
      agx::Vec4i8 const& indexSet() const;

      /// \return reference to the shape1pos attribute
      agx::Vec3& shape1pos();
      /// \return const reference to the shape1pos attribute
      agx::Vec3 const& shape1pos() const;

      /// \return reference to the shape2pos attribute
      agx::Vec3& shape2pos();
      /// \return const reference to the shape2pos attribute
      agx::Vec3 const& shape2pos() const;

      /// \return reference to the shape1normal attribute
      agx::Vec3& shape1normal();
      /// \return const reference to the shape1normal attribute
      agx::Vec3 const& shape1normal() const;

    };
    DOXYGEN_END_INTERNAL_BLOCK()



    typedef agx::VectorPOD<WarmStartingDataPtr> WarmStartingDataPtrVector;
    typedef agxData::Array<WarmStartingDataPtr> WarmStartingDataPtrArray;



    DOXYGEN_START_INTERNAL_BLOCK()
    /* Implementation */
    //-----------------------------------------------------------------------------------------------------
    //-----------------------------------------------------------------------------------------------------
    inline WarmStartingDataInstance agx::Physics::WarmStartingDataData::operator[] (size_t index) { return WarmStartingDataInstance(this, (agx::Index)index); }
    //-----------------------------------------------------------------------------------------------------
    AGX_FORCE_INLINE WarmStartingDataPtr::WarmStartingDataPtr() {}
    AGX_FORCE_INLINE WarmStartingDataPtr::WarmStartingDataPtr(agxData::EntityStorage* storage, agx::Index id) : agxData::EntityPtr(storage, id) {}
    AGX_FORCE_INLINE WarmStartingDataPtr::WarmStartingDataPtr(const agxData::EntityPtr& ptr) : agxData::EntityPtr(ptr)
    {
      agxAssertN(!ptr || ptr.isInstanceOf(WarmStartingDataModel::instance()),
        "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
        EntityPtr::getModel()->fullPath().c_str(), WarmStartingDataModel::instance()->fullPath().c_str());
    }

    AGX_FORCE_INLINE WarmStartingDataPtr::WarmStartingDataPtr(const agxData::EntityInstance& instance) : agxData::EntityPtr(instance)
    {
      agxAssertN(!instance || instance.isInstanceOf(WarmStartingDataModel::instance()),
        "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
        EntityPtr::getModel()->fullPath().c_str(), WarmStartingDataModel::instance()->fullPath().c_str());
    }

    AGX_FORCE_INLINE WarmStartingDataPtr& WarmStartingDataPtr::operator= (const agxData::EntityPtr& ptr)
    {
      agxData::EntityPtr::operator= (ptr);
      agxAssertN(!ptr || ptr.isInstanceOf(WarmStartingDataModel::instance()),
        "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
        EntityPtr::getModel()->fullPath().c_str(), WarmStartingDataModel::instance()->fullPath().c_str());
      return *this;
    }

    AGX_FORCE_INLINE WarmStartingDataPtr& WarmStartingDataPtr::operator= (const agxData::EntityInstance& instance)
    {
      agxData::EntityPtr::operator= (instance);
      agxAssertN(!instance || instance.isInstanceOf(WarmStartingDataModel::instance()),
        "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
        EntityPtr::getModel()->fullPath().c_str(), WarmStartingDataModel::instance()->fullPath().c_str());
      return *this;
    }

    inline WarmStartingDataInstance WarmStartingDataPtr::instance() { return agxData::EntityPtr::instance(); }
    inline const WarmStartingDataInstance WarmStartingDataPtr::instance() const { return agxData::EntityPtr::instance(); }
    AGX_FORCE_INLINE WarmStartingDataSemantics* WarmStartingDataPtr::operator->() { return (WarmStartingDataSemantics* )this; }
    AGX_FORCE_INLINE const WarmStartingDataSemantics* WarmStartingDataPtr::operator->() const { return (const WarmStartingDataSemantics* )this; }
    AGX_FORCE_INLINE WarmStartingDataData* WarmStartingDataPtr::getData() { return static_cast<WarmStartingDataData* >(agxData::EntityPtr::getData()); }
    AGX_FORCE_INLINE const WarmStartingDataData* WarmStartingDataPtr::getData() const { return static_cast<const WarmStartingDataData* >(agxData::EntityPtr::getData()); }

    AGX_FORCE_INLINE agx::Vec3& WarmStartingDataPtr::localForce() { verifyIndex(); return getData()->localForce[calculateIndex()]; }
    AGX_FORCE_INLINE agx::Vec3 const& WarmStartingDataPtr::localForce() const { verifyIndex(); return getData()->localForce[calculateIndex()]; }

    AGX_FORCE_INLINE agx::Vec4i8& WarmStartingDataPtr::indexSet() { verifyIndex(); return getData()->indexSet[calculateIndex()]; }
    AGX_FORCE_INLINE agx::Vec4i8 const& WarmStartingDataPtr::indexSet() const { verifyIndex(); return getData()->indexSet[calculateIndex()]; }

    AGX_FORCE_INLINE agx::Vec3& WarmStartingDataPtr::shape1pos() { verifyIndex(); return getData()->shape1pos[calculateIndex()]; }
    AGX_FORCE_INLINE agx::Vec3 const& WarmStartingDataPtr::shape1pos() const { verifyIndex(); return getData()->shape1pos[calculateIndex()]; }

    AGX_FORCE_INLINE agx::Vec3& WarmStartingDataPtr::shape2pos() { verifyIndex(); return getData()->shape2pos[calculateIndex()]; }
    AGX_FORCE_INLINE agx::Vec3 const& WarmStartingDataPtr::shape2pos() const { verifyIndex(); return getData()->shape2pos[calculateIndex()]; }

    AGX_FORCE_INLINE agx::Vec3& WarmStartingDataPtr::shape1normal() { verifyIndex(); return getData()->shape1normal[calculateIndex()]; }
    AGX_FORCE_INLINE agx::Vec3 const& WarmStartingDataPtr::shape1normal() const { verifyIndex(); return getData()->shape1normal[calculateIndex()]; }

    //-----------------------------------------------------------------------------------------------------
    AGX_FORCE_INLINE WarmStartingDataInstance::WarmStartingDataInstance() {}
    AGX_FORCE_INLINE WarmStartingDataInstance::WarmStartingDataInstance(WarmStartingDataData* data, agx::Index index) : agxData::EntityInstance(data, index) {}
    AGX_FORCE_INLINE WarmStartingDataInstance::WarmStartingDataInstance(agxData::EntityStorage* storage, agx::Index index) : agxData::EntityInstance(storage, index) {}
    AGX_FORCE_INLINE WarmStartingDataInstance::WarmStartingDataInstance(const agxData::EntityInstance& other) : agxData::EntityInstance(other)
    {
      agxAssertN(!other || other.isInstanceOf(WarmStartingDataModel::instance()),
        "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
        EntityInstance::getModel()->fullPath().c_str(), WarmStartingDataModel::instance()->fullPath().c_str());
    }

    AGX_FORCE_INLINE WarmStartingDataInstance::WarmStartingDataInstance(const agxData::EntityPtr& ptr) : agxData::EntityInstance(ptr)
    {
      agxAssertN(!ptr || ptr.isInstanceOf(WarmStartingDataModel::instance()),
        "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
        EntityInstance::getModel()->fullPath().c_str(), WarmStartingDataModel::instance()->fullPath().c_str());
    }


    AGX_FORCE_INLINE WarmStartingDataData* WarmStartingDataInstance::getData() { return static_cast<WarmStartingDataData* >(agxData::EntityInstance::getData()); }
    AGX_FORCE_INLINE const WarmStartingDataData* WarmStartingDataInstance::getData() const { return static_cast<const WarmStartingDataData* >(agxData::EntityInstance::getData()); }

    AGX_FORCE_INLINE agx::Vec3& WarmStartingDataInstance::localForce() { verifyIndex(); return getData()->localForce[getIndex()]; }
    AGX_FORCE_INLINE agx::Vec3 const& WarmStartingDataInstance::localForce() const { verifyIndex(); return getData()->localForce[getIndex()]; }

    AGX_FORCE_INLINE agx::Vec4i8& WarmStartingDataInstance::indexSet() { verifyIndex(); return getData()->indexSet[getIndex()]; }
    AGX_FORCE_INLINE agx::Vec4i8 const& WarmStartingDataInstance::indexSet() const { verifyIndex(); return getData()->indexSet[getIndex()]; }

    AGX_FORCE_INLINE agx::Vec3& WarmStartingDataInstance::shape1pos() { verifyIndex(); return getData()->shape1pos[getIndex()]; }
    AGX_FORCE_INLINE agx::Vec3 const& WarmStartingDataInstance::shape1pos() const { verifyIndex(); return getData()->shape1pos[getIndex()]; }

    AGX_FORCE_INLINE agx::Vec3& WarmStartingDataInstance::shape2pos() { verifyIndex(); return getData()->shape2pos[getIndex()]; }
    AGX_FORCE_INLINE agx::Vec3 const& WarmStartingDataInstance::shape2pos() const { verifyIndex(); return getData()->shape2pos[getIndex()]; }

    AGX_FORCE_INLINE agx::Vec3& WarmStartingDataInstance::shape1normal() { verifyIndex(); return getData()->shape1normal[getIndex()]; }
    AGX_FORCE_INLINE agx::Vec3 const& WarmStartingDataInstance::shape1normal() const { verifyIndex(); return getData()->shape1normal[getIndex()]; }

    //-----------------------------------------------------------------------------------------------------
    AGX_FORCE_INLINE WarmStartingDataSemantics::WarmStartingDataSemantics() {}
    //-----------------------------------------------------------------------------------------------------
    DOXYGEN_END_INTERNAL_BLOCK()
  }
}

AGX_TYPE_BINDING(agx::Physics::WarmStartingDataPtr, "Physics.WarmStartingDataPtr")
AGX_TYPE_BINDING(agx::Physics::WarmStartingDataInstance, "Physics.WarmStartingDataInstance")

#ifdef _MSC_VER
# pragma warning(pop)
#endif

#undef AGX_ENTITY_WRAPPER
#undef AGX_ENTITY_NAMESPACE
#endif

