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

#ifndef GENERATED_AGX_PHYSICS_BINARYCONSTRAINT_H_PLUGIN
#define GENERATED_AGX_PHYSICS_BINARYCONSTRAINT_H_PLUGIN

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
#include <agx/Physics/ConstraintBaseEntity.h>
#include <agx/Integer.h>


namespace agx
{
  namespace Physics
  {

    class BinaryConstraintModel;
    class BinaryConstraintData;
    class BinaryConstraintPtr;
    class BinaryConstraintInstance;
    class BinaryConstraintSemantics;


    AGX_DECLARE_POINTER_TYPES(BinaryConstraintModel);

    /** 
    Abstract description of the data attributes for the Physics.BinaryConstraint entity.
    */ 
    class AGXPHYSICS_EXPORT BinaryConstraintModel : public agx::Physics::ConstraintBaseModel
    {
    public:
      typedef BinaryConstraintPtr PtrT;

      BinaryConstraintModel(const agx::String& name = "BinaryConstraint");

      /// \return The entity model singleton.
      static BinaryConstraintModel* instance();

      /// Create and return a pointer to a new instance in the default storage for this entity model.
      static BinaryConstraintPtr createInstance();

      /// \return The default storage for this entity model.
      static agxData::EntityStorage* defaultStorage();

      /// This is part of internal cleanup and should not be called by users
      virtual void shutdownCleanup() override;



      /* Attributes */
      static agxData::ScalarAttributeT< agx::UInt >* body1Attribute;
      static agxData::ScalarAttributeT< agx::UInt >* body2Attribute;

    protected:
      virtual ~BinaryConstraintModel();
      virtual agxData::EntityData* createData(agxData::EntityStorage* storage) override;
      virtual void configure(agx::TiXmlElement* eEntity) override;
      virtual void initAttributeAccessors() override;
      virtual void construct(agxData::EntityPtr instance) override;
      void construct(agx::Physics::BinaryConstraintPtr binaryConstraint);
    };


    DOXYGEN_START_INTERNAL_BLOCK()
    #ifndef AGX_PHYSICS_BINARYCONSTRAINT_DATA_SET_OVERRIDE
    #define AGX_PHYSICS_BINARYCONSTRAINT_DATA_SET
    class AGXPHYSICS_EXPORT BinaryConstraintData : public agx::Physics::ConstraintBaseData
    {
    public:
      BinaryConstraintInstance operator[] (size_t index);

    public:
      agxData::Array< BinaryConstraintPtr >& instance;
      agxData::Array< agx::UInt > body1;
      agxData::Array< agx::UInt > body2;

    public:
      typedef agx::UInt body1Type;
      typedef agx::UInt body2Type;

    public:
      BinaryConstraintData(agxData::EntityStorage* storage);
      BinaryConstraintData();

    protected:
      virtual ~BinaryConstraintData() {}
      virtual void setNumElements(agx::Index numElements) override;

    private:
      BinaryConstraintData& operator= (const BinaryConstraintData&) { return *this; }

    };
    #endif
    DOXYGEN_END_INTERNAL_BLOCK()


    DOXYGEN_START_INTERNAL_BLOCK()
    class AGXPHYSICS_EXPORT BinaryConstraintSemantics : public agx::Physics::ConstraintBaseSemantics
    {
    public:

      // Automatic getters
      agx::UInt const& getBody1() const;
      agx::UInt const& getBody2() const;

      // Semantics defined by explicit kernels

      // Automatic setters
      void setBody1(agx::UInt const& value);
      void setBody2(agx::UInt const& value);


    protected:
      friend class BinaryConstraintPtr;
      friend class BinaryConstraintInstance;
      BinaryConstraintSemantics();
    };
    DOXYGEN_END_INTERNAL_BLOCK()


    /**
    Pointer to a entity instance of type Physics.BinaryConstraint
    */
    class CALLABLE BinaryConstraintPtr : public agx::Physics::ConstraintBasePtr
    {
    public:
      typedef BinaryConstraintModel ModelType;
      typedef BinaryConstraintData DataType;
      typedef BinaryConstraintInstance InstanceType;

    public:
      AGXPHYSICS_EXPORT BinaryConstraintPtr();
      AGXPHYSICS_EXPORT BinaryConstraintPtr(agxData::EntityStorage* storage, agx::Index id);
      AGXPHYSICS_EXPORT BinaryConstraintPtr(const agxData::EntityPtr& ptr);
      AGXPHYSICS_EXPORT BinaryConstraintPtr(const agxData::EntityInstance& instance);
      AGXPHYSICS_EXPORT BinaryConstraintPtr& operator= (const agxData::EntityPtr& ptr);
      AGXPHYSICS_EXPORT BinaryConstraintPtr& operator= (const agxData::EntityInstance& instance);
      AGXPHYSICS_EXPORT BinaryConstraintInstance instance();
      AGXPHYSICS_EXPORT const BinaryConstraintInstance instance() const;

      AGXPHYSICS_EXPORT BinaryConstraintSemantics* operator->();
      AGXPHYSICS_EXPORT const BinaryConstraintSemantics* operator->() const;

      BinaryConstraintData* getData();
      const BinaryConstraintData* getData() const;


      /// \return reference to the body1 attribute
      AGXPHYSICS_EXPORT agx::UInt& body1();
      /// \return const reference to the body1 attribute
      AGXPHYSICS_EXPORT agx::UInt const& body1() const;

      /// \return reference to the body2 attribute
      AGXPHYSICS_EXPORT agx::UInt& body2();
      /// \return const reference to the body2 attribute
      AGXPHYSICS_EXPORT agx::UInt const& body2() const;

      DOXYGEN_START_INTERNAL_BLOCK()
      // Only used by tolua!
      #ifdef AGX_GENERATED_TOLUA_WRAPPER
      inline agx::UInt const& getBody1() const { return (*this)->getBody1(); }
      inline agx::UInt const& getBody2() const { return (*this)->getBody2(); }


      inline void setBody1(agx::UInt const& value) { (*this)->setBody1(value); }
      inline void setBody2(agx::UInt const& value) { (*this)->setBody2(value); }
      #endif
      DOXYGEN_END_INTERNAL_BLOCK()

    };


    DOXYGEN_START_INTERNAL_BLOCK()
    class AGXPHYSICS_EXPORT BinaryConstraintInstance : public agx::Physics::ConstraintBaseInstance
    {
    public:
      BinaryConstraintInstance();
      BinaryConstraintInstance(BinaryConstraintData* data, agx::Index index);
      BinaryConstraintInstance(agxData::EntityStorage *storage, agx::Index index);
      BinaryConstraintInstance(const agxData::EntityInstance& other);
      BinaryConstraintInstance(const agxData::EntityPtr& ptr);

      BinaryConstraintData* getData();
      const BinaryConstraintData* getData() const;

    public:
      /// \return reference to the body1 attribute
      agx::UInt& body1();
      /// \return const reference to the body1 attribute
      agx::UInt const& body1() const;

      /// \return reference to the body2 attribute
      agx::UInt& body2();
      /// \return const reference to the body2 attribute
      agx::UInt const& body2() const;

    };
    DOXYGEN_END_INTERNAL_BLOCK()



    typedef agx::VectorPOD<BinaryConstraintPtr> BinaryConstraintPtrVector;
    typedef agxData::Array<BinaryConstraintPtr> BinaryConstraintPtrArray;



    DOXYGEN_START_INTERNAL_BLOCK()
    /* Implementation */
    //-----------------------------------------------------------------------------------------------------
    //-----------------------------------------------------------------------------------------------------
    inline BinaryConstraintInstance agx::Physics::BinaryConstraintData::operator[] (size_t index) { return BinaryConstraintInstance(this, (agx::Index)index); }
    //-----------------------------------------------------------------------------------------------------
    AGX_FORCE_INLINE BinaryConstraintPtr::BinaryConstraintPtr() {}
    AGX_FORCE_INLINE BinaryConstraintPtr::BinaryConstraintPtr(agxData::EntityStorage* storage, agx::Index id) : agx::Physics::ConstraintBasePtr(storage, id) {}
    AGX_FORCE_INLINE BinaryConstraintPtr::BinaryConstraintPtr(const agxData::EntityPtr& ptr) : agx::Physics::ConstraintBasePtr(ptr)
    {
      agxAssertN(!ptr || ptr.isInstanceOf(BinaryConstraintModel::instance()),
        "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
        EntityPtr::getModel()->fullPath().c_str(), BinaryConstraintModel::instance()->fullPath().c_str());
    }

    AGX_FORCE_INLINE BinaryConstraintPtr::BinaryConstraintPtr(const agxData::EntityInstance& instance) : agx::Physics::ConstraintBasePtr(instance)
    {
      agxAssertN(!instance || instance.isInstanceOf(BinaryConstraintModel::instance()),
        "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
        EntityPtr::getModel()->fullPath().c_str(), BinaryConstraintModel::instance()->fullPath().c_str());
    }

    AGX_FORCE_INLINE BinaryConstraintPtr& BinaryConstraintPtr::operator= (const agxData::EntityPtr& ptr)
    {
      agxData::EntityPtr::operator= (ptr);
      agxAssertN(!ptr || ptr.isInstanceOf(BinaryConstraintModel::instance()),
        "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
        EntityPtr::getModel()->fullPath().c_str(), BinaryConstraintModel::instance()->fullPath().c_str());
      return *this;
    }

    AGX_FORCE_INLINE BinaryConstraintPtr& BinaryConstraintPtr::operator= (const agxData::EntityInstance& instance)
    {
      agxData::EntityPtr::operator= (instance);
      agxAssertN(!instance || instance.isInstanceOf(BinaryConstraintModel::instance()),
        "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
        EntityPtr::getModel()->fullPath().c_str(), BinaryConstraintModel::instance()->fullPath().c_str());
      return *this;
    }

    inline BinaryConstraintInstance BinaryConstraintPtr::instance() { return agxData::EntityPtr::instance(); }
    inline const BinaryConstraintInstance BinaryConstraintPtr::instance() const { return agxData::EntityPtr::instance(); }
    AGX_FORCE_INLINE BinaryConstraintSemantics* BinaryConstraintPtr::operator->() { return (BinaryConstraintSemantics* )this; }
    AGX_FORCE_INLINE const BinaryConstraintSemantics* BinaryConstraintPtr::operator->() const { return (const BinaryConstraintSemantics* )this; }
    AGX_FORCE_INLINE BinaryConstraintData* BinaryConstraintPtr::getData() { return static_cast<BinaryConstraintData* >(agxData::EntityPtr::getData()); }
    AGX_FORCE_INLINE const BinaryConstraintData* BinaryConstraintPtr::getData() const { return static_cast<const BinaryConstraintData* >(agxData::EntityPtr::getData()); }

    AGX_FORCE_INLINE agx::UInt& BinaryConstraintPtr::body1() { verifyIndex(); return getData()->body1[calculateIndex()]; }
    AGX_FORCE_INLINE agx::UInt const& BinaryConstraintPtr::body1() const { verifyIndex(); return getData()->body1[calculateIndex()]; }

    AGX_FORCE_INLINE agx::UInt& BinaryConstraintPtr::body2() { verifyIndex(); return getData()->body2[calculateIndex()]; }
    AGX_FORCE_INLINE agx::UInt const& BinaryConstraintPtr::body2() const { verifyIndex(); return getData()->body2[calculateIndex()]; }

    //-----------------------------------------------------------------------------------------------------
    AGX_FORCE_INLINE BinaryConstraintInstance::BinaryConstraintInstance() {}
    AGX_FORCE_INLINE BinaryConstraintInstance::BinaryConstraintInstance(BinaryConstraintData* data, agx::Index index) : agx::Physics::ConstraintBaseInstance(data, index) {}
    AGX_FORCE_INLINE BinaryConstraintInstance::BinaryConstraintInstance(agxData::EntityStorage* storage, agx::Index index) : agx::Physics::ConstraintBaseInstance(storage, index) {}
    AGX_FORCE_INLINE BinaryConstraintInstance::BinaryConstraintInstance(const agxData::EntityInstance& other) : agx::Physics::ConstraintBaseInstance(other)
    {
      agxAssertN(!other || other.isInstanceOf(BinaryConstraintModel::instance()),
        "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
        EntityInstance::getModel()->fullPath().c_str(), BinaryConstraintModel::instance()->fullPath().c_str());
    }

    AGX_FORCE_INLINE BinaryConstraintInstance::BinaryConstraintInstance(const agxData::EntityPtr& ptr) : agx::Physics::ConstraintBaseInstance(ptr)
    {
      agxAssertN(!ptr || ptr.isInstanceOf(BinaryConstraintModel::instance()),
        "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
        EntityInstance::getModel()->fullPath().c_str(), BinaryConstraintModel::instance()->fullPath().c_str());
    }


    AGX_FORCE_INLINE BinaryConstraintData* BinaryConstraintInstance::getData() { return static_cast<BinaryConstraintData* >(agxData::EntityInstance::getData()); }
    AGX_FORCE_INLINE const BinaryConstraintData* BinaryConstraintInstance::getData() const { return static_cast<const BinaryConstraintData* >(agxData::EntityInstance::getData()); }

    AGX_FORCE_INLINE agx::UInt& BinaryConstraintInstance::body1() { verifyIndex(); return getData()->body1[getIndex()]; }
    AGX_FORCE_INLINE agx::UInt const& BinaryConstraintInstance::body1() const { verifyIndex(); return getData()->body1[getIndex()]; }

    AGX_FORCE_INLINE agx::UInt& BinaryConstraintInstance::body2() { verifyIndex(); return getData()->body2[getIndex()]; }
    AGX_FORCE_INLINE agx::UInt const& BinaryConstraintInstance::body2() const { verifyIndex(); return getData()->body2[getIndex()]; }

    //-----------------------------------------------------------------------------------------------------
    AGX_FORCE_INLINE BinaryConstraintSemantics::BinaryConstraintSemantics() {}
    //-----------------------------------------------------------------------------------------------------
    DOXYGEN_END_INTERNAL_BLOCK()
  }
}

AGX_TYPE_BINDING(agx::Physics::BinaryConstraintPtr, "Physics.BinaryConstraintPtr")
AGX_TYPE_BINDING(agx::Physics::BinaryConstraintInstance, "Physics.BinaryConstraintInstance")

#ifdef _MSC_VER
# pragma warning(pop)
#endif

#undef AGX_ENTITY_WRAPPER
#undef AGX_ENTITY_NAMESPACE
#endif

