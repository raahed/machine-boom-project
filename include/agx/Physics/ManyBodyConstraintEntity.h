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

#ifndef GENERATED_AGX_PHYSICS_MANYBODYCONSTRAINT_H_PLUGIN
#define GENERATED_AGX_PHYSICS_MANYBODYCONSTRAINT_H_PLUGIN

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

    class ManyBodyConstraintModel;
    class ManyBodyConstraintData;
    class ManyBodyConstraintPtr;
    class ManyBodyConstraintInstance;
    class ManyBodyConstraintSemantics;


    AGX_DECLARE_POINTER_TYPES(ManyBodyConstraintModel);

    /** 
    Abstract description of the data attributes for the Physics.ManyBodyConstraint entity.
    */ 
    class AGXPHYSICS_EXPORT ManyBodyConstraintModel : public agx::Physics::ConstraintBaseModel
    {
    public:
      typedef ManyBodyConstraintPtr PtrT;

      ManyBodyConstraintModel(const agx::String& name = "ManyBodyConstraint");

      /// \return The entity model singleton.
      static ManyBodyConstraintModel* instance();

      /// Create and return a pointer to a new instance in the default storage for this entity model.
      static ManyBodyConstraintPtr createInstance();

      /// \return The default storage for this entity model.
      static agxData::EntityStorage* defaultStorage();

      /// This is part of internal cleanup and should not be called by users
      virtual void shutdownCleanup() override;



      /* Attributes */
      static agxData::ArrayAttributeT< agx::UInt >* bodiesAttribute;

    protected:
      virtual ~ManyBodyConstraintModel();
      virtual agxData::EntityData* createData(agxData::EntityStorage* storage) override;
      virtual void configure(agx::TiXmlElement* eEntity) override;
      virtual void initAttributeAccessors() override;
      virtual void construct(agxData::EntityPtr instance) override;
      void construct(agx::Physics::ManyBodyConstraintPtr manyBodyConstraint);
    };


    DOXYGEN_START_INTERNAL_BLOCK()
    #ifndef AGX_PHYSICS_MANYBODYCONSTRAINT_DATA_SET_OVERRIDE
    #define AGX_PHYSICS_MANYBODYCONSTRAINT_DATA_SET
    class AGXPHYSICS_EXPORT ManyBodyConstraintData : public agx::Physics::ConstraintBaseData
    {
    public:
      ManyBodyConstraintInstance operator[] (size_t index);

    public:
      agxData::Array< ManyBodyConstraintPtr >& instance;
      agxData::Array< agxData::Array< agx::UInt > > bodies;

    public:
      typedef agxData::Array< agx::UInt > bodiesType;

    public:
      ManyBodyConstraintData(agxData::EntityStorage* storage);
      ManyBodyConstraintData();

    protected:
      virtual ~ManyBodyConstraintData() {}
      virtual void setNumElements(agx::Index numElements) override;

    private:
      ManyBodyConstraintData& operator= (const ManyBodyConstraintData&) { return *this; }

    };
    #endif
    DOXYGEN_END_INTERNAL_BLOCK()


    DOXYGEN_START_INTERNAL_BLOCK()
    class AGXPHYSICS_EXPORT ManyBodyConstraintSemantics : public agx::Physics::ConstraintBaseSemantics
    {
    public:

      // Automatic getters
      agxData::Array< agx::UInt > const& getBodies() const;

      // Semantics defined by explicit kernels

      // Automatic setters
      void setBodies(agxData::Array< agx::UInt > const& value);


    protected:
      friend class ManyBodyConstraintPtr;
      friend class ManyBodyConstraintInstance;
      ManyBodyConstraintSemantics();
    };
    DOXYGEN_END_INTERNAL_BLOCK()


    /**
    Pointer to a entity instance of type Physics.ManyBodyConstraint
    */
    class CALLABLE ManyBodyConstraintPtr : public agx::Physics::ConstraintBasePtr
    {
    public:
      typedef ManyBodyConstraintModel ModelType;
      typedef ManyBodyConstraintData DataType;
      typedef ManyBodyConstraintInstance InstanceType;

    public:
      AGXPHYSICS_EXPORT ManyBodyConstraintPtr();
      AGXPHYSICS_EXPORT ManyBodyConstraintPtr(agxData::EntityStorage* storage, agx::Index id);
      AGXPHYSICS_EXPORT ManyBodyConstraintPtr(const agxData::EntityPtr& ptr);
      AGXPHYSICS_EXPORT ManyBodyConstraintPtr(const agxData::EntityInstance& instance);
      AGXPHYSICS_EXPORT ManyBodyConstraintPtr& operator= (const agxData::EntityPtr& ptr);
      AGXPHYSICS_EXPORT ManyBodyConstraintPtr& operator= (const agxData::EntityInstance& instance);
      AGXPHYSICS_EXPORT ManyBodyConstraintInstance instance();
      AGXPHYSICS_EXPORT const ManyBodyConstraintInstance instance() const;

      AGXPHYSICS_EXPORT ManyBodyConstraintSemantics* operator->();
      AGXPHYSICS_EXPORT const ManyBodyConstraintSemantics* operator->() const;

      ManyBodyConstraintData* getData();
      const ManyBodyConstraintData* getData() const;


      /// \return reference to the bodies attribute
      AGXPHYSICS_EXPORT agxData::Array< agx::UInt >& bodies();
      /// \return const reference to the bodies attribute
      AGXPHYSICS_EXPORT agxData::Array< agx::UInt > const& bodies() const;

      DOXYGEN_START_INTERNAL_BLOCK()
      // Only used by tolua!
      #ifdef AGX_GENERATED_TOLUA_WRAPPER
      inline agxData::Array< agx::UInt > const& getBodies() const { return (*this)->getBodies(); }


      inline void setBodies(agxData::Array< agx::UInt > const& value) { (*this)->setBodies(value); }
      #endif
      DOXYGEN_END_INTERNAL_BLOCK()

    };


    DOXYGEN_START_INTERNAL_BLOCK()
    class AGXPHYSICS_EXPORT ManyBodyConstraintInstance : public agx::Physics::ConstraintBaseInstance
    {
    public:
      ManyBodyConstraintInstance();
      ManyBodyConstraintInstance(ManyBodyConstraintData* data, agx::Index index);
      ManyBodyConstraintInstance(agxData::EntityStorage *storage, agx::Index index);
      ManyBodyConstraintInstance(const agxData::EntityInstance& other);
      ManyBodyConstraintInstance(const agxData::EntityPtr& ptr);

      ManyBodyConstraintData* getData();
      const ManyBodyConstraintData* getData() const;

    public:
      /// \return reference to the bodies attribute
      agxData::Array< agx::UInt >& bodies();
      /// \return const reference to the bodies attribute
      agxData::Array< agx::UInt > const& bodies() const;

    };
    DOXYGEN_END_INTERNAL_BLOCK()



    typedef agx::VectorPOD<ManyBodyConstraintPtr> ManyBodyConstraintPtrVector;
    typedef agxData::Array<ManyBodyConstraintPtr> ManyBodyConstraintPtrArray;



    DOXYGEN_START_INTERNAL_BLOCK()
    /* Implementation */
    //-----------------------------------------------------------------------------------------------------
    //-----------------------------------------------------------------------------------------------------
    inline ManyBodyConstraintInstance agx::Physics::ManyBodyConstraintData::operator[] (size_t index) { return ManyBodyConstraintInstance(this, (agx::Index)index); }
    //-----------------------------------------------------------------------------------------------------
    AGX_FORCE_INLINE ManyBodyConstraintPtr::ManyBodyConstraintPtr() {}
    AGX_FORCE_INLINE ManyBodyConstraintPtr::ManyBodyConstraintPtr(agxData::EntityStorage* storage, agx::Index id) : agx::Physics::ConstraintBasePtr(storage, id) {}
    AGX_FORCE_INLINE ManyBodyConstraintPtr::ManyBodyConstraintPtr(const agxData::EntityPtr& ptr) : agx::Physics::ConstraintBasePtr(ptr)
    {
      agxAssertN(!ptr || ptr.isInstanceOf(ManyBodyConstraintModel::instance()),
        "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
        EntityPtr::getModel()->fullPath().c_str(), ManyBodyConstraintModel::instance()->fullPath().c_str());
    }

    AGX_FORCE_INLINE ManyBodyConstraintPtr::ManyBodyConstraintPtr(const agxData::EntityInstance& instance) : agx::Physics::ConstraintBasePtr(instance)
    {
      agxAssertN(!instance || instance.isInstanceOf(ManyBodyConstraintModel::instance()),
        "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
        EntityPtr::getModel()->fullPath().c_str(), ManyBodyConstraintModel::instance()->fullPath().c_str());
    }

    AGX_FORCE_INLINE ManyBodyConstraintPtr& ManyBodyConstraintPtr::operator= (const agxData::EntityPtr& ptr)
    {
      agxData::EntityPtr::operator= (ptr);
      agxAssertN(!ptr || ptr.isInstanceOf(ManyBodyConstraintModel::instance()),
        "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
        EntityPtr::getModel()->fullPath().c_str(), ManyBodyConstraintModel::instance()->fullPath().c_str());
      return *this;
    }

    AGX_FORCE_INLINE ManyBodyConstraintPtr& ManyBodyConstraintPtr::operator= (const agxData::EntityInstance& instance)
    {
      agxData::EntityPtr::operator= (instance);
      agxAssertN(!instance || instance.isInstanceOf(ManyBodyConstraintModel::instance()),
        "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
        EntityPtr::getModel()->fullPath().c_str(), ManyBodyConstraintModel::instance()->fullPath().c_str());
      return *this;
    }

    inline ManyBodyConstraintInstance ManyBodyConstraintPtr::instance() { return agxData::EntityPtr::instance(); }
    inline const ManyBodyConstraintInstance ManyBodyConstraintPtr::instance() const { return agxData::EntityPtr::instance(); }
    AGX_FORCE_INLINE ManyBodyConstraintSemantics* ManyBodyConstraintPtr::operator->() { return (ManyBodyConstraintSemantics* )this; }
    AGX_FORCE_INLINE const ManyBodyConstraintSemantics* ManyBodyConstraintPtr::operator->() const { return (const ManyBodyConstraintSemantics* )this; }
    AGX_FORCE_INLINE ManyBodyConstraintData* ManyBodyConstraintPtr::getData() { return static_cast<ManyBodyConstraintData* >(agxData::EntityPtr::getData()); }
    AGX_FORCE_INLINE const ManyBodyConstraintData* ManyBodyConstraintPtr::getData() const { return static_cast<const ManyBodyConstraintData* >(agxData::EntityPtr::getData()); }

    AGX_FORCE_INLINE agxData::Array< agx::UInt >& ManyBodyConstraintPtr::bodies() { verifyIndex(); return getData()->bodies[calculateIndex()]; }
    AGX_FORCE_INLINE agxData::Array< agx::UInt > const& ManyBodyConstraintPtr::bodies() const { verifyIndex(); return getData()->bodies[calculateIndex()]; }

    //-----------------------------------------------------------------------------------------------------
    AGX_FORCE_INLINE ManyBodyConstraintInstance::ManyBodyConstraintInstance() {}
    AGX_FORCE_INLINE ManyBodyConstraintInstance::ManyBodyConstraintInstance(ManyBodyConstraintData* data, agx::Index index) : agx::Physics::ConstraintBaseInstance(data, index) {}
    AGX_FORCE_INLINE ManyBodyConstraintInstance::ManyBodyConstraintInstance(agxData::EntityStorage* storage, agx::Index index) : agx::Physics::ConstraintBaseInstance(storage, index) {}
    AGX_FORCE_INLINE ManyBodyConstraintInstance::ManyBodyConstraintInstance(const agxData::EntityInstance& other) : agx::Physics::ConstraintBaseInstance(other)
    {
      agxAssertN(!other || other.isInstanceOf(ManyBodyConstraintModel::instance()),
        "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
        EntityInstance::getModel()->fullPath().c_str(), ManyBodyConstraintModel::instance()->fullPath().c_str());
    }

    AGX_FORCE_INLINE ManyBodyConstraintInstance::ManyBodyConstraintInstance(const agxData::EntityPtr& ptr) : agx::Physics::ConstraintBaseInstance(ptr)
    {
      agxAssertN(!ptr || ptr.isInstanceOf(ManyBodyConstraintModel::instance()),
        "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
        EntityInstance::getModel()->fullPath().c_str(), ManyBodyConstraintModel::instance()->fullPath().c_str());
    }


    AGX_FORCE_INLINE ManyBodyConstraintData* ManyBodyConstraintInstance::getData() { return static_cast<ManyBodyConstraintData* >(agxData::EntityInstance::getData()); }
    AGX_FORCE_INLINE const ManyBodyConstraintData* ManyBodyConstraintInstance::getData() const { return static_cast<const ManyBodyConstraintData* >(agxData::EntityInstance::getData()); }

    AGX_FORCE_INLINE agxData::Array< agx::UInt >& ManyBodyConstraintInstance::bodies() { verifyIndex(); return getData()->bodies[getIndex()]; }
    AGX_FORCE_INLINE agxData::Array< agx::UInt > const& ManyBodyConstraintInstance::bodies() const { verifyIndex(); return getData()->bodies[getIndex()]; }

    //-----------------------------------------------------------------------------------------------------
    AGX_FORCE_INLINE ManyBodyConstraintSemantics::ManyBodyConstraintSemantics() {}
    //-----------------------------------------------------------------------------------------------------
    DOXYGEN_END_INTERNAL_BLOCK()
  }
}

AGX_TYPE_BINDING(agx::Physics::ManyBodyConstraintPtr, "Physics.ManyBodyConstraintPtr")
AGX_TYPE_BINDING(agx::Physics::ManyBodyConstraintInstance, "Physics.ManyBodyConstraintInstance")

#ifdef _MSC_VER
# pragma warning(pop)
#endif

#undef AGX_ENTITY_WRAPPER
#undef AGX_ENTITY_NAMESPACE
#endif

