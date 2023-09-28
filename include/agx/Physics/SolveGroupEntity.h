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

#ifndef GENERATED_AGX_PHYSICS_SOLVEGROUP_H_PLUGIN
#define GENERATED_AGX_PHYSICS_SOLVEGROUP_H_PLUGIN

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
#include <agx/Physics/InteractionGroupEntity.h>
#include <agx/Physics/SolveIslandEntity.h>
namespace agx { class SolveModel; }

namespace agx { namespace Physics { class SolveIslandPtr; }}

namespace agx
{
  namespace Physics
  {

    class SolveGroupModel;
    class SolveGroupData;
    class SolveGroupPtr;
    class SolveGroupInstance;
    class SolveGroupSemantics;


    AGX_DECLARE_POINTER_TYPES(SolveGroupModel);

    /** 
    Abstract description of the data attributes for the Physics.SolveGroup entity.
    */ 
    class AGXPHYSICS_EXPORT SolveGroupModel : public agx::Physics::InteractionGroupModel
    {
    public:
      typedef SolveGroupPtr PtrT;

      SolveGroupModel(const agx::String& name = "SolveGroup");

      /// \return The entity model singleton.
      static SolveGroupModel* instance();

      /// Create and return a pointer to a new instance in the default storage for this entity model.
      static SolveGroupPtr createInstance();

      /// \return The default storage for this entity model.
      static agxData::EntityStorage* defaultStorage();

      /// This is part of internal cleanup and should not be called by users
      virtual void shutdownCleanup() override;



      /* Attributes */
      static agxData::ScalarAttributeT< agx::Physics::SolveIslandPtr >* islandAttribute;
      static agxData::PointerAttributeT< agx::SolveModel*>* solveModelAttribute;

    protected:
      virtual ~SolveGroupModel();
      virtual agxData::EntityData* createData(agxData::EntityStorage* storage) override;
      virtual void configure(agx::TiXmlElement* eEntity) override;
      virtual void initAttributeAccessors() override;
      virtual void construct(agxData::EntityPtr instance) override;
      void construct(agx::Physics::SolveGroupPtr solveGroup);
    };


    DOXYGEN_START_INTERNAL_BLOCK()
    #ifndef AGX_PHYSICS_SOLVEGROUP_DATA_SET_OVERRIDE
    #define AGX_PHYSICS_SOLVEGROUP_DATA_SET
    class AGXPHYSICS_EXPORT SolveGroupData : public agx::Physics::InteractionGroupData
    {
    public:
      SolveGroupInstance operator[] (size_t index);

    public:
      agxData::Array< SolveGroupPtr >& instance;
      agxData::Array< agx::Physics::SolveIslandPtr > island;
      agxData::Array< agx::SolveModel* > solveModel;

    public:
      typedef agx::Physics::SolveIslandPtr islandType;
      typedef agx::SolveModel* solveModelType;

    public:
      SolveGroupData(agxData::EntityStorage* storage);
      SolveGroupData();

    protected:
      virtual ~SolveGroupData() {}
      virtual void setNumElements(agx::Index numElements) override;

    private:
      SolveGroupData& operator= (const SolveGroupData&) { return *this; }

    };
    #endif
    DOXYGEN_END_INTERNAL_BLOCK()


    DOXYGEN_START_INTERNAL_BLOCK()
    class AGXPHYSICS_EXPORT SolveGroupSemantics : public agx::Physics::InteractionGroupSemantics
    {
    public:

      // Automatic getters
      agx::Physics::SolveIslandPtr const& getIsland() const;
      agx::SolveModel* const& getSolveModel() const;

      // Semantics defined by explicit kernels

      // Automatic setters
      void setIsland(agx::Physics::SolveIslandPtr const& value);
      void setSolveModel(agx::SolveModel* const& value);


    protected:
      friend class SolveGroupPtr;
      friend class SolveGroupInstance;
      SolveGroupSemantics();
    };
    DOXYGEN_END_INTERNAL_BLOCK()


    /**
    Pointer to a entity instance of type Physics.SolveGroup
    */
    class CALLABLE SolveGroupPtr : public agx::Physics::InteractionGroupPtr
    {
    public:
      typedef SolveGroupModel ModelType;
      typedef SolveGroupData DataType;
      typedef SolveGroupInstance InstanceType;

    public:
      AGXPHYSICS_EXPORT SolveGroupPtr();
      AGXPHYSICS_EXPORT SolveGroupPtr(agxData::EntityStorage* storage, agx::Index id);
      AGXPHYSICS_EXPORT SolveGroupPtr(const agxData::EntityPtr& ptr);
      AGXPHYSICS_EXPORT SolveGroupPtr(const agxData::EntityInstance& instance);
      AGXPHYSICS_EXPORT SolveGroupPtr& operator= (const agxData::EntityPtr& ptr);
      AGXPHYSICS_EXPORT SolveGroupPtr& operator= (const agxData::EntityInstance& instance);
      AGXPHYSICS_EXPORT SolveGroupInstance instance();
      AGXPHYSICS_EXPORT const SolveGroupInstance instance() const;

      AGXPHYSICS_EXPORT SolveGroupSemantics* operator->();
      AGXPHYSICS_EXPORT const SolveGroupSemantics* operator->() const;

      SolveGroupData* getData();
      const SolveGroupData* getData() const;


      /// \return reference to the island attribute
      AGXPHYSICS_EXPORT agx::Physics::SolveIslandPtr& island();
      /// \return const reference to the island attribute
      AGXPHYSICS_EXPORT agx::Physics::SolveIslandPtr const& island() const;

      /// \return reference to the solveModel attribute
      AGXPHYSICS_EXPORT agx::SolveModel*& solveModel();
      /// \return const reference to the solveModel attribute
      AGXPHYSICS_EXPORT agx::SolveModel* const& solveModel() const;

      DOXYGEN_START_INTERNAL_BLOCK()
      // Only used by tolua!
      #ifdef AGX_GENERATED_TOLUA_WRAPPER
      inline agx::Physics::SolveIslandPtr const& getIsland() const { return (*this)->getIsland(); }
      inline agx::SolveModel* const& getSolveModel() const { return (*this)->getSolveModel(); }


      inline void setIsland(agx::Physics::SolveIslandPtr const& value) { (*this)->setIsland(value); }
      inline void setSolveModel(agx::SolveModel* const& value) { (*this)->setSolveModel(value); }
      #endif
      DOXYGEN_END_INTERNAL_BLOCK()

    };


    DOXYGEN_START_INTERNAL_BLOCK()
    class AGXPHYSICS_EXPORT SolveGroupInstance : public agx::Physics::InteractionGroupInstance
    {
    public:
      SolveGroupInstance();
      SolveGroupInstance(SolveGroupData* data, agx::Index index);
      SolveGroupInstance(agxData::EntityStorage *storage, agx::Index index);
      SolveGroupInstance(const agxData::EntityInstance& other);
      SolveGroupInstance(const agxData::EntityPtr& ptr);

      SolveGroupData* getData();
      const SolveGroupData* getData() const;

    public:
      /// \return reference to the island attribute
      agx::Physics::SolveIslandPtr& island();
      /// \return const reference to the island attribute
      agx::Physics::SolveIslandPtr const& island() const;

      /// \return reference to the solveModel attribute
      agx::SolveModel*& solveModel();
      /// \return const reference to the solveModel attribute
      agx::SolveModel* const& solveModel() const;

    };
    DOXYGEN_END_INTERNAL_BLOCK()



    typedef agx::VectorPOD<SolveGroupPtr> SolveGroupPtrVector;
    typedef agxData::Array<SolveGroupPtr> SolveGroupPtrArray;



    DOXYGEN_START_INTERNAL_BLOCK()
    /* Implementation */
    //-----------------------------------------------------------------------------------------------------
    //-----------------------------------------------------------------------------------------------------
    inline SolveGroupInstance agx::Physics::SolveGroupData::operator[] (size_t index) { return SolveGroupInstance(this, (agx::Index)index); }
    //-----------------------------------------------------------------------------------------------------
    AGX_FORCE_INLINE SolveGroupPtr::SolveGroupPtr() {}
    AGX_FORCE_INLINE SolveGroupPtr::SolveGroupPtr(agxData::EntityStorage* storage, agx::Index id) : agx::Physics::InteractionGroupPtr(storage, id) {}
    AGX_FORCE_INLINE SolveGroupPtr::SolveGroupPtr(const agxData::EntityPtr& ptr) : agx::Physics::InteractionGroupPtr(ptr)
    {
      agxAssertN(!ptr || ptr.isInstanceOf(SolveGroupModel::instance()),
        "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
        EntityPtr::getModel()->fullPath().c_str(), SolveGroupModel::instance()->fullPath().c_str());
    }

    AGX_FORCE_INLINE SolveGroupPtr::SolveGroupPtr(const agxData::EntityInstance& instance) : agx::Physics::InteractionGroupPtr(instance)
    {
      agxAssertN(!instance || instance.isInstanceOf(SolveGroupModel::instance()),
        "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
        EntityPtr::getModel()->fullPath().c_str(), SolveGroupModel::instance()->fullPath().c_str());
    }

    AGX_FORCE_INLINE SolveGroupPtr& SolveGroupPtr::operator= (const agxData::EntityPtr& ptr)
    {
      agxData::EntityPtr::operator= (ptr);
      agxAssertN(!ptr || ptr.isInstanceOf(SolveGroupModel::instance()),
        "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
        EntityPtr::getModel()->fullPath().c_str(), SolveGroupModel::instance()->fullPath().c_str());
      return *this;
    }

    AGX_FORCE_INLINE SolveGroupPtr& SolveGroupPtr::operator= (const agxData::EntityInstance& instance)
    {
      agxData::EntityPtr::operator= (instance);
      agxAssertN(!instance || instance.isInstanceOf(SolveGroupModel::instance()),
        "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
        EntityPtr::getModel()->fullPath().c_str(), SolveGroupModel::instance()->fullPath().c_str());
      return *this;
    }

    inline SolveGroupInstance SolveGroupPtr::instance() { return agxData::EntityPtr::instance(); }
    inline const SolveGroupInstance SolveGroupPtr::instance() const { return agxData::EntityPtr::instance(); }
    AGX_FORCE_INLINE SolveGroupSemantics* SolveGroupPtr::operator->() { return (SolveGroupSemantics* )this; }
    AGX_FORCE_INLINE const SolveGroupSemantics* SolveGroupPtr::operator->() const { return (const SolveGroupSemantics* )this; }
    AGX_FORCE_INLINE SolveGroupData* SolveGroupPtr::getData() { return static_cast<SolveGroupData* >(agxData::EntityPtr::getData()); }
    AGX_FORCE_INLINE const SolveGroupData* SolveGroupPtr::getData() const { return static_cast<const SolveGroupData* >(agxData::EntityPtr::getData()); }

    AGX_FORCE_INLINE agx::Physics::SolveIslandPtr& SolveGroupPtr::island() { verifyIndex(); return getData()->island[calculateIndex()]; }
    AGX_FORCE_INLINE agx::Physics::SolveIslandPtr const& SolveGroupPtr::island() const { verifyIndex(); return getData()->island[calculateIndex()]; }

    AGX_FORCE_INLINE agx::SolveModel*& SolveGroupPtr::solveModel() { verifyIndex(); return getData()->solveModel[calculateIndex()]; }
    AGX_FORCE_INLINE agx::SolveModel* const& SolveGroupPtr::solveModel() const { verifyIndex(); return getData()->solveModel[calculateIndex()]; }

    //-----------------------------------------------------------------------------------------------------
    AGX_FORCE_INLINE SolveGroupInstance::SolveGroupInstance() {}
    AGX_FORCE_INLINE SolveGroupInstance::SolveGroupInstance(SolveGroupData* data, agx::Index index) : agx::Physics::InteractionGroupInstance(data, index) {}
    AGX_FORCE_INLINE SolveGroupInstance::SolveGroupInstance(agxData::EntityStorage* storage, agx::Index index) : agx::Physics::InteractionGroupInstance(storage, index) {}
    AGX_FORCE_INLINE SolveGroupInstance::SolveGroupInstance(const agxData::EntityInstance& other) : agx::Physics::InteractionGroupInstance(other)
    {
      agxAssertN(!other || other.isInstanceOf(SolveGroupModel::instance()),
        "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
        EntityInstance::getModel()->fullPath().c_str(), SolveGroupModel::instance()->fullPath().c_str());
    }

    AGX_FORCE_INLINE SolveGroupInstance::SolveGroupInstance(const agxData::EntityPtr& ptr) : agx::Physics::InteractionGroupInstance(ptr)
    {
      agxAssertN(!ptr || ptr.isInstanceOf(SolveGroupModel::instance()),
        "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
        EntityInstance::getModel()->fullPath().c_str(), SolveGroupModel::instance()->fullPath().c_str());
    }


    AGX_FORCE_INLINE SolveGroupData* SolveGroupInstance::getData() { return static_cast<SolveGroupData* >(agxData::EntityInstance::getData()); }
    AGX_FORCE_INLINE const SolveGroupData* SolveGroupInstance::getData() const { return static_cast<const SolveGroupData* >(agxData::EntityInstance::getData()); }

    AGX_FORCE_INLINE agx::Physics::SolveIslandPtr& SolveGroupInstance::island() { verifyIndex(); return getData()->island[getIndex()]; }
    AGX_FORCE_INLINE agx::Physics::SolveIslandPtr const& SolveGroupInstance::island() const { verifyIndex(); return getData()->island[getIndex()]; }

    AGX_FORCE_INLINE agx::SolveModel*& SolveGroupInstance::solveModel() { verifyIndex(); return getData()->solveModel[getIndex()]; }
    AGX_FORCE_INLINE agx::SolveModel* const& SolveGroupInstance::solveModel() const { verifyIndex(); return getData()->solveModel[getIndex()]; }

    //-----------------------------------------------------------------------------------------------------
    AGX_FORCE_INLINE SolveGroupSemantics::SolveGroupSemantics() {}
    //-----------------------------------------------------------------------------------------------------
    DOXYGEN_END_INTERNAL_BLOCK()
  }
}

AGX_TYPE_BINDING(agx::Physics::SolveGroupPtr, "Physics.SolveGroupPtr")
AGX_TYPE_BINDING(agx::Physics::SolveGroupInstance, "Physics.SolveGroupInstance")

#ifdef _MSC_VER
# pragma warning(pop)
#endif

#undef AGX_ENTITY_WRAPPER
#undef AGX_ENTITY_NAMESPACE
#endif

