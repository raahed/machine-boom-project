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

#ifndef GENERATED_AGX_PHYSICS_INTERACTION_H_PLUGIN
#define GENERATED_AGX_PHYSICS_INTERACTION_H_PLUGIN

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
#include <agx/Physics/GraphNodeEntity.h>
namespace agx { class SolveModel; }

namespace agx { namespace Physics { class GraphNodePtr; }}

namespace agx
{
  namespace Physics
  {

    class InteractionModel;
    class InteractionData;
    class InteractionPtr;
    class InteractionInstance;
    class InteractionSemantics;


    AGX_DECLARE_POINTER_TYPES(InteractionModel);

    /** 
    Abstract description of the data attributes for the Physics.Interaction entity.
    */ 
    class AGXPHYSICS_EXPORT InteractionModel : public agxData::EntityModel
    {
    public:
      typedef InteractionPtr PtrT;

      InteractionModel(const agx::String& name = "Interaction");

      /// \return The entity model singleton.
      static InteractionModel* instance();

      /// Create and return a pointer to a new instance in the default storage for this entity model.
      static InteractionPtr createInstance();

      /// \return The default storage for this entity model.
      static agxData::EntityStorage* defaultStorage();

      /// This is part of internal cleanup and should not be called by users
      virtual void shutdownCleanup() override;



      /* Attributes */
      static agxData::PointerAttributeT< agx::SolveModel*>* solveModelAttribute;
      static agxData::ScalarAttributeT< agx::Physics::GraphNodePtr >* graphNodeAttribute;

    protected:
      virtual ~InteractionModel();
      virtual agxData::EntityData* createData(agxData::EntityStorage* storage) override;
      virtual void configure(agx::TiXmlElement* eEntity) override;
      virtual void initAttributeAccessors() override;
      virtual void construct(agxData::EntityPtr instance) override;
      void construct(agx::Physics::InteractionPtr interaction);
    };


    DOXYGEN_START_INTERNAL_BLOCK()
    #ifndef AGX_PHYSICS_INTERACTION_DATA_SET_OVERRIDE
    #define AGX_PHYSICS_INTERACTION_DATA_SET
    class AGXPHYSICS_EXPORT InteractionData : public agxData::EntityData
    {
    public:
      InteractionInstance operator[] (size_t index);

    public:
      agxData::Array< InteractionPtr >& instance;
      agxData::Array< agx::SolveModel* > solveModel;
      agxData::Array< agx::Physics::GraphNodePtr > graphNode;

    public:
      typedef agx::SolveModel* solveModelType;
      typedef agx::Physics::GraphNodePtr graphNodeType;

    public:
      InteractionData(agxData::EntityStorage* storage);
      InteractionData();

    protected:
      virtual ~InteractionData() {}
      virtual void setNumElements(agx::Index numElements) override;

    private:
      InteractionData& operator= (const InteractionData&) { return *this; }

    };
    #endif
    DOXYGEN_END_INTERNAL_BLOCK()


    DOXYGEN_START_INTERNAL_BLOCK()
    class AGXPHYSICS_EXPORT InteractionSemantics : protected agxData::EntityPtr
    {
    public:

      // Automatic getters
      agx::SolveModel* const& getSolveModel() const;
      agx::Physics::GraphNodePtr const& getGraphNode() const;

      // Semantics defined by explicit kernels

      // Automatic setters
      void setSolveModel(agx::SolveModel* const& value);
      void setGraphNode(agx::Physics::GraphNodePtr const& value);


    protected:
      friend class InteractionPtr;
      friend class InteractionInstance;
      InteractionSemantics();
    };
    DOXYGEN_END_INTERNAL_BLOCK()


    /**
    Pointer to a entity instance of type Physics.Interaction
    */
    class CALLABLE InteractionPtr : public agxData::EntityPtr
    {
    public:
      typedef InteractionModel ModelType;
      typedef InteractionData DataType;
      typedef InteractionInstance InstanceType;

    public:
      AGXPHYSICS_EXPORT InteractionPtr();
      AGXPHYSICS_EXPORT InteractionPtr(agxData::EntityStorage* storage, agx::Index id);
      AGXPHYSICS_EXPORT InteractionPtr(const agxData::EntityPtr& ptr);
      AGXPHYSICS_EXPORT InteractionPtr(const agxData::EntityInstance& instance);
      AGXPHYSICS_EXPORT InteractionPtr& operator= (const agxData::EntityPtr& ptr);
      AGXPHYSICS_EXPORT InteractionPtr& operator= (const agxData::EntityInstance& instance);
      AGXPHYSICS_EXPORT InteractionInstance instance();
      AGXPHYSICS_EXPORT const InteractionInstance instance() const;

      AGXPHYSICS_EXPORT InteractionSemantics* operator->();
      AGXPHYSICS_EXPORT const InteractionSemantics* operator->() const;

      InteractionData* getData();
      const InteractionData* getData() const;


      /// \return reference to the solveModel attribute
      AGXPHYSICS_EXPORT agx::SolveModel*& solveModel();
      /// \return const reference to the solveModel attribute
      AGXPHYSICS_EXPORT agx::SolveModel* const& solveModel() const;

      /// \return reference to the graphNode attribute
      AGXPHYSICS_EXPORT agx::Physics::GraphNodePtr& graphNode();
      /// \return const reference to the graphNode attribute
      AGXPHYSICS_EXPORT agx::Physics::GraphNodePtr const& graphNode() const;

      DOXYGEN_START_INTERNAL_BLOCK()
      // Only used by tolua!
      #ifdef AGX_GENERATED_TOLUA_WRAPPER
      inline agx::SolveModel* const& getSolveModel() const { return (*this)->getSolveModel(); }
      inline agx::Physics::GraphNodePtr const& getGraphNode() const { return (*this)->getGraphNode(); }


      inline void setSolveModel(agx::SolveModel* const& value) { (*this)->setSolveModel(value); }
      inline void setGraphNode(agx::Physics::GraphNodePtr const& value) { (*this)->setGraphNode(value); }
      #endif
      DOXYGEN_END_INTERNAL_BLOCK()

    };


    DOXYGEN_START_INTERNAL_BLOCK()
    class AGXPHYSICS_EXPORT InteractionInstance : public agxData::EntityInstance
    {
    public:
      InteractionInstance();
      InteractionInstance(InteractionData* data, agx::Index index);
      InteractionInstance(agxData::EntityStorage *storage, agx::Index index);
      InteractionInstance(const agxData::EntityInstance& other);
      InteractionInstance(const agxData::EntityPtr& ptr);

      InteractionData* getData();
      const InteractionData* getData() const;

    public:
      /// \return reference to the solveModel attribute
      agx::SolveModel*& solveModel();
      /// \return const reference to the solveModel attribute
      agx::SolveModel* const& solveModel() const;

      /// \return reference to the graphNode attribute
      agx::Physics::GraphNodePtr& graphNode();
      /// \return const reference to the graphNode attribute
      agx::Physics::GraphNodePtr const& graphNode() const;

    };
    DOXYGEN_END_INTERNAL_BLOCK()



    typedef agx::VectorPOD<InteractionPtr> InteractionPtrVector;
    typedef agxData::Array<InteractionPtr> InteractionPtrArray;



    DOXYGEN_START_INTERNAL_BLOCK()
    /* Implementation */
    //-----------------------------------------------------------------------------------------------------
    //-----------------------------------------------------------------------------------------------------
    inline InteractionInstance agx::Physics::InteractionData::operator[] (size_t index) { return InteractionInstance(this, (agx::Index)index); }
    //-----------------------------------------------------------------------------------------------------
    AGX_FORCE_INLINE InteractionPtr::InteractionPtr() {}
    AGX_FORCE_INLINE InteractionPtr::InteractionPtr(agxData::EntityStorage* storage, agx::Index id) : agxData::EntityPtr(storage, id) {}
    AGX_FORCE_INLINE InteractionPtr::InteractionPtr(const agxData::EntityPtr& ptr) : agxData::EntityPtr(ptr)
    {
      agxAssertN(!ptr || ptr.isInstanceOf(InteractionModel::instance()),
        "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
        EntityPtr::getModel()->fullPath().c_str(), InteractionModel::instance()->fullPath().c_str());
    }

    AGX_FORCE_INLINE InteractionPtr::InteractionPtr(const agxData::EntityInstance& instance) : agxData::EntityPtr(instance)
    {
      agxAssertN(!instance || instance.isInstanceOf(InteractionModel::instance()),
        "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
        EntityPtr::getModel()->fullPath().c_str(), InteractionModel::instance()->fullPath().c_str());
    }

    AGX_FORCE_INLINE InteractionPtr& InteractionPtr::operator= (const agxData::EntityPtr& ptr)
    {
      agxData::EntityPtr::operator= (ptr);
      agxAssertN(!ptr || ptr.isInstanceOf(InteractionModel::instance()),
        "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
        EntityPtr::getModel()->fullPath().c_str(), InteractionModel::instance()->fullPath().c_str());
      return *this;
    }

    AGX_FORCE_INLINE InteractionPtr& InteractionPtr::operator= (const agxData::EntityInstance& instance)
    {
      agxData::EntityPtr::operator= (instance);
      agxAssertN(!instance || instance.isInstanceOf(InteractionModel::instance()),
        "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
        EntityPtr::getModel()->fullPath().c_str(), InteractionModel::instance()->fullPath().c_str());
      return *this;
    }

    inline InteractionInstance InteractionPtr::instance() { return agxData::EntityPtr::instance(); }
    inline const InteractionInstance InteractionPtr::instance() const { return agxData::EntityPtr::instance(); }
    AGX_FORCE_INLINE InteractionSemantics* InteractionPtr::operator->() { return (InteractionSemantics* )this; }
    AGX_FORCE_INLINE const InteractionSemantics* InteractionPtr::operator->() const { return (const InteractionSemantics* )this; }
    AGX_FORCE_INLINE InteractionData* InteractionPtr::getData() { return static_cast<InteractionData* >(agxData::EntityPtr::getData()); }
    AGX_FORCE_INLINE const InteractionData* InteractionPtr::getData() const { return static_cast<const InteractionData* >(agxData::EntityPtr::getData()); }

    AGX_FORCE_INLINE agx::SolveModel*& InteractionPtr::solveModel() { verifyIndex(); return getData()->solveModel[calculateIndex()]; }
    AGX_FORCE_INLINE agx::SolveModel* const& InteractionPtr::solveModel() const { verifyIndex(); return getData()->solveModel[calculateIndex()]; }

    AGX_FORCE_INLINE agx::Physics::GraphNodePtr& InteractionPtr::graphNode() { verifyIndex(); return getData()->graphNode[calculateIndex()]; }
    AGX_FORCE_INLINE agx::Physics::GraphNodePtr const& InteractionPtr::graphNode() const { verifyIndex(); return getData()->graphNode[calculateIndex()]; }

    //-----------------------------------------------------------------------------------------------------
    AGX_FORCE_INLINE InteractionInstance::InteractionInstance() {}
    AGX_FORCE_INLINE InteractionInstance::InteractionInstance(InteractionData* data, agx::Index index) : agxData::EntityInstance(data, index) {}
    AGX_FORCE_INLINE InteractionInstance::InteractionInstance(agxData::EntityStorage* storage, agx::Index index) : agxData::EntityInstance(storage, index) {}
    AGX_FORCE_INLINE InteractionInstance::InteractionInstance(const agxData::EntityInstance& other) : agxData::EntityInstance(other)
    {
      agxAssertN(!other || other.isInstanceOf(InteractionModel::instance()),
        "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
        EntityInstance::getModel()->fullPath().c_str(), InteractionModel::instance()->fullPath().c_str());
    }

    AGX_FORCE_INLINE InteractionInstance::InteractionInstance(const agxData::EntityPtr& ptr) : agxData::EntityInstance(ptr)
    {
      agxAssertN(!ptr || ptr.isInstanceOf(InteractionModel::instance()),
        "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
        EntityInstance::getModel()->fullPath().c_str(), InteractionModel::instance()->fullPath().c_str());
    }


    AGX_FORCE_INLINE InteractionData* InteractionInstance::getData() { return static_cast<InteractionData* >(agxData::EntityInstance::getData()); }
    AGX_FORCE_INLINE const InteractionData* InteractionInstance::getData() const { return static_cast<const InteractionData* >(agxData::EntityInstance::getData()); }

    AGX_FORCE_INLINE agx::SolveModel*& InteractionInstance::solveModel() { verifyIndex(); return getData()->solveModel[getIndex()]; }
    AGX_FORCE_INLINE agx::SolveModel* const& InteractionInstance::solveModel() const { verifyIndex(); return getData()->solveModel[getIndex()]; }

    AGX_FORCE_INLINE agx::Physics::GraphNodePtr& InteractionInstance::graphNode() { verifyIndex(); return getData()->graphNode[getIndex()]; }
    AGX_FORCE_INLINE agx::Physics::GraphNodePtr const& InteractionInstance::graphNode() const { verifyIndex(); return getData()->graphNode[getIndex()]; }

    //-----------------------------------------------------------------------------------------------------
    AGX_FORCE_INLINE InteractionSemantics::InteractionSemantics() {}
    //-----------------------------------------------------------------------------------------------------
    DOXYGEN_END_INTERNAL_BLOCK()
  }
}

AGX_TYPE_BINDING(agx::Physics::InteractionPtr, "Physics.InteractionPtr")
AGX_TYPE_BINDING(agx::Physics::InteractionInstance, "Physics.InteractionInstance")

#ifdef _MSC_VER
# pragma warning(pop)
#endif

#undef AGX_ENTITY_WRAPPER
#undef AGX_ENTITY_NAMESPACE
#endif

