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

#ifndef GENERATED_AGX_PHYSICS_INTERACTIONGROUP_H_PLUGIN
#define GENERATED_AGX_PHYSICS_INTERACTIONGROUP_H_PLUGIN

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
#include <agx/Integer.h>
#include <agx/Vec4.h>


namespace agx
{
  namespace Physics
  {

    class InteractionGroupModel;
    class InteractionGroupData;
    class InteractionGroupPtr;
    class InteractionGroupInstance;
    class InteractionGroupSemantics;


    AGX_DECLARE_POINTER_TYPES(InteractionGroupModel);

    /** 
    Abstract description of the data attributes for the Physics.InteractionGroup entity.
    */ 
    class AGXPHYSICS_EXPORT InteractionGroupModel : public agxData::EntityModel
    {
    public:
      typedef InteractionGroupPtr PtrT;

      InteractionGroupModel(const agx::String& name = "InteractionGroup");

      /// \return The entity model singleton.
      static InteractionGroupModel* instance();

      /// Create and return a pointer to a new instance in the default storage for this entity model.
      static InteractionGroupPtr createInstance();

      /// \return The default storage for this entity model.
      static agxData::EntityStorage* defaultStorage();

      /// This is part of internal cleanup and should not be called by users
      virtual void shutdownCleanup() override;



      /* Attributes */
      static agxData::ArrayAttributeT< agx::UInt32 >* nodesAttribute;
      static agxData::ScalarAttributeT< agx::Vec4f >* colorAttribute;

    protected:
      virtual ~InteractionGroupModel();
      virtual agxData::EntityData* createData(agxData::EntityStorage* storage) override;
      virtual void configure(agx::TiXmlElement* eEntity) override;
      virtual void initAttributeAccessors() override;
      virtual void construct(agxData::EntityPtr instance) override;
      void construct(agx::Physics::InteractionGroupPtr interactionGroup);
    };


    DOXYGEN_START_INTERNAL_BLOCK()
    #ifndef AGX_PHYSICS_INTERACTIONGROUP_DATA_SET_OVERRIDE
    #define AGX_PHYSICS_INTERACTIONGROUP_DATA_SET
    class AGXPHYSICS_EXPORT InteractionGroupData : public agxData::EntityData
    {
    public:
      InteractionGroupInstance operator[] (size_t index);

    public:
      agxData::Array< InteractionGroupPtr >& instance;
      agxData::Array< agxData::Array< agx::UInt32 > > nodes;
      agxData::Array< agx::Vec4f > color;

    public:
      typedef agxData::Array< agx::UInt32 > nodesType;
      typedef agx::Vec4f colorType;

    public:
      InteractionGroupData(agxData::EntityStorage* storage);
      InteractionGroupData();

    protected:
      virtual ~InteractionGroupData() {}
      virtual void setNumElements(agx::Index numElements) override;

    private:
      InteractionGroupData& operator= (const InteractionGroupData&) { return *this; }

    };
    #endif
    DOXYGEN_END_INTERNAL_BLOCK()


    DOXYGEN_START_INTERNAL_BLOCK()
    class AGXPHYSICS_EXPORT InteractionGroupSemantics : protected agxData::EntityPtr
    {
    public:

      // Automatic getters
      agxData::Array< agx::UInt32 > const& getNodes() const;
      agx::Vec4f const& getColor() const;

      // Semantics defined by explicit kernels

      // Automatic setters
      void setNodes(agxData::Array< agx::UInt32 > const& value);
      void setColor(agx::Vec4f const& value);


    protected:
      friend class InteractionGroupPtr;
      friend class InteractionGroupInstance;
      InteractionGroupSemantics();
    };
    DOXYGEN_END_INTERNAL_BLOCK()


    /**
    Pointer to a entity instance of type Physics.InteractionGroup
    */
    class CALLABLE InteractionGroupPtr : public agxData::EntityPtr
    {
    public:
      typedef InteractionGroupModel ModelType;
      typedef InteractionGroupData DataType;
      typedef InteractionGroupInstance InstanceType;

    public:
      AGXPHYSICS_EXPORT InteractionGroupPtr();
      AGXPHYSICS_EXPORT InteractionGroupPtr(agxData::EntityStorage* storage, agx::Index id);
      AGXPHYSICS_EXPORT InteractionGroupPtr(const agxData::EntityPtr& ptr);
      AGXPHYSICS_EXPORT InteractionGroupPtr(const agxData::EntityInstance& instance);
      AGXPHYSICS_EXPORT InteractionGroupPtr& operator= (const agxData::EntityPtr& ptr);
      AGXPHYSICS_EXPORT InteractionGroupPtr& operator= (const agxData::EntityInstance& instance);
      AGXPHYSICS_EXPORT InteractionGroupInstance instance();
      AGXPHYSICS_EXPORT const InteractionGroupInstance instance() const;

      AGXPHYSICS_EXPORT InteractionGroupSemantics* operator->();
      AGXPHYSICS_EXPORT const InteractionGroupSemantics* operator->() const;

      InteractionGroupData* getData();
      const InteractionGroupData* getData() const;


      /// \return reference to the nodes attribute
      AGXPHYSICS_EXPORT agxData::Array< agx::UInt32 >& nodes();
      /// \return const reference to the nodes attribute
      AGXPHYSICS_EXPORT agxData::Array< agx::UInt32 > const& nodes() const;

      /// \return reference to the color attribute
      AGXPHYSICS_EXPORT agx::Vec4f& color();
      /// \return const reference to the color attribute
      AGXPHYSICS_EXPORT agx::Vec4f const& color() const;

      DOXYGEN_START_INTERNAL_BLOCK()
      // Only used by tolua!
      #ifdef AGX_GENERATED_TOLUA_WRAPPER
      inline agxData::Array< agx::UInt32 > const& getNodes() const { return (*this)->getNodes(); }
      inline agx::Vec4f const& getColor() const { return (*this)->getColor(); }


      inline void setNodes(agxData::Array< agx::UInt32 > const& value) { (*this)->setNodes(value); }
      inline void setColor(agx::Vec4f const& value) { (*this)->setColor(value); }
      #endif
      DOXYGEN_END_INTERNAL_BLOCK()

    };


    DOXYGEN_START_INTERNAL_BLOCK()
    class AGXPHYSICS_EXPORT InteractionGroupInstance : public agxData::EntityInstance
    {
    public:
      InteractionGroupInstance();
      InteractionGroupInstance(InteractionGroupData* data, agx::Index index);
      InteractionGroupInstance(agxData::EntityStorage *storage, agx::Index index);
      InteractionGroupInstance(const agxData::EntityInstance& other);
      InteractionGroupInstance(const agxData::EntityPtr& ptr);

      InteractionGroupData* getData();
      const InteractionGroupData* getData() const;

    public:
      /// \return reference to the nodes attribute
      agxData::Array< agx::UInt32 >& nodes();
      /// \return const reference to the nodes attribute
      agxData::Array< agx::UInt32 > const& nodes() const;

      /// \return reference to the color attribute
      agx::Vec4f& color();
      /// \return const reference to the color attribute
      agx::Vec4f const& color() const;

    };
    DOXYGEN_END_INTERNAL_BLOCK()



    typedef agx::VectorPOD<InteractionGroupPtr> InteractionGroupPtrVector;
    typedef agxData::Array<InteractionGroupPtr> InteractionGroupPtrArray;



    DOXYGEN_START_INTERNAL_BLOCK()
    /* Implementation */
    //-----------------------------------------------------------------------------------------------------
    //-----------------------------------------------------------------------------------------------------
    inline InteractionGroupInstance agx::Physics::InteractionGroupData::operator[] (size_t index) { return InteractionGroupInstance(this, (agx::Index)index); }
    //-----------------------------------------------------------------------------------------------------
    AGX_FORCE_INLINE InteractionGroupPtr::InteractionGroupPtr() {}
    AGX_FORCE_INLINE InteractionGroupPtr::InteractionGroupPtr(agxData::EntityStorage* storage, agx::Index id) : agxData::EntityPtr(storage, id) {}
    AGX_FORCE_INLINE InteractionGroupPtr::InteractionGroupPtr(const agxData::EntityPtr& ptr) : agxData::EntityPtr(ptr)
    {
      agxAssertN(!ptr || ptr.isInstanceOf(InteractionGroupModel::instance()),
        "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
        EntityPtr::getModel()->fullPath().c_str(), InteractionGroupModel::instance()->fullPath().c_str());
    }

    AGX_FORCE_INLINE InteractionGroupPtr::InteractionGroupPtr(const agxData::EntityInstance& instance) : agxData::EntityPtr(instance)
    {
      agxAssertN(!instance || instance.isInstanceOf(InteractionGroupModel::instance()),
        "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
        EntityPtr::getModel()->fullPath().c_str(), InteractionGroupModel::instance()->fullPath().c_str());
    }

    AGX_FORCE_INLINE InteractionGroupPtr& InteractionGroupPtr::operator= (const agxData::EntityPtr& ptr)
    {
      agxData::EntityPtr::operator= (ptr);
      agxAssertN(!ptr || ptr.isInstanceOf(InteractionGroupModel::instance()),
        "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
        EntityPtr::getModel()->fullPath().c_str(), InteractionGroupModel::instance()->fullPath().c_str());
      return *this;
    }

    AGX_FORCE_INLINE InteractionGroupPtr& InteractionGroupPtr::operator= (const agxData::EntityInstance& instance)
    {
      agxData::EntityPtr::operator= (instance);
      agxAssertN(!instance || instance.isInstanceOf(InteractionGroupModel::instance()),
        "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
        EntityPtr::getModel()->fullPath().c_str(), InteractionGroupModel::instance()->fullPath().c_str());
      return *this;
    }

    inline InteractionGroupInstance InteractionGroupPtr::instance() { return agxData::EntityPtr::instance(); }
    inline const InteractionGroupInstance InteractionGroupPtr::instance() const { return agxData::EntityPtr::instance(); }
    AGX_FORCE_INLINE InteractionGroupSemantics* InteractionGroupPtr::operator->() { return (InteractionGroupSemantics* )this; }
    AGX_FORCE_INLINE const InteractionGroupSemantics* InteractionGroupPtr::operator->() const { return (const InteractionGroupSemantics* )this; }
    AGX_FORCE_INLINE InteractionGroupData* InteractionGroupPtr::getData() { return static_cast<InteractionGroupData* >(agxData::EntityPtr::getData()); }
    AGX_FORCE_INLINE const InteractionGroupData* InteractionGroupPtr::getData() const { return static_cast<const InteractionGroupData* >(agxData::EntityPtr::getData()); }

    AGX_FORCE_INLINE agxData::Array< agx::UInt32 >& InteractionGroupPtr::nodes() { verifyIndex(); return getData()->nodes[calculateIndex()]; }
    AGX_FORCE_INLINE agxData::Array< agx::UInt32 > const& InteractionGroupPtr::nodes() const { verifyIndex(); return getData()->nodes[calculateIndex()]; }

    AGX_FORCE_INLINE agx::Vec4f& InteractionGroupPtr::color() { verifyIndex(); return getData()->color[calculateIndex()]; }
    AGX_FORCE_INLINE agx::Vec4f const& InteractionGroupPtr::color() const { verifyIndex(); return getData()->color[calculateIndex()]; }

    //-----------------------------------------------------------------------------------------------------
    AGX_FORCE_INLINE InteractionGroupInstance::InteractionGroupInstance() {}
    AGX_FORCE_INLINE InteractionGroupInstance::InteractionGroupInstance(InteractionGroupData* data, agx::Index index) : agxData::EntityInstance(data, index) {}
    AGX_FORCE_INLINE InteractionGroupInstance::InteractionGroupInstance(agxData::EntityStorage* storage, agx::Index index) : agxData::EntityInstance(storage, index) {}
    AGX_FORCE_INLINE InteractionGroupInstance::InteractionGroupInstance(const agxData::EntityInstance& other) : agxData::EntityInstance(other)
    {
      agxAssertN(!other || other.isInstanceOf(InteractionGroupModel::instance()),
        "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
        EntityInstance::getModel()->fullPath().c_str(), InteractionGroupModel::instance()->fullPath().c_str());
    }

    AGX_FORCE_INLINE InteractionGroupInstance::InteractionGroupInstance(const agxData::EntityPtr& ptr) : agxData::EntityInstance(ptr)
    {
      agxAssertN(!ptr || ptr.isInstanceOf(InteractionGroupModel::instance()),
        "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
        EntityInstance::getModel()->fullPath().c_str(), InteractionGroupModel::instance()->fullPath().c_str());
    }


    AGX_FORCE_INLINE InteractionGroupData* InteractionGroupInstance::getData() { return static_cast<InteractionGroupData* >(agxData::EntityInstance::getData()); }
    AGX_FORCE_INLINE const InteractionGroupData* InteractionGroupInstance::getData() const { return static_cast<const InteractionGroupData* >(agxData::EntityInstance::getData()); }

    AGX_FORCE_INLINE agxData::Array< agx::UInt32 >& InteractionGroupInstance::nodes() { verifyIndex(); return getData()->nodes[getIndex()]; }
    AGX_FORCE_INLINE agxData::Array< agx::UInt32 > const& InteractionGroupInstance::nodes() const { verifyIndex(); return getData()->nodes[getIndex()]; }

    AGX_FORCE_INLINE agx::Vec4f& InteractionGroupInstance::color() { verifyIndex(); return getData()->color[getIndex()]; }
    AGX_FORCE_INLINE agx::Vec4f const& InteractionGroupInstance::color() const { verifyIndex(); return getData()->color[getIndex()]; }

    //-----------------------------------------------------------------------------------------------------
    AGX_FORCE_INLINE InteractionGroupSemantics::InteractionGroupSemantics() {}
    //-----------------------------------------------------------------------------------------------------
    DOXYGEN_END_INTERNAL_BLOCK()
  }
}

AGX_TYPE_BINDING(agx::Physics::InteractionGroupPtr, "Physics.InteractionGroupPtr")
AGX_TYPE_BINDING(agx::Physics::InteractionGroupInstance, "Physics.InteractionGroupInstance")

#ifdef _MSC_VER
# pragma warning(pop)
#endif

#undef AGX_ENTITY_WRAPPER
#undef AGX_ENTITY_NAMESPACE
#endif

