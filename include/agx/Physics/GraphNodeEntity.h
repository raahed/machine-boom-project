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

#ifndef GENERATED_AGX_PHYSICS_GRAPHNODE_H_PLUGIN
#define GENERATED_AGX_PHYSICS_GRAPHNODE_H_PLUGIN

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

namespace agxData { class EntityPtr; }

namespace agx
{
  namespace Physics
  {

    class GraphNodeModel;
    class GraphNodeData;
    class GraphNodePtr;
    class GraphNodeInstance;
    class GraphNodeSemantics;


    AGX_DECLARE_POINTER_TYPES(GraphNodeModel);

    /** 
    Abstract description of the data attributes for the Physics.GraphNode entity.
    */ 
    class AGXPHYSICS_EXPORT GraphNodeModel : public agxData::EntityModel
    {
    public:
      typedef GraphNodePtr PtrT;

      GraphNodeModel(const agx::String& name = "GraphNode");

      /// \return The entity model singleton.
      static GraphNodeModel* instance();

      /// Create and return a pointer to a new instance in the default storage for this entity model.
      static GraphNodePtr createInstance();

      /// \return The default storage for this entity model.
      static agxData::EntityStorage* defaultStorage();

      /// This is part of internal cleanup and should not be called by users
      virtual void shutdownCleanup() override;



      /* Attributes */
      static agxData::ScalarAttributeT< agxData::EntityPtr >* sourceAttribute;
      static agxData::ScalarAttributeT< agx::UInt8 >* typeAttribute;
      static agxData::ScalarAttributeT< agx::UInt8 >* stateAttribute;
      static agxData::ScalarAttributeT< agx::UInt32 >* colorAttribute;
      static agxData::ScalarAttributeT< agx::UInt32 >* tagAttribute;
      static agxData::ArrayAttributeT< agx::UInt >* edgesAttribute;
      static agxData::ScalarAttributeT< agx::UInt32 >* islandAttribute;
      static agxData::ScalarAttributeT< agx::UInt32 >* subindexAttribute;

    protected:
      virtual ~GraphNodeModel();
      virtual agxData::EntityData* createData(agxData::EntityStorage* storage) override;
      virtual void configure(agx::TiXmlElement* eEntity) override;
      virtual void initAttributeAccessors() override;
      virtual void construct(agxData::EntityPtr instance) override;
      void construct(agx::Physics::GraphNodePtr graphNode);
    };


    DOXYGEN_START_INTERNAL_BLOCK()
    #ifndef AGX_PHYSICS_GRAPHNODE_DATA_SET_OVERRIDE
    #define AGX_PHYSICS_GRAPHNODE_DATA_SET
    class AGXPHYSICS_EXPORT GraphNodeData : public agxData::EntityData
    {
    public:
      GraphNodeInstance operator[] (size_t index);

    public:
      agxData::Array< GraphNodePtr >& instance;
      agxData::Array< agxData::EntityPtr > source;
      agxData::Array< agx::UInt8 > type;
      agxData::Array< agx::UInt8 > state;
      agxData::Array< agx::UInt32 > color;
      agxData::Array< agx::UInt32 > tag;
      agxData::Array< agxData::Array< agx::UInt > > edges;
      agxData::Array< agx::UInt32 > island;
      agxData::Array< agx::UInt32 > subindex;

    public:
      typedef agxData::EntityPtr sourceType;
      typedef agx::UInt8 typeType;
      typedef agx::UInt8 stateType;
      typedef agx::UInt32 colorType;
      typedef agx::UInt32 tagType;
      typedef agxData::Array< agx::UInt > edgesType;
      typedef agx::UInt32 islandType;
      typedef agx::UInt32 subindexType;

    public:
      GraphNodeData(agxData::EntityStorage* storage);
      GraphNodeData();

    protected:
      virtual ~GraphNodeData() {}
      virtual void setNumElements(agx::Index numElements) override;

    private:
      GraphNodeData& operator= (const GraphNodeData&) { return *this; }

    };
    #endif
    DOXYGEN_END_INTERNAL_BLOCK()


    DOXYGEN_START_INTERNAL_BLOCK()
    class AGXPHYSICS_EXPORT GraphNodeSemantics : protected agxData::EntityPtr
    {
    public:

      // Automatic getters
      agxData::EntityPtr const& getSource() const;
      agx::UInt8 const& getType() const;
      agx::UInt8 const& getState() const;
      agx::UInt32 const& getColor() const;
      agx::UInt32 const& getTag() const;
      agxData::Array< agx::UInt > const& getEdges() const;
      agx::UInt32 const& getIsland() const;
      agx::UInt32 const& getSubindex() const;

      // Semantics defined by explicit kernels

      // Automatic setters
      void setSource(agxData::EntityPtr const& value);
      void setType(agx::UInt8 const& value);
      void setState(agx::UInt8 const& value);
      void setColor(agx::UInt32 const& value);
      void setTag(agx::UInt32 const& value);
      void setEdges(agxData::Array< agx::UInt > const& value);
      void setIsland(agx::UInt32 const& value);
      void setSubindex(agx::UInt32 const& value);


    protected:
      friend class GraphNodePtr;
      friend class GraphNodeInstance;
      GraphNodeSemantics();
    };
    DOXYGEN_END_INTERNAL_BLOCK()


    /**
    Pointer to a entity instance of type Physics.GraphNode
    */
    class CALLABLE GraphNodePtr : public agxData::EntityPtr
    {
    public:
      typedef GraphNodeModel ModelType;
      typedef GraphNodeData DataType;
      typedef GraphNodeInstance InstanceType;

    public:
      AGXPHYSICS_EXPORT GraphNodePtr();
      AGXPHYSICS_EXPORT GraphNodePtr(agxData::EntityStorage* storage, agx::Index id);
      AGXPHYSICS_EXPORT GraphNodePtr(const agxData::EntityPtr& ptr);
      AGXPHYSICS_EXPORT GraphNodePtr(const agxData::EntityInstance& instance);
      AGXPHYSICS_EXPORT GraphNodePtr& operator= (const agxData::EntityPtr& ptr);
      AGXPHYSICS_EXPORT GraphNodePtr& operator= (const agxData::EntityInstance& instance);
      AGXPHYSICS_EXPORT GraphNodeInstance instance();
      AGXPHYSICS_EXPORT const GraphNodeInstance instance() const;

      AGXPHYSICS_EXPORT GraphNodeSemantics* operator->();
      AGXPHYSICS_EXPORT const GraphNodeSemantics* operator->() const;

      GraphNodeData* getData();
      const GraphNodeData* getData() const;


      /// \return reference to the source attribute
      AGXPHYSICS_EXPORT agxData::EntityPtr& source();
      /// \return const reference to the source attribute
      AGXPHYSICS_EXPORT agxData::EntityPtr const& source() const;

      /// \return reference to the type attribute
      AGXPHYSICS_EXPORT agx::UInt8& type();
      /// \return const reference to the type attribute
      AGXPHYSICS_EXPORT agx::UInt8 const& type() const;

      /// \return reference to the state attribute
      AGXPHYSICS_EXPORT agx::UInt8& state();
      /// \return const reference to the state attribute
      AGXPHYSICS_EXPORT agx::UInt8 const& state() const;

      /// \return reference to the color attribute
      AGXPHYSICS_EXPORT agx::UInt32& color();
      /// \return const reference to the color attribute
      AGXPHYSICS_EXPORT agx::UInt32 const& color() const;

      /// \return reference to the tag attribute
      AGXPHYSICS_EXPORT agx::UInt32& tag();
      /// \return const reference to the tag attribute
      AGXPHYSICS_EXPORT agx::UInt32 const& tag() const;

      /// \return reference to the edges attribute
      AGXPHYSICS_EXPORT agxData::Array< agx::UInt >& edges();
      /// \return const reference to the edges attribute
      AGXPHYSICS_EXPORT agxData::Array< agx::UInt > const& edges() const;

      /// \return reference to the island attribute
      AGXPHYSICS_EXPORT agx::UInt32& island();
      /// \return const reference to the island attribute
      AGXPHYSICS_EXPORT agx::UInt32 const& island() const;

      /// \return reference to the subindex attribute
      AGXPHYSICS_EXPORT agx::UInt32& subindex();
      /// \return const reference to the subindex attribute
      AGXPHYSICS_EXPORT agx::UInt32 const& subindex() const;

      DOXYGEN_START_INTERNAL_BLOCK()
      // Only used by tolua!
      #ifdef AGX_GENERATED_TOLUA_WRAPPER
      inline agxData::EntityPtr const& getSource() const { return (*this)->getSource(); }
      inline agx::UInt8 const& getType() const { return (*this)->getType(); }
      inline agx::UInt8 const& getState() const { return (*this)->getState(); }
      inline agx::UInt32 const& getColor() const { return (*this)->getColor(); }
      inline agx::UInt32 const& getTag() const { return (*this)->getTag(); }
      inline agxData::Array< agx::UInt > const& getEdges() const { return (*this)->getEdges(); }
      inline agx::UInt32 const& getIsland() const { return (*this)->getIsland(); }
      inline agx::UInt32 const& getSubindex() const { return (*this)->getSubindex(); }


      inline void setSource(agxData::EntityPtr const& value) { (*this)->setSource(value); }
      inline void setType(agx::UInt8 const& value) { (*this)->setType(value); }
      inline void setState(agx::UInt8 const& value) { (*this)->setState(value); }
      inline void setColor(agx::UInt32 const& value) { (*this)->setColor(value); }
      inline void setTag(agx::UInt32 const& value) { (*this)->setTag(value); }
      inline void setEdges(agxData::Array< agx::UInt > const& value) { (*this)->setEdges(value); }
      inline void setIsland(agx::UInt32 const& value) { (*this)->setIsland(value); }
      inline void setSubindex(agx::UInt32 const& value) { (*this)->setSubindex(value); }
      #endif
      DOXYGEN_END_INTERNAL_BLOCK()

    };


    DOXYGEN_START_INTERNAL_BLOCK()
    class AGXPHYSICS_EXPORT GraphNodeInstance : public agxData::EntityInstance
    {
    public:
      GraphNodeInstance();
      GraphNodeInstance(GraphNodeData* data, agx::Index index);
      GraphNodeInstance(agxData::EntityStorage *storage, agx::Index index);
      GraphNodeInstance(const agxData::EntityInstance& other);
      GraphNodeInstance(const agxData::EntityPtr& ptr);

      GraphNodeData* getData();
      const GraphNodeData* getData() const;

    public:
      /// \return reference to the source attribute
      agxData::EntityPtr& source();
      /// \return const reference to the source attribute
      agxData::EntityPtr const& source() const;

      /// \return reference to the type attribute
      agx::UInt8& type();
      /// \return const reference to the type attribute
      agx::UInt8 const& type() const;

      /// \return reference to the state attribute
      agx::UInt8& state();
      /// \return const reference to the state attribute
      agx::UInt8 const& state() const;

      /// \return reference to the color attribute
      agx::UInt32& color();
      /// \return const reference to the color attribute
      agx::UInt32 const& color() const;

      /// \return reference to the tag attribute
      agx::UInt32& tag();
      /// \return const reference to the tag attribute
      agx::UInt32 const& tag() const;

      /// \return reference to the edges attribute
      agxData::Array< agx::UInt >& edges();
      /// \return const reference to the edges attribute
      agxData::Array< agx::UInt > const& edges() const;

      /// \return reference to the island attribute
      agx::UInt32& island();
      /// \return const reference to the island attribute
      agx::UInt32 const& island() const;

      /// \return reference to the subindex attribute
      agx::UInt32& subindex();
      /// \return const reference to the subindex attribute
      agx::UInt32 const& subindex() const;

    };
    DOXYGEN_END_INTERNAL_BLOCK()



    typedef agx::VectorPOD<GraphNodePtr> GraphNodePtrVector;
    typedef agxData::Array<GraphNodePtr> GraphNodePtrArray;



    DOXYGEN_START_INTERNAL_BLOCK()
    /* Implementation */
    //-----------------------------------------------------------------------------------------------------
    //-----------------------------------------------------------------------------------------------------
    inline GraphNodeInstance agx::Physics::GraphNodeData::operator[] (size_t index) { return GraphNodeInstance(this, (agx::Index)index); }
    //-----------------------------------------------------------------------------------------------------
    AGX_FORCE_INLINE GraphNodePtr::GraphNodePtr() {}
    AGX_FORCE_INLINE GraphNodePtr::GraphNodePtr(agxData::EntityStorage* storage, agx::Index id) : agxData::EntityPtr(storage, id) {}
    AGX_FORCE_INLINE GraphNodePtr::GraphNodePtr(const agxData::EntityPtr& ptr) : agxData::EntityPtr(ptr)
    {
      agxAssertN(!ptr || ptr.isInstanceOf(GraphNodeModel::instance()),
        "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
        EntityPtr::getModel()->fullPath().c_str(), GraphNodeModel::instance()->fullPath().c_str());
    }

    AGX_FORCE_INLINE GraphNodePtr::GraphNodePtr(const agxData::EntityInstance& instance) : agxData::EntityPtr(instance)
    {
      agxAssertN(!instance || instance.isInstanceOf(GraphNodeModel::instance()),
        "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
        EntityPtr::getModel()->fullPath().c_str(), GraphNodeModel::instance()->fullPath().c_str());
    }

    AGX_FORCE_INLINE GraphNodePtr& GraphNodePtr::operator= (const agxData::EntityPtr& ptr)
    {
      agxData::EntityPtr::operator= (ptr);
      agxAssertN(!ptr || ptr.isInstanceOf(GraphNodeModel::instance()),
        "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
        EntityPtr::getModel()->fullPath().c_str(), GraphNodeModel::instance()->fullPath().c_str());
      return *this;
    }

    AGX_FORCE_INLINE GraphNodePtr& GraphNodePtr::operator= (const agxData::EntityInstance& instance)
    {
      agxData::EntityPtr::operator= (instance);
      agxAssertN(!instance || instance.isInstanceOf(GraphNodeModel::instance()),
        "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
        EntityPtr::getModel()->fullPath().c_str(), GraphNodeModel::instance()->fullPath().c_str());
      return *this;
    }

    inline GraphNodeInstance GraphNodePtr::instance() { return agxData::EntityPtr::instance(); }
    inline const GraphNodeInstance GraphNodePtr::instance() const { return agxData::EntityPtr::instance(); }
    AGX_FORCE_INLINE GraphNodeSemantics* GraphNodePtr::operator->() { return (GraphNodeSemantics* )this; }
    AGX_FORCE_INLINE const GraphNodeSemantics* GraphNodePtr::operator->() const { return (const GraphNodeSemantics* )this; }
    AGX_FORCE_INLINE GraphNodeData* GraphNodePtr::getData() { return static_cast<GraphNodeData* >(agxData::EntityPtr::getData()); }
    AGX_FORCE_INLINE const GraphNodeData* GraphNodePtr::getData() const { return static_cast<const GraphNodeData* >(agxData::EntityPtr::getData()); }

    AGX_FORCE_INLINE agxData::EntityPtr& GraphNodePtr::source() { verifyIndex(); return getData()->source[calculateIndex()]; }
    AGX_FORCE_INLINE agxData::EntityPtr const& GraphNodePtr::source() const { verifyIndex(); return getData()->source[calculateIndex()]; }

    AGX_FORCE_INLINE agx::UInt8& GraphNodePtr::type() { verifyIndex(); return getData()->type[calculateIndex()]; }
    AGX_FORCE_INLINE agx::UInt8 const& GraphNodePtr::type() const { verifyIndex(); return getData()->type[calculateIndex()]; }

    AGX_FORCE_INLINE agx::UInt8& GraphNodePtr::state() { verifyIndex(); return getData()->state[calculateIndex()]; }
    AGX_FORCE_INLINE agx::UInt8 const& GraphNodePtr::state() const { verifyIndex(); return getData()->state[calculateIndex()]; }

    AGX_FORCE_INLINE agx::UInt32& GraphNodePtr::color() { verifyIndex(); return getData()->color[calculateIndex()]; }
    AGX_FORCE_INLINE agx::UInt32 const& GraphNodePtr::color() const { verifyIndex(); return getData()->color[calculateIndex()]; }

    AGX_FORCE_INLINE agx::UInt32& GraphNodePtr::tag() { verifyIndex(); return getData()->tag[calculateIndex()]; }
    AGX_FORCE_INLINE agx::UInt32 const& GraphNodePtr::tag() const { verifyIndex(); return getData()->tag[calculateIndex()]; }

    AGX_FORCE_INLINE agxData::Array< agx::UInt >& GraphNodePtr::edges() { verifyIndex(); return getData()->edges[calculateIndex()]; }
    AGX_FORCE_INLINE agxData::Array< agx::UInt > const& GraphNodePtr::edges() const { verifyIndex(); return getData()->edges[calculateIndex()]; }

    AGX_FORCE_INLINE agx::UInt32& GraphNodePtr::island() { verifyIndex(); return getData()->island[calculateIndex()]; }
    AGX_FORCE_INLINE agx::UInt32 const& GraphNodePtr::island() const { verifyIndex(); return getData()->island[calculateIndex()]; }

    AGX_FORCE_INLINE agx::UInt32& GraphNodePtr::subindex() { verifyIndex(); return getData()->subindex[calculateIndex()]; }
    AGX_FORCE_INLINE agx::UInt32 const& GraphNodePtr::subindex() const { verifyIndex(); return getData()->subindex[calculateIndex()]; }

    //-----------------------------------------------------------------------------------------------------
    AGX_FORCE_INLINE GraphNodeInstance::GraphNodeInstance() {}
    AGX_FORCE_INLINE GraphNodeInstance::GraphNodeInstance(GraphNodeData* data, agx::Index index) : agxData::EntityInstance(data, index) {}
    AGX_FORCE_INLINE GraphNodeInstance::GraphNodeInstance(agxData::EntityStorage* storage, agx::Index index) : agxData::EntityInstance(storage, index) {}
    AGX_FORCE_INLINE GraphNodeInstance::GraphNodeInstance(const agxData::EntityInstance& other) : agxData::EntityInstance(other)
    {
      agxAssertN(!other || other.isInstanceOf(GraphNodeModel::instance()),
        "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
        EntityInstance::getModel()->fullPath().c_str(), GraphNodeModel::instance()->fullPath().c_str());
    }

    AGX_FORCE_INLINE GraphNodeInstance::GraphNodeInstance(const agxData::EntityPtr& ptr) : agxData::EntityInstance(ptr)
    {
      agxAssertN(!ptr || ptr.isInstanceOf(GraphNodeModel::instance()),
        "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
        EntityInstance::getModel()->fullPath().c_str(), GraphNodeModel::instance()->fullPath().c_str());
    }


    AGX_FORCE_INLINE GraphNodeData* GraphNodeInstance::getData() { return static_cast<GraphNodeData* >(agxData::EntityInstance::getData()); }
    AGX_FORCE_INLINE const GraphNodeData* GraphNodeInstance::getData() const { return static_cast<const GraphNodeData* >(agxData::EntityInstance::getData()); }

    AGX_FORCE_INLINE agxData::EntityPtr& GraphNodeInstance::source() { verifyIndex(); return getData()->source[getIndex()]; }
    AGX_FORCE_INLINE agxData::EntityPtr const& GraphNodeInstance::source() const { verifyIndex(); return getData()->source[getIndex()]; }

    AGX_FORCE_INLINE agx::UInt8& GraphNodeInstance::type() { verifyIndex(); return getData()->type[getIndex()]; }
    AGX_FORCE_INLINE agx::UInt8 const& GraphNodeInstance::type() const { verifyIndex(); return getData()->type[getIndex()]; }

    AGX_FORCE_INLINE agx::UInt8& GraphNodeInstance::state() { verifyIndex(); return getData()->state[getIndex()]; }
    AGX_FORCE_INLINE agx::UInt8 const& GraphNodeInstance::state() const { verifyIndex(); return getData()->state[getIndex()]; }

    AGX_FORCE_INLINE agx::UInt32& GraphNodeInstance::color() { verifyIndex(); return getData()->color[getIndex()]; }
    AGX_FORCE_INLINE agx::UInt32 const& GraphNodeInstance::color() const { verifyIndex(); return getData()->color[getIndex()]; }

    AGX_FORCE_INLINE agx::UInt32& GraphNodeInstance::tag() { verifyIndex(); return getData()->tag[getIndex()]; }
    AGX_FORCE_INLINE agx::UInt32 const& GraphNodeInstance::tag() const { verifyIndex(); return getData()->tag[getIndex()]; }

    AGX_FORCE_INLINE agxData::Array< agx::UInt >& GraphNodeInstance::edges() { verifyIndex(); return getData()->edges[getIndex()]; }
    AGX_FORCE_INLINE agxData::Array< agx::UInt > const& GraphNodeInstance::edges() const { verifyIndex(); return getData()->edges[getIndex()]; }

    AGX_FORCE_INLINE agx::UInt32& GraphNodeInstance::island() { verifyIndex(); return getData()->island[getIndex()]; }
    AGX_FORCE_INLINE agx::UInt32 const& GraphNodeInstance::island() const { verifyIndex(); return getData()->island[getIndex()]; }

    AGX_FORCE_INLINE agx::UInt32& GraphNodeInstance::subindex() { verifyIndex(); return getData()->subindex[getIndex()]; }
    AGX_FORCE_INLINE agx::UInt32 const& GraphNodeInstance::subindex() const { verifyIndex(); return getData()->subindex[getIndex()]; }

    //-----------------------------------------------------------------------------------------------------
    AGX_FORCE_INLINE GraphNodeSemantics::GraphNodeSemantics() {}
    //-----------------------------------------------------------------------------------------------------
    DOXYGEN_END_INTERNAL_BLOCK()
  }
}

AGX_TYPE_BINDING(agx::Physics::GraphNodePtr, "Physics.GraphNodePtr")
AGX_TYPE_BINDING(agx::Physics::GraphNodeInstance, "Physics.GraphNodeInstance")

#ifdef _MSC_VER
# pragma warning(pop)
#endif

#undef AGX_ENTITY_WRAPPER
#undef AGX_ENTITY_NAMESPACE
#endif

