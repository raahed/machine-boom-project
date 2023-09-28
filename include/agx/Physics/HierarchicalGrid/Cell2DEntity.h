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

#ifndef GENERATED_AGX_PHYSICS_HIERARCHICALGRID_CELL2D_H_PLUGIN
#define GENERATED_AGX_PHYSICS_HIERARCHICALGRID_CELL2D_H_PLUGIN

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
#include <agx/Vec2.h>
#include <agx/Real.h>
#include <agx/IndexRange.h>


namespace agx
{
  namespace Physics
  {
    namespace HierarchicalGrid
    {

      class Cell2DModel;
      class Cell2DData;
      class Cell2DPtr;
      class Cell2DInstance;
      class Cell2DSemantics;


      AGX_DECLARE_POINTER_TYPES(Cell2DModel);

      /** 
      Abstract description of the data attributes for the Physics.HierarchicalGrid.Cell2D entity.
      */ 
      class AGXPHYSICS_EXPORT Cell2DModel : public agxData::EntityModel
      {
      public:
        typedef Cell2DPtr PtrT;

        Cell2DModel(const agx::String& name = "Cell2D");

        /// \return The entity model singleton.
        static Cell2DModel* instance();

        /// Create and return a pointer to a new instance in the default storage for this entity model.
        static Cell2DPtr createInstance();

        /// \return The default storage for this entity model.
        static agxData::EntityStorage* defaultStorage();

        /// This is part of internal cleanup and should not be called by users
        virtual void shutdownCleanup() override;



        /* Attributes */
        static agxData::ScalarAttributeT< agx::UInt8 >* tierAttribute;
        static agxData::ScalarAttributeT< agx::Vec2i >* idAttribute;
        static agxData::ScalarAttributeT< agx::Vec2f >* positionAttribute;
        static agxData::ScalarAttributeT< agx::UInt64 >* linearIdAttribute;
        static agxData::ScalarAttributeT< agx::UInt8 >* stateAttribute;
        static agxData::ScalarAttributeT< agx::Real32 >* emptyTimeAttribute;
        static agxData::ScalarAttributeT< agx::IndexRange32 >* collisionObjectsAttribute;
        static agxData::ScalarAttributeT< agx::UInt8 >* zoneTypeAttribute;
        static agxData::ScalarAttributeT< agx::UInt32 >* parentAttribute;
        static agxData::ScalarAttributeT< agx::UInt32 >* childrenAttribute;
        static const size_t childrenArraySize = 4;
        static agxData::ScalarAttributeT< agx::UInt8 >* numChildrenAttribute;
        static agxData::ScalarAttributeT< agx::UInt32 >* neighborsAttribute;
        static const size_t neighborsArraySize = 12;
        static agxData::ScalarAttributeT< agx::UInt32 >* searchCostAttribute;
        static agxData::SharedAttributeT< agx::Real32 >* emptyThresholdAttribute;
        static agxData::SharedAttributeT< agx::Real >* sizeAlignmentAttribute;

      protected:
        virtual ~Cell2DModel();
        virtual agxData::EntityData* createData(agxData::EntityStorage* storage) override;
        virtual void configure(agx::TiXmlElement* eEntity) override;
        virtual void initAttributeAccessors() override;
        virtual void construct(agxData::EntityPtr instance) override;
        void construct(agx::Physics::HierarchicalGrid::Cell2DPtr cell2D);
      };


      DOXYGEN_START_INTERNAL_BLOCK()
      #ifndef AGX_PHYSICS_HIERARCHICALGRID_CELL2D_DATA_SET_OVERRIDE
      #define AGX_PHYSICS_HIERARCHICALGRID_CELL2D_DATA_SET
      class AGXPHYSICS_EXPORT Cell2DData : public agxData::EntityData
      {
      public:
        Cell2DInstance operator[] (size_t index);

      public:
        agxData::Array< Cell2DPtr >& instance;
        agxData::Array< agx::UInt8 > tier;
        agxData::Array< agx::Vec2i > id;
        agxData::Array< agx::Vec2f > position;
        agxData::Array< agx::UInt64 > linearId;
        agxData::Array< agx::UInt8 > state;
        agxData::Array< agx::Real32 > emptyTime;
        agxData::Array< agx::IndexRange32 > collisionObjects;
        agxData::Array< agx::UInt8 > zoneType;
        agxData::Array< agx::UInt32 > parent;
        agxData::Array< agx::UInt32 > children;
        static const size_t childrenArraySize = 4;
        agxData::Array< agx::UInt8 > numChildren;
        agxData::Array< agx::UInt32 > neighbors;
        static const size_t neighborsArraySize = 12;
        agxData::Array< agx::UInt32 > searchCost;
        agx::Real32 emptyThreshold;
        agx::Real sizeAlignment;

      public:
        typedef agx::UInt8 tierType;
        typedef agx::Vec2i idType;
        typedef agx::Vec2f positionType;
        typedef agx::UInt64 linearIdType;
        typedef agx::UInt8 stateType;
        typedef agx::Real32 emptyTimeType;
        typedef agx::IndexRange32 collisionObjectsType;
        typedef agx::UInt8 zoneTypeType;
        typedef agx::UInt32 parentType;
        typedef agx::UInt32 childrenType;
        typedef agx::UInt8 numChildrenType;
        typedef agx::UInt32 neighborsType;
        typedef agx::UInt32 searchCostType;
        typedef agx::Real32 emptyThresholdType;
        typedef agx::Real sizeAlignmentType;

      public:
        Cell2DData(agxData::EntityStorage* storage);
        Cell2DData();

      protected:
        virtual ~Cell2DData() {}
        virtual void setNumElements(agx::Index numElements) override;
        virtual void synchronizeSharedAttribute(agxData::Value* value) override;

      private:
        Cell2DData& operator= (const Cell2DData&) { return *this; }

      };
      #endif
      DOXYGEN_END_INTERNAL_BLOCK()


      DOXYGEN_START_INTERNAL_BLOCK()
      class AGXPHYSICS_EXPORT Cell2DSemantics : protected agxData::EntityPtr
      {
      public:

        // Automatic getters
        agx::UInt8 const& getTier() const;
        agx::Vec2i const& getId() const;
        agx::Vec2f const& getPosition() const;
        agx::UInt64 const& getLinearId() const;
        agx::UInt8 const& getState() const;
        agx::Real32 const& getEmptyTime() const;
        agx::IndexRange32 const& getCollisionObjects() const;
        agx::UInt8 const& getZoneType() const;
        agx::UInt32 const& getParent() const;
        agx::UInt8 const& getNumChildren() const;
        agx::UInt32 const& getSearchCost() const;

        // Semantics defined by explicit kernels

        // Automatic setters
        void setTier(agx::UInt8 const& value);
        void setId(agx::Vec2i const& value);
        void setPosition(agx::Vec2f const& value);
        void setLinearId(agx::UInt64 const& value);
        void setState(agx::UInt8 const& value);
        void setEmptyTime(agx::Real32 const& value);
        void setCollisionObjects(agx::IndexRange32 const& value);
        void setZoneType(agx::UInt8 const& value);
        void setParent(agx::UInt32 const& value);
        void setNumChildren(agx::UInt8 const& value);
        void setSearchCost(agx::UInt32 const& value);


      protected:
        friend class Cell2DPtr;
        friend class Cell2DInstance;
        Cell2DSemantics();
      };
      DOXYGEN_END_INTERNAL_BLOCK()


      /**
      Pointer to a entity instance of type Physics.HierarchicalGrid.Cell2D
      */
      class CALLABLE Cell2DPtr : public agxData::EntityPtr
      {
      public:
        typedef Cell2DModel ModelType;
        typedef Cell2DData DataType;
        typedef Cell2DInstance InstanceType;

      public:
        AGXPHYSICS_EXPORT Cell2DPtr();
        AGXPHYSICS_EXPORT Cell2DPtr(agxData::EntityStorage* storage, agx::Index id);
        AGXPHYSICS_EXPORT Cell2DPtr(const agxData::EntityPtr& ptr);
        AGXPHYSICS_EXPORT Cell2DPtr(const agxData::EntityInstance& instance);
        AGXPHYSICS_EXPORT Cell2DPtr& operator= (const agxData::EntityPtr& ptr);
        AGXPHYSICS_EXPORT Cell2DPtr& operator= (const agxData::EntityInstance& instance);
        AGXPHYSICS_EXPORT Cell2DInstance instance();
        AGXPHYSICS_EXPORT const Cell2DInstance instance() const;

        AGXPHYSICS_EXPORT Cell2DSemantics* operator->();
        AGXPHYSICS_EXPORT const Cell2DSemantics* operator->() const;

        Cell2DData* getData();
        const Cell2DData* getData() const;


        /// \return reference to the tier attribute
        AGXPHYSICS_EXPORT agx::UInt8& tier();
        /// \return const reference to the tier attribute
        AGXPHYSICS_EXPORT agx::UInt8 const& tier() const;

        /// \return reference to the id attribute
        AGXPHYSICS_EXPORT agx::Vec2i& id();
        /// \return const reference to the id attribute
        AGXPHYSICS_EXPORT agx::Vec2i const& id() const;

        /// \return reference to the position attribute
        AGXPHYSICS_EXPORT agx::Vec2f& position();
        /// \return const reference to the position attribute
        AGXPHYSICS_EXPORT agx::Vec2f const& position() const;

        /// \return reference to the linearId attribute
        AGXPHYSICS_EXPORT agx::UInt64& linearId();
        /// \return const reference to the linearId attribute
        AGXPHYSICS_EXPORT agx::UInt64 const& linearId() const;

        /// \return reference to the state attribute
        AGXPHYSICS_EXPORT agx::UInt8& state();
        /// \return const reference to the state attribute
        AGXPHYSICS_EXPORT agx::UInt8 const& state() const;

        /// \return reference to the emptyTime attribute
        AGXPHYSICS_EXPORT agx::Real32& emptyTime();
        /// \return const reference to the emptyTime attribute
        AGXPHYSICS_EXPORT agx::Real32 const& emptyTime() const;

        /// \return reference to the collisionObjects attribute
        AGXPHYSICS_EXPORT agx::IndexRange32& collisionObjects();
        /// \return const reference to the collisionObjects attribute
        AGXPHYSICS_EXPORT agx::IndexRange32 const& collisionObjects() const;

        /// \return reference to the zoneType attribute
        AGXPHYSICS_EXPORT agx::UInt8& zoneType();
        /// \return const reference to the zoneType attribute
        AGXPHYSICS_EXPORT agx::UInt8 const& zoneType() const;

        /// \return reference to the parent attribute
        AGXPHYSICS_EXPORT agx::UInt32& parent();
        /// \return const reference to the parent attribute
        AGXPHYSICS_EXPORT agx::UInt32 const& parent() const;

        AGXPHYSICS_EXPORT agxData::Array< agx::UInt32 > children();
        AGXPHYSICS_EXPORT agxData::Array< agx::UInt32 > const children() const;
        AGXPHYSICS_EXPORT agx::UInt32& children(size_t index);
        AGXPHYSICS_EXPORT agx::UInt32 const& children(size_t index) const;

        /// \return reference to the numChildren attribute
        AGXPHYSICS_EXPORT agx::UInt8& numChildren();
        /// \return const reference to the numChildren attribute
        AGXPHYSICS_EXPORT agx::UInt8 const& numChildren() const;

        AGXPHYSICS_EXPORT agxData::Array< agx::UInt32 > neighbors();
        AGXPHYSICS_EXPORT agxData::Array< agx::UInt32 > const neighbors() const;
        AGXPHYSICS_EXPORT agx::UInt32& neighbors(size_t index);
        AGXPHYSICS_EXPORT agx::UInt32 const& neighbors(size_t index) const;

        /// \return reference to the searchCost attribute
        AGXPHYSICS_EXPORT agx::UInt32& searchCost();
        /// \return const reference to the searchCost attribute
        AGXPHYSICS_EXPORT agx::UInt32 const& searchCost() const;

        /// \return reference to the emptyThreshold attribute
        AGXPHYSICS_EXPORT agx::Real32& emptyThreshold();
        /// \return const reference to the emptyThreshold attribute
        AGXPHYSICS_EXPORT agx::Real32 const& emptyThreshold() const;

        /// \return reference to the sizeAlignment attribute
        AGXPHYSICS_EXPORT agx::Real& sizeAlignment();
        /// \return const reference to the sizeAlignment attribute
        AGXPHYSICS_EXPORT agx::Real const& sizeAlignment() const;

        DOXYGEN_START_INTERNAL_BLOCK()
        // Only used by tolua!
        #ifdef AGX_GENERATED_TOLUA_WRAPPER
        inline agx::UInt8 const& getTier() const { return (*this)->getTier(); }
        inline agx::Vec2i const& getId() const { return (*this)->getId(); }
        inline agx::Vec2f const& getPosition() const { return (*this)->getPosition(); }
        inline agx::UInt64 const& getLinearId() const { return (*this)->getLinearId(); }
        inline agx::UInt8 const& getState() const { return (*this)->getState(); }
        inline agx::Real32 const& getEmptyTime() const { return (*this)->getEmptyTime(); }
        inline agx::IndexRange32 const& getCollisionObjects() const { return (*this)->getCollisionObjects(); }
        inline agx::UInt8 const& getZoneType() const { return (*this)->getZoneType(); }
        inline agx::UInt32 const& getParent() const { return (*this)->getParent(); }
        inline agx::UInt8 const& getNumChildren() const { return (*this)->getNumChildren(); }
        inline agx::UInt32 const& getSearchCost() const { return (*this)->getSearchCost(); }


        inline void setTier(agx::UInt8 const& value) { (*this)->setTier(value); }
        inline void setId(agx::Vec2i const& value) { (*this)->setId(value); }
        inline void setPosition(agx::Vec2f const& value) { (*this)->setPosition(value); }
        inline void setLinearId(agx::UInt64 const& value) { (*this)->setLinearId(value); }
        inline void setState(agx::UInt8 const& value) { (*this)->setState(value); }
        inline void setEmptyTime(agx::Real32 const& value) { (*this)->setEmptyTime(value); }
        inline void setCollisionObjects(agx::IndexRange32 const& value) { (*this)->setCollisionObjects(value); }
        inline void setZoneType(agx::UInt8 const& value) { (*this)->setZoneType(value); }
        inline void setParent(agx::UInt32 const& value) { (*this)->setParent(value); }
        inline void setNumChildren(agx::UInt8 const& value) { (*this)->setNumChildren(value); }
        inline void setSearchCost(agx::UInt32 const& value) { (*this)->setSearchCost(value); }
        #endif
        DOXYGEN_END_INTERNAL_BLOCK()

      };


      DOXYGEN_START_INTERNAL_BLOCK()
      class AGXPHYSICS_EXPORT Cell2DInstance : public agxData::EntityInstance
      {
      public:
        Cell2DInstance();
        Cell2DInstance(Cell2DData* data, agx::Index index);
        Cell2DInstance(agxData::EntityStorage *storage, agx::Index index);
        Cell2DInstance(const agxData::EntityInstance& other);
        Cell2DInstance(const agxData::EntityPtr& ptr);

        Cell2DData* getData();
        const Cell2DData* getData() const;

      public:
        /// \return reference to the tier attribute
        agx::UInt8& tier();
        /// \return const reference to the tier attribute
        agx::UInt8 const& tier() const;

        /// \return reference to the id attribute
        agx::Vec2i& id();
        /// \return const reference to the id attribute
        agx::Vec2i const& id() const;

        /// \return reference to the position attribute
        agx::Vec2f& position();
        /// \return const reference to the position attribute
        agx::Vec2f const& position() const;

        /// \return reference to the linearId attribute
        agx::UInt64& linearId();
        /// \return const reference to the linearId attribute
        agx::UInt64 const& linearId() const;

        /// \return reference to the state attribute
        agx::UInt8& state();
        /// \return const reference to the state attribute
        agx::UInt8 const& state() const;

        /// \return reference to the emptyTime attribute
        agx::Real32& emptyTime();
        /// \return const reference to the emptyTime attribute
        agx::Real32 const& emptyTime() const;

        /// \return reference to the collisionObjects attribute
        agx::IndexRange32& collisionObjects();
        /// \return const reference to the collisionObjects attribute
        agx::IndexRange32 const& collisionObjects() const;

        /// \return reference to the zoneType attribute
        agx::UInt8& zoneType();
        /// \return const reference to the zoneType attribute
        agx::UInt8 const& zoneType() const;

        /// \return reference to the parent attribute
        agx::UInt32& parent();
        /// \return const reference to the parent attribute
        agx::UInt32 const& parent() const;

        agxData::Array< agx::UInt32 > children();
        agxData::Array< agx::UInt32 > const children() const;
        agx::UInt32& children(size_t index);
        agx::UInt32 const& children(size_t index) const;

        /// \return reference to the numChildren attribute
        agx::UInt8& numChildren();
        /// \return const reference to the numChildren attribute
        agx::UInt8 const& numChildren() const;

        agxData::Array< agx::UInt32 > neighbors();
        agxData::Array< agx::UInt32 > const neighbors() const;
        agx::UInt32& neighbors(size_t index);
        agx::UInt32 const& neighbors(size_t index) const;

        /// \return reference to the searchCost attribute
        agx::UInt32& searchCost();
        /// \return const reference to the searchCost attribute
        agx::UInt32 const& searchCost() const;

        /// \return reference to the emptyThreshold attribute
        agx::Real32& emptyThreshold();
        /// \return const reference to the emptyThreshold attribute
        agx::Real32 const& emptyThreshold() const;

        /// \return reference to the sizeAlignment attribute
        agx::Real& sizeAlignment();
        /// \return const reference to the sizeAlignment attribute
        agx::Real const& sizeAlignment() const;

      };
      DOXYGEN_END_INTERNAL_BLOCK()



      typedef agx::VectorPOD<Cell2DPtr> Cell2DPtrVector;
      typedef agxData::Array<Cell2DPtr> Cell2DPtrArray;



      DOXYGEN_START_INTERNAL_BLOCK()
      /* Implementation */
      //-----------------------------------------------------------------------------------------------------
      //-----------------------------------------------------------------------------------------------------
      inline Cell2DInstance agx::Physics::HierarchicalGrid::Cell2DData::operator[] (size_t index) { return Cell2DInstance(this, (agx::Index)index); }
      //-----------------------------------------------------------------------------------------------------
      AGX_FORCE_INLINE Cell2DPtr::Cell2DPtr() {}
      AGX_FORCE_INLINE Cell2DPtr::Cell2DPtr(agxData::EntityStorage* storage, agx::Index id) : agxData::EntityPtr(storage, id) {}
      AGX_FORCE_INLINE Cell2DPtr::Cell2DPtr(const agxData::EntityPtr& ptr) : agxData::EntityPtr(ptr)
      {
        agxAssertN(!ptr || ptr.isInstanceOf(Cell2DModel::instance()),
          "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
          EntityPtr::getModel()->fullPath().c_str(), Cell2DModel::instance()->fullPath().c_str());
      }

      AGX_FORCE_INLINE Cell2DPtr::Cell2DPtr(const agxData::EntityInstance& instance) : agxData::EntityPtr(instance)
      {
        agxAssertN(!instance || instance.isInstanceOf(Cell2DModel::instance()),
          "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
          EntityPtr::getModel()->fullPath().c_str(), Cell2DModel::instance()->fullPath().c_str());
      }

      AGX_FORCE_INLINE Cell2DPtr& Cell2DPtr::operator= (const agxData::EntityPtr& ptr)
      {
        agxData::EntityPtr::operator= (ptr);
        agxAssertN(!ptr || ptr.isInstanceOf(Cell2DModel::instance()),
          "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
          EntityPtr::getModel()->fullPath().c_str(), Cell2DModel::instance()->fullPath().c_str());
        return *this;
      }

      AGX_FORCE_INLINE Cell2DPtr& Cell2DPtr::operator= (const agxData::EntityInstance& instance)
      {
        agxData::EntityPtr::operator= (instance);
        agxAssertN(!instance || instance.isInstanceOf(Cell2DModel::instance()),
          "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
          EntityPtr::getModel()->fullPath().c_str(), Cell2DModel::instance()->fullPath().c_str());
        return *this;
      }

      inline Cell2DInstance Cell2DPtr::instance() { return agxData::EntityPtr::instance(); }
      inline const Cell2DInstance Cell2DPtr::instance() const { return agxData::EntityPtr::instance(); }
      AGX_FORCE_INLINE Cell2DSemantics* Cell2DPtr::operator->() { return (Cell2DSemantics* )this; }
      AGX_FORCE_INLINE const Cell2DSemantics* Cell2DPtr::operator->() const { return (const Cell2DSemantics* )this; }
      AGX_FORCE_INLINE Cell2DData* Cell2DPtr::getData() { return static_cast<Cell2DData* >(agxData::EntityPtr::getData()); }
      AGX_FORCE_INLINE const Cell2DData* Cell2DPtr::getData() const { return static_cast<const Cell2DData* >(agxData::EntityPtr::getData()); }

      AGX_FORCE_INLINE agx::UInt8& Cell2DPtr::tier() { verifyIndex(); return getData()->tier[calculateIndex()]; }
      AGX_FORCE_INLINE agx::UInt8 const& Cell2DPtr::tier() const { verifyIndex(); return getData()->tier[calculateIndex()]; }

      AGX_FORCE_INLINE agx::Vec2i& Cell2DPtr::id() { verifyIndex(); return getData()->id[calculateIndex()]; }
      AGX_FORCE_INLINE agx::Vec2i const& Cell2DPtr::id() const { verifyIndex(); return getData()->id[calculateIndex()]; }

      AGX_FORCE_INLINE agx::Vec2f& Cell2DPtr::position() { verifyIndex(); return getData()->position[calculateIndex()]; }
      AGX_FORCE_INLINE agx::Vec2f const& Cell2DPtr::position() const { verifyIndex(); return getData()->position[calculateIndex()]; }

      AGX_FORCE_INLINE agx::UInt64& Cell2DPtr::linearId() { verifyIndex(); return getData()->linearId[calculateIndex()]; }
      AGX_FORCE_INLINE agx::UInt64 const& Cell2DPtr::linearId() const { verifyIndex(); return getData()->linearId[calculateIndex()]; }

      AGX_FORCE_INLINE agx::UInt8& Cell2DPtr::state() { verifyIndex(); return getData()->state[calculateIndex()]; }
      AGX_FORCE_INLINE agx::UInt8 const& Cell2DPtr::state() const { verifyIndex(); return getData()->state[calculateIndex()]; }

      AGX_FORCE_INLINE agx::Real32& Cell2DPtr::emptyTime() { verifyIndex(); return getData()->emptyTime[calculateIndex()]; }
      AGX_FORCE_INLINE agx::Real32 const& Cell2DPtr::emptyTime() const { verifyIndex(); return getData()->emptyTime[calculateIndex()]; }

      AGX_FORCE_INLINE agx::IndexRange32& Cell2DPtr::collisionObjects() { verifyIndex(); return getData()->collisionObjects[calculateIndex()]; }
      AGX_FORCE_INLINE agx::IndexRange32 const& Cell2DPtr::collisionObjects() const { verifyIndex(); return getData()->collisionObjects[calculateIndex()]; }

      AGX_FORCE_INLINE agx::UInt8& Cell2DPtr::zoneType() { verifyIndex(); return getData()->zoneType[calculateIndex()]; }
      AGX_FORCE_INLINE agx::UInt8 const& Cell2DPtr::zoneType() const { verifyIndex(); return getData()->zoneType[calculateIndex()]; }

      AGX_FORCE_INLINE agx::UInt32& Cell2DPtr::parent() { verifyIndex(); return getData()->parent[calculateIndex()]; }
      AGX_FORCE_INLINE agx::UInt32 const& Cell2DPtr::parent() const { verifyIndex(); return getData()->parent[calculateIndex()]; }

      AGX_FORCE_INLINE agxData::Array< agx::UInt32 > Cell2DPtr::children() { verifyIndex(); return getData()->children.slice(agx::IndexRange32(calculateIndex() * (agx::Index)Cell2DData::childrenArraySize, (calculateIndex()+1) * (agx::Index)Cell2DData::childrenArraySize)); }
      AGX_FORCE_INLINE const agxData::Array< agx::UInt32 > Cell2DPtr::children() const { verifyIndex(); return getData()->children.slice(agx::IndexRange32(calculateIndex() * (agx::Index)Cell2DData::childrenArraySize, (calculateIndex()+1) * (agx::Index)Cell2DData::childrenArraySize)); }
      AGX_FORCE_INLINE agx::UInt32& Cell2DPtr::children(size_t index) { return this->children()[index]; }
      AGX_FORCE_INLINE agx::UInt32 const& Cell2DPtr::children(size_t index) const { return this->children()[index]; }

      AGX_FORCE_INLINE agx::UInt8& Cell2DPtr::numChildren() { verifyIndex(); return getData()->numChildren[calculateIndex()]; }
      AGX_FORCE_INLINE agx::UInt8 const& Cell2DPtr::numChildren() const { verifyIndex(); return getData()->numChildren[calculateIndex()]; }

      AGX_FORCE_INLINE agxData::Array< agx::UInt32 > Cell2DPtr::neighbors() { verifyIndex(); return getData()->neighbors.slice(agx::IndexRange32(calculateIndex() * (agx::Index)Cell2DData::neighborsArraySize, (calculateIndex()+1) * (agx::Index)Cell2DData::neighborsArraySize)); }
      AGX_FORCE_INLINE const agxData::Array< agx::UInt32 > Cell2DPtr::neighbors() const { verifyIndex(); return getData()->neighbors.slice(agx::IndexRange32(calculateIndex() * (agx::Index)Cell2DData::neighborsArraySize, (calculateIndex()+1) * (agx::Index)Cell2DData::neighborsArraySize)); }
      AGX_FORCE_INLINE agx::UInt32& Cell2DPtr::neighbors(size_t index) { return this->neighbors()[index]; }
      AGX_FORCE_INLINE agx::UInt32 const& Cell2DPtr::neighbors(size_t index) const { return this->neighbors()[index]; }

      AGX_FORCE_INLINE agx::UInt32& Cell2DPtr::searchCost() { verifyIndex(); return getData()->searchCost[calculateIndex()]; }
      AGX_FORCE_INLINE agx::UInt32 const& Cell2DPtr::searchCost() const { verifyIndex(); return getData()->searchCost[calculateIndex()]; }

      AGX_FORCE_INLINE agx::Real32& Cell2DPtr::emptyThreshold() { verifyIndex(); return getData()->emptyThreshold; }
      AGX_FORCE_INLINE agx::Real32 const& Cell2DPtr::emptyThreshold() const { verifyIndex(); return getData()->emptyThreshold; }

      AGX_FORCE_INLINE agx::Real& Cell2DPtr::sizeAlignment() { verifyIndex(); return getData()->sizeAlignment; }
      AGX_FORCE_INLINE agx::Real const& Cell2DPtr::sizeAlignment() const { verifyIndex(); return getData()->sizeAlignment; }

      //-----------------------------------------------------------------------------------------------------
      AGX_FORCE_INLINE Cell2DInstance::Cell2DInstance() {}
      AGX_FORCE_INLINE Cell2DInstance::Cell2DInstance(Cell2DData* data, agx::Index index) : agxData::EntityInstance(data, index) {}
      AGX_FORCE_INLINE Cell2DInstance::Cell2DInstance(agxData::EntityStorage* storage, agx::Index index) : agxData::EntityInstance(storage, index) {}
      AGX_FORCE_INLINE Cell2DInstance::Cell2DInstance(const agxData::EntityInstance& other) : agxData::EntityInstance(other)
      {
        agxAssertN(!other || other.isInstanceOf(Cell2DModel::instance()),
          "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
          EntityInstance::getModel()->fullPath().c_str(), Cell2DModel::instance()->fullPath().c_str());
      }

      AGX_FORCE_INLINE Cell2DInstance::Cell2DInstance(const agxData::EntityPtr& ptr) : agxData::EntityInstance(ptr)
      {
        agxAssertN(!ptr || ptr.isInstanceOf(Cell2DModel::instance()),
          "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
          EntityInstance::getModel()->fullPath().c_str(), Cell2DModel::instance()->fullPath().c_str());
      }


      AGX_FORCE_INLINE Cell2DData* Cell2DInstance::getData() { return static_cast<Cell2DData* >(agxData::EntityInstance::getData()); }
      AGX_FORCE_INLINE const Cell2DData* Cell2DInstance::getData() const { return static_cast<const Cell2DData* >(agxData::EntityInstance::getData()); }

      AGX_FORCE_INLINE agx::UInt8& Cell2DInstance::tier() { verifyIndex(); return getData()->tier[getIndex()]; }
      AGX_FORCE_INLINE agx::UInt8 const& Cell2DInstance::tier() const { verifyIndex(); return getData()->tier[getIndex()]; }

      AGX_FORCE_INLINE agx::Vec2i& Cell2DInstance::id() { verifyIndex(); return getData()->id[getIndex()]; }
      AGX_FORCE_INLINE agx::Vec2i const& Cell2DInstance::id() const { verifyIndex(); return getData()->id[getIndex()]; }

      AGX_FORCE_INLINE agx::Vec2f& Cell2DInstance::position() { verifyIndex(); return getData()->position[getIndex()]; }
      AGX_FORCE_INLINE agx::Vec2f const& Cell2DInstance::position() const { verifyIndex(); return getData()->position[getIndex()]; }

      AGX_FORCE_INLINE agx::UInt64& Cell2DInstance::linearId() { verifyIndex(); return getData()->linearId[getIndex()]; }
      AGX_FORCE_INLINE agx::UInt64 const& Cell2DInstance::linearId() const { verifyIndex(); return getData()->linearId[getIndex()]; }

      AGX_FORCE_INLINE agx::UInt8& Cell2DInstance::state() { verifyIndex(); return getData()->state[getIndex()]; }
      AGX_FORCE_INLINE agx::UInt8 const& Cell2DInstance::state() const { verifyIndex(); return getData()->state[getIndex()]; }

      AGX_FORCE_INLINE agx::Real32& Cell2DInstance::emptyTime() { verifyIndex(); return getData()->emptyTime[getIndex()]; }
      AGX_FORCE_INLINE agx::Real32 const& Cell2DInstance::emptyTime() const { verifyIndex(); return getData()->emptyTime[getIndex()]; }

      AGX_FORCE_INLINE agx::IndexRange32& Cell2DInstance::collisionObjects() { verifyIndex(); return getData()->collisionObjects[getIndex()]; }
      AGX_FORCE_INLINE agx::IndexRange32 const& Cell2DInstance::collisionObjects() const { verifyIndex(); return getData()->collisionObjects[getIndex()]; }

      AGX_FORCE_INLINE agx::UInt8& Cell2DInstance::zoneType() { verifyIndex(); return getData()->zoneType[getIndex()]; }
      AGX_FORCE_INLINE agx::UInt8 const& Cell2DInstance::zoneType() const { verifyIndex(); return getData()->zoneType[getIndex()]; }

      AGX_FORCE_INLINE agx::UInt32& Cell2DInstance::parent() { verifyIndex(); return getData()->parent[getIndex()]; }
      AGX_FORCE_INLINE agx::UInt32 const& Cell2DInstance::parent() const { verifyIndex(); return getData()->parent[getIndex()]; }

      AGX_FORCE_INLINE agxData::Array< agx::UInt32 > Cell2DInstance::children() { verifyIndex(); return getData()->children.slice(agx::IndexRange32(getIndex() * (agx::Index)Cell2DData::childrenArraySize, (getIndex()+1) * (agx::Index)Cell2DData::childrenArraySize)); }
      AGX_FORCE_INLINE const agxData::Array< agx::UInt32 > Cell2DInstance::children() const { verifyIndex(); return getData()->children.slice(agx::IndexRange32(getIndex() * (agx::Index)Cell2DData::childrenArraySize, (getIndex()+1) * (agx::Index)Cell2DData::childrenArraySize)); }
      AGX_FORCE_INLINE agx::UInt32& Cell2DInstance::children(size_t index) { return this->children()[index]; }
      AGX_FORCE_INLINE agx::UInt32 const& Cell2DInstance::children(size_t index) const { return this->children()[index]; }

      AGX_FORCE_INLINE agx::UInt8& Cell2DInstance::numChildren() { verifyIndex(); return getData()->numChildren[getIndex()]; }
      AGX_FORCE_INLINE agx::UInt8 const& Cell2DInstance::numChildren() const { verifyIndex(); return getData()->numChildren[getIndex()]; }

      AGX_FORCE_INLINE agxData::Array< agx::UInt32 > Cell2DInstance::neighbors() { verifyIndex(); return getData()->neighbors.slice(agx::IndexRange32(getIndex() * (agx::Index)Cell2DData::neighborsArraySize, (getIndex()+1) * (agx::Index)Cell2DData::neighborsArraySize)); }
      AGX_FORCE_INLINE const agxData::Array< agx::UInt32 > Cell2DInstance::neighbors() const { verifyIndex(); return getData()->neighbors.slice(agx::IndexRange32(getIndex() * (agx::Index)Cell2DData::neighborsArraySize, (getIndex()+1) * (agx::Index)Cell2DData::neighborsArraySize)); }
      AGX_FORCE_INLINE agx::UInt32& Cell2DInstance::neighbors(size_t index) { return this->neighbors()[index]; }
      AGX_FORCE_INLINE agx::UInt32 const& Cell2DInstance::neighbors(size_t index) const { return this->neighbors()[index]; }

      AGX_FORCE_INLINE agx::UInt32& Cell2DInstance::searchCost() { verifyIndex(); return getData()->searchCost[getIndex()]; }
      AGX_FORCE_INLINE agx::UInt32 const& Cell2DInstance::searchCost() const { verifyIndex(); return getData()->searchCost[getIndex()]; }

      AGX_FORCE_INLINE agx::Real32& Cell2DInstance::emptyThreshold() { verifyIndex(); return getData()->emptyThreshold; }
      AGX_FORCE_INLINE agx::Real32 const& Cell2DInstance::emptyThreshold() const { verifyIndex(); return getData()->emptyThreshold; }

      AGX_FORCE_INLINE agx::Real& Cell2DInstance::sizeAlignment() { verifyIndex(); return getData()->sizeAlignment; }
      AGX_FORCE_INLINE agx::Real const& Cell2DInstance::sizeAlignment() const { verifyIndex(); return getData()->sizeAlignment; }

      //-----------------------------------------------------------------------------------------------------
      AGX_FORCE_INLINE Cell2DSemantics::Cell2DSemantics() {}
      //-----------------------------------------------------------------------------------------------------
      DOXYGEN_END_INTERNAL_BLOCK()
    }
  }
}

AGX_TYPE_BINDING(agx::Physics::HierarchicalGrid::Cell2DPtr, "Physics.HierarchicalGrid.Cell2DPtr")
AGX_TYPE_BINDING(agx::Physics::HierarchicalGrid::Cell2DInstance, "Physics.HierarchicalGrid.Cell2DInstance")

#ifdef _MSC_VER
# pragma warning(pop)
#endif

#undef AGX_ENTITY_WRAPPER
#undef AGX_ENTITY_NAMESPACE
#endif

