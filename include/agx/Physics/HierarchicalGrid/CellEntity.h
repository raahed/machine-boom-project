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

#ifndef GENERATED_AGX_PHYSICS_HIERARCHICALGRID_CELL_H_PLUGIN
#define GENERATED_AGX_PHYSICS_HIERARCHICALGRID_CELL_H_PLUGIN

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
#include <agx/Vec3.h>
#include <agx/Real.h>
#include <agx/IndexRange.h>
#include <agx/Physics/HierarchicalGrid/ContactZoneEntity.h>

namespace agx { namespace Physics { namespace HierarchicalGrid { class ContactZonePtr; }}}

namespace agx
{
  namespace Physics
  {
    namespace HierarchicalGrid
    {

      class CellModel;
      class CellData;
      class CellPtr;
      class CellInstance;
      class CellSemantics;


      AGX_DECLARE_POINTER_TYPES(CellModel);

      /** 
      Abstract description of the data attributes for the Physics.HierarchicalGrid.Cell entity.
      */ 
      class AGXPHYSICS_EXPORT CellModel : public agxData::EntityModel
      {
      public:
        typedef CellPtr PtrT;

        CellModel(const agx::String& name = "Cell");

        /// \return The entity model singleton.
        static CellModel* instance();

        /// Create and return a pointer to a new instance in the default storage for this entity model.
        static CellPtr createInstance();

        /// \return The default storage for this entity model.
        static agxData::EntityStorage* defaultStorage();

        /// This is part of internal cleanup and should not be called by users
        virtual void shutdownCleanup() override;



        /* Attributes */
        static agxData::ScalarAttributeT< agx::UInt8 >* tierAttribute;
        static agxData::ScalarAttributeT< agx::UInt8 >* invDepthAttribute;
        static agxData::ScalarAttributeT< agx::Vec3i >* idAttribute;
        static agxData::ScalarAttributeT< agx::Vec3 >* positionAttribute;
        static agxData::ScalarAttributeT< agx::UInt8 >* stateAttribute;
        static agxData::ScalarAttributeT< agx::Real32 >* emptyTimeAttribute;
        static agxData::ScalarAttributeT< agx::IndexRange32 >* collisionObjectsAttribute;
        static agxData::ScalarAttributeT< agx::UInt32 >* parentAttribute;
        static agxData::ScalarAttributeT< agx::UInt32 >* childrenAttribute;
        static const size_t childrenArraySize = 8;
        static agxData::ScalarAttributeT< agx::UInt8 >* numChildrenAttribute;
        static agxData::ScalarAttributeT< agx::UInt32 >* neighborsAttribute;
        static const size_t neighborsArraySize = 32;
        static agxData::ScalarAttributeT< agx::UInt32 >* searchCostAttribute;
        static agxData::ScalarAttributeT< agx::UInt32 >* sortedIndexAttribute;
        static agxData::ScalarAttributeT< agx::Physics::HierarchicalGrid::ContactZonePtr >* internalZoneAttribute;
        static agxData::ScalarAttributeT< agx::UInt8 >* zoneTypeAttribute;
        static agxData::PointerAttributeT< agx::AtomicValue*>* solveBodyOffsetsAttribute;
        static agxData::SharedAttributeT< agx::Real32 >* emptyThresholdAttribute;
        static agxData::SharedAttributeT< agx::Real >* reorderingFrequencyAttribute;
        static agxData::SharedAttributeT< agx::Real >* sizeAlignmentAttribute;

      protected:
        virtual ~CellModel();
        virtual agxData::EntityData* createData(agxData::EntityStorage* storage) override;
        virtual void configure(agx::TiXmlElement* eEntity) override;
        virtual void initAttributeAccessors() override;
        virtual void construct(agxData::EntityPtr instance) override;
        void construct(agx::Physics::HierarchicalGrid::CellPtr cell);
      };


      DOXYGEN_START_INTERNAL_BLOCK()
      #ifndef AGX_PHYSICS_HIERARCHICALGRID_CELL_DATA_SET_OVERRIDE
      #define AGX_PHYSICS_HIERARCHICALGRID_CELL_DATA_SET
      class AGXPHYSICS_EXPORT CellData : public agxData::EntityData
      {
      public:
        CellInstance operator[] (size_t index);

      public:
        agxData::Array< CellPtr >& instance;
        agxData::Array< agx::UInt8 > tier;
        agxData::Array< agx::UInt8 > invDepth;
        agxData::Array< agx::Vec3i > id;
        agxData::Array< agx::Vec3 > position;
        agxData::Array< agx::UInt8 > state;
        agxData::Array< agx::Real32 > emptyTime;
        agxData::Array< agx::IndexRange32 > collisionObjects;
        agxData::Array< agx::UInt32 > parent;
        agxData::Array< agx::UInt32 > children;
        static const size_t childrenArraySize = 8;
        agxData::Array< agx::UInt8 > numChildren;
        agxData::Array< agx::UInt32 > neighbors;
        static const size_t neighborsArraySize = 32;
        agxData::Array< agx::UInt32 > searchCost;
        agxData::Array< agx::UInt32 > sortedIndex;
        agxData::Array< agx::Physics::HierarchicalGrid::ContactZonePtr > internalZone;
        agxData::Array< agx::UInt8 > zoneType;
        agxData::Array< agx::AtomicValue* > solveBodyOffsets;
        agx::Real32 emptyThreshold;
        agx::Real reorderingFrequency;
        agx::Real sizeAlignment;

      public:
        typedef agx::UInt8 tierType;
        typedef agx::UInt8 invDepthType;
        typedef agx::Vec3i idType;
        typedef agx::Vec3 positionType;
        typedef agx::UInt8 stateType;
        typedef agx::Real32 emptyTimeType;
        typedef agx::IndexRange32 collisionObjectsType;
        typedef agx::UInt32 parentType;
        typedef agx::UInt32 childrenType;
        typedef agx::UInt8 numChildrenType;
        typedef agx::UInt32 neighborsType;
        typedef agx::UInt32 searchCostType;
        typedef agx::UInt32 sortedIndexType;
        typedef agx::Physics::HierarchicalGrid::ContactZonePtr internalZoneType;
        typedef agx::UInt8 zoneTypeType;
        typedef agx::AtomicValue* solveBodyOffsetsType;
        typedef agx::Real32 emptyThresholdType;
        typedef agx::Real reorderingFrequencyType;
        typedef agx::Real sizeAlignmentType;

      public:
        CellData(agxData::EntityStorage* storage);
        CellData();

      protected:
        virtual ~CellData() {}
        virtual void setNumElements(agx::Index numElements) override;
        virtual void synchronizeSharedAttribute(agxData::Value* value) override;

      private:
        CellData& operator= (const CellData&) { return *this; }

      };
      #endif
      DOXYGEN_END_INTERNAL_BLOCK()


      DOXYGEN_START_INTERNAL_BLOCK()
      class AGXPHYSICS_EXPORT CellSemantics : protected agxData::EntityPtr
      {
      public:

        // Automatic getters
        agx::UInt8 const& getTier() const;
        agx::UInt8 const& getInvDepth() const;
        agx::Vec3i const& getId() const;
        agx::Vec3 const& getPosition() const;
        agx::UInt8 const& getState() const;
        agx::Real32 const& getEmptyTime() const;
        agx::IndexRange32 const& getCollisionObjects() const;
        agx::UInt32 const& getParent() const;
        agx::UInt8 const& getNumChildren() const;
        agx::UInt32 const& getSearchCost() const;
        agx::UInt32 const& getSortedIndex() const;
        agx::Physics::HierarchicalGrid::ContactZonePtr const& getInternalZone() const;
        agx::UInt8 const& getZoneType() const;
        agx::AtomicValue* const& getSolveBodyOffsets() const;

        // Semantics defined by explicit kernels

        // Automatic setters
        void setTier(agx::UInt8 const& value);
        void setInvDepth(agx::UInt8 const& value);
        void setId(agx::Vec3i const& value);
        void setPosition(agx::Vec3 const& value);
        void setState(agx::UInt8 const& value);
        void setEmptyTime(agx::Real32 const& value);
        void setCollisionObjects(agx::IndexRange32 const& value);
        void setParent(agx::UInt32 const& value);
        void setNumChildren(agx::UInt8 const& value);
        void setSearchCost(agx::UInt32 const& value);
        void setSortedIndex(agx::UInt32 const& value);
        void setInternalZone(agx::Physics::HierarchicalGrid::ContactZonePtr const& value);
        void setZoneType(agx::UInt8 const& value);
        void setSolveBodyOffsets(agx::AtomicValue* const& value);


      protected:
        friend class CellPtr;
        friend class CellInstance;
        CellSemantics();
      };
      DOXYGEN_END_INTERNAL_BLOCK()


      /**
      Pointer to a entity instance of type Physics.HierarchicalGrid.Cell
      */
      class CALLABLE CellPtr : public agxData::EntityPtr
      {
      public:
        typedef CellModel ModelType;
        typedef CellData DataType;
        typedef CellInstance InstanceType;

      public:
        AGXPHYSICS_EXPORT CellPtr();
        AGXPHYSICS_EXPORT CellPtr(agxData::EntityStorage* storage, agx::Index id);
        AGXPHYSICS_EXPORT CellPtr(const agxData::EntityPtr& ptr);
        AGXPHYSICS_EXPORT CellPtr(const agxData::EntityInstance& instance);
        AGXPHYSICS_EXPORT CellPtr& operator= (const agxData::EntityPtr& ptr);
        AGXPHYSICS_EXPORT CellPtr& operator= (const agxData::EntityInstance& instance);
        AGXPHYSICS_EXPORT CellInstance instance();
        AGXPHYSICS_EXPORT const CellInstance instance() const;

        AGXPHYSICS_EXPORT CellSemantics* operator->();
        AGXPHYSICS_EXPORT const CellSemantics* operator->() const;

        CellData* getData();
        const CellData* getData() const;


        /// \return reference to the tier attribute
        AGXPHYSICS_EXPORT agx::UInt8& tier();
        /// \return const reference to the tier attribute
        AGXPHYSICS_EXPORT agx::UInt8 const& tier() const;

        /// \return reference to the invDepth attribute
        AGXPHYSICS_EXPORT agx::UInt8& invDepth();
        /// \return const reference to the invDepth attribute
        AGXPHYSICS_EXPORT agx::UInt8 const& invDepth() const;

        /// \return reference to the id attribute
        AGXPHYSICS_EXPORT agx::Vec3i& id();
        /// \return const reference to the id attribute
        AGXPHYSICS_EXPORT agx::Vec3i const& id() const;

        /// \return reference to the position attribute
        AGXPHYSICS_EXPORT agx::Vec3& position();
        /// \return const reference to the position attribute
        AGXPHYSICS_EXPORT agx::Vec3 const& position() const;

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

        /// \return reference to the sortedIndex attribute
        AGXPHYSICS_EXPORT agx::UInt32& sortedIndex();
        /// \return const reference to the sortedIndex attribute
        AGXPHYSICS_EXPORT agx::UInt32 const& sortedIndex() const;

        /// \return reference to the internalZone attribute
        AGXPHYSICS_EXPORT agx::Physics::HierarchicalGrid::ContactZonePtr& internalZone();
        /// \return const reference to the internalZone attribute
        AGXPHYSICS_EXPORT agx::Physics::HierarchicalGrid::ContactZonePtr const& internalZone() const;

        /// \return reference to the zoneType attribute
        AGXPHYSICS_EXPORT agx::UInt8& zoneType();
        /// \return const reference to the zoneType attribute
        AGXPHYSICS_EXPORT agx::UInt8 const& zoneType() const;

        /// \return reference to the solveBodyOffsets attribute
        AGXPHYSICS_EXPORT agx::AtomicValue*& solveBodyOffsets();
        /// \return const reference to the solveBodyOffsets attribute
        AGXPHYSICS_EXPORT agx::AtomicValue* const& solveBodyOffsets() const;

        /// \return reference to the emptyThreshold attribute
        AGXPHYSICS_EXPORT agx::Real32& emptyThreshold();
        /// \return const reference to the emptyThreshold attribute
        AGXPHYSICS_EXPORT agx::Real32 const& emptyThreshold() const;

        /// \return reference to the reorderingFrequency attribute
        AGXPHYSICS_EXPORT agx::Real& reorderingFrequency();
        /// \return const reference to the reorderingFrequency attribute
        AGXPHYSICS_EXPORT agx::Real const& reorderingFrequency() const;

        /// \return reference to the sizeAlignment attribute
        AGXPHYSICS_EXPORT agx::Real& sizeAlignment();
        /// \return const reference to the sizeAlignment attribute
        AGXPHYSICS_EXPORT agx::Real const& sizeAlignment() const;

        DOXYGEN_START_INTERNAL_BLOCK()
        // Only used by tolua!
        #ifdef AGX_GENERATED_TOLUA_WRAPPER
        inline agx::UInt8 const& getTier() const { return (*this)->getTier(); }
        inline agx::UInt8 const& getInvDepth() const { return (*this)->getInvDepth(); }
        inline agx::Vec3i const& getId() const { return (*this)->getId(); }
        inline agx::Vec3 const& getPosition() const { return (*this)->getPosition(); }
        inline agx::UInt8 const& getState() const { return (*this)->getState(); }
        inline agx::Real32 const& getEmptyTime() const { return (*this)->getEmptyTime(); }
        inline agx::IndexRange32 const& getCollisionObjects() const { return (*this)->getCollisionObjects(); }
        inline agx::UInt32 const& getParent() const { return (*this)->getParent(); }
        inline agx::UInt8 const& getNumChildren() const { return (*this)->getNumChildren(); }
        inline agx::UInt32 const& getSearchCost() const { return (*this)->getSearchCost(); }
        inline agx::UInt32 const& getSortedIndex() const { return (*this)->getSortedIndex(); }
        inline agx::Physics::HierarchicalGrid::ContactZonePtr const& getInternalZone() const { return (*this)->getInternalZone(); }
        inline agx::UInt8 const& getZoneType() const { return (*this)->getZoneType(); }
        inline agx::AtomicValue* const& getSolveBodyOffsets() const { return (*this)->getSolveBodyOffsets(); }


        inline void setTier(agx::UInt8 const& value) { (*this)->setTier(value); }
        inline void setInvDepth(agx::UInt8 const& value) { (*this)->setInvDepth(value); }
        inline void setId(agx::Vec3i const& value) { (*this)->setId(value); }
        inline void setPosition(agx::Vec3 const& value) { (*this)->setPosition(value); }
        inline void setState(agx::UInt8 const& value) { (*this)->setState(value); }
        inline void setEmptyTime(agx::Real32 const& value) { (*this)->setEmptyTime(value); }
        inline void setCollisionObjects(agx::IndexRange32 const& value) { (*this)->setCollisionObjects(value); }
        inline void setParent(agx::UInt32 const& value) { (*this)->setParent(value); }
        inline void setNumChildren(agx::UInt8 const& value) { (*this)->setNumChildren(value); }
        inline void setSearchCost(agx::UInt32 const& value) { (*this)->setSearchCost(value); }
        inline void setSortedIndex(agx::UInt32 const& value) { (*this)->setSortedIndex(value); }
        inline void setInternalZone(agx::Physics::HierarchicalGrid::ContactZonePtr const& value) { (*this)->setInternalZone(value); }
        inline void setZoneType(agx::UInt8 const& value) { (*this)->setZoneType(value); }
        inline void setSolveBodyOffsets(agx::AtomicValue* const& value) { (*this)->setSolveBodyOffsets(value); }
        #endif
        DOXYGEN_END_INTERNAL_BLOCK()

      };


      DOXYGEN_START_INTERNAL_BLOCK()
      class AGXPHYSICS_EXPORT CellInstance : public agxData::EntityInstance
      {
      public:
        CellInstance();
        CellInstance(CellData* data, agx::Index index);
        CellInstance(agxData::EntityStorage *storage, agx::Index index);
        CellInstance(const agxData::EntityInstance& other);
        CellInstance(const agxData::EntityPtr& ptr);

        CellData* getData();
        const CellData* getData() const;

      public:
        /// \return reference to the tier attribute
        agx::UInt8& tier();
        /// \return const reference to the tier attribute
        agx::UInt8 const& tier() const;

        /// \return reference to the invDepth attribute
        agx::UInt8& invDepth();
        /// \return const reference to the invDepth attribute
        agx::UInt8 const& invDepth() const;

        /// \return reference to the id attribute
        agx::Vec3i& id();
        /// \return const reference to the id attribute
        agx::Vec3i const& id() const;

        /// \return reference to the position attribute
        agx::Vec3& position();
        /// \return const reference to the position attribute
        agx::Vec3 const& position() const;

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

        /// \return reference to the sortedIndex attribute
        agx::UInt32& sortedIndex();
        /// \return const reference to the sortedIndex attribute
        agx::UInt32 const& sortedIndex() const;

        /// \return reference to the internalZone attribute
        agx::Physics::HierarchicalGrid::ContactZonePtr& internalZone();
        /// \return const reference to the internalZone attribute
        agx::Physics::HierarchicalGrid::ContactZonePtr const& internalZone() const;

        /// \return reference to the zoneType attribute
        agx::UInt8& zoneType();
        /// \return const reference to the zoneType attribute
        agx::UInt8 const& zoneType() const;

        /// \return reference to the solveBodyOffsets attribute
        agx::AtomicValue*& solveBodyOffsets();
        /// \return const reference to the solveBodyOffsets attribute
        agx::AtomicValue* const& solveBodyOffsets() const;

        /// \return reference to the emptyThreshold attribute
        agx::Real32& emptyThreshold();
        /// \return const reference to the emptyThreshold attribute
        agx::Real32 const& emptyThreshold() const;

        /// \return reference to the reorderingFrequency attribute
        agx::Real& reorderingFrequency();
        /// \return const reference to the reorderingFrequency attribute
        agx::Real const& reorderingFrequency() const;

        /// \return reference to the sizeAlignment attribute
        agx::Real& sizeAlignment();
        /// \return const reference to the sizeAlignment attribute
        agx::Real const& sizeAlignment() const;

      };
      DOXYGEN_END_INTERNAL_BLOCK()



      typedef agx::VectorPOD<CellPtr> CellPtrVector;
      typedef agxData::Array<CellPtr> CellPtrArray;



      DOXYGEN_START_INTERNAL_BLOCK()
      /* Implementation */
      //-----------------------------------------------------------------------------------------------------
      //-----------------------------------------------------------------------------------------------------
      inline CellInstance agx::Physics::HierarchicalGrid::CellData::operator[] (size_t index) { return CellInstance(this, (agx::Index)index); }
      //-----------------------------------------------------------------------------------------------------
      AGX_FORCE_INLINE CellPtr::CellPtr() {}
      AGX_FORCE_INLINE CellPtr::CellPtr(agxData::EntityStorage* storage, agx::Index id) : agxData::EntityPtr(storage, id) {}
      AGX_FORCE_INLINE CellPtr::CellPtr(const agxData::EntityPtr& ptr) : agxData::EntityPtr(ptr)
      {
        agxAssertN(!ptr || ptr.isInstanceOf(CellModel::instance()),
          "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
          EntityPtr::getModel()->fullPath().c_str(), CellModel::instance()->fullPath().c_str());
      }

      AGX_FORCE_INLINE CellPtr::CellPtr(const agxData::EntityInstance& instance) : agxData::EntityPtr(instance)
      {
        agxAssertN(!instance || instance.isInstanceOf(CellModel::instance()),
          "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
          EntityPtr::getModel()->fullPath().c_str(), CellModel::instance()->fullPath().c_str());
      }

      AGX_FORCE_INLINE CellPtr& CellPtr::operator= (const agxData::EntityPtr& ptr)
      {
        agxData::EntityPtr::operator= (ptr);
        agxAssertN(!ptr || ptr.isInstanceOf(CellModel::instance()),
          "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
          EntityPtr::getModel()->fullPath().c_str(), CellModel::instance()->fullPath().c_str());
        return *this;
      }

      AGX_FORCE_INLINE CellPtr& CellPtr::operator= (const agxData::EntityInstance& instance)
      {
        agxData::EntityPtr::operator= (instance);
        agxAssertN(!instance || instance.isInstanceOf(CellModel::instance()),
          "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
          EntityPtr::getModel()->fullPath().c_str(), CellModel::instance()->fullPath().c_str());
        return *this;
      }

      inline CellInstance CellPtr::instance() { return agxData::EntityPtr::instance(); }
      inline const CellInstance CellPtr::instance() const { return agxData::EntityPtr::instance(); }
      AGX_FORCE_INLINE CellSemantics* CellPtr::operator->() { return (CellSemantics* )this; }
      AGX_FORCE_INLINE const CellSemantics* CellPtr::operator->() const { return (const CellSemantics* )this; }
      AGX_FORCE_INLINE CellData* CellPtr::getData() { return static_cast<CellData* >(agxData::EntityPtr::getData()); }
      AGX_FORCE_INLINE const CellData* CellPtr::getData() const { return static_cast<const CellData* >(agxData::EntityPtr::getData()); }

      AGX_FORCE_INLINE agx::UInt8& CellPtr::tier() { verifyIndex(); return getData()->tier[calculateIndex()]; }
      AGX_FORCE_INLINE agx::UInt8 const& CellPtr::tier() const { verifyIndex(); return getData()->tier[calculateIndex()]; }

      AGX_FORCE_INLINE agx::UInt8& CellPtr::invDepth() { verifyIndex(); return getData()->invDepth[calculateIndex()]; }
      AGX_FORCE_INLINE agx::UInt8 const& CellPtr::invDepth() const { verifyIndex(); return getData()->invDepth[calculateIndex()]; }

      AGX_FORCE_INLINE agx::Vec3i& CellPtr::id() { verifyIndex(); return getData()->id[calculateIndex()]; }
      AGX_FORCE_INLINE agx::Vec3i const& CellPtr::id() const { verifyIndex(); return getData()->id[calculateIndex()]; }

      AGX_FORCE_INLINE agx::Vec3& CellPtr::position() { verifyIndex(); return getData()->position[calculateIndex()]; }
      AGX_FORCE_INLINE agx::Vec3 const& CellPtr::position() const { verifyIndex(); return getData()->position[calculateIndex()]; }

      AGX_FORCE_INLINE agx::UInt8& CellPtr::state() { verifyIndex(); return getData()->state[calculateIndex()]; }
      AGX_FORCE_INLINE agx::UInt8 const& CellPtr::state() const { verifyIndex(); return getData()->state[calculateIndex()]; }

      AGX_FORCE_INLINE agx::Real32& CellPtr::emptyTime() { verifyIndex(); return getData()->emptyTime[calculateIndex()]; }
      AGX_FORCE_INLINE agx::Real32 const& CellPtr::emptyTime() const { verifyIndex(); return getData()->emptyTime[calculateIndex()]; }

      AGX_FORCE_INLINE agx::IndexRange32& CellPtr::collisionObjects() { verifyIndex(); return getData()->collisionObjects[calculateIndex()]; }
      AGX_FORCE_INLINE agx::IndexRange32 const& CellPtr::collisionObjects() const { verifyIndex(); return getData()->collisionObjects[calculateIndex()]; }

      AGX_FORCE_INLINE agx::UInt32& CellPtr::parent() { verifyIndex(); return getData()->parent[calculateIndex()]; }
      AGX_FORCE_INLINE agx::UInt32 const& CellPtr::parent() const { verifyIndex(); return getData()->parent[calculateIndex()]; }

      AGX_FORCE_INLINE agxData::Array< agx::UInt32 > CellPtr::children() { verifyIndex(); return getData()->children.slice(agx::IndexRange32(calculateIndex() * (agx::Index)CellData::childrenArraySize, (calculateIndex()+1) * (agx::Index)CellData::childrenArraySize)); }
      AGX_FORCE_INLINE const agxData::Array< agx::UInt32 > CellPtr::children() const { verifyIndex(); return getData()->children.slice(agx::IndexRange32(calculateIndex() * (agx::Index)CellData::childrenArraySize, (calculateIndex()+1) * (agx::Index)CellData::childrenArraySize)); }
      AGX_FORCE_INLINE agx::UInt32& CellPtr::children(size_t index) { return this->children()[index]; }
      AGX_FORCE_INLINE agx::UInt32 const& CellPtr::children(size_t index) const { return this->children()[index]; }

      AGX_FORCE_INLINE agx::UInt8& CellPtr::numChildren() { verifyIndex(); return getData()->numChildren[calculateIndex()]; }
      AGX_FORCE_INLINE agx::UInt8 const& CellPtr::numChildren() const { verifyIndex(); return getData()->numChildren[calculateIndex()]; }

      AGX_FORCE_INLINE agxData::Array< agx::UInt32 > CellPtr::neighbors() { verifyIndex(); return getData()->neighbors.slice(agx::IndexRange32(calculateIndex() * (agx::Index)CellData::neighborsArraySize, (calculateIndex()+1) * (agx::Index)CellData::neighborsArraySize)); }
      AGX_FORCE_INLINE const agxData::Array< agx::UInt32 > CellPtr::neighbors() const { verifyIndex(); return getData()->neighbors.slice(agx::IndexRange32(calculateIndex() * (agx::Index)CellData::neighborsArraySize, (calculateIndex()+1) * (agx::Index)CellData::neighborsArraySize)); }
      AGX_FORCE_INLINE agx::UInt32& CellPtr::neighbors(size_t index) { return this->neighbors()[index]; }
      AGX_FORCE_INLINE agx::UInt32 const& CellPtr::neighbors(size_t index) const { return this->neighbors()[index]; }

      AGX_FORCE_INLINE agx::UInt32& CellPtr::searchCost() { verifyIndex(); return getData()->searchCost[calculateIndex()]; }
      AGX_FORCE_INLINE agx::UInt32 const& CellPtr::searchCost() const { verifyIndex(); return getData()->searchCost[calculateIndex()]; }

      AGX_FORCE_INLINE agx::UInt32& CellPtr::sortedIndex() { verifyIndex(); return getData()->sortedIndex[calculateIndex()]; }
      AGX_FORCE_INLINE agx::UInt32 const& CellPtr::sortedIndex() const { verifyIndex(); return getData()->sortedIndex[calculateIndex()]; }

      AGX_FORCE_INLINE agx::Physics::HierarchicalGrid::ContactZonePtr& CellPtr::internalZone() { verifyIndex(); return getData()->internalZone[calculateIndex()]; }
      AGX_FORCE_INLINE agx::Physics::HierarchicalGrid::ContactZonePtr const& CellPtr::internalZone() const { verifyIndex(); return getData()->internalZone[calculateIndex()]; }

      AGX_FORCE_INLINE agx::UInt8& CellPtr::zoneType() { verifyIndex(); return getData()->zoneType[calculateIndex()]; }
      AGX_FORCE_INLINE agx::UInt8 const& CellPtr::zoneType() const { verifyIndex(); return getData()->zoneType[calculateIndex()]; }

      AGX_FORCE_INLINE agx::AtomicValue*& CellPtr::solveBodyOffsets() { verifyIndex(); return getData()->solveBodyOffsets[calculateIndex()]; }
      AGX_FORCE_INLINE agx::AtomicValue* const& CellPtr::solveBodyOffsets() const { verifyIndex(); return getData()->solveBodyOffsets[calculateIndex()]; }

      AGX_FORCE_INLINE agx::Real32& CellPtr::emptyThreshold() { verifyIndex(); return getData()->emptyThreshold; }
      AGX_FORCE_INLINE agx::Real32 const& CellPtr::emptyThreshold() const { verifyIndex(); return getData()->emptyThreshold; }

      AGX_FORCE_INLINE agx::Real& CellPtr::reorderingFrequency() { verifyIndex(); return getData()->reorderingFrequency; }
      AGX_FORCE_INLINE agx::Real const& CellPtr::reorderingFrequency() const { verifyIndex(); return getData()->reorderingFrequency; }

      AGX_FORCE_INLINE agx::Real& CellPtr::sizeAlignment() { verifyIndex(); return getData()->sizeAlignment; }
      AGX_FORCE_INLINE agx::Real const& CellPtr::sizeAlignment() const { verifyIndex(); return getData()->sizeAlignment; }

      //-----------------------------------------------------------------------------------------------------
      AGX_FORCE_INLINE CellInstance::CellInstance() {}
      AGX_FORCE_INLINE CellInstance::CellInstance(CellData* data, agx::Index index) : agxData::EntityInstance(data, index) {}
      AGX_FORCE_INLINE CellInstance::CellInstance(agxData::EntityStorage* storage, agx::Index index) : agxData::EntityInstance(storage, index) {}
      AGX_FORCE_INLINE CellInstance::CellInstance(const agxData::EntityInstance& other) : agxData::EntityInstance(other)
      {
        agxAssertN(!other || other.isInstanceOf(CellModel::instance()),
          "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
          EntityInstance::getModel()->fullPath().c_str(), CellModel::instance()->fullPath().c_str());
      }

      AGX_FORCE_INLINE CellInstance::CellInstance(const agxData::EntityPtr& ptr) : agxData::EntityInstance(ptr)
      {
        agxAssertN(!ptr || ptr.isInstanceOf(CellModel::instance()),
          "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
          EntityInstance::getModel()->fullPath().c_str(), CellModel::instance()->fullPath().c_str());
      }


      AGX_FORCE_INLINE CellData* CellInstance::getData() { return static_cast<CellData* >(agxData::EntityInstance::getData()); }
      AGX_FORCE_INLINE const CellData* CellInstance::getData() const { return static_cast<const CellData* >(agxData::EntityInstance::getData()); }

      AGX_FORCE_INLINE agx::UInt8& CellInstance::tier() { verifyIndex(); return getData()->tier[getIndex()]; }
      AGX_FORCE_INLINE agx::UInt8 const& CellInstance::tier() const { verifyIndex(); return getData()->tier[getIndex()]; }

      AGX_FORCE_INLINE agx::UInt8& CellInstance::invDepth() { verifyIndex(); return getData()->invDepth[getIndex()]; }
      AGX_FORCE_INLINE agx::UInt8 const& CellInstance::invDepth() const { verifyIndex(); return getData()->invDepth[getIndex()]; }

      AGX_FORCE_INLINE agx::Vec3i& CellInstance::id() { verifyIndex(); return getData()->id[getIndex()]; }
      AGX_FORCE_INLINE agx::Vec3i const& CellInstance::id() const { verifyIndex(); return getData()->id[getIndex()]; }

      AGX_FORCE_INLINE agx::Vec3& CellInstance::position() { verifyIndex(); return getData()->position[getIndex()]; }
      AGX_FORCE_INLINE agx::Vec3 const& CellInstance::position() const { verifyIndex(); return getData()->position[getIndex()]; }

      AGX_FORCE_INLINE agx::UInt8& CellInstance::state() { verifyIndex(); return getData()->state[getIndex()]; }
      AGX_FORCE_INLINE agx::UInt8 const& CellInstance::state() const { verifyIndex(); return getData()->state[getIndex()]; }

      AGX_FORCE_INLINE agx::Real32& CellInstance::emptyTime() { verifyIndex(); return getData()->emptyTime[getIndex()]; }
      AGX_FORCE_INLINE agx::Real32 const& CellInstance::emptyTime() const { verifyIndex(); return getData()->emptyTime[getIndex()]; }

      AGX_FORCE_INLINE agx::IndexRange32& CellInstance::collisionObjects() { verifyIndex(); return getData()->collisionObjects[getIndex()]; }
      AGX_FORCE_INLINE agx::IndexRange32 const& CellInstance::collisionObjects() const { verifyIndex(); return getData()->collisionObjects[getIndex()]; }

      AGX_FORCE_INLINE agx::UInt32& CellInstance::parent() { verifyIndex(); return getData()->parent[getIndex()]; }
      AGX_FORCE_INLINE agx::UInt32 const& CellInstance::parent() const { verifyIndex(); return getData()->parent[getIndex()]; }

      AGX_FORCE_INLINE agxData::Array< agx::UInt32 > CellInstance::children() { verifyIndex(); return getData()->children.slice(agx::IndexRange32(getIndex() * (agx::Index)CellData::childrenArraySize, (getIndex()+1) * (agx::Index)CellData::childrenArraySize)); }
      AGX_FORCE_INLINE const agxData::Array< agx::UInt32 > CellInstance::children() const { verifyIndex(); return getData()->children.slice(agx::IndexRange32(getIndex() * (agx::Index)CellData::childrenArraySize, (getIndex()+1) * (agx::Index)CellData::childrenArraySize)); }
      AGX_FORCE_INLINE agx::UInt32& CellInstance::children(size_t index) { return this->children()[index]; }
      AGX_FORCE_INLINE agx::UInt32 const& CellInstance::children(size_t index) const { return this->children()[index]; }

      AGX_FORCE_INLINE agx::UInt8& CellInstance::numChildren() { verifyIndex(); return getData()->numChildren[getIndex()]; }
      AGX_FORCE_INLINE agx::UInt8 const& CellInstance::numChildren() const { verifyIndex(); return getData()->numChildren[getIndex()]; }

      AGX_FORCE_INLINE agxData::Array< agx::UInt32 > CellInstance::neighbors() { verifyIndex(); return getData()->neighbors.slice(agx::IndexRange32(getIndex() * (agx::Index)CellData::neighborsArraySize, (getIndex()+1) * (agx::Index)CellData::neighborsArraySize)); }
      AGX_FORCE_INLINE const agxData::Array< agx::UInt32 > CellInstance::neighbors() const { verifyIndex(); return getData()->neighbors.slice(agx::IndexRange32(getIndex() * (agx::Index)CellData::neighborsArraySize, (getIndex()+1) * (agx::Index)CellData::neighborsArraySize)); }
      AGX_FORCE_INLINE agx::UInt32& CellInstance::neighbors(size_t index) { return this->neighbors()[index]; }
      AGX_FORCE_INLINE agx::UInt32 const& CellInstance::neighbors(size_t index) const { return this->neighbors()[index]; }

      AGX_FORCE_INLINE agx::UInt32& CellInstance::searchCost() { verifyIndex(); return getData()->searchCost[getIndex()]; }
      AGX_FORCE_INLINE agx::UInt32 const& CellInstance::searchCost() const { verifyIndex(); return getData()->searchCost[getIndex()]; }

      AGX_FORCE_INLINE agx::UInt32& CellInstance::sortedIndex() { verifyIndex(); return getData()->sortedIndex[getIndex()]; }
      AGX_FORCE_INLINE agx::UInt32 const& CellInstance::sortedIndex() const { verifyIndex(); return getData()->sortedIndex[getIndex()]; }

      AGX_FORCE_INLINE agx::Physics::HierarchicalGrid::ContactZonePtr& CellInstance::internalZone() { verifyIndex(); return getData()->internalZone[getIndex()]; }
      AGX_FORCE_INLINE agx::Physics::HierarchicalGrid::ContactZonePtr const& CellInstance::internalZone() const { verifyIndex(); return getData()->internalZone[getIndex()]; }

      AGX_FORCE_INLINE agx::UInt8& CellInstance::zoneType() { verifyIndex(); return getData()->zoneType[getIndex()]; }
      AGX_FORCE_INLINE agx::UInt8 const& CellInstance::zoneType() const { verifyIndex(); return getData()->zoneType[getIndex()]; }

      AGX_FORCE_INLINE agx::AtomicValue*& CellInstance::solveBodyOffsets() { verifyIndex(); return getData()->solveBodyOffsets[getIndex()]; }
      AGX_FORCE_INLINE agx::AtomicValue* const& CellInstance::solveBodyOffsets() const { verifyIndex(); return getData()->solveBodyOffsets[getIndex()]; }

      AGX_FORCE_INLINE agx::Real32& CellInstance::emptyThreshold() { verifyIndex(); return getData()->emptyThreshold; }
      AGX_FORCE_INLINE agx::Real32 const& CellInstance::emptyThreshold() const { verifyIndex(); return getData()->emptyThreshold; }

      AGX_FORCE_INLINE agx::Real& CellInstance::reorderingFrequency() { verifyIndex(); return getData()->reorderingFrequency; }
      AGX_FORCE_INLINE agx::Real const& CellInstance::reorderingFrequency() const { verifyIndex(); return getData()->reorderingFrequency; }

      AGX_FORCE_INLINE agx::Real& CellInstance::sizeAlignment() { verifyIndex(); return getData()->sizeAlignment; }
      AGX_FORCE_INLINE agx::Real const& CellInstance::sizeAlignment() const { verifyIndex(); return getData()->sizeAlignment; }

      //-----------------------------------------------------------------------------------------------------
      AGX_FORCE_INLINE CellSemantics::CellSemantics() {}
      //-----------------------------------------------------------------------------------------------------
      DOXYGEN_END_INTERNAL_BLOCK()
    }
  }
}

AGX_TYPE_BINDING(agx::Physics::HierarchicalGrid::CellPtr, "Physics.HierarchicalGrid.CellPtr")
AGX_TYPE_BINDING(agx::Physics::HierarchicalGrid::CellInstance, "Physics.HierarchicalGrid.CellInstance")

#ifdef _MSC_VER
# pragma warning(pop)
#endif

#undef AGX_ENTITY_WRAPPER
#undef AGX_ENTITY_NAMESPACE
#endif

