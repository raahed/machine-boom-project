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

#ifndef GENERATED_AGX_PHYSICS_HIERARCHICALGRID_GRIDTIER_H_PLUGIN
#define GENERATED_AGX_PHYSICS_HIERARCHICALGRID_GRIDTIER_H_PLUGIN

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
#include <agx/Real.h>
#include <agx/Physics/HierarchicalGrid/GridTables.h>


namespace agx
{
  namespace Physics
  {
    namespace HierarchicalGrid
    {

      class GridTierModel;
      class GridTierData;
      class GridTierPtr;
      class GridTierInstance;
      class GridTierSemantics;


      AGX_DECLARE_POINTER_TYPES(GridTierModel);

      /** 
      Abstract description of the data attributes for the Physics.HierarchicalGrid.GridTier entity.
      */ 
      class AGXPHYSICS_EXPORT GridTierModel : public agxData::EntityModel
      {
      public:
        typedef GridTierPtr PtrT;

        GridTierModel(const agx::String& name = "GridTier");

        /// \return The entity model singleton.
        static GridTierModel* instance();

        /// Create and return a pointer to a new instance in the default storage for this entity model.
        static GridTierPtr createInstance();

        /// \return The default storage for this entity model.
        static agxData::EntityStorage* defaultStorage();

        /// This is part of internal cleanup and should not be called by users
        virtual void shutdownCleanup() override;



        /* Attributes */
        static agxData::ScalarAttributeT< agx::Real >* sizeAttribute;
        static agxData::ScalarAttributeT< agx::Real >* invSizeAttribute;
        static agxData::ScalarAttributeT< agx::Real32 >* boundingRadiusAttribute;
        static agxData::PointerAttributeT< agx::GridCellTable*>* cellTableAttribute;
        static agxData::PointerAttributeT< agx::ContactZoneTable*>* zoneTableAttribute;

      protected:
        virtual ~GridTierModel();
        virtual agxData::EntityData* createData(agxData::EntityStorage* storage) override;
        virtual void configure(agx::TiXmlElement* eEntity) override;
        virtual void initAttributeAccessors() override;
        virtual void construct(agxData::EntityPtr instance) override;
        void construct(agx::Physics::HierarchicalGrid::GridTierPtr gridTier);
      };


      DOXYGEN_START_INTERNAL_BLOCK()
      #ifndef AGX_PHYSICS_HIERARCHICALGRID_GRIDTIER_DATA_SET_OVERRIDE
      #define AGX_PHYSICS_HIERARCHICALGRID_GRIDTIER_DATA_SET
      class AGXPHYSICS_EXPORT GridTierData : public agxData::EntityData
      {
      public:
        GridTierInstance operator[] (size_t index);

      public:
        agxData::Array< GridTierPtr >& instance;
        agxData::Array< agx::Real > size;
        agxData::Array< agx::Real > invSize;
        agxData::Array< agx::Real32 > boundingRadius;
        agxData::Array< agx::GridCellTable* > cellTable;
        agxData::Array< agx::ContactZoneTable* > zoneTable;

      public:
        typedef agx::Real sizeType;
        typedef agx::Real invSizeType;
        typedef agx::Real32 boundingRadiusType;
        typedef agx::GridCellTable* cellTableType;
        typedef agx::ContactZoneTable* zoneTableType;

      public:
        GridTierData(agxData::EntityStorage* storage);
        GridTierData();

      protected:
        virtual ~GridTierData() {}
        virtual void setNumElements(agx::Index numElements) override;

      private:
        GridTierData& operator= (const GridTierData&) { return *this; }

      };
      #endif
      DOXYGEN_END_INTERNAL_BLOCK()


      DOXYGEN_START_INTERNAL_BLOCK()
      class AGXPHYSICS_EXPORT GridTierSemantics : protected agxData::EntityPtr
      {
      public:

        // Automatic getters
        agx::Real const& getSize() const;
        agx::Real const& getInvSize() const;
        agx::Real32 const& getBoundingRadius() const;
        agx::GridCellTable* const& getCellTable() const;
        agx::ContactZoneTable* const& getZoneTable() const;

        // Semantics defined by explicit kernels

        // Automatic setters
        void setSize(agx::Real const& value);
        void setInvSize(agx::Real const& value);
        void setBoundingRadius(agx::Real32 const& value);
        void setCellTable(agx::GridCellTable* const& value);
        void setZoneTable(agx::ContactZoneTable* const& value);


      protected:
        friend class GridTierPtr;
        friend class GridTierInstance;
        GridTierSemantics();
      };
      DOXYGEN_END_INTERNAL_BLOCK()


      /**
      Pointer to a entity instance of type Physics.HierarchicalGrid.GridTier
      */
      class CALLABLE GridTierPtr : public agxData::EntityPtr
      {
      public:
        typedef GridTierModel ModelType;
        typedef GridTierData DataType;
        typedef GridTierInstance InstanceType;

      public:
        AGXPHYSICS_EXPORT GridTierPtr();
        AGXPHYSICS_EXPORT GridTierPtr(agxData::EntityStorage* storage, agx::Index id);
        AGXPHYSICS_EXPORT GridTierPtr(const agxData::EntityPtr& ptr);
        AGXPHYSICS_EXPORT GridTierPtr(const agxData::EntityInstance& instance);
        AGXPHYSICS_EXPORT GridTierPtr& operator= (const agxData::EntityPtr& ptr);
        AGXPHYSICS_EXPORT GridTierPtr& operator= (const agxData::EntityInstance& instance);
        AGXPHYSICS_EXPORT GridTierInstance instance();
        AGXPHYSICS_EXPORT const GridTierInstance instance() const;

        AGXPHYSICS_EXPORT GridTierSemantics* operator->();
        AGXPHYSICS_EXPORT const GridTierSemantics* operator->() const;

        GridTierData* getData();
        const GridTierData* getData() const;


        /// \return reference to the size attribute
        AGXPHYSICS_EXPORT agx::Real& size();
        /// \return const reference to the size attribute
        AGXPHYSICS_EXPORT agx::Real const& size() const;

        /// \return reference to the invSize attribute
        AGXPHYSICS_EXPORT agx::Real& invSize();
        /// \return const reference to the invSize attribute
        AGXPHYSICS_EXPORT agx::Real const& invSize() const;

        /// \return reference to the boundingRadius attribute
        AGXPHYSICS_EXPORT agx::Real32& boundingRadius();
        /// \return const reference to the boundingRadius attribute
        AGXPHYSICS_EXPORT agx::Real32 const& boundingRadius() const;

        /// \return reference to the cellTable attribute
        AGXPHYSICS_EXPORT agx::GridCellTable*& cellTable();
        /// \return const reference to the cellTable attribute
        AGXPHYSICS_EXPORT agx::GridCellTable* const& cellTable() const;

        /// \return reference to the zoneTable attribute
        AGXPHYSICS_EXPORT agx::ContactZoneTable*& zoneTable();
        /// \return const reference to the zoneTable attribute
        AGXPHYSICS_EXPORT agx::ContactZoneTable* const& zoneTable() const;

        DOXYGEN_START_INTERNAL_BLOCK()
        // Only used by tolua!
        #ifdef AGX_GENERATED_TOLUA_WRAPPER
        inline agx::Real const& getSize() const { return (*this)->getSize(); }
        inline agx::Real const& getInvSize() const { return (*this)->getInvSize(); }
        inline agx::Real32 const& getBoundingRadius() const { return (*this)->getBoundingRadius(); }
        inline agx::GridCellTable* const& getCellTable() const { return (*this)->getCellTable(); }
        inline agx::ContactZoneTable* const& getZoneTable() const { return (*this)->getZoneTable(); }


        inline void setSize(agx::Real const& value) { (*this)->setSize(value); }
        inline void setInvSize(agx::Real const& value) { (*this)->setInvSize(value); }
        inline void setBoundingRadius(agx::Real32 const& value) { (*this)->setBoundingRadius(value); }
        inline void setCellTable(agx::GridCellTable* const& value) { (*this)->setCellTable(value); }
        inline void setZoneTable(agx::ContactZoneTable* const& value) { (*this)->setZoneTable(value); }
        #endif
        DOXYGEN_END_INTERNAL_BLOCK()

      };


      DOXYGEN_START_INTERNAL_BLOCK()
      class AGXPHYSICS_EXPORT GridTierInstance : public agxData::EntityInstance
      {
      public:
        GridTierInstance();
        GridTierInstance(GridTierData* data, agx::Index index);
        GridTierInstance(agxData::EntityStorage *storage, agx::Index index);
        GridTierInstance(const agxData::EntityInstance& other);
        GridTierInstance(const agxData::EntityPtr& ptr);

        GridTierData* getData();
        const GridTierData* getData() const;

      public:
        /// \return reference to the size attribute
        agx::Real& size();
        /// \return const reference to the size attribute
        agx::Real const& size() const;

        /// \return reference to the invSize attribute
        agx::Real& invSize();
        /// \return const reference to the invSize attribute
        agx::Real const& invSize() const;

        /// \return reference to the boundingRadius attribute
        agx::Real32& boundingRadius();
        /// \return const reference to the boundingRadius attribute
        agx::Real32 const& boundingRadius() const;

        /// \return reference to the cellTable attribute
        agx::GridCellTable*& cellTable();
        /// \return const reference to the cellTable attribute
        agx::GridCellTable* const& cellTable() const;

        /// \return reference to the zoneTable attribute
        agx::ContactZoneTable*& zoneTable();
        /// \return const reference to the zoneTable attribute
        agx::ContactZoneTable* const& zoneTable() const;

      };
      DOXYGEN_END_INTERNAL_BLOCK()



      typedef agx::VectorPOD<GridTierPtr> GridTierPtrVector;
      typedef agxData::Array<GridTierPtr> GridTierPtrArray;



      DOXYGEN_START_INTERNAL_BLOCK()
      /* Implementation */
      //-----------------------------------------------------------------------------------------------------
      //-----------------------------------------------------------------------------------------------------
      inline GridTierInstance agx::Physics::HierarchicalGrid::GridTierData::operator[] (size_t index) { return GridTierInstance(this, (agx::Index)index); }
      //-----------------------------------------------------------------------------------------------------
      AGX_FORCE_INLINE GridTierPtr::GridTierPtr() {}
      AGX_FORCE_INLINE GridTierPtr::GridTierPtr(agxData::EntityStorage* storage, agx::Index id) : agxData::EntityPtr(storage, id) {}
      AGX_FORCE_INLINE GridTierPtr::GridTierPtr(const agxData::EntityPtr& ptr) : agxData::EntityPtr(ptr)
      {
        agxAssertN(!ptr || ptr.isInstanceOf(GridTierModel::instance()),
          "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
          EntityPtr::getModel()->fullPath().c_str(), GridTierModel::instance()->fullPath().c_str());
      }

      AGX_FORCE_INLINE GridTierPtr::GridTierPtr(const agxData::EntityInstance& instance) : agxData::EntityPtr(instance)
      {
        agxAssertN(!instance || instance.isInstanceOf(GridTierModel::instance()),
          "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
          EntityPtr::getModel()->fullPath().c_str(), GridTierModel::instance()->fullPath().c_str());
      }

      AGX_FORCE_INLINE GridTierPtr& GridTierPtr::operator= (const agxData::EntityPtr& ptr)
      {
        agxData::EntityPtr::operator= (ptr);
        agxAssertN(!ptr || ptr.isInstanceOf(GridTierModel::instance()),
          "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
          EntityPtr::getModel()->fullPath().c_str(), GridTierModel::instance()->fullPath().c_str());
        return *this;
      }

      AGX_FORCE_INLINE GridTierPtr& GridTierPtr::operator= (const agxData::EntityInstance& instance)
      {
        agxData::EntityPtr::operator= (instance);
        agxAssertN(!instance || instance.isInstanceOf(GridTierModel::instance()),
          "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
          EntityPtr::getModel()->fullPath().c_str(), GridTierModel::instance()->fullPath().c_str());
        return *this;
      }

      inline GridTierInstance GridTierPtr::instance() { return agxData::EntityPtr::instance(); }
      inline const GridTierInstance GridTierPtr::instance() const { return agxData::EntityPtr::instance(); }
      AGX_FORCE_INLINE GridTierSemantics* GridTierPtr::operator->() { return (GridTierSemantics* )this; }
      AGX_FORCE_INLINE const GridTierSemantics* GridTierPtr::operator->() const { return (const GridTierSemantics* )this; }
      AGX_FORCE_INLINE GridTierData* GridTierPtr::getData() { return static_cast<GridTierData* >(agxData::EntityPtr::getData()); }
      AGX_FORCE_INLINE const GridTierData* GridTierPtr::getData() const { return static_cast<const GridTierData* >(agxData::EntityPtr::getData()); }

      AGX_FORCE_INLINE agx::Real& GridTierPtr::size() { verifyIndex(); return getData()->size[calculateIndex()]; }
      AGX_FORCE_INLINE agx::Real const& GridTierPtr::size() const { verifyIndex(); return getData()->size[calculateIndex()]; }

      AGX_FORCE_INLINE agx::Real& GridTierPtr::invSize() { verifyIndex(); return getData()->invSize[calculateIndex()]; }
      AGX_FORCE_INLINE agx::Real const& GridTierPtr::invSize() const { verifyIndex(); return getData()->invSize[calculateIndex()]; }

      AGX_FORCE_INLINE agx::Real32& GridTierPtr::boundingRadius() { verifyIndex(); return getData()->boundingRadius[calculateIndex()]; }
      AGX_FORCE_INLINE agx::Real32 const& GridTierPtr::boundingRadius() const { verifyIndex(); return getData()->boundingRadius[calculateIndex()]; }

      AGX_FORCE_INLINE agx::GridCellTable*& GridTierPtr::cellTable() { verifyIndex(); return getData()->cellTable[calculateIndex()]; }
      AGX_FORCE_INLINE agx::GridCellTable* const& GridTierPtr::cellTable() const { verifyIndex(); return getData()->cellTable[calculateIndex()]; }

      AGX_FORCE_INLINE agx::ContactZoneTable*& GridTierPtr::zoneTable() { verifyIndex(); return getData()->zoneTable[calculateIndex()]; }
      AGX_FORCE_INLINE agx::ContactZoneTable* const& GridTierPtr::zoneTable() const { verifyIndex(); return getData()->zoneTable[calculateIndex()]; }

      //-----------------------------------------------------------------------------------------------------
      AGX_FORCE_INLINE GridTierInstance::GridTierInstance() {}
      AGX_FORCE_INLINE GridTierInstance::GridTierInstance(GridTierData* data, agx::Index index) : agxData::EntityInstance(data, index) {}
      AGX_FORCE_INLINE GridTierInstance::GridTierInstance(agxData::EntityStorage* storage, agx::Index index) : agxData::EntityInstance(storage, index) {}
      AGX_FORCE_INLINE GridTierInstance::GridTierInstance(const agxData::EntityInstance& other) : agxData::EntityInstance(other)
      {
        agxAssertN(!other || other.isInstanceOf(GridTierModel::instance()),
          "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
          EntityInstance::getModel()->fullPath().c_str(), GridTierModel::instance()->fullPath().c_str());
      }

      AGX_FORCE_INLINE GridTierInstance::GridTierInstance(const agxData::EntityPtr& ptr) : agxData::EntityInstance(ptr)
      {
        agxAssertN(!ptr || ptr.isInstanceOf(GridTierModel::instance()),
          "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
          EntityInstance::getModel()->fullPath().c_str(), GridTierModel::instance()->fullPath().c_str());
      }


      AGX_FORCE_INLINE GridTierData* GridTierInstance::getData() { return static_cast<GridTierData* >(agxData::EntityInstance::getData()); }
      AGX_FORCE_INLINE const GridTierData* GridTierInstance::getData() const { return static_cast<const GridTierData* >(agxData::EntityInstance::getData()); }

      AGX_FORCE_INLINE agx::Real& GridTierInstance::size() { verifyIndex(); return getData()->size[getIndex()]; }
      AGX_FORCE_INLINE agx::Real const& GridTierInstance::size() const { verifyIndex(); return getData()->size[getIndex()]; }

      AGX_FORCE_INLINE agx::Real& GridTierInstance::invSize() { verifyIndex(); return getData()->invSize[getIndex()]; }
      AGX_FORCE_INLINE agx::Real const& GridTierInstance::invSize() const { verifyIndex(); return getData()->invSize[getIndex()]; }

      AGX_FORCE_INLINE agx::Real32& GridTierInstance::boundingRadius() { verifyIndex(); return getData()->boundingRadius[getIndex()]; }
      AGX_FORCE_INLINE agx::Real32 const& GridTierInstance::boundingRadius() const { verifyIndex(); return getData()->boundingRadius[getIndex()]; }

      AGX_FORCE_INLINE agx::GridCellTable*& GridTierInstance::cellTable() { verifyIndex(); return getData()->cellTable[getIndex()]; }
      AGX_FORCE_INLINE agx::GridCellTable* const& GridTierInstance::cellTable() const { verifyIndex(); return getData()->cellTable[getIndex()]; }

      AGX_FORCE_INLINE agx::ContactZoneTable*& GridTierInstance::zoneTable() { verifyIndex(); return getData()->zoneTable[getIndex()]; }
      AGX_FORCE_INLINE agx::ContactZoneTable* const& GridTierInstance::zoneTable() const { verifyIndex(); return getData()->zoneTable[getIndex()]; }

      //-----------------------------------------------------------------------------------------------------
      AGX_FORCE_INLINE GridTierSemantics::GridTierSemantics() {}
      //-----------------------------------------------------------------------------------------------------
      DOXYGEN_END_INTERNAL_BLOCK()
    }
  }
}

AGX_TYPE_BINDING(agx::Physics::HierarchicalGrid::GridTierPtr, "Physics.HierarchicalGrid.GridTierPtr")
AGX_TYPE_BINDING(agx::Physics::HierarchicalGrid::GridTierInstance, "Physics.HierarchicalGrid.GridTierInstance")

#ifdef _MSC_VER
# pragma warning(pop)
#endif

#undef AGX_ENTITY_WRAPPER
#undef AGX_ENTITY_NAMESPACE
#endif

