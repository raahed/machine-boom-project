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

#ifndef GENERATED_AGX_PHYSICS_HIERARCHICALGRID_GRIDTIER2D_H_PLUGIN
#define GENERATED_AGX_PHYSICS_HIERARCHICALGRID_GRIDTIER2D_H_PLUGIN

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
#include <agx/Physics/HierarchicalGrid/Common.h>


namespace agx
{
  namespace Physics
  {
    namespace HierarchicalGrid
    {

      class GridTier2DModel;
      class GridTier2DData;
      class GridTier2DPtr;
      class GridTier2DInstance;
      class GridTier2DSemantics;


      AGX_DECLARE_POINTER_TYPES(GridTier2DModel);

      /** 
      Abstract description of the data attributes for the Physics.HierarchicalGrid.GridTier2D entity.
      */ 
      class AGXPHYSICS_EXPORT GridTier2DModel : public agxData::EntityModel
      {
      public:
        typedef GridTier2DPtr PtrT;

        GridTier2DModel(const agx::String& name = "GridTier2D");

        /// \return The entity model singleton.
        static GridTier2DModel* instance();

        /// Create and return a pointer to a new instance in the default storage for this entity model.
        static GridTier2DPtr createInstance();

        /// \return The default storage for this entity model.
        static agxData::EntityStorage* defaultStorage();

        /// This is part of internal cleanup and should not be called by users
        virtual void shutdownCleanup() override;



        /* Attributes */
        static agxData::ScalarAttributeT< agx::Real >* sizeAttribute;
        static agxData::ScalarAttributeT< agx::Real >* invSizeAttribute;
        static agxData::ScalarAttributeT< agx::Real >* boundingRadiusAttribute;
        static agxData::PointerAttributeT< agx::GridCell2DTable*>* cellTableAttribute;

      protected:
        virtual ~GridTier2DModel();
        virtual agxData::EntityData* createData(agxData::EntityStorage* storage) override;
        virtual void configure(agx::TiXmlElement* eEntity) override;
        virtual void initAttributeAccessors() override;
        virtual void construct(agxData::EntityPtr instance) override;
        void construct(agx::Physics::HierarchicalGrid::GridTier2DPtr gridTier2D);
      };


      DOXYGEN_START_INTERNAL_BLOCK()
      #ifndef AGX_PHYSICS_HIERARCHICALGRID_GRIDTIER2D_DATA_SET_OVERRIDE
      #define AGX_PHYSICS_HIERARCHICALGRID_GRIDTIER2D_DATA_SET
      class AGXPHYSICS_EXPORT GridTier2DData : public agxData::EntityData
      {
      public:
        GridTier2DInstance operator[] (size_t index);

      public:
        agxData::Array< GridTier2DPtr >& instance;
        agxData::Array< agx::Real > size;
        agxData::Array< agx::Real > invSize;
        agxData::Array< agx::Real > boundingRadius;
        agxData::Array< agx::GridCell2DTable* > cellTable;

      public:
        typedef agx::Real sizeType;
        typedef agx::Real invSizeType;
        typedef agx::Real boundingRadiusType;
        typedef agx::GridCell2DTable* cellTableType;

      public:
        GridTier2DData(agxData::EntityStorage* storage);
        GridTier2DData();

      protected:
        virtual ~GridTier2DData() {}
        virtual void setNumElements(agx::Index numElements) override;

      private:
        GridTier2DData& operator= (const GridTier2DData&) { return *this; }

      };
      #endif
      DOXYGEN_END_INTERNAL_BLOCK()


      DOXYGEN_START_INTERNAL_BLOCK()
      class AGXPHYSICS_EXPORT GridTier2DSemantics : protected agxData::EntityPtr
      {
      public:

        // Automatic getters
        agx::Real const& getSize() const;
        agx::Real const& getInvSize() const;
        agx::Real const& getBoundingRadius() const;
        agx::GridCell2DTable* const& getCellTable() const;

        // Semantics defined by explicit kernels

        // Automatic setters
        void setSize(agx::Real const& value);
        void setInvSize(agx::Real const& value);
        void setBoundingRadius(agx::Real const& value);
        void setCellTable(agx::GridCell2DTable* const& value);


      protected:
        friend class GridTier2DPtr;
        friend class GridTier2DInstance;
        GridTier2DSemantics();
      };
      DOXYGEN_END_INTERNAL_BLOCK()


      /**
      Pointer to a entity instance of type Physics.HierarchicalGrid.GridTier2D
      */
      class CALLABLE GridTier2DPtr : public agxData::EntityPtr
      {
      public:
        typedef GridTier2DModel ModelType;
        typedef GridTier2DData DataType;
        typedef GridTier2DInstance InstanceType;

      public:
        AGXPHYSICS_EXPORT GridTier2DPtr();
        AGXPHYSICS_EXPORT GridTier2DPtr(agxData::EntityStorage* storage, agx::Index id);
        AGXPHYSICS_EXPORT GridTier2DPtr(const agxData::EntityPtr& ptr);
        AGXPHYSICS_EXPORT GridTier2DPtr(const agxData::EntityInstance& instance);
        AGXPHYSICS_EXPORT GridTier2DPtr& operator= (const agxData::EntityPtr& ptr);
        AGXPHYSICS_EXPORT GridTier2DPtr& operator= (const agxData::EntityInstance& instance);
        AGXPHYSICS_EXPORT GridTier2DInstance instance();
        AGXPHYSICS_EXPORT const GridTier2DInstance instance() const;

        AGXPHYSICS_EXPORT GridTier2DSemantics* operator->();
        AGXPHYSICS_EXPORT const GridTier2DSemantics* operator->() const;

        GridTier2DData* getData();
        const GridTier2DData* getData() const;


        /// \return reference to the size attribute
        AGXPHYSICS_EXPORT agx::Real& size();
        /// \return const reference to the size attribute
        AGXPHYSICS_EXPORT agx::Real const& size() const;

        /// \return reference to the invSize attribute
        AGXPHYSICS_EXPORT agx::Real& invSize();
        /// \return const reference to the invSize attribute
        AGXPHYSICS_EXPORT agx::Real const& invSize() const;

        /// \return reference to the boundingRadius attribute
        AGXPHYSICS_EXPORT agx::Real& boundingRadius();
        /// \return const reference to the boundingRadius attribute
        AGXPHYSICS_EXPORT agx::Real const& boundingRadius() const;

        /// \return reference to the cellTable attribute
        AGXPHYSICS_EXPORT agx::GridCell2DTable*& cellTable();
        /// \return const reference to the cellTable attribute
        AGXPHYSICS_EXPORT agx::GridCell2DTable* const& cellTable() const;

        DOXYGEN_START_INTERNAL_BLOCK()
        // Only used by tolua!
        #ifdef AGX_GENERATED_TOLUA_WRAPPER
        inline agx::Real const& getSize() const { return (*this)->getSize(); }
        inline agx::Real const& getInvSize() const { return (*this)->getInvSize(); }
        inline agx::Real const& getBoundingRadius() const { return (*this)->getBoundingRadius(); }
        inline agx::GridCell2DTable* const& getCellTable() const { return (*this)->getCellTable(); }


        inline void setSize(agx::Real const& value) { (*this)->setSize(value); }
        inline void setInvSize(agx::Real const& value) { (*this)->setInvSize(value); }
        inline void setBoundingRadius(agx::Real const& value) { (*this)->setBoundingRadius(value); }
        inline void setCellTable(agx::GridCell2DTable* const& value) { (*this)->setCellTable(value); }
        #endif
        DOXYGEN_END_INTERNAL_BLOCK()

      };


      DOXYGEN_START_INTERNAL_BLOCK()
      class AGXPHYSICS_EXPORT GridTier2DInstance : public agxData::EntityInstance
      {
      public:
        GridTier2DInstance();
        GridTier2DInstance(GridTier2DData* data, agx::Index index);
        GridTier2DInstance(agxData::EntityStorage *storage, agx::Index index);
        GridTier2DInstance(const agxData::EntityInstance& other);
        GridTier2DInstance(const agxData::EntityPtr& ptr);

        GridTier2DData* getData();
        const GridTier2DData* getData() const;

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
        agx::Real& boundingRadius();
        /// \return const reference to the boundingRadius attribute
        agx::Real const& boundingRadius() const;

        /// \return reference to the cellTable attribute
        agx::GridCell2DTable*& cellTable();
        /// \return const reference to the cellTable attribute
        agx::GridCell2DTable* const& cellTable() const;

      };
      DOXYGEN_END_INTERNAL_BLOCK()



      typedef agx::VectorPOD<GridTier2DPtr> GridTier2DPtrVector;
      typedef agxData::Array<GridTier2DPtr> GridTier2DPtrArray;



      DOXYGEN_START_INTERNAL_BLOCK()
      /* Implementation */
      //-----------------------------------------------------------------------------------------------------
      //-----------------------------------------------------------------------------------------------------
      inline GridTier2DInstance agx::Physics::HierarchicalGrid::GridTier2DData::operator[] (size_t index) { return GridTier2DInstance(this, (agx::Index)index); }
      //-----------------------------------------------------------------------------------------------------
      AGX_FORCE_INLINE GridTier2DPtr::GridTier2DPtr() {}
      AGX_FORCE_INLINE GridTier2DPtr::GridTier2DPtr(agxData::EntityStorage* storage, agx::Index id) : agxData::EntityPtr(storage, id) {}
      AGX_FORCE_INLINE GridTier2DPtr::GridTier2DPtr(const agxData::EntityPtr& ptr) : agxData::EntityPtr(ptr)
      {
        agxAssertN(!ptr || ptr.isInstanceOf(GridTier2DModel::instance()),
          "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
          EntityPtr::getModel()->fullPath().c_str(), GridTier2DModel::instance()->fullPath().c_str());
      }

      AGX_FORCE_INLINE GridTier2DPtr::GridTier2DPtr(const agxData::EntityInstance& instance) : agxData::EntityPtr(instance)
      {
        agxAssertN(!instance || instance.isInstanceOf(GridTier2DModel::instance()),
          "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
          EntityPtr::getModel()->fullPath().c_str(), GridTier2DModel::instance()->fullPath().c_str());
      }

      AGX_FORCE_INLINE GridTier2DPtr& GridTier2DPtr::operator= (const agxData::EntityPtr& ptr)
      {
        agxData::EntityPtr::operator= (ptr);
        agxAssertN(!ptr || ptr.isInstanceOf(GridTier2DModel::instance()),
          "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
          EntityPtr::getModel()->fullPath().c_str(), GridTier2DModel::instance()->fullPath().c_str());
        return *this;
      }

      AGX_FORCE_INLINE GridTier2DPtr& GridTier2DPtr::operator= (const agxData::EntityInstance& instance)
      {
        agxData::EntityPtr::operator= (instance);
        agxAssertN(!instance || instance.isInstanceOf(GridTier2DModel::instance()),
          "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
          EntityPtr::getModel()->fullPath().c_str(), GridTier2DModel::instance()->fullPath().c_str());
        return *this;
      }

      inline GridTier2DInstance GridTier2DPtr::instance() { return agxData::EntityPtr::instance(); }
      inline const GridTier2DInstance GridTier2DPtr::instance() const { return agxData::EntityPtr::instance(); }
      AGX_FORCE_INLINE GridTier2DSemantics* GridTier2DPtr::operator->() { return (GridTier2DSemantics* )this; }
      AGX_FORCE_INLINE const GridTier2DSemantics* GridTier2DPtr::operator->() const { return (const GridTier2DSemantics* )this; }
      AGX_FORCE_INLINE GridTier2DData* GridTier2DPtr::getData() { return static_cast<GridTier2DData* >(agxData::EntityPtr::getData()); }
      AGX_FORCE_INLINE const GridTier2DData* GridTier2DPtr::getData() const { return static_cast<const GridTier2DData* >(agxData::EntityPtr::getData()); }

      AGX_FORCE_INLINE agx::Real& GridTier2DPtr::size() { verifyIndex(); return getData()->size[calculateIndex()]; }
      AGX_FORCE_INLINE agx::Real const& GridTier2DPtr::size() const { verifyIndex(); return getData()->size[calculateIndex()]; }

      AGX_FORCE_INLINE agx::Real& GridTier2DPtr::invSize() { verifyIndex(); return getData()->invSize[calculateIndex()]; }
      AGX_FORCE_INLINE agx::Real const& GridTier2DPtr::invSize() const { verifyIndex(); return getData()->invSize[calculateIndex()]; }

      AGX_FORCE_INLINE agx::Real& GridTier2DPtr::boundingRadius() { verifyIndex(); return getData()->boundingRadius[calculateIndex()]; }
      AGX_FORCE_INLINE agx::Real const& GridTier2DPtr::boundingRadius() const { verifyIndex(); return getData()->boundingRadius[calculateIndex()]; }

      AGX_FORCE_INLINE agx::GridCell2DTable*& GridTier2DPtr::cellTable() { verifyIndex(); return getData()->cellTable[calculateIndex()]; }
      AGX_FORCE_INLINE agx::GridCell2DTable* const& GridTier2DPtr::cellTable() const { verifyIndex(); return getData()->cellTable[calculateIndex()]; }

      //-----------------------------------------------------------------------------------------------------
      AGX_FORCE_INLINE GridTier2DInstance::GridTier2DInstance() {}
      AGX_FORCE_INLINE GridTier2DInstance::GridTier2DInstance(GridTier2DData* data, agx::Index index) : agxData::EntityInstance(data, index) {}
      AGX_FORCE_INLINE GridTier2DInstance::GridTier2DInstance(agxData::EntityStorage* storage, agx::Index index) : agxData::EntityInstance(storage, index) {}
      AGX_FORCE_INLINE GridTier2DInstance::GridTier2DInstance(const agxData::EntityInstance& other) : agxData::EntityInstance(other)
      {
        agxAssertN(!other || other.isInstanceOf(GridTier2DModel::instance()),
          "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
          EntityInstance::getModel()->fullPath().c_str(), GridTier2DModel::instance()->fullPath().c_str());
      }

      AGX_FORCE_INLINE GridTier2DInstance::GridTier2DInstance(const agxData::EntityPtr& ptr) : agxData::EntityInstance(ptr)
      {
        agxAssertN(!ptr || ptr.isInstanceOf(GridTier2DModel::instance()),
          "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
          EntityInstance::getModel()->fullPath().c_str(), GridTier2DModel::instance()->fullPath().c_str());
      }


      AGX_FORCE_INLINE GridTier2DData* GridTier2DInstance::getData() { return static_cast<GridTier2DData* >(agxData::EntityInstance::getData()); }
      AGX_FORCE_INLINE const GridTier2DData* GridTier2DInstance::getData() const { return static_cast<const GridTier2DData* >(agxData::EntityInstance::getData()); }

      AGX_FORCE_INLINE agx::Real& GridTier2DInstance::size() { verifyIndex(); return getData()->size[getIndex()]; }
      AGX_FORCE_INLINE agx::Real const& GridTier2DInstance::size() const { verifyIndex(); return getData()->size[getIndex()]; }

      AGX_FORCE_INLINE agx::Real& GridTier2DInstance::invSize() { verifyIndex(); return getData()->invSize[getIndex()]; }
      AGX_FORCE_INLINE agx::Real const& GridTier2DInstance::invSize() const { verifyIndex(); return getData()->invSize[getIndex()]; }

      AGX_FORCE_INLINE agx::Real& GridTier2DInstance::boundingRadius() { verifyIndex(); return getData()->boundingRadius[getIndex()]; }
      AGX_FORCE_INLINE agx::Real const& GridTier2DInstance::boundingRadius() const { verifyIndex(); return getData()->boundingRadius[getIndex()]; }

      AGX_FORCE_INLINE agx::GridCell2DTable*& GridTier2DInstance::cellTable() { verifyIndex(); return getData()->cellTable[getIndex()]; }
      AGX_FORCE_INLINE agx::GridCell2DTable* const& GridTier2DInstance::cellTable() const { verifyIndex(); return getData()->cellTable[getIndex()]; }

      //-----------------------------------------------------------------------------------------------------
      AGX_FORCE_INLINE GridTier2DSemantics::GridTier2DSemantics() {}
      //-----------------------------------------------------------------------------------------------------
      DOXYGEN_END_INTERNAL_BLOCK()
    }
  }
}

AGX_TYPE_BINDING(agx::Physics::HierarchicalGrid::GridTier2DPtr, "Physics.HierarchicalGrid.GridTier2DPtr")
AGX_TYPE_BINDING(agx::Physics::HierarchicalGrid::GridTier2DInstance, "Physics.HierarchicalGrid.GridTier2DInstance")

#ifdef _MSC_VER
# pragma warning(pop)
#endif

#undef AGX_ENTITY_WRAPPER
#undef AGX_ENTITY_NAMESPACE
#endif

