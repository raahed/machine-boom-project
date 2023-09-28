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

#ifndef GENERATED_AGX_PHYSICS_HIERARCHICALGRID_ORIENTEDGEOMETRYBOUND_H_PLUGIN
#define GENERATED_AGX_PHYSICS_HIERARCHICALGRID_ORIENTEDGEOMETRYBOUND_H_PLUGIN

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
#include <agx/Physics/GeometryEntity.h>
#include <agx/Bound.h>
#include <agx/Vec3.h>
#include <agx/AffineMatrix4x4.h>

namespace agx { namespace Physics { class GeometryPtr; }}

namespace agx
{
  namespace Physics
  {
    namespace HierarchicalGrid
    {

      class OrientedGeometryBoundModel;
      class OrientedGeometryBoundData;
      class OrientedGeometryBoundPtr;
      class OrientedGeometryBoundInstance;
      class OrientedGeometryBoundSemantics;


      AGX_DECLARE_POINTER_TYPES(OrientedGeometryBoundModel);

      /** 
      Abstract description of the data attributes for the Physics.HierarchicalGrid.OrientedGeometryBound entity.
      */ 
      class AGXPHYSICS_EXPORT OrientedGeometryBoundModel : public agxData::EntityModel
      {
      public:
        typedef OrientedGeometryBoundPtr PtrT;

        OrientedGeometryBoundModel(const agx::String& name = "OrientedGeometryBound");

        /// \return The entity model singleton.
        static OrientedGeometryBoundModel* instance();

        /// Create and return a pointer to a new instance in the default storage for this entity model.
        static OrientedGeometryBoundPtr createInstance();

        /// \return The default storage for this entity model.
        static agxData::EntityStorage* defaultStorage();

        /// This is part of internal cleanup and should not be called by users
        virtual void shutdownCleanup() override;



        /* Attributes */
        static agxData::ScalarAttributeT< agx::Physics::GeometryPtr >* geometryAttribute;
        static agxData::ScalarAttributeT< agx::Bound3 >* localBoundAttribute;
        static agxData::ScalarAttributeT< agx::Vec3 >* halfExtentsAttribute;
        static agxData::ScalarAttributeT< agx::AffineMatrix4x4 >* invTransformAttribute;

      protected:
        virtual ~OrientedGeometryBoundModel();
        virtual agxData::EntityData* createData(agxData::EntityStorage* storage) override;
        virtual void configure(agx::TiXmlElement* eEntity) override;
        virtual void initAttributeAccessors() override;
        virtual void construct(agxData::EntityPtr instance) override;
        void construct(agx::Physics::HierarchicalGrid::OrientedGeometryBoundPtr orientedGeometryBound);
      };


      DOXYGEN_START_INTERNAL_BLOCK()
      #ifndef AGX_PHYSICS_HIERARCHICALGRID_ORIENTEDGEOMETRYBOUND_DATA_SET_OVERRIDE
      #define AGX_PHYSICS_HIERARCHICALGRID_ORIENTEDGEOMETRYBOUND_DATA_SET
      class AGXPHYSICS_EXPORT OrientedGeometryBoundData : public agxData::EntityData
      {
      public:
        OrientedGeometryBoundInstance operator[] (size_t index);

      public:
        agxData::Array< OrientedGeometryBoundPtr >& instance;
        agxData::Array< agx::Physics::GeometryPtr > geometry;
        agxData::Array< agx::Bound3 > localBound;
        agxData::Array< agx::Vec3 > halfExtents;
        agxData::Array< agx::AffineMatrix4x4 > invTransform;

      public:
        typedef agx::Physics::GeometryPtr geometryType;
        typedef agx::Bound3 localBoundType;
        typedef agx::Vec3 halfExtentsType;
        typedef agx::AffineMatrix4x4 invTransformType;

      public:
        OrientedGeometryBoundData(agxData::EntityStorage* storage);
        OrientedGeometryBoundData();

      protected:
        virtual ~OrientedGeometryBoundData() {}
        virtual void setNumElements(agx::Index numElements) override;

      private:
        OrientedGeometryBoundData& operator= (const OrientedGeometryBoundData&) { return *this; }

      };
      #endif
      DOXYGEN_END_INTERNAL_BLOCK()


      DOXYGEN_START_INTERNAL_BLOCK()
      class AGXPHYSICS_EXPORT OrientedGeometryBoundSemantics : protected agxData::EntityPtr
      {
      public:

        // Automatic getters
        agx::Physics::GeometryPtr const& getGeometry() const;
        agx::Bound3 const& getLocalBound() const;
        agx::Vec3 const& getHalfExtents() const;
        agx::AffineMatrix4x4 const& getInvTransform() const;

        // Semantics defined by explicit kernels

        // Automatic setters
        void setGeometry(agx::Physics::GeometryPtr const& value);
        void setLocalBound(agx::Bound3 const& value);
        void setHalfExtents(agx::Vec3 const& value);
        void setInvTransform(agx::AffineMatrix4x4 const& value);


      protected:
        friend class OrientedGeometryBoundPtr;
        friend class OrientedGeometryBoundInstance;
        OrientedGeometryBoundSemantics();
      };
      DOXYGEN_END_INTERNAL_BLOCK()


      /**
      Pointer to a entity instance of type Physics.HierarchicalGrid.OrientedGeometryBound
      */
      class CALLABLE OrientedGeometryBoundPtr : public agxData::EntityPtr
      {
      public:
        typedef OrientedGeometryBoundModel ModelType;
        typedef OrientedGeometryBoundData DataType;
        typedef OrientedGeometryBoundInstance InstanceType;

      public:
        AGXPHYSICS_EXPORT OrientedGeometryBoundPtr();
        AGXPHYSICS_EXPORT OrientedGeometryBoundPtr(agxData::EntityStorage* storage, agx::Index id);
        AGXPHYSICS_EXPORT OrientedGeometryBoundPtr(const agxData::EntityPtr& ptr);
        AGXPHYSICS_EXPORT OrientedGeometryBoundPtr(const agxData::EntityInstance& instance);
        AGXPHYSICS_EXPORT OrientedGeometryBoundPtr& operator= (const agxData::EntityPtr& ptr);
        AGXPHYSICS_EXPORT OrientedGeometryBoundPtr& operator= (const agxData::EntityInstance& instance);
        AGXPHYSICS_EXPORT OrientedGeometryBoundInstance instance();
        AGXPHYSICS_EXPORT const OrientedGeometryBoundInstance instance() const;

        AGXPHYSICS_EXPORT OrientedGeometryBoundSemantics* operator->();
        AGXPHYSICS_EXPORT const OrientedGeometryBoundSemantics* operator->() const;

        OrientedGeometryBoundData* getData();
        const OrientedGeometryBoundData* getData() const;


        /// \return reference to the geometry attribute
        AGXPHYSICS_EXPORT agx::Physics::GeometryPtr& geometry();
        /// \return const reference to the geometry attribute
        AGXPHYSICS_EXPORT agx::Physics::GeometryPtr const& geometry() const;

        /// \return reference to the localBound attribute
        AGXPHYSICS_EXPORT agx::Bound3& localBound();
        /// \return const reference to the localBound attribute
        AGXPHYSICS_EXPORT agx::Bound3 const& localBound() const;

        /// \return reference to the halfExtents attribute
        AGXPHYSICS_EXPORT agx::Vec3& halfExtents();
        /// \return const reference to the halfExtents attribute
        AGXPHYSICS_EXPORT agx::Vec3 const& halfExtents() const;

        /// \return reference to the invTransform attribute
        AGXPHYSICS_EXPORT agx::AffineMatrix4x4& invTransform();
        /// \return const reference to the invTransform attribute
        AGXPHYSICS_EXPORT agx::AffineMatrix4x4 const& invTransform() const;

        DOXYGEN_START_INTERNAL_BLOCK()
        // Only used by tolua!
        #ifdef AGX_GENERATED_TOLUA_WRAPPER
        inline agx::Physics::GeometryPtr const& getGeometry() const { return (*this)->getGeometry(); }
        inline agx::Bound3 const& getLocalBound() const { return (*this)->getLocalBound(); }
        inline agx::Vec3 const& getHalfExtents() const { return (*this)->getHalfExtents(); }
        inline agx::AffineMatrix4x4 const& getInvTransform() const { return (*this)->getInvTransform(); }


        inline void setGeometry(agx::Physics::GeometryPtr const& value) { (*this)->setGeometry(value); }
        inline void setLocalBound(agx::Bound3 const& value) { (*this)->setLocalBound(value); }
        inline void setHalfExtents(agx::Vec3 const& value) { (*this)->setHalfExtents(value); }
        inline void setInvTransform(agx::AffineMatrix4x4 const& value) { (*this)->setInvTransform(value); }
        #endif
        DOXYGEN_END_INTERNAL_BLOCK()

      };


      DOXYGEN_START_INTERNAL_BLOCK()
      class AGXPHYSICS_EXPORT OrientedGeometryBoundInstance : public agxData::EntityInstance
      {
      public:
        OrientedGeometryBoundInstance();
        OrientedGeometryBoundInstance(OrientedGeometryBoundData* data, agx::Index index);
        OrientedGeometryBoundInstance(agxData::EntityStorage *storage, agx::Index index);
        OrientedGeometryBoundInstance(const agxData::EntityInstance& other);
        OrientedGeometryBoundInstance(const agxData::EntityPtr& ptr);

        OrientedGeometryBoundData* getData();
        const OrientedGeometryBoundData* getData() const;

      public:
        /// \return reference to the geometry attribute
        agx::Physics::GeometryPtr& geometry();
        /// \return const reference to the geometry attribute
        agx::Physics::GeometryPtr const& geometry() const;

        /// \return reference to the localBound attribute
        agx::Bound3& localBound();
        /// \return const reference to the localBound attribute
        agx::Bound3 const& localBound() const;

        /// \return reference to the halfExtents attribute
        agx::Vec3& halfExtents();
        /// \return const reference to the halfExtents attribute
        agx::Vec3 const& halfExtents() const;

        /// \return reference to the invTransform attribute
        agx::AffineMatrix4x4& invTransform();
        /// \return const reference to the invTransform attribute
        agx::AffineMatrix4x4 const& invTransform() const;

      };
      DOXYGEN_END_INTERNAL_BLOCK()



      typedef agx::VectorPOD<OrientedGeometryBoundPtr> OrientedGeometryBoundPtrVector;
      typedef agxData::Array<OrientedGeometryBoundPtr> OrientedGeometryBoundPtrArray;



      DOXYGEN_START_INTERNAL_BLOCK()
      /* Implementation */
      //-----------------------------------------------------------------------------------------------------
      //-----------------------------------------------------------------------------------------------------
      inline OrientedGeometryBoundInstance agx::Physics::HierarchicalGrid::OrientedGeometryBoundData::operator[] (size_t index) { return OrientedGeometryBoundInstance(this, (agx::Index)index); }
      //-----------------------------------------------------------------------------------------------------
      AGX_FORCE_INLINE OrientedGeometryBoundPtr::OrientedGeometryBoundPtr() {}
      AGX_FORCE_INLINE OrientedGeometryBoundPtr::OrientedGeometryBoundPtr(agxData::EntityStorage* storage, agx::Index id) : agxData::EntityPtr(storage, id) {}
      AGX_FORCE_INLINE OrientedGeometryBoundPtr::OrientedGeometryBoundPtr(const agxData::EntityPtr& ptr) : agxData::EntityPtr(ptr)
      {
        agxAssertN(!ptr || ptr.isInstanceOf(OrientedGeometryBoundModel::instance()),
          "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
          EntityPtr::getModel()->fullPath().c_str(), OrientedGeometryBoundModel::instance()->fullPath().c_str());
      }

      AGX_FORCE_INLINE OrientedGeometryBoundPtr::OrientedGeometryBoundPtr(const agxData::EntityInstance& instance) : agxData::EntityPtr(instance)
      {
        agxAssertN(!instance || instance.isInstanceOf(OrientedGeometryBoundModel::instance()),
          "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
          EntityPtr::getModel()->fullPath().c_str(), OrientedGeometryBoundModel::instance()->fullPath().c_str());
      }

      AGX_FORCE_INLINE OrientedGeometryBoundPtr& OrientedGeometryBoundPtr::operator= (const agxData::EntityPtr& ptr)
      {
        agxData::EntityPtr::operator= (ptr);
        agxAssertN(!ptr || ptr.isInstanceOf(OrientedGeometryBoundModel::instance()),
          "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
          EntityPtr::getModel()->fullPath().c_str(), OrientedGeometryBoundModel::instance()->fullPath().c_str());
        return *this;
      }

      AGX_FORCE_INLINE OrientedGeometryBoundPtr& OrientedGeometryBoundPtr::operator= (const agxData::EntityInstance& instance)
      {
        agxData::EntityPtr::operator= (instance);
        agxAssertN(!instance || instance.isInstanceOf(OrientedGeometryBoundModel::instance()),
          "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
          EntityPtr::getModel()->fullPath().c_str(), OrientedGeometryBoundModel::instance()->fullPath().c_str());
        return *this;
      }

      inline OrientedGeometryBoundInstance OrientedGeometryBoundPtr::instance() { return agxData::EntityPtr::instance(); }
      inline const OrientedGeometryBoundInstance OrientedGeometryBoundPtr::instance() const { return agxData::EntityPtr::instance(); }
      AGX_FORCE_INLINE OrientedGeometryBoundSemantics* OrientedGeometryBoundPtr::operator->() { return (OrientedGeometryBoundSemantics* )this; }
      AGX_FORCE_INLINE const OrientedGeometryBoundSemantics* OrientedGeometryBoundPtr::operator->() const { return (const OrientedGeometryBoundSemantics* )this; }
      AGX_FORCE_INLINE OrientedGeometryBoundData* OrientedGeometryBoundPtr::getData() { return static_cast<OrientedGeometryBoundData* >(agxData::EntityPtr::getData()); }
      AGX_FORCE_INLINE const OrientedGeometryBoundData* OrientedGeometryBoundPtr::getData() const { return static_cast<const OrientedGeometryBoundData* >(agxData::EntityPtr::getData()); }

      AGX_FORCE_INLINE agx::Physics::GeometryPtr& OrientedGeometryBoundPtr::geometry() { verifyIndex(); return getData()->geometry[calculateIndex()]; }
      AGX_FORCE_INLINE agx::Physics::GeometryPtr const& OrientedGeometryBoundPtr::geometry() const { verifyIndex(); return getData()->geometry[calculateIndex()]; }

      AGX_FORCE_INLINE agx::Bound3& OrientedGeometryBoundPtr::localBound() { verifyIndex(); return getData()->localBound[calculateIndex()]; }
      AGX_FORCE_INLINE agx::Bound3 const& OrientedGeometryBoundPtr::localBound() const { verifyIndex(); return getData()->localBound[calculateIndex()]; }

      AGX_FORCE_INLINE agx::Vec3& OrientedGeometryBoundPtr::halfExtents() { verifyIndex(); return getData()->halfExtents[calculateIndex()]; }
      AGX_FORCE_INLINE agx::Vec3 const& OrientedGeometryBoundPtr::halfExtents() const { verifyIndex(); return getData()->halfExtents[calculateIndex()]; }

      AGX_FORCE_INLINE agx::AffineMatrix4x4& OrientedGeometryBoundPtr::invTransform() { verifyIndex(); return getData()->invTransform[calculateIndex()]; }
      AGX_FORCE_INLINE agx::AffineMatrix4x4 const& OrientedGeometryBoundPtr::invTransform() const { verifyIndex(); return getData()->invTransform[calculateIndex()]; }

      //-----------------------------------------------------------------------------------------------------
      AGX_FORCE_INLINE OrientedGeometryBoundInstance::OrientedGeometryBoundInstance() {}
      AGX_FORCE_INLINE OrientedGeometryBoundInstance::OrientedGeometryBoundInstance(OrientedGeometryBoundData* data, agx::Index index) : agxData::EntityInstance(data, index) {}
      AGX_FORCE_INLINE OrientedGeometryBoundInstance::OrientedGeometryBoundInstance(agxData::EntityStorage* storage, agx::Index index) : agxData::EntityInstance(storage, index) {}
      AGX_FORCE_INLINE OrientedGeometryBoundInstance::OrientedGeometryBoundInstance(const agxData::EntityInstance& other) : agxData::EntityInstance(other)
      {
        agxAssertN(!other || other.isInstanceOf(OrientedGeometryBoundModel::instance()),
          "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
          EntityInstance::getModel()->fullPath().c_str(), OrientedGeometryBoundModel::instance()->fullPath().c_str());
      }

      AGX_FORCE_INLINE OrientedGeometryBoundInstance::OrientedGeometryBoundInstance(const agxData::EntityPtr& ptr) : agxData::EntityInstance(ptr)
      {
        agxAssertN(!ptr || ptr.isInstanceOf(OrientedGeometryBoundModel::instance()),
          "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
          EntityInstance::getModel()->fullPath().c_str(), OrientedGeometryBoundModel::instance()->fullPath().c_str());
      }


      AGX_FORCE_INLINE OrientedGeometryBoundData* OrientedGeometryBoundInstance::getData() { return static_cast<OrientedGeometryBoundData* >(agxData::EntityInstance::getData()); }
      AGX_FORCE_INLINE const OrientedGeometryBoundData* OrientedGeometryBoundInstance::getData() const { return static_cast<const OrientedGeometryBoundData* >(agxData::EntityInstance::getData()); }

      AGX_FORCE_INLINE agx::Physics::GeometryPtr& OrientedGeometryBoundInstance::geometry() { verifyIndex(); return getData()->geometry[getIndex()]; }
      AGX_FORCE_INLINE agx::Physics::GeometryPtr const& OrientedGeometryBoundInstance::geometry() const { verifyIndex(); return getData()->geometry[getIndex()]; }

      AGX_FORCE_INLINE agx::Bound3& OrientedGeometryBoundInstance::localBound() { verifyIndex(); return getData()->localBound[getIndex()]; }
      AGX_FORCE_INLINE agx::Bound3 const& OrientedGeometryBoundInstance::localBound() const { verifyIndex(); return getData()->localBound[getIndex()]; }

      AGX_FORCE_INLINE agx::Vec3& OrientedGeometryBoundInstance::halfExtents() { verifyIndex(); return getData()->halfExtents[getIndex()]; }
      AGX_FORCE_INLINE agx::Vec3 const& OrientedGeometryBoundInstance::halfExtents() const { verifyIndex(); return getData()->halfExtents[getIndex()]; }

      AGX_FORCE_INLINE agx::AffineMatrix4x4& OrientedGeometryBoundInstance::invTransform() { verifyIndex(); return getData()->invTransform[getIndex()]; }
      AGX_FORCE_INLINE agx::AffineMatrix4x4 const& OrientedGeometryBoundInstance::invTransform() const { verifyIndex(); return getData()->invTransform[getIndex()]; }

      //-----------------------------------------------------------------------------------------------------
      AGX_FORCE_INLINE OrientedGeometryBoundSemantics::OrientedGeometryBoundSemantics() {}
      //-----------------------------------------------------------------------------------------------------
      DOXYGEN_END_INTERNAL_BLOCK()
    }
  }
}

AGX_TYPE_BINDING(agx::Physics::HierarchicalGrid::OrientedGeometryBoundPtr, "Physics.HierarchicalGrid.OrientedGeometryBoundPtr")
AGX_TYPE_BINDING(agx::Physics::HierarchicalGrid::OrientedGeometryBoundInstance, "Physics.HierarchicalGrid.OrientedGeometryBoundInstance")

#ifdef _MSC_VER
# pragma warning(pop)
#endif

#undef AGX_ENTITY_WRAPPER
#undef AGX_ENTITY_NAMESPACE
#endif

