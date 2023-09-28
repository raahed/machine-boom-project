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

#ifndef GENERATED_AGX_PHYSICS_GEOMETRY_CAPSULE_H_PLUGIN
#define GENERATED_AGX_PHYSICS_GEOMETRY_CAPSULE_H_PLUGIN

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
#include <agx/Physics/Geometry/ShapeEntity.h>
#include <agx/Real.h>


namespace agx
{
  namespace Physics
  {
    namespace Geometry
    {

      class CapsuleModel;
      class CapsuleData;
      class CapsulePtr;
      class CapsuleInstance;
      class CapsuleSemantics;


      AGX_DECLARE_POINTER_TYPES(CapsuleModel);

      /** 
      Abstract description of the data attributes for the Physics.Geometry.Capsule entity.
      */ 
      class AGXPHYSICS_EXPORT CapsuleModel : public agx::Physics::Geometry::ShapeModel
      {
      public:
        typedef CapsulePtr PtrT;

        CapsuleModel(const agx::String& name = "Capsule");

        /// \return The entity model singleton.
        static CapsuleModel* instance();

        /// Create and return a pointer to a new instance in the default storage for this entity model.
        static CapsulePtr createInstance();

        /// \return The default storage for this entity model.
        static agxData::EntityStorage* defaultStorage();

        /// This is part of internal cleanup and should not be called by users
        virtual void shutdownCleanup() override;



        /* Attributes */
        static agxData::ScalarAttributeT< agx::Real >* heightAttribute;
        static agxData::ScalarAttributeT< agx::Real >* radiusAttribute;

      protected:
        virtual ~CapsuleModel();
        virtual agxData::EntityData* createData(agxData::EntityStorage* storage) override;
        virtual void configure(agx::TiXmlElement* eEntity) override;
        virtual void initAttributeAccessors() override;
        virtual void construct(agxData::EntityPtr instance) override;
        void construct(agx::Physics::Geometry::CapsulePtr capsule);
      };


      DOXYGEN_START_INTERNAL_BLOCK()
      #ifndef AGX_PHYSICS_GEOMETRY_CAPSULE_DATA_SET_OVERRIDE
      #define AGX_PHYSICS_GEOMETRY_CAPSULE_DATA_SET
      class AGXPHYSICS_EXPORT CapsuleData : public agx::Physics::Geometry::ShapeData
      {
      public:
        CapsuleInstance operator[] (size_t index);

      public:
        agxData::Array< CapsulePtr >& instance;
        agxData::Array< agx::Real > height;
        agxData::Array< agx::Real > radius;

      public:
        typedef agx::Real heightType;
        typedef agx::Real radiusType;

      public:
        CapsuleData(agxData::EntityStorage* storage);
        CapsuleData();

      protected:
        virtual ~CapsuleData() {}
        virtual void setNumElements(agx::Index numElements) override;

      private:
        CapsuleData& operator= (const CapsuleData&) { return *this; }

      };
      #endif
      DOXYGEN_END_INTERNAL_BLOCK()


      DOXYGEN_START_INTERNAL_BLOCK()
      class AGXPHYSICS_EXPORT CapsuleSemantics : public agx::Physics::Geometry::ShapeSemantics
      {
      public:

        // Automatic getters
        agx::Real const& getHeight() const;
        agx::Real const& getRadius() const;

        // Semantics defined by explicit kernels

        // Automatic setters
        void setHeight(agx::Real const& value);
        void setRadius(agx::Real const& value);


      protected:
        friend class CapsulePtr;
        friend class CapsuleInstance;
        CapsuleSemantics();
      };
      DOXYGEN_END_INTERNAL_BLOCK()


      /**
      Pointer to a entity instance of type Physics.Geometry.Capsule
      */
      class CALLABLE CapsulePtr : public agx::Physics::Geometry::ShapePtr
      {
      public:
        typedef CapsuleModel ModelType;
        typedef CapsuleData DataType;
        typedef CapsuleInstance InstanceType;

      public:
        AGXPHYSICS_EXPORT CapsulePtr();
        AGXPHYSICS_EXPORT CapsulePtr(agxData::EntityStorage* storage, agx::Index id);
        AGXPHYSICS_EXPORT CapsulePtr(const agxData::EntityPtr& ptr);
        AGXPHYSICS_EXPORT CapsulePtr(const agxData::EntityInstance& instance);
        AGXPHYSICS_EXPORT CapsulePtr& operator= (const agxData::EntityPtr& ptr);
        AGXPHYSICS_EXPORT CapsulePtr& operator= (const agxData::EntityInstance& instance);
        AGXPHYSICS_EXPORT CapsuleInstance instance();
        AGXPHYSICS_EXPORT const CapsuleInstance instance() const;

        AGXPHYSICS_EXPORT CapsuleSemantics* operator->();
        AGXPHYSICS_EXPORT const CapsuleSemantics* operator->() const;

        CapsuleData* getData();
        const CapsuleData* getData() const;


        /// \return reference to the height attribute
        AGXPHYSICS_EXPORT agx::Real& height();
        /// \return const reference to the height attribute
        AGXPHYSICS_EXPORT agx::Real const& height() const;

        /// \return reference to the radius attribute
        AGXPHYSICS_EXPORT agx::Real& radius();
        /// \return const reference to the radius attribute
        AGXPHYSICS_EXPORT agx::Real const& radius() const;

        DOXYGEN_START_INTERNAL_BLOCK()
        // Only used by tolua!
        #ifdef AGX_GENERATED_TOLUA_WRAPPER
        inline agx::Real const& getHeight() const { return (*this)->getHeight(); }
        inline agx::Real const& getRadius() const { return (*this)->getRadius(); }


        inline void setHeight(agx::Real const& value) { (*this)->setHeight(value); }
        inline void setRadius(agx::Real const& value) { (*this)->setRadius(value); }
        #endif
        DOXYGEN_END_INTERNAL_BLOCK()

      };

      // Entity is Referenced
      typedef agxData::EntityRef< CapsulePtr > CapsuleRef;


      DOXYGEN_START_INTERNAL_BLOCK()
      class AGXPHYSICS_EXPORT CapsuleInstance : public agx::Physics::Geometry::ShapeInstance
      {
      public:
        CapsuleInstance();
        CapsuleInstance(CapsuleData* data, agx::Index index);
        CapsuleInstance(agxData::EntityStorage *storage, agx::Index index);
        CapsuleInstance(const agxData::EntityInstance& other);
        CapsuleInstance(const agxData::EntityPtr& ptr);

        CapsuleData* getData();
        const CapsuleData* getData() const;

      public:
        /// \return reference to the height attribute
        agx::Real& height();
        /// \return const reference to the height attribute
        agx::Real const& height() const;

        /// \return reference to the radius attribute
        agx::Real& radius();
        /// \return const reference to the radius attribute
        agx::Real const& radius() const;

      };
      DOXYGEN_END_INTERNAL_BLOCK()



      typedef agx::VectorPOD<CapsulePtr> CapsulePtrVector;
      typedef agxData::Array<CapsulePtr> CapsulePtrArray;



      DOXYGEN_START_INTERNAL_BLOCK()
      /* Implementation */
      //-----------------------------------------------------------------------------------------------------
      //-----------------------------------------------------------------------------------------------------
      inline CapsuleInstance agx::Physics::Geometry::CapsuleData::operator[] (size_t index) { return CapsuleInstance(this, (agx::Index)index); }
      //-----------------------------------------------------------------------------------------------------
      AGX_FORCE_INLINE CapsulePtr::CapsulePtr() {}
      AGX_FORCE_INLINE CapsulePtr::CapsulePtr(agxData::EntityStorage* storage, agx::Index id) : agx::Physics::Geometry::ShapePtr(storage, id) {}
      AGX_FORCE_INLINE CapsulePtr::CapsulePtr(const agxData::EntityPtr& ptr) : agx::Physics::Geometry::ShapePtr(ptr)
      {
        agxAssertN(!ptr || ptr.isInstanceOf(CapsuleModel::instance()),
          "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
          EntityPtr::getModel()->fullPath().c_str(), CapsuleModel::instance()->fullPath().c_str());
      }

      AGX_FORCE_INLINE CapsulePtr::CapsulePtr(const agxData::EntityInstance& instance) : agx::Physics::Geometry::ShapePtr(instance)
      {
        agxAssertN(!instance || instance.isInstanceOf(CapsuleModel::instance()),
          "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
          EntityPtr::getModel()->fullPath().c_str(), CapsuleModel::instance()->fullPath().c_str());
      }

      AGX_FORCE_INLINE CapsulePtr& CapsulePtr::operator= (const agxData::EntityPtr& ptr)
      {
        agxData::EntityPtr::operator= (ptr);
        agxAssertN(!ptr || ptr.isInstanceOf(CapsuleModel::instance()),
          "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
          EntityPtr::getModel()->fullPath().c_str(), CapsuleModel::instance()->fullPath().c_str());
        return *this;
      }

      AGX_FORCE_INLINE CapsulePtr& CapsulePtr::operator= (const agxData::EntityInstance& instance)
      {
        agxData::EntityPtr::operator= (instance);
        agxAssertN(!instance || instance.isInstanceOf(CapsuleModel::instance()),
          "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
          EntityPtr::getModel()->fullPath().c_str(), CapsuleModel::instance()->fullPath().c_str());
        return *this;
      }

      inline CapsuleInstance CapsulePtr::instance() { return agxData::EntityPtr::instance(); }
      inline const CapsuleInstance CapsulePtr::instance() const { return agxData::EntityPtr::instance(); }
      AGX_FORCE_INLINE CapsuleSemantics* CapsulePtr::operator->() { return (CapsuleSemantics* )this; }
      AGX_FORCE_INLINE const CapsuleSemantics* CapsulePtr::operator->() const { return (const CapsuleSemantics* )this; }
      AGX_FORCE_INLINE CapsuleData* CapsulePtr::getData() { return static_cast<CapsuleData* >(agxData::EntityPtr::getData()); }
      AGX_FORCE_INLINE const CapsuleData* CapsulePtr::getData() const { return static_cast<const CapsuleData* >(agxData::EntityPtr::getData()); }

      AGX_FORCE_INLINE agx::Real& CapsulePtr::height() { verifyIndex(); return getData()->height[calculateIndex()]; }
      AGX_FORCE_INLINE agx::Real const& CapsulePtr::height() const { verifyIndex(); return getData()->height[calculateIndex()]; }

      AGX_FORCE_INLINE agx::Real& CapsulePtr::radius() { verifyIndex(); return getData()->radius[calculateIndex()]; }
      AGX_FORCE_INLINE agx::Real const& CapsulePtr::radius() const { verifyIndex(); return getData()->radius[calculateIndex()]; }

      //-----------------------------------------------------------------------------------------------------
      AGX_FORCE_INLINE CapsuleInstance::CapsuleInstance() {}
      AGX_FORCE_INLINE CapsuleInstance::CapsuleInstance(CapsuleData* data, agx::Index index) : agx::Physics::Geometry::ShapeInstance(data, index) {}
      AGX_FORCE_INLINE CapsuleInstance::CapsuleInstance(agxData::EntityStorage* storage, agx::Index index) : agx::Physics::Geometry::ShapeInstance(storage, index) {}
      AGX_FORCE_INLINE CapsuleInstance::CapsuleInstance(const agxData::EntityInstance& other) : agx::Physics::Geometry::ShapeInstance(other)
      {
        agxAssertN(!other || other.isInstanceOf(CapsuleModel::instance()),
          "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
          EntityInstance::getModel()->fullPath().c_str(), CapsuleModel::instance()->fullPath().c_str());
      }

      AGX_FORCE_INLINE CapsuleInstance::CapsuleInstance(const agxData::EntityPtr& ptr) : agx::Physics::Geometry::ShapeInstance(ptr)
      {
        agxAssertN(!ptr || ptr.isInstanceOf(CapsuleModel::instance()),
          "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
          EntityInstance::getModel()->fullPath().c_str(), CapsuleModel::instance()->fullPath().c_str());
      }


      AGX_FORCE_INLINE CapsuleData* CapsuleInstance::getData() { return static_cast<CapsuleData* >(agxData::EntityInstance::getData()); }
      AGX_FORCE_INLINE const CapsuleData* CapsuleInstance::getData() const { return static_cast<const CapsuleData* >(agxData::EntityInstance::getData()); }

      AGX_FORCE_INLINE agx::Real& CapsuleInstance::height() { verifyIndex(); return getData()->height[getIndex()]; }
      AGX_FORCE_INLINE agx::Real const& CapsuleInstance::height() const { verifyIndex(); return getData()->height[getIndex()]; }

      AGX_FORCE_INLINE agx::Real& CapsuleInstance::radius() { verifyIndex(); return getData()->radius[getIndex()]; }
      AGX_FORCE_INLINE agx::Real const& CapsuleInstance::radius() const { verifyIndex(); return getData()->radius[getIndex()]; }

      //-----------------------------------------------------------------------------------------------------
      AGX_FORCE_INLINE CapsuleSemantics::CapsuleSemantics() {}
      //-----------------------------------------------------------------------------------------------------
      DOXYGEN_END_INTERNAL_BLOCK()
    }
  }
}

AGX_TYPE_BINDING(agx::Physics::Geometry::CapsulePtr, "Physics.Geometry.CapsulePtr")
AGX_TYPE_BINDING(agx::Physics::Geometry::CapsuleInstance, "Physics.Geometry.CapsuleInstance")

#ifdef _MSC_VER
# pragma warning(pop)
#endif

#undef AGX_ENTITY_WRAPPER
#undef AGX_ENTITY_NAMESPACE
#endif

