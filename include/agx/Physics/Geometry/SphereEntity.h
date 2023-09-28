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

#ifndef GENERATED_AGX_PHYSICS_GEOMETRY_SPHERE_H_PLUGIN
#define GENERATED_AGX_PHYSICS_GEOMETRY_SPHERE_H_PLUGIN

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

      class SphereModel;
      class SphereData;
      class SpherePtr;
      class SphereInstance;
      class SphereSemantics;


      AGX_DECLARE_POINTER_TYPES(SphereModel);

      /** 
      Abstract description of the data attributes for the Physics.Geometry.Sphere entity.
      */ 
      class AGXPHYSICS_EXPORT SphereModel : public agx::Physics::Geometry::ShapeModel
      {
      public:
        typedef SpherePtr PtrT;

        SphereModel(const agx::String& name = "Sphere");

        /// \return The entity model singleton.
        static SphereModel* instance();

        /// Create and return a pointer to a new instance in the default storage for this entity model.
        static SpherePtr createInstance();

        /// \return The default storage for this entity model.
        static agxData::EntityStorage* defaultStorage();

        /// This is part of internal cleanup and should not be called by users
        virtual void shutdownCleanup() override;



        /* Attributes */
        static agxData::ScalarAttributeT< agx::Real >* radiusAttribute;

      protected:
        virtual ~SphereModel();
        virtual agxData::EntityData* createData(agxData::EntityStorage* storage) override;
        virtual void configure(agx::TiXmlElement* eEntity) override;
        virtual void initAttributeAccessors() override;
        virtual void construct(agxData::EntityPtr instance) override;
        void construct(agx::Physics::Geometry::SpherePtr sphere);
      };


      DOXYGEN_START_INTERNAL_BLOCK()
      #ifndef AGX_PHYSICS_GEOMETRY_SPHERE_DATA_SET_OVERRIDE
      #define AGX_PHYSICS_GEOMETRY_SPHERE_DATA_SET
      class AGXPHYSICS_EXPORT SphereData : public agx::Physics::Geometry::ShapeData
      {
      public:
        SphereInstance operator[] (size_t index);

      public:
        agxData::Array< SpherePtr >& instance;
        agxData::Array< agx::Real > radius;

      public:
        typedef agx::Real radiusType;

      public:
        SphereData(agxData::EntityStorage* storage);
        SphereData();

      protected:
        virtual ~SphereData() {}
        virtual void setNumElements(agx::Index numElements) override;

      private:
        SphereData& operator= (const SphereData&) { return *this; }

      };
      #endif
      DOXYGEN_END_INTERNAL_BLOCK()


      DOXYGEN_START_INTERNAL_BLOCK()
      class AGXPHYSICS_EXPORT SphereSemantics : public agx::Physics::Geometry::ShapeSemantics
      {
      public:

        // Automatic getters
        agx::Real const& getRadius() const;

        // Semantics defined by explicit kernels

        // Automatic setters
        void setRadius(agx::Real const& value);


      protected:
        friend class SpherePtr;
        friend class SphereInstance;
        SphereSemantics();
      };
      DOXYGEN_END_INTERNAL_BLOCK()


      /**
      Pointer to a entity instance of type Physics.Geometry.Sphere
      */
      class CALLABLE SpherePtr : public agx::Physics::Geometry::ShapePtr
      {
      public:
        typedef SphereModel ModelType;
        typedef SphereData DataType;
        typedef SphereInstance InstanceType;

      public:
        AGXPHYSICS_EXPORT SpherePtr();
        AGXPHYSICS_EXPORT SpherePtr(agxData::EntityStorage* storage, agx::Index id);
        AGXPHYSICS_EXPORT SpherePtr(const agxData::EntityPtr& ptr);
        AGXPHYSICS_EXPORT SpherePtr(const agxData::EntityInstance& instance);
        AGXPHYSICS_EXPORT SpherePtr& operator= (const agxData::EntityPtr& ptr);
        AGXPHYSICS_EXPORT SpherePtr& operator= (const agxData::EntityInstance& instance);
        AGXPHYSICS_EXPORT SphereInstance instance();
        AGXPHYSICS_EXPORT const SphereInstance instance() const;

        AGXPHYSICS_EXPORT SphereSemantics* operator->();
        AGXPHYSICS_EXPORT const SphereSemantics* operator->() const;

        SphereData* getData();
        const SphereData* getData() const;


        /// \return reference to the radius attribute
        AGXPHYSICS_EXPORT agx::Real& radius();
        /// \return const reference to the radius attribute
        AGXPHYSICS_EXPORT agx::Real const& radius() const;

        DOXYGEN_START_INTERNAL_BLOCK()
        // Only used by tolua!
        #ifdef AGX_GENERATED_TOLUA_WRAPPER
        inline agx::Real const& getRadius() const { return (*this)->getRadius(); }


        inline void setRadius(agx::Real const& value) { (*this)->setRadius(value); }
        #endif
        DOXYGEN_END_INTERNAL_BLOCK()

      };

      // Entity is Referenced
      typedef agxData::EntityRef< SpherePtr > SphereRef;


      DOXYGEN_START_INTERNAL_BLOCK()
      class AGXPHYSICS_EXPORT SphereInstance : public agx::Physics::Geometry::ShapeInstance
      {
      public:
        SphereInstance();
        SphereInstance(SphereData* data, agx::Index index);
        SphereInstance(agxData::EntityStorage *storage, agx::Index index);
        SphereInstance(const agxData::EntityInstance& other);
        SphereInstance(const agxData::EntityPtr& ptr);

        SphereData* getData();
        const SphereData* getData() const;

      public:
        /// \return reference to the radius attribute
        agx::Real& radius();
        /// \return const reference to the radius attribute
        agx::Real const& radius() const;

      };
      DOXYGEN_END_INTERNAL_BLOCK()



      typedef agx::VectorPOD<SpherePtr> SpherePtrVector;
      typedef agxData::Array<SpherePtr> SpherePtrArray;



      DOXYGEN_START_INTERNAL_BLOCK()
      /* Implementation */
      //-----------------------------------------------------------------------------------------------------
      //-----------------------------------------------------------------------------------------------------
      inline SphereInstance agx::Physics::Geometry::SphereData::operator[] (size_t index) { return SphereInstance(this, (agx::Index)index); }
      //-----------------------------------------------------------------------------------------------------
      AGX_FORCE_INLINE SpherePtr::SpherePtr() {}
      AGX_FORCE_INLINE SpherePtr::SpherePtr(agxData::EntityStorage* storage, agx::Index id) : agx::Physics::Geometry::ShapePtr(storage, id) {}
      AGX_FORCE_INLINE SpherePtr::SpherePtr(const agxData::EntityPtr& ptr) : agx::Physics::Geometry::ShapePtr(ptr)
      {
        agxAssertN(!ptr || ptr.isInstanceOf(SphereModel::instance()),
          "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
          EntityPtr::getModel()->fullPath().c_str(), SphereModel::instance()->fullPath().c_str());
      }

      AGX_FORCE_INLINE SpherePtr::SpherePtr(const agxData::EntityInstance& instance) : agx::Physics::Geometry::ShapePtr(instance)
      {
        agxAssertN(!instance || instance.isInstanceOf(SphereModel::instance()),
          "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
          EntityPtr::getModel()->fullPath().c_str(), SphereModel::instance()->fullPath().c_str());
      }

      AGX_FORCE_INLINE SpherePtr& SpherePtr::operator= (const agxData::EntityPtr& ptr)
      {
        agxData::EntityPtr::operator= (ptr);
        agxAssertN(!ptr || ptr.isInstanceOf(SphereModel::instance()),
          "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
          EntityPtr::getModel()->fullPath().c_str(), SphereModel::instance()->fullPath().c_str());
        return *this;
      }

      AGX_FORCE_INLINE SpherePtr& SpherePtr::operator= (const agxData::EntityInstance& instance)
      {
        agxData::EntityPtr::operator= (instance);
        agxAssertN(!instance || instance.isInstanceOf(SphereModel::instance()),
          "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
          EntityPtr::getModel()->fullPath().c_str(), SphereModel::instance()->fullPath().c_str());
        return *this;
      }

      inline SphereInstance SpherePtr::instance() { return agxData::EntityPtr::instance(); }
      inline const SphereInstance SpherePtr::instance() const { return agxData::EntityPtr::instance(); }
      AGX_FORCE_INLINE SphereSemantics* SpherePtr::operator->() { return (SphereSemantics* )this; }
      AGX_FORCE_INLINE const SphereSemantics* SpherePtr::operator->() const { return (const SphereSemantics* )this; }
      AGX_FORCE_INLINE SphereData* SpherePtr::getData() { return static_cast<SphereData* >(agxData::EntityPtr::getData()); }
      AGX_FORCE_INLINE const SphereData* SpherePtr::getData() const { return static_cast<const SphereData* >(agxData::EntityPtr::getData()); }

      AGX_FORCE_INLINE agx::Real& SpherePtr::radius() { verifyIndex(); return getData()->radius[calculateIndex()]; }
      AGX_FORCE_INLINE agx::Real const& SpherePtr::radius() const { verifyIndex(); return getData()->radius[calculateIndex()]; }

      //-----------------------------------------------------------------------------------------------------
      AGX_FORCE_INLINE SphereInstance::SphereInstance() {}
      AGX_FORCE_INLINE SphereInstance::SphereInstance(SphereData* data, agx::Index index) : agx::Physics::Geometry::ShapeInstance(data, index) {}
      AGX_FORCE_INLINE SphereInstance::SphereInstance(agxData::EntityStorage* storage, agx::Index index) : agx::Physics::Geometry::ShapeInstance(storage, index) {}
      AGX_FORCE_INLINE SphereInstance::SphereInstance(const agxData::EntityInstance& other) : agx::Physics::Geometry::ShapeInstance(other)
      {
        agxAssertN(!other || other.isInstanceOf(SphereModel::instance()),
          "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
          EntityInstance::getModel()->fullPath().c_str(), SphereModel::instance()->fullPath().c_str());
      }

      AGX_FORCE_INLINE SphereInstance::SphereInstance(const agxData::EntityPtr& ptr) : agx::Physics::Geometry::ShapeInstance(ptr)
      {
        agxAssertN(!ptr || ptr.isInstanceOf(SphereModel::instance()),
          "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
          EntityInstance::getModel()->fullPath().c_str(), SphereModel::instance()->fullPath().c_str());
      }


      AGX_FORCE_INLINE SphereData* SphereInstance::getData() { return static_cast<SphereData* >(agxData::EntityInstance::getData()); }
      AGX_FORCE_INLINE const SphereData* SphereInstance::getData() const { return static_cast<const SphereData* >(agxData::EntityInstance::getData()); }

      AGX_FORCE_INLINE agx::Real& SphereInstance::radius() { verifyIndex(); return getData()->radius[getIndex()]; }
      AGX_FORCE_INLINE agx::Real const& SphereInstance::radius() const { verifyIndex(); return getData()->radius[getIndex()]; }

      //-----------------------------------------------------------------------------------------------------
      AGX_FORCE_INLINE SphereSemantics::SphereSemantics() {}
      //-----------------------------------------------------------------------------------------------------
      DOXYGEN_END_INTERNAL_BLOCK()
    }
  }
}

AGX_TYPE_BINDING(agx::Physics::Geometry::SpherePtr, "Physics.Geometry.SpherePtr")
AGX_TYPE_BINDING(agx::Physics::Geometry::SphereInstance, "Physics.Geometry.SphereInstance")

#ifdef _MSC_VER
# pragma warning(pop)
#endif

#undef AGX_ENTITY_WRAPPER
#undef AGX_ENTITY_NAMESPACE
#endif

