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

#ifndef GENERATED_AGX_PHYSICS_GEOMETRY_HOLLOWCYLINDER_H_PLUGIN
#define GENERATED_AGX_PHYSICS_GEOMETRY_HOLLOWCYLINDER_H_PLUGIN

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

      class HollowCylinderModel;
      class HollowCylinderData;
      class HollowCylinderPtr;
      class HollowCylinderInstance;
      class HollowCylinderSemantics;


      AGX_DECLARE_POINTER_TYPES(HollowCylinderModel);

      /** 
      Abstract description of the data attributes for the Physics.Geometry.HollowCylinder entity.
      */ 
      class AGXPHYSICS_EXPORT HollowCylinderModel : public agx::Physics::Geometry::ShapeModel
      {
      public:
        typedef HollowCylinderPtr PtrT;

        HollowCylinderModel(const agx::String& name = "HollowCylinder");

        /// \return The entity model singleton.
        static HollowCylinderModel* instance();

        /// Create and return a pointer to a new instance in the default storage for this entity model.
        static HollowCylinderPtr createInstance();

        /// \return The default storage for this entity model.
        static agxData::EntityStorage* defaultStorage();

        /// This is part of internal cleanup and should not be called by users
        virtual void shutdownCleanup() override;



        /* Attributes */
        static agxData::ScalarAttributeT< agx::Real >* heightAttribute;
        static agxData::ScalarAttributeT< agx::Real >* radiusAttribute;
        static agxData::ScalarAttributeT< agx::Real >* thicknessAttribute;

      protected:
        virtual ~HollowCylinderModel();
        virtual agxData::EntityData* createData(agxData::EntityStorage* storage) override;
        virtual void configure(agx::TiXmlElement* eEntity) override;
        virtual void initAttributeAccessors() override;
        virtual void construct(agxData::EntityPtr instance) override;
        void construct(agx::Physics::Geometry::HollowCylinderPtr hollowCylinder);
      };


      DOXYGEN_START_INTERNAL_BLOCK()
      #ifndef AGX_PHYSICS_GEOMETRY_HOLLOWCYLINDER_DATA_SET_OVERRIDE
      #define AGX_PHYSICS_GEOMETRY_HOLLOWCYLINDER_DATA_SET
      class AGXPHYSICS_EXPORT HollowCylinderData : public agx::Physics::Geometry::ShapeData
      {
      public:
        HollowCylinderInstance operator[] (size_t index);

      public:
        agxData::Array< HollowCylinderPtr >& instance;
        agxData::Array< agx::Real > height;
        agxData::Array< agx::Real > radius;
        agxData::Array< agx::Real > thickness;

      public:
        typedef agx::Real heightType;
        typedef agx::Real radiusType;
        typedef agx::Real thicknessType;

      public:
        HollowCylinderData(agxData::EntityStorage* storage);
        HollowCylinderData();

      protected:
        virtual ~HollowCylinderData() {}
        virtual void setNumElements(agx::Index numElements) override;

      private:
        HollowCylinderData& operator= (const HollowCylinderData&) { return *this; }

      };
      #endif
      DOXYGEN_END_INTERNAL_BLOCK()


      DOXYGEN_START_INTERNAL_BLOCK()
      class AGXPHYSICS_EXPORT HollowCylinderSemantics : public agx::Physics::Geometry::ShapeSemantics
      {
      public:

        // Automatic getters
        agx::Real const& getHeight() const;
        agx::Real const& getRadius() const;
        agx::Real const& getThickness() const;

        // Semantics defined by explicit kernels

        // Automatic setters
        void setHeight(agx::Real const& value);
        void setRadius(agx::Real const& value);
        void setThickness(agx::Real const& value);


      protected:
        friend class HollowCylinderPtr;
        friend class HollowCylinderInstance;
        HollowCylinderSemantics();
      };
      DOXYGEN_END_INTERNAL_BLOCK()


      /**
      Pointer to a entity instance of type Physics.Geometry.HollowCylinder
      */
      class CALLABLE HollowCylinderPtr : public agx::Physics::Geometry::ShapePtr
      {
      public:
        typedef HollowCylinderModel ModelType;
        typedef HollowCylinderData DataType;
        typedef HollowCylinderInstance InstanceType;

      public:
        AGXPHYSICS_EXPORT HollowCylinderPtr();
        AGXPHYSICS_EXPORT HollowCylinderPtr(agxData::EntityStorage* storage, agx::Index id);
        AGXPHYSICS_EXPORT HollowCylinderPtr(const agxData::EntityPtr& ptr);
        AGXPHYSICS_EXPORT HollowCylinderPtr(const agxData::EntityInstance& instance);
        AGXPHYSICS_EXPORT HollowCylinderPtr& operator= (const agxData::EntityPtr& ptr);
        AGXPHYSICS_EXPORT HollowCylinderPtr& operator= (const agxData::EntityInstance& instance);
        AGXPHYSICS_EXPORT HollowCylinderInstance instance();
        AGXPHYSICS_EXPORT const HollowCylinderInstance instance() const;

        AGXPHYSICS_EXPORT HollowCylinderSemantics* operator->();
        AGXPHYSICS_EXPORT const HollowCylinderSemantics* operator->() const;

        HollowCylinderData* getData();
        const HollowCylinderData* getData() const;


        /// \return reference to the height attribute
        AGXPHYSICS_EXPORT agx::Real& height();
        /// \return const reference to the height attribute
        AGXPHYSICS_EXPORT agx::Real const& height() const;

        /// \return reference to the radius attribute
        AGXPHYSICS_EXPORT agx::Real& radius();
        /// \return const reference to the radius attribute
        AGXPHYSICS_EXPORT agx::Real const& radius() const;

        /// \return reference to the thickness attribute
        AGXPHYSICS_EXPORT agx::Real& thickness();
        /// \return const reference to the thickness attribute
        AGXPHYSICS_EXPORT agx::Real const& thickness() const;

        DOXYGEN_START_INTERNAL_BLOCK()
        // Only used by tolua!
        #ifdef AGX_GENERATED_TOLUA_WRAPPER
        inline agx::Real const& getHeight() const { return (*this)->getHeight(); }
        inline agx::Real const& getRadius() const { return (*this)->getRadius(); }
        inline agx::Real const& getThickness() const { return (*this)->getThickness(); }


        inline void setHeight(agx::Real const& value) { (*this)->setHeight(value); }
        inline void setRadius(agx::Real const& value) { (*this)->setRadius(value); }
        inline void setThickness(agx::Real const& value) { (*this)->setThickness(value); }
        #endif
        DOXYGEN_END_INTERNAL_BLOCK()

      };

      // Entity is Referenced
      typedef agxData::EntityRef< HollowCylinderPtr > HollowCylinderRef;


      DOXYGEN_START_INTERNAL_BLOCK()
      class AGXPHYSICS_EXPORT HollowCylinderInstance : public agx::Physics::Geometry::ShapeInstance
      {
      public:
        HollowCylinderInstance();
        HollowCylinderInstance(HollowCylinderData* data, agx::Index index);
        HollowCylinderInstance(agxData::EntityStorage *storage, agx::Index index);
        HollowCylinderInstance(const agxData::EntityInstance& other);
        HollowCylinderInstance(const agxData::EntityPtr& ptr);

        HollowCylinderData* getData();
        const HollowCylinderData* getData() const;

      public:
        /// \return reference to the height attribute
        agx::Real& height();
        /// \return const reference to the height attribute
        agx::Real const& height() const;

        /// \return reference to the radius attribute
        agx::Real& radius();
        /// \return const reference to the radius attribute
        agx::Real const& radius() const;

        /// \return reference to the thickness attribute
        agx::Real& thickness();
        /// \return const reference to the thickness attribute
        agx::Real const& thickness() const;

      };
      DOXYGEN_END_INTERNAL_BLOCK()



      typedef agx::VectorPOD<HollowCylinderPtr> HollowCylinderPtrVector;
      typedef agxData::Array<HollowCylinderPtr> HollowCylinderPtrArray;



      DOXYGEN_START_INTERNAL_BLOCK()
      /* Implementation */
      //-----------------------------------------------------------------------------------------------------
      //-----------------------------------------------------------------------------------------------------
      inline HollowCylinderInstance agx::Physics::Geometry::HollowCylinderData::operator[] (size_t index) { return HollowCylinderInstance(this, (agx::Index)index); }
      //-----------------------------------------------------------------------------------------------------
      AGX_FORCE_INLINE HollowCylinderPtr::HollowCylinderPtr() {}
      AGX_FORCE_INLINE HollowCylinderPtr::HollowCylinderPtr(agxData::EntityStorage* storage, agx::Index id) : agx::Physics::Geometry::ShapePtr(storage, id) {}
      AGX_FORCE_INLINE HollowCylinderPtr::HollowCylinderPtr(const agxData::EntityPtr& ptr) : agx::Physics::Geometry::ShapePtr(ptr)
      {
        agxAssertN(!ptr || ptr.isInstanceOf(HollowCylinderModel::instance()),
          "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
          EntityPtr::getModel()->fullPath().c_str(), HollowCylinderModel::instance()->fullPath().c_str());
      }

      AGX_FORCE_INLINE HollowCylinderPtr::HollowCylinderPtr(const agxData::EntityInstance& instance) : agx::Physics::Geometry::ShapePtr(instance)
      {
        agxAssertN(!instance || instance.isInstanceOf(HollowCylinderModel::instance()),
          "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
          EntityPtr::getModel()->fullPath().c_str(), HollowCylinderModel::instance()->fullPath().c_str());
      }

      AGX_FORCE_INLINE HollowCylinderPtr& HollowCylinderPtr::operator= (const agxData::EntityPtr& ptr)
      {
        agxData::EntityPtr::operator= (ptr);
        agxAssertN(!ptr || ptr.isInstanceOf(HollowCylinderModel::instance()),
          "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
          EntityPtr::getModel()->fullPath().c_str(), HollowCylinderModel::instance()->fullPath().c_str());
        return *this;
      }

      AGX_FORCE_INLINE HollowCylinderPtr& HollowCylinderPtr::operator= (const agxData::EntityInstance& instance)
      {
        agxData::EntityPtr::operator= (instance);
        agxAssertN(!instance || instance.isInstanceOf(HollowCylinderModel::instance()),
          "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
          EntityPtr::getModel()->fullPath().c_str(), HollowCylinderModel::instance()->fullPath().c_str());
        return *this;
      }

      inline HollowCylinderInstance HollowCylinderPtr::instance() { return agxData::EntityPtr::instance(); }
      inline const HollowCylinderInstance HollowCylinderPtr::instance() const { return agxData::EntityPtr::instance(); }
      AGX_FORCE_INLINE HollowCylinderSemantics* HollowCylinderPtr::operator->() { return (HollowCylinderSemantics* )this; }
      AGX_FORCE_INLINE const HollowCylinderSemantics* HollowCylinderPtr::operator->() const { return (const HollowCylinderSemantics* )this; }
      AGX_FORCE_INLINE HollowCylinderData* HollowCylinderPtr::getData() { return static_cast<HollowCylinderData* >(agxData::EntityPtr::getData()); }
      AGX_FORCE_INLINE const HollowCylinderData* HollowCylinderPtr::getData() const { return static_cast<const HollowCylinderData* >(agxData::EntityPtr::getData()); }

      AGX_FORCE_INLINE agx::Real& HollowCylinderPtr::height() { verifyIndex(); return getData()->height[calculateIndex()]; }
      AGX_FORCE_INLINE agx::Real const& HollowCylinderPtr::height() const { verifyIndex(); return getData()->height[calculateIndex()]; }

      AGX_FORCE_INLINE agx::Real& HollowCylinderPtr::radius() { verifyIndex(); return getData()->radius[calculateIndex()]; }
      AGX_FORCE_INLINE agx::Real const& HollowCylinderPtr::radius() const { verifyIndex(); return getData()->radius[calculateIndex()]; }

      AGX_FORCE_INLINE agx::Real& HollowCylinderPtr::thickness() { verifyIndex(); return getData()->thickness[calculateIndex()]; }
      AGX_FORCE_INLINE agx::Real const& HollowCylinderPtr::thickness() const { verifyIndex(); return getData()->thickness[calculateIndex()]; }

      //-----------------------------------------------------------------------------------------------------
      AGX_FORCE_INLINE HollowCylinderInstance::HollowCylinderInstance() {}
      AGX_FORCE_INLINE HollowCylinderInstance::HollowCylinderInstance(HollowCylinderData* data, agx::Index index) : agx::Physics::Geometry::ShapeInstance(data, index) {}
      AGX_FORCE_INLINE HollowCylinderInstance::HollowCylinderInstance(agxData::EntityStorage* storage, agx::Index index) : agx::Physics::Geometry::ShapeInstance(storage, index) {}
      AGX_FORCE_INLINE HollowCylinderInstance::HollowCylinderInstance(const agxData::EntityInstance& other) : agx::Physics::Geometry::ShapeInstance(other)
      {
        agxAssertN(!other || other.isInstanceOf(HollowCylinderModel::instance()),
          "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
          EntityInstance::getModel()->fullPath().c_str(), HollowCylinderModel::instance()->fullPath().c_str());
      }

      AGX_FORCE_INLINE HollowCylinderInstance::HollowCylinderInstance(const agxData::EntityPtr& ptr) : agx::Physics::Geometry::ShapeInstance(ptr)
      {
        agxAssertN(!ptr || ptr.isInstanceOf(HollowCylinderModel::instance()),
          "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
          EntityInstance::getModel()->fullPath().c_str(), HollowCylinderModel::instance()->fullPath().c_str());
      }


      AGX_FORCE_INLINE HollowCylinderData* HollowCylinderInstance::getData() { return static_cast<HollowCylinderData* >(agxData::EntityInstance::getData()); }
      AGX_FORCE_INLINE const HollowCylinderData* HollowCylinderInstance::getData() const { return static_cast<const HollowCylinderData* >(agxData::EntityInstance::getData()); }

      AGX_FORCE_INLINE agx::Real& HollowCylinderInstance::height() { verifyIndex(); return getData()->height[getIndex()]; }
      AGX_FORCE_INLINE agx::Real const& HollowCylinderInstance::height() const { verifyIndex(); return getData()->height[getIndex()]; }

      AGX_FORCE_INLINE agx::Real& HollowCylinderInstance::radius() { verifyIndex(); return getData()->radius[getIndex()]; }
      AGX_FORCE_INLINE agx::Real const& HollowCylinderInstance::radius() const { verifyIndex(); return getData()->radius[getIndex()]; }

      AGX_FORCE_INLINE agx::Real& HollowCylinderInstance::thickness() { verifyIndex(); return getData()->thickness[getIndex()]; }
      AGX_FORCE_INLINE agx::Real const& HollowCylinderInstance::thickness() const { verifyIndex(); return getData()->thickness[getIndex()]; }

      //-----------------------------------------------------------------------------------------------------
      AGX_FORCE_INLINE HollowCylinderSemantics::HollowCylinderSemantics() {}
      //-----------------------------------------------------------------------------------------------------
      DOXYGEN_END_INTERNAL_BLOCK()
    }
  }
}

AGX_TYPE_BINDING(agx::Physics::Geometry::HollowCylinderPtr, "Physics.Geometry.HollowCylinderPtr")
AGX_TYPE_BINDING(agx::Physics::Geometry::HollowCylinderInstance, "Physics.Geometry.HollowCylinderInstance")

#ifdef _MSC_VER
# pragma warning(pop)
#endif

#undef AGX_ENTITY_WRAPPER
#undef AGX_ENTITY_NAMESPACE
#endif

