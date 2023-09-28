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

#ifndef GENERATED_AGX_PHYSICS_GEOMETRY_HOLLOWCONE_H_PLUGIN
#define GENERATED_AGX_PHYSICS_GEOMETRY_HOLLOWCONE_H_PLUGIN

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

      class HollowConeModel;
      class HollowConeData;
      class HollowConePtr;
      class HollowConeInstance;
      class HollowConeSemantics;


      AGX_DECLARE_POINTER_TYPES(HollowConeModel);

      /** 
      Abstract description of the data attributes for the Physics.Geometry.HollowCone entity.
      */ 
      class AGXPHYSICS_EXPORT HollowConeModel : public agx::Physics::Geometry::ShapeModel
      {
      public:
        typedef HollowConePtr PtrT;

        HollowConeModel(const agx::String& name = "HollowCone");

        /// \return The entity model singleton.
        static HollowConeModel* instance();

        /// Create and return a pointer to a new instance in the default storage for this entity model.
        static HollowConePtr createInstance();

        /// \return The default storage for this entity model.
        static agxData::EntityStorage* defaultStorage();

        /// This is part of internal cleanup and should not be called by users
        virtual void shutdownCleanup() override;



        /* Attributes */
        static agxData::ScalarAttributeT< agx::Real >* heightAttribute;
        static agxData::ScalarAttributeT< agx::Real >* topRadiusAttribute;
        static agxData::ScalarAttributeT< agx::Real >* bottomRadiusAttribute;
        static agxData::ScalarAttributeT< agx::Real >* thicknessAttribute;

      protected:
        virtual ~HollowConeModel();
        virtual agxData::EntityData* createData(agxData::EntityStorage* storage) override;
        virtual void configure(agx::TiXmlElement* eEntity) override;
        virtual void initAttributeAccessors() override;
        virtual void construct(agxData::EntityPtr instance) override;
        void construct(agx::Physics::Geometry::HollowConePtr hollowCone);
      };


      DOXYGEN_START_INTERNAL_BLOCK()
      #ifndef AGX_PHYSICS_GEOMETRY_HOLLOWCONE_DATA_SET_OVERRIDE
      #define AGX_PHYSICS_GEOMETRY_HOLLOWCONE_DATA_SET
      class AGXPHYSICS_EXPORT HollowConeData : public agx::Physics::Geometry::ShapeData
      {
      public:
        HollowConeInstance operator[] (size_t index);

      public:
        agxData::Array< HollowConePtr >& instance;
        agxData::Array< agx::Real > height;
        agxData::Array< agx::Real > topRadius;
        agxData::Array< agx::Real > bottomRadius;
        agxData::Array< agx::Real > thickness;

      public:
        typedef agx::Real heightType;
        typedef agx::Real topRadiusType;
        typedef agx::Real bottomRadiusType;
        typedef agx::Real thicknessType;

      public:
        HollowConeData(agxData::EntityStorage* storage);
        HollowConeData();

      protected:
        virtual ~HollowConeData() {}
        virtual void setNumElements(agx::Index numElements) override;

      private:
        HollowConeData& operator= (const HollowConeData&) { return *this; }

      };
      #endif
      DOXYGEN_END_INTERNAL_BLOCK()


      DOXYGEN_START_INTERNAL_BLOCK()
      class AGXPHYSICS_EXPORT HollowConeSemantics : public agx::Physics::Geometry::ShapeSemantics
      {
      public:

        // Automatic getters
        agx::Real const& getHeight() const;
        agx::Real const& getTopRadius() const;
        agx::Real const& getBottomRadius() const;
        agx::Real const& getThickness() const;

        // Semantics defined by explicit kernels

        // Automatic setters
        void setHeight(agx::Real const& value);
        void setTopRadius(agx::Real const& value);
        void setBottomRadius(agx::Real const& value);
        void setThickness(agx::Real const& value);


      protected:
        friend class HollowConePtr;
        friend class HollowConeInstance;
        HollowConeSemantics();
      };
      DOXYGEN_END_INTERNAL_BLOCK()


      /**
      Pointer to a entity instance of type Physics.Geometry.HollowCone
      */
      class CALLABLE HollowConePtr : public agx::Physics::Geometry::ShapePtr
      {
      public:
        typedef HollowConeModel ModelType;
        typedef HollowConeData DataType;
        typedef HollowConeInstance InstanceType;

      public:
        AGXPHYSICS_EXPORT HollowConePtr();
        AGXPHYSICS_EXPORT HollowConePtr(agxData::EntityStorage* storage, agx::Index id);
        AGXPHYSICS_EXPORT HollowConePtr(const agxData::EntityPtr& ptr);
        AGXPHYSICS_EXPORT HollowConePtr(const agxData::EntityInstance& instance);
        AGXPHYSICS_EXPORT HollowConePtr& operator= (const agxData::EntityPtr& ptr);
        AGXPHYSICS_EXPORT HollowConePtr& operator= (const agxData::EntityInstance& instance);
        AGXPHYSICS_EXPORT HollowConeInstance instance();
        AGXPHYSICS_EXPORT const HollowConeInstance instance() const;

        AGXPHYSICS_EXPORT HollowConeSemantics* operator->();
        AGXPHYSICS_EXPORT const HollowConeSemantics* operator->() const;

        HollowConeData* getData();
        const HollowConeData* getData() const;


        /// \return reference to the height attribute
        AGXPHYSICS_EXPORT agx::Real& height();
        /// \return const reference to the height attribute
        AGXPHYSICS_EXPORT agx::Real const& height() const;

        /// \return reference to the topRadius attribute
        AGXPHYSICS_EXPORT agx::Real& topRadius();
        /// \return const reference to the topRadius attribute
        AGXPHYSICS_EXPORT agx::Real const& topRadius() const;

        /// \return reference to the bottomRadius attribute
        AGXPHYSICS_EXPORT agx::Real& bottomRadius();
        /// \return const reference to the bottomRadius attribute
        AGXPHYSICS_EXPORT agx::Real const& bottomRadius() const;

        /// \return reference to the thickness attribute
        AGXPHYSICS_EXPORT agx::Real& thickness();
        /// \return const reference to the thickness attribute
        AGXPHYSICS_EXPORT agx::Real const& thickness() const;

        DOXYGEN_START_INTERNAL_BLOCK()
        // Only used by tolua!
        #ifdef AGX_GENERATED_TOLUA_WRAPPER
        inline agx::Real const& getHeight() const { return (*this)->getHeight(); }
        inline agx::Real const& getTopRadius() const { return (*this)->getTopRadius(); }
        inline agx::Real const& getBottomRadius() const { return (*this)->getBottomRadius(); }
        inline agx::Real const& getThickness() const { return (*this)->getThickness(); }


        inline void setHeight(agx::Real const& value) { (*this)->setHeight(value); }
        inline void setTopRadius(agx::Real const& value) { (*this)->setTopRadius(value); }
        inline void setBottomRadius(agx::Real const& value) { (*this)->setBottomRadius(value); }
        inline void setThickness(agx::Real const& value) { (*this)->setThickness(value); }
        #endif
        DOXYGEN_END_INTERNAL_BLOCK()

      };

      // Entity is Referenced
      typedef agxData::EntityRef< HollowConePtr > HollowConeRef;


      DOXYGEN_START_INTERNAL_BLOCK()
      class AGXPHYSICS_EXPORT HollowConeInstance : public agx::Physics::Geometry::ShapeInstance
      {
      public:
        HollowConeInstance();
        HollowConeInstance(HollowConeData* data, agx::Index index);
        HollowConeInstance(agxData::EntityStorage *storage, agx::Index index);
        HollowConeInstance(const agxData::EntityInstance& other);
        HollowConeInstance(const agxData::EntityPtr& ptr);

        HollowConeData* getData();
        const HollowConeData* getData() const;

      public:
        /// \return reference to the height attribute
        agx::Real& height();
        /// \return const reference to the height attribute
        agx::Real const& height() const;

        /// \return reference to the topRadius attribute
        agx::Real& topRadius();
        /// \return const reference to the topRadius attribute
        agx::Real const& topRadius() const;

        /// \return reference to the bottomRadius attribute
        agx::Real& bottomRadius();
        /// \return const reference to the bottomRadius attribute
        agx::Real const& bottomRadius() const;

        /// \return reference to the thickness attribute
        agx::Real& thickness();
        /// \return const reference to the thickness attribute
        agx::Real const& thickness() const;

      };
      DOXYGEN_END_INTERNAL_BLOCK()



      typedef agx::VectorPOD<HollowConePtr> HollowConePtrVector;
      typedef agxData::Array<HollowConePtr> HollowConePtrArray;



      DOXYGEN_START_INTERNAL_BLOCK()
      /* Implementation */
      //-----------------------------------------------------------------------------------------------------
      //-----------------------------------------------------------------------------------------------------
      inline HollowConeInstance agx::Physics::Geometry::HollowConeData::operator[] (size_t index) { return HollowConeInstance(this, (agx::Index)index); }
      //-----------------------------------------------------------------------------------------------------
      AGX_FORCE_INLINE HollowConePtr::HollowConePtr() {}
      AGX_FORCE_INLINE HollowConePtr::HollowConePtr(agxData::EntityStorage* storage, agx::Index id) : agx::Physics::Geometry::ShapePtr(storage, id) {}
      AGX_FORCE_INLINE HollowConePtr::HollowConePtr(const agxData::EntityPtr& ptr) : agx::Physics::Geometry::ShapePtr(ptr)
      {
        agxAssertN(!ptr || ptr.isInstanceOf(HollowConeModel::instance()),
          "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
          EntityPtr::getModel()->fullPath().c_str(), HollowConeModel::instance()->fullPath().c_str());
      }

      AGX_FORCE_INLINE HollowConePtr::HollowConePtr(const agxData::EntityInstance& instance) : agx::Physics::Geometry::ShapePtr(instance)
      {
        agxAssertN(!instance || instance.isInstanceOf(HollowConeModel::instance()),
          "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
          EntityPtr::getModel()->fullPath().c_str(), HollowConeModel::instance()->fullPath().c_str());
      }

      AGX_FORCE_INLINE HollowConePtr& HollowConePtr::operator= (const agxData::EntityPtr& ptr)
      {
        agxData::EntityPtr::operator= (ptr);
        agxAssertN(!ptr || ptr.isInstanceOf(HollowConeModel::instance()),
          "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
          EntityPtr::getModel()->fullPath().c_str(), HollowConeModel::instance()->fullPath().c_str());
        return *this;
      }

      AGX_FORCE_INLINE HollowConePtr& HollowConePtr::operator= (const agxData::EntityInstance& instance)
      {
        agxData::EntityPtr::operator= (instance);
        agxAssertN(!instance || instance.isInstanceOf(HollowConeModel::instance()),
          "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
          EntityPtr::getModel()->fullPath().c_str(), HollowConeModel::instance()->fullPath().c_str());
        return *this;
      }

      inline HollowConeInstance HollowConePtr::instance() { return agxData::EntityPtr::instance(); }
      inline const HollowConeInstance HollowConePtr::instance() const { return agxData::EntityPtr::instance(); }
      AGX_FORCE_INLINE HollowConeSemantics* HollowConePtr::operator->() { return (HollowConeSemantics* )this; }
      AGX_FORCE_INLINE const HollowConeSemantics* HollowConePtr::operator->() const { return (const HollowConeSemantics* )this; }
      AGX_FORCE_INLINE HollowConeData* HollowConePtr::getData() { return static_cast<HollowConeData* >(agxData::EntityPtr::getData()); }
      AGX_FORCE_INLINE const HollowConeData* HollowConePtr::getData() const { return static_cast<const HollowConeData* >(agxData::EntityPtr::getData()); }

      AGX_FORCE_INLINE agx::Real& HollowConePtr::height() { verifyIndex(); return getData()->height[calculateIndex()]; }
      AGX_FORCE_INLINE agx::Real const& HollowConePtr::height() const { verifyIndex(); return getData()->height[calculateIndex()]; }

      AGX_FORCE_INLINE agx::Real& HollowConePtr::topRadius() { verifyIndex(); return getData()->topRadius[calculateIndex()]; }
      AGX_FORCE_INLINE agx::Real const& HollowConePtr::topRadius() const { verifyIndex(); return getData()->topRadius[calculateIndex()]; }

      AGX_FORCE_INLINE agx::Real& HollowConePtr::bottomRadius() { verifyIndex(); return getData()->bottomRadius[calculateIndex()]; }
      AGX_FORCE_INLINE agx::Real const& HollowConePtr::bottomRadius() const { verifyIndex(); return getData()->bottomRadius[calculateIndex()]; }

      AGX_FORCE_INLINE agx::Real& HollowConePtr::thickness() { verifyIndex(); return getData()->thickness[calculateIndex()]; }
      AGX_FORCE_INLINE agx::Real const& HollowConePtr::thickness() const { verifyIndex(); return getData()->thickness[calculateIndex()]; }

      //-----------------------------------------------------------------------------------------------------
      AGX_FORCE_INLINE HollowConeInstance::HollowConeInstance() {}
      AGX_FORCE_INLINE HollowConeInstance::HollowConeInstance(HollowConeData* data, agx::Index index) : agx::Physics::Geometry::ShapeInstance(data, index) {}
      AGX_FORCE_INLINE HollowConeInstance::HollowConeInstance(agxData::EntityStorage* storage, agx::Index index) : agx::Physics::Geometry::ShapeInstance(storage, index) {}
      AGX_FORCE_INLINE HollowConeInstance::HollowConeInstance(const agxData::EntityInstance& other) : agx::Physics::Geometry::ShapeInstance(other)
      {
        agxAssertN(!other || other.isInstanceOf(HollowConeModel::instance()),
          "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
          EntityInstance::getModel()->fullPath().c_str(), HollowConeModel::instance()->fullPath().c_str());
      }

      AGX_FORCE_INLINE HollowConeInstance::HollowConeInstance(const agxData::EntityPtr& ptr) : agx::Physics::Geometry::ShapeInstance(ptr)
      {
        agxAssertN(!ptr || ptr.isInstanceOf(HollowConeModel::instance()),
          "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
          EntityInstance::getModel()->fullPath().c_str(), HollowConeModel::instance()->fullPath().c_str());
      }


      AGX_FORCE_INLINE HollowConeData* HollowConeInstance::getData() { return static_cast<HollowConeData* >(agxData::EntityInstance::getData()); }
      AGX_FORCE_INLINE const HollowConeData* HollowConeInstance::getData() const { return static_cast<const HollowConeData* >(agxData::EntityInstance::getData()); }

      AGX_FORCE_INLINE agx::Real& HollowConeInstance::height() { verifyIndex(); return getData()->height[getIndex()]; }
      AGX_FORCE_INLINE agx::Real const& HollowConeInstance::height() const { verifyIndex(); return getData()->height[getIndex()]; }

      AGX_FORCE_INLINE agx::Real& HollowConeInstance::topRadius() { verifyIndex(); return getData()->topRadius[getIndex()]; }
      AGX_FORCE_INLINE agx::Real const& HollowConeInstance::topRadius() const { verifyIndex(); return getData()->topRadius[getIndex()]; }

      AGX_FORCE_INLINE agx::Real& HollowConeInstance::bottomRadius() { verifyIndex(); return getData()->bottomRadius[getIndex()]; }
      AGX_FORCE_INLINE agx::Real const& HollowConeInstance::bottomRadius() const { verifyIndex(); return getData()->bottomRadius[getIndex()]; }

      AGX_FORCE_INLINE agx::Real& HollowConeInstance::thickness() { verifyIndex(); return getData()->thickness[getIndex()]; }
      AGX_FORCE_INLINE agx::Real const& HollowConeInstance::thickness() const { verifyIndex(); return getData()->thickness[getIndex()]; }

      //-----------------------------------------------------------------------------------------------------
      AGX_FORCE_INLINE HollowConeSemantics::HollowConeSemantics() {}
      //-----------------------------------------------------------------------------------------------------
      DOXYGEN_END_INTERNAL_BLOCK()
    }
  }
}

AGX_TYPE_BINDING(agx::Physics::Geometry::HollowConePtr, "Physics.Geometry.HollowConePtr")
AGX_TYPE_BINDING(agx::Physics::Geometry::HollowConeInstance, "Physics.Geometry.HollowConeInstance")

#ifdef _MSC_VER
# pragma warning(pop)
#endif

#undef AGX_ENTITY_WRAPPER
#undef AGX_ENTITY_NAMESPACE
#endif

