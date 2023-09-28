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

#ifndef GENERATED_AGX_PHYSICS_GEOMETRY_CONE_H_PLUGIN
#define GENERATED_AGX_PHYSICS_GEOMETRY_CONE_H_PLUGIN

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

      class ConeModel;
      class ConeData;
      class ConePtr;
      class ConeInstance;
      class ConeSemantics;


      AGX_DECLARE_POINTER_TYPES(ConeModel);

      /** 
      Abstract description of the data attributes for the Physics.Geometry.Cone entity.
      */ 
      class AGXPHYSICS_EXPORT ConeModel : public agx::Physics::Geometry::ShapeModel
      {
      public:
        typedef ConePtr PtrT;

        ConeModel(const agx::String& name = "Cone");

        /// \return The entity model singleton.
        static ConeModel* instance();

        /// Create and return a pointer to a new instance in the default storage for this entity model.
        static ConePtr createInstance();

        /// \return The default storage for this entity model.
        static agxData::EntityStorage* defaultStorage();

        /// This is part of internal cleanup and should not be called by users
        virtual void shutdownCleanup() override;



        /* Attributes */
        static agxData::ScalarAttributeT< agx::Real >* heightAttribute;
        static agxData::ScalarAttributeT< agx::Real >* topRadiusAttribute;
        static agxData::ScalarAttributeT< agx::Real >* bottomRadiusAttribute;

      protected:
        virtual ~ConeModel();
        virtual agxData::EntityData* createData(agxData::EntityStorage* storage) override;
        virtual void configure(agx::TiXmlElement* eEntity) override;
        virtual void initAttributeAccessors() override;
        virtual void construct(agxData::EntityPtr instance) override;
        void construct(agx::Physics::Geometry::ConePtr cone);
      };


      DOXYGEN_START_INTERNAL_BLOCK()
      #ifndef AGX_PHYSICS_GEOMETRY_CONE_DATA_SET_OVERRIDE
      #define AGX_PHYSICS_GEOMETRY_CONE_DATA_SET
      class AGXPHYSICS_EXPORT ConeData : public agx::Physics::Geometry::ShapeData
      {
      public:
        ConeInstance operator[] (size_t index);

      public:
        agxData::Array< ConePtr >& instance;
        agxData::Array< agx::Real > height;
        agxData::Array< agx::Real > topRadius;
        agxData::Array< agx::Real > bottomRadius;

      public:
        typedef agx::Real heightType;
        typedef agx::Real topRadiusType;
        typedef agx::Real bottomRadiusType;

      public:
        ConeData(agxData::EntityStorage* storage);
        ConeData();

      protected:
        virtual ~ConeData() {}
        virtual void setNumElements(agx::Index numElements) override;

      private:
        ConeData& operator= (const ConeData&) { return *this; }

      };
      #endif
      DOXYGEN_END_INTERNAL_BLOCK()


      DOXYGEN_START_INTERNAL_BLOCK()
      class AGXPHYSICS_EXPORT ConeSemantics : public agx::Physics::Geometry::ShapeSemantics
      {
      public:

        // Automatic getters
        agx::Real const& getHeight() const;
        agx::Real const& getTopRadius() const;
        agx::Real const& getBottomRadius() const;

        // Semantics defined by explicit kernels

        // Automatic setters
        void setHeight(agx::Real const& value);
        void setTopRadius(agx::Real const& value);
        void setBottomRadius(agx::Real const& value);


      protected:
        friend class ConePtr;
        friend class ConeInstance;
        ConeSemantics();
      };
      DOXYGEN_END_INTERNAL_BLOCK()


      /**
      Pointer to a entity instance of type Physics.Geometry.Cone
      */
      class CALLABLE ConePtr : public agx::Physics::Geometry::ShapePtr
      {
      public:
        typedef ConeModel ModelType;
        typedef ConeData DataType;
        typedef ConeInstance InstanceType;

      public:
        AGXPHYSICS_EXPORT ConePtr();
        AGXPHYSICS_EXPORT ConePtr(agxData::EntityStorage* storage, agx::Index id);
        AGXPHYSICS_EXPORT ConePtr(const agxData::EntityPtr& ptr);
        AGXPHYSICS_EXPORT ConePtr(const agxData::EntityInstance& instance);
        AGXPHYSICS_EXPORT ConePtr& operator= (const agxData::EntityPtr& ptr);
        AGXPHYSICS_EXPORT ConePtr& operator= (const agxData::EntityInstance& instance);
        AGXPHYSICS_EXPORT ConeInstance instance();
        AGXPHYSICS_EXPORT const ConeInstance instance() const;

        AGXPHYSICS_EXPORT ConeSemantics* operator->();
        AGXPHYSICS_EXPORT const ConeSemantics* operator->() const;

        ConeData* getData();
        const ConeData* getData() const;


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

        DOXYGEN_START_INTERNAL_BLOCK()
        // Only used by tolua!
        #ifdef AGX_GENERATED_TOLUA_WRAPPER
        inline agx::Real const& getHeight() const { return (*this)->getHeight(); }
        inline agx::Real const& getTopRadius() const { return (*this)->getTopRadius(); }
        inline agx::Real const& getBottomRadius() const { return (*this)->getBottomRadius(); }


        inline void setHeight(agx::Real const& value) { (*this)->setHeight(value); }
        inline void setTopRadius(agx::Real const& value) { (*this)->setTopRadius(value); }
        inline void setBottomRadius(agx::Real const& value) { (*this)->setBottomRadius(value); }
        #endif
        DOXYGEN_END_INTERNAL_BLOCK()

      };

      // Entity is Referenced
      typedef agxData::EntityRef< ConePtr > ConeRef;


      DOXYGEN_START_INTERNAL_BLOCK()
      class AGXPHYSICS_EXPORT ConeInstance : public agx::Physics::Geometry::ShapeInstance
      {
      public:
        ConeInstance();
        ConeInstance(ConeData* data, agx::Index index);
        ConeInstance(agxData::EntityStorage *storage, agx::Index index);
        ConeInstance(const agxData::EntityInstance& other);
        ConeInstance(const agxData::EntityPtr& ptr);

        ConeData* getData();
        const ConeData* getData() const;

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

      };
      DOXYGEN_END_INTERNAL_BLOCK()



      typedef agx::VectorPOD<ConePtr> ConePtrVector;
      typedef agxData::Array<ConePtr> ConePtrArray;



      DOXYGEN_START_INTERNAL_BLOCK()
      /* Implementation */
      //-----------------------------------------------------------------------------------------------------
      //-----------------------------------------------------------------------------------------------------
      inline ConeInstance agx::Physics::Geometry::ConeData::operator[] (size_t index) { return ConeInstance(this, (agx::Index)index); }
      //-----------------------------------------------------------------------------------------------------
      AGX_FORCE_INLINE ConePtr::ConePtr() {}
      AGX_FORCE_INLINE ConePtr::ConePtr(agxData::EntityStorage* storage, agx::Index id) : agx::Physics::Geometry::ShapePtr(storage, id) {}
      AGX_FORCE_INLINE ConePtr::ConePtr(const agxData::EntityPtr& ptr) : agx::Physics::Geometry::ShapePtr(ptr)
      {
        agxAssertN(!ptr || ptr.isInstanceOf(ConeModel::instance()),
          "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
          EntityPtr::getModel()->fullPath().c_str(), ConeModel::instance()->fullPath().c_str());
      }

      AGX_FORCE_INLINE ConePtr::ConePtr(const agxData::EntityInstance& instance) : agx::Physics::Geometry::ShapePtr(instance)
      {
        agxAssertN(!instance || instance.isInstanceOf(ConeModel::instance()),
          "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
          EntityPtr::getModel()->fullPath().c_str(), ConeModel::instance()->fullPath().c_str());
      }

      AGX_FORCE_INLINE ConePtr& ConePtr::operator= (const agxData::EntityPtr& ptr)
      {
        agxData::EntityPtr::operator= (ptr);
        agxAssertN(!ptr || ptr.isInstanceOf(ConeModel::instance()),
          "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
          EntityPtr::getModel()->fullPath().c_str(), ConeModel::instance()->fullPath().c_str());
        return *this;
      }

      AGX_FORCE_INLINE ConePtr& ConePtr::operator= (const agxData::EntityInstance& instance)
      {
        agxData::EntityPtr::operator= (instance);
        agxAssertN(!instance || instance.isInstanceOf(ConeModel::instance()),
          "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
          EntityPtr::getModel()->fullPath().c_str(), ConeModel::instance()->fullPath().c_str());
        return *this;
      }

      inline ConeInstance ConePtr::instance() { return agxData::EntityPtr::instance(); }
      inline const ConeInstance ConePtr::instance() const { return agxData::EntityPtr::instance(); }
      AGX_FORCE_INLINE ConeSemantics* ConePtr::operator->() { return (ConeSemantics* )this; }
      AGX_FORCE_INLINE const ConeSemantics* ConePtr::operator->() const { return (const ConeSemantics* )this; }
      AGX_FORCE_INLINE ConeData* ConePtr::getData() { return static_cast<ConeData* >(agxData::EntityPtr::getData()); }
      AGX_FORCE_INLINE const ConeData* ConePtr::getData() const { return static_cast<const ConeData* >(agxData::EntityPtr::getData()); }

      AGX_FORCE_INLINE agx::Real& ConePtr::height() { verifyIndex(); return getData()->height[calculateIndex()]; }
      AGX_FORCE_INLINE agx::Real const& ConePtr::height() const { verifyIndex(); return getData()->height[calculateIndex()]; }

      AGX_FORCE_INLINE agx::Real& ConePtr::topRadius() { verifyIndex(); return getData()->topRadius[calculateIndex()]; }
      AGX_FORCE_INLINE agx::Real const& ConePtr::topRadius() const { verifyIndex(); return getData()->topRadius[calculateIndex()]; }

      AGX_FORCE_INLINE agx::Real& ConePtr::bottomRadius() { verifyIndex(); return getData()->bottomRadius[calculateIndex()]; }
      AGX_FORCE_INLINE agx::Real const& ConePtr::bottomRadius() const { verifyIndex(); return getData()->bottomRadius[calculateIndex()]; }

      //-----------------------------------------------------------------------------------------------------
      AGX_FORCE_INLINE ConeInstance::ConeInstance() {}
      AGX_FORCE_INLINE ConeInstance::ConeInstance(ConeData* data, agx::Index index) : agx::Physics::Geometry::ShapeInstance(data, index) {}
      AGX_FORCE_INLINE ConeInstance::ConeInstance(agxData::EntityStorage* storage, agx::Index index) : agx::Physics::Geometry::ShapeInstance(storage, index) {}
      AGX_FORCE_INLINE ConeInstance::ConeInstance(const agxData::EntityInstance& other) : agx::Physics::Geometry::ShapeInstance(other)
      {
        agxAssertN(!other || other.isInstanceOf(ConeModel::instance()),
          "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
          EntityInstance::getModel()->fullPath().c_str(), ConeModel::instance()->fullPath().c_str());
      }

      AGX_FORCE_INLINE ConeInstance::ConeInstance(const agxData::EntityPtr& ptr) : agx::Physics::Geometry::ShapeInstance(ptr)
      {
        agxAssertN(!ptr || ptr.isInstanceOf(ConeModel::instance()),
          "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
          EntityInstance::getModel()->fullPath().c_str(), ConeModel::instance()->fullPath().c_str());
      }


      AGX_FORCE_INLINE ConeData* ConeInstance::getData() { return static_cast<ConeData* >(agxData::EntityInstance::getData()); }
      AGX_FORCE_INLINE const ConeData* ConeInstance::getData() const { return static_cast<const ConeData* >(agxData::EntityInstance::getData()); }

      AGX_FORCE_INLINE agx::Real& ConeInstance::height() { verifyIndex(); return getData()->height[getIndex()]; }
      AGX_FORCE_INLINE agx::Real const& ConeInstance::height() const { verifyIndex(); return getData()->height[getIndex()]; }

      AGX_FORCE_INLINE agx::Real& ConeInstance::topRadius() { verifyIndex(); return getData()->topRadius[getIndex()]; }
      AGX_FORCE_INLINE agx::Real const& ConeInstance::topRadius() const { verifyIndex(); return getData()->topRadius[getIndex()]; }

      AGX_FORCE_INLINE agx::Real& ConeInstance::bottomRadius() { verifyIndex(); return getData()->bottomRadius[getIndex()]; }
      AGX_FORCE_INLINE agx::Real const& ConeInstance::bottomRadius() const { verifyIndex(); return getData()->bottomRadius[getIndex()]; }

      //-----------------------------------------------------------------------------------------------------
      AGX_FORCE_INLINE ConeSemantics::ConeSemantics() {}
      //-----------------------------------------------------------------------------------------------------
      DOXYGEN_END_INTERNAL_BLOCK()
    }
  }
}

AGX_TYPE_BINDING(agx::Physics::Geometry::ConePtr, "Physics.Geometry.ConePtr")
AGX_TYPE_BINDING(agx::Physics::Geometry::ConeInstance, "Physics.Geometry.ConeInstance")

#ifdef _MSC_VER
# pragma warning(pop)
#endif

#undef AGX_ENTITY_WRAPPER
#undef AGX_ENTITY_NAMESPACE
#endif

