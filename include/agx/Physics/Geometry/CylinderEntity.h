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

#ifndef GENERATED_AGX_PHYSICS_GEOMETRY_CYLINDER_H_PLUGIN
#define GENERATED_AGX_PHYSICS_GEOMETRY_CYLINDER_H_PLUGIN

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

      class CylinderModel;
      class CylinderData;
      class CylinderPtr;
      class CylinderInstance;
      class CylinderSemantics;


      AGX_DECLARE_POINTER_TYPES(CylinderModel);

      /** 
      Abstract description of the data attributes for the Physics.Geometry.Cylinder entity.
      */ 
      class AGXPHYSICS_EXPORT CylinderModel : public agx::Physics::Geometry::ShapeModel
      {
      public:
        typedef CylinderPtr PtrT;

        CylinderModel(const agx::String& name = "Cylinder");

        /// \return The entity model singleton.
        static CylinderModel* instance();

        /// Create and return a pointer to a new instance in the default storage for this entity model.
        static CylinderPtr createInstance();

        /// \return The default storage for this entity model.
        static agxData::EntityStorage* defaultStorage();

        /// This is part of internal cleanup and should not be called by users
        virtual void shutdownCleanup() override;



        /* Attributes */
        static agxData::ScalarAttributeT< agx::Real >* heightAttribute;
        static agxData::ScalarAttributeT< agx::Real >* radiusAttribute;

      protected:
        virtual ~CylinderModel();
        virtual agxData::EntityData* createData(agxData::EntityStorage* storage) override;
        virtual void configure(agx::TiXmlElement* eEntity) override;
        virtual void initAttributeAccessors() override;
        virtual void construct(agxData::EntityPtr instance) override;
        void construct(agx::Physics::Geometry::CylinderPtr cylinder);
      };


      DOXYGEN_START_INTERNAL_BLOCK()
      #ifndef AGX_PHYSICS_GEOMETRY_CYLINDER_DATA_SET_OVERRIDE
      #define AGX_PHYSICS_GEOMETRY_CYLINDER_DATA_SET
      class AGXPHYSICS_EXPORT CylinderData : public agx::Physics::Geometry::ShapeData
      {
      public:
        CylinderInstance operator[] (size_t index);

      public:
        agxData::Array< CylinderPtr >& instance;
        agxData::Array< agx::Real > height;
        agxData::Array< agx::Real > radius;

      public:
        typedef agx::Real heightType;
        typedef agx::Real radiusType;

      public:
        CylinderData(agxData::EntityStorage* storage);
        CylinderData();

      protected:
        virtual ~CylinderData() {}
        virtual void setNumElements(agx::Index numElements) override;

      private:
        CylinderData& operator= (const CylinderData&) { return *this; }

      };
      #endif
      DOXYGEN_END_INTERNAL_BLOCK()


      DOXYGEN_START_INTERNAL_BLOCK()
      class AGXPHYSICS_EXPORT CylinderSemantics : public agx::Physics::Geometry::ShapeSemantics
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
        friend class CylinderPtr;
        friend class CylinderInstance;
        CylinderSemantics();
      };
      DOXYGEN_END_INTERNAL_BLOCK()


      /**
      Pointer to a entity instance of type Physics.Geometry.Cylinder
      */
      class CALLABLE CylinderPtr : public agx::Physics::Geometry::ShapePtr
      {
      public:
        typedef CylinderModel ModelType;
        typedef CylinderData DataType;
        typedef CylinderInstance InstanceType;

      public:
        AGXPHYSICS_EXPORT CylinderPtr();
        AGXPHYSICS_EXPORT CylinderPtr(agxData::EntityStorage* storage, agx::Index id);
        AGXPHYSICS_EXPORT CylinderPtr(const agxData::EntityPtr& ptr);
        AGXPHYSICS_EXPORT CylinderPtr(const agxData::EntityInstance& instance);
        AGXPHYSICS_EXPORT CylinderPtr& operator= (const agxData::EntityPtr& ptr);
        AGXPHYSICS_EXPORT CylinderPtr& operator= (const agxData::EntityInstance& instance);
        AGXPHYSICS_EXPORT CylinderInstance instance();
        AGXPHYSICS_EXPORT const CylinderInstance instance() const;

        AGXPHYSICS_EXPORT CylinderSemantics* operator->();
        AGXPHYSICS_EXPORT const CylinderSemantics* operator->() const;

        CylinderData* getData();
        const CylinderData* getData() const;


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
      typedef agxData::EntityRef< CylinderPtr > CylinderRef;


      DOXYGEN_START_INTERNAL_BLOCK()
      class AGXPHYSICS_EXPORT CylinderInstance : public agx::Physics::Geometry::ShapeInstance
      {
      public:
        CylinderInstance();
        CylinderInstance(CylinderData* data, agx::Index index);
        CylinderInstance(agxData::EntityStorage *storage, agx::Index index);
        CylinderInstance(const agxData::EntityInstance& other);
        CylinderInstance(const agxData::EntityPtr& ptr);

        CylinderData* getData();
        const CylinderData* getData() const;

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



      typedef agx::VectorPOD<CylinderPtr> CylinderPtrVector;
      typedef agxData::Array<CylinderPtr> CylinderPtrArray;



      DOXYGEN_START_INTERNAL_BLOCK()
      /* Implementation */
      //-----------------------------------------------------------------------------------------------------
      //-----------------------------------------------------------------------------------------------------
      inline CylinderInstance agx::Physics::Geometry::CylinderData::operator[] (size_t index) { return CylinderInstance(this, (agx::Index)index); }
      //-----------------------------------------------------------------------------------------------------
      AGX_FORCE_INLINE CylinderPtr::CylinderPtr() {}
      AGX_FORCE_INLINE CylinderPtr::CylinderPtr(agxData::EntityStorage* storage, agx::Index id) : agx::Physics::Geometry::ShapePtr(storage, id) {}
      AGX_FORCE_INLINE CylinderPtr::CylinderPtr(const agxData::EntityPtr& ptr) : agx::Physics::Geometry::ShapePtr(ptr)
      {
        agxAssertN(!ptr || ptr.isInstanceOf(CylinderModel::instance()),
          "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
          EntityPtr::getModel()->fullPath().c_str(), CylinderModel::instance()->fullPath().c_str());
      }

      AGX_FORCE_INLINE CylinderPtr::CylinderPtr(const agxData::EntityInstance& instance) : agx::Physics::Geometry::ShapePtr(instance)
      {
        agxAssertN(!instance || instance.isInstanceOf(CylinderModel::instance()),
          "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
          EntityPtr::getModel()->fullPath().c_str(), CylinderModel::instance()->fullPath().c_str());
      }

      AGX_FORCE_INLINE CylinderPtr& CylinderPtr::operator= (const agxData::EntityPtr& ptr)
      {
        agxData::EntityPtr::operator= (ptr);
        agxAssertN(!ptr || ptr.isInstanceOf(CylinderModel::instance()),
          "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
          EntityPtr::getModel()->fullPath().c_str(), CylinderModel::instance()->fullPath().c_str());
        return *this;
      }

      AGX_FORCE_INLINE CylinderPtr& CylinderPtr::operator= (const agxData::EntityInstance& instance)
      {
        agxData::EntityPtr::operator= (instance);
        agxAssertN(!instance || instance.isInstanceOf(CylinderModel::instance()),
          "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
          EntityPtr::getModel()->fullPath().c_str(), CylinderModel::instance()->fullPath().c_str());
        return *this;
      }

      inline CylinderInstance CylinderPtr::instance() { return agxData::EntityPtr::instance(); }
      inline const CylinderInstance CylinderPtr::instance() const { return agxData::EntityPtr::instance(); }
      AGX_FORCE_INLINE CylinderSemantics* CylinderPtr::operator->() { return (CylinderSemantics* )this; }
      AGX_FORCE_INLINE const CylinderSemantics* CylinderPtr::operator->() const { return (const CylinderSemantics* )this; }
      AGX_FORCE_INLINE CylinderData* CylinderPtr::getData() { return static_cast<CylinderData* >(agxData::EntityPtr::getData()); }
      AGX_FORCE_INLINE const CylinderData* CylinderPtr::getData() const { return static_cast<const CylinderData* >(agxData::EntityPtr::getData()); }

      AGX_FORCE_INLINE agx::Real& CylinderPtr::height() { verifyIndex(); return getData()->height[calculateIndex()]; }
      AGX_FORCE_INLINE agx::Real const& CylinderPtr::height() const { verifyIndex(); return getData()->height[calculateIndex()]; }

      AGX_FORCE_INLINE agx::Real& CylinderPtr::radius() { verifyIndex(); return getData()->radius[calculateIndex()]; }
      AGX_FORCE_INLINE agx::Real const& CylinderPtr::radius() const { verifyIndex(); return getData()->radius[calculateIndex()]; }

      //-----------------------------------------------------------------------------------------------------
      AGX_FORCE_INLINE CylinderInstance::CylinderInstance() {}
      AGX_FORCE_INLINE CylinderInstance::CylinderInstance(CylinderData* data, agx::Index index) : agx::Physics::Geometry::ShapeInstance(data, index) {}
      AGX_FORCE_INLINE CylinderInstance::CylinderInstance(agxData::EntityStorage* storage, agx::Index index) : agx::Physics::Geometry::ShapeInstance(storage, index) {}
      AGX_FORCE_INLINE CylinderInstance::CylinderInstance(const agxData::EntityInstance& other) : agx::Physics::Geometry::ShapeInstance(other)
      {
        agxAssertN(!other || other.isInstanceOf(CylinderModel::instance()),
          "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
          EntityInstance::getModel()->fullPath().c_str(), CylinderModel::instance()->fullPath().c_str());
      }

      AGX_FORCE_INLINE CylinderInstance::CylinderInstance(const agxData::EntityPtr& ptr) : agx::Physics::Geometry::ShapeInstance(ptr)
      {
        agxAssertN(!ptr || ptr.isInstanceOf(CylinderModel::instance()),
          "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
          EntityInstance::getModel()->fullPath().c_str(), CylinderModel::instance()->fullPath().c_str());
      }


      AGX_FORCE_INLINE CylinderData* CylinderInstance::getData() { return static_cast<CylinderData* >(agxData::EntityInstance::getData()); }
      AGX_FORCE_INLINE const CylinderData* CylinderInstance::getData() const { return static_cast<const CylinderData* >(agxData::EntityInstance::getData()); }

      AGX_FORCE_INLINE agx::Real& CylinderInstance::height() { verifyIndex(); return getData()->height[getIndex()]; }
      AGX_FORCE_INLINE agx::Real const& CylinderInstance::height() const { verifyIndex(); return getData()->height[getIndex()]; }

      AGX_FORCE_INLINE agx::Real& CylinderInstance::radius() { verifyIndex(); return getData()->radius[getIndex()]; }
      AGX_FORCE_INLINE agx::Real const& CylinderInstance::radius() const { verifyIndex(); return getData()->radius[getIndex()]; }

      //-----------------------------------------------------------------------------------------------------
      AGX_FORCE_INLINE CylinderSemantics::CylinderSemantics() {}
      //-----------------------------------------------------------------------------------------------------
      DOXYGEN_END_INTERNAL_BLOCK()
    }
  }
}

AGX_TYPE_BINDING(agx::Physics::Geometry::CylinderPtr, "Physics.Geometry.CylinderPtr")
AGX_TYPE_BINDING(agx::Physics::Geometry::CylinderInstance, "Physics.Geometry.CylinderInstance")

#ifdef _MSC_VER
# pragma warning(pop)
#endif

#undef AGX_ENTITY_WRAPPER
#undef AGX_ENTITY_NAMESPACE
#endif

