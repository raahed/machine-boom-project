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

#ifndef GENERATED_AGX_PHYSICS_GEOMETRY_TRIANGLE_H_PLUGIN
#define GENERATED_AGX_PHYSICS_GEOMETRY_TRIANGLE_H_PLUGIN

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
#include <agx/Vec3.h>
#include <agx/Vec4.h>
#include <agx/Integer.h>
#include <agx/Real.h>


namespace agx
{
  namespace Physics
  {
    namespace Geometry
    {

      class TriangleModel;
      class TriangleData;
      class TrianglePtr;
      class TriangleInstance;
      class TriangleSemantics;


      AGX_DECLARE_POINTER_TYPES(TriangleModel);

      /** 
      Abstract description of the data attributes for the Physics.Geometry.Triangle entity.
      */ 
      class AGXPHYSICS_EXPORT TriangleModel : public agxData::EntityModel
      {
      public:
        typedef TrianglePtr PtrT;

        TriangleModel(const agx::String& name = "Triangle");

        /// \return The entity model singleton.
        static TriangleModel* instance();

        /// Create and return a pointer to a new instance in the default storage for this entity model.
        static TrianglePtr createInstance();

        /// \return The default storage for this entity model.
        static agxData::EntityStorage* defaultStorage();

        /// This is part of internal cleanup and should not be called by users
        virtual void shutdownCleanup() override;



        /* Attributes */
        static agxData::ScalarAttributeT< agx::Vec3f >* verticesAttribute;
        static const size_t verticesArraySize = 3;
        static agxData::ScalarAttributeT< agx::Vec3f >* normalsAttribute;
        static const size_t normalsArraySize = 3;
        static agxData::ScalarAttributeT< agx::Vec4f >* colorsAttribute;
        static const size_t colorsArraySize = 3;
        static agxData::ScalarAttributeT< agx::UInt32 >* indicesAttribute;
        static const size_t indicesArraySize = 3;
        static agxData::ScalarAttributeT< agx::Real >* cameraDistanceAttribute;

      protected:
        virtual ~TriangleModel();
        virtual agxData::EntityData* createData(agxData::EntityStorage* storage) override;
        virtual void configure(agx::TiXmlElement* eEntity) override;
        virtual void initAttributeAccessors() override;
        virtual void construct(agxData::EntityPtr instance) override;
        void construct(agx::Physics::Geometry::TrianglePtr triangle);
      };


      DOXYGEN_START_INTERNAL_BLOCK()
      #ifndef AGX_PHYSICS_GEOMETRY_TRIANGLE_DATA_SET_OVERRIDE
      #define AGX_PHYSICS_GEOMETRY_TRIANGLE_DATA_SET
      class AGXPHYSICS_EXPORT TriangleData : public agxData::EntityData
      {
      public:
        TriangleInstance operator[] (size_t index);

      public:
        agxData::Array< TrianglePtr >& instance;
        agxData::Array< agx::Vec3f > vertices;
        static const size_t verticesArraySize = 3;
        agxData::Array< agx::Vec3f > normals;
        static const size_t normalsArraySize = 3;
        agxData::Array< agx::Vec4f > colors;
        static const size_t colorsArraySize = 3;
        agxData::Array< agx::UInt32 > indices;
        static const size_t indicesArraySize = 3;
        agxData::Array< agx::Real > cameraDistance;

      public:
        typedef agx::Vec3f verticesType;
        typedef agx::Vec3f normalsType;
        typedef agx::Vec4f colorsType;
        typedef agx::UInt32 indicesType;
        typedef agx::Real cameraDistanceType;

      public:
        TriangleData(agxData::EntityStorage* storage);
        TriangleData();

      protected:
        virtual ~TriangleData() {}
        virtual void setNumElements(agx::Index numElements) override;

      private:
        TriangleData& operator= (const TriangleData&) { return *this; }

      };
      #endif
      DOXYGEN_END_INTERNAL_BLOCK()


      DOXYGEN_START_INTERNAL_BLOCK()
      class AGXPHYSICS_EXPORT TriangleSemantics : protected agxData::EntityPtr
      {
      public:

        // Automatic getters
        agx::Real const& getCameraDistance() const;

        // Semantics defined by explicit kernels

        // Automatic setters
        void setCameraDistance(agx::Real const& value);


      protected:
        friend class TrianglePtr;
        friend class TriangleInstance;
        TriangleSemantics();
      };
      DOXYGEN_END_INTERNAL_BLOCK()


      /**
      Pointer to a entity instance of type Physics.Geometry.Triangle
      */
      class CALLABLE TrianglePtr : public agxData::EntityPtr
      {
      public:
        typedef TriangleModel ModelType;
        typedef TriangleData DataType;
        typedef TriangleInstance InstanceType;

      public:
        AGXPHYSICS_EXPORT TrianglePtr();
        AGXPHYSICS_EXPORT TrianglePtr(agxData::EntityStorage* storage, agx::Index id);
        AGXPHYSICS_EXPORT TrianglePtr(const agxData::EntityPtr& ptr);
        AGXPHYSICS_EXPORT TrianglePtr(const agxData::EntityInstance& instance);
        AGXPHYSICS_EXPORT TrianglePtr& operator= (const agxData::EntityPtr& ptr);
        AGXPHYSICS_EXPORT TrianglePtr& operator= (const agxData::EntityInstance& instance);
        AGXPHYSICS_EXPORT TriangleInstance instance();
        AGXPHYSICS_EXPORT const TriangleInstance instance() const;

        AGXPHYSICS_EXPORT TriangleSemantics* operator->();
        AGXPHYSICS_EXPORT const TriangleSemantics* operator->() const;

        TriangleData* getData();
        const TriangleData* getData() const;


        AGXPHYSICS_EXPORT agxData::Array< agx::Vec3f > vertices();
        AGXPHYSICS_EXPORT agxData::Array< agx::Vec3f > const vertices() const;
        AGXPHYSICS_EXPORT agx::Vec3f& vertices(size_t index);
        AGXPHYSICS_EXPORT agx::Vec3f const& vertices(size_t index) const;

        AGXPHYSICS_EXPORT agxData::Array< agx::Vec3f > normals();
        AGXPHYSICS_EXPORT agxData::Array< agx::Vec3f > const normals() const;
        AGXPHYSICS_EXPORT agx::Vec3f& normals(size_t index);
        AGXPHYSICS_EXPORT agx::Vec3f const& normals(size_t index) const;

        AGXPHYSICS_EXPORT agxData::Array< agx::Vec4f > colors();
        AGXPHYSICS_EXPORT agxData::Array< agx::Vec4f > const colors() const;
        AGXPHYSICS_EXPORT agx::Vec4f& colors(size_t index);
        AGXPHYSICS_EXPORT agx::Vec4f const& colors(size_t index) const;

        AGXPHYSICS_EXPORT agxData::Array< agx::UInt32 > indices();
        AGXPHYSICS_EXPORT agxData::Array< agx::UInt32 > const indices() const;
        AGXPHYSICS_EXPORT agx::UInt32& indices(size_t index);
        AGXPHYSICS_EXPORT agx::UInt32 const& indices(size_t index) const;

        /// \return reference to the cameraDistance attribute
        AGXPHYSICS_EXPORT agx::Real& cameraDistance();
        /// \return const reference to the cameraDistance attribute
        AGXPHYSICS_EXPORT agx::Real const& cameraDistance() const;

        DOXYGEN_START_INTERNAL_BLOCK()
        // Only used by tolua!
        #ifdef AGX_GENERATED_TOLUA_WRAPPER
        inline agx::Real const& getCameraDistance() const { return (*this)->getCameraDistance(); }


        inline void setCameraDistance(agx::Real const& value) { (*this)->setCameraDistance(value); }
        #endif
        DOXYGEN_END_INTERNAL_BLOCK()

      };


      DOXYGEN_START_INTERNAL_BLOCK()
      class AGXPHYSICS_EXPORT TriangleInstance : public agxData::EntityInstance
      {
      public:
        TriangleInstance();
        TriangleInstance(TriangleData* data, agx::Index index);
        TriangleInstance(agxData::EntityStorage *storage, agx::Index index);
        TriangleInstance(const agxData::EntityInstance& other);
        TriangleInstance(const agxData::EntityPtr& ptr);

        TriangleData* getData();
        const TriangleData* getData() const;

      public:
        agxData::Array< agx::Vec3f > vertices();
        agxData::Array< agx::Vec3f > const vertices() const;
        agx::Vec3f& vertices(size_t index);
        agx::Vec3f const& vertices(size_t index) const;

        agxData::Array< agx::Vec3f > normals();
        agxData::Array< agx::Vec3f > const normals() const;
        agx::Vec3f& normals(size_t index);
        agx::Vec3f const& normals(size_t index) const;

        agxData::Array< agx::Vec4f > colors();
        agxData::Array< agx::Vec4f > const colors() const;
        agx::Vec4f& colors(size_t index);
        agx::Vec4f const& colors(size_t index) const;

        agxData::Array< agx::UInt32 > indices();
        agxData::Array< agx::UInt32 > const indices() const;
        agx::UInt32& indices(size_t index);
        agx::UInt32 const& indices(size_t index) const;

        /// \return reference to the cameraDistance attribute
        agx::Real& cameraDistance();
        /// \return const reference to the cameraDistance attribute
        agx::Real const& cameraDistance() const;

      };
      DOXYGEN_END_INTERNAL_BLOCK()



      typedef agx::VectorPOD<TrianglePtr> TrianglePtrVector;
      typedef agxData::Array<TrianglePtr> TrianglePtrArray;



      DOXYGEN_START_INTERNAL_BLOCK()
      /* Implementation */
      //-----------------------------------------------------------------------------------------------------
      //-----------------------------------------------------------------------------------------------------
      inline TriangleInstance agx::Physics::Geometry::TriangleData::operator[] (size_t index) { return TriangleInstance(this, (agx::Index)index); }
      //-----------------------------------------------------------------------------------------------------
      AGX_FORCE_INLINE TrianglePtr::TrianglePtr() {}
      AGX_FORCE_INLINE TrianglePtr::TrianglePtr(agxData::EntityStorage* storage, agx::Index id) : agxData::EntityPtr(storage, id) {}
      AGX_FORCE_INLINE TrianglePtr::TrianglePtr(const agxData::EntityPtr& ptr) : agxData::EntityPtr(ptr)
      {
        agxAssertN(!ptr || ptr.isInstanceOf(TriangleModel::instance()),
          "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
          EntityPtr::getModel()->fullPath().c_str(), TriangleModel::instance()->fullPath().c_str());
      }

      AGX_FORCE_INLINE TrianglePtr::TrianglePtr(const agxData::EntityInstance& instance) : agxData::EntityPtr(instance)
      {
        agxAssertN(!instance || instance.isInstanceOf(TriangleModel::instance()),
          "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
          EntityPtr::getModel()->fullPath().c_str(), TriangleModel::instance()->fullPath().c_str());
      }

      AGX_FORCE_INLINE TrianglePtr& TrianglePtr::operator= (const agxData::EntityPtr& ptr)
      {
        agxData::EntityPtr::operator= (ptr);
        agxAssertN(!ptr || ptr.isInstanceOf(TriangleModel::instance()),
          "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
          EntityPtr::getModel()->fullPath().c_str(), TriangleModel::instance()->fullPath().c_str());
        return *this;
      }

      AGX_FORCE_INLINE TrianglePtr& TrianglePtr::operator= (const agxData::EntityInstance& instance)
      {
        agxData::EntityPtr::operator= (instance);
        agxAssertN(!instance || instance.isInstanceOf(TriangleModel::instance()),
          "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
          EntityPtr::getModel()->fullPath().c_str(), TriangleModel::instance()->fullPath().c_str());
        return *this;
      }

      inline TriangleInstance TrianglePtr::instance() { return agxData::EntityPtr::instance(); }
      inline const TriangleInstance TrianglePtr::instance() const { return agxData::EntityPtr::instance(); }
      AGX_FORCE_INLINE TriangleSemantics* TrianglePtr::operator->() { return (TriangleSemantics* )this; }
      AGX_FORCE_INLINE const TriangleSemantics* TrianglePtr::operator->() const { return (const TriangleSemantics* )this; }
      AGX_FORCE_INLINE TriangleData* TrianglePtr::getData() { return static_cast<TriangleData* >(agxData::EntityPtr::getData()); }
      AGX_FORCE_INLINE const TriangleData* TrianglePtr::getData() const { return static_cast<const TriangleData* >(agxData::EntityPtr::getData()); }

      AGX_FORCE_INLINE agxData::Array< agx::Vec3f > TrianglePtr::vertices() { verifyIndex(); return getData()->vertices.slice(agx::IndexRange32(calculateIndex() * (agx::Index)TriangleData::verticesArraySize, (calculateIndex()+1) * (agx::Index)TriangleData::verticesArraySize)); }
      AGX_FORCE_INLINE const agxData::Array< agx::Vec3f > TrianglePtr::vertices() const { verifyIndex(); return getData()->vertices.slice(agx::IndexRange32(calculateIndex() * (agx::Index)TriangleData::verticesArraySize, (calculateIndex()+1) * (agx::Index)TriangleData::verticesArraySize)); }
      AGX_FORCE_INLINE agx::Vec3f& TrianglePtr::vertices(size_t index) { return this->vertices()[index]; }
      AGX_FORCE_INLINE agx::Vec3f const& TrianglePtr::vertices(size_t index) const { return this->vertices()[index]; }

      AGX_FORCE_INLINE agxData::Array< agx::Vec3f > TrianglePtr::normals() { verifyIndex(); return getData()->normals.slice(agx::IndexRange32(calculateIndex() * (agx::Index)TriangleData::normalsArraySize, (calculateIndex()+1) * (agx::Index)TriangleData::normalsArraySize)); }
      AGX_FORCE_INLINE const agxData::Array< agx::Vec3f > TrianglePtr::normals() const { verifyIndex(); return getData()->normals.slice(agx::IndexRange32(calculateIndex() * (agx::Index)TriangleData::normalsArraySize, (calculateIndex()+1) * (agx::Index)TriangleData::normalsArraySize)); }
      AGX_FORCE_INLINE agx::Vec3f& TrianglePtr::normals(size_t index) { return this->normals()[index]; }
      AGX_FORCE_INLINE agx::Vec3f const& TrianglePtr::normals(size_t index) const { return this->normals()[index]; }

      AGX_FORCE_INLINE agxData::Array< agx::Vec4f > TrianglePtr::colors() { verifyIndex(); return getData()->colors.slice(agx::IndexRange32(calculateIndex() * (agx::Index)TriangleData::colorsArraySize, (calculateIndex()+1) * (agx::Index)TriangleData::colorsArraySize)); }
      AGX_FORCE_INLINE const agxData::Array< agx::Vec4f > TrianglePtr::colors() const { verifyIndex(); return getData()->colors.slice(agx::IndexRange32(calculateIndex() * (agx::Index)TriangleData::colorsArraySize, (calculateIndex()+1) * (agx::Index)TriangleData::colorsArraySize)); }
      AGX_FORCE_INLINE agx::Vec4f& TrianglePtr::colors(size_t index) { return this->colors()[index]; }
      AGX_FORCE_INLINE agx::Vec4f const& TrianglePtr::colors(size_t index) const { return this->colors()[index]; }

      AGX_FORCE_INLINE agxData::Array< agx::UInt32 > TrianglePtr::indices() { verifyIndex(); return getData()->indices.slice(agx::IndexRange32(calculateIndex() * (agx::Index)TriangleData::indicesArraySize, (calculateIndex()+1) * (agx::Index)TriangleData::indicesArraySize)); }
      AGX_FORCE_INLINE const agxData::Array< agx::UInt32 > TrianglePtr::indices() const { verifyIndex(); return getData()->indices.slice(agx::IndexRange32(calculateIndex() * (agx::Index)TriangleData::indicesArraySize, (calculateIndex()+1) * (agx::Index)TriangleData::indicesArraySize)); }
      AGX_FORCE_INLINE agx::UInt32& TrianglePtr::indices(size_t index) { return this->indices()[index]; }
      AGX_FORCE_INLINE agx::UInt32 const& TrianglePtr::indices(size_t index) const { return this->indices()[index]; }

      AGX_FORCE_INLINE agx::Real& TrianglePtr::cameraDistance() { verifyIndex(); return getData()->cameraDistance[calculateIndex()]; }
      AGX_FORCE_INLINE agx::Real const& TrianglePtr::cameraDistance() const { verifyIndex(); return getData()->cameraDistance[calculateIndex()]; }

      //-----------------------------------------------------------------------------------------------------
      AGX_FORCE_INLINE TriangleInstance::TriangleInstance() {}
      AGX_FORCE_INLINE TriangleInstance::TriangleInstance(TriangleData* data, agx::Index index) : agxData::EntityInstance(data, index) {}
      AGX_FORCE_INLINE TriangleInstance::TriangleInstance(agxData::EntityStorage* storage, agx::Index index) : agxData::EntityInstance(storage, index) {}
      AGX_FORCE_INLINE TriangleInstance::TriangleInstance(const agxData::EntityInstance& other) : agxData::EntityInstance(other)
      {
        agxAssertN(!other || other.isInstanceOf(TriangleModel::instance()),
          "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
          EntityInstance::getModel()->fullPath().c_str(), TriangleModel::instance()->fullPath().c_str());
      }

      AGX_FORCE_INLINE TriangleInstance::TriangleInstance(const agxData::EntityPtr& ptr) : agxData::EntityInstance(ptr)
      {
        agxAssertN(!ptr || ptr.isInstanceOf(TriangleModel::instance()),
          "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
          EntityInstance::getModel()->fullPath().c_str(), TriangleModel::instance()->fullPath().c_str());
      }


      AGX_FORCE_INLINE TriangleData* TriangleInstance::getData() { return static_cast<TriangleData* >(agxData::EntityInstance::getData()); }
      AGX_FORCE_INLINE const TriangleData* TriangleInstance::getData() const { return static_cast<const TriangleData* >(agxData::EntityInstance::getData()); }

      AGX_FORCE_INLINE agxData::Array< agx::Vec3f > TriangleInstance::vertices() { verifyIndex(); return getData()->vertices.slice(agx::IndexRange32(getIndex() * (agx::Index)TriangleData::verticesArraySize, (getIndex()+1) * (agx::Index)TriangleData::verticesArraySize)); }
      AGX_FORCE_INLINE const agxData::Array< agx::Vec3f > TriangleInstance::vertices() const { verifyIndex(); return getData()->vertices.slice(agx::IndexRange32(getIndex() * (agx::Index)TriangleData::verticesArraySize, (getIndex()+1) * (agx::Index)TriangleData::verticesArraySize)); }
      AGX_FORCE_INLINE agx::Vec3f& TriangleInstance::vertices(size_t index) { return this->vertices()[index]; }
      AGX_FORCE_INLINE agx::Vec3f const& TriangleInstance::vertices(size_t index) const { return this->vertices()[index]; }

      AGX_FORCE_INLINE agxData::Array< agx::Vec3f > TriangleInstance::normals() { verifyIndex(); return getData()->normals.slice(agx::IndexRange32(getIndex() * (agx::Index)TriangleData::normalsArraySize, (getIndex()+1) * (agx::Index)TriangleData::normalsArraySize)); }
      AGX_FORCE_INLINE const agxData::Array< agx::Vec3f > TriangleInstance::normals() const { verifyIndex(); return getData()->normals.slice(agx::IndexRange32(getIndex() * (agx::Index)TriangleData::normalsArraySize, (getIndex()+1) * (agx::Index)TriangleData::normalsArraySize)); }
      AGX_FORCE_INLINE agx::Vec3f& TriangleInstance::normals(size_t index) { return this->normals()[index]; }
      AGX_FORCE_INLINE agx::Vec3f const& TriangleInstance::normals(size_t index) const { return this->normals()[index]; }

      AGX_FORCE_INLINE agxData::Array< agx::Vec4f > TriangleInstance::colors() { verifyIndex(); return getData()->colors.slice(agx::IndexRange32(getIndex() * (agx::Index)TriangleData::colorsArraySize, (getIndex()+1) * (agx::Index)TriangleData::colorsArraySize)); }
      AGX_FORCE_INLINE const agxData::Array< agx::Vec4f > TriangleInstance::colors() const { verifyIndex(); return getData()->colors.slice(agx::IndexRange32(getIndex() * (agx::Index)TriangleData::colorsArraySize, (getIndex()+1) * (agx::Index)TriangleData::colorsArraySize)); }
      AGX_FORCE_INLINE agx::Vec4f& TriangleInstance::colors(size_t index) { return this->colors()[index]; }
      AGX_FORCE_INLINE agx::Vec4f const& TriangleInstance::colors(size_t index) const { return this->colors()[index]; }

      AGX_FORCE_INLINE agxData::Array< agx::UInt32 > TriangleInstance::indices() { verifyIndex(); return getData()->indices.slice(agx::IndexRange32(getIndex() * (agx::Index)TriangleData::indicesArraySize, (getIndex()+1) * (agx::Index)TriangleData::indicesArraySize)); }
      AGX_FORCE_INLINE const agxData::Array< agx::UInt32 > TriangleInstance::indices() const { verifyIndex(); return getData()->indices.slice(agx::IndexRange32(getIndex() * (agx::Index)TriangleData::indicesArraySize, (getIndex()+1) * (agx::Index)TriangleData::indicesArraySize)); }
      AGX_FORCE_INLINE agx::UInt32& TriangleInstance::indices(size_t index) { return this->indices()[index]; }
      AGX_FORCE_INLINE agx::UInt32 const& TriangleInstance::indices(size_t index) const { return this->indices()[index]; }

      AGX_FORCE_INLINE agx::Real& TriangleInstance::cameraDistance() { verifyIndex(); return getData()->cameraDistance[getIndex()]; }
      AGX_FORCE_INLINE agx::Real const& TriangleInstance::cameraDistance() const { verifyIndex(); return getData()->cameraDistance[getIndex()]; }

      //-----------------------------------------------------------------------------------------------------
      AGX_FORCE_INLINE TriangleSemantics::TriangleSemantics() {}
      //-----------------------------------------------------------------------------------------------------
      DOXYGEN_END_INTERNAL_BLOCK()
    }
  }
}

AGX_TYPE_BINDING(agx::Physics::Geometry::TrianglePtr, "Physics.Geometry.TrianglePtr")
AGX_TYPE_BINDING(agx::Physics::Geometry::TriangleInstance, "Physics.Geometry.TriangleInstance")

#ifdef _MSC_VER
# pragma warning(pop)
#endif

#undef AGX_ENTITY_WRAPPER
#undef AGX_ENTITY_NAMESPACE
#endif

