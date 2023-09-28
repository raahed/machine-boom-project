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

#ifndef GENERATED_AGX_PHYSICS_GEOMETRY_SHAPEGROUP_H_PLUGIN
#define GENERATED_AGX_PHYSICS_GEOMETRY_SHAPEGROUP_H_PLUGIN

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
#include <agx/Integer.h>
#include <agx/Physics/Geometry/ShapeEntity.h>
#include <agx/AffineMatrix4x4.h>

namespace agx { namespace Physics { namespace Geometry { class ShapePtr; }}}

namespace agx
{
  namespace Physics
  {
    namespace Geometry
    {

      class ShapeGroupModel;
      class ShapeGroupData;
      class ShapeGroupPtr;
      class ShapeGroupInstance;
      class ShapeGroupSemantics;


      AGX_DECLARE_POINTER_TYPES(ShapeGroupModel);

      /** 
      Abstract description of the data attributes for the Physics.Geometry.ShapeGroup entity.
      */ 
      class AGXPHYSICS_EXPORT ShapeGroupModel : public agx::Physics::Geometry::ShapeModel
      {
      public:
        typedef ShapeGroupPtr PtrT;

        ShapeGroupModel(const agx::String& name = "ShapeGroup");

        /// \return The entity model singleton.
        static ShapeGroupModel* instance();

        /// Create and return a pointer to a new instance in the default storage for this entity model.
        static ShapeGroupPtr createInstance();

        /// \return The default storage for this entity model.
        static agxData::EntityStorage* defaultStorage();

        /// This is part of internal cleanup and should not be called by users
        virtual void shutdownCleanup() override;



        /* Attributes */
        static agxData::ScalarAttributeT< agx::UInt >* tagAttribute;
        static agxData::ScalarAttributeT< agx::Vector< agx::Physics::Geometry::ShapePtr > >* childShapesAttribute;
        static agxData::ScalarAttributeT< agx::Vector< agx::AffineMatrix4x4 > >* childTransformsAttribute;

      protected:
        virtual ~ShapeGroupModel();
        virtual agxData::EntityData* createData(agxData::EntityStorage* storage) override;
        virtual void configure(agx::TiXmlElement* eEntity) override;
        virtual void initAttributeAccessors() override;
        virtual void construct(agxData::EntityPtr instance) override;
        void construct(agx::Physics::Geometry::ShapeGroupPtr shapeGroup);
      };


      DOXYGEN_START_INTERNAL_BLOCK()
      #ifndef AGX_PHYSICS_GEOMETRY_SHAPEGROUP_DATA_SET_OVERRIDE
      #define AGX_PHYSICS_GEOMETRY_SHAPEGROUP_DATA_SET
      class AGXPHYSICS_EXPORT ShapeGroupData : public agx::Physics::Geometry::ShapeData
      {
      public:
        ShapeGroupInstance operator[] (size_t index);

      public:
        agxData::Array< ShapeGroupPtr >& instance;
        agxData::Array< agx::UInt > tag;
        agxData::Array< agx::Vector< agx::Physics::Geometry::ShapePtr > > childShapes;
        agxData::Array< agx::Vector< agx::AffineMatrix4x4 > > childTransforms;

      public:
        typedef agx::UInt tagType;
        typedef agx::Vector< agx::Physics::Geometry::ShapePtr > childShapesType;
        typedef agx::Vector< agx::AffineMatrix4x4 > childTransformsType;

      public:
        ShapeGroupData(agxData::EntityStorage* storage);
        ShapeGroupData();

      protected:
        virtual ~ShapeGroupData() {}
        virtual void setNumElements(agx::Index numElements) override;

      private:
        ShapeGroupData& operator= (const ShapeGroupData&) { return *this; }

      };
      #endif
      DOXYGEN_END_INTERNAL_BLOCK()


      DOXYGEN_START_INTERNAL_BLOCK()
      class AGXPHYSICS_EXPORT ShapeGroupSemantics : public agx::Physics::Geometry::ShapeSemantics
      {
      public:

        // Automatic getters
        agx::UInt const& getTag() const;
        agx::Vector< agx::Physics::Geometry::ShapePtr > const& getChildShapes() const;
        agx::Vector< agx::AffineMatrix4x4 > const& getChildTransforms() const;

        // Semantics defined by explicit kernels

        // Automatic setters
        void setTag(agx::UInt const& value);
        void setChildShapes(agx::Vector< agx::Physics::Geometry::ShapePtr > const& value);
        void setChildTransforms(agx::Vector< agx::AffineMatrix4x4 > const& value);


      protected:
        friend class ShapeGroupPtr;
        friend class ShapeGroupInstance;
        ShapeGroupSemantics();
      };
      DOXYGEN_END_INTERNAL_BLOCK()


      /**
      Pointer to a entity instance of type Physics.Geometry.ShapeGroup
      */
      class CALLABLE ShapeGroupPtr : public agx::Physics::Geometry::ShapePtr
      {
      public:
        typedef ShapeGroupModel ModelType;
        typedef ShapeGroupData DataType;
        typedef ShapeGroupInstance InstanceType;

      public:
        AGXPHYSICS_EXPORT ShapeGroupPtr();
        AGXPHYSICS_EXPORT ShapeGroupPtr(agxData::EntityStorage* storage, agx::Index id);
        AGXPHYSICS_EXPORT ShapeGroupPtr(const agxData::EntityPtr& ptr);
        AGXPHYSICS_EXPORT ShapeGroupPtr(const agxData::EntityInstance& instance);
        AGXPHYSICS_EXPORT ShapeGroupPtr& operator= (const agxData::EntityPtr& ptr);
        AGXPHYSICS_EXPORT ShapeGroupPtr& operator= (const agxData::EntityInstance& instance);
        AGXPHYSICS_EXPORT ShapeGroupInstance instance();
        AGXPHYSICS_EXPORT const ShapeGroupInstance instance() const;

        AGXPHYSICS_EXPORT ShapeGroupSemantics* operator->();
        AGXPHYSICS_EXPORT const ShapeGroupSemantics* operator->() const;

        ShapeGroupData* getData();
        const ShapeGroupData* getData() const;


        /// \return reference to the tag attribute
        AGXPHYSICS_EXPORT agx::UInt& tag();
        /// \return const reference to the tag attribute
        AGXPHYSICS_EXPORT agx::UInt const& tag() const;

        /// \return reference to the childShapes attribute
        AGXPHYSICS_EXPORT agx::Vector< agx::Physics::Geometry::ShapePtr >& childShapes();
        /// \return const reference to the childShapes attribute
        AGXPHYSICS_EXPORT agx::Vector< agx::Physics::Geometry::ShapePtr > const& childShapes() const;

        /// \return reference to the childTransforms attribute
        AGXPHYSICS_EXPORT agx::Vector< agx::AffineMatrix4x4 >& childTransforms();
        /// \return const reference to the childTransforms attribute
        AGXPHYSICS_EXPORT agx::Vector< agx::AffineMatrix4x4 > const& childTransforms() const;

        DOXYGEN_START_INTERNAL_BLOCK()
        // Only used by tolua!
        #ifdef AGX_GENERATED_TOLUA_WRAPPER
        inline agx::UInt const& getTag() const { return (*this)->getTag(); }
        inline agx::Vector< agx::Physics::Geometry::ShapePtr > const& getChildShapes() const { return (*this)->getChildShapes(); }
        inline agx::Vector< agx::AffineMatrix4x4 > const& getChildTransforms() const { return (*this)->getChildTransforms(); }


        inline void setTag(agx::UInt const& value) { (*this)->setTag(value); }
        inline void setChildShapes(agx::Vector< agx::Physics::Geometry::ShapePtr > const& value) { (*this)->setChildShapes(value); }
        inline void setChildTransforms(agx::Vector< agx::AffineMatrix4x4 > const& value) { (*this)->setChildTransforms(value); }
        #endif
        DOXYGEN_END_INTERNAL_BLOCK()

      };

      // Entity is Referenced
      typedef agxData::EntityRef< ShapeGroupPtr > ShapeGroupRef;


      DOXYGEN_START_INTERNAL_BLOCK()
      class AGXPHYSICS_EXPORT ShapeGroupInstance : public agx::Physics::Geometry::ShapeInstance
      {
      public:
        ShapeGroupInstance();
        ShapeGroupInstance(ShapeGroupData* data, agx::Index index);
        ShapeGroupInstance(agxData::EntityStorage *storage, agx::Index index);
        ShapeGroupInstance(const agxData::EntityInstance& other);
        ShapeGroupInstance(const agxData::EntityPtr& ptr);

        ShapeGroupData* getData();
        const ShapeGroupData* getData() const;

      public:
        /// \return reference to the tag attribute
        agx::UInt& tag();
        /// \return const reference to the tag attribute
        agx::UInt const& tag() const;

        /// \return reference to the childShapes attribute
        agx::Vector< agx::Physics::Geometry::ShapePtr >& childShapes();
        /// \return const reference to the childShapes attribute
        agx::Vector< agx::Physics::Geometry::ShapePtr > const& childShapes() const;

        /// \return reference to the childTransforms attribute
        agx::Vector< agx::AffineMatrix4x4 >& childTransforms();
        /// \return const reference to the childTransforms attribute
        agx::Vector< agx::AffineMatrix4x4 > const& childTransforms() const;

      };
      DOXYGEN_END_INTERNAL_BLOCK()



      typedef agx::VectorPOD<ShapeGroupPtr> ShapeGroupPtrVector;
      typedef agxData::Array<ShapeGroupPtr> ShapeGroupPtrArray;



      DOXYGEN_START_INTERNAL_BLOCK()
      /* Implementation */
      //-----------------------------------------------------------------------------------------------------
      //-----------------------------------------------------------------------------------------------------
      inline ShapeGroupInstance agx::Physics::Geometry::ShapeGroupData::operator[] (size_t index) { return ShapeGroupInstance(this, (agx::Index)index); }
      //-----------------------------------------------------------------------------------------------------
      AGX_FORCE_INLINE ShapeGroupPtr::ShapeGroupPtr() {}
      AGX_FORCE_INLINE ShapeGroupPtr::ShapeGroupPtr(agxData::EntityStorage* storage, agx::Index id) : agx::Physics::Geometry::ShapePtr(storage, id) {}
      AGX_FORCE_INLINE ShapeGroupPtr::ShapeGroupPtr(const agxData::EntityPtr& ptr) : agx::Physics::Geometry::ShapePtr(ptr)
      {
        agxAssertN(!ptr || ptr.isInstanceOf(ShapeGroupModel::instance()),
          "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
          EntityPtr::getModel()->fullPath().c_str(), ShapeGroupModel::instance()->fullPath().c_str());
      }

      AGX_FORCE_INLINE ShapeGroupPtr::ShapeGroupPtr(const agxData::EntityInstance& instance) : agx::Physics::Geometry::ShapePtr(instance)
      {
        agxAssertN(!instance || instance.isInstanceOf(ShapeGroupModel::instance()),
          "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
          EntityPtr::getModel()->fullPath().c_str(), ShapeGroupModel::instance()->fullPath().c_str());
      }

      AGX_FORCE_INLINE ShapeGroupPtr& ShapeGroupPtr::operator= (const agxData::EntityPtr& ptr)
      {
        agxData::EntityPtr::operator= (ptr);
        agxAssertN(!ptr || ptr.isInstanceOf(ShapeGroupModel::instance()),
          "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
          EntityPtr::getModel()->fullPath().c_str(), ShapeGroupModel::instance()->fullPath().c_str());
        return *this;
      }

      AGX_FORCE_INLINE ShapeGroupPtr& ShapeGroupPtr::operator= (const agxData::EntityInstance& instance)
      {
        agxData::EntityPtr::operator= (instance);
        agxAssertN(!instance || instance.isInstanceOf(ShapeGroupModel::instance()),
          "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
          EntityPtr::getModel()->fullPath().c_str(), ShapeGroupModel::instance()->fullPath().c_str());
        return *this;
      }

      inline ShapeGroupInstance ShapeGroupPtr::instance() { return agxData::EntityPtr::instance(); }
      inline const ShapeGroupInstance ShapeGroupPtr::instance() const { return agxData::EntityPtr::instance(); }
      AGX_FORCE_INLINE ShapeGroupSemantics* ShapeGroupPtr::operator->() { return (ShapeGroupSemantics* )this; }
      AGX_FORCE_INLINE const ShapeGroupSemantics* ShapeGroupPtr::operator->() const { return (const ShapeGroupSemantics* )this; }
      AGX_FORCE_INLINE ShapeGroupData* ShapeGroupPtr::getData() { return static_cast<ShapeGroupData* >(agxData::EntityPtr::getData()); }
      AGX_FORCE_INLINE const ShapeGroupData* ShapeGroupPtr::getData() const { return static_cast<const ShapeGroupData* >(agxData::EntityPtr::getData()); }

      AGX_FORCE_INLINE agx::UInt& ShapeGroupPtr::tag() { verifyIndex(); return getData()->tag[calculateIndex()]; }
      AGX_FORCE_INLINE agx::UInt const& ShapeGroupPtr::tag() const { verifyIndex(); return getData()->tag[calculateIndex()]; }

      AGX_FORCE_INLINE agx::Vector< agx::Physics::Geometry::ShapePtr >& ShapeGroupPtr::childShapes() { verifyIndex(); return getData()->childShapes[calculateIndex()]; }
      AGX_FORCE_INLINE agx::Vector< agx::Physics::Geometry::ShapePtr > const& ShapeGroupPtr::childShapes() const { verifyIndex(); return getData()->childShapes[calculateIndex()]; }

      AGX_FORCE_INLINE agx::Vector< agx::AffineMatrix4x4 >& ShapeGroupPtr::childTransforms() { verifyIndex(); return getData()->childTransforms[calculateIndex()]; }
      AGX_FORCE_INLINE agx::Vector< agx::AffineMatrix4x4 > const& ShapeGroupPtr::childTransforms() const { verifyIndex(); return getData()->childTransforms[calculateIndex()]; }

      //-----------------------------------------------------------------------------------------------------
      AGX_FORCE_INLINE ShapeGroupInstance::ShapeGroupInstance() {}
      AGX_FORCE_INLINE ShapeGroupInstance::ShapeGroupInstance(ShapeGroupData* data, agx::Index index) : agx::Physics::Geometry::ShapeInstance(data, index) {}
      AGX_FORCE_INLINE ShapeGroupInstance::ShapeGroupInstance(agxData::EntityStorage* storage, agx::Index index) : agx::Physics::Geometry::ShapeInstance(storage, index) {}
      AGX_FORCE_INLINE ShapeGroupInstance::ShapeGroupInstance(const agxData::EntityInstance& other) : agx::Physics::Geometry::ShapeInstance(other)
      {
        agxAssertN(!other || other.isInstanceOf(ShapeGroupModel::instance()),
          "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
          EntityInstance::getModel()->fullPath().c_str(), ShapeGroupModel::instance()->fullPath().c_str());
      }

      AGX_FORCE_INLINE ShapeGroupInstance::ShapeGroupInstance(const agxData::EntityPtr& ptr) : agx::Physics::Geometry::ShapeInstance(ptr)
      {
        agxAssertN(!ptr || ptr.isInstanceOf(ShapeGroupModel::instance()),
          "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
          EntityInstance::getModel()->fullPath().c_str(), ShapeGroupModel::instance()->fullPath().c_str());
      }


      AGX_FORCE_INLINE ShapeGroupData* ShapeGroupInstance::getData() { return static_cast<ShapeGroupData* >(agxData::EntityInstance::getData()); }
      AGX_FORCE_INLINE const ShapeGroupData* ShapeGroupInstance::getData() const { return static_cast<const ShapeGroupData* >(agxData::EntityInstance::getData()); }

      AGX_FORCE_INLINE agx::UInt& ShapeGroupInstance::tag() { verifyIndex(); return getData()->tag[getIndex()]; }
      AGX_FORCE_INLINE agx::UInt const& ShapeGroupInstance::tag() const { verifyIndex(); return getData()->tag[getIndex()]; }

      AGX_FORCE_INLINE agx::Vector< agx::Physics::Geometry::ShapePtr >& ShapeGroupInstance::childShapes() { verifyIndex(); return getData()->childShapes[getIndex()]; }
      AGX_FORCE_INLINE agx::Vector< agx::Physics::Geometry::ShapePtr > const& ShapeGroupInstance::childShapes() const { verifyIndex(); return getData()->childShapes[getIndex()]; }

      AGX_FORCE_INLINE agx::Vector< agx::AffineMatrix4x4 >& ShapeGroupInstance::childTransforms() { verifyIndex(); return getData()->childTransforms[getIndex()]; }
      AGX_FORCE_INLINE agx::Vector< agx::AffineMatrix4x4 > const& ShapeGroupInstance::childTransforms() const { verifyIndex(); return getData()->childTransforms[getIndex()]; }

      //-----------------------------------------------------------------------------------------------------
      AGX_FORCE_INLINE ShapeGroupSemantics::ShapeGroupSemantics() {}
      //-----------------------------------------------------------------------------------------------------
      DOXYGEN_END_INTERNAL_BLOCK()
    }
  }
}

AGX_TYPE_BINDING(agx::Physics::Geometry::ShapeGroupPtr, "Physics.Geometry.ShapeGroupPtr")
AGX_TYPE_BINDING(agx::Physics::Geometry::ShapeGroupInstance, "Physics.Geometry.ShapeGroupInstance")

#ifdef _MSC_VER
# pragma warning(pop)
#endif

#undef AGX_ENTITY_WRAPPER
#undef AGX_ENTITY_NAMESPACE
#endif

