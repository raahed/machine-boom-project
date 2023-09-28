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

#ifndef GENERATED_AGX_PHYSICS_GEOMETRY_WIRESHAPE_H_PLUGIN
#define GENERATED_AGX_PHYSICS_GEOMETRY_WIRESHAPE_H_PLUGIN

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
#include <agx/Physics/Geometry/CapsuleEntity.h>
#include <agx/Vec3.h>
#include <agx/Integer.h>


namespace agx
{
  namespace Physics
  {
    namespace Geometry
    {

      class WireShapeModel;
      class WireShapeData;
      class WireShapePtr;
      class WireShapeInstance;
      class WireShapeSemantics;


      AGX_DECLARE_POINTER_TYPES(WireShapeModel);

      /** 
      Abstract description of the data attributes for the Physics.Geometry.WireShape entity.
      */ 
      class AGXPHYSICS_EXPORT WireShapeModel : public agx::Physics::Geometry::CapsuleModel
      {
      public:
        typedef WireShapePtr PtrT;

        WireShapeModel(const agx::String& name = "WireShape");

        /// \return The entity model singleton.
        static WireShapeModel* instance();

        /// Create and return a pointer to a new instance in the default storage for this entity model.
        static WireShapePtr createInstance();

        /// \return The default storage for this entity model.
        static agxData::EntityStorage* defaultStorage();

        /// This is part of internal cleanup and should not be called by users
        virtual void shutdownCleanup() override;



        /* Attributes */
        static agxData::ScalarAttributeT< agx::Vec3 >* previousPoint0Attribute;
        static agxData::ScalarAttributeT< agx::Vec3 >* previousPoint1Attribute;
        static agxData::ScalarAttributeT< agx::UInt32 >* wireIdAttribute;

      protected:
        virtual ~WireShapeModel();
        virtual agxData::EntityData* createData(agxData::EntityStorage* storage) override;
        virtual void configure(agx::TiXmlElement* eEntity) override;
        virtual void initAttributeAccessors() override;
        virtual void construct(agxData::EntityPtr instance) override;
        void construct(agx::Physics::Geometry::WireShapePtr wireShape);
      };


      DOXYGEN_START_INTERNAL_BLOCK()
      #ifndef AGX_PHYSICS_GEOMETRY_WIRESHAPE_DATA_SET_OVERRIDE
      #define AGX_PHYSICS_GEOMETRY_WIRESHAPE_DATA_SET
      class AGXPHYSICS_EXPORT WireShapeData : public agx::Physics::Geometry::CapsuleData
      {
      public:
        WireShapeInstance operator[] (size_t index);

      public:
        agxData::Array< WireShapePtr >& instance;
        agxData::Array< agx::Vec3 > previousPoint0;
        agxData::Array< agx::Vec3 > previousPoint1;
        agxData::Array< agx::UInt32 > wireId;

      public:
        typedef agx::Vec3 previousPoint0Type;
        typedef agx::Vec3 previousPoint1Type;
        typedef agx::UInt32 wireIdType;

      public:
        WireShapeData(agxData::EntityStorage* storage);
        WireShapeData();

      protected:
        virtual ~WireShapeData() {}
        virtual void setNumElements(agx::Index numElements) override;

      private:
        WireShapeData& operator= (const WireShapeData&) { return *this; }

      };
      #endif
      DOXYGEN_END_INTERNAL_BLOCK()


      DOXYGEN_START_INTERNAL_BLOCK()
      class AGXPHYSICS_EXPORT WireShapeSemantics : public agx::Physics::Geometry::CapsuleSemantics
      {
      public:

        // Automatic getters
        agx::Vec3 const& getPreviousPoint0() const;
        agx::Vec3 const& getPreviousPoint1() const;
        agx::UInt32 const& getWireId() const;

        // Semantics defined by explicit kernels

        // Automatic setters
        void setPreviousPoint0(agx::Vec3 const& value);
        void setPreviousPoint1(agx::Vec3 const& value);
        void setWireId(agx::UInt32 const& value);


      protected:
        friend class WireShapePtr;
        friend class WireShapeInstance;
        WireShapeSemantics();
      };
      DOXYGEN_END_INTERNAL_BLOCK()


      /**
      Pointer to a entity instance of type Physics.Geometry.WireShape
      */
      class CALLABLE WireShapePtr : public agx::Physics::Geometry::CapsulePtr
      {
      public:
        typedef WireShapeModel ModelType;
        typedef WireShapeData DataType;
        typedef WireShapeInstance InstanceType;

      public:
        AGXPHYSICS_EXPORT WireShapePtr();
        AGXPHYSICS_EXPORT WireShapePtr(agxData::EntityStorage* storage, agx::Index id);
        AGXPHYSICS_EXPORT WireShapePtr(const agxData::EntityPtr& ptr);
        AGXPHYSICS_EXPORT WireShapePtr(const agxData::EntityInstance& instance);
        AGXPHYSICS_EXPORT WireShapePtr& operator= (const agxData::EntityPtr& ptr);
        AGXPHYSICS_EXPORT WireShapePtr& operator= (const agxData::EntityInstance& instance);
        AGXPHYSICS_EXPORT WireShapeInstance instance();
        AGXPHYSICS_EXPORT const WireShapeInstance instance() const;

        AGXPHYSICS_EXPORT WireShapeSemantics* operator->();
        AGXPHYSICS_EXPORT const WireShapeSemantics* operator->() const;

        WireShapeData* getData();
        const WireShapeData* getData() const;


        /// \return reference to the previousPoint0 attribute
        AGXPHYSICS_EXPORT agx::Vec3& previousPoint0();
        /// \return const reference to the previousPoint0 attribute
        AGXPHYSICS_EXPORT agx::Vec3 const& previousPoint0() const;

        /// \return reference to the previousPoint1 attribute
        AGXPHYSICS_EXPORT agx::Vec3& previousPoint1();
        /// \return const reference to the previousPoint1 attribute
        AGXPHYSICS_EXPORT agx::Vec3 const& previousPoint1() const;

        /// \return reference to the wireId attribute
        AGXPHYSICS_EXPORT agx::UInt32& wireId();
        /// \return const reference to the wireId attribute
        AGXPHYSICS_EXPORT agx::UInt32 const& wireId() const;

        DOXYGEN_START_INTERNAL_BLOCK()
        // Only used by tolua!
        #ifdef AGX_GENERATED_TOLUA_WRAPPER
        inline agx::Vec3 const& getPreviousPoint0() const { return (*this)->getPreviousPoint0(); }
        inline agx::Vec3 const& getPreviousPoint1() const { return (*this)->getPreviousPoint1(); }
        inline agx::UInt32 const& getWireId() const { return (*this)->getWireId(); }


        inline void setPreviousPoint0(agx::Vec3 const& value) { (*this)->setPreviousPoint0(value); }
        inline void setPreviousPoint1(agx::Vec3 const& value) { (*this)->setPreviousPoint1(value); }
        inline void setWireId(agx::UInt32 const& value) { (*this)->setWireId(value); }
        #endif
        DOXYGEN_END_INTERNAL_BLOCK()

      };

      // Entity is Referenced
      typedef agxData::EntityRef< WireShapePtr > WireShapeRef;


      DOXYGEN_START_INTERNAL_BLOCK()
      class AGXPHYSICS_EXPORT WireShapeInstance : public agx::Physics::Geometry::CapsuleInstance
      {
      public:
        WireShapeInstance();
        WireShapeInstance(WireShapeData* data, agx::Index index);
        WireShapeInstance(agxData::EntityStorage *storage, agx::Index index);
        WireShapeInstance(const agxData::EntityInstance& other);
        WireShapeInstance(const agxData::EntityPtr& ptr);

        WireShapeData* getData();
        const WireShapeData* getData() const;

      public:
        /// \return reference to the previousPoint0 attribute
        agx::Vec3& previousPoint0();
        /// \return const reference to the previousPoint0 attribute
        agx::Vec3 const& previousPoint0() const;

        /// \return reference to the previousPoint1 attribute
        agx::Vec3& previousPoint1();
        /// \return const reference to the previousPoint1 attribute
        agx::Vec3 const& previousPoint1() const;

        /// \return reference to the wireId attribute
        agx::UInt32& wireId();
        /// \return const reference to the wireId attribute
        agx::UInt32 const& wireId() const;

      };
      DOXYGEN_END_INTERNAL_BLOCK()



      typedef agx::VectorPOD<WireShapePtr> WireShapePtrVector;
      typedef agxData::Array<WireShapePtr> WireShapePtrArray;



      DOXYGEN_START_INTERNAL_BLOCK()
      /* Implementation */
      //-----------------------------------------------------------------------------------------------------
      //-----------------------------------------------------------------------------------------------------
      inline WireShapeInstance agx::Physics::Geometry::WireShapeData::operator[] (size_t index) { return WireShapeInstance(this, (agx::Index)index); }
      //-----------------------------------------------------------------------------------------------------
      AGX_FORCE_INLINE WireShapePtr::WireShapePtr() {}
      AGX_FORCE_INLINE WireShapePtr::WireShapePtr(agxData::EntityStorage* storage, agx::Index id) : agx::Physics::Geometry::CapsulePtr(storage, id) {}
      AGX_FORCE_INLINE WireShapePtr::WireShapePtr(const agxData::EntityPtr& ptr) : agx::Physics::Geometry::CapsulePtr(ptr)
      {
        agxAssertN(!ptr || ptr.isInstanceOf(WireShapeModel::instance()),
          "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
          EntityPtr::getModel()->fullPath().c_str(), WireShapeModel::instance()->fullPath().c_str());
      }

      AGX_FORCE_INLINE WireShapePtr::WireShapePtr(const agxData::EntityInstance& instance) : agx::Physics::Geometry::CapsulePtr(instance)
      {
        agxAssertN(!instance || instance.isInstanceOf(WireShapeModel::instance()),
          "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
          EntityPtr::getModel()->fullPath().c_str(), WireShapeModel::instance()->fullPath().c_str());
      }

      AGX_FORCE_INLINE WireShapePtr& WireShapePtr::operator= (const agxData::EntityPtr& ptr)
      {
        agxData::EntityPtr::operator= (ptr);
        agxAssertN(!ptr || ptr.isInstanceOf(WireShapeModel::instance()),
          "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
          EntityPtr::getModel()->fullPath().c_str(), WireShapeModel::instance()->fullPath().c_str());
        return *this;
      }

      AGX_FORCE_INLINE WireShapePtr& WireShapePtr::operator= (const agxData::EntityInstance& instance)
      {
        agxData::EntityPtr::operator= (instance);
        agxAssertN(!instance || instance.isInstanceOf(WireShapeModel::instance()),
          "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
          EntityPtr::getModel()->fullPath().c_str(), WireShapeModel::instance()->fullPath().c_str());
        return *this;
      }

      inline WireShapeInstance WireShapePtr::instance() { return agxData::EntityPtr::instance(); }
      inline const WireShapeInstance WireShapePtr::instance() const { return agxData::EntityPtr::instance(); }
      AGX_FORCE_INLINE WireShapeSemantics* WireShapePtr::operator->() { return (WireShapeSemantics* )this; }
      AGX_FORCE_INLINE const WireShapeSemantics* WireShapePtr::operator->() const { return (const WireShapeSemantics* )this; }
      AGX_FORCE_INLINE WireShapeData* WireShapePtr::getData() { return static_cast<WireShapeData* >(agxData::EntityPtr::getData()); }
      AGX_FORCE_INLINE const WireShapeData* WireShapePtr::getData() const { return static_cast<const WireShapeData* >(agxData::EntityPtr::getData()); }

      AGX_FORCE_INLINE agx::Vec3& WireShapePtr::previousPoint0() { verifyIndex(); return getData()->previousPoint0[calculateIndex()]; }
      AGX_FORCE_INLINE agx::Vec3 const& WireShapePtr::previousPoint0() const { verifyIndex(); return getData()->previousPoint0[calculateIndex()]; }

      AGX_FORCE_INLINE agx::Vec3& WireShapePtr::previousPoint1() { verifyIndex(); return getData()->previousPoint1[calculateIndex()]; }
      AGX_FORCE_INLINE agx::Vec3 const& WireShapePtr::previousPoint1() const { verifyIndex(); return getData()->previousPoint1[calculateIndex()]; }

      AGX_FORCE_INLINE agx::UInt32& WireShapePtr::wireId() { verifyIndex(); return getData()->wireId[calculateIndex()]; }
      AGX_FORCE_INLINE agx::UInt32 const& WireShapePtr::wireId() const { verifyIndex(); return getData()->wireId[calculateIndex()]; }

      //-----------------------------------------------------------------------------------------------------
      AGX_FORCE_INLINE WireShapeInstance::WireShapeInstance() {}
      AGX_FORCE_INLINE WireShapeInstance::WireShapeInstance(WireShapeData* data, agx::Index index) : agx::Physics::Geometry::CapsuleInstance(data, index) {}
      AGX_FORCE_INLINE WireShapeInstance::WireShapeInstance(agxData::EntityStorage* storage, agx::Index index) : agx::Physics::Geometry::CapsuleInstance(storage, index) {}
      AGX_FORCE_INLINE WireShapeInstance::WireShapeInstance(const agxData::EntityInstance& other) : agx::Physics::Geometry::CapsuleInstance(other)
      {
        agxAssertN(!other || other.isInstanceOf(WireShapeModel::instance()),
          "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
          EntityInstance::getModel()->fullPath().c_str(), WireShapeModel::instance()->fullPath().c_str());
      }

      AGX_FORCE_INLINE WireShapeInstance::WireShapeInstance(const agxData::EntityPtr& ptr) : agx::Physics::Geometry::CapsuleInstance(ptr)
      {
        agxAssertN(!ptr || ptr.isInstanceOf(WireShapeModel::instance()),
          "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
          EntityInstance::getModel()->fullPath().c_str(), WireShapeModel::instance()->fullPath().c_str());
      }


      AGX_FORCE_INLINE WireShapeData* WireShapeInstance::getData() { return static_cast<WireShapeData* >(agxData::EntityInstance::getData()); }
      AGX_FORCE_INLINE const WireShapeData* WireShapeInstance::getData() const { return static_cast<const WireShapeData* >(agxData::EntityInstance::getData()); }

      AGX_FORCE_INLINE agx::Vec3& WireShapeInstance::previousPoint0() { verifyIndex(); return getData()->previousPoint0[getIndex()]; }
      AGX_FORCE_INLINE agx::Vec3 const& WireShapeInstance::previousPoint0() const { verifyIndex(); return getData()->previousPoint0[getIndex()]; }

      AGX_FORCE_INLINE agx::Vec3& WireShapeInstance::previousPoint1() { verifyIndex(); return getData()->previousPoint1[getIndex()]; }
      AGX_FORCE_INLINE agx::Vec3 const& WireShapeInstance::previousPoint1() const { verifyIndex(); return getData()->previousPoint1[getIndex()]; }

      AGX_FORCE_INLINE agx::UInt32& WireShapeInstance::wireId() { verifyIndex(); return getData()->wireId[getIndex()]; }
      AGX_FORCE_INLINE agx::UInt32 const& WireShapeInstance::wireId() const { verifyIndex(); return getData()->wireId[getIndex()]; }

      //-----------------------------------------------------------------------------------------------------
      AGX_FORCE_INLINE WireShapeSemantics::WireShapeSemantics() {}
      //-----------------------------------------------------------------------------------------------------
      DOXYGEN_END_INTERNAL_BLOCK()
    }
  }
}

AGX_TYPE_BINDING(agx::Physics::Geometry::WireShapePtr, "Physics.Geometry.WireShapePtr")
AGX_TYPE_BINDING(agx::Physics::Geometry::WireShapeInstance, "Physics.Geometry.WireShapeInstance")

#ifdef _MSC_VER
# pragma warning(pop)
#endif

#undef AGX_ENTITY_WRAPPER
#undef AGX_ENTITY_NAMESPACE
#endif

