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

#ifndef GENERATED_AGX_PHYSICS_GEOMETRY_RENDERBOX_H_PLUGIN
#define GENERATED_AGX_PHYSICS_GEOMETRY_RENDERBOX_H_PLUGIN

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
#include <agxGL/RenderBox.h>


namespace agx
{
  namespace Physics
  {
    namespace Geometry
    {

      class RenderBoxModel;
      class RenderBoxData;
      class RenderBoxPtr;
      class RenderBoxInstance;
      class RenderBoxSemantics;


      AGX_DECLARE_POINTER_TYPES(RenderBoxModel);

      /** 
      Abstract description of the data attributes for the Physics.Geometry.RenderBox entity.
      */ 
      class AGXPHYSICS_EXPORT RenderBoxModel : public agxData::EntityModel
      {
      public:
        typedef RenderBoxPtr PtrT;

        RenderBoxModel(const agx::String& name = "RenderBox");

        /// \return The entity model singleton.
        static RenderBoxModel* instance();

        /// Create and return a pointer to a new instance in the default storage for this entity model.
        static RenderBoxPtr createInstance();

        /// \return The default storage for this entity model.
        static agxData::EntityStorage* defaultStorage();

        /// This is part of internal cleanup and should not be called by users
        virtual void shutdownCleanup() override;



        /* Attributes */
        static agxData::ScalarAttributeT< agxGL::RenderBoxVertices >* verticesAttribute;
        static agxData::ScalarAttributeT< agxGL::RenderBoxNormals >* normalsAttribute;
        static agxData::ScalarAttributeT< agxGL::RenderBoxOutlines >* outlinesAttribute;

      protected:
        virtual ~RenderBoxModel();
        virtual agxData::EntityData* createData(agxData::EntityStorage* storage) override;
        virtual void configure(agx::TiXmlElement* eEntity) override;
        virtual void initAttributeAccessors() override;
        virtual void construct(agxData::EntityPtr instance) override;
        void construct(agx::Physics::Geometry::RenderBoxPtr renderBox);
      };


      DOXYGEN_START_INTERNAL_BLOCK()
      #ifndef AGX_PHYSICS_GEOMETRY_RENDERBOX_DATA_SET_OVERRIDE
      #define AGX_PHYSICS_GEOMETRY_RENDERBOX_DATA_SET
      class AGXPHYSICS_EXPORT RenderBoxData : public agxData::EntityData
      {
      public:
        RenderBoxInstance operator[] (size_t index);

      public:
        agxData::Array< RenderBoxPtr >& instance;
        agxData::Array< agxGL::RenderBoxVertices > vertices;
        agxData::Array< agxGL::RenderBoxNormals > normals;
        agxData::Array< agxGL::RenderBoxOutlines > outlines;

      public:
        typedef agxGL::RenderBoxVertices verticesType;
        typedef agxGL::RenderBoxNormals normalsType;
        typedef agxGL::RenderBoxOutlines outlinesType;

      public:
        RenderBoxData(agxData::EntityStorage* storage);
        RenderBoxData();

      protected:
        virtual ~RenderBoxData() {}
        virtual void setNumElements(agx::Index numElements) override;

      private:
        RenderBoxData& operator= (const RenderBoxData&) { return *this; }

      };
      #endif
      DOXYGEN_END_INTERNAL_BLOCK()


      DOXYGEN_START_INTERNAL_BLOCK()
      class AGXPHYSICS_EXPORT RenderBoxSemantics : protected agxData::EntityPtr
      {
      public:

        // Automatic getters
        agxGL::RenderBoxVertices const& getVertices() const;
        agxGL::RenderBoxNormals const& getNormals() const;
        agxGL::RenderBoxOutlines const& getOutlines() const;

        // Semantics defined by explicit kernels

        // Automatic setters
        void setVertices(agxGL::RenderBoxVertices const& value);
        void setNormals(agxGL::RenderBoxNormals const& value);
        void setOutlines(agxGL::RenderBoxOutlines const& value);


      protected:
        friend class RenderBoxPtr;
        friend class RenderBoxInstance;
        RenderBoxSemantics();
      };
      DOXYGEN_END_INTERNAL_BLOCK()


      /**
      Pointer to a entity instance of type Physics.Geometry.RenderBox
      */
      class CALLABLE RenderBoxPtr : public agxData::EntityPtr
      {
      public:
        typedef RenderBoxModel ModelType;
        typedef RenderBoxData DataType;
        typedef RenderBoxInstance InstanceType;

      public:
        AGXPHYSICS_EXPORT RenderBoxPtr();
        AGXPHYSICS_EXPORT RenderBoxPtr(agxData::EntityStorage* storage, agx::Index id);
        AGXPHYSICS_EXPORT RenderBoxPtr(const agxData::EntityPtr& ptr);
        AGXPHYSICS_EXPORT RenderBoxPtr(const agxData::EntityInstance& instance);
        AGXPHYSICS_EXPORT RenderBoxPtr& operator= (const agxData::EntityPtr& ptr);
        AGXPHYSICS_EXPORT RenderBoxPtr& operator= (const agxData::EntityInstance& instance);
        AGXPHYSICS_EXPORT RenderBoxInstance instance();
        AGXPHYSICS_EXPORT const RenderBoxInstance instance() const;

        AGXPHYSICS_EXPORT RenderBoxSemantics* operator->();
        AGXPHYSICS_EXPORT const RenderBoxSemantics* operator->() const;

        RenderBoxData* getData();
        const RenderBoxData* getData() const;


        /// \return reference to the vertices attribute
        AGXPHYSICS_EXPORT agxGL::RenderBoxVertices& vertices();
        /// \return const reference to the vertices attribute
        AGXPHYSICS_EXPORT agxGL::RenderBoxVertices const& vertices() const;

        /// \return reference to the normals attribute
        AGXPHYSICS_EXPORT agxGL::RenderBoxNormals& normals();
        /// \return const reference to the normals attribute
        AGXPHYSICS_EXPORT agxGL::RenderBoxNormals const& normals() const;

        /// \return reference to the outlines attribute
        AGXPHYSICS_EXPORT agxGL::RenderBoxOutlines& outlines();
        /// \return const reference to the outlines attribute
        AGXPHYSICS_EXPORT agxGL::RenderBoxOutlines const& outlines() const;

        DOXYGEN_START_INTERNAL_BLOCK()
        // Only used by tolua!
        #ifdef AGX_GENERATED_TOLUA_WRAPPER
        inline agxGL::RenderBoxVertices const& getVertices() const { return (*this)->getVertices(); }
        inline agxGL::RenderBoxNormals const& getNormals() const { return (*this)->getNormals(); }
        inline agxGL::RenderBoxOutlines const& getOutlines() const { return (*this)->getOutlines(); }


        inline void setVertices(agxGL::RenderBoxVertices const& value) { (*this)->setVertices(value); }
        inline void setNormals(agxGL::RenderBoxNormals const& value) { (*this)->setNormals(value); }
        inline void setOutlines(agxGL::RenderBoxOutlines const& value) { (*this)->setOutlines(value); }
        #endif
        DOXYGEN_END_INTERNAL_BLOCK()

      };


      DOXYGEN_START_INTERNAL_BLOCK()
      class AGXPHYSICS_EXPORT RenderBoxInstance : public agxData::EntityInstance
      {
      public:
        RenderBoxInstance();
        RenderBoxInstance(RenderBoxData* data, agx::Index index);
        RenderBoxInstance(agxData::EntityStorage *storage, agx::Index index);
        RenderBoxInstance(const agxData::EntityInstance& other);
        RenderBoxInstance(const agxData::EntityPtr& ptr);

        RenderBoxData* getData();
        const RenderBoxData* getData() const;

      public:
        /// \return reference to the vertices attribute
        agxGL::RenderBoxVertices& vertices();
        /// \return const reference to the vertices attribute
        agxGL::RenderBoxVertices const& vertices() const;

        /// \return reference to the normals attribute
        agxGL::RenderBoxNormals& normals();
        /// \return const reference to the normals attribute
        agxGL::RenderBoxNormals const& normals() const;

        /// \return reference to the outlines attribute
        agxGL::RenderBoxOutlines& outlines();
        /// \return const reference to the outlines attribute
        agxGL::RenderBoxOutlines const& outlines() const;

      };
      DOXYGEN_END_INTERNAL_BLOCK()



      typedef agx::VectorPOD<RenderBoxPtr> RenderBoxPtrVector;
      typedef agxData::Array<RenderBoxPtr> RenderBoxPtrArray;



      DOXYGEN_START_INTERNAL_BLOCK()
      /* Implementation */
      //-----------------------------------------------------------------------------------------------------
      //-----------------------------------------------------------------------------------------------------
      inline RenderBoxInstance agx::Physics::Geometry::RenderBoxData::operator[] (size_t index) { return RenderBoxInstance(this, (agx::Index)index); }
      //-----------------------------------------------------------------------------------------------------
      AGX_FORCE_INLINE RenderBoxPtr::RenderBoxPtr() {}
      AGX_FORCE_INLINE RenderBoxPtr::RenderBoxPtr(agxData::EntityStorage* storage, agx::Index id) : agxData::EntityPtr(storage, id) {}
      AGX_FORCE_INLINE RenderBoxPtr::RenderBoxPtr(const agxData::EntityPtr& ptr) : agxData::EntityPtr(ptr)
      {
        agxAssertN(!ptr || ptr.isInstanceOf(RenderBoxModel::instance()),
          "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
          EntityPtr::getModel()->fullPath().c_str(), RenderBoxModel::instance()->fullPath().c_str());
      }

      AGX_FORCE_INLINE RenderBoxPtr::RenderBoxPtr(const agxData::EntityInstance& instance) : agxData::EntityPtr(instance)
      {
        agxAssertN(!instance || instance.isInstanceOf(RenderBoxModel::instance()),
          "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
          EntityPtr::getModel()->fullPath().c_str(), RenderBoxModel::instance()->fullPath().c_str());
      }

      AGX_FORCE_INLINE RenderBoxPtr& RenderBoxPtr::operator= (const agxData::EntityPtr& ptr)
      {
        agxData::EntityPtr::operator= (ptr);
        agxAssertN(!ptr || ptr.isInstanceOf(RenderBoxModel::instance()),
          "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
          EntityPtr::getModel()->fullPath().c_str(), RenderBoxModel::instance()->fullPath().c_str());
        return *this;
      }

      AGX_FORCE_INLINE RenderBoxPtr& RenderBoxPtr::operator= (const agxData::EntityInstance& instance)
      {
        agxData::EntityPtr::operator= (instance);
        agxAssertN(!instance || instance.isInstanceOf(RenderBoxModel::instance()),
          "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
          EntityPtr::getModel()->fullPath().c_str(), RenderBoxModel::instance()->fullPath().c_str());
        return *this;
      }

      inline RenderBoxInstance RenderBoxPtr::instance() { return agxData::EntityPtr::instance(); }
      inline const RenderBoxInstance RenderBoxPtr::instance() const { return agxData::EntityPtr::instance(); }
      AGX_FORCE_INLINE RenderBoxSemantics* RenderBoxPtr::operator->() { return (RenderBoxSemantics* )this; }
      AGX_FORCE_INLINE const RenderBoxSemantics* RenderBoxPtr::operator->() const { return (const RenderBoxSemantics* )this; }
      AGX_FORCE_INLINE RenderBoxData* RenderBoxPtr::getData() { return static_cast<RenderBoxData* >(agxData::EntityPtr::getData()); }
      AGX_FORCE_INLINE const RenderBoxData* RenderBoxPtr::getData() const { return static_cast<const RenderBoxData* >(agxData::EntityPtr::getData()); }

      AGX_FORCE_INLINE agxGL::RenderBoxVertices& RenderBoxPtr::vertices() { verifyIndex(); return getData()->vertices[calculateIndex()]; }
      AGX_FORCE_INLINE agxGL::RenderBoxVertices const& RenderBoxPtr::vertices() const { verifyIndex(); return getData()->vertices[calculateIndex()]; }

      AGX_FORCE_INLINE agxGL::RenderBoxNormals& RenderBoxPtr::normals() { verifyIndex(); return getData()->normals[calculateIndex()]; }
      AGX_FORCE_INLINE agxGL::RenderBoxNormals const& RenderBoxPtr::normals() const { verifyIndex(); return getData()->normals[calculateIndex()]; }

      AGX_FORCE_INLINE agxGL::RenderBoxOutlines& RenderBoxPtr::outlines() { verifyIndex(); return getData()->outlines[calculateIndex()]; }
      AGX_FORCE_INLINE agxGL::RenderBoxOutlines const& RenderBoxPtr::outlines() const { verifyIndex(); return getData()->outlines[calculateIndex()]; }

      //-----------------------------------------------------------------------------------------------------
      AGX_FORCE_INLINE RenderBoxInstance::RenderBoxInstance() {}
      AGX_FORCE_INLINE RenderBoxInstance::RenderBoxInstance(RenderBoxData* data, agx::Index index) : agxData::EntityInstance(data, index) {}
      AGX_FORCE_INLINE RenderBoxInstance::RenderBoxInstance(agxData::EntityStorage* storage, agx::Index index) : agxData::EntityInstance(storage, index) {}
      AGX_FORCE_INLINE RenderBoxInstance::RenderBoxInstance(const agxData::EntityInstance& other) : agxData::EntityInstance(other)
      {
        agxAssertN(!other || other.isInstanceOf(RenderBoxModel::instance()),
          "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
          EntityInstance::getModel()->fullPath().c_str(), RenderBoxModel::instance()->fullPath().c_str());
      }

      AGX_FORCE_INLINE RenderBoxInstance::RenderBoxInstance(const agxData::EntityPtr& ptr) : agxData::EntityInstance(ptr)
      {
        agxAssertN(!ptr || ptr.isInstanceOf(RenderBoxModel::instance()),
          "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
          EntityInstance::getModel()->fullPath().c_str(), RenderBoxModel::instance()->fullPath().c_str());
      }


      AGX_FORCE_INLINE RenderBoxData* RenderBoxInstance::getData() { return static_cast<RenderBoxData* >(agxData::EntityInstance::getData()); }
      AGX_FORCE_INLINE const RenderBoxData* RenderBoxInstance::getData() const { return static_cast<const RenderBoxData* >(agxData::EntityInstance::getData()); }

      AGX_FORCE_INLINE agxGL::RenderBoxVertices& RenderBoxInstance::vertices() { verifyIndex(); return getData()->vertices[getIndex()]; }
      AGX_FORCE_INLINE agxGL::RenderBoxVertices const& RenderBoxInstance::vertices() const { verifyIndex(); return getData()->vertices[getIndex()]; }

      AGX_FORCE_INLINE agxGL::RenderBoxNormals& RenderBoxInstance::normals() { verifyIndex(); return getData()->normals[getIndex()]; }
      AGX_FORCE_INLINE agxGL::RenderBoxNormals const& RenderBoxInstance::normals() const { verifyIndex(); return getData()->normals[getIndex()]; }

      AGX_FORCE_INLINE agxGL::RenderBoxOutlines& RenderBoxInstance::outlines() { verifyIndex(); return getData()->outlines[getIndex()]; }
      AGX_FORCE_INLINE agxGL::RenderBoxOutlines const& RenderBoxInstance::outlines() const { verifyIndex(); return getData()->outlines[getIndex()]; }

      //-----------------------------------------------------------------------------------------------------
      AGX_FORCE_INLINE RenderBoxSemantics::RenderBoxSemantics() {}
      //-----------------------------------------------------------------------------------------------------
      DOXYGEN_END_INTERNAL_BLOCK()
    }
  }
}

AGX_TYPE_BINDING(agx::Physics::Geometry::RenderBoxPtr, "Physics.Geometry.RenderBoxPtr")
AGX_TYPE_BINDING(agx::Physics::Geometry::RenderBoxInstance, "Physics.Geometry.RenderBoxInstance")

#ifdef _MSC_VER
# pragma warning(pop)
#endif

#undef AGX_ENTITY_WRAPPER
#undef AGX_ENTITY_NAMESPACE
#endif

