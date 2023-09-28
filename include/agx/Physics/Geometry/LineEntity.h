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

#ifndef GENERATED_AGX_PHYSICS_GEOMETRY_LINE_H_PLUGIN
#define GENERATED_AGX_PHYSICS_GEOMETRY_LINE_H_PLUGIN

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
#include <agx/Line.h>


namespace agx
{
  namespace Physics
  {
    namespace Geometry
    {

      class LineModel;
      class LineData;
      class LinePtr;
      class LineInstance;
      class LineSemantics;


      AGX_DECLARE_POINTER_TYPES(LineModel);

      /** 
      Abstract description of the data attributes for the Physics.Geometry.Line entity.
      */ 
      class AGXPHYSICS_EXPORT LineModel : public agx::Physics::Geometry::ShapeModel
      {
      public:
        typedef LinePtr PtrT;

        LineModel(const agx::String& name = "Line");

        /// \return The entity model singleton.
        static LineModel* instance();

        /// Create and return a pointer to a new instance in the default storage for this entity model.
        static LinePtr createInstance();

        /// \return The default storage for this entity model.
        static agxData::EntityStorage* defaultStorage();

        /// This is part of internal cleanup and should not be called by users
        virtual void shutdownCleanup() override;



        /* Attributes */
        static agxData::ScalarAttributeT< agx::Line >* lineAttribute;

      protected:
        virtual ~LineModel();
        virtual agxData::EntityData* createData(agxData::EntityStorage* storage) override;
        virtual void configure(agx::TiXmlElement* eEntity) override;
        virtual void initAttributeAccessors() override;
        virtual void construct(agxData::EntityPtr instance) override;
        void construct(agx::Physics::Geometry::LinePtr line);
      };


      DOXYGEN_START_INTERNAL_BLOCK()
      #ifndef AGX_PHYSICS_GEOMETRY_LINE_DATA_SET_OVERRIDE
      #define AGX_PHYSICS_GEOMETRY_LINE_DATA_SET
      class AGXPHYSICS_EXPORT LineData : public agx::Physics::Geometry::ShapeData
      {
      public:
        LineInstance operator[] (size_t index);

      public:
        agxData::Array< LinePtr >& instance;
        agxData::Array< agx::Line > line;

      public:
        typedef agx::Line lineType;

      public:
        LineData(agxData::EntityStorage* storage);
        LineData();

      protected:
        virtual ~LineData() {}
        virtual void setNumElements(agx::Index numElements) override;

      private:
        LineData& operator= (const LineData&) { return *this; }

      };
      #endif
      DOXYGEN_END_INTERNAL_BLOCK()


      DOXYGEN_START_INTERNAL_BLOCK()
      class AGXPHYSICS_EXPORT LineSemantics : public agx::Physics::Geometry::ShapeSemantics
      {
      public:

        // Automatic getters
        agx::Line const& getLine() const;

        // Semantics defined by explicit kernels

        // Automatic setters
        void setLine(agx::Line const& value);


      protected:
        friend class LinePtr;
        friend class LineInstance;
        LineSemantics();
      };
      DOXYGEN_END_INTERNAL_BLOCK()


      /**
      Pointer to a entity instance of type Physics.Geometry.Line
      */
      class CALLABLE LinePtr : public agx::Physics::Geometry::ShapePtr
      {
      public:
        typedef LineModel ModelType;
        typedef LineData DataType;
        typedef LineInstance InstanceType;

      public:
        AGXPHYSICS_EXPORT LinePtr();
        AGXPHYSICS_EXPORT LinePtr(agxData::EntityStorage* storage, agx::Index id);
        AGXPHYSICS_EXPORT LinePtr(const agxData::EntityPtr& ptr);
        AGXPHYSICS_EXPORT LinePtr(const agxData::EntityInstance& instance);
        AGXPHYSICS_EXPORT LinePtr& operator= (const agxData::EntityPtr& ptr);
        AGXPHYSICS_EXPORT LinePtr& operator= (const agxData::EntityInstance& instance);
        AGXPHYSICS_EXPORT LineInstance instance();
        AGXPHYSICS_EXPORT const LineInstance instance() const;

        AGXPHYSICS_EXPORT LineSemantics* operator->();
        AGXPHYSICS_EXPORT const LineSemantics* operator->() const;

        LineData* getData();
        const LineData* getData() const;


        /// \return reference to the line attribute
        AGXPHYSICS_EXPORT agx::Line& line();
        /// \return const reference to the line attribute
        AGXPHYSICS_EXPORT agx::Line const& line() const;

        DOXYGEN_START_INTERNAL_BLOCK()
        // Only used by tolua!
        #ifdef AGX_GENERATED_TOLUA_WRAPPER
        inline agx::Line const& getLine() const { return (*this)->getLine(); }


        inline void setLine(agx::Line const& value) { (*this)->setLine(value); }
        #endif
        DOXYGEN_END_INTERNAL_BLOCK()

      };

      // Entity is Referenced
      typedef agxData::EntityRef< LinePtr > LineRef;


      DOXYGEN_START_INTERNAL_BLOCK()
      class AGXPHYSICS_EXPORT LineInstance : public agx::Physics::Geometry::ShapeInstance
      {
      public:
        LineInstance();
        LineInstance(LineData* data, agx::Index index);
        LineInstance(agxData::EntityStorage *storage, agx::Index index);
        LineInstance(const agxData::EntityInstance& other);
        LineInstance(const agxData::EntityPtr& ptr);

        LineData* getData();
        const LineData* getData() const;

      public:
        /// \return reference to the line attribute
        agx::Line& line();
        /// \return const reference to the line attribute
        agx::Line const& line() const;

      };
      DOXYGEN_END_INTERNAL_BLOCK()



      typedef agx::VectorPOD<LinePtr> LinePtrVector;
      typedef agxData::Array<LinePtr> LinePtrArray;



      DOXYGEN_START_INTERNAL_BLOCK()
      /* Implementation */
      //-----------------------------------------------------------------------------------------------------
      //-----------------------------------------------------------------------------------------------------
      inline LineInstance agx::Physics::Geometry::LineData::operator[] (size_t index) { return LineInstance(this, (agx::Index)index); }
      //-----------------------------------------------------------------------------------------------------
      AGX_FORCE_INLINE LinePtr::LinePtr() {}
      AGX_FORCE_INLINE LinePtr::LinePtr(agxData::EntityStorage* storage, agx::Index id) : agx::Physics::Geometry::ShapePtr(storage, id) {}
      AGX_FORCE_INLINE LinePtr::LinePtr(const agxData::EntityPtr& ptr) : agx::Physics::Geometry::ShapePtr(ptr)
      {
        agxAssertN(!ptr || ptr.isInstanceOf(LineModel::instance()),
          "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
          EntityPtr::getModel()->fullPath().c_str(), LineModel::instance()->fullPath().c_str());
      }

      AGX_FORCE_INLINE LinePtr::LinePtr(const agxData::EntityInstance& instance) : agx::Physics::Geometry::ShapePtr(instance)
      {
        agxAssertN(!instance || instance.isInstanceOf(LineModel::instance()),
          "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
          EntityPtr::getModel()->fullPath().c_str(), LineModel::instance()->fullPath().c_str());
      }

      AGX_FORCE_INLINE LinePtr& LinePtr::operator= (const agxData::EntityPtr& ptr)
      {
        agxData::EntityPtr::operator= (ptr);
        agxAssertN(!ptr || ptr.isInstanceOf(LineModel::instance()),
          "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
          EntityPtr::getModel()->fullPath().c_str(), LineModel::instance()->fullPath().c_str());
        return *this;
      }

      AGX_FORCE_INLINE LinePtr& LinePtr::operator= (const agxData::EntityInstance& instance)
      {
        agxData::EntityPtr::operator= (instance);
        agxAssertN(!instance || instance.isInstanceOf(LineModel::instance()),
          "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
          EntityPtr::getModel()->fullPath().c_str(), LineModel::instance()->fullPath().c_str());
        return *this;
      }

      inline LineInstance LinePtr::instance() { return agxData::EntityPtr::instance(); }
      inline const LineInstance LinePtr::instance() const { return agxData::EntityPtr::instance(); }
      AGX_FORCE_INLINE LineSemantics* LinePtr::operator->() { return (LineSemantics* )this; }
      AGX_FORCE_INLINE const LineSemantics* LinePtr::operator->() const { return (const LineSemantics* )this; }
      AGX_FORCE_INLINE LineData* LinePtr::getData() { return static_cast<LineData* >(agxData::EntityPtr::getData()); }
      AGX_FORCE_INLINE const LineData* LinePtr::getData() const { return static_cast<const LineData* >(agxData::EntityPtr::getData()); }

      AGX_FORCE_INLINE agx::Line& LinePtr::line() { verifyIndex(); return getData()->line[calculateIndex()]; }
      AGX_FORCE_INLINE agx::Line const& LinePtr::line() const { verifyIndex(); return getData()->line[calculateIndex()]; }

      //-----------------------------------------------------------------------------------------------------
      AGX_FORCE_INLINE LineInstance::LineInstance() {}
      AGX_FORCE_INLINE LineInstance::LineInstance(LineData* data, agx::Index index) : agx::Physics::Geometry::ShapeInstance(data, index) {}
      AGX_FORCE_INLINE LineInstance::LineInstance(agxData::EntityStorage* storage, agx::Index index) : agx::Physics::Geometry::ShapeInstance(storage, index) {}
      AGX_FORCE_INLINE LineInstance::LineInstance(const agxData::EntityInstance& other) : agx::Physics::Geometry::ShapeInstance(other)
      {
        agxAssertN(!other || other.isInstanceOf(LineModel::instance()),
          "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
          EntityInstance::getModel()->fullPath().c_str(), LineModel::instance()->fullPath().c_str());
      }

      AGX_FORCE_INLINE LineInstance::LineInstance(const agxData::EntityPtr& ptr) : agx::Physics::Geometry::ShapeInstance(ptr)
      {
        agxAssertN(!ptr || ptr.isInstanceOf(LineModel::instance()),
          "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
          EntityInstance::getModel()->fullPath().c_str(), LineModel::instance()->fullPath().c_str());
      }


      AGX_FORCE_INLINE LineData* LineInstance::getData() { return static_cast<LineData* >(agxData::EntityInstance::getData()); }
      AGX_FORCE_INLINE const LineData* LineInstance::getData() const { return static_cast<const LineData* >(agxData::EntityInstance::getData()); }

      AGX_FORCE_INLINE agx::Line& LineInstance::line() { verifyIndex(); return getData()->line[getIndex()]; }
      AGX_FORCE_INLINE agx::Line const& LineInstance::line() const { verifyIndex(); return getData()->line[getIndex()]; }

      //-----------------------------------------------------------------------------------------------------
      AGX_FORCE_INLINE LineSemantics::LineSemantics() {}
      //-----------------------------------------------------------------------------------------------------
      DOXYGEN_END_INTERNAL_BLOCK()
    }
  }
}

AGX_TYPE_BINDING(agx::Physics::Geometry::LinePtr, "Physics.Geometry.LinePtr")
AGX_TYPE_BINDING(agx::Physics::Geometry::LineInstance, "Physics.Geometry.LineInstance")

#ifdef _MSC_VER
# pragma warning(pop)
#endif

#undef AGX_ENTITY_WRAPPER
#undef AGX_ENTITY_NAMESPACE
#endif

