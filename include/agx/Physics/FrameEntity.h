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

#ifndef GENERATED_AGX_PHYSICS_FRAME_H_PLUGIN
#define GENERATED_AGX_PHYSICS_FRAME_H_PLUGIN

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
#include <agx/ReferencedEntity.h>
#include <agx/AffineMatrix4x4.h>


namespace agx
{
  namespace Physics
  {

    class FrameModel;
    class FrameData;
    class FramePtr;
    class FrameInstance;
    class FrameSemantics;


    AGX_DECLARE_POINTER_TYPES(FrameModel);

    /** 
    Abstract description of the data attributes for the Physics.Frame entity.
    */ 
    class AGXPHYSICS_EXPORT FrameModel : public agx::ReferencedModel
    {
    public:
      typedef FramePtr PtrT;

      FrameModel(const agx::String& name = "Frame");

      /// \return The entity model singleton.
      static FrameModel* instance();

      /// Create and return a pointer to a new instance in the default storage for this entity model.
      static FramePtr createInstance();

      /// \return The default storage for this entity model.
      static agxData::EntityStorage* defaultStorage();

      /// This is part of internal cleanup and should not be called by users
      virtual void shutdownCleanup() override;



      /* Attributes */
      static agxData::ScalarAttributeT< agx::AffineMatrix4x4 >* globalTransformAttribute;
      static agxData::ScalarAttributeT< agx::AffineMatrix4x4 >* localTransformAttribute;

    protected:
      virtual ~FrameModel();
      virtual agxData::EntityData* createData(agxData::EntityStorage* storage) override;
      virtual void configure(agx::TiXmlElement* eEntity) override;
      virtual void initAttributeAccessors() override;
      virtual void construct(agxData::EntityPtr instance) override;
      void construct(agx::Physics::FramePtr frame);
    };


    DOXYGEN_START_INTERNAL_BLOCK()
    #ifndef AGX_PHYSICS_FRAME_DATA_SET_OVERRIDE
    #define AGX_PHYSICS_FRAME_DATA_SET
    class AGXPHYSICS_EXPORT FrameData : public agx::ReferencedData
    {
    public:
      FrameInstance operator[] (size_t index);

    public:
      agxData::Array< FramePtr >& instance;
      agxData::Array< agx::AffineMatrix4x4 > globalTransform;
      agxData::Array< agx::AffineMatrix4x4 > localTransform;

    public:
      typedef agx::AffineMatrix4x4 globalTransformType;
      typedef agx::AffineMatrix4x4 localTransformType;

    public:
      FrameData(agxData::EntityStorage* storage);
      FrameData();

    protected:
      virtual ~FrameData() {}
      virtual void setNumElements(agx::Index numElements) override;

    private:
      FrameData& operator= (const FrameData&) { return *this; }

    };
    #endif
    DOXYGEN_END_INTERNAL_BLOCK()


    DOXYGEN_START_INTERNAL_BLOCK()
    class AGXPHYSICS_EXPORT FrameSemantics : public agx::ReferencedSemantics
    {
    public:

      // Automatic getters
      agx::AffineMatrix4x4 const& getGlobalTransform() const;
      agx::AffineMatrix4x4 const& getLocalTransform() const;

      // Semantics defined by explicit kernels

      // Automatic setters
      void setGlobalTransform(agx::AffineMatrix4x4 const& value);
      void setLocalTransform(agx::AffineMatrix4x4 const& value);


    protected:
      friend class FramePtr;
      friend class FrameInstance;
      FrameSemantics();
    };
    DOXYGEN_END_INTERNAL_BLOCK()


    /**
    Pointer to a entity instance of type Physics.Frame
    */
    class CALLABLE FramePtr : public agx::ReferencedPtr
    {
    public:
      typedef FrameModel ModelType;
      typedef FrameData DataType;
      typedef FrameInstance InstanceType;

    public:
      AGXPHYSICS_EXPORT FramePtr();
      AGXPHYSICS_EXPORT FramePtr(agxData::EntityStorage* storage, agx::Index id);
      AGXPHYSICS_EXPORT FramePtr(const agxData::EntityPtr& ptr);
      AGXPHYSICS_EXPORT FramePtr(const agxData::EntityInstance& instance);
      AGXPHYSICS_EXPORT FramePtr& operator= (const agxData::EntityPtr& ptr);
      AGXPHYSICS_EXPORT FramePtr& operator= (const agxData::EntityInstance& instance);
      AGXPHYSICS_EXPORT FrameInstance instance();
      AGXPHYSICS_EXPORT const FrameInstance instance() const;

      AGXPHYSICS_EXPORT FrameSemantics* operator->();
      AGXPHYSICS_EXPORT const FrameSemantics* operator->() const;

      FrameData* getData();
      const FrameData* getData() const;


      /// \return reference to the globalTransform attribute
      AGXPHYSICS_EXPORT agx::AffineMatrix4x4& globalTransform();
      /// \return const reference to the globalTransform attribute
      AGXPHYSICS_EXPORT agx::AffineMatrix4x4 const& globalTransform() const;

      /// \return reference to the localTransform attribute
      AGXPHYSICS_EXPORT agx::AffineMatrix4x4& localTransform();
      /// \return const reference to the localTransform attribute
      AGXPHYSICS_EXPORT agx::AffineMatrix4x4 const& localTransform() const;

      DOXYGEN_START_INTERNAL_BLOCK()
      // Only used by tolua!
      #ifdef AGX_GENERATED_TOLUA_WRAPPER
      inline agx::AffineMatrix4x4 const& getGlobalTransform() const { return (*this)->getGlobalTransform(); }
      inline agx::AffineMatrix4x4 const& getLocalTransform() const { return (*this)->getLocalTransform(); }


      inline void setGlobalTransform(agx::AffineMatrix4x4 const& value) { (*this)->setGlobalTransform(value); }
      inline void setLocalTransform(agx::AffineMatrix4x4 const& value) { (*this)->setLocalTransform(value); }
      #endif
      DOXYGEN_END_INTERNAL_BLOCK()

    };

    // Entity is Referenced
    typedef agxData::EntityRef< FramePtr > FrameRef;


    DOXYGEN_START_INTERNAL_BLOCK()
    class AGXPHYSICS_EXPORT FrameInstance : public agx::ReferencedInstance
    {
    public:
      FrameInstance();
      FrameInstance(FrameData* data, agx::Index index);
      FrameInstance(agxData::EntityStorage *storage, agx::Index index);
      FrameInstance(const agxData::EntityInstance& other);
      FrameInstance(const agxData::EntityPtr& ptr);

      FrameData* getData();
      const FrameData* getData() const;

    public:
      /// \return reference to the globalTransform attribute
      agx::AffineMatrix4x4& globalTransform();
      /// \return const reference to the globalTransform attribute
      agx::AffineMatrix4x4 const& globalTransform() const;

      /// \return reference to the localTransform attribute
      agx::AffineMatrix4x4& localTransform();
      /// \return const reference to the localTransform attribute
      agx::AffineMatrix4x4 const& localTransform() const;

    };
    DOXYGEN_END_INTERNAL_BLOCK()



    typedef agx::VectorPOD<FramePtr> FramePtrVector;
    typedef agxData::Array<FramePtr> FramePtrArray;



    DOXYGEN_START_INTERNAL_BLOCK()
    /* Implementation */
    //-----------------------------------------------------------------------------------------------------
    //-----------------------------------------------------------------------------------------------------
    inline FrameInstance agx::Physics::FrameData::operator[] (size_t index) { return FrameInstance(this, (agx::Index)index); }
    //-----------------------------------------------------------------------------------------------------
    AGX_FORCE_INLINE FramePtr::FramePtr() {}
    AGX_FORCE_INLINE FramePtr::FramePtr(agxData::EntityStorage* storage, agx::Index id) : agx::ReferencedPtr(storage, id) {}
    AGX_FORCE_INLINE FramePtr::FramePtr(const agxData::EntityPtr& ptr) : agx::ReferencedPtr(ptr)
    {
      agxAssertN(!ptr || ptr.isInstanceOf(FrameModel::instance()),
        "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
        EntityPtr::getModel()->fullPath().c_str(), FrameModel::instance()->fullPath().c_str());
    }

    AGX_FORCE_INLINE FramePtr::FramePtr(const agxData::EntityInstance& instance) : agx::ReferencedPtr(instance)
    {
      agxAssertN(!instance || instance.isInstanceOf(FrameModel::instance()),
        "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
        EntityPtr::getModel()->fullPath().c_str(), FrameModel::instance()->fullPath().c_str());
    }

    AGX_FORCE_INLINE FramePtr& FramePtr::operator= (const agxData::EntityPtr& ptr)
    {
      agxData::EntityPtr::operator= (ptr);
      agxAssertN(!ptr || ptr.isInstanceOf(FrameModel::instance()),
        "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
        EntityPtr::getModel()->fullPath().c_str(), FrameModel::instance()->fullPath().c_str());
      return *this;
    }

    AGX_FORCE_INLINE FramePtr& FramePtr::operator= (const agxData::EntityInstance& instance)
    {
      agxData::EntityPtr::operator= (instance);
      agxAssertN(!instance || instance.isInstanceOf(FrameModel::instance()),
        "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
        EntityPtr::getModel()->fullPath().c_str(), FrameModel::instance()->fullPath().c_str());
      return *this;
    }

    inline FrameInstance FramePtr::instance() { return agxData::EntityPtr::instance(); }
    inline const FrameInstance FramePtr::instance() const { return agxData::EntityPtr::instance(); }
    AGX_FORCE_INLINE FrameSemantics* FramePtr::operator->() { return (FrameSemantics* )this; }
    AGX_FORCE_INLINE const FrameSemantics* FramePtr::operator->() const { return (const FrameSemantics* )this; }
    AGX_FORCE_INLINE FrameData* FramePtr::getData() { return static_cast<FrameData* >(agxData::EntityPtr::getData()); }
    AGX_FORCE_INLINE const FrameData* FramePtr::getData() const { return static_cast<const FrameData* >(agxData::EntityPtr::getData()); }

    AGX_FORCE_INLINE agx::AffineMatrix4x4& FramePtr::globalTransform() { verifyIndex(); return getData()->globalTransform[calculateIndex()]; }
    AGX_FORCE_INLINE agx::AffineMatrix4x4 const& FramePtr::globalTransform() const { verifyIndex(); return getData()->globalTransform[calculateIndex()]; }

    AGX_FORCE_INLINE agx::AffineMatrix4x4& FramePtr::localTransform() { verifyIndex(); return getData()->localTransform[calculateIndex()]; }
    AGX_FORCE_INLINE agx::AffineMatrix4x4 const& FramePtr::localTransform() const { verifyIndex(); return getData()->localTransform[calculateIndex()]; }

    //-----------------------------------------------------------------------------------------------------
    AGX_FORCE_INLINE FrameInstance::FrameInstance() {}
    AGX_FORCE_INLINE FrameInstance::FrameInstance(FrameData* data, agx::Index index) : agx::ReferencedInstance(data, index) {}
    AGX_FORCE_INLINE FrameInstance::FrameInstance(agxData::EntityStorage* storage, agx::Index index) : agx::ReferencedInstance(storage, index) {}
    AGX_FORCE_INLINE FrameInstance::FrameInstance(const agxData::EntityInstance& other) : agx::ReferencedInstance(other)
    {
      agxAssertN(!other || other.isInstanceOf(FrameModel::instance()),
        "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
        EntityInstance::getModel()->fullPath().c_str(), FrameModel::instance()->fullPath().c_str());
    }

    AGX_FORCE_INLINE FrameInstance::FrameInstance(const agxData::EntityPtr& ptr) : agx::ReferencedInstance(ptr)
    {
      agxAssertN(!ptr || ptr.isInstanceOf(FrameModel::instance()),
        "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
        EntityInstance::getModel()->fullPath().c_str(), FrameModel::instance()->fullPath().c_str());
    }


    AGX_FORCE_INLINE FrameData* FrameInstance::getData() { return static_cast<FrameData* >(agxData::EntityInstance::getData()); }
    AGX_FORCE_INLINE const FrameData* FrameInstance::getData() const { return static_cast<const FrameData* >(agxData::EntityInstance::getData()); }

    AGX_FORCE_INLINE agx::AffineMatrix4x4& FrameInstance::globalTransform() { verifyIndex(); return getData()->globalTransform[getIndex()]; }
    AGX_FORCE_INLINE agx::AffineMatrix4x4 const& FrameInstance::globalTransform() const { verifyIndex(); return getData()->globalTransform[getIndex()]; }

    AGX_FORCE_INLINE agx::AffineMatrix4x4& FrameInstance::localTransform() { verifyIndex(); return getData()->localTransform[getIndex()]; }
    AGX_FORCE_INLINE agx::AffineMatrix4x4 const& FrameInstance::localTransform() const { verifyIndex(); return getData()->localTransform[getIndex()]; }

    //-----------------------------------------------------------------------------------------------------
    AGX_FORCE_INLINE FrameSemantics::FrameSemantics() {}
    //-----------------------------------------------------------------------------------------------------
    DOXYGEN_END_INTERNAL_BLOCK()
  }
}

AGX_TYPE_BINDING(agx::Physics::FramePtr, "Physics.FramePtr")
AGX_TYPE_BINDING(agx::Physics::FrameInstance, "Physics.FrameInstance")

#ifdef _MSC_VER
# pragma warning(pop)
#endif

#undef AGX_ENTITY_WRAPPER
#undef AGX_ENTITY_NAMESPACE
#endif

