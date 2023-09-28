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

#ifndef GENERATED_AGX_PHYSICS_SOLVEMATERIAL32_H_PLUGIN
#define GENERATED_AGX_PHYSICS_SOLVEMATERIAL32_H_PLUGIN

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
#include <agx/Real.h>
#include <agx/Vec2.h>


namespace agx
{
  namespace Physics
  {

    class SolveMaterial32Model;
    class SolveMaterial32Data;
    class SolveMaterial32Ptr;
    class SolveMaterial32Instance;
    class SolveMaterial32Semantics;


    AGX_DECLARE_POINTER_TYPES(SolveMaterial32Model);

    /** 
    Abstract description of the data attributes for the Physics.SolveMaterial32 entity.
    */ 
    class AGXPHYSICS_EXPORT SolveMaterial32Model : public agxData::EntityModel
    {
    public:
      typedef SolveMaterial32Ptr PtrT;

      SolveMaterial32Model(const agx::String& name = "SolveMaterial32");

      /// \return The entity model singleton.
      static SolveMaterial32Model* instance();

      /// Create and return a pointer to a new instance in the default storage for this entity model.
      static SolveMaterial32Ptr createInstance();

      /// \return The default storage for this entity model.
      static agxData::EntityStorage* defaultStorage();

      /// This is part of internal cleanup and should not be called by users
      virtual void shutdownCleanup() override;



      /* Attributes */
      static agxData::ScalarAttributeT< agx::Real32 >* adhesiveImpulseAttribute;
      static agxData::ScalarAttributeT< agx::Vec2f >* frictionAttribute;

    protected:
      virtual ~SolveMaterial32Model();
      virtual agxData::EntityData* createData(agxData::EntityStorage* storage) override;
      virtual void configure(agx::TiXmlElement* eEntity) override;
      virtual void initAttributeAccessors() override;
      virtual void construct(agxData::EntityPtr instance) override;
      void construct(agx::Physics::SolveMaterial32Ptr solveMaterial32);
    };


    DOXYGEN_START_INTERNAL_BLOCK()
    #ifndef AGX_PHYSICS_SOLVEMATERIAL32_DATA_SET_OVERRIDE
    #define AGX_PHYSICS_SOLVEMATERIAL32_DATA_SET
    class AGXPHYSICS_EXPORT SolveMaterial32Data : public agxData::EntityData
    {
    public:
      SolveMaterial32Instance operator[] (size_t index);

    public:
      agxData::Array< SolveMaterial32Ptr >& instance;
      agxData::Array< agx::Real32 > adhesiveImpulse;
      agxData::Array< agx::Vec2f > friction;

    public:
      typedef agx::Real32 adhesiveImpulseType;
      typedef agx::Vec2f frictionType;

    public:
      SolveMaterial32Data(agxData::EntityStorage* storage);
      SolveMaterial32Data();

    protected:
      virtual ~SolveMaterial32Data() {}
      virtual void setNumElements(agx::Index numElements) override;

    private:
      SolveMaterial32Data& operator= (const SolveMaterial32Data&) { return *this; }

    };
    #endif
    DOXYGEN_END_INTERNAL_BLOCK()


    DOXYGEN_START_INTERNAL_BLOCK()
    class AGXPHYSICS_EXPORT SolveMaterial32Semantics : protected agxData::EntityPtr
    {
    public:

      // Automatic getters
      agx::Real32 const& getAdhesiveImpulse() const;
      agx::Vec2f const& getFriction() const;

      // Semantics defined by explicit kernels

      // Automatic setters
      void setAdhesiveImpulse(agx::Real32 const& value);
      void setFriction(agx::Vec2f const& value);


    protected:
      friend class SolveMaterial32Ptr;
      friend class SolveMaterial32Instance;
      SolveMaterial32Semantics();
    };
    DOXYGEN_END_INTERNAL_BLOCK()


    /**
    Pointer to a entity instance of type Physics.SolveMaterial32
    */
    class CALLABLE SolveMaterial32Ptr : public agxData::EntityPtr
    {
    public:
      typedef SolveMaterial32Model ModelType;
      typedef SolveMaterial32Data DataType;
      typedef SolveMaterial32Instance InstanceType;

    public:
      AGXPHYSICS_EXPORT SolveMaterial32Ptr();
      AGXPHYSICS_EXPORT SolveMaterial32Ptr(agxData::EntityStorage* storage, agx::Index id);
      AGXPHYSICS_EXPORT SolveMaterial32Ptr(const agxData::EntityPtr& ptr);
      AGXPHYSICS_EXPORT SolveMaterial32Ptr(const agxData::EntityInstance& instance);
      AGXPHYSICS_EXPORT SolveMaterial32Ptr& operator= (const agxData::EntityPtr& ptr);
      AGXPHYSICS_EXPORT SolveMaterial32Ptr& operator= (const agxData::EntityInstance& instance);
      AGXPHYSICS_EXPORT SolveMaterial32Instance instance();
      AGXPHYSICS_EXPORT const SolveMaterial32Instance instance() const;

      AGXPHYSICS_EXPORT SolveMaterial32Semantics* operator->();
      AGXPHYSICS_EXPORT const SolveMaterial32Semantics* operator->() const;

      SolveMaterial32Data* getData();
      const SolveMaterial32Data* getData() const;


      /// \return reference to the adhesiveImpulse attribute
      AGXPHYSICS_EXPORT agx::Real32& adhesiveImpulse();
      /// \return const reference to the adhesiveImpulse attribute
      AGXPHYSICS_EXPORT agx::Real32 const& adhesiveImpulse() const;

      /// \return reference to the friction attribute
      AGXPHYSICS_EXPORT agx::Vec2f& friction();
      /// \return const reference to the friction attribute
      AGXPHYSICS_EXPORT agx::Vec2f const& friction() const;

      DOXYGEN_START_INTERNAL_BLOCK()
      // Only used by tolua!
      #ifdef AGX_GENERATED_TOLUA_WRAPPER
      inline agx::Real32 const& getAdhesiveImpulse() const { return (*this)->getAdhesiveImpulse(); }
      inline agx::Vec2f const& getFriction() const { return (*this)->getFriction(); }


      inline void setAdhesiveImpulse(agx::Real32 const& value) { (*this)->setAdhesiveImpulse(value); }
      inline void setFriction(agx::Vec2f const& value) { (*this)->setFriction(value); }
      #endif
      DOXYGEN_END_INTERNAL_BLOCK()

    };


    DOXYGEN_START_INTERNAL_BLOCK()
    class AGXPHYSICS_EXPORT SolveMaterial32Instance : public agxData::EntityInstance
    {
    public:
      SolveMaterial32Instance();
      SolveMaterial32Instance(SolveMaterial32Data* data, agx::Index index);
      SolveMaterial32Instance(agxData::EntityStorage *storage, agx::Index index);
      SolveMaterial32Instance(const agxData::EntityInstance& other);
      SolveMaterial32Instance(const agxData::EntityPtr& ptr);

      SolveMaterial32Data* getData();
      const SolveMaterial32Data* getData() const;

    public:
      /// \return reference to the adhesiveImpulse attribute
      agx::Real32& adhesiveImpulse();
      /// \return const reference to the adhesiveImpulse attribute
      agx::Real32 const& adhesiveImpulse() const;

      /// \return reference to the friction attribute
      agx::Vec2f& friction();
      /// \return const reference to the friction attribute
      agx::Vec2f const& friction() const;

    };
    DOXYGEN_END_INTERNAL_BLOCK()



    typedef agx::VectorPOD<SolveMaterial32Ptr> SolveMaterial32PtrVector;
    typedef agxData::Array<SolveMaterial32Ptr> SolveMaterial32PtrArray;



    DOXYGEN_START_INTERNAL_BLOCK()
    /* Implementation */
    //-----------------------------------------------------------------------------------------------------
    //-----------------------------------------------------------------------------------------------------
    inline SolveMaterial32Instance agx::Physics::SolveMaterial32Data::operator[] (size_t index) { return SolveMaterial32Instance(this, (agx::Index)index); }
    //-----------------------------------------------------------------------------------------------------
    AGX_FORCE_INLINE SolveMaterial32Ptr::SolveMaterial32Ptr() {}
    AGX_FORCE_INLINE SolveMaterial32Ptr::SolveMaterial32Ptr(agxData::EntityStorage* storage, agx::Index id) : agxData::EntityPtr(storage, id) {}
    AGX_FORCE_INLINE SolveMaterial32Ptr::SolveMaterial32Ptr(const agxData::EntityPtr& ptr) : agxData::EntityPtr(ptr)
    {
      agxAssertN(!ptr || ptr.isInstanceOf(SolveMaterial32Model::instance()),
        "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
        EntityPtr::getModel()->fullPath().c_str(), SolveMaterial32Model::instance()->fullPath().c_str());
    }

    AGX_FORCE_INLINE SolveMaterial32Ptr::SolveMaterial32Ptr(const agxData::EntityInstance& instance) : agxData::EntityPtr(instance)
    {
      agxAssertN(!instance || instance.isInstanceOf(SolveMaterial32Model::instance()),
        "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
        EntityPtr::getModel()->fullPath().c_str(), SolveMaterial32Model::instance()->fullPath().c_str());
    }

    AGX_FORCE_INLINE SolveMaterial32Ptr& SolveMaterial32Ptr::operator= (const agxData::EntityPtr& ptr)
    {
      agxData::EntityPtr::operator= (ptr);
      agxAssertN(!ptr || ptr.isInstanceOf(SolveMaterial32Model::instance()),
        "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
        EntityPtr::getModel()->fullPath().c_str(), SolveMaterial32Model::instance()->fullPath().c_str());
      return *this;
    }

    AGX_FORCE_INLINE SolveMaterial32Ptr& SolveMaterial32Ptr::operator= (const agxData::EntityInstance& instance)
    {
      agxData::EntityPtr::operator= (instance);
      agxAssertN(!instance || instance.isInstanceOf(SolveMaterial32Model::instance()),
        "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
        EntityPtr::getModel()->fullPath().c_str(), SolveMaterial32Model::instance()->fullPath().c_str());
      return *this;
    }

    inline SolveMaterial32Instance SolveMaterial32Ptr::instance() { return agxData::EntityPtr::instance(); }
    inline const SolveMaterial32Instance SolveMaterial32Ptr::instance() const { return agxData::EntityPtr::instance(); }
    AGX_FORCE_INLINE SolveMaterial32Semantics* SolveMaterial32Ptr::operator->() { return (SolveMaterial32Semantics* )this; }
    AGX_FORCE_INLINE const SolveMaterial32Semantics* SolveMaterial32Ptr::operator->() const { return (const SolveMaterial32Semantics* )this; }
    AGX_FORCE_INLINE SolveMaterial32Data* SolveMaterial32Ptr::getData() { return static_cast<SolveMaterial32Data* >(agxData::EntityPtr::getData()); }
    AGX_FORCE_INLINE const SolveMaterial32Data* SolveMaterial32Ptr::getData() const { return static_cast<const SolveMaterial32Data* >(agxData::EntityPtr::getData()); }

    AGX_FORCE_INLINE agx::Real32& SolveMaterial32Ptr::adhesiveImpulse() { verifyIndex(); return getData()->adhesiveImpulse[calculateIndex()]; }
    AGX_FORCE_INLINE agx::Real32 const& SolveMaterial32Ptr::adhesiveImpulse() const { verifyIndex(); return getData()->adhesiveImpulse[calculateIndex()]; }

    AGX_FORCE_INLINE agx::Vec2f& SolveMaterial32Ptr::friction() { verifyIndex(); return getData()->friction[calculateIndex()]; }
    AGX_FORCE_INLINE agx::Vec2f const& SolveMaterial32Ptr::friction() const { verifyIndex(); return getData()->friction[calculateIndex()]; }

    //-----------------------------------------------------------------------------------------------------
    AGX_FORCE_INLINE SolveMaterial32Instance::SolveMaterial32Instance() {}
    AGX_FORCE_INLINE SolveMaterial32Instance::SolveMaterial32Instance(SolveMaterial32Data* data, agx::Index index) : agxData::EntityInstance(data, index) {}
    AGX_FORCE_INLINE SolveMaterial32Instance::SolveMaterial32Instance(agxData::EntityStorage* storage, agx::Index index) : agxData::EntityInstance(storage, index) {}
    AGX_FORCE_INLINE SolveMaterial32Instance::SolveMaterial32Instance(const agxData::EntityInstance& other) : agxData::EntityInstance(other)
    {
      agxAssertN(!other || other.isInstanceOf(SolveMaterial32Model::instance()),
        "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
        EntityInstance::getModel()->fullPath().c_str(), SolveMaterial32Model::instance()->fullPath().c_str());
    }

    AGX_FORCE_INLINE SolveMaterial32Instance::SolveMaterial32Instance(const agxData::EntityPtr& ptr) : agxData::EntityInstance(ptr)
    {
      agxAssertN(!ptr || ptr.isInstanceOf(SolveMaterial32Model::instance()),
        "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
        EntityInstance::getModel()->fullPath().c_str(), SolveMaterial32Model::instance()->fullPath().c_str());
    }


    AGX_FORCE_INLINE SolveMaterial32Data* SolveMaterial32Instance::getData() { return static_cast<SolveMaterial32Data* >(agxData::EntityInstance::getData()); }
    AGX_FORCE_INLINE const SolveMaterial32Data* SolveMaterial32Instance::getData() const { return static_cast<const SolveMaterial32Data* >(agxData::EntityInstance::getData()); }

    AGX_FORCE_INLINE agx::Real32& SolveMaterial32Instance::adhesiveImpulse() { verifyIndex(); return getData()->adhesiveImpulse[getIndex()]; }
    AGX_FORCE_INLINE agx::Real32 const& SolveMaterial32Instance::adhesiveImpulse() const { verifyIndex(); return getData()->adhesiveImpulse[getIndex()]; }

    AGX_FORCE_INLINE agx::Vec2f& SolveMaterial32Instance::friction() { verifyIndex(); return getData()->friction[getIndex()]; }
    AGX_FORCE_INLINE agx::Vec2f const& SolveMaterial32Instance::friction() const { verifyIndex(); return getData()->friction[getIndex()]; }

    //-----------------------------------------------------------------------------------------------------
    AGX_FORCE_INLINE SolveMaterial32Semantics::SolveMaterial32Semantics() {}
    //-----------------------------------------------------------------------------------------------------
    DOXYGEN_END_INTERNAL_BLOCK()
  }
}

AGX_TYPE_BINDING(agx::Physics::SolveMaterial32Ptr, "Physics.SolveMaterial32Ptr")
AGX_TYPE_BINDING(agx::Physics::SolveMaterial32Instance, "Physics.SolveMaterial32Instance")

#ifdef _MSC_VER
# pragma warning(pop)
#endif

#undef AGX_ENTITY_WRAPPER
#undef AGX_ENTITY_NAMESPACE
#endif

