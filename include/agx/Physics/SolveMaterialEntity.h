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

#ifndef GENERATED_AGX_PHYSICS_SOLVEMATERIAL_H_PLUGIN
#define GENERATED_AGX_PHYSICS_SOLVEMATERIAL_H_PLUGIN

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

    class SolveMaterialModel;
    class SolveMaterialData;
    class SolveMaterialPtr;
    class SolveMaterialInstance;
    class SolveMaterialSemantics;


    AGX_DECLARE_POINTER_TYPES(SolveMaterialModel);

    /** 
    Abstract description of the data attributes for the Physics.SolveMaterial entity.
    */ 
    class AGXPHYSICS_EXPORT SolveMaterialModel : public agxData::EntityModel
    {
    public:
      typedef SolveMaterialPtr PtrT;

      SolveMaterialModel(const agx::String& name = "SolveMaterial");

      /// \return The entity model singleton.
      static SolveMaterialModel* instance();

      /// Create and return a pointer to a new instance in the default storage for this entity model.
      static SolveMaterialPtr createInstance();

      /// \return The default storage for this entity model.
      static agxData::EntityStorage* defaultStorage();

      /// This is part of internal cleanup and should not be called by users
      virtual void shutdownCleanup() override;



      /* Attributes */
      static agxData::ScalarAttributeT< agx::Real >* adhesiveImpulseAttribute;
      static agxData::ScalarAttributeT< agx::Vec2 >* frictionAttribute;

    protected:
      virtual ~SolveMaterialModel();
      virtual agxData::EntityData* createData(agxData::EntityStorage* storage) override;
      virtual void configure(agx::TiXmlElement* eEntity) override;
      virtual void initAttributeAccessors() override;
      virtual void construct(agxData::EntityPtr instance) override;
      void construct(agx::Physics::SolveMaterialPtr solveMaterial);
    };


    DOXYGEN_START_INTERNAL_BLOCK()
    #ifndef AGX_PHYSICS_SOLVEMATERIAL_DATA_SET_OVERRIDE
    #define AGX_PHYSICS_SOLVEMATERIAL_DATA_SET
    class AGXPHYSICS_EXPORT SolveMaterialData : public agxData::EntityData
    {
    public:
      SolveMaterialInstance operator[] (size_t index);

    public:
      agxData::Array< SolveMaterialPtr >& instance;
      agxData::Array< agx::Real > adhesiveImpulse;
      agxData::Array< agx::Vec2 > friction;

    public:
      typedef agx::Real adhesiveImpulseType;
      typedef agx::Vec2 frictionType;

    public:
      SolveMaterialData(agxData::EntityStorage* storage);
      SolveMaterialData();

    protected:
      virtual ~SolveMaterialData() {}
      virtual void setNumElements(agx::Index numElements) override;

    private:
      SolveMaterialData& operator= (const SolveMaterialData&) { return *this; }

    };
    #endif
    DOXYGEN_END_INTERNAL_BLOCK()


    DOXYGEN_START_INTERNAL_BLOCK()
    class AGXPHYSICS_EXPORT SolveMaterialSemantics : protected agxData::EntityPtr
    {
    public:

      // Automatic getters
      agx::Real const& getAdhesiveImpulse() const;
      agx::Vec2 const& getFriction() const;

      // Semantics defined by explicit kernels

      // Automatic setters
      void setAdhesiveImpulse(agx::Real const& value);
      void setFriction(agx::Vec2 const& value);


    protected:
      friend class SolveMaterialPtr;
      friend class SolveMaterialInstance;
      SolveMaterialSemantics();
    };
    DOXYGEN_END_INTERNAL_BLOCK()


    /**
    Pointer to a entity instance of type Physics.SolveMaterial
    */
    class CALLABLE SolveMaterialPtr : public agxData::EntityPtr
    {
    public:
      typedef SolveMaterialModel ModelType;
      typedef SolveMaterialData DataType;
      typedef SolveMaterialInstance InstanceType;

    public:
      AGXPHYSICS_EXPORT SolveMaterialPtr();
      AGXPHYSICS_EXPORT SolveMaterialPtr(agxData::EntityStorage* storage, agx::Index id);
      AGXPHYSICS_EXPORT SolveMaterialPtr(const agxData::EntityPtr& ptr);
      AGXPHYSICS_EXPORT SolveMaterialPtr(const agxData::EntityInstance& instance);
      AGXPHYSICS_EXPORT SolveMaterialPtr& operator= (const agxData::EntityPtr& ptr);
      AGXPHYSICS_EXPORT SolveMaterialPtr& operator= (const agxData::EntityInstance& instance);
      AGXPHYSICS_EXPORT SolveMaterialInstance instance();
      AGXPHYSICS_EXPORT const SolveMaterialInstance instance() const;

      AGXPHYSICS_EXPORT SolveMaterialSemantics* operator->();
      AGXPHYSICS_EXPORT const SolveMaterialSemantics* operator->() const;

      SolveMaterialData* getData();
      const SolveMaterialData* getData() const;


      /// \return reference to the adhesiveImpulse attribute
      AGXPHYSICS_EXPORT agx::Real& adhesiveImpulse();
      /// \return const reference to the adhesiveImpulse attribute
      AGXPHYSICS_EXPORT agx::Real const& adhesiveImpulse() const;

      /// \return reference to the friction attribute
      AGXPHYSICS_EXPORT agx::Vec2& friction();
      /// \return const reference to the friction attribute
      AGXPHYSICS_EXPORT agx::Vec2 const& friction() const;

      DOXYGEN_START_INTERNAL_BLOCK()
      // Only used by tolua!
      #ifdef AGX_GENERATED_TOLUA_WRAPPER
      inline agx::Real const& getAdhesiveImpulse() const { return (*this)->getAdhesiveImpulse(); }
      inline agx::Vec2 const& getFriction() const { return (*this)->getFriction(); }


      inline void setAdhesiveImpulse(agx::Real const& value) { (*this)->setAdhesiveImpulse(value); }
      inline void setFriction(agx::Vec2 const& value) { (*this)->setFriction(value); }
      #endif
      DOXYGEN_END_INTERNAL_BLOCK()

    };


    DOXYGEN_START_INTERNAL_BLOCK()
    class AGXPHYSICS_EXPORT SolveMaterialInstance : public agxData::EntityInstance
    {
    public:
      SolveMaterialInstance();
      SolveMaterialInstance(SolveMaterialData* data, agx::Index index);
      SolveMaterialInstance(agxData::EntityStorage *storage, agx::Index index);
      SolveMaterialInstance(const agxData::EntityInstance& other);
      SolveMaterialInstance(const agxData::EntityPtr& ptr);

      SolveMaterialData* getData();
      const SolveMaterialData* getData() const;

    public:
      /// \return reference to the adhesiveImpulse attribute
      agx::Real& adhesiveImpulse();
      /// \return const reference to the adhesiveImpulse attribute
      agx::Real const& adhesiveImpulse() const;

      /// \return reference to the friction attribute
      agx::Vec2& friction();
      /// \return const reference to the friction attribute
      agx::Vec2 const& friction() const;

    };
    DOXYGEN_END_INTERNAL_BLOCK()



    typedef agx::VectorPOD<SolveMaterialPtr> SolveMaterialPtrVector;
    typedef agxData::Array<SolveMaterialPtr> SolveMaterialPtrArray;



    DOXYGEN_START_INTERNAL_BLOCK()
    /* Implementation */
    //-----------------------------------------------------------------------------------------------------
    //-----------------------------------------------------------------------------------------------------
    inline SolveMaterialInstance agx::Physics::SolveMaterialData::operator[] (size_t index) { return SolveMaterialInstance(this, (agx::Index)index); }
    //-----------------------------------------------------------------------------------------------------
    AGX_FORCE_INLINE SolveMaterialPtr::SolveMaterialPtr() {}
    AGX_FORCE_INLINE SolveMaterialPtr::SolveMaterialPtr(agxData::EntityStorage* storage, agx::Index id) : agxData::EntityPtr(storage, id) {}
    AGX_FORCE_INLINE SolveMaterialPtr::SolveMaterialPtr(const agxData::EntityPtr& ptr) : agxData::EntityPtr(ptr)
    {
      agxAssertN(!ptr || ptr.isInstanceOf(SolveMaterialModel::instance()),
        "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
        EntityPtr::getModel()->fullPath().c_str(), SolveMaterialModel::instance()->fullPath().c_str());
    }

    AGX_FORCE_INLINE SolveMaterialPtr::SolveMaterialPtr(const agxData::EntityInstance& instance) : agxData::EntityPtr(instance)
    {
      agxAssertN(!instance || instance.isInstanceOf(SolveMaterialModel::instance()),
        "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
        EntityPtr::getModel()->fullPath().c_str(), SolveMaterialModel::instance()->fullPath().c_str());
    }

    AGX_FORCE_INLINE SolveMaterialPtr& SolveMaterialPtr::operator= (const agxData::EntityPtr& ptr)
    {
      agxData::EntityPtr::operator= (ptr);
      agxAssertN(!ptr || ptr.isInstanceOf(SolveMaterialModel::instance()),
        "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
        EntityPtr::getModel()->fullPath().c_str(), SolveMaterialModel::instance()->fullPath().c_str());
      return *this;
    }

    AGX_FORCE_INLINE SolveMaterialPtr& SolveMaterialPtr::operator= (const agxData::EntityInstance& instance)
    {
      agxData::EntityPtr::operator= (instance);
      agxAssertN(!instance || instance.isInstanceOf(SolveMaterialModel::instance()),
        "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
        EntityPtr::getModel()->fullPath().c_str(), SolveMaterialModel::instance()->fullPath().c_str());
      return *this;
    }

    inline SolveMaterialInstance SolveMaterialPtr::instance() { return agxData::EntityPtr::instance(); }
    inline const SolveMaterialInstance SolveMaterialPtr::instance() const { return agxData::EntityPtr::instance(); }
    AGX_FORCE_INLINE SolveMaterialSemantics* SolveMaterialPtr::operator->() { return (SolveMaterialSemantics* )this; }
    AGX_FORCE_INLINE const SolveMaterialSemantics* SolveMaterialPtr::operator->() const { return (const SolveMaterialSemantics* )this; }
    AGX_FORCE_INLINE SolveMaterialData* SolveMaterialPtr::getData() { return static_cast<SolveMaterialData* >(agxData::EntityPtr::getData()); }
    AGX_FORCE_INLINE const SolveMaterialData* SolveMaterialPtr::getData() const { return static_cast<const SolveMaterialData* >(agxData::EntityPtr::getData()); }

    AGX_FORCE_INLINE agx::Real& SolveMaterialPtr::adhesiveImpulse() { verifyIndex(); return getData()->adhesiveImpulse[calculateIndex()]; }
    AGX_FORCE_INLINE agx::Real const& SolveMaterialPtr::adhesiveImpulse() const { verifyIndex(); return getData()->adhesiveImpulse[calculateIndex()]; }

    AGX_FORCE_INLINE agx::Vec2& SolveMaterialPtr::friction() { verifyIndex(); return getData()->friction[calculateIndex()]; }
    AGX_FORCE_INLINE agx::Vec2 const& SolveMaterialPtr::friction() const { verifyIndex(); return getData()->friction[calculateIndex()]; }

    //-----------------------------------------------------------------------------------------------------
    AGX_FORCE_INLINE SolveMaterialInstance::SolveMaterialInstance() {}
    AGX_FORCE_INLINE SolveMaterialInstance::SolveMaterialInstance(SolveMaterialData* data, agx::Index index) : agxData::EntityInstance(data, index) {}
    AGX_FORCE_INLINE SolveMaterialInstance::SolveMaterialInstance(agxData::EntityStorage* storage, agx::Index index) : agxData::EntityInstance(storage, index) {}
    AGX_FORCE_INLINE SolveMaterialInstance::SolveMaterialInstance(const agxData::EntityInstance& other) : agxData::EntityInstance(other)
    {
      agxAssertN(!other || other.isInstanceOf(SolveMaterialModel::instance()),
        "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
        EntityInstance::getModel()->fullPath().c_str(), SolveMaterialModel::instance()->fullPath().c_str());
    }

    AGX_FORCE_INLINE SolveMaterialInstance::SolveMaterialInstance(const agxData::EntityPtr& ptr) : agxData::EntityInstance(ptr)
    {
      agxAssertN(!ptr || ptr.isInstanceOf(SolveMaterialModel::instance()),
        "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
        EntityInstance::getModel()->fullPath().c_str(), SolveMaterialModel::instance()->fullPath().c_str());
    }


    AGX_FORCE_INLINE SolveMaterialData* SolveMaterialInstance::getData() { return static_cast<SolveMaterialData* >(agxData::EntityInstance::getData()); }
    AGX_FORCE_INLINE const SolveMaterialData* SolveMaterialInstance::getData() const { return static_cast<const SolveMaterialData* >(agxData::EntityInstance::getData()); }

    AGX_FORCE_INLINE agx::Real& SolveMaterialInstance::adhesiveImpulse() { verifyIndex(); return getData()->adhesiveImpulse[getIndex()]; }
    AGX_FORCE_INLINE agx::Real const& SolveMaterialInstance::adhesiveImpulse() const { verifyIndex(); return getData()->adhesiveImpulse[getIndex()]; }

    AGX_FORCE_INLINE agx::Vec2& SolveMaterialInstance::friction() { verifyIndex(); return getData()->friction[getIndex()]; }
    AGX_FORCE_INLINE agx::Vec2 const& SolveMaterialInstance::friction() const { verifyIndex(); return getData()->friction[getIndex()]; }

    //-----------------------------------------------------------------------------------------------------
    AGX_FORCE_INLINE SolveMaterialSemantics::SolveMaterialSemantics() {}
    //-----------------------------------------------------------------------------------------------------
    DOXYGEN_END_INTERNAL_BLOCK()
  }
}

AGX_TYPE_BINDING(agx::Physics::SolveMaterialPtr, "Physics.SolveMaterialPtr")
AGX_TYPE_BINDING(agx::Physics::SolveMaterialInstance, "Physics.SolveMaterialInstance")

#ifdef _MSC_VER
# pragma warning(pop)
#endif

#undef AGX_ENTITY_WRAPPER
#undef AGX_ENTITY_NAMESPACE
#endif

