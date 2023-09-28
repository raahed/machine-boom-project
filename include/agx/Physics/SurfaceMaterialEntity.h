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

#ifndef GENERATED_AGX_PHYSICS_SURFACEMATERIAL_H_PLUGIN
#define GENERATED_AGX_PHYSICS_SURFACEMATERIAL_H_PLUGIN

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
#include <agx/Real.h>
#include <agx/Integer.h>


namespace agx
{
  namespace Physics
  {

    class SurfaceMaterialModel;
    class SurfaceMaterialData;
    class SurfaceMaterialPtr;
    class SurfaceMaterialInstance;
    class SurfaceMaterialSemantics;


    AGX_DECLARE_POINTER_TYPES(SurfaceMaterialModel);

    /** 
    Abstract description of the data attributes for the Physics.SurfaceMaterial entity.
    */ 
    class AGXPHYSICS_EXPORT SurfaceMaterialModel : public agx::ReferencedModel
    {
    public:
      typedef SurfaceMaterialPtr PtrT;

      SurfaceMaterialModel(const agx::String& name = "SurfaceMaterial");

      /// \return The entity model singleton.
      static SurfaceMaterialModel* instance();

      /// Create and return a pointer to a new instance in the default storage for this entity model.
      static SurfaceMaterialPtr createInstance();

      /// \return The default storage for this entity model.
      static agxData::EntityStorage* defaultStorage();

      /// This is part of internal cleanup and should not be called by users
      virtual void shutdownCleanup() override;



      /* Attributes */
      static agxData::ScalarAttributeT< agx::Real >* roughnessAttribute;
      static agxData::ScalarAttributeT< agx::Real >* adhesionAttribute;
      static agxData::ScalarAttributeT< agx::Real >* adhesiveOverlapAttribute;
      static agxData::ScalarAttributeT< agx::Real >* viscosityAttribute;
      static agxData::ScalarAttributeT< agx::Bool >* frictionEnabledAttribute;
      static agxData::ScalarAttributeT< agx::Bool >* dirtyAttribute;

    protected:
      virtual ~SurfaceMaterialModel();
      virtual agxData::EntityData* createData(agxData::EntityStorage* storage) override;
      virtual void configure(agx::TiXmlElement* eEntity) override;
      virtual void initAttributeAccessors() override;
      virtual void construct(agxData::EntityPtr instance) override;
      void construct(agx::Physics::SurfaceMaterialPtr surfaceMaterial);
    };


    DOXYGEN_START_INTERNAL_BLOCK()
    #ifndef AGX_PHYSICS_SURFACEMATERIAL_DATA_SET_OVERRIDE
    #define AGX_PHYSICS_SURFACEMATERIAL_DATA_SET
    class AGXPHYSICS_EXPORT SurfaceMaterialData : public agx::ReferencedData
    {
    public:
      SurfaceMaterialInstance operator[] (size_t index);

    public:
      agxData::Array< SurfaceMaterialPtr >& instance;
      agxData::Array< agx::Real > roughness;
      agxData::Array< agx::Real > adhesion;
      agxData::Array< agx::Real > adhesiveOverlap;
      agxData::Array< agx::Real > viscosity;
      agxData::Array< agx::Bool > frictionEnabled;
      agxData::Array< agx::Bool > dirty;

    public:
      typedef agx::Real roughnessType;
      typedef agx::Real adhesionType;
      typedef agx::Real adhesiveOverlapType;
      typedef agx::Real viscosityType;
      typedef agx::Bool frictionEnabledType;
      typedef agx::Bool dirtyType;

    public:
      SurfaceMaterialData(agxData::EntityStorage* storage);
      SurfaceMaterialData();

    protected:
      virtual ~SurfaceMaterialData() {}
      virtual void setNumElements(agx::Index numElements) override;

    private:
      SurfaceMaterialData& operator= (const SurfaceMaterialData&) { return *this; }

    };
    #endif
    DOXYGEN_END_INTERNAL_BLOCK()


    DOXYGEN_START_INTERNAL_BLOCK()
    class AGXPHYSICS_EXPORT SurfaceMaterialSemantics : public agx::ReferencedSemantics
    {
    public:

      // Automatic getters
      agx::Real const& getRoughness() const;
      agx::Real const& getAdhesion() const;
      agx::Real const& getAdhesiveOverlap() const;
      agx::Real const& getViscosity() const;
      agx::Bool const& getFrictionEnabled() const;
      agx::Bool const& getDirty() const;

      // Semantics defined by explicit kernels

      // Automatic setters
      void setRoughness(agx::Real const& value);
      void setAdhesion(agx::Real const& value);
      void setAdhesiveOverlap(agx::Real const& value);
      void setViscosity(agx::Real const& value);
      void setFrictionEnabled(agx::Bool const& value);
      void setDirty(agx::Bool const& value);


    protected:
      friend class SurfaceMaterialPtr;
      friend class SurfaceMaterialInstance;
      SurfaceMaterialSemantics();
    };
    DOXYGEN_END_INTERNAL_BLOCK()


    /**
    Pointer to a entity instance of type Physics.SurfaceMaterial
    */
    class CALLABLE SurfaceMaterialPtr : public agx::ReferencedPtr
    {
    public:
      typedef SurfaceMaterialModel ModelType;
      typedef SurfaceMaterialData DataType;
      typedef SurfaceMaterialInstance InstanceType;

    public:
      AGXPHYSICS_EXPORT SurfaceMaterialPtr();
      AGXPHYSICS_EXPORT SurfaceMaterialPtr(agxData::EntityStorage* storage, agx::Index id);
      AGXPHYSICS_EXPORT SurfaceMaterialPtr(const agxData::EntityPtr& ptr);
      AGXPHYSICS_EXPORT SurfaceMaterialPtr(const agxData::EntityInstance& instance);
      AGXPHYSICS_EXPORT SurfaceMaterialPtr& operator= (const agxData::EntityPtr& ptr);
      AGXPHYSICS_EXPORT SurfaceMaterialPtr& operator= (const agxData::EntityInstance& instance);
      AGXPHYSICS_EXPORT SurfaceMaterialInstance instance();
      AGXPHYSICS_EXPORT const SurfaceMaterialInstance instance() const;

      AGXPHYSICS_EXPORT SurfaceMaterialSemantics* operator->();
      AGXPHYSICS_EXPORT const SurfaceMaterialSemantics* operator->() const;

      SurfaceMaterialData* getData();
      const SurfaceMaterialData* getData() const;


      /// \return reference to the roughness attribute
      AGXPHYSICS_EXPORT agx::Real& roughness();
      /// \return const reference to the roughness attribute
      AGXPHYSICS_EXPORT agx::Real const& roughness() const;

      /// \return reference to the adhesion attribute
      AGXPHYSICS_EXPORT agx::Real& adhesion();
      /// \return const reference to the adhesion attribute
      AGXPHYSICS_EXPORT agx::Real const& adhesion() const;

      /// \return reference to the adhesiveOverlap attribute
      AGXPHYSICS_EXPORT agx::Real& adhesiveOverlap();
      /// \return const reference to the adhesiveOverlap attribute
      AGXPHYSICS_EXPORT agx::Real const& adhesiveOverlap() const;

      /// \return reference to the viscosity attribute
      AGXPHYSICS_EXPORT agx::Real& viscosity();
      /// \return const reference to the viscosity attribute
      AGXPHYSICS_EXPORT agx::Real const& viscosity() const;

      /// \return reference to the frictionEnabled attribute
      AGXPHYSICS_EXPORT agx::Bool& frictionEnabled();
      /// \return const reference to the frictionEnabled attribute
      AGXPHYSICS_EXPORT agx::Bool const& frictionEnabled() const;

      /// \return reference to the dirty attribute
      AGXPHYSICS_EXPORT agx::Bool& dirty();
      /// \return const reference to the dirty attribute
      AGXPHYSICS_EXPORT agx::Bool const& dirty() const;

      DOXYGEN_START_INTERNAL_BLOCK()
      // Only used by tolua!
      #ifdef AGX_GENERATED_TOLUA_WRAPPER
      inline agx::Real const& getRoughness() const { return (*this)->getRoughness(); }
      inline agx::Real const& getAdhesion() const { return (*this)->getAdhesion(); }
      inline agx::Real const& getAdhesiveOverlap() const { return (*this)->getAdhesiveOverlap(); }
      inline agx::Real const& getViscosity() const { return (*this)->getViscosity(); }
      inline agx::Bool const& getFrictionEnabled() const { return (*this)->getFrictionEnabled(); }
      inline agx::Bool const& getDirty() const { return (*this)->getDirty(); }


      inline void setRoughness(agx::Real const& value) { (*this)->setRoughness(value); }
      inline void setAdhesion(agx::Real const& value) { (*this)->setAdhesion(value); }
      inline void setAdhesiveOverlap(agx::Real const& value) { (*this)->setAdhesiveOverlap(value); }
      inline void setViscosity(agx::Real const& value) { (*this)->setViscosity(value); }
      inline void setFrictionEnabled(agx::Bool const& value) { (*this)->setFrictionEnabled(value); }
      inline void setDirty(agx::Bool const& value) { (*this)->setDirty(value); }
      #endif
      DOXYGEN_END_INTERNAL_BLOCK()

    };

    // Entity is Referenced
    typedef agxData::EntityRef< SurfaceMaterialPtr > SurfaceMaterialRef;


    DOXYGEN_START_INTERNAL_BLOCK()
    class AGXPHYSICS_EXPORT SurfaceMaterialInstance : public agx::ReferencedInstance
    {
    public:
      SurfaceMaterialInstance();
      SurfaceMaterialInstance(SurfaceMaterialData* data, agx::Index index);
      SurfaceMaterialInstance(agxData::EntityStorage *storage, agx::Index index);
      SurfaceMaterialInstance(const agxData::EntityInstance& other);
      SurfaceMaterialInstance(const agxData::EntityPtr& ptr);

      SurfaceMaterialData* getData();
      const SurfaceMaterialData* getData() const;

    public:
      /// \return reference to the roughness attribute
      agx::Real& roughness();
      /// \return const reference to the roughness attribute
      agx::Real const& roughness() const;

      /// \return reference to the adhesion attribute
      agx::Real& adhesion();
      /// \return const reference to the adhesion attribute
      agx::Real const& adhesion() const;

      /// \return reference to the adhesiveOverlap attribute
      agx::Real& adhesiveOverlap();
      /// \return const reference to the adhesiveOverlap attribute
      agx::Real const& adhesiveOverlap() const;

      /// \return reference to the viscosity attribute
      agx::Real& viscosity();
      /// \return const reference to the viscosity attribute
      agx::Real const& viscosity() const;

      /// \return reference to the frictionEnabled attribute
      agx::Bool& frictionEnabled();
      /// \return const reference to the frictionEnabled attribute
      agx::Bool const& frictionEnabled() const;

      /// \return reference to the dirty attribute
      agx::Bool& dirty();
      /// \return const reference to the dirty attribute
      agx::Bool const& dirty() const;

    };
    DOXYGEN_END_INTERNAL_BLOCK()



    typedef agx::VectorPOD<SurfaceMaterialPtr> SurfaceMaterialPtrVector;
    typedef agxData::Array<SurfaceMaterialPtr> SurfaceMaterialPtrArray;



    DOXYGEN_START_INTERNAL_BLOCK()
    /* Implementation */
    //-----------------------------------------------------------------------------------------------------
    //-----------------------------------------------------------------------------------------------------
    inline SurfaceMaterialInstance agx::Physics::SurfaceMaterialData::operator[] (size_t index) { return SurfaceMaterialInstance(this, (agx::Index)index); }
    //-----------------------------------------------------------------------------------------------------
    AGX_FORCE_INLINE SurfaceMaterialPtr::SurfaceMaterialPtr() {}
    AGX_FORCE_INLINE SurfaceMaterialPtr::SurfaceMaterialPtr(agxData::EntityStorage* storage, agx::Index id) : agx::ReferencedPtr(storage, id) {}
    AGX_FORCE_INLINE SurfaceMaterialPtr::SurfaceMaterialPtr(const agxData::EntityPtr& ptr) : agx::ReferencedPtr(ptr)
    {
      agxAssertN(!ptr || ptr.isInstanceOf(SurfaceMaterialModel::instance()),
        "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
        EntityPtr::getModel()->fullPath().c_str(), SurfaceMaterialModel::instance()->fullPath().c_str());
    }

    AGX_FORCE_INLINE SurfaceMaterialPtr::SurfaceMaterialPtr(const agxData::EntityInstance& instance) : agx::ReferencedPtr(instance)
    {
      agxAssertN(!instance || instance.isInstanceOf(SurfaceMaterialModel::instance()),
        "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
        EntityPtr::getModel()->fullPath().c_str(), SurfaceMaterialModel::instance()->fullPath().c_str());
    }

    AGX_FORCE_INLINE SurfaceMaterialPtr& SurfaceMaterialPtr::operator= (const agxData::EntityPtr& ptr)
    {
      agxData::EntityPtr::operator= (ptr);
      agxAssertN(!ptr || ptr.isInstanceOf(SurfaceMaterialModel::instance()),
        "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
        EntityPtr::getModel()->fullPath().c_str(), SurfaceMaterialModel::instance()->fullPath().c_str());
      return *this;
    }

    AGX_FORCE_INLINE SurfaceMaterialPtr& SurfaceMaterialPtr::operator= (const agxData::EntityInstance& instance)
    {
      agxData::EntityPtr::operator= (instance);
      agxAssertN(!instance || instance.isInstanceOf(SurfaceMaterialModel::instance()),
        "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
        EntityPtr::getModel()->fullPath().c_str(), SurfaceMaterialModel::instance()->fullPath().c_str());
      return *this;
    }

    inline SurfaceMaterialInstance SurfaceMaterialPtr::instance() { return agxData::EntityPtr::instance(); }
    inline const SurfaceMaterialInstance SurfaceMaterialPtr::instance() const { return agxData::EntityPtr::instance(); }
    AGX_FORCE_INLINE SurfaceMaterialSemantics* SurfaceMaterialPtr::operator->() { return (SurfaceMaterialSemantics* )this; }
    AGX_FORCE_INLINE const SurfaceMaterialSemantics* SurfaceMaterialPtr::operator->() const { return (const SurfaceMaterialSemantics* )this; }
    AGX_FORCE_INLINE SurfaceMaterialData* SurfaceMaterialPtr::getData() { return static_cast<SurfaceMaterialData* >(agxData::EntityPtr::getData()); }
    AGX_FORCE_INLINE const SurfaceMaterialData* SurfaceMaterialPtr::getData() const { return static_cast<const SurfaceMaterialData* >(agxData::EntityPtr::getData()); }

    AGX_FORCE_INLINE agx::Real& SurfaceMaterialPtr::roughness() { verifyIndex(); return getData()->roughness[calculateIndex()]; }
    AGX_FORCE_INLINE agx::Real const& SurfaceMaterialPtr::roughness() const { verifyIndex(); return getData()->roughness[calculateIndex()]; }

    AGX_FORCE_INLINE agx::Real& SurfaceMaterialPtr::adhesion() { verifyIndex(); return getData()->adhesion[calculateIndex()]; }
    AGX_FORCE_INLINE agx::Real const& SurfaceMaterialPtr::adhesion() const { verifyIndex(); return getData()->adhesion[calculateIndex()]; }

    AGX_FORCE_INLINE agx::Real& SurfaceMaterialPtr::adhesiveOverlap() { verifyIndex(); return getData()->adhesiveOverlap[calculateIndex()]; }
    AGX_FORCE_INLINE agx::Real const& SurfaceMaterialPtr::adhesiveOverlap() const { verifyIndex(); return getData()->adhesiveOverlap[calculateIndex()]; }

    AGX_FORCE_INLINE agx::Real& SurfaceMaterialPtr::viscosity() { verifyIndex(); return getData()->viscosity[calculateIndex()]; }
    AGX_FORCE_INLINE agx::Real const& SurfaceMaterialPtr::viscosity() const { verifyIndex(); return getData()->viscosity[calculateIndex()]; }

    AGX_FORCE_INLINE agx::Bool& SurfaceMaterialPtr::frictionEnabled() { verifyIndex(); return getData()->frictionEnabled[calculateIndex()]; }
    AGX_FORCE_INLINE agx::Bool const& SurfaceMaterialPtr::frictionEnabled() const { verifyIndex(); return getData()->frictionEnabled[calculateIndex()]; }

    AGX_FORCE_INLINE agx::Bool& SurfaceMaterialPtr::dirty() { verifyIndex(); return getData()->dirty[calculateIndex()]; }
    AGX_FORCE_INLINE agx::Bool const& SurfaceMaterialPtr::dirty() const { verifyIndex(); return getData()->dirty[calculateIndex()]; }

    //-----------------------------------------------------------------------------------------------------
    AGX_FORCE_INLINE SurfaceMaterialInstance::SurfaceMaterialInstance() {}
    AGX_FORCE_INLINE SurfaceMaterialInstance::SurfaceMaterialInstance(SurfaceMaterialData* data, agx::Index index) : agx::ReferencedInstance(data, index) {}
    AGX_FORCE_INLINE SurfaceMaterialInstance::SurfaceMaterialInstance(agxData::EntityStorage* storage, agx::Index index) : agx::ReferencedInstance(storage, index) {}
    AGX_FORCE_INLINE SurfaceMaterialInstance::SurfaceMaterialInstance(const agxData::EntityInstance& other) : agx::ReferencedInstance(other)
    {
      agxAssertN(!other || other.isInstanceOf(SurfaceMaterialModel::instance()),
        "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
        EntityInstance::getModel()->fullPath().c_str(), SurfaceMaterialModel::instance()->fullPath().c_str());
    }

    AGX_FORCE_INLINE SurfaceMaterialInstance::SurfaceMaterialInstance(const agxData::EntityPtr& ptr) : agx::ReferencedInstance(ptr)
    {
      agxAssertN(!ptr || ptr.isInstanceOf(SurfaceMaterialModel::instance()),
        "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
        EntityInstance::getModel()->fullPath().c_str(), SurfaceMaterialModel::instance()->fullPath().c_str());
    }


    AGX_FORCE_INLINE SurfaceMaterialData* SurfaceMaterialInstance::getData() { return static_cast<SurfaceMaterialData* >(agxData::EntityInstance::getData()); }
    AGX_FORCE_INLINE const SurfaceMaterialData* SurfaceMaterialInstance::getData() const { return static_cast<const SurfaceMaterialData* >(agxData::EntityInstance::getData()); }

    AGX_FORCE_INLINE agx::Real& SurfaceMaterialInstance::roughness() { verifyIndex(); return getData()->roughness[getIndex()]; }
    AGX_FORCE_INLINE agx::Real const& SurfaceMaterialInstance::roughness() const { verifyIndex(); return getData()->roughness[getIndex()]; }

    AGX_FORCE_INLINE agx::Real& SurfaceMaterialInstance::adhesion() { verifyIndex(); return getData()->adhesion[getIndex()]; }
    AGX_FORCE_INLINE agx::Real const& SurfaceMaterialInstance::adhesion() const { verifyIndex(); return getData()->adhesion[getIndex()]; }

    AGX_FORCE_INLINE agx::Real& SurfaceMaterialInstance::adhesiveOverlap() { verifyIndex(); return getData()->adhesiveOverlap[getIndex()]; }
    AGX_FORCE_INLINE agx::Real const& SurfaceMaterialInstance::adhesiveOverlap() const { verifyIndex(); return getData()->adhesiveOverlap[getIndex()]; }

    AGX_FORCE_INLINE agx::Real& SurfaceMaterialInstance::viscosity() { verifyIndex(); return getData()->viscosity[getIndex()]; }
    AGX_FORCE_INLINE agx::Real const& SurfaceMaterialInstance::viscosity() const { verifyIndex(); return getData()->viscosity[getIndex()]; }

    AGX_FORCE_INLINE agx::Bool& SurfaceMaterialInstance::frictionEnabled() { verifyIndex(); return getData()->frictionEnabled[getIndex()]; }
    AGX_FORCE_INLINE agx::Bool const& SurfaceMaterialInstance::frictionEnabled() const { verifyIndex(); return getData()->frictionEnabled[getIndex()]; }

    AGX_FORCE_INLINE agx::Bool& SurfaceMaterialInstance::dirty() { verifyIndex(); return getData()->dirty[getIndex()]; }
    AGX_FORCE_INLINE agx::Bool const& SurfaceMaterialInstance::dirty() const { verifyIndex(); return getData()->dirty[getIndex()]; }

    //-----------------------------------------------------------------------------------------------------
    AGX_FORCE_INLINE SurfaceMaterialSemantics::SurfaceMaterialSemantics() {}
    //-----------------------------------------------------------------------------------------------------
    DOXYGEN_END_INTERNAL_BLOCK()
  }
}

AGX_TYPE_BINDING(agx::Physics::SurfaceMaterialPtr, "Physics.SurfaceMaterialPtr")
AGX_TYPE_BINDING(agx::Physics::SurfaceMaterialInstance, "Physics.SurfaceMaterialInstance")

#ifdef _MSC_VER
# pragma warning(pop)
#endif

#undef AGX_ENTITY_WRAPPER
#undef AGX_ENTITY_NAMESPACE
#endif

