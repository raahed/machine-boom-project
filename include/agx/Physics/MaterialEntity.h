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

#ifndef GENERATED_AGX_PHYSICS_MATERIAL_H_PLUGIN
#define GENERATED_AGX_PHYSICS_MATERIAL_H_PLUGIN

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
#include <agx/Name.h>
#include <agx/Physics/BulkMaterialEntity.h>
#include <agx/Physics/SurfaceMaterialEntity.h>
#include <agx/Physics/WireMaterialEntity.h>
namespace agx { class Material; }

namespace agx { namespace Physics { class BulkMaterialPtr; }}
namespace agx { namespace Physics { class SurfaceMaterialPtr; }}
namespace agx { namespace Physics { class WireMaterialPtr; }}

namespace agx
{
  namespace Physics
  {

    class MaterialModel;
    class MaterialData;
    class MaterialPtr;
    class MaterialInstance;
    class MaterialSemantics;


    AGX_DECLARE_POINTER_TYPES(MaterialModel);

    /** 
    Abstract description of the data attributes for the Physics.Material entity.
    */ 
    class AGXPHYSICS_EXPORT MaterialModel : public agx::ReferencedModel
    {
    public:
      typedef MaterialPtr PtrT;

      MaterialModel(const agx::String& name = "Material");

      /// \return The entity model singleton.
      static MaterialModel* instance();

      /// Create and return a pointer to a new instance in the default storage for this entity model.
      static MaterialPtr createInstance();

      /// \return The default storage for this entity model.
      static agxData::EntityStorage* defaultStorage();

      /// This is part of internal cleanup and should not be called by users
      virtual void shutdownCleanup() override;



      /* Attributes */
      static agxData::PointerAttributeT< agx::Material*>* modelAttribute;
      static agxData::ScalarAttributeT< agx::Name >* nameAttribute;
      static agxData::ScalarAttributeT< agx::Name >* materialLibraryNameAttribute;
      static agxData::ScalarAttributeT< agx::Physics::BulkMaterialPtr >* bulkMaterialAttribute;
      static agxData::ScalarAttributeT< agx::Physics::SurfaceMaterialPtr >* surfaceMaterialAttribute;
      static agxData::ScalarAttributeT< agx::Physics::WireMaterialPtr >* wireMaterialAttribute;

    protected:
      virtual ~MaterialModel();
      virtual agxData::EntityData* createData(agxData::EntityStorage* storage) override;
      virtual void configure(agx::TiXmlElement* eEntity) override;
      virtual void initAttributeAccessors() override;
      virtual void construct(agxData::EntityPtr instance) override;
      void construct(agx::Physics::MaterialPtr material);
    };


    DOXYGEN_START_INTERNAL_BLOCK()
    #ifndef AGX_PHYSICS_MATERIAL_DATA_SET_OVERRIDE
    #define AGX_PHYSICS_MATERIAL_DATA_SET
    class AGXPHYSICS_EXPORT MaterialData : public agx::ReferencedData
    {
    public:
      MaterialInstance operator[] (size_t index);

    public:
      agxData::Array< MaterialPtr >& instance;
      agxData::Array< agx::Material* > model;
      agxData::Array< agx::Name > name;
      agxData::Array< agx::Name > materialLibraryName;
      agxData::Array< agx::Physics::BulkMaterialPtr > bulkMaterial;
      agxData::Array< agx::Physics::SurfaceMaterialPtr > surfaceMaterial;
      agxData::Array< agx::Physics::WireMaterialPtr > wireMaterial;

    public:
      typedef agx::Material* modelType;
      typedef agx::Name nameType;
      typedef agx::Name materialLibraryNameType;
      typedef agx::Physics::BulkMaterialPtr bulkMaterialType;
      typedef agx::Physics::SurfaceMaterialPtr surfaceMaterialType;
      typedef agx::Physics::WireMaterialPtr wireMaterialType;

    public:
      MaterialData(agxData::EntityStorage* storage);
      MaterialData();

    protected:
      virtual ~MaterialData() {}
      virtual void setNumElements(agx::Index numElements) override;

    private:
      MaterialData& operator= (const MaterialData&) { return *this; }

    };
    #endif
    DOXYGEN_END_INTERNAL_BLOCK()


    DOXYGEN_START_INTERNAL_BLOCK()
    class AGXPHYSICS_EXPORT MaterialSemantics : public agx::ReferencedSemantics
    {
    public:

      // Automatic getters
      agx::Material* const& getModel() const;
      agx::Name const& getName() const;
      agx::Name const& getMaterialLibraryName() const;
      agx::Physics::BulkMaterialPtr const& getBulkMaterial() const;
      agx::Physics::SurfaceMaterialPtr const& getSurfaceMaterial() const;
      agx::Physics::WireMaterialPtr const& getWireMaterial() const;

      // Semantics defined by explicit kernels

      // Automatic setters
      void setModel(agx::Material* const& value);
      void setName(agx::Name const& value);
      void setMaterialLibraryName(agx::Name const& value);
      void setBulkMaterial(agx::Physics::BulkMaterialPtr const& value);
      void setSurfaceMaterial(agx::Physics::SurfaceMaterialPtr const& value);
      void setWireMaterial(agx::Physics::WireMaterialPtr const& value);


    protected:
      friend class MaterialPtr;
      friend class MaterialInstance;
      MaterialSemantics();
    };
    DOXYGEN_END_INTERNAL_BLOCK()


    /**
    Pointer to a entity instance of type Physics.Material
    */
    class CALLABLE MaterialPtr : public agx::ReferencedPtr
    {
    public:
      typedef MaterialModel ModelType;
      typedef MaterialData DataType;
      typedef MaterialInstance InstanceType;

    public:
      AGXPHYSICS_EXPORT MaterialPtr();
      AGXPHYSICS_EXPORT MaterialPtr(agxData::EntityStorage* storage, agx::Index id);
      AGXPHYSICS_EXPORT MaterialPtr(const agxData::EntityPtr& ptr);
      AGXPHYSICS_EXPORT MaterialPtr(const agxData::EntityInstance& instance);
      AGXPHYSICS_EXPORT MaterialPtr& operator= (const agxData::EntityPtr& ptr);
      AGXPHYSICS_EXPORT MaterialPtr& operator= (const agxData::EntityInstance& instance);
      AGXPHYSICS_EXPORT MaterialInstance instance();
      AGXPHYSICS_EXPORT const MaterialInstance instance() const;

      AGXPHYSICS_EXPORT MaterialSemantics* operator->();
      AGXPHYSICS_EXPORT const MaterialSemantics* operator->() const;

      MaterialData* getData();
      const MaterialData* getData() const;


      /// \return reference to the model attribute
      AGXPHYSICS_EXPORT agx::Material*& model();
      /// \return const reference to the model attribute
      AGXPHYSICS_EXPORT agx::Material* const& model() const;

      /// \return reference to the name attribute
      AGXPHYSICS_EXPORT agx::Name& name();
      /// \return const reference to the name attribute
      AGXPHYSICS_EXPORT agx::Name const& name() const;

      /// \return reference to the materialLibraryName attribute
      AGXPHYSICS_EXPORT agx::Name& materialLibraryName();
      /// \return const reference to the materialLibraryName attribute
      AGXPHYSICS_EXPORT agx::Name const& materialLibraryName() const;

      /// \return reference to the bulkMaterial attribute
      AGXPHYSICS_EXPORT agx::Physics::BulkMaterialPtr& bulkMaterial();
      /// \return const reference to the bulkMaterial attribute
      AGXPHYSICS_EXPORT agx::Physics::BulkMaterialPtr const& bulkMaterial() const;

      /// \return reference to the surfaceMaterial attribute
      AGXPHYSICS_EXPORT agx::Physics::SurfaceMaterialPtr& surfaceMaterial();
      /// \return const reference to the surfaceMaterial attribute
      AGXPHYSICS_EXPORT agx::Physics::SurfaceMaterialPtr const& surfaceMaterial() const;

      /// \return reference to the wireMaterial attribute
      AGXPHYSICS_EXPORT agx::Physics::WireMaterialPtr& wireMaterial();
      /// \return const reference to the wireMaterial attribute
      AGXPHYSICS_EXPORT agx::Physics::WireMaterialPtr const& wireMaterial() const;

      DOXYGEN_START_INTERNAL_BLOCK()
      // Only used by tolua!
      #ifdef AGX_GENERATED_TOLUA_WRAPPER
      inline agx::Material* const& getModel() const { return (*this)->getModel(); }
      inline agx::Name const& getName() const { return (*this)->getName(); }
      inline agx::Name const& getMaterialLibraryName() const { return (*this)->getMaterialLibraryName(); }
      inline agx::Physics::BulkMaterialPtr const& getBulkMaterial() const { return (*this)->getBulkMaterial(); }
      inline agx::Physics::SurfaceMaterialPtr const& getSurfaceMaterial() const { return (*this)->getSurfaceMaterial(); }
      inline agx::Physics::WireMaterialPtr const& getWireMaterial() const { return (*this)->getWireMaterial(); }


      inline void setModel(agx::Material* const& value) { (*this)->setModel(value); }
      inline void setName(agx::Name const& value) { (*this)->setName(value); }
      inline void setMaterialLibraryName(agx::Name const& value) { (*this)->setMaterialLibraryName(value); }
      inline void setBulkMaterial(agx::Physics::BulkMaterialPtr const& value) { (*this)->setBulkMaterial(value); }
      inline void setSurfaceMaterial(agx::Physics::SurfaceMaterialPtr const& value) { (*this)->setSurfaceMaterial(value); }
      inline void setWireMaterial(agx::Physics::WireMaterialPtr const& value) { (*this)->setWireMaterial(value); }
      #endif
      DOXYGEN_END_INTERNAL_BLOCK()

    };

    // Entity is Referenced
    typedef agxData::EntityRef< MaterialPtr > MaterialRef;


    DOXYGEN_START_INTERNAL_BLOCK()
    class AGXPHYSICS_EXPORT MaterialInstance : public agx::ReferencedInstance
    {
    public:
      MaterialInstance();
      MaterialInstance(MaterialData* data, agx::Index index);
      MaterialInstance(agxData::EntityStorage *storage, agx::Index index);
      MaterialInstance(const agxData::EntityInstance& other);
      MaterialInstance(const agxData::EntityPtr& ptr);

      MaterialData* getData();
      const MaterialData* getData() const;

    public:
      /// \return reference to the model attribute
      agx::Material*& model();
      /// \return const reference to the model attribute
      agx::Material* const& model() const;

      /// \return reference to the name attribute
      agx::Name& name();
      /// \return const reference to the name attribute
      agx::Name const& name() const;

      /// \return reference to the materialLibraryName attribute
      agx::Name& materialLibraryName();
      /// \return const reference to the materialLibraryName attribute
      agx::Name const& materialLibraryName() const;

      /// \return reference to the bulkMaterial attribute
      agx::Physics::BulkMaterialPtr& bulkMaterial();
      /// \return const reference to the bulkMaterial attribute
      agx::Physics::BulkMaterialPtr const& bulkMaterial() const;

      /// \return reference to the surfaceMaterial attribute
      agx::Physics::SurfaceMaterialPtr& surfaceMaterial();
      /// \return const reference to the surfaceMaterial attribute
      agx::Physics::SurfaceMaterialPtr const& surfaceMaterial() const;

      /// \return reference to the wireMaterial attribute
      agx::Physics::WireMaterialPtr& wireMaterial();
      /// \return const reference to the wireMaterial attribute
      agx::Physics::WireMaterialPtr const& wireMaterial() const;

    };
    DOXYGEN_END_INTERNAL_BLOCK()



    typedef agx::VectorPOD<MaterialPtr> MaterialPtrVector;
    typedef agxData::Array<MaterialPtr> MaterialPtrArray;



    DOXYGEN_START_INTERNAL_BLOCK()
    /* Implementation */
    //-----------------------------------------------------------------------------------------------------
    //-----------------------------------------------------------------------------------------------------
    inline MaterialInstance agx::Physics::MaterialData::operator[] (size_t index) { return MaterialInstance(this, (agx::Index)index); }
    //-----------------------------------------------------------------------------------------------------
    AGX_FORCE_INLINE MaterialPtr::MaterialPtr() {}
    AGX_FORCE_INLINE MaterialPtr::MaterialPtr(agxData::EntityStorage* storage, agx::Index id) : agx::ReferencedPtr(storage, id) {}
    AGX_FORCE_INLINE MaterialPtr::MaterialPtr(const agxData::EntityPtr& ptr) : agx::ReferencedPtr(ptr)
    {
      agxAssertN(!ptr || ptr.isInstanceOf(MaterialModel::instance()),
        "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
        EntityPtr::getModel()->fullPath().c_str(), MaterialModel::instance()->fullPath().c_str());
    }

    AGX_FORCE_INLINE MaterialPtr::MaterialPtr(const agxData::EntityInstance& instance) : agx::ReferencedPtr(instance)
    {
      agxAssertN(!instance || instance.isInstanceOf(MaterialModel::instance()),
        "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
        EntityPtr::getModel()->fullPath().c_str(), MaterialModel::instance()->fullPath().c_str());
    }

    AGX_FORCE_INLINE MaterialPtr& MaterialPtr::operator= (const agxData::EntityPtr& ptr)
    {
      agxData::EntityPtr::operator= (ptr);
      agxAssertN(!ptr || ptr.isInstanceOf(MaterialModel::instance()),
        "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
        EntityPtr::getModel()->fullPath().c_str(), MaterialModel::instance()->fullPath().c_str());
      return *this;
    }

    AGX_FORCE_INLINE MaterialPtr& MaterialPtr::operator= (const agxData::EntityInstance& instance)
    {
      agxData::EntityPtr::operator= (instance);
      agxAssertN(!instance || instance.isInstanceOf(MaterialModel::instance()),
        "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
        EntityPtr::getModel()->fullPath().c_str(), MaterialModel::instance()->fullPath().c_str());
      return *this;
    }

    inline MaterialInstance MaterialPtr::instance() { return agxData::EntityPtr::instance(); }
    inline const MaterialInstance MaterialPtr::instance() const { return agxData::EntityPtr::instance(); }
    AGX_FORCE_INLINE MaterialSemantics* MaterialPtr::operator->() { return (MaterialSemantics* )this; }
    AGX_FORCE_INLINE const MaterialSemantics* MaterialPtr::operator->() const { return (const MaterialSemantics* )this; }
    AGX_FORCE_INLINE MaterialData* MaterialPtr::getData() { return static_cast<MaterialData* >(agxData::EntityPtr::getData()); }
    AGX_FORCE_INLINE const MaterialData* MaterialPtr::getData() const { return static_cast<const MaterialData* >(agxData::EntityPtr::getData()); }

    AGX_FORCE_INLINE agx::Material*& MaterialPtr::model() { verifyIndex(); return getData()->model[calculateIndex()]; }
    AGX_FORCE_INLINE agx::Material* const& MaterialPtr::model() const { verifyIndex(); return getData()->model[calculateIndex()]; }

    AGX_FORCE_INLINE agx::Name& MaterialPtr::name() { verifyIndex(); return getData()->name[calculateIndex()]; }
    AGX_FORCE_INLINE agx::Name const& MaterialPtr::name() const { verifyIndex(); return getData()->name[calculateIndex()]; }

    AGX_FORCE_INLINE agx::Name& MaterialPtr::materialLibraryName() { verifyIndex(); return getData()->materialLibraryName[calculateIndex()]; }
    AGX_FORCE_INLINE agx::Name const& MaterialPtr::materialLibraryName() const { verifyIndex(); return getData()->materialLibraryName[calculateIndex()]; }

    AGX_FORCE_INLINE agx::Physics::BulkMaterialPtr& MaterialPtr::bulkMaterial() { verifyIndex(); return getData()->bulkMaterial[calculateIndex()]; }
    AGX_FORCE_INLINE agx::Physics::BulkMaterialPtr const& MaterialPtr::bulkMaterial() const { verifyIndex(); return getData()->bulkMaterial[calculateIndex()]; }

    AGX_FORCE_INLINE agx::Physics::SurfaceMaterialPtr& MaterialPtr::surfaceMaterial() { verifyIndex(); return getData()->surfaceMaterial[calculateIndex()]; }
    AGX_FORCE_INLINE agx::Physics::SurfaceMaterialPtr const& MaterialPtr::surfaceMaterial() const { verifyIndex(); return getData()->surfaceMaterial[calculateIndex()]; }

    AGX_FORCE_INLINE agx::Physics::WireMaterialPtr& MaterialPtr::wireMaterial() { verifyIndex(); return getData()->wireMaterial[calculateIndex()]; }
    AGX_FORCE_INLINE agx::Physics::WireMaterialPtr const& MaterialPtr::wireMaterial() const { verifyIndex(); return getData()->wireMaterial[calculateIndex()]; }

    //-----------------------------------------------------------------------------------------------------
    AGX_FORCE_INLINE MaterialInstance::MaterialInstance() {}
    AGX_FORCE_INLINE MaterialInstance::MaterialInstance(MaterialData* data, agx::Index index) : agx::ReferencedInstance(data, index) {}
    AGX_FORCE_INLINE MaterialInstance::MaterialInstance(agxData::EntityStorage* storage, agx::Index index) : agx::ReferencedInstance(storage, index) {}
    AGX_FORCE_INLINE MaterialInstance::MaterialInstance(const agxData::EntityInstance& other) : agx::ReferencedInstance(other)
    {
      agxAssertN(!other || other.isInstanceOf(MaterialModel::instance()),
        "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
        EntityInstance::getModel()->fullPath().c_str(), MaterialModel::instance()->fullPath().c_str());
    }

    AGX_FORCE_INLINE MaterialInstance::MaterialInstance(const agxData::EntityPtr& ptr) : agx::ReferencedInstance(ptr)
    {
      agxAssertN(!ptr || ptr.isInstanceOf(MaterialModel::instance()),
        "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
        EntityInstance::getModel()->fullPath().c_str(), MaterialModel::instance()->fullPath().c_str());
    }


    AGX_FORCE_INLINE MaterialData* MaterialInstance::getData() { return static_cast<MaterialData* >(agxData::EntityInstance::getData()); }
    AGX_FORCE_INLINE const MaterialData* MaterialInstance::getData() const { return static_cast<const MaterialData* >(agxData::EntityInstance::getData()); }

    AGX_FORCE_INLINE agx::Material*& MaterialInstance::model() { verifyIndex(); return getData()->model[getIndex()]; }
    AGX_FORCE_INLINE agx::Material* const& MaterialInstance::model() const { verifyIndex(); return getData()->model[getIndex()]; }

    AGX_FORCE_INLINE agx::Name& MaterialInstance::name() { verifyIndex(); return getData()->name[getIndex()]; }
    AGX_FORCE_INLINE agx::Name const& MaterialInstance::name() const { verifyIndex(); return getData()->name[getIndex()]; }

    AGX_FORCE_INLINE agx::Name& MaterialInstance::materialLibraryName() { verifyIndex(); return getData()->materialLibraryName[getIndex()]; }
    AGX_FORCE_INLINE agx::Name const& MaterialInstance::materialLibraryName() const { verifyIndex(); return getData()->materialLibraryName[getIndex()]; }

    AGX_FORCE_INLINE agx::Physics::BulkMaterialPtr& MaterialInstance::bulkMaterial() { verifyIndex(); return getData()->bulkMaterial[getIndex()]; }
    AGX_FORCE_INLINE agx::Physics::BulkMaterialPtr const& MaterialInstance::bulkMaterial() const { verifyIndex(); return getData()->bulkMaterial[getIndex()]; }

    AGX_FORCE_INLINE agx::Physics::SurfaceMaterialPtr& MaterialInstance::surfaceMaterial() { verifyIndex(); return getData()->surfaceMaterial[getIndex()]; }
    AGX_FORCE_INLINE agx::Physics::SurfaceMaterialPtr const& MaterialInstance::surfaceMaterial() const { verifyIndex(); return getData()->surfaceMaterial[getIndex()]; }

    AGX_FORCE_INLINE agx::Physics::WireMaterialPtr& MaterialInstance::wireMaterial() { verifyIndex(); return getData()->wireMaterial[getIndex()]; }
    AGX_FORCE_INLINE agx::Physics::WireMaterialPtr const& MaterialInstance::wireMaterial() const { verifyIndex(); return getData()->wireMaterial[getIndex()]; }

    //-----------------------------------------------------------------------------------------------------
    AGX_FORCE_INLINE MaterialSemantics::MaterialSemantics() {}
    //-----------------------------------------------------------------------------------------------------
    DOXYGEN_END_INTERNAL_BLOCK()
  }
}

AGX_TYPE_BINDING(agx::Physics::MaterialPtr, "Physics.MaterialPtr")
AGX_TYPE_BINDING(agx::Physics::MaterialInstance, "Physics.MaterialInstance")

#ifdef _MSC_VER
# pragma warning(pop)
#endif

#undef AGX_ENTITY_WRAPPER
#undef AGX_ENTITY_NAMESPACE
#endif

