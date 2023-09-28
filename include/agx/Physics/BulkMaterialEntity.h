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

#ifndef GENERATED_AGX_PHYSICS_BULKMATERIAL_H_PLUGIN
#define GENERATED_AGX_PHYSICS_BULKMATERIAL_H_PLUGIN

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

    class BulkMaterialModel;
    class BulkMaterialData;
    class BulkMaterialPtr;
    class BulkMaterialInstance;
    class BulkMaterialSemantics;


    AGX_DECLARE_POINTER_TYPES(BulkMaterialModel);

    /** 
    Abstract description of the data attributes for the Physics.BulkMaterial entity.
    */ 
    class AGXPHYSICS_EXPORT BulkMaterialModel : public agx::ReferencedModel
    {
    public:
      typedef BulkMaterialPtr PtrT;

      BulkMaterialModel(const agx::String& name = "BulkMaterial");

      /// \return The entity model singleton.
      static BulkMaterialModel* instance();

      /// Create and return a pointer to a new instance in the default storage for this entity model.
      static BulkMaterialPtr createInstance();

      /// \return The default storage for this entity model.
      static agxData::EntityStorage* defaultStorage();

      /// This is part of internal cleanup and should not be called by users
      virtual void shutdownCleanup() override;



      /* Attributes */
      static agxData::ScalarAttributeT< agx::Real >* densityAttribute;
      static agxData::ScalarAttributeT< agx::Real >* youngsModulusAttribute;
      static agxData::ScalarAttributeT< agx::Real >* viscosityAttribute;
      static agxData::ScalarAttributeT< agx::Real >* dampingAttribute;
      static agxData::ScalarAttributeT< agx::Real >* minElasticRestLengthAttribute;
      static agxData::ScalarAttributeT< agx::Real >* maxElasticRestLengthAttribute;
      static agxData::ScalarAttributeT< agx::Bool >* dirtyAttribute;

    protected:
      virtual ~BulkMaterialModel();
      virtual agxData::EntityData* createData(agxData::EntityStorage* storage) override;
      virtual void configure(agx::TiXmlElement* eEntity) override;
      virtual void initAttributeAccessors() override;
      virtual void construct(agxData::EntityPtr instance) override;
      void construct(agx::Physics::BulkMaterialPtr bulkMaterial);
    };


    DOXYGEN_START_INTERNAL_BLOCK()
    #ifndef AGX_PHYSICS_BULKMATERIAL_DATA_SET_OVERRIDE
    #define AGX_PHYSICS_BULKMATERIAL_DATA_SET
    class AGXPHYSICS_EXPORT BulkMaterialData : public agx::ReferencedData
    {
    public:
      BulkMaterialInstance operator[] (size_t index);

    public:
      agxData::Array< BulkMaterialPtr >& instance;
      agxData::Array< agx::Real > density;
      agxData::Array< agx::Real > youngsModulus;
      agxData::Array< agx::Real > viscosity;
      agxData::Array< agx::Real > damping;
      agxData::Array< agx::Real > minElasticRestLength;
      agxData::Array< agx::Real > maxElasticRestLength;
      agxData::Array< agx::Bool > dirty;

    public:
      typedef agx::Real densityType;
      typedef agx::Real youngsModulusType;
      typedef agx::Real viscosityType;
      typedef agx::Real dampingType;
      typedef agx::Real minElasticRestLengthType;
      typedef agx::Real maxElasticRestLengthType;
      typedef agx::Bool dirtyType;

    public:
      BulkMaterialData(agxData::EntityStorage* storage);
      BulkMaterialData();

    protected:
      virtual ~BulkMaterialData() {}
      virtual void setNumElements(agx::Index numElements) override;

    private:
      BulkMaterialData& operator= (const BulkMaterialData&) { return *this; }

    };
    #endif
    DOXYGEN_END_INTERNAL_BLOCK()


    DOXYGEN_START_INTERNAL_BLOCK()
    class AGXPHYSICS_EXPORT BulkMaterialSemantics : public agx::ReferencedSemantics
    {
    public:

      // Automatic getters
      agx::Real const& getDensity() const;
      agx::Real const& getYoungsModulus() const;
      agx::Real const& getViscosity() const;
      agx::Real const& getDamping() const;
      agx::Real const& getMinElasticRestLength() const;
      agx::Real const& getMaxElasticRestLength() const;
      agx::Bool const& getDirty() const;

      // Semantics defined by explicit kernels

      // Automatic setters
      void setDensity(agx::Real const& value);
      void setYoungsModulus(agx::Real const& value);
      void setViscosity(agx::Real const& value);
      void setDamping(agx::Real const& value);
      void setMinElasticRestLength(agx::Real const& value);
      void setMaxElasticRestLength(agx::Real const& value);
      void setDirty(agx::Bool const& value);


    protected:
      friend class BulkMaterialPtr;
      friend class BulkMaterialInstance;
      BulkMaterialSemantics();
    };
    DOXYGEN_END_INTERNAL_BLOCK()


    /**
    Pointer to a entity instance of type Physics.BulkMaterial
    */
    class CALLABLE BulkMaterialPtr : public agx::ReferencedPtr
    {
    public:
      typedef BulkMaterialModel ModelType;
      typedef BulkMaterialData DataType;
      typedef BulkMaterialInstance InstanceType;

    public:
      AGXPHYSICS_EXPORT BulkMaterialPtr();
      AGXPHYSICS_EXPORT BulkMaterialPtr(agxData::EntityStorage* storage, agx::Index id);
      AGXPHYSICS_EXPORT BulkMaterialPtr(const agxData::EntityPtr& ptr);
      AGXPHYSICS_EXPORT BulkMaterialPtr(const agxData::EntityInstance& instance);
      AGXPHYSICS_EXPORT BulkMaterialPtr& operator= (const agxData::EntityPtr& ptr);
      AGXPHYSICS_EXPORT BulkMaterialPtr& operator= (const agxData::EntityInstance& instance);
      AGXPHYSICS_EXPORT BulkMaterialInstance instance();
      AGXPHYSICS_EXPORT const BulkMaterialInstance instance() const;

      AGXPHYSICS_EXPORT BulkMaterialSemantics* operator->();
      AGXPHYSICS_EXPORT const BulkMaterialSemantics* operator->() const;

      BulkMaterialData* getData();
      const BulkMaterialData* getData() const;


      /// \return reference to the density attribute
      AGXPHYSICS_EXPORT agx::Real& density();
      /// \return const reference to the density attribute
      AGXPHYSICS_EXPORT agx::Real const& density() const;

      /// \return reference to the youngsModulus attribute
      AGXPHYSICS_EXPORT agx::Real& youngsModulus();
      /// \return const reference to the youngsModulus attribute
      AGXPHYSICS_EXPORT agx::Real const& youngsModulus() const;

      /// \return reference to the viscosity attribute
      AGXPHYSICS_EXPORT agx::Real& viscosity();
      /// \return const reference to the viscosity attribute
      AGXPHYSICS_EXPORT agx::Real const& viscosity() const;

      /// \return reference to the damping attribute
      AGXPHYSICS_EXPORT agx::Real& damping();
      /// \return const reference to the damping attribute
      AGXPHYSICS_EXPORT agx::Real const& damping() const;

      /// \return reference to the minElasticRestLength attribute
      AGXPHYSICS_EXPORT agx::Real& minElasticRestLength();
      /// \return const reference to the minElasticRestLength attribute
      AGXPHYSICS_EXPORT agx::Real const& minElasticRestLength() const;

      /// \return reference to the maxElasticRestLength attribute
      AGXPHYSICS_EXPORT agx::Real& maxElasticRestLength();
      /// \return const reference to the maxElasticRestLength attribute
      AGXPHYSICS_EXPORT agx::Real const& maxElasticRestLength() const;

      /// \return reference to the dirty attribute
      AGXPHYSICS_EXPORT agx::Bool& dirty();
      /// \return const reference to the dirty attribute
      AGXPHYSICS_EXPORT agx::Bool const& dirty() const;

      DOXYGEN_START_INTERNAL_BLOCK()
      // Only used by tolua!
      #ifdef AGX_GENERATED_TOLUA_WRAPPER
      inline agx::Real const& getDensity() const { return (*this)->getDensity(); }
      inline agx::Real const& getYoungsModulus() const { return (*this)->getYoungsModulus(); }
      inline agx::Real const& getViscosity() const { return (*this)->getViscosity(); }
      inline agx::Real const& getDamping() const { return (*this)->getDamping(); }
      inline agx::Real const& getMinElasticRestLength() const { return (*this)->getMinElasticRestLength(); }
      inline agx::Real const& getMaxElasticRestLength() const { return (*this)->getMaxElasticRestLength(); }
      inline agx::Bool const& getDirty() const { return (*this)->getDirty(); }


      inline void setDensity(agx::Real const& value) { (*this)->setDensity(value); }
      inline void setYoungsModulus(agx::Real const& value) { (*this)->setYoungsModulus(value); }
      inline void setViscosity(agx::Real const& value) { (*this)->setViscosity(value); }
      inline void setDamping(agx::Real const& value) { (*this)->setDamping(value); }
      inline void setMinElasticRestLength(agx::Real const& value) { (*this)->setMinElasticRestLength(value); }
      inline void setMaxElasticRestLength(agx::Real const& value) { (*this)->setMaxElasticRestLength(value); }
      inline void setDirty(agx::Bool const& value) { (*this)->setDirty(value); }
      #endif
      DOXYGEN_END_INTERNAL_BLOCK()

    };

    // Entity is Referenced
    typedef agxData::EntityRef< BulkMaterialPtr > BulkMaterialRef;


    DOXYGEN_START_INTERNAL_BLOCK()
    class AGXPHYSICS_EXPORT BulkMaterialInstance : public agx::ReferencedInstance
    {
    public:
      BulkMaterialInstance();
      BulkMaterialInstance(BulkMaterialData* data, agx::Index index);
      BulkMaterialInstance(agxData::EntityStorage *storage, agx::Index index);
      BulkMaterialInstance(const agxData::EntityInstance& other);
      BulkMaterialInstance(const agxData::EntityPtr& ptr);

      BulkMaterialData* getData();
      const BulkMaterialData* getData() const;

    public:
      /// \return reference to the density attribute
      agx::Real& density();
      /// \return const reference to the density attribute
      agx::Real const& density() const;

      /// \return reference to the youngsModulus attribute
      agx::Real& youngsModulus();
      /// \return const reference to the youngsModulus attribute
      agx::Real const& youngsModulus() const;

      /// \return reference to the viscosity attribute
      agx::Real& viscosity();
      /// \return const reference to the viscosity attribute
      agx::Real const& viscosity() const;

      /// \return reference to the damping attribute
      agx::Real& damping();
      /// \return const reference to the damping attribute
      agx::Real const& damping() const;

      /// \return reference to the minElasticRestLength attribute
      agx::Real& minElasticRestLength();
      /// \return const reference to the minElasticRestLength attribute
      agx::Real const& minElasticRestLength() const;

      /// \return reference to the maxElasticRestLength attribute
      agx::Real& maxElasticRestLength();
      /// \return const reference to the maxElasticRestLength attribute
      agx::Real const& maxElasticRestLength() const;

      /// \return reference to the dirty attribute
      agx::Bool& dirty();
      /// \return const reference to the dirty attribute
      agx::Bool const& dirty() const;

    };
    DOXYGEN_END_INTERNAL_BLOCK()



    typedef agx::VectorPOD<BulkMaterialPtr> BulkMaterialPtrVector;
    typedef agxData::Array<BulkMaterialPtr> BulkMaterialPtrArray;



    DOXYGEN_START_INTERNAL_BLOCK()
    /* Implementation */
    //-----------------------------------------------------------------------------------------------------
    //-----------------------------------------------------------------------------------------------------
    inline BulkMaterialInstance agx::Physics::BulkMaterialData::operator[] (size_t index) { return BulkMaterialInstance(this, (agx::Index)index); }
    //-----------------------------------------------------------------------------------------------------
    AGX_FORCE_INLINE BulkMaterialPtr::BulkMaterialPtr() {}
    AGX_FORCE_INLINE BulkMaterialPtr::BulkMaterialPtr(agxData::EntityStorage* storage, agx::Index id) : agx::ReferencedPtr(storage, id) {}
    AGX_FORCE_INLINE BulkMaterialPtr::BulkMaterialPtr(const agxData::EntityPtr& ptr) : agx::ReferencedPtr(ptr)
    {
      agxAssertN(!ptr || ptr.isInstanceOf(BulkMaterialModel::instance()),
        "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
        EntityPtr::getModel()->fullPath().c_str(), BulkMaterialModel::instance()->fullPath().c_str());
    }

    AGX_FORCE_INLINE BulkMaterialPtr::BulkMaterialPtr(const agxData::EntityInstance& instance) : agx::ReferencedPtr(instance)
    {
      agxAssertN(!instance || instance.isInstanceOf(BulkMaterialModel::instance()),
        "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
        EntityPtr::getModel()->fullPath().c_str(), BulkMaterialModel::instance()->fullPath().c_str());
    }

    AGX_FORCE_INLINE BulkMaterialPtr& BulkMaterialPtr::operator= (const agxData::EntityPtr& ptr)
    {
      agxData::EntityPtr::operator= (ptr);
      agxAssertN(!ptr || ptr.isInstanceOf(BulkMaterialModel::instance()),
        "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
        EntityPtr::getModel()->fullPath().c_str(), BulkMaterialModel::instance()->fullPath().c_str());
      return *this;
    }

    AGX_FORCE_INLINE BulkMaterialPtr& BulkMaterialPtr::operator= (const agxData::EntityInstance& instance)
    {
      agxData::EntityPtr::operator= (instance);
      agxAssertN(!instance || instance.isInstanceOf(BulkMaterialModel::instance()),
        "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
        EntityPtr::getModel()->fullPath().c_str(), BulkMaterialModel::instance()->fullPath().c_str());
      return *this;
    }

    inline BulkMaterialInstance BulkMaterialPtr::instance() { return agxData::EntityPtr::instance(); }
    inline const BulkMaterialInstance BulkMaterialPtr::instance() const { return agxData::EntityPtr::instance(); }
    AGX_FORCE_INLINE BulkMaterialSemantics* BulkMaterialPtr::operator->() { return (BulkMaterialSemantics* )this; }
    AGX_FORCE_INLINE const BulkMaterialSemantics* BulkMaterialPtr::operator->() const { return (const BulkMaterialSemantics* )this; }
    AGX_FORCE_INLINE BulkMaterialData* BulkMaterialPtr::getData() { return static_cast<BulkMaterialData* >(agxData::EntityPtr::getData()); }
    AGX_FORCE_INLINE const BulkMaterialData* BulkMaterialPtr::getData() const { return static_cast<const BulkMaterialData* >(agxData::EntityPtr::getData()); }

    AGX_FORCE_INLINE agx::Real& BulkMaterialPtr::density() { verifyIndex(); return getData()->density[calculateIndex()]; }
    AGX_FORCE_INLINE agx::Real const& BulkMaterialPtr::density() const { verifyIndex(); return getData()->density[calculateIndex()]; }

    AGX_FORCE_INLINE agx::Real& BulkMaterialPtr::youngsModulus() { verifyIndex(); return getData()->youngsModulus[calculateIndex()]; }
    AGX_FORCE_INLINE agx::Real const& BulkMaterialPtr::youngsModulus() const { verifyIndex(); return getData()->youngsModulus[calculateIndex()]; }

    AGX_FORCE_INLINE agx::Real& BulkMaterialPtr::viscosity() { verifyIndex(); return getData()->viscosity[calculateIndex()]; }
    AGX_FORCE_INLINE agx::Real const& BulkMaterialPtr::viscosity() const { verifyIndex(); return getData()->viscosity[calculateIndex()]; }

    AGX_FORCE_INLINE agx::Real& BulkMaterialPtr::damping() { verifyIndex(); return getData()->damping[calculateIndex()]; }
    AGX_FORCE_INLINE agx::Real const& BulkMaterialPtr::damping() const { verifyIndex(); return getData()->damping[calculateIndex()]; }

    AGX_FORCE_INLINE agx::Real& BulkMaterialPtr::minElasticRestLength() { verifyIndex(); return getData()->minElasticRestLength[calculateIndex()]; }
    AGX_FORCE_INLINE agx::Real const& BulkMaterialPtr::minElasticRestLength() const { verifyIndex(); return getData()->minElasticRestLength[calculateIndex()]; }

    AGX_FORCE_INLINE agx::Real& BulkMaterialPtr::maxElasticRestLength() { verifyIndex(); return getData()->maxElasticRestLength[calculateIndex()]; }
    AGX_FORCE_INLINE agx::Real const& BulkMaterialPtr::maxElasticRestLength() const { verifyIndex(); return getData()->maxElasticRestLength[calculateIndex()]; }

    AGX_FORCE_INLINE agx::Bool& BulkMaterialPtr::dirty() { verifyIndex(); return getData()->dirty[calculateIndex()]; }
    AGX_FORCE_INLINE agx::Bool const& BulkMaterialPtr::dirty() const { verifyIndex(); return getData()->dirty[calculateIndex()]; }

    //-----------------------------------------------------------------------------------------------------
    AGX_FORCE_INLINE BulkMaterialInstance::BulkMaterialInstance() {}
    AGX_FORCE_INLINE BulkMaterialInstance::BulkMaterialInstance(BulkMaterialData* data, agx::Index index) : agx::ReferencedInstance(data, index) {}
    AGX_FORCE_INLINE BulkMaterialInstance::BulkMaterialInstance(agxData::EntityStorage* storage, agx::Index index) : agx::ReferencedInstance(storage, index) {}
    AGX_FORCE_INLINE BulkMaterialInstance::BulkMaterialInstance(const agxData::EntityInstance& other) : agx::ReferencedInstance(other)
    {
      agxAssertN(!other || other.isInstanceOf(BulkMaterialModel::instance()),
        "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
        EntityInstance::getModel()->fullPath().c_str(), BulkMaterialModel::instance()->fullPath().c_str());
    }

    AGX_FORCE_INLINE BulkMaterialInstance::BulkMaterialInstance(const agxData::EntityPtr& ptr) : agx::ReferencedInstance(ptr)
    {
      agxAssertN(!ptr || ptr.isInstanceOf(BulkMaterialModel::instance()),
        "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
        EntityInstance::getModel()->fullPath().c_str(), BulkMaterialModel::instance()->fullPath().c_str());
    }


    AGX_FORCE_INLINE BulkMaterialData* BulkMaterialInstance::getData() { return static_cast<BulkMaterialData* >(agxData::EntityInstance::getData()); }
    AGX_FORCE_INLINE const BulkMaterialData* BulkMaterialInstance::getData() const { return static_cast<const BulkMaterialData* >(agxData::EntityInstance::getData()); }

    AGX_FORCE_INLINE agx::Real& BulkMaterialInstance::density() { verifyIndex(); return getData()->density[getIndex()]; }
    AGX_FORCE_INLINE agx::Real const& BulkMaterialInstance::density() const { verifyIndex(); return getData()->density[getIndex()]; }

    AGX_FORCE_INLINE agx::Real& BulkMaterialInstance::youngsModulus() { verifyIndex(); return getData()->youngsModulus[getIndex()]; }
    AGX_FORCE_INLINE agx::Real const& BulkMaterialInstance::youngsModulus() const { verifyIndex(); return getData()->youngsModulus[getIndex()]; }

    AGX_FORCE_INLINE agx::Real& BulkMaterialInstance::viscosity() { verifyIndex(); return getData()->viscosity[getIndex()]; }
    AGX_FORCE_INLINE agx::Real const& BulkMaterialInstance::viscosity() const { verifyIndex(); return getData()->viscosity[getIndex()]; }

    AGX_FORCE_INLINE agx::Real& BulkMaterialInstance::damping() { verifyIndex(); return getData()->damping[getIndex()]; }
    AGX_FORCE_INLINE agx::Real const& BulkMaterialInstance::damping() const { verifyIndex(); return getData()->damping[getIndex()]; }

    AGX_FORCE_INLINE agx::Real& BulkMaterialInstance::minElasticRestLength() { verifyIndex(); return getData()->minElasticRestLength[getIndex()]; }
    AGX_FORCE_INLINE agx::Real const& BulkMaterialInstance::minElasticRestLength() const { verifyIndex(); return getData()->minElasticRestLength[getIndex()]; }

    AGX_FORCE_INLINE agx::Real& BulkMaterialInstance::maxElasticRestLength() { verifyIndex(); return getData()->maxElasticRestLength[getIndex()]; }
    AGX_FORCE_INLINE agx::Real const& BulkMaterialInstance::maxElasticRestLength() const { verifyIndex(); return getData()->maxElasticRestLength[getIndex()]; }

    AGX_FORCE_INLINE agx::Bool& BulkMaterialInstance::dirty() { verifyIndex(); return getData()->dirty[getIndex()]; }
    AGX_FORCE_INLINE agx::Bool const& BulkMaterialInstance::dirty() const { verifyIndex(); return getData()->dirty[getIndex()]; }

    //-----------------------------------------------------------------------------------------------------
    AGX_FORCE_INLINE BulkMaterialSemantics::BulkMaterialSemantics() {}
    //-----------------------------------------------------------------------------------------------------
    DOXYGEN_END_INTERNAL_BLOCK()
  }
}

AGX_TYPE_BINDING(agx::Physics::BulkMaterialPtr, "Physics.BulkMaterialPtr")
AGX_TYPE_BINDING(agx::Physics::BulkMaterialInstance, "Physics.BulkMaterialInstance")

#ifdef _MSC_VER
# pragma warning(pop)
#endif

#undef AGX_ENTITY_WRAPPER
#undef AGX_ENTITY_NAMESPACE
#endif

