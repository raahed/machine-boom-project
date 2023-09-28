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

#ifndef GENERATED_AGX_PHYSICS_WIREMATERIAL_H_PLUGIN
#define GENERATED_AGX_PHYSICS_WIREMATERIAL_H_PLUGIN

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

    class WireMaterialModel;
    class WireMaterialData;
    class WireMaterialPtr;
    class WireMaterialInstance;
    class WireMaterialSemantics;


    AGX_DECLARE_POINTER_TYPES(WireMaterialModel);

    /** 
    Abstract description of the data attributes for the Physics.WireMaterial entity.
    */ 
    class AGXPHYSICS_EXPORT WireMaterialModel : public agx::ReferencedModel
    {
    public:
      typedef WireMaterialPtr PtrT;

      WireMaterialModel(const agx::String& name = "WireMaterial");

      /// \return The entity model singleton.
      static WireMaterialModel* instance();

      /// Create and return a pointer to a new instance in the default storage for this entity model.
      static WireMaterialPtr createInstance();

      /// \return The default storage for this entity model.
      static agxData::EntityStorage* defaultStorage();

      /// This is part of internal cleanup and should not be called by users
      virtual void shutdownCleanup() override;



      /* Attributes */
      static agxData::ScalarAttributeT< agx::Real >* youngsModulusStretchAttribute;
      static agxData::ScalarAttributeT< agx::Real >* youngsModulusBendAttribute;
      static agxData::ScalarAttributeT< agx::Real >* dampingStretchAttribute;
      static agxData::ScalarAttributeT< agx::Real >* dampingBendAttribute;
      static agxData::ScalarAttributeT< agx::Bool >* dirtyAttribute;

    protected:
      virtual ~WireMaterialModel();
      virtual agxData::EntityData* createData(agxData::EntityStorage* storage) override;
      virtual void configure(agx::TiXmlElement* eEntity) override;
      virtual void initAttributeAccessors() override;
      virtual void construct(agxData::EntityPtr instance) override;
      void construct(agx::Physics::WireMaterialPtr wireMaterial);
    };


    DOXYGEN_START_INTERNAL_BLOCK()
    #ifndef AGX_PHYSICS_WIREMATERIAL_DATA_SET_OVERRIDE
    #define AGX_PHYSICS_WIREMATERIAL_DATA_SET
    class AGXPHYSICS_EXPORT WireMaterialData : public agx::ReferencedData
    {
    public:
      WireMaterialInstance operator[] (size_t index);

    public:
      agxData::Array< WireMaterialPtr >& instance;
      agxData::Array< agx::Real > youngsModulusStretch;
      agxData::Array< agx::Real > youngsModulusBend;
      agxData::Array< agx::Real > dampingStretch;
      agxData::Array< agx::Real > dampingBend;
      agxData::Array< agx::Bool > dirty;

    public:
      typedef agx::Real youngsModulusStretchType;
      typedef agx::Real youngsModulusBendType;
      typedef agx::Real dampingStretchType;
      typedef agx::Real dampingBendType;
      typedef agx::Bool dirtyType;

    public:
      WireMaterialData(agxData::EntityStorage* storage);
      WireMaterialData();

    protected:
      virtual ~WireMaterialData() {}
      virtual void setNumElements(agx::Index numElements) override;

    private:
      WireMaterialData& operator= (const WireMaterialData&) { return *this; }

    };
    #endif
    DOXYGEN_END_INTERNAL_BLOCK()


    DOXYGEN_START_INTERNAL_BLOCK()
    class AGXPHYSICS_EXPORT WireMaterialSemantics : public agx::ReferencedSemantics
    {
    public:

      // Automatic getters
      agx::Real const& getYoungsModulusStretch() const;
      agx::Real const& getYoungsModulusBend() const;
      agx::Real const& getDampingStretch() const;
      agx::Real const& getDampingBend() const;
      agx::Bool const& getDirty() const;

      // Semantics defined by explicit kernels

      // Automatic setters
      void setYoungsModulusStretch(agx::Real const& value);
      void setYoungsModulusBend(agx::Real const& value);
      void setDampingStretch(agx::Real const& value);
      void setDampingBend(agx::Real const& value);
      void setDirty(agx::Bool const& value);


    protected:
      friend class WireMaterialPtr;
      friend class WireMaterialInstance;
      WireMaterialSemantics();
    };
    DOXYGEN_END_INTERNAL_BLOCK()


    /**
    Pointer to a entity instance of type Physics.WireMaterial
    */
    class CALLABLE WireMaterialPtr : public agx::ReferencedPtr
    {
    public:
      typedef WireMaterialModel ModelType;
      typedef WireMaterialData DataType;
      typedef WireMaterialInstance InstanceType;

    public:
      AGXPHYSICS_EXPORT WireMaterialPtr();
      AGXPHYSICS_EXPORT WireMaterialPtr(agxData::EntityStorage* storage, agx::Index id);
      AGXPHYSICS_EXPORT WireMaterialPtr(const agxData::EntityPtr& ptr);
      AGXPHYSICS_EXPORT WireMaterialPtr(const agxData::EntityInstance& instance);
      AGXPHYSICS_EXPORT WireMaterialPtr& operator= (const agxData::EntityPtr& ptr);
      AGXPHYSICS_EXPORT WireMaterialPtr& operator= (const agxData::EntityInstance& instance);
      AGXPHYSICS_EXPORT WireMaterialInstance instance();
      AGXPHYSICS_EXPORT const WireMaterialInstance instance() const;

      AGXPHYSICS_EXPORT WireMaterialSemantics* operator->();
      AGXPHYSICS_EXPORT const WireMaterialSemantics* operator->() const;

      WireMaterialData* getData();
      const WireMaterialData* getData() const;


      /// \return reference to the youngsModulusStretch attribute
      AGXPHYSICS_EXPORT agx::Real& youngsModulusStretch();
      /// \return const reference to the youngsModulusStretch attribute
      AGXPHYSICS_EXPORT agx::Real const& youngsModulusStretch() const;

      /// \return reference to the youngsModulusBend attribute
      AGXPHYSICS_EXPORT agx::Real& youngsModulusBend();
      /// \return const reference to the youngsModulusBend attribute
      AGXPHYSICS_EXPORT agx::Real const& youngsModulusBend() const;

      /// \return reference to the dampingStretch attribute
      AGXPHYSICS_EXPORT agx::Real& dampingStretch();
      /// \return const reference to the dampingStretch attribute
      AGXPHYSICS_EXPORT agx::Real const& dampingStretch() const;

      /// \return reference to the dampingBend attribute
      AGXPHYSICS_EXPORT agx::Real& dampingBend();
      /// \return const reference to the dampingBend attribute
      AGXPHYSICS_EXPORT agx::Real const& dampingBend() const;

      /// \return reference to the dirty attribute
      AGXPHYSICS_EXPORT agx::Bool& dirty();
      /// \return const reference to the dirty attribute
      AGXPHYSICS_EXPORT agx::Bool const& dirty() const;

      DOXYGEN_START_INTERNAL_BLOCK()
      // Only used by tolua!
      #ifdef AGX_GENERATED_TOLUA_WRAPPER
      inline agx::Real const& getYoungsModulusStretch() const { return (*this)->getYoungsModulusStretch(); }
      inline agx::Real const& getYoungsModulusBend() const { return (*this)->getYoungsModulusBend(); }
      inline agx::Real const& getDampingStretch() const { return (*this)->getDampingStretch(); }
      inline agx::Real const& getDampingBend() const { return (*this)->getDampingBend(); }
      inline agx::Bool const& getDirty() const { return (*this)->getDirty(); }


      inline void setYoungsModulusStretch(agx::Real const& value) { (*this)->setYoungsModulusStretch(value); }
      inline void setYoungsModulusBend(agx::Real const& value) { (*this)->setYoungsModulusBend(value); }
      inline void setDampingStretch(agx::Real const& value) { (*this)->setDampingStretch(value); }
      inline void setDampingBend(agx::Real const& value) { (*this)->setDampingBend(value); }
      inline void setDirty(agx::Bool const& value) { (*this)->setDirty(value); }
      #endif
      DOXYGEN_END_INTERNAL_BLOCK()

    };

    // Entity is Referenced
    typedef agxData::EntityRef< WireMaterialPtr > WireMaterialRef;


    DOXYGEN_START_INTERNAL_BLOCK()
    class AGXPHYSICS_EXPORT WireMaterialInstance : public agx::ReferencedInstance
    {
    public:
      WireMaterialInstance();
      WireMaterialInstance(WireMaterialData* data, agx::Index index);
      WireMaterialInstance(agxData::EntityStorage *storage, agx::Index index);
      WireMaterialInstance(const agxData::EntityInstance& other);
      WireMaterialInstance(const agxData::EntityPtr& ptr);

      WireMaterialData* getData();
      const WireMaterialData* getData() const;

    public:
      /// \return reference to the youngsModulusStretch attribute
      agx::Real& youngsModulusStretch();
      /// \return const reference to the youngsModulusStretch attribute
      agx::Real const& youngsModulusStretch() const;

      /// \return reference to the youngsModulusBend attribute
      agx::Real& youngsModulusBend();
      /// \return const reference to the youngsModulusBend attribute
      agx::Real const& youngsModulusBend() const;

      /// \return reference to the dampingStretch attribute
      agx::Real& dampingStretch();
      /// \return const reference to the dampingStretch attribute
      agx::Real const& dampingStretch() const;

      /// \return reference to the dampingBend attribute
      agx::Real& dampingBend();
      /// \return const reference to the dampingBend attribute
      agx::Real const& dampingBend() const;

      /// \return reference to the dirty attribute
      agx::Bool& dirty();
      /// \return const reference to the dirty attribute
      agx::Bool const& dirty() const;

    };
    DOXYGEN_END_INTERNAL_BLOCK()



    typedef agx::VectorPOD<WireMaterialPtr> WireMaterialPtrVector;
    typedef agxData::Array<WireMaterialPtr> WireMaterialPtrArray;



    DOXYGEN_START_INTERNAL_BLOCK()
    /* Implementation */
    //-----------------------------------------------------------------------------------------------------
    //-----------------------------------------------------------------------------------------------------
    inline WireMaterialInstance agx::Physics::WireMaterialData::operator[] (size_t index) { return WireMaterialInstance(this, (agx::Index)index); }
    //-----------------------------------------------------------------------------------------------------
    AGX_FORCE_INLINE WireMaterialPtr::WireMaterialPtr() {}
    AGX_FORCE_INLINE WireMaterialPtr::WireMaterialPtr(agxData::EntityStorage* storage, agx::Index id) : agx::ReferencedPtr(storage, id) {}
    AGX_FORCE_INLINE WireMaterialPtr::WireMaterialPtr(const agxData::EntityPtr& ptr) : agx::ReferencedPtr(ptr)
    {
      agxAssertN(!ptr || ptr.isInstanceOf(WireMaterialModel::instance()),
        "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
        EntityPtr::getModel()->fullPath().c_str(), WireMaterialModel::instance()->fullPath().c_str());
    }

    AGX_FORCE_INLINE WireMaterialPtr::WireMaterialPtr(const agxData::EntityInstance& instance) : agx::ReferencedPtr(instance)
    {
      agxAssertN(!instance || instance.isInstanceOf(WireMaterialModel::instance()),
        "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
        EntityPtr::getModel()->fullPath().c_str(), WireMaterialModel::instance()->fullPath().c_str());
    }

    AGX_FORCE_INLINE WireMaterialPtr& WireMaterialPtr::operator= (const agxData::EntityPtr& ptr)
    {
      agxData::EntityPtr::operator= (ptr);
      agxAssertN(!ptr || ptr.isInstanceOf(WireMaterialModel::instance()),
        "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
        EntityPtr::getModel()->fullPath().c_str(), WireMaterialModel::instance()->fullPath().c_str());
      return *this;
    }

    AGX_FORCE_INLINE WireMaterialPtr& WireMaterialPtr::operator= (const agxData::EntityInstance& instance)
    {
      agxData::EntityPtr::operator= (instance);
      agxAssertN(!instance || instance.isInstanceOf(WireMaterialModel::instance()),
        "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
        EntityPtr::getModel()->fullPath().c_str(), WireMaterialModel::instance()->fullPath().c_str());
      return *this;
    }

    inline WireMaterialInstance WireMaterialPtr::instance() { return agxData::EntityPtr::instance(); }
    inline const WireMaterialInstance WireMaterialPtr::instance() const { return agxData::EntityPtr::instance(); }
    AGX_FORCE_INLINE WireMaterialSemantics* WireMaterialPtr::operator->() { return (WireMaterialSemantics* )this; }
    AGX_FORCE_INLINE const WireMaterialSemantics* WireMaterialPtr::operator->() const { return (const WireMaterialSemantics* )this; }
    AGX_FORCE_INLINE WireMaterialData* WireMaterialPtr::getData() { return static_cast<WireMaterialData* >(agxData::EntityPtr::getData()); }
    AGX_FORCE_INLINE const WireMaterialData* WireMaterialPtr::getData() const { return static_cast<const WireMaterialData* >(agxData::EntityPtr::getData()); }

    AGX_FORCE_INLINE agx::Real& WireMaterialPtr::youngsModulusStretch() { verifyIndex(); return getData()->youngsModulusStretch[calculateIndex()]; }
    AGX_FORCE_INLINE agx::Real const& WireMaterialPtr::youngsModulusStretch() const { verifyIndex(); return getData()->youngsModulusStretch[calculateIndex()]; }

    AGX_FORCE_INLINE agx::Real& WireMaterialPtr::youngsModulusBend() { verifyIndex(); return getData()->youngsModulusBend[calculateIndex()]; }
    AGX_FORCE_INLINE agx::Real const& WireMaterialPtr::youngsModulusBend() const { verifyIndex(); return getData()->youngsModulusBend[calculateIndex()]; }

    AGX_FORCE_INLINE agx::Real& WireMaterialPtr::dampingStretch() { verifyIndex(); return getData()->dampingStretch[calculateIndex()]; }
    AGX_FORCE_INLINE agx::Real const& WireMaterialPtr::dampingStretch() const { verifyIndex(); return getData()->dampingStretch[calculateIndex()]; }

    AGX_FORCE_INLINE agx::Real& WireMaterialPtr::dampingBend() { verifyIndex(); return getData()->dampingBend[calculateIndex()]; }
    AGX_FORCE_INLINE agx::Real const& WireMaterialPtr::dampingBend() const { verifyIndex(); return getData()->dampingBend[calculateIndex()]; }

    AGX_FORCE_INLINE agx::Bool& WireMaterialPtr::dirty() { verifyIndex(); return getData()->dirty[calculateIndex()]; }
    AGX_FORCE_INLINE agx::Bool const& WireMaterialPtr::dirty() const { verifyIndex(); return getData()->dirty[calculateIndex()]; }

    //-----------------------------------------------------------------------------------------------------
    AGX_FORCE_INLINE WireMaterialInstance::WireMaterialInstance() {}
    AGX_FORCE_INLINE WireMaterialInstance::WireMaterialInstance(WireMaterialData* data, agx::Index index) : agx::ReferencedInstance(data, index) {}
    AGX_FORCE_INLINE WireMaterialInstance::WireMaterialInstance(agxData::EntityStorage* storage, agx::Index index) : agx::ReferencedInstance(storage, index) {}
    AGX_FORCE_INLINE WireMaterialInstance::WireMaterialInstance(const agxData::EntityInstance& other) : agx::ReferencedInstance(other)
    {
      agxAssertN(!other || other.isInstanceOf(WireMaterialModel::instance()),
        "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
        EntityInstance::getModel()->fullPath().c_str(), WireMaterialModel::instance()->fullPath().c_str());
    }

    AGX_FORCE_INLINE WireMaterialInstance::WireMaterialInstance(const agxData::EntityPtr& ptr) : agx::ReferencedInstance(ptr)
    {
      agxAssertN(!ptr || ptr.isInstanceOf(WireMaterialModel::instance()),
        "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
        EntityInstance::getModel()->fullPath().c_str(), WireMaterialModel::instance()->fullPath().c_str());
    }


    AGX_FORCE_INLINE WireMaterialData* WireMaterialInstance::getData() { return static_cast<WireMaterialData* >(agxData::EntityInstance::getData()); }
    AGX_FORCE_INLINE const WireMaterialData* WireMaterialInstance::getData() const { return static_cast<const WireMaterialData* >(agxData::EntityInstance::getData()); }

    AGX_FORCE_INLINE agx::Real& WireMaterialInstance::youngsModulusStretch() { verifyIndex(); return getData()->youngsModulusStretch[getIndex()]; }
    AGX_FORCE_INLINE agx::Real const& WireMaterialInstance::youngsModulusStretch() const { verifyIndex(); return getData()->youngsModulusStretch[getIndex()]; }

    AGX_FORCE_INLINE agx::Real& WireMaterialInstance::youngsModulusBend() { verifyIndex(); return getData()->youngsModulusBend[getIndex()]; }
    AGX_FORCE_INLINE agx::Real const& WireMaterialInstance::youngsModulusBend() const { verifyIndex(); return getData()->youngsModulusBend[getIndex()]; }

    AGX_FORCE_INLINE agx::Real& WireMaterialInstance::dampingStretch() { verifyIndex(); return getData()->dampingStretch[getIndex()]; }
    AGX_FORCE_INLINE agx::Real const& WireMaterialInstance::dampingStretch() const { verifyIndex(); return getData()->dampingStretch[getIndex()]; }

    AGX_FORCE_INLINE agx::Real& WireMaterialInstance::dampingBend() { verifyIndex(); return getData()->dampingBend[getIndex()]; }
    AGX_FORCE_INLINE agx::Real const& WireMaterialInstance::dampingBend() const { verifyIndex(); return getData()->dampingBend[getIndex()]; }

    AGX_FORCE_INLINE agx::Bool& WireMaterialInstance::dirty() { verifyIndex(); return getData()->dirty[getIndex()]; }
    AGX_FORCE_INLINE agx::Bool const& WireMaterialInstance::dirty() const { verifyIndex(); return getData()->dirty[getIndex()]; }

    //-----------------------------------------------------------------------------------------------------
    AGX_FORCE_INLINE WireMaterialSemantics::WireMaterialSemantics() {}
    //-----------------------------------------------------------------------------------------------------
    DOXYGEN_END_INTERNAL_BLOCK()
  }
}

AGX_TYPE_BINDING(agx::Physics::WireMaterialPtr, "Physics.WireMaterialPtr")
AGX_TYPE_BINDING(agx::Physics::WireMaterialInstance, "Physics.WireMaterialInstance")

#ifdef _MSC_VER
# pragma warning(pop)
#endif

#undef AGX_ENTITY_WRAPPER
#undef AGX_ENTITY_NAMESPACE
#endif

