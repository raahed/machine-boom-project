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

#ifndef GENERATED_AGX_PHYSICS_CONSTRAINTBASE_H_PLUGIN
#define GENERATED_AGX_PHYSICS_CONSTRAINTBASE_H_PLUGIN

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
#include <agx/Physics/InteractionEntity.h>
#include <agx/Integer.h>
namespace agx { class ConstraintImplementation; }


namespace agx
{
  namespace Physics
  {

    class ConstraintBaseModel;
    class ConstraintBaseData;
    class ConstraintBasePtr;
    class ConstraintBaseInstance;
    class ConstraintBaseSemantics;


    AGX_DECLARE_POINTER_TYPES(ConstraintBaseModel);

    /** 
    Abstract description of the data attributes for the Physics.ConstraintBase entity.
    */ 
    class AGXPHYSICS_EXPORT ConstraintBaseModel : public agx::Physics::InteractionModel
    {
    public:
      typedef ConstraintBasePtr PtrT;

      ConstraintBaseModel(const agx::String& name = "ConstraintBase");

      /// \return The entity model singleton.
      static ConstraintBaseModel* instance();

      /// Create and return a pointer to a new instance in the default storage for this entity model.
      static ConstraintBasePtr createInstance();

      /// \return The default storage for this entity model.
      static agxData::EntityStorage* defaultStorage();

      /// This is part of internal cleanup and should not be called by users
      virtual void shutdownCleanup() override;



      /* Attributes */
      static agxData::PointerAttributeT< agx::ConstraintImplementation*>* modelAttribute;
      static agxData::ScalarAttributeT< agx::UInt >* rowIndexAttribute;
      static agxData::ScalarAttributeT< agx::UInt >* numRowsAttribute;
      static agxData::ScalarAttributeT< agx::UInt >* jacobianIndexAttribute;
      static agxData::ScalarAttributeT< agx::UInt >* blockRowIndexAttribute;
      static agxData::ScalarAttributeT< agx::Bool >* isImpactingAttribute;

    protected:
      virtual ~ConstraintBaseModel();
      virtual agxData::EntityData* createData(agxData::EntityStorage* storage) override;
      virtual void configure(agx::TiXmlElement* eEntity) override;
      virtual void initAttributeAccessors() override;
      virtual void construct(agxData::EntityPtr instance) override;
      void construct(agx::Physics::ConstraintBasePtr constraintBase);
    };


    DOXYGEN_START_INTERNAL_BLOCK()
    #ifndef AGX_PHYSICS_CONSTRAINTBASE_DATA_SET_OVERRIDE
    #define AGX_PHYSICS_CONSTRAINTBASE_DATA_SET
    class AGXPHYSICS_EXPORT ConstraintBaseData : public agx::Physics::InteractionData
    {
    public:
      ConstraintBaseInstance operator[] (size_t index);

    public:
      agxData::Array< ConstraintBasePtr >& instance;
      agxData::Array< agx::ConstraintImplementation* > model;
      agxData::Array< agx::UInt > rowIndex;
      agxData::Array< agx::UInt > numRows;
      agxData::Array< agx::UInt > jacobianIndex;
      agxData::Array< agx::UInt > blockRowIndex;
      agxData::Array< agx::Bool > isImpacting;

    public:
      typedef agx::ConstraintImplementation* modelType;
      typedef agx::UInt rowIndexType;
      typedef agx::UInt numRowsType;
      typedef agx::UInt jacobianIndexType;
      typedef agx::UInt blockRowIndexType;
      typedef agx::Bool isImpactingType;

    public:
      ConstraintBaseData(agxData::EntityStorage* storage);
      ConstraintBaseData();

    protected:
      virtual ~ConstraintBaseData() {}
      virtual void setNumElements(agx::Index numElements) override;

    private:
      ConstraintBaseData& operator= (const ConstraintBaseData&) { return *this; }

    };
    #endif
    DOXYGEN_END_INTERNAL_BLOCK()


    DOXYGEN_START_INTERNAL_BLOCK()
    class AGXPHYSICS_EXPORT ConstraintBaseSemantics : public agx::Physics::InteractionSemantics
    {
    public:

      // Automatic getters
      agx::ConstraintImplementation* const& getModel() const;
      agx::UInt const& getRowIndex() const;
      agx::UInt const& getNumRows() const;
      agx::UInt const& getJacobianIndex() const;
      agx::UInt const& getBlockRowIndex() const;
      agx::Bool const& getIsImpacting() const;

      // Semantics defined by explicit kernels

      // Automatic setters
      void setModel(agx::ConstraintImplementation* const& value);
      void setRowIndex(agx::UInt const& value);
      void setNumRows(agx::UInt const& value);
      void setJacobianIndex(agx::UInt const& value);
      void setBlockRowIndex(agx::UInt const& value);
      void setIsImpacting(agx::Bool const& value);


    protected:
      friend class ConstraintBasePtr;
      friend class ConstraintBaseInstance;
      ConstraintBaseSemantics();
    };
    DOXYGEN_END_INTERNAL_BLOCK()


    /**
    Pointer to a entity instance of type Physics.ConstraintBase
    */
    class CALLABLE ConstraintBasePtr : public agx::Physics::InteractionPtr
    {
    public:
      typedef ConstraintBaseModel ModelType;
      typedef ConstraintBaseData DataType;
      typedef ConstraintBaseInstance InstanceType;

    public:
      AGXPHYSICS_EXPORT ConstraintBasePtr();
      AGXPHYSICS_EXPORT ConstraintBasePtr(agxData::EntityStorage* storage, agx::Index id);
      AGXPHYSICS_EXPORT ConstraintBasePtr(const agxData::EntityPtr& ptr);
      AGXPHYSICS_EXPORT ConstraintBasePtr(const agxData::EntityInstance& instance);
      AGXPHYSICS_EXPORT ConstraintBasePtr& operator= (const agxData::EntityPtr& ptr);
      AGXPHYSICS_EXPORT ConstraintBasePtr& operator= (const agxData::EntityInstance& instance);
      AGXPHYSICS_EXPORT ConstraintBaseInstance instance();
      AGXPHYSICS_EXPORT const ConstraintBaseInstance instance() const;

      AGXPHYSICS_EXPORT ConstraintBaseSemantics* operator->();
      AGXPHYSICS_EXPORT const ConstraintBaseSemantics* operator->() const;

      ConstraintBaseData* getData();
      const ConstraintBaseData* getData() const;


      /// \return reference to the model attribute
      AGXPHYSICS_EXPORT agx::ConstraintImplementation*& model();
      /// \return const reference to the model attribute
      AGXPHYSICS_EXPORT agx::ConstraintImplementation* const& model() const;

      /// \return reference to the rowIndex attribute
      AGXPHYSICS_EXPORT agx::UInt& rowIndex();
      /// \return const reference to the rowIndex attribute
      AGXPHYSICS_EXPORT agx::UInt const& rowIndex() const;

      /// \return reference to the numRows attribute
      AGXPHYSICS_EXPORT agx::UInt& numRows();
      /// \return const reference to the numRows attribute
      AGXPHYSICS_EXPORT agx::UInt const& numRows() const;

      /// \return reference to the jacobianIndex attribute
      AGXPHYSICS_EXPORT agx::UInt& jacobianIndex();
      /// \return const reference to the jacobianIndex attribute
      AGXPHYSICS_EXPORT agx::UInt const& jacobianIndex() const;

      /// \return reference to the blockRowIndex attribute
      AGXPHYSICS_EXPORT agx::UInt& blockRowIndex();
      /// \return const reference to the blockRowIndex attribute
      AGXPHYSICS_EXPORT agx::UInt const& blockRowIndex() const;

      /// \return reference to the isImpacting attribute
      AGXPHYSICS_EXPORT agx::Bool& isImpacting();
      /// \return const reference to the isImpacting attribute
      AGXPHYSICS_EXPORT agx::Bool const& isImpacting() const;

      DOXYGEN_START_INTERNAL_BLOCK()
      // Only used by tolua!
      #ifdef AGX_GENERATED_TOLUA_WRAPPER
      inline agx::ConstraintImplementation* const& getModel() const { return (*this)->getModel(); }
      inline agx::UInt const& getRowIndex() const { return (*this)->getRowIndex(); }
      inline agx::UInt const& getNumRows() const { return (*this)->getNumRows(); }
      inline agx::UInt const& getJacobianIndex() const { return (*this)->getJacobianIndex(); }
      inline agx::UInt const& getBlockRowIndex() const { return (*this)->getBlockRowIndex(); }
      inline agx::Bool const& getIsImpacting() const { return (*this)->getIsImpacting(); }


      inline void setModel(agx::ConstraintImplementation* const& value) { (*this)->setModel(value); }
      inline void setRowIndex(agx::UInt const& value) { (*this)->setRowIndex(value); }
      inline void setNumRows(agx::UInt const& value) { (*this)->setNumRows(value); }
      inline void setJacobianIndex(agx::UInt const& value) { (*this)->setJacobianIndex(value); }
      inline void setBlockRowIndex(agx::UInt const& value) { (*this)->setBlockRowIndex(value); }
      inline void setIsImpacting(agx::Bool const& value) { (*this)->setIsImpacting(value); }
      #endif
      DOXYGEN_END_INTERNAL_BLOCK()

    };


    DOXYGEN_START_INTERNAL_BLOCK()
    class AGXPHYSICS_EXPORT ConstraintBaseInstance : public agx::Physics::InteractionInstance
    {
    public:
      ConstraintBaseInstance();
      ConstraintBaseInstance(ConstraintBaseData* data, agx::Index index);
      ConstraintBaseInstance(agxData::EntityStorage *storage, agx::Index index);
      ConstraintBaseInstance(const agxData::EntityInstance& other);
      ConstraintBaseInstance(const agxData::EntityPtr& ptr);

      ConstraintBaseData* getData();
      const ConstraintBaseData* getData() const;

    public:
      /// \return reference to the model attribute
      agx::ConstraintImplementation*& model();
      /// \return const reference to the model attribute
      agx::ConstraintImplementation* const& model() const;

      /// \return reference to the rowIndex attribute
      agx::UInt& rowIndex();
      /// \return const reference to the rowIndex attribute
      agx::UInt const& rowIndex() const;

      /// \return reference to the numRows attribute
      agx::UInt& numRows();
      /// \return const reference to the numRows attribute
      agx::UInt const& numRows() const;

      /// \return reference to the jacobianIndex attribute
      agx::UInt& jacobianIndex();
      /// \return const reference to the jacobianIndex attribute
      agx::UInt const& jacobianIndex() const;

      /// \return reference to the blockRowIndex attribute
      agx::UInt& blockRowIndex();
      /// \return const reference to the blockRowIndex attribute
      agx::UInt const& blockRowIndex() const;

      /// \return reference to the isImpacting attribute
      agx::Bool& isImpacting();
      /// \return const reference to the isImpacting attribute
      agx::Bool const& isImpacting() const;

    };
    DOXYGEN_END_INTERNAL_BLOCK()



    typedef agx::VectorPOD<ConstraintBasePtr> ConstraintBasePtrVector;
    typedef agxData::Array<ConstraintBasePtr> ConstraintBasePtrArray;



    DOXYGEN_START_INTERNAL_BLOCK()
    /* Implementation */
    //-----------------------------------------------------------------------------------------------------
    //-----------------------------------------------------------------------------------------------------
    inline ConstraintBaseInstance agx::Physics::ConstraintBaseData::operator[] (size_t index) { return ConstraintBaseInstance(this, (agx::Index)index); }
    //-----------------------------------------------------------------------------------------------------
    AGX_FORCE_INLINE ConstraintBasePtr::ConstraintBasePtr() {}
    AGX_FORCE_INLINE ConstraintBasePtr::ConstraintBasePtr(agxData::EntityStorage* storage, agx::Index id) : agx::Physics::InteractionPtr(storage, id) {}
    AGX_FORCE_INLINE ConstraintBasePtr::ConstraintBasePtr(const agxData::EntityPtr& ptr) : agx::Physics::InteractionPtr(ptr)
    {
      agxAssertN(!ptr || ptr.isInstanceOf(ConstraintBaseModel::instance()),
        "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
        EntityPtr::getModel()->fullPath().c_str(), ConstraintBaseModel::instance()->fullPath().c_str());
    }

    AGX_FORCE_INLINE ConstraintBasePtr::ConstraintBasePtr(const agxData::EntityInstance& instance) : agx::Physics::InteractionPtr(instance)
    {
      agxAssertN(!instance || instance.isInstanceOf(ConstraintBaseModel::instance()),
        "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
        EntityPtr::getModel()->fullPath().c_str(), ConstraintBaseModel::instance()->fullPath().c_str());
    }

    AGX_FORCE_INLINE ConstraintBasePtr& ConstraintBasePtr::operator= (const agxData::EntityPtr& ptr)
    {
      agxData::EntityPtr::operator= (ptr);
      agxAssertN(!ptr || ptr.isInstanceOf(ConstraintBaseModel::instance()),
        "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
        EntityPtr::getModel()->fullPath().c_str(), ConstraintBaseModel::instance()->fullPath().c_str());
      return *this;
    }

    AGX_FORCE_INLINE ConstraintBasePtr& ConstraintBasePtr::operator= (const agxData::EntityInstance& instance)
    {
      agxData::EntityPtr::operator= (instance);
      agxAssertN(!instance || instance.isInstanceOf(ConstraintBaseModel::instance()),
        "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
        EntityPtr::getModel()->fullPath().c_str(), ConstraintBaseModel::instance()->fullPath().c_str());
      return *this;
    }

    inline ConstraintBaseInstance ConstraintBasePtr::instance() { return agxData::EntityPtr::instance(); }
    inline const ConstraintBaseInstance ConstraintBasePtr::instance() const { return agxData::EntityPtr::instance(); }
    AGX_FORCE_INLINE ConstraintBaseSemantics* ConstraintBasePtr::operator->() { return (ConstraintBaseSemantics* )this; }
    AGX_FORCE_INLINE const ConstraintBaseSemantics* ConstraintBasePtr::operator->() const { return (const ConstraintBaseSemantics* )this; }
    AGX_FORCE_INLINE ConstraintBaseData* ConstraintBasePtr::getData() { return static_cast<ConstraintBaseData* >(agxData::EntityPtr::getData()); }
    AGX_FORCE_INLINE const ConstraintBaseData* ConstraintBasePtr::getData() const { return static_cast<const ConstraintBaseData* >(agxData::EntityPtr::getData()); }

    AGX_FORCE_INLINE agx::ConstraintImplementation*& ConstraintBasePtr::model() { verifyIndex(); return getData()->model[calculateIndex()]; }
    AGX_FORCE_INLINE agx::ConstraintImplementation* const& ConstraintBasePtr::model() const { verifyIndex(); return getData()->model[calculateIndex()]; }

    AGX_FORCE_INLINE agx::UInt& ConstraintBasePtr::rowIndex() { verifyIndex(); return getData()->rowIndex[calculateIndex()]; }
    AGX_FORCE_INLINE agx::UInt const& ConstraintBasePtr::rowIndex() const { verifyIndex(); return getData()->rowIndex[calculateIndex()]; }

    AGX_FORCE_INLINE agx::UInt& ConstraintBasePtr::numRows() { verifyIndex(); return getData()->numRows[calculateIndex()]; }
    AGX_FORCE_INLINE agx::UInt const& ConstraintBasePtr::numRows() const { verifyIndex(); return getData()->numRows[calculateIndex()]; }

    AGX_FORCE_INLINE agx::UInt& ConstraintBasePtr::jacobianIndex() { verifyIndex(); return getData()->jacobianIndex[calculateIndex()]; }
    AGX_FORCE_INLINE agx::UInt const& ConstraintBasePtr::jacobianIndex() const { verifyIndex(); return getData()->jacobianIndex[calculateIndex()]; }

    AGX_FORCE_INLINE agx::UInt& ConstraintBasePtr::blockRowIndex() { verifyIndex(); return getData()->blockRowIndex[calculateIndex()]; }
    AGX_FORCE_INLINE agx::UInt const& ConstraintBasePtr::blockRowIndex() const { verifyIndex(); return getData()->blockRowIndex[calculateIndex()]; }

    AGX_FORCE_INLINE agx::Bool& ConstraintBasePtr::isImpacting() { verifyIndex(); return getData()->isImpacting[calculateIndex()]; }
    AGX_FORCE_INLINE agx::Bool const& ConstraintBasePtr::isImpacting() const { verifyIndex(); return getData()->isImpacting[calculateIndex()]; }

    //-----------------------------------------------------------------------------------------------------
    AGX_FORCE_INLINE ConstraintBaseInstance::ConstraintBaseInstance() {}
    AGX_FORCE_INLINE ConstraintBaseInstance::ConstraintBaseInstance(ConstraintBaseData* data, agx::Index index) : agx::Physics::InteractionInstance(data, index) {}
    AGX_FORCE_INLINE ConstraintBaseInstance::ConstraintBaseInstance(agxData::EntityStorage* storage, agx::Index index) : agx::Physics::InteractionInstance(storage, index) {}
    AGX_FORCE_INLINE ConstraintBaseInstance::ConstraintBaseInstance(const agxData::EntityInstance& other) : agx::Physics::InteractionInstance(other)
    {
      agxAssertN(!other || other.isInstanceOf(ConstraintBaseModel::instance()),
        "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
        EntityInstance::getModel()->fullPath().c_str(), ConstraintBaseModel::instance()->fullPath().c_str());
    }

    AGX_FORCE_INLINE ConstraintBaseInstance::ConstraintBaseInstance(const agxData::EntityPtr& ptr) : agx::Physics::InteractionInstance(ptr)
    {
      agxAssertN(!ptr || ptr.isInstanceOf(ConstraintBaseModel::instance()),
        "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
        EntityInstance::getModel()->fullPath().c_str(), ConstraintBaseModel::instance()->fullPath().c_str());
    }


    AGX_FORCE_INLINE ConstraintBaseData* ConstraintBaseInstance::getData() { return static_cast<ConstraintBaseData* >(agxData::EntityInstance::getData()); }
    AGX_FORCE_INLINE const ConstraintBaseData* ConstraintBaseInstance::getData() const { return static_cast<const ConstraintBaseData* >(agxData::EntityInstance::getData()); }

    AGX_FORCE_INLINE agx::ConstraintImplementation*& ConstraintBaseInstance::model() { verifyIndex(); return getData()->model[getIndex()]; }
    AGX_FORCE_INLINE agx::ConstraintImplementation* const& ConstraintBaseInstance::model() const { verifyIndex(); return getData()->model[getIndex()]; }

    AGX_FORCE_INLINE agx::UInt& ConstraintBaseInstance::rowIndex() { verifyIndex(); return getData()->rowIndex[getIndex()]; }
    AGX_FORCE_INLINE agx::UInt const& ConstraintBaseInstance::rowIndex() const { verifyIndex(); return getData()->rowIndex[getIndex()]; }

    AGX_FORCE_INLINE agx::UInt& ConstraintBaseInstance::numRows() { verifyIndex(); return getData()->numRows[getIndex()]; }
    AGX_FORCE_INLINE agx::UInt const& ConstraintBaseInstance::numRows() const { verifyIndex(); return getData()->numRows[getIndex()]; }

    AGX_FORCE_INLINE agx::UInt& ConstraintBaseInstance::jacobianIndex() { verifyIndex(); return getData()->jacobianIndex[getIndex()]; }
    AGX_FORCE_INLINE agx::UInt const& ConstraintBaseInstance::jacobianIndex() const { verifyIndex(); return getData()->jacobianIndex[getIndex()]; }

    AGX_FORCE_INLINE agx::UInt& ConstraintBaseInstance::blockRowIndex() { verifyIndex(); return getData()->blockRowIndex[getIndex()]; }
    AGX_FORCE_INLINE agx::UInt const& ConstraintBaseInstance::blockRowIndex() const { verifyIndex(); return getData()->blockRowIndex[getIndex()]; }

    AGX_FORCE_INLINE agx::Bool& ConstraintBaseInstance::isImpacting() { verifyIndex(); return getData()->isImpacting[getIndex()]; }
    AGX_FORCE_INLINE agx::Bool const& ConstraintBaseInstance::isImpacting() const { verifyIndex(); return getData()->isImpacting[getIndex()]; }

    //-----------------------------------------------------------------------------------------------------
    AGX_FORCE_INLINE ConstraintBaseSemantics::ConstraintBaseSemantics() {}
    //-----------------------------------------------------------------------------------------------------
    DOXYGEN_END_INTERNAL_BLOCK()
  }
}

AGX_TYPE_BINDING(agx::Physics::ConstraintBasePtr, "Physics.ConstraintBasePtr")
AGX_TYPE_BINDING(agx::Physics::ConstraintBaseInstance, "Physics.ConstraintBaseInstance")

#ifdef _MSC_VER
# pragma warning(pop)
#endif

#undef AGX_ENTITY_WRAPPER
#undef AGX_ENTITY_NAMESPACE
#endif

