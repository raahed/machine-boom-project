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

#ifndef GENERATED_AGX_PHYSICS_SOLVEISLAND_H_PLUGIN
#define GENERATED_AGX_PHYSICS_SOLVEISLAND_H_PLUGIN

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
#include <agx/Physics/InteractionGroupEntity.h>
#include <agx/Integer.h>
#include <agx/Physics/SolveGroupEntity.h>

namespace agx { namespace Physics { class SolveGroupPtr; }}

namespace agx
{
  namespace Physics
  {

    class SolveIslandModel;
    class SolveIslandData;
    class SolveIslandPtr;
    class SolveIslandInstance;
    class SolveIslandSemantics;


    AGX_DECLARE_POINTER_TYPES(SolveIslandModel);

    /** 
    Abstract description of the data attributes for the Physics.SolveIsland entity.
    */ 
    class AGXPHYSICS_EXPORT SolveIslandModel : public agx::Physics::InteractionGroupModel
    {
    public:
      typedef SolveIslandPtr PtrT;

      SolveIslandModel(const agx::String& name = "SolveIsland");

      /// \return The entity model singleton.
      static SolveIslandModel* instance();

      /// Create and return a pointer to a new instance in the default storage for this entity model.
      static SolveIslandPtr createInstance();

      /// \return The default storage for this entity model.
      static agxData::EntityStorage* defaultStorage();

      /// This is part of internal cleanup and should not be called by users
      virtual void shutdownCleanup() override;



      /* Attributes */
      static agxData::ScalarAttributeT< agx::Bool >* isImpactingAttribute;
      static agxData::ScalarAttributeT< agx::Bool >* iterativeFrictionRefinementAttribute;
      static agxData::ScalarAttributeT< agx::UInt >* directSolveCostEstimateAttribute;
      static agxData::ScalarAttributeT< agx::UInt >* currentIterationAttribute;
      static agxData::ArrayAttributeT< agx::Physics::SolveGroupPtr >* groupsAttribute;
      static agxData::ArrayAttributeT< agx::UInt32 >* directBodiesAttribute;

    protected:
      virtual ~SolveIslandModel();
      virtual agxData::EntityData* createData(agxData::EntityStorage* storage) override;
      virtual void configure(agx::TiXmlElement* eEntity) override;
      virtual void initAttributeAccessors() override;
      virtual void construct(agxData::EntityPtr instance) override;
      void construct(agx::Physics::SolveIslandPtr solveIsland);
    };


    DOXYGEN_START_INTERNAL_BLOCK()
    #ifndef AGX_PHYSICS_SOLVEISLAND_DATA_SET_OVERRIDE
    #define AGX_PHYSICS_SOLVEISLAND_DATA_SET
    class AGXPHYSICS_EXPORT SolveIslandData : public agx::Physics::InteractionGroupData
    {
    public:
      SolveIslandInstance operator[] (size_t index);

    public:
      agxData::Array< SolveIslandPtr >& instance;
      agxData::Array< agx::Bool > isImpacting;
      agxData::Array< agx::Bool > iterativeFrictionRefinement;
      agxData::Array< agx::UInt > directSolveCostEstimate;
      agxData::Array< agx::UInt > currentIteration;
      agxData::Array< agxData::Array< agx::Physics::SolveGroupPtr > > groups;
      agxData::Array< agxData::Array< agx::UInt32 > > directBodies;

    public:
      typedef agx::Bool isImpactingType;
      typedef agx::Bool iterativeFrictionRefinementType;
      typedef agx::UInt directSolveCostEstimateType;
      typedef agx::UInt currentIterationType;
      typedef agxData::Array< agx::Physics::SolveGroupPtr > groupsType;
      typedef agxData::Array< agx::UInt32 > directBodiesType;

    public:
      SolveIslandData(agxData::EntityStorage* storage);
      SolveIslandData();

    protected:
      virtual ~SolveIslandData() {}
      virtual void setNumElements(agx::Index numElements) override;

    private:
      SolveIslandData& operator= (const SolveIslandData&) { return *this; }

    };
    #endif
    DOXYGEN_END_INTERNAL_BLOCK()


    DOXYGEN_START_INTERNAL_BLOCK()
    class AGXPHYSICS_EXPORT SolveIslandSemantics : public agx::Physics::InteractionGroupSemantics
    {
    public:

      // Automatic getters
      agx::Bool const& getIsImpacting() const;
      agx::Bool const& getIterativeFrictionRefinement() const;
      agx::UInt const& getDirectSolveCostEstimate() const;
      agx::UInt const& getCurrentIteration() const;
      agxData::Array< agx::Physics::SolveGroupPtr > const& getGroups() const;
      agxData::Array< agx::UInt32 > const& getDirectBodies() const;

      // Semantics defined by explicit kernels

      // Automatic setters
      void setIsImpacting(agx::Bool const& value);
      void setIterativeFrictionRefinement(agx::Bool const& value);
      void setDirectSolveCostEstimate(agx::UInt const& value);
      void setCurrentIteration(agx::UInt const& value);
      void setGroups(agxData::Array< agx::Physics::SolveGroupPtr > const& value);
      void setDirectBodies(agxData::Array< agx::UInt32 > const& value);


    protected:
      friend class SolveIslandPtr;
      friend class SolveIslandInstance;
      SolveIslandSemantics();
    };
    DOXYGEN_END_INTERNAL_BLOCK()


    /**
    Pointer to a entity instance of type Physics.SolveIsland
    */
    class CALLABLE SolveIslandPtr : public agx::Physics::InteractionGroupPtr
    {
    public:
      typedef SolveIslandModel ModelType;
      typedef SolveIslandData DataType;
      typedef SolveIslandInstance InstanceType;

    public:
      AGXPHYSICS_EXPORT SolveIslandPtr();
      AGXPHYSICS_EXPORT SolveIslandPtr(agxData::EntityStorage* storage, agx::Index id);
      AGXPHYSICS_EXPORT SolveIslandPtr(const agxData::EntityPtr& ptr);
      AGXPHYSICS_EXPORT SolveIslandPtr(const agxData::EntityInstance& instance);
      AGXPHYSICS_EXPORT SolveIslandPtr& operator= (const agxData::EntityPtr& ptr);
      AGXPHYSICS_EXPORT SolveIslandPtr& operator= (const agxData::EntityInstance& instance);
      AGXPHYSICS_EXPORT SolveIslandInstance instance();
      AGXPHYSICS_EXPORT const SolveIslandInstance instance() const;

      AGXPHYSICS_EXPORT SolveIslandSemantics* operator->();
      AGXPHYSICS_EXPORT const SolveIslandSemantics* operator->() const;

      SolveIslandData* getData();
      const SolveIslandData* getData() const;


      /// \return reference to the isImpacting attribute
      AGXPHYSICS_EXPORT agx::Bool& isImpacting();
      /// \return const reference to the isImpacting attribute
      AGXPHYSICS_EXPORT agx::Bool const& isImpacting() const;

      /// \return reference to the iterativeFrictionRefinement attribute
      AGXPHYSICS_EXPORT agx::Bool& iterativeFrictionRefinement();
      /// \return const reference to the iterativeFrictionRefinement attribute
      AGXPHYSICS_EXPORT agx::Bool const& iterativeFrictionRefinement() const;

      /// \return reference to the directSolveCostEstimate attribute
      AGXPHYSICS_EXPORT agx::UInt& directSolveCostEstimate();
      /// \return const reference to the directSolveCostEstimate attribute
      AGXPHYSICS_EXPORT agx::UInt const& directSolveCostEstimate() const;

      /// \return reference to the currentIteration attribute
      AGXPHYSICS_EXPORT agx::UInt& currentIteration();
      /// \return const reference to the currentIteration attribute
      AGXPHYSICS_EXPORT agx::UInt const& currentIteration() const;

      /// \return reference to the groups attribute
      AGXPHYSICS_EXPORT agxData::Array< agx::Physics::SolveGroupPtr >& groups();
      /// \return const reference to the groups attribute
      AGXPHYSICS_EXPORT agxData::Array< agx::Physics::SolveGroupPtr > const& groups() const;

      /// \return reference to the directBodies attribute
      AGXPHYSICS_EXPORT agxData::Array< agx::UInt32 >& directBodies();
      /// \return const reference to the directBodies attribute
      AGXPHYSICS_EXPORT agxData::Array< agx::UInt32 > const& directBodies() const;

      DOXYGEN_START_INTERNAL_BLOCK()
      // Only used by tolua!
      #ifdef AGX_GENERATED_TOLUA_WRAPPER
      inline agx::Bool const& getIsImpacting() const { return (*this)->getIsImpacting(); }
      inline agx::Bool const& getIterativeFrictionRefinement() const { return (*this)->getIterativeFrictionRefinement(); }
      inline agx::UInt const& getDirectSolveCostEstimate() const { return (*this)->getDirectSolveCostEstimate(); }
      inline agx::UInt const& getCurrentIteration() const { return (*this)->getCurrentIteration(); }
      inline agxData::Array< agx::Physics::SolveGroupPtr > const& getGroups() const { return (*this)->getGroups(); }
      inline agxData::Array< agx::UInt32 > const& getDirectBodies() const { return (*this)->getDirectBodies(); }


      inline void setIsImpacting(agx::Bool const& value) { (*this)->setIsImpacting(value); }
      inline void setIterativeFrictionRefinement(agx::Bool const& value) { (*this)->setIterativeFrictionRefinement(value); }
      inline void setDirectSolveCostEstimate(agx::UInt const& value) { (*this)->setDirectSolveCostEstimate(value); }
      inline void setCurrentIteration(agx::UInt const& value) { (*this)->setCurrentIteration(value); }
      inline void setGroups(agxData::Array< agx::Physics::SolveGroupPtr > const& value) { (*this)->setGroups(value); }
      inline void setDirectBodies(agxData::Array< agx::UInt32 > const& value) { (*this)->setDirectBodies(value); }
      #endif
      DOXYGEN_END_INTERNAL_BLOCK()

    };


    DOXYGEN_START_INTERNAL_BLOCK()
    class AGXPHYSICS_EXPORT SolveIslandInstance : public agx::Physics::InteractionGroupInstance
    {
    public:
      SolveIslandInstance();
      SolveIslandInstance(SolveIslandData* data, agx::Index index);
      SolveIslandInstance(agxData::EntityStorage *storage, agx::Index index);
      SolveIslandInstance(const agxData::EntityInstance& other);
      SolveIslandInstance(const agxData::EntityPtr& ptr);

      SolveIslandData* getData();
      const SolveIslandData* getData() const;

    public:
      /// \return reference to the isImpacting attribute
      agx::Bool& isImpacting();
      /// \return const reference to the isImpacting attribute
      agx::Bool const& isImpacting() const;

      /// \return reference to the iterativeFrictionRefinement attribute
      agx::Bool& iterativeFrictionRefinement();
      /// \return const reference to the iterativeFrictionRefinement attribute
      agx::Bool const& iterativeFrictionRefinement() const;

      /// \return reference to the directSolveCostEstimate attribute
      agx::UInt& directSolveCostEstimate();
      /// \return const reference to the directSolveCostEstimate attribute
      agx::UInt const& directSolveCostEstimate() const;

      /// \return reference to the currentIteration attribute
      agx::UInt& currentIteration();
      /// \return const reference to the currentIteration attribute
      agx::UInt const& currentIteration() const;

      /// \return reference to the groups attribute
      agxData::Array< agx::Physics::SolveGroupPtr >& groups();
      /// \return const reference to the groups attribute
      agxData::Array< agx::Physics::SolveGroupPtr > const& groups() const;

      /// \return reference to the directBodies attribute
      agxData::Array< agx::UInt32 >& directBodies();
      /// \return const reference to the directBodies attribute
      agxData::Array< agx::UInt32 > const& directBodies() const;

    };
    DOXYGEN_END_INTERNAL_BLOCK()



    typedef agx::VectorPOD<SolveIslandPtr> SolveIslandPtrVector;
    typedef agxData::Array<SolveIslandPtr> SolveIslandPtrArray;



    DOXYGEN_START_INTERNAL_BLOCK()
    /* Implementation */
    //-----------------------------------------------------------------------------------------------------
    //-----------------------------------------------------------------------------------------------------
    inline SolveIslandInstance agx::Physics::SolveIslandData::operator[] (size_t index) { return SolveIslandInstance(this, (agx::Index)index); }
    //-----------------------------------------------------------------------------------------------------
    AGX_FORCE_INLINE SolveIslandPtr::SolveIslandPtr() {}
    AGX_FORCE_INLINE SolveIslandPtr::SolveIslandPtr(agxData::EntityStorage* storage, agx::Index id) : agx::Physics::InteractionGroupPtr(storage, id) {}
    AGX_FORCE_INLINE SolveIslandPtr::SolveIslandPtr(const agxData::EntityPtr& ptr) : agx::Physics::InteractionGroupPtr(ptr)
    {
      agxAssertN(!ptr || ptr.isInstanceOf(SolveIslandModel::instance()),
        "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
        EntityPtr::getModel()->fullPath().c_str(), SolveIslandModel::instance()->fullPath().c_str());
    }

    AGX_FORCE_INLINE SolveIslandPtr::SolveIslandPtr(const agxData::EntityInstance& instance) : agx::Physics::InteractionGroupPtr(instance)
    {
      agxAssertN(!instance || instance.isInstanceOf(SolveIslandModel::instance()),
        "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
        EntityPtr::getModel()->fullPath().c_str(), SolveIslandModel::instance()->fullPath().c_str());
    }

    AGX_FORCE_INLINE SolveIslandPtr& SolveIslandPtr::operator= (const agxData::EntityPtr& ptr)
    {
      agxData::EntityPtr::operator= (ptr);
      agxAssertN(!ptr || ptr.isInstanceOf(SolveIslandModel::instance()),
        "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
        EntityPtr::getModel()->fullPath().c_str(), SolveIslandModel::instance()->fullPath().c_str());
      return *this;
    }

    AGX_FORCE_INLINE SolveIslandPtr& SolveIslandPtr::operator= (const agxData::EntityInstance& instance)
    {
      agxData::EntityPtr::operator= (instance);
      agxAssertN(!instance || instance.isInstanceOf(SolveIslandModel::instance()),
        "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
        EntityPtr::getModel()->fullPath().c_str(), SolveIslandModel::instance()->fullPath().c_str());
      return *this;
    }

    inline SolveIslandInstance SolveIslandPtr::instance() { return agxData::EntityPtr::instance(); }
    inline const SolveIslandInstance SolveIslandPtr::instance() const { return agxData::EntityPtr::instance(); }
    AGX_FORCE_INLINE SolveIslandSemantics* SolveIslandPtr::operator->() { return (SolveIslandSemantics* )this; }
    AGX_FORCE_INLINE const SolveIslandSemantics* SolveIslandPtr::operator->() const { return (const SolveIslandSemantics* )this; }
    AGX_FORCE_INLINE SolveIslandData* SolveIslandPtr::getData() { return static_cast<SolveIslandData* >(agxData::EntityPtr::getData()); }
    AGX_FORCE_INLINE const SolveIslandData* SolveIslandPtr::getData() const { return static_cast<const SolveIslandData* >(agxData::EntityPtr::getData()); }

    AGX_FORCE_INLINE agx::Bool& SolveIslandPtr::isImpacting() { verifyIndex(); return getData()->isImpacting[calculateIndex()]; }
    AGX_FORCE_INLINE agx::Bool const& SolveIslandPtr::isImpacting() const { verifyIndex(); return getData()->isImpacting[calculateIndex()]; }

    AGX_FORCE_INLINE agx::Bool& SolveIslandPtr::iterativeFrictionRefinement() { verifyIndex(); return getData()->iterativeFrictionRefinement[calculateIndex()]; }
    AGX_FORCE_INLINE agx::Bool const& SolveIslandPtr::iterativeFrictionRefinement() const { verifyIndex(); return getData()->iterativeFrictionRefinement[calculateIndex()]; }

    AGX_FORCE_INLINE agx::UInt& SolveIslandPtr::directSolveCostEstimate() { verifyIndex(); return getData()->directSolveCostEstimate[calculateIndex()]; }
    AGX_FORCE_INLINE agx::UInt const& SolveIslandPtr::directSolveCostEstimate() const { verifyIndex(); return getData()->directSolveCostEstimate[calculateIndex()]; }

    AGX_FORCE_INLINE agx::UInt& SolveIslandPtr::currentIteration() { verifyIndex(); return getData()->currentIteration[calculateIndex()]; }
    AGX_FORCE_INLINE agx::UInt const& SolveIslandPtr::currentIteration() const { verifyIndex(); return getData()->currentIteration[calculateIndex()]; }

    AGX_FORCE_INLINE agxData::Array< agx::Physics::SolveGroupPtr >& SolveIslandPtr::groups() { verifyIndex(); return getData()->groups[calculateIndex()]; }
    AGX_FORCE_INLINE agxData::Array< agx::Physics::SolveGroupPtr > const& SolveIslandPtr::groups() const { verifyIndex(); return getData()->groups[calculateIndex()]; }

    AGX_FORCE_INLINE agxData::Array< agx::UInt32 >& SolveIslandPtr::directBodies() { verifyIndex(); return getData()->directBodies[calculateIndex()]; }
    AGX_FORCE_INLINE agxData::Array< agx::UInt32 > const& SolveIslandPtr::directBodies() const { verifyIndex(); return getData()->directBodies[calculateIndex()]; }

    //-----------------------------------------------------------------------------------------------------
    AGX_FORCE_INLINE SolveIslandInstance::SolveIslandInstance() {}
    AGX_FORCE_INLINE SolveIslandInstance::SolveIslandInstance(SolveIslandData* data, agx::Index index) : agx::Physics::InteractionGroupInstance(data, index) {}
    AGX_FORCE_INLINE SolveIslandInstance::SolveIslandInstance(agxData::EntityStorage* storage, agx::Index index) : agx::Physics::InteractionGroupInstance(storage, index) {}
    AGX_FORCE_INLINE SolveIslandInstance::SolveIslandInstance(const agxData::EntityInstance& other) : agx::Physics::InteractionGroupInstance(other)
    {
      agxAssertN(!other || other.isInstanceOf(SolveIslandModel::instance()),
        "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
        EntityInstance::getModel()->fullPath().c_str(), SolveIslandModel::instance()->fullPath().c_str());
    }

    AGX_FORCE_INLINE SolveIslandInstance::SolveIslandInstance(const agxData::EntityPtr& ptr) : agx::Physics::InteractionGroupInstance(ptr)
    {
      agxAssertN(!ptr || ptr.isInstanceOf(SolveIslandModel::instance()),
        "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
        EntityInstance::getModel()->fullPath().c_str(), SolveIslandModel::instance()->fullPath().c_str());
    }


    AGX_FORCE_INLINE SolveIslandData* SolveIslandInstance::getData() { return static_cast<SolveIslandData* >(agxData::EntityInstance::getData()); }
    AGX_FORCE_INLINE const SolveIslandData* SolveIslandInstance::getData() const { return static_cast<const SolveIslandData* >(agxData::EntityInstance::getData()); }

    AGX_FORCE_INLINE agx::Bool& SolveIslandInstance::isImpacting() { verifyIndex(); return getData()->isImpacting[getIndex()]; }
    AGX_FORCE_INLINE agx::Bool const& SolveIslandInstance::isImpacting() const { verifyIndex(); return getData()->isImpacting[getIndex()]; }

    AGX_FORCE_INLINE agx::Bool& SolveIslandInstance::iterativeFrictionRefinement() { verifyIndex(); return getData()->iterativeFrictionRefinement[getIndex()]; }
    AGX_FORCE_INLINE agx::Bool const& SolveIslandInstance::iterativeFrictionRefinement() const { verifyIndex(); return getData()->iterativeFrictionRefinement[getIndex()]; }

    AGX_FORCE_INLINE agx::UInt& SolveIslandInstance::directSolveCostEstimate() { verifyIndex(); return getData()->directSolveCostEstimate[getIndex()]; }
    AGX_FORCE_INLINE agx::UInt const& SolveIslandInstance::directSolveCostEstimate() const { verifyIndex(); return getData()->directSolveCostEstimate[getIndex()]; }

    AGX_FORCE_INLINE agx::UInt& SolveIslandInstance::currentIteration() { verifyIndex(); return getData()->currentIteration[getIndex()]; }
    AGX_FORCE_INLINE agx::UInt const& SolveIslandInstance::currentIteration() const { verifyIndex(); return getData()->currentIteration[getIndex()]; }

    AGX_FORCE_INLINE agxData::Array< agx::Physics::SolveGroupPtr >& SolveIslandInstance::groups() { verifyIndex(); return getData()->groups[getIndex()]; }
    AGX_FORCE_INLINE agxData::Array< agx::Physics::SolveGroupPtr > const& SolveIslandInstance::groups() const { verifyIndex(); return getData()->groups[getIndex()]; }

    AGX_FORCE_INLINE agxData::Array< agx::UInt32 >& SolveIslandInstance::directBodies() { verifyIndex(); return getData()->directBodies[getIndex()]; }
    AGX_FORCE_INLINE agxData::Array< agx::UInt32 > const& SolveIslandInstance::directBodies() const { verifyIndex(); return getData()->directBodies[getIndex()]; }

    //-----------------------------------------------------------------------------------------------------
    AGX_FORCE_INLINE SolveIslandSemantics::SolveIslandSemantics() {}
    //-----------------------------------------------------------------------------------------------------
    DOXYGEN_END_INTERNAL_BLOCK()
  }
}

AGX_TYPE_BINDING(agx::Physics::SolveIslandPtr, "Physics.SolveIslandPtr")
AGX_TYPE_BINDING(agx::Physics::SolveIslandInstance, "Physics.SolveIslandInstance")

#ifdef _MSC_VER
# pragma warning(pop)
#endif

#undef AGX_ENTITY_WRAPPER
#undef AGX_ENTITY_NAMESPACE
#endif

