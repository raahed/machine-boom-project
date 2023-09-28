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

#ifndef GENERATED_AGX_PHYSICS_CONSTRAINTROW_H_PLUGIN
#define GENERATED_AGX_PHYSICS_CONSTRAINTROW_H_PLUGIN

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
#include <agx/Integer.h>
#include <agx/Real.h>
#include <agx/Range.h>


namespace agx
{
  namespace Physics
  {

    class ConstraintRowModel;
    class ConstraintRowData;
    class ConstraintRowPtr;
    class ConstraintRowInstance;
    class ConstraintRowSemantics;


    AGX_DECLARE_POINTER_TYPES(ConstraintRowModel);

    /** 
    Abstract description of the data attributes for the Physics.ConstraintRow entity.
    */ 
    class AGXPHYSICS_EXPORT ConstraintRowModel : public agxData::EntityModel
    {
    public:
      typedef ConstraintRowPtr PtrT;

      ConstraintRowModel(const agx::String& name = "ConstraintRow");

      /// \return The entity model singleton.
      static ConstraintRowModel* instance();

      /// Create and return a pointer to a new instance in the default storage for this entity model.
      static ConstraintRowPtr createInstance();

      /// \return The default storage for this entity model.
      static agxData::EntityStorage* defaultStorage();

      /// This is part of internal cleanup and should not be called by users
      virtual void shutdownCleanup() override;



      /* Attributes */
      static agxData::ScalarAttributeT< agx::Bool >* holonomicAttribute;
      static agxData::ScalarAttributeT< agx::Real >* dampingAttribute;
      static agxData::ScalarAttributeT< agx::Real >* epsilonAttribute;
      static agxData::ScalarAttributeT< agx::Real >* lambdaAttribute;
      static agxData::ScalarAttributeT< agx::Real >* totalLambdaAttribute;
      static agxData::ScalarAttributeT< agx::Real >* rhsAttribute;
      static agxData::ScalarAttributeT< agx::Real >* invDAttribute;
      static agxData::ScalarAttributeT< agx::Real >* impactSpeedAttribute;
      static agxData::ScalarAttributeT< agx::Real >* velocityAttribute;
      static agxData::ScalarAttributeT< agx::Real >* violationAttribute;
      static agxData::ScalarAttributeT< agx::RangeReal >* boundAttribute;
      static agxData::ScalarAttributeT< agx::Real >* residualAttribute;
      static agxData::ScalarAttributeT< agx::Int8 >* indexSetStateAttribute;

    protected:
      virtual ~ConstraintRowModel();
      virtual agxData::EntityData* createData(agxData::EntityStorage* storage) override;
      virtual void configure(agx::TiXmlElement* eEntity) override;
      virtual void initAttributeAccessors() override;
      virtual void construct(agxData::EntityPtr instance) override;
      void construct(agx::Physics::ConstraintRowPtr constraintRow);
    };


    DOXYGEN_START_INTERNAL_BLOCK()
    #ifndef AGX_PHYSICS_CONSTRAINTROW_DATA_SET_OVERRIDE
    #define AGX_PHYSICS_CONSTRAINTROW_DATA_SET
    class AGXPHYSICS_EXPORT ConstraintRowData : public agxData::EntityData
    {
    public:
      ConstraintRowInstance operator[] (size_t index);

    public:
      agxData::Array< ConstraintRowPtr >& instance;
      agxData::Array< agx::Bool > holonomic;
      agxData::Array< agx::Real > damping;
      agxData::Array< agx::Real > epsilon;
      agxData::Array< agx::Real > lambda;
      agxData::Array< agx::Real > totalLambda;
      agxData::Array< agx::Real > rhs;
      agxData::Array< agx::Real > invD;
      agxData::Array< agx::Real > impactSpeed;
      agxData::Array< agx::Real > velocity;
      agxData::Array< agx::Real > violation;
      agxData::Array< agx::RangeReal > bound;
      agxData::Array< agx::Real > residual;
      agxData::Array< agx::Int8 > indexSetState;

    public:
      typedef agx::Bool holonomicType;
      typedef agx::Real dampingType;
      typedef agx::Real epsilonType;
      typedef agx::Real lambdaType;
      typedef agx::Real totalLambdaType;
      typedef agx::Real rhsType;
      typedef agx::Real invDType;
      typedef agx::Real impactSpeedType;
      typedef agx::Real velocityType;
      typedef agx::Real violationType;
      typedef agx::RangeReal boundType;
      typedef agx::Real residualType;
      typedef agx::Int8 indexSetStateType;

    public:
      ConstraintRowData(agxData::EntityStorage* storage);
      ConstraintRowData();

    protected:
      virtual ~ConstraintRowData() {}
      virtual void setNumElements(agx::Index numElements) override;

    private:
      ConstraintRowData& operator= (const ConstraintRowData&) { return *this; }

    };
    #endif
    DOXYGEN_END_INTERNAL_BLOCK()


    DOXYGEN_START_INTERNAL_BLOCK()
    class AGXPHYSICS_EXPORT ConstraintRowSemantics : protected agxData::EntityPtr
    {
    public:

      // Automatic getters
      agx::Bool const& getHolonomic() const;
      agx::Real const& getDamping() const;
      agx::Real const& getEpsilon() const;
      agx::Real const& getLambda() const;
      agx::Real const& getTotalLambda() const;
      agx::Real const& getRhs() const;
      agx::Real const& getInvD() const;
      agx::Real const& getImpactSpeed() const;
      agx::Real const& getVelocity() const;
      agx::Real const& getViolation() const;
      agx::RangeReal const& getBound() const;
      agx::Real const& getResidual() const;
      agx::Int8 const& getIndexSetState() const;

      // Semantics defined by explicit kernels

      // Automatic setters
      void setHolonomic(agx::Bool const& value);
      void setDamping(agx::Real const& value);
      void setEpsilon(agx::Real const& value);
      void setLambda(agx::Real const& value);
      void setTotalLambda(agx::Real const& value);
      void setRhs(agx::Real const& value);
      void setInvD(agx::Real const& value);
      void setImpactSpeed(agx::Real const& value);
      void setVelocity(agx::Real const& value);
      void setViolation(agx::Real const& value);
      void setBound(agx::RangeReal const& value);
      void setResidual(agx::Real const& value);
      void setIndexSetState(agx::Int8 const& value);


    protected:
      friend class ConstraintRowPtr;
      friend class ConstraintRowInstance;
      ConstraintRowSemantics();
    };
    DOXYGEN_END_INTERNAL_BLOCK()


    /**
    Pointer to a entity instance of type Physics.ConstraintRow
    */
    class CALLABLE ConstraintRowPtr : public agxData::EntityPtr
    {
    public:
      typedef ConstraintRowModel ModelType;
      typedef ConstraintRowData DataType;
      typedef ConstraintRowInstance InstanceType;

    public:
      AGXPHYSICS_EXPORT ConstraintRowPtr();
      AGXPHYSICS_EXPORT ConstraintRowPtr(agxData::EntityStorage* storage, agx::Index id);
      AGXPHYSICS_EXPORT ConstraintRowPtr(const agxData::EntityPtr& ptr);
      AGXPHYSICS_EXPORT ConstraintRowPtr(const agxData::EntityInstance& instance);
      AGXPHYSICS_EXPORT ConstraintRowPtr& operator= (const agxData::EntityPtr& ptr);
      AGXPHYSICS_EXPORT ConstraintRowPtr& operator= (const agxData::EntityInstance& instance);
      AGXPHYSICS_EXPORT ConstraintRowInstance instance();
      AGXPHYSICS_EXPORT const ConstraintRowInstance instance() const;

      AGXPHYSICS_EXPORT ConstraintRowSemantics* operator->();
      AGXPHYSICS_EXPORT const ConstraintRowSemantics* operator->() const;

      ConstraintRowData* getData();
      const ConstraintRowData* getData() const;


      /// \return reference to the holonomic attribute
      AGXPHYSICS_EXPORT agx::Bool& holonomic();
      /// \return const reference to the holonomic attribute
      AGXPHYSICS_EXPORT agx::Bool const& holonomic() const;

      /// \return reference to the damping attribute
      AGXPHYSICS_EXPORT agx::Real& damping();
      /// \return const reference to the damping attribute
      AGXPHYSICS_EXPORT agx::Real const& damping() const;

      /// \return reference to the epsilon attribute
      AGXPHYSICS_EXPORT agx::Real& epsilon();
      /// \return const reference to the epsilon attribute
      AGXPHYSICS_EXPORT agx::Real const& epsilon() const;

      /// \return reference to the lambda attribute
      AGXPHYSICS_EXPORT agx::Real& lambda();
      /// \return const reference to the lambda attribute
      AGXPHYSICS_EXPORT agx::Real const& lambda() const;

      /// \return reference to the totalLambda attribute
      AGXPHYSICS_EXPORT agx::Real& totalLambda();
      /// \return const reference to the totalLambda attribute
      AGXPHYSICS_EXPORT agx::Real const& totalLambda() const;

      /// \return reference to the rhs attribute
      AGXPHYSICS_EXPORT agx::Real& rhs();
      /// \return const reference to the rhs attribute
      AGXPHYSICS_EXPORT agx::Real const& rhs() const;

      /// \return reference to the invD attribute
      AGXPHYSICS_EXPORT agx::Real& invD();
      /// \return const reference to the invD attribute
      AGXPHYSICS_EXPORT agx::Real const& invD() const;

      /// \return reference to the impactSpeed attribute
      AGXPHYSICS_EXPORT agx::Real& impactSpeed();
      /// \return const reference to the impactSpeed attribute
      AGXPHYSICS_EXPORT agx::Real const& impactSpeed() const;

      /// \return reference to the velocity attribute
      AGXPHYSICS_EXPORT agx::Real& velocity();
      /// \return const reference to the velocity attribute
      AGXPHYSICS_EXPORT agx::Real const& velocity() const;

      /// \return reference to the violation attribute
      AGXPHYSICS_EXPORT agx::Real& violation();
      /// \return const reference to the violation attribute
      AGXPHYSICS_EXPORT agx::Real const& violation() const;

      /// \return reference to the bound attribute
      AGXPHYSICS_EXPORT agx::RangeReal& bound();
      /// \return const reference to the bound attribute
      AGXPHYSICS_EXPORT agx::RangeReal const& bound() const;

      /// \return reference to the residual attribute
      AGXPHYSICS_EXPORT agx::Real& residual();
      /// \return const reference to the residual attribute
      AGXPHYSICS_EXPORT agx::Real const& residual() const;

      /// \return reference to the indexSetState attribute
      AGXPHYSICS_EXPORT agx::Int8& indexSetState();
      /// \return const reference to the indexSetState attribute
      AGXPHYSICS_EXPORT agx::Int8 const& indexSetState() const;

      DOXYGEN_START_INTERNAL_BLOCK()
      // Only used by tolua!
      #ifdef AGX_GENERATED_TOLUA_WRAPPER
      inline agx::Bool const& getHolonomic() const { return (*this)->getHolonomic(); }
      inline agx::Real const& getDamping() const { return (*this)->getDamping(); }
      inline agx::Real const& getEpsilon() const { return (*this)->getEpsilon(); }
      inline agx::Real const& getLambda() const { return (*this)->getLambda(); }
      inline agx::Real const& getTotalLambda() const { return (*this)->getTotalLambda(); }
      inline agx::Real const& getRhs() const { return (*this)->getRhs(); }
      inline agx::Real const& getInvD() const { return (*this)->getInvD(); }
      inline agx::Real const& getImpactSpeed() const { return (*this)->getImpactSpeed(); }
      inline agx::Real const& getVelocity() const { return (*this)->getVelocity(); }
      inline agx::Real const& getViolation() const { return (*this)->getViolation(); }
      inline agx::RangeReal const& getBound() const { return (*this)->getBound(); }
      inline agx::Real const& getResidual() const { return (*this)->getResidual(); }
      inline agx::Int8 const& getIndexSetState() const { return (*this)->getIndexSetState(); }


      inline void setHolonomic(agx::Bool const& value) { (*this)->setHolonomic(value); }
      inline void setDamping(agx::Real const& value) { (*this)->setDamping(value); }
      inline void setEpsilon(agx::Real const& value) { (*this)->setEpsilon(value); }
      inline void setLambda(agx::Real const& value) { (*this)->setLambda(value); }
      inline void setTotalLambda(agx::Real const& value) { (*this)->setTotalLambda(value); }
      inline void setRhs(agx::Real const& value) { (*this)->setRhs(value); }
      inline void setInvD(agx::Real const& value) { (*this)->setInvD(value); }
      inline void setImpactSpeed(agx::Real const& value) { (*this)->setImpactSpeed(value); }
      inline void setVelocity(agx::Real const& value) { (*this)->setVelocity(value); }
      inline void setViolation(agx::Real const& value) { (*this)->setViolation(value); }
      inline void setBound(agx::RangeReal const& value) { (*this)->setBound(value); }
      inline void setResidual(agx::Real const& value) { (*this)->setResidual(value); }
      inline void setIndexSetState(agx::Int8 const& value) { (*this)->setIndexSetState(value); }
      #endif
      DOXYGEN_END_INTERNAL_BLOCK()

    };


    DOXYGEN_START_INTERNAL_BLOCK()
    class AGXPHYSICS_EXPORT ConstraintRowInstance : public agxData::EntityInstance
    {
    public:
      ConstraintRowInstance();
      ConstraintRowInstance(ConstraintRowData* data, agx::Index index);
      ConstraintRowInstance(agxData::EntityStorage *storage, agx::Index index);
      ConstraintRowInstance(const agxData::EntityInstance& other);
      ConstraintRowInstance(const agxData::EntityPtr& ptr);

      ConstraintRowData* getData();
      const ConstraintRowData* getData() const;

    public:
      /// \return reference to the holonomic attribute
      agx::Bool& holonomic();
      /// \return const reference to the holonomic attribute
      agx::Bool const& holonomic() const;

      /// \return reference to the damping attribute
      agx::Real& damping();
      /// \return const reference to the damping attribute
      agx::Real const& damping() const;

      /// \return reference to the epsilon attribute
      agx::Real& epsilon();
      /// \return const reference to the epsilon attribute
      agx::Real const& epsilon() const;

      /// \return reference to the lambda attribute
      agx::Real& lambda();
      /// \return const reference to the lambda attribute
      agx::Real const& lambda() const;

      /// \return reference to the totalLambda attribute
      agx::Real& totalLambda();
      /// \return const reference to the totalLambda attribute
      agx::Real const& totalLambda() const;

      /// \return reference to the rhs attribute
      agx::Real& rhs();
      /// \return const reference to the rhs attribute
      agx::Real const& rhs() const;

      /// \return reference to the invD attribute
      agx::Real& invD();
      /// \return const reference to the invD attribute
      agx::Real const& invD() const;

      /// \return reference to the impactSpeed attribute
      agx::Real& impactSpeed();
      /// \return const reference to the impactSpeed attribute
      agx::Real const& impactSpeed() const;

      /// \return reference to the velocity attribute
      agx::Real& velocity();
      /// \return const reference to the velocity attribute
      agx::Real const& velocity() const;

      /// \return reference to the violation attribute
      agx::Real& violation();
      /// \return const reference to the violation attribute
      agx::Real const& violation() const;

      /// \return reference to the bound attribute
      agx::RangeReal& bound();
      /// \return const reference to the bound attribute
      agx::RangeReal const& bound() const;

      /// \return reference to the residual attribute
      agx::Real& residual();
      /// \return const reference to the residual attribute
      agx::Real const& residual() const;

      /// \return reference to the indexSetState attribute
      agx::Int8& indexSetState();
      /// \return const reference to the indexSetState attribute
      agx::Int8 const& indexSetState() const;

    };
    DOXYGEN_END_INTERNAL_BLOCK()



    typedef agx::VectorPOD<ConstraintRowPtr> ConstraintRowPtrVector;
    typedef agxData::Array<ConstraintRowPtr> ConstraintRowPtrArray;



    DOXYGEN_START_INTERNAL_BLOCK()
    /* Implementation */
    //-----------------------------------------------------------------------------------------------------
    //-----------------------------------------------------------------------------------------------------
    inline ConstraintRowInstance agx::Physics::ConstraintRowData::operator[] (size_t index) { return ConstraintRowInstance(this, (agx::Index)index); }
    //-----------------------------------------------------------------------------------------------------
    AGX_FORCE_INLINE ConstraintRowPtr::ConstraintRowPtr() {}
    AGX_FORCE_INLINE ConstraintRowPtr::ConstraintRowPtr(agxData::EntityStorage* storage, agx::Index id) : agxData::EntityPtr(storage, id) {}
    AGX_FORCE_INLINE ConstraintRowPtr::ConstraintRowPtr(const agxData::EntityPtr& ptr) : agxData::EntityPtr(ptr)
    {
      agxAssertN(!ptr || ptr.isInstanceOf(ConstraintRowModel::instance()),
        "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
        EntityPtr::getModel()->fullPath().c_str(), ConstraintRowModel::instance()->fullPath().c_str());
    }

    AGX_FORCE_INLINE ConstraintRowPtr::ConstraintRowPtr(const agxData::EntityInstance& instance) : agxData::EntityPtr(instance)
    {
      agxAssertN(!instance || instance.isInstanceOf(ConstraintRowModel::instance()),
        "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
        EntityPtr::getModel()->fullPath().c_str(), ConstraintRowModel::instance()->fullPath().c_str());
    }

    AGX_FORCE_INLINE ConstraintRowPtr& ConstraintRowPtr::operator= (const agxData::EntityPtr& ptr)
    {
      agxData::EntityPtr::operator= (ptr);
      agxAssertN(!ptr || ptr.isInstanceOf(ConstraintRowModel::instance()),
        "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
        EntityPtr::getModel()->fullPath().c_str(), ConstraintRowModel::instance()->fullPath().c_str());
      return *this;
    }

    AGX_FORCE_INLINE ConstraintRowPtr& ConstraintRowPtr::operator= (const agxData::EntityInstance& instance)
    {
      agxData::EntityPtr::operator= (instance);
      agxAssertN(!instance || instance.isInstanceOf(ConstraintRowModel::instance()),
        "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
        EntityPtr::getModel()->fullPath().c_str(), ConstraintRowModel::instance()->fullPath().c_str());
      return *this;
    }

    inline ConstraintRowInstance ConstraintRowPtr::instance() { return agxData::EntityPtr::instance(); }
    inline const ConstraintRowInstance ConstraintRowPtr::instance() const { return agxData::EntityPtr::instance(); }
    AGX_FORCE_INLINE ConstraintRowSemantics* ConstraintRowPtr::operator->() { return (ConstraintRowSemantics* )this; }
    AGX_FORCE_INLINE const ConstraintRowSemantics* ConstraintRowPtr::operator->() const { return (const ConstraintRowSemantics* )this; }
    AGX_FORCE_INLINE ConstraintRowData* ConstraintRowPtr::getData() { return static_cast<ConstraintRowData* >(agxData::EntityPtr::getData()); }
    AGX_FORCE_INLINE const ConstraintRowData* ConstraintRowPtr::getData() const { return static_cast<const ConstraintRowData* >(agxData::EntityPtr::getData()); }

    AGX_FORCE_INLINE agx::Bool& ConstraintRowPtr::holonomic() { verifyIndex(); return getData()->holonomic[calculateIndex()]; }
    AGX_FORCE_INLINE agx::Bool const& ConstraintRowPtr::holonomic() const { verifyIndex(); return getData()->holonomic[calculateIndex()]; }

    AGX_FORCE_INLINE agx::Real& ConstraintRowPtr::damping() { verifyIndex(); return getData()->damping[calculateIndex()]; }
    AGX_FORCE_INLINE agx::Real const& ConstraintRowPtr::damping() const { verifyIndex(); return getData()->damping[calculateIndex()]; }

    AGX_FORCE_INLINE agx::Real& ConstraintRowPtr::epsilon() { verifyIndex(); return getData()->epsilon[calculateIndex()]; }
    AGX_FORCE_INLINE agx::Real const& ConstraintRowPtr::epsilon() const { verifyIndex(); return getData()->epsilon[calculateIndex()]; }

    AGX_FORCE_INLINE agx::Real& ConstraintRowPtr::lambda() { verifyIndex(); return getData()->lambda[calculateIndex()]; }
    AGX_FORCE_INLINE agx::Real const& ConstraintRowPtr::lambda() const { verifyIndex(); return getData()->lambda[calculateIndex()]; }

    AGX_FORCE_INLINE agx::Real& ConstraintRowPtr::totalLambda() { verifyIndex(); return getData()->totalLambda[calculateIndex()]; }
    AGX_FORCE_INLINE agx::Real const& ConstraintRowPtr::totalLambda() const { verifyIndex(); return getData()->totalLambda[calculateIndex()]; }

    AGX_FORCE_INLINE agx::Real& ConstraintRowPtr::rhs() { verifyIndex(); return getData()->rhs[calculateIndex()]; }
    AGX_FORCE_INLINE agx::Real const& ConstraintRowPtr::rhs() const { verifyIndex(); return getData()->rhs[calculateIndex()]; }

    AGX_FORCE_INLINE agx::Real& ConstraintRowPtr::invD() { verifyIndex(); return getData()->invD[calculateIndex()]; }
    AGX_FORCE_INLINE agx::Real const& ConstraintRowPtr::invD() const { verifyIndex(); return getData()->invD[calculateIndex()]; }

    AGX_FORCE_INLINE agx::Real& ConstraintRowPtr::impactSpeed() { verifyIndex(); return getData()->impactSpeed[calculateIndex()]; }
    AGX_FORCE_INLINE agx::Real const& ConstraintRowPtr::impactSpeed() const { verifyIndex(); return getData()->impactSpeed[calculateIndex()]; }

    AGX_FORCE_INLINE agx::Real& ConstraintRowPtr::velocity() { verifyIndex(); return getData()->velocity[calculateIndex()]; }
    AGX_FORCE_INLINE agx::Real const& ConstraintRowPtr::velocity() const { verifyIndex(); return getData()->velocity[calculateIndex()]; }

    AGX_FORCE_INLINE agx::Real& ConstraintRowPtr::violation() { verifyIndex(); return getData()->violation[calculateIndex()]; }
    AGX_FORCE_INLINE agx::Real const& ConstraintRowPtr::violation() const { verifyIndex(); return getData()->violation[calculateIndex()]; }

    AGX_FORCE_INLINE agx::RangeReal& ConstraintRowPtr::bound() { verifyIndex(); return getData()->bound[calculateIndex()]; }
    AGX_FORCE_INLINE agx::RangeReal const& ConstraintRowPtr::bound() const { verifyIndex(); return getData()->bound[calculateIndex()]; }

    AGX_FORCE_INLINE agx::Real& ConstraintRowPtr::residual() { verifyIndex(); return getData()->residual[calculateIndex()]; }
    AGX_FORCE_INLINE agx::Real const& ConstraintRowPtr::residual() const { verifyIndex(); return getData()->residual[calculateIndex()]; }

    AGX_FORCE_INLINE agx::Int8& ConstraintRowPtr::indexSetState() { verifyIndex(); return getData()->indexSetState[calculateIndex()]; }
    AGX_FORCE_INLINE agx::Int8 const& ConstraintRowPtr::indexSetState() const { verifyIndex(); return getData()->indexSetState[calculateIndex()]; }

    //-----------------------------------------------------------------------------------------------------
    AGX_FORCE_INLINE ConstraintRowInstance::ConstraintRowInstance() {}
    AGX_FORCE_INLINE ConstraintRowInstance::ConstraintRowInstance(ConstraintRowData* data, agx::Index index) : agxData::EntityInstance(data, index) {}
    AGX_FORCE_INLINE ConstraintRowInstance::ConstraintRowInstance(agxData::EntityStorage* storage, agx::Index index) : agxData::EntityInstance(storage, index) {}
    AGX_FORCE_INLINE ConstraintRowInstance::ConstraintRowInstance(const agxData::EntityInstance& other) : agxData::EntityInstance(other)
    {
      agxAssertN(!other || other.isInstanceOf(ConstraintRowModel::instance()),
        "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
        EntityInstance::getModel()->fullPath().c_str(), ConstraintRowModel::instance()->fullPath().c_str());
    }

    AGX_FORCE_INLINE ConstraintRowInstance::ConstraintRowInstance(const agxData::EntityPtr& ptr) : agxData::EntityInstance(ptr)
    {
      agxAssertN(!ptr || ptr.isInstanceOf(ConstraintRowModel::instance()),
        "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
        EntityInstance::getModel()->fullPath().c_str(), ConstraintRowModel::instance()->fullPath().c_str());
    }


    AGX_FORCE_INLINE ConstraintRowData* ConstraintRowInstance::getData() { return static_cast<ConstraintRowData* >(agxData::EntityInstance::getData()); }
    AGX_FORCE_INLINE const ConstraintRowData* ConstraintRowInstance::getData() const { return static_cast<const ConstraintRowData* >(agxData::EntityInstance::getData()); }

    AGX_FORCE_INLINE agx::Bool& ConstraintRowInstance::holonomic() { verifyIndex(); return getData()->holonomic[getIndex()]; }
    AGX_FORCE_INLINE agx::Bool const& ConstraintRowInstance::holonomic() const { verifyIndex(); return getData()->holonomic[getIndex()]; }

    AGX_FORCE_INLINE agx::Real& ConstraintRowInstance::damping() { verifyIndex(); return getData()->damping[getIndex()]; }
    AGX_FORCE_INLINE agx::Real const& ConstraintRowInstance::damping() const { verifyIndex(); return getData()->damping[getIndex()]; }

    AGX_FORCE_INLINE agx::Real& ConstraintRowInstance::epsilon() { verifyIndex(); return getData()->epsilon[getIndex()]; }
    AGX_FORCE_INLINE agx::Real const& ConstraintRowInstance::epsilon() const { verifyIndex(); return getData()->epsilon[getIndex()]; }

    AGX_FORCE_INLINE agx::Real& ConstraintRowInstance::lambda() { verifyIndex(); return getData()->lambda[getIndex()]; }
    AGX_FORCE_INLINE agx::Real const& ConstraintRowInstance::lambda() const { verifyIndex(); return getData()->lambda[getIndex()]; }

    AGX_FORCE_INLINE agx::Real& ConstraintRowInstance::totalLambda() { verifyIndex(); return getData()->totalLambda[getIndex()]; }
    AGX_FORCE_INLINE agx::Real const& ConstraintRowInstance::totalLambda() const { verifyIndex(); return getData()->totalLambda[getIndex()]; }

    AGX_FORCE_INLINE agx::Real& ConstraintRowInstance::rhs() { verifyIndex(); return getData()->rhs[getIndex()]; }
    AGX_FORCE_INLINE agx::Real const& ConstraintRowInstance::rhs() const { verifyIndex(); return getData()->rhs[getIndex()]; }

    AGX_FORCE_INLINE agx::Real& ConstraintRowInstance::invD() { verifyIndex(); return getData()->invD[getIndex()]; }
    AGX_FORCE_INLINE agx::Real const& ConstraintRowInstance::invD() const { verifyIndex(); return getData()->invD[getIndex()]; }

    AGX_FORCE_INLINE agx::Real& ConstraintRowInstance::impactSpeed() { verifyIndex(); return getData()->impactSpeed[getIndex()]; }
    AGX_FORCE_INLINE agx::Real const& ConstraintRowInstance::impactSpeed() const { verifyIndex(); return getData()->impactSpeed[getIndex()]; }

    AGX_FORCE_INLINE agx::Real& ConstraintRowInstance::velocity() { verifyIndex(); return getData()->velocity[getIndex()]; }
    AGX_FORCE_INLINE agx::Real const& ConstraintRowInstance::velocity() const { verifyIndex(); return getData()->velocity[getIndex()]; }

    AGX_FORCE_INLINE agx::Real& ConstraintRowInstance::violation() { verifyIndex(); return getData()->violation[getIndex()]; }
    AGX_FORCE_INLINE agx::Real const& ConstraintRowInstance::violation() const { verifyIndex(); return getData()->violation[getIndex()]; }

    AGX_FORCE_INLINE agx::RangeReal& ConstraintRowInstance::bound() { verifyIndex(); return getData()->bound[getIndex()]; }
    AGX_FORCE_INLINE agx::RangeReal const& ConstraintRowInstance::bound() const { verifyIndex(); return getData()->bound[getIndex()]; }

    AGX_FORCE_INLINE agx::Real& ConstraintRowInstance::residual() { verifyIndex(); return getData()->residual[getIndex()]; }
    AGX_FORCE_INLINE agx::Real const& ConstraintRowInstance::residual() const { verifyIndex(); return getData()->residual[getIndex()]; }

    AGX_FORCE_INLINE agx::Int8& ConstraintRowInstance::indexSetState() { verifyIndex(); return getData()->indexSetState[getIndex()]; }
    AGX_FORCE_INLINE agx::Int8 const& ConstraintRowInstance::indexSetState() const { verifyIndex(); return getData()->indexSetState[getIndex()]; }

    //-----------------------------------------------------------------------------------------------------
    AGX_FORCE_INLINE ConstraintRowSemantics::ConstraintRowSemantics() {}
    //-----------------------------------------------------------------------------------------------------
    DOXYGEN_END_INTERNAL_BLOCK()
  }
}

AGX_TYPE_BINDING(agx::Physics::ConstraintRowPtr, "Physics.ConstraintRowPtr")
AGX_TYPE_BINDING(agx::Physics::ConstraintRowInstance, "Physics.ConstraintRowInstance")

#ifdef _MSC_VER
# pragma warning(pop)
#endif

#undef AGX_ENTITY_WRAPPER
#undef AGX_ENTITY_NAMESPACE
#endif

