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

#ifndef GENERATED_AGX_PHYSICS_PARTICLE_CONTACTCONSTRAINTROW_H_PLUGIN
#define GENERATED_AGX_PHYSICS_PARTICLE_CONTACTCONSTRAINTROW_H_PLUGIN

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


namespace agx
{
  namespace Physics
  {
    namespace Particle
    {

      class ContactConstraintRowModel;
      class ContactConstraintRowData;
      class ContactConstraintRowPtr;
      class ContactConstraintRowInstance;
      class ContactConstraintRowSemantics;


      AGX_DECLARE_POINTER_TYPES(ContactConstraintRowModel);

      /** 
      Abstract description of the data attributes for the Physics.Particle.ContactConstraintRow entity.
      */ 
      class AGXPHYSICS_EXPORT ContactConstraintRowModel : public agxData::EntityModel
      {
      public:
        typedef ContactConstraintRowPtr PtrT;

        ContactConstraintRowModel(const agx::String& name = "ContactConstraintRow");

        /// \return The entity model singleton.
        static ContactConstraintRowModel* instance();

        /// Create and return a pointer to a new instance in the default storage for this entity model.
        static ContactConstraintRowPtr createInstance();

        /// \return The default storage for this entity model.
        static agxData::EntityStorage* defaultStorage();

        /// This is part of internal cleanup and should not be called by users
        virtual void shutdownCleanup() override;



        /* Attributes */
        static agxData::ScalarAttributeT< agx::Real >* lambdaAttribute;
        static agxData::ScalarAttributeT< agx::Real >* impactLambdaAttribute;
        static agxData::ScalarAttributeT< agx::Real >* rhsAttribute;
        static agxData::ScalarAttributeT< agx::Real >* epsilonAttribute;
        static agxData::ScalarAttributeT< agx::Real >* velocityAttribute;
        static agxData::ScalarAttributeT< agx::Real >* invDAttribute;
        static agxData::ScalarAttributeT< agx::Real >* violationAttribute;
        static agxData::ScalarAttributeT< agx::Real >* residualAttribute;

      protected:
        virtual ~ContactConstraintRowModel();
        virtual agxData::EntityData* createData(agxData::EntityStorage* storage) override;
        virtual void configure(agx::TiXmlElement* eEntity) override;
        virtual void initAttributeAccessors() override;
        virtual void construct(agxData::EntityPtr instance) override;
        void construct(agx::Physics::Particle::ContactConstraintRowPtr contactConstraintRow);
      };


      DOXYGEN_START_INTERNAL_BLOCK()
      #ifndef AGX_PHYSICS_PARTICLE_CONTACTCONSTRAINTROW_DATA_SET_OVERRIDE
      #define AGX_PHYSICS_PARTICLE_CONTACTCONSTRAINTROW_DATA_SET
      class AGXPHYSICS_EXPORT ContactConstraintRowData : public agxData::EntityData
      {
      public:
        ContactConstraintRowInstance operator[] (size_t index);

      public:
        agxData::Array< ContactConstraintRowPtr >& instance;
        agxData::Array< agx::Real > lambda;
        agxData::Array< agx::Real > impactLambda;
        agxData::Array< agx::Real > rhs;
        agxData::Array< agx::Real > epsilon;
        agxData::Array< agx::Real > velocity;
        agxData::Array< agx::Real > invD;
        agxData::Array< agx::Real > violation;
        agxData::Array< agx::Real > residual;

      public:
        typedef agx::Real lambdaType;
        typedef agx::Real impactLambdaType;
        typedef agx::Real rhsType;
        typedef agx::Real epsilonType;
        typedef agx::Real velocityType;
        typedef agx::Real invDType;
        typedef agx::Real violationType;
        typedef agx::Real residualType;

      public:
        ContactConstraintRowData(agxData::EntityStorage* storage);
        ContactConstraintRowData();

      protected:
        virtual ~ContactConstraintRowData() {}
        virtual void setNumElements(agx::Index numElements) override;

      private:
        ContactConstraintRowData& operator= (const ContactConstraintRowData&) { return *this; }

      };
      #endif
      DOXYGEN_END_INTERNAL_BLOCK()


      DOXYGEN_START_INTERNAL_BLOCK()
      class AGXPHYSICS_EXPORT ContactConstraintRowSemantics : protected agxData::EntityPtr
      {
      public:

        // Automatic getters
        agx::Real const& getLambda() const;
        agx::Real const& getImpactLambda() const;
        agx::Real const& getRhs() const;
        agx::Real const& getEpsilon() const;
        agx::Real const& getVelocity() const;
        agx::Real const& getInvD() const;
        agx::Real const& getViolation() const;
        agx::Real const& getResidual() const;

        // Semantics defined by explicit kernels

        // Automatic setters
        void setLambda(agx::Real const& value);
        void setImpactLambda(agx::Real const& value);
        void setRhs(agx::Real const& value);
        void setEpsilon(agx::Real const& value);
        void setVelocity(agx::Real const& value);
        void setInvD(agx::Real const& value);
        void setViolation(agx::Real const& value);
        void setResidual(agx::Real const& value);


      protected:
        friend class ContactConstraintRowPtr;
        friend class ContactConstraintRowInstance;
        ContactConstraintRowSemantics();
      };
      DOXYGEN_END_INTERNAL_BLOCK()


      /**
      Pointer to a entity instance of type Physics.Particle.ContactConstraintRow
      */
      class CALLABLE ContactConstraintRowPtr : public agxData::EntityPtr
      {
      public:
        typedef ContactConstraintRowModel ModelType;
        typedef ContactConstraintRowData DataType;
        typedef ContactConstraintRowInstance InstanceType;

      public:
        AGXPHYSICS_EXPORT ContactConstraintRowPtr();
        AGXPHYSICS_EXPORT ContactConstraintRowPtr(agxData::EntityStorage* storage, agx::Index id);
        AGXPHYSICS_EXPORT ContactConstraintRowPtr(const agxData::EntityPtr& ptr);
        AGXPHYSICS_EXPORT ContactConstraintRowPtr(const agxData::EntityInstance& instance);
        AGXPHYSICS_EXPORT ContactConstraintRowPtr& operator= (const agxData::EntityPtr& ptr);
        AGXPHYSICS_EXPORT ContactConstraintRowPtr& operator= (const agxData::EntityInstance& instance);
        AGXPHYSICS_EXPORT ContactConstraintRowInstance instance();
        AGXPHYSICS_EXPORT const ContactConstraintRowInstance instance() const;

        AGXPHYSICS_EXPORT ContactConstraintRowSemantics* operator->();
        AGXPHYSICS_EXPORT const ContactConstraintRowSemantics* operator->() const;

        ContactConstraintRowData* getData();
        const ContactConstraintRowData* getData() const;


        /// \return reference to the lambda attribute
        AGXPHYSICS_EXPORT agx::Real& lambda();
        /// \return const reference to the lambda attribute
        AGXPHYSICS_EXPORT agx::Real const& lambda() const;

        /// \return reference to the impactLambda attribute
        AGXPHYSICS_EXPORT agx::Real& impactLambda();
        /// \return const reference to the impactLambda attribute
        AGXPHYSICS_EXPORT agx::Real const& impactLambda() const;

        /// \return reference to the rhs attribute
        AGXPHYSICS_EXPORT agx::Real& rhs();
        /// \return const reference to the rhs attribute
        AGXPHYSICS_EXPORT agx::Real const& rhs() const;

        /// \return reference to the epsilon attribute
        AGXPHYSICS_EXPORT agx::Real& epsilon();
        /// \return const reference to the epsilon attribute
        AGXPHYSICS_EXPORT agx::Real const& epsilon() const;

        /// \return reference to the velocity attribute
        AGXPHYSICS_EXPORT agx::Real& velocity();
        /// \return const reference to the velocity attribute
        AGXPHYSICS_EXPORT agx::Real const& velocity() const;

        /// \return reference to the invD attribute
        AGXPHYSICS_EXPORT agx::Real& invD();
        /// \return const reference to the invD attribute
        AGXPHYSICS_EXPORT agx::Real const& invD() const;

        /// \return reference to the violation attribute
        AGXPHYSICS_EXPORT agx::Real& violation();
        /// \return const reference to the violation attribute
        AGXPHYSICS_EXPORT agx::Real const& violation() const;

        /// \return reference to the residual attribute
        AGXPHYSICS_EXPORT agx::Real& residual();
        /// \return const reference to the residual attribute
        AGXPHYSICS_EXPORT agx::Real const& residual() const;

        DOXYGEN_START_INTERNAL_BLOCK()
        // Only used by tolua!
        #ifdef AGX_GENERATED_TOLUA_WRAPPER
        inline agx::Real const& getLambda() const { return (*this)->getLambda(); }
        inline agx::Real const& getImpactLambda() const { return (*this)->getImpactLambda(); }
        inline agx::Real const& getRhs() const { return (*this)->getRhs(); }
        inline agx::Real const& getEpsilon() const { return (*this)->getEpsilon(); }
        inline agx::Real const& getVelocity() const { return (*this)->getVelocity(); }
        inline agx::Real const& getInvD() const { return (*this)->getInvD(); }
        inline agx::Real const& getViolation() const { return (*this)->getViolation(); }
        inline agx::Real const& getResidual() const { return (*this)->getResidual(); }


        inline void setLambda(agx::Real const& value) { (*this)->setLambda(value); }
        inline void setImpactLambda(agx::Real const& value) { (*this)->setImpactLambda(value); }
        inline void setRhs(agx::Real const& value) { (*this)->setRhs(value); }
        inline void setEpsilon(agx::Real const& value) { (*this)->setEpsilon(value); }
        inline void setVelocity(agx::Real const& value) { (*this)->setVelocity(value); }
        inline void setInvD(agx::Real const& value) { (*this)->setInvD(value); }
        inline void setViolation(agx::Real const& value) { (*this)->setViolation(value); }
        inline void setResidual(agx::Real const& value) { (*this)->setResidual(value); }
        #endif
        DOXYGEN_END_INTERNAL_BLOCK()

      };


      DOXYGEN_START_INTERNAL_BLOCK()
      class AGXPHYSICS_EXPORT ContactConstraintRowInstance : public agxData::EntityInstance
      {
      public:
        ContactConstraintRowInstance();
        ContactConstraintRowInstance(ContactConstraintRowData* data, agx::Index index);
        ContactConstraintRowInstance(agxData::EntityStorage *storage, agx::Index index);
        ContactConstraintRowInstance(const agxData::EntityInstance& other);
        ContactConstraintRowInstance(const agxData::EntityPtr& ptr);

        ContactConstraintRowData* getData();
        const ContactConstraintRowData* getData() const;

      public:
        /// \return reference to the lambda attribute
        agx::Real& lambda();
        /// \return const reference to the lambda attribute
        agx::Real const& lambda() const;

        /// \return reference to the impactLambda attribute
        agx::Real& impactLambda();
        /// \return const reference to the impactLambda attribute
        agx::Real const& impactLambda() const;

        /// \return reference to the rhs attribute
        agx::Real& rhs();
        /// \return const reference to the rhs attribute
        agx::Real const& rhs() const;

        /// \return reference to the epsilon attribute
        agx::Real& epsilon();
        /// \return const reference to the epsilon attribute
        agx::Real const& epsilon() const;

        /// \return reference to the velocity attribute
        agx::Real& velocity();
        /// \return const reference to the velocity attribute
        agx::Real const& velocity() const;

        /// \return reference to the invD attribute
        agx::Real& invD();
        /// \return const reference to the invD attribute
        agx::Real const& invD() const;

        /// \return reference to the violation attribute
        agx::Real& violation();
        /// \return const reference to the violation attribute
        agx::Real const& violation() const;

        /// \return reference to the residual attribute
        agx::Real& residual();
        /// \return const reference to the residual attribute
        agx::Real const& residual() const;

      };
      DOXYGEN_END_INTERNAL_BLOCK()



      typedef agx::VectorPOD<ContactConstraintRowPtr> ContactConstraintRowPtrVector;
      typedef agxData::Array<ContactConstraintRowPtr> ContactConstraintRowPtrArray;



      DOXYGEN_START_INTERNAL_BLOCK()
      /* Implementation */
      //-----------------------------------------------------------------------------------------------------
      //-----------------------------------------------------------------------------------------------------
      inline ContactConstraintRowInstance agx::Physics::Particle::ContactConstraintRowData::operator[] (size_t index) { return ContactConstraintRowInstance(this, (agx::Index)index); }
      //-----------------------------------------------------------------------------------------------------
      AGX_FORCE_INLINE ContactConstraintRowPtr::ContactConstraintRowPtr() {}
      AGX_FORCE_INLINE ContactConstraintRowPtr::ContactConstraintRowPtr(agxData::EntityStorage* storage, agx::Index id) : agxData::EntityPtr(storage, id) {}
      AGX_FORCE_INLINE ContactConstraintRowPtr::ContactConstraintRowPtr(const agxData::EntityPtr& ptr) : agxData::EntityPtr(ptr)
      {
        agxAssertN(!ptr || ptr.isInstanceOf(ContactConstraintRowModel::instance()),
          "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
          EntityPtr::getModel()->fullPath().c_str(), ContactConstraintRowModel::instance()->fullPath().c_str());
      }

      AGX_FORCE_INLINE ContactConstraintRowPtr::ContactConstraintRowPtr(const agxData::EntityInstance& instance) : agxData::EntityPtr(instance)
      {
        agxAssertN(!instance || instance.isInstanceOf(ContactConstraintRowModel::instance()),
          "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
          EntityPtr::getModel()->fullPath().c_str(), ContactConstraintRowModel::instance()->fullPath().c_str());
      }

      AGX_FORCE_INLINE ContactConstraintRowPtr& ContactConstraintRowPtr::operator= (const agxData::EntityPtr& ptr)
      {
        agxData::EntityPtr::operator= (ptr);
        agxAssertN(!ptr || ptr.isInstanceOf(ContactConstraintRowModel::instance()),
          "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
          EntityPtr::getModel()->fullPath().c_str(), ContactConstraintRowModel::instance()->fullPath().c_str());
        return *this;
      }

      AGX_FORCE_INLINE ContactConstraintRowPtr& ContactConstraintRowPtr::operator= (const agxData::EntityInstance& instance)
      {
        agxData::EntityPtr::operator= (instance);
        agxAssertN(!instance || instance.isInstanceOf(ContactConstraintRowModel::instance()),
          "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
          EntityPtr::getModel()->fullPath().c_str(), ContactConstraintRowModel::instance()->fullPath().c_str());
        return *this;
      }

      inline ContactConstraintRowInstance ContactConstraintRowPtr::instance() { return agxData::EntityPtr::instance(); }
      inline const ContactConstraintRowInstance ContactConstraintRowPtr::instance() const { return agxData::EntityPtr::instance(); }
      AGX_FORCE_INLINE ContactConstraintRowSemantics* ContactConstraintRowPtr::operator->() { return (ContactConstraintRowSemantics* )this; }
      AGX_FORCE_INLINE const ContactConstraintRowSemantics* ContactConstraintRowPtr::operator->() const { return (const ContactConstraintRowSemantics* )this; }
      AGX_FORCE_INLINE ContactConstraintRowData* ContactConstraintRowPtr::getData() { return static_cast<ContactConstraintRowData* >(agxData::EntityPtr::getData()); }
      AGX_FORCE_INLINE const ContactConstraintRowData* ContactConstraintRowPtr::getData() const { return static_cast<const ContactConstraintRowData* >(agxData::EntityPtr::getData()); }

      AGX_FORCE_INLINE agx::Real& ContactConstraintRowPtr::lambda() { verifyIndex(); return getData()->lambda[calculateIndex()]; }
      AGX_FORCE_INLINE agx::Real const& ContactConstraintRowPtr::lambda() const { verifyIndex(); return getData()->lambda[calculateIndex()]; }

      AGX_FORCE_INLINE agx::Real& ContactConstraintRowPtr::impactLambda() { verifyIndex(); return getData()->impactLambda[calculateIndex()]; }
      AGX_FORCE_INLINE agx::Real const& ContactConstraintRowPtr::impactLambda() const { verifyIndex(); return getData()->impactLambda[calculateIndex()]; }

      AGX_FORCE_INLINE agx::Real& ContactConstraintRowPtr::rhs() { verifyIndex(); return getData()->rhs[calculateIndex()]; }
      AGX_FORCE_INLINE agx::Real const& ContactConstraintRowPtr::rhs() const { verifyIndex(); return getData()->rhs[calculateIndex()]; }

      AGX_FORCE_INLINE agx::Real& ContactConstraintRowPtr::epsilon() { verifyIndex(); return getData()->epsilon[calculateIndex()]; }
      AGX_FORCE_INLINE agx::Real const& ContactConstraintRowPtr::epsilon() const { verifyIndex(); return getData()->epsilon[calculateIndex()]; }

      AGX_FORCE_INLINE agx::Real& ContactConstraintRowPtr::velocity() { verifyIndex(); return getData()->velocity[calculateIndex()]; }
      AGX_FORCE_INLINE agx::Real const& ContactConstraintRowPtr::velocity() const { verifyIndex(); return getData()->velocity[calculateIndex()]; }

      AGX_FORCE_INLINE agx::Real& ContactConstraintRowPtr::invD() { verifyIndex(); return getData()->invD[calculateIndex()]; }
      AGX_FORCE_INLINE agx::Real const& ContactConstraintRowPtr::invD() const { verifyIndex(); return getData()->invD[calculateIndex()]; }

      AGX_FORCE_INLINE agx::Real& ContactConstraintRowPtr::violation() { verifyIndex(); return getData()->violation[calculateIndex()]; }
      AGX_FORCE_INLINE agx::Real const& ContactConstraintRowPtr::violation() const { verifyIndex(); return getData()->violation[calculateIndex()]; }

      AGX_FORCE_INLINE agx::Real& ContactConstraintRowPtr::residual() { verifyIndex(); return getData()->residual[calculateIndex()]; }
      AGX_FORCE_INLINE agx::Real const& ContactConstraintRowPtr::residual() const { verifyIndex(); return getData()->residual[calculateIndex()]; }

      //-----------------------------------------------------------------------------------------------------
      AGX_FORCE_INLINE ContactConstraintRowInstance::ContactConstraintRowInstance() {}
      AGX_FORCE_INLINE ContactConstraintRowInstance::ContactConstraintRowInstance(ContactConstraintRowData* data, agx::Index index) : agxData::EntityInstance(data, index) {}
      AGX_FORCE_INLINE ContactConstraintRowInstance::ContactConstraintRowInstance(agxData::EntityStorage* storage, agx::Index index) : agxData::EntityInstance(storage, index) {}
      AGX_FORCE_INLINE ContactConstraintRowInstance::ContactConstraintRowInstance(const agxData::EntityInstance& other) : agxData::EntityInstance(other)
      {
        agxAssertN(!other || other.isInstanceOf(ContactConstraintRowModel::instance()),
          "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
          EntityInstance::getModel()->fullPath().c_str(), ContactConstraintRowModel::instance()->fullPath().c_str());
      }

      AGX_FORCE_INLINE ContactConstraintRowInstance::ContactConstraintRowInstance(const agxData::EntityPtr& ptr) : agxData::EntityInstance(ptr)
      {
        agxAssertN(!ptr || ptr.isInstanceOf(ContactConstraintRowModel::instance()),
          "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
          EntityInstance::getModel()->fullPath().c_str(), ContactConstraintRowModel::instance()->fullPath().c_str());
      }


      AGX_FORCE_INLINE ContactConstraintRowData* ContactConstraintRowInstance::getData() { return static_cast<ContactConstraintRowData* >(agxData::EntityInstance::getData()); }
      AGX_FORCE_INLINE const ContactConstraintRowData* ContactConstraintRowInstance::getData() const { return static_cast<const ContactConstraintRowData* >(agxData::EntityInstance::getData()); }

      AGX_FORCE_INLINE agx::Real& ContactConstraintRowInstance::lambda() { verifyIndex(); return getData()->lambda[getIndex()]; }
      AGX_FORCE_INLINE agx::Real const& ContactConstraintRowInstance::lambda() const { verifyIndex(); return getData()->lambda[getIndex()]; }

      AGX_FORCE_INLINE agx::Real& ContactConstraintRowInstance::impactLambda() { verifyIndex(); return getData()->impactLambda[getIndex()]; }
      AGX_FORCE_INLINE agx::Real const& ContactConstraintRowInstance::impactLambda() const { verifyIndex(); return getData()->impactLambda[getIndex()]; }

      AGX_FORCE_INLINE agx::Real& ContactConstraintRowInstance::rhs() { verifyIndex(); return getData()->rhs[getIndex()]; }
      AGX_FORCE_INLINE agx::Real const& ContactConstraintRowInstance::rhs() const { verifyIndex(); return getData()->rhs[getIndex()]; }

      AGX_FORCE_INLINE agx::Real& ContactConstraintRowInstance::epsilon() { verifyIndex(); return getData()->epsilon[getIndex()]; }
      AGX_FORCE_INLINE agx::Real const& ContactConstraintRowInstance::epsilon() const { verifyIndex(); return getData()->epsilon[getIndex()]; }

      AGX_FORCE_INLINE agx::Real& ContactConstraintRowInstance::velocity() { verifyIndex(); return getData()->velocity[getIndex()]; }
      AGX_FORCE_INLINE agx::Real const& ContactConstraintRowInstance::velocity() const { verifyIndex(); return getData()->velocity[getIndex()]; }

      AGX_FORCE_INLINE agx::Real& ContactConstraintRowInstance::invD() { verifyIndex(); return getData()->invD[getIndex()]; }
      AGX_FORCE_INLINE agx::Real const& ContactConstraintRowInstance::invD() const { verifyIndex(); return getData()->invD[getIndex()]; }

      AGX_FORCE_INLINE agx::Real& ContactConstraintRowInstance::violation() { verifyIndex(); return getData()->violation[getIndex()]; }
      AGX_FORCE_INLINE agx::Real const& ContactConstraintRowInstance::violation() const { verifyIndex(); return getData()->violation[getIndex()]; }

      AGX_FORCE_INLINE agx::Real& ContactConstraintRowInstance::residual() { verifyIndex(); return getData()->residual[getIndex()]; }
      AGX_FORCE_INLINE agx::Real const& ContactConstraintRowInstance::residual() const { verifyIndex(); return getData()->residual[getIndex()]; }

      //-----------------------------------------------------------------------------------------------------
      AGX_FORCE_INLINE ContactConstraintRowSemantics::ContactConstraintRowSemantics() {}
      //-----------------------------------------------------------------------------------------------------
      DOXYGEN_END_INTERNAL_BLOCK()
    }
  }
}

AGX_TYPE_BINDING(agx::Physics::Particle::ContactConstraintRowPtr, "Physics.Particle.ContactConstraintRowPtr")
AGX_TYPE_BINDING(agx::Physics::Particle::ContactConstraintRowInstance, "Physics.Particle.ContactConstraintRowInstance")

#ifdef _MSC_VER
# pragma warning(pop)
#endif

#undef AGX_ENTITY_WRAPPER
#undef AGX_ENTITY_NAMESPACE
#endif

