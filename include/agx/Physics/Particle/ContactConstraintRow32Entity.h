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

#ifndef GENERATED_AGX_PHYSICS_PARTICLE_CONTACTCONSTRAINTROW32_H_PLUGIN
#define GENERATED_AGX_PHYSICS_PARTICLE_CONTACTCONSTRAINTROW32_H_PLUGIN

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

      class ContactConstraintRow32Model;
      class ContactConstraintRow32Data;
      class ContactConstraintRow32Ptr;
      class ContactConstraintRow32Instance;
      class ContactConstraintRow32Semantics;


      AGX_DECLARE_POINTER_TYPES(ContactConstraintRow32Model);

      /** 
      Abstract description of the data attributes for the Physics.Particle.ContactConstraintRow32 entity.
      */ 
      class AGXPHYSICS_EXPORT ContactConstraintRow32Model : public agxData::EntityModel
      {
      public:
        typedef ContactConstraintRow32Ptr PtrT;

        ContactConstraintRow32Model(const agx::String& name = "ContactConstraintRow32");

        /// \return The entity model singleton.
        static ContactConstraintRow32Model* instance();

        /// Create and return a pointer to a new instance in the default storage for this entity model.
        static ContactConstraintRow32Ptr createInstance();

        /// \return The default storage for this entity model.
        static agxData::EntityStorage* defaultStorage();

        /// This is part of internal cleanup and should not be called by users
        virtual void shutdownCleanup() override;



        /* Attributes */
        static agxData::ScalarAttributeT< agx::Real32 >* lambdaAttribute;
        static agxData::ScalarAttributeT< agx::Real32 >* impactLambdaAttribute;
        static agxData::ScalarAttributeT< agx::Real32 >* rhsAttribute;
        static agxData::ScalarAttributeT< agx::Real32 >* epsilonAttribute;
        static agxData::ScalarAttributeT< agx::Real32 >* velocityAttribute;
        static agxData::ScalarAttributeT< agx::Real32 >* invDAttribute;
        static agxData::ScalarAttributeT< agx::Real32 >* violationAttribute;
        static agxData::ScalarAttributeT< agx::Real >* residualAttribute;

      protected:
        virtual ~ContactConstraintRow32Model();
        virtual agxData::EntityData* createData(agxData::EntityStorage* storage) override;
        virtual void configure(agx::TiXmlElement* eEntity) override;
        virtual void initAttributeAccessors() override;
        virtual void construct(agxData::EntityPtr instance) override;
        void construct(agx::Physics::Particle::ContactConstraintRow32Ptr contactConstraintRow32);
      };


      DOXYGEN_START_INTERNAL_BLOCK()
      #ifndef AGX_PHYSICS_PARTICLE_CONTACTCONSTRAINTROW32_DATA_SET_OVERRIDE
      #define AGX_PHYSICS_PARTICLE_CONTACTCONSTRAINTROW32_DATA_SET
      class AGXPHYSICS_EXPORT ContactConstraintRow32Data : public agxData::EntityData
      {
      public:
        ContactConstraintRow32Instance operator[] (size_t index);

      public:
        agxData::Array< ContactConstraintRow32Ptr >& instance;
        agxData::Array< agx::Real32 > lambda;
        agxData::Array< agx::Real32 > impactLambda;
        agxData::Array< agx::Real32 > rhs;
        agxData::Array< agx::Real32 > epsilon;
        agxData::Array< agx::Real32 > velocity;
        agxData::Array< agx::Real32 > invD;
        agxData::Array< agx::Real32 > violation;
        agxData::Array< agx::Real > residual;

      public:
        typedef agx::Real32 lambdaType;
        typedef agx::Real32 impactLambdaType;
        typedef agx::Real32 rhsType;
        typedef agx::Real32 epsilonType;
        typedef agx::Real32 velocityType;
        typedef agx::Real32 invDType;
        typedef agx::Real32 violationType;
        typedef agx::Real residualType;

      public:
        ContactConstraintRow32Data(agxData::EntityStorage* storage);
        ContactConstraintRow32Data();

      protected:
        virtual ~ContactConstraintRow32Data() {}
        virtual void setNumElements(agx::Index numElements) override;

      private:
        ContactConstraintRow32Data& operator= (const ContactConstraintRow32Data&) { return *this; }

      };
      #endif
      DOXYGEN_END_INTERNAL_BLOCK()


      DOXYGEN_START_INTERNAL_BLOCK()
      class AGXPHYSICS_EXPORT ContactConstraintRow32Semantics : protected agxData::EntityPtr
      {
      public:

        // Automatic getters
        agx::Real32 const& getLambda() const;
        agx::Real32 const& getImpactLambda() const;
        agx::Real32 const& getRhs() const;
        agx::Real32 const& getEpsilon() const;
        agx::Real32 const& getVelocity() const;
        agx::Real32 const& getInvD() const;
        agx::Real32 const& getViolation() const;
        agx::Real const& getResidual() const;

        // Semantics defined by explicit kernels

        // Automatic setters
        void setLambda(agx::Real32 const& value);
        void setImpactLambda(agx::Real32 const& value);
        void setRhs(agx::Real32 const& value);
        void setEpsilon(agx::Real32 const& value);
        void setVelocity(agx::Real32 const& value);
        void setInvD(agx::Real32 const& value);
        void setViolation(agx::Real32 const& value);
        void setResidual(agx::Real const& value);


      protected:
        friend class ContactConstraintRow32Ptr;
        friend class ContactConstraintRow32Instance;
        ContactConstraintRow32Semantics();
      };
      DOXYGEN_END_INTERNAL_BLOCK()


      /**
      Pointer to a entity instance of type Physics.Particle.ContactConstraintRow32
      */
      class CALLABLE ContactConstraintRow32Ptr : public agxData::EntityPtr
      {
      public:
        typedef ContactConstraintRow32Model ModelType;
        typedef ContactConstraintRow32Data DataType;
        typedef ContactConstraintRow32Instance InstanceType;

      public:
        AGXPHYSICS_EXPORT ContactConstraintRow32Ptr();
        AGXPHYSICS_EXPORT ContactConstraintRow32Ptr(agxData::EntityStorage* storage, agx::Index id);
        AGXPHYSICS_EXPORT ContactConstraintRow32Ptr(const agxData::EntityPtr& ptr);
        AGXPHYSICS_EXPORT ContactConstraintRow32Ptr(const agxData::EntityInstance& instance);
        AGXPHYSICS_EXPORT ContactConstraintRow32Ptr& operator= (const agxData::EntityPtr& ptr);
        AGXPHYSICS_EXPORT ContactConstraintRow32Ptr& operator= (const agxData::EntityInstance& instance);
        AGXPHYSICS_EXPORT ContactConstraintRow32Instance instance();
        AGXPHYSICS_EXPORT const ContactConstraintRow32Instance instance() const;

        AGXPHYSICS_EXPORT ContactConstraintRow32Semantics* operator->();
        AGXPHYSICS_EXPORT const ContactConstraintRow32Semantics* operator->() const;

        ContactConstraintRow32Data* getData();
        const ContactConstraintRow32Data* getData() const;


        /// \return reference to the lambda attribute
        AGXPHYSICS_EXPORT agx::Real32& lambda();
        /// \return const reference to the lambda attribute
        AGXPHYSICS_EXPORT agx::Real32 const& lambda() const;

        /// \return reference to the impactLambda attribute
        AGXPHYSICS_EXPORT agx::Real32& impactLambda();
        /// \return const reference to the impactLambda attribute
        AGXPHYSICS_EXPORT agx::Real32 const& impactLambda() const;

        /// \return reference to the rhs attribute
        AGXPHYSICS_EXPORT agx::Real32& rhs();
        /// \return const reference to the rhs attribute
        AGXPHYSICS_EXPORT agx::Real32 const& rhs() const;

        /// \return reference to the epsilon attribute
        AGXPHYSICS_EXPORT agx::Real32& epsilon();
        /// \return const reference to the epsilon attribute
        AGXPHYSICS_EXPORT agx::Real32 const& epsilon() const;

        /// \return reference to the velocity attribute
        AGXPHYSICS_EXPORT agx::Real32& velocity();
        /// \return const reference to the velocity attribute
        AGXPHYSICS_EXPORT agx::Real32 const& velocity() const;

        /// \return reference to the invD attribute
        AGXPHYSICS_EXPORT agx::Real32& invD();
        /// \return const reference to the invD attribute
        AGXPHYSICS_EXPORT agx::Real32 const& invD() const;

        /// \return reference to the violation attribute
        AGXPHYSICS_EXPORT agx::Real32& violation();
        /// \return const reference to the violation attribute
        AGXPHYSICS_EXPORT agx::Real32 const& violation() const;

        /// \return reference to the residual attribute
        AGXPHYSICS_EXPORT agx::Real& residual();
        /// \return const reference to the residual attribute
        AGXPHYSICS_EXPORT agx::Real const& residual() const;

        DOXYGEN_START_INTERNAL_BLOCK()
        // Only used by tolua!
        #ifdef AGX_GENERATED_TOLUA_WRAPPER
        inline agx::Real32 const& getLambda() const { return (*this)->getLambda(); }
        inline agx::Real32 const& getImpactLambda() const { return (*this)->getImpactLambda(); }
        inline agx::Real32 const& getRhs() const { return (*this)->getRhs(); }
        inline agx::Real32 const& getEpsilon() const { return (*this)->getEpsilon(); }
        inline agx::Real32 const& getVelocity() const { return (*this)->getVelocity(); }
        inline agx::Real32 const& getInvD() const { return (*this)->getInvD(); }
        inline agx::Real32 const& getViolation() const { return (*this)->getViolation(); }
        inline agx::Real const& getResidual() const { return (*this)->getResidual(); }


        inline void setLambda(agx::Real32 const& value) { (*this)->setLambda(value); }
        inline void setImpactLambda(agx::Real32 const& value) { (*this)->setImpactLambda(value); }
        inline void setRhs(agx::Real32 const& value) { (*this)->setRhs(value); }
        inline void setEpsilon(agx::Real32 const& value) { (*this)->setEpsilon(value); }
        inline void setVelocity(agx::Real32 const& value) { (*this)->setVelocity(value); }
        inline void setInvD(agx::Real32 const& value) { (*this)->setInvD(value); }
        inline void setViolation(agx::Real32 const& value) { (*this)->setViolation(value); }
        inline void setResidual(agx::Real const& value) { (*this)->setResidual(value); }
        #endif
        DOXYGEN_END_INTERNAL_BLOCK()

      };


      DOXYGEN_START_INTERNAL_BLOCK()
      class AGXPHYSICS_EXPORT ContactConstraintRow32Instance : public agxData::EntityInstance
      {
      public:
        ContactConstraintRow32Instance();
        ContactConstraintRow32Instance(ContactConstraintRow32Data* data, agx::Index index);
        ContactConstraintRow32Instance(agxData::EntityStorage *storage, agx::Index index);
        ContactConstraintRow32Instance(const agxData::EntityInstance& other);
        ContactConstraintRow32Instance(const agxData::EntityPtr& ptr);

        ContactConstraintRow32Data* getData();
        const ContactConstraintRow32Data* getData() const;

      public:
        /// \return reference to the lambda attribute
        agx::Real32& lambda();
        /// \return const reference to the lambda attribute
        agx::Real32 const& lambda() const;

        /// \return reference to the impactLambda attribute
        agx::Real32& impactLambda();
        /// \return const reference to the impactLambda attribute
        agx::Real32 const& impactLambda() const;

        /// \return reference to the rhs attribute
        agx::Real32& rhs();
        /// \return const reference to the rhs attribute
        agx::Real32 const& rhs() const;

        /// \return reference to the epsilon attribute
        agx::Real32& epsilon();
        /// \return const reference to the epsilon attribute
        agx::Real32 const& epsilon() const;

        /// \return reference to the velocity attribute
        agx::Real32& velocity();
        /// \return const reference to the velocity attribute
        agx::Real32 const& velocity() const;

        /// \return reference to the invD attribute
        agx::Real32& invD();
        /// \return const reference to the invD attribute
        agx::Real32 const& invD() const;

        /// \return reference to the violation attribute
        agx::Real32& violation();
        /// \return const reference to the violation attribute
        agx::Real32 const& violation() const;

        /// \return reference to the residual attribute
        agx::Real& residual();
        /// \return const reference to the residual attribute
        agx::Real const& residual() const;

      };
      DOXYGEN_END_INTERNAL_BLOCK()



      typedef agx::VectorPOD<ContactConstraintRow32Ptr> ContactConstraintRow32PtrVector;
      typedef agxData::Array<ContactConstraintRow32Ptr> ContactConstraintRow32PtrArray;



      DOXYGEN_START_INTERNAL_BLOCK()
      /* Implementation */
      //-----------------------------------------------------------------------------------------------------
      //-----------------------------------------------------------------------------------------------------
      inline ContactConstraintRow32Instance agx::Physics::Particle::ContactConstraintRow32Data::operator[] (size_t index) { return ContactConstraintRow32Instance(this, (agx::Index)index); }
      //-----------------------------------------------------------------------------------------------------
      AGX_FORCE_INLINE ContactConstraintRow32Ptr::ContactConstraintRow32Ptr() {}
      AGX_FORCE_INLINE ContactConstraintRow32Ptr::ContactConstraintRow32Ptr(agxData::EntityStorage* storage, agx::Index id) : agxData::EntityPtr(storage, id) {}
      AGX_FORCE_INLINE ContactConstraintRow32Ptr::ContactConstraintRow32Ptr(const agxData::EntityPtr& ptr) : agxData::EntityPtr(ptr)
      {
        agxAssertN(!ptr || ptr.isInstanceOf(ContactConstraintRow32Model::instance()),
          "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
          EntityPtr::getModel()->fullPath().c_str(), ContactConstraintRow32Model::instance()->fullPath().c_str());
      }

      AGX_FORCE_INLINE ContactConstraintRow32Ptr::ContactConstraintRow32Ptr(const agxData::EntityInstance& instance) : agxData::EntityPtr(instance)
      {
        agxAssertN(!instance || instance.isInstanceOf(ContactConstraintRow32Model::instance()),
          "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
          EntityPtr::getModel()->fullPath().c_str(), ContactConstraintRow32Model::instance()->fullPath().c_str());
      }

      AGX_FORCE_INLINE ContactConstraintRow32Ptr& ContactConstraintRow32Ptr::operator= (const agxData::EntityPtr& ptr)
      {
        agxData::EntityPtr::operator= (ptr);
        agxAssertN(!ptr || ptr.isInstanceOf(ContactConstraintRow32Model::instance()),
          "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
          EntityPtr::getModel()->fullPath().c_str(), ContactConstraintRow32Model::instance()->fullPath().c_str());
        return *this;
      }

      AGX_FORCE_INLINE ContactConstraintRow32Ptr& ContactConstraintRow32Ptr::operator= (const agxData::EntityInstance& instance)
      {
        agxData::EntityPtr::operator= (instance);
        agxAssertN(!instance || instance.isInstanceOf(ContactConstraintRow32Model::instance()),
          "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
          EntityPtr::getModel()->fullPath().c_str(), ContactConstraintRow32Model::instance()->fullPath().c_str());
        return *this;
      }

      inline ContactConstraintRow32Instance ContactConstraintRow32Ptr::instance() { return agxData::EntityPtr::instance(); }
      inline const ContactConstraintRow32Instance ContactConstraintRow32Ptr::instance() const { return agxData::EntityPtr::instance(); }
      AGX_FORCE_INLINE ContactConstraintRow32Semantics* ContactConstraintRow32Ptr::operator->() { return (ContactConstraintRow32Semantics* )this; }
      AGX_FORCE_INLINE const ContactConstraintRow32Semantics* ContactConstraintRow32Ptr::operator->() const { return (const ContactConstraintRow32Semantics* )this; }
      AGX_FORCE_INLINE ContactConstraintRow32Data* ContactConstraintRow32Ptr::getData() { return static_cast<ContactConstraintRow32Data* >(agxData::EntityPtr::getData()); }
      AGX_FORCE_INLINE const ContactConstraintRow32Data* ContactConstraintRow32Ptr::getData() const { return static_cast<const ContactConstraintRow32Data* >(agxData::EntityPtr::getData()); }

      AGX_FORCE_INLINE agx::Real32& ContactConstraintRow32Ptr::lambda() { verifyIndex(); return getData()->lambda[calculateIndex()]; }
      AGX_FORCE_INLINE agx::Real32 const& ContactConstraintRow32Ptr::lambda() const { verifyIndex(); return getData()->lambda[calculateIndex()]; }

      AGX_FORCE_INLINE agx::Real32& ContactConstraintRow32Ptr::impactLambda() { verifyIndex(); return getData()->impactLambda[calculateIndex()]; }
      AGX_FORCE_INLINE agx::Real32 const& ContactConstraintRow32Ptr::impactLambda() const { verifyIndex(); return getData()->impactLambda[calculateIndex()]; }

      AGX_FORCE_INLINE agx::Real32& ContactConstraintRow32Ptr::rhs() { verifyIndex(); return getData()->rhs[calculateIndex()]; }
      AGX_FORCE_INLINE agx::Real32 const& ContactConstraintRow32Ptr::rhs() const { verifyIndex(); return getData()->rhs[calculateIndex()]; }

      AGX_FORCE_INLINE agx::Real32& ContactConstraintRow32Ptr::epsilon() { verifyIndex(); return getData()->epsilon[calculateIndex()]; }
      AGX_FORCE_INLINE agx::Real32 const& ContactConstraintRow32Ptr::epsilon() const { verifyIndex(); return getData()->epsilon[calculateIndex()]; }

      AGX_FORCE_INLINE agx::Real32& ContactConstraintRow32Ptr::velocity() { verifyIndex(); return getData()->velocity[calculateIndex()]; }
      AGX_FORCE_INLINE agx::Real32 const& ContactConstraintRow32Ptr::velocity() const { verifyIndex(); return getData()->velocity[calculateIndex()]; }

      AGX_FORCE_INLINE agx::Real32& ContactConstraintRow32Ptr::invD() { verifyIndex(); return getData()->invD[calculateIndex()]; }
      AGX_FORCE_INLINE agx::Real32 const& ContactConstraintRow32Ptr::invD() const { verifyIndex(); return getData()->invD[calculateIndex()]; }

      AGX_FORCE_INLINE agx::Real32& ContactConstraintRow32Ptr::violation() { verifyIndex(); return getData()->violation[calculateIndex()]; }
      AGX_FORCE_INLINE agx::Real32 const& ContactConstraintRow32Ptr::violation() const { verifyIndex(); return getData()->violation[calculateIndex()]; }

      AGX_FORCE_INLINE agx::Real& ContactConstraintRow32Ptr::residual() { verifyIndex(); return getData()->residual[calculateIndex()]; }
      AGX_FORCE_INLINE agx::Real const& ContactConstraintRow32Ptr::residual() const { verifyIndex(); return getData()->residual[calculateIndex()]; }

      //-----------------------------------------------------------------------------------------------------
      AGX_FORCE_INLINE ContactConstraintRow32Instance::ContactConstraintRow32Instance() {}
      AGX_FORCE_INLINE ContactConstraintRow32Instance::ContactConstraintRow32Instance(ContactConstraintRow32Data* data, agx::Index index) : agxData::EntityInstance(data, index) {}
      AGX_FORCE_INLINE ContactConstraintRow32Instance::ContactConstraintRow32Instance(agxData::EntityStorage* storage, agx::Index index) : agxData::EntityInstance(storage, index) {}
      AGX_FORCE_INLINE ContactConstraintRow32Instance::ContactConstraintRow32Instance(const agxData::EntityInstance& other) : agxData::EntityInstance(other)
      {
        agxAssertN(!other || other.isInstanceOf(ContactConstraintRow32Model::instance()),
          "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
          EntityInstance::getModel()->fullPath().c_str(), ContactConstraintRow32Model::instance()->fullPath().c_str());
      }

      AGX_FORCE_INLINE ContactConstraintRow32Instance::ContactConstraintRow32Instance(const agxData::EntityPtr& ptr) : agxData::EntityInstance(ptr)
      {
        agxAssertN(!ptr || ptr.isInstanceOf(ContactConstraintRow32Model::instance()),
          "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
          EntityInstance::getModel()->fullPath().c_str(), ContactConstraintRow32Model::instance()->fullPath().c_str());
      }


      AGX_FORCE_INLINE ContactConstraintRow32Data* ContactConstraintRow32Instance::getData() { return static_cast<ContactConstraintRow32Data* >(agxData::EntityInstance::getData()); }
      AGX_FORCE_INLINE const ContactConstraintRow32Data* ContactConstraintRow32Instance::getData() const { return static_cast<const ContactConstraintRow32Data* >(agxData::EntityInstance::getData()); }

      AGX_FORCE_INLINE agx::Real32& ContactConstraintRow32Instance::lambda() { verifyIndex(); return getData()->lambda[getIndex()]; }
      AGX_FORCE_INLINE agx::Real32 const& ContactConstraintRow32Instance::lambda() const { verifyIndex(); return getData()->lambda[getIndex()]; }

      AGX_FORCE_INLINE agx::Real32& ContactConstraintRow32Instance::impactLambda() { verifyIndex(); return getData()->impactLambda[getIndex()]; }
      AGX_FORCE_INLINE agx::Real32 const& ContactConstraintRow32Instance::impactLambda() const { verifyIndex(); return getData()->impactLambda[getIndex()]; }

      AGX_FORCE_INLINE agx::Real32& ContactConstraintRow32Instance::rhs() { verifyIndex(); return getData()->rhs[getIndex()]; }
      AGX_FORCE_INLINE agx::Real32 const& ContactConstraintRow32Instance::rhs() const { verifyIndex(); return getData()->rhs[getIndex()]; }

      AGX_FORCE_INLINE agx::Real32& ContactConstraintRow32Instance::epsilon() { verifyIndex(); return getData()->epsilon[getIndex()]; }
      AGX_FORCE_INLINE agx::Real32 const& ContactConstraintRow32Instance::epsilon() const { verifyIndex(); return getData()->epsilon[getIndex()]; }

      AGX_FORCE_INLINE agx::Real32& ContactConstraintRow32Instance::velocity() { verifyIndex(); return getData()->velocity[getIndex()]; }
      AGX_FORCE_INLINE agx::Real32 const& ContactConstraintRow32Instance::velocity() const { verifyIndex(); return getData()->velocity[getIndex()]; }

      AGX_FORCE_INLINE agx::Real32& ContactConstraintRow32Instance::invD() { verifyIndex(); return getData()->invD[getIndex()]; }
      AGX_FORCE_INLINE agx::Real32 const& ContactConstraintRow32Instance::invD() const { verifyIndex(); return getData()->invD[getIndex()]; }

      AGX_FORCE_INLINE agx::Real32& ContactConstraintRow32Instance::violation() { verifyIndex(); return getData()->violation[getIndex()]; }
      AGX_FORCE_INLINE agx::Real32 const& ContactConstraintRow32Instance::violation() const { verifyIndex(); return getData()->violation[getIndex()]; }

      AGX_FORCE_INLINE agx::Real& ContactConstraintRow32Instance::residual() { verifyIndex(); return getData()->residual[getIndex()]; }
      AGX_FORCE_INLINE agx::Real const& ContactConstraintRow32Instance::residual() const { verifyIndex(); return getData()->residual[getIndex()]; }

      //-----------------------------------------------------------------------------------------------------
      AGX_FORCE_INLINE ContactConstraintRow32Semantics::ContactConstraintRow32Semantics() {}
      //-----------------------------------------------------------------------------------------------------
      DOXYGEN_END_INTERNAL_BLOCK()
    }
  }
}

AGX_TYPE_BINDING(agx::Physics::Particle::ContactConstraintRow32Ptr, "Physics.Particle.ContactConstraintRow32Ptr")
AGX_TYPE_BINDING(agx::Physics::Particle::ContactConstraintRow32Instance, "Physics.Particle.ContactConstraintRow32Instance")

#ifdef _MSC_VER
# pragma warning(pop)
#endif

#undef AGX_ENTITY_WRAPPER
#undef AGX_ENTITY_NAMESPACE
#endif

