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

#ifndef GENERATED_AGX_PHYSICS_PARTICLE_CONTACTCONSTRAINT_H_PLUGIN
#define GENERATED_AGX_PHYSICS_PARTICLE_CONTACTCONSTRAINT_H_PLUGIN

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
#include <agx/Jacobian.h>


namespace agx
{
  namespace Physics
  {
    namespace Particle
    {

      class ContactConstraintModel;
      class ContactConstraintData;
      class ContactConstraintPtr;
      class ContactConstraintInstance;
      class ContactConstraintSemantics;


      AGX_DECLARE_POINTER_TYPES(ContactConstraintModel);

      /** 
      Abstract description of the data attributes for the Physics.Particle.ContactConstraint entity.
      */ 
      class AGXPHYSICS_EXPORT ContactConstraintModel : public agx::Physics::InteractionModel
      {
      public:
        typedef ContactConstraintPtr PtrT;

        ContactConstraintModel(const agx::String& name = "ContactConstraint");

        /// \return The entity model singleton.
        static ContactConstraintModel* instance();

        /// Create and return a pointer to a new instance in the default storage for this entity model.
        static ContactConstraintPtr createInstance();

        /// \return The default storage for this entity model.
        static agxData::EntityStorage* defaultStorage();

        /// This is part of internal cleanup and should not be called by users
        virtual void shutdownCleanup() override;



        /* Attributes */
        static agxData::ScalarAttributeT< agx::UInt32 >* particle1Attribute;
        static agxData::ScalarAttributeT< agx::UInt32 >* particle2Attribute;
        static agxData::ScalarAttributeT< agx::UInt32 >* contactIndexAttribute;
        static agxData::ScalarAttributeT< agx::UInt32 >* materialIndexAttribute;
        static agxData::ScalarAttributeT< agx::Bool >* isImpactingAttribute;
        static agxData::ScalarAttributeT< agx::ParticleParticleContactJacobian >* GAttribute;
        static agxData::ScalarAttributeT< agx::ParticleParticleContactJacobian >* GMInvAttribute;

      protected:
        virtual ~ContactConstraintModel();
        virtual agxData::EntityData* createData(agxData::EntityStorage* storage) override;
        virtual void configure(agx::TiXmlElement* eEntity) override;
        virtual void initAttributeAccessors() override;
        virtual void construct(agxData::EntityPtr instance) override;
        void construct(agx::Physics::Particle::ContactConstraintPtr contactConstraint);
      };


      DOXYGEN_START_INTERNAL_BLOCK()
      #ifndef AGX_PHYSICS_PARTICLE_CONTACTCONSTRAINT_DATA_SET_OVERRIDE
      #define AGX_PHYSICS_PARTICLE_CONTACTCONSTRAINT_DATA_SET
      class AGXPHYSICS_EXPORT ContactConstraintData : public agx::Physics::InteractionData
      {
      public:
        ContactConstraintInstance operator[] (size_t index);

      public:
        agxData::Array< ContactConstraintPtr >& instance;
        agxData::Array< agx::UInt32 > particle1;
        agxData::Array< agx::UInt32 > particle2;
        agxData::Array< agx::UInt32 > contactIndex;
        agxData::Array< agx::UInt32 > materialIndex;
        agxData::Array< agx::Bool > isImpacting;
        agxData::Array< agx::ParticleParticleContactJacobian > G;
        agxData::Array< agx::ParticleParticleContactJacobian > GMInv;

      public:
        typedef agx::UInt32 particle1Type;
        typedef agx::UInt32 particle2Type;
        typedef agx::UInt32 contactIndexType;
        typedef agx::UInt32 materialIndexType;
        typedef agx::Bool isImpactingType;
        typedef agx::ParticleParticleContactJacobian GType;
        typedef agx::ParticleParticleContactJacobian GMInvType;

      public:
        ContactConstraintData(agxData::EntityStorage* storage);
        ContactConstraintData();

      protected:
        virtual ~ContactConstraintData() {}
        virtual void setNumElements(agx::Index numElements) override;

      private:
        ContactConstraintData& operator= (const ContactConstraintData&) { return *this; }

      };
      #endif
      DOXYGEN_END_INTERNAL_BLOCK()


      DOXYGEN_START_INTERNAL_BLOCK()
      class AGXPHYSICS_EXPORT ContactConstraintSemantics : public agx::Physics::InteractionSemantics
      {
      public:

        // Automatic getters
        agx::UInt32 const& getParticle1() const;
        agx::UInt32 const& getParticle2() const;
        agx::UInt32 const& getContactIndex() const;
        agx::UInt32 const& getMaterialIndex() const;
        agx::Bool const& getIsImpacting() const;
        agx::ParticleParticleContactJacobian const& getG() const;
        agx::ParticleParticleContactJacobian const& getGMInv() const;

        // Semantics defined by explicit kernels

        // Automatic setters
        void setParticle1(agx::UInt32 const& value);
        void setParticle2(agx::UInt32 const& value);
        void setContactIndex(agx::UInt32 const& value);
        void setMaterialIndex(agx::UInt32 const& value);
        void setIsImpacting(agx::Bool const& value);
        void setG(agx::ParticleParticleContactJacobian const& value);
        void setGMInv(agx::ParticleParticleContactJacobian const& value);


      protected:
        friend class ContactConstraintPtr;
        friend class ContactConstraintInstance;
        ContactConstraintSemantics();
      };
      DOXYGEN_END_INTERNAL_BLOCK()


      /**
      Pointer to a entity instance of type Physics.Particle.ContactConstraint
      */
      class CALLABLE ContactConstraintPtr : public agx::Physics::InteractionPtr
      {
      public:
        typedef ContactConstraintModel ModelType;
        typedef ContactConstraintData DataType;
        typedef ContactConstraintInstance InstanceType;

      public:
        AGXPHYSICS_EXPORT ContactConstraintPtr();
        AGXPHYSICS_EXPORT ContactConstraintPtr(agxData::EntityStorage* storage, agx::Index id);
        AGXPHYSICS_EXPORT ContactConstraintPtr(const agxData::EntityPtr& ptr);
        AGXPHYSICS_EXPORT ContactConstraintPtr(const agxData::EntityInstance& instance);
        AGXPHYSICS_EXPORT ContactConstraintPtr& operator= (const agxData::EntityPtr& ptr);
        AGXPHYSICS_EXPORT ContactConstraintPtr& operator= (const agxData::EntityInstance& instance);
        AGXPHYSICS_EXPORT ContactConstraintInstance instance();
        AGXPHYSICS_EXPORT const ContactConstraintInstance instance() const;

        AGXPHYSICS_EXPORT ContactConstraintSemantics* operator->();
        AGXPHYSICS_EXPORT const ContactConstraintSemantics* operator->() const;

        ContactConstraintData* getData();
        const ContactConstraintData* getData() const;


        /// \return reference to the particle1 attribute
        AGXPHYSICS_EXPORT agx::UInt32& particle1();
        /// \return const reference to the particle1 attribute
        AGXPHYSICS_EXPORT agx::UInt32 const& particle1() const;

        /// \return reference to the particle2 attribute
        AGXPHYSICS_EXPORT agx::UInt32& particle2();
        /// \return const reference to the particle2 attribute
        AGXPHYSICS_EXPORT agx::UInt32 const& particle2() const;

        /// \return reference to the contactIndex attribute
        AGXPHYSICS_EXPORT agx::UInt32& contactIndex();
        /// \return const reference to the contactIndex attribute
        AGXPHYSICS_EXPORT agx::UInt32 const& contactIndex() const;

        /// \return reference to the materialIndex attribute
        AGXPHYSICS_EXPORT agx::UInt32& materialIndex();
        /// \return const reference to the materialIndex attribute
        AGXPHYSICS_EXPORT agx::UInt32 const& materialIndex() const;

        /// \return reference to the isImpacting attribute
        AGXPHYSICS_EXPORT agx::Bool& isImpacting();
        /// \return const reference to the isImpacting attribute
        AGXPHYSICS_EXPORT agx::Bool const& isImpacting() const;

        /// \return reference to the G attribute
        AGXPHYSICS_EXPORT agx::ParticleParticleContactJacobian& G();
        /// \return const reference to the G attribute
        AGXPHYSICS_EXPORT agx::ParticleParticleContactJacobian const& G() const;

        /// \return reference to the GMInv attribute
        AGXPHYSICS_EXPORT agx::ParticleParticleContactJacobian& GMInv();
        /// \return const reference to the GMInv attribute
        AGXPHYSICS_EXPORT agx::ParticleParticleContactJacobian const& GMInv() const;

        DOXYGEN_START_INTERNAL_BLOCK()
        // Only used by tolua!
        #ifdef AGX_GENERATED_TOLUA_WRAPPER
        inline agx::UInt32 const& getParticle1() const { return (*this)->getParticle1(); }
        inline agx::UInt32 const& getParticle2() const { return (*this)->getParticle2(); }
        inline agx::UInt32 const& getContactIndex() const { return (*this)->getContactIndex(); }
        inline agx::UInt32 const& getMaterialIndex() const { return (*this)->getMaterialIndex(); }
        inline agx::Bool const& getIsImpacting() const { return (*this)->getIsImpacting(); }
        inline agx::ParticleParticleContactJacobian const& getG() const { return (*this)->getG(); }
        inline agx::ParticleParticleContactJacobian const& getGMInv() const { return (*this)->getGMInv(); }


        inline void setParticle1(agx::UInt32 const& value) { (*this)->setParticle1(value); }
        inline void setParticle2(agx::UInt32 const& value) { (*this)->setParticle2(value); }
        inline void setContactIndex(agx::UInt32 const& value) { (*this)->setContactIndex(value); }
        inline void setMaterialIndex(agx::UInt32 const& value) { (*this)->setMaterialIndex(value); }
        inline void setIsImpacting(agx::Bool const& value) { (*this)->setIsImpacting(value); }
        inline void setG(agx::ParticleParticleContactJacobian const& value) { (*this)->setG(value); }
        inline void setGMInv(agx::ParticleParticleContactJacobian const& value) { (*this)->setGMInv(value); }
        #endif
        DOXYGEN_END_INTERNAL_BLOCK()

      };


      DOXYGEN_START_INTERNAL_BLOCK()
      class AGXPHYSICS_EXPORT ContactConstraintInstance : public agx::Physics::InteractionInstance
      {
      public:
        ContactConstraintInstance();
        ContactConstraintInstance(ContactConstraintData* data, agx::Index index);
        ContactConstraintInstance(agxData::EntityStorage *storage, agx::Index index);
        ContactConstraintInstance(const agxData::EntityInstance& other);
        ContactConstraintInstance(const agxData::EntityPtr& ptr);

        ContactConstraintData* getData();
        const ContactConstraintData* getData() const;

      public:
        /// \return reference to the particle1 attribute
        agx::UInt32& particle1();
        /// \return const reference to the particle1 attribute
        agx::UInt32 const& particle1() const;

        /// \return reference to the particle2 attribute
        agx::UInt32& particle2();
        /// \return const reference to the particle2 attribute
        agx::UInt32 const& particle2() const;

        /// \return reference to the contactIndex attribute
        agx::UInt32& contactIndex();
        /// \return const reference to the contactIndex attribute
        agx::UInt32 const& contactIndex() const;

        /// \return reference to the materialIndex attribute
        agx::UInt32& materialIndex();
        /// \return const reference to the materialIndex attribute
        agx::UInt32 const& materialIndex() const;

        /// \return reference to the isImpacting attribute
        agx::Bool& isImpacting();
        /// \return const reference to the isImpacting attribute
        agx::Bool const& isImpacting() const;

        /// \return reference to the G attribute
        agx::ParticleParticleContactJacobian& G();
        /// \return const reference to the G attribute
        agx::ParticleParticleContactJacobian const& G() const;

        /// \return reference to the GMInv attribute
        agx::ParticleParticleContactJacobian& GMInv();
        /// \return const reference to the GMInv attribute
        agx::ParticleParticleContactJacobian const& GMInv() const;

      };
      DOXYGEN_END_INTERNAL_BLOCK()



      typedef agx::VectorPOD<ContactConstraintPtr> ContactConstraintPtrVector;
      typedef agxData::Array<ContactConstraintPtr> ContactConstraintPtrArray;



      DOXYGEN_START_INTERNAL_BLOCK()
      /* Implementation */
      //-----------------------------------------------------------------------------------------------------
      //-----------------------------------------------------------------------------------------------------
      inline ContactConstraintInstance agx::Physics::Particle::ContactConstraintData::operator[] (size_t index) { return ContactConstraintInstance(this, (agx::Index)index); }
      //-----------------------------------------------------------------------------------------------------
      AGX_FORCE_INLINE ContactConstraintPtr::ContactConstraintPtr() {}
      AGX_FORCE_INLINE ContactConstraintPtr::ContactConstraintPtr(agxData::EntityStorage* storage, agx::Index id) : agx::Physics::InteractionPtr(storage, id) {}
      AGX_FORCE_INLINE ContactConstraintPtr::ContactConstraintPtr(const agxData::EntityPtr& ptr) : agx::Physics::InteractionPtr(ptr)
      {
        agxAssertN(!ptr || ptr.isInstanceOf(ContactConstraintModel::instance()),
          "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
          EntityPtr::getModel()->fullPath().c_str(), ContactConstraintModel::instance()->fullPath().c_str());
      }

      AGX_FORCE_INLINE ContactConstraintPtr::ContactConstraintPtr(const agxData::EntityInstance& instance) : agx::Physics::InteractionPtr(instance)
      {
        agxAssertN(!instance || instance.isInstanceOf(ContactConstraintModel::instance()),
          "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
          EntityPtr::getModel()->fullPath().c_str(), ContactConstraintModel::instance()->fullPath().c_str());
      }

      AGX_FORCE_INLINE ContactConstraintPtr& ContactConstraintPtr::operator= (const agxData::EntityPtr& ptr)
      {
        agxData::EntityPtr::operator= (ptr);
        agxAssertN(!ptr || ptr.isInstanceOf(ContactConstraintModel::instance()),
          "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
          EntityPtr::getModel()->fullPath().c_str(), ContactConstraintModel::instance()->fullPath().c_str());
        return *this;
      }

      AGX_FORCE_INLINE ContactConstraintPtr& ContactConstraintPtr::operator= (const agxData::EntityInstance& instance)
      {
        agxData::EntityPtr::operator= (instance);
        agxAssertN(!instance || instance.isInstanceOf(ContactConstraintModel::instance()),
          "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
          EntityPtr::getModel()->fullPath().c_str(), ContactConstraintModel::instance()->fullPath().c_str());
        return *this;
      }

      inline ContactConstraintInstance ContactConstraintPtr::instance() { return agxData::EntityPtr::instance(); }
      inline const ContactConstraintInstance ContactConstraintPtr::instance() const { return agxData::EntityPtr::instance(); }
      AGX_FORCE_INLINE ContactConstraintSemantics* ContactConstraintPtr::operator->() { return (ContactConstraintSemantics* )this; }
      AGX_FORCE_INLINE const ContactConstraintSemantics* ContactConstraintPtr::operator->() const { return (const ContactConstraintSemantics* )this; }
      AGX_FORCE_INLINE ContactConstraintData* ContactConstraintPtr::getData() { return static_cast<ContactConstraintData* >(agxData::EntityPtr::getData()); }
      AGX_FORCE_INLINE const ContactConstraintData* ContactConstraintPtr::getData() const { return static_cast<const ContactConstraintData* >(agxData::EntityPtr::getData()); }

      AGX_FORCE_INLINE agx::UInt32& ContactConstraintPtr::particle1() { verifyIndex(); return getData()->particle1[calculateIndex()]; }
      AGX_FORCE_INLINE agx::UInt32 const& ContactConstraintPtr::particle1() const { verifyIndex(); return getData()->particle1[calculateIndex()]; }

      AGX_FORCE_INLINE agx::UInt32& ContactConstraintPtr::particle2() { verifyIndex(); return getData()->particle2[calculateIndex()]; }
      AGX_FORCE_INLINE agx::UInt32 const& ContactConstraintPtr::particle2() const { verifyIndex(); return getData()->particle2[calculateIndex()]; }

      AGX_FORCE_INLINE agx::UInt32& ContactConstraintPtr::contactIndex() { verifyIndex(); return getData()->contactIndex[calculateIndex()]; }
      AGX_FORCE_INLINE agx::UInt32 const& ContactConstraintPtr::contactIndex() const { verifyIndex(); return getData()->contactIndex[calculateIndex()]; }

      AGX_FORCE_INLINE agx::UInt32& ContactConstraintPtr::materialIndex() { verifyIndex(); return getData()->materialIndex[calculateIndex()]; }
      AGX_FORCE_INLINE agx::UInt32 const& ContactConstraintPtr::materialIndex() const { verifyIndex(); return getData()->materialIndex[calculateIndex()]; }

      AGX_FORCE_INLINE agx::Bool& ContactConstraintPtr::isImpacting() { verifyIndex(); return getData()->isImpacting[calculateIndex()]; }
      AGX_FORCE_INLINE agx::Bool const& ContactConstraintPtr::isImpacting() const { verifyIndex(); return getData()->isImpacting[calculateIndex()]; }

      AGX_FORCE_INLINE agx::ParticleParticleContactJacobian& ContactConstraintPtr::G() { verifyIndex(); return getData()->G[calculateIndex()]; }
      AGX_FORCE_INLINE agx::ParticleParticleContactJacobian const& ContactConstraintPtr::G() const { verifyIndex(); return getData()->G[calculateIndex()]; }

      AGX_FORCE_INLINE agx::ParticleParticleContactJacobian& ContactConstraintPtr::GMInv() { verifyIndex(); return getData()->GMInv[calculateIndex()]; }
      AGX_FORCE_INLINE agx::ParticleParticleContactJacobian const& ContactConstraintPtr::GMInv() const { verifyIndex(); return getData()->GMInv[calculateIndex()]; }

      //-----------------------------------------------------------------------------------------------------
      AGX_FORCE_INLINE ContactConstraintInstance::ContactConstraintInstance() {}
      AGX_FORCE_INLINE ContactConstraintInstance::ContactConstraintInstance(ContactConstraintData* data, agx::Index index) : agx::Physics::InteractionInstance(data, index) {}
      AGX_FORCE_INLINE ContactConstraintInstance::ContactConstraintInstance(agxData::EntityStorage* storage, agx::Index index) : agx::Physics::InteractionInstance(storage, index) {}
      AGX_FORCE_INLINE ContactConstraintInstance::ContactConstraintInstance(const agxData::EntityInstance& other) : agx::Physics::InteractionInstance(other)
      {
        agxAssertN(!other || other.isInstanceOf(ContactConstraintModel::instance()),
          "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
          EntityInstance::getModel()->fullPath().c_str(), ContactConstraintModel::instance()->fullPath().c_str());
      }

      AGX_FORCE_INLINE ContactConstraintInstance::ContactConstraintInstance(const agxData::EntityPtr& ptr) : agx::Physics::InteractionInstance(ptr)
      {
        agxAssertN(!ptr || ptr.isInstanceOf(ContactConstraintModel::instance()),
          "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
          EntityInstance::getModel()->fullPath().c_str(), ContactConstraintModel::instance()->fullPath().c_str());
      }


      AGX_FORCE_INLINE ContactConstraintData* ContactConstraintInstance::getData() { return static_cast<ContactConstraintData* >(agxData::EntityInstance::getData()); }
      AGX_FORCE_INLINE const ContactConstraintData* ContactConstraintInstance::getData() const { return static_cast<const ContactConstraintData* >(agxData::EntityInstance::getData()); }

      AGX_FORCE_INLINE agx::UInt32& ContactConstraintInstance::particle1() { verifyIndex(); return getData()->particle1[getIndex()]; }
      AGX_FORCE_INLINE agx::UInt32 const& ContactConstraintInstance::particle1() const { verifyIndex(); return getData()->particle1[getIndex()]; }

      AGX_FORCE_INLINE agx::UInt32& ContactConstraintInstance::particle2() { verifyIndex(); return getData()->particle2[getIndex()]; }
      AGX_FORCE_INLINE agx::UInt32 const& ContactConstraintInstance::particle2() const { verifyIndex(); return getData()->particle2[getIndex()]; }

      AGX_FORCE_INLINE agx::UInt32& ContactConstraintInstance::contactIndex() { verifyIndex(); return getData()->contactIndex[getIndex()]; }
      AGX_FORCE_INLINE agx::UInt32 const& ContactConstraintInstance::contactIndex() const { verifyIndex(); return getData()->contactIndex[getIndex()]; }

      AGX_FORCE_INLINE agx::UInt32& ContactConstraintInstance::materialIndex() { verifyIndex(); return getData()->materialIndex[getIndex()]; }
      AGX_FORCE_INLINE agx::UInt32 const& ContactConstraintInstance::materialIndex() const { verifyIndex(); return getData()->materialIndex[getIndex()]; }

      AGX_FORCE_INLINE agx::Bool& ContactConstraintInstance::isImpacting() { verifyIndex(); return getData()->isImpacting[getIndex()]; }
      AGX_FORCE_INLINE agx::Bool const& ContactConstraintInstance::isImpacting() const { verifyIndex(); return getData()->isImpacting[getIndex()]; }

      AGX_FORCE_INLINE agx::ParticleParticleContactJacobian& ContactConstraintInstance::G() { verifyIndex(); return getData()->G[getIndex()]; }
      AGX_FORCE_INLINE agx::ParticleParticleContactJacobian const& ContactConstraintInstance::G() const { verifyIndex(); return getData()->G[getIndex()]; }

      AGX_FORCE_INLINE agx::ParticleParticleContactJacobian& ContactConstraintInstance::GMInv() { verifyIndex(); return getData()->GMInv[getIndex()]; }
      AGX_FORCE_INLINE agx::ParticleParticleContactJacobian const& ContactConstraintInstance::GMInv() const { verifyIndex(); return getData()->GMInv[getIndex()]; }

      //-----------------------------------------------------------------------------------------------------
      AGX_FORCE_INLINE ContactConstraintSemantics::ContactConstraintSemantics() {}
      //-----------------------------------------------------------------------------------------------------
      DOXYGEN_END_INTERNAL_BLOCK()
    }
  }
}

AGX_TYPE_BINDING(agx::Physics::Particle::ContactConstraintPtr, "Physics.Particle.ContactConstraintPtr")
AGX_TYPE_BINDING(agx::Physics::Particle::ContactConstraintInstance, "Physics.Particle.ContactConstraintInstance")

#ifdef _MSC_VER
# pragma warning(pop)
#endif

#undef AGX_ENTITY_WRAPPER
#undef AGX_ENTITY_NAMESPACE
#endif

