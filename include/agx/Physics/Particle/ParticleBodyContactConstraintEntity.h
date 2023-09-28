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

#ifndef GENERATED_AGX_PHYSICS_PARTICLE_PARTICLEBODYCONTACTCONSTRAINT_H_PLUGIN
#define GENERATED_AGX_PHYSICS_PARTICLE_PARTICLEBODYCONTACTCONSTRAINT_H_PLUGIN

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

      class ParticleBodyContactConstraintModel;
      class ParticleBodyContactConstraintData;
      class ParticleBodyContactConstraintPtr;
      class ParticleBodyContactConstraintInstance;
      class ParticleBodyContactConstraintSemantics;


      AGX_DECLARE_POINTER_TYPES(ParticleBodyContactConstraintModel);

      /** 
      Abstract description of the data attributes for the Physics.Particle.ParticleBodyContactConstraint entity.
      */ 
      class AGXPHYSICS_EXPORT ParticleBodyContactConstraintModel : public agx::Physics::InteractionModel
      {
      public:
        typedef ParticleBodyContactConstraintPtr PtrT;

        ParticleBodyContactConstraintModel(const agx::String& name = "ParticleBodyContactConstraint");

        /// \return The entity model singleton.
        static ParticleBodyContactConstraintModel* instance();

        /// Create and return a pointer to a new instance in the default storage for this entity model.
        static ParticleBodyContactConstraintPtr createInstance();

        /// \return The default storage for this entity model.
        static agxData::EntityStorage* defaultStorage();

        /// This is part of internal cleanup and should not be called by users
        virtual void shutdownCleanup() override;



        /* Attributes */
        static agxData::ScalarAttributeT< agx::UInt32 >* particleAttribute;
        static agxData::ScalarAttributeT< agx::UInt32 >* bodyAttribute;
        static agxData::ScalarAttributeT< agx::UInt32 >* contactIndexAttribute;
        static agxData::ScalarAttributeT< agx::UInt32 >* materialIndexAttribute;
        static agxData::ScalarAttributeT< agx::Bool >* isImpactingAttribute;
        static agxData::ScalarAttributeT< agx::ParticleBodyContactJacobian >* GAttribute;
        static agxData::ScalarAttributeT< agx::ParticleBodyContactJacobian >* GMInvAttribute;

      protected:
        virtual ~ParticleBodyContactConstraintModel();
        virtual agxData::EntityData* createData(agxData::EntityStorage* storage) override;
        virtual void configure(agx::TiXmlElement* eEntity) override;
        virtual void initAttributeAccessors() override;
        virtual void construct(agxData::EntityPtr instance) override;
        void construct(agx::Physics::Particle::ParticleBodyContactConstraintPtr particleBodyContactConstraint);
      };


      DOXYGEN_START_INTERNAL_BLOCK()
      #ifndef AGX_PHYSICS_PARTICLE_PARTICLEBODYCONTACTCONSTRAINT_DATA_SET_OVERRIDE
      #define AGX_PHYSICS_PARTICLE_PARTICLEBODYCONTACTCONSTRAINT_DATA_SET
      class AGXPHYSICS_EXPORT ParticleBodyContactConstraintData : public agx::Physics::InteractionData
      {
      public:
        ParticleBodyContactConstraintInstance operator[] (size_t index);

      public:
        agxData::Array< ParticleBodyContactConstraintPtr >& instance;
        agxData::Array< agx::UInt32 > particle;
        agxData::Array< agx::UInt32 > body;
        agxData::Array< agx::UInt32 > contactIndex;
        agxData::Array< agx::UInt32 > materialIndex;
        agxData::Array< agx::Bool > isImpacting;
        agxData::Array< agx::ParticleBodyContactJacobian > G;
        agxData::Array< agx::ParticleBodyContactJacobian > GMInv;

      public:
        typedef agx::UInt32 particleType;
        typedef agx::UInt32 bodyType;
        typedef agx::UInt32 contactIndexType;
        typedef agx::UInt32 materialIndexType;
        typedef agx::Bool isImpactingType;
        typedef agx::ParticleBodyContactJacobian GType;
        typedef agx::ParticleBodyContactJacobian GMInvType;

      public:
        ParticleBodyContactConstraintData(agxData::EntityStorage* storage);
        ParticleBodyContactConstraintData();

      protected:
        virtual ~ParticleBodyContactConstraintData() {}
        virtual void setNumElements(agx::Index numElements) override;

      private:
        ParticleBodyContactConstraintData& operator= (const ParticleBodyContactConstraintData&) { return *this; }

      };
      #endif
      DOXYGEN_END_INTERNAL_BLOCK()


      DOXYGEN_START_INTERNAL_BLOCK()
      class AGXPHYSICS_EXPORT ParticleBodyContactConstraintSemantics : public agx::Physics::InteractionSemantics
      {
      public:

        // Automatic getters
        agx::UInt32 const& getParticle() const;
        agx::UInt32 const& getBody() const;
        agx::UInt32 const& getContactIndex() const;
        agx::UInt32 const& getMaterialIndex() const;
        agx::Bool const& getIsImpacting() const;
        agx::ParticleBodyContactJacobian const& getG() const;
        agx::ParticleBodyContactJacobian const& getGMInv() const;

        // Semantics defined by explicit kernels

        // Automatic setters
        void setParticle(agx::UInt32 const& value);
        void setBody(agx::UInt32 const& value);
        void setContactIndex(agx::UInt32 const& value);
        void setMaterialIndex(agx::UInt32 const& value);
        void setIsImpacting(agx::Bool const& value);
        void setG(agx::ParticleBodyContactJacobian const& value);
        void setGMInv(agx::ParticleBodyContactJacobian const& value);


      protected:
        friend class ParticleBodyContactConstraintPtr;
        friend class ParticleBodyContactConstraintInstance;
        ParticleBodyContactConstraintSemantics();
      };
      DOXYGEN_END_INTERNAL_BLOCK()


      /**
      Pointer to a entity instance of type Physics.Particle.ParticleBodyContactConstraint
      */
      class CALLABLE ParticleBodyContactConstraintPtr : public agx::Physics::InteractionPtr
      {
      public:
        typedef ParticleBodyContactConstraintModel ModelType;
        typedef ParticleBodyContactConstraintData DataType;
        typedef ParticleBodyContactConstraintInstance InstanceType;

      public:
        AGXPHYSICS_EXPORT ParticleBodyContactConstraintPtr();
        AGXPHYSICS_EXPORT ParticleBodyContactConstraintPtr(agxData::EntityStorage* storage, agx::Index id);
        AGXPHYSICS_EXPORT ParticleBodyContactConstraintPtr(const agxData::EntityPtr& ptr);
        AGXPHYSICS_EXPORT ParticleBodyContactConstraintPtr(const agxData::EntityInstance& instance);
        AGXPHYSICS_EXPORT ParticleBodyContactConstraintPtr& operator= (const agxData::EntityPtr& ptr);
        AGXPHYSICS_EXPORT ParticleBodyContactConstraintPtr& operator= (const agxData::EntityInstance& instance);
        AGXPHYSICS_EXPORT ParticleBodyContactConstraintInstance instance();
        AGXPHYSICS_EXPORT const ParticleBodyContactConstraintInstance instance() const;

        AGXPHYSICS_EXPORT ParticleBodyContactConstraintSemantics* operator->();
        AGXPHYSICS_EXPORT const ParticleBodyContactConstraintSemantics* operator->() const;

        ParticleBodyContactConstraintData* getData();
        const ParticleBodyContactConstraintData* getData() const;


        /// \return reference to the particle attribute
        AGXPHYSICS_EXPORT agx::UInt32& particle();
        /// \return const reference to the particle attribute
        AGXPHYSICS_EXPORT agx::UInt32 const& particle() const;

        /// \return reference to the body attribute
        AGXPHYSICS_EXPORT agx::UInt32& body();
        /// \return const reference to the body attribute
        AGXPHYSICS_EXPORT agx::UInt32 const& body() const;

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
        AGXPHYSICS_EXPORT agx::ParticleBodyContactJacobian& G();
        /// \return const reference to the G attribute
        AGXPHYSICS_EXPORT agx::ParticleBodyContactJacobian const& G() const;

        /// \return reference to the GMInv attribute
        AGXPHYSICS_EXPORT agx::ParticleBodyContactJacobian& GMInv();
        /// \return const reference to the GMInv attribute
        AGXPHYSICS_EXPORT agx::ParticleBodyContactJacobian const& GMInv() const;

        DOXYGEN_START_INTERNAL_BLOCK()
        // Only used by tolua!
        #ifdef AGX_GENERATED_TOLUA_WRAPPER
        inline agx::UInt32 const& getParticle() const { return (*this)->getParticle(); }
        inline agx::UInt32 const& getBody() const { return (*this)->getBody(); }
        inline agx::UInt32 const& getContactIndex() const { return (*this)->getContactIndex(); }
        inline agx::UInt32 const& getMaterialIndex() const { return (*this)->getMaterialIndex(); }
        inline agx::Bool const& getIsImpacting() const { return (*this)->getIsImpacting(); }
        inline agx::ParticleBodyContactJacobian const& getG() const { return (*this)->getG(); }
        inline agx::ParticleBodyContactJacobian const& getGMInv() const { return (*this)->getGMInv(); }


        inline void setParticle(agx::UInt32 const& value) { (*this)->setParticle(value); }
        inline void setBody(agx::UInt32 const& value) { (*this)->setBody(value); }
        inline void setContactIndex(agx::UInt32 const& value) { (*this)->setContactIndex(value); }
        inline void setMaterialIndex(agx::UInt32 const& value) { (*this)->setMaterialIndex(value); }
        inline void setIsImpacting(agx::Bool const& value) { (*this)->setIsImpacting(value); }
        inline void setG(agx::ParticleBodyContactJacobian const& value) { (*this)->setG(value); }
        inline void setGMInv(agx::ParticleBodyContactJacobian const& value) { (*this)->setGMInv(value); }
        #endif
        DOXYGEN_END_INTERNAL_BLOCK()

      };


      DOXYGEN_START_INTERNAL_BLOCK()
      class AGXPHYSICS_EXPORT ParticleBodyContactConstraintInstance : public agx::Physics::InteractionInstance
      {
      public:
        ParticleBodyContactConstraintInstance();
        ParticleBodyContactConstraintInstance(ParticleBodyContactConstraintData* data, agx::Index index);
        ParticleBodyContactConstraintInstance(agxData::EntityStorage *storage, agx::Index index);
        ParticleBodyContactConstraintInstance(const agxData::EntityInstance& other);
        ParticleBodyContactConstraintInstance(const agxData::EntityPtr& ptr);

        ParticleBodyContactConstraintData* getData();
        const ParticleBodyContactConstraintData* getData() const;

      public:
        /// \return reference to the particle attribute
        agx::UInt32& particle();
        /// \return const reference to the particle attribute
        agx::UInt32 const& particle() const;

        /// \return reference to the body attribute
        agx::UInt32& body();
        /// \return const reference to the body attribute
        agx::UInt32 const& body() const;

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
        agx::ParticleBodyContactJacobian& G();
        /// \return const reference to the G attribute
        agx::ParticleBodyContactJacobian const& G() const;

        /// \return reference to the GMInv attribute
        agx::ParticleBodyContactJacobian& GMInv();
        /// \return const reference to the GMInv attribute
        agx::ParticleBodyContactJacobian const& GMInv() const;

      };
      DOXYGEN_END_INTERNAL_BLOCK()



      typedef agx::VectorPOD<ParticleBodyContactConstraintPtr> ParticleBodyContactConstraintPtrVector;
      typedef agxData::Array<ParticleBodyContactConstraintPtr> ParticleBodyContactConstraintPtrArray;



      DOXYGEN_START_INTERNAL_BLOCK()
      /* Implementation */
      //-----------------------------------------------------------------------------------------------------
      //-----------------------------------------------------------------------------------------------------
      inline ParticleBodyContactConstraintInstance agx::Physics::Particle::ParticleBodyContactConstraintData::operator[] (size_t index) { return ParticleBodyContactConstraintInstance(this, (agx::Index)index); }
      //-----------------------------------------------------------------------------------------------------
      AGX_FORCE_INLINE ParticleBodyContactConstraintPtr::ParticleBodyContactConstraintPtr() {}
      AGX_FORCE_INLINE ParticleBodyContactConstraintPtr::ParticleBodyContactConstraintPtr(agxData::EntityStorage* storage, agx::Index id) : agx::Physics::InteractionPtr(storage, id) {}
      AGX_FORCE_INLINE ParticleBodyContactConstraintPtr::ParticleBodyContactConstraintPtr(const agxData::EntityPtr& ptr) : agx::Physics::InteractionPtr(ptr)
      {
        agxAssertN(!ptr || ptr.isInstanceOf(ParticleBodyContactConstraintModel::instance()),
          "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
          EntityPtr::getModel()->fullPath().c_str(), ParticleBodyContactConstraintModel::instance()->fullPath().c_str());
      }

      AGX_FORCE_INLINE ParticleBodyContactConstraintPtr::ParticleBodyContactConstraintPtr(const agxData::EntityInstance& instance) : agx::Physics::InteractionPtr(instance)
      {
        agxAssertN(!instance || instance.isInstanceOf(ParticleBodyContactConstraintModel::instance()),
          "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
          EntityPtr::getModel()->fullPath().c_str(), ParticleBodyContactConstraintModel::instance()->fullPath().c_str());
      }

      AGX_FORCE_INLINE ParticleBodyContactConstraintPtr& ParticleBodyContactConstraintPtr::operator= (const agxData::EntityPtr& ptr)
      {
        agxData::EntityPtr::operator= (ptr);
        agxAssertN(!ptr || ptr.isInstanceOf(ParticleBodyContactConstraintModel::instance()),
          "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
          EntityPtr::getModel()->fullPath().c_str(), ParticleBodyContactConstraintModel::instance()->fullPath().c_str());
        return *this;
      }

      AGX_FORCE_INLINE ParticleBodyContactConstraintPtr& ParticleBodyContactConstraintPtr::operator= (const agxData::EntityInstance& instance)
      {
        agxData::EntityPtr::operator= (instance);
        agxAssertN(!instance || instance.isInstanceOf(ParticleBodyContactConstraintModel::instance()),
          "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
          EntityPtr::getModel()->fullPath().c_str(), ParticleBodyContactConstraintModel::instance()->fullPath().c_str());
        return *this;
      }

      inline ParticleBodyContactConstraintInstance ParticleBodyContactConstraintPtr::instance() { return agxData::EntityPtr::instance(); }
      inline const ParticleBodyContactConstraintInstance ParticleBodyContactConstraintPtr::instance() const { return agxData::EntityPtr::instance(); }
      AGX_FORCE_INLINE ParticleBodyContactConstraintSemantics* ParticleBodyContactConstraintPtr::operator->() { return (ParticleBodyContactConstraintSemantics* )this; }
      AGX_FORCE_INLINE const ParticleBodyContactConstraintSemantics* ParticleBodyContactConstraintPtr::operator->() const { return (const ParticleBodyContactConstraintSemantics* )this; }
      AGX_FORCE_INLINE ParticleBodyContactConstraintData* ParticleBodyContactConstraintPtr::getData() { return static_cast<ParticleBodyContactConstraintData* >(agxData::EntityPtr::getData()); }
      AGX_FORCE_INLINE const ParticleBodyContactConstraintData* ParticleBodyContactConstraintPtr::getData() const { return static_cast<const ParticleBodyContactConstraintData* >(agxData::EntityPtr::getData()); }

      AGX_FORCE_INLINE agx::UInt32& ParticleBodyContactConstraintPtr::particle() { verifyIndex(); return getData()->particle[calculateIndex()]; }
      AGX_FORCE_INLINE agx::UInt32 const& ParticleBodyContactConstraintPtr::particle() const { verifyIndex(); return getData()->particle[calculateIndex()]; }

      AGX_FORCE_INLINE agx::UInt32& ParticleBodyContactConstraintPtr::body() { verifyIndex(); return getData()->body[calculateIndex()]; }
      AGX_FORCE_INLINE agx::UInt32 const& ParticleBodyContactConstraintPtr::body() const { verifyIndex(); return getData()->body[calculateIndex()]; }

      AGX_FORCE_INLINE agx::UInt32& ParticleBodyContactConstraintPtr::contactIndex() { verifyIndex(); return getData()->contactIndex[calculateIndex()]; }
      AGX_FORCE_INLINE agx::UInt32 const& ParticleBodyContactConstraintPtr::contactIndex() const { verifyIndex(); return getData()->contactIndex[calculateIndex()]; }

      AGX_FORCE_INLINE agx::UInt32& ParticleBodyContactConstraintPtr::materialIndex() { verifyIndex(); return getData()->materialIndex[calculateIndex()]; }
      AGX_FORCE_INLINE agx::UInt32 const& ParticleBodyContactConstraintPtr::materialIndex() const { verifyIndex(); return getData()->materialIndex[calculateIndex()]; }

      AGX_FORCE_INLINE agx::Bool& ParticleBodyContactConstraintPtr::isImpacting() { verifyIndex(); return getData()->isImpacting[calculateIndex()]; }
      AGX_FORCE_INLINE agx::Bool const& ParticleBodyContactConstraintPtr::isImpacting() const { verifyIndex(); return getData()->isImpacting[calculateIndex()]; }

      AGX_FORCE_INLINE agx::ParticleBodyContactJacobian& ParticleBodyContactConstraintPtr::G() { verifyIndex(); return getData()->G[calculateIndex()]; }
      AGX_FORCE_INLINE agx::ParticleBodyContactJacobian const& ParticleBodyContactConstraintPtr::G() const { verifyIndex(); return getData()->G[calculateIndex()]; }

      AGX_FORCE_INLINE agx::ParticleBodyContactJacobian& ParticleBodyContactConstraintPtr::GMInv() { verifyIndex(); return getData()->GMInv[calculateIndex()]; }
      AGX_FORCE_INLINE agx::ParticleBodyContactJacobian const& ParticleBodyContactConstraintPtr::GMInv() const { verifyIndex(); return getData()->GMInv[calculateIndex()]; }

      //-----------------------------------------------------------------------------------------------------
      AGX_FORCE_INLINE ParticleBodyContactConstraintInstance::ParticleBodyContactConstraintInstance() {}
      AGX_FORCE_INLINE ParticleBodyContactConstraintInstance::ParticleBodyContactConstraintInstance(ParticleBodyContactConstraintData* data, agx::Index index) : agx::Physics::InteractionInstance(data, index) {}
      AGX_FORCE_INLINE ParticleBodyContactConstraintInstance::ParticleBodyContactConstraintInstance(agxData::EntityStorage* storage, agx::Index index) : agx::Physics::InteractionInstance(storage, index) {}
      AGX_FORCE_INLINE ParticleBodyContactConstraintInstance::ParticleBodyContactConstraintInstance(const agxData::EntityInstance& other) : agx::Physics::InteractionInstance(other)
      {
        agxAssertN(!other || other.isInstanceOf(ParticleBodyContactConstraintModel::instance()),
          "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
          EntityInstance::getModel()->fullPath().c_str(), ParticleBodyContactConstraintModel::instance()->fullPath().c_str());
      }

      AGX_FORCE_INLINE ParticleBodyContactConstraintInstance::ParticleBodyContactConstraintInstance(const agxData::EntityPtr& ptr) : agx::Physics::InteractionInstance(ptr)
      {
        agxAssertN(!ptr || ptr.isInstanceOf(ParticleBodyContactConstraintModel::instance()),
          "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
          EntityInstance::getModel()->fullPath().c_str(), ParticleBodyContactConstraintModel::instance()->fullPath().c_str());
      }


      AGX_FORCE_INLINE ParticleBodyContactConstraintData* ParticleBodyContactConstraintInstance::getData() { return static_cast<ParticleBodyContactConstraintData* >(agxData::EntityInstance::getData()); }
      AGX_FORCE_INLINE const ParticleBodyContactConstraintData* ParticleBodyContactConstraintInstance::getData() const { return static_cast<const ParticleBodyContactConstraintData* >(agxData::EntityInstance::getData()); }

      AGX_FORCE_INLINE agx::UInt32& ParticleBodyContactConstraintInstance::particle() { verifyIndex(); return getData()->particle[getIndex()]; }
      AGX_FORCE_INLINE agx::UInt32 const& ParticleBodyContactConstraintInstance::particle() const { verifyIndex(); return getData()->particle[getIndex()]; }

      AGX_FORCE_INLINE agx::UInt32& ParticleBodyContactConstraintInstance::body() { verifyIndex(); return getData()->body[getIndex()]; }
      AGX_FORCE_INLINE agx::UInt32 const& ParticleBodyContactConstraintInstance::body() const { verifyIndex(); return getData()->body[getIndex()]; }

      AGX_FORCE_INLINE agx::UInt32& ParticleBodyContactConstraintInstance::contactIndex() { verifyIndex(); return getData()->contactIndex[getIndex()]; }
      AGX_FORCE_INLINE agx::UInt32 const& ParticleBodyContactConstraintInstance::contactIndex() const { verifyIndex(); return getData()->contactIndex[getIndex()]; }

      AGX_FORCE_INLINE agx::UInt32& ParticleBodyContactConstraintInstance::materialIndex() { verifyIndex(); return getData()->materialIndex[getIndex()]; }
      AGX_FORCE_INLINE agx::UInt32 const& ParticleBodyContactConstraintInstance::materialIndex() const { verifyIndex(); return getData()->materialIndex[getIndex()]; }

      AGX_FORCE_INLINE agx::Bool& ParticleBodyContactConstraintInstance::isImpacting() { verifyIndex(); return getData()->isImpacting[getIndex()]; }
      AGX_FORCE_INLINE agx::Bool const& ParticleBodyContactConstraintInstance::isImpacting() const { verifyIndex(); return getData()->isImpacting[getIndex()]; }

      AGX_FORCE_INLINE agx::ParticleBodyContactJacobian& ParticleBodyContactConstraintInstance::G() { verifyIndex(); return getData()->G[getIndex()]; }
      AGX_FORCE_INLINE agx::ParticleBodyContactJacobian const& ParticleBodyContactConstraintInstance::G() const { verifyIndex(); return getData()->G[getIndex()]; }

      AGX_FORCE_INLINE agx::ParticleBodyContactJacobian& ParticleBodyContactConstraintInstance::GMInv() { verifyIndex(); return getData()->GMInv[getIndex()]; }
      AGX_FORCE_INLINE agx::ParticleBodyContactJacobian const& ParticleBodyContactConstraintInstance::GMInv() const { verifyIndex(); return getData()->GMInv[getIndex()]; }

      //-----------------------------------------------------------------------------------------------------
      AGX_FORCE_INLINE ParticleBodyContactConstraintSemantics::ParticleBodyContactConstraintSemantics() {}
      //-----------------------------------------------------------------------------------------------------
      DOXYGEN_END_INTERNAL_BLOCK()
    }
  }
}

AGX_TYPE_BINDING(agx::Physics::Particle::ParticleBodyContactConstraintPtr, "Physics.Particle.ParticleBodyContactConstraintPtr")
AGX_TYPE_BINDING(agx::Physics::Particle::ParticleBodyContactConstraintInstance, "Physics.Particle.ParticleBodyContactConstraintInstance")

#ifdef _MSC_VER
# pragma warning(pop)
#endif

#undef AGX_ENTITY_WRAPPER
#undef AGX_ENTITY_NAMESPACE
#endif

