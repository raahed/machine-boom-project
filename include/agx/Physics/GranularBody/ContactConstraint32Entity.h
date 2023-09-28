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

#ifndef GENERATED_AGX_PHYSICS_GRANULARBODY_CONTACTCONSTRAINT32_H_PLUGIN
#define GENERATED_AGX_PHYSICS_GRANULARBODY_CONTACTCONSTRAINT32_H_PLUGIN

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
#include <agx/Jacobian.h>


namespace agx
{
  namespace Physics
  {
    namespace GranularBody
    {

      class ContactConstraint32Model;
      class ContactConstraint32Data;
      class ContactConstraint32Ptr;
      class ContactConstraint32Instance;
      class ContactConstraint32Semantics;


      AGX_DECLARE_POINTER_TYPES(ContactConstraint32Model);

      /** 
      Abstract description of the data attributes for the Physics.GranularBody.ContactConstraint32 entity.
      */ 
      class AGXPHYSICS_EXPORT ContactConstraint32Model : public agxData::EntityModel
      {
      public:
        typedef ContactConstraint32Ptr PtrT;

        ContactConstraint32Model(const agx::String& name = "ContactConstraint32");

        /// \return The entity model singleton.
        static ContactConstraint32Model* instance();

        /// Create and return a pointer to a new instance in the default storage for this entity model.
        static ContactConstraint32Ptr createInstance();

        /// \return The default storage for this entity model.
        static agxData::EntityStorage* defaultStorage();

        /// This is part of internal cleanup and should not be called by users
        virtual void shutdownCleanup() override;



        /* Attributes */
        static agxData::ScalarAttributeT< agx::UInt32 >* body1Attribute;
        static agxData::ScalarAttributeT< agx::UInt32 >* body2Attribute;
        static agxData::ScalarAttributeT< agx::UInt32 >* solveBody1Attribute;
        static agxData::ScalarAttributeT< agx::UInt32 >* solveBody2Attribute;
        static agxData::ScalarAttributeT< agx::UInt32 >* cachedContactIdAttribute;
        static agxData::ScalarAttributeT< agx::UInt32 >* contactIndexAttribute;
        static agxData::ScalarAttributeT< agx::UInt32 >* jacobianIndexAttribute;
        static agxData::ScalarAttributeT< agx::UInt32 >* rowIndexAttribute;
        static agxData::ScalarAttributeT< agx::UInt32 >* materialIndexAttribute;
        static agxData::ScalarAttributeT< agx::UInt32 >* iterationCountAttribute;
        static agxData::ScalarAttributeT< agx::Bool >* isImpactingAttribute;
        static agxData::ScalarAttributeT< agx::UInt >* activeCountAttribute;
        static agxData::ScalarAttributeT< agx::Real32 >* restingComplianceAttribute;
        static agxData::ScalarAttributeT< agx::Real32 >* charContactTimeAttribute;
        static agxData::ScalarAttributeT< agx::Real32 >* charMassAttribute;
        static agxData::ScalarAttributeT< agx::JacobianMeta32 >* GMetaAttribute;
        static agxData::ScalarAttributeT< agx::Real32 >* nonlinearMultiplierAttribute;
        static agxData::ScalarAttributeT< agx::Real32 >* rollingResistanceMuAttribute;
        static agxData::ScalarAttributeT< agx::Real32 >* twistLimitMultiplierAttribute;

      protected:
        virtual ~ContactConstraint32Model();
        virtual agxData::EntityData* createData(agxData::EntityStorage* storage) override;
        virtual void configure(agx::TiXmlElement* eEntity) override;
        virtual void initAttributeAccessors() override;
        virtual void construct(agxData::EntityPtr instance) override;
        void construct(agx::Physics::GranularBody::ContactConstraint32Ptr contactConstraint32);
      };


      DOXYGEN_START_INTERNAL_BLOCK()
      #ifndef AGX_PHYSICS_GRANULARBODY_CONTACTCONSTRAINT32_DATA_SET_OVERRIDE
      #define AGX_PHYSICS_GRANULARBODY_CONTACTCONSTRAINT32_DATA_SET
      class AGXPHYSICS_EXPORT ContactConstraint32Data : public agxData::EntityData
      {
      public:
        ContactConstraint32Instance operator[] (size_t index);

      public:
        agxData::Array< ContactConstraint32Ptr >& instance;
        agxData::Array< agx::UInt32 > body1;
        agxData::Array< agx::UInt32 > body2;
        agxData::Array< agx::UInt32 > solveBody1;
        agxData::Array< agx::UInt32 > solveBody2;
        agxData::Array< agx::UInt32 > cachedContactId;
        agxData::Array< agx::UInt32 > contactIndex;
        agxData::Array< agx::UInt32 > jacobianIndex;
        agxData::Array< agx::UInt32 > rowIndex;
        agxData::Array< agx::UInt32 > materialIndex;
        agxData::Array< agx::UInt32 > iterationCount;
        agxData::Array< agx::Bool > isImpacting;
        agxData::Array< agx::UInt > activeCount;
        agxData::Array< agx::Real32 > restingCompliance;
        agxData::Array< agx::Real32 > charContactTime;
        agxData::Array< agx::Real32 > charMass;
        agxData::Array< agx::JacobianMeta32 > GMeta;
        agxData::Array< agx::Real32 > nonlinearMultiplier;
        agxData::Array< agx::Real32 > rollingResistanceMu;
        agxData::Array< agx::Real32 > twistLimitMultiplier;

      public:
        typedef agx::UInt32 body1Type;
        typedef agx::UInt32 body2Type;
        typedef agx::UInt32 solveBody1Type;
        typedef agx::UInt32 solveBody2Type;
        typedef agx::UInt32 cachedContactIdType;
        typedef agx::UInt32 contactIndexType;
        typedef agx::UInt32 jacobianIndexType;
        typedef agx::UInt32 rowIndexType;
        typedef agx::UInt32 materialIndexType;
        typedef agx::UInt32 iterationCountType;
        typedef agx::Bool isImpactingType;
        typedef agx::UInt activeCountType;
        typedef agx::Real32 restingComplianceType;
        typedef agx::Real32 charContactTimeType;
        typedef agx::Real32 charMassType;
        typedef agx::JacobianMeta32 GMetaType;
        typedef agx::Real32 nonlinearMultiplierType;
        typedef agx::Real32 rollingResistanceMuType;
        typedef agx::Real32 twistLimitMultiplierType;

      public:
        ContactConstraint32Data(agxData::EntityStorage* storage);
        ContactConstraint32Data();

      protected:
        virtual ~ContactConstraint32Data() {}
        virtual void setNumElements(agx::Index numElements) override;

      private:
        ContactConstraint32Data& operator= (const ContactConstraint32Data&) { return *this; }

      };
      #endif
      DOXYGEN_END_INTERNAL_BLOCK()


      DOXYGEN_START_INTERNAL_BLOCK()
      class AGXPHYSICS_EXPORT ContactConstraint32Semantics : protected agxData::EntityPtr
      {
      public:

        // Automatic getters
        agx::UInt32 const& getBody1() const;
        agx::UInt32 const& getBody2() const;
        agx::UInt32 const& getSolveBody1() const;
        agx::UInt32 const& getSolveBody2() const;
        agx::UInt32 const& getCachedContactId() const;
        agx::UInt32 const& getContactIndex() const;
        agx::UInt32 const& getJacobianIndex() const;
        agx::UInt32 const& getRowIndex() const;
        agx::UInt32 const& getMaterialIndex() const;
        agx::UInt32 const& getIterationCount() const;
        agx::Bool const& getIsImpacting() const;
        agx::UInt const& getActiveCount() const;
        agx::Real32 const& getRestingCompliance() const;
        agx::Real32 const& getCharContactTime() const;
        agx::Real32 const& getCharMass() const;
        agx::JacobianMeta32 const& getGMeta() const;
        agx::Real32 const& getNonlinearMultiplier() const;
        agx::Real32 const& getRollingResistanceMu() const;
        agx::Real32 const& getTwistLimitMultiplier() const;

        // Semantics defined by explicit kernels

        // Automatic setters
        void setBody1(agx::UInt32 const& value);
        void setBody2(agx::UInt32 const& value);
        void setSolveBody1(agx::UInt32 const& value);
        void setSolveBody2(agx::UInt32 const& value);
        void setCachedContactId(agx::UInt32 const& value);
        void setContactIndex(agx::UInt32 const& value);
        void setJacobianIndex(agx::UInt32 const& value);
        void setRowIndex(agx::UInt32 const& value);
        void setMaterialIndex(agx::UInt32 const& value);
        void setIterationCount(agx::UInt32 const& value);
        void setIsImpacting(agx::Bool const& value);
        void setActiveCount(agx::UInt const& value);
        void setRestingCompliance(agx::Real32 const& value);
        void setCharContactTime(agx::Real32 const& value);
        void setCharMass(agx::Real32 const& value);
        void setGMeta(agx::JacobianMeta32 const& value);
        void setNonlinearMultiplier(agx::Real32 const& value);
        void setRollingResistanceMu(agx::Real32 const& value);
        void setTwistLimitMultiplier(agx::Real32 const& value);


      protected:
        friend class ContactConstraint32Ptr;
        friend class ContactConstraint32Instance;
        ContactConstraint32Semantics();
      };
      DOXYGEN_END_INTERNAL_BLOCK()


      /**
      Pointer to a entity instance of type Physics.GranularBody.ContactConstraint32
      */
      class CALLABLE ContactConstraint32Ptr : public agxData::EntityPtr
      {
      public:
        typedef ContactConstraint32Model ModelType;
        typedef ContactConstraint32Data DataType;
        typedef ContactConstraint32Instance InstanceType;

      public:
        AGXPHYSICS_EXPORT ContactConstraint32Ptr();
        AGXPHYSICS_EXPORT ContactConstraint32Ptr(agxData::EntityStorage* storage, agx::Index id);
        AGXPHYSICS_EXPORT ContactConstraint32Ptr(const agxData::EntityPtr& ptr);
        AGXPHYSICS_EXPORT ContactConstraint32Ptr(const agxData::EntityInstance& instance);
        AGXPHYSICS_EXPORT ContactConstraint32Ptr& operator= (const agxData::EntityPtr& ptr);
        AGXPHYSICS_EXPORT ContactConstraint32Ptr& operator= (const agxData::EntityInstance& instance);
        AGXPHYSICS_EXPORT ContactConstraint32Instance instance();
        AGXPHYSICS_EXPORT const ContactConstraint32Instance instance() const;

        AGXPHYSICS_EXPORT ContactConstraint32Semantics* operator->();
        AGXPHYSICS_EXPORT const ContactConstraint32Semantics* operator->() const;

        ContactConstraint32Data* getData();
        const ContactConstraint32Data* getData() const;


        /// \return reference to the body1 attribute
        AGXPHYSICS_EXPORT agx::UInt32& body1();
        /// \return const reference to the body1 attribute
        AGXPHYSICS_EXPORT agx::UInt32 const& body1() const;

        /// \return reference to the body2 attribute
        AGXPHYSICS_EXPORT agx::UInt32& body2();
        /// \return const reference to the body2 attribute
        AGXPHYSICS_EXPORT agx::UInt32 const& body2() const;

        /// \return reference to the solveBody1 attribute
        AGXPHYSICS_EXPORT agx::UInt32& solveBody1();
        /// \return const reference to the solveBody1 attribute
        AGXPHYSICS_EXPORT agx::UInt32 const& solveBody1() const;

        /// \return reference to the solveBody2 attribute
        AGXPHYSICS_EXPORT agx::UInt32& solveBody2();
        /// \return const reference to the solveBody2 attribute
        AGXPHYSICS_EXPORT agx::UInt32 const& solveBody2() const;

        /// \return reference to the cachedContactId attribute
        AGXPHYSICS_EXPORT agx::UInt32& cachedContactId();
        /// \return const reference to the cachedContactId attribute
        AGXPHYSICS_EXPORT agx::UInt32 const& cachedContactId() const;

        /// \return reference to the contactIndex attribute
        AGXPHYSICS_EXPORT agx::UInt32& contactIndex();
        /// \return const reference to the contactIndex attribute
        AGXPHYSICS_EXPORT agx::UInt32 const& contactIndex() const;

        /// \return reference to the jacobianIndex attribute
        AGXPHYSICS_EXPORT agx::UInt32& jacobianIndex();
        /// \return const reference to the jacobianIndex attribute
        AGXPHYSICS_EXPORT agx::UInt32 const& jacobianIndex() const;

        /// \return reference to the rowIndex attribute
        AGXPHYSICS_EXPORT agx::UInt32& rowIndex();
        /// \return const reference to the rowIndex attribute
        AGXPHYSICS_EXPORT agx::UInt32 const& rowIndex() const;

        /// \return reference to the materialIndex attribute
        AGXPHYSICS_EXPORT agx::UInt32& materialIndex();
        /// \return const reference to the materialIndex attribute
        AGXPHYSICS_EXPORT agx::UInt32 const& materialIndex() const;

        /// \return reference to the iterationCount attribute
        AGXPHYSICS_EXPORT agx::UInt32& iterationCount();
        /// \return const reference to the iterationCount attribute
        AGXPHYSICS_EXPORT agx::UInt32 const& iterationCount() const;

        /// \return reference to the isImpacting attribute
        AGXPHYSICS_EXPORT agx::Bool& isImpacting();
        /// \return const reference to the isImpacting attribute
        AGXPHYSICS_EXPORT agx::Bool const& isImpacting() const;

        /// \return reference to the activeCount attribute
        AGXPHYSICS_EXPORT agx::UInt& activeCount();
        /// \return const reference to the activeCount attribute
        AGXPHYSICS_EXPORT agx::UInt const& activeCount() const;

        /// \return reference to the restingCompliance attribute
        AGXPHYSICS_EXPORT agx::Real32& restingCompliance();
        /// \return const reference to the restingCompliance attribute
        AGXPHYSICS_EXPORT agx::Real32 const& restingCompliance() const;

        /// \return reference to the charContactTime attribute
        AGXPHYSICS_EXPORT agx::Real32& charContactTime();
        /// \return const reference to the charContactTime attribute
        AGXPHYSICS_EXPORT agx::Real32 const& charContactTime() const;

        /// \return reference to the charMass attribute
        AGXPHYSICS_EXPORT agx::Real32& charMass();
        /// \return const reference to the charMass attribute
        AGXPHYSICS_EXPORT agx::Real32 const& charMass() const;

        /// \return reference to the GMeta attribute
        AGXPHYSICS_EXPORT agx::JacobianMeta32& GMeta();
        /// \return const reference to the GMeta attribute
        AGXPHYSICS_EXPORT agx::JacobianMeta32 const& GMeta() const;

        /// \return reference to the nonlinearMultiplier attribute
        AGXPHYSICS_EXPORT agx::Real32& nonlinearMultiplier();
        /// \return const reference to the nonlinearMultiplier attribute
        AGXPHYSICS_EXPORT agx::Real32 const& nonlinearMultiplier() const;

        /// \return reference to the rollingResistanceMu attribute
        AGXPHYSICS_EXPORT agx::Real32& rollingResistanceMu();
        /// \return const reference to the rollingResistanceMu attribute
        AGXPHYSICS_EXPORT agx::Real32 const& rollingResistanceMu() const;

        /// \return reference to the twistLimitMultiplier attribute
        AGXPHYSICS_EXPORT agx::Real32& twistLimitMultiplier();
        /// \return const reference to the twistLimitMultiplier attribute
        AGXPHYSICS_EXPORT agx::Real32 const& twistLimitMultiplier() const;

        DOXYGEN_START_INTERNAL_BLOCK()
        // Only used by tolua!
        #ifdef AGX_GENERATED_TOLUA_WRAPPER
        inline agx::UInt32 const& getBody1() const { return (*this)->getBody1(); }
        inline agx::UInt32 const& getBody2() const { return (*this)->getBody2(); }
        inline agx::UInt32 const& getSolveBody1() const { return (*this)->getSolveBody1(); }
        inline agx::UInt32 const& getSolveBody2() const { return (*this)->getSolveBody2(); }
        inline agx::UInt32 const& getCachedContactId() const { return (*this)->getCachedContactId(); }
        inline agx::UInt32 const& getContactIndex() const { return (*this)->getContactIndex(); }
        inline agx::UInt32 const& getJacobianIndex() const { return (*this)->getJacobianIndex(); }
        inline agx::UInt32 const& getRowIndex() const { return (*this)->getRowIndex(); }
        inline agx::UInt32 const& getMaterialIndex() const { return (*this)->getMaterialIndex(); }
        inline agx::UInt32 const& getIterationCount() const { return (*this)->getIterationCount(); }
        inline agx::Bool const& getIsImpacting() const { return (*this)->getIsImpacting(); }
        inline agx::UInt const& getActiveCount() const { return (*this)->getActiveCount(); }
        inline agx::Real32 const& getRestingCompliance() const { return (*this)->getRestingCompliance(); }
        inline agx::Real32 const& getCharContactTime() const { return (*this)->getCharContactTime(); }
        inline agx::Real32 const& getCharMass() const { return (*this)->getCharMass(); }
        inline agx::JacobianMeta32 const& getGMeta() const { return (*this)->getGMeta(); }
        inline agx::Real32 const& getNonlinearMultiplier() const { return (*this)->getNonlinearMultiplier(); }
        inline agx::Real32 const& getRollingResistanceMu() const { return (*this)->getRollingResistanceMu(); }
        inline agx::Real32 const& getTwistLimitMultiplier() const { return (*this)->getTwistLimitMultiplier(); }


        inline void setBody1(agx::UInt32 const& value) { (*this)->setBody1(value); }
        inline void setBody2(agx::UInt32 const& value) { (*this)->setBody2(value); }
        inline void setSolveBody1(agx::UInt32 const& value) { (*this)->setSolveBody1(value); }
        inline void setSolveBody2(agx::UInt32 const& value) { (*this)->setSolveBody2(value); }
        inline void setCachedContactId(agx::UInt32 const& value) { (*this)->setCachedContactId(value); }
        inline void setContactIndex(agx::UInt32 const& value) { (*this)->setContactIndex(value); }
        inline void setJacobianIndex(agx::UInt32 const& value) { (*this)->setJacobianIndex(value); }
        inline void setRowIndex(agx::UInt32 const& value) { (*this)->setRowIndex(value); }
        inline void setMaterialIndex(agx::UInt32 const& value) { (*this)->setMaterialIndex(value); }
        inline void setIterationCount(agx::UInt32 const& value) { (*this)->setIterationCount(value); }
        inline void setIsImpacting(agx::Bool const& value) { (*this)->setIsImpacting(value); }
        inline void setActiveCount(agx::UInt const& value) { (*this)->setActiveCount(value); }
        inline void setRestingCompliance(agx::Real32 const& value) { (*this)->setRestingCompliance(value); }
        inline void setCharContactTime(agx::Real32 const& value) { (*this)->setCharContactTime(value); }
        inline void setCharMass(agx::Real32 const& value) { (*this)->setCharMass(value); }
        inline void setGMeta(agx::JacobianMeta32 const& value) { (*this)->setGMeta(value); }
        inline void setNonlinearMultiplier(agx::Real32 const& value) { (*this)->setNonlinearMultiplier(value); }
        inline void setRollingResistanceMu(agx::Real32 const& value) { (*this)->setRollingResistanceMu(value); }
        inline void setTwistLimitMultiplier(agx::Real32 const& value) { (*this)->setTwistLimitMultiplier(value); }
        #endif
        DOXYGEN_END_INTERNAL_BLOCK()

      };


      DOXYGEN_START_INTERNAL_BLOCK()
      class AGXPHYSICS_EXPORT ContactConstraint32Instance : public agxData::EntityInstance
      {
      public:
        ContactConstraint32Instance();
        ContactConstraint32Instance(ContactConstraint32Data* data, agx::Index index);
        ContactConstraint32Instance(agxData::EntityStorage *storage, agx::Index index);
        ContactConstraint32Instance(const agxData::EntityInstance& other);
        ContactConstraint32Instance(const agxData::EntityPtr& ptr);

        ContactConstraint32Data* getData();
        const ContactConstraint32Data* getData() const;

      public:
        /// \return reference to the body1 attribute
        agx::UInt32& body1();
        /// \return const reference to the body1 attribute
        agx::UInt32 const& body1() const;

        /// \return reference to the body2 attribute
        agx::UInt32& body2();
        /// \return const reference to the body2 attribute
        agx::UInt32 const& body2() const;

        /// \return reference to the solveBody1 attribute
        agx::UInt32& solveBody1();
        /// \return const reference to the solveBody1 attribute
        agx::UInt32 const& solveBody1() const;

        /// \return reference to the solveBody2 attribute
        agx::UInt32& solveBody2();
        /// \return const reference to the solveBody2 attribute
        agx::UInt32 const& solveBody2() const;

        /// \return reference to the cachedContactId attribute
        agx::UInt32& cachedContactId();
        /// \return const reference to the cachedContactId attribute
        agx::UInt32 const& cachedContactId() const;

        /// \return reference to the contactIndex attribute
        agx::UInt32& contactIndex();
        /// \return const reference to the contactIndex attribute
        agx::UInt32 const& contactIndex() const;

        /// \return reference to the jacobianIndex attribute
        agx::UInt32& jacobianIndex();
        /// \return const reference to the jacobianIndex attribute
        agx::UInt32 const& jacobianIndex() const;

        /// \return reference to the rowIndex attribute
        agx::UInt32& rowIndex();
        /// \return const reference to the rowIndex attribute
        agx::UInt32 const& rowIndex() const;

        /// \return reference to the materialIndex attribute
        agx::UInt32& materialIndex();
        /// \return const reference to the materialIndex attribute
        agx::UInt32 const& materialIndex() const;

        /// \return reference to the iterationCount attribute
        agx::UInt32& iterationCount();
        /// \return const reference to the iterationCount attribute
        agx::UInt32 const& iterationCount() const;

        /// \return reference to the isImpacting attribute
        agx::Bool& isImpacting();
        /// \return const reference to the isImpacting attribute
        agx::Bool const& isImpacting() const;

        /// \return reference to the activeCount attribute
        agx::UInt& activeCount();
        /// \return const reference to the activeCount attribute
        agx::UInt const& activeCount() const;

        /// \return reference to the restingCompliance attribute
        agx::Real32& restingCompliance();
        /// \return const reference to the restingCompliance attribute
        agx::Real32 const& restingCompliance() const;

        /// \return reference to the charContactTime attribute
        agx::Real32& charContactTime();
        /// \return const reference to the charContactTime attribute
        agx::Real32 const& charContactTime() const;

        /// \return reference to the charMass attribute
        agx::Real32& charMass();
        /// \return const reference to the charMass attribute
        agx::Real32 const& charMass() const;

        /// \return reference to the GMeta attribute
        agx::JacobianMeta32& GMeta();
        /// \return const reference to the GMeta attribute
        agx::JacobianMeta32 const& GMeta() const;

        /// \return reference to the nonlinearMultiplier attribute
        agx::Real32& nonlinearMultiplier();
        /// \return const reference to the nonlinearMultiplier attribute
        agx::Real32 const& nonlinearMultiplier() const;

        /// \return reference to the rollingResistanceMu attribute
        agx::Real32& rollingResistanceMu();
        /// \return const reference to the rollingResistanceMu attribute
        agx::Real32 const& rollingResistanceMu() const;

        /// \return reference to the twistLimitMultiplier attribute
        agx::Real32& twistLimitMultiplier();
        /// \return const reference to the twistLimitMultiplier attribute
        agx::Real32 const& twistLimitMultiplier() const;

      };
      DOXYGEN_END_INTERNAL_BLOCK()



      typedef agx::VectorPOD<ContactConstraint32Ptr> ContactConstraint32PtrVector;
      typedef agxData::Array<ContactConstraint32Ptr> ContactConstraint32PtrArray;



      DOXYGEN_START_INTERNAL_BLOCK()
      /* Implementation */
      //-----------------------------------------------------------------------------------------------------
      //-----------------------------------------------------------------------------------------------------
      inline ContactConstraint32Instance agx::Physics::GranularBody::ContactConstraint32Data::operator[] (size_t index) { return ContactConstraint32Instance(this, (agx::Index)index); }
      //-----------------------------------------------------------------------------------------------------
      AGX_FORCE_INLINE ContactConstraint32Ptr::ContactConstraint32Ptr() {}
      AGX_FORCE_INLINE ContactConstraint32Ptr::ContactConstraint32Ptr(agxData::EntityStorage* storage, agx::Index id) : agxData::EntityPtr(storage, id) {}
      AGX_FORCE_INLINE ContactConstraint32Ptr::ContactConstraint32Ptr(const agxData::EntityPtr& ptr) : agxData::EntityPtr(ptr)
      {
        agxAssertN(!ptr || ptr.isInstanceOf(ContactConstraint32Model::instance()),
          "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
          EntityPtr::getModel()->fullPath().c_str(), ContactConstraint32Model::instance()->fullPath().c_str());
      }

      AGX_FORCE_INLINE ContactConstraint32Ptr::ContactConstraint32Ptr(const agxData::EntityInstance& instance) : agxData::EntityPtr(instance)
      {
        agxAssertN(!instance || instance.isInstanceOf(ContactConstraint32Model::instance()),
          "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
          EntityPtr::getModel()->fullPath().c_str(), ContactConstraint32Model::instance()->fullPath().c_str());
      }

      AGX_FORCE_INLINE ContactConstraint32Ptr& ContactConstraint32Ptr::operator= (const agxData::EntityPtr& ptr)
      {
        agxData::EntityPtr::operator= (ptr);
        agxAssertN(!ptr || ptr.isInstanceOf(ContactConstraint32Model::instance()),
          "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
          EntityPtr::getModel()->fullPath().c_str(), ContactConstraint32Model::instance()->fullPath().c_str());
        return *this;
      }

      AGX_FORCE_INLINE ContactConstraint32Ptr& ContactConstraint32Ptr::operator= (const agxData::EntityInstance& instance)
      {
        agxData::EntityPtr::operator= (instance);
        agxAssertN(!instance || instance.isInstanceOf(ContactConstraint32Model::instance()),
          "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
          EntityPtr::getModel()->fullPath().c_str(), ContactConstraint32Model::instance()->fullPath().c_str());
        return *this;
      }

      inline ContactConstraint32Instance ContactConstraint32Ptr::instance() { return agxData::EntityPtr::instance(); }
      inline const ContactConstraint32Instance ContactConstraint32Ptr::instance() const { return agxData::EntityPtr::instance(); }
      AGX_FORCE_INLINE ContactConstraint32Semantics* ContactConstraint32Ptr::operator->() { return (ContactConstraint32Semantics* )this; }
      AGX_FORCE_INLINE const ContactConstraint32Semantics* ContactConstraint32Ptr::operator->() const { return (const ContactConstraint32Semantics* )this; }
      AGX_FORCE_INLINE ContactConstraint32Data* ContactConstraint32Ptr::getData() { return static_cast<ContactConstraint32Data* >(agxData::EntityPtr::getData()); }
      AGX_FORCE_INLINE const ContactConstraint32Data* ContactConstraint32Ptr::getData() const { return static_cast<const ContactConstraint32Data* >(agxData::EntityPtr::getData()); }

      AGX_FORCE_INLINE agx::UInt32& ContactConstraint32Ptr::body1() { verifyIndex(); return getData()->body1[calculateIndex()]; }
      AGX_FORCE_INLINE agx::UInt32 const& ContactConstraint32Ptr::body1() const { verifyIndex(); return getData()->body1[calculateIndex()]; }

      AGX_FORCE_INLINE agx::UInt32& ContactConstraint32Ptr::body2() { verifyIndex(); return getData()->body2[calculateIndex()]; }
      AGX_FORCE_INLINE agx::UInt32 const& ContactConstraint32Ptr::body2() const { verifyIndex(); return getData()->body2[calculateIndex()]; }

      AGX_FORCE_INLINE agx::UInt32& ContactConstraint32Ptr::solveBody1() { verifyIndex(); return getData()->solveBody1[calculateIndex()]; }
      AGX_FORCE_INLINE agx::UInt32 const& ContactConstraint32Ptr::solveBody1() const { verifyIndex(); return getData()->solveBody1[calculateIndex()]; }

      AGX_FORCE_INLINE agx::UInt32& ContactConstraint32Ptr::solveBody2() { verifyIndex(); return getData()->solveBody2[calculateIndex()]; }
      AGX_FORCE_INLINE agx::UInt32 const& ContactConstraint32Ptr::solveBody2() const { verifyIndex(); return getData()->solveBody2[calculateIndex()]; }

      AGX_FORCE_INLINE agx::UInt32& ContactConstraint32Ptr::cachedContactId() { verifyIndex(); return getData()->cachedContactId[calculateIndex()]; }
      AGX_FORCE_INLINE agx::UInt32 const& ContactConstraint32Ptr::cachedContactId() const { verifyIndex(); return getData()->cachedContactId[calculateIndex()]; }

      AGX_FORCE_INLINE agx::UInt32& ContactConstraint32Ptr::contactIndex() { verifyIndex(); return getData()->contactIndex[calculateIndex()]; }
      AGX_FORCE_INLINE agx::UInt32 const& ContactConstraint32Ptr::contactIndex() const { verifyIndex(); return getData()->contactIndex[calculateIndex()]; }

      AGX_FORCE_INLINE agx::UInt32& ContactConstraint32Ptr::jacobianIndex() { verifyIndex(); return getData()->jacobianIndex[calculateIndex()]; }
      AGX_FORCE_INLINE agx::UInt32 const& ContactConstraint32Ptr::jacobianIndex() const { verifyIndex(); return getData()->jacobianIndex[calculateIndex()]; }

      AGX_FORCE_INLINE agx::UInt32& ContactConstraint32Ptr::rowIndex() { verifyIndex(); return getData()->rowIndex[calculateIndex()]; }
      AGX_FORCE_INLINE agx::UInt32 const& ContactConstraint32Ptr::rowIndex() const { verifyIndex(); return getData()->rowIndex[calculateIndex()]; }

      AGX_FORCE_INLINE agx::UInt32& ContactConstraint32Ptr::materialIndex() { verifyIndex(); return getData()->materialIndex[calculateIndex()]; }
      AGX_FORCE_INLINE agx::UInt32 const& ContactConstraint32Ptr::materialIndex() const { verifyIndex(); return getData()->materialIndex[calculateIndex()]; }

      AGX_FORCE_INLINE agx::UInt32& ContactConstraint32Ptr::iterationCount() { verifyIndex(); return getData()->iterationCount[calculateIndex()]; }
      AGX_FORCE_INLINE agx::UInt32 const& ContactConstraint32Ptr::iterationCount() const { verifyIndex(); return getData()->iterationCount[calculateIndex()]; }

      AGX_FORCE_INLINE agx::Bool& ContactConstraint32Ptr::isImpacting() { verifyIndex(); return getData()->isImpacting[calculateIndex()]; }
      AGX_FORCE_INLINE agx::Bool const& ContactConstraint32Ptr::isImpacting() const { verifyIndex(); return getData()->isImpacting[calculateIndex()]; }

      AGX_FORCE_INLINE agx::UInt& ContactConstraint32Ptr::activeCount() { verifyIndex(); return getData()->activeCount[calculateIndex()]; }
      AGX_FORCE_INLINE agx::UInt const& ContactConstraint32Ptr::activeCount() const { verifyIndex(); return getData()->activeCount[calculateIndex()]; }

      AGX_FORCE_INLINE agx::Real32& ContactConstraint32Ptr::restingCompliance() { verifyIndex(); return getData()->restingCompliance[calculateIndex()]; }
      AGX_FORCE_INLINE agx::Real32 const& ContactConstraint32Ptr::restingCompliance() const { verifyIndex(); return getData()->restingCompliance[calculateIndex()]; }

      AGX_FORCE_INLINE agx::Real32& ContactConstraint32Ptr::charContactTime() { verifyIndex(); return getData()->charContactTime[calculateIndex()]; }
      AGX_FORCE_INLINE agx::Real32 const& ContactConstraint32Ptr::charContactTime() const { verifyIndex(); return getData()->charContactTime[calculateIndex()]; }

      AGX_FORCE_INLINE agx::Real32& ContactConstraint32Ptr::charMass() { verifyIndex(); return getData()->charMass[calculateIndex()]; }
      AGX_FORCE_INLINE agx::Real32 const& ContactConstraint32Ptr::charMass() const { verifyIndex(); return getData()->charMass[calculateIndex()]; }

      AGX_FORCE_INLINE agx::JacobianMeta32& ContactConstraint32Ptr::GMeta() { verifyIndex(); return getData()->GMeta[calculateIndex()]; }
      AGX_FORCE_INLINE agx::JacobianMeta32 const& ContactConstraint32Ptr::GMeta() const { verifyIndex(); return getData()->GMeta[calculateIndex()]; }

      AGX_FORCE_INLINE agx::Real32& ContactConstraint32Ptr::nonlinearMultiplier() { verifyIndex(); return getData()->nonlinearMultiplier[calculateIndex()]; }
      AGX_FORCE_INLINE agx::Real32 const& ContactConstraint32Ptr::nonlinearMultiplier() const { verifyIndex(); return getData()->nonlinearMultiplier[calculateIndex()]; }

      AGX_FORCE_INLINE agx::Real32& ContactConstraint32Ptr::rollingResistanceMu() { verifyIndex(); return getData()->rollingResistanceMu[calculateIndex()]; }
      AGX_FORCE_INLINE agx::Real32 const& ContactConstraint32Ptr::rollingResistanceMu() const { verifyIndex(); return getData()->rollingResistanceMu[calculateIndex()]; }

      AGX_FORCE_INLINE agx::Real32& ContactConstraint32Ptr::twistLimitMultiplier() { verifyIndex(); return getData()->twistLimitMultiplier[calculateIndex()]; }
      AGX_FORCE_INLINE agx::Real32 const& ContactConstraint32Ptr::twistLimitMultiplier() const { verifyIndex(); return getData()->twistLimitMultiplier[calculateIndex()]; }

      //-----------------------------------------------------------------------------------------------------
      AGX_FORCE_INLINE ContactConstraint32Instance::ContactConstraint32Instance() {}
      AGX_FORCE_INLINE ContactConstraint32Instance::ContactConstraint32Instance(ContactConstraint32Data* data, agx::Index index) : agxData::EntityInstance(data, index) {}
      AGX_FORCE_INLINE ContactConstraint32Instance::ContactConstraint32Instance(agxData::EntityStorage* storage, agx::Index index) : agxData::EntityInstance(storage, index) {}
      AGX_FORCE_INLINE ContactConstraint32Instance::ContactConstraint32Instance(const agxData::EntityInstance& other) : agxData::EntityInstance(other)
      {
        agxAssertN(!other || other.isInstanceOf(ContactConstraint32Model::instance()),
          "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
          EntityInstance::getModel()->fullPath().c_str(), ContactConstraint32Model::instance()->fullPath().c_str());
      }

      AGX_FORCE_INLINE ContactConstraint32Instance::ContactConstraint32Instance(const agxData::EntityPtr& ptr) : agxData::EntityInstance(ptr)
      {
        agxAssertN(!ptr || ptr.isInstanceOf(ContactConstraint32Model::instance()),
          "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
          EntityInstance::getModel()->fullPath().c_str(), ContactConstraint32Model::instance()->fullPath().c_str());
      }


      AGX_FORCE_INLINE ContactConstraint32Data* ContactConstraint32Instance::getData() { return static_cast<ContactConstraint32Data* >(agxData::EntityInstance::getData()); }
      AGX_FORCE_INLINE const ContactConstraint32Data* ContactConstraint32Instance::getData() const { return static_cast<const ContactConstraint32Data* >(agxData::EntityInstance::getData()); }

      AGX_FORCE_INLINE agx::UInt32& ContactConstraint32Instance::body1() { verifyIndex(); return getData()->body1[getIndex()]; }
      AGX_FORCE_INLINE agx::UInt32 const& ContactConstraint32Instance::body1() const { verifyIndex(); return getData()->body1[getIndex()]; }

      AGX_FORCE_INLINE agx::UInt32& ContactConstraint32Instance::body2() { verifyIndex(); return getData()->body2[getIndex()]; }
      AGX_FORCE_INLINE agx::UInt32 const& ContactConstraint32Instance::body2() const { verifyIndex(); return getData()->body2[getIndex()]; }

      AGX_FORCE_INLINE agx::UInt32& ContactConstraint32Instance::solveBody1() { verifyIndex(); return getData()->solveBody1[getIndex()]; }
      AGX_FORCE_INLINE agx::UInt32 const& ContactConstraint32Instance::solveBody1() const { verifyIndex(); return getData()->solveBody1[getIndex()]; }

      AGX_FORCE_INLINE agx::UInt32& ContactConstraint32Instance::solveBody2() { verifyIndex(); return getData()->solveBody2[getIndex()]; }
      AGX_FORCE_INLINE agx::UInt32 const& ContactConstraint32Instance::solveBody2() const { verifyIndex(); return getData()->solveBody2[getIndex()]; }

      AGX_FORCE_INLINE agx::UInt32& ContactConstraint32Instance::cachedContactId() { verifyIndex(); return getData()->cachedContactId[getIndex()]; }
      AGX_FORCE_INLINE agx::UInt32 const& ContactConstraint32Instance::cachedContactId() const { verifyIndex(); return getData()->cachedContactId[getIndex()]; }

      AGX_FORCE_INLINE agx::UInt32& ContactConstraint32Instance::contactIndex() { verifyIndex(); return getData()->contactIndex[getIndex()]; }
      AGX_FORCE_INLINE agx::UInt32 const& ContactConstraint32Instance::contactIndex() const { verifyIndex(); return getData()->contactIndex[getIndex()]; }

      AGX_FORCE_INLINE agx::UInt32& ContactConstraint32Instance::jacobianIndex() { verifyIndex(); return getData()->jacobianIndex[getIndex()]; }
      AGX_FORCE_INLINE agx::UInt32 const& ContactConstraint32Instance::jacobianIndex() const { verifyIndex(); return getData()->jacobianIndex[getIndex()]; }

      AGX_FORCE_INLINE agx::UInt32& ContactConstraint32Instance::rowIndex() { verifyIndex(); return getData()->rowIndex[getIndex()]; }
      AGX_FORCE_INLINE agx::UInt32 const& ContactConstraint32Instance::rowIndex() const { verifyIndex(); return getData()->rowIndex[getIndex()]; }

      AGX_FORCE_INLINE agx::UInt32& ContactConstraint32Instance::materialIndex() { verifyIndex(); return getData()->materialIndex[getIndex()]; }
      AGX_FORCE_INLINE agx::UInt32 const& ContactConstraint32Instance::materialIndex() const { verifyIndex(); return getData()->materialIndex[getIndex()]; }

      AGX_FORCE_INLINE agx::UInt32& ContactConstraint32Instance::iterationCount() { verifyIndex(); return getData()->iterationCount[getIndex()]; }
      AGX_FORCE_INLINE agx::UInt32 const& ContactConstraint32Instance::iterationCount() const { verifyIndex(); return getData()->iterationCount[getIndex()]; }

      AGX_FORCE_INLINE agx::Bool& ContactConstraint32Instance::isImpacting() { verifyIndex(); return getData()->isImpacting[getIndex()]; }
      AGX_FORCE_INLINE agx::Bool const& ContactConstraint32Instance::isImpacting() const { verifyIndex(); return getData()->isImpacting[getIndex()]; }

      AGX_FORCE_INLINE agx::UInt& ContactConstraint32Instance::activeCount() { verifyIndex(); return getData()->activeCount[getIndex()]; }
      AGX_FORCE_INLINE agx::UInt const& ContactConstraint32Instance::activeCount() const { verifyIndex(); return getData()->activeCount[getIndex()]; }

      AGX_FORCE_INLINE agx::Real32& ContactConstraint32Instance::restingCompliance() { verifyIndex(); return getData()->restingCompliance[getIndex()]; }
      AGX_FORCE_INLINE agx::Real32 const& ContactConstraint32Instance::restingCompliance() const { verifyIndex(); return getData()->restingCompliance[getIndex()]; }

      AGX_FORCE_INLINE agx::Real32& ContactConstraint32Instance::charContactTime() { verifyIndex(); return getData()->charContactTime[getIndex()]; }
      AGX_FORCE_INLINE agx::Real32 const& ContactConstraint32Instance::charContactTime() const { verifyIndex(); return getData()->charContactTime[getIndex()]; }

      AGX_FORCE_INLINE agx::Real32& ContactConstraint32Instance::charMass() { verifyIndex(); return getData()->charMass[getIndex()]; }
      AGX_FORCE_INLINE agx::Real32 const& ContactConstraint32Instance::charMass() const { verifyIndex(); return getData()->charMass[getIndex()]; }

      AGX_FORCE_INLINE agx::JacobianMeta32& ContactConstraint32Instance::GMeta() { verifyIndex(); return getData()->GMeta[getIndex()]; }
      AGX_FORCE_INLINE agx::JacobianMeta32 const& ContactConstraint32Instance::GMeta() const { verifyIndex(); return getData()->GMeta[getIndex()]; }

      AGX_FORCE_INLINE agx::Real32& ContactConstraint32Instance::nonlinearMultiplier() { verifyIndex(); return getData()->nonlinearMultiplier[getIndex()]; }
      AGX_FORCE_INLINE agx::Real32 const& ContactConstraint32Instance::nonlinearMultiplier() const { verifyIndex(); return getData()->nonlinearMultiplier[getIndex()]; }

      AGX_FORCE_INLINE agx::Real32& ContactConstraint32Instance::rollingResistanceMu() { verifyIndex(); return getData()->rollingResistanceMu[getIndex()]; }
      AGX_FORCE_INLINE agx::Real32 const& ContactConstraint32Instance::rollingResistanceMu() const { verifyIndex(); return getData()->rollingResistanceMu[getIndex()]; }

      AGX_FORCE_INLINE agx::Real32& ContactConstraint32Instance::twistLimitMultiplier() { verifyIndex(); return getData()->twistLimitMultiplier[getIndex()]; }
      AGX_FORCE_INLINE agx::Real32 const& ContactConstraint32Instance::twistLimitMultiplier() const { verifyIndex(); return getData()->twistLimitMultiplier[getIndex()]; }

      //-----------------------------------------------------------------------------------------------------
      AGX_FORCE_INLINE ContactConstraint32Semantics::ContactConstraint32Semantics() {}
      //-----------------------------------------------------------------------------------------------------
      DOXYGEN_END_INTERNAL_BLOCK()
    }
  }
}

AGX_TYPE_BINDING(agx::Physics::GranularBody::ContactConstraint32Ptr, "Physics.GranularBody.ContactConstraint32Ptr")
AGX_TYPE_BINDING(agx::Physics::GranularBody::ContactConstraint32Instance, "Physics.GranularBody.ContactConstraint32Instance")

#ifdef _MSC_VER
# pragma warning(pop)
#endif

#undef AGX_ENTITY_WRAPPER
#undef AGX_ENTITY_NAMESPACE
#endif

