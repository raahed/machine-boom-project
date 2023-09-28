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

#ifndef GENERATED_AGX_PHYSICS_GRANULARBODY_CONTACTCONSTRAINT_H_PLUGIN
#define GENERATED_AGX_PHYSICS_GRANULARBODY_CONTACTCONSTRAINT_H_PLUGIN

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

      class ContactConstraintModel;
      class ContactConstraintData;
      class ContactConstraintPtr;
      class ContactConstraintInstance;
      class ContactConstraintSemantics;


      AGX_DECLARE_POINTER_TYPES(ContactConstraintModel);

      /** 
      Abstract description of the data attributes for the Physics.GranularBody.ContactConstraint entity.
      */ 
      class AGXPHYSICS_EXPORT ContactConstraintModel : public agxData::EntityModel
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
        static agxData::ScalarAttributeT< agx::Real >* restingComplianceAttribute;
        static agxData::ScalarAttributeT< agx::Real >* charContactTimeAttribute;
        static agxData::ScalarAttributeT< agx::Real >* charMassAttribute;
        static agxData::ScalarAttributeT< agx::JacobianMeta >* GMetaAttribute;
        static agxData::ScalarAttributeT< agx::Real >* nonlinearMultiplierAttribute;
        static agxData::ScalarAttributeT< agx::Real >* rollingResistanceMuAttribute;
        static agxData::ScalarAttributeT< agx::Real >* twistLimitMultiplierAttribute;

      protected:
        virtual ~ContactConstraintModel();
        virtual agxData::EntityData* createData(agxData::EntityStorage* storage) override;
        virtual void configure(agx::TiXmlElement* eEntity) override;
        virtual void initAttributeAccessors() override;
        virtual void construct(agxData::EntityPtr instance) override;
        void construct(agx::Physics::GranularBody::ContactConstraintPtr contactConstraint);
      };


      DOXYGEN_START_INTERNAL_BLOCK()
      #ifndef AGX_PHYSICS_GRANULARBODY_CONTACTCONSTRAINT_DATA_SET_OVERRIDE
      #define AGX_PHYSICS_GRANULARBODY_CONTACTCONSTRAINT_DATA_SET
      class AGXPHYSICS_EXPORT ContactConstraintData : public agxData::EntityData
      {
      public:
        ContactConstraintInstance operator[] (size_t index);

      public:
        agxData::Array< ContactConstraintPtr >& instance;
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
        agxData::Array< agx::Real > restingCompliance;
        agxData::Array< agx::Real > charContactTime;
        agxData::Array< agx::Real > charMass;
        agxData::Array< agx::JacobianMeta > GMeta;
        agxData::Array< agx::Real > nonlinearMultiplier;
        agxData::Array< agx::Real > rollingResistanceMu;
        agxData::Array< agx::Real > twistLimitMultiplier;

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
        typedef agx::Real restingComplianceType;
        typedef agx::Real charContactTimeType;
        typedef agx::Real charMassType;
        typedef agx::JacobianMeta GMetaType;
        typedef agx::Real nonlinearMultiplierType;
        typedef agx::Real rollingResistanceMuType;
        typedef agx::Real twistLimitMultiplierType;

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
      class AGXPHYSICS_EXPORT ContactConstraintSemantics : protected agxData::EntityPtr
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
        agx::Real const& getRestingCompliance() const;
        agx::Real const& getCharContactTime() const;
        agx::Real const& getCharMass() const;
        agx::JacobianMeta const& getGMeta() const;
        agx::Real const& getNonlinearMultiplier() const;
        agx::Real const& getRollingResistanceMu() const;
        agx::Real const& getTwistLimitMultiplier() const;

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
        void setRestingCompliance(agx::Real const& value);
        void setCharContactTime(agx::Real const& value);
        void setCharMass(agx::Real const& value);
        void setGMeta(agx::JacobianMeta const& value);
        void setNonlinearMultiplier(agx::Real const& value);
        void setRollingResistanceMu(agx::Real const& value);
        void setTwistLimitMultiplier(agx::Real const& value);


      protected:
        friend class ContactConstraintPtr;
        friend class ContactConstraintInstance;
        ContactConstraintSemantics();
      };
      DOXYGEN_END_INTERNAL_BLOCK()


      /**
      Pointer to a entity instance of type Physics.GranularBody.ContactConstraint
      */
      class CALLABLE ContactConstraintPtr : public agxData::EntityPtr
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

        /// \return reference to the restingCompliance attribute
        AGXPHYSICS_EXPORT agx::Real& restingCompliance();
        /// \return const reference to the restingCompliance attribute
        AGXPHYSICS_EXPORT agx::Real const& restingCompliance() const;

        /// \return reference to the charContactTime attribute
        AGXPHYSICS_EXPORT agx::Real& charContactTime();
        /// \return const reference to the charContactTime attribute
        AGXPHYSICS_EXPORT agx::Real const& charContactTime() const;

        /// \return reference to the charMass attribute
        AGXPHYSICS_EXPORT agx::Real& charMass();
        /// \return const reference to the charMass attribute
        AGXPHYSICS_EXPORT agx::Real const& charMass() const;

        /// \return reference to the GMeta attribute
        AGXPHYSICS_EXPORT agx::JacobianMeta& GMeta();
        /// \return const reference to the GMeta attribute
        AGXPHYSICS_EXPORT agx::JacobianMeta const& GMeta() const;

        /// \return reference to the nonlinearMultiplier attribute
        AGXPHYSICS_EXPORT agx::Real& nonlinearMultiplier();
        /// \return const reference to the nonlinearMultiplier attribute
        AGXPHYSICS_EXPORT agx::Real const& nonlinearMultiplier() const;

        /// \return reference to the rollingResistanceMu attribute
        AGXPHYSICS_EXPORT agx::Real& rollingResistanceMu();
        /// \return const reference to the rollingResistanceMu attribute
        AGXPHYSICS_EXPORT agx::Real const& rollingResistanceMu() const;

        /// \return reference to the twistLimitMultiplier attribute
        AGXPHYSICS_EXPORT agx::Real& twistLimitMultiplier();
        /// \return const reference to the twistLimitMultiplier attribute
        AGXPHYSICS_EXPORT agx::Real const& twistLimitMultiplier() const;

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
        inline agx::Real const& getRestingCompliance() const { return (*this)->getRestingCompliance(); }
        inline agx::Real const& getCharContactTime() const { return (*this)->getCharContactTime(); }
        inline agx::Real const& getCharMass() const { return (*this)->getCharMass(); }
        inline agx::JacobianMeta const& getGMeta() const { return (*this)->getGMeta(); }
        inline agx::Real const& getNonlinearMultiplier() const { return (*this)->getNonlinearMultiplier(); }
        inline agx::Real const& getRollingResistanceMu() const { return (*this)->getRollingResistanceMu(); }
        inline agx::Real const& getTwistLimitMultiplier() const { return (*this)->getTwistLimitMultiplier(); }


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
        inline void setRestingCompliance(agx::Real const& value) { (*this)->setRestingCompliance(value); }
        inline void setCharContactTime(agx::Real const& value) { (*this)->setCharContactTime(value); }
        inline void setCharMass(agx::Real const& value) { (*this)->setCharMass(value); }
        inline void setGMeta(agx::JacobianMeta const& value) { (*this)->setGMeta(value); }
        inline void setNonlinearMultiplier(agx::Real const& value) { (*this)->setNonlinearMultiplier(value); }
        inline void setRollingResistanceMu(agx::Real const& value) { (*this)->setRollingResistanceMu(value); }
        inline void setTwistLimitMultiplier(agx::Real const& value) { (*this)->setTwistLimitMultiplier(value); }
        #endif
        DOXYGEN_END_INTERNAL_BLOCK()

      };


      DOXYGEN_START_INTERNAL_BLOCK()
      class AGXPHYSICS_EXPORT ContactConstraintInstance : public agxData::EntityInstance
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

        /// \return reference to the restingCompliance attribute
        agx::Real& restingCompliance();
        /// \return const reference to the restingCompliance attribute
        agx::Real const& restingCompliance() const;

        /// \return reference to the charContactTime attribute
        agx::Real& charContactTime();
        /// \return const reference to the charContactTime attribute
        agx::Real const& charContactTime() const;

        /// \return reference to the charMass attribute
        agx::Real& charMass();
        /// \return const reference to the charMass attribute
        agx::Real const& charMass() const;

        /// \return reference to the GMeta attribute
        agx::JacobianMeta& GMeta();
        /// \return const reference to the GMeta attribute
        agx::JacobianMeta const& GMeta() const;

        /// \return reference to the nonlinearMultiplier attribute
        agx::Real& nonlinearMultiplier();
        /// \return const reference to the nonlinearMultiplier attribute
        agx::Real const& nonlinearMultiplier() const;

        /// \return reference to the rollingResistanceMu attribute
        agx::Real& rollingResistanceMu();
        /// \return const reference to the rollingResistanceMu attribute
        agx::Real const& rollingResistanceMu() const;

        /// \return reference to the twistLimitMultiplier attribute
        agx::Real& twistLimitMultiplier();
        /// \return const reference to the twistLimitMultiplier attribute
        agx::Real const& twistLimitMultiplier() const;

      };
      DOXYGEN_END_INTERNAL_BLOCK()



      typedef agx::VectorPOD<ContactConstraintPtr> ContactConstraintPtrVector;
      typedef agxData::Array<ContactConstraintPtr> ContactConstraintPtrArray;



      DOXYGEN_START_INTERNAL_BLOCK()
      /* Implementation */
      //-----------------------------------------------------------------------------------------------------
      //-----------------------------------------------------------------------------------------------------
      inline ContactConstraintInstance agx::Physics::GranularBody::ContactConstraintData::operator[] (size_t index) { return ContactConstraintInstance(this, (agx::Index)index); }
      //-----------------------------------------------------------------------------------------------------
      AGX_FORCE_INLINE ContactConstraintPtr::ContactConstraintPtr() {}
      AGX_FORCE_INLINE ContactConstraintPtr::ContactConstraintPtr(agxData::EntityStorage* storage, agx::Index id) : agxData::EntityPtr(storage, id) {}
      AGX_FORCE_INLINE ContactConstraintPtr::ContactConstraintPtr(const agxData::EntityPtr& ptr) : agxData::EntityPtr(ptr)
      {
        agxAssertN(!ptr || ptr.isInstanceOf(ContactConstraintModel::instance()),
          "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
          EntityPtr::getModel()->fullPath().c_str(), ContactConstraintModel::instance()->fullPath().c_str());
      }

      AGX_FORCE_INLINE ContactConstraintPtr::ContactConstraintPtr(const agxData::EntityInstance& instance) : agxData::EntityPtr(instance)
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

      AGX_FORCE_INLINE agx::UInt32& ContactConstraintPtr::body1() { verifyIndex(); return getData()->body1[calculateIndex()]; }
      AGX_FORCE_INLINE agx::UInt32 const& ContactConstraintPtr::body1() const { verifyIndex(); return getData()->body1[calculateIndex()]; }

      AGX_FORCE_INLINE agx::UInt32& ContactConstraintPtr::body2() { verifyIndex(); return getData()->body2[calculateIndex()]; }
      AGX_FORCE_INLINE agx::UInt32 const& ContactConstraintPtr::body2() const { verifyIndex(); return getData()->body2[calculateIndex()]; }

      AGX_FORCE_INLINE agx::UInt32& ContactConstraintPtr::solveBody1() { verifyIndex(); return getData()->solveBody1[calculateIndex()]; }
      AGX_FORCE_INLINE agx::UInt32 const& ContactConstraintPtr::solveBody1() const { verifyIndex(); return getData()->solveBody1[calculateIndex()]; }

      AGX_FORCE_INLINE agx::UInt32& ContactConstraintPtr::solveBody2() { verifyIndex(); return getData()->solveBody2[calculateIndex()]; }
      AGX_FORCE_INLINE agx::UInt32 const& ContactConstraintPtr::solveBody2() const { verifyIndex(); return getData()->solveBody2[calculateIndex()]; }

      AGX_FORCE_INLINE agx::UInt32& ContactConstraintPtr::cachedContactId() { verifyIndex(); return getData()->cachedContactId[calculateIndex()]; }
      AGX_FORCE_INLINE agx::UInt32 const& ContactConstraintPtr::cachedContactId() const { verifyIndex(); return getData()->cachedContactId[calculateIndex()]; }

      AGX_FORCE_INLINE agx::UInt32& ContactConstraintPtr::contactIndex() { verifyIndex(); return getData()->contactIndex[calculateIndex()]; }
      AGX_FORCE_INLINE agx::UInt32 const& ContactConstraintPtr::contactIndex() const { verifyIndex(); return getData()->contactIndex[calculateIndex()]; }

      AGX_FORCE_INLINE agx::UInt32& ContactConstraintPtr::jacobianIndex() { verifyIndex(); return getData()->jacobianIndex[calculateIndex()]; }
      AGX_FORCE_INLINE agx::UInt32 const& ContactConstraintPtr::jacobianIndex() const { verifyIndex(); return getData()->jacobianIndex[calculateIndex()]; }

      AGX_FORCE_INLINE agx::UInt32& ContactConstraintPtr::rowIndex() { verifyIndex(); return getData()->rowIndex[calculateIndex()]; }
      AGX_FORCE_INLINE agx::UInt32 const& ContactConstraintPtr::rowIndex() const { verifyIndex(); return getData()->rowIndex[calculateIndex()]; }

      AGX_FORCE_INLINE agx::UInt32& ContactConstraintPtr::materialIndex() { verifyIndex(); return getData()->materialIndex[calculateIndex()]; }
      AGX_FORCE_INLINE agx::UInt32 const& ContactConstraintPtr::materialIndex() const { verifyIndex(); return getData()->materialIndex[calculateIndex()]; }

      AGX_FORCE_INLINE agx::UInt32& ContactConstraintPtr::iterationCount() { verifyIndex(); return getData()->iterationCount[calculateIndex()]; }
      AGX_FORCE_INLINE agx::UInt32 const& ContactConstraintPtr::iterationCount() const { verifyIndex(); return getData()->iterationCount[calculateIndex()]; }

      AGX_FORCE_INLINE agx::Bool& ContactConstraintPtr::isImpacting() { verifyIndex(); return getData()->isImpacting[calculateIndex()]; }
      AGX_FORCE_INLINE agx::Bool const& ContactConstraintPtr::isImpacting() const { verifyIndex(); return getData()->isImpacting[calculateIndex()]; }

      AGX_FORCE_INLINE agx::Real& ContactConstraintPtr::restingCompliance() { verifyIndex(); return getData()->restingCompliance[calculateIndex()]; }
      AGX_FORCE_INLINE agx::Real const& ContactConstraintPtr::restingCompliance() const { verifyIndex(); return getData()->restingCompliance[calculateIndex()]; }

      AGX_FORCE_INLINE agx::Real& ContactConstraintPtr::charContactTime() { verifyIndex(); return getData()->charContactTime[calculateIndex()]; }
      AGX_FORCE_INLINE agx::Real const& ContactConstraintPtr::charContactTime() const { verifyIndex(); return getData()->charContactTime[calculateIndex()]; }

      AGX_FORCE_INLINE agx::Real& ContactConstraintPtr::charMass() { verifyIndex(); return getData()->charMass[calculateIndex()]; }
      AGX_FORCE_INLINE agx::Real const& ContactConstraintPtr::charMass() const { verifyIndex(); return getData()->charMass[calculateIndex()]; }

      AGX_FORCE_INLINE agx::JacobianMeta& ContactConstraintPtr::GMeta() { verifyIndex(); return getData()->GMeta[calculateIndex()]; }
      AGX_FORCE_INLINE agx::JacobianMeta const& ContactConstraintPtr::GMeta() const { verifyIndex(); return getData()->GMeta[calculateIndex()]; }

      AGX_FORCE_INLINE agx::Real& ContactConstraintPtr::nonlinearMultiplier() { verifyIndex(); return getData()->nonlinearMultiplier[calculateIndex()]; }
      AGX_FORCE_INLINE agx::Real const& ContactConstraintPtr::nonlinearMultiplier() const { verifyIndex(); return getData()->nonlinearMultiplier[calculateIndex()]; }

      AGX_FORCE_INLINE agx::Real& ContactConstraintPtr::rollingResistanceMu() { verifyIndex(); return getData()->rollingResistanceMu[calculateIndex()]; }
      AGX_FORCE_INLINE agx::Real const& ContactConstraintPtr::rollingResistanceMu() const { verifyIndex(); return getData()->rollingResistanceMu[calculateIndex()]; }

      AGX_FORCE_INLINE agx::Real& ContactConstraintPtr::twistLimitMultiplier() { verifyIndex(); return getData()->twistLimitMultiplier[calculateIndex()]; }
      AGX_FORCE_INLINE agx::Real const& ContactConstraintPtr::twistLimitMultiplier() const { verifyIndex(); return getData()->twistLimitMultiplier[calculateIndex()]; }

      //-----------------------------------------------------------------------------------------------------
      AGX_FORCE_INLINE ContactConstraintInstance::ContactConstraintInstance() {}
      AGX_FORCE_INLINE ContactConstraintInstance::ContactConstraintInstance(ContactConstraintData* data, agx::Index index) : agxData::EntityInstance(data, index) {}
      AGX_FORCE_INLINE ContactConstraintInstance::ContactConstraintInstance(agxData::EntityStorage* storage, agx::Index index) : agxData::EntityInstance(storage, index) {}
      AGX_FORCE_INLINE ContactConstraintInstance::ContactConstraintInstance(const agxData::EntityInstance& other) : agxData::EntityInstance(other)
      {
        agxAssertN(!other || other.isInstanceOf(ContactConstraintModel::instance()),
          "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
          EntityInstance::getModel()->fullPath().c_str(), ContactConstraintModel::instance()->fullPath().c_str());
      }

      AGX_FORCE_INLINE ContactConstraintInstance::ContactConstraintInstance(const agxData::EntityPtr& ptr) : agxData::EntityInstance(ptr)
      {
        agxAssertN(!ptr || ptr.isInstanceOf(ContactConstraintModel::instance()),
          "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
          EntityInstance::getModel()->fullPath().c_str(), ContactConstraintModel::instance()->fullPath().c_str());
      }


      AGX_FORCE_INLINE ContactConstraintData* ContactConstraintInstance::getData() { return static_cast<ContactConstraintData* >(agxData::EntityInstance::getData()); }
      AGX_FORCE_INLINE const ContactConstraintData* ContactConstraintInstance::getData() const { return static_cast<const ContactConstraintData* >(agxData::EntityInstance::getData()); }

      AGX_FORCE_INLINE agx::UInt32& ContactConstraintInstance::body1() { verifyIndex(); return getData()->body1[getIndex()]; }
      AGX_FORCE_INLINE agx::UInt32 const& ContactConstraintInstance::body1() const { verifyIndex(); return getData()->body1[getIndex()]; }

      AGX_FORCE_INLINE agx::UInt32& ContactConstraintInstance::body2() { verifyIndex(); return getData()->body2[getIndex()]; }
      AGX_FORCE_INLINE agx::UInt32 const& ContactConstraintInstance::body2() const { verifyIndex(); return getData()->body2[getIndex()]; }

      AGX_FORCE_INLINE agx::UInt32& ContactConstraintInstance::solveBody1() { verifyIndex(); return getData()->solveBody1[getIndex()]; }
      AGX_FORCE_INLINE agx::UInt32 const& ContactConstraintInstance::solveBody1() const { verifyIndex(); return getData()->solveBody1[getIndex()]; }

      AGX_FORCE_INLINE agx::UInt32& ContactConstraintInstance::solveBody2() { verifyIndex(); return getData()->solveBody2[getIndex()]; }
      AGX_FORCE_INLINE agx::UInt32 const& ContactConstraintInstance::solveBody2() const { verifyIndex(); return getData()->solveBody2[getIndex()]; }

      AGX_FORCE_INLINE agx::UInt32& ContactConstraintInstance::cachedContactId() { verifyIndex(); return getData()->cachedContactId[getIndex()]; }
      AGX_FORCE_INLINE agx::UInt32 const& ContactConstraintInstance::cachedContactId() const { verifyIndex(); return getData()->cachedContactId[getIndex()]; }

      AGX_FORCE_INLINE agx::UInt32& ContactConstraintInstance::contactIndex() { verifyIndex(); return getData()->contactIndex[getIndex()]; }
      AGX_FORCE_INLINE agx::UInt32 const& ContactConstraintInstance::contactIndex() const { verifyIndex(); return getData()->contactIndex[getIndex()]; }

      AGX_FORCE_INLINE agx::UInt32& ContactConstraintInstance::jacobianIndex() { verifyIndex(); return getData()->jacobianIndex[getIndex()]; }
      AGX_FORCE_INLINE agx::UInt32 const& ContactConstraintInstance::jacobianIndex() const { verifyIndex(); return getData()->jacobianIndex[getIndex()]; }

      AGX_FORCE_INLINE agx::UInt32& ContactConstraintInstance::rowIndex() { verifyIndex(); return getData()->rowIndex[getIndex()]; }
      AGX_FORCE_INLINE agx::UInt32 const& ContactConstraintInstance::rowIndex() const { verifyIndex(); return getData()->rowIndex[getIndex()]; }

      AGX_FORCE_INLINE agx::UInt32& ContactConstraintInstance::materialIndex() { verifyIndex(); return getData()->materialIndex[getIndex()]; }
      AGX_FORCE_INLINE agx::UInt32 const& ContactConstraintInstance::materialIndex() const { verifyIndex(); return getData()->materialIndex[getIndex()]; }

      AGX_FORCE_INLINE agx::UInt32& ContactConstraintInstance::iterationCount() { verifyIndex(); return getData()->iterationCount[getIndex()]; }
      AGX_FORCE_INLINE agx::UInt32 const& ContactConstraintInstance::iterationCount() const { verifyIndex(); return getData()->iterationCount[getIndex()]; }

      AGX_FORCE_INLINE agx::Bool& ContactConstraintInstance::isImpacting() { verifyIndex(); return getData()->isImpacting[getIndex()]; }
      AGX_FORCE_INLINE agx::Bool const& ContactConstraintInstance::isImpacting() const { verifyIndex(); return getData()->isImpacting[getIndex()]; }

      AGX_FORCE_INLINE agx::Real& ContactConstraintInstance::restingCompliance() { verifyIndex(); return getData()->restingCompliance[getIndex()]; }
      AGX_FORCE_INLINE agx::Real const& ContactConstraintInstance::restingCompliance() const { verifyIndex(); return getData()->restingCompliance[getIndex()]; }

      AGX_FORCE_INLINE agx::Real& ContactConstraintInstance::charContactTime() { verifyIndex(); return getData()->charContactTime[getIndex()]; }
      AGX_FORCE_INLINE agx::Real const& ContactConstraintInstance::charContactTime() const { verifyIndex(); return getData()->charContactTime[getIndex()]; }

      AGX_FORCE_INLINE agx::Real& ContactConstraintInstance::charMass() { verifyIndex(); return getData()->charMass[getIndex()]; }
      AGX_FORCE_INLINE agx::Real const& ContactConstraintInstance::charMass() const { verifyIndex(); return getData()->charMass[getIndex()]; }

      AGX_FORCE_INLINE agx::JacobianMeta& ContactConstraintInstance::GMeta() { verifyIndex(); return getData()->GMeta[getIndex()]; }
      AGX_FORCE_INLINE agx::JacobianMeta const& ContactConstraintInstance::GMeta() const { verifyIndex(); return getData()->GMeta[getIndex()]; }

      AGX_FORCE_INLINE agx::Real& ContactConstraintInstance::nonlinearMultiplier() { verifyIndex(); return getData()->nonlinearMultiplier[getIndex()]; }
      AGX_FORCE_INLINE agx::Real const& ContactConstraintInstance::nonlinearMultiplier() const { verifyIndex(); return getData()->nonlinearMultiplier[getIndex()]; }

      AGX_FORCE_INLINE agx::Real& ContactConstraintInstance::rollingResistanceMu() { verifyIndex(); return getData()->rollingResistanceMu[getIndex()]; }
      AGX_FORCE_INLINE agx::Real const& ContactConstraintInstance::rollingResistanceMu() const { verifyIndex(); return getData()->rollingResistanceMu[getIndex()]; }

      AGX_FORCE_INLINE agx::Real& ContactConstraintInstance::twistLimitMultiplier() { verifyIndex(); return getData()->twistLimitMultiplier[getIndex()]; }
      AGX_FORCE_INLINE agx::Real const& ContactConstraintInstance::twistLimitMultiplier() const { verifyIndex(); return getData()->twistLimitMultiplier[getIndex()]; }

      //-----------------------------------------------------------------------------------------------------
      AGX_FORCE_INLINE ContactConstraintSemantics::ContactConstraintSemantics() {}
      //-----------------------------------------------------------------------------------------------------
      DOXYGEN_END_INTERNAL_BLOCK()
    }
  }
}

AGX_TYPE_BINDING(agx::Physics::GranularBody::ContactConstraintPtr, "Physics.GranularBody.ContactConstraintPtr")
AGX_TYPE_BINDING(agx::Physics::GranularBody::ContactConstraintInstance, "Physics.GranularBody.ContactConstraintInstance")

#ifdef _MSC_VER
# pragma warning(pop)
#endif

#undef AGX_ENTITY_WRAPPER
#undef AGX_ENTITY_NAMESPACE
#endif

