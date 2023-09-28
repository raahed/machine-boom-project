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

#ifndef GENERATED_AGX_PHYSICS_CONSTRAINTFORCES_H_PLUGIN
#define GENERATED_AGX_PHYSICS_CONSTRAINTFORCES_H_PLUGIN

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
#include <agx/Name.h>
#include <agx/Vec3.h>
#include <agx/Real.h>


namespace agx
{
  namespace Physics
  {

    class ConstraintForcesModel;
    class ConstraintForcesData;
    class ConstraintForcesPtr;
    class ConstraintForcesInstance;
    class ConstraintForcesSemantics;


    AGX_DECLARE_POINTER_TYPES(ConstraintForcesModel);

    /** 
    Abstract description of the data attributes for the Physics.ConstraintForces entity.
    */ 
    class AGXPHYSICS_EXPORT ConstraintForcesModel : public agxData::EntityModel
    {
    public:
      typedef ConstraintForcesPtr PtrT;

      ConstraintForcesModel(const agx::String& name = "ConstraintForces");

      /// \return The entity model singleton.
      static ConstraintForcesModel* instance();

      /// Create and return a pointer to a new instance in the default storage for this entity model.
      static ConstraintForcesPtr createInstance();

      /// \return The default storage for this entity model.
      static agxData::EntityStorage* defaultStorage();

      /// This is part of internal cleanup and should not be called by users
      virtual void shutdownCleanup() override;



      /* Attributes */
      static agxData::ScalarAttributeT< agx::Name >* nameAttribute;
      static agxData::ScalarAttributeT< agx::Name >* typeAttribute;
      static agxData::ScalarAttributeT< agx::Vec3 >* body1ForceAttribute;
      static agxData::ScalarAttributeT< agx::Vec3 >* body2ForceAttribute;
      static agxData::ScalarAttributeT< agx::Vec3 >* body1TorqueAttribute;
      static agxData::ScalarAttributeT< agx::Vec3 >* body2TorqueAttribute;
      static agxData::ScalarAttributeT< agx::Real >* motor1ForceAttribute;
      static agxData::ScalarAttributeT< agx::Real >* motor2ForceAttribute;
      static agxData::ScalarAttributeT< agx::Real >* lock1ForceAttribute;
      static agxData::ScalarAttributeT< agx::Real >* lock2ForceAttribute;
      static agxData::ScalarAttributeT< agx::Real >* range1ForceAttribute;
      static agxData::ScalarAttributeT< agx::Real >* range2ForceAttribute;
      static agxData::ScalarAttributeT< agx::Real >* angle1Attribute;
      static agxData::ScalarAttributeT< agx::Real >* angle2Attribute;
      static agxData::ScalarAttributeT< agx::Real >* currentSpeed1Attribute;
      static agxData::ScalarAttributeT< agx::Real >* currentSpeed2Attribute;

    protected:
      virtual ~ConstraintForcesModel();
      virtual agxData::EntityData* createData(agxData::EntityStorage* storage) override;
      virtual void configure(agx::TiXmlElement* eEntity) override;
      virtual void initAttributeAccessors() override;
      virtual void construct(agxData::EntityPtr instance) override;
      void construct(agx::Physics::ConstraintForcesPtr constraintForces);
    };


    DOXYGEN_START_INTERNAL_BLOCK()
    #ifndef AGX_PHYSICS_CONSTRAINTFORCES_DATA_SET_OVERRIDE
    #define AGX_PHYSICS_CONSTRAINTFORCES_DATA_SET
    class AGXPHYSICS_EXPORT ConstraintForcesData : public agxData::EntityData
    {
    public:
      ConstraintForcesInstance operator[] (size_t index);

    public:
      agxData::Array< ConstraintForcesPtr >& instance;
      agxData::Array< agx::Name > name;
      agxData::Array< agx::Name > type;
      agxData::Array< agx::Vec3 > body1Force;
      agxData::Array< agx::Vec3 > body2Force;
      agxData::Array< agx::Vec3 > body1Torque;
      agxData::Array< agx::Vec3 > body2Torque;
      agxData::Array< agx::Real > motor1Force;
      agxData::Array< agx::Real > motor2Force;
      agxData::Array< agx::Real > lock1Force;
      agxData::Array< agx::Real > lock2Force;
      agxData::Array< agx::Real > range1Force;
      agxData::Array< agx::Real > range2Force;
      agxData::Array< agx::Real > angle1;
      agxData::Array< agx::Real > angle2;
      agxData::Array< agx::Real > currentSpeed1;
      agxData::Array< agx::Real > currentSpeed2;

    public:
      typedef agx::Name nameType;
      typedef agx::Name typeType;
      typedef agx::Vec3 body1ForceType;
      typedef agx::Vec3 body2ForceType;
      typedef agx::Vec3 body1TorqueType;
      typedef agx::Vec3 body2TorqueType;
      typedef agx::Real motor1ForceType;
      typedef agx::Real motor2ForceType;
      typedef agx::Real lock1ForceType;
      typedef agx::Real lock2ForceType;
      typedef agx::Real range1ForceType;
      typedef agx::Real range2ForceType;
      typedef agx::Real angle1Type;
      typedef agx::Real angle2Type;
      typedef agx::Real currentSpeed1Type;
      typedef agx::Real currentSpeed2Type;

    public:
      ConstraintForcesData(agxData::EntityStorage* storage);
      ConstraintForcesData();

    protected:
      virtual ~ConstraintForcesData() {}
      virtual void setNumElements(agx::Index numElements) override;

    private:
      ConstraintForcesData& operator= (const ConstraintForcesData&) { return *this; }

    };
    #endif
    DOXYGEN_END_INTERNAL_BLOCK()


    DOXYGEN_START_INTERNAL_BLOCK()
    class AGXPHYSICS_EXPORT ConstraintForcesSemantics : protected agxData::EntityPtr
    {
    public:

      // Automatic getters
      agx::Name const& getName() const;
      agx::Name const& getType() const;
      agx::Vec3 const& getBody1Force() const;
      agx::Vec3 const& getBody2Force() const;
      agx::Vec3 const& getBody1Torque() const;
      agx::Vec3 const& getBody2Torque() const;
      agx::Real const& getMotor1Force() const;
      agx::Real const& getMotor2Force() const;
      agx::Real const& getLock1Force() const;
      agx::Real const& getLock2Force() const;
      agx::Real const& getRange1Force() const;
      agx::Real const& getRange2Force() const;
      agx::Real const& getAngle1() const;
      agx::Real const& getAngle2() const;
      agx::Real const& getCurrentSpeed1() const;
      agx::Real const& getCurrentSpeed2() const;

      // Semantics defined by explicit kernels

      // Automatic setters
      void setName(agx::Name const& value);
      void setType(agx::Name const& value);
      void setBody1Force(agx::Vec3 const& value);
      void setBody2Force(agx::Vec3 const& value);
      void setBody1Torque(agx::Vec3 const& value);
      void setBody2Torque(agx::Vec3 const& value);
      void setMotor1Force(agx::Real const& value);
      void setMotor2Force(agx::Real const& value);
      void setLock1Force(agx::Real const& value);
      void setLock2Force(agx::Real const& value);
      void setRange1Force(agx::Real const& value);
      void setRange2Force(agx::Real const& value);
      void setAngle1(agx::Real const& value);
      void setAngle2(agx::Real const& value);
      void setCurrentSpeed1(agx::Real const& value);
      void setCurrentSpeed2(agx::Real const& value);


    protected:
      friend class ConstraintForcesPtr;
      friend class ConstraintForcesInstance;
      ConstraintForcesSemantics();
    };
    DOXYGEN_END_INTERNAL_BLOCK()


    /**
    Pointer to a entity instance of type Physics.ConstraintForces
    */
    class CALLABLE ConstraintForcesPtr : public agxData::EntityPtr
    {
    public:
      typedef ConstraintForcesModel ModelType;
      typedef ConstraintForcesData DataType;
      typedef ConstraintForcesInstance InstanceType;

    public:
      AGXPHYSICS_EXPORT ConstraintForcesPtr();
      AGXPHYSICS_EXPORT ConstraintForcesPtr(agxData::EntityStorage* storage, agx::Index id);
      AGXPHYSICS_EXPORT ConstraintForcesPtr(const agxData::EntityPtr& ptr);
      AGXPHYSICS_EXPORT ConstraintForcesPtr(const agxData::EntityInstance& instance);
      AGXPHYSICS_EXPORT ConstraintForcesPtr& operator= (const agxData::EntityPtr& ptr);
      AGXPHYSICS_EXPORT ConstraintForcesPtr& operator= (const agxData::EntityInstance& instance);
      AGXPHYSICS_EXPORT ConstraintForcesInstance instance();
      AGXPHYSICS_EXPORT const ConstraintForcesInstance instance() const;

      AGXPHYSICS_EXPORT ConstraintForcesSemantics* operator->();
      AGXPHYSICS_EXPORT const ConstraintForcesSemantics* operator->() const;

      ConstraintForcesData* getData();
      const ConstraintForcesData* getData() const;


      /// \return reference to the name attribute
      AGXPHYSICS_EXPORT agx::Name& name();
      /// \return const reference to the name attribute
      AGXPHYSICS_EXPORT agx::Name const& name() const;

      /// \return reference to the type attribute
      AGXPHYSICS_EXPORT agx::Name& type();
      /// \return const reference to the type attribute
      AGXPHYSICS_EXPORT agx::Name const& type() const;

      /// \return reference to the body1Force attribute
      AGXPHYSICS_EXPORT agx::Vec3& body1Force();
      /// \return const reference to the body1Force attribute
      AGXPHYSICS_EXPORT agx::Vec3 const& body1Force() const;

      /// \return reference to the body2Force attribute
      AGXPHYSICS_EXPORT agx::Vec3& body2Force();
      /// \return const reference to the body2Force attribute
      AGXPHYSICS_EXPORT agx::Vec3 const& body2Force() const;

      /// \return reference to the body1Torque attribute
      AGXPHYSICS_EXPORT agx::Vec3& body1Torque();
      /// \return const reference to the body1Torque attribute
      AGXPHYSICS_EXPORT agx::Vec3 const& body1Torque() const;

      /// \return reference to the body2Torque attribute
      AGXPHYSICS_EXPORT agx::Vec3& body2Torque();
      /// \return const reference to the body2Torque attribute
      AGXPHYSICS_EXPORT agx::Vec3 const& body2Torque() const;

      /// \return reference to the motor1Force attribute
      AGXPHYSICS_EXPORT agx::Real& motor1Force();
      /// \return const reference to the motor1Force attribute
      AGXPHYSICS_EXPORT agx::Real const& motor1Force() const;

      /// \return reference to the motor2Force attribute
      AGXPHYSICS_EXPORT agx::Real& motor2Force();
      /// \return const reference to the motor2Force attribute
      AGXPHYSICS_EXPORT agx::Real const& motor2Force() const;

      /// \return reference to the lock1Force attribute
      AGXPHYSICS_EXPORT agx::Real& lock1Force();
      /// \return const reference to the lock1Force attribute
      AGXPHYSICS_EXPORT agx::Real const& lock1Force() const;

      /// \return reference to the lock2Force attribute
      AGXPHYSICS_EXPORT agx::Real& lock2Force();
      /// \return const reference to the lock2Force attribute
      AGXPHYSICS_EXPORT agx::Real const& lock2Force() const;

      /// \return reference to the range1Force attribute
      AGXPHYSICS_EXPORT agx::Real& range1Force();
      /// \return const reference to the range1Force attribute
      AGXPHYSICS_EXPORT agx::Real const& range1Force() const;

      /// \return reference to the range2Force attribute
      AGXPHYSICS_EXPORT agx::Real& range2Force();
      /// \return const reference to the range2Force attribute
      AGXPHYSICS_EXPORT agx::Real const& range2Force() const;

      /// \return reference to the angle1 attribute
      AGXPHYSICS_EXPORT agx::Real& angle1();
      /// \return const reference to the angle1 attribute
      AGXPHYSICS_EXPORT agx::Real const& angle1() const;

      /// \return reference to the angle2 attribute
      AGXPHYSICS_EXPORT agx::Real& angle2();
      /// \return const reference to the angle2 attribute
      AGXPHYSICS_EXPORT agx::Real const& angle2() const;

      /// \return reference to the currentSpeed1 attribute
      AGXPHYSICS_EXPORT agx::Real& currentSpeed1();
      /// \return const reference to the currentSpeed1 attribute
      AGXPHYSICS_EXPORT agx::Real const& currentSpeed1() const;

      /// \return reference to the currentSpeed2 attribute
      AGXPHYSICS_EXPORT agx::Real& currentSpeed2();
      /// \return const reference to the currentSpeed2 attribute
      AGXPHYSICS_EXPORT agx::Real const& currentSpeed2() const;

      DOXYGEN_START_INTERNAL_BLOCK()
      // Only used by tolua!
      #ifdef AGX_GENERATED_TOLUA_WRAPPER
      inline agx::Name const& getName() const { return (*this)->getName(); }
      inline agx::Name const& getType() const { return (*this)->getType(); }
      inline agx::Vec3 const& getBody1Force() const { return (*this)->getBody1Force(); }
      inline agx::Vec3 const& getBody2Force() const { return (*this)->getBody2Force(); }
      inline agx::Vec3 const& getBody1Torque() const { return (*this)->getBody1Torque(); }
      inline agx::Vec3 const& getBody2Torque() const { return (*this)->getBody2Torque(); }
      inline agx::Real const& getMotor1Force() const { return (*this)->getMotor1Force(); }
      inline agx::Real const& getMotor2Force() const { return (*this)->getMotor2Force(); }
      inline agx::Real const& getLock1Force() const { return (*this)->getLock1Force(); }
      inline agx::Real const& getLock2Force() const { return (*this)->getLock2Force(); }
      inline agx::Real const& getRange1Force() const { return (*this)->getRange1Force(); }
      inline agx::Real const& getRange2Force() const { return (*this)->getRange2Force(); }
      inline agx::Real const& getAngle1() const { return (*this)->getAngle1(); }
      inline agx::Real const& getAngle2() const { return (*this)->getAngle2(); }
      inline agx::Real const& getCurrentSpeed1() const { return (*this)->getCurrentSpeed1(); }
      inline agx::Real const& getCurrentSpeed2() const { return (*this)->getCurrentSpeed2(); }


      inline void setName(agx::Name const& value) { (*this)->setName(value); }
      inline void setType(agx::Name const& value) { (*this)->setType(value); }
      inline void setBody1Force(agx::Vec3 const& value) { (*this)->setBody1Force(value); }
      inline void setBody2Force(agx::Vec3 const& value) { (*this)->setBody2Force(value); }
      inline void setBody1Torque(agx::Vec3 const& value) { (*this)->setBody1Torque(value); }
      inline void setBody2Torque(agx::Vec3 const& value) { (*this)->setBody2Torque(value); }
      inline void setMotor1Force(agx::Real const& value) { (*this)->setMotor1Force(value); }
      inline void setMotor2Force(agx::Real const& value) { (*this)->setMotor2Force(value); }
      inline void setLock1Force(agx::Real const& value) { (*this)->setLock1Force(value); }
      inline void setLock2Force(agx::Real const& value) { (*this)->setLock2Force(value); }
      inline void setRange1Force(agx::Real const& value) { (*this)->setRange1Force(value); }
      inline void setRange2Force(agx::Real const& value) { (*this)->setRange2Force(value); }
      inline void setAngle1(agx::Real const& value) { (*this)->setAngle1(value); }
      inline void setAngle2(agx::Real const& value) { (*this)->setAngle2(value); }
      inline void setCurrentSpeed1(agx::Real const& value) { (*this)->setCurrentSpeed1(value); }
      inline void setCurrentSpeed2(agx::Real const& value) { (*this)->setCurrentSpeed2(value); }
      #endif
      DOXYGEN_END_INTERNAL_BLOCK()

    };


    DOXYGEN_START_INTERNAL_BLOCK()
    class AGXPHYSICS_EXPORT ConstraintForcesInstance : public agxData::EntityInstance
    {
    public:
      ConstraintForcesInstance();
      ConstraintForcesInstance(ConstraintForcesData* data, agx::Index index);
      ConstraintForcesInstance(agxData::EntityStorage *storage, agx::Index index);
      ConstraintForcesInstance(const agxData::EntityInstance& other);
      ConstraintForcesInstance(const agxData::EntityPtr& ptr);

      ConstraintForcesData* getData();
      const ConstraintForcesData* getData() const;

    public:
      /// \return reference to the name attribute
      agx::Name& name();
      /// \return const reference to the name attribute
      agx::Name const& name() const;

      /// \return reference to the type attribute
      agx::Name& type();
      /// \return const reference to the type attribute
      agx::Name const& type() const;

      /// \return reference to the body1Force attribute
      agx::Vec3& body1Force();
      /// \return const reference to the body1Force attribute
      agx::Vec3 const& body1Force() const;

      /// \return reference to the body2Force attribute
      agx::Vec3& body2Force();
      /// \return const reference to the body2Force attribute
      agx::Vec3 const& body2Force() const;

      /// \return reference to the body1Torque attribute
      agx::Vec3& body1Torque();
      /// \return const reference to the body1Torque attribute
      agx::Vec3 const& body1Torque() const;

      /// \return reference to the body2Torque attribute
      agx::Vec3& body2Torque();
      /// \return const reference to the body2Torque attribute
      agx::Vec3 const& body2Torque() const;

      /// \return reference to the motor1Force attribute
      agx::Real& motor1Force();
      /// \return const reference to the motor1Force attribute
      agx::Real const& motor1Force() const;

      /// \return reference to the motor2Force attribute
      agx::Real& motor2Force();
      /// \return const reference to the motor2Force attribute
      agx::Real const& motor2Force() const;

      /// \return reference to the lock1Force attribute
      agx::Real& lock1Force();
      /// \return const reference to the lock1Force attribute
      agx::Real const& lock1Force() const;

      /// \return reference to the lock2Force attribute
      agx::Real& lock2Force();
      /// \return const reference to the lock2Force attribute
      agx::Real const& lock2Force() const;

      /// \return reference to the range1Force attribute
      agx::Real& range1Force();
      /// \return const reference to the range1Force attribute
      agx::Real const& range1Force() const;

      /// \return reference to the range2Force attribute
      agx::Real& range2Force();
      /// \return const reference to the range2Force attribute
      agx::Real const& range2Force() const;

      /// \return reference to the angle1 attribute
      agx::Real& angle1();
      /// \return const reference to the angle1 attribute
      agx::Real const& angle1() const;

      /// \return reference to the angle2 attribute
      agx::Real& angle2();
      /// \return const reference to the angle2 attribute
      agx::Real const& angle2() const;

      /// \return reference to the currentSpeed1 attribute
      agx::Real& currentSpeed1();
      /// \return const reference to the currentSpeed1 attribute
      agx::Real const& currentSpeed1() const;

      /// \return reference to the currentSpeed2 attribute
      agx::Real& currentSpeed2();
      /// \return const reference to the currentSpeed2 attribute
      agx::Real const& currentSpeed2() const;

    };
    DOXYGEN_END_INTERNAL_BLOCK()



    typedef agx::VectorPOD<ConstraintForcesPtr> ConstraintForcesPtrVector;
    typedef agxData::Array<ConstraintForcesPtr> ConstraintForcesPtrArray;



    DOXYGEN_START_INTERNAL_BLOCK()
    /* Implementation */
    //-----------------------------------------------------------------------------------------------------
    //-----------------------------------------------------------------------------------------------------
    inline ConstraintForcesInstance agx::Physics::ConstraintForcesData::operator[] (size_t index) { return ConstraintForcesInstance(this, (agx::Index)index); }
    //-----------------------------------------------------------------------------------------------------
    AGX_FORCE_INLINE ConstraintForcesPtr::ConstraintForcesPtr() {}
    AGX_FORCE_INLINE ConstraintForcesPtr::ConstraintForcesPtr(agxData::EntityStorage* storage, agx::Index id) : agxData::EntityPtr(storage, id) {}
    AGX_FORCE_INLINE ConstraintForcesPtr::ConstraintForcesPtr(const agxData::EntityPtr& ptr) : agxData::EntityPtr(ptr)
    {
      agxAssertN(!ptr || ptr.isInstanceOf(ConstraintForcesModel::instance()),
        "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
        EntityPtr::getModel()->fullPath().c_str(), ConstraintForcesModel::instance()->fullPath().c_str());
    }

    AGX_FORCE_INLINE ConstraintForcesPtr::ConstraintForcesPtr(const agxData::EntityInstance& instance) : agxData::EntityPtr(instance)
    {
      agxAssertN(!instance || instance.isInstanceOf(ConstraintForcesModel::instance()),
        "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
        EntityPtr::getModel()->fullPath().c_str(), ConstraintForcesModel::instance()->fullPath().c_str());
    }

    AGX_FORCE_INLINE ConstraintForcesPtr& ConstraintForcesPtr::operator= (const agxData::EntityPtr& ptr)
    {
      agxData::EntityPtr::operator= (ptr);
      agxAssertN(!ptr || ptr.isInstanceOf(ConstraintForcesModel::instance()),
        "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
        EntityPtr::getModel()->fullPath().c_str(), ConstraintForcesModel::instance()->fullPath().c_str());
      return *this;
    }

    AGX_FORCE_INLINE ConstraintForcesPtr& ConstraintForcesPtr::operator= (const agxData::EntityInstance& instance)
    {
      agxData::EntityPtr::operator= (instance);
      agxAssertN(!instance || instance.isInstanceOf(ConstraintForcesModel::instance()),
        "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
        EntityPtr::getModel()->fullPath().c_str(), ConstraintForcesModel::instance()->fullPath().c_str());
      return *this;
    }

    inline ConstraintForcesInstance ConstraintForcesPtr::instance() { return agxData::EntityPtr::instance(); }
    inline const ConstraintForcesInstance ConstraintForcesPtr::instance() const { return agxData::EntityPtr::instance(); }
    AGX_FORCE_INLINE ConstraintForcesSemantics* ConstraintForcesPtr::operator->() { return (ConstraintForcesSemantics* )this; }
    AGX_FORCE_INLINE const ConstraintForcesSemantics* ConstraintForcesPtr::operator->() const { return (const ConstraintForcesSemantics* )this; }
    AGX_FORCE_INLINE ConstraintForcesData* ConstraintForcesPtr::getData() { return static_cast<ConstraintForcesData* >(agxData::EntityPtr::getData()); }
    AGX_FORCE_INLINE const ConstraintForcesData* ConstraintForcesPtr::getData() const { return static_cast<const ConstraintForcesData* >(agxData::EntityPtr::getData()); }

    AGX_FORCE_INLINE agx::Name& ConstraintForcesPtr::name() { verifyIndex(); return getData()->name[calculateIndex()]; }
    AGX_FORCE_INLINE agx::Name const& ConstraintForcesPtr::name() const { verifyIndex(); return getData()->name[calculateIndex()]; }

    AGX_FORCE_INLINE agx::Name& ConstraintForcesPtr::type() { verifyIndex(); return getData()->type[calculateIndex()]; }
    AGX_FORCE_INLINE agx::Name const& ConstraintForcesPtr::type() const { verifyIndex(); return getData()->type[calculateIndex()]; }

    AGX_FORCE_INLINE agx::Vec3& ConstraintForcesPtr::body1Force() { verifyIndex(); return getData()->body1Force[calculateIndex()]; }
    AGX_FORCE_INLINE agx::Vec3 const& ConstraintForcesPtr::body1Force() const { verifyIndex(); return getData()->body1Force[calculateIndex()]; }

    AGX_FORCE_INLINE agx::Vec3& ConstraintForcesPtr::body2Force() { verifyIndex(); return getData()->body2Force[calculateIndex()]; }
    AGX_FORCE_INLINE agx::Vec3 const& ConstraintForcesPtr::body2Force() const { verifyIndex(); return getData()->body2Force[calculateIndex()]; }

    AGX_FORCE_INLINE agx::Vec3& ConstraintForcesPtr::body1Torque() { verifyIndex(); return getData()->body1Torque[calculateIndex()]; }
    AGX_FORCE_INLINE agx::Vec3 const& ConstraintForcesPtr::body1Torque() const { verifyIndex(); return getData()->body1Torque[calculateIndex()]; }

    AGX_FORCE_INLINE agx::Vec3& ConstraintForcesPtr::body2Torque() { verifyIndex(); return getData()->body2Torque[calculateIndex()]; }
    AGX_FORCE_INLINE agx::Vec3 const& ConstraintForcesPtr::body2Torque() const { verifyIndex(); return getData()->body2Torque[calculateIndex()]; }

    AGX_FORCE_INLINE agx::Real& ConstraintForcesPtr::motor1Force() { verifyIndex(); return getData()->motor1Force[calculateIndex()]; }
    AGX_FORCE_INLINE agx::Real const& ConstraintForcesPtr::motor1Force() const { verifyIndex(); return getData()->motor1Force[calculateIndex()]; }

    AGX_FORCE_INLINE agx::Real& ConstraintForcesPtr::motor2Force() { verifyIndex(); return getData()->motor2Force[calculateIndex()]; }
    AGX_FORCE_INLINE agx::Real const& ConstraintForcesPtr::motor2Force() const { verifyIndex(); return getData()->motor2Force[calculateIndex()]; }

    AGX_FORCE_INLINE agx::Real& ConstraintForcesPtr::lock1Force() { verifyIndex(); return getData()->lock1Force[calculateIndex()]; }
    AGX_FORCE_INLINE agx::Real const& ConstraintForcesPtr::lock1Force() const { verifyIndex(); return getData()->lock1Force[calculateIndex()]; }

    AGX_FORCE_INLINE agx::Real& ConstraintForcesPtr::lock2Force() { verifyIndex(); return getData()->lock2Force[calculateIndex()]; }
    AGX_FORCE_INLINE agx::Real const& ConstraintForcesPtr::lock2Force() const { verifyIndex(); return getData()->lock2Force[calculateIndex()]; }

    AGX_FORCE_INLINE agx::Real& ConstraintForcesPtr::range1Force() { verifyIndex(); return getData()->range1Force[calculateIndex()]; }
    AGX_FORCE_INLINE agx::Real const& ConstraintForcesPtr::range1Force() const { verifyIndex(); return getData()->range1Force[calculateIndex()]; }

    AGX_FORCE_INLINE agx::Real& ConstraintForcesPtr::range2Force() { verifyIndex(); return getData()->range2Force[calculateIndex()]; }
    AGX_FORCE_INLINE agx::Real const& ConstraintForcesPtr::range2Force() const { verifyIndex(); return getData()->range2Force[calculateIndex()]; }

    AGX_FORCE_INLINE agx::Real& ConstraintForcesPtr::angle1() { verifyIndex(); return getData()->angle1[calculateIndex()]; }
    AGX_FORCE_INLINE agx::Real const& ConstraintForcesPtr::angle1() const { verifyIndex(); return getData()->angle1[calculateIndex()]; }

    AGX_FORCE_INLINE agx::Real& ConstraintForcesPtr::angle2() { verifyIndex(); return getData()->angle2[calculateIndex()]; }
    AGX_FORCE_INLINE agx::Real const& ConstraintForcesPtr::angle2() const { verifyIndex(); return getData()->angle2[calculateIndex()]; }

    AGX_FORCE_INLINE agx::Real& ConstraintForcesPtr::currentSpeed1() { verifyIndex(); return getData()->currentSpeed1[calculateIndex()]; }
    AGX_FORCE_INLINE agx::Real const& ConstraintForcesPtr::currentSpeed1() const { verifyIndex(); return getData()->currentSpeed1[calculateIndex()]; }

    AGX_FORCE_INLINE agx::Real& ConstraintForcesPtr::currentSpeed2() { verifyIndex(); return getData()->currentSpeed2[calculateIndex()]; }
    AGX_FORCE_INLINE agx::Real const& ConstraintForcesPtr::currentSpeed2() const { verifyIndex(); return getData()->currentSpeed2[calculateIndex()]; }

    //-----------------------------------------------------------------------------------------------------
    AGX_FORCE_INLINE ConstraintForcesInstance::ConstraintForcesInstance() {}
    AGX_FORCE_INLINE ConstraintForcesInstance::ConstraintForcesInstance(ConstraintForcesData* data, agx::Index index) : agxData::EntityInstance(data, index) {}
    AGX_FORCE_INLINE ConstraintForcesInstance::ConstraintForcesInstance(agxData::EntityStorage* storage, agx::Index index) : agxData::EntityInstance(storage, index) {}
    AGX_FORCE_INLINE ConstraintForcesInstance::ConstraintForcesInstance(const agxData::EntityInstance& other) : agxData::EntityInstance(other)
    {
      agxAssertN(!other || other.isInstanceOf(ConstraintForcesModel::instance()),
        "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
        EntityInstance::getModel()->fullPath().c_str(), ConstraintForcesModel::instance()->fullPath().c_str());
    }

    AGX_FORCE_INLINE ConstraintForcesInstance::ConstraintForcesInstance(const agxData::EntityPtr& ptr) : agxData::EntityInstance(ptr)
    {
      agxAssertN(!ptr || ptr.isInstanceOf(ConstraintForcesModel::instance()),
        "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
        EntityInstance::getModel()->fullPath().c_str(), ConstraintForcesModel::instance()->fullPath().c_str());
    }


    AGX_FORCE_INLINE ConstraintForcesData* ConstraintForcesInstance::getData() { return static_cast<ConstraintForcesData* >(agxData::EntityInstance::getData()); }
    AGX_FORCE_INLINE const ConstraintForcesData* ConstraintForcesInstance::getData() const { return static_cast<const ConstraintForcesData* >(agxData::EntityInstance::getData()); }

    AGX_FORCE_INLINE agx::Name& ConstraintForcesInstance::name() { verifyIndex(); return getData()->name[getIndex()]; }
    AGX_FORCE_INLINE agx::Name const& ConstraintForcesInstance::name() const { verifyIndex(); return getData()->name[getIndex()]; }

    AGX_FORCE_INLINE agx::Name& ConstraintForcesInstance::type() { verifyIndex(); return getData()->type[getIndex()]; }
    AGX_FORCE_INLINE agx::Name const& ConstraintForcesInstance::type() const { verifyIndex(); return getData()->type[getIndex()]; }

    AGX_FORCE_INLINE agx::Vec3& ConstraintForcesInstance::body1Force() { verifyIndex(); return getData()->body1Force[getIndex()]; }
    AGX_FORCE_INLINE agx::Vec3 const& ConstraintForcesInstance::body1Force() const { verifyIndex(); return getData()->body1Force[getIndex()]; }

    AGX_FORCE_INLINE agx::Vec3& ConstraintForcesInstance::body2Force() { verifyIndex(); return getData()->body2Force[getIndex()]; }
    AGX_FORCE_INLINE agx::Vec3 const& ConstraintForcesInstance::body2Force() const { verifyIndex(); return getData()->body2Force[getIndex()]; }

    AGX_FORCE_INLINE agx::Vec3& ConstraintForcesInstance::body1Torque() { verifyIndex(); return getData()->body1Torque[getIndex()]; }
    AGX_FORCE_INLINE agx::Vec3 const& ConstraintForcesInstance::body1Torque() const { verifyIndex(); return getData()->body1Torque[getIndex()]; }

    AGX_FORCE_INLINE agx::Vec3& ConstraintForcesInstance::body2Torque() { verifyIndex(); return getData()->body2Torque[getIndex()]; }
    AGX_FORCE_INLINE agx::Vec3 const& ConstraintForcesInstance::body2Torque() const { verifyIndex(); return getData()->body2Torque[getIndex()]; }

    AGX_FORCE_INLINE agx::Real& ConstraintForcesInstance::motor1Force() { verifyIndex(); return getData()->motor1Force[getIndex()]; }
    AGX_FORCE_INLINE agx::Real const& ConstraintForcesInstance::motor1Force() const { verifyIndex(); return getData()->motor1Force[getIndex()]; }

    AGX_FORCE_INLINE agx::Real& ConstraintForcesInstance::motor2Force() { verifyIndex(); return getData()->motor2Force[getIndex()]; }
    AGX_FORCE_INLINE agx::Real const& ConstraintForcesInstance::motor2Force() const { verifyIndex(); return getData()->motor2Force[getIndex()]; }

    AGX_FORCE_INLINE agx::Real& ConstraintForcesInstance::lock1Force() { verifyIndex(); return getData()->lock1Force[getIndex()]; }
    AGX_FORCE_INLINE agx::Real const& ConstraintForcesInstance::lock1Force() const { verifyIndex(); return getData()->lock1Force[getIndex()]; }

    AGX_FORCE_INLINE agx::Real& ConstraintForcesInstance::lock2Force() { verifyIndex(); return getData()->lock2Force[getIndex()]; }
    AGX_FORCE_INLINE agx::Real const& ConstraintForcesInstance::lock2Force() const { verifyIndex(); return getData()->lock2Force[getIndex()]; }

    AGX_FORCE_INLINE agx::Real& ConstraintForcesInstance::range1Force() { verifyIndex(); return getData()->range1Force[getIndex()]; }
    AGX_FORCE_INLINE agx::Real const& ConstraintForcesInstance::range1Force() const { verifyIndex(); return getData()->range1Force[getIndex()]; }

    AGX_FORCE_INLINE agx::Real& ConstraintForcesInstance::range2Force() { verifyIndex(); return getData()->range2Force[getIndex()]; }
    AGX_FORCE_INLINE agx::Real const& ConstraintForcesInstance::range2Force() const { verifyIndex(); return getData()->range2Force[getIndex()]; }

    AGX_FORCE_INLINE agx::Real& ConstraintForcesInstance::angle1() { verifyIndex(); return getData()->angle1[getIndex()]; }
    AGX_FORCE_INLINE agx::Real const& ConstraintForcesInstance::angle1() const { verifyIndex(); return getData()->angle1[getIndex()]; }

    AGX_FORCE_INLINE agx::Real& ConstraintForcesInstance::angle2() { verifyIndex(); return getData()->angle2[getIndex()]; }
    AGX_FORCE_INLINE agx::Real const& ConstraintForcesInstance::angle2() const { verifyIndex(); return getData()->angle2[getIndex()]; }

    AGX_FORCE_INLINE agx::Real& ConstraintForcesInstance::currentSpeed1() { verifyIndex(); return getData()->currentSpeed1[getIndex()]; }
    AGX_FORCE_INLINE agx::Real const& ConstraintForcesInstance::currentSpeed1() const { verifyIndex(); return getData()->currentSpeed1[getIndex()]; }

    AGX_FORCE_INLINE agx::Real& ConstraintForcesInstance::currentSpeed2() { verifyIndex(); return getData()->currentSpeed2[getIndex()]; }
    AGX_FORCE_INLINE agx::Real const& ConstraintForcesInstance::currentSpeed2() const { verifyIndex(); return getData()->currentSpeed2[getIndex()]; }

    //-----------------------------------------------------------------------------------------------------
    AGX_FORCE_INLINE ConstraintForcesSemantics::ConstraintForcesSemantics() {}
    //-----------------------------------------------------------------------------------------------------
    DOXYGEN_END_INTERNAL_BLOCK()
  }
}

AGX_TYPE_BINDING(agx::Physics::ConstraintForcesPtr, "Physics.ConstraintForcesPtr")
AGX_TYPE_BINDING(agx::Physics::ConstraintForcesInstance, "Physics.ConstraintForcesInstance")

#ifdef _MSC_VER
# pragma warning(pop)
#endif

#undef AGX_ENTITY_WRAPPER
#undef AGX_ENTITY_NAMESPACE
#endif

