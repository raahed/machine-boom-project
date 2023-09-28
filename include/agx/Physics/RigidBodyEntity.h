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

#ifndef GENERATED_AGX_PHYSICS_RIGIDBODY_H_PLUGIN
#define GENERATED_AGX_PHYSICS_RIGIDBODY_H_PLUGIN

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
#include <agx/ReferencedEntity.h>
#include <agx/Integer.h>
#include <agx/RigidBodyState.h>
#include <agx/Name.h>
#include <agx/Real.h>
#include <agx/Vec3.h>
#include <agx/SPDMatrix3x3.h>
#include <agx/Matrix3x3.h>
#include <agx/AffineMatrix4x4.h>
#include <agx/Physics/GeometryEntity.h>
#include <agx/Physics/GraphNodeEntity.h>
#include <agx/Range6.h>
namespace agx { class RigidBody; }

namespace agx { namespace Physics { class GeometryPtr; }}
namespace agx { namespace Physics { class GraphNodePtr; }}

namespace agx
{
  namespace Physics
  {

    class RigidBodyModel;
    class RigidBodyData;
    class RigidBodyPtr;
    class RigidBodyInstance;
    class RigidBodySemantics;


    AGX_DECLARE_POINTER_TYPES(RigidBodyModel);

    /** 
    Abstract description of the data attributes for the Physics.RigidBody entity.
    */ 
    class AGXPHYSICS_EXPORT RigidBodyModel : public agx::ReferencedModel
    {
    public:
      typedef RigidBodyPtr PtrT;

      RigidBodyModel(const agx::String& name = "RigidBody");

      /// \return The entity model singleton.
      static RigidBodyModel* instance();

      /// Create and return a pointer to a new instance in the default storage for this entity model.
      static RigidBodyPtr createInstance();

      /// \return The default storage for this entity model.
      static agxData::EntityStorage* defaultStorage();

      /// This is part of internal cleanup and should not be called by users
      virtual void shutdownCleanup() override;



      /* Attributes */
      static agxData::ScalarAttributeT< agx::UInt32 >* idAttribute;
      static agxData::ScalarAttributeT< agx::RigidBodyState >* stateAttribute;
      static agxData::ScalarAttributeT< agx::Name >* nameAttribute;
      static agxData::ScalarAttributeT< agx::Real >* massAttribute;
      static agxData::ScalarAttributeT< agx::Real >* invMassAttribute;
      static agxData::ScalarAttributeT< agx::Vec3 >* effectiveMassAttribute;
      static agxData::ScalarAttributeT< agx::Vec3 >* effectiveMassCoefficientsAttribute;
      static agxData::ScalarAttributeT< agx::Vec3 >* effectiveMassTransformDiagonalAttribute;
      static agxData::ScalarAttributeT< agx::SPDMatrix3x3 >* inertiaAttribute;
      static agxData::ScalarAttributeT< agx::SPDMatrix3x3 >* effectiveInertiaAttribute;
      static agxData::ScalarAttributeT< agx::Vec3 >* effectiveInertiaCoefficientsAttribute;
      static agxData::ScalarAttributeT< agx::Matrix3x3 >* worldMassMatrixAttribute;
      static agxData::ScalarAttributeT< agx::Matrix3x3 >* invWorldMassMatrixAttribute;
      static agxData::ScalarAttributeT< agx::Matrix3x3 >* effectiveWorldMassMatrixAttribute;
      static agxData::ScalarAttributeT< agx::Matrix3x3 >* effectiveInvWorldMassMatrixAttribute;
      static agxData::ScalarAttributeT< agx::Matrix3x3 >* worldInertiaAttribute;
      static agxData::ScalarAttributeT< agx::Matrix3x3 >* invWorldInertiaAttribute;
      static agxData::ScalarAttributeT< agx::Matrix3x3 >* effectiveWorldInertiaAttribute;
      static agxData::ScalarAttributeT< agx::Matrix3x3 >* effectiveInvWorldInertiaAttribute;
      static agxData::ScalarAttributeT< agx::Vec3 >* forceAttribute;
      static agxData::ScalarAttributeT< agx::Vec3 >* torqueAttribute;
      static agxData::ScalarAttributeT< agx::Vec3 >* lastForceAttribute;
      static agxData::ScalarAttributeT< agx::Vec3 >* lastTorqueAttribute;
      static agxData::ScalarAttributeT< agx::AffineMatrix4x4 >* modelTransformAttribute;
      static agxData::ScalarAttributeT< agx::AffineMatrix4x4 >* localModelTransformAttribute;
      static agxData::ScalarAttributeT< agx::AffineMatrix4x4 >* cmTransformAttribute;
      static agxData::ScalarAttributeT< agx::AffineMatrix4x4 >* localCmTransformAttribute;
      static agxData::ScalarAttributeT< agx::Vec3 >* velocityAttribute;
      static agxData::ScalarAttributeT< agx::Vec3 >* angularVelocityAttribute;
      static agxData::ScalarAttributeT< agx::Vec3 >* linearAccelerationAttribute;
      static agxData::ScalarAttributeT< agx::Vec3 >* angularAccelerationAttribute;
      static agxData::ScalarAttributeT< agx::Vec3f >* linearVelocityDampingAttribute;
      static agxData::ScalarAttributeT< agx::Vec3f >* angularVelocityDampingAttribute;
      static agxData::ScalarAttributeT< agx::Vec3f >* linearVelocityZeroDampingAttribute;
      static agxData::ScalarAttributeT< agx::Vec3f >* angularVelocityZeroDampingAttribute;
      static agxData::ScalarAttributeT< agx::Physics::GeometryPtr >* geometryAttribute;
      static agxData::ScalarAttributeT< agx::Physics::GraphNodePtr >* graphNodeAttribute;
      static agxData::ScalarAttributeT< agx::Range6 >* velocityRangesAttribute;
      static agxData::PointerAttributeT< agx::RigidBody*>* modelAttribute;
      static agxData::ScalarAttributeT< agx::UInt32 >* solveIndexAttribute;
      static agxData::ScalarAttributeT< agx::UInt16 >* numConstraintsAttribute;

    protected:
      virtual ~RigidBodyModel();
      virtual agxData::EntityData* createData(agxData::EntityStorage* storage) override;
      virtual void configure(agx::TiXmlElement* eEntity) override;
      virtual void initAttributeAccessors() override;
      virtual void construct(agxData::EntityPtr instance) override;
      void construct(agx::Physics::RigidBodyPtr rigidBody);
    };


    DOXYGEN_START_INTERNAL_BLOCK()
    #ifndef AGX_PHYSICS_RIGIDBODY_DATA_SET_OVERRIDE
    #define AGX_PHYSICS_RIGIDBODY_DATA_SET
    class AGXPHYSICS_EXPORT RigidBodyData : public agx::ReferencedData
    {
    public:
      RigidBodyInstance operator[] (size_t index);

    public:
      agxData::Array< RigidBodyPtr >& instance;
      agxData::Array< agx::UInt32 > id;
      agxData::Array< agx::RigidBodyState > state;
      agxData::Array< agx::Name > name;
      agxData::Array< agx::Real > mass;
      agxData::Array< agx::Real > invMass;
      agxData::Array< agx::Vec3 > effectiveMass;
      agxData::Array< agx::Vec3 > effectiveMassCoefficients;
      agxData::Array< agx::Vec3 > effectiveMassTransformDiagonal;
      agxData::Array< agx::SPDMatrix3x3 > inertia;
      agxData::Array< agx::SPDMatrix3x3 > effectiveInertia;
      agxData::Array< agx::Vec3 > effectiveInertiaCoefficients;
      agxData::Array< agx::Matrix3x3 > worldMassMatrix;
      agxData::Array< agx::Matrix3x3 > invWorldMassMatrix;
      agxData::Array< agx::Matrix3x3 > effectiveWorldMassMatrix;
      agxData::Array< agx::Matrix3x3 > effectiveInvWorldMassMatrix;
      agxData::Array< agx::Matrix3x3 > worldInertia;
      agxData::Array< agx::Matrix3x3 > invWorldInertia;
      agxData::Array< agx::Matrix3x3 > effectiveWorldInertia;
      agxData::Array< agx::Matrix3x3 > effectiveInvWorldInertia;
      agxData::Array< agx::Vec3 > force;
      agxData::Array< agx::Vec3 > torque;
      agxData::Array< agx::Vec3 > lastForce;
      agxData::Array< agx::Vec3 > lastTorque;
      agxData::Array< agx::AffineMatrix4x4 > modelTransform;
      agxData::Array< agx::AffineMatrix4x4 > localModelTransform;
      agxData::Array< agx::AffineMatrix4x4 > cmTransform;
      agxData::Array< agx::AffineMatrix4x4 > localCmTransform;
      agxData::Array< agx::Vec3 > velocity;
      agxData::Array< agx::Vec3 > angularVelocity;
      agxData::Array< agx::Vec3 > linearAcceleration;
      agxData::Array< agx::Vec3 > angularAcceleration;
      agxData::Array< agx::Vec3f > linearVelocityDamping;
      agxData::Array< agx::Vec3f > angularVelocityDamping;
      agxData::Array< agx::Vec3f > linearVelocityZeroDamping;
      agxData::Array< agx::Vec3f > angularVelocityZeroDamping;
      agxData::Array< agx::Physics::GeometryPtr > geometry;
      agxData::Array< agx::Physics::GraphNodePtr > graphNode;
      agxData::Array< agx::Range6 > velocityRanges;
      agxData::Array< agx::RigidBody* > model;
      agxData::Array< agx::UInt32 > solveIndex;
      agxData::Array< agx::UInt16 > numConstraints;

    public:
      typedef agx::UInt32 idType;
      typedef agx::RigidBodyState stateType;
      typedef agx::Name nameType;
      typedef agx::Real massType;
      typedef agx::Real invMassType;
      typedef agx::Vec3 effectiveMassType;
      typedef agx::Vec3 effectiveMassCoefficientsType;
      typedef agx::Vec3 effectiveMassTransformDiagonalType;
      typedef agx::SPDMatrix3x3 inertiaType;
      typedef agx::SPDMatrix3x3 effectiveInertiaType;
      typedef agx::Vec3 effectiveInertiaCoefficientsType;
      typedef agx::Matrix3x3 worldMassMatrixType;
      typedef agx::Matrix3x3 invWorldMassMatrixType;
      typedef agx::Matrix3x3 effectiveWorldMassMatrixType;
      typedef agx::Matrix3x3 effectiveInvWorldMassMatrixType;
      typedef agx::Matrix3x3 worldInertiaType;
      typedef agx::Matrix3x3 invWorldInertiaType;
      typedef agx::Matrix3x3 effectiveWorldInertiaType;
      typedef agx::Matrix3x3 effectiveInvWorldInertiaType;
      typedef agx::Vec3 forceType;
      typedef agx::Vec3 torqueType;
      typedef agx::Vec3 lastForceType;
      typedef agx::Vec3 lastTorqueType;
      typedef agx::AffineMatrix4x4 modelTransformType;
      typedef agx::AffineMatrix4x4 localModelTransformType;
      typedef agx::AffineMatrix4x4 cmTransformType;
      typedef agx::AffineMatrix4x4 localCmTransformType;
      typedef agx::Vec3 velocityType;
      typedef agx::Vec3 angularVelocityType;
      typedef agx::Vec3 linearAccelerationType;
      typedef agx::Vec3 angularAccelerationType;
      typedef agx::Vec3f linearVelocityDampingType;
      typedef agx::Vec3f angularVelocityDampingType;
      typedef agx::Vec3f linearVelocityZeroDampingType;
      typedef agx::Vec3f angularVelocityZeroDampingType;
      typedef agx::Physics::GeometryPtr geometryType;
      typedef agx::Physics::GraphNodePtr graphNodeType;
      typedef agx::Range6 velocityRangesType;
      typedef agx::RigidBody* modelType;
      typedef agx::UInt32 solveIndexType;
      typedef agx::UInt16 numConstraintsType;

    public:
      RigidBodyData(agxData::EntityStorage* storage);
      RigidBodyData();

    protected:
      virtual ~RigidBodyData() {}
      virtual void setNumElements(agx::Index numElements) override;

    private:
      RigidBodyData& operator= (const RigidBodyData&) { return *this; }

    };
    #endif
    DOXYGEN_END_INTERNAL_BLOCK()


    DOXYGEN_START_INTERNAL_BLOCK()
    class AGXPHYSICS_EXPORT RigidBodySemantics : public agx::ReferencedSemantics
    {
    public:

      // Automatic getters
      agx::UInt32 const& getId() const;
      agx::RigidBodyState const& getState() const;
      agx::Name const& getName() const;
      agx::Real const& getMass() const;
      agx::Real const& getInvMass() const;
      agx::Vec3 const& getEffectiveMass() const;
      agx::Vec3 const& getEffectiveMassCoefficients() const;
      agx::Vec3 const& getEffectiveMassTransformDiagonal() const;
      agx::SPDMatrix3x3 const& getInertia() const;
      agx::SPDMatrix3x3 const& getEffectiveInertia() const;
      agx::Vec3 const& getEffectiveInertiaCoefficients() const;
      agx::Matrix3x3 const& getWorldMassMatrix() const;
      agx::Matrix3x3 const& getInvWorldMassMatrix() const;
      agx::Matrix3x3 const& getEffectiveWorldMassMatrix() const;
      agx::Matrix3x3 const& getEffectiveInvWorldMassMatrix() const;
      agx::Matrix3x3 const& getWorldInertia() const;
      agx::Matrix3x3 const& getInvWorldInertia() const;
      agx::Matrix3x3 const& getEffectiveWorldInertia() const;
      agx::Matrix3x3 const& getEffectiveInvWorldInertia() const;
      agx::Vec3 const& getForce() const;
      agx::Vec3 const& getTorque() const;
      agx::Vec3 const& getLastForce() const;
      agx::Vec3 const& getLastTorque() const;
      agx::AffineMatrix4x4 const& getModelTransform() const;
      agx::AffineMatrix4x4 const& getLocalModelTransform() const;
      agx::AffineMatrix4x4 const& getCmTransform() const;
      agx::AffineMatrix4x4 const& getLocalCmTransform() const;
      agx::Vec3 const& getVelocity() const;
      agx::Vec3 const& getAngularVelocity() const;
      agx::Vec3 const& getLinearAcceleration() const;
      agx::Vec3 const& getAngularAcceleration() const;
      agx::Vec3f const& getLinearVelocityDamping() const;
      agx::Vec3f const& getAngularVelocityDamping() const;
      agx::Vec3f const& getLinearVelocityZeroDamping() const;
      agx::Vec3f const& getAngularVelocityZeroDamping() const;
      agx::Physics::GeometryPtr const& getGeometry() const;
      agx::Physics::GraphNodePtr const& getGraphNode() const;
      agx::Range6 const& getVelocityRanges() const;
      agx::RigidBody* const& getModel() const;
      agx::UInt32 const& getSolveIndex() const;
      agx::UInt16 const& getNumConstraints() const;

      // Semantics defined by explicit kernels

      // Automatic setters
      void setId(agx::UInt32 const& value);
      void setState(agx::RigidBodyState const& value);
      void setName(agx::Name const& value);
      void setMass(agx::Real const& value);
      void setInvMass(agx::Real const& value);
      void setEffectiveMass(agx::Vec3 const& value);
      void setEffectiveMassCoefficients(agx::Vec3 const& value);
      void setEffectiveMassTransformDiagonal(agx::Vec3 const& value);
      void setInertia(agx::SPDMatrix3x3 const& value);
      void setEffectiveInertia(agx::SPDMatrix3x3 const& value);
      void setEffectiveInertiaCoefficients(agx::Vec3 const& value);
      void setWorldMassMatrix(agx::Matrix3x3 const& value);
      void setInvWorldMassMatrix(agx::Matrix3x3 const& value);
      void setEffectiveWorldMassMatrix(agx::Matrix3x3 const& value);
      void setEffectiveInvWorldMassMatrix(agx::Matrix3x3 const& value);
      void setWorldInertia(agx::Matrix3x3 const& value);
      void setInvWorldInertia(agx::Matrix3x3 const& value);
      void setEffectiveWorldInertia(agx::Matrix3x3 const& value);
      void setEffectiveInvWorldInertia(agx::Matrix3x3 const& value);
      void setForce(agx::Vec3 const& value);
      void setTorque(agx::Vec3 const& value);
      void setLastForce(agx::Vec3 const& value);
      void setLastTorque(agx::Vec3 const& value);
      void setModelTransform(agx::AffineMatrix4x4 const& value);
      void setLocalModelTransform(agx::AffineMatrix4x4 const& value);
      void setCmTransform(agx::AffineMatrix4x4 const& value);
      void setLocalCmTransform(agx::AffineMatrix4x4 const& value);
      void setVelocity(agx::Vec3 const& value);
      void setAngularVelocity(agx::Vec3 const& value);
      void setLinearAcceleration(agx::Vec3 const& value);
      void setAngularAcceleration(agx::Vec3 const& value);
      void setLinearVelocityDamping(agx::Vec3f const& value);
      void setAngularVelocityDamping(agx::Vec3f const& value);
      void setLinearVelocityZeroDamping(agx::Vec3f const& value);
      void setAngularVelocityZeroDamping(agx::Vec3f const& value);
      void setGeometry(agx::Physics::GeometryPtr const& value);
      void setGraphNode(agx::Physics::GraphNodePtr const& value);
      void setVelocityRanges(agx::Range6 const& value);
      void setModel(agx::RigidBody* const& value);
      void setSolveIndex(agx::UInt32 const& value);
      void setNumConstraints(agx::UInt16 const& value);


    protected:
      friend class RigidBodyPtr;
      friend class RigidBodyInstance;
      RigidBodySemantics();
    };
    DOXYGEN_END_INTERNAL_BLOCK()


    /**
    Pointer to a entity instance of type Physics.RigidBody
    */
    class CALLABLE RigidBodyPtr : public agx::ReferencedPtr
    {
    public:
      typedef RigidBodyModel ModelType;
      typedef RigidBodyData DataType;
      typedef RigidBodyInstance InstanceType;

    public:
      AGXPHYSICS_EXPORT RigidBodyPtr();
      AGXPHYSICS_EXPORT RigidBodyPtr(agxData::EntityStorage* storage, agx::Index id);
      AGXPHYSICS_EXPORT RigidBodyPtr(const agxData::EntityPtr& ptr);
      AGXPHYSICS_EXPORT RigidBodyPtr(const agxData::EntityInstance& instance);
      AGXPHYSICS_EXPORT RigidBodyPtr& operator= (const agxData::EntityPtr& ptr);
      AGXPHYSICS_EXPORT RigidBodyPtr& operator= (const agxData::EntityInstance& instance);
      AGXPHYSICS_EXPORT RigidBodyInstance instance();
      AGXPHYSICS_EXPORT const RigidBodyInstance instance() const;

      AGXPHYSICS_EXPORT RigidBodySemantics* operator->();
      AGXPHYSICS_EXPORT const RigidBodySemantics* operator->() const;

      RigidBodyData* getData();
      const RigidBodyData* getData() const;


      /// \return reference to the id attribute
      AGXPHYSICS_EXPORT agx::UInt32& id();
      /// \return const reference to the id attribute
      AGXPHYSICS_EXPORT agx::UInt32 const& id() const;

      /// \return reference to the state attribute
      AGXPHYSICS_EXPORT agx::RigidBodyState& state();
      /// \return const reference to the state attribute
      AGXPHYSICS_EXPORT agx::RigidBodyState const& state() const;

      /// \return reference to the name attribute
      AGXPHYSICS_EXPORT agx::Name& name();
      /// \return const reference to the name attribute
      AGXPHYSICS_EXPORT agx::Name const& name() const;

      /// \return reference to the mass attribute
      AGXPHYSICS_EXPORT agx::Real& mass();
      /// \return const reference to the mass attribute
      AGXPHYSICS_EXPORT agx::Real const& mass() const;

      /// \return reference to the invMass attribute
      AGXPHYSICS_EXPORT agx::Real& invMass();
      /// \return const reference to the invMass attribute
      AGXPHYSICS_EXPORT agx::Real const& invMass() const;

      /// \return reference to the effectiveMass attribute
      AGXPHYSICS_EXPORT agx::Vec3& effectiveMass();
      /// \return const reference to the effectiveMass attribute
      AGXPHYSICS_EXPORT agx::Vec3 const& effectiveMass() const;

      /// \return reference to the effectiveMassCoefficients attribute
      AGXPHYSICS_EXPORT agx::Vec3& effectiveMassCoefficients();
      /// \return const reference to the effectiveMassCoefficients attribute
      AGXPHYSICS_EXPORT agx::Vec3 const& effectiveMassCoefficients() const;

      /// \return reference to the effectiveMassTransformDiagonal attribute
      AGXPHYSICS_EXPORT agx::Vec3& effectiveMassTransformDiagonal();
      /// \return const reference to the effectiveMassTransformDiagonal attribute
      AGXPHYSICS_EXPORT agx::Vec3 const& effectiveMassTransformDiagonal() const;

      /// \return reference to the inertia attribute
      AGXPHYSICS_EXPORT agx::SPDMatrix3x3& inertia();
      /// \return const reference to the inertia attribute
      AGXPHYSICS_EXPORT agx::SPDMatrix3x3 const& inertia() const;

      /// \return reference to the effectiveInertia attribute
      AGXPHYSICS_EXPORT agx::SPDMatrix3x3& effectiveInertia();
      /// \return const reference to the effectiveInertia attribute
      AGXPHYSICS_EXPORT agx::SPDMatrix3x3 const& effectiveInertia() const;

      /// \return reference to the effectiveInertiaCoefficients attribute
      AGXPHYSICS_EXPORT agx::Vec3& effectiveInertiaCoefficients();
      /// \return const reference to the effectiveInertiaCoefficients attribute
      AGXPHYSICS_EXPORT agx::Vec3 const& effectiveInertiaCoefficients() const;

      /// \return reference to the worldMassMatrix attribute
      AGXPHYSICS_EXPORT agx::Matrix3x3& worldMassMatrix();
      /// \return const reference to the worldMassMatrix attribute
      AGXPHYSICS_EXPORT agx::Matrix3x3 const& worldMassMatrix() const;

      /// \return reference to the invWorldMassMatrix attribute
      AGXPHYSICS_EXPORT agx::Matrix3x3& invWorldMassMatrix();
      /// \return const reference to the invWorldMassMatrix attribute
      AGXPHYSICS_EXPORT agx::Matrix3x3 const& invWorldMassMatrix() const;

      /// \return reference to the effectiveWorldMassMatrix attribute
      AGXPHYSICS_EXPORT agx::Matrix3x3& effectiveWorldMassMatrix();
      /// \return const reference to the effectiveWorldMassMatrix attribute
      AGXPHYSICS_EXPORT agx::Matrix3x3 const& effectiveWorldMassMatrix() const;

      /// \return reference to the effectiveInvWorldMassMatrix attribute
      AGXPHYSICS_EXPORT agx::Matrix3x3& effectiveInvWorldMassMatrix();
      /// \return const reference to the effectiveInvWorldMassMatrix attribute
      AGXPHYSICS_EXPORT agx::Matrix3x3 const& effectiveInvWorldMassMatrix() const;

      /// \return reference to the worldInertia attribute
      AGXPHYSICS_EXPORT agx::Matrix3x3& worldInertia();
      /// \return const reference to the worldInertia attribute
      AGXPHYSICS_EXPORT agx::Matrix3x3 const& worldInertia() const;

      /// \return reference to the invWorldInertia attribute
      AGXPHYSICS_EXPORT agx::Matrix3x3& invWorldInertia();
      /// \return const reference to the invWorldInertia attribute
      AGXPHYSICS_EXPORT agx::Matrix3x3 const& invWorldInertia() const;

      /// \return reference to the effectiveWorldInertia attribute
      AGXPHYSICS_EXPORT agx::Matrix3x3& effectiveWorldInertia();
      /// \return const reference to the effectiveWorldInertia attribute
      AGXPHYSICS_EXPORT agx::Matrix3x3 const& effectiveWorldInertia() const;

      /// \return reference to the effectiveInvWorldInertia attribute
      AGXPHYSICS_EXPORT agx::Matrix3x3& effectiveInvWorldInertia();
      /// \return const reference to the effectiveInvWorldInertia attribute
      AGXPHYSICS_EXPORT agx::Matrix3x3 const& effectiveInvWorldInertia() const;

      /// \return reference to the force attribute
      AGXPHYSICS_EXPORT agx::Vec3& force();
      /// \return const reference to the force attribute
      AGXPHYSICS_EXPORT agx::Vec3 const& force() const;

      /// \return reference to the torque attribute
      AGXPHYSICS_EXPORT agx::Vec3& torque();
      /// \return const reference to the torque attribute
      AGXPHYSICS_EXPORT agx::Vec3 const& torque() const;

      /// \return reference to the lastForce attribute
      AGXPHYSICS_EXPORT agx::Vec3& lastForce();
      /// \return const reference to the lastForce attribute
      AGXPHYSICS_EXPORT agx::Vec3 const& lastForce() const;

      /// \return reference to the lastTorque attribute
      AGXPHYSICS_EXPORT agx::Vec3& lastTorque();
      /// \return const reference to the lastTorque attribute
      AGXPHYSICS_EXPORT agx::Vec3 const& lastTorque() const;

      /// \return reference to the modelTransform attribute
      AGXPHYSICS_EXPORT agx::AffineMatrix4x4& modelTransform();
      /// \return const reference to the modelTransform attribute
      AGXPHYSICS_EXPORT agx::AffineMatrix4x4 const& modelTransform() const;

      /// \return reference to the localModelTransform attribute
      AGXPHYSICS_EXPORT agx::AffineMatrix4x4& localModelTransform();
      /// \return const reference to the localModelTransform attribute
      AGXPHYSICS_EXPORT agx::AffineMatrix4x4 const& localModelTransform() const;

      /// \return reference to the cmTransform attribute
      AGXPHYSICS_EXPORT agx::AffineMatrix4x4& cmTransform();
      /// \return const reference to the cmTransform attribute
      AGXPHYSICS_EXPORT agx::AffineMatrix4x4 const& cmTransform() const;

      /// \return reference to the localCmTransform attribute
      AGXPHYSICS_EXPORT agx::AffineMatrix4x4& localCmTransform();
      /// \return const reference to the localCmTransform attribute
      AGXPHYSICS_EXPORT agx::AffineMatrix4x4 const& localCmTransform() const;

      /// \return reference to the velocity attribute
      AGXPHYSICS_EXPORT agx::Vec3& velocity();
      /// \return const reference to the velocity attribute
      AGXPHYSICS_EXPORT agx::Vec3 const& velocity() const;

      /// \return reference to the angularVelocity attribute
      AGXPHYSICS_EXPORT agx::Vec3& angularVelocity();
      /// \return const reference to the angularVelocity attribute
      AGXPHYSICS_EXPORT agx::Vec3 const& angularVelocity() const;

      /// \return reference to the linearAcceleration attribute
      AGXPHYSICS_EXPORT agx::Vec3& linearAcceleration();
      /// \return const reference to the linearAcceleration attribute
      AGXPHYSICS_EXPORT agx::Vec3 const& linearAcceleration() const;

      /// \return reference to the angularAcceleration attribute
      AGXPHYSICS_EXPORT agx::Vec3& angularAcceleration();
      /// \return const reference to the angularAcceleration attribute
      AGXPHYSICS_EXPORT agx::Vec3 const& angularAcceleration() const;

      /// \return reference to the linearVelocityDamping attribute
      AGXPHYSICS_EXPORT agx::Vec3f& linearVelocityDamping();
      /// \return const reference to the linearVelocityDamping attribute
      AGXPHYSICS_EXPORT agx::Vec3f const& linearVelocityDamping() const;

      /// \return reference to the angularVelocityDamping attribute
      AGXPHYSICS_EXPORT agx::Vec3f& angularVelocityDamping();
      /// \return const reference to the angularVelocityDamping attribute
      AGXPHYSICS_EXPORT agx::Vec3f const& angularVelocityDamping() const;

      /// \return reference to the linearVelocityZeroDamping attribute
      AGXPHYSICS_EXPORT agx::Vec3f& linearVelocityZeroDamping();
      /// \return const reference to the linearVelocityZeroDamping attribute
      AGXPHYSICS_EXPORT agx::Vec3f const& linearVelocityZeroDamping() const;

      /// \return reference to the angularVelocityZeroDamping attribute
      AGXPHYSICS_EXPORT agx::Vec3f& angularVelocityZeroDamping();
      /// \return const reference to the angularVelocityZeroDamping attribute
      AGXPHYSICS_EXPORT agx::Vec3f const& angularVelocityZeroDamping() const;

      /// \return reference to the geometry attribute
      AGXPHYSICS_EXPORT agx::Physics::GeometryPtr& geometry();
      /// \return const reference to the geometry attribute
      AGXPHYSICS_EXPORT agx::Physics::GeometryPtr const& geometry() const;

      /// \return reference to the graphNode attribute
      AGXPHYSICS_EXPORT agx::Physics::GraphNodePtr& graphNode();
      /// \return const reference to the graphNode attribute
      AGXPHYSICS_EXPORT agx::Physics::GraphNodePtr const& graphNode() const;

      /// \return reference to the velocityRanges attribute
      AGXPHYSICS_EXPORT agx::Range6& velocityRanges();
      /// \return const reference to the velocityRanges attribute
      AGXPHYSICS_EXPORT agx::Range6 const& velocityRanges() const;

      /// \return reference to the model attribute
      AGXPHYSICS_EXPORT agx::RigidBody*& model();
      /// \return const reference to the model attribute
      AGXPHYSICS_EXPORT agx::RigidBody* const& model() const;

      /// \return reference to the solveIndex attribute
      AGXPHYSICS_EXPORT agx::UInt32& solveIndex();
      /// \return const reference to the solveIndex attribute
      AGXPHYSICS_EXPORT agx::UInt32 const& solveIndex() const;

      /// \return reference to the numConstraints attribute
      AGXPHYSICS_EXPORT agx::UInt16& numConstraints();
      /// \return const reference to the numConstraints attribute
      AGXPHYSICS_EXPORT agx::UInt16 const& numConstraints() const;

      DOXYGEN_START_INTERNAL_BLOCK()
      // Only used by tolua!
      #ifdef AGX_GENERATED_TOLUA_WRAPPER
      inline agx::UInt32 const& getId() const { return (*this)->getId(); }
      inline agx::RigidBodyState const& getState() const { return (*this)->getState(); }
      inline agx::Name const& getName() const { return (*this)->getName(); }
      inline agx::Real const& getMass() const { return (*this)->getMass(); }
      inline agx::Real const& getInvMass() const { return (*this)->getInvMass(); }
      inline agx::Vec3 const& getEffectiveMass() const { return (*this)->getEffectiveMass(); }
      inline agx::Vec3 const& getEffectiveMassCoefficients() const { return (*this)->getEffectiveMassCoefficients(); }
      inline agx::Vec3 const& getEffectiveMassTransformDiagonal() const { return (*this)->getEffectiveMassTransformDiagonal(); }
      inline agx::SPDMatrix3x3 const& getInertia() const { return (*this)->getInertia(); }
      inline agx::SPDMatrix3x3 const& getEffectiveInertia() const { return (*this)->getEffectiveInertia(); }
      inline agx::Vec3 const& getEffectiveInertiaCoefficients() const { return (*this)->getEffectiveInertiaCoefficients(); }
      inline agx::Matrix3x3 const& getWorldMassMatrix() const { return (*this)->getWorldMassMatrix(); }
      inline agx::Matrix3x3 const& getInvWorldMassMatrix() const { return (*this)->getInvWorldMassMatrix(); }
      inline agx::Matrix3x3 const& getEffectiveWorldMassMatrix() const { return (*this)->getEffectiveWorldMassMatrix(); }
      inline agx::Matrix3x3 const& getEffectiveInvWorldMassMatrix() const { return (*this)->getEffectiveInvWorldMassMatrix(); }
      inline agx::Matrix3x3 const& getWorldInertia() const { return (*this)->getWorldInertia(); }
      inline agx::Matrix3x3 const& getInvWorldInertia() const { return (*this)->getInvWorldInertia(); }
      inline agx::Matrix3x3 const& getEffectiveWorldInertia() const { return (*this)->getEffectiveWorldInertia(); }
      inline agx::Matrix3x3 const& getEffectiveInvWorldInertia() const { return (*this)->getEffectiveInvWorldInertia(); }
      inline agx::Vec3 const& getForce() const { return (*this)->getForce(); }
      inline agx::Vec3 const& getTorque() const { return (*this)->getTorque(); }
      inline agx::Vec3 const& getLastForce() const { return (*this)->getLastForce(); }
      inline agx::Vec3 const& getLastTorque() const { return (*this)->getLastTorque(); }
      inline agx::AffineMatrix4x4 const& getModelTransform() const { return (*this)->getModelTransform(); }
      inline agx::AffineMatrix4x4 const& getLocalModelTransform() const { return (*this)->getLocalModelTransform(); }
      inline agx::AffineMatrix4x4 const& getCmTransform() const { return (*this)->getCmTransform(); }
      inline agx::AffineMatrix4x4 const& getLocalCmTransform() const { return (*this)->getLocalCmTransform(); }
      inline agx::Vec3 const& getVelocity() const { return (*this)->getVelocity(); }
      inline agx::Vec3 const& getAngularVelocity() const { return (*this)->getAngularVelocity(); }
      inline agx::Vec3 const& getLinearAcceleration() const { return (*this)->getLinearAcceleration(); }
      inline agx::Vec3 const& getAngularAcceleration() const { return (*this)->getAngularAcceleration(); }
      inline agx::Vec3f const& getLinearVelocityDamping() const { return (*this)->getLinearVelocityDamping(); }
      inline agx::Vec3f const& getAngularVelocityDamping() const { return (*this)->getAngularVelocityDamping(); }
      inline agx::Vec3f const& getLinearVelocityZeroDamping() const { return (*this)->getLinearVelocityZeroDamping(); }
      inline agx::Vec3f const& getAngularVelocityZeroDamping() const { return (*this)->getAngularVelocityZeroDamping(); }
      inline agx::Physics::GeometryPtr const& getGeometry() const { return (*this)->getGeometry(); }
      inline agx::Physics::GraphNodePtr const& getGraphNode() const { return (*this)->getGraphNode(); }
      inline agx::Range6 const& getVelocityRanges() const { return (*this)->getVelocityRanges(); }
      inline agx::RigidBody* const& getModel() const { return (*this)->getModel(); }
      inline agx::UInt32 const& getSolveIndex() const { return (*this)->getSolveIndex(); }
      inline agx::UInt16 const& getNumConstraints() const { return (*this)->getNumConstraints(); }


      inline void setId(agx::UInt32 const& value) { (*this)->setId(value); }
      inline void setState(agx::RigidBodyState const& value) { (*this)->setState(value); }
      inline void setName(agx::Name const& value) { (*this)->setName(value); }
      inline void setMass(agx::Real const& value) { (*this)->setMass(value); }
      inline void setInvMass(agx::Real const& value) { (*this)->setInvMass(value); }
      inline void setEffectiveMass(agx::Vec3 const& value) { (*this)->setEffectiveMass(value); }
      inline void setEffectiveMassCoefficients(agx::Vec3 const& value) { (*this)->setEffectiveMassCoefficients(value); }
      inline void setEffectiveMassTransformDiagonal(agx::Vec3 const& value) { (*this)->setEffectiveMassTransformDiagonal(value); }
      inline void setInertia(agx::SPDMatrix3x3 const& value) { (*this)->setInertia(value); }
      inline void setEffectiveInertia(agx::SPDMatrix3x3 const& value) { (*this)->setEffectiveInertia(value); }
      inline void setEffectiveInertiaCoefficients(agx::Vec3 const& value) { (*this)->setEffectiveInertiaCoefficients(value); }
      inline void setWorldMassMatrix(agx::Matrix3x3 const& value) { (*this)->setWorldMassMatrix(value); }
      inline void setInvWorldMassMatrix(agx::Matrix3x3 const& value) { (*this)->setInvWorldMassMatrix(value); }
      inline void setEffectiveWorldMassMatrix(agx::Matrix3x3 const& value) { (*this)->setEffectiveWorldMassMatrix(value); }
      inline void setEffectiveInvWorldMassMatrix(agx::Matrix3x3 const& value) { (*this)->setEffectiveInvWorldMassMatrix(value); }
      inline void setWorldInertia(agx::Matrix3x3 const& value) { (*this)->setWorldInertia(value); }
      inline void setInvWorldInertia(agx::Matrix3x3 const& value) { (*this)->setInvWorldInertia(value); }
      inline void setEffectiveWorldInertia(agx::Matrix3x3 const& value) { (*this)->setEffectiveWorldInertia(value); }
      inline void setEffectiveInvWorldInertia(agx::Matrix3x3 const& value) { (*this)->setEffectiveInvWorldInertia(value); }
      inline void setForce(agx::Vec3 const& value) { (*this)->setForce(value); }
      inline void setTorque(agx::Vec3 const& value) { (*this)->setTorque(value); }
      inline void setLastForce(agx::Vec3 const& value) { (*this)->setLastForce(value); }
      inline void setLastTorque(agx::Vec3 const& value) { (*this)->setLastTorque(value); }
      inline void setModelTransform(agx::AffineMatrix4x4 const& value) { (*this)->setModelTransform(value); }
      inline void setLocalModelTransform(agx::AffineMatrix4x4 const& value) { (*this)->setLocalModelTransform(value); }
      inline void setCmTransform(agx::AffineMatrix4x4 const& value) { (*this)->setCmTransform(value); }
      inline void setLocalCmTransform(agx::AffineMatrix4x4 const& value) { (*this)->setLocalCmTransform(value); }
      inline void setVelocity(agx::Vec3 const& value) { (*this)->setVelocity(value); }
      inline void setAngularVelocity(agx::Vec3 const& value) { (*this)->setAngularVelocity(value); }
      inline void setLinearAcceleration(agx::Vec3 const& value) { (*this)->setLinearAcceleration(value); }
      inline void setAngularAcceleration(agx::Vec3 const& value) { (*this)->setAngularAcceleration(value); }
      inline void setLinearVelocityDamping(agx::Vec3f const& value) { (*this)->setLinearVelocityDamping(value); }
      inline void setAngularVelocityDamping(agx::Vec3f const& value) { (*this)->setAngularVelocityDamping(value); }
      inline void setLinearVelocityZeroDamping(agx::Vec3f const& value) { (*this)->setLinearVelocityZeroDamping(value); }
      inline void setAngularVelocityZeroDamping(agx::Vec3f const& value) { (*this)->setAngularVelocityZeroDamping(value); }
      inline void setGeometry(agx::Physics::GeometryPtr const& value) { (*this)->setGeometry(value); }
      inline void setGraphNode(agx::Physics::GraphNodePtr const& value) { (*this)->setGraphNode(value); }
      inline void setVelocityRanges(agx::Range6 const& value) { (*this)->setVelocityRanges(value); }
      inline void setModel(agx::RigidBody* const& value) { (*this)->setModel(value); }
      inline void setSolveIndex(agx::UInt32 const& value) { (*this)->setSolveIndex(value); }
      inline void setNumConstraints(agx::UInt16 const& value) { (*this)->setNumConstraints(value); }
      #endif
      DOXYGEN_END_INTERNAL_BLOCK()

    };

    // Entity is Referenced
    typedef agxData::EntityRef< RigidBodyPtr > RigidBodyRef;


    DOXYGEN_START_INTERNAL_BLOCK()
    class AGXPHYSICS_EXPORT RigidBodyInstance : public agx::ReferencedInstance
    {
    public:
      RigidBodyInstance();
      RigidBodyInstance(RigidBodyData* data, agx::Index index);
      RigidBodyInstance(agxData::EntityStorage *storage, agx::Index index);
      RigidBodyInstance(const agxData::EntityInstance& other);
      RigidBodyInstance(const agxData::EntityPtr& ptr);

      RigidBodyData* getData();
      const RigidBodyData* getData() const;

    public:
      /// \return reference to the id attribute
      agx::UInt32& id();
      /// \return const reference to the id attribute
      agx::UInt32 const& id() const;

      /// \return reference to the state attribute
      agx::RigidBodyState& state();
      /// \return const reference to the state attribute
      agx::RigidBodyState const& state() const;

      /// \return reference to the name attribute
      agx::Name& name();
      /// \return const reference to the name attribute
      agx::Name const& name() const;

      /// \return reference to the mass attribute
      agx::Real& mass();
      /// \return const reference to the mass attribute
      agx::Real const& mass() const;

      /// \return reference to the invMass attribute
      agx::Real& invMass();
      /// \return const reference to the invMass attribute
      agx::Real const& invMass() const;

      /// \return reference to the effectiveMass attribute
      agx::Vec3& effectiveMass();
      /// \return const reference to the effectiveMass attribute
      agx::Vec3 const& effectiveMass() const;

      /// \return reference to the effectiveMassCoefficients attribute
      agx::Vec3& effectiveMassCoefficients();
      /// \return const reference to the effectiveMassCoefficients attribute
      agx::Vec3 const& effectiveMassCoefficients() const;

      /// \return reference to the effectiveMassTransformDiagonal attribute
      agx::Vec3& effectiveMassTransformDiagonal();
      /// \return const reference to the effectiveMassTransformDiagonal attribute
      agx::Vec3 const& effectiveMassTransformDiagonal() const;

      /// \return reference to the inertia attribute
      agx::SPDMatrix3x3& inertia();
      /// \return const reference to the inertia attribute
      agx::SPDMatrix3x3 const& inertia() const;

      /// \return reference to the effectiveInertia attribute
      agx::SPDMatrix3x3& effectiveInertia();
      /// \return const reference to the effectiveInertia attribute
      agx::SPDMatrix3x3 const& effectiveInertia() const;

      /// \return reference to the effectiveInertiaCoefficients attribute
      agx::Vec3& effectiveInertiaCoefficients();
      /// \return const reference to the effectiveInertiaCoefficients attribute
      agx::Vec3 const& effectiveInertiaCoefficients() const;

      /// \return reference to the worldMassMatrix attribute
      agx::Matrix3x3& worldMassMatrix();
      /// \return const reference to the worldMassMatrix attribute
      agx::Matrix3x3 const& worldMassMatrix() const;

      /// \return reference to the invWorldMassMatrix attribute
      agx::Matrix3x3& invWorldMassMatrix();
      /// \return const reference to the invWorldMassMatrix attribute
      agx::Matrix3x3 const& invWorldMassMatrix() const;

      /// \return reference to the effectiveWorldMassMatrix attribute
      agx::Matrix3x3& effectiveWorldMassMatrix();
      /// \return const reference to the effectiveWorldMassMatrix attribute
      agx::Matrix3x3 const& effectiveWorldMassMatrix() const;

      /// \return reference to the effectiveInvWorldMassMatrix attribute
      agx::Matrix3x3& effectiveInvWorldMassMatrix();
      /// \return const reference to the effectiveInvWorldMassMatrix attribute
      agx::Matrix3x3 const& effectiveInvWorldMassMatrix() const;

      /// \return reference to the worldInertia attribute
      agx::Matrix3x3& worldInertia();
      /// \return const reference to the worldInertia attribute
      agx::Matrix3x3 const& worldInertia() const;

      /// \return reference to the invWorldInertia attribute
      agx::Matrix3x3& invWorldInertia();
      /// \return const reference to the invWorldInertia attribute
      agx::Matrix3x3 const& invWorldInertia() const;

      /// \return reference to the effectiveWorldInertia attribute
      agx::Matrix3x3& effectiveWorldInertia();
      /// \return const reference to the effectiveWorldInertia attribute
      agx::Matrix3x3 const& effectiveWorldInertia() const;

      /// \return reference to the effectiveInvWorldInertia attribute
      agx::Matrix3x3& effectiveInvWorldInertia();
      /// \return const reference to the effectiveInvWorldInertia attribute
      agx::Matrix3x3 const& effectiveInvWorldInertia() const;

      /// \return reference to the force attribute
      agx::Vec3& force();
      /// \return const reference to the force attribute
      agx::Vec3 const& force() const;

      /// \return reference to the torque attribute
      agx::Vec3& torque();
      /// \return const reference to the torque attribute
      agx::Vec3 const& torque() const;

      /// \return reference to the lastForce attribute
      agx::Vec3& lastForce();
      /// \return const reference to the lastForce attribute
      agx::Vec3 const& lastForce() const;

      /// \return reference to the lastTorque attribute
      agx::Vec3& lastTorque();
      /// \return const reference to the lastTorque attribute
      agx::Vec3 const& lastTorque() const;

      /// \return reference to the modelTransform attribute
      agx::AffineMatrix4x4& modelTransform();
      /// \return const reference to the modelTransform attribute
      agx::AffineMatrix4x4 const& modelTransform() const;

      /// \return reference to the localModelTransform attribute
      agx::AffineMatrix4x4& localModelTransform();
      /// \return const reference to the localModelTransform attribute
      agx::AffineMatrix4x4 const& localModelTransform() const;

      /// \return reference to the cmTransform attribute
      agx::AffineMatrix4x4& cmTransform();
      /// \return const reference to the cmTransform attribute
      agx::AffineMatrix4x4 const& cmTransform() const;

      /// \return reference to the localCmTransform attribute
      agx::AffineMatrix4x4& localCmTransform();
      /// \return const reference to the localCmTransform attribute
      agx::AffineMatrix4x4 const& localCmTransform() const;

      /// \return reference to the velocity attribute
      agx::Vec3& velocity();
      /// \return const reference to the velocity attribute
      agx::Vec3 const& velocity() const;

      /// \return reference to the angularVelocity attribute
      agx::Vec3& angularVelocity();
      /// \return const reference to the angularVelocity attribute
      agx::Vec3 const& angularVelocity() const;

      /// \return reference to the linearAcceleration attribute
      agx::Vec3& linearAcceleration();
      /// \return const reference to the linearAcceleration attribute
      agx::Vec3 const& linearAcceleration() const;

      /// \return reference to the angularAcceleration attribute
      agx::Vec3& angularAcceleration();
      /// \return const reference to the angularAcceleration attribute
      agx::Vec3 const& angularAcceleration() const;

      /// \return reference to the linearVelocityDamping attribute
      agx::Vec3f& linearVelocityDamping();
      /// \return const reference to the linearVelocityDamping attribute
      agx::Vec3f const& linearVelocityDamping() const;

      /// \return reference to the angularVelocityDamping attribute
      agx::Vec3f& angularVelocityDamping();
      /// \return const reference to the angularVelocityDamping attribute
      agx::Vec3f const& angularVelocityDamping() const;

      /// \return reference to the linearVelocityZeroDamping attribute
      agx::Vec3f& linearVelocityZeroDamping();
      /// \return const reference to the linearVelocityZeroDamping attribute
      agx::Vec3f const& linearVelocityZeroDamping() const;

      /// \return reference to the angularVelocityZeroDamping attribute
      agx::Vec3f& angularVelocityZeroDamping();
      /// \return const reference to the angularVelocityZeroDamping attribute
      agx::Vec3f const& angularVelocityZeroDamping() const;

      /// \return reference to the geometry attribute
      agx::Physics::GeometryPtr& geometry();
      /// \return const reference to the geometry attribute
      agx::Physics::GeometryPtr const& geometry() const;

      /// \return reference to the graphNode attribute
      agx::Physics::GraphNodePtr& graphNode();
      /// \return const reference to the graphNode attribute
      agx::Physics::GraphNodePtr const& graphNode() const;

      /// \return reference to the velocityRanges attribute
      agx::Range6& velocityRanges();
      /// \return const reference to the velocityRanges attribute
      agx::Range6 const& velocityRanges() const;

      /// \return reference to the model attribute
      agx::RigidBody*& model();
      /// \return const reference to the model attribute
      agx::RigidBody* const& model() const;

      /// \return reference to the solveIndex attribute
      agx::UInt32& solveIndex();
      /// \return const reference to the solveIndex attribute
      agx::UInt32 const& solveIndex() const;

      /// \return reference to the numConstraints attribute
      agx::UInt16& numConstraints();
      /// \return const reference to the numConstraints attribute
      agx::UInt16 const& numConstraints() const;

    };
    DOXYGEN_END_INTERNAL_BLOCK()



    typedef agx::VectorPOD<RigidBodyPtr> RigidBodyPtrVector;
    typedef agxData::Array<RigidBodyPtr> RigidBodyPtrArray;



    DOXYGEN_START_INTERNAL_BLOCK()
    /* Implementation */
    //-----------------------------------------------------------------------------------------------------
    //-----------------------------------------------------------------------------------------------------
    inline RigidBodyInstance agx::Physics::RigidBodyData::operator[] (size_t index) { return RigidBodyInstance(this, (agx::Index)index); }
    //-----------------------------------------------------------------------------------------------------
    AGX_FORCE_INLINE RigidBodyPtr::RigidBodyPtr() {}
    AGX_FORCE_INLINE RigidBodyPtr::RigidBodyPtr(agxData::EntityStorage* storage, agx::Index id) : agx::ReferencedPtr(storage, id) {}
    AGX_FORCE_INLINE RigidBodyPtr::RigidBodyPtr(const agxData::EntityPtr& ptr) : agx::ReferencedPtr(ptr)
    {
      agxAssertN(!ptr || ptr.isInstanceOf(RigidBodyModel::instance()),
        "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
        EntityPtr::getModel()->fullPath().c_str(), RigidBodyModel::instance()->fullPath().c_str());
    }

    AGX_FORCE_INLINE RigidBodyPtr::RigidBodyPtr(const agxData::EntityInstance& instance) : agx::ReferencedPtr(instance)
    {
      agxAssertN(!instance || instance.isInstanceOf(RigidBodyModel::instance()),
        "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
        EntityPtr::getModel()->fullPath().c_str(), RigidBodyModel::instance()->fullPath().c_str());
    }

    AGX_FORCE_INLINE RigidBodyPtr& RigidBodyPtr::operator= (const agxData::EntityPtr& ptr)
    {
      agxData::EntityPtr::operator= (ptr);
      agxAssertN(!ptr || ptr.isInstanceOf(RigidBodyModel::instance()),
        "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
        EntityPtr::getModel()->fullPath().c_str(), RigidBodyModel::instance()->fullPath().c_str());
      return *this;
    }

    AGX_FORCE_INLINE RigidBodyPtr& RigidBodyPtr::operator= (const agxData::EntityInstance& instance)
    {
      agxData::EntityPtr::operator= (instance);
      agxAssertN(!instance || instance.isInstanceOf(RigidBodyModel::instance()),
        "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
        EntityPtr::getModel()->fullPath().c_str(), RigidBodyModel::instance()->fullPath().c_str());
      return *this;
    }

    inline RigidBodyInstance RigidBodyPtr::instance() { return agxData::EntityPtr::instance(); }
    inline const RigidBodyInstance RigidBodyPtr::instance() const { return agxData::EntityPtr::instance(); }
    AGX_FORCE_INLINE RigidBodySemantics* RigidBodyPtr::operator->() { return (RigidBodySemantics* )this; }
    AGX_FORCE_INLINE const RigidBodySemantics* RigidBodyPtr::operator->() const { return (const RigidBodySemantics* )this; }
    AGX_FORCE_INLINE RigidBodyData* RigidBodyPtr::getData() { return static_cast<RigidBodyData* >(agxData::EntityPtr::getData()); }
    AGX_FORCE_INLINE const RigidBodyData* RigidBodyPtr::getData() const { return static_cast<const RigidBodyData* >(agxData::EntityPtr::getData()); }

    AGX_FORCE_INLINE agx::UInt32& RigidBodyPtr::id() { verifyIndex(); return getData()->id[calculateIndex()]; }
    AGX_FORCE_INLINE agx::UInt32 const& RigidBodyPtr::id() const { verifyIndex(); return getData()->id[calculateIndex()]; }

    AGX_FORCE_INLINE agx::RigidBodyState& RigidBodyPtr::state() { verifyIndex(); return getData()->state[calculateIndex()]; }
    AGX_FORCE_INLINE agx::RigidBodyState const& RigidBodyPtr::state() const { verifyIndex(); return getData()->state[calculateIndex()]; }

    AGX_FORCE_INLINE agx::Name& RigidBodyPtr::name() { verifyIndex(); return getData()->name[calculateIndex()]; }
    AGX_FORCE_INLINE agx::Name const& RigidBodyPtr::name() const { verifyIndex(); return getData()->name[calculateIndex()]; }

    AGX_FORCE_INLINE agx::Real& RigidBodyPtr::mass() { verifyIndex(); return getData()->mass[calculateIndex()]; }
    AGX_FORCE_INLINE agx::Real const& RigidBodyPtr::mass() const { verifyIndex(); return getData()->mass[calculateIndex()]; }

    AGX_FORCE_INLINE agx::Real& RigidBodyPtr::invMass() { verifyIndex(); return getData()->invMass[calculateIndex()]; }
    AGX_FORCE_INLINE agx::Real const& RigidBodyPtr::invMass() const { verifyIndex(); return getData()->invMass[calculateIndex()]; }

    AGX_FORCE_INLINE agx::Vec3& RigidBodyPtr::effectiveMass() { verifyIndex(); return getData()->effectiveMass[calculateIndex()]; }
    AGX_FORCE_INLINE agx::Vec3 const& RigidBodyPtr::effectiveMass() const { verifyIndex(); return getData()->effectiveMass[calculateIndex()]; }

    AGX_FORCE_INLINE agx::Vec3& RigidBodyPtr::effectiveMassCoefficients() { verifyIndex(); return getData()->effectiveMassCoefficients[calculateIndex()]; }
    AGX_FORCE_INLINE agx::Vec3 const& RigidBodyPtr::effectiveMassCoefficients() const { verifyIndex(); return getData()->effectiveMassCoefficients[calculateIndex()]; }

    AGX_FORCE_INLINE agx::Vec3& RigidBodyPtr::effectiveMassTransformDiagonal() { verifyIndex(); return getData()->effectiveMassTransformDiagonal[calculateIndex()]; }
    AGX_FORCE_INLINE agx::Vec3 const& RigidBodyPtr::effectiveMassTransformDiagonal() const { verifyIndex(); return getData()->effectiveMassTransformDiagonal[calculateIndex()]; }

    AGX_FORCE_INLINE agx::SPDMatrix3x3& RigidBodyPtr::inertia() { verifyIndex(); return getData()->inertia[calculateIndex()]; }
    AGX_FORCE_INLINE agx::SPDMatrix3x3 const& RigidBodyPtr::inertia() const { verifyIndex(); return getData()->inertia[calculateIndex()]; }

    AGX_FORCE_INLINE agx::SPDMatrix3x3& RigidBodyPtr::effectiveInertia() { verifyIndex(); return getData()->effectiveInertia[calculateIndex()]; }
    AGX_FORCE_INLINE agx::SPDMatrix3x3 const& RigidBodyPtr::effectiveInertia() const { verifyIndex(); return getData()->effectiveInertia[calculateIndex()]; }

    AGX_FORCE_INLINE agx::Vec3& RigidBodyPtr::effectiveInertiaCoefficients() { verifyIndex(); return getData()->effectiveInertiaCoefficients[calculateIndex()]; }
    AGX_FORCE_INLINE agx::Vec3 const& RigidBodyPtr::effectiveInertiaCoefficients() const { verifyIndex(); return getData()->effectiveInertiaCoefficients[calculateIndex()]; }

    AGX_FORCE_INLINE agx::Matrix3x3& RigidBodyPtr::worldMassMatrix() { verifyIndex(); return getData()->worldMassMatrix[calculateIndex()]; }
    AGX_FORCE_INLINE agx::Matrix3x3 const& RigidBodyPtr::worldMassMatrix() const { verifyIndex(); return getData()->worldMassMatrix[calculateIndex()]; }

    AGX_FORCE_INLINE agx::Matrix3x3& RigidBodyPtr::invWorldMassMatrix() { verifyIndex(); return getData()->invWorldMassMatrix[calculateIndex()]; }
    AGX_FORCE_INLINE agx::Matrix3x3 const& RigidBodyPtr::invWorldMassMatrix() const { verifyIndex(); return getData()->invWorldMassMatrix[calculateIndex()]; }

    AGX_FORCE_INLINE agx::Matrix3x3& RigidBodyPtr::effectiveWorldMassMatrix() { verifyIndex(); return getData()->effectiveWorldMassMatrix[calculateIndex()]; }
    AGX_FORCE_INLINE agx::Matrix3x3 const& RigidBodyPtr::effectiveWorldMassMatrix() const { verifyIndex(); return getData()->effectiveWorldMassMatrix[calculateIndex()]; }

    AGX_FORCE_INLINE agx::Matrix3x3& RigidBodyPtr::effectiveInvWorldMassMatrix() { verifyIndex(); return getData()->effectiveInvWorldMassMatrix[calculateIndex()]; }
    AGX_FORCE_INLINE agx::Matrix3x3 const& RigidBodyPtr::effectiveInvWorldMassMatrix() const { verifyIndex(); return getData()->effectiveInvWorldMassMatrix[calculateIndex()]; }

    AGX_FORCE_INLINE agx::Matrix3x3& RigidBodyPtr::worldInertia() { verifyIndex(); return getData()->worldInertia[calculateIndex()]; }
    AGX_FORCE_INLINE agx::Matrix3x3 const& RigidBodyPtr::worldInertia() const { verifyIndex(); return getData()->worldInertia[calculateIndex()]; }

    AGX_FORCE_INLINE agx::Matrix3x3& RigidBodyPtr::invWorldInertia() { verifyIndex(); return getData()->invWorldInertia[calculateIndex()]; }
    AGX_FORCE_INLINE agx::Matrix3x3 const& RigidBodyPtr::invWorldInertia() const { verifyIndex(); return getData()->invWorldInertia[calculateIndex()]; }

    AGX_FORCE_INLINE agx::Matrix3x3& RigidBodyPtr::effectiveWorldInertia() { verifyIndex(); return getData()->effectiveWorldInertia[calculateIndex()]; }
    AGX_FORCE_INLINE agx::Matrix3x3 const& RigidBodyPtr::effectiveWorldInertia() const { verifyIndex(); return getData()->effectiveWorldInertia[calculateIndex()]; }

    AGX_FORCE_INLINE agx::Matrix3x3& RigidBodyPtr::effectiveInvWorldInertia() { verifyIndex(); return getData()->effectiveInvWorldInertia[calculateIndex()]; }
    AGX_FORCE_INLINE agx::Matrix3x3 const& RigidBodyPtr::effectiveInvWorldInertia() const { verifyIndex(); return getData()->effectiveInvWorldInertia[calculateIndex()]; }

    AGX_FORCE_INLINE agx::Vec3& RigidBodyPtr::force() { verifyIndex(); return getData()->force[calculateIndex()]; }
    AGX_FORCE_INLINE agx::Vec3 const& RigidBodyPtr::force() const { verifyIndex(); return getData()->force[calculateIndex()]; }

    AGX_FORCE_INLINE agx::Vec3& RigidBodyPtr::torque() { verifyIndex(); return getData()->torque[calculateIndex()]; }
    AGX_FORCE_INLINE agx::Vec3 const& RigidBodyPtr::torque() const { verifyIndex(); return getData()->torque[calculateIndex()]; }

    AGX_FORCE_INLINE agx::Vec3& RigidBodyPtr::lastForce() { verifyIndex(); return getData()->lastForce[calculateIndex()]; }
    AGX_FORCE_INLINE agx::Vec3 const& RigidBodyPtr::lastForce() const { verifyIndex(); return getData()->lastForce[calculateIndex()]; }

    AGX_FORCE_INLINE agx::Vec3& RigidBodyPtr::lastTorque() { verifyIndex(); return getData()->lastTorque[calculateIndex()]; }
    AGX_FORCE_INLINE agx::Vec3 const& RigidBodyPtr::lastTorque() const { verifyIndex(); return getData()->lastTorque[calculateIndex()]; }

    AGX_FORCE_INLINE agx::AffineMatrix4x4& RigidBodyPtr::modelTransform() { verifyIndex(); return getData()->modelTransform[calculateIndex()]; }
    AGX_FORCE_INLINE agx::AffineMatrix4x4 const& RigidBodyPtr::modelTransform() const { verifyIndex(); return getData()->modelTransform[calculateIndex()]; }

    AGX_FORCE_INLINE agx::AffineMatrix4x4& RigidBodyPtr::localModelTransform() { verifyIndex(); return getData()->localModelTransform[calculateIndex()]; }
    AGX_FORCE_INLINE agx::AffineMatrix4x4 const& RigidBodyPtr::localModelTransform() const { verifyIndex(); return getData()->localModelTransform[calculateIndex()]; }

    AGX_FORCE_INLINE agx::AffineMatrix4x4& RigidBodyPtr::cmTransform() { verifyIndex(); return getData()->cmTransform[calculateIndex()]; }
    AGX_FORCE_INLINE agx::AffineMatrix4x4 const& RigidBodyPtr::cmTransform() const { verifyIndex(); return getData()->cmTransform[calculateIndex()]; }

    AGX_FORCE_INLINE agx::AffineMatrix4x4& RigidBodyPtr::localCmTransform() { verifyIndex(); return getData()->localCmTransform[calculateIndex()]; }
    AGX_FORCE_INLINE agx::AffineMatrix4x4 const& RigidBodyPtr::localCmTransform() const { verifyIndex(); return getData()->localCmTransform[calculateIndex()]; }

    AGX_FORCE_INLINE agx::Vec3& RigidBodyPtr::velocity() { verifyIndex(); return getData()->velocity[calculateIndex()]; }
    AGX_FORCE_INLINE agx::Vec3 const& RigidBodyPtr::velocity() const { verifyIndex(); return getData()->velocity[calculateIndex()]; }

    AGX_FORCE_INLINE agx::Vec3& RigidBodyPtr::angularVelocity() { verifyIndex(); return getData()->angularVelocity[calculateIndex()]; }
    AGX_FORCE_INLINE agx::Vec3 const& RigidBodyPtr::angularVelocity() const { verifyIndex(); return getData()->angularVelocity[calculateIndex()]; }

    AGX_FORCE_INLINE agx::Vec3& RigidBodyPtr::linearAcceleration() { verifyIndex(); return getData()->linearAcceleration[calculateIndex()]; }
    AGX_FORCE_INLINE agx::Vec3 const& RigidBodyPtr::linearAcceleration() const { verifyIndex(); return getData()->linearAcceleration[calculateIndex()]; }

    AGX_FORCE_INLINE agx::Vec3& RigidBodyPtr::angularAcceleration() { verifyIndex(); return getData()->angularAcceleration[calculateIndex()]; }
    AGX_FORCE_INLINE agx::Vec3 const& RigidBodyPtr::angularAcceleration() const { verifyIndex(); return getData()->angularAcceleration[calculateIndex()]; }

    AGX_FORCE_INLINE agx::Vec3f& RigidBodyPtr::linearVelocityDamping() { verifyIndex(); return getData()->linearVelocityDamping[calculateIndex()]; }
    AGX_FORCE_INLINE agx::Vec3f const& RigidBodyPtr::linearVelocityDamping() const { verifyIndex(); return getData()->linearVelocityDamping[calculateIndex()]; }

    AGX_FORCE_INLINE agx::Vec3f& RigidBodyPtr::angularVelocityDamping() { verifyIndex(); return getData()->angularVelocityDamping[calculateIndex()]; }
    AGX_FORCE_INLINE agx::Vec3f const& RigidBodyPtr::angularVelocityDamping() const { verifyIndex(); return getData()->angularVelocityDamping[calculateIndex()]; }

    AGX_FORCE_INLINE agx::Vec3f& RigidBodyPtr::linearVelocityZeroDamping() { verifyIndex(); return getData()->linearVelocityZeroDamping[calculateIndex()]; }
    AGX_FORCE_INLINE agx::Vec3f const& RigidBodyPtr::linearVelocityZeroDamping() const { verifyIndex(); return getData()->linearVelocityZeroDamping[calculateIndex()]; }

    AGX_FORCE_INLINE agx::Vec3f& RigidBodyPtr::angularVelocityZeroDamping() { verifyIndex(); return getData()->angularVelocityZeroDamping[calculateIndex()]; }
    AGX_FORCE_INLINE agx::Vec3f const& RigidBodyPtr::angularVelocityZeroDamping() const { verifyIndex(); return getData()->angularVelocityZeroDamping[calculateIndex()]; }

    AGX_FORCE_INLINE agx::Physics::GeometryPtr& RigidBodyPtr::geometry() { verifyIndex(); return getData()->geometry[calculateIndex()]; }
    AGX_FORCE_INLINE agx::Physics::GeometryPtr const& RigidBodyPtr::geometry() const { verifyIndex(); return getData()->geometry[calculateIndex()]; }

    AGX_FORCE_INLINE agx::Physics::GraphNodePtr& RigidBodyPtr::graphNode() { verifyIndex(); return getData()->graphNode[calculateIndex()]; }
    AGX_FORCE_INLINE agx::Physics::GraphNodePtr const& RigidBodyPtr::graphNode() const { verifyIndex(); return getData()->graphNode[calculateIndex()]; }

    AGX_FORCE_INLINE agx::Range6& RigidBodyPtr::velocityRanges() { verifyIndex(); return getData()->velocityRanges[calculateIndex()]; }
    AGX_FORCE_INLINE agx::Range6 const& RigidBodyPtr::velocityRanges() const { verifyIndex(); return getData()->velocityRanges[calculateIndex()]; }

    AGX_FORCE_INLINE agx::RigidBody*& RigidBodyPtr::model() { verifyIndex(); return getData()->model[calculateIndex()]; }
    AGX_FORCE_INLINE agx::RigidBody* const& RigidBodyPtr::model() const { verifyIndex(); return getData()->model[calculateIndex()]; }

    AGX_FORCE_INLINE agx::UInt32& RigidBodyPtr::solveIndex() { verifyIndex(); return getData()->solveIndex[calculateIndex()]; }
    AGX_FORCE_INLINE agx::UInt32 const& RigidBodyPtr::solveIndex() const { verifyIndex(); return getData()->solveIndex[calculateIndex()]; }

    AGX_FORCE_INLINE agx::UInt16& RigidBodyPtr::numConstraints() { verifyIndex(); return getData()->numConstraints[calculateIndex()]; }
    AGX_FORCE_INLINE agx::UInt16 const& RigidBodyPtr::numConstraints() const { verifyIndex(); return getData()->numConstraints[calculateIndex()]; }

    //-----------------------------------------------------------------------------------------------------
    AGX_FORCE_INLINE RigidBodyInstance::RigidBodyInstance() {}
    AGX_FORCE_INLINE RigidBodyInstance::RigidBodyInstance(RigidBodyData* data, agx::Index index) : agx::ReferencedInstance(data, index) {}
    AGX_FORCE_INLINE RigidBodyInstance::RigidBodyInstance(agxData::EntityStorage* storage, agx::Index index) : agx::ReferencedInstance(storage, index) {}
    AGX_FORCE_INLINE RigidBodyInstance::RigidBodyInstance(const agxData::EntityInstance& other) : agx::ReferencedInstance(other)
    {
      agxAssertN(!other || other.isInstanceOf(RigidBodyModel::instance()),
        "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
        EntityInstance::getModel()->fullPath().c_str(), RigidBodyModel::instance()->fullPath().c_str());
    }

    AGX_FORCE_INLINE RigidBodyInstance::RigidBodyInstance(const agxData::EntityPtr& ptr) : agx::ReferencedInstance(ptr)
    {
      agxAssertN(!ptr || ptr.isInstanceOf(RigidBodyModel::instance()),
        "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
        EntityInstance::getModel()->fullPath().c_str(), RigidBodyModel::instance()->fullPath().c_str());
    }


    AGX_FORCE_INLINE RigidBodyData* RigidBodyInstance::getData() { return static_cast<RigidBodyData* >(agxData::EntityInstance::getData()); }
    AGX_FORCE_INLINE const RigidBodyData* RigidBodyInstance::getData() const { return static_cast<const RigidBodyData* >(agxData::EntityInstance::getData()); }

    AGX_FORCE_INLINE agx::UInt32& RigidBodyInstance::id() { verifyIndex(); return getData()->id[getIndex()]; }
    AGX_FORCE_INLINE agx::UInt32 const& RigidBodyInstance::id() const { verifyIndex(); return getData()->id[getIndex()]; }

    AGX_FORCE_INLINE agx::RigidBodyState& RigidBodyInstance::state() { verifyIndex(); return getData()->state[getIndex()]; }
    AGX_FORCE_INLINE agx::RigidBodyState const& RigidBodyInstance::state() const { verifyIndex(); return getData()->state[getIndex()]; }

    AGX_FORCE_INLINE agx::Name& RigidBodyInstance::name() { verifyIndex(); return getData()->name[getIndex()]; }
    AGX_FORCE_INLINE agx::Name const& RigidBodyInstance::name() const { verifyIndex(); return getData()->name[getIndex()]; }

    AGX_FORCE_INLINE agx::Real& RigidBodyInstance::mass() { verifyIndex(); return getData()->mass[getIndex()]; }
    AGX_FORCE_INLINE agx::Real const& RigidBodyInstance::mass() const { verifyIndex(); return getData()->mass[getIndex()]; }

    AGX_FORCE_INLINE agx::Real& RigidBodyInstance::invMass() { verifyIndex(); return getData()->invMass[getIndex()]; }
    AGX_FORCE_INLINE agx::Real const& RigidBodyInstance::invMass() const { verifyIndex(); return getData()->invMass[getIndex()]; }

    AGX_FORCE_INLINE agx::Vec3& RigidBodyInstance::effectiveMass() { verifyIndex(); return getData()->effectiveMass[getIndex()]; }
    AGX_FORCE_INLINE agx::Vec3 const& RigidBodyInstance::effectiveMass() const { verifyIndex(); return getData()->effectiveMass[getIndex()]; }

    AGX_FORCE_INLINE agx::Vec3& RigidBodyInstance::effectiveMassCoefficients() { verifyIndex(); return getData()->effectiveMassCoefficients[getIndex()]; }
    AGX_FORCE_INLINE agx::Vec3 const& RigidBodyInstance::effectiveMassCoefficients() const { verifyIndex(); return getData()->effectiveMassCoefficients[getIndex()]; }

    AGX_FORCE_INLINE agx::Vec3& RigidBodyInstance::effectiveMassTransformDiagonal() { verifyIndex(); return getData()->effectiveMassTransformDiagonal[getIndex()]; }
    AGX_FORCE_INLINE agx::Vec3 const& RigidBodyInstance::effectiveMassTransformDiagonal() const { verifyIndex(); return getData()->effectiveMassTransformDiagonal[getIndex()]; }

    AGX_FORCE_INLINE agx::SPDMatrix3x3& RigidBodyInstance::inertia() { verifyIndex(); return getData()->inertia[getIndex()]; }
    AGX_FORCE_INLINE agx::SPDMatrix3x3 const& RigidBodyInstance::inertia() const { verifyIndex(); return getData()->inertia[getIndex()]; }

    AGX_FORCE_INLINE agx::SPDMatrix3x3& RigidBodyInstance::effectiveInertia() { verifyIndex(); return getData()->effectiveInertia[getIndex()]; }
    AGX_FORCE_INLINE agx::SPDMatrix3x3 const& RigidBodyInstance::effectiveInertia() const { verifyIndex(); return getData()->effectiveInertia[getIndex()]; }

    AGX_FORCE_INLINE agx::Vec3& RigidBodyInstance::effectiveInertiaCoefficients() { verifyIndex(); return getData()->effectiveInertiaCoefficients[getIndex()]; }
    AGX_FORCE_INLINE agx::Vec3 const& RigidBodyInstance::effectiveInertiaCoefficients() const { verifyIndex(); return getData()->effectiveInertiaCoefficients[getIndex()]; }

    AGX_FORCE_INLINE agx::Matrix3x3& RigidBodyInstance::worldMassMatrix() { verifyIndex(); return getData()->worldMassMatrix[getIndex()]; }
    AGX_FORCE_INLINE agx::Matrix3x3 const& RigidBodyInstance::worldMassMatrix() const { verifyIndex(); return getData()->worldMassMatrix[getIndex()]; }

    AGX_FORCE_INLINE agx::Matrix3x3& RigidBodyInstance::invWorldMassMatrix() { verifyIndex(); return getData()->invWorldMassMatrix[getIndex()]; }
    AGX_FORCE_INLINE agx::Matrix3x3 const& RigidBodyInstance::invWorldMassMatrix() const { verifyIndex(); return getData()->invWorldMassMatrix[getIndex()]; }

    AGX_FORCE_INLINE agx::Matrix3x3& RigidBodyInstance::effectiveWorldMassMatrix() { verifyIndex(); return getData()->effectiveWorldMassMatrix[getIndex()]; }
    AGX_FORCE_INLINE agx::Matrix3x3 const& RigidBodyInstance::effectiveWorldMassMatrix() const { verifyIndex(); return getData()->effectiveWorldMassMatrix[getIndex()]; }

    AGX_FORCE_INLINE agx::Matrix3x3& RigidBodyInstance::effectiveInvWorldMassMatrix() { verifyIndex(); return getData()->effectiveInvWorldMassMatrix[getIndex()]; }
    AGX_FORCE_INLINE agx::Matrix3x3 const& RigidBodyInstance::effectiveInvWorldMassMatrix() const { verifyIndex(); return getData()->effectiveInvWorldMassMatrix[getIndex()]; }

    AGX_FORCE_INLINE agx::Matrix3x3& RigidBodyInstance::worldInertia() { verifyIndex(); return getData()->worldInertia[getIndex()]; }
    AGX_FORCE_INLINE agx::Matrix3x3 const& RigidBodyInstance::worldInertia() const { verifyIndex(); return getData()->worldInertia[getIndex()]; }

    AGX_FORCE_INLINE agx::Matrix3x3& RigidBodyInstance::invWorldInertia() { verifyIndex(); return getData()->invWorldInertia[getIndex()]; }
    AGX_FORCE_INLINE agx::Matrix3x3 const& RigidBodyInstance::invWorldInertia() const { verifyIndex(); return getData()->invWorldInertia[getIndex()]; }

    AGX_FORCE_INLINE agx::Matrix3x3& RigidBodyInstance::effectiveWorldInertia() { verifyIndex(); return getData()->effectiveWorldInertia[getIndex()]; }
    AGX_FORCE_INLINE agx::Matrix3x3 const& RigidBodyInstance::effectiveWorldInertia() const { verifyIndex(); return getData()->effectiveWorldInertia[getIndex()]; }

    AGX_FORCE_INLINE agx::Matrix3x3& RigidBodyInstance::effectiveInvWorldInertia() { verifyIndex(); return getData()->effectiveInvWorldInertia[getIndex()]; }
    AGX_FORCE_INLINE agx::Matrix3x3 const& RigidBodyInstance::effectiveInvWorldInertia() const { verifyIndex(); return getData()->effectiveInvWorldInertia[getIndex()]; }

    AGX_FORCE_INLINE agx::Vec3& RigidBodyInstance::force() { verifyIndex(); return getData()->force[getIndex()]; }
    AGX_FORCE_INLINE agx::Vec3 const& RigidBodyInstance::force() const { verifyIndex(); return getData()->force[getIndex()]; }

    AGX_FORCE_INLINE agx::Vec3& RigidBodyInstance::torque() { verifyIndex(); return getData()->torque[getIndex()]; }
    AGX_FORCE_INLINE agx::Vec3 const& RigidBodyInstance::torque() const { verifyIndex(); return getData()->torque[getIndex()]; }

    AGX_FORCE_INLINE agx::Vec3& RigidBodyInstance::lastForce() { verifyIndex(); return getData()->lastForce[getIndex()]; }
    AGX_FORCE_INLINE agx::Vec3 const& RigidBodyInstance::lastForce() const { verifyIndex(); return getData()->lastForce[getIndex()]; }

    AGX_FORCE_INLINE agx::Vec3& RigidBodyInstance::lastTorque() { verifyIndex(); return getData()->lastTorque[getIndex()]; }
    AGX_FORCE_INLINE agx::Vec3 const& RigidBodyInstance::lastTorque() const { verifyIndex(); return getData()->lastTorque[getIndex()]; }

    AGX_FORCE_INLINE agx::AffineMatrix4x4& RigidBodyInstance::modelTransform() { verifyIndex(); return getData()->modelTransform[getIndex()]; }
    AGX_FORCE_INLINE agx::AffineMatrix4x4 const& RigidBodyInstance::modelTransform() const { verifyIndex(); return getData()->modelTransform[getIndex()]; }

    AGX_FORCE_INLINE agx::AffineMatrix4x4& RigidBodyInstance::localModelTransform() { verifyIndex(); return getData()->localModelTransform[getIndex()]; }
    AGX_FORCE_INLINE agx::AffineMatrix4x4 const& RigidBodyInstance::localModelTransform() const { verifyIndex(); return getData()->localModelTransform[getIndex()]; }

    AGX_FORCE_INLINE agx::AffineMatrix4x4& RigidBodyInstance::cmTransform() { verifyIndex(); return getData()->cmTransform[getIndex()]; }
    AGX_FORCE_INLINE agx::AffineMatrix4x4 const& RigidBodyInstance::cmTransform() const { verifyIndex(); return getData()->cmTransform[getIndex()]; }

    AGX_FORCE_INLINE agx::AffineMatrix4x4& RigidBodyInstance::localCmTransform() { verifyIndex(); return getData()->localCmTransform[getIndex()]; }
    AGX_FORCE_INLINE agx::AffineMatrix4x4 const& RigidBodyInstance::localCmTransform() const { verifyIndex(); return getData()->localCmTransform[getIndex()]; }

    AGX_FORCE_INLINE agx::Vec3& RigidBodyInstance::velocity() { verifyIndex(); return getData()->velocity[getIndex()]; }
    AGX_FORCE_INLINE agx::Vec3 const& RigidBodyInstance::velocity() const { verifyIndex(); return getData()->velocity[getIndex()]; }

    AGX_FORCE_INLINE agx::Vec3& RigidBodyInstance::angularVelocity() { verifyIndex(); return getData()->angularVelocity[getIndex()]; }
    AGX_FORCE_INLINE agx::Vec3 const& RigidBodyInstance::angularVelocity() const { verifyIndex(); return getData()->angularVelocity[getIndex()]; }

    AGX_FORCE_INLINE agx::Vec3& RigidBodyInstance::linearAcceleration() { verifyIndex(); return getData()->linearAcceleration[getIndex()]; }
    AGX_FORCE_INLINE agx::Vec3 const& RigidBodyInstance::linearAcceleration() const { verifyIndex(); return getData()->linearAcceleration[getIndex()]; }

    AGX_FORCE_INLINE agx::Vec3& RigidBodyInstance::angularAcceleration() { verifyIndex(); return getData()->angularAcceleration[getIndex()]; }
    AGX_FORCE_INLINE agx::Vec3 const& RigidBodyInstance::angularAcceleration() const { verifyIndex(); return getData()->angularAcceleration[getIndex()]; }

    AGX_FORCE_INLINE agx::Vec3f& RigidBodyInstance::linearVelocityDamping() { verifyIndex(); return getData()->linearVelocityDamping[getIndex()]; }
    AGX_FORCE_INLINE agx::Vec3f const& RigidBodyInstance::linearVelocityDamping() const { verifyIndex(); return getData()->linearVelocityDamping[getIndex()]; }

    AGX_FORCE_INLINE agx::Vec3f& RigidBodyInstance::angularVelocityDamping() { verifyIndex(); return getData()->angularVelocityDamping[getIndex()]; }
    AGX_FORCE_INLINE agx::Vec3f const& RigidBodyInstance::angularVelocityDamping() const { verifyIndex(); return getData()->angularVelocityDamping[getIndex()]; }

    AGX_FORCE_INLINE agx::Vec3f& RigidBodyInstance::linearVelocityZeroDamping() { verifyIndex(); return getData()->linearVelocityZeroDamping[getIndex()]; }
    AGX_FORCE_INLINE agx::Vec3f const& RigidBodyInstance::linearVelocityZeroDamping() const { verifyIndex(); return getData()->linearVelocityZeroDamping[getIndex()]; }

    AGX_FORCE_INLINE agx::Vec3f& RigidBodyInstance::angularVelocityZeroDamping() { verifyIndex(); return getData()->angularVelocityZeroDamping[getIndex()]; }
    AGX_FORCE_INLINE agx::Vec3f const& RigidBodyInstance::angularVelocityZeroDamping() const { verifyIndex(); return getData()->angularVelocityZeroDamping[getIndex()]; }

    AGX_FORCE_INLINE agx::Physics::GeometryPtr& RigidBodyInstance::geometry() { verifyIndex(); return getData()->geometry[getIndex()]; }
    AGX_FORCE_INLINE agx::Physics::GeometryPtr const& RigidBodyInstance::geometry() const { verifyIndex(); return getData()->geometry[getIndex()]; }

    AGX_FORCE_INLINE agx::Physics::GraphNodePtr& RigidBodyInstance::graphNode() { verifyIndex(); return getData()->graphNode[getIndex()]; }
    AGX_FORCE_INLINE agx::Physics::GraphNodePtr const& RigidBodyInstance::graphNode() const { verifyIndex(); return getData()->graphNode[getIndex()]; }

    AGX_FORCE_INLINE agx::Range6& RigidBodyInstance::velocityRanges() { verifyIndex(); return getData()->velocityRanges[getIndex()]; }
    AGX_FORCE_INLINE agx::Range6 const& RigidBodyInstance::velocityRanges() const { verifyIndex(); return getData()->velocityRanges[getIndex()]; }

    AGX_FORCE_INLINE agx::RigidBody*& RigidBodyInstance::model() { verifyIndex(); return getData()->model[getIndex()]; }
    AGX_FORCE_INLINE agx::RigidBody* const& RigidBodyInstance::model() const { verifyIndex(); return getData()->model[getIndex()]; }

    AGX_FORCE_INLINE agx::UInt32& RigidBodyInstance::solveIndex() { verifyIndex(); return getData()->solveIndex[getIndex()]; }
    AGX_FORCE_INLINE agx::UInt32 const& RigidBodyInstance::solveIndex() const { verifyIndex(); return getData()->solveIndex[getIndex()]; }

    AGX_FORCE_INLINE agx::UInt16& RigidBodyInstance::numConstraints() { verifyIndex(); return getData()->numConstraints[getIndex()]; }
    AGX_FORCE_INLINE agx::UInt16 const& RigidBodyInstance::numConstraints() const { verifyIndex(); return getData()->numConstraints[getIndex()]; }

    //-----------------------------------------------------------------------------------------------------
    AGX_FORCE_INLINE RigidBodySemantics::RigidBodySemantics() {}
    //-----------------------------------------------------------------------------------------------------
    DOXYGEN_END_INTERNAL_BLOCK()
  }
}

AGX_TYPE_BINDING(agx::Physics::RigidBodyPtr, "Physics.RigidBodyPtr")
AGX_TYPE_BINDING(agx::Physics::RigidBodyInstance, "Physics.RigidBodyInstance")

#ifdef _MSC_VER
# pragma warning(pop)
#endif

#undef AGX_ENTITY_WRAPPER
#undef AGX_ENTITY_NAMESPACE
#endif

