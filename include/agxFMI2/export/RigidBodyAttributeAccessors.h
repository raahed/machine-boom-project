/*
Copyright 2007-2023. Algoryx Simulation AB.

All AGX source code, intellectual property, documentation, sample code,
tutorials, scene files and technical white papers, are copyrighted, proprietary
and confidential material of Algoryx Simulation AB. You may not download, read,
store, distribute, publish, copy or otherwise disseminate, use or expose this
material unless having a written signed agreement with Algoryx Simulation AB, or having been
advised so by Algoryx Simulation AB for a time limited evaluation, or having purchased a
valid commercial license from Algoryx Simulation AB.

Algoryx Simulation AB disclaims all responsibilities for loss or damage caused
from using this software, unless otherwise stated in written agreements with
Algoryx Simulation AB.
*/

#ifndef AGXFMI2_RIGIDBODYATTRIBUTEACCESSORS_H
#define AGXFMI2_RIGIDBODYATTRIBUTEACCESSORS_H

#include <agxFMI2/export/Variable.h>
#include <agx/Physics/RigidBodyEntity.h>
#include <agx/Solver.h>
#include <agxData/Attribute.h>

extern "C"
{
  #include <external/fmi/2.0/fmi2Functions.h>
}

namespace agx {
  class RigidBody;
}

namespace agxFMI2
{
  namespace Export
  {
    //////////////////////////////////////////////////////////////////////////////////////////
    // INPUT VARIABLES
    //////////////////////////////////////////////////////////////////////////////////////////
    AGX_DECLARE_POINTER_TYPES(RigidBodyAttributeInputVariableReal);

    class AGXFMI_EXPORT RigidBodyAttributeInputVariableReal : public InputVariable_Real
    {
    public:
      RigidBodyAttributeInputVariableReal(agx::RigidBody *body, const agx::String& name);

      agx::Physics::RigidBodyRef getBody();
      void setBody(agx::Physics::RigidBodyRef body);
    protected:
      virtual ~RigidBodyAttributeInputVariableReal();

    protected:
      agx::Physics::RigidBodyRef m_body;
    };

    AGX_DECLARE_POINTER_TYPES(RigidBodyVec3AttributeInputVariable);

    // Generic agx::Vec3 attribute
    class AGXFMI_EXPORT RigidBodyVec3AttributeInputVariable : public RigidBodyAttributeInputVariableReal
    {
    public:
      typedef agxData::ScalarAttributeT<agx::Vec3> AttributeT;

    public:
      RigidBodyVec3AttributeInputVariable(agx::RigidBody *body, AttributeT *attribute, agx::UInt component, bool causalityName = false);
      RigidBodyVec3AttributeInputVariable(agx::RigidBody *body, AttributeT *attribute, agx::UInt component, const agx::String& name);
      virtual fmi2Status set(agx::Real value) override;
      virtual fmi2Status get(agx::Real& value) override;

      void setDirectionalDerivativeInput(agx::MobilitySolver *solver, agx::Real value);

    protected:
      virtual ~RigidBodyVec3AttributeInputVariable();

    private:
      using InputVariable_Real::get;
      agx::Real get();

    protected:
      agx::ref_ptr<AttributeT> m_attribute;
      agx::UInt m_component;

    };

#define RB_VEC3_INPUT_VARIABLE(NAME, ATTRIBUTE_NAME)                                                                        \
    class AGXFMI_EXPORT RigidBody ## NAME ## InputVariable : public RigidBodyVec3AttributeInputVariable                    \
    {                                                                                                                        \
    public:                                                                                                                  \
      RigidBody ## NAME ## InputVariable(agx::RigidBody *body, agx::UInt component, bool causalityName = false) : RigidBodyVec3AttributeInputVariable(body, agx::Physics::RigidBodyModel::ATTRIBUTE_NAME ## Attribute, component, causalityName) {} \
                                                                                                                             \
    protected:                                                                                                               \
      virtual ~RigidBody ## NAME ## InputVariable() {}                                                                            \
    }

    RB_VEC3_INPUT_VARIABLE(Force, force);
    RB_VEC3_INPUT_VARIABLE(Torque, torque);
    RB_VEC3_INPUT_VARIABLE(Velocity, velocity);
    RB_VEC3_INPUT_VARIABLE(AngularVelocity, angularVelocity);

    AGX_DECLARE_POINTER_TYPES(RigidBodyForceAccumulatorInputVariable);

    class AGXFMI_EXPORT RigidBodyForceAccumulatorInputVariable : public RigidBodyVec3AttributeInputVariable
    {
    public:
      RigidBodyForceAccumulatorInputVariable(agx::RigidBody *body, agx::UInt component, bool causalityName = false);
      virtual fmi2Status set(agx::Real value) override;
      void apply();

    protected:
      virtual ~RigidBodyForceAccumulatorInputVariable();

    private:
      agx::Real m_storedValue;
    };

    AGX_DECLARE_POINTER_TYPES(RigidBodyTorqueAccumulatorInputVariable);

    class AGXFMI_EXPORT RigidBodyTorqueAccumulatorInputVariable : public RigidBodyVec3AttributeInputVariable
    {
    public:
      RigidBodyTorqueAccumulatorInputVariable(agx::RigidBody *body, agx::UInt component, bool causalityName = false);
      virtual fmi2Status set(agx::Real value) override;
      void apply();

    protected:
      virtual ~RigidBodyTorqueAccumulatorInputVariable();

    private:
      agx::Real m_storedValue;
    };


    class AGXFMI_EXPORT RigidBodyPositionInputVariable : public RigidBodyAttributeInputVariableReal
    {
    public:
      RigidBodyPositionInputVariable(agx::RigidBody *body, agx::UInt component, bool causalityName = false);
      virtual fmi2Status set(agx::Real value) override;
      virtual fmi2Status get(agx::Real& value) override;

    protected:
      virtual ~RigidBodyPositionInputVariable();

    private:
      agx::UInt m_component;
    };

    class AGXFMI_EXPORT RigidBodyRotationQuatInputVariable : public RigidBodyAttributeInputVariableReal
    {
    public:
      RigidBodyRotationQuatInputVariable(agx::RigidBody *body, agx::UInt component, bool causalityName = false);
      virtual fmi2Status set(agx::Real value) override;
      virtual fmi2Status get(agx::Real& value) override;

    protected:
      virtual ~RigidBodyRotationQuatInputVariable();

    private:
      agx::UInt m_component;
    };

    class AGXFMI_EXPORT RigidBodyRotationEulerInputVariable : public RigidBodyAttributeInputVariableReal
    {
    public:
      RigidBodyRotationEulerInputVariable(agx::RigidBody *body, agx::UInt component, bool causalityName = false);
      virtual fmi2Status set(agx::Real value) override;
      virtual fmi2Status get(agx::Real& value) override;

    protected:
      virtual ~RigidBodyRotationEulerInputVariable();

    private:
      agx::UInt m_component;
    };


    //////////////////////////////////////////////////////////////////////////////////////////
    // OUTPUT VARIABLES
    //////////////////////////////////////////////////////////////////////////////////////////

    // Generic agx::Vec3 attribute

    class AGXFMI_EXPORT RigidBodyVec3AttributeOutputVariable : public OutputVariable_Real
    {
    public:
      typedef agxData::ScalarAttributeT<agx::Vec3> AttributeT;

    public:
      RigidBodyVec3AttributeOutputVariable(agx::RigidBody *body, AttributeT *attribute, agx::UInt component, bool causalityName = false);
      virtual fmi2Status get(agx::Real& value) override;
      agx::Real getDirectionalDerivativeOutput(agx::MobilitySolver *solver, agx::MobilitySolver *referenceSolver);
      agx::Real get(agx::MobilitySolver *solver);

    protected:
      virtual ~RigidBodyVec3AttributeOutputVariable();

    private:
      agx::Physics::RigidBodyRef m_body;
      agx::ref_ptr<AttributeT> m_attribute;
      agx::UInt m_component;
    };

#define RB_VEC3_OUTPUT_VARIABLE(NAME, ATTRIBUTE_NAME)                                                                        \
    class AGXFMI_EXPORT RigidBody ## NAME ## OutputVariable : public RigidBodyVec3AttributeOutputVariable                    \
    {                                                                                                                        \
    public:                                                                                                                  \
      RigidBody ## NAME ## OutputVariable(agx::RigidBody *body, agx::UInt component, bool causalityName = false) : RigidBodyVec3AttributeOutputVariable(body, agx::Physics::RigidBodyModel::ATTRIBUTE_NAME ## Attribute, component, causalityName) {} \
                                                                                                                             \
    protected:                                                                                                               \
      virtual ~RigidBody ## NAME ## OutputVariable() {}                                                                            \
    }

    RB_VEC3_OUTPUT_VARIABLE(Force, force);
    RB_VEC3_OUTPUT_VARIABLE(Torque, torque);
    RB_VEC3_OUTPUT_VARIABLE(AppliedForce, lastForce);
    RB_VEC3_OUTPUT_VARIABLE(AppliedTorque, lastTorque);
    RB_VEC3_OUTPUT_VARIABLE(Acceleration, linearAcceleration);
    RB_VEC3_OUTPUT_VARIABLE(AngularAcceleration, angularAcceleration);
    RB_VEC3_OUTPUT_VARIABLE(Velocity, velocity);
    RB_VEC3_OUTPUT_VARIABLE(AngularVelocity, angularVelocity);


    class AGXFMI_EXPORT RigidBodyPositionOutputVariable : public OutputVariable_Real
    {
    public:
      RigidBodyPositionOutputVariable(agx::RigidBody *body, agx::UInt component, bool causalityName = false);
      virtual fmi2Status get(agx::Real& value) override;

    protected:
      virtual ~RigidBodyPositionOutputVariable();

    private:
      agx::Physics::RigidBodyRef m_body;
      agx::UInt m_component;
    };

    class AGXFMI_EXPORT RigidBodyRotationQuatOutputVariable : public OutputVariable_Real
    {
    public:
      RigidBodyRotationQuatOutputVariable(agx::RigidBody *body, agx::UInt component, bool causalityName = false);
      virtual fmi2Status get(agx::Real& value) override;

    protected:
      virtual ~RigidBodyRotationQuatOutputVariable();

    private:
      agx::Physics::RigidBodyRef m_body;
      agx::UInt m_component;
    };

    class AGXFMI_EXPORT RigidBodyRotationEulerOutputVariable : public OutputVariable_Real
    {
    public:
      RigidBodyRotationEulerOutputVariable(agx::RigidBody *body, agx::UInt component, bool causalityName = false);
      virtual fmi2Status get(agx::Real& value) override;

    protected:
      virtual ~RigidBodyRotationEulerOutputVariable();

    private:
      agx::Physics::RigidBodyRef m_body;
      agx::UInt m_component;
    };
  }
}

#endif /* AGXFMI2_RIGIDBODYATTRIBUTEACCESSORS_H */
