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

#ifndef AGXFMI2_GEOMETRYATTRIBUTEACCESSORS_H
#define AGXFMI2_GEOMETRYATTRIBUTEACCESSORS_H

#include <agxFMI2/export/Variable.h>
#include <agx/Physics/GeometryEntity.h>
#include <agx/Solver.h>
#include <agxData/Attribute.h>

extern "C"
{
  #include <external/fmi/2.0/fmi2Functions.h>
}

namespace agxCollide {
  class Geometry;
}

namespace agxFMI2
{
  namespace Export
  {
    //////////////////////////////////////////////////////////////////////////////////////////
    // INPUT VARIABLES
    //////////////////////////////////////////////////////////////////////////////////////////

    // Generic agx::Vec3 attribute
    class AGXFMI_EXPORT GeometryVec3AttributeInputVariable : public InputVariable_Real
    {
    public:
      typedef agxData::ScalarAttributeT<agx::Vec3f> AttributeT;

    public:
      GeometryVec3AttributeInputVariable(agxCollide::Geometry *geometry, AttributeT *attribute, agx::UInt component, bool causalityName = false);
      virtual fmi2Status set(agx::Real value) override;

      void setDirectionalDerivativeInput(agx::MobilitySolver *solver, agx::Real value);

    protected:
      virtual ~GeometryVec3AttributeInputVariable();

    private:
      using InputVariable_Real::get;
      agx::Real get();

    private:
      agx::Physics::GeometryRef m_geometry;
      agx::ref_ptr<AttributeT> m_attribute;
      agx::UInt m_component;
    };


    // Surface Velocity
    class AGXFMI_EXPORT GeometrySurfaceVelocityInputVariable : public GeometryVec3AttributeInputVariable
    {
    public:
      GeometrySurfaceVelocityInputVariable(agxCollide::Geometry *geometry, agx::UInt component, bool causalityName = false) : GeometryVec3AttributeInputVariable(geometry, agx::Physics::GeometryModel::surfaceVelocityAttribute, component, causalityName) {}

    protected:
      virtual ~GeometrySurfaceVelocityInputVariable() {}
    };

    //////////////////////////////////////////////////////////////////////////////////////////
    // OUTPUT VARIABLES
    //////////////////////////////////////////////////////////////////////////////////////////

    // Generic agx::Vec3 attribute

    class AGXFMI_EXPORT GeometryVec3AttributeOutputVariable : public OutputVariable_Real
    {
    public:
      typedef agxData::ScalarAttributeT<agx::Vec3f> AttributeT;

    public:
      GeometryVec3AttributeOutputVariable(agxCollide::Geometry* geometry, AttributeT *attribute, agx::UInt component, bool causalityName = false);
      virtual fmi2Status get(agx::Real& value) override;
      agx::Real getDirectionalDerivativeOutput(agx::MobilitySolver *solver, agx::MobilitySolver *referenceSolver);

    protected:
      virtual ~GeometryVec3AttributeOutputVariable();

    private:
      agx::Real get(agx::MobilitySolver *solver);

    private:
      agx::Physics::GeometryRef m_geometry;
      agx::ref_ptr<AttributeT> m_attribute;
      agx::UInt m_component;
    };

    class AGXFMI_EXPORT GeometrySurfaceVelocityOutputVariable : public GeometryVec3AttributeOutputVariable
    {
    public:
      GeometrySurfaceVelocityOutputVariable(agxCollide::Geometry *geometry, agx::UInt component, bool causalityName = false) : GeometryVec3AttributeOutputVariable(geometry, agx::Physics::GeometryModel::surfaceVelocityAttribute, component, causalityName) {}

    protected:
      virtual ~GeometrySurfaceVelocityOutputVariable() {}
    };
  }
}

#endif /* AGXFMI2_GEOMETRYATTRIBUTEACCESSORS_H */
