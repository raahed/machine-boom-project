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

# pragma once

#include <agxModel/export.h>

#include <agx/Frame.h>
#include <agx/Physics/GranularBodySystem.h>

#include <agx/ref_ptr.h>

namespace agxModel
{
  class GranularFieldPostProcessor;

  typedef std::pair<agx::Physics::GranularBodyPtr, agx::Real> ParticleWeightPair;
  typedef agx::Vector<ParticleWeightPair> ParticleWeightPairVector;

  AGX_DECLARE_POINTER_TYPES(GranularField);
  AGX_DECLARE_POINTER_TYPES(GranularMassField);
  AGX_DECLARE_POINTER_TYPES(GranularMomentumField);
  AGX_DECLARE_POINTER_TYPES(GranularVelocityField);
  AGX_DECLARE_POINTER_TYPES(GranularCustomBufferField);


  AGX_DECLARE_VECTOR_TYPES(GranularField);
  /**
  Base class for a GranularField. Inherit from this when implementing new fields
  */
  class AGXMODEL_EXPORT GranularField : public agx::Referenced
  {
  public:
    /**
    GranularField base class
    \param dimensions - How many dimension the one field element has. Mass field has dimensions=1 and Velocity field dimensions=3
    \param name - Field name
    */
    GranularField(agx::UInt dimensions, agx::String name);

    /**
    Implement this method in derived classes to define how the particle weight
    pair should be added to the field element with index index
    \param index - What voxel to add the particle field data to
    \param pair - The particle and the its weight
    \return - true if successful
    */
    virtual bool addToField(agx::UInt index, ParticleWeightPair* pair);

    /**
    Implement this method in derived class to do a final computation on the field
    when all the particles have been added. Possible to use if the field depends
    on other fields to be computed first.
    \return - true if successful
    */
    virtual bool calculateFieldElement(agx::UInt /* index */);

    /**
    Virtual destructor.
    */
    virtual ~GranularField();

    /**
    Get ptr to field data
    \return ptr to field data
    */
    agx::Real* getFieldData();

    /**
    Clear the field
    */
    void clearField();

    /**
    Allocate memory for the field
    \return - true if successful
    */
    bool allocateField();

    /**
    Get the field size. That is, the number of voxels in the field times the number of field elements.
    \return - field size
    */
    agx::UInt getFieldSize();

    /**
    Get the shape of the field. x, y, z is the number of voxels the fields granular postprocessor have. w is number of dimensions for the field.
    \return - field shape
    */
    agx::Vec4u getFieldShape();

    /**
    Get the granular field post processor
    \return - GranularFieldPostProcessor
    */
    agxModel::GranularFieldPostProcessor* getGranularFieldPostProcessor();

    /**
    Get field name
    \return - field name
    */
    agx::String getName();

    /**
    Set GranularFieldPostProcessor
    \param pp - Set the granularFieldPostProcessor for this granularField
    */
    void setGranularFieldPostProcesser(agxModel::GranularFieldPostProcessor* pp);

    /**
    Get the names of the fields that this field depends on
    \return - Vector of required fields
    */
    const agx::Vector<agx::String>& getRequiredFields();


  protected:
    agx::Vec4u  m_fieldShape;
    agx::Vector<agx::Real>  m_field;
    agx::Vector<agx::String> m_requiredFields;
    agx::String m_name;

  private:
    agxModel::GranularFieldPostProcessor* m_granularFieldPostProcessor;
  };

  class AGXMODEL_EXPORT GranularMomentumField : public GranularField
  {
  public:
    GranularMomentumField();

    virtual bool addToField(agx::UInt index, ParticleWeightPair* pair);
    virtual bool calculateFieldElement(agx::UInt index);

  protected:
    ~GranularMomentumField();

  private:

  };

  class AGXMODEL_EXPORT GranularMassField : public GranularField
  {
  public:
    GranularMassField();

    virtual bool addToField(agx::UInt index, ParticleWeightPair* pair);
    virtual bool calculateFieldElement(agx::UInt index);

    // Used for debugging mass conservation
    agx::Real calculateTotalMassInField();
  protected:
    ~GranularMassField();

  private:

  };

  class AGXMODEL_EXPORT GranularVelocityField : public GranularField
  {
  public:
    GranularVelocityField();

    virtual bool addToField(agx::UInt index, ParticleWeightPair* pair);
    virtual bool calculateFieldElement(agx::UInt index);

  protected:
    ~GranularVelocityField();

  private:
  };

  class AGXMODEL_EXPORT GranularCustomBufferField : public GranularField
  {
  public:
    /**
    Custom buffer field. Requires that the granularfield name is the same as a custom buffer in the particle system
    \param bufferName - The field name and the name of the custom buffer in the particle system
    */
    GranularCustomBufferField(agx::String bufferName);

    virtual bool addToField(agx::UInt index, ParticleWeightPair* pair);
    virtual bool calculateFieldElement(agx::UInt index);

  protected:
    ~GranularCustomBufferField();
  };
}
