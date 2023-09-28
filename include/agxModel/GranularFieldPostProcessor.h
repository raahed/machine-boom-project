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
#include <agxModel/GranularFields.h>

#include <agx/ref_ptr.h>

namespace agxModel
{
  class GranularField;

  typedef std::pair<agx::Physics::GranularBodyPtr, agx::Real> ParticleWeightPair;
  typedef agx::Vector<ParticleWeightPair> ParticleWeightPairVector;

  AGX_DECLARE_POINTER_TYPES(GranularFieldPostProcessor);

  /**
  Class for calculating different granular fields. The fields are
  discretized into voxels and every particle in the system is sorted into
  voxels together with a weight. The field value in every voxel is then the
  weighted sum of the particles in that voxel.
  */
  class AGXMODEL_EXPORT GranularFieldPostProcessor : public agx::Referenced
  {
  public:

    /**
    GranularFieldPostProcessor that computes voxel discretized granular fields for a given granular body system inside a rectangular volume
    \param gbs - Granular body system for the field
    \param voxelCount - How many voxel in each dimension to fit inside the volume
    \param halfExtents - Half size of volumes sides
    */
    GranularFieldPostProcessor(agx::Physics::GranularBodySystem* gbs, const agx::Vec3u& voxelCount, const agx::Vec3& halfExtents);

    /**
    Sorts particles into voxels and calculate the fields that are added to this class
    \return - true if successful
    */
    bool calculateFields();

    /**
    Adds a new field. This allocates memory in the field class given the shape of the field defined in this class
    \param gf - GranularField that should be calculated by the GranularFieldPostProcessor
    */
    bool addField(agxModel::GranularField* gf);

    /**
    Remove field with this name
    \param name - Remove granularField with this name
    */
    bool removeField(const agx::String& name);

    /**
    \return Voxel volume
    */
    agx::Real getVoxelVolume();

    /**
    \return voxel count
    */
    agx::Vec3u getVoxelCount();

    /**
    \return half extents of the volume
    */
    agx::Vec3 getHalfExtents();

    /**
    \return the size of one voxel
    */
    agx::Vec3 getVoxelSize();

    /**
    \return frame for the field
    */
    agx::Frame* getFrame();

    /**
    Get a field added to the GranularFieldPostProcessor
    \param name - Name of the field to get
    \return field or nullptr if the field was not added
    */
    agxModel::GranularField* getField(const agx::String& name);

    /**
    \return a vector of all the fields
    */
    const agxModel::GranularFieldRefVector& getFields();

    /**
    \return the granular body system.
    */
    agx::Physics::GranularBodySystem* getGranularBodySystem();

    /**
    Set position of the field volume in world coordinates
    \param pos - position in world coordinates
    */
    void setPosition(const agx::Vec3& pos);

    /**
    Set rotation of the field volume in world coordinates
    \param quat - rotation in world coordinates
    */
    void setRotation(const agx::Quat & quat);

    /**
    Set position of the field volume in world coordinates
    \param simulation - simulation
    */
    void createDebugVisualization(agxSDK::Simulation* simulation);

  protected:
    virtual ~GranularFieldPostProcessor();

  private:
    agx::Physics::GranularBodySystemRef m_granularBodySystem;
    agx::FrameRef                       m_frame;
    agx::Vec3u                          m_voxelCount; // x is width, y is depth, z is height
    agx::Vec3                           m_halfExtents;
    agx::Vec3                           m_voxelSize;

    agxModel::GranularFieldRefVector m_fields;

  public:
    agxData::BufferRef m_particleInfo;
    agxData::BufferRef m_particleVoxelIndices;
    agxData::BufferRef m_particleVoxelWeights;

  private:
    /* Sorts the particles into voxels. The result is in the particleWeightPairVector. */
    bool sortParticlesIntoVoxels();
    agx::Vec3i getVoxelIndexFromPosition(const agx::Vec3& position) const;
    agx::Vec3 getPositionRelativeFirstVoxel(const agx::Vec3& position) const;

    agx::UInt getFlatIndex(const agx::Vec3i& index) const;

    void addRequiredFields(const agx::Vector<agx::String>& fields);

  };

}
