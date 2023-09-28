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

#ifndef AGX_DEFORMABLEPARTICLEMESH_H
#define AGX_DEFORMABLEPARTICLEMESH_H

#include <agxSDK/Assembly.h>

#include <agxUtil/RawMesh.h>
#include <agx/DistanceJoint.h>
#include <agxModel/export.h>
#include <agxStream/Serializable.h>

namespace agxModel
{

  class AGXMODEL_EXPORT DeformableParticleMesh : public agxSDK::Assembly
  {
  public:
    // Used for serialization
    DeformableParticleMesh();

    /**
    Create a deformable mesh from the RawMesh specified. Vertices become particles, and edges become
    distance joints.

    Use setMeshProperties/setConstraintProperties/setParticleProperties to set the default values of the
    different parts of the DeformableParticleMesh, and then call init() to actually create the structure and
    add it to the Assembly.

    \param rawMesh the mesh to form the structure of the DeformableParticleMesh
    */
    DeformableParticleMesh(
      agxUtil::RawMesh* rawMesh);

    /**
    Set the properties of the entire mesh.
    \param mass The mass of the mesh, will be evenly distributed over the particles
    */
    void setMass(agx::Real mass);

    /**
    Set the solve type for the distance joints connecting the particles.
    \param solveType The type of solver used for the constraint
    */
    void setConstraintSolveType(agx::Constraint::SolveType solveType);

    /**
    Set the compliance of the distance joints connecting the particles.
    \param compliance the default compliance of the constraints
    */
    void setConstraintCompliance(agx::Real compliance);

    /**
    Set the radius of all the particles. The geometry used for the particles is a sphere.
    \param radius the radius of the particle spheres
    */
    void setParticleRadius(agx::Real radius);

    /**
    Set the material of all the particles.
    \param material the material of the particles
    */
    void setParticleMaterial(agx::Material* material);

    /**
    Add a collision group to all particles in the mesh.
    \param name the name of the group
    */
    void addParticleGroup(const agx::Name& name);

    /**
    Add a collision group to all particles in the mesh.
    \param id the id of the group
    */
    void addParticleGroup(agx::UInt32 id);

    /**
    Remove a collision group from all particles in the mesh.
    \param name the name of the group
    */
    void removeParticleGroup(const agx::Name& name);

    /**
    Remove a collision group from all particles in the mesh.
    \param id the id of the group
    */
    void removeParticleGroup(agx::UInt32 id);

    /**
    Get the indices of all particles in each face. Index over
    getParticles.
    */
    agx::Vector<agx::UIntVector> getFaces() const;

    /**
    Get a list of all the particles in the mesh.
    */
    const agx::RigidBodyRefVector& getParticles() const;

    /**
    Get a list of all the joints in the mesh.
    These are not guaranteed to be ordered in any particular way.
    */
    const agx::Vector<agx::DistanceJointRef>& getConstraints() const;

    /**
      Get the mesh without deformation.
    */
    agxUtil::RawMesh* getSourceMesh() const;

    AGXSTREAM_DECLARE_SERIALIZABLE(agxModel::DeformableParticleMesh);

  private:
    void addJoint(size_t firstParticle, size_t secondParticle);

    agx::RigidBodyRefVector m_particles;
    agx::Vector<agx::DistanceJointRef> m_joints;

    agx::MaterialRef m_particleMaterial;

    agxUtil::RawMeshRef m_sourceMesh;
  };

}
#endif // AGX_DEFORMABLEPARTICLEMESH_H
