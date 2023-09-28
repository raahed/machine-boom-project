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

#ifndef AGXOSG_FBXUTILS_H
#define AGXOSG_FBXUTILS_H

#if defined (SWIG) || defined(_MSC_VER)
#include <agx/config/AGX_USE_FBX.h>
#if defined(SWIG) || AGX_USE_FBX()

#define FBXSDK_NEW_API

#include <agx/agx.h>
#include <agxUtil/agxUtil.h>
#include <agxOSG/export.h>
#include <agxSDK/SimulationController.h>

namespace agxOSG {

  DOXYGEN_START_INTERNAL_BLOCK()
  class FBXInterface;
  DOXYGEN_END_INTERNAL_BLOCK()

  /**
  Writer for the FBX 3D format including animations.
  This class cannot handle a simulation that changes in structure regarding bodies/geometries.
  So do not call add/remove on geometries during the simulation/playback.
  */
  class AGXOSG_EXPORT FBXWriter : public agx::Referenced
  {
  public:
    /**
    Create a FBXWriter that will write to the specified file in FBX 7.3 binary format
    \path - File path to the generated fbx file
    */
    FBXWriter();

    /**
    This method will clear any previously stored data and make the writer ready for creating
    a new fbx file.
    \path - File path to the generated fbx file
    */
    void init(const agx::String& path);


    /**
    Connect the FBXWriter to the simulation, it will use a StepEventListener to animate each time step.
    \param simulation - The Simulation to connect too.
    \return true if successful
    */
    bool connect(agxSDK::Simulation *simulation);

    /**
    Connect to an existing simulation controller with a journal. At playback the kinematic data
    for the registered bodies will be recorded into the FBX file.
    \param simulationController a SimulationController containing the Journal to run.
    */
    bool connect(agxSDK::SimulationController *simulationController);

    /**
    Disconnect FBXWriter from simulation/journal, will smooth rotations and disconnect listeners
    */
    bool disconnect();

    /**
    Add a mesh to the fbx file from object path, the mesh can then be animated using its id.
    This method MUST be called for each object that are supposed to be part of the simulation
    \param id - The node id to be used when animating the mesh
    \return true if successful
    */
    bool addMesh(agx::UInt32 id,
      const agx::Vec3Vector& vertices,
      const agx::UInt32Vector& indices,
      const agx::Vec3Vector& normals,
      const agx::Vec2Vector& texCoordinates,
      const agx::AffineMatrix4x4& transform,
      agxCollide::RenderMaterial *material = nullptr);

    /**
    Write the actual fbx file from the added data.
    \return true if successful
    */
    bool writeToFile();

    DOXYGEN_START_INTERNAL_BLOCK()
    /// \return the interface for the FBX API.
    agxOSG::FBXInterface *getInterface();
    DOXYGEN_END_INTERNAL_BLOCK()


  protected:
    virtual ~FBXWriter();
  protected:
    agx::observer_ptr<agxSDK::StepEventListener> m_listener;

    agx::ref_ptr<agx::Referenced> m_fbxInterface;
  private:
  };
}

// AGX_USE_FBX
#endif

#endif
#endif
