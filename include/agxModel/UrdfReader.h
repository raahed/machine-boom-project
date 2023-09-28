/*
Copyright 2007-2023. Algoryx Simulation AB.

All AGX source code, intellectual property, documentation, sample code,
tutorials, scene files and technical white papers, is copyrighted, proprietary
and confidential material of Algoryx Simulation AB. You may not download, read,
store, distribute, publish, copy or otherwise disseminate, use or expose this
material unless having a written signed agreement with Algoryx Simulation AB, or having been
advised so by Algoryx Simulation AB for a time limited evaluation, or having purchased a
valid commercial license from Algoryx Simulation AB.

Algoryx Simulation AB disclaims all responsibilities for loss or damage caused
from using this software, unless otherwise stated in written agreements with
Algoryx Simulation AB.
*/

#pragma once

#include <agxModel/export.h>
#include <agxUtil/agxUtil.h>

namespace agxSDK
{
  class Assembly;
}

namespace agxModel
{
  /// Class for reading a URDF file into an agxSDK::Assembly
  class AGXMODEL_EXPORT UrdfReader
  {
  public:
    // Disallow creating an instance of this class.
    UrdfReader() = delete;

    /**
    Settings for controlling details regarding the URDF reader.
    */
    struct AGXMODEL_EXPORT Settings
    {
      /**
      \param fixToWorld_ - if set to true the base link will be attached to the world with a agx::LockJoint
      \param disableLinkedBodies_ - If true, the collision detection between linked/constrained will be disabled.
      \param mergeKinematicLinks_ - If true, and if a Link is missing the "inertial" element, it is according to the specification to be treated as a kinematic link. 
                                    This link will get a negnegligible mass and will be merged with its parent and the constraint between this link and its parent will be disabled.
                                    If false, the body will get its mass properties calculated from the shape/volume/density.

      */
      explicit Settings(bool fixToWorld_ = false, bool disableLinkedBodies_ = false, bool mergeKinematicLinks_ = true);

      bool fixToWorld = false;
      bool disableLinkedBodies = false;
      bool mergeKinematicLinks = true;    
    };

    /**
    Reads a URDF file and returns an agxSDK::Assembly representation of it.
    \param filePath - the path to the URDF file. This can be a relative path using agxIO::Environment::instance()->getFilePath(agxIO::Environment::RESOURCE_PATH) to locate the file.
    \param packagePath - the absolute path to the 'package://' portion of any filename attributes in the URDF file
    \param joints - initial joint positions. Note that only revolute, continuous and prismatic joints are counted so
                    the number of joint elements should match the total number of those joint types in the URDF model
    \param settings - Settings for the URDF reader.
    \return an agxSDK::AssemblyRef representing the URDF model
    */
    static agxSDK::AssemblyRef read(const agx::String& filePath,
                                    const agx::String& packagePath = "", 
                                    const agx::RealVector* joints = nullptr,
                                    const Settings& settings = Settings());
  };
}

