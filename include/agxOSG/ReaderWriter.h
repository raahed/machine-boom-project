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

#ifndef AGXOSG_READERWRITER_H
#define AGXOSG_READERWRITER_H

#include <agxOSG/export.h>
#include <agxOSG/RigidBodyRenderCache.h>
#include <string>

#include <agx/PushDisableWarnings.h> // Disabling warnings. Include agx/PopDisableWarnings.h below!
#include <osgDB/ReadFile>
#include <agx/PopDisableWarnings.h> // End of disabled warnings.

#include <agx/Math.h>
#include <agxSDK/Simulation.h>

DOXYGEN_START_INTERNAL_BLOCK()


namespace osg {
  class Node;
}

namespace agxSDK
{
   class Assembly;
}
DOXYGEN_END_INTERNAL_BLOCK()

namespace agxOSG
{
  class ExampleApplication;

  /**
  Read a specified filename from disk and return it as a OpenSceneGraph node.
  The special thing with this is that if searchForIve is true, it will try to find a file named [path]/[filename].ive instead of the
  specified extension.
  Will add the option "noRotation" so that osgDB's plugin for reading .obj-files will not rotate them 90 degrees
  about the (-1, 0, 0) axis (as it does by default).

  \param filename - path to the model that is supposed to be read
  \param searchForIve - if true, a filename with the ive extension will be read instead if found.
  \return true if model is loaded successfully.
  */
  AGXOSG_EXPORT osg::Node* readNodeFile(const std::string& filename, bool searchForIve=false);

  /**
  Read a specified filename from disk and return it as a OpenSceneGraph node.
  The special thing with this is that if searchForIve is true, it will try to find a file named [path]/[filename].ive instead of the
  specified extension.
  \param filename - path to the model that is supposed to be read
  \param options - Options for the call to osgDB::readNodeFile.
    Recommendation: add the option "noRotation" so that osgDB's plugin for reading .obj-files will
  not rotate them 90 degrees about the (-1, 0, 0) axis (as it does by default).
  \param searchForIve - if true, a filename with the ive extension will be read instead if found.
  \return true if model is loaded successfully.
  */
  AGXOSG_EXPORT osg::Node* readNodeFile(const std::string& filename, const osgDB::ReaderWriter::Options* options, bool searchForIve=false );

  /**
  Utility function for reading various files:

  - .agx/.aagx (serialization of simulations)

  For the .scene/.agxScene format: If an parent assembly is specified, nothing will be added to the simulation, only into
  the parent assembly. So you will need to add this assembly to a simulation to be able to simulate the objects.

  The file will be searched for using AGX_ENVIRONMENT().getFilePath(agxIO::Environment::RESOURCE_PATH).find() which means that it will use the AGX_FILE_PATH environment variable
  and path's added to the RESOURCE_PATH.

  The data will be added to the simulation given as an argument
  \param filename - Filename to be opened and read
  \param simulation - The simulation where the content of the data will be added
  \param root - The openscenegraph root node where visuals (if available) will be put
  \param parent - Objects read from the file will be put in this assembly if it is != nullptr
  \param detailRatio - The detail of the visual models, default=0.34, higher is more tessellated
  \param createAxis - If true XYZ axis will be attached to the visual model
  \param selection - Selection of things to read from file.
  See agxSDK::Simulation::ReadSelectionMask. Only for .agx or .aagx.
  \return true if loading was successful, otherwise false.
  */
  AGXOSG_EXPORT bool readFile(const std::string& filename, agxSDK::Simulation *simulation,
                              osg::Group *root, agxSDK::Assembly *parent=nullptr,
                              agx::Real detailRatio=0.35, bool createAxis=false,
                              agx::UInt selection = agxSDK::Simulation::READ_DEFAULT,
                              agxOSG::RigidBodyRenderCache* cache = nullptr );



} // Namespace agxOSG

#endif

