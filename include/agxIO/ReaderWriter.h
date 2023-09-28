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

#ifndef AGXIO_READERWRITER_H
#define AGXIO_READERWRITER_H

#include <agx/agxPhysics_export.h>
#include <string>
#include <agx/String.h>
#include <agx/Vec3.h>
#include <agxSDK/Simulation.h>

namespace agxSDK
{
  class Assembly;
}

namespace agxIO
{
  /**
  Utility function for reading various files:

  - .scene/.agxScene (ConfigScript description of a scene)
  - .agx/.aagx (serialization of simulations)

  For the .scene/.agxScene format: If an parent assembly is specified, nothing will be added to the simulation, only into
  the parent assembly. So you will need to add this assembly to a simulation to be able to simulate the objects.

  The file will be searched for using AGX_ENVIRONMENT().getFilePath(agxIO::Environment::RESOURCE_PATH).find() which means that it will use the AGX_FILE_PATH environment variable
  and path's added to the RESOURCE_PATH.

  The data will be added to the simulation given as an argument
  \param filename - Filename to be opened and read
  \param simulation - The simulation where the content of the data will be added
  \param parent - If specified (!= nullptr) all objects will be added to this parent
  \param selection - Selection of things to read from file.
  See agxSDK::Simulation::ReadSelectionMask. Only for .agx or .aagx.
  \return true if reading was successful.
  */
  AGXPHYSICS_EXPORT bool readFile(const agx::String& filename, agxSDK::Simulation* simulation,
    agxSDK::Assembly* parent = nullptr,
    agx::UInt selection = agxSDK::Simulation::READ_DEFAULT);

  /// Specify the type of file
  enum FileType {
    FILETYPE_AGX, /*<! .agx binary format */
    FILETYPE_AAGX, /*<! .aagx ascii format */
    FILETYPE_POV /*<! Povray file format */
  };

  /**
  Utility function for write a simulation to a stream.

  \param type - Specifies in what format, the simulation will be written to the stream
  \param outStream - Stream that will contain the simulation
  \param simulation - The simulation containing the objects to be written to file
  \param from - Camera position (only for .pov format)
  \param at - Camera looking at point (only for .pov format)
  \param up - Camera up direction (only for .pov format)
  \param fovy - Field of view  (only for .pov format)
  \param aspectRatio ratio for the window size  (only for .pov format)
  \return true if writing was successful.
  */
  AGXPHYSICS_EXPORT bool writeFile(FileType type, std::ostream& outStream,
    agxSDK::Simulation* simulation, agx::Vec3 from = agx::Vec3(),
    agx::Vec3 at = agx::Vec3(), agx::Vec3 up = agx::Vec3(),
    double fovy = 0, double aspectRatio = 0 );

  /**
  Utility function for write a simulation into various files:

  - .agx/.aagx (serialization of simulations)
  - .pov (Povray format)

  \param filename - Filename that will contain the simulation
  \param simulation - The simulation containing the objects to be written to file
  \param from - Camera position (only for .pov format)
  \param at - Camera looking at point (only for .pov format)
  \param up - Camera up direction (only for .pov format)
  \param fovy - Field of view  (only for .pov format)
  \param aspectRatio ratio for the window size  (only for .pov format)
  \return true if writing was successful.
  */
  AGXPHYSICS_EXPORT bool writeFile(const agx::String& filename, agxSDK::Simulation* simulation,
    agx::Vec3 from = agx::Vec3(), agx::Vec3 at = agx::Vec3(),
    agx::Vec3 up = agx::Vec3(), double fovy = 0, double aspectRatio = 0 );



  /**
  Function which will create a zipped file of name \p archiveName containing the files listed in \p files

  Valid file extensions are:

  - .agxz (archive will contain a compressed .agx file)
  - .aagxz (archive will contain a compressed .aagx file)
  - .agxPyz (archive will contain a .agxPy file of the same name as the archive + any other files/directories with files required for the .agxPy file to run)
  - .agxLuaz (archive will contain a .agxLua file of the same name as the archive + any other files/directories with files required for the .agxLua file to run)
  - .mpyz (archive will contain a .agx file of the same name as the archive + one or more .mpy files)

  \param archiveName - Name of created archive
  \param files - List of file to be compressed into archive
  \return true if the archive is successfully created and the list of files contains the above listed ones.
  */
  AGXPHYSICS_EXPORT bool createCompressedArchive(const agx::String& archiveName, const agx::StringVector& files);

  /**
  Function which will de-compress an archive to a target directory

  - .agxz (archive need to contain a compressed .agx file)
  - .aagxz (archive need to contain a compressed .aagx file)
  - .agxPyz (archive need to contain a .agxPy file of the same name as the archive + any other files/directories with files required for the .agxPy file to run)
  - .agxLuaz (archive need to contain a .agxLua file of the same name as the archive + any other files/directories with files required for the .agxLua file to run)
  - .mpyz (archive need to contain a .agx file of the same name as the archive + one or more .mpy files)


  \param archiveFile - Name of zipped archive
  \param directory - Path to where files are unzipped.

  \return true if files are successfully unzipped and contains the files as listed above
  */
  AGXPHYSICS_EXPORT bool decompressArchive(const agx::String& archiveFile, const agx::String& directory);
}

#endif
