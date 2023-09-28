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

#pragma once

#include <agxModel/export.h>
#include <agx/String.h>

namespace agxModel {


  class BeamModelProperties;

  /**
  Class with methods to read and write BeamModelProperties from and to json files or strings.
  */
  class AGXMODEL_EXPORT BeamModelPropertiesReaderWriter
  {
    public:

      /**
      Read parameters from BeamModelProperties and write result to a String
      \param properties The BeamModelProperties whose values should be read
      \param jsonOutput String that will contain the output
      \return True if json output could be assigned to the string
      */
      static bool writeJsonString( const agxModel::BeamModelProperties* properties, agx::String& jsonOutput );


      /**
      Read parameters from BeamModelProperties and write result to a BeamModelProperties definition file
      \param properties The BeamModelProperties whose values should be read
      \param filename Name of file where data should be written
      \return True if file could successfully be written
      */
      static bool writeJson( const agxModel::BeamModelProperties* properties, const agx::String& filename );

      /**
      Read parameters from a string with json data and assign values to BeamModelProperties
      \param properties The Material to modify
      \param jsonContents json data
      */
      static bool readJsonString( agxModel::BeamModelProperties* properties, const agx::String& jsonContents );


      /**
      Read parameters from a file with json data and assign values to BeamModelProperties
      \param properties The Material to modify
      \param filename Name of input file
      */
      static bool readJson( agxModel::BeamModelProperties* properties, const agx::String& filename );




  };


}

