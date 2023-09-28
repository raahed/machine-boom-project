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

#include <agxCable/export.h>
#include <agx/String.h>

namespace agxCable {


  class CableProperties;

  /**
  Class with methods to read and write CableProperties from and to json files or strings.
  */
  class AGXCABLE_EXPORT CablePropertiesReaderWriter
  {
    public:

      /**
      Read parameters from CableProperties and write result to a String
      \param properties The CableProperties whose values should be read
      \param jsonOutput String that will contain the output
      \return True if json output could be assigned to the string
      */
      static bool writeJsonString( const agxCable::CableProperties* properties, agx::String& jsonOutput );


      /**
      Read parameters from CableProperties and write result to a CableProperties definition file
      \param properties The CableProperties whose values should be read
      \param filename Name of file where data should be written
      \return True if file could successfully be written
      */
      static bool writeJson( const agxCable::CableProperties* properties, const agx::String& filename );

      /**
      Read parameters from a string with json data and assign values to CableProperties
      \param properties The Material to modify
      \param jsonContents json data
      \return True if json data could be read and assigned to the CableProperties
      */
      static bool readJsonString( agxCable::CableProperties* properties, const agx::String& jsonContents );


      /**
      Read parameters from a file with json data and assign values to CableProperties
      \param properties The Material to modify
      \param filename Name of input file
      \return True if json data could be read and assigned to the CableProperties
      */
      static bool readJson( agxCable::CableProperties* properties, const agx::String& filename );




  };


}

