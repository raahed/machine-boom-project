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

#include <agx/agxPhysics_export.h>
#include <agx/String.h>

namespace agx {

  class Material;
  class ContactMaterial;

  /**
  Utility class for reading and writing Materials.
  */
  class AGXPHYSICS_EXPORT MaterialReaderWriter
  {
    public:

      /**
      Read parameters from a Material and write result to a String
      \param material The Material whose values should be read
      \param jsonOutput String that will contain the output
      \return True if json output could be assigned to the string
      */
      static bool writeJsonString( const agx::Material* material, agx::String& jsonOutput );


      /**
      Read parameters from a Material and write result to a Material definition file
      \param material The Material whose values should be read
      \param filename Name of file where data should be written
      \return True if file could successfully be written
      */
      static bool writeJson( const agx::Material* material, const agx::String& filename );

      /**
      Read parameters from a string with json data and assign values to a Material
      \param material The Material to modify
      \param jsonContents json data
      \return True if json data could be read and assigned to the Material
      */
      static bool readJsonString( agx::Material* material, const agx::String& jsonContents );


      /**
      Read parameters from a file with json data and assign values to a Material
      \param material The Material to modify
      \param filename Name of input file
      \return True if json data could be read and assigned to the Material
      */
      static bool readJson( agx::Material* material, const agx::String& filename );




      /**
      Read parameters from a ContactMaterial and write result to a String
      \param contactMaterial The ContactMaterial whose values should be read
      \param jsonOutput String that will contain the output
      \return True if json output could be assigned to the string
      */
      static bool writeJsonString( const agx::ContactMaterial* contactMaterial, agx::String& jsonOutput );


      /**
      Read parameters from a ContactMaterial and write result to a ContactMaterial definition file
      \param contactMaterial The ContactMaterial whose values should be read
      \param filename Name of file where data should be written
      \return True if file could successfully be written
      */
      static bool writeJson( const agx::ContactMaterial* contactMaterial, const agx::String& filename );

      /**
      Read parameters from a string with json data and assign values to a ContactMaterial
      \param contactMaterial The ContactMaterial to modify
      \param jsonContents json data
      \return True if json data could be read and assigned to the ContactMaterial
      */
      static bool readJsonString( agx::ContactMaterial* contactMaterial, const agx::String& jsonContents );

      /**
      Read parameters from a file with json data and assign values to a ContactMaterial
      \param contactMaterial The ContactMaterial to modify
      \param filename Name of input file
      \return True if json data could be read and assigned to the ContactMaterial
      */
      static bool readJson( agx::ContactMaterial* contactMaterial, const agx::String& filename );

  };

}

