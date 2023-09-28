/*
Copyright 2007-2023. Algoryx Simulation AB.

All AGX source code, intellectual property, documentation, sample code,
tutorials, scene files and technical white papers, are copyrighted, proprietary
and confidential material of Algoryx Simulation AB. You may not download, read,
store, distribute, publish, copy or otherwise disseminate, use or expose this
material unless having a written signed agreement with Algoryx Simulation AB, or
having been advised so by Algoryx Simulation AB for a time limited evaluation,
or having purchased a valid commercial license from Algoryx Simulation AB.

Algoryx Simulation AB disclaims all responsibilities for loss or damage caused
from using this software, unless otherwise stated in written agreements with
Algoryx Simulation AB.
*/

#pragma once

#include <agx/config/AGX_USE_AGXTERRAIN.h>
#include <agxTerrain/export.h>
#include <agx/String.h>



namespace agxTerrain
{
  class TerrainMaterial;

  /**
  Utility class for reading and writing TerrainMaterial objects to JSON format. The data can be written to both
  files and strings.
  */
  class AGXTERRAIN_EXPORT TerrainMaterialReaderWriter
  {

  public:
    /**
    Read TerrainMateiral configuration from a file in JSON format into a TerrainMaterial object.
    \param filename - the filename of the file containing the terrain material configuration in JSON format.
    \param terrainMaterial - the TerrainMaterial object that will be initialized with the data from the JSON file.
    \return true if the file was successfully read, false otherwise.
    */
    static bool readJson( TerrainMaterial* terrainMaterial, const agx::String& filename );

    /**
    Write a TerrainMaterial configuration to a file in JSON format.
    \param terrainMaterial - the TerrainMaterial object that will be written to the JSON file.
    \param filename - the filename of the file where the the terrain material configuration will be written in JSON format.
    \return true if the file was successfully written, false otherwise.
    */
    static bool writeJson( const TerrainMaterial* terrainMaterial, const agx::String& filename );

    /**
    Write a TerrainMaterial configuration to a string in JSON format.
    \param terrainMaterial - the TerrainMaterial object that will be written to the JSON string.
    \param jsonString - the string where the JSON data will be written.
    \return true if the data was successfully written to the JSON string, false otherwise.
    */
    static bool writeJsonString( const TerrainMaterial* terrainMaterial, agx::String& jsonString );

    /**
    Read a TerrainMaterial configuration from a string in JSON format.
    \param jsonString - the string where the JSON data will be read from.
    \param terrainMaterial - the TerrainMaterial object that will be written to from the JSON string.
    \return true if the data was successfully read to the JSON string, false otherwise.
    */
    static bool readJsonString( TerrainMaterial* terrainMaterial, const agx::String& jsonString );
  };
}
