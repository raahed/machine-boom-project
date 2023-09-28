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

#include <agx/config/AGX_USE_AGXTERRAIN.h>
#include <agxTerrain/export.h>
#include <agxTerrain/TerrainMaterial.h>

#define DEFAULT_TERRAIN_MATERIAL_LIBRARY_FOLDER "MaterialLibrary/TerrainMaterials/"

namespace agxTerrain
{
  /**
  Class that is used for interfacing with different TerrainMaterial presets with calibrated
  bulk and contact properties of different archetypes of soil, such as dirt, gravel and
  sand. It parses the available preset files that exists in a specified direction, validates them,
  and provides easy interfacing for constructing terrain materials.
  The default location if a folder containing the default material preset files of AGX Dynamics.
  The files contain TerrainMaterial specifications in JSON format. The user is not limited
  to using these preset files but can also load custom material files through the Terrain class.
  */
  class AGXTERRAIN_EXPORT TerrainMaterialLibrary
  {
  public:

    /**
    Configure a agxTerrain::TerrainMaterial object from one of the existing library
    presets, derived from the .json files in the specified library folder. The default location
    of the preset folder contains the default material preset of AGX Dynamics.
    Note - Existing library materials can be extracted via the getAvailableLibraryMaterials() method:
    \param materialName - the name of the material preset to load.
    \param terrainMaterial - Pointer to the terrain material that should be modified
    \param librayFolder - The path to the folder containing all the terrain material preset files.
    \return a TerrainMaterial object created from the specified material preset or nullptr if the creation failed.
    */
    static bool loadMaterialProfile( const agx::String& materialName,
                                     agxTerrain::TerrainMaterial* terrainMaterial,
                                     const agx::String& librayFolder = DEFAULT_TERRAIN_MATERIAL_LIBRARY_FOLDER );

    /**
    Get the available TerrainMaterial presets from the existing.json files in the specified
    material library folder. The default location of the preset folder contains the default
    material preset of AGX Dynamics.
    \param librayFolder - The path to the folder containing all the terrain material preset files.
    \return a vector containing the available TerrainMaterial presets from the existing
            .json files in the specified material library folder.
    */
    static agx::StringVector getAvailableLibraryMaterials( const agx::String& librayFolder = DEFAULT_TERRAIN_MATERIAL_LIBRARY_FOLDER );
  };
}


