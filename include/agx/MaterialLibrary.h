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
#include <agx/Material.h>

namespace agx
{

  /**
  Class with functionality to view items in the Material Library and load Materials.
  */
  class AGXPHYSICS_EXPORT MaterialLibrary {
    public:

      /**
      Enum describing different item types in the MaterialLibrary
      */
      enum LibraryItemType {
        MATERIAL,
        CONTACT_MATERIAL,
        TERRAIN_MATERIAL,
        CABLE_PROPERTIES,
        BEAMMODEL_PROPERTIES
      };



      /**
      Load a Material from the MaterialLibrary.
      \return A ref pointer to a Material. The ref-pointer is invalid if a Material with the given name could not be found
      */
      static agx::MaterialRef loadMaterial( agx::String name );


      /**
      Return the name for a LibraryItemType.
      */
      static agx::String getItemTypeName( LibraryItemType itemType );


      /**
      Return the relative path for a named library item of specified type
      */
      static agx::String getFilePath( LibraryItemType itemType, agx::String name );


      /**
      Returns a vector with all the names for requested item type
      */
      static agx::StringVector getAvailableItems(LibraryItemType itemType);
  };



}
