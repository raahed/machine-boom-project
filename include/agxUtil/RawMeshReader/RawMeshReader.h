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

#ifndef AGXUTIL_RAWMESHREADER_H
#define AGXUTIL_RAWMESHREADER_H

#include <agx/agx.h>
#include <agxUtil/RawMesh.h>
#include <agx/Matrix3x3.h>

namespace agxUtil {

 /// Functions for generating Trimesh shapes from meshes, and for serializing them.
 namespace RawMeshReader {

  /**
  Create a mesh shape from a file containing mesh data (only obj is supported formats today).
  \param filename File to create mesh from
  \param vertexRotateAndScale Rotation and scale of mesh
  \param vertexTranslate Translation of mesh
  \return pointer to new mesh shape, nullptr if file could not be loaded
  */
  AGXPHYSICS_EXPORT agxUtil::RawMeshRefVector createRawMeshFromFile(
   const agx::String& filename,
   const agx::Matrix3x3& vertexRotateAndScale = agx::Matrix3x3(),
   const agx::Vec3& vertexTranslate = agx::Vec3());
 }
}

#endif
