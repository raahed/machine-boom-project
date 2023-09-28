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

#ifndef AGXUTIL_TRIMESHREADERWRITER_H
#define AGXUTIL_TRIMESHREADERWRITER_H

#include <agx/agx.h>
#include <agxCollide/Trimesh.h>

namespace agxCollide {
  class HeightField;
}

namespace agxUtil {

  /// Functions for generating Trimesh shapes from meshes, and for serializing them.
  namespace TrimeshReaderWriter {

    /**
    Create a Trimesh shape from a supported mesh file format (See agxIO::MeshReader::FileType) containing mesh data.
    \param filename File to create Trimesh from
    \param trimeshOptions Options set to the created Trimesh
    \param vertexRotateAndScale Rotation and scale of Trimesh
    \param vertexTranslate Translation of Trimesh
    \return pointer to new Trimesh shape, nullptr if file could not be loaded
    */
    AGXPHYSICS_EXPORT agxCollide::Trimesh* createTrimesh(
      const agx::String& filename,
      uint32_t trimeshOptions = agxCollide::Trimesh::REMOVE_DUPLICATE_VERTICES,
      const agx::Matrix3x3& vertexRotateAndScale = agx::Matrix3x3(),
      const agx::Vec3& vertexTranslate = agx::Vec3());

    /**
    Create a Trimesh shape from vertices and indices
    \param vertices - The vertices from which we will create a trimesh
    \param indices - The index array that defines the triangles of the triangle mesh
    \param trimeshOptions Options set to the created Trimesh
    \param vertexRotateAndScale Rotation and scale of Trimesh
    \param vertexTranslate Translation of Trimesh
    \return pointer to new Trimesh shape, nullptr if file could not be loaded
    */
    AGXPHYSICS_EXPORT agxCollide::Trimesh* createTrimesh(
      const agx::Vec3Vector& vertices,
      const agx::UInt32Vector& indices,
      uint32_t trimeshOptions = agxCollide::Trimesh::REMOVE_DUPLICATE_VERTICES,
      const agx::Matrix3x3& vertexRotateAndScale = agx::Matrix3x3(),
      const agx::Vec3& vertexTranslate = agx::Vec3());


    /**
    Create a Terrain shape from a supported mesh model file (See agxIO::MeshReader::FileType).
    \note The mesh model must be a non closed mesh with a typical direction of the normals. Meaning that normals should all have a well defined "up" direction.
    Closed mesh models (such as a sphere) will generate in invalid Terrain meshes.
    \param filename File to create Terrain from
    \param bottomMargin Bottom margin of the Terrain Trimesh
    \param trimeshOptions Options set to the created Trimesh
    \param vertexRotateAndScale Rotation and scale of Trimesh
    \param vertexTranslate Translation of Trimesh
    \return pointer to new Terrain shape, nullptr if file could not be loaded
    */
    AGXPHYSICS_EXPORT agxCollide::Trimesh* createTrimeshTerrain(
      const agx::String& filename,
      agx::Real bottomMargin,
      uint32_t trimeshOptions = agxCollide::Trimesh::REMOVE_DUPLICATE_VERTICES,
      const agx::Matrix3x3& vertexRotateAndScale=agx::Matrix3x3(),
      const agx::Vec3& vertexTranslate = agx::Vec3());

    /**
    Creates a trimesh from an agxCollide::HeightField.
    \param heightField HeightField to create Trimesh from
    \return pointer to new Trimesh shape
    */
    AGXPHYSICS_EXPORT agxCollide::Trimesh* createTrimesh(
      const agxCollide::HeightField* heightField);

    /**
    Exports trimesh vertex and index data to a mesh file (currently only supporting wavefront obj).
    This can be useful when creating trimeshes from other sources than 3D data files.
    \param trimesh - The trimesh to serialize.
    \param filename - The desired file name
    \return Was the write operation successful?
    */
    AGXPHYSICS_EXPORT bool exportTrimesh( const agxCollide::Trimesh* trimesh, const agx::String& filename );

    /**
    Create a RenderData object from a supported mesh file (See agxIO::MeshReader::FileType).
    This RenderData object can then be attached to a shape for rendering of the shape.
    \param filename File to create Trimesh from
    \param vertexRotateAndScale Rotation and scale of Trimesh
    \param vertexTranslate Translation of Trimesh
    \return pointer to new agxCollide::RenderData object, nullptr if file could not be loaded
    */
    AGXPHYSICS_EXPORT agxCollide::RenderData* createRenderData(
      const agx::String& filename,
      const agx::Matrix3x3& vertexRotateAndScale=agx::Matrix3x3(),
      const agx::Vec3& vertexTranslate = agx::Vec3());
  }
}

#endif

