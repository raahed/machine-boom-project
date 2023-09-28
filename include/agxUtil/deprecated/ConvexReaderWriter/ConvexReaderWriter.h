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

#include <agxUtil/deprecated/ConvexReaderWriter/ConvexReaderWriter.h>
#include <agxUtil/ConvexReaderWriter/ConvexReaderWriter.h>
#include <agxCollide/deprecated/ConvexBuilder.h>

namespace agxUtil {
  namespace deprecated {

    /// Functions for generating Convex shapes from meshes, and for serializing them.
    namespace ConvexReaderWriter {

      /**
      Creates convex shapes using convex decomposition from a supported mesh file format (see agxIO::MeshReader::FileType),
      using John Ratcliff's algorithm from 2006 for convex decomposition.
      (http://codesuppository.blogspot.com/2006/04/approximate-convex-decomposition.html).
      \note Might take long for medium to larger meshes.

      \deprecated Use agxUtil::ConvexReaderWriter::createConvexDecomposition instead

      \param filename The file name.
      \param results The result vector.
      \param builder A convex builder for more control. Optional.
      \param transformation Scaling and rotation of each vertex.
      \param translation Translation of each vertex (executed after transformation).
      \return Was reading the file and creating the results successful?
      */
      AGXPHYSICS_EXPORT bool createConvexDecomposition(const agx::String& filename,
        agxCollide::ConvexRefVector& results,
        agxCollide::deprecated::ConvexBuilder* builder = nullptr,
        const agx::Matrix3x3& transformation = agx::Matrix3x3(),
        const agx::Vec3& translation = agx::Vec3());

      /**
      Creates convex shapes using convex decomposition from vertices and indices (triangle lists),
      using John Ratcliff's algorithm from 2006 for convex decomposition.
      (http://codesuppository.blogspot.com/2006/04/approximate-convex-decomposition.html).
      \note Might take long for medium to larger meshes.

      \deprecated Use agxUtil::ConvexReaderWriter::createConvexDecomposition instead

      \param vertices The vertices.
      \param indices The indices.
      \param results The result vector.
      \param builder A convex builder for more control. Optional.
      \param transformation Scaling and rotation of each vertex.
      \param translation Translation of each vertex (executed after transformation).
      \return Was reading the file and creating the results successful?
      */
      AGXPHYSICS_EXPORT bool createConvexDecomposition(const agx::Vec3Vector& vertices,
        const agx::UInt32Vector& indices,
        agxCollide::ConvexRefVector& results,
        agxCollide::deprecated::ConvexBuilder* builder = nullptr,
        const agx::Matrix3x3& transformation = agx::Matrix3x3(),
        const agx::Vec3& translation = agx::Vec3());


      /**
      Creates a convex shape from vertices only, building their convex hull.
      Vertices lying within the convex hull will be ignored.

      \deprecated Use agxUtil::ConvexReaderWriter::createConvex instead

      \note Might take long for medium to larger meshes.
      \param vertices The vertices.
      \param transformation Scaling and rotation of each vertex.
      \param translation Translation of each vertex (executed after transformation).
      \return The created Convex. Might be 0 if an error occurred.
      */
      AGXPHYSICS_EXPORT agxCollide::Convex* createConvex(const agx::Vec3Vector& vertices,
        const agx::Matrix3x3& transformation = agx::Matrix3x3(),
        const agx::Vec3& translation = agx::Vec3());
    }
  }
}
