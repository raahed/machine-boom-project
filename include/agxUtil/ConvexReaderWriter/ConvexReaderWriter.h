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
#include <agx/agx.h>
#include <agxCollide/Convex.h>


namespace agxCollide {
  typedef agx::Vector<agxCollide::ConvexRef> ConvexRefVector;
}

namespace agxUtil {

  /// Functions for generating Convex shapes from meshes, and for serializing them.
  namespace ConvexReaderWriter {

    /**
    Creates a set of convex shapes that approximates the triangle mesh defined by \p filename
    using V-HACD which uses a volumetric algorithm.
    The computation time can be high, if a large voxel resolution is used.

    \param filename Name of mesh file.
    \param results The result vector.
    \param elementResolutionPerAxis Resolution parameter. Range 20-400. Larger values take more time.
    \param transformation Scaling and rotation of each vertex.
    \param translation Translation of each vertex (executed after transformation).
    \return True if convex decomposition was successful and generated output
    */
    AGXPHYSICS_EXPORT bool createConvexDecomposition( const agx::String& filename,
                                                      agxCollide::ConvexRefVector& results,
                                                      size_t elementResolutionPerAxis = 50,
                                                      const agx::Matrix3x3& transformation = agx::Matrix3x3(),
                                                      const agx::Vec3& translation = agx::Vec3() );

    /**
    Creates a set of convex shapes that approximates the triangle mesh defined by \p filename
    using V-HACD which uses a volumetric algorithm.
    The computation time can be high, if a large voxel resolution is used.

    \param vertices The vertices.
    \param indices The indices.
    \param results The result vector.
    \param elementResolutionPerAxis Resolution parameter. Range 20-400. Larger values take more time.
    \param transformation Scaling and rotation of each vertex.
    \param translation Translation of each vertex (executed after transformation).
    \return True if convex decomposition was successful and generated output
    */
    AGXPHYSICS_EXPORT bool createConvexDecomposition( const agx::Vec3Vector& vertices,
                                                      const agx::UInt32Vector& indices,
                                                      agxCollide::ConvexRefVector& results,
                                                      size_t elementResolutionPerAxis = 50,
                                                      const agx::Matrix3x3& transformation = agx::Matrix3x3(),
                                                      const agx::Vec3& translation = agx::Vec3() );

    
    /**
    Creates a convex shape from vertices only, building their convex hull.
    Vertices lying within the convex hull will be ignored.

    \note The algorithm is using float precision. Using vertices with large coordinate values
          might result in an inexact convex hull.

    \param vertices The vertices.
    \param transformation Scaling and rotation of each vertex.
    \param translation Translation of each vertex (executed after transformation).
    \return The created Convex. Might be 0 if an error occurred.
    */
    AGXPHYSICS_EXPORT agxCollide::Convex* createConvex( const agx::Vec3Vector& vertices,
                                                        const agx::Matrix3x3& transformation = agx::Matrix3x3(),
                                                        const agx::Vec3& translation = agx::Vec3() );

    /**
    Creates a convex shape from a trimesh file. The mesh in the file can be concave, the create
    convex will encapsulate the whole triangle mesh into the hull.

    \param filename - The Mesh filename
    \param transformation Scaling and rotation of each vertex.
    \param translation Translation of each vertex (executed after transformation).
    \return The created Convex. Might be 0 if an error occurred.
    */
    AGXPHYSICS_EXPORT agxCollide::Convex* createConvex(const agx::String& filename,
                                                       const agx::Matrix3x3& transformation = agx::Matrix3x3(),
                                                       const agx::Vec3& translation = agx::Vec3());

    /**
    Creates a set of convex shapes that approximates the triangle mesh defined by \p filename
    using V-HACD which uses a volumetric algorithm.
    The computation time can be high, if a large voxel resolution is used.

    \param filename Name of mesh file.
    \param results The result vector.
    \param elementResolutionPerAxis Resolution parameter. Will be clamped between 20-400. Larger values take more time.
    \param transformation Scaling and rotation of each vertex.
    \param translation Translation of each vertex (executed after transformation).
    \return True if convex decomposition was successful and generated output
    */
    AGXPHYSICS_EXPORT bool createVHACDConvexDecomposition( const agx::String& filename,
                                                           agxCollide::ConvexRefVector& results,
                                                           size_t elementResolutionPerAxis = 50,
                                                           const agx::Matrix3x3& transformation = agx::Matrix3x3(),
                                                           const agx::Vec3& translation = agx::Vec3() );

    /**
    Creates a set of convex shapes that approximates the triangle mesh defined by \p vertices and \p indices
    using V-HACD which uses a volumetric algorithm. 
    The computation time can be high, if a large voxel resolution is used.

    \param vertices The triangle mesh vertices.
    \param indices The triangle mesh indices.
    \param results The generated convex meshes.
    \param elementResolutionPerAxis Resolution parameter. Range 20-400. Larger values take more time.
    \param transformation Scaling and rotation of each vertex.
    \param translation Translation of each vertex (executed after transformation).
    \return True if convex decomposition was successful and generated output
    */
    AGXPHYSICS_EXPORT bool createVHACDConvexDecomposition( const agx::Vec3Vector& vertices,
                                                           const agx::UInt32Vector& indices,
                                                           agxCollide::ConvexRefVector& results,
                                                           size_t elementResolutionPerAxis = 50,
                                                           const agx::Matrix3x3& transformation = agx::Matrix3x3(),
                                                           const agx::Vec3& translation = agx::Vec3() );

    /**
    Creates a convex from vertices which are known to be convex.
    \note No convex decomposition will be done.
    \param vertices The triangle mesh vertices.
    \param indices The triangle mesh indices.
    \param sourceName The name of the data source.
    \param transformation Scaling and rotation of each vertex.
    \param translation Translation of each vertex (executed after transformation).
    \retval The convex.
    */
    AGXPHYSICS_EXPORT agxCollide::Convex* createFromConvex( const agx::Vec3Vector& vertices,
                                                            const agx::UInt32Vector& indices,
                                                            const agx::String& sourceName = "convexFromReaderWriter",
                                                            const agx::Matrix3x3& transformation = agx::Matrix3x3(),
                                                            const agx::Vec3& translation = agx::Vec3() );


    /**
    Creates a convex from a supported mesh-file (see agxIO::MeshReader::FileType) which is known to be convex.
    \note No convex decomposition will be done, also no Convex generation will be done if the mesh is not convex \see createConvex
    \param filename The file name.
    \param transformation Scaling and rotation of each vertex.
    \param translation Translation of each vertex (executed after transformation).
    \return The convex.
    */
    AGXPHYSICS_EXPORT agxCollide::Convex* createFromConvex( const agx::String& filename,
                                                            const agx::Matrix3x3& transformation = agx::Matrix3x3(),
                                                            const agx::Vec3& translation = agx::Vec3() );

    /**
    Exports convex vertex and index data to a wavefront obj file on a storage device.
    This can be useful when creating convex meshes from other sources than 3D data files.
    \param convex The convex to serialize.
    \param filename The desired file name (one would probably want it to end on .obj)
    \return Was the write operation successful?
    */
    AGXPHYSICS_EXPORT bool exportConvex( const agxCollide::Convex& convex, const agx::String& filename );
  }
}


