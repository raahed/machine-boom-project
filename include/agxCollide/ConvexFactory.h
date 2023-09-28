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
#include <agx/Referenced.h>
#include <agx/Vector.h>
#include <agxCollide/Convex.h>
#include <agxIO/MeshReader.h>
#include <agx/agx_vector_types.h>
#include <agx/debug.h>

namespace agxCollide
{

  typedef agx::Vector<agxCollide::ConvexRef> ConvexRefVector;

  AGX_DECLARE_POINTER_TYPES(ConvexFactory);
  /**
  Class for creating convex meshes from a set of vertices and indices.
  First the method build() is called. If this was successful (return > 0)
  the created ConvexShapes will be available in a vector (getConvexShapes())

  Parameters must be set before calling the build method
  */
  class AGXPHYSICS_EXPORT ConvexFactory : public agx::Referenced
  {
    public:

      struct Parameters : public agx::Referenced
      {
      };
      typedef agx::ref_ptr< Parameters > ParametersRef;
      typedef agx::ref_ptr< const Parameters > ParametersConstRef;

      struct VHACDParameters : public Parameters
      {
        /// maximum number of voxels generated during the voxelization stage. This will be recalculated as resolution ^3 before it reaches VHACD. Default: 50, range: [2, 400]. 
        uint32_t resolution = 50;
        
        /// maximum concavity. Default 0.0025, range [0.0, 1.0]
        double concavity = 0.0025;
        
        /// Controls the granularity of the search for the "best" clipping plane. Default 4, range: [1, 16]
        uint32_t planeDownsampling = 4;

        /// Controls the precision of the convex-hull generation process during the clipping plane selection stage. Default 4, range: [1, 16]
        uint32_t convexhullDownsampling = 4;

        /// Controls the bias toward clipping along symmetry planes. Default 0.05, range: [0.0, 1.0]
        double alpha = 0.05;
        
        /// Controls the bias toward clipping along revolution axes. Default 0.05, range: [0.0, 1.0]
        double beta = 0.05;

        /// Enable/disable normalizing the mesh before applying the convex decomposition. Default 0, range: [false, true]
        bool pca = false;

        /// 0: voxel-based approximate convex decomposition, 1: tetrahedron-based approximate convex decomposition. Default 0.
        uint32_t mode = 0; // 0: voxel-based (recommended), 1: tetrahedron-based

        // Controls the maximum number of triangles per convex - hull. Default 64, range: [0, 1024]
        uint32_t maxNumVerticesPerConvex = 64;

        /// Controls the adaptive sampling of the generated convex hulls. Default 0.0001, range: [0.0, 0.01]. 
        double minVolumePerConvex = 0.0001;
                
        bool convexhullApproximation = true;
        
        /// Maximum number of convex hulls. Default 1024, range: [1, 1024]
        uint32_t maxConvexHulls = 1024;
      };
      typedef agx::ref_ptr< VHACDParameters > VHACDParametersRef;
      typedef agx::ref_ptr< const VHACDParameters > VHACDParametersConstRef;
     

      /// Constructor
      ConvexFactory( );
  
      /**
      Constructs a number of Convex shapes from a triangle list.
      */
      size_t build( const agx::Vec3Vector& vertices, 
                    const agx::UInt32Vector& indices, 
                    const Parameters* parameters = nullptr,
                    const agx::Matrix3x3& transformation = agx::Matrix3x3(),
                    const agx::Vec3& translation = agx::Vec3()
                    );

      /**
      Constructs a number of Convex shapes from the triangles read with the MeshReader.
      The reader MUST be successfully initialized with data before this call.
      \return number of created Convex meshes, 0 if error occurs.
      */
      size_t build( const agxIO::MeshReader *reader, 
                    const Parameters* parameters = nullptr,
                    const agx::Matrix3x3& transformation = agx::Matrix3x3(),
                    const agx::Vec3& translation = agx::Vec3()
                    );

      /// \return a vector of constructed Convex shapes
      const ConvexRefVector& getConvexShapes() const;

      /// \return the ith shape
      agxCollide::ConvexRef getShape(size_t i);

      /// return num shapes
      size_t getNumShapes() const;

  protected:
    virtual ~ConvexFactory( );

  private:
    
    size_t buildImplementation(const agx::Vec3Vector& vertices,
                               const agx::UInt32Vector& indices, 
                               const VHACDParameters* parameters,
                               const agx::Matrix3x3& transformation = agx::Matrix3x3(),
                               const agx::Vec3& translation = agx::Vec3()
                              );

    agxIO::MeshReaderRef m_reader;

    ConvexRefVector m_convexShapes;

    ParametersConstRef m_parameters;
  };
}



