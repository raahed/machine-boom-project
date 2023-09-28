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

#ifndef AGXCOLLIDE_MESH_H
#define AGXCOLLIDE_MESH_H

#include <agx/agxPhysics_export.h>
#include <agx/agx_vector_types.h>

#include <agxData/LocalVector.h>

#include <external/GIMPACT/AabbTree.h>

#include <agxCollide/agxCollide.h>
#include <agxCollide/MeshData.h>
#include <agxCollide/Shape.h>

#include <agxStream/Serializable.h>


namespace agxCollide
{
  AGX_DECLARE_POINTER_TYPES(Mesh);

  /**
  * Mesh is a common base class for triangle meshes, such as Mesh or HeightField.
  * It is a purely abstract class and should not be used by itself.
  *
  * All triangles are stored as triangle lists in counter clockwise winding.
  * The raw data is stored in the CollisionMeshData class as arrays.
  * For each triangle, edge i points from vertex i to vertex (i + 1) modulo 3,
  * 0 <= i < 3 and are thus stored in the vertex indices (CollisionMeshData::indices).
  *
  * There are three ways of accessing the mesh data:
  * - Either directly via getMeshData()
  * - For the data not related to mesh traversal,
  *   there exist also dedicated getter functions.
  * - For easier use of mesh traversal (given a half-edge structure),
  *   there is a Triangle class (including a small overhead).
  * - For easier use of mesh traversal (given a half-edge structure),
  *   there is a VertexEdgeCirculator and a FaceEdgeCirculator class
  *   (including a small overhead).
  *
  * Meshes such as Trimeshes or HeightFields can have a
  * half edge structure which can be used for traversing the mesh.
  * There, each edge has a half edge partner,
  * which is the corresponding edge in the neighboring triangle.
  * At holes (or at borders of a HeightField), this neighbor triangle will be
  * invalid.
  *
  * Due to the internal representation, errors can occur if the
  * number of triangles exceeds the maximum of integers / 3 - 1,
  * (for 32 bit: 1431655764). Therefore, input data with larger numbers of
  * triangles will be cut off at this point.
  */
  class AGXPHYSICS_EXPORT Mesh : public Shape
  {

    public:
      /**
      * Class for more intuitive access to the Mesh's mesh data.
      * Involves a small overhead compared to using the more low-level access
      * to the raw data by using getMeshData().
      */
      class AGXPHYSICS_EXPORT Triangle
      {
          friend class Mesh;
        public:
          /**
          * \param localVertexIndex The vertex's index within the triangle (0 to 2).
          * \return Specified vertex
          */
          agx::Vec3 getVertex( uint_fast8_t localVertexIndex ) const;

          /**
          * This function returns the triangle normal (as stored in the collision mesh data).
          * \return triangle normal.
          */
          agx::Vec3 getNormal() const;

          /**
          * \return True if the Triangle is valid (its m_triangleIndex is positive
          *         and smaller than the Mesh's number of Triangles).
          *         Triangles can become invalid when created from invalid
          *         Meshes or from half edge functions when no half edge
          *         structure is available (e.g. at terrain borders).
          */
          bool isValid() const;

          /**
          * Returns the global index of the vertex.
          * \param localVertexIndex The vertex's index within the triangle (0 to 2).
          * \return The global vertex index of the specified triangle vertex.
          */
          size_t getGlobalVertexIndex( uint_fast8_t localVertexIndex ) const;

          /**
          * \param localEdgeIndex The edge's index within the triangle (0 to 2).
          * \return The starting vertex of the specified edge.
          */
          agx::Vec3 getEdgeStartVertex( uint_fast8_t localEdgeIndex ) const;

          /**
          * \param localEdgeIndex The edge's index within the triangle (0 to 2).
          * \return The ending vertex of the specified edge.
          */
          agx::Vec3 getEdgeEndVertex( uint_fast8_t localEdgeIndex ) const;

          /// \return The length of the longest edge in the triangle.
          agx::Real getMaximumEdgeLength() const;

          /**
          * \param localEdgeIndex: The edge's index within the triangle (0 to 2).
          * \return Returns if the triangle is valid and has a valid half edge partner.
          */
          bool hasHalfEdgePartner( uint_fast8_t localEdgeIndex ) const;

          /**
          * \param localEdgeIndex: The edge's index within the triangle (0 to 2).
          * \return Returns the half edge partner of this triangle across an edge.
          *         If no half edge information for this edge exists, the Triangle
          *         will be invalid.
          *         This should be tested for before the result is used.
          *
          */
          const Triangle getHalfEdgePartner( uint_fast8_t localEdgeIndex ) const;

          /**
          * \param localEdgeIndex: The edge's index within the triangle (0 to 2).
          * \return The global edge index of the specified Triangle edge's
          *         half edge partner. The global edge index is defined as
          *         3 * global triangle index + local edge index.
          *
          *         If no half edge information can be found, returns a value which
          *           is
          *         This should be tested for before the result is used.
          */
          size_t getHalfEdgePartnerGlobalEdgeIndex( uint_fast8_t localEdgeIndex ) const;

          /**
          * \param localEdgeIndex: The edge's index within the triangle (0 to 2).
          * \return The local edge index of the edge within the specified Triangle
          *         edge's half edge partner corresponding to the specified edge.
          *         If no half edge information for the local edge exists,
          *         will be invalid.
          *         If no half edge information can be found, returns a value > 2.
          *         This should be tested for before the result is used.
          */
          size_t getHalfEdgePartnerLocalEdgeIndex( uint_fast8_t localEdgeIndex ) const;

          /**
          * \return Triangle's triangle index. Can be numTriangles
          *        (= getHalfEdgeNeighborInvalidTriangle()) if invalid.
          */
          size_t getTriangleIndex() const;

          /// \return A pointer to the Mesh this triangle is based on.
          const Mesh* getMesh() const;


        protected:
          /**
          * This constructor is hidden and should only be called from the Mesh
          * class.
          * \param mesh The Mesh class which the Triangle belongs to.
          * \param triangleIndex The Triangle's index in the Mesh.
          */
          Triangle( const Mesh* mesh, size_t triangleIndex );

          /**
          * This constructor is hidden and should only be called from the Mesh
          * class and its children.
          */
          Triangle();

        protected:
          MeshConstRef m_mesh;
          size_t m_triangleIndex;

      };


      /**
      * Class in order to circulate over the edges connected to a Triangle's
      * vertex.
      * Involves a small overhead compared to using the more low-level access
      * to the raw data by using getMeshData().
      */
      class AGXPHYSICS_EXPORT VertexEdgeCirculator
      {
          friend class Mesh;

        public:
          /**
          * Move circulator one step clockwise.
          * Moves to next triangle bordering to the vertex and
          *  this triangle's edge starting from the vertex. If no half-edge
          *  information is available for the bordering edge (a hole), the circulator
          *  will go counterclockwise backwards around the vertex until a hole
          * is found there. */
          void operator++(int);

          /**
          * Move circulator one step clockwise.
          * Moves to next triangle bordering to the vertex and
          *  this triangle's edge starting from the vertex. If no half-edge
          *  information is available for the bordering edge (a hole), the circulator
          *  will go counterclockwise backwards around the vertex until a hole
          * is found there. */
          VertexEdgeCirculator& operator++ ();

          /**
          * \return True if the circulator is valid.
          * Circulators become invalid if constructed from or assigned with invalid data.
          */
          bool isValid() const;

          /**
          * \return True if circulator is valid, at start position and has been increased.
          * Watch out in while loops: will return false for non-valid circulators.
          */
          bool atEnd() const;

          /// \return Starting vertex of the edge the circulator is pointing to.
          agx::Vec3 getEdgeStartVertex() const;

          ///  \return Ending vertex of the edge the circulator is pointing to.
          agx::Vec3 getEdgeEndVertex() const;

          /// \return The triangle the circulator is pointing to.
          inline const Triangle getTriangle() const;

          /// \return The local edge index of the edge the circulator is pointing to.
          uint_fast8_t getLocalEdgeIndex() const;

          /// \return Has the iterator just jumped over a hole (and is on its other side).
          bool hasJustJumpedOverHole() const;

          /// \return Global index of triangle the circulator is pointing at right now.
          size_t getTriangleIndex() const;

          /// \return A pointer to the Mesh this triangle is based on.
          const Mesh* getMesh() const;

          /// A newly created and unassigned VertexEdgeCirculator will be invalid.
          VertexEdgeCirculator();

        protected:
          /**
          * This hidden constructor can only be called from the Mesh class.
          * \param mesh -  Pointer to a mesh
          * \param triangleIndex Within the meshData, the index of the Triangle
          *                      the circulator points to when constructed.
          * \param localVertexIndex The local index of the Triangle's vertex that
          *                      the circulator points to when constructed.
          */
          VertexEdgeCirculator( const Mesh* mesh,
                                size_t triangleIndex, uint_fast8_t localVertexIndex );

        protected:
          MeshConstRef m_mesh;
          size_t m_triangleStartIndex;
          size_t m_triangleIndex;
          uint_fast8_t m_localEdgeStartIndex;
          uint_fast8_t m_localEdgeIndex;
          bool m_hasMoved;
          bool m_hasJustJumpedOverHole;
          bool m_valid;
      };

      /**
      * Class in order to circulate over the edges connected to a Triangle's
      * face.
      * Involves a small overhead compared to using the more low-level access
      * to the raw data by using getMeshData().
      */
      class AGXPHYSICS_EXPORT FaceEdgeCirculator
      {
          friend class Mesh;

        public:
          /**
          * Move circulator one step counter clockwise around the face.
          */
          void operator++(int);

          /**
          * Move circulator one step counter clockwise around the face.
          */
          FaceEdgeCirculator& operator++();

          /**
          * \return True if the circulator is valid.
          * Circulators become invalid if constructed from or assigned with
          * on invalid data.
          */
          bool isValid() const;

          /**
          * \return True if circulator is valid, at start position and has been increased.
          * Watch out in while loops: Also test for isValid() to avoid endless loops.
          */
          bool atEnd() const;

          /// \return Starting vertex of the edge the circulator is pointing to.
          agx::Vec3 getEdgeStartVertex() const;

          ///  \return Ending vertex of the edge the circulator is pointing to.
          agx::Vec3 getEdgeEndVertex() const;

          /// \return The triangle the circulator is pointing to.
          const Triangle getTriangle() const;

          /// \return The local edge index of the edge the circulator is pointing to.
          uint_fast8_t getLocalEdgeIndex() const;

          /// \return A pointer to the Mesh this circulator is based on.
          const Mesh* getMesh() const;

          /// A newly created and unassigned FaceEdgeCirculator will be invalid.
          FaceEdgeCirculator();

        protected:
          /**
          * This hidden constructor can only be called from the Mesh class.
          * \param mesh - Pointer to the mesh
          * \param triangleIndex Within the meshData, the index of the Triangle
          *                      the circulator points to when constructed.
          */
          FaceEdgeCirculator( const Mesh* mesh, size_t triangleIndex );

        protected:
          MeshConstRef m_mesh;
          size_t m_triangleIndex;
          uint_fast8_t m_localEdgeIndex;
          bool m_hasMoved;
          bool m_valid;
      };


      // Methods of the class Mesh.
    public:

      /// \return The number of vertices in the mesh data.
      size_t getNumVertices() const;

      /**
      * \param globalVertexIndex Index of the vertex in the MeshData's vertices array.
      * \return Vertex specified by globalVertexIndex
      */
      agx::Vec3 getVertex( size_t globalVertexIndex ) const;

      /// \return The number of Triangles in the mesh data.
      size_t getNumTriangles() const;

      /**
      * Returns one of the three vertices in a triangle.
      * \param triangleIndex Index of the triangle the vertex belongs to.
      * \param localVertexIndex The vertex's index within the triangle (0 to 2).
      * \return The vertex specified by the parameters.
      */
      agx::Vec3 getTriangleVertex( size_t triangleIndex, uint_fast8_t localVertexIndex ) const;

      /**
      * \param triangleIndex The index of the Triangle.
      * \return The normal of the Triangle.
      */
      agx::Vec3 getTriangleNormal( size_t triangleIndex ) const;

      /**
      * Returns the global index of one of the three vertices in a triangle.
      * \param triangleIndex Index of the triangle the vertex belongs to.
      * \param localVertexIndex The vertex's index within the triangle (0 to 2).
      * \return The global index of the vertex specified by the parameters.
      */
      size_t getGlobalVertexIndex( size_t triangleIndex, uint_fast8_t localVertexIndex ) const;

      /**
      * \param triangleIndex The index of the Triangle.
      * \return The length of the longest edge in the Triangle.
      */
      agx::Real getTriangleMaximumEdgeLength( size_t triangleIndex ) const;

      /// \return The maximum length of all triangle edges.
      agx::Real getTotalMaximumEdgeLength() const;

      /// \return The global edge index of an edge's half edge partner
      uint32_t getHalfEdgePartnerEdge( size_t globalEdgeIndex ) const;

      /**
      * \return True if the Mesh is valid and can be used for collision
      *         detection and other purposes; false otherwise.
      */
      bool isValid() const;

      /**
      * \return True if the Mesh is valid and can be used for collision
      *         detection and other purposes, and if it is a Manifold;
      *         false otherwise.
      */
      bool isValidAndClosed() const;

      /// \return The center of the Mesh. Inherited from class Shape.
      virtual agx::Vec3 getCenter() const override;

      /// \return  The volume of the Mesh. Inherited from class Shape.
      virtual agx::Real getVolume() const override;

      ///Returns the inertia of the shape, scaled with the mass. Inherited from class Shape.
      virtual agx::SPDMatrix3x3 calculateInertia( agx::Real mass ) const override;

      /// \return True if Mesh is terrain (it will not be a Manifold/closed then).
      bool isTerrain() const;

      /// \return True if the center point of the mesh is inside the mesh, eg false for torus
      bool hasInternalCenterPoint() const;

      /**
      * \return Bottom safety margin.  This is the depth that is added below
      *         the lowest point in the terrain in local z direction, to prevent
      *         tunneling in this direction.
      */
      agx::Real getBottomMargin() const;

      /**
      * Sets the depth that is added below
      *         the lowest point in the terrain in local z direction, to prevent
      *         tunneling in this direction.
      *         Will also update the bounding volume.
      * \param newMargin The desired depth. Negative values will be ignored.
      */
      void setBottomMargin(agx::Real newMargin);

      /**
      * \param globalEdgeIndex The global index to examine.
      * \return Is the given edge index a valid edge index within this Mesh?
      */
      bool isValidEdgeIndex(size_t globalEdgeIndex ) const;

      /**
      * \param globalTriangleIndex The global index to examine.
      * \return Is the given triangle index a valid edge index within this Mesh?
      */
      bool isValidTriangleIndex(size_t globalTriangleIndex ) const;

      /**
      * \param triangleIndex The index of the triangle.
      * \return Triangle
      */
      inline const Triangle getTriangle( size_t triangleIndex ) const;

      /**
      * \param globalEdgeIndex The global edge index.
      * \return The local edge index which the edge (given as a global index) corresponds to.
      */
      static size_t getLocalEdgeIndex( size_t globalEdgeIndex );

      /**
      * \param globalEdgeIndex The global edge index.
      * \return The global Triangle index which the edge (given as a global index) corresponds to.
      */
      static size_t getGlobalTriangleIndex( size_t globalEdgeIndex );

      /**
      * Computes the global edge index from a given triangle index and
      * the local edge index within this triangle.
      * \param triangleIndex The global triangle index.
      * \param localEdgeIndex The local edge index within this triangle.
      * \return The global edge index.
      */
      static size_t getGlobalEdgeIndex ( size_t triangleIndex, uint_fast8_t localEdgeIndex );

      /**
      * Calculates an array of triangle indices of the triangles that are within a
      * bounding volume.
      * \param volume The bounding box.
      * \param meshToWorld A transformation from the mesh's local frame to world coordinates.
      * \param volumeToWorld A transformation from the volume's local frame to world coordinates.
      * \param triangleIndices A vector of triangle indices. It will be emptied before writing to it.
      */

      void calculateTrianglesInVolume(
        const BoundingAABB& volume, const agx::AffineMatrix4x4& meshToWorld, const agx::AffineMatrix4x4& volumeToWorld,
        agx::UInt32Vector& triangleIndices ) const;

      void calculateTrianglesInVolume(
        const BoundingAABB& volume, const agx::AffineMatrix4x4& meshToWorld, const agx::AffineMatrix4x4& volumeToWorld,
        agxData::LocalVector<agx::UInt32>& triangleIndices ) const;


      /// Struct used in calculateTriangleOverlapsAlongLineSegment.
      struct TriangleLineSegmentResult
      {
        TriangleLineSegmentResult(agx::UInt32 newTriangleIndex,
          agx::Real newLineParameter, bool newIsFrontFace) :
        lineParameter(newLineParameter), triangleIndex(newTriangleIndex),
          isFrontFace(newIsFrontFace)
        {}
        agx::Real lineParameter; // Parameter along line, between 0 and 1.
        agx::UInt32 triangleIndex; // The triangle index.
        bool isFrontFace; // Is triangle hit from front, or back?
      };

      typedef agx::Vector<TriangleLineSegmentResult> TriangleLineSegmentResultVector;

      /**
      Calculates all triangle overlaps of a line segment with the mesh.
      Tries to fix some corner cases such as:
      - A line segment tangent to a mesh edge should get 2 contacts (one for each triangle).
      - A line segment intersecting through a mesh edge should get 1 contact (for one of the triangles).
      - Vertices will be treated as several edges (with 1-2 contacts per edge, see above).
      \param segmentStartLocal The segment starting point in mesh coordinates.
      \param segmentEndLocal The segment ending point in mesh coordinates.
      \param triangleResults - Vector of resulting overlapped triangles
      */
      void calculateTriangleOverlapsAlongLineSegment(
        const agx::Vec3& segmentStartLocal,
        const agx::Vec3& segmentEndLocal,
        TriangleLineSegmentResultVector& triangleResults) const;

      /**
      Calculates an array of triangle indices of the triangles whose bounding volumes
      get intersected by a line segment
      \param segmentStartLocal The segment starting point in mesh coordinates.
      \param segmentEndLocal The segment ending point in mesh coordinates.
      \param triangleIndices A vector of triangle indices. It will be emptied before being written to.
      */
      virtual void calculatePossibleTriangleOverlapsAlongLineSegment(
        const agx::Vec3& segmentStartLocal,
        const agx::Vec3& segmentEndLocal,
        agx::UInt32Vector& triangleIndices) const = 0;


      /**
      Calculates an array of triangle indices of the triangles whose bounding volumes
      get intersected by a line segment
      \param segmentStartLocal The segment starting point in mesh coordinates.
      \param segmentEndLocal The segment ending point in mesh coordinates.
      \param triangleIndices A vector of triangle indices. It will be emptied before being written to.
      */
      virtual void calculatePossibleTriangleOverlapsAlongLineSegment(
        const agx::Vec3& segmentStartLocal,
        const agx::Vec3& segmentEndLocal,
        agxData::LocalVector<agx::UInt32>& triangleIndices) const = 0;

      /**
      Creates an VertexEdgeCirculator pointing to this Triangle and
      a Voronoi region.
      \param triangleIndex - Index of the triangle
      \param localVertexIndex: The local index of the vertex that the
             VertexEdgeCirculator should point to.
      \return The created VertexEdgeCirculator
      */
      VertexEdgeCirculator createVertexEdgeCirculator(
        size_t triangleIndex,
        uint_fast8_t localVertexIndex ) const;

      /**
      * Creates an FaceEdgeCirculator pointing to this Triangle and
      * a Voronoi region.
      * \return The created FaceEdgeCirculator
      */
      FaceEdgeCirculator createFaceEdgeCirculator(
        size_t triangleIndex ) const;

      /**
      Is an edge convex?
      Returns always true if half edge information is missing
      on this edge/the whole mesh.
      Exactly straight edges will also be considered convex.
      \param globalEdgeIndex
      \retval Is the edge convex?
      */
      bool isConvexEdge(size_t globalEdgeIndex) const;

      /**
      Is an vertex convex?
      Returns always true if half edge information is missing
      on this vertex/the whole mesh.
      Exactly straight edges will also be considered convex.
      Saddle points will return false.
      When encountering a hole, the edges until the whole
      will be considered.
      \param triangleIndex The global index of the triangle
      \param localVertexIndex The local index of the vertex in the triangle
      \retval Is the edge convex?
      */
      bool isConvexVertex(size_t triangleIndex, size_t localVertexIndex) const;

      /// \return The CollisionMeshData containing all the mesh information.
      const CollisionMeshData* getMeshData() const;

      /// \return True if the Mesh has a half edge structure.
      bool hasHalfEdge() const;

      /// \return The Trimesh's bounding volume hierarchy.
      const AabbTree* getBvhTree() const;

      virtual const BoundingAABB& updateBoundingVolume() override;
      virtual BoundingAABB calculateLocalBound() const override;
      const BoundingAABB& getLocalBound() const;

      // AGXSTREAM_DECLARE_SERIALIZABLE( agxCollide::Mesh );

    protected:
      /// Hidden, should only be called by child classes.
      Mesh (Shape::Type type, agx::Physics::Geometry::ShapePtr entity);

      /// Hide destructor.
      virtual ~Mesh();

      void store( class agxStream::OutputArchive& out ) const override;
      void restore( class agxStream::InputArchive& in ) override;

      // Update the bounding volume hierarchy.
      virtual void updateBvhTree() = 0;


    protected:
      CollisionMeshDataRef m_collisionMeshData;
      AabbTreeRef m_aabbTree;
      BoundingAABB m_localBound;
  };


  /// Implementations

  /// Triangle class
  AGX_FORCE_INLINE size_t Mesh::Triangle::getTriangleIndex() const
  {
    return m_triangleIndex;
  }


  AGX_FORCE_INLINE size_t Mesh::Triangle::getGlobalVertexIndex( uint_fast8_t localVertexIndex ) const
  {
    return m_mesh->getGlobalVertexIndex( m_triangleIndex, localVertexIndex );
  }


  AGX_FORCE_INLINE bool Mesh::Triangle::isValid() const
  {
    return m_mesh->isValid() && m_mesh->isValidTriangleIndex( m_triangleIndex );
  }


  AGX_FORCE_INLINE agx::Vec3 Mesh::Triangle::getNormal() const
  {
    return m_mesh->getTriangleNormal( m_triangleIndex );
  }


  AGX_FORCE_INLINE agx::Vec3 Mesh::Triangle::getVertex( uint_fast8_t localVertexIndex ) const
  {
    return m_mesh->getTriangleVertex( m_triangleIndex, localVertexIndex );
  }


  AGX_FORCE_INLINE agx::Vec3 Mesh::Triangle::getEdgeStartVertex( uint_fast8_t localEdgeIndex ) const
  {
    return m_mesh->getTriangleVertex( m_triangleIndex, localEdgeIndex );
  }


  AGX_FORCE_INLINE agx::Vec3 Mesh::Triangle::getEdgeEndVertex( uint_fast8_t localEdgeIndex ) const
  {
    return m_mesh->getTriangleVertex( m_triangleIndex, (uint_fast8_t)((localEdgeIndex + (uint_fast8_t)1) % (uint_fast8_t)3) );
  }


  AGX_FORCE_INLINE agx::Real Mesh::Triangle::getMaximumEdgeLength() const
  {
    return m_mesh->getTriangleMaximumEdgeLength( m_triangleIndex );
  }


  inline size_t Mesh::Triangle::getHalfEdgePartnerGlobalEdgeIndex( uint_fast8_t localEdgeIndex ) const
  {
    return m_mesh->getHalfEdgePartnerEdge( Mesh::getGlobalEdgeIndex( m_triangleIndex, localEdgeIndex ) );
  }


  inline const Mesh::Triangle Mesh::Triangle::getHalfEdgePartner( uint_fast8_t localEdgeIndex ) const
  {
    agxAssert( localEdgeIndex < 3 );
    return Triangle( m_mesh, this->getHalfEdgePartnerGlobalEdgeIndex( localEdgeIndex ) / 3 ) ;
  }


  AGX_FORCE_INLINE size_t Mesh::Triangle::getHalfEdgePartnerLocalEdgeIndex( uint_fast8_t localEdgeIndex ) const
  {
    agxAssert( localEdgeIndex < 3 );
    return this->getHalfEdgePartnerGlobalEdgeIndex( localEdgeIndex ) % 3;
  }



  AGX_FORCE_INLINE bool Mesh::Triangle::hasHalfEdgePartner( uint_fast8_t localEdgeIndex )
  const
  {
    return m_mesh->isValidEdgeIndex( this->getHalfEdgePartnerGlobalEdgeIndex( localEdgeIndex ) );
  }


  AGX_FORCE_INLINE const Mesh* Mesh::Triangle::getMesh() const
  {
    return m_mesh.get();
  }


  /// Mesh class
  AGX_FORCE_INLINE size_t Mesh::getGlobalEdgeIndex ( size_t triangleIndex, uint_fast8_t localEdgeIndex )
  {
    return 3 * triangleIndex + localEdgeIndex;
  }


  inline const Mesh::Triangle Mesh::getTriangle( size_t triangleIndex ) const
  {
    return Mesh::Triangle( this, triangleIndex );
  }


  AGX_FORCE_INLINE size_t Mesh::getLocalEdgeIndex( size_t globalEdgeIndex )
  {
    return globalEdgeIndex % 3;
  }


  AGX_FORCE_INLINE size_t Mesh::getGlobalTriangleIndex( size_t globalEdgeIndex )
  {
    return globalEdgeIndex / 3;
  }


  AGX_FORCE_INLINE size_t Mesh::getNumVertices() const
  {
    return m_collisionMeshData->m_vertices.size();
  }


  AGX_FORCE_INLINE agx::Vec3 Mesh::getVertex( size_t globalVertexIndex ) const
  {
    return m_collisionMeshData->m_vertices[ globalVertexIndex ];
  }


  AGX_FORCE_INLINE size_t Mesh::getNumTriangles() const
  {
    return m_collisionMeshData->m_numTriangles;
  }


  AGX_FORCE_INLINE agx::Vec3 Mesh::getTriangleVertex( size_t triangleIndex, uint_fast8_t localVertexIndex ) const
  {
    return m_collisionMeshData->m_vertices[m_collisionMeshData->m_indices[3 * triangleIndex + localVertexIndex]];
  }


  AGX_FORCE_INLINE agx::Vec3 Mesh::getTriangleNormal( size_t triangleIndex ) const
  {
    return m_collisionMeshData->m_normals[triangleIndex];
  }


  AGX_FORCE_INLINE size_t Mesh::getGlobalVertexIndex( size_t triangleIndex, uint_fast8_t localVertexIndex ) const
  {
    return m_collisionMeshData->m_indices[3 * triangleIndex + localVertexIndex];
  }


  AGX_FORCE_INLINE agx::Real Mesh::getTriangleMaximumEdgeLength( size_t triangleIndex ) const
  {
    return m_collisionMeshData->m_triangleMaximumEdgeLengths[triangleIndex];
  }


  AGX_FORCE_INLINE agx::Real Mesh::getTotalMaximumEdgeLength() const
  {
    return m_collisionMeshData->m_totalMaximumEdgeLength;
  }


  AGX_FORCE_INLINE uint32_t Mesh::getHalfEdgePartnerEdge( size_t globalEdgeIndex ) const
  {
    return m_collisionMeshData->m_halfEdges[globalEdgeIndex];
  }


  AGX_FORCE_INLINE bool Mesh::isValid() const
  {
    return m_collisionMeshData->m_valid;
  }


  AGX_FORCE_INLINE bool Mesh::isValidAndClosed() const
  {
    return (m_collisionMeshData->m_valid && m_collisionMeshData->m_closed);
  }

  AGX_FORCE_INLINE agx::Vec3 Mesh::getCenter() const
  {
    return m_collisionMeshData->m_center;
  }


  AGX_FORCE_INLINE agx::Real Mesh::getVolume() const
  {
    return m_collisionMeshData->m_volume;
  }


  AGX_FORCE_INLINE agx::SPDMatrix3x3 Mesh::calculateInertia( agx::Real mass ) const
  {
    return mass * m_collisionMeshData->m_inertia;
  }


  AGX_FORCE_INLINE bool Mesh::isTerrain() const
  {
    return m_collisionMeshData->m_isTerrain;
  }

  AGX_FORCE_INLINE bool Mesh::hasInternalCenterPoint() const
  {
    return m_collisionMeshData->m_hasInternalCenterPoint;
  }


  AGX_FORCE_INLINE agx::Real Mesh::getBottomMargin() const
  {
    return m_collisionMeshData->m_bottomMargin;
  }


  AGX_FORCE_INLINE void Mesh::setBottomMargin(agx::Real newMargin)
  {
    if (newMargin >= 0) {
      m_collisionMeshData->m_bottomMargin = newMargin;
      incrementModifiedCount();
      updateBoundingVolume();
    }
  }


  AGX_FORCE_INLINE bool Mesh::isValidEdgeIndex(size_t globalEdgeIndex ) const
  {
    return globalEdgeIndex < m_collisionMeshData->m_indices.size();
  }


  AGX_FORCE_INLINE bool Mesh::isValidTriangleIndex(size_t globalTriangleIndex ) const
  {
    return globalTriangleIndex < m_collisionMeshData->m_numTriangles;
  }


  AGX_FORCE_INLINE const CollisionMeshData* Mesh::getMeshData() const
  {
    return m_collisionMeshData.get();
  }


  AGX_FORCE_INLINE bool Mesh::hasHalfEdge() const
  {
    return m_collisionMeshData->m_hasHalfEdge;
  }


  AGX_FORCE_INLINE const AabbTree* Mesh::getBvhTree() const
  {
    return m_aabbTree;
  }



  // From class VertexEdgeCirculator.

  AGX_FORCE_INLINE void Mesh::VertexEdgeCirculator::operator++(int)
  {
    if (m_valid) {
      if (m_hasJustJumpedOverHole) {
        // must have moved already to come here
        // keep same triangle, just move local index
        m_localEdgeIndex = (uint_fast8_t)((m_localEdgeIndex + (uint_fast8_t)1) % (uint_fast8_t)3);
        m_hasJustJumpedOverHole = false;
      } else {
        m_hasMoved = true;
        size_t he = m_mesh->getHalfEdgePartnerEdge(3 * m_triangleIndex + m_localEdgeIndex);
        if (!m_mesh->isValidEdgeIndex(he)) {
          // Found hole.
          // Go all the way in counter clockwise direction from start.
          m_triangleIndex = m_triangleStartIndex;
          m_localEdgeIndex = (uint_fast8_t)((m_localEdgeStartIndex + (uint_fast8_t)2) % (uint_fast8_t)3);
          m_hasJustJumpedOverHole = true;
          size_t he2 = m_mesh->getHalfEdgePartnerEdge(3 * m_triangleIndex + m_localEdgeIndex);
          while (m_mesh->isValidEdgeIndex( he2 )) {
            m_triangleIndex = he2 / 3;
            m_localEdgeIndex = (uint_fast8_t)((he2 + 2) % 3);
            he2 = m_mesh->getHalfEdgePartnerEdge(3 * m_triangleIndex + m_localEdgeIndex);
          }
        } else {
          // normal case not influenced by hole: continue in clockwise direction
          m_triangleIndex = he / 3;
          m_localEdgeIndex = (uint_fast8_t)((he + 1) % 3);
        }
      }
    }
  }


  AGX_FORCE_INLINE Mesh::VertexEdgeCirculator& Mesh::VertexEdgeCirculator::operator++()
  {
    (*this)++;
    return *this;
  }



  AGX_FORCE_INLINE bool Mesh::VertexEdgeCirculator::isValid() const
  {
    return m_valid;
  }



  AGX_FORCE_INLINE bool Mesh::VertexEdgeCirculator::atEnd() const
  {
    return (m_valid && m_hasMoved && m_localEdgeIndex == m_localEdgeStartIndex &&
            m_triangleIndex == m_triangleStartIndex);
  }



  AGX_FORCE_INLINE agx::Vec3 Mesh::VertexEdgeCirculator::getEdgeStartVertex()
  const
  {
    return m_mesh->getMeshData()->m_vertices[
             m_mesh->getMeshData()->m_indices[3 * m_triangleIndex + m_localEdgeIndex]];
  }



  AGX_FORCE_INLINE agx::Vec3 Mesh::VertexEdgeCirculator::getEdgeEndVertex()
  const
  {
    return m_mesh->getMeshData()->m_vertices[
             m_mesh->getMeshData()->m_indices[3 * m_triangleIndex + (m_localEdgeIndex + 1) % 3]];
  }


  AGX_FORCE_INLINE bool Mesh::VertexEdgeCirculator::hasJustJumpedOverHole() const
  {
    return m_hasJustJumpedOverHole;
  }


  inline const Mesh::Triangle Mesh::VertexEdgeCirculator::getTriangle()
  const
  {
    return Triangle( m_mesh->getTriangle( m_triangleIndex ));
  }


  AGX_FORCE_INLINE uint_fast8_t Mesh::VertexEdgeCirculator::getLocalEdgeIndex() const
  {
    return m_localEdgeIndex;
  }


  AGX_FORCE_INLINE size_t Mesh::VertexEdgeCirculator::getTriangleIndex() const
  {
    return m_triangleIndex;
  }


  AGX_FORCE_INLINE const Mesh* Mesh::VertexEdgeCirculator::getMesh() const
  {
    return m_mesh.get();
  }


  AGX_FORCE_INLINE void Mesh::calculateTrianglesInVolume( const BoundingAABB& volume,
      const agx::AffineMatrix4x4& meshToWorld, const agx::AffineMatrix4x4& volumeToWorld,
      agx::UInt32Vector& triangleIndices ) const
  {
    m_aabbTree->findCollision(&volume, std::back_inserter(triangleIndices), meshToWorld, volumeToWorld );
  }



  AGX_FORCE_INLINE void Mesh::calculateTrianglesInVolume( const BoundingAABB& volume,
      const agx::AffineMatrix4x4& meshToWorld, const agx::AffineMatrix4x4& volumeToWorld,
      agxData::LocalVector<agx::UInt32>& triangleIndices ) const
  {
    m_aabbTree->findCollision(&volume, std::back_inserter(triangleIndices), meshToWorld, volumeToWorld );
  }


  inline Mesh::VertexEdgeCirculator Mesh::createVertexEdgeCirculator(
    size_t triangleIndex,
    uint_fast8_t localVertexIndex ) const
  {
    return VertexEdgeCirculator( this, triangleIndex, localVertexIndex );
  }



  // From class FaceEdgeCirculator.

  AGX_FORCE_INLINE void Mesh::FaceEdgeCirculator::operator++(int)
  {
    if (m_valid) {
      m_hasMoved = true;
      m_localEdgeIndex = (uint_fast8_t)((m_localEdgeIndex + (uint_fast8_t)1) % (uint_fast8_t)3);
    }
  }


  AGX_FORCE_INLINE Mesh::FaceEdgeCirculator& Mesh::FaceEdgeCirculator::operator++()
  {
    (*this)++;
    return *this;
  }


  AGX_FORCE_INLINE bool Mesh::FaceEdgeCirculator::isValid() const
  {
    return m_valid;
  }



  AGX_FORCE_INLINE bool Mesh::FaceEdgeCirculator::atEnd() const
  {
    return (m_valid && m_hasMoved && m_localEdgeIndex == 0);
  }



  AGX_FORCE_INLINE agx::Vec3 Mesh::FaceEdgeCirculator::getEdgeStartVertex()
  const
  {
    return m_mesh->getMeshData()->m_vertices[
             m_mesh->getMeshData()->m_indices[3 * m_triangleIndex + m_localEdgeIndex]];
  }



  AGX_FORCE_INLINE agx::Vec3 Mesh::FaceEdgeCirculator::getEdgeEndVertex()
  const
  {
    return m_mesh->getMeshData()->m_vertices[
             m_mesh->getMeshData()->m_indices[3 * m_triangleIndex + (m_localEdgeIndex + 1) % 3]];
  }


  inline const Mesh::Triangle Mesh::FaceEdgeCirculator::getTriangle()
  const
  {
    return m_mesh->getTriangle( m_triangleIndex );
  }


  AGX_FORCE_INLINE uint_fast8_t Mesh::FaceEdgeCirculator::getLocalEdgeIndex() const
  {
    return m_localEdgeIndex;
  }



  inline Mesh::FaceEdgeCirculator Mesh::createFaceEdgeCirculator(
    size_t triangleIndex ) const
  {
    return FaceEdgeCirculator( this, triangleIndex );
  }


  AGX_FORCE_INLINE const Mesh* Mesh::FaceEdgeCirculator::getMesh() const
  {
    return m_mesh.get();
  }

  AGX_FORCE_INLINE const BoundingAABB& Mesh::getLocalBound() const
  {
    return m_localBound;
  }


}

#endif /* AGXCOLLIDE_MESH_H */
