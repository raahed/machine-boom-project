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

#ifndef AGXCOLLIDE_MESHDATA_H
#define AGXCOLLIDE_MESHDATA_H

#include <agx/agx_vector_types.h>
#include <agx/ref_ptr.h>
#include <agxCollide/BoundingAABB.h>
#include <agx/SPDMatrix3x3.h>
#include <agxStream/Serializable.h>

namespace agxCollide
{
  /**
  * Stores triangle mesh data as triangle lists, i.e. always three indices
  * refer to a triangle. The triangles are thus formed by indices
  * 012, 345, 678 and so on.
  * Each index contains a unsigned int referring to a position in the vertices
  * vector. Thus, vertices can be reused in several triangles at once,
  * saving storage space.
  * The number of triangles is indices.size() / 3, but is cached in
  * numTriangles.
  */
  class AGXPHYSICS_EXPORT MeshData : public agx::Referenced, public agxStream::Serializable
  {
    public:
      friend class Mesh;
      friend class Trimesh;
      friend class HeightField;
      friend class Convex;

      MeshData();

      agx::Vec3Vector& getVertices();
      const agx::Vec3Vector& getVertices() const ;

      agx::UInt32Vector& getIndices();
      const agx::UInt32Vector& getIndices() const;

      size_t getNumTriangles() const;

      const BoundingAABB& getBound() const;

      AGXSTREAM_DECLARE_SERIALIZABLE( agxCollide::MeshData );

    protected:
      /// Hiding destructor.
      virtual ~MeshData();
    protected:
      agx::Vec3Vector m_vertices;
      agx::UInt32Vector m_indices;
      BoundingAABB m_bound; // Double storing (also in shape entity), here it is only used for graphics.
      size_t m_numTriangles;
  };

  typedef agx::ref_ptr<MeshData> MeshDataRef;
  typedef agx::ref_ptr<const MeshData> ConstMeshDataRef;


  /// Class for data sharing only to be used by Mesh, its internal classes and children.
  class AGXPHYSICS_EXPORT CollisionMeshData : public MeshData
  {
    public:
      friend class Mesh;
      friend class Trimesh;
      friend class HeightField;
      friend class Convex;

      CollisionMeshData();

      /// Creates a deep copy of all data. Caller takes ownership of pointer.
      CollisionMeshData* deepCopy() const;

      /// \returns A vector containing the triangle normals.
      const agx::Vec3Vector& getNormals() const;

      const agx::Real32Vector& getTriangleMaximumEdgeLengths() const;

      const agx::UInt32Vector& getHalfEdges() const;

      agx::Real getTotalMaximumEdgeLength() const;

      bool isValid() const;

      bool hasHalfEdge() const;

      const agx::SPDMatrix3x3& getInertia() const;

      agx::Vec3 getCenter() const;

      agx::Real getVolume() const;

      agx::Real getBottomMargin() const;

      bool isTerrain() const;

      bool hasInternalCenterPoint() const;

      AGXSTREAM_DECLARE_SERIALIZABLE( agxCollide::CollisionMeshData );

    protected:
      /// Hiding destructor.
      virtual ~CollisionMeshData();

    protected:
      // Everything here should be added to the deepCopy()-method.
      agx::Vec3Vector m_normals;
      agx::Real32Vector m_triangleMaximumEdgeLengths;
      agx::UInt32Vector m_halfEdges;
      agx::SPDMatrix3x3 m_inertia;
      agx::Vec3 m_center;
      agx::Real m_totalMaximumEdgeLength;
      agx::Real m_volume;
      agx::Real m_bottomMargin;
      bool m_valid;
      bool m_closed;
      bool m_hasHalfEdge;
      bool m_isTerrain;
      bool m_hasInternalCenterPoint;
  };

  typedef agx::ref_ptr<CollisionMeshData> CollisionMeshDataRef;
  typedef agx::ref_ptr<const CollisionMeshData> ConstCollisionMeshDataRef;


  /// Class for data sharing only to be used by HeightField.
  class AGXPHYSICS_EXPORT HeightFieldMeshData : public agx::Referenced
  {
    public:
      friend class HeightField;

      HeightFieldMeshData();

      ///\return the size of the Heightfield
      agx::Vec2 getSize() const { return m_size; }

      ///\return the resolution of the heightfield
      agx::Vec2u32 getResolution() const;

      /// Creates a deep copy of all data. Caller takes ownership of pointer.
      HeightFieldMeshData* deepCopy() const;

    protected:
      /// Hiding destructor.
      virtual ~HeightFieldMeshData();
    protected:
      // Everything here should be added to the deepCopy()-method.
      agx::UIntVector m_boundIndices;
      agx::Vec2 m_size;
      agx::Vec2 m_scale;
      agx::Vec2 m_invScale; // Cached inverses of m_scale.
      agx::Real m_minAllowedHeight;
      size_t m_resolutionX;
      size_t m_resolutionY;
      bool m_dynamic; // Are we interested of mass properties?
  };

  typedef agx::ref_ptr<HeightFieldMeshData> HeightFieldMeshDataRef;
  typedef agx::ref_ptr<const HeightFieldMeshData> ConstHeightFieldMeshDataRef;



  // Implementations

  AGX_FORCE_INLINE agx::Vec3Vector& MeshData::getVertices()
  {
    return m_vertices;
  }


  AGX_FORCE_INLINE const agx::Vec3Vector& MeshData::getVertices() const
  {
    return m_vertices;
  }


  AGX_FORCE_INLINE agx::UInt32Vector& MeshData::getIndices()
  {
    return m_indices;
  }


  AGX_FORCE_INLINE const agx::UInt32Vector& MeshData::getIndices() const
  {
    return m_indices;
  }


  AGX_FORCE_INLINE size_t MeshData::getNumTriangles() const
  {
    return m_numTriangles;
  }


  AGX_FORCE_INLINE const BoundingAABB& MeshData::getBound() const
  {
    return m_bound;
  }


}

#endif

