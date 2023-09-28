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

#ifndef AGXCOLLIDE_TRIMESH_H
#define AGXCOLLIDE_TRIMESH_H

#include <agx/agx_vector_types.h>
#include <agx/SPDMatrix3x3.h>

#include <agx/agxPhysics_export.h>
#include <agxCollide/MeshData.h>
#include <agxCollide/Mesh.h>
#include <agxCollide/TriangleVoronoiRegion.h>

#include <agxStream/Serializable.h>

#include <agxData/LocalVector.h>



namespace agxCollide
{
  AGX_DECLARE_POINTER_TYPES(Trimesh);

  /**
  Triangle mesh for geometric intersection tests.
  More detailed ways of creating it can be found in agxUtil::TrimeshReaderWriter.
  */
  class AGXPHYSICS_EXPORT Trimesh : public agxCollide::Mesh
  {
  public:

    /**
    Parameters for trimesh creation from data set.
    */
    enum TrimeshOptionsFlags {
      /**
      Counterclockwise orientation is standard,
      set this flag in order to change to clockwise orientation.
      */
      CLOCKWISE_ORIENTATION = 0x1,
      /**
      Deactivates warnings about flaws in the mesh topology which will lead
      to bad simulation results. Error warnings about illegal mesh data will still be
      issued.
      If trimesh data which will lead to warnings has to be used, it can be a good idea
      to use this flag for performance reasons:
      1. This will avoid having to send warning strings to LOGGER_WARNING
      2. It will also avoid having to store them, increasing memory footprint and serialization
         file size.
      However, the best alternative is to avoid/fix the warnings, since they indicate that
      the mesh might behave badly in the simulation!
      */
      NO_WARNINGS = 0x10,
      /**
      Set this flag in order to have the mesh to behave as a terrain,
      with z up as the terrain up normal.
      */
      TERRAIN = 0x20,
      /**
      Removes exactly identical duplicates in the vertices, as well as
      (resulting or previously existing) triangles which use the same vertex more than once.
      */
      REMOVE_DUPLICATE_VERTICES = 0x40,
      /**
      Tries to rearrange indices for all triangles so that normals for neighboring triangles
      (that share edges) use the edges in the same direction.
      */
      RECALCULATE_NORMALS_GIVEN_FIRST_TRIANGLE = 0x80
    };


    /////////////////////////////////
    // Trimesh member declarations //
    /////////////////////////////////

    // Constructors //

    /**
    Constructs a new Trimesh object from a vector of vertices and indices each.
    The Trimesh should either be a manifold or a terrain.
    If it is a terrain, this should be specified in the optionsMask.
      A terrain trimesh should be used only for static objects.
      The terrain mesh should not contain holes except the outer bounds, and
      it should be oriented towards the normal [0 0 1] (z up).
    The constructor can result in warnings of two different types:
      'Trimesh creation error' will be printed to LOGGER_WARNING.
        They indicate that the trimesh will not be usable due to corrupt input data.
      'Trimesh creation warning' will be printed to LOGGER_WARNING by default,
        but this can be turned off by setting the NO_WARNINGS-flag in the optionsMask.
        The warnings in this category can be obtained later by the getWarnings()-method
        (if warnings have not been turned off).
        The warnings of this category warn that the data for this trimesh has severe
        shortcomings and might not behave well in the simulation.
        General hints when creating trimeshes:
        Triangle meshes for collision detection should be closed/manifolds.
        Consider merging close vertices.
        Consider also closing holes in the mesh.
        Consider orienting all triangle faces outwards.
        Try modeling in way that lets the trimesh represent a closed volume,
        with a clearly defined inside and outside (given by winding).
        For simulating terrain, please use the appropriate constructor.

    \param vertices Pointer to a Vector of vertices.
                    Vertices should not be duplicates of each other.
                    Has to have at least size 1.
                    All content will be copied. No ownership will be taken.
    \param indices Pointer to a Vector of size_t indicating the vertices
                   for each triangle by referring to the vertices parameter.
                   All triangles described by a set of 3 indices each have
                   to have counter-clockwise winding (all-clockwise-winding
                   can be specified in TrimeshOptionsFlags).
                   Has to have at least size 3.
                   All content will be copied. No ownership will be taken.
    \param sourceName Name of the data source for debugging.
    \param optionsMask Options can be set as in enum TrimeshOptionsFlags.
    \param bottomMargin A safety threshold for catching collisions below
           the terrain surface, in the terrain case.
    */
    Trimesh(
      const agx::Vec3Vector* vertices, const agx::UInt32Vector* indices,
      const char* sourceName, uint32_t optionsMask = Trimesh::REMOVE_DUPLICATE_VERTICES, agx::Real bottomMargin = 0);

    // Accessors //

    /**
    Calculates an array of triangle indices of the triangles whose bounding volumes
    get intersected by a line segment
    \param segmentStartLocal The segment starting point in trimesh coordinates.
    \param segmentEndLocal The segment ending point in trimesh coordinates.
    \param triangleIndices A vector of triangle indices. It will be emptied before being written to.
    */
    virtual void calculatePossibleTriangleOverlapsAlongLineSegment(
      const agx::Vec3& segmentStartLocal,
      const agx::Vec3& segmentEndLocal,
      agx::UInt32Vector& triangleIndices) const override;

    /**
    Calculates an array of triangle indices of the triangles whose bounding volumes
    get intersected by a line segment
    \param segmentStartLocal The segment starting point in trimesh coordinates.
    \param segmentEndLocal The segment ending point in trimesh coordinates.
    \param triangleIndices A vector of triangle indices. It will be emptied before being written to.
    */
    virtual void calculatePossibleTriangleOverlapsAlongLineSegment(
      const agx::Vec3& segmentStartLocal,
      const agx::Vec3& segmentEndLocal,
      agxData::LocalVector<agx::UInt32>& triangleIndices) const override;

    virtual Shape* clone() const override;

    /**
    Creates a new Trimesh which shares data with this one.
    (should not be used for data which will be changed afterwards).
    Quick to create, minimum memory overhead.
    Will only copy members from Mesh/Trimesh, not Shape (such as RenderData).
    The caller takes over possession of the pointer.
    */
    Trimesh* shallowCopy() const;

    /**
    Creates a new Trimesh which copies data from this one.
    (modifying one's data will not modify the other's).
    Less quick to create than shallowCopy (but faster than
    creating mesh from input data), double memory usage.
    Will only copy members from Mesh/Trimesh, not Shape (such as RenderData).
    The caller takes over possession of the pointer.
    */
    Trimesh* deepCopy() const;

    /**
    Updates the mesh geometry data like normals or edge convexity.
    Will update mass properties, bounding box and bounding volume hierarchy.
    Will be called after construction.
    It should also be called if vertex positions get modified after construction.
    \param showWarnings Should warnings be printed out to cout?
    \param recalculateMassProperties Should the mass properties be computed/recomputed?
    */
    void updateMeshGeometry(bool showWarnings, bool recalculateMassProperties = true);

    /// Gets option mask used when creating (can be interesting for debugging meshes).
    uint32_t getOptionsMask() const;

    /// Gets source name used when creating (can be interesting for debugging meshes).
    const agx::String& getSourceName() const;

    /**
    Gets warnings occurring when creating (can be interesting for debugging meshes).
    Will be empty if there where no warnings, or if warnings have been deactivated
    (see getOptionsMask()).
    */
    const agx::String& getWarnings() const;

    /**
    \return true if the Trimesh is a shallow copy of another Trimesh, false otherwise.
    */
    bool isShallowCopy() const;

    /**
    \return the shallow copy source of the Trimesh if it exists, nullptr otherwise.
    */
    const Trimesh* getShallowCopySource();

    AGXSTREAM_DECLARE_SERIALIZABLE(agxCollide::Trimesh);

  private:
    /// Hiding default constructor
    Trimesh();

    /// Hiding copy constructor
    Trimesh(const Trimesh& trimesh);

    /// Hiding assignment
    Trimesh& operator=(const Trimesh& trimesh);

    /**
    Adds trimesh data to an existing Trimesh.
    Should only be called within a Trimesh constructor.
    \param vertices Pointer to a Vector of vertices.
                    Vertices should not be duplicates of each other.
                    Has to have at least size 1.
                    All content will be copied. No ownership will be taken.
    \param indices Pointer to a Vector of size_t indicating the vertices
                   for each triangle by referring to the vertices parameter.
                   All triangles described by a set of 3 indices each have
                   to have counter-clockwise winding (all-clockwise-winding
                   can be specified in TrimeshOptionsFlags).
                   Has to have at least size 3.
                   All content will be copied. No ownership will be taken.
    \param sourceName Name of the data source for debugging.
    \param optionsMask Options can be set as in enum TrimeshOptionsFlags.
    */
    void addConstructionData(const agx::Vec3Vector* vertices,
      const agx::UInt32Vector* indices,
      const char* sourceName,
      uint32_t optionsMask);

    /**
    Finalizes the mesh. Should only be called within a constructor.
    \param sourceName Name of the data sources for debugging.
    */
    void finalize(const char* sourceName, bool showWarnings, uint32_t optionsMask);

    /**
    Flip normal of triangle with index 'triangleIndex'
    \param triangleIndex    - index of triangle to flip
    */
    void flipNormal(const size_t triangleIndex);

    /**
    Find triangle neighbor index and local edge index
    */
    bool getTriangleHalfEdgePartnerTriangleIndexAndLocalEdgeIndex(const size_t triangleIndex, const size_t localEdgeIndex, size_t& neighborIndex, size_t& neighborLocalEdgeIndex);



  protected:
    /**
    Calculates mass properties. Should only be called within finalize.
    */
    void calculateMassProperties(bool showWarnings);

    /**
    Finalizes the mesh topology, including mesh legality checks
    as well as normal and maximum error bound calculation.
    Should only be called within finalize.
    \param showWarnings - write warnings to output
    \param optionsMask  - TrimeshOptionsFlags
    */
    void finalizeMeshTopology(bool showWarnings, uint32_t optionsMask);

    /// Hidden, should only be called by child classes.
    Trimesh(Type type, agx::Physics::Geometry::ShapePtr entity, const agx::Vec3Vector* vertices, const agx::UInt32Vector* indices,
      const char* sourceName, uint32_t optionsMask = 0);

    /// Hiding default constructor for child classes.
    Trimesh(Type type, agx::Physics::Geometry::ShapePtr entity);

    // Sets the values in a shallow-copy way.
    void setValuesAsShallowCopy(const Trimesh* other);

    /// Hiding destructor
    virtual ~Trimesh();

    /// Update the bounding volume hierarchy. Internal method.
    virtual void updateBvhTree() override;


  protected:
    agx::String m_sourceName;
    agx::String m_warnings;
    uint32_t m_optionsMask;
    agx::ref_ptr<const Trimesh> m_shallowCopySource;

  };


  // Implementations

  inline void Trimesh::calculatePossibleTriangleOverlapsAlongLineSegment(
    const agx::Vec3& segmentStartLocal,
    const agx::Vec3& segmentEndLocal,
    agx::UInt32Vector& triangleIndices) const
  {
    this->getBvhTree()->findCollisionWithLineSegment(segmentStartLocal, segmentEndLocal, triangleIndices);
  }



  inline void Trimesh::calculatePossibleTriangleOverlapsAlongLineSegment(
    const agx::Vec3& segmentStartLocal,
    const agx::Vec3& segmentEndLocal,
    agxData::LocalVector<agx::UInt32>& triangleIndices) const
  {
    this->getBvhTree()->findCollisionWithLineSegment(segmentStartLocal, segmentEndLocal, triangleIndices);
  }

}

#endif
