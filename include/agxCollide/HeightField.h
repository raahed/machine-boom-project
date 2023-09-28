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

#ifndef AGXCOLLIDE_HEIGHTFIELD_H
#define AGXCOLLIDE_HEIGHTFIELD_H

#include <agx/agxPhysics_export.h>

#include <agx/Vec2.h>
#include <agx/agx_vector_types.h>

#include <agxCollide/Mesh.h>
#include <agxStream/Serializable.h>
#include <external/GIMPACT/AabbTree.h>

#include <agxData/LocalVector.h>

namespace agxIO
{
  class Image;
}

namespace agxCollide
{
  AGX_DECLARE_POINTER_TYPES(HeightField);

  /**
  * A HeightField is a collision shape that can be created from a grid structure,
  * like an image. Each grid point defines a height, and the difference in height
  * between the grid points is interpolated by triangles which are then used for
  * collision detection. Each (rectangular) grid cell is divided in a lower left
  * and an upper right triangle.
  * The grid is assumed to be in dimension x and y, and z is assumed to be upwards
  * in positive height. In the x and y dimension, the grid is centered at the origin
  * in its local frame.
  * The grid is stored internally in row major order.
  */
  class AGXPHYSICS_EXPORT HeightField : public Mesh
  {
    public:
      static agxCollide::HeightField* createFromImage( agxIO::Image* image,
          agx::Real sizeX, agx::Real sizeY,
          agx::Real low, agx::Real high,
          agx::Real bottomMargin = agx::Real(1));

      /**
      \param filename - Path to an image which will be read as a height map.
      \param sizeX - World size in the x dimension
      \param sizeY - World size in the y dimension
      \param low   - The lowest height in the z dimension, maps to the lowest value in the image
      \param high  - The highest height in the z dimension, maps to the highest value in the image
      \param bottomMargin How deep is the HeightField under its lowest point?
      \return a HeightField shape with extents in X/Y and height values along Z
      */
      static agxCollide::HeightField* createFromFile( const agx::String& filename,
          agx::Real sizeX, agx::Real sizeY,
          agx::Real low, agx::Real high,
          agx::Real bottomMargin = agx::Real(1));

    public:

      /**
      * Constructor for the HeightField class. This will create a HeightField with
      * height 0 everywhere.
      * Use setHeights to set the heights at all points at once, or setHeight to
      * set it at especially defined positions.
      * \param resolutionX  The number of sample points in dimension x.
      * \param resolutionY  The number of sample points in dimension y.
      * \param sizeX The physical size in dimension x.
      * \param sizeY The physical size in dimension y.
      * \param bottomMargin How deep is the HeightField under its lowest point? Positive value.
      */
      HeightField(size_t resolutionX, size_t resolutionY, agx::Real sizeX,
                  agx::Real sizeY, agx::Real bottomMargin = agx::Real(1));

      /**
      * Constructor for the HeightField class. This will create a HeightField with
      * height 0 everywhere.
      * Use setHeights to set the heights at all points at once, or setHeight to
      * set it at especially defined positions.
      * \param resolutionX  The number of sample points in dimension x.
      * \param resolutionY  The number of sample points in dimension y.
      * \param sizeX The physical size in dimension x.
      * \param sizeY The physical size in dimension y.
      * \param heights - must be a row major matrix with dimensions
      *                  (resolutionX, resolutionY).
      * \param flipY - set to true if y decreases with increasing indices in heights.
      * \param bottomMargin How deep is the HeightField under its lowest point? Positive value.
      */
      HeightField(size_t resolutionX, size_t resolutionY, agx::Real sizeX,
                  agx::Real sizeY, const agx::RealVector& heights, bool flipY = false, agx::Real bottomMargin = agx::Real(1));

      /**
      * Set height values of all sampled points.
      * \param heights - must be a row major matrix with dimensions
      *                  (resolutionX, resolutionY).
      * \param flipY - set to true if y decreases with increasing indices in heights.
      */
      void setHeights(const agx::RealVector& heights, bool flipY = false);

      /**
      * Set height values given a list of terrain indices and corresponding heights
      * \param indices - list of indices to be updated. Must be of the same
      *                  dimensions as the height vector.
      *
      * \param heights - list of heights belonging to the indices to be updated. Must be
      *                  of the same dimensions as the indices vector.
      */
      void setHeightsFromList(const agx::Vec2uVector& indices, const agx::RealVector& heights);

      /**
      * Set height values of all grid nodes within a rectangle.
      * All triangles which belong to any of the points will be recalculated.
      * This is cheaper than calling setHeight(...) for all points in the rectangle,
      * since all triangles have to be computed just once instead of up to 6 times.
      *
      * \param heights A vector of size (maxVertexX + 1 - minVertexX) *
      *        (maxVertexY + 1 - minVertexY), giving the heights for the rectangle.
      * \param minVertexX The minimum vertex x index.
      * \param minVertexY The minimum vertex y index.
      * \param maxVertexX The maximum vertex x index.
      * \param maxVertexY The maximum vertex y index.
      * \retval true of succeeded (parameters within bounds and heights
      *                            has correct size).
      * For dynamic HeightFields, only the volume and center of mass of
      * the HeightField will be updated automatically, the unscaled inertia will
      * remain the same.
      * (Call the potentially costly calculateMassProperties after
      * all desired heights have been set in order to calculate the new mass
      * properties including inertia, if a correctly updated inertia is important).
      */
      bool setHeights(const agx::RealVector& heights,
                      size_t minVertexX, size_t maxVertexX,
                      size_t minVertexY, size_t maxVertexY);

      /**
      * Sets the height at a specific grid node.
      * \param xIndex The index of the grid node in dimension x.
      * \param yIndex The index of the grid node in dimension y.
      * \param height The desired height at this position.
      *
      * This method will recompute all adjacent triangles (up to 6).
      * If several adjacent heights of are changed, it can be more efficient
      * to call setHeights for the (rectangular) part of the grid that
      * should be changed.
      *
      * For dynamic HeightFields, only the volume and center of mass of
      * the HeightField will be updated automatically, the unscaled inertia will
      * remain the same.
      * (Call the potentially costly calculateMassProperties after
      * all desired heights have been set in order to calculate the new mass
      * properties including inertia, if a correctly updated inertia is important).
      */
      void setHeight(size_t xIndex, size_t yIndex, agx::Real height);

      /**
      * Gets the height at a grid point.
      * \param x x coordinate on grid.
      * \param y y coordinate on grid.
      * \retval height at this position.
      */
      agx::Real getHeight(size_t x, size_t y) const;

      /// Returns the resolution (the number of height values) in dimension x.
      size_t getResolutionX() const;

      /// Returns the resolution (the number of height values) in dimension y.
      size_t getResolutionY() const;

      /**
      * Returns a triangle given the 2D-projection of a point on the xy-plane in the grid.
      * The triangle will be invalid if the point's projection is outside the grid.
      */
      Mesh::Triangle getTriangleFromPoint(const agx::Vec3& point) const;

      /**
      * Projects a point in local HeightField coordinate system to the
      * HeightField grid (which is moved to the first quadrant).
      * \param point Point in height field coordinate system (height field center in origin).
      * \retval Point in height field grid. Will be on grid if values are between
      *   (0, 0) and (getResolutionX() - 1, getResolutionY() - 1).
      * Drop decimals for lower lower grid coordinates, round upwards for higher ones.
      */
      agx::Vec2 projectPointToGrid(const agx::Vec3& point) const;

      /// Returns the (global) minimum height of the HeightField.
      agx::Real getMinHeight() const;

      /// Returns the (global) maximum height of the HeightField.
      agx::Real getMaxHeight() const;

      /// Returns the size of the HeightField (in dimension x and y).
      agx::Vec2 getSize() const;

      /// Returns the scaling factor for a single cell in the HeightField (in dimension x and y).
      agx::Vec2 getScale() const;

      /// Returns the vertex at a grid position.
      agx::Vec3 getVertexFromGrid(size_t xIndex, size_t yIndex) const;

      /**
      * Sets the new minAllowedHeight, which is at
      * the same time the lower bound for a dynamic HeightField (for volume computation).
      * If the new minAllowedHeight is larger than the smallest
      * existing height, it will be set to the smallest existing height.
      * If the HeightField is dynamic, the mass properties will be recalculated,
      * which is expensive in large HeightFields (linear in number of triangles).
      * This function will not change existing height values, so that
      * there might be heights below the minimum allowed heights if the
      * the minimum allowed height is increased.
      * \param minAllowedHeight The new minimum height.
      */
      void setMinAllowedHeight(agx::Real minAllowedHeight);

      /**
      * Returns the minimum allowed height, which is at the same time
      * the lower bound for a dynamic HeightField (for volume computation).
      */
      agx::Real getMinAllowedHeight() const;

      /**
      * Sets if the HeightField is dynamic, which means if it should keep
      * its MassProperties updated.
      */
      void setDynamic(bool dynamic);

      /**
      * Return if the HeightField is dynamic, which means if it should keep
      * its MassProperties updated.
      */
      bool getDynamic() const;

      /**
      * Calculates the minimum and maximum vertex indices of the square that a
      * bounding box resides in.
      * \param volume The bounding box in its local coordinates.
      * \param meshToWorld A transformation from the mesh's local frame to world coordinates.
      * \param volumeToWorld A transformation from the volume's local frame to world coordinates.
      * \param gridMin The minimum vertex index x and y values.
      * \param gridMax The maximum vertex index x and y values.
      * \retval Does the volume have any overlap with the grid?
      */
      bool calculateVertexIndicesTouchedByAABB(
        const BoundingAABB& volume,
        const agx::AffineMatrix4x4& meshToWorld,
        const agx::AffineMatrix4x4& volumeToWorld,
        agx::Vec2i& gridMin,
        agx::Vec2i& gridMax) const;

      /**
      * Calculates an array of triangle indices of the triangles whose bounding volumes
      * get intersected by a line segment
      * \param segmentStartLocal The segment starting point in heightfield coordinates.
      * \param segmentEndLocal The segment ending point in heightfield coordinates.
      * \param triangleIndices A vector of triangle indices. It will be emptied before writing to it.
      */
      virtual void calculatePossibleTriangleOverlapsAlongLineSegment(
        const agx::Vec3& segmentStartLocal,
        const agx::Vec3& segmentEndLocal,
        agx::UInt32Vector& triangleIndices) const override;


      /**
      * Calculates an array of triangle indices of the triangles whose bounding volumes
      * get intersected by a line segment
      * \param segmentStartLocal The segment starting point in heightfield coordinates.
      * \param segmentEndLocal The segment ending point in heightfield coordinates.
      * \param triangleIndices A vector of triangle indices. It will be emptied before writing to it.
      */
      virtual void calculatePossibleTriangleOverlapsAlongLineSegment(
        const agx::Vec3& segmentStartLocal,
        const agx::Vec3& segmentEndLocal,
        agxData::LocalVector<agx::UInt32>& triangleIndices) const override;

      /**
      * Calculates the vertex indices for x and y given a global vertex index.
      * \param vertexIndex The global vertex index.
      * \param vertexIndexX A return value: The vertex index's x component.
      * \param vertexIndexY A return value: The vertex index's y component.
      */
      void calculateVertexCoordinates(
        size_t vertexIndex,
        size_t& vertexIndexX, size_t& vertexIndexY) const;

      /**
      * Computes the global triangle index within the HeightField.
      * \param indexX The triangle's x index (valid values: 0  to (getResolutionX() - 2)).
      * \param indexY The triangle's y index (valid values: 0  to (getResolutionY() - 2)).
      * \param upper Is it a lower left triangle (0) or upper right triangle (1)?
      */
      size_t getTriangleIndex(
        size_t indexX, size_t indexY,
        uint_fast8_t upper) const;

      /**
      * Creates a new HeightField which shares data with this one.
      * (should not be used for data which will be changed afterwards).
      * Quick to create, minimum memory overhead.
      * The caller takes over possession of the pointer.
      */
      HeightField* shallowCopy() const;

      /**
      * Creates a new HeightField which copies data from this one.
      * (modifying one's data will not modify the other's).
      * Less quick to create than shallowCopy (but faster than
      * creating mesh from input data), double memory usage.
      * The caller takes over possession of the pointer.
      */
      HeightField* deepCopy() const;

      /**
      * Calculates "mass properties" such as volume, unscaled inertia tensor and center.
      * Will give standard values for non-dynamic HeightFields.
      * For dynamic HeightFields, the mass properties will be recalculated per triangle
      * which is expensive in large HeightFields (linear in number of triangles).
      */
      void calculateMassProperties();

      /**
      \return a const pointer to the mesh data specific for this height field.
      */
      const HeightFieldMeshData* getHeightFieldMeshData() const;

      virtual Shape* clone() const override;

      DOXYGEN_START_INTERNAL_BLOCK()

      AGXSTREAM_DECLARE_SERIALIZABLE( agxCollide::HeightField );
      DOXYGEN_END_INTERNAL_BLOCK()

    protected:

      DOXYGEN_START_INTERNAL_BLOCK()

      // Update the bounding volume hierarchy.
      virtual void updateBvhTree() override;

      /**
      * Recalculates a specific triangle normal and maximum edge length
      *  and updates the bounding volume hierarchy.
      * \param triangleIndex The index of the triangle.
      * \param shouldUpdateBvhNode Should the bvh node be updated?
      */
      void updateTriangle(size_t triangleIndex, bool shouldUpdateBvhNode = true);

      void addQuadrantVolume(int xIndex, int yIndex, agx::Real factor);

      void addTriangleVolume(const agx::Vec3& v0, const agx::Vec3& v1,
                             const agx::Vec3& v2, agx::Real factor);


      // Builds a partial tree of the bounding volume hierarchy.
      void buildSubTree(
        agx::Vec2u start,
        agx::Vec2u end,
        int& nextNodeIndex,
        int parentIndex );

      /** Internal method, called by constructor or when restoring.
      * \param updateMassProperties Should mass properties be updated?
      * \param heights - must be a row major matrix with dimensions
      *                  (resolutionX, resolutionY).
      * \param flipY - set to true if y decreases with increasing indices in heights.
      */
      void finalize(bool updateMassProperties, const agx::RealVector& heights, bool flipY = false);

      /** Internal method, called by constructor or when restoring.
      * \param shouldUpdateMassProperties Should mass properties be updated?
      * \param heights - must be a row major matrix with dimensions
      *                  (resolutionX, resolutionY).
      * \param flipY - set to true if y decreases with increasing indices in heights.
      */
      void setHeights(bool shouldUpdateMassProperties, const agx::RealVector& heights, bool flipY = false);

      // Hiding default constructor.
      HeightField();

      // hiding assignment operator
      HeightField& operator=(const HeightField&);

      DOXYGEN_END_INTERNAL_BLOCK()

      // hiding destructor
      virtual ~HeightField();


    protected:
      HeightFieldMeshDataRef m_heightFieldData;
  };



  /* IMPLEMENTATIONS */

  AGX_FORCE_INLINE agx::Real HeightField::getMinHeight() const
  {
    return m_localBound.min().z() + m_collisionMeshData->m_bottomMargin;
  }


  AGX_FORCE_INLINE agx::Real HeightField::getMaxHeight() const
  {
    return m_localBound.max().z();
  }


  AGX_FORCE_INLINE size_t HeightField::getResolutionX() const
  {
    return m_heightFieldData->m_resolutionX;
  }


  AGX_FORCE_INLINE size_t HeightField::getResolutionY() const
  {
    return m_heightFieldData->m_resolutionY;
  }


  AGX_FORCE_INLINE agx::Real HeightField::getHeight(size_t x, size_t y) const
  {
    return m_collisionMeshData->m_vertices[x + y * m_heightFieldData->m_resolutionX][2];
  }


  AGX_FORCE_INLINE agx::Vec2 HeightField::getSize() const
  {
    return m_heightFieldData->m_size;
  }


  AGX_FORCE_INLINE agx::Vec2 HeightField::getScale() const
  {
    return m_heightFieldData->m_scale;
  }


  AGX_FORCE_INLINE void HeightField::calculateVertexCoordinates( size_t vertexIndex,
      size_t& vertexIndexX, size_t& vertexIndexY) const
  {
    vertexIndexX = vertexIndex % m_heightFieldData->m_resolutionX;
    vertexIndexY = vertexIndex / m_heightFieldData->m_resolutionX;
  }


  AGX_FORCE_INLINE size_t HeightField::getTriangleIndex(size_t indexX, size_t indexY, uint_fast8_t upper) const
  {
    return 2 * (indexX + indexY * (m_heightFieldData->m_resolutionX - 1)) + upper;
  }



  AGX_FORCE_INLINE agx::Vec3 HeightField::getVertexFromGrid(size_t xIndex, size_t yIndex) const
  {
    return getVertex(yIndex * m_heightFieldData->m_resolutionX + xIndex);
  }


  AGX_FORCE_INLINE void HeightField::setMinAllowedHeight( agx::Real minAllowedHeight )
  {
    // Get the lowest value z from bounding box.
    m_heightFieldData->m_minAllowedHeight = std::min( minAllowedHeight, m_localBound.min().z() + m_collisionMeshData->m_bottomMargin );
    if (m_heightFieldData->m_dynamic)
      calculateMassProperties();
  }


  AGX_FORCE_INLINE agx::Real HeightField::getMinAllowedHeight() const
  {
    return m_heightFieldData->m_minAllowedHeight;
  }


  AGX_FORCE_INLINE bool HeightField::getDynamic() const
  {
    return m_heightFieldData->m_dynamic;
  }


  AGX_FORCE_INLINE void HeightField::setDynamic( bool dynamic )
  {
    m_heightFieldData->m_dynamic = dynamic;
    calculateMassProperties();
  }


  AGX_FORCE_INLINE const HeightFieldMeshData* HeightField::getHeightFieldMeshData() const
  {
    return m_heightFieldData;
  }


  AGX_FORCE_INLINE agx::Vec2 HeightField::projectPointToGrid(const agx::Vec3& point) const
  {
    const agx::Vec2 projectedPoint = agx::Vec2(point[0], point[1]) + m_heightFieldData->m_size * agx::Real(0.5);
    const agx::Vec2 pointInGrid(projectedPoint[0] * m_heightFieldData->m_invScale[0], projectedPoint[1] * m_heightFieldData->m_invScale[1]);
    return pointInGrid;
  }

}


#endif /* AGXCOLLIDE_HEIGHTFIELD_H */
