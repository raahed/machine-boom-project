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

#ifndef AGX_MARCHINGCUBES_H
#define AGX_MARCHINGCUBES_H

#if AGX_USE_OPENGL()

#include <agx/config/AGX_USE_OPENGL.h>
#include <agx/agx_vector_types.h>
#include <agx/Referenced.h>
#include <agx/Vec3.h>
#include <agx/HashTable.h>

namespace agx
{
  class AGXPHYSICS_EXPORT MarchingCubes : public Referenced
  {
  public:
    MarchingCubes(Real resolution, Real searchRadius, Real surfaceThreshold = -1);

    void clearParticles();
    void addParticles(Vec3Vector& particlePositions);
    const Vec3Vector& calculateSurface();
    void drawCells();
    void drawSurfaceVertices();

  protected:
    virtual ~MarchingCubes();

  private:
    class SamplingCell;
    struct McVertex;
    struct McCell;

    typedef HashTable<Vec3i, SamplingCell *> SamplingCellHash;
    inline Vec3i calculateCellId(const Vec3& pos);

  private:
    SamplingCellHash m_cellHash;
    // Real m_searchRadius; // Not used yet.
    Real m_cellSize;
    Real m_invCellSize;
    Real m_surfaceThreshold;
    int m_sampleDimensions;
    Vec3Vector m_surfaceVertices;
  };

  struct AGXPHYSICS_EXPORT MarchingCubes::McVertex
  {
    McVertex() {}
    McVertex(const Vec3& pos) : value(0), position(pos) {}
    Real value;
    Vec3 position;
  };

  struct AGXPHYSICS_EXPORT  MarchingCubes::McCell
  {
    McCell() : type(-1) {}

    int type;
    McVertex *vertices[8];

    void calculateType(Real surfaceThreshold);
    void addSurfaceVertices(Vec3Vector& resultBuffer, Real invThreshold);
    Vec3 interpolateEdgeVertex(int edge, Real invThreshold);
  };


  class AGXPHYSICS_EXPORT MarchingCubes::SamplingCell
  {
  public:

    SamplingCell(const Vec3i& id, Real size, int sampleDimensions, SamplingCellHash& cellHash);
    ~SamplingCell();
    inline const Vec3Vector& getParticles();
    inline void clearParticles();
    inline void addParticle(const Vec3& particlePosition);
    inline const Vec3i& getId() const;

    void addNeighborConnections(SamplingCellHash& cellHash);
    void removeNeighborConnections();
    void calculateSurface(Real surfaceThreshold, Vec3Vector& vertexBuffer);

    inline McVertex *operator() (unsigned int x, unsigned int y, unsigned int z);

    SamplingCell **getNeighbors();

    void draw(Real surfaceThreshold);
  private:
    friend class MarchingCubes;
    void updateSamplePoints(Real surfaceThreshold);

    Vec3i m_id;
    Real m_size;
    Vec3 m_offset;
    SamplingCell *m_neighbors[26];
    Vec3Vector m_particles;
    Vector<McVertex> m_msVertices;
    Vector<McCell> m_msCells;
    unsigned int m_emptyCounter;
    Vec3 m_innerBoundsMin;
    Vec3 m_innerBoundsMax;
    unsigned int m_sampleDimensions;
  };



  typedef ref_ptr<MarchingCubes> MarchingCubesRef;
}

#endif

#endif /* _AGX_MARCHINGCUBES_H_ */
