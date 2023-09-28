/*
Copyright 2007-2023. Algoryx Simulation AB.

All AGX source code, intellectual property, documentation, sample code,
tutorials, scene files and technical white papers, are copyrighted, proprietary
and confidential material of Algoryx Simulation AB.You may not download, read,
store, distribute, publish, copy or otherwise disseminate, use or expose this
material unless having a written signed agreement with Algoryx Simulation AB, or having been
advised so by Algoryx Simulation AB for a time limited evaluation, or having purchased a
valid commercial license from Algoryx Simulation AB.

Algoryx Simulation AB disclaims all responsibilities for loss or damage caused
from using this software, unless otherwise stated in written agreements with
Algoryx Simulation AB.
*/

#pragma once

#include <agxTerrain/export.h>
#include <agx/Referenced.h>
#include <agx/agx_vector_types.h>
#include <agx/StackArray.h>
#include <agx/HashVector.h>

namespace agxTerrain
{
  class Terrain;

  /**
  Class responsible for triggering avalanches in the terrain.
  */
  AGX_DECLARE_POINTER_TYPES(AvalancheController);
  class AGXTERRAIN_EXPORT AvalancheController : public agx::Referenced
  {
  public:
    using TerrainIndexSet = agx::HashSet<agx::Vec2i>;

  public:
    /**
    Construct given terrain instance.
    \param terrain - terrain instance
    */
    AvalancheController(Terrain* terrain);

    /**
    \internal

    Default constructor used in serialization.
    */
    AvalancheController();

    void resetIndices();

    void update();

    void setForbiddenVertices(const TerrainIndexSet& forbiddenVertices);

    void setIndicesToAvalanche(const TerrainIndexSet& indicesToAvalanche);

    void addIndicesToAvalanche(const TerrainIndexSet& indicesToAvalanche);

    void setAvalancheDecayPrecent(agx::Real avalancheDecayPrecent);

    agx::Real getAvalancheDecayPrecent() const;

    void setAvalancheMaxHeightGrowth(agx::Real maxHeightGrowth);

    agx::Real getAvalancheMaxHeightGrowth() const;

    const TerrainIndexSet& getModifiedVertices() const;

    const TerrainIndexSet& getAvalanchedVertices() const;

  public:
    DOXYGEN_START_INTERNAL_BLOCK()
    AGXTERRAIN_STORE_RESTORE_INTERFACE;
    DOXYGEN_END_INTERNAL_BLOCK()

  protected:
    virtual ~AvalancheController();


    // Used to tracks surface during avalanching and cache information so
    // nothing has to be read from the heightfield(s).
    // Terrain (x,y) -> ( surfaceVoxel, height)
    typedef agx::HashTable<agx::Vec2i, std::pair<agx::Vec3i, agx::Real> > SurfaceInfoTable;
    SurfaceInfoTable m_surfaceInfoTable;

    void applyGrowthMap();

    agx::Real findReposeHeightDifference(const agx::Vec2i& v1, const agx::Vec2i& v2);

    void generateVertexNeighborList(const agx::Vec2i& surfaceVertex, agx::StackArray<agx::Vec2i, 8>& neighborList );

    agx::Real calculateOccupancyTransfer(const agx::Vec2i& surfaceVertex, agx::Real heightDifference);

    agx::Real calculateMaximumOccupancyTransfer();

  private:
    class AvalancheIterationMap
    {
      public:
        AvalancheIterationMap(agx::Int kmin, agx::Int kmax)
            : m_kmin(kmin)
            , m_kmax(kmax)

        {
          agxAssert(kmin <= kmax);
          if (kmin <= kmax) {
            m_avalancheIterationVector.resize((kmax - kmin + 1), agx::HashVector<agx::Vec2i>());
          }
        }

        const agx::HashVector<agx::Vec2i>& operator[](agx::Int k)
        {
          resizeFromK(k);
          return m_avalancheIterationVector[toIndex(k)];
        }

        void push_back(agx::Int k, const agx::Vec2i& index)
        {
          resizeFromK(k);
          size_t vectorIndex = toIndex(k);
          if (!m_avalancheIterationVector[vectorIndex].contains(index))
          {
            m_avalancheIterationVector[vectorIndex].push_back(index,index);
          }
        }

        bool contains(agx::Int k, const agx::Vec2i& index)
        {
          size_t vectorIndex = toIndex(k);
          if (vectorIndex > m_avalancheIterationVector.size() - 1)
            return false;

          return m_avalancheIterationVector[vectorIndex].contains(index);
        }

        size_t toIndex(agx::Int k)
        {
          return m_kmax - k;
        }

        size_t toK(agx::Int i)
        {
          return m_kmax - i;
        }

        void resizeFromK(agx::Int k)
        {
          size_t vectorIndex = toIndex(k);
          if (vectorIndex > m_avalancheIterationVector.size() - 1)
          {
            m_avalancheIterationVector.resize(vectorIndex + 1);
            m_kmin = k;
          }
        }

        agx::Int kmin()
        {
          return m_kmin;
        }

        agx::Int kmax()
        {
          return m_kmax;
        }

        void clear()
        {
          m_avalancheIterationVector.clear();
        }

        size_t size()
        {
          return m_avalancheIterationVector.size();
        }

    private:
      agx::Vector<agx::HashVector<agx::Vec2i>> m_avalancheIterationVector;
      agx::Int m_kmin;
      agx::Int m_kmax;
    };

  private:
    void avalanche(const agx::Vec2i vertexXY, AvalancheIterationMap& avalancheMap);

  private:
    Terrain*                   m_terrain;
    TerrainIndexSet                        m_forbiddenTerrainVertices;
    TerrainIndexSet                        m_avalanchedTerrainVertices;
    TerrainIndexSet                        m_indicesToAvalanche;
    TerrainIndexSet                        m_modifiedTerrainVertices;
    agx::HashTable<agx::Vec2i, agx::Real>  m_growthTable;
    agx::Real                              m_avalancheDecayPrecent;
    agx::Real                              m_maxHeightGrowth;
  };

  AGX_FORCE_INLINE const AvalancheController::TerrainIndexSet& AvalancheController::getModifiedVertices() const
  {
    return m_modifiedTerrainVertices;
  }

  AGX_FORCE_INLINE const AvalancheController::TerrainIndexSet& AvalancheController::getAvalanchedVertices() const
  {
    return m_avalanchedTerrainVertices;
  }
}
