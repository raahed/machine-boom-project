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

#include <agxOSG/export.h>

#include <agxTerrain/Terrain.h>

#include <agx/Task.h>

#include <agx/PushDisableWarnings.h> // Disabling warnings. Include agx/PopDisableWarnings.h below!
#include <osg/Drawable>
#include <osg/Version>
#include <agx/PopDisableWarnings.h> // End of disabled warnings.

namespace osgSim
{
  class ColorRange;
}


namespace agxOSG
{
  /**
  Internal drawable class for drawing the Voxel grid structure.
  */
  class AGXOSG_EXPORT TerrainVoxelDrawable : public osg::Drawable
  {
    public:
      TerrainVoxelDrawable(agxTerrain::Terrain* terrain);

      void setRenderVoxelSolidMass( bool enable );
      bool getRenderVoxelSolidMass() const;

      void setRenderVoxelFluidMass( bool enable );
      bool getRenderVoxelFluidMass() const;

      void setRenderVoxelBoundingBox( bool enable );
      bool getRenderVoxelBoundingBox() const;

      void setRenderVoxelCompaction( bool enable );
      bool getRenderVoxelCompaction() const;

      void setRenderVelocityField( bool enable );
      bool getRenderVelocityField() const;

      void setVelocityFieldLineColor(const agx::Vec4& lineColor);

      void setMaxVoxelValueColor(const agx::Vec4f& maxVoxelColor);

      /**
      \return true if at least one feature is enabled
      */
      bool hasEnabledFeatures() const;

      /**
      \return true when the render task has been added to the simulation
      */
      bool isInitialized() const;

      virtual osg::Object* cloneType() const { return new TerrainVoxelDrawable(m_terrain); }
      virtual osg::Object* clone(const osg::CopyOp&) const { return new TerrainVoxelDrawable(*this); }
      virtual bool isSameKindAs(const osg::Object* obj) const { return dynamic_cast<const TerrainVoxelDrawable *>(obj) != nullptr; }
      virtual const char* libraryName() const { return "agxOSG"; }
      virtual const char* className() const { return "TerrainVoxelDrawable"; }

      /** Compute the bounding box around Drawables's geometry.*/
#if OSG_VERSION_GREATER_OR_EQUAL(3,4,0)
      virtual osg::BoundingSphere computeBound() const;
      virtual osg::BoundingBox computeBoundingBox() const;
#else
      virtual osg::BoundingBox computeBound() const;
#endif

      /// The actual draw method for the merged bodies
      virtual void drawImplementation(osg::RenderInfo& renderInfo) const;

      void updateDrawable();

    public:
      void onAddNotification( agxSDK::Simulation* simulation );
      void onRemoveNotification( agxSDK::Simulation* simulation );

    protected:
      virtual ~TerrainVoxelDrawable();


      void renderVelocityTableLines(
        const agxTerrain::BasicGrid* solid,
        const agx::AffineMatrix4x4& terrainTransform,
        const agx::HashTable< agx::Vec3i, agx::Vec3f >& velocityTable,
        agx::Real elementSize,
        bool enableVelocityFieldRendering,
        agxData::Buffer* linePositions,
        agxData::Buffer* lineOffsets);

      void renderVoxels(
        const agxTerrain::BasicGrid* solid,
        const agxTerrain::BasicGrid* fluid,
        const agxTerrain::BasicGrid* compaction,
        const agx::AffineMatrix4x4& terrainTransform,
        agx::Real elementSize,
        bool enableRenderVoxelSolidMass,
        bool enableRenderVoxelFluidMass,
        bool enableRenderVoxelCompaction,
        agxData::Buffer* verticesBuffer,
        agxData::Buffer* colorBuffer,
        agxData::Buffer* voxelSizeBuffer,
        agxData::Buffer* enableRenderBuffer,
        std::function<agx::Vec4f(float)> getColor);

      void renderInternalNodes(
        const agxTerrain::BasicGrid* solid,
        const agx::AffineMatrix4x4& terrainTransform,
        agxData::Buffer* nodesVertices,
        agxData::Buffer* nodesColorBuffer,
        agxData::Buffer* nodesFirstBuffer,
        agxData::Buffer* nodesCountBuffer,
        bool enableRenderVoxelBoundingBox);


      //////////////////////////////////////////////////////////////////////////
      // Variables
      //////////////////////////////////////////////////////////////////////////
    protected:
      agxTerrain::TerrainRef            m_terrain;
      osg::Group*                       m_group;
      agx::TaskGroupRef                 m_renderTask;
      osg::ref_ptr<osgSim::ColorRange>  m_colorRange;
      bool                              m_enable;
      float                             m_rScaling;
      bool                              m_enableRenderVoxelSolidMass;
      bool                              m_enableRenderVoxelFluidMass;
      bool                              m_enableRenderVoxelCompaction;
      bool                              m_enableRenderVoxelBoundingBox;
      bool                              m_enableVelocityFieldRendering;

      agxData::BufferRef m_verticesBuffer;
      agxData::BufferRef m_colorBuffer;
      agxData::BufferRef m_voxelSizeBuffer;
      agxData::BufferRef m_voxelEnableRenderBuffer;
      agx::Vec4f         m_maxVoxelColor;

      // Bounding box data
      agxData::BufferRef m_nodesVertices;
      agxData::BufferRef m_nodesColorBuffer;
      agxData::BufferRef m_nodesCountBuffer;
      agxData::BufferRef m_nodesFirstBuffer;

      // Velocity field data
      agxData::BufferRef m_linePositions;
      agxData::BufferRef m_lineOffsets;
      agx::TaskRef       m_lineKernel;
  };
}
