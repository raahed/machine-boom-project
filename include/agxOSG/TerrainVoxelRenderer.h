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

#include <agx/config/AGX_USE_AGXTERRAIN.h>
#if defined(SWIG) || AGX_USE_AGXTERRAIN()

#include <agxOSG/Node.h>
#include <agxOSG/SoilParticleSystemDrawable.h>
#include <agxOSG/TerrainVoxelDrawable.h>

#include <agx/PushDisableWarnings.h> // Disabling warnings. Include agx/PopDisableWarnings.h below!
#include <osg/Drawable>
#include <osgViewer/Viewer>
#include <osgSim/ColorRange>
#include <osgSim/ScalarsToColors>
#include <osg/Version>
#include <agx/PopDisableWarnings.h> // End of disabled warnings.

#include <agx/config/AGX_USE_AGXTERRAIN.h>
#include <agxSDK/StepEventListener.h>
#include <agxTerrain/Terrain.h>
#include <agx/Task.h>

namespace agxOSG
{
  AGX_DECLARE_POINTER_TYPES(TerrainVoxelRenderer);

  /**
  Rendering class for the agxTerrain::Terrain.
  */
  class AGXOSG_EXPORT TerrainVoxelRenderer : public agxSDK::StepEventListener
  {
    public:
      /**
      Finds terrain renderer given terrain and simulation.
      \return terrain renderer if exist, otherwise nullptr
      */
      static TerrainVoxelRenderer* find( const agxTerrain::Terrain* terrain, const agxSDK::Simulation* simulation );

    public:
      enum RenderMode
      {
        DEPTH,
        COMPACTION
      };

    public:
      /**
      Construct given terrain, color range (height or compaction), render mode,
      wire frame and render group.
      \param terrain - terrain to render
      \param group - render group the terrain should be part of
      */
      TerrainVoxelRenderer( agxTerrain::Terrain* terrain, osg::Group* group );

      /**
      \return the terrain
      */
      agxTerrain::Terrain* getTerrain() const;

      /**
      \return the geometry node if valid, otherwise nullptr
      */
      agxOSG::GeometryNode* getNode() const;

      /**
      Enable/disable rendering of solid mass. Default: Disabled.
      \param enable - true to enable, false to disable
      */
      void setRenderVoxelSolidMass(bool enable);

      /**
      \return true if enabled, otherwise false
      */
      bool getRenderVoxelSolidMass() const;

      /**
      Enable/disable rendering of fluid mass. Default: Disabled.
      \param enable - true to enable, false to disable
      */
      void setRenderVoxelFluidMass(bool enable);

      /**
      \return true if enabled, otherwise false
      */
      bool getRenderVoxelFluidMass() const;

      /**
      Enable/disable rendering of voxel bounding boxes. Default: Disabled.
      \param enable - true to enable, false to disable
      */
      void setRenderVoxelBoundingBox(bool enable);

      /**
      \return true if enabled, otherwise false
      */
      bool getRenderVoxelBoundingBox() const;

      /**
      Enable/disable rendering of the terrain. Default: Enabled.
      \param enable - true to enable, false to disable
      */
      void setRenderHeightField(bool enable);

      /**
      \return true if enabled, otherwise false
      */
      bool getRenderHeightField() const;

      /**
      Enable/disable rendering of compaction in the voxel grid. Default: Disabled.
      \param enable - true to enable, false to disable
      */
      void setRenderVoxelCompaction(bool enable);

      /**
      \return true if enabled, otherwise false
      */
      bool getRenderVoxelCompaction() const;

      /**
      Enable/disable rendering of the mass velocity field. Default: Disabled.
      \param enable - true to enable, false to disable
      */
      void setRenderVelocityField(bool enable);

      /**
      \return true if enabled, otherwise false
      */
      bool getRenderVelocityField() const;

      /**
      Enable/disable rendering of solid particles. Default: Enabled.
      \param enable - true to enable, false to disable
      */
      void setRenderSoilParticles(bool enable);

      /**
      \return true if enabled, otherwise false
      */
      bool getRenderSoilParticles() const;

      /**
      Set enable render particles as triangle meshes. Default: Disabled.
      \param enable - true to enable, false to disable
      */
      void setRenderSoilParticlesMesh(bool enable);

      /**
      \return true if enabled, otherwise false
      */
      bool getRenderSoilParticlesMesh() const;

      /**
      Load soil particle mesh model and texture (optional).
      \param modelFilename - model filename and including relative path and optional
                             translate and scale ("model.obj.[2, 2, 2].scale.[0, 1, 2].trans"
      */
      void setSoilParticleMeshData( const agx::String& modelFilename, const agx::String& textureFilename = "" );

      /**
      \return model filename currently in use when rendering soil particles as mesh
      */
      const agx::String& getSoilParticleMeshModelFilename() const;

      /**
      \return model texture filename currently in use when rendering soil particles as mesh
      */
      const agx::String& getSoilParticleMeshTextureFilename() const;

      /**
      Enable/disable rendering of terrain height as a color range ranging
      from lowest as blue, neutral as green and highest as red. If enable
      and heightRange is default (-inf, inf) the min and max heights are
      calculated given current state of the terrain.
      \param enable - true to enable, false to disable
      \param heightRange - height range defining lowest and highest hight of the terrain
      */
      void setRenderHeights(bool enable, agx::RangeReal heightRange = agx::RangeReal());

      /**
      \return true if enabled, otherwise false
      */
      bool getRenderHeights() const;

      /**
      Enable/disable rendering of compaction.
      \param enable - true to enable, false to disable
      \param compactionRange - min and max compaction
      */
      void setRenderCompaction(bool enable, agx::RangeReal compactionRange = agx::RangeReal( 0.75, 1.5 ));

      /**
      \return true if enabled, otherwise false
      */
      bool getRenderCompaction() const;

      /**
      Set color of the terrain.
      \param color - color
      */
      void setColor(agx::Vec4f color);

      /**
      \return current color of the terrain
      */
      agx::Vec4f getColor() const;

      /**
      Assign line color of velocity fields in the voxel renderer.
      \param lineColor - line color
      */
      void setVelocityFieldLineColor(const agx::Vec4& lineColor);

      /**
      Assign max value color of velocity fields in the voxel renderer.
      \param maxVoxelColor - max voxel color
      */
      void setMaxVoxelValueColor(const agx::Vec4f& maxVoxelColor);

#ifndef SWIG
    public:
      struct ContextData
      {
        ContextData();
        ~ContextData();

        agx::Bool isValid() const;

        osg::ref_ptr<agxOSG::GeometryNode> node;
        osg::observer_ptr<osg::Group>      parent;
        osg::ref_ptr<osg::Geometry>        geometry;
        osg::ref_ptr<osg::Geode>           geode;
        osg::StateSet*                     stateSet;

        osg::ref_ptr<osgSim::ColorRange> colorRange;
        osg::ref_ptr<osg::Vec4Array>     colorArray;
      };

      struct SoilParticleMeshData
      {
        osg::ref_ptr<osg::Node> node;
        osg::ref_ptr<osg::Group> group;
        agx::String modelFilename;
        agx::String textureFilename;
      };

    public:
      virtual void addNotification() override;
      virtual void removeNotification() override;
      virtual void last( const agx::TimeStamp& ) override;
#endif

    protected:
      virtual ~TerrainVoxelRenderer();

    protected:
      agxTerrain::TerrainObserver              m_terrain;
      ContextData                              m_surfaceContextData;
      ContextData                              m_heightsContextData;
      osg::Vec4                                m_color;
      osg::ref_ptr<TerrainVoxelDrawable>       m_voxelDrawable;
      osg::ref_ptr<SoilParticleSystemDrawable> m_soilParticleDrawable;
      SoilParticleMeshData                     m_soilParticleMeshData;
  };
}
#endif
