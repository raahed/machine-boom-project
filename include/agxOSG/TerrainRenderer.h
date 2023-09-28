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


#ifndef AGXOSG_TERRAINRENDERER_H
#define AGXOSG_TERRAINRENDERER_H

#include <agx/PushDisableWarnings.h> // Disabling warnings. Include agx/PopDisableWarnings.h below!
#include <osgSim/ColorRange>
#include <osgSim/ScalarsToColors>
#include <osg/ref_ptr>
#include <agx/PopDisableWarnings.h> // End of disabled warnings.

#include <agxOSG/GeometryNode.h>
#include <agxRender/Color.h>
#include <agxSDK/StepEventListener.h>
#include <agxModel/Terrain.h>
#include <agxOSG/utils.h>

DOXYGEN_START_INTERNAL_BLOCK()


namespace agxOSG
{
  class Group;
  class Node;
  class GeometryNode;
}

namespace osg
{
  class HeightField;
  class Geometry;
}
DOXYGEN_END_INTERNAL_BLOCK()

namespace agxOSG
{
  class AGXOSG_EXPORT TerrainRenderer : public agxSDK::StepEventListener
  {
  public:
    TerrainRenderer( agxModel::Terrain* terrain, agx::RangeReal youngsRange, osg::Group* root );
    void post(const agx::TimeStamp& );
    agxOSG::GeometryNode *getTerrainNode() { return m_terrainNode; }

  protected:
    virtual ~TerrainRenderer();

  private:
    agxModel::TerrainRef m_terrain;
    agxOSG::GeometryNode *m_terrainNode;
    osg::HeightField* m_renderHeightField;
    agx::RangeReal m_youngsRange;
    osg::ref_ptr<osg::Group> m_particleGroup;
    osg::ref_ptr<osg::Geode> m_sphereInstance;
    osg::ref_ptr<osg::Vec4Array> m_colorArray;
  };

  class AGXOSG_EXPORT TerrainHeightColorRenderer : public agxSDK::StepEventListener
  {
  public:
    TerrainHeightColorRenderer(agxModel::Terrain* terrain,
      const agxRender::ColorVector& colorRange,
      agx::RangeReal youngsRange,
      bool renderWireFrame,
      osg::Group* root);

    void post(const agx::TimeStamp&);
    agxOSG::GeometryNode *getTerrainNode() { return m_terrainNode; }

  protected:
    virtual ~TerrainHeightColorRenderer();
  private:
    agxModel::TerrainRef m_terrain;
    agxOSG::GeometryNode *m_terrainNode;
    agx::RangeReal m_youngsRange;
    osg::ref_ptr<osg::Group> m_particleGroup;
    osg::ref_ptr<osg::Geode> m_sphereInstance;
    osg::ref_ptr<osgSim::ColorRange> m_colorRange;
    osg::ref_ptr<osg::Vec4Array> m_colorArray;
  };

  class AGXOSG_EXPORT TerrainGridRenderer : public agxSDK::StepEventListener
  {
  public:
    TerrainGridRenderer(agxModel::Terrain* terrain, const agxRender::Color& gridColor, agx::Real lineThickness, osg::Group* root);
    void post(const agx::TimeStamp&);
    agxOSG::GeometryNode *getTerrainNode() { return m_terrainNode; }

  protected:
    virtual ~TerrainGridRenderer();
  private:
    agxModel::TerrainRef m_terrain;
    agxOSG::GeometryNode *m_terrainNode;
    osg::HeightField* m_renderHeightField;
    osg::ref_ptr<osg::Vec4Array> m_colorArray;
  };
}

#endif
