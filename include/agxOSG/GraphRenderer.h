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

#ifndef AGXOSG_GRAPHRENDERER_H
#define AGXOSG_GRAPHRENDERER_H

#include <agxRender/RenderManager.h>
#include <agxOSG/export.h>
#include <agxRender/Graph.h>

#include <agx/PushDisableWarnings.h> // Disabling warnings. Include agx/PopDisableWarnings.h below!
#include <osg/Group>
#include <osg/Geometry>
#include <osg/Geode>
#include <osg/Array>
#include <agx/PopDisableWarnings.h> // End of disabled warnings.

namespace agxOSG
{
  class RenderProxyFactory;

  AGX_DECLARE_POINTER_TYPES(GraphRenderer);
  class AGXOSG_EXPORT GraphRenderer : public agxRender::Graph::GraphRenderer
  {
  public:
    GraphRenderer( agxRender::RenderManager* renderManager );

    void drawText( const agx::Vec2& pos, const agx::String& str);
    void setColor( const agx::Vec4& color );
    void drawLine( const agx::Vec2& p1, const agx::Vec2& p2 ) const;
    void preDraw();
    void postDraw();
    void addChannel();
    void removeChannel();
    size_t getNumChannels() const;

    virtual void setEnable( bool flag );

    void drawData( size_t channelIndex, const agxRender::Graph::DataVector& data );

    osg::Node *getNode( ) { return m_parent; }

    void clear();

    void setRenderManager( agxRender::RenderManager *mgr ) { m_mgr= mgr; }

  protected:

    virtual ~GraphRenderer();
    void updateLineGeometry( size_t channelIndex, const agxRender::Graph::DataVector& data );

    osg::ref_ptr<osg::Group> m_parent;
    agxRender::RenderManager *m_mgr;
    agx::Vec4 m_currentColor;

    struct OSGChannel {
      osg::ref_ptr<osg::Geode> geode;
      osg::ref_ptr<osg::Geometry> geometry;
      osg::DrawArrays*    primitiveSet;
      osg::ref_ptr<osg::Vec4Array> colors;
      osg::ref_ptr<osg::Vec3Array> vertices;
    };

    typedef agx::Vector<OSGChannel> Channels;
    Channels m_channels;
  };
}

#endif
