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

#ifndef AGXOSG_RENDERINGWRAPPER_H
#define AGXOSG_RENDERINGWRAPPER_H

#if defined(SWIG) || defined(_MSC_VER)
#include <agx/PushDisableWarnings.h> // Disabling warnings. Include agx/PopDisableWarnings.h below!

#include <GL/gl.h>

#include <osgViewer/Viewer>
#include <osg/Version>
#include <osgGA/TrackBallManipulator>
#if defined(OSG_VERSION_GREATER_OR_EQUAL)
# if OSG_VERSION_GREATER_OR_EQUAL(3,0,0)
#   include <osgViewer/api/Win32/GraphicsWindowWin32>
# else
#  include <osgViewer/GraphicsWindowWin32>
# endif
#else
#  include <osgViewer/GraphicsWindowWin32>
#endif
#include <agx/PopDisableWarnings.h> // End of disabled warnings.

#include <agxOSG/export.h>
#include <agx/Integer.h>
#include <agxSDK/Simulation.h>
#include <agxOSG/RenderProxy.h>

#include <memory>

namespace agxOSG
{
  struct RenderingWrapperWindowsTypes;
}

namespace agxOSG
{
  class AGXOSG_EXPORT GraphicsNode
  {
  public:
    GraphicsNode(osg::Node* node);
    ~GraphicsNode();

    osg::Node* getNode() const;

    /**
    Read an image from disk and apply to the specified node as a 2D Texture in OVERRIDE|PROTECTED mode.
    \return false if image cannot be read.
    */
    bool setTexture(const agx::String& imagePath, bool repeat = false, agxOSG::TextureMode textureMode = agxOSG::DIFFUSE_TEXTURE, float scaleX = 1.0f, float scaleY = 1.0f);

    /**
    Set the specular part of a material for a node. If the node  does not have a material, a new one
    will be created and assigned to the node.
    */
    void setSpecularColor(const agx::Vec4f& color);

    /**
    Set the shininess exponent for the Phong specular model
    */
    void setShininess(float shininess);

    /**
    Set the ambient part of a material for a node. If the node  does not have a material, a new one
    will be created and assigned to the node.
    */
    void setAmbientColor(const agx::Vec4f& color);

    /**
    Set the alpha part of the material for a node. 0 completely transparent, 1 - opaque
    */
    void setAlpha(float alpha);

    /**
    Set the diffuse part of a material for a node. If the node  does not have a material, a new one
    will be created and assigned to the node.
    */
    void setDiffuseColor(const agx::Vec4f& color);

  private:
    osg::observer_ptr< osg::Node > m_node;
  };

  class AGXOSG_EXPORT GraphicsWindow
  {
  public:
    enum WindowFunctionDisable
    {
      NONE = 0x0,
      FULLSCREEN = 0x1
    };

    GraphicsWindow();
    bool init(agx::Int64 handle, int disabledFunctions = NONE);

    void frame();

    void setActiveSimulation(agxSDK::Simulation *simulation);
    agxSDK::Simulation *getActiveSimulation() const;
    void removeSimulation(agxSDK::Simulation *simulation);

    osg::Group* getRoot();

    void fitSceneIntoView();

    agxOSG::GraphicsNode* getVisual(const agx::RigidBody* rb) const;
    agxOSG::GraphicsNode* getOrCreateVisual(agx::RigidBody* rb);
    agxOSG::GraphicsNode* getVisual(const agxCollide::Geometry* geometry) const;
    agxOSG::GraphicsNode* getOrCreateVisual(agxCollide::Geometry* geometry);

    ~GraphicsWindow();

    friend RenderingWrapperWindowsTypes;
  private:
    void initCamera(osg::Camera *camera);

    std::unique_ptr<RenderingWrapperWindowsTypes> m_winTypes;
    osg::ref_ptr<osgViewer::Viewer> m_viewer;
    agxSDK::Simulation *m_activeSimulation;
    osg::ref_ptr<osg::Group> m_root;
    osgGA::StandardManipulator *m_manipulator;

    osg::ref_ptr<osg::GraphicsContext::Traits> m_traits;
    osg::ref_ptr<osg::GraphicsContext> m_graphicsContext;
    struct SimulationViewContext {
      agx::ref_ptr<agxOSG::RenderProxyFactory>  factory;
      osg::ref_ptr<osg::Camera> camera;
    };

    typedef agx::HashTable<agxSDK::Simulation *, SimulationViewContext> SimulationFactoryTable;
    SimulationFactoryTable m_factoryTable;

    typedef agx::HashVector< const agx::RigidBody*, GraphicsNode > RigidBodyGraphicsNodeContainer;
    typedef agx::HashVector< const agxCollide::Geometry*, GraphicsNode > GeometryGraphicsNodeContainer;
    RigidBodyGraphicsNodeContainer m_rbNodes;
    GeometryGraphicsNodeContainer m_geometryNodes;
  };
}


#endif // _MSC_VER
#endif
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

#ifndef AGXOSG_RENDERINGWRAPPER_H
#define AGXOSG_RENDERINGWRAPPER_H

#if defined(SWIG) || defined(_MSC_VER)
#include <agx/PushDisableWarnings.h> // Disabling warnings. Include agx/PopDisableWarnings.h below!

#include <GL/gl.h>

#include <osgViewer/Viewer>
#include <osg/Version>
#include <osgGA/TrackBallManipulator>
#if defined(OSG_VERSION_GREATER_OR_EQUAL)
# if OSG_VERSION_GREATER_OR_EQUAL(3,0,0)
#   include <osgViewer/api/Win32/GraphicsWindowWin32>
# else
#  include <osgViewer/GraphicsWindowWin32>
# endif
#else
#  include <osgViewer/GraphicsWindowWin32>
#endif
#include <agx/PopDisableWarnings.h> // End of disabled warnings.

#include <agxOSG/export.h>
#include <agx/Integer.h>
#include <agxSDK/Simulation.h>
#include <agxOSG/RenderProxy.h>

struct RenderingWrapperWindowsTypes;


namespace agxOSG
{
  class AGXOSG_EXPORT GraphicsNode
  {
    public:
      GraphicsNode( osg::Node* node );
      ~GraphicsNode();

      osg::Node* getNode() const;

      /**
      Read an image from disk and apply to the specified node as a 2D Texture in OVERRIDE|PROTECTED mode.
      \return false if image cannot be read.
      */
      bool setTexture( const agx::String& imagePath, bool repeat=false, agxOSG::TextureMode textureMode = agxOSG::DIFFUSE_TEXTURE, float scaleX = 1.0f, float scaleY = 1.0f );

      /**
      Set the specular part of a material for a node. If the node  does not have a material, a new one
      will be created and assigned to the node.
      */
      void setSpecularColor( const agx::Vec4f& color );

      /**
      Set the shininess exponent for the Phong specular model
      */
      void setShininess( float shininess );

      /**
      Set the ambient part of a material for a node. If the node  does not have a material, a new one
      will be created and assigned to the node.
      */
      void setAmbientColor( const agx::Vec4f& color );

      /**
      Set the alpha part of the material for a node. 0 completely transparent, 1 - opaque
      */
      void setAlpha( float alpha );

      /**
      Set the diffuse part of a material for a node. If the node  does not have a material, a new one
      will be created and assigned to the node.
      */
      void setDiffuseColor( const agx::Vec4f& color );

    private:
      osg::observer_ptr< osg::Node > m_node;
  };

  class AGXOSG_EXPORT GraphicsWindow
  {
    public:
      enum WindowFunctionDisable
      {
        NONE = 0x0,
        FULLSCREEN = 0x1
      };

      GraphicsWindow( );
      bool init( agx::Int64 handle, int disabledFunctions = NONE );

      bool initEmbeddedQT(int x, int y, int width, int height, float scale = 1.0f);
      void resize(int x, int y, int width, int height);
      osgGA::EventQueue* getEventQueue() const;

      void mouseMoveEvent(int x, int y);

      void mousePressEvent(int x, int y, int button);

      void mouseReleaseEvent(int x, int y, int button);

      void keyPressEvent(int key, int mod_key);
      void keyReleaseEvent(int key, int mod_key);

      void wheelEvent(int motion);

      void frame();

      void setActiveSimulation( agxSDK::Simulation *simulation );
      agxSDK::Simulation *getActiveSimulation() const;
      void removeSimulation( agxSDK::Simulation *simulation );

      osg::Group* getRoot();
      agxOSG::SceneDecorator* getSceneDecorator();

      void fitSceneIntoView();

      agxOSG::GraphicsNode* getVisual( const agx::RigidBody* rb ) const;
      agxOSG::GraphicsNode* getOrCreateVisual( agx::RigidBody* rb );
      agxOSG::GraphicsNode* getVisual( const agxCollide::Geometry* geometry ) const;
      agxOSG::GraphicsNode* getOrCreateVisual( agxCollide::Geometry* geometry );

      ~GraphicsWindow();

      void clear();

    private:
      void initCamera( osg::Camera *camera );

      std::unique_ptr<RenderingWrapperWindowsTypes> m_winTypes;
      osg::ref_ptr<osgViewer::Viewer> m_viewer;
      agxSDK::Simulation *m_activeSimulation;
      osg::ref_ptr<osg::Group> m_root;
      osg::ref_ptr<agxOSG::SceneDecorator> m_decorator;
      osgGA::StandardManipulator *m_manipulator;

      osg::ref_ptr<osg::GraphicsContext::Traits> m_traits;
      osg::ref_ptr<osg::GraphicsContext> m_graphicsContext;
      struct SimulationViewContext {
        agx::ref_ptr<agxOSG::RenderProxyFactory>  factory;
        osg::ref_ptr<osg::Camera> camera;
      };

      typedef agx::HashTable<agxSDK::Simulation *, SimulationViewContext> SimulationFactoryTable;
      SimulationFactoryTable m_factoryTable;

      typedef agx::HashVector< const agx::RigidBody*, GraphicsNode > RigidBodyGraphicsNodeContainer;
      typedef agx::HashVector< const agxCollide::Geometry*, GraphicsNode > GeometryGraphicsNodeContainer;
      RigidBodyGraphicsNodeContainer m_rbNodes;
      GeometryGraphicsNodeContainer m_geometryNodes;
      osg::ref_ptr<osgViewer::GraphicsWindowEmbedded> m_embeddedWindow;
      float m_scale;
  };
}


#endif // _MSC_VER
#endif
