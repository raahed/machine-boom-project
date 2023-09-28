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
/* This code comes from an example in OpenSceneGraph: example/osgoit.cpp */

#ifndef AGXOSG_DEPTHPEELING_H
#define AGXOSG_DEPTHPEELING_H

#include <osg/Version>
#ifdef OSG_VERSION_GREATER_OR_EQUAL
#if   OSG_VERSION_GREATER_OR_EQUAL(3,0,0)
# define AGX_HAVE_DEPTHPEELING 1
#endif
#endif

#ifdef AGX_HAVE_DEPTHPEELING

#include <agxOSG/export.h>

#include <agx/PushDisableWarnings.h> // Disabling warnings. Include agx/PopDisableWarnings.h below!
#include <osg/Referenced>
#include <osg/Group>
#include <osg/TextureRectangle>
#include <osgViewer/ViewerEventHandlers>
#include <agx/PopDisableWarnings.h> // End of disabled warnings.

// Some choices for the kind of textures we can use ...
#define USE_TEXTURE_RECTANGLE
// #define USE_NON_POWER_OF_TWO_TEXTURE
#define USE_PACKED_DEPTH_STENCIL


#ifdef _MSC_VER
# pragma warning(push)
//  warning C4251:  'X' : class 'Y' needs to have dll-interface to be used by clients of class 'Z'
# pragma warning( disable : 4251 )
#endif

namespace osg
{
  class Node;
}

namespace agxOSG
{

  /// Class that implements order independent transparency
  class AGXOSG_EXPORT DepthPeeling : public osg::Referenced {

  public:

    DepthPeeling(unsigned width, unsigned height);

    void setScene(osg::Node* scene);

    osg::Node* getRoot();

    void resize(int width, int height);

    void setNumPasses(unsigned numPasses);
    unsigned getNumPasses() const { return m_numPasses; }

    void setTexUnit(unsigned texUnit);

    void setShowAllLayers(bool showAllLayers);
    bool getShowAllLayers() const;

    void setEnable(bool depthPeelingEnabled);
    bool getEnable() const { return m_depthPeelingEnabled; }
    bool isEnabled() const { return m_depthPeelingEnabled; }

    void setOffsetValue(unsigned offsetValue);

    unsigned getOffsetValue() const { return m_offsetValue; }

    class EventHandler : public osgGA::GUIEventHandler {
    public:
      EventHandler(DepthPeeling* depthPeeling);
    protected:
      bool handle(const osgGA::GUIEventAdapter& ea, osgGA::GUIActionAdapter&, osg::Object*, osg::NodeVisitor*);
      using osgGA::GUIEventHandler::handle;
      osg::ref_ptr<DepthPeeling> m_depthPeeling;
    };


  protected:

    osg::Node*createQuad(unsigned layerNumber, unsigned numTiles);

    class CullCallback : public osg::NodeCallback {
    public:
      CullCallback(unsigned texUnit, unsigned texWidth, unsigned texHeight, unsigned offsetValue);

      virtual void operator()(osg::Node* node, osg::NodeVisitor* nv);

    private:
      unsigned m_texUnit;
      unsigned m_texWidth;
      unsigned m_texHeight;
      unsigned m_offsetValue;
    };

    void createPeeling();



    unsigned m_numPasses;
    unsigned m_texUnit;
    unsigned m_texWidth;
    unsigned m_texHeight;
    bool m_showAllLayers;
    bool m_depthPeelingEnabled;
    unsigned m_offsetValue;

    // The root node that is handed over to the viewer
    osg::ref_ptr<osg::Group> m_root;

    // The scene that is displayed
    osg::ref_ptr<osg::Group> m_scene;

    // The final camera that composites the pre rendered textures to the final picture
    osg::ref_ptr<osg::Camera> m_compositeCamera;

#ifdef USE_TEXTURE_RECTANGLE
    osg::ref_ptr<osg::TextureRectangle> m_depthTextures[2];
    std::vector<osg::ref_ptr<osg::TextureRectangle> > m_colorTextures;
#else
    osg::ref_ptr<osg::Texture2D> m_depthTextures[2];
    std::vector<osg::ref_ptr<osg::Texture2D> > m_colorTextures;
#endif
  };


} // namespace agxOSG

#ifdef _MSC_VER
# pragma warning(pop)
#endif


#endif

#endif
