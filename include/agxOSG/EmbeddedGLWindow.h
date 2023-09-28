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

#ifndef AGXOSG_EMBEDDEDGLWINDOW_H
#define AGXOSG_EMBEDDEDGLWINDOW_H


#include <agx/PushDisableWarnings.h> // Disabling warnings. Include agx/PopDisableWarnings.h below!

#include <osgViewer/Viewer>
#include <osg/Version>
#include <osgGA/TrackballManipulator>
#include <osgViewer/GraphicsWindow>

#include <agx/PopDisableWarnings.h> // End of disabled warnings.

#include <agxOSG/export.h>
#include <agx/Integer.h>
#include <agxSDK/Simulation.h>
#include <agxOSG/RenderProxy.h>

namespace agxOSG
{
  /**
  This class is for embedding an OSG based window inside another OpenGL contex, for example in QT
  */
  class AGXOSG_EXPORT EmbeddedGLWindow
  {
    public:

      /// Constructor
      EmbeddedGLWindow( );

      /**
      Will initialize the scenegraph, camera, viewer
      A valid OpenGL context should be initialized and active for this to work
      \param x,y - Position of the window
      \param width, height - Size of the window
      */
      bool init(int x, int y, int width, int height);

      /**
      Called at resize of the window
      \param x,y - new position of the window
      \param width, height - new size of the window
      */
      void resize(int x, int y, int width, int height);
      osgGA::EventQueue* getEventQueue() const;

      /**
      Called upon mouse event
      \param x,y - position of the mouse pointer in the window
      */
      void mouseMoveEvent(int x, int y);

      /**
      Called upon pressing a mouse button
      \param x,y - position of the mouse pointer in the window
      \param button - which button is pressed, 0,1,2
      */
      void mousePressEvent(int x, int y, int button);

      /**
      Called upon releasing a mouse button
      \param x,y - position of the mouse pointer in the window
      \param button - which button is released, 0,1,2
      */
      void mouseReleaseEvent(int x, int y, int button);

      /**
      Called upon a keypress event
      \param key - which key is pressed
      \param mod_mask - mask for the modification keys
      */
      void keyPressEvent(int key, int mod_mask);

      /**
      Called upon a keyrelease event
      \param key - which key is released
      \param mod_mask - mask for the modification keys
      */
      void keyReleaseEvent(int key, int mod_mask);

      /**
      Called when the mouse wheel is used
      \param motion - which direction the mouse wheel is moved, 0,1 (up/down)
      */
      void wheelEvent(int motion);

      /**
      Should be called to update the rendering frame
      */
      void frame();

      /**
      \return the camera
      */
      osg::Camera *getCamera();

      /**
      \return the camera
      */
      const osg::Camera *getCamera() const;

      /**
      Associate a Simulation to this window. Will be used to perform debug rendering.
      However, the simulation will not be step forward. That has to be handled explicitly
      */
      void setSimulation( agxSDK::Simulation *simulation );

      /**
      \return the associated simulation
      */
      agxSDK::Simulation *getSimulation() const;

      /**
      Deassociates any registered simulation
      */
      void removeSimulation( );

      /**
      \return the root node
      */
      osg::Group* getRoot();

      /**
      \return the root node
      */
      const osg::Group* getRoot() const;

      /**
      \return the SceneDecorator node
      */
      agxOSG::SceneDecorator* getSceneDecorator();

      /**
      Will move the camera so that the whole scene is visible
      */
      void fitSceneIntoView();

      /// Destructor
      ~EmbeddedGLWindow();

      /**
      Will remove simulation, deallocate all scene graph elements
      After call to this method, init is required
      */
      void clear();

    private:

      osg::ref_ptr<osgViewer::Viewer> m_viewer;
      agxSDK::SimulationRef m_simulation;
      osg::ref_ptr<osg::Group> m_root;
      osg::ref_ptr<agxOSG::SceneDecorator> m_decorator;
      osgGA::StandardManipulator *m_manipulator;

      osg::ref_ptr<osgViewer::GraphicsWindowEmbedded> m_embeddedWindow;
  };
}

#endif
