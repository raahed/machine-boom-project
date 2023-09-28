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

#include <agx/PushDisableWarnings.h> // Disabling warnings. Include agx/PopDisableWarnings.h below!
#include <osgText/Text>
#include <osg/Geometry>
#include <osgViewer/Viewer>
#include <agx/PopDisableWarnings.h> // End of disabled warnings.

#include <agxOSG/GeometryNode.h>

#include <agx/RigidBody.h>
#include <agxSDK/PickHandler.h>

namespace agxOSG
{
  class AGXOSG_EXPORT Panel : public osg::Geode
  {
    public:
      Panel();
      Panel( const agx::Vec2& position, const agx::Vec2& size, const agx::Vec4& color );

      void setPosition( const agx::Vec2& position );
      const agx::Vec2& getPosition() const { return m_position; }

      void setSize( const agx::Vec2& size );
      const agx::Vec2& getSize() const { return m_size; }

      void setColor( const agx::Vec4& color );
      agx::Vec4 getColor() const;

      virtual void uninitialize( osg::Group* sceneData ) = 0;
      virtual bool isValid() const = 0;
      virtual void update() = 0;
      virtual bool match( const void* obj ) const = 0;

    protected:
      virtual ~Panel() {}
      void initialize();

    protected:
      agx::Vec2 m_position;
      agx::Vec2 m_size;

      osg::ref_ptr<osg::Geometry>         m_geometry;
      osg::ref_ptr<osg::Vec3Array>        m_vertices;
      osg::ref_ptr<osg::DrawElementsUInt> m_indices;
      osg::ref_ptr<osg::Vec4Array>        m_color;
      osg::ref_ptr<osg::Vec3Array>        m_normals;
  };

  typedef osg::ref_ptr< Panel > PanelRef;

  typedef agx::Vector< PanelRef > PanelContainer;

  // class to handle events with a pick

  // Testclass for the new design of PickHandler
  AGX_DECLARE_POINTER_TYPES(PickHandler);
  class AGXOSG_EXPORT PickHandler : public agxSDK::PickHandler
  {
  public:
    typedef agxSDK::PickHandler base;

    PickHandler( osgViewer::Viewer *viewer, osg::Group *scene=nullptr);


    /// Will detach any constrained rigidbody.
    void reset();

    /// Specify the modkeymask that will activate the PickHandler, for example MODKEY_LEFT_SHIFT
    void setActivateModKeyMask(unsigned int modKey );

    /// Specify the mousekey that will activate the PickHandler, for example RIGHT_MOUSE_BUTTON
    void setActivateMouseButton(unsigned int mouse);

    /// \return the current modkey mask that will activate the pick handler
    unsigned int getActivateModKeyMask() const;

    /// \return the current mouse button that will activate the pick handler
    unsigned int getActivateMouseButton() const;

    /// Intersect any geometry in the screen
    using agxSDK::GuiEventListener::intersect;
    agxSDK::PickResult intersect(float screenX, float screenY, bool onlyDynamics);

    /// Get the near and far points based on the current camera pose
    void getNearFarPoints(const osg::Camera* camera, agx::Vec3& near, agx::Vec3& far, float x, float y, bool useOldMatrix = false) const;

    /// \return the current camera
    const osg::Camera *getCamera() const;

    virtual bool keyboard(int key, unsigned int modKeyMask, float x, float y, bool keydown);

    /**
    \param button - Which mouse button was used
    \param state - What is the state of the button
    \param x,y - Position of mouse pointer
    */
    virtual bool mouse(MouseButtonMask button, MouseState state, float x, float y);
    virtual bool mouseDragged(MouseButtonMask buttonMask, float x, float y);
    virtual void update(float x, float y);

  protected:


    virtual ~PickHandler();

  private:

    unsigned int m_modKeyMask;
    unsigned int m_key;
    osg::Matrix m_cameraMatrix;
    osg::observer_ptr< osgViewer::Viewer > m_viewer;
    osg::observer_ptr< osg::Group > m_scene;
    osg::observer_ptr< osg::Camera > m_camera;

    unsigned int m_activateModKeyMask;
    unsigned int m_activateMouseKey;


  public: // Old interface
    virtual agxSDK::PickResult pick( float screenX, float screenY );


    void createMouseSpringRendering(float scale);
    void setSimulation( agxSDK::Simulation* simulation ) { agxSDK::EventListener::setSimulation(simulation); }
    void updateRenderState();
    void setCamera(osg::Camera *camera);

  private:
    void init();

    bool isInitialized() const { return m_initialized; }
    void updateClickedGeometry(MouseState state, float x, float y);

    // void createConstraint( const agx::Vec3& worldPoint );
    bool intersect( float screenX, float screenY, const osg::Camera* camera, agxCollide::LocalGeometryContactVector& result ) const;
    void startPick(float x, float y);
    void updatePanels(float x, float y);
    void updateUnphysical(agx::Vec3 direction, agx::Vec3 currentMousePoint);


    void traverseParticles();

    // Internal helper methods for mouse handling.
    bool mouseDown(MouseButtonMask button, float x, float y);
    bool mouseDownLeftButton(float x, float y);
    bool mouseDownMiddleButton(float x, float y);
    bool mouseDownRightButton(float x, float y);



  private: // Old interface

    agx::RigidBodyRef m_mousePickBody;

    bool m_initialized;
    bool m_useLock;
    bool m_moveUnphysically;
    bool m_inDragging;

    agx::Vec3 m_rotationFromStart;
    agx::Quat m_oldBodyRotation;
    agx::Vec3 m_oldBodyCMOffset;
    agx::Vec3 m_localPoint;
    agx::Real m_distance;



    osg::PositionAttitudeTransform *m_mouseAttachTransform;
    osg::PositionAttitudeTransform *m_mousePositionTransform;
    osg::Geometry *m_mouseSpringLine;
    osg::observer_ptr< osg::Switch > m_mouseMarkerSwitch;
    PanelContainer m_panels;

    osg::observer_ptr< osg::Switch > m_sphereSwitch;

    float m_mousePickingGraphicsScale;

  };
}
