/*
Copyright 2007-2023. Algoryx Simulation AB.

All AGX source code, intellectual property, documentation, sample code,
tutorials, scene files and technical white papers, are copyrighted, proprietary
and confidential material of Algoryx Simulation AB. You may not download, read,
store, distribute, publish, copy or otherwise disseminate, use or expose this
material unless having a written signed agreement with Algoryx Simulation AB, or
having been advised so by Algoryx Simulation AB for a time limited evaluation,
or having purchased a valid commercial license from Algoryx Simulation AB.

Algoryx Simulation AB disclaims all responsibilities for loss or damage caused
from using this software, unless otherwise stated in written agreements with
Algoryx Simulation AB.
*/

#pragma once

#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wconversion"
#endif

#include <agx/config/AGX_USE_OPENGL.h>
#include <agx/config/AGX_USE_FFMPEG.h>
#include <agx/config/AGX_USE_PYTHON.h>

#include <agx/config.h>

#include <agx/PushDisableWarnings.h> // Disabling warnings. Include agx/PopDisableWarnings.h below!
#include <osg/Switch>
#include <osgViewer/Viewer>
#include <osg/Version>
#include <osg/ref_ptr>
#include <osg/MatrixTransform>
#include <agx/PopDisableWarnings.h> // End of disabled warnings.

#include <agxOSG/DepthPeeling.h>
#include <agxOSG/export.h>
#include <agxOSG/GuiEventAdapter.h>
#include <agxOSG/PickHandler.h>
#include <agxOSG/RenderTarget.h>
#include <agxOSG/RenderToTexture.h>
#include <agxOSG/VideoFFMPEGPipeCapture.h>
#include <agxOSG/RigidBodyRenderCache.h>

#if AGX_USE_WEBSOCKETS()
#include <agxOSG/ExampleApplicationController.h>
#endif

#include <agxSDK/Simulation.h>
#include <agxCollide/GeometryPair.h>
#include <agxSDK/GuiEventListener.h>
#include <agxSDK/ContactEventListener.h>
#include <agxOSG/ImageCapture.h>
#include <agxCFG/ConfigScript.h>

#include <agx/Timer.h>
#include <agxOSG/StatisticsRenderer.h>

#include <agxIO/ArgumentParser.h>
#include <agx/HighAccuracyTimer.h>

#include <cstddef>

#if AGX_USE_OPENGL()
#include <agxGL/Camera.h>
#include <agxGL/Lights.h>
#endif

#if AGX_USE_PYTHON()
#  include <agxPython/ScriptManager.h>
#  include <agxPython/ScriptContext.h>
#  include <agx/config/AGX_PYTHON_VERSION.h>
#endif

#ifdef __clang__
#pragma clang diagnostic pop
#endif

namespace osgGA
{
  class StandardManipulator;
  class MatrixManipulator;
}

namespace agxFMI2
{
  namespace Export
  {
    class Module;
  }
}


namespace agxSDK
{
  class Assembly;
}

namespace agxNet
{
  class CoSimulationServer;
}

namespace agxOSG {
  class ExampleApplication;
  class SceneDecorator;
  class RenderProxyFactory;
  class GeometryNode;
#if AGX_USE_WEBSOCKETS()
  class ExampleApplicationController;
#endif

  typedef osg::Group * (*BuildScenePtr)(agxSDK::Simulation *simulation, ExampleApplication *application);

  /// A pod-struct for holding information about scenes to load by the application.
  struct AGXOSG_EXPORT SceneDescription {

    /// A non-valid scene.
    SceneDescription();

    /// Creates a scene from a function pointer.
    SceneDescription(BuildScenePtr sc, bool isTest = false, float stop = float(), bool isSlowUnittest = false);

    /// Creates a scene from a script file and a function within the file.
    SceneDescription(const std::string& scriptFileName, const std::string& scriptFunctionName, bool isTest = false, float stop = float(), bool isSlowUnittest = false);

    /// Creates a scene from any kind of agx-readable file.
    SceneDescription(const std::string& sceneFile, bool isTest = false, float stop = float(), bool isSlowUnittest = false);

    bool isValid() const { return valid; }


    BuildScenePtr scene;
    bool unittest;
    float stopAfter;
    bool slowUnittest;
    bool valid;
    std::string scriptFunction;
    std::string fileName;
  };


#if defined(OSG_VERSION_GREATER_OR_EQUAL)
# if OSG_VERSION_GREATER_OR_EQUAL(2,9,11)
    typedef osgGA::StandardManipulator CameraManipulatorType;
# else
    typedef osgGA::MatrixManipulator CameraManipulatorType;
# endif
#else
  typedef osgGA::MatrixManipulator CameraManipulatorType;
#endif



#if AGX_USE_OPENGL()
  class CameraSynchronization : public osg::Camera::DrawCallback
  {
  public:
    CameraSynchronization(osg::Camera *mainCamera, agxGL::Camera *glCamera) : m_mainCamera(mainCamera), m_glCamera(glCamera)
    {
    }

    virtual ~CameraSynchronization()
    {
    }

    virtual void operator () (osg::RenderInfo& /* renderInfo */) const
    {
      // Explicit support for Real=float agx with Real=double osg
      agxData::Val<agx::Matrix4x4d> projection(agx::Matrix4x4d(m_mainCamera->getProjectionMatrix().ptr()));
      agx::Matrix4x4 m = projection.transform<agx::Matrix4x4 > ();

      if (m != m_glCamera->getProjectionMatrix())
        m_glCamera->setProjectionMatrix(m);

      agxData::Val<agx::Matrix4x4d> view(agx::Matrix4x4d(m_mainCamera->getViewMatrix().ptr()));
      agx::Matrix4x4 m2 = view.transform<agx::Matrix4x4 > ();

      if (m2 != m_glCamera->getViewMatrix())
        m_glCamera->setViewMatrix(m2);

      auto viewport = m_mainCamera->getViewport();
      m_glCamera->setViewPort(static_cast<int>(viewport->width()), static_cast<int>(m_mainCamera->getViewport()->height()));
    }
    using osg::Camera::DrawCallback::operator();

  private:
    osg::Camera *m_mainCamera;
    agxGL::Camera *m_glCamera;
  };

  class LightsSynchronization : public osg::Camera::DrawCallback
  {
  public:
    LightsSynchronization(agxGL::Lights *mainLights, agxOSG::SceneDecorator *decorator)
      : m_glLights(mainLights),
      m_decorator(decorator)
    {
    }

    virtual ~LightsSynchronization()
    {
    }

    virtual void operator () (osg::RenderInfo& /* renderInfo */) const
    {

      auto updateLight = [this](size_t index, agxOSG::LightSource l)
      {
        m_glLights->setLightPosition(index, l.getPosition());
        m_glLights->setLightDirection(index, l.getDirection());
      };

      updateLight(0, m_decorator->getLightSource(agxOSG::SceneDecorator::LIGHT0));
      updateLight(1, m_decorator->getLightSource(agxOSG::SceneDecorator::LIGHT1));
      updateLight(2, m_decorator->getLightSource(agxOSG::SceneDecorator::LIGHT2));
    }

    using osg::Camera::DrawCallback::operator();

  private:
    agxGL::Lights  *m_glLights;
    SceneDecorator *m_decorator;
  };
#endif

  class AGXOSG_EXPORT CameraManipulatorFactory : public agx::Singleton
  {
    public:
      enum Types { TRACKBALL,
#if defined(OSG_VERSION_GREATER_OR_EQUAL)
# if OSG_VERSION_GREATER_OR_EQUAL(2,9,11)
                   FPS,
# endif
#endif
                   NUM_TYPES };

      CameraManipulatorFactory();

      static CameraManipulatorFactory* instance();

      CameraManipulatorType* next( osgViewer::Viewer* viewer = nullptr );
      CameraManipulatorType* next( osgViewer::GraphicsWindow* window = nullptr );
      CameraManipulatorType* create( osgViewer::Viewer* viewer = nullptr );
      CameraManipulatorType* create( Types type, osgViewer::Viewer* viewer = nullptr );
      CameraManipulatorType* create( Types type, osgViewer::GraphicsWindow* window = nullptr );
      CameraManipulatorType* create( osgViewer::GraphicsWindow* window = nullptr );

      void setType( Types type ) { m_currentType = type; }
      Types getType() const { return m_currentType; }

    protected:
      SINGLETON_CLASSNAME_METHOD();
      virtual void shutdown() override { s_instance = nullptr; }
      osgViewer::GraphicsWindow* getWindow( osgViewer::Viewer* viewer, size_t num = 0 ) const;

      static CameraManipulatorFactory* s_instance;
      Types m_currentType;
  };

  struct AGXOSG_EXPORT CameraData
  {
    /**
    Uses camera view matrix for eye, center and up.
    */
    CameraData( const osg::Camera* camera );

    /**
    Uses camera manipulator for eye, center and up.
    */
    CameraData( const osgViewer::Viewer* viewer );

    /**
    Apply current settings to a camera.
    */
    void applyTo( osg::Camera* camera ) const;

    /**
    Apply current settings to a viewer which assigns home position of the camera manipulator.
    */
    void applyTo( osgViewer::Viewer* viewer ) const;

    double nearClippingPlane;
    double farClippingPlane;
    double fieldOfView;
    double aspectRatio;
    bool valid;
    double nearFarRatio;

    agx::Vec3 eye;
    agx::Vec3 center;
    agx::Vec3 up;

    private:
      CameraData();
      void initialize( const osg::Camera* camera );
  };

  AGX_DECLARE_POINTER_TYPES( ExampleApplicationListener );
  class AGXOSG_EXPORT ExampleApplicationListener : public agx::Referenced {

  public:
    virtual void preFrame(ExampleApplication* app);
    virtual void postFrame(ExampleApplication* app);

  protected:
    virtual ~ExampleApplicationListener() override;
  };


  AGX_DECLARE_POINTER_TYPES( ExampleApplication );

  /**
  Class that encapsulates rendering and simulation using OpenSceneGraph
  */
  class AGXOSG_EXPORT ExampleApplication : public agx::Referenced
  {
  public:
    typedef agx::Event1<bool> AutoStepEvent;
    AutoStepEvent autoStepEvent;

    /**
    Specifies node masks for different parts of the rendering scene.
    Makes it possible to later disable rendering of certain parts for different cameras.
    For example the HUD should not be rendered to RenderTargets (sensor cameras) by default.
    */
    enum CameraMask
    {
      MAIN_SCENE_MASK = 1 << 1,
      DEBUG_RENDER_MASK = 1 << 2,
      OSG_RENDER_MASK = 1 << 3,
      HUD_MASK = 1 << 4,
      DECORATOR_MASK = 1 << 5,
    };

    enum SolverType
    {
      MULTI,      // New.
      ITERATIVE,  // New forced to iterative.
      NUM_SOLVERS
    };


    void setSolverType( SolverType t ) { m_solverType = t; }
    SolverType getSolverType() const { return m_solverType; }

    std::string getSolverName( SolverType t ) const;
    SolverType getSolverType( const std::string& name ) const;

    /**
    Add/remove a pair of bodies/geometries that when colliding will cause the simulation to pause.
    */
    void addAutoPausePair(const agx::RigidBody *body1, const agx::RigidBody *body2);
    void addAutoPausePair(const agxCollide::Geometry *geometry1, const agxCollide::Geometry *geometry2);
    void removeAutoPausePair(const agx::RigidBody *body1, const agx::RigidBody *body2);
    void removeAutoPausePair(const agxCollide::Geometry *geometry1, const agxCollide::Geometry *geometry2);

    /**
    Enable/disable auto-pausing.
    */
    void setEnableAutoPausing(bool flag);
    bool getEnableAutoPausing();

    /// Constructor
    ExampleApplication( agxSDK::Simulation *simulation=nullptr );

    bool init( agxIO::ArgumentParser* arguments, bool agxOnly = false );

    void addListener(ExampleApplicationListener *listener);

#ifndef SWIG
    bool init( int argc, char** argv );
#endif

    agxIO::ArgumentParser* getArguments() { return m_arguments.get(); }

    agxOSG::RenderProxyFactory *getRenderProxyFactory() { return m_renderProxyFactory; }

    void setTimeStep( agx::Real dt );
    agx::Real getTimeStep() const;

    bool addScene( const std::string& scriptFile, const std::string& scriptFunction, int key, bool isPartofUnitTest=true, float stopAfter=float(), bool isSlowUnittest=false );
    bool addScene( const std::string& scriptFile, const std::string& scriptFunction, bool isPartofUnitTest=true, float stopAfter=float(), bool isSlowUnittest=false );

    bool addScene( BuildScenePtr sceneFunction, int key, bool isPartofUnitTest=true, float stopAfter=float(), bool isSlowUnittest=false );
    bool addScene( BuildScenePtr sceneFunction, bool isPartofUnitTest=true, float stopAfter=float(), bool isSlowUnittest=false );

    bool addScene( SceneDescription sceneDescription, int key);
    bool addScene( SceneDescription sceneDescription);

    /// Remove all added scenes
    void clearAddedScenes() { m_keybindingVector.clear(); }

    agx::Callback getStepCallback() const { return m_stepCallback; }
    void setStepCallback(agx::Callback callback) { m_stepCallback = callback; }

    bool initialized() const { return m_initialized; }

    agxOSG::SceneDecorator *getSceneDecorator() { return m_decorator; }

    virtual int run();

    void setOsgNotifyLevel(osg::NotifySeverity severity) const;

    /**
    Stop (=exit) the application at next opportunity. Meaning, exit the run loop.
    \param exitCode - The exit code that will be used when returning from the ::run loop.
    */
    void stop(int exitCode = 0) { m_shouldStop = true; m_exitCode = exitCode; }

    /// Will cancel the stop if possible. Continue the ::run loop.
    void cancelStop() { m_shouldStop = false; }

    /// \return true if someone has called stop.
    bool shouldStop() const { return m_shouldStop; }


    /**
    Set the position and size of the window.
    \param x, y - X/Y position in pixels of the graphics window
    \param width, height - size in pixels of the graphics window
    \returns true if the window is present and the resize was successful. False if there are no windows to be modified.
    */
    bool setWindowRectangle(unsigned int x, unsigned int y, unsigned int width, unsigned int height);

    /**
    Set the title of the window.
    \param title - Title string to set on the window
    \returns true if the window is present and the title was successfully updated. False if there are no windows to be modified.
    */
    bool setWindowTitle(const std::string& title);

    /// \return true if someone called stop OR viewer indicates we should stop (ESC pressed)
    bool breakRequested() const;

    virtual void setupViewer( bool lightingEnabled = true, bool visibleWindow = true );

    void setupVideoCaptureRenderTotexture();

    //void initScene();

    osg::Group *getScene() { return m_scene.get(); }
    void setScene(osg::Group *scene ) { m_scene = scene; }
    osg::Group *getSceneRoot() { return m_sceneRoot.get(); }
    osg::Group* getSceneAgxOSGRoot();
    osg::Group *getSceneSwitch() { return m_sceneSwitch.get(); }

#ifdef AGX_HAVE_DEPTHPEELING
    void setEnableDepthPeeling( bool flag );
    bool getEnableDepthPeeling() const;
    agxOSG::DepthPeeling* getDepthPeeling();
#endif

    /**
    Set enabling the debug renderer.
    Works only when running with graphics (no --agxOnly) and a simulation exists.
    \param flag Should the debug renderer be enabled or not?
    \retval Was setting the value successful?
    Will e.g. return false if running without graphics, or if no simulation exists.
    */
    bool setEnableDebugRenderer( bool flag );

    /// Is the debug renderer enabled? Will also return false if no simulation exists.
    bool getEnableDebugRenderer() const;

    void setEnableTextDebugRendering( bool flag );
    bool getEnableTextDebugRendering() const;

    void setEnableCaptureSyncWithSimulation( bool enableSync );
    bool getEnableCaptureSyncWithSimulation(  ) const;

    void setEnableOSGRenderer( bool flag );
    bool getEnableOSGRenderer();

    void setEnableVSync( bool flag, bool forceUpdate = false );
    bool getEnableVSync() const;

    void getCameraHome(       agx::Vec3& eye,       agx::Vec3& center,       agx::Vec3& up);
    void setCameraHome( const agx::Vec3& eye, const agx::Vec3& center, const agx::Vec3& up=agx::Vec3(0,0,1) );

    /**
    Add a render target to the current viewer. This can be used to add an additional camera which renders to a texture.
    \param rtt - The RenderTarget to be added to the viewer
    \param targetSceneNode - If specified, this will be the part of the scene which is rendered to the target. Default is getSceneDecorator()
    */
    void addRenderTarget(agxOSG::RenderTarget* rtt, osg::Node *targetSceneNode=nullptr);

    /**
    Remove a specified renderTarget
    \returns true if target is removed
    */
    bool removeRenderTarget(agxOSG::RenderTarget* rtt);

    /**
    Transform the debug rendering AND pick handler origin by this matrix. This is handy when you are creating scenes which
    have coordinates which are very large. For example on the surface of the earth. Find the object which you want to transform into origin, get its translation
    and call this method with its inverse transform (usually translation only).
    */
    void setDebugRenderInverseMatrix( const agx::AffineMatrix4x4 &m );

#if defined(OSG_VERSION_GREATER_OR_EQUAL)
# if OSG_VERSION_GREATER_OR_EQUAL(2,9,11)
    osgGA::CameraManipulator* getCameraManipulator();
    const osgGA::CameraManipulator* getCameraManipulator() const;
#else
    osgGA::MatrixManipulator* getCameraManipulator();
    const osgGA::MatrixManipulator* getCameraManipulator() const;
#endif
#else
    osgGA::MatrixManipulator* getCameraManipulator();
    const osgGA::MatrixManipulator* getCameraManipulator() const;
#endif
    agxSDK::Simulation *getSimulation() { return m_simulation.get(); }

    bool restoreFromFile( const std::string& filename );
    void journalRenderLoader(agxSDK::Assembly *assembly, bool keyFrame);


    /// Should the application step automatically? False for 'pause', true for 'play'.
    void setAutoStepping( bool flag );

    /// Should the application step automatically? False for 'pause', true for 'play'.
    bool getAutoStepping( ) const;


    void setEnablePauseUpdate( bool flag ) { m_pauseUpdate = flag; }
    bool getEnablePauseUpdate( ) const { return m_pauseUpdate; }

    /**
    Should ExampleApplication try to hold simulation in real time?
    Will wait for wall clock if going to fast, but might still go too slow
    if simulation or rendering time take too long.
    */
    void setRealTimeSync(bool flag);
    bool getRealTimeSync() const;

    agx::Journal *getJournal();
    const std::string& getJournalConfigurationPath() const;
    const std::string& getJournalPlaybackPath() const;

    void setJournalConfigurationPath(const agx::String& path);

    void setEnableJournalIncrementalStructure( bool enable );
    bool getEnableJournalIncrementalStructure() const;

    SceneDescription getCallbackIdx( size_t idx );

    SceneDescription getCallback( int key );
    bool createSceneFromKey( int key );
    bool createSceneFromIndex( int idx, bool firstStartUp = false );


    void reloadScene();
    void createNextScene();
    void createPreviousScene();

    size_t getNumScenes() const;

    void step();

    void changeSolver();

    const std::string& getSaveSceneFilename() const { return m_sceneFilename; }

    osg::Group *getRoot() { return m_root.get(); }

    void fitSceneIntoView();
    void initSimulation(agxSDK::Simulation *simulation = nullptr, bool initializeGraphics = true);
    void initViewer(int width, int height, bool osgWindow = true);

    void initGraphics();
    void initRpc();
    bool readCFGFile();

    /**
    Lets the application stop after a certain simulation time.
    Same as using --stopAfter command line argument.
    \param stopTime Simulation time after which the application should stop.
    */
    void stopAfter(const agx::Real stopTime);

    osgViewer::Viewer *getViewer() { return m_viewer.get(); }
    osg::Camera *getCamera() { if (m_viewer.valid()) return m_viewer->getCamera(); else return nullptr; }

    bool hasOffscreenWindow() const;

    /**
    \return the camera data for main camera
    */
    agxOSG::CameraData getCameraData() const;

    /**
    Apply camera data. Typically: camData = app->getCameraData(); camData.nearClippingPlane = 0.01f; app->applyCameraData( camData );
    */
    void applyCameraData( const agxOSG::CameraData& cameraData );

    intptr_t getHWND() const;

    agx::Vec3 getViewDirection();

    PickHandler* getPickHandler() { return m_pickHandler; }
    const PickHandler* getPickHandler() const { return m_pickHandler; }

    std::string getScriptFile() const { return m_scriptFile; }

    /// Get file path to most recent lua file.
    std::string getLuaFilePath() const
    {
      return m_luaFilePath;
    }

    void handleReactiveScriptErrors( bool abortSimulation = true );

    agx::Real getTimeStamp() const;

    void clearScene();

    void setEnableSimulationDump( bool flag );
    bool getEnableSimulationDump( ) const;

    void updateCoSimulationServer();
    bool updatePythonCoSimulationServer(const agx::String& filename, std::stringstream& buffer);
    bool coSimulationCallPython(agxNet::CoSimulationServer* server);

    void updateRemoteDebugger();
    void centerScene();
    void stepSimulation();

    bool useCoSimulation() const {return m_useCoSimulation;}

    agx::Vec3 getGravity() const { return m_gravity; }


    agxOSG::ImageCapture *getImageCapture() { return m_imageCapture; }
    const agxOSG::ImageCapture *getImageCapture() const { return m_imageCapture; }

    agxOSG::VideoFFMPEGPipeCapture *getVideoServerCapture() { return m_videoCapture; }

    osg::Camera *getTextureCamera() { return m_textureCamera; }


    bool isCameraHomeSet() const { return m_cameraHomeSet; }

    static void registerRemovedGeometry(GeometryNode *node);

    void storeInitialJournalState();
    void initRecordJournal();

    void journalPlayback();
    void listSessionNames(const std::string& journalToList);

    /**
    Sets an orbit camera following an node.
    \param node The node.
    \param heading The camera heading.
    \param elevation The camera elevation.
    \param distance The camera distance
    \param trackerMode See osgGA::NodeTrackerManipulator::TrackerNode.
    */
    void setOrbitCamera(agxOSG::GeometryNode* node,
      double heading, double elevation, double distance,
      int trackerMode = 0);

    /**
    Sets an orbit camera following an node.
    \param node The node.
    \param eye The camera eye.
    \param center The camera center relative geometry center.
    \param up The camera up vector
    \param trackerMode See osgGA::NodeTrackerManipulator::TrackerNode.
    */
    void setOrbitCamera(agxOSG::GeometryNode* node,
      const agx::Vec3& eye, const agx::Vec3& center, const agx::Vec3& up = agx::Vec3::Z_AXIS(),
      int trackerMode = 0);

    bool executeOneStepWithGraphics();
    bool executeOneStepWithoutGraphics();

    void updateServices();


    /**
    If no graphics is used, this method always return false
    \return true when ESC key has been pressed in the graphics window, indicating that the application is shutting down.
    */
    bool done() const;


    /**
    If set to true (default) pressing the ESC key will exit the run loop and application will shut down.
    If false, pressing esc will not quit the application.
    */
    void setQuitEventSetsDone(bool flag);
    bool getQuitEventSetsDone() const;

    /**
    Is window resizing allowed?
    */
    bool getAllowWindowResizing() const;

    /**
    Set if window resizing should be allowed.
    */
    void setAllowWindowResizing(bool flag);

    /**
    Takes a screen shot of the scene.
    If filename is left empty, it will default to "agx_screen_%05d.png".
    */
    void takeScreenShot(const agx::String& filename="");

    void synchronizedStep(bool blocking = false);

    void stepWrapper();

    void initScene();
    void initScene(osg::Group *root);


    /// Gets journal format.
    agx::UInt getJournalFormat() const;

    /// Sets journal format. Only valid before starting journal recording/playback.
    void setJournalFormat(agx::UInt journalFormat);

    /// Set to true if journal should save data in 32bit float.
    void setEnableJournal32bitMode(bool enable);

    /// Set if a journal should be recorded from the simulation
    void setEnableJournalRecord(bool enable);

    /// Set the requested recording frequency for attached journal
    void setRequestedJournalFrequency(agx::Real freq);

    /// Set the path for the recorded journal
    void setJournalRecordPath(const agx::String& journalPath);

    /// Sets the transformation for the visual coordinate system and grid.
    void setCoordinateSystemTransform( const agx::AffineMatrix4x4& coordinateSystemTransform);

    /// Gets the transformation for the visual coordinate system and grid.
    agx::AffineMatrix4x4 getCoordinateSystemTransform() const;

    /// Sets the grid size for the visual grid.
    void setGridSize(const agx::Vec2& gridSize);

    /// Gets the grid size for the visual grid.
    agx::Vec2 getGridSize() const;

    /// Sets the grid resolution for the visual grid.
    void setGridResolution(const agx::Vec2u gridResolution);

    /// Gets the grid resolution for the visual grid.
    agx::Vec2u getGridResolution() const;

    /// Enables/disables the visual grid.
    void setEnableGrid(bool flag);

    /// Is the visual grid enabled/disabled?
    bool getEnableGrid() const;

    /// Enables/disables the visual coordinate system.
    void setEnableCoordinateSystem(bool flag);

    /// Is the visual coordinate system enabled/disabled?
    bool getEnableCoordinateSystem() const;

    // Throttle calls to avoid excessive CPU usage
    // Is automatically called from `executeWithoutGraphics`
    void throttleIdleCpuUsageBySleeping();

    agxOSG::RenderToTexture * getVideoCaptureRendertoTexture() { return m_videoCaptureRenderToTexture; }

    std::string getProfilingJournalPath() const;

#if AGX_USE_WEBSOCKETS()
    agxOSG::ExampleApplicationController *getController();

    agx::UInt16 getControlChannelPort();
    void pushParametersToControlChannel();
    void setEnableControlChannelTickSignals(bool flag);
    void pushFrameToRemoteViewer();
#endif

    void postExecution();
    void setEnableThreadTimeline(bool flag);
    void setEnableTaskProfile(bool flag);
    int getExitCode() { return m_exitCode; }

#if AGX_USE_PYTHON()
    agxPython::ScriptContextInterface* initPythonContext(agxPython::ScriptManager *scriptManager);
#endif

    void createVisual(agxSDK::Simulation *simulation, float detailRatio = 1.0f, bool createAxes = false);
    void createVisual(agxSDK::Assembly *assembly, float detailRatio = 1.0f, bool createAxes = false);
    agx::Mutex& getFrameMutex() { return m_frameMutex; }

  protected:

    virtual ~ExampleApplication();

    void drawGrid();
    void drawCoordinateSystem();

    void granularCreated( agxSDK::Simulation* sim, agx::RigidBody* body );
    void addGranularCreateVisualCallbacks();

  protected:
    class SimulationListener;

    void execute();
    void executeWithoutGraphics();
    bool executeOneStepWithoutGraphics(bool& saveAfterEnabled, agx::Real saveAfterTime, agx::HighAccuracyTimer& timer);


    void executeWithGraphics();
    bool executeOneStepWithGraphics(bool& saveAfterEnabled, agx::Real saveAfterTime, agx::HighAccuracyTimer& timer);

    void generateQuickProfiling();
    void journalSceneLoader(const agx::String& path);
    void removeGeometryNode(agxSDK::Simulation*, agxCollide::Geometry*);
    void setArgumentsPostSceneCreation(); // This function will override a number of settings with their command line argument specifications
    agxJson::Value extractSimulationStructureToJson();
    void applyInitialParameters(const std::string& filePath);
    void initThreadTimeline();
    bool initBrickInterop();

    void triggerPreFrameListeners();
    void triggerPostFrameListeners();

    bool attachScripts( osg::Group* root );
    osg::Group *executeLuaScript( const agx::String& file, const agx::String& function, bool& success );
    osg::Group *executePythonScript( const agx::String& file, const agx::String& function, bool& success );
    osg::Group *loadBrickModel( const agx::String& file, const agx::String& model, bool& success );
    osg::Group *executeMPyScript(const agx::String& file, bool& success);

    osg::Group *createScene( const SceneDescription& desc, bool &success );

    osg::ref_ptr<agxOSG::SceneDecorator> m_decorator;
    agxSDK::SimulationRef m_simulation;
    osg::ref_ptr<agxOSG::GuiEventAdapter> m_eventAdapter;
    osg::ref_ptr<osgViewer::Viewer> m_viewer;

    agxOSG::ImageCaptureRef m_imageCapture;

    osg::ref_ptr<osg::Group> m_root;
    osg::ref_ptr<osg::Group> m_sceneRoot;
    osg::ref_ptr<osg::Group> m_scene;
    osg::ref_ptr<osg::Switch> m_sceneSwitch;

    bool m_osgWindow;

    int m_sceneIndex;

    typedef agx::HashTable<int, int> KeyBindings;

    typedef agx::Vector<SceneDescription> KeyBindingsVector;

    KeyBindings m_keyBindings;
    KeyBindingsVector m_keybindingVector;

    SolverType m_solverType;
    agxIO::ArgumentParserRef m_arguments;

    std::string m_sceneFilename;
    std::string m_parameterFilePath;

    bool m_initialized;
    std::string m_cfgFileName;

    mutable agx::Real m_timestep;

    bool m_automaticStop;
    bool m_realTime;
    agx::Real m_relativeStopTime;
    agx::Real m_stopTime;
    agx::UInt m_stopFrame;
    bool m_newSimulation;
    bool m_autoCycleThroughScenes;
    bool m_agxOnly;
    StatisticsRendererRef m_statisticsRenderer;
    // std::string m_exportMayaAnimationPath;

#if AGX_USE_WEBSOCKETS()
    friend class ExampleApplicationController;
#endif
    friend class AutoPauseListener;
    bool m_autoPausingEnabled;
    agx::HashSet<agxCollide::GeometryPair> m_autoPausePairs;
    agxSDK::ContactEventListenerRef m_autoPauseListener;

    PickHandlerRef m_pickHandler;

    agxCFG::ConfigScriptRef m_settings;

    std::string m_xmlPlotFile;
    bool m_shouldLoadXmlPlot;

    std::string m_scriptFile;
    bool m_shouldStop;
    int m_exitCode;
    agx::ref_ptr<agx::Referenced> m_coSimulationServer;
    agx::ref_ptr<agx::Referenced> m_debugClient;

    agx::Vec3 m_gravity;
#if AGX_USE_OPENGL()
    agxGL::CameraRef m_GlCamera;
    agxGL::LightsRef  m_GlLights;
#endif
    bool m_cameraHomeSet;
    agx::Vec3 m_cameraHomeEye;
    agx::Vec3 m_cameraHomeCenter;
    agx::Vec3 m_cameraHomeUp;
    bool m_quickProfiling;
    bool m_generateThreadTimeline;
    std::string m_profilingRoot;
    agx::Vector<agx::String> m_timelineFormats;
    bool m_generateTaskProfile;
    bool m_useCoSimulation;
    std::string m_coSimulationInputFile;
    std::string m_coSimulationOutputFile;
    agx::RealVector m_coSimulationInputData;
    agx::RealVector m_coSimulationOutputData;
    bool m_vSync;
    std::string m_luaFilePath;
    std::string m_pythonFilePath;
    std::string m_journalPlaybackPath;
    std::string m_journalConfigurationPath;
    bool m_explicitJournalConfigurationPath;
    agx::UInt m_journalFormat;
    agx::Real m_journalStartTime;
    agx::Real m_journalStopTime;
    std::string m_journalEofMode;
    std::string m_journalRecordPath;
    std::string m_journalSessionName;
    std::string m_extractSimulationStructureToJsonPath;
    agx::Real m_journalRecordStartTime;
    // agx::UInt m_journalStride;
    agx::Real m_journalFrequency;
    agx::RealModeEnum m_journalRealMode;
    agx::ref_ptr<agx::Referenced> m_journal;

    agx::Vector<std::pair<std::string, int> > m_attachedScripts;
    agx::ref_ptr<agx::Referenced> m_controller;

    agxSDK::Simulation::RigidBodyEvent::CallbackType m_addBodyCallback;

#ifdef AGX_HAVE_DEPTHPEELING
    osg::ref_ptr<DepthPeeling> m_depthPeeling;
#endif
    agxOSG::RenderProxyFactory* m_renderProxyFactory;
    bool m_enableStepping;
    int m_startScene;
    agxSDK::Simulation::GeometryEvent::CallbackType m_removeGeometryCallback;
    osg::ref_ptr<osg::Camera> m_textureCamera;
    agx::Mutex m_frameMutex;
    agx::Mutex m_serviceMutex;
    agx::Block m_frameBlock;
    bool m_haveLicense;
    bool m_allowWindowResizing;
    agx::Real m_dt;
    agx::Real m_realTimeRest;
    agx::Timer m_stepTimer;
    agx::Timer m_frameSleepTimer;
    bool m_didStep;
    RenderTargetRefVector m_renderTargetSet;
    osg::ref_ptr<osg::GraphicsContext> m_pBuffer;
    osg::ref_ptr<osg::Camera> m_pBufferCamera;
    agx::Real m_profilingFrequency;

  public:
    bool m_loadedSceneFile;
    bool m_isLoadingScene;
    bool m_isFMU;
    agx::Event m_sceneLoadEvent;
    std::string m_hostName;

  protected:
    agx::AffineMatrix4x4 m_coordinateSystemTransform;
    agx::Vec2u m_gridResolution;
    agx::Vec2 m_gridSize;
    bool m_coordinateSystemEnabled;
    bool m_gridEnabled;
    osg::ref_ptr<osg::Geode> m_grid;
    osg::ref_ptr<osg::MatrixTransform> m_coordinateSystem;
    bool m_pauseUpdate;
    bool m_enableJournalRecording;
    bool m_enableJournalIncrementalStructure;
    bool m_interactiveRemoteClient;
    bool m_explicitFmu;
    agx::Callback m_stepCallback;

    std::string m_videoName;
    VideoFFMPEGPipeCaptureRef    m_videoCapture;
    RenderToTextureRef m_videoCaptureRenderToTexture;
    agx::Vector<ExampleApplicationListenerRef> m_listeners;

    agxOSG::RigidBodyRenderCacheRef m_cache;
  };

}
