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

#ifndef AGXQT_QTAGXWRAPPER_H
#define AGXQT_QTAGXWRAPPER_H

#include <agx/config/AGX_USE_LUA.h>
#include <agx/config/AGX_USE_PYTHON.h>
#include <agxQt/export.h>
#include <agx/Math.h>

#include <agx/PushDisableWarnings.h> // Disabling warnings. Include agx/PopDisableWarnings.h below!
#include <QObject>
#include <QTimer>
#include <QElapsedTimer>
#include <osgGA/TrackballManipulator>
#include <osgGA/StateSetManipulator>
#include <osgViewer/ViewerEventHandlers>
#include <osgViewer/Renderer>
#include <agx/PopDisableWarnings.h> // End of disabled warnings.

#include <agxQt/UtilWidgets.h>
#include <agxQt/QtOSGUtils.h>
#include <agxQt/export.h>
#include <agxQt/VideoCapture.h>
#include <agxQt/CameraViewSet.h>

#include <agxSDK/SimulationController.h>
#include <agxCFG/ConfigScript.h>
#include <agx/ConfigSingleton.h>
#include <agxIO/ArgumentParser.h>
#include <agxOSG/ExampleApplication.h>
#include <agxSDK/Simulation.h>

#include <agxOSG/SimulationObject.h>
#include <agx/Journal.h>
#include <agx/AnalysisBox.h>

#include <agxOSG/RenderProxy.h>
#include <agxOSG/DepthPeeling.h>
#include <agxOSG/ClipPlane.h>
#include <agxOSG/SimulationDrawable.h>
#include <agxQt/GranularImpactsDrawable.h>

#if AGX_USE_LUA()
#  include <agxLua/ScriptManager.h>
#  include <agxLua/luaUtils.h>
#  include <agxLua/LuaCall.h>
#endif

namespace agxQt
{
  class QtAgxWrapperStepCallback;
  /**
  * Class that wraps SimulationController with an interface that enables QT Gui signals to control the simulation controller
  * and it's contents.
  */
  class AGXQT_EXPORT QtAgxWrapper : public QObject, public agx::Referenced
  {
    Q_OBJECT

    friend class QtAgxWrapperStepCallback;
    friend class agxQt::VideoCapture;

  public:
    // Convenience enum for mapping background color to int
    enum BACKGROUND_COLOR
    {
      GRAY_GRADIENT=0,
      WHITE=1,
      WHITE_GRADIENT=2,
      SKY_BLUE=3,
      LIGHT_BLUE = 4,
      BLACK=5,
      NUM_COLORS=6
    };

  public:
    QtAgxWrapper();

    /// Init
    bool init(osgViewer::Viewer* viewer, agxIO::ArgumentParser* arguments);

    /// Loads a file into the wrapper
    bool loadFile(const std::string& filename);

    /// Reloads the active loaded scene
    void reloadActiveScene();

    /// Set the viewer to sync the simulation content to
    void setViewer(osgViewer::Viewer* viewer);
    osgViewer::Viewer* getViewer();

    /// Returns pointer to osg structure of the scene
    void setScene(osg::Group* scene);
    osg::Group* getScene();

    /// Returns the description of the active scene
    agxOSG::SceneDescription getSceneDescription() const { return m_sceneDescription; }

    /// Get Scene decorator
    agxOSG::SceneDecorator* getSceneDecorator();

    /// Get the camera view manager
    agxQt::OsgCameraViewManager* getCameraViewManager();

    // Return the particle render utils object
    agxQt::ParticleSystemRenderUtility* getParticleRenderUtillity();

    // Return the trajectory renderer
    agxQt::TrajectoryRenderer* getTrajectoryRenderer();

    /// Set the time step used in the simulation
    void setTimeStep(float dt);

    /// Get the current timestamp of the simulation.
    agx::TimeStamp getTimeStamp();

    /// Returns the active simulation in the wrapper. This is kind of a "hack" to let the renderer get a simulation lock to sync rendering with stepping.
    static agxSDK::SimulationController* getActiveSimulation();

    /// Returns the size of the render window
    agx::Vec2u getRenderWindowSize() const;

    /// Image capturing for video
    void startCapture();
    void stopCapture();

    /// Take screenshot
    void takeScreenShot();

    /// Save simulation state to file
    bool saveSimulationState(const std::string& filename);

    /// Save the active camera view set to a file
    bool writeCameraViewsToFile(const std::string& filename);

    agxQt::VideoCapture* getVideoCapture();

    agxSDK::Simulation* getSimulation();

    agx::String getCurrentSessionName() const;

    bool isSimulationRunning() const;

    agx::AnalysisBox* getAnalysisBox();

    agxOSG::ClipPlane* getClipPlane();
    agxQt::ClipPlaneController* getClipPlaneController();

    bool journalContainsImpactData();

    bool attachScripts(osg::Group*root);

    void finalizeAndResetVideoCapture();

    agx::Real getSimulationUpdateTimeStep();

    osg::Group* getRoot() { return m_root; }

    agxOSG::RigidBodyRenderCache* getRenderCache();

  signals:
    // simulation
    void signalSimulationUpdate(float timeStamp);
    void signalNewPlayback(float timeStart, float timeEnd, uint numFrames);
    void signalNewSimulation();
    void signalSimulationPause();
    void signalJournalTrackReachedEnd();
    void signalJournalContainsImpactData(bool hasData);

    void signalUpdateSimulationStructure();

    void signalActiveCameraViewChanged();

    // Renderer
    void signalRenderUpdate();

    void signalNewSimulationLoaded();

    public slots:
    // The main loop of the wrapper
    void slotUpdateLoop();

    // Play control
    void slotPauseSimulation();
    void slotPlaySimulation();
    void slotStepForwardSimulation();
    void slotStepBackSimulation();
    void slotStopSimulation();
    void slotChangeSimulationTime(float);
    void slotJumpToEnd();
    void slotJumpToStart();
    void slotSetJournalStride(uint stride);
    void slotEnableJournalLooping(bool enable);

    // OSG
    void slotEnableDebugRenderMode(bool);
    void slotEnableOSGRenderMode(bool);
    void slotEnbleDrawStatistics(bool);
    void slotChangeDrawMode(int mode);
    void slotCenterScene();
    void slotChangeParticleShaderMode(int mode);
    void slotChangeAlpha(agx::Real alpha);
    void slotSetEnableShaders(bool enable);
    void slotChangeBackgroundColor( int mode );
    void slotSetEnableRenderDataRendering(bool enable);
    void slotSetEnableCollisionDataRendering(bool enable);
    void slotMoveLightToLookAt();
    void slotUpdateAnalysisBoxDrawable();
    void slotUpdateClipplingPlaneDrawable();
    void slotHandleStopSimulation();

    // Camera functions
    void slotIncrementCamera();
    void slotDecrementCamera();
    void slotStoreCurrentViewInCameraSet();
    void slotSetViewFromActiveCameraDescription();
    void slotIncrementClipPlane();
    void slotDecrementClipPlane();

    // Simulation
    void stepSimulation();
    void slotClearSimulation();
    void slotReloadSimulation();
    void slotUpdateSpace();

  private:

  protected:
    virtual ~QtAgxWrapper();
    void loadDefaultCameraConfig(const agx::String& filename);
    void setSceneDescription(const agxOSG::SceneDescription& desc);
    void lockSimulationAndTriggerEvents(); // See comments in function to see purpose
    void handleArguments(agxIO::ArgumentParser* arguments);
    void handleFileNameArguments(agxIO::ArgumentParser* arguments);
    void journalPlayback(const std::string& journalPlaybackPath, const std::string asessionName = "");
    void clearJournalSessionData();
    void journalSceneLoader(const agx::String& path);
    void journalRenderLoader(agxSDK::Assembly* assembly, bool keyFrame);
    bool restoreFromFile(const std::string& file);
    bool executeLuaScript(const agx::String& file, const agx::String& function);
    bool executePythonScript(const agx::String& file, const agx::String& function);
    bool executeMPyScript(const agx::String& file);
    void initSimulationController();
    void setupViewer();
    void fitSceneIntoView();
    void clearSimulation();
    void clearScene();
    bool loadActiveScene();
    osg::Group* createScene(const agxOSG::SceneDescription& desc, bool &success);
    bool readCFGFile();
    void setupUpdateLoop();
    void handleJournalLooping();
    bool playbackIsAtEnd();
    void setupPostProcessor();
    void setEnableDataRenderFlag(bool enable, agxOSG::GeometryNode::RenderFlag flag);
    void updateAnalysisBox();
    void triggetStopInScripts();

    /// Static helper function to get active simulation controller so that other classes can lock the active simulation. The osg renderer is such an example
    static void setActiveSimulationController(agxSDK::SimulationController* controller);

    //////////////////////////////////////////////////////////////////////////
    // Variables
    //////////////////////////////////////////////////////////////////////////
  private:
    agx::Block m_waitForSimulationStepBlock;

  protected:
    agxSDK::SimulationControllerRef      m_simulationController;

    agxCFG::ConfigScriptRef              m_settings;
    agxOSG::SceneDescription             m_sceneDescription;
    agx::StringVector                    m_attachedScripts;
    std::string                          m_pythonFilePath;

    // Journal session related variables
    agx::UInt   m_journalStride;
    std::string m_journalSessionName;
    std::string m_currentCaptureFolder;

    std::string m_cfgFileName;
    std::string m_luaFilePath;

    bool m_initalized;
    bool m_simulationPaused;
    bool m_loopJournal;

    // Image capture
    agxQt::VideoCaptureRef m_videoCapture;

    // Camera handler
    OsgCameraViewManagerRef m_cameraManager;

    // Particle post processing
    ParticleSystemRenderUtilityRef           m_particleRenderUtil;
    // Trajectory Renderer
    TrajectoryRendererRef                    m_trajectoryRenderer;

    // Analysis Box
    agx::AnalysisBoxRef                      m_analysisBox;
    osg::ref_ptr<osg::Group>                 m_analysisBoxDrawable;

    // Clip Plane
    osg::ref_ptr < agxOSG::ClipPlane >       m_clipPlane;
    agxQt::ClipPlaneControllerRef            m_clipPlaneController;
    osg::ref_ptr<osg::Group>                 m_clipPlaneDrawable;

    // OSG variables
    agxOSG::PickHandlerRef                   m_pickHandler;
    agxOSG::RenderProxyFactory*              m_renderProxyFactory;
    osg::ref_ptr<osgViewer::Viewer>          m_viewer;
    osg::ref_ptr<agxOSG::SceneDecorator>     m_decorator;
    osg::ref_ptr<agxOSG::GuiEventAdapter>    m_eventAdapter;
    osg::ref_ptr<osg::Group>                 m_root;
    osg::ref_ptr<osg::Group>                 m_postProcessRoot;
    osg::ref_ptr<osg::Group>                 m_sceneRoot;
    osg::ref_ptr<osg::Group>                 m_scene;
    osg::ref_ptr<osg::Switch>                m_sceneSwitch;
    osg::ref_ptr<osgGA::StateSetManipulator> m_ssm;
    agxGL::CameraRef                         m_GlCamera;
    agxGL::LightsRef                         m_GlLights;

    agxOSG::RigidBodyRenderCacheRef          m_renderBodyCache;

    // Static pointer to active simulation that is used to lock simulation while viewer is rendering the scene.
    static agxSDK::SimulationController* s_activeController;
    QTimer m_updateTimer;
    int m_updates;
  };

  // Implementation
  AGX_FORCE_INLINE void QtAgxWrapper::setViewer(osgViewer::Viewer* viewer) { m_viewer = viewer; }

  AGX_FORCE_INLINE osgViewer::Viewer* QtAgxWrapper::getViewer() { return m_viewer; }

  AGX_FORCE_INLINE agxOSG::SceneDecorator* QtAgxWrapper::getSceneDecorator() { return m_decorator; }

  AGX_FORCE_INLINE agxSDK::Simulation* QtAgxWrapper::getSimulation() { return m_simulationController ? m_simulationController->getSimulation() : nullptr; }

  AGX_FORCE_INLINE void QtAgxWrapper::setScene(osg::Group* scene) { m_scene = scene; }

  AGX_FORCE_INLINE void QtAgxWrapper::setTimeStep(float dt) { m_simulationController->getSimulation()->setTimeStep(dt); }

  AGX_FORCE_INLINE osg::Group* QtAgxWrapper::getScene() { return m_scene.get(); }

  AGX_FORCE_INLINE agxSDK::SimulationController* QtAgxWrapper::getActiveSimulation() { return s_activeController; }

  AGX_FORCE_INLINE void QtAgxWrapper::setActiveSimulationController(agxSDK::SimulationController* controller) { s_activeController = controller; }

  AGX_FORCE_INLINE void QtAgxWrapper::setSceneDescription(const agxOSG::SceneDescription& desc) { m_sceneDescription = desc; }

  AGX_FORCE_INLINE agxQt::OsgCameraViewManager* QtAgxWrapper::getCameraViewManager() { return m_cameraManager.get(); }

  AGX_FORCE_INLINE  agxQt::ParticleSystemRenderUtility* QtAgxWrapper::getParticleRenderUtillity() { return m_particleRenderUtil.get(); }

  AGX_FORCE_INLINE  agxQt::TrajectoryRenderer* QtAgxWrapper::getTrajectoryRenderer() { return m_trajectoryRenderer.get(); }

  AGX_FORCE_INLINE  bool QtAgxWrapper::isSimulationRunning() const { return !m_simulationPaused; }

  AGX_FORCE_INLINE  agx::AnalysisBox* QtAgxWrapper::getAnalysisBox() { return m_analysisBox; }
}

#endif