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

#ifndef AGXQT_QTVIEWERMAINWINDOW_H
#define AGXQT_QTVIEWERMAINWINDOW_H

#include <agxQt/export.h>
#include <agx/config/AGX_USE_KEYSHOT.h>
#include <agxQt/QtAgxWrapper.h>
#include <agxQt/UtilWidgets.h>
#include <agxQt/OsgRenderer.h>
#include <agxQt/PlaybackController.h>
#include <agxQt/DataExporterDialog.h>
#include <agxQt/CameraListWidget.h>
#include <agxQt/ClipPlaneWidget.h>
#include <agxQt/VideoCaptureSettingsWidget.h>
#include <agxQt/KeyshotExporterDialog.h>
#include <agxQt/AnalysisBoundWidget.h>
#include <agxQt/SimulationStructureWidget.h>
#include <agxQt/DockWidgetWithCloseSignal.h>
#include <agxQt/JournalInfoWidget.h>

#include <agx/PushDisableWarnings.h> // Disabling warnings. Include agx/PopDisableWarnings.h below!
#include <agxQt/ui_QtViewerMainWindow.h>
#include <QMainWindow>
#include <QWidget>
#include <QOpenGLContext>
#include <QOpenGLFunctions>
#include <QObject>
#include <agx/PopDisableWarnings.h> // End of disabled warnings.

namespace agxQt
{
  class ParticleRenderWidget;

  //////////////////////////////////////////////////////////////////////////
  // QtAgxViewerMainWindow
  //////////////////////////////////////////////////////////////////////////
  /**
  * Main window of the agx application
  */
  class AGXQT_EXPORT QtAgxViewerMainWindow : public QMainWindow
  {
    Q_OBJECT

  public:
    enum RESOLUTION_MODE
    {
      RES_HD = 0,
      RES_FULL_HD = 1,
      RES_4K=2
    };

  public:
    explicit QtAgxViewerMainWindow(QWidget *parent = 0);
    virtual ~QtAgxViewerMainWindow();
    OsgRenderer* getRenderer();

    QtAgxWrapper* getWrapper() { return m_qtagxWrapper; }

    bool init(agxIO::ArgumentParser* arguments);

    bool load(const std::string& fn);

    void expandWindow( int deltaX, int deltaY );

  signals:
    void signalChangeDrawMode(int mode);
    void signalChangeParticleShaderMode(int mode);
    void signalChangeBackgroundColor(int mode);

    public slots:
    // Main
    void slotCloseApplication();
    void testSlot();
    void slotEnableCapture(bool enable);
    void slotTakeScreenShot();

    // Simulation
    void slotJournalTrackReachedEnd();
    void slotPressCaptureButton();
    void slotRefreshRendering();

    private slots:
    void slotOpenFile(const std::string& overrideFilter = "");
    void slotSaveSimulationState();
    void slotOnSimulationLoaded();

    // Camera config slots
    void slotOpenConfigFile() { slotOpenFile("Camera configuration (*.cfg)"); }
    void slotSaveCameraViews();
    void slotUpdateCameraViewWidget();
    void slotToggleShowCameraViews(bool);

    void slotToggleModePolygon();
    void slotToggleModeWireframe();
    void slotToggleModePoints();
    void slotCyclePolygonMode();

    void slotToggleViewSideDocket(bool);

    void slotChangeParticleShaderSprites();
    void slotChangeParticleShaderRotationalSprites();
    void slotChangeParticleShaderAlphaSprites();
    void slotIncrementParticleAlpha();
    void slotDecrementParticleAlpha();
    void slotUpdateParticleRenderWidget();
    void slotUpdatePostProcessingBound();
    void slotOpenDataExporterDialog();


    void slotSetResolutionHD();
    void slotSetResolutionFullHD();
    void slotSetResolution4K();
    void slotSetResolutionMode( int mode );

    void slotToggleBackgroundColor( int state );
    void slotToggleBackgroundGrayGradient();
    void slotToggleBackgroundWhite();
    void slotToggleBackgroundWhiteGradient();
    void slotToggleBackgroundSkyBlue();
    void slotToggleBackgroundLightBlue();
    void slotToggleBackgroundBlack();

    void slotEnableUseAlgoryxLogo(bool enable);

    // Simulation Structure
    void slotUpdateSimulationStructure();

    // Resize
    void slotSetEnableResize(bool shouldResize);
    void slotToggleOsgFullscreen();

    // media generation
    void slotShowMediaWidget();
    void slotGenerateVideo();

    // Keyshot
    void slotExportCurrentFrameToBip();
    void slotExportWholeJournalToBipFiles();
    void slotUpdateKeyshotExporterDialog();
    void slotOpenKeyshotExporterDialog();

    // Clip plane updates
    void slotTranslateClipPlane();
    void slotDeTranslateClipPlane();

    // Contact Exporter
    void slotTranslateBoundX();
    void slotDeTranslateBoundX();
    void slotTranslateBoundY();
    void slotDeTranslateBoundY();
    void slotUpdateContactExporterWidget();
    void slotEnableImpactPostProcessing(bool enable);

    // Info
    void slotOpenJournalInfo();

  protected:
    void setFilenameTitle(const std::string& filename);
    void setupGui();
    void initConnectionsGui();
    void initConnectionsActions();
    void togglePolgyonRendering(int state);

    void setupCameraViewMenu();
    void updateSimulationStructure();

    virtual void keyPressEvent(QKeyEvent* event);
    virtual void keyReleaseEvent(QKeyEvent* event);
    virtual void resizeEvent(QResizeEvent* event);

  private:
    void closeEvent(QCloseEvent * event);

    //////////////////////////////////////////////////////////////////////////
    // Variables
    //////////////////////////////////////////////////////////////////////////

  protected:
    // GUI
    Ui::QtViewerMainWindow * m_ui;

    // agxQT
    std::unique_ptr<agxQt::OsgRenderer>                  m_renderer;
    std::unique_ptr<agxQt::PlaybackControllerWidget>     m_controllerWidget;
    std::unique_ptr<agxQt::CameraListWidget>             m_cameraListWidget;
    std::unique_ptr<agxQt::VideoCaptureSettingsWidget>   m_videoSettingsWidget;
    std::unique_ptr<agxQt::ParticleRenderWidget>         m_particleRenderWidget;
#if AGX_USE_KEYSHOT()
    std::unique_ptr<agxQt::KeyshotExporterDialog>        m_keyshotExporterDialog;
#endif
    std::unique_ptr<agxQt::DataExporterDialog>           m_contactWriterDialog;
    std::unique_ptr<agxQt::AnalysisBoundWidget>          m_analysisBoundWidget;
    std::unique_ptr<agxQt::ClipPlaneWidget>              m_clipPlaneWidget;
    agx::ref_ptr<QtAgxWrapper>                           m_qtagxWrapper;
    std::unique_ptr<agxQt::SimulationStructureWidget>    m_treeStructureWidget;
    std::unique_ptr<agxQt::JournalInfoWidget>            m_journalInfoWidget;
    bool                                                 m_isFullScreen;

    // Qt
    QDockWidget*               m_playbackDockWidget;
    DockWidgetWithCloseSignal* m_cameraListDockWidget;
    QWidget*                   m_recordWidget;
    QDockWidget*               m_leftDockWidget;

    // Qt custom actions
    QAction* m_centerSceneAction;
    QAction* m_cyclePolyModeAction;
    QAction* m_pauseAction;
    QAction* m_stepForwardAction;
    QAction* m_stepBackAction;
    QAction* m_incrementSpeedAction;
    QAction* m_decrementSpeedAction;
    QAction* m_saveSimulationStateAction;
  };
}

#endif
