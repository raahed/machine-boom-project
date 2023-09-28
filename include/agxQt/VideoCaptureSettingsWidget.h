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

#ifndef AGXQT_VIDEOCAPTURESETTINGSWIDGET_H
#define AGXQT_VIDEOCAPTURESETTINGSWIDGET_H

#include <agx/PushDisableWarnings.h> // Disabling warnings. Include agx/PopDisableWarnings.h below!
#include <QWidget>
#include <QProcess>
#include <osgViewer/ViewerEventHandlers>
#include <osgViewer/Renderer>
#include <agxQt/ui_VideoCaptureSettingsWidget.h>
#include <agx/PopDisableWarnings.h> // End of disabled warnings.

#include <agxQt/export.h>
#include <agxQt/VideoCapture.h>
#include <agx/Referenced.h>

#define DEFAULT_VIDEO_FPS 30
#define DEFAULT_CAPTURE_FPS 30

#define QUALITY_INTERVAL 20
#define DEFAULT_QUALITY_TICK 17

namespace agxQt
{
  class AGXQT_EXPORT VideoCaptureSettingsWidget : public QWidget
  {
    Q_OBJECT

  public:
    explicit VideoCaptureSettingsWidget(QWidget *parent = 0);
    virtual ~VideoCaptureSettingsWidget();

    /// Set initial data to the inputs in the widget
    void setData(
      agxQt::VideoCapture* vc,
      agx::Real simulationTimestep);

    /// Get output name for the video
    std::string getVideoOutputName();

    /// Get FPS from input
    agx::Real getVideoFPS() const;

    /// Get real time factor
    agx::Real getRealTimeFactor() const;

    /// Returns the quality value from the video [0-1]
    agx::Real getVideoQuality();
    void setVideoQuality(agx::Real quality);

    void updateVideoCapture();

signals:
    void signalGenerateMovie();
    void signalSynchMediaSettingsWithWrapper();

  protected:
    void setOutputPath();
    void synchLabels();

  protected slots:
    // Tab - Video slots
    void slotFPSEditFinished();
    void slotEditRealTimeFactor();
    void slotSetOupoutPathButtonPressed();
    void slotFinalizePressed();
    void slotCloseWidget();
    void slotQualitySliderChanged();

  //////////////////////////////////////////////////////////////////////////
  // Variables
  //////////////////////////////////////////////////////////////////////////
  private:
    agx::Real m_videoFPS;
    agx::Real m_videoQuality;
    QString   m_videoOutputPath;
    agx::Real m_realTimeFactor;
    agx::Real m_timeStep;
    agxQt::VideoCapture*            m_videoCapture;
    Ui::VideoCaptureSettingsWidget* m_ui;
  };
}

#endif // QT_VIDEOSETTINGS_WIDGET