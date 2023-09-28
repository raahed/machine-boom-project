/*
Copyright 2007-2023. Algoryx Simulation AB.

All AgX source code, intellectual property, documentation, sample code,
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

#ifndef AGXQT_KEYSHOTEXPORTERDIALOG_H
#define AGXQT_KEYSHOTEXPORTERDIALOG_H

#include <agx/PushDisableWarnings.h> // Disabling warnings. Include agx/PopDisableWarnings.h below!
#include <QWidget>
#include <QThread>
#include <agxQt/ui_KeyshotExporterDialog.h>
#include <agx/PopDisableWarnings.h> // End of disabled warnings.

#include <agx/config/AGX_USE_KEYSHOT.h>
#include <agx/TimeStamp.h>
#include <agx/agx.h>
#include <agx/String.h>
#include <agxSDK/SimulationController.h>
#include <agxQt/export.h>
#include <iostream>
#include <agxControl/MeasurementSensor.h>

#if AGX_USE_KEYSHOT()

namespace agxQt
{
  class KeyshotWriterThread;
  class QtAgxWrapper;
  class CameraDescription;
  class ParticleSystemRenderUtility;

  // Time settings struct. Used in the export function for exporting data in time intervals.
  struct KeyshotTimeSettings
  {
    agx::Real startTime;
    agx::Real endTime;
    agx::Int  fps;
    bool      useSnapShot;
  };

  /**
  * Keyshot export dialog wrapper for qtViewer
  */
  class AGXQT_EXPORT KeyshotExporterDialog : public QWidget
  {
    Q_OBJECT

  public:
    explicit KeyshotExporterDialog(QWidget *parent = 0);
    virtual ~KeyshotExporterDialog();

    void init();
    void updateJournal(agxQt::QtAgxWrapper * wrapper);

    bool verifyAgainstActiveJournal();

    public slots:
    void slotUpdateSlider(int);
    void slotUpdateTimeLeft(double);
    void slotStartThread();
    void slotStopThread();
    void slotWriteCompleted();
    void slotUseSnapshotBoxChecked();
    void slotTimeSettingsChanged();
    void synchronizeGUIFromUpdatedJournal();
    void updateGUI();

  protected:
    agx::UInt verifyFPS(agx::UInt fps);

    //////////////////////////////////////////////////////////////////////////
    // Variables
    //////////////////////////////////////////////////////////////////////////
  private:
    Ui::KeyshotExporterDialog *             m_ui;
    KeyshotWriterThread    *                m_writerThread;
    agx::Real                               m_startTime;
    agx::Real                               m_endTime;
    agx::Int                                m_fps;
    bool                                    m_useTimeSnapShot;
    agxControl::ExponentialFilterRef        m_filter;
    agxQt::QtAgxWrapper*                    m_wrapper;
  };

  class AGXQT_EXPORT KeyshotWriterThread : public QThread
  {
    Q_OBJECT

  public:
    KeyshotWriterThread(agxQt::QtAgxWrapper * wrapper, const agxQt::KeyshotTimeSettings& timeSettings);

    /// Updates the export information used in the thread
    void update(agxQt::QtAgxWrapper * wrapper);

    /// Updates the time settings used in the export thread
    void setTimeSettings(const agxQt::KeyshotTimeSettings& timeSettings);

    /// Reset the writer thread
    void reset();

    public slots:
    void slotAbortThread();

  signals:
    void update(int);
    void updateTimeLeft(double);

  protected:
    void run();

    //////////////////////////////////////////////////////////////////////////
    // Variables
    //////////////////////////////////////////////////////////////////////////
  public:
    bool                                        m_shouldAbort;
    bool                                        m_running;
  protected:
    agxQt::KeyshotTimeSettings                  m_timeSettings;
    agxQt::QtAgxWrapper*                        m_wrapper;
  };

  class AGXQT_EXPORT KeyshotBipFilesWrtier
  {
  public:
    typedef std::function<bool(int, double)> StepforwardCallback;

    /// Write kesyhot .bip files
    static bool writeBipFilesFromJournal(
      const agx::String& journalFilename,
      const agx::String& sessionName,
      agxQt::CameraDescription* camera,
      const agxQt::KeyshotTimeSettings& timeSettings,
      agxQt::ParticleSystemRenderUtility * renderUtil,
      StepforwardCallback callback);

  private:
    static bool _writeBipFilesFromJournal(
      const agx::String& journalFilename,
      const agx::String& sessionName,
      agxQt::CameraDescription* camera,
      const agxQt::KeyshotTimeSettings& timeSettings,
      agxQt::ParticleSystemRenderUtility * renderUtil,
      StepforwardCallback callback);
  };
}

#endif
#endif