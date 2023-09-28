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

#ifndef AGXQT_PLAYBACKCONTROLLER_H
#define AGXQT_PLAYBACKCONTROLLER_H

#include <agx/PushDisableWarnings.h> // Disabling warnings. Include agx/PopDisableWarnings.h below!
#include <QWidget>
#include <agxQt/ui_Controller.h>
#include <agx/PopDisableWarnings.h> // End of disabled warnings.

#include <agx/TimeStamp.h>
#include <agx/agx.h>
#include <agxQt/export.h>
#include <iostream>

namespace agxQt
{
  /**
  * Controller for playback
  */
  class AGXQT_EXPORT PlaybackControllerWidget : public QWidget
  {
    Q_OBJECT

  public:
    enum ControllerState
    {
      PLAY,
      PAUSED
    };

  public:

    explicit PlaybackControllerWidget(QWidget *parent = 0);
    virtual ~PlaybackControllerWidget();

    void setEnable(bool enable);

    bool eventFilter(QObject * watched, QEvent * event);

    public slots:
      void slotSetPause();
      void slotSetPlay();
      void slotInitNewPlayback(float timeStart, float timeEnd, uint maxFrames );
      void slotInitNewSimulation();
      void slotTimeChanged(float time);

      // Public slots used to push buttons the in the GUI
      void slotPushPlayPause();
      void slotPushStepForward();
      void slotPushStepBack();
      void slotPushIncrementSpeed();
      void slotPushDecrementSpeed();

      void slotSetEnableRecordingNotifier(bool enable);

   signals:
     void signalSliderTimeJump(float time);
     void signalPlay();
     void signalPause();
     void signalStop();
     void signalStepForward();
     void signalStepBack();
     void signalSpeedChanged(uint speed);
     void signalEnableJournalLoop(bool enable);

  // Internal slots
  private slots:
     void slotSliderChanged(int value);
     void slotStopPushed();
     void slotForwardPushed();
     void slotBackPushed();
     void slotPlayPausePushed();
     void slotSliderPressed();
     void slotSliderReleased();

     void slotIncrementSpeedPushed();
     void slotDecrementSpeedPushed();
     void slotLoopCheckBoxPressed();

  protected:
    void setEnablePlaybackGui(bool enable);
    void setControllerState(ControllerState state);
    void connectSignals();
    void init();
    void updateTimeLabels();
    void updateSpeedButtonLabel();
    QString constructHoverValueString();
    float getSingleStepValue();
    float getCurrentTimeFromPlaybackSliderValue(int val);

    //////////////////////////////////////////////////////////////////////////
    // Variables
    //////////////////////////////////////////////////////////////////////////
  private:
    Ui::Controller * m_ui;

    bool m_enable;
    bool isInPlayback;
    agx::UInt m_speedModifier;
    ControllerState m_state;

    agx::UInt m_maxFrames;
    agx::TimeStamp m_startTime;
    agx::TimeStamp m_endTime;

    QIcon m_playIcon;
    QIcon m_stopIcon;
    QIcon m_pauseIcon;
    QIcon m_stepForwardIcon;
    QIcon m_speedForwardIcon;
    QIcon m_speedBackwardIcon;
    QIcon m_stepBackIcon;

    bool m_sliderPressed;
  };
}

#endif
