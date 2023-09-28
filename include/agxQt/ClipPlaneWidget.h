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
#include <QWidget>
#include <agxQt/ui_Controller.h>
#include <agxQt/ui_ClipPlaneWidget.h>
#include <agx/PopDisableWarnings.h> // End of disabled warnings.

#include <agx/EulerAngles.h>
#include <agx/Quat.h>
#include <agx/Vec3.h>
#include <agxQt/export.h>

namespace agxQt
{
  class ClipPlaneController;
}

namespace agxQt
{
  /**
  * ClipPlaneWidget for controlling the ClipPlane in qtViewer
  */
  class AGXQT_EXPORT ClipPlaneWidget : public QWidget
  {
    Q_OBJECT

  public:
    explicit ClipPlaneWidget(QWidget *parent = 0);

    virtual ~ClipPlaneWidget();

    void setClipPlaneController(agxQt::ClipPlaneController * clipPlane);

    void connectSignals();

  private:
    void setEnableGUI(bool enable);

  public:
    signals:
      void signalClipPlaneUpdated();

  public slots:
    void slotUpdateGUI();

  private slots:
    void slotCenterEditFinished();
    void slotRotationEditFinished();
    void slotEnableChecked();
    void slotEnableRenderChecked();
    void slotUpdateGUIFromClipPlane();

    //////////////////////////////////////////////////////////////////////////
    // Variables
    //////////////////////////////////////////////////////////////////////////
  private:
    Ui::ClipPlaneWidget*              m_ui;
    agxQt::ClipPlaneController*       m_clipPlaneController;
  };
}
