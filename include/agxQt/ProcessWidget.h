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

#ifndef AGXQT_QPROCESS_WIDGET_H
#define AGXQT_QPROCESS_WIDGET_H

#include <agx/PushDisableWarnings.h> // Disabling warnings. Include agx/PopDisableWarnings.h below!
#include <agxQt/ui_ProcessOutputWidget.h>
#include <agxQt/export.h>
#include <QWidget>
#include <QProcess>
#include <osgViewer/ViewerEventHandlers>
#include <osgViewer/Renderer>
#include <agx/PopDisableWarnings.h> // End of disabled warnings.

#include <agx/Referenced.h>

namespace agxQt
{
  class AGXQT_EXPORT ProcessOutputWidget : public QWidget
  {
    Q_OBJECT

  public:
    explicit ProcessOutputWidget(QWidget *parent = 0);
    virtual ~ProcessOutputWidget();

    void connectProcess(QProcess *p);
    void appendTextToOutputConsole(QString text);

    public slots:
      void updateOutput();
      void updateError();

  private slots:
      void slotCloseButtonPressed();
      void processFinished(int exitCode, QProcess::ExitStatus exitStatus);

  //////////////////////////////////////////////////////////////////////////
  // Variables
  //////////////////////////////////////////////////////////////////////////
  private:
    Ui::QProcessWidget* m_ui;
    QProcess *          m_process;
  };
}



#endif
