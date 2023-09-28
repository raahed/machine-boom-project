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

#ifndef AGXQT_ANALYSISBOUNDWIDGET_H
#define AGXQT_ANALYSISBOUNDWIDGET_H

#include <agx/PushDisableWarnings.h> // Disabling warnings. Include agx/PopDisableWarnings.h below!
#include <QWidget>
#include <agxQt/ui_Controller.h>
#include <agxQt/ui_AnalysisBoundWidget.h>
#include <agx/PopDisableWarnings.h> // End of disabled warnings.

#include <agxQt/export.h>

namespace agx
{
  class AnalysisBox;
}

namespace agxQt
{
  /**
  * AnalysisBoundWidget for controlling the AnalysisBound in qtViewer
  */
  class AGXQT_EXPORT AnalysisBoundWidget : public QWidget
  {
    Q_OBJECT

  public:
    explicit AnalysisBoundWidget(QWidget *parent = 0);

    virtual ~AnalysisBoundWidget();

    void setAnalysisBox(agx::AnalysisBox * box);

    void connectSignals();

  private:
    void setEnableGUI(bool enable);

  public:
    signals:
      void signalAnalysisBoxUpdated();

  public slots:
    void slotUpdateGUI();

  private slots:
    void slotCenterEditFinished();
    void slotSizeEditFinished();
    void slotAnalysisCheckBoxChecked();
    void slotUpdateGUIFromAnalysisBox();

    //////////////////////////////////////////////////////////////////////////
    // Variables
    //////////////////////////////////////////////////////////////////////////
  private:
    Ui::AnalysisBoundWidget* m_ui;
    agx::AnalysisBox       * m_analysisBox;
  };
}

#endif