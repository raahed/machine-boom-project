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

#ifndef AGXQT_CONTACTWRITERDIALOG_H
#define AGXQT_CONTACTWRITERDIALOG_H

#include <agx/PushDisableWarnings.h> // Disabling warnings. Include agx/PopDisableWarnings.h below!
#include <QWidget>
#include <agxQt/ui_DataExporterDialog.h>
#include <agx/PopDisableWarnings.h> // End of disabled warnings.

#include <agx/TimeStamp.h>
#include <agx/agx.h>
#include <agxQt/ImpactAnalysis.h>
#include <agxQt/export.h>
#include <iostream>
#include <agxControl/MeasurementOperations.h>

namespace agxQt
{
  /**
  * Data export dialog wrapper for qtViewer
  */
  class AGXQT_EXPORT DataExporterDialog : public QWidget
  {
    Q_OBJECT

  public:
    explicit DataExporterDialog(QWidget *parent = 0);
    virtual ~DataExporterDialog();

    void init();
    void updateJournal(agx::String filename, agx::String sessionName, agx::AnalysisBox* bound);

    bool verifyAgainstActiveJournal();

    public slots:
    void slotUpdateSlider(int);
    void slotUpdateTimeLeft(double);
    void slotStartThread();
    void slotStopThread();
    void slotWriteCompleted();
    void slotUseAnalysisBoxChecked();
    void slotUseSnapshotBoxChecked();
    void slotExportTypeChanged(int);
    void slotTimeSettingsChanged();
    void synchronizeGUIFromUpdatedJournal();
    void updateGUI();

  protected:

    //////////////////////////////////////////////////////////////////////////
    // Variables
    //////////////////////////////////////////////////////////////////////////
  private:
    Ui::DataExporterDialog *                    m_ui;
    ContactWriterThread    *                    m_writerThread;
    agx::GranularDataExporter::ExporterDataType m_type;
    agxControl::ExponentialFilterRef            m_filter;
    agx::Real                                   m_startTime;
    agx::Real                                   m_endTime;
    bool                                        m_useTimeSnapShot;
  };
}

#endif