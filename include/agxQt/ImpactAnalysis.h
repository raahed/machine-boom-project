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

#ifndef AGXQT_IMPACTANALYSIS_H
#define AGXQT_IMPACTANALYSIS_H

#include <functional>
#include <agx/PushDisableWarnings.h>
#include <QThread>
#include <agx/PopDisableWarnings.h>
#include <agxSDK/SimulationController.h>
#include <agx/GranularDataExporter.h>

namespace agxQt
{
  // Time settings struct. Used in the export function for exporting data in time intervals.
  struct TimeSettings
  {
    agx::Real startTime;
    agx::Real endTime;
    bool      useSnapShot;
  };

  /**
  Class containing utility functions for exporting simulation data.
  */
  class ImpactInformationWriter
  {
  public:
    typedef std::function<bool(int, double)> StepforwardCallback;

    /// Write simulation data to file
    static bool writeSimulationDataToFile(agx::String filename,
      agx::String sessionName,
      const agx::Bound3& bound,
      const TimeSettings& settings,
      agx::GranularDataExporter::ExporterDataType exportType,
      StepforwardCallback callback);
  };

  /// Thread class for writing simulation  information to file
  class ContactWriterThread : public QThread
  {
    Q_OBJECT

  public:
    /**
    Data Export thread class that opens up the specified journal and writes the specified export information to file.

    \param type Type of data to be exported from the journal.
    \param filename Name of the journal file to open.
    \param sessionName Name of session to open in the specified journal file.
    \param timeSettings The time settings to use when determining what time interval to export in.
    \param bound Analysis box to use in the export, to limit data entities to those that are inside the box.
    */
    ContactWriterThread(agx::GranularDataExporter::ExporterDataType type, agx::String filename, agx::String sessionName, const TimeSettings& timeSettings, agx::AnalysisBox* bound);

    /// Set export type of the writing thread
    void setExportType(agx::GranularDataExporter::ExporterDataType type);

    /// Updates the export information used in the thread
    void update(agx::String filename, agx::String sessionName, agx::AnalysisBox* bound);

    /// Updates the time settings used in the export thread
    void setTimeSettings(const TimeSettings& timeSettings);

    /// Reset the writer thread
    void reset();

    // Set if the AnalysisBox should be used in the export.
    void setShouldUseBound(bool enable);

    // Get if the AnalysisBox should be used in the export.
    bool getShouldUseBound() const;

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
    agxSDK::SimulationController*               m_controller;
    agx::String                                 m_fileName;
    agx::String                                 m_sessionName;
    agx::GranularDataExporter::ExporterDataType m_exportType;
    agx::AnalysisBox*                           m_bound;
    TimeSettings                                m_timeSettings;
    bool                                        m_useBound;
  };
}

#endif