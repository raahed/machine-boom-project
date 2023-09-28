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

#ifndef AGXPLOT_SYSTEM_H
#define AGXPLOT_SYSTEM_H

#include <agxPlot/Window.h>
#include <agxPlot/Output.h>
#include <agxPlot/Queue.h>
#include <agxPlot/DataSeries.h>

#include <agx/Thread.h>
#include <agx/ThreadSynchronization.h>
#include <agx/Timer.h>

#include <agxSDK/Simulation.h>

namespace agxPlot
{
  class Output;
  AGX_DECLARE_POINTER_TYPES(Output);

  class AGXPHYSICS_EXPORT System : public agx::Referenced
  {
    public:
      System();


      /**
       Get the default plot associated to the system. This is where plot curves
       are added if nothing else is specified.
       \param windowId the ID associated with the window. If created this way, the default
                       name will be the same as the ID.
       \return the default plot window
       */
      agxPlot::Window* getOrCreateWindow(const agx::String& windowId = "");

      /**
         When automatically generating values, this is the rate which the plot system
         tries to hold.

         It will add value from the timestep closest after if stride is not perfect.
         There is no average calculation etc.
         \param timeStep The target timeStep for the plot, if lower  than the simulation
                         timestep values from all timesteps will be used.
      */
      void setTimeStep(agx::Real timeStep);

      /**
          Check if it is time for a new plot frame. If true, values will be automatically
          generated for this timestep.
      */
      bool shouldUpdate() const;

      /**
       Add an output to listen to the System. When updated, the output should be informed.
       Also adds all current data to the output queue.
       \param output Output that listens to the system
       */
      void add(agxPlot::Output* output);

      /**
       INTERNAL.

       Use addNewOutput and this will remove old systems from the output.
       \param output Output that should stop listening to the system
       */
      void remove(agxPlot::Output* output);

      /**
       Update all series after a time step
       */
      void generateData();

      /**
       Check for new data and push it to all plot outputs. Should be safe to do, always.
       */
      void refresh();

      /**
       Add a data series to be updated. If it is not connected to a plot,
       it will not be exported automatically. No two data series of with
       the same ID should exist.
       \param dataSeries Should not be nullptr.
       \retval Was adding the series successful?
       */
      bool addDataSeries(agxPlot::DataSeries* dataSeries);


      /**
       Get a dataseries from the ID it holds.
      */
      agxPlot::DataSeries* getDataSeries(agx::UInt id) const;

      /// \return the registered outputs
      const OutputRefVector& getOutputs() const;

      /**
       Returns true if the system has any windows.
      */
      bool hasAnyWindow() const;

      /**
       Retrieve the description as follows.

       ["windows"] = an array of all agxPlot::Window descriptions.
       ["data"] = an array of tuples. Contains ["id"] with the data id
                  and ["values"] which is an array of real values.
                  This field only exists if includeData is true.
      */
      void getDescription(agxJson::Value& root, bool includeData) const;
      /**
       INTERNAL

       set from the description generated from getDescription.
      */
      void setFromDescription(const agxJson::Value& root);

      /**
       Store the current description in an attached RECORD journal.
       Will automatically be called upon normal exit.
      */
      void save() const;
      /**
       Load from an attach PLAYBACK journal.
       Will automatically be called upon attaching a journal.
      */
      void load();

      /**
       Tell all outputs attached that no more data will be coming.
       Close file streams and the like.
      */
      void closeOutputs();

      /**
       Clear all old data from DataSeries
      */
      void clearDataSeries();

      /**
       Sets all windows in the system to disabled.
      */
      void disableWindows();

      typedef agx::Event1<agxPlot::Output*> OutputAddedEvent;
      OutputAddedEvent outputAdded;

      /// Disable all plotting
      static void disableAllPlotting();
      static bool plottingDisabled();

    public:
      /***************************************/
      /* HELPER FUNCTIONS FOR CREATING PLOTS */
      /***************************************/
    protected:
      ~System();
    private:
      void updatedRepresentation();
      /**
      Add a new window to the plot system. Each window has its own collection of
      curves, its own view of the simulation.
      \param window Window to add to the system
      */
      void add(Window* window);

      bool hasData() const;

      void simulationTickCallback(agx::Clock *clock);

      AGX_DECLARE_POINTER_TYPES(OutputThread);

    private:
      friend class agxSDK::Simulation;
      void setSimulation(agxSDK::Simulation* simulation);

    private:
      agxPlot::DataSeriesRefVector m_dataSeries;
      agx::Clock::TickEvent::CallbackType m_simulationTickCallback;
      WindowRefVector m_windows;
      agx::observer_ptr< agxSDK::Simulation> m_simulation;
      OutputRefVector m_outputs;
      OutputThreadRef m_outputThread;
      bool m_loadingJournal;

      bool m_updatedRepresentation;
      agx::Real m_plotTimeStep;
      agx::Real m_lastPlotTime;

      agx::Event::CallbackType m_windowRepresentationChangedCallback;
    private:
      friend class OutputThread;
      agx::Mutex m_outputMutex;
  };
  AGX_DECLARE_POINTER_TYPES(System);


  ///////////////////////////////////////////////////////

  class System::OutputThread : public agx::BasicThread, public agx::Referenced
  {
  public:
    OutputThread(System *system, agx::Real frequency = 30);

    void setFrequency(agx::Real frequency);
    agx::Real getFrequency() const;

  protected:
    virtual ~OutputThread();

  private:
    virtual void run();

    void start();
    void stop();
    void updateOutputs();

  private:
    System *m_system;
    bool m_running;
    agx::Block m_startBlock;
    agx::Timer m_timer;
    agx::Real m_frequency;
  };
}

#endif
