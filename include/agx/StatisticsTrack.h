/*
Copyright 2007-2023. Algoryx Simulation AB.

All AGX source code, intellectual property, documentation, sample code,
tutorials, scene files and technical white papers, is copyrighted, proprietary
and confidential material of Algoryx Simulation AB. You may not download, read,
store, distribute, publish, copy or otherwise disseminate, use or expose this
material unless having a written signed agreement with Algoryx Simulation AB, or having been
advised so by Algoryx Simulation AB for a time limited evaluation, or having purchased a
valid commercial license from Algoryx Simulation AB.

Algoryx Simulation AB disclaims all responsibilities for loss or damage caused
from using this software, unless otherwise stated in written agreements with
Algoryx Simulation AB.
*/

#ifndef AGX_STATISTICSTRACK_H
#define AGX_STATISTICSTRACK_H

#include <agx/Statistics.h>
#include <agx/Clock.h>
#include <agxData/FrameIO.h>

namespace agxSDK
{
  AGX_DECLARE_POINTER_TYPES(Simulation);
}

namespace agx
{
  class StatisticsFrameReader : public agxData::FrameReader
  {
  public:
    StatisticsFrameReader(agxSDK::Simulation *simulation, Clock *clock, Statistics *statistics = Statistics::instance());


    virtual agxData::Frame *readFrame() override;

  protected:
    virtual ~StatisticsFrameReader();

  private:
    class DataBinding : public agxData::Frame::DataBinding
    {
    public:
      DataBinding(const agx::Path& internalPath, const agx::Path& externalPath, Statistics::AbstractData *data = nullptr);

      Statistics::AbstractData *getData();
      void setData(Statistics::AbstractData *data);

    protected:
      virtual ~DataBinding();

    private:
      Statistics::AbstractDataObserver m_data;
    };

    virtual agxData::Frame::DataBinding *createDataBinding(const agx::Path& internalPath, const agx::Path& externalPath) override;
    void updateCallback(Clock *clock);

  private:
    agxSDK::SimulationRef m_simulation;
    ClockRef m_clock;
    Statistics *m_statistics;
    TaskRef m_updateTask;
    mutable agx::Block m_clockBlock;
    mutable agx::Block m_readerBlock;
    Clock::TickEvent::CallbackType m_updateCallback;
  };

}


#endif /* AGX_STATISTICSTRACK_H */
