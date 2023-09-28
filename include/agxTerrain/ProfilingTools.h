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

#include <agxTerrain/export.h>
#include <agx/Referenced.h>
#include <agx/Name.h>
#include <agx/HighAccuracyTimer.h>
#include <agx/HashTable.h>
#include <agx/HashVector.h>
#include <iostream>
#include <iomanip>

namespace agxTerrain
{
  namespace Profiler
  {
    typedef agx::HashVector<agx::Name, agx::Real> TimingStorage;
    using Bucket = std::pair<agx::Name, agx::Real>;

    AGXTERRAIN_EXPORT TimingStorage& getStorage();
    AGXTERRAIN_EXPORT void clear();
    AGXTERRAIN_EXPORT void printTimers(const agx::String& title);
    AGXTERRAIN_EXPORT void printToFile(agx::Real time, const agx::String& filename);
    AGXTERRAIN_EXPORT void resetFile(const agx::String& filename);
    AGXTERRAIN_EXPORT void registerTiming(const agx::Name& name, agx::Real value);
    AGXTERRAIN_EXPORT agx::Real& currentTime();
  }

  AGX_DECLARE_POINTER_TYPES(ScopeTimer);
  class AGXTERRAIN_EXPORT ScopeTimer : public agx::Referenced
  {
  public:
    explicit ScopeTimer(const agx::Name& name)
        : m_timer(false)
        , m_name(name)
    {
      m_timer.start();
    }

  protected:
    virtual ~ScopeTimer()
    {
      m_timer.stop();
      Profiler::registerTiming(m_name, m_timer.getTime());
    }

    agx::HighAccuracyTimer m_timer;
    agx::Name m_name;
  };
}

#define SCOPE_PROFILER(a) agxTerrain::ScopeTimerRef ref = new agxTerrain::ScopeTimer(a);
#define PRINT_TIMINGS() Profiler::printTimers();


#if defined(USE_EXTENDED_TERRAIN_PROFILING) && USE_EXTENDED_TERRAIN_PROFILING
  #define GET_TIMESTAMP(varname) agx::UInt64 varname = agx::Timer::getCurrentTick();
  #define THREAD_LOG_ITEM( textname, var1, var2 ) agx::Thread::getCurrentThread()->reportSystemJob( var1, var2, textname );
#else
  #define GET_TIMESTAMP(varname)
  #define THREAD_LOG_ITEM( textname, var1, var2 )
#endif


