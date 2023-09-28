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

#ifndef AGX_THREADTASK_H
#define AGX_THREADTASK_H

#include <agx/Task.h>

namespace agx
{
  AGX_DECLARE_POINTER_TYPES(ThreadTask);
  class AGXCORE_EXPORT ThreadTask : public ParallelTask
  {
  public:
    /**
    Constructor. Create a thread task with an serial preparation stage followed by a parallel section (which is called once for each thread).
    */
    ThreadTask(const Name& name, const Job::Callback& serialCallback, const Job::Callback& parallelCallback, UInt numThreads);
    ThreadTask(const Name& name, const Job::Callback& parallelCallback, UInt numThreads);
    ThreadTask(const Job::Callback& parallelCallback, UInt numThreads);

    /**
    Set the number of threads (which will be clamped to the global agx::setNumThreads setting during execution)
    */
    void setNumThreads(UInt numThreads);

    /**
    \return The number of threads requested for the task.
    */
    UInt getNumThreads() const;

  protected:
    virtual ~ThreadTask();
    void run();

  private:
    Vector<Job> m_jobs;
    agxData::ValueT<UInt> *m_numThreads;
    Job::Callback m_serialCallback;
    Job::Callback m_parallelCallback;
  };
}

#endif /* AGX_THREADTASK_H */
