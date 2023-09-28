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

#ifndef AGX_JOB_H
#define AGX_JOB_H

#ifdef _MSC_VER
# pragma warning(push)
# pragma warning(disable:4355) // Disable warnings about passing the 'this' pointer to base class constructors.
# pragma warning(disable: 6011) // Disable warningC6011: dereferencing nullptr pointer
#endif

#include <agx/agxCore_export.h>
#include <agx/IndexRange.h>
#include <agx/Callback.h>

#include <agx/Name.h>
#include <agxData/Type.h>
#include <agx/Event.h>
#include <agx/Vector.h>
#include <agx/Timer.h>
#include <agx/ThreadSynchronization.h>
#include <agx/SyncTag.h>
#include <deque>




namespace agx
{
  class Device;
  class Task;
  class Thread;

  typedef class Job* JobPtr;


  typedef VectorPOD<class Job *> JobPtrVector;
  typedef std::deque<class Job *> JobPtrQueue;

  class RangeJob;
  using RangeCallback = agx::Callback1<const RangeJob&>;

  /**
  An abstract job/workblock representation, which allows work
  threads to execute arbitrary tasks. The jobs have specified
  dependecies which enables them to be organized in an execution graph.
  */
  class AGXCORE_EXPORT Job : public Callback
  {
  public:
    Job();
    Job(const Job& other);
    Job(const Callback& callback, agx::UInt32 costEstimate = 100);
    virtual ~Job();

    Job& operator=(const Job& other);

    /// Initialize the job
    void init(const Callback& callback = Callback(), agx::UInt32 costEstimate = 100);

    /**
    Add a dependency to a job.
    \param parent The parent dependency
    */
    void addDependency(Job *parent);

    /**
    Remove a dependency from a job.
    \param parent The parent dependency
    */
    void removeDependency(Job *parent);

    /**
    \param parent - The parent job to check
    \return True if the specified job is a direct parent dependency, not recursive.
    */
    bool hasDependency(Job *parent);

    /**
    Remove all parent dependencies.
    NOTE: Child dependencies in parents are _not_ updated
    */
    void removeAllDependencies();

    /**
    Remove all child dependencies.
    NOTE: Parent dependencies in children are _not_ updated
    */
    void clearChildDependencies();

    /**
    Explicitly set the number of parent dependencies.
    NOTE: Only use if you know what you are doing....
    */
    void setNumDependencies(agx::UInt32 numDependencies);

    /**
    \return The number of parent dependencies.
    */
    agx::UInt32 getNumDependencies() const;

    /**
    Explicitly increment the number of parent dependencies.
    */
    void incrementNumDependencies(agx::UInt32 size = 1);

    /**
    Explicitly decrement the number of parent dependencies.
    */
    void decrementNumDependencies(agx::UInt32 size = 1);

    /**
    \return The list of child dependencies.
    */
    const JobPtrVector& getChildDependencies() const;

    /**
    \return The parent jobs. NOTE Only available in debug build, in release the vector is always empty.
    */
    const JobPtrVector& getParentDependencies() const;

    /**
    Set the callback which is called when execute() runs.
    */
    void setCallback(const Callback& callback);

    /**
    Return the current callback.
    */
    const Callback& getCallback() const;

    /**
    Increase the dependency counter for the job, and queue job for execution if all dependencies are resolved.
    Automatically called as parent dependencies are completed.
    \return true if all dependencies are resolved
    */
    bool resolve();

    /**
    Schedule job for execution in thread pool.
    */
    void spawn();

    /**
    \return The thread this job is bound to. nullptr if no explicit thread binding.
    */
    Thread *getTargetThread();

    /**
    Set the target thread which the job will be executed on.
    */
    void setTargetThread(Thread *thread);

    /**
    \return The task which this job is part of.
    */
    Task *getTask();

    /**
    Set the task which the job belongs to. Normally handled automatically.
    */
    void setTask(Task *task);

    /**
    \return The synchronization tag for this job.
    */
    SyncTag *getSyncTag();

    /**
    Set the synchronization tag which must enter completion status before the job is executed.
    */
    void setSyncTag(SyncTag *tag);

    /**
    \return The execution time, in ms.
    */
    Real getExecutionTime() const;

    /**
    \return The execution timer.
    */
    const Timer& getTimer() const;

    /**
    Specify the cost estimate for this job.
    */
    void setCostEstimate(agx::UInt32 cost);

    /**
    \return The estimated job size.
    */
    agx::UInt32 getCostEstimate() const;

    /**
    \return The thread which executed the job.
    */
    Thread *getAssignedThread();


    /**
    \return The job tag.
    */
    agx::UInt32 getTag() const;

    /**
    Set the job tag, can be used to keep track of different job classes etc.
    */
    void setTag(agx::UInt32 tag);

    /**
    Set the name of the job, for debugging.
    */
    void setName(const char *name);

    /**
    \return The name of the job.
    */
    const char *getName() const;


    /**
    \return The index of the job (for parallel kernels)
    */
    agx::Index& index();
    const agx::Index& index() const;

  public:

    /**
    Job implementation, override in subclass. Default implementation use a generic callback.
    */
    virtual void implementation();

  private:
    /**
    Execute the job callback, and trigger post job event.
    \param resolvedChildren - Vector which is filled with resolved child dependencies.
    */
    void execute(JobPtrVector& resolvedChildren, Thread* thread);

  protected:
    friend class Thread;
    friend class JobGroup;

    const char *m_name;
    JobPtrVector m_children;
    JobPtrVector m_parents;
    Callback m_callback;
    Timer m_timer;
    agx::UInt32 m_numDependencies;
    std::atomic<Int32> m_dependencyCounter;
    Task *m_task;
    Thread *m_targetThread;
    Thread *m_assignedThread;
    SyncTagRef m_syncTag;
    agx::UInt32 m_cost;
    agx::UInt32 m_tag;
    agx::Index m_index;
  };

  typedef Vector<Job> JobVector;

  //---------------------------------------------------------------

  /**
  A range job executes a kernel implementation on a subset
  of the data.
  */
  class AGXCORE_EXPORT RangeJob : public Job
  {
  public:
    using RangeCallback = agx::RangeCallback;

  public:
    RangeJob();
    RangeJob(const RangeCallback& callback, const IndexRange& range, agx::UInt32 costPerElement = 1);
    RangeJob(const RangeJob& other);
    virtual ~RangeJob();

    void init(const RangeCallback& callback, const IndexRange& range, agx::UInt32 costPerElement = 1);
    void init(const IndexRange& range, agx::UInt32 costPerElement = 1);

    RangeJob& operator= (const RangeJob& other);

    /**
    \return The work range allocated to this job.
    */
    IndexRange& range();
    const IndexRange& range() const;

  protected:
    virtual void implementation() override;

  private:
    // void dispatch();

  protected:
    RangeCallback m_rangeCallback;
    IndexRange m_range;
  };

  typedef Vector<RangeJob> RangeJobVector;

  //---------------------------------------------------------------

  template <class T>
  class RangeJobT : public RangeJob
  {
  public:
    RangeJobT();
    RangeJobT(const RangeCallback& callback, const IndexRange& range, agx::UInt32 costPerElement = 1);
    RangeJobT(const RangeJobT<T>& other);

    virtual ~RangeJobT();

  protected:
    virtual void implementation() override;
  };

  //---------------------------------------------------------------

  /**
  A group of jobs to be executed.
  Less overhead than having a sequence of jobs with serial dependencies.
  */
  class AGXCORE_EXPORT JobGroup : public Job
  {
  public:
    JobGroup();
    virtual ~JobGroup();

    void init();

    void addJob(Job *job);
    const JobPtrVector& getSubJobs() const;

    // void execute();

  protected:
    virtual void implementation() override;

  private:
    JobPtrVector m_jobs;
  };

  //---------------------------------------------------------------

  /**
  A job used as a synchronization utility when waiting for another job to complete.
  */
  class AGXCORE_EXPORT WaitJob : public Job
  {
  public:
    WaitJob(Job *job);
    virtual ~WaitJob();

    /**
    Blocks until the targeted job is completed.
    */
    void wait();

    /**
    Release the waiting thread. Automatically called when the target job completes.
    */
    void release();

    /**
    \return The target job to wait for.
    */
    Job *getTarget();
    const Job *getTarget() const;

  private:
    Job *m_target;
    Block m_block;
  };

  //---------------------------------------------------------------

  /**
  A task job is a job which executes an agx::Task
  */
  class AGXCORE_EXPORT TaskJob : public Job
  {
  public:
    TaskJob();
    TaskJob(Task *task);
    virtual ~TaskJob();

    void init(Task *task);
    void init();

    void execute();

  private:
    Task *m_target;
  };





  /* Implementation */

  //// JOB ////////////////////////////////////////////////////////
  AGX_FORCE_INLINE Job::Job() :
    Callback(&Job::spawn, this),
    m_name(""),
    m_numDependencies(0),
    m_dependencyCounter(0),
    m_task(nullptr),
    m_targetThread(nullptr),
    m_assignedThread(nullptr),
    m_cost(0),
    m_tag(0),
    m_index(InvalidIndex)
  {
  }

  AGX_FORCE_INLINE Job::Job(const Job& other) :
      Callback(&Job::spawn, this),
      m_name(other.m_name),
      m_children(other.m_children),
      m_parents(other.m_parents),
      m_callback(other.m_callback),
      m_timer(other.m_timer),
      m_numDependencies(other.m_numDependencies),
      m_dependencyCounter(other.m_dependencyCounter.load()),
      m_task(other.m_task),
      m_targetThread(other.m_targetThread),
      m_assignedThread(other.m_assignedThread),
      m_syncTag(other.m_syncTag),
      m_cost(other.m_cost),
      m_tag(other.m_tag),
      m_index(other.m_index)
  {
  }



  AGX_FORCE_INLINE Job::Job(const Callback& callback, agx::UInt32 costEstimate) :
    Callback(&Job::spawn, this),
    m_name(""),
    m_callback(callback),
    m_numDependencies(0),
    m_dependencyCounter(0),
    m_task(nullptr),
    m_targetThread(nullptr),
    m_assignedThread(nullptr),
    m_cost(costEstimate),
    m_tag(0),
    m_index(InvalidIndex)
  {
  }

  AGX_FORCE_INLINE Job& Job::operator=(const Job& other)
  {
    m_name = other.m_name;
    m_children = other.m_children;
    m_parents = other.m_parents;
    m_callback = other.m_callback;
    m_timer = other.m_timer;
    m_numDependencies = other.m_numDependencies;
    m_dependencyCounter = other.m_dependencyCounter.load();
    m_task = other.m_task;
    m_targetThread = other.m_targetThread;
    m_assignedThread = other.m_assignedThread;
    m_syncTag = other.m_syncTag;
    m_cost = other.m_cost;
    m_tag = other.m_tag;
    m_index = other.m_index;

    return *this;
  }

  AGX_FORCE_INLINE const char *Job::getName() const { return m_name; }
  AGX_FORCE_INLINE void Job::setName(const char *name) { m_name = name; }

  AGX_FORCE_INLINE Thread *Job::getTargetThread() { return m_targetThread; }
  AGX_FORCE_INLINE void Job::setTargetThread(Thread *thread) { m_targetThread = thread; }
  AGX_FORCE_INLINE Task *Job::getTask() { return m_task; }
  AGX_FORCE_INLINE void Job::setTask(Task *task) { m_task = task; }
  AGX_FORCE_INLINE SyncTag *Job::getSyncTag() { return m_syncTag; }
  AGX_FORCE_INLINE void Job::setSyncTag(SyncTag *tag) { m_syncTag = tag; }
  AGX_FORCE_INLINE Thread *Job::getAssignedThread() { return m_assignedThread; }

  AGX_FORCE_INLINE agx::UInt32 Job::getTag() const { return m_tag; }
  AGX_FORCE_INLINE void Job::setTag(agx::UInt32 tag) { m_tag = tag; }


  AGX_FORCE_INLINE void Job::addDependency(Job *parent)
  {
    agxAssert(parent);
    // agxAssert(!parent->m_children.contains(this));
    parent->m_children.push_back(this);
    m_numDependencies++;
    #ifdef AGX_DEBUG
    m_parents.push_back(parent);
    #endif
  }


  AGX_FORCE_INLINE void Job::removeDependency(Job *parent)
  {
    agxAssert(parent);
    // agxAssert(parent->m_children.contains(this));
    if (parent->m_children.findAndErase(this)) {
      m_numDependencies--;
      #ifdef AGX_DEBUG
      agxVerify(m_parents.findAndErase(parent));
      #endif
    }
  }


  AGX_FORCE_INLINE bool Job::hasDependency(Job *parent)
  {
    agxAssert(parent);
    return parent->m_children.contains(this);
  }

  AGX_FORCE_INLINE const JobPtrVector& Job::getParentDependencies() const { return m_parents; }

  AGX_FORCE_INLINE void Job::clearChildDependencies() { m_children.clear(); }
  AGX_FORCE_INLINE void Job::setNumDependencies(agx::UInt32 numDependencies) { m_numDependencies = numDependencies; }
  AGX_FORCE_INLINE void Job::removeAllDependencies()
  {
    m_numDependencies = 0;
    #ifdef AGX_DEBUG
    m_parents.clear();
    #endif
  }

  AGX_FORCE_INLINE agx::UInt32 Job::getNumDependencies() const { return m_numDependencies; }
  AGX_FORCE_INLINE void Job::incrementNumDependencies(agx::UInt32 size) { m_numDependencies += size; }
  AGX_FORCE_INLINE void Job::decrementNumDependencies(agx::UInt32 size) { m_numDependencies -= size; }

  AGX_FORCE_INLINE const JobPtrVector& Job::getChildDependencies() const { return m_children; }

  AGX_FORCE_INLINE void Job::setCallback(const Callback& callback) { m_callback = callback; }
  AGX_FORCE_INLINE const Job::Callback& Job::getCallback() const { return m_callback; }


  AGX_FORCE_INLINE void Job::init(const Callback& callback, agx::UInt32 costEstimate)
  {
    this->setNumDependencies(0);
    m_children.clear();

    #ifdef AGX_DEBUG
    m_parents.clear();
    #endif

    m_callback = callback;
    m_cost = costEstimate;
    m_targetThread = nullptr;
    m_assignedThread = nullptr;
    m_tag = 0;
    m_index = 0;
  }

  AGX_FORCE_INLINE Real Job::getExecutionTime() const { return Real(m_timer.getTime()); }
  AGX_FORCE_INLINE const Timer& Job::getTimer() const { return m_timer; }
  AGX_FORCE_INLINE agx::UInt32 Job::getCostEstimate() const { return m_cost; }
  AGX_FORCE_INLINE void Job::setCostEstimate(agx::UInt32 cost) { m_cost = cost; }

  AGX_FORCE_INLINE agx::Index& Job::index() { return m_index; }
  AGX_FORCE_INLINE const agx::Index& Job::index() const { return m_index; }


  //// JOB GROUP /////////////////////////////////////////////////////

  AGX_FORCE_INLINE JobGroup::JobGroup() // : Job(Callback(&JobGroup::execute, this))
  {
  }

  AGX_FORCE_INLINE void JobGroup::init()
  {
    // Job::init(Callback(&JobGroup::execute, this));
    Job::init(Callback());
    m_jobs.clear();
    m_cost = 0;
  }

  AGX_FORCE_INLINE void JobGroup::addJob(Job *job)
  {
    m_jobs.push_back(job);
    m_cost += job->getCostEstimate();
  }

  AGX_FORCE_INLINE const JobPtrVector& JobGroup::getSubJobs() const
  {
    return m_jobs;
  }



  //// RANGE JOB /////////////////////////////////////////////////////
  AGX_FORCE_INLINE RangeJob::RangeJob() // : Job(Callback(&RangeJob::dispatch, this))
  {}

  AGX_FORCE_INLINE RangeJob::RangeJob(const RangeCallback& callback, const IndexRange& range, agx::UInt32 costPerElement) // : Job(Callback(&RangeJob::dispatch, this))
  {
    this->init(callback, range, costPerElement);
  }

  AGX_FORCE_INLINE RangeJob::RangeJob(const RangeJob& other) : Job(other) // : Job(Callback(&RangeJob::dispatch, this))
  {
    *this = other;
  }

  AGX_FORCE_INLINE RangeJob& RangeJob::operator= (const RangeJob& other)
  {
    this->init(other.m_rangeCallback, other.m_range);
    return *this;
  }

  AGX_FORCE_INLINE IndexRange& RangeJob::range() { return m_range; }
  AGX_FORCE_INLINE const IndexRange& RangeJob::range() const { return m_range; }


  AGX_FORCE_INLINE void RangeJob::init(const RangeCallback& callback, const IndexRange& range, agx::UInt32 costPerElement)
  {
    m_range = range;
    m_rangeCallback = callback;

    Job::init(Callback(), (agx::UInt32)m_range.size() * costPerElement);
  }

  AGX_FORCE_INLINE void RangeJob::init(const IndexRange& range, agx::UInt32 costPerElement)
  {
    m_range = range;
    Job::init(Callback(), (agx::UInt32)m_range.size() * costPerElement);
  }

  /////////////////////////////////////////////////////////

  template <class T>
  AGX_FORCE_INLINE RangeJobT<T>::RangeJobT()
  {}

  template <class T>
  AGX_FORCE_INLINE RangeJobT<T>::RangeJobT(const RangeCallback& callback, const IndexRange& range, agx::UInt32 costPerElement)
      : RangeJob(callback, range, costPerElement)
  {}

  template <class T>
  AGX_FORCE_INLINE RangeJobT<T>::RangeJobT(const RangeJobT<T>& other) : RangeJob(other)
  {}

  template <class T>
  RangeJobT<T>::~RangeJobT()
  {
  }

  template <class T>
  void RangeJobT<T>::implementation()
  {
    agxAssert(dynamic_cast<T *>(m_task));
    static_cast<T *>(m_task)->dispatch(*this);
  }

  //// TASK JOB /////////////////////////////////////////////////////
  AGX_FORCE_INLINE TaskJob::TaskJob() : Job(Callback(&TaskJob::execute, this)), m_target(nullptr)
  {
  }

  AGX_FORCE_INLINE TaskJob::TaskJob(Task* task) : Job(Callback(&TaskJob::execute, this)), m_target(task)
  {
  }

  AGX_FORCE_INLINE void TaskJob::init(Task *task)
  {
    Job::init(Callback(&TaskJob::execute, this));
    m_target = task;
  }

  AGX_FORCE_INLINE void TaskJob::init()
  {
    Job::init(Callback(&TaskJob::execute, this));
  }

  //// WAIT JOB /////////////////////////////////////////////////////

  AGX_FORCE_INLINE Job *WaitJob::getTarget() { return m_target; }
  AGX_FORCE_INLINE const Job *WaitJob::getTarget() const { return m_target; }

}

AGX_TYPE_BINDING(agx::Job, "Job")
AGX_TYPE_BINDING(agx::RangeJob, "RangeJob")
AGX_TYPE_BINDING(agx::JobGroup, "JobGroup")
AGX_TYPE_BINDING(agx::TaskJob, "TaskJob")


#ifdef _MSC_VER
# pragma warning(pop)
#endif

#endif /* _AGX_JOB_H_ */
