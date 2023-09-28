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

#ifndef AGX_TASK_H
#define AGX_TASK_H

#include <agx/Job.h>
#include <agx/Device.h>
#include <agx/HashSet.h>
#include <agx/Component.h>
#include <agx/Parameter.h>
#include <agx/Referenced.h>
#include <agx/String.h>
#include <agx/Vector.h>
#include <agx/Statistics.h>

#include <agxData/Buffer.h>
#include <agx/ParameterBinding.h>


namespace agx
{
  class TiXmlElement;
  class Solver;
  class Thread;
  class Clock;
  class TaskTimingReportHandle;

  AGX_DECLARE_POINTER_TYPES(Task);
  AGX_DECLARE_VECTOR_TYPES(Task);
  AGX_DECLARE_POINTER_TYPES(SerialTask);
  AGX_DECLARE_POINTER_TYPES(ParallelTask);
  AGX_DECLARE_POINTER_TYPES(TaskGroup);




  /**
  A representation of a generic task. The task is defined by its implementation
  and a set of parameters. The parameters provide a common interface to bind the data
  accessed by the task. The tasks have dependencies to other tasks and can be
  organized in a task graph.
  */
  class AGXCORE_EXPORT Task : public Component
  {
  public:
    static agx::Model *ClassModel();

    // Allow concatenated syntax: path.to.tasks:implementationName
    static Task* load(const String& path, Device* device = CpuDevice::instance());
    static Task* load(const char* path, Device* device = CpuDevice::instance());

    static Task* load(TiXmlElement *eTask, Device* device);
    static Task* load(const Path& path, Device* device = CpuDevice::instance());
    // static Task *load(const String& path, Device *device, const String& implementation);
    static Task* load(const Path& path, const Name& implementation, Device* device = CpuDevice::instance());

    static Task* load(const Path& path, const Name& implementation, const agx::ParameterBindingVector& parameterList, Device* device = CpuDevice::instance());
    static Task* load(const Path& path, const Name& implementation, agx::ParameterBindingVector& parameterList, bool allowFormatPromotions, Device* device = CpuDevice::instance());

    virtual void configure(TiXmlElement* eTask) override;
    void configure(TiXmlElement* eTask, bool loadParameters);

    void save(agx::TiXmlElement* parent) const override;

    enum ProfilingMode
    {
      PROFILING_NONE,
      PROFILING_FAST,
      PROFILING_FULL
    };

    static void setProfilingMode(ProfilingMode mode);
    static ProfilingMode getProfilingMode();

  public:
    typedef Event1<Task *> ExecutionEvent;
    typedef Event2<Task *, Parameter *> ParameterEvent;

    /// Triggered before the task begins
    ExecutionEvent startEvent;

    /// Triggered after the task has completed
    ExecutionEvent completionEvent;

    /// Triggered when a parameter is added
    ParameterEvent addParameterEvent;

    /// Triggered when a parameter is removed
    ParameterEvent removeParameterEvent;

    /// Enable/disable the task, normally use setEnable/isEnabled instead
    agxData::Val<Bool> enableParameter;

  public:

    /// Used for tracing mismatched parameter bindings
    enum ParameterMatch {
        NO_EVENTS = 0,

        // Bits set to indicate events that does not imply a failed match.
        EMPTY_BINDING_LIST       = (1<<0),  // Matched by default since the list of supplied parameters was empty.
        PARAMETER_VALID          = (1<<1),  // A parameter was valid and therefore didn't need a new binding.
        PARAMETER_NOT_REQUIRED   = (1<<2),  // A parameter was marked as not required and therefore didn't need a new binding.
        PARAMETER_BOUND          = (1<<3),  // A parameter was already bound and therefore didn't need a new binding.
        PARAMETER_HAS_BOUND_PATH = (1<<4),  // A parameter has a bind path already and therefore didn't need a new binding.
        FORMATS_MATCH            = (1<<5),  // A parameter and a new binding had the same format.
        NEED_DEVICE_TRANSFORM    = (1<<6),  // The device has requested to get data of the new binding type in the format of the parameter. Transformation will be required.

        // Bits set to signal a failed match.
        MISSING_BINDING          = (1<<7),  // When a task has a parameter that needs a new binding but didn't get one.
        BINDING_LACKS_TYPE       = (1<<8),  // A new binding didn't have a type.
        FORMATS_MISSMATCH        = (1<<9),  // The format of a new binding was incompatible with the parameter format.
        EXTRA_BINDING            = (1<<10), // There were new bindings that didn't have a corresponding parameter.
        INCONSISTEN_PARAMETERS   = (1<<11), // Don't think this can ever happen. Means that a loop over all parameters didn't visit all of them.

        // Mask of all events that doesn't imply a failed match.
        // Note that the NO_MATCH_MASK test has precendence, so match-testing should only use that.
        MATCH_MASK = EMPTY_BINDING_LIST | PARAMETER_VALID | PARAMETER_NOT_REQUIRED | PARAMETER_BOUND | PARAMETER_HAS_BOUND_PATH | FORMATS_MATCH | NEED_DEVICE_TRANSFORM,

        // The following is the union of all events signaling a failed parameter match. Bitwise and between this mask and a ParameterMatch is == 0 when there is a match.
        NO_MATCH_MASK = MISSING_BINDING | BINDING_LACKS_TYPE | FORMATS_MISSMATCH | EXTRA_BINDING | INCONSISTEN_PARAMETERS
    };
    static void printParameterMatch( UInt32 parameterMatch );

  public:
    /**
    Constructor
    \param name The name of the task.
    */
    Task(const agx::Name& name, agx::Device *device = agx::CpuDevice::instance());

    /**
    Execute the task. The call will not return until the task execution has completed.
    */
    void execute();

    /**
    Block until the task has completed execution.
    */
    virtual void wait();

    /**
    \return True if the task is running.
    */
    bool isRunning() const;

    /**
    Enable/disable a task. When disabled, it is bypassed in the execution path.
    */
    void setEnable(bool flag);

    /**
    \return True if enabled.
    */
    bool isEnabled() const;

    /**
    Set a specific thread which must execute this task.
    */
    virtual void setTargetThread(agx::Thread *thread);

    /**
    \return The target thread.
    */
    agx::Thread *getTargetThread();

    /**
    \return The thread which called execute()
    */
    agx::Thread *getDispatchThread();

    /**
    Add a parameter.
    */
    void addParameter(agx::Parameter *parameter);
    void addParameter(agx::Parameter *parameter, size_t index);

    /**
    Remove a parameter.
    */
    void removeParameter(agx::Parameter *parameter);
    void removeParameter(size_t index);

    /**
    \return The parameter list.
    */
    const agx::ParameterPtrVector& getParameters() const;

    /**
    \return The parameter with a specified name.
    */
    agx::Parameter *getParameter(const agx::Name& name);
    const agx::Parameter *getParameter(const agx::Name& name) const;

    // ScalarParameter
    agx::ScalarParameter *getScalarParameter(const agx::Name& name);
    const agx::ScalarParameter *getScalarParameter(const agx::Name& name) const;

    // ArrayParameter
    agx::ArrayParameter *getArrayParameter(const agx::Name& name);
    const agx::ArrayParameter *getArrayParameter(const agx::Name& name) const;

    // PointerParameter
    agx::PointerParameter *getPointerParameter(const agx::Name& name);
    const agx::PointerParameter *getPointerParameter(const agx::Name& name) const;

    /**
    \return The parameter with a specified index.
    */
    agx::Parameter *getParameter(size_t index);
    const agx::Parameter *getParameter(size_t index) const;


    UInt32 matchParameters(const ParameterTable& bindings);
    UInt32 matchParameters(const agx::ParameterBindingVector& bindings);
    UInt32 matchParameters(agx::ParameterBindingVector& bindings, bool allowFormatPromotions);

    /**
    \return The task group which this task is part of.
    */
    agx::TaskGroup *getGroup();
    const agx::TaskGroup *getGroup() const;


    /**
    \return The dependencies for this task.
    */
    const agx::TaskPtrVector& getParents() const;

    /**
    \return The child tasks which depends on this task.
    */
    const agx::TaskPtrVector& getChildren() const;



    /**
    Add a dependency.
    \param parent The parent dependency.
    */
    void addDependency(agx::Task *parent);
    void addDependency(const agx::Path& path);
    void addDependencies(const agx::String& paths); // Semi-colon separated

    /**
    Remove a dependency.
    \param parent The parent dependency.
    */
    void removeDependency(agx::Task* parent);

    // Search recursively for a parent dependency
    bool hasDependency(agx::Task *parent) const;

    /**
    Replace the task with another task by rerouting the dependencies.
    */
    void replace(agx::Task *other);

    void traverse(const Callback1<Task *>& callback, bool breadthFirst = true);

    void print() const;
    void print(std::ostream& stream) const;
    void printBindings();
    void printBindings(std::ostream& stream);

    virtual void setContext( agx::Object* context ) override;
    void setModel(agx::Model *model);

  public:
    /**
    \return The pre-job for the task.
    */
    virtual agx::Job *getPreJob() = 0;

    /**
    \return The post-job for the task.
    */
    virtual agx::Job *getPostJob() = 0;



  public:
    virtual void rebind() override;

    void addTriggerPath(const agx::Path& path);
    void addTriggerPaths(const agx::String& paths);
    void addResolvePath(const agx::Path& path);
    void addResolvePaths(const agx::String& paths);

  public:

    /// Performance statistics


    /// \return The wall-time of the previous execution
    agx::Real32 getWallTime() const;

    /// \return The overhead time of the previous execution
    agx::Real32 getOverheadTime() const;

    /// \return The compute cost of the previous execution
    agx::Real32 getComputeCost() const;

    agx::Real32 getAccumulatedWallTime() const;
    agx::Real32 getAccumulatedOverheadTime() const;
    agx::Real32 getAccumulatedComputeCost() const;
    agx::UInt getNumAccumulationSamples() const;

    /// Reset accumulated performance statistics
    void resetAccumulationTimers(bool recursive = true);

    // private
    void beginCallback();
    void endCallback(agx::Real32 computeCost, agx::Real32 overheadCost = 0);

  protected:

    virtual ~Task();


    // void setGroup(Task *group);
    void enableCallback(agxData::Value *enable);

    static void addConnection(Task *parent, Task *child);
    static void removeConnection(Task *parent, Task *child);

    virtual void parameterAddedCallback(Parameter* /*parameter*/) {}
    virtual void parameterRemovedCallback(Parameter* /*parameter*/) {}
    virtual void subtaskAddedCallback(Task* /*subtask*/) {}

    // Must be called after the number of kernel jobs is known, during dispatch
    void prepareGlobalResultSorting(size_t numJobs);

  protected:
    friend class Component;
    friend class UnresolvedTask;
    friend class TaskManager;
    friend class Solver;

    void init();

    // void defaultRunCallback();
    void registerBufferReadersAndWriters();
    void unregisterBufferReadersAndWriters();
    void verifyParameters();
    void disconnectFromParent(Task* ancestor);
    void disconnectFromChild(Task* descendant);
    void connectToParent(Task* ancestor);
    void connectToChild(Task* descendant);
    void dispatchBlocking(Thread *thread);

    friend class TaskGroup;
    void setGroup(TaskGroup *group);


    friend class Thread;
    static void signalUnhandledException();
    static bool hasUnhandledException();

#ifndef SWIG
    virtual Object *getResourceImpl(const Path& path, agx::Model *model) override;
#endif

  private:
    bool m_enabled;
    std::atomic<bool> m_running;

    Thread *m_targetThread;
    Thread *m_dispatchThread;
    Timer m_overheadTimer;
    // Timer m_runTimer;
    Timer m_wallTimer;



    // Currently not decided how task timer reporting will be handled, with internal Real32s or through Statistics.
    // Keeping both as part of the object for now since we do not want to have to change this header file when
    // changing the reporting strategy.
    Real32 m_wallTime;
    Real32 m_computeCost;
    Real32 m_overheadTime;
    Real32 m_accumulatedWallTime;
    Real32 m_accumulatedComputeCost;
    Real32 m_accumulatedOverheadTime;
    UInt m_numAccumulations;

    TaskTimingReportHandleRef m_statisticsHandle;


    TaskGroup *m_group;
    TaskPtrVector m_parents;
    TaskPtrVector m_children;

    ParameterPtrVector m_parameters;
    ParameterPtrVector m_globalResultParameters;
    agxData::Value::Event::CallbackType m_enableCallback;

    struct BindPath
    {
      enum Type
      {
        INVALID=-1,
        DEPENDENCY,
        RESOLVE,
        TRIGGER
      };

      BindPath() : type((Type)agx::InvalidIndex), binding(nullptr) {}
      BindPath(const Path& _path, Type _type) : path(_path), type(_type), binding(nullptr) {}

      Path path;
      Type type;
      ObjectRef binding;
    };

    Vector<BindPath> m_bindPaths;
    Path m_bindContextPath;
    ObjectObserver m_bindContext;

    AGX_DECLARE_POINTER_TYPES(DisableHelper);
    class DisableHelper : public Referenced
    {
    public:
      DisableHelper(Task *task);

      void restore();

      void registerSubtask(Task* subtask);
      void removeSubtask(Task* subtask);

    protected:
      virtual ~DisableHelper();

    private:
      void emptyPre();
      void emptyPost();

    private:
      Task *m_task;
      JobPtrVector m_oldPreChildren;
      UInt32 m_oldNumPostDependencies;
      Callback m_oldPre;
      Callback m_oldPost;
    };

    DisableHelperRef m_disableHelper;

    static ProfilingMode s_profilingMode;
  };


  /**
  SerialTask
  */
  class AGXCORE_EXPORT SerialTask : public Task
  {
  public:
    SerialTask(const agx::Name& name, agx::Device *device = agx::CpuDevice::instance());

    SerialTask(const agx::Name& name, const Job::Callback& runCallback, agx::Device* device = agx::CpuDevice::instance());

    /**
    Specify the task dispatch to execute.
    */
    void setDispatch(const Job::Callback& dispatchCallback);


    /**
    Execution callback. Override in subclass, only called if no dispatchCallback is registered.
    */
    virtual void run() {}


    Job *getJob();

    virtual agx::Job *getPreJob() override;
    virtual agx::Job *getPostJob() override;
    virtual void setTargetThread(agx::Thread *thread) override;

  protected:
    virtual ~SerialTask();

  private:
    class SerialJob : public Job
    {
    public:
      virtual ~SerialJob();

    private:
      virtual void implementation() override;
    };

  private:
    SerialJob m_job;
  };


  /**
  ParallelTask
  */
  class AGXCORE_EXPORT ParallelTask : public Task
  {
  public:
    ParallelTask(const agx::Name& name, agx::Device *device = agx::CpuDevice::instance());
    ParallelTask(const agx::Name& name, const agx::Job::Callback& dispatchCallback, agx::Device* device = agx::CpuDevice::instance());

    /**
    Add a job to the task.
    */
    void addJob(agx::Job *job);

    /**
    Remove all jobs (except the original pre/dispatch/post jobs)
    */
    void removeAllJobs();

    Job *getDispatchJob();

    void setDispatch(const agx::Job::Callback& dispatchCallback);

    virtual agx::Job *getPreJob() override;
    virtual agx::Job *getPostJob() override;
    virtual void setTargetThread(agx::Thread *thread) override;

  protected:
    virtual ~ParallelTask();

    friend class Solver;
    void endCallback();

  private:
    class PreJob : public Job
    {
    public:
      virtual ~PreJob();

    private:
      virtual void implementation() override;
    };

    class PostJob : public Job
    {
    public:
      virtual ~PostJob();

    private:
      virtual void implementation() override;
    };

    class DispatchJob : public Job
    {
    public:
      virtual ~DispatchJob();

    private:
      virtual void implementation() override;
    };

  private:
    PreJob m_preJob;
    PostJob m_postJob;
    DispatchJob m_dispatchJob;
    JobPtrVector m_jobStack;
    Index m_jobIndexCounter;
  };


  /**
  TaskGroup
  */
  class AGXCORE_EXPORT TaskGroup : public Task
  {
  public:
    TaskGroup(const agx::Name& name, bool isSequential = true, agx::Device *device = agx::CpuDevice::instance());

    /// \return True if the task group is sequential
    bool isSequential() const;

    void setSequential(bool flag);

    /**
    Add a subtask. If parallel dispatch is enabled subtasks will not receive any dependencies to each other,
    otherwise dependencies are added between subtasks (in the order they are added) to ensure serial execution.
    */
    void addSubtask(agx::Task *subtask);

    /// Add a subtask between two existing tasks
    void addSubtaskBetween(agx::Task *subtask, agx::Task *parent, agx::Task *child);
    void addSubtaskBetween(agx::Task *subtask, const agx::String& parent, const agx::String& child);

    /// Add a subtask to be executed after a specified task
    void addSubtaskAfter(agx::Task *subtask, agx::Task *parent);
    void addSubtaskAfter(agx::Task *subtask, const agx::String& parent);

    /// Add a subtask to be executed before a specified task
    void addSubtaskBefore(agx::Task *subtask, agx::Task *child);
    void addSubtaskBefore(agx::Task *subtask, const agx::String& child);

    /**
    Remove a subtask.
    \param subtask The subtask
    */
    void removeSubtask(agx::Task *subtask);

    /**
    Remove a subtask by name.
    \param name The name of the subtask.
    */
    void removeSubtask(const agx::Name& name);

    /**
    Remove all subtasks.
    */
    void removeAllSubtasks();

    /**
    Serialize the task. Non-reversible.
    */
    void serialize(bool recursive = false);

    /**
    \return The subtasks for the task.
    */
    const agx::TaskRefVector& getSubtasks() const;

    /**
    \return A subtask with a specified name, the index is used if there are multiple subtasks with the same name.
    */
    agx::Task *getSubtask(const agx::String& name, size_t index = 0);


    virtual void traverse(const Callback1<Task *>& callback, bool breadthFirst = true);

    void writeDependencyDotGraph() const;


    virtual agx::Job* getPreJob() override;
    virtual agx::Job* getPostJob() override;
    virtual void setTargetThread(agx::Thread* thread) override;

  protected:
    virtual ~TaskGroup();

  private:
    friend class Task;
    void registerSubtask(Task* subtask);
    void endCallback();

  private:
    class PreJob : public Job
    {
    public:
      virtual ~PreJob();

    private:
      virtual void implementation() override;
    };

    class PostJob : public Job
    {
    public:
      virtual ~PostJob();

    private:
      virtual void implementation() override;
    };

  private:
    friend class Solver;

    bool m_sequential;
    PreJob m_preJob;
    PostJob m_postJob;
    TaskRefVector m_subtasks;
  };

  /* Implementation */
  AGX_FORCE_INLINE Task::ProfilingMode Task::getProfilingMode() { return s_profilingMode; }

  AGX_FORCE_INLINE bool Task::isEnabled() const { return m_enabled; }
  AGX_FORCE_INLINE bool Task::isRunning() const { return m_running; }
  AGX_FORCE_INLINE TaskGroup *Task::getGroup() { return m_group; }
  AGX_FORCE_INLINE const TaskGroup *Task::getGroup() const { return m_group; }


  AGX_FORCE_INLINE const TaskPtrVector& Task::getParents() const { return m_parents; }
  AGX_FORCE_INLINE const TaskPtrVector& Task::getChildren() const { return m_children; }
  AGX_FORCE_INLINE Thread *Task::getTargetThread() { return m_targetThread; }


  AGX_FORCE_INLINE const ParameterPtrVector& Task::getParameters() const { return m_parameters; }
  AGX_FORCE_INLINE Parameter *Task::getParameter(const Name& name) { return this->getObject<Parameter>(name); }
  AGX_FORCE_INLINE const Parameter *Task::getParameter(const Name& name) const { return const_cast<Task *>(this)->getParameter(name); }

  AGX_FORCE_INLINE ScalarParameter *Task::getScalarParameter(const Name& name) { return this->getObject<ScalarParameter>(name); }
  AGX_FORCE_INLINE const ScalarParameter *Task::getScalarParameter(const Name& name) const { return const_cast<Task *>(this)->getScalarParameter(name); }

  AGX_FORCE_INLINE ArrayParameter *Task::getArrayParameter(const Name& name) { return this->getObject<ArrayParameter>(name); }
  AGX_FORCE_INLINE const ArrayParameter *Task::getArrayParameter(const Name& name) const { return const_cast<Task *>(this)->getArrayParameter(name); }

  AGX_FORCE_INLINE PointerParameter *Task::getPointerParameter(const Name& name) { return this->getObject<PointerParameter>(name); }
  AGX_FORCE_INLINE const PointerParameter *Task::getPointerParameter(const Name& name) const { return const_cast<Task *>(this)->getPointerParameter(name); }

  AGX_FORCE_INLINE Parameter *Task::getParameter(size_t index) { return m_parameters[index]; }
  AGX_FORCE_INLINE const Parameter *Task::getParameter(size_t index) const { return m_parameters[index]; }

  AGX_FORCE_INLINE Thread *Task::getDispatchThread() { return m_dispatchThread; }

  AGX_FORCE_INLINE Real32 Task::getWallTime() const { return m_wallTime; }
  AGX_FORCE_INLINE Real32 Task::getComputeCost() const { return m_computeCost; }
  AGX_FORCE_INLINE Real32 Task::getOverheadTime() const { return m_overheadTime; }
  AGX_FORCE_INLINE Real32 Task::getAccumulatedWallTime() const { return m_accumulatedWallTime; }
  AGX_FORCE_INLINE Real32 Task::getAccumulatedOverheadTime() const { return m_accumulatedOverheadTime; }
  AGX_FORCE_INLINE Real32 Task::getAccumulatedComputeCost() const { return m_accumulatedComputeCost; }
  AGX_FORCE_INLINE UInt Task::getNumAccumulationSamples() const { return m_numAccumulations; }



  // TaskGroup
  AGX_FORCE_INLINE bool TaskGroup::isSequential() const { return m_sequential; }
  AGX_FORCE_INLINE const TaskRefVector& TaskGroup::getSubtasks() const { return m_subtasks; }


}

#endif /* _AGX_TASK_H_ */
