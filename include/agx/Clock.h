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
#ifndef AGX_CLOCK_H
#define AGX_CLOCK_H


#include <agxData/Value.h>
#include <agx/Task.h>


namespace agx
{
  class TiXmlElement;

  AGX_DECLARE_POINTER_TYPES(Clock);
  AGX_DECLARE_VECTOR_TYPES(Clock);

  /**
  Clock class keeps representation of virtual/simulated time.
  Clocks can be connected in a hierarchy using different frequencies.
  Tasks can be triggered when the clock ticks.
  Events can be scheduled with specified interval.
  */
  class AGXCORE_EXPORT Clock : public Component
  {
  public:
    static agx::Model *ClassModel();

    static Clock *load(TiXmlElement *eClock, Device *device);
    virtual void configure(TiXmlElement *eClock) override;
    virtual void save(TiXmlElement *eParent) const override;


    AGX_DECLARE_POINTER_TYPES(FrameWrapper);
    AGX_DECLARE_VECTOR_TYPES(FrameWrapper);

    AGX_DECLARE_POINTER_TYPES(Event);
    AGX_DECLARE_VECTOR_TYPES(Event);

  public:
    typedef Event1<Clock *> TickEvent;
    TickEvent preTick;
    TickEvent postTick;

  public:
    Clock(const agx::Name& name = "clock", agx::Real frequency = 60);

    /**
    Advance the clock to the next time step.
    \param blocking Will wait for all tick tasks if true
    */
    void tick(bool blocking = true);

    /**
    Advance the clock to a specified time.
    \param time The specified time
    */
    void stepTo(agx::Real time);

    /**
    Advance the clock a specified interval.
    \param delta The interval
    */
    void stepForward(agx::Real delta);

    /**
    Set the time explicitly, without ticking.
    */
    void setTime(agx::Real64 time);

    /**
    \return The current time.
    */
    agx::Real getTime() const;

    /**
    \return true if the clock is enabled.
    */
    agx::Bool isEnabled() const;

    /**
    Set to false to disable the clock, tick will be void operation.
    */
    void setEnable(agx::Bool flag);

    /**
    \return The frequency of the clock, 1/timeStep
    */
    agx::Real getFrequency() const;

    /**
    Set the clock frequency.
    */
    void setFrequency(agx::Real frequency);

    /**
    \return The current frame index.
    */
    agx::UInt getFrame() const;

    /**
    \return The current time step.
    */
    agx::Real getTimeStep() const;

    /**
    Set the time step of the clock, 1/frequency
    */
    bool setTimeStep(agx::Real dt);

    /**
    \return The time modulation of the clock.
    */
    agx::Real getTimeModulation() const;

    /**
    Set the time modulation of the clock, default is 1.0
    */
    void setTimeModulation(agx::Real modulation);

    /**
    Add an event to the clock to be triggered as specified by the event.
    \param event The event
    */
    void addEvent(agx::Clock::Event *event);

    /**
    Remove an event from the clock.
    \param event The event
    */
    void removeEvent(agx::Clock::Event *event);

    /**
    Add a job to be executed on each tick.
    \param job The job
    */
    void addFrameJob(agx::Job *job);

    /**
    Remove a frame job.
    \param job The job
    */
    void removeFrameJob(agx::Job *job);

    /**
    Add a task to be executed on each tick.
    \param task The task
    */
    void addFrameTask(agx::Task *task);

    /**
    Remove a frame task.
    \param task The task
    */
    void removeFrameTask(agx::Task *task);

    /**
    Add a slave clock, to be driven by this clock.
    \param slave The slave clock
    */
    void addSlave(agx::Clock *slave);

    /**
    Explicitly set the current frame index.
    */
    void setFrame(UInt frame);

    /**
    Remove a slave clock.
    \param slave The slave clock
    */
    void removeSlave(agx::Clock *slave);

    /**
    \return The list of slaves.
    */
    const agx::ClockPtrVector& getSlaves() const;

    /**
    \return The reference clock to which this clock is a slave.
    */
    agx::Clock *getReference();


    /**
    Reset to frame zero (and time=0), remove all events.
    */
    void reset();

    /// Bind the clock to a reference clock using a bindpath.
    void bind(const agx::Path& path);

    /// \return The bind path
    const agx::Path& getBindPath() const;

    virtual void rebind() override;

    /**
    Control whether this Clock is the main clock for thread timeline generation.
    Default is true. Set to false for example when using a sub-simulation that
    is stepped from within the main simulation.
    */
    void setIsThreadTimelineResponsible(bool isResponsible);

    /// \return True if this clock is the main thread timeline clock. False otherwise.
    bool getIsThreadTimelineResponsible() const;

  protected:
    virtual ~Clock();

    // virtual bool bind(Object *object) override;

  private:
    void slaveTick();
    void rebuild( Real oldTimeStep );
    void schedule(Event *event);
    void schedule(Event *event, Real timeUntilDispatch);
    void unschedule(Event *event);
    void reschedule(Event *event);

    void enableCallback(agxData::Value *);
    void calculateTimestep();

  private:
    agxData::ValueRefT<Bool> m_enabled;
    agxData::ValueRefT<Real64> m_time;
    agxData::ValueRefT<Real> m_frequency;
    agxData::ValueRefT<Real> m_timeModulation;
    agxData::ValueRefT<Real> m_timeStep;
    agxData::ValueRefT<UInt> m_frame;

    agxData::Value::Event::CallbackType m_updateTimeStepCallback;
    agxData::Value::Event::CallbackType m_enableCallback;

    Path m_bindPath;

    ClockRef m_reference;
    ClockPtrVector m_slaves;
    EventRef m_syncEvent;

    class Frame;
    typedef HashTable<UInt, Frame> FrameTable;
    // typedef VectorPOD<Frame *> FramePtrVector;
    FrameTable m_frameTable;
    EventPtrVector m_events;
    // EventPtrVector m_tickEvents;
    TaskObserverVector m_frameTasks;
    JobPtrVector m_frameJobs;
    // Job m_framePostJob;
    FrameWrapperRef m_frameTaskWrapper;
    bool m_slaveTrigger;
    bool m_isTimelineResponsible = true;
  };

  //////////////////////////////////////////////////////////////////////////////////////////////////

  /**
  An event to be triggered after a specified time. Optionally repeating with a specified interval.
  */
  class AGXCORE_EXPORT Clock::Event : public Component, public agx::Event
  {
  public:
    static agx::Model *ClassModel();

    static Clock::Event *load(TiXmlElement *eEvent, Device *device);
    virtual void configure(TiXmlElement *eEvent) override;

  public:

    /** Constructor */
    Event(const agx::Name& name = "Event");
    Event(const agx::Callback& callback, agx::Real startDelay, agx::Real interval = -1.0);

    /**
    \return The clock which is responsible for triggering the event.
    */
    agx::Clock *getClock();

    /**
    Set the start delay when added to a clock or reset.
    \param time - in seconds
    */
    void setStartDelay(agx::Real time);


    /**
    \return the start delay when added to a clock or reset.
    */
    agx::Real getStartDelay() const;

    /**
    Set the trigger interval.
    \todo Use <= 0 to indicate repeat on/off?
    */
    void setInterval(agx::Real interval);


    /**
    Get the trigger interval.
    \return the trigger interval
    */
    agx::Real getInterval() const;

    /**
    Set if event trigger multiple times.
    \todo Set number of iterations/repetitions instead?
    */
    void setRepeating(bool flag);

    /**
    \return true if event trigger multiple times.
    */
    bool isRepeating() const;

    /**
    Pause/unpause
    */
    void setEnable(bool flag);

    /// Same as setEnable(true)
    void start();

    /// Same as setEnable(false)
    void stop();

    /**
    Reset event, with start delay.
    */
    void reset();

    ///
    size_t getIndex() const;

  protected:
    virtual ~Event();

  private:
    Real calculateTimeUntilDispatch( Real timeStep );
    Real calculateTimeUntilDispatch();
    void valueCallback(agxData::Value *value);

    friend class Clock;
    void setIndex(size_t index);
    Event *m_next;

    size_t m_index;
    Clock *m_clock;
    agxData::ValueRefT<Real> m_startDelay;
    agxData::ValueRefT<Real> m_interval;
    agxData::ValueRefT<Bool> m_repeating;
    agxData::ValueRefT<Bool> m_enabled;

    // CallbackVector m_callbacks;
    Callback m_mainCallback;
    Real m_subInterval; // Accumulator
    UInt m_scheduledFrame;
    agxData::Value::Event::CallbackType m_valueCallback;
  };


  //////////////////////////////////////////////////////////////////////////////////////////////////

  /// Internal class
  class Clock::Frame
  {
  public:
    Frame() {}
    Frame(UInt frameId) : id(frameId), eventList(nullptr) {}
    UInt id;
    Event *eventList;
    // Real m_time;
  };


  /// Internal class
  class Clock::FrameWrapper : public TaskGroup
  {
  public:
    FrameWrapper(Clock *clock);
    void addFrameTask(Task *child);

    void execute();
    Clock* getClock();

  protected:
    virtual ~FrameWrapper();

  private:
    Clock *m_clock;
    TaskRefVector m_children;
  };

  //////////////////////////////////////////////////////////////////////////////////////////////////

  DOXYGEN_START_INTERNAL_BLOCK()
  /* Implementation */
  AGX_FORCE_INLINE Clock *Clock::getReference() { return m_reference; }
  AGX_FORCE_INLINE const ClockPtrVector& Clock::getSlaves() const { return m_slaves; }

  AGX_FORCE_INLINE bool Clock::isEnabled() const { return m_enabled->get(); }
  AGX_FORCE_INLINE Real Clock::getTime() const { return agx::Real(m_time->get()); }
  AGX_FORCE_INLINE Real Clock::getFrequency() const { return m_frequency->get(); }
  AGX_FORCE_INLINE Real Clock::getTimeModulation() const { return m_timeModulation->get(); }
  AGX_FORCE_INLINE Real Clock::getTimeStep() const { return m_timeStep->get(); }
  AGX_FORCE_INLINE UInt Clock::getFrame() const { return m_frame->get(); }
  AGX_FORCE_INLINE const Path& Clock::getBindPath() const { return m_bindPath; }

  AGX_FORCE_INLINE Real Clock::Event::getStartDelay() const { return m_startDelay->get(); }
  AGX_FORCE_INLINE Real Clock::Event::getInterval() const { return m_interval->get(); }
  AGX_FORCE_INLINE bool Clock::Event::isRepeating() const { return m_interval->get() >= 0; }
  AGX_FORCE_INLINE Clock *Clock::Event::getClock() { return m_clock; }
  AGX_FORCE_INLINE size_t Clock::Event::getIndex() const { return m_index; }
  DOXYGEN_END_INTERNAL_BLOCK()

  // AGX_FORCE_INLINE const CallbackVector& Clock::Event::getCallbacks() const { return m_callbacks; }
}

#endif /* _AGX_CLOCK_H_ */
