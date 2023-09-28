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

#ifndef AGX_STATISTICS_H
#define AGX_STATISTICS_H

#include <agx/config/AGX_USE_STATISTICS.h>
#include <agx/Referenced.h>
#include <agx/Path.h>
#include <agx/Timer.h>
#include <agx/Math.h>
#include <agx/Vector.h>
#include <agx/HashSet.h>
#include <agxData/Format.h>

#include <agx/Logger.h>
#include <agxData/Value.h>

#if AGX_USE_STATISTICS()

#define AGX_STATISTICS_REGISTER_DATA_PROVIDER(INSTANCE_ADDRESS, MODULE_NAME, CONTEXT_ADDRESS) \
  agx::Statistics::instance()->registerDataProvider( INSTANCE_ADDRESS, MODULE_NAME, CONTEXT_ADDRESS )

#define AGX_STATISTICS_REGISTER_ROOT_DATA_PROVIDER(INSTANCE_ADDRESS, MODULE_NAME) \
  agx::Statistics::instance()->registerDataProvider( INSTANCE_ADDRESS, MODULE_NAME )

#define AGX_STATISTICS_UNREGISTER_DATA_PROVIDER(INSTANCE_ADDRESS) \
  agx::Statistics::instance()->unregisterDataProvider( INSTANCE_ADDRESS )


#define AGX_STATISTICS_REPORT_DATA(INSTANCE_ADDRESS, DATA_NAME, VALUE) \
  agx::Statistics::instance()->report( INSTANCE_ADDRESS, DATA_NAME, VALUE )

#define AGX_STATISTICS_REPORT_TIMING(INSTANCE_ADDRESS, DATA_NAME, VALUE) \
  agx::Statistics::instance()->reportTiming( INSTANCE_ADDRESS, DATA_NAME, VALUE )

#define AGX_STATISTICS_REPORT_ADDITIONAL_TIMING(INSTANCE_ADDRESS, DATA_NAME, VALUE) \
  agx::Statistics::instance()->reportAdditionalTiming( INSTANCE_ADDRESS, DATA_NAME, VALUE )


#else


#define AGX_STATISTICS_REGISTER_DATA_PROVIDER(INSTANCE_ADDRESS, MODULE_NAME, CONTEXT_ADDRESS)

#define AGX_STATISTICS_REGISTER_ROOT_DATA_PROVIDER(INSTANCE_ADDRESS, MODULE_NAME)

#define AGX_STATISTICS_UNREGISTER_DATA_PROVIDER(INSTANCE_ADDRESS)


#define AGX_STATISTICS_REPORT_DATA(INSTANCE_ADDRESS, DATA_NAME, VALUE)

#define AGX_STATISTICS_REPORT_TIMING(INSTANCE_ADDRESS, DATA_NAME, VALUE)

#define AGX_STATISTICS_REPORT_ADDITIONAL_TIMING(INSTANCE_ADDRESS, DATA_NAME, VALUE)


#endif




namespace agx
{
  class Task;
  AGX_DECLARE_POINTER_TYPES(TaskTimingReportHandle);

  struct TimingInfo {
    // Most people report these as milliseconds.
    double current;
    double accumulated;
  };


  /**
  Base statistics gathering class. It is a Singleton  and the
  default instance can be accessed using 'Statistics::instance()'. Any object
  can report data to an instance of this class and all tasks will report their
  profiling timer values to the default Statistics object.

  Data is stored hierarchically, so each provider must register themselves with
  a context. The context is what defines the parent node in the statistics
  tree.
  */
  class AGXCORE_EXPORT Statistics : public Singleton
  {
  public:
    AGX_DECLARE_POINTER_TYPES(ProviderHandle);
    AGX_DECLARE_VECTOR_TYPES(ProviderHandle);
    AGX_DECLARE_POINTER_TYPES(AbstractData);

    template <typename T> class Data;

    typedef HashTable<const char*, AbstractDataRef> DataTable;

  public:

    /** Enables or disables gathering of statistics data.*/
    void setEnable( bool );

    /** Returns true if this Statistics object is currently accepting new data.*/
    bool getEnable() const;

    /** Returns true if this Statistics object is currently accepting new data.*/
    bool isEnabled() const;

    /**
    Register a new data provider. This must be done before any data can be
    reported from that provider.

    The 'context' parameter defines the location of the new provider handle node
    in the statistics tree. It be the same value as the 'provider' argument
    for another call to 'registerDataProvider(*)', past or future. It may also be
    nullptr, which will place the new provider handle node below the root of the
    tree.

    \param provider  A unique identifier for the new provider. Often the 'this' pointer in the calling code.
    \param name      A name for the new provider.
    \param context   Provider for the parent node in the statistics tree. May be nullptr.
    */
    bool registerDataProvider( void* provider, const char* name, void* context = (void*)nullptr );

    /**
    Convenience method, calls the const char* version of registerDataProvider(*).
    */
    bool registerDataProvider( void* provider, const String& name, void* context = (void*)nullptr );

    /**
    Used by the tasks to create a provider handle designed for task timer reporting.
    */
    TaskTimingReportHandle* registerTaskTimerProvider( void* provider, const char* name, void* context );


    /**
    Removes the provider handle for the given provider from the statistics tree.
    Any remaining children are moved to the removed node's parent.
    */
    bool unregisterDataProvider( void* provider );

    /**
    Move a sub-tree to a new location.
    \param instance  The identifier for the statistics node that should be moved.
    \param context   The identifier for the statistics node that should be the new parent.
    */
    void setContext(void* instance, void* context);

    /**
    Report statistics. The data element named 'name' in the provider handle for the
    provider 'instance' will be set to 'value' and the corresponding accumulated value
    will be incremented.
    \param instance  A unique identifier for a data provider. Must have been part of a prior

                     call to 'registerDataProvider(*) or 'registerTaskTimerProvider(*)'.
    \param name      The name of the reported data element.
    \param value     The value being reported.
    \return A handle to the updated statistics value.
    */
    template <typename T>
    AbstractData* report( void* instance, const char* name, const T& value );

    /**
    Report a timing statistics. Functionally very similar to 'report(*)'.

    \param instance  A unique identifier for a data provider. Must have been part of a prior
                     call to 'registerDataProvider(*) or 'registerTaskTimerProvider(*)'.
    \param name      The name of the reported data element.
    \param timing    The timing value being reported.
    \return A handle to the updated statistics value.
    */
    template <typename T>
    AbstractData* reportTiming( void* instance, const char* name, const T& timing );

    /**
    Convenience report method that reads the current time from a timer.
    */
    AbstractData* reportTiming( void* instance, const char* name, const Timer& timer );


    /**
    Increments the chosen timing value instead of setting it.
    */
    template <typename T>
    AbstractData* reportAdditionalTiming( void* instance, const char* name, const T& timing );

    /**
    Convenience method that reads the current value from a timer.
    */
    AbstractData* reportAdditionalTiming( void* instance, const char* name, const Timer& timer );

#if 0
// This method was believed to be required, but apparently not yet.

    /**
    Convenience method that does both reporting and a tree reorganization, if necessary.

    \param instance  A unique identifier for a data provider. Must have been part of a prior
                     call to 'registerDataProvider(*) or 'registerTaskTimerProvider(*)'.
    \param context   Identifier for that node that should be the parent of the reporting node.
    \param name      The name of the reported data element.
    \param value     The value being reported.
    */
    template <typename T>
    AbstractData* report(void* instance, void* context, const char* name, const T& value);
#endif

    /**
    Get a data handle from a statistics handle.
    \param instance  Identifier for the data provider.
    \param name      The name of the data to fetch.
    */
    AbstractData* getData( void* instance, const char* name );


    /**
    Get a data handle from the statistics tree. Since there may be many providers with the same name,
    a index is used to identify a particular provider.

    \param providerName  The name of a data provider that reported the data sample we should get.
    \param dataName      The name of the data sample to get.
    \param providerIndex Index into the list of providers with the given name.
    */
    AbstractData* getData( const char* providerName, const char* dataName, size_t providerIndex = 0 );


    /**
    Get a data handle from the statistics tree. Convenience method using strings instead of pointers.

    \param providerName  The name of a data provider that reported the data sample we should get.
    \param dataName      The name of the data sample to get.
    \param providerIndex Index into the list of providers with the given name.
    */
    AbstractData* getData( const String& providerName, const String& dataName, size_t providerIndex = 0 );

    /**
    Helper for C# swig binding. Will find a provider by name, and data by name, and return a string with timing info, or an empty string on fail.
    */
    String getTimingString( const String& providerName, const String& dataName, size_t providerIndex = 0 );

    /**
    Will find a provider by name, and data by name, and return the stored statistics data.
    Returns -1,-1 on fail.
    */
    TimingInfo getTimingInfo( const String& providerName, const String& dataName, size_t providerIndex = 0 );


    /**
    Get a data handle from the statistics tree. The tree is traversed from the provider identified by the
    'instance' along the given path. If there are ambiguous steps in the path, i.e., multiple children
    with the same name, then the first match in each such case will be used.

    \param instance  Identifier for the statistics node where the path traversal starts.
    \param path      Path from the statistics node for the given identifier to the wanted statistics element.
    */
    AbstractData* getData( void* instance, const Path& path );

    /**
    Templated convenience method that returns the result from the non-templated 'getData(*)' with the same
    arguments as a typed data handle.
    */
    template <typename T>
    Data<T>* getData(void* instance, const char* name);


    /**
    Returns the provider handle for the given statistics provider.
    */
    ProviderHandle* getProviderHandle( void* instance );

    /**
    Returns a handle to the root of the statistics tree.
    */
    ProviderHandle* getRootProviderHandle();

    /**
    Returns handles to all providers that have been registered with the given name.
    */
    const ProviderHandleRefVector& getProviderHandles( const char* name );

    /**
    Reset all statistics values, setting both the current and accumulated value using the
    type's default constructor.
    */
    void clear();

    /**
    Reset all statistics values, setting the current value using the type's default constructor.
    */
    void clearCurrent();

    /**
    Print print statistics in a human-readable form to the given stream. If 'includeTaskTimers' is
    true, then the task timer values will be included as well, which may produce quite a bit of output.

    \param stream            The stream to print to.
    \param includeTaskTimers If true, then the output will include profiling information for every executed task.
    */
    void print( std::ostream& stream, bool includeTaskTimers = false );

    /**
    Print print statistics in a human-readable form to the given stream. If 'includeTaskTimers' is
    true, then the task timer values will be included as well, which may produce quite a bit of output.

    \param includeTaskTimers If true, then the output will include profiling information for every executed task.
    */
    String printed( bool includeTaskTimers = false );



    Statistics();


    static Statistics *instance();
    static bool hasInstance();
    SINGLETON_CLASSNAME_METHOD();


  protected:

    void createRootNode();

    /** Remove implicit childless nodes up a branch.*/
    void decayBranch( ProviderHandle* leaf );

    /**
    All strings held inside the statistics tree is owned by the statistics tree. This method
    is used to make a copy of incoming strings so we know how and when to delete them.
    */
    static const char* copyString( const char* );

    virtual ~Statistics();

  private:
    ProviderHandle* getOrCreateParentNode( void* context );

    friend class ProviderHandle;
    /**
    Creates name binding.
    \param name - Must not be nullptr.
    \param provider - Must not be nullptr.
    \retval: Did it succeed?
    */
    bool createNameBinding( const char* name, ProviderHandle* provider );

    /**
    Remove name binding.
    \param provider - Must not be nullptr.
    \retval: Did it succeed?
    */
    bool removeNameBinding( ProviderHandle* provider );

    static Statistics* createInstance();

    ReentrantMutex& getTypeSystemMutex() const;

  private:

    static Statistics* s_instance;
    ReentrantMutex m_mutex;


    typedef HashTable< const char*, ProviderHandleRefVector> DataProviderNameTable;
    typedef HashTable< void*, ProviderHandleRef > DataProviderTable;
    DataProviderNameTable m_providerNames;
    DataProviderTable m_providers;
    ProviderHandleRef m_rootProvider;
    bool m_enabled;
  };


  /**
  Internal representation of the data providers. Keeps a list of child ProviderHandles and a list
  of data reported by the reporter for the ProviderHandle.
  */
  class AGXCORE_EXPORT Statistics::ProviderHandle : public Referenced
  {
  public:

    /**
    \param owner A unique identifier for the new ProviderHandle.
    \param statistics  The statistics object that is responsible for the new ProviderHandle.
    */
    ProviderHandle( void* owner, Statistics* statistics );

    /**
    \return The name given to this ProviderHandle.
    */
    const char* getName();

    /**
    Set the name of the provider
    */
    void setName(const char *name);

    /**
    \return The 'provider' passed when the data provider was registered.
    */
    void *getOwner();
    const void *getOwner() const;


    /**
    Convenience method that test if the owner was of a specific type and if so returns a
    pointer to the owner cast to that type.
    */
    template <typename T>
    T *getOwner();

    template <typename T>
    const T *getOwner() const;

    /**
    \param name  The name of a data element reported to this ProviderHandle.
    \return The statistics data handle associated with the given name, or nullptr if no such data exists.
    */
    AbstractData* getData( const char* name ) const;

    /**
    \return The parent node. Is nullptr for the root node.
    */
    ProviderHandle* getParent();

    /**
     Add the given ProviderHandle to the list of children, and set the parent pointer of the
     child to the new parent. Will fail if the child already has a parent. See 'adoptChild(*)'.
    */
    void addChild( ProviderHandle* child );

    /**
    \todo There may be several children with the same name, need a way to choose between them.

    \return A child with the given name, or nullptr if no such child exists.
    */
    ProviderHandle* getChild( const char* name );

    /**
    \return The child at the given index.
    */
    ProviderHandle* getChild( size_t index );

    /**
    \return The number of children.
    */
    size_t getNumChildren() const;

    /**
    \return The list of child nodes.
    */
    const ProviderHandleRefVector& getChildren() const;


    /**
    Returns true if this ProviderHandle was created as a handle for a Task.
    */
    bool isTaskNode() const;


    /**
    Returns true if this ProviderHandle was implicitly created. When a providers registers itself
    with a context but the context have not yet been registered, then an implicit ProviderHandle
    is created for the context.
    */
    bool isImplicit() const;


    /**
    Returns true if this node has a nullptr parent.
    */
    bool isRoot() const;

    /**
    Clears the data help by this ProviderHandle, using the type's default constructor.
    \param recursive   If true, then the same operations will be performed on the children as well.
    \param clearAccumulated If true, then the accumulated values are cleared as well.
    */
    void clear( bool recursive = false, bool clearAccumulated = true );


    /**
    Recursively print the data in this ProviderHandle and all its children.
    */
    void print( std::ostream& stream, bool includeTaskTimers, unsigned depth = 0 );

    /**
    \return The reported data elements for this provider.
    */
    const DataTable& getDataTable() const;

  protected:
    virtual ~ProviderHandle();

    friend class Statistics;

    /**
    Makes sure that a data handle with the given names exists.
    */
    template< typename T >
    Data<T>* getOrCreateData( const char* name );


    /**
    Move all children from the given ProviderHandle into 'this'.
    */
    void takeChildren( ProviderHandle* currentParent );

    /**
    Move a ProviderHandle from its current parent into 'this'.
    */
    void adoptChild( ProviderHandle* child );

    /**
    Tests if the given data handle corresponds to one of the special task timing values.
    */
    virtual bool isTaskTimer( AbstractData* data );

    /**
    Returns true if the ProviderHandle contains task timing values.
    */
    void setIsTaskNode( bool isTaskNode );

    /**
    Recursively updates the SUBTREE_IS_EMPTY flags. Returns true if subtree is
    empty. An ProviderHandle is considered empty if the only data recored in the
    subtree are task timer values.
    */
    bool updateEmptyFlags();

  protected:
    void *m_owner;
    const char* m_name;
    ProviderHandle *m_parent;
    Statistics* m_statistics;
    ProviderHandleRefVector m_children;
    DataTable m_dataTable; // All data elements reported to this handle.
    Int32 m_flags;

    // Enum listing the masks that may be used with the 'm_flags' member variable.
    enum FlagFields {
      IS_TASK_NODE = 0x1,
      SUBTREE_IS_EMPTY = 0x2
    };
  };


  /**
  An abstract handle for a statistics value. Actual data storage is managed by templated
  subclasses.
  */
  class AGXCORE_EXPORT Statistics::AbstractData : public Referenced
  {
  public:
    AbstractData( const char* name, const agxData::Format* format );

    const agxData::Format *getFormat() const;

    const char* getName() const;
    bool isTiming() const;

    /**
    Converts the current value to a human readable string using the "ascii" format for
    the type of the value.
    */
    void toString( String& result ) const;

    /**
    Reference-less method to be used with Lua scripts.
    */
    String toString() const;

    /**
    \return The data as an agxData::Value
   */
    virtual agxData::Value *toValue() const = 0;

    /**
    Converts the accumulated value to a human readable string using the "ascii" format
    for the type of the value.
    */
    void accumulatedToString( String& result ) const;

    /**
    Reference-less. Converts the accumulated value to a human readable string using the "ascii" format
    for the type of the value.
    */
    String accumulatedToString() const;

    /**
    Converts this abstract representation to a type specified data handle. Will return
    true if the given type is wrong.
    */
    template< typename T >
    Data<T>* asTypedData();

    /**
    Sets the current and accumulated values to zero, or the equivalent for complex types.
    Uses default constructor.
    */
    virtual void clear( bool clearAccumulated = true ) = 0;


    void setIsTiming( bool isTiming = true );

  protected:
    virtual ~AbstractData();

    /**
    Returns a raw pointer to the memory location where the current value is stored. Use with caution.
    */
    virtual const void* valuePtr() const = 0;
    /**
    Returns a raw pointer to the memory location where the current value is stored. Use with caution.
    */
    virtual const void* accumulatedPtr() const = 0;

  private:
    const char* m_name;
    const agxData::Format* m_format;
    bool m_isTiming;
  };

  /**
  Statistics data storage class. Records both the most recently reported values and an accumulation
  of all reportings.
  */
  template <typename T>
  class Statistics::Data : public AbstractData
  {
  public:
    Data( const char* name);

    virtual agxData::Value *toValue() const override;

    /**
    Returns the current value of this statistics data handle. In the absence of 'reportAdditionalTiming(*)'
    calls this is the value passed to the last call to 'report(*)'.
    */
    T& value();
    const T& value() const;

    /**
    Returns the accumulated value of this statistics data handle. This is the sum of all reported
    values since the last reset.
    */
    T& accumulatedValue();
    const T& accumulatedValue() const;


    agx::UInt getNumAccumulated() const;

    /*
    Sets the current value to 'value' and increments the accumulated value by the same amount.
    */
    void report( const T& value );

    /**
    Sets the current and accumulated values to zero, or the equivalent for complex types.
    Uses default constructor.
    */
    virtual void clear( bool clearAccumulated = true ) override;

  protected:
    virtual ~Data();

    virtual const void* valuePtr() const override;
    virtual const void* accumulatedPtr() const override;

  private:
    T m_value;
    T m_accumulatedValue;
    agx::UInt m_numAccumulated;
  };



  DOXYGEN_START_INTERNAL_BLOCK()

  /**
  A provider handle used by the Tasks to report profiling times.
  Not declared with the other nested classes since we want to be able to forward declare
  it in agx/Task.h.
  */
  class AGXCORE_EXPORT TaskTimingReportHandle : public Statistics::ProviderHandle
  {
  public:
    TaskTimingReportHandle( void* owner, Statistics* statistics );
//    void report( Real wallTime, Real computeCost, Real overhead, void* context );
    void report( Real wallTime, Real computeCost, Real overhead );
    void reportWallAndOverhead( Real wallTime, Real overhead, void* context );
    void reportWallAndOverhead( Real wallTime, Real overhead );
    void reportCostAndAdditionalOverhead( Real computeCost, Real additionalOverhead );

    Statistics::Data<Real>* getWallTime();
    Statistics::Data<Real>* getOverhead();
    Statistics::Data<Real>* getComputeCost();

  protected:
    virtual bool isTaskTimer( Statistics::AbstractData* data );
  private:
    Statistics::Data<Real> *m_computeCost;
    Statistics::Data<Real> *m_wallTime;
    Statistics::Data<Real> *m_overhead;
  };

  class StatisticsScopeTimer {
  public:
    StatisticsScopeTimer(void *instance, const char *name) : m_instance(instance), m_name(name)
    {
      m_timer.start();
    }

    ~StatisticsScopeTimer()
    {
      m_timer.stop();
      AGX_STATISTICS_REPORT_ADDITIONAL_TIMING(m_instance, m_name, (Real32)m_timer.getTime());
    }

  private:
    void *m_instance;
    const char *m_name;
    Timer m_timer;
  };


  /* Implementation*/

  template <typename T>
  Statistics::AbstractData* Statistics::report( void* instance, const char* name, const T& value )
  {
    if ( ! m_enabled )
      return nullptr;

    // The typesystem my register task timer providers. Hence the typesystem lock must be taken first
    // to avoid potential lock order inversion.
    //
    // The call to provider->getOrCreateData<T>( name ) might take the typesystem lock so
    // we must take the typesystem lock here before m_mutex
    ScopeLock<ReentrantMutex> typeSystemLock( getTypeSystemMutex() );


    ScopeLock<ReentrantMutex> lock( m_mutex );

    // Get the provider handle.
    DataProviderTable::iterator providerIt = m_providers.find( instance );
    if ( providerIt == m_providers.end() )
    {
      LOGGER_WARNING() << "Trying to report statistics from an instance that hasn't been registered as a data provider yet." << LOGGER_ENDL();
      return nullptr;
    }

    // Get the data handle. Make sure it has the expected type.
    ProviderHandle* provider = providerIt->second;
    Data<T>* data = provider->getOrCreateData<T>( name );
    if ( data == nullptr )
    {
      const agxData::Format* actualFormat = provider->getData( name )->getFormat();
      LOGGER_WARNING() << "Reporting data of type \'" <<
          agxData::getType<T>()->getName() << ":" << agxData::getFormat<T>()->getName() << "\'" <<
          " to data point \'" << name << ", which has already received data of type \'" <<
          actualFormat->getType()->getName() << ":" << actualFormat->getName() << "\'. " <<
          "Cannot mix data types, so newest data point is ignored." << LOGGER_ENDL();
      return nullptr;
    }

    // Update data.
    data->value() = value;
    data->accumulatedValue() += value;
    return data;
  }

  template< typename T >
  AGX_FORCE_INLINE Statistics::AbstractData* Statistics::reportTiming( void* instance, const char* name, const T& timing )
  {
    AbstractData* timingData = this->report( instance, name, (Real)timing );
    if ( timingData )
      timingData->setIsTiming( true );
    return timingData;
  }

  AGX_FORCE_INLINE Statistics::AbstractData* Statistics::reportTiming( void* instance, const char* name, const Timer& timer )
  {
    AbstractData* timingData = this->report( instance, name, (Real)timer.getTime() );
    if ( timingData )
      timingData->setIsTiming( true );
    return timingData;
  }

  template <typename T>
  AGX_FORCE_INLINE Statistics::AbstractData* Statistics::reportAdditionalTiming( void* instance, const char* name, const T& timing )
  {
    if ( !m_enabled )
      return nullptr;

    ScopeLock<ReentrantMutex> lock( m_mutex );

    // Get the data that should be incremented. Make sure it has the expected type.
    Data<T>* data = this->getData<T>( instance, name );
    if ( data == nullptr )
    {
      return reportTiming( instance, name, timing );
    }

    // Do the increment.
    data->value() += timing;
    data->accumulatedValue() += timing;

    return data;
  }

  AGX_FORCE_INLINE Statistics::AbstractData* Statistics::reportAdditionalTiming( void* instance, const char* name, const Timer& timer )
  {
    return this->reportAdditionalTiming( instance, name, (Real)timer.getTime() );
  }


#if 0
// This method was believed to be required, but apparently not yet.
  template <typename T>
  AGX_FORCE_INLINE Statistics::AbstractData* Statistics::report( void* /*instance*/, void* /*context*/, const char* /*name*/, const T& /*value*/ )
  {
    if ( ! m_enabled )
      return;
    agxAbort1("Context altering reporting not yet implemented.");
  }
#endif


  AGX_FORCE_INLINE Statistics::AbstractData* Statistics::getData(void *instance, const char* name)
  {
    DataProviderTable::iterator it = m_providers.find( instance );
    return it != m_providers.end() ? it->second->getData( name ) : nullptr;
  }


  template <typename T>
  AGX_FORCE_INLINE Statistics::Data<T>* Statistics::getData(void* instance, const char* name)
  {
    AbstractData *data = this->getData( instance, name );
    if ( data == nullptr || dynamic_cast<Data<T> *>(data) == nullptr )
      return nullptr;
    return static_cast<Data<T> *>( data );
  }


  AGX_FORCE_INLINE Statistics::ProviderHandle* Statistics::getProviderHandle( void* instance )
  {
    DataProviderTable::iterator it = m_providers.find( instance );
    return it != m_providers.end() ? it->second  :  nullptr;
  }

  AGX_FORCE_INLINE Statistics::ProviderHandle* Statistics::getRootProviderHandle()
  {
    return m_rootProvider;
  }

  AGX_FORCE_INLINE const Statistics::ProviderHandleRefVector& Statistics::getProviderHandles( const char* name )
  {
    DataProviderNameTable::const_iterator it = m_providerNames.find( name );
    if ( it == m_providerNames.end() )
      LOGGER_ERROR() << "Statistics: Was asked to return a reference to the list of providers with the name \'" << name << "\'" <<
      ", but there are no such providers and thus no vector to return." << LOGGER_ENDL();

    return it->second;
  }



  //---------------------------------------------------------------

  AGX_FORCE_INLINE Statistics::ProviderHandle::ProviderHandle( void* owner, Statistics* statistics )
  : m_owner(owner), m_name(nullptr), m_parent(nullptr), m_statistics(statistics), m_flags(0)
  {
    this->setIsTaskNode( false );
  }


  AGX_FORCE_INLINE Statistics::ProviderHandle::~ProviderHandle()
  {
  }


  AGX_FORCE_INLINE const char* Statistics::ProviderHandle::getName()
  {
    return this->isImplicit() ? "<Implicit node>"  : m_name;
  }

  AGX_FORCE_INLINE void* Statistics::ProviderHandle::getOwner()
  {
    return m_owner;
  }

  AGX_FORCE_INLINE const void* Statistics::ProviderHandle::getOwner() const
  {
    return m_owner;
  }

  AGX_FORCE_INLINE Statistics::ProviderHandle* Statistics::ProviderHandle::getParent()
  {
    return m_parent;
  }

  template< typename T >
  AGX_FORCE_INLINE T* Statistics::ProviderHandle::getOwner()
  {
    return dynamic_cast<T*>( this->getOwner() );
  }

  template< typename T >
  AGX_FORCE_INLINE const T* Statistics::ProviderHandle::getOwner() const
  {
    return dynamic_cast<const T*>( this->getOwner() );
  }

  AGX_FORCE_INLINE Statistics::ProviderHandle* Statistics::ProviderHandle::getChild( const char* name )
  {
    // Linear search for the child with the wanted name.
    for ( size_t i = 0 ; i < m_children.size() ; ++i )
      if ( !m_children[i]->isImplicit() && ::strcmp(name, m_children[i]->m_name) == 0 )
        return m_children[i];

    return nullptr;
  }

  AGX_FORCE_INLINE Statistics::ProviderHandle* Statistics::ProviderHandle::getChild( size_t index ) { return m_children[index]; }
  AGX_FORCE_INLINE const Statistics::ProviderHandleRefVector& Statistics::ProviderHandle::getChildren() const { return m_children; }
  AGX_FORCE_INLINE size_t Statistics::ProviderHandle::getNumChildren() const { return m_children.size(); }

  AGX_FORCE_INLINE bool Statistics::ProviderHandle::isTaskNode() const { return m_flags & IS_TASK_NODE; }
  AGX_FORCE_INLINE bool Statistics::ProviderHandle::isImplicit() const { return m_name == nullptr; }
  AGX_FORCE_INLINE const Statistics::DataTable& Statistics::ProviderHandle::getDataTable() const { return m_dataTable; }

  AGX_FORCE_INLINE bool Statistics::ProviderHandle::isRoot() const
  {
    if ( !m_parent &&  this != m_statistics->getRootProviderHandle() )
      LOGGER_ERROR() << "Internal Error: Found a root that isn't the same is the Statistics root node. This should not happen?" << LOGGER_ENDL();

    return !m_parent;
  }



  template< typename T >
  Statistics::Data<T>* Statistics::ProviderHandle::getOrCreateData( const char* name )
  {
    DataTable::iterator it = m_dataTable.find( name );
    if ( it != m_dataTable.end() )
    {
      AbstractData* data = it->second;
      return dynamic_cast< Data<T>* >( data );
    }
    else
    {
      const char* internalName = Statistics::copyString( name );
      Data<T>* newData = new Data<T>( internalName ); // Pass ownership of 'internalName' to 'newData'.
      m_dataTable.insert( newData->getName(), newData );
      return newData;
    }
  }


  AGX_FORCE_INLINE bool Statistics::ProviderHandle::isTaskTimer( AbstractData* )
  {
    return false;
  }


  AGX_FORCE_INLINE void Statistics::ProviderHandle::setIsTaskNode( bool isTaskNode )
  {
    if ( isTaskNode )
      m_flags |= IS_TASK_NODE;
    else
      m_flags &= ~IS_TASK_NODE;
  }

  //---------------------------------------------------------------

  AGX_FORCE_INLINE TaskTimingReportHandle::TaskTimingReportHandle(void* owner, Statistics* statistics) :
      Statistics::ProviderHandle( owner, statistics )
  {
    this->setIsTaskNode( true );
    m_computeCost = this->getOrCreateData<Real>( "ComputeCost" );
    m_wallTime = this->getOrCreateData<Real>( "WallTime" );
    m_overhead = this->getOrCreateData<Real>( "Overhead" );
  }

#if 0
  AGX_FORCE_INLINE void TaskTimingReportHandle::report( Real wallTime, Real computeCost, Real overhead, void* context )
  {
    if ( !m_statistics->getEnable() )
      return;

    if ( context != m_parent->getOwner() )
      m_statistics->setContext( m_owner, context );

    this->report( wallTime, computeCost, overhead );
  }
#endif

  AGX_FORCE_INLINE void TaskTimingReportHandle::report( Real wallTime, Real computeCost, Real overhead )
  {
    if ( !m_statistics || !m_statistics->getEnable() )
      return;

    m_wallTime->report( wallTime );
    m_computeCost->report( computeCost );
    m_overhead->report( overhead );
  }

  AGX_FORCE_INLINE void TaskTimingReportHandle::reportWallAndOverhead( Real wallTime, Real overhead, void* context )
  {
    if ( !m_statistics || !m_statistics->getEnable() )
      return;

    if ( context != m_parent->getOwner() )
      m_statistics->setContext( m_owner, context );

    this->reportWallAndOverhead( wallTime, overhead );
  }

  AGX_FORCE_INLINE void TaskTimingReportHandle::reportWallAndOverhead( Real wallTime, Real overhead )
  {
    if ( !m_statistics || !m_statistics->getEnable() )
      return;

    m_wallTime->report( wallTime );
    m_overhead->report( overhead );
  }

  AGX_FORCE_INLINE void TaskTimingReportHandle::reportCostAndAdditionalOverhead( Real computeCost, Real additionalOverhead )
  {
    if ( !m_statistics || !m_statistics->getEnable() )
      return;

    m_computeCost->report( computeCost );
    m_overhead->report( additionalOverhead );
  }


  AGX_FORCE_INLINE bool TaskTimingReportHandle::isTaskTimer( Statistics::AbstractData* data )
  {
    return data == m_computeCost  ||  data == m_wallTime  ||  data == m_overhead;
  }

  //---------------------------------------------------------------

  AGX_FORCE_INLINE Statistics::AbstractData::AbstractData( const char* name, const agxData::Format* format )
  : m_name(name), m_format(format), m_isTiming( false )
  {
  }

  AGX_FORCE_INLINE Statistics::AbstractData::~AbstractData()
  {
    delete [] m_name;
    m_name = nullptr;
  }


  AGX_FORCE_INLINE const agxData::Format* Statistics::AbstractData::getFormat() const
  {
    return m_format;
  }

  AGX_FORCE_INLINE const char* Statistics::AbstractData::getName() const
  {
    return m_name;
  }

  AGX_FORCE_INLINE bool Statistics::AbstractData::isTiming() const
  {
    return m_isTiming;
  }

  AGX_FORCE_INLINE void Statistics::AbstractData::toString( String& result ) const
  {
    agxAssert(m_format);
    if (m_format) {
      if ( m_format->getType()->getFormat("ascii") == nullptr )
      {
        result =  String("Can not print value of type \'") + m_format->getType()->getName().c_str() + String("\', no ascii format available");
        return;
      }
      agxData::transform(&result, m_format->getType()->getFormat("ascii"), this->valuePtr(), m_format, 1);
    }
  }

  AGX_FORCE_INLINE String Statistics::AbstractData::toString() const
  {
    String result;
    this->toString( result );
    return result;
  }

  AGX_FORCE_INLINE String Statistics::AbstractData::accumulatedToString() const
  {
    String result;
    this->accumulatedToString(result);
    return result;
  }


  AGX_FORCE_INLINE void Statistics::AbstractData::accumulatedToString( String& result ) const
  {
    agxAssert(m_format);
    if (m_format) {
      agxAssertN(m_format->getType()->getFormat("ascii"), "Can not print value of type \'%s\', no ascii format available...", m_format->getType()->getName().c_str());
      if ( m_format->getType()->getFormat("ascii") == nullptr )
      {
        LOGGER_WARNING() << "Can not print value of type \'" << m_format->getType()->getName().c_str() << "\' to String, no ascii format available";
        return;
      }
      agxData::transform(&result, m_format->getType()->getFormat("ascii"), this->accumulatedPtr(), m_format, 1);
    }
  }

  template< typename T >
  AGX_FORCE_INLINE Statistics::Data<T>* Statistics::AbstractData::asTypedData()
  {
    return dynamic_cast< Data<T>* >( this );
  }

  AGX_FORCE_INLINE void Statistics::AbstractData::setIsTiming( bool isTiming )
  {
    m_isTiming = isTiming;
  }

  //---------------------------------------------------------------

  template <typename T>
  AGX_FORCE_INLINE Statistics::Data<T>::Data( const char* name)
  : AbstractData( name, agxData::getFormat<T>() ), m_value(T()), m_accumulatedValue(T()), m_numAccumulated(0)
  {
  }

  template< typename T >
  AGX_FORCE_INLINE Statistics::Data<T>::~Data()
  {
  }

  template< typename T >
  agxData::Value *Statistics::Data<T>::toValue() const
  {
    agxData::Value *value = new agxData::Value(this->getName(), agxData::getFormat<T>());
    value->set<T>(m_value);
    return value;
  }


  template <typename T>
  AGX_FORCE_INLINE T& Statistics::Data<T>::value() { return m_value; }

  template <typename T>
  AGX_FORCE_INLINE const T& Statistics::Data<T>::value() const { return m_value; }

  template <typename T>
  AGX_FORCE_INLINE T& Statistics::Data<T>::accumulatedValue() { return m_accumulatedValue; }

  template <typename T>
  AGX_FORCE_INLINE const T& Statistics::Data<T>::accumulatedValue() const { return m_accumulatedValue; }

  template <typename T>
  AGX_FORCE_INLINE agx::UInt Statistics::Data<T>::getNumAccumulated() const { return m_numAccumulated; }

  template <typename T>
  AGX_FORCE_INLINE void Statistics::Data<T>::report( const T& value )
  {
    m_value = value;
    m_accumulatedValue += value;
    ++m_numAccumulated;
  }

  template <typename T>
  AGX_FORCE_INLINE void Statistics::Data<T>::clear( bool clearAccumulated )
  {
    m_value = T();
    if ( clearAccumulated ) {
      m_accumulatedValue = T();
      m_numAccumulated = 0;
    }
  }

  template <typename T>
  const void* Statistics::Data<T>::valuePtr() const
  {
    return static_cast< const void* > ( &m_value );
  }

  template <typename T>
  const void* Statistics::Data<T>::accumulatedPtr() const
  {
    return static_cast< const void* > ( &m_accumulatedValue );
  }

  //---------------------------------------------------------------


  DOXYGEN_END_INTERNAL_BLOCK()

}


#endif /* AGX_STATISTICS_H */
