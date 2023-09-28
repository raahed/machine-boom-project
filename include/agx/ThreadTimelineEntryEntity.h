/*
Copyright 2007-2023. Algoryx Simulation AB.

All AGX source code, intellectual property, documentation, sample code,
tutorials, scene files and technical white papers, are copyrighted, proprietary
and confidential material of Algoryx Simulation AB. You may not download, read,
store, distribute, publish, copy or otherwise disseminate, use or expose this
material unless having a written signed agreement with Algoryx Simulation AB, or
having been advised so by Algoryx Simulation AB for a time limited evaluation,
or having purchased a valid commercial license from Algoryx Simulation AB.

Algoryx Simulation AB disclaims all responsibilities for loss or damage caused
from using this software, unless otherwise stated in written agreements with
Algoryx Simulation AB.
*/

//////////////////////////////////////////////////
// AUTOMATICALLY GENERATED ENTITY, DO NOT EDIT! //
//////////////////////////////////////////////////

#ifndef GENERATED_AGX_THREADTIMELINEENTRY_H_PLUGIN
#define GENERATED_AGX_THREADTIMELINEENTRY_H_PLUGIN

#define AGX_ENTITY_WRAPPER 1


#ifdef _MSC_VER
# pragma warning(push)
// warning C4505: 'agxData::VectorAttributeT<T>::print' : unreferenced local function has been removed
# pragma warning( disable : 4505 )
//  warning C4251:  'X' : class 'Y' needs to have dll-interface to be used by clients of class 'Z'
# pragma warning( disable : 4251 )
//  warning C4355: 'this' : used in base member initializer list
# pragma warning( disable : 4355 )
//  marked as __forceinline not inlined
# pragma warning( disable: 4714 )
#endif

#include <agxData/EntityModel.h>
#include <agxData/EntityStorage.h>
#include <agxData/EntityRef.h>
#include <agxData/EntityPtr.h>
#include <agxData/EntityInstance.h>
#include <agx/Integer.h>
#include <agx/Real.h>
#include <agx/macros.h>
#include <agx/Integer.h>
#include <agx/Real.h>
#include <agx/String.h>


namespace agx
{

  class ThreadTimelineEntryModel;
  class ThreadTimelineEntryData;
  class ThreadTimelineEntryPtr;
  class ThreadTimelineEntryInstance;
  class ThreadTimelineEntrySemantics;


  AGX_DECLARE_POINTER_TYPES(ThreadTimelineEntryModel);

  /** 
  Abstract description of the data attributes for the ThreadTimelineEntry entity.
  */ 
  class AGXCORE_EXPORT ThreadTimelineEntryModel : public agxData::EntityModel
  {
  public:
    typedef ThreadTimelineEntryPtr PtrT;

    ThreadTimelineEntryModel(const agx::String& name = "ThreadTimelineEntry");

    /// \return The entity model singleton.
    static ThreadTimelineEntryModel* instance();

    /// Create and return a pointer to a new instance in the default storage for this entity model.
    static ThreadTimelineEntryPtr createInstance();

    /// \return The default storage for this entity model.
    static agxData::EntityStorage* defaultStorage();

    /// This is part of internal cleanup and should not be called by users
    virtual void shutdownCleanup() override;



    /* Attributes */
    static agxData::ScalarAttributeT< agx::UInt32 >* taskIdAttribute;
    static agxData::ScalarAttributeT< agx::Real64 >* startTimeGlobalAttribute;
    static agxData::ScalarAttributeT< agx::Real64 >* endTimeGlobalAttribute;
    static agxData::ScalarAttributeT< agx::Real64 >* startTimeFrameAttribute;
    static agxData::ScalarAttributeT< agx::Real64 >* endTimeFrameAttribute;
    static agxData::ScalarAttributeT< agx::String >* descriptionAttribute;
    static agxData::ScalarAttributeT< agx::UInt32 >* costEstimateAttribute;
    static agxData::ScalarAttributeT< agx::UInt32 >* poolSizeAttribute;
    static agxData::ScalarAttributeT< agx::UInt8 >* jobTypeAttribute;
    static agxData::ScalarAttributeT< agx::UInt64 >* jobPtrAttribute;
    static agxData::ScalarAttributeT< agx::String >* extraDataTitleAttribute;
    static agxData::ScalarAttributeT< agx::Real64 >* extraDataAttribute;

  protected:
    virtual ~ThreadTimelineEntryModel();
    virtual agxData::EntityData* createData(agxData::EntityStorage* storage) override;
    virtual void configure(agx::TiXmlElement* eEntity) override;
    virtual void initAttributeAccessors() override;
    virtual void construct(agxData::EntityPtr instance) override;
    void construct(agx::ThreadTimelineEntryPtr threadTimelineEntry);
  };


  DOXYGEN_START_INTERNAL_BLOCK()
  #ifndef AGX_THREADTIMELINEENTRY_DATA_SET_OVERRIDE
  #define AGX_THREADTIMELINEENTRY_DATA_SET
  class AGXCORE_EXPORT ThreadTimelineEntryData : public agxData::EntityData
  {
  public:
    ThreadTimelineEntryInstance operator[] (size_t index);

  public:
    agxData::Array< ThreadTimelineEntryPtr >& instance;
    agxData::Array< agx::UInt32 > taskId;
    agxData::Array< agx::Real64 > startTimeGlobal;
    agxData::Array< agx::Real64 > endTimeGlobal;
    agxData::Array< agx::Real64 > startTimeFrame;
    agxData::Array< agx::Real64 > endTimeFrame;
    agxData::Array< agx::String > description;
    agxData::Array< agx::UInt32 > costEstimate;
    agxData::Array< agx::UInt32 > poolSize;
    agxData::Array< agx::UInt8 > jobType;
    agxData::Array< agx::UInt64 > jobPtr;
    agxData::Array< agx::String > extraDataTitle;
    agxData::Array< agx::Real64 > extraData;

  public:
    typedef agx::UInt32 taskIdType;
    typedef agx::Real64 startTimeGlobalType;
    typedef agx::Real64 endTimeGlobalType;
    typedef agx::Real64 startTimeFrameType;
    typedef agx::Real64 endTimeFrameType;
    typedef agx::String descriptionType;
    typedef agx::UInt32 costEstimateType;
    typedef agx::UInt32 poolSizeType;
    typedef agx::UInt8 jobTypeType;
    typedef agx::UInt64 jobPtrType;
    typedef agx::String extraDataTitleType;
    typedef agx::Real64 extraDataType;

  public:
    ThreadTimelineEntryData(agxData::EntityStorage* storage);
    ThreadTimelineEntryData();

  protected:
    virtual ~ThreadTimelineEntryData() {}
    virtual void setNumElements(agx::Index numElements) override;

  private:
    ThreadTimelineEntryData& operator= (const ThreadTimelineEntryData&) { return *this; }

  };
  #endif
  DOXYGEN_END_INTERNAL_BLOCK()


  DOXYGEN_START_INTERNAL_BLOCK()
  class AGXCORE_EXPORT ThreadTimelineEntrySemantics : protected agxData::EntityPtr
  {
  public:

    // Automatic getters
    agx::UInt32 const& getTaskId() const;
    agx::Real64 const& getStartTimeGlobal() const;
    agx::Real64 const& getEndTimeGlobal() const;
    agx::Real64 const& getStartTimeFrame() const;
    agx::Real64 const& getEndTimeFrame() const;
    agx::String const& getDescription() const;
    agx::UInt32 const& getCostEstimate() const;
    agx::UInt32 const& getPoolSize() const;
    agx::UInt8 const& getJobType() const;
    agx::UInt64 const& getJobPtr() const;
    agx::String const& getExtraDataTitle() const;
    agx::Real64 const& getExtraData() const;

    // Semantics defined by explicit kernels

    // Automatic setters
    void setTaskId(agx::UInt32 const& value);
    void setStartTimeGlobal(agx::Real64 const& value);
    void setEndTimeGlobal(agx::Real64 const& value);
    void setStartTimeFrame(agx::Real64 const& value);
    void setEndTimeFrame(agx::Real64 const& value);
    void setDescription(agx::String const& value);
    void setCostEstimate(agx::UInt32 const& value);
    void setPoolSize(agx::UInt32 const& value);
    void setJobType(agx::UInt8 const& value);
    void setJobPtr(agx::UInt64 const& value);
    void setExtraDataTitle(agx::String const& value);
    void setExtraData(agx::Real64 const& value);


  protected:
    friend class ThreadTimelineEntryPtr;
    friend class ThreadTimelineEntryInstance;
    ThreadTimelineEntrySemantics();
  };
  DOXYGEN_END_INTERNAL_BLOCK()


  /**
  Pointer to a entity instance of type ThreadTimelineEntry
  */
  class CALLABLE ThreadTimelineEntryPtr : public agxData::EntityPtr
  {
  public:
    typedef ThreadTimelineEntryModel ModelType;
    typedef ThreadTimelineEntryData DataType;
    typedef ThreadTimelineEntryInstance InstanceType;

  public:
    AGXCORE_EXPORT ThreadTimelineEntryPtr();
    AGXCORE_EXPORT ThreadTimelineEntryPtr(agxData::EntityStorage* storage, agx::Index id);
    AGXCORE_EXPORT ThreadTimelineEntryPtr(const agxData::EntityPtr& ptr);
    AGXCORE_EXPORT ThreadTimelineEntryPtr(const agxData::EntityInstance& instance);
    AGXCORE_EXPORT ThreadTimelineEntryPtr& operator= (const agxData::EntityPtr& ptr);
    AGXCORE_EXPORT ThreadTimelineEntryPtr& operator= (const agxData::EntityInstance& instance);
    AGXCORE_EXPORT ThreadTimelineEntryInstance instance();
    AGXCORE_EXPORT const ThreadTimelineEntryInstance instance() const;

    AGXCORE_EXPORT ThreadTimelineEntrySemantics* operator->();
    AGXCORE_EXPORT const ThreadTimelineEntrySemantics* operator->() const;

    ThreadTimelineEntryData* getData();
    const ThreadTimelineEntryData* getData() const;


    /// \return reference to the taskId attribute
    AGXCORE_EXPORT agx::UInt32& taskId();
    /// \return const reference to the taskId attribute
    AGXCORE_EXPORT agx::UInt32 const& taskId() const;

    /// \return reference to the startTimeGlobal attribute
    AGXCORE_EXPORT agx::Real64& startTimeGlobal();
    /// \return const reference to the startTimeGlobal attribute
    AGXCORE_EXPORT agx::Real64 const& startTimeGlobal() const;

    /// \return reference to the endTimeGlobal attribute
    AGXCORE_EXPORT agx::Real64& endTimeGlobal();
    /// \return const reference to the endTimeGlobal attribute
    AGXCORE_EXPORT agx::Real64 const& endTimeGlobal() const;

    /// \return reference to the startTimeFrame attribute
    AGXCORE_EXPORT agx::Real64& startTimeFrame();
    /// \return const reference to the startTimeFrame attribute
    AGXCORE_EXPORT agx::Real64 const& startTimeFrame() const;

    /// \return reference to the endTimeFrame attribute
    AGXCORE_EXPORT agx::Real64& endTimeFrame();
    /// \return const reference to the endTimeFrame attribute
    AGXCORE_EXPORT agx::Real64 const& endTimeFrame() const;

    /// \return reference to the description attribute
    AGXCORE_EXPORT agx::String& description();
    /// \return const reference to the description attribute
    AGXCORE_EXPORT agx::String const& description() const;

    /// \return reference to the costEstimate attribute
    AGXCORE_EXPORT agx::UInt32& costEstimate();
    /// \return const reference to the costEstimate attribute
    AGXCORE_EXPORT agx::UInt32 const& costEstimate() const;

    /// \return reference to the poolSize attribute
    AGXCORE_EXPORT agx::UInt32& poolSize();
    /// \return const reference to the poolSize attribute
    AGXCORE_EXPORT agx::UInt32 const& poolSize() const;

    /// \return reference to the jobType attribute
    AGXCORE_EXPORT agx::UInt8& jobType();
    /// \return const reference to the jobType attribute
    AGXCORE_EXPORT agx::UInt8 const& jobType() const;

    /// \return reference to the jobPtr attribute
    AGXCORE_EXPORT agx::UInt64& jobPtr();
    /// \return const reference to the jobPtr attribute
    AGXCORE_EXPORT agx::UInt64 const& jobPtr() const;

    /// \return reference to the extraDataTitle attribute
    AGXCORE_EXPORT agx::String& extraDataTitle();
    /// \return const reference to the extraDataTitle attribute
    AGXCORE_EXPORT agx::String const& extraDataTitle() const;

    /// \return reference to the extraData attribute
    AGXCORE_EXPORT agx::Real64& extraData();
    /// \return const reference to the extraData attribute
    AGXCORE_EXPORT agx::Real64 const& extraData() const;

    DOXYGEN_START_INTERNAL_BLOCK()
    // Only used by tolua!
    #ifdef AGX_GENERATED_TOLUA_WRAPPER
    inline agx::UInt32 const& getTaskId() const { return (*this)->getTaskId(); }
    inline agx::Real64 const& getStartTimeGlobal() const { return (*this)->getStartTimeGlobal(); }
    inline agx::Real64 const& getEndTimeGlobal() const { return (*this)->getEndTimeGlobal(); }
    inline agx::Real64 const& getStartTimeFrame() const { return (*this)->getStartTimeFrame(); }
    inline agx::Real64 const& getEndTimeFrame() const { return (*this)->getEndTimeFrame(); }
    inline agx::String const& getDescription() const { return (*this)->getDescription(); }
    inline agx::UInt32 const& getCostEstimate() const { return (*this)->getCostEstimate(); }
    inline agx::UInt32 const& getPoolSize() const { return (*this)->getPoolSize(); }
    inline agx::UInt8 const& getJobType() const { return (*this)->getJobType(); }
    inline agx::UInt64 const& getJobPtr() const { return (*this)->getJobPtr(); }
    inline agx::String const& getExtraDataTitle() const { return (*this)->getExtraDataTitle(); }
    inline agx::Real64 const& getExtraData() const { return (*this)->getExtraData(); }


    inline void setTaskId(agx::UInt32 const& value) { (*this)->setTaskId(value); }
    inline void setStartTimeGlobal(agx::Real64 const& value) { (*this)->setStartTimeGlobal(value); }
    inline void setEndTimeGlobal(agx::Real64 const& value) { (*this)->setEndTimeGlobal(value); }
    inline void setStartTimeFrame(agx::Real64 const& value) { (*this)->setStartTimeFrame(value); }
    inline void setEndTimeFrame(agx::Real64 const& value) { (*this)->setEndTimeFrame(value); }
    inline void setDescription(agx::String const& value) { (*this)->setDescription(value); }
    inline void setCostEstimate(agx::UInt32 const& value) { (*this)->setCostEstimate(value); }
    inline void setPoolSize(agx::UInt32 const& value) { (*this)->setPoolSize(value); }
    inline void setJobType(agx::UInt8 const& value) { (*this)->setJobType(value); }
    inline void setJobPtr(agx::UInt64 const& value) { (*this)->setJobPtr(value); }
    inline void setExtraDataTitle(agx::String const& value) { (*this)->setExtraDataTitle(value); }
    inline void setExtraData(agx::Real64 const& value) { (*this)->setExtraData(value); }
    #endif
    DOXYGEN_END_INTERNAL_BLOCK()

  };


  DOXYGEN_START_INTERNAL_BLOCK()
  class AGXCORE_EXPORT ThreadTimelineEntryInstance : public agxData::EntityInstance
  {
  public:
    ThreadTimelineEntryInstance();
    ThreadTimelineEntryInstance(ThreadTimelineEntryData* data, agx::Index index);
    ThreadTimelineEntryInstance(agxData::EntityStorage *storage, agx::Index index);
    ThreadTimelineEntryInstance(const agxData::EntityInstance& other);
    ThreadTimelineEntryInstance(const agxData::EntityPtr& ptr);

    ThreadTimelineEntryData* getData();
    const ThreadTimelineEntryData* getData() const;

  public:
    /// \return reference to the taskId attribute
    agx::UInt32& taskId();
    /// \return const reference to the taskId attribute
    agx::UInt32 const& taskId() const;

    /// \return reference to the startTimeGlobal attribute
    agx::Real64& startTimeGlobal();
    /// \return const reference to the startTimeGlobal attribute
    agx::Real64 const& startTimeGlobal() const;

    /// \return reference to the endTimeGlobal attribute
    agx::Real64& endTimeGlobal();
    /// \return const reference to the endTimeGlobal attribute
    agx::Real64 const& endTimeGlobal() const;

    /// \return reference to the startTimeFrame attribute
    agx::Real64& startTimeFrame();
    /// \return const reference to the startTimeFrame attribute
    agx::Real64 const& startTimeFrame() const;

    /// \return reference to the endTimeFrame attribute
    agx::Real64& endTimeFrame();
    /// \return const reference to the endTimeFrame attribute
    agx::Real64 const& endTimeFrame() const;

    /// \return reference to the description attribute
    agx::String& description();
    /// \return const reference to the description attribute
    agx::String const& description() const;

    /// \return reference to the costEstimate attribute
    agx::UInt32& costEstimate();
    /// \return const reference to the costEstimate attribute
    agx::UInt32 const& costEstimate() const;

    /// \return reference to the poolSize attribute
    agx::UInt32& poolSize();
    /// \return const reference to the poolSize attribute
    agx::UInt32 const& poolSize() const;

    /// \return reference to the jobType attribute
    agx::UInt8& jobType();
    /// \return const reference to the jobType attribute
    agx::UInt8 const& jobType() const;

    /// \return reference to the jobPtr attribute
    agx::UInt64& jobPtr();
    /// \return const reference to the jobPtr attribute
    agx::UInt64 const& jobPtr() const;

    /// \return reference to the extraDataTitle attribute
    agx::String& extraDataTitle();
    /// \return const reference to the extraDataTitle attribute
    agx::String const& extraDataTitle() const;

    /// \return reference to the extraData attribute
    agx::Real64& extraData();
    /// \return const reference to the extraData attribute
    agx::Real64 const& extraData() const;

  };
  DOXYGEN_END_INTERNAL_BLOCK()



  typedef agx::VectorPOD<ThreadTimelineEntryPtr> ThreadTimelineEntryPtrVector;
  typedef agxData::Array<ThreadTimelineEntryPtr> ThreadTimelineEntryPtrArray;



  DOXYGEN_START_INTERNAL_BLOCK()
  /* Implementation */
  //-----------------------------------------------------------------------------------------------------
  //-----------------------------------------------------------------------------------------------------
  inline ThreadTimelineEntryInstance agx::ThreadTimelineEntryData::operator[] (size_t index) { return ThreadTimelineEntryInstance(this, (agx::Index)index); }
  //-----------------------------------------------------------------------------------------------------
  AGX_FORCE_INLINE ThreadTimelineEntryPtr::ThreadTimelineEntryPtr() {}
  AGX_FORCE_INLINE ThreadTimelineEntryPtr::ThreadTimelineEntryPtr(agxData::EntityStorage* storage, agx::Index id) : agxData::EntityPtr(storage, id) {}
  AGX_FORCE_INLINE ThreadTimelineEntryPtr::ThreadTimelineEntryPtr(const agxData::EntityPtr& ptr) : agxData::EntityPtr(ptr)
  {
    agxAssertN(!ptr || ptr.isInstanceOf(ThreadTimelineEntryModel::instance()),
      "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
      EntityPtr::getModel()->fullPath().c_str(), ThreadTimelineEntryModel::instance()->fullPath().c_str());
  }

  AGX_FORCE_INLINE ThreadTimelineEntryPtr::ThreadTimelineEntryPtr(const agxData::EntityInstance& instance) : agxData::EntityPtr(instance)
  {
    agxAssertN(!instance || instance.isInstanceOf(ThreadTimelineEntryModel::instance()),
      "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
      EntityPtr::getModel()->fullPath().c_str(), ThreadTimelineEntryModel::instance()->fullPath().c_str());
  }

  AGX_FORCE_INLINE ThreadTimelineEntryPtr& ThreadTimelineEntryPtr::operator= (const agxData::EntityPtr& ptr)
  {
    agxData::EntityPtr::operator= (ptr);
    agxAssertN(!ptr || ptr.isInstanceOf(ThreadTimelineEntryModel::instance()),
      "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
      EntityPtr::getModel()->fullPath().c_str(), ThreadTimelineEntryModel::instance()->fullPath().c_str());
    return *this;
  }

  AGX_FORCE_INLINE ThreadTimelineEntryPtr& ThreadTimelineEntryPtr::operator= (const agxData::EntityInstance& instance)
  {
    agxData::EntityPtr::operator= (instance);
    agxAssertN(!instance || instance.isInstanceOf(ThreadTimelineEntryModel::instance()),
      "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
      EntityPtr::getModel()->fullPath().c_str(), ThreadTimelineEntryModel::instance()->fullPath().c_str());
    return *this;
  }

  inline ThreadTimelineEntryInstance ThreadTimelineEntryPtr::instance() { return agxData::EntityPtr::instance(); }
  inline const ThreadTimelineEntryInstance ThreadTimelineEntryPtr::instance() const { return agxData::EntityPtr::instance(); }
  AGX_FORCE_INLINE ThreadTimelineEntrySemantics* ThreadTimelineEntryPtr::operator->() { return (ThreadTimelineEntrySemantics* )this; }
  AGX_FORCE_INLINE const ThreadTimelineEntrySemantics* ThreadTimelineEntryPtr::operator->() const { return (const ThreadTimelineEntrySemantics* )this; }
  AGX_FORCE_INLINE ThreadTimelineEntryData* ThreadTimelineEntryPtr::getData() { return static_cast<ThreadTimelineEntryData* >(agxData::EntityPtr::getData()); }
  AGX_FORCE_INLINE const ThreadTimelineEntryData* ThreadTimelineEntryPtr::getData() const { return static_cast<const ThreadTimelineEntryData* >(agxData::EntityPtr::getData()); }

  AGX_FORCE_INLINE agx::UInt32& ThreadTimelineEntryPtr::taskId() { verifyIndex(); return getData()->taskId[calculateIndex()]; }
  AGX_FORCE_INLINE agx::UInt32 const& ThreadTimelineEntryPtr::taskId() const { verifyIndex(); return getData()->taskId[calculateIndex()]; }

  AGX_FORCE_INLINE agx::Real64& ThreadTimelineEntryPtr::startTimeGlobal() { verifyIndex(); return getData()->startTimeGlobal[calculateIndex()]; }
  AGX_FORCE_INLINE agx::Real64 const& ThreadTimelineEntryPtr::startTimeGlobal() const { verifyIndex(); return getData()->startTimeGlobal[calculateIndex()]; }

  AGX_FORCE_INLINE agx::Real64& ThreadTimelineEntryPtr::endTimeGlobal() { verifyIndex(); return getData()->endTimeGlobal[calculateIndex()]; }
  AGX_FORCE_INLINE agx::Real64 const& ThreadTimelineEntryPtr::endTimeGlobal() const { verifyIndex(); return getData()->endTimeGlobal[calculateIndex()]; }

  AGX_FORCE_INLINE agx::Real64& ThreadTimelineEntryPtr::startTimeFrame() { verifyIndex(); return getData()->startTimeFrame[calculateIndex()]; }
  AGX_FORCE_INLINE agx::Real64 const& ThreadTimelineEntryPtr::startTimeFrame() const { verifyIndex(); return getData()->startTimeFrame[calculateIndex()]; }

  AGX_FORCE_INLINE agx::Real64& ThreadTimelineEntryPtr::endTimeFrame() { verifyIndex(); return getData()->endTimeFrame[calculateIndex()]; }
  AGX_FORCE_INLINE agx::Real64 const& ThreadTimelineEntryPtr::endTimeFrame() const { verifyIndex(); return getData()->endTimeFrame[calculateIndex()]; }

  AGX_FORCE_INLINE agx::String& ThreadTimelineEntryPtr::description() { verifyIndex(); return getData()->description[calculateIndex()]; }
  AGX_FORCE_INLINE agx::String const& ThreadTimelineEntryPtr::description() const { verifyIndex(); return getData()->description[calculateIndex()]; }

  AGX_FORCE_INLINE agx::UInt32& ThreadTimelineEntryPtr::costEstimate() { verifyIndex(); return getData()->costEstimate[calculateIndex()]; }
  AGX_FORCE_INLINE agx::UInt32 const& ThreadTimelineEntryPtr::costEstimate() const { verifyIndex(); return getData()->costEstimate[calculateIndex()]; }

  AGX_FORCE_INLINE agx::UInt32& ThreadTimelineEntryPtr::poolSize() { verifyIndex(); return getData()->poolSize[calculateIndex()]; }
  AGX_FORCE_INLINE agx::UInt32 const& ThreadTimelineEntryPtr::poolSize() const { verifyIndex(); return getData()->poolSize[calculateIndex()]; }

  AGX_FORCE_INLINE agx::UInt8& ThreadTimelineEntryPtr::jobType() { verifyIndex(); return getData()->jobType[calculateIndex()]; }
  AGX_FORCE_INLINE agx::UInt8 const& ThreadTimelineEntryPtr::jobType() const { verifyIndex(); return getData()->jobType[calculateIndex()]; }

  AGX_FORCE_INLINE agx::UInt64& ThreadTimelineEntryPtr::jobPtr() { verifyIndex(); return getData()->jobPtr[calculateIndex()]; }
  AGX_FORCE_INLINE agx::UInt64 const& ThreadTimelineEntryPtr::jobPtr() const { verifyIndex(); return getData()->jobPtr[calculateIndex()]; }

  AGX_FORCE_INLINE agx::String& ThreadTimelineEntryPtr::extraDataTitle() { verifyIndex(); return getData()->extraDataTitle[calculateIndex()]; }
  AGX_FORCE_INLINE agx::String const& ThreadTimelineEntryPtr::extraDataTitle() const { verifyIndex(); return getData()->extraDataTitle[calculateIndex()]; }

  AGX_FORCE_INLINE agx::Real64& ThreadTimelineEntryPtr::extraData() { verifyIndex(); return getData()->extraData[calculateIndex()]; }
  AGX_FORCE_INLINE agx::Real64 const& ThreadTimelineEntryPtr::extraData() const { verifyIndex(); return getData()->extraData[calculateIndex()]; }

  //-----------------------------------------------------------------------------------------------------
  AGX_FORCE_INLINE ThreadTimelineEntryInstance::ThreadTimelineEntryInstance() {}
  AGX_FORCE_INLINE ThreadTimelineEntryInstance::ThreadTimelineEntryInstance(ThreadTimelineEntryData* data, agx::Index index) : agxData::EntityInstance(data, index) {}
  AGX_FORCE_INLINE ThreadTimelineEntryInstance::ThreadTimelineEntryInstance(agxData::EntityStorage* storage, agx::Index index) : agxData::EntityInstance(storage, index) {}
  AGX_FORCE_INLINE ThreadTimelineEntryInstance::ThreadTimelineEntryInstance(const agxData::EntityInstance& other) : agxData::EntityInstance(other)
  {
    agxAssertN(!other || other.isInstanceOf(ThreadTimelineEntryModel::instance()),
      "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
      EntityInstance::getModel()->fullPath().c_str(), ThreadTimelineEntryModel::instance()->fullPath().c_str());
  }

  AGX_FORCE_INLINE ThreadTimelineEntryInstance::ThreadTimelineEntryInstance(const agxData::EntityPtr& ptr) : agxData::EntityInstance(ptr)
  {
    agxAssertN(!ptr || ptr.isInstanceOf(ThreadTimelineEntryModel::instance()),
      "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
      EntityInstance::getModel()->fullPath().c_str(), ThreadTimelineEntryModel::instance()->fullPath().c_str());
  }


  AGX_FORCE_INLINE ThreadTimelineEntryData* ThreadTimelineEntryInstance::getData() { return static_cast<ThreadTimelineEntryData* >(agxData::EntityInstance::getData()); }
  AGX_FORCE_INLINE const ThreadTimelineEntryData* ThreadTimelineEntryInstance::getData() const { return static_cast<const ThreadTimelineEntryData* >(agxData::EntityInstance::getData()); }

  AGX_FORCE_INLINE agx::UInt32& ThreadTimelineEntryInstance::taskId() { verifyIndex(); return getData()->taskId[getIndex()]; }
  AGX_FORCE_INLINE agx::UInt32 const& ThreadTimelineEntryInstance::taskId() const { verifyIndex(); return getData()->taskId[getIndex()]; }

  AGX_FORCE_INLINE agx::Real64& ThreadTimelineEntryInstance::startTimeGlobal() { verifyIndex(); return getData()->startTimeGlobal[getIndex()]; }
  AGX_FORCE_INLINE agx::Real64 const& ThreadTimelineEntryInstance::startTimeGlobal() const { verifyIndex(); return getData()->startTimeGlobal[getIndex()]; }

  AGX_FORCE_INLINE agx::Real64& ThreadTimelineEntryInstance::endTimeGlobal() { verifyIndex(); return getData()->endTimeGlobal[getIndex()]; }
  AGX_FORCE_INLINE agx::Real64 const& ThreadTimelineEntryInstance::endTimeGlobal() const { verifyIndex(); return getData()->endTimeGlobal[getIndex()]; }

  AGX_FORCE_INLINE agx::Real64& ThreadTimelineEntryInstance::startTimeFrame() { verifyIndex(); return getData()->startTimeFrame[getIndex()]; }
  AGX_FORCE_INLINE agx::Real64 const& ThreadTimelineEntryInstance::startTimeFrame() const { verifyIndex(); return getData()->startTimeFrame[getIndex()]; }

  AGX_FORCE_INLINE agx::Real64& ThreadTimelineEntryInstance::endTimeFrame() { verifyIndex(); return getData()->endTimeFrame[getIndex()]; }
  AGX_FORCE_INLINE agx::Real64 const& ThreadTimelineEntryInstance::endTimeFrame() const { verifyIndex(); return getData()->endTimeFrame[getIndex()]; }

  AGX_FORCE_INLINE agx::String& ThreadTimelineEntryInstance::description() { verifyIndex(); return getData()->description[getIndex()]; }
  AGX_FORCE_INLINE agx::String const& ThreadTimelineEntryInstance::description() const { verifyIndex(); return getData()->description[getIndex()]; }

  AGX_FORCE_INLINE agx::UInt32& ThreadTimelineEntryInstance::costEstimate() { verifyIndex(); return getData()->costEstimate[getIndex()]; }
  AGX_FORCE_INLINE agx::UInt32 const& ThreadTimelineEntryInstance::costEstimate() const { verifyIndex(); return getData()->costEstimate[getIndex()]; }

  AGX_FORCE_INLINE agx::UInt32& ThreadTimelineEntryInstance::poolSize() { verifyIndex(); return getData()->poolSize[getIndex()]; }
  AGX_FORCE_INLINE agx::UInt32 const& ThreadTimelineEntryInstance::poolSize() const { verifyIndex(); return getData()->poolSize[getIndex()]; }

  AGX_FORCE_INLINE agx::UInt8& ThreadTimelineEntryInstance::jobType() { verifyIndex(); return getData()->jobType[getIndex()]; }
  AGX_FORCE_INLINE agx::UInt8 const& ThreadTimelineEntryInstance::jobType() const { verifyIndex(); return getData()->jobType[getIndex()]; }

  AGX_FORCE_INLINE agx::UInt64& ThreadTimelineEntryInstance::jobPtr() { verifyIndex(); return getData()->jobPtr[getIndex()]; }
  AGX_FORCE_INLINE agx::UInt64 const& ThreadTimelineEntryInstance::jobPtr() const { verifyIndex(); return getData()->jobPtr[getIndex()]; }

  AGX_FORCE_INLINE agx::String& ThreadTimelineEntryInstance::extraDataTitle() { verifyIndex(); return getData()->extraDataTitle[getIndex()]; }
  AGX_FORCE_INLINE agx::String const& ThreadTimelineEntryInstance::extraDataTitle() const { verifyIndex(); return getData()->extraDataTitle[getIndex()]; }

  AGX_FORCE_INLINE agx::Real64& ThreadTimelineEntryInstance::extraData() { verifyIndex(); return getData()->extraData[getIndex()]; }
  AGX_FORCE_INLINE agx::Real64 const& ThreadTimelineEntryInstance::extraData() const { verifyIndex(); return getData()->extraData[getIndex()]; }

  //-----------------------------------------------------------------------------------------------------
  AGX_FORCE_INLINE ThreadTimelineEntrySemantics::ThreadTimelineEntrySemantics() {}
  //-----------------------------------------------------------------------------------------------------
  DOXYGEN_END_INTERNAL_BLOCK()
}

AGX_TYPE_BINDING(agx::ThreadTimelineEntryPtr, "ThreadTimelineEntryPtr")
AGX_TYPE_BINDING(agx::ThreadTimelineEntryInstance, "ThreadTimelineEntryInstance")

#ifdef _MSC_VER
# pragma warning(pop)
#endif

#undef AGX_ENTITY_WRAPPER
#undef AGX_ENTITY_NAMESPACE
#endif

