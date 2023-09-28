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

#include <agxData/Frame.h>
#include <agxData/FrameIO.h>
#include <memory>
#include <agx/DynamicBitSet.h>
#include <agx/Statistics.h>
#include <agx/TaskTimingEntity.h>
#include <agxSDK/Simulation.h>
#include <agxStream/OutputArchive.h>
#include <agxStream/InputArchive.h>

namespace agxCollide
{
  class Geometry;
}

namespace agxSDK
{
  class Assembly;

  AGX_DECLARE_POINTER_TYPES(SimulationFrameReader);
  class AGXPHYSICS_EXPORT SimulationFrameReader : public agxData::FrameReader
  {
  public:
    SimulationFrameReader(Simulation *simulation, bool passive = false);


    virtual agxData::Frame *readFrame() override;

    /**
    Generate an explicit key frame. Does not reset dirty
    flags for incremental frame sequence.
    */
    agxData::Frame *generateKeyFrame();

    void start();
    void stop();
    void disconnect();

    void setReadFrameOnNextPreTick(bool flag);

    void forcePushCurrentFrame();

    /// Set keyframe interval
    void setKeyFrameInterval(agx::UInt interval);

    void setStoreConstraintData(bool flag);
    bool getStoreConstraintData() const;

    /// \return The key frame interval
    agx::UInt getKeyFrameInterval() const;

    virtual agx::Real getTimeStep() override;
    virtual agx::UInt getFrameStride()  override;

    /**
    Enable/disable incremental simulation structure recording. Enabled by default.
    */
    void setEnableIncrementalStructure(bool flag);

    /**
    \return True if incremental structure recording is enabled.
    */
    bool getEnableIncrementalStructure() const;


  protected:
    void tryPushCurrentFrameToTrack();

    virtual ~SimulationFrameReader();

  private:
    AGX_DECLARE_POINTER_TYPES(DataBinding);
    AGX_DECLARE_POINTER_TYPES(DirtySet);


    class ValueListener : public agxData::Value::EventListener
    {
    public:
      ValueListener(SimulationFrameReader *reader);

      virtual void destroyCallback(agxData::Value* value) override;
      virtual void updateCallback(agxData::Value* value) override;

    private:
      SimulationFrameReader *reader();
    };

    class BufferListener : public agxData::Buffer::EventListener
    {
    public:
      BufferListener(SimulationFrameReader *reader);

      virtual void destroyCallback(agxData::Buffer* buffer) override;
      virtual void resizeCallback(agxData::Buffer* buffer, agx::Index size, agx::Index oldSize) override;
      virtual void updateCallback(agxData::Buffer* buffer) override;
      virtual void updateCallback(agxData::Buffer* buffer, agx::Index index) override;
      virtual void updateCallback(agxData::Buffer* buffer, agx::IndexRange range) override;
      virtual void updateCallback(agxData::Buffer* buffer, agxData::Array< agx::Index > indexSet) override;

    private:
      SimulationFrameReader *reader();
    };

    class StorageListener : public agxData::EntityStorage::EventListener
    {
    public:
      StorageListener(SimulationFrameReader *reader);

      virtual void destroyCallback(agxData::EntityStorage* storage) override;
      virtual void createInstanceCallback(agxData::EntityStorage* storage, agxData::EntityPtr instance) override;
      virtual void createInstancesCallback(agxData::EntityStorage* storage, agxData::EntityRange instances) override;
      virtual void createInstancesCallback(agxData::EntityStorage* storage, agxData::Array<agxData::EntityPtr> instances) override;
      virtual void destroyInstanceCallback(agxData::EntityStorage* storage, agxData::EntityPtr instance) override;
      virtual void destroyInstancesCallback(agxData::EntityStorage* storage, agxData::EntityRange instances) override;
      virtual void destroyInstancesCallback(agxData::EntityStorage* storage, agxData::Array<agxData::EntityPtr> instances) override;
      virtual void permuteCallback(agxData::EntityStorage* storage, agxData::Array< agx::Index > permutation) override;

    private:
      SimulationFrameReader *reader();
    };

    class IncrementalArchive : public agx::Observer
    {
    public:
      IncrementalArchive();
      virtual ~IncrementalArchive();

      agxStream::OutputArchive& createArchive(bool binary);
      // void attach(agxStream::OutputArchive *target);

      std::stringstream& getStream();
      void clear();
      void detach();

      agx::UInt32 getIdCounter();
      void setIdCounter(agx::UInt32 id);

    private:
      virtual void objectDeleted(void *target);
      void storeCallback(const agxStream::Serializable *object, agx::UInt32 id);

    private:
      std::unique_ptr<std::stringstream> m_stream;
      agxStream::OutputArchiveRef m_target;
      agxStream::OutputArchive::StoreEvent::CallbackType m_storeCallback;
      agxStream::OutputArchive::ObjectToIdMapContext m_mapContext;
      agx::HashTable<const agx::Referenced*, const agxStream::Serializable*> m_refSerializableTable;
      agx::Vector<const agxStream::Serializable *> m_nonRef;
    };


    virtual agxData::Frame::DataBinding *createDataBinding(const agx::Path& internalPath, const agx::Path& externalPath) override;

    friend class agx::Journal;
    void preTickCallback(agx::Clock *clock);
    void postTickCallback(agx::Clock *clock);
    void synchronizeConstraintForceData();
    void postSaveCallback();

    void addListenerToValue(agxData::Value *value);
    void addListenerToBuffer(agxData::Buffer *buffer);
    void addListenerToStorage(agxData::EntityStorage *storage);

    void addStatisticsToFrame(agxData::Frame *frame);
    void addStatisticsToFrame(agxData::Frame *frame, agx::Component *frameNode, agx::Statistics::ProviderHandle *providerNode);

    void addObjectToFrame(agxData::Frame *frame, agx::Object *object, const agx::Path& bindPath, bool resetDirtyFlags, bool recursive);
    void addValueToFrame(agxData::Frame *frame, agxData::Value *value, const agx::Path& bindPath, bool resetDirtyFlags);
    void addBufferToFrame(agxData::Frame *frame, agxData::Buffer *buffer, const agx::Path& bindPath, bool resetDirtyFlags);
    void addStorageToFrame(agxData::Frame *frame, agxData::EntityStorage *storage, const agx::Path& bindPath, bool resetDirtyFlags);
    void addIndexToIdBufferToFrame(agxData::Frame *frame, agxData::EntityStorage *storage, const agx::Path& bindPath, bool resetDirtyFlags);
    void initializeBinding(DataBinding *binding, bool doingActualSimulation);
    void initializeBinding(agx::Object *object, bool doingActualSimulation, bool recursive);
    agxData::Frame *readFrameImpl(bool isSequenceFrame);

    void addConstraintDataToFrame( agxData::Frame *frame );


    agxData::EntityStorage *extractTaskProfilingData(agx::Task *root);
    void extractTaskProfilingData(agx::Task *task, agx::TaskTimingData& timingData);
    agxData::Frame::EntityStorage *exportBuffersToFrameStorage(agxData::EntityStorage *storage);


  private:
    friend class ValueListener;
    ValueListener m_valueListener;

    friend class BufferListener;
    BufferListener m_bufferListener;

    friend class StorageListener;
    StorageListener m_storageListener;

    typedef agx::HashTable<agxData::Value *, bool> DirtyValueTable;
    DirtyValueTable m_activeValues;

    typedef agx::HashTable<agxData::Buffer *, DirtySetRef> DirtyBufferTable;
    DirtyBufferTable m_activeBuffers;

    typedef agx::HashSet<agxData::EntityStorage *> StorageTable;
    StorageTable m_activeStorages;

    agx::observer_ptr<Simulation> m_simulation;
    agx::Clock::TickEvent::CallbackType m_preTickCallback;
    agx::Clock::TickEvent::CallbackType m_postTickCallback;
    agx::UInt m_frameCount;
    agx::UInt m_frameToRead;
    agx::Block m_simBlock;
    bool m_shutdown;
    agxData::EntityModelRef m_constraintForceModel;
    agx::Object::EventRefVector m_events;
    bool m_saveFrameOnNextPreTick;
    agx::UInt m_keyFrameInterval;
    agx::UInt m_framesSinceLastKeyFrame;
    bool m_storeConstraintData;
    agx::Real m_nextProfilingTimeStamp;
    agx::Task::ProfilingMode m_taskProfilingMode;
    bool m_enableThreadTimeline;
    bool m_isProfilingFrame;
    bool m_passive;


    typedef agx::HashTable<agx::Task *, agx::UInt> TaskProfilingSizeTable;
    TaskProfilingSizeTable m_taskProfilingSizeTable;

    agxData::BufferRefVector m_frameExportBuffers;

    IncrementalArchive m_incrementalArchive;

  private:
    agxData::EntityStorage *extractCustomTaskProfiling();
    void extractCustomTaskProfiling(agx::TaskTimingReportHandle* node, agx::TaskTimingData& timingData);

    agx::Component *createFrameArchive(agxData::Frame* frame, bool binary = true);
    agxData::Buffer *createKeyFrameArchive(bool binary);

    void addRigidBodyCallback(Simulation *, agx::RigidBody *body);
    void removeRigidBodyCallback(Simulation *, agx::RigidBody *body);

    void addGeometryCallback(Simulation *, agxCollide::Geometry *geometry);
    void removeGeometryCallback(Simulation *, agxCollide::Geometry *geometry);

    void addConstraintCallback(Simulation *, agx::Constraint *constraint);
    void removeConstraintCallback(Simulation *, agx::Constraint *constraint);

    Simulation::RigidBodyEvent::CallbackType m_addRigidBodyCallback;
    Simulation::RigidBodyEvent::CallbackType m_removeRigidBodyCallback;

    Simulation::ConstraintEvent::CallbackType m_addConstraintCallback;
    Simulation::ConstraintEvent::CallbackType m_removeConstraintCallback;

    Simulation::GeometryEvent::CallbackType m_addGeometryCallback;
    Simulation::GeometryEvent::CallbackType m_removeGeometryCallback;

  public:

    struct ArchiveEvent
    {
      enum Type
      {
        RIGID_BODY_ADD,
        RIGID_BODY_REMOVE,
        CONSTRAINT_ADD,
        CONSTRAINT_REMOVE,
        GEOMETRY_ADD,
        GEOMETRY_REMOVE,
        GARBAGE_COLLECT
      };

      Type type;
      agx::ref_ptr<agx::Referenced> object;
    };


  private:
    agx::Vector<ArchiveEvent> m_archiveEvents;
    bool m_enableIncrementalStructure;
  };

  //////////////////////////////////////////////////////////////////////////////////////////////////////

  class SimulationFrameReader::DataBinding : public agxData::Frame::DataBinding
  {
  public:
    DataBinding(const agx::Path& internalPath, const agx::Path& externalPath, agx::Object *object = nullptr);

    agx::Object *getObject();
    void setObject(agx::Object *object);

  protected:
    virtual ~DataBinding();

  private:
    agx::ObjectObserver m_object;
  };

  //////////////////////////////////////////////////////////////////////////////////////////////////////


  class SimulationFrameReader::DirtySet : public agx::Referenced
  {
  public:
    DirtySet(bool markAllDirty = false);

    bool empty() const; // Return true if size is 0 and isAllDirty is false

    size_t size() const;

    bool isAllDirty() const;
    void markAllDirty();
    void reset();

    void addIndex(agx::Index index);
    const agx::IndexVector& getIndexList() const;

  private:
    bool m_allDirty;
    agx::IndexVector m_indexList;
    agx::DynamicBitSet m_bitSet;
  };



  /* Implementation */
  AGX_FORCE_INLINE agx::UInt SimulationFrameReader::getKeyFrameInterval() const { return m_keyFrameInterval; }
  AGX_FORCE_INLINE bool SimulationFrameReader::DirtySet::empty() const { return m_indexList.empty() && !m_allDirty; }
  AGX_FORCE_INLINE size_t SimulationFrameReader::DirtySet::size() const { return m_indexList.size(); }
  AGX_FORCE_INLINE bool SimulationFrameReader::DirtySet::isAllDirty() const { return m_allDirty; }
  AGX_FORCE_INLINE const agx::IndexVector& SimulationFrameReader::DirtySet::getIndexList() const { return m_indexList; }

  AGX_FORCE_INLINE SimulationFrameReader *SimulationFrameReader::ValueListener::reader() { return this->getOwner<SimulationFrameReader>(); }
  AGX_FORCE_INLINE SimulationFrameReader *SimulationFrameReader::BufferListener::reader() { return this->getOwner<SimulationFrameReader>(); }
  AGX_FORCE_INLINE SimulationFrameReader *SimulationFrameReader::StorageListener::reader() { return this->getOwner<SimulationFrameReader>(); }

}
