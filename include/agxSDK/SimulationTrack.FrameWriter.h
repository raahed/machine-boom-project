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

  AGX_DECLARE_POINTER_TYPES(SimulationFrameWriter);
  class AGXPHYSICS_EXPORT SimulationFrameWriter : public agxData::FrameWriter
  {
  public:
    SimulationFrameWriter(Simulation *simulation, bool passive = false);

    virtual void writeFrame(const agxData::Frame *frame) override;

    virtual void forceFlush() override;

    agx::Task *getUpdateTask() { return m_updateTask; }

    void setBlocking(bool blocking) { m_blocking = blocking; }

    // Bypass thread synchronization
    void writeFrameImpl(const agxData::Frame *frame);


  protected:
    virtual ~SimulationFrameWriter();

  private:
    void writeIncrementalArchive(const agxData::Buffer *archiveData, agx::UInt64 skipFrame, agxSDK::Assembly *assembly);
    void updateCallback();
    void write(const agx::Component *component, const agxData::Frame *frame);
    void write(const agxData::Buffer *buffer, const agxData::Frame *frame);
    void write(const agxData::Frame::PartialBuffer *buffer, const agxData::Frame *frame);
    void write(const agxData::Frame::EntityStorage *storage, const agxData::Frame *frame);
    void write(const agxData::Value *value, const agxData::Frame *frame);
    void evaluateEventsPre(const agxData::Frame *frame);
    void evaluateEventsPost(const agxData::Frame *frame);
    void writeConstraintData(const agxData::Frame *frame);
    bool writeFrameArchive(const agxData::Frame *frame);
    agx::UInt64 writeKeyFrameArchive(const agxData::Buffer* keyFrameArchive, agxSDK::Assembly* renderAssembly );
    void updateArrayBuffers();

    void debugPrint(const agxData::Frame *frame);
    void debugPrint(const agx::Component *component, const agxData::Frame *frame);
    void debugPrint(const agxData::Frame::PartialBuffer *buffer, const agxData::Frame *frame);
    void debugPrint(const agxData::Buffer *buffer, const agxData::Frame *frame);
    void debugPrint(const agxData::EntityStorage *storage, const agxData::Frame *frame);
    void debugPrint(const agxData::Value *value, const agxData::Frame *frame);

    void setClock();

    void handleGeometryRestore(agxCollide::Geometry* geometry);
    void storeAndReplaceMaterialPtr(agxCollide::Geometry* geometry);
    void storeTrimeshShallowCopyRefs(agxCollide::Geometry* geometry);

    template <typename T>
    T *getTarget(const Object *source);

    template <typename T>
    T *getTarget(agx::UInt32 id);

    template <typename T>
    agxData::EntityStorage *getEventStorage(T *event);

    struct ArchiveObjectEntry {
      agx::UInt32 id;
      agxStream::Serializable* serializablePtr;
    };

    class IncrementalArchive : public agx::Observer
    {
    public:
      IncrementalArchive();
      virtual ~IncrementalArchive();

      agxStream::InputArchive& createArchive(const agxData::Buffer *archiveData, bool binary = true);
      // void attach(agxStream::OutputArchive *target);

      std::stringstream& getStream();
      void clear();
      void detach();

    private:
      virtual void objectDeleted(void *target);
      void restoreCallback(agx::UInt32 id, agxStream::Serializable *object, const std::string& className);

    private:
      std::unique_ptr<std::stringstream> m_stream;
      agxStream::InputArchiveRef m_target;
      agxStream::InputArchive::RestoreEvent::CallbackType m_restoreCallback;
      agxStream::InputArchive::IdToObjectVector m_objectCache;
      agxStream::InputArchive::IdToClassNameVector m_nameCache;
      agx::HashTable<agx::Referenced*, ArchiveObjectEntry> m_objectToId;
      agx::Vector<agx::UInt32> m_nonRef;
    };

    agxData::EntityStorage *getStorage(agxData::EntityStorage *encodedPtr);
    agxData::Buffer *getBuffer(agxData::Buffer *encodedPtr);
    void rebuildObjectTable(const agx::Object *node);

  private:
    agx::observer_ptr<Simulation> m_simulation;
    bool m_blocking;

    agx::UInt m_frameIndex;
    agx::Real m_timeStamp;
    agx::Real m_timeStep;
    bool m_needClockUpdate;

    agx::TaskGroupRef m_updateTask;
    bool m_forceFlush;
    agx::Clock::TickEvent::CallbackType m_postTickCallback;

    typedef agx::HashTable<agx::UInt32, agx::ObjectObserver> ObjectIdTable;
    ObjectIdTable m_objectIdTable;
    agxData::BufferRefVector m_arrayBuffers;
    bool m_passive;
    IncrementalArchive m_incrementalArchive;
    agx::HashSet<agx::MaterialRef> m_incrementalJournalMaterials;
    agx::HashSet<agxCollide::ShapeConstRef> m_uniqueTrimeshes;
    agx::HashSet<agxCollide::RenderDataRef> m_uniqueRenderData;
  };

}
