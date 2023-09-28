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

#ifndef AGXDATA_FRAME_H
#define AGXDATA_FRAME_H

#include <agx/Device.h>
#include <agx/Component.h>
#include <agx/Clock.h>

#ifdef _MSC_VER
#pragma warning( push )
#pragma warning( disable : 4275 ) // non dll-interface class 'std::runtime_error' used as base for dll-interface class 'agxData::Frame::MergeException'
#pragma warning( disable : 4290 ) // C++ exception specification ignored except to indicate a function is not __declspec(nothrow)
#endif


DOXYGEN_START_INTERNAL_BLOCK()

namespace agxData
{
  class Track;

  AGX_DECLARE_POINTER_TYPES(Frame);
  AGX_DECLARE_VECTOR_TYPES(Frame);
  class AGXCORE_EXPORT Frame : public agx::Component
  {
  public:
    static agx::Model *ClassModel();

  public:
    AGX_DECLARE_POINTER_TYPES(DataBinding);
    AGX_DECLARE_VECTOR_TYPES(DataBinding);

    AGX_DECLARE_POINTER_TYPES(PartialBuffer);
    AGX_DECLARE_VECTOR_TYPES(PartialBuffer);
    AGX_DECLARE_POINTER_TYPES(EntityStorage);
    AGX_DECLARE_VECTOR_TYPES(EntityStorage);

  public:
    struct Header
    {
      Header() : index(agx::InvalidIndex), timeStep(-1), timeStamp(-1), id(agx::InvalidIndex), isKeyFrame(false) {}
      agx::UInt index;
      agx::Real timeStep;
      agx::Real timeStamp;
      agx::UInt32 id;
      bool isKeyFrame;
    };

  public:
    Frame(agx::UInt index, agx::Real timeStamp, agx::Real timeStep);
    Frame(agx::Clock *clock);

    /**
    \return the frame header.
    */
    Header getHeader() const;

    /**
    Set to true if frame is keyframe. Default is false.
    */
    void setKeyFrame(bool flag);

    /**
    \return true if frame is keyframe.
    */
    bool isKeyFrame() const;

    /**
    \return The index of the frame.
    */
    agx::UInt getIndex() const;

    /**
    Set the frame index.
    */
    void setIndex(agx::UInt index);

    /**
    \return The timestamp of the frame.
    */
    agx::Real getTimeStamp() const;

    /**
    Set the time stamp.
    */
    void setTimeStamp(agx::Real timeStamp);

    /**
    \return The timestep of the frame.
    */
    agx::Real getTimeStep() const;

    /**
    Set the time step.
    */
    void setTimeStep(agx::Real dt);

    /**
    \return The number of bytes stored in the frame.
    */
    agx::UInt getNumBytes() const;

    /**
    \return The track which the frame belongs to.
    */
    Track *getTrack();
    const Track *getTrack() const;

    class AGXCORE_EXPORT MergeException : public std::runtime_error
    {
    public:
      MergeException(const Object* node, const Object* collisionNode);

      const Object *getNode() const;
      const Object *getCollisionNode() const;

    private:
      const Object *m_node;
      const Object *m_collisionNode;
    };

    /**
    Merge content from another frame into this frame.
    \param other The other frame
    \param allowOverwrite Set to true if path conflicts are allowed, if so, use data from 'other' frame
    */
    void merge(const Frame *other, bool allowOverwrite = false);

    // Alias for frame->getResource<agxData::Buffer>(path);
    Buffer *getBuffer(const agx::Path& path);

    void addStorageMetaData(agxData::EntityStorage *storage, const agx::Path& path);
    void addComponent(Component *component, const agx::Path& path);
    void addStorage(EntityStorage *storage, const agx::Path& path);
    void addBuffer(Buffer *buffer, const agx::Path& path);
    void addBuffer(PartialBuffer *buffer, const agx::Path& path);
    void addValue(agxData::Value *value, const agx::Path& path);

    // Used for tracking attributes of individual entity instances
    void addInstanceAttribute(agxData::EntityPtr instance, agxData::Attribute *attribute);


    // TODO: Should always be handled automatically by frame?
    void setNumBytes(agx::UInt numBytes);


    void addEvent(agx::Object::Event *event);
    void addEvents(const agx::Object::EventRefVector& events);
    agx::Object::EventRefVector& getEvents();
    const agx::Object::EventRefVector& getEvents() const;

    void evaluateEventsLocally();
    bool isQueued() const;
    bool isDiscardedFromCache() const;
    bool isJumpFrame() const;
    bool isSerialized() const;

    void setIncrementalFrameArchiveBuffer(agxData::Buffer *buffer);
    const agxData::Buffer *getIncrementalFrameArchiveBuffer() const;
    agxData::Buffer *getIncrementalFrameArchiveBuffer();


  protected:
    virtual ~Frame();

  private:
    friend class Track;
    void setTrack(Track *track);

    void setQueued(bool flag);

    void setDiscardedFromCache(bool flag);

    void setJumpFrame(bool flag);

    void setSerialized(bool flag);

    agx::Component *getContext(const agx::Path& path);


    Frame::EntityStorage *getStorageWithId(agx::Component *node, agx::UInt32 id);

    typedef agx::HashTable<agx::Object *, agx::Object *> FrameMap;
    void buildFrameMap(agx::Component *node, FrameMap& map, const Frame *root);


    void merge(agx::Component *targetParent, const agxData::Value *value, bool allowOverwrite);
    void merge(agx::Component *targetParent, const agxData::Buffer *buffer, bool allowOverwrite);
    void merge(agx::Component *targetParent, const PartialBuffer *buffer, bool allowOverwrite);
    void merge(agx::Component *targetParent, const EntityStorage *storage, bool allowOverwrite);
    void merge(agx::Component *targetParent, const agx::Component *component, bool allowOverwrite);
    void merge(agx::Component *targetParent, const agx::Object *node, bool allowOverwrite);
    void merge(agx::Component *targetParent, const agx::ObjectRefVector& objects, bool allowOverwrite);

    void addNode(Component *parent, Object *object);
    void removeNode(Object *object);
    static size_t calculateNumBytes(Object *node);

  private:
    Track *m_track;
    bool m_isQueued;
    bool m_isDiscardedFromCache;
    bool m_isJumpFrame;
    bool m_isKeyFrame;
    bool m_serialized;
    agx::UInt m_index;
    agx::Real m_timeStamp;
    agx::Real m_timeStep;
    agx::UInt m_numBytes;
    agx::Object::EventRefVector m_events;
    agxData::BufferRef m_incrementalFrameBuffer;
    const agxData::Frame* m_mergeFrame;

    // typedef HashTable<agx::Path, Buffer *> BufferTable;
    // BufferTable m_bufferTable;
  };


  class AGXCORE_EXPORT Frame::DataBinding : public agx::Referenced
  {
  public:
    DataBinding(const agx::Path& path);
    DataBinding(const agx::Path& internalPath, const agx::Path& externalPath);

    // TODO Specify format transformations in binding?
    // DataBinding(const agx::Path& internalPath, const Format *internalFormat, const agx::Path& externalPath, const Format *externalFormat);

    /**
    The internal binding is the binding in the common frame format.
    */
    const agx::Path& internalPath() const;

    /**
    The external binding is the implementation specific binding.
    */
    const agx::Path& externalPath() const;

    /// \return True if the binding is recursive
    bool isRecursive() const;

    /// Set true to enable recursive binding
    void setRecursive(bool flag);

  protected:
    virtual ~DataBinding() {}

  private:
    agx::Path m_internalPath;
    agx::Path m_externalPath;
    bool m_isRecursive;
  };

  class AGXCORE_EXPORT Frame::PartialBuffer : public agx::Component
  {
  public:
    static agx::Model *ClassModel();

  public:
    // PartialBuffer();
    PartialBuffer(const agx::Name& name, agxData::Format *dataFormat, size_t size, size_t fullSize);
    PartialBuffer(const agx::Name& name, size_t size, size_t fullSize);
    PartialBuffer(const agx::Name& name, agxData::Buffer *indexBuffer, agxData::Buffer *dataBuffer, size_t fullSize);

    bool empty() const;
    size_t size() const;
    size_t fullSize() const;
    // size_t fullCapacity() const;


    void resize(size_t size);
    const agxData::Format *getFormat() const;

    agxData::Buffer *getDataBuffer();
    const agxData::Buffer *getDataBuffer() const;

    agxData::Buffer *getIndexBuffer();
    const agxData::Buffer *getIndexBuffer() const;

    void setDataBuffer(agxData::Buffer *buffer);

    size_t getNumBytes() const;

    void merge(const PartialBuffer *other, bool allowOverwrite);

    PartialBuffer *clone() const;

  protected:
    virtual ~PartialBuffer();

  private:
    agxData::BufferRef m_indexBuffer;
    agxData::BufferRef m_dataBuffer;
    size_t m_fullSize;
    // size_t m_fullCapacity;
  };

  class AGXCORE_EXPORT Frame::EntityStorage : public agx::Component
  {
  public:
    static agx::Model *ClassModel();

  public:
    EntityStorage(agxData::EntityStorage *storage);
    EntityStorage(const agx::Name& name, agxData::EntityModel *model, size_t size, size_t capacity);

    EntityModel *getModel();
    const EntityModel *getModel() const;

    bool empty() const;
    size_t size() const;
    size_t capacity() const;

    // Explicitly set size parameter
    void setSize(size_t size);
    void setCapacity(size_t capacity);

    virtual void addObject(agx::Object *object, bool assignContext = true);
    virtual void removeObject(agx::Object *object);


    // EntityStorage semantics for evaluating events locally during frame merge
    void resize(size_t size);
    void reserve(size_t capacity);
    void destroyInstance(agx::Index id);

  protected:
    virtual ~EntityStorage();

  private:
    agxData::EntityModelRef m_model;
    size_t m_size;
    size_t m_capacity;
    agxData::Buffer *m_idToIndexBuffer;
    agxData::Buffer *m_indexToIdBuffer;
    typedef agx::HashSet<Buffer *> BufferHash;
    BufferHash m_attributeBuffers;
    // PartialBufferPtrVector m_partialBuffers;
  };



  AGXCORE_EXPORT std::ostream& operator<< ( std::ostream& stream, const agxData::Frame::Header& header );

  //---------------------------------------------------------------



  /* Implementation */
  AGX_FORCE_INLINE Track *Frame::getTrack() { return m_track; }
  AGX_FORCE_INLINE const Track *Frame::getTrack() const { return m_track; }

  AGX_FORCE_INLINE bool Frame::isQueued() const { return m_isQueued; }
  AGX_FORCE_INLINE bool Frame::isJumpFrame() const { return m_isJumpFrame; }
  AGX_FORCE_INLINE bool Frame::isDiscardedFromCache() const { return m_isDiscardedFromCache; }

  AGX_FORCE_INLINE bool Frame::isKeyFrame() const { return m_isKeyFrame; }
  AGX_FORCE_INLINE agx::UInt Frame::getIndex() const { return m_index; }
  AGX_FORCE_INLINE agx::Real Frame::getTimeStamp() const { return m_timeStamp; }
  AGX_FORCE_INLINE agx::Real Frame::getTimeStep() const { return m_timeStep; }
  AGX_FORCE_INLINE agx::UInt Frame::getNumBytes() const { return m_numBytes; }

  AGX_FORCE_INLINE const agx::Object::EventRefVector& Frame::getEvents() const { return m_events; }
  AGX_FORCE_INLINE agx::Object::EventRefVector& Frame::getEvents() { return m_events; }

  //---------------------------------------------------------------

  AGX_FORCE_INLINE bool Frame::PartialBuffer::empty() const { return size() == 0; }
  AGX_FORCE_INLINE size_t Frame::PartialBuffer::size() const { return m_dataBuffer ? m_dataBuffer->size() : 0; }
  AGX_FORCE_INLINE size_t Frame::PartialBuffer::fullSize() const { return m_fullSize; }
  // AGX_FORCE_INLINE size_t Frame::PartialBuffer::fullCapacity() const { return m_fullCapacity; }

  AGX_FORCE_INLINE const agxData::Format *Frame::PartialBuffer::getFormat() const { return m_dataBuffer ? m_dataBuffer->getFormat() : 0; }

  AGX_FORCE_INLINE agxData::Buffer *Frame::PartialBuffer::getDataBuffer() { return m_dataBuffer; }
  AGX_FORCE_INLINE const agxData::Buffer *Frame::PartialBuffer::getDataBuffer() const { return m_dataBuffer; }
  AGX_FORCE_INLINE agxData::Buffer *Frame::PartialBuffer::getIndexBuffer() { return m_indexBuffer; }
  AGX_FORCE_INLINE const agxData::Buffer *Frame::PartialBuffer::getIndexBuffer() const { return m_indexBuffer; }

  //---------------------------------------------------------------

  AGX_FORCE_INLINE EntityModel *Frame::EntityStorage::getModel() { return m_model; }
  AGX_FORCE_INLINE const EntityModel *Frame::EntityStorage::getModel() const { return m_model; }
  AGX_FORCE_INLINE bool Frame::EntityStorage::empty() const { return size() == 0; }
  AGX_FORCE_INLINE size_t Frame::EntityStorage::size() const { return m_size; }
  AGX_FORCE_INLINE size_t Frame::EntityStorage::capacity() const { return m_capacity; }

}

DOXYGEN_END_INTERNAL_BLOCK()

#ifdef _MSC_VER
#pragma warning( pop )
#endif

#endif /* AGXDATA_FRAME_H */
