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

//-----------------------------------------------
// AUTOMATICALLY GENERATED EVENT, DO NOT EDIT!   
//-----------------------------------------------

#ifndef GENERATED_AGX_BUFFER_H_DECLARATION
#define GENERATED_AGX_BUFFER_H_DECLARATION

#include <agx/macros.h>

DOXYGEN_START_INTERNAL_BLOCK()
class EventDispatch;

class AGXCORE_EXPORT EventListener
{
public:

  EventListener(agx::Object *owner = nullptr);
  virtual ~EventListener();

  agx::Object* getOwner();
  template <typename T>
  T* getOwner();

  virtual void destroyCallback(agxData::Buffer* buffer);
  virtual void resizeCallback(agxData::Buffer* buffer, agx::Index size, agx::Index oldSize);
  virtual void updateCallback(agxData::Buffer* buffer);
  virtual void updateCallback(agxData::Buffer* buffer, agx::Index index);
  virtual void updateCallback(agxData::Buffer* buffer, agx::IndexRange range);
  virtual void updateCallback(agxData::Buffer* buffer, agxData::Array< agx::Index > indexSet);

private:
  friend class EventDispatch;
private:
  agx::Object* m_owner;
  agx::UInt32 m_numRegistered;
};

class AGXCORE_EXPORT EventDispatch
{
public:
  EventDispatch();
  ~EventDispatch();

public:
  void addListener(EventListener* listener);
  void removeListener(EventListener* listener);
  bool hasListener(EventListener* listener) const;

  typedef agx::VectorPOD< EventListener* > EventListenerPtrVector;
  const EventListenerPtrVector& getListeners() const;

public:
  void triggerDestroyEvent(agxData::Buffer* buffer);
  void triggerResizeEvent(agxData::Buffer* buffer, agx::Index size, agx::Index oldSize);
  void triggerUpdateEvent(agxData::Buffer* buffer);
  void triggerUpdateEvent(agxData::Buffer* buffer, agx::Index index);
  void triggerUpdateEvent(agxData::Buffer* buffer, agx::IndexRange range);
  void triggerUpdateEvent(agxData::Buffer* buffer, agxData::Array< agx::Index > indexSet);

private:
  EventListenerPtrVector m_listeners;
  agx::Thread *m_activeThread;
  agx::UInt32 m_iterationIndex;
};

class AGXCORE_EXPORT DestroyEvent : public agx::Object::Event
{
public:
  DestroyEvent(agxData::Buffer* buffer);

  agxData::Buffer* getBuffer();

protected:
  virtual ~DestroyEvent();

private:
  agxData::Buffer* m_buffer;
};

class AGXCORE_EXPORT ResizeEvent : public agx::Object::Event
{
public:
  ResizeEvent(agxData::Buffer* buffer, agx::Index size, agx::Index oldSize);

  agxData::Buffer* getBuffer();
  agx::Index getSize();
  agx::Index getOldSize();

protected:
  virtual ~ResizeEvent();

private:
  agxData::Buffer* m_buffer;
  agx::Index m_size;
  agx::Index m_oldSize;
};

class AGXCORE_EXPORT UpdateEvent : public agx::Object::Event
{
public:
  UpdateEvent(agxData::Buffer* buffer);

  agxData::Buffer* getBuffer();

protected:
  virtual ~UpdateEvent();

private:
  agxData::Buffer* m_buffer;
};

class AGXCORE_EXPORT UpdateEvent2 : public agx::Object::Event
{
public:
  UpdateEvent2(agxData::Buffer* buffer, agx::Index index);

  agxData::Buffer* getBuffer();
  agx::Index getIndex();

protected:
  virtual ~UpdateEvent2();

private:
  agxData::Buffer* m_buffer;
  agx::Index m_index;
};

class AGXCORE_EXPORT UpdateEvent3 : public agx::Object::Event
{
public:
  UpdateEvent3(agxData::Buffer* buffer, agx::IndexRange range);

  agxData::Buffer* getBuffer();
  agx::IndexRange getRange();

protected:
  virtual ~UpdateEvent3();

private:
  agxData::Buffer* m_buffer;
  agx::IndexRange m_range;
};

class AGXCORE_EXPORT UpdateEvent4 : public agx::Object::Event
{
public:
  UpdateEvent4(agxData::Buffer* buffer, agxData::Array< agx::Index > indexSet);

  agxData::Buffer* getBuffer();
  agxData::Array< agx::Index > getIndexSet();

protected:
  virtual ~UpdateEvent4();

private:
  agxData::Buffer* m_buffer;
  agxData::Array< agx::Index > m_indexSet;
};


DOXYGEN_END_INTERNAL_BLOCK()
#endif // GENERATED_AGX_BUFFER_H_DECLARATION
