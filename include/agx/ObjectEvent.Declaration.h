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

#ifndef GENERATED_AGX_OBJECT_H_DECLARATION
#define GENERATED_AGX_OBJECT_H_DECLARATION

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

  virtual void destroyCallback(agx::Object* object);

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
  void triggerDestroyEvent(agx::Object* object);

private:
  EventListenerPtrVector m_listeners;
  agx::Thread *m_activeThread;
  agx::UInt32 m_iterationIndex;
};

class AGXCORE_EXPORT DestroyEvent : public agx::Object::Event
{
public:
  DestroyEvent(agx::Object* object);

  agx::Object* getObject();

protected:
  virtual ~DestroyEvent();

private:
  agx::Object* m_object;
};


DOXYGEN_END_INTERNAL_BLOCK()
#endif // GENERATED_AGX_OBJECT_H_DECLARATION
