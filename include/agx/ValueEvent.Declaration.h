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

#ifndef GENERATED_AGX_VALUE_H_DECLARATION
#define GENERATED_AGX_VALUE_H_DECLARATION

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

  virtual void destroyCallback(agxData::Value* value);
  virtual void updateCallback(agxData::Value* value);
  virtual void bindCallback(agxData::Value* value, agxData::Value* binding, agxData::Value* oldBinding);

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
  void triggerDestroyEvent(agxData::Value* value);
  void triggerUpdateEvent(agxData::Value* value);
  void triggerBindEvent(agxData::Value* value, agxData::Value* binding, agxData::Value* oldBinding);

private:
  EventListenerPtrVector m_listeners;
  agx::Thread *m_activeThread;
  agx::UInt32 m_iterationIndex;
};

class AGXCORE_EXPORT DestroyEvent : public agx::Object::Event
{
public:
  DestroyEvent(agxData::Value* value);

  agxData::Value* getValue();

protected:
  virtual ~DestroyEvent();

private:
  agxData::Value* m_value;
};

class AGXCORE_EXPORT UpdateEvent : public agx::Object::Event
{
public:
  UpdateEvent(agxData::Value* value);

  agxData::Value* getValue();

protected:
  virtual ~UpdateEvent();

private:
  agxData::Value* m_value;
};

class AGXCORE_EXPORT BindEvent : public agx::Object::Event
{
public:
  BindEvent(agxData::Value* value, agxData::Value* binding, agxData::Value* oldBinding);

  agxData::Value* getValue();
  agxData::Value* getBinding();
  agxData::Value* getOldBinding();

protected:
  virtual ~BindEvent();

private:
  agxData::Value* m_value;
  agxData::Value* m_binding;
  agxData::Value* m_oldBinding;
};


DOXYGEN_END_INTERNAL_BLOCK()
#endif // GENERATED_AGX_VALUE_H_DECLARATION
