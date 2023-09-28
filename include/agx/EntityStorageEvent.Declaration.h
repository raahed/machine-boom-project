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

#ifndef GENERATED_AGX_ENTITYSTORAGE_H_DECLARATION
#define GENERATED_AGX_ENTITYSTORAGE_H_DECLARATION

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

  virtual void destroyCallback(agxData::EntityStorage* storage);
  virtual void createInstanceCallback(agxData::EntityStorage* storage, agxData::EntityPtr instance);
  virtual void createInstancesCallback(agxData::EntityStorage* storage, agxData::EntityRange range);
  virtual void createInstancesCallback(agxData::EntityStorage* storage, agxData::Array< agxData::EntityPtr > instances);
  virtual void destroyInstanceCallback(agxData::EntityStorage* storage, agxData::EntityPtr instance);
  virtual void destroyInstancesCallback(agxData::EntityStorage* storage, agxData::EntityRange range);
  virtual void destroyInstancesCallback(agxData::EntityStorage* storage, agxData::Array< agxData::EntityPtr > instances);
  virtual void permuteCallback(agxData::EntityStorage* storage, agxData::Array< agx::Index > permutation);

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
  void triggerDestroyEvent(agxData::EntityStorage* storage);
  void triggerCreateInstanceEvent(agxData::EntityStorage* storage, agxData::EntityPtr instance);
  void triggerCreateInstancesEvent(agxData::EntityStorage* storage, agxData::EntityRange range);
  void triggerCreateInstancesEvent(agxData::EntityStorage* storage, agxData::Array< agxData::EntityPtr > instances);
  void triggerDestroyInstanceEvent(agxData::EntityStorage* storage, agxData::EntityPtr instance);
  void triggerDestroyInstancesEvent(agxData::EntityStorage* storage, agxData::EntityRange range);
  void triggerDestroyInstancesEvent(agxData::EntityStorage* storage, agxData::Array< agxData::EntityPtr > instances);
  void triggerPermuteEvent(agxData::EntityStorage* storage, agxData::Array< agx::Index > permutation);

private:
  EventListenerPtrVector m_listeners;
  agx::Thread *m_activeThread;
  agx::UInt32 m_iterationIndex;
};

class AGXCORE_EXPORT DestroyEvent : public agx::Object::Event
{
public:
  DestroyEvent(agxData::EntityStorage* storage);

  agxData::EntityStorage* getStorage();

protected:
  virtual ~DestroyEvent();

private:
  agxData::EntityStorage* m_storage;
};

class AGXCORE_EXPORT CreateInstanceEvent : public agx::Object::Event
{
public:
  CreateInstanceEvent(agxData::EntityStorage* storage, agxData::EntityPtr instance);

  agxData::EntityStorage* getStorage();
  agxData::EntityPtr getInstance();

protected:
  virtual ~CreateInstanceEvent();

private:
  agxData::EntityStorage* m_storage;
  agxData::EntityPtr m_instance;
};

class AGXCORE_EXPORT CreateInstancesEvent : public agx::Object::Event
{
public:
  CreateInstancesEvent(agxData::EntityStorage* storage, agxData::EntityRange range);

  agxData::EntityStorage* getStorage();
  agxData::EntityRange getRange();

protected:
  virtual ~CreateInstancesEvent();

private:
  agxData::EntityStorage* m_storage;
  agxData::EntityRange m_range;
};

class AGXCORE_EXPORT CreateInstancesEvent2 : public agx::Object::Event
{
public:
  CreateInstancesEvent2(agxData::EntityStorage* storage, agxData::Array< agxData::EntityPtr > instances);

  agxData::EntityStorage* getStorage();
  agxData::Array< agxData::EntityPtr > getInstances();

protected:
  virtual ~CreateInstancesEvent2();

private:
  agxData::EntityStorage* m_storage;
  agxData::Array< agxData::EntityPtr > m_instances;
};

class AGXCORE_EXPORT DestroyInstanceEvent : public agx::Object::Event
{
public:
  DestroyInstanceEvent(agxData::EntityStorage* storage, agxData::EntityPtr instance);

  agxData::EntityStorage* getStorage();
  agxData::EntityPtr getInstance();

protected:
  virtual ~DestroyInstanceEvent();

private:
  agxData::EntityStorage* m_storage;
  agxData::EntityPtr m_instance;
};

class AGXCORE_EXPORT DestroyInstancesEvent : public agx::Object::Event
{
public:
  DestroyInstancesEvent(agxData::EntityStorage* storage, agxData::EntityRange range);

  agxData::EntityStorage* getStorage();
  agxData::EntityRange getRange();

protected:
  virtual ~DestroyInstancesEvent();

private:
  agxData::EntityStorage* m_storage;
  agxData::EntityRange m_range;
};

class AGXCORE_EXPORT DestroyInstancesEvent2 : public agx::Object::Event
{
public:
  DestroyInstancesEvent2(agxData::EntityStorage* storage, agxData::Array< agxData::EntityPtr > instances);

  agxData::EntityStorage* getStorage();
  agxData::Array< agxData::EntityPtr > getInstances();

protected:
  virtual ~DestroyInstancesEvent2();

private:
  agxData::EntityStorage* m_storage;
  agxData::Array< agxData::EntityPtr > m_instances;
};

class AGXCORE_EXPORT PermuteEvent : public agx::Object::Event
{
public:
  PermuteEvent(agxData::EntityStorage* storage, agxData::Array< agx::Index > permutation);

  agxData::EntityStorage* getStorage();
  agxData::Array< agx::Index > getPermutation();

protected:
  virtual ~PermuteEvent();

private:
  agxData::EntityStorage* m_storage;
  agxData::Array< agx::Index > m_permutation;
};


DOXYGEN_END_INTERNAL_BLOCK()
#endif // GENERATED_AGX_ENTITYSTORAGE_H_DECLARATION
