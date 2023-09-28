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

#ifndef AGXIO_KEYEVENT_H
#define AGXIO_KEYEVENT_H


#include <agxIO/Events.h>
#include <agx/Object.h>
#include <agx/Task.h>

namespace agxIO
{
  AGX_DECLARE_POINTER_TYPES(KeyEvent);
  class AGXCORE_EXPORT KeyEvent : public agx::Object
  {
  public:
    static KeyEvent *load(agx::TiXmlElement *eKey, agx::Device *device);
    virtual void configure(agx::TiXmlElement *eKey) override;

  public:
    agxData::Val<bool> downState;
    agxData::Val<bool> toggleState;

  public:
    KeyEvent(const agx::String& key, const agx::String& scope = "");

    void addTrigger(agx::Task *task);
    void removeTrigger(agx::Task *task);

    virtual void rebind() override;
    virtual agx::Object *getResourceImpl(const agx::Path& path, agx::Model *model) override;
  protected:
    virtual ~KeyEvent();
    void keyCallback(int key, agxIO::EventType type);

  private:
    agx::String m_scope;
    InputEvent::CallbackType m_keyCallback;
    agx::TaskRefVector m_tasks;
  };

}

#endif /* _AGXIO_KEYEVENT_H_ */
