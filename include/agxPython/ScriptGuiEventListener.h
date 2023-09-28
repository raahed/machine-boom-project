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

#ifndef AGXPYTHON_SCRIPTGUIEVENTLISTENER_H
#define AGXPYTHON_SCRIPTGUIEVENTLISTENER_H


#include <agx/config/AGX_USE_PYTHON.h>
#include <agx/config.h>

#if 0
#if AGX_USE_PYTHON()

#include <agx/PushDisableWarnings.h> // Disabling warnings. Include agx/PopDisableWarnings.h below!
#include <Python.h>
#include <agx/PopDisableWarnings.h> // End of disabled warnings.

namespace agxSDK
{
  class Simulation;
}

#include <agxPython/export.h>
#include <agx/Referenced.h>
#include <agx/ref_ptr.h>
#include <agxSDK/GuiEventListener.h>

namespace agxPython
{

  class ScriptContext;

  class AGXPYTHON_EXPORT GuiEvent
  {

  public:

    friend class ScriptGuiEventListener;

    enum GuiEventType
    {
      UPDATE = 0x1,
      MOUSE = 0x2,
      KEYBOARD = 0x4,
    };

    bool isUpdate() const { return (m_type == UPDATE); }

    bool isKeyboard() const { return (m_type == KEYBOARD); }

    bool isMouse() const { return (m_type == MOUSE); }

    bool isMouseDragged() const { return (isMouse() && (m_mouseState & agxSDK::GuiEventListener::MouseState::MOUSE_DOWN) == agxSDK::GuiEventListener::MouseState::MOUSE_DOWN); }

    bool isMouseMoved() const { return (!isMouseDragged() && (m_mouseState & agxSDK::GuiEventListener::MouseState::NO_MOUSE_STATE) == agxSDK::GuiEventListener::MouseState::NO_MOUSE_STATE); }

    float x() const { return m_x; }

    float y() const { return m_y; }

    int mouseButton() const { return m_mouseButtonMask; }

    int mouseState() const { return m_mouseState; }

    int key() const { return m_key; }

    unsigned int modKey() const { return m_modKeyMask; }

    bool keyIsDown() const { return m_keyDown; }

  protected:

    GuiEvent(float x, float y) : m_type(UPDATE), m_x(x), m_y(y) {}
    GuiEvent(int key, int modKeyMask, bool keyDown, float x, float y) : m_type(KEYBOARD), m_x(x), m_y(y), m_key(key), m_modKeyMask(modKeyMask), m_keyDown(keyDown) {}
    GuiEvent(agxSDK::GuiEventListener::MouseButtonMask button, agxSDK::GuiEventListener::MouseState state, float x, float y) : m_type(MOUSE), m_x(x), m_y(y), m_mouseButtonMask(button), m_mouseState(state) {}

    PyObject *makeDict();

  private:

    GuiEventType m_type;

    float m_x;
    float m_y;

    int m_mouseButtonMask;
    int m_mouseState;

    int m_key;
    unsigned int m_modKeyMask;
    bool m_keyDown;
  };

  class AGXPYTHON_EXPORT ScriptGuiEventListener : public agxSDK::GuiEventListener
  {

  public:

    friend class ScriptContext_listeners;

#ifndef SWIG
    virtual bool mouseDragged(MouseButtonMask /*buttonMask*/, float /*x*/, float /*y*/) override;

    virtual bool mouseMoved(float /*x*/, float /*y*/) override;

    virtual bool mouse(MouseButtonMask /*button*/, MouseState /*state*/, float /*x*/, float /*y*/) override;

    virtual bool keyboard(int /*key*/, unsigned int /*modKeyMask*/, float /*x*/, float /*y*/, bool /*keydown*/) override;

    virtual void update(float /*x*/, float /*y*/) override;
#endif

    void onEvent(PyObject *callback);

  protected:

    ScriptGuiEventListener(ScriptContext *scriptContext);

    virtual ~ScriptGuiEventListener();

    void callbackSet(PyObject *pyCallback);

  private:

    ScriptGuiEventListener();

    ScriptContext *m_scriptContext;

    PyObject *m_pyCallback;

  };

}


#endif

#endif

#endif
