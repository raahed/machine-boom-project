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
#ifndef AGXPYTHON_SCRIPTIDE_H
#define AGXPYTHON_SCRIPTIDE_H 1

#include <agx/config/AGX_USE_PYTHON.h>

#if AGX_USE_PYTHON()

#include <agxPython/export.h>
#include <agxPython/ScriptIDEAutocomplete.h>
#include <agxPython/ScriptIDEInterpreter.h>

#include <agx/Referenced.h>
#include <agx/Name.h>

namespace agxPython
{

  AGX_DECLARE_POINTER_TYPES(ScriptIDE);
  class AGXPYTHON_EXPORT ScriptIDE : public agx::Referenced
  {

  public:

    friend class ScriptManager;

    void updateSourceCode(const agx::String& sourceCode);
    void queryAutoComplete(const agx::String& sourceCode, int caretLine, int caretColumn, bool readOnly, bool scopeChange);

    void setAutocomplete(ScriptIDEAutocomplete* autocomplete);

    ScriptIDEAutocomplete* getAutocomplete() const;

    void setInterpreter(ScriptIDEInterpreter* interpreter);

    ScriptIDEInterpreter* getInterpreter() const;

    const agx::Name& getIdentifyingName() const;

    const agx::String& getSourceCode() const;

    int getCaretX() const;

    int getCaretY() const;

  protected:
    ScriptIDE(const agx::Name& identifyingName);

    virtual ~ScriptIDE();

  private:

    agx::Name m_identifyingName;

    agx::String m_sourceCode;

    std::pair<int, int> m_caret;

    agx::ref_ptr<ScriptIDEAutocomplete> m_autocomplete;

    agx::ref_ptr<ScriptIDEInterpreter> m_interpreter;

  };

}

#endif

#endif
