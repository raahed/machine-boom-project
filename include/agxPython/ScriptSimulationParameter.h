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

#include <agx/config/AGX_USE_PYTHON.h>
#include <agx/config.h>

#if AGX_USE_PYTHON()

namespace agxSDK
{
  class Simulation;
}

#include <agxPython/export.h>
#include <agx/Referenced.h>
#include <agx/ref_ptr.h>
#include <agxSDK/SimulationParameter.h>
#include <agxPython/ScriptContext.h>

namespace agxPython
{

  //class ScriptContext;

  class AGXPYTHON_EXPORT ScriptSimulationParameter : public agxSDK::SimulationParameterT<agx::Real>
  {
  public:
    ScriptSimulationParameter(const agx::Name& name, ScriptContext *scriptContext);

    virtual ~ScriptSimulationParameter();

  private:

    ScriptSimulationParameter();

    ScriptContext *m_scriptContext;
  };
}


#endif
