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

#ifndef AGXMEX_LUA_SIMULATION_CONTROL_ARGUMENTS_H
#define AGXMEX_LUA_SIMULATION_CONTROL_ARGUMENTS_H

#include <agxMex/export.h>
#include <agx/String.h>
#include <agxSDK/SimulationControl.h>


namespace agxMex {

  AGX_DECLARE_POINTER_TYPES(LuaInputArgument);
  /// Class handling input arguments from lua.
  class AGXMEX_EXPORT LuaInputArgument : public agxSDK::SimulationControlArgument
  {
  public:
    LuaInputArgument(
      const agx::String& functionName,
      size_t numValues);

    agx::String getFunctionName() const;

    virtual bool applyInput(const agxSDK::Simulation* simulation, const agx::RealVector& input) const;

  protected:
    // Just to please the compiler, should never reach this
    virtual bool obtainOutput(const agxSDK::Simulation* simulation, const agx::RealVector& output) const;

    agx::String m_functionName;
  };


  AGX_DECLARE_POINTER_TYPES(LuaOutputArgument);
  /// Class handling output arguments from lua.
  class AGXMEX_EXPORT LuaOutputArgument : public agxSDK::SimulationControlArgument {
  public:
    LuaOutputArgument(
      const agx::String& functionName,
      size_t numValues);

    agx::String getFunctionName() const;

    virtual bool obtainOutput(const agxSDK::Simulation* simulation, const agx::RealVector& output) const;

  protected:

    // Just to please the compiler, should never reach this
    virtual bool applyInput(const agxSDK::Simulation* simulation, const agx::RealVector& input) const;

    agx::String m_functionName;
  };

}

#endif
