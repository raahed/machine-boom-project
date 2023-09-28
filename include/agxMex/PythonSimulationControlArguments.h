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

#ifndef AGXMEX_PYTHON_SIMULATION_CONTROL_ARGUMENTS_H
#define AGXMEX_PYTHON_SIMULATION_CONTROL_ARGUMENTS_H

#include <agx/config/AGX_USE_PYTHON.h>
#include <agx/config/AGX_USE_AGXMEX.h>
#if AGX_USE_PYTHON() && AGX_USE_AGXMEX()

#include <agxMex/export.h>
#include <agx/String.h>
#include <agx/Referenced.h>
#include <agxSDK/SimulationControl.h>

namespace agxMex
{


  AGX_DECLARE_POINTER_TYPES(PythonControlArgument);
  /// Class handling input arguments from python.
  class AGXMEX_EXPORT PythonControlArgument : public agxSDK::SimulationControlArgument
  {

  public:
    PythonControlArgument(size_t numInput, size_t numOutput);

    virtual bool input(const agxSDK::Simulation * /*sim*/, const agx::RealVector * /*input*/) const { return false; }
    virtual bool output(const agxSDK::Simulation * /*sim*/, const agx::RealVector * /*output*/) const { return false; }

  protected:
    PythonControlArgument(const PythonControlArgument&) = delete;

    virtual bool applyInput(const agxSDK::Simulation* simulation, const agx::RealVector& inputData) const override;
    virtual bool obtainOutput(const agxSDK::Simulation* simulation, const agx::RealVector& outputData) const override;

  private:

  };

}

#endif // AGX_USE_PYTHON

#endif // AGXMEX_PYTHON_SIMULATION_CONTROL_ARGUMENTS_H
