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


#include <agx/TimeStamp.h>
#include <agx/Real.h>
#include <agx/Referenced.h>
#include <agxModel/export.h>

namespace agxModel
{

  AGX_DECLARE_POINTER_TYPES(Controller1D);

  /**
  A simple base class for a controller with a set point and an out signal.

  This class is not for direct use; use any of the derived classes instead, e.g. agxModel::PidController1D.
  */
  class AGXMODEL_EXPORT Controller1D : public virtual agx::Referenced
  {
    public:

      /// Update the measured Process Variable and calculate a new Manipulated Variable.
      /// The measuredProcessVariable is the Plant measured Process Variable, PV in a Control System
      /// and time is the current simulation time
      virtual void update(const agx::TimeStamp& /*time*/, agx::Real /*measuredProcessVariable*/) {};

      /// Reset all calculated values to initial conditions. This will keep the Set Point.
      virtual void reset() {};

      /// get the Manipulated Variable (MV)
      virtual agx::Real getManipulatedVariable() const { return 0; };

      /// Set the current Set Point for the controller
      virtual void setSetPoint(agx::Real /*setPoint*/) { };

      /// Get the current Set Point for the controller
      virtual agx::Real getSetPoint() const { return 0; };

  };
}
