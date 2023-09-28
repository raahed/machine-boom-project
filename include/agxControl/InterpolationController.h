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


/// \cond CONTROL

#ifndef AGXCONTROL_INTERPOLATION_CONTROLLER_H
#define AGXCONTROL_INTERPOLATION_CONTROLLER_H

#include <agx/config.h>

#include <agx/Referenced.h>
#include <agxStream/Serializable.h>

namespace agxControl
{
  AGX_DECLARE_POINTER_TYPES(InterpolationController);


  class AGXPHYSICS_EXPORT InterpolationController : public agx::Referenced, public agxStream::Serializable
  {
  public:
    virtual agx::Real interpolate(agx::Real startValue, agx::Real endValue, agx::Real startTime, agx::Real endTime, agx::Real time) = 0;

    AGXSTREAM_DECLARE_ABSTRACT_SERIALIZABLE( agxControl::InterpolationController );

  protected:
    virtual ~InterpolationController() {}
  };


  AGX_DECLARE_POINTER_TYPES(LinearInterpolationController);
  class AGXPHYSICS_EXPORT LinearInterpolationController : public agxControl::InterpolationController
  {
  public:
    virtual agx::Real interpolate(agx::Real startValue, agx::Real endValue, agx::Real startTime, agx::Real endTime, agx::Real time) override;
    AGXSTREAM_DECLARE_SERIALIZABLE( agxControl::LinearInterpolationController );

  protected:
    virtual ~LinearInterpolationController() {}
  };

  AGX_DECLARE_POINTER_TYPES(HermiteInterpolationController);
  class AGXPHYSICS_EXPORT HermiteInterpolationController : public agxControl::InterpolationController
  {
  public:
    virtual agx::Real interpolate(agx::Real startValue, agx::Real endValue, agx::Real startTime, agx::Real endTime, agx::Real time) override;
    agx::Real interpolate(agx::Real startValue, agx::Real endValue, agx::Real startTime, agx::Real endTime, agx::Real time, agx::Real startSlope, agx::Real endSlope);
    AGXSTREAM_DECLARE_SERIALIZABLE( agxControl::HermiteInterpolationController );

  protected:
    virtual ~HermiteInterpolationController() {}
  };
}

// Include guard
#endif

/// \endcond CONTROL

