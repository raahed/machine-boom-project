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
#ifndef AGX_QUAT_H
#define AGX_QUAT_H

#include <agx/QuatTemplate.h>
#include <agxData/Type.h>

namespace agx
{
  typedef QuatT<Real> Quat;

  typedef QuatT<Real32> Quat32;
  typedef QuatT<Real64> Quat64;
}

AGX_TYPE_BINDING(agx::Quat64, "Quat")
AGX_TYPE_BINDING(agx::Quat32, "Quat")

#endif
