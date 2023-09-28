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

#include <agx/Vec3.h>
#include <agx/Quat.h>


namespace agx
{
  class LockJoint;
}



namespace agxStream
{
  class InputArchive;
  class OutputArchive;
}



namespace agxCable
{
  struct DeformationData
  {
    agx::Vec3 translationalForce;
    agx::Vec3 rotationalForce;
    agx::Quat relativeRotation;

    DeformationData();
    DeformationData(const agx::Vec3& translationalForce, const agx::Vec3& rotationalForce,
                    const agx::Quat& relativeRotation);

    void update(const agx::LockJoint& lock, const agx::Quat& previousRotation, const agx::Quat& currentRotation);

    void store(agxStream::OutputArchive& out) const;
    void restore(agxStream::InputArchive& in);
  };
}
