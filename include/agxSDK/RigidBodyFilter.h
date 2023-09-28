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

#ifndef AGXSDK_RIGIDBODY_FILTER_H
#define AGXSDK_RIGIDBODY_FILTER_H

#include <agx/agxPhysics_export.h>
#include <agx/stdint.h>
#include <agx/Referenced.h>
#include <agxCollide/GeometryPair.h>
#include <agxCollide/Contacts.h>
#include <agxSDK/ExecuteFilter.h>
#include <agxSDK/GeometryFilter.h>
#include <agx/RigidBody.h>


namespace agxSDK
{

  template <>
  inline bool matchFilter(const agxCollide::Geometry* geo, const agx::RigidBody* rb)
  {
    const bool matches = (!rb || geo->getRigidBody() == rb);
    return matches;
  }

  // RigidBodyFilter.
  template class ExecuteFilterT<agx::RigidBody, agx::RigidBody>;
  typedef ExecuteFilterT<agx::RigidBody, agx::RigidBody> RigidBodyFilter;
  typedef agx::ref_ptr<RigidBodyFilter> RigidBodyFilterRef;


  // RigidBodyGeometry  Filter.
  template class ExecuteFilterT<agx::RigidBody, agxCollide::Geometry>;
  typedef ExecuteFilterT<agx::RigidBody, agxCollide::Geometry> RigidBodyGeometryFilter;
  typedef agx::ref_ptr<RigidBodyGeometryFilter> RigidBodyGeometryFilterRef;




} // namespace agxSDK

#endif


