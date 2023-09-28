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

#include <agx/agxPhysics_export.h>
#include <agx/Referenced.h>
#include <agxSDK/ExecuteFilter.h>
#include <agxSDK/GeometryFilter.h>
#include <agxSDK/RigidBodyFilter.h>
#include <agxSDK/Assembly.h>

namespace agxSDK
{

  template <>
  inline bool matchFilter(const agxCollide::Geometry* geo, const agxSDK::Assembly* a)
  {
    const bool matches = (!a || a->getGeometries().contains(geo));
    return matches;
  }

  /** AssemblyFilter.
  Note that it only includes geometries explicitly added to the assembly,
  (as in agxSDK::Assembly::getGeometries()).
  */
  template class ExecuteFilterT<agxSDK::Assembly, agxSDK::Assembly>;
  typedef ExecuteFilterT<agxSDK::Assembly, agxSDK::Assembly> AssemblyFilter;
  typedef agx::ref_ptr<AssemblyFilter> AssemblyFilterRef;

  /** AssemblyGeometryFilter.
  Note that it on the assembly-side only includes geometries explicitly added to the assembly,
  (as in agxSDK::Assembly::getGeometries()).
  */
  template class ExecuteFilterT<agxSDK::Assembly, agxCollide::Geometry>;
  typedef ExecuteFilterT<agxSDK::Assembly, agxCollide::Geometry> AssemblyGeometryFilter;
  typedef agx::ref_ptr<AssemblyGeometryFilter> AssemblyGeometryFilterRef;

  /** AssemblyRigidBodyFilter.
  Note that it on the assembly-side only includes geometries explicitly added to the assembly,
  (as in agxSDK::Assembly::getGeometries()).
  */
  template class ExecuteFilterT<agxSDK::Assembly, agx::RigidBody>;
  typedef ExecuteFilterT<agxSDK::Assembly, agx::RigidBody> AssemblyRigidBodyFilter;
  typedef agx::ref_ptr<AssemblyRigidBodyFilter> AssemblyRigidBodyFilterRef;

} // namespace agxSDK

