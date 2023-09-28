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

#ifndef AGXGL_PRIMITIVE_RENDERER_H
#define AGXGL_PRIMITIVE_RENDERER_H

#include <agx/Math.h>
#include <agx/Vec4.h>
#include <agx/AffineMatrix4x4.h>
#include <agx/agxPhysics_export.h>

namespace agxGL
{
  void AGXPHYSICS_EXPORT renderSphere(const agx::AffineMatrix4x4& transform, agx::Real radius, const agx::Vec4& color = agx::Vec4(1));
  void AGXPHYSICS_EXPORT renderBox(const agx::AffineMatrix4x4& transform, const agx::Vec3& halfLengths, const agx::Vec4& color = agx::Vec4(1));
  void AGXPHYSICS_EXPORT renderCylinder(const agx::AffineMatrix4x4& transform, agx::Real radius, agx::Real height, const agx::Vec4& color = agx::Vec4(1));
  void AGXPHYSICS_EXPORT renderCapsule(const agx::AffineMatrix4x4& transform, agx::Real radius, agx::Real height, const agx::Vec4& color = agx::Vec4(1));
  void AGXPHYSICS_EXPORT renderLine(const agx::AffineMatrix4x4& transform, const agx::Vec3& start, const agx::Vec3& end, const agx::Vec4& color = agx::Vec4(1));
  void AGXPHYSICS_EXPORT renderPlane(const agx::AffineMatrix4x4& trans, const agx::Vec3& normal, agx::Real distance, const agx::Vec4& color);

  void AGXPHYSICS_EXPORT renderSphere(agx::Real radius, const agx::Vec4& color = agx::Vec4(1));
  void AGXPHYSICS_EXPORT renderBox(const agx::Vec3& halfLengths, const agx::Vec4& color = agx::Vec4(1));
  void AGXPHYSICS_EXPORT renderBox(const agx::Vec3& min, const agx::Vec3& max, const agx::Vec4& color = agx::Vec4(1));
  void AGXPHYSICS_EXPORT renderCylinder(agx::Real radius, agx::Real height, const agx::Vec4& color = agx::Vec4(1));
  void AGXPHYSICS_EXPORT renderCapsule(agx::Real radius, agx::Real height, const agx::Vec4& color = agx::Vec4(1));
  void AGXPHYSICS_EXPORT renderLine(const agx::Vec3& start, const agx::Vec3& end, const agx::Vec4& color = agx::Vec4(1));
  // void AGXPHYSICS_EXPORT renderMesh(agx::Vec3Vector& vertices, agx::UInt32Vector indices, const agx::Vec4& color = agx::Vec4(1));
}

#endif // #ifndef  AGXGL_PRIMITIVE_RENDERER_H
