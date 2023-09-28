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

#ifndef AGXGL_UTIL_H
#define AGXGL_UTIL_H

#include <agx/Vec3.h>
#include <agx/Line.h>
#include <agx/AffineMatrix4x4.h>

namespace agxGL
{

  agx::UInt AGXPHYSICS_EXPORT generateBoxLines(agx::Vec3f min, agx::Vec3f max, agx::Line32* lines, agx::UInt lineIndex);


  agx::UInt AGXPHYSICS_EXPORT generateBoxLines( const agx::Vec3f& min, const agx::Vec3f& max, const agx::AffineMatrix4x4f& transform,
    agx::Line32* lines, agx::UInt lineIndex );

}


#endif /* AGXGL_UTIL_H */
