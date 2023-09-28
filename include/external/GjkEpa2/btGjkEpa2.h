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

/*
This source code has been taken and modified by Algoryx Simulation AB
from the source and under the license given below.
*/

/*
Bullet Continuous Collision Detection and Physics Library
Copyright (c) 2003-2008 Erwin Coumans  http://continuousphysics.com/Bullet/

This software is provided 'as-is', without any express or implied warranty.
In no event will the authors be held liable for any damages arising from the
use of this software.
Permission is granted to anyone to use this software for any purpose,
including commercial applications, and to alter it and redistribute it
freely,
subject to the following restrictions:

1. The origin of this software must not be misrepresented; you must not
claim that you wrote the original software. If you use this software in a
product, an acknowledgment in the product documentation would be appreciated
but is not required.
2. Altered source versions must be plainly marked as such, and must not be
misrepresented as being the original software.
3. This notice may not be removed or altered from any source distribution.
*/

/*
GJK-EPA collision solver by Nathanael Presson, 2008
*/
#ifndef BT_GJK_EPA2_H
#define BT_GJK_EPA2_H

#include <agx/macros.h>

DOXYGEN_START_INTERNAL_BLOCK()

//#include <agxCollide/bullet/LinearMath/btVector3.h>
#include <agx/Vec3.h>
#include <agx/AffineMatrix4x4.h>

namespace agxCollide
{
  class InitGjkEpa {
  public:
    friend class Space;
    static void init();
  };

  class Shape;
}

/// btGjkEpaSolver contributed under zlib by Nathanael Presson
struct  btGjkEpaSolver2
{
  struct  GjkEpaResults
  {
    enum eStatus
    {
      Separated,    /* Shapes do not penetrate                        */
      Penetrating,  /* Shapes are penetrating                        */
      GJK_Failed,    /* GJK phase fail, no big issue, shapes are probably just 'touching'  */
      EPA_Failed    /* EPA phase fail, bigger problem, need to save parameters, and debug  */
    }    status;
    agx::Vec3  witnesses[2];
    agx::Vec3  normal;
    agx::Real  distance;
  };

  AGXPHYSICS_EXPORT static bool  distance(  const agxCollide::Shape* shape0, const agx::AffineMatrix4x4& transform0,
    const agxCollide::Shape* shape1, const agx::AffineMatrix4x4& transform1,
    const agx::Vec3& guess, GjkEpaResults& results);

  AGXPHYSICS_EXPORT static bool penetration(const agxCollide::Shape* shape0, const agx::AffineMatrix4x4& transform0,
    const agxCollide::Shape* shape1, const agx::AffineMatrix4x4& transform1,
    const agx::Vec3& guess, GjkEpaResults& results, const agx::Real epaAccuracy = agx::Real(1e-6));
};

DOXYGEN_END_INTERNAL_BLOCK()


#endif //BT_GJK_EPA2_H

