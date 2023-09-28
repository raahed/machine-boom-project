/*
Copyright 2007-2023. Algoryx Simulation AB.

All AGX source code, intellectual property, documentation, sample code,
tutorials, scene files and technical white papers, is copyrighted, proprietary
and confidential material of Algoryx Simulation AB. You may not download, read,
store, distribute, publish, copy or otherwise disseminate, use or expose this
material unless having a written signed agreement with Algoryx Simulation AB, or having been
advised so by Algoryx Simulation AB for a time limited evaluation, or having purchased a
valid commercial license from Algoryx Simulation AB.

Algoryx Simulation AB disclaims all responsibilities for loss or damage caused
from using this software, unless otherwise stated in written agreements with
Algoryx Simulation AB.
*/

#ifndef AGX_LINE_H
#define AGX_LINE_H

#include <agx/Vec3.h>

namespace agx
{
  template <typename T>
  class LineT
  {
  public:
    typedef T Type;

  public:
    T p1;
    T p2;

  public:
    inline LineT() { }
    inline LineT(const T& _p1, const T& _p2) : p1(_p1), p2(_p2) {}
  };

  typedef LineT<Vec3> Line;
  typedef LineT<Vec3f> Line32;
  typedef LineT<Vec3d> Line64;


  template <typename T>
  std::ostream& operator<<(std::ostream& output, const LineT<T>& line)
  {
    output << line.p1 << "->" << line.p2;
    return output;
  }
}

AGX_TYPE_BINDING(agx::Line32, "Line")
AGX_TYPE_BINDING(agx::Line64, "Line")

#endif


