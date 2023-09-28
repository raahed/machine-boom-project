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



#include <agx/agxCore_export.h>
#include <agx/version.h>

#include <agx/String.h>
#include <agx/Vec2.h>
#include <agx/Vec3.h>
#include <agx/Quat.h>
#include <agx/Range.h>
#include <agx/Matrix3x3.h>
#include <agx/SPDMatrix3x3.h>



namespace agxStream
{
  namespace ElementTypes
  {
    enum Type {
      FLOAT, DOUBLE, VEC2F, VEC2D, VEC3F, VEC3D, VEC4F, VEC4D, QUATF, QUATD, OTHER
    };
    inline agx::UInt32 getElementType(const float&)
    {
      return FLOAT;
    }

    inline agx::UInt32 getElementType(const double&)
    {
      return DOUBLE;
    }

    template <typename T>
    agx::UInt32 getElementType(const agx::Vec2T<T>&)
    {
      if (sizeof(T) == sizeof(float))
        return VEC2F;
      else
        return VEC2D;
    }

    template <typename T>
    agx::UInt32 getElementType(const agx::Vec3T<T>&)
    {
      if (sizeof(T) == sizeof(float))
        return VEC3F;
      else
        return VEC3D;
    }

    template <typename T>
    agx::UInt32 getElementType(const agx::Vec4T<T>&)
    {
      if (sizeof(T) == sizeof(float))
        return VEC4F;
      else
        return VEC4D;
    }

    template <typename T>
    agx::UInt32 getElementType(const agx::QuatT<T>&)
    {
      if (sizeof(T) == sizeof(float))
        return QUATF;
      else
        return QUATD;
    }

    template <typename T>
    agx::UInt32 getElementType(const T&)
    {
      return OTHER;
    }
  }
}

