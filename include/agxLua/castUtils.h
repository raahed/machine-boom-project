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

#ifndef AGXLUA_CAST_UTILS_H
#define AGXLUA_CAST_UTILS_H

#include <agx/Hinge.h>
#include <agx/CylindricalJoint.h>
#include <agx/Prismatic.h>

namespace agxLua
{

  // Dynamic cast is not available in tolua so we need this to do downward cast, specially for classes with multiple inheritance

  inline
    agx::Hinge* castSerializableToHinge( agxStream::Serializable* obj )
  {
    return dynamic_cast<agx::Hinge *>(obj);
  }

  inline
    agx::Prismatic* castSerializableToPrismatic( agxStream::Serializable* obj )
  {
    return dynamic_cast<agx::Prismatic *>(obj);
  }

  inline
    agx::CylindricalJoint* castSerializableToCylindricalJoint( agxStream::Serializable* obj )
  {
    return dynamic_cast<agx::CylindricalJoint *>(obj);
  }

}


#endif // AGXLUA_CAST_UTILS_H