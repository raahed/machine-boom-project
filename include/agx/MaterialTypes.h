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

#ifndef AGX_MATERIALTYPES_H
#define AGX_MATERIALTYPES_H

#include <agxData/Type.h>

namespace agx
{
  struct SimpleContactMaterial
  {
    SimpleContactMaterial();
    SimpleContactMaterial(Real friction_, Real restitution_);
    Real friction;
    Real restitution;
  };

  /* Implementation */
  inline SimpleContactMaterial::SimpleContactMaterial() {}
  inline SimpleContactMaterial::SimpleContactMaterial(Real friction_, Real restitution_)
    : friction(friction_)
    , restitution(restitution_)
  {
  }

  AGX_FORCE_INLINE std::ostream& operator<<( std::ostream& stream, const SimpleContactMaterial& /*material*/ )
  {
    stream << "TODO Implement ostream << SimpleContactMaterial" << std::endl;
    return stream;
  }
}


AGX_TYPE_BINDING(agx::SimpleContactMaterial, "ContactMaterial")

#endif /* AGX_MATERIALTYPES_H */
