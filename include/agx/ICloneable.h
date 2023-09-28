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

#ifndef AGX_CLONEABLE_H
#define AGX_CLONEABLE_H

#include <agx/Referenced.h>

namespace agx
{
  AGX_DECLARE_POINTER_TYPES( ICloneable );

  /**
  Interface class for a cloneable reference counted object.
  */
  class AGXPHYSICS_EXPORT ICloneable : public agx::Referenced
  {
    public:
      /**
      Creates a clone of this object. If not null, \p child is the pointer to the clone
      to be returned, but has been instantiated in a child class implementation.
      \param child - pointer to clone, if already created
      \return a clone of this object
      */
      virtual agx::ICloneableRef clone( agx::ICloneable* child = nullptr ) = 0;
  };
}

#endif
