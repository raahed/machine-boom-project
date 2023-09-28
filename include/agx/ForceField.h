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

#ifndef AGX_FORCEFIELD_H
#define AGX_FORCEFIELD_H

#include <agx/Interaction.h>

namespace agx
{

  class DynamicsSystem;

  /**
  Fields produce force on physical bodies which have the appropriate charge.

  A field produces a force on a physical body according to a function
  which considers the charge registered for each body.
  */
  class AGXPHYSICS_EXPORT ForceField : public Interaction
  {
    public:
      ForceField();

      /// Copy constructor
      ForceField( const ForceField& );

    protected:

      /**
      ForceField objects must be removed from the DynamicsSystem
      they belong when they are deleted.
      */
      virtual ~ForceField();

    private:
  };
} //namespace agx

#endif
