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

#ifndef AGXMODEL_TIRE_H
#define AGXMODEL_TIRE_H

#include <agxModel/export.h>

#include <agxSDK/Assembly.h>


namespace agxModel
{

  AGX_DECLARE_POINTER_TYPES(Tire);
  /**
  \note: This is an experimental feature, not fully tested yet.

  A simple base class for tire models.
  Not for direct use; use any of the derived classes
  instead.
  */
  class AGXMODEL_EXPORT Tire : public agxSDK::Assembly
  {
    public:

      /// Gets radius.
      agx::Real getRadius() const;

      /// Gets the current loaded radius.
      virtual agx::Real getLoadedRadius() const;

    protected:
      Tire();
      virtual ~Tire();

    protected:
      agx::Real m_radius;
  };
}

#endif // AGXMODEL_TIRE_H
