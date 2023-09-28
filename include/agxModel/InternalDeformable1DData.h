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

#include <agx/Referenced.h>

namespace agxModel
{
  class Deformable1D;

  AGX_DECLARE_POINTER_TYPES(InternalDeformable1DData);

  /**
  Bridge between the InternalData system and the Deformable1D. Cannot store the
  Deformable1D directly because of virtual inheritance between Deformable1D and
  Referenced.
  */
  class InternalDeformable1DData : public agx::Referenced
  {
    public:
      explicit InternalDeformable1DData(agxModel::Deformable1D* deformable)
          : m_deformable(deformable)
      {
      }

      agxModel::Deformable1D* getDeformable()
      {
        return m_deformable;
      }

    private:
      agxModel::Deformable1D* m_deformable;
  };
}
