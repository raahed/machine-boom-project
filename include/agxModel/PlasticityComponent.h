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

#include <agxModel/Deformable1DComponent.h>

namespace agxModel
{
  /**
  Deformable1DComponent that adds plasticity to the deformable. Forcing a
  deformation with a force or torque larger than the deformable's bulk material's
  yield point cases permanent deformation of the deformable.
  */
  class AGXMODEL_EXPORT PlasticityComponent : public Deformable1DComponent
  {
    public:
      virtual void pre(Deformable1D& deformable) override;
      virtual void post(Deformable1D& deformable) override;

      AGXSTREAM_DECLARE_SERIALIZABLE(agxModel::PlasticityComponent);

    protected:
      virtual ~PlasticityComponent() {}
  };
}
