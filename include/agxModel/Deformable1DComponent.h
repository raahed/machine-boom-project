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

#include <agxModel/export.h>

#include <agxStream/Serializable.h>


namespace agxModel
{
  class Deformable1D;
}

namespace agxModel
{
  AGX_DECLARE_POINTER_TYPES(Deformable1DComponent);
  AGX_DECLARE_VECTOR_TYPES(Deformable1DComponent);

  /**
  Base class for all classes that add behavior to or inspect a Deformable1D.
  The pre and post callbacks are called by the Deformable1D during the simulation
  process.
  */
  class AGXMODEL_EXPORT Deformable1DComponent : public agx::Referenced, public agxStream::Serializable
  {
    public:
      virtual void pre(agxModel::Deformable1D& deformable) = 0;
      virtual void post(agxModel::Deformable1D& deformable) = 0;

      AGXSTREAM_DECLARE_ABSTRACT_SERIALIZABLE(agxModel::Deformable1DComponent);
      virtual void store(agxStream::OutputArchive& /*out*/) const override {}
      virtual void restore(agxStream::InputArchive& /*in*/) override {}

    protected:
      virtual ~Deformable1DComponent() {}
  };
}
