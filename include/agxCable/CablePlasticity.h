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


#include <agxSDK/LinkedStructureComponent.h>

#include <agxCable/export.h>
#include <agxCable/Direction.h>
#include <agxCable/DeformationData.h>

namespace agxCable
{
  AGX_DECLARE_POINTER_TYPES(CablePlasticity);

  class AGXCABLE_EXPORT CablePlasticity : public agxSDK::LinkedStructureComponent
  {
    public:
      CablePlasticity();

      void setYieldPoint(agx::Real yieldPoint, agxCable::Direction direction);
      agx::Real getYieldPoint(agxCable::Direction direction) const;

      AGXSTREAM_DECLARE_SERIALIZABLE(agxCable::CablePlasticity);

    DOXYGEN_START_INTERNAL_BLOCK()
    public:
      CablePlasticity(const agx::Real (& yieldPoints)[ALL_DIRECTIONS],
                      agx::Vector<agxCable::DeformationData>&& deformations);

      virtual void onPreStep() override;
      virtual void onPostStep() override;
    DOXYGEN_END_INTERNAL_BLOCK()

    protected:
      virtual ~CablePlasticity();

    private:
      agx::Real m_yieldPoints[ALL_DIRECTIONS];
      agx::Vector<agxCable::DeformationData> m_deformations;
  };
}
