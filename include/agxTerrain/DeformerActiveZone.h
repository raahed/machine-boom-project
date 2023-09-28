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

#include <agx/config/AGX_USE_AGXTERRAIN.h>

#include <agxTerrain/ActiveZone.h>


namespace agxTerrain
{

  class DeformerCollection;

  AGX_DECLARE_POINTER_TYPES(DeformerActiveZone);

  class AGXTERRAIN_EXPORT DeformerActiveZone : public ActiveZone
  {
  public:
    DeformerActiveZone(agx::Frame* parentFrame, DeformerCollection* deformer);

    ActiveZone::Type getType() const override;

    void onPreCollide(Terrain* terrain, const agx::Line& cuttingEdge, const agx::Line& topEdge, const agx::Vec3& forwardVector) override;

    void onPre(Terrain* terrain, const agx::Line& cuttingEdge, const agx::Line& topEdge, const agx::Vec3& forwardVector) override;

    void onPost(Terrain* terrain, const agx::Line& cuttingEdge, const agx::Line& topEdge, const agx::Vec3& forwardVector) override;

  protected:
    virtual ~DeformerActiveZone();

  private:

    void calculateProjectedVectors(Terrain* terrain,
                                   const agx::Line& cuttingEdge,
                                   const agx::Line& topEdge,
                                   const agx::Vec3& forwardVector,
                                   agx::Line& cuttingEdgeWorld,
                                   agx::Line& topEdgeWorld);

  private:
    agx::Vec3 m_projectedForwardVector;
    agx::Vec3 m_flatForwardVector;

    DeformerCollection* m_deformer;
  };
}