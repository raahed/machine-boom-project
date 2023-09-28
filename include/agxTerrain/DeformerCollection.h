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

#include <agxTerrain/DeformerActiveZone.h>
#include <agxTerrain/SoilParticleAggregate.h>

namespace agxTerrain
{

  class TerrainToolCollection;

  AGX_DECLARE_POINTER_TYPES(DeformerCollection);

  class AGXTERRAIN_EXPORT DeformerCollection : public agx::Referenced
  {
  public:
    DeformerCollection(agx::Frame* parentFrame,
                       agx::Line cuttingEdge,
                       agx::Line topEdge,
                       agx::Vec3 forwardVector);

    void addNotification(agxSDK::Simulation* simulation, Terrain* terrain);

    void removeNotification(agxSDK::Simulation* simulation, Terrain* terrain);

    const agx::Line& getCuttingEdge() const;
    const agx::Line& getTopEdge() const;
    const agx::Vec3& getForwardVector() const;

    DeformerActiveZone* getActiveZone();

    SoilParticleAggregate* getAggregate();

    void setCuttingEdge(agx::Line cuttingEdge);
    void setTopEdge(agx::Line topEdge);
    void setForwardVector(agx::Vec3 forwardVector);

    void onPreCollide(TerrainToolCollection* collection);
    void onPre(TerrainToolCollection* collection);
    void onPost();

    void onEnableChanged(bool enable);

  protected:
    virtual ~DeformerCollection();

  private:

  private:
    agx::Line m_cuttingEdge;
    agx::Line m_topEdge;
    agx::Vec3 m_forwardVector;

    DeformerActiveZoneRef    m_activeZone;
    SoilParticleAggregateRef m_aggregate;
  };
}