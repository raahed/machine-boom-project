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

#include <agxTerrain/export.h>
#include <agx/agx_vector_types.h>
#include <agx/Prismatic.h>
#include <agx/Line.h>

namespace agxTerrain
{
  class Shovel;
  class TerrainToolCollection;
  class TerrainMaterial;

  AGX_DECLARE_POINTER_TYPES(SoilPenetrationResistance);
  class AGXTERRAIN_EXPORT SoilPenetrationResistance : public agx::Referenced
  {
  public:

    enum class PenetrationModel
    {
      NO_MODEL,
      ELASTIC_PLASTIC_LIMIT,
      PLASTIC
    };

    SoilPenetrationResistance();

    /**
    \internal

    On addNotification of Terrain and Shovel.
    */
    void addNotification(agxSDK::Simulation* simulation, Shovel* shovel);

    /**
    \internal

    On removeNotification of Terrain and Shovel.
    */
    void removeNotification(agxSDK::Simulation* simulation, Shovel* shovel);

    /**
    \internal

    On pre step of Terrain and Shovel.
    */
    void onPre(TerrainToolCollection* collection);

    PenetrationModel getPenetrationModel();

    void setPenetrationModel(PenetrationModel model);

    agx::Prismatic* getPenetrationPrismatic();

    void setupPenetrationResistance(TerrainToolCollection* collection);

    bool isActive();

    void onEnableChange( bool enable );

    /**
    \internal

    Set whether the penetration resistance should be active or not by setting the enable flag on the
    penetration prismatic motor.
    \param enable - set to true/false if the penetration resistance should be active or not
    */
    void setActive(bool enable);

  protected:
    virtual ~SoilPenetrationResistance();

  private:

    agx::Real calculateToothPressureNoModel(TerrainMaterial* material,
                                            agx::Real E,
                                            agx::Real p0,
                                            agx::Real a,
                                            agx::Real a0);

    agx::Real calculateToothPressureElasticPlasticLimit(TerrainMaterial* material,
                                                        agx::Real E,
                                                        agx::Real p0,
                                                        agx::Real a,
                                                        agx::Real a0);

    agx::Real calculateToothPressurePlastic(TerrainMaterial* material,
                                            agx::Real E,
                                            agx::Real p0,
                                            agx::Real dilatancyAngle,
                                            agx::Real a,
                                            agx::Real a0);

    void calculateToothRadiusAndDepth(TerrainToolCollection* collection, agx::Real& a, agx::Real& h);

    agx::Real calculateSeparatingPlateArea(TerrainToolCollection* collection);

    agx::Real calculateSeparatingPlateMassParticles(TerrainToolCollection* collection);

    void getPressureAndYoungsModulusAtTeeth( TerrainToolCollection* collection,
                                             agx::UInt nTeeth,
                                             agx::Real& p0,
                                             agx::Real& E,
                                             agx::Real& compaction,
                                             agx::Real& meanDilatancyAngle,
                                             agx::Real& meanDepth );

    agx::Real calculateSeparatingPlateForce(TerrainToolCollection* collection);

    agx::Real calculateToothForce(TerrainToolCollection* collection);

    void calculatePowerSeriesCoefficients(const agx::Real gamma,
                                          const agx::Real xi,
                                          const agx::UInt n,
                                          agx::RealVector& coefficients);

    agx::Real calculatePowerSeries(const agx::Real x,
                                   const agx::Real gamma,
                                   const agx::UInt n,
                                   const agx::RealVector& coefficients);

    agx::Real calculatePressureLimit(const agx::Real m,
                                     const agx::Real beta,
                                     const agx::Real gamma,
                                     const agx::Real delta,
                                     const agx::Real eta,
                                     const agx::UInt n,
                                     const agx::RealVector& coefficients);

    agx::Real calculatePressureRatio(const agx::Real R_limit,
                                     const agx::Real a,
                                     const agx::Real a0,
                                     const agx::Real m,
                                     const agx::Real beta,
                                     const agx::Real gamma,
                                     const agx::Real delta,
                                     const agx::Real eta,
                                     const agx::UInt n,
                                     const agx::RealVector& coefficients);

  private:
    agx::PrismaticRef m_penetrationPrismatic;
    PenetrationModel m_model;
  };
}