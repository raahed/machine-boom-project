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
#include <agxTerrain/export.h>
#include <agx/Referenced.h>
#include <agx/BitState.h>
#include <array>

namespace agxTerrain
{
  AGX_DECLARE_POINTER_TYPES(TerrainProperties);
  /**
  Class for containing and managing the different agxTerrain::Terrain settings.
  */
  class AGXTERRAIN_EXPORT TerrainProperties : public agx::Referenced
  {
  public:

    /**
    \return the absolute lower speed (m/s) threshold where soil particles are allowed to merge with the terrain.
    */
    agx::Real getSoilMergeSpeedThreshold() const;

    /**
    \return whether or not to use locked borders. I.e the borders of the terrain are not allowed to
            change from excavation and avalanching.
    */
    bool getEnableLockedBorders() const;

    /**
    \return whether or not to use soil compaction calculation in the terrain.
    */
    bool getEnableSoilCompaction() const;

    /**
    \return whether or not to use avalanching in the terrain.
    */
    bool getEnableAvalanching() const;

    /**
    \return true if particles outside the terrain bounds should be deleted - otherwise false (default).
    */
    bool getDeleteSoilParticlesOutsideBounds() const;

    /**
    \return the lifetime (s) of the created soil particles. (Default: Infinity).
    */
    agx::Real getSoilParticleLifeTime() const;

    /**
    \return the penetration force velocity scaling constant. This will scale the penetration force
            with the shovel velocity squared in the cutting direction according to: ( 1.0 + C * v^2 ) (Default: 0.0)
    */
    agx::Real getPenetrationForceVelocityScaling() const;

    /**
    Get the maximum allowed height transfer per time step due to avalanching.
    */
    agx::Real getAvalancheMaxHeightGrowth() const;

    /**
    \return the fraction of the height difference that violates the angle of repose condition that will be
            transferred in each time step during avalanching.
    */
    agx::Real getAvalancheDecayFraction() const;

    /**
    \return the the fractional error threshold for the height difference that will trigger avalanching in a grid point.
    \note - the height difference error threshold is calculated using: errorThreshold * elementSize.
    */
    agx::Real getAvalancheErrorThreshold() const;

    /**
    Set the compliance of the soil aggregate lock joint the local constraint dimensions to relax or increase the force
    feedback interaction from shovel - terrain interaction.
    \note The z - direction(dof = 5) is perpendicular to the plane that the cutting edge and the top edge exists in, which
          is typically in the digging direction.
    dof: 0 - 2 = translational direction
    dof: 3 - 5 = rotational direction
    \return the current compliance in the lock joint in the specified degree of freedom.
    */
    agx::Real getSoilAggregateLockCompliance(agx::Int dof) const;

    /**
    \return whether the terrain should create particles or not during shovel interactions.
    */
    bool getCreateParticles() const;

    /**
    \return the maximum volume of active zone wedges that should wake particles.
    */
    agx::Real getMaximumParticleActivationVolume() const;

    /**
    Get the shovel speed threshold in the excavation plane of shovel deformers when static mass in the active
    zone is converted to dynamic mass.
    */
    agx::Real getActivationSpeed() const;

    /**
    \return the rate factor in the algorithm that governs particle growth from fluid mass and resizing. (Default: 0.05)
    */
    agx::Real getSoilParticleGrowthRate() const;

    /**
    \return true if dynamic mass should be created during excavation. If false, this will prevent the
            creation of fluid and soil particle mass during shovel excavation; thus only solid removal
            will be active.
    */
    bool getEnableCreateDynamicMass() const;

    /**
    \return true if the terrain should be deformable. If false, no avalanching will occur, no dynamic particles will be created and
            the soil will not be compacted.
    */
    bool getEnableDeformation() const;

    /**
    Set the absolute lower speed (m/s) threshold where soil particles are allowed to merge with the terrain.
    \param speedThreshold - The lower speed threshold where soil particles are allowed to
           merge with the terrain. ( Default: 4 )
    */
    void setSoilMergeSpeedThreshold(agx::Real speedThreshold);

    /**
    Set the merge rate for soil particles into the terrain. The merge rate is defined as the fraction
    of the current particle mass that should be merged into the terrain for each second.
    \param mergeRate - the fraction of the current particle mass that should be merged
                       into the terrain for each second. ( Default: 9)
    */
    void setSoilParticleMergeRate( agx::Real mergeRate );

    /**
    \return the merge rate for soil particles in the terrain. The merge rate is defined as the fraction
    of the current particle mass that should be merged into the terrain for each second.
    */
    agx::Real getSoilParticleMergeRate() const;

    /**
    Set the fraction of the particle nominal radius that will trigger instant merge of soil particles
    into the terrain during merging.
    \param radiusFraction - the fraction of the particle nominal radius that will trigger instant
                            merge for a particle. (Default: 0.4)
    \note - radiusFractionvalue must be in the interval (0, 1.0] to be valid ( zero excluded ).
    */
    void setSoilParticleMergeThreshold( agx::Real radiusFraction );

    /**
    \return the fraction of the particle nominal radius that will trigger instant merge of particles
            into the terrain during merging.
    */
    agx::Real getSoilParticleMergeThreshold() const;

    /**
    Set the lifetime (s) of created soil particles in the terrain. The particle will be deleted after existing
    for the specified lifetime.
    \param lifeTime - The lifetime of created particles. ( Default: Infinity )
    */
    void setSoilParticleLifeTime(agx::Real lifeTime);

    /**
    Set the fraction of the height difference that violates the angle of repose condition that will be
    transferred in each time step during avalanching.
    \param decayFraction - The fraction of the height difference that violates the angle of repose
                           that will be transferred. (Default: 0.1)
    */
    void setAvalancheDecayFraction(agx::Real decayFraction);

    /**
    Set the maximum allowed height (m) transfer per time step due to avalanching.
    \param maxHeightGrowth - The maximum height allowed during a height (m) transfer. (Default: infinity)
    */
    void setAvalancheMaxHeightGrowth(agx::Real maxHeightGrowth);

    /**
    Set the fractional error threshold for the height difference that will trigger avalanching in a grid point.
    \param errorThreshold - the fractional error threshold that will determine if a height difference will trigger avalanching. (Default: 1E-4)
    \note - the height difference error threshold is calculated using: errorThreshold / elementSize.
    */
    void setAvalancheErrorThreshold( agx::Real errorThreshold );

    /**
    Set whether the terrain should create particles or not during shovel interactions.
    \param enable - true if particles should be created from excavation - false otherwise.
    */
    void setCreateParticles(bool enable);

    /**
    Set whether or not to use the soil compaction calculations in the terrain.
    \param enable - true if compaction should be calculated in the terrain - false otherwise.
    */
    void setEnableSoilCompaction(bool enable);

    /**
    Set if terrain should delete soil particles outside of terrain bounds.
    \param enable - true if particle should be deleted when outside terrain bounds - false otherwise.
    */
    void setDeleteSoilParticlesOutsideBounds(bool enable);

    /**
    Set whether to fixate the height of the borders in the terrain. I.e the borders of the terrain
    are not allowed to change from excavation and avalanching.
    \param enable - true if borders should be locked - false otherwise.
    */
    void setEnableLockedBorders(bool enable);

    /**
    Set whether to enable avalanching in the terrain.
    \param enable - true if avalanching should be enabled - false otherwise.
    */
    void setEnableAvalanching(bool enable);

    /**
    Set the rate factor in the algorithm for growing soil particles from fluid mass and particle resizing.
    \param growthRateFactor - the rate factor in the algorithm that governs particle growth from fluid mass and resizing. (Default: 0.125)
    */
    void setSoilParticleGrowthRate( agx::Real growthRateFactor );

    /**
    Set the compliance of the soil aggregate lock joint the local constraint dimensions to relax or increase the force
    feedback interaction from shovel -terrain interaction.
    \note The z-direction (dof = 5) is perpendicular to the plane that the cutting edge and the top edge exists in, which
          is typically in the digging direction.
    dof: 0-2 = translational direction
    dof: 3-5 = rotational direction
    \param compliance - the compliance to set in the specified degree of freedom
    \param dof - Target dof for the change of compliance
    */
    void setSoilAggregateLockCompliance(agx::Real compliance, agx::Int dof);

    /**
    Set the penetration force velocity scaling constant. This will scale the penetration force
    with the shovel velocity squared in the cutting direction according to: ( 1.0 + C * v^2 )
    \param penetrationForceVelocityScaling - The coefficient for scaling the penetration force with the
                                             shovel velocity in the cutting direction (Default: 0.0)
    */
    void setPenetrationForceVelocityScaling(agx::Real penetrationForceVelocityScaling);

    /**
    Sets the maximum volume (m3) of active zone wedges that should wake particles.
    \param volume - the maximum volume of active zone wedges that should wake particles. ( Default:Infinity )
    */
    void setMaximumParticleActivationVolume(agx::Real volume);

    /**
    Sets if dynamic mass should be created during excavation. Setting this to false will prevent the
    creation of fluid and soil particle mass during shovel excavation; thus only solid removal will be active.
    \param enable - True if dynamic mass should be created during excavation, false otherwise ( Default: True )
    */
    void setEnableCreateDynamicMass( bool enable );


    /**
    Sets if terrain deformation should be enabled. Setting this to false will prevent the creation of fluid and soil particle mass
    and prevent both avalanching and compaction of the terrain.
    \param enable - True if terrain deformations should occur, false otherwise (Default: True)
    */
    void setEnableDeformation( bool enable );

    /**
    Set the shovel speed threshold in the excavation plane when static mass in the active
    zone is converted to dynamic mass.
    \param activationSpeed - the specified speed threshold where static mass is
                             converted to dynamic mass in the active zone. (Default: 0.1 m/s)
    */
    void setActivationSpeed( agx::Real activationSpeed );

    /**
    Get soil particle size scaling factor.
    */
    float getSoilParticleSizeScaling();

    /**
    Set soil particle size scaling factor. The soil particle size scaling factor scales the
    nominal radius that the algorithm will aim for during the dynamic
    resizing of particles that occur during terrain interaction. This is used to alter the desired number of
    soil particles in the terrain. Default value: 1.0
    */
    void setSoilParticleSizeScaling(float scalingFactor);

    /**
    \return whether the TerainProperties is dirty.
    */
    bool isDirty() const;

    /**
    Set dirty flag on the TerrainProperties
    \param isDirty - set whether the properties should be marked as dirty or not.
    */
    void setDirty(bool isDirty);

    DOXYGEN_START_INTERNAL_BLOCK()

    /**
    Default constructor
    */
    TerrainProperties();

    AGXTERRAIN_STORE_RESTORE_INTERFACE;

    DOXYGEN_END_INTERNAL_BLOCK()
  protected:
    virtual ~TerrainProperties();

    enum StateFlags : agx::UInt32
    {
      LOCKED_BORDERS = 1 << 0,
      CREATE_PATICLES = 1 << 1,
      DELETE_PARTICLES_OUTSIDE_BOUNDS = 1 << 2,
      ENABLE_SOIL_COMPACTION = 1 << 3,
      ENABLE_AVALANCHING = 1 << 4,
      ENABLE_DYNAMIC_MASS = 1 << 5,
      ENABLE_DEFORMATION = 1 << 6,
    };
    using Flags = agx::BitState<StateFlags, agx::UInt32>;

  private:
    Flags                    m_flags;
    agx::Real                m_avalancheDecayFraction;
    agx::Real                m_avalancheMaxHeightGrowth;
    agx::Real                m_penetrationForceVelocityScaling;
    agx::Real                m_soilMergeSpeedThreshold;
    agx::Real                m_maximumActivationVolume;
    agx::Real                m_soilParticleLifeTime;
    agx::Real                m_avalancheErrorThreshold;
    std::array<agx::Real, 6> m_soilAggregateCompliance;
    agx::Real                m_activationSpeed;
    agx::Real                m_particleGrowthRateFactor;
    agx::Real                m_soilParticleMergeRate;
    agx::Real                m_soilParticleMergeThreshold;
    float                    m_particleSizeScaling;
    bool                     m_isDirty;
  };


  inline bool TerrainProperties::getEnableDeformation() const
  {
    return m_flags.Is( StateFlags::ENABLE_DEFORMATION );
  }

  inline bool TerrainProperties::getEnableLockedBorders() const
  {
    return m_flags.Is(StateFlags::LOCKED_BORDERS);
  }



  inline agx::Real TerrainProperties::getAvalancheMaxHeightGrowth() const
  {
    return m_avalancheMaxHeightGrowth;
  }



  inline agx::Real TerrainProperties::getAvalancheDecayFraction() const
  {
    return m_avalancheDecayFraction;
  }



  inline bool TerrainProperties::getDeleteSoilParticlesOutsideBounds() const
  {
    return m_flags.Is(StateFlags::DELETE_PARTICLES_OUTSIDE_BOUNDS);
  }



  inline agx::Real TerrainProperties::getPenetrationForceVelocityScaling() const
  {
    return m_penetrationForceVelocityScaling;
  }



  inline agx::Real TerrainProperties::getMaximumParticleActivationVolume() const
  {
    return m_maximumActivationVolume;
  }



  inline agx::Real TerrainProperties::getSoilAggregateLockCompliance(agx::Int dof) const
  {
    return m_soilAggregateCompliance[dof];
  }



  inline bool TerrainProperties::getEnableCreateDynamicMass() const
  {
    return m_flags.Is( StateFlags::ENABLE_DYNAMIC_MASS );
  }


}

