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

#include <agxTerrain/Terrain.h>

namespace agxTerrain
{
  /**
  Utility class where most of the physics calculation used in agxTerrain is gathered.
  */
  class TerrainPhysics
  {
  public:
    /**
    Estimate new compaction in a voxel from an applied stress using the density log10 method. Compaction occurs only if the applied stress is bigger than
    the pre-consolidation stress in the element. The compaction curve follows the following expression:

    delta_rho = (C * log10( stress / preconsolidationStress )) * timeFactor
    new_rho = old_rho + delta_rho

    where 'stress' is the applied stress in the voxel, 'preconsolidationStress' is the pre-consolidation stress of the material in nominal compaction given by the User in BulkMaterial,
    'C' is the compression index of the material, 'old_rho' the density of the material given nominal compaction of the material, 'delta_rho' is the change in density, timeFactor is
    a factor the relaxes the rate change in density/compaction over time given a relaxation constant (See 'calculateCompactionTimeFactor') and 'new_rho' is the new density.

    The new compaction of the material is given by the ratio between the new and the old density:

    new_compaction = new_rho/old_rho

    \param terrain - The Terrain that will be processed
    \param voxelIndex - The specified index to check for compression
    \param stress - The applied stress (Pa) in the voxel.
    \param timeFactor- The time factor (0.0-1.0) which modifies the calculated change in compaction from the applied stress. Used for compaction rate relaxation.
    \param newCompaction - The new compaction of the voxel, or the old one if no compaction occurred.
    \return True if compaction occurred in the voxel, false otherwise.
    */
    static bool calculateNewCompactionFromStressLog10(Terrain* terrain,
                                                      const agx::Vec3i& voxelIndex,
                                                      agx::Real stress,
                                                      agx::Real timeFactor,
                                                      agx::Real& newCompaction);

    /**
    Estimate new compaction in a voxel from an applied stress using the void ratio method. Compaction occurs only if the applied stress is bigger than
    the pre-consolidation stress in the element. The compaction curve follows the following expression:

        rho_new = rho_old * ( 1.0 / ( 1.0 - phi0 * C * ln( stress / preconsolidationStress ) ) ) * timeFactor

    where 'C' is the compression index, 'rho_new' is the new density of the material, 'rho_old' is the old density
    before compaction, 'preconsolidationStress' is stress used to initialize the soil in the bank sate, 'stress' is the
    applied compaction stress, timeFactor is a factor the relaxes the rate change in density/compaction over time given
    a relaxation constant (See 'calculateCompactionTimeFactor') and 'new_rho' is the new density. and phi zero is derived
    from the initial void ratio e0 ( Default phi0: 0.5 ) in the soil:

        phi0 = 1.0 / ( 1.0 + e0 )

    The new compaction of the material is given by the ratio between the new and the old density:

      new_compaction = new_rho/old_rho

    \param terrain - The Terrain that will be processed
    \param voxelIndex - The specified index to check for compression
    \param stress - The applied stress (Pa) in the voxel.
    \param timeFactor- The time factor (0.0-1.0) which modifies the calculated change in compaction from the applied stress. Used for compaction rate relaxation.
    \param newCompaction - The new compaction of the voxel, or the old one if no compaction occurred.
    \return True if compaction occurred in the voxel, false otherwise.
    */
    static bool calculateNewCompactionFromStressVoidRatio(Terrain* terrain,
                                                          const agx::Vec3i& voxelIndex,
                                                          agx::Real stress,
                                                          agx::Real timeFactor,
                                                          agx::Real& newCompaction);

    /**
    Calculate stress at a given depth directly under the middle axis of a uniform disk, given
    in "An Introduction to Soil Mechanics pg. 219." This is the Boussinesq method integrated
    over a uniform disk.
    \param depth - The specified depth under the disk to calculate the stress.
    \param surfaceLoad - The applied surface stress (Pa) from the uniform disk.
    \param diskRadius - The radius of the uniform surface disk.
    \return The stress (Pa) at the specified depth given by the uniform disk method.
    */
    static agx::Real calculateStressFromUniformDiskLoad(agx::Real depth, agx::Real surfaceLoad, agx::Real diskRadius);

    /**
    Calculate stress at a given depth and distance from a surface point force using the Boussinesq-Frolich method.
    \param depth - The specified depth at which to calculate the stress.
    \param surfaceForce - The applied point force (N) at the surface.
    \param concentrationFactor - The Frolich concentration factor which determines the elasticity of the soil. Default value is 3, which is perfectly elastic.
    \param rk_vec - The distance vector from the point force to the location in at which to calculate the stress.
    \return The stress (Pa) at the specified depth given by the Boussinesq.
    */
    static agx::Real calculateBoussinesqFrolichSubSoilStress(agx::Real depth, agx::Real surfaceForce, agx::Real concentrationFactor, const agx::Vec3& rk_vec);

    /**
    Calculate the compression multiplier which is used to modify the Young's Modulus of terrain contacts given the soil compaction at the contact location.
    \param hardnessRate - The specified hardness rate of the material, i.e how fast it gets stiffer with compaction.
    \param compaction - The specified compaction of the soil.
    \return The compression multiplier used to modify the stiffness of the material.
    */
    static agx::Real calculateCompressionMultiplier( agx::Real hardnessRate, agx::Real compaction );

    /**
    Calculate the compaction time factor used to relax the compaction rate of the material. The relaxation of the compaction rate follows the following formula:

    new_compaction = old_compaction + delta_compaction * time_factor
    time_factor = 1.0 - exp(-contactTime / tau_c)

    The longer a contact has been active, the faster the compaction rate will increase until it reaches the maximum change in compaction given by the applied stress.
    \param contactTime - The specified contact time.
    \param tau_c - The relaxation time constant.
    \return The time factor used to modify the compaction rate.
    */
    static agx::Real calculateCompactionTimeFactor(agx::Real contactTime, agx::Real tau_c);

    /**
    Calculate the multiplier which is used to modify the local angle of repose of the material given the compaction in a voxel.
    \param terrain - The terrain object
    \param voxelIndex - The specified voxel index
    \return The angle of repose multiplier in the voxel element given by the specified index.
    */
    static agx::Real getAngleOfReposeMultiplier(Terrain* terrain, const agx::Vec3i& voxelIndex);

    /**
    Calculate the multiplier which is used to modify the local angle of repose of the material given the compaction in a voxel.
    \param terrain - The terrain object
    \param compaction - The specified compaction
    \return The angle of repose multiplier in the voxel element given by the specified index.
    */
    static agx::Real getAngleOfReposeMultiplier(Terrain* terrain, const agx::Real compaction);

    /**
    Calculate the local repose height, i.e the height difference where above that avalanching will begin, between two terrain indices, which is
    determined by the material angle of repose and the local compaction.
    \param terrain - The terrain object
    \param receivingIndex - The receiving index, the place where material will flow TO.
    \param reposingIndex - The reposing index, the place where material will flow FROM.
    \return the repose height between the two terrain indices. Any height difference that his higher than this will cause avalanching.
    */
    static agx::Real calculateReposeHeight(Terrain* terrain, const agx::Vec2i& receivingIndex, const agx::Vec2i& reposingIndex);


    /**
    Performs the same calculations as calculateReposeHeight and also computes the height difference for the reverse flow
    */
    static agx::Vec2 calculateReposeHeights( Terrain* terrain,
                                             const agx::Vec3i& receivingSurfaceVoxelIndex,
                                             const agx::Vec3i& reposingSurfaceVoxelIndex,
                                             const agx::Vec2& tanReposeAngles);


    /**
    Performs the same calculations as calculateReposeHeight and also computes the height difference for the reverse flow
    */
    static agx::Vec2 calculateReposeHeights( Terrain* terrain,
                                             const agx::Vec3i& receivingSurfaceVoxelIndex,
                                             const agx::Vec3i& reposingSurfaceVoxelIndex,
                                             const agx::Vec2& tanReposeAngles,
                                             const agx::Real& receivingSurfaceCompaction,
                                             const agx::Real& reposingSurfaceCompaction);

    /**
    Calculate a new Young's modulus given a hardening rate and compaction.
    \param originalYoungsModulus - The original Young's modulus of the material.
    \param hardeningRate - The hardening rate of the material.
    \param compaction - The local compaction of the material.
    \return The modified Young's modulus given by the hardening rate and the local compaction of the material.
    */
    static agx::Real calculateYoungsModulus_POW2(agx::Real originalYoungsModulus, agx::Real hardeningRate, agx::Real compaction);

    /**
    Calculate a new Young's modulus using the Cam-Clay model from critical-state theories. This is the following expression:

      E_eff = E_0 * ( 1.0 + sign( compaction - 1.0 ) * k_e * ( abs( compaction - 1.0 ) ^ n_e ) )

    where 'E_eff' is the effective Young's Modulus, 'E_0' is the original Young's modulus, 'compaction' is the local compaction of the mateiral,
    'k_e' is and 'n_e' are hardening parameters that for a constant packing ratio has values 1.0 and 0.5 respectively.

    \param originalYoungsModulus - The original Young's modulus of the material.
    \param k_e - One of the hardening parameters of the model
    \param n_e - One of the hardening parameters of the model
    \param compaction - The local compaction of the material.
    \return The modified Young's modulus given by the Cam-Clay model with hardening constants and the local compaction of the material.
    */
    static agx::Real calculateYoungsModulus_CAM_CLAY(agx::Real originalYoungsModulus, agx::Real k_e, agx::Real n_e, agx::Real compaction);

    /**
    Calculate the hardness multiplier for the the Cam-Clay model from critical-state theories. This is the following expression:

      h = ( 1.0 + sign( compaction - 1.0 ) * k_e * ( abs( compaction - 1.0 ) ^ n_e ) )

    where 'compaction' is the local compaction of the mateiral, 'k_e' is and 'n_e' are hardening parameters that for a constant
    packing ratio has values 1.0 and 0.5 respectively.

    \param k_e - One of the hardening parameters of the model
    \param n_e - One of the hardening parameters of the model
    \param compaction - The local compaction of the material.
    \return The hardness multiplier for the CAM-Clay model.
    */
    static agx::Real calculateHardnessMultiplier_CAM_CLAY(agx::Real k_e, agx::Real n_e, agx::Real compaction);

    /**
    Calculate the pre-consolidation stress of the material in a voxel with specified index using the void ratio method.
    \param terrain - The terrain object
    \param voxelIndex - The specified voxel index
    \return The pre-consolidation of solid mass in the voxel given by the specified index.
    */
    static agx::Real calculatePreconsolidationStressLog10(Terrain* terrain, const agx::Vec3i& voxelIndex);

    /**
    Calculate the pre-consolidation stress of the material in a voxel with specified index using the void ratio method.
    \param terrain - The terrain object
    \param voxelIndex - The specified voxel index
    \return The pre-consolidation of solid mass in the voxel given by the specified index.
    */
    static agx::Real calculatePreconsolidationStressVoidRatio(Terrain* terrain, const agx::Vec3i& voxelIndex);

    /**
    Calculate the local dilatancy angle in a specified voxel in the terrain. The dilatancy angle is
    a function of the local compaction, the critical compaction and the nominal dilatancy angle:

    local_dilatancy_angle = dilatancyAngleScalingFactor * ( local_compaction - critical_compaction )

    \param terrain - The terrain object
    \param voxelIndex - The specified voxel index
    \return The local dilatancy angle in the Terrain at specified voxel.
    */
    static agx::Real calculateDilatancyAngleVoxel( const Terrain* terrain, const agx::Vec3i& voxelIndex );

    /**
    Calculate the dilatancy angle for a TerrainMaterial given a specific compaction. The dilatancy
    angle is a function of the local compaction, the critical compaction and the nominal dilatancy
    angle:

    local_dilatancy_angle = dilatancyAngleScalingFactor * ( local_compaction - critical_compaction )

    \param terrainMaterial - The TerrainMaterial object
    \param compaction - The specified compaction that is used to calculate the dilatancy angle
    \return The dilatancy angle of the TerrainMaterial given a specific compaction.
    */
    static agx::Real calculateDilatancyAngle( const TerrainMaterial* terrainMaterial, agx::Real compaction );

    /**
    Calculate the effective friction angle of a voxel in a specific terrain. The effective friction
    angle is expressed as:

    eff_friction_angle = base_friction_angle + voxel_dilatnacy_angle

    \param terrain - The TerrainMaterial object
    \param voxelIndex - The specified voxel index where the effective friction angle of the terrain will be calculated.
    \return The effective friction angle of a specified voxel index in the specified terrain.
    */
    static agx::Real calculateEffectiveFrictionAngleVoxel( const Terrain* terrain, const agx::Vec3i& voxelIndex );

    /**
    Calculate the effective repose angle of a specific terrain, unmodified by compaction. The repose
    angle is expressed as:

    effective_repose_angle = internal_friction_angle + delta_repose_angle

    \param terrain - The TerrainMaterial object
    \return The effective repose angle of the terrain.
    */
    static agx::Real calculateEffectiveReposeAngle(const Terrain* terrain);

    /**
    Calculate the effective friction angle at the surface of a specific terrain with a given terrain index.
    The effective friction angle is expressed as:

    eff_friction_angle = base_friction_angle + voxel_dilatnacy_angle

    \param terrain - The TerrainMaterial object
    \param terrainIndex - The specified terrain index where the effective friction angle of the terrain will be calculated.
    \return The effective friction angle of a specified terrain at a given terrain index.
    */
    static agx::Real calculateEffectiveFrictionAngleTerrainIndex( const Terrain* terrain, const agx::Vec2i& terrainIndex );

    /**
    Calculate the effective friction angle of a TerrainMaterial given a specified compaction. The effective friction angle
    can be expressed as:

    eff_friction_angle = base_friction_angle + voxel_dilatnacy_angle

    \param terrainMaterial - The TerrainMaterial object
    \param compaction - The specified compaction that is used to calculate the effective friction angle
    \return The effective friction angle of a specified terrain at a given terrain index.
    */
    static agx::Real calculateEffectiveFrictionAngle( const TerrainMaterial* terrainMaterial, agx::Real compaction );

    /**
    Estimate the critical compaction of a TerrainMaterial which is used in calculating the local dilatancy angle
    in the terrain from varying compaction. The formula is expressed as:

    critical_compaction = nominal_compaction - dilatancyAngle / dilatancyAngleScalingFactor

    \param terrainMaterial - The TerrainMaterial object
    \return The estimated critical compaction of the TerrainMaterial.
    */
    static agx::Real estimateCriticalCompaction( const TerrainMaterial* terrainMaterial );
  };
}
