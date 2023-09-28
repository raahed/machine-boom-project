/*
Copyright 2007-2023. Algoryx Simulation AB.

All AGX source code, intellectual property, documentation, sample code,
tutorials, scene files and technical white papers, are copyrighted, proprietary
and confidential material of Algoryx Simulation AB. You may not download, read,
store, distribute, publish, copy or otherwise disseminate, use or expose this
material unless having a written signed agreement with Algoryx Simulation AB, or
having been advised so by Algoryx Simulation AB for a time limited evaluation,
or having purchased a valid commercial license from Algoryx Simulation AB.

Algoryx Simulation AB disclaims all responsibilities for loss or damage caused
from using this software, unless otherwise stated in written agreements with
Algoryx Simulation AB.
*/

#pragma once

#include <agx/config/AGX_USE_AGXTERRAIN.h>
#include <agxTerrain/export.h>
#include <agx/Material.h>
#include <array>

namespace agxTerrain
{
  class Terrain;
  AGX_DECLARE_POINTER_TYPES(TerrainMaterial);

  /**
  This class is used to describe the general material properties of the Terrain. When a TerrainMaterial
  instance is added to a terrain object, it affects the soil failure zone characteristics and feedback forces
  and also calibrates the internal contact materials governing soil dynamics. All the different parameters
  are grouped into different objects that reflects the phenomena that they affect:

    BulkProperties         - Holds the overarching bulk properties of the terrain and mainly affect excavation
                             zone shapes, penetration resistance and contact mateiral properties for excavation
                             feedback forces.

    CompactionProperties   - Properties with regards to the compaction algorithm used in the terrain.

    ParticleProperties     - Properties that govern the soil dynamics of created soil particles during excavation,
                             which is expressed via the contact material parameters within the soil particles and without.

    ExcavationContactProperties - Properties that govern the contact properties in the shovel <-> soil aggregate
                                  and soil aggregate <-> terrain contacts during excavation.

   These objects can be extracted from the TerrainMaterial instance in order to change the Terrain characteristics.

   Note that using any of the set methods on a TerrainMaterial instance will cause it to recalibrate the set method's
   governing contact material parameter in the terrain instance. This can override any previous manual user setting
   of the terrain contact material.
  */
  class AGXTERRAIN_EXPORT TerrainMaterial : public agx::Referenced
  {
  public:
    AGX_DECLARE_POINTER_TYPES(BulkProperties);
    AGX_DECLARE_POINTER_TYPES(ParticleProperties);
    AGX_DECLARE_POINTER_TYPES(CompactionProperties);
    AGX_DECLARE_POINTER_TYPES(ExcavationContactProperties);

  public:
    /**
    Default constructor.
    */
    TerrainMaterial(const agx::String& description="");

    /**
    \return Then name of the object.
    */
    const agx::String& getDescription() const;

    /**
    \return the compaction properties related to the soil compaction model of the terrain.
    */
    BulkProperties* getBulkProperties() const;

    /**
    \return the compaction properties related to the soil compaction model of the terrain.
    */
    CompactionProperties* getCompactionProperties() const;

    /**
    \return the soil particle properties of the agxTerrain material.
    */
    ParticleProperties* getParticleProperties() const;

    /**
    \return the excavation contact properties.
    */
    ExcavationContactProperties* getExcavationContactProperties() const;

    /**
    \return the packing fraction of the material
    */
    agx::Real getSoilPackingFraction() const;

    /**
    Set the name of the object.
    */
    void setDescription(const agx::String& description );

    /**
    \return the last error during construction of the TerrainMaterial.
    \note - Construction is typically done in TerrainMaterialReaderWriter.
    */
    agx::String getLastError() const;

    /**
    Print the internal material parameters in a formatted way.
    */
    void printParameters() const;

  public:

    /**
     Class containing the soil bulk properties of and agxTerrain object. These properties typically affect the shape and force feedback
     of the failure zones that appear during excavation but also affect the shovel penetration resistance.

     The bulk material properties governs the following internal contact material parameters:

     YoungsModulus     - The stiffness of the material in Pa. This affects the Young's Modulus parameter of the
                         contact materials.
     Cohesion          - Describes the internal cohesion of the material in Newton. This translates to the adhesion
                         parameter of the internal contact material.
     Density           - Describes the density of the soil in kg/m3. This translates to the specific density of the
                         solid soil material and the bulk density of the dynamic soil (i.e Soil Particles).
     FrictionAngle     - Describes the internal friction angle of the material. This affects the
                         angle of repose in a 1-to 1 ration for the avalanching algorithm.
     PoissonsRatio     - Describes the Poisson's ratio of the material. This affects the soil
                         penetration force calculations.
     DilatancyAngle    - Describes the dilatancy angle of the material. This affects the soil
                         penetration force calculations.
    */
    class AGXTERRAIN_EXPORT BulkProperties : public agx::Referenced
    {
    public:
      /**
      Default constructor.
      */
      BulkProperties(TerrainMaterial* terrainMaterial);

      /**
     \return the bulk density (kg/m3) of the bulk material.
     */
      agx::Real getDensity() const;

      /**
      \return the maximum density (kg/m3) of the bulk material.
      */
      agx::Real getMaximumDensity() const;

      /**
      \return the bulk Young's modulus (Pa) of the bulk material.
      */
      agx::Real getYoungsModulus() const;

      /**
      \return the bulk Poisson's ratio of the bulk material.
      */
      agx::Real getPoissonsRatio() const;

      /**
      \return the friction angle (rad) of the bulk material.
      */
      agx::Real getFrictionAngle() const;

      /**
      \return the dilatancy angle (rad) of the bulk material.
      */
      agx::Real getDilatancyAngle() const;

      /**
      \return the bulk cohesion (Pa) of the bulk material.
      */
      agx::Real getCohesion() const;

      /**
      \return the swell factor of the material, i.e the volume expansion of created dynamic soil during excavation.
      */
      agx::Real getSwellFactor() const;

      /**
      \return the delta repose angle of the material, used to caluclate the
              angle of repose for a material. See `setDeltaReposeAngle`.
      */
      agx::Real getDeltaReposeAngle() const;

      /**
      Set the bulk density (kg/m^3) of the bulk material.
      \param density - the specified bulk density to set to the bulk material. (Default: 1.4e3 kg/m^3)
      */
      void setDensity(agx::Real density);

      /**
      Set the maximum density (kg/m^3) of the bulk material. This is the upper limit for the soil compaction.
      \param maxDensity - the specified density to set to the bulk material. (Default: 1.6e3 kg/m^3)
      */
      void setMaximumDensity(agx::Real maxDensity);

      /**
      Set the bulk Young's modulus (Pa) of the bulk material.
      \param youngsModulus - the specified bulk Young's modulus to set to the bulk material. (Default: 1e7 Pa)
      */
      void setYoungsModulus(agx::Real youngsModulus);

      /**
      Set the bulk Poisson's ratio of the bulk material. This affects the soil
      penetration force calculations.
      \param poissonsRatio - the specified bulk Poisson's ratio to set to the bulk material. (Default: 0.1)
      */
      void setPoissonsRatio(agx::Real poissonsRatio);

      /**
      Set the internal friction angle (rad) of the bulk material. This affects the shape of the
      active zones during excavations and also the angle of repose in a 1-to 1 ration for the
      avalanching algorithm.
      \param frictionAngle - the specified internal friction angle to set to the bulk material.
             (Default: 45.0 * agx::PI / 180.0 radians)
      */
      void setFrictionAngle(agx::Real frictionAngle);

      /**
      Set the dilatancy angle (rad) of the bulk material. This determines the material properties
      during expansion and compression. This affects the soil penetration force calculations.
      \param dilatancyAngle - the specified dilatancy angle to set to the bulk material.
             (Default: 15.0 * agx::PI / 180.0 radians)
      */
      void setDilatancyAngle( agx::Real dilatancyAngle );

      /**
      Set the bulk cohesion (Pa) of the bulk material.
      \param cohesion - the specified bulk cohesion to set to the bulk material. (Default: 0.0 Pa)
      */
      void setCohesion(agx::Real cohesion);

      /**
      Set the swell factor of the material, i.e how much the material will expand during excavation.
      The volume of the dynamic material created will increase according to the swell factor:
      volume dynamic = volume excavated * swellFactor
      \param swellFactor - The specified swell factor to set to the material. (Default: 1.0)
      */
      void setSwellFactor(agx::Real swellFactor);

      /**
      Set the delta repose angle. This will increase the angle of repose of the material according to the following formula:

      angle_of_repose = atan( tan(internal_friction angle + delta_repose_angle) * compaction_factor )

      \param deltaReposeAngle - the specified delta repose to set to the material.
      */
      void setDeltaReposeAngle(agx::Real deltaReposeAngle);

      AGXTERRAIN_STORE_RESTORE_INTERFACE;

    protected:
      virtual ~BulkProperties() {}
      void onChanged();

    public:
      agx::Real m_density;
      agx::Real m_maxDensity;
      agx::Real m_youngsModulus;
      agx::Real m_poissonsRatio;
      agx::Real m_frictionAngle;
      agx::Real m_cohesion;
      agx::Real m_dilatancyAngle;
      agx::Real m_swellFactor;
      agx::Real m_deltaReposeAngle;

      TerrainMaterial* m_terrainMaterial;
    };



    /**
    Class containing the properties of the soil compaction model used in agxTerrain, such as compression index and
    soil hardening constants.

    CompressionIndex                 - Describes how quickly the density of the material increases with compression.
    SwellFactor                      - Describes the amount the soil swells when going from its bank state to
                                       its loose state.
    HardeningRate                    - Describes how quickly the Young's modulus of the material increases
                                       with compression.
    PreconsolidationStress           - Describes the pressure the soil was consolidated by.
    CompactionTimeRelaxationConstant - Describes how quickly the material is compressed when exposed to a compressing stress.
    StressCutOffFraction             - Describes the fraction of the surface stress that must be reached before
                                       cutting off stress calculations downward in the Terrain.
    angleOfReposeCompactionRate      - Describes how quickly the angle of repose of the material increases with compression.
   */
    class AGXTERRAIN_EXPORT CompactionProperties : public agx::Referenced
    {
    public:
      CompactionProperties();

      /**
      \return the hardening constant k_e of the bulk material, i.e how the Young's modulus of the terrain contacts should scale with
              increasing/decreasing compaction.
      \note - See 'setHardeningConstantsKE' for scaling law
      */
      agx::Real getHardeningConstantKE() const;

      /**
      \return the hardening constant n_e of the bulk material, i.e how the Young's modulus of the terrain contacts should scale with
              increasing/decreasing compaction.
      \note - See 'setHardeningConstantsNE' for scaling law
      */
      agx::Real getHardeningConstantNE() const;

      /**
      \return the pre-consolidation stress (Pa) of the material, i.e how much stress is needed in order to
              compact the soil further than the starting nominal compaction.
      \note - Nominal compaction is defined as 1.0.
      */
      agx::Real getPreconsolidationStress() const;

      /**
      \return the bank state phi0 used in the compression algorithm (See 'setBankStatePhi'), defined as
              compaction = 1.0 of the material.
      */
      agx::Real getBankStatePhi() const;

      /**
      \return the compression index of the material, i.e how fast the material should
              increase in density from increased stress. See `setCompressionIndex` for scaling law.
      */
      agx::Real getCompressionIndex() const;

      /**
      \return the time relaxation constant for compaction. See setCompactionTimeRelaxationConstant.
              for explanation of compaction rate relaxation.
      */
      agx::Real getCompactionTimeRelaxationConstant() const;

      /**
      \return the fraction of the surface stress that should serve as a cutoff value from when the stress propagation
              from the surface downward into the soil should stop.
      */
      agx::Real getStressCutOffFraction() const;

      /**
      \return how the compaction should affect the angle of repose.
      \note See 'setAngleOfReposeCompactionRate' for explanation of scaling law.
      */
      agx::Real getAngleOfReposeCompactionRate() const;

      /**
      Set the hardening constant k_e of the bulk material, i.e how the Young's modulus of the terrain contacts should scale with
      increasing/decreasing compaction. The formula for this is:

        E_eff = E_0 * ( 1.0 + sign( compaction - 1.0 ) * k_e * ( abs( compaction - 1.0 ) ^ n_e ) )

        where 'E_eff' is the effective Young's Modulus, 'E_0' is the original Young's modulus, 'compaction' is the local compaction of the material,
        'k_e' is and 'n_e' are hardening parameters that for a constant packing ratio has values 1.0 and 0.5 respectively.

      \param k_e - The hardening constant k_e of the material. (Default: 1.0)
      */
      void setHardeningConstantKE(agx::Real k_e);

      /**
      Set the hardening constant n_e of the bulk material, i.e how the Young's modulus of the terrain contacts should scale with
      increasing/decreasing compaction. The formula for this is:

        E_eff = E_0 * ( 1.0 + sign( compaction - 1.0 ) * k_e * ( abs( compaction - 1.0 ) ^ n_e ) )

        where 'E_eff' is the effective Young's Modulus, 'E_0' is the original Young's modulus, 'compaction' is the local compaction of the material,
        'k_e' is and 'n_e' are hardening parameters that for a constant packing ratio has values 1.0 and 0.5 respectively.

      \param n_e - The hardening constant n_e of the material. (Default: 0.5)
      */
      void setHardeningConstantNE(agx::Real n_e);

      /**
      Set the stress at which the soil in the default state was compressed in, i.e when it has nominal compaction 1.0.
      In order to compress the soil further, the applied stress on the soil has to exceed this.
      \param preconsolidationStress - the specified preconsolidationStress to set on the bulk material when in
                                      nominal compaction. (Default: 98e3 Pa)
      */
      void setPreconsolidationStress(agx::Real preconsolidationStress);

      /**
      Set the phi0 value of the bank state soil. This is used in the compaction calculation where soil stress generates compacted soil.
      note See 'setCompressionIndex' for an explanation of how phi0 is used in compaction calculations.
      \param phi0 - The specified phi0 of the soil in the bank state. (Default: 0.5)
      */
      void setBankStatePhi(agx::Real phi0);

      /**
      Sets the compression index for the soil, which is the constant that determines how fast
      the soil should compress given increased surface stress. The formula for computing the compaction
      curve is given by the following expression:

          rho_new = rho_old * ( 1.0 / ( 1.0 - phi0 * C * ln( stress / preconsolidationStress ) ) )

      where 'C' is the compression index, 'rho_new' is the new density of the material, 'rho_old' is the old density
      before compaction, 'preconsolidationStress' is stress used to initialize the soil in the bank sate, 'stress' is the
      applied compaction stress and phi zero is derived from the initial void ratio e0 ( Default phi0: 0.5 ) in the soil:

          phi0 = e0 / ( 1.0 + e0 )

      \param compressionIndex - The compression index C used in the formula above. (Default: 0.1)
      */
      void setCompressionIndex(agx::Real compressionIndex);

      /**
      Set time relaxation for compaction. The factor is used to cap the change in density during the contact according to:
          change in density during time step = change in density from compaction * timeFactor
      where:
          timeFactor = 1.0 - exp(contactTime/tau)
      \param compactionTimeRelaxationConstant - the specified time constant to be set on the bulk material. (Default: 0.05)
      */
      void setCompactionTimeRelaxationConstant(agx::Real compactionTimeRelaxationConstant);

      /**
      Set the fraction of the surface stress that should serve as a cutoff value from when the stress propagation
      from the surface downward into the soil should stop.
      \param stressCutOffFraction - the fraction of the surface stress that serves as the cutoff. (Default: 0.01)
      */
      void setStressCutOffFraction(agx::Real stressCutOffFraction);

      /**
      Set how the compaction should increase the angle of repose. The tan of the angle of repose is increased by the following factor:
      The applied angle of repose of the material is active when the soil is in loose compaction, i.e has compaction 1.0 / swellFactor.

      m  = 2.0 ^ ( angleOfReposeCompactionRate * ( compaction - (1.0/swellFactor) ) )

      \param angleOfReposeCompactionRate - the specified compaction angle of repose rate to set to the bulk material. (Default: 1.0)
      */
      void setAngleOfReposeCompactionRate(agx::Real angleOfReposeCompactionRate);

      /**
      Set the dilatancy angle scaling factor with respect to compaction in the terrain. The dilatancy angle
      modifies the effective local angle of friction in terrain by the following formula:

      eff_friction_angle = friction_angle + scalingFactor * ( voxelCompaction - criticalCompaction )

      \param scalingFactor - The specified dilatancy angle scaling factor to set. ( Default: 20 * agx::PI / 180 radians )
      */
      void setDilatancyAngleScalingFactor( agx::Real scalingFactor );

      /**
      Get the dilatancy angle scaling factor with respect to the compaction in the terrain. The dilatancy angle
      modifies the effective local angle of friction in terrain by the following formula:

      eff_friction_angle = friction_angle + scalingFactor * ( voxelCompaction - criticalCompaction )

      \return the dilatancy angle scaling factor
      */
      agx::Real getDilatancyAngleScalingFactor() const;

      AGXTERRAIN_STORE_RESTORE_INTERFACE;

    protected:
      virtual ~CompactionProperties() {}

    private:
      agx::Real m_hardening_ke;
      agx::Real m_hardening_ne;
      agx::Real m_preconsolidationStress;
      agx::Real m_compressionIndex;
      agx::Real m_phi0;
      agx::Real m_compactionTimeRelaxationConstant;
      agx::Real m_stressCutOffFraction;
      agx::Real m_angleOfReposeCompactionRate;
      agx::Real m_dilatancyAngleScalingFactor;
    };



    /**
    Class containing the properties of the created soil particles in the Terrain, such as particle density and contact parameters
    amongst soil particles and between soil particles and the terrain surface.
    */
    class AGXTERRAIN_EXPORT ParticleProperties : public agx::Referenced
    {
    public:
      ParticleProperties(TerrainMaterial* terrainMaterial);

      /**
      \return the specific density of the generated soil particles. This is automatically derived from the
              terrain bulk density and the resulting packing fraction.
      */
      agx::Real getParticleDensity() const;

      /**
      \return particle Young's modulus (Pa) of the bulk material used in the contacts between particles.
      */
      agx::Real getParticleYoungsModulus() const;

      /**
      \return the particle restitution of the bulk material used in the contacts between particles.
      */
      agx::Real getParticleRestitution() const;

      /**
      \return particle-particle surface friction of the bulk material used in the contacts between particles.
      */
      agx::Real getParticleSurfaceFriction() const;

      /**
      \return particle rolling resistance of the bulk material used in the contacts between particles.
      */
      agx::Real getParticleRollingResistance() const;

      /**
      \return particle cohesion (Pa) of the bulk material used in the contacts between particles.
      */
      agx::Real getParticleCohesion() const;

      /**
      \return particle Young's modulus of against the contacts on the terrain surface.
      */
      agx::Real getParticleTerrainYoungsModulus() const;

      /**
      \return the restitution of the particle <-> terrain contacts.
      */
      agx::Real getParticleTerrainRestitution() const;

      /**
      \return the friction of the particle <-> terrain contacts.
      */
      agx::Real getParticleTerrainSurfaceFriction() const;

      /**
      \return the rolling resistance of the particle <-> terrain contacts.
      */
      agx::Real getParticleTerrainRollingResistance() const;

      /**
      \return the cohesion of the particle <-> terrain contacts.
      */
      agx::Real getParticleTerrainCohesion() const;

      /**
      \return adhesion overlap factor of the particle properties.
      */
      agx::Real getAdhesionOverlapFactor() const;

      /**
      Set the particle Young's modulus (Pa) of the bulk material.
      \param youngsModulus - the specified particle Young's modulus to set to the bulk material. (Default: 1e9 Pa)
      */
      void setParticleYoungsModulus(agx::Real youngsModulus);

      /**
      Set the particle restitution of the bulk material.
      \param restitution - the specified particle restitution to set to the bulk material. (Default: 0.5)
      */
      void setParticleRestitution(agx::Real restitution);

      /**
      Set the particle surface friction of the bulk material.
      \param friction - the specified particle surface friction to set to the bulk material. (Default: 0.4)
      */
      void setParticleSurfaceFriction(agx::Real friction);

      /**
      Set the particle rolling resistance of the bulk material.
      \param rollingResistance - the specified particle rolling resistance to set to the bulk material. (Default: 0.3)
      */
      void setParticleRollingResistance(agx::Real rollingResistance);

      /**
      Set the particle cohesion (Pa) of the bulk material.
      \param cohesion - the specified particle rolling resistance to set to the bulk material. (Default: 0.0 Pa)
      */
      void setParticleCohesion(agx::Real cohesion);

      /**
      Set particle <-> Terrain Young's modulus (Pa) of the bulk material.
      \param youngsModulus - the specified particle <-> terrain Young's modulus to set to the bulk material. (Default: 1e8 Pa)
      */
      void setParticleTerrainYoungsModulus(agx::Real youngsModulus);

      /**
      Set particle <-> terrain restitution of the bulk material.
      \param restitution - the specified particle <-> terrain restitution to set to the bulk material. (Default: 0.0)
      */
      void setParticleTerrainRestitution( agx::Real restitution );

      /**
      Set the particle <-> terrain surface friction of the bulk material.
      \param friction - the specified particle surface friction to set to the bulk material. (Default: 0.8)
      */
      void setParticleTerrainSurfaceFriction(agx::Real friction);

      /**
      Set the particle <-> terrain rolling resistance of the bulk material.
      Set the particle rolling resistance of the bulk material.
      \param rollingResistance - the specified particle rolling resistance to set to the bulk material. (Default: 0.3)
      */
      void setParticleTerrainRollingResistance(agx::Real rollingResistance);

      /**
       Set the particle <-> terrain cohesion (Pa) of the bulk material.
       \param cohesion - the specified particle <-> terrain cohesion to set to the bulk material. (Default: 0.0 Pa)
       */
      void setParticleTerrainCohesion(agx::Real cohesion);

      /**
      Set the adhesion overlap factor of particle contacts, i.e what fraction of the particle radius
      is allowed to overlap to simulate adhesion.
      \param adhesionOverlapFactor - The specified adhesion overlap factor to set to the material. (Default: 0.05)
      */
      void setAdhesionOverlapFactor( agx::Real adhesionOverlapFactor );

      AGXTERRAIN_STORE_RESTORE_INTERFACE;

    protected:
      virtual ~ParticleProperties() {}
      void onChanged();

    protected:
      struct ParticleParticleParameters
      {
        ParticleParticleParameters();

        agx::Real youngsModulus;
        agx::Real restitution;
        agx::Real surfaceFriction;
        agx::Real rollingResistance;
        agx::Real cohesion;
      };

      struct ParticleTerrainParameters
      {
        ParticleTerrainParameters();

        agx::Real youngsModulus;
        agx::Real restitution;
        agx::Real surfaceFriction;
        agx::Real rollingResistance;
        agx::Real cohesion;
      };

    protected:
      ParticleParticleParameters m_particleParticleParameters;
      ParticleTerrainParameters  m_particleTerrainParameters;
      agx::Real                  m_adhesionOverlapFactor;

      TerrainMaterial* m_terrainMaterial;
    };

  public:
    DOXYGEN_START_INTERNAL_BLOCK()

    /**
    Internal method
    Set last error string during construction of the TerrainMaterial class.
    */
    void setLastError( const agx::String& errorString );

    /**
    Called by Terrain in preCollide. Update terrain if TerrainMaterial is dirty.
    */
    void onPreCollideStep( Terrain * terrain );

    /**
    Called by Terrain in pre. Update terrain if TerrainMaterial is dirty.
    */
    void onPreStep( Terrain * terrain );

    /**
    Called by Terrain in pre. Update terrain if TerrainMaterial is dirty.
    */
    void onPostStep( Terrain * terrain );

    /**
    * Internal method
    * \return whether the TerainMaterial is dirty.
    */
    bool isDirty() const;

    /**
    * Internal method
    * Set dirty flag on the TerrainMaterial
    * \param isDirty - set whether the material should be marked as dirty or not.
    */
    void setDirty(bool isDirty);

    /**
     * Internal method
     * \return the allowed overlap used to simulate adhesion
     */
    agx::Real calculateAdhesionOverlap( Terrain * terrain, agx::Real adhesion );

    /**
    * Internal method
    * Synchronizes internal terrain parameters from TerrainMaterial
    * \param terrain - the terrain that uses this bulk material
    */
    void synchronize(Terrain* terrain);

    AGXTERRAIN_STORE_RESTORE_INTERFACE;
    DOXYGEN_END_INTERNAL_BLOCK()

  protected:
    virtual ~TerrainMaterial();

    void updateParticleDensity( Terrain* terrain );
    void synchronizeBulkProperties( Terrain* terrain );
    void synchronizeParticleProperties( Terrain* terrain );

  private:
    agx::String                     m_description;
    agx::String                     m_lastError;
    BulkPropertiesRef               m_bulkProperties;
    ParticlePropertiesRef           m_particleProperties;
    CompactionPropertiesRef         m_compactionProperties;
    ExcavationContactPropertiesRef  m_excavationContactProperties;
    bool                            m_isDirty;
  };



  /**
  Properties that govern the contact properties in the shovel <-> soil aggregate and soil aggregate <-> terrain contacts during excavation.
  The contacts have virtual depth that increases when the separation plane of the excavation presses against the plane of the aggregate <->
  terrain contact. This class contains properties for governing the depth and force properties of those contacts.
  */
  class AGXTERRAIN_EXPORT TerrainMaterial::ExcavationContactProperties : public agx::Referenced
  {
  public:
    /**
    Default constructor
    */
    ExcavationContactProperties(TerrainMaterial* terrainMaterial);

    /**
    \return the maximum depth (m) of an soil aggregate <-> terrain contact. This increases when the separation direction
            of the excavation intersects the contact plane, causing virtual soil compression between soil aggregates and
            the terrain.
    */
    agx::Real getMaximumContactDepth() const;

    /**
    \return the depth decay of an soil aggregate <-> terrain contact. This determines how rapidly the stored depth
            will decay during separation when the active zone moves away from the soil aggregate <-> terrain contact
            plane.
    */
    agx::Real getDepthDecayFactor() const;

    /**
    \return the depth increase factor of an soil aggregate <-> terrain contact. This governs how fast the depth should increase
            when the separation direction of the excavation intersects the contact plane, causing virtual soil compression between
            soil aggregates and the terrain.
    */
    agx::Real getDepthIncreaseFactor() const;

    /**
    \return the maximum force that the soil aggregate <-> terrain contacts are allowed to have. Default maximum values
            are determined by the soil mechanics properties of the terrain.
    */
    agx::Real getMaximumAggregateNormalForce() const;

    /**
    Internal Method

    Get the contact stiffness multiplier for the generated contacts between the soil aggregates <-> terrain for excavation
    and deformation. The final Young's modulus value that will be used in the contact material thus becomes:

      YM_final = BulkYoungsModulus * stiffnessMultiplier

    \return the specified stiffness multiplier for the Young's Modulus in soil aggregate contacts
            for excavation and deformation.
    */
    agx::Real getAggregateStiffnessMultiplier() const;

    /**
    Get the contact stiffness multiplier for the generated contacts between the soil aggregates <-> shovels in primary excavation.
    The final Young's modulus value that will be used in the contact material thus becomes:

      YM_final = BulkYoungsModulus * stiffnessMultiplier

    \return stiffnessMultiplier - Get the specified stiffness multiplier for the Young's Modulus in soil aggregate <-> shovel contacts
                                  for excavation.
    */
    agx::Real getExcavationStiffnessMultiplier() const;

    /**
    Return the lower angle threshold between the separation normal and the aggregate <-> terrain
    contact plane where depth increments in the contact plane should increase the contact model
    depth used in terrain <-> aggregate contact points.
    \return - the angle threshold for when aggregate <-> terrain contact model depth should increase.
    */
    agx::Real getDepthAngleThreshold() const;

    /**
    Internal Method

    Set the contact stiffness multiplier for the generated contacts between the soil aggregates <-> terrain for excavation
    and deformation. The final Young's modulus value that will be used in the contact material thus becomes:

      YM_final = BulkYoungsModulus * stiffnessMultiplier

    \param stiffnessMultiplier - Set the specified stiffness multiplier for the Young's Modulus in soil aggregate contacts
                                 for excavation and deformation. (Default: 0.002)
    */
    void setAggregateStiffnessMultiplier(agx::Real stiffnessMultiplier);

    /**
    Set the contact stiffness multiplier for the generated contacts between the soil aggregates <-> shovels in primary excavation.
    The final Young's modulus value that will be used in the contact material thus becomes:

      YM_final = BulkYoungsModulus * stiffnessMultiplier

    \param excavationStiffnessMultiplier - Set the specified stiffness multiplier for the Young's Modulus in
                                           soil aggregate <-> shovel contacts for excavation. (Default: 1.0)
    */
    void setExcavationStiffnessMultiplier(agx::Real excavationStiffnessMultiplier);

    /**
    Set the maximum depth (m) of a soil aggregate <-> terrain contact. This increases when the separation direction
    of the excavation intersects the contact plane, causing virtual soil compression between soil aggregates and
    the terrain.
    \param maximumDepth - the maximum depth that the soil aggregate <-> terrain contacts are allowed to have. (Default: 1.0)
    */
    void setMaximumContactDepth(agx::Real maximumDepth);

    /**
    Set the depth decay factor of a soil aggregate <-> terrain contact.  This determines how rapidly the stored depth
    in a terrain <-> aggregate contact will decay during separation when the active zone moves away from the soil
    aggregate <-> terrain contact plane.
    \param depthDecay - the decay factor of the depth of a terrain <-> aggregate contact during separation. (Default: 2.0)
    */
    void setDepthDecayFactor(agx::Real depthDecay);

    /**
    Set the depth increase factor of a soil aggregate <-> terrain contact. This governs how fast the depth should increase
    when the separation direction of the excavation intersects the contact plane, causing virtual soil compression between
    soil aggregates and the terrain.
    \param depthIncreaseFactor - The factor the governs how fast the depth should increase when the separation direction intersects
                                 with the contact plane. (Default: 1.0)
    */
    void setDepthIncreaseFactor(agx::Real depthIncreaseFactor);

    /**
    Set the maximum force that the soil aggregate <-> terrain contacts are allowed to have. Default maximum values
    are determined by the soil mechanics properties of the terrain.
    \param maximumAggregateForce - The maximum aggregate force that a terrain contact are allowed to have. (Default: inf)
    */
    void setMaximumAggregateNormalForce(agx::Real maximumAggregateForce);

    /**
    Set the lower angle threshold between the separation normal and the aggregate <-> terrain contact plane
    where depth increments the contact plane should increase the contact model depth used in terrain <-> aggregate
    contact points.
    \param depthAngleThreshold - The angle threshold for when aggregate <-> terrain contact model depth
                                 should increase. ( Default: pi / 4 ( 45 deg ) )
    */
    void setDepthAngleThreshold( agx::Real depthAngleThreshold );

    AGXTERRAIN_STORE_RESTORE_INTERFACE;

  protected:
    virtual ~ExcavationContactProperties();

  private:
    agx::Real m_maximumDepth;
    agx::Real m_depthDecayFactor;
    agx::Real m_depthIncreaseFactor;
    agx::Real m_maximumAggregateNormalForce;
    agx::Real m_aggregateStiffnessMultiplier;
    agx::Real m_excavationStiffnessMultiplier;
    agx::Real m_depthAngleThreshold;
    TerrainMaterial* m_terrainMaterial;
  };



  inline TerrainMaterial::BulkProperties* TerrainMaterial::getBulkProperties() const
  {
    return m_bulkProperties;
  }


  inline TerrainMaterial::CompactionProperties* TerrainMaterial::getCompactionProperties() const
  {
    return m_compactionProperties;
  }


  inline TerrainMaterial::ParticleProperties* TerrainMaterial::getParticleProperties() const
  {
    return m_particleProperties;
  }


  inline TerrainMaterial::ExcavationContactProperties* TerrainMaterial::getExcavationContactProperties() const
  {
    return m_excavationContactProperties;
  }


  inline agx::Real TerrainMaterial::BulkProperties::getDensity() const
  {
    return m_density;
  }


  inline agx::Real TerrainMaterial::BulkProperties::getMaximumDensity() const
  {
    return m_maxDensity;
  }


  inline agx::Real TerrainMaterial::BulkProperties::getYoungsModulus() const
  {
    return m_youngsModulus;
  }


  inline agx::Real TerrainMaterial::BulkProperties::getPoissonsRatio() const
  {
    return m_poissonsRatio;
  }


  inline agx::Real TerrainMaterial::BulkProperties::getFrictionAngle() const
  {
    return m_frictionAngle;
  }

  inline agx::Real TerrainMaterial::BulkProperties::getDilatancyAngle() const
  {
    return m_dilatancyAngle;
  }

  inline agx::Real TerrainMaterial::BulkProperties::getCohesion() const
  {
    return m_cohesion;
  }

  inline agx::Real TerrainMaterial::BulkProperties::getSwellFactor() const
  {
    return m_swellFactor;
  }

  inline agx::Real TerrainMaterial::BulkProperties::getDeltaReposeAngle() const
  {
    return m_deltaReposeAngle;
  }

  inline agx::Real TerrainMaterial::ParticleProperties::getAdhesionOverlapFactor() const
  {
    return m_adhesionOverlapFactor;
  }

  inline agx::Real TerrainMaterial::CompactionProperties::getDilatancyAngleScalingFactor() const
  {
    return m_dilatancyAngleScalingFactor;
  }
}

