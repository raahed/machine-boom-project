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

#ifndef AGXMODEL_TERRAIN_PARTICLES_H
#define AGXMODEL_TERRAIN_PARTICLES_H

#include <agxModel/export.h>


#include <agxCollide/Contacts.h>
#include <agx/Material.h>

#include <agxCollide/agxCollide.h>


#define TERRAIN_USE_PARTICLE_SYSTEM 0

#if TERRAIN_USE_PARTICLE_SYSTEM
#include <agx/ParticleSystem.h>
#endif

namespace agxModel
{
  class Terrain;

  typedef agx::Vec3 ParticleCoordinateHeightData;

  typedef agx::Vector< std::pair< agx::Vec2i, ParticleCoordinateHeightData > > Vec2iHeightDataPairVector;

  typedef agx::Vector< std::pair < agx::Real, size_t > > RealSizeTPairVector;

  typedef agx::Vector< std::pair<size_t, size_t> > SizeTPairVector;

  typedef agx::HashVector< size_t, agx::Real > SizeTRealHashVector;

  /**
  \deprecated This class is deprecated and will be removed in a future version of AGX. \sa agxTerrain::Terrain
  Class to handle terrain digged up particles.
  */
  class AGXMODEL_EXPORT ParticleAttributes : public agxStream::Serializable
  {

    public:

      /**
      Enum for type of soil.
      */
      enum SoilType {
        PARTICLE,                  /**< */
        PARTICLE_SYSTEM_PARTICLE,  /**< */
        CONVEX_BODY                /**< */
      };

      ParticleAttributes();

      /**
      Set the type of soil to be created when it's time to create particles.
      */
      void setType( SoilType soilType );

      /**
      \returns the current choice of SoilParticle type.
      */
      SoilType getType( );

      /**
      Set unique Id for soil particles. To make it possible to disable against other groups.
      */
      void setParticleUniqueId( agx::UInt32 uniqueParticleId );

      /**
      \returns unique particle id for disabling collisions against them (for example).
      */
      agx::UInt32 getParticleUniqueId() const;

      /**
      Set how much stiffer the particle Young's modulus to be than the terrain material.
      The terrain material is usually very soft, and if the particle contacts are solved iteratively that will make them behave even softer.
      */
      void setParticleYoungsModulusMultiplier(agx::Real multiplier);

      /**
      \returns how many times stiffer the particles should be than the actual Young's modulus for a material.
      */
      agx::Real getParticleYoungsModulusMultiplier() const;

      /**
      Set the merge velocity for soil particles. I.e the minimum speed of particles before they can be merged with the terrain.
      */
      void setMergeVelocity( agx::Real velocity );

      /**
      \returns the square of the merge velocity..
      */
      agx::Real getMergeVelocity2() const;

      /**
      \returns the adhesive overlap used for the material of the soil particles.
      */
      agx::Real getAdhesiveOverlap( ) const;

      /**
      \returns the outer radius for the soil particles.
      */
      agx::Real getRadius( ) const;

      /**
      \returns the core radius of soil particles (outer radius minus the adhesion).
      */
      agx::Real getCoreRadius( ) const;

      /**
      \returns the volume a soil particle will represent.
      */
      agx::Real getVolume( ) const;

      /**
      Set the soil particle radius.
      The radius is controlled by agxModel::Terrain through agxModel::TerrainDataInterface.
      The spheres are conformed in an FCC lattice (terrain fixes this).
      The core radius should be set so that when introduced particles are mathematically just touching.
      The radius can be set to a value leq to the height field vertex distance.
      */
      void setRadius( agx::Real radius);

      /**
      Set the particle core radius.
      Changing the core radius will effect the dynamics of the SoilParticles.
      User could set core radius explicitly for wanted particle behavior.
      The possible adhesion overlap distance will change.
      */
      void setCoreRadius( agx::Real coreRadius );

      /**
      \returns true if the radius has been changed.
      */
      bool getRadiusIsDirty( ) const;

      AGXSTREAM_DECLARE_SERIALIZABLE(agxModel::ParticleAttributes);

    protected:
      friend class Terrain;
    private:

      void setRadiusIsDirty( bool dirty );

      SoilType m_type;
      agx::Real m_radius;
      agx::Real m_coreRadius;
      agx::Real m_mergeVelocity2;
      bool m_radiusIsDirty;
      agx::UInt32 m_uniqueParticleId;
      agx::Real m_particleYoungsMultiplier;
  };

  /**
  When converting Terrain to particles, we create SoilParticles.
  Its geometry/ies will have propertyBool "SoilParticle"
  Their type could be set PARTICLE or COMPOSITE_SPHERE
  */
  class AGXMODEL_EXPORT SoilParticle : public agx::RigidBody
  {
    public:

      /// Specifies the type of the SoilParticle
      SoilParticle( agx::Real radius, ParticleAttributes::SoilType type );

      /**
      \returns which type of soil this is.
      */
      ParticleAttributes::SoilType getType() const;

      /**
      \returns the soil particle radius.
      */
      agx::Real getRadius( ) const;

      /**
      \returns the volume of the soil particle.
      */
      virtual agx::Real getVolume( ) const;

      /**
      Set a parent to the particle in a soil particle contact graph.
      This is used if the soil particle simulation is activated. if ( USE_SOIL_PARTICLE_SIMULATION == 1 )
      */
      void setParent( SoilParticle* parent );

      /**
      Set that this particle is a seed.
      */
      void setSeedContact( );

      /**
      \returns true if this is a seed.
      */
      bool getSeedContact( ) const;

      /**
      \returns the parent to this particle.
      */
      SoilParticle* getParent( ) const;

      /**
      \returns true if this particle has a parent.
      */
      bool hasParent() const;

      /**
      \returns true if this particle is a parent.
      */
      bool isParent() const;

      AGXSTREAM_DECLARE_SERIALIZABLE(agxModel::SoilParticle);

    protected:
      SoilParticle() {}
      ParticleAttributes::SoilType m_type;
    private:
      agx::Real m_radius;
      SoilParticle* m_parent; //particles
      bool m_seedContact;

  };

  typedef agx::ref_ptr< SoilParticle > SoilParticleRef;
  typedef agx::observer_ptr< SoilParticle > SoilParticleObs;

  class AGXMODEL_EXPORT ConvexSoil : public SoilParticle
  {
    public:
      /**
      Create a convex body representing digged up soil.
      */
      ConvexSoil( const agx::Vec3Vector* vertices, const agx::UInt32Vector* indices );

      /**
      \returns the volume of the convex soil.
      */
      virtual agx::Real getVolume( ) const override;

      AGXSTREAM_DECLARE_SERIALIZABLE(agxModel::ConvexSoil);
    protected:
      ConvexSoil() {}
    private:

  };

#if TERRAIN_USE_PARTICLE_SYSTEM
  typedef agx::VectorPOD< agx::Physics::ParticlePtr > SoilParticleObsVector;
  typedef SoilParticleObsVector SoilParticleRefVector;
  typedef SoilParticleRefVector SoilParticlePtrVector;
#else
  typedef agx::Vector< SoilParticleObs > SoilParticleObsVector;
  typedef agx::Vector< SoilParticleRef > SoilParticleRefVector;
  typedef agx::Vector<SoilParticle*> SoilParticlePtrVector;
#endif

}

#endif //AGXMODEL_TERRAIN_PARTICLES_H
