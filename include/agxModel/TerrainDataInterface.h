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

#ifndef AGXMODEL_TERRAIN_DATA_INTERFACE_H
#define AGXMODEL_TERRAIN_DATA_INTERFACE_H

#include <agxModel/export.h>


#include <agxCollide/Geometry.h>
#include <agxCollide/Contacts.h>
#include <agx/Material.h>

#include <agx/FrictionModel.h>


namespace agxCollide
{
  class HeightField;
}

namespace agxSDK
{
  class Simulation;
}

#define MAX_NUMBER_MATERIALS 256

namespace agxModel
{
  class Terrain;

  /**
  \deprecated This class is deprecated and will be removed in a future version of AGX. \sa agxTerrain::Terrain
  Class for storage of contact point info that will help to calculate volume of heightfield overlaps
  */
  class ImpactVolumeData : public agx::Referenced
  {
    public:
      ImpactVolumeData();

      size_t getLocalVertexIndexWithLargestInfluence() const;

      agx::Vec3 point;
      agx::Vec3 normal;
      agx::Real depth;
      agx::Vec3 percent;
      agx::Real normalForce;
      agx::Vec3i vectorIndices;
      agxCollide::ContactPoint* contactPoint;
      agx::Vec2i coordinates[3];
      agx::Vec2  positions[3];

      void clear( );

  };

  typedef agx::ref_ptr< ImpactVolumeData > ImpactVolumeDataRef;

  typedef agx::Vector< ImpactVolumeDataRef > ImpactVolumeDataRefVector;

  typedef agx::Vector< ImpactVolumeData* > ImpactVolumeDataPtrVector;

  typedef agx::Vector< agx::ref_ptr< const agx::Material > > ConstMaterialRefVector;

  typedef std::pair<  agx::ref_ptr<const agx::Material>, agx::ref_ptr<const agx::Material> > ConstMaterialRefPair;

  typedef agx::HashTable< size_t, ConstMaterialRefPair > SizeTConstMaterialRefPairTable;

  typedef agx::Vector< size_t > SizeTVector;

  typedef agx::Vector< const agx::ContactMaterial* > ConstContactMaterialVector;

  /**
  This class implements an interface for how the terrain accesses the data it needs to access
  data items for each grid point of the terrain.

  This class is responsible for holding and allowing a Terrain to access data for
  writing and reading.

  It holds data for:

  - Mapping all gridpoints to a pair of agx::Material.
    The first material in the material pair is for the Terrain and second is for the particles created
    at a position on the Terrain. The difference between the materials is the adhesion value.

  - Mapping all grid points to a deformability value. A value describing how fast a material becomes harder
    due to deformation.

  - Storage of deformation at each grid point.


  User may set:

  - The MaxYoungsChangePerTimestep. At big penetrations, we don't want large deformations. This is a value
    determining the maximum number of times the ground can become harder(youngs modulus), during one timestep.

  - The ShovelFrictionMultiplier. When a shovel is cutting through the terrain, the friction is measured using
    the friction coefficient from the shovel/Terrain contact material, and a normal force found from the
    penetration depth, youngs modulus and adhesion. This multiplier could tweak that value, for wanted behavior.

  - Materials as ExternalMaterialsInteractingWithParticles. For example wheel-particle interaction could is something
    of low interest. Then, by adding the wheel material as an external material interacting with particles, we get
    iterative friction, and less time in the solver. The bucket/shovel-material maybe also could be added, but
    that means there is possible penetration through the bucket/shovel. But by making the shovel geometries thick,
    that might be avoided.

  */
  class AGXMODEL_EXPORT TerrainDataInterface : public agx::Referenced, public agxStream::Serializable
  {
    public:

      /**
      \param heightField            - size of this will give the internal data structures a correct size.
      \param defaultTerrainMaterial - If there are grid points which the user has not given any material, they will get this material.
      */
      TerrainDataInterface(  const agxCollide::HeightField* heightField, const agx::Material* defaultTerrainMaterial = nullptr );

      /**
      Adds a material to the terrain.
      \param material         - a material for a positions at the Terrain
      \param materialIndex    - will map all grid points in the materialIndexMap with this positive integer to material
      \param particleAdhesion - the adhesion between particles created at the grid points with materialIndex
      \param deformability    - agxData::Value of how fast the terrain gets harder at grid points with materialIndex
      \param angleOfRepose    - the angle of repose of the material. (for piles for example), compressed ground has larger angle (scaled linearly with hardness from compression, and wetness of material)
      */
      void add( const agx::Material* material, const size_t materialIndex, const agx::Real particleAdhesion = agx::Real(0), const agx::Real deformability = agx::Real(0), agx::Real angleOfRepose = agx::Real(0) );

      /**
      Removes a material from the terrain.
      */
      void remove( const agx::Material* material );

      /**
      Maps materials to grid points. Sets values between 0 and MAX_NUMBER_MATERIALS to all grid points
      \param filePath - path to .PNG file
      */
      virtual void setMaterialMapping( const std::string& filePath );

      /**
      Maps materials to grid points. Sets values between 0 and MAX_NUMBER_MATERIALS to all grid points
      \param x - x position in height field.
      \param y - y position in height field.
      \param materialIndex - index of material in m_indexToMaterial hash table.
      */
      void setMaterialMapping( const size_t x, const size_t y, const size_t materialIndex );

      /**
      Due to high velocities we may have deep Terrain penetrations. The terrain is deformed using the depth of the contacts.
      We need a rule for how much the terrain can deform during one timestep. Here a maxium value of the increase of stiffness can be set.
      \param maxScaleOfYoungsOneTimestep - the number of times youngs modulus can increase during one timestep
      */
      void setMaxYoungsChangePerTimestep( agx::Real maxScaleOfYoungsOneTimestep );

      /**
      When a shovel is cutting through the terrain,
      the user has the ability to set a minimum force that is needed to cut through the ground.
      */
      void setShovelGroundCuttingForce( agx::Real force );

      /**
      \return the ShovelGroundCuttingForce
      */
      agx::Real getShovelGroundCuttingForce() const;

      /**
      Returns begin iterator to the materials on this terrain
      */
      SizeTConstMaterialRefPairTable::const_iterator getMaterialIndexBeginIterator() const;

      /**
      Returns end iterator to the materials on this terrain
      */
      SizeTConstMaterialRefPairTable::const_iterator getMaterialIndexEndIterator() const;

      /**
      Add an external material to the terrain. (Make terrain know about it)
      Add those materials that will collide with particles.
      These materials will have solve type SPLIT against particles created by digging in the terrain.
      For all materials not added, the contact material between them and the particles created by the terrain will be implicit,
      and very soft.
      */
      void addExternalMaterial( agx::Material* material, agx::Real surfaceViscosity = agx::Real(1E-3) );

      /**
      Set the solve type of the particles when they interact with each other.
      The solveType will define the type that the friction will get between particles.
      ITERATIVE, SPLIT or DIRECT (default is ITERATIVE)
      */
      void setParticleFrictionSolveType( agx::FrictionModel::SolveType solveType );

      /**
      \returns the solve type of the particles when they interact with each other.
      */
      agx::FrictionModel::SolveType getParticleFrictionSolveType() const;

      /**
      \returns the current deformation at position x,y (length units)
      */
      virtual agx::Real getCompression( size_t x, size_t y ) const;

      /**
      Set the compression at a specified grid point (x,y)
      \param x,y coordinates of grid point
      \param val - New value to be set at specified grid point.
      */
      virtual void setCompression( size_t x, size_t y , const agx::Real val );

      /**
      \param deformability       - value of how easy it is to deform the terrain locally.
      \param oldCompression      - the previous compression at that local position.
      \param maxYoungsMultiplier - the maximum number of times harder the ground is allowed to become due to a new impact.
      \returns calculated value (in length units) of how much the ground could be compressed at a certain vertex.
      */
      agx::Real getPossibleCompression( agx::Real deformability, agx::Real oldCompression, agx::Real maxYoungsMultiplier );

      /**
      Set how much harder than default Young's modulus the ground can get by being compressed?
      \param maxCompressionMultiplier - the number of times harder, must be >= 1 and a multiple of 2.(2,4,8,16,32,64...). Will be clamped to the nearest integer that gives = log2(maxCompressionMultiplier^2)
      */
      void setMaxCompressionMultiplier( agx::Real maxCompressionMultiplier );

      /**
      \returns how much harder than default Young's modulus the ground can get by being compressed?
      */
      agx::Real getMaxCompressionMultiplier() const;

      /**
      Set the depth for maximum compression.
      When digging, the ground is more compressed the deeper we get. Here we can set at which depth the ground gets maximum compression
      */
      void setDepthForMaxCompression( agx::Real depthForMaxCompression );

      /**
      \returns the depth for maximum compression.
      */
      agx::Real getDepthForMaxCompression() const;

      /**
      Set the inverse friction multiplier. (This is only used for directly solved particles).
      Since the friction coefficient is infinite when using directly solved friction, we scale the viscosity instead.
      The viscosity has an inverse relation to how large the friction force can get.
      Setting the inverse friction multiplier to zero will give infinitely dry friction. All other values will give some sort of viscous friction.
      The higher value the less friction.
      */
      void setInverseFrictionMultiplier( agx::Real inverseFrictionMultiplier );

      /**
      returns the inverse friction multiplier.
      */
      agx::Real getInverseFrictionMultiplier( ) const;

      /**
      \param x - grid x coordinate
      \param y - grid y coordinate
      \returns agx::Material for particle to be created at grid point [x,y]
      */
      const agx::Material* getParticleMaterial( const size_t x, const size_t y ) const;

      /**
      \param materialIndex - the index that maps to the wanted agx::Material for particles
      \return agx::Material for particle to be created at grid point with materialIndex
      */
      const agx::Material* getParticleMaterial( const size_t materialIndex ) const;

      /**
      \param x - grid x coordinate
      \param y - grid y coordinate
      \return deformability at grid point [x,y]
      */
      agx::Real getDeformability(const size_t x, const size_t y) const;

      /**
      \param x - grid x coordinate
      \param y - grid y coordinate
      \return angle of repose at grid point [x,y]
      */
      agx::Real getTanOfTheAngleOfRepose(const size_t x, const size_t y) const;

      /**
      \param x - grid x coordinate
      \param y - grid y coordinate
      \return material index at coordinates[x,y]
      */
      size_t getMaterialIndex(const size_t x, const size_t y) const;

      /**
      \param x - grid x coordinate
      \param y - grid y coordinate
      \return agx::Material for Terrain at grid point [x,y]
      */
      const agx::Material* getMaterial(const size_t x, const size_t y) const;

      /**
      \param materialIndex - the index that maps to the wanted agx::Material for Terrain
      \return agx::Material for Terrain at grid point with materialIndex
      */
      const agx::Material* getMaterial(const size_t materialIndex) const;

      /**
      \return the resolution in X of the TerrainData
      */
      size_t getResolutionX() const;

      /**
      \return the resolution in Y of the TerrainData
      */
      size_t getResolutionY() const;

      /**
      \return the Number of grid points for the TerrainData
      */
      size_t getNumberOfVertices() const;

      /**
      \return true if resolution is != 0.
      */
      bool isValid() const;

      /**
      \return an array index given two grid coordinates.
      */
      size_t getVectorIndex( size_t x, size_t y ) const;

      /**
      Get the x,y coordinate of the i:th point in the grid (starting from 0,0)
      */
      void getCoordinates( size_t i, size_t& x, size_t& y ) const;

      /**
      Get the x,y coordinate of the i:th point in the grid (starting from 0,0)
      */
      void getCoordinates( size_t i, int& x, int& y ) const;

      /**
      Calculate the YoungsModulus value for a given grid point.
      - If grid point is invalid, the defaultYoungsModulus will be returned
      \param x, y                 - Coordinate of grid point
      \param defaultYoungsModulus - the value of Young's modulus, if the ground was not deformed
      \param deformability        - value of how fast the ground gets harder
      \return the current YoungsModulus value for the requested grid point.
      */
      virtual agx::Real calculateYoungsModulus(size_t x, size_t y, agx::Real defaultYoungsModulus, agx::Real deformability, agx::Real localDepth = agx::Real(0)) const;

      AGXSTREAM_DECLARE_SERIALIZABLE(agxModel::TerrainDataInterface);

    protected:
      TerrainDataInterface();

      virtual ~TerrainDataInterface();

            /**
      Set the number of time steps left for avalanches at a vertex
      \param vectorIndex - index for vertex
      \param value - the number of time steps left for triggering an avalanche
      */
      void updateAvalanceHeatMap( const size_t vectorIndex, const size_t value );
      void updateAvalanceHeatMap( const size_t x, const size_t y, const size_t value );

      /**
      integrate heat map
      */
      void tickDownAvalancheHeatMap();

      size_t getAvalancheHeatMapSize();

      size_t getAvalancheHeatMapValue( const size_t index );

      /**
      \param material         - material for Terrain.
      \param particleAdhesion - wanted adhesive force between particles created from Terrain at gridpoint with material.
      \return                 - new agx::Material, identical to material, except for the adhesion value
      */
      virtual agx::MaterialRef createParticleMaterial( const agx::Material* material, const agx::Real particleAdhesion );

      agx::Real calculateSurfaceViscosity( agx::Real frictionCoefficient );

      agx::ContactMaterialRef createParticleContactMaterial( agxSDK::Simulation* simulation, const agx::Material* mat1, const agx::Material* mat2, const bool adhesiveForce, agx::Real adhesiveOverlap, agx::Real youngsParticleMultiplier, agx::Real surfaceViscosity = agx::Real(0), agx::FrictionModel::SolveType frictionSolveType = agx::FrictionModel::SPLIT );

      agx::RigidBodyRefVector& getDynamicHeightFieldParticles();

      void allocateDynamicHeightFieldParticles( Terrain* terrain);

      void disableDynamicHeightFieldParticles( size_t numberOfUsedParticles );

      bool hasParticles();

      agx::RigidBody* getParticle();

      void regretParticleFromBack( size_t indexFromBack );

      //When the height shrinks at a vertex due to digging, the ground below harder, the deltaHeightShrink and the m_depthForMaxCompression parameter
      //will give a new compression. Also the compression variable tells how much the ground is compressed due to the compressed ground (overlap with geometry)
      /**
      \param x                        - vertex X height field position
      \param y                        - vertex Y height field position
      \param deltaDiggingHeightShrink - height change due to digging (could be negative, which leads to less compressed ground, when particles merge or soil is pusched TO (x,y) vertex)
      \param compressionFromOverlap   - the compression of the geometry impact.
      */
      void updateCompression( size_t x, size_t y, agx::Real deltaDiggingHeightShrink, agx::Real compressionFromOverlap );

      friend class Terrain;

      /**
      Uses all found contactMaterials for a geometry contact, and creates a temporary contact material.
      \param contactMaterials   - vector containing all found contact materials (usually for a set of contact points)
      \param newCM              - a temporary contact material (with isInternal == true) to modify.
      \param totalYoungsModulus - the total sum of all Young's modulus at the very same contact points where we found
                                  the contact materials. But the values summed up is found using calculateYoungsModulus() (above)
      \return the first found FrictionModel held by any of the contactMaterials.
      */
      virtual const agx::FrictionModel* calculateContactMaterialProperties(
        ConstContactMaterialVector& contactMaterials, agx::ContactMaterial& newCM, const agx::Real totalYoungsModulus);

      /**
      2^( 1 + lastDeformation * deformability ) = newYoungs / lastYoungs
      we have set maxvalue of RHS

      Scale lastDeformation if necessary. If maxYoungsChange is set to zero, the terrain can't be modified
      \param lastDeformation - the deformation due to geometry overlap
      \param deformability - the deformability value for the vertex with the geometry overlap
      \return - the scaled deformation value
      */
      virtual agx::Real scaleDeformation( agx::Real lastDeformation, agx::Real deformability ) const;

      /**
      Clear existing contact materials created for particles
      */
      void removeExistingParticleParticleContactMaterials( agxSDK::Simulation* simulation );

      /**
      Create contact materials for everything interacting with particles (including particle-particle contact materials)
      return nullptr if no material was created
      */
      void createContactMaterials( agxSDK::Simulation* simulation, agx::Real adhesiveOverlap, agx::Real particleYoungsMultiplier );

      /**
      Set the deformation at all grid points
      \param val - New value to be set at specified grid point.
      */
      void setCompression( agx::Real val);

      /**
      returns true if DEFORMABILITY or YOUNGS_SCALER has been set
      */
      bool getMaterialsAreDirty() const;

      /**
      Resets the number of impactVolumeDatas used. Pool is still of same size.
      */
      void clearImpactVolumeData( );

      /**
      Calculate a multiplier for how much harder the ground gets given a compression and its deformability.
      \param deformability - the local terrain deformability(see terrain.h for definition)
      \param compression - how compressed the ground got (length units).
      */
      static agx::Real compressionMultiplier( agx::Real deformability, agx::Real compression );

      /**
      Calculates ImpactVolumeData from a given contact point
      */
      ImpactVolumeData* getImpactVolumeData( agxCollide::ContactPoint* cp, const Terrain* terrain );
      bool getData( const agxModel::Terrain* terrain, const agx::Vec3 pos, ImpactVolumeData& ivd ) const;
      bool getData( const agxModel::Terrain* terrain, const agxCollide::ContactPoint& cp, ImpactVolumeData& ivd ) const;
      bool getTriangleIndexPoints( const Terrain* terrain, const agx::Vec3 point, ImpactVolumeData& ivd ) const;
      bool collectNearbyVertices( const Terrain* terrain, ImpactVolumeData& ivd ) const;


    protected:

      ImpactVolumeDataRefVector m_impactVolumeDataStorage;

      agx::RealVector m_compression;
      SizeTVector m_materialIndexMap;
      SizeTVector m_avalancheHeatMap;
      agx::RigidBodyRefVector m_dynamicHeightFieldParticles;

      size_t m_numberOfUsedParticles;

      size_t m_numberOfUsedImpactVolumes;

      size_t m_resolutionX;
      size_t m_resolutionY;

      agx::Real m_maxYoungsChangePerTimestepPOW2;
      agx::Real m_shovelGroundCuttingForce;

      bool m_materialsAreDirty;

      agx::FrictionModel::SolveType m_particleFrictionSolveType;

      agx::Real m_inverseFrictionMultiplier;

      agx::Real m_maxCompressionMultiplier;

      agx::Real m_depthForMaxCompression;

      SizeTConstMaterialRefPairTable m_indexToMaterial;

      ConstMaterialRefVector m_externalMaterialsInteractingWithParticles;
      agx::RealVector m_externalMaterialSurfaceViscosity;

      agx::RealVector m_materialIndexToDeformability;
      agx::RealVector m_materialIndexToTanOfAngleOfRepose;
  };
  typedef agx::ref_ptr<TerrainDataInterface> TerrainDataInterfaceRef;

}

#endif //AGXMODEL_TERRAIN_DATA_INTERFACE_H
