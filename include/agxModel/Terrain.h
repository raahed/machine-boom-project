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

#ifndef AGXMODEL_TERRAIN_H
#define AGXMODEL_TERRAIN_H

#include <agxModel/export.h>


#include <agxSDK/StepEventListener.h>
#include <agxCollide/Geometry.h>
#include <agxCollide/Contacts.h>
#include <agx/Material.h>

#include <agx/FrictionModel.h>

#include <agxCollide/agxCollide.h>
#include <agxSDK/Simulation.h>
#include <agx/Range.h>

#include <agxModel/TerrainDataInterface.h>
#include <agxModel/TerrainParticles.h>
#include <agxModel/TerrainUtils.h>
#include <agxModel/SoilParticleSimulation.h>


namespace agxCollide
{
  class HeightField;
}


namespace agxModel
{
  /**

  \deprecated This class is deprecated and will be removed in a future version of AGX. \sa agxTerrain::Terrain
  Terrain contains a agxCollide::Geometry containing one shape - a agxCollide::HeightField.

  Its purpose is to act as deformable terrain. This is implemented by letting the height field's internal height data change.

  Terrain can be deformed in two ways, either by compression or by "shoveling".

  All geometries colliding with the Terrain can compress it.
  Geometries containing property "ShovelUpProperty" and "ShovelForwardProperty" will be able to convert the Terrain into SoilParticles (or rigid bodies).

  The two above mentioned properties are Vec3 properties, and define forward direction (sharp part of shovel) and normal direction (length defines shovel thickness)


  ______________________________________________________
  |                       /\                             |
  |    Shovel             || (a non normalized vector)   |  --------> ShovelForwardProperty (with length of the distance from center of mass to the cutting edge)
  |                       ||  ShovelUpProperty       |
  |______________________________________________________|

  Note! The property vectors are relative to the geometry!

  A combination of multiple shovel geometries is possible, but if there are conflicting forward directions, there could be trouble.

  A Terrain accesses "maps" of terrain data through the TerrainDataInterface. The user is free to implement their own
  specialization of a TerrainDataInterface. The default is TerrainDataInterface which is used if nothing else is specified.

  Geometries having a "DynamicHeightFieldDeformer" property will also be able to "push" mass within the height field to deform it. Adding the DynamicHeightFieldDeformer property bool,
  involves one choice. Will the contact constraints between the deformer and the terrain have contact depth in gravity direction or not.
  Adding the property bool as true, will give contact depths, adding it as false results in contacts always being satisfied (makes objects slowly fall through the ground).
  But for a bucket, you don't like it to be pushed out from a pile of sand/soil. For buckets you choose the DynamicHeightFieldDeformer property to be false.

  */

  class AGXMODEL_EXPORT Terrain : public agxSDK::StepEventListener
  {
    public:

      /**
      Constructor. Create a terrain given a specified height field.
      */
      Terrain( agxCollide::HeightField* heightField );

      /**
      Constructor. Create a terrain given a specified Geometry, having only one shape, A height field.
      The shape should not have any transformation relative to the geometry.
      */
      Terrain( agxCollide::Geometry* terrainGeometry );

      /**
      \returns the Terrain geometry (which is created in Terrain constructor)
      */
      agxCollide::Geometry* getGeometry() const;

      /**
      \returns pointer to data interface of terrain. (one is created by default, but can be replaced)
      */
      agxModel::TerrainDataInterface* getDataInterface() const;

      /**
      \returns the largest particle radius for heightfield scale. Assuming scale.x()==scale.y().
      */
      agx::Real getMaximumParticleRadius() const;

      /**
      Set the cut off (threshold) velocity for an activation of the dynamic height field functionality at a contact/impact.
      */
      void setDynamicHeightFieldCutOffVelocity( agx::Real cutOffVelocity );

      /**
      \returns the cut off (threshold) velocity for activation of the dynamic height field functionality at a contact/impact.
      */
      agx::Real getDynamicHeightFieldCutOffVelocity() const;

      /**
      Set the cut off (threshold) angular velocity for an activation of the dynamic height field functionality at a contact/impact.
      */
      void setDynamicHeightFieldCutOffAngularVelocity( agx::Real cutOffAngularVelocity );
      /**
      \returns the cut off (threshold) angular velocity for activation of the dynamic height field functionality at a contact/impact.
      */
      agx::Real getDynamicHeightFieldCutOffAngularVelocity() const;


#if TERRAIN_USE_PARTICLE_SYSTEM
      agx::ParticleSystem* getParticleSystem() const;
#endif
      /**
      Set a material to the terrain geometry, which will be used for non material-mapped grid points.
      */
      virtual void setMaterial(agx::Material* material);

      /**
      Set TerrainDataInterface. (TerrainDataInterface is used default) but user could implement a class inheriting from TerrainDataInterface.
      */
      void set( agxModel::TerrainDataInterface* dtdc );

      /**
      Use a particle simulation to deform the terrain (replaces pure compression).
      */
      void setEnableDynamicHeightField( bool enable );

      /**
      \returns true if dynamic height field simulation is enabled.
      */
      bool getEnableDynamicHeightField() const;

      /**
      \returns true if geometry qualifies to be a shovel
      */
      bool isGeometryShovel(const agxCollide::Geometry* otherGeometry) const;

      /**
      \returns true if geometry qualifies to be a shovel moving forward
      */
      bool isGeometryShovelMovingForward(const agxCollide::Geometry* otherGeometry, agx::Vec3& shovelForwardW) const;

      /**
      Fixate the height of the height field borders.
      */
      void setEnableLockedBorders( bool enable, bool triggerAvalancheMassLoss = true );

      /**
      Set the vertical Young's modulus area scaler.
      For a geometry with the DYNAMIC_HEIGHT_FIELD_PROPERTY:
       The vertical Young's modulus scaler decides for how large contact area the Young's modulus,
       for a contact material (between the terrain and other), will be defined.
       The Young's modulus is scaled linearly relative to contact area in vertical (gravitational) direction.
      */
      void setVerticalYoungsModulusAreaScaler( agx::Real scaler );

      /**
      \returns the Young's modulus area scaler for the terrain.
      */
      agx::Real getVerticalYoungsModulusAreaScaler() const;

      /**
      Set the horizontal Young's modulus area scaler.
      For a geometry with the DYNAMIC_HEIGHT_FIELD_PROPERTY:
        The horizontal Young's modulus scaler decides for how large contact area the Young's modulus,
       for a contact material (between the terrain and other), will be defined.
       The Young's modulus is scaled NON-linearly relative to contact area in horizontal (NON-gravitational) direction.
       The nonlinearity is due to that the mass that is moved in the horizontal plane is not linearly depending on the contact area.
      */
      void setHorizontalYoungsModulusAreaScaler( agx::Real scaler );

      /**
      \returns the horizontal Young's modulus area scaler for the terrain.
      */
      agx::Real getHorizontalYoungsModulusAreaScaler() const;

      /**
      Goes through all vertices 'iterations' times and triggers avalanches.
      Use this to relax for example an imported height field.
      */
      void avalancheAll( size_t iterations );

      /**
      Set the wetness scale.
      Scale between 0 and 1 (1 == 100%)
      The increment of the angle of repose due to compression will be scaled with the wetness scale.
      The material will at zero (0) wetness always have the same angle of repose.
      At wetness == 1, the avalanche algorithm will use an angle of repose that is linearly scaled using the amount of compression of the ground.
      */
      void setWetnessScale( agx::Real wetnessScaler );

      /**
      \returns the of wetness of the terrain.
      */
      agx::Real getWetnessScale() const;

      /**
      Set the avalanche decay percent.
      The terrain will start avalanches everywhere a deformer (geometry with DYNAMIC_HEIGHT_FIELD_PROPERTY )has been (and is).
      The avalancheDecayPercent will decide how large percent of the height of a peak, that will "slide" during one time step.
      The height will change until the height difference with all neighbors satisfies the angle of repose.
      */
      void setAvalancheDecayPercent( agx::Real scaler );

      /**
      \returns the avalanche decay percent.
      */
      agx::Real getAvalancheDecayPercent() const;

      /**
      Set the number of time steps the avalanche will continue at a vertex after the last time it was deformed.
      */
      void setAvalancheStepLifeTime( size_t numSteps );

      /**
      \returns the number of time steps the avalanche will continue at a vertex after the last time it was deformed.
      */
      size_t getAvalancheStepLifeTime() const;

      /**
      Set how many vertices away from a geometry with deform property an avalanche could be triggered.
      */
      void setAvalancheVertexDistanceExtent( size_t numVertices );

      /**
      \returns how many vertices away from a geometry with deform property an avalanche could be triggered.
      */
      size_t getAvalancheVertexDistanceExtent() const;

      /**
      Set the angle between the Shovel forward vector and the height field "up" direction (gravitational direction by definition).
      When digging, the particles appear ni the volume above the shovel (in shovel up/normal direction).
      This volume is zero when the shovel forward direction points in vertical direction.
      Here it is possible to set at which angle particles could appear. (default is PI/10)
      */
      void setShovelMinimumDigAngleToHorizontalPlane( agx::Real angle );

      /**
      \returns the angle between the Shovel forward vector and the height field "up" direction (gravitational direction by definition).
      */
      agx::Real getShovelMinimumDigAngleToHorizontalPlane() const;

      /**
      Returns the relative velocity between the terrain and a point on a geometry. Assuming the terrain has no angular velocity.
      */
      agx::Vec3 pointRelativeVelocityLocal(const agxCollide::Geometry* shovelGeometry, const agx::Vec3 pointWorld) const;

      /**
      Finds the relative velocity between a point on the shovel and the Terrain, along the normalLocal direction.
      */
      agx::Real calculateShovelContactDepth(const agxCollide::Geometry* shovelGeometry, const agx::Vec3 normalLocal, const agx::Vec3 pointWorld) const;

      /**
      \returns pointer to the terrain height controller.
      */
      const agxModel::HeightController* getHeightController() const;

      /**
      \returns pointer to the terrain particle attributes.
      Use ParticleAttributes to set/get radius, core radius, volume and type for soil particles.
      like setRadius(), setType() (type could be SoilParticle::PARTICLE or SoilParticle::COMPOSITE_SPHERE)
      */
      const ParticleAttributes& getParticleAttributes() const;
      ParticleAttributes& getParticleAttributes();

      /**
      \returns radius of the particles used in the dynamic height field simulation.
      */
      agx::Real getDynamicHeightFieldParticleRadius() const;

      /**
      \returns a reference to a vector containing all indices in the height field where the height, deformation or deformability has been changed
       could be that it is only height that is interesting, but maybe rendering is deformation dependent?
       Anyhow, deformation should not be changed if the height is not also changed. If the user doesn't set it deformation explicitly.
       Softness will also only be set by the user.
      */
      const agx::Vec2iVector& getModifiedIndices() const;

      /**
      \returns a reference to a vector containing all created particles during the last Terrain update.
                This vector is cleared in the beginning of a time step (PRE_EVENT)
      */
      const SoilParticleObsVector& getCreatedParticles() const;

      /**
      \returns a reference to a vector containing all removed particles (particles that have merged with the Terrain) during the last time step.
          This vector is cleared in the beginning of a time step (PRE_EVENT)
      */
#if TERRAIN_USE_PARTICLE_SYSTEM
      const agx::VectorPOD<agxData::EntityPtr>& getRemovedParticles() const;
#else
      const SoilParticleRefVector& getRemovedParticles() const;
#endif
      const agxCollide::HeightField* getHeightField( ) const;

      /**
      \returns pointer to the simulation of the dynamic height field.
      */
      const DynamicHeightFieldSimulation* getDynamicHeightFieldSimulation() const;

      /**
      Transform positions/vectors to/from terrain/world
      */
      agx::Vec3 transformPointToTerrain( const agx::Vec3 point ) const;
      agx::Vec3 transformPointToWorld( const agx::Vec3 point ) const;
      agx::Vec3 transformVectorToTerrain( const agx::Vec3 vector ) const;
      agx::Vec3 transformVectorToWorld( const agx::Vec3 vector ) const;

      /// Function below must be public. But should not be called from external functions.

      /**
      Finds all geometry contacts including the Terrain geometry and introduces temporary contact materials for them.
      Finds Particles to merge.
      */
      virtual void pre(const agx::TimeStamp&) override;

      /**
      Finds terrain vertices to shrink, and thereby introduce particles at the correct positions.
      Merges particles with the terrain (found in ::pre() ).
      */
      virtual void post(const agx::TimeStamp&) override;

      /**
      Adds terrain geometry to simulation.
      */
      virtual void addNotification() override;

      /**
      Removes terrain geometry from space, but not simulation.
      */
      virtual void removeNotification() override;

#if defined(USE_SOIL_PARTICLE_SIMULATION) && USE_SOIL_PARTICLE_SIMULATION == 1
      /**
      \returns pointer to the soil particle simulation.
      */
      SoilParticleSimulation* getParticleSimulation();
#endif
      AGXSTREAM_DECLARE_SERIALIZABLE(agxModel::Terrain);

    protected:

      Terrain();
      Terrain( const Terrain& original );
      virtual ~Terrain();

    private:

      struct ShovelPlane
      {
      public:
        size_t hfXorigin;
        size_t hfYorigin;
        agx::Real hfZorigin;
        agx::Vec3 normal;

        agx::Real getHeight(const agx::Real hfX, const agx::Real hfY) const
        {
          agx::Real dx = hfX - agx::Real(hfXorigin);
          agx::Real dy = hfY - agx::Real(hfYorigin);

          if (agx::equalsZero(dx) && agx::equalsZero(dy))
            return hfZorigin;

          agx::Vec3 inHfPlaneVector = agx::Vec3(dx, dy, 0);
          agx::Vec3 inHfPlaneOrthoVector = inHfPlaneVector - (inHfPlaneVector*normal) * normal;

          const size_t variable = agx::equalsZero(dx) ? 1 : 0;

          agx::Real percentOfActual = inHfPlaneOrthoVector[variable]/inHfPlaneOrthoVector[variable];

          agx::Real actualHeight = inHfPlaneOrthoVector.z() / percentOfActual;

          return actualHeight;
        }

        agx::Bool isAbovePlane(agx::Vec3 hfPos, const agx::Real particleRadius) const
        {
          agx::Real planePosAtCoordinates = getHeight(hfPos.x(), hfPos.y());
          return hfPos.z() > (planePosAtCoordinates + particleRadius);
        }

      };

      void prepareForContactModification();

      /**
      Introduces particle (when digging) and adds it to m_createdParticles.
      */
      SoilParticle* createParticle( agx::Real radius );
      SoilParticle* getParticle( const size_t& x, const size_t& y, agxSDK::Simulation* simulation, bool forDynamicHeightField );

      void createTerrain();

      agxCollide::HeightField* _getHeightField();

      void replaceDynamicHeightFieldParticles();

      agx::Real calculateTotalVolume( agx::Vector< size_t >& allChangedVertices );

      void moveVolumeFromParticleMovement( agx::Vector<size_t>& grownIndices, agx::RigidBodyRefVector& bodies, const size_t numGeometriesColliding, agx::Real totalVolume, const agx::RigidBody* deformerBody );

      void simulateDynamicHeightField( const agxCollide::Geometry* geometry, agx::Vector< size_t >& localShrinkedVertices);

      void fakeSimulatedDynamicHeightField( const agxCollide::Geometry* geometry, agx::Vector< size_t >& localShrinkedVertices );

      void createProjectedGeometryContact(
        const agxCollide::GeometryContact* geometryContact, agx::ContactMaterial* contactMaterialGravityDirection,
        const agx::Vec3 gravityDirection, const bool inGravityDirection, const bool useGravityDepth);

      void addParticleParticleGeometryContact( size_t index1, size_t index2, agx::RigidBodyRefVector& bodies );

      /**
      Go through a bunch of criteria for merging a SoilParticle
      */
#if TERRAIN_USE_PARTICLE_SYSTEM
      bool validateParticleToMerge(agx::Physics::ParticlePtr particle);
#else
      bool validateParticleToMerge(SoilParticle* particle) const;
#endif
      bool handleParticleContactMaterial(SoilParticle* sp, agxCollide::GeometryContact* geometryContact, agxCollide::Geometry* otherGeometry);


      void splitDeformingGeometryContact(DeformingGeometryContactProperties& dgp, agxCollide::GeometryContact* geometryContact, ConstContactMaterialVector& contactMaterials, agx::ContactMaterial* materialForFrictionOnly);

      void createGeometryContactGivenContactPoint(const agxCollide::GeometryContact* parentGeometryContact, ContactPointData& cpd, const agxCollide::ContactPoint* cp,const agx::ContactMaterial* contactMaterial);

      void createOldShovelDefinition(DeformingGeometryContactProperties& dgp, agxCollide::GeometryContact* geometryContact, ConstContactMaterialVector& contactMaterials, agx::ContactMaterial* contactMaterial);

      void resetNoGrowIndices();

      void updateAvalanceIndices(size_t minX, size_t maxX, size_t minY, size_t maxY);

      //Does the vertex have a body compressing int (or hovering above eps away?)
      bool isNoGrowIndex( size_t index, agx::Real eps = agx::Real(0.1));

      agx::Real findReposeHeight( size_t indexHF, size_t x, size_t y, agx::Real compression, agx::Real hfScale ) const;

      /**
      Will spread an avalanche at vertex in height field.
      \param indexHF        - index in height field
      \param vertexDistance - how many vertices away this should propagate recursively.
      */
      void spreadAvalanche( size_t indexHF, const size_t vertexDistance );

      void avalanche();

      agx::ContactMaterial* createContactMaterial(const agx::ContactMaterial* originalCM, const agx::Real& newYoungs);

      /**
      Create a temporary (internal) "mean" contact material using a vector of existing contact materials and the sum of their calculated youngs moduluses.
      */
      agx::ContactMaterial* createMeanMaterial( ConstContactMaterialVector& contactMaterials, const agx::Real totalYoungsModulus );

      /**
      Validates a 3 Terrain grid points, to see if they are inside an area of any shovel projected to the Terrain.
      */
      bool isInsideNonMergeArea( agx::Vec2i* coordinates );

      /**
      Validates one Terrain grid point, to see if they are inside an area of any shovel projected to the Terrain.
      */
      bool isInsideNonMergeArea( agx::Vec2i& coordinates );

      /**
      Given a position of a SoilParticle, this function returns the amount each vertex should raise when merging it to the Terrain.
      \param pos      - position of a SoilParticle
      \param volume   - volume of the SoilParticle
      \param dh0      - storage of the change in height of vertex at coordinate 0 in ivd
      \param dh1      - storage of the change in height of vertex at coordinate 1 in ivd
      \param dh2      - storage of the change in height of vertex at coordinate 2 in ivd
      \param ivd      - Terrain data for position of SoilParticle
      \return success
      */
      bool getDeltaHeights( agx::Vec3 pos, const agx::Real volume, agx::Real& dh0, agx::Real& dh1, agx::Real& dh2, ImpactVolumeData& ivd );

      /**
      Merge a SoilParticle with the Terrain.
      */
#if TERRAIN_USE_PARTICLE_SYSTEM
      bool mergeParticle( agx::Physics::ParticlePtr particle );
#else
      bool mergeParticle( SoilParticle* particle );
#endif

      void addGeometryContactsForAddedParticles( SizeTPairVector& bodyIndexContactPairs,
          agx::RigidBodyRefVector& bodies, RealSizeTPairVector& localMatrixHeights,
          size_t localX, size_t localY, const size_t sizeX, const size_t sizeY
                                               );

      /**
      Temporary convert surface of height field to particles to deform terrain.
      Return the number of particles in the bodies vector that are empty.
      */
      size_t generateVolumeOfParticlesWithContacts( agx::HashSet<size_t>& insertedFccIndices, agx::Vector< size_t >& changedVertices, agxSDK::Simulation* simulation, agx::RigidBodyRefVector& bodies, SizeTPairVector& bodyIndexContactPairs, const agx::RigidBody* deformBody );

      /**
      Tries to merge all particles in m_particlesConsideredToBeRemoved with the Terrain.
      */
      void mergeParticles();

      /**
      Finds all geometry contacts containing the Terrain geometry. Stores them in m_geometryContactsWithImpactVolumeData in GeometryContactInfo class.
      */
      void collectContacts();

      /**
      Clears m_geometryContactsWithImpactVolumeData
      */
      void clearContacts();

      /**
      Make sure no geometries have changed bodies
      */
      void updateContacts( const agxCollide::GeometryContactPtrVector& contacts, bool removeContacts = false);

      /**
      Go through the grown vertices, and scale their height change with the precentChange scaler
      */
      agx::Real scaleGrowth( agx::Vector<size_t>& changedIndices, const bool justFindScaler, agx::Real scaler = agx::Real(1));

      /**
      Given all deformations during the last time step, saved to m_tempCompressions, this function will deform the height field.
      It will only shrink the heights if shrink is true, and only grow if shrink is false.
      It returns (if volumeCalculation is true) the changed height field volume.
      The compressionPercent is a value of how large amount of the height shrink that is compressed (the rest is moved sideways)(only for shrinking).
      \returns a Vec3 with the x = overlapping volume, y = compressedVolume, z = pushed sidewayVolume , if volumeCalculation is set true
      */
      VolumeOverlap changeHeights( agx::Vector< size_t >& changedVertices, const bool shrink = true, const bool volumeCalculation = false, const agx::Real compressionPercent = agx::Real(0), const bool onlyCompression = true);

      /**
      Calculates the volume under a triangle of the height field, above the lowest zPosition.
      */
      agx::Real calculateVolumeForTriangle( agx::Vec3 xPositions, agx::Vec3 yPositions, agx::Vec3 zPositions );

      /**
      Finds the volume change after a compression at a grid point on the Terrain.
      Can be used with a negative sign of compression, to get volume increment (negative return value)
      */
      agx::Real calculateVolumeDiffAroundOneVertex( size_t vectorIndex, agx::Real compression = 0 );

      /**
      Handles the volume change for shovels. Will make a call to createParticles if necessary.
      */
      bool shrinkShovelVolume( agx::Vector< size_t >& localChangedVertices, const bool shovelIsDeformer, const agxCollide::Geometry* shovel, const agx::Real shovelThickness, const agx::Vec3 shovelForwardW );

      /**
      Create particles around one particle position in the particle coordinate system
      */
      agx::RigidBody* createVolumeReplacingParticle( agx::HashSet<size_t>& insertedFccIndices, agxSDK::Simulation* simulation, const agx::Real minHeightLevel, const agx::Real heightFieldHeight, const size_t particleX, const size_t particleY, const size_t particleZ, ParticleCoordinateSystem::DirectionFCC fccPos );

      bool accumulateVertexY( agx::Vec3& meanPos, agx::Vec3Vector& vertices, agx::Vector<agx::RangeReal>& localChangedHeights, const size_t minX, const size_t minY, const size_t sizeX, const size_t sizeY, const size_t localX, const size_t localY);
      size_t accumulateVerticesX( agx::Vec3& meanPos, agx::Vec3Vector& vertices, agx::Vector<agx::RangeReal>& localChangedHeights, const size_t minX, const size_t minY, const size_t sizeX, const size_t sizeY, const size_t localX, const size_t localY, const size_t maxNum);
      void createConvexBody(agx::Vector< size_t >& localChangedVertices, const agx::Real shovelThickness);

      /**
      Creates particles of the correct volume where the height field has been removed
      \return number of particles created.
      */
      size_t createParticles( agx::Vector< size_t >& localChangedVertices, const agx::Real volume, const agxCollide::Geometry* shovel, const agx::Real shovelThickness, const ShovelPlane& plane  );

      class GeometryContactInfo;
      /**
      Find which vertices that should shrink due to one geometry contact.
      */
      void collectDeformationFromGeometryContact( agxModel::GeometryContactInfo* geometryContactInfo, agx::Vector< size_t >& allDeformedVertices);

      /**
      Finds Terrain penetrations from a geometry contact.
      */
      agx::Real handleContact(agxModel::GeometryContactInfo* geometryContactInfo, agx::Vector< size_t >& localChangedVertices, const bool isShovel, const bool deformsDynamicHeightField );

      /**
      We have to keep track of which vertices that have been visited (and raycasted) for a specific geometryContact,
      so that we don't raycast any vertices twice. This could maybe be done using a hash table (at least if the terrain grid is large.)
      */
      bool isVisited( const size_t vertexIndex ) const;

      /**
      Notify that a specific vertex has been visited
      */
      void setVisited( const size_t vertexIndex );

      /**
      Add vertex neighbors to queue, for raycasting.
      */
      void enqueueNeighbors( std::queue<size_t>& queue, int x, int y );

      void updateTempCompression(const size_t vertexIndex, agx::Real deltaCompression);
      void setTempCompression(const size_t vertexIndex ,agx::Real compression);
      agx::Real getTempCompression(const size_t vertexIndex) const;

      void notifyModifiedIndex(const size_t x, const size_t y, const size_t currentIndex, agx::Real& depth);

      /**
      Finds displacement for vertices due to contact.
      return max depth for vertices found.
      */
      agx::Real findVertexDisplacements( agxCollide::Geometry* collidingGeometry, ImpactVolumeData& ivd, agx::Vector< size_t >& localChangedVertices, const bool isShovel, const bool deformsDynamicHeightField );

      /**
      Find penetration depth for geom in Terrain.
      */
      bool raycastVertex( agxCollide::Geometry* geom , agx::Real& depth, agx::Vec3 hfLocalPosition, const bool isShovelWithoutDeformer, agx::Real& maxThickness ) const;

      /**
      Get the non-Terrain(this) geometry from a geometry contact containing Terrain.
      */
      const agxCollide::Geometry* getOtherGeometry( const agxCollide::LocalGeometryContact* geometryContact ) const;
      agxCollide::Geometry* getOtherGeometry( agxCollide::LocalGeometryContact* geometryContact );

      const agxCollide::Geometry* getOtherGeometry( const agxModel::GeometryContactInfo* geometryContactInfo ) const;
      agxCollide::Geometry* getOtherGeometry( agxModel::GeometryContactInfo* geometryContactInfo );

      /**
      Notify that a specific height has been changed for the height field.
      */
      void setModifiedIndex( size_t x, size_t y );

      agxCollide::GeometryRef m_geometry;

      // Vectors with temporary values (could maybe be moved to TerrainDataInterface, and maybe replaced with hash tables for large height fields)
      agx::RealVector m_tempCompression;
      agx::RealVector m_noGrowIndices;
      agx::BoolVector m_visitedIndices;

      GeometryContactInfoRefVector m_geometryContactsWithImpactVolumeData;

      /**
      Dynamic Height Field variables
      */
      bool m_useDynamicHeightField;
      agx::Real m_cutOffVelocity;
      agx::Real m_cutOffAngularVelocity;
      agx::Real m_verticalYoungsModulusAreaScaler;
      agx::Real m_horizontalYoungsModulusAreaScaler;

      /**
      Avalanche variables
      */
      agx::Real m_wetnessScale;
      agx::Real m_avalancheDecayPercent;
      size_t    m_avalancheStepLifeTime;
      size_t    m_avalanceVertexDistanceExtent;

      /**
      Shovel digging parameters
      */
      agx::Real m_shovelMinAngleToHorizontalPlane;

      ParticleAttributes m_particleAttribute;
      ParticleCoordinateSystem m_particleCoordinateSystem;
      HeightControllerRef m_heightController;
#if defined(USE_SOIL_PARTICLE_SIMULATION) && USE_SOIL_PARTICLE_SIMULATION == 1
      agxSDK::SimulationRef m_particleSimulation;
#endif

      DynamicHeightFieldSimulationRef m_dynamicHeightFieldSimulation;
      ContactGeneratorRef m_shovelContactGenerator;

      agx::Vector<agx::ContactMaterialRef> m_temporaryContactMaterials;
      agx::Vector< agx::ContactMaterialRef > m_originalContactMaterialStorage;
      agx::Vector< agx::FrictionModelRef > m_shovelFrictionMaterialPool;

      //reset each timestep
      agx::Vec2iVector m_modifiedIndices;
#if TERRAIN_USE_PARTICLE_SYSTEM
      agx::ParticleSystemRef m_particleSystem;
      agx::List<agx::Physics::ParticlePtr> m_particlesConsideredToBeRemoved;
      SoilParticleObsVector m_createdParticles;
      agx::VectorPOD<agxData::EntityPtr> m_removedParticles; //This vector contain the last ref ptr to these bodies
#else
      agx::List<SoilParticle*> m_particlesConsideredToBeRemoved;
      SoilParticleObsVector m_createdParticles;
      SoilParticleRefVector m_removedParticles; //This vector contain the last ref ptr to these bodies
#endif
      // Access to data for the terrain
      agxModel::TerrainDataInterfaceRef m_data;

      agx::ref_ptr<TerrainDebugRenderer> m_terrainDebugRenderer;

      agx::Vector<int> m_xDiff;
      agx::Vector<int> m_yDiff;

      agx::Vector< agx::Vec2i > m_minNonMergeCoordinates;
      agx::Vector< agx::Vec2i > m_maxNonMergeCoordinates;

      agx::Real m_cachedDigVolume;
      size_t m_particlesLost;
  };

  typedef agx::ref_ptr<Terrain> TerrainRef;

}

#endif
