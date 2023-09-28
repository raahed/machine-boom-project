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
#include <agxTerrain/AvalancheController.h>
#include <agxTerrain/PrimaryActiveZone.h>
#include <agxTerrain/TerrainMaterial.h>
#include <agxTerrain/Grid.h>
#include <agxTerrain/ColumnHeightGrid.h>
#include <agxTerrain/SoilPenetrationResistance.h>
#include <agxTerrain/TerrainGridControl.h>
#include <agxTerrain/TerrainToolCollection.h>
#include <agxTerrain/TerrainPhysics.h>
#include <agxTerrain/TerrainMassController.h>
#include <agxTerrain/CompactionController.h>
#include <agxTerrain/TerrainContactGenerator.h>
#include <agxTerrain/SoilSimulationInterface.h>
#include <agxTerrain/SoilParticleStatistics.h>
#include <agxTerrain/TerrainMaterialLibrary.h>
#include <agxTerrain/TerrainDataAtlas.h>
#include <agxTerrain/TerrainProperties.h>

#include <agxSDK/TerrainInstance.h>

#include <agx/HashSet.h>
#include <agx/HighAccuracyTimer.h>
#include <agx/BitState.h>
#include <agx/IndexLambdaKernel.h>

#include <agxCollide/Plane.h>
#include <agxCollide/Geometry.h>
#include <agxCollide/HeightField.h>

namespace agxTerrain
{
  typedef agx::Physics::GranularBodyPtr SoilParticlePtr;
  typedef agx::Physics::GranularBodyPtrVector SoilParticlePtrVector;
  typedef agx::HashSet<agx::Vec3i> VoxelIndexSet;
  typedef std::pair<SoilParticlePtr, agx::Real> ParticleWeightPair;
  typedef agx::Vector<ParticleWeightPair> ParticleWeightPairVector;
  typedef agx::HashTable<agx::Vec3i, ParticleWeightPairVector> IndexToParticleWeightPairVector;
  typedef agx::SetVector<agx::Vec2i> ModifiedVerticesSetVector;
  typedef agx::Vector<agx::Vec2i> ModifiedVerticesVector;
  typedef agx::Vector<std::pair<SoilParticlePtr, float>> ParticleDeltaVector;
  using TerrainIndexSet = agx::HashSet<agx::Vec2i>;

  AGX_DECLARE_POINTER_TYPES(Terrain);
  AGX_DECLARE_VECTOR_TYPES(Terrain);

  /**
  A terrain model based a 3D grid model with overlapping height field that can be deformed by interacting
  shovels objects performing digging motions, converting solid mass to dynamic mass which can be moved.

  The underlying terrain data model consists of a hierarchical sparse 3D grid of cells, containing terrain data such as mass.
  The terrain surface is represented as a 2D height field that is updated when the underlying grid structure is changed.
  Mass is moved in the terrain via interacting shovels that converts solid cells to dynamic mass. Each shovel has an "Active Zone"
  which is defined by specifying two edges on the shovel's constructor, a top edge and a cutting edge. These edges should be perpendicular
  to the shovel's digging direction and parallel to the shovel's cross section. When the active zone intersects the terrain,
  dynamics mass is created in the overlapping area.

  /// \todo mention something about material parametrization.
  The dynamic mass is expressed in the form of 6-degrees of freedom soil particles with hertzian contact dynamics, coulomb friction
  and rolling resistance. The dynamic soil is pushed via the shovel which in turn returns a resistance force against the shovel that
  is computed from the mass aggregated in the shovel's active zone. The dynamic soil is merged back into the terrain when approaching
  a steady state on the terrain surface.

  openvdb, which is a library for representing large sparse 3D hierarchal voxel structures, is used for cell storage and representation.
  */
  class AGXTERRAIN_EXPORT Terrain : public agxSDK::TerrainInstance
  {
  public:
    enum class MaterialType
    {
      TERRAIN,
      PARTICLE,
      AGGREGATE
    };

    enum class MassType
    {
      SOLID,
      PARTICLE,
      FLUID
    };

  public:
    /**
    \return true if particles sees given rigid body as kinematic (regardless of motion control)
    */
    static agx::Bool getEnableKinematicForParticles( const agx::RigidBody* rb );

    /**
    Enable/disable particles to see this rigid body as kinematic regardless of current motion control.
    \param rb - rigid body
    \param enable - true to enable this feature, false to disable
    */
    static void setEnableKinematicForParticles( agx::RigidBody* rb, agx::Bool enable );

    /**
    Find first terrain with given name.
    \param simulation - simulation the terrain is part of
    \param name - name of the terrain
    \return terrain instance if found - otherwise nullptr
    */
    static Terrain* find( const agxSDK::Simulation* simulation, const agx::Name& name );

    /**
    Find terrain with given UUID.
    \param simulation - simulation the terrain is part of
    \param uuid - UUID of the terrain
    \return terrain instance if found - otherwise nullptr
    */
    static Terrain* find( const agxSDK::Simulation* simulation, const agx::Uuid& uuid );

    /**
    Find all terrains with given name.
    \param simulation - simulation the terrain is part of
    \param name - name of the terrain
    \return vector of terrains
    */
    static TerrainPtrVector findAll( const agxSDK::Simulation* simulation, const agx::Name& name );

    /**
    Finds all terrains in the given simulation.
    \param simulation - simulation with terrains.
    \return vector of terrains
    */
    static TerrainPtrVector findAll( const agxSDK::Simulation* simulation );

  public:
    /**
    Creates a terrain object by using an existing height field as a basis. The height field data is copied into
    the internal terrain structure. The existing height field MUST have uniform scale in X and Y axis, otherwise
    the function returns nullptr.
    \note The specified height field is copied and never used inside Terrain object.
    \param heightField - The height field object that will be used as a basis for creating
    \param maximumDepth - The maximum depth of the terrain.
    \note - If the heightfield contains heights which are lower than the specified maximumDepth
            parameter, then the heightfield heights will be clamped upwards to the maximumDepth.
    \return A terrain object create from the specified height field or nullptr if an invalid height field is given.
    */
    static Terrain* createFromHeightField(const agxCollide::HeightField* heightField, agx::Real maximumDepth);

    /**
    Constructor. Creates a terrain object with a specific resolution of 2D grid points with elementSize as the length between them.
    The terrain length in each direction thus becomes (resolution - 1) * elementSize.
    \param resolutionX - The number of height points to use in the Terrain in the X-direction.
    \param resolutionY - The number of height points to use in the Terrain in the Y-direction.
    \param elementSize - The distance between the height points in the terrain.
    \param maximumDepth - The maximum depth that the terrain can have.
    */
    Terrain(size_t resolutionX, size_t resolutionY, agx::Real elementSize, agx::Real maximumDepth);

    /**
    Constructor. Creates a terrain object with a specific resolution of 2D grid points with elementSize as the length between them.
    The terrain length in each direction thus becomes (resolution - 1) * elementSize.
    \param resolutionX - The number of height points to use in the Terrain in the X-direction.
    \param resolutionY - The number of height points to use in the Terrain in the Y-direction.
    \param elementSize - The distance between the height points in the terrain.
    \param heights - A row major matrix, containing the specified heights, with dimensions (resolutionX, resolutionY).
    \param flipY - set to true if y decreases with increasing indices in heights.
    \param maximumDepth - The maximum depth that the terrain can have.
    */
    Terrain(
      size_t resolutionX, size_t resolutionY, agx::Real elementSize, const agx::RealVector& heights,
      bool flipY, agx::Real maximumDepth);

    /**
    Set a compaction for all current active solid grid elements in the Terrain object.
    \note - the terrain grid elements all start with a the nominal compaction value of 1.0, of which all
    user specified bulk parameters correspond to.
    \param compaction - The compaction value to set to every voxel in the terrain (Default: 1.0)
    \param replaceCompactedSoil - True if all voxels which already have a compaction value (compaction != 1.0)
    should get overwritten.
    */
    void setCompaction( agx::Real compaction, bool replaceCompactedSoil=true ) const;

    /**
    Set the transform of this terrain.
    \param transform - new transform of this terrain, given in world coordinate frame
    */
    void setTransform(const agx::AffineMatrix4x4& transform);

    /**
    \return transform of this terrain
    */
    agx::AffineMatrix4x4 getTransform() const;

    /**
    \return inverse transform of this terrain
    */
    agx::AffineMatrix4x4 getInverseTransform() const;

    /**
    Set if terrain computations should be enabled.
    \note - updates for ToolCollections will still be executed if false.
    \param enable - true if terrain computations should be enabled, false otherwise.
    */
    void setEnable( bool enable );

    /**
    \return true if terrain computations should be enabled, false otherwise.
    \note - updates for ToolCollections will still be executed if false.
    */
    bool getEnable() const;

    /**
    Set the position of this terrain.
    \param position - new position of this terrain, given in world coordinate frame
    */
    void setPosition(const agx::Vec3& position);

    /**
    \return rotation of this terrain, in world coordinate frame
    */
    agx::Vec3 getPosition() const;

    /**
    Set the rotation of this terrain.
    \param rotation - new rotation of this terrain, given in world coordinate frame
    */
    void setRotation(const agx::Quat& rotation);

    /**
    \return rotation of this terrain, in world coordinate frame
    */
    agx::Quat getRotation() const;

    /**
    Transform point given in world coordinate frame to local coordinate
    frame of this terrain.
    \param point - point in world coordinate frame
    \return point in local coordinate frame of this terrain
    */
    agx::Vec3 transformPointToTerrain(const agx::Vec3 point) const;

    /**
    Transform point given in local coordinate frame of this terrain,
    to world coordinate frame.
    \param point - point in local coordinate frame of this terrain
    \return point in world coordinate frame
    */
    agx::Vec3 transformPointToWorld(const agx::Vec3 point) const;

    /**
    Transform vector given in world coordinate frame to local coordinate
    frame of this terrain.
    \param vector - vector in world coordinate frame
    \return vector in local coordinate frame of this terrain
    */
    agx::Vec3 transformVectorToTerrain(const agx::Vec3 vector) const;

    /**
    Transform vector given in local coordinate frame of this terrain,
    to world coordinate frame.
    \param vector - vector in local coordinate frame of this terrain
    \return vector in world coordinate frame
    */
    agx::Vec3 transformVectorToWorld(const agx::Vec3 vector) const;

    /**
    \note It's undefined to modify the terrain height field instance.
    \return the height field shape of this terrain
    */
    const agxCollide::HeightField* getHeightField() const;

    /**
    \note It's undefined to modify this geometry instance.
    \return the geometry of this terrain
    */
    virtual agxCollide::Geometry* getGeometry() const override;

    /**
    Get the soil particle height field consisting of the highest point of either
    the terrain surface or the highest particle position in each terrain index.
    \note It's undefined to modify the terrain height field instance.
    \return the soil particle height field shape of this terrain
    */
    const agxCollide::HeightField* getSoilParticleBoundedHeightField() const;

    /**
    Sets if deformation should be enabled on this terrain. If this is set to false no dynamic mass particles or fluid mass is created.
    Additionally, no avalanching or compaction occurs on the terrain.
    \param deformable - Whether or not to deformations should occur on this terrain
    */
    void setDeformable( bool deformable );

    /**
    Gets whether or not deformations are enabled for this terrain. If false, then no dynamic mass particles or fluid mass is created.
    Additionally, no avalanching or compaction occurs on the terrain.
    \return true if deformations are enabled for the terrain, false otherwise
    */
    inline bool getDeformable() const;

    /**
    Add disable merge and avalanche a number of indices from the terrain edge
    */
    void setNoMergeEdgeMargin(size_t noMergeEdgeMargin);

    /**
    Prevent merge of dynamic mass to this terrain
    */
    void setNoMerge(bool noMerge);

    /**
    Return whether this terrain is prevented from merging dynamic mass to it
     */
    bool getNoMerge();

    /**
    Add disable merge and avalanche inside geometry bound with added extension distance
    */
    bool addNoMergeZoneToGeometry( agxCollide::Geometry* geometry, agx::Real extensionDistance);

    /**
    remove disable merge and avalanche inside geometry bound with added extension distance
    */
    bool removeNoMergeZoneToGeometry( agxCollide::Geometry* geometry);

    /**
    Returns the terrain vertices which voxel columns has modified compaction during the last time step.
    The method getCompactionController()->getCompressedTerrainIndices only returns terrain vertices that
    has been compressed by terrain contacts. This method takes vertices that has modified compaction by
    avalanching and dynamic mass mergeing back into the terrain into account aswell.
    \return a HashVector containing the modified terrain indices which voxel columns has modified compaction
    during the last time step.
    */
    const ModifiedVerticesVector& getModifiedVoxelColumns() const;

    /**
    Returns the modified terrain height field vertices during the last time step.
    \return a HashVector containing the modified terrain indices for the height field during the last time step.
    */
    const ModifiedVerticesVector& getModifiedVertices() const;

    /**
    Returns the modified terrain height field (including particles) vertices during the last time step.
    \return a HashVector containing the modified particle height field indices for the last time step.
    */
    const ModifiedVerticesVector& getModifiedParticleVertices() const;

    /**
    Removes a vertex from the vector with vertices that were modified during the last time step.
    */
    void removeModifiedParticleVertice(agx::Vec2i terrainIndex);

    /**
    Returns the terrain vertices that contains soil particles during the last time step.
    \return a HashVector containing the terrain indices containing particles during the last time step.
    */
    const ModifiedVerticesVector& getParticleVertices() const;

    /**
    Returns the total surface size of he terrain in X Y coordinate system.
    */
    agx::Vec2 getSize() const;

    /**
    Returns the element resolution of the terrain in the local X-axis.
    */
    size_t getResolutionX() const;

    /**
    Returns the element resolution of the terrain in the local Y-axis.
    */
    size_t getResolutionY() const;

    /**
    Returns the element size of a cell in agxTerrain, which is incidentally the distance between the center of each grid point
    in the surface height field.
    \return the element size of a single cell in Terrain.
    */
    agx::Real getElementSize() const;

    /**
    Returns the memory usage in bytes in agxTerrain.
    */
    size_t getMemoryUsage();

    /**
    \return the minimum allowed height of this terrain
    */
    agx::Real getMinimumAllowedHeight() const;

    /**
    Finds the terrain surface position, given in local coordinate frame, for a specified terrain index.
    \param - terrainIndex - the terrain index, going from (0,0) -> (resolutionX-1, resolutionY-1)
    \param includeSurfaceParticles - Also consider surface particles when calculating the height, default false.
    \returns the position of the terrain surface in the grid point, given in local coordinate frame
    */
    agx::Vec3 getSurfacePosition(const agx::Vec2i& terrainIndex, bool includeSurfaceParticles = false) const;

    /**
    Finds the terrain surface world position for a specified terrain index.
    \param - terrainIndex - the terrain index, going from (0,0) -> (resolutionX-1, resolutionY-1)
    \param includeSurfaceParticles - Also consider surface particles when calculating the height, default false.
    \returns the world position of the terrain surface in the grid point
    */
    agx::Vec3 getSurfacePositionWorld(const agx::Vec2i& terrainIndex, bool includeSurfaceParticles = false) const;

    /**
    Finds the closes terrain grid point given any point in world.
    \param worldPosition - position in world
    \param clampToBound - true if the grid point should be clamped to border if position is outside bounds
    \returns the closest terrain grid index, going from (0,0) -> (resolutionX-1, resolutionY-1)
    */
    agx::Vec2i getClosestGridPoint(const agx::Vec3& worldPosition, bool clampToBound=false) const;

    /**
    Project a world point to the grid surface.
    \param worldPosition - position in world
    \param result - position on this terrain, given in world coordinates
    \return true if \p worldPosition successfully projected onto this terrain, false
            if \p worldPosition is outside
    */
    agx::Bool projectPointToSurface(const agx::Vec3& worldPosition, agx::Vec3& result) const;

    /**
    Checks if the specified index is at the border of the terrain
    \param terrainIndex - the terrain index to check
    \returns true if the given index is at the border, else false
    */
    bool isBorderIndex(const agx::Vec2i& terrainIndex) const;

    /**
    Checks if the specified index is within bound of the terrain [(0,0) -> (resolutionX, resolutionY)]
    \param terrainIndex - the terrain index to check
    \returns true if the given index is within bounds, else false
    */
    bool isIndexWithinBounds(const agx::Vec2i& terrainIndex) const;

    /**
    Checks whether the position is within terrain (x,y) position bounds in world space.
    \param position - the position to be checked against the terrain bound.
    \returns true if the position is within bounds of terrain, false otherwise.
    */
    bool isWorldPositionWithinBounds(const agx::Vec3& position) const;

    /**
    Checks whether the position is within terrain (x,y) position bounds in local space.
    \param position - the position to be checked against the terrain bound.
    \returns true if the position is within bounds of terrain, false otherwise.
    */
    bool isLocalPositionWithinBounds(const agx::Vec3& position) const;

    /**
    Sets Terrain height at a specific terrain grid node.
    \param terrainIndex An (x,y) index specifying the grid point to set the height to
    \param height the new height in the height field.
    */
    void setHeight(const agx::Vec2i& terrainIndex, agx::Real height);

    /**
    Gets the terrain height at a specific terrain grid node.
    \param terrainIndex An (x,y) index specifying the grid point.
    \param includeSurfaceParticles - Also consider particles when returning height. Default false.
    \returns the height at the specific grid node.
    */
    agx::Real getHeight(const agx::Vec2i& terrainIndex, bool includeSurfaceParticles = false) const;

    /**
    \return the calculated total terrain volume.
    */
    agx::Real calculateVolume();

    /**
    \return the calculated total solid mass in the terrain in kg.
    */
    agx::Real calculateTotalSolidMass() const;

    /**
    \return the calculated current total fluid mass in the terrain in kg.
    */
    agx::Real calculateTotalFluidMass() const;

    /**
    \return the calculated current total dynamic mass in the terrain in kg. This includes created soil particles and fluid mass.
    */
    agx::Real calculateTotalDynamicMass() const;

    /**
    \return the calculated the total mass in the terrain, consisting of both dynamic and solid mass.
    */
    agx::Real calculateTotalMass() const;

    /**
    Adds a shovel object in the terrain. The shovel objects interacts with the terrain by
    converting solid soil to dynamic soil in it's active zone.
    \param shovel - The specified shovel object.
    \return true if the shovel object was successfully registered in the terrain object, false otherwise.
    */
    bool add(Shovel* shovel);

    /**
    Removes a shovel object in the terrain, if it exists.
    \param shovel - The specified shovel object.
    \return true if the shovel object was successfully removed in the terrain object, false otherwise.
    */
    bool remove(Shovel* shovel);

    /**
    \return registered tool collections for this terrain instance
    */
    const TerrainToolCollectionRefVector& getToolCollections() const;

    /**
    \param shovel - shovel instance
    \return tool collection given shovel - nullptr when shovel hasn't been added to this terrain
    */
    TerrainToolCollection* getToolCollection( const Shovel* shovel ) const;

    /**
    Return the shovels currently added in the terrain.
    */
    ShovelRefVector getShovels() const;

    /**
    The result includes the active force and torque from the penetration resistance from the terrain on the shovel if the shovel
    is digging in the terrain
    \param shovel - interacting shovel
    \param force - the penetration force
    \param torque - the penetration torque
    \return true if resulting force and torque was written to \p force and \p torque - otherwise false
    */
    bool getPenetrationForce(const Shovel* shovel, agx::Vec3& force, agx::Vec3& torque) const;

    /**
    Given geometry contacts exists and the solver has solved them - calculates
    total contact force between the terrain soil particle aggregate and the given shovel. This represents the
    separation force that is required to move the excavated soil in the shovel active zone.
    \param shovel - interacting shovel
    \return the total contact force between given shovel and the soil particle aggregate in the terrain.
    */
    agx::Vec3 getSeparationContactForce( const Shovel* shovel ) const;

    /**
    Given geometry contacts exists and the solver has solved them - calculates
    total contact force between the terrain deformation soil aggregate and the given shovel. This represents the
    deformation force that is required to move soil via shovel deformation instead of excavation, i.e not excavation or digging.
    Examples of this would be side movement and backwards grading of the soil.
    \param shovel - interacting shovel
    \return the total contact force between given shovel and the deformer soil aggregates in the terrain.
    */
    agx::Vec3 getDeformationContactForce(const Shovel* shovel) const;

    /**
    calculates total contact force between the soil aggregate associated with the specified excavation mode and the given shovel.
    This represents the deformation force that is required to move soil via shovel deformation instead of excavation, i.e not
    excavation or digging. Examples of this would be side movement and backwards grading of the soil.
    \param shovel - interacting shovel
    \param excavationMode - the excavation mode that the specified soil aggregate belongs to
    \return the total contact force between given shovel and the soil aggregate specified by the excavation mode
    */
    agx::Vec3 getExcavationModeContactForce(const Shovel* shovel, Shovel::ExcavationMode excavationMode) const;

    /**
    Given geometry contacts exists and the solver has solved them - calculates
    total shovel contact force between this terrain and the given shovel. This is the contact force
    that prevents the shovel from falling through the terrain when not in excavation mode, where contact
    feedback is generated from the soil aggregates.
    \note - This method returns regular contact forces ONLY when no soil aggregates are present to generate excavation feedback!
    \param shovel - interacting shovel
    \return the total non-excavation contact force between this terrain and the given shovel.
    */
    agx::Vec3 getContactForce( const Shovel* shovel ) const;

    /**
    Get the shovel <-> aggregate contacts with the terrain given an excavation mode and a shovel.
    \param shovel - the specified shovel
    \param excavationMode - the excavation mode that the soil aggregate belongs too ( PRIMARY, DEFORM_BACK, DEFORM_RIGHT, DEFORM_LEFT )
    \return vector containing the geometry contacts between the specified shovel and the aggregate.
    */
    agxCollide::GeometryContactPtrVector getShovelAggregateContacts( const Shovel* shovel, Shovel::ExcavationMode excavationMode ) const;

    /**
    Get the last computed dead load fraction of the shovel, i.e how much of it's inner volume that is filled with dynamic soil.
    The dead load fraction ranges from 0.0 (empty), to 1.0 (full).
    \param shovel - interacting shovel
    \return the last computed dead load fraction
    */
    agx::Real getLastDeadLoadFraction(const Shovel* shovel) const;

    /**
    \return the shovel inner volume that is used in the dead load calculations
    */
    agx::Real getInnerVolume(const Shovel* shovel) const;

    /**
    \return the last excavated volume (m3) of the terrain.
    */
    agx::Real getLastExcavatedVolume() const;

    /**
    Check if the specified shovel is currently in digging mode with the terrain, i.e if the cutting edge is submerged.
    \param shovel - interacting shovel
    \return the current penetration force between given shovel and this terrain
    */
    bool isDigging(const Shovel* shovel) const;

    /**
    \param shovel - interacting shovel
    \return dynamic mass in given shovel, including both particles and fluid mass
    */
    agx::Real getDynamicMass( const Shovel* shovel ) const;

    /**
    Returns the total soil aggregate mass in a terrain given a shovel and a specific excavation mode. This
    function can be used to extract the active mass that the shovel is trying to displace in the failure
    zones during digging and deformation.
    \param shovel - interacting shovel
    \param excavationMode - The excavation mode of the aggregate that will be used to extract the mass.
    \return the total aggregate mass, including inner shape if excavation mode is PRIMARY_EXCAVATION.
    */
    agx::Real getSoilAggregateMass(const Shovel* shovel, Shovel::ExcavationMode excavationMode) const;

    /**
    Check is a shovel object is registered in the terrain.
    \param shovel - the shovel object that is checked against the terrain.
    \return true if the shovel object is registered in the terrain, false otherwise.
    */
    bool contains(const Shovel* shovel) const;

    /**
    Checks if a specified rigid body is contained in any of the terrain object's registered shovels.
    \param body - the specified body to check against the registered shovels.
    \return true if the specified body is contained in any of the registered shovels.
    */
    bool isRigidBodyShovel(const agx::RigidBody* body) const;

    /**
    Checks if the specified geometry's rigid body is contained in any of the terrain object's registered shovels.
    \param geometry - the specified geometry for which its body should be checked against the registered shovels.
    \return true if the specified geometry's body is contained in any of the registered shovels.
    */
    bool isGeometryShovel(const agxCollide::Geometry* geometry) const;

    /**
    Return true if the specified geometry bounding volume overlaps the terrain geometry bounding volume
    */
    bool hasOverlap(const agxCollide::Geometry* geometry) const;

    /**
    Returns one of the internal materials in agxTerrain. TODO fix this.
    MaterialType::TERRAIN   - The internal material used on the terrain surface height field structure.
    MaterialType::PARTICLE  - The internal material used on the terrain particles.
    MaterialType::AGGREGATE - The internal material used on aggregate used to represent the particle mass.

    These materials can be used to configure contact materials with objects interacting with the terrain.

    \returns The specified internal material in agxTerrain.
    */
    agx::Material* getMaterial( Terrain::MaterialType type = Terrain::MaterialType::TERRAIN ) const;

    /**
    Assign new material to this terrain. The material instance will be assigned
    to the specified part of the terrain:
    MaterialType::Terrain   - The geometry holding the height field (getHeightFieldGeometry()) will be assigned the material.
    MaterialType::Particle  - The granular body system simulating dynamic soil particles will be assigned the material.
    MaterialType::Aggregate - NOT SUPPORTED
    Internal contact materials terrain <-> particles, terrain <-> soil particle aggregate and particles <-> particles
    will be updated.

    Note that the properties in the material will be overwritten by the terrain based on the assigned TerrainMaterial.
    \param material - new material for this terrain
    \param type - specifies which material to set.
    */
    void setMaterial( agx::Material* material, Terrain::MaterialType type = Terrain::MaterialType::TERRAIN);

    /**
    Returns one of the internal contact materials in agxTerrain. TODO fix this.

    Contact materials exist between
    MaterialType::TERRAIN  - MaterialType::PARTICLE
    MaterialType::PARTICLE - MaterialType::PARTICLE
    MaterialType::TERRAIN  - MaterialType::AGGREGATE

    \returns The specified internal contact material in agxTerrain.
    */
    agx::ContactMaterial* getContactMaterial( Terrain::MaterialType type1, Terrain::MaterialType type2 );

    /**
    Returns the terrain material which is used to derive material and contact material parameters.

    \returns The set terrain material.
    */
    TerrainMaterial* getTerrainMaterial() const;

    /**
    Set the material for the bulk of the terrain which is used to derive
    material and contact material parameters.
    */
    void setTerrainMaterial( TerrainMaterial* terrainMaterial );

    /**
    Loads a TerrainMaterial preset in the TerrainMaterialLibrary that contains calibrated bulk and contact
    properties for a predefined material archetype.
    Note - This method uses the convenience enum MaterialPreset that points to existing preset files.
    \param materialName - The name of the material preset that should be loaded.
    \return true if the material was successfully loaded, false otherwise.
    */
    bool loadLibraryMaterial( const agx::String materialName );

    /**
    Get the available TerrainMaterial presets from the TerrainMaterialLibrary that contains calibrated
    bulk and contact properties for a predefined material archetype.
    \return a vector containing the available TerrainMaterial presets in the TerrainMaterialLibrary.
    */
    agx::StringVector getAvailableLibraryMaterials() const;

    /**
    Load a TerrainMaterial from a specification file in JSON data format.
    \param filename - the name of the JSON file containing the specified material data.
    \return true if the specification was successfully loaded, false otherwise.
    */
    bool loadMaterialFile( const agx::String& filename );

    /**
    Explicitly set contact material properties in a shovel-aggregate contact for a specific
    excavation mode for a specific shovel-terrain pair. This overrides the shovel-terrain contact
    material properties that are used in the default case.
    \param shovel - The specified shovel.
    \param contactMaterial - The contact material to be set in the aggregate contact.
    \param mode - The specified excavation mode that corresponds to the aggregate.
    \note - this returns false if the shovel is not set to the terrain.
    \return true if the contact material was successfully set, false otherwise.
    */
    bool setShovelAggregateContactMaterial( const Shovel* shovel,
                                            agx::ContactMaterial* contactMaterial,
                                            Shovel::ExcavationMode mode = Shovel::ExcavationMode::PRIMARY );

    /**
    Get the explicitly set contact material in a shovel-aggregate contact corresponding to a specified
    excavation mode for a specific shovel-terrain pair. This overrides the shovel-terrain contact
    material properties that are used in the default case.
    \param shovel - The specified shovel.
    \param mode - The specified excavation mode that corresponds to the aggregate.
    \note - this returns nullptr if the shovel is not set to the terrain.
    \return a contact material if one has been explicitly set, nullptr otherwise.
    */
    agx::ContactMaterial* getShovelAggregateContactMaterial( const Shovel* shovel,
                                                             Shovel::ExcavationMode mode = Shovel::ExcavationMode::PRIMARY ) const;

    /**
    Set height values of all sampled points.
    \param heights - must be a row major matrix with dimensions (resolutionX, resolutionY).
    \param flipY - set to true if y decreases with increasing indices in heights. ( Default: false )
    \param resetCompaction - Set to true if the compaction data should be erased to default compaction,
                             false if compaction data should be preserved. ( Default: true )
    \return true if heights were set successfully
    */
    bool setHeights( const agx::RealVector& heights, bool flipY = false, bool resetCompaction = true );

    /**
    * Converts the terrain occupancy that overlaps the specified shape into dynamic mass
    * \param shape - The shape within which to convert occupancy into dynamic mass
    */
    void convertToDynamicMassInShape(agxCollide::Shape* shape);

    /**
    Triggers initial avalanching on all indices in the terrain. If any height differences in the terrain violates the angle of
    repose requirement, avalanching will begin on those indices.
    */
    void triggerForceAvalancheAll();

    /**
    Calculate a reduced version of the height field structure. Heights in terrain index segments with dimensions [factorX, factorY]
    are averaged and put into a reduced structure with dimensions [resolutionX/factorX, resolutionY/factorY].
    \param factorX - The factor to resize the height field in the X dimension.
    \param factorY - The factor to resize the height field in the Y dimension.
    \return a row major matrix of heights which corresponds to the reduced height field structure.
    */
    agx::RealVector getResizedHeightField(size_t factorX, size_t factorY) const;

    /**
    Get the soil particle system interface used to represent the dynamic mass in the terrain. This can be
    used to access and modify soil particle data.

    \return the interface to the soil particle system used to represent the dynamic mass in the terrain.
    */
    SoilSimulationInterface* getSoilSimulationInterface() const;

    /**
    Get control interface of the terrain grid data. This interface can modify and extract data from the
    terrain in grid cell level, properties such as mass and compaction.

    \return the control interface of the terrain grid data.
    */
    TerrainGridControl* getTerrainGridControl() const;

    /**
    Get the total mass (kg) of the active soil particles in the simulation.
    */
    agx::Real getTotalSoilParticleMass() const;

    /**
    Get basic statistics of the soil particles active in the simulation.
    */
    SoilParticleStatistics getSoilParticleStatistics() const;

    /**
    \return the current active terrain contacts with external geometries on this terrain instance.
    */
    const TerrainContactRefVector& getTerrainContacts() const;

    /**
    \return the properties object for the terrain.
    */
    TerrainProperties* getProperties() const;

    /**
    Get intersecting 3D grid element with specified geometry. The x,y coordinate of the returned 3D points
    is the terrain index while the z coordinate is a depth index. See the 'TerrainDataGridInterface' class
    for further explanation.
    \param geometry - The geometry to test intersection with the terrain grid.
    \param gridPoints - Reference to the container where the intersecting terrain grid elements will be put.
    \note - This function updates the bound of the specified geometry in order to properly do intersection tests.
    \return true if there are intersecting voxels within terrain bound, false if not.
    */
    bool getIntersectingGridElements( agxCollide::Geometry* geometry,
                                      agx::Vec3iVector& gridPoints ) const;

    /**
    Get intersecting 3D grid element with specified geometry that is active/occupied with a specified mass type.
    The x,y coordinate of the returned 3D points is the terrain index while the z coordinate is a depth
    index. See the 'TerrainDataGridInterface' class for further explanation.
    \param geometry - The geometry to test intersection with the terrain grid.
    \param gridPoints - Reference to the container where the intersecting terrain grid elements will be put.
    \param massType - The specific mass type (SOLID, FLUID or PARTICLE) that is active/occupied in grid
                      points that will be extracted.
    \note - This function updates the bound of the specified geometry in order to properly do intersection tests.
    \return true if there are intersecting voxels with specified mass type within terrain bound, false if not.
    */
    bool getIntersectingActiveGridElements( agxCollide::Geometry* geometry,
                                            agx::Vec3iVector& gridPoints,
                                            MassType massType ) const;

    /**
    Remove fluid mass (in kg, NOT occupancy) in the intersecting 3D grid element with specified geometry.
    The x,y coordinate of the returned 3D points is the terrain index while the z coordinate is a depth
    index. See the 'TerrainDataGridInterface' class for further explanation.
    \param geometry - The geometry to test intersection with the terrain grid.
    \param fluidMass - Amount of mass to be removed
    \note - This function updates the bound of the specified geometry in order to properly do intersection tests.
    \return The amount of fluid mass (in kg) that was removed in the grid points of the intersection.
    */
    float removeFluidMassInGeometry(agxCollide::Geometry* geometry, float fluidMass);

    /**
    Get the soil particle nominal radius from the simulation. The nominal radius
    is the particle radius that the terrain algorithm will aim for during the dynamic
    resizing of particles that occur during terrain interaction. This is calculated
    from the resulting element size upon construction of the terrain object.
    */
    float getParticleNominalRadius();

    /**
    Return the number of soil particles active in the simulation.
    */
    size_t getNumSoilParticles() const;

    /**
    Return the number of soil particles created during excavation this time step.
    \note - particles may disappear during dynamic mass distribution stage
    */
    size_t getNumCreatedParticles() const;

    /**
    Returns the world up direction of the terrain.
    */
    agx::Vec3 getUpDirection() const;

    /**
    \return the area of a voxel
    */
    agx::Real getVoxelArea() const;

    /**
    \return the volume of a voxel
    */
    agx::Real getVoxelVolume() const;

    /**
    \return the packing ratio of soil particles generated from the terrain.
    */
    static agx::Real getTerrainParticlePackingRatio();

    DOXYGEN_START_INTERNAL_BLOCK()

    AGXSTREAM_DECLARE_SERIALIZABLE( agxTerrain::Terrain );

    /// Returns the number of active voxels in the terrain.
    size_t getNumActiveVoxels();

    /// Returns the number of modified voxels during the last update.
    size_t getNumModifiedVoxels();

    /**
    Verify that a mass column in the terrain grid is properly filled, i.e that all submerged voxels under the surface holds the maximum
    allowed mass given by the voxel compaction. There should not be any "holes" in the terrain.
    \note - If the check fails, it will trigger an agxVerify and crash.
    \param terrainIndex - The specified terrain surface index to check.
    \return true if the terrain solid column satisfies mass integrity conditions, false otherwise.
    */
    bool verifyMassColumnIntegrity(const agx::Vec2i& terrainIndex);

    /**
    Returns true if the specified voxel index is below the maximum depth of the terrain.
    */
    bool isVoxelIndexBelowMaximumDepth(const agx::Vec3i& index) const;

    /**
    Transform a line given in world coordinate frame to local coordinate
    frame of this terrain.
    \param line - the line in world coordinate frame
    \return the line in local coordinate frame of this terrain
    */
    agx::Line transformLineToTerrain(const agx::Line& line) const;

    /**
    Transform a line given in local coordinate frame to world coordinate
    frame of this terrain.
    \param line - the line in local coordinate frame
    \return the line in world coordinate frame of this terrain
    */
    agx::Line transformLineToWorld(const agx::Line& line) const;

    /**
    \return true if the specified voxel index is inside the terrain collision geometry bound
    */
    bool isVoxelIndexInsideGeometryBound(const agx::Vec3i& index) const;

    /**
    \return whether or not the specified terrain index is on border and if borders are locked.
    */
    bool isBorderIndexAndBordersLocked(const agx::Vec2i& terrainIndex) const;

    CompactionController* getCompactionController() const;

    TerrainDataAtlas* getTerrainDataAtlas() const;

    virtual void runBuildIslandsTask() override;

    void clearAllSoilParticles();

    /**
    * Internal method
    * Adds fluid mass in column to a terrain index.
    * \param terrainIndex- terrain index index.
    * \param fluidMass - the amount of fluid mass to add to terrain index.
    * \return the amount of fluid mass added to the terrain index.
    */
    float addFluidMassToColumn(const agx::Vec2i& terrainIndex, float fluidMass) const;

    /**
    * Internal method
    * Removes fluid mass in column given a terrain index.
    * \param terrainIndex- terrain index index.
    * \param fluidMass - the amount of fluid mass to remove from the terrain index.
    * \return the amount of fluid mass that could not be removed from the terrain.
    */
    float removeFluidMassFromColumn(const agx::Vec2i& terrainIndex, float fluidMass);

    /**
    * Internal method
    * Adds fluid mass to a voxel given a voxel index up to the maximum capacity of the voxel.
    * \param voxelIndex - openVDB voxel index.
    * \param fluidMass - the amount of fluid mass to add to voxel.
    * \return the amount of fluid mass added to the voxel.
    */
    float addFluidMass(const agx::Vec3i& voxelIndex, float fluidMass) const;

    /**
    * Internal method
    * Remove fluid mass to a voxel given a voxel index up to the maximum available fluid mass of the voxel.
    * \param voxelIndex - openVDB voxel index.
    * \param fluidMass - the amount of fluid mass to remove from voxel.
    * \return the amount of fluid mass that was removed from the voxel.
    */
    float removeFluidMass(const agx::Vec3i& voxelIndex, float fluidMass);

    /**
    Internal method

    \return the amount of fluid occupancy.
    \note - Used only for testing
    */
    agx::Real calculateFluidOccupancy() const;

    /**
    Internal method

    \return the total amount of soil particle occupancy.
    \note - Used only for testing
    */
    agx::Real getTotalSoilParticleOccupancy() const;

    /**
    * Internal method
    * \param voxelIndex - openVDB voxel index.
    * \return the position in terrain local space corresponding to the center of the given voxel.
    */
    agx::Vec3 getLocalPositionFromVoxelIndex(const agx::Vec3i& voxelIndex) const;

    /**
    * Internal method
    * \param voxelIndex - openVDB voxel index.
    * \return the position in world space corresponding to the center of the given voxel.
    */
    agx::Vec3 getWorldPositionFromVoxelIndex(const agx::Vec3i& voxelIndex) const;

    /**
    * Internal method
    * \param position - position in terrain local space.
    * \return the openVDB voxel index of the closest voxel to the given position.
    */
    agx::Vec3i getVoxelIndexFromLocalPosition(agx::Vec3 position) const;

    /**
    * Internal method
    * \param position - position in world space.
    * \return the openVDB voxel index of the closest voxel to the given position.
    */
    agx::Vec3i getVoxelIndexFromWorldPosition(const agx::Vec3& position) const;

    /**
    * Internal method
    * \param terrainIndex - terrain index
    * \return the openVDB voxel index at the surface at the given terrain index.
    */
    agx::Vec3i getSurfaceVoxelIndexFromTerrainIndex(const agx::Vec2i& terrainIndex) const;

    /**
    * Internal method
    * \param voxelXY - x- and y-coordinates of an openVDB voxel index.
    * \return the position in local terrain space of the surface in the given openVDB voxel column.
    */
    agx::Vec3 getSurfacePositionFromVoxelXY(const agx::Vec2i& voxelXY) const;

    /**
    * Internal method
    * \param voxelXY - x- and y-coordinates of an openVDB voxel index.
    * \return the position in world space of the surface in the given openVDB voxel column.
    */
    agx::Vec3 getSurfacePositionWorldFromVoxelXY(const agx::Vec2i& voxelXY) const;

    /**
    * Internal method
    * \param voxelXY - x- and y-coordinates of an openVDB voxel index.
    * \return the terrain index corresponding to the given openVDB voxel column.
    */
    agx::Vec2i getTerrainIndexFromVoxelIndexXY(const agx::Vec2i& voxelXY) const;

    /**
    * Internal method
    * \param terrainIndex - terrain index
    * \return the x- and y-coordinates of the openVDB voxel index corresponding to the given terrain index.
    */
    agx::Vec2i getVoxelIndexXYFromTerrainIndex(const agx::Vec2i& terrainIndex) const;

    /**
    * Internal method
    * \param voxelXY - x- and y-coordinates of an openVDB voxel index.
    * \return the openVDB voxel index corresponding to the surface voxel in the given openVDB voxel column.
    */
    agx::Vec3i getSurfaceVoxelIndexFromVoxelXY(const agx::Vec2i& voxelXY) const;

    /**
    * Internal method
    * Set the voxels where no DynamicMass can be created or moved into. Note that solidOccupancy that are removed in
    * these voxels will just disappear. So using this breaks mass conservation.
    * \param voxelIndices - Vector with voxelIndices where no DynamicMass should be created or moved to
    */
    void setForbiddenVoxelsForDynamicMass(const agx::Vec3iVector& voxelIndices);

    /**
    * Internal method
    * \param voxelIndices - Vector with voxelIndices where no DynamicMass should be created or moved to
    * \return true if there are any forbidden voxels
    */
    bool getForbiddenVoxelsForDynamicMass(agx::Vec3iVector& voxelIndices) const;

    /**
    * Internal method
    * Get the voxels where no DynamicMass can be created or moved into.
    * \param voxel - voxel to check if it is a forbidden voxel for dynamic mass
    * \return true if the
    */
    bool isForbiddenVoxelsForDynamicMass(const agx::Vec3i& voxelIndex) const;

    /**
    * Internal method
    * \param voxelIndex - openVDB voxel index.
    * \return whether given openVDB voxel index is occupied by a shovel geometry.
    */
    bool voxelContainsShovel(const agx::Vec3i& voxelIndex) const;

    /**
    * Internal method
    * Checks if a particle was created by this terrain.
    * \param particle - openVDB voxel index.
    * \return whether the specified particle was created by this terrain.
    */
    bool hasCreatedParticle( agx::Physics::ParticlePtr particle ) const;

    /**
    * Internal method
    * Checks if a particle has the same particle material as the terrain. This should indicate that the particle was
    * created from this terrain instance.
    * \param particle - openVDB voxel index.
    * \return whether the specified particle has the same particle mateiral as the terrain
    */
    bool hasSameParticleMaterial(agx::Physics::ParticlePtr particle) const;

    /**
    * Internal method
    * Using the TerrainMassController to modify the terrain data can invalidate the Terrain instance.
    * \return the TerrainMassController of the Terrain.
    */
    TerrainMassController* getMassController() const;

    /**
    Internal method
    Get the aggregate contact force with the terrain given an excavation mode and a shovel
    \param shovel - the specified shovel
    \param excavationMode - the excavation mode that the aggregate belongs too ( PRIMARY, DEFORM_BACK, DEFORM_RIGHT, DEFORM_LEFT )
    \return the total force (N) acting on the aggregate in the terrain-aggregate contact
    */
    agx::Vec3 getAggregateTerrainContactForce(const Shovel* shovel, Shovel::ExcavationMode excavationMode) const;

    /**
    Internal method
    Get the aggregate normal force with the terrain given an excavation mode and a shovel
    \param shovel - the specified shovel
    \param excavationMode - the excavation mode that the aggregate belongs too ( PRIMARY, DEFORM_BACK, DEFORM_RIGHT, DEFORM_LEFT )
    \return the total force (N) acting on the aggregate in the terrain-aggregate contact
    */
    agx::Vec3 getAggregateTerrainNormalForce(const Shovel* shovel, Shovel::ExcavationMode excavationMode) const;

    /**
    Internal method
    Get the aggregate normal and shear contact force with the terrain given an excavation mode and a shovel
    \param shovel - the specified shovel
    \param excavationMode - the excavation mode that the aggregate belongs too ( PRIMARY, DEFORM_BACK, DEFORM_RIGHT, DEFORM_LEFT )
    \return vector containing the geometry contacts between the specified aggregate and the terrain
    */
    agxCollide::GeometryContactPtrVector getAggregateTerrainContacts(const Shovel* shovel, Shovel::ExcavationMode excavationMode) const;

    void adjustFluidMassInGrid(agx::Vec3i binIndex, int binVoxels, float availableFluidMass, float remainingFluidMass);

    agx::Vec3iVector getVoxelsOccupiedByShovel(TerrainToolCollection *collection) const;

    bool voxelHasParticles(const agx::Vec3i& voxelIndex) const;

    bool isForbiddenIndex(const agx::Vec3i& index);

    void addTerrainIndexToAvalanche(const agx::Vec2i& index);

    int getLowestAllowableVoxelIndexZ() const;

    agx::Vec3 constructHeightFieldOffset() const;

    void setIsTerrainPagerTile(bool isPagerTile);

    bool getIsTerrainPagerTile() const;

    bool hasEnabledToolCollections() const;

    bool shouldUpdateTerrain() const;

    bool hasParticlesInTerrain() const;

    void debugRenderFluidMass();

    void clampHeights(agx::Real minValue);

    DOXYGEN_END_INTERNAL_BLOCK()

  protected:
    /**
    Default constructor used in serialization.
    */
    Terrain();

    /**
    Add notification executed when terrain is added to a simulation
    */
    virtual void addNotification() override;

    /**
    Remove notification executed when terrain is removed to a simulation
    */
    virtual void removeNotification() override;

    /**
    Executes pre-collide events for agxTerrain in the simulation.
    */
    virtual void preCollide() override;

    /**
    Executes pre-step events for agxTerrain in the simulation.
    */
    virtual void pre() override;

    /**
    Executes post-step events for agxTerrain in the simulation.
    */
    virtual void post() override;

    /**
    Callback to be executed at the end of the time step
    */
    virtual void last() override;

  protected:
    typedef struct offsetIndexInfo
    {
      agx::UInt size;
      agx::UInt groundParticleIndex;
      agx::UInt stopMergeIndex;
      agx::UInt offset;
      agx::UInt currentIndex;
    } offsetIndexInfo;

    using GoToStopCondition = std::function<bool(const agx::Vec3i& index)>;
    using ParticleIslandVector = agx::Vector<agx::Vector<std::pair<agx::Vec3i, offsetIndexInfo>>>;

    /// Surface voxel functions
    bool isSurfaceVoxel(const agx::Vec3i& index) const;
    bool isSolidBoundaryVoxel(const agx::Vec3i& index) const;
    bool isSubmergedVoxel(const agx::Vec3i& index) const;
    bool isFilledSolidVoxel(const agx::Vec3i& index) const;
    bool isSolidVoxel(const agx::Vec3i& index) const;
    bool isActiveSolidVoxel(const agx::Vec3i& index) const;
    bool hasVoxelCompaction(const agx::Vec3i& index) const;
    bool isVoxelWithinTerrainIndexBounds(const agx::Vec3i& index) const;
    bool getSurfaceVoxel(const agx::Vec3i& voxelIndex, agx::Vec3i& result) const;
    bool findPathToSurfaceVoxel(const agx::Vec3i& startVoxel, std::deque<agx::Vec3i>& result);
    bool findSurfaceVoxel(const agx::Vec3i& index, agx::Vec3i& result) const;
    agx::Vec3i goToVoxelKernel(const agx::Vec3i& start, const GoToStopCondition& stopCondition) const;
    agx::Vec3i goToSurfaceVoxel(const agx::Vec3i& start) const;
    agx::Vec3i goToBoundaryVoxel(const agx::Vec3i& start) const;
    void updateSurfaceHeightFieldFromDirtyVoxels();
    void updateSurfaceHeightFieldFromVoxelIndices(const agx::Vec3iVector& updateIndices);
    void updateSurfaceHeightFieldFromVoxelIndices(const agxData::LocalVector<agx::Vec3i>& updateIndices);
    void updateSurfaceHeightFieldFromVoxelIndex(const agx::Vec3i& voxelIndex);
    void updateSurfaceHeightFieldFromHeightsTable( agx::HashTable<agx::Vec2i, std::pair<agx::Vec3i, agx::Real> >& table );
    void traceGridSurfaceVoxel(const agx::Vec2i& voxelIndexXY, agx::Vec3i& surfaceVoxel) const;
    agx::Real traceGridSurfaceHeightFromVoxelIndex(const agx::Vec3i& index);
    agx::Real calculateSurfaceHeightFromVoxelIndex(const agx::Vec3i& voxelIndex);
    void initializeSurfaceHeightFieldFromVoxelGrid();
    void initVoxelStructureFromHeightField( bool adjustForCompaction = false );
    void convertVoxelsInActiveZonesToDynamicMass();
    bool shouldPrimaryActiveZoneRemoveMass(TerrainToolCollection* toolCollection);
    bool shouldDeformerActiveZoneRemoveMass(TerrainToolCollection* toolCollection, DeformerCollection* deformerCollection);
    void convertRemovedOccupancyToDynamicMass();
    void createSoilParticles();
    agx::Vec3 getValidParticlePosition(const agx::Real radius, const agx::Vec3i& index, const agx::Vec3& pos);
    void addDirtyVoxelsToAvalanche();
    void removeSoilParticlesOutsideTerrainBounds();
    void updateTransformationFromParentBody();
    void storeModifiedTerrainIndex( const agx::Vec2i& terrainVertex );
    void addModifiedParticleTerrainIndex(const agx::Vec2i& terrainIndex);
    void updateIndicesForAvalanche(TerrainIndexSet& forbiddenVertices);
    agx::Real findNewHeightFromVoxelIndex(const agx::Vec3i& index);

    SoilParticlePtrVector calculateParticlesInContactWithSurfaceGraph();
    void storeVoxelContactHeight( const agx::Vec2i& voxelXY, agx::Real currentHeight );
    void storeParticleGeometryContact( agx::Physics::ParticleGeometryContactPtr contact );

    bool _getIntersectingGridElements(
      BasicGrid* grid,
      agxCollide::Geometry* geometry,
      agx::Vec3iVector& gridPoints,
      bool isOccupied = true) const;

    void sortParticlesIntoVoxels();
    void sortParticlesIntoColumns();
    agx::HashTable<agx::Vec3i, agx::UInt64Vector> sortParticlesIntoBins(int binVoxels);

    void preSortSynchronizeParticleHeightField();
    void postSortSynchronizeParticleHeightField();

    // Sampling method
    void sampleParticleMassVelocity();

    void preUpdateTerrainContacts();
    void postUpdateCompaction();
    bool isDirty();
    void markVoxelToAvalanche(const agx::Vec3i& changedVoxel);
    void postSolveShovels();
    void updateOccupancyFromHeightChange(const agx::Vec2i& heightFieldIndex);
    void avalanche();
    void computeForbiddenBounds();
    void initializeFromHeightField();
    void sanitizeMaxDepthData(agx::Real maximumDepth);

    // Resize soil particles
    void resizeSoilParticles();
    void computeParticleGrowthTables(const agx::UInt64Vector& particleIndices, ParticleDeltaVector& particles);
    void growParticlesFromFluidMass(agx::Vec3i binIndex, int binVoxels, float availableFluidMass, ParticleDeltaVector& particles);
    void growSmallParticlesFromParticleMass(ParticleDeltaVector& particles, float particleRadiusUpperLimit);
    void growLargeParticlesFromParticleMass(ParticleDeltaVector& particles, float particleRadiusUpperLimit);

    // Merge soil particles functions
    void mergeSoilParticlesAndFluidMass();
    void addNeighbours(agxData::LocalVector<agx::Vec3i>& indexQueue, agx::Vec3i index);
    bool mergeFluidMass();
    void createParticleIslands( ParticleIslandVector& particleIslands );
    void mergeParticleIslands( const ParticleIslandVector& particleIslands, agxData::LocalVector<agx::Vec3i>& mergedParticleIndices );
    void resizeOrRemoveParticles( const agx::HashTable<SoilParticlePtr, std::array<agx::Real, 2>>& particlesToUpdate );
    agx::Real getMergeSpeedSquared() const;

    // Merge to cutting edge functions
    void mergeSoilToCuttingEdge();
    agx::Vec3iVector mergeCuttingEdgeColumn(agx::Vec3i voxel, size_t numVoxelsToSurfaceThreshold, agx::Plane cuttingPlane);
    float convertParticlesToSolidMass(agx::Vec3i voxel, ParticleWeightPairVector particles, float massToRemove,  agx::Vec3iVector& changedIndices);
    void moveParticlesToSurface(agx::Vec3iVector changedVoxelIndices);

    // Step fluid mass
    void stepFluidMass();
    void calculateFluidMassTransport(agx::HashTable<agx::Vec3i, float>& fluidMassTable);
    void applyFluidMassChanges(agx::HashTable<agx::Vec3i, float>& fluidMassTable);
    void enforceIncompressibilityColumnwise();
    agx::Vec3f getVoxelVelocity(const agx::Vec3i& voxelIndex, agx::HashTable<agx::Vec3i, size_t>& shovelVoxels);
    void createFluidMassInVoxel(const agx::Vec3i& voxelIndex, agx::Real32 amount);
    void createFluidMassOnTerrainSurface(const agx::Vec2i& terrainIndex, int surfaceVoxelOffset, agx::Real32 amount);

    // Conversion functions for particle radius and general occupancy
    float convertParticleRadiusToOccupancy(float radius) const;
    float convertOccupancyToParticleRadius(float occupancy) const;
    float getDynamicMassCompaction() const;
    bool soilParticleShouldMerge(SoilParticlePtr& ptr) const;
    bool particleIslandShouldMerge(const agx::Vector<std::pair<agx::Vec3i, offsetIndexInfo>>& island);
    bool voxelColumnHasParticles(const agx::Vec2i& vertex) const;
    void debugRenderGeometryContactVoxels() const;

    void calculateShovelVoxelIntersection();

    agx::HashVector<agx::Vec2i, agx::Int> getShovelHeightPerColumn();

    agx::Real convertOccupancyToMass(agx::Real occupnacy) const;
    agx::Real convertMassToOccupancy(agx::Real mass) const;

    agx::Real generateRandomNumber();
    agx::Vec3 generateRandomSoilParticlePositionPerturbation(agx::Real maxDelta);
    agx::Quat generateRandomSoilParticleRotation();

    void debugRenderForbiddenBounds();
    void debugRenderShovelForbiddenBounds();

    AvalancheController* getAvalancheController() const;

    agx::Bool isIndexWithinBoundsIssueWarning( const agx::Vec2i& terrainIndex ) const;

    void renderVoxelAABB(agx::Vec3i index, agx::Vec4f color);

    void toolCollectionsPreCollide();
    agx::TaskGroupRef createTask_preCollide();
    void initBuildIslandTask();

    const agx::AffineMatrix4x4& _getTransform() const;
    const agx::AffineMatrix4x4& _getInverseTransform() const;

    void moveEntityDataToCurrentThread();

    void sanityCheckHeightFieldAgainstMaxDepth();

    friend class AvalancheController;
    friend class SoilSimulationInterface;
    friend class TerrainMaterial;
    friend class SoilParticleAggregate;
    friend class TerrainGridControl;
    friend class TerrainMassController;
    friend class CompactionController;
    friend class TerrainPhysics;
    friend class TerrainCache;

  protected:
    virtual ~Terrain();

    Terrain(agxCollide::HeightField* heightField, agx::Real maximumDepth);

  private:
    using BoundingAABBVector             = agx::Vector<agxCollide::BoundingAABB>;
    using VoxelIndexOffsetIndexInfoTable = agx::HashTable<agx::Vec3i, offsetIndexInfo>;

  private:
    agxCollide::HeightFieldRef                m_surfaceHeightField;
    agxCollide::HeightFieldRef                m_particleHeightField;
    agx::Real                                 m_voxelSize;
    TerrainPropertiesRef                      m_properties;
    agx::UniformRealGenerator                 m_generator;
    float                                     m_nominalSoilParticleRadius;

    agxCollide::GeometryRef                   m_surfaceGeometry;
    agx::AffineMatrix4x4                      m_transform;
    agx::AffineMatrix4x4                      m_invTransform;
    agx::Vec3                                 m_heightFieldOffset;
    agx::Vec2i                                m_voxelToHeightFieldIndexSpaceOffset;

    TerrainDataAtlasRef                       m_terrainDataAtlas;
    TerrainMassControllerRef                  m_massController;
    CompactionControllerRef                   m_compactionController;
    TerrainContactGeneratorRef                m_terrainContactGenerator;
    AvalancheControllerRef                    m_avalancheController;
    SoilSimulationInterfaceRef                m_soilSimulationInterface;
    TerrainGridControlRef                     m_terrainGridInterface;

    TerrainMaterialRef                        m_terrainMaterial;
    agx::MaterialRef                          m_material;
    agx::MaterialRef                          m_particleMaterial;
    agx::MaterialRef                          m_aggregateMaterial;

    agx::ContactMaterialRef                   m_terrainParticleContactMaterial;
    agx::ContactMaterialRef                   m_terrainAggregateContactMaterial;
    agx::ContactMaterialRef                   m_particleParticleContactMaterial;

    TerrainToolCollectionRefVector            m_toolCollections;
    AvalancheController::TerrainIndexSet      m_terrainIndicesToAvalanche;

    agx::HashTable<agxCollide::Geometry*, agx::Real>     m_noMergeGeometryToDistance;
    size_t                                    m_noMergeEdgeMargin;
    bool                                      m_noMerge;
    BoundingAABBVector                        m_forbiddenBounds;
    agx::HashTable<agx::Vec3i, size_t>        m_voxelIndexToShovelIndex;
    agx::HashTable<TerrainToolCollection*, agx::Vec3iVector> m_voxelsInPrimaryActiveZoneLastTimeStep;
    agx::Vec3iVector                          m_dirtyVoxelIndices;
    agx::Real32Vector                         m_removedOccupancy;
    agx::Real                                 m_lastRemovedVolume;
    VoxelIndexSet                             m_voxelHasCreatedParticle;
    VoxelIndexSet                             m_forbiddenVoxelsForDynamicMass;
    TerrainIndexSet                           m_mergedVoxelXYVertices;
    ModifiedVerticesSetVector                 m_modifiedVoxelColumns;
    ModifiedVerticesSetVector                 m_modifiedTerrainVertices;
    ModifiedVerticesSetVector                 m_modifiedParticleVertices;
    ModifiedVerticesSetVector                 m_particleTerrainVertices;
    agx::HashTable<agx::Index, agx::Real>     m_particleToContactHeights;
    agx::HashTable<agx::Vec2i, agx::Real>     m_geometryContactHeights;

    size_t                                    m_numCreatedParticles;
    float                                     m_averageSoilParticleRadius;

    // Sort particles into voxels
    VoxelIndexOffsetIndexInfoTable            m_voxelParticleOffsetTable;
    ParticleWeightPairVector                  m_voxelParticleWeightPairVector;

    // Sort particles into columns
    VoxelIndexOffsetIndexInfoTable            m_columnParticleOffsetTable;
    ParticleWeightPairVector                  m_columnParticleWeightPairVector;

    agx::HashTable<agx::Vec2i, agx::Real>     m_columnParticleHeight;
    agx::HighAccuracyTimer                    m_timer;

    agx::IndexLambdaKernelRef                 m_indexKernel;

    bool                                      m_isTerrainPagerTile;
    bool                                      m_enable;

    // !HACK! !BEWARE! Islands partitioning
    // The build islands task is not executed when using PPGS.
    // No island partitioning means no parallelization of solving the rigid body system.
    // However, we still need islands in Terrain to determine which particles are in contact with the ground.
    // Thus, we manually load and execute the task in Terrain.
    // The task is loaded and bound for each Terrain that is added but is only executed once by agxSDK::Simulation.
    // The task is executed using the runBuildIslandsTask() method.
    agx::TaskRef m_buildIslandsTask;
    agx::TaskGroupRef m_preCollideUpdateTask;
    agx::TaskRef m_sortParticlesToVoxelsTask;
    agx::TaskRef m_sortParticlesToColumnsTask;
    agx::TaskRef m_calculateShovelVoxelIntersection;
    agx::TaskRef m_convertSolidToDynamicMassTask;

    agx::ParallelTaskRef m_updateHeightfieldsTask;
    agx::Job             m_updateHeightfieldsJobs[2];
    agx::ParallelTaskRef m_voxelCollisionTask;

    bool                                      m_enableFluidMerge; // Only used for testing. Should not be serialized.

    agx::Vector<agxCollide::ShapeRef> m_userInputFailureShapes;
  };


  AGX_FORCE_INLINE agx::Vec3i Terrain::getVoxelIndexFromLocalPosition(agx::Vec3 position) const
  {
    position *= (1.0 / m_voxelSize);
    // Note: ( position + 0.5 * m_voxelSize ) / m_voxelSize
    return agx::Vec3i((agx::Int)std::floor(position.x() + 0.5),
      (agx::Int)std::floor(position.y() + 0.5),
      (agx::Int)std::floor(position.z() + 0.5));
  }


  inline TerrainDataAtlas* Terrain::getTerrainDataAtlas() const
  {
    return m_terrainDataAtlas;
  }


  inline TerrainMaterial* Terrain::getTerrainMaterial() const
  {
    return m_terrainMaterial;
  }


  inline size_t Terrain::getResolutionX() const
  {
    return m_surfaceHeightField->getResolutionX();
  }


  inline size_t Terrain::getResolutionY() const
  {
    return m_surfaceHeightField->getResolutionY();
  }


  inline agx::Real Terrain::getElementSize() const
  {
    return m_voxelSize;
  }


  inline agx::Vec2i Terrain::getTerrainIndexFromVoxelIndexXY(const agx::Vec2i& voxelXY) const
  {
    return voxelXY + m_voxelToHeightFieldIndexSpaceOffset;
  }


  inline agx::Vec2i Terrain::getVoxelIndexXYFromTerrainIndex(const agx::Vec2i& terrainIndex) const
  {
    return terrainIndex - m_voxelToHeightFieldIndexSpaceOffset;
  }


  inline const agx::AffineMatrix4x4& Terrain::_getTransform() const
  {
    return m_transform;
  }



  inline const agx::AffineMatrix4x4& Terrain::_getInverseTransform() const
  {
    return m_invTransform;
  }


  inline agx::Vec3i Terrain::getSurfaceVoxelIndexFromTerrainIndex(const agx::Vec2i& terrainIndex) const
  {
    agx::Vec2i voxelXY = getVoxelIndexXYFromTerrainIndex(terrainIndex);

    auto solidGrid = static_cast<ColumnHeightGrid*>(m_terrainDataAtlas->getGridMassData(TerrainDataAtlas::MassType::SOLID));
    int voxelHeight = solidGrid->getSurfaceHeightIndex(terrainIndex);

    return agx::Vec3i(voxelXY.x(), voxelXY.y(), voxelHeight);
  }


  inline agx::Vec3 Terrain::getSurfacePositionFromVoxelXY( const agx::Vec2i& voxelXY ) const
  {
    return getSurfacePosition( getTerrainIndexFromVoxelIndexXY( voxelXY ) );
  }


  inline agx::Vec3 Terrain::getSurfacePositionWorldFromVoxelXY( const agx::Vec2i& voxelXY ) const
  {
    return getSurfacePositionWorld( getTerrainIndexFromVoxelIndexXY( voxelXY ) );
  }


  inline bool Terrain::getIsTerrainPagerTile() const
  {
    return m_isTerrainPagerTile;
  }

  inline bool Terrain::getDeformable() const
  {
    return m_properties->getEnableDeformation();
  }

}
