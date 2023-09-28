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

#ifndef AGX_PARTICLESYSTEM_H
#define AGX_PARTICLESYSTEM_H

#include <agx/Referenced.h>
#include <agx/Component.h>
#include <agx/Bound.h>
#include <agx/Physics/ParticleEntity.h>
#include <agx/Physics/ParticlePairContactEntity.h>
#include <agx/Physics/ParticleGeometryContactEntity.h>
#include <agx/Material.h>
#include <agx/Emitter.h>
#include <agxStream/Serializable.h>
#include <agx/Task.h>
#include <agxCollide/Geometry.h>
#include <agxCollide/Space.h>
#include <typeinfo>
#include <agx/Logger.h>

namespace agxData
{
  typedef agx::SymmetricPair<agx::Physics::ParticlePtr> ParticlePair;
}

namespace agxIO
{
  class Image;
}

namespace agxSDK
{
  class Simulation;
}

AGX_TYPE_BINDING( agxData::ParticlePair, "ParticlePair" )

namespace agx
{
  AGX_DECLARE_POINTER_TYPES( ParticleSystem );
  AGX_DECLARE_VECTOR_TYPES( ParticleSystem );
  typedef agx::SetVector< agx::ref_ptr<agx::ParticleSystem> > ParticleSystemRefSetVector;


  /**
  * A basic particle system that contains all the storages and buffers required for basic operation and also
  * a rendering task. It does not contain any update task. A simulated particle system is created by either
  * adding an update task using setUpdateTask(Task*), or by creating one of the pre-configured particle systems
  * available as sub-classes. Currently available are RigidParticleSystem and GranularBodySystem.
  *
  * ConstraintFluid is currently disabled.
  */
  class AGXPHYSICS_EXPORT ParticleSystem : public Component, public agxStream::Serializable
  {
  public:
    static Model* ClassModel();

    /**
     * Set the task that will be executed whenever it is time to step the simulation.
     */
    virtual bool setUpdateTask( Task* updateTask );

    /**
    \return The current update task.
    */
    Task* getUpdateTask();

    /**
    Triggers when particlesystem is added to the simulation
    */
    virtual void addNotification( agxSDK::Simulation* simulation );

    /**
    Create a new particle in the particle system.
    */
    Physics::ParticlePtr createParticle();

    /**
    Create a set of particles in the particle system.
    */
    agxData::EntityRange createParticles( size_t numParticles );

    /**
    * Returns a particle from the particle system given a particle id
    */
    Physics::ParticlePtr getParticle( agx::Index particleId );

    /**
    Destroy a particle.
    */
    void destroyParticle( Physics::ParticlePtr particle );

    /**
    Destroy a list of particles.
    */
    void destroyParticles( const Physics::ParticlePtrVector& particles );
    void destroyParticles( const agxData::EntityPtrVector& particles );

    /**
    Destroy a range of particles.
    */
    void destroyParticles( agxData::EntityRange particles );

    /**
    Tag a particle for removal at the end of the time step.
    \param particle - the specified particlePtr to mark for removal.
    \return - true if particle was successfully removed, false otherwise.
    \note - This is a more preferable way to remove particles than
            destroyParticle since it will not change the internal particle
            data structure during the time step
    */
    bool removeParticle( Physics::ParticlePtr particle );

    /**
    \return true if particle is not destroyed or marked to be removed
    */
    bool isValid( Physics::ParticlePtr particle ) const;

    /**
    \return true if particle geometry contact is valid
    */
    bool isParticleGeometryContactValid( Physics::ParticleGeometryContactPtr contact );

    /**
    \return true if particle-particle contact is valid
    */
    bool isParticlePairContactValid( Physics::ParticlePairContactPtr contact );

#if (!defined SWIG ) || ( defined SWIGCSHARP || defined SWIGJAVA)
    /**
    \return All active particles in the particle system (only valid until storage is modified, e.g. by deleting a particle).
    */
    const agx::Physics::ParticlePtrArray getParticles() const;
    agx::Physics::ParticlePtrArray getParticles();
#endif

    /// Spawn particles with defined positions
    agxData::EntityRange spawnParticles( const agx::Vec3Vector& positions );

    /**
    Spawns particles in a specified bound
    \param bound the specified bound to spawn in
    \param spacing spacing between the particle center points that are spawned
    \param jitterFactor the factor of the distance that is used to randomize particle positions
    \return A vector containing the created particles.
    */
    agx::Physics::ParticlePtrVector spawnParticlesInBound( const agx::Bound3& bound, Real radius, const agx::Vec3& spacing, agx::Real jitterFactor = agx::Real( 0 ) );

    /**
    Spawns particles In a specified bound from a distribution table
    \param bound The specified bound to spawn in
    \param sourceTable The particle model distribution table used to spawn particles
    \param spacing spacing between the particle center points that are spawned
    \param jitterFactor The factor of the distance that is used to randomize particle positions
    \return A vector containing the created particles.
    */
    agx::Physics::ParticlePtrVector spawnParticlesInBound( const agx::Bound3& bound, agx::Emitter::DistributionTable* sourceTable, const agx::Vec3& spacing, agx::Real jitterFactor = agx::Real( 0 ) );

    /**
    Spawns particles In a specified bound using a Hexagonal-Close-Packing (HCP) lattice
    \param bound The specified bound to spawn in
    \param radius The radius for the new particles
    \param spacing Spacing between the particle center points that are spawned
    \param jitterFactor The factor of the distance that is used to randomize particle positions
    \return A vector containing the created particles.
    */
    agx::Physics::ParticlePtrVector spawnParticlesInBoundHCP( const agx::Bound3& bound, Real radius, const agx::Vec3& spacing, agx::Real jitterFactor = agx::Real( 0 ) );

    /**
    Spawns particles in a specified bound from a distribution table using a Hexagonal-Close-Packing (HCP) lattice
    \param bound The specified bound to spawn in
    \param sourceTable The particle model distribution table used to spawn particles
    \param spacing Spacing between the particle center points that are spawned
    \param jitterFactor The factor of the distance that is used to randomize particle positions
    \return A vector containing the created particles.
    */
    agx::Physics::ParticlePtrVector spawnParticlesInBoundHCP( const agx::Bound3& bound, agx::Emitter::DistributionTable* sourceTable, const agx::Vec3& spacing, agx::Real jitterFactor = agx::Real( 0 ) );

    /**
    Spawns particles In a specified geometry
    \param geometry The specified geometry to spawn particles in
    \param spacing Spacing between the particle center points that are spawned
    \param jitterFactor The factor of the distance that is used to randomize particle positions
    \return A vector containing the created particles.
    */
    agx::Physics::ParticlePtrVector spawnParticlesInGeometry( agxCollide::Geometry* geometry, Real radius, const agx::Vec3& spacing, agx::Real jitterFactor = agx::Real( 0 ) );

    /**
    Spawns particles In a specified geometry using a Hexagonal-Close-Packing (HCP) lattice
    \param geometry The specified geometry to spawn particles in
    \param spacing Spacing between the particle center points that are spawned
    \param jitterFactor The factor of the distance that is used to randomize particle positions
    \return A vector containing the created particles.
    */
    agx::Physics::ParticlePtrVector spawnParticlesInGeometryHCP( agxCollide::Geometry* geometry, Real radius, const agx::Vec3& spacing, agx::Real jitterFactor = agx::Real( 0 ) );

    /**
    Spawns particles In a specified geometry from a distribution table
    \param geometry The specified geometry to spawn particles in
    \param sourceTable The particle model distribution table used to spawn particles
    \param spacing Spacing between the particle center points that are spawned
    \param jitterFactor The factor of the distance that is used to randomize particle positions
    \return A vector containing the created particles.
    */
    agx::Physics::ParticlePtrVector spawnParticlesInGeometry( agxCollide::Geometry* geometry, agx::Emitter::DistributionTable* sourceTable, const agx::Vec3& spacing, agx::Real jitterFactor = agx::Real( 0 ) );

    /**
    Spawns particles In a specified geometry from a distribution table using a Hexagonal-Close-Packing (HCP) lattice
    \param geometry The specified geometry to spawn particles in
    \param sourceTable The particle model distribution table used to spawn particles
    \param spacing Spacing between the particle center points that are spawned
    \param jitterFactor The factor of the distance that is used to randomize particle positions
    \return A vector containing the created particles.
    */
    agx::Physics::ParticlePtrVector spawnParticlesInGeometryHCP( agxCollide::Geometry* geometry, agx::Emitter::DistributionTable* sourceTable, const agx::Vec3& spacing, agx::Real jitterFactor = agx::Real( 0 ) );

    /**
    Colors particles with different levels in the vertical z-direction inside a specified bound
    \param bound The bound that is used to color the particles
    \param colors Vector containing the colors that will be used in the coloring
    */
    void colorParticlesWithLevelsInBound( const agx::Bound3& bound, const agx::Vec4fVector& colors );

    /**
    \return The current number of particles in the system.
    */
    size_t getNumParticles() const;

    /**
    \return The default particle radius of created particles.
    */
    Real getParticleRadius() const;

    /**
    \return The default particle mass. By default, the mass is derived from the assigned default agx::Material of the system.
    */
    Real getParticleMass() const;

    /**
    Set the default particle radius of created particles, with option of updating existing particles.
    \note - When updating the radius of existing particles, their mass properties will be updated
            according to the density of each individual particle material. This will override any manual
            mass value that has been set to the particles.
    \param radius - the specified default radius to set to the particle system.
    \param updateExistingParticles - true if the existing particles should be updated, false otherwise. ( Default: false )
    */
    void setParticleRadius( Real radius, bool updateExistingParticles = false );

    /**
    Set particle mass. Note that the default particle mass is normally derived from the default material
    of the particle system when particles are created. This function is used to set the default
    value of the mass buffer and also to manually set the mass on all particles.
    \note - Setting a material on either the particle system or an individual particle will override this mass value.
    \param mass - the specified mass to set as default and to all particles if that option is used.
    \param updateExistingParticles - true if the mass of all existing particles should be updated, false otherwise. ( Default: false )
    */
    virtual void setParticleMass( Real mass, bool updateExistingParticles = false );

    /**
    Set the default particle material. The default particle mass is derived from the density of the default material.
    \note - When updating the material of existing particles, their mass properties will be updated
            according to the radius of each individual particle. This will override any manual
            mass value that has been set to the particles.
    \param material - the specific material to set as the default particle material.
    \param updateExistingParticles - true if existing particles should be updated, false otherwise. ( Default: false )
    */
    void setMaterial( Material* material, bool updateExistingParticles = false );

    /**
    Removes and returns particles in the specified particle system outside the given 3D bound.
    \param bound - the specified 3D bound which all particle center positions
                   will be tested against.
    \return a vector with the successfully removed particles.
    */
    agx::Physics::ParticlePtrVector removeParticlesOutsideBound( const agx::Bound3& bound );

    /**
    \return The default particle material of created particles.
    */
    Material* getMaterial();
    const Material* getMaterial() const;

    /**
    * This should be called if the density of the bulk material is changed, so the particle default mass is updated accordingly
    */
    void evaluateParticleDefaultMass();

    /**
    \return The storage where the particle data is stored.
    */
    agxData::EntityStorage* getParticleStorage();

    /**
    \return The storage where the particle data is stored.
    */
    const agxData::EntityStorage* getParticleStorage() const;

    /**
    \return The storage for the particle pair contacts.
    */
    agxData::EntityStorage* getParticlePairContactStorage();

    /**
    \return The storage for Geometry-Particle contact lists.
    */
    agxData::EntityStorage* getGeometryParticleContactListStorage();

    /**
    \return The storage for Particle-Geometry contacts.
    */
    agxData::EntityStorage* getParticleGeometryContactStorage();

    /**
    Returns the current ParticleGeometryContacts in the simulation.
    */
    agx::Vector<agx::Physics::ParticleGeometryContactPtr> getParticleGeometryContacts();

    /**
    Returns the current ParticleParticle contacts in the simulation.
    */
    agx::Vector<agx::Physics::ParticlePairContactPtr> getParticleParticleContacts();

    /**
    Returns the number of particle-geometry contacts.
    */
    size_t getNumParticleGeometryContacts();

    /**
    Returns the number of particle-particle contacts.
    */
    size_t getNumParticleParticleContacts();

    /**
    Returns the number of ACTIVE particle-particle contacts, after the last conducted contact filtering.
    */
    size_t getNumActiveParticleParticleContacts();

    /**
    Returns the number of ACTIVE particle-geometry contacts, after the last conducted contact filtering.
    */
    size_t getNumActiveParticleGeometryContacts();

    /** Sums force magnitudes from particle system onto rigid body.
    Only valid for iterative solver.
    \param body The rigid body.
    */
    Vec3 sumForceMagnitudes( RigidBody* body );

    // Returns the total contact forces between the particles in the system
    agx::Real getTotalParticleParticleContactForces();

    // Returns the total particle-geometry contact forces in the system.
    agx::Real getTotalParticleGeometryContactForces();

    // Calculates the total particle volume in the particle system
    agx::Real calculateTotalParticleVolume() const;

    /// Calculates the total particle mass in the particle system
    agx::Real calculateTotalParticleMass() const;

    //// Collision culling

    /**
    Specify whether \p otherGeometry is allowed to collide with this geometry
    */
    void setEnableCollisions( const agxCollide::Geometry* geometry, bool flag );

    /**
    \return true if this geometry is allowed to collide with \p geometry.
    */
    bool getEnableCollisions( const agxCollide::Geometry* geometry ) const;

    /**
    \return The set of disabled geometry collisions.
    */
    const agxCollide::GeometryHashVector& getDisabledCollisions() const;


    /**
    \return true if the ParticleSystem can collide with a specified geometry.

    Following has to be true to make this method return true.

    - if they belong to a group pair that is not disabled
    - collisions for g1 is not disabled against g2 (g1->setEnableCollisions(g2, false)
    */
    // bool canCollide( const agxCollide::Geometry* geometry ) const;

    /**
    Add the specified \p id to a vector of group id, this will make a all newly created particles part
    of the group \p id. By default the particles are not part of any group.

    This can be used to partition the simulation where some groups cannot collide with other groups.
    Which groups that can collide or not is determined by Space and can be set using the method
    agxCollide::Space::setEnablePair( id1, id2, enable );
    */
    void addCollisionGroup( agx::UInt32 id, bool updateExistingParticles = true );
    void addCollisionGroup( const agx::Name& id, bool updateExistingParticles = true );

    /**
    Remove a group id from the vector of group ids.
    \param id - The group id to be removed
    */
    void removeCollisionGroup( agx::UInt32 id, bool updateExistingParticles = true );
    void removeCollisionGroup( const agx::Name& id, bool updateExistingParticles = true );

    /**
    Add the specified \p id to a vector of group id, this make the specific particle part
    of the group \p id. By default the specific particle is not part of any group.

    This can be used to partition the simulation where some groups cannot collide with other groups.
    Which groups that can collide or not is determined by Space and can be set using the method
    agxCollide::Space::setEnablePair( id1, id2, enable );
    */
    void addCollisionGroupParticle( agx::UInt32 id, agx::Physics::ParticlePtr ptr );
    void addCollisionGroupParticle( const agx::Name& id, agx::Physics::ParticlePtr ptr );

    /**
    Remove a group id from the vector of group ids.
    \param id - The group id to be removed
    */
    void removeCollisionGroupParticle( agx::UInt32 id, agx::Physics::ParticlePtr ptr );
    void removeCollisionGroupParticle( const agx::Name& id, agx::Physics::ParticlePtr ptr );

    /**
    This is performing a linear search among the group id for this particle.

    \param id - The group id we are looking for.
    \return true if the Geometry is part of the group \p id
    */
    bool hasCollisionGroupParticle( agx::UInt32 id, agx::Physics::ParticlePtr ptr ) const;
    bool hasCollisionGroupParticle( const agx::Name& id, agx::Physics::ParticlePtr ptr ) const;

    /**
    This is performing a linear search among the group id for this particle.

    \param id - The group id we are looking for.
    \return true if the Geometry is part of the group \p id
    */
    bool hasCollisionGroup( agx::UInt32 id ) const;
    bool hasCollisionGroup( const agx::Name& id ) const;

    /**
    \return a GroupSet with all the group id:s for the particle system
    */
    agx::Physics::CollisionGroupSetPtr getCollisionGroupSet() const;

    const agx::Bound3& getParticleBound();

    /**
    Returns true if the particle system has a buffer with the specified name and format.
    */
    bool hasCustomBuffer( const agx::String& buffername, const agx::String& formatName );

    /**
    Adds a custom attribute Real buffer for the particle system. Allows for setting custom
    properties of type Real for each particle in the particle system.
    \param buffername The name of the attribute buffer
    */
    void addCustomBufferReal( const agx::String& buffername );

    /**
    Sets a custom Real buffer value to a particle
    \param id Id of the particle
    \param buffername The name of the custom Real attribute buffer
    \param val The Real value to be set on the particle
    */
    void setCustomBufferValueReal( agx::Index id, const agx::String& buffername, agx::Real val );

    /**
    Returns a custom Real buffer value from a particle
    \param id Id of the particle
    \param buffername The name of the custom Real attribute buffer
    \return The custom Real value of the particle.
    */
    agx::Real getCustomBufferValueReal( agx::Index id, const agx::String& buffername );

    /**
    Adds a custom attribute Int buffer for the particle system. Allows for setting custom
    properties of type Int for each particle in the particle system.
    \param buffername The name of the attribute buffer
    */
    void addCustomBufferInt( const agx::String& buffername );

    /**
    Sets a custom Int buffer value to a particle
    \param id Id of the particle
    \param buffername The name of the custom Int attribute buffer
    \param val The Int value to be set on the particle
    */
    void setCustomBufferValueInt( agx::Index id, const agx::String& buffername, agx::Int val );

    /**
    Returns a custom Int buffer value from a particle
    \param id Id of the particle
    \param buffername The name of the custom Int attribute buffer
    \return The custom Int value of the particle.
    */
    agx::Int getCustomBufferValueInt( agx::Index id, const agx::String& buffername );

    /**
    Adds a custom attribute Real32 buffer for the particle system. Allows for setting custom
    properties of type Real32 for each particle in the particle system.
    \param buffername The name of the attribute buffer
    */
    void addCustomBufferReal32( const agx::String& buffername );

    /**
    Sets a custom Real32 buffer value to a particle
    \param id Id of the particle
    \param buffername The name of the custom Real attribute buffer
    \param val The Real32 value to be set on the particle
    */
    void setCustomBufferValueReal32( agx::Index id, const agx::String& buffername, agx::Real32 val );

    /**
    Returns a custom Real32 buffer value from a particle
    \param id Id of the particle
    \param buffername The name of the custom Real32 attribute buffer
    \return The custom Real value of the particle.
    */
    agx::Real32 getCustomBufferValueReal32( agx::Index id, const agx::String& buffername );

    /**
    Adds a custom attribute Vec3 buffer for the particle system. Allows for setting custom
    properties of type Vec3 for each particle in the particle system.
    \param buffername The name of the Vec3 attribute buffer to add
    */
    void addCustomBufferVec3( const agx::String& buffername );

    /**
    Sets a custom Vec3 buffer value to a particle
    \param id Id of the particle
    \param buffername The name of the custom Vec3 attribute buffer
    \param val The Vec3 value to be set on the particle
    */
    void setCustomBufferValueVec3( agx::Index id, const agx::String& buffername, agx::Vec3 val );

    /**
    Returns a custom Vec3 buffer value from a particle
    \param id Id of the particle
    \param buffername The name of the custom Vec3 attribute buffer
    \return The custom Vec3 value of the particle.
    */
    agx::Vec3 getCustomBufferValueVec3( agx::Index id, const agx::String& buffername );

    agx::Physics::ParticlePairContactPtr createParticlePairContact( agx::Physics::ParticlePtr p1, agx::Physics::ParticlePtr p2 );
    agx::Physics::ParticleGeometryContactPtr createParticleGeometryContact( agx::Physics::ParticlePtr particle, agx::Physics::GeometryPtr geometry );

    /// Destroy particle with specified id
    virtual void destroySingleParticle( agx::Index particleId );

    /// Set particle radius with specified id
    virtual void setSingleParticleRadius( agx::Index particleId, agx::Real radius );

    /// Get particle radius with specified id
    virtual agx::Real getSingleParticleRadius( agx::Index particleId );

    /// Set particle radius with specified id
    virtual void setSingleParticleMass( agx::Index particleId, agx::Real mass );

    /// Get particle radius with specified id
    virtual agx::Real getSingleParticleMass( agx::Index particleId );

    /// Set particle color with specified id
    virtual void setSingleParticleColor( agx::Index particleId, const agx::Vec4f& color );

    /// Get particle color with specified id
    virtual agx::Vec4f getSingleParticleColor( agx::Index particleId );

    /// Set particle velocity with specified id
    virtual void setSingleParticleVelocity( agx::Index particleId, const agx::Vec3& velocity );

    /// Get particle velocity with specified id
    virtual agx::Vec3 getSingleParticleVelocity( agx::Index particleId );

    /// Get particle position with specified id
    virtual agx::Vec3 getSingleParticlePosition( agx::Index particleId );

    /// Set particle force with specified id
    virtual void setSingleParticleForce( agx::Index particleId, const agx::Vec3& force );

    /// Get particle velocity with specified id
    virtual agx::Vec3 getSingleParticleForce( agx::Index particleId );

    /// Apply air drag to the particles according to formula: cd * velocity^2 + cr * velocity
    void applyAirForceDragSimple( agx::Real cd, agx::Real cr );

    /// Check if there is a particle with a specified id
    bool particleExistsWithId( agx::Index id );

    /**
    Return the particle id for the specified particle entity array index.
    \param index - specified entity array index.
    \return the particle id given a specified entity array index. Will return InvalidIndex if
            index is outside array range.
    */
    agx::Index getParticleIdFromIndex( agx::Index index ) const;

    /**
    Utility method for clearing all particles in the particle system.
    */
    void clearAllParticles();

    /**
    Update the shouldRender variable in particles given renders states such as clipping and
    filtering. This determines if an individual particle should be rendered or not.
    */
    void updateShouldRender();

    /**
    Executes the "commit" command on the buffer so that particle rendering can be updated properly.
    */
    void commitColorBuffer();

    /**
    Executes the "commit" command all buffers in the particle system. This is necessary if the
    raw data buffers been modified during a simulation where a journal is recorded.
    */
    void commitAllBuffers();

    /**
    Binds relevant buffer pointers to objects after inserting the particle system into the simulation.
    */
    void attach();

    /**
    Unbinds relevant buffer pointers from objects after removing the particle system
    into the simulation.
    */
    void detach();

    // private
    void _setGravityTask( Task* task );
    void setSpace( agxCollide::Space* space );
    void clearContactData();

    DOXYGEN_START_INTERNAL_BLOCK()

      agx::Vector<agx::Physics::ParticlePtr> createParticlesFromImage( agxIO::Image* image, agx::Real particleRadius, const agx::Vec3& center, agx::Real radiusJitter );

    /// Calculate volume packing
    static Vec3u calculateFilledBoundPacking( agx::Vec3Vector& result, agx::Bound3 bound, const agx::Vec3& spacing, agx::Real boundarySpacing = agx::Real( 0 ), agx::Real jitterFactor = agx::Real( 0 ) );
    static size_t calculateFilledBoundPackingHCP( agx::Vec3Vector& result, agx::Bound3 bound, const agx::Vec3& distance, agx::Real boundarySpacing = agx::Real( 0 ), agx::Real jitterFactor = agx::Real( 0 ) );
    static void calculateFilledGeometryPacking( agx::Vec3Vector& result, agxCollide::Geometry* geometry, const agx::Vec3& spacing, agx::Real boundarySpacing = agx::Real( 0 ), agx::Real jitterFactor = agx::Real( 0 ) );
    static void calculateFilledGeometryPackingHCP( agx::Vec3Vector& result, agxCollide::Geometry* geometry, const agx::Vec3& spacing, agx::Real boundarySpacing = agx::Real( 0 ), agx::Real jitterFactor = agx::Real( 0 ) );

    template<typename T>
    void addCustomBufferT( const agx::String& name );

    template<typename T>
    void setCustomBufferValueT( agx::Index id, const agx::String& buffername, const T valueT );

    template<typename T>
    T getCustomBufferValueT( agx::Index id, const agx::String& buffername );

    agxSDK::Simulation* getSimulation() const;

    agx::Physics::ParticlePtrVector convertRangeToParticlePtrVector( agxData::EntityRange& entityRange );

    AGXSTREAM_DECLARE_SERIALIZABLE( agx::ParticleSystem );

    DOXYGEN_END_INTERNAL_BLOCK()

  protected:
    ParticleSystem( const String& model, Device* device = CpuDevice::instance() );
    ParticleSystem(); // Only used for restoring from serialization

    virtual ~ParticleSystem();

    void init( bool createMaterial = true );
    void setCollisionGroupSet( agx::Physics::CollisionGroupSetPtr );
    void invalidLicenseCallback();
    TaskGroupRef m_updateTask;
    void setRadiusOnParticles( agxData::EntityRange& entityRange, agx::Real radius );

  protected:
    /**
    Particle material EventListener that is used during journal playback in order to restore particle material ptr from an uuid string.
    The journal does not support storage of MaterialPtrs, thus an uuid string is instead stored in the particles which
    is used to lookup the material in the simulation context and manually set it on particles when the material buffer
    has changed.
    */
    AGX_DECLARE_POINTER_TYPES( MaterialUuidBufferListener );
    class AGXPHYSICS_EXPORT MaterialUuidBufferListener : public agxData::Buffer::EventListener, public agx::Referenced
    {
    public:
      MaterialUuidBufferListener( ParticleSystem* system, agxSDK::Simulation* simulation );
      bool isJournalPlayback();
      void setSimulation( agxSDK::Simulation* sim );
      virtual void updateCallback( agxData::Buffer* buffer ) override;
      virtual void updateCallback( agxData::Buffer* buffer, agx::Index index ) override;
      virtual void updateCallback( agxData::Buffer* buffer, agx::IndexRange range ) override;
      virtual void updateCallback( agxData::Buffer* buffer, agxData::Array< agx::Index > indexSet ) override;

    protected:
      virtual ~MaterialUuidBufferListener();
      //////////////////////////////////////////////////////////////////////////
      // Variables
      //////////////////////////////////////////////////////////////////////////
    private:
      agx::observer_ptr<agx::ParticleSystem>  m_system;
      agxSDK::Simulation*                     m_simulation;
    };

    /**
    Used to apply enableRendering flag on particles loaded from old .agxJournals pre
    AGXSTREAM_ARCHIVE_MODIFICATION_EMITTER_STATE hash. Only added during restore of
    agx file that does not have the hash.
    */
    AGX_DECLARE_POINTER_TYPES( UpdateRenderingListener );
    class AGXPHYSICS_EXPORT UpdateRenderingListener : public agxData::Buffer::EventListener, public agx::Referenced
    {
    public:
      UpdateRenderingListener( ParticleSystem* system, agxSDK::Simulation* simulation );
      bool isJournalPlayback();
      void setSimulation( agxSDK::Simulation* sim );
      virtual void updateCallback( agxData::Buffer* buffer ) override;
      virtual void updateCallback( agxData::Buffer* buffer, agx::Index index ) override;
      virtual void updateCallback( agxData::Buffer* buffer, agx::IndexRange range ) override;
      virtual void updateCallback( agxData::Buffer* buffer, agxData::Array< agx::Index > indexSet ) override;

    protected:
      virtual ~UpdateRenderingListener();
      //////////////////////////////////////////////////////////////////////////
      // Variables
      //////////////////////////////////////////////////////////////////////////
    private:
      agx::observer_ptr<agx::ParticleSystem>  m_system;
      agxSDK::Simulation* m_simulation;
    };

  protected:
    agxData::EntityStorageRef m_particleStorage;
  private:
    agxData::EntityStorageRef m_particleCellStorage;
    agxData::EntityStorageRef m_cellBlockStorage;
    agxData::EntityStorageRef m_particleZoneStorage;
    agxData::EntityStorageRef m_collisionParticleStorage;
    agxData::EntityStorageRef m_particleGeometryContactStorage;
    agxData::EntityStorageRef m_particlePairContactStorage;
    agxData::EntityStorageRef m_geometryParticleContactListStorage;

    agxData::BufferRef m_materialUuidBuffer;
    agxData::BufferRef m_stateBuffer;
    agxData::BufferRef m_cellBlockPermutation;
    agxData::BufferRef m_deadCellBlocks;
    agxData::BufferRef m_particlesWithMissingTargetCell;
    agxData::BufferRef m_cellBlockOverlapPairs;
    agxData::ValueRef m_particleBound;
    agxData::ValueRef m_collisionGroupSet;
    agxData::ValueRef m_defaultRadius;
    agxData::ValueRef m_defaultMass;
    agxData::ValueRef m_defaultMaterial;

    MaterialRef m_material;
    TaskRef m_gravityTask;
    bool m_initialized;
    bool m_dirtyBound;

    agx::TaskRef m_updateBoundTask;

    agxCollide::GeometryHashVector m_disabledCollisionsHash;
    agxCollide::SpaceRef m_space;

    MaterialUuidBufferListenerRef m_uuidMaterialListenerRef;
    UpdateRenderingListenerRef m_updateRenderingListenerRef;

    agxData::ValueObserver m_invMassValue;
    void massChanged( agxData::Value* massValue );
    agxData::Value::Event::CallbackType m_massChangedCallback;
  };

  /* Implementation */
  AGX_FORCE_INLINE Task* ParticleSystem::getUpdateTask() { return m_updateTask; }

  AGX_FORCE_INLINE const agx::Physics::ParticlePtrArray ParticleSystem::getParticles() const
  {
    return m_particleStorage->getInstances<Physics::ParticlePtr>();
  }

  AGX_FORCE_INLINE agx::Physics::ParticlePtrArray ParticleSystem::getParticles()
  {
    return m_particleStorage->getInstances<Physics::ParticlePtr>();
  }

  AGX_FORCE_INLINE agxData::EntityStorage* ParticleSystem::getParticleStorage() { return m_particleStorage; }
  AGX_FORCE_INLINE const agxData::EntityStorage* ParticleSystem::getParticleStorage() const { return m_particleStorage; }
  AGX_FORCE_INLINE agxData::EntityStorage* ParticleSystem::getParticleGeometryContactStorage() { return m_particleGeometryContactStorage; }
  AGX_FORCE_INLINE agxData::EntityStorage* ParticleSystem::getGeometryParticleContactListStorage() { return m_geometryParticleContactListStorage; }
  AGX_FORCE_INLINE agxData::EntityStorage* ParticleSystem::getParticlePairContactStorage() { return m_particlePairContactStorage; }

  AGX_FORCE_INLINE const agxCollide::GeometryHashVector& ParticleSystem::getDisabledCollisions() const { return m_disabledCollisionsHash; }
  // AGX_FORCE_INLINE bool ParticleSystem::hasCollisionGroup( agx::UInt32 id ) const { return m_collisionGroups.contains(id); }
  inline agx::Physics::CollisionGroupSetPtr ParticleSystem::getCollisionGroupSet() const { return m_collisionGroupSet->get<Physics::CollisionGroupSetPtr>(); }

  // Real
  AGX_FORCE_INLINE void ParticleSystem::addCustomBufferReal( const agx::String& buffername ) { addCustomBufferT<agx::Real>( buffername ); }
  AGX_FORCE_INLINE void ParticleSystem::setCustomBufferValueReal( agx::Index id, const agx::String& buffername, agx::Real val ) { setCustomBufferValueT<agx::Real>( id, buffername, val ); }
  AGX_FORCE_INLINE agx::Real ParticleSystem::getCustomBufferValueReal( agx::Index id, const agx::String& buffername ) { return getCustomBufferValueT<agx::Real>( id, buffername ); }

  // Int
  AGX_FORCE_INLINE void ParticleSystem::addCustomBufferInt( const agx::String& buffername ) { addCustomBufferT<agx::Int>( buffername ); }
  AGX_FORCE_INLINE void ParticleSystem::setCustomBufferValueInt( agx::Index id, const agx::String& buffername, agx::Int val ) { setCustomBufferValueT<agx::Int>( id, buffername, val ); }
  AGX_FORCE_INLINE agx::Int ParticleSystem::getCustomBufferValueInt( agx::Index id, const agx::String& buffername ) { return getCustomBufferValueT<agx::Int>( id, buffername ); }

  // Real32
  AGX_FORCE_INLINE void ParticleSystem::addCustomBufferReal32( const agx::String& buffername ) { addCustomBufferT<agx::Real32>( buffername ); }
  AGX_FORCE_INLINE void ParticleSystem::setCustomBufferValueReal32( agx::Index id, const agx::String& buffername, agx::Real32 val ) { setCustomBufferValueT<agx::Real32>( id, buffername, val ); }
  AGX_FORCE_INLINE agx::Real32 ParticleSystem::getCustomBufferValueReal32( agx::Index id, const agx::String& buffername ) { return getCustomBufferValueT<agx::Real32>( id, buffername ); }

  // Vec3
  AGX_FORCE_INLINE void ParticleSystem::addCustomBufferVec3( const agx::String& buffername ) { addCustomBufferT<agx::Vec3>( buffername ); }
  AGX_FORCE_INLINE void ParticleSystem::setCustomBufferValueVec3( agx::Index id, const agx::String& buffername, agx::Vec3 val ) { setCustomBufferValueT<agx::Vec3>( id, buffername, val ); }
  AGX_FORCE_INLINE agx::Vec3 ParticleSystem::getCustomBufferValueVec3( agx::Index id, const agx::String& buffername ) { return getCustomBufferValueT<agx::Vec3>( id, buffername ); }
}

// AGX_PARTICLESYSTEM_H
#endif
