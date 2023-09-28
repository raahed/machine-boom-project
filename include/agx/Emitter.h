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

#include <agx/agxPhysics_export.h>
#include <agx/Physics/EmitterEntity.h>
#include <agxStream/Serializable.h>
#include <agx/Task.h>
#include <agxCollide/Geometry.h>
#include <agxCollide/Space.h>
#include <agx/Random.h>
#include <agx/SetVector.h>
#include <agx/BitState.h>

namespace agxSDK {
  class Simulation;
}

namespace agx
{
  AGX_DECLARE_POINTER_TYPES(Emitter);
  AGX_DECLARE_VECTOR_TYPES(Emitter);
  typedef agx::SetVector<ref_ptr<Emitter> >      EmitterRefSetVector;


  /**
  * Spawns new bodys inside a given volume.
  */
  class CALLABLE AGXPHYSICS_EXPORT Emitter : public SerialTask, public agxStream::Serializable
  {
  public:
    enum Quantity
    {
      QUANTITY_COUNT,
      QUANTITY_VOLUME,
      QUANTITY_MASS
    };

    AGX_DECLARE_POINTER_TYPES(DistributionModel);
    AGX_DECLARE_VECTOR_TYPES(DistributionModel);
    AGX_DECLARE_POINTER_TYPES(DistributionTable);
    AGX_DECLARE_POINTER_TYPES(EmitterData);

  public:

    /**
    * Create an emitter
    */
    Emitter(Quantity quantity = QUANTITY_COUNT);

    /**
    Set the volume that bodies will be spawned inside.
    */
    void setGeometry(agxCollide::Geometry* geometry);
    agxCollide::Geometry* getGeometry();


    /**
    Set the emitting quantity
    QUANTITY_COUNT - number of emitted objects
    QUANTITY_VOLUME - volume of emitted objects
    QUANTITY_MASS - mass of emitted objects
    */
    void setQuantity(Quantity quantity);

    /**
    \return The current emit quantity.
    */
    Quantity getQuantity() const;

    /**
    Set the rate in which the emitter emits objects, measured in the specified quantity
    QUANTITY_COUNT - number of objects / time unit
    QUANTITY_VOLUME - volume of objects / time unit
    QUANTITY_MASS - mass of objects / time unit
    */
    void setRate(Real rate);

    /**
    \return the rate by which this emitter spawns bodys
    */
    Real getRate() const;

    /**
    Set the initial velocity for the bodys when spawned, in local geometry coordinates
    \param velocity - Initial velocity
    */
    void setVelocity(const Vec3& velocity);

    /**
    \return the initial velocity (in local geometry coordinates) the spawned bodys should receive
    */
    const Vec3& getVelocity() const;

    /**
    Set the seed for the random number generator of the emitter
    */
    void setSeed(const agx::UInt32 seed);

    /**
    Set a distribution table to be used for body attributes.
    */
    void setDistributionTable(DistributionTable *table);

    /**
    Set if fixed position should be used on objects created from this emitter.
    See also setFixedPositionOffset.
    \param enable - true if a fixed position should be used when creating objects
                    from this emitter.
    */
    void setUseFixedPosition(bool enable);

    /**
    \return true if a fixed position should be when creating objects from the emitter.
    */
    bool getUseFixedPosition() const;

    /**
    Set if a fixed rotation should be used on objects created from this emitter.
    See also setFixedRotationOffset.
    \param enable - true if a fixed rotation should be used when creating objects
                    from this emitter.
    */
    void setUseFixedRotation( bool enable );

    /**
    \return true if fixed rotation should be used, false otherwise.
    */
    bool getUseFixedRotation() const;

    /**
    Set the fixed position offset of the objects created from the emitter.
    \note - getUseFixedPosition() must be true for this to be used.
    \param fixedPositionOffset - the fixed position offset from emitter center that
                                   should be used as the initial position of the
                                   created objects.
    */
    void setFixedPositionOffset( const agx::Vec3& fixedPositionOffset );

    /**
    Set the fixed rotation offset of the objects created from the emitter.
    \note - getUseFixedRotation() must be true for this to be used.
    \param fixedRotationOffset - the fixed initial rotation that should be used
                                 on objects created from the emitter.
    */
    void setFixedRotationOffset( const agx::Quat& fixedRotationOffset );

    /**
    * \return the fixed position offset of the objects created from the emitter.
    */
    agx::Vec3 getFixedPositionOffset() const;

    /**
    * \return the fixed rotation offset of the objects created from the emitter.
    */
    agx::Quat getFixedRotationOffset() const;

    /**
    \return The body distribution table. Can be nullptr.
    */
    DistributionTable *getDistributionTable();
    const DistributionTable *getDistributionTable() const;

    /**
    \return the number of bodies that this emitter has emitted
    */
    agx::UInt getNumEmitted() const;

    /**
    * Sets the maximum amount of objects that this emitter is allowed to emit.
    QUANTITY_COUNT - max number of objects
    QUANTITY_VOLUME - max total volume of objects
    QUANTITY_MASS - max total mass of objects
    */
    void setMaximumEmittedQuantity(const agx::Real max);

    /**
    * Returns the quantity amount that this emitter has emitted.
    */
    agx::Real getEmittedQuantity() const;

    /**
    * Get the maximum amount of quantity that this emitter is allowed to emit.
    */
    agx::Real getMaximumEmittedQuantity() const;

    /**
    Add the specified \p id to a vector of group id, this will make a bodies emitted from this emitter part
    of the group \p id. By default a body emitted is not part of any group.

    This can be used to partition the simulation where some groups cannot collide with other groups.
    Which groups that can collide or not is determined by Space and can be set using the method
    agxCollide::Space::setEnablePair(id1,id2);
    */
    void addCollisionGroup(agx::UInt32 id);
    void addCollisionGroup(const agx::Name& name);

    /**
    Remove a group id from the vector of group ids.
    \param id - The group id to be removed
    */
    void removeCollisionGroup(agx::UInt32 id);
    void removeCollisionGroup(const agx::Name& name);

    /**
    This is performing a linear search among the group id for this Emitter.

    \param id - The group id we are looking for.
    \return true if the Emitter has the group \p id
    */
    bool hasCollisionGroup(agx::UInt32 id) const;
    bool hasCollisionGroup(agx::Name id) const;

    /**
    \return a GroupSet with all the group id:s for the particle system
    */
    agx::Physics::CollisionGroupSetPtr getCollisionGroupSet() const;

    /**
    Sets whenever the render settings of emitted objects should be overriden. Mostly relevant
    for the "setEnableRenderEmittedObjects" settings.
    \param enable - true if the the emitter should override emitted object render settings. (Default: false)
    */
    void setEnableOverrideRenderSettings(bool enable);

    /**
    \return if the render settings of the emitted objects should be overriden. (Default: false)
    */
    bool getEnableOverrideRenderSettings() const;

    /**
    Sets whenever the emitted objects should be rendered or not, if possible. REQUIRES that
    getEnableOverrideRenderSettings is true.
    \param enable - whenever rendering should be enabled for spawned objects or not. (Default: true)
    note: The emitter uses the available render information on spawned objects
          to adjust this settings. It is the responsibility of Emitter child implementations
          to keep track and modify the render data for it's objects based on the flag data.
    */
    void setEnableRenderEmittedObjects(bool enable);

    /**
    \return whenever the emitted objects should be rendered or not, if possible. REQUIRES that
            "getEnableOverrideRenderSettings" is true. (Default: true)
    */
    bool getEnableRenderEmittedObjects() const;

    /**
    Set emitter name
    */
    void setName(const agx::String& name);

    agx::String getName() const;

    DOXYGEN_START_INTERNAL_BLOCK()

    Physics::EmitterPtr getEntity() const;

    virtual void transfer(agxSDK::Simulation *simulation);
    void setSpace(agxCollide::Space* space);

    AGXSTREAM_DECLARE_ABSTRACT_SERIALIZABLE(agx::Emitter);
    static agxStream::Serializable* create() { return nullptr; }
    static agxStream::Serializable* create(agxStream::InputArchive&) { return nullptr; }

    DOXYGEN_END_INTERNAL_BLOCK()

  protected:
    virtual void run() override;

    /// Default destructor
    virtual ~Emitter();

    virtual void emitBody(const Vec3& position,
                          const Quat& rotation,
                          const Vec3& velocity,
                          Emitter::DistributionModel *model) = 0;
    virtual bool preEmit() = 0;
    virtual void postEmit() = 0;
    virtual Real getDefaultElementQuantity(Quantity quantity);

    virtual void store( agxStream::OutputArchive& out ) const override;
    virtual void restore( agxStream::InputArchive& in ) override;

    void setEmittedQuantity(agx::Real quantity);
    void setEmittedCount(agx::UInt emittedCount);

    // Generates a random number from the generator in the emitter
    agx::Real generateRandomNumber();

    // State flags for emitter
    enum StateFlags : agx::UInt32
    {
      OVERRIDE_RENDER_SETTINGS = 1 << 0,
      SHOULD_RENDER_EMITTED_OBJECTS = 1 << 1,
      FIXED_POSITION = 1 << 2,
      FIXED_ROTATION = 1 << 3
    };
    using Flags = agx::BitState<StateFlags, agx::UInt32>;

  private:
    // Creates and initializes the parameters.
    void init();

    // Few helper functions for randomly selecting points in space.
    Vec3 randomPointInBox(const agxCollide::BoundingAABB* bound);
    Vec3 randomPointInSphere(agx::Real maxRadius);
    Vec3 randomPointInGeometry(agxCollide::Geometry* geometry, agxCollide::Geometry* proxySphere);

    void setCollisionGroupSet(agx::Physics::CollisionGroupSetPtr set);

  protected:
    Physics::EmitterPtr m_entity;

    DistributionTableRef m_distributionTable;
    agxCollide::GeometryRef m_geometry;
    agxCollide::GeometryRef m_proxySphere;
    agxCollide::SpaceRef m_space;
    ScalarParameterRef m_timestep;

    // Random number generator for body emitter
    UniformRealGenerator m_generator;

    agx::Vec3 m_fixedPositionOffset;
    agx::Quat m_fixedRotationOffset;

    Quantity m_quantity;
    Real m_rest;
    Flags m_state;
  };


  class AGXPHYSICS_EXPORT Emitter::DistributionModel : public Referenced, public virtual agxStream::Serializable
  {
  public:
    DistributionModel();
    DistributionModel(agx::Real probabilityWeight);

    /**
    Set the random probability weight.
    */
    void setProbabilityWeight(agx::Real weight);

    /**
    \return The random probability weight.
    */
    agx::Real getProbabilityWeight() const;


    /**
    \return The number of times this model was randomly chosen.
    */
    agx::UInt getActivationCount() const;

    agx::Real getVolume() const;
    agx::Real getMass() const;

    DOXYGEN_START_INTERNAL_BLOCK()

    AGXSTREAM_DECLARE_ABSTRACT_SERIALIZABLE(agx::Emitter::DistributionModel);
    static agxStream::Serializable* create() { return nullptr; }
    static agxStream::Serializable* create(agxStream::InputArchive&) { return nullptr; }

    DOXYGEN_END_INTERNAL_BLOCK()

  protected:
    virtual ~DistributionModel();
    virtual agx::Real calculateVolume() = 0;
    virtual agx::Real calculateMass() = 0;
    virtual void store( agxStream::OutputArchive& out ) const override;
    virtual void restore( agxStream::InputArchive& in ) override;

  private:
    friend class DistributionTable;
    void setDistribution(DistributionTable *distribution);

    static bool rangeCompare(DistributionModel *model, agx::Real value);

  protected:
    DistributionTable *m_distribution;
    agx::Real          m_probabilityWeight;
    agx::UInt          m_activationCount;
    agx::RangeReal     m_range;
    agx::Real          m_volume;
    agx::Real          m_mass;
  };

  class AGXPHYSICS_EXPORT Emitter::DistributionTable : public Referenced, public virtual agxStream::Serializable
  {
  public:
    DistributionTable(Quantity quantity = QUANTITY_COUNT);

    /**
    Add a particle model
    */
    void addModel(DistributionModel *model);

    /**
    \return the list of particle models.
    */
    const DistributionModelRefVector& getModels() const;

    /**
    \return A random particle model, using the weighted probabilities.
    */
    DistributionModel *getRandomModel();

    /**
    Set the seed of the random number generator in the distribution table
    */
    void setSeed(agx::UInt32 seed);

    /**
    \return The total random probability weight of all models.
    */
    agx::Real calculateTotalWeight() const;

    /**
    Set The probability quantity.
    */
    void setProbabilityQuantity(Quantity quantity);

    /**
    \return The probability quantity.
    */
    Quantity getProbabilityQuantity() const;

    DOXYGEN_START_INTERNAL_BLOCK()

    AGXSTREAM_DECLARE_SERIALIZABLE(agx::Emitter::DistributionTable);

    DOXYGEN_END_INTERNAL_BLOCK()

  protected:
    virtual ~DistributionTable();

  private:
    friend class DistributionModel;
    void rebuildProbabilityTable();

    agx::Real getQuantifiedProbabilityWeight(DistributionModel *model) const;
    agx::Real calculateTotalQuantifiedWeight() const;


  private:
    Quantity                              m_quantity;
    DistributionModelRefVector            m_models;
    UniformRealGenerator                  m_randomGenerator;
    agx::UInt32                           m_seed;
    bool                                  m_needRebuild;
  };

  /* Implementation */
  AGX_FORCE_INLINE agxCollide::Geometry* agx::Emitter::getGeometry() { return m_geometry; }

  inline agx::Physics::CollisionGroupSetPtr agx::Emitter::getCollisionGroupSet() const { return m_entity.collisionGroupSet(); }

  AGX_FORCE_INLINE bool agx::Emitter::hasCollisionGroup(agx::UInt32 id) const
  {
    return agxCollide::CollisionGroupManager::hasGroup(getCollisionGroupSet(), id);
  }

  AGX_FORCE_INLINE bool agx::Emitter::hasCollisionGroup(const agx::Name id) const
  {
    return agxCollide::CollisionGroupManager::hasGroup(getCollisionGroupSet(), id);
  }

  AGX_FORCE_INLINE void agx::Emitter::setCollisionGroupSet(agx::Physics::CollisionGroupSetPtr set)
  {
    m_entity.collisionGroupSet() = set;
  }
}

