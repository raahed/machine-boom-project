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

#include <agxTerrain/Shovel.h>
#include <agxTerrain/AggregateContactGenerator.h>
#include <agxTerrain/ShovelAggregateContactUtils.h>
#include <agxTerrain/DeformController.h>

#include <agxStream/Serializable.h>

namespace agxTerrain
{
  AGX_DECLARE_POINTER_TYPES( TerrainToolCollection );
  AGX_DECLARE_VECTOR_TYPES( TerrainToolCollection );

  /**
  Collection of terrain tool (currently Shovel) related objects which are
  terrain instance specific.
  */
  class AGXTERRAIN_EXPORT TerrainToolCollection : public agx::Referenced, public agxStream::Serializable
  {
  public:
    /**
    Construct given terrain and shovel.
    \param terrain - terrain instance this collection belongs to
    \param shovel - user shovel instance
    */
    TerrainToolCollection( Terrain* terrain, Shovel* shovel );

    /**
    \return terrain instance this collection belongs to
    */
    Terrain* getTerrain() const;

    /**
    \return user shovel instance
    */
    Shovel* getShovel() const;

    /**
    \return shovel active zone specific for this terrain instance
    */
    PrimaryActiveZone* getActiveZone() const;

    /**
    Assign active zone for the shovel in this terrain.
    \param activeZone - active zone instance
    */
    void setActiveZone( PrimaryActiveZone* activeZone );

    /**
    \return soil particle aggregate specific for this terrain instance
    */
    SoilParticleAggregate* getSoilParticleAggregate() const;

    /**
    Assign soil particle aggregate for the shovel in this terrain.
    \param soilParticleAggregate - soil particle aggregate instance
    */
    void setSoilParticleAggregate( SoilParticleAggregate* soilParticleAggregate );

    /**
    \return aggregate contact generator specific for this terrain instance
    */
    AggregateContactGenerator* getAggregateContactGenerator() const;

    /**
    Assign aggregate contact generator for the soil particle aggregate in this terrain.
    \param aggregateContactGenerator - aggregate contact generator instance
    */
    void setAggregateContactGenerator(AggregateContactGenerator* aggregateContactGenerator);

    /**
    \return aggregate contact generator specific for this terrain instance
    */
    DeformController* getDeformController() const;

    /**
    Assign aggregate contact generator for the soil particle aggregate in this terrain.
    \param deformController - aggregate contact generator instance
    */
    void setDeformController(DeformController* deformController);

    /**
    \return shovel penetration resistance specific for this terrain instance
    */
    SoilPenetrationResistance* getPenetrationResistance() const;

    /**
    Assign penetration resistance instance for the shovel in this terrain.
    \param penetrationResistance - penetration resistance instance
    */
    void setPenetrationResistance(SoilPenetrationResistance* penetrationResistance);

    /**
    \return the shovel geometries colliding with voxels in the terrain
    */
    const agxCollide::GeometryRefVector& getVoxelCollisionGeometries() const;

    /**
    Assign collision geometries used to test voxel overlaps.
    \param geometries - voxel collision geometries
    */
    void setVoxelCollisionGeometries( const agxCollide::GeometryRefVector& geometries );

    /**
    \return the ShovelAggregateContactMaterialContainer that contains all contact materials explicitly set for
            this tool collection.
    */
    ShovelAggregateContactMaterialContainer* getShovelTerrainContactMaterialContainer();


    /**
    \return parent frame located at model center of the shovel
    */
    agx::Frame* getParentFrame() const;

    /**
    \return true if initialized (initializes in addNotification) - otherwise false
    */
    agx::Bool initialized() const;

    /**
    \return true if the ToolCollection should update in preCollide, pre and post
    */
    agx::Bool shouldUpdate();

    /**
    Set if regular geometry contacts should be created between the shovel
    geometries and the associated terrain geometry.
    */
    void setEnableShovelTerrainGeometryContacts(bool enable);

    /**
    Set whenever the innerbody aggregate should have zero velocity before solve step as
    opposed to using particle or cached post-solve velocities from the previous time step.
    \param enable - true if the innerbody aggregate should have zero velocity, false otherwise.
    */
    void setZeroAggregateVelocity( bool enable );

    /**
    \return true if the innerbody aggregate should have zero velocity, false otherwise.
    */
    agx::Bool getSetZeroAggregateVelocity() const;

  public:
    /**
    \internal

    Add notification when a shovel is added to an initialized terrain
    or during addNotification of the terrain.
    \param simulation - simulation the terrain is part of
    */
    virtual agx::Bool initialize( agxSDK::Simulation* simulation );

    /**
    \internal

    Remove notification when a shovel is removed from an initialized terrain
    or when the terrain is being removed from a simulation.
    \param simulation - simulation the terrain is part of (or being removed from)
    */
    virtual void uninitialize( agxSDK::Simulation* simulation );

    /**
    Callback from terrain preCollide.
    */
    virtual void onPreCollide();

    /**
    Callback from terrain pre-solve.
    */
    virtual void onPre();

    /**
    Callback from terrain post-solve.
    */
    virtual void onPost();

    /**
    Check if any of the collision geometries are inside terrain bounds
    */
    bool collisionGeometriesAreInsideTerrain() const;

    /**
    Update ToolCollection if shovel enable has changed
    */
    void handleEnableChange();

    /**
    Set enable/disable of the ToolCollection
    */
    void enable(bool enabled);

    /**
    \return true if the ToolCollection is enabled
    */
    bool getEnable();

    /**
    \return true if shovel enable has changed, mismatching the collection internal shovel enable variable
    */
    bool shovelEnableHasChanged() const;

    /**
    Clear the active zone wedges of the primary active zones and deformers
    */
    void clearActiveZoneWedges() const;

    /**
    \return the current wedge aggregate in the specified collection given an excavation mode if it exists, nullptr otherwise.
    */
    const agx::RigidBody* getWedgeAggregate(Shovel::ExcavationMode excavationMode) const;

    /**
    \return the current wedge aggregate depth in the specified collection given an excavation mode.
    */
    agx::Real getAggregateTerrainContactDepth(Shovel::ExcavationMode excavationMode) const;

    DOXYGEN_START_INTERNAL_BLOCK()
      AGXSTREAM_DECLARE_SERIALIZABLE( agxTerrain::TerrainToolCollection );

    virtual void onPostRestore( Terrain* terrain );
    DOXYGEN_END_INTERNAL_BLOCK()

    protected:
      /**
      Default constructor used in serialization.
      */
      TerrainToolCollection();

      /**
      Reference counted object - protected destructor.
      */
      virtual ~TerrainToolCollection();

      enum StateFlags : agx::UInt32
      {
        ENABLED = 1 << 0,
        SHOVEL_ENABLED = 1 << 1,
        SET_ZERO_AGGREGATE_VELOCITY = 1 << 2
      };
      using Flags = agx::BitState<StateFlags, agx::UInt32>;

      bool getShovelEnabled() const;

      void setShovelEnabled( bool enabled );

      void setEnable( bool enabled );

    private:
      Terrain*                                   m_terrain;
      ShovelRef                                  m_shovel;
      PrimaryActiveZoneRef                       m_activeZone;
      SoilParticleAggregateRef                   m_soilParticleAggregate;
      AggregateContactGeneratorRef               m_aggregateContactGenerator;
      DeformControllerRef                        m_deformController;
      ShovelAggregateContactMaterialContainerRef m_shovelAggregateCMContainer;
      SoilPenetrationResistanceRef               m_penetrationResistance;
      Flags                                      m_flags;

      agx::FrameRef m_parentFrame; /**< Parent frame for shovel geometries synchronized with model center of shovel body. */
      agxCollide::GeometryRefVector m_voxelCollisionGeometries; /**< Geometries used to find shovel <-> voxel collisions. */
  };
}
