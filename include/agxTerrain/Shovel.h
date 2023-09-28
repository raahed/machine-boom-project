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

#include <agx/BitState.h>
#include <agxTerrain/PrimaryActiveZone.h>
#include <agxTerrain/SoilParticleAggregate.h>
#include <agxTerrain/SoilPenetrationResistance.h>

#include <agxStream/Serializable.h>

namespace agxCollide
{
  class Geometry;
  class Shape;
}

namespace agxTerrain
{
  class Terrain;
  class TerrainMaterial;
  class TerrainToolCollection;

  /**
  Shovel object used to interact with the terrain via an active zone that converts solid terrain to dynamic terrain
  which can be moved by the shovel rigid body.

  A shovel object consists of a rigid body and two edges, one top edge and a bottom cutting edge. These edges, which together
  with a fracture angle, constructs the active zone in front of the shovel object. As the shovel moves, the active zone converts
  solid mass into dynamic mass, consisting and internal fluid and particle mass. The dynamic mass is simulated in an internal
  simulation and is kinematically affected by objects that are added to it. The shovel receives feedback forces through a rigid
  body aggregate constructed from the inertia of soil particles in the active zone.
  */
  AGX_DECLARE_POINTER_TYPES(Shovel);
  AGX_DECLARE_VECTOR_TYPES(Shovel);
  class AGXTERRAIN_EXPORT Shovel : public agx::Referenced, public agxStream::Serializable
  {
  public:
    class ExcavationSettings;


    /**
    Basic constructor.
    \param shovelBody - The base body of the shovel.
    \param topEdge - The top edge of the active zone, specified in the LOCAL shovelBody frame
                     that will be created in front of the shovel.
    \param cuttingEdge - The lowest edge on the shovel, specified in the LOCAL shovelBody frame
                         that serves as the cutting edge of the active zone.
    \param cuttingDirection - The cutting direction of the shovel where the penetration resistance will be active,
                              which is usually parallel to the lowest shovel plate that is used to initially
                              penetrate the soil.
    */
    Shovel(agx::RigidBody* shovelBody,
           const agx::Line& topEdge,
           const agx::Line& cuttingEdge,
           const agx::Vec3& cuttingDirection);

    /**
    Enum for describing the different excavation modes of a shovel:
      PRIMARY - The primary mode of excavation, where the shovel digs along the cutting
                direction in the terrain horizontal plane.
      DEFORM_BACK  - The mode of excavation where the shovel is digging in the opposite
                     direction of the cutting direction in the terrain horizontal plane.
                     ( Example: Backside grading )
      DEFORM_LEFT  - The mode of excavation where the shovel is digging to the counter-clock wise
                     (Left) direction orthogonal to the cutting direction in the terrain horizontal plane.
                     ( Example: Side push/grading )
      DEFORM_RIGHT - The mode of excavation where the shovel is digging to the clock wise
                     (Right) direction orthogonal to the cutting direction in the terrain horizontal plane.
                     ( Example: Side push/grading )
    */
    enum class ExcavationMode : agx::UInt32
    {
      PRIMARY = 0,
      DEFORM_BACK = 1,
      DEFORM_RIGHT = 2,
      DEFORM_LEFT = 3
    };

    /**
    \return the base rigid body from the shovel object.
    */
    agx::RigidBody* getRigidBody() const;

    /**
    \return the excavation settings for an excavation mode in the shovel.
    */
    ExcavationSettings& getExcavationSettings(ExcavationMode mode);

    /**
    \returns the top edge in local space.
    */
    agx::Line getTopEdge() const;

    /**
    \returns the top edge in world space.
    */
    agx::Line getTopEdgeWorld() const;

    /**
    \returns the cutting edge in local space.
    */
    agx::Line getCuttingEdge() const;

    /**
    \returns the cutting edge in world space.
    */
    agx::Line getCuttingEdgeWorld() const;

    /**
    \returns the forward vector in local space.
    */
    agx::Vec3 getCuttingDirection() const;

    /**
    \returns the forward vector in world space.
    */
    agx::Vec3 getCuttingDirectionWorld() const;

    /**
    \returns the tooth length of the shovel.
    */
    agx::Real getToothLength() const;

    /**
    \returns the minimum radius of the shovel teeth.
    */
    agx::Real getToothMinimumRadius() const;

    /**
    \returns the maximum radius of the shovel teeth.
    */
    agx::Real getToothMaximumRadius() const;

    /**
    \returns the teeth count of the shovel.
    */
    agx::UInt getNumberOfTeeth() const;

    /**
    \return the computed cutting edge length of the Shovel.
    */
    agx::Real computeCuttingEdgeLength() const;

    /**
    Set a settings object for an ExcavationMode of the shovel. Excavation modes are as follows:
      PRIMARY - The primary mode of excavation, where the shovel digs along the cutting
                direction in the terrain horizontal plane.
      DEFORM_BACK  - The mode of excavation where the shovel is digging in the opposite
                     direction of the cutting direction in the terrain horizontal plane.
                     ( Example: Backside grading )
      DEFORM_LEFT  - The mode of excavation where the shovel is digging to the counter-clock wise
                     (Left) direction orthogonal to the cutting direction in the terrain horizontal plane.
                     ( Example: Side push/grading )
      DEFORM_RIGHT - The mode of excavation where the shovel is digging to the clock wise
                     (Right) direction orthogonal to the cutting direction in the terrain horizontal plane.
                     ( Example: Side push/grading )
    \param mode - The specified excavation mode that the settings will apply to.
    \param excavationSettings - The specified excavation settings that will apply to the excavation mode.
    */
    void setExcavationSettings( ExcavationMode mode, ExcavationSettings excavationSettings );

    /**
    Set the top edge in local space.
    \param topEdge - new top edge in local space.
    */
    void setTopEdge(agx::Line topEdge);

    /**
    Set the cutting edge in local space.
    \param cuttingEdge - new cutting edge in local space.
    */
    void setCuttingEdge(agx::Line cuttingEdge);

    /**
    \param cuttingDirection - new cutting direction in local space.
    */
    void setCuttingDirection(agx::Vec3 cuttingDirection);

    /**
    \param length - new tooth length.
    */
    void setToothLength(agx::Real length);

    /**
    \param radius - new minimum radius of the shovel teeth.
    */
    void setToothMinimumRadius(agx::Real radius);

    /**
    \param radius - new maximum radius of the shovel teeth.
    */
    void setToothMaximumRadius(agx::Real radius);

    /**
    \param count - new teeth count of shovel.
    */
    void setNumberOfTeeth(agx::UInt count);

    /**
    Set the extension outside the shovel bounding box where soil particle merging is forbidden.
    */
    void setNoMergeExtensionDistance(agx::Real extensionDistance);

    /**
    Set the minimum submerged cutting edge length fraction (0-1) that generates submerged cutting. Default: 0.5
    */
    void setMinimumSubmergedContactLengthFraction(agx::Real minimumSubmergedContactLengthFraction);

    /**
    \return get current separation plane for each excavation mode
    */
    agx::Plane getCurrentSeparationPlane( Terrain* terrain, ExcavationMode mode ) const;

    /**
    Get the minimum submerged cutting edge length fraction (0-1) that generates submerged cutting. Default: 0.5
    */
    agx::Real getMinimumSubmergedContactLengthFraction() const;

    /**
    Get the margin outside the shovel bonding box where soil particle merging is forbidden.
    */
    agx::Real getNoMergeExtensionDistance() const;

    /**
    Sets the vertical distance under the blade cutting edge that the soil is
    allowed to instantly merge up to.
    */
    void setVerticalBladeSoilMergeDistance(agx::Real verticalSoilBladeMergeDistance);

    /**
    Sets the dead-load limit where secondary separation will start to activate where the forward direction
    starts to change according to the virtual separation plate created by the material inside the shovel
    \param secondarySeparationLimit - The dead load limit where secondary separation will start to activate (Default: 0.8)
    */
    void setSecondarySeparationDeadloadLimit( agx::Real secondarySeparationLimit );

    /**
    Get the vertical distance under the blade cutting edge that the soil is
    allowed to instantly merge up to.
    */
    agx::Real getVerticalBladeSoilMergeDistance() const;

    /**
    Get the soil penetration model used for the specified shovel in the given terrain. Available models are:

      NO_MODEL - Only use the soil pressure above the cutting edge to determine the penetration resistance.
      ELASTIC_PLASTIC_LIMIT - Calculate the expected penetration resistance based in the elastic plastic limit. (Default)
      PLASTIC - Calculate the expected penetration resistance based on plastic expansion due to tooth pressure.

    \param model - The output variable where the penetration model that is used will be put, if successful.
    \param terrain - The specified terrain where this model applies.
    \return the specified if the model was successfully extracted, false otherwise.
    */
    bool getSoilPenetrationModel(agxTerrain::Terrain* terrain, agxTerrain::SoilPenetrationResistance::PenetrationModel& model) const;

    /**
    Set the vertical penetration depth threshold for when the shovel tooth for penetration
    resistance should reach full effectiveness. The penetration depth is defined as the vertical
    distance between the tip of a shovel tooth and the surface position of the height field. The
    penetration resistance will increase from a baseline of 10% until maximum effectiveness is reached
    when the vertical penetration depth of the shovel reaches the specified value. (Default: 0.5 m)
    \param depthThreshold - the vertical pressure threshold that will cause penetration
                            resistance to start.
    */
    void setPenetrationDepthThreshold(agx::Real depthThreshold);

    /**
    Set the soil penetration model used for the specified shovel in the given terrain. Available models are:

      NO_MODEL - Only use the soil pressure above the cutting edge to determine the penetration resistance.
      ELASTIC_PLASTIC_LIMIT - Calculate the expected penetration resistance based in the elastic plastic limit. (Default)
      PLASTIC - Calculate the expected penetration resistance based on plastic expansion due to tooth pressure.

    \param model - The specified penetration model to use in the shovel against the terrain. ( Default: ELASTIC_PLASTIC_LIMIT )
    \param terrain - The specified terrain where this model will apply.
    \return true if the model was successfully changed, false otherwise.
    */
    bool setSoilPenetrationModel(agxTerrain::SoilPenetrationResistance::PenetrationModel model, agxTerrain::Terrain* terrain);

    /**
    Get the vertical penetration depth threshold for when the shovel tooth for penetration
    resistance should reach full effectiveness. The penetration depth is defined as the vertical
    distance between the tip of a shovel tooth and the surface position of the height field. The
    penetration resistance will increase from a baseline of 10% until maximum effectiveness is reached
    when the vertical penetration depth of the shovel reaches the specified value. (Default: 0.2 m)
    \return the vertical depth penetration threshold that will cause penetration resistance to
            reach full effectiveness.
    */
    agx::Real getPenetrationDepthThreshold() const;

    /**
    Set the linear scaling coefficient for the penetration force (Default: 1.0)
    \param penetrationForceScaling - The coefficient for scaling the penetration force that the terrain will give on this shovel
    */
    void setPenetrationForceScaling(agx::Real penetrationForceScaling);

    /**
    \return the linear scaling coefficient for the penetration force (Default: 1.0)
    */
    agx::Real getPenetrationForceScaling() const;

    /**
    Set true/false if the shovel deformers should make particle free deformations. (Default: false)
    Note, if this is true all excavation modes will make particle free deformations.
    Even if enableCreateDynamicMass is set to false for one or more excavation modes.
    \param enable - true if the shovel deformers mode should make particle free deformations.
    */
    void setEnableParticleFreeDeformers(bool enable);

    /**
    \return whether the shovel deformers should create dynamic mass.
    */
    bool getEnableParticleFreeDeformers() const;

    /**
    Set to true/false if shovel <-> terrain contacts should always be removed
    param enable - set to true/false if shovel <-> terrain contacts should always be removed
    */
    void setAlwaysRemoveShovelContacts(bool enable);

    /**
    \return true/false if shovel <-> terrain contacts are removed.
    */
    bool getAlwaysRemoveShovelContacts() const;

    /**
    Set a maximum limit on penetration force (N) that the terrain will generate on this shovel. (Default: Infinity)
    \param maxPenetrationForce - The maximum penetration force that the terrain will act on this shovel
    */
    void setMaxPenetrationForce(agx::Real maxPenetrationForce);

    /**
    Set the maximum limit on penetration force (N) that the terrain will generate on this shovel. (Default: Infinity)
    \return The maximum penetration force that the terrain will act on this shovel
    */
    agx::Real getMaxPenetrationForce() const;

    /**
    \return the dead-load limit where secondary separation will start to active where the forward direction
            starts to change according to the virtual separation plate created by the material inside the shovel
    */
    agx::Real getSecondarySeparationDeadloadLimit() const;

    /**
    \return the secondary cutting direction of the shovel in a specified terrain
    */
    agx::Vec3 getSecondaryCuttingDirection( Terrain* terrain ) const;

    /**
    \return the secondary separation forward vector of the shovel in a specified terrain
    */
    agx::Real getSecondarySeparationAngle( Terrain* terrain ) const;

    /**
    Get the last computed dead load fraction of the shovel, i.e how much of it's inner volume that is filled with dynamic soil.
    The dead load fraction ranges from 0.0 (empty), to 1.0 (full).
    \return the last computed dead load fraction
    */
    agx::Real getDeadLoadFraction() const;

    /**
    Get the last computed inner contact area of the shovel, i.e the estimated cross-section area
    of the inner volume that is filled with dynamic soil.
    \note - This is used in the shovel-aggregate contact in primary excavation to calculate stiffness and cohesion.
    \return the last computed inner contact area of the shovel.
    */
    agx::Real getInnerContactArea() const;

    /**
    Set the distance threshold from the shovel planes where regular geometry contacts between
    the shovel underside and the terrain are allowed to be created. ( Default: cuttingEdgeLength / 10.0 )
    \param bottomContactThreshold - The contact distance threshold from the shovel planes where regular
                                    geometry contacts between the shovel underside and the terrain are
                                    allowed to be created.
    */
    void setBottomContactThreshold( agx::Real bottomContactThreshold );

    /**
    Get the distance threshold from the shovel planes where regular geometry contacts between
    the shovel underside and the terrain are allowed to be created. ( Default: cuttingEdgeLength / 10.0 )
    */
    agx::Real getBottomContactThreshold() const;

    /**
    Set if inner shape alone should always create dynamic mass. The alternative is to only
    create dynamic mass in the inner shape when primary excavation soil wedges create mass.
    Default: true.
    \param enable - true if inner shape should always create dynamic mass (default), false otherwise.
    */
    void setEnableInnerShapeCreateDynamicMass( bool enable );

    /**
    Get if inner shape alone should always create dynamic mass. The alternative is to only
    create dynamic mass in the inner shape when primary excavation soil wedges create mass.
    \return true if inner shape should always create dynamic mass (default), false otherwise.
    */
    bool getEnableInnerShapeCreateDynamicMass() const;

    /**
    Set whenever the excavation force feedback during PRIMARY excavation should be
    generated from particle contacts instead of aggregate contacts.
    \param enable - true if particles should generate contact
                    forces on the shovel, false otherwise. (Default: false)
    */
    void setEnableParticleForceFeedback( bool enable );

    /**
    \return whenever the excavation force feedback during PRIMARY excavation should be
    generated from particle contacts instead of aggregate contacts. (Default: false)
    */
    bool getEnableParticleForceFeedback();

    /**
    Change state enable of this shovel
    Default: true.
    \param enable - true to enable, false to disable
    */
    void setEnable(bool enable);

    /**
    Access the state enable flag.
    \return true if the body is enabled (default) - otherwise false
    */
    bool getEnable() const;

  public:

    /**
    Class containing the settings for the different ExcavationModes for a shovel.
    */
    class AGXTERRAIN_EXPORT ExcavationSettings
    {
    public:
      /**
      Default constructor
      */
      ExcavationSettings();

      /**
      Constructor taking an internal state variable from another excavation setting object.
      \param state - The internal state data from another excavation setting object.
      */
      ExcavationSettings(agx::UInt32 state);

      /**
      Set whether the excavation mode should be enabled, creating dynamic mass and generating force feedback.
      \param enable - true/false if the excavation mode should generate force feedback and create dynamic mass.s
      */
      void setEnable(bool enable);

      /**
      Set true/false if the excavation mode should create dynamic mass.
      \param enable - true if the excavation mode should create dynamic mass.
      */
      void setEnableCreateDynamicMass(bool enable);

      /**
      Set true/false if the excavation mode should generate force feedback from created aggregates.
      \param enable - true if the excavation mode should generate force feedback from created aggregates.
      */
      void setEnableForceFeedback(bool enable);

      /**
      \return whether the excavation mode should be enabled, creating dynamic mass and generating force feedback.
      */
      bool getEnable() const;

      /**
      \return whether the excavation mode should generate force feedback from created aggregates.
      */
      bool getEnableForceFeedback() const;

      /**
      \return whether the excavation mode should create dynamic mass.
      */
      bool getEnableCreateDynamicMass() const;

      /**
      \return the internal state variable for the shovel settings.
      */
      agx::UInt32 getState() const;

      void store( agxStream::OutputArchive& out ) const { m_settings.store( out ); }
      void restore( agxStream::InputArchive& in ) { m_settings.restore( in ); }

    private:
      enum StateFlags : agx::UInt32
      {
        ENABLED = 1 << 0,
        ENABLE_CREATE_DYNAMIC_MASS = 1 << 1,
        ENABLE_FORCE_FEEDBACK = 1 << 2,
      };
      using Flags = agx::BitState<StateFlags, agx::UInt32>;
      Flags m_settings;
    };

   public:

    DOXYGEN_START_INTERNAL_BLOCK()
    /**
    Add notification when a shovel is added to an initialized terrain or when
    the terrain is added to a simulation (i.e., on Terrain::addNotification).
    */
    agx::Bool initializeCollection( TerrainToolCollection* collection, agxSDK::Simulation* simulation );

    /**
    Remove notification when this shove is removed from a terrain or when the terrain is
    being removed from a simulation.
    */
    void uninitializeCollection( TerrainToolCollection* collection, agxSDK::Simulation* simulation );

    /**
    Callback from terrain pre-solve.
    \param terrain - terrain instance
    */
    void onPre( TerrainToolCollection* collection );

    /**
    Callback from terrain post-solve.
    \param terrain - terrain instance
    */
    void onPost( TerrainToolCollection* collection );

    bool getShovelPlanes(TerrainToolCollection* collection,
                         agx::Plane& bottomPlane,
                         agx::Plane& backPlane,
                         agx::Plane& leftPlane,
                         agx::Plane& rightPlane);

    void removeTerrainShovelContacts(TerrainToolCollection* collection);

    void removeTerrainShovelContactsDeformer(TerrainToolCollection* collection);

    bool isEdgeSubmerged(const agx::Line& edge, const Terrain* terrain) const;

    bool isEdgeSubmergedInSolid(const agx::Line& edge, const Terrain* terrain) const;

    bool isEdgeSubmergedInHeightfield(const agx::Line& edge, const agxCollide::HeightField* heightfield, const Terrain* terrain) const;

    agx::Real getSubmergedEdgeLength(const agx::Line& edge, const Terrain* terrain) const;

    agx::Vec3 getCuttingEdgeVelocity();

    void correctShovelVectors();

    void calculateDeadLoadFraction( const Terrain* terrain,
                                    const ActiveZone* activeZone,
                                    const agx::Physics::GranularBodyPtrVector& innerParticles );

    void setInnerContactArea( agx::Real contactArea );

    void setSettingsAreDirty(bool enable);

    bool getSettingsAreDirty() const;

    agx::Real computeDefaultBottomContactThreshold() const;

    AGXSTREAM_DECLARE_SERIALIZABLE( agxTerrain::Shovel );
    DOXYGEN_END_INTERNAL_BLOCK()

  protected:
    /**
    Default constructor used in serialization.
    */
    Shovel();

    /**
    Reference counted object - protected destructor.
    */
    virtual ~Shovel();

    agxCollide::GeometryRefVector createVoxelCollisionGeometries( agx::Real tessellationLevel ) const;

  protected:
    struct SoilPenetrationParameters
    {
      agx::Real depthThreshold;
      agx::Real forceScaling;
      agx::Real maxForce;
    };

    enum StateFlags : agx::UInt32
    {
      ENABLED = 1 << 0,
      REMOVE_SHOVEL_CONTACTS = 1 << 1,
      INNER_SHAPE_CREATE_DYNAMIC_MASS = 1 << 2,
      ENABLE_PARTICLE_FREE_DEFORMATION = 1 << 3,
      ENABLE_PARTICLE_FORCE_FEEDBACK = 1 << 4
    };
    using Flags = agx::BitState<StateFlags, agx::UInt32>;

    agx::RigidBodyObserver    m_shovelBody;
    agx::Line                 m_cuttingEdge;
    agx::Line                 m_topEdge;
    agx::Vec3                 m_cuttingDirection;
    agx::Real                 m_toothLength;
    agx::Real                 m_toothMinRadius;
    agx::Real                 m_toothMaxRadius;
    agx::UInt                 m_nTeeth;
    agx::Real                 m_noMergeExtensionDistance;
    agx::Real                 m_verticalSoilBladeMergeDistance;
    agx::Real                 m_minimumSubmergedContactLengthFraction;
    agx::Real                 m_secondarySeparationDeadLoadLimit;
    Flags                     m_flags;
    SoilPenetrationParameters m_soilPenetrationParameters;

    std::array< ExcavationSettings, 4 > m_excavationSettings;

    agx::Real                           m_contactThreshold;

    // Caching of dead load
    agx::Real                 m_deadLoadFraction;
    // Caching of inner contact area
    agx::Real                 m_innerContactArea;

    // Geometries used to find shovel-voxel collisions
    agxCollide::GeometryRefVector m_voxelCollisionGeometries;
    bool                          m_settingsAreDirty;
  };
}
