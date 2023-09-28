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

#include <agxModel/export.h>

#include <agx/Real.h>
#include <agx/Frame.h>
#include <agx/Vec2.h>
#include <agx/Constraint.h>
#include <agx/Referenced.h>
#include <agxUtil/agxUtil.h>
#include <agxUtil/TrimeshDeformer.h>
#include <agxStream/Serializable.h>
#include <agxSDK/Assembly.h>
#include <agxCollide/Line.h>
#include <agxCollide/Sphere.h>
#include <agxCollide/Mesh.h>
#include <agx/LockJoint.h>

namespace agxSDK
{
  class Simulation;
}


namespace agxModel
{


  AGX_DECLARE_POINTER_TYPES(SuctionCupSensorFilter);
  /**
   Class for filtering objects of interest nearby a suction cup.

   It will filter all contact involving geometries in
   either the m_sensors or the m_collisionGeometries vector.
   */
  class SuctionCupSensorFilter : public agxSDK::ExecuteFilter
  {
  public:

    SuctionCupSensorFilter()
    {}

    void addGeometry(agxCollide::Geometry* geometry)
    {
      m_collisionGeometries.push_back(geometry);
    }

    const agxCollide::GeometryRefVector& getGeometries() const
    {
      return m_collisionGeometries;
    }

    /// Inherited from agxSDK::ExecuteFilter.
    virtual bool match(const agxCollide::Geometry* g0, const agxCollide::Geometry* g1) const override
    {
      const agxCollide::Geometry* geos[] = { g0, g1 };

      for (size_t j = 0; j < 2; ++j)
      {
        for (size_t i = 0; i < m_collisionGeometries.size(); ++i)
        {
          if (geos[j] == m_collisionGeometries[i])
            return true;
        }
      }
      return false;
    }

    using ExecuteFilter::match;

  protected:
    virtual ~SuctionCupSensorFilter()
    {
      m_collisionGeometries.clear();
    };


  protected:
    agxCollide::GeometryRefVector m_sensors;
    agxCollide::GeometryRefVector m_collisionGeometries;
  };



  AGX_DECLARE_POINTER_TYPES(SuctionCup);
  AGX_DECLARE_VECTOR_TYPES(SuctionCup);

  AGX_DECLARE_POINTER_TYPES(VacuumSystem);
  AGX_DECLARE_VECTOR_TYPES(VacuumSystem);

  /**
  The SuctionCupInteraction is a book keeper of one interaction
  found by the collision detection, given the geometries of the SuctionCupSensorFilter.

  The data is found from contact events and only valid during the pre step event.

  It will collect geometry contacts with non sensor geometries to
  be able to sum the total friction and normal forces
  together with the maximum relative velocity for the contact points.
  */
  AGX_DECLARE_POINTER_TYPES(SuctionCupInteraction);
  AGX_DECLARE_VECTOR_TYPES(SuctionCupInteraction);
  class AGXMODEL_EXPORT SuctionCupInteraction : public agx::Referenced
  {
    public:
      /**
      A class for collecting and computing information
      for one RigidBody nearby a suction cup.
      */
      SuctionCupInteraction();


      /**
      \returns the number of seal sensors any of the Geometries of the RigidBody is colliding with.
      */
      size_t getSealCount() const;

      /**
      \returns if the rigid body of the interaction is centered under the suction cup.
      */
      bool getCentered() const;

      /**
      \returns distance to object along a vector considered positioned at the
      center of the suction cup, normal to the suction cup plane.
      Note that the distance is only valid if also getCentered() returns true.
      */
      agx::Real getCenterDistance() const;

      /**
      \returns the surface normal of the object at the point at the
      center under the suction cup.
      Note that the normal is only valid if also getCentered() returns true.
      */
      agx::Vec3 getCenterNormalWorld() const;

      /**
      \returns the scalar force calculated for the rigid body
      during the call to updateVacuumForce() which is called from
      SuctionGripper::pre.
      */
      agx::Real getCalculatedScalarForce() const;

      /**
      \returns the 3D force in world frame calculated for the rigid body
      during the call to updateVacuumForce() which is called from
      SuctionGripper::pre.
      */
      agx::Vec3 getCalculatedForce() const;

      /**
      \returns the world position for applying the vacuum force on the rigid body
      during the call to updateVacuumForce() which is called from
      SuctionGripper::pre.
      */
      agx::Vec3 getFoundSurfacePositionWorld() const;

      /**
      \returns the rigid body which will be affected by the vacuum force.
      */
      agx::RigidBody* getRigidBody();

      /**
      \returns the accumulated friction forces (from solver) between the suction cup and the body which is in interaction with.
      */
      agx::Vec3 getCurrentFrictionForce() const;

      /**
      \returns the accumulated normal forces (from solver) between the suction cup and the body which is in interaction with.
      */
      agx::Vec3 getCurrentContactNormalForce() const;

      /**
      \returns the maximal found velocity between the suction cup and the body which is in interaction with.
      */
      agx::Real getCurrentRelativeVelocity() const;


    protected:
      virtual ~SuctionCupInteraction();

      friend class SuctionCup;
      friend class SuctionGripper;

      /*
      Update the latest position found for where to apply the cvacuum force
      */
      void setFoundSurfacePositionWorld(agx::Vec3 worldPosition);

      /**
      \param - body, the rigid body which will be affected by the calculated vacuum force.
      */
      void setRigidBody(agx::RigidBody* body);

      /**
      Updates the surface position of an interacting object where to apply the force.
      Will always be along the suction cup normal,
      but only updated with the depth is shorter than any previous registrated depth.
      */
      void updateEstimatedForcePosition(const SuctionCup* cup, agx::Real depth);

      /**
      Set distance to object along a vector considered positioned at the
      center of the suction cup, normal to the suction cup plane.
      */
      void setCenterDistance(agx::Real distance);

      /**
      Set the surface normal of the object at the point at the
      center under the suction cup.
      */
      void setCenterNormalWorld(agx::Vec3 normal);

      void updateInteractedSeals(size_t sealIndex, agx::Real depth);

      void setCalculatedForce(agx::Vec3 force);
      void setCalculatedScalarForce(agx::Real scalarForce);

      agx::Real getSealDepth(size_t sealIndex) const;

      bool hasSealInteraction(size_t sealIndex) const;

      /**
      Function responsible for updating the force from the vacuum.
      By calling setCalculatedScalarForce and setCalculatedForce
      \param cup - the suction cup interacting with objects
      \param vacuumSystem - the vacuum system for the suction gripper where this suction cup is attached
      \param fraction - the fraction of the total force that this interaction is allowed to use, given seal sensor interactions
      */
      virtual bool updateVacuumForce(const SuctionCup* cup, const VacuumSystem* vacuumSystem, agx::Real fraction);

      void reset();

      void addTemporaryGeometryContact(agxCollide::GeometryContact* gc);

      void renderInteraction(const SuctionCup* cup);

      bool m_centered;
      agx::Real m_centerDistance;
      agx::Vec3 m_centerNormalWorld;
      agx::Vec3 m_foundSurfacePosition;
      agx::Real m_calculatedScalarForce;
      agx::Vec3 m_calculatedForce;
      agx::Vec3 m_calculatedForcePosition;
      agx::RigidBodyObserver m_body;
      agx::HashVector<size_t, agx::Real> m_interactedSealDepth;
      agxCollide::GeometryContactPtrVector m_tempContacts;
  };


  AGX_DECLARE_POINTER_TYPES(SuctionGripper);
  AGX_DECLARE_VECTOR_TYPES(SuctionGripper);

  /**
  Basic model for a vacuum system used by a SuctionGripper.
  virtual function `calculateAvailableVacuumForce` need to compute the
  current force a suction cup will add to the rigid bodies
  near the cup.

  Unless anything else is said, all vacuum system are considered
  as one shapeless volume, with homogenius internal pressure.
  */
  class AGXMODEL_EXPORT VacuumSystem : public virtual agx::Referenced, public virtual agxStream::Serializable
  {
    public:
      /**
      Create a vacuum system with a constant desired vacuum level.
      The level must be between 0-1, where 0 is atmospheric pressure
      and 1 is perfect vacuum.
      \param desiredVacuum - the vacuum level (0-1)

      This vacuum system assume all interactions will get the full vacuum force possible.
      */
      VacuumSystem(agx::Real desiredVacuum);

      /**
      \returns the maximum force a specific cup can apply to interacted objects.
      \param cup - a suction cup.
      */
      agx::Real calculateMaximumSuctionForce(SuctionCup* cup) const;

      /**
      A virtual function which is responsible to calculate the actual force
      a suction cup can apply to objects, given the current situation.
      */
      virtual agx::Real calculateAvailableVacuumForce(SuctionCup* cup) const;

      /**
      A virtual function for updating the vacuum system.
      The default model will not be affected,
      but this function invites to have a time dependent vacuum system.
      */
      virtual void step(agx::Real dt, const SuctionGripper* gripper);

      /**
      \param enable - if true, the pump will become activated, otherwise disabled.
      */
      void setEnablePump(const bool& enable);

      /**
      \returns true if the pump is active.
      */
      bool getEnablePump() const;

      /**
      \returns the current vacuum level (0-1).
      */
      agx::Real getCurrentVacuum() const;

      /**
      Set the current vacuum level (0-1) for the system
      If set outside the valid range, the value is clamped to be inside.
      \param vacuumLevel - the current vacuum level
      */
      void setDesiredVacuum(agx::Real vacuumLevel);

      /**
      Set the desired vacuum level (0-1) for the system
      */
      agx::Real getDesiredVacuum() const;

      /**
      Set the pressure outside the vacuum system in Pascal [Pa]
      */
      void setOutsidePressure(agx::Real pressure);

      /**
      Get the pressure outside the vacuum system in Pascal[Pa]
      */
      agx::Real getOutsidePressure() const;

      /**
      \returns estimated gas flow through the suction cup lip
      */
      agx::Real calculateGasSpeedAtLip(const SuctionCup* cup) const;

      /**
      Set the gas density outside the vacuum system [kg / m^3]
      */
      void setOutsideGasDensity(agx::Real density);

      /**
      Get the gas density outside the vacuum system [kg / m^3]
      */
      agx::Real getOutsideGasDensity() const;

      AGXSTREAM_DECLARE_SERIALIZABLE(agxModel::VacuumSystem);

    protected:
      virtual ~VacuumSystem();
      VacuumSystem();
      /**
      Set the current vacuum for the system
      \param vacuum - the current vacuum level
      */
      void setCurrentVacuum(agx::Real vacuum);
      agx::Real m_desiredVacuum;
      agx::Real m_currentVacuum;
      bool m_pumpEnabled;
      agx::Real m_timeOfPumpStateToggled;
      agx::Real m_currentTime;
      agx::Real m_outsidePressure;
      agx::Real m_outsideGasDensity;
  };


  /**
  The SuctionCup model owns a body which is mean to be attached somewhere on a
  SuctionGripper holder.

  The body of the suction cup can have several geometries and sensors, but all geometries which
  are added to the list of sensors of the SuctionCupSensorFilter are used for discover
  nearby objects and collect surface data for the a vacuum force model.
  The handleContact implementation is responsible for creating and populating
  SuctionCupInteraction models, which are responsible for applying forces on objects.

  The default suction cup model will create two types of collision sensors,
  one line sensor, and a number of sphere sensors calles seal sensors.
  The seal sensors will be distributed around the center of the suction cup in the locka XY plane.

  The line will point define the suction cup plane.
  The local line vector is defines the line shape.
  The seal sensors are there do find the distance to the object near the lip.
  A leakage area can thereby be calculated.
  It is up to the vacuum force model to make sense of the collision data from the sensors.

  If not colliding with the line, the object is not centered below the suctioncup.

  The SuctionCupInteraction contain the contact point and normal of where the line collides.

          | |
          | | <- Holder
      ____|_|____
     |           | <- Cup
      `o-o-o-o-o` <- Spheres are seal sensors.
           |
           | <- Line
           |

  */
  typedef agx::HashSet< SuctionCupRef > SuctionCupRefSet;
  class AGXMODEL_EXPORT SuctionCup : public virtual agx::Referenced, public virtual agxStream::Serializable
  {
  public:
    /**
    The SuctionCup contains a rigid body with a local attachment frame
    which will be used to position it relative to a SuctionGripper assembly.
    \param cupBody - the rigid body of the suction cup, if set to nullptr, a body will be created with cylinder geometry
    \param lipRadius - the radius of the suction cup outermost part when in contact with objects
    \param lipHeight - the height of the collision geometry of the cup. Note, only used when cupBody argument is null.
    \param sealResolution - the discretization of the contact surface needed to be sealed for creating a vacuum.
    \param localLineSensorVector - the line sensor vector out from the suction cup ring plane, toward objects to gripp.
    \param sealSensorReach - size of the seal sensor radiusr

    */
    SuctionCup(agx::RigidBody* cupBody,
      agx::Real lipRadius = agx::Real(0.025),
      agx::Real lipHeight = agx::Real(0.01),
      size_t sealResolution = size_t(0),
      agx::Vec3 localLineSensorVector = agx::Vec3(0,0,-0.1),
      agx::Real sealSensorReach = agx::Real(0.01));


    /**
    \returns normalized world vector which is normal to the suction cup surface.
    */
    agx::Vec3 getNormal() const;

    /**
    \returns area of suction cup, calculated from the lip radius.
    which is used for vacuum force computation. F = P * A
    */
    agx::Real getArea() const;

    /**
    \returns pointer to the suction cup RigidBody
    */
    const agx::RigidBody* getRigidBody() const;

    /**
    \returns pointer to the suction cup RigidBody
    */
    agx::RigidBody* getRigidBody();

    /**
    \returns the lip radius of the suction cup.
    */
    agx::Real getLipRadius() const;

    /**
    Set the mounting radius.
    Will default to the half of the lip radius
    Could effect the airflow for more advanced
    VacuumSystem models.
    */
    void setMountingRadius(agx::Real radius);

    /**
    \returns the mouinting radius.
    */
    agx::Real getMountingRadius() const;

    /**
    * Verify if the cup is valid
    */
    bool isValid() const;

    /**
    \returns the from seal sensors calculated leakage area.
    */
    agx::Real getCurrentLeakageArea() const;

    /**
    \return the seal resolution of the suction cup.
    */
    size_t getSealResolution() const;

    /**
    \returns the distance between two seal sensors.
    */
    agx::Real getSensorDistance() const;

    /**
    \returns pointer to the collision filer implementation.
    */
    SuctionCupSensorFilter* getSensorFilter();

    /**
    The available vacuum force is a pre calculated value by the
    virtual VacuumSystem::calculateAvailableVacuumForce function.
    This function is just a getter for the latest found value,
    which is updated in the SuctionGripper::pre callback.
    \returns - calculated maximum vacuum force an object can be affected by when near this suction cup.
    */
    agx::Real getAvailableVacuumForce() const;

    /**
    \returns the seal sensors for the cup.
    */
    const agxCollide::GeometryRefSetVector& getSealSensors() const;

    /**
    \returns a temporary vector with the current active interactions
    only valid after preCollide callback and during the rest of the timestep.
    */
    agx::Vector < agxModel::SuctionCupInteraction*> getActiveInteractionsVector() const;

    /**
    calculates the leakage area. The last result can be accessed through getCurrentLeakageArea().
    \returns the leakage area between the suction cup and its interactions.
    */
    virtual void updateLeakageArea();

    /**
    Responsible for applying forces to rigid bodies of the cup interactions.
    */
    void applyVacuumForces(const VacuumSystem* vacuumSystem);


    /**
    Will call the virtual function
    VacuumSystem::calculateAvailableVacuumForce;
    then
    apply the vacuum forces
    */
    void updateVacuumForces(const VacuumSystem* vacuumSystem);


    AGXSTREAM_DECLARE_SERIALIZABLE(agxModel::SuctionCup);

  protected:
    SuctionCup();

    virtual ~SuctionCup();

    // Contact events.
    friend class agxUtil::GeneralContactEventListener< SuctionCup >;

    friend class SuctionGripper;

    void invalidateInteractions();

    /**
    \returns a pre-cooked body with collision cylinders according to the cup dimensions (lipRAdius, height)
    */
    agx::RigidBody* createDefaultCupBody();

    const agx::HashVector<agx::RigidBody*, SuctionCupInteractionRef>& getActiveInteractions() const;

    const SuctionCupInteractionRefVector& getDisabledInteractions() const;

    /**
    Function creating an interaction.
    Virtual - enabling other cup implementations
    i.e. the virtual SuctionCupInteraction::updateVacuumForce function
    may include new approaches of the actual force applied to the bodies.
    */
    virtual agxModel::SuctionCupInteraction* createInteraction();

    /**
    Makes sure all suction cup bodies and constraints are
    given to the simulation.
    Called from SuctionGripper::addNotification()
    */
    void addNotification(agxSDK::Simulation* simulation);
    void removeNotification(agxSDK::Simulation* simulation);


    friend class SuctionGripper;

    agxSDK::ContactEventListener::KeepContactPolicy impact(
      const agx::TimeStamp&, agxCollide::GeometryContact* gc);

    agxSDK::ContactEventListener::KeepContactPolicy contact(
      const agx::TimeStamp&, agxCollide::GeometryContact* gc);

    void separation(
      const agx::TimeStamp&, agxCollide::GeometryPair&);

    agxSDK::ContactEventListener::KeepContactPolicy handleContacts(
      const agx::TimeStamp&, agxCollide::GeometryContact* gc);

    agx::Vec3 calculateLineP2() const;
    void createLineSensor();
    agxCollide::Geometry* createSealSensor(agx::RigidBody* body, agx::Vec3 bodyLocalPosition);
    void createCupSealSensors();

    agx::Vec3 findSurfaceNormal(agxCollide::Shape* collidedShape,
                                agx::Vec3 pointAtSurface,
                                bool& success);

    SuctionCupInteraction* getActiveInteraction(agx::RigidBody* rigidBody);

    void collectCollisionGeometries();

  protected:
    agx::RigidBodyRef m_body;
    agx::Real m_lipRadius;
    agx::Real m_lipHeight;
    agx::Real m_mountingRadius;
    agx::Real m_currentLeakageArea;
    size_t m_sealResolution;
    agx::Vec3 m_localLineSensorVector;
    agx::Real m_sealSensorReach;
    agx::ref_ptr<agxCollide::Line> m_line;
    agx::AffineMatrix4x4 m_miniSphereTransform;
    agxCollide::SphereRef m_miniSphere;
    agxCollide::GeometryRef m_lineSensor;
    SuctionCupSensorFilterRef m_collisionFilter;
    agx::Real m_calculatedAvailableVacuumForce;
    agxCollide::GeometryRefSetVector m_sealSensors;
    agx::RealVector m_sealDepthMeasurements;
    agx::ref_ptr< agxUtil::GeneralContactEventListener< SuctionCup > > m_contactCallback;
    // Implementation dependent temporary variables, only valid per time step.
    agx::HashVector<agx::RigidBody*, SuctionCupInteractionRef> m_activeInteractions;
    SuctionCupInteractionRefVector m_interactionsPool;

  };

  /**
  This should be considered a template class used as a
  base for Suction Gripper simulation models.

  A Suction Gripper model has one holder body and at least
  one suction cup body.
  It is up to the user to position and attach
  each suction cup to the hold relative.
  The SuctionGripper has a VacuumSystem for
  vacuum force calculations.
  How this force is distributed and applied on
  the objects the suction cup interactis with
  is the responsability of the SuctionCupInteraction model
  of each individual suction cup.
  */

  class AGXMODEL_EXPORT SuctionGripper : public agxSDK::Assembly
  {
    public:
      /**
      The SuctionGripper must be initialized with a holder body.

      It is created given a default VaccumSystem, which is possible to replace.
      \param holderBody - the rigid body to which the cups are intended to be attached
      */
      SuctionGripper( agx::RigidBody* holderBody );

      /**
      Set the rigid body of the holder.
      */
      bool setHolderRigidBody(agx::RigidBody* holderBody);

      /**
      \return the rigid body of the holder.
      */
      agx::RigidBody* getHolderRigidBody();

      /**
      \return the rigid body of the holder.
      */
      const agx::RigidBody* getHolderRigidBody() const;

      /**
      \return a vector of all suction cups of the gripper.
      */
      const SuctionCupRefVector& getSuctionCups() const;

      /**
      \return a vector of all suction cups of the gripper.
      */
      SuctionCupRefVector& getSuctionCups();

      /**
      Add one suction cup to the gripper.
      The user is responsible for positioning and constraining
      the cup rigid body to the holder body.
      */
      void addSuctionCup(SuctionCup* cup);

      /// \return Is the SuctionGripper valid?
      bool isValid() const;

      /**
      \returns the summed leakage for all suction cups.
      */
      agx::Real getCurrentLeakage() const;

      /**
      \returns pointer to the vacuum system.
      */
      const VacuumSystem* getVacuumSystem() const;

      /**
      \returns pointer to the vacuum system.
      */
      VacuumSystem* getVacuumSystem();

      /**
      Set a new vacuum system to the gripper.
      */
      void setVacuumSystem(VacuumSystem* vacuumSystem);

      static agx::LockJoint* positionAndLockSuctionCup( agxModel::SuctionGripper* gripper,
                                                        agxModel::SuctionCup* cup,
                                                        agx::Vec3 holderLocalTranslate,
                                                        agx::Vec3 cupLocalTranslate);

      AGXSTREAM_DECLARE_SERIALIZABLE(SuctionGripper);

    protected:
      SuctionGripper();

      virtual void updateValid();

      virtual ~SuctionGripper();

      // Adding to/removing from simulation.
      virtual void addNotification( agxSDK::Simulation* simulation ) override;

      using agxSDK::Assembly::addNotification;

      virtual void removeNotification( agxSDK::Simulation* simulation )override;

      using agxSDK::Assembly::removeNotification;


      // Step events.
      friend class agxUtil::GeneralStepListener< SuctionGripper >;

      virtual void preCollide(const agx::TimeStamp&);
      virtual void pre(const agx::TimeStamp&);
      virtual void post(const agx::TimeStamp&);
      virtual void last(const agx::TimeStamp&);



    protected:
      // More persistent variables, will be serialized.
      agx::RigidBodyRef m_holderBody;
      bool m_valid;
      VacuumSystemRef m_vacuumSystem;
      agx::ref_ptr< agxUtil::GeneralStepListener< SuctionGripper > > m_stepCallback;
      agxModel::SuctionCupRefVector m_suctionCups;

  };

  AGX_DECLARE_POINTER_TYPES(SingleCupSuctionGripper);
  AGX_DECLARE_VECTOR_TYPES(SingleCupSuctionGripper);
  class AGXMODEL_EXPORT SingleCupSuctionGripper : public SuctionGripper
  {
  public:
    /**
    Create a SuctionGripper with one holder body
    and one suction cup. Both with cylindrical collision geometry.
    The default holder body can be ignored by the holderBody argument.
    The default suction cup geometry can be ignored by the cupBody argument
    If no deformableMesh is given, a default one is created.
    The deformable mesh does not collide, an its solo purpose is to
    populate any visual resource with updated vertex positions.
    \param sealResolution - number of seal sensors for the suction cup
    \param gripperRadius  - radius for both cylinder geometries (holder and cup)
    \param gripperHeight  - height for the cylinder geometry of the holder
    \param cupHeight      - height for the suction cup in relaxed state
    \param lipHeight      - height for the cylinder geometry for the suction cup
    \param deformableMesh - will be transformed according to the cup / holder relative transform
    \param cupMaterial    - the material for the cup geometry, if cupBody is nullptr
    \param holderBody     - will override the creation of a default holder rigid body
    \param cupBody        - will override the default suction cup body

    */
    SingleCupSuctionGripper(size_t sealResolution,
      agx::Real gripperRadius = agx::Real(0.025),
      agx::Real gripperHeight = agx::Real(0.01),
      agx::Real cupHeight = agx::Real(0.05),
      agx::Real lipHeight = agx::Real(0.01),
      agxCollide::Trimesh* deformableMesh = nullptr,
      agx::Material* cupMaterial = nullptr,
      agx::RigidBody* holderBody = nullptr,
      agx::RigidBody* cupBody = nullptr);

    /**
    \return the created lock joint between the holder and cup.
    */
    agx::LockJoint* getLockJoint() const;

    /**
    \return the geometry created if the deformable mesh is not nullptr
    */
    agxCollide::Geometry* getTrimeshGeometry() const;

    /**
    \return the trimesh shape assigned by the deformableMesh attribute
    */
    agxCollide::Trimesh* getDeformableTrimesh() const;

    /**
    \return the trimesh deformer
    */
    agxUtil::TrimeshDeformer* getTrimeshDeformer() const;

    AGXSTREAM_DECLARE_SERIALIZABLE(SingleCupSuctionGripper);

  protected:
    agx::RigidBody* createDefaultHolderBody(agx::Real radius, agx::Real height) const;
    void setupDeformableTrimesh(SuctionCup* cup, agx::Frame* holderFrame);

    SingleCupSuctionGripper();
    virtual ~SingleCupSuctionGripper();

    agx::ref_ptr<agx::LockJoint> m_lock;
    agxCollide::TrimeshRef m_trimesh;
    agxCollide::GeometryRef m_trimeshGeometry;
    agxUtil::TrimeshDeformerRef m_trimeshDeformer;

  };
}

