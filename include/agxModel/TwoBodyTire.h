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

#ifndef AGXMODEL_TWO_BODY_TIRE_H
#define AGXMODEL_TWO_BODY_TIRE_H

#include <agxModel/export.h>

#include <agxModel/Tire.h>
#include <agx/Vec2.h>
#include <agx/Hinge.h>
#include <agxUtil/agxUtil.h>

namespace agxSDK
{
  class Simulation;
}


namespace agxModel
{
  AGX_DECLARE_POINTER_TYPES(TireFilter);
    // Class for allowing contacts as long as any of the bodies is involved.
  class TireFilter : public agxSDK::ExecuteFilter
  {
  public:

    TireFilter()
    {}

    void addBody(agx::RigidBody* body)
    {
      m_bodies.push_back(body);
    }

    /// Inherited from agxSDK::ExecuteFilter.
    virtual bool match(const agxCollide::Geometry* g0, const agxCollide::Geometry* g1) const override
    {
      const agxCollide::Geometry* geos[] = { g0, g1 };

      for (size_t j = 0; j < 2; ++j)
      if (geos[j]->getRigidBody())
      for (size_t i = 0; i < m_bodies.size(); ++i)
      if (geos[j]->getRigidBody() == m_bodies[i])
        return true;

      return false;
    }

    using ExecuteFilter::match;

  protected:
    virtual ~TireFilter() {};


  protected:
    agx::RigidBodyRefVector m_bodies;
  };


  AGX_DECLARE_POINTER_TYPES(TireFriction);
  /**
  Friction model that follows a tire's orientation in the world,
  based on the tire's frame.
  The Y-axis is assumed to be the rotation axis.
  For contacts orthogonal to it, the first (main) friction direction
  will be n^Y-axis, and the second one will be the Y-Axis.
  */
  class AGXMODEL_EXPORT TireFriction : public agx::IterativeProjectedConeFriction
  {
    public:
      /**
      Constructor given frame.
      \param referenceFrame The reference frame. For details, see comments on class.
      */
      TireFriction( const agx::Frame* referenceFrame );

      /// Inherited from IterativeProjectedConeFriction, the only change.
      virtual void calculateTangentPlane(
        const agxCollide::Geometry* geometry1,
        const agxCollide::Geometry* geometry2,
        const agx::Vec3& point,
        const agx::Vec3& normal,
        const agx::Real depth,
        agx::Vec3& ret_u,
        agx::Vec3& ret_v ) const override;

    protected:
      virtual ~TireFriction();

    protected:
      agx::FrameConstObserver m_refFrame;
  };


  AGX_DECLARE_POINTER_TYPES( TwoBodyTire );
  AGX_DECLARE_VECTOR_TYPES( TwoBodyTire );
  /**
  \note: This is an experimental feature, not fully tested yet.

  A simple tire model with two rigid bodies - the hub and the tire.
  To model any deformation/elastic behavior, the joint settings between
  hub and tire are being modified.

  Notes on stiffness and damping:

  There are four internal deformation modes for the tire which can be expressed via its
  constraint:
  Radial stiffness/damping affects translation orthogonal to tire rotation axis.
  Lateral stiffness/damping affects translation in axis of rotation.
  Bending stiffness/damping affects rotation orthogonal to axis of rotation.
  Torsional stiffness/damping affects rotation in axis of rotation.

  The unit for translational stiffness is force/displacement (if using SI: N/m)
  The unit for rotational stiffness is torque/angular displacement (if using SI: Nm/rad)
  The unit for the translational damping coefficient is force * time/displacement (if using SI: Ns/m)
  The unit for the rotational damping coefficient is torque * time/angular displacement (if using SI: Nms/rad)

  Implementation-details:
  1.) stiffness and damping coefficient are set directly on the constraint
  when using the setStiffness and setDampingCoefficient methods.
  Since the agx-internal damping is based on stiffness (or rather, compliance) as well,
  setStiffness will have an effect on the result of getDampingCoefficient.
  (The constraint can also directly be worked with via getHinge
  - if setting values there, getStiffness and getDampingCoefficient might not work
  as expected.)
  2.) The tire will copy a GeometryContacts' ContactMaterial and modify the copy to
  apply special tire friction. This is done both for implicit and explicit ContactMaterials.
  Both are queried for via MaterialManager (internal materials will be ignored).
  */
  class AGXMODEL_EXPORT TwoBodyTire : public agxModel::Tire
  {
    public:
      /**
      Construct tire given rigid body (with wheel geometry/geometries) and radius.
      By definition tire axis is assumed to be y direction in body coordinates.
      This is also the secondary friction direction.
      Contacts between the tireBody and the hingeBody will be disabled.
      \param tireBody - rigid body of this tire (including geometries)
      \param outerRadius - radius of this tire
      \param hubBody - rigid body that the tire is fixed upon (including geometries)
      \param innerRadius - inner radius of the tire (and outer radius of hub)
      \param localTransform - local transform for hinge and friction directions,
             local in tireBody's coordinate system.
             Expects rotation axis to be in Y-axis.
      */
      TwoBodyTire( agx::RigidBody* tireBody, agx::Real outerRadius,
        agx::RigidBody* hubBody, agx::Real innerRadius,
        const agx::AffineMatrix4x4& localTransform = agx::AffineMatrix4x4() );

      enum DeformationMode {
        RADIAL,      // Translation orthogonal to rotation axis.
        LATERAL,     // Translation in rotation axis.
        BENDING,     // Rotation orthogonal to rotation axis.
        TORSIONAL    // Rotation in rotation axis.
      };

      /**
      \return the rigid body of the tire.
      */
      agx::RigidBody* getTireRigidBody();

      /**
      \return the rigid body of the tire.
      */
      const agx::RigidBody* getTireRigidBody() const;

      /**
      \return the rigid body of the hub.
      */
      agx::RigidBody* getHubRigidBody();

      /**
      \return the rigid body of the hub.
      */
      const agx::RigidBody* getHubRigidBody() const;

      /**
      \return The outer radius of the tire. Identical to getRadius().
      */
      agx::Real getOuterRadius() const;

      /**
      \return The inner radius of the tire (and outer of the hub).
      */
      agx::Real getInnerRadius() const;

      /// \return Is the tire valid?
      virtual bool isValid() const;

      /// \return the reference frame for friction directions.
      const agx::Frame* getReferenceFrame() const;

      /**
      \return The friction multiplier (for friction different directions).
      See setImplicitFrictionMultiplier for more details.
      */
      agx::Vec2 getImplicitFrictionMultiplier() const;

      /**
      Set the implicit friction multiplier in order to get different behavior
      for different friction directions (forwards, sideways).
      This is only necessary for implicit contact materials,
      since for explicit ones, this can be set directly at the
      contact material instead.
      */
      void setImplicitFrictionMultiplier(const agx::Vec2& implicitFrictionMultiplier);

      /**
      Gets the current loaded radius. Recomputed in POST_STEP.
      Usually, loaded radius is defined per contact.
      Since the tire can interact with several contacts and be influenced by other forces,
      it is defined as the distance between the hub center and the tire center instead
      in the TwoBodyTire-model.
      */
      virtual agx::Real getLoadedRadius() const override;

      /**
      Sets the stiffness for a deformation mode.
      \param stiffness The stiffness.
      \param mode The deformation mode.
      \retval True if setting was successful.
      \note see comments in class definition about stiffness and damping.
      */
      virtual bool setStiffness(const agx::Real stiffness, DeformationMode mode);

      /**
      Gets the stiffness for a deformation mode.
      \param mode The deformation mode.
      \retval The stiffness. 0 if invalid mode.
      \note see comments in class definition about stiffness and damping.
      */
      virtual agx::Real getStiffness(DeformationMode mode);

      /**
      Sets the damping coefficient for a deformation mode.
      \param dampingCoefficient The damping Coefficient.
      \param mode The deformation mode.
      \retval True if setting was successful.
      \note see comments in class definition about stiffness and damping.
      */
      virtual bool setDampingCoefficient(const agx::Real dampingCoefficient, DeformationMode mode);

      /**
      Gets the damping coefficient for a deformation mode.
      \param mode The deformation mode.
      \retval The damping Coefficient. 0 if invalid mode.
      \note see comments in class definition about stiffness and damping.
      */
      virtual agx::Real getDampingCoefficient(DeformationMode mode);

      /**
      Search the given simulation for a TwoBodyTire with the given name.
      \param simulation - The simulation to search in.
      \param name - The name of the TwoBodyTire to search for.
      \return A TwoBodyTire with name \p name in simulation \p simulation, or nullptr.
      */
      static TwoBodyTire* find( agxSDK::Simulation* simulation, const agx::Name& name );

      /**
      Search the given assembly for a TwoBodyTire with the given name.
      \param assembly - The assembly to search in.
      \param name - The name of the TwoBodyTire to search for.
      \return A TwoBodyTire with name \p name in assembly \p assembly, or nullptr.
      */
      static TwoBodyTire* find( agxSDK::Assembly* assembly, const agx::Name& name );

      /**
      Find all TwoBodyTires with given name.
      \param simulation - simulation the tire is part of
      \param name - name of the TwoBodyTire
      \return vector of TwoBodyTire
      */
      static TwoBodyTirePtrVector findAll( const agxSDK::Simulation* simulation, const agx::Name& name );

      /**
      Finds all TwoBodyTire in the given simulation.
      \param simulation - simulation with TwoBodyTire.
      \return vector of TwoBodyTire
      */
      static TwoBodyTirePtrVector findAll( const agxSDK::Simulation* simulation );

      /**
      Gets the hinge (with active Lock1D) connecting the tire to the hub.
      This setup is an implementation detail, and might change in future versions of AGX.
      Having direct control over the hinge can be helpful in order to change
      fine-grained options such as compliance and damping.
      Use with care.
      \retval The hinge between tire and hub. 0 if isValid is false.
      */
      agx::Hinge* getHinge();

      AGXSTREAM_DECLARE_SERIALIZABLE(agxModel::TwoBodyTire);

    protected:
      TwoBodyTire();

      void initBodies(agx::Real outerRadius, agx::Real innerRadius);

      virtual ~TwoBodyTire();

      // Adding to/removing from simulation.
      virtual void addNotification( agxSDK::Simulation* simulation ) override;

      using agxSDK::Assembly::addNotification;

      virtual void removeNotification( agxSDK::Simulation* simulation )override;

      using agxSDK::Assembly::removeNotification;

      // Contact events.
      friend class agxUtil::GeneralContactEventListener< TwoBodyTire >;

      agxSDK::ContactEventListener::KeepContactPolicy impact(
        const agx::TimeStamp&, agxCollide::GeometryContact* gc );

      agxSDK::ContactEventListener::KeepContactPolicy contact(
        const agx::TimeStamp&, agxCollide::GeometryContact* gc );

      void separation(
        const agx::TimeStamp&, agxCollide::GeometryPair& );

      agxSDK::ContactEventListener::KeepContactPolicy handleContacts(
        const agx::TimeStamp&, agxCollide::GeometryContact* gc);

      agx::Vec3 getTireCenterWorld() const;

      agx::Vec3 getTireAxisWorld() const;

      void reduceContacts(agxCollide::LocalGeometryContact& gc);

      // Step events.
      friend class agxUtil::GeneralStepListener< TwoBodyTire >;

      virtual void preCollide( const agx::TimeStamp& );
      virtual void pre( const agx::TimeStamp& );
      virtual void post( const agx::TimeStamp& );
      virtual void last( const agx::TimeStamp& );


    protected:
      // More persistent variables, will be serialized.
      agx::RigidBodyObserver m_tireBody;
      agx::RigidBodyObserver m_hubBody;
      agx::Real m_innerRadius;
      agx::Vec2 m_implicitFrictionMultiplier;
      agx::HingeRef m_hinge;
      agx::FrameRef m_refFrame;
      bool m_initialized;
      agx::Real m_loadedRadius;
      // Implementation dependent temporary variables, only valid per simulation.
      agx::UInt32 m_collisionGroup;
      agx::ref_ptr< agxUtil::GeneralContactEventListener< TwoBodyTire > > m_contactCallback;
      agx::ref_ptr< agxUtil::GeneralStepListener< TwoBodyTire > > m_stepCallback;
      // Implementation dependent temporary variables, only valid per time step.
      agx::Vector<agxCollide::LocalGeometryContact> m_contacts;
      agx::Vector<agx::ContactMaterialRef> m_contactMaterials;
  };
}

#endif // AGXMODEL_TWO_BODY_TIRE_H
