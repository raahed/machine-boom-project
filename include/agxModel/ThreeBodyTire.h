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

#ifndef AGXMODEL_THREE_BODY_TIRE_H
#define AGXMODEL_THREE_BODY_TIRE_H


#include <agxModel/TwoBodyTire.h>

namespace agxModel
{

  AGX_DECLARE_POINTER_TYPES(ThreeBodyTire);
  /**
  \note: This is an experimental feature, not fully tested yet.

  A simple tire model with three rigid bodies - the hub and the tire represented by two bodies.
  One body with rotational degrees of freedom relative the hub and another with spatial degrees of freedom relative the first tire body.
  To model any deformation/elastic behavior, the joint settings between
  hub and tire are being modified.

  Notes on stiffness and damping:
  See TwoBodyTire
  */
  class AGXMODEL_EXPORT ThreeBodyTire : public agxModel::TwoBodyTire
  {
    public:
      /**
      Construct tire given rigid body (with wheel geometry/geometries) and radius.
      By definition tire axis is assumed to be y direction in body coordinates.
      This is also the secondary friction direction.
      Contacts between the tireBody and the hingeBody will be disabled.
      \param tireBodyCollision - rigid body of this tire with spatial degrees of freedom relative the tireBodyVisual (including geometries)
      \param tireBodyVisual - rigid body of this tire with rotational degree of freedom relative the hub (no geometries)
      \param outerRadius - radius of this tire
      \param hubBody - rigid body that the tire is fixed upon (including geometries)
      \param innerRadius - inner radius of the tire (and outer radius of hub)
      \param localTransform - local transform for hinge and friction directions,
             local in tireBody's coordinate system.
             Expects rotation axis to be in Y-axis.
      */
      ThreeBodyTire(agx::RigidBody* tireBodyCollision, agx::RigidBody* tireBodyVisual, agx::Real outerRadius,
        agx::RigidBody* hubBody, agx::Real innerRadius,
        const agx::AffineMatrix4x4& localTransform = agx::AffineMatrix4x4() );

      /**
      \return the rigid body of the tire.
      */
      agx::RigidBody* getTireVisualRigidBody();

      /**
      \return the rigid body of the tire.
      */
      const agx::RigidBody* getTireCollisionRigidBody() const;


      /**
      Gets the hinge (with active Lock1D) connecting the tire to the hub.
      This setup is an implementation detail, and might change in future versions of AGX.
      Having direct control over the hinge can be helpful in order to change
      fine-grained options such as compliance and damping.
      Use with care.
      \retval The hinge between tire and hub. 0 if isValid is false.
      */
      agx::Hinge* getOuterHinge();

      /**
      Sets the stiffness for a deformation mode.
      \param stiffness The stiffness.
      \param mode The deformation mode.
      \retval True if setting was successful.
      \note see comments in class definition about stiffness and damping.
      */
      virtual bool setStiffness(const agx::Real stiffness, DeformationMode mode) override;

      /**
      Gets the stiffness for a deformation mode.
      \param mode The deformation mode.
      \retval The stiffness. 0 if invalid mode.
      \note see comments in class definition about stiffness and damping.
      */
      virtual agx::Real getStiffness(DeformationMode mode) override;

      /**
      Sets the damping coefficient for a deformation mode.
      \param dampingCoefficient The damping Coefficient.
      \param mode The deformation mode.
      \retval True if setting was successful.
      \note see comments in class definition about stiffness and damping.
      */
      virtual bool setDampingCoefficient(const agx::Real dampingCoefficient, DeformationMode mode) override;

      /**
      Gets the damping coefficient for a deformation mode.
      \param mode The deformation mode.
      \retval The damping Coefficient. 0 if invalid mode.
      \note see comments in class definition about stiffness and damping.
      */
      virtual agx::Real getDampingCoefficient(DeformationMode mode) override;


      AGXSTREAM_DECLARE_SERIALIZABLE(agxModel::ThreeBodyTire);

    protected:
      ThreeBodyTire();

      virtual ~ThreeBodyTire();

      virtual bool isValid() const override;

      // Contact events.
      friend class agxUtil::GeneralContactEventListener< TwoBodyTire >;

      agxSDK::ContactEventListener::KeepContactPolicy impact(
        const agx::TimeStamp&, agxCollide::GeometryContact* gc );

      agxSDK::ContactEventListener::KeepContactPolicy contact(
        const agx::TimeStamp&, agxCollide::GeometryContact* gc );

      void separation(
        const agx::TimeStamp&, agxCollide::GeometryPair& );

      agxSDK::ContactEventListener::KeepContactPolicy handleContacts(
        const agx::TimeStamp&,
        agxCollide::GeometryContact* gc,
        bool impact);

      void reduceContacts(agxCollide::LocalGeometryContact& gc);

      // Step events.
      friend class agxUtil::GeneralStepListener< TwoBodyTire >;

      virtual void post( const agx::TimeStamp& ) override;


    protected:

      // Adding to/removing from simulation.
      virtual void addNotification(agxSDK::Simulation* simulation) override;

      using agxSDK::Assembly::addNotification;

      virtual void removeNotification(agxSDK::Simulation* simulation)override;

      using agxSDK::Assembly::removeNotification;

      // More persistent variables, will be serialized.
      agx::RigidBodyObserver m_tireVisualBody;
      agx::HingeRef m_outerHinge;

  };
}

#endif // AGXMODEL_THREE_BODY_TIRE_H
