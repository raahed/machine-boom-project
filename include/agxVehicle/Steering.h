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
#include <agxVehicle/export.h>
#include <agxVehicle/SteeringParameters.h>
#include <agx/Constraint.h>

namespace agxVehicle
{
  AGX_DECLARE_POINTER_TYPES(Steering);
  AGX_DECLARE_POINTER_TYPES(Ackermann);
  AGX_DECLARE_POINTER_TYPES(BellCrank);
  AGX_DECLARE_POINTER_TYPES(RackPinion);
  AGX_DECLARE_POINTER_TYPES(Davis);

  class WheelJoint;

  /**
  A Steering mechanism is designed to align wheels simultaneously to minimize the
  slip, making them move along the tangents of concentric circles with radii
  determined by the distances between the wheels, and the distance between
  front and rear.  In addition to the mechanical linkage that connects the
  wheel, there has to be also a connection to the steering wheel.

  AGX provides Ackerman mechanism with direct steering input on one of the
  wheels, a Bell-crank or central-lever linkage which is used for vehicle with a
  steering column in the center, a rack and pinion mechanism which is the
  most commonly used in passenger cars, and the Davis mechanism which
  provides ideal steering but is never used in practice.
  */
  class AGXVEHICLE_EXPORT Steering : public agx::Constraint
  {
  public:

    /**
    Set the value of the steering angle.
    \param angle - desired angle of rotation of the steering wheel
    in radians.
    */
    void setSteeringAngle(agx::Real angle);

    /**
    \return - the value of the steering angle
    */
    agx::Real getSteeringAngle() const;

    /**
    \return - the maximum allowed steering angle
    */
    agx::Real getMaximumSteeringAngle(agx::UInt side) const;

    /**
    Warning: there is no setSteeringParameters() method, since that requires reinitialization.
    It is best to delete and create new instance.
    */
    const agxVehicle::SteeringParameters& getSteeringParameters() const;

    /**
    Setup debug rendering of the steering geometry.
    */
    void render(class agxRender::RenderManager* mgr, float scale) const override;

    /**
    \return - the number of degree of freedom
    */
    int getNumDOF() const override;

    AGXSTREAM_DECLARE_SERIALIZABLE(agxVehicle::Steering);

  protected:
    Steering();
    virtual ~Steering();
    class SteeringImplementation* m_impl;
  };



  class AGXVEHICLE_EXPORT BellCrank : public Steering
  {
  public:
    /**
    Construct the bell-crank steering geometry.
    \param leftWheel - WheelJoint on the left of the chassis when looking ahead.
    \param rightWheel - WheelJoint on the right of the chassis.
    \param params - defaults to optimized values.
    */
    BellCrank(WheelJoint* leftWheel, WheelJoint* rightWheel,
              SteeringParameters params = SteeringParameters::BellCrank());

    BellCrank() = default;
    AGXSTREAM_DECLARE_SERIALIZABLE(agxVehicle::BellCrank);

  protected:
    virtual ~BellCrank();
  };



  class AGXVEHICLE_EXPORT RackPinion: public Steering
  {
  public:
    /**
    Construct the Rack-pinion steering mechanism.
    \param leftWheel - WheelJoint on the left of the chassis when looking ahead.
    \param rightWheel - WheelJoint on the right of the chassis.
    \param params - defaults to optimized values.
    */
    RackPinion(WheelJoint* leftWheel, WheelJoint* rightWheel,
               SteeringParameters params = SteeringParameters::RackPinion());

    RackPinion() = default;
    AGXSTREAM_DECLARE_SERIALIZABLE(agxVehicle::RackPinion);

  protected:
    virtual ~RackPinion();
  };



  class AGXVEHICLE_EXPORT Davis : public Steering
  {
  public:
    /**
    Construct the Davis steering mechanism. The Davis steering mechanism provides perfect steering geometry.
    It is not use much in practice because it is subject to wear.
    \param leftWheel - WheelJoint on the left of the chassis when looking ahead.
    \param rightWheel - WheelJoint on the right of the chassis.
    \param params - defaults to optimized values.
    */
    Davis(WheelJoint* leftWheel, WheelJoint* rightWheel,
          SteeringParameters params = SteeringParameters::Davis());

    Davis() = default;
    AGXSTREAM_DECLARE_SERIALIZABLE(agxVehicle::Davis);

  protected:
    virtual ~Davis();
  };



  class AGXVEHICLE_EXPORT Ackermann : public Steering
  {
  public:
    /**
    Construct the Ackermann steering mechanism.
    Ackermann four bar linkage with steering applied directly one one wheel.
    \param leftWheel - WheelJoint on the left of the chassis when looking ahead.
    \param rightWheel - WheelJoint on the right of the chassis.
    \param params - defaults to optimized values.
    */
    Ackermann(WheelJoint* leftWheel, WheelJoint* rightWheel,
              SteeringParameters params = SteeringParameters::Ackermann());

    int getNumDOF() const override;

    Ackermann() = default;
    AGXSTREAM_DECLARE_SERIALIZABLE(agxVehicle::Ackermann);

  protected:
    virtual ~Ackermann();
  };
}
