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
#include <agx/Math.h>
#include <agxVehicle/export.h>

namespace agxVehicle
{
  struct AGXVEHICLE_EXPORT SteeringParameters
  {
    /**
    The steering parameters are generally complicated so sensible defaults are provided.
    They are as follows:
    \param phi0   - Initial angle of the kingpin or kunckle of the right wheel,
                    measured from the direction along the axle, and pointing to the right.
                    With phi0 = 0 the knuckel points directly to the right along the axle (a bad thing),
                    and with phi0 = pi/2 makes the kingpin parallel to the wheel (not good either)
    \param l      - Length of the knuckle normalized to the wheel base. Wheel base is the distance between the centers of the two wheels on the same axle.
    \param alpha0 - Initial angle of the right tie rod which connects the knuckle to the steering column or the rack.
                    This is also measured from the axle of the right wheel
    \param lc     - Distance of the steering column from the tie rods from the
                    line connecting the wheels in units of l, when applicable (Bell crank)
    \param lr     - Rack length in units of the wheel base for rack and pinion
    \param gear   - Gear ratio between steering wheel and the control angle of the linkage mechanism
    \param side   - Side of steering column position: 0 refers to left wheel, 1 refer to right wheel. This is specially used for Ackermann steering mechanism.
    */
    agx::Real phi0;
    agx::Real l;
    agx::Real alpha0;
    agx::Real lc;
    agx::Real lr;
    agx::Real gear;
    agx::UInt side;


    /**
    Bell Crank steering configuration parameters:
        phi0: -1.885
        l: 0.14
        alpha0: 0.0
        lc: 1.0
        lr: 0.0
        gear: 1.0
        side: 0
    */
    static SteeringParameters BellCrank();


    /**
    Ackermann steering configuration parameters:
        phi0: -2.007
        l: 0.16
        alpha0: 0.0
        lc: 0.0
        lr: 0.0
        gear: 1.0
        side: 0
    */
    static SteeringParameters Ackermann();


    /**
    RackPinion steering configuration parameters:
        phi0: -1.885
        l: 0.14
        alpha0: 0.0
        lc: 1.0
        lr: 0.25
        gear: 1.0
        side: 0
    */
    static SteeringParameters RackPinion();


    /**
    Davis steering configuration parameters:
        phi0: 1.815
        l: 0.14
        alpha0: 1.32645
        lc: -2.0
        lr: 0.75
        gear: 1.0
        side: 0
    */
    static SteeringParameters Davis();
  };
}
