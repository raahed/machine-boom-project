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

#ifndef AGXHYDRAULICS_HYDRAULIC_UTILS_H
#define AGXHYDRAULICS_HYDRAULIC_UTILS_H

#include <agxHydraulics/export.h>
#include <agx/Real.h>

namespace agxPowerLine
{
  class PowerLine;
}

namespace agxHydraulics
{
  /**
  At set of small utility function intended to help when building hydraulics
  systems.
  */
  namespace utils
  {
    // Pressure conversions.
    AGXHYDRAULICS_EXPORT agx::Real pascalToKPascal(agx::Real pascals);
    AGXHYDRAULICS_EXPORT agx::Real pascalToBar(agx::Real pascals);
    AGXHYDRAULICS_EXPORT agx::Real barToPascal(agx::Real bar);
    AGXHYDRAULICS_EXPORT agx::Real pascalToPsi(agx::Real pascals);
    AGXHYDRAULICS_EXPORT agx::Real psiToPascal(agx::Real psi);

    // Angular velocity conversions.
    AGXHYDRAULICS_EXPORT agx::Real revolutionsPerMinuteToRadiansPerSecond(agx::Real revolutionsPerMinute);
    AGXHYDRAULICS_EXPORT agx::Real radiansPerSecondToRevolutionsPerMinute(agx::Real radiansPerSecond);

    // Flow rate conversions. The US liquid gallon is used.
    AGXHYDRAULICS_EXPORT agx::Real gallonsPerMinuteToCubicMetersPerSecond(agx::Real gallonsPerMinute);
    AGXHYDRAULICS_EXPORT agx::Real cubicMetersPerSecondToGallonsPerMinute(agx::Real cubicMetersPerSecond);
    AGXHYDRAULICS_EXPORT agx::Real cubicMetersPerSecondToLitersPerMinute(agx::Real cubicMetersPerSecond);
    AGXHYDRAULICS_EXPORT agx::Real cubicMetersPerSecondToCubicInches(agx::Real cubicMetersPerSecond);

    // Displacement conversions.
    AGXHYDRAULICS_EXPORT agx::Real cubicInchesPerRevolutionToCubicMetersPerRadian(agx::Real cubicInchesPerRevolution);
    AGXHYDRAULICS_EXPORT agx::Real cubicMetersPerRadianToCubicInchesPerRevolution(agx::Real cubicMetersPerRadian);

    // Circular area conversions.
    AGXHYDRAULICS_EXPORT agx::Real radiusToArea(agx::Real radius);
    AGXHYDRAULICS_EXPORT agx::Real diameterToArea(agx::Real diameter);



    /**
    Set the fluid density of all FlowUnits in a PowerLine.

    Will only affect the FlowUnits currently in the PowerLine. Any FlowUnits
    added later will be unaffected.
    */
    AGXHYDRAULICS_EXPORT void setFluidDensity(agxPowerLine::PowerLine* powerLine, agx::Real density);

    /**
    Set the fluid viscosity of all FlowUnits in a PowerLine.

    Will only affect the FlowUnits currently in the PowerLine. Any FlowUnits
    added later will be unaffected.
    */
    AGXHYDRAULICS_EXPORT void setFluidViscosity(agxPowerLine::PowerLine* powerLine, agx::Real viscosity);

    /**
    Set the tank pressure of the global implicit tank. Any unconnected pipe end
    will be subjected to this pressure.
    */
    AGXHYDRAULICS_EXPORT void setTankPressure(agx::Real tankPressure);

    /**
    \return The pressure of the global implicit tank.
    */
    AGXHYDRAULICS_EXPORT agx::Real getTankPressure();
  }
}


// Include guard.
#endif
