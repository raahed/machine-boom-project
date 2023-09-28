/*
Copyright 2007-2023. Algoryx Simulation AB.

All AGX source code, intellectual property, documentation, sample code,
tutorials, scene files and technical white papers, are copyrighted, proprietary
and confidential material of Algoryx Simulation AB. You may not download, read,
store, distribute, publish, copy or otherwise disseminate, use or expose this
material unless having a written signed agreement with Algoryx Simulation AB, or
having been advised so by Algoryx Simulation AB for a time limited evaluation,
or having purchased a valid commercial license from Algoryx Simulation AB.

Algoryx Simulation AB disclaims all responsibilities for loss or damage caused
from using this software, unless otherwise stated in written agreements with
Algoryx Simulation AB.
*/

#ifndef AGXTERRAIN_TRAJECTORYCALCULATOR_H
#define AGXTERRAIN_TRAJECTORYCALCULATOR_H

#include <queue>
#include <numeric>

#include <agxTerrain/export.h>

#include <agx/agx_vector_types.h>
#include <agxData/agxData.h>

#include <agxCollide/HeightField.h>
#include <agxCollide/Geometry.h>
#include <agxCollide/Line.h>
#include <agxCollide/Contacts.h>
#include <agxCollide/BasicPrimitiveTests.h>

namespace agxTerrain
{
  struct TrajectorySettings
  {
    agx::Real cutDepth;
    agx::Real cutAngle;
    agx::Real bladeSpeed;
    agx::Real goalHeight;
    agx::Real resolution;
    agx::Real timestep;
  };

  class AGXTERRAIN_EXPORT TrajectoryCalculator
  {
  public:
    TrajectoryCalculator();
    TrajectoryCalculator(agxCollide::HeightField* heightField, const TrajectorySettings& settings);

    void setTrajectorySettings(const TrajectorySettings& settings);
    void setHeightField(agxCollide::HeightField* heightField);
    void setCutDepth(agx::Real cutDepth);
    void setCutAngle(agx::Real cutAngle);
    void setBladeSpeed(agx::Real bladeSpeed);
    void setGoalHeight(agx::Real goalHeight);
    void setResolution(agx::Real resolution);
    void setTimestep(agx::Real timestep);

    TrajectorySettings getTrajectorySettings() const;
    agx::Real getCutDepth() const;
    agx::Real getCutAngle() const;
    agx::Real getBladeSpeed() const;
    agx::Real getGoalHeight() const;
    agx::Real getResolution() const;
    agx::Real getTimestep() const;

    agx::Bool hasHeightField() const;
    agx::Bool isValid() const;

    void createTrajectory(const agx::Vec2& startPoint, const agx::Vec2& endPoint);
    agx::Vec3Vector getTrajectory();
    agx::Vec3 getInitialPosition();
    agx::Vec3 getVelocity();
    void discardTrajectory();

    // Debug rendering methods
    void DEBUG_drawHeightCurve();
    void DEBUG_drawTrajectory();
    void DEBUG_drawVelocities();

  private:
    agx::RealVector applyFilters(const agx::RealVector& heightCurve);
    void computeHeightCurve(const agx::Vec2& startPoint, const agx::Vec2& endPoint);
    void computeTrajectory(const agx::RealVector& heightCurve);
    void computeVelocities();

    agx::RealVector movingAverageStddev(const agx::RealVector& data, const size_t range);
    agx::RealVector movingAverageFilter(const agx::RealVector& data, const size_t range);
    agx::RealVector movingAverageFilterLowSlope(const agx::RealVector& data, size_t range, const agx::Real threshold);
    agx::Real interpolate(agx::Real x, const agx::RealVector& xDiscrete, const agx::RealVector& yDiscrete);
    agx::RealVector computeArcLength(const agx::RealVector& x, const agx::RealVector& y);
    agx::Real computeInverseArcLength(agx::Real distance, const agx::RealVector& x, const agx::RealVector& arcLength);

    // Heightfield and resulting 2D heightcurve
    agxCollide::HeightFieldRef m_heightField;
    agx::RealVector m_heightCurve;
    agx::RealVector m_x;

    // Settings
    TrajectorySettings m_settings;

    // Start and endpoint in xy-plane along with rotation matrix for velocities
    agx::Vec2 m_startPoint;
    agx::Vec2 m_endPoint;
    agx::Matrix4x4 m_rotateMatrix;

    // Resulting trajectory and velocities
    agx::RealVector m_trajectory;
    std::queue<agx::Vec3> m_velocities;

    // Boolean for valid trajectory
    agx::Bool m_validTrajectory;
  };
}

#endif
