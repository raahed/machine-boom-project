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
#include <agx/Vec2.h>
#include <agx/Vec3.h>

#define MAXIMUM_DISTANCE_BETWEEN_NODE_AND_GEOMETRY 0.05

namespace agxStream
{
  class OutputArchive;
  class InputArchive;
}

namespace agxWire
{
  /**
  Class to hold specific parameters for the wire - this makes the
  wires highly configurable.
  */
  class AGXPHYSICS_EXPORT WireParameterController
  {
    public:
      /**
      Structure with compliance parameters for the prismatic constraint used by winches.
      Two spatial parameters, three rotational, one for the motor, one for the range and
      the final one is for the lock.
      */
      struct WinchPrismaticCompliance
      {
        agx::Vec2 spatial;      /**< Compliance */
        agx::Vec2 spatial_d;    /**< Damping */
        agx::Vec3 rotational;   /**< Compliance */
        agx::Vec3 rotational_d; /**< Damping */
        agx::Real motor;        /**< Compliance */
        agx::Real motor_d;      /**< Damping */
        agx::Real range;        /**< Compliance */
        agx::Real range_d;      /**< Damping */
        agx::Real lock;         /**< Compliance */
        agx::Real lock_d;       /**< Damping */
      };

      WireParameterController();

      ~WireParameterController();

      /**
      Reset all parameters to default.
      */
      void resetToDefault();

      /**
      Set the minimum distance allowed between nodes. I.e., a lumped element node will NOT
      be created closer than this distance from routed nodes.
      Default: 0.05. Valid: Real > 0.
      \param minDistanceBetweenNodes - new minimum distance between nodes definition
      */
      inline void setMinimumDistanceBetweenNodes(agx::Real minDistanceBetweenNodes);

      /**
      \return minimum distance between nodes
      \sa setMinimumDistanceBetweenNodes
      */
      agx::Real getMinimumDistanceBetweenNodes() const;

      /**
      Set the distance between the lump node and the stop node for winches. This also
      defines the theoretic maximum speed, in fact, in practice the speed can be higher
      but the behavior can be strange. Default: 1 m, equivalent to winch speed of 60 m/s
      in 60 Hz.
      \param stopNodeReferenceDistance - distance between the lump node and stop node for winches
      */
      inline void setStopNodeReferenceDistance(agx::Real stopNodeReferenceDistance);

      /**
      \return the distance between the lump node and stop node for winches
      \sa setStopNodeReferenceDistance
      */
      agx::Real getStopNodeReferenceDistance() const;

      /**
      Set the fraction of stop node reference distance that defines the range end of the prismatic
      constraint used by most winches. Default: 0.05 which means 5 cm if stop node reference distance is 1 m.
      \param stopNodeLumpMinDistanceFraction - fraction of the stop node reference distance to define the length of the prismatic range
      */
      void setStopNodeLumpMinDistanceFraction(agx::Real stopNodeLumpMinDistanceFraction);

      /**
      \return the fraction of the stop node reference distance to define the length of the prismatic range
      \sa setStopNodeLumpMinDistanceFraction
      */
      agx::Real getStopNodeLumpMinDistanceFraction() const;

      /**
      The scale constant controls the insert/remove of lumped nodes in a wire. The parameter has an analytical
      value derived given the Nyquist frequency. The probability to have more lumped nodes in the wire increases
      with this scale constant. Default: 0.35. Valid: Real > 0.
      \param scaleConstant - new scale constant (Default: 0.35)
      */
      void setScaleConstant(agx::Real scaleConstant);

      /**
      \return the scale constant
      \sa setScaleConstant
      */
      agx::Real getScaleConstant() const;

      /**
      Set the wire sphere geometry radius multiplier. It's convenient to have larger geometry spheres for the
      lumped elements in a wire. It looks more natural and it helps the contact handling in the wires. Default: 1.5.
      Valid: Real > 0.
      \param radiusMultiplier - wire lump geometry sphere radius multiplier
      */
      void setRadiusMultiplier(agx::Real radiusMultiplier);

      agx::Real getNonScaledRadiusMultiplier() const;

      /**
      \return the wire lump geometry sphere radius multiplier
      */
      agx::Real getRadiusMultiplier(agx::Real wireRadius) const;

      /**
      This value should be related the size of objects the wire is interacting with, to avoid tunneling
      If we are simulating

      */
      void setMaximumContactMovementOneTimestep(agx::Real maxMovement);

      agx::Real getMaximumContactMovementOneTimestep() const;

      /**
      \return the sensor radius, It is the same value as the max movement per timestep, to catch all geometries that are interesting
      (if wire diameter is larger than maximum movement we return wire diameter instead)
      */
      inline agx::Real getSensorRadius(agx::Real wireRadius) const;

      /**
      Set a multiplier for tension scaling when deciding if a constraint could be replaced with a force.
      */
      void setSplitTensionMultiplier(agx::Real multiplier);

      /**
      \return the split tension multiplier.
      */
      agx::Real getSplitTensionMultiplier() const;

      /**
      Set a scale for the damping of the dynamics solver for ShapeContactNodes
      This will scale the damping of the damping found in
      the material parameters for an interaction of a wire/other contact.
      */
      void setWireContactDynamicsSolverDampingScale(agx::Real scale);

      /**
      \return the  WireContactDynamicSolverDampingScale. Default == 2
      */
      agx::Real getWireContactDynamicsSolverDampingScale() const;

      DOXYGEN_START_INTERNAL_BLOCK()

      agx::Real getWireContactPrecision() const;
      agx::Real getContactNodeEdgeDefaultLength() const;
      agx::Bool getSolverUseExtraLength() const;
      agx::Bool getSolverMoveNodesBeforeMainSolver() const;

      /**
      param precision - 0 < precision <= 1. where 0 (zero) would mean no allowed overlap, and 1 means wireRadius in overlap is acceptable.
      */
      void setWireContactPrecision(agx::Real precision);

      void setContactNodeEdgeDefaultLength(agx::Real length);
      void setSolverUseExtraLength(agx::Bool useExtraLength);
      void setSolverMoveNodesBeforeMainSolver(agx::Bool moveNodesBeforeSolver);

      /**
      Get/set compliance parameters for the winch prismatic constraint.
      */
      WinchPrismaticCompliance& getWinchPrismaticCompliance();

      /**
      Get compliance parameters for the winch prismatic constraint.
      */
      const WinchPrismaticCompliance& getWinchPrismaticCompliance() const;

      /**
      Save.
      */
      void store( agxStream::OutputArchive& out ) const;

      /**
      Restore.
      */
      void restore( agxStream::InputArchive& in );

      DOXYGEN_END_INTERNAL_BLOCK()


    private:
      agx::Real                 m_minimumDistanceBetweenNodes;
      agx::Real                 m_stopNodeReferenceDistance;
      agx::Real                 m_stopNodeLumpMinDistanceFraction;
      WinchPrismaticCompliance  m_winchPrismaticCompliance;
      agx::Real                 m_scaleConstant;
      agx::Real                 m_radiusMultiplier;
      agx::Real                 m_maximumContactMovementOneTimestep;
      agx::Real                 m_splitTensionMultiplier;
      agx::Real                 m_wireContactPrecision;
      agx::Real                 m_wireContactDynamicsSolverDampingScale;
      agx::Real                 m_contactNodeEdgeDefaultLength;
      agx::Bool                 m_useExtraLength;
      agx::Bool                 m_moveNodesBeforeSolver;
      agx::Bool                 m_clampNodesToEdges;
  };

  inline WireParameterController::WinchPrismaticCompliance& WireParameterController::getWinchPrismaticCompliance()
  {
    return m_winchPrismaticCompliance;
  }

  inline const WireParameterController::WinchPrismaticCompliance& WireParameterController::getWinchPrismaticCompliance() const
  {
    return m_winchPrismaticCompliance;
  }

  inline void WireParameterController::setMinimumDistanceBetweenNodes(agx::Real minDistanceBetweenNodes)
  {
    agxAssert(minDistanceBetweenNodes > 0);
    m_minimumDistanceBetweenNodes = minDistanceBetweenNodes;
  }


  inline agx::Real WireParameterController::getMinimumDistanceBetweenNodes() const
  {
    return m_minimumDistanceBetweenNodes;
  }


  inline void WireParameterController::setStopNodeReferenceDistance(agx::Real stopNodeReferenceDistance)
  {
    m_stopNodeReferenceDistance = stopNodeReferenceDistance;
  }

  inline agx::Real WireParameterController::getStopNodeReferenceDistance() const
  {
    return m_stopNodeReferenceDistance;
  }


  inline void WireParameterController::setStopNodeLumpMinDistanceFraction(agx::Real stopNodeLumpMinDistanceFraction)
  {
    m_stopNodeLumpMinDistanceFraction = stopNodeLumpMinDistanceFraction;
  }


  inline agx::Real WireParameterController::getStopNodeLumpMinDistanceFraction() const
  {
    return m_stopNodeLumpMinDistanceFraction;
  }



  inline void WireParameterController::setScaleConstant(agx::Real scaleConstant)
  {
    agxAssert(scaleConstant > 0);
    m_scaleConstant = scaleConstant;
  }


  inline agx::Real WireParameterController::getScaleConstant() const
  {
    return m_scaleConstant;
  }


  inline void WireParameterController::setRadiusMultiplier(agx::Real radiusMultiplier)
  {
    agxAssert(radiusMultiplier > 0);
    m_radiusMultiplier = radiusMultiplier;
  }

  inline agx::Real WireParameterController::getNonScaledRadiusMultiplier() const
  {
    return m_radiusMultiplier;
  }

  inline agx::Real WireParameterController::getRadiusMultiplier(agx::Real wireRadius) const
  {
    agxAssert(!agx::equalsZero(wireRadius));

    agx::Real radiusInMultiplier = wireRadius * m_radiusMultiplier;

    agx::Real extra = agx::clamp(radiusInMultiplier - wireRadius, agx::Real(0), agx::Real(MAXIMUM_DISTANCE_BETWEEN_NODE_AND_GEOMETRY));//Max contact node distance from geometry == 0.05

    return (wireRadius + extra) / wireRadius;
  }

  inline void WireParameterController::setMaximumContactMovementOneTimestep(agx::Real maxMovement)
  {
    maxMovement = std::abs(maxMovement); //must be positive
    m_maximumContactMovementOneTimestep = maxMovement;
  }

  inline agx::Real WireParameterController::getMaximumContactMovementOneTimestep() const
  {
    return m_maximumContactMovementOneTimestep;
  }

  inline agx::Real WireParameterController::getSensorRadius(agx::Real wireRadius) const
  {
    return m_maximumContactMovementOneTimestep * 2 + wireRadius * 4;
  }
}
