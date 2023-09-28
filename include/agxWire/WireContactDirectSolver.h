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


#include <agx/agx.h>
#include <agx/Referenced.h>
#include <agxWire/Node.h>
#include <agxWire/ShapeContactNode.h>

#include <agxWire/WireSolver/WireSolveData.h>

DOXYGEN_START_INTERNAL_BLOCK()

namespace agxWire
{
  class WireDistanceCompositeConstraint;

  typedef agx::Vector<ShapeContactNode*> ContactNodePtrVector;

  /**
  */
  class AGXPHYSICS_EXPORT WireContactDirectSolver : public agx::Referenced
  {
    public:
      WireContactDirectSolver();

      void solveSystem(agxWire::WireDistanceCompositeConstraint* wire, const agxWire::NodePtrVector& allNodes, const ContactNodePtrVector& contactNodes, agx::RealVector& newNodeVelocities,
                       agx::Real damping);

      void solveSystemKinematic(ContactNodePtrVector& contactNodes, agx::RealVector& newNodeVelocities, WireDistanceCompositeConstraint* ldcc, const agx::Real h);

    private:
      bool doKinematicMove(WireDistanceCompositeConstraint* ldcc, ShapeContactNode* cln, agx::Real maxLengthDifference, bool ignoreFrictionAndMaxmovement = false, agx::Vec3 newShapePosition = agx::Vec3(0, 0, 0));
      void moveRowByKinematics(WireDistanceCompositeConstraint* ldcc, agx::Vector< NodeIterator >& movableContactsOnOneMassless, const agx::Real extendedLength);
      ~WireContactDirectSolver();

      struct SolveData
      {
        agx::Vec3Vector particlePoints; // p[0, n].
        agx::Vec3Vector edgeNormals;    // u[1, n-1].
        agx::Vec3Vector surfaceNormals; // n[1, n-1].
        agx::RealValarrayPOD segmentLengths; // l[0, n-1].
        agx::Vec3Vector segmentNormals; // w[0, n-1].
        agx::RealValarrayPOD wireTensions; // lambda[0, n].
        agx::RealValarrayPOD gravityForces;
        agx::Vec3Vector forces; // f[1, n-1].
        agx::RealValarrayPOD masses;
        agx::RealValarrayPOD forcesAlongEdgeNormals; // f_u[1, n-1].
        agx::RealValarrayPOD normalForces; // f_n[1, n-1].
        agx::RealValarrayPOD normalForceFromGravity;
        agx::RealValarrayPOD frictionBounds; // b[1, n-1].
      };

      SolveData m_solveData;
      std::shared_ptr<agxWire::WireSolver::WireSimulation> m_solver;
  };

}

DOXYGEN_END_INTERNAL_BLOCK()
