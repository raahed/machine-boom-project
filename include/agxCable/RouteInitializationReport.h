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

#include <agx/Real.h>
#include <agx/Integer.h>

#include <agxCable/export.h>
#include <agxCable/CableSegment.h>
#include <agxCable/Node.h>

namespace agxCable
{
  class AGXCABLE_EXPORT RouteInitializationReport
  {
    public:
      /**
      Create a report with values indication that no initialization has happened
      yet, or that the initialization failed.

      \p RouteInitializationReport::successful will return false for this
      initialization report.
      */
      RouteInitializationReport();

      /**
      Create a report with the given contents.
      */
      RouteInitializationReport(agx::Real targetError, agx::Real actualError, agx::Real resolution, size_t numNodes);

      /**
      \return true if actual error is less than or equal to the target error.
      */
      agx::Bool successful() const;

      /**
      \return the target error
      */
      agx::Real getTargetError() const;

      /**
      \return the maximum error after routing
      */
      agx::Real getActualError() const;

      /**
      \return The number of segments in the initialized route.
      */
      size_t getNumSegments() const;

      /**
      \return The resolution of the routed path.
      */
      agx::Real getResolution() const;

      /**
      \return The total length of the created segments. May be different from
              the total distances between the routing node positions due to
              divisibility.
      */
      agx::Real getLength() const;


    public:
      void store(agxStream::OutputArchive& out) const;
      void restore(agxStream::InputArchive& in);

    private:
      agx::Real m_targetError;
      agx::Real m_actualError;
      agx::Real m_resolution;
      size_t m_numSegments;
  };
}
