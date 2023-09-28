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

#pragma once

#include <agx/agx_vector_types.h>
#include <agx/RigidBody.h>

#include <agxSDK/StepEventListener.h>

#include <agxUtil/agxUtil.h>
#include <agxUtil/Statistic.h>

#include <agx/ContactForceFilterParticipant.h>

namespace agx
{
  /**
  A ContactForceFilter provide resting contact forces as seen through either an
  exponential moving average filter or a median filter. Impact forces are scaled
  using the Hertz impact model and provided separately.
  */
  class AGXPHYSICS_EXPORT ContactForceFilter : public agxSDK::StepEventListener
  {
  public:
    /// This constructor will be removed. Used only while experimenting.
    ContactForceFilter();

    ContactForceFilter(agx::RigidBody& body1, agx::RigidBody& body2);
    ContactForceFilter(agx::RigidBody& body, agxCollide::Geometry& geometry);

    /// \return The exponential moving average of the resting force.
    agx::Real getRestingMedian() const;

    /// \return The median of the resting force of the last few time steps.
    agx::Real getRestingAverage() const;

    /// \return The impact force, using the Hertz model, of the most recent time step.
    agx::Real getImpact() const;

  public: // Methods called by AGX.
    virtual void preCollide(const agx::TimeStamp& time) override;
    virtual void post(const agx::TimeStamp& time) override;

  protected:
    ~ContactForceFilter() = default;

  private:
    /// Constructor the public constructors delegate to.
    ContactForceFilter(
      agx::detail::ContactForceFilterParticipant participant1,
      agx::detail::ContactForceFilterParticipant participant2);

  private:
    agx::detail::ContactForceFilterParticipant m_participant1;
    agx::detail::ContactForceFilterParticipant m_participant2;

    agx::Real m_impactForce;

    agxUtil::MedianStatistic m_median;
    agxUtil::ExponentialMovingAverageStatistic m_average;

    // Set to true if we had at least one contact during the last update.
    bool m_hadContact;
  };
}
