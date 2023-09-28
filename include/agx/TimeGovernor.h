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

#ifndef AGX_TIME_GOVERNOR_H
#define AGX_TIME_GOVERNOR_H

#include <agx/agx.h>
#include <agx/Referenced.h>
#include <agx/TimeStamp.h>
#include <agx/Clock.h>

namespace agx
{
  /**

  */
  class AGXPHYSICS_EXPORT TimeGovernor : public agx::Referenced
  {
  public:
    TimeGovernor();
    TimeGovernor(agx::ClockRef clock);
    virtual ~TimeGovernor();

    /** Set the step length of the integrator
    \param h - The step length
    */
    virtual void setTimeStep(Real h);
    virtual Real getTimeStep() const;
    virtual void operator()( const agx::TimeStamp& ) {}
  protected:
    agx::ClockRef m_clock;
  };

  typedef agx::ref_ptr<TimeGovernor> TimeGovernorRef;
}

#endif
