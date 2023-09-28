/*
Copyright 2007-2023. Algoryx Simulation AB.

All AGX source code, intellectual property, documentation, sample code,
tutorials, scene files and technical white papers, is copyrighted, proprietary
and confidential material of Algoryx Simulation AB. You may not download, read,
store, distribute, publish, copy or otherwise disseminate, use or expose this
material unless having a written signed agreement with Algoryx Simulation AB, or having been
advised so by Algoryx Simulation AB for a time limited evaluation, or having purchased a
valid commercial license from Algoryx Simulation AB.

Algoryx Simulation AB disclaims all responsibilities for loss or damage caused
from using this software, unless otherwise stated in written agreements with
Algoryx Simulation AB.
*/

#ifndef AGX_REALTIMECLOCK_H
#define AGX_REALTIMECLOCK_H

#include <agx/Clock.h>
#include <agx/Timer.h>


namespace agx
{
  AGX_DECLARE_POINTER_TYPES(RealTimeClock);
  class AGXCORE_EXPORT RealTimeClock : public Clock
  {
  public:
    static RealTimeClock *instance();

    void tick();

  protected:
    RealTimeClock(const Name& name, Real frequency = 720);
    virtual ~RealTimeClock();

  private:
    bool m_first;
    Real m_rest;
    Timer m_timer;
    Timer m_tickTimer;
    Real m_tickTime;
  };

}


#endif /* _AGX_REALTIMECLOCK_H_ */
