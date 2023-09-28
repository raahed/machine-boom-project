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


#ifndef AGX_REALTIMETRIGGER_H
#define AGX_REALTIMETRIGGER_H

#include <agx/Timer.h>

namespace agx
{

  class AGXCORE_EXPORT RealTimeTrigger
  {
  public:
    RealTimeTrigger(Real frequency = 0);

    /// Set the trigger frequency
    void setFrequency(Real frequency);

    /// \return The trigger frequency
    Real getFrequency() const;

    /// Reset the trigger
    void reset();

    /**
    Update the trigger to the current time.
    \return True if we triggered an update
    */
    bool update();

  private:
    Timer m_timer;
    Real m_frequency;
    Real m_rest;
  };
}

#endif //AGX_REALTIMETRIGGER_H
