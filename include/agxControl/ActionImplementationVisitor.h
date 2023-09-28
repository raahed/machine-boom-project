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


/// \cond CONTROL


#ifndef AGXCONTROL_ACTION_IMPLEMENTATION_VISITOR_H
#define AGXCONTROL_ACTION_IMPLEMENTATION_VISITOR_H


#include <agx/config.h>

#include <agx/Real.h>

namespace agxControl
{
  class Action;
  class ActionImplementation;
  class CallbackActionImplementation;
  template<typename T> class Callback1ActionImplementation;
  template<typename R, typename T> class CallbackR1ActionImplementation;
  template<typename T1, typename T2> class Callback2ActionImplementation;


  class ActionImplementationVisitor
  {
  public:
    virtual void update(agxControl::ActionImplementation* action, agx::Real time) = 0;
    virtual void update(agxControl::CallbackActionImplementation* action, agx::Real time) = 0;
    virtual void update(agxControl::Callback1ActionImplementation<agx::Real>* action, agx::Real time) = 0;
    virtual void update(agxControl::CallbackR1ActionImplementation<void, agx::Real>* action, agx::Real time) = 0;
    virtual void update(agxControl::Callback2ActionImplementation<agx::Real, agx::Int>* action, agx::Real time) = 0;


    virtual void update(agxControl::ActionImplementation* begin, agxControl::ActionImplementation* end, agx::Real time) = 0;
    virtual void update(agxControl::CallbackActionImplementation* begin, agxControl::CallbackActionImplementation* end, agx::Real time) = 0;
    virtual void update(agxControl::Callback1ActionImplementation<agx::Real>* begin, agxControl::Callback1ActionImplementation<agx::Real>* end, agx::Real time) = 0;
    virtual void update(agxControl::CallbackR1ActionImplementation<void, agx::Real>* begin, agxControl::CallbackR1ActionImplementation<void, agx::Real>* end, agx::Real time) = 0;
    virtual void update(agxControl::Callback2ActionImplementation<agx::Real, agx::Int>* begin, agxControl::Callback2ActionImplementation<agx::Real, agx::Int>* end, agx::Real time) = 0;
  };
}


// Include guard.
#endif


/// \endcond
