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


#ifndef AGXCONTROL_OPERATION_H
#define AGXCONTROL_OPERATION_H


#include <agx/config.h>

#include <agx/Referenced.h>
#include <agxStream/Serializable.h>

#include <agxControl/Action.h>
#include <agxControl/ActionImplementationVisitor.h>


namespace agxSDK
{
  class Simulation;
}

namespace agxControl
{

  class Action;
  class ActionManager;
  class ActionImplementation;
  class CallbackActionImplementation;
  template<typename T> class Callback1ActionImplementation;
  template<typename R, typename T> class CallbackR1ActionImplementation;
  template<typename T1, typename T2> class Callback2ActionImplementation;

  AGX_DECLARE_POINTER_TYPES(Operation);

  /**
  An operation represents an Action that happens over time. The Operation is
  updated every time step, triggering some internal Action.


  \internal

  What we need Operations for:
   - Interpolation of single-value, Real actions.
   - Goal-based operations, i.e.,
     "run the motor until the constraint reaches a given angle."

  The interpolation operation can take two Actions and interpolate between them.
  The problem is how to determine the value to interpolate when the Action has
  more than one parameter. Also, what type should the creator function (probably
  constructor) take? Should we have a general framework for accepting anything
  and interpolate whatever, or should we list a given set of Action subclasses
  that we can interpolate? What would those subclasses be?

  Should the Operation layer work on Actions or ActionImplementations? It must at
  least have the Actions for store/restore. The actual updates can be done via
  the visitor pattern. Operation calls a visit method on the ActionImplementation,
  which answers by calling a highly overloaded update method on the Operation. The
  overloading contains one method signature for every supported ActionImplementation
  subclass.

  Things to implement first:
   - A motor that starts and stops with an operation.
  */
  class AGXPHYSICS_EXPORT Operation : public agx::Referenced, public agxStream::Serializable
  {
  public:

    Operation(agxControl::Action* begin, agxControl::Action* end);

    virtual void activate(agx::Real time);
    virtual void update(agx::Real time);
    virtual void deactivate(agx::Real time);

    agx::Real getBeginTime() const;
    agx::Real getEndTime() const;

    agxControl::Action* getBeginAction();
    const agxControl::Action* getBeginAction() const;
    agxControl::Action* getEndAction();
    const agxControl::Action* getEndAction() const;

    const agxSDK::Simulation* getSimulation() const;


    AGXSTREAM_DECLARE_SERIALIZABLE(agxControl::Operation);
    /// Used only by store/restore.
    Operation();

  private:
    friend class agxSDK::Simulation;
    friend class agxControl::ActionManager;
    void setSimulation(agxSDK::Simulation* simulation);

  private:
    agx::ref_ptr<agxControl::Action> m_beginAction;
    agx::ref_ptr<agxControl::Action> m_endAction;
    /// \todo Want an observer pointer here. How should I structure my includes?
    agxSDK::Simulation* m_simulation;
  };




}

// Include guard.
#endif

/// \endcond
