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


#ifndef AGXCONTROL_INTERPOLATING_OPERATION_H
#define AGXCONTROL_INTERPOLATING_OPERATION_H

#include <agx/config.h>

#include <agxControl/InterpolationController.h>
#include <agxControl/Operation.h>
#include <agxControl/ActionImplementationVisitor.h>

namespace agxControl
{


  class AGXPHYSICS_EXPORT InterpolatingOperation : public agxControl::Operation, public agxControl::ActionImplementationVisitor
  {
  public:
    InterpolatingOperation(agxControl::Action* begin, agxControl::Action* end, InterpolationController* controller);

    virtual void activate(agx::Real time) override;
    virtual void update(agx::Real time) override;
    virtual void deactivate(agx::Real time) override;


    AGXSTREAM_DECLARE_SERIALIZABLE(agxControl::InterpolatingOperation);
    /// Used only by store/restore.
    InterpolatingOperation() {};

  public:

    /*
    Single dispatch visitor pattern. Is this ever used? When will we need to visit
    only a single action?
    */
    void update(agxControl::ActionImplementation* action, agx::Real time) override;
    void update(agxControl::CallbackActionImplementation* action, agx::Real time) override;
    void update(agxControl::Callback1ActionImplementation<agx::Real>* action, agx::Real time) override;
    void update(agxControl::CallbackR1ActionImplementation<void, agx::Real>* action, agx::Real time) override;
    void update(agxControl::Callback2ActionImplementation<agx::Real, agx::Int>* action, agx::Real time) override;


    /*
    Dual dispatch visitor pattern. The Operation contains two Actions and we want
    to run one of the update methods below. This is achieved by calling the virtual
    visit(Visitor*, ActionImplementation*, .) method on the begin ActionImplementation,
    which in turn will call the virtual dispatchVisit(Visitor*, <something>Implementation<something>*, .)
    on the received ActionImplementation. Implementations for all supported variants
    of the virtual dispatchVisit method is given in the base ActionImplementation,
    but they all Throw exception. Implementation in the derived classes are given
    for the methods that take a derivative of ActionImplementation being the same
    type as the class providing the implementation. That implementation calls one
    of the update methods below.
    */
    void update(agxControl::ActionImplementation* begin, agxControl::ActionImplementation* end, agx::Real time) override;
    void update(agxControl::CallbackActionImplementation* begin, agxControl::CallbackActionImplementation* end, agx::Real time) override;
    void update(agxControl::Callback1ActionImplementation<agx::Real>* begin, agxControl::Callback1ActionImplementation<agx::Real>* end, agx::Real time) override;
    void update(agxControl::CallbackR1ActionImplementation<void, agx::Real>* begin, agxControl::CallbackR1ActionImplementation<void, agx::Real>* end, agx::Real time) override;
    void update(agxControl::Callback2ActionImplementation<agx::Real, agx::Int>* begin, agxControl::Callback2ActionImplementation<agx::Real, agx::Int>* end, agx::Real time) override;


  protected:
    virtual ~InterpolatingOperation() {}
  private:
    agxControl::InterpolationControllerRef m_controller;
  };

}

// Include guard.
#endif

/// \endcond
