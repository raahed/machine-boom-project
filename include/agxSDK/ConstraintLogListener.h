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

#ifndef AGXSDK_CONSTRAINTLOGLISTENER_H
#define AGXSDK_CONSTRAINTLOGLISTENER_H

#include <agxSDK/agxSDK.h>
#include <agxSDK/LogListener.h>
#include <agx/Constraint.h>
#include <agx/TimeStamp.h>
#include <string>


namespace agxSDK
{
  /// Class that will log various properties of constraints
  class AGXPHYSICS_EXPORT ConstraintLogListener : public LogListener
  {
  public:

    /**
    ConstraintLogListener constructor.
    \param fileName File name for logging. If empty, will write to standard out.
          If file exists, it will be overwritten.
    \param newFilePerTimeStep Should each time step be logged to a new file?
    \param valueSeparator String for separating the different columns
    */
    ConstraintLogListener(const std::string& fileName = "",
      bool newFilePerTimeStep = false,
      const std::string& valueSeparator = "\t");

    /**
    Adds a constraint for logging.
    \param constraint Pointer to constraint
    \param bodyIndex Index of attached rigid body at constraint.
    */
    inline void addConstraint( agx::Constraint* constraint, int bodyIndex = 0);


  protected:
    virtual void obtainHeader( std::stringstream& header );
    virtual void collectData ( std::stringstream& data, const agx::TimeStamp& /*time*/ );


  protected:
    typedef std::pair<agx::observer_ptr<agx::Constraint>, int> AttachedConstrait;
    agx::Vector< AttachedConstrait > m_constraints;
  };

  typedef agx::ref_ptr<ConstraintLogListener> ConstraintLogListenerRef;

  inline void ConstraintLogListener::addConstraint( agx::Constraint* constraint,
    int bodyIndex /*=0*/)
  {
    m_constraints.push_back( AttachedConstrait( constraint, bodyIndex ) );
    constraint->setEnableComputeForces( true );
  }
}

#endif /* AGXSDK_CONSTRAINTLOGLISTENER_H */
