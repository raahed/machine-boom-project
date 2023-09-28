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

#ifndef AGXSDK_RIGIDBODYLOGLISTENER_H
#define AGXSDK_RIGIDBODYLOGLISTENER_H

#include <agxSDK/agxSDK.h>
#include <agxSDK/LogListener.h>
#include <agx/RigidBody.h>
#include <string>

namespace agxSDK
{

  /**
  Class that will log various properties of rigid bodies.
  */
  class AGXPHYSICS_EXPORT RigidBodyLogListener : public LogListener
  {
  public:

    /**
    RigidBodyLogListener constructor.
    \param fileName File name for logging. If empty, will write to standard out.
          If file exists, it will be overwritten.
    \param newFilePerTimeStep Should each time step be logged to a new file?
    \param valueSeparator String for separating the different columns
    */
    RigidBodyLogListener(const std::string& fileName = "",
      bool newFilePerTimeStep = false,
      const std::string& valueSeparator = "\t");

    /// Adds a body for logging.
    inline void addBody( agx::RigidBody* body );

  protected:
    virtual void obtainHeader( std::stringstream& header );
    virtual void collectData ( std::stringstream& data, const agx::TimeStamp& /*time*/ );


  protected:
    agx::Vector<agx::observer_ptr<agx::RigidBody> > m_bodies;
  };

  typedef agx::ref_ptr<RigidBodyLogListener> RigidBodyLogListenerRef;



  inline void RigidBodyLogListener::addBody( agx::RigidBody* body)
  {
    m_bodies.push_back( body );
  }

}

#endif /* AGXSDK_RIGIDBODYLOGLISTENER_H */
