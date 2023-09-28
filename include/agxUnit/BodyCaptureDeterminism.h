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

#ifndef AGXUNIT_BODYCAPTUREDETERMINISM_H
#define AGXUNIT_BODYCAPTUREDETERMINISM_H

#include <agx/agxPhysics_export.h>

#include <agxSDK/StepEventListener.h>
#include <agx/RigidBody.h>
#include <fstream>
namespace agxUnit
{


/// Class for capturing data for measuring determinism
class AGXPHYSICS_EXPORT BodyCaptureDeterminism : public agxSDK::StepEventListener
{
public:

  struct Session : public agx::Referenced
  {
    agx::Vector<agx::Vec3> position;
    agx::Vector<agx::Quat> orientation;
    agx::Vector<float> time;
  };

  BodyCaptureDeterminism(const agx::String &file );
  void setRigidBody( agx::RigidBody *body) { m_body=body; newSession(); }

  void post(const agx::TimeStamp& t);
  void capture(const agx::TimeStamp& t);

  void removeNotification();

  void newSession() { m_sessions.push_back( new Session ); }
  void writeLine( Session *session, size_t i );

  void calcDiff( Session *s0, Session *s1, size_t i, agx::Real data[9] );

  agx::Real getError();

protected:
  typedef agx::Vector<agx::ref_ptr<Session> > SessionVector;
  SessionVector m_sessions;

  agx::String m_file;
  std::ofstream m_stream;
  agx::observer_ptr<agx::RigidBody> m_body;
  agx::Real m_error;
};

}

#endif

