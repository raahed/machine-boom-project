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

#include <agxCollide/Geometry.h>
#include <agx/RigidBody.h>

namespace agxCollide
{
  class Space;
}

namespace agx
{
  namespace detail
  {
    class ContactForceFilterParticipant
    {
      public:
        ContactForceFilterParticipant();
        ContactForceFilterParticipant(agx::RigidBody& body);
        ContactForceFilterParticipant(agxCollide::Geometry& geometry);

        agx::RigidBody* getBody();
        agxCollide::Geometry* getGeometry();

        bool isValid() const;

        static agxCollide::GeometryContactVector getContacts(
          ContactForceFilterParticipant& lhs, ContactForceFilterParticipant& rhs, agxCollide::Space& space);

      private:
        agx::RigidBodyObserver m_body;
        agxCollide::GeometryObserver m_geometry;
    };
  }
}