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

#ifndef AGXUNIT_AGXOSGUNITTESTHELPERFUNCTIONS_H
#define AGXUNIT_AGXOSGUNITTESTHELPERFUNCTIONS_H


#include <agxOSG/export.h>
#include <agx/agx_vector_types.h>
#include <agxUnit/UnitMacros.h>
#include <agx/RigidBody.h>

namespace osg
{
  class Group;
}

namespace agx
{
  class RigidBody;
}
namespace agxCollide
{
  class GeometryContact;
}

namespace agxOSG {
  class Group;
  class ExampleApplication;
}

namespace agxSDK {
  class Simulation;
}


namespace agxUnit {

  typedef void (*ScenePtr)(
    agxSDK::Simulation* simulation,
    osg::Group* root );

  typedef agx::RigidBody* (*OneBodyRestingScenePtr)(
    agxSDK::Simulation* simulation,
    osg::Group* root );

  typedef agx::RigidBodyPtrVector (*ManyBodiesRestingScenePtr)(
    agxSDK::Simulation* simulation,
    osg::Group* root );


  AGXOSG_EXPORT osg::Group* testResting(
    agxSDK::Simulation* simulation,
    agxOSG::ExampleApplication* application,
    OneBodyRestingScenePtr restingScene,
    const char* groupName,
    const char* testName,
    agx::Real preTestingTime = agx::Real(1));

  AGXOSG_EXPORT osg::Group* testResting(
    agxSDK::Simulation* simulation,
    agxOSG::ExampleApplication* application,
    ManyBodiesRestingScenePtr restingScene,
    const char* groupName,
    const char* testName,
    agx::Real preTestingTime = agx::Real(1));

  enum NormalType {
    NORMAL,
    NORMAL_AND_NEGATIVE,
    ORTHOGONAL_TO_NORMAL,
    NORMAL_NO_DEPTH,
    IGNORE_NORMAL_AND_DEPTH
  };

  AGXOSG_EXPORT osg::Group* testContainsContact(
    agxSDK::Simulation* simulation,
    agxOSG::ExampleApplication* application,
    ScenePtr scene,
    const char* groupName,
    const char* testName,
    agx::Vec3f normal,
    agx::Real depth,
    NormalType normalType = NORMAL,
    unsigned int minNumContacts = 1);

  // Utility method for passing normal as Vec3 instead of Vec3f.
  AGXOSG_EXPORT osg::Group* testContainsContact(
    agxSDK::Simulation* simulation,
    agxOSG::ExampleApplication* application,
    ScenePtr scene,
    const char* groupName,
    const char* testName,
    agx::Vec3 normal,
    agx::Real depth,
    NormalType normalType = NORMAL,
    unsigned int minNumContacts = 1);



  AGXOSG_EXPORT osg::Group* testContainsNoContact(
    agxSDK::Simulation* simulation,
    agxOSG::ExampleApplication* application,
    ScenePtr scene,
    const char* groupName,
    const char* testName );
}

#endif

