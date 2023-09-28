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

#ifndef AGXOSG_ROCKGENERATOR_H
#define AGXOSG_ROCKGENERATOR_H

#include <agxOSG/export.h>
#include <agx/Referenced.h>
#include <agx/Vec3.h>

#include <agx/PushDisableWarnings.h> // Disabling warnings. Include agx/PopDisableWarnings.h below!
#include <osg/Node>
#include <agx/PopDisableWarnings.h> // End of disabled warnings.

namespace agxOSG {
  class Node;
}

namespace agxCollide {
  class Space;
}

namespace agx {
  class Material;
  class RigidBody;
}

namespace agxSDK {
  class Assembly;
}

namespace agxOSG
{
  class AGXOSG_EXPORT RockGenerator : public agx::Referenced
  {
    public:
      enum RockType
      {
        COMPOSITE_ONE,
        COMPOSITE_TWO,
        COMPOSITE_THREE,
        COMPOSITE_FOUR,
        MAYA_ROCK1,
        MAYA_ROCK2,
        MAYA_ROCK3,
        NUM_ROCK_TYPES
      };

      RockGenerator();

      void generateRocks( agxSDK::Assembly *root,
                          agxCollide::Space *space,
                          int numRocks,
                          osg::Group *visualRoot,
                          agx::Real emitterRadius,
                          agx::Real elementRadius,
                          agx::Material *rockMaterial,
                          agx::Real emitVelocityMagnitude = 0.0,
                          agx::Real elementPositionJitter = 0.5,
                          agx::Real elementRadiusJitter = 0.5,
                          agx::Real elementSpacing = 0.0 );


    protected:
      virtual ~RockGenerator();

      agx::RigidBody *generateRock(osg::Group *visualRoot, RockType type, agx::Real elementRadius, agx::Real elementPositionJitter, agx::Real elementRadiusJitter, agx::Real elementSpacing, agx::Material *rockMaterial);
      void generateMayaRock(osg::Group *visualRoot, RockType type, agx::RigidBody *body, agx::Real elementRadius, agx::Real elementPositionJitter, agx::Material *rockMaterial);
      void addSphereElement(agx::RigidBody *body, osg::Group *visualRoot, const agx::Vec3& localOffset, agx::Real elementRadius, agx::Real elementPositionJitter, agx::Real elementRadiusJitter, agx::Material *rockMaterial);

    private:
      osg::ref_ptr<osg::Node> m_visualRock1;
      osg::ref_ptr<osg::Node> m_visualRock2;
      osg::ref_ptr<osg::Node> m_visualRock3;
  };

  typedef agx::ref_ptr<RockGenerator> RockGeneratorRef;
}

#endif /* AGXOSG_ROCKGENERATOR_H */
