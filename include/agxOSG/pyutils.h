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
#ifndef AGXOSG_PYUTILS_H
#define AGXOSG_PYUTILS_H

#include <agxOSG/export.h>
#include <agx/Vec4.h>

#include <agxUtil/agxUtil.h>
#include <agxOSG/Node.h>
#include <agxOSG/ParticleSystemDrawable.h>
#include <agxSDK/GuiEventListener.h>
#include <agxOSG/utils.h>

#include <agx/PushDisableWarnings.h> // Disabling warnings. Include agx/PopDisableWarnings.h below!
#include <osg/observer_ptr>
#include <agx/PopDisableWarnings.h> // End of disabled warnings.

namespace osg
{
  class Group;
}

namespace agxModel
{
  class FractureGenerator;
}

namespace osg
{
  AGXOSG_EXPORT void Group_reference(osg::Group *node);
  AGXOSG_EXPORT void Group_unreference(osg::Group *node);
}

namespace agxOSG
{

  class Text;

  AGXOSG_EXPORT void GeometryNode_reference(agxOSG::GeometryNode *node);
  AGXOSG_EXPORT void GeometryNode_unreference(agxOSG::GeometryNode *node);


  AGXOSG_EXPORT void Text_reference(agxOSG::Text *node);
  AGXOSG_EXPORT void Text_unreference(agxOSG::Text *node);

  AGXOSG_EXPORT void Transform_reference(agxOSG::Transform *node);
  AGXOSG_EXPORT void Transform_unreference(agxOSG::Transform *node);

}

#endif /* _AGXOSG_UTILS_H_ */
