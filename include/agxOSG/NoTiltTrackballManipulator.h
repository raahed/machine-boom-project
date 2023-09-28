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


#ifndef AGXOSG_NOTILTTRACKBALLMANIPULATOR_H
#define AGXOSG_NOTILTTRACKBALLMANIPULATOR_H

#include <agx/PushDisableWarnings.h> // Disabling warnings. Include agx/PopDisableWarnings.h below!
#include <osgGA/TrackballManipulator>
#include <agx/PopDisableWarnings.h> // End of disabled warnings.

#include <agxOSG/export.h>
#include <agx/macros.h>

namespace agxOSG
{
  class AGXOSG_EXPORT NoTiltTrackballManipulator : public osgGA::TrackballManipulator
  {
  public:
    virtual void home(const osgGA::GUIEventAdapter& ea, osgGA::GUIActionAdapter& us) override
    {
      osgGA::TrackballManipulator::home(ea, us);
    }

    virtual void home(double d) override
    {
      osgGA::TrackballManipulator::home(d);
    }

    bool calcMovement();
    virtual bool handle(const osgGA::GUIEventAdapter& ea, osgGA::GUIActionAdapter& us) override;
    using osgGA::GUIEventHandler::handle;
  private:

  };
}

#endif /* _AGXOSG_NOTILTTRACKBALLMANIPULATOR_H_ */
