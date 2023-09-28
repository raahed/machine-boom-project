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
#ifndef AGXOSG_GUIEVENTADAPTER_H
#define AGXOSG_GUIEVENTADAPTER_H

#include <agx/PushDisableWarnings.h> // Disabling warnings. Include agx/PopDisableWarnings.h below!
#include <agxOSG/export.h>
#include <agxSDK/GuiEventAdapter.h>
#include <osgGA/GUIEventHandler>
#include <agx/PopDisableWarnings.h> // End of disabled warnings.

DOXYGEN_START_INTERNAL_BLOCK()


namespace osgGA
{

  class GuiEventAdapter;
  class GUIActionAdapter;
}
DOXYGEN_END_INTERNAL_BLOCK()

namespace agxOSG
{
  /// Class for inserting mouse and keyboard event from OSG into agxSDK::Simulation
  class AGXOSG_EXPORT GuiEventAdapter : public agxSDK::GuiEventAdapter, public osgGA::GUIEventHandler
  {

  public:

    GuiEventAdapter( agxSDK::Simulation *simulation );
    virtual ~GuiEventAdapter();

  protected:

    virtual bool handle( const osgGA::GUIEventAdapter& ea, osgGA::GUIActionAdapter& );
    using osgGA::GUIEventHandler::handle;
    bool handleMouse( osgGA::GUIEventAdapter::EventType eventType, const osgGA::GUIEventAdapter& ea, osgGA::GUIActionAdapter& );
    bool handleKeyboard( osgGA::GUIEventAdapter::EventType eventType, const osgGA::GUIEventAdapter& ea, osgGA::GUIActionAdapter& );
    bool handleMouseMove( osgGA::GUIEventAdapter::EventType eventType, const osgGA::GUIEventAdapter& ea, osgGA::GUIActionAdapter& );
    bool handleUpdate( osgGA::GUIEventAdapter::EventType eventType, const osgGA::GUIEventAdapter& ea, osgGA::GUIActionAdapter& );

  };
}

#endif

