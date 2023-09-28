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

#ifndef AGX_XMLLOADER_H
#define AGX_XMLLOADER_H

#include <agx/agx.h>
#include <agx/String.h>

#include <functional>

namespace agx
{
  class TiXmlElement;
  class Device;
  class Object;

  typedef std::function<Object *(TiXmlElement *, Device *)> XmlLoadFunction;

    struct AGXCORE_EXPORT XmlLoaderRegistrator
    {
      XmlLoaderRegistrator(const String& name, XmlLoadFunction loader);

      static bool             hasLoadFunction( const String& name );
      static XmlLoadFunction& getLoadFunction( const String& name );
    };

}

#endif

