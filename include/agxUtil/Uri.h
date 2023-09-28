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

#include <agx/Integer.h>
#include <agx/String.h>

#include <agx/agxPhysics_export.h>

#ifdef _MSC_VER
# pragma warning(push)
# pragma warning(disable: 4251) // warning C4251: class X needs to have dll-interface to be used by clients of class Y
#endif

namespace agxUtil
{
  struct AGXPHYSICS_EXPORT Uri
  {
    public:
      Uri(const agx::String& uri);

      agx::String protocol;
      agx::String host;
      agx::UInt16 port = agx::InvalidIndex;
      agx::String path;
      agx::String parameters;

      bool isValid = false;

      /**
      Decode percent-encoded characters in an URL.

      \param url The URL to decode percent-encoded characters in.
      \return A string containing the decoded URL.
      */
      static agx::String decodeUrl(const agx::String& url);
  };
}

#ifdef _MSC_VER
#  pragma warning(pop)
#endif
