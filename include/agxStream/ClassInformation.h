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
#pragma once


#include <string>
#include <agx/agx.h>
#include <agx/agxCore_export.h>

#ifdef _MSC_VER
# pragma warning(push)
# pragma warning(disable: 4251) // warning C4251: class X needs to have dll-interface to be used by clients of class Y
#endif


namespace agxStream
{
  /// Class that extracts namespace, class name and method name when called with the AGX_FUNCTION macro.
  struct AGXCORE_EXPORT ClassInformation {
    ClassInformation( const char *functionString );

    /// \return true if the string contained a valid method and className
    bool valid() const {
      return method.length() > 0 && className.length() > 0 ;
    }

    /// \return namespace::className
    std::string getClassType() const;

    std::string nameSpace;
    std::string method;
    std::string className;
  };

} // Namespace agxStream

#ifdef _MSC_VER
#  pragma warning(pop)
#endif

