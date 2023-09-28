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

#ifndef AGX_ENCODING_H
#define AGX_ENCODING_H

#include <agx/agx.h>
#include <string>

namespace agx
{


  bool AGXCORE_EXPORT isASCII( const char* str );

  bool AGXCORE_EXPORT isASCII( const std::string& s );


  std::wstring AGXCORE_EXPORT utf8ToWide( const char* str );
  std::wstring AGXCORE_EXPORT utf8ToWide( const std::string& s );

  std::string  AGXCORE_EXPORT wideToUtf8( const std::wstring& wide );

#ifdef _MSC_VER
  inline AGXCORE_EXPORT std::wstring  UnicodeName( const char *s )
  {
    return agx::utf8ToWide(s);
  }

  inline AGXCORE_EXPORT std::wstring  UnicodeName(const std::string& s )
  {
    return agx::utf8ToWide(s);
  }

#else
  inline AGXCORE_EXPORT const char*  UnicodeName( const char *s )
  {
    return s;
  }

  inline AGXCORE_EXPORT const std::string&  UnicodeName(const std::string& s )
  {
    return s;
  }
#endif

}


#endif

