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

#ifndef AGXDATA_TYPEMAP_H
#define AGXDATA_TYPEMAP_H

#include <agx/String.h>
#include <agx/HashTable.h>
#include <agx/HashSet.h>
#include <agx/SetVector.h>

namespace agx
{
  class TiXmlElement;
}

namespace agxData
{
  struct AGXCORE_EXPORT TypeImplementation
  {
    TypeImplementation() {}
    TypeImplementation(const agx::String& impl, const agx::String& inc) : implementation(impl), include(inc) {}
    agx::String implementation;
    agx::String include;
  };

  typedef agx::HashTable<agx::String, TypeImplementation> TypeMap;

  // TypeMap buildTypeMap();

  agx::String AGXCORE_EXPORT getTypeImplementationName(const agx::String& abstractTypeName);
  agx::String AGXCORE_EXPORT getTypeImplementationName(const agx::String& abstractTypeName, agx::TiXmlElement *element, agx::SetVector<agx::String> *includeHash = nullptr);

  agx::String AGXCORE_EXPORT getIncludePrefix(agx::TiXmlElement *element);
  agx::String AGXCORE_EXPORT getIncludePrefix(const agx::String& rootPath);
}


#endif /* AGXDATA_TYPEMAP_H */
