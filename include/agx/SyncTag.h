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

#ifndef AGX_SYNCTAG_H
#define AGX_SYNCTAG_H

#include <agx/Integer.h>
#include <agx/Referenced.h>
#include <iosfwd>

DOXYGEN_START_INTERNAL_BLOCK()

#ifdef _WIN32
#ifdef ERROR
#undef ERROR
#endif
#endif

namespace agx
{
  AGX_DECLARE_POINTER_TYPES(SyncTag);
  class AGXCORE_EXPORT SyncTag : public Referenced
  {
  public:
    enum Status
    {
      SYNC_TAG_ERROR = -1,
      SYNC_TAG_RUNNING,
      SYNC_TAG_COMPLETED
    };

  public:
    virtual Status wait() = 0;
    virtual Status getStatus() = 0;
  };

  /* Implementation */
  inline std::ostream& operator << ( std::ostream& output, const SyncTag& tag)
  {
    output << &tag;
    return output;
  }
}


#endif /* _AGX_SYNCTAG_H_ */

DOXYGEN_END_INTERNAL_BLOCK()
