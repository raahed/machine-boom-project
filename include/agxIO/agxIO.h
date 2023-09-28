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

#ifndef AGXIO_AGXIO_DOCS
#define AGXIO_AGXIO_DOCS

#include <agx/String.h>
#include <agx/agxCore_export.h>
/**
\namespace agxIO
\brief The agxIO namespace contains classes for reading, writing and finding files.
*/
namespace agxIO
{
  void AGXCORE_EXPORT readFile(agx::String& result, const agx::String& filePath);

  /**
   * Read the content of a file into a char array allocated by the function. Ownership
   * of the array is passed to the caller, who then becomes responsible for calling
   * 'delete []' on it.
   */
  AGXCORE_EXPORT char* readBinaryFile(size_t* size, const agx::String& filePath );
}

#endif

