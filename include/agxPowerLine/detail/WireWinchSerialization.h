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


#ifndef AGXPOWERLINE_DETAIL_WIRE_WINCH_SERIALIZATION_H
#define AGXPOWERLINE_DETAIL_WIRE_WINCH_SERIALIZATION_H

#include <agxStream/shouldReadDataFromArchive.h>
#include <agx/version.h>


namespace agxPowerLine
{
  namespace detail
  {
    inline bool shouldReadWireWinchConnectorData(size_t archiveAgxVersion, agx::UInt16 archiveSerializationVersion)
    {
      /*
      Using AGX version instead of serialization version because of serialization
      breaking change in a release candidate (RC).

      A bug fix was required in 2.13.2 and the fix required changes to the
      serialization format. The fix was implemented in trunk and the serialization
      version increased to 61. However, we cannot bump the serialization format in
      the RC because there are intermediate changes to the serialization in trunk
      that we don't want in the RC. We therefore need another way to determine if
      the archive being read contains the new data or not.

      By looking at the AGX version instead fo the serialization version we can
      determine if the archive was written with the extra data or not regardless of
      whether it was written from trunk or the RC
      */
      agx::UInt16 const serializationVersionInRc = 58;
      agx::UInt16 const serializationVersionAtChange = 61;
      size_t const agxVersionInRc = AGX_CALC_VERSION(2, 13, 2, 5);
      size_t const agxVersionInTrunkAfterRc = AGX_CALC_VERSION(2, 14, 0, 0);

      return agxStream::detail::shouldReadDataFromArchive(
            archiveAgxVersion,
            archiveSerializationVersion,
            serializationVersionInRc,
            serializationVersionAtChange,
            agxVersionInRc,
            agxVersionInTrunkAfterRc);
    }
  }
}

#endif
