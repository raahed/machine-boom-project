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




#include <agx/agxPhysics_export.h>
#include <agx/Integer.h>


namespace agxStream
{
  DOXYGEN_START_INTERNAL_BLOCK()
  namespace detail
  {

    /**
    Test if a particular piece of serialization data is available in the
    archive. The issue is that we cannot trust the serialization version number
    alone because a subset of the archive changes may have been merged into an
    RC. In that case we cannot bump the serialization version. The
    serialization version says that ALL changes prior and including the seen
    version must be part of the archive. The RC, once created, will never
    changes its serialization version. To handle cherry picking of serialization
    changes we also look at the AGX version that wrote the archive.

    There is a range of serialization versions in which the data may be present
    and within that range we use the AGX version.

    We assume that the AGX version in the RC is lower than the AGX version at
    which the change was introduced in trunk, that the serialization version in
    the RC never changes, and that the AGX version was bumped in the RC when
    the cherry-pick merge was made.

    \param archiveAgxVersion - The AGX version that was used to write the archive.
    \param archiveSerializationVersion - The serialization version of the archive. All changes at or below this must be present.
    \param serializationVersionInRc - The serialization version in the RC.
    \param serializationVersionAtChange - The serialization version that the change got in trunk.
    \param agxVersionInRc - The AGX version that the RC was bumped to when the change was merged from trunk to the RC.
    \param agxVersionInTrunkAfterRc - The lowest AGX version that is higher than any version that the RC may ever reach.

    \return True if the data is present in the archive. False otherwise.
    */
    bool AGXPHYSICS_EXPORT shouldReadDataFromArchive(
        size_t const archiveAgxVersion,
        agx::UInt16 const archiveSerializationVersion,
        agx::UInt16 const serializationVersionInRc,
        agx::UInt16 const serializationVersionAtChange,
        size_t const agxVersionInRc,
        size_t const agxVersionInTrunkAfterRc);
  }
  DOXYGEN_END_INTERNAL_BLOCK()
}

