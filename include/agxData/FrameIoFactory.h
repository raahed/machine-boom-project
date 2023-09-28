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


#ifndef AGXDATA_FRAME_IO_FACTORY
#define AGXDATA_FRAME_IO_FACTORY

#include <agxData/DiskTrack.h>
DOXYGEN_START_INTERNAL_BLOCK()

namespace agxData
{
  namespace FrameIoFactory
  {
    /** Given a JournalArchive::Session, create a suitable FrameReader for the session. Will
    return nullptr if there is no suitable reader implemented for the journal format. */
    AGXCORE_EXPORT agxData::DiskFrameReader* createFrameReader( agxData::JournalArchive::Session* session );
    /** Given a JournalArchive::Session, create a suitable FrameWriter for the session. Will
       return nullptr if there is no suitable reader implemented for the journal format. */
    AGXCORE_EXPORT agxData::DiskFrameWriter* createFrameWriter( agxData::JournalArchive::Session* session );
  }
}

DOXYGEN_END_INTERNAL_BLOCK()

// Include guard.
#endif
