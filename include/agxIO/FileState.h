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

#ifndef AGXIO_FILESTATE_H
#define AGXIO_FILESTATE_H

#include <agx/agx.h>
#include <agx/agxCore_export.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <agx/String.h>
#include <agx/Vector.h>
#include <agxIO/FileSystem.h>

namespace agxIO
{

  /// Class for storing and comparing a change date for a file on disk
  class AGXCORE_EXPORT FileState
  {
    public:
      /**
      Constructor
      \param filename - The file for which the date should be compared.
      */
      FileState( const agx::String& filename);

      /// Default constructor, not valid
      FileState();

      /// Is the modified date different from the last call to reset()/Constructor?
      bool hasChanged() const;

      /**
      Refresh the last changed info from the file
      \return returns true if reset was successful, false if file did not exist and was not readable
      */
      bool reset();

      /// \return true if the two file states are equal in modified time
      bool operator==(const FileState& other);

      /// \return true if the two file states are NOT equal in modified time
      bool operator!=(const FileState& other);

    private:
      agx::String m_filename;
      bool m_valid;
      FileSystem::StatStruct m_buf;
  };

  /**
  vector for holding a set of file states,
  where the vector can be asked if any of the files has been changed.
  */
  class AGXCORE_EXPORT FileStateVector : public agx::Vector<agxIO::FileState>
  {
      typedef agx::Vector<agxIO::FileState> Base;
    public:

      /// \return true if any of the FileStates have been changed
      bool hasChanged() const;

      /// Add a FileState to the vector
      void push_back( const FileState& fs );

      /// Add a filename to the vector
      void push_back( const agx::String& filename );
  };


} // namespace agxIO

#endif
