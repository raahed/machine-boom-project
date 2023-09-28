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

#ifndef AGXIO_FILEPATHCONTAINER_H
#define AGXIO_FILEPATHCONTAINER_H

#ifdef _MSC_VER
# pragma warning(push)
# pragma warning(disable:4786) // Disable warnings about long names
#endif

#include <agxIO/FileSystem.h>
#include <agx/agxCore_export.h>
#include <agx/List.h>

namespace agxIO
{
  /// class for locating files
  class AGXCORE_EXPORT FilePathContainer
  {

    public:
      typedef agx::List< agx::String> PathList;

    public:
      /**
      Constructor
      */
      FilePathContainer();

      /**
      Constructor - set the file path found in given environment variable
      */
      FilePathContainer(const agx::String& env);

      /** Destructor */
      ~FilePathContainer();

      /**
      Clear all added path
      */
      void clear();

      /**
      Set the file path using a single string delimited with given delimiter
      \param paths - PATH_DELIMITER separated string with paths
      \param clearPrevious - clear previously added information to the path list of this instance
      \param delimiter - the delimiter (PATH_DELIMITER) defaults to the os where the code is compiled
      */
      void setFilePathList(const agx::String& paths,
                           bool clearPrevious = true,
                           const agx::String& delimiter = agxIO::FileSystem::PATH_DELIMITER);

      /** Set the file path found in given environment variable
      \return true if env var found
      */
      bool setFromEnvironmentVariable(const agx::String& env, bool clearPrevious = true );

      /**
      Set the file path directly. Clear previously set data.
      */
      void setFilePath(const agx::String& path);

      /**
      Adds a list of paths to the end previous one
      \param path - string of paths each separated with PATH_DELIMITER
      */
      void addFilePath(const agx::String& path);

      /**
      Removes a list of paths.
      \param path - string of paths each separated with PATH_DELIMITER
      */
      void removeFilePath(const agx::String& path);

      /**
      Adds a filepath to the end of the list
      \param path - string of paths each separated with PATH_DELIMITER
      \
      */
      void pushbackPath(const agx::String& path);

      /**
      Adds a filepath to the beginning of the list
      \param path - string of paths each separated with PATH_DELIMITER
      */
      void pushfrontPath(const agx::String& path);

      /**
      \return file path list
      */
      PathList& getFilePathList();

      /**
      \return file path list
      */
      const PathList& getFilePathList() const;

      /**
      First it will try to locate 'filename' in one of the specified directories.
      Then it will search for it as 'filename' (current dir).
      \return path to given filename if file exists in any directory
      */
      agx::String find(const agx::String& filename) const;

    private:

      /** Path list */
      PathList m_path_list;


      /** Convert given string to list with sub strings (separated by delimiter in string) */
      void convertStringToList(const agx::String& str, PathList& lst, const agx::String& delimiter);

      void removeDuplicates();

      /** Delimiter used when parsing environment variable */
      agx::String m_delimiter;

  };

}

#ifdef _MSC_VER
# pragma warning(pop)
#endif
#endif // AGXIO_FILEPATHCONTAINER_H
