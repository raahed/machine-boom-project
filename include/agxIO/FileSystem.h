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

#include <agx/agx.h>
#include <agx/agxCore_export.h>
#include <agx/Callback.h>
#include <agx/String.h>
#include <agx/PushDisableWarnings.h> // Disabling warnings. Include agx/PopDisableWarnings.h below!
#include <sys/stat.h>
#include <agx/PopDisableWarnings.h> // End of disabled warnings.

namespace agxIO
{
  struct FileSystem {

    #ifdef _MSC_VER
    typedef struct _stat64 StatStruct;
    #else
    typedef struct stat StatStruct;
    #endif

#ifndef SWIG
    /// Separator between directories in a file path: on unix "/", on windows "\\"
    AGXCORE_EXPORT static const char* PATH_SEPARATOR;

    /// Path separator under UNIX "/"
    AGXCORE_EXPORT static const char* UNIX_PATH_SEPARATOR;

    /// Path separator under Windows "\\"
    AGXCORE_EXPORT static const char* WINDOWS_PATH_SEPARATOR;

    /// Separator between paths in a list of paths, on unix: ":", on windows ";"
    AGXCORE_EXPORT static const char* PATH_DELIMITER;

    /// Filetype for dynamically loaded libraries (dll / so / dylib)
    AGXCORE_EXPORT static const char* DLL_SUFFIX;
#endif

    /**
    Open a file. The filename should be in ascii or utf-8 format.
    */
    AGXCORE_EXPORT static FILE* openFile( const char* file, const char* mode );

    /**
    Open a file. The filename should be in ascii or utf-8 format.
    */
    AGXCORE_EXPORT static FILE* openFile( const agx::String& str, const char* mode );

    /**
    Close the FILE* stream.
    \return true on success.
    */
    AGXCORE_EXPORT static bool closeFile( FILE* stream );

    /**
    Read the contents of filename and return it as a String.
    */
    static AGXCORE_EXPORT agx::String readFile( const agx::String& filename );

    /**
    Creates a directory on the host file system.
    \param path - Path to the directory that will be created
    \param recursive - If true, non existing directories in the path will be created, not just the last.
    \return true if successful
    */
    static AGXCORE_EXPORT bool mkdir(const agx::String& path, bool recursive = false);

    /**
    Scan a directory.
    */
    typedef agx::Callback1<const agx::String&> ScanCallback;
    static AGXCORE_EXPORT bool scan(const agx::String& path, ScanCallback callback);

    /**
    Test if a path is a directory.
    */
    static AGXCORE_EXPORT bool isDir(const agx::String& path);

    /**
     * \returns true if the given path is an absolute path.
     */
    static AGXCORE_EXPORT bool isAbsolute(const agx::String& path);

    /**
    \return true if the path is a link. Under windows this method always return false.
    */
    static AGXCORE_EXPORT bool isLink( const agx::String& path );

    /**
    /return The size of the file.
    */
    static AGXCORE_EXPORT size_t size( const agx::String& path );

    /**
    Test if a path exists.
    \return true if file exists.
    */
    static AGXCORE_EXPORT bool exists(const agx::String& path);

    /**
    Test if a path exists and is readable.
    \return true if path exists and is readable.
    */
    static AGXCORE_EXPORT bool readable(const agx::String& path);

    /**
    Test if a path exists and is writeable.
    \return true if path exists and is writeable.
    */
    static AGXCORE_EXPORT bool writeable(const agx::String& path);

    /**
    \return A platform specific directory for temporary files.
    */
    static AGXCORE_EXPORT agx::String getSystemTmpDirectory();

    /**
    Create a temporary directory.
    \return Empty string if failed
    */
    static AGXCORE_EXPORT agx::String mkTmpDirectory(const agx::String& prefix = "");
    static AGXCORE_EXPORT agx::String mkTmpDirectory(const agx::String& parentDirectory, const agx::String& prefix);

    /**
    \return difference between last modified time of the two files \p path1 and \p path 2,
    OR 0 if none of the two files could be accessed OR 1 if \p path1 could not be accessed OR -1 of
    \p path2 could not be accessed.
    */
    static AGXCORE_EXPORT time_t compareDate(const agx::String& path1, const agx::String& path2);

    static AGXCORE_EXPORT bool rename(const agx::String& oldPath, const agx::String& newPath);

    static AGXCORE_EXPORT bool copy(const agx::String& source, const agx::String& destination);

    static AGXCORE_EXPORT bool remove(const agx::String& path, bool recursive = false);

    /**
    \param unixStyle - Path in unix style (with / separators)
    \return path in Windows style (with \\)
    */
    static AGXCORE_EXPORT agx::String convertUnixFilePathToWindows( const agx::String& unixStyle );

    /**
    \param windowsStyle - Path in windows style (with \ separators)
    \return path in Unix style (with \)
    */
    static AGXCORE_EXPORT agx::String convertWindowsFilePathToUnix( const agx::String& windowsStyle );

    /// \return \p filepath converted to the current platforms filepath specification
    static AGXCORE_EXPORT agx::String convertToNativeFilePath( const agx::String& filePath );

    /**
    Take a filepath, convert it to Unix format (using UNIX_PATH_SEPARATOR) and finally convert it
    to a format that Collada DAE api accepts.

    For windows this means that "c:\path" -> "/c:/path"

    If the string "://"   can be found in \p filePath then no conversion will take place, as it
    assumes that the path is already in the correct format.
    \return filePath converted to collada compliant file path format.
    */
    static AGXCORE_EXPORT agx::String convertToColladaFilePath( const agx::String& filePath );

    /// \return the extension (without .) of the file
    static AGXCORE_EXPORT agx::String getFileExtension(const agx::String& filePath);

    /// \return the filename only (no extension no path)
    static AGXCORE_EXPORT agx::String getFileWithoutExtension(const agx::String& filePath);

    /// \return the filename including file extension (no path)
    static AGXCORE_EXPORT agx::String getFileWithExtension(const agx::String& filePath);

    /// \return the file system directory containing the file
    static AGXCORE_EXPORT agx::String getDirectory(const agx::String& filePath);

    /**
    Returns a list of all files in a directory
    \param result - The vector that will be populated with all files in \p path.
    \param path - The path to the directory where files will be found. If empty current directory ("./") will be used.
    \param recursive - If true, a recursive search for files will be performed.
    \return number of found files
    */
    static AGXCORE_EXPORT size_t getFilesInDirectory(agx::StringVector& result, const agx::String& path, bool recursive = false);

    /**
    Combine two paths with a separator.
    This method will return the path in native format
    Any double occurrence of path separators will be removed.
    \param path1 - First path to be combined
    \param path2 - Second path to be combined
    \return path1 and path2 combined with a separator
    */
    static AGXCORE_EXPORT agx::String combinePath(const agx::String& path1, const agx::String& path2);


    ///\ return the path without any ending filename (only the directory/disk) in native format
    static agx::String AGXCORE_EXPORT getPath(const agx::String& path);

    /**
    If \p is a relative path (isabsolute() returns false), the path will be expanded into a full path
    using current working directory.
    \return \p relativePath expanded into an absolute path
    */
    static agx::String AGXCORE_EXPORT getAbsolutePath( const agx::String& relativePath );

    /**
    \return The current working directory in native path format.
    */
    static agx::String AGXCORE_EXPORT getCurrentWorkingDirectory();

    /**
    Set the current working directory.
    */
    static bool AGXCORE_EXPORT setCurrentWorkingDirectory(const agx::String& path);

    /**
    Wrapper around the stat-function to work around a bug in Visual Studio
    when passing an empty string to stat.
    \param path The path to the file (c-string in original stat)
    \param buf The pointer to a preallocated stat-struct (see original stat-function).
    */
    static AGXCORE_EXPORT int stat(const agx::String& path, StatStruct* buf);

    /**
    \return a string with a unique filename including a path which can be used to create a temporary file
    */
    static AGXCORE_EXPORT agx::String getTmpFilename();

  }; // struct FileSystem
} // Namespace agxIO
