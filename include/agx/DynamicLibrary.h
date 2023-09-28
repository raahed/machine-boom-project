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
#ifndef AGX_DYNAMIC_LIBRARY_H
#define AGX_DYNAMIC_LIBRARY_H


#include <agx/agxCore_export.h>
#include <agx/Referenced.h>
#include <string>
#include <agxIO/FilePathContainer.h>

namespace agx {

  class FilePathContainer;

  /**
  */
  AGX_DECLARE_POINTER_TYPES( DynamicLibrary );
  class AGXCORE_EXPORT DynamicLibrary : public agx::Referenced
  {
  public:

    typedef void*   HANDLE;
    typedef void*   PROC_ADDRESS;

    /**
    Load a named library (without file extension). Library should be located in one of the
    directories in the given FilePathContainer.
    \param libraryName - Name of dynamic library that should be loaded (without extension).
    \param fp - FilePathContainer containing list of paths where the library will be searched.
    \return Pointer to a DynamicLibrary object on successfully opening of library returns nullptr on failure.
    */
    static DynamicLibrary* loadLibrary(const agx::String& libraryName, agxIO::FilePathContainer &fp,
                                       const agx::String& additionalDirectory="");

    /**
    \return Name of library stripped of path.
    */
    const agx::String& getName() const     { return m_name; }

    /**
    \return Name of library including full path.
    */
    const agx::String& getFullName() const { return m_fullName; }

    /**
    \return Handle to the loaded dynamic library (.dso/.dll).
    */
    HANDLE             getHandle() const   { return m_handle; }

    /**
    \param procName - Name of the function that should be located.
    \return Address of the named function located in library.
    */
    PROC_ADDRESS       getProcAddress(const agx::String& procName);

  protected:

    /** get handle to library file */
    static HANDLE getLibraryHandle( const agx::String& libraryName);

    /** disallow default constructor.*/
    DynamicLibrary() {}
    /** disallow copy constructor.*/
    DynamicLibrary(const DynamicLibrary&) : Referenced() {}
    /** disallow copy operator.*/
    DynamicLibrary& operator == (const DynamicLibrary&) { return *this; }

    /**
      Disallow public construction so that users have to go
      through loadLibrary() above which returns nullptr on
      failure, a valid DynamicLibrary object on success.
    */
    DynamicLibrary(const agx::String& name,HANDLE handle);
    virtual ~DynamicLibrary();

    HANDLE          m_handle;
    agx::String     m_name;
    agx::String     m_fullName;

  };

} // namespace agx


#endif //
