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

#include <agx/agxCore_export.h>

#include <string>
#include <cstddef>
#include <stdexcept>
#include <agx/Integer.h>
#include <agx/Referenced.h>

namespace agxStream
{
  inline char littleEndian()
  {
    int i = 1;
    return *((char*)&i);
  }

  /// magic number for archives
  const agx::UInt32 ARCHIVE_MAGIC = 0xFADDED0;

  // header tags
  enum HeaderTag {
    OBJECT_TAG = 0x1,
    CLASS_TAG = 0x2
  };


  // storage modes
  enum StorageMode {
    STORE_BY_VALUE = 0x4,
    STORE_BY_REF = 0x8,
    STORE_NULL_POINTER = 0x10
  };


  /**
  Abstract base class for input/output storage of Serializable classes
  */
  class AGXCORE_EXPORT Archive : public agx::Referenced
  {
  public:
    Archive() : m_isOpen(false), m_binaryMode(true) {}

    /// \return true if archive is used in binary mode
    bool getBinaryMode() const {
      return m_binaryMode;
    }

    /// \param m - If true this archive is used in binary mode
    void setBinaryMode(bool m) {
      m_binaryMode = m;
    }

    /// Called whenever a new item is initiated
    virtual void newAttribute() {}

    ///\return true if archive is open for reading/writing
    bool isOpen() const {
      return m_isOpen;
    }

    virtual void close() {
      m_isOpen = false;
    }

    virtual void beginSection(const char *) = 0;
    virtual void endSection(const char *) = 0;

    /**
    Ties a beginSecion/endSection pair to a scope.
    */
    class AGXCORE_EXPORT ScopedSection
    {
    public:
      /**
      The data pointed to by 'sectionName' is not copied. It is the
      responsibility of the caller to ensure that the pointer is valid for
      the duration of the ScopedSection's lifetime.
      \param archive - The archive in which a beginSection/endSection will be created
      \param sectionName - The name of the created section
      \param shouldWeCreateSection - If false no section will be created. Default true.
      */
      ScopedSection(Archive& archive, const char* sectionName, bool shouldWeCreateSection=true);
      ~ScopedSection();
    private:
      ScopedSection(const ScopedSection&);
      ScopedSection& operator=(const ScopedSection&);
    private:
      Archive& m_archive;
      const char* m_sectionName;
      bool m_shouldCreate;
    };

  protected:

    /// Destructor
    virtual ~Archive() {  }


    void open() {
      m_isOpen = true;
    }

    bool m_isOpen;
  private:
    bool m_binaryMode;
  };
  typedef agx::ref_ptr<Archive> ArchiveRef;

  class ArchiveException : public std::runtime_error
  {
  public:
    explicit ArchiveException(const std::string& msg) : runtime_error(msg.c_str()) { }

  };

  class ArchiveEofException : public std::runtime_error
  {
  public:
    explicit ArchiveEofException(const std::string& msg) : runtime_error(msg.c_str()) { }

  };

  class EndOfArchiveException : public std::runtime_error
  {
  public:
    explicit EndOfArchiveException(const std::string& msg) : runtime_error(msg.c_str()) { }

  };

}


