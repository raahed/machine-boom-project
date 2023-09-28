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
#ifndef AGXSTREAM_INPUT_ARCHIVE_H
#define AGXSTREAM_INPUT_ARCHIVE_H

#ifdef _MSC_VER
# pragma warning(push)
# pragma warning( disable : 4290 )  // C++ exception specification ignored except to indicate a function is not __declspec(nothrow)
#endif

#include <agx/agx.h>
#include <agx/agxCore_export.h>

#include <agx/Vector.h>
#include <agxStream/Archive.h>

#include <agx/String.h>
#include <agx/Name.h>
#include <agx/Vec3.h>
#include <agx/Vec2.h>
#include <agx/Quat.h>
#include <agx/Range.h>
#include <agx/Matrix3x3.h>
#include <agx/SPDMatrix3x3.h>
#include <agxStream/Serializable.h>
#include <agxStream/RestoreListener.h>
#include <agx/Uuid.h>
#include <sstream>

namespace agxStream
{
  class Serializable;
  class StorageAgent;

  /**
  Class for reading a binary stream of serialized data.
  */
  class AGXCORE_EXPORT InputArchive : public Archive
  {

  public:
    typedef agx::Event3<agx::UInt32, Serializable *, const std::string&> RestoreEvent;
    RestoreEvent restoreEvent;


    typedef agx::VectorPOD< Serializable* > IdToObjectVector;
    typedef agx::Vector< std::string > IdToClassNameVector;
    typedef agx::Vector< StorageAgent* > IdToStorageAgentVector;

  public:
    InputArchive(std::istream& inputStream);
    InputArchive(std::istream& inputStream, IdToObjectVector& objectCache, IdToClassNameVector& nameCache);

    void open();

    /**
    Add a listener that will be called for each restored Serializable object
    \return true if successfully added. False if listener already is added.
    */
    bool addRestoreListener(RestoreListener *listener);

    /**
    Remove a RestoreListener
    \return true if listener was located and removed
    */
    bool removeRestoreListener(RestoreListener *listener);

    /// Return a reference to the inputstream
    std::istream& getStream();

    /// \return the string written for application
    agx::String getAppName() const;

    /// \return the generation version specified during writing
    agx::UInt8 getGenerationVersion() const;

    /// \return the major version specified during writing
    agx::UInt8 getMajorVersion() const;

    /// \return the minor version specified during writing
    agx::UInt8 getMinorVersion() const;

    /// \return the patch version specified during writing
    agx::UInt8 getPatchVersion() const;

    /// \return the revision specified during writing, "" if no revision was written/available
    agx::String getRevision() const;

    /// \return the build date of agx which wrote the archive
    agx::String getBuildDate() const;

    /// \return the build time of agx which wrote the archive
    agx::String getBuildTime() const;

    /// \return the date/time in ISO format YYYY-MM-DDTHH:MM:SS when the archive was written
    agx::String getArchiveDate() const;


    agx::UInt16 getSerializationVersion() const;

    typedef agx::HashTable<agx::UInt, bool> BuildConfigurationTable;
    /// \return a table with various flags which was used to build the agx version used at writing this archive.
    const BuildConfigurationTable& getBuildFlags() const;

    /// \return the version of the restored data as a size_t for version comparison using AGX_CALC_VERSION
    size_t getVersion() const;

    virtual void read(char& val);
    virtual void read(bool& val);
    virtual void read(agx::Int8& val);
    virtual void read(agx::UInt8& val);
    virtual void read(agx::Int16& val);
    virtual void read(agx::UInt16& val);

    virtual void read(agx::UInt32& val);
    virtual void read(agx::Int32& val);

    virtual void read(std::string& val);
    virtual void read(agx::UInt64& val);
    virtual void read(agx::Int64& val);
    virtual void read(float& val);
    virtual void read(double& val);

#ifdef __APPLE__
    virtual void read( size_t& val );
#endif

    virtual void read(agx::RangeReal& val);
    virtual void read(agx::Vec2f& val);
    virtual void read(agx::Vec2d& val);
    virtual void read(agx::Vec2i32& val);
    virtual void read(agx::Vec2i64& val);

    virtual void read(agx::Vec3f& val);
    virtual void read(agx::Vec3d& val);
    virtual void read(agx::Vec3i32& val);
    virtual void read(agx::Vec3i64& val);
    virtual void read(agx::Vec4f& val);
    virtual void read(agx::Vec4d& val);

    virtual void read(agx::Quat& val);
    virtual void read(agx::SPDMatrix3x3& val);
    virtual void read(agx::Matrix3x3d& val);
    virtual void read(agx::Matrix3x3f& val);
    virtual void read(agx::AffineMatrix4x4f& val);
    virtual void read(agx::AffineMatrix4x4d& val);
    virtual void read(agx::Uuid& val);

    /**
    This method will read raw data from the archive.
    Notice that you should always have called beginSection() prior to calling this method.
    Otherwise it is not compatible with XML input

    \param buf - Pointer to the buffer where data will be written
    \param len - Number of bytes to read
    */
    virtual void read(void* buf, size_t len);


    virtual void read(agx::String& val);

    virtual void read(agx::Name& val);

    template<typename T>
    void readRef(T** obj, StorageMode mode)
    {
      if (mode == STORE_BY_REF) {

        Serializable* sObj;

        sObj = readObjectByRef();

        *obj = dynamic_cast<T*>(sObj);
        if (sObj != 0 && *obj == 0) {
          agx::String str = agx::String::format("Type mismatch: read type \'%s\' Expected \'%s\'",
            sObj->getClassName(),
            T::getConstructClassId());

          agxThrow ArchiveException(str);
        }
      }
      else
      {
        agxThrow ArchiveException("Invalid storage mode specified for read");
      }
    }

    void readVal(Serializable& obj, StorageMode mode);

    Serializable* readObjectByRef();
    bool readObjectByVal(Serializable& obj);

    /**
    Must be called when archive is de-serialized successfully.
    If not, all dynamically allocated objects will be deleted.
    */
    void setFinished();

    /// \return the number of objects read so far
    int getNumObjectsRead() const;

    /// \return true if the input archive matches the current platforms endian
    bool getEndianMatch() const;

    enum RestoreMode
    {
      TRY_BY_VALUE,     ///<! Restore by value, even if its a pointer that is given to the serialization. If pointer==nullptr, restore by reference.
      TRY_BY_REFERENCE, ///<! Restore by reference when possible. Even if pointer given is != nullptr, use BY_REFERENCE
      DEFAULT = TRY_BY_REFERENCE
    };

    void setRestoreMode(RestoreMode mode);
    RestoreMode getRestoreMode() const;

    virtual void beginSection(const char* title) override;
    virtual void endSection(const char* title) override;

    virtual void getAttribute(const char *name, double& value);
    virtual void getAttribute(const char *name, float& value);
    virtual void getAttribute(const char *name, bool& value);
    virtual void getAttribute(const char *name, std::string& value);
    virtual void getAttribute(const char *name, char& value);
    virtual void getAttribute(const char *name, agx::UInt16& value);
    virtual void getAttribute(const char *name, agx::UInt64& value);
    virtual void getAttribute(const char *name, agx::UInt32& value);
    virtual void getAttribute(const char *name, agx::Int32& value);
    virtual void getAttribute(const char *name, agx::Int8& value);
    virtual void getAttribute(const char *name, agx::UInt8& value);
    virtual void getAttribute(const char *name, agx::Uuid& value);

#ifdef __APPLE__
    virtual void getAttribute(const char *name, size_t& value);
#endif

    /**
    \return true if archive contains a specified modification
    */
    bool hasModification(const char *str) const;

    /**
    \return true if all \p modifications is represented in the current archive
    */
    bool hasModifications(const agx::StringVector& modifications) const;


    /// \return true if eof is reached during reading
    bool eof() const;

    /**
    Read an array of data. Only works for agx::VectorPOD<> types.
    \param name - name of the vector data
    \param val - Target of the read vector data.
    */
    template<typename VECTOR_TYPE>
    AGXCORE_EXPORT void read_vector(const agx::String& name, VECTOR_TYPE& val);


  protected:
    /// Destructor
    virtual ~InputArchive();

    void  readCompareBitSize();

    virtual void readArchiveHeader();
    virtual void readObjectHeader(StorageMode& storageMode, StorageAgent*& agent);
    virtual void readTypeHeader(StorageAgent*& agent);


    virtual size_t getNumBytesRead() const;

  protected:
    friend class Serializable;
    friend class StorageAgent;

    /// Remove the serialized object from list
    bool removeObject(Serializable *object);


    std::istream& m_inputStream;
    IdToObjectVector m_defaultObjectCache;
    IdToObjectVector& m_IdToObjectVector;
    IdToClassNameVector m_defaultNameCache;
    IdToClassNameVector& m_idToClassNameVector;
    IdToStorageAgentVector m_IdToStorageAgentVector;

    // copy constructor and assignment operator prohibited
    InputArchive(const InputArchive& inputStream);
    InputArchive& operator=(const InputArchive&);

    typedef agx::HashTable<RestoreListener *, RestoreListenerRef> RestoreListenerHash;
    RestoreListenerHash m_restoreListeners;
  protected:

    /**
    \return a string with AGX version and Archive version used for error message reporting.
    */
    agx::String getVersionStringMessage() const;

    /**
    \return a string with AGX serialization version and Archive serialization version used for error message reporting.
    */
    agx::String getSerializationVersionStringMessage() const;

    void readHeaderData();
    void eof(bool flag);
  private:

    template<typename T>
    void readCorrectNumBytes(T& val);

    /// \return a string of the current section stack for error message/debugging purposes
    agx::String getSectionStackString() const;

    virtual agx::String getErrorMessage(const agx::String& message);


    BuildConfigurationTable m_buildFlags;


    agx::String m_appName;
    agx::UInt8 m_generationVersion, m_majorVersion, m_minorVersion, m_patchVersion;
    agx::String m_revision;
    agx::UInt16 m_serialization_version;

    size_t m_numBytesRead;
    int m_numObjectsRead; // for diagnostic purposes

    bool m_finished;
    char m_endianMatch;
    RestoreMode m_restoreMode;
    bool m_atEof;
    agx::String m_buildDate;
    agx::String m_buildTime;
    agx::Vector<agx::String> m_sectionStack;
    agx::String m_archiveDate;

    void readModificationData();
    agx::HashSet<agx::String> m_modificationHash;
  };

  typedef agx::ref_ptr<InputArchive> InputArchiveRef;


  inline bool InputArchive::eof() const
  {
    return m_atEof;
  }
  inline void InputArchive::eof(bool flag)
  {
    m_atEof = flag;
  }

  template <typename T>
  struct InputVal {
    InputVal(const char *name, T& obj) : m_name(name), m_obj(obj), m_mode(STORE_BY_VALUE) { m_objPtr = &m_obj; }
    InputVal(const char *name, T* obj) : m_name(name), m_obj(*obj), m_objPtr(obj), m_mode(STORE_BY_REF) {}

    const char *m_name;
    T & m_obj;
    T *m_objPtr;

    StorageMode m_mode;
    InputVal & operator=(const InputVal& other) { m_obj = other.m_obj; m_name = other.m_name; m_mode = other.m_mode; return *this; }
  };

  template <typename T>
  struct InputRef {
    InputRef(const char *name, T& obj) : m_name(name), m_obj(obj), m_mode(STORE_BY_REF) {}

    const char *m_name;
    T & m_obj;

    StorageMode m_mode;
    InputRef & operator=(const InputRef& other) { m_obj = other.m_obj; m_name = other.m_name; m_mode = other.m_mode; return *this; }
  };


  /**
  Create an object with a name and a reference to the object that should be restored (usually a pointer).
  This should be used for restoring pointers to Serializable objects.

  \param name - name of the object requested to be restored.
  \param obj - reference to the object to be restored
  \return an object containing the reference to an object that will be restored (usually a pointer).
  */
  template <typename T>
  InputRef<T> in(const char *name, T& obj) { return InputRef<T>(name, obj); }

  /**
  Create an object with a name and a reference to the object that should be restored.
  This should be used for restoring by value. For double, float, Vec3 and references to Serializable objects.

  \param name - name of the object requested to be restored.
  \param obj - reference to the object to be restored
  \return an object containing the reference to an object that will be restored by value.
  */
  template <typename T>
  InputVal<T> in_val(const char *name, T& obj) { return InputVal<T>(name, obj); }


#define DECLARE_INPUT_BASIC_TYPE(TYPE) \
  AGXCORE_EXPORT inline agxStream::InputArchive& operator>>( InputArchive& in, InputVal<TYPE> val ) \
  { \
  in.beginSection(val.m_name);  \
  in.read( val.m_obj ); \
  in.endSection(val.m_name); \
  return in;\
  } \
  inline InputVal<TYPE> in(const char *name, TYPE& obj) { return InputVal<TYPE>(name, obj); }

  DECLARE_INPUT_BASIC_TYPE(std::string)
  DECLARE_INPUT_BASIC_TYPE(agx::String)
  DECLARE_INPUT_BASIC_TYPE(agx::Name)
  DECLARE_INPUT_BASIC_TYPE(bool)
  DECLARE_INPUT_BASIC_TYPE(char)
  DECLARE_INPUT_BASIC_TYPE(agx::Int8)
  DECLARE_INPUT_BASIC_TYPE(agx::UInt8)
  DECLARE_INPUT_BASIC_TYPE(agx::Int16)
  DECLARE_INPUT_BASIC_TYPE(agx::UInt16)
  DECLARE_INPUT_BASIC_TYPE(agx::Int32)
  DECLARE_INPUT_BASIC_TYPE(agx::UInt32)
  DECLARE_INPUT_BASIC_TYPE(agx::Int64)
  DECLARE_INPUT_BASIC_TYPE(agx::UInt64)
#ifdef __APPLE__
  DECLARE_INPUT_BASIC_TYPE(size_t)
#endif

  DECLARE_INPUT_BASIC_TYPE(float)
  DECLARE_INPUT_BASIC_TYPE(double)
  DECLARE_INPUT_BASIC_TYPE(agx::RangeReal)
  DECLARE_INPUT_BASIC_TYPE(agx::Vec2f)
  DECLARE_INPUT_BASIC_TYPE(agx::Vec2d)
  DECLARE_INPUT_BASIC_TYPE(agx::Vec2i32)
  DECLARE_INPUT_BASIC_TYPE(agx::Vec2i64)
  DECLARE_INPUT_BASIC_TYPE(agx::Vec3f)
  DECLARE_INPUT_BASIC_TYPE(agx::Vec3d)
  DECLARE_INPUT_BASIC_TYPE(agx::Vec3i32)
  DECLARE_INPUT_BASIC_TYPE(agx::Vec3i64)
  DECLARE_INPUT_BASIC_TYPE(agx::Vec4f)
  DECLARE_INPUT_BASIC_TYPE(agx::Vec4d)
  DECLARE_INPUT_BASIC_TYPE(agx::Quat)
  DECLARE_INPUT_BASIC_TYPE(agx::SPDMatrix3x3)
  DECLARE_INPUT_BASIC_TYPE(agx::AffineMatrix4x4)
  DECLARE_INPUT_BASIC_TYPE(agx::Matrix3x3f)
  DECLARE_INPUT_BASIC_TYPE(agx::Matrix3x3d)
  DECLARE_INPUT_BASIC_TYPE(agx::Uuid)

  template <typename T>
  inline InputArchive& operator>>(InputArchive& in, InputVal<T> val)
  {
    in.beginSection(val.m_name);
    in.readVal(val.m_obj, val.m_mode);
    in.endSection(val.m_name);
    return in;
  }
  template <typename T>
  inline InputArchive& operator>>(InputArchive& in, InputRef<T> val)
  {
    in.beginSection(val.m_name);
    in.readRef(&(val.m_obj), val.m_mode);
    in.endSection(val.m_name);
    return in;
  }



  template<typename T>
  void restoreRefPtr(agxStream::InputArchive& in, const char* name, agx::ref_ptr<T>& refPtr)
  {
    T* object = nullptr;
    in >> agxStream::in(name, object);
    refPtr = object;
  }

  template<typename T>
  void restoreObserverPtr(agxStream::InputArchive& in, const char* name, agx::observer_ptr<T>& observerPtr)
  {
    T* object = nullptr;
    in >> agxStream::in(name, object);
    observerPtr = object;
  }

}

#ifdef _MSC_VER
# pragma warning(pop)
#endif

#endif
