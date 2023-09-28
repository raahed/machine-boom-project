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
#include <agx/HashTable.h>
#include <agx/Name.h>

#include <agxStream/Archive.h>
#include <agx/String.h>
#include <agx/Vec2.h>
#include <agx/Vec3.h>
#include <agx/Quat.h>
#include <agx/Range.h>
#include <agx/Matrix3x3.h>
#include <agx/SPDMatrix3x3.h>
#include <agx/Uuid.h>

namespace agx
{
  class Uuid;
}

namespace agxStream
{

  class StorageAgent;
  class Serializable;



  /**
  Class for writing serialized data in binary format to a stream.

  \todo If multiple objects of same pointer address are serialized they are considered to be instances.
  One example where this fails is: Ptr *p1 = new Ptr; archive << p1; delete p1; Ptr p2 = new Ptr; archive << p2;
  These are two separated objects, but if p1 == p2 it will fail due to that they have the same address.
  */
  class AGXCORE_EXPORT OutputArchive : public Archive
  {
  public:
      typedef agx::Event2<const Serializable *, agx::UInt32> StoreEvent;
      StoreEvent storeEvent;

      struct IdEntry {
        IdEntry(agx::UInt32 _id = agx::InvalidIndex) : id(_id), valid(true) {}
        agx::UInt32 id;
        bool valid;
      };

      typedef agx::HashTable< const Serializable*,  IdEntry > ObjectToIdMap;
      typedef agx::HashTable< StorageAgent*,  agx::UInt32 > StorageAgentToIdMap;

      /**
      This struct is used to handle id-collisions when using incremental journal. Using
      the size of the table as the next id does not work since addresses are deleted and reallocated
      during restore. Having a separate idCounter for each table allows us to handle the issue.
      */
      struct ObjectToIdMapContext {
        agx::UInt32 idCounter;
        ObjectToIdMap table;

        ObjectToIdMapContext() : idCounter(0) {}
      };

    public:

    /**
    Create and associate an OutputArchive for writing to an open stream
    */
    OutputArchive( std::ostream& outputStream );
    OutputArchive( std::ostream& outputStream, ObjectToIdMapContext& cache );

    /// Return a reference to the output stream
    std::ostream& getStream() { return m_outputStream; }

    /**
    Open a new stream and write an archive header with some version information
    */
    void open(const std::string& appName);

    /**
    \return number of objects stored so far
    */
    int getNumObjectsWritten() const {
      return m_numObjectsWritten;
    }

    /**
    \return number of objects skipped in writing due to that they have setEnableSerialization(false)
    */
    int getNumObjectsSkipped() const {
      return m_numObjectsSkipped;
    }

    /**
    Specify if the current date (and time) at execution time should be used when creating archive,
    or if the build date should be used. Default is enabled.
    */
    static void setUseCurrentDate(bool flag);

    /**
    Return if the current date (and time) at execution time should be used when creating archive,
    or if the build date should be used.
    */
    bool getUseCurrentDate();

    virtual void write(char val);
    virtual void write(bool val);
    virtual void write(agx::UInt8 val);
    virtual void write(agx::Int8 val);

    virtual void write(agx::Int16 val);
    virtual void write(agx::UInt16 val);

    virtual void write(agx::Int32 val);
    virtual void write(agx::UInt32 val);

    virtual void write(agx::Int64  val);
    virtual void write(agx::UInt64 val);
#ifdef __APPLE__
    virtual void write(size_t val) { write((agx::UInt64)val); }
#endif


    virtual void write(float  val);
    virtual void write(double val);

    virtual void write(const std::string& val);
    virtual void write(const agx::String& val);
    virtual void write(const agx::Name& val)
    {
      agx::String tmp = val;
      write(tmp);
    }
    virtual void write(const agx::Uuid& val);


    void write(const agx::RangeReal& val);
    void write(const agx::Vec2f& val);
    void write(const agx::Vec2d& val);
    void write(const agx::Vec2i32& val);
    void write(const agx::Vec2i64& val);

    void write(const agx::Vec3f& val);
    void write(const agx::Vec3d& val);
    void write(const agx::Vec3i32& val);
    void write(const agx::Vec3i64& val);
    void write(const agx::Vec4f& val);
    void write(const agx::Vec4d& val);
    void write(const agx::Quat& val);
    void write(const agx::SPDMatrix3x3& val);
    void write(const agx::Matrix3x3d& val);
    void write(const agx::Matrix3x3f& val);
    void write(const agx::AffineMatrix4x4f& val);
    void write(const agx::AffineMatrix4x4d& val);

    void write(const Serializable *serializable, StorageMode mode);
    void write(const Serializable &serializable, StorageMode mode);


    virtual void addAttribute(const char *name, bool value);
    virtual void addAttribute(const char *name, const char* value);
    virtual void addAttribute(const char *name, char value);
    virtual void addAttribute(const char *name, agx::Int8 value);
    virtual void addAttribute(const char *name, agx::UInt8 value);
    virtual void addAttribute(const char *name, agx::UInt16 value);
    virtual void addAttribute(const char *name, agx::Int32 value);
    virtual void addAttribute(const char *name, agx::UInt32 value);
    virtual void addAttribute(const char *name, agx::UInt64 value);
    virtual void addAttribute(const char *name, float value);
    virtual void addAttribute(const char *name, double value);
    virtual void addAttribute(const char *name, const agx::Uuid& value);

#ifdef __APPLE__
    virtual void addAttribute(const char *name, size_t value) { addAttribute(name, (agx::UInt64)value); }
#endif

    bool getEndianMatch() const {
      return m_endianMatch == 1;
    }

    virtual void beginSection(const char *) override {}
    virtual void endSection(const char *) override {}

    virtual void writeObjectByRef(const Serializable* obj);
    virtual void writeObjectByVal(const Serializable& obj);

    /**
    This method will write raw data into the archive.
    Notice that you should always have called beginSection() prior to calling this method.
    Otherwise it is not compatible with XML output.

    \param buf - Pointer to the data to be written
    \param len - Number of bytes to write
    */
    virtual void write(const void* buf, size_t len);

    /// Make sure all data in buffer is written to associated stream
    void flush();

    /// Call flush
    void close() override;

    /**
    Specify a new buffer size for the write buffer. Default = 8192 bytes
    A call to this method will flush the buffer to the stream if \p bufferSize is less than the current size
    */
    void setWriteBufferSize(agx::UInt32 bufferSize);

    /// \return current write buffer
    agx::UInt32 getWriteBufferSize() const;

    /**
    Add a modification string that will be stored in the archive.
    This method need to be called before the open method (because thats when the header is written).
    \return false if modification already exists
    */
    bool addModification( const char *str);

    /**
    \returns true if this archive has the specified \p modification hash
    */
    bool hasModification(const agx::String& modification) const;

    /**
    Write an array of data. Only works for agx::VectorPOD<> types.
    \param name - name of the vector data
    \param val - The array data to be written.
    */
    template<typename VECTOR_TYPE>
    AGXCORE_EXPORT void write_vector(const agx::String& name, const VECTOR_TYPE& val);

  protected:

    // Need this to verify the modifications
    friend class InputArchive;

    // This constructor is only used by InputArchive during modification verification
    OutputArchive();

    /// Destructor
    virtual ~OutputArchive();

    virtual void writeHeaderData(agx::UInt32 magic,
      const std::string& appName,
      agx::UInt8 generationVersion,
      agx::UInt8 majorVersion,
      agx::UInt8 minorVersion,
      agx::UInt8 patchVersion,
      const std::string& revision,
      agx::UInt16 serializationVersion);

    virtual void writeArchiveHeader(const std::string& appName,
      agx::UInt8 generationVersion,
      agx::UInt8 majorVersion,
      agx::UInt8 minorVersion,
      agx::UInt8 patchVersion,
      const std::string& revision,
      agx::UInt16 serializationVersion);

    virtual  void writeObjectHeader(StorageMode mode, StorageAgent* agent);
    virtual void writeTypeHeader(StorageAgent* agent);


  protected:
    void  writeBitSize();
    std::ostream& m_outputStream;
    ObjectToIdMapContext m_defaultCache;
    ObjectToIdMapContext& m_objectToIdMap;
    StorageAgentToIdMap m_storageAgentToIdMap;


    int m_numObjectsWritten; // for diagnostic purposes
    int m_numObjectsSkipped;

    // copy constructor and assignment operator prohibited
    OutputArchive(const OutputArchive&);
    OutputArchive& operator=(const OutputArchive&);

  private:
    char m_endianMatch;
    agx::Vector<char> m_buffer;
    agx::UInt32 m_bufferPosition;
    agx::UInt32 m_bufferSize;

    void addModifications();
    void writeModificationData();
    bool m_hasAddedModifications;
    static bool s_useCurrentDate;

    agx::HashSet<agx::String> m_modificationHash;
  };



  template <typename T>
  struct OutputRef
  {
    OutputRef(const char *name, const T& obj, StorageMode mode) :
      m_name(name),
      m_obj(&obj),
      m_mode(mode)
    {
    }

    OutputRef(const char *name, const T* obj, StorageMode mode) :
      m_name(name),
      m_obj(obj),
      m_mode(mode)
    {
    }

    OutputRef& operator=(const OutputRef& other)
    {
      m_obj = other.m_obj;
      m_name = other.m_name;
      m_mode = other.m_mode;
      return *this;
    }

    const char *m_name;
    const T *m_obj;
    StorageMode m_mode;
  };
  typedef agx::ref_ptr<OutputArchive> OutputArchiveRef;



  template <typename T>
  struct OutputVal {
    OutputVal(const char *name, const T obj) : m_name(name), m_obj(obj), m_mode(STORE_BY_VALUE) {}

    const char *m_name;
    T m_obj;

    StorageMode m_mode;
    OutputVal & operator=(const OutputVal& other) { m_obj = other.m_obj; m_name = other.m_name; m_mode = other.m_mode; return *this; }
  };


#define DECLARE_OUTPUT_BASIC_TYPE(TYPE) \
  AGXCORE_EXPORT inline agxStream::OutputArchive& operator<<( OutputArchive& out, OutputVal<TYPE> val ) \
  { \
  out.beginSection(val.m_name);  \
  out.write( val.m_obj ); \
  out.endSection(val.m_name); \
  return out;\
  }\
  AGXCORE_EXPORT inline OutputVal<TYPE> out(const char *name, TYPE obj) { return OutputVal<TYPE>(name, obj); }

  DECLARE_OUTPUT_BASIC_TYPE(std::string)
  DECLARE_OUTPUT_BASIC_TYPE(agx::String)
  DECLARE_OUTPUT_BASIC_TYPE(agx::Name)
  DECLARE_OUTPUT_BASIC_TYPE(bool)
  DECLARE_OUTPUT_BASIC_TYPE(char)
  DECLARE_OUTPUT_BASIC_TYPE(agx::Int8)
  DECLARE_OUTPUT_BASIC_TYPE(agx::UInt8)
  DECLARE_OUTPUT_BASIC_TYPE(agx::Int16)
  DECLARE_OUTPUT_BASIC_TYPE(agx::UInt16)
  DECLARE_OUTPUT_BASIC_TYPE(agx::Int32)
  DECLARE_OUTPUT_BASIC_TYPE(agx::UInt32)
  DECLARE_OUTPUT_BASIC_TYPE(agx::Int64)
  DECLARE_OUTPUT_BASIC_TYPE(agx::UInt64)
#ifdef __APPLE__
  DECLARE_OUTPUT_BASIC_TYPE(size_t)
#endif
  DECLARE_OUTPUT_BASIC_TYPE(float)
  DECLARE_OUTPUT_BASIC_TYPE(double)
  DECLARE_OUTPUT_BASIC_TYPE(agx::RangeReal)
  DECLARE_OUTPUT_BASIC_TYPE(agx::Vec2f)
  DECLARE_OUTPUT_BASIC_TYPE(agx::Vec2d)
  DECLARE_OUTPUT_BASIC_TYPE(agx::Vec2i32)
  DECLARE_OUTPUT_BASIC_TYPE(agx::Vec2i64)
  DECLARE_OUTPUT_BASIC_TYPE(agx::Vec3f)
  DECLARE_OUTPUT_BASIC_TYPE(agx::Vec3d)
  DECLARE_OUTPUT_BASIC_TYPE(agx::Vec3i32)
  DECLARE_OUTPUT_BASIC_TYPE(agx::Vec3i64)
  DECLARE_OUTPUT_BASIC_TYPE(agx::Vec4f)
  DECLARE_OUTPUT_BASIC_TYPE(agx::Vec4d)
  DECLARE_OUTPUT_BASIC_TYPE(agx::Quat)
  DECLARE_OUTPUT_BASIC_TYPE(agx::SPDMatrix3x3)
  DECLARE_OUTPUT_BASIC_TYPE(agx::AffineMatrix4x4)
  DECLARE_OUTPUT_BASIC_TYPE(agx::Matrix3x3f)
  DECLARE_OUTPUT_BASIC_TYPE(agx::Matrix3x3d)
  DECLARE_OUTPUT_BASIC_TYPE(agx::Uuid)


  /**
  Return an object that contain the name and the object that should be serialized to an archive.
  This should be used for pointers to Serializable objects.
  \param name - name of the object/data
  \param obj - Pointer to a Serializable object
  */
  inline OutputRef<Serializable> out(const char *name, const Serializable* obj)
  {
    return OutputRef<Serializable>(name, obj, STORE_BY_REF);
  }

  /**
  Return an object that contain the name and the object that should be serialized to an archive.
  This should be used for serializing objects by value.
  \param name - name of the object/data
  \param obj - Reference to a Serializable object or a basic type (double, Vec3, RangeReal etc.)
  */
  template <typename T>
  OutputRef<T> out_val(const char *name, const T& obj) { return OutputRef<T>(name, obj, STORE_BY_VALUE); }


  template <typename T>
  inline OutputArchive& operator<<(OutputArchive& out, OutputRef<T> val)
  {
    out.beginSection(val.m_name);
    out.write(val.m_obj, val.m_mode);
    out.endSection(val.m_name);
    return out;
  }

  template <typename T>
  inline OutputArchive& operator<<(OutputArchive& out, OutputVal<T> val)
  {
    out.beginSection(val.m_name);
    out.write(val.m_obj);
    out.endSection(val.m_name);
    return out;
  }
}

