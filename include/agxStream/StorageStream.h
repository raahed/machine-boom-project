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

#include <agx/agxPhysics_export.h>
#include <iostream>
#include <agx/Vec3.h>
#include <agx/AffineMatrix4x4.h>
#include <stack>
#include <memory>

namespace agx
{
  class RigidBody;
}

namespace agxWire
{
  class Node;
}

namespace agxCollide
{
  class Geometry;
}

namespace agxStream
{
  /**
  Abstract base class for storing/restoring a line/drums with version control.
  A StorageControl can either be used for WRITING data to, or READING data from, not both.
  The argument to the constructor will define if this is an input or an output StorageStream.
  */
  class AGXPHYSICS_EXPORT StorageStream
  {
  public:

    enum Mode
    {
      RESTORE,
      STORE
    };

    enum FloatMode
    {
      AS_DOUBLE = 0,
      AS_FLOAT = 1
    };

    /**
    \param mode - specifies if the storage stream is for STORE or RESTORE
    \param useBlocks - if true, block sizes will be written to enable skipBlock over blocks. This will add overhead in the STORE process.
    If false (default), a zero value for the block sizes will be written and skipping will not be possible.
    */
    StorageStream(Mode mode, bool useBlocks=false);

    /// Destructor
    virtual ~StorageStream();

    /**
    Read data from the buffer into the stream for restore
    */
    bool read(const agx::Vector<char>& buffer);

    /**
    Write data from the stream into the buffer for external storage
    */
    bool write(agx::Vector<char>& buffer);

    /**
    Re-initialize the stream so it has no data stored, just as it was when it was created initially.
    */
    void clear();

    enum Status
    {
      SUCCESS,
      MISSING_BODY,
      MISSING_GEOMETRY,
      INVALID_MD5,
      INVALID_HEADER,
      ERROR_WRITING_TO_OUTSTREAM,
      ERROR_READING_FROM_INSTREAM,
      ERROR_WRITING_TO_DATASTREAM,
      ERROR_READING_FROM_DATASTREAM,
      MIXING_MODES
    };

    /// \return version conglomerated into one agx::UInt32 for easier comparison using AGX_CALC_VERSION()
    agx::UInt32 getVersion() const;

    /// \return serialization version stored in the storage stream
    agx::UInt16 getSerializationVersion() const;

    /// \return Storage stream version stored in the storage stream. Introduced in 2.19.0.0.
    agx::UInt16 getStorageStreamVersion() const;

    ///\return the current status of this Storage
    Status getStatus() const { return m_status; }
#ifndef SWIG
    /**
    Called when we need a valid pointer to a body. The derived class
    should set the *body pointer to a valid one.

    \param body - Pointer to a rigidbody pointer which should be set to a valid rigidbody
    \return false if something went wrong in the read method, indicates that restoring body failed.
    */
    virtual bool read(agx::RigidBody** body);

    /**
    Called when we want to write a rigid body.
    The derived class should store this pointer/id/values somewhere so that we can later call read and get a valid
    pointer back.

    \param body - the body that is written/stored
    \return false indicates that something went wrong when storing the body.
    */
    virtual bool write(agx::RigidBody* body);

    /**
    Called when we need a valid pointer to a geometry. The derived class
    should set the *geometry pointer to a valid one.

    \param geometry - Pointer to a geometry pointer which should be set to a valid geometry
    \return false if something went wrong in the read method, indicates that restoring geometry failed.
    */
    virtual bool read(agxCollide::Geometry** geometry);

    /**
    Called when we want to write a geometry.
    The derived class should store this pointer/id/values somewhere so that we can later call read and get a valid
    pointer back.

    \param geometry - the geometry that is written/stored
    \return false indicates that something went wrong when storing the geometry.
    */
    virtual bool write(agxCollide::Geometry* geometry);

    /**
    Called with a valid pointer to a line node. The derived class can assign user data etc during this call.
    \param node - valid node to assign any user specified data
    \return true if successful - otherwise false
    */
    virtual bool read(agxWire::Node* node);

    /**
    Called when we want to write a LineNode.
    The derived class should store this pointer/id/values somewhere so that we can later call read and get a valid
    pointer back.

    \param node - the LineNode that is written/stored
    \return false indicates that something went wrong when storing the node.
    */
    virtual bool write(agxWire::Node* node);

    /**
    Method that will write this object to an output stream, derive your own if you want to
    write any local data in your derived class.
    */
    virtual std::ostream& write(std::ostream& str);

    /**
    Method that will read data from a stream into this object.Derive your own if you want to
    read any local data into your derived class.
    */
    virtual std::istream& read(std::istream& str);

    inline Mode getMode() const;

    AGXPHYSICS_EXPORT friend std::ostream& operator<<(std::ostream& str, StorageStream& storage);
    AGXPHYSICS_EXPORT friend std::istream& operator>>(std::istream& str, StorageStream& storage);

    // --------- Restoring/reading
    virtual void read(agx::UInt8& val);
    virtual void read(agx::UInt16& val);
    virtual void read(agx::UInt64& val);
    virtual void read(agx::Int32& val);

    virtual void read(char& val);
    virtual void read(bool& val);

    /// \p readAsFloat there to make this method compatible with write( double, bool ), it will be read as float, no matter what
    virtual void read(float& val, bool readAsFloat = false);
    /// If \p readAsFloat==true, then this variable will be read from the stream as a float
    virtual void read(double& val, bool readAsFloat = false);

    virtual void read(std::string& val);

    /// If \p readAsFloat==true, then this Vec3 will be read from the stream as a 3 floats
    virtual void read(agx::Vec3d& val, bool readAsFloat = false);

    /// \p readAsFloat there to make this method compatible with write( double, bool ), it will always be read as a float
    virtual void read(agx::Vec3f& val, bool readAsFloat = false);

    /// If \p readAsFloat==true, then this Vec3 will be read from the stream as a 3 floats
    virtual void read(agx::Quat64& val, bool readAsFloat = false);

    /// \p readAsFloat there to make this method compatible with write( double, bool ), it will always be read as a float
    virtual void read(agx::Quat32& val, bool readAsFloat = false);

    /// If \p readAsFloat==true, then this matrix will be read from the stream as a 16 floats
    virtual void read(agx::AffineMatrix4x4d& val, bool readAsFloat = false);

    /// \p readAsFloat there to make this method compatible with write( double, bool ), it will always be read as a float
    virtual void read(agx::AffineMatrix4x4f& val, bool readAsFloat = false);

    // -------- Storing/writing
    virtual void write(agx::UInt8 val);
    virtual void write(agx::UInt16 val);
    virtual void write(agx::UInt64 val);
    virtual void write(agx::Int32  val);

    virtual void write(char val);
    virtual void write(bool val);


    /// \p writeAsFloat there to make this method compatible with write( double, bool ), it WILL be written as float no matter what
    virtual void write(float  val, FloatMode writeAsFloat = AS_FLOAT);

    /// If \p writeAsFloat==true, then this variable will be written to the stream as a float
    virtual void write(double val, FloatMode writeAsFloat = AS_DOUBLE);

    virtual void write(const std::string& val);

    /// If \p writeAsFloat==true, then this vector will be written to the stream as 3*float
    virtual void write(const agx::Vec3d& val, FloatMode writeAsFloat = AS_DOUBLE);

    /// \p writeAsFloat there to make this method compatible with write( double, bool ), it WILL be written as float no matter what
    virtual void write(const agx::Vec3f& val, FloatMode writeAsFloat = AS_DOUBLE);

    /// If \p writeAsFloat==true, then this vector will be written to the stream as 3*float
    virtual void write(const agx::Quat64& val, FloatMode writeAsFloat = AS_DOUBLE);

    /// \p writeAsFloat there to make this method compatible with write( double, bool ), it WILL be written as float no matter what
    virtual void write(const agx::Quat32& val, FloatMode writeAsFloat = AS_DOUBLE);

    /// If \p writeAsFloat==true, then this Matrix will be written to the stream as 12*float
    virtual void write(const agx::AffineMatrix4x4d& val, FloatMode writeAsFloat = AS_DOUBLE);

    /// \p writeAsFloat there to make this method compatible with write( double, bool ), it WILL be written as float no matter what
    virtual void write(const agx::AffineMatrix4x4f& val, FloatMode writeAsFloat = AS_DOUBLE);

    /**
    Begin a new logical block within the stream.
    Each call to beginBlock should be matched with a corresponding call to
    \p endBlock with the same name. Blocks may be nested.

    \note Read/write errors may leave the block structure in an inconsistent state.
    */
    void beginBlock(const char* blockName);

    /**
    End the block started by the corresponding \p beginBlock.
    \return True if the stream for the ended block is good. False otherwise.
    */
    bool endBlock(const char* blockName);

    void skipBlock(const char* blockName);

    /// \return True if the current block stream is good.
    bool good() const;

    /**
    Ties a beginBlock/endBlock pair to a scope.
    */
    class AGXPHYSICS_EXPORT ScopedBlock
    {
      public:
        /**
        The data pointed to by 'blockName' is not copied. It is the responsibility
        of the caller to ensure that the pointer is valid for the duration of
        the ScopedBlock's lifetime.
        */
        ScopedBlock(StorageStream& stream, const char* blockName);
        ~ScopedBlock();
       private:
        ScopedBlock(const ScopedBlock&);
        ScopedBlock& operator=(const ScopedBlock&);
      private:
        StorageStream& m_stream;
        const char* m_blockName;
    };

    /**
    Enable/Disable the MD5 check of the data. Takes more time if enabled.
    \param flag - If true, MD5 check is enabled. A MD5 checksum will be written together with the data.
    */
    void setEnableMD5(bool flag) { m_checkMD5 = flag; }

    /**
    \return true if MD5 check is enabled
    */
    bool getEnableMD5() const { return m_checkMD5; }

    /// Return a direct reference to the current block stream, use with caution
    std::stringstream &getStream();
    const std::stringstream &getStream() const;

#endif /* SWIG */
  protected:

    void readHeader(std::istream& str);
    void writeHeader(std::ostream& str, const char* data);


    void checkMD5();

    void checkMode(Mode newMode);

    StorageStream(const StorageStream&) {}
    StorageStream() {}


  private:

    // Internal class for storing stack of blocks
    class AGXPHYSICS_EXPORT StorageStack
    {
    public:

      /// Constructor
      StorageStack(std::shared_ptr<std::stringstream> root);
      StorageStack() { agxAbort(); }

      /// Destructor
      virtual ~StorageStack();

      /// \return name of current block
      std::string name() const;

      /// \return the stream of the current block
      std::shared_ptr<std::stringstream> top();

      std::shared_ptr<const std::stringstream> top() const;

      /// \return the number of blocks on the stack
      uint32_t size() { return uint32_t(m_stack.size()); }

      /// Push a new block to the stack
      void push(const std::string& blockName, std::shared_ptr<std::stringstream>);

      /// Remove the last pushed block from the stack
      void pop();

      /// Remove all entries in the stack
      void clear(std::shared_ptr <std::stringstream> root);

      /// \return the number of bytes written to the current block
      uint32_t numBytes();
      std::shared_ptr<std::stringstream> getNewStream();


      void flush();

    protected:

      void writeBuffer(const char* data, size_t numBytes, std::stringstream* stream);

      typedef std::stack<std::pair<std::string, std::shared_ptr<std::stringstream> > > StreamStack;
      StreamStack m_stack;
      std::shared_ptr<std::stringstream> m_root;
      std::stack<std::shared_ptr<std::stringstream> > m_streamPool;

      agx::VectorPOD<char> m_writeBuffer;
      agx::UInt32 m_bufferPosition;
      agx::UInt32 m_bufferSize;

    };

  private:


    bool m_checkMD5;

    std::string m_md5;
    agx::UInt8 m_generation, m_major, m_minor, m_patch;
    agx::UInt16 m_serializationVersion;
    agx::UInt16 m_storageStreamVersion;

    Status m_status;
    Mode m_mode;

    std::shared_ptr<std::stringstream> m_data;
    agx::UInt32 m_dataSize;

    StorageStack m_stack;
    bool m_useBlocks;
  };

  AGXPHYSICS_EXPORT std::ostream& operator<<(std::ostream& str, StorageStream& storage);
  AGXPHYSICS_EXPORT std::istream& operator>>(std::istream& str, StorageStream& storage);

  inline bool StorageStream::read(agx::RigidBody ** /*body*/) { return false; }
  inline bool StorageStream::write(agx::RigidBody * /*body*/) { return false; }
  inline bool StorageStream::read(agxCollide::Geometry ** /*geometry*/) { return false; }
  inline bool StorageStream::write(agxCollide::Geometry * /*geometry*/) { return false; }
  inline bool StorageStream::read(agxWire::Node * /*node*/) { return false; }
  inline bool StorageStream::write(agxWire::Node * /*node*/) { return false; }


  // ------------------- Reading
  inline StorageStream& operator>>(StorageStream& in, agx::UInt8& val)
  {
    in.read(val);
    return in;
  }

  inline StorageStream& operator>>(StorageStream& in, agx::UInt16& val)
  {
    in.read(val);
    return in;
  }

  inline StorageStream& operator>>(StorageStream& in, char& val)
  {
    in.read(val);
    return in;
  }

  inline StorageStream& operator>>(StorageStream& in, bool& val)
  {
    in.read(val);
    return in;
  }

  inline StorageStream& operator>>(StorageStream& in, int& val)
  {
    int32_t readVal = 0;
    in.read(readVal);
    val = int(readVal);
    return in;
  }

  inline StorageStream& operator>>(StorageStream& in, size_t& val)
  {
    uint64_t readVal = 0;
    in.read(readVal);
    val = size_t(readVal);
    return in;
  }

#ifdef __APPLE__
  inline StorageStream& operator>>(StorageStream& in, agx::UInt& val)
  {
    uint64_t readVal = 0;
    in.read(readVal);
    val = agx::UInt(readVal);
    return in;
  }
#endif

  inline StorageStream& operator>>(StorageStream& in, float& val)
  {
    in.read(val);
    return in;
  }

  inline StorageStream& operator>>(StorageStream& in, double& val)
  {
    in.read(val);
    return in;
  }

  inline StorageStream& operator>>(StorageStream& in, std::string& val)
  {
    in.read(val);
    return in;
  }

  inline StorageStream& operator>>(StorageStream& in, agx::RigidBody** rb)
  {
    in.read(rb);
    return in;
  }

  inline StorageStream& operator>>(StorageStream& in, agxCollide::Geometry** geometry)
  {
    in.read(geometry);
    return in;
  }

  inline StorageStream& operator>>(StorageStream& in, agx::Vec3d& val)
  {
    in.read(val);
    return in;
  }

  inline StorageStream& operator>>(StorageStream& in, agx::Vec3f& val)
  {
    in.read(val);
    return in;
  }

  inline StorageStream& operator>>(StorageStream& in, agx::Quat64& q)
  {
    in.read(q);
    return in;
  }

  inline StorageStream& operator>>(StorageStream& in, agx::Quat32& q)
  {
    in.read(q);
    return in;
  }

  inline StorageStream& operator>>(StorageStream& in, agx::AffineMatrix4x4d& val)
  {
    in.read(val);
    return in;
  }

  inline StorageStream& operator>>(StorageStream& in, agx::AffineMatrix4x4f& val)
  {
    in.read(val);
    return in;
  }

  inline StorageStream& operator>>(StorageStream& in, agxWire::Node* node)
  {
    in.read(node);
    return in;
  }

  // ---------------- Writing
  inline StorageStream& operator<<(StorageStream& out, agx::UInt8 val)
  {
    out.write(val);
    return out;
  }

  inline StorageStream& operator<<(StorageStream& out, agx::UInt16 val)
  {
    out.write(val);
    return out;
  }


  inline StorageStream& operator<<(StorageStream& out, char val)
  {
    out.write(val);
    return out;
  }

  inline StorageStream& operator<<(StorageStream& out, bool val)
  {
    out.write(val);
    return out;
  }


  inline StorageStream& operator<<(StorageStream& out, int val)
  {
    out.write(int32_t(val));
    return out;
  }

  inline StorageStream& operator<<(StorageStream& out, size_t val)
  {
    out.write(uint64_t(val));
    return out;
  }

#ifdef __APPLE__
  inline StorageStream& operator<<(StorageStream& out, agx::UInt val)
  {
    out.write(uint64_t(val));
    return out;
  }
#endif

  inline StorageStream& operator<<(StorageStream& out, float val)
  {
    out.write(val);
    return out;
  }

  inline StorageStream& operator<<(StorageStream& out, double val)
  {
    out.write(val);
    return out;
  }

  inline StorageStream& operator<<(StorageStream& out, const std::string& val)
  {
    out.write(val);
    return out;
  }

  inline StorageStream& operator<<(StorageStream& out, const agx::RigidBody* rb)
  {
    out.write(const_cast< agx::RigidBody* >(rb));
    return out;
  }

  inline StorageStream& operator<<(StorageStream& out, const agxCollide::Geometry* geometry)
  {
    out.write(const_cast< agxCollide::Geometry* >(geometry));
    return out;
  }

  inline StorageStream& operator<<(StorageStream& out, const agx::AffineMatrix4x4f& v)
  {
    out.write(v);
    return out;
  }

  inline StorageStream& operator<<(StorageStream& out, const agx::AffineMatrix4x4d& v)
  {
    out.write(v);
    return out;
  }

  inline StorageStream& operator<<(StorageStream& out, const agx::Vec3d& v)
  {
    out.write(v);
    return out;
  }

  inline StorageStream& operator<<(StorageStream& out, const agx::Vec3f& v)
  {
    out.write(v);
    return out;
  }

  inline StorageStream& operator<<(StorageStream& out, const agx::Quat64& q)
  {
    out.write(q);
    return out;
  }

  inline StorageStream& operator<<(StorageStream& out, const agx::Quat32& q)
  {
    out.write(q);
    return out;
  }

  inline StorageStream& operator<<(StorageStream& out, const agxWire::Node* node)
  {
    out.write(const_cast< agxWire::Node* >(node));
    return out;
  }

  inline StorageStream::Mode StorageStream::getMode() const
  {
    return m_mode;
  }
} // namespace agxStream

