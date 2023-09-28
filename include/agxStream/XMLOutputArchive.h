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


#ifdef _MSC_VER
# pragma warning( push )
# pragma warning( disable : 4290 )
#endif

#include <agx/agxPhysics_export.h>


#include <string>
#include <sstream>

#include <agx/byteswap.h>
#include <agxStream/OutputArchive.h>

namespace agx
{
  class TiXmlNode;
  class TiXmlElement;
  class TiXmlDocument;
  class Uuid;
}


namespace agxStream
{

  /**
  XML version of an OutputArchive (for storing serialized objects).
  The data sent to the outputStream will be human readable.
  Take more storage and is slower than binary archive, but handy for debugging purposes.
  */
  class AGXCORE_EXPORT XMLOutputArchive : public OutputArchive
  {
    public:

      /**
      Constructor that creates a XML based output archive.
      The precision used for writing real values will be acquired from the stream (stream.precision()).
      However the minimum precision is 8 significant numbers.
      Data will only be written to the stream when the archive is either flushed or closed. (or at the destructor of the Archive).
      */
      XMLOutputArchive( std::ostream& outputStream ) ;
      XMLOutputArchive( std::ostream& outputStream, ObjectToIdMapContext& mapCache );


      virtual void write( char   val ) ;
      virtual void write( bool   val ) ;
      virtual void write( agx::UInt8  val ) ;
      virtual void write( agx::Int8  val ) ;
      virtual void write( agx::Int16  val ) ;
      virtual void write(  agx::UInt16 val ) ;

      virtual void write( agx::UInt32 val ) ;
      virtual void write( agx::Int32 val ) ;

      virtual void write( agx::UInt64 val ) ;
      virtual void write( agx::Int64  val ) ;
      virtual void write( float  val ) ;
      virtual void write( double val ) ;

      virtual void write( const char* val ) ;
      virtual void write( const std::string& val ) ;

      virtual void write( const agx::Vec3f& val ) ;
      virtual void write( const agx::Vec3d& val ) ;
      virtual void write( const agx::Uuid& val ) ;

      /**
      This method will write raw data into the archive.
      Notice that you should always have called beginSection() prior to calling this method.
      Otherwise it is not compatible with XML output.

      The raw data will be encoded before written as string into the XML tree

      \param buf - Pointer to the data to be written
      \param len - Number of bytes to write
      */
      virtual void write( const void* buf, size_t len ) ;

      using OutputArchive::write;

      virtual void beginSection( const char* title  );
      virtual void endSection( const char* title );


      virtual void addAttribute(const char *name, bool value);
      virtual void addAttribute(const char *name, const char* value);
      virtual void addAttribute(const char *name, char value);
      virtual void addAttribute(const char *name, agx::UInt8 value);
      virtual void addAttribute(const char *name, agx::Int8 value);
      virtual void addAttribute(const char *name, agx::UInt16 value);
      virtual void addAttribute(const char *name, agx::UInt32 value);
      virtual void addAttribute(const char *name, agx::Int32 value);
      virtual void addAttribute(const char *name, agx::UInt64 value);
      virtual void addAttribute(const char *name, float value);
      virtual void addAttribute(const char *name, double value);
      virtual void addAttribute(const char *name, const agx::Uuid& value);

      /// Method for closing the archive, important to call this as it will flush the data to the outputstream.
      void close() { OutputArchive::close(); flush(); }
  protected:
    /// Destructor
    virtual ~XMLOutputArchive();

    void addString(const char *str);

    bool flush();

    // copy constructor and assignment operator prohibited
    XMLOutputArchive( const XMLOutputArchive& );
    XMLOutputArchive& operator=( const XMLOutputArchive& );
    //void createXMLNode( const char *value );

    private:

      agx::TiXmlDocument *m_document;
      agx::TiXmlElement *m_element;
      agx::TiXmlElement *m_parent;
      agx::Int32 m_precision;

      std::stringstream m_stringBuf;

  };
  typedef agx::ref_ptr<XMLOutputArchive> XMLOutputArchiveRef;

}

#ifdef _MSC_VER
# pragma warning( pop )
#endif
