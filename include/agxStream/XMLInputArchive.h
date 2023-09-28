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

#include <agxStream/InputArchive.h>

#ifdef _MSC_VER
# pragma warning(push)
# pragma warning( disable : 4251 ) // class X needs to have dll-interface to be used by clients of class Y
#endif


namespace agx
{
  class TiXmlDocument;
  class TiXmlElement;
  class TiXmlNode;
}

namespace agxStream
{

  /**
  XML version of an input archive for restoring data written with the XMLOutputArchive.
  */
  class AGXCORE_EXPORT XMLInputArchive : public InputArchive
  {
      // copy constructor and assignment operator prohibited
      XMLInputArchive( const XMLInputArchive& inputStream );

      XMLInputArchive& operator=( const XMLInputArchive& );

    public:

      /**
      Constructor creating a XMLInputArchive for reading XML streams. Data will be read from the \p inputStream
      during the constructor.
      */
      XMLInputArchive( std::istream& inputStream );
      XMLInputArchive(std::istream& inputStream, IdToObjectVector& idCache, IdToClassNameVector& nameCache);


      using InputArchive::read;
      virtual void read( char& val ) ;
      virtual void read( bool& val ) ;
      virtual void read( agx::Int8& val ) ;
      virtual void read( agx::UInt8& val ) ;
      virtual void read( agx::Int16& val ) ;
      virtual void read( agx::UInt16& val ) ;

      virtual void read(  agx::Vec3d& val ) ;
      virtual void read(  agx::Vec3f& val ) ;
      virtual void read(  agx::Uuid& val ) ;

      virtual void read(  agx::UInt32& val ) ;
      virtual void read( agx::Int32& val ) ;
      virtual void read( std::string& val ) ;
      virtual void read( agx::UInt64& val ) ;
      virtual void read( agx::Int64& val ) ;
      virtual void read( float& val ) ;
      virtual void read( double& val ) ;


      /**
      This method will read raw data from the archive.
      Notice that you should always have called beginSection() prior to calling this method.
      Otherwise it is not compatible with XML input

      \param buf - Pointer to the buffer where data will be read
      \param size - Number of bytes to read
      */
      virtual void read(void * buf, size_t size);

      virtual size_t getNumBytesRead() const;

      virtual void beginSection( const char* title );
      virtual void endSection( const char* title );




      agx::String getErrorMessage( const agx::String& message );

      virtual void getAttribute(const char *name, std::string& value);
      virtual void getAttribute(const char *name, float& value);
      virtual void getAttribute(const char *name, double& value);
      virtual void getAttribute(const char *name, bool& value);
      virtual void getAttribute(const char *name, char& value);
      virtual void getAttribute(const char *name, agx::Int8& value);
      virtual void getAttribute(const char *name, agx::UInt8& value);
      virtual void getAttribute(const char *name, agx::UInt16& value);
      virtual void getAttribute(const char *name, agx::UInt32& value);
      virtual void getAttribute(const char *name, agx::Int32& value);
      virtual void getAttribute(const char *name, agx::UInt64& value);
      virtual void getAttribute(const char *name, agx::Uuid& value);

    protected:
      /// Destructor
      virtual ~XMLInputArchive();

      void init();
      agx::TiXmlElement* getElement( const char *variable );

      agx::TiXmlDocument *m_document;
      //agx::TiXmlElement *m_element;
      //agx::TiXmlElement *m_parent;

      struct StackElement {
        agx::TiXmlNode *parent;
        agx::TiXmlNode *prev;
        agx::TiXmlNode *curr;

        StackElement(agx::TiXmlNode *parent_arg, agx::TiXmlNode *prev_arg, agx::TiXmlNode *element_arg) :
          parent(parent_arg), prev(prev_arg), curr(element_arg)
        {
          agxAssert(parent);
          }
      };
      agx::Vector<StackElement> m_sectionStack;
      std::stringstream m_stringBuf;
      std::string m_currTitle;
      agx::TiXmlElement *m_currentElement;
      agx::TiXmlNode *m_lastNode;
  };
  typedef agx::ref_ptr<XMLInputArchive> XMLInputArchiveRef;

}

#ifdef _MSC_VER
#ifdef _MSC_VER
# pragma warning(pop)
#endif
#endif

