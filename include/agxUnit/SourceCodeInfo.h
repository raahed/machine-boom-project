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

#ifndef AGXUNIT_SOURCECODEINFO_H
#define AGXUNIT_SOURCECODEINFO_H

#include <agx/config.h>
#include <agx/config/AGX_USE_UNIT_TESTS.h>


#if AGX_USE_UNIT_TESTS()

#include <agx/agxPhysics_export.h>
#include <agx/String.h>
#include <iosfwd>


#ifdef _MSC_VER
# pragma warning(push)
# pragma warning(disable: 4251) // warning C4251: class X needs to have dll-interface to be used by clients of class Y
#endif


namespace agxUnit
{
  class AGXPHYSICS_EXPORT SourceCodeInfo
  {
  public:
    SourceCodeInfo() : m_line(0) {}
    SourceCodeInfo(int line, const agx::String& file, const agx::String& function) : m_line(line), m_file(file), m_function(function) {}

    SourceCodeInfo & operator =( const SourceCodeInfo &other )
    {
      if ( this != &other )
      {
        m_file = other.m_file.c_str();
        m_line = other.m_line;
        m_function = other.m_function.c_str();
      }
      return *this;
    }

    SourceCodeInfo( const SourceCodeInfo &other )
      : m_line( other.m_line ), m_file( other.m_file.c_str() ), m_function( other.m_function.c_str() )
    {
    }


    bool operator ==( const SourceCodeInfo &other ) const {
      return m_line == other.m_line && m_file == other.m_file&& m_function==other.m_function;
    }

    bool operator !=( const SourceCodeInfo &other ) const { return !(*this == other); }

    agx::String filename() const { return m_file; }
    agx::String function() const { return m_function; }
    int line() const { return m_line; }

    friend std::ostream& operator <<( std::ostream& os, const SourceCodeInfo &info ) {
      os << "File: " << info.filename();
#ifdef __linux__
      os << ":" << info.line() << " ";
#else
      os << "(" << info.line() << ") ";
#endif
      os << "Function: " << info.function();
      return os;
    }

  private:
    int m_line;
    agx::String m_file;
    agx::String m_function;
  };


}

#ifdef _MSC_VER
#  pragma warning(pop)
#endif


#endif // AGX_USE_UNIT_TESTS()



#endif //AGXUNIT_SOURCECODEINFO_H
