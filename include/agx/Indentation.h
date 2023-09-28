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

#ifndef AGX_INDENTATION_H
#define AGX_INDENTATION_H

#include <agx/Integer.h>
#include <agx/String.h>
#include <iostream>

namespace agx
{
  class AGXCORE_EXPORT Indentation
  {
  public:
    static const agx::UInt TabSize = 2;


    Indentation() : m_level(0) {}
    Indentation(agx::UInt level) : m_level(level) {}

    agx::UInt getLevel() const { return m_level; }
    agx::UInt getIndentation() const { return m_level * TabSize; }

    /**
    Indent a text, applied for each line in input string.
    \param string The input text string
    \return The indented result
    */
    String apply(const String& string);

    Indentation& operator+= (int increment) { m_level += increment; return *this; }
    Indentation& operator-= (int decrement) { m_level -= decrement; return *this; }

    Indentation operator+ (int increment) const { return Indentation(m_level + increment); }
    Indentation operator- (int decrement) const { return Indentation(m_level - decrement); }

    Indentation& operator++() { m_level++; return *this; }
    Indentation operator++(int) { m_level++; return Indentation(m_level - 1); }

    Indentation& operator--() { m_level--; return *this; }
    Indentation operator--(int) { m_level--; return Indentation(m_level + 1); }

    String str() const { return String(getIndentation(), ' '); }
    operator String() const { return str(); }

  private:
    agx::UInt m_level;
  };


  template<typename T>
  AgXString<T> operator+ ( const Indentation& indentation, const AgXString<T>& str )
  {
    return indentation.str() + str;
  }


  inline std::ostream& operator<< (std::ostream& stream, const Indentation& indentation)
  {
    stream << indentation.str();
    return stream;
  }

}


#endif /* AGX_INDENTATION_H */
