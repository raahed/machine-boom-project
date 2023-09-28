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


#ifndef AGX_STRING_H
#define AGX_STRING_H


#include <agx/Integer.h>
#include <agx/Real.h>
#include <agx/Vector.h>

#include <string>


namespace agx
{
  template<typename T = std::string> class AgXString;

#ifndef SWIG
  typedef Vector<AgXString<std::string> > StringVector;
#else
  typedef Vector<String> StringVector;
#endif

  template<typename T>
  class AgXString : public T
  {
    public:
      typedef std::string::size_type      size_type;
      typedef std::string::iterator       iterator;
      typedef std::string::const_iterator const_iterator;

    public:
      /**
      Default constructor.
      */
      AgXString();

      /**
      Copy constructors.
      */
      AgXString( const AgXString& str );

      /**
      Create a AgXString from stl string.
      */
      AgXString( const std::string& str );

      /**
      Element type pointer constructor.
      */
      AgXString( const char* str );

      /**
      Construct \p length number of \p ch.
      \param length - final length of this string
      \param ch - filling
      */
      AgXString( size_type length, char ch );

      /**
      Construct given \p str, final length of this string will be \p length.
      \param str - string
      \param length - final length of this string
      \note If length > strlen( str ) it's not defined which characters this string is extended with.
      */
      AgXString( const char* str, size_type length );

      /**
      Construct sub-string of \p str from \p index and \p length characters forward.
      \param str - a string
      \param index - index from where this sub-string should start
      \param length - length from \p index
      */
      AgXString( const char* str, size_type index, size_type length );

      /**
      Construct sub-string given iterators.
      \param begin - start iterator
      \param end - end iterator
      */
      AgXString( const_iterator begin, const_iterator end );

      /**
      Operator +, AgXString b = thisAgXString + other.
      */
      AgXString operator + ( const AgXString& other ) const;

      /**
      Operator +, AgXString b = thisAgXString + "other".
      */
      AgXString operator + ( const char* other ) const;

      /**
      Operator +, AgXString b = thisAgXString + 'o'.
      */
      AgXString operator + ( const char elem ) const;

      /**
      Copy method, if implementation of this class is changed.
      */
      AgXString copy() const;

      /**
      \return sub-string of this from \p index to end
      */
      AgXString substr( size_type index ) const;

      /**
      \return sub-string of this given \p index and \p length
      */
      AgXString substr( size_type index, size_type length ) const;

      /**
      C printf formatting of a string. E.g., agx::AgXString s = agx::AgXString::format( "%s%d", "a number: ", 10 );.
      \return string of given format
      */
      static AgXString format( const char* format, ... );
      static AgXString formatVA( const char* format, va_list ap);


      /**
      \return true if this string is equal to \p other
      */
      bool operator== (const std::string& other) const;

      /**
      \return true if this string is equal to \p other
      */
      bool operator== (const char* other) const;

      /**
      \return true if this string is NOT equal to \p other
      */
      bool operator!= (const std::string& other) const;

      /**
      \return true if this string is NOT equal to \p other
      */
      bool operator!= (const char* other) const;

      /* Pystring extensions */
      ///////////////////////////////////////////////////////////////////////////////
      // Copyright (c) 2008, Sony Pictures Imageworks
      // All rights reserved.
      //
      // Redistribution and use in source and binary forms, with or without
      // modification, are permitted provided that the following conditions are
      // met:
      //
      // Redistributions of source code must retain the above copyright notice,
      // this list of conditions and the following disclaimer.
      // Redistributions in binary form must reproduce the above copyright
      // notice, this list of conditions and the following disclaimer in the
      // documentation and/or other materials provided with the distribution.
      // Neither the name of the organization Sony Pictures Imageworks nor the
      // names of its contributors
      // may be used to endorse or promote products derived from this software
      // without specific prior written permission.
      // THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
      // "AS
      // IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
      // TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
      // PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
      // OWNER
      // OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
      // SPECIAL,
      // EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
      // PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
      // PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
      // LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
      // NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
      // SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
      ///////////////////////////////////////////////////////////////////////////////

#define MAX_32BIT_INT 2147483647

      /**
      \return a copy of the string with only its first character capitalized.
      */
      AgXString capitalize() const;

      /**
      \return centered in a string of length width. Padding is done using spaces.
      */
      AgXString center(int width) const;

      /**
      \return the number of occurrences of substring sub in string S[start:end]. Optional
      arguments start and end are interpreted as in slice notation.
      */
      int count(const AgXString& substr, int start = 0, int end = MAX_32BIT_INT) const;

      /**
      \return True if the string ends with the specified suffix, otherwise return False. With
      optional start, test beginning at that position. With optional end, stop comparing at that position.
      */
      bool endswith(const AgXString& suffix, int start = 0, int end = MAX_32BIT_INT) const;
      bool endswithCI(const AgXString& suffix, int start = 0, int end = MAX_32BIT_INT) const;

      /**
      \return a copy of the string where all tab characters are expanded using spaces. If tabsize
      is not given, a tab size of 8 characters is assumed.
      */
      AgXString expandtabs(int tabsize = 8) const;

      /**
      \return the lowest index in the string where substring sub is found, such that sub is
      contained in the range [start, end). Optional arguments start and end are interpreted as
      in slice notation. Return -1 if sub is not found.
      */
      bool contains(const AgXString& sub, size_t start = 0) const;

      /**
      Synonym of find right now. Python version throws exceptions. This one currently doesn't
      */
      int index(const AgXString& sub, int start = 0, int end = MAX_32BIT_INT) const;

      /**
      \return true if all characters in the string are alphanumeric and there is at least one
      character, false otherwise.
      */
      bool isalnum() const;

      /**
      \return true if all characters in the string are alphabetic and there is at least one
      character, false otherwise
      */
      bool isalpha() const;

      /**
      \return true if all characters in the string are digits and there is at least one
      character, false otherwise.
      */
      bool isdigit() const;

      /**
      \return true if all cased characters in the string are lowercase and there is at least one
      cased character, false otherwise.
      */
      bool islower() const;

      /**
      \return true if there are only whitespace characters in the string and there is at least
      one character, false otherwise.
      */
      bool isspace() const;

      /**
      \return true if the string is a titlecased string and there is at least one character,
      i.e. uppercase characters may only follow uncased characters and lowercase characters only
      cased ones. Return false otherwise.
      */
      bool istitle() const;

      /**
      \return true if all cased characters in the string are uppercase and there is at least one
      cased character, false otherwise.
      */
      bool isupper() const;

      /**
      \return the string left justified in a string of length width. Padding is done using
      spaces. The original string is returned if width is less than str.size().
      */
      AgXString ljust(int width) const;

      /**
      \return a copy of the string converted to lowercase.
      */
      AgXString lower() const;

      /**
      \return a copy of the string with leading characters removed. If chars is omitted or None,
      whitespace characters are removed. If given and not "", chars must be a string; the
      characters in the string will be stripped from the beginning of the string this method
      is called on (argument "str" ).
      */
      AgXString lstrip(const AgXString& chars = "") const;

      /**
      If this string begins with \p toBeRemoved, it will be removed and returned.
      \return this string with the leading string \p toBeRemoved removed. Otherwise the string is returned unmodified.
      */
      AgXString stripLeadingString(const AgXString& toBeRemoved) const;

      /**
      Split the string around first occurrence of sep.
      Three strings will always placed into result. If sep is found, the strings will
      be the text before sep, sep itself, and the remaining text. If sep is
      not found, the original string will be returned with two empty strings.
      */
      void partition(const AgXString& sep, StringVector& result ) const;

      /**
      \return a copy of the string with all occurrences of substring old replaced by new. If
      the optional argument count is given, only the first count occurrences are replaced.
      */
      AgXString replace(const AgXString& oldstr, const AgXString& newstr, int count = -1) const;

      /**
      \return the lowest index in the string where substring sub is found, such that sub is
      contained in the range [start, end). Optional arguments start and end are interpreted as
      in slice notation. Return npos if sub is not found.
      */
      size_t find(const AgXString& sub, int start = 0, int end = MAX_32BIT_INT ) const;

      /**
      \return the highest index in the string where substring sub is found, such that sub is
      contained within s[start,end]. Optional arguments start and end are interpreted as in
      slice notation. Return npos if sub is not found.
      */
      size_t rfind(const AgXString& sub, int start = 0, int end = MAX_32BIT_INT ) const;

      /**
      \return the string right justified in a string of length width. Padding is done using
      spaces. The original string is returned if width is less than str.size().
      */
      AgXString rjust(int width) const;

      /**
      Split the string around last occurrence of sep.
      Three strings will always placed into result. If sep is found, the strings will
      be the text before sep, sep itself, and the remaining text. If sep is
      not found, the original string will be returned with two empty strings.
      */
      void rpartition(const AgXString& sep, StringVector& result ) const;

      /**
      \return a copy of the string with trailing characters removed. If chars is "", whitespace
      characters are removed. If not "", the characters in the string will be stripped from the
      end of the string this method is called on.
      */
      AgXString rstrip(const AgXString& chars = "") const;

      /**
      Fills the "result" list with the words in the string, using sep as the delimiter string.
      If maxsplit is > -1, at most maxsplit splits are done. If sep is "",
      any whitespace string is a separator.
      */
      void split(StringVector& result, const AgXString& sep = "", int maxsplit = -1) const;

      /**
      Fills the "result" list with the words in the string, using sep as the delimiter string.
      Does a number of splits starting at the end of the string, the result still has the
      split strings in their original order.
      If maxsplit is > -1, at most maxsplit splits are done. If sep is "",
      any whitespace string is a separator.
      */
      void rsplit(StringVector& result, const AgXString& sep = "", int maxsplit = -1) const;

      /**
      \return a list of the lines in the string, breaking at line boundaries. Line breaks
      are not included in the resulting list unless keepends is given and true.
      */
      void splitlines(StringVector& result, bool keepends = false ) const;

      /**
      \return True if string starts with the prefix, otherwise return False. With optional start,
      test string beginning at that position. With optional end, stop comparing string at that
      position
      */
      bool startswith(const AgXString& prefix, int start = 0, int end = MAX_32BIT_INT ) const;
      bool startswithCI(const AgXString& prefix, int start = 0, int end = MAX_32BIT_INT ) const;

      /**
      \return a copy of the string with leading and trailing characters removed. If chars is "",
      whitespace characters are removed. If given not "",  the characters in the string will be
      stripped from the both ends of the string this method is called on.
      */
      AgXString strip(const AgXString& chars = "") const;

      /**
      \return a copy of the string with uppercase characters converted to lowercase and vice versa.
      */
      AgXString swapcase() const;

      /**
      \return a titlecased version of the string: words start with uppercase characters,
      all remaining cased characters are lowercase.
      */
      AgXString title() const;

      /**
      \return a copy of the string where all characters occurring in the optional argument
      deletechars are removed, and the remaining characters have been mapped through the given
      translation table, which must be a string of length 256.
      */
      AgXString translate(const AgXString& table, const AgXString& deletechars = "") const;

      /**
      \return a copy of the string converted to uppercase.
      */
      AgXString upper() const;

      /**
      \return the numeric string left filled with zeros in a string of length width. The original
      string is returned if width is less than str.size().
      */
      AgXString zfill(int width) const;

      /**
      function matching python's slice functionality.
      */
      AgXString slice(int start = 0, int end = MAX_32BIT_INT) const;
  };

#ifndef SWIG
  typedef AgXString<std::string> String;
#endif

  /**
   * Return a reference to an empty string that is owned by an AGX Dynamics
   * shared library. Such a string reference is required when returning to
   * calling code that was build with allocator incompatible compiler or linker
   * settings and thus exhibit memory errors when destroying strings allocated
   * from within an AGX Dynamics shared library.
   */
  AGXCORE_EXPORT const agx::String& getEmptyString();

  // Utility function to print bit patterns
  // NOTE: Ignores endianess, handles data as a block of bytes
  template <typename T>
  agx::String printBin(const T& val);

  template <typename T>
  agx::String printHex(const T& val);

  agx::String AGXCORE_EXPORT printBin(const void* data, UInt numBytes);
  agx::String AGXCORE_EXPORT printHex(const void* data, UInt numBytes);


  // Specializations for basic types, handles endianess
  agx::String AGXCORE_EXPORT printBin(UInt32 val);
  agx::String AGXCORE_EXPORT printBin(UInt64 val);
  agx::String AGXCORE_EXPORT printBin(Real32 val);
  agx::String AGXCORE_EXPORT printBin(Real64 val);

  agx::String AGXCORE_EXPORT printHex(UInt32 val);
  agx::String AGXCORE_EXPORT printHex(UInt64 val);
  agx::String AGXCORE_EXPORT printHex(Real32 val);
  agx::String AGXCORE_EXPORT printHex(Real64 val);

} // namespace agx


#include <external/pystring/pystring.h>

namespace agx
{

  template<typename T>
  inline AgXString<T> AgXString<T>::format(const char* format, ...)
  {
    va_list ap;
    va_start(ap, format);
    AgXString<T> result = AgXString<T>::formatVA(format, ap);
    va_end(ap);
    return result;
  }

  template<typename T>
  inline AgXString<T> AgXString<T>::formatVA(const char* format, va_list ap)
  {
    char* buf = 0;

#ifdef _MSC_VER
    int len = _vscprintf( format, ap ) + 1;
    buf = (char*) malloc( len * sizeof(char) );
    if (buf)
      vsprintf_s( buf, len, format, ap );
#else
    int ret = vasprintf(&buf, format, ap);
    agxVerifyN(ret >= 0, "Could not allocate memory for formatted string: \'%s\'", format);
#endif

    agxAssert(buf);
    if (buf) {
      AgXString<T> result(buf);
      free(buf);

      return result;
    }
    else
      return AgXString<T>(); // Fall back.
  }

  // Implementation

  template <typename T>
  inline AgXString<T>::AgXString() : T()
  {

  }

  template <typename T>
  inline AgXString<T>::AgXString( const AgXString<T>& str )
    : T( str )

  {
  }

  template <typename T>
  inline AgXString<T>::AgXString( const std::string& str )
    : T( str )
  {
  }

  template <typename T>
  inline AgXString<T>::AgXString( const char* str )
    : T( str )
  {
  }

  template <typename T>
  inline AgXString<T>::AgXString( size_type length, char ch )
    : T( length, ch )
  {
  }

  template <typename T>
  inline AgXString<T>::AgXString( const char* str, size_type length )
    : T( str, length )

  {
  }

  template <typename T>
  inline AgXString<T>::AgXString( const char* str, size_type index, size_type length )
    : T( str, index, length )

  {
  }

  template <typename T>
  inline AgXString<T>::AgXString( const_iterator begin, const_iterator end )
    : T( begin, end )
  {
  }

  template <typename T>
  inline AgXString<T> AgXString<T>::operator + ( const AgXString<T>& other ) const
  {
    AgXString ret = *this;
    ret += other;
    return ret;
  }

  template <typename T>
  inline AgXString<T> AgXString<T>::operator + ( const char* other ) const
  {
    AgXString ret = *this;
    ret += other;
    return ret;
  }

  template <typename T>
  inline AgXString<T>  AgXString<T>::operator + ( const char elem ) const
  {
    AgXString ret = *this;
    ret += elem;
    return ret;
  }

  template <typename T>
  inline AgXString<T> AgXString<T>::copy() const
  {
    return AgXString( *this );
  }

  template <typename T>
  inline AgXString<T> AgXString<T>::substr( size_type index ) const
  {
    return AgXString( this->c_str(), index, this->size() );
  }

  template <typename T>
  inline AgXString<T> AgXString<T>::substr( size_type index, size_type length ) const
  {
    return AgXString( this->c_str(), index, length );
  }

  template <typename T>
  AGX_FORCE_INLINE bool AgXString<T>::operator== (const std::string& other) const
  {
    return *(std::string*)this == other;
  }

  template <typename T>
  AGX_FORCE_INLINE bool AgXString<T>::operator== (const char* other) const
  {
    return *(std::string*)this == other;
  }

  template <typename T>
  AGX_FORCE_INLINE bool AgXString<T>::operator!= (const std::string& other) const
  {
    return *(std::string*)this != other;
  }

  template <typename T>
  AGX_FORCE_INLINE bool AgXString<T>::operator!= (const char* other) const
  {
    return *(std::string*)this != other;
  }



  template<typename T>
  AgXString<T> operator + ( const char* str1, const AgXString<T>& str2 )
  {
    return AgXString<T>( str1 ) + str2;
  }

  template<typename T>
  AgXString<T> operator + ( const char elem, const AgXString<T>& str )
  {
    return AgXString<T>( 1, elem ) + str;
  }

  template<typename T>
  inline bool AgXString<T>::contains(const AgXString<T>& sub, size_t start) const
  {
    return this->find(sub, int(start)) != AgXString<T>::npos;
  }


  template<typename T>
  AgXString<T> AgXString<T>::capitalize() const
  {
    return pystring::capitalize(*this);
  }

  template<typename T>
  AgXString<T> AgXString<T>::center(int width) const
  {
    return pystring::center(*this, width);
  }

  template<typename T>
  int AgXString<T>::count(const AgXString<T>& substr, int start, int end) const
  {
    return pystring::count(*this, substr, start, end);
  }

  template<typename T>
  bool AgXString<T>::endswith(const AgXString<T>& suffix, int start, int end) const
  {
    return pystring::endswith(*this, suffix, start, end);
  }

  template<typename T>
  bool AgXString<T>::endswithCI(const AgXString<T>& suffix, int start, int end) const
  {
    return pystring::endswithCI(*this, suffix, start, end);
  }

  template<typename T>
  AgXString<T> AgXString<T>::expandtabs(int tabsize) const
  {
    return pystring::expandtabs(*this, tabsize);
  }

  /*
  int AgXString<T>::find(const AgXString<T>& sub, int start, int end) const
  {
    return pystring::find(*this, sub, start, end);
  }
  */

  template<typename T>
  int AgXString<T>::index(const AgXString<T>& sub, int start, int end) const
  {
    return pystring::index(*this, sub, start, end);
  }

  template<typename T>
  bool AgXString<T>::isalnum() const
  {
    return pystring::isalnum(*this);
  }

  template<typename T>
  bool AgXString<T>::isalpha() const
  {
    return pystring::isalpha(*this);
  }

  template<typename T>
  bool AgXString<T>::isdigit() const
  {
    return pystring::isdigit(*this);
  }

  template<typename T>
  bool AgXString<T>::islower() const
  {
    return pystring::islower(*this);
  }

  template<typename T>
  bool AgXString<T>::isspace() const
  {
    return pystring::isspace(*this);
  }

  template<typename T>
  bool AgXString<T>::istitle() const
  {
    return pystring::istitle(*this);
  }

  template<typename T>
  bool AgXString<T>::isupper() const
  {
    return pystring::isupper(*this);
  }


  template<typename T>
  AgXString<T> AgXString<T>::ljust(int width) const
  {
    return pystring::ljust(*this, width);
  }

  template<typename T>
  AgXString<T> AgXString<T>::lower() const
  {
    return pystring::lower(*this);
  }


  template<typename T>
  AgXString<T> AgXString<T>::stripLeadingString(const AgXString<T>& toBeRemoved) const
  {
    std::size_t start = this->find(toBeRemoved); // Find the string to be removed
    if (start == 0 ) {// Did we find it, and its from the start of the string?
      if (T::length() > toBeRemoved.size()) {
        return this->substr(toBeRemoved.size(), T::length() ); // Remove it.
      } else
        return "";
    }
    return *this;
  }


  template<typename T>
  AgXString<T> AgXString<T>::lstrip(const AgXString<T>& chars) const
  {
    return pystring::lstrip(*this, chars);
  }

  template<typename T>
  void AgXString<T>::partition(const AgXString<T>& sep, StringVector& result ) const
  {
    return pystring::partition(*this, sep, result);
  }

  template<typename T>
  AgXString<T> AgXString<T>::replace(const AgXString<T>& oldstr, const AgXString<T>& newstr, int count) const
  {
    return pystring::replace(*this, oldstr, newstr, count);
  }

  template<typename T>
  size_t AgXString<T>::find(const AgXString<T>& sub, int start, int end) const
  {
    int index = pystring::find(*this, sub, start, end);
    return index < 0 ? AgXString<T>::npos : (size_t)index;
  }

  template<typename T>
  size_t AgXString<T>::rfind(const AgXString<T>& sub, int start, int end) const
  {
    int index = pystring::rfind(*this, sub, start, end);
    return index < 0 ? AgXString<T>::npos : (size_t)index;
  }


  template<typename T>
  AgXString<T> AgXString<T>::rjust(int width) const
  {
    return pystring::rjust(*this, width);
  }

  template<typename T>
  void AgXString<T>::rpartition(const AgXString<T>& sep, StringVector& result ) const
  {
    return pystring::rpartition(*this, sep, result);
  }

  template<typename T>
  AgXString<T> AgXString<T>::rstrip(const AgXString<T>& chars) const
  {
    return pystring::rstrip(*this, chars);
  }

  template<typename T>
  void AgXString<T>::split(StringVector& result, const AgXString<T>& sep, int maxsplit) const
  {
    return pystring::split(*this, result, sep, maxsplit);
  }

  template<typename T>
  void AgXString<T>::rsplit(StringVector& result, const AgXString<T>& sep, int maxsplit) const
  {
    return pystring::rsplit(*this, result, sep, maxsplit);
  }

  template<typename T>
  void AgXString<T>::splitlines(StringVector& result, bool keepends) const
  {
    return pystring::splitlines(*this, result, keepends);
  }

  template<typename T>
  bool AgXString<T>::startswith(const AgXString<T>& prefix, int start, int end) const
  {
    return pystring::startswith(*this, prefix, start, end);
  }

  template<typename T>
  bool AgXString<T>::startswithCI(const AgXString<T>& prefix, int start, int end) const
  {
    return pystring::startswithCI(*this, prefix, start, end);
  }

  template<typename T>
  AgXString<T> AgXString<T>::strip(const AgXString<T>& chars) const
  {
    if (this->size() == 0)
      return *this;

    return pystring::strip(*this, chars);
  }

  template<typename T>
  AgXString<T> AgXString<T>::swapcase() const
  {
    return pystring::swapcase(*this);
  }

  template<typename T>
  AgXString<T> AgXString<T>::title() const
  {
    return pystring::title(*this);
  }

  template<typename T>
  AgXString<T> AgXString<T>::translate(const AgXString<T>& table, const AgXString<T>& deletechars) const
  {
    return pystring::translate(*this, table, deletechars);
  }

  template<typename T>
  AgXString<T> AgXString<T>::upper() const
  {
    return pystring::upper(*this);
  }

  template<typename T>
  AgXString<T> AgXString<T>::zfill(int width) const
  {
    return pystring::zfill(*this, width);
  }

  template<typename T>
  AgXString<T> AgXString<T>::slice(int start, int end) const
  {
    return pystring::slice(*this, start, end);
  }


  template <typename T>
  agx::String printBin(const T& val)
  {
    return printBin(&val, sizeof(T));
  }

  template <typename T>
  agx::String printHex(const T& val)
  {
    return printHex(&val, sizeof(T));
  }

}

namespace std
{
  AGX_FORCE_INLINE void swap(agx::String& lhs, agx::String& rhs)
  {
    std::swap(static_cast<std::string&>(lhs), static_cast<std::string&>(rhs));
  }
}


#endif /* _AGX_STRING_H_ */
