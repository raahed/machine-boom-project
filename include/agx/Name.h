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

#ifndef AGX_NAME_H
#define AGX_NAME_H

#include <agx/String.h>
#include <agx/AtomicValue.h>
#include <agx/HashFunction.h>
#include <string.h>
#include <iosfwd>

namespace agx
{
  /**
  Representation of a name string. Identical names share string buffer, so
  copying a name is a cheap operation, just a reference increment.
  */
  class AGXCORE_EXPORT Name
  {
  public:
    Name();
    Name(const char *name);
    Name(const agx::String& name);
    Name(const std::string& name);
    Name(const Name& name);

    ~Name();

    /**
    Cast operators
    */
    operator agx::String() const;
    operator std::string() const;
    // operator const char *() const;

    /**
    \return The agx::String representation.
    */
    agx::String str() const;

    /**
    \return The c-string representation.
    */
    const char *c_str() const;

    /**
    \return The length of the name.
    */
    agx::UInt32 length() const;


    /**
    \return The length of the name.
    */
    agx::UInt32 size() const;

    /**
    \return True if the name is empty.
    */
    bool empty() const;


    /**
    \return The character at a specified index.
    */
    char operator[] (size_t index) const;

    /** Copy operators */
    agx::Name& operator= (const agx::Name& other);
    agx::Name& operator= (const std::string& other);
    agx::Name& operator= (const agx::String& other);
    agx::Name& operator= (const char *other);


    /** Comparison operators */
    bool operator== (const agx::Name& other) const;
    bool operator== (const std::string& other) const;
    bool operator== (const agx::String& other) const;
    bool operator== (const char *other) const;

    bool operator!= (const agx::Name& other) const;
    bool operator!= (const std::string& other) const;
    bool operator!= (const agx::String& other) const;
    bool operator!= (const char *other) const;

    /** Concatenation operator */
    agx::String operator+ (const agx::Name& other) const;
    agx::String operator+ (const std::string& other) const;
    agx::String operator+ (const agx::String& other) const;
    agx::String operator+ (const char *other) const;


    // Hash function
    UInt32 hash() const;

    void swap(Name& rhs);

  private:
    struct Header
    {
      agx::AtomicValue refCount;
      agx::UInt32 length;
    };

  private:
    void init(const char *name);
    static char *allocateSharedString(const char *string);

    static const char *getSharedString(const char *str);
    static void destroySharedString(const char *str);
    static Header& getHeader(const char *buf);

    Header& header();
    const Header& header() const;
    void incRef();
    void decRef();


  private:
    static const char *s_emptyString; // Important that empty name share string buffer, which is not guaranteed when using "" literals

    /**
    Name buffer allocation layout:
    -------------------------------------
    |    header    | char buffer...     |
    -------------------------------------
    ^              ^
    allocation     m_buf
    */
    const char *m_buf;
  };

  bool operator== (const std::string& str, const agx::Name& name);
  bool operator== (const agx::String& str, const agx::Name& name);
  bool operator== (const char *str, const agx::Name& name);

  bool operator!= (const std::string& str, const agx::Name& name);
  bool operator!= (const agx::String& str, const agx::Name& name);
  bool operator!= (const char *str, const agx::Name& name);

  AGXCORE_EXPORT std::ostream& operator << ( std::ostream& stream, const Name& name );





  /* Implementation */
  AGX_FORCE_INLINE Name::Name()
  {
    if (!s_emptyString)
    {
      // getSharedString return an already incremented ref, to avoid race conditions
      s_emptyString = Name::getSharedString("");
      m_buf = s_emptyString;
    }
    else
    {
      m_buf = s_emptyString;
      this->incRef();
    }
  }

  AGX_FORCE_INLINE Name::Name(const Name& name) : m_buf(name.m_buf)
  {
    this->incRef();
  }


  AGX_FORCE_INLINE Name::~Name()
  {
    this->decRef();
  }


  AGX_FORCE_INLINE const char *Name::c_str() const { return m_buf; }
  // AGX_FORCE_INLINE Name::operator const char *() const { return m_buf; }

  AGX_FORCE_INLINE agx::UInt32 Name::length() const { return header().length; }
  AGX_FORCE_INLINE agx::UInt32 Name::size() const { return length(); }
  AGX_FORCE_INLINE bool Name::empty() const { return length() == 0; }

  // Dangerous cast from char*, with alignment requirement 1, to Header*, with
  // alignment requirement 4. We guarantee that the cast is safe because the
  // casted pointer is in fact what was returned from malloc.
  //
  // A better way to solve this would be to keep the malloced pointer, the pointer
  // pointing to the Header, in Name::m_buf and add either a char* or a char[0]
  // member to Header to get at the string data.
  //
  // It is not enough to keep the Header as is and compute the string pointer as
  //    const char* string = reinterpret_cast<const char*>(m_buf) + sizeof(Header)
  // because that would create, and most likely later dereference, a pointer
  // pointing past the end of the object from which the original pointer was created.
  // May be a special case here since we stay within a single malloc'ed buffer,
  // but I'm not sure.
  #include <agx/PushDisableWarnings.h>
  AGX_FORCE_INLINE Name::Header& Name::getHeader(const char *buf) { agxAssert(buf); return *(Header *)(buf - sizeof(Header)); }
  #include <agx/PopDisableWarnings.h>

  AGX_FORCE_INLINE Name::Header& Name::header() { return Name::getHeader(m_buf); }
  AGX_FORCE_INLINE const Name::Header& Name::header() const { return Name::getHeader(m_buf); }

  AGX_FORCE_INLINE void Name::incRef() { this->header().refCount.inc(); }
  AGX_FORCE_INLINE void Name::decRef()
  {
    UInt32 refCount = (UInt32)this->header().refCount.dec()-1;

    if (refCount == 0)
    {
      Name::destroySharedString(m_buf);
      m_buf = nullptr;
    }
  }

  AGX_FORCE_INLINE UInt32 Name::hash() const
  {
    // Can hash on buffer ptr rather than string content, because identical names always share buffer
    return agx::hash((void *)m_buf);
  }

  AGX_FORCE_INLINE char Name::operator[] (size_t index) const
  {
    agxAssert(index < length());
    return m_buf[index];
  }


  AGX_FORCE_INLINE agx::Name& Name::operator= (const std::string& other)
  {
    (*this) = other.c_str();
    return *this;
  }

  AGX_FORCE_INLINE agx::Name& Name::operator= (const agx::String& other)
  {
    (*this) = other.c_str();
    return *this;
  }

  AGX_FORCE_INLINE agx::Name& Name::operator= (const char *other)
  {
    this->decRef();
    this->init(other);
    return *this;
  }

  AGX_FORCE_INLINE agx::Name& Name::operator= (const agx::Name& other)
  {
    const_cast<Name&>(other).incRef(); // incRef before decRef to handle case where name is assigned to itself
    this->decRef();
    m_buf = other.m_buf;
    return *this;
  }


  AGX_FORCE_INLINE bool Name::operator== (const Name& other) const
  {
    // Only need to compare string buffer pointer since it is shared by identical names
    return m_buf == other.m_buf;
  }

  AGX_FORCE_INLINE bool Name::operator== (const std::string& other) const
  {
    // Use default std::string comparison
    return m_buf == other;
  }

  AGX_FORCE_INLINE bool Name::operator== (const agx::String& other) const
  {
    // Use default std::string comparison
    return m_buf == other;
  }


  AGX_FORCE_INLINE bool Name::operator== (const char *other) const
  {
    return strcmp(m_buf, other) == 0;
  }


  AGX_FORCE_INLINE bool Name::operator!= (const Name& other) const { return !this->operator==(other); }
  AGX_FORCE_INLINE bool Name::operator!= (const std::string& other) const { return !this->operator==(other); }
  AGX_FORCE_INLINE bool Name::operator!= (const agx::String& other) const { return !this->operator==(other); }
  AGX_FORCE_INLINE bool Name::operator!= (const char *other) const { return !this->operator==(other); }


  AGX_FORCE_INLINE bool operator== (const std::string& str, const Name& name) { return name == str; }
  AGX_FORCE_INLINE bool operator== (const agx::String& str, const Name& name) { return name == str; }
  AGX_FORCE_INLINE bool operator== (const char *str, const Name& name) { return name == str; }

  AGX_FORCE_INLINE bool operator!= (const std::string& str, const Name& name) { return name != str; }
  AGX_FORCE_INLINE bool operator!= (const agx::String& str, const Name& name) { return name != str; }
  AGX_FORCE_INLINE bool operator!= (const char *str, const Name& name) { return name != str; }

  AGX_FORCE_INLINE void Name::swap(Name& rhs)
  {
    std::swap(m_buf, rhs.m_buf);
  }


}

AGXCORE_EXPORT agx::String operator+ (const std::string& str, const agx::Name& name);
AGXCORE_EXPORT agx::String operator+ (const agx::String& str, const agx::Name& name);
AGXCORE_EXPORT agx::String operator+ (const char *str, const agx::Name& name);

namespace std
{
  AGX_FORCE_INLINE void swap(agx::Name& lhs, agx::Name& rhs)
  {
    lhs.swap(rhs);
  }
}



#endif /* AGX_NAME_H */
