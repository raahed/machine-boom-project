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

#ifndef AGX_UUID_H
#define AGX_UUID_H

#include <agx/agxCore_export.h>
#include <agx/Integer.h>
#include <agx/String.h>
#include <agxData/Type.h>
#include <random>

#ifdef _MSC_VER
#pragma warning( push )
#pragma warning( disable : 4251 ) // class 'x' needs to have dll-interface to be used by clients of class 'y'
#endif


namespace agx
{

  /**
  A UUID, or Universally unique identifier, is intended to uniquely identify information in a distributed environment without significant central coordination.
  It can be used to tag objects with very short lifetimes, or to reliably identify very persistent objects across a network.
  Generator of UUID values based on V4
  http://en.wikipedia.org/wiki/Universally_unique_identifier
  */
  class AGXCORE_EXPORT Uuid
  {
  public:

    /// Constructor
    Uuid();

    /**
    Constructor, will construct a uuid based on the given Uuis formatted string
    \param uuidString - THe string with a uuid value
    */
    explicit Uuid(const agx::String& uuidString);

    inline bool operator==(const Uuid& rhs) const
    {
      return m_data.ab == rhs.m_data.ab && m_data.cd == rhs.m_data.cd;
    }

    inline bool operator!=(const Uuid& rhs) const
    {
      return !(m_data.ab == rhs.m_data.ab && m_data.cd == rhs.m_data.cd);
    }

    inline bool operator<(const Uuid& rhs) const
    {
      if (m_data.ab < rhs.m_data.ab) return true;
      if (m_data.ab > rhs.m_data.ab) return false;
      if (m_data.cd < rhs.m_data.cd) return true;
      return false;
    }

    inline bool operator>(const Uuid& rhs) const
    {
      if (m_data.ab > rhs.m_data.ab) return true;
      if (m_data.ab < rhs.m_data.ab) return false;
      if (m_data.cd > rhs.m_data.cd) return true;
      return false;
    }

    /**
    If this method returns true, the str() method will return a string of zeros
    \return true if this uuid has not been initialized to a value.
    */
    inline bool isNil() const
    {
      return (m_data.ab == m_data.cd) && (m_data.ab == 0);
    }

    /// \return the binary storage size of a uuid
    static size_t size() { return 2 * sizeof(uint64_t); }

    /// \return a const pointer to the underlying storage of an uuid
    const agx::UInt8 *data() const;

    /// \return pointer to the underlying storage of an uuid
    agx::UInt8 *data();

    friend class UuidGenerator;

    friend AGXCORE_EXPORT std::ostream& operator <<(std::ostream& stream, const Uuid& uuid);

#ifndef SWIG
    /// Sorting functor of an uuid
    template<class T>
    struct Less {
      // functor for operator<
      bool operator()(const T& _Left, const T& _Right) const {
        // apply operator< to operands
        return (_Left->getUuid() < _Right->getUuid());
      }
    };
#endif


    /**
    Generate a valid UUID string from a specified input string.
    The algorithm will use the specified string as a seed to generate an Uuid in the
    standard form.
    There is no guarantee that the Uuid is truly unique.
    \param seedString - A string used as a seed to create a unique identifier
    \return an Uuid based on the input string
    */
    static Uuid generateFromString(const agx::String& seedString);

    /// \return the Uuid in string format
    agx::String str() const;

    AGX_FORCE_INLINE UInt32 hash() const;

  protected:
    friend class UuidGenerator2;

    struct Data {
      Data(uint64_t xab, uint64_t xcd) : ab(xab), cd(xcd) {}
      Data() : ab(0), cd(0) {}

      uint64_t ab;
      uint64_t cd;
    };

    Data m_data;


  };


  AGXCORE_EXPORT std::ostream& operator <<(std::ostream& stream, const Uuid& uuid);


  template <typename ch, typename char_traits>
  std::basic_istream<ch, char_traits>& operator>>(std::basic_istream<ch, char_traits> &is, Uuid &u)
  {
    const typename std::basic_istream<ch, char_traits>::sentry ok(is);
    if (ok) {
      unsigned char data[16] = {0};

      typedef std::ctype<ch> ctype_t;
      ctype_t const& ctype = std::use_facet<ctype_t>(is.getloc());

      ch xdigits[16];
      {
        char szdigits[] = "0123456789ABCDEF";
        ctype.widen(szdigits, szdigits + 16, xdigits);
      }
      ch*const xdigits_end = xdigits + 16;

      ch c;
      for (std::size_t i = 0; i<u.size() && is; ++i) {
        is >> c;
        c = ctype.toupper(c);

        ch* f = std::find(xdigits, xdigits_end, c);
        if (f == xdigits_end) {
          is.setstate(std::ios_base::failbit);
          break;
        }

        unsigned char byte = static_cast<unsigned char>(std::distance(&xdigits[0], f));

        is >> c;
        c = ctype.toupper(c);
        f = std::find(xdigits, xdigits_end, c);
        if (f == xdigits_end) {
          is.setstate(std::ios_base::failbit);
          break;
        }

        byte = static_cast<unsigned char>(byte << 4);
        byte = static_cast<unsigned char>(byte | static_cast<unsigned char>(std::distance(&xdigits[0], f)));

        data[i] = byte;

        if (is) {
          if (i == 3 || i == 5 || i == 7 || i == 9) {
            is >> c;
            if (c != is.widen('-')) is.setstate(std::ios_base::failbit);
          }
        }
      }

      if (is) {
        memcpy(u.data(), data, 16);
        //std::copy(data, data + 16, u.data());
      }
    }
    return is;
  }

#ifdef _MSC_VER
#pragma warning( push )
#pragma warning( disable : 4127 ) // warning C4127: conditional expression is constant
#endif
  AGX_FORCE_INLINE UInt32 Uuid::hash() const
  {
    if (sizeof(size_t) > 4) {
      return agx::hash(size_t(m_data.ab ^ m_data.cd));
    }
    else {
      uint64_t hash64 = m_data.ab ^ m_data.cd;
      return agx::hash(size_t(uint32_t(hash64 >> 32) ^ uint32_t(hash64)));
    }
  }
#ifdef _MSC_VER
#pragma warning( pop )
#endif

  /**
  Generator of UUID values based on V4
  http://en.wikipedia.org/wiki/Universally_unique_identifier
  */
  class AGXCORE_EXPORT UuidGenerator
  {
  public:
    /// Constructor
    UuidGenerator();

    /// Destructor
    ~UuidGenerator();

    /// \return a newly created uuid
    Uuid generate();

  private:
    std::uniform_int_distribution<uint64_t> m_dist;

  };


} //namespace agx

#ifndef SWIG
AGX_TYPE_BINDING(agx::Uuid, "Uuid")

#endif


#ifdef _MSC_VER
#pragma warning( pop )
#endif

#endif
