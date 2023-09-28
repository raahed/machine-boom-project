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

#ifndef AGXDATA_BYTESTREAM_H
#define AGXDATA_BYTESTREAM_H

#include <agxData/agxData.h>

namespace agxData
{
  /**
  Utility class for parsing/writing a byte stream.
  */
  class AGXCORE_EXPORT ByteStream
  {
  public:
    ByteStream();
    ByteStream(BytePtr begin, BytePtr end);
    ByteStream(BytePtr begin, BytePtr end, BytePtr head);

    /**
    \return The stream buffer start address.
    */
    BytePtr& begin();
    const BytePtr& begin() const;

    /**
    \return The stream buffer end address.
    */
    BytePtr& end();
    const BytePtr& end() const;

    /**
    \return The current stream position.
    */
    BytePtr& head();
    const BytePtr& head() const;

    /**
    \return The size of the stream (in bytes)
    */
    size_t size() const;

    /**
    \return The current offset of the head in bytes
    */
    size_t getCurrentOffset() const;

    /**
    \return The front element.
    */
    template <typename T>
    T topElement() const;

    /**
    \return The front element and advance the stream head.
    */
    template <typename T>
    T popElement();


    /**
    \return The stream head and advance the stream the requested number of bytes.
    */
    BytePtr popBytes(size_t numBytes);

    /**
    Consume an expected value from the stream.
    */
    template <typename T>
    void consume(const T& value);

    /**
    Write a value to the head of the stream and advance the head.
    */
    template <typename T>
    void write(const T& value);

    /**
    Write a buffer.
    */
    void write(const Byte *buffer, size_t numBytes);
    void write(const char *buffer, size_t numBytes);

    /**
    Read a buffer.
    */
    void read(Byte *buffer, size_t numBytes);
    void read(char *buffer, size_t numBytes);


    ByteStream operator+ (size_t offset) const;
    ByteStream& operator+= (size_t offset);

  private:
    BytePtr m_begin;
    BytePtr m_end;
    mutable BytePtr m_head;
  };



  /* Implementation */
  AGX_FORCE_INLINE ByteStream::ByteStream() : m_begin(nullptr), m_end(nullptr), m_head(nullptr)
  {
  }

  AGX_FORCE_INLINE ByteStream::ByteStream(BytePtr begin, BytePtr end) : m_begin(begin), m_end(end), m_head(begin)
  {
  }

  AGX_FORCE_INLINE ByteStream::ByteStream(BytePtr begin, BytePtr end, BytePtr head) : m_begin(begin), m_end(end), m_head(head)
  {
    agxAssert(head >= begin && head <= end);
  }


  AGX_FORCE_INLINE BytePtr& ByteStream::begin() { return m_begin; }
  AGX_FORCE_INLINE const BytePtr& ByteStream::begin() const { return m_begin; }

  AGX_FORCE_INLINE BytePtr& ByteStream::end() { return m_end; }
  AGX_FORCE_INLINE const BytePtr& ByteStream::end() const { return m_end; }

  AGX_FORCE_INLINE BytePtr& ByteStream::head() { return m_head; }
  AGX_FORCE_INLINE const BytePtr& ByteStream::head() const { return m_head; }

  AGX_FORCE_INLINE size_t ByteStream::size() const { return (size_t)(m_end - m_begin); }
  AGX_FORCE_INLINE size_t ByteStream::getCurrentOffset() const { return (size_t)(m_head - m_begin); }

  template <typename T>
  inline T ByteStream::topElement() const
  {
    agxVerify((size_t)(m_end - m_head) >= sizeof(T));
    T element;
    ::memcpy((void *)&element, (const void *)m_head, sizeof(T));
    return element;
  }

  template <typename T>
  AGX_FORCE_INLINE T ByteStream::popElement()
  {
    T element = this->topElement<T>();
    m_head += sizeof(T);
    return element;
  }


  template <typename T>
  AGX_FORCE_INLINE void ByteStream::consume(const T& value)
  {
    T element = this->popElement<T>();
    agxVerify(element == value);
  }

  template <typename T>
  AGX_FORCE_INLINE void ByteStream::write(const T& value)
  {
    agxVerify((size_t)(m_end - m_head) >= sizeof(T));
    ::memcpy((void *)m_head, (const void *)&value, sizeof(T));
    m_head += sizeof(T);
  }

  AGX_FORCE_INLINE BytePtr ByteStream::popBytes(size_t numBytes)
  {
    agxVerify((size_t)(m_end-m_head) >= numBytes);
    BytePtr ptr = m_head;
    m_head += numBytes;
    return ptr;
  }


  AGX_FORCE_INLINE void ByteStream::write(const Byte *buffer, size_t numBytes)
  {
    agxVerify((size_t)(m_end-m_head) >= numBytes);
    ::memcpy((void *)m_head, (const void *)buffer, numBytes);
    m_head += numBytes;
  }

  AGX_FORCE_INLINE void ByteStream::write(const char *buffer, size_t numBytes)
  {
    this->write((const Byte *)buffer, numBytes);
  }

  AGX_FORCE_INLINE void ByteStream::read(Byte *buffer, size_t numBytes)
  {
    agxVerify((size_t)(m_end-m_head) >= numBytes);
    ::memcpy((void *)buffer, (const void *)m_head, numBytes);
    m_head += numBytes;
  }

  AGX_FORCE_INLINE void ByteStream::read(char *buffer, size_t numBytes)
  {
    this->read((Byte *)buffer, numBytes);
  }



  AGX_FORCE_INLINE ByteStream ByteStream::operator+ (size_t offset) const { return ByteStream(m_begin, m_end, m_head + offset); }
  AGX_FORCE_INLINE ByteStream& ByteStream::operator+= (size_t offset) { m_head += offset; return *this; }
}


#endif /* AGXDATA_BYTESTREAM_H */
