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


#ifndef AGXDATA_BINARYDATA_H
#define AGXDATA_BINARYDATA_H

#include <agx/Referenced.h>
#include <agx/Integer.h>
#include <agxData/ByteStream.h>

namespace agxData
{

  /**
   * A handle to a chunk of generic binary data. Support both data ownership and borrowing.
   */
  AGX_DECLARE_POINTER_TYPES(BinaryData);
  AGX_DECLARE_VECTOR_TYPES(BinaryData);
  class AGXCORE_EXPORT BinaryData : public agx::Referenced
  {
  public:

    enum DataAllocationType
    {
      DATA_IS_MALLOCED,  // m_data was allocated using malloc, will be deallocated using free.
      DATA_IS_NEWED,     // m_data was allocated using new, will be deallocated using delete.
      DATA_IS_ARRAY,     // m_data was allocated using new [], will be deallocated using delete [];
      DATA_IS_STATIC,    // m_data should not be deallocated at all. Not necessarily really static
                         // data, can also be used to create a BinaryData object that points into
                         // data owned by someone else. It is the responsibility of the user of this
                         // BinaryData to ensure that the memory hasn't been deallocated when the
                         // pointer returned by ptr() is used.
      DATA_IS_ALIGNED    // m_data was allocated using the static member functions of ByteAllocator.
    };


    BinaryData();

    // Allocates the requested size.
    BinaryData(size_t size);

    // Allocates the requested size on the specified alignment.
    BinaryData(size_t size, size_t alignment);

    /**
     * Create a binary data that is a handle to an externally allocated memory area. The given memory
     * area will be deallocated according to the instruction implied by the given 'allocationType'. To
     * avoid deallocation, pass DATA_IS_STATIC.
     *
     * \param allocationType  The way the BinaryData should treat allocation/deallocation of the data.
     * \param size            The size of the memory block, in bytes.
     * \param data            Pointer to the data. Pass 0 to have the constructor do the allocation.
     */
    BinaryData(Byte* data, size_t size, DataAllocationType allocationType);

    BinaryData(Byte* data, size_t size, agx::Referenced *reference);

    /**
     * Create a BinaryData that contains a copy of the given memory area.
     */
    static BinaryData *copy(const Byte* data, size_t size);

    /**
    Load binary data from a file.
    */
    static BinaryData *readFromFile(const agx::String& path);

    /**
    \return The size in bytes
    */
    size_t size() const;

    /**
    \return The generic data.
    */
    Byte* ptr();
    const Byte* ptr() const;

    // Templated
    template <typename T>
    T* ptr();

    template <typename T>
    const T* ptr() const;

    /**
    \return The data as a byte stream.
    */
    ByteStream getByteStream();
    const ByteStream getByteStream() const;


    /**
     Change the size that will be returned by calls to 'size()'. The memory buffer remains unchanged,
     so use with care.
     */
    void setApparentSize( size_t size );

  protected:
    ~BinaryData();

  private:
    Byte* m_data;
    size_t m_size;

    DataAllocationType m_allocationType;
    agx::ref_ptr<agx::Referenced> m_reference;
  };


  /* Implementation */
  AGX_FORCE_INLINE size_t BinaryData::size() const { return m_size; }

  AGX_FORCE_INLINE Byte* BinaryData::ptr() { return m_data; }
  AGX_FORCE_INLINE const Byte* BinaryData::ptr() const { return m_data; }
  AGX_FORCE_INLINE ByteStream BinaryData::getByteStream() { return ByteStream(m_data, m_data + m_size); }
  AGX_FORCE_INLINE const ByteStream BinaryData::getByteStream() const { return ByteStream(m_data, m_data + m_size); }

  template <typename T>
  AGX_FORCE_INLINE T* BinaryData::ptr() { return reinterpret_cast<T *>(m_data); }

  template <typename T>
  AGX_FORCE_INLINE const T* BinaryData::ptr() const { return static_cast<const T *>(m_data); }


}


#endif /* AGXDATA_BINARYDATA_H */
