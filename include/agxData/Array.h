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

#ifndef AGXDATA_ARRAY_H
#define AGXDATA_ARRAY_H

#include <agxData/Data.h>
#include <agx/GlobalResult.h>
#include <agx/Vector.h>

#ifdef _MSC_VER
# pragma warning(push)
# pragma warning( disable : 4251 ) // class X needs to have dll-interface to be used by clients of class Y
#endif

namespace agxData
{

  /**
   * Arrays are fast accessors into the data, or a portion of it, held by a Buffer. The AbstractArray provides
   * common base class for the template Array classes.
   */
  class AGXCORE_EXPORT AbstractArray : public Data
  {
    public:

      /** Create a new invalid array. Reading elements through this array will fail. */
      AbstractArray();

      /** Create a new array that covers the entire given buffer. */
      AbstractArray(agxData::Buffer* buffer);

      /** Create a new array for the given buffer that covers the specified range. */
      AbstractArray(agxData::Buffer* buffer, agx::IndexRange32 range);

      /** Create a new array that covers a subset of another array. */
      AbstractArray(const agxData::AbstractArray& other, agx::IndexRange32 localRange);

      /// \return The number of elements accessible through the array.
      size_t size() const;

      /// \return True if size is zero.
      bool empty() const;

      /// \return The range of the back end buffer that this array covers.
      agx::IndexRange32& range();
      const agx::IndexRange32& range() const;

      /// \return The buffer that the array indexes into. Can be nullptr.
      const agxData::Buffer* buffer() const;
      agxData::Buffer* buffer();

      /// Set the buffer that the array will access, as well as the range of the buffer that is to be covered.
      void init(agxData::Buffer* buffer, agx::IndexRange32 range);

      /// Set the buffer that the array will access. The range is set to the entire buffer.
      void init(agxData::Buffer* buffer);

      /// Set to empty array
      void init();

      /// Set the range of the buffer that the array will cover.
      void setRange(size_t startIndex, size_t endIndex);

      /// Set the range of the buffer that the array will cover.
      void setRange(agx::IndexRange32 range);

      /// \return The format of the covered Buffer. Will return nullptr if the array is invalid.
      agxData::Format* getFormat();
      const agxData::Format* getFormat() const;

      /// Updates the internal data access pointer. Called when the back end buffer is reallocated.
      void sync();

      /// \return True if the array is valid, i.e., is bound to a back end buffer.
      bool isValid() const;

      /// Set the buffer while maintaining the current range, should normally not be used
      void _setBuffer(agxData::Buffer *buffer);

    protected:
      friend class SerializedFrame;
      agxData::Buffer* m_buffer;
      agx::IndexRange32 m_range;
  };



  /**
   * Type-specific Array used for fast access into the data held by a Buffer.
   */
  template <typename T>
  class Array : public agxData::AbstractArray
  {
    public:
      typedef T                                  Type;
      typedef T                                  value_type;
      typedef T*                                 pointer;
      typedef const T*                           const_pointer;
      typedef T&                                 reference;
      typedef const T&                           const_reference;
      typedef size_t                             size_type;
      typedef T*                                 iterator;
      typedef const T*                           const_iterator;
      typedef std::random_access_iterator_tag    iterator_category;
      typedef ptrdiff_t                          difference_type;

      static const bool IsScalar = false;

      /**
       * Array creation helper method that creates an array covering the given memory area. The created array will
       * not have a back end Buffer object and is therefore invalid. The given memory area will be interpreted
       * as an array of elements of type T.
       *
       * \param buffer Memory area that the array should index into.
       * \param size   The length, in number of elements, that the array should cover.
       * \return A new array that covers the given memory area.
       */
      static agxData::Array<T> raw(T* buffer, agx::Index size);
      static const agxData::Array<T> raw(const T* buffer, agx::Index size);

    public:
      /// Create a new invalid array.
      Array();

      /// Create a new array covering the entire given buffer.
      Array(agxData::Buffer* buffer);

      /// Create a new array covering the given range of the given buffer.
      Array(agxData::Buffer* buffer, agx::IndexRange32 range);

      /// Create a new array that covers a subset of another array.
      Array(const agxData::Array<T>& other, agx::IndexRange32 localRange);

      /// Create a new array view from a vector
      Array(agx::Vector<T>& vec);
      Array(agx::Vector<T>& vec, agx::IndexRange32 range);
      Array(agx::VectorPOD<T>& vec);
      Array(agx::VectorPOD<T>& vec, agx::IndexRange32 range);

      /// \return A reference to the last element of the array.
      T& back();
      const T& back() const;

      /// \return A reference to the first element of the array.
      T& front();
      const T& front() const;

      /// \return A reference to the element at the given index in the array.
      T& operator[] (size_t index);
      const T& operator[] (size_t index) const;

      /// \return A reference to the element at the given index in the array.
      T& at(size_t index);
      const T& at(size_t index) const;

      /// \return Create a new array that covers a subset of the current array.
      agxData::Array<T> slice(agx::IndexRange32 subRange) const;

      /// \return Create a new array that covers a subset of the current array.
      agxData::Array<T> operator[] (agx::IndexRange32 subRange) const;

      /// \return An iterator to the first element of the array.
      iterator begin();
      const_iterator begin() const;

      /// \return An iterator to one past the last element of the array.
      iterator end();
      const_iterator end() const;

      /// \return A pointer to the start of the memory area that the array covers.
      T* ptr();
      const T* ptr() const;

      /**
       * The index of the first occurrence of an element that is equal to the given element. If there is no such
       * element, then InvalidIndex is returned.
       *
       * \param element  The value to search for.
       * \return Index of the first match, or if there is no match InvalidIndex.
       */
      template <typename T2>
      size_t find(const T2& element) const;

      /// \return True if the array contains at least one element that is equal to the given element.
      template <typename T2>
      bool contains(const T2& element) const;


      //
      // GlobalResult functionality used by parallel kernels that create a dynamic amount of output data.
      // These are implemented in agxData/Buffer.h
      //

      /// \return The global result object associated with the buffer that the array is bound to.
      agx::GlobalResult* getGlobalResult();

      /// \see agx::GlobalResult::allocateResult
      agx::GlobalResult::Transaction allocateResult(size_t numElements);
      /// \see agx::GlobalResult::commitResult
      size_t commitResult(size_t numElements, const void* localResult, agx::Index sortIndex);
      size_t commitResult(size_t numElements, const void* localResult);

      /// \see agx::GlobalResult::commitResult
      template <typename T2>
      size_t commitResult(const T2& vector, agx::Index sortIndex);
      template <typename T2>
      size_t commitResult(const T2& vector);

  };


  /* Implementation */
  // See Buffer.h
}

#ifdef _MSC_VER
# pragma warning(pop)
#endif

AGX_TEMPLATED_TYPE_BINDING(agxData::Array, "Array")


#endif /* _AGXDATA_ARRAY_H_ */
