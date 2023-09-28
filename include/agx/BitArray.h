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
#ifndef AGX_BITARRAY_H
#define AGX_BITARRAY_H

#include <agx/macros.h>
#include <stdlib.h>
#include <memory.h>

#ifdef _MSC_VER
# pragma warning(push)
# pragma warning(disable: 6011) // Disable warningC6011: dereferencing nullptr pointer
#endif


#ifndef CHAR_BIT
#define CHAR_BIT 8
#endif

// Calculate index into array for the specified bit
#define BITARRAY_BIT_INDEX(bit)         ((bit) / CHAR_BIT)

/* Calculates a bit mask to mask out in a byte the bit:th bit modulus 8*/
#define BITARRAY_BIT_MASK(bit) (1 << (bit % CHAR_BIT))

namespace agx
{
  DOXYGEN_START_INTERNAL_BLOCK()

  namespace {
#ifdef _MSC_VER
    // MS CL and gcc does not agree upon what is allowed.
    // Using non literals in const expression is just fone according to MS.
    template <size_t val>
    struct BitSize {
      static const unsigned short result = ( val / 8.0 - ( int )( val / 8.0 ) ) == 0 ? ( int )( val / 8.0 ) : ( int )( val / 8.0 ) + 1;
    };

#else

#if defined(__INTEL_COMPILER) || defined(__clang__)
    template <size_t val>
    struct BitSize {
      static const unsigned short result = ( val / 8.0 - ( int )( val / 8.0 ) ) == 0 ? ( int )( val / 8.0 ) : ( int )( val / 8.0 ) + 1;
    };

#else

    // On the other hand, gcc requires that the non-literal in an expression
    // must be outside of the template. So here we put it into a namespace.
    // This code generates zero-bytes on a MS compiler...
    double const BITS=8.0;

    template <size_t val>
    struct BitSize {
      static const unsigned short result;
    };

    template <size_t val>
    const unsigned short BitSize<val>::result = (unsigned short)(( val / BITS - ( int )( val / BITS ) ) == 0 ? ( int )( val / BITS ) : ( int )( val / BITS ) + 1);
#endif
#endif
  }

  DOXYGEN_END_INTERNAL_BLOCK()

  /**
  Class for stack-storage of a (at compile time) specified number
  of bits with a minimal size of storage.
  */
  template <size_t NUM_BITS>
  class BitArray
  {
    public:
      /// Constructor, initializes all bits to zero
      AGX_FORCE_INLINE BitArray() {
        memset( m_data, 0, NUM_BYTES );
      }

      /// Destructor
      ~BitArray() {}

      /**
      Set the n:th bit in the bitarray to \p flag
      \param n - select the n:th bit to be set
      \param flag - set the n:th bit to flag
      */
      AGX_FORCE_INLINE void set( size_t n, bool flag ) {
         if ( flag )
           m_data[n / CHAR_BIT] |= (char)(BITARRAY_BIT_MASK( n ));
         else
           m_data[n / CHAR_BIT] &= (char)(~BITARRAY_BIT_MASK( n ));
      }

      /**
      \return the value of the n:th bit
      */
      AGX_FORCE_INLINE bool get( size_t n ) const {
        return( ( m_data[BITARRAY_BIT_INDEX( n )] & ( 1 << (n % CHAR_BIT) ) ) != 0 );
      }

      /// Class which is a reference into an existing bitarray
      class reference
      {
        public:
          AGX_FORCE_INLINE reference( BitArray *array, size_t index ) {
            m_bitArray = array;
            m_index = index;
          }

          /**
          Cast operator to read a referenced value as a bool
          \return value of the referenced bit in the referenced bitarray
          */
          AGX_FORCE_INLINE operator bool() const {
            agxAssert1( m_bitArray, "Null reference to bitarray" );
            agxAssert1( m_bitArray->size() > m_index, "Out of bounds indexing of bitarray" );

            return m_bitArray->get( m_index );
          }

          /**
          Assignment of a bool value into the bitarray index
          */
          AGX_FORCE_INLINE reference& operator=( const bool val ) {
            agxAssert1( m_bitArray, "Null reference to bitarray" );
            agxAssert1( m_bitArray->size() > m_index, "Out of bounds indexing of bitarray" );

            m_bitArray->set( m_index, val );
            return *this;
          }

          /// Copy constructor
          inline reference( const reference& other ) {
            m_index = other.m_index;
            m_bitArray = other.m_bitArray;
          }

          /// assignment operator
          inline reference& operator=( const reference& other ) {
            if ( this == &other )
              return *this;

            m_index = other.m_index;
            m_bitArray = other.m_bitArray;

            return *this;
          }

        private:

          reference() {}

          BitArray *m_bitArray;        /* array index applies to */
          size_t m_index;             /* index of bit in array */
      };

      /// Class which is a reference into an existing bitarray
      class const_reference
      {
        public:
          AGX_FORCE_INLINE const_reference( const BitArray *array, size_t index ) {
            m_bitArray = array;
            m_index = index;
          }

          /**
          Cast operator to read a referenced value as a bool
          \return value of the referenced bit in the referenced bitarray
          */
          AGX_FORCE_INLINE operator bool() const {
            agxAssert1( m_bitArray, "Null reference to bitarray" );
            agxAssert1( m_bitArray->size() > m_index, "Out of bounds indexing of bitarray" );

            return m_bitArray->get( m_index );
          }

          /// Copy constructor
          inline const_reference( const const_reference& other ) {
            m_index = other.m_index;
            m_bitArray = other.m_bitArray;
          }

          /// assignment operator
          inline const_reference& operator=( const const_reference& other ) {
            if ( this == &other )
              return *this;

            m_index = other.m_index;
            m_bitArray = other.m_bitArray;

            return *this;
          }

        private:

          const_reference() {}

          const BitArray *m_bitArray;        /* array index applies to */
          size_t m_index;             /* index of bit in array */
      };

      AGX_FORCE_INLINE size_t size() const {
        return NUM_BITS;
      }
      AGX_FORCE_INLINE size_t sizeOf() const {
        return NUM_BYTES;
      }

      AGX_FORCE_INLINE reference operator[]( size_t bit ) {
        reference result( this, bit );
        return result;
      }

      AGX_FORCE_INLINE const_reference operator[]( size_t bit ) const {
        const_reference result( this, bit );
        return result;
      }

    protected:
      BitArray( const BitArray& ) {}
      BitArray& operator=( const BitArray& ) {
        return *this;
      }

    private:

      DOXYGEN_START_INTERNAL_BLOCK()

      enum Size
      {
        NUM_BYTES = int(BitSize<NUM_BITS>::result)
      };

      DOXYGEN_END_INTERNAL_BLOCK()

      char m_data[NUM_BYTES];
  };

#ifdef BITARRAY_BIT_MASK
#undef BITARRAY_BIT_MASK
#endif
} // namespace agx

#ifdef _MSC_VER
# pragma warning(pop)
#endif

#endif
