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

#ifndef SABRE_MARKS_H
#define SABRE_MARKS_H

#include <agxSabre/export.h>
#include <algorithm>
#include <cstring>

namespace agxSabre
{

  /**
  Class for holding char marks (and stack for fast clear).
  */
  class AGXSABRE_EXPORT Marks
  {
    public:

      /**
      Constructor
      */
      inline Marks();

      /**
      Constructor
      */
      inline Marks( unsigned int size );

      /**
      Destructor
      */
      inline ~Marks();

      /**
      Assignment operator
      */
      Marks& operator= (const Marks& rhs);

      /**
      Read only []-operator
      */
      const char& operator[]( size_t index ) const;


      /**
      \return Size of the marks vector
      */
      inline unsigned int getSize() const;

      /**
      Resize the marks.
      */
      inline void setSize( unsigned int size );


      /**
      \return Pointer the actual marks
      */
      inline const char* getMarks() const;

      /**
      */
      inline const unsigned int* getMarkIndices() const;

      /**
      \return Number of marked elements
      */
      inline unsigned int getNumMarks() const;

      /**
      Place a mark on index idx
      */
      inline void mark(unsigned int idx);

      inline void mark(unsigned int idx, char m );

      /**
      Changes the mark on a index
      */
      inline void remark( unsigned int idx, char m );

      /**
      \return true if element idx is marked
      */
      inline bool isMarked(unsigned int idx) const;

      /**
      Clear the marks using the stack
      */
      inline void clear();


      /**
      Clears all the marks
      */
      inline void wipe();

      /**
      Sorts the index-vector used for marks
      */
      inline void sort();

    private:
      Marks( const Marks& /*other*/ ) : m_marks(0), m_stack(0),m_numMarked(0),m_size(0),m_allocatedSize(0) {}

      char* m_marks;
      unsigned int* m_stack;

      unsigned int m_numMarked;
      unsigned int m_size;
      unsigned int m_allocatedSize;


      inline void allocateMemory(unsigned int size);
      void releaseMemory();
  };



  inline Marks::Marks()
    : m_marks(0), m_stack(0), m_numMarked(0), m_size(0), m_allocatedSize(0)
  {

  }

  inline Marks::Marks(unsigned int size)
    : m_marks(0), m_stack(0), m_numMarked(0), m_size(size), m_allocatedSize(0)
  {
    allocateMemory( m_size );
  }

  inline Marks::~Marks()
  {
    releaseMemory();
  }

  inline Marks& Marks::operator= (const Marks& rhs)
  {
    if ( &rhs == this )
      return *this;

    setSize( rhs.getSize() );
    m_numMarked = rhs.m_numMarked;

    memcpy( m_marks, rhs.m_marks, sizeof(char) * rhs.getSize() );
    memcpy( m_stack, rhs.m_stack, sizeof(int) * rhs.getSize() );

    return *this;
  }


  inline const char& Marks::operator[]( size_t index ) const
  {
    return m_marks[index];
  }

  inline unsigned int Marks::getSize() const
  {
    return m_size;
  }

  inline void Marks::setSize( unsigned int size )
  {
    if ( size > m_allocatedSize )
    {
      releaseMemory();
      allocateMemory( size );
    }

    m_size = size;
  }


  inline const char* Marks::getMarks() const
  {
    return m_marks;
  }


  inline const unsigned int* Marks::getMarkIndices() const
  {
    return m_stack;
  }


  inline unsigned int Marks::getNumMarks() const
  {
    return m_numMarked;
  }

  inline void Marks::mark(unsigned int idx)
  {
    m_stack[ m_numMarked++ ] = idx;
    m_marks[idx] = 1;
  }

  inline void Marks::mark(unsigned int idx, char m )
  {
    m_stack[ m_numMarked++ ] = idx;
    m_marks[idx] = m;
  }


  inline void Marks::remark( unsigned int idx, char m )
  {
    if ( !isMarked(idx) )
      mark(idx,m);
    else
      m_marks[idx] = m;
  }


  inline bool Marks::isMarked(unsigned int idx) const
  {
    return m_marks[idx] != 0;
  }

  inline void Marks::clear()
  {
    while( m_numMarked > 0 )
    {
      m_marks[ m_stack[ --m_numMarked ] ] = 0;
    }
  }


  inline void Marks::wipe()
  {
    if ( m_marks )
      memset( m_marks, 0, m_allocatedSize * sizeof(char) );

    m_numMarked = 0;
  }


  inline void Marks::sort()
  {
    std::sort(m_stack, m_stack+m_numMarked);
  }


  inline void Marks::allocateMemory(unsigned int size)
  {
    m_allocatedSize = size;

    m_marks = new char[ m_allocatedSize ];
    m_stack = new unsigned int[ m_allocatedSize ];

    memset(m_marks, 0, sizeof(char) * m_allocatedSize);

    m_numMarked = 0;
  }



}


#endif

