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

#ifndef AGXSABRE_SCRATCHPAD_H
#define AGXSABRE_SCRATCHPAD_H


#include <agx/agx.h>
#include <agx/HashTable.h>
#include <agxSabre/export.h>

namespace agxSabre
{

  /**
  ScratchPad holds aligned memory of different types (int, float, etc)
  and tries to minimize repeated calls to new/delete.

  There is no safety:
    - Memory is NOT cleard between calls.
    - Memory does not have to be returned, it is assumed not to be
      used when next get-call with same tag number is performed.
    - Requesting same tag with different sizes can cause the previous pointer
      to be invalidated.

  */
  class AGXSABRE_EXPORT ScratchPad
  {
    public:

      ScratchPad();
      ~ScratchPad();

      template<typename T>
      T*      getTArray( size_t tag, size_t size);

      unsigned char* getUCharArray(size_t tag, size_t size);
      int*           getIntArray(size_t tag, size_t size);
      float*         getFloatArray(size_t tag, size_t size);
      double*        getDoubleArray(size_t tag, size_t size);

      void deallocate();

    private:

      template< typename T >
      struct Allocation {
        T* m_ptr;
        T* m_aligned;
        size_t m_size;
        size_t m_tag;
      };

      typedef agx::HashTable< size_t, Allocation<unsigned char> > UCharTable;
      typedef agx::HashTable< size_t, Allocation<int> >    IntTable;
      typedef agx::HashTable< size_t, Allocation<float> >  FloatTable;
      typedef agx::HashTable< size_t, Allocation<double> > DoubleTable;

      IntTable    m_intData;
      UCharTable  m_uCharData;
      FloatTable  m_floatData;
      DoubleTable m_doubleData;

  };


  template<>
  AGX_FORCE_INLINE int*  ScratchPad::getTArray( size_t tag, size_t size)
  {
    return getIntArray(tag,size);
  }


  template<>
  AGX_FORCE_INLINE float*  ScratchPad::getTArray( size_t tag, size_t size)
  {
    return getFloatArray(tag,size);
  }


  template<>
  AGX_FORCE_INLINE double*  ScratchPad::getTArray( size_t tag, size_t size)
  {
    return getDoubleArray(tag,size);
  }



}

#endif

