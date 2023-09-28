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

#ifndef AGXSABRE_METALAYOUT_H
#define AGXSABRE_METALAYOUT_H

#include <agxSabre/agxSabre.h>


#ifdef _MSC_VER
# pragma warning(push)
# pragma warning(disable: 4251) // warning C4251: class X needs to have dll-interface to be used by clients of class Y
#endif


namespace agxSabre
{

  /**
  Internal structure.

  Used for column access of row compressed data.
  */
  typedef struct AGXSABRE_EXPORT MetaLayout
  {
    UInt32Vector              colPointers;
    UInt32Vector              rowIndices;
    agx::VectorPOD<double* >  blkData;


    void configure( uint32_t N, uint32_t B );

    void clear();

    void print();

  } MetaLayout_t;


}

#ifdef _MSC_VER
#  pragma warning(pop)
#endif


#endif
