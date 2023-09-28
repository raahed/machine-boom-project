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

#ifndef AGXSABRE_AGXSABRE_H
#define AGXSABRE_AGXSABRE_H

#include <set>

#include <agx/macros.h>
#include <agx/stdint.h>
#include <agx/Vector.h>

#include <agxSabre/export.h>


#define SABRE_DEBUG 0

#if defined(SABRE_DEBUG) && SABRE_DEBUG
#define SABRE_ANNOTATE( ... ) fprintf(stderr, __VA_ARGS__ )
#else
#define SABRE_ANNOTATE( ... )
#endif


/**
The agxSabre namespace contains classes and methods for working with sparse matrices.
*/
namespace agxSabre
{


  /**
  Different types of factorizers.
  Internal matrix/elimination tree analysis determines which type to use.
  */
  enum Factorizer {
    SERIAL,
    MULTIFRONTAL
  };


  /**
  */
  enum MatrixMethod {
    MATRIX_AUTO_SELECT_METHOD,
    MATRIX_RANK_UPDATE,
    MATRIX_PARTIAL_FACTOR
  };

  /**
  For tree traversal etc.
  */
  enum {
    UNVISITED = 0,
    SEEN = 1,
    VISITED = 2
  };


  // forward declarations of classes
  class Permutation;
  class Tree;


  typedef std::pair< agx::UInt32, agx::UInt32 > UInt32Pair;
  typedef agx::VectorPOD< agx::UInt32 >         UInt32Vector;
  typedef agx::VectorPOD< UInt32Pair >          UInt32PairVector;
  typedef std::set< agx::UInt32 >               UInt32Set;




  /**
  Generic template to handle size requirements and padding
  */
  template<typename T>
  inline size_t paddedSize( size_t i );

  template<>
  inline size_t paddedSize<float>( size_t i )
  {
    const size_t three = 3;
    return (i+three) & ~three;
  }

  template<>
  inline size_t paddedSize<double>( size_t i )
  {
    const size_t one = 1;
    return (i+one) & ~one;
  }

  template<typename T>
  T* alignPtr( T* ptr )
  {
    const size_t number = 31;
    return (T*)( (((size_t)(ptr))+number) & ~number );
  }


  AGX_FORCE_INLINE bool isZero( float f )
  {
    return ( f < FLT_EPSILON && f > -FLT_EPSILON );
  }

  AGX_FORCE_INLINE bool isZero( double d )
  {
    return ( d < DBL_EPSILON && d > -DBL_EPSILON );
  }


  inline bool pairLess( const UInt32Pair& a, const UInt32Pair& b )
  {
    return a.first < b.first;
  }

  inline bool pairGreater( const UInt32Pair& a, const UInt32Pair& b )
  {
    return a.first > b.first;
  }

}

#endif

