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

#pragma once

#ifdef _MSC_VER
#  pragma warning(disable: 4714) // Disable warnings about not able to __forceinline
#endif

#include <agx/Vector.h>
#include <agx/agx_valarray_types.h>

namespace agx
{
  class SparseRangeReal;

  class IndexSet
  {
    public:
      struct Ipair
      {
        Ipair( UInt globalIndex, UInt localIndex ) : global( globalIndex ), local( localIndex ) {}
        UInt global;
        UInt local;
      };
      typedef agx::Vector< Ipair > IxSet;

      typedef IntValarray IVector;
      typedef BoolValarray BVector;

    public:
      enum IndexType
      {
        EQUALITY     = -1,
        FREE         = 0,
        LOWER        = 2,
        UPPER        = 4,
        IGNORE_INDEX = 8
      };

    public:
      IndexSet();
      IndexSet( agx::UInt n );

      ~IndexSet();

      void resize( agx::UInt n );

      void reset( const agx::SparseRangeReal& bounds );

      void update( const agx::SparseRangeReal& bounds );

      const agx::IndexSet::IxSet& getAtBounds() const;

      const agx::IndexSet::IVector& getFull() const;
      const agx::IndexSet::IVector& getIndexSet() const;

      agx::Int& operator() ( agx::UInt i );
      agx::Int operator() ( agx::UInt i ) const;

      bool hasIgnoredIndices() const;

      uint32_t hash();

    private:
      IVector m_full;
      IVector m_ix;
      mutable IxSet m_bag;
  };

  AGX_FORCE_INLINE Int& agx::IndexSet::operator()( UInt i )
  {
    return m_ix[i];
  }

  AGX_FORCE_INLINE Int agx::IndexSet::operator()( UInt i ) const
  {
    return m_ix[i];
  }
}
