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

#include <agx/Vector.h>
#include <agx/NlmcpCallback.h>

namespace agx
{
  using NlmcpCallbacks = agx::Vector<NlmcpCallbackRef>;

  /**
  Data type to bind an index and ranges associated with it.
  */
  struct IndexedRangeReal
  {
    struct Trace
    {
      enum Buffer
      {
        BINARY,
        MANY_BODY,
        CONTACT,
        UNKNOWN
      };

      Buffer buffer;
      agx::UInt row;
    };

    IndexedRangeReal( agx::UInt idx, agx::RangeReal rr )
      : IndexedRangeReal( idx,
                          rr,
                          { Trace::UNKNOWN, agx::InvalidIndex } )
    {
    }

    IndexedRangeReal( agx::UInt idx, agx::RangeReal rr, Trace trace )
      : index( idx ),
        range( rr ),
        trace( trace )
    {
    }

    agx::UInt index;
    agx::RangeReal range;
    Trace trace;

    private:
      friend class Vector<IndexedRangeReal>;
      IndexedRangeReal()
        : IndexedRangeReal( agx::InvalidIndex,
                            RangeReal() )
      {
      }
  };

  /** Essentially a sparse array of bounds.*/
  class SparseRangeReal : public Vector<IndexedRangeReal>
  {
    public:
      /**
      Default constructor.
      */
      SparseRangeReal();

      /**
      Add nl-callback to receive callbacks during solves.
      \param callback - callback
      \param localRow - local start row of the non-linear bound
      \param numRows  - number of rows the nl-callback handles
      \return true if added - otherwise false
      */
      agx::Bool add( agx::NlmcpCallback* callback, agx::UInt localRow, agx::UInt numRows );

      /**
      Begin scope where NlmcpCallback instances are added given
      permuted matrix row. When a new callback is added the local
      row is given which will be an offset of permutedMatrixRow.
      */
      void beginNlAddScope( agx::UInt permutedMatrixRow, agx::UInt iterativeRow );

      /**
      End 'add new nl-callbacks' scope. It's not valid to add
      callbacks after calling this method.
      \return number of added callbacks
      */
      agx::UInt endNlAddScope();

      /**
      \return current scope permuted matrix start row - if not in a scope, InvalidIndex is returned.
      */
      agx::UInt getScopePermutedMatrixRow() const;

      /**
      \return current scope iterative start row - if not in a scope, InvalidIndex is returned.
      */
      agx::UInt getScopeIterativeRow() const;

      /**
      \param row - permuted matrix row
      \param searchDepth - number of callbacks to check from back of array
      \return true if the given row is bounded by a nl-callback - otherwise false
      */
      agx::Bool isBoundedByNlCallback( agx::UInt row, agx::UInt searchDepth ) const;

      /**
      Sort given 'index' in IndexedRangeReal and updates indices
      of added nl-callbacks.
      */
      void sort();

      /**
      \return the added nl-callbacks
      */
      const NlmcpCallbacks& getCallbacks() const;

    public:
      /** write the array to an hdf5 stream, here disguised as the void * pointer  */
      int save_hdf5(void *group, const agx::String & name) const;
      /** write the array to file in hdf5 format, with given name. */
      int save_hdf5(const agx::String & filename , const agx::String & name) const;
      /** read  the array from an hdf5 stream, here disguised as the void * pointer  */
      int load_hdf5(void *group, const agx::String & name);
      /** read the array to file in hdf5 format, with given name. */
      int load_hdf5(const agx::String & filename , const agx::String & name);

    private:
      struct NlAddScopeData
      {
        NlAddScopeData()
          : directRow( agx::InvalidIndex ),
            iterativeRow( agx::InvalidIndex ),
            numCallbacks( agx::InvalidIndex )
        {
        }

        agx::UInt directRow;
        agx::UInt iterativeRow;
        agx::UInt numCallbacks;
      };

    private:
      agx::NlmcpCallbacks m_nlCallbacks;
      NlAddScopeData m_nlAddScopeData;
  };
}
