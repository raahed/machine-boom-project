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

#ifndef AGX_SPARSE_INVERSE_H
#define AGX_SPARSE_INVERSE_H

#include <agx/agx_valarray_types.h>
#include <agx/IndexSet.h>
#include <agx/SparseMatrix.h>
#include <map>

#include <agxSabre/DenseTypes.h>

#ifdef _MSC_VER
# pragma warning(push)
// Disable warning about assignment operator could not be generated due to reference members.
#pragma warning( disable: 4512 )
#endif


namespace agx
{

  /**
     SparseInverse holds a subset of the columns of the inverse of a given
     sparse matrix.  To compute these columns, we simply call the solve(x)
     method of the matrix object with x[i] = 0 for i!=j, and x[j] = 0.

     The columns are stored as SparseVector so only nonzero elements are
     considered.  In addition, only a subset of indices are considered at
     all, those contained in the SparseRangeReal object.



   The columns are held in an std::map so they are
   sorted according to their index  in increasing order.  This makes it
   easy to find out if a given column is present or if need
   */


    /**
        An entry associates a real value with an index.
    */

  class RealEntry : public std::pair<int, Real> {
    public:
    RealEntry() : std::pair<int, Real>() {}
    RealEntry(const int& i, const Real &x) : std::pair<int, Real> (i, x) {}
    bool operator< (const RealEntry & e) { return first < e.first ; }
    bool operator== (const RealEntry & e) { return first == e.first ; }
    bool operator<= (const RealEntry & e) { return first <= e.first ; }
  };

  /**
       A sparse vector here is an Vector of entries with a few
       additional operations for creation and sorting.
  */

  class SparseVector : public Vector<RealEntry> {
    public:
    SparseVector() {}
    SparseVector(const RealValarray &x, const SparseRangeReal &bounds, Real threshold = 0 );
    void extract_subvector(const IndexSet::IxSet& s, SparseVector &V ) const;
    };


  class SparseInverse {
    private:
      typedef std::map<UInt,  SparseVector >  ISVMap;
      ISVMap m_map;
      void update_inverse( SparseMatrix& sparseMatrix, const IndexSet::IxSet& s, const SparseRangeReal& bounds, const RealValarray& signs );
      void build_submatrix( SparseMatrix& sparseMatrix, const IndexSet::IxSet &s, const SparseRangeReal& bounds, agxSabre::DenseMatrix<agx::Real> &m, const RealValarray& signs );
    public:
      SparseInverse() {}
      ~SparseInverse();

      void clear();
      void solve_subproblem( SparseMatrix& sparseMatrix, const IndexSet::IxSet& s, const SparseRangeReal& bounds, RealValarray &b, const RealValarray& signs );
  };

}

#ifdef _MSC_VER
# pragma warning(pop)
#endif

#endif
