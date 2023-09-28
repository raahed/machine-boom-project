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

#ifndef AGX_SPARSEMATRIXREPSABRE_H
#define AGX_SPARSEMATRIXREPSABRE_H

#include <agx/config/AGX_USE_HDF5.h>
#include <agx/SparseMatrix.h>
#include <agxSabre/Sabre.h>
#include <agxSabre/SabreData.h>

namespace agx
{

  class SparseMatrixRepSabre : public SparseMatrixRep
  {
    public:
      // constructor
      SparseMatrixRepSabre();

      // Optional constructor that takes compressed matrix as input
      SparseMatrixRepSabre( size_t* blkSizes, size_t numBlocks,  size_t i, size_t j, size_t nnz, int* r, int* c, double* data );

      // destructor
      virtual ~SparseMatrixRepSabre();

      virtual agx::Real* getBlock(int i, int j, bool& writeTransposed, unsigned int& ld) override;

      virtual void reset() override;

      virtual void factor() override;
      virtual void solve( agx::Real* ) override;
      virtual size_t unitSolve( agx::Real*, int startRow = -1 ) override;

      virtual void multiply( const agx::Real* x, agx::Real* r ) override;
      virtual void multiplyAdd( const agx::Real* x, agx::Real* r ) override;
      virtual void multiplyAdd( const agx::Real& alpha, const agx::Real* x, const agx::Real& beta, agx::Real* r ) override;

      virtual void setLogicalStructure( size_t n, const unsigned int* pointers, const unsigned int* indices, const unsigned int* dimensions, const int* flags, unsigned int numThreads ) override;

      virtual agx::Bool getRow( agx::UInt rowNumber, const agx::RealValarray& signs, agx::RealValarray& retRow ) const override;
      virtual agx::Bool getColumn( agx::UInt colNumber, agx::RealValarray& retCol, agx::Bool& signFlip ) const override;

      virtual agx::UInt getBlockPermutedStartRowIndex( agx::UInt blockIndex ) const override;
      virtual agx::UInt permuteRowIndex( agx::UInt rowIndex ) const override;
      virtual agx::UInt invPermuteRowIndex( agx::UInt permutedRowIndex ) const override;

      virtual unsigned int getNumRows() const override;
      virtual unsigned int getNumColumns() const override;

      virtual void print() const override;

      virtual void setNumBodies(size_t num) override;

      #if AGX_USE_HDF5()
        virtual bool saveH5(void* group, const agx::String& name ) const override;
        virtual bool loadH5(void* group, const agx::String& name ) override;
      #endif

      virtual agx::Real getFactorTime() const override { return m_data.getFactorTime(); }
      virtual agx::Real getSolveTime() const override { return m_data.getSolveTime(); }

      agxSabre::SabreData<agx::Real>& getSabreData() { return m_data; }
      const agxSabre::SabreData<agx::Real>& getSabreData() const { return m_data; }

    private:
      agxSabre::Sabre m_sabre;
      agxSabre::SabreData<agx::Real> m_data;
  };

}

#endif
