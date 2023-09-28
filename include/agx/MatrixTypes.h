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

#ifndef MATRIX_TYPES_H
#define MATRIX_TYPES_H

namespace agx
{

  /**
  A class to distinguish between various types of matrices, both in terms
  of storage properties, i.e., dense or sparse, or lower or upper triangular
  in the case of symmetric matrices.   These storage types should include
  banded storage in the future.

  The matrix type itself refers to mathematical properties, such as
  whether or not it is square, symmetric, positive definite, or a saddle
  point matrix.

  The set functions here all take a boolean argument which defaults to
  true, and they return a reference to *this.  This means that one can
  do things like
  t.setSaddlePoint().t.setPositiveDefinite(false).setDense(false).setUpperTriangle();

  The MatrixType class holds little data and that is directly on the
  stack in any case.  Thus, it is possible to create such objects on the
  fly inside constructors of matrices.  For instance, one could do
  int n = 100;
  SparseMatrix m(n, MatrixType().setSaddlePoint().t.setPositiveDefinite(false).setDense(false).setUpperTriangle());
  to get a square sparse matrix of saddle point type stored as upper
  triangle only.  In this example, the default solver library would be
  used.  The class SolverLibs is defined in another file because it is
  part of our own specific matrix library class hierarchy.
  */
  class MatrixType  {
    public:

      /**
      How the matrix is stored.  If not dense, then, it is sparse.
      Dense or sparse, it is possible that only upper or lower triangles
      are stores, as is the case for symmetric matrices.
      */
      enum Storage {
        DENSE           =  2,       /**< The entire matrix is stored, or it is sparse. */
        UPPER_TRIANGLE  =  4,       /**< Only upper triangle is stored, dense or sparse */
        LOWER_TRIANGLE  =  8        /**< Only lower triangle is stored, dense or sparse */
      };

      /**
      The kind of matrix we are dealing with.   Users of these classes are
      not to worry much about these enums since the inline methods below do the
      work.
      */
      enum Type {
        GENERIC           = 0,      /**< No special property. */
        SQUARE            = 2,
        SYMMETRIC         = 4,      /**< Symmetric matrices. */
        POSITIVE_DEFINITE = 8,      /**< Positive definite matrices */
        SADDLE_POINT      = 16      /**< Saddle point matrices, which may be
                                         symmetric or not, and positive definite
                                         or not. */
      };

      MatrixType() : m_storage(DENSE), m_type(GENERIC) { }


      MatrixType(int type, int storage) {
        m_storage = storage;
        m_type = type;
      }

      MatrixType & setSymmetric (bool t = true) {
        (t)?  m_type |= SYMMETRIC
          : m_type &= ~SYMMETRIC;
        return setGeneric(false);
      }

      MatrixType & setSquare (bool t = true) {
        (t)?  m_type |= SQUARE
          : m_type &= ~SQUARE;
        return setGeneric(false);
      }

      MatrixType & setSaddlePoint(bool t = true) {
        (t)?  m_type |= SADDLE_POINT
          : m_type &= ~SADDLE_POINT;
        return setGeneric(false).setSquare(true);
      }

      MatrixType & setPositiveDefinite (bool t = true) {
        (t)?  m_type |= POSITIVE_DEFINITE
          : m_type &= ~POSITIVE_DEFINITE;
        return setGeneric(!t).setSquare(true);
      }

      MatrixType & setGeneric (bool t = true )  {
        if (t) {  m_type = GENERIC ; }
        return *this;
      }

      MatrixType &setDense(bool t = true ) {
        (t)? m_storage |= DENSE
          : m_storage &= ~DENSE;
        return *this;
      }

      MatrixType &setSparse(bool t = true ) {
        return setDense(!t);
      }

      MatrixType &setUpperTriangle(bool t = true ) {
        (t)? m_storage |= UPPER_TRIANGLE
          : m_storage &= ~UPPER_TRIANGLE;
        return setLowerTriangle(!t);
      }

      MatrixType &setLowerTriangle(bool t = true ) {
        (t)? m_storage |= LOWER_TRIANGLE
          : m_storage &= ~LOWER_TRIANGLE;
        return this->setSquare(true);
      }

      bool isGeneric() const { return m_type == GENERIC ; }
      bool isPositiveDefinite () const { return (m_type&POSITIVE_DEFINITE) != 0; }
      bool isSaddlePoint () const { return (m_type&SADDLE_POINT) != 0; }
      bool isSymmetric () const { return (m_type&SYMMETRIC) != 0; }
      bool isDense() const { return (m_storage&DENSE) != 0; }
      bool isSparse() const { return ! isDense() ; }
      bool isUpperTriangle() const { return (m_storage&UPPER_TRIANGLE) != 0; }
      bool isLowerTriangle() const { return (m_storage&LOWER_TRIANGLE) != 0; }
      bool isSquare() const { return (m_type&SQUARE) != 0; }

    private:
      int m_storage;
      int m_type;
  };

}
#endif
