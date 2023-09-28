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

#include <iostream>
#include <vector>
#include <algorithm>
#include <string.h>
#include <stdlib.h>

#include <memory>
#include <agxWire/WireSolver/wireSolverTypes.h>
#include <agxWire/WireSolver/Diag4.h>
#include <agxWire/WireSolver/QPSolver.h>

DOXYGEN_START_INTERNAL_BLOCK()

#ifdef USE_IOH5
#include "ioh5.h"
#include <sstream>
#include <iomanip>
#endif
#ifdef USE_OCTAVE_IO
#include "octaveio.h"
#include <iostream>
#include <sstream>
#include <fstream>
#include <iomanip>
#endif

#if defined(_WIN32)
#define alloca _alloca
#endif

#include <agx/Vector.h>
namespace agxWire
{

  namespace WireSolver
  {


    typedef agx::Vector< std::shared_ptr<agx::RealValarrayPOD> >   table;

    /*
    This is to factorize symmetric or bisymmetric banded matrices with
    bandwidth W, allowing for factorization update and downdate and needed
    in an LCP solver.

    We store the diagonal and sub-diagonals.

    With this storage scheme, the elements i of column j below any
    diagonal are

    table[ i ]->[ j ],  i = 1, 2, ... , W -1,

    as long as j + i < N

    This means that element (I, J)  with I >= J is found at
    table[ I - J ]->[ J ],   with I - J < W,

    which are the same as the elements of row j to the right of diagonal.

    If we want the elements in column j above the diagonal then we access

    table[ i ]->[ j - i ],  i = 1, 2, ... , W - 1,


    We store the matrix in a vector of arrays so that the first array
    contains the diagonal, the second contains the first sub-diagonal, etc.
    */
    struct banded_matrix : public qp_matrix {

      agx::RealValarrayPOD negated;  /// which variables are negated
      table   data;
      std::shared_ptr<banded_matrix> original;       /// copy of original data


      void resize(size_t N, size_t W, bool bisym = false, bool orig = false);

      banded_matrix(size_t N, size_t W, bool bisym = false, bool orig = false);

      virtual ~banded_matrix();

      inline void negate(size_t i) { negated[i] = agx::Real(-1.0); }

      /**
      This will return the diagonal and sub-diagonal arrays.
      if i = 0, then you get the diagonal
      if i = 1, then you get the first sub-diagonal
      etc.
      */
      agx::RealValarrayPOD & get_subdiagonal(size_t i) { return  *data[i]; }


      // Boiler plate stuff for accessors and mutators to save typing
      inline size_t size()      const { return active.size(); }
      inline size_t last()      const { return active.size() - 1; }
      inline size_t bandwidth() const { return data.size(); }
      agx::RealValarrayPOD & operator[](size_t i) { return *original->data[i]; }

      /// \returns number of elements *below* the diagonal.
      inline size_t col_bandwidth(size_t j) const {
        return std::min(data.size(), size() - j);
      }

      /// Number of elements to the *left* of the diagonal
      inline size_t row_bandwidth(size_t i) const {
        return std::min(data.size(), i + 1);
      }

      /// saves a bit of typing
      inline double & diag_element(size_t i) {
        return (*data[0])[i];
      }

      /// saves a bit of typing
      inline double  diag_element(size_t i) const {
        return (*data[0])[i];
      }


      /// Element below the diagonal within the bandwidth.  No check performed.
      inline double & lower_element_raw(size_t i, size_t j) {
        return   (*data[i - j])[j];
      }

      ///
      inline double  lower_element_raw(size_t i, size_t j) const {
        return   (*data[i - j])[j];
      }

      ///
      inline double  lower_element_raw_original(size_t i, size_t j) const {
        return   (*(original->data[i - j]))[j];
      }

      /// fetch an element
      inline agx::Real elem(size_t i, size_t j) const {
        double r = 0;

        if (i >= j) {
          if (i - j < data.size()) {
            r = (*(original->data[i - j]))[j];
          }
        }
        else if (j - i < data.size()) {
          r = elem(j, i);
        }
        return r;
      }


      /**
      Assuming the matrix has been reset, we zero out a row and column of
      the matrix, relaxing the diagonal element with 1.
      */
      void delete_row_col(size_t k);

      /// Copy the *factored* coloum below the diagonal in a buffer
      void get_current_lower_column(size_t j, agx::Real * c) const;

      /// Copy the original column below the diagonal in a buffer
      void get_current_lower_column_original(size_t j, agx::Real * c) const;

      /// Utility for the factorization which is right looking.
      void update_column(size_t j);


      /**
      Utility: makes the code simpler to read.
      Note that the main diagonal contains 1/D in the LDLT factorization.
      */
      inline void update_diagonal(size_t j) {
        (*data[0])[j] = (double) 1.0 / (*data[0])[j];
        return;
      }


      /**
      Here we update column k after having processed column j.  This is at
      most a rank K update where K is the bandwidth.  This will touch the
      elements in k which are less than k-j below the diagonal
      */
      void rank_update_column(size_t j, size_t k, double *cj, double d);


      /**
      Combine lower triangular solve and multiplication with D inverse Here
      we walk along the columns.  In each column j we solve for b[ j ] and
      then update the b vector below it.  After that we multiply b[ j ]
      with the inverse diagonal element.
      */
      void forward_elimination(agx::RealValarrayPOD & x);

      ///
      void back_substitution(agx::RealValarrayPOD & x);

      /// Reset the  data to the original.
      void sync();

      /// \TODO: apparently unused.
      inline virtual bool keep_it(const int & left_set, const size_t& i) const {
        return (left_set == ALL) || (left_set == FREE &&  active[i]) || (left_set == TIGHT && !active[i]);
      }

      /**
      Implementation of pure virtuals in the base class.  Silly enough,
      all masking is done in the base class already and unused here.
      TODO: clean this up.

       y = alpha * y + beta * M * x;
      */
      virtual void multiply(
        const agx::RealValarrayPOD &x,
        agx::RealValarrayPOD &y,
        double alpha = 0.0,
        double beta = 1.0,
        int  /* left_set */ = ALL,
        int /* right_set */ = ALL);


      /**
      Apply left looking LDLT algorithm here without pivoting.  We keep 1/d
      on the diagonal.

      We perform a rank-k update where k here is the bandwidth so that the
      matrix is updated on the right and below the current point.

       The start variable is systematically ignored at this time, meaning
       that we factor the entire matrix.
       TODO: fix this to skip the part which does not need refactorization,
       i.e., all rows before start.  The issue here is to recompute the
       the parts of the rank updates from above start row which affect
       stuff below it, taking the masks into account.
      */
      virtual void factor(int /*start = -1*/);



      /// As declared in base class.
      virtual void solve(agx::RealValarrayPOD & b, size_t variable);

      /**
      Needed by the base class.
      \TODO: this is a duplication of previous code: diag_element
      */
      inline virtual double  get_diagonal_element(size_t i) const {
        return negated[i] * (*(original->data[0]))[i];
      }


      /// Fetch the column  'variable', both below and above the diagonal
      virtual void get_column(agx::RealValarrayPOD & v, size_t variable, agx::Real sign);

      /// As needed to implement bi-symmetry
      virtual void flip(agx::RealValarrayPOD & x) {
        for (size_t i = 0; i < negated.size(); ++i) {
          x[i] *= negated[i];
        }
      }

      /// Defined as pure virtual in base class but most likely not needed
      virtual void multiply_add_column(agx::RealValarrayPOD & v, agx::Real alpha, size_t variable);

      //
      // Utilities for octave.  These are defined in `diag4utils.h'
      //
#ifdef OCTAVE
      ///
      /// Reset the factor of matrix from data available in octave.
      ///
      virtual void read_factor(const octave_value_list& args, int arg);


      ///
      /// Write the data in the matrix to an octave struct.
      ///
      virtual void convert_matrix_oct(octave_scalar_map & st);


#endif

#ifdef USE_IOH5
      virtual void write_hdf5(H5::Group &g, std::string name = "banded_matrix") {

        h5::write_scalar(g, std::string("bandwidth"), 2 * original->data.size() - 1);
        h5::write_scalar(g, std::string("size"), original->data[0]->size());
        h5::write_vector(g, std::string("signs"), negated);

        for (size_t i = 0; i < original->data.size(); ++i) {
          std::stringstream band_name;
          band_name << std::string("band") << std::setw(4) << std::setfill('0') << i;
          h5::write_vector(g, band_name.str(), *original->data[i]);
        }


      }

      virtual void write_hdf5(std::string filename, std::string prefix = "matrix") {
        H5::H5File * file = h5::append_or_create(filename);
        H5::Group   g = h5::append_problem(file, prefix);
        write_hdf5(g, "banded_matrix");
        g.close();
        file->close();
      }


      virtual void read_hdf5(std::string filename, std::string prefix, int id) {};

      virtual void read_hdf5(H5::Group &g) {};

#endif
#ifdef USE_OCTAVE_IO
      virtual std::ostream &  write_octave(std::ostream & os, const std::string & name) {

        /// compute the *real* bandwidth, i.e., bands that aren't empty.
        size_t bw = 0;
        for (size_t i = 0; i < original->data.size(); ++i) {
          if (original->data[i]->size())
            ++bw;
        }

        octaveio::write_struct(os, name, 3);
        octaveio::write_vector(os, "signs", negated);
        octaveio::write_vector(os, "active", active);


        size_t N = original->data[0]->size();

        os << "# name: bands" << std::endl;
        os << "# type: matrix" << std::endl;
        os << "# rows: " << 2 * bw - 1 << std::endl;
        os << "# columns: " << N << std::endl;


        for (size_t i = bw - 1; i != 0; --i) {
          octaveio::write_vector_patched(os, *(original->data[i]), N, false);
        }
        for (size_t i = 0; i < bw; ++i) {
          octaveio::write_vector_patched(os, *(original->data[i]), N, true);
        }
        return os;
      }

      virtual void  write_octave(const std::string & filename, const std::string & name) {
        static int i = 0;
        std::stringstream real_name;
        real_name << filename << "_" << std::setw(4) << std::setfill('0') << i++;

        std::fstream fs;
        fs.open(real_name.str(), std::fstream::in | std::fstream::out | std::fstream::app);
        write_octave(fs, name);
        fs.close();
        return;

      }


      template <class T>
      std::ostream & write_cpp_array(std::ostream& os, const std::string & name, T & v) {

        for (size_t i = 0; i < v.size(); ++i) {
          os << name << "[ " << i << " ] = " << v[i] << ";" << std::endl;
        }
        return os;
      }


      virtual std::ostream & write_cpp(std::ostream& os, std::string name) {
        //  os << "namespace "  << name  << " { " << std::endl;

        os << "banded_matrix & matrix ()  {" << std::endl;
        os << " static banded_matrix matrix( " << size() << ", " << data.size() << ", " << bisymmetric << ");" << std::endl;
        write_cpp_array(os, "matrix.negated", negated);
        write_cpp_array(os, "matrix.active", active);
        for (size_t i = 0; i < original->data.size(); ++i) {
          agx::RealValarrayPOD & c = *original->data[i];

          os << "{" << std::endl;
          os << "std::valarray< double > & b = ( *matrix.original->data[ " << i << " ] ); " << std::endl;
          write_cpp_array(os, "b", c);
          os << "}" << std::endl;
        }
        return os << " return " << name << ";" << std::endl << "}" << std::endl;
      }



#endif


    };




    /**
    This extends any qp_matrix to have an extra row and column representing
    a global length constraint.


    As such, it reuses all the methods from diag for,
    but ads a small correction to them.   This does mean overloading all
    methods, but the amount of extra work is minimal.
    With this, one can have both curvature energy and a global length
    constraint.

    \TODO: biggest problem here is the semantic of the active variables.
    */
    struct banded_length : public qp_matrix {
      std::shared_ptr<qp_matrix>   M;               // base matrix: we are adding a row and
                                                    // column to this

      agx::RealValarrayPOD J0;       // original data: Jacobian + diagonal.
      agx::RealValarrayPOD J;        // This gets overwritten with inv(M) * J0
                            // to compute the Schur complement.
      agx::Real i_schur;                  // inverse of  the Schur complement
      bool active_last;             /// Whether or not the last variable is
                                    /// active
      agx::Real sign_last;

      inline size_t last() const {
        if (J0.size() == 0) {
          return std::numeric_limits<size_t>::max();
        }
        return J0.size() - 1;
      }

      virtual void flip(agx::RealValarrayPOD & x) {
        M->flip(x);
        x[last()] *= sign_last;
      }

      /// start with all empty
      banded_length();

      banded_length(qp_matrix * _M, bool _bisymmetric = false);

      banded_length(std::shared_ptr<qp_matrix>  _M, bool _bisymmetric = false);

      void resize(const std::shared_ptr<qp_matrix>&  _M, bool _bisymmetric = false);

      inline virtual size_t size() const { return J0.size(); }

      virtual void negate(size_t i) {
        if (i < M->size()) {
          M->negate(i);
        }
        else {
          sign_last = -agx::Real(1.0);
        }
      }


      ///
      virtual void multiply(
        const agx::RealValarrayPOD& b,
        agx::RealValarrayPOD& x,
        double alpha = 0.0,
        double beta = 1.0,
        int  /*left_set*/ = ALL,
        int /*right_set*/ = ALL);

      /**
      Just as the name says.  This is straight forward.  We cache
      inv(M)*J0(1:end-1) since that's used when solving.
      */
      virtual void get_schur_complement();

      ///
      virtual void factor(int variable = -1);


      virtual void solve(agx::RealValarrayPOD  & x, size_t start = 1) {

        double a = x[last()]; // save this to solve for separately
        x[last()] = 0;          // nix this for safety

        if (dirty)
          factor();

        M->solve(x, start);    // x is now  inv(M) * x

        if (active_last) {

          for (size_t i = 0; i < M->active.size(); ++i) {
            if (!M->active[i])
              x[i] = 0.0;
          }
          // This line below gives: G * inv(M) * x0, with x0 original x
          double gamma = std::inner_product(std::begin(J0), std::end(J0) - (size_t)1, std::begin(x), agx::Real(0.0));
          double alpha = i_schur * (a - gamma);
          // update x with the last variable
          x -= J * sign_last * alpha;
          //      x -= sign_last * alpha * J ;
          x[last()] = alpha;

        }
        else {
          x[last()] = a;
        }
      }

      /**
      Self-explanatory: used by LCP solver, and is *variable* based, not
      *block* based.  Subclasses have `blockwise'  version of these where
      the index is the index of a Lagrange multiplier, not just a variable.
      */
      virtual bool toggle_active(size_t i) {
        dirty = true;
        if (i < M->size())
          return M->toggle_active(i);
        else
          active_last = !active_last;
        return active_last;
      }

      virtual void set_active() {
        dirty = true;
        M->set_active();
        active_last = true;
      }

      virtual void set_active(size_t i, bool a) {
        dirty = true;
        if (i < M->size())
          M->set_active(i, a);
        else
          active_last = a;
      }

      /// self-explanatory: for LCP solver: variable based, not  block  based.
      virtual void set_active(const agx::IntValarrayPOD& idx) {
        dirty = true;
        for (size_t i = 0; i < M->size(); ++i)
          M->set_active(i, idx[i] <= 0);
        active_last = idx[idx.size() - 1] <= 0;
        return;
      }

      // What follows are operations needed for the LCP solver

      /**
      For the bisymmetric case, the alternate entries on the diagonal are
      negated in comparison to the un-symmetrized matrix.
      */
      inline virtual agx::Real get_diagonal_element(size_t i) const {
        if (i < last())
          return M->get_diagonal_element(i);
        else
          return  sign_last * J0[last()];
      }

      void get_column(agx::RealValarrayPOD & v, size_t variable, agx::Real sign);

      /**
      Multiply and add a column to a vector, making use of the fact that
      the columns are sparse here.
      */
      virtual void multiply_add_column(agx::RealValarrayPOD & w, agx::Real alpha, size_t variable);


      virtual void multiply_submatrix(const agx::RealValarrayPOD& b, agx::RealValarrayPOD& x,
        double alpha = 0.0, double beta = 1.0,
        int  left_set = ALL, int right_set = ALL);

      /**
      Here we solve for  v in M(active, active ) v = -M(active, variable)
      and also compute the  mtt  variable needed by the qp solver
      Because of the masks we need to apply, a temp copy is necessary.
      This essentially computes the `variable'  column of the inverse of the
      principal submatrix M(active, active)
      */
      virtual bool get_search_direction(size_t variable, agx::RealValarrayPOD & v, agx::Real & mtt);

      virtual bool is_active(size_t i) const {
        return (i < last()) ? M->is_active(i) : active_last;
      }

#ifdef OCTAVE
      // utility to workf from octave
      //  banded_length();
      virtual void convert_matrix_oct(octave_scalar_map & st) {};
      // utility to work from octave: sets the factor directly
      virtual void read_factor(const octave_value_list& args, int arg) {};
#endif


#ifdef USE_IOH5
      virtual void write_hdf5(std::string & filename, std::string prefix = "matrix") { };
      virtual void write_hdf5(H5::Group &g, std::string & name = "matrix") {};
      virtual void read_hdf5(std::string & filename, std::string & prefix, int id) {};
      virtual void read_hdf5(H5::Group &g) {};
#endif

#ifdef USE_OCTAVE_IO

      /**
      \todo The other write_octave previously hid the base class version of this
      method. That's a compiler warning.
      This implementation calls the base class version, which is what would
      have happened in some cases previously as well.
      May want to instead open an ofstream and call the other overload.
      */
      virtual void write_octave(const std::string& filename, const std::string& name)
      {
        return this->qp_matrix::write_octave(filename, name);
      }



      virtual std::ostream &  write_octave(std::ostream & os, const std::string & name)
      {
        octaveio::write_struct(os, name, 3);
        M->write_octave(os, "submatrix");
        octaveio::write_vector(os, "last_row", J0);
        octaveio::write_scalar(os, "sign", flip_sign);
        return os;
      }

      virtual std::ostream & write_cpp(std::ostream& os, std::string /*name*/) { return os; };
#endif
    };
  }
}


DOXYGEN_END_INTERNAL_BLOCK()
