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

#define USE_OCTAVE_IO


#ifdef USE_IOH5
#include "ioh5.h"
#endif
#ifdef USE_OCTAVE_IO
#include "octaveio.h"
#include <iostream>
#endif

#include <agxWire/WireSolver/wireSolverTypes.h>

#include <numeric>

#include <iostream>
#include <vector>
//#include <chrono>
//#include <ctime>
#include <functional>
#include <assert.h>
#include <valarray>

/// needed for linking with octave
#ifdef OCTAVE
class ColumnVector;
class octave_value_list;
class octave_scalar_map;
#endif

DOXYGEN_START_INTERNAL_BLOCK()


namespace agxWire {
  namespace WireSolver {


    /**
    *
    * Block LDLT decomposition.
    * All blocks are 2x2
    * Diagonal blocks have one of the two form before factorization:
    * [a 1]      or  [a 0]
    * [1 e]          [0 1]
    *
    *  These account for mass and friction and the Jacobian is just 1.
    *  a = mass + (h^2/4) * ( k1 + k2)
    *  where k1,k2 are the left and right spring constant, i.e.,
    *  tension / length
    *
    *  The off-diagonal blocks are block-subdiagonal and have the form
    *
    *  [b 0]  before and     [b f]  after factorization
    *  [0 0]                 [0 0]
    *
    *  These account for elasticity and b  = (h^2/4)k  where k is the spring
    *  constant from the coupling to the *previous* particle.
    *
    *  We compute and store the inverse of the diagonal blocks
    *  [e f]   =   ( 1 / (e*h - g*h) ) * [ h -g]
    *  [g h]                             [-f  e]
    *
    *  We don't factor in place since we need to update and down-date this
    *  factorization by changing between one of the two forms of the diagonal
    *  block.
    *
    *  In order to reset if needed, one must down-date and then update the
    *  first equation.
    *
    *
    * Assumption made here is that we are simulating a wire which is anchored
    * at both ends.  Each particle is linked to the previous and next one with
    * potentially different spring constants.  These are tension/length in
    * practice.
    *
    * NOTE: One could unhook the first and last particle but the code then
    * needs small revision to insert a switch to activate or deactivate the
    * anchor.
    */
    struct qp_matrix {

      /** Enumeration of the states of variables as needed in the LCP solver.
      This is intrusive but yet needed for various operations best
      encapsulated here than in the qp class itself, e.g., `solve principal
      subproblem'  for instance.

      EQ    is for variables which have both infinite upper and lower bounds
      FREE  is for variables which are not currently at their bound
      TIGHT is for variables currently clamped at one of their bounds
      UPPER is for variables currently clamped at their upper bound
      LOWER is for variables currently clamped at their lower bounds
      */
      enum states :int { EQ = -1, FREE = 0, LOWER = 2, TIGHT = 2, UPPER = 4, ALL = 6 };
      bool bisymmetric;             /// \brief whether we solve the bisymmetric problem



      bool dirty;                   /// \brief whether data was changed and matrix
                                    /// needs re-factorization.



                                    /** NOTE: we use symmetric representation of bisymmetric matrices since
                                    that's all the LCP solver can handle, but symmetric representation is
                                    more efficient. */
      agx::Real  flip_sign;         /** Negative for the bisymmetric case.

                                        Flipping signs is delegated to derived class since formats can vary:
                                        variables corresponding to Lagrange multipliers have their signs
                                        flipped. */


      virtual void negate(size_t i) = 0; /** \brief Sets which variables
                                         have to be negated to account
                                         for bi-symmetry */

      virtual void flip(agx::RealValarrayPOD & x) = 0; /** \brief Flips the signs of all variables which need it

                                              Whether or not a variable is active, i.e., the variable is free (solved
                                              for) or not.  Lagrange multiplier variables are the only ones that
                                              switch on and off as decided by the LCP solver. */

      agx::BoolValarrayPOD active;
      agx::RealValarrayPOD tmp_sub;    //! \brief a temporary array for multiply and solve
      agx::RealValarrayPOD tmp_search; //! \brief  a temporary array for the search  direction method





      /**
      Assign bisymmetric state and allocate memory
      for the active state as well as tmp arrays.
      */
      qp_matrix(size_t _n, bool b = false);

      /// Minimal constructor: only bisymmetric state set
      qp_matrix(bool b = false);

      /// Reset to initial state with a given size.
      void resize(size_t _n, bool b);

      virtual ~qp_matrix();

      /// \returns number of variables
      virtual size_t size() const { return active.size(); }

      /**
      Self-explanatory: used by LCP solver, and is *variable* based, not
      *block* based.  Subclasses have `blockwise'  version of these where
      the index is the index of a Lagrange multiplier, not just a variable.
      */
      virtual bool toggle_active(size_t i)
      {
        dirty = true;
        active[i] = !active[i];
        return active[i];
      }

      virtual bool is_active(size_t i) const { return active[i]; }

      virtual void set_active()
      {
        dirty = true;
        active = true;
      }

      virtual void set_active(size_t i, bool a)
      {
        dirty = true;
        active[i] = a;
      }

      /// self-explanatory: for LCP solver: variable based, not  block  based.
      virtual void set_active(const agx::IntValarrayPOD& idx);

      /**
      Factorize or update/down-date a factorization if  variable
      is specified.  If so, the variable will toggle between active and
      inactive, and the factorization will restart there, updating what's
      already done.
      */
      virtual void factor(int variable = -1) = 0;

      /// Fetch the column  variable
      virtual void get_column(agx::RealValarrayPOD & v, size_t variable, agx::Real sign) = 0;

      ///
      virtual void multiply_add_column(agx::RealValarrayPOD & v, agx::Real alpha, size_t variable) = 0;


      /**
      Solve a linear system of equations. The  start  argument
      indicates the position of the first non-zero argument.
      \TODO: why is this 1 by default?
      */
      virtual void solve(agx::RealValarrayPOD  & x, size_t start = 1) = 0;


      /**
      Standard multiply add:
      x = alpha * x + beta * M * b
      */
      virtual void multiply(
        const agx::RealValarrayPOD& b,
        agx::RealValarrayPOD& x,
        double alpha = 0.0,
        double beta = 1.0,
        int  left_set = ALL,
        int right_set = ALL) = 0;


      /// Needed by LCP solver
      virtual agx::Real get_diagonal_element(size_t i) const = 0;

      /**
      Multiply add operations with sub-matrices, given a set of active and
      inactive variables.  The following four cases are covered.

      In all cases, the other variables in x are left untouched

      To do this, we zero out the part of b that isn't wanted.  This
      however will still write on x in unwanted places, i.e., for the first
      case, if we set b(~active) = 0, then we will write over x(~active)
      which we don't want.
      Therefore, we cache x as well and restore the parts we overwrite
      x( active ) = alpha * x( active ) + beta * M( active, active ) * b( active )

      x( active ) = alpha * x( active ) + beta * M( active, ~active ) * b( ~active )

      x(~active)  = alpha * x(~active) + beta * M(~active, active ) b( active )

      x(~active)  = alpha * x(~active) + beta * M(~active, ~active ) b( ~active )

      We use two bool's to control whether we use `active'  or `!active',
      one on each side.  `true'  corresponds to `active', `false' to
      `!active'

      multiply-add is delegated to the derived class, but the outer logic
      is independent of that.
      */
      virtual void multiply_submatrix(
        const agx::RealValarrayPOD& b,
        agx::RealValarrayPOD& x,
        double alpha = 0.0,
        double beta = 1.0,
        int  left_set = ALL,
        int right_set = ALL);



      /**
      Here we solve for  v in M(active, active ) v = -M(active, variable)
      and also compute the  mtt  variable needed by the qp solver
      Because of the masks we need to apply, a temp copy is necessary.
      This essentially computes the `variable'  column of the inverse of the
      principal submatrix M(active, active)
      */
      virtual bool get_search_direction(size_t variable, agx::RealValarrayPOD & v, agx::Real & mtt);

      /**
      Given a search direction as previously computed and a steplength
      determined by the LCP solver, we move variables according to rules
      defined below.

      For a given step length `step', the z( driving ) variable moves by
      z( driving ) += step
      and then w( driving ) moves in proportion with the search direction.
      */
      virtual bool do_step(agx::RealValarrayPOD & z, agx::RealValarrayPOD & w,
        agx::RealValarrayPOD & v, agx::Real step, size_t driving);

      /**
      Here we solve for z in
      M(F, F)z(F) = -q(F) - M(F, T)z(T)
      where z(T) are the variables at their bounds.  This is to support the
      LCP solver
      */
      virtual void solve_subproblem(
        const agx::RealValarrayPOD& q,
        const agx::RealValarrayPOD& l,
        const agx::RealValarrayPOD& u,
        const agx::IntValarrayPOD & idx,
        agx::RealValarrayPOD& z);

#ifdef OCTAVE

      // Reset the factor of matrix from data available in octave.
      virtual void read_factor(const octave_value_list& args, int arg) = 0;

      /// Write the data in the matrix to an octave struct.
      virtual void convert_matrix_oct(octave_scalar_map & st) = 0;

#endif
#ifdef USE_IOH5
      virtual void write_hdf5(std::string & /*filename*/, std::string /*prefix*/ = "matrix") {}

      virtual void write_hdf5(H5::Group& /*g*/, std::string & /*name*/ = "matrix") {}

      virtual void read_hdf5(std::string& /*filename*/, std::string& /*prefix*/, int /*id*/) {}

      virtual void read_hdf5(H5::Group& /*g*/) {};
#endif
#ifdef USE_OCTAVE_IO
      virtual void write_octave(const std::string& /*filename*/, const std::string& /*name*/) {}
      virtual std::ostream& write_octave(std::ostream& os, const std::string& /*name*/) { return os; }
      virtual std::ostream& write_cpp(std::ostream& os, std::string /*name*/) { return os; };
#endif
    };


    /**
    This is the bare minimum data needed to store the original entries in a
    tridiagonal matrix matrix.  This is needed to keep the original data as
    needed when updating or down-dating the matrix factors.
    */
    struct diag_data {
      agx::RealValarrayPOD diag;    /// main diagonal: 2 entries per block
      agx::RealValarrayPOD sub;     /// first sub-diagonal: 1 entry per block
      agx::RealValarrayPOD ssub;    /// second sub-diagonal: even entries,
      diag_data(size_t _n) : diag(2 * _n), sub(2 * _n), ssub(2 * _n) {}
      diag_data() : diag(), sub(), ssub() {}

    };



    /**
    Here we have a matrix which corresponds to a chain of particles
    attached with spring and dampers, and each of the particles is subject
    to dry friction.  This is permuted here to be a banded matrix of
    bandwidth 4 using a packed format:

    Main diagonal where each other element is a mass, and each other
    element is the compliance of the contact.

    Sub-diagonal is all ones, one element for each mass.  This entry is on
    the same row as the compliance

    Sub-subdiagonal:  every other element: these are the spring constants

    fill  : on sub-diagonal as well.  These become non-zero during
    factorization
    */
    struct diag4 : public qp_matrix {

      size_t n_blocks;             ///  number of blocks
      agx::RealValarrayPOD diag;    /// main diagonal: 2 entries per block
      agx::RealValarrayPOD sub;     /// first sub-diagonal: 1 entry per block
      agx::RealValarrayPOD ssub;    /// second sub-diagonal: even entries,
                           /// one per block except for the last block.
      agx::RealValarrayPOD fill;    /// second sub-diagonal: odd entries, one per
                           /// block except for the last block.


      std::slice odd;

      diag_data original;          /** Hold the original data for matrix-vector
                                   multiply and to simplify update/down-date */


      inline void negate(size_t /*i*/) {} /** This does nothing because the
                                          internal structure of this matrix
                                          already determines what is and
                                          what isn't negated.

                                          default constructor.
                                          \TODO Find a way to get rid of this */

      diag4();

      /**
      Allocate all arrays as well as the matrix to hold the original data.
      Filling the arrays is left to the user.
      This delegates to the next constructor.
      */
      diag4(size_t _n, bool _bisymmetric = false);

      /**
      Here we allocate the diag4 matrix with the right size but also make
      sure that the qp_solver class allocates the basic data with the right
      sizes, as defined by inherited classes which can add rows and columns
      to this.
      The point here is that qp_matrix might have a different size from diag4
      This will be invoked by inherited classes in which case _n will not be
      equal to _na
      */
      diag4(size_t _n, size_t _na, bool _bisymmetric = false);


      /**
      Delegate allocation to previous constructor and copy basic data. All
      variables active by default.
      */
      diag4(const agx::RealValarrayPOD& _diag, const agx::RealValarrayPOD & _sub,
        const agx::RealValarrayPOD& _ssub,
        const bool _bisymmetric = false,
        const agx::BoolValarrayPOD _active = agx::BoolValarrayPOD());

      /**
      This does the real work in terms of allocation and storing data.
      */
      diag4(size_t _na, const agx::RealValarrayPOD& _diag, const agx::RealValarrayPOD & _sub,
        const agx::RealValarrayPOD& _ssub,
        const bool _bisymmetric,
        const agx::BoolValarrayPOD _active = agx::BoolValarrayPOD());

      inline virtual size_t size() const { return diag.size(); }

      /**
      This cache the original data for the matrix.
      */
      void sync() {
        original.diag = diag;
        original.sub = sub;
        original.ssub = ssub;
      }

#ifdef OCTAVE
      /**
      Utility to work from octave.  Here we take data which comes from a
      script to construct the matrix.
      */
      diag4(ColumnVector _diag, ColumnVector _sub, ColumnVector _ssub, ColumnVector _fill, ColumnVector _active, bool _bisymmetric = false);
      virtual void read_factor(const octave_value_list& args, int arg);
      virtual void convert_matrix_oct(octave_scalar_map & st);
#endif

      /**
      Negate the signs of the Lagrange multipliers for bisymmetric mode.
      These are the odd variables here, which are the friction forces.
      */
      inline virtual void flip(agx::RealValarrayPOD & x) {
        if (bisymmetric)
        {
          for (auto i = odd.start(); i < odd.size(); i += odd.stride())
          {
            x[i] = flip_sign * x[i];
          }
        }
      }

      /**
      Internal: the block is active if the second variable in that block is
      active.
      */
      inline bool active_block(size_t i) const {
        return active[2 * i + 1];
      }

      /**
      invert the diagonal block
      Given a 2x2 matrix [ a b ]
      [ b c ]
      the inverse is
      (1/det)[  c -b ]
      [ -b  a ]
      and det = (a * c - b *b ), the determinant.
      For this particular problem, only "a" is updated by the operations
      which precede this one and that is found in the factored matrix.
      Since other values in the block haven't been touched yet, the only
      valid data is in the original matrix.
      */
      inline bool invert_active_block(size_t i) {
        //!  determinant: need diagnostic for the case where the matrix isn't P
        agx::Real delta = agx::Real(1.0) / (diag[2 * i] * original.diag[2 * i + 1] - original.sub[i] * original.sub[i]);
        //! swap diagonal entries using the original data for everything except the "a" value
        diag[2 * i + 1] = delta * diag[2 * i];
        diag[2 * i] = delta * original.diag[2 * i + 1];
        //! flip the sign of the off-diagonal entry (matrix is symmetric)
        sub[i] = -delta * original.sub[i];
        return true;
      }

      /**
      Inactive block invert.

      No need for determinant for this case since the inactive block is
      just
      [ a  0 ]
      [ 0  1 ]
      */
      inline bool invert_inactive_block(size_t i) {
        //! @todo  Need diagnostic for the case where the matrix isn't P
        diag[2 * i] = (agx::Real) 1.0 / diag[2 * i];
        diag[2 * i + 1] = agx::Real(1.0);
        sub[i] = agx::Real(0);
        return true;
      }


      /// Wrapper for previous methods.
      inline bool invert_block(size_t i) {
        (active_block(i)) ? invert_active_block(i) : invert_inactive_block(i);
        return true;
      };

      /**
      Perform the work on the next subdiagonal using updated diagonal
      block.  This cannot be performed on the last element.

      What this will do is take care of the off diagonal elements below the
      block that was just inverted, update the next diagonal block, and
      create the fill.

      This corresponds to the standard matrix factorization operation of
      multiplying the current column with the inverse of the current,
      factored, diagonal block.  The column in this case has only one entry:
      the row after the block.

      This, in turn, according to standard matrix factorization algorithm,
      modifies the next diagonal block.  See Golub and van Loan for
      details.
      */
      inline bool push_forward(size_t i) {
        ssub[i] = original.ssub[i] * diag[2 * i];
        diag[2 * i + 2] = original.diag[2 * i + 2] - ssub[i] * original.ssub[i];
        fill[i] = original.ssub[i] * sub[i];
        return true;
      }

      /**
      Factor the entire matrix as LDLT factors.

      The special trick here is that `D'  is a 2x2 block diagonal matrix.
      This leaves `L'  to have 1s on the diagonal, a sub-subdiagonal.
      The sub-diagonal is contained  in the blocks.

      The reason to put the push_forward function in the condition
      preceeding the for-loop is to avoid an 'if' statement in the loop
      itself.  'block' gets executed n times, but push_forward is executed
      only n-1 times.  Note that `variable' here is blockwise, i.e., an odd
      number only.

      We stop short by 1 since `push_forward'  at block n-1 does all the work
      on block n.
      */
      virtual void factor(int variable = -1);

      /**
      Diagonal solve in the LDLT process.
      This just multiplies the 2x2 block stored in diag and sub with the
      RHS, since we store the inverse of D
      */
      inline bool solve_block_forward(agx::RealValarrayPOD &x, size_t i) {
        agx::Real tmp0 = x[2 * i - 2];
        x[2 * i - 2] = diag[2 * i - 2] * tmp0 + x[2 * i - 1] * sub[i - 1];
        x[2 * i - 1] = sub[i - 1] * tmp0 + x[2 * i - 1] * diag[2 * i - 1];
        return true;
      }

      /**
      All-in-one, lower triangular and diagonal solve: LDx = b.
      Work in-place, and do the digonal solve as we go.

      The lower triangle contains only ones on the sub-subdiagonal along
      with the fills.   Everything else is contained in the D blocks.

      This means in particular that there are only 1's in the first two
      rows.

      We move down block-wise and in all case encounter

      [ ssub   0  1  0 ]
      [   0  fill 0  1 ]
      */
      bool forward_elimination(agx::RealValarrayPOD & x, size_t start = 1);


      /**
      Solve LT y = b   in a very standard way.

      Use an int counter here to mind the 0 crossing
      */
      bool backward_substitution(agx::RealValarrayPOD& x);

      /**
      Driver.  Allow to start the solve at a variable different from the
      beginning in case x is all zeros up to `start'  Note that we never
      start at 0.
      */
      virtual void solve(agx::RealValarrayPOD  & x, size_t start = 1);


      /**
      Here we update a block which has changed from active to inactive or
      vice versa.  If that block was active, it had the form (before
      inverse)
      [ a + d  1 ]
      [ 1      e ]
      and `d' is a value added from what came before.   This has been
      inverted at this point though and so we re-invert the matrix to
      recover a + d, then set the new form
      [ a + d  0 ]
      [ 0      1 ]
      of an inactive block.  Then invert that and proceed.  For the other
      case, we go the other way around.  The argument `activate'  determines
      whether we go from active contact or the other way around.
      */
      inline void update_block(size_t variable);

      /// will mask the result
      inline virtual bool keep_it(const int & left_set, const size_t& i) const {
        return (left_set == ALL) || (left_set == FREE &&  active[i]) || (left_set == TIGHT && !active[i]);
      }


      // Standard  multiply add but with variable masking.
      virtual void multiply(
        const agx::RealValarrayPOD& b,
        agx::RealValarrayPOD& x,
        double alpha = 0.0,
        double beta = 1.0,
        int  left_set = ALL,
        int right_set = ALL);

      /**
      Specialized multiplication which only considers the even variables.
      Here alpha and beta are vectors so that we can do

      x =   A .* x + B .*  ( H(even, even) *  x );

      The reason for this is that we are considering cases where   H = M + U
      where M is diagonal, but we want to compute
      ( W + a * U ) * b

      Where W is not the diagonal of H
      and we compute instead  a * ( ( 1 / a ) * ( M + (W - M) )  + U  );
      */
      virtual void multiply_velocities(const agx::RealValarrayPOD& b,
        agx::RealValarrayPOD& x, const agx::RealValarrayPOD & D,
        const agx::RealValarrayPOD & G, agx::Real alpha = 0.0, agx::Real  beta = 1.0);

      /**
      For the bisymmetric case, the alternate entries on the diagonal are
      negated in comparison to the un-symmetrized matrix.  Needed for the
      LCP solver.
      */
      inline virtual agx::Real get_diagonal_element(size_t i) const {
        return  (i % 2) ? flip_sign * original.diag[i] : original.diag[i];
      }

      /**
      Since `variable' is one which can be switched, there are only two
      elements in that column: diagonal and first super diagonal.
      However, we are only interested in the parts corresponding to the
      active variables and that's just the super diagonal.
      We want  v = - M(F, F) \ M(F, variable ) where F is the set of
      active variables.
      \todo: this needs to allow for a sparse version so that optimization
      from the call can be made.
      */
      virtual void get_column(agx::RealValarrayPOD & v, size_t variable, agx::Real sign) {
        v[variable - 1] = sign * flip_sign * original.sub[(variable - 1) / 2];
        v[variable] = sign * flip_sign * original.diag[variable];
      }

      inline virtual void multiply_add_column(agx::RealValarrayPOD & w, agx::Real alpha, size_t variable) {

        if (!active[variable - 1]) {
          w[variable - 1] += alpha * flip_sign * original.sub[(variable - 1) / 2];
        }
        if (!active[variable]) {
          w[variable] += alpha *  flip_sign * original.diag[variable];
        }
      }

    };


    /**
    diag4length class extends the diag4 4-diagonal matrix by one full row
    and one full column.  As such, it reuses all the methods from diag for,
    but ads a small correction to them.   This does mean overloading all
    methods, but the amount of extra work is minimal.
    With this, one can have both curvature energy and a global lenght
    constraint.

    \TODO: biggest problem here is the semantic of the active variables.
    */
    struct diag4length : public diag4 {
      agx::RealValarrayPOD J0;       /// original data: Jacobian + diagonal
      agx::RealValarrayPOD J;        /** the Jacobian for the length part and diagonal element.  This gets overwritten
                            with inv(M) * J0 to compute the Schur
                            complement. */
      agx::Real i_schur;    /// inverse of  the Schur complement

      /**
      Does  nothing since sign structure is
      already implied in this special matrix.
      */
      virtual void negate(size_t /*i*/) { }

      ///
      inline size_t last() const {
        if (J0.size() == 0) {
          return std::numeric_limits<size_t>::max();
        }
        return J0.size() - 1;
      }

      ///
      diag4length();

      /// start with all empty
      diag4length(size_t _n, bool _bisymmetric = false);

      /// double delegation here, via the diag4 constructor
      diag4length(const agx::RealValarrayPOD& _diag,
        const agx::RealValarrayPOD & _sub,
        const agx::RealValarrayPOD& _ssub,
        const agx::RealValarrayPOD& _J,
        const bool _bisymmetric,
        const agx::BoolValarrayPOD _active = agx::BoolValarrayPOD());

      inline virtual size_t size() const { return J0.size(); }

#ifdef OCTAVE
      // utility to workf from octave
      diag4length(ColumnVector _diag, ColumnVector _sub, ColumnVector _ssub, ColumnVector _fill,
        ColumnVector _J, ColumnVector _active, bool _bisymmetric = false);
      virtual void convert_matrix_oct(octave_scalar_map & st);
      // utility to work from octave: sets the factor directly
      virtual void read_factor(const octave_value_list& args, int arg);
#endif

      ///
      virtual void multiply(
        const agx::RealValarrayPOD& b,
        agx::RealValarrayPOD& x, double
        alpha = 0.0,
        double beta = 1.0,
        int  /*left_set*/ = ALL,
        int /*right_set*/ = ALL);

      /**
      Just as the name says.  This is straight forward.  We cache
      inv(M)*J0(1:end-1) since that's used when solving.
      */
      virtual void get_schur_complement();

      ///
      virtual void factor(int variable = -1)
      {
        diag4::factor(variable);
        get_schur_complement();     // this will automatically factor via
        dirty = false;
      }

      ///
      virtual void solve(agx::RealValarrayPOD  & x, size_t start = 1);

      /**
      For the bisymmetric case, the alternate entries on the diagonal are
      negated in comparison to the un-symmetrized matrix.  Needed for the
      LCP solver.
      */
      inline virtual agx::Real get_diagonal_element(size_t i) const {
        if (i < last())
          return diag4::get_diagonal_element(i);
        else
          return  flip_sign * J0[last()];
      }

      virtual void get_column(agx::RealValarrayPOD & v, size_t variable, agx::Real sign) {

        if (variable < last()) {
          diag4::get_column(v, variable, sign);
          // WARNING: explicit assumption that the last row is never negated
          // by bi-symmetry
          v[last()] = sign * J0[variable];
        }
        else {
          for (size_t i = 0; i < J0.size(); ++i)
            v[i] = sign * flip_sign * J0[i];
        }
      }

      /**
      Multiply and add a column to a vector, making use of the fact that
      the columns are sparse here.
      */
      virtual void multiply_add_column(agx::RealValarrayPOD & w, agx::Real alpha, size_t variable);
    };

    /**
    qp_matrix_length class extends any qp_matrix to also contain a global
    length constraint, i.e., an extra row and column.

    This delegates most of the work to a qp_matrix allocated outside of
    this class.

    TODO: biggest problem here is the semantic of the active variables.
    The current method wastes memory by double allocating the arrays in the
    qp_matrix base class.
    */
    struct qp_matrix_length : public qp_matrix {
      qp_matrix * M;
      agx::RealValarrayPOD J0;       // original data: Jacobian + diagonal
      agx::RealValarrayPOD J;        // the Jacobian for the length part and
                            // diagonal element.  This gets overwritten
                            // with inv(M) * J0 to compute the Schur
                            // complement.
      agx::Real i_schur;                  // inverse of  the Schur complement

      inline size_t last() const {
        if (J0.size() == 0) {
          return std::numeric_limits<size_t>::max();
        }
        return J0.size() - 1;
      }

      /// Default constructor
      qp_matrix_length();

      /// start with all empty
      qp_matrix_length(qp_matrix * _M);

      virtual size_t size() const { return J0.size(); }



      /**
      Self-explanatory: used by LCP solver, and is *variable* based, not
      *block* based.  Subclasses have `blockwise'  version of these where
      the index is the index of a Lagrange multiplier, not just a variable.
      */
      virtual bool toggle_active(size_t i) {
        dirty = true;
        if (i < size() - 1)
          return M->toggle_active(i);
        else
          return true;
      }

      virtual void set_active() {
        M->set_active();
        return;
      }


      /// self-explanatory: for LCP solver: variable based, not  block  based.
      virtual void set_active(const agx::IntValarrayPOD& idx) {
        dirty = true;
        for (size_t i = 0; i < M->size(); ++i)
          M->set_active(i, idx[i] <= 0);
        return;
      }

      /**
      \todo This was previously hidden, which is a compiler warning.
      Implemented to call the base class, which is what would
      have happened in some cases previously as well.
      Is this the proper way to do it?
      */
      virtual void set_active(size_t i, bool a)
      {
        this->qp_matrix::set_active(i, a);
      }


#if 0
      //#ifdef OCTAVE
      /// utility to work from octave
      XXXX TODO
        diag4length(ColumnVector _diag, ColumnVector _sub, ColumnVector _ssub, ColumnVector _fill,
          ColumnVector _J, ColumnVector _active, bool _bisymmetric = false);
      virtual void convert_matrix_oct(octave_scalar_map & st);
      /// utility to work from octave: sets the factor directly
      virtual void read_factor(const octave_value_list& args, int arg);
#endif

      virtual void multiply(
        const agx::RealValarrayPOD& b,
        agx::RealValarrayPOD& x,
        double alpha = 0.0,
        double beta = 1.0,
        int  /*left_set*/ = ALL,
        int /*right_set*/ = ALL);

      virtual void negate(size_t i) {
        if (i < J0.size() - 1) {
          M->negate(i);
        }
      }

      /**
      Just as the name says.  This is straight forward.  We cache
      inv(M)*J0(1:end-1) since that's used when solving.
      */
      virtual void get_schur_complement();

      virtual void factor(int variable = -1) {
        M->factor(variable);
        get_schur_complement();     // this will automatically factor via
        dirty = false;
      }


      virtual void solve(agx::RealValarrayPOD  & x, size_t start = 1);

      /**
      For the bisymmetric case, the alternate entries on the diagonal are
      negated in comparison to the un-symmetrized matrix.  Needed for the
      LCP solver.
      */
      inline virtual agx::Real get_diagonal_element(size_t i) const {
        if (i < last())
          return M->get_diagonal_element(i);
        else
          return  flip_sign * J0[last()];
      }

      virtual void get_column(agx::RealValarrayPOD & v, size_t variable, agx::Real sign) {

        if (variable < last()) {
          M->get_column(v, variable, sign);
          // WARNING: explicit assumption that the last row is never negated
          // by bi-symmetry
          v[last()] = sign * J0[variable];
        }
        else {
          for (size_t i = 0; i < J0.size(); i += 2)
            v[i] = sign * flip_sign * J0[i];
        }
      }
      /**
      Multiply and add a column to a vector, making use of the fact that
      the columns are sparse here.
      */
      virtual void multiply_add_column(agx::RealValarrayPOD & w, agx::Real alpha, size_t variable);
    };
  }
}

DOXYGEN_END_INTERNAL_BLOCK()
