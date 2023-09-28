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

#include <agxWire/WireSolver/Diag4.h>
#include <agxWire/WireSolver/wireSolverTypes.h>


#ifdef USE_IOH5
#include "ioh5.h"
#endif

#include <math.h>
#include <vector>
#include <limits>
#include <algorithm>
#include <assert.h>
#include <exception>

#ifdef USE_OCTAVE_IO

#include "octaveio.h"
#include <iostream>
#include <sstream>
#include <fstream>
#include <iomanip>

#endif

#define DUMP_PROBLEM()

DOXYGEN_START_INTERNAL_BLOCK()


#define DO_TIME_MEASURE 0

namespace agxWire {

  namespace WireSolver {



    /**
    Utility.
    Keeps the four possible steps and indicates which one is chosen.
    The reason to have a whole class for this is that the logic is
    confusing.  This keeps the qp class simpler and cleaner.

    The computation of the min steps is done in the qp solver

    TODO: too little precaution is taken against getting an invalid step.
    */
    struct min_step {
      enum { INVALID = -1, NATURAL = 0, Z_BOUND = 1, S_LOWER = 2, S_UPPER = 3 };

      agx::Real steps[4];         /// the four candidates
      size_t  indices[4];         /// index of variables. May be (size_t)-1 to signal none. (Well, it can be -1. Not sure what it would mean.)
      int step_type;              /// which of the four alternatives
      bool ready;                 /// flag telling that computation is over

      min_step(int _driving, agx::Real theta0, agx::Real theta1) { init(_driving, theta0, theta1); }

      /// Initialize in a silly state
      min_step() : min_step(-1, std::numeric_limits<agx::Real>::infinity(), std::numeric_limits<agx::Real>::infinity()) {}

      void reset() { init(size_t(-1), std::numeric_limits<agx::Real>::infinity(), std::numeric_limits<agx::Real>::infinity()); }

      /**
      Prepare the computation: set the two natural steps, and reset the
      other two options to infinity.
      */
      void init(size_t _driving, agx::Real theta0, agx::Real theta1);

      ///
      ///  Used as we perform the minimum ratio step.   This is just a running
      ///  'min'  computation, catching the idex of the min as we go along
      ///  TODO:  I cannot see how to use the standard library for this.
      inline void update(agx::Real x, int i, int var) {
        if (x < steps[var]) {
          steps[var] = x;
          indices[var] = i;
        }
      }

      ///
      /// Returns which of the four alternative steps we are taking.
      ///
      inline int get_step_type() const {
        return (int)step_type;
      }

      ///
      /// This will return the `driving'  variable when we take a natural
      /// pivot, i.e., one which 'frees' a variable and makes its slack vanish.
      ///
      /// Otherwise we get the index of the variable which prevents taking a
      /// full step.
      ///
      inline int get_blocking() const {
        return (int)indices[(int)step_type];
      }

      ///
      /// Return the min step.
      /// The computation  is not done here, though it could be a good idea to
      /// do it as needed.
      agx::Real get_min_step(agx::Real pivot);
    };

    ///
    /// A QP solver using Keller's algorithm
    ///
    ///
    struct QPSolver {
      enum STATES { EQ = -1, FREE = 0, LOWER = 2, UPPER = 3 };
      enum STATUS { SUCCESS = 0, MAX_ITERATIONS = 2, UNBOUNDED_RAY = 4 };

      const agx::Real inf = std::numeric_limits<agx::Real>::infinity();

      qp_matrix * M;                  /// System matrix
      agx::IntValarrayPOD idx;            ///  index set
      agx::RealValarrayPOD w;                    /// slacks
      agx::RealValarrayPOD v;                    /// search direction.


                                        /// these are variables for the solve which are shared between the
                                        /// different methods.
      agx::Real pivot;                   /// Whether we are working on a lower bound
                                         /// or upper bound: the latter case
                                         /// involves a -1 sign
      min_step step;                /// The step
      size_t driving;               /// The driving variable, i.e., the one for
                                    /// which are are trying to find a feasible
                                    /// slack
      size_t blocking;              /// A blocking variable, i.e., one which
                                    /// prevents making the slack of the
                                    /// driving variable feasible.
      agx::Real mtt;                     /// The rate at which the slack changes
                                         /// with respect to the corresponding
                                         /// variable.
      int iterations;               /// Number of iterations for main loop.
      int projection_iterations;       /// Iterations of projections for feasible
                                       /// point.
      agx::Real zero_tolerance;          /// How much slack we give for exact zeros.
      bool done;                    /// Done.

      //std::chrono::duration<double> solve_time;
      //std::chrono::duration<double> projection_time;

      virtual ~QPSolver();

      /**
      Default constructor takes references to dummy variables.  This is a
      bit silly since we do not have a resize functionality now.
      */
      QPSolver();


      /**
      Standard constructor.
      \TODO: the `const'  has to be dropped if we want to reuse an instance
      of this class, both here and in declarations.
      Also, the 'z' vector should be a reference to avoid copy going in and
      out.
      */
      QPSolver(qp_matrix * _m);



      /**
      Resize the memory used to fit the specified matrix.
      */
      void resize(qp_matrix * _m);

      void init();



      /// \returns the number of variables

      inline size_t size() const {
        return idx.size();
      }

      /**
      Compute how the z vector changes for a unit change in the driving
      variable
      */
      bool get_search_direction(const agx::RealValarrayPOD & q);

      size_t get_free();

      /// Rearrange the index set to be consistent with the step just taken.
      bool do_pivot(const agx::RealValarrayPOD& /*z*/, const agx::RealValarrayPOD& /*q*/);


      /// Just compute   w = M*z + q
      bool get_complementarity_point(const agx::RealValarrayPOD& q,
        const agx::RealValarrayPOD & lo, const agx::RealValarrayPOD & up,
        agx::RealValarrayPOD & z);

      /**
      This will find the largest slack and determine whether or not this
      corresponds to a lower or upper bound.
      */
      bool get_driving_variable();

      /**
      Figure out how far we can push the driving variable without breaking
        other bounds.
      */
      bool get_step_length(const agx::RealValarrayPOD & lo, const agx::RealValarrayPOD & up, const agx::RealValarrayPOD & z);

      int project_feasible(const agx::RealValarrayPOD & q,
                           const agx::RealValarrayPOD & lo, const agx::RealValarrayPOD & up,
                           agx::RealValarrayPOD &  z, bool warm = false);

      int active_set_switch(const agx::RealValarrayPOD & lo, const agx::RealValarrayPOD & up, const agx::RealValarrayPOD &  z);

      size_t get_at_bound();



      /// Simple feasible point: all inequalities inactive.
      void get_feasible_point(const agx::RealValarrayPOD & q, const agx::RealValarrayPOD & lo, const agx::RealValarrayPOD & up, agx::RealValarrayPOD & z);

      ///
      size_t active_count() const;

      ///
      bool is_feasible(const agx::RealValarrayPOD & lo, const agx::RealValarrayPOD & up, agx::RealValarrayPOD & z)  const;

      ///
      agx::Real get_feasibility_error(const agx::RealValarrayPOD & lo, const  agx::RealValarrayPOD & up, agx::RealValarrayPOD & z) const;

      ///
      bool feasible(const agx::RealValarrayPOD & lo, const agx::RealValarrayPOD & up, const agx::RealValarrayPOD & z) const;

      /**
      This could of course take arguments so the same class could be
      references as class members.
      */
      int solve(const agx::RealValarrayPOD& q, const agx::RealValarrayPOD& lo,
        const agx::RealValarrayPOD& up, agx::RealValarrayPOD & z, const bool warmStart = false);


      /// See how well the solver did
      inline int get_iterations() const { return iterations; }



      /// This will check if all complementarity conditions are satisfied.
      agx::Real get_complementarity_error(const agx::RealValarrayPOD & z);

      std::ostream &   report_stats(std::ostream & io,
        const agx::RealValarrayPOD & z,
        const agx::RealValarrayPOD & lo,
        const agx::RealValarrayPOD & up);

#ifdef USE_IOH5

      virtual void write_hdf5(const std::string & filename,
        const std::valarray<int>& q,
        const std::valarray<int>& lo,
        const std::valarray<int>& up,
        std::string prefix = std::string("problem"));
      virtual void read_hdf5(std::string filname, const std::string & prefix, int id);
      virtual void read_hdf5(H5::Group);
#endif
#ifdef USE_OCTAVE_IO

      virtual void write_octave(const std::string & filename,
        const agx::RealValarrayPOD& q,
        const agx::RealValarrayPOD& lo,
        const agx::RealValarrayPOD& up,
        const agx::RealValarrayPOD& z,
        std::string /*prefix*/ = std::string("problem"));


      template <class T>
      std::ostream & write_cpp_array(std::ostream& os, const std::string & name, T & vv) {
        std::string space(",");
        os << "static std::valarray<double> " << name << "({" << std::endl;
        for (size_t i = 0; i < vv.size(); ++i) {
          os << vv[i] << space << std::endl;
          if (i == vv.size() - 2) {
            space = "";
          }
        }
        return  os << "});";
      }

      std::ostream & write_cpp_array_int(std::ostream& os, const std::string & name, agx::IntValarrayPOD & vv);

      std::ostream & write_cpp_array_int(std::ostream& os, const std::string & name, const std::valarray<int> & vv);

      void write_cpp(const std::string & filename,
        const agx::RealValarrayPOD& q,
        const agx::RealValarrayPOD& lo,
        const agx::RealValarrayPOD& up,
        std::string /*prefix*/ = std::string("problem"));

#endif
      private:

        // To silence compiler
        QPSolver& operator=(const QPSolver&) = delete;

    };

  }
}

DOXYGEN_END_INTERNAL_BLOCK()

