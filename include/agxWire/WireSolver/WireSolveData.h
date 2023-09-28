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

#include <cmath>
#include <math.h>
#include <limits>

#include <agxWire/WireSolver/Banded.h>
#include <agxWire/WireSolver/wireSolverTypes.h>

DOXYGEN_START_INTERNAL_BLOCK()

// needed for linking with octave
#ifdef OCTAVE
class octave_value_list;
#endif

namespace agxWire
{
  namespace WireSolver
  {

    /**
    The WireSolveData struct holds information necessary to perform the
    simulation. Computations of the necessary quantities such as spring
    constants and distance Jacobians are left to other classes.
    */
    struct WireSolveData {
      agx::RealValarrayPOD x;                /// Position of the particle along the edge on which it lies

      agx::RealValarrayPOD lowerEdgeLimit;   /// Edge limit along negative edge direction (-inf as default).
      agx::RealValarrayPOD upperEdgeLimit;   /// Edge limit along positive edge direction (+inf as default).

      agx::RealValarrayPOD v;                /// Velocity of the particle along same.

      agx::RealValarrayPOD masses;             /// Each particle has a different mass proportional to the length of the segments to the neighbors

      agx::RealValarrayPOD diagSprings;          /** There is one of these for each segment
                                            of the wire as well as one on each of
                                            the endpoints: that's one more than
                                           the number of particles*/



      agx::RealValarrayPOD offDiagSprings;          /** There is one of these for each segment
                                           of the wire as well as one on each of
                                           the endpoints: that's one more than
                                           the number of particles */



                                           /** Computation of the friction bounds is left to the application since
                                           that involves the contact properties and normal forces on the
                                           points. */

      agx::RealValarrayPOD friction_lo;      /// Lower bound for the friction force

      agx::RealValarrayPOD friction_up;      /** Upper bound for the friction force:
                                    both are needed since the QP solver
                                    takes const ref to these so we cannot
                                    just negate.*/
      agx::RealValarrayPOD lower_bounds;     /// ALL bounds for the QP solver, including those for the range constraints.
      agx::RealValarrayPOD upper_bounds;
      agx::Real bounds_compliance;

      agx::RealValarrayPOD forces;           /// External force on each particle

      agx::Real length;             /// The rest length of the wire
      agx::Real current_length;     /// The actual length of the wire start of the solve.
      agx::Real length_speed;       /// Speed of elongation.
      agx::Real length_compliance;  /// Compliance of length constraints

      agx::RealValarrayPOD lengths;          /// length of segments between particles.

      agx::RealValarrayPOD Jacobian;         /**This is going to contain the rate of
                                    change of length as a function of rate
                                    of change of x.  Edge data is needed to
                                    get this.   For convenience, this
                                    contains the last diagonal element of
                                    the matrix.*/

                                    /**
                                    TODO: This is where we need access to edge data of sort so that we
                                    can correctly compute the distance between the control points we are
                                    considering.
                                    This information can be in here, or we can have references to the
                                    data. */


                                    /// \returns The number of point masses.
      inline size_t size() const { return masses.size(); }

      /**
        This should be the most common constructor.  After this point, the
        values to be put on the arrays are written directly.  There is no
        safety here but accessors/mutators are a pain.
      */
      WireSolveData(size_t N, agx::Real _length);

      void reset(size_t N, agx::Real _length);


      /**
       This will compute the rate of change of length wrt. rate of change of
       particle position

       \TODO: this computation is correct if edges are parallel.  Otherwise
       we need the angle between the two.

       The distance between two nodes is
       d[ i ] = hypot( lengths[ i ], x[ i + 1 ] - x[ i ]);
       Note that x[ i ] appears twice here.
       */
      inline void compute_jacobians() {}


      /**
       Every 4th element on the main diagonal are the masses + spring
       constants multiplied by factors dependent on integration.

       On the first subdiagonal we have the unit Jacobian of point particle
       friction constraint.

        Second subdiagonal has a 1 every 4th entry for the lower bound
        constraint

        Third subdiagonal has a 1 every 4th entry for the upper bound
        constraint.

        Fourth subdiagonal has a coupling spring constant every 4th
        element

        To account for bisymmetry, all Lagrange multipliers must be
       negated.

       NOTE: the damping forces have to be added as well but with a
       different factor!
      */
      void assemble_matrix(banded_matrix & MB, agx::Real gamma, agx::Real pertFrictionSlip, agx::Real pertBounds = agx::Real(1E-10));

      /**
      */
      void assemble_matrix(banded_length & ML, agx::Real pertBounds = agx::Real(1E-10), agx::Real pertLength = agx::Real(1e-10));

#ifdef OCTAVE
      /// Utilities for octave so that we can easily put sensible numbers in
      /// this.
      WireSolveData(const octave_value_list& args, int arg);
#endif


      /**
       Here we write the spring forces on a vector taking care of a stride.

       Because the `q' vector is the one that will go in the QP solver, we
       negate everything if needed
      */
      void get_spring_forces(agx::RealValarrayPOD & q, size_t stride, agx::Real step, bool negate = true);

      /**
      Compute distance between point i and point i+1.   Here I assume that
      the particles have distance 1 when x[ i ] = x[ i + 1 ]
      */
      inline agx::Real get_length_violation() {

        agx::Real l = 0;
        for (size_t i = 0; i < x.size() - 1; ++i) {
          l += std::hypot(lengths[i], x[i] - x[i + 1]);
        }

        current_length = l;
        return  length - l;
      }


      /**
      This computes the rate of change of the length of the string: should
      be adjusted to account for the changes in the Jacobian as the
      particles move.
      */
      inline agx::Real get_length_speed()
      {
        agx::Real sum(0);
        for (size_t i = 0; i < v.size(); ++i) {
          sum += Jacobian[i] * v[i];
        }

        length_speed = sum;

        return sum;
      }

      //
      // Standard stepping
      //
      inline void step(agx::Real step) {
        for (size_t i = 0; i < v.size(); i++)
          x[i] += step * v[i];
      }

    };


    /**
    This contains the solver and the stepper.

    This class will perform necessary computations based on edges and
    tension of
    */
    class WireSimulation {

    private:

      agx::Real h;                       /**Time step: the class is small and this
                                         shows up in two places only so the one
                                         letter variable won't hurt.
                                         This is private so that the Spook
                                         parameters are readjusted when it is
                                         changed.*/

    public:

      WireSolveData m_wire;                  /// Notice the inconsistent nomenclature

      std::shared_ptr<banded_matrix> MB;                /// System matrix with arbitrary
                                                        /// bandwidth

      banded_length MBL;

      QPSolver QP;                    /// An instance of the LCP solver.  OK,
                                       /// OK, it really is a QP solver disguised
                                       /// as an LCP one.


      agx::RealValarrayPOD q;                /// This is the "negated RHS"  for the
                                    /// stepper.  It's 'q' to be consistent
                                    /// with the internals of the solver


      agx::RealValarrayPOD z;                /**That's the vector we solve for.  It
                                    contains both velocities and Lagrange
                                    multipliers intertwined in weird and
                                    wonderful ways.   For the model with
                                    just curvature, velocities are in the
                                    even entries, and odd entries contain
                                    contact forces.  For the one with the
                                    global distance constraint, the last
                                    entry is the tension for that
                                    constraint.*/



      agx::Real slip;              /**Contact friction creep.  This can be
                                   zero here because the matrix is always
                                   well conditioned.*/

      agx::Real relaxation;              /// Standard Spook parameter

      agx::Real spook_a;                 /// The parameter in front of constraint
                                         /// violation

      agx::Real spook_b;                 /// The parameter in front of constraint
                                         /// velocity

      agx::Real spook_c;                 /// The diagonal perturbations.

      bool ready;                   /// True once the matrix is assembled

      bool positive_tension;

      ///
      ///  Alternate constructor to use the banded matrix.
      ///
      WireSimulation(size_t N, agx::Real H = agx::Real(1.0 / 60.0), bool _positive_tension = true);

      /// Reset the simulation to initial state but with size of vectors retained according to \param N
      void reset(size_t N, agx::Real H, bool _positive_tension);

      inline agx::Real get_time_step() const { return h; }

      /**
      Setting the time step triggers re-computation of the parameters used
      in the stepper.
      */
      void set_time_step(agx::Real H);

      void get_rhs(bool negate = true);
      void get_length_rhs(bool negate = true);

      /// Perform n steps of size h
      void do_step(size_t n, size_t step = 0);

    private:

      // To silence the compiler warning about assignment operator
      WireSimulation& operator=(const WireSimulation&) = delete;

    };
  } // WireSolver
} // agxWire

DOXYGEN_END_INTERNAL_BLOCK()
