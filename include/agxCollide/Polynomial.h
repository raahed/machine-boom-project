/*
Copyright 2007-2023. Algoryx Simulation AB.

All AGX source code, intellectual property, documentation, sample code,
tutorials, scene files and technical white papers, is copyrighted, proprietary
and confidential material of Algoryx Simulation AB. You may not download, read,
store, distribute, publish, copy or otherwise disseminate, use or expose this
material unless having a written signed agreement with Algoryx Simulation AB, or having been
advised so by Algoryx Simulation AB for a time limited evaluation, or having purchased a
valid commercial license from Algoryx Simulation AB.

Algoryx Simulation AB disclaims all responsibilities for loss or damage caused
from using this software, unless otherwise stated in written agreements with
Algoryx Simulation AB.
*/

#ifndef AGXCOLLIDE_POLYNOMIAL_H
#define AGXCOLLIDE_POLYNOMIAL_H

#include <agx/Math.h>
#include <agxCollide/agxCollide.h>
#include <agx/StackArray.h>


namespace agxCollide
{
  template <size_t N>
  class Polynomial {

  public:

    /** Constructor: creates a Polynomial with given coefficients.
    \param coefficients Coefficients given in decreasing degree */
    Polynomial( const agx::StackArray<agx::Real, N>& coefficients );

    /** Calculates agx::Real roots of a single variable polynomial within a given boundary.
    \param intervalStart start of search interval
    \param intervalEnd end of search interval
    \param roots: Returns the found roots within interval, in increasing order.
                  Multiple roots are only give once.
    */
    void findRealRootsInInterval(
      const agx::Real intervalStart,
      const agx::Real intervalEnd,
      agx::StackArray<agx::Real, N - 1>& roots ) const;

    /** Calculates derivation of a polygon */
    AGX_FORCE_INLINE Polynomial<N-1> derive() const;

    /** Evaluates polynomial given by polyCoefficients at value x. */
    AGX_FORCE_INLINE agx::Real evaluateAt( const agx::Real x ) const;

    /** Solves linear function ax + b = 0; returns only agx::Real solutions within given interval
    Assume it has not been called with all coefficients == 0. */
    static void findRootsOfLinearInInterval(
      const agx::Real a,
      const agx::Real b,
      const agx::Real intervalStart,
      const agx::Real intervalEnd,
      agx::StackArray<agx::Real, 1>& root );

    /** Solves quadratic function a x^2 + bx + c, returns only agx::Real solutions within time interval.
    Assume it has not been called with all coefficients == 0 */
    static void findRootsOfQuadraticInInterval(
      const agx::Real a,
      const agx::Real b,
      const agx::Real c,
      const agx::Real intervalStart,
      const agx::Real intervalEnd,
      agx::StackArray<agx::Real, 2>& roots );

  private:
    agx::StackArray<agx::Real, N> m_coefficients;

  };


  /// Implementations

  template <size_t N>
  inline Polynomial<N>::Polynomial( const agx::StackArray<agx::Real, N>& coefficients ) {
    m_coefficients = coefficients;
  }

  template<>
  inline void Polynomial<size_t(2)>::findRealRootsInInterval(
    const agx::Real intervalStart,
    const agx::Real intervalEnd,
    agx::StackArray<agx::Real, 1>& roots ) const
  {
    switch(m_coefficients.size()) {
    case 2:
      {
        // linear case
        agx::StackArray<agx::Real, 1> rootsMinor;
        findRootsOfLinearInInterval( m_coefficients[0], m_coefficients[1],
          intervalStart, intervalEnd, rootsMinor );
        roots = rootsMinor;
        break;
      }
    default:
      break;
    }
    return;
  }


  template <size_t N>
  void Polynomial<N>::findRealRootsInInterval(
    const agx::Real intervalStart,
    const agx::Real intervalEnd,
    agx::StackArray<agx::Real, N - 1>& roots ) const
  {
    switch (m_coefficients.size()) {
    case 0:
      break;
    case 1:
      break;
    case 2:
      {
        // linear case
        agx::StackArray<agx::Real, 1> rootsMinor;
        findRootsOfLinearInInterval( m_coefficients[0], m_coefficients[1],
          intervalStart, intervalEnd, rootsMinor );
        roots = rootsMinor;
      }
      break;
    default:
      {
        if (agx::equalsZero( m_coefficients[0] )) { // leading coefficient zero
          agx::StackArray<agx::Real, N - 1> coefficientsWithoutLeading;
          for (size_t i = 1; i < m_coefficients.size(); ++i)
            coefficientsWithoutLeading.push_back( m_coefficients[i] );
          Polynomial<N - 1> withoutLeadingCoefficient( coefficientsWithoutLeading );
          agx::StackArray<agx::Real, N - 2> rootsMinor;
          withoutLeadingCoefficient.findRealRootsInInterval( intervalStart, intervalEnd, rootsMinor );
          roots = rootsMinor;
          return;
        }

        Polynomial<N - 1> polyDer = derive();
        agx::StackArray<agx::Real, N - 2> rootsOfDerivative;
        polyDer.findRealRootsInInterval( intervalStart, intervalEnd, rootsOfDerivative );

        // calculate interval borders
        agx::StackArray<agx::Real, N> intervalBorders;
        intervalBorders.push_back( intervalStart );
        if (rootsOfDerivative.size() > 0) {
          intervalBorders.push_back( rootsOfDerivative[0] );
          for (size_t i = 1; i < rootsOfDerivative.size(); ++i) {
            if (!agx::equalsZero( rootsOfDerivative[i] - rootsOfDerivative[i - 1] ))
              intervalBorders.push_back( rootsOfDerivative[i] ); // not interested in multiples of roots
          }
        }
        intervalBorders.push_back( intervalEnd );

        const agx::Real eps(agx::REAL_SQRT_EPSILON * 10); // For evaluating derivatives along interval.
        const agx::Real zeroTolerance(agx::REAL_SQRT_EPSILON * 0.001);
        const size_t maxIter = 50;

        for (size_t intervalNr = 0; intervalNr < intervalBorders.size() - 1; intervalNr++) {
          const agx::Real localMin = intervalBorders[intervalNr];
          const agx::Real localMax = intervalBorders[intervalNr + 1];
          const agx::Real localEps = std::min(eps, (localMax - localMin) * 0.001);
          // check at interval borders (derivative might be zero here)
          if (agx::equalsZero( evaluateAt( localMin ), zeroTolerance ) && (roots.size() == 0 || !agx::equivalent(roots[roots.size() - 1], localMin, eps))) {
            roots.push_back( localMin );
          }
          else {

            agx::Real fAtMin = evaluateAt( localMin + localEps);
            agx::Real fAtMax = evaluateAt( localMax - localEps);
            if (agx::sign( fAtMin ) != agx::sign( fAtMax )) { // only interested in intervals where sign changes happen
              agx::Real xTmp = agx::Real(0.5) * (localMin + localMax);
              agx::Real f = evaluateAt( xTmp );
              size_t localIter = 0;
              // For cycle detection: Make update factor smaller if in cycle.
              agx::Real fPrevious = f;
              agx::Real factor = 1.0;
              while (!agx::equalsZero( f, zeroTolerance ) &&  localIter < maxIter) {
                agx::Real df = polyDer.evaluateAt( xTmp );
                xTmp -= factor * f/df; // we never allow df to become zero, by construction
                xTmp = agx::clamp(xTmp, localMin + localEps, localMax - localEps);
                fPrevious = f;
                f = evaluateAt( xTmp );
                if (agx::geq(agx::absolute(f), agx::absolute(fPrevious)))
                  factor *= 0.9;
                ++localIter;
              }
              if (agx::equalsZero( f, zeroTolerance ) && (roots.size() == 0 || !agx::equivalent(roots[roots.size() - 1], xTmp, eps))) {
                roots.push_back( xTmp );
              }
            }
          }
        }
      }
    }
  }


  template <size_t N>
  Polynomial<N - 1> Polynomial<N>::derive() const
  {
    agx::StackArray<agx::Real, N - 1> deriveCoefficients;

    for (int i = 0; i < int(m_coefficients.size()) - 1; ++i)
      deriveCoefficients.push_back( (agx::Real)(m_coefficients.size() - 1 - i) * m_coefficients[i] );

    return Polynomial<N - 1>( deriveCoefficients );
  }


  template <size_t N>
  agx::Real Polynomial<N>::evaluateAt( const agx::Real x ) const
  {
    // evaluating using Horner's scheme
    agx::Real polyVal = 0;
    for (size_t i = 0; i < m_coefficients.size(); ++i)
      polyVal = m_coefficients[i] + polyVal * x;

    return polyVal;
  }


  template <size_t N>
  void Polynomial<N>::findRootsOfLinearInInterval(
    const agx::Real a,
    const agx::Real b,
    const agx::Real intervalStart,
    const agx::Real intervalEnd,
    agx::StackArray<agx::Real, 1>& root)
  {
    if (!agx::equalsZero( a )) { // non-constant case
      agx::Real xTmp = -b / a;
      if (xTmp >= intervalStart && xTmp <= intervalEnd)
        root.push_back(xTmp);
    }
    return;
  }


  template <size_t N>
  void Polynomial<N>::findRootsOfQuadraticInInterval(
    const agx::Real a,
    const agx::Real b,
    const agx::Real c,
    const agx::Real intervalStart,
    const agx::Real intervalEnd,
    agx::StackArray<agx::Real, 2>& roots )
  {
    if (agx::equalsZero( a )) {  // linear case
      agx::StackArray<agx::Real, 1> rootsMinor;
      findRootsOfLinearInInterval( b, c, intervalStart, intervalEnd, rootsMinor );
      roots = rootsMinor;
    }
    else {
      agx::Real D = b * b - 4 * a * c;

      if (agx::equalsZero( D )) { // determinant equals zero, 1 solution
        agx::Real xTmp = -agx::Real(0.5) * b / a;
        if (xTmp >= intervalStart && xTmp <= intervalEnd)
          roots.push_back( xTmp );
      }
      else {
        if (D > 0) {
          agx::Real aInv = 1 / a; // two agx::Real solutions
          agx::Real sqrtD = std::sqrt(D);
          agx::Real xTmp = agx::Real(0.5) * (-b - sqrtD) * aInv;
          if (xTmp >= intervalStart && xTmp <= intervalEnd)
            roots.push_back( xTmp );
          agx::Real xTmp2 = agx::Real(0.5) * (-b + sqrtD) * aInv;
          if (xTmp2 >= intervalStart && xTmp2 <= intervalEnd)
            roots.push_back( xTmp2 );
        }
      }
    }
  }

}


#endif // AGXCOLLIDE_POLYNOMIAL_H
