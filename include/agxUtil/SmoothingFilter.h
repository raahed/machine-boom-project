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
#ifndef AGXUTIL_SMOOTHINGFILTER_H
#define AGXUTIL_SMOOTHINGFILTER_H

#include <agx/agxPhysics_export.h>
#include <agx/Math.h>
#include <agx/Referenced.h>

namespace agxUtil
{
  class AGXPHYSICS_EXPORT SmoothingFilter : public agx::Referenced
  {
    public:
      SmoothingFilter() {}

      /**
      Update statistic variable \p s with current observation \p observation.
      \param observation - new observation
      \param s - current statistic variable
      \return new statistic variable (smoothed value)
      */
      virtual agx::Real update( agx::Real observation, agx::Real s ) const = 0;

    protected:
      virtual ~SmoothingFilter() {}
  };

  typedef agx::ref_ptr< SmoothingFilter > SmoothingFilterRef;

  /**
  Variable Smooth Factor Exponential Moving Average filter.
  During update, this class calculates a suitable smoothing factor
  to eliminate any sudden peaks in the observations (i.e., the smoothing factor goes to zero).
  */
  class AGXPHYSICS_EXPORT VariableSmoothingFactorFilter : public SmoothingFilter
  {
    public:
      /**
      Construct given maximum smoothing factor.
      \param maximumSmoothingFactor - the maximum smoothing factor to be used for this filter (valid 0 <= maximumSmoothingFactor <= 1)
      */
      VariableSmoothingFactorFilter( agx::Real maximumSmoothingFactor )
        : m_maxSmoothingFactor( maximumSmoothingFactor ) {}

      /**
      Update the current statistic variable \p s given a new observation.
      Will calculate a smoothing factor given the deviation of the new observation
      from the current value of s. Large deviations will give smaller smoothing
      factor, i.e., only a small fraction of the new observation will be part
      of s after this update.
      */
      virtual agx::Real update( agx::Real observation, agx::Real s ) const;

      /**
      \return the maximum smoothing factor
      */
      agx::Real getMaximumSmoothingFactor() const
      {
        return m_maxSmoothingFactor;
      }

      /**
      Set the maximum smoothing factor for this filter.
      \param maxSmoothingFactor - the maximum smoothing factor
      */
      void setMaximumSmoothingFactor( agx::Real maxSmoothingFactor )
      {
        m_maxSmoothingFactor = maxSmoothingFactor;
      }

    protected:
      virtual ~VariableSmoothingFactorFilter() {}

    protected:
      agx::Real m_maxSmoothingFactor;
  };

  typedef agx::ref_ptr< VariableSmoothingFactorFilter > VariableSmoothingFactorFilterRef;

}
#endif
