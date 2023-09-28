/*
Copyright 2007-2023. Algoryx Simulation AB.

All AGX source code, intellectual property, documentation, sample code,
tutorials, scene files and technical white papers, are copyrighted, proprietary
and confidential material of Algoryx Simulation AB. You may not download, read,
store, distribute, publish, copy or otherwise disseminate, use or expose this
material unless having a written signed agreement with Algoryx Simulation AB, or
having been advised so by Algoryx Simulation AB for a time limited evaluation,
or having purchased a valid commercial license from Algoryx Simulation AB.

Algoryx Simulation AB disclaims all responsibilities for loss or damage caused
from using this software, unless otherwise stated in written agreements with
Algoryx Simulation AB.
*/

#pragma once


#include <agx/Real.h>
#include <agx/Integer.h>
#include <agx/Math.h>
#include <agx/Referenced.h>
#include <agx/agx_vector_types.h>

#include <agx/agxPhysics_export.h>

#include <memory>


namespace agxUtil
{
  /**
  Pure virtual statistic class containing the fundamentals for implementing a statistics method
  */
  class AGXPHYSICS_EXPORT Statistic
  {
    public:
      /**
      Updates the statistic given the current observation and returns the current value of this statistic
      \param observation - the current observation
      \return the current value of this statistic
      */
      virtual agx::Real update( agx::Real observation ) = 0;

      /**
      \return the current value of this statistic
      */
      virtual agx::Real get() const = 0;

      virtual ~Statistic() {}
  };



  /**
  Extra level of indirection provided for use by garbage collected languages. By
  packaging the actual Statistic object in a reference counted handle we work
  around the owership issue between the garbage collector in the other language
  and the unique_ptr held by e.g. agxPlot::DataSeries.
  */
  class AGXPHYSICS_EXPORT StatisticHandle : public agx::Referenced
  {
    public:
      StatisticHandle(Statistic* statistic)
          : m_statistic(statistic)
      {
      }

#ifndef SWIG
      std::unique_ptr<Statistic> takeStatistic()
      {
        return std::move(m_statistic);
      }
#endif

    protected:
      virtual ~StatisticHandle() {}

    private:
      std::unique_ptr<Statistic> m_statistic;
  };

  AGX_DECLARE_POINTER_TYPES(StatisticHandle);


  /**
  Exponential moving average statistic.
  At observation \f$Y_n\f$ the statistic \f$S_n\f$ is calculated as:
  \f$S_n = S_{n-1} + \alpha \left( Y_n - S_{n-1} \right)\f$,
  where \f$ 0 < \alpha < 1\f$ is the smoothing factor.
  */
  class AGXPHYSICS_EXPORT ExponentialMovingAverageStatistic : public Statistic
  {
    public:
      /**
      \param smoothFactor - the smoothing factor used to propagate the average (must be between zero and one)
      \param initialObservation - initial observation, default zero (optional)
      */
      ExponentialMovingAverageStatistic( agx::Real smoothFactor, agx::Real initialObservation = 0 )
        : m_smoothFactor( smoothFactor ), m_s( initialObservation ) {}

      virtual inline agx::Real update( agx::Real observation )
      {
        m_s += m_smoothFactor * ( observation - m_s );
        return m_s;
      }

      /**
      Set the raw value of the statistic scalar.
      This will be stored without any smoothing. Can be used to initialize the object to a
      specific value.
      */
      void set( agx::Real val ) { m_s = val; }

      virtual inline agx::Real get() const { return m_s; }

      /**
      \return the smoothing factor
      */
      inline agx::Real getSmoothFactor() const { return m_smoothFactor; }

      /**
      \param smoothFactor - the smoothing factor
      */
      inline void setSmoothFactor( agx::Real smoothFactor ) { m_smoothFactor = smoothFactor; }

      virtual ~ExponentialMovingAverageStatistic() {}

      static StatisticHandleRef make_statistic(
        agx::Real maxSmoothingFactor, agx::Real initialObservation = 0)
      {
        return new StatisticHandle(
          new ExponentialMovingAverageStatistic(maxSmoothingFactor, initialObservation));
      }

    protected:
      agx::Real m_smoothFactor;
      agx::Real m_s;
  };

  /**
  Variable Smooth Factor Exponential Moving Average statistic.
  During update, this class calculates a suitable smoothing factor
  to eliminate any sudden peaks in the observations (i.e., the smoothing factor goes to zero).
  */
  class AGXPHYSICS_EXPORT VariableSmoothFactorEMAStatistic : public ExponentialMovingAverageStatistic
  {
    public:
      /**
      \param maxSmoothingFactor - the maximum smoothing factor (min smoothing factor is zero)
      \param initialObservation - the initial observation
      */
      VariableSmoothFactorEMAStatistic( agx::Real maxSmoothingFactor, agx::Real initialObservation = 0 )
        : ExponentialMovingAverageStatistic( maxSmoothingFactor, initialObservation ) {}

      virtual inline agx::Real update( agx::Real observation )
      {
        agx::Real absDeltaObservation = std::abs( m_s - observation );
        agx::Real absStatistic = std::abs( m_s );
        agx::Real alpha = ( absDeltaObservation > 0 && absStatistic > 0 ? absStatistic / absDeltaObservation : m_smoothFactor );
        alpha = agx::clamp( alpha, agx::Real( 0 ), m_smoothFactor );
        m_s += alpha * ( observation - m_s );
        return m_s;
      }

      virtual ~VariableSmoothFactorEMAStatistic() {}

      static StatisticHandleRef make_statistic(
        agx::Real maxSmoothingFactor, agx::Real initialObservation = 0)
      {
        return new StatisticHandle(new VariableSmoothFactorEMAStatistic(maxSmoothingFactor, initialObservation));
      }
  };

  /**
  Median statistic that keeps a history of a number of observations and reports
  the median, i.e., middle element in sorted order, of that.
  */
  class AGXPHYSICS_EXPORT MedianStatistic : public Statistic
  {
    public:
      /**
      \param maxHistorySize - The number of observations to keep in the history.
      */
      MedianStatistic(size_t maxHistorySize)
          : m_maxHistorySize(maxHistorySize)
      {
        m_historyByAge.reserve(m_maxHistorySize);
        m_historyByValue.reserve(m_maxHistorySize);
      }

      /**
      \return The median of the history.
      */
      virtual agx::Real get() const override
      {
        size_t n = this->getCurrentHistorySize();

        if (n == 0) {
          /// \todo What should be removed here?
          return agx::Real(0.0);
        }

        size_t middle = n / 2;
        if (n % 2 == 0) {
          // Two in the middle, return their average.
          return (m_historyByValue[middle - 1] + m_historyByValue[middle]) / agx::Real(2.0);
        }
        else {
          return m_historyByValue[middle];
        }
      }

      /**
      Ad a new observation to the history, removing the oldest if the maximum
      history size has been reached.
      */
      virtual agx::Real update(agx::Real observation) override
      {
        // Remove old values.
        while (this->getCurrentHistorySize() >= this->getMaxHistorySize()) {
          agx::Real v = m_historyByAge.back();
          m_historyByAge.pop_back();
          m_historyByValue.findAndErase(v, false);
        }

        // Insert new value.
        using namespace std;
        auto insertAt = lower_bound(begin(m_historyByValue), end(m_historyByValue), observation);
        m_historyByValue.insert(insertAt, observation);
        m_historyByAge.insert(size_t(0), observation);

        return this->get();
      }


      void clear()
      {
        m_historyByAge.clear();
        m_historyByValue.clear();
      }


      size_t getMaxHistorySize() const
      {
        return m_maxHistorySize;
      }



      size_t getCurrentHistorySize() const
      {
        agxAssert(m_historyByAge.size() == m_historyByValue.size());
        size_t n = m_historyByAge.size();
        return n;
      }


      const agx::RealVector& getHistoryByAge() const
      {
        return m_historyByAge;
      }


      const agx::RealVector& getHistoryByValue() const
      {
        return m_historyByValue;
      }

      static StatisticHandleRef make_statistic(size_t maxHistorySize)
      {
        return new StatisticHandle(new MedianStatistic(maxHistorySize));
      }

    private:
      size_t m_maxHistorySize;
      agx::RealVector m_historyByAge;
      agx::RealVector m_historyByValue;
  };



}
