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

#include <agx/Vec3.h>
#include <agxModel/Controller1D.h>
#include <agxStream/Serializable.h>
#include <agxModel/export.h>

#include <deque>

namespace agxModel
{

  AGX_DECLARE_POINTER_TYPES(PidController1D);
  /**
  A PID controller with the classic gain coefficients for proportional, integral, and derivative gain.

  The PID controller has an integral windup algorithm with both "clamping" and/or an integral
  back-calculation algorithm. The backward calculation is done by recomputing the integral term
  to its previous state before the windup started and by that disabling the integration.

  The PID controller may also add constraints on the Manipulated Variable (MV) and/or the acceleration of the Manipulated Variable (dMV/dt).
  For example, the Manipulated Variable limits might be set to a winch max/min speed and max/min force.

  To reduce noise in the time derivative of the error, dE/dt, the PID controller could use a longer time interval to differentiate the error, e.g. the five-point stencil method.
  */
  class AGXMODEL_EXPORT PidController1D : public Controller1D
  {
    public:
      enum class DerivationMethod : agx::UInt32
      {
        FIRSTORDER = 1 << 0,
        FIVEPOINTSTENCIL = 1 << 1,
        THREEPOINTSKEWEDSTENCIL = 1 << 2,
      };

      AGX_DECLARE_POINTER_TYPES(Derivative);

      /**
      Differentiate a PID controller signal with a two point estimation.
      */
      class AGXMODEL_EXPORT Derivative : public agx::Referenced
      {
      public:
        Derivative() : m_lastTime(0), m_lastValue(0) {};

        virtual agx::Real calculate(const agx::TimeStamp& time, agx::Real value);

      protected:
        agx::TimeStamp m_lastTime;
      private:
        agx::Real m_lastValue;
      };

      AGX_DECLARE_POINTER_TYPES(DerivativeFivePointStencil);
      /**
      Differentiate a PID controller signal with the "five-point stencil" method. This will use 5 points
      to calculate the derivative centered around -2*h in the chain of time steps (-4*h, -3*h, -2*h, -1*h, 0),
      where h is the time step. Note that this will filter high frequency signals but introduce a 2 time step
      lag in the derivative.
      */
      class AGXMODEL_EXPORT DerivativeFivePointStencil : public Derivative
      {
      public:
        DerivativeFivePointStencil() : m_fivePoints(5, agx::Real(0)) {};
        /// Calculate five-point stencil derivative.
        /// \param time The current time. Note that only regular time steps are allowed
        /// \param value The current function value
        virtual agx::Real calculate(const agx::TimeStamp& time, agx::Real value) override;

      private:
        std::vector<agx::Real> m_fivePoints;
      };

      AGX_DECLARE_POINTER_TYPES(DerivativeThreePointSkewdStencil);
      /**
      Differentiate a PID controller signal with the "three-point stencil" method. This will the 3 last points
      to calculate the derivative in the chain of time steps (-2*h, -1*h, 0), where h is the time step.
      */
      class AGXMODEL_EXPORT DerivativeThreePointSkewedStencil : public Derivative
      {
      public:
        DerivativeThreePointSkewedStencil() : m_threePoints(3, agx::Real(0)) {};
        /// Calculate three-point skewed stencil derivative.
        /// \param time The current time. Note that only regular time steps are allowed
        /// \param value The current function value
        virtual agx::Real calculate(const agx::TimeStamp& time, agx::Real value) override;
      private:
        std::vector<agx::Real> m_threePoints;
      };

    public:
      PidController1D();

      /// Update the measured Process Variable and calculate a new value for the Manipulated Variable.
      /// Calculate the function u(t) = K_p * e(t) + K_i * Int( e(tau ) dtau, 0..t) + K_d * de(t)/dt, where
      /// e is the error between the measured Process Variable and the Set Point
      /// \param measuredProcessVariable The Plant measured Process Variable, PV in a Control System
      virtual void update(const agx::TimeStamp& time, agx::Real measuredProcessVariable) override;

      /// \return The Manipulated Variable (MV), or the output signal, from the PID controller.
      /// The Manipulated Variable is calculated in the last call to update().
      /// If no call to update is done then 0 is returned.
      virtual agx::Real getManipulatedVariable() const override;

      /// Set the current Set Point for the controller
      virtual void setSetPoint(agx::Real setPoint) override;

      /// \return The current Set Point for the controller
      virtual agx::Real getSetPoint() const override;

      /// Reset calculated gain, error, and cost to 0. Note that this will keep the Set Point and Manipulated Variable as it is.
      /// The PID controller keeps track of the simulation time and this simulation time will not be reset to 0 when this method
      /// is called.
      virtual void reset() override;

      /// \return The current value of the measured Process Variable as given in the call to update.
      agx::Real getMeasuredProcessVariable() const;

      /// Limit the Manipulated Variable between this lower and upper limit. Default limits are +/- agx::Real max value.
      void setManipulatedVariableLimit(agx::Real lowerLimit, agx::Real upperLimit);

      /// \return The upper and lower limit of the Manipulated Variable. First value in the returned  vector is the lower limit
      /// and the second value in the vector is the upper limit.
      const agx::Vec2& getManipulatedVariableLimit() const;

      /// Limit the Manipulated Variable acceleration dMV/dt between this lower and upper limit. Default limits are +/- agx::Real max value.
      void setManipulatedVariableAccelerationLimit(agx::Real lowerLimit, agx::Real upperLimit);

      /// \return The upper and lower limit of the Manipulated Variable acceleration limitation. First value in the returned vector is the lower limit
      /// and the second value in the vector is the upper limit.
      const agx::Vec2& getManipulatedVariableAccelerationLimit() const;

      /// Enable or disable the integral windup backward calculation. Note that the anti windup algorithm is on by default.
      /// If this is disabled then the integral gain will not be limited when the Manipulated Variable is outside
      /// its limits defined by getManipulatedVariableLimit(), and by that cause an integral windup.
      /// The backward calculation is done by recomputing the integral term to its previous state before the windup started
      /// and by that disabling the integration.
      void setIntegralWindupProtection(bool doIntegralWindupProtection);

      bool hasEnabledIntegralWindupProtection() const;

      /// Set the lower and upper limit of the integral gain term, getIntegraTerm().
      void setIntegralWindupClamping(agx::Real lowerLimitIntegralTerm, agx::Real upperLimitIntegralTerm);

      /// \return the value of the upper and lower limit of the integral gain term. First value in the vector is the lower limit
      /// and the second value in the vector is the upper limit.
      const agx::Vec2& getIntegralWindupClamping() const;

      /// Enable the PID controller.
      void enablePidController();

      /// Disable the PID controller. The Manipulated Value will be zero.
      void disablePidController();

      /// \return return true if the PID controller is enabled. Default is true.
      bool isEnabled();

      /// \return The gains stored in a vector, where x is proportional gain K_p, y is integral gain K_i and z is derivative gain K_d
      agx::Vec3 getGains() const;

      /// Set the gains
      void setGains(agx::Real proportionalGain, agx::Real integralGain, agx::Real derivativeGain);

      /// \return The proportional gain, K_p
      agx::Real getProportionalGain() const;

      /// Set the proportional gain, K_p
      void setProportionalGain(agx::Real proportionalGain);

      /// \return The integral gain, K_i
      agx::Real getIntegralGain() const;

      /// Set the integral gain, K_i
      void setIntegralGain(agx::Real integralGain);

      /// \return the integral time defined as T_i = K_p / K_i. If K_i is zero then agx::RealMax is returned.
      agx::Real getIntegralTime();

      /// Set the integral time defined as T_i = K_p / K_i. This call will update the integral gain coefficient, K_i.
      /// If T_i is set to agx::Real::maxval then K_i is set to 0. If T_i is set to 0 then K_i is set to agx::RealMax
      /// If the proportional gain coefficient is zero then K_i will be set to zero as well regardless of the value of the integralTime.
      void setIntegralTime(agx::Real integralTime);

      /// \return The derivative gain, K_d
      agx::Real getDerivativeGain() const;

      /// Set the derivative gain, K_d
      void setDerivativeGain(agx::Real derivativeGain);

      /// \return the derivative time defined as T_d = K_d / K_p. If the proportional gain is zero the agx::RealMax is returned.
      agx::Real getDerivativeTime();

      /// Set the derivative time defined as T_d = K_d / K_p. This will update the derivative gain, K_d.
      /// If the proportional gain is zero then K_d will be set to zero as well regardless of the value of the derivativeTime.
      /// If the derivative time is zero the derivative gain is set to zero.
      void setDerivativeTime(agx::Real derivativeTime);

      /// Set PID frequency for gain compensation of the derivative and integral term, default values is 1.
      void setFrequency(agx::Real frequency);

      /// \return PID frequency for gain compensation of the derivative and integral term
      agx::Real getFrequency() const;

      /// \return The error between the Process Variable and the Set Point, e = SP - PV
      agx::Real getError() const;

      /// \return The current proportional term, K_p * e(t)
      agx::Real getProportionalTerm() const;

      /// \return The current integral term, K_i * Int( e(tau ) dtau, 0..t) * f, where f is the defined PID frequency
      agx::Real getIntegraTerm() const;

      /// \return The current derivative term, K_d * de(t)/dt * f, where f is the defined PID frequency
      agx::Real getDerivativeTerm() const;

      /// This will calculate the cost function value between the last time update() was called and
      /// last time minus deltaTime so that (t - t0) = deltaTime, where t is the last time update() was called.
      /// Note that the cost function history is not infinite and is truncated at 1000 time steps.
      ///
      /// \return The cost function value, 1 / (t - t0) * Int( e(tau )^2 dtau, t0..t)
      /// \param deltaTime This will determine t0 so that t0 = t - deltaTime, where t is the last time update() was called.
      agx::Real calculateCostFunctionValue(agx::Real deltaTime) const;

      /// Change the derivation method. Default is the first order numerical derivation method.
      void setDerivationMethod(Derivative *derivationMethod);

      /// Change the derivation method. Default is the first order numerical derivation method DerivationMethod::FirstOrder
      void setDerivationType(DerivationMethod derivationMethod);

    private:
      agx::Real integralWindupLimitAcceleration(agx::Real manipulatedVariablePrevious, agx::Real dt) const;

      agx::Real integralWindupDisableIntegralCalculation(agx::Real u, agx::Real error, agx::Real dt, agx::Real& integralGainTerm) const;

      agx::Real integralWindupClamping(agx::Real integralGainTerm, agx::Vec2 limits) const;

      void calculateCost(const agx::TimeStamp& time, const agx::TimeStamp& dt, agx::Real error);


    private:
      agx::TimeStamp m_lastTime;
      agx::Real m_manipulatedVariable;
      agx::Real m_setPoint;
      agx::Real m_error;
      agx::Real m_processVariable;
      agx::Vec2 m_manipulatedVariableLimit;
      agx::Vec2 m_manipulatedVariableAccelerationLimit;
      agx::Real m_proportionalGainCoefficient;
      agx::Real m_integralGainCoefficient;
      agx::Real m_derivativeGainCoefficient;
      // scale derivative/integral gain with this frequency, default is 1
      agx::Real m_frequency;

      std::deque<agx::Real> m_cost;
      std::deque<agx::TimeStamp> m_costTime;

      // Current value of integral gain, Int(e, 0..t), without coefficient
      agx::Real m_integralGain;
      // Current value of derivative gain, dE/dt, without coefficient
      agx::Real m_derivativeGain;
      agx::Vec2 m_integralTermLimit;

      bool m_isEnabled;
      bool m_doIntegralWindupProtection;
      DerivativeRef m_derivative;

      static const int MAX_COST_HISTORY;
  };
}

