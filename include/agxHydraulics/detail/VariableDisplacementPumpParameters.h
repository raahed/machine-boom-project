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


#ifndef AGXHYDRAULICS_INTERNAL_UTILS_H
#define AGXHYDRAULICS_INTERNAL_UTILS_H

#include <agx/Referenced.h>

#include <agxHydraulics/FlowUnit.h>

namespace agxHydraulics
{
  class Pipe;
  class Pump;
  class PumpUnit;

  namespace detail
  {
    AGX_DECLARE_POINTER_TYPES(VariableDisplacementPumpParameters);

    /**
     * \internal
     * Computes derived parameters from the ones supplied by the user.
     */
    class AGXHYDRAULICS_EXPORT VariableDisplacementPumpParameters : public agx::Referenced
    {
      public:
        VariableDisplacementPumpParameters(
            agx::Real cutoffPressure, agx::Real deadheadPressure,
            agx::Real internalLeakage, agx::Real flowRatio,
            agx::Real compensatorArea)
        :
            m_cutoffPressure(cutoffPressure),
            m_deadheadPressure(deadheadPressure),
            m_internalLeakage(internalLeakage),
            m_flowRatio(flowRatio),
            m_compensatorArea(compensatorArea),

            m_springConstant(force(deadheadPressure) - force(cutoffPressure)),
            m_noFlowSpringElongation(elongation(deadheadPressure)),
            m_fullFlowSpringElongation(elongation(cutoffPressure)),
            m_noFlowPoppetPosition(poppetPosition(deadheadPressure)),
            m_fullFlowPoppetPosition(poppetPosition(cutoffPressure))
        {
  //        this->print(std::cout);
          agxVerify(agx::equivalent(m_noFlowSpringElongation - m_fullFlowSpringElongation, agx::Real(1)));
        }

        agx::Real cutoffPressure() const {  return m_cutoffPressure;  }
        agx::Real deadheadPressure() const { return m_deadheadPressure; }
        agx::Real internalLeakage() const { return m_internalLeakage; }
        agx::Real flowRatio() const { return m_flowRatio; }
        agx::Real compensatorArea() const { return m_compensatorArea; }
        agx::Real springConstant() const { return m_springConstant; }
        agx::Real noFlowSpringElongation() const { return m_noFlowSpringElongation; }
        agx::Real fullFlowSpringElongation() const { return m_fullFlowSpringElongation; }
        agx::Real springRestPosition() const { return m_noFlowSpringElongation; }
        agx::Real noFlowPoppetPosition() const { return m_noFlowPoppetPosition; }
        agx::Real fullFlowPoppetPosition() const { return m_fullFlowPoppetPosition; }

        agx::Real force(agx::Real pressure)
        {
          return pressure * m_compensatorArea;
        }

        agx::Real elongation(agx::Real pressure)
        {
          return force(pressure) / m_springConstant;
        }

        agx::Real poppetPosition(agx::Real pressure)
        {
          return m_noFlowSpringElongation - elongation(pressure);
        }


        void print(std::ostream& stream) const;



      private:
        /**
        The pump parameters is a non-assignable struct because all it's members
        are const. Use copy construction if you want a copy.

        /// \todo Use "= delete" here when Visual Studio supports it.
        */
        VariableDisplacementPumpParameters& operator=(const VariableDisplacementPumpParameters&)
        {
          return *this;
        }

      private:
        const agx::Real m_cutoffPressure;
        const agx::Real m_deadheadPressure;
        const agx::Real m_internalLeakage;
        const agx::Real m_flowRatio;
        const agx::Real m_compensatorArea;

        const agx::Real m_springConstant;
        const agx::Real m_noFlowSpringElongation;
        const agx::Real m_fullFlowSpringElongation;
        const agx::Real m_noFlowPoppetPosition;
        const agx::Real m_fullFlowPoppetPosition;
    };


    std::ostream& operator<<(std::ostream& stream, const VariableDisplacementPumpParameters& pumpParameters);

  } // namespace detail

} // namespace agxHydraulics



#endif
