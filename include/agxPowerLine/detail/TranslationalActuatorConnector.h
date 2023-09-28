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

#ifndef AGXPOWERLINE_TRANSLATIONAL_ACTUATOR_CONNECTOR_H
#define AGXPOWERLINE_TRANSLATIONAL_ACTUATOR_CONNECTOR_H

#include <agxPowerLine/ActuatorConnector.h>

/// \cond INTERNAL_DOCUMENTATION

namespace agxPowerLine
{
  class TranslationalActuator;

  namespace detail
  {
    AGX_DECLARE_POINTER_TYPES(TranslationalActuatorConnector);

    class AGXMODEL_EXPORT TranslationalActuatorConnector : public agxPowerLine::ActuatorConnector
    {
      public:
        TranslationalActuatorConnector(agxPowerLine::TranslationalActuator* actuator);

        virtual bool getReverseOrder() const override;

        AGXSTREAM_DECLARE_SERIALIZABLE(agxPowerLine::detail::TranslationalActuatorConnector);


      // Methods called by the rest of the PowerLine framework.
      public:
        virtual agxPowerLine::PhysicalDimensionMultiBodyConstraintImplementation* createConstraint() override;

        virtual agx::RegularizationParameters::VariableType calculateComplianceAndDamping(
            const agx::Real timeStep, agx::Real& compliance, agx::Real& damping) override;

        virtual agx::Real calculateViolation() const override;

        virtual bool store(agxStream::StorageStream& out) const override;
        virtual bool restore(agxStream::StorageStream& in) override;

      protected:
        TranslationalActuatorConnector();
        virtual ~TranslationalActuatorConnector() {}

      private:
        agx::observer_ptr<agxPowerLine::TranslationalActuator> m_translationalActuator;
    };
  }
}

/// \endcond

#endif
