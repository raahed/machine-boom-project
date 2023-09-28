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

#ifndef AGXPOWERLINE_WIRE_WINCH_CONNECTOR_H
#define AGXPOWERLINE_WIRE_WINCH_CONNECTOR_H

#include <agxPowerLine/ActuatorConnector.h>
#include <agxPowerLine/detail/WireWinchConnectorConstraints.h>


namespace agxPowerLine
{
  class WireWinchActuator;

  namespace detail
  {

    AGX_DECLARE_POINTER_TYPES(WireWinchConnector);


    class AGXMODEL_EXPORT WireWinchConnector : public agxPowerLine::ActuatorConnector
    {
      public:
        WireWinchConnector(agxPowerLine::WireWinchActuator* winchActuator);

        virtual bool getReverseOrder() const override;


        AGXSTREAM_DECLARE_SERIALIZABLE(agxPowerLine::detail::WireWinchConnector);




      // Methods called by the rest of the PowerLine framework.
      public:
        /// \cond INTERNAL_DOCUMENTATION
        virtual agxPowerLine::PhysicalDimensionMultiBodyConstraintImplementation* createConstraint() override;

        virtual agx::RegularizationParameters::VariableType calculateComplianceAndDamping(const agx::Real timeStep, agx::Real& compliance, agx::Real& damping) override;

        /**
        Called by the PowerLine during store. Stores internal data into stream.
        */
        virtual bool store(agxStream::StorageStream& stream) const override;

        /**
        Called by the PowerLine during restore. Restores internal data from
        stream.
        */
        virtual bool restore(agxStream::StorageStream& stream) override;
        /// \endcond

      protected:
        /// \cond INTERNAL_DOCUMENTATION
        WireWinchConnector();
        virtual ~WireWinchConnector() {};
        /// \endcond

      protected:
        agx::observer_ptr<agxPowerLine::WireWinchActuator> m_wireWinchActuator;
    };
  }
}

#endif
