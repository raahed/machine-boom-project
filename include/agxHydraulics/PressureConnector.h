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



#ifndef AGXHYDRAULICS_PRESSURE_CONNECTOR_H
#define AGXHYDRAULICS_PRESSURE_CONNECTOR_H

#include <agx/Real.h>
#include <agxPowerLine/Connector.h>
#include <agxHydraulics/export.h>


namespace agxHydraulics
{
  AGX_DECLARE_POINTER_TYPES(PressureConnector);
  AGX_DECLARE_VECTOR_TYPES(PressureConnector);

  /**
  The PressureConnector is an abstract base class for all Connectors that has
  a pressure.
  */
  class AGXHYDRAULICS_EXPORT PressureConnector : public agxPowerLine::Connector
  {
    public:
      /**
      \return The current pressure generated by the PressureConnector.
      */
      virtual agx::Real getPressure() const = 0;


    // Methods called by the rest of the PowerLine/Hydraulics frame work.
    public:
#ifndef SWIG
      /**
      Stores internal data into stream.
      */
      virtual bool store(agxStream::StorageStream& str) const;

      using agxPowerLine::Connector::store;

      /**
      Restores internal data from stream.
      */
      virtual bool restore(agxStream::StorageStream& str);

      using agxPowerLine::Connector::restore;
#endif
    protected:
      PressureConnector();
      virtual ~PressureConnector();

  };
}


#endif
