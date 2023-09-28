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

#include <agxSDK/SimulationParameter.h>
#include <agxCallable/CallableAccessor.h>

namespace agxCallable
{
  AGX_DECLARE_POINTER_TYPES(SimulationParameter);
  class AGXCALLABLE_EXPORT SimulationParameter : public agxSDK::SimulationParameter
  {
  public:
    SimulationParameter(const agx::String& command);

    virtual const agxData::Format *getFormat() const override;
    virtual const agxData::Value *getValue() const override;
    virtual void setValue(agxData::Value *value) override;
    virtual bool hasSetter() const override;
    virtual bool hasGetter() const override;

  protected:
    virtual ~SimulationParameter();
    virtual void addNotification(agxSDK::Simulation *simulation);
    virtual void removeNotification(agxSDK::Simulation *simulation);

  private:
    agx::String m_command;
    agxCallable::CallableAccessorRef m_callable;
  };

}
