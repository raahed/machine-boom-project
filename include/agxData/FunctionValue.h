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

#ifndef AGX_FUNCTIONVALUE_H
#define AGX_FUNCTIONVALUE_H

#include <agxData/Value.h>
#include <agx/Task.h>
#include <agx/UnresolvedTask.h>

namespace agxData
{
  class AGXCORE_EXPORT FunctionValue : public agxData::Value
  {
  public:
    // FunctionValue(const Name& name, const agxData::Type *type, Task *function);
    FunctionValue(const agxData::Type *type, const agx::String& expression);
    FunctionValue(const agxData::Format *format, const agx::String& expression);

    void injectFunctionInstance(agx::UnresolvedTask *proxy);

    virtual void rebind();

    void setContext(agx::Object *component);

  protected:
    virtual ~FunctionValue();

  private:
    void parameterUpdateCallback(agx::Parameter *parameter);

  private:
    agx::TaskRef m_function;
    agx::Parameter::Event::CallbackType m_parameterCallback;
    agx::ParameterRef m_result;
  };
}

#endif /* _AGX_FUNCTIONVALUE_H_ */
