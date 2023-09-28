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

#ifndef AGX_CALLALBLEACCESSOR_IMPL_H
#define AGX_CALLALBLEACCESSOR_IMPL_H

#include <agxCallable/CallableAccessor.h>
#include <agxCallable/Callable.h>

namespace agxCallable
{
  template <typename ObjectT, typename ValueT>
  CallableAccessorT<ObjectT, ValueT>::CallableAccessorT(ObjectT *object, const agx::String& cmd, const agx::String& fullCmd) : m_object(object), m_cmd(fullCmd)
  {
    m_get = agxCallable::makeCallable< ValueT() >(object, cmd);
    m_set = agxCallable::makeCallable< void(typename SetterT<ValueT>::Type) >(object, cmd);
    agxVerify(m_get);

    m_currentValue = new agxData::ValueT<ValueT>(m_cmd);
    m_values = new agxData::BufferT<ValueT>(m_cmd);
  }

  template <typename ObjectT, typename ValueT>
  const agx::String& CallableAccessorT<ObjectT, ValueT>::getCommandString() const
  {
    return m_cmd;
  }

  template <typename ObjectT, typename ValueT>
  const agxData::Format *CallableAccessorT<ObjectT, ValueT>::getFormat() const
  {
    return m_currentValue->getFormat();
  }

  template <typename ObjectT, typename ValueT>
  bool CallableAccessorT<ObjectT, ValueT>::hasSetter() const {
    return m_set != nullptr;
  }

  template <typename ObjectT, typename ValueT>
  bool CallableAccessorT<ObjectT, ValueT>::hasGetter() const {
    return m_get != nullptr;
  }

  template <typename ObjectT, typename ValueT>
  const agxData::Value *CallableAccessorT<ObjectT, ValueT>::getCurrentValue()
  {
    m_currentValue->set(m_get());
    return m_currentValue.get();
  }

  template <typename ObjectT, typename ValueT>
  void CallableAccessorT<ObjectT, ValueT>::setValue(agxData::Value *value)
  {
    if (!m_set)
      throw std::runtime_error(agx::String::format("No setter for %s", m_cmd.c_str()).c_str());

    auto val = value->get<ValueT>();
    m_set(val);
  }

  template <typename ObjectT, typename ValueT>
  void CallableAccessorT<ObjectT, ValueT>::recordValue()
  {
    m_values->as<agxData::BufferT<ValueT>>()->push_back(m_get());
  }

  template <typename ObjectT, typename ValueT>
  const agxData::Buffer *CallableAccessorT<ObjectT, ValueT>::getRecordedValues() const
  {
    return m_values;
  }

}

#endif //AGX_CALLALBLEACCESSOR_IMPL_H
