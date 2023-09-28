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

#ifndef AGX_CALLABLERECORDER_H
#define AGX_CALLABLERECORDER_H

#include <agxCallable/export.h>
#include <agx/Referenced.h>
#include <agxData/Buffer.h>
#include <agxData/Value.h>

namespace agxSDK { class Simulation; }

#ifdef _MSC_VER
#  pragma warning(push)
#  pragma warning(disable: 4512) // warning C4512: assignment operator could not be generated
#endif


namespace agxCallable
{
  AGX_DECLARE_POINTER_TYPES(CallableAccessor);
  AGX_DECLARE_VECTOR_TYPES(CallableAccessor);
  class AGXCALLABLE_EXPORT CallableAccessor : public agx::Referenced {
  public:
    static CallableAccessor *create(agxSDK::Simulation *sim, const agx::String& cmd);

  public:
    /// \return The command string bound to this callable
    virtual const agx::String& getCommandString() const = 0;

    /// \return The current value from the callable
    virtual const agxData::Value *getCurrentValue() = 0;

    /// Read the current value and append it to the record-buffer
    virtual void recordValue() = 0;

    /// \return The buffer of recorded values
    virtual const agxData::Buffer *getRecordedValues() const = 0;

    /// Set a new value on the callable
    virtual void setValue(agxData::Value *value) = 0;

    /// \return The format of the callable
    virtual const agxData::Format *getFormat() const = 0;

    /// \return True if setter functionality is available
    virtual bool hasSetter() const = 0;

    /// \return True if getter functionality is available
    virtual bool hasGetter() const = 0;

  private:
  };

  // Template specialization because callable setters take "const T&" or just "T" type arguments depending on type
  template <typename T> struct SetterT { typedef const T& Type; };
  template <> struct SetterT<agx::Real> { typedef agx::Real Type; };



  template <typename ObjectT, typename ValueT>
  class CallableAccessorT : public CallableAccessor {
  public:
    CallableAccessorT(ObjectT *object, const agx::String& cmd, const agx::String& fullCmd);

    virtual const agx::String& getCommandString() const override;
    virtual const agxData::Value *getCurrentValue() override;
    virtual void recordValue() override;
    virtual const agxData::Buffer *getRecordedValues() const override;
    virtual void setValue(agxData::Value *value) override;
    virtual const agxData::Format *getFormat() const override;
    virtual bool hasSetter() const override;
    virtual bool hasGetter() const override;

  private:
    ObjectT *m_object;
    const agx::String m_cmd;
    std::function<ValueT()> m_get;
    std::function<void(typename SetterT<ValueT>::Type)> m_set;
    agxData::ValueRefT<ValueT> m_currentValue;
    agxData::BufferRef m_values;
  };

#ifdef _MSC_VER
#  pragma warning(pop)
#endif


}
#endif //AGX_CALLABLERECORDER_H
