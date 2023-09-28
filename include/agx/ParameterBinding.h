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


#ifndef AGXFN_PARAMETERBINDING_H
#define AGXFN_PARAMETERBINDING_H

#include <agxData/Scalar.h>
#include <agxData/Array.h>
#include <agx/Parameter.h>

namespace agx
{
  /**
  A binding between a formal function parameter and an actual value.
  */
  class AGXCORE_EXPORT ParameterBinding
  {
  public:
    ParameterBinding();
    ParameterBinding(agxData::Data *value, const agxData::Format *format, agx::Parameter::MetaType metaType, bool isWritable, bool allowFormatPromotion);

    agxData::Data *getValue();
    const agxData::Data *getValue() const;

    const agxData::Format *getFormat() const;

    void setValue(agxData::Data *value);
    void setFormat(const agxData::Format *format);
    void setAllowFormatPromotion(bool flag);
    void setMetaType(agx::UInt metaType);
    void setWritable(bool writable);


    agx::UInt getMetaType() const;
    bool isScalar() const;
    bool isArray() const;
    bool isPointer() const;

    bool isWritable() const;
    bool allowFormatPromotion() const;
    bool isValid() const;



    template <typename T>
    T& getScalar();

    template <typename T>
    const T& getScalar() const;


    template <typename T>
    agxData::Array<T>& getArray();

    template <typename T>
    const agxData::Array<T>& getArray() const;


    template <typename T>
    T *& getPointer();

    template <typename T>
    const T *& getPointer() const;

  private:
    agxData::Data *m_value;
    const agxData::Format *m_format;
    agx::UInt m_metaType;
    bool m_isWritable;
    bool m_allowFormatPromotion;
  };

  typedef agx::VectorPOD<ParameterBinding> ParameterBindingVector;


  /* Implementation */
  AGX_FORCE_INLINE ParameterBinding::ParameterBinding() : m_value(nullptr), m_format(nullptr), m_metaType(agx::InvalidIndex), m_isWritable(false), m_allowFormatPromotion(false)
  {}

  AGX_FORCE_INLINE ParameterBinding::ParameterBinding(agxData::Data *value, const agxData::Format *format, agx::Parameter::MetaType metaType, bool isWritable, bool allowFormatPromotion) : m_value(value), m_format(format), m_metaType(metaType), m_isWritable(isWritable), m_allowFormatPromotion(allowFormatPromotion)
  {}

  AGX_FORCE_INLINE agxData::Data *ParameterBinding::getValue() { return m_value; }
  AGX_FORCE_INLINE const agxData::Data *ParameterBinding::getValue() const { return m_value; }
  AGX_FORCE_INLINE const agxData::Format *ParameterBinding::getFormat() const { return m_format; }
  AGX_FORCE_INLINE agx::UInt ParameterBinding::getMetaType() const { return m_metaType; }
  AGX_FORCE_INLINE bool ParameterBinding::isScalar() const { return m_metaType == agx::Parameter::SCALAR; }
  AGX_FORCE_INLINE bool ParameterBinding::isArray() const { return m_metaType == agx::Parameter::ARRAY; }
  AGX_FORCE_INLINE bool ParameterBinding::isPointer() const { return m_metaType == agx::Parameter::POINTER; }
  AGX_FORCE_INLINE bool ParameterBinding::isWritable() const { return m_isWritable; }
  AGX_FORCE_INLINE bool ParameterBinding::allowFormatPromotion() const { return m_allowFormatPromotion; }
  AGX_FORCE_INLINE bool ParameterBinding::isValid() const { return m_value && m_format; }


  template <typename T>
  AGX_FORCE_INLINE T& ParameterBinding::getScalar()
  {
    agxAssert(this->isValid());
    agxAssert(m_metaType == agx::Parameter::SCALAR);
    agxAssert(m_format->is(agxData::getFormat<T>()));
    return static_cast< agxData::Scalar<T> *>(m_value)->get();
  }

  template <typename T>
  AGX_FORCE_INLINE const T& ParameterBinding::getScalar() const
  {
    return const_cast<ParameterBinding *>(this)->getScalar<T>();
  }

  template <typename T>
  AGX_FORCE_INLINE agxData::Array<T>& ParameterBinding::getArray()
  {
    agxAssert(this->isValid());
    agxAssert(m_metaType == agx::Parameter::ARRAY);
    agxAssert(m_format->is(agxData::getFormat<T>()));
    return *static_cast< agxData::Array<T> *>(m_value);
  }

  template <typename T>
  AGX_FORCE_INLINE const agxData::Array<T>& ParameterBinding::getArray() const
  {
    return const_cast<ParameterBinding *>(this)->getArray<T>();
  }

  template <typename T>
  AGX_FORCE_INLINE T *& ParameterBinding::getPointer()
  {
    agxAssert(this->isValid());
    agxAssert(m_metaType == agx::Parameter::POINTER);
    agxAssert(m_format->is(agxData::getFormat<T>()));
    static T *blah;
    return blah;
    // return static_cast< agxData::Pointer<T> *>(m_value)->get(); // ?
  }

  template <typename T>
  AGX_FORCE_INLINE const T *& ParameterBinding::getPointer() const
  {
    return const_cast<ParameterBinding *>(this)->getPointer<T>();
  }


}

#endif /* AGXFN_PARAMETERBINDING_H */
