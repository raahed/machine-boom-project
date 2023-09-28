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

#ifndef AGXDATA_TYPE2_H
#define AGXDATA_TYPE2_H

#include <typeinfo>
#include <agxData/agxData.h>
#include <agxData/Format.h>
#include <agx/Vector.h>
#include <agx/HashTable.h>
#include <agx/HashVector.h>
#include <agx/Integer.h>
#include <agx/Referenced.h>
#include <agx/Object.h>
#include <agx/AtomicValue.h>
#include <agx/SpinMutex.h>

namespace agx
{
  class TiXmlElement;
  class Device;
  class Object;
}

namespace agxData
{
  #define AGX_TYPE_NAME_MAX_LENGTH 1023

  AGX_DECLARE_POINTER_TYPES(Type);

  typedef agx::HashVector<agx::Name, TypeRef > TypeTable;
  typedef agx::HashTable<agx::Name, FormatRef> FormatTable;

  /**
  \return The type with a specified name.
  */
  AGXCORE_EXPORT Type *getType(const agx::String& name);
  AGXCORE_EXPORT Type *getOrCreateType(const agx::String& name);


  /**
  \return The format with a specified name, using the "type:format" syntax.
  */
  AGXCORE_EXPORT Format* getFormat(const agx::String& typeFormatName);
  AGXCORE_EXPORT Format* getFormat(agx::TiXmlElement* eTypeFormat, agx::Device* device = nullptr);

  /**
  \return A generic format, where only the size of the format is important.
  */
  AGXCORE_EXPORT Format *getGenericStructFormat(size_t numBytes);

  /**
  \return The type with a specified id.
  */
  AGXCORE_EXPORT Type *getType(agx::UInt id);

  /**
  \return The abstract type corresponding to a language type.
  */
  template <typename T>
  Type *getType();

  /**
  \return The abstract format corresponding to a language type.
  */
  template <typename T>
  Format *getFormat();


  /**
  \return All registered types.
  */
  const TypeTable& getTypes();

  /**
  Print all types and formats to std::cout.
  */
  void printAllTypesAndFormats();

  /**
  Abstracted type.
  */
  class AGXCORE_EXPORT Type : public agx::Model
  {
  public:
    static agx::Model *ClassModel();

  public:
    Type(const agx::Name& name);

    /**
    \return The id of the type.
    */
    agx::UInt getId() const;

    /**
    \return All registered formats for this type.
    */
    const FormatTable& getFormats() const;

    /**
    \return The format with a specified name.
    */
    Format *getFormat(const agx::Name& name);
    const Format *getFormat(const agx::Name& name) const;

    /**
    \return The format corresponding to a specific C++ type.
    */
    template <typename T>
    Format *getFormat();

    template <typename T>
    const Format *getFormat() const;

    /**
    \return The default type format.
    */
    Format *getDefaultFormat();
    const Format *getDefaultFormat() const;

    void addFormat(Format *format, bool isDefault = false);
    void setDefaultFormat(Format *format);
  protected:
    virtual ~Type();

  private:
    friend class Format;
    // void registerFormat(const Format *format, bool isDefault) const;

  private:
    agx::UInt m_id;
    FormatRef m_defaultFormat;
    FormatTable m_formats;
  };


  /**
  Type binding, templated type -> abstract type.
  */
  template <typename T>
  struct TypeBinding
  {
    static Type *getType()
    {
      Type *type = agxData::getOrCreateType(typeid(T).name());
      if (!type->getDefaultFormat())
        type->addFormat(new FormatT<T>("default"), true);

      return type;
    }
  };

  template <typename T>
  struct FormatBinding
  {
    static Format *getFormat()
    {
      Type *type = getType<T>();
      agxAssertN(type->getFormat<T>(), "No format \'%s\' in type \'%s\'", typeid(T).name(), type->getName().c_str());
      return type->getFormat<T>();
    }
  };

  #define AGX_TYPE_BINDING(_Type, _Name)         \
  namespace agxData                              \
  {                                              \
    template <>                                  \
    struct TypeBinding<_Type>                    \
    {                                            \
      static Type *getType()                     \
      {                                          \
        return agxData::getOrCreateType(_Name);  \
      }                                          \
    };                                           \
  }

  #define AGX_TEMPLATED_TYPE_BINDING(_Type, _Name)                                                                                  \
  namespace agxData                                                                                                                 \
  {                                                                                                                                 \
    template <typename T>                                                                                                           \
    struct TypeBinding< _Type<T> >                                                                                                  \
    {                                                                                                                               \
      static Type *getType()                                                                                                        \
      {                                                                                                                             \
        Format *format = agxData::getFormat<T>();                                                                                   \
        const agx::String& elementName = format->getName() == "default" ? format->getType()->getName().str() : format->fullName();  \
        agx::String typeName = _Name "<" + elementName + ">";                                                                       \
        Type *type = agxData::getOrCreateType(typeName);                                                                            \
                                                                                                                                    \
        if (!type->getDefaultFormat())                                                                                              \
          type->addFormat(new FormatT< _Type<T> >("default"), true);                                                                \
                                                                                                                                    \
        return type;                                                                                                                \
      }                                                                                                                             \
    };                                                                                                                              \
                                                                                                                                    \
    template <typename T>                                                                                                           \
    struct FormatBinding< _Type<T> >                                                                                                \
    {                                                                                                                               \
      static Format *getFormat()                                                                                                    \
      {                                                                                                                             \
        return getType< _Type<T> >()->getDefaultFormat();                                                                           \
      }                                                                                                                             \
    };                                                                                                                              \
  }



  /* Implementation */
  inline agx::UInt Type::getId() const { return m_id; }
  inline const FormatTable& Type::getFormats() const { return m_formats; }
  inline Format *Type::getDefaultFormat() { return m_defaultFormat; }
  inline const Format *Type::getDefaultFormat() const { return m_defaultFormat; }

  inline Format *Type::getFormat(const agx::Name& name)
  {
    FormatTable::iterator it = m_formats.find(name);
    return it == m_formats.end() ? nullptr : it->second;
  }

  inline const Format *Type::getFormat(const agx::Name& name) const
  {
    return const_cast<Type *>(this)->getFormat(name);
  }



  template <typename T>
  Format *Type::getFormat()
  {
    for (FormatTable::iterator it = m_formats.begin(); it != m_formats.end(); ++it)
      if (it->second->getImplementationName() == typeid(T).name())
        return it->second;

    return nullptr;
  }

  template <typename T>
  const Format *Type::getFormat() const
  {
    return const_cast<Type *>(this)->getFormat<T>();
  }



  template <typename T>
  Type *getType()
  {
    return TypeBinding<T>::getType();
  }


  template <typename T>
  Format *getFormat()
  {
    return FormatBinding<T>::getFormat();
  }

}

#include <agx/IntegerTemplates.h>
#include <agx/RealTemplates.h>
#include <agx/IndexRangeTemplates.h>

AGX_TYPE_BINDING(agx::String, "String")
AGX_TYPE_BINDING(agx::Name, "Name")
AGX_TYPE_BINDING(agx::Object *, "ObjectPointer")
AGX_TYPE_BINDING(agx::AtomicValue, "AtomicValue")
AGX_TYPE_BINDING(agx::SpinMutex, "SpinMutex")

AGX_TEMPLATED_TYPE_BINDING(agx::Vector, "Vector")
AGX_TEMPLATED_TYPE_BINDING(agx::HashSet, "HashSet")

// AGX_TYPE_BINDING(void *, "Pointer")

namespace agxData
{
  template <typename T>
  struct TypeBinding<T*>
  {
    static Type *getType()
    {
      return agxData::getOrCreateType("Pointer");
    }
  };

  template <typename T>
  struct FormatBinding<T*>
  {
    static Format *getFormat()
    {
      return getType<T*>()->getDefaultFormat();
    }
  };

  template <> AGX_FORCE_INLINE Format *getFormat<void>() { return Format::Void(); }
  template <> AGX_FORCE_INLINE Format *getFormat<agx::String>() { return agxData::getType<agx::String>()->getDefaultFormat(); }
}

#endif /* _AGXDATA_TYPE2_H_ */
