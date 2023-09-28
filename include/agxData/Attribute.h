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

#ifndef AGXDATA_ATTRIBUTE_H
#define AGXDATA_ATTRIBUTE_H

#ifdef _MSC_VER
# pragma warning(push)
// warning C4505: 'agxData::VectorAttributeT<T>::print' : unreferenced local function has been removed
# pragma warning( disable : 4505 )
#endif

#include <agx/agx.h>
#include <agx/agxCore_export.h>
#include <agx/String.h>
#include <agx/Object.h>
#include <agx/Vector.h>
#include <agxData/Value.h>
#include <iosfwd>


namespace agx
{
  class TiXmlElement;
}

namespace agxData
{
  class EntityModel;
  class Type;

  template<typename T>
  class Array;

  AGX_DECLARE_POINTER_TYPES(Attribute);
  AGX_DECLARE_POINTER_TYPES(ScalarAttribute);
  AGX_DECLARE_POINTER_TYPES(ArrayAttribute);
  AGX_DECLARE_POINTER_TYPES(SharedAttribute);


  /**
  An abstract attribute description, part of an Entity.
  */
  class AGXCORE_EXPORT Attribute : public agx::Object
  {
  public:
    static agx::Model *ClassModel();

  public:

    /**
    Constructor.
    \param name The name of the attribute.
    \param type The data type of the attribute.
    */
    Attribute(const agx::Name& name, agxData::Type *type, agxData::Format *defaultFormat = nullptr, agxData::Value *defaultValue = nullptr, agxData::Value *numElements = nullptr);

    /**
    \return the id of this attribute (unique in the pools which the attribute is part of)
    */
    agx::Index getId() const;

    /**
    \return the data type of the attribute.
    */
    agxData::Type *getType();
    const agxData::Type *getType() const;

    /**
    \return The full name, "entityagx::Name.attributeagx::Name"
    */
    agx::String fullName() const;

    /**
    \return the default data format for the attribute.
    */
    agxData::Format *getDefaultFormat();
    const agxData::Format *getDefaultFormat() const;

    /**
    \return The number of elements per attribute instance, default is 1.
    */
    agxData::Value *getNumElements();
    const agxData::Value *getNumElements() const;

    /**
    \return The data initializer.
    */
    agxData::Value *getDefaultValue();
    const agxData::Value *getDefaultValue() const;


    /**
    \return The entity which the attribute is part of. nullptr if not an entity attribute.
    */
    EntityModel *getEntity() const;

    void setDefaultFormat(agxData::Format *format);
    void setDefaultValue(agxData::Value *value);

  protected:
    virtual ~Attribute();
    Attribute() : Object(""), m_entity(0) {}
    friend class EntityModel;
    friend class EntityStorage;
    void setId(agx::Index id);
    void setEntity(EntityModel *entity);

  private:
    agx::Index m_id;
    TypeRef m_type;
    FormatRef m_defaultFormat;
    agxData::ValueRef m_defaultValue;
    agxData::ValueRef m_numElements;
    EntityModel *m_entity;
  };


  std::ostream& operator << (std::ostream& output, const Attribute& attribute);


  /**
  A scalar attribute.
  */
  class AGXCORE_EXPORT ScalarAttribute : public Attribute
  {
  public:
    static ScalarAttribute *load(agx::TiXmlElement *eAttribute);
    static agx::Model *ClassModel();

    ScalarAttribute(
      const agx::Name& name, Type* type, Format* defaultFormat = nullptr, agxData::Value* defaultValue = nullptr,
      agxData::Value* numElementsExpression = nullptr);

  protected:
    virtual ~ScalarAttribute();

  private:
  };

  /**
  A pointer attribute.
  */
  class AGXCORE_EXPORT PointerAttribute : public Attribute
  {
  public:
    static PointerAttribute *load(agx::TiXmlElement *eAttribute);
    static agx::Model *ClassModel();

    PointerAttribute(const agx::Name& name, Type *type, Format *defaultFormat = nullptr, agxData::Value *defaultValue = nullptr);

  protected:
    virtual ~PointerAttribute();

  private:
  };

  /**
  A shared attribute.
  */
  class AGXCORE_EXPORT SharedAttribute : public Attribute
  {
  public:
    static SharedAttribute *load(agx::TiXmlElement *eAttribute);
    static agx::Model *ClassModel();

  public:
    SharedAttribute(const agx::Name& name, Format *format, const agx::String& expression);

    // agxData::Value *getValue() { return m_value; }
    const agx::String& getExpression() const { return m_expression; }

  protected:
    SharedAttribute() {}
  private:
    // agxData::ValueRef m_value;
    agx::String m_expression;
  };



  /**
  A container attribute (Array/Vector/HashSet)
  */
  class AGXCORE_EXPORT ContainerAttribute : public Attribute
  {
  public:
    static agx::Model *ClassModel();

    ContainerAttribute(const agx::Name& name, ScalarAttribute *elementAttribute, Type *containerType);
    ScalarAttribute *getElementAttribute();
    const ScalarAttribute *getElementAttribute() const;
    virtual ~ContainerAttribute();

  protected:
    ContainerAttribute() {}
  private:
    ScalarAttributeRef m_elementAttribute;
  };


  /**
  An array attribute.
  */
  class AGXCORE_EXPORT ArrayAttribute : public ContainerAttribute
  {
  public:
    static ArrayAttribute *load(agx::TiXmlElement *eAttribute);
    static agx::Model *ClassModel();


  public:
    ArrayAttribute(const agx::Name& name, ScalarAttribute *elementAttribute, Type *arrayType);
    virtual ~ArrayAttribute();

  protected:
    ArrayAttribute() {}
  private:
  };


  /**
  A vector attribute.
  */
  class AGXCORE_EXPORT VectorAttribute : public ContainerAttribute
  {
  public:
    static VectorAttribute *load(agx::TiXmlElement *eAttribute);
    static agx::Model *ClassModel();


  public:
    VectorAttribute(const agx::Name& name, ScalarAttribute *elementAttribute, Type *vectorType);
    virtual ~VectorAttribute();

  protected:
    VectorAttribute() {}
  private:
  };


  /**
  A hash set attribute.
  */
  class AGXCORE_EXPORT HashSetAttribute : public ContainerAttribute
  {
  public:
    static HashSetAttribute *load(agx::TiXmlElement *eAttribute);
    static agx::Model *ClassModel();


  public:
    HashSetAttribute(const agx::Name& name, ScalarAttribute *elementAttribute, Type* hashSetType);
    virtual ~HashSetAttribute();

  protected:
    HashSetAttribute() {}
  private:
  };



  //--- TEMPLATED ----

  template <typename T>
  class ScalarAttributeT : public ScalarAttribute
  {
  public:
    typedef T Type;

    ScalarAttributeT(const agx::Name& name) : ScalarAttribute(name, agxData::getType<T>(), agxData::getFormat<T>()) {}
    //ScalarAttributeT() {}
    virtual ~ScalarAttributeT() {}
  private:
    // T m_defaultValue;
  };

  template <typename T>
  class PointerAttributeT : public PointerAttribute
  {
  public:
    typedef T Type;

  private:
    PointerAttributeT() {}
    virtual ~PointerAttributeT() {}
    // T m_defaultValue;
  };

  template <typename T>
  class ArrayAttributeT : public ArrayAttribute
  {
  public:
    typedef Array<T> Type;
    typedef T ElementType;

  private:
    ArrayAttributeT() {}
    virtual ~ArrayAttributeT() {}
    // Array<T> m_defaultValue;
    // ScalarAttributeT<T> m_elementAttribute;
  };

  template <typename T>
  class VectorAttributeT : public VectorAttribute
  {
  public:
    typedef agx::Vector<T> Type;
    typedef T ElementType;

  private:
    VectorAttributeT() {}
    virtual ~VectorAttributeT() {}
    // Type m_defaultValue;
    ScalarAttributeT<T> m_elementAttribute;
  };

  template <typename T>
  class HashSetAttributeT : public HashSetAttribute
  {
  public:
    typedef agx::HashSet<T> Type; // NOTE agx::HashSet instead of agxData::HashSet due to pointer invalidation (element buffer) in recursive HashSet::rebuild calls
    typedef T ElementType;

  private:
    HashSetAttributeT() {}
    virtual ~HashSetAttributeT() {}
    // Type m_defaultValue;
    ScalarAttributeT<T> m_elementAttribute;
  };



  template <typename T>
  class SharedAttributeT : public SharedAttribute
  {
  public:
    typedef T Type;

  private:
    SharedAttributeT() {}
    virtual ~SharedAttributeT() {}
    // T m_defaultValue;
  };




    /* Implementation */

  //-- ATTRIBUTE ----------------------
  AGX_FORCE_INLINE agx::Index Attribute::getId() const { return m_id; }
  AGX_FORCE_INLINE Type *Attribute::getType() { return m_type; }
  AGX_FORCE_INLINE const Type *Attribute::getType() const { return m_type; }
  AGX_FORCE_INLINE Format *Attribute::getDefaultFormat() { return m_defaultFormat; }
  AGX_FORCE_INLINE const Format *Attribute::getDefaultFormat() const { return m_defaultFormat; }
  AGX_FORCE_INLINE agxData::Value *Attribute::getNumElements() { return m_numElements; }
  AGX_FORCE_INLINE const agxData::Value *Attribute::getNumElements() const { return m_numElements; }
  AGX_FORCE_INLINE agxData::Value *Attribute::getDefaultValue() { return m_defaultValue; }
  AGX_FORCE_INLINE const agxData::Value *Attribute::getDefaultValue() const { return m_defaultValue; }
  AGX_FORCE_INLINE EntityModel *Attribute::getEntity() const { return m_entity; }



  //-- CONTAINER ATTRIBUTE --------------------------
  AGX_FORCE_INLINE const ScalarAttribute *ContainerAttribute::getElementAttribute() const { return m_elementAttribute; }
  AGX_FORCE_INLINE ScalarAttribute *ContainerAttribute::getElementAttribute() { return m_elementAttribute; }

}

#ifdef _MSC_VER
# pragma warning(pop)
#endif
#endif /* _AGXDATA_ATTRIBUTE_H_ */
