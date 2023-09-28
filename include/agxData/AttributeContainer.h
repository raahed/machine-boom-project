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

#ifndef AGXDATA_ATTRIBUTECONTAINER_H
#define AGXDATA_ATTRIBUTECONTAINER_H

#include <agx/agx.h>
#include <agxData/Attribute.h>
#include <agx/agxCore_export.h>
#include <agx/Allocator.h>
#include <agx/HashTable.h>

namespace agxData
{
  /**
  An attribute container.
  */
  class AGXCORE_EXPORT AttributeContainer
  {
  public:
    AttributeContainer();
    ~AttributeContainer();

    /**
    \return The value corresponding to an attribute.
    */
    void *getAttribute(const Attribute *attribute);
    const void *getAttribute(const Attribute *attribute) const;


    // Templated accessors
    template <typename T>
    T& getAttribute(const ScalarAttributeT<T> *attribute);

    template <typename T>
    const T& getAttribute(const ScalarAttributeT<T> *attribute) const;


    /**
    \return True if an attribute is active/allocated.
    */
    bool hasAttribute(const Attribute *attribute) const;

  private:
    typedef agx::HashTable<AttributeConstRef, void *> ScalarAttributeTable;
    mutable ScalarAttributeTable m_attributeTable;
  };


  /* Implementation */
  template <typename T>
  T& AttributeContainer::getAttribute(const ScalarAttributeT<T> *attribute)
  {
    return *static_cast<T *>(this->getAttribute(static_cast<const Attribute *>(attribute)));
  }

  template <typename T>
  const T& AttributeContainer::getAttribute(const ScalarAttributeT<T> *attribute) const
  {
    return const_cast<AttributeContainer *>(this)->getAttribute(attribute);
  }

}


#endif /* _AGXDATA_ATTRIBUTECONTAINER_H_ */
