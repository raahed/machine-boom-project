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
#ifndef AGXPYTHON_SCRIPTIDEDATA_H
#define AGXPYTHON_SCRIPTIDEDATA_H 1

#include <agx/config/AGX_USE_PYTHON.h>

#if AGX_USE_PYTHON()

#include <agxPython/export.h>

#include <agx/Referenced.h>
#include <agx/Name.h>

#include <map>
#include <vector>
#include <string>
#include <iostream>

namespace agxPython
{

  AGX_DECLARE_POINTER_TYPES(ScriptIDEData);
  class AGXPYTHON_EXPORT ScriptIDEData : public agx::Referenced
  {

  public:

    ScriptIDEData(size_t numberOfColumns);

    /// Copy constructor
    ScriptIDEData(const ScriptIDEData& copy);

#ifndef SWIG
    agx::String& getData(size_t i);
#endif

    agx::String operator[](size_t i) const;
    agx::String& operator[](size_t i);

    size_t size() const;
    agx::String getData(size_t i) const;
    agx::String returnData(size_t i) const;

    virtual bool matching(const agx::String& regex) const;

    virtual size_t hash() const;

    virtual agx::String str() const =0;

    virtual agx::String type() const = 0;

    virtual bool isGood() const = 0;

    struct compare
    {
      bool operator()(size_t lhs, size_t rhs) const
      {
        if (lhs == rhs)
          return false;
        return (lhs > rhs);
      }
    };

  protected:

    virtual ~ScriptIDEData();

    virtual bool matching_single(size_t i, const agx::String& regex) const;

  private:

    agx::Vector<agx::String> m_data;

  };



  AGX_DECLARE_POINTER_TYPES(ScriptIDEDataCollection);
  class AGXPYTHON_EXPORT ScriptIDEDataCollection : public agx::Referenced
  {

  public:

    typedef ScriptIDEDataRef elem_type;
    typedef std::map<size_t, elem_type, ScriptIDEData::compare> container_type;
    typedef container_type::iterator iterator_type;
    typedef std::vector<elem_type> range_type;

    const agx::Name& getCollectionTypeName() const;

    virtual bool add(ScriptIDEDataRef item);

    virtual void clear();

    virtual range_type equal_range(const agx::String& regexFilter = "");

    iterator_type begin() { return m_data.begin(); }
    iterator_type end() { return m_data.end(); }

  protected:
    ScriptIDEDataCollection(const agx::Name& collectionTypeName);

    virtual ~ScriptIDEDataCollection();

  private:

    agx::Name m_collectionTypeName;

    container_type m_data;

  };

}

#endif

#endif
