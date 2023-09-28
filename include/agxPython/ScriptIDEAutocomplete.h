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
#ifndef AGXPYTHON_SCRIPTIDEAUTOCOMPLETE_H
#define AGXPYTHON_SCRIPTIDEAUTOCOMPLETE_H 1

#include <agx/config/AGX_USE_PYTHON.h>

#if AGX_USE_PYTHON()

#include <agxPython/export.h>

#include <agx/Referenced.h>

#include <agxPython/ScriptIDEData.h>

#include <thread>
#include <mutex>
#include <vector>
#include <agx/HashTable.h>

namespace agxPython
{

  class ScriptIDE;
  AGX_DECLARE_POINTER_TYPES(ScriptIDEAutocompleteCollection);
  class AGXPYTHON_EXPORT ScriptIDEAutocompleteCollection : public agxPython::ScriptIDEDataCollection
  {
  public:

    friend class ScriptIDEAutocomplete;

    enum ItemAttribute
    {
      Category = 0,
      Type,
      Name,
      FullName,
      ModuleName,
      IsEnabled,
      Docstring,
      Completion,
      NumberOfAttributes,
    };

    class AGXPYTHON_EXPORT Item : public agxPython::ScriptIDEData
    {

    public:

      Item();

      // Copy constructor
      Item(const Item& copy);

      Item(void* jediObject, ScriptIDEAutocompleteCollection *collection);

      virtual agx::String str() const override;

      virtual agx::String type() const override {
        return "AutocompleteItem";
      }

      virtual bool isGood() const override;

    protected:

      virtual ~Item();

    private:

      bool m_isGood;

    };
    typedef agx::ref_ptr<Item> ItemRef;
    typedef agx::Vector< ItemRef > ItemRefVector;

    ScriptIDEAutocompleteCollection();

    bool add(ScriptIDEDataRef item) override;

    ItemRefVector getItems();

  protected:

    void addToDocstringHash(const agx::String& key, const agx::String& value);
    agx::String getFromDocstringHash(const agx::String& key) const;


    virtual ~ScriptIDEAutocompleteCollection();

    typedef agx::HashTable<agx::String, agx::String> StringHashTable;
    StringHashTable m_cached_docstring;



  private:

    std::mutex m_lock;

  };

  AGX_DECLARE_POINTER_TYPES(ScriptIDEAutocomplete);
  class AGXPYTHON_EXPORT ScriptIDEAutocomplete : public agx::Referenced
  {

  public:

    friend class ScriptIDE;
    class JediThread;

    ScriptIDEAutocomplete();

    virtual bool onUpdate(const agx::String& sourceCode) = 0;

    ScriptIDE* getIDE();

    ScriptIDEAutocompleteCollection* getCollection();

    static void preloadMomentumModules();

    // call this to stop the JediThread. After a call to this method no more auto completion queries could be made
    void stop();
    
  protected:

    virtual ~ScriptIDEAutocomplete();

  private:

    void refresh(bool readOnly, bool scopeChange);

    JediThread* m_worker;

    ScriptIDE* m_ide;

    ScriptIDEAutocompleteCollectionRef m_suggested;
  };

}

#endif

#endif
