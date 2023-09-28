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

#include <agxStream/agxStream.h>
#include <agx/DynamicLibrary.h>
#include <agx/Singleton.h>

#include <string>
#include <agx/HashTable.h>
#include <agx/Referenced.h>

DOXYGEN_START_INTERNAL_BLOCK()


namespace agxStream
{
  class Serializable;
  class InputArchive;

  /// This class is the base class for creating instances of Serializable classes.
  class AGXCORE_EXPORT StorageAgent
  {
  private:
    std::string m_className;

    // copy constructor and assignment operator prohibited
    StorageAgent(const StorageAgent&);
    StorageAgent& operator=(const StorageAgent&);


  protected:
    StorageAgent(const char* className);

  public:
    /// Destructor
    virtual ~StorageAgent();

    /// \return the name of the class that this instance can create.
    const char* getClassName() const;

    //virtual Serializable* createInstance() = 0;
    virtual Serializable* createInstance(InputArchive& in) = 0;

    /**
    \param className - Name of class for which a StorageAgent will be searched.
    \param archive - The input archive that is asking for the StorageAgent.
    \return A pointer to the StorageAgent capable of creating the class of name \p className
    */
    static StorageAgent* lookup( const char* className, const agxStream::InputArchive* archive );

    static std::string stripClassName(const char* functionName, const char* className);

  };

  template< class T >
  class DefStorageAgent : public StorageAgent
  {
  public:
    DefStorageAgent() : StorageAgent(T::getConstructClassId()) { }
    virtual ~DefStorageAgent() { }

    virtual Serializable* createInstance(InputArchive& in) override {
      return T::create(in);
    }
  };

  AGX_FORCE_INLINE const char* StorageAgent::getClassName() const
  {
    return m_className.c_str();
  }


  AGX_DECLARE_POINTER_TYPES(StorageManager);
  class AGXCORE_EXPORT StorageManager
  {
  public:

    StorageManager();
    ~StorageManager();

    static StorageManager* instance();
    void cleanup();

    void shutdown() { cleanup(); }

    struct ProxyBase {};

    template<class T>
    class RegisterStorageProxy : public ProxyBase
    {
    public:

      RegisterStorageProxy() {
        if (StorageManager::instance()) {
          m_registerStorageProxy = new T;
          StorageManager::instance()->add(m_registerStorageProxy);

        }
      }

      virtual ~RegisterStorageProxy() {
        delete m_registerStorageProxy;
      }

      T* getStorageAgent() const {
        return m_registerStorageProxy;
      }

    protected:
      T* m_registerStorageProxy;
    };

    class AGXCORE_EXPORT ModificationCallback : public agx::Referenced
    {
    public:
      ModificationCallback(const agx::StringVector& modifications);
      ModificationCallback();

      bool operator() (const agxStream::InputArchive* archive) const;

    private:
      ~ModificationCallback();

      agx::StringVector m_modifications;
    };



    /**
    This class will register two StorageAgents to the storage manager. One default and one deprecated class.
    In addition it will store a callback that determines which one that should be used. If the callback returns true, the default is used.
    */
    template<class T, class Deprecated_T>
    class DeprecatedRegisterStorageProxy : public ProxyBase
    {
    public:


      DeprecatedRegisterStorageProxy(const agx::StringVector& modificationList) {
        if (StorageManager::instance()) {

          // Add the StorageAgent to the list of storage agents for the StorageManager
          m_registerStorageProxy = new T;
          m_deprecatedRegisterStorageProxy = new Deprecated_T;
          StorageManager::instance()->add(m_registerStorageProxy, modificationList, m_deprecatedRegisterStorageProxy);
        }
      }

      virtual ~DeprecatedRegisterStorageProxy() {
        delete m_registerStorageProxy;

        if (m_deprecatedRegisterStorageProxy)
          delete m_deprecatedRegisterStorageProxy;
      }


    protected:
      T* m_registerStorageProxy;
      Deprecated_T* m_deprecatedRegisterStorageProxy;
    };


    /// Add a StorageAgent for the given className
    void add(const char *className, StorageAgent* agent);

    /// Add a StorageAgent
    void add(StorageAgent* agent);

    /// Add a StorageAgent for a default class, and a deprecated class. If the list of modifications are all present in InputArchive, the default class will be used
    void add(StorageAgent* defaultAgent, const agx::StringVector& modificationList, StorageAgent* deprecatedAgent);

    /// Remove a storage agent
    void remove(StorageAgent* agent);

    /// \return a storage agent for the given className. The archive might be involved in the test if it is a deprecated class involved.
    StorageAgent* find(const char* className, const agxStream::InputArchive *archive = nullptr);

  protected:

    typedef agx::ref_ptr<agx::DynamicLibrary> DynamicLibraryRef;


    // This class stores a StorageAgent for a class
    // In addition it can store another StorageAgent which is a deprecated fallback that will be used IF the callback returns true.
    class AgentStorage : public agx::Referenced {
    public:
      AgentStorage(StorageAgent* defaultAgent, const agx::StringVector& modificationList, StorageAgent* deprecatedAgent);

      StorageAgent* getStorageAgent(const agxStream::InputArchive* archive);

    private:
      agx::ref_ptr<ModificationCallback> m_callback;
      StorageAgent* m_defaultAgent;
      StorageAgent* m_deprecatedAgent;

    private:
      ~AgentStorage();
    };

    typedef agx::HashTable< std::string, agx::ref_ptr<AgentStorage> > StorageAgentMap;
    StorageAgentMap m_storageAgents;

  };


  /**
  This structure should be used to instantiate a StorageAgent for classes that has a deprecated representation.
  For example, you have a class TorqueConverter that you now have reimplemented.
  To be able to support restoring old serializations with the old torque converter, you can use this class to
  specify the New class, Old class and a callback that indicates which implementation that should be used.
  Example: DeprecatedStorage<MyClass, DeprecatedClass> MyClassStorage;
  */
  template <typename DefaultClass_T, typename DeprecatedClass_T>
  struct DeprecatedStorage : public StorageManager::DeprecatedRegisterStorageProxy< DefStorageAgent<DefaultClass_T>, DefStorageAgent<DeprecatedClass_T> >
  {
    DeprecatedStorage(const agx::StringVector& modificationList) : StorageManager::DeprecatedRegisterStorageProxy< DefStorageAgent<DefaultClass_T>, DefStorageAgent<DeprecatedClass_T> >(modificationList)
    {
    }

    ~DeprecatedStorage() {}
  };


  /**
  This structure should be used to instantiate a StorageAgent for all serializable classes.
  Example: Storage<MyClass> MyClassStorage;
  */
  template <typename T>
  struct Storage : public StorageManager::RegisterStorageProxy< DefStorageAgent<T> >
  {
    Storage()
    {
    }

    ~Storage()
    {
    }
  };


} // namespace

DOXYGEN_END_INTERNAL_BLOCK()

