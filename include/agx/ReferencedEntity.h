/*
Copyright 2007-2023. Algoryx Simulation AB.

All AGX source code, intellectual property, documentation, sample code,
tutorials, scene files and technical white papers, are copyrighted, proprietary
and confidential material of Algoryx Simulation AB. You may not download, read,
store, distribute, publish, copy or otherwise disseminate, use or expose this
material unless having a written signed agreement with Algoryx Simulation AB, or
having been advised so by Algoryx Simulation AB for a time limited evaluation,
or having purchased a valid commercial license from Algoryx Simulation AB.

Algoryx Simulation AB disclaims all responsibilities for loss or damage caused
from using this software, unless otherwise stated in written agreements with
Algoryx Simulation AB.
*/

//////////////////////////////////////////////////
// AUTOMATICALLY GENERATED ENTITY, DO NOT EDIT! //
//////////////////////////////////////////////////

#ifndef GENERATED_AGX_REFERENCED_H_PLUGIN
#define GENERATED_AGX_REFERENCED_H_PLUGIN

#define AGX_ENTITY_WRAPPER 1


#ifdef _MSC_VER
# pragma warning(push)
// warning C4505: 'agxData::VectorAttributeT<T>::print' : unreferenced local function has been removed
# pragma warning( disable : 4505 )
//  warning C4251:  'X' : class 'Y' needs to have dll-interface to be used by clients of class 'Z'
# pragma warning( disable : 4251 )
//  warning C4355: 'this' : used in base member initializer list
# pragma warning( disable : 4355 )
//  marked as __forceinline not inlined
# pragma warning( disable: 4714 )
#endif

#include <agxData/EntityModel.h>
#include <agxData/EntityStorage.h>
#include <agxData/EntityRef.h>
#include <agxData/EntityPtr.h>
#include <agxData/EntityInstance.h>
#include <agx/Integer.h>
#include <agx/Real.h>
#include <agx/macros.h>
#include <agx/Integer.h>
#include <agx/SpinMutex.h>


namespace agx
{

  class ReferencedModel;
  class ReferencedData;
  class ReferencedPtr;
  class ReferencedInstance;
  class ReferencedSemantics;


  AGX_DECLARE_POINTER_TYPES(ReferencedModel);

  /** 
  Abstract description of the data attributes for the Referenced entity.
  */ 
  class AGXCORE_EXPORT ReferencedModel : public agxData::EntityModel
  {
  public:
    typedef ReferencedPtr PtrT;

    ReferencedModel(const agx::String& name = "Referenced");

    /// \return The entity model singleton.
    static ReferencedModel* instance();

    /// Create and return a pointer to a new instance in the default storage for this entity model.
    static ReferencedPtr createInstance();

    /// \return The default storage for this entity model.
    static agxData::EntityStorage* defaultStorage();

    /// This is part of internal cleanup and should not be called by users
    virtual void shutdownCleanup() override;



    /* Attributes */
    static agxData::ScalarAttributeT< agx::UInt32 >* referenceCountAttribute;
    static agxData::ScalarAttributeT< agx::SpinMutex >* observerMutexAttribute;
    static agxData::ScalarAttributeT< agx::Vector< agxData::EntityPtr * > >* observersAttribute;

  protected:
    virtual ~ReferencedModel();
    virtual agxData::EntityData* createData(agxData::EntityStorage* storage) override;
    virtual void configure(agx::TiXmlElement* eEntity) override;
    virtual void initAttributeAccessors() override;
    virtual void construct(agxData::EntityPtr instance) override;
    void construct(agx::ReferencedPtr referenced);
  };


  DOXYGEN_START_INTERNAL_BLOCK()
  #ifndef AGX_REFERENCED_DATA_SET_OVERRIDE
  #define AGX_REFERENCED_DATA_SET
  class AGXCORE_EXPORT ReferencedData : public agxData::EntityData
  {
  public:
    ReferencedInstance operator[] (size_t index);

  public:
    agxData::Array< ReferencedPtr >& instance;
    agxData::Array< agx::UInt32 > referenceCount;
    agxData::Array< agx::SpinMutex > observerMutex;
    agxData::Array< agx::Vector< agxData::EntityPtr * > > observers;

  public:
    typedef agx::UInt32 referenceCountType;
    typedef agx::SpinMutex observerMutexType;
    typedef agx::Vector< agxData::EntityPtr * > observersType;

  public:
    ReferencedData(agxData::EntityStorage* storage);
    ReferencedData();

  protected:
    virtual ~ReferencedData() {}
    virtual void setNumElements(agx::Index numElements) override;

  private:
    ReferencedData& operator= (const ReferencedData&) { return *this; }

  };
  #endif
  DOXYGEN_END_INTERNAL_BLOCK()


  DOXYGEN_START_INTERNAL_BLOCK()
  class AGXCORE_EXPORT ReferencedSemantics : protected agxData::EntityPtr
  {
  public:

    // Automatic getters
    agx::UInt32 const& getReferenceCount() const;
    agx::SpinMutex const& getObserverMutex() const;
    agx::Vector< agxData::EntityPtr * > const& getObservers() const;

    // Semantics defined by explicit kernels

    // Automatic setters
    void setReferenceCount(agx::UInt32 const& value);
    void setObserverMutex(agx::SpinMutex const& value);
    void setObservers(agx::Vector< agxData::EntityPtr * > const& value);


  protected:
    friend class ReferencedPtr;
    friend class ReferencedInstance;
    ReferencedSemantics();
  };
  DOXYGEN_END_INTERNAL_BLOCK()


  /**
  Pointer to a entity instance of type Referenced
  */
  class CALLABLE ReferencedPtr : public agxData::EntityPtr
  {
  public:
    typedef ReferencedModel ModelType;
    typedef ReferencedData DataType;
    typedef ReferencedInstance InstanceType;

  public:
    AGXCORE_EXPORT ReferencedPtr();
    AGXCORE_EXPORT ReferencedPtr(agxData::EntityStorage* storage, agx::Index id);
    AGXCORE_EXPORT ReferencedPtr(const agxData::EntityPtr& ptr);
    AGXCORE_EXPORT ReferencedPtr(const agxData::EntityInstance& instance);
    AGXCORE_EXPORT ReferencedPtr& operator= (const agxData::EntityPtr& ptr);
    AGXCORE_EXPORT ReferencedPtr& operator= (const agxData::EntityInstance& instance);
    AGXCORE_EXPORT ReferencedInstance instance();
    AGXCORE_EXPORT const ReferencedInstance instance() const;

    AGXCORE_EXPORT ReferencedSemantics* operator->();
    AGXCORE_EXPORT const ReferencedSemantics* operator->() const;

    ReferencedData* getData();
    const ReferencedData* getData() const;


    /// \return reference to the referenceCount attribute
    AGXCORE_EXPORT agx::UInt32& referenceCount();
    /// \return const reference to the referenceCount attribute
    AGXCORE_EXPORT agx::UInt32 const& referenceCount() const;

    /// \return reference to the observerMutex attribute
    AGXCORE_EXPORT agx::SpinMutex& observerMutex();
    /// \return const reference to the observerMutex attribute
    AGXCORE_EXPORT agx::SpinMutex const& observerMutex() const;

    /// \return reference to the observers attribute
    AGXCORE_EXPORT agx::Vector< agxData::EntityPtr * >& observers();
    /// \return const reference to the observers attribute
    AGXCORE_EXPORT agx::Vector< agxData::EntityPtr * > const& observers() const;

    DOXYGEN_START_INTERNAL_BLOCK()
    // Only used by tolua!
    #ifdef AGX_GENERATED_TOLUA_WRAPPER
    inline agx::UInt32 const& getReferenceCount() const { return (*this)->getReferenceCount(); }
    inline agx::SpinMutex const& getObserverMutex() const { return (*this)->getObserverMutex(); }
    inline agx::Vector< agxData::EntityPtr * > const& getObservers() const { return (*this)->getObservers(); }


    inline void setReferenceCount(agx::UInt32 const& value) { (*this)->setReferenceCount(value); }
    inline void setObserverMutex(agx::SpinMutex const& value) { (*this)->setObserverMutex(value); }
    inline void setObservers(agx::Vector< agxData::EntityPtr * > const& value) { (*this)->setObservers(value); }
    #endif
    DOXYGEN_END_INTERNAL_BLOCK()

  };

  // Entity is Referenced
  typedef agxData::EntityRef< ReferencedPtr > ReferencedRef;


  DOXYGEN_START_INTERNAL_BLOCK()
  class AGXCORE_EXPORT ReferencedInstance : public agxData::EntityInstance
  {
  public:
    ReferencedInstance();
    ReferencedInstance(ReferencedData* data, agx::Index index);
    ReferencedInstance(agxData::EntityStorage *storage, agx::Index index);
    ReferencedInstance(const agxData::EntityInstance& other);
    ReferencedInstance(const agxData::EntityPtr& ptr);

    ReferencedData* getData();
    const ReferencedData* getData() const;

  public:
    /// \return reference to the referenceCount attribute
    agx::UInt32& referenceCount();
    /// \return const reference to the referenceCount attribute
    agx::UInt32 const& referenceCount() const;

    /// \return reference to the observerMutex attribute
    agx::SpinMutex& observerMutex();
    /// \return const reference to the observerMutex attribute
    agx::SpinMutex const& observerMutex() const;

    /// \return reference to the observers attribute
    agx::Vector< agxData::EntityPtr * >& observers();
    /// \return const reference to the observers attribute
    agx::Vector< agxData::EntityPtr * > const& observers() const;

  };
  DOXYGEN_END_INTERNAL_BLOCK()



  typedef agx::VectorPOD<ReferencedPtr> ReferencedPtrVector;
  typedef agxData::Array<ReferencedPtr> ReferencedPtrArray;



  DOXYGEN_START_INTERNAL_BLOCK()
  /* Implementation */
  //-----------------------------------------------------------------------------------------------------
  //-----------------------------------------------------------------------------------------------------
  inline ReferencedInstance agx::ReferencedData::operator[] (size_t index) { return ReferencedInstance(this, (agx::Index)index); }
  //-----------------------------------------------------------------------------------------------------
  AGX_FORCE_INLINE ReferencedPtr::ReferencedPtr() {}
  AGX_FORCE_INLINE ReferencedPtr::ReferencedPtr(agxData::EntityStorage* storage, agx::Index id) : agxData::EntityPtr(storage, id) {}
  AGX_FORCE_INLINE ReferencedPtr::ReferencedPtr(const agxData::EntityPtr& ptr) : agxData::EntityPtr(ptr)
  {
    agxAssertN(!ptr || ptr.isInstanceOf(ReferencedModel::instance()),
      "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
      EntityPtr::getModel()->fullPath().c_str(), ReferencedModel::instance()->fullPath().c_str());
  }

  AGX_FORCE_INLINE ReferencedPtr::ReferencedPtr(const agxData::EntityInstance& instance) : agxData::EntityPtr(instance)
  {
    agxAssertN(!instance || instance.isInstanceOf(ReferencedModel::instance()),
      "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
      EntityPtr::getModel()->fullPath().c_str(), ReferencedModel::instance()->fullPath().c_str());
  }

  AGX_FORCE_INLINE ReferencedPtr& ReferencedPtr::operator= (const agxData::EntityPtr& ptr)
  {
    agxData::EntityPtr::operator= (ptr);
    agxAssertN(!ptr || ptr.isInstanceOf(ReferencedModel::instance()),
      "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
      EntityPtr::getModel()->fullPath().c_str(), ReferencedModel::instance()->fullPath().c_str());
    return *this;
  }

  AGX_FORCE_INLINE ReferencedPtr& ReferencedPtr::operator= (const agxData::EntityInstance& instance)
  {
    agxData::EntityPtr::operator= (instance);
    agxAssertN(!instance || instance.isInstanceOf(ReferencedModel::instance()),
      "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
      EntityPtr::getModel()->fullPath().c_str(), ReferencedModel::instance()->fullPath().c_str());
    return *this;
  }

  inline ReferencedInstance ReferencedPtr::instance() { return agxData::EntityPtr::instance(); }
  inline const ReferencedInstance ReferencedPtr::instance() const { return agxData::EntityPtr::instance(); }
  AGX_FORCE_INLINE ReferencedSemantics* ReferencedPtr::operator->() { return (ReferencedSemantics* )this; }
  AGX_FORCE_INLINE const ReferencedSemantics* ReferencedPtr::operator->() const { return (const ReferencedSemantics* )this; }
  AGX_FORCE_INLINE ReferencedData* ReferencedPtr::getData() { return static_cast<ReferencedData* >(agxData::EntityPtr::getData()); }
  AGX_FORCE_INLINE const ReferencedData* ReferencedPtr::getData() const { return static_cast<const ReferencedData* >(agxData::EntityPtr::getData()); }

  AGX_FORCE_INLINE agx::UInt32& ReferencedPtr::referenceCount() { verifyIndex(); return getData()->referenceCount[calculateIndex()]; }
  AGX_FORCE_INLINE agx::UInt32 const& ReferencedPtr::referenceCount() const { verifyIndex(); return getData()->referenceCount[calculateIndex()]; }

  AGX_FORCE_INLINE agx::SpinMutex& ReferencedPtr::observerMutex() { verifyIndex(); return getData()->observerMutex[calculateIndex()]; }
  AGX_FORCE_INLINE agx::SpinMutex const& ReferencedPtr::observerMutex() const { verifyIndex(); return getData()->observerMutex[calculateIndex()]; }

  AGX_FORCE_INLINE agx::Vector< agxData::EntityPtr * >& ReferencedPtr::observers() { verifyIndex(); return getData()->observers[calculateIndex()]; }
  AGX_FORCE_INLINE agx::Vector< agxData::EntityPtr * > const& ReferencedPtr::observers() const { verifyIndex(); return getData()->observers[calculateIndex()]; }

  //-----------------------------------------------------------------------------------------------------
  AGX_FORCE_INLINE ReferencedInstance::ReferencedInstance() {}
  AGX_FORCE_INLINE ReferencedInstance::ReferencedInstance(ReferencedData* data, agx::Index index) : agxData::EntityInstance(data, index) {}
  AGX_FORCE_INLINE ReferencedInstance::ReferencedInstance(agxData::EntityStorage* storage, agx::Index index) : agxData::EntityInstance(storage, index) {}
  AGX_FORCE_INLINE ReferencedInstance::ReferencedInstance(const agxData::EntityInstance& other) : agxData::EntityInstance(other)
  {
    agxAssertN(!other || other.isInstanceOf(ReferencedModel::instance()),
      "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
      EntityInstance::getModel()->fullPath().c_str(), ReferencedModel::instance()->fullPath().c_str());
  }

  AGX_FORCE_INLINE ReferencedInstance::ReferencedInstance(const agxData::EntityPtr& ptr) : agxData::EntityInstance(ptr)
  {
    agxAssertN(!ptr || ptr.isInstanceOf(ReferencedModel::instance()),
      "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
      EntityInstance::getModel()->fullPath().c_str(), ReferencedModel::instance()->fullPath().c_str());
  }


  AGX_FORCE_INLINE ReferencedData* ReferencedInstance::getData() { return static_cast<ReferencedData* >(agxData::EntityInstance::getData()); }
  AGX_FORCE_INLINE const ReferencedData* ReferencedInstance::getData() const { return static_cast<const ReferencedData* >(agxData::EntityInstance::getData()); }

  AGX_FORCE_INLINE agx::UInt32& ReferencedInstance::referenceCount() { verifyIndex(); return getData()->referenceCount[getIndex()]; }
  AGX_FORCE_INLINE agx::UInt32 const& ReferencedInstance::referenceCount() const { verifyIndex(); return getData()->referenceCount[getIndex()]; }

  AGX_FORCE_INLINE agx::SpinMutex& ReferencedInstance::observerMutex() { verifyIndex(); return getData()->observerMutex[getIndex()]; }
  AGX_FORCE_INLINE agx::SpinMutex const& ReferencedInstance::observerMutex() const { verifyIndex(); return getData()->observerMutex[getIndex()]; }

  AGX_FORCE_INLINE agx::Vector< agxData::EntityPtr * >& ReferencedInstance::observers() { verifyIndex(); return getData()->observers[getIndex()]; }
  AGX_FORCE_INLINE agx::Vector< agxData::EntityPtr * > const& ReferencedInstance::observers() const { verifyIndex(); return getData()->observers[getIndex()]; }

  //-----------------------------------------------------------------------------------------------------
  AGX_FORCE_INLINE ReferencedSemantics::ReferencedSemantics() {}
  //-----------------------------------------------------------------------------------------------------
  DOXYGEN_END_INTERNAL_BLOCK()
}

AGX_TYPE_BINDING(agx::ReferencedPtr, "ReferencedPtr")
AGX_TYPE_BINDING(agx::ReferencedInstance, "ReferencedInstance")

#ifdef _MSC_VER
# pragma warning(pop)
#endif

#undef AGX_ENTITY_WRAPPER
#undef AGX_ENTITY_NAMESPACE
#endif

