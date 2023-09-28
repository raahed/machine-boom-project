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

#ifndef GENERATED_AGX_TASKTIMING_H_PLUGIN
#define GENERATED_AGX_TASKTIMING_H_PLUGIN

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
#include <agx/Name.h>
#include <agx/Integer.h>
#include <agx/Real.h>


namespace agx
{

  class TaskTimingModel;
  class TaskTimingData;
  class TaskTimingPtr;
  class TaskTimingInstance;
  class TaskTimingSemantics;


  AGX_DECLARE_POINTER_TYPES(TaskTimingModel);

  /** 
  Abstract description of the data attributes for the TaskTiming entity.
  */ 
  class AGXPHYSICS_EXPORT TaskTimingModel : public agxData::EntityModel
  {
  public:
    typedef TaskTimingPtr PtrT;

    TaskTimingModel(const agx::String& name = "TaskTiming");

    /// \return The entity model singleton.
    static TaskTimingModel* instance();

    /// Create and return a pointer to a new instance in the default storage for this entity model.
    static TaskTimingPtr createInstance();

    /// \return The default storage for this entity model.
    static agxData::EntityStorage* defaultStorage();

    /// This is part of internal cleanup and should not be called by users
    virtual void shutdownCleanup() override;



    /* Attributes */
    static agxData::ScalarAttributeT< agx::Name >* nameAttribute;
    static agxData::ScalarAttributeT< agx::Name >* implementationAttribute;
    static agxData::ScalarAttributeT< agx::UInt >* numSubtasksAttribute;
    static agxData::ScalarAttributeT< agx::Real64 >* wallTimeAttribute;
    static agxData::ScalarAttributeT< agx::Real64 >* computeCostAttribute;
    static agxData::ScalarAttributeT< agx::Real64 >* overheadTimeAttribute;

  protected:
    virtual ~TaskTimingModel();
    virtual agxData::EntityData* createData(agxData::EntityStorage* storage) override;
    virtual void configure(agx::TiXmlElement* eEntity) override;
    virtual void initAttributeAccessors() override;
    virtual void construct(agxData::EntityPtr instance) override;
    void construct(agx::TaskTimingPtr taskTiming);
  };


  DOXYGEN_START_INTERNAL_BLOCK()
  #ifndef AGX_TASKTIMING_DATA_SET_OVERRIDE
  #define AGX_TASKTIMING_DATA_SET
  class AGXPHYSICS_EXPORT TaskTimingData : public agxData::EntityData
  {
  public:
    TaskTimingInstance operator[] (size_t index);

  public:
    agxData::Array< TaskTimingPtr >& instance;
    agxData::Array< agx::Name > name;
    agxData::Array< agx::Name > implementation;
    agxData::Array< agx::UInt > numSubtasks;
    agxData::Array< agx::Real64 > wallTime;
    agxData::Array< agx::Real64 > computeCost;
    agxData::Array< agx::Real64 > overheadTime;

  public:
    typedef agx::Name nameType;
    typedef agx::Name implementationType;
    typedef agx::UInt numSubtasksType;
    typedef agx::Real64 wallTimeType;
    typedef agx::Real64 computeCostType;
    typedef agx::Real64 overheadTimeType;

  public:
    TaskTimingData(agxData::EntityStorage* storage);
    TaskTimingData();

  protected:
    virtual ~TaskTimingData() {}
    virtual void setNumElements(agx::Index numElements) override;

  private:
    TaskTimingData& operator= (const TaskTimingData&) { return *this; }

  };
  #endif
  DOXYGEN_END_INTERNAL_BLOCK()


  DOXYGEN_START_INTERNAL_BLOCK()
  class AGXPHYSICS_EXPORT TaskTimingSemantics : protected agxData::EntityPtr
  {
  public:

    // Automatic getters
    agx::Name const& getName() const;
    agx::Name const& getImplementation() const;
    agx::UInt const& getNumSubtasks() const;
    agx::Real64 const& getWallTime() const;
    agx::Real64 const& getComputeCost() const;
    agx::Real64 const& getOverheadTime() const;

    // Semantics defined by explicit kernels

    // Automatic setters
    void setName(agx::Name const& value);
    void setImplementation(agx::Name const& value);
    void setNumSubtasks(agx::UInt const& value);
    void setWallTime(agx::Real64 const& value);
    void setComputeCost(agx::Real64 const& value);
    void setOverheadTime(agx::Real64 const& value);


  protected:
    friend class TaskTimingPtr;
    friend class TaskTimingInstance;
    TaskTimingSemantics();
  };
  DOXYGEN_END_INTERNAL_BLOCK()


  /**
  Pointer to a entity instance of type TaskTiming
  */
  class CALLABLE TaskTimingPtr : public agxData::EntityPtr
  {
  public:
    typedef TaskTimingModel ModelType;
    typedef TaskTimingData DataType;
    typedef TaskTimingInstance InstanceType;

  public:
    AGXPHYSICS_EXPORT TaskTimingPtr();
    AGXPHYSICS_EXPORT TaskTimingPtr(agxData::EntityStorage* storage, agx::Index id);
    AGXPHYSICS_EXPORT TaskTimingPtr(const agxData::EntityPtr& ptr);
    AGXPHYSICS_EXPORT TaskTimingPtr(const agxData::EntityInstance& instance);
    AGXPHYSICS_EXPORT TaskTimingPtr& operator= (const agxData::EntityPtr& ptr);
    AGXPHYSICS_EXPORT TaskTimingPtr& operator= (const agxData::EntityInstance& instance);
    AGXPHYSICS_EXPORT TaskTimingInstance instance();
    AGXPHYSICS_EXPORT const TaskTimingInstance instance() const;

    AGXPHYSICS_EXPORT TaskTimingSemantics* operator->();
    AGXPHYSICS_EXPORT const TaskTimingSemantics* operator->() const;

    TaskTimingData* getData();
    const TaskTimingData* getData() const;


    /// \return reference to the name attribute
    AGXPHYSICS_EXPORT agx::Name& name();
    /// \return const reference to the name attribute
    AGXPHYSICS_EXPORT agx::Name const& name() const;

    /// \return reference to the implementation attribute
    AGXPHYSICS_EXPORT agx::Name& implementation();
    /// \return const reference to the implementation attribute
    AGXPHYSICS_EXPORT agx::Name const& implementation() const;

    /// \return reference to the numSubtasks attribute
    AGXPHYSICS_EXPORT agx::UInt& numSubtasks();
    /// \return const reference to the numSubtasks attribute
    AGXPHYSICS_EXPORT agx::UInt const& numSubtasks() const;

    /// \return reference to the wallTime attribute
    AGXPHYSICS_EXPORT agx::Real64& wallTime();
    /// \return const reference to the wallTime attribute
    AGXPHYSICS_EXPORT agx::Real64 const& wallTime() const;

    /// \return reference to the computeCost attribute
    AGXPHYSICS_EXPORT agx::Real64& computeCost();
    /// \return const reference to the computeCost attribute
    AGXPHYSICS_EXPORT agx::Real64 const& computeCost() const;

    /// \return reference to the overheadTime attribute
    AGXPHYSICS_EXPORT agx::Real64& overheadTime();
    /// \return const reference to the overheadTime attribute
    AGXPHYSICS_EXPORT agx::Real64 const& overheadTime() const;

    DOXYGEN_START_INTERNAL_BLOCK()
    // Only used by tolua!
    #ifdef AGX_GENERATED_TOLUA_WRAPPER
    inline agx::Name const& getName() const { return (*this)->getName(); }
    inline agx::Name const& getImplementation() const { return (*this)->getImplementation(); }
    inline agx::UInt const& getNumSubtasks() const { return (*this)->getNumSubtasks(); }
    inline agx::Real64 const& getWallTime() const { return (*this)->getWallTime(); }
    inline agx::Real64 const& getComputeCost() const { return (*this)->getComputeCost(); }
    inline agx::Real64 const& getOverheadTime() const { return (*this)->getOverheadTime(); }


    inline void setName(agx::Name const& value) { (*this)->setName(value); }
    inline void setImplementation(agx::Name const& value) { (*this)->setImplementation(value); }
    inline void setNumSubtasks(agx::UInt const& value) { (*this)->setNumSubtasks(value); }
    inline void setWallTime(agx::Real64 const& value) { (*this)->setWallTime(value); }
    inline void setComputeCost(agx::Real64 const& value) { (*this)->setComputeCost(value); }
    inline void setOverheadTime(agx::Real64 const& value) { (*this)->setOverheadTime(value); }
    #endif
    DOXYGEN_END_INTERNAL_BLOCK()

  };


  DOXYGEN_START_INTERNAL_BLOCK()
  class AGXPHYSICS_EXPORT TaskTimingInstance : public agxData::EntityInstance
  {
  public:
    TaskTimingInstance();
    TaskTimingInstance(TaskTimingData* data, agx::Index index);
    TaskTimingInstance(agxData::EntityStorage *storage, agx::Index index);
    TaskTimingInstance(const agxData::EntityInstance& other);
    TaskTimingInstance(const agxData::EntityPtr& ptr);

    TaskTimingData* getData();
    const TaskTimingData* getData() const;

  public:
    /// \return reference to the name attribute
    agx::Name& name();
    /// \return const reference to the name attribute
    agx::Name const& name() const;

    /// \return reference to the implementation attribute
    agx::Name& implementation();
    /// \return const reference to the implementation attribute
    agx::Name const& implementation() const;

    /// \return reference to the numSubtasks attribute
    agx::UInt& numSubtasks();
    /// \return const reference to the numSubtasks attribute
    agx::UInt const& numSubtasks() const;

    /// \return reference to the wallTime attribute
    agx::Real64& wallTime();
    /// \return const reference to the wallTime attribute
    agx::Real64 const& wallTime() const;

    /// \return reference to the computeCost attribute
    agx::Real64& computeCost();
    /// \return const reference to the computeCost attribute
    agx::Real64 const& computeCost() const;

    /// \return reference to the overheadTime attribute
    agx::Real64& overheadTime();
    /// \return const reference to the overheadTime attribute
    agx::Real64 const& overheadTime() const;

  };
  DOXYGEN_END_INTERNAL_BLOCK()



  typedef agx::VectorPOD<TaskTimingPtr> TaskTimingPtrVector;
  typedef agxData::Array<TaskTimingPtr> TaskTimingPtrArray;



  DOXYGEN_START_INTERNAL_BLOCK()
  /* Implementation */
  //-----------------------------------------------------------------------------------------------------
  //-----------------------------------------------------------------------------------------------------
  inline TaskTimingInstance agx::TaskTimingData::operator[] (size_t index) { return TaskTimingInstance(this, (agx::Index)index); }
  //-----------------------------------------------------------------------------------------------------
  AGX_FORCE_INLINE TaskTimingPtr::TaskTimingPtr() {}
  AGX_FORCE_INLINE TaskTimingPtr::TaskTimingPtr(agxData::EntityStorage* storage, agx::Index id) : agxData::EntityPtr(storage, id) {}
  AGX_FORCE_INLINE TaskTimingPtr::TaskTimingPtr(const agxData::EntityPtr& ptr) : agxData::EntityPtr(ptr)
  {
    agxAssertN(!ptr || ptr.isInstanceOf(TaskTimingModel::instance()),
      "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
      EntityPtr::getModel()->fullPath().c_str(), TaskTimingModel::instance()->fullPath().c_str());
  }

  AGX_FORCE_INLINE TaskTimingPtr::TaskTimingPtr(const agxData::EntityInstance& instance) : agxData::EntityPtr(instance)
  {
    agxAssertN(!instance || instance.isInstanceOf(TaskTimingModel::instance()),
      "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
      EntityPtr::getModel()->fullPath().c_str(), TaskTimingModel::instance()->fullPath().c_str());
  }

  AGX_FORCE_INLINE TaskTimingPtr& TaskTimingPtr::operator= (const agxData::EntityPtr& ptr)
  {
    agxData::EntityPtr::operator= (ptr);
    agxAssertN(!ptr || ptr.isInstanceOf(TaskTimingModel::instance()),
      "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
      EntityPtr::getModel()->fullPath().c_str(), TaskTimingModel::instance()->fullPath().c_str());
    return *this;
  }

  AGX_FORCE_INLINE TaskTimingPtr& TaskTimingPtr::operator= (const agxData::EntityInstance& instance)
  {
    agxData::EntityPtr::operator= (instance);
    agxAssertN(!instance || instance.isInstanceOf(TaskTimingModel::instance()),
      "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
      EntityPtr::getModel()->fullPath().c_str(), TaskTimingModel::instance()->fullPath().c_str());
    return *this;
  }

  inline TaskTimingInstance TaskTimingPtr::instance() { return agxData::EntityPtr::instance(); }
  inline const TaskTimingInstance TaskTimingPtr::instance() const { return agxData::EntityPtr::instance(); }
  AGX_FORCE_INLINE TaskTimingSemantics* TaskTimingPtr::operator->() { return (TaskTimingSemantics* )this; }
  AGX_FORCE_INLINE const TaskTimingSemantics* TaskTimingPtr::operator->() const { return (const TaskTimingSemantics* )this; }
  AGX_FORCE_INLINE TaskTimingData* TaskTimingPtr::getData() { return static_cast<TaskTimingData* >(agxData::EntityPtr::getData()); }
  AGX_FORCE_INLINE const TaskTimingData* TaskTimingPtr::getData() const { return static_cast<const TaskTimingData* >(agxData::EntityPtr::getData()); }

  AGX_FORCE_INLINE agx::Name& TaskTimingPtr::name() { verifyIndex(); return getData()->name[calculateIndex()]; }
  AGX_FORCE_INLINE agx::Name const& TaskTimingPtr::name() const { verifyIndex(); return getData()->name[calculateIndex()]; }

  AGX_FORCE_INLINE agx::Name& TaskTimingPtr::implementation() { verifyIndex(); return getData()->implementation[calculateIndex()]; }
  AGX_FORCE_INLINE agx::Name const& TaskTimingPtr::implementation() const { verifyIndex(); return getData()->implementation[calculateIndex()]; }

  AGX_FORCE_INLINE agx::UInt& TaskTimingPtr::numSubtasks() { verifyIndex(); return getData()->numSubtasks[calculateIndex()]; }
  AGX_FORCE_INLINE agx::UInt const& TaskTimingPtr::numSubtasks() const { verifyIndex(); return getData()->numSubtasks[calculateIndex()]; }

  AGX_FORCE_INLINE agx::Real64& TaskTimingPtr::wallTime() { verifyIndex(); return getData()->wallTime[calculateIndex()]; }
  AGX_FORCE_INLINE agx::Real64 const& TaskTimingPtr::wallTime() const { verifyIndex(); return getData()->wallTime[calculateIndex()]; }

  AGX_FORCE_INLINE agx::Real64& TaskTimingPtr::computeCost() { verifyIndex(); return getData()->computeCost[calculateIndex()]; }
  AGX_FORCE_INLINE agx::Real64 const& TaskTimingPtr::computeCost() const { verifyIndex(); return getData()->computeCost[calculateIndex()]; }

  AGX_FORCE_INLINE agx::Real64& TaskTimingPtr::overheadTime() { verifyIndex(); return getData()->overheadTime[calculateIndex()]; }
  AGX_FORCE_INLINE agx::Real64 const& TaskTimingPtr::overheadTime() const { verifyIndex(); return getData()->overheadTime[calculateIndex()]; }

  //-----------------------------------------------------------------------------------------------------
  AGX_FORCE_INLINE TaskTimingInstance::TaskTimingInstance() {}
  AGX_FORCE_INLINE TaskTimingInstance::TaskTimingInstance(TaskTimingData* data, agx::Index index) : agxData::EntityInstance(data, index) {}
  AGX_FORCE_INLINE TaskTimingInstance::TaskTimingInstance(agxData::EntityStorage* storage, agx::Index index) : agxData::EntityInstance(storage, index) {}
  AGX_FORCE_INLINE TaskTimingInstance::TaskTimingInstance(const agxData::EntityInstance& other) : agxData::EntityInstance(other)
  {
    agxAssertN(!other || other.isInstanceOf(TaskTimingModel::instance()),
      "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
      EntityInstance::getModel()->fullPath().c_str(), TaskTimingModel::instance()->fullPath().c_str());
  }

  AGX_FORCE_INLINE TaskTimingInstance::TaskTimingInstance(const agxData::EntityPtr& ptr) : agxData::EntityInstance(ptr)
  {
    agxAssertN(!ptr || ptr.isInstanceOf(TaskTimingModel::instance()),
      "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
      EntityInstance::getModel()->fullPath().c_str(), TaskTimingModel::instance()->fullPath().c_str());
  }


  AGX_FORCE_INLINE TaskTimingData* TaskTimingInstance::getData() { return static_cast<TaskTimingData* >(agxData::EntityInstance::getData()); }
  AGX_FORCE_INLINE const TaskTimingData* TaskTimingInstance::getData() const { return static_cast<const TaskTimingData* >(agxData::EntityInstance::getData()); }

  AGX_FORCE_INLINE agx::Name& TaskTimingInstance::name() { verifyIndex(); return getData()->name[getIndex()]; }
  AGX_FORCE_INLINE agx::Name const& TaskTimingInstance::name() const { verifyIndex(); return getData()->name[getIndex()]; }

  AGX_FORCE_INLINE agx::Name& TaskTimingInstance::implementation() { verifyIndex(); return getData()->implementation[getIndex()]; }
  AGX_FORCE_INLINE agx::Name const& TaskTimingInstance::implementation() const { verifyIndex(); return getData()->implementation[getIndex()]; }

  AGX_FORCE_INLINE agx::UInt& TaskTimingInstance::numSubtasks() { verifyIndex(); return getData()->numSubtasks[getIndex()]; }
  AGX_FORCE_INLINE agx::UInt const& TaskTimingInstance::numSubtasks() const { verifyIndex(); return getData()->numSubtasks[getIndex()]; }

  AGX_FORCE_INLINE agx::Real64& TaskTimingInstance::wallTime() { verifyIndex(); return getData()->wallTime[getIndex()]; }
  AGX_FORCE_INLINE agx::Real64 const& TaskTimingInstance::wallTime() const { verifyIndex(); return getData()->wallTime[getIndex()]; }

  AGX_FORCE_INLINE agx::Real64& TaskTimingInstance::computeCost() { verifyIndex(); return getData()->computeCost[getIndex()]; }
  AGX_FORCE_INLINE agx::Real64 const& TaskTimingInstance::computeCost() const { verifyIndex(); return getData()->computeCost[getIndex()]; }

  AGX_FORCE_INLINE agx::Real64& TaskTimingInstance::overheadTime() { verifyIndex(); return getData()->overheadTime[getIndex()]; }
  AGX_FORCE_INLINE agx::Real64 const& TaskTimingInstance::overheadTime() const { verifyIndex(); return getData()->overheadTime[getIndex()]; }

  //-----------------------------------------------------------------------------------------------------
  AGX_FORCE_INLINE TaskTimingSemantics::TaskTimingSemantics() {}
  //-----------------------------------------------------------------------------------------------------
  DOXYGEN_END_INTERNAL_BLOCK()
}

AGX_TYPE_BINDING(agx::TaskTimingPtr, "TaskTimingPtr")
AGX_TYPE_BINDING(agx::TaskTimingInstance, "TaskTimingInstance")

#ifdef _MSC_VER
# pragma warning(pop)
#endif

#undef AGX_ENTITY_WRAPPER
#undef AGX_ENTITY_NAMESPACE
#endif

