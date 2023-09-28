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

#ifndef GENERATED_AGX_PHYSICS_HIERARCHICALGRID_CORNERZONE_H_PLUGIN
#define GENERATED_AGX_PHYSICS_HIERARCHICALGRID_CORNERZONE_H_PLUGIN

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
#include <agx/Vec3.h>
#include <agx/Physics/HierarchicalGrid/ContactZoneEntity.h>
#include <agx/Integer.h>
namespace agx { class Job; }

namespace agx { namespace Physics { namespace HierarchicalGrid { class ContactZonePtr; }}}
namespace agx { namespace Physics { namespace HierarchicalGrid { class ContactZonePtr; }}}
namespace agx { namespace Physics { namespace HierarchicalGrid { class ContactZonePtr; }}}

namespace agx
{
  namespace Physics
  {
    namespace HierarchicalGrid
    {

      class CornerZoneModel;
      class CornerZoneData;
      class CornerZonePtr;
      class CornerZoneInstance;
      class CornerZoneSemantics;


      AGX_DECLARE_POINTER_TYPES(CornerZoneModel);

      /** 
      Abstract description of the data attributes for the Physics.HierarchicalGrid.CornerZone entity.
      */ 
      class AGXPHYSICS_EXPORT CornerZoneModel : public agxData::EntityModel
      {
      public:
        typedef CornerZonePtr PtrT;

        CornerZoneModel(const agx::String& name = "CornerZone");

        /// \return The entity model singleton.
        static CornerZoneModel* instance();

        /// Create and return a pointer to a new instance in the default storage for this entity model.
        static CornerZonePtr createInstance();

        /// \return The default storage for this entity model.
        static agxData::EntityStorage* defaultStorage();

        /// This is part of internal cleanup and should not be called by users
        virtual void shutdownCleanup() override;



        /* Attributes */
        static agxData::ScalarAttributeT< agx::Vec3i >* idAttribute;
        static agxData::ScalarAttributeT< agx::Physics::HierarchicalGrid::ContactZonePtr >* diagonal1Attribute;
        static agxData::ScalarAttributeT< agx::Physics::HierarchicalGrid::ContactZonePtr >* diagonal2Attribute;
        static agxData::ScalarAttributeT< agx::Physics::HierarchicalGrid::ContactZonePtr >* internalZonesAttribute;
        static const size_t internalZonesArraySize = 4;
        static agxData::ArrayAttributeT< agx::UInt32 >* particleParticleContactsAttribute;
        static agxData::ArrayAttributeT< agx::UInt32 >* particleGeometryContactsAttribute;
        static agxData::ArrayAttributeT< agx::UInt32 >* geometryGeometryContactsAttribute;
        static agxData::PointerAttributeT< agx::Job*>* jobAttribute;

      protected:
        virtual ~CornerZoneModel();
        virtual agxData::EntityData* createData(agxData::EntityStorage* storage) override;
        virtual void configure(agx::TiXmlElement* eEntity) override;
        virtual void initAttributeAccessors() override;
        virtual void construct(agxData::EntityPtr instance) override;
        void construct(agx::Physics::HierarchicalGrid::CornerZonePtr cornerZone);
      };


      DOXYGEN_START_INTERNAL_BLOCK()
      #ifndef AGX_PHYSICS_HIERARCHICALGRID_CORNERZONE_DATA_SET_OVERRIDE
      #define AGX_PHYSICS_HIERARCHICALGRID_CORNERZONE_DATA_SET
      class AGXPHYSICS_EXPORT CornerZoneData : public agxData::EntityData
      {
      public:
        CornerZoneInstance operator[] (size_t index);

      public:
        agxData::Array< CornerZonePtr >& instance;
        agxData::Array< agx::Vec3i > id;
        agxData::Array< agx::Physics::HierarchicalGrid::ContactZonePtr > diagonal1;
        agxData::Array< agx::Physics::HierarchicalGrid::ContactZonePtr > diagonal2;
        agxData::Array< agx::Physics::HierarchicalGrid::ContactZonePtr > internalZones;
        static const size_t internalZonesArraySize = 4;
        agxData::Array< agxData::Array< agx::UInt32 > > particleParticleContacts;
        agxData::Array< agxData::Array< agx::UInt32 > > particleGeometryContacts;
        agxData::Array< agxData::Array< agx::UInt32 > > geometryGeometryContacts;
        agxData::Array< agx::Job* > job;

      public:
        typedef agx::Vec3i idType;
        typedef agx::Physics::HierarchicalGrid::ContactZonePtr diagonal1Type;
        typedef agx::Physics::HierarchicalGrid::ContactZonePtr diagonal2Type;
        typedef agx::Physics::HierarchicalGrid::ContactZonePtr internalZonesType;
        typedef agxData::Array< agx::UInt32 > particleParticleContactsType;
        typedef agxData::Array< agx::UInt32 > particleGeometryContactsType;
        typedef agxData::Array< agx::UInt32 > geometryGeometryContactsType;
        typedef agx::Job* jobType;

      public:
        CornerZoneData(agxData::EntityStorage* storage);
        CornerZoneData();

      protected:
        virtual ~CornerZoneData() {}
        virtual void setNumElements(agx::Index numElements) override;

      private:
        CornerZoneData& operator= (const CornerZoneData&) { return *this; }

      };
      #endif
      DOXYGEN_END_INTERNAL_BLOCK()


      DOXYGEN_START_INTERNAL_BLOCK()
      class AGXPHYSICS_EXPORT CornerZoneSemantics : protected agxData::EntityPtr
      {
      public:

        // Automatic getters
        agx::Vec3i const& getId() const;
        agx::Physics::HierarchicalGrid::ContactZonePtr const& getDiagonal1() const;
        agx::Physics::HierarchicalGrid::ContactZonePtr const& getDiagonal2() const;
        agxData::Array< agx::UInt32 > const& getParticleParticleContacts() const;
        agxData::Array< agx::UInt32 > const& getParticleGeometryContacts() const;
        agxData::Array< agx::UInt32 > const& getGeometryGeometryContacts() const;
        agx::Job* const& getJob() const;

        // Semantics defined by explicit kernels

        // Automatic setters
        void setId(agx::Vec3i const& value);
        void setDiagonal1(agx::Physics::HierarchicalGrid::ContactZonePtr const& value);
        void setDiagonal2(agx::Physics::HierarchicalGrid::ContactZonePtr const& value);
        void setParticleParticleContacts(agxData::Array< agx::UInt32 > const& value);
        void setParticleGeometryContacts(agxData::Array< agx::UInt32 > const& value);
        void setGeometryGeometryContacts(agxData::Array< agx::UInt32 > const& value);
        void setJob(agx::Job* const& value);


      protected:
        friend class CornerZonePtr;
        friend class CornerZoneInstance;
        CornerZoneSemantics();
      };
      DOXYGEN_END_INTERNAL_BLOCK()


      /**
      Pointer to a entity instance of type Physics.HierarchicalGrid.CornerZone
      */
      class CALLABLE CornerZonePtr : public agxData::EntityPtr
      {
      public:
        typedef CornerZoneModel ModelType;
        typedef CornerZoneData DataType;
        typedef CornerZoneInstance InstanceType;

      public:
        AGXPHYSICS_EXPORT CornerZonePtr();
        AGXPHYSICS_EXPORT CornerZonePtr(agxData::EntityStorage* storage, agx::Index id);
        AGXPHYSICS_EXPORT CornerZonePtr(const agxData::EntityPtr& ptr);
        AGXPHYSICS_EXPORT CornerZonePtr(const agxData::EntityInstance& instance);
        AGXPHYSICS_EXPORT CornerZonePtr& operator= (const agxData::EntityPtr& ptr);
        AGXPHYSICS_EXPORT CornerZonePtr& operator= (const agxData::EntityInstance& instance);
        AGXPHYSICS_EXPORT CornerZoneInstance instance();
        AGXPHYSICS_EXPORT const CornerZoneInstance instance() const;

        AGXPHYSICS_EXPORT CornerZoneSemantics* operator->();
        AGXPHYSICS_EXPORT const CornerZoneSemantics* operator->() const;

        CornerZoneData* getData();
        const CornerZoneData* getData() const;


        /// \return reference to the id attribute
        AGXPHYSICS_EXPORT agx::Vec3i& id();
        /// \return const reference to the id attribute
        AGXPHYSICS_EXPORT agx::Vec3i const& id() const;

        /// \return reference to the diagonal1 attribute
        AGXPHYSICS_EXPORT agx::Physics::HierarchicalGrid::ContactZonePtr& diagonal1();
        /// \return const reference to the diagonal1 attribute
        AGXPHYSICS_EXPORT agx::Physics::HierarchicalGrid::ContactZonePtr const& diagonal1() const;

        /// \return reference to the diagonal2 attribute
        AGXPHYSICS_EXPORT agx::Physics::HierarchicalGrid::ContactZonePtr& diagonal2();
        /// \return const reference to the diagonal2 attribute
        AGXPHYSICS_EXPORT agx::Physics::HierarchicalGrid::ContactZonePtr const& diagonal2() const;

        AGXPHYSICS_EXPORT agxData::Array< agx::Physics::HierarchicalGrid::ContactZonePtr > internalZones();
        AGXPHYSICS_EXPORT agxData::Array< agx::Physics::HierarchicalGrid::ContactZonePtr > const internalZones() const;
        AGXPHYSICS_EXPORT agx::Physics::HierarchicalGrid::ContactZonePtr& internalZones(size_t index);
        AGXPHYSICS_EXPORT agx::Physics::HierarchicalGrid::ContactZonePtr const& internalZones(size_t index) const;

        /// \return reference to the particleParticleContacts attribute
        AGXPHYSICS_EXPORT agxData::Array< agx::UInt32 >& particleParticleContacts();
        /// \return const reference to the particleParticleContacts attribute
        AGXPHYSICS_EXPORT agxData::Array< agx::UInt32 > const& particleParticleContacts() const;

        /// \return reference to the particleGeometryContacts attribute
        AGXPHYSICS_EXPORT agxData::Array< agx::UInt32 >& particleGeometryContacts();
        /// \return const reference to the particleGeometryContacts attribute
        AGXPHYSICS_EXPORT agxData::Array< agx::UInt32 > const& particleGeometryContacts() const;

        /// \return reference to the geometryGeometryContacts attribute
        AGXPHYSICS_EXPORT agxData::Array< agx::UInt32 >& geometryGeometryContacts();
        /// \return const reference to the geometryGeometryContacts attribute
        AGXPHYSICS_EXPORT agxData::Array< agx::UInt32 > const& geometryGeometryContacts() const;

        /// \return reference to the job attribute
        AGXPHYSICS_EXPORT agx::Job*& job();
        /// \return const reference to the job attribute
        AGXPHYSICS_EXPORT agx::Job* const& job() const;

        DOXYGEN_START_INTERNAL_BLOCK()
        // Only used by tolua!
        #ifdef AGX_GENERATED_TOLUA_WRAPPER
        inline agx::Vec3i const& getId() const { return (*this)->getId(); }
        inline agx::Physics::HierarchicalGrid::ContactZonePtr const& getDiagonal1() const { return (*this)->getDiagonal1(); }
        inline agx::Physics::HierarchicalGrid::ContactZonePtr const& getDiagonal2() const { return (*this)->getDiagonal2(); }
        inline agxData::Array< agx::UInt32 > const& getParticleParticleContacts() const { return (*this)->getParticleParticleContacts(); }
        inline agxData::Array< agx::UInt32 > const& getParticleGeometryContacts() const { return (*this)->getParticleGeometryContacts(); }
        inline agxData::Array< agx::UInt32 > const& getGeometryGeometryContacts() const { return (*this)->getGeometryGeometryContacts(); }
        inline agx::Job* const& getJob() const { return (*this)->getJob(); }


        inline void setId(agx::Vec3i const& value) { (*this)->setId(value); }
        inline void setDiagonal1(agx::Physics::HierarchicalGrid::ContactZonePtr const& value) { (*this)->setDiagonal1(value); }
        inline void setDiagonal2(agx::Physics::HierarchicalGrid::ContactZonePtr const& value) { (*this)->setDiagonal2(value); }
        inline void setParticleParticleContacts(agxData::Array< agx::UInt32 > const& value) { (*this)->setParticleParticleContacts(value); }
        inline void setParticleGeometryContacts(agxData::Array< agx::UInt32 > const& value) { (*this)->setParticleGeometryContacts(value); }
        inline void setGeometryGeometryContacts(agxData::Array< agx::UInt32 > const& value) { (*this)->setGeometryGeometryContacts(value); }
        inline void setJob(agx::Job* const& value) { (*this)->setJob(value); }
        #endif
        DOXYGEN_END_INTERNAL_BLOCK()

      };


      DOXYGEN_START_INTERNAL_BLOCK()
      class AGXPHYSICS_EXPORT CornerZoneInstance : public agxData::EntityInstance
      {
      public:
        CornerZoneInstance();
        CornerZoneInstance(CornerZoneData* data, agx::Index index);
        CornerZoneInstance(agxData::EntityStorage *storage, agx::Index index);
        CornerZoneInstance(const agxData::EntityInstance& other);
        CornerZoneInstance(const agxData::EntityPtr& ptr);

        CornerZoneData* getData();
        const CornerZoneData* getData() const;

      public:
        /// \return reference to the id attribute
        agx::Vec3i& id();
        /// \return const reference to the id attribute
        agx::Vec3i const& id() const;

        /// \return reference to the diagonal1 attribute
        agx::Physics::HierarchicalGrid::ContactZonePtr& diagonal1();
        /// \return const reference to the diagonal1 attribute
        agx::Physics::HierarchicalGrid::ContactZonePtr const& diagonal1() const;

        /// \return reference to the diagonal2 attribute
        agx::Physics::HierarchicalGrid::ContactZonePtr& diagonal2();
        /// \return const reference to the diagonal2 attribute
        agx::Physics::HierarchicalGrid::ContactZonePtr const& diagonal2() const;

        agxData::Array< agx::Physics::HierarchicalGrid::ContactZonePtr > internalZones();
        agxData::Array< agx::Physics::HierarchicalGrid::ContactZonePtr > const internalZones() const;
        agx::Physics::HierarchicalGrid::ContactZonePtr& internalZones(size_t index);
        agx::Physics::HierarchicalGrid::ContactZonePtr const& internalZones(size_t index) const;

        /// \return reference to the particleParticleContacts attribute
        agxData::Array< agx::UInt32 >& particleParticleContacts();
        /// \return const reference to the particleParticleContacts attribute
        agxData::Array< agx::UInt32 > const& particleParticleContacts() const;

        /// \return reference to the particleGeometryContacts attribute
        agxData::Array< agx::UInt32 >& particleGeometryContacts();
        /// \return const reference to the particleGeometryContacts attribute
        agxData::Array< agx::UInt32 > const& particleGeometryContacts() const;

        /// \return reference to the geometryGeometryContacts attribute
        agxData::Array< agx::UInt32 >& geometryGeometryContacts();
        /// \return const reference to the geometryGeometryContacts attribute
        agxData::Array< agx::UInt32 > const& geometryGeometryContacts() const;

        /// \return reference to the job attribute
        agx::Job*& job();
        /// \return const reference to the job attribute
        agx::Job* const& job() const;

      };
      DOXYGEN_END_INTERNAL_BLOCK()



      typedef agx::VectorPOD<CornerZonePtr> CornerZonePtrVector;
      typedef agxData::Array<CornerZonePtr> CornerZonePtrArray;



      DOXYGEN_START_INTERNAL_BLOCK()
      /* Implementation */
      //-----------------------------------------------------------------------------------------------------
      //-----------------------------------------------------------------------------------------------------
      inline CornerZoneInstance agx::Physics::HierarchicalGrid::CornerZoneData::operator[] (size_t index) { return CornerZoneInstance(this, (agx::Index)index); }
      //-----------------------------------------------------------------------------------------------------
      AGX_FORCE_INLINE CornerZonePtr::CornerZonePtr() {}
      AGX_FORCE_INLINE CornerZonePtr::CornerZonePtr(agxData::EntityStorage* storage, agx::Index id) : agxData::EntityPtr(storage, id) {}
      AGX_FORCE_INLINE CornerZonePtr::CornerZonePtr(const agxData::EntityPtr& ptr) : agxData::EntityPtr(ptr)
      {
        agxAssertN(!ptr || ptr.isInstanceOf(CornerZoneModel::instance()),
          "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
          EntityPtr::getModel()->fullPath().c_str(), CornerZoneModel::instance()->fullPath().c_str());
      }

      AGX_FORCE_INLINE CornerZonePtr::CornerZonePtr(const agxData::EntityInstance& instance) : agxData::EntityPtr(instance)
      {
        agxAssertN(!instance || instance.isInstanceOf(CornerZoneModel::instance()),
          "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
          EntityPtr::getModel()->fullPath().c_str(), CornerZoneModel::instance()->fullPath().c_str());
      }

      AGX_FORCE_INLINE CornerZonePtr& CornerZonePtr::operator= (const agxData::EntityPtr& ptr)
      {
        agxData::EntityPtr::operator= (ptr);
        agxAssertN(!ptr || ptr.isInstanceOf(CornerZoneModel::instance()),
          "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
          EntityPtr::getModel()->fullPath().c_str(), CornerZoneModel::instance()->fullPath().c_str());
        return *this;
      }

      AGX_FORCE_INLINE CornerZonePtr& CornerZonePtr::operator= (const agxData::EntityInstance& instance)
      {
        agxData::EntityPtr::operator= (instance);
        agxAssertN(!instance || instance.isInstanceOf(CornerZoneModel::instance()),
          "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
          EntityPtr::getModel()->fullPath().c_str(), CornerZoneModel::instance()->fullPath().c_str());
        return *this;
      }

      inline CornerZoneInstance CornerZonePtr::instance() { return agxData::EntityPtr::instance(); }
      inline const CornerZoneInstance CornerZonePtr::instance() const { return agxData::EntityPtr::instance(); }
      AGX_FORCE_INLINE CornerZoneSemantics* CornerZonePtr::operator->() { return (CornerZoneSemantics* )this; }
      AGX_FORCE_INLINE const CornerZoneSemantics* CornerZonePtr::operator->() const { return (const CornerZoneSemantics* )this; }
      AGX_FORCE_INLINE CornerZoneData* CornerZonePtr::getData() { return static_cast<CornerZoneData* >(agxData::EntityPtr::getData()); }
      AGX_FORCE_INLINE const CornerZoneData* CornerZonePtr::getData() const { return static_cast<const CornerZoneData* >(agxData::EntityPtr::getData()); }

      AGX_FORCE_INLINE agx::Vec3i& CornerZonePtr::id() { verifyIndex(); return getData()->id[calculateIndex()]; }
      AGX_FORCE_INLINE agx::Vec3i const& CornerZonePtr::id() const { verifyIndex(); return getData()->id[calculateIndex()]; }

      AGX_FORCE_INLINE agx::Physics::HierarchicalGrid::ContactZonePtr& CornerZonePtr::diagonal1() { verifyIndex(); return getData()->diagonal1[calculateIndex()]; }
      AGX_FORCE_INLINE agx::Physics::HierarchicalGrid::ContactZonePtr const& CornerZonePtr::diagonal1() const { verifyIndex(); return getData()->diagonal1[calculateIndex()]; }

      AGX_FORCE_INLINE agx::Physics::HierarchicalGrid::ContactZonePtr& CornerZonePtr::diagonal2() { verifyIndex(); return getData()->diagonal2[calculateIndex()]; }
      AGX_FORCE_INLINE agx::Physics::HierarchicalGrid::ContactZonePtr const& CornerZonePtr::diagonal2() const { verifyIndex(); return getData()->diagonal2[calculateIndex()]; }

      AGX_FORCE_INLINE agxData::Array< agx::Physics::HierarchicalGrid::ContactZonePtr > CornerZonePtr::internalZones() { verifyIndex(); return getData()->internalZones.slice(agx::IndexRange32(calculateIndex() * (agx::Index)CornerZoneData::internalZonesArraySize, (calculateIndex()+1) * (agx::Index)CornerZoneData::internalZonesArraySize)); }
      AGX_FORCE_INLINE const agxData::Array< agx::Physics::HierarchicalGrid::ContactZonePtr > CornerZonePtr::internalZones() const { verifyIndex(); return getData()->internalZones.slice(agx::IndexRange32(calculateIndex() * (agx::Index)CornerZoneData::internalZonesArraySize, (calculateIndex()+1) * (agx::Index)CornerZoneData::internalZonesArraySize)); }
      AGX_FORCE_INLINE agx::Physics::HierarchicalGrid::ContactZonePtr& CornerZonePtr::internalZones(size_t index) { return this->internalZones()[index]; }
      AGX_FORCE_INLINE agx::Physics::HierarchicalGrid::ContactZonePtr const& CornerZonePtr::internalZones(size_t index) const { return this->internalZones()[index]; }

      AGX_FORCE_INLINE agxData::Array< agx::UInt32 >& CornerZonePtr::particleParticleContacts() { verifyIndex(); return getData()->particleParticleContacts[calculateIndex()]; }
      AGX_FORCE_INLINE agxData::Array< agx::UInt32 > const& CornerZonePtr::particleParticleContacts() const { verifyIndex(); return getData()->particleParticleContacts[calculateIndex()]; }

      AGX_FORCE_INLINE agxData::Array< agx::UInt32 >& CornerZonePtr::particleGeometryContacts() { verifyIndex(); return getData()->particleGeometryContacts[calculateIndex()]; }
      AGX_FORCE_INLINE agxData::Array< agx::UInt32 > const& CornerZonePtr::particleGeometryContacts() const { verifyIndex(); return getData()->particleGeometryContacts[calculateIndex()]; }

      AGX_FORCE_INLINE agxData::Array< agx::UInt32 >& CornerZonePtr::geometryGeometryContacts() { verifyIndex(); return getData()->geometryGeometryContacts[calculateIndex()]; }
      AGX_FORCE_INLINE agxData::Array< agx::UInt32 > const& CornerZonePtr::geometryGeometryContacts() const { verifyIndex(); return getData()->geometryGeometryContacts[calculateIndex()]; }

      AGX_FORCE_INLINE agx::Job*& CornerZonePtr::job() { verifyIndex(); return getData()->job[calculateIndex()]; }
      AGX_FORCE_INLINE agx::Job* const& CornerZonePtr::job() const { verifyIndex(); return getData()->job[calculateIndex()]; }

      //-----------------------------------------------------------------------------------------------------
      AGX_FORCE_INLINE CornerZoneInstance::CornerZoneInstance() {}
      AGX_FORCE_INLINE CornerZoneInstance::CornerZoneInstance(CornerZoneData* data, agx::Index index) : agxData::EntityInstance(data, index) {}
      AGX_FORCE_INLINE CornerZoneInstance::CornerZoneInstance(agxData::EntityStorage* storage, agx::Index index) : agxData::EntityInstance(storage, index) {}
      AGX_FORCE_INLINE CornerZoneInstance::CornerZoneInstance(const agxData::EntityInstance& other) : agxData::EntityInstance(other)
      {
        agxAssertN(!other || other.isInstanceOf(CornerZoneModel::instance()),
          "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
          EntityInstance::getModel()->fullPath().c_str(), CornerZoneModel::instance()->fullPath().c_str());
      }

      AGX_FORCE_INLINE CornerZoneInstance::CornerZoneInstance(const agxData::EntityPtr& ptr) : agxData::EntityInstance(ptr)
      {
        agxAssertN(!ptr || ptr.isInstanceOf(CornerZoneModel::instance()),
          "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
          EntityInstance::getModel()->fullPath().c_str(), CornerZoneModel::instance()->fullPath().c_str());
      }


      AGX_FORCE_INLINE CornerZoneData* CornerZoneInstance::getData() { return static_cast<CornerZoneData* >(agxData::EntityInstance::getData()); }
      AGX_FORCE_INLINE const CornerZoneData* CornerZoneInstance::getData() const { return static_cast<const CornerZoneData* >(agxData::EntityInstance::getData()); }

      AGX_FORCE_INLINE agx::Vec3i& CornerZoneInstance::id() { verifyIndex(); return getData()->id[getIndex()]; }
      AGX_FORCE_INLINE agx::Vec3i const& CornerZoneInstance::id() const { verifyIndex(); return getData()->id[getIndex()]; }

      AGX_FORCE_INLINE agx::Physics::HierarchicalGrid::ContactZonePtr& CornerZoneInstance::diagonal1() { verifyIndex(); return getData()->diagonal1[getIndex()]; }
      AGX_FORCE_INLINE agx::Physics::HierarchicalGrid::ContactZonePtr const& CornerZoneInstance::diagonal1() const { verifyIndex(); return getData()->diagonal1[getIndex()]; }

      AGX_FORCE_INLINE agx::Physics::HierarchicalGrid::ContactZonePtr& CornerZoneInstance::diagonal2() { verifyIndex(); return getData()->diagonal2[getIndex()]; }
      AGX_FORCE_INLINE agx::Physics::HierarchicalGrid::ContactZonePtr const& CornerZoneInstance::diagonal2() const { verifyIndex(); return getData()->diagonal2[getIndex()]; }

      AGX_FORCE_INLINE agxData::Array< agx::Physics::HierarchicalGrid::ContactZonePtr > CornerZoneInstance::internalZones() { verifyIndex(); return getData()->internalZones.slice(agx::IndexRange32(getIndex() * (agx::Index)CornerZoneData::internalZonesArraySize, (getIndex()+1) * (agx::Index)CornerZoneData::internalZonesArraySize)); }
      AGX_FORCE_INLINE const agxData::Array< agx::Physics::HierarchicalGrid::ContactZonePtr > CornerZoneInstance::internalZones() const { verifyIndex(); return getData()->internalZones.slice(agx::IndexRange32(getIndex() * (agx::Index)CornerZoneData::internalZonesArraySize, (getIndex()+1) * (agx::Index)CornerZoneData::internalZonesArraySize)); }
      AGX_FORCE_INLINE agx::Physics::HierarchicalGrid::ContactZonePtr& CornerZoneInstance::internalZones(size_t index) { return this->internalZones()[index]; }
      AGX_FORCE_INLINE agx::Physics::HierarchicalGrid::ContactZonePtr const& CornerZoneInstance::internalZones(size_t index) const { return this->internalZones()[index]; }

      AGX_FORCE_INLINE agxData::Array< agx::UInt32 >& CornerZoneInstance::particleParticleContacts() { verifyIndex(); return getData()->particleParticleContacts[getIndex()]; }
      AGX_FORCE_INLINE agxData::Array< agx::UInt32 > const& CornerZoneInstance::particleParticleContacts() const { verifyIndex(); return getData()->particleParticleContacts[getIndex()]; }

      AGX_FORCE_INLINE agxData::Array< agx::UInt32 >& CornerZoneInstance::particleGeometryContacts() { verifyIndex(); return getData()->particleGeometryContacts[getIndex()]; }
      AGX_FORCE_INLINE agxData::Array< agx::UInt32 > const& CornerZoneInstance::particleGeometryContacts() const { verifyIndex(); return getData()->particleGeometryContacts[getIndex()]; }

      AGX_FORCE_INLINE agxData::Array< agx::UInt32 >& CornerZoneInstance::geometryGeometryContacts() { verifyIndex(); return getData()->geometryGeometryContacts[getIndex()]; }
      AGX_FORCE_INLINE agxData::Array< agx::UInt32 > const& CornerZoneInstance::geometryGeometryContacts() const { verifyIndex(); return getData()->geometryGeometryContacts[getIndex()]; }

      AGX_FORCE_INLINE agx::Job*& CornerZoneInstance::job() { verifyIndex(); return getData()->job[getIndex()]; }
      AGX_FORCE_INLINE agx::Job* const& CornerZoneInstance::job() const { verifyIndex(); return getData()->job[getIndex()]; }

      //-----------------------------------------------------------------------------------------------------
      AGX_FORCE_INLINE CornerZoneSemantics::CornerZoneSemantics() {}
      //-----------------------------------------------------------------------------------------------------
      DOXYGEN_END_INTERNAL_BLOCK()
    }
  }
}

AGX_TYPE_BINDING(agx::Physics::HierarchicalGrid::CornerZonePtr, "Physics.HierarchicalGrid.CornerZonePtr")
AGX_TYPE_BINDING(agx::Physics::HierarchicalGrid::CornerZoneInstance, "Physics.HierarchicalGrid.CornerZoneInstance")

#ifdef _MSC_VER
# pragma warning(pop)
#endif

#undef AGX_ENTITY_WRAPPER
#undef AGX_ENTITY_NAMESPACE
#endif

