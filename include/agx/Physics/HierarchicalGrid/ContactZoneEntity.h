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

#ifndef GENERATED_AGX_PHYSICS_HIERARCHICALGRID_CONTACTZONE_H_PLUGIN
#define GENERATED_AGX_PHYSICS_HIERARCHICALGRID_CONTACTZONE_H_PLUGIN

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
#include <agx/AtomicValue.h>
#include <agx/Physics/SolveGroupEntity.h>
#include <agx/Physics/HierarchicalGrid/CellEntity.h>
#include <agx/Physics/HierarchicalGrid/GridTierEntity.h>
#include <agx/Physics/HierarchicalGrid/ContactZoneDependencyEntity.h>
#include <agx/Vec3.h>
#include <agx/Vec4.h>
namespace agx { class Job; }

namespace agx { namespace Physics { namespace HierarchicalGrid { class CellPtr; }}}
namespace agx { namespace Physics { namespace HierarchicalGrid { class GridTierPtr; }}}
namespace agx { namespace Physics { namespace HierarchicalGrid { class ContactZoneDependencyPtr; }}}
namespace agx { namespace Physics { class SolveGroupPtr; }}

namespace agx
{
  namespace Physics
  {
    namespace HierarchicalGrid
    {

      class ContactZoneModel;
      class ContactZoneData;
      class ContactZonePtr;
      class ContactZoneInstance;
      class ContactZoneSemantics;


      AGX_DECLARE_POINTER_TYPES(ContactZoneModel);

      /** 
      Abstract description of the data attributes for the Physics.HierarchicalGrid.ContactZone entity.
      */ 
      class AGXPHYSICS_EXPORT ContactZoneModel : public agxData::EntityModel
      {
      public:
        typedef ContactZonePtr PtrT;

        ContactZoneModel(const agx::String& name = "ContactZone");

        /// \return The entity model singleton.
        static ContactZoneModel* instance();

        /// Create and return a pointer to a new instance in the default storage for this entity model.
        static ContactZonePtr createInstance();

        /// \return The default storage for this entity model.
        static agxData::EntityStorage* defaultStorage();

        /// This is part of internal cleanup and should not be called by users
        virtual void shutdownCleanup() override;



        /* Attributes */
        static agxData::ArrayAttributeT< agx::UInt32 >* particleParticleContactsAttribute;
        static agxData::ArrayAttributeT< agx::UInt32 >* particleGeometryContactsAttribute;
        static agxData::ArrayAttributeT< agx::UInt32 >* geometryGeometryContactsAttribute;
        static agxData::ScalarAttributeT< agx::UInt32 >* totNumContactsAttribute;
        static agxData::ScalarAttributeT< agx::AtomicValue >* numParticleParticleContactsAttribute;
        static agxData::ScalarAttributeT< agx::AtomicValue >* numParticleGeometryContactsAttribute;
        static agxData::ScalarAttributeT< agx::AtomicValue >* numGeometryContactsAttribute;
        static agxData::ArrayAttributeT< agx::Physics::SolveGroupPtr >* solveGroupsAttribute;
        static agxData::ScalarAttributeT< agx::Physics::HierarchicalGrid::CellPtr >* spatialCellAttribute;
        static agxData::ScalarAttributeT< agx::Physics::HierarchicalGrid::GridTierPtr >* tierAttribute;
        static agxData::ScalarAttributeT< agx::Physics::HierarchicalGrid::ContactZoneDependencyPtr >* interTierDependencyListAttribute;
        static agxData::ScalarAttributeT< agx::UInt8 >* typeAttribute;
        static agxData::ScalarAttributeT< agx::Vec3i >* idAttribute;
        static agxData::PointerAttributeT< agx::Job*>* jobAttribute;
        static agxData::ScalarAttributeT< agx::Vec4f >* colorAttribute;

      protected:
        virtual ~ContactZoneModel();
        virtual agxData::EntityData* createData(agxData::EntityStorage* storage) override;
        virtual void configure(agx::TiXmlElement* eEntity) override;
        virtual void initAttributeAccessors() override;
        virtual void construct(agxData::EntityPtr instance) override;
        void construct(agx::Physics::HierarchicalGrid::ContactZonePtr contactZone);
      };


      DOXYGEN_START_INTERNAL_BLOCK()
      #ifndef AGX_PHYSICS_HIERARCHICALGRID_CONTACTZONE_DATA_SET_OVERRIDE
      #define AGX_PHYSICS_HIERARCHICALGRID_CONTACTZONE_DATA_SET
      class AGXPHYSICS_EXPORT ContactZoneData : public agxData::EntityData
      {
      public:
        ContactZoneInstance operator[] (size_t index);

      public:
        agxData::Array< ContactZonePtr >& instance;
        agxData::Array< agxData::Array< agx::UInt32 > > particleParticleContacts;
        agxData::Array< agxData::Array< agx::UInt32 > > particleGeometryContacts;
        agxData::Array< agxData::Array< agx::UInt32 > > geometryGeometryContacts;
        agxData::Array< agx::UInt32 > totNumContacts;
        agxData::Array< agx::AtomicValue > numParticleParticleContacts;
        agxData::Array< agx::AtomicValue > numParticleGeometryContacts;
        agxData::Array< agx::AtomicValue > numGeometryContacts;
        agxData::Array< agxData::Array< agx::Physics::SolveGroupPtr > > solveGroups;
        agxData::Array< agx::Physics::HierarchicalGrid::CellPtr > spatialCell;
        agxData::Array< agx::Physics::HierarchicalGrid::GridTierPtr > tier;
        agxData::Array< agx::Physics::HierarchicalGrid::ContactZoneDependencyPtr > interTierDependencyList;
        agxData::Array< agx::UInt8 > type;
        agxData::Array< agx::Vec3i > id;
        agxData::Array< agx::Job* > job;
        agxData::Array< agx::Vec4f > color;

      public:
        typedef agxData::Array< agx::UInt32 > particleParticleContactsType;
        typedef agxData::Array< agx::UInt32 > particleGeometryContactsType;
        typedef agxData::Array< agx::UInt32 > geometryGeometryContactsType;
        typedef agx::UInt32 totNumContactsType;
        typedef agx::AtomicValue numParticleParticleContactsType;
        typedef agx::AtomicValue numParticleGeometryContactsType;
        typedef agx::AtomicValue numGeometryContactsType;
        typedef agxData::Array< agx::Physics::SolveGroupPtr > solveGroupsType;
        typedef agx::Physics::HierarchicalGrid::CellPtr spatialCellType;
        typedef agx::Physics::HierarchicalGrid::GridTierPtr tierType;
        typedef agx::Physics::HierarchicalGrid::ContactZoneDependencyPtr interTierDependencyListType;
        typedef agx::UInt8 typeType;
        typedef agx::Vec3i idType;
        typedef agx::Job* jobType;
        typedef agx::Vec4f colorType;

      public:
        ContactZoneData(agxData::EntityStorage* storage);
        ContactZoneData();

      protected:
        virtual ~ContactZoneData() {}
        virtual void setNumElements(agx::Index numElements) override;

      private:
        ContactZoneData& operator= (const ContactZoneData&) { return *this; }

      };
      #endif
      DOXYGEN_END_INTERNAL_BLOCK()


      DOXYGEN_START_INTERNAL_BLOCK()
      class AGXPHYSICS_EXPORT ContactZoneSemantics : protected agxData::EntityPtr
      {
      public:

        // Automatic getters
        agxData::Array< agx::UInt32 > const& getParticleParticleContacts() const;
        agxData::Array< agx::UInt32 > const& getParticleGeometryContacts() const;
        agxData::Array< agx::UInt32 > const& getGeometryGeometryContacts() const;
        agx::UInt32 const& getTotNumContacts() const;
        agx::AtomicValue const& getNumParticleParticleContacts() const;
        agx::AtomicValue const& getNumParticleGeometryContacts() const;
        agx::AtomicValue const& getNumGeometryContacts() const;
        agxData::Array< agx::Physics::SolveGroupPtr > const& getSolveGroups() const;
        agx::Physics::HierarchicalGrid::CellPtr const& getSpatialCell() const;
        agx::Physics::HierarchicalGrid::GridTierPtr const& getTier() const;
        agx::Physics::HierarchicalGrid::ContactZoneDependencyPtr const& getInterTierDependencyList() const;
        agx::UInt8 const& getType() const;
        agx::Vec3i const& getId() const;
        agx::Job* const& getJob() const;
        agx::Vec4f const& getColor() const;

        // Semantics defined by explicit kernels

        // Automatic setters
        void setParticleParticleContacts(agxData::Array< agx::UInt32 > const& value);
        void setParticleGeometryContacts(agxData::Array< agx::UInt32 > const& value);
        void setGeometryGeometryContacts(agxData::Array< agx::UInt32 > const& value);
        void setTotNumContacts(agx::UInt32 const& value);
        void setNumParticleParticleContacts(agx::AtomicValue const& value);
        void setNumParticleGeometryContacts(agx::AtomicValue const& value);
        void setNumGeometryContacts(agx::AtomicValue const& value);
        void setSolveGroups(agxData::Array< agx::Physics::SolveGroupPtr > const& value);
        void setSpatialCell(agx::Physics::HierarchicalGrid::CellPtr const& value);
        void setTier(agx::Physics::HierarchicalGrid::GridTierPtr const& value);
        void setInterTierDependencyList(agx::Physics::HierarchicalGrid::ContactZoneDependencyPtr const& value);
        void setType(agx::UInt8 const& value);
        void setId(agx::Vec3i const& value);
        void setJob(agx::Job* const& value);
        void setColor(agx::Vec4f const& value);


      protected:
        friend class ContactZonePtr;
        friend class ContactZoneInstance;
        ContactZoneSemantics();
      };
      DOXYGEN_END_INTERNAL_BLOCK()


      /**
      Pointer to a entity instance of type Physics.HierarchicalGrid.ContactZone
      */
      class CALLABLE ContactZonePtr : public agxData::EntityPtr
      {
      public:
        typedef ContactZoneModel ModelType;
        typedef ContactZoneData DataType;
        typedef ContactZoneInstance InstanceType;

      public:
        AGXPHYSICS_EXPORT ContactZonePtr();
        AGXPHYSICS_EXPORT ContactZonePtr(agxData::EntityStorage* storage, agx::Index id);
        AGXPHYSICS_EXPORT ContactZonePtr(const agxData::EntityPtr& ptr);
        AGXPHYSICS_EXPORT ContactZonePtr(const agxData::EntityInstance& instance);
        AGXPHYSICS_EXPORT ContactZonePtr& operator= (const agxData::EntityPtr& ptr);
        AGXPHYSICS_EXPORT ContactZonePtr& operator= (const agxData::EntityInstance& instance);
        AGXPHYSICS_EXPORT ContactZoneInstance instance();
        AGXPHYSICS_EXPORT const ContactZoneInstance instance() const;

        AGXPHYSICS_EXPORT ContactZoneSemantics* operator->();
        AGXPHYSICS_EXPORT const ContactZoneSemantics* operator->() const;

        ContactZoneData* getData();
        const ContactZoneData* getData() const;


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

        /// \return reference to the totNumContacts attribute
        AGXPHYSICS_EXPORT agx::UInt32& totNumContacts();
        /// \return const reference to the totNumContacts attribute
        AGXPHYSICS_EXPORT agx::UInt32 const& totNumContacts() const;

        /// \return reference to the numParticleParticleContacts attribute
        AGXPHYSICS_EXPORT agx::AtomicValue& numParticleParticleContacts();
        /// \return const reference to the numParticleParticleContacts attribute
        AGXPHYSICS_EXPORT agx::AtomicValue const& numParticleParticleContacts() const;

        /// \return reference to the numParticleGeometryContacts attribute
        AGXPHYSICS_EXPORT agx::AtomicValue& numParticleGeometryContacts();
        /// \return const reference to the numParticleGeometryContacts attribute
        AGXPHYSICS_EXPORT agx::AtomicValue const& numParticleGeometryContacts() const;

        /// \return reference to the numGeometryContacts attribute
        AGXPHYSICS_EXPORT agx::AtomicValue& numGeometryContacts();
        /// \return const reference to the numGeometryContacts attribute
        AGXPHYSICS_EXPORT agx::AtomicValue const& numGeometryContacts() const;

        /// \return reference to the solveGroups attribute
        AGXPHYSICS_EXPORT agxData::Array< agx::Physics::SolveGroupPtr >& solveGroups();
        /// \return const reference to the solveGroups attribute
        AGXPHYSICS_EXPORT agxData::Array< agx::Physics::SolveGroupPtr > const& solveGroups() const;

        /// \return reference to the spatialCell attribute
        AGXPHYSICS_EXPORT agx::Physics::HierarchicalGrid::CellPtr& spatialCell();
        /// \return const reference to the spatialCell attribute
        AGXPHYSICS_EXPORT agx::Physics::HierarchicalGrid::CellPtr const& spatialCell() const;

        /// \return reference to the tier attribute
        AGXPHYSICS_EXPORT agx::Physics::HierarchicalGrid::GridTierPtr& tier();
        /// \return const reference to the tier attribute
        AGXPHYSICS_EXPORT agx::Physics::HierarchicalGrid::GridTierPtr const& tier() const;

        /// \return reference to the interTierDependencyList attribute
        AGXPHYSICS_EXPORT agx::Physics::HierarchicalGrid::ContactZoneDependencyPtr& interTierDependencyList();
        /// \return const reference to the interTierDependencyList attribute
        AGXPHYSICS_EXPORT agx::Physics::HierarchicalGrid::ContactZoneDependencyPtr const& interTierDependencyList() const;

        /// \return reference to the type attribute
        AGXPHYSICS_EXPORT agx::UInt8& type();
        /// \return const reference to the type attribute
        AGXPHYSICS_EXPORT agx::UInt8 const& type() const;

        /// \return reference to the id attribute
        AGXPHYSICS_EXPORT agx::Vec3i& id();
        /// \return const reference to the id attribute
        AGXPHYSICS_EXPORT agx::Vec3i const& id() const;

        /// \return reference to the job attribute
        AGXPHYSICS_EXPORT agx::Job*& job();
        /// \return const reference to the job attribute
        AGXPHYSICS_EXPORT agx::Job* const& job() const;

        /// \return reference to the color attribute
        AGXPHYSICS_EXPORT agx::Vec4f& color();
        /// \return const reference to the color attribute
        AGXPHYSICS_EXPORT agx::Vec4f const& color() const;

        DOXYGEN_START_INTERNAL_BLOCK()
        // Only used by tolua!
        #ifdef AGX_GENERATED_TOLUA_WRAPPER
        inline agxData::Array< agx::UInt32 > const& getParticleParticleContacts() const { return (*this)->getParticleParticleContacts(); }
        inline agxData::Array< agx::UInt32 > const& getParticleGeometryContacts() const { return (*this)->getParticleGeometryContacts(); }
        inline agxData::Array< agx::UInt32 > const& getGeometryGeometryContacts() const { return (*this)->getGeometryGeometryContacts(); }
        inline agx::UInt32 const& getTotNumContacts() const { return (*this)->getTotNumContacts(); }
        inline agx::AtomicValue const& getNumParticleParticleContacts() const { return (*this)->getNumParticleParticleContacts(); }
        inline agx::AtomicValue const& getNumParticleGeometryContacts() const { return (*this)->getNumParticleGeometryContacts(); }
        inline agx::AtomicValue const& getNumGeometryContacts() const { return (*this)->getNumGeometryContacts(); }
        inline agxData::Array< agx::Physics::SolveGroupPtr > const& getSolveGroups() const { return (*this)->getSolveGroups(); }
        inline agx::Physics::HierarchicalGrid::CellPtr const& getSpatialCell() const { return (*this)->getSpatialCell(); }
        inline agx::Physics::HierarchicalGrid::GridTierPtr const& getTier() const { return (*this)->getTier(); }
        inline agx::Physics::HierarchicalGrid::ContactZoneDependencyPtr const& getInterTierDependencyList() const { return (*this)->getInterTierDependencyList(); }
        inline agx::UInt8 const& getType() const { return (*this)->getType(); }
        inline agx::Vec3i const& getId() const { return (*this)->getId(); }
        inline agx::Job* const& getJob() const { return (*this)->getJob(); }
        inline agx::Vec4f const& getColor() const { return (*this)->getColor(); }


        inline void setParticleParticleContacts(agxData::Array< agx::UInt32 > const& value) { (*this)->setParticleParticleContacts(value); }
        inline void setParticleGeometryContacts(agxData::Array< agx::UInt32 > const& value) { (*this)->setParticleGeometryContacts(value); }
        inline void setGeometryGeometryContacts(agxData::Array< agx::UInt32 > const& value) { (*this)->setGeometryGeometryContacts(value); }
        inline void setTotNumContacts(agx::UInt32 const& value) { (*this)->setTotNumContacts(value); }
        inline void setNumParticleParticleContacts(agx::AtomicValue const& value) { (*this)->setNumParticleParticleContacts(value); }
        inline void setNumParticleGeometryContacts(agx::AtomicValue const& value) { (*this)->setNumParticleGeometryContacts(value); }
        inline void setNumGeometryContacts(agx::AtomicValue const& value) { (*this)->setNumGeometryContacts(value); }
        inline void setSolveGroups(agxData::Array< agx::Physics::SolveGroupPtr > const& value) { (*this)->setSolveGroups(value); }
        inline void setSpatialCell(agx::Physics::HierarchicalGrid::CellPtr const& value) { (*this)->setSpatialCell(value); }
        inline void setTier(agx::Physics::HierarchicalGrid::GridTierPtr const& value) { (*this)->setTier(value); }
        inline void setInterTierDependencyList(agx::Physics::HierarchicalGrid::ContactZoneDependencyPtr const& value) { (*this)->setInterTierDependencyList(value); }
        inline void setType(agx::UInt8 const& value) { (*this)->setType(value); }
        inline void setId(agx::Vec3i const& value) { (*this)->setId(value); }
        inline void setJob(agx::Job* const& value) { (*this)->setJob(value); }
        inline void setColor(agx::Vec4f const& value) { (*this)->setColor(value); }
        #endif
        DOXYGEN_END_INTERNAL_BLOCK()

      };


      DOXYGEN_START_INTERNAL_BLOCK()
      class AGXPHYSICS_EXPORT ContactZoneInstance : public agxData::EntityInstance
      {
      public:
        ContactZoneInstance();
        ContactZoneInstance(ContactZoneData* data, agx::Index index);
        ContactZoneInstance(agxData::EntityStorage *storage, agx::Index index);
        ContactZoneInstance(const agxData::EntityInstance& other);
        ContactZoneInstance(const agxData::EntityPtr& ptr);

        ContactZoneData* getData();
        const ContactZoneData* getData() const;

      public:
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

        /// \return reference to the totNumContacts attribute
        agx::UInt32& totNumContacts();
        /// \return const reference to the totNumContacts attribute
        agx::UInt32 const& totNumContacts() const;

        /// \return reference to the numParticleParticleContacts attribute
        agx::AtomicValue& numParticleParticleContacts();
        /// \return const reference to the numParticleParticleContacts attribute
        agx::AtomicValue const& numParticleParticleContacts() const;

        /// \return reference to the numParticleGeometryContacts attribute
        agx::AtomicValue& numParticleGeometryContacts();
        /// \return const reference to the numParticleGeometryContacts attribute
        agx::AtomicValue const& numParticleGeometryContacts() const;

        /// \return reference to the numGeometryContacts attribute
        agx::AtomicValue& numGeometryContacts();
        /// \return const reference to the numGeometryContacts attribute
        agx::AtomicValue const& numGeometryContacts() const;

        /// \return reference to the solveGroups attribute
        agxData::Array< agx::Physics::SolveGroupPtr >& solveGroups();
        /// \return const reference to the solveGroups attribute
        agxData::Array< agx::Physics::SolveGroupPtr > const& solveGroups() const;

        /// \return reference to the spatialCell attribute
        agx::Physics::HierarchicalGrid::CellPtr& spatialCell();
        /// \return const reference to the spatialCell attribute
        agx::Physics::HierarchicalGrid::CellPtr const& spatialCell() const;

        /// \return reference to the tier attribute
        agx::Physics::HierarchicalGrid::GridTierPtr& tier();
        /// \return const reference to the tier attribute
        agx::Physics::HierarchicalGrid::GridTierPtr const& tier() const;

        /// \return reference to the interTierDependencyList attribute
        agx::Physics::HierarchicalGrid::ContactZoneDependencyPtr& interTierDependencyList();
        /// \return const reference to the interTierDependencyList attribute
        agx::Physics::HierarchicalGrid::ContactZoneDependencyPtr const& interTierDependencyList() const;

        /// \return reference to the type attribute
        agx::UInt8& type();
        /// \return const reference to the type attribute
        agx::UInt8 const& type() const;

        /// \return reference to the id attribute
        agx::Vec3i& id();
        /// \return const reference to the id attribute
        agx::Vec3i const& id() const;

        /// \return reference to the job attribute
        agx::Job*& job();
        /// \return const reference to the job attribute
        agx::Job* const& job() const;

        /// \return reference to the color attribute
        agx::Vec4f& color();
        /// \return const reference to the color attribute
        agx::Vec4f const& color() const;

      };
      DOXYGEN_END_INTERNAL_BLOCK()



      typedef agx::VectorPOD<ContactZonePtr> ContactZonePtrVector;
      typedef agxData::Array<ContactZonePtr> ContactZonePtrArray;



      DOXYGEN_START_INTERNAL_BLOCK()
      /* Implementation */
      //-----------------------------------------------------------------------------------------------------
      //-----------------------------------------------------------------------------------------------------
      inline ContactZoneInstance agx::Physics::HierarchicalGrid::ContactZoneData::operator[] (size_t index) { return ContactZoneInstance(this, (agx::Index)index); }
      //-----------------------------------------------------------------------------------------------------
      AGX_FORCE_INLINE ContactZonePtr::ContactZonePtr() {}
      AGX_FORCE_INLINE ContactZonePtr::ContactZonePtr(agxData::EntityStorage* storage, agx::Index id) : agxData::EntityPtr(storage, id) {}
      AGX_FORCE_INLINE ContactZonePtr::ContactZonePtr(const agxData::EntityPtr& ptr) : agxData::EntityPtr(ptr)
      {
        agxAssertN(!ptr || ptr.isInstanceOf(ContactZoneModel::instance()),
          "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
          EntityPtr::getModel()->fullPath().c_str(), ContactZoneModel::instance()->fullPath().c_str());
      }

      AGX_FORCE_INLINE ContactZonePtr::ContactZonePtr(const agxData::EntityInstance& instance) : agxData::EntityPtr(instance)
      {
        agxAssertN(!instance || instance.isInstanceOf(ContactZoneModel::instance()),
          "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
          EntityPtr::getModel()->fullPath().c_str(), ContactZoneModel::instance()->fullPath().c_str());
      }

      AGX_FORCE_INLINE ContactZonePtr& ContactZonePtr::operator= (const agxData::EntityPtr& ptr)
      {
        agxData::EntityPtr::operator= (ptr);
        agxAssertN(!ptr || ptr.isInstanceOf(ContactZoneModel::instance()),
          "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
          EntityPtr::getModel()->fullPath().c_str(), ContactZoneModel::instance()->fullPath().c_str());
        return *this;
      }

      AGX_FORCE_INLINE ContactZonePtr& ContactZonePtr::operator= (const agxData::EntityInstance& instance)
      {
        agxData::EntityPtr::operator= (instance);
        agxAssertN(!instance || instance.isInstanceOf(ContactZoneModel::instance()),
          "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
          EntityPtr::getModel()->fullPath().c_str(), ContactZoneModel::instance()->fullPath().c_str());
        return *this;
      }

      inline ContactZoneInstance ContactZonePtr::instance() { return agxData::EntityPtr::instance(); }
      inline const ContactZoneInstance ContactZonePtr::instance() const { return agxData::EntityPtr::instance(); }
      AGX_FORCE_INLINE ContactZoneSemantics* ContactZonePtr::operator->() { return (ContactZoneSemantics* )this; }
      AGX_FORCE_INLINE const ContactZoneSemantics* ContactZonePtr::operator->() const { return (const ContactZoneSemantics* )this; }
      AGX_FORCE_INLINE ContactZoneData* ContactZonePtr::getData() { return static_cast<ContactZoneData* >(agxData::EntityPtr::getData()); }
      AGX_FORCE_INLINE const ContactZoneData* ContactZonePtr::getData() const { return static_cast<const ContactZoneData* >(agxData::EntityPtr::getData()); }

      AGX_FORCE_INLINE agxData::Array< agx::UInt32 >& ContactZonePtr::particleParticleContacts() { verifyIndex(); return getData()->particleParticleContacts[calculateIndex()]; }
      AGX_FORCE_INLINE agxData::Array< agx::UInt32 > const& ContactZonePtr::particleParticleContacts() const { verifyIndex(); return getData()->particleParticleContacts[calculateIndex()]; }

      AGX_FORCE_INLINE agxData::Array< agx::UInt32 >& ContactZonePtr::particleGeometryContacts() { verifyIndex(); return getData()->particleGeometryContacts[calculateIndex()]; }
      AGX_FORCE_INLINE agxData::Array< agx::UInt32 > const& ContactZonePtr::particleGeometryContacts() const { verifyIndex(); return getData()->particleGeometryContacts[calculateIndex()]; }

      AGX_FORCE_INLINE agxData::Array< agx::UInt32 >& ContactZonePtr::geometryGeometryContacts() { verifyIndex(); return getData()->geometryGeometryContacts[calculateIndex()]; }
      AGX_FORCE_INLINE agxData::Array< agx::UInt32 > const& ContactZonePtr::geometryGeometryContacts() const { verifyIndex(); return getData()->geometryGeometryContacts[calculateIndex()]; }

      AGX_FORCE_INLINE agx::UInt32& ContactZonePtr::totNumContacts() { verifyIndex(); return getData()->totNumContacts[calculateIndex()]; }
      AGX_FORCE_INLINE agx::UInt32 const& ContactZonePtr::totNumContacts() const { verifyIndex(); return getData()->totNumContacts[calculateIndex()]; }

      AGX_FORCE_INLINE agx::AtomicValue& ContactZonePtr::numParticleParticleContacts() { verifyIndex(); return getData()->numParticleParticleContacts[calculateIndex()]; }
      AGX_FORCE_INLINE agx::AtomicValue const& ContactZonePtr::numParticleParticleContacts() const { verifyIndex(); return getData()->numParticleParticleContacts[calculateIndex()]; }

      AGX_FORCE_INLINE agx::AtomicValue& ContactZonePtr::numParticleGeometryContacts() { verifyIndex(); return getData()->numParticleGeometryContacts[calculateIndex()]; }
      AGX_FORCE_INLINE agx::AtomicValue const& ContactZonePtr::numParticleGeometryContacts() const { verifyIndex(); return getData()->numParticleGeometryContacts[calculateIndex()]; }

      AGX_FORCE_INLINE agx::AtomicValue& ContactZonePtr::numGeometryContacts() { verifyIndex(); return getData()->numGeometryContacts[calculateIndex()]; }
      AGX_FORCE_INLINE agx::AtomicValue const& ContactZonePtr::numGeometryContacts() const { verifyIndex(); return getData()->numGeometryContacts[calculateIndex()]; }

      AGX_FORCE_INLINE agxData::Array< agx::Physics::SolveGroupPtr >& ContactZonePtr::solveGroups() { verifyIndex(); return getData()->solveGroups[calculateIndex()]; }
      AGX_FORCE_INLINE agxData::Array< agx::Physics::SolveGroupPtr > const& ContactZonePtr::solveGroups() const { verifyIndex(); return getData()->solveGroups[calculateIndex()]; }

      AGX_FORCE_INLINE agx::Physics::HierarchicalGrid::CellPtr& ContactZonePtr::spatialCell() { verifyIndex(); return getData()->spatialCell[calculateIndex()]; }
      AGX_FORCE_INLINE agx::Physics::HierarchicalGrid::CellPtr const& ContactZonePtr::spatialCell() const { verifyIndex(); return getData()->spatialCell[calculateIndex()]; }

      AGX_FORCE_INLINE agx::Physics::HierarchicalGrid::GridTierPtr& ContactZonePtr::tier() { verifyIndex(); return getData()->tier[calculateIndex()]; }
      AGX_FORCE_INLINE agx::Physics::HierarchicalGrid::GridTierPtr const& ContactZonePtr::tier() const { verifyIndex(); return getData()->tier[calculateIndex()]; }

      AGX_FORCE_INLINE agx::Physics::HierarchicalGrid::ContactZoneDependencyPtr& ContactZonePtr::interTierDependencyList() { verifyIndex(); return getData()->interTierDependencyList[calculateIndex()]; }
      AGX_FORCE_INLINE agx::Physics::HierarchicalGrid::ContactZoneDependencyPtr const& ContactZonePtr::interTierDependencyList() const { verifyIndex(); return getData()->interTierDependencyList[calculateIndex()]; }

      AGX_FORCE_INLINE agx::UInt8& ContactZonePtr::type() { verifyIndex(); return getData()->type[calculateIndex()]; }
      AGX_FORCE_INLINE agx::UInt8 const& ContactZonePtr::type() const { verifyIndex(); return getData()->type[calculateIndex()]; }

      AGX_FORCE_INLINE agx::Vec3i& ContactZonePtr::id() { verifyIndex(); return getData()->id[calculateIndex()]; }
      AGX_FORCE_INLINE agx::Vec3i const& ContactZonePtr::id() const { verifyIndex(); return getData()->id[calculateIndex()]; }

      AGX_FORCE_INLINE agx::Job*& ContactZonePtr::job() { verifyIndex(); return getData()->job[calculateIndex()]; }
      AGX_FORCE_INLINE agx::Job* const& ContactZonePtr::job() const { verifyIndex(); return getData()->job[calculateIndex()]; }

      AGX_FORCE_INLINE agx::Vec4f& ContactZonePtr::color() { verifyIndex(); return getData()->color[calculateIndex()]; }
      AGX_FORCE_INLINE agx::Vec4f const& ContactZonePtr::color() const { verifyIndex(); return getData()->color[calculateIndex()]; }

      //-----------------------------------------------------------------------------------------------------
      AGX_FORCE_INLINE ContactZoneInstance::ContactZoneInstance() {}
      AGX_FORCE_INLINE ContactZoneInstance::ContactZoneInstance(ContactZoneData* data, agx::Index index) : agxData::EntityInstance(data, index) {}
      AGX_FORCE_INLINE ContactZoneInstance::ContactZoneInstance(agxData::EntityStorage* storage, agx::Index index) : agxData::EntityInstance(storage, index) {}
      AGX_FORCE_INLINE ContactZoneInstance::ContactZoneInstance(const agxData::EntityInstance& other) : agxData::EntityInstance(other)
      {
        agxAssertN(!other || other.isInstanceOf(ContactZoneModel::instance()),
          "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
          EntityInstance::getModel()->fullPath().c_str(), ContactZoneModel::instance()->fullPath().c_str());
      }

      AGX_FORCE_INLINE ContactZoneInstance::ContactZoneInstance(const agxData::EntityPtr& ptr) : agxData::EntityInstance(ptr)
      {
        agxAssertN(!ptr || ptr.isInstanceOf(ContactZoneModel::instance()),
          "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
          EntityInstance::getModel()->fullPath().c_str(), ContactZoneModel::instance()->fullPath().c_str());
      }


      AGX_FORCE_INLINE ContactZoneData* ContactZoneInstance::getData() { return static_cast<ContactZoneData* >(agxData::EntityInstance::getData()); }
      AGX_FORCE_INLINE const ContactZoneData* ContactZoneInstance::getData() const { return static_cast<const ContactZoneData* >(agxData::EntityInstance::getData()); }

      AGX_FORCE_INLINE agxData::Array< agx::UInt32 >& ContactZoneInstance::particleParticleContacts() { verifyIndex(); return getData()->particleParticleContacts[getIndex()]; }
      AGX_FORCE_INLINE agxData::Array< agx::UInt32 > const& ContactZoneInstance::particleParticleContacts() const { verifyIndex(); return getData()->particleParticleContacts[getIndex()]; }

      AGX_FORCE_INLINE agxData::Array< agx::UInt32 >& ContactZoneInstance::particleGeometryContacts() { verifyIndex(); return getData()->particleGeometryContacts[getIndex()]; }
      AGX_FORCE_INLINE agxData::Array< agx::UInt32 > const& ContactZoneInstance::particleGeometryContacts() const { verifyIndex(); return getData()->particleGeometryContacts[getIndex()]; }

      AGX_FORCE_INLINE agxData::Array< agx::UInt32 >& ContactZoneInstance::geometryGeometryContacts() { verifyIndex(); return getData()->geometryGeometryContacts[getIndex()]; }
      AGX_FORCE_INLINE agxData::Array< agx::UInt32 > const& ContactZoneInstance::geometryGeometryContacts() const { verifyIndex(); return getData()->geometryGeometryContacts[getIndex()]; }

      AGX_FORCE_INLINE agx::UInt32& ContactZoneInstance::totNumContacts() { verifyIndex(); return getData()->totNumContacts[getIndex()]; }
      AGX_FORCE_INLINE agx::UInt32 const& ContactZoneInstance::totNumContacts() const { verifyIndex(); return getData()->totNumContacts[getIndex()]; }

      AGX_FORCE_INLINE agx::AtomicValue& ContactZoneInstance::numParticleParticleContacts() { verifyIndex(); return getData()->numParticleParticleContacts[getIndex()]; }
      AGX_FORCE_INLINE agx::AtomicValue const& ContactZoneInstance::numParticleParticleContacts() const { verifyIndex(); return getData()->numParticleParticleContacts[getIndex()]; }

      AGX_FORCE_INLINE agx::AtomicValue& ContactZoneInstance::numParticleGeometryContacts() { verifyIndex(); return getData()->numParticleGeometryContacts[getIndex()]; }
      AGX_FORCE_INLINE agx::AtomicValue const& ContactZoneInstance::numParticleGeometryContacts() const { verifyIndex(); return getData()->numParticleGeometryContacts[getIndex()]; }

      AGX_FORCE_INLINE agx::AtomicValue& ContactZoneInstance::numGeometryContacts() { verifyIndex(); return getData()->numGeometryContacts[getIndex()]; }
      AGX_FORCE_INLINE agx::AtomicValue const& ContactZoneInstance::numGeometryContacts() const { verifyIndex(); return getData()->numGeometryContacts[getIndex()]; }

      AGX_FORCE_INLINE agxData::Array< agx::Physics::SolveGroupPtr >& ContactZoneInstance::solveGroups() { verifyIndex(); return getData()->solveGroups[getIndex()]; }
      AGX_FORCE_INLINE agxData::Array< agx::Physics::SolveGroupPtr > const& ContactZoneInstance::solveGroups() const { verifyIndex(); return getData()->solveGroups[getIndex()]; }

      AGX_FORCE_INLINE agx::Physics::HierarchicalGrid::CellPtr& ContactZoneInstance::spatialCell() { verifyIndex(); return getData()->spatialCell[getIndex()]; }
      AGX_FORCE_INLINE agx::Physics::HierarchicalGrid::CellPtr const& ContactZoneInstance::spatialCell() const { verifyIndex(); return getData()->spatialCell[getIndex()]; }

      AGX_FORCE_INLINE agx::Physics::HierarchicalGrid::GridTierPtr& ContactZoneInstance::tier() { verifyIndex(); return getData()->tier[getIndex()]; }
      AGX_FORCE_INLINE agx::Physics::HierarchicalGrid::GridTierPtr const& ContactZoneInstance::tier() const { verifyIndex(); return getData()->tier[getIndex()]; }

      AGX_FORCE_INLINE agx::Physics::HierarchicalGrid::ContactZoneDependencyPtr& ContactZoneInstance::interTierDependencyList() { verifyIndex(); return getData()->interTierDependencyList[getIndex()]; }
      AGX_FORCE_INLINE agx::Physics::HierarchicalGrid::ContactZoneDependencyPtr const& ContactZoneInstance::interTierDependencyList() const { verifyIndex(); return getData()->interTierDependencyList[getIndex()]; }

      AGX_FORCE_INLINE agx::UInt8& ContactZoneInstance::type() { verifyIndex(); return getData()->type[getIndex()]; }
      AGX_FORCE_INLINE agx::UInt8 const& ContactZoneInstance::type() const { verifyIndex(); return getData()->type[getIndex()]; }

      AGX_FORCE_INLINE agx::Vec3i& ContactZoneInstance::id() { verifyIndex(); return getData()->id[getIndex()]; }
      AGX_FORCE_INLINE agx::Vec3i const& ContactZoneInstance::id() const { verifyIndex(); return getData()->id[getIndex()]; }

      AGX_FORCE_INLINE agx::Job*& ContactZoneInstance::job() { verifyIndex(); return getData()->job[getIndex()]; }
      AGX_FORCE_INLINE agx::Job* const& ContactZoneInstance::job() const { verifyIndex(); return getData()->job[getIndex()]; }

      AGX_FORCE_INLINE agx::Vec4f& ContactZoneInstance::color() { verifyIndex(); return getData()->color[getIndex()]; }
      AGX_FORCE_INLINE agx::Vec4f const& ContactZoneInstance::color() const { verifyIndex(); return getData()->color[getIndex()]; }

      //-----------------------------------------------------------------------------------------------------
      AGX_FORCE_INLINE ContactZoneSemantics::ContactZoneSemantics() {}
      //-----------------------------------------------------------------------------------------------------
      DOXYGEN_END_INTERNAL_BLOCK()
    }
  }
}

AGX_TYPE_BINDING(agx::Physics::HierarchicalGrid::ContactZonePtr, "Physics.HierarchicalGrid.ContactZonePtr")
AGX_TYPE_BINDING(agx::Physics::HierarchicalGrid::ContactZoneInstance, "Physics.HierarchicalGrid.ContactZoneInstance")

#ifdef _MSC_VER
# pragma warning(pop)
#endif

#undef AGX_ENTITY_WRAPPER
#undef AGX_ENTITY_NAMESPACE
#endif

