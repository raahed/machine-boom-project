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

#ifndef GENERATED_AGX_PHYSICS_GRANULARBODY_CACHEDGRANULARCONTACT_H_PLUGIN
#define GENERATED_AGX_PHYSICS_GRANULARBODY_CACHEDGRANULARCONTACT_H_PLUGIN

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
#include <agx/Physics/GranularBody/CachedContact.h>
#include <agx/Vec3.h>


namespace agx
{
  namespace Physics
  {
    namespace GranularBody
    {

      class CachedGranularContactModel;
      class CachedGranularContactData;
      class CachedGranularContactPtr;
      class CachedGranularContactInstance;
      class CachedGranularContactSemantics;


      AGX_DECLARE_POINTER_TYPES(CachedGranularContactModel);

      /** 
      Abstract description of the data attributes for the Physics.GranularBody.CachedGranularContact entity.
      */ 
      class AGXPHYSICS_EXPORT CachedGranularContactModel : public agxData::EntityModel
      {
      public:
        typedef CachedGranularContactPtr PtrT;

        CachedGranularContactModel(const agx::String& name = "CachedGranularContact");

        /// \return The entity model singleton.
        static CachedGranularContactModel* instance();

        /// Create and return a pointer to a new instance in the default storage for this entity model.
        static CachedGranularContactPtr createInstance();

        /// \return The default storage for this entity model.
        static agxData::EntityStorage* defaultStorage();

        /// This is part of internal cleanup and should not be called by users
        virtual void shutdownCleanup() override;



        /* Attributes */
        static agxData::ScalarAttributeT< agx::Bool >* keepAttribute;
        static agxData::ScalarAttributeT< agx::Bool >* duplicateAttribute;
        static agxData::ScalarAttributeT< agx::Physics::GranularBody::ContactLambda >* contactLambdasAttribute;
        static agxData::ScalarAttributeT< agx::Vec3f >* contactNormalAttribute;
        static agxData::ScalarAttributeT< agx::Bool >* shouldWarmStartAttribute;

      protected:
        virtual ~CachedGranularContactModel();
        virtual agxData::EntityData* createData(agxData::EntityStorage* storage) override;
        virtual void configure(agx::TiXmlElement* eEntity) override;
        virtual void initAttributeAccessors() override;
        virtual void construct(agxData::EntityPtr instance) override;
        void construct(agx::Physics::GranularBody::CachedGranularContactPtr cachedGranularContact);
      };


      DOXYGEN_START_INTERNAL_BLOCK()
      #ifndef AGX_PHYSICS_GRANULARBODY_CACHEDGRANULARCONTACT_DATA_SET_OVERRIDE
      #define AGX_PHYSICS_GRANULARBODY_CACHEDGRANULARCONTACT_DATA_SET
      class AGXPHYSICS_EXPORT CachedGranularContactData : public agxData::EntityData
      {
      public:
        CachedGranularContactInstance operator[] (size_t index);

      public:
        agxData::Array< CachedGranularContactPtr >& instance;
        agxData::Array< agx::Bool > keep;
        agxData::Array< agx::Bool > duplicate;
        agxData::Array< agx::Physics::GranularBody::ContactLambda > contactLambdas;
        agxData::Array< agx::Vec3f > contactNormal;
        agxData::Array< agx::Bool > shouldWarmStart;

      public:
        typedef agx::Bool keepType;
        typedef agx::Bool duplicateType;
        typedef agx::Physics::GranularBody::ContactLambda contactLambdasType;
        typedef agx::Vec3f contactNormalType;
        typedef agx::Bool shouldWarmStartType;

      public:
        CachedGranularContactData(agxData::EntityStorage* storage);
        CachedGranularContactData();

      protected:
        virtual ~CachedGranularContactData() {}
        virtual void setNumElements(agx::Index numElements) override;

      private:
        CachedGranularContactData& operator= (const CachedGranularContactData&) { return *this; }

      };
      #endif
      DOXYGEN_END_INTERNAL_BLOCK()


      DOXYGEN_START_INTERNAL_BLOCK()
      class AGXPHYSICS_EXPORT CachedGranularContactSemantics : protected agxData::EntityPtr
      {
      public:

        // Automatic getters
        agx::Bool const& getKeep() const;
        agx::Bool const& getDuplicate() const;
        agx::Physics::GranularBody::ContactLambda const& getContactLambdas() const;
        agx::Vec3f const& getContactNormal() const;
        agx::Bool const& getShouldWarmStart() const;

        // Semantics defined by explicit kernels

        // Automatic setters
        void setKeep(agx::Bool const& value);
        void setDuplicate(agx::Bool const& value);
        void setContactLambdas(agx::Physics::GranularBody::ContactLambda const& value);
        void setContactNormal(agx::Vec3f const& value);
        void setShouldWarmStart(agx::Bool const& value);


      protected:
        friend class CachedGranularContactPtr;
        friend class CachedGranularContactInstance;
        CachedGranularContactSemantics();
      };
      DOXYGEN_END_INTERNAL_BLOCK()


      /**
      Pointer to a entity instance of type Physics.GranularBody.CachedGranularContact
      */
      class CALLABLE CachedGranularContactPtr : public agxData::EntityPtr
      {
      public:
        typedef CachedGranularContactModel ModelType;
        typedef CachedGranularContactData DataType;
        typedef CachedGranularContactInstance InstanceType;

      public:
        AGXPHYSICS_EXPORT CachedGranularContactPtr();
        AGXPHYSICS_EXPORT CachedGranularContactPtr(agxData::EntityStorage* storage, agx::Index id);
        AGXPHYSICS_EXPORT CachedGranularContactPtr(const agxData::EntityPtr& ptr);
        AGXPHYSICS_EXPORT CachedGranularContactPtr(const agxData::EntityInstance& instance);
        AGXPHYSICS_EXPORT CachedGranularContactPtr& operator= (const agxData::EntityPtr& ptr);
        AGXPHYSICS_EXPORT CachedGranularContactPtr& operator= (const agxData::EntityInstance& instance);
        AGXPHYSICS_EXPORT CachedGranularContactInstance instance();
        AGXPHYSICS_EXPORT const CachedGranularContactInstance instance() const;

        AGXPHYSICS_EXPORT CachedGranularContactSemantics* operator->();
        AGXPHYSICS_EXPORT const CachedGranularContactSemantics* operator->() const;

        CachedGranularContactData* getData();
        const CachedGranularContactData* getData() const;


        /// \return reference to the keep attribute
        AGXPHYSICS_EXPORT agx::Bool& keep();
        /// \return const reference to the keep attribute
        AGXPHYSICS_EXPORT agx::Bool const& keep() const;

        /// \return reference to the duplicate attribute
        AGXPHYSICS_EXPORT agx::Bool& duplicate();
        /// \return const reference to the duplicate attribute
        AGXPHYSICS_EXPORT agx::Bool const& duplicate() const;

        /// \return reference to the contactLambdas attribute
        AGXPHYSICS_EXPORT agx::Physics::GranularBody::ContactLambda& contactLambdas();
        /// \return const reference to the contactLambdas attribute
        AGXPHYSICS_EXPORT agx::Physics::GranularBody::ContactLambda const& contactLambdas() const;

        /// \return reference to the contactNormal attribute
        AGXPHYSICS_EXPORT agx::Vec3f& contactNormal();
        /// \return const reference to the contactNormal attribute
        AGXPHYSICS_EXPORT agx::Vec3f const& contactNormal() const;

        /// \return reference to the shouldWarmStart attribute
        AGXPHYSICS_EXPORT agx::Bool& shouldWarmStart();
        /// \return const reference to the shouldWarmStart attribute
        AGXPHYSICS_EXPORT agx::Bool const& shouldWarmStart() const;

        DOXYGEN_START_INTERNAL_BLOCK()
        // Only used by tolua!
        #ifdef AGX_GENERATED_TOLUA_WRAPPER
        inline agx::Bool const& getKeep() const { return (*this)->getKeep(); }
        inline agx::Bool const& getDuplicate() const { return (*this)->getDuplicate(); }
        inline agx::Physics::GranularBody::ContactLambda const& getContactLambdas() const { return (*this)->getContactLambdas(); }
        inline agx::Vec3f const& getContactNormal() const { return (*this)->getContactNormal(); }
        inline agx::Bool const& getShouldWarmStart() const { return (*this)->getShouldWarmStart(); }


        inline void setKeep(agx::Bool const& value) { (*this)->setKeep(value); }
        inline void setDuplicate(agx::Bool const& value) { (*this)->setDuplicate(value); }
        inline void setContactLambdas(agx::Physics::GranularBody::ContactLambda const& value) { (*this)->setContactLambdas(value); }
        inline void setContactNormal(agx::Vec3f const& value) { (*this)->setContactNormal(value); }
        inline void setShouldWarmStart(agx::Bool const& value) { (*this)->setShouldWarmStart(value); }
        #endif
        DOXYGEN_END_INTERNAL_BLOCK()

      };


      DOXYGEN_START_INTERNAL_BLOCK()
      class AGXPHYSICS_EXPORT CachedGranularContactInstance : public agxData::EntityInstance
      {
      public:
        CachedGranularContactInstance();
        CachedGranularContactInstance(CachedGranularContactData* data, agx::Index index);
        CachedGranularContactInstance(agxData::EntityStorage *storage, agx::Index index);
        CachedGranularContactInstance(const agxData::EntityInstance& other);
        CachedGranularContactInstance(const agxData::EntityPtr& ptr);

        CachedGranularContactData* getData();
        const CachedGranularContactData* getData() const;

      public:
        /// \return reference to the keep attribute
        agx::Bool& keep();
        /// \return const reference to the keep attribute
        agx::Bool const& keep() const;

        /// \return reference to the duplicate attribute
        agx::Bool& duplicate();
        /// \return const reference to the duplicate attribute
        agx::Bool const& duplicate() const;

        /// \return reference to the contactLambdas attribute
        agx::Physics::GranularBody::ContactLambda& contactLambdas();
        /// \return const reference to the contactLambdas attribute
        agx::Physics::GranularBody::ContactLambda const& contactLambdas() const;

        /// \return reference to the contactNormal attribute
        agx::Vec3f& contactNormal();
        /// \return const reference to the contactNormal attribute
        agx::Vec3f const& contactNormal() const;

        /// \return reference to the shouldWarmStart attribute
        agx::Bool& shouldWarmStart();
        /// \return const reference to the shouldWarmStart attribute
        agx::Bool const& shouldWarmStart() const;

      };
      DOXYGEN_END_INTERNAL_BLOCK()



      typedef agx::VectorPOD<CachedGranularContactPtr> CachedGranularContactPtrVector;
      typedef agxData::Array<CachedGranularContactPtr> CachedGranularContactPtrArray;



      DOXYGEN_START_INTERNAL_BLOCK()
      /* Implementation */
      //-----------------------------------------------------------------------------------------------------
      //-----------------------------------------------------------------------------------------------------
      inline CachedGranularContactInstance agx::Physics::GranularBody::CachedGranularContactData::operator[] (size_t index) { return CachedGranularContactInstance(this, (agx::Index)index); }
      //-----------------------------------------------------------------------------------------------------
      AGX_FORCE_INLINE CachedGranularContactPtr::CachedGranularContactPtr() {}
      AGX_FORCE_INLINE CachedGranularContactPtr::CachedGranularContactPtr(agxData::EntityStorage* storage, agx::Index id) : agxData::EntityPtr(storage, id) {}
      AGX_FORCE_INLINE CachedGranularContactPtr::CachedGranularContactPtr(const agxData::EntityPtr& ptr) : agxData::EntityPtr(ptr)
      {
        agxAssertN(!ptr || ptr.isInstanceOf(CachedGranularContactModel::instance()),
          "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
          EntityPtr::getModel()->fullPath().c_str(), CachedGranularContactModel::instance()->fullPath().c_str());
      }

      AGX_FORCE_INLINE CachedGranularContactPtr::CachedGranularContactPtr(const agxData::EntityInstance& instance) : agxData::EntityPtr(instance)
      {
        agxAssertN(!instance || instance.isInstanceOf(CachedGranularContactModel::instance()),
          "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
          EntityPtr::getModel()->fullPath().c_str(), CachedGranularContactModel::instance()->fullPath().c_str());
      }

      AGX_FORCE_INLINE CachedGranularContactPtr& CachedGranularContactPtr::operator= (const agxData::EntityPtr& ptr)
      {
        agxData::EntityPtr::operator= (ptr);
        agxAssertN(!ptr || ptr.isInstanceOf(CachedGranularContactModel::instance()),
          "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
          EntityPtr::getModel()->fullPath().c_str(), CachedGranularContactModel::instance()->fullPath().c_str());
        return *this;
      }

      AGX_FORCE_INLINE CachedGranularContactPtr& CachedGranularContactPtr::operator= (const agxData::EntityInstance& instance)
      {
        agxData::EntityPtr::operator= (instance);
        agxAssertN(!instance || instance.isInstanceOf(CachedGranularContactModel::instance()),
          "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
          EntityPtr::getModel()->fullPath().c_str(), CachedGranularContactModel::instance()->fullPath().c_str());
        return *this;
      }

      inline CachedGranularContactInstance CachedGranularContactPtr::instance() { return agxData::EntityPtr::instance(); }
      inline const CachedGranularContactInstance CachedGranularContactPtr::instance() const { return agxData::EntityPtr::instance(); }
      AGX_FORCE_INLINE CachedGranularContactSemantics* CachedGranularContactPtr::operator->() { return (CachedGranularContactSemantics* )this; }
      AGX_FORCE_INLINE const CachedGranularContactSemantics* CachedGranularContactPtr::operator->() const { return (const CachedGranularContactSemantics* )this; }
      AGX_FORCE_INLINE CachedGranularContactData* CachedGranularContactPtr::getData() { return static_cast<CachedGranularContactData* >(agxData::EntityPtr::getData()); }
      AGX_FORCE_INLINE const CachedGranularContactData* CachedGranularContactPtr::getData() const { return static_cast<const CachedGranularContactData* >(agxData::EntityPtr::getData()); }

      AGX_FORCE_INLINE agx::Bool& CachedGranularContactPtr::keep() { verifyIndex(); return getData()->keep[calculateIndex()]; }
      AGX_FORCE_INLINE agx::Bool const& CachedGranularContactPtr::keep() const { verifyIndex(); return getData()->keep[calculateIndex()]; }

      AGX_FORCE_INLINE agx::Bool& CachedGranularContactPtr::duplicate() { verifyIndex(); return getData()->duplicate[calculateIndex()]; }
      AGX_FORCE_INLINE agx::Bool const& CachedGranularContactPtr::duplicate() const { verifyIndex(); return getData()->duplicate[calculateIndex()]; }

      AGX_FORCE_INLINE agx::Physics::GranularBody::ContactLambda& CachedGranularContactPtr::contactLambdas() { verifyIndex(); return getData()->contactLambdas[calculateIndex()]; }
      AGX_FORCE_INLINE agx::Physics::GranularBody::ContactLambda const& CachedGranularContactPtr::contactLambdas() const { verifyIndex(); return getData()->contactLambdas[calculateIndex()]; }

      AGX_FORCE_INLINE agx::Vec3f& CachedGranularContactPtr::contactNormal() { verifyIndex(); return getData()->contactNormal[calculateIndex()]; }
      AGX_FORCE_INLINE agx::Vec3f const& CachedGranularContactPtr::contactNormal() const { verifyIndex(); return getData()->contactNormal[calculateIndex()]; }

      AGX_FORCE_INLINE agx::Bool& CachedGranularContactPtr::shouldWarmStart() { verifyIndex(); return getData()->shouldWarmStart[calculateIndex()]; }
      AGX_FORCE_INLINE agx::Bool const& CachedGranularContactPtr::shouldWarmStart() const { verifyIndex(); return getData()->shouldWarmStart[calculateIndex()]; }

      //-----------------------------------------------------------------------------------------------------
      AGX_FORCE_INLINE CachedGranularContactInstance::CachedGranularContactInstance() {}
      AGX_FORCE_INLINE CachedGranularContactInstance::CachedGranularContactInstance(CachedGranularContactData* data, agx::Index index) : agxData::EntityInstance(data, index) {}
      AGX_FORCE_INLINE CachedGranularContactInstance::CachedGranularContactInstance(agxData::EntityStorage* storage, agx::Index index) : agxData::EntityInstance(storage, index) {}
      AGX_FORCE_INLINE CachedGranularContactInstance::CachedGranularContactInstance(const agxData::EntityInstance& other) : agxData::EntityInstance(other)
      {
        agxAssertN(!other || other.isInstanceOf(CachedGranularContactModel::instance()),
          "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
          EntityInstance::getModel()->fullPath().c_str(), CachedGranularContactModel::instance()->fullPath().c_str());
      }

      AGX_FORCE_INLINE CachedGranularContactInstance::CachedGranularContactInstance(const agxData::EntityPtr& ptr) : agxData::EntityInstance(ptr)
      {
        agxAssertN(!ptr || ptr.isInstanceOf(CachedGranularContactModel::instance()),
          "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
          EntityInstance::getModel()->fullPath().c_str(), CachedGranularContactModel::instance()->fullPath().c_str());
      }


      AGX_FORCE_INLINE CachedGranularContactData* CachedGranularContactInstance::getData() { return static_cast<CachedGranularContactData* >(agxData::EntityInstance::getData()); }
      AGX_FORCE_INLINE const CachedGranularContactData* CachedGranularContactInstance::getData() const { return static_cast<const CachedGranularContactData* >(agxData::EntityInstance::getData()); }

      AGX_FORCE_INLINE agx::Bool& CachedGranularContactInstance::keep() { verifyIndex(); return getData()->keep[getIndex()]; }
      AGX_FORCE_INLINE agx::Bool const& CachedGranularContactInstance::keep() const { verifyIndex(); return getData()->keep[getIndex()]; }

      AGX_FORCE_INLINE agx::Bool& CachedGranularContactInstance::duplicate() { verifyIndex(); return getData()->duplicate[getIndex()]; }
      AGX_FORCE_INLINE agx::Bool const& CachedGranularContactInstance::duplicate() const { verifyIndex(); return getData()->duplicate[getIndex()]; }

      AGX_FORCE_INLINE agx::Physics::GranularBody::ContactLambda& CachedGranularContactInstance::contactLambdas() { verifyIndex(); return getData()->contactLambdas[getIndex()]; }
      AGX_FORCE_INLINE agx::Physics::GranularBody::ContactLambda const& CachedGranularContactInstance::contactLambdas() const { verifyIndex(); return getData()->contactLambdas[getIndex()]; }

      AGX_FORCE_INLINE agx::Vec3f& CachedGranularContactInstance::contactNormal() { verifyIndex(); return getData()->contactNormal[getIndex()]; }
      AGX_FORCE_INLINE agx::Vec3f const& CachedGranularContactInstance::contactNormal() const { verifyIndex(); return getData()->contactNormal[getIndex()]; }

      AGX_FORCE_INLINE agx::Bool& CachedGranularContactInstance::shouldWarmStart() { verifyIndex(); return getData()->shouldWarmStart[getIndex()]; }
      AGX_FORCE_INLINE agx::Bool const& CachedGranularContactInstance::shouldWarmStart() const { verifyIndex(); return getData()->shouldWarmStart[getIndex()]; }

      //-----------------------------------------------------------------------------------------------------
      AGX_FORCE_INLINE CachedGranularContactSemantics::CachedGranularContactSemantics() {}
      //-----------------------------------------------------------------------------------------------------
      DOXYGEN_END_INTERNAL_BLOCK()
    }
  }
}

AGX_TYPE_BINDING(agx::Physics::GranularBody::CachedGranularContactPtr, "Physics.GranularBody.CachedGranularContactPtr")
AGX_TYPE_BINDING(agx::Physics::GranularBody::CachedGranularContactInstance, "Physics.GranularBody.CachedGranularContactInstance")

#ifdef _MSC_VER
# pragma warning(pop)
#endif

#undef AGX_ENTITY_WRAPPER
#undef AGX_ENTITY_NAMESPACE
#endif

