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

#ifndef GENERATED_AGX_DEMO_DEMO3_PARTICLE6DEMO_H_PLUGIN
#define GENERATED_AGX_DEMO_DEMO3_PARTICLE6DEMO_H_PLUGIN

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
#include <agx/Physics/ParticleEntity.h>
#include <agx/Matrix3x3.h>
#include <agx/Vec3.h>


namespace agx
{
  namespace Demo
  {
    namespace Demo3
    {

      class Particle6DemoModel;
      class Particle6DemoData;
      class Particle6DemoPtr;
      class Particle6DemoInstance;
      class Particle6DemoSemantics;


      AGX_DECLARE_POINTER_TYPES(Particle6DemoModel);

      /** 
      Abstract description of the data attributes for the Demo.Demo3.Particle6Demo entity.
      */ 
      class AGXPHYSICS_EXPORT Particle6DemoModel : public agx::Physics::ParticleModel
      {
      public:
        typedef Particle6DemoPtr PtrT;

        Particle6DemoModel(const agx::String& name = "Particle6Demo");

        /// \return The entity model singleton.
        static Particle6DemoModel* instance();

        /// Create and return a pointer to a new instance in the default storage for this entity model.
        static Particle6DemoPtr createInstance();

        /// \return The default storage for this entity model.
        static agxData::EntityStorage* defaultStorage();

        /// This is part of internal cleanup and should not be called by users
        virtual void shutdownCleanup() override;



        /* Attributes */
        static agxData::ScalarAttributeT< agx::Matrix3x3 >* rotationAttribute;
        static agxData::ScalarAttributeT< agx::Vec3 >* torqueAttribute;

      protected:
        virtual ~Particle6DemoModel();
        virtual agxData::EntityData* createData(agxData::EntityStorage* storage) override;
        virtual void configure(agx::TiXmlElement* eEntity) override;
        virtual void initAttributeAccessors() override;
        virtual void construct(agxData::EntityPtr instance) override;
        void construct(agx::Demo::Demo3::Particle6DemoPtr particle6Demo);
      };


      DOXYGEN_START_INTERNAL_BLOCK()
      #ifndef AGX_DEMO_DEMO3_PARTICLE6DEMO_DATA_SET_OVERRIDE
      #define AGX_DEMO_DEMO3_PARTICLE6DEMO_DATA_SET
      class AGXPHYSICS_EXPORT Particle6DemoData : public agx::Physics::ParticleData
      {
      public:
        Particle6DemoInstance operator[] (size_t index);

      public:
        agxData::Array< Particle6DemoPtr >& instance;
        agxData::Array< agx::Matrix3x3 > rotation;
        agxData::Array< agx::Vec3 > torque;

      public:
        typedef agx::Matrix3x3 rotationType;
        typedef agx::Vec3 torqueType;

      public:
        Particle6DemoData(agxData::EntityStorage* storage);
        Particle6DemoData();

      protected:
        virtual ~Particle6DemoData() {}
        virtual void setNumElements(agx::Index numElements) override;

      private:
        Particle6DemoData& operator= (const Particle6DemoData&) { return *this; }

      };
      #endif
      DOXYGEN_END_INTERNAL_BLOCK()


      DOXYGEN_START_INTERNAL_BLOCK()
      class AGXPHYSICS_EXPORT Particle6DemoSemantics : public agx::Physics::ParticleSemantics
      {
      public:

        // Automatic getters
        agx::Matrix3x3 const& getRotation() const;
        agx::Vec3 const& getTorque() const;

        // Semantics defined by explicit kernels

        // Automatic setters
        void setRotation(agx::Matrix3x3 const& value);
        void setTorque(agx::Vec3 const& value);


      protected:
        friend class Particle6DemoPtr;
        friend class Particle6DemoInstance;
        Particle6DemoSemantics();
      };
      DOXYGEN_END_INTERNAL_BLOCK()


      /**
      Pointer to a entity instance of type Demo.Demo3.Particle6Demo
      */
      class CALLABLE Particle6DemoPtr : public agx::Physics::ParticlePtr
      {
      public:
        typedef Particle6DemoModel ModelType;
        typedef Particle6DemoData DataType;
        typedef Particle6DemoInstance InstanceType;

      public:
        AGXPHYSICS_EXPORT Particle6DemoPtr();
        AGXPHYSICS_EXPORT Particle6DemoPtr(agxData::EntityStorage* storage, agx::Index id);
        AGXPHYSICS_EXPORT Particle6DemoPtr(const agxData::EntityPtr& ptr);
        AGXPHYSICS_EXPORT Particle6DemoPtr(const agxData::EntityInstance& instance);
        AGXPHYSICS_EXPORT Particle6DemoPtr& operator= (const agxData::EntityPtr& ptr);
        AGXPHYSICS_EXPORT Particle6DemoPtr& operator= (const agxData::EntityInstance& instance);
        AGXPHYSICS_EXPORT Particle6DemoInstance instance();
        AGXPHYSICS_EXPORT const Particle6DemoInstance instance() const;

        AGXPHYSICS_EXPORT Particle6DemoSemantics* operator->();
        AGXPHYSICS_EXPORT const Particle6DemoSemantics* operator->() const;

        Particle6DemoData* getData();
        const Particle6DemoData* getData() const;


        /// \return reference to the rotation attribute
        AGXPHYSICS_EXPORT agx::Matrix3x3& rotation();
        /// \return const reference to the rotation attribute
        AGXPHYSICS_EXPORT agx::Matrix3x3 const& rotation() const;

        /// \return reference to the torque attribute
        AGXPHYSICS_EXPORT agx::Vec3& torque();
        /// \return const reference to the torque attribute
        AGXPHYSICS_EXPORT agx::Vec3 const& torque() const;

        DOXYGEN_START_INTERNAL_BLOCK()
        // Only used by tolua!
        #ifdef AGX_GENERATED_TOLUA_WRAPPER
        inline agx::Matrix3x3 const& getRotation() const { return (*this)->getRotation(); }
        inline agx::Vec3 const& getTorque() const { return (*this)->getTorque(); }


        inline void setRotation(agx::Matrix3x3 const& value) { (*this)->setRotation(value); }
        inline void setTorque(agx::Vec3 const& value) { (*this)->setTorque(value); }
        #endif
        DOXYGEN_END_INTERNAL_BLOCK()

      };


      DOXYGEN_START_INTERNAL_BLOCK()
      class AGXPHYSICS_EXPORT Particle6DemoInstance : public agx::Physics::ParticleInstance
      {
      public:
        Particle6DemoInstance();
        Particle6DemoInstance(Particle6DemoData* data, agx::Index index);
        Particle6DemoInstance(agxData::EntityStorage *storage, agx::Index index);
        Particle6DemoInstance(const agxData::EntityInstance& other);
        Particle6DemoInstance(const agxData::EntityPtr& ptr);

        Particle6DemoData* getData();
        const Particle6DemoData* getData() const;

      public:
        /// \return reference to the rotation attribute
        agx::Matrix3x3& rotation();
        /// \return const reference to the rotation attribute
        agx::Matrix3x3 const& rotation() const;

        /// \return reference to the torque attribute
        agx::Vec3& torque();
        /// \return const reference to the torque attribute
        agx::Vec3 const& torque() const;

      };
      DOXYGEN_END_INTERNAL_BLOCK()



      typedef agx::VectorPOD<Particle6DemoPtr> Particle6DemoPtrVector;
      typedef agxData::Array<Particle6DemoPtr> Particle6DemoPtrArray;



      DOXYGEN_START_INTERNAL_BLOCK()
      /* Implementation */
      //-----------------------------------------------------------------------------------------------------
      //-----------------------------------------------------------------------------------------------------
      inline Particle6DemoInstance agx::Demo::Demo3::Particle6DemoData::operator[] (size_t index) { return Particle6DemoInstance(this, (agx::Index)index); }
      //-----------------------------------------------------------------------------------------------------
      AGX_FORCE_INLINE Particle6DemoPtr::Particle6DemoPtr() {}
      AGX_FORCE_INLINE Particle6DemoPtr::Particle6DemoPtr(agxData::EntityStorage* storage, agx::Index id) : agx::Physics::ParticlePtr(storage, id) {}
      AGX_FORCE_INLINE Particle6DemoPtr::Particle6DemoPtr(const agxData::EntityPtr& ptr) : agx::Physics::ParticlePtr(ptr)
      {
        agxAssertN(!ptr || ptr.isInstanceOf(Particle6DemoModel::instance()),
          "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
          EntityPtr::getModel()->fullPath().c_str(), Particle6DemoModel::instance()->fullPath().c_str());
      }

      AGX_FORCE_INLINE Particle6DemoPtr::Particle6DemoPtr(const agxData::EntityInstance& instance) : agx::Physics::ParticlePtr(instance)
      {
        agxAssertN(!instance || instance.isInstanceOf(Particle6DemoModel::instance()),
          "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
          EntityPtr::getModel()->fullPath().c_str(), Particle6DemoModel::instance()->fullPath().c_str());
      }

      AGX_FORCE_INLINE Particle6DemoPtr& Particle6DemoPtr::operator= (const agxData::EntityPtr& ptr)
      {
        agxData::EntityPtr::operator= (ptr);
        agxAssertN(!ptr || ptr.isInstanceOf(Particle6DemoModel::instance()),
          "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
          EntityPtr::getModel()->fullPath().c_str(), Particle6DemoModel::instance()->fullPath().c_str());
        return *this;
      }

      AGX_FORCE_INLINE Particle6DemoPtr& Particle6DemoPtr::operator= (const agxData::EntityInstance& instance)
      {
        agxData::EntityPtr::operator= (instance);
        agxAssertN(!instance || instance.isInstanceOf(Particle6DemoModel::instance()),
          "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
          EntityPtr::getModel()->fullPath().c_str(), Particle6DemoModel::instance()->fullPath().c_str());
        return *this;
      }

      inline Particle6DemoInstance Particle6DemoPtr::instance() { return agxData::EntityPtr::instance(); }
      inline const Particle6DemoInstance Particle6DemoPtr::instance() const { return agxData::EntityPtr::instance(); }
      AGX_FORCE_INLINE Particle6DemoSemantics* Particle6DemoPtr::operator->() { return (Particle6DemoSemantics* )this; }
      AGX_FORCE_INLINE const Particle6DemoSemantics* Particle6DemoPtr::operator->() const { return (const Particle6DemoSemantics* )this; }
      AGX_FORCE_INLINE Particle6DemoData* Particle6DemoPtr::getData() { return static_cast<Particle6DemoData* >(agxData::EntityPtr::getData()); }
      AGX_FORCE_INLINE const Particle6DemoData* Particle6DemoPtr::getData() const { return static_cast<const Particle6DemoData* >(agxData::EntityPtr::getData()); }

      AGX_FORCE_INLINE agx::Matrix3x3& Particle6DemoPtr::rotation() { verifyIndex(); return getData()->rotation[calculateIndex()]; }
      AGX_FORCE_INLINE agx::Matrix3x3 const& Particle6DemoPtr::rotation() const { verifyIndex(); return getData()->rotation[calculateIndex()]; }

      AGX_FORCE_INLINE agx::Vec3& Particle6DemoPtr::torque() { verifyIndex(); return getData()->torque[calculateIndex()]; }
      AGX_FORCE_INLINE agx::Vec3 const& Particle6DemoPtr::torque() const { verifyIndex(); return getData()->torque[calculateIndex()]; }

      //-----------------------------------------------------------------------------------------------------
      AGX_FORCE_INLINE Particle6DemoInstance::Particle6DemoInstance() {}
      AGX_FORCE_INLINE Particle6DemoInstance::Particle6DemoInstance(Particle6DemoData* data, agx::Index index) : agx::Physics::ParticleInstance(data, index) {}
      AGX_FORCE_INLINE Particle6DemoInstance::Particle6DemoInstance(agxData::EntityStorage* storage, agx::Index index) : agx::Physics::ParticleInstance(storage, index) {}
      AGX_FORCE_INLINE Particle6DemoInstance::Particle6DemoInstance(const agxData::EntityInstance& other) : agx::Physics::ParticleInstance(other)
      {
        agxAssertN(!other || other.isInstanceOf(Particle6DemoModel::instance()),
          "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
          EntityInstance::getModel()->fullPath().c_str(), Particle6DemoModel::instance()->fullPath().c_str());
      }

      AGX_FORCE_INLINE Particle6DemoInstance::Particle6DemoInstance(const agxData::EntityPtr& ptr) : agx::Physics::ParticleInstance(ptr)
      {
        agxAssertN(!ptr || ptr.isInstanceOf(Particle6DemoModel::instance()),
          "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
          EntityInstance::getModel()->fullPath().c_str(), Particle6DemoModel::instance()->fullPath().c_str());
      }


      AGX_FORCE_INLINE Particle6DemoData* Particle6DemoInstance::getData() { return static_cast<Particle6DemoData* >(agxData::EntityInstance::getData()); }
      AGX_FORCE_INLINE const Particle6DemoData* Particle6DemoInstance::getData() const { return static_cast<const Particle6DemoData* >(agxData::EntityInstance::getData()); }

      AGX_FORCE_INLINE agx::Matrix3x3& Particle6DemoInstance::rotation() { verifyIndex(); return getData()->rotation[getIndex()]; }
      AGX_FORCE_INLINE agx::Matrix3x3 const& Particle6DemoInstance::rotation() const { verifyIndex(); return getData()->rotation[getIndex()]; }

      AGX_FORCE_INLINE agx::Vec3& Particle6DemoInstance::torque() { verifyIndex(); return getData()->torque[getIndex()]; }
      AGX_FORCE_INLINE agx::Vec3 const& Particle6DemoInstance::torque() const { verifyIndex(); return getData()->torque[getIndex()]; }

      //-----------------------------------------------------------------------------------------------------
      AGX_FORCE_INLINE Particle6DemoSemantics::Particle6DemoSemantics() {}
      //-----------------------------------------------------------------------------------------------------
      DOXYGEN_END_INTERNAL_BLOCK()
    }
  }
}

AGX_TYPE_BINDING(agx::Demo::Demo3::Particle6DemoPtr, "Demo.Demo3.Particle6DemoPtr")
AGX_TYPE_BINDING(agx::Demo::Demo3::Particle6DemoInstance, "Demo.Demo3.Particle6DemoInstance")

#ifdef _MSC_VER
# pragma warning(pop)
#endif

#undef AGX_ENTITY_WRAPPER
#undef AGX_ENTITY_NAMESPACE
#endif

