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

#ifndef GENERATED_AGX_PHYSICS_PARTICLE_H_PLUGIN
#define GENERATED_AGX_PHYSICS_PARTICLE_H_PLUGIN

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
#include <agx/ParticleState.h>
#include <agx/Real.h>
#include <agx/Physics/MaterialEntity.h>
#include <agx/Physics/CollisionGroupSetEntity.h>
#include <agx/Vec3.h>
#include <agx/Uuid.h>
#include <agx/Integer.h>
#include <agx/Physics/GraphNodeEntity.h>
#include <agx/Vec4.h>

namespace agx { namespace Physics { class MaterialPtr; }}
namespace agx { namespace Physics { class CollisionGroupSetPtr; }}
namespace agx { namespace Physics { class GraphNodePtr; }}

namespace agx
{
  namespace Physics
  {

    class ParticleModel;
    class ParticleData;
    class ParticlePtr;
    class ParticleInstance;
    class ParticleSemantics;


    AGX_DECLARE_POINTER_TYPES(ParticleModel);

    /** 
    Abstract description of the data attributes for the Physics.Particle entity.
    */ 
    class AGXPHYSICS_EXPORT ParticleModel : public agxData::EntityModel
    {
    public:
      typedef ParticlePtr PtrT;

      ParticleModel(const agx::String& name = "Particle");

      /// \return The entity model singleton.
      static ParticleModel* instance();

      /// Create and return a pointer to a new instance in the default storage for this entity model.
      static ParticlePtr createInstance();

      /// \return The default storage for this entity model.
      static agxData::EntityStorage* defaultStorage();

      /// This is part of internal cleanup and should not be called by users
      virtual void shutdownCleanup() override;



      /* Attributes */
      static agxData::ScalarAttributeT< agx::ParticleState >* stateAttribute;
      static agxData::SharedAttributeT< agx::Real >* defaultRadiusAttribute;
      static agxData::SharedAttributeT< agx::Real >* defaultMassAttribute;
      static agxData::SharedAttributeT< agx::Physics::MaterialPtr >* defaultMaterialAttribute;
      static agxData::SharedAttributeT< agx::Physics::CollisionGroupSetPtr >* defaultCollisionGroupSetAttribute;
      static agxData::SharedAttributeT< agx::Real >* defaultInvMassAttribute;
      static agxData::ScalarAttributeT< agx::Vec3 >* positionAttribute;
      static agxData::ScalarAttributeT< agx::Vec3 >* velocityAttribute;
      static agxData::ScalarAttributeT< agx::Vec3 >* oldVelocityAttribute;
      static agxData::ScalarAttributeT< agx::Vec3 >* forceAttribute;
      static agxData::ScalarAttributeT< agx::Real >* lifeAttribute;
      static agxData::ScalarAttributeT< agx::Real >* radiusAttribute;
      static agxData::ScalarAttributeT< agx::Real >* massAttribute;
      static agxData::ScalarAttributeT< agx::Real >* invMassAttribute;
      static agxData::ScalarAttributeT< agx::Physics::MaterialPtr >* materialAttribute;
      static agxData::ScalarAttributeT< agx::Uuid >* materialUuidAttribute;
      static agxData::ScalarAttributeT< agx::Physics::CollisionGroupSetPtr >* collisionGroupSetAttribute;
      static agxData::ScalarAttributeT< agx::UInt8 >* tierAttribute;
      static agxData::ScalarAttributeT< agx::UInt32 >* cellIndexAttribute;
      static agxData::ScalarAttributeT< agx::UInt16 >* cellSlotAttribute;
      static agxData::ScalarAttributeT< agx::UInt32 >* oldCellAttribute;
      static agxData::ScalarAttributeT< agx::UInt32 >* geometryContactListAttribute;
      static agxData::ScalarAttributeT< agx::UInt32 >* solveBodyIndexAttribute;
      static agxData::ScalarAttributeT< agx::Physics::GraphNodePtr >* graphNodeAttribute;
      static agxData::ScalarAttributeT< agx::Vec4f >* colorAttribute;
      static agxData::ScalarAttributeT< agx::Vec3 >* geometryContactVectorAttribute;
      static agxData::ScalarAttributeT< agx::Real32 >* enableRenderingAttribute;

    protected:
      virtual ~ParticleModel();
      virtual agxData::EntityData* createData(agxData::EntityStorage* storage) override;
      virtual void configure(agx::TiXmlElement* eEntity) override;
      virtual void initAttributeAccessors() override;
      virtual void construct(agxData::EntityPtr instance) override;
      void construct(agx::Physics::ParticlePtr particle);
    };


    DOXYGEN_START_INTERNAL_BLOCK()
    #ifndef AGX_PHYSICS_PARTICLE_DATA_SET_OVERRIDE
    #define AGX_PHYSICS_PARTICLE_DATA_SET
    class AGXPHYSICS_EXPORT ParticleData : public agxData::EntityData
    {
    public:
      ParticleInstance operator[] (size_t index);

    public:
      agxData::Array< ParticlePtr >& instance;
      agxData::Array< agx::ParticleState > state;
      agx::Real defaultRadius;
      agx::Real defaultMass;
      agx::Physics::MaterialPtr defaultMaterial;
      agx::Physics::CollisionGroupSetPtr defaultCollisionGroupSet;
      agx::Real defaultInvMass;
      agxData::Array< agx::Vec3 > position;
      agxData::Array< agx::Vec3 > velocity;
      agxData::Array< agx::Vec3 > oldVelocity;
      agxData::Array< agx::Vec3 > force;
      agxData::Array< agx::Real > life;
      agxData::Array< agx::Real > radius;
      agxData::Array< agx::Real > mass;
      agxData::Array< agx::Real > invMass;
      agxData::Array< agx::Physics::MaterialPtr > material;
      agxData::Array< agx::Uuid > materialUuid;
      agxData::Array< agx::Physics::CollisionGroupSetPtr > collisionGroupSet;
      agxData::Array< agx::UInt8 > tier;
      agxData::Array< agx::UInt32 > cellIndex;
      agxData::Array< agx::UInt16 > cellSlot;
      agxData::Array< agx::UInt32 > oldCell;
      agxData::Array< agx::UInt32 > geometryContactList;
      agxData::Array< agx::UInt32 > solveBodyIndex;
      agxData::Array< agx::Physics::GraphNodePtr > graphNode;
      agxData::Array< agx::Vec4f > color;
      agxData::Array< agx::Vec3 > geometryContactVector;
      agxData::Array< agx::Real32 > enableRendering;

    public:
      typedef agx::ParticleState stateType;
      typedef agx::Real defaultRadiusType;
      typedef agx::Real defaultMassType;
      typedef agx::Physics::MaterialPtr defaultMaterialType;
      typedef agx::Physics::CollisionGroupSetPtr defaultCollisionGroupSetType;
      typedef agx::Real defaultInvMassType;
      typedef agx::Vec3 positionType;
      typedef agx::Vec3 velocityType;
      typedef agx::Vec3 oldVelocityType;
      typedef agx::Vec3 forceType;
      typedef agx::Real lifeType;
      typedef agx::Real radiusType;
      typedef agx::Real massType;
      typedef agx::Real invMassType;
      typedef agx::Physics::MaterialPtr materialType;
      typedef agx::Uuid materialUuidType;
      typedef agx::Physics::CollisionGroupSetPtr collisionGroupSetType;
      typedef agx::UInt8 tierType;
      typedef agx::UInt32 cellIndexType;
      typedef agx::UInt16 cellSlotType;
      typedef agx::UInt32 oldCellType;
      typedef agx::UInt32 geometryContactListType;
      typedef agx::UInt32 solveBodyIndexType;
      typedef agx::Physics::GraphNodePtr graphNodeType;
      typedef agx::Vec4f colorType;
      typedef agx::Vec3 geometryContactVectorType;
      typedef agx::Real32 enableRenderingType;

    public:
      ParticleData(agxData::EntityStorage* storage);
      ParticleData();

    protected:
      virtual ~ParticleData() {}
      virtual void setNumElements(agx::Index numElements) override;
      virtual void synchronizeSharedAttribute(agxData::Value* value) override;

    private:
      ParticleData& operator= (const ParticleData&) { return *this; }

    };
    #endif
    DOXYGEN_END_INTERNAL_BLOCK()


    DOXYGEN_START_INTERNAL_BLOCK()
    class AGXPHYSICS_EXPORT ParticleSemantics : protected agxData::EntityPtr
    {
    public:

      // Automatic getters
      agx::ParticleState const& getState() const;
      agx::Vec3 const& getPosition() const;
      agx::Vec3 const& getVelocity() const;
      agx::Vec3 const& getOldVelocity() const;
      agx::Vec3 const& getForce() const;
      agx::Real const& getLife() const;
      agx::Real const& getRadius() const;
      agx::Real const& getMass() const;
      agx::Real const& getInvMass() const;
      agx::Physics::MaterialPtr const& getMaterial() const;
      agx::Uuid const& getMaterialUuid() const;
      agx::Physics::CollisionGroupSetPtr const& getCollisionGroupSet() const;
      agx::UInt8 const& getTier() const;
      agx::UInt32 const& getCellIndex() const;
      agx::UInt16 const& getCellSlot() const;
      agx::UInt32 const& getOldCell() const;
      agx::UInt32 const& getGeometryContactList() const;
      agx::UInt32 const& getSolveBodyIndex() const;
      agx::Physics::GraphNodePtr const& getGraphNode() const;
      agx::Vec4f const& getColor() const;
      agx::Vec3 const& getGeometryContactVector() const;
      agx::Real32 const& getEnableRendering() const;

      // Semantics defined by explicit kernels
      void integrate(const agx::Real& clock_timeStep);
      void integratePosition(const agx::Real& clock_timeStep);
      void integrateVelocity(const agx::Real& clock_timeStep);
      void setColor(const agx::Vec4f& color);
      void setMass(const agx::Real& mass);
      void setMaterial(const agx::Physics::MaterialPtr& material);
      void setPosition(const agx::Vec3& position);
      void setRadius(const agx::Real& radius);
      void setVelocity(const agx::Vec3& velocity);

      // Automatic setters
      void setState(agx::ParticleState const& value);
      void setOldVelocity(agx::Vec3 const& value);
      void setForce(agx::Vec3 const& value);
      void setLife(agx::Real const& value);
      void setInvMass(agx::Real const& value);
      void setMaterialUuid(agx::Uuid const& value);
      void setCollisionGroupSet(agx::Physics::CollisionGroupSetPtr const& value);
      void setTier(agx::UInt8 const& value);
      void setCellIndex(agx::UInt32 const& value);
      void setCellSlot(agx::UInt16 const& value);
      void setOldCell(agx::UInt32 const& value);
      void setGeometryContactList(agx::UInt32 const& value);
      void setSolveBodyIndex(agx::UInt32 const& value);
      void setGraphNode(agx::Physics::GraphNodePtr const& value);
      void setGeometryContactVector(agx::Vec3 const& value);
      void setEnableRendering(agx::Real32 const& value);


    protected:
      friend class ParticlePtr;
      friend class ParticleInstance;
      ParticleSemantics();
    };
    DOXYGEN_END_INTERNAL_BLOCK()


    /**
    Pointer to a entity instance of type Physics.Particle
    */
    class CALLABLE ParticlePtr : public agxData::EntityPtr
    {
    public:
      typedef ParticleModel ModelType;
      typedef ParticleData DataType;
      typedef ParticleInstance InstanceType;

    public:
      AGXPHYSICS_EXPORT ParticlePtr();
      AGXPHYSICS_EXPORT ParticlePtr(agxData::EntityStorage* storage, agx::Index id);
      AGXPHYSICS_EXPORT ParticlePtr(const agxData::EntityPtr& ptr);
      AGXPHYSICS_EXPORT ParticlePtr(const agxData::EntityInstance& instance);
      AGXPHYSICS_EXPORT ParticlePtr& operator= (const agxData::EntityPtr& ptr);
      AGXPHYSICS_EXPORT ParticlePtr& operator= (const agxData::EntityInstance& instance);
      AGXPHYSICS_EXPORT ParticleInstance instance();
      AGXPHYSICS_EXPORT const ParticleInstance instance() const;

      AGXPHYSICS_EXPORT ParticleSemantics* operator->();
      AGXPHYSICS_EXPORT const ParticleSemantics* operator->() const;

      ParticleData* getData();
      const ParticleData* getData() const;


      /// \return reference to the state attribute
      AGXPHYSICS_EXPORT agx::ParticleState& state();
      /// \return const reference to the state attribute
      AGXPHYSICS_EXPORT agx::ParticleState const& state() const;

      /// \return reference to the defaultRadius attribute
      AGXPHYSICS_EXPORT agx::Real& defaultRadius();
      /// \return const reference to the defaultRadius attribute
      AGXPHYSICS_EXPORT agx::Real const& defaultRadius() const;

      /// \return reference to the defaultMass attribute
      AGXPHYSICS_EXPORT agx::Real& defaultMass();
      /// \return const reference to the defaultMass attribute
      AGXPHYSICS_EXPORT agx::Real const& defaultMass() const;

      /// \return reference to the defaultMaterial attribute
      AGXPHYSICS_EXPORT agx::Physics::MaterialPtr& defaultMaterial();
      /// \return const reference to the defaultMaterial attribute
      AGXPHYSICS_EXPORT agx::Physics::MaterialPtr const& defaultMaterial() const;

      /// \return reference to the defaultCollisionGroupSet attribute
      AGXPHYSICS_EXPORT agx::Physics::CollisionGroupSetPtr& defaultCollisionGroupSet();
      /// \return const reference to the defaultCollisionGroupSet attribute
      AGXPHYSICS_EXPORT agx::Physics::CollisionGroupSetPtr const& defaultCollisionGroupSet() const;

      /// \return reference to the defaultInvMass attribute
      AGXPHYSICS_EXPORT agx::Real& defaultInvMass();
      /// \return const reference to the defaultInvMass attribute
      AGXPHYSICS_EXPORT agx::Real const& defaultInvMass() const;

      /// \return reference to the position attribute
      AGXPHYSICS_EXPORT agx::Vec3& position();
      /// \return const reference to the position attribute
      AGXPHYSICS_EXPORT agx::Vec3 const& position() const;

      /// \return reference to the velocity attribute
      AGXPHYSICS_EXPORT agx::Vec3& velocity();
      /// \return const reference to the velocity attribute
      AGXPHYSICS_EXPORT agx::Vec3 const& velocity() const;

      /// \return reference to the oldVelocity attribute
      AGXPHYSICS_EXPORT agx::Vec3& oldVelocity();
      /// \return const reference to the oldVelocity attribute
      AGXPHYSICS_EXPORT agx::Vec3 const& oldVelocity() const;

      /// \return reference to the force attribute
      AGXPHYSICS_EXPORT agx::Vec3& force();
      /// \return const reference to the force attribute
      AGXPHYSICS_EXPORT agx::Vec3 const& force() const;

      /// \return reference to the life attribute
      AGXPHYSICS_EXPORT agx::Real& life();
      /// \return const reference to the life attribute
      AGXPHYSICS_EXPORT agx::Real const& life() const;

      /// \return reference to the radius attribute
      AGXPHYSICS_EXPORT agx::Real& radius();
      /// \return const reference to the radius attribute
      AGXPHYSICS_EXPORT agx::Real const& radius() const;

      /// \return reference to the mass attribute
      AGXPHYSICS_EXPORT agx::Real& mass();
      /// \return const reference to the mass attribute
      AGXPHYSICS_EXPORT agx::Real const& mass() const;

      /// \return reference to the invMass attribute
      AGXPHYSICS_EXPORT agx::Real& invMass();
      /// \return const reference to the invMass attribute
      AGXPHYSICS_EXPORT agx::Real const& invMass() const;

      /// \return reference to the material attribute
      AGXPHYSICS_EXPORT agx::Physics::MaterialPtr& material();
      /// \return const reference to the material attribute
      AGXPHYSICS_EXPORT agx::Physics::MaterialPtr const& material() const;

      /// \return reference to the materialUuid attribute
      AGXPHYSICS_EXPORT agx::Uuid& materialUuid();
      /// \return const reference to the materialUuid attribute
      AGXPHYSICS_EXPORT agx::Uuid const& materialUuid() const;

      /// \return reference to the collisionGroupSet attribute
      AGXPHYSICS_EXPORT agx::Physics::CollisionGroupSetPtr& collisionGroupSet();
      /// \return const reference to the collisionGroupSet attribute
      AGXPHYSICS_EXPORT agx::Physics::CollisionGroupSetPtr const& collisionGroupSet() const;

      /// \return reference to the tier attribute
      AGXPHYSICS_EXPORT agx::UInt8& tier();
      /// \return const reference to the tier attribute
      AGXPHYSICS_EXPORT agx::UInt8 const& tier() const;

      /// \return reference to the cellIndex attribute
      AGXPHYSICS_EXPORT agx::UInt32& cellIndex();
      /// \return const reference to the cellIndex attribute
      AGXPHYSICS_EXPORT agx::UInt32 const& cellIndex() const;

      /// \return reference to the cellSlot attribute
      AGXPHYSICS_EXPORT agx::UInt16& cellSlot();
      /// \return const reference to the cellSlot attribute
      AGXPHYSICS_EXPORT agx::UInt16 const& cellSlot() const;

      /// \return reference to the oldCell attribute
      AGXPHYSICS_EXPORT agx::UInt32& oldCell();
      /// \return const reference to the oldCell attribute
      AGXPHYSICS_EXPORT agx::UInt32 const& oldCell() const;

      /// \return reference to the geometryContactList attribute
      AGXPHYSICS_EXPORT agx::UInt32& geometryContactList();
      /// \return const reference to the geometryContactList attribute
      AGXPHYSICS_EXPORT agx::UInt32 const& geometryContactList() const;

      /// \return reference to the solveBodyIndex attribute
      AGXPHYSICS_EXPORT agx::UInt32& solveBodyIndex();
      /// \return const reference to the solveBodyIndex attribute
      AGXPHYSICS_EXPORT agx::UInt32 const& solveBodyIndex() const;

      /// \return reference to the graphNode attribute
      AGXPHYSICS_EXPORT agx::Physics::GraphNodePtr& graphNode();
      /// \return const reference to the graphNode attribute
      AGXPHYSICS_EXPORT agx::Physics::GraphNodePtr const& graphNode() const;

      /// \return reference to the color attribute
      AGXPHYSICS_EXPORT agx::Vec4f& color();
      /// \return const reference to the color attribute
      AGXPHYSICS_EXPORT agx::Vec4f const& color() const;

      /// \return reference to the geometryContactVector attribute
      AGXPHYSICS_EXPORT agx::Vec3& geometryContactVector();
      /// \return const reference to the geometryContactVector attribute
      AGXPHYSICS_EXPORT agx::Vec3 const& geometryContactVector() const;

      /// \return reference to the enableRendering attribute
      AGXPHYSICS_EXPORT agx::Real32& enableRendering();
      /// \return const reference to the enableRendering attribute
      AGXPHYSICS_EXPORT agx::Real32 const& enableRendering() const;

      DOXYGEN_START_INTERNAL_BLOCK()
      // Only used by tolua!
      #ifdef AGX_GENERATED_TOLUA_WRAPPER
      inline agx::ParticleState const& getState() const { return (*this)->getState(); }
      inline agx::Vec3 const& getPosition() const { return (*this)->getPosition(); }
      inline agx::Vec3 const& getVelocity() const { return (*this)->getVelocity(); }
      inline agx::Vec3 const& getOldVelocity() const { return (*this)->getOldVelocity(); }
      inline agx::Vec3 const& getForce() const { return (*this)->getForce(); }
      inline agx::Real const& getLife() const { return (*this)->getLife(); }
      inline agx::Real const& getRadius() const { return (*this)->getRadius(); }
      inline agx::Real const& getMass() const { return (*this)->getMass(); }
      inline agx::Real const& getInvMass() const { return (*this)->getInvMass(); }
      inline agx::Physics::MaterialPtr const& getMaterial() const { return (*this)->getMaterial(); }
      inline agx::Uuid const& getMaterialUuid() const { return (*this)->getMaterialUuid(); }
      inline agx::Physics::CollisionGroupSetPtr const& getCollisionGroupSet() const { return (*this)->getCollisionGroupSet(); }
      inline agx::UInt8 const& getTier() const { return (*this)->getTier(); }
      inline agx::UInt32 const& getCellIndex() const { return (*this)->getCellIndex(); }
      inline agx::UInt16 const& getCellSlot() const { return (*this)->getCellSlot(); }
      inline agx::UInt32 const& getOldCell() const { return (*this)->getOldCell(); }
      inline agx::UInt32 const& getGeometryContactList() const { return (*this)->getGeometryContactList(); }
      inline agx::UInt32 const& getSolveBodyIndex() const { return (*this)->getSolveBodyIndex(); }
      inline agx::Physics::GraphNodePtr const& getGraphNode() const { return (*this)->getGraphNode(); }
      inline agx::Vec4f const& getColor() const { return (*this)->getColor(); }
      inline agx::Vec3 const& getGeometryContactVector() const { return (*this)->getGeometryContactVector(); }
      inline agx::Real32 const& getEnableRendering() const { return (*this)->getEnableRendering(); }

      inline void integrate(const agx::Real& clock_timeStep) { (*this)->integrate(clock_timeStep); }
      inline void integratePosition(const agx::Real& clock_timeStep) { (*this)->integratePosition(clock_timeStep); }
      inline void integrateVelocity(const agx::Real& clock_timeStep) { (*this)->integrateVelocity(clock_timeStep); }
      inline void setColor(const agx::Vec4f& color) { (*this)->setColor(color); }
      inline void setMass(const agx::Real& mass) { (*this)->setMass(mass); }
      inline void setMaterial(const agx::Physics::MaterialPtr& material) { (*this)->setMaterial(material); }
      inline void setPosition(const agx::Vec3& position) { (*this)->setPosition(position); }
      inline void setRadius(const agx::Real& radius) { (*this)->setRadius(radius); }
      inline void setVelocity(const agx::Vec3& velocity) { (*this)->setVelocity(velocity); }

      inline void setState(agx::ParticleState const& value) { (*this)->setState(value); }
      inline void setOldVelocity(agx::Vec3 const& value) { (*this)->setOldVelocity(value); }
      inline void setForce(agx::Vec3 const& value) { (*this)->setForce(value); }
      inline void setLife(agx::Real const& value) { (*this)->setLife(value); }
      inline void setInvMass(agx::Real const& value) { (*this)->setInvMass(value); }
      inline void setMaterialUuid(agx::Uuid const& value) { (*this)->setMaterialUuid(value); }
      inline void setCollisionGroupSet(agx::Physics::CollisionGroupSetPtr const& value) { (*this)->setCollisionGroupSet(value); }
      inline void setTier(agx::UInt8 const& value) { (*this)->setTier(value); }
      inline void setCellIndex(agx::UInt32 const& value) { (*this)->setCellIndex(value); }
      inline void setCellSlot(agx::UInt16 const& value) { (*this)->setCellSlot(value); }
      inline void setOldCell(agx::UInt32 const& value) { (*this)->setOldCell(value); }
      inline void setGeometryContactList(agx::UInt32 const& value) { (*this)->setGeometryContactList(value); }
      inline void setSolveBodyIndex(agx::UInt32 const& value) { (*this)->setSolveBodyIndex(value); }
      inline void setGraphNode(agx::Physics::GraphNodePtr const& value) { (*this)->setGraphNode(value); }
      inline void setGeometryContactVector(agx::Vec3 const& value) { (*this)->setGeometryContactVector(value); }
      inline void setEnableRendering(agx::Real32 const& value) { (*this)->setEnableRendering(value); }
      #endif
      DOXYGEN_END_INTERNAL_BLOCK()

    };


    DOXYGEN_START_INTERNAL_BLOCK()
    class AGXPHYSICS_EXPORT ParticleInstance : public agxData::EntityInstance
    {
    public:
      ParticleInstance();
      ParticleInstance(ParticleData* data, agx::Index index);
      ParticleInstance(agxData::EntityStorage *storage, agx::Index index);
      ParticleInstance(const agxData::EntityInstance& other);
      ParticleInstance(const agxData::EntityPtr& ptr);

      ParticleData* getData();
      const ParticleData* getData() const;

    public:
      /// \return reference to the state attribute
      agx::ParticleState& state();
      /// \return const reference to the state attribute
      agx::ParticleState const& state() const;

      /// \return reference to the defaultRadius attribute
      agx::Real& defaultRadius();
      /// \return const reference to the defaultRadius attribute
      agx::Real const& defaultRadius() const;

      /// \return reference to the defaultMass attribute
      agx::Real& defaultMass();
      /// \return const reference to the defaultMass attribute
      agx::Real const& defaultMass() const;

      /// \return reference to the defaultMaterial attribute
      agx::Physics::MaterialPtr& defaultMaterial();
      /// \return const reference to the defaultMaterial attribute
      agx::Physics::MaterialPtr const& defaultMaterial() const;

      /// \return reference to the defaultCollisionGroupSet attribute
      agx::Physics::CollisionGroupSetPtr& defaultCollisionGroupSet();
      /// \return const reference to the defaultCollisionGroupSet attribute
      agx::Physics::CollisionGroupSetPtr const& defaultCollisionGroupSet() const;

      /// \return reference to the defaultInvMass attribute
      agx::Real& defaultInvMass();
      /// \return const reference to the defaultInvMass attribute
      agx::Real const& defaultInvMass() const;

      /// \return reference to the position attribute
      agx::Vec3& position();
      /// \return const reference to the position attribute
      agx::Vec3 const& position() const;

      /// \return reference to the velocity attribute
      agx::Vec3& velocity();
      /// \return const reference to the velocity attribute
      agx::Vec3 const& velocity() const;

      /// \return reference to the oldVelocity attribute
      agx::Vec3& oldVelocity();
      /// \return const reference to the oldVelocity attribute
      agx::Vec3 const& oldVelocity() const;

      /// \return reference to the force attribute
      agx::Vec3& force();
      /// \return const reference to the force attribute
      agx::Vec3 const& force() const;

      /// \return reference to the life attribute
      agx::Real& life();
      /// \return const reference to the life attribute
      agx::Real const& life() const;

      /// \return reference to the radius attribute
      agx::Real& radius();
      /// \return const reference to the radius attribute
      agx::Real const& radius() const;

      /// \return reference to the mass attribute
      agx::Real& mass();
      /// \return const reference to the mass attribute
      agx::Real const& mass() const;

      /// \return reference to the invMass attribute
      agx::Real& invMass();
      /// \return const reference to the invMass attribute
      agx::Real const& invMass() const;

      /// \return reference to the material attribute
      agx::Physics::MaterialPtr& material();
      /// \return const reference to the material attribute
      agx::Physics::MaterialPtr const& material() const;

      /// \return reference to the materialUuid attribute
      agx::Uuid& materialUuid();
      /// \return const reference to the materialUuid attribute
      agx::Uuid const& materialUuid() const;

      /// \return reference to the collisionGroupSet attribute
      agx::Physics::CollisionGroupSetPtr& collisionGroupSet();
      /// \return const reference to the collisionGroupSet attribute
      agx::Physics::CollisionGroupSetPtr const& collisionGroupSet() const;

      /// \return reference to the tier attribute
      agx::UInt8& tier();
      /// \return const reference to the tier attribute
      agx::UInt8 const& tier() const;

      /// \return reference to the cellIndex attribute
      agx::UInt32& cellIndex();
      /// \return const reference to the cellIndex attribute
      agx::UInt32 const& cellIndex() const;

      /// \return reference to the cellSlot attribute
      agx::UInt16& cellSlot();
      /// \return const reference to the cellSlot attribute
      agx::UInt16 const& cellSlot() const;

      /// \return reference to the oldCell attribute
      agx::UInt32& oldCell();
      /// \return const reference to the oldCell attribute
      agx::UInt32 const& oldCell() const;

      /// \return reference to the geometryContactList attribute
      agx::UInt32& geometryContactList();
      /// \return const reference to the geometryContactList attribute
      agx::UInt32 const& geometryContactList() const;

      /// \return reference to the solveBodyIndex attribute
      agx::UInt32& solveBodyIndex();
      /// \return const reference to the solveBodyIndex attribute
      agx::UInt32 const& solveBodyIndex() const;

      /// \return reference to the graphNode attribute
      agx::Physics::GraphNodePtr& graphNode();
      /// \return const reference to the graphNode attribute
      agx::Physics::GraphNodePtr const& graphNode() const;

      /// \return reference to the color attribute
      agx::Vec4f& color();
      /// \return const reference to the color attribute
      agx::Vec4f const& color() const;

      /// \return reference to the geometryContactVector attribute
      agx::Vec3& geometryContactVector();
      /// \return const reference to the geometryContactVector attribute
      agx::Vec3 const& geometryContactVector() const;

      /// \return reference to the enableRendering attribute
      agx::Real32& enableRendering();
      /// \return const reference to the enableRendering attribute
      agx::Real32 const& enableRendering() const;

    };
    DOXYGEN_END_INTERNAL_BLOCK()



    typedef agx::VectorPOD<ParticlePtr> ParticlePtrVector;
    typedef agxData::Array<ParticlePtr> ParticlePtrArray;



    DOXYGEN_START_INTERNAL_BLOCK()
    /* Implementation */
    //-----------------------------------------------------------------------------------------------------
    //-----------------------------------------------------------------------------------------------------
    inline ParticleInstance agx::Physics::ParticleData::operator[] (size_t index) { return ParticleInstance(this, (agx::Index)index); }
    //-----------------------------------------------------------------------------------------------------
    AGX_FORCE_INLINE ParticlePtr::ParticlePtr() {}
    AGX_FORCE_INLINE ParticlePtr::ParticlePtr(agxData::EntityStorage* storage, agx::Index id) : agxData::EntityPtr(storage, id) {}
    AGX_FORCE_INLINE ParticlePtr::ParticlePtr(const agxData::EntityPtr& ptr) : agxData::EntityPtr(ptr)
    {
      agxAssertN(!ptr || ptr.isInstanceOf(ParticleModel::instance()),
        "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
        EntityPtr::getModel()->fullPath().c_str(), ParticleModel::instance()->fullPath().c_str());
    }

    AGX_FORCE_INLINE ParticlePtr::ParticlePtr(const agxData::EntityInstance& instance) : agxData::EntityPtr(instance)
    {
      agxAssertN(!instance || instance.isInstanceOf(ParticleModel::instance()),
        "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
        EntityPtr::getModel()->fullPath().c_str(), ParticleModel::instance()->fullPath().c_str());
    }

    AGX_FORCE_INLINE ParticlePtr& ParticlePtr::operator= (const agxData::EntityPtr& ptr)
    {
      agxData::EntityPtr::operator= (ptr);
      agxAssertN(!ptr || ptr.isInstanceOf(ParticleModel::instance()),
        "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
        EntityPtr::getModel()->fullPath().c_str(), ParticleModel::instance()->fullPath().c_str());
      return *this;
    }

    AGX_FORCE_INLINE ParticlePtr& ParticlePtr::operator= (const agxData::EntityInstance& instance)
    {
      agxData::EntityPtr::operator= (instance);
      agxAssertN(!instance || instance.isInstanceOf(ParticleModel::instance()),
        "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
        EntityPtr::getModel()->fullPath().c_str(), ParticleModel::instance()->fullPath().c_str());
      return *this;
    }

    inline ParticleInstance ParticlePtr::instance() { return agxData::EntityPtr::instance(); }
    inline const ParticleInstance ParticlePtr::instance() const { return agxData::EntityPtr::instance(); }
    AGX_FORCE_INLINE ParticleSemantics* ParticlePtr::operator->() { return (ParticleSemantics* )this; }
    AGX_FORCE_INLINE const ParticleSemantics* ParticlePtr::operator->() const { return (const ParticleSemantics* )this; }
    AGX_FORCE_INLINE ParticleData* ParticlePtr::getData() { return static_cast<ParticleData* >(agxData::EntityPtr::getData()); }
    AGX_FORCE_INLINE const ParticleData* ParticlePtr::getData() const { return static_cast<const ParticleData* >(agxData::EntityPtr::getData()); }

    AGX_FORCE_INLINE agx::ParticleState& ParticlePtr::state() { verifyIndex(); return getData()->state[calculateIndex()]; }
    AGX_FORCE_INLINE agx::ParticleState const& ParticlePtr::state() const { verifyIndex(); return getData()->state[calculateIndex()]; }

    AGX_FORCE_INLINE agx::Real& ParticlePtr::defaultRadius() { verifyIndex(); return getData()->defaultRadius; }
    AGX_FORCE_INLINE agx::Real const& ParticlePtr::defaultRadius() const { verifyIndex(); return getData()->defaultRadius; }

    AGX_FORCE_INLINE agx::Real& ParticlePtr::defaultMass() { verifyIndex(); return getData()->defaultMass; }
    AGX_FORCE_INLINE agx::Real const& ParticlePtr::defaultMass() const { verifyIndex(); return getData()->defaultMass; }

    AGX_FORCE_INLINE agx::Physics::MaterialPtr& ParticlePtr::defaultMaterial() { verifyIndex(); return getData()->defaultMaterial; }
    AGX_FORCE_INLINE agx::Physics::MaterialPtr const& ParticlePtr::defaultMaterial() const { verifyIndex(); return getData()->defaultMaterial; }

    AGX_FORCE_INLINE agx::Physics::CollisionGroupSetPtr& ParticlePtr::defaultCollisionGroupSet() { verifyIndex(); return getData()->defaultCollisionGroupSet; }
    AGX_FORCE_INLINE agx::Physics::CollisionGroupSetPtr const& ParticlePtr::defaultCollisionGroupSet() const { verifyIndex(); return getData()->defaultCollisionGroupSet; }

    AGX_FORCE_INLINE agx::Real& ParticlePtr::defaultInvMass() { verifyIndex(); return getData()->defaultInvMass; }
    AGX_FORCE_INLINE agx::Real const& ParticlePtr::defaultInvMass() const { verifyIndex(); return getData()->defaultInvMass; }

    AGX_FORCE_INLINE agx::Vec3& ParticlePtr::position() { verifyIndex(); return getData()->position[calculateIndex()]; }
    AGX_FORCE_INLINE agx::Vec3 const& ParticlePtr::position() const { verifyIndex(); return getData()->position[calculateIndex()]; }

    AGX_FORCE_INLINE agx::Vec3& ParticlePtr::velocity() { verifyIndex(); return getData()->velocity[calculateIndex()]; }
    AGX_FORCE_INLINE agx::Vec3 const& ParticlePtr::velocity() const { verifyIndex(); return getData()->velocity[calculateIndex()]; }

    AGX_FORCE_INLINE agx::Vec3& ParticlePtr::oldVelocity() { verifyIndex(); return getData()->oldVelocity[calculateIndex()]; }
    AGX_FORCE_INLINE agx::Vec3 const& ParticlePtr::oldVelocity() const { verifyIndex(); return getData()->oldVelocity[calculateIndex()]; }

    AGX_FORCE_INLINE agx::Vec3& ParticlePtr::force() { verifyIndex(); return getData()->force[calculateIndex()]; }
    AGX_FORCE_INLINE agx::Vec3 const& ParticlePtr::force() const { verifyIndex(); return getData()->force[calculateIndex()]; }

    AGX_FORCE_INLINE agx::Real& ParticlePtr::life() { verifyIndex(); return getData()->life[calculateIndex()]; }
    AGX_FORCE_INLINE agx::Real const& ParticlePtr::life() const { verifyIndex(); return getData()->life[calculateIndex()]; }

    AGX_FORCE_INLINE agx::Real& ParticlePtr::radius() { verifyIndex(); return getData()->radius[calculateIndex()]; }
    AGX_FORCE_INLINE agx::Real const& ParticlePtr::radius() const { verifyIndex(); return getData()->radius[calculateIndex()]; }

    AGX_FORCE_INLINE agx::Real& ParticlePtr::mass() { verifyIndex(); return getData()->mass[calculateIndex()]; }
    AGX_FORCE_INLINE agx::Real const& ParticlePtr::mass() const { verifyIndex(); return getData()->mass[calculateIndex()]; }

    AGX_FORCE_INLINE agx::Real& ParticlePtr::invMass() { verifyIndex(); return getData()->invMass[calculateIndex()]; }
    AGX_FORCE_INLINE agx::Real const& ParticlePtr::invMass() const { verifyIndex(); return getData()->invMass[calculateIndex()]; }

    AGX_FORCE_INLINE agx::Physics::MaterialPtr& ParticlePtr::material() { verifyIndex(); return getData()->material[calculateIndex()]; }
    AGX_FORCE_INLINE agx::Physics::MaterialPtr const& ParticlePtr::material() const { verifyIndex(); return getData()->material[calculateIndex()]; }

    AGX_FORCE_INLINE agx::Uuid& ParticlePtr::materialUuid() { verifyIndex(); return getData()->materialUuid[calculateIndex()]; }
    AGX_FORCE_INLINE agx::Uuid const& ParticlePtr::materialUuid() const { verifyIndex(); return getData()->materialUuid[calculateIndex()]; }

    AGX_FORCE_INLINE agx::Physics::CollisionGroupSetPtr& ParticlePtr::collisionGroupSet() { verifyIndex(); return getData()->collisionGroupSet[calculateIndex()]; }
    AGX_FORCE_INLINE agx::Physics::CollisionGroupSetPtr const& ParticlePtr::collisionGroupSet() const { verifyIndex(); return getData()->collisionGroupSet[calculateIndex()]; }

    AGX_FORCE_INLINE agx::UInt8& ParticlePtr::tier() { verifyIndex(); return getData()->tier[calculateIndex()]; }
    AGX_FORCE_INLINE agx::UInt8 const& ParticlePtr::tier() const { verifyIndex(); return getData()->tier[calculateIndex()]; }

    AGX_FORCE_INLINE agx::UInt32& ParticlePtr::cellIndex() { verifyIndex(); return getData()->cellIndex[calculateIndex()]; }
    AGX_FORCE_INLINE agx::UInt32 const& ParticlePtr::cellIndex() const { verifyIndex(); return getData()->cellIndex[calculateIndex()]; }

    AGX_FORCE_INLINE agx::UInt16& ParticlePtr::cellSlot() { verifyIndex(); return getData()->cellSlot[calculateIndex()]; }
    AGX_FORCE_INLINE agx::UInt16 const& ParticlePtr::cellSlot() const { verifyIndex(); return getData()->cellSlot[calculateIndex()]; }

    AGX_FORCE_INLINE agx::UInt32& ParticlePtr::oldCell() { verifyIndex(); return getData()->oldCell[calculateIndex()]; }
    AGX_FORCE_INLINE agx::UInt32 const& ParticlePtr::oldCell() const { verifyIndex(); return getData()->oldCell[calculateIndex()]; }

    AGX_FORCE_INLINE agx::UInt32& ParticlePtr::geometryContactList() { verifyIndex(); return getData()->geometryContactList[calculateIndex()]; }
    AGX_FORCE_INLINE agx::UInt32 const& ParticlePtr::geometryContactList() const { verifyIndex(); return getData()->geometryContactList[calculateIndex()]; }

    AGX_FORCE_INLINE agx::UInt32& ParticlePtr::solveBodyIndex() { verifyIndex(); return getData()->solveBodyIndex[calculateIndex()]; }
    AGX_FORCE_INLINE agx::UInt32 const& ParticlePtr::solveBodyIndex() const { verifyIndex(); return getData()->solveBodyIndex[calculateIndex()]; }

    AGX_FORCE_INLINE agx::Physics::GraphNodePtr& ParticlePtr::graphNode() { verifyIndex(); return getData()->graphNode[calculateIndex()]; }
    AGX_FORCE_INLINE agx::Physics::GraphNodePtr const& ParticlePtr::graphNode() const { verifyIndex(); return getData()->graphNode[calculateIndex()]; }

    AGX_FORCE_INLINE agx::Vec4f& ParticlePtr::color() { verifyIndex(); return getData()->color[calculateIndex()]; }
    AGX_FORCE_INLINE agx::Vec4f const& ParticlePtr::color() const { verifyIndex(); return getData()->color[calculateIndex()]; }

    AGX_FORCE_INLINE agx::Vec3& ParticlePtr::geometryContactVector() { verifyIndex(); return getData()->geometryContactVector[calculateIndex()]; }
    AGX_FORCE_INLINE agx::Vec3 const& ParticlePtr::geometryContactVector() const { verifyIndex(); return getData()->geometryContactVector[calculateIndex()]; }

    AGX_FORCE_INLINE agx::Real32& ParticlePtr::enableRendering() { verifyIndex(); return getData()->enableRendering[calculateIndex()]; }
    AGX_FORCE_INLINE agx::Real32 const& ParticlePtr::enableRendering() const { verifyIndex(); return getData()->enableRendering[calculateIndex()]; }

    //-----------------------------------------------------------------------------------------------------
    AGX_FORCE_INLINE ParticleInstance::ParticleInstance() {}
    AGX_FORCE_INLINE ParticleInstance::ParticleInstance(ParticleData* data, agx::Index index) : agxData::EntityInstance(data, index) {}
    AGX_FORCE_INLINE ParticleInstance::ParticleInstance(agxData::EntityStorage* storage, agx::Index index) : agxData::EntityInstance(storage, index) {}
    AGX_FORCE_INLINE ParticleInstance::ParticleInstance(const agxData::EntityInstance& other) : agxData::EntityInstance(other)
    {
      agxAssertN(!other || other.isInstanceOf(ParticleModel::instance()),
        "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
        EntityInstance::getModel()->fullPath().c_str(), ParticleModel::instance()->fullPath().c_str());
    }

    AGX_FORCE_INLINE ParticleInstance::ParticleInstance(const agxData::EntityPtr& ptr) : agxData::EntityInstance(ptr)
    {
      agxAssertN(!ptr || ptr.isInstanceOf(ParticleModel::instance()),
        "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
        EntityInstance::getModel()->fullPath().c_str(), ParticleModel::instance()->fullPath().c_str());
    }


    AGX_FORCE_INLINE ParticleData* ParticleInstance::getData() { return static_cast<ParticleData* >(agxData::EntityInstance::getData()); }
    AGX_FORCE_INLINE const ParticleData* ParticleInstance::getData() const { return static_cast<const ParticleData* >(agxData::EntityInstance::getData()); }

    AGX_FORCE_INLINE agx::ParticleState& ParticleInstance::state() { verifyIndex(); return getData()->state[getIndex()]; }
    AGX_FORCE_INLINE agx::ParticleState const& ParticleInstance::state() const { verifyIndex(); return getData()->state[getIndex()]; }

    AGX_FORCE_INLINE agx::Real& ParticleInstance::defaultRadius() { verifyIndex(); return getData()->defaultRadius; }
    AGX_FORCE_INLINE agx::Real const& ParticleInstance::defaultRadius() const { verifyIndex(); return getData()->defaultRadius; }

    AGX_FORCE_INLINE agx::Real& ParticleInstance::defaultMass() { verifyIndex(); return getData()->defaultMass; }
    AGX_FORCE_INLINE agx::Real const& ParticleInstance::defaultMass() const { verifyIndex(); return getData()->defaultMass; }

    AGX_FORCE_INLINE agx::Physics::MaterialPtr& ParticleInstance::defaultMaterial() { verifyIndex(); return getData()->defaultMaterial; }
    AGX_FORCE_INLINE agx::Physics::MaterialPtr const& ParticleInstance::defaultMaterial() const { verifyIndex(); return getData()->defaultMaterial; }

    AGX_FORCE_INLINE agx::Physics::CollisionGroupSetPtr& ParticleInstance::defaultCollisionGroupSet() { verifyIndex(); return getData()->defaultCollisionGroupSet; }
    AGX_FORCE_INLINE agx::Physics::CollisionGroupSetPtr const& ParticleInstance::defaultCollisionGroupSet() const { verifyIndex(); return getData()->defaultCollisionGroupSet; }

    AGX_FORCE_INLINE agx::Real& ParticleInstance::defaultInvMass() { verifyIndex(); return getData()->defaultInvMass; }
    AGX_FORCE_INLINE agx::Real const& ParticleInstance::defaultInvMass() const { verifyIndex(); return getData()->defaultInvMass; }

    AGX_FORCE_INLINE agx::Vec3& ParticleInstance::position() { verifyIndex(); return getData()->position[getIndex()]; }
    AGX_FORCE_INLINE agx::Vec3 const& ParticleInstance::position() const { verifyIndex(); return getData()->position[getIndex()]; }

    AGX_FORCE_INLINE agx::Vec3& ParticleInstance::velocity() { verifyIndex(); return getData()->velocity[getIndex()]; }
    AGX_FORCE_INLINE agx::Vec3 const& ParticleInstance::velocity() const { verifyIndex(); return getData()->velocity[getIndex()]; }

    AGX_FORCE_INLINE agx::Vec3& ParticleInstance::oldVelocity() { verifyIndex(); return getData()->oldVelocity[getIndex()]; }
    AGX_FORCE_INLINE agx::Vec3 const& ParticleInstance::oldVelocity() const { verifyIndex(); return getData()->oldVelocity[getIndex()]; }

    AGX_FORCE_INLINE agx::Vec3& ParticleInstance::force() { verifyIndex(); return getData()->force[getIndex()]; }
    AGX_FORCE_INLINE agx::Vec3 const& ParticleInstance::force() const { verifyIndex(); return getData()->force[getIndex()]; }

    AGX_FORCE_INLINE agx::Real& ParticleInstance::life() { verifyIndex(); return getData()->life[getIndex()]; }
    AGX_FORCE_INLINE agx::Real const& ParticleInstance::life() const { verifyIndex(); return getData()->life[getIndex()]; }

    AGX_FORCE_INLINE agx::Real& ParticleInstance::radius() { verifyIndex(); return getData()->radius[getIndex()]; }
    AGX_FORCE_INLINE agx::Real const& ParticleInstance::radius() const { verifyIndex(); return getData()->radius[getIndex()]; }

    AGX_FORCE_INLINE agx::Real& ParticleInstance::mass() { verifyIndex(); return getData()->mass[getIndex()]; }
    AGX_FORCE_INLINE agx::Real const& ParticleInstance::mass() const { verifyIndex(); return getData()->mass[getIndex()]; }

    AGX_FORCE_INLINE agx::Real& ParticleInstance::invMass() { verifyIndex(); return getData()->invMass[getIndex()]; }
    AGX_FORCE_INLINE agx::Real const& ParticleInstance::invMass() const { verifyIndex(); return getData()->invMass[getIndex()]; }

    AGX_FORCE_INLINE agx::Physics::MaterialPtr& ParticleInstance::material() { verifyIndex(); return getData()->material[getIndex()]; }
    AGX_FORCE_INLINE agx::Physics::MaterialPtr const& ParticleInstance::material() const { verifyIndex(); return getData()->material[getIndex()]; }

    AGX_FORCE_INLINE agx::Uuid& ParticleInstance::materialUuid() { verifyIndex(); return getData()->materialUuid[getIndex()]; }
    AGX_FORCE_INLINE agx::Uuid const& ParticleInstance::materialUuid() const { verifyIndex(); return getData()->materialUuid[getIndex()]; }

    AGX_FORCE_INLINE agx::Physics::CollisionGroupSetPtr& ParticleInstance::collisionGroupSet() { verifyIndex(); return getData()->collisionGroupSet[getIndex()]; }
    AGX_FORCE_INLINE agx::Physics::CollisionGroupSetPtr const& ParticleInstance::collisionGroupSet() const { verifyIndex(); return getData()->collisionGroupSet[getIndex()]; }

    AGX_FORCE_INLINE agx::UInt8& ParticleInstance::tier() { verifyIndex(); return getData()->tier[getIndex()]; }
    AGX_FORCE_INLINE agx::UInt8 const& ParticleInstance::tier() const { verifyIndex(); return getData()->tier[getIndex()]; }

    AGX_FORCE_INLINE agx::UInt32& ParticleInstance::cellIndex() { verifyIndex(); return getData()->cellIndex[getIndex()]; }
    AGX_FORCE_INLINE agx::UInt32 const& ParticleInstance::cellIndex() const { verifyIndex(); return getData()->cellIndex[getIndex()]; }

    AGX_FORCE_INLINE agx::UInt16& ParticleInstance::cellSlot() { verifyIndex(); return getData()->cellSlot[getIndex()]; }
    AGX_FORCE_INLINE agx::UInt16 const& ParticleInstance::cellSlot() const { verifyIndex(); return getData()->cellSlot[getIndex()]; }

    AGX_FORCE_INLINE agx::UInt32& ParticleInstance::oldCell() { verifyIndex(); return getData()->oldCell[getIndex()]; }
    AGX_FORCE_INLINE agx::UInt32 const& ParticleInstance::oldCell() const { verifyIndex(); return getData()->oldCell[getIndex()]; }

    AGX_FORCE_INLINE agx::UInt32& ParticleInstance::geometryContactList() { verifyIndex(); return getData()->geometryContactList[getIndex()]; }
    AGX_FORCE_INLINE agx::UInt32 const& ParticleInstance::geometryContactList() const { verifyIndex(); return getData()->geometryContactList[getIndex()]; }

    AGX_FORCE_INLINE agx::UInt32& ParticleInstance::solveBodyIndex() { verifyIndex(); return getData()->solveBodyIndex[getIndex()]; }
    AGX_FORCE_INLINE agx::UInt32 const& ParticleInstance::solveBodyIndex() const { verifyIndex(); return getData()->solveBodyIndex[getIndex()]; }

    AGX_FORCE_INLINE agx::Physics::GraphNodePtr& ParticleInstance::graphNode() { verifyIndex(); return getData()->graphNode[getIndex()]; }
    AGX_FORCE_INLINE agx::Physics::GraphNodePtr const& ParticleInstance::graphNode() const { verifyIndex(); return getData()->graphNode[getIndex()]; }

    AGX_FORCE_INLINE agx::Vec4f& ParticleInstance::color() { verifyIndex(); return getData()->color[getIndex()]; }
    AGX_FORCE_INLINE agx::Vec4f const& ParticleInstance::color() const { verifyIndex(); return getData()->color[getIndex()]; }

    AGX_FORCE_INLINE agx::Vec3& ParticleInstance::geometryContactVector() { verifyIndex(); return getData()->geometryContactVector[getIndex()]; }
    AGX_FORCE_INLINE agx::Vec3 const& ParticleInstance::geometryContactVector() const { verifyIndex(); return getData()->geometryContactVector[getIndex()]; }

    AGX_FORCE_INLINE agx::Real32& ParticleInstance::enableRendering() { verifyIndex(); return getData()->enableRendering[getIndex()]; }
    AGX_FORCE_INLINE agx::Real32 const& ParticleInstance::enableRendering() const { verifyIndex(); return getData()->enableRendering[getIndex()]; }

    //-----------------------------------------------------------------------------------------------------
    AGX_FORCE_INLINE ParticleSemantics::ParticleSemantics() {}
    //-----------------------------------------------------------------------------------------------------
    DOXYGEN_END_INTERNAL_BLOCK()
  }
}

AGX_TYPE_BINDING(agx::Physics::ParticlePtr, "Physics.ParticlePtr")
AGX_TYPE_BINDING(agx::Physics::ParticleInstance, "Physics.ParticleInstance")

#ifdef _MSC_VER
# pragma warning(pop)
#endif

#undef AGX_ENTITY_WRAPPER
#undef AGX_ENTITY_NAMESPACE
#endif

