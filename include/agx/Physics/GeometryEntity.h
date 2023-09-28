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

#ifndef GENERATED_AGX_PHYSICS_GEOMETRY_H_PLUGIN
#define GENERATED_AGX_PHYSICS_GEOMETRY_H_PLUGIN

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
#include <agx/ReferencedEntity.h>
#include <agx/Name.h>
#include <agxCollide/GeometryState.h>
#include <agx/Physics/Geometry/ShapeEntity.h>
#include <agx/AffineMatrix4x4.h>
#include <agxCollide/BoundingAABB.h>
#include <agx/Real.h>
#include <agx/Vec3.h>
#include <agx/Integer.h>
#include <agx/Physics/CollisionGroupSetEntity.h>
#include <agx/Physics/RigidBodyEntity.h>
#include <agx/Physics/MaterialEntity.h>
namespace agxCollide { class Geometry; }

namespace agx { namespace Physics { namespace Geometry { class ShapePtr; }}}
namespace agx { namespace Physics { class CollisionGroupSetPtr; }}
namespace agx { namespace Physics { class RigidBodyPtr; }}
namespace agx { namespace Physics { class MaterialPtr; }}
namespace agx { namespace Physics { class GeometryPtr; }}

namespace agx
{
  namespace Physics
  {

    class GeometryModel;
    class GeometryData;
    class GeometryPtr;
    class GeometryInstance;
    class GeometrySemantics;


    AGX_DECLARE_POINTER_TYPES(GeometryModel);

    /** 
    Abstract description of the data attributes for the Physics.Geometry entity.
    */ 
    class AGXPHYSICS_EXPORT GeometryModel : public agx::ReferencedModel
    {
    public:
      typedef GeometryPtr PtrT;

      GeometryModel(const agx::String& name = "Geometry");

      /// \return The entity model singleton.
      static GeometryModel* instance();

      /// Create and return a pointer to a new instance in the default storage for this entity model.
      static GeometryPtr createInstance();

      /// \return The default storage for this entity model.
      static agxData::EntityStorage* defaultStorage();

      /// This is part of internal cleanup and should not be called by users
      virtual void shutdownCleanup() override;



      /* Attributes */
      static agxData::ScalarAttributeT< agx::Name >* nameAttribute;
      static agxData::ScalarAttributeT< agxCollide::GeometryState >* stateAttribute;
      static agxData::ScalarAttributeT< agx::Physics::Geometry::ShapePtr >* shapeAttribute;
      static agxData::ScalarAttributeT< agx::AffineMatrix4x4 >* transformAttribute;
      static agxData::ScalarAttributeT< agx::AffineMatrix4x4 >* localTransformAttribute;
      static agxData::ScalarAttributeT< agxCollide::BoundingAABB >* boundingAABBAttribute;
      static agxData::ScalarAttributeT< agx::Real >* boundingRadiusAttribute;
      static agxData::ScalarAttributeT< agx::Vec3 >* localBoundCenterAttribute;
      static agxData::ScalarAttributeT< agx::UInt8 >* tierAttribute;
      static agxData::ScalarAttributeT< agx::UInt32 >* cellIndexAttribute;
      static agxData::ScalarAttributeT< agx::UInt16 >* cellSlotAttribute;
      static agxData::ScalarAttributeT< agx::UInt32 >* oldCellAttribute;
      static agxData::ScalarAttributeT< agx::UInt >* orientedBoundIdAttribute;
      static agxData::ScalarAttributeT< agx::Physics::CollisionGroupSetPtr >* collisionGroupSetAttribute;
      static agxData::ScalarAttributeT< agx::UInt32 >* sweepAndPruneIndexAttribute;
      static agxData::ScalarAttributeT< agx::Physics::RigidBodyPtr >* bodyAttribute;
      static agxData::ScalarAttributeT< agx::Vec3f >* surfaceVelocityAttribute;
      static agxData::ScalarAttributeT< agx::Physics::MaterialPtr >* materialAttribute;
      static agxData::PointerAttributeT< agxCollide::Geometry*>* modelAttribute;
      static agxData::ScalarAttributeT< agx::Physics::GeometryPtr >* nextAttribute;

    protected:
      virtual ~GeometryModel();
      virtual agxData::EntityData* createData(agxData::EntityStorage* storage) override;
      virtual void configure(agx::TiXmlElement* eEntity) override;
      virtual void initAttributeAccessors() override;
      virtual void construct(agxData::EntityPtr instance) override;
      void construct(agx::Physics::GeometryPtr geometry);
    };


    DOXYGEN_START_INTERNAL_BLOCK()
    #ifndef AGX_PHYSICS_GEOMETRY_DATA_SET_OVERRIDE
    #define AGX_PHYSICS_GEOMETRY_DATA_SET
    class AGXPHYSICS_EXPORT GeometryData : public agx::ReferencedData
    {
    public:
      GeometryInstance operator[] (size_t index);

    public:
      agxData::Array< GeometryPtr >& instance;
      agxData::Array< agx::Name > name;
      agxData::Array< agxCollide::GeometryState > state;
      agxData::Array< agx::Physics::Geometry::ShapePtr > shape;
      agxData::Array< agx::AffineMatrix4x4 > transform;
      agxData::Array< agx::AffineMatrix4x4 > localTransform;
      agxData::Array< agxCollide::BoundingAABB > boundingAABB;
      agxData::Array< agx::Real > boundingRadius;
      agxData::Array< agx::Vec3 > localBoundCenter;
      agxData::Array< agx::UInt8 > tier;
      agxData::Array< agx::UInt32 > cellIndex;
      agxData::Array< agx::UInt16 > cellSlot;
      agxData::Array< agx::UInt32 > oldCell;
      agxData::Array< agx::UInt > orientedBoundId;
      agxData::Array< agx::Physics::CollisionGroupSetPtr > collisionGroupSet;
      agxData::Array< agx::UInt32 > sweepAndPruneIndex;
      agxData::Array< agx::Physics::RigidBodyPtr > body;
      agxData::Array< agx::Vec3f > surfaceVelocity;
      agxData::Array< agx::Physics::MaterialPtr > material;
      agxData::Array< agxCollide::Geometry* > model;
      agxData::Array< agx::Physics::GeometryPtr > next;

    public:
      typedef agx::Name nameType;
      typedef agxCollide::GeometryState stateType;
      typedef agx::Physics::Geometry::ShapePtr shapeType;
      typedef agx::AffineMatrix4x4 transformType;
      typedef agx::AffineMatrix4x4 localTransformType;
      typedef agxCollide::BoundingAABB boundingAABBType;
      typedef agx::Real boundingRadiusType;
      typedef agx::Vec3 localBoundCenterType;
      typedef agx::UInt8 tierType;
      typedef agx::UInt32 cellIndexType;
      typedef agx::UInt16 cellSlotType;
      typedef agx::UInt32 oldCellType;
      typedef agx::UInt orientedBoundIdType;
      typedef agx::Physics::CollisionGroupSetPtr collisionGroupSetType;
      typedef agx::UInt32 sweepAndPruneIndexType;
      typedef agx::Physics::RigidBodyPtr bodyType;
      typedef agx::Vec3f surfaceVelocityType;
      typedef agx::Physics::MaterialPtr materialType;
      typedef agxCollide::Geometry* modelType;
      typedef agx::Physics::GeometryPtr nextType;

    public:
      GeometryData(agxData::EntityStorage* storage);
      GeometryData();

    protected:
      virtual ~GeometryData() {}
      virtual void setNumElements(agx::Index numElements) override;

    private:
      GeometryData& operator= (const GeometryData&) { return *this; }

    };
    #endif
    DOXYGEN_END_INTERNAL_BLOCK()


    DOXYGEN_START_INTERNAL_BLOCK()
    class AGXPHYSICS_EXPORT GeometrySemantics : public agx::ReferencedSemantics
    {
    public:

      // Automatic getters
      agx::Name const& getName() const;
      agxCollide::GeometryState const& getState() const;
      agx::Physics::Geometry::ShapePtr const& getShape() const;
      agx::AffineMatrix4x4 const& getTransform() const;
      agx::AffineMatrix4x4 const& getLocalTransform() const;
      agxCollide::BoundingAABB const& getBoundingAABB() const;
      agx::Real const& getBoundingRadius() const;
      agx::Vec3 const& getLocalBoundCenter() const;
      agx::UInt8 const& getTier() const;
      agx::UInt32 const& getCellIndex() const;
      agx::UInt16 const& getCellSlot() const;
      agx::UInt32 const& getOldCell() const;
      agx::UInt const& getOrientedBoundId() const;
      agx::Physics::CollisionGroupSetPtr const& getCollisionGroupSet() const;
      agx::UInt32 const& getSweepAndPruneIndex() const;
      agx::Physics::RigidBodyPtr const& getBody() const;
      agx::Vec3f const& getSurfaceVelocity() const;
      agx::Physics::MaterialPtr const& getMaterial() const;
      agxCollide::Geometry* const& getModel() const;
      agx::Physics::GeometryPtr const& getNext() const;

      // Semantics defined by explicit kernels

      // Automatic setters
      void setName(agx::Name const& value);
      void setState(agxCollide::GeometryState const& value);
      void setShape(agx::Physics::Geometry::ShapePtr const& value);
      void setTransform(agx::AffineMatrix4x4 const& value);
      void setLocalTransform(agx::AffineMatrix4x4 const& value);
      void setBoundingAABB(agxCollide::BoundingAABB const& value);
      void setBoundingRadius(agx::Real const& value);
      void setLocalBoundCenter(agx::Vec3 const& value);
      void setTier(agx::UInt8 const& value);
      void setCellIndex(agx::UInt32 const& value);
      void setCellSlot(agx::UInt16 const& value);
      void setOldCell(agx::UInt32 const& value);
      void setOrientedBoundId(agx::UInt const& value);
      void setCollisionGroupSet(agx::Physics::CollisionGroupSetPtr const& value);
      void setSweepAndPruneIndex(agx::UInt32 const& value);
      void setBody(agx::Physics::RigidBodyPtr const& value);
      void setSurfaceVelocity(agx::Vec3f const& value);
      void setMaterial(agx::Physics::MaterialPtr const& value);
      void setModel(agxCollide::Geometry* const& value);
      void setNext(agx::Physics::GeometryPtr const& value);


    protected:
      friend class GeometryPtr;
      friend class GeometryInstance;
      GeometrySemantics();
    };
    DOXYGEN_END_INTERNAL_BLOCK()


    /**
    Pointer to a entity instance of type Physics.Geometry
    */
    class CALLABLE GeometryPtr : public agx::ReferencedPtr
    {
    public:
      typedef GeometryModel ModelType;
      typedef GeometryData DataType;
      typedef GeometryInstance InstanceType;

    public:
      AGXPHYSICS_EXPORT GeometryPtr();
      AGXPHYSICS_EXPORT GeometryPtr(agxData::EntityStorage* storage, agx::Index id);
      AGXPHYSICS_EXPORT GeometryPtr(const agxData::EntityPtr& ptr);
      AGXPHYSICS_EXPORT GeometryPtr(const agxData::EntityInstance& instance);
      AGXPHYSICS_EXPORT GeometryPtr& operator= (const agxData::EntityPtr& ptr);
      AGXPHYSICS_EXPORT GeometryPtr& operator= (const agxData::EntityInstance& instance);
      AGXPHYSICS_EXPORT GeometryInstance instance();
      AGXPHYSICS_EXPORT const GeometryInstance instance() const;

      AGXPHYSICS_EXPORT GeometrySemantics* operator->();
      AGXPHYSICS_EXPORT const GeometrySemantics* operator->() const;

      GeometryData* getData();
      const GeometryData* getData() const;


      /// \return reference to the name attribute
      AGXPHYSICS_EXPORT agx::Name& name();
      /// \return const reference to the name attribute
      AGXPHYSICS_EXPORT agx::Name const& name() const;

      /// \return reference to the state attribute
      AGXPHYSICS_EXPORT agxCollide::GeometryState& state();
      /// \return const reference to the state attribute
      AGXPHYSICS_EXPORT agxCollide::GeometryState const& state() const;

      /// \return reference to the shape attribute
      AGXPHYSICS_EXPORT agx::Physics::Geometry::ShapePtr& shape();
      /// \return const reference to the shape attribute
      AGXPHYSICS_EXPORT agx::Physics::Geometry::ShapePtr const& shape() const;

      /// \return reference to the transform attribute
      AGXPHYSICS_EXPORT agx::AffineMatrix4x4& transform();
      /// \return const reference to the transform attribute
      AGXPHYSICS_EXPORT agx::AffineMatrix4x4 const& transform() const;

      /// \return reference to the localTransform attribute
      AGXPHYSICS_EXPORT agx::AffineMatrix4x4& localTransform();
      /// \return const reference to the localTransform attribute
      AGXPHYSICS_EXPORT agx::AffineMatrix4x4 const& localTransform() const;

      /// \return reference to the boundingAABB attribute
      AGXPHYSICS_EXPORT agxCollide::BoundingAABB& boundingAABB();
      /// \return const reference to the boundingAABB attribute
      AGXPHYSICS_EXPORT agxCollide::BoundingAABB const& boundingAABB() const;

      /// \return reference to the boundingRadius attribute
      AGXPHYSICS_EXPORT agx::Real& boundingRadius();
      /// \return const reference to the boundingRadius attribute
      AGXPHYSICS_EXPORT agx::Real const& boundingRadius() const;

      /// \return reference to the localBoundCenter attribute
      AGXPHYSICS_EXPORT agx::Vec3& localBoundCenter();
      /// \return const reference to the localBoundCenter attribute
      AGXPHYSICS_EXPORT agx::Vec3 const& localBoundCenter() const;

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

      /// \return reference to the orientedBoundId attribute
      AGXPHYSICS_EXPORT agx::UInt& orientedBoundId();
      /// \return const reference to the orientedBoundId attribute
      AGXPHYSICS_EXPORT agx::UInt const& orientedBoundId() const;

      /// \return reference to the collisionGroupSet attribute
      AGXPHYSICS_EXPORT agx::Physics::CollisionGroupSetPtr& collisionGroupSet();
      /// \return const reference to the collisionGroupSet attribute
      AGXPHYSICS_EXPORT agx::Physics::CollisionGroupSetPtr const& collisionGroupSet() const;

      /// \return reference to the sweepAndPruneIndex attribute
      AGXPHYSICS_EXPORT agx::UInt32& sweepAndPruneIndex();
      /// \return const reference to the sweepAndPruneIndex attribute
      AGXPHYSICS_EXPORT agx::UInt32 const& sweepAndPruneIndex() const;

      /// \return reference to the body attribute
      AGXPHYSICS_EXPORT agx::Physics::RigidBodyPtr& body();
      /// \return const reference to the body attribute
      AGXPHYSICS_EXPORT agx::Physics::RigidBodyPtr const& body() const;

      /// \return reference to the surfaceVelocity attribute
      AGXPHYSICS_EXPORT agx::Vec3f& surfaceVelocity();
      /// \return const reference to the surfaceVelocity attribute
      AGXPHYSICS_EXPORT agx::Vec3f const& surfaceVelocity() const;

      /// \return reference to the material attribute
      AGXPHYSICS_EXPORT agx::Physics::MaterialPtr& material();
      /// \return const reference to the material attribute
      AGXPHYSICS_EXPORT agx::Physics::MaterialPtr const& material() const;

      /// \return reference to the model attribute
      AGXPHYSICS_EXPORT agxCollide::Geometry*& model();
      /// \return const reference to the model attribute
      AGXPHYSICS_EXPORT agxCollide::Geometry* const& model() const;

      /// \return reference to the next attribute
      AGXPHYSICS_EXPORT agx::Physics::GeometryPtr& next();
      /// \return const reference to the next attribute
      AGXPHYSICS_EXPORT agx::Physics::GeometryPtr const& next() const;

      DOXYGEN_START_INTERNAL_BLOCK()
      // Only used by tolua!
      #ifdef AGX_GENERATED_TOLUA_WRAPPER
      inline agx::Name const& getName() const { return (*this)->getName(); }
      inline agxCollide::GeometryState const& getState() const { return (*this)->getState(); }
      inline agx::Physics::Geometry::ShapePtr const& getShape() const { return (*this)->getShape(); }
      inline agx::AffineMatrix4x4 const& getTransform() const { return (*this)->getTransform(); }
      inline agx::AffineMatrix4x4 const& getLocalTransform() const { return (*this)->getLocalTransform(); }
      inline agxCollide::BoundingAABB const& getBoundingAABB() const { return (*this)->getBoundingAABB(); }
      inline agx::Real const& getBoundingRadius() const { return (*this)->getBoundingRadius(); }
      inline agx::Vec3 const& getLocalBoundCenter() const { return (*this)->getLocalBoundCenter(); }
      inline agx::UInt8 const& getTier() const { return (*this)->getTier(); }
      inline agx::UInt32 const& getCellIndex() const { return (*this)->getCellIndex(); }
      inline agx::UInt16 const& getCellSlot() const { return (*this)->getCellSlot(); }
      inline agx::UInt32 const& getOldCell() const { return (*this)->getOldCell(); }
      inline agx::UInt const& getOrientedBoundId() const { return (*this)->getOrientedBoundId(); }
      inline agx::Physics::CollisionGroupSetPtr const& getCollisionGroupSet() const { return (*this)->getCollisionGroupSet(); }
      inline agx::UInt32 const& getSweepAndPruneIndex() const { return (*this)->getSweepAndPruneIndex(); }
      inline agx::Physics::RigidBodyPtr const& getBody() const { return (*this)->getBody(); }
      inline agx::Vec3f const& getSurfaceVelocity() const { return (*this)->getSurfaceVelocity(); }
      inline agx::Physics::MaterialPtr const& getMaterial() const { return (*this)->getMaterial(); }
      inline agxCollide::Geometry* const& getModel() const { return (*this)->getModel(); }
      inline agx::Physics::GeometryPtr const& getNext() const { return (*this)->getNext(); }


      inline void setName(agx::Name const& value) { (*this)->setName(value); }
      inline void setState(agxCollide::GeometryState const& value) { (*this)->setState(value); }
      inline void setShape(agx::Physics::Geometry::ShapePtr const& value) { (*this)->setShape(value); }
      inline void setTransform(agx::AffineMatrix4x4 const& value) { (*this)->setTransform(value); }
      inline void setLocalTransform(agx::AffineMatrix4x4 const& value) { (*this)->setLocalTransform(value); }
      inline void setBoundingAABB(agxCollide::BoundingAABB const& value) { (*this)->setBoundingAABB(value); }
      inline void setBoundingRadius(agx::Real const& value) { (*this)->setBoundingRadius(value); }
      inline void setLocalBoundCenter(agx::Vec3 const& value) { (*this)->setLocalBoundCenter(value); }
      inline void setTier(agx::UInt8 const& value) { (*this)->setTier(value); }
      inline void setCellIndex(agx::UInt32 const& value) { (*this)->setCellIndex(value); }
      inline void setCellSlot(agx::UInt16 const& value) { (*this)->setCellSlot(value); }
      inline void setOldCell(agx::UInt32 const& value) { (*this)->setOldCell(value); }
      inline void setOrientedBoundId(agx::UInt const& value) { (*this)->setOrientedBoundId(value); }
      inline void setCollisionGroupSet(agx::Physics::CollisionGroupSetPtr const& value) { (*this)->setCollisionGroupSet(value); }
      inline void setSweepAndPruneIndex(agx::UInt32 const& value) { (*this)->setSweepAndPruneIndex(value); }
      inline void setBody(agx::Physics::RigidBodyPtr const& value) { (*this)->setBody(value); }
      inline void setSurfaceVelocity(agx::Vec3f const& value) { (*this)->setSurfaceVelocity(value); }
      inline void setMaterial(agx::Physics::MaterialPtr const& value) { (*this)->setMaterial(value); }
      inline void setModel(agxCollide::Geometry* const& value) { (*this)->setModel(value); }
      inline void setNext(agx::Physics::GeometryPtr const& value) { (*this)->setNext(value); }
      #endif
      DOXYGEN_END_INTERNAL_BLOCK()

    };

    // Entity is Referenced
    typedef agxData::EntityRef< GeometryPtr > GeometryRef;


    DOXYGEN_START_INTERNAL_BLOCK()
    class AGXPHYSICS_EXPORT GeometryInstance : public agx::ReferencedInstance
    {
    public:
      GeometryInstance();
      GeometryInstance(GeometryData* data, agx::Index index);
      GeometryInstance(agxData::EntityStorage *storage, agx::Index index);
      GeometryInstance(const agxData::EntityInstance& other);
      GeometryInstance(const agxData::EntityPtr& ptr);

      GeometryData* getData();
      const GeometryData* getData() const;

    public:
      /// \return reference to the name attribute
      agx::Name& name();
      /// \return const reference to the name attribute
      agx::Name const& name() const;

      /// \return reference to the state attribute
      agxCollide::GeometryState& state();
      /// \return const reference to the state attribute
      agxCollide::GeometryState const& state() const;

      /// \return reference to the shape attribute
      agx::Physics::Geometry::ShapePtr& shape();
      /// \return const reference to the shape attribute
      agx::Physics::Geometry::ShapePtr const& shape() const;

      /// \return reference to the transform attribute
      agx::AffineMatrix4x4& transform();
      /// \return const reference to the transform attribute
      agx::AffineMatrix4x4 const& transform() const;

      /// \return reference to the localTransform attribute
      agx::AffineMatrix4x4& localTransform();
      /// \return const reference to the localTransform attribute
      agx::AffineMatrix4x4 const& localTransform() const;

      /// \return reference to the boundingAABB attribute
      agxCollide::BoundingAABB& boundingAABB();
      /// \return const reference to the boundingAABB attribute
      agxCollide::BoundingAABB const& boundingAABB() const;

      /// \return reference to the boundingRadius attribute
      agx::Real& boundingRadius();
      /// \return const reference to the boundingRadius attribute
      agx::Real const& boundingRadius() const;

      /// \return reference to the localBoundCenter attribute
      agx::Vec3& localBoundCenter();
      /// \return const reference to the localBoundCenter attribute
      agx::Vec3 const& localBoundCenter() const;

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

      /// \return reference to the orientedBoundId attribute
      agx::UInt& orientedBoundId();
      /// \return const reference to the orientedBoundId attribute
      agx::UInt const& orientedBoundId() const;

      /// \return reference to the collisionGroupSet attribute
      agx::Physics::CollisionGroupSetPtr& collisionGroupSet();
      /// \return const reference to the collisionGroupSet attribute
      agx::Physics::CollisionGroupSetPtr const& collisionGroupSet() const;

      /// \return reference to the sweepAndPruneIndex attribute
      agx::UInt32& sweepAndPruneIndex();
      /// \return const reference to the sweepAndPruneIndex attribute
      agx::UInt32 const& sweepAndPruneIndex() const;

      /// \return reference to the body attribute
      agx::Physics::RigidBodyPtr& body();
      /// \return const reference to the body attribute
      agx::Physics::RigidBodyPtr const& body() const;

      /// \return reference to the surfaceVelocity attribute
      agx::Vec3f& surfaceVelocity();
      /// \return const reference to the surfaceVelocity attribute
      agx::Vec3f const& surfaceVelocity() const;

      /// \return reference to the material attribute
      agx::Physics::MaterialPtr& material();
      /// \return const reference to the material attribute
      agx::Physics::MaterialPtr const& material() const;

      /// \return reference to the model attribute
      agxCollide::Geometry*& model();
      /// \return const reference to the model attribute
      agxCollide::Geometry* const& model() const;

      /// \return reference to the next attribute
      agx::Physics::GeometryPtr& next();
      /// \return const reference to the next attribute
      agx::Physics::GeometryPtr const& next() const;

    };
    DOXYGEN_END_INTERNAL_BLOCK()



    typedef agx::VectorPOD<GeometryPtr> GeometryPtrVector;
    typedef agxData::Array<GeometryPtr> GeometryPtrArray;



    DOXYGEN_START_INTERNAL_BLOCK()
    /* Implementation */
    //-----------------------------------------------------------------------------------------------------
    //-----------------------------------------------------------------------------------------------------
    inline GeometryInstance agx::Physics::GeometryData::operator[] (size_t index) { return GeometryInstance(this, (agx::Index)index); }
    //-----------------------------------------------------------------------------------------------------
    AGX_FORCE_INLINE GeometryPtr::GeometryPtr() {}
    AGX_FORCE_INLINE GeometryPtr::GeometryPtr(agxData::EntityStorage* storage, agx::Index id) : agx::ReferencedPtr(storage, id) {}
    AGX_FORCE_INLINE GeometryPtr::GeometryPtr(const agxData::EntityPtr& ptr) : agx::ReferencedPtr(ptr)
    {
      agxAssertN(!ptr || ptr.isInstanceOf(GeometryModel::instance()),
        "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
        EntityPtr::getModel()->fullPath().c_str(), GeometryModel::instance()->fullPath().c_str());
    }

    AGX_FORCE_INLINE GeometryPtr::GeometryPtr(const agxData::EntityInstance& instance) : agx::ReferencedPtr(instance)
    {
      agxAssertN(!instance || instance.isInstanceOf(GeometryModel::instance()),
        "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
        EntityPtr::getModel()->fullPath().c_str(), GeometryModel::instance()->fullPath().c_str());
    }

    AGX_FORCE_INLINE GeometryPtr& GeometryPtr::operator= (const agxData::EntityPtr& ptr)
    {
      agxData::EntityPtr::operator= (ptr);
      agxAssertN(!ptr || ptr.isInstanceOf(GeometryModel::instance()),
        "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
        EntityPtr::getModel()->fullPath().c_str(), GeometryModel::instance()->fullPath().c_str());
      return *this;
    }

    AGX_FORCE_INLINE GeometryPtr& GeometryPtr::operator= (const agxData::EntityInstance& instance)
    {
      agxData::EntityPtr::operator= (instance);
      agxAssertN(!instance || instance.isInstanceOf(GeometryModel::instance()),
        "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
        EntityPtr::getModel()->fullPath().c_str(), GeometryModel::instance()->fullPath().c_str());
      return *this;
    }

    inline GeometryInstance GeometryPtr::instance() { return agxData::EntityPtr::instance(); }
    inline const GeometryInstance GeometryPtr::instance() const { return agxData::EntityPtr::instance(); }
    AGX_FORCE_INLINE GeometrySemantics* GeometryPtr::operator->() { return (GeometrySemantics* )this; }
    AGX_FORCE_INLINE const GeometrySemantics* GeometryPtr::operator->() const { return (const GeometrySemantics* )this; }
    AGX_FORCE_INLINE GeometryData* GeometryPtr::getData() { return static_cast<GeometryData* >(agxData::EntityPtr::getData()); }
    AGX_FORCE_INLINE const GeometryData* GeometryPtr::getData() const { return static_cast<const GeometryData* >(agxData::EntityPtr::getData()); }

    AGX_FORCE_INLINE agx::Name& GeometryPtr::name() { verifyIndex(); return getData()->name[calculateIndex()]; }
    AGX_FORCE_INLINE agx::Name const& GeometryPtr::name() const { verifyIndex(); return getData()->name[calculateIndex()]; }

    AGX_FORCE_INLINE agxCollide::GeometryState& GeometryPtr::state() { verifyIndex(); return getData()->state[calculateIndex()]; }
    AGX_FORCE_INLINE agxCollide::GeometryState const& GeometryPtr::state() const { verifyIndex(); return getData()->state[calculateIndex()]; }

    AGX_FORCE_INLINE agx::Physics::Geometry::ShapePtr& GeometryPtr::shape() { verifyIndex(); return getData()->shape[calculateIndex()]; }
    AGX_FORCE_INLINE agx::Physics::Geometry::ShapePtr const& GeometryPtr::shape() const { verifyIndex(); return getData()->shape[calculateIndex()]; }

    AGX_FORCE_INLINE agx::AffineMatrix4x4& GeometryPtr::transform() { verifyIndex(); return getData()->transform[calculateIndex()]; }
    AGX_FORCE_INLINE agx::AffineMatrix4x4 const& GeometryPtr::transform() const { verifyIndex(); return getData()->transform[calculateIndex()]; }

    AGX_FORCE_INLINE agx::AffineMatrix4x4& GeometryPtr::localTransform() { verifyIndex(); return getData()->localTransform[calculateIndex()]; }
    AGX_FORCE_INLINE agx::AffineMatrix4x4 const& GeometryPtr::localTransform() const { verifyIndex(); return getData()->localTransform[calculateIndex()]; }

    AGX_FORCE_INLINE agxCollide::BoundingAABB& GeometryPtr::boundingAABB() { verifyIndex(); return getData()->boundingAABB[calculateIndex()]; }
    AGX_FORCE_INLINE agxCollide::BoundingAABB const& GeometryPtr::boundingAABB() const { verifyIndex(); return getData()->boundingAABB[calculateIndex()]; }

    AGX_FORCE_INLINE agx::Real& GeometryPtr::boundingRadius() { verifyIndex(); return getData()->boundingRadius[calculateIndex()]; }
    AGX_FORCE_INLINE agx::Real const& GeometryPtr::boundingRadius() const { verifyIndex(); return getData()->boundingRadius[calculateIndex()]; }

    AGX_FORCE_INLINE agx::Vec3& GeometryPtr::localBoundCenter() { verifyIndex(); return getData()->localBoundCenter[calculateIndex()]; }
    AGX_FORCE_INLINE agx::Vec3 const& GeometryPtr::localBoundCenter() const { verifyIndex(); return getData()->localBoundCenter[calculateIndex()]; }

    AGX_FORCE_INLINE agx::UInt8& GeometryPtr::tier() { verifyIndex(); return getData()->tier[calculateIndex()]; }
    AGX_FORCE_INLINE agx::UInt8 const& GeometryPtr::tier() const { verifyIndex(); return getData()->tier[calculateIndex()]; }

    AGX_FORCE_INLINE agx::UInt32& GeometryPtr::cellIndex() { verifyIndex(); return getData()->cellIndex[calculateIndex()]; }
    AGX_FORCE_INLINE agx::UInt32 const& GeometryPtr::cellIndex() const { verifyIndex(); return getData()->cellIndex[calculateIndex()]; }

    AGX_FORCE_INLINE agx::UInt16& GeometryPtr::cellSlot() { verifyIndex(); return getData()->cellSlot[calculateIndex()]; }
    AGX_FORCE_INLINE agx::UInt16 const& GeometryPtr::cellSlot() const { verifyIndex(); return getData()->cellSlot[calculateIndex()]; }

    AGX_FORCE_INLINE agx::UInt32& GeometryPtr::oldCell() { verifyIndex(); return getData()->oldCell[calculateIndex()]; }
    AGX_FORCE_INLINE agx::UInt32 const& GeometryPtr::oldCell() const { verifyIndex(); return getData()->oldCell[calculateIndex()]; }

    AGX_FORCE_INLINE agx::UInt& GeometryPtr::orientedBoundId() { verifyIndex(); return getData()->orientedBoundId[calculateIndex()]; }
    AGX_FORCE_INLINE agx::UInt const& GeometryPtr::orientedBoundId() const { verifyIndex(); return getData()->orientedBoundId[calculateIndex()]; }

    AGX_FORCE_INLINE agx::Physics::CollisionGroupSetPtr& GeometryPtr::collisionGroupSet() { verifyIndex(); return getData()->collisionGroupSet[calculateIndex()]; }
    AGX_FORCE_INLINE agx::Physics::CollisionGroupSetPtr const& GeometryPtr::collisionGroupSet() const { verifyIndex(); return getData()->collisionGroupSet[calculateIndex()]; }

    AGX_FORCE_INLINE agx::UInt32& GeometryPtr::sweepAndPruneIndex() { verifyIndex(); return getData()->sweepAndPruneIndex[calculateIndex()]; }
    AGX_FORCE_INLINE agx::UInt32 const& GeometryPtr::sweepAndPruneIndex() const { verifyIndex(); return getData()->sweepAndPruneIndex[calculateIndex()]; }

    AGX_FORCE_INLINE agx::Physics::RigidBodyPtr& GeometryPtr::body() { verifyIndex(); return getData()->body[calculateIndex()]; }
    AGX_FORCE_INLINE agx::Physics::RigidBodyPtr const& GeometryPtr::body() const { verifyIndex(); return getData()->body[calculateIndex()]; }

    AGX_FORCE_INLINE agx::Vec3f& GeometryPtr::surfaceVelocity() { verifyIndex(); return getData()->surfaceVelocity[calculateIndex()]; }
    AGX_FORCE_INLINE agx::Vec3f const& GeometryPtr::surfaceVelocity() const { verifyIndex(); return getData()->surfaceVelocity[calculateIndex()]; }

    AGX_FORCE_INLINE agx::Physics::MaterialPtr& GeometryPtr::material() { verifyIndex(); return getData()->material[calculateIndex()]; }
    AGX_FORCE_INLINE agx::Physics::MaterialPtr const& GeometryPtr::material() const { verifyIndex(); return getData()->material[calculateIndex()]; }

    AGX_FORCE_INLINE agxCollide::Geometry*& GeometryPtr::model() { verifyIndex(); return getData()->model[calculateIndex()]; }
    AGX_FORCE_INLINE agxCollide::Geometry* const& GeometryPtr::model() const { verifyIndex(); return getData()->model[calculateIndex()]; }

    AGX_FORCE_INLINE agx::Physics::GeometryPtr& GeometryPtr::next() { verifyIndex(); return getData()->next[calculateIndex()]; }
    AGX_FORCE_INLINE agx::Physics::GeometryPtr const& GeometryPtr::next() const { verifyIndex(); return getData()->next[calculateIndex()]; }

    //-----------------------------------------------------------------------------------------------------
    AGX_FORCE_INLINE GeometryInstance::GeometryInstance() {}
    AGX_FORCE_INLINE GeometryInstance::GeometryInstance(GeometryData* data, agx::Index index) : agx::ReferencedInstance(data, index) {}
    AGX_FORCE_INLINE GeometryInstance::GeometryInstance(agxData::EntityStorage* storage, agx::Index index) : agx::ReferencedInstance(storage, index) {}
    AGX_FORCE_INLINE GeometryInstance::GeometryInstance(const agxData::EntityInstance& other) : agx::ReferencedInstance(other)
    {
      agxAssertN(!other || other.isInstanceOf(GeometryModel::instance()),
        "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
        EntityInstance::getModel()->fullPath().c_str(), GeometryModel::instance()->fullPath().c_str());
    }

    AGX_FORCE_INLINE GeometryInstance::GeometryInstance(const agxData::EntityPtr& ptr) : agx::ReferencedInstance(ptr)
    {
      agxAssertN(!ptr || ptr.isInstanceOf(GeometryModel::instance()),
        "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
        EntityInstance::getModel()->fullPath().c_str(), GeometryModel::instance()->fullPath().c_str());
    }


    AGX_FORCE_INLINE GeometryData* GeometryInstance::getData() { return static_cast<GeometryData* >(agxData::EntityInstance::getData()); }
    AGX_FORCE_INLINE const GeometryData* GeometryInstance::getData() const { return static_cast<const GeometryData* >(agxData::EntityInstance::getData()); }

    AGX_FORCE_INLINE agx::Name& GeometryInstance::name() { verifyIndex(); return getData()->name[getIndex()]; }
    AGX_FORCE_INLINE agx::Name const& GeometryInstance::name() const { verifyIndex(); return getData()->name[getIndex()]; }

    AGX_FORCE_INLINE agxCollide::GeometryState& GeometryInstance::state() { verifyIndex(); return getData()->state[getIndex()]; }
    AGX_FORCE_INLINE agxCollide::GeometryState const& GeometryInstance::state() const { verifyIndex(); return getData()->state[getIndex()]; }

    AGX_FORCE_INLINE agx::Physics::Geometry::ShapePtr& GeometryInstance::shape() { verifyIndex(); return getData()->shape[getIndex()]; }
    AGX_FORCE_INLINE agx::Physics::Geometry::ShapePtr const& GeometryInstance::shape() const { verifyIndex(); return getData()->shape[getIndex()]; }

    AGX_FORCE_INLINE agx::AffineMatrix4x4& GeometryInstance::transform() { verifyIndex(); return getData()->transform[getIndex()]; }
    AGX_FORCE_INLINE agx::AffineMatrix4x4 const& GeometryInstance::transform() const { verifyIndex(); return getData()->transform[getIndex()]; }

    AGX_FORCE_INLINE agx::AffineMatrix4x4& GeometryInstance::localTransform() { verifyIndex(); return getData()->localTransform[getIndex()]; }
    AGX_FORCE_INLINE agx::AffineMatrix4x4 const& GeometryInstance::localTransform() const { verifyIndex(); return getData()->localTransform[getIndex()]; }

    AGX_FORCE_INLINE agxCollide::BoundingAABB& GeometryInstance::boundingAABB() { verifyIndex(); return getData()->boundingAABB[getIndex()]; }
    AGX_FORCE_INLINE agxCollide::BoundingAABB const& GeometryInstance::boundingAABB() const { verifyIndex(); return getData()->boundingAABB[getIndex()]; }

    AGX_FORCE_INLINE agx::Real& GeometryInstance::boundingRadius() { verifyIndex(); return getData()->boundingRadius[getIndex()]; }
    AGX_FORCE_INLINE agx::Real const& GeometryInstance::boundingRadius() const { verifyIndex(); return getData()->boundingRadius[getIndex()]; }

    AGX_FORCE_INLINE agx::Vec3& GeometryInstance::localBoundCenter() { verifyIndex(); return getData()->localBoundCenter[getIndex()]; }
    AGX_FORCE_INLINE agx::Vec3 const& GeometryInstance::localBoundCenter() const { verifyIndex(); return getData()->localBoundCenter[getIndex()]; }

    AGX_FORCE_INLINE agx::UInt8& GeometryInstance::tier() { verifyIndex(); return getData()->tier[getIndex()]; }
    AGX_FORCE_INLINE agx::UInt8 const& GeometryInstance::tier() const { verifyIndex(); return getData()->tier[getIndex()]; }

    AGX_FORCE_INLINE agx::UInt32& GeometryInstance::cellIndex() { verifyIndex(); return getData()->cellIndex[getIndex()]; }
    AGX_FORCE_INLINE agx::UInt32 const& GeometryInstance::cellIndex() const { verifyIndex(); return getData()->cellIndex[getIndex()]; }

    AGX_FORCE_INLINE agx::UInt16& GeometryInstance::cellSlot() { verifyIndex(); return getData()->cellSlot[getIndex()]; }
    AGX_FORCE_INLINE agx::UInt16 const& GeometryInstance::cellSlot() const { verifyIndex(); return getData()->cellSlot[getIndex()]; }

    AGX_FORCE_INLINE agx::UInt32& GeometryInstance::oldCell() { verifyIndex(); return getData()->oldCell[getIndex()]; }
    AGX_FORCE_INLINE agx::UInt32 const& GeometryInstance::oldCell() const { verifyIndex(); return getData()->oldCell[getIndex()]; }

    AGX_FORCE_INLINE agx::UInt& GeometryInstance::orientedBoundId() { verifyIndex(); return getData()->orientedBoundId[getIndex()]; }
    AGX_FORCE_INLINE agx::UInt const& GeometryInstance::orientedBoundId() const { verifyIndex(); return getData()->orientedBoundId[getIndex()]; }

    AGX_FORCE_INLINE agx::Physics::CollisionGroupSetPtr& GeometryInstance::collisionGroupSet() { verifyIndex(); return getData()->collisionGroupSet[getIndex()]; }
    AGX_FORCE_INLINE agx::Physics::CollisionGroupSetPtr const& GeometryInstance::collisionGroupSet() const { verifyIndex(); return getData()->collisionGroupSet[getIndex()]; }

    AGX_FORCE_INLINE agx::UInt32& GeometryInstance::sweepAndPruneIndex() { verifyIndex(); return getData()->sweepAndPruneIndex[getIndex()]; }
    AGX_FORCE_INLINE agx::UInt32 const& GeometryInstance::sweepAndPruneIndex() const { verifyIndex(); return getData()->sweepAndPruneIndex[getIndex()]; }

    AGX_FORCE_INLINE agx::Physics::RigidBodyPtr& GeometryInstance::body() { verifyIndex(); return getData()->body[getIndex()]; }
    AGX_FORCE_INLINE agx::Physics::RigidBodyPtr const& GeometryInstance::body() const { verifyIndex(); return getData()->body[getIndex()]; }

    AGX_FORCE_INLINE agx::Vec3f& GeometryInstance::surfaceVelocity() { verifyIndex(); return getData()->surfaceVelocity[getIndex()]; }
    AGX_FORCE_INLINE agx::Vec3f const& GeometryInstance::surfaceVelocity() const { verifyIndex(); return getData()->surfaceVelocity[getIndex()]; }

    AGX_FORCE_INLINE agx::Physics::MaterialPtr& GeometryInstance::material() { verifyIndex(); return getData()->material[getIndex()]; }
    AGX_FORCE_INLINE agx::Physics::MaterialPtr const& GeometryInstance::material() const { verifyIndex(); return getData()->material[getIndex()]; }

    AGX_FORCE_INLINE agxCollide::Geometry*& GeometryInstance::model() { verifyIndex(); return getData()->model[getIndex()]; }
    AGX_FORCE_INLINE agxCollide::Geometry* const& GeometryInstance::model() const { verifyIndex(); return getData()->model[getIndex()]; }

    AGX_FORCE_INLINE agx::Physics::GeometryPtr& GeometryInstance::next() { verifyIndex(); return getData()->next[getIndex()]; }
    AGX_FORCE_INLINE agx::Physics::GeometryPtr const& GeometryInstance::next() const { verifyIndex(); return getData()->next[getIndex()]; }

    //-----------------------------------------------------------------------------------------------------
    AGX_FORCE_INLINE GeometrySemantics::GeometrySemantics() {}
    //-----------------------------------------------------------------------------------------------------
    DOXYGEN_END_INTERNAL_BLOCK()
  }
}

AGX_TYPE_BINDING(agx::Physics::GeometryPtr, "Physics.GeometryPtr")
AGX_TYPE_BINDING(agx::Physics::GeometryInstance, "Physics.GeometryInstance")

#ifdef _MSC_VER
# pragma warning(pop)
#endif

#undef AGX_ENTITY_WRAPPER
#undef AGX_ENTITY_NAMESPACE
#endif

