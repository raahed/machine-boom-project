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

#ifndef GENERATED_AGX_PHYSICS_CONTACTPOINT_H_PLUGIN
#define GENERATED_AGX_PHYSICS_CONTACTPOINT_H_PLUGIN

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
#include <agx/Real.h>
#include <agx/Integer.h>
#include <agx/Vec4.h>
#include <agx/Physics/Geometry/ShapeEntity.h>

namespace agx { namespace Physics { namespace Geometry { class ShapePtr; }}}
namespace agx { namespace Physics { namespace Geometry { class ShapePtr; }}}

namespace agx
{
  namespace Physics
  {

    class ContactPointModel;
    class ContactPointData;
    class ContactPointPtr;
    class ContactPointInstance;
    class ContactPointSemantics;


    AGX_DECLARE_POINTER_TYPES(ContactPointModel);

    /** 
    Abstract description of the data attributes for the Physics.ContactPoint entity.
    */ 
    class AGXPHYSICS_EXPORT ContactPointModel : public agxData::EntityModel
    {
    public:
      typedef ContactPointPtr PtrT;

      ContactPointModel(const agx::String& name = "ContactPoint");

      /// \return The entity model singleton.
      static ContactPointModel* instance();

      /// Create and return a pointer to a new instance in the default storage for this entity model.
      static ContactPointPtr createInstance();

      /// \return The default storage for this entity model.
      static agxData::EntityStorage* defaultStorage();

      /// This is part of internal cleanup and should not be called by users
      virtual void shutdownCleanup() override;



      /* Attributes */
      static agxData::ScalarAttributeT< agx::Vec3 >* pointAttribute;
      static agxData::ScalarAttributeT< agx::Vec3f >* normalAttribute;
      static agxData::ScalarAttributeT< agx::Vec3f >* tangentUAttribute;
      static agxData::ScalarAttributeT< agx::Vec3f >* tangentVAttribute;
      static agxData::ScalarAttributeT< agx::Real >* depthAttribute;
      static agxData::ScalarAttributeT< agx::UInt8 >* stateAttribute;
      static agxData::ScalarAttributeT< agx::Vec3f >* velocityAttribute;
      static agxData::ScalarAttributeT< agx::Vec3 >* localForceAttribute;
      static agxData::ScalarAttributeT< agx::Vec4i8 >* indexSetAttribute;
      static agxData::ScalarAttributeT< agx::Real >* maxNormalForceAttribute;
      static agxData::ScalarAttributeT< agx::Bool >* enabledAttribute;
      static agxData::ScalarAttributeT< agx::UInt32 >* faceIndex1Attribute;
      static agxData::ScalarAttributeT< agx::UInt32 >* faceIndex2Attribute;
      static agxData::ScalarAttributeT< agx::UInt8 >* faceFeature1Attribute;
      static agxData::ScalarAttributeT< agx::UInt8 >* faceFeature2Attribute;
      static agxData::ScalarAttributeT< agx::Physics::Geometry::ShapePtr >* shape1Attribute;
      static agxData::ScalarAttributeT< agx::Physics::Geometry::ShapePtr >* shape2Attribute;
      static agxData::ScalarAttributeT< agx::Real >* areaAttribute;
      static agxData::ScalarAttributeT< agx::Real >* elasticRestLengthShape1Attribute;
      static agxData::ScalarAttributeT< agx::Real >* elasticRestLengthShape2Attribute;

    protected:
      virtual ~ContactPointModel();
      virtual agxData::EntityData* createData(agxData::EntityStorage* storage) override;
      virtual void configure(agx::TiXmlElement* eEntity) override;
      virtual void initAttributeAccessors() override;
      virtual void construct(agxData::EntityPtr instance) override;
      void construct(agx::Physics::ContactPointPtr contactPoint);
    };


    DOXYGEN_START_INTERNAL_BLOCK()
    #ifndef AGX_PHYSICS_CONTACTPOINT_DATA_SET_OVERRIDE
    #define AGX_PHYSICS_CONTACTPOINT_DATA_SET
    class AGXPHYSICS_EXPORT ContactPointData : public agxData::EntityData
    {
    public:
      ContactPointInstance operator[] (size_t index);

    public:
      agxData::Array< ContactPointPtr >& instance;
      agxData::Array< agx::Vec3 > point;
      agxData::Array< agx::Vec3f > normal;
      agxData::Array< agx::Vec3f > tangentU;
      agxData::Array< agx::Vec3f > tangentV;
      agxData::Array< agx::Real > depth;
      agxData::Array< agx::UInt8 > state;
      agxData::Array< agx::Vec3f > velocity;
      agxData::Array< agx::Vec3 > localForce;
      agxData::Array< agx::Vec4i8 > indexSet;
      agxData::Array< agx::Real > maxNormalForce;
      agxData::Array< agx::Bool > enabled;
      agxData::Array< agx::UInt32 > faceIndex1;
      agxData::Array< agx::UInt32 > faceIndex2;
      agxData::Array< agx::UInt8 > faceFeature1;
      agxData::Array< agx::UInt8 > faceFeature2;
      agxData::Array< agx::Physics::Geometry::ShapePtr > shape1;
      agxData::Array< agx::Physics::Geometry::ShapePtr > shape2;
      agxData::Array< agx::Real > area;
      agxData::Array< agx::Real > elasticRestLengthShape1;
      agxData::Array< agx::Real > elasticRestLengthShape2;

    public:
      typedef agx::Vec3 pointType;
      typedef agx::Vec3f normalType;
      typedef agx::Vec3f tangentUType;
      typedef agx::Vec3f tangentVType;
      typedef agx::Real depthType;
      typedef agx::UInt8 stateType;
      typedef agx::Vec3f velocityType;
      typedef agx::Vec3 localForceType;
      typedef agx::Vec4i8 indexSetType;
      typedef agx::Real maxNormalForceType;
      typedef agx::Bool enabledType;
      typedef agx::UInt32 faceIndex1Type;
      typedef agx::UInt32 faceIndex2Type;
      typedef agx::UInt8 faceFeature1Type;
      typedef agx::UInt8 faceFeature2Type;
      typedef agx::Physics::Geometry::ShapePtr shape1Type;
      typedef agx::Physics::Geometry::ShapePtr shape2Type;
      typedef agx::Real areaType;
      typedef agx::Real elasticRestLengthShape1Type;
      typedef agx::Real elasticRestLengthShape2Type;

    public:
      ContactPointData(agxData::EntityStorage* storage);
      ContactPointData();

    protected:
      virtual ~ContactPointData() {}
      virtual void setNumElements(agx::Index numElements) override;

    private:
      ContactPointData& operator= (const ContactPointData&) { return *this; }

    };
    #endif
    DOXYGEN_END_INTERNAL_BLOCK()


    DOXYGEN_START_INTERNAL_BLOCK()
    class AGXPHYSICS_EXPORT ContactPointSemantics : protected agxData::EntityPtr
    {
    public:

      // Automatic getters
      agx::Vec3 const& getPoint() const;
      agx::Vec3f const& getNormal() const;
      agx::Vec3f const& getTangentU() const;
      agx::Vec3f const& getTangentV() const;
      agx::Real const& getDepth() const;
      agx::UInt8 const& getState() const;
      agx::Vec3f const& getVelocity() const;
      agx::Vec3 const& getLocalForce() const;
      agx::Vec4i8 const& getIndexSet() const;
      agx::Real const& getMaxNormalForce() const;
      agx::Bool const& getEnabled() const;
      agx::UInt32 const& getFaceIndex1() const;
      agx::UInt32 const& getFaceIndex2() const;
      agx::UInt8 const& getFaceFeature1() const;
      agx::UInt8 const& getFaceFeature2() const;
      agx::Physics::Geometry::ShapePtr const& getShape1() const;
      agx::Physics::Geometry::ShapePtr const& getShape2() const;
      agx::Real const& getArea() const;
      agx::Real const& getElasticRestLengthShape1() const;
      agx::Real const& getElasticRestLengthShape2() const;

      // Semantics defined by explicit kernels

      // Automatic setters
      void setPoint(agx::Vec3 const& value);
      void setNormal(agx::Vec3f const& value);
      void setTangentU(agx::Vec3f const& value);
      void setTangentV(agx::Vec3f const& value);
      void setDepth(agx::Real const& value);
      void setState(agx::UInt8 const& value);
      void setVelocity(agx::Vec3f const& value);
      void setLocalForce(agx::Vec3 const& value);
      void setIndexSet(agx::Vec4i8 const& value);
      void setMaxNormalForce(agx::Real const& value);
      void setEnabled(agx::Bool const& value);
      void setFaceIndex1(agx::UInt32 const& value);
      void setFaceIndex2(agx::UInt32 const& value);
      void setFaceFeature1(agx::UInt8 const& value);
      void setFaceFeature2(agx::UInt8 const& value);
      void setShape1(agx::Physics::Geometry::ShapePtr const& value);
      void setShape2(agx::Physics::Geometry::ShapePtr const& value);
      void setArea(agx::Real const& value);
      void setElasticRestLengthShape1(agx::Real const& value);
      void setElasticRestLengthShape2(agx::Real const& value);


    protected:
      friend class ContactPointPtr;
      friend class ContactPointInstance;
      ContactPointSemantics();
    };
    DOXYGEN_END_INTERNAL_BLOCK()


    /**
    Pointer to a entity instance of type Physics.ContactPoint
    */
    class CALLABLE ContactPointPtr : public agxData::EntityPtr
    {
    public:
      typedef ContactPointModel ModelType;
      typedef ContactPointData DataType;
      typedef ContactPointInstance InstanceType;

    public:
      AGXPHYSICS_EXPORT ContactPointPtr();
      AGXPHYSICS_EXPORT ContactPointPtr(agxData::EntityStorage* storage, agx::Index id);
      AGXPHYSICS_EXPORT ContactPointPtr(const agxData::EntityPtr& ptr);
      AGXPHYSICS_EXPORT ContactPointPtr(const agxData::EntityInstance& instance);
      AGXPHYSICS_EXPORT ContactPointPtr& operator= (const agxData::EntityPtr& ptr);
      AGXPHYSICS_EXPORT ContactPointPtr& operator= (const agxData::EntityInstance& instance);
      AGXPHYSICS_EXPORT ContactPointInstance instance();
      AGXPHYSICS_EXPORT const ContactPointInstance instance() const;

      AGXPHYSICS_EXPORT ContactPointSemantics* operator->();
      AGXPHYSICS_EXPORT const ContactPointSemantics* operator->() const;

      ContactPointData* getData();
      const ContactPointData* getData() const;


      /// \return reference to the point attribute
      AGXPHYSICS_EXPORT agx::Vec3& point();
      /// \return const reference to the point attribute
      AGXPHYSICS_EXPORT agx::Vec3 const& point() const;

      /// \return reference to the normal attribute
      AGXPHYSICS_EXPORT agx::Vec3f& normal();
      /// \return const reference to the normal attribute
      AGXPHYSICS_EXPORT agx::Vec3f const& normal() const;

      /// \return reference to the tangentU attribute
      AGXPHYSICS_EXPORT agx::Vec3f& tangentU();
      /// \return const reference to the tangentU attribute
      AGXPHYSICS_EXPORT agx::Vec3f const& tangentU() const;

      /// \return reference to the tangentV attribute
      AGXPHYSICS_EXPORT agx::Vec3f& tangentV();
      /// \return const reference to the tangentV attribute
      AGXPHYSICS_EXPORT agx::Vec3f const& tangentV() const;

      /// \return reference to the depth attribute
      AGXPHYSICS_EXPORT agx::Real& depth();
      /// \return const reference to the depth attribute
      AGXPHYSICS_EXPORT agx::Real const& depth() const;

      /// Contact point state, such as impacting, resting etc.
      AGXPHYSICS_EXPORT agx::UInt8& state();
      /// Contact point state, such as impacting, resting etc.
      AGXPHYSICS_EXPORT agx::UInt8 const& state() const;

      /// \return reference to the velocity attribute
      AGXPHYSICS_EXPORT agx::Vec3f& velocity();
      /// \return const reference to the velocity attribute
      AGXPHYSICS_EXPORT agx::Vec3f const& velocity() const;

      /// \return reference to the localForce attribute
      AGXPHYSICS_EXPORT agx::Vec3& localForce();
      /// \return const reference to the localForce attribute
      AGXPHYSICS_EXPORT agx::Vec3 const& localForce() const;

      /// \return reference to the indexSet attribute
      AGXPHYSICS_EXPORT agx::Vec4i8& indexSet();
      /// \return const reference to the indexSet attribute
      AGXPHYSICS_EXPORT agx::Vec4i8 const& indexSet() const;

      /// \return reference to the maxNormalForce attribute
      AGXPHYSICS_EXPORT agx::Real& maxNormalForce();
      /// \return const reference to the maxNormalForce attribute
      AGXPHYSICS_EXPORT agx::Real const& maxNormalForce() const;

      /// \return reference to the enabled attribute
      AGXPHYSICS_EXPORT agx::Bool& enabled();
      /// \return const reference to the enabled attribute
      AGXPHYSICS_EXPORT agx::Bool const& enabled() const;

      /// The face index of shape 1 in contact (e.g. index of triangle for trimesh).
      AGXPHYSICS_EXPORT agx::UInt32& faceIndex1();
      /// The face index of shape 1 in contact (e.g. index of triangle for trimesh).
      AGXPHYSICS_EXPORT agx::UInt32 const& faceIndex1() const;

      /// The face index of shape 2 in contact (e.g. index of triangle for trimesh).
      AGXPHYSICS_EXPORT agx::UInt32& faceIndex2();
      /// The face index of shape 2 in contact (e.g. index of triangle for trimesh).
      AGXPHYSICS_EXPORT agx::UInt32 const& faceIndex2() const;

      /// The face feature of shape 1 in contact (e.g. voronoi region on triangle for trimesh).
      AGXPHYSICS_EXPORT agx::UInt8& faceFeature1();
      /// The face feature of shape 1 in contact (e.g. voronoi region on triangle for trimesh).
      AGXPHYSICS_EXPORT agx::UInt8 const& faceFeature1() const;

      /// The face feature of shape 2 in contact (e.g. voronoi region on triangle for trimesh).
      AGXPHYSICS_EXPORT agx::UInt8& faceFeature2();
      /// The face feature of shape 2 in contact (e.g. voronoi region on triangle for trimesh).
      AGXPHYSICS_EXPORT agx::UInt8 const& faceFeature2() const;

      /// Shape number 1 in contact
      AGXPHYSICS_EXPORT agx::Physics::Geometry::ShapePtr& shape1();
      /// Shape number 1 in contact
      AGXPHYSICS_EXPORT agx::Physics::Geometry::ShapePtr const& shape1() const;

      /// Shape number 2 in contact
      AGXPHYSICS_EXPORT agx::Physics::Geometry::ShapePtr& shape2();
      /// Shape number 2 in contact
      AGXPHYSICS_EXPORT agx::Physics::Geometry::ShapePtr const& shape2() const;

      /// \return reference to the area attribute
      AGXPHYSICS_EXPORT agx::Real& area();
      /// \return const reference to the area attribute
      AGXPHYSICS_EXPORT agx::Real const& area() const;

      /// \return reference to the elasticRestLengthShape1 attribute
      AGXPHYSICS_EXPORT agx::Real& elasticRestLengthShape1();
      /// \return const reference to the elasticRestLengthShape1 attribute
      AGXPHYSICS_EXPORT agx::Real const& elasticRestLengthShape1() const;

      /// \return reference to the elasticRestLengthShape2 attribute
      AGXPHYSICS_EXPORT agx::Real& elasticRestLengthShape2();
      /// \return const reference to the elasticRestLengthShape2 attribute
      AGXPHYSICS_EXPORT agx::Real const& elasticRestLengthShape2() const;

      DOXYGEN_START_INTERNAL_BLOCK()
      // Only used by tolua!
      #ifdef AGX_GENERATED_TOLUA_WRAPPER
      inline agx::Vec3 const& getPoint() const { return (*this)->getPoint(); }
      inline agx::Vec3f const& getNormal() const { return (*this)->getNormal(); }
      inline agx::Vec3f const& getTangentU() const { return (*this)->getTangentU(); }
      inline agx::Vec3f const& getTangentV() const { return (*this)->getTangentV(); }
      inline agx::Real const& getDepth() const { return (*this)->getDepth(); }
      inline agx::UInt8 const& getState() const { return (*this)->getState(); }
      inline agx::Vec3f const& getVelocity() const { return (*this)->getVelocity(); }
      inline agx::Vec3 const& getLocalForce() const { return (*this)->getLocalForce(); }
      inline agx::Vec4i8 const& getIndexSet() const { return (*this)->getIndexSet(); }
      inline agx::Real const& getMaxNormalForce() const { return (*this)->getMaxNormalForce(); }
      inline agx::Bool const& getEnabled() const { return (*this)->getEnabled(); }
      inline agx::UInt32 const& getFaceIndex1() const { return (*this)->getFaceIndex1(); }
      inline agx::UInt32 const& getFaceIndex2() const { return (*this)->getFaceIndex2(); }
      inline agx::UInt8 const& getFaceFeature1() const { return (*this)->getFaceFeature1(); }
      inline agx::UInt8 const& getFaceFeature2() const { return (*this)->getFaceFeature2(); }
      inline agx::Physics::Geometry::ShapePtr const& getShape1() const { return (*this)->getShape1(); }
      inline agx::Physics::Geometry::ShapePtr const& getShape2() const { return (*this)->getShape2(); }
      inline agx::Real const& getArea() const { return (*this)->getArea(); }
      inline agx::Real const& getElasticRestLengthShape1() const { return (*this)->getElasticRestLengthShape1(); }
      inline agx::Real const& getElasticRestLengthShape2() const { return (*this)->getElasticRestLengthShape2(); }


      inline void setPoint(agx::Vec3 const& value) { (*this)->setPoint(value); }
      inline void setNormal(agx::Vec3f const& value) { (*this)->setNormal(value); }
      inline void setTangentU(agx::Vec3f const& value) { (*this)->setTangentU(value); }
      inline void setTangentV(agx::Vec3f const& value) { (*this)->setTangentV(value); }
      inline void setDepth(agx::Real const& value) { (*this)->setDepth(value); }
      inline void setState(agx::UInt8 const& value) { (*this)->setState(value); }
      inline void setVelocity(agx::Vec3f const& value) { (*this)->setVelocity(value); }
      inline void setLocalForce(agx::Vec3 const& value) { (*this)->setLocalForce(value); }
      inline void setIndexSet(agx::Vec4i8 const& value) { (*this)->setIndexSet(value); }
      inline void setMaxNormalForce(agx::Real const& value) { (*this)->setMaxNormalForce(value); }
      inline void setEnabled(agx::Bool const& value) { (*this)->setEnabled(value); }
      inline void setFaceIndex1(agx::UInt32 const& value) { (*this)->setFaceIndex1(value); }
      inline void setFaceIndex2(agx::UInt32 const& value) { (*this)->setFaceIndex2(value); }
      inline void setFaceFeature1(agx::UInt8 const& value) { (*this)->setFaceFeature1(value); }
      inline void setFaceFeature2(agx::UInt8 const& value) { (*this)->setFaceFeature2(value); }
      inline void setShape1(agx::Physics::Geometry::ShapePtr const& value) { (*this)->setShape1(value); }
      inline void setShape2(agx::Physics::Geometry::ShapePtr const& value) { (*this)->setShape2(value); }
      inline void setArea(agx::Real const& value) { (*this)->setArea(value); }
      inline void setElasticRestLengthShape1(agx::Real const& value) { (*this)->setElasticRestLengthShape1(value); }
      inline void setElasticRestLengthShape2(agx::Real const& value) { (*this)->setElasticRestLengthShape2(value); }
      #endif
      DOXYGEN_END_INTERNAL_BLOCK()

    };


    DOXYGEN_START_INTERNAL_BLOCK()
    class AGXPHYSICS_EXPORT ContactPointInstance : public agxData::EntityInstance
    {
    public:
      ContactPointInstance();
      ContactPointInstance(ContactPointData* data, agx::Index index);
      ContactPointInstance(agxData::EntityStorage *storage, agx::Index index);
      ContactPointInstance(const agxData::EntityInstance& other);
      ContactPointInstance(const agxData::EntityPtr& ptr);

      ContactPointData* getData();
      const ContactPointData* getData() const;

    public:
      /// \return reference to the point attribute
      agx::Vec3& point();
      /// \return const reference to the point attribute
      agx::Vec3 const& point() const;

      /// \return reference to the normal attribute
      agx::Vec3f& normal();
      /// \return const reference to the normal attribute
      agx::Vec3f const& normal() const;

      /// \return reference to the tangentU attribute
      agx::Vec3f& tangentU();
      /// \return const reference to the tangentU attribute
      agx::Vec3f const& tangentU() const;

      /// \return reference to the tangentV attribute
      agx::Vec3f& tangentV();
      /// \return const reference to the tangentV attribute
      agx::Vec3f const& tangentV() const;

      /// \return reference to the depth attribute
      agx::Real& depth();
      /// \return const reference to the depth attribute
      agx::Real const& depth() const;

      /// Contact point state, such as impacting, resting etc.
      agx::UInt8& state();
      /// Contact point state, such as impacting, resting etc.
      agx::UInt8 const& state() const;

      /// \return reference to the velocity attribute
      agx::Vec3f& velocity();
      /// \return const reference to the velocity attribute
      agx::Vec3f const& velocity() const;

      /// \return reference to the localForce attribute
      agx::Vec3& localForce();
      /// \return const reference to the localForce attribute
      agx::Vec3 const& localForce() const;

      /// \return reference to the indexSet attribute
      agx::Vec4i8& indexSet();
      /// \return const reference to the indexSet attribute
      agx::Vec4i8 const& indexSet() const;

      /// \return reference to the maxNormalForce attribute
      agx::Real& maxNormalForce();
      /// \return const reference to the maxNormalForce attribute
      agx::Real const& maxNormalForce() const;

      /// \return reference to the enabled attribute
      agx::Bool& enabled();
      /// \return const reference to the enabled attribute
      agx::Bool const& enabled() const;

      /// The face index of shape 1 in contact (e.g. index of triangle for trimesh).
      agx::UInt32& faceIndex1();
      /// The face index of shape 1 in contact (e.g. index of triangle for trimesh).
      agx::UInt32 const& faceIndex1() const;

      /// The face index of shape 2 in contact (e.g. index of triangle for trimesh).
      agx::UInt32& faceIndex2();
      /// The face index of shape 2 in contact (e.g. index of triangle for trimesh).
      agx::UInt32 const& faceIndex2() const;

      /// The face feature of shape 1 in contact (e.g. voronoi region on triangle for trimesh).
      agx::UInt8& faceFeature1();
      /// The face feature of shape 1 in contact (e.g. voronoi region on triangle for trimesh).
      agx::UInt8 const& faceFeature1() const;

      /// The face feature of shape 2 in contact (e.g. voronoi region on triangle for trimesh).
      agx::UInt8& faceFeature2();
      /// The face feature of shape 2 in contact (e.g. voronoi region on triangle for trimesh).
      agx::UInt8 const& faceFeature2() const;

      /// Shape number 1 in contact
      agx::Physics::Geometry::ShapePtr& shape1();
      /// Shape number 1 in contact
      agx::Physics::Geometry::ShapePtr const& shape1() const;

      /// Shape number 2 in contact
      agx::Physics::Geometry::ShapePtr& shape2();
      /// Shape number 2 in contact
      agx::Physics::Geometry::ShapePtr const& shape2() const;

      /// \return reference to the area attribute
      agx::Real& area();
      /// \return const reference to the area attribute
      agx::Real const& area() const;

      /// \return reference to the elasticRestLengthShape1 attribute
      agx::Real& elasticRestLengthShape1();
      /// \return const reference to the elasticRestLengthShape1 attribute
      agx::Real const& elasticRestLengthShape1() const;

      /// \return reference to the elasticRestLengthShape2 attribute
      agx::Real& elasticRestLengthShape2();
      /// \return const reference to the elasticRestLengthShape2 attribute
      agx::Real const& elasticRestLengthShape2() const;

    };
    DOXYGEN_END_INTERNAL_BLOCK()



    typedef agx::VectorPOD<ContactPointPtr> ContactPointPtrVector;
    typedef agxData::Array<ContactPointPtr> ContactPointPtrArray;



    DOXYGEN_START_INTERNAL_BLOCK()
    /* Implementation */
    //-----------------------------------------------------------------------------------------------------
    //-----------------------------------------------------------------------------------------------------
    inline ContactPointInstance agx::Physics::ContactPointData::operator[] (size_t index) { return ContactPointInstance(this, (agx::Index)index); }
    //-----------------------------------------------------------------------------------------------------
    AGX_FORCE_INLINE ContactPointPtr::ContactPointPtr() {}
    AGX_FORCE_INLINE ContactPointPtr::ContactPointPtr(agxData::EntityStorage* storage, agx::Index id) : agxData::EntityPtr(storage, id) {}
    AGX_FORCE_INLINE ContactPointPtr::ContactPointPtr(const agxData::EntityPtr& ptr) : agxData::EntityPtr(ptr)
    {
      agxAssertN(!ptr || ptr.isInstanceOf(ContactPointModel::instance()),
        "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
        EntityPtr::getModel()->fullPath().c_str(), ContactPointModel::instance()->fullPath().c_str());
    }

    AGX_FORCE_INLINE ContactPointPtr::ContactPointPtr(const agxData::EntityInstance& instance) : agxData::EntityPtr(instance)
    {
      agxAssertN(!instance || instance.isInstanceOf(ContactPointModel::instance()),
        "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
        EntityPtr::getModel()->fullPath().c_str(), ContactPointModel::instance()->fullPath().c_str());
    }

    AGX_FORCE_INLINE ContactPointPtr& ContactPointPtr::operator= (const agxData::EntityPtr& ptr)
    {
      agxData::EntityPtr::operator= (ptr);
      agxAssertN(!ptr || ptr.isInstanceOf(ContactPointModel::instance()),
        "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
        EntityPtr::getModel()->fullPath().c_str(), ContactPointModel::instance()->fullPath().c_str());
      return *this;
    }

    AGX_FORCE_INLINE ContactPointPtr& ContactPointPtr::operator= (const agxData::EntityInstance& instance)
    {
      agxData::EntityPtr::operator= (instance);
      agxAssertN(!instance || instance.isInstanceOf(ContactPointModel::instance()),
        "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
        EntityPtr::getModel()->fullPath().c_str(), ContactPointModel::instance()->fullPath().c_str());
      return *this;
    }

    inline ContactPointInstance ContactPointPtr::instance() { return agxData::EntityPtr::instance(); }
    inline const ContactPointInstance ContactPointPtr::instance() const { return agxData::EntityPtr::instance(); }
    AGX_FORCE_INLINE ContactPointSemantics* ContactPointPtr::operator->() { return (ContactPointSemantics* )this; }
    AGX_FORCE_INLINE const ContactPointSemantics* ContactPointPtr::operator->() const { return (const ContactPointSemantics* )this; }
    AGX_FORCE_INLINE ContactPointData* ContactPointPtr::getData() { return static_cast<ContactPointData* >(agxData::EntityPtr::getData()); }
    AGX_FORCE_INLINE const ContactPointData* ContactPointPtr::getData() const { return static_cast<const ContactPointData* >(agxData::EntityPtr::getData()); }

    AGX_FORCE_INLINE agx::Vec3& ContactPointPtr::point() { verifyIndex(); return getData()->point[calculateIndex()]; }
    AGX_FORCE_INLINE agx::Vec3 const& ContactPointPtr::point() const { verifyIndex(); return getData()->point[calculateIndex()]; }

    AGX_FORCE_INLINE agx::Vec3f& ContactPointPtr::normal() { verifyIndex(); return getData()->normal[calculateIndex()]; }
    AGX_FORCE_INLINE agx::Vec3f const& ContactPointPtr::normal() const { verifyIndex(); return getData()->normal[calculateIndex()]; }

    AGX_FORCE_INLINE agx::Vec3f& ContactPointPtr::tangentU() { verifyIndex(); return getData()->tangentU[calculateIndex()]; }
    AGX_FORCE_INLINE agx::Vec3f const& ContactPointPtr::tangentU() const { verifyIndex(); return getData()->tangentU[calculateIndex()]; }

    AGX_FORCE_INLINE agx::Vec3f& ContactPointPtr::tangentV() { verifyIndex(); return getData()->tangentV[calculateIndex()]; }
    AGX_FORCE_INLINE agx::Vec3f const& ContactPointPtr::tangentV() const { verifyIndex(); return getData()->tangentV[calculateIndex()]; }

    AGX_FORCE_INLINE agx::Real& ContactPointPtr::depth() { verifyIndex(); return getData()->depth[calculateIndex()]; }
    AGX_FORCE_INLINE agx::Real const& ContactPointPtr::depth() const { verifyIndex(); return getData()->depth[calculateIndex()]; }

    AGX_FORCE_INLINE agx::UInt8& ContactPointPtr::state() { verifyIndex(); return getData()->state[calculateIndex()]; }
    AGX_FORCE_INLINE agx::UInt8 const& ContactPointPtr::state() const { verifyIndex(); return getData()->state[calculateIndex()]; }

    AGX_FORCE_INLINE agx::Vec3f& ContactPointPtr::velocity() { verifyIndex(); return getData()->velocity[calculateIndex()]; }
    AGX_FORCE_INLINE agx::Vec3f const& ContactPointPtr::velocity() const { verifyIndex(); return getData()->velocity[calculateIndex()]; }

    AGX_FORCE_INLINE agx::Vec3& ContactPointPtr::localForce() { verifyIndex(); return getData()->localForce[calculateIndex()]; }
    AGX_FORCE_INLINE agx::Vec3 const& ContactPointPtr::localForce() const { verifyIndex(); return getData()->localForce[calculateIndex()]; }

    AGX_FORCE_INLINE agx::Vec4i8& ContactPointPtr::indexSet() { verifyIndex(); return getData()->indexSet[calculateIndex()]; }
    AGX_FORCE_INLINE agx::Vec4i8 const& ContactPointPtr::indexSet() const { verifyIndex(); return getData()->indexSet[calculateIndex()]; }

    AGX_FORCE_INLINE agx::Real& ContactPointPtr::maxNormalForce() { verifyIndex(); return getData()->maxNormalForce[calculateIndex()]; }
    AGX_FORCE_INLINE agx::Real const& ContactPointPtr::maxNormalForce() const { verifyIndex(); return getData()->maxNormalForce[calculateIndex()]; }

    AGX_FORCE_INLINE agx::Bool& ContactPointPtr::enabled() { verifyIndex(); return getData()->enabled[calculateIndex()]; }
    AGX_FORCE_INLINE agx::Bool const& ContactPointPtr::enabled() const { verifyIndex(); return getData()->enabled[calculateIndex()]; }

    AGX_FORCE_INLINE agx::UInt32& ContactPointPtr::faceIndex1() { verifyIndex(); return getData()->faceIndex1[calculateIndex()]; }
    AGX_FORCE_INLINE agx::UInt32 const& ContactPointPtr::faceIndex1() const { verifyIndex(); return getData()->faceIndex1[calculateIndex()]; }

    AGX_FORCE_INLINE agx::UInt32& ContactPointPtr::faceIndex2() { verifyIndex(); return getData()->faceIndex2[calculateIndex()]; }
    AGX_FORCE_INLINE agx::UInt32 const& ContactPointPtr::faceIndex2() const { verifyIndex(); return getData()->faceIndex2[calculateIndex()]; }

    AGX_FORCE_INLINE agx::UInt8& ContactPointPtr::faceFeature1() { verifyIndex(); return getData()->faceFeature1[calculateIndex()]; }
    AGX_FORCE_INLINE agx::UInt8 const& ContactPointPtr::faceFeature1() const { verifyIndex(); return getData()->faceFeature1[calculateIndex()]; }

    AGX_FORCE_INLINE agx::UInt8& ContactPointPtr::faceFeature2() { verifyIndex(); return getData()->faceFeature2[calculateIndex()]; }
    AGX_FORCE_INLINE agx::UInt8 const& ContactPointPtr::faceFeature2() const { verifyIndex(); return getData()->faceFeature2[calculateIndex()]; }

    AGX_FORCE_INLINE agx::Physics::Geometry::ShapePtr& ContactPointPtr::shape1() { verifyIndex(); return getData()->shape1[calculateIndex()]; }
    AGX_FORCE_INLINE agx::Physics::Geometry::ShapePtr const& ContactPointPtr::shape1() const { verifyIndex(); return getData()->shape1[calculateIndex()]; }

    AGX_FORCE_INLINE agx::Physics::Geometry::ShapePtr& ContactPointPtr::shape2() { verifyIndex(); return getData()->shape2[calculateIndex()]; }
    AGX_FORCE_INLINE agx::Physics::Geometry::ShapePtr const& ContactPointPtr::shape2() const { verifyIndex(); return getData()->shape2[calculateIndex()]; }

    AGX_FORCE_INLINE agx::Real& ContactPointPtr::area() { verifyIndex(); return getData()->area[calculateIndex()]; }
    AGX_FORCE_INLINE agx::Real const& ContactPointPtr::area() const { verifyIndex(); return getData()->area[calculateIndex()]; }

    AGX_FORCE_INLINE agx::Real& ContactPointPtr::elasticRestLengthShape1() { verifyIndex(); return getData()->elasticRestLengthShape1[calculateIndex()]; }
    AGX_FORCE_INLINE agx::Real const& ContactPointPtr::elasticRestLengthShape1() const { verifyIndex(); return getData()->elasticRestLengthShape1[calculateIndex()]; }

    AGX_FORCE_INLINE agx::Real& ContactPointPtr::elasticRestLengthShape2() { verifyIndex(); return getData()->elasticRestLengthShape2[calculateIndex()]; }
    AGX_FORCE_INLINE agx::Real const& ContactPointPtr::elasticRestLengthShape2() const { verifyIndex(); return getData()->elasticRestLengthShape2[calculateIndex()]; }

    //-----------------------------------------------------------------------------------------------------
    AGX_FORCE_INLINE ContactPointInstance::ContactPointInstance() {}
    AGX_FORCE_INLINE ContactPointInstance::ContactPointInstance(ContactPointData* data, agx::Index index) : agxData::EntityInstance(data, index) {}
    AGX_FORCE_INLINE ContactPointInstance::ContactPointInstance(agxData::EntityStorage* storage, agx::Index index) : agxData::EntityInstance(storage, index) {}
    AGX_FORCE_INLINE ContactPointInstance::ContactPointInstance(const agxData::EntityInstance& other) : agxData::EntityInstance(other)
    {
      agxAssertN(!other || other.isInstanceOf(ContactPointModel::instance()),
        "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
        EntityInstance::getModel()->fullPath().c_str(), ContactPointModel::instance()->fullPath().c_str());
    }

    AGX_FORCE_INLINE ContactPointInstance::ContactPointInstance(const agxData::EntityPtr& ptr) : agxData::EntityInstance(ptr)
    {
      agxAssertN(!ptr || ptr.isInstanceOf(ContactPointModel::instance()),
        "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
        EntityInstance::getModel()->fullPath().c_str(), ContactPointModel::instance()->fullPath().c_str());
    }


    AGX_FORCE_INLINE ContactPointData* ContactPointInstance::getData() { return static_cast<ContactPointData* >(agxData::EntityInstance::getData()); }
    AGX_FORCE_INLINE const ContactPointData* ContactPointInstance::getData() const { return static_cast<const ContactPointData* >(agxData::EntityInstance::getData()); }

    AGX_FORCE_INLINE agx::Vec3& ContactPointInstance::point() { verifyIndex(); return getData()->point[getIndex()]; }
    AGX_FORCE_INLINE agx::Vec3 const& ContactPointInstance::point() const { verifyIndex(); return getData()->point[getIndex()]; }

    AGX_FORCE_INLINE agx::Vec3f& ContactPointInstance::normal() { verifyIndex(); return getData()->normal[getIndex()]; }
    AGX_FORCE_INLINE agx::Vec3f const& ContactPointInstance::normal() const { verifyIndex(); return getData()->normal[getIndex()]; }

    AGX_FORCE_INLINE agx::Vec3f& ContactPointInstance::tangentU() { verifyIndex(); return getData()->tangentU[getIndex()]; }
    AGX_FORCE_INLINE agx::Vec3f const& ContactPointInstance::tangentU() const { verifyIndex(); return getData()->tangentU[getIndex()]; }

    AGX_FORCE_INLINE agx::Vec3f& ContactPointInstance::tangentV() { verifyIndex(); return getData()->tangentV[getIndex()]; }
    AGX_FORCE_INLINE agx::Vec3f const& ContactPointInstance::tangentV() const { verifyIndex(); return getData()->tangentV[getIndex()]; }

    AGX_FORCE_INLINE agx::Real& ContactPointInstance::depth() { verifyIndex(); return getData()->depth[getIndex()]; }
    AGX_FORCE_INLINE agx::Real const& ContactPointInstance::depth() const { verifyIndex(); return getData()->depth[getIndex()]; }

    AGX_FORCE_INLINE agx::UInt8& ContactPointInstance::state() { verifyIndex(); return getData()->state[getIndex()]; }
    AGX_FORCE_INLINE agx::UInt8 const& ContactPointInstance::state() const { verifyIndex(); return getData()->state[getIndex()]; }

    AGX_FORCE_INLINE agx::Vec3f& ContactPointInstance::velocity() { verifyIndex(); return getData()->velocity[getIndex()]; }
    AGX_FORCE_INLINE agx::Vec3f const& ContactPointInstance::velocity() const { verifyIndex(); return getData()->velocity[getIndex()]; }

    AGX_FORCE_INLINE agx::Vec3& ContactPointInstance::localForce() { verifyIndex(); return getData()->localForce[getIndex()]; }
    AGX_FORCE_INLINE agx::Vec3 const& ContactPointInstance::localForce() const { verifyIndex(); return getData()->localForce[getIndex()]; }

    AGX_FORCE_INLINE agx::Vec4i8& ContactPointInstance::indexSet() { verifyIndex(); return getData()->indexSet[getIndex()]; }
    AGX_FORCE_INLINE agx::Vec4i8 const& ContactPointInstance::indexSet() const { verifyIndex(); return getData()->indexSet[getIndex()]; }

    AGX_FORCE_INLINE agx::Real& ContactPointInstance::maxNormalForce() { verifyIndex(); return getData()->maxNormalForce[getIndex()]; }
    AGX_FORCE_INLINE agx::Real const& ContactPointInstance::maxNormalForce() const { verifyIndex(); return getData()->maxNormalForce[getIndex()]; }

    AGX_FORCE_INLINE agx::Bool& ContactPointInstance::enabled() { verifyIndex(); return getData()->enabled[getIndex()]; }
    AGX_FORCE_INLINE agx::Bool const& ContactPointInstance::enabled() const { verifyIndex(); return getData()->enabled[getIndex()]; }

    AGX_FORCE_INLINE agx::UInt32& ContactPointInstance::faceIndex1() { verifyIndex(); return getData()->faceIndex1[getIndex()]; }
    AGX_FORCE_INLINE agx::UInt32 const& ContactPointInstance::faceIndex1() const { verifyIndex(); return getData()->faceIndex1[getIndex()]; }

    AGX_FORCE_INLINE agx::UInt32& ContactPointInstance::faceIndex2() { verifyIndex(); return getData()->faceIndex2[getIndex()]; }
    AGX_FORCE_INLINE agx::UInt32 const& ContactPointInstance::faceIndex2() const { verifyIndex(); return getData()->faceIndex2[getIndex()]; }

    AGX_FORCE_INLINE agx::UInt8& ContactPointInstance::faceFeature1() { verifyIndex(); return getData()->faceFeature1[getIndex()]; }
    AGX_FORCE_INLINE agx::UInt8 const& ContactPointInstance::faceFeature1() const { verifyIndex(); return getData()->faceFeature1[getIndex()]; }

    AGX_FORCE_INLINE agx::UInt8& ContactPointInstance::faceFeature2() { verifyIndex(); return getData()->faceFeature2[getIndex()]; }
    AGX_FORCE_INLINE agx::UInt8 const& ContactPointInstance::faceFeature2() const { verifyIndex(); return getData()->faceFeature2[getIndex()]; }

    AGX_FORCE_INLINE agx::Physics::Geometry::ShapePtr& ContactPointInstance::shape1() { verifyIndex(); return getData()->shape1[getIndex()]; }
    AGX_FORCE_INLINE agx::Physics::Geometry::ShapePtr const& ContactPointInstance::shape1() const { verifyIndex(); return getData()->shape1[getIndex()]; }

    AGX_FORCE_INLINE agx::Physics::Geometry::ShapePtr& ContactPointInstance::shape2() { verifyIndex(); return getData()->shape2[getIndex()]; }
    AGX_FORCE_INLINE agx::Physics::Geometry::ShapePtr const& ContactPointInstance::shape2() const { verifyIndex(); return getData()->shape2[getIndex()]; }

    AGX_FORCE_INLINE agx::Real& ContactPointInstance::area() { verifyIndex(); return getData()->area[getIndex()]; }
    AGX_FORCE_INLINE agx::Real const& ContactPointInstance::area() const { verifyIndex(); return getData()->area[getIndex()]; }

    AGX_FORCE_INLINE agx::Real& ContactPointInstance::elasticRestLengthShape1() { verifyIndex(); return getData()->elasticRestLengthShape1[getIndex()]; }
    AGX_FORCE_INLINE agx::Real const& ContactPointInstance::elasticRestLengthShape1() const { verifyIndex(); return getData()->elasticRestLengthShape1[getIndex()]; }

    AGX_FORCE_INLINE agx::Real& ContactPointInstance::elasticRestLengthShape2() { verifyIndex(); return getData()->elasticRestLengthShape2[getIndex()]; }
    AGX_FORCE_INLINE agx::Real const& ContactPointInstance::elasticRestLengthShape2() const { verifyIndex(); return getData()->elasticRestLengthShape2[getIndex()]; }

    //-----------------------------------------------------------------------------------------------------
    AGX_FORCE_INLINE ContactPointSemantics::ContactPointSemantics() {}
    //-----------------------------------------------------------------------------------------------------
    DOXYGEN_END_INTERNAL_BLOCK()
  }
}

AGX_TYPE_BINDING(agx::Physics::ContactPointPtr, "Physics.ContactPointPtr")
AGX_TYPE_BINDING(agx::Physics::ContactPointInstance, "Physics.ContactPointInstance")

#ifdef _MSC_VER
# pragma warning(pop)
#endif

#undef AGX_ENTITY_WRAPPER
#undef AGX_ENTITY_NAMESPACE
#endif

