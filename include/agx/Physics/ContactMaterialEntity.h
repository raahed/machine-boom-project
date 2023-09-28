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

#ifndef GENERATED_AGX_PHYSICS_CONTACTMATERIAL_H_PLUGIN
#define GENERATED_AGX_PHYSICS_CONTACTMATERIAL_H_PLUGIN

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
#include <agx/Integer.h>
#include <agx/Physics/MaterialEntity.h>
#include <agx/Vec3.h>
#include <agx/Real.h>
namespace agx { class ContactMaterial; }

namespace agx { namespace Physics { class MaterialPtr; }}
namespace agx { namespace Physics { class MaterialPtr; }}

namespace agx
{
  namespace Physics
  {

    class ContactMaterialModel;
    class ContactMaterialData;
    class ContactMaterialPtr;
    class ContactMaterialInstance;
    class ContactMaterialSemantics;


    AGX_DECLARE_POINTER_TYPES(ContactMaterialModel);

    /** 
    Abstract description of the data attributes for the Physics.ContactMaterial entity.
    */ 
    class AGXPHYSICS_EXPORT ContactMaterialModel : public agx::ReferencedModel
    {
    public:
      typedef ContactMaterialPtr PtrT;

      ContactMaterialModel(const agx::String& name = "ContactMaterial");

      /// \return The entity model singleton.
      static ContactMaterialModel* instance();

      /// Create and return a pointer to a new instance in the default storage for this entity model.
      static ContactMaterialPtr createInstance();

      /// \return The default storage for this entity model.
      static agxData::EntityStorage* defaultStorage();

      /// This is part of internal cleanup and should not be called by users
      virtual void shutdownCleanup() override;



      /* Attributes */
      static agxData::ScalarAttributeT< agx::Bool >* isExplicitAttribute;
      static agxData::ScalarAttributeT< agx::Physics::MaterialPtr >* material1Attribute;
      static agxData::ScalarAttributeT< agx::Physics::MaterialPtr >* material2Attribute;
      static agxData::ScalarAttributeT< agx::Vec3 >* restitutionAttribute;
      static agxData::ScalarAttributeT< agx::Vec3 >* frictionAttribute;
      static agxData::ScalarAttributeT< agx::Real >* adhesionAttribute;
      static agxData::ScalarAttributeT< agx::Real >* adhesiveOverlapAttribute;
      static agxData::ScalarAttributeT< agx::Vec3 >* viscosityAttribute;
      static agxData::ScalarAttributeT< agx::Bool >* surfaceFrictionEnabledAttribute;
      static agxData::ScalarAttributeT< agx::UInt8 >* contactReductionModeAttribute;
      static agxData::ScalarAttributeT< agx::UInt8 >* contactReductionBinResolutionAttribute;
      static agxData::ScalarAttributeT< agx::Bool >* useContactAreaApproachAttribute;
      static agxData::ScalarAttributeT< agx::Vec3 >* wireFrictionAttribute;
      static agxData::ScalarAttributeT< agx::Real >* youngsModulusAttribute;
      static agxData::ScalarAttributeT< agx::Real >* dampingAttribute;
      static agxData::ScalarAttributeT< agx::Real >* minElasticRestLengthAttribute;
      static agxData::ScalarAttributeT< agx::Real >* maxElasticRestLengthAttribute;
      static agxData::ScalarAttributeT< agx::Real >* impactComplianceAttribute;
      static agxData::ScalarAttributeT< agx::Real >* rollingResistanceCoefficientAttribute;
      static agxData::ScalarAttributeT< agx::Real >* twistingResistanceCoefficientAttribute;
      static agxData::ScalarAttributeT< agx::Real >* rollingResistanceComplianceAttribute;
      static agxData::ScalarAttributeT< agx::Real >* twistingResistanceComplianceAttribute;
      static agxData::PointerAttributeT< agx::ContactMaterial*>* modelAttribute;

    protected:
      virtual ~ContactMaterialModel();
      virtual agxData::EntityData* createData(agxData::EntityStorage* storage) override;
      virtual void configure(agx::TiXmlElement* eEntity) override;
      virtual void initAttributeAccessors() override;
      virtual void construct(agxData::EntityPtr instance) override;
      void construct(agx::Physics::ContactMaterialPtr contactMaterial);
    };


    DOXYGEN_START_INTERNAL_BLOCK()
    #ifndef AGX_PHYSICS_CONTACTMATERIAL_DATA_SET_OVERRIDE
    #define AGX_PHYSICS_CONTACTMATERIAL_DATA_SET
    class AGXPHYSICS_EXPORT ContactMaterialData : public agx::ReferencedData
    {
    public:
      ContactMaterialInstance operator[] (size_t index);

    public:
      agxData::Array< ContactMaterialPtr >& instance;
      agxData::Array< agx::Bool > isExplicit;
      agxData::Array< agx::Physics::MaterialPtr > material1;
      agxData::Array< agx::Physics::MaterialPtr > material2;
      agxData::Array< agx::Vec3 > restitution;
      agxData::Array< agx::Vec3 > friction;
      agxData::Array< agx::Real > adhesion;
      agxData::Array< agx::Real > adhesiveOverlap;
      agxData::Array< agx::Vec3 > viscosity;
      agxData::Array< agx::Bool > surfaceFrictionEnabled;
      agxData::Array< agx::UInt8 > contactReductionMode;
      agxData::Array< agx::UInt8 > contactReductionBinResolution;
      agxData::Array< agx::Bool > useContactAreaApproach;
      agxData::Array< agx::Vec3 > wireFriction;
      agxData::Array< agx::Real > youngsModulus;
      agxData::Array< agx::Real > damping;
      agxData::Array< agx::Real > minElasticRestLength;
      agxData::Array< agx::Real > maxElasticRestLength;
      agxData::Array< agx::Real > impactCompliance;
      agxData::Array< agx::Real > rollingResistanceCoefficient;
      agxData::Array< agx::Real > twistingResistanceCoefficient;
      agxData::Array< agx::Real > rollingResistanceCompliance;
      agxData::Array< agx::Real > twistingResistanceCompliance;
      agxData::Array< agx::ContactMaterial* > model;

    public:
      typedef agx::Bool isExplicitType;
      typedef agx::Physics::MaterialPtr material1Type;
      typedef agx::Physics::MaterialPtr material2Type;
      typedef agx::Vec3 restitutionType;
      typedef agx::Vec3 frictionType;
      typedef agx::Real adhesionType;
      typedef agx::Real adhesiveOverlapType;
      typedef agx::Vec3 viscosityType;
      typedef agx::Bool surfaceFrictionEnabledType;
      typedef agx::UInt8 contactReductionModeType;
      typedef agx::UInt8 contactReductionBinResolutionType;
      typedef agx::Bool useContactAreaApproachType;
      typedef agx::Vec3 wireFrictionType;
      typedef agx::Real youngsModulusType;
      typedef agx::Real dampingType;
      typedef agx::Real minElasticRestLengthType;
      typedef agx::Real maxElasticRestLengthType;
      typedef agx::Real impactComplianceType;
      typedef agx::Real rollingResistanceCoefficientType;
      typedef agx::Real twistingResistanceCoefficientType;
      typedef agx::Real rollingResistanceComplianceType;
      typedef agx::Real twistingResistanceComplianceType;
      typedef agx::ContactMaterial* modelType;

    public:
      ContactMaterialData(agxData::EntityStorage* storage);
      ContactMaterialData();

    protected:
      virtual ~ContactMaterialData() {}
      virtual void setNumElements(agx::Index numElements) override;

    private:
      ContactMaterialData& operator= (const ContactMaterialData&) { return *this; }

    };
    #endif
    DOXYGEN_END_INTERNAL_BLOCK()


    DOXYGEN_START_INTERNAL_BLOCK()
    class AGXPHYSICS_EXPORT ContactMaterialSemantics : public agx::ReferencedSemantics
    {
    public:

      // Automatic getters
      agx::Bool const& getIsExplicit() const;
      agx::Physics::MaterialPtr const& getMaterial1() const;
      agx::Physics::MaterialPtr const& getMaterial2() const;
      agx::Vec3 const& getRestitution() const;
      agx::Vec3 const& getFriction() const;
      agx::Real const& getAdhesion() const;
      agx::Real const& getAdhesiveOverlap() const;
      agx::Vec3 const& getViscosity() const;
      agx::Bool const& getSurfaceFrictionEnabled() const;
      agx::UInt8 const& getContactReductionMode() const;
      agx::UInt8 const& getContactReductionBinResolution() const;
      agx::Bool const& getUseContactAreaApproach() const;
      agx::Vec3 const& getWireFriction() const;
      agx::Real const& getYoungsModulus() const;
      agx::Real const& getDamping() const;
      agx::Real const& getMinElasticRestLength() const;
      agx::Real const& getMaxElasticRestLength() const;
      agx::Real const& getImpactCompliance() const;
      agx::Real const& getRollingResistanceCoefficient() const;
      agx::Real const& getTwistingResistanceCoefficient() const;
      agx::Real const& getRollingResistanceCompliance() const;
      agx::Real const& getTwistingResistanceCompliance() const;
      agx::ContactMaterial* const& getModel() const;

      // Semantics defined by explicit kernels

      // Automatic setters
      void setIsExplicit(agx::Bool const& value);
      void setMaterial1(agx::Physics::MaterialPtr const& value);
      void setMaterial2(agx::Physics::MaterialPtr const& value);
      void setRestitution(agx::Vec3 const& value);
      void setFriction(agx::Vec3 const& value);
      void setAdhesion(agx::Real const& value);
      void setAdhesiveOverlap(agx::Real const& value);
      void setViscosity(agx::Vec3 const& value);
      void setSurfaceFrictionEnabled(agx::Bool const& value);
      void setContactReductionMode(agx::UInt8 const& value);
      void setContactReductionBinResolution(agx::UInt8 const& value);
      void setUseContactAreaApproach(agx::Bool const& value);
      void setWireFriction(agx::Vec3 const& value);
      void setYoungsModulus(agx::Real const& value);
      void setDamping(agx::Real const& value);
      void setMinElasticRestLength(agx::Real const& value);
      void setMaxElasticRestLength(agx::Real const& value);
      void setImpactCompliance(agx::Real const& value);
      void setRollingResistanceCoefficient(agx::Real const& value);
      void setTwistingResistanceCoefficient(agx::Real const& value);
      void setRollingResistanceCompliance(agx::Real const& value);
      void setTwistingResistanceCompliance(agx::Real const& value);
      void setModel(agx::ContactMaterial* const& value);


    protected:
      friend class ContactMaterialPtr;
      friend class ContactMaterialInstance;
      ContactMaterialSemantics();
    };
    DOXYGEN_END_INTERNAL_BLOCK()


    /**
    Pointer to a entity instance of type Physics.ContactMaterial
    */
    class CALLABLE ContactMaterialPtr : public agx::ReferencedPtr
    {
    public:
      typedef ContactMaterialModel ModelType;
      typedef ContactMaterialData DataType;
      typedef ContactMaterialInstance InstanceType;

    public:
      AGXPHYSICS_EXPORT ContactMaterialPtr();
      AGXPHYSICS_EXPORT ContactMaterialPtr(agxData::EntityStorage* storage, agx::Index id);
      AGXPHYSICS_EXPORT ContactMaterialPtr(const agxData::EntityPtr& ptr);
      AGXPHYSICS_EXPORT ContactMaterialPtr(const agxData::EntityInstance& instance);
      AGXPHYSICS_EXPORT ContactMaterialPtr& operator= (const agxData::EntityPtr& ptr);
      AGXPHYSICS_EXPORT ContactMaterialPtr& operator= (const agxData::EntityInstance& instance);
      AGXPHYSICS_EXPORT ContactMaterialInstance instance();
      AGXPHYSICS_EXPORT const ContactMaterialInstance instance() const;

      AGXPHYSICS_EXPORT ContactMaterialSemantics* operator->();
      AGXPHYSICS_EXPORT const ContactMaterialSemantics* operator->() const;

      ContactMaterialData* getData();
      const ContactMaterialData* getData() const;


      /// \return reference to the isExplicit attribute
      AGXPHYSICS_EXPORT agx::Bool& isExplicit();
      /// \return const reference to the isExplicit attribute
      AGXPHYSICS_EXPORT agx::Bool const& isExplicit() const;

      /// \return reference to the material1 attribute
      AGXPHYSICS_EXPORT agx::Physics::MaterialPtr& material1();
      /// \return const reference to the material1 attribute
      AGXPHYSICS_EXPORT agx::Physics::MaterialPtr const& material1() const;

      /// \return reference to the material2 attribute
      AGXPHYSICS_EXPORT agx::Physics::MaterialPtr& material2();
      /// \return const reference to the material2 attribute
      AGXPHYSICS_EXPORT agx::Physics::MaterialPtr const& material2() const;

      /// \return reference to the restitution attribute
      AGXPHYSICS_EXPORT agx::Vec3& restitution();
      /// \return const reference to the restitution attribute
      AGXPHYSICS_EXPORT agx::Vec3 const& restitution() const;

      /// \return reference to the friction attribute
      AGXPHYSICS_EXPORT agx::Vec3& friction();
      /// \return const reference to the friction attribute
      AGXPHYSICS_EXPORT agx::Vec3 const& friction() const;

      /// \return reference to the adhesion attribute
      AGXPHYSICS_EXPORT agx::Real& adhesion();
      /// \return const reference to the adhesion attribute
      AGXPHYSICS_EXPORT agx::Real const& adhesion() const;

      /// \return reference to the adhesiveOverlap attribute
      AGXPHYSICS_EXPORT agx::Real& adhesiveOverlap();
      /// \return const reference to the adhesiveOverlap attribute
      AGXPHYSICS_EXPORT agx::Real const& adhesiveOverlap() const;

      /// \return reference to the viscosity attribute
      AGXPHYSICS_EXPORT agx::Vec3& viscosity();
      /// \return const reference to the viscosity attribute
      AGXPHYSICS_EXPORT agx::Vec3 const& viscosity() const;

      /// \return reference to the surfaceFrictionEnabled attribute
      AGXPHYSICS_EXPORT agx::Bool& surfaceFrictionEnabled();
      /// \return const reference to the surfaceFrictionEnabled attribute
      AGXPHYSICS_EXPORT agx::Bool const& surfaceFrictionEnabled() const;

      /// \return reference to the contactReductionMode attribute
      AGXPHYSICS_EXPORT agx::UInt8& contactReductionMode();
      /// \return const reference to the contactReductionMode attribute
      AGXPHYSICS_EXPORT agx::UInt8 const& contactReductionMode() const;

      /// \return reference to the contactReductionBinResolution attribute
      AGXPHYSICS_EXPORT agx::UInt8& contactReductionBinResolution();
      /// \return const reference to the contactReductionBinResolution attribute
      AGXPHYSICS_EXPORT agx::UInt8 const& contactReductionBinResolution() const;

      /// \return reference to the useContactAreaApproach attribute
      AGXPHYSICS_EXPORT agx::Bool& useContactAreaApproach();
      /// \return const reference to the useContactAreaApproach attribute
      AGXPHYSICS_EXPORT agx::Bool const& useContactAreaApproach() const;

      /// \return reference to the wireFriction attribute
      AGXPHYSICS_EXPORT agx::Vec3& wireFriction();
      /// \return const reference to the wireFriction attribute
      AGXPHYSICS_EXPORT agx::Vec3 const& wireFriction() const;

      /// \return reference to the youngsModulus attribute
      AGXPHYSICS_EXPORT agx::Real& youngsModulus();
      /// \return const reference to the youngsModulus attribute
      AGXPHYSICS_EXPORT agx::Real const& youngsModulus() const;

      /// \return reference to the damping attribute
      AGXPHYSICS_EXPORT agx::Real& damping();
      /// \return const reference to the damping attribute
      AGXPHYSICS_EXPORT agx::Real const& damping() const;

      /// \return reference to the minElasticRestLength attribute
      AGXPHYSICS_EXPORT agx::Real& minElasticRestLength();
      /// \return const reference to the minElasticRestLength attribute
      AGXPHYSICS_EXPORT agx::Real const& minElasticRestLength() const;

      /// \return reference to the maxElasticRestLength attribute
      AGXPHYSICS_EXPORT agx::Real& maxElasticRestLength();
      /// \return const reference to the maxElasticRestLength attribute
      AGXPHYSICS_EXPORT agx::Real const& maxElasticRestLength() const;

      /// \return reference to the impactCompliance attribute
      AGXPHYSICS_EXPORT agx::Real& impactCompliance();
      /// \return const reference to the impactCompliance attribute
      AGXPHYSICS_EXPORT agx::Real const& impactCompliance() const;

      /// \return reference to the rollingResistanceCoefficient attribute
      AGXPHYSICS_EXPORT agx::Real& rollingResistanceCoefficient();
      /// \return const reference to the rollingResistanceCoefficient attribute
      AGXPHYSICS_EXPORT agx::Real const& rollingResistanceCoefficient() const;

      /// \return reference to the twistingResistanceCoefficient attribute
      AGXPHYSICS_EXPORT agx::Real& twistingResistanceCoefficient();
      /// \return const reference to the twistingResistanceCoefficient attribute
      AGXPHYSICS_EXPORT agx::Real const& twistingResistanceCoefficient() const;

      /// \return reference to the rollingResistanceCompliance attribute
      AGXPHYSICS_EXPORT agx::Real& rollingResistanceCompliance();
      /// \return const reference to the rollingResistanceCompliance attribute
      AGXPHYSICS_EXPORT agx::Real const& rollingResistanceCompliance() const;

      /// \return reference to the twistingResistanceCompliance attribute
      AGXPHYSICS_EXPORT agx::Real& twistingResistanceCompliance();
      /// \return const reference to the twistingResistanceCompliance attribute
      AGXPHYSICS_EXPORT agx::Real const& twistingResistanceCompliance() const;

      /// \return reference to the model attribute
      AGXPHYSICS_EXPORT agx::ContactMaterial*& model();
      /// \return const reference to the model attribute
      AGXPHYSICS_EXPORT agx::ContactMaterial* const& model() const;

      DOXYGEN_START_INTERNAL_BLOCK()
      // Only used by tolua!
      #ifdef AGX_GENERATED_TOLUA_WRAPPER
      inline agx::Bool const& getIsExplicit() const { return (*this)->getIsExplicit(); }
      inline agx::Physics::MaterialPtr const& getMaterial1() const { return (*this)->getMaterial1(); }
      inline agx::Physics::MaterialPtr const& getMaterial2() const { return (*this)->getMaterial2(); }
      inline agx::Vec3 const& getRestitution() const { return (*this)->getRestitution(); }
      inline agx::Vec3 const& getFriction() const { return (*this)->getFriction(); }
      inline agx::Real const& getAdhesion() const { return (*this)->getAdhesion(); }
      inline agx::Real const& getAdhesiveOverlap() const { return (*this)->getAdhesiveOverlap(); }
      inline agx::Vec3 const& getViscosity() const { return (*this)->getViscosity(); }
      inline agx::Bool const& getSurfaceFrictionEnabled() const { return (*this)->getSurfaceFrictionEnabled(); }
      inline agx::UInt8 const& getContactReductionMode() const { return (*this)->getContactReductionMode(); }
      inline agx::UInt8 const& getContactReductionBinResolution() const { return (*this)->getContactReductionBinResolution(); }
      inline agx::Bool const& getUseContactAreaApproach() const { return (*this)->getUseContactAreaApproach(); }
      inline agx::Vec3 const& getWireFriction() const { return (*this)->getWireFriction(); }
      inline agx::Real const& getYoungsModulus() const { return (*this)->getYoungsModulus(); }
      inline agx::Real const& getDamping() const { return (*this)->getDamping(); }
      inline agx::Real const& getMinElasticRestLength() const { return (*this)->getMinElasticRestLength(); }
      inline agx::Real const& getMaxElasticRestLength() const { return (*this)->getMaxElasticRestLength(); }
      inline agx::Real const& getImpactCompliance() const { return (*this)->getImpactCompliance(); }
      inline agx::Real const& getRollingResistanceCoefficient() const { return (*this)->getRollingResistanceCoefficient(); }
      inline agx::Real const& getTwistingResistanceCoefficient() const { return (*this)->getTwistingResistanceCoefficient(); }
      inline agx::Real const& getRollingResistanceCompliance() const { return (*this)->getRollingResistanceCompliance(); }
      inline agx::Real const& getTwistingResistanceCompliance() const { return (*this)->getTwistingResistanceCompliance(); }
      inline agx::ContactMaterial* const& getModel() const { return (*this)->getModel(); }


      inline void setIsExplicit(agx::Bool const& value) { (*this)->setIsExplicit(value); }
      inline void setMaterial1(agx::Physics::MaterialPtr const& value) { (*this)->setMaterial1(value); }
      inline void setMaterial2(agx::Physics::MaterialPtr const& value) { (*this)->setMaterial2(value); }
      inline void setRestitution(agx::Vec3 const& value) { (*this)->setRestitution(value); }
      inline void setFriction(agx::Vec3 const& value) { (*this)->setFriction(value); }
      inline void setAdhesion(agx::Real const& value) { (*this)->setAdhesion(value); }
      inline void setAdhesiveOverlap(agx::Real const& value) { (*this)->setAdhesiveOverlap(value); }
      inline void setViscosity(agx::Vec3 const& value) { (*this)->setViscosity(value); }
      inline void setSurfaceFrictionEnabled(agx::Bool const& value) { (*this)->setSurfaceFrictionEnabled(value); }
      inline void setContactReductionMode(agx::UInt8 const& value) { (*this)->setContactReductionMode(value); }
      inline void setContactReductionBinResolution(agx::UInt8 const& value) { (*this)->setContactReductionBinResolution(value); }
      inline void setUseContactAreaApproach(agx::Bool const& value) { (*this)->setUseContactAreaApproach(value); }
      inline void setWireFriction(agx::Vec3 const& value) { (*this)->setWireFriction(value); }
      inline void setYoungsModulus(agx::Real const& value) { (*this)->setYoungsModulus(value); }
      inline void setDamping(agx::Real const& value) { (*this)->setDamping(value); }
      inline void setMinElasticRestLength(agx::Real const& value) { (*this)->setMinElasticRestLength(value); }
      inline void setMaxElasticRestLength(agx::Real const& value) { (*this)->setMaxElasticRestLength(value); }
      inline void setImpactCompliance(agx::Real const& value) { (*this)->setImpactCompliance(value); }
      inline void setRollingResistanceCoefficient(agx::Real const& value) { (*this)->setRollingResistanceCoefficient(value); }
      inline void setTwistingResistanceCoefficient(agx::Real const& value) { (*this)->setTwistingResistanceCoefficient(value); }
      inline void setRollingResistanceCompliance(agx::Real const& value) { (*this)->setRollingResistanceCompliance(value); }
      inline void setTwistingResistanceCompliance(agx::Real const& value) { (*this)->setTwistingResistanceCompliance(value); }
      inline void setModel(agx::ContactMaterial* const& value) { (*this)->setModel(value); }
      #endif
      DOXYGEN_END_INTERNAL_BLOCK()

    };

    // Entity is Referenced
    typedef agxData::EntityRef< ContactMaterialPtr > ContactMaterialRef;


    DOXYGEN_START_INTERNAL_BLOCK()
    class AGXPHYSICS_EXPORT ContactMaterialInstance : public agx::ReferencedInstance
    {
    public:
      ContactMaterialInstance();
      ContactMaterialInstance(ContactMaterialData* data, agx::Index index);
      ContactMaterialInstance(agxData::EntityStorage *storage, agx::Index index);
      ContactMaterialInstance(const agxData::EntityInstance& other);
      ContactMaterialInstance(const agxData::EntityPtr& ptr);

      ContactMaterialData* getData();
      const ContactMaterialData* getData() const;

    public:
      /// \return reference to the isExplicit attribute
      agx::Bool& isExplicit();
      /// \return const reference to the isExplicit attribute
      agx::Bool const& isExplicit() const;

      /// \return reference to the material1 attribute
      agx::Physics::MaterialPtr& material1();
      /// \return const reference to the material1 attribute
      agx::Physics::MaterialPtr const& material1() const;

      /// \return reference to the material2 attribute
      agx::Physics::MaterialPtr& material2();
      /// \return const reference to the material2 attribute
      agx::Physics::MaterialPtr const& material2() const;

      /// \return reference to the restitution attribute
      agx::Vec3& restitution();
      /// \return const reference to the restitution attribute
      agx::Vec3 const& restitution() const;

      /// \return reference to the friction attribute
      agx::Vec3& friction();
      /// \return const reference to the friction attribute
      agx::Vec3 const& friction() const;

      /// \return reference to the adhesion attribute
      agx::Real& adhesion();
      /// \return const reference to the adhesion attribute
      agx::Real const& adhesion() const;

      /// \return reference to the adhesiveOverlap attribute
      agx::Real& adhesiveOverlap();
      /// \return const reference to the adhesiveOverlap attribute
      agx::Real const& adhesiveOverlap() const;

      /// \return reference to the viscosity attribute
      agx::Vec3& viscosity();
      /// \return const reference to the viscosity attribute
      agx::Vec3 const& viscosity() const;

      /// \return reference to the surfaceFrictionEnabled attribute
      agx::Bool& surfaceFrictionEnabled();
      /// \return const reference to the surfaceFrictionEnabled attribute
      agx::Bool const& surfaceFrictionEnabled() const;

      /// \return reference to the contactReductionMode attribute
      agx::UInt8& contactReductionMode();
      /// \return const reference to the contactReductionMode attribute
      agx::UInt8 const& contactReductionMode() const;

      /// \return reference to the contactReductionBinResolution attribute
      agx::UInt8& contactReductionBinResolution();
      /// \return const reference to the contactReductionBinResolution attribute
      agx::UInt8 const& contactReductionBinResolution() const;

      /// \return reference to the useContactAreaApproach attribute
      agx::Bool& useContactAreaApproach();
      /// \return const reference to the useContactAreaApproach attribute
      agx::Bool const& useContactAreaApproach() const;

      /// \return reference to the wireFriction attribute
      agx::Vec3& wireFriction();
      /// \return const reference to the wireFriction attribute
      agx::Vec3 const& wireFriction() const;

      /// \return reference to the youngsModulus attribute
      agx::Real& youngsModulus();
      /// \return const reference to the youngsModulus attribute
      agx::Real const& youngsModulus() const;

      /// \return reference to the damping attribute
      agx::Real& damping();
      /// \return const reference to the damping attribute
      agx::Real const& damping() const;

      /// \return reference to the minElasticRestLength attribute
      agx::Real& minElasticRestLength();
      /// \return const reference to the minElasticRestLength attribute
      agx::Real const& minElasticRestLength() const;

      /// \return reference to the maxElasticRestLength attribute
      agx::Real& maxElasticRestLength();
      /// \return const reference to the maxElasticRestLength attribute
      agx::Real const& maxElasticRestLength() const;

      /// \return reference to the impactCompliance attribute
      agx::Real& impactCompliance();
      /// \return const reference to the impactCompliance attribute
      agx::Real const& impactCompliance() const;

      /// \return reference to the rollingResistanceCoefficient attribute
      agx::Real& rollingResistanceCoefficient();
      /// \return const reference to the rollingResistanceCoefficient attribute
      agx::Real const& rollingResistanceCoefficient() const;

      /// \return reference to the twistingResistanceCoefficient attribute
      agx::Real& twistingResistanceCoefficient();
      /// \return const reference to the twistingResistanceCoefficient attribute
      agx::Real const& twistingResistanceCoefficient() const;

      /// \return reference to the rollingResistanceCompliance attribute
      agx::Real& rollingResistanceCompliance();
      /// \return const reference to the rollingResistanceCompliance attribute
      agx::Real const& rollingResistanceCompliance() const;

      /// \return reference to the twistingResistanceCompliance attribute
      agx::Real& twistingResistanceCompliance();
      /// \return const reference to the twistingResistanceCompliance attribute
      agx::Real const& twistingResistanceCompliance() const;

      /// \return reference to the model attribute
      agx::ContactMaterial*& model();
      /// \return const reference to the model attribute
      agx::ContactMaterial* const& model() const;

    };
    DOXYGEN_END_INTERNAL_BLOCK()



    typedef agx::VectorPOD<ContactMaterialPtr> ContactMaterialPtrVector;
    typedef agxData::Array<ContactMaterialPtr> ContactMaterialPtrArray;



    DOXYGEN_START_INTERNAL_BLOCK()
    /* Implementation */
    //-----------------------------------------------------------------------------------------------------
    //-----------------------------------------------------------------------------------------------------
    inline ContactMaterialInstance agx::Physics::ContactMaterialData::operator[] (size_t index) { return ContactMaterialInstance(this, (agx::Index)index); }
    //-----------------------------------------------------------------------------------------------------
    AGX_FORCE_INLINE ContactMaterialPtr::ContactMaterialPtr() {}
    AGX_FORCE_INLINE ContactMaterialPtr::ContactMaterialPtr(agxData::EntityStorage* storage, agx::Index id) : agx::ReferencedPtr(storage, id) {}
    AGX_FORCE_INLINE ContactMaterialPtr::ContactMaterialPtr(const agxData::EntityPtr& ptr) : agx::ReferencedPtr(ptr)
    {
      agxAssertN(!ptr || ptr.isInstanceOf(ContactMaterialModel::instance()),
        "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
        EntityPtr::getModel()->fullPath().c_str(), ContactMaterialModel::instance()->fullPath().c_str());
    }

    AGX_FORCE_INLINE ContactMaterialPtr::ContactMaterialPtr(const agxData::EntityInstance& instance) : agx::ReferencedPtr(instance)
    {
      agxAssertN(!instance || instance.isInstanceOf(ContactMaterialModel::instance()),
        "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
        EntityPtr::getModel()->fullPath().c_str(), ContactMaterialModel::instance()->fullPath().c_str());
    }

    AGX_FORCE_INLINE ContactMaterialPtr& ContactMaterialPtr::operator= (const agxData::EntityPtr& ptr)
    {
      agxData::EntityPtr::operator= (ptr);
      agxAssertN(!ptr || ptr.isInstanceOf(ContactMaterialModel::instance()),
        "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
        EntityPtr::getModel()->fullPath().c_str(), ContactMaterialModel::instance()->fullPath().c_str());
      return *this;
    }

    AGX_FORCE_INLINE ContactMaterialPtr& ContactMaterialPtr::operator= (const agxData::EntityInstance& instance)
    {
      agxData::EntityPtr::operator= (instance);
      agxAssertN(!instance || instance.isInstanceOf(ContactMaterialModel::instance()),
        "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
        EntityPtr::getModel()->fullPath().c_str(), ContactMaterialModel::instance()->fullPath().c_str());
      return *this;
    }

    inline ContactMaterialInstance ContactMaterialPtr::instance() { return agxData::EntityPtr::instance(); }
    inline const ContactMaterialInstance ContactMaterialPtr::instance() const { return agxData::EntityPtr::instance(); }
    AGX_FORCE_INLINE ContactMaterialSemantics* ContactMaterialPtr::operator->() { return (ContactMaterialSemantics* )this; }
    AGX_FORCE_INLINE const ContactMaterialSemantics* ContactMaterialPtr::operator->() const { return (const ContactMaterialSemantics* )this; }
    AGX_FORCE_INLINE ContactMaterialData* ContactMaterialPtr::getData() { return static_cast<ContactMaterialData* >(agxData::EntityPtr::getData()); }
    AGX_FORCE_INLINE const ContactMaterialData* ContactMaterialPtr::getData() const { return static_cast<const ContactMaterialData* >(agxData::EntityPtr::getData()); }

    AGX_FORCE_INLINE agx::Bool& ContactMaterialPtr::isExplicit() { verifyIndex(); return getData()->isExplicit[calculateIndex()]; }
    AGX_FORCE_INLINE agx::Bool const& ContactMaterialPtr::isExplicit() const { verifyIndex(); return getData()->isExplicit[calculateIndex()]; }

    AGX_FORCE_INLINE agx::Physics::MaterialPtr& ContactMaterialPtr::material1() { verifyIndex(); return getData()->material1[calculateIndex()]; }
    AGX_FORCE_INLINE agx::Physics::MaterialPtr const& ContactMaterialPtr::material1() const { verifyIndex(); return getData()->material1[calculateIndex()]; }

    AGX_FORCE_INLINE agx::Physics::MaterialPtr& ContactMaterialPtr::material2() { verifyIndex(); return getData()->material2[calculateIndex()]; }
    AGX_FORCE_INLINE agx::Physics::MaterialPtr const& ContactMaterialPtr::material2() const { verifyIndex(); return getData()->material2[calculateIndex()]; }

    AGX_FORCE_INLINE agx::Vec3& ContactMaterialPtr::restitution() { verifyIndex(); return getData()->restitution[calculateIndex()]; }
    AGX_FORCE_INLINE agx::Vec3 const& ContactMaterialPtr::restitution() const { verifyIndex(); return getData()->restitution[calculateIndex()]; }

    AGX_FORCE_INLINE agx::Vec3& ContactMaterialPtr::friction() { verifyIndex(); return getData()->friction[calculateIndex()]; }
    AGX_FORCE_INLINE agx::Vec3 const& ContactMaterialPtr::friction() const { verifyIndex(); return getData()->friction[calculateIndex()]; }

    AGX_FORCE_INLINE agx::Real& ContactMaterialPtr::adhesion() { verifyIndex(); return getData()->adhesion[calculateIndex()]; }
    AGX_FORCE_INLINE agx::Real const& ContactMaterialPtr::adhesion() const { verifyIndex(); return getData()->adhesion[calculateIndex()]; }

    AGX_FORCE_INLINE agx::Real& ContactMaterialPtr::adhesiveOverlap() { verifyIndex(); return getData()->adhesiveOverlap[calculateIndex()]; }
    AGX_FORCE_INLINE agx::Real const& ContactMaterialPtr::adhesiveOverlap() const { verifyIndex(); return getData()->adhesiveOverlap[calculateIndex()]; }

    AGX_FORCE_INLINE agx::Vec3& ContactMaterialPtr::viscosity() { verifyIndex(); return getData()->viscosity[calculateIndex()]; }
    AGX_FORCE_INLINE agx::Vec3 const& ContactMaterialPtr::viscosity() const { verifyIndex(); return getData()->viscosity[calculateIndex()]; }

    AGX_FORCE_INLINE agx::Bool& ContactMaterialPtr::surfaceFrictionEnabled() { verifyIndex(); return getData()->surfaceFrictionEnabled[calculateIndex()]; }
    AGX_FORCE_INLINE agx::Bool const& ContactMaterialPtr::surfaceFrictionEnabled() const { verifyIndex(); return getData()->surfaceFrictionEnabled[calculateIndex()]; }

    AGX_FORCE_INLINE agx::UInt8& ContactMaterialPtr::contactReductionMode() { verifyIndex(); return getData()->contactReductionMode[calculateIndex()]; }
    AGX_FORCE_INLINE agx::UInt8 const& ContactMaterialPtr::contactReductionMode() const { verifyIndex(); return getData()->contactReductionMode[calculateIndex()]; }

    AGX_FORCE_INLINE agx::UInt8& ContactMaterialPtr::contactReductionBinResolution() { verifyIndex(); return getData()->contactReductionBinResolution[calculateIndex()]; }
    AGX_FORCE_INLINE agx::UInt8 const& ContactMaterialPtr::contactReductionBinResolution() const { verifyIndex(); return getData()->contactReductionBinResolution[calculateIndex()]; }

    AGX_FORCE_INLINE agx::Bool& ContactMaterialPtr::useContactAreaApproach() { verifyIndex(); return getData()->useContactAreaApproach[calculateIndex()]; }
    AGX_FORCE_INLINE agx::Bool const& ContactMaterialPtr::useContactAreaApproach() const { verifyIndex(); return getData()->useContactAreaApproach[calculateIndex()]; }

    AGX_FORCE_INLINE agx::Vec3& ContactMaterialPtr::wireFriction() { verifyIndex(); return getData()->wireFriction[calculateIndex()]; }
    AGX_FORCE_INLINE agx::Vec3 const& ContactMaterialPtr::wireFriction() const { verifyIndex(); return getData()->wireFriction[calculateIndex()]; }

    AGX_FORCE_INLINE agx::Real& ContactMaterialPtr::youngsModulus() { verifyIndex(); return getData()->youngsModulus[calculateIndex()]; }
    AGX_FORCE_INLINE agx::Real const& ContactMaterialPtr::youngsModulus() const { verifyIndex(); return getData()->youngsModulus[calculateIndex()]; }

    AGX_FORCE_INLINE agx::Real& ContactMaterialPtr::damping() { verifyIndex(); return getData()->damping[calculateIndex()]; }
    AGX_FORCE_INLINE agx::Real const& ContactMaterialPtr::damping() const { verifyIndex(); return getData()->damping[calculateIndex()]; }

    AGX_FORCE_INLINE agx::Real& ContactMaterialPtr::minElasticRestLength() { verifyIndex(); return getData()->minElasticRestLength[calculateIndex()]; }
    AGX_FORCE_INLINE agx::Real const& ContactMaterialPtr::minElasticRestLength() const { verifyIndex(); return getData()->minElasticRestLength[calculateIndex()]; }

    AGX_FORCE_INLINE agx::Real& ContactMaterialPtr::maxElasticRestLength() { verifyIndex(); return getData()->maxElasticRestLength[calculateIndex()]; }
    AGX_FORCE_INLINE agx::Real const& ContactMaterialPtr::maxElasticRestLength() const { verifyIndex(); return getData()->maxElasticRestLength[calculateIndex()]; }

    AGX_FORCE_INLINE agx::Real& ContactMaterialPtr::impactCompliance() { verifyIndex(); return getData()->impactCompliance[calculateIndex()]; }
    AGX_FORCE_INLINE agx::Real const& ContactMaterialPtr::impactCompliance() const { verifyIndex(); return getData()->impactCompliance[calculateIndex()]; }

    AGX_FORCE_INLINE agx::Real& ContactMaterialPtr::rollingResistanceCoefficient() { verifyIndex(); return getData()->rollingResistanceCoefficient[calculateIndex()]; }
    AGX_FORCE_INLINE agx::Real const& ContactMaterialPtr::rollingResistanceCoefficient() const { verifyIndex(); return getData()->rollingResistanceCoefficient[calculateIndex()]; }

    AGX_FORCE_INLINE agx::Real& ContactMaterialPtr::twistingResistanceCoefficient() { verifyIndex(); return getData()->twistingResistanceCoefficient[calculateIndex()]; }
    AGX_FORCE_INLINE agx::Real const& ContactMaterialPtr::twistingResistanceCoefficient() const { verifyIndex(); return getData()->twistingResistanceCoefficient[calculateIndex()]; }

    AGX_FORCE_INLINE agx::Real& ContactMaterialPtr::rollingResistanceCompliance() { verifyIndex(); return getData()->rollingResistanceCompliance[calculateIndex()]; }
    AGX_FORCE_INLINE agx::Real const& ContactMaterialPtr::rollingResistanceCompliance() const { verifyIndex(); return getData()->rollingResistanceCompliance[calculateIndex()]; }

    AGX_FORCE_INLINE agx::Real& ContactMaterialPtr::twistingResistanceCompliance() { verifyIndex(); return getData()->twistingResistanceCompliance[calculateIndex()]; }
    AGX_FORCE_INLINE agx::Real const& ContactMaterialPtr::twistingResistanceCompliance() const { verifyIndex(); return getData()->twistingResistanceCompliance[calculateIndex()]; }

    AGX_FORCE_INLINE agx::ContactMaterial*& ContactMaterialPtr::model() { verifyIndex(); return getData()->model[calculateIndex()]; }
    AGX_FORCE_INLINE agx::ContactMaterial* const& ContactMaterialPtr::model() const { verifyIndex(); return getData()->model[calculateIndex()]; }

    //-----------------------------------------------------------------------------------------------------
    AGX_FORCE_INLINE ContactMaterialInstance::ContactMaterialInstance() {}
    AGX_FORCE_INLINE ContactMaterialInstance::ContactMaterialInstance(ContactMaterialData* data, agx::Index index) : agx::ReferencedInstance(data, index) {}
    AGX_FORCE_INLINE ContactMaterialInstance::ContactMaterialInstance(agxData::EntityStorage* storage, agx::Index index) : agx::ReferencedInstance(storage, index) {}
    AGX_FORCE_INLINE ContactMaterialInstance::ContactMaterialInstance(const agxData::EntityInstance& other) : agx::ReferencedInstance(other)
    {
      agxAssertN(!other || other.isInstanceOf(ContactMaterialModel::instance()),
        "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
        EntityInstance::getModel()->fullPath().c_str(), ContactMaterialModel::instance()->fullPath().c_str());
    }

    AGX_FORCE_INLINE ContactMaterialInstance::ContactMaterialInstance(const agxData::EntityPtr& ptr) : agx::ReferencedInstance(ptr)
    {
      agxAssertN(!ptr || ptr.isInstanceOf(ContactMaterialModel::instance()),
        "Entity type mismatch. Storage contains entity data for type \'%s\', instance pointer is of type \'%s\'",
        EntityInstance::getModel()->fullPath().c_str(), ContactMaterialModel::instance()->fullPath().c_str());
    }


    AGX_FORCE_INLINE ContactMaterialData* ContactMaterialInstance::getData() { return static_cast<ContactMaterialData* >(agxData::EntityInstance::getData()); }
    AGX_FORCE_INLINE const ContactMaterialData* ContactMaterialInstance::getData() const { return static_cast<const ContactMaterialData* >(agxData::EntityInstance::getData()); }

    AGX_FORCE_INLINE agx::Bool& ContactMaterialInstance::isExplicit() { verifyIndex(); return getData()->isExplicit[getIndex()]; }
    AGX_FORCE_INLINE agx::Bool const& ContactMaterialInstance::isExplicit() const { verifyIndex(); return getData()->isExplicit[getIndex()]; }

    AGX_FORCE_INLINE agx::Physics::MaterialPtr& ContactMaterialInstance::material1() { verifyIndex(); return getData()->material1[getIndex()]; }
    AGX_FORCE_INLINE agx::Physics::MaterialPtr const& ContactMaterialInstance::material1() const { verifyIndex(); return getData()->material1[getIndex()]; }

    AGX_FORCE_INLINE agx::Physics::MaterialPtr& ContactMaterialInstance::material2() { verifyIndex(); return getData()->material2[getIndex()]; }
    AGX_FORCE_INLINE agx::Physics::MaterialPtr const& ContactMaterialInstance::material2() const { verifyIndex(); return getData()->material2[getIndex()]; }

    AGX_FORCE_INLINE agx::Vec3& ContactMaterialInstance::restitution() { verifyIndex(); return getData()->restitution[getIndex()]; }
    AGX_FORCE_INLINE agx::Vec3 const& ContactMaterialInstance::restitution() const { verifyIndex(); return getData()->restitution[getIndex()]; }

    AGX_FORCE_INLINE agx::Vec3& ContactMaterialInstance::friction() { verifyIndex(); return getData()->friction[getIndex()]; }
    AGX_FORCE_INLINE agx::Vec3 const& ContactMaterialInstance::friction() const { verifyIndex(); return getData()->friction[getIndex()]; }

    AGX_FORCE_INLINE agx::Real& ContactMaterialInstance::adhesion() { verifyIndex(); return getData()->adhesion[getIndex()]; }
    AGX_FORCE_INLINE agx::Real const& ContactMaterialInstance::adhesion() const { verifyIndex(); return getData()->adhesion[getIndex()]; }

    AGX_FORCE_INLINE agx::Real& ContactMaterialInstance::adhesiveOverlap() { verifyIndex(); return getData()->adhesiveOverlap[getIndex()]; }
    AGX_FORCE_INLINE agx::Real const& ContactMaterialInstance::adhesiveOverlap() const { verifyIndex(); return getData()->adhesiveOverlap[getIndex()]; }

    AGX_FORCE_INLINE agx::Vec3& ContactMaterialInstance::viscosity() { verifyIndex(); return getData()->viscosity[getIndex()]; }
    AGX_FORCE_INLINE agx::Vec3 const& ContactMaterialInstance::viscosity() const { verifyIndex(); return getData()->viscosity[getIndex()]; }

    AGX_FORCE_INLINE agx::Bool& ContactMaterialInstance::surfaceFrictionEnabled() { verifyIndex(); return getData()->surfaceFrictionEnabled[getIndex()]; }
    AGX_FORCE_INLINE agx::Bool const& ContactMaterialInstance::surfaceFrictionEnabled() const { verifyIndex(); return getData()->surfaceFrictionEnabled[getIndex()]; }

    AGX_FORCE_INLINE agx::UInt8& ContactMaterialInstance::contactReductionMode() { verifyIndex(); return getData()->contactReductionMode[getIndex()]; }
    AGX_FORCE_INLINE agx::UInt8 const& ContactMaterialInstance::contactReductionMode() const { verifyIndex(); return getData()->contactReductionMode[getIndex()]; }

    AGX_FORCE_INLINE agx::UInt8& ContactMaterialInstance::contactReductionBinResolution() { verifyIndex(); return getData()->contactReductionBinResolution[getIndex()]; }
    AGX_FORCE_INLINE agx::UInt8 const& ContactMaterialInstance::contactReductionBinResolution() const { verifyIndex(); return getData()->contactReductionBinResolution[getIndex()]; }

    AGX_FORCE_INLINE agx::Bool& ContactMaterialInstance::useContactAreaApproach() { verifyIndex(); return getData()->useContactAreaApproach[getIndex()]; }
    AGX_FORCE_INLINE agx::Bool const& ContactMaterialInstance::useContactAreaApproach() const { verifyIndex(); return getData()->useContactAreaApproach[getIndex()]; }

    AGX_FORCE_INLINE agx::Vec3& ContactMaterialInstance::wireFriction() { verifyIndex(); return getData()->wireFriction[getIndex()]; }
    AGX_FORCE_INLINE agx::Vec3 const& ContactMaterialInstance::wireFriction() const { verifyIndex(); return getData()->wireFriction[getIndex()]; }

    AGX_FORCE_INLINE agx::Real& ContactMaterialInstance::youngsModulus() { verifyIndex(); return getData()->youngsModulus[getIndex()]; }
    AGX_FORCE_INLINE agx::Real const& ContactMaterialInstance::youngsModulus() const { verifyIndex(); return getData()->youngsModulus[getIndex()]; }

    AGX_FORCE_INLINE agx::Real& ContactMaterialInstance::damping() { verifyIndex(); return getData()->damping[getIndex()]; }
    AGX_FORCE_INLINE agx::Real const& ContactMaterialInstance::damping() const { verifyIndex(); return getData()->damping[getIndex()]; }

    AGX_FORCE_INLINE agx::Real& ContactMaterialInstance::minElasticRestLength() { verifyIndex(); return getData()->minElasticRestLength[getIndex()]; }
    AGX_FORCE_INLINE agx::Real const& ContactMaterialInstance::minElasticRestLength() const { verifyIndex(); return getData()->minElasticRestLength[getIndex()]; }

    AGX_FORCE_INLINE agx::Real& ContactMaterialInstance::maxElasticRestLength() { verifyIndex(); return getData()->maxElasticRestLength[getIndex()]; }
    AGX_FORCE_INLINE agx::Real const& ContactMaterialInstance::maxElasticRestLength() const { verifyIndex(); return getData()->maxElasticRestLength[getIndex()]; }

    AGX_FORCE_INLINE agx::Real& ContactMaterialInstance::impactCompliance() { verifyIndex(); return getData()->impactCompliance[getIndex()]; }
    AGX_FORCE_INLINE agx::Real const& ContactMaterialInstance::impactCompliance() const { verifyIndex(); return getData()->impactCompliance[getIndex()]; }

    AGX_FORCE_INLINE agx::Real& ContactMaterialInstance::rollingResistanceCoefficient() { verifyIndex(); return getData()->rollingResistanceCoefficient[getIndex()]; }
    AGX_FORCE_INLINE agx::Real const& ContactMaterialInstance::rollingResistanceCoefficient() const { verifyIndex(); return getData()->rollingResistanceCoefficient[getIndex()]; }

    AGX_FORCE_INLINE agx::Real& ContactMaterialInstance::twistingResistanceCoefficient() { verifyIndex(); return getData()->twistingResistanceCoefficient[getIndex()]; }
    AGX_FORCE_INLINE agx::Real const& ContactMaterialInstance::twistingResistanceCoefficient() const { verifyIndex(); return getData()->twistingResistanceCoefficient[getIndex()]; }

    AGX_FORCE_INLINE agx::Real& ContactMaterialInstance::rollingResistanceCompliance() { verifyIndex(); return getData()->rollingResistanceCompliance[getIndex()]; }
    AGX_FORCE_INLINE agx::Real const& ContactMaterialInstance::rollingResistanceCompliance() const { verifyIndex(); return getData()->rollingResistanceCompliance[getIndex()]; }

    AGX_FORCE_INLINE agx::Real& ContactMaterialInstance::twistingResistanceCompliance() { verifyIndex(); return getData()->twistingResistanceCompliance[getIndex()]; }
    AGX_FORCE_INLINE agx::Real const& ContactMaterialInstance::twistingResistanceCompliance() const { verifyIndex(); return getData()->twistingResistanceCompliance[getIndex()]; }

    AGX_FORCE_INLINE agx::ContactMaterial*& ContactMaterialInstance::model() { verifyIndex(); return getData()->model[getIndex()]; }
    AGX_FORCE_INLINE agx::ContactMaterial* const& ContactMaterialInstance::model() const { verifyIndex(); return getData()->model[getIndex()]; }

    //-----------------------------------------------------------------------------------------------------
    AGX_FORCE_INLINE ContactMaterialSemantics::ContactMaterialSemantics() {}
    //-----------------------------------------------------------------------------------------------------
    DOXYGEN_END_INTERNAL_BLOCK()
  }
}

AGX_TYPE_BINDING(agx::Physics::ContactMaterialPtr, "Physics.ContactMaterialPtr")
AGX_TYPE_BINDING(agx::Physics::ContactMaterialInstance, "Physics.ContactMaterialInstance")

#ifdef _MSC_VER
# pragma warning(pop)
#endif

#undef AGX_ENTITY_WRAPPER
#undef AGX_ENTITY_NAMESPACE
#endif

