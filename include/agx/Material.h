/*
Copyright 2007-2023. Algoryx Simulation AB.

All AGX source code, intellectual property, documentation, sample code,
tutorials, scene files and technical white papers, are copyrighted, proprietary
and confidential material of Algoryx Simulation AB. You may not download, read,
store, distribute, publish, copy or otherwise disseminate, use or expose this
material unless having a written signed agreement with Algoryx Simulation AB, or having been
advised so by Algoryx Simulation AB for a time limited evaluation, or having purchased a
valid commercial license from Algoryx Simulation AB.

Algoryx Simulation AB disclaims all responsibilities for loss or damage caused
from using this software, unless otherwise stated in written agreements with
Algoryx Simulation AB.
*/

#pragma once


#include <agx/agxPhysics_export.h>
#include <agxStream/Serializable.h>
#include <agx/FrictionModel.h>
#include <agx/Physics/MaterialEntity.h>
#include <agx/Physics/BulkMaterialEntity.h>
#include <agx/Physics/SurfaceMaterialEntity.h>
#include <agx/Physics/WireMaterialEntity.h>
#include <agx/Physics/ContactMaterialEntity.h>
#include <agx/macros.h>
#include <agx/SetVector.h>


#ifdef _MSC_VER
# pragma warning(push)
# pragma warning(disable: 4251) // warning C4251: class X needs to have dll-interface to be used by clients of class Y
#endif

namespace agxSDK
{
  class MaterialManager;
}
namespace agxIO
{
  class SceneExporter;
}
namespace agxCollide
{
  class GeometryContact;
}
namespace agxData
{
  class EntityStorage;
}

namespace agx
{
  AGX_DECLARE_POINTER_TYPES(BulkMaterial);
  AGX_DECLARE_POINTER_TYPES(SurfaceMaterial);
  AGX_DECLARE_POINTER_TYPES(WireMaterial);
  AGX_DECLARE_POINTER_TYPES(Material);
  AGX_DECLARE_POINTER_TYPES(ContactMaterial);
  typedef agx::SetVector< agx::ref_ptr<agx::ContactMaterial> > ContactMaterialRefSetVector;
  typedef agx::Vector<agx::ref_ptr<agx::ContactMaterial> > ContactMaterialRefVector;


  /**
  Physical properties for the bulk of geometries made up by a Material.
  */
  class AGXPHYSICS_EXPORT BulkMaterial : public Referenced, public virtual agxStream::Serializable
  {
    public:

      /// Default constructor
      BulkMaterial();

      /**
      Set the density of a material.
      The density can be used for automatic calculation of massProperties.
      Observe that changing the density does NOT lead to an automatically new calculated
      mass/inertia for any RigidBodies that have Geometries with this material.
      To reflect a change in density, an explicit call to RigidBody::updateMassProperties() has to be done.
      */
      void setDensity(Real density);

      /**
      Get the density of a material.
      \return The density of the material in SI units.
      */
      Real getDensity() const;

      /**
      Set the Young's modulus of the material, same as spring coefficient k.
      */
      void setYoungsModulus(Real value);

      /**
      \return the Young's modulus of the material.
      */
      Real getYoungsModulus() const;     

      /**
      This is the damping factor that is used by the contact constraint.
      The value is the time the contact constraint has to fulfill its violation.
      */
      Real getDamping() const;

      /**
      Set the damping factor used by the contact constraint.
      The value is the time the contact constraint has to fulfill its violation.
      \param damping - the new damping.
      */
      void setDamping( Real damping );

      /**
      Set the minimum and maximum elastic rest length of the bulk material.
      This is only used if the contact area approach is used (getUseContactAreaApproach()==true).
      It sets the minimum and maximum allowed value for the elastic rest length of the material - values
      below will be clamped to the min, values above will be clamped to the max.
      In contact between materials m1 and m2, the min elastic rest of the contact material cm(m1, m2) will
      have the sum of m1's and m2's min elastic rest length, and the max elastic rest length of cm(m1, m2) will
      be the sum of m1's and m2's max elastic rest length, unless these values are set at the contact material directly.
      For the effect of the elastic rest length, see the user manual.
      \param minElasticRestLength: The min elastic rest length. Only values > 0 allowed. Has to be <= maxElasticRestLength.
      \param maxElasticRestLength: The max elastic rest length. Only values > 0 allowed. Has to be >= minElasticRestLength.
      \retval Success - Are the values in the allowed range and have thus been applied?
      */
      bool setMinMaxElasticRestLength(Real minElasticRestLength, Real maxElasticRestLength);

      /**
      Get the minimum elastic rest length of the bulk material.
      This is only used if the contact area approach is used (getUseContactAreaApproach()==true).
      For the effect of the elastic rest length, see the user manual.
      \retval minElasticRestLength: The min elastic rest length.
      */
      Real getMinElasticRestLength() const;

      /**
      Get the maximum elastic rest length of the bulk material.
      This is only used if the contact area approach is used (getUseContactAreaApproach()==true).
      For the effect of the elastic rest length, see the user manual.
      \retval maxElasticRestLength: The max elastic rest length.
      */
      Real getMaxElasticRestLength() const;

      /// Copy all the attributes of \p copy_material to this.
      void copy( const BulkMaterial* source );

      /**
      Set the bulk viscosity coefficient of the material (1 - restitution coefficient).
      \param viscosity - The new viscosity
      */
      void setViscosity( Real viscosity );

      /**
      \return the bulk viscosity coefficient (1 - restitution coefficient) of this material.
      */
      Real getViscosity() const;

      /**
      \return true if any of the attributes in the BulkMaterial is changed since last setDirty( false );
      */
      bool isDirty() const;

      DOXYGEN_START_INTERNAL_BLOCK()


      AGXSTREAM_DECLARE_SERIALIZABLE( agx::BulkMaterial );

      /// \return reference to the internal entity storage
      Physics::BulkMaterialPtr& getEntity();

      /// \return reference to the internal entity storage
      const Physics::BulkMaterialPtr& getEntity() const;

    protected:

      friend class Material;

      AGX_FORCE_INLINE void setDirty( bool flag ) const;
      DOXYGEN_END_INTERNAL_BLOCK()

      /// Destructor
      virtual ~BulkMaterial();



    private:
      mutable Physics::BulkMaterialRef m_entity;
  };


  /**
  Physical properties for the surface of a Material.
  */
  class AGXPHYSICS_EXPORT SurfaceMaterial :  public Referenced, public virtual agxStream::Serializable
  {
    public:

      /// Default constructor
      SurfaceMaterial();

      /**
      Set the roughness parameter. A unit less parameter that corresponds to the final
      friction coefficient.
      */
      void setRoughness( Real roughness );

      /**
      \return the roughness parameter. A unit less parameter that corresponds to the final
      friction coefficient.
      */
      Real getRoughness() const;

      /**
      Set the adhesive force and allowed overlap between two colliding objects.
      \param adhesion - adhesive force >= 0, i.e., the attractive force.
        The value is clamped between zero and INFINITY.
      \param adhesiveOverlap - allowed overlap (length >= 0) from surface for resting contact.
        At this overlap, no force is applied. At lower overlap, the adhesion force will work,
        at higher overlap, the (usual) contact force.
        The value is clamped between zero and INFINITY.
      */
      void setAdhesion( Real adhesion, Real adhesiveOverlap );

      /**
      \return the adhesion parameter, i.e., the attractive force between two colliding objects (default: 0)
      */
      Real getAdhesion() const;

      /**
      \return the adhesive overlap for this surface material
      */
      Real getAdhesiveOverlap() const;

      /**
      Set the viscosity parameter of this surface. This parameter tells how dry/wet the surface is.
      For larger values, the surface is wetter, and contacting objects will creep more.
      It's like compliance for the friction constraints. Default: 5E-9.
      */
      void setViscosity( Real viscosity );

      /**
      \return the viscosity for this surface (see setViscosity) (default: 5E-9)
      */
      Real getViscosity() const;

      /**
      If this is set to false, the solver will NOT calculate any friction when this material is in contact with another material.
      \return true if the friction should be used when solving contacts for this Material.
      */
      bool getFrictionEnabled() const;

      /**
      Specify if the friction should be used when solving contacts for this Material.

      \param flag - If this is set to false, the solver will NOT calculate any friction when this material is in contact with another material.
      */
      void setFrictionEnabled( bool flag );

      /// \return true if material has changed
      bool isDirty() const;

      DOXYGEN_START_INTERNAL_BLOCK()

      AGXSTREAM_DECLARE_SERIALIZABLE( agx::SurfaceMaterial );


      /**
      Copy all attributes from \p copy_material to this material
      */
      void copy( const SurfaceMaterial* source );

      Physics::SurfaceMaterialPtr& getEntity();

      const Physics::SurfaceMaterialPtr& getEntity() const;

    protected:

      friend class Material;
      void setDirty( bool flag ) const;

      DOXYGEN_END_INTERNAL_BLOCK()

      /// Destructor
      virtual ~SurfaceMaterial();

    private:
      mutable Physics::SurfaceMaterialRef m_entity;
  };


  /**
  Physical properties for the contact between a Geometry and an agx::Wire
  */
  class AGXPHYSICS_EXPORT WireMaterial :  public Referenced, public virtual agxStream::Serializable
  {
    public:

      /// Default constructor
      WireMaterial();

      /**
      \return Young's modulus for stretching (default: 6E10, valid > 0)
      */
      Real getYoungsModulusStretch() const;

      /**
      \return damping in the wire stretch constraints (default: 0.075, valid > 0)
      */
      Real getDampingStretch() const;

      /**
      \return Young's modulus for bend resistance (default equal to stretch)
      */
      Real getYoungsModulusBend() const;

      /**
      \return damping in the wire bend constraints (default: 0.075, valid > 0)
      */
      Real getDampingBend() const;

      /**
      Set the Young's modulus controlling the stretchiness of a wire.
      \param youngsStretch - stretch Young's modulus given in units of force per unit area
      */
      void setYoungsModulusStretch( Real youngsStretch );

      /**
      Set the damping in wire stretch constraints. (default: 0.075, valid > 0)
      \param dampingStretch - damping in wire stretch constraints
      */
      void setDampingStretch( Real dampingStretch );

      /**
      Set the Young's modulus controlling the bend resistance of a wire. 0 => chain like behavior, large => stiff wire or rod
      \param youngsBend - bend Young's modulus given in units of force per unit area
      */
      void setYoungsModulusBend( Real youngsBend );

      /**
      Assign damping in wire bend constraints. (default: 0.075, valid > 0)
      \param dampingBend - damping in wire bend constraints
      */
      void setDampingBend( Real dampingBend );

      DOXYGEN_START_INTERNAL_BLOCK()


      AGXSTREAM_DECLARE_SERIALIZABLE( agx::WireMaterial );

      bool isDirty() const;
      void setDirty(bool flag) const;

      /// Copy all the attributes of \p copy_material to this.
      void copy( const WireMaterial* source );

      Physics::WireMaterialPtr& getEntity();

      const Physics::WireMaterialPtr& getEntity() const;

    protected:

      friend class Material;
      DOXYGEN_END_INTERNAL_BLOCK()

      /// Destructor
      virtual ~WireMaterial();

    private:
      mutable Physics::WireMaterialRef m_entity;
  };



  /** Main material class which acts as a holder of a Surface Material and a Bulk material. */
  class AGXPHYSICS_EXPORT Material :  public Referenced, public virtual agxStream::Serializable
  {
    public:

      /**
      Constructor. If a parent is specified, the new material is initially
      given the same parameters as the parent. If the parent is 0 default
      material parameters will be used.
      */
      Material( const agx::Name& name, const Material* parent = nullptr );

      /**
      Construct a material from friction and restitution values.
      The surface and bulk properties are generated from these values using
      an approximative model, which is not guaranteed to be accurate.
      Note that this is NOT the preferred way of working with the material
      system.
      */
      Material( const agx::Name& name, Real restitution, Real friction );

      /**
      Copy constructor
      */
      Material( const Material& material );

      /**
      \return the surface material properties for this material.
      */
      SurfaceMaterial* getSurfaceMaterial();

      /**
      \return the surface material properties for this material.
      */
      const SurfaceMaterial* getSurfaceMaterial() const;

      /**
      \return the bulk material properties for this material.
      */
      BulkMaterial* getBulkMaterial();

      /**
      \return the bulk material properties for this material.
      */
      const BulkMaterial* getBulkMaterial() const;

      /**
      \return the wire material properties for this material.
      */
      WireMaterial* getWireMaterial();

      /**
      \return the wire material properties for this material.
      */
      const WireMaterial* getWireMaterial() const;

      /**
      \return the name of the material
      */
      const agx::Name& getName() const;

      /**
      Set the name of the material
      */
      void setName( const agx::Name& name ) {
        m_entity.name() = name;
      }

      /// \return a pointer to the default Material (Singleton)
      static Material* getDefaultMaterial();

      DOXYGEN_START_INTERNAL_BLOCK()

      /**
      \return true if any of the attributes in the Material is changed since last setDirty( false );
      */
      bool isDirty() const;


      /**
      Set the properties for this Material to the values defined by a named
      Material in the Material Library.
      \return True if the material definition file could be used to update this Material
      */
      bool loadLibraryMaterial( agx::String name );


      /**
      \return Return all the Material names found via the MaterialLibrary
      */
      static agx::StringVector getAvailableLibraryMaterials();


      /**
      Returns true if the material is loaded from the Material Library
      or if it is based upon such a material.
      */
      bool isLibraryMaterial() const;

      AGXSTREAM_DECLARE_SERIALIZABLE( agx::Material );


      Physics::MaterialPtr& getEntity();

      const Physics::MaterialPtr& getEntity() const;

      /// Copy all attributes from \p copy_material
      void copy(const Material* other);

      DOXYGEN_END_INTERNAL_BLOCK()


    protected:

      /// Default constructor
      Material();

      /// Destructor
      virtual ~Material();

    private:
      friend class agxSDK::MaterialManager;
      void setDirty( bool flag ) const;

      /// Initialize material
      void init(const Name& name);

    private:
      BulkMaterialRef m_bulkMaterial;
      SurfaceMaterialRef m_surfaceMaterial;
      WireMaterialRef m_wireMaterial;

      mutable Physics::MaterialRef m_entity;
  };


  /**
  This class store the combine material properties between two agx::Material's.
  It can both be implicit, i.e. calculated (by the MaterialManager) during impact between two Geometry's materials.
  Or it can be explicit, i.e. created by the user and parameterized through the class interface.
  */
  class AGXPHYSICS_EXPORT ContactMaterial : public Referenced, public virtual agxStream::Serializable
  {
    public:
      /// Specifies the direction of friction
      enum FrictionDirection {
        PRIMARY_DIRECTION,
        SECONDARY_DIRECTION,
        BOTH_PRIMARY_AND_SECONDARY
      };

      /// Specifies the mode for contact reduction
      enum ContactReductionMode {
        REDUCE_NONE,               /**< No contact reduction enabled */
        REDUCE_GEOMETRY,           /**< Default: Reduce contacts between geometries */
        REDUCE_ALL                 /**< Two step reduction: first between geometries, and then between rigid bodies */
      };

    public:

      /**
      Constructor which associates this ContactMaterial to the two given materials \p m1 and \p m2.
      \param m1 - A Material
      \param m2 - A Material
      \param implicit - If true this will be marked as an implicit ContactMaterial.
      */
      ContactMaterial( const Material* m1, const Material* m2, bool implicit=false );

      /**
      Create a ContactMaterial copying attributes from \p other.
      \param other - Attributes will be copied from this ContactMaterial
      \param implicit - If true this will be marked as an implicit ContactMaterial.
      */
      ContactMaterial( const ContactMaterial* other, bool implicit );

      /**
      Set the restitution parameter.
      The value is clamped between zero and INFINITY
      \param restitution - New restitution value.
      */
      void setRestitution( Real restitution );

      /**
      Set the tangential restitution coefficient for a given tangential direction, or both.
      The default tangential restitutions are zero and \p restitution will be clamped
      between zero and infinity.
      \param restitution - New tangential restitution value.
      \param direction - Specifies for which direction the tangential restitution value will be set. Default is BOTH_PRIMARY_AND_SECONDARY.
      */
      void setTangentialRestitution( Real restitution, FrictionDirection direction = BOTH_PRIMARY_AND_SECONDARY );

      /**
      Set the friction coefficient.
      The value is clamped between zero and INFINITY
      \param friction - New friction coefficient
      \param direction - Specifies for which direction the friction value will be set. Default is BOTH_PRIMARY_AND_SECONDARY
      */
      void setFrictionCoefficient( Real friction, FrictionDirection direction = BOTH_PRIMARY_AND_SECONDARY );

      /**
      Assign the surface viscosity, i.e., how wet the friction is between the colliding materials. Default: ~1E-8.
      It's possible to have different viscosities in primary and secondary directions.
      \param viscosity - New viscosity value.
      \param direction - Specifies for which direction the viscosity value will be set. Default is BOTH_PRIMARY_AND_SECONDARY
      */
      void setSurfaceViscosity( Real viscosity, FrictionDirection direction = BOTH_PRIMARY_AND_SECONDARY );

      /**
      Set the adhesive force and allowed overlap between two colliding objects.
      \param adhesion - adhesive force >= 0, i.e., the attractive force.
        The value is clamped between zero and INFINITY.
      \param adhesiveOverlap - allowed overlap (length >= 0) from surface for resting contact.
        At this overlap, no force is applied. At lower overlap, the adhesion force will work,
        at higher overlap, the (usual) contact force.
        The value is clamped between zero and INFINITY.
      */
      void setAdhesion( Real adhesion, Real adhesiveOverlap );

      /**
      \return the adhesion parameter, i.e., the attractive force between two colliding objects.
      */
      Real getAdhesion() const;

      /**
      \return the adhesive overlap for this surface material
      */
      Real getAdhesiveOverlap() const;

      /**
      \return the restitution parameter.
      */
      Real getRestitution() const;

      /**
      \return the tangential resitution parameter for given tangential direction.
      */
      Real getTangentialRestitution( FrictionDirection direction = PRIMARY_DIRECTION ) const;

      /**
      \return the surface friction parameter.
      */
      Real getFrictionCoefficient( FrictionDirection direction = PRIMARY_DIRECTION ) const;

      /**
      \return the surface viscosity in the primary or secondary direction.
      */
      Real getSurfaceViscosity( FrictionDirection direction = PRIMARY_DIRECTION ) const;

      /**
      \return true if surface friction should be used when calculating friction in the solver.
      */
      bool getSurfaceFrictionEnabled() const;

      /**
      Specify if surface friction should be calculated in the solver for this contact material.
      */
      void setEnableSurfaceFriction( bool flag );

      /// Set the YoungsModulus of the material, same as spring coefficient k.
      void setYoungsModulus( Real youngsModulus );

      /// \return the YoungsModulus of the material
      Real getYoungsModulus() const ;

      /**
      Set the damping of the material.
      The value is the time the contact constraint has to fulfill its violation.
      \param damping: The new damping value
      */
      void setDamping( Real damping );

      /**
      Set the minimum and maximum elastic rest length of the contact material.
      This is only used if the contact area approach is used (getUseContactAreaApproach()==true).
      It sets the minimum and maximum allowed value for the elastic rest length of the material - values
      below will be clamped to the min, values above will be clamped to the max.
      If the value has not been set: in contact between materials m1 and m2, the min elastic rest of the contact material cm(m1, m2) will
      have the sum of m1's and m2's min elastic rest length, and the max elastic rest length of cm(m1, m2) will
      be the sum of m1's and m2's max elastic rest length.
      For the effect of the elastic rest length, see the user manual.
      \param minElasticRestLength: The min elastic rest length. Only values > 0 allowed. Has to be <= maxElasticRestLength.
      \param maxElasticRestLength: The max elastic rest length. Only values > 0 allowed. Has to be >= minElasticRestLength.
      \retval Success - Are the values in the allowed range and have thus been applied?
      */
      bool setMinMaxElasticRestLength(Real minElasticRestLength, Real maxElasticRestLength);

      /**
      Get the minimum elastic rest length of the contact material.
      This is only used if the contact area approach is used (getUseContactAreaApproach()==true).
      For the effect of the elastic rest length, see the user manual.
      \retval minElasticRestLength: The min elastic rest length.
      */
      Real getMinElasticRestLength() const;

      /**
      Get the maximum elastic rest length of the contact material.
      This is only used if the contact area approach is used (getUseContactAreaApproach()==true).
      For the effect of the elastic rest length, see the user manual.
      \retval maxElasticRestLength: The max elastic rest length.
      */
      Real getMaxElasticRestLength() const;

      /**
      The value is the time the contact constraint has to fulfill its violation.
      \return the damping of the material
      */
      Real getDamping() const ;

      /**
      \return the friction model used when two objects with this contact material collides
      */
      FrictionModel* getFrictionModel() const;

      /**
      Set friction model for this contact material.
      */
      void setFrictionModel( FrictionModel* frictionModel );

      /**
      \return first material in this contact material
      */
      const Material* getMaterial1() const;

      /**
      \return second material in this contact material
      */
      const Material* getMaterial2() const;

      /**
      \return true if contact material is explicit (created by user)
      */
      bool isExplicit() const;

      /**
      Specify if contact reduction should be enabled.
      \param mode - The mode, default is REDUCE_GEOMETRY
      */
      void setContactReductionMode( ContactReductionMode mode );

      /**
      Returns whether contact reduction is enabled for this surface material
      \return true if contact reduction is enabled.
      */
      ContactReductionMode getContactReductionMode( ) const;

      /**
      Specify the resolution used when evaluating contacts for reduction between geometry contacts.
      A high value will keep more contacts, lower will result in more aggressive reduction the default.
      Commonly a value of 2 or 3 will give good result.
      Values from 0 to 10 are valid, where a value of 0 means this will be overridden by the ContactReductionBinResolution from Space.
      \param binResolution - parameter to ContactReducer.
      */
      void setContactReductionBinResolution( agx::UInt8 binResolution );

      /**
      \return bin resolution used in contact reduction between geometry contacts
      */
      agx::UInt8 getContactReductionBinResolution() const;

      /**
      Should contacts using this contact material use the new approach for
      computing contact area? Default false.
      \note: experimental feature, not fully tested yet.
      */
      void setUseContactAreaApproach( bool useContactAreaApproach );

      /**
      Should contacts using this contact material use the new approach for
      computing contact area? Default false.
      \note: experimental feature, not fully tested yet.
      */
      bool getUseContactAreaApproach() const;

      /**
      * Sets the rolling resistance coefficient of the contact material.
      * Only works with contacts that include particles from GranularBodySystem.
      *
      * \param rollingCoefficient - The coefficient of the rolling resistance.
      */
      void setRollingResistanceCoefficient(agx::Real rollingCoefficient);

      /**
      * Sets the twisting resistance coefficient of the contact material.
      * Only works with contacts that include particles from GranularBodySystem.
      *
      * \param twistingCoefficient - The coefficient of the twisting resistance.
      */
      void setTwistingResistanceCoefficient(agx::Real twistingCoefficient);

     /**
      * \return The coefficient of the rolling resistance in the contact material.
      */
      agx::Real getRollingResistanceCoefficient() const;

      /**
      * \return The coefficient of the twisting resistance in the contact material.
      */
      agx::Real getTwistingResistanceCoefficient() const;

      /**
      * Sets the rolling resistance compliance in the contact material.
      * Only works with contacts that include particles from GranularBodySystem.
      *
      * \param compliance The compliance of the rolling resistance constraint.
      */
      void setRollingResistanceCompliance(agx::Real compliance);

      /**
      * Sets the twisting resistance compliance in the contact material.
      * Only works with contacts that include particles from GranularBodySystem.
      *
      * \param compliance The compliance of the twisting resistance constraint.
      */
      void setTwistingResistanceCompliance(agx::Real compliance);

      /**
      * \return Set the compliance of the rolling resistance constraint.
      */
      agx::Real getRollingResistanceCompliance() const;

      /**
      * \return The compliance of the twisting resistance constraint.
      */
      agx::Real getTwistingResistanceCompliance() const;

      /**
      * Sets the compliance used for calculating impact forces. Only used by granular body system impacts.
      *
      * \param impactCompliance The compliance of the impact constraint.
      */
      void setImpactCompliance(agx::Real impactCompliance);

      /**
      * \return The compliance for calculating impact forces. Only used by granular body system impacts.
      */
      agx::Real getImpactCompliance() const;

      /**
      Set the friction coefficient used by the contact nodes in the wires. The primary direction
      is along the wire and the secondary direction is long the contact edge on the object the
      wire contact is sliding (close to orthogonal to the wire direction).
      The value is clamped between zero and INFINITY
      \param friction - New friction coefficient
      \param direction - Specifies for which direction the friction value will be set. Default is BOTH_PRIMARY_AND_SECONDARY.
      */
      void setWireFrictionCoefficient( Real friction, FrictionDirection direction = BOTH_PRIMARY_AND_SECONDARY );

      /**
      \return the wire friction parameter along the given direction
      */
      Real getWireFrictionCoefficient( FrictionDirection direction = PRIMARY_DIRECTION ) const;

      DOXYGEN_START_INTERNAL_BLOCK()

      /**
      Creates and returns clone of this contact material given material pair.
      The cloned instance will be explicit and, unlike this, have \p material1
      and \p material2 as reference materials.
      \note The friction model instance will be assigned to the clone.
      \param material1 - first material (invalid if nullptr)
      \param material2 - second material (invalid if nullptr)
      \return new contact material instance with identical values but different materials - nullptr for invalid arguments
      */
      agx::ContactMaterial* clone( const agx::Material* material1, const agx::Material* material2 ) const;


      /**
      Copies the ContactMaterial settings into an existing material.
      */
      void copyInto( ContactMaterial& target ) const;

      void transfer( agxData::EntityStorage* storage );


      AGXSTREAM_DECLARE_SERIALIZABLE( agx::ContactMaterial );

      Physics::ContactMaterialPtr& getEntity();

      const Physics::ContactMaterialPtr& getEntity() const;

      /**
      Restitution coefficients are stored in a Vec3 and
      this enum defines their indicies and is only used
      internally in this class or in the solver.
      */
      enum class RestitutionCoefficient
      {
        NORMAL,
        TANGENTIAL_PRIMARY,
        TANGENTIAL_SECONDARY
      };

      DOXYGEN_END_INTERNAL_BLOCK()

    protected:
      ContactMaterial();
      virtual ~ContactMaterial();

    private:

      ContactMaterial( const ContactMaterial& m );

      ContactMaterial& operator = ( const ContactMaterial& );

      void setDefaultParameters();

      friend class agxSDK::MaterialManager;
      friend class agxCollide::GeometryContact;
      friend class agxIO::SceneExporter;
      void setIsExplicit( bool is_explicit );


    private:
      ref_ptr<const Material> m_material1;
      ref_ptr<const Material> m_material2;
      FrictionModelRef m_frictionModel;

      mutable Physics::ContactMaterialRef m_entity;
  };


  /* Implementation */


  // SurfaceMaterial AGX_FORCE_INLINE implementation
  AGX_FORCE_INLINE bool SurfaceMaterial::getFrictionEnabled() const
  {
    return m_entity.frictionEnabled();
  }

  AGX_FORCE_INLINE Real SurfaceMaterial::getRoughness() const
  {
    return m_entity.roughness();
  }

  AGX_FORCE_INLINE Real SurfaceMaterial::getAdhesion() const
  {
    return m_entity.adhesion();
  }

  AGX_FORCE_INLINE Real SurfaceMaterial::getAdhesiveOverlap() const
  {
    return m_entity.adhesiveOverlap();
  }

  AGX_FORCE_INLINE Real SurfaceMaterial::getViscosity() const
  {
    return m_entity.viscosity();
  }

  DOXYGEN_START_INTERNAL_BLOCK()
  AGX_FORCE_INLINE bool SurfaceMaterial::isDirty() const
  {
    return m_entity.dirty();
  }

  AGX_FORCE_INLINE void SurfaceMaterial::setDirty( bool flag ) const
  {
    m_entity.dirty() = flag;
  }
  DOXYGEN_END_INTERNAL_BLOCK()

  DOXYGEN_START_INTERNAL_BLOCK()
  AGX_FORCE_INLINE bool WireMaterial::isDirty() const
  {
    return m_entity.dirty();
  }

  AGX_FORCE_INLINE void WireMaterial::setDirty( bool flag ) const
  {
    m_entity.dirty() = flag;
  }
  DOXYGEN_END_INTERNAL_BLOCK()

  AGX_FORCE_INLINE Real WireMaterial::getYoungsModulusStretch() const
  {
    return m_entity.youngsModulusStretch();
  }

  AGX_FORCE_INLINE Real WireMaterial::getDampingStretch() const
  {
    return m_entity.dampingStretch();
  }

  AGX_FORCE_INLINE Real WireMaterial::getYoungsModulusBend() const
  {
    return m_entity.youngsModulusBend();
  }

  AGX_FORCE_INLINE Real WireMaterial::getDampingBend() const
  {
    return m_entity.dampingBend();
  }

  AGX_FORCE_INLINE void WireMaterial::setYoungsModulusStretch( Real youngsStretch )
  {
    setDirty( true );
    m_entity.youngsModulusStretch() = youngsStretch;
  }

  AGX_FORCE_INLINE void WireMaterial::setDampingStretch( Real dampingStretch )
  {
    setDirty( true );
    m_entity.dampingStretch() = dampingStretch;
  }

  AGX_FORCE_INLINE void WireMaterial::setYoungsModulusBend( Real youngsBend )
  {
    setDirty( true );
    m_entity.youngsModulusBend() = youngsBend;
  }

  AGX_FORCE_INLINE void WireMaterial::setDampingBend( Real dampingBend )
  {
    setDirty( true );
    m_entity.dampingBend() = dampingBend;
  }

  // BulkMaterial AGX_FORCE_INLINE implementation
  AGX_FORCE_INLINE Real BulkMaterial::getViscosity() const
  {
    return m_entity.viscosity();
  }

  AGX_FORCE_INLINE Real BulkMaterial::getDamping() const
  {
    return m_entity.damping();
  }

  AGX_FORCE_INLINE Real BulkMaterial::getYoungsModulus() const
  {
    return m_entity.youngsModulus();
  }

  AGX_FORCE_INLINE Real BulkMaterial::getDensity() const
  {
    return m_entity.density();
  }

  AGX_FORCE_INLINE bool BulkMaterial::isDirty() const
  {
    return m_entity.dirty();
  }

  DOXYGEN_START_INTERNAL_BLOCK()
  AGX_FORCE_INLINE void BulkMaterial::setDirty( bool flag ) const
  {
    m_entity.dirty() = flag;
  }
  DOXYGEN_END_INTERNAL_BLOCK()

  // Material AGX_FORCE_INLINE implementation
  AGX_FORCE_INLINE SurfaceMaterial* Material::getSurfaceMaterial()
  {
    return m_surfaceMaterial;
  }

  AGX_FORCE_INLINE const SurfaceMaterial* Material::getSurfaceMaterial() const
  {
    return m_surfaceMaterial;
  }

  AGX_FORCE_INLINE BulkMaterial* Material::getBulkMaterial()
  {
    return m_bulkMaterial;
  }

  AGX_FORCE_INLINE const BulkMaterial* Material::getBulkMaterial() const
  {
    return m_bulkMaterial;
  }

  AGX_FORCE_INLINE WireMaterial* Material::getWireMaterial()
  {
    return m_wireMaterial;
  }

  AGX_FORCE_INLINE const WireMaterial* Material::getWireMaterial() const
  {
    return m_wireMaterial;
  }

  DOXYGEN_START_INTERNAL_BLOCK()
  AGX_FORCE_INLINE bool Material::isDirty() const
  {
    return m_surfaceMaterial->isDirty() || m_bulkMaterial->isDirty() || m_wireMaterial->isDirty();
  }

  AGX_FORCE_INLINE void Material::setDirty( bool flag ) const
  {
    m_surfaceMaterial->setDirty(flag);
    m_bulkMaterial->setDirty(flag);
    m_wireMaterial->setDirty(flag);
  }
  DOXYGEN_END_INTERNAL_BLOCK()


  AGX_FORCE_INLINE const agx::Name& Material::getName() const
  {
    return m_entity.name();
  }

  // ContactMaterial AGX_FORCE_INLINE implementation
  AGX_FORCE_INLINE Real ContactMaterial::getRestitution() const
  {
    return m_entity.restitution()[ (int)RestitutionCoefficient::NORMAL ];
  }

  AGX_FORCE_INLINE Real ContactMaterial::getFrictionCoefficient(FrictionDirection direction) const
  {
    return m_entity.friction()[ agx::clamp< int >( direction, 0, 1 ) ];
  }

  AGX_FORCE_INLINE Real ContactMaterial::getSurfaceViscosity( FrictionDirection direction /* = PRIMARY_DIRECTION */ ) const
  {
    return m_entity.viscosity()[ agx::clamp< int >( direction, 0, 1 ) ];
  }

  AGX_FORCE_INLINE Real ContactMaterial::getAdhesion() const
  {
    return m_entity.adhesion();
  }

  AGX_FORCE_INLINE Real ContactMaterial::getAdhesiveOverlap() const
  {
    return m_entity.adhesiveOverlap();
  }

  AGX_FORCE_INLINE bool ContactMaterial::getSurfaceFrictionEnabled() const
  {
    return m_entity.surfaceFrictionEnabled();
  }

  AGX_FORCE_INLINE Real ContactMaterial::getYoungsModulus() const
  {
    return m_entity.youngsModulus();
  }

  AGX_FORCE_INLINE Real ContactMaterial::getDamping() const
  {
    return m_entity.damping();
  }

  AGX_FORCE_INLINE FrictionModel* ContactMaterial::getFrictionModel() const
  {
    return m_frictionModel;
  }

  AGX_FORCE_INLINE bool ContactMaterial::isExplicit() const
  {
    return m_entity.isExplicit();
  }

  AGX_FORCE_INLINE const Material* ContactMaterial::getMaterial1() const
  {
    return m_material1;
  }

  AGX_FORCE_INLINE const Material* ContactMaterial::getMaterial2() const
  {
    return m_material2;
  }

  AGX_FORCE_INLINE ContactMaterial::ContactReductionMode ContactMaterial::getContactReductionMode( ) const
  {
    return (ContactReductionMode)m_entity.contactReductionMode();
  }

  AGX_FORCE_INLINE agx::UInt8 ContactMaterial::getContactReductionBinResolution() const
  {
    return m_entity.contactReductionBinResolution();
  }

  DOXYGEN_START_INTERNAL_BLOCK()

  AGX_FORCE_INLINE Physics::BulkMaterialPtr& BulkMaterial::getEntity()
  {
    return m_entity;
  }
  AGX_FORCE_INLINE const Physics::BulkMaterialPtr& BulkMaterial::getEntity() const
  {
    return m_entity;
  }

  AGX_FORCE_INLINE Physics::SurfaceMaterialPtr& SurfaceMaterial::getEntity()
  {
    return m_entity;
  }

  AGX_FORCE_INLINE const Physics::SurfaceMaterialPtr& SurfaceMaterial::getEntity() const
  {
    return m_entity;
  }


  AGX_FORCE_INLINE Physics::WireMaterialPtr& WireMaterial::getEntity()
  {
    return m_entity;
  }

  AGX_FORCE_INLINE const Physics::WireMaterialPtr& WireMaterial::getEntity() const
  {
    return m_entity;
  }


  AGX_FORCE_INLINE Physics::MaterialPtr& Material::getEntity()
  {
    return m_entity;
  }

  AGX_FORCE_INLINE const Physics::MaterialPtr& Material::getEntity() const
  {
    return m_entity;
  }


  AGX_FORCE_INLINE Physics::ContactMaterialPtr& ContactMaterial::getEntity()
  {
    return m_entity;
  }

  AGX_FORCE_INLINE const Physics::ContactMaterialPtr& ContactMaterial::getEntity() const
  {
    return m_entity;
  }
  DOXYGEN_END_INTERNAL_BLOCK()

  AGX_FORCE_INLINE agx::Real ContactMaterial::getRollingResistanceCoefficient() const
  {
    return m_entity.rollingResistanceCoefficient();
  }

  AGX_FORCE_INLINE agx::Real ContactMaterial::getRollingResistanceCompliance() const
  {
    return m_entity.rollingResistanceCompliance();
  }

  AGX_FORCE_INLINE agx::Real ContactMaterial::getTwistingResistanceCompliance() const
  {
    return m_entity.twistingResistanceCompliance();
  }

  AGX_FORCE_INLINE agx::Real agx::ContactMaterial::getImpactCompliance() const
  {
    return m_entity.impactCompliance();
  }

  AGX_FORCE_INLINE bool agx::ContactMaterial::getUseContactAreaApproach() const
  {
    return m_entity.useContactAreaApproach();
  }

  AGX_FORCE_INLINE Real ContactMaterial::getWireFrictionCoefficient(FrictionDirection direction) const
  {
    return m_entity.wireFriction()[ agx::clamp< int >( direction, 0, 1 ) ];
  }
} // namespace agx

#ifdef _MSC_VER
#  pragma warning(pop)
#endif


