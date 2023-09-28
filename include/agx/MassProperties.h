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

#include <agx/agx.h>
#include <agx/Vec3.h>
#include <agx/SPDMatrix3x3.h>
#include <agxStream/Serializable.h>
#include <agx/Physics/RigidBodyEntity.h>


namespace agx
{

  /**
     Class for rigid body inertia tensor and mass.

     The mass properties is a combined object which
     includes the mass and the inertia tensor (in body
     reference frame).

     In some situations the effective mass of the body may differ
     from the normal mass which can be controlled with the setEffective* methods.
  */
  class AGXPHYSICS_EXPORT MassProperties : public agxStream::Serializable
  {
    public:

      /**
      Specifies a bit mask for controlling what should be automatically calculated
      when a body/geometry is changed.
      */
      enum AutoGenerateFlags {
        AUTO_GENERATE_NONE = 0x0,                         /**< Specifies that none of the attributes should be automatically updated */
        MASS = 0x1,                                       /**< Specifies that the mass should be automatically updated */
        CM_OFFSET = 0x2,                                  /**< Specifies that the center of mass should be automatically updated */
        INERTIA = 0x4,                                    /**< Specifies that the inertia should be automatically updated */
        AUTO_GENERATE_ALL = MASS | CM_OFFSET | INERTIA    /**< Specifies that the all of the attributes should be automatically updated */
      };


      /**
      Constructor (for internal use) which associates this object to the storage for a RigidBody.
      */
      MassProperties(agx::Physics::RigidBodyPtr entity);

      /**
      Set the mass.
      This will automatically scale the inertia matrix to reflect the change of mass.

      If autoGenerate is set to true, the mass will be recalculated when the geometry
      configuration of the body changes.
      \param m - The new mass. Has to be > DBL_EPSILON and finite.
      \param autoGenerate - if true, the setAutoGenerateMask will enable the MASS update flag.
      \retval Was the mass correctly set?
      */
      bool setMass( Real m, bool autoGenerate = false );

      /**
      \return the mass.
      */
      inline Real getMass() const;

      /**
      \return the inverse mass.
      */
      inline const Real& getInvMass() const;

      /**
      Set the inertia tensor as a full 3x3 tensor matrix.

      If autoGenerate is set to true, the tensor will
      be recalculated when the geometry configuration of the body changes.
      \param m - The new inertia tensor. Should be a valid finite symmetric positive definite matrix (check its isValid()-method).
      \param autoGenerate - if true, the setAutoGenerateMask will enable the INERTIA update flag.
      \retval Was setting the inertia tensor valid?
      */
      bool setInertiaTensor( const SPDMatrix3x3& m, bool autoGenerate = false);

      /**
      Set the inertia tensor given the diagonal.

      If autoGenerate is set to true, the tensor will
      be recalculated when the geometry configuration of the body changes.
      \param inertiaDiagonal - The new inertia tensor diagonal. Each component has to be > DBL_EPSILON and finite.
      \param autoGenerate - if true, the setAutoGenerateMask will enable the INERTIA update flag.
      \retval Was setting the inertia tensor valid?
      */
      bool setInertiaTensor( const Vec3& inertiaDiagonal, bool autoGenerate = false);

      /**
      \return the inverse mass vector.
      */
      Vec3 getInvMassVector() const;

      /**
      \return the inverse mass matrix.
      */
      SPDMatrix3x3 getInvMassMatrix() const;

      /**
      \return the inertia tensor.
      */
      const SPDMatrix3x3& getInertiaTensor() const;

      /**
      Specify which parameters should be automatically calculated from the geometries of the body given a bitmask of AutoGenerateFlags.
      Default is AUTO_GENERATE_ALL, which mean that all mass properties are updated whenever a geometry associated to a body is changed/removed/added.
      \note Currently, automatic mass property calculation (incremental or not) assumes all geometries and shapes to
            be disjoint (non-overlapping). If they overlap, the effect of this assumption is that the overlapping
            volume will be counted twice into mass and inertia computation. It is recommended to set the mass properties manually in
            this case.
      */
      void setAutoGenerateMask(uint32_t mask);

      /**
      \return the mask with parameters to be auto-generated.
      */
      agx::UInt32 getAutoGenerateMask() const;

      /**
      \return the diagonal of the inertia matrix
      */
      Vec3 getPrincipalInertiae() const;

      /**
      Set mass coefficients. Default is 0,0,0
      Side effect: will also update the effective mass.
      The mass coefficients can be used to change the mass used in the RHS in the solver.
      It will not change the gravitational mass.
      \param coefficients All coefficients have to be >= 0 and finite.
      This allows for different masses along the 3 translational DOF of a body.
      \retval Could the coefficients be set successfully?
      */
      bool setMassCoefficients( const Vec3& coefficients );

      /**
      \return a const reference to the current mass coefficients.
      */
      const Vec3& getMassCoefficients( ) const;

      /**
      \return the effective mass
      */
      const Vec3& getEffectiveMass() const;

      /**
      Set inertia tensor coefficients. Default is 0,0,0
      Side effect: will also update the effective inertia.
      The inertia tensor coefficients can be used to change the inertia used in the RHS in the solver.

      This allows for different inertia for the 3 rotational DOF of a body.
      \param coefficients All coefficients have to be >= 0 and finite.
      \retval Could the coefficients be set successfully?
      */
      bool setInertiaTensorCoefficients( const Vec3& coefficients );

      /**
      \return the current inertia tensor coefficients
      */
      const Vec3& getInertiaTensorCoefficients() const;

      /**
      \return the effective inertia matrix.
      If setInertiaTensorCoefficients==0,0,0, this method will return the same as getInertiaTensor()
      */
      const SPDMatrix3x3& getEffectiveInertiaTensor() const;

      /**
      \return the effective inverted mass as a Vec3
      */
      Vec3 getEffectiveInvMassVector() const;

      /**
      \return the effective inverted mass as a Matrix
      */
      SPDMatrix3x3 getEffectiveInvMassMatrix() const;

      /**
      Set the transform to transform the effective coefficients to the
      body/mass frame. For example if the mass coefficients are given in
      a left handed body attached frame the effective mass transform should
      be: 1  0  0
          0 -1  0
          0  0  1
      \note This transform needs only to be set if there are off diagonal
            elements in the inertia tensor.
      */
      void setEffectiveMassTransformDiagonal( const Vec3& effectiveMassTransformDiagonal );

      /**
      \return the current transformation for transforming the effective coefficients to the body/mass frame.
      */
      const Vec3& getEffectiveMassTransformDiagonal() const;


      /// Assignment operator.
      MassProperties& operator= (const MassProperties& copy);

      /// Destructor
      virtual ~MassProperties();

      DOXYGEN_START_INTERNAL_BLOCK()


      AGXSTREAM_DECLARE_SERIALIZABLE( agx::MassProperties );


    public:
      /**
      Internal method
      */
      static Matrix3x3 calculateOffsetTensor( const Matrix3x3& matrix, const Vec3& v, Real mass );

      /// Internal method
      void setDefaultValues();

      DOXYGEN_END_INTERNAL_BLOCK()

    private:
      MassProperties();
      friend class RigidBody;

      MassProperties(const MassProperties& copy);


    private:

      agx::UInt32 m_autoGenerateMask;
      agx::Physics::RigidBodyPtr m_entity;
  };


  /* Implementation */


  DOXYGEN_START_INTERNAL_BLOCK()
  inline Matrix3x3 MassProperties::calculateOffsetTensor(const Matrix3x3& matrix, const Vec3& v, Real mass)
  {
    Real xx, xy, xz, yz, yy, zz, xtx;

    xx = v[0] * v[0];
    xy = v[0] * v[1];
    xz = v[0] * v[2];

    yy = v[1] * v[1];
    yz = v[1] * v[2];

    zz = v[2] * v[2];

    xtx = v.length2();

    Matrix3x3 tmp(xtx - xx, -xy, -xz,
                  -xy, xtx - yy, -yz,
                  -xz, -yz, xtx - zz);

    tmp *= mass;

    return matrix + tmp;
  }
  DOXYGEN_END_INTERNAL_BLOCK()

  inline Real MassProperties::getMass() const
  {
    agxAssert(m_entity.isValid());
    return m_entity.mass();
  }


  inline const Vec3& MassProperties::getEffectiveMass() const
  {
    agxAssert(m_entity.isValid());
    return m_entity.effectiveMass();
  }

  inline const Vec3& MassProperties::getMassCoefficients( ) const
  {
    agxAssert(m_entity.isValid());
    return m_entity.effectiveMassCoefficients();
  }

  inline const Vec3& MassProperties::getInertiaTensorCoefficients() const
  {
    agxAssert(m_entity.isValid());
    return m_entity.effectiveInertiaCoefficients();
  }

  inline agx::UInt32 MassProperties::getAutoGenerateMask() const
  {
    return m_autoGenerateMask;
  }

  inline const SPDMatrix3x3& MassProperties::getInertiaTensor() const
  {
    agxAssert(m_entity.isValid());
    return m_entity.inertia();
  }

  inline const SPDMatrix3x3& MassProperties::getEffectiveInertiaTensor() const
  {
    agxAssert(m_entity.isValid());
    return m_entity.effectiveInertia();
  }

  inline const Vec3& MassProperties::getEffectiveMassTransformDiagonal() const
  {
    agxAssert(m_entity.isValid());
    return m_entity.effectiveMassTransformDiagonal();
  }


  inline Vec3 MassProperties::getPrincipalInertiae() const
  {
    agxAssert(m_entity.isValid());
    return m_entity.inertia().getDiagonal();
  }

  inline const Real& MassProperties::getInvMass() const
  {
    agxAssert(m_entity.isValid());
    return m_entity.invMass();
  }

  inline Vec3 MassProperties::getInvMassVector() const
  {
    agxAssert(m_entity.isValid());
    return Vec3(m_entity.invMass(), m_entity.invMass(), m_entity.invMass());
  }

  inline SPDMatrix3x3 MassProperties::getInvMassMatrix() const
  {
    agxAssert(m_entity.isValid());
    return SPDMatrix3x3( Vec3(m_entity.invMass(), m_entity.invMass(), m_entity.invMass()) );
  }

} // namespace agx
