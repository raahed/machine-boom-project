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

#include <agx/ElementaryConstraintData.h>
#include <agx/Jacobian.h>
#include <agx/RegularizationParameters.h>
#include <agx/BitState.h>

#include <agxStream/Serializable.h>

#ifdef _MSC_VER
# pragma warning(push)
# pragma warning(disable: 6011) // Disable warningC6011: dereferencing nullptr pointer
#endif

namespace agxStream
{
  class OutputArchive;
  class InputArchive;
}

namespace agx
{
  class RigidBody;
  class Frame;
  class ConstraintNlmcpCallback;

  DOXYGEN_START_INTERNAL_BLOCK()
  namespace GWriteState
  {
    enum Enum { NORMAL, IGNORE_FIRST, WRITE_SECOND_FIRST };
  }
  DOXYGEN_END_INTERNAL_BLOCK()

  /**
  Elementary constraint base class with interface and global constraint functionality.
  */
  class CALLABLE AGXPHYSICS_EXPORT ElementaryConstraint : public agx::Referenced, public agxStream::Serializable
  {
    public:
      typedef agx::Vector< agx::ref_ptr< agx::ElementaryConstraint > > RefContainer;
      typedef agx::VectorPOD< agx::ElementaryConstraint* > PtrContainer;
      typedef agx::Name NameType;

    public:
      /**
      \param row - row < getNumRows()
      \return regularization parameters (holding e.g., compliance and damping) for row \p row (default: 0)
      */
      agx::RegularizationParameters*       getRegularizationParameters( const agx::UInt row = 0 );

      /**
      \param row - row < getNumRows()
      \return regularization parameters (holding e.g., compliance and damping) for row \p row (default: 0)
      */
      const agx::RegularizationParameters* getRegularizationParameters( const agx::UInt row = 0 ) const;

      /**
      Assign compliance for a given row, or all rows if \p row = -1 (default).
      \param compliance - new compliance
      \param row - row to assign compliance. If i < 0 to assign all rows.
      */
      void setCompliance( agx::Real compliance, int row );
      void setCompliance( agx::Real compliance );

      /**
      Assign elasticity for a given row, or all rows if \p row = -1 (default).
      \param elasticity - new elasticity
      \param row - row to assign elasticity. If i < 0 to assign all rows.
      */
      void setElasticity(agx::Real elasticity, int row);
      void setElasticity(agx::Real elasticity);

      /**
      \param row - row < getNumRows()
      \return compliance for row \p row (row < getNumRows())
      */
      agx::Real getCompliance( agx::UInt row = 0 ) const;

      /**
      \param row - row < getNumRows()
      \return elasticity for row \p row (row < getNumRows())
      */
      agx::Real getElasticity(agx::UInt row = 0) const;

      /**
      Assign damping for a given row, or all rows if \p row = -1 (default).
      \note Calling this method for non-holonomic rows (e.g. Motor1D) has no effect.
      \param damping - new damping
      \param row - row to assign damping. If i < 0 to assign all rows.
      */
      void setDamping( agx::Real damping, int row );
      void setDamping( agx::Real damping );

      /**
      \param row - row < getNumRows()
      \return damping for row \p row (row < getNumRows())
      */
      agx::Real getDamping( agx::UInt row = 0 ) const;

      /**
      \return number of rows in this elementary constraint
      */
      agx::UInt getNumRows() const;

      /**
      Assign force range to a given row.
      \note \p row is default 0. I.e., it is not defined to assign \p forceRange to all rows in one call.
      \param forceRange - new force range
      \param row - row < getNumRows()
      */
      void setForceRange( agx::RangeReal forceRange, const agx::UInt row = 0 );

      /**
      Assign force range to a given row.
      \note \p row is default 0. I.e., it is not defined to assign \p lower and \p upper to all rows in one call.
      \param lower - lower force range (typically <= 0)
      \param upper - upper force range (typically >= 0)
      \param row - row < getNumRows()
      */
      void setForceRange( agx::Real lower, agx::Real upper, const agx::UInt row = 0 );

      /**
      \param row - row < getNumRows()
      \return current force range at row \p row
      */
      agx::RangeReal&       getForceRange( const agx::UInt row = 0 );

      /**
      \param row - row < getNumRows()
      \return current force range at row \p row
      */
      const agx::RangeReal& getForceRange( const agx::UInt row = 0 ) const;

      /**
      Name this elementary constraint.
      \param name - new name of this elementary constraint
      */
      void setName( const agx::ElementaryConstraint::NameType& name );

      /**
      \return the name of this elementary constraint
      */
      const agx::ElementaryConstraint::NameType& getName() const;

      /**
      \return array of current forces - length = getNumRows()
      */
      const agx::Real* getCurrentForces() const;

      /**
      Returns the force applied last time step by this elementary constraint.
      \param row - row < getNumRows()
      \return the force (given row) this elementary constraint applied during the last solve
      */
      virtual agx::Real getCurrentForce( const agx::UInt row = 0 ) const;

      /**
      Enable/disable this elementary constraint.
      \param enable - true to enable, false to disable
      */
      virtual void setEnable( agx::Bool enable );

      /**
      \return true if this elementary constraint is enabled - otherwise false
      */
      virtual agx::Bool getEnable() const;

      /**
      An elementary constraint is defined to be \p active if enabled. Yet constraints
      can be enabled but not active (e.g., RangeController when the range isn't exceeded).
      \return true if this elementary constraint is active - otherwise false
      */
      virtual agx::Bool isActive() const;

      /**
      Prepare call. After this call the elementary constraint may be activated.
      */
      virtual void prepare();

      /**
      Reset call. Call this to zero the solution vector for the ElementaryConstraint.
      */
      void reset();

      /**
      \return true of this elementary constraint is impacting
      */
      virtual agx::Bool isImpacting() const;

      /**
      Returns the Jacobian of this elementary constraint. Note that the Jacobian
      buffer has to be provided by the caller.
      \param G - Jacobian buffer
      \param numBlocks - number of blocks in the global Jacobian (e.g., number of bodies in the main constraint)
      \param row - start row in Jacobian buffer
      \param writeState - state of the main constraint (the order in which the Jacobian is written defines this state)
      \return the number of elements used in \p G
      */
      virtual agx::UInt getJacobian( agx::Jacobian6DOFElement* G, agx::UInt numBlocks, agx::UInt row, agx::GWriteState::Enum writeState );

      /**
      Returns the violation of this elementary constraint. Note that the violation
      buffer has to be provided by the caller.
      \param g - violation buffer
      \param row - start row in violation buffer g
      \return the number of rows used (= elements written) in violation buffer
      */
      virtual agx::UInt getViolation( agx::Real* g, agx::UInt row );

      /**
      Returns the velocity of this elementary constraint (extra, driving term on
      the right hand side). Note that the velocity buffer has to be provided by
      the caller.
      \param v - velocity buffer
      \param row - start row in velocity buffer v
      \return the number of rows used (= elements written) in velocity buffer
      */
      virtual agx::UInt getVelocity( agx::Real* v, agx::UInt row ) const;

      /**
      Returns the bounds of the Lagrange Multipliers (default (-inf, +inf)).
      Note that the bounds buffer has to be provided by the caller.
      \note Dimension of the Lagrange Multipliers are Force * Time (argument \p h = time step length).
      \param bounds - bounds buffer
      \param row - start row in bounds buffer \p bounds
      \param h - current time step length
      \return the number of rows used (= elements written) in the bounds buffer
      */
      virtual agx::UInt getBounds( agx::RangeReal* bounds, agx::UInt row, agx::Real h ) const;

      /**
       Modify epsilon so that this ElementaryConstraint appear to be linear in
       the "normal" deformantion space instead any other space that the
       ElementaryConstraint may be using. For most ElementaryConstraints this
       does nothing. For QuatLock the epsilon is modified to account for the
       nonlinear nature of quaternion's relation to the angle of rotation they
       represent.

       This function is called by the engine during constraint preparation for
       solve.
       */
      virtual void linearizeCompliance(const agx::Real* violations, agx::Real* epsilons, agx::Real minPerturbation) const;

#ifndef SWIG
      /**
      \return solver callback instance if supported by this elementary constraint
      */
      virtual agx::ConstraintNlmcpCallback* getNlCallback() const;
#endif

      /**
      Callback after solve, with the solution.
      \param solution - solution of the last solve
      \param row - start row in solution for this elementary constraint
      \param h - current time step length
      \return number of rows read (= number of rows of this elementary constraint)
      */
      virtual agx::UInt postSolveCallback( const agx::Real* solution, agx::UInt row, agx::Real h );

      /**
      Call for elementary constraints dependent on attachment pairs. Ignored
      by this base class.
      */
      virtual void setAttachmentPair( const agx::AttachmentPair* ap );

      /**
      Store this object to stream.
      */
      virtual void store( agxStream::OutputArchive& out ) const override;

      /**
      Restore this object from stream.
      */
      virtual void restore( agxStream::InputArchive& in ) override;

      AGXSTREAM_DECLARE_ABSTRACT_SERIALIZABLE( agx::ElementaryConstraint );

      /**
      Store structural independent data to stream.
      */
      virtual void storeLightData( agxStream::StorageStream& str ) const override;

      /**
      Restore structural independent data from stream.
      */
      virtual void restoreLightData( agxStream::StorageStream& str ) override;

      /**
       * Copy configuration row from the given ElementaryConstraint into \p this.
       * \param source - The ElementaryConstraint to copy from.
       * \param sourceRow - The index of the row in the given ElementaryConstraint to copy from.
       * \param destRow - The index to the row in \p this to which the configuration should be copied.
       */
      void copyRowFrom(const ElementaryConstraint* source, agx::UInt sourceRow, agx::UInt destRow);

    protected:
      /**
      Construct given data. Sub-classes are responsible of providing regularization
      parameters-, force- and bounds data of size \p numRows.
      \param enable - default enabled or disabled
      \param numRows - number of equations of this elementary constraint
      \param regParamsBuffer - buffer (of size numRows) with regularization parameters
      \param lastForcesBuffer - buffer (of size numRows) to store last/current forces
      \param userBoundsBuffer - buffer (of size numRows) with bounds (read/written through set/getForceRange)
      */
      ElementaryConstraint( agx::Bool enable,
                            agx::UInt numRows,
                            agx::RegularizationParameters* regParamsBuffer,
                            agx::Real* lastForcesBuffer,
                            agx::RangeReal* userBoundsBuffer );

      /**
      Reference counted object, protected destructor.
      */
      virtual ~ElementaryConstraint();

    protected:
      agx::Bool                           m_enable;         /**< enabled */
      agx::UInt                           m_numRows;        /**< number of rows */
      agx::ElementaryConstraint::NameType m_name;           /**< name */
      agx::RegularizationParameters*      m_rps;            /**< regularization parameters */
      agx::Real*                          m_lfs;            /**< last forces */
      agx::RangeReal*                     m_ubs;            /**< user bounds  */
      agx::ref_ptr<agx::Referenced>       m_internalData;   /**< custom, internal data */

    private:
      DOXYGEN_START_INTERNAL_BLOCK()

      friend class InternalData;

      /**
      \return internal data for merge split
      */
      agx::Referenced* getInternalData() const;

      /**
      Assign merge split data for this elementary constraint.
      */
      void setInternalData( agx::Referenced* data );

      DOXYGEN_END_INTERNAL_BLOCK()
  };

  typedef agx::ref_ptr< ElementaryConstraint > ElementaryConstraintRef;

  /**
  Helper class to hold data that is general for elementary constraints.
  N = number of rows.
  DataT = specific data for the elementary constraint. If no data, set to e.g., void*.
  */
  template< agx::UInt N >
  class ElementaryConstraintN : public ElementaryConstraint
  {
    public:
      /**
      Construct given enable flag.
      \param enable - default flag for enable/disable
      */
      ElementaryConstraintN( agx::Bool enable )
        : ElementaryConstraint( enable, N, m_regParams, m_forces, m_bounds ) {}

    protected:
      /**
      Used during serialization restore.
      */
      ElementaryConstraintN()
        : ElementaryConstraint( false, N, m_regParams, m_forces, m_bounds ) {}

    protected:
      agx::RegularizationParameters m_regParams[ N ]; /**< Regularization parameters buffer. */
      agx::Real                     m_forces[ N ];    /**< Forces buffer. */
      agx::RangeReal                m_bounds[ N ];    /**< Bounds buffer. */
  };

#ifdef SWIG
  %template(ElementaryConstraint1) agx::ElementaryConstraintN< 1 >;
  %template(ElementaryConstraint3) agx::ElementaryConstraintN< 3 >;
#endif

  template< agx::UInt N, typename DataT >
  class ElementaryConstraintNData : public ElementaryConstraintN< N >
  {
    public:
      /**
      Construct given data and enable flag.
      \param data - data for the elementary constraint
      \param enable - default flag for enable/disable
      */
      ElementaryConstraintNData( const DataT& data, agx::Bool enable )
        : ElementaryConstraintN< N >( enable ), m_data( data ) {}

      /**
      \return the elementary constraint data
      */
      const DataT& getData() const { return m_data; }

      /**
      \return the elementary constraint data
      */
      DataT& getData() { return m_data; }

      /**
      Data has to support assignment of attachment pair.
      */
      virtual void setAttachmentPair( const agx::AttachmentPair* ap ) override { m_data.set( ap ); }

    protected:
      ElementaryConstraintNData()
        : ElementaryConstraintN< N >(), m_data() {}

    private:
      ElementaryConstraintNData& operator=(const ElementaryConstraintNData&) {return *this;}
    protected:
      DataT m_data; /**< Data. */
  };

#ifdef SWIG
  %template(Dot11Dot1Data) agx::ElementaryConstraintNData< 1, agx::Dot1Data >;
#endif

  /**
  Dot1 elementary constraint: Two orthogonal vectors.
  */
  class AGXPHYSICS_EXPORT Dot1 : public ElementaryConstraintNData< 1, Dot1Data >
  {
    public:
      /**
      Construct given Dot1Data.
      \param data - data for this elementary constraint
      */
      Dot1( const agx::Dot1Data& data );

      AGXSTREAM_DECLARE_SERIALIZABLE(agx::Dot1);

    protected:
      /**
      Used during serialization restore.
      */
      Dot1();

      /**
      Reference counted object, protected destructor.
      */
      virtual ~Dot1();

      virtual agx::UInt getJacobian( agx::Jacobian6DOFElement* G, agx::UInt numBlocks, agx::UInt row, agx::GWriteState::Enum writeState ) override;
      virtual agx::UInt getViolation( agx::Real* g, agx::UInt row ) override;
  };

  typedef agx::ref_ptr< Dot1 > Dot1Ref;

  /**
  Swing elementary constraint: Two orthogonal vectors with hookean behaviour.
  */
  class AGXPHYSICS_EXPORT Swing : public ElementaryConstraintNData<2, SwingData> {
    public:
      /**
      Construct given SwingData.
      \param data - data for this elementary constraint
      */
      Swing(const agx::SwingData& data);


      AGXSTREAM_DECLARE_SERIALIZABLE( agx::Swing );

    protected:
      /**
      Used during serialization restore.
      */
      Swing();

      /**
      Reference counted object, protected destructor.
      */
      virtual ~Swing();

      virtual agx::UInt getJacobian(agx::Jacobian6DOFElement* G, agx::UInt numBlocks, agx::UInt row,
                                    agx::GWriteState::Enum writeState) override;

      virtual agx::UInt getViolation(agx::Real* g, agx::UInt row) override;

      agx::Real m_sinc;
      agx::Real m_xi;
      agx::Real m_epsilon;

  };

  typedef agx::ref_ptr<Swing> SwingRef;


#ifdef SWIG
  %template(Dot21Dot2Data) agx::ElementaryConstraintNData< 1, agx::Dot2Data >;
#endif

  /**
  Dot2 elementary constraint: Separation vector orthogonal to direction vector.
  */
  class AGXPHYSICS_EXPORT Dot2 : public ElementaryConstraintNData< 1, Dot2Data >
  {
    public:
      /**
      Construct given Dot2Data.
      */
      Dot2( const agx::Dot2Data& data );

      AGXSTREAM_DECLARE_SERIALIZABLE(agx::Dot2);

    protected:
      /**
      Used during serialization restore.
      */
      Dot2();

      /**
      Reference counted object, protected destructor.
      */
      virtual ~Dot2();

      virtual agx::UInt getJacobian( agx::Jacobian6DOFElement* G, agx::UInt numBlocks, agx::UInt row, agx::GWriteState::Enum writeState ) override;
      virtual agx::UInt getViolation( agx::Real* g, agx::UInt row ) override;
  };

  typedef agx::ref_ptr< Dot2 > Dot2Ref;

#ifdef SWIG
  %template(ElementaryConstraint3ElementaryConstraintData) agx::ElementaryConstraintNData< 3, agx::ElementaryConstraintData >;
#endif

  /**
  Spherical elementary constraint: Two points in world coincide. Note that this
  elementary constraint has world as reference (unlike SphericalRel which has
  body1 as reference).
  */
  class AGXPHYSICS_EXPORT Spherical : public ElementaryConstraintNData< 3, ElementaryConstraintData >
  {
    public:
      /**
      Construct given SphericalData.
      */
      Spherical( const agx::ElementaryConstraintData& data );

      AGXSTREAM_DECLARE_SERIALIZABLE(agx::Spherical);

    protected:
      /**
      Used during serialization restore.
      */
      Spherical();

      /**
      Reference counted object, protected destructor.
      */
      virtual ~Spherical();

      virtual agx::UInt getJacobian( agx::Jacobian6DOFElement* G, agx::UInt numBlocks, agx::UInt row, agx::GWriteState::Enum writeState ) override;
      virtual agx::UInt getViolation( agx::Real* g, agx::UInt row ) override;
  };

  typedef agx::ref_ptr< Spherical > SphericalRef;

  /**
  Spherical elementary constraint: Two points in world coincide.
  */
  class AGXPHYSICS_EXPORT SphericalRel : public ElementaryConstraintNData< 3, ElementaryConstraintData >
  {
    public:
      /**
      Construct given SphericalRelData.
      */
      SphericalRel( const agx::ElementaryConstraintData& data );

      AGXSTREAM_DECLARE_SERIALIZABLE(agx::SphericalRel);

    protected:
      /**
      Used during serialization restore.
      */
      SphericalRel();

      /**
      Reference counted object, protected destructor.
      */
      virtual ~SphericalRel();

      virtual agx::UInt getJacobian( agx::Jacobian6DOFElement* G, agx::UInt numBlocks, agx::UInt row, agx::GWriteState::Enum writeState ) override;
      virtual agx::UInt getViolation( agx::Real* g, agx::UInt row ) override;
  };

  typedef agx::ref_ptr< SphericalRel > SphericalRelRef;

#ifdef SWIG
  %template(QuatLock3QuatLockData) agx::ElementaryConstraintNData< 3, agx::QuatLockData >;
#endif

  /**
  Locks the three rotational degrees of freedom (relative coordinates).
  */
  class AGXPHYSICS_EXPORT QuatLock : public ElementaryConstraintNData< 3, QuatLockData >
  {
    public:
      /**
      Construct given data.
      */
      QuatLock( const agx::QuatLockData& data );

      AGXSTREAM_DECLARE_SERIALIZABLE(agx::QuatLock);

    protected:
      /**
      Used during serialization restore.
      */
      QuatLock();

      /**
      Reference counted object, protected destructor.
      */
      virtual ~QuatLock();

      /**
      Propagates \p ap to custom data from QuatLockData.
      */
      virtual void setAttachmentPair( const agx::AttachmentPair* ap ) override;

      virtual agx::UInt getJacobian( agx::Jacobian6DOFElement* G, agx::UInt numBlocks, agx::UInt row, agx::GWriteState::Enum writeState ) override;
      virtual agx::UInt getViolation( agx::Real* g, agx::UInt row ) override;

      virtual void linearizeCompliance(const agx::Real* violations, agx::Real* epsilons, agx::Real minPerturbation) const override;
  };

  typedef agx::ref_ptr< QuatLock > QuatLockRef;

#ifdef SWIG
  %template(BasicControllerConstraint1ConstraintAngleBasedData) agx::ElementaryConstraintNData< 1, agx::ConstraintAngleBasedData >;
#endif

  /**
  Base class for constraint angle based elementary constraints (secondary constraints).
  */
  class CALLABLE AGXPHYSICS_EXPORT BasicControllerConstraint : public ElementaryConstraintNData< 1, ConstraintAngleBasedData >
  {
    public:
      /**
      Construct given data containing angle and default enable flag.
      \param data - data containing angle
      \param enable - default enable flag
      */
      BasicControllerConstraint( const agx::ConstraintAngleBasedData& data, agx::Bool enable );

      AGXSTREAM_DECLARE_ABSTRACT_SERIALIZABLE( agx::BasicControllerConstraint );

    protected:
      /**
      Used during serialization restore.
      */
      BasicControllerConstraint();

      /**
      Store this object to stream.
      */
      virtual void store( agxStream::OutputArchive& out ) const override;

      /**
      Restore this object from stream.
      */
      virtual void restore( agxStream::InputArchive& in ) override;

      /**
      Default constraint callback for writing Jacobian. Default this method
      uses angle axis, angle type and ConstraintAngleBasedData::getDir for
      Jacobian. Override this method for custom implementation and call
      utility method writeJacobian.
      */
      virtual agx::UInt getJacobian( agx::Jacobian6DOFElement* G,
                                     agx::UInt numBlocks,
                                     agx::UInt row,
                                     agx::GWriteState::Enum writeState ) override;

      /**
      Internal utility method to write the Jacobian given direction and angle type.
      */
      virtual agx::UInt writeJacobian( agx::Jacobian6DOFElement* G,
                                       agx::UInt numBlocks,
                                       agx::UInt row,
                                       agx::GWriteState::Enum writeState,
                                       const agx::Vec3& dir1,
                                       const agx::Vec3& dir2,
                                       agx::Angle::Type type ) const;
  };

  /**
  Elementary secondary constraint to drive something given
  target speed (translational or rotational).
  */
  class CALLABLE AGXPHYSICS_EXPORT TargetSpeedController : public BasicControllerConstraint
  {
    public:
      /**
      Construct given constraint angle based data.
      \param data - constraint angle based data defining this controller
      */
      TargetSpeedController( const agx::ConstraintAngleBasedData& data );

      /**
      Assign target speed for this controller.
      \param speed - new target speed for this controller
      */
      void setSpeed( agx::Real speed );

      /**
      \return the current target speed for this controller
      */
      agx::Real getSpeed() const;

      /**
      If \p is true the target speed will be interpreted as zero and the
      current angle is the reference angle. I.e., this motor will try
      to hold this angle given the available force range. Disabled
      when \p locked is false.
      */
      void setLocked( agx::Bool locked );

      /**
      \return true if this motor is currently locked at an angle
      */
      agx::Bool getLocked() const;

      /**
      If this state is set and the speed is set to zero, the behavior
      is identical to 'setLocked( true )'. I.e., when this controller
      is set to have zero speed with this state set, it will not drift
      away from that angle if the assigned force range is enough to
      hold that angle.
      */
      void setLockedAtZeroSpeed( agx::Bool lockedAtZeroSpeed );

      /**
      \return true if this motor will lock at current angle when speed is set to zero
      */
      agx::Bool getLockedAtZeroSpeed() const;

      /**
      Utility cast method.
      */
      static agx::TargetSpeedController* safeCast( const agx::ElementaryConstraint* targetSpeedController );

      AGXSTREAM_DECLARE_SERIALIZABLE(agx::TargetSpeedController);

      /**
      Store structural independent data to stream.
      */
      virtual void storeLightData( agxStream::StorageStream& str ) const override;

      /**
      Restore structural independent data from stream.
      */
      virtual void restoreLightData( agxStream::StorageStream& str ) override;

    protected:
      enum InternalState { NONE = 0, LOCKED = (1<<0), LOCKED_AT_ZERO_SPEED = (1<<1) };

    protected:
      /**
      Used during serialization restore.
      */
      TargetSpeedController();

      /**
      Reference counted object, protected destructor.
      */
      virtual ~TargetSpeedController();

      /**
      Updates state dependent variables.
      */
      void adjustToCurrentState();

      /**
      \return true if any state has this controller in locked state
      */
      agx::Bool isLocked() const;

      virtual agx::UInt getViolation( agx::Real* g, agx::UInt row ) override;
      virtual agx::UInt getVelocity( agx::Real* v, agx::UInt row ) const override;

    protected:
      agx::Real  m_speed;
      agx::Int32 m_internalState;
      agx::Real  m_lockAngle;
  };

  typedef agx::ref_ptr< TargetSpeedController > TargetSpeedControllerRef;

  /**
  Translational or rotational friction controller for Hinge, Prismatic and CylindricalJoint.
  */
  class AGXPHYSICS_EXPORT FrictionController : public BasicControllerConstraint
  {
    public:
      /**
      Construct given angle based data.

      This controller is by default disabled.
      */
      FrictionController( const agx::ConstraintAngleBasedData& data );

      /**
      Assign friction coefficient. Default: 0.4167.

      \note If this controller is rotational (Hinge or CylindriclJoint) the radius
            of the axle should be included in the friction coefficient for the
            comparisons with the normal force to be dimensionally correct. I.e.,
              friction_torque <= friction_coefficient * axle_radius * normal_force

      \param frictionCoefficient - friction coefficient
      */
      void setFrictionCoefficient( agx::Real frictionCoefficient );

      /**
      \return the friction coefficient
      */
      agx::Real getFrictionCoefficient() const;

      /**
      Enable/disable non-linear update of the friction conditions given
      current normal force from the direct solver. When enabled - this
      feature is similar to scale box friction models with solve type DIRECT.

      Default: Disabled

      \note This feature only supports constraint solve types DIRECT and
            DIRECT_AND_ITERATIVE - meaning, if the constraint has solve
            type ITERATIVE, this feature is ignored.

      \param enable - true to enable, false to disable (Disabled by default.)
      */
      void setEnableNonLinearDirectSolveUpdate( agx::Bool enable );

      /**
      \return true if non-linear direct solve update is enabled - otherwise false
      */
      agx::Bool getEnableNonLinearDirectSolveUpdate() const;

      /**
      Set the minimum force range that this friction controller can apply. Can be used to simulate static friction.
      Default: (0, 0)
      */
      void setMinimumStaticFrictionForceRange( agx::RangeReal bound );

      /**
      Get the minimum force range that this friction controller can apply. \sa setMinimumStaticFrictionForceRange
      */
      agx::RangeReal getMinimumStaticFrictionForceRange() const;

      /**
      \return true when this controller is enabled and has a friction coefficient larger than zero - otherwise false
      */
      virtual agx::Bool isActive() const override;

      /**
      Utility cast method.
      */
      static agx::FrictionController* safeCast( const agx::ElementaryConstraint* frictionController );

      AGXSTREAM_DECLARE_SERIALIZABLE( agx::FrictionController );

      /**
      /internal

      Store structural independent data to stream.
      */
      virtual void storeLightData( agxStream::StorageStream& str ) const override;

      /**
      /internal

      Restore structural independent data from stream.
      */
      virtual void restoreLightData( agxStream::StorageStream& str ) override;

#ifndef SWIG
    public:
      ConstraintNlmcpCallback* getNlCallback() const override;
      agx::UInt iterativePostSolveCallback( const agx::ConstraintImplementation* constraint,
                                            const agx::Real* solution,
                                            agx::UInt row,
                                            agx::Real h );
#endif

    protected:
      FrictionController();
      virtual ~FrictionController();

      virtual agx::UInt getViolation( agx::Real* g, agx::UInt row ) override;
      virtual agx::UInt getBounds( agx::RangeReal* bounds, agx::UInt row, agx::Real h ) const override;
      virtual agx::UInt postSolveCallback( const agx::Real* solution, agx::UInt row, agx::Real h ) override;

    private:
      enum Flag : agx::UInt16
      {
        NON_LINEAR = 1 << 0
      };
      using Flags = agx::BitState<Flag, agx::UInt16>;

    private:
      class FrictionControllerNlCallback* getCallback() const;

    private:
      Real m_frictionCoefficient;
      Real m_lastNormalForce;
      RangeReal m_minimumStaticFrictionBound;
      ref_ptr<Referenced> m_nlmcpCallback;
      Flags m_flags;
  };

  /**
  Implementation of a electric motor controller.
  */
  class AGXPHYSICS_EXPORT ElectricMotorController : public BasicControllerConstraint
  {
    public:
      /**
      Construct given basic controller data.
      */
      ElectricMotorController( const agx::ConstraintAngleBasedData& data );

      /**
      Available voltage or voltage drop across the terminals of this motor.
      Default value: 24V.
      \param voltage - voltage (default: 24V)
      */
      void setVoltage( agx::Real voltage );

      /**
      \return the currently used voltage
      */
      agx::Real getVoltage() const;

      /**
      Assign armature resistance (Ohm). Resistance in the armature circuit.
      Default value: 1 Ohm.
      \param armatureResistance - armature resistance (default: 1 Ohm)
      */
      void setArmatureResistance( agx::Real armatureResistance );

      /**
      \return the currently used armature resistance
      */
      agx::Real getArmatureResistance() const;

      /**
      Assign torque constant of this motor in unit torque per Ampere. This value
      couples the torque out to current in. Default value: 1.
      \param torqueConstant - torque constant (default: 1)
      */
      void setTorqueConstant( agx::Real torqueConstant );

      /**
      \return the currently used torque constant
      */
      agx::Real getTorqueConstant() const;

      /**
      Utility cast method.
      */
      static agx::ElectricMotorController* safeCast( const agx::ElementaryConstraint* electricMotorController );

      AGXSTREAM_DECLARE_SERIALIZABLE(agx::ElectricMotorController);

      /**
      Store structural independent data to stream.
      */
      virtual void storeLightData( agxStream::StorageStream& str ) const override;

      /**
      Restore structural independent data from stream.
      */
      virtual void restoreLightData( agxStream::StorageStream& str ) override;

    protected:
      ElectricMotorController();
      virtual ~ElectricMotorController();

      /**
      Not defined to explicitly set compliance. Use armature resistance and torque constant
      instead. Compliance = armature resistance / ( torque constant )^2.
      */
      using ElementaryConstraint::setCompliance;

      /**
      Pure non-holonomic constraint. Damping parameter isn't used.
      */
      using ElementaryConstraint::setDamping;

      virtual void prepare() override;
      virtual agx::UInt getViolation( agx::Real* v, agx::UInt row ) override;
      virtual agx::UInt getVelocity( agx::Real* v, agx::UInt row ) const override;

    protected:
      agx::Real m_voltage;
      agx::Real m_armatureResistance;
      agx::Real m_torqueConstant;
  };

  typedef agx::ref_ptr< ElectricMotorController > ElectricMotorControllerRef;

  /**
  Elementary secondary constraint to keep constraint angle within
  two given values.
  */
  class CALLABLE AGXPHYSICS_EXPORT RangeController : public BasicControllerConstraint
  {
    public:
      /**
      Construct given constraint angle based data.
      \param data - constraint angle based data defining this controller
      */
      RangeController( const agx::ConstraintAngleBasedData& data );

      /**
      Assign range for the constraint angle to between.
      \param range - range for the angle to be between
      */
      void setRange( agx::RangeReal range );

      /**
      Assign range for the constraint angle to between.
      \param lower - lower range
      \param upper - upper range
      */
      void setRange( agx::Real lower, agx::Real upper );

      /**
      \return the current range (reference, may be changed)
      */
      agx::RangeReal& getRange();

      /**
      \return the current range
      */
      const agx::RangeReal& getRange() const;

      /**
      \return true if enabled and active at a bound, otherwise false
      */
      virtual agx::Bool isActive() const override;

      /**
      Utility cast method.
      */
      static agx::RangeController* safeCast( const agx::ElementaryConstraint* rangeController );

      AGXSTREAM_DECLARE_SERIALIZABLE(agx::RangeController);

      /**
      Store structural independent data to stream.
      */
      virtual void storeLightData( agxStream::StorageStream& str ) const override;

      /**
      Restore structural independent data from stream.
      */
      virtual void restoreLightData( agxStream::StorageStream& str ) override;

    protected:
      /**
      Used during serialization restore.
      */
      RangeController();

      /**
      Reference counted object, protected destructor.
      */
      virtual ~RangeController();

      virtual void prepare() override;
      virtual agx::UInt getViolation( agx::Real* g, agx::UInt row ) override;
      virtual agx::UInt getBounds( agx::RangeReal* bounds, agx::UInt row, agx::Real h ) const override;

    protected:
      agx::RangeReal m_range;
      agx::Real      m_violation;
  };

  typedef agx::ref_ptr< RangeController > RangeControllerRef;

  /**
  Elementary secondary constraint to keep constraint angle
  at a given target position.
  */
  class CALLABLE AGXPHYSICS_EXPORT LockController : public BasicControllerConstraint
  {
    public:
      /**
      Construct given constraint angle based data.
      \param data - constraint angle based data defining this controller
      */
      LockController( const agx::ConstraintAngleBasedData& data );

      /**
      Assign new target position.
      \param position - target constraint angle position
      */
      void setPosition( agx::Real position );

      /**
      \return the current target constraint angle position
      */
      agx::Real getPosition() const;

      /**
      Utility cast method.
      */
      static agx::LockController* safeCast( const agx::ElementaryConstraint* lockController );

      AGXSTREAM_DECLARE_SERIALIZABLE(agx::LockController);

      /**
      Store structural independent data to stream.
      */
      virtual void storeLightData( agxStream::StorageStream& str ) const override;

      /**
      Restore structural independent data from stream.
      */
      virtual void restoreLightData( agxStream::StorageStream& str ) override;

    protected:
      /**
      Used during serialization restore.
      */
      LockController();

      /**
      Reference counted object, protected destructor.
      */
      virtual ~LockController();

      virtual agx::UInt getViolation( agx::Real* g, agx::UInt row ) override;

    protected:
      agx::Real m_position;
  };

  typedef agx::ref_ptr< LockController > LockControllerRef;

  class CALLABLE AGXPHYSICS_EXPORT ScrewController : public agx::BasicControllerConstraint
  {
    public:
      /**
      Construct given basic controller data and default lead of the screw.
      Default is a lead of 0, that is no coupling between the two DOF
      \param data - constraint angle based data defining this controller
      */
      ScrewController( const agx::ConstraintAngleBasedData& data );

      /**
      Assign lead, i.e., the distance along the screw's axis that is covered by one complete rotation.
      \param lead - is the distance along the screw's axis that is covered by one complete rotation of the screw (360 degrees)
      */
      void setLead( agx::Real lead );

      /**
      \return the lead of this screw controller
      */
      agx::Real getLead() const;

      /**
      Utility cast method.
      */
      static agx::ScrewController* safeCast( const agx::ElementaryConstraint* screwController );

      AGXSTREAM_DECLARE_SERIALIZABLE(agx::ScrewController);

      /**
      Store structural independent data to stream.
      */
      virtual void storeLightData( agxStream::StorageStream& str ) const override;

      /**
      Restore structural independent data from stream.
      */
      virtual void restoreLightData( agxStream::StorageStream& str ) override;

    protected:
      /**
      Default constructor used during restore.
      */
      ScrewController();

      /**
      Reference counted object, protected destructor.
      */
      virtual ~ScrewController();

      virtual agx::UInt getJacobian( agx::Jacobian6DOFElement* G, agx::UInt numBlocks, agx::UInt row, agx::GWriteState::Enum writeState ) override;
      virtual agx::UInt getViolation( agx::Real* g, agx::UInt row ) override;

    protected:
      agx::Real m_lead;
  };


#ifdef SWIG
  %template(TwistTwistData) agx::ElementaryConstraintNData< 1, agx::TwistData >;
#endif

  /**
  Twist elementary constraint.
  */
  class AGXPHYSICS_EXPORT Twist : public ElementaryConstraintNData<1, TwistData> {
    public:
      /**
      Construct given TwistData
      \param data - data for this elementary constraint
      */
      Twist(const agx::TwistData& data);

      AGXSTREAM_DECLARE_SERIALIZABLE( agx::Twist );

    protected:
      virtual void setAttachmentPair(const agx::AttachmentPair* ap) override;

      /**
      Used during serialization restore.
      */
      Twist();

      /**
      Reference counted object, protected destructor.
      */
      virtual ~Twist();

      virtual agx::UInt getJacobian(agx::Jacobian6DOFElement* G, agx::UInt numBlocks, agx::UInt row,
                                    agx::GWriteState::Enum writeState) override;

      virtual agx::UInt getViolation(agx::Real* g, agx::UInt row) override;
  };

  typedef agx::ref_ptr<Twist> TwistRef;

  AGX_FORCE_INLINE RegularizationParameters* ElementaryConstraint::getRegularizationParameters( const UInt row /* = 0 */ )
  {
    agxAssert( m_rps != nullptr && row < m_numRows );
    return &m_rps[ row ];
  }

  AGX_FORCE_INLINE const RegularizationParameters* ElementaryConstraint::getRegularizationParameters( const UInt row /* = 0 */ ) const
  {
    agxAssert( m_rps != nullptr && row < m_numRows );
    return &m_rps[ row ];
  }

  AGX_FORCE_INLINE UInt ElementaryConstraint::getNumRows() const
  {
    return m_numRows;
  }

  AGX_FORCE_INLINE const ElementaryConstraint::NameType& ElementaryConstraint::getName() const
  {
    return m_name;
  }

  inline void ElementaryConstraint::linearizeCompliance(
      const agx::Real* /*violations*/, agx::Real* /*epsilons*/, agx::Real /*minPerturbation*/) const
  {
  }

  DOXYGEN_START_INTERNAL_BLOCK()
  inline Referenced* ElementaryConstraint::getInternalData() const
  {
    return m_internalData;
  }
  DOXYGEN_END_INTERNAL_BLOCK()
}

#ifdef _MSC_VER
# pragma warning(pop)
#endif
