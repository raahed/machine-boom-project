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
#include <agx/RigidBody.h>

#include <agx/Physics/BinaryConstraintEntity.h>
#include <agx/Physics/ManyBodyConstraintEntity.h>
#include <agx/Physics/ContactConstraintEntity.h>
#include <agx/Physics/ConstraintRowEntity.h>
#include <agx/Jacobian.h>
#include <agx/Constraint.h>
#include <agxSDK/StepEventListener.h>

#include <agxSDK/ConstraintEnergyData.h>

// Forward declarations
namespace agxSDK
{
  class Simulation;
  typedef agx::observer_ptr< Simulation > SimulationObserver;
}

namespace agxSDK
{
  AGX_DECLARE_POINTER_TYPES( EnergyManager );

  /**
  Keeps track of the energy flow in the system.
  */
  class AGXPHYSICS_EXPORT EnergyManager : public agxStream::Serializable, public agx::Referenced
  {

  public:
    AGXSTREAM_DECLARE_SERIALIZABLE( agxSDK::EnergyManager );

  public:

    /**
    Enables energy calculations for a rigid body.
    \param rigidBody - The rigid body.
    */
    void add( const agx::RigidBody* rigidBody );

    /**
    Enables energy calculations for a constraint.
    \param constraint - The constraint.
    */
    void add( const agx::Constraint* constraint );

    /**
    Removes the rigid body from the EnergyManager.
    \param rigidBody - The rigid body to be removed.
    */
    void remove( const agx::RigidBody* rigidBody );

    /**
    Removes the constraint from the EnergyManager.
    \param constraint - The constraint to be removed.
    */
    void remove( const agx::Constraint* constraint );

    /**
    Get the change in gravity potential for a rigid body during the last timestep.
    \param rb - The rigid body.
    \return The change in gravity potential for the rigid body.
    */
    agx::Real getPotentialChange( const agx::RigidBody* rb ) const;

    /**
    Get the dissipation from linear and angular velocity damping and external forces. All external forces is
    considered to be dissipative.
    \param rb - The rigid body.
    \return The dissipation.
    */
    agx::Real getDissipation( const agx::RigidBody* rb ) const;

    /**
    Get the change in potential for a constraint during the last timestep.
    \param constraint - The constraint.
    \return The change of potential stored in the constraint.
    */
    agx::Real getPotentialChange( const agx::Constraint* constraint ) const;

    /**
    Get the dissipation for a constraint during the last timestep.
    \param constraint - The constraint.
    \return The energy dissipation from the constraint.
    */
    agx::Real getDissipation( const agx::Constraint* constraint ) const;

    /**
    Get the change in potential for a secondary constraint during the last timestep.
    \param constraint - The secondary constraint.
    \return The change of potential stored in the secondary constraint.
    */
    agx::Real getPotentialChange( const agx::ElementaryConstraint* constraint ) const;

    /**
    Get the dissipation for a secondary constraint during the last timestep.
    \param constraint - The secondary constraint.
    \return The energy dissipation from the secondary constraint.
    */
    agx::Real getDissipation( const agx::ElementaryConstraint* constraint ) const;

    /**
    Get the power from a constraint motor. The motor can be a Motor1D or an ElectricMotorController.
    \param motor - The constraint motor.
    \return The power used by the motor.
    */
    agx::Real getPower( const agx::ElementaryConstraint* motor ) const;

    /**
    Get the change in kinetic energy for a rigid body during the last time step.
    \param rb - The rigid body.
    \return The change in kinetic energy .
    */
    agx::Real getKineticEnergyChange( const agx::RigidBody* rb ) const;

    /**
    Get the change in kinetic energy from the translational motion for a rigid body during the last time step.
    \param rb - The rigid body.
    \return The change in kinetic energy from translational motion.
    */
    agx::Real getTranslationalEnergyChange( const agx::RigidBody* rb ) const;

    /**
    Get the change in kinetic energy from the rotational motion for a rigid body during the last time step.
    \param rb - The rigid body.
    \return The change in kinetic energy from rotational motion.
    */
    agx::Real getRotationalEnergyChange( const agx::RigidBody* rb ) const;

    /**
    Get the kinetic energy for a rigid body.
    \param rb - The rigid body.
    \return The kinetic energy from translational and rotational motion.
    */
    static agx::Real getKineticEnergy( const agx::RigidBody* rb );

    /**
    Get the translational part of the kinetic energy of a rigid body.
    \param rb - The rigid body.
    \return The kinetic energy from translational motion.
    */
    static agx::Real getTranslationalEnergy( const agx::RigidBody* rb );

    /**
    Get the rotational part of the kinetic energy of a rigid body.
    \param rb - The rigid body.
    \return The kinetic energy from rotational motion.
    */
    static agx::Real getRotationalEnergy( const agx::RigidBody* rb );

    DOXYGEN_START_INTERNAL_BLOCK()
    /**
    Calculates work done by external forces. For internal use.
    */
    void calculateWorkFromExternalForces( const agx::Physics::RigidBodyData& rigidBody );
    /**
    Calculates work done on bodies by binary constraints. For internal use.
    */
    void calculateWorkFromBinaryConstraints( const agx::Physics::BinaryConstraintData& binaryConstraint,
                                             const agx::Physics::ConstraintRowData& binaryConstraintRow,
                                             const agxData::Array< agx::Jacobian6DOFElement >& binaryConstraintJacobian,
                                             const agx::Physics::RigidBodyData& rigidBody );
    /**
    Calculates work done on bodies by many body constraints. For internal use.
    */
    void calculateWorkFromManyBodyConstraints( const agx::Physics::ManyBodyConstraintData& manyBodyConstraint,
                                               const agx::Physics::ConstraintRowData& manyBodyConstraintRow,
                                               const agxData::Array< agx::Jacobian6DOFElement >& manyBodyConstraintJacobian,
                                               const agx::Physics::RigidBodyData& rigidBody );

    /**
    Updates the data that should be stored for the next timestep. For internal use.
    */
    void updateData();

    /**
    Resets all values for work done in the system. Called before the energy calculations starts. For internal use.
    */
    void resetEnergy();


  protected:
    // Default constructor
    EnergyManager() = default;

    /**
    Destructor
    */
    virtual ~EnergyManager();

  private:
    class RigidBodyData
    {
    public:
      RigidBodyData( const agx::RigidBody* rb );

      agx::Vec3 velocity() const;
      agx::Vec3 angularVelocity() const;

      agx::Bool update();

    private:
      agx::Vec3 m_velocity;
      agx::Vec3 m_angularVelocity;
      agx::RigidBodyConstObserver m_rb;
    };

    struct EnergyChange
    {
      EnergyChange( agx::Real initValue = 0 )
        : potential( initValue ), dissipation( initValue )
      {
      }

      agx::Real potential;
      agx::Real dissipation;

      void reset()
      {
        potential = 0;
        dissipation = 0;
      };
    };

    using RigidBodyDataContainer = agx::HashVector< const agx::RigidBody*, EnergyManager::RigidBodyData >;
    using EnergyChangeBodyContainer = agx::HashVector< const agx::RigidBody*, EnergyManager::EnergyChange >;
    using EnergyChangeConstraintContainer = agx::HashVector< const agx::ConstraintImplementation*, EnergyManager::EnergyChange >;
    using EnergyChangeScContainer = agx::HashVector< const agx::ElementaryConstraint*, EnergyChange >;
    using RigidBodyConstRefVector = agx::Vector< agx::RigidBodyConstRef >;
    using ConstraintConstRefVector = agx::Vector< agx::ConstraintConstRef >;

    AGX_DECLARE_POINTER_TYPES( UpdateListener );

    class UpdateListener : public agxSDK::StepEventListener
    {
    public:
      UpdateListener( EnergyManager* energyManager );
      virtual void pre( const agx::TimeStamp& t ) override;

    protected:
      virtual ~UpdateListener();

    private:
      EnergyManagerObserver m_energyManager;
    };

  private:
    friend class Simulation;

    /**
    Constructor
    */
    EnergyManager( agxSDK::Simulation* simulation );

    /**
    Set simulation
    */
    void setSimulation( agxSDK::Simulation* simulation );

    /**
    Clear the energy manager.
    */
    void clear();

    template<class T>
    void calculateEnergyForElementaryConstraint( agx::Real& potential, agx::Real& dissipation,
                                                 ConstraintEnergyData* data, agx::UInt numRows, agx::Bool isActive,
                                                 agx::UInt& rowIndex, const T& rbIndices,
                                                 const agx::Physics::ConstraintRowData& binaryConstraintRow,
                                                 agx::UInt& gIndex,
                                                 const agxData::Array< agx::Jacobian6DOFElement >& binaryConstraintJacobian,
                                                 const agx::Physics::RigidBodyData& rigidBody );

    void calculateWorkOnBody( agx::Real& potential, agx::Real& dissipation, const EnergyManager::RigidBodyData& rbData,
                              const agx::Vec3& v, const agx::Vec3& w, agx::Bool isHolonomic, agx::Real lambdaDot,
                              agx::Real tau, const agx::Jacobian6DOFElement& G, agx::Real lambda );

    DOXYGEN_END_INTERNAL_BLOCK()

  private:
    RigidBodyDataContainer          m_rbData;
    EnergyChangeBodyContainer       m_energyChangeBodies;
    EnergyChangeConstraintContainer m_energyChangeConstraints;
    EnergyChangeScContainer         m_energyChangeSc;

    RigidBodyConstRefVector         m_rigidBodies;
    ConstraintConstRefVector        m_constraints;

    agxSDK::SimulationObserver      m_simulation;
    UpdateListenerRef               m_updateListener;
  };
}
