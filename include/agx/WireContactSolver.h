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

#ifndef AGX_WIRE_CONTACT_SOLVER_H
#define AGX_WIRE_CONTACT_SOLVER_H

#include <agx/Referenced.h>
#include <agx/Range.h>
#include <agx/agx_vector_types.h>

namespace agx
{

#define NUM_CONTACTS_AT_END 2

  class WireParticle : public Referenced
  {
    public:
    WireParticle( Vec3 position, Vec3 edgeStart, Vec3 edgeEnd,Real mass, Real velocity = Real(0), Real normalForce = 0, Real frictionCoeff = 0)
      : m_position(position),
  m_velocity(velocity),
  m_edgeStart(edgeStart),
  m_edgeEnd(edgeEnd),
  m_invMass(std::max( Real(1)/mass, Real(1E-6)) ),
  m_totalForce(0),
  m_normalForce(normalForce),
  m_externalForce(0),
  m_frictionCoeff(frictionCoeff)
    {
      m_edge = m_edgeEnd-m_edgeStart;
      m_edge.normalize();
    }

    AGX_FORCE_INLINE Vec3 getPosition() const { return m_position; }
    AGX_FORCE_INLINE void setPosition( Vec3 pos ) { m_position = pos; }
    AGX_FORCE_INLINE void setVelocity(Real velocity) { m_velocity = velocity; }
    AGX_FORCE_INLINE Real getVelocity( ) const { return m_velocity; }
    AGX_FORCE_INLINE Real getMass() { return Real(1)/m_invMass; }
    AGX_FORCE_INLINE Real getInvMass() { return m_invMass; }
    AGX_FORCE_INLINE Vec3 getEdge() { return m_edge; }
    void setEdge( Vec3 edge )
    {
      edge.normalize();
      m_edge = edge;
    }
    AGX_FORCE_INLINE Real getNormalForce() const { return m_normalForce; }
    void setNormalForce( Real normalForce ) { m_normalForce = normalForce; }
    void setFrictionCoefficient( Real frictionCoeff ) { m_frictionCoeff = frictionCoeff; }
    AGX_FORCE_INLINE Real getFrictionCoefficient( ) { return m_frictionCoeff; }
    void applyVelocityDamping ();
    AGX_FORCE_INLINE void updatePosition( Real timestep, Real maxVelocity = agx::Infinity )
    {
      Real velocity = agx::clamp( m_velocity, -maxVelocity,maxVelocity);

      m_position = m_position + m_edge*velocity*timestep;
    }

    AGX_FORCE_INLINE void integrateExternalForce( Real timestep )
    {
      m_velocity = m_velocity + m_externalForce*timestep*m_invMass;
    }

    AGX_FORCE_INLINE void setExternalForce( Vec3 externalForce )
    {
      m_externalForce = externalForce*m_edge;
    }

    AGX_FORCE_INLINE Real getExternalForce( ) const { return m_externalForce; }

    AGX_FORCE_INLINE void resetTotalForce( ) { m_totalForce = 0; }

    AGX_FORCE_INLINE void addToTotalForce( Real constraintForce ) { m_totalForce += constraintForce; }

    AGX_FORCE_INLINE Real getTotalForce( ) { return m_totalForce; }

    protected:
      virtual ~WireParticle() {}

    private:
      Vec3 m_position;
      Real m_velocity;
      Vec3 m_edgeStart;
      Vec3 m_edgeEnd;
      Vec3 m_edge;
      Real m_invMass;
      Real m_totalForce;
      Real m_normalForce;
      Real m_externalForce;
      Real m_frictionCoeff;
  };

  typedef ref_ptr<WireParticle> WireParticleRef;

  typedef agx::Vector<WireParticleRef> WireParticleRefVector;

  class ParticleContactWire : public Referenced
  {
    public:
      ParticleContactWire(  Real restlength, Real density, Real youngsModulus, Real compliance, Real radius,Real tension,bool)
      :m_youngsModulus(youngsModulus),m_compliance(compliance),m_restlength(restlength),m_density(density),m_radius(radius),m_tension(tension)
      {}


      AGX_FORCE_INLINE Real getYoungsModulus() const { return m_youngsModulus; }
      AGX_FORCE_INLINE Real getDensity() const { return m_density; }
      AGX_FORCE_INLINE Real getCompliance() const { return m_compliance; }
      AGX_FORCE_INLINE Real getRestlength() const { return m_restlength; }
      AGX_FORCE_INLINE Real getRadius() const { return m_radius; }
      AGX_FORCE_INLINE Real getTension() const { return m_tension; }

    protected:
      virtual ~ParticleContactWire() {}

    private:
      Real m_youngsModulus;
      Real m_compliance;
      Real m_restlength;
      Real m_density;
      Real m_radius;
      Real m_tension;
  };

  typedef ref_ptr<ParticleContactWire> ParticleContactWireRef;


  /**
  A one row constraint
  */
  class WireContactConstraint : public Referenced
  {
    public:
      WireContactConstraint();

      virtual void postCallback(Real /*timestep*/) {}

      virtual void calculateJacobian() = 0;

      virtual void calculateViolation() = 0;

      void calculateInitialLambda( Real timestep )
      {
        m_lambda = timestep*m_violation/m_compliance;
      }

      bool addWireParticle( WireParticle* wp, size_t contactIndex )
      {
        if ( m_particles.empty() )
          m_firstParticleGlobalIndex = contactIndex;

        m_particles.push_back( wp );
        m_jacobianVector.push_back(agx::Real(0));

        return true;
      }
      bool getJacobian( size_t particleIndex, Real& G )
      {
        if ( m_jacobianVector.empty() )
          return false;

        if ( particleIndex < m_firstParticleGlobalIndex || m_jacobianVector.size() <= particleIndex - m_firstParticleGlobalIndex )
          return false;

        G = m_jacobianVector[particleIndex-m_firstParticleGlobalIndex];

        return true;
      }

      Real getInvMass( size_t particleIndex )
      {
        if ( m_particles.empty() )
          return -1;

        if ( particleIndex < m_firstParticleGlobalIndex || m_particles.size() <= particleIndex - m_firstParticleGlobalIndex )
          return -1;

        return m_particles[particleIndex-m_firstParticleGlobalIndex]->getInvMass();
      }

      Real getMass( size_t particleIndex )
      {
        if ( m_particles.empty() )
          return -1;

        if ( particleIndex < m_firstParticleGlobalIndex || m_particles.size() <= particleIndex - m_firstParticleGlobalIndex )
          return -1;

        return m_particles[particleIndex-m_firstParticleGlobalIndex]->getMass();
      }

      size_t getFirstParticleGlobalIndex() const {  return m_firstParticleGlobalIndex; }

      AGX_FORCE_INLINE Real getRHS() const { return m_rhs; }
      AGX_FORCE_INLINE virtual void calculateRHS( Real invTimestep, Real violation )
      {
        Real gamma = 1/(1+4*m_damping*invTimestep);

        Real GV = 0;
        for ( size_t i = 0; i < m_particles.size(); ++i )
        {
          Real G = m_jacobianVector[i];
          GV += G*m_particles[i]->getVelocity();
        }

        m_rhs = -4*gamma*violation*invTimestep + gamma*GV;
      }
      AGX_FORCE_INLINE Real getLambda() const { return m_lambda; }
      void setLambda( Real lambda ) { m_lambda = lambda; }
      AGX_FORCE_INLINE Real getEpsilon() const { return m_epsilon; }
      AGX_FORCE_INLINE Real getD() const { return m_D; }
      AGX_FORCE_INLINE Real getViolation () const {return m_violation;}

      AGX_FORCE_INLINE void setReachedBounds( bool reached ) { m_reachedBounds = reached; }
      AGX_FORCE_INLINE bool getReachedBounds( ) { return m_reachedBounds; }

      AGX_FORCE_INLINE void calculateD( Real timestep)
      {
        size_t indexOffset = getFirstParticleGlobalIndex();

        calculateEpsilon(timestep);

        m_D = getEpsilon();

        for ( size_t i = 0; i < m_jacobianVector.size(); ++i )
        {
          size_t particleIndex = i + indexOffset;
          Real g = 0;
          bool success = getJacobian(particleIndex,g);
          if ( !success )
          {
            m_D = 0;
            return;
          }

          m_D += g*getInvMass(particleIndex)*g;
        }

      }

      AGX_FORCE_INLINE WireParticleRefVector& getParticles() {return m_particles;}
      AGX_FORCE_INLINE void calculateEpsilon( Real timestep )
      {
        m_epsilon = Real(4.0) / ( timestep * timestep * ( Real(1.0) + Real(4.0) * m_damping / timestep ) ) * m_compliance;
      }

      Real getForceOnParticle( size_t globalIndex, Real timestep )
      {
        if ( globalIndex < m_firstParticleGlobalIndex || globalIndex - m_firstParticleGlobalIndex >= m_jacobianVector.size() )
          return 0;

        return m_lambda*m_jacobianVector[ globalIndex - m_firstParticleGlobalIndex ]/timestep;
      }

      AGX_FORCE_INLINE bool getHasBounds() { return m_hasBounds; }
      void setBounds(RangeReal bounds)
      {
        m_lowerBounds = bounds.lower();
        m_upperBounds = bounds.upper();
        m_hasBounds = true;
      }
      AGX_FORCE_INLINE Real getLowerBounds() const { return m_lowerBounds; }
      AGX_FORCE_INLINE Real getUpperBounds() const { return m_upperBounds; }

      void setCompliance( Real compliance ) { m_compliance = compliance; }

    protected:
      virtual ~WireContactConstraint() {}

    protected:
      WireParticleRefVector m_particles;
      RealVector m_jacobianVector;
      RealVector m_normalForces;
      Real m_violation;
      Real m_compliance;
      size_t m_firstParticleGlobalIndex;
      bool m_hasBounds;
      Real m_rhs;
      Real m_damping;

    private:

      /**
      The wire is always a chain of bodies, The m_bodyJacobianPairVector must stay sorted. Then it is trivial to calculate Jacobians for the constraints
      */


      Real m_lambda;

      Real m_epsilon;
      Real m_D; //Epsilon + G M^-1 G^T for iterative constraints
      Real m_upperBounds;
      Real m_lowerBounds;
      bool m_reachedBounds;


  };

  typedef ref_ptr<WireContactConstraint> WireContactConstraintRef;

  typedef agx::Vector<WireContactConstraintRef> WireContactConstraintRefVector;

  class WireContactDistanceConstraint : public WireContactConstraint
  {
    public:
      WireContactDistanceConstraint( Real restlength, Real compliance, Vec3 beginVelocity, Vec3 endVelocity ) :
  m_beginParticle(nullptr),
  m_endParticle(nullptr),
  m_beginParticleVelocity( beginVelocity ),
  m_endParticleVelocity( endVelocity ),
  m_restlength(restlength)
      {
        m_compliance = compliance;
      }

      virtual void postCallback( Real timestep ) override
      {
        m_beginParticle->setPosition( m_beginParticle->getPosition() + m_beginParticleVelocity*timestep );
        m_endParticle->setPosition( m_endParticle->getPosition() + m_endParticleVelocity*timestep );
      }
      virtual void calculateJacobian() override;
      virtual void calculateViolation() override;
      void setBeginParticle( WireParticle* bwp ) { m_beginParticle = bwp; }
      void setEndParticle( WireParticle* ewp ) { m_endParticle = ewp; }
    protected:
      virtual ~WireContactDistanceConstraint() {}
    private:
      WireParticleRef m_beginParticle;
      WireParticleRef m_endParticle;
      Vec3 m_beginParticleVelocity;
      Vec3 m_endParticleVelocity;
      Real m_restlength;

  };

  typedef ref_ptr<WireContactDistanceConstraint> WireContactDistanceConstraintRef;

  class WireContactBendConstraint : public WireContactConstraint
  {
    public:
      WireContactBendConstraint( WireParticle* bwp, WireParticle* mwp, WireParticle* ewp, Real radius, Real youngsModulus, size_t beginIndex ) :
  m_beginParticle(bwp),
  m_middleParticle(mwp),
  m_endParticle(ewp),
  m_angle(0)
      {
        Real l1 = (bwp->getPosition() - mwp->getPosition() ).length();
        Real l2 = (mwp->getPosition() - ewp->getPosition() ).length();
        addWireParticle(m_beginParticle,beginIndex);
        addWireParticle(m_middleParticle,beginIndex+1);
        addWireParticle(m_endParticle,beginIndex+2);
        m_damping = Real(0.5);
        m_firstParticleGlobalIndex = beginIndex;

        Real A = agx::PI * radius * radius;

        // l1 + l2 can be close to zero, clamp above a minimum compliance.
        m_compliance = std::max( Real(400) *( l1 + l2 ) / ( A * radius * radius * youngsModulus ), Real(1E-4));
      }

      virtual void calculateJacobian() override;
      virtual void calculateViolation() override;

    protected:
      virtual ~WireContactBendConstraint() {}

    private:
      WireParticleRef m_beginParticle;
      WireParticleRef m_middleParticle;
      WireParticleRef m_endParticle;
      Real m_angle;
  };

  typedef ref_ptr<WireContactBendConstraint> WireContactBendConstraintRef;

  /**
    Ad-hoc constraint to find normalforces (it is a distanceconstraint in normaldirection of the contact nodes)
  */
  class WireContactSpringConstraint : public WireContactConstraint
  {
    public:
      WireContactSpringConstraint(){}
      WireContactSpringConstraint( WireParticle* particle, Vec3 edge,Real compliance,size_t beginIndex ) :
  m_particle(particle),
  m_restLength(0)
      {
        m_restLength = edge.length();
        m_compliance = compliance;
        m_firstParticleGlobalIndex = beginIndex;
        m_fixPoint = particle->getPosition() - edge;
        addWireParticle(particle,beginIndex);
      }

      virtual void calculateJacobian( ) override
      {
        if ( m_jacobianVector.empty() )
          return;

        m_jacobianVector[0] = 1;
      }
      virtual void calculateViolation( ) override
      {
        m_violation = m_particle->getPosition().distance( m_fixPoint ) - m_restLength;
      }

    protected:
      virtual ~WireContactSpringConstraint() {}

    protected:
      WireParticleRef m_particle;
      Vec3 m_fixPoint;
      Real m_restLength;

  };

  typedef ref_ptr<WireContactSpringConstraint> WireContactSpringConstraintRef;

  class WireContactFrictionConstraint : public WireContactSpringConstraint
  {
    public:
      WireContactFrictionConstraint( WireParticle* particle, Vec3 edge,Real compliance,size_t beginIndex  )
      {
        m_particle = particle;
        m_hasBounds = true;
        m_restLength = edge.length();
        m_compliance = compliance;
        m_firstParticleGlobalIndex = beginIndex;
        m_fixPoint = particle->getPosition() - edge;
        addWireParticle(particle,beginIndex);
      }


      //virtual void calculateJacobian( );
      virtual void calculateViolation( ) override
      {
        m_violation = 0;
      }

      virtual void calculateRHS( Real /*timestep*/, Real /*violation*/ ) override
      {
        m_rhs = 0;
      }
    protected:
      virtual ~WireContactFrictionConstraint() {}

  };

  typedef ref_ptr<WireContactFrictionConstraint> WireContactFrictionConstraintRef;

  /**
  Given a wire, and positions along the wire, we solve for new contact positions
  */
  class AGXPHYSICS_EXPORT WireContactSolver : public Referenced
  {
  public:
    WireContactSolver( Real outsideTimestep, Real beginMass, Real endMass) :
  m_createSpringConstraints(false),
  m_timestep(agx::Real(1E-2)),
  m_outsideTimestep(outsideTimestep),
  m_maxVelocity(agx::Infinity),
  m_beginMass(beginMass),
  m_endMass(endMass)
    {}

    agx::Real getDistanceViolation();

    void createWireParticles( const Vec3Vector& positions, const Vec3Vector& edgeStartPoints, const Vec3Vector& edgeEndPoints,const RealVector& velocities, RealVector& normalForces, bool createSprings, const agx::Real maxExternalForce = 0 );

    void createWire( const Real radius, const Real density, const Real restlength, const Real youngsModulus, const Real compliance,const Real tension, bool useBend);

    void createConstraints( bool distanceBounds = false );

    bool iterateConstraints( bool onlyOneStep = false, bool updatePositionsAndVelocities = true );

    void updateParticleForces( Real timestep  );

    const WireParticleRefVector& getParticles() const { return m_particles; }

    const agx::Vector< WireContactSpringConstraint* > getSpringConstraints() const { return m_springConstraints; }

    Real getTimestep( ) const { return m_timestep; }

    void setMaxVelocity( Real maxVelocity ) { m_maxVelocity = std::abs(maxVelocity); }

    /**
    Maximum value for tension where we use contact solver even if wire is compressed.
    Note!
    The tension is usually positive, but the main solve has made the segment compressed.
    This leads to contact nodes standing still when they should move. This tension cut off is a way of
    choosing the closest path algorithm instead when we know the WireContactSolver will fail.
    */
    Real calculateTensionCutOff( Real dist, Real mass, Real timeStep );

    void clear()
    {
      m_constraints.clear();
      m_wire = nullptr;
      m_particles.clear();
    }

  protected:
    virtual ~WireContactSolver() {}

  private:
    void updateResidual( Real& r, WireParticleRefVector& particleVector, WireContactConstraint* c );

    void updateVelocity( Real dl, WireParticleRefVector& particleVector, WireContactConstraint* c );

    void setTimestep( Real timestep ) { m_timestep = timestep; }

    Real calculateTimestep( Real dist1,Real dist2, Real mass, Real tension );


    WireContactConstraintRefVector m_constraints;
    agx::Vector< WireContactSpringConstraint* > m_springConstraints;
    ParticleContactWireRef m_wire;
    WireParticleRefVector m_particles;
    bool m_createSpringConstraints;
    Real m_timestep;
    Real m_outsideTimestep;
    Real m_maxVelocity;
    Real m_beginMass;
    Real m_endMass;
  };

  typedef ref_ptr<WireContactSolver> WireContactSolverRef;
}
#endif

