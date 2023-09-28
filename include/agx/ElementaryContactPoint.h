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

#include <agx/ConstraintImplementation.h>

namespace agx
{
  AGX_DECLARE_POINTER_TYPES( ElementaryContactPoint );
  AGX_DECLARE_POINTER_TYPES( ElementaryContactPointFactory );

  /**
  Base class of custom implementations of contact point constraints.
  These contact point constraints are treated as ordinary constraints,
  meaning the default contact solving pipelines of the solver are bypassed.

  The forces/torques applied by this contact point implementation are written
  back to the geometry contact before they're exposed to the users (or AMOR).

  NOTE: The default implementations in this base class assumes one normal and
        the rest of the rows to be tangents. Only the normal row 0 can generate
        impact if the speed of m_velocity[0] > 0. The pre-calculated m_normalViolation
        and m_normalForceRange are written in getViolation and getBounds to local row 0,
        and the rest of the rows are 0 (or [-0, 0] for force range) for the rest of the
        rows. Override these methods if the implementation needs/should do something else.
  */
  class AGXPHYSICS_EXPORT ElementaryContactPoint : public ElementaryConstraint
  {
    public:
      /**
      Input data of the geometry contact, which is constant for
      all contact points.
      */
      struct InputData
      {
        Real h; /**< Time step size. */
        Real impactCondition; /**< Minimum speed >= 0 that can generate an impact. */
        const ConstraintImplementation::SolverBodyView bodies; /**< Bodies in the geometry contact. */
        const ContactMaterial& contactMaterial; /**< Contact material in the geometry contact. */

        InputData& operator = ( const InputData& ) = delete;
      };

    public:
      /**
      Material properties of the normal using utility functions used in the solver.
      Note that the utility functions are calculating the diagonal perturbation, but
      the constraint (when solved) will, again, be asked to calculate the diagonal
      perturbation. Reversing the calculated epsilon to a compliance by:
          c_eps = h^2 (1 + 4d / h) * eps / 4
      This will results in 'eps' reaching the solver at the end.
      
      Note that the return value of this method is dependent on the state of the
      "use contact area approach" in the contact material.

      \param contactPoint - contact point with contact area (only used when contactMaterial.getUseContactAreaApproach() == true)
      \param inputData - input data for the contact
      \return the diagonal perturbation as a compliance such that the diagonal perturbation
              will be used in the solver at the end
      */
      static Real calculateNormalEpsilonAsCompliance( const agxCollide::ContactPoint& contactPoint, const InputData& inputData );

    public:
      /**
      Calculates the impact speed > 0 along the normal of the given contact point.
      If the speed < inputData.impactCondition, 0.0 is returned, i.e., the contact
      is considered to be impacting if this method returns a value > 0.0.
      \param contactPoint - contact point with normal
      \param inputData - input data for the contact
      \return impact speed >= 0 along the normal
      */
      Real calculateNormalImpactSpeed( const agxCollide::ContactPoint& contactPoint, const InputData& inputData ) const;

      /**
      Calculates the impact speed > 0 along the a given direction.
      If the speed < inputData.impactCondition, 0.0 is returned, i.e., the contact
      is considered to be impacting if this method returns a value > 0.0.
      \param restitution - restitution >= 0 in the given direction
      \param direction - direction along the impact speed should be calculated
      \param inputData - input data for the contact
      \return impact speed >= 0 along the given direction
      */
      Real calculateImpactSpeed( Real restitution, Vec3f direction, const InputData& inputData ) const;

      /**
      \return the offset vector from the center of mass of body 1 to the contact point
      */
      Vec3 getCmOffset1() const;

      /**
      \return the offset vector from the center of mass of body 2 to the contact point
      */
      Vec3 getCmOffset2() const;

      /**
      \return the position of the contact point in world coordinate frame
      */
      Vec3 getContactPointPosition() const;

    public:
      /**
      Initializes some convenient data and calls 'configure' which is the
      implementation dependent initialization.
      */
      virtual void initialize( agxCollide::ContactPoint& contactPoint, const InputData& inputData ) final;

      /**
      Configure the contact point constraint given contact point and geometry contact
      related data. All inputs has been validated and this method should not fail. I.e.,
      the contact point is valid and enabled so the constraints should be created.
      \param contactPoint - contact point to create the constraint for
      \param inputData - input data for the contact
      */
      virtual void configure( agxCollide::ContactPoint& contactPoint, const InputData& inputData ) = 0;

      /**
      Render debug things using agxRender::RenderSingleton. This method is
      called from a step listener, post.
      */
      virtual void renderDebug() const;

      /**
      Callback when the solver is done. Write back "localForce", "tangentU" and "tangentV"
      to the contact point for all post-data to be in sync with ordinary contacts.

      The "local" force is represented as:
          [0] = x: normal force
          [1] = y: primary tangential/friction force
          [2] = z: secondary tangential/friction force
      For example, if this is an implementation of a normal with two tangents, local force
      should be set as:
          contactPoint.localForce() = Vec3( solution[ 0 ], solution[ 1 ], solution[ 2 ] ) / h;
      Note that the solution has to be divided by the time step.
      \param contactPoint - contact point for this elementary constraint to write data to
      \param solution - solution from the solver, indices [0, getNumRows()) accessible
      \param h - time step size
      */
      virtual void onPostSolve( agxCollide::ContactPoint& contactPoint, const Real* solution, Real h ) const = 0;

      /**
      Writes the contact point violation at row 0 by default.
      */
      virtual UInt getViolation( Real* g, UInt row ) override;

      /**
      Writes m_normalForceRange of the contact point at row 0 by default and
      setting all other rows to [-0, 0].
      */
      virtual UInt getBounds( RangeReal* bounds, UInt row, Real h ) const override;

      /**
      Writes the local m_velocity buffer to the solver.
      */
      virtual UInt getVelocity( Real* v, UInt row ) const override;

      /**
      Impacting if the velocity at row 0 is larger than zero.
      */
      virtual Bool isImpacting() const override;

    protected:
      /**
      Construct given total number of rows per contact point. Serialization is and should
      be disabled because the contact points this constraint is based is not stored and
      will be regenerated when a simulation has been read.
      \param numRows - total number of rows per contact point
      */
      ElementaryContactPoint( UInt numRows );

      virtual ~ElementaryContactPoint();

    protected:
      Vec3 m_contactPointPosition;
      Vec3 m_cmOffset1;
      Vec3 m_cmOffset2;
      Vec3 m_relativeSurfaceVelocity;
      Real m_normalViolation;
      RangeReal m_normalForceRange;
      Bool m_frictionEnabled;
      RealValarray m_velocity;
  };

  inline Vec3 ElementaryContactPoint::getCmOffset1() const
  {
    return m_cmOffset1;
  }

  inline Vec3 ElementaryContactPoint::getCmOffset2() const
  {
    return m_cmOffset2;
  }

  inline Vec3 ElementaryContactPoint::getContactPointPosition() const
  {
    return m_contactPointPosition;
  }

  using ElementaryContactPointRefVector = Vector<ElementaryContactPointRef>;

  /**
  Interface factory for custom contact point constraint implementations.
  The createInstance method is only called when the requested index
  exceeds the cache, i.e., it's assumed the elementary constraints can
  be reused between solves (yet 'configure' is called every step).
  */
  class AGXPHYSICS_EXPORT ElementaryContactPointFactory : public Referenced
  {
    public:
      /**
      Returns an instance at the given index in the cache, e.g., this
      method can be used as (in a constraint implementation):
          factory->begin();
          m_ec.push_back( factory->get( m_ec.size() ) );
          m_ec.push_back( factory->get( m_ec.size() ) );
          factory->end();
      
      The cache is increased with the given index.

      \param localIndex - local index in the cache
      \return elementary contact point instance at the given index in the cache
      */
      ElementaryContactPoint* get( size_t localIndex );

      /**
      Begin using 'get' given local index. The current usage, i.e., start index
      in the cache, is returned.
      \return start index in the cache
      */
      size_t begin();

      /**
      Flag that the constraint won't call 'get' anymore.
      */
      void end();

      /**
      Reset the cache counter.
      */
      void reset();

      /**
      \return the current cache
      */
      const ElementaryContactPointRefVector& getCache() const;

    public:
      /**
      Create a new instance when the requested index exceeds the size of the cache.
      This method can be called multiple times and may not be related to the returned
      elementary contact point to be passed to a constraint and solved.
      \return new instance, invalid if nullptr
      */
      virtual ElementaryContactPoint* createInstance() = 0;

    protected:
      /**
      Increases the cache with the given size.
      \param size - size to increase the cache with
      */
      void increaseCache( size_t size );

    protected:
      ElementaryContactPointFactory();
      virtual ~ElementaryContactPointFactory();

    private:
      ElementaryContactPointRefVector m_ecCache;
      size_t m_currentStartIndex;
      size_t m_maxLocalIndex;
  };
}
