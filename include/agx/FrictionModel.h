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

#ifndef AGX_FRICTIONMODEL_H
#define AGX_FRICTIONMODEL_H

#include <agx/Referenced.h>
#include <agx/Range.h>
#include <agx/Vec3.h>
#include <agxStream/Serializable.h>
#include <agx/Physics/GeometryContactEntity.h>

#if defined(_MSC_VER)
#pragma warning( push )
#pragma warning( disable : 4589 ) // Disable: 4589: Constructor of abstract class 'agx::FrictionModel' ignores initializer for virtual base class 'agxStream::Serializable'
#endif


// Forward declarations
namespace agxCollide
{
  class Geometry;
  class GeometryContact;
}

namespace agxSDK
{
  class Simulation;
}

namespace agx
{
  AGX_DECLARE_POINTER_TYPES(FrictionModel);
  /**
  Base class for all friction models with interface and some utility/necessary methods implemented.
  */
  class AGXPHYSICS_EXPORT FrictionModel : public Referenced, public virtual agxStream::Serializable
  {
    public:
      // NOTE: Used to index in arrays.
      //   Think carefully before changing order and/or adding before DIRECT_AND_ITERATIVE.
      enum SolveType
      {
         NOT_DEFINED,
         DIRECT,              /**< Normal and friction equation calculated only in the DIRECT solver */
         ITERATIVE,           /**< Normal and friction equation calculated only in the ITERATIVE solver */
         SPLIT,               /**< First Normal equation is calculated in the DIRECT solver, then in a second pass
                                   the normal and the friction equations are solved in the ITERATIVE solver */
         DIRECT_AND_ITERATIVE /**< Normal and friction equation calculated both in the ITERATIVE and the DIRECT solver */
      };

      enum FrictionCallbackType
      {
        NONE                               = 0,      /**< Not initialized, scale bounds nor iterative projected. */
        INITIALIZE                         = 1 << 0, /**< Initialize bounds callback. */
        SCALE_BOUNDS                       = 1 << 1, /**< Scaled bounds, if direct, callbacks from CP-solver. */
        ITERATIVE_PROJECTION               = 1 << 2, /**< Iterative projected bounds. */
        HAS_CUSTOM_CONTACT_IMPLEMENTATION  = 1 << 3, /**< Custom implementation of the contact constraints associated to this
                                                          friction model with this state. Geometry contacts with a friction model
                                                          containing this state will not reach the solver. */
        ACCURATE_CONE_NL_DIRECT            = 1 << 4, /**< Iterative projected cone friction with direct solve type supports exact
                                                          (anisotropic) projections to the friction cone. This feature isn't enabled
                                                          by default because it's computationally more expensive when the convergence
                                                          rate is lower and the solver may fail more often. */
      };

    public:
      /**
      Solve type for a friction model can be either DIRECT (all equations to the direct solver),
      ITERATIVE (all equations to the iterative solver),
      SPLIT (normal equations both direct and iterative and tangential equations only iterative)
      and DIRECT_AND_ITERATIVE (both normal and tangential
      equations are solved both direct and iterative).
      \param solveType - solve type definition
      */
      void setSolveType( SolveType solveType );

      /**
      \return the current solve type for this friction model
      */
      SolveType getSolveType() const;

      /**
      \return callback mask, used by solvers to determine which callbacks this friction model needs
      */
      int getCallbackMask() const;

      /**
      \param mask - callback mask to check
      \return true if the given callback mask is enabled
      */
      bool getEnableCallbackMask( FrictionCallbackType mask ) const;

      /**
      Calculates friction plane given normal. Only valid if \p normal, \p ret_u and \p ret_v are orthogonal.
      \param geometry1 - first geometry in the geometry contact
      \param geometry2 - the other geometry in the geometry contact
      \param point - contact point in world coordinate system
      \param normal - contact normal in world coordinate system
      \param depth - penetration depth
      \param ret_u - returned, primary direction in friction plane
      \param ret_v - returned, secondary direction in the friction plane
      */
      virtual void calculateTangentPlane( const agxCollide::Geometry* geometry1,
                                          const agxCollide::Geometry* geometry2,
                                          const agx::Vec3& point,
                                          const agx::Vec3& normal,
                                          const agx::Real depth,
                                          agx::Vec3& ret_u,
                                          agx::Vec3& ret_v ) const;

      /**
      Initializes bounds given contact point (pointIndex).
      \param geometryContact - geometry contact
      \param pointIndex - point in geometry contact
      \param dt - time step
      \param ret_boundU - returned, primary direction bounds
      \param ret_boundV - returned, secondary direction bounds
      */
      virtual void initialize( const Physics::GeometryContactPtr geometryContact,
                               size_t pointIndex,
                               agx::Real dt,
                               agx::RangeReal& ret_boundU,
                               agx::RangeReal& ret_boundV ) const = 0;

      /**
      \return true if the friction model needs callbacks for the
        Non-Linear Mixed Complementary Problem solver
        (e.g., ScaleBoxFrictionModel)
      */
      bool needsNlmcpCallbacks() const;

      /**
      \internal

      \return true if this friction model generates custom contact and friction constraints
      */
      bool hasCustomContactImplementation() const;

      /**
      \internal

      Match contact if hasCustomContactImplementation() == true.
      */
      virtual void matchForCustomContactImplementation( agxCollide::GeometryContact* geometryContact,
                                                        agxSDK::Simulation* simulation );

      AGXSTREAM_DECLARE_ABSTRACT_SERIALIZABLE( agx::FrictionModel );

    protected:
      /**
      Protected constructor given callback bask and preferred solve type.
      \param callbackMask - callback mask from FrictionCallbackType
      \param preferredSolveType - if no one assigns solve type, this is the "default" solve type
      */
      FrictionModel( int callbackMask, SolveType preferredSolveType );

      /**
      Enable/disable the given callback mask.
      \param mask - callback mask to change
      \param enable - enable flag
      */
      void setEnableCallbackMask( FrictionCallbackType mask, bool enable );

      /**
      Destructor.
      */
      virtual ~FrictionModel();

      /**
      Restore.
      Abstract serializable, so child classes must explicitly call these methods.
      */
      void restore( agxStream::InputArchive& in ) override;

      /**
      Store.
      Abstract serializable, so child classes must explicitly call these methods.
      */
      void store( agxStream::OutputArchive& out ) const override;

    private:
      SolveType m_solveType;
      int m_callbackMask;
  };



  AGX_DECLARE_POINTER_TYPES(BoxFrictionModel);

  /**
  Box friction. Static bounds during solve. The friction box is by default aligned with the world axes.
  It's possible to implement 'calculateTangentPlane' to align the box with some other direction. This
  friction model supports primary and secondary direction friction coefficients.

  In this implementation the normal force is estimated (impact speed, mass and gravity) or
  if the geometry contact has been solved before, the last normal force is used.
  */
  class AGXPHYSICS_EXPORT BoxFrictionModel : public FrictionModel
  {
    public:
      /**
      Default constructor with default (preferred) solve type SPLIT.
      */
      BoxFrictionModel( agx::FrictionModel::SolveType solveType = agx::FrictionModel::SPLIT );

      /**
      Initializes the bounds, i.e., max force for the tangential equations.
      \param geometryContact - geometry contact
      \param pointIndex - point in geometry contact
      \param dt - time step
      \param ret_boundU - returned, primary direction bounds
      \param ret_boundV - returned, secondary direction bounds
      */
      virtual void initialize( const Physics::GeometryContactPtr geometryContact, size_t pointIndex, agx::Real dt, agx::RangeReal& ret_boundU, agx::RangeReal& ret_boundV ) const override;

      /**
      Calculates/estimates the normal force at the current point.
      */
      virtual agx::Real calculateNormalForce( const agxCollide::Geometry* geometry1, const agxCollide::Geometry* geometry2, const agx::Vec3& point, const agx::Vec3& normal, const agx::Real depth, const agx::Real dt, const size_t numPoints ) const;

      AGXSTREAM_DECLARE_SERIALIZABLE(agx::BoxFrictionModel);

    protected:
      virtual ~BoxFrictionModel() {}
  };


  AGX_DECLARE_POINTER_TYPES(ScaleBoxFrictionModel);

  /**
  Scale box friction. Callbacks from the NLMCP (Non-Linear Mixed Complementarity Problem)
  solver with the current normal force between objects.

  The friction box is by default aligned with the world axes.
  It's possible to implement 'calculateTangentPlane' to align the box is some other direction.
  This friction model supports primary and secondary direction friction coefficients.
  */
  class AGXPHYSICS_EXPORT ScaleBoxFrictionModel : public FrictionModel
  {
    public:
      /**
      Default constructor with default (preferred) solve type SPLIT.
      */
      ScaleBoxFrictionModel( agx::FrictionModel::SolveType solveType = agx::FrictionModel::SPLIT );

      /**
      Initializes the bounds to 0.
      */
      virtual void initialize( const Physics::GeometryContactPtr geometryContact, size_t pointIndex, Real dt, RangeReal& ret_boundU, RangeReal& ret_boundV ) const override;

      AGXSTREAM_DECLARE_SERIALIZABLE(agx::ScaleBoxFrictionModel);

    protected:
      ScaleBoxFrictionModel( int callbackMask, SolveType preferredSolveType ) : FrictionModel( callbackMask, preferredSolveType ) {}
      virtual ~ScaleBoxFrictionModel() {}
  };


  AGX_DECLARE_POINTER_TYPES(IterativeProjectedConeFriction);

  /**
  Iterative Projected Cone Friction (IPC friction) model.
  This is the default friction model, with solve type = SPLIT.
  Given the normal force, this friction model projects
  the friction forces onto the friction cone, which means that the
  alignment of the friction box can be neglected.

  Solve type ITERATIVE and SPLIT works the same with projections
  onto the circle/ellipse given current normal force.

  Solve type DIRECT_AND_ITERATIVE works identical to ScaleBoxFrictionModel
  where the friction bounds are estimated using the iterative solver which
  then are used as fixed bounds in the direct solver.

  Solve type DIRECT is by default estimating and maximizing the friction
  bounds given the current normal force (non-linear) where the size of the
  bounds depends on the reactive forces, such that the resulting friction
  force should stay within the friction circle/ellipse, as close as possible.
  When the friction bounds are expanded, there are certain sliding directions
  that results in higher friction than expected, but this approach has better
  convergence, resulting in better performance in and more stable results
  from the solver.

  To disable the approximation of expanding the friction bounds, flag true
  to IterativeProjectedConeFriction::setEnableDirectExactConeProjection and
  the projections will be precise within the friction cone in any resulting
  direction.
  */
  class AGXPHYSICS_EXPORT IterativeProjectedConeFriction : public ScaleBoxFrictionModel
  {
    public:
      /**
      Default constructor.

      INITIALIZE and SCALE_BOUNDS if solveType != SPLIT.
      For split solves we only get project callbacks during iterations.
      DIRECT_AND_ITERATIVE ScaleBoxFriction::update gets calls
      from the NLMCP and this::project during iterations.
      */
      IterativeProjectedConeFriction( agx::FrictionModel::SolveType solveType = agx::FrictionModel::SPLIT );

      /**
      Enable/disable exact cone projection when solve type is DIRECT. This feature
      is disabled by default when it may affect the convergence of the solver. When
      enabled, the friction forces will be within the friction cone at any given
      resulting direction.
      \param enable - enable/disable exact cone projection when solve type is DIRECT
      */
      void setEnableDirectExactConeProjection( bool enable );

      /**
      \return true when exact direct cone projection is enabled for solve type DIRECT (disabled by default)
      */
      bool getEnableDirectExactConeProjection() const;

      AGXSTREAM_DECLARE_SERIALIZABLE(agx::IterativeProjectedConeFriction);

    protected:
      virtual ~IterativeProjectedConeFriction();
  };

  AGX_FORCE_INLINE FrictionModel::SolveType FrictionModel::getSolveType() const
  {
    return m_solveType;
  }

  AGX_FORCE_INLINE int FrictionModel::getCallbackMask() const
  {
    return m_callbackMask;
  }

  AGX_FORCE_INLINE bool FrictionModel::needsNlmcpCallbacks() const
  {
    return (m_callbackMask & SCALE_BOUNDS) != 0 && (m_solveType == DIRECT || m_solveType == DIRECT_AND_ITERATIVE);
  }

  inline bool FrictionModel::hasCustomContactImplementation() const
  {
    return ( m_callbackMask & HAS_CUSTOM_CONTACT_IMPLEMENTATION ) != 0;
  }

  inline void FrictionModel::matchForCustomContactImplementation( agxCollide::GeometryContact* /*geometryContact*/,
                                                                  agxSDK::Simulation* /*simulation*/ )
  {
  }
}

#if defined(_MSC_VER)
#pragma warning( pop )
#endif

#endif
