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

#include <agx/FrictionModel.h>
#include <agx/Attachment.h>
#include <agx/BitState.h>

namespace agx
{
  /**
  Oriented friction box base implementation containing a frame of
  reference and the primary direction given in that frame.
  */
  template<typename T>
  class OrientedFrictionModelImpl : public T
  {
    public:
      using Model = T;

    public:
      /**
      Construct given frame of reference and primary friction direction.
      \param refFrame - frame of reference
      \param primaryDirection - primary friction direction given in the reference frame
      \param solveType - friction model solve type
      */
      OrientedFrictionModelImpl( const agx::Frame* refFrame,
                                 const agx::Vec3& primaryDirection,
                                 agx::FrictionModel::SolveType solveType );

      /**
      \return the frame of reference
      */
      const agx::Frame* getReferenceFrame() const;

      /**
      Assign frame of reference.
      \param refFrame - frame of reference
      */
      void setReferenceFrame( const agx::Frame* refFrame );

      /**
      \return primary friction direction given in reference frame
      */
      const agx::Vec3& getPrimaryDirection() const;

      /**
      Assign primary friction direction given in reference frame.
      \param primaryDirection - primary friction direction
      */
      void setPrimaryDirection( const agx::Vec3& primaryDirection );

    public:
      /**
      Calculates primary and secondary friction directions in world
      coordinates given contact point.
      */
      virtual void calculateTangentPlane( const agxCollide::Geometry* geometry1,
                                          const agxCollide::Geometry* geometry2,
                                          const agx::Vec3& point,
                                          const agx::Vec3& normal,
                                          const agx::Real depth,
                                          agx::Vec3& ret_u,
                                          agx::Vec3& ret_v ) const override;

    protected:
      virtual ~OrientedFrictionModelImpl();

    private:
      agx::FrameConstObserver m_refFrame;
      agx::Vec3 m_primaryDirection;
  };

#ifdef SWIG
  %template(OrientedBoxFrictionModelImpl) OrientedFrictionModelImpl<agx::BoxFrictionModel>;
  %template(OrientedScaleBoxFrictionModelImpl) OrientedFrictionModelImpl<agx::ScaleBoxFrictionModel>;
  %template(OrientedIterativeProjectedConeFrictionModelImpl) OrientedFrictionModelImpl<agx::IterativeProjectedConeFriction>;
#endif

  /**
  Box friction model with oriented friction box.
  */
  class AGXPHYSICS_EXPORT OrientedBoxFrictionModel : public OrientedFrictionModelImpl<agx::BoxFrictionModel>
  {
    public:
      using Base = OrientedFrictionModelImpl<agx::BoxFrictionModel>;

    public:
      /**
      Construct given frame of reference and primary friction direction.
      \param refFrame - frame of reference
      \param primaryDirection - primary friction direction given in the reference frame
      \param solveType - friction model solve type
      */
      OrientedBoxFrictionModel( const agx::Frame* refFrame,
                                const agx::Vec3& primaryDirection,
                                agx::FrictionModel::SolveType solveType = agx::FrictionModel::SPLIT );

    public:
      AGXSTREAM_DECLARE_SERIALIZABLE( agx::OrientedBoxFrictionModel );

    protected:
      OrientedBoxFrictionModel();

      virtual ~OrientedBoxFrictionModel();
  };

  /**
  Scale box friction model with oriented friction box.
  */
  class AGXPHYSICS_EXPORT OrientedScaleBoxFrictionModel : public OrientedFrictionModelImpl<agx::ScaleBoxFrictionModel>
  {
    public:
      using Base = OrientedFrictionModelImpl<agx::ScaleBoxFrictionModel>;

    public:
      /**
      Construct given frame of reference and primary friction direction.
      \param refFrame - frame of reference
      \param primaryDirection - primary friction direction given in the reference frame
      \param solveType - friction model solve type
      */
      OrientedScaleBoxFrictionModel( const agx::Frame* refFrame,
                                     const agx::Vec3& primaryDirection,
                                     agx::FrictionModel::SolveType solveType = agx::FrictionModel::SPLIT );

    public:
      AGXSTREAM_DECLARE_SERIALIZABLE( agx::OrientedScaleBoxFrictionModel );

    protected:
      OrientedScaleBoxFrictionModel();

      virtual ~OrientedScaleBoxFrictionModel();
  };

  /**
  Iterative projected cone friction model with oriented friction box.
  */
  class AGXPHYSICS_EXPORT OrientedIterativeProjectedConeFrictionModel : public OrientedFrictionModelImpl<agx::IterativeProjectedConeFriction>
  {
    public:
      using Base = OrientedFrictionModelImpl<agx::IterativeProjectedConeFriction>;

    public:
      /**
      Construct given frame of reference and primary friction direction.
      \param refFrame - frame of reference
      \param primaryDirection - primary friction direction given in the reference frame
      \param solveType - friction model solve type
      */
      OrientedIterativeProjectedConeFrictionModel( const agx::Frame* refFrame,
                                                   const agx::Vec3& primaryDirection,
                                                   agx::FrictionModel::SolveType solveType = agx::FrictionModel::SPLIT );

    public:
      AGXSTREAM_DECLARE_SERIALIZABLE( agx::OrientedIterativeProjectedConeFrictionModel );

    protected:
      OrientedIterativeProjectedConeFrictionModel();

      virtual ~OrientedIterativeProjectedConeFrictionModel();
  };

  /**
  Oriented box friction model that uses the same normal force magnitude for all
  contact points associated to this friction model. This means that the size of
  the friction box always will be:
    primary direction: primary_friction_coefficient * given_normal_force
    secondary direction: secondary_friction_coefficient * given_normal_force
  The given normal force can also be scaled with the contact point depth resulting
  in a maximum friction force:
    depth * primary_friction_coefficient * given_normal_force
    depth * secondary_friction_coefficient * given_normal_force
  */
  class AGXPHYSICS_EXPORT ConstantNormalForceOrientedBoxFrictionModel : public OrientedBoxFrictionModel
  {
    public:
      /**
      Construct given normal force magnitude, reference frame and primary direction.
      \param normalForceMagnitude - normal force for all contact points
      \param refFrame - frame of reference
      \param primaryDirection - primary friction direction
      \param solveType - friction model solve type
      \param scaleWithDepth - true to scale the given normal force with the contact point depth
      */
      ConstantNormalForceOrientedBoxFrictionModel( agx::Real normalForceMagnitude,
                                                   const agx::Frame* refFrame,
                                                   const agx::Vec3& primaryDirection,
                                                   agx::FrictionModel::SolveType solveType = agx::FrictionModel::SPLIT,
                                                   agx::Bool scaleWithDepth = false );

      /**
      Assign normal force magnitude.
      \param normalForceMagnitude - normal force magnitude
      */
      void setNormalForceMagnitude( agx::Real normalForceMagnitude );

      /**
      \return the given normal force magnitude
      */
      agx::Real getNormalForceMagnitude() const;

      /**
      Enable/disable scale of the given normal force with the contact
      point depth resulting in a maximum friction force:
        depth * primary_friction_coefficient * given_normal_force
        depth * secondary_friction_coefficient * given_normal_force
      Default: false.
      \param enable - true to enable, false to disable
      */
      void setEnableScaleWithDepth( agx::Bool enable );

      /**
      \return true if the given normal force is scaled with the contact point depth
      */
      agx::Bool getEnableScaleWithDepth() const;

    public:
      /**
      \return the given normal force magnitude for friction bounds in the solver
      */
      virtual agx::Real calculateNormalForce( const agxCollide::Geometry* geometry1,
                                              const agxCollide::Geometry* geometry2,
                                              const agx::Vec3& point,
                                              const agx::Vec3& normal,
                                              const agx::Real depth,
                                              const agx::Real dt,
                                              const size_t numPoints ) const override;

      AGXSTREAM_DECLARE_SERIALIZABLE( agx::ConstantNormalForceOrientedBoxFrictionModel );

    protected:
      ConstantNormalForceOrientedBoxFrictionModel();

      virtual ~ConstantNormalForceOrientedBoxFrictionModel();

    private:
      enum Flag : agx::UInt16
      {
        SCALE_WITH_DEPTH = 1 << 0
      };
      using Flags = agx::BitState<Flag, agx::UInt16>;

    private:
      agx::Real m_normalForceMagnitude;
      Flags m_flags;
  };

  template<typename T>
  OrientedFrictionModelImpl<T>::OrientedFrictionModelImpl( const agx::Frame* refFrame,
                                                           const agx::Vec3& primaryDirection,
                                                           agx::FrictionModel::SolveType solveType )
    : T( solveType ), m_refFrame( refFrame ), m_primaryDirection( primaryDirection )
  {
  }

  template<typename T>
  OrientedFrictionModelImpl<T>::~OrientedFrictionModelImpl()
  {
  }

  template<typename T>
  const agx::Frame* OrientedFrictionModelImpl<T>::getReferenceFrame() const
  {
    return m_refFrame;
  }

  template<typename T>
  void OrientedFrictionModelImpl<T>::setReferenceFrame( const agx::Frame* refFrame )
  {
    m_refFrame = refFrame;
  }

  template<typename T>
  const agx::Vec3& OrientedFrictionModelImpl<T>::getPrimaryDirection() const
  {
    return m_primaryDirection;
  }

  template<typename T>
  void OrientedFrictionModelImpl<T>::setPrimaryDirection( const agx::Vec3& primaryDirection )
  {
    m_primaryDirection = primaryDirection;
  }

  template<typename T>
  void OrientedFrictionModelImpl<T>::calculateTangentPlane( const agxCollide::Geometry* /*geometry1*/,
                                                            const agxCollide::Geometry* /*geometry2*/,
                                                            const agx::Vec3& /*point*/,
                                                            const agx::Vec3& normal,
                                                            const agx::Real /*depth*/,
                                                            agx::Vec3& ret_u,
                                                            agx::Vec3& ret_v ) const
  {
    // If the frame is null we treat the primary direction as given in world frame.
    ret_u  = m_refFrame != nullptr ?
               m_refFrame->transformVectorToWorld( m_primaryDirection ) :
               m_primaryDirection;
    ret_u -= ( normal * ret_u ) * normal;
    if ( ret_u.normalize() < agx::RealEpsilon ) {
      agx::Attachment::createAttachmentBase( normal, ret_u, ret_v );
      return;
    }

    ret_v = normal.cross( ret_u );
  }
}
