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

#include <agxVehicle/export.h>

#include <agx/Hinge.h>
#include <agx/BitState.h>
#include <agx/agx_vector_types.h>

namespace agxVehicle
{
  AGX_DECLARE_POINTER_TYPES( TrackProperties );

  /**
  Object containing properties of an agxVehicle::Track.
  */
  class AGXVEHICLE_EXPORT TrackProperties : public agx::Referenced, public agxStream::Serializable
  {
    public:
      /**
      Default constructor - all flags and values are set do default.
      */
      TrackProperties();

      /**
      Assign compliance (for all hinge degrees of freedom) used in the hinges between track nodes.
      \param compliance - new compliance (default: 1.0E-10)
      */
      void setHingeCompliance( agx::Real compliance );

      /**
      Assign compliance for a given degree of freedom in the hinges between track nodes.
      \param compliance - new compliance (default: 1.0E-10)
      \param dof - hinge degree of freedom
      */
      void setHingeCompliance( agx::Real compliance, agx::Hinge::DOF dof );

      /**
      Assign compliance for the translational degrees of freedom in the hinges between track nodes.
      \param compliance - new compliance (default: 1.0E-10)
      */
      void setHingeComplianceTranslational( agx::Real compliance );

      /**
      Assign compliance for the rotational degrees of freedom in the hinges between track nodes.
      \param compliance - new compliance (default: 1.0E-10)
      */
      void setHingeComplianceRotational( agx::Real compliance );

      /**
      \return the compliance used in the hinges between track nodes
      */
      agx::Real getHingeCompliance( agx::Hinge::DOF dof ) const;

      /**
      Assign damping (for all hinge degrees of freedom) used in the hinges between track nodes.
      \param damping - new damping (default: 0.0333)
      */
      void setHingeDamping( agx::Real damping );

      /**
      Assign damping for a given degree of freedom in the hinges between track nodes.
      \param damping - new damping (default: 0.0333)
      \param dof - hinge degree of freedom
      */
      void setHingeDamping(agx::Real damping, agx::Hinge::DOF dof);

      /**
      Assign damping for the translational degrees of freedom in the hinges between track nodes.
      \param damping - new damping (default: 0.0333)
      */
      void setHingeDampingTranslational(agx::Real damping);

      /**
      Assign damping for the rotational degrees of freedom in the hinges between track nodes.
      \param damping - new damping (default: 0.0333)
      */
      void setHingeDampingRotational(agx::Real damping);

      /**
      \return the damping used in the hinges between track nodes
      */
      agx::Real getHingeDamping( agx::Hinge::DOF dof ) const;

      /**
      True to enable the range in the hinges between the track nodes to define how
      the track may bend.
      \param enable - true to enable, false to disable (default: true)
      */
      void setEnableHingeRange( agx::Bool enable );

      /**
      \return true if the hinge ranges are enabled between the track nodes
      */
      agx::Bool getEnableHingeRange() const;

      /**
      Assign range used if the hinge range between the nodes are enabled.
      \param lowerAngle - lower angle in radians, begin of range (default: -2 pi / 3)
      \param upperAngle - upper angle in radians, end of range (default: degreesToRadians( 20 ))
      */
      void setHingeRangeRange( agx::Real lowerAngle, agx::Real upperAngle );

      /**
      \return the hinge range
      */
      agx::RangeReal getHingeRangeRange() const;

      /**
      When the track has been initialized some nodes are in contact with the wheels.
      If this flag is true the interacting nodes will be merged to the wheel directly
      after initialize, if false the nodes will be merged during the first (or later)
      time step.
      \param enable - true to enable, false to disable (default: false)
      */
      void setEnableOnInitializeMergeNodesToWheels( agx::Bool enable );

      /**
      \return true if initial merge is enabled, false if disabled
      */
      agx::Bool getEnableOnInitializeMergeNodesToWheels() const;

      /**
      True to position/transform the track nodes to the surface of the wheels after
      the track has been initialized. When false, the routing algorithm positions
      are used.
      \param enable - true to enable, false to disable (default: true)
      */
      void setEnableOnInitializeTransformNodesToWheels( agx::Bool enable );

      /**
      \return true if initial transform of nodes to wheels is enabled, false if disabled
      */
      agx::Bool getEnableOnInitializeTransformNodesToWheels() const;

      /**
      When the nodes are transformed to the wheels, this is the final target overlap.
      \param overlap - target overlap (default: 1.0E-3)
      */
      void setTransformNodesToWheelsOverlap( agx::Real overlap );

      /**
      \return the target overlap when transforming nodes to wheels
      */
      agx::Real getTransformNodesToWheelsOverlap() const;

      /**
      Threshold when to merge a node to a wheel. Given a reference direction in the
      track, this value is the projection of the deviation (from the reference direction)
      of the node direction onto the wheel radial direction vector. I.e., when the
      projection is negative the node can be considered "wrapped" on the wheel.
      Default value: -0.1.
      \param mergeThreshold - merge threshold (default: -0.1)
      */
      void setNodesToWheelsMergeThreshold( agx::Real mergeThreshold );

      /**
      \return node to wheel merge threshold
      */
      agx::Real getNodesToWheelsMergeThreshold() const;

      /**
      Threshold when to split a node from a wheel. Given a reference direction in the
      track, this value is the projection of the deviation (from the reference direction)
      of the node direction onto the wheel radial direction vector. I.e., when the
      projection is negative the node can be considered "wrapped" on the wheel.
      Default value: -0.05.
      \param splitThreshold - split threshold (default: -0.05)
      */
      void setNodesToWheelsSplitThreshold( agx::Real splitThreshold );

      /**
      \return node to wheel split threshold
      */
      agx::Real getNodesToWheelsSplitThreshold() const;

      /**
      Average direction of non-merged nodes entering or exiting a wheel is used as
      reference direction to split of a merged node. This is the number of nodes to
      include into this average direction.
      \param numIncludedNodes - number of nodes to include in average direction (default: 3)
      */
      void setNumNodesIncludedInAverageDirection( agx::UInt numIncludedNodes );

      /**
      \return the number of nodes included in node average direction calculations
      */
      agx::UInt getNumNodesIncludedInAverageDirection() const;

      /**
      Minimum value of the normal force (the hinge force along the track) used in "internal"
      friction calculations. I.e., when the track is compressed, this value is used with
      the friction coefficient as a minimum stabilizing compliance. If this value is negative
      there will be stabilization when the track is compressed. Default: 100.
      \param minNormalForce - minimum normal force to calculate internal friction in the node hinges
      */
      void setMinStabilizingHingeNormalForce( agx::Real minNormalForce );

      /**
      \return the minimum normal force to calculate internal friction in the node hinges (default: 100)
      */
      agx::Real getMinStabilizingHingeNormalForce() const;

      /**
      Friction parameter of the internal friction in the node hinges. This parameter scales
      the normal force in the hinge. Default: 1.0
      \note This parameter can not be identified as a real friction coefficient when it's used
            to stabilize tracks under tension.
      \param frictionParameter - normal force scale parameter
      */
      void setStabilizingHingeFrictionParameter( agx::Real frictionParameter );

      /**
      \return normal force scale parameter
      */
      agx::Real getStabilizingHingeFrictionParameter() const;

      /**
      Resets all flags and values to default.
      */
      void resetToDefault();

      /**
      Writes compliance and damping to the given hinge instance.
      \param hinge - hinge to write compliance and damping to
      */
      void updateComplianceDamping( agx::Hinge& hinge ) const;

    public:
      AGXSTREAM_DECLARE_SERIALIZABLE( agxVehicle::TrackProperties );

    protected:
      enum FlagsEnum : agx::UInt32
      {
        ON_INITIALIZE_MERGE_NODES_TO_WHEELS     = 1 << 0,
        ON_INITIALIZE_TRANSFORM_NODES_TO_WHEELS = 1 << 1,
        HINGE_RANGE_ENABLED                     = 1 << 2,
      };
      using Flags = agx::BitState<FlagsEnum, agx::UInt32>;

      enum RealValue : agx::UInt
      {
        HINGE_RANGE_LOWER,
        HINGE_RANGE_UPPER,
        NODE_TO_WHEEL_OVERLAP,
        NODE_TO_WHEEL_MERGE_THRESHOLD,
        NODE_TO_WHEEL_SPLIT_THRESHOLD,
        NUM_NODES_IN_AVERAGE_DIRECTION,
        MIN_STABILIZING_NORMAL_FORCE,
        STABILIZING_FRICTION_PARAMETER,
        NUM_REAL_VALUES
      };
      using RealValues = agx::RealVector;

    protected:
      /**
      Reference counted object - protected destructor.
      */
      virtual ~TrackProperties();

    private:
      Flags m_flags;
      RealValues m_realValues;
      agx::Real m_hingeCompliance[ 5 ];
      agx::Real m_hingeDamping[ 5 ];
  };

  inline agx::Real TrackProperties::getHingeCompliance( agx::Hinge::DOF dof ) const
  {
    return m_hingeCompliance[ (agx::UInt)std::max<int>( dof, 0 ) ];
  }

  inline agx::Real TrackProperties::getHingeDamping( agx::Hinge::DOF dof ) const
  {
    return m_hingeDamping[ (agx::UInt)std::max<int>( dof, 0 ) ];
  }

  inline agx::Bool TrackProperties::getEnableHingeRange() const
  {
    return m_flags.Is( HINGE_RANGE_ENABLED );
  }

  inline agx::RangeReal TrackProperties::getHingeRangeRange() const
  {
    return agx::RangeReal{ m_realValues[ HINGE_RANGE_LOWER ], m_realValues[ HINGE_RANGE_UPPER ] };
  }
}
