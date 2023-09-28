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

#include <agxModel/export.h>

#include <agx/Referenced.h>

#include <agxStream/Serializable.h>

namespace agxModel
{
  AGX_DECLARE_POINTER_TYPES( BeamModelProperties );

  /**
  Bulk beam model properties present in all agxModel::BeamModel instances, which
  in turn propagates the bulk material properties to the constraints in an agxModel::Beam.
  Example usage:

      agxModel::BeamModelPropertiesRef steelProperties = new agxModel::BeamModelProperties( 210.0E9, 0.31 );

      // Creating a rectangular and an I-beam with steel bulk properties.
      agxModel::BeamModelRef rectangularModel = new agxModel::RectangularBeam( 0.15, 0.10, steelProperties );
      agxModel::BeamModelRef iBeamModel = new agxModel::IBeam( 0.15, 0.10, 0.012, 0.012, steelProperties );

      const auto beamResolution = 10u;
      agxModel::BeamRef rectangularBeam7m = new agxModel::Beam( rectangularModel, beamResolution, 7.0 );
      agxModel::BeamRef rectangularBeam4m = new agxModel::Beam( rectangularModel, beamResolution, 4.0 );
      agxModel::BeamRef iBeam3m = new agxModel::Beam( iBeamModel, beamResolution, 3.0 );
      agxModel::BeamRef iBeam4_5m = new agxModel::Beam( iBeamModel, beamResolution, 4.5 );

  Default values:
      Young's modulus: 200.0E9
      Poisson's ratio: 0.3333
      Damping time:    2.0 / 60.0 = 0.03333

  Damping time:
      For real-time simulations, i.e., time step size >> 0 ~ 0.01, there is no correlation between the
      material properties of a beam and the damping time parameter. If the time step size is decreased
      close to the true damping time of the simulated material, there's possible to map the damping
      time parameter to the viscosity. For real-time simulations it's important that the damping time
      parameter is larger than the time step, typically two times the time step size, for unconditionally
      stable simulations.
  */
  class AGXMODEL_EXPORT BeamModelProperties : public agx::Referenced, public agxStream::Serializable
  {
    public:
      /**
      Default constructor setting Young's modulus to 200.0E9, Poisson's ratio to 0.3333 and
      damping time to 0.03333.
      */
      BeamModelProperties();

      /**
      Construct given Young's modulus and Poisson's ratio. The damping time parameter will
      be set to 0.03333.
      \param youngsModulus - Young's modulus, will be clamped to the range in which it's valid: (0.0, infinity)
      \param poissonsRatio - Poisson's ratio, will be clamped to the range in which it's valid: (-0.5, 1.0]
      */
      BeamModelProperties( agx::Real youngsModulus, agx::Real poissonsRatio );

      /**
      Assigns new Young's modulus parameter. The value will be clamped to (0.0, infinity). Default: 200.0E9
      \param youngsModulus - Young's modulus, will be clamped to the range in which it's valid: (0.0, infinity)
      */
      void setYoungsModulus( agx::Real youngsModulus );

      /**
      \return the current value of Young's modulus
      */
      agx::Real getYoungsModulus() const;

      /**
      Assigns new Poisson's ratio parameter. The value will be clamped to (-0.5, 1.0]. Default: 0.3333
      \param poissonsRatio - Poisson's ratio, will be clamped to the range in which it's valid: (-0.5, 1.0]
      */
      void setPoissonsRatio( agx::Real poissonsRatio );

      /**
      \return the current value of Poisson's ratio
      */
      agx::Real getPoissonsRatio() const;

      /**
      Assigns new damping time parameter. This value will be clamped to [0.0, infinity). Default: 0.03333
      */
      void setDampingTime( agx::Real dampingTime );

      /**
      \return the current value of damping time
      */
      agx::Real getDampingTime() const;

    public:
      DOXYGEN_START_INTERNAL_BLOCK()

      AGXSTREAM_DECLARE_SERIALIZABLE( agxModel::BeamModelProperties );

    protected:
      virtual ~BeamModelProperties();

      DOXYGEN_END_INTERNAL_BLOCK()

    private:
      agx::Real m_youngsModulus;
      agx::Real m_poissonsRatio;
      agx::Real m_dampingTime;
  };
}
