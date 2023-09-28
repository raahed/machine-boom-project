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

#ifndef AGXMODEL_WINDANDWATERPARAMETERS_H
#define AGXMODEL_WINDANDWATERPARAMETERS_H

#include <agxModel/AddedMassDb.h>

#include <agx/Referenced.h>

namespace agxModel
{
  class WindAndWaterController;

  class AGXMODEL_EXPORT WindAndWaterParameters : public agx::Referenced
  {
    public:
      /**
      Tessellation levels of native, non-mesh shapes, e.g,
      sphere, capsule and cylinder.
      */
      enum ShapeTessellation
      {
        LOW,
        MEDIUM,
        HIGH,
        ULTRA_HIGH,
        DEFAULT_TESSELLATION = MEDIUM
      };

      enum Coefficient {
        PRESSURE_DRAG,
        VISCOUS_DRAG,
        LIFT,
        BUOYANCY
      };

      /**
      Translate from ShapeTessellation to scale of default meshing technique.
      */
      static agx::Real shapeTessellationLevelToResolution( agxModel::WindAndWaterParameters::ShapeTessellation tessellationLevel );

    public:

      /**
      Set pressure drag, viscous drag, lift or buoyancy coefficient
      \param type - PRESSURE_DRAG, VISCOUS_DRAG, LIFT or BUOYANCY
      \param coefficient - new coefficient
      */
      void setCoefficient( Coefficient type, agx::Real coefficient );

      /**
      \return coefficient of type PRESSURE_DRAG, VISCOUS_DRAG, LIFT or BUOYANCY
      */
      agx::Real getCoefficient( Coefficient type ) const;

      /**
      Set a hydrodynamic coefficient for a rigid body
      \param controller - The WindAndWaterController used.
      \param rb - The rigid body whose coefficient we want to change.
      \param type - The coefficient we want to change; PRESSURE_DRAG, VISCOUS_DRAG, LIFT or BUOYANCY.
      \param coefficient - Value of the new coefficient.
      */
      static void setHydrodynamicCoefficient( WindAndWaterController* controller, agx::RigidBody* rb, Coefficient type, agx::Real coefficient );

      /**
      Set a aerodynamic coefficient for a rigid body
      \param controller - The WindAndWaterController used.
      \param rb - The rigid body whose coefficient we want to change.
      \param type - The coefficient we want to change; PRESSURE_DRAG, VISCOUS_DRAG or LIFT.
      \param coefficient - Value of the new coefficient.
      */
      static void setAerodynamicCoefficient( WindAndWaterController* controller, agx::RigidBody* rb, Coefficient type, agx::Real coefficient );

      /**
      Assign tessellation level for native shapes (sphere, capsule and cylinder) used
      during aero- and hydrodynamics integrations. This parameter is ignored for
      mesh shape types and box.
      \param tessellation - tessellation level
      */
      void setShapeTessellationLevel( agxModel::WindAndWaterParameters::ShapeTessellation tessellation );

      /**
      \return the current shape tessellation level
      */
      agxModel::WindAndWaterParameters::ShapeTessellation getShapeTessellationLevel() const;

      /**
      \return the shape these parameters are associated with
      */
      const agxCollide::Shape* getShape() const;

      /**
      Internal method.
      */
      virtual void store( agxStream::OutputArchive& out ) const;

      /**
      Internal method.
      */
      virtual void restore( agxStream::InputArchive& in );

    protected:
      WindAndWaterParameters( const agxCollide::Shape* shape, agxModel::AddedMassDbRef addedMassDb );
      virtual ~WindAndWaterParameters();

      DOXYGEN_START_INTERNAL_BLOCK()
      virtual agx::Bool checkParameter( Coefficient type ) const = 0;
      DOXYGEN_END_INTERNAL_BLOCK()

    protected:
      agx::Real m_pressureDragCoefficient;
      agx::Real m_liftCoefficient;
      agx::Real m_viscousDragCoefficient;
      agx::Real m_buoyancyScaling;
      agxModel::AddedMassStorage::Identifier m_addedMassIdentifier;
      agxModel::AddedMassDbRef m_addedMassDb;
      agxCollide::ShapeConstObserver m_shape;
      ShapeTessellation m_tessellation;
  };

  typedef agx::ref_ptr< WindAndWaterParameters > WindAndWaterParametersRef;

  class AGXMODEL_EXPORT AerodynamicsParameters : public WindAndWaterParameters
  {
    protected:
      friend class WindAndWaterController;
      AerodynamicsParameters( const agxCollide::Shape* shape, agxModel::AddedMassDbRef addedMassDb );
      virtual ~AerodynamicsParameters();

      DOXYGEN_START_INTERNAL_BLOCK()
      virtual agx::Bool checkParameter( Coefficient type ) const override;
      DOXYGEN_END_INTERNAL_BLOCK()
  };

  typedef agx::ref_ptr< AerodynamicsParameters > AerodynamicsParametersRef;

  class AGXMODEL_EXPORT HydrodynamicsParameters : public WindAndWaterParameters
  {
    public:
      /**
      Initializes the added mass storage given filename to store and/or restore the data for this given shape.
      \note First time, i.e., if the file doesn't exist, this could potentially take a long time.
      \return true if successful - otherwise false
      */
      agx::Bool initializeAddedMassStorage( const agx::String& filename );

      /**
      \return added mass storage if present, otherwise null
      */
      const agxModel::AddedMassStorage* getAddedMassStorage() const;

    protected:
      friend class WindAndWaterController;
      HydrodynamicsParameters( const agxCollide::Shape* shape, agxModel::AddedMassDbRef addedMassDb );
      virtual ~HydrodynamicsParameters();

      DOXYGEN_START_INTERNAL_BLOCK()
      virtual agx::Bool checkParameter( Coefficient type ) const override;
      DOXYGEN_END_INTERNAL_BLOCK()
  };

  typedef agx::ref_ptr< HydrodynamicsParameters > HydrodynamicsParametersRef;

  AGX_FORCE_INLINE WindAndWaterParameters::ShapeTessellation WindAndWaterParameters::getShapeTessellationLevel() const
  {
    return m_tessellation;
  }
}

#endif // AGXMODEL_WINDANDWATERPARAMETERS_H
