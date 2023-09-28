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
#include <agx/GeometryContactConstraint.h>

namespace agxSDK
{
  class Simulation;
}

namespace agx
{
  AGX_DECLARE_POINTER_TYPES( DirectMultiTangentsFrictionModel );

  /**
  Friction model with an arbitrary number of tangents in the friction plane. This
  model is only valid with solve type: agx::FrictionModel::DIRECT.

  Friction is non-linear where the friction force depends on the normal force and
  the normal force depends on the friction force. During the non-linear update in
  the direct solver, this model is selecting two active tangent directions given the
  current direction of the friction force such that the active tangents aligns with
  and is orthogonal to the current friction force.
  */
  class AGXPHYSICS_EXPORT DirectMultiTangentsFrictionModel : public ScaleBoxFrictionModel
  {
    public:
      /**
      Construct given number of tangents.
      \param numTangents - number of tangents in the friction plane
      */
      DirectMultiTangentsFrictionModel( UInt numTangents );

      /**
      Set number of tangents in the friction plane.
      \param numTangents - number of tangents
      */
      void setNumTangents( UInt numTangents );

      /**
      \return number of tangents in the friction plane
      */
      UInt getNumTangents() const;

    public:
      /**
      Matching the given geometry contact, if match, the geometry contact will be solved
      using agx::GeometryContactConstraint in the given simulation.
      \param geometryContact - geometry contact to match
      \param simulation - simulation the geometry contact is part of
      */
      virtual void matchForCustomContactImplementation( agxCollide::GeometryContact* geometryContact,
                                                        agxSDK::Simulation* simulation ) override;

      AGXSTREAM_DECLARE_SERIALIZABLE( agx::DirectMultiTangentsFrictionModel );

    protected:
      /**
      Default constructor used if restored.
      */
      DirectMultiTangentsFrictionModel();

      /**
      Reference counted object, protected destructor.
      */
      virtual ~DirectMultiTangentsFrictionModel();

    private:
      GeometryContactConstraintRef m_geometryContactConstraints;
  };
}
