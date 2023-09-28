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

#ifndef AGX_GRAVITYFIELD_H
#define AGX_GRAVITYFIELD_H

#include <agx/Component.h>
#include <agx/Vec3.h>

#include <agxStream/Serializable.h>

namespace agx
{
  class Task;

  AGX_DECLARE_POINTER_TYPES(GravityField);
  /**
     The class GravityField is responsible for calculating and applying a gravity force on
     a set of bodies and particles. It can also calculate the acceleration at arbitrary
     points in space.

     All sub-systems of a Simulation shares a single GravityField object, but the GravityField
     creates separate worker tasks for each sub-system.
  */
  class AGXPHYSICS_EXPORT GravityField : public agx::Component, public agxStream::Serializable
  {
    public:
      GravityField();

      /**
      Given \p position, a gravity acceleration will be calculated and returned.
      \param position - The position at which the acceleration will be given.
      \return The calculated acceleration at position.
      */
      virtual agx::Vec3 calculateGravity( const agx::Vec3& position ) const = 0;

    public:
      /// Internal use
      virtual Task* createRigidBodyTask() = 0;

      /// Internal use
      virtual Task* createParticleSystemTask() = 0;

      AGXSTREAM_DECLARE_ABSTRACT_SERIALIZABLE( agx::GravityField );

    protected:
      virtual void store( agxStream::OutputArchive& out ) const override;
      virtual void restore( agxStream::InputArchive& in ) override;

      /// Destructor
      virtual ~GravityField();
  };

} //namespace agx


#endif
