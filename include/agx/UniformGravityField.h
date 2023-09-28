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

#ifndef AGX_UNIFORMGRAVITYFIELD_H
#define AGX_UNIFORMGRAVITYFIELD_H

#include <agx/GravityField.h>

#include <agxData/Value.h>

namespace agx
{
  AGX_DECLARE_POINTER_TYPES(UniformGravityField);

  /**
  The class UniformGravityField calculates a gravity which is uniform in
  magnitude over the entire space and directed along a specified vector.
  */
  class AGXPHYSICS_EXPORT UniformGravityField : public GravityField
  {
    public:
      /**
      Constructor.
      \param gravity  Acceleration vector, specifying the gravity acceleration.
      */
      UniformGravityField( Vec3 gravity = Vec3(0, 0, -agx::GRAVITY_ACCELERATION) );

      /// Set the gravity vector
      void setGravity( const Vec3& gravity );

      /// \return the gravity vector
      Vec3 getGravity( ) const;

      /**
      \param position - The position at which the acceleration will be given (not used in this model).
      \return The uniform gravity
      */
      virtual Vec3 calculateGravity( const Vec3& position ) const override;

      /// Internal use
      virtual Task* createRigidBodyTask() override;

      /// Internal use
      virtual Task* createParticleSystemTask() override;


    protected:
      AGXSTREAM_DECLARE_SERIALIZABLE(agx::UniformGravityField);

      virtual ~UniformGravityField();

    private:
      agxData::ValueRefT<Vec3> m_gravity;
  };


  //////////////////// IMPLEMENTATION ////////////////

  inline Vec3 UniformGravityField::calculateGravity( const Vec3& ) const
  {
    return m_gravity->get();
  }

  inline void UniformGravityField::setGravity( const Vec3& gravity )
  {
    m_gravity->set(gravity);
  }

  inline Vec3 UniformGravityField::getGravity( ) const
  {
    return m_gravity->get();
  }
} //namespace agx
#endif
