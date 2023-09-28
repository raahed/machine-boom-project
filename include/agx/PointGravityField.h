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

#ifndef AGX_POINTGRAVITYFIELD_H
#define AGX_POINTGRAVITYFIELD_H

#include <agx/GravityField.h>

#include <agxData/Value.h>

namespace agx
{

  AGX_DECLARE_POINTER_TYPES(PointGravityField);
  /**
  The class PointGravityField calculates a gravity which is uniform in
  magnitude over the entire space and directed from the given position
  outwards from origin. Inwards for negative magnitudes.
  */
  class AGXPHYSICS_EXPORT PointGravityField : public GravityField
  {
    public:
      /**
      Constructor.
      \param center - The center of the gravity field
      \param gravityMagnitude the magnitude for the PointGravity (direction is calculated later based in pos)
      */
      PointGravityField( const Vec3& center = Vec3(), Real gravityMagnitude = agx::GRAVITY_ACCELERATION );

      /// Set the gravity direction/magnitude
      void setGravity( const Real gravityMagnitude );

      //The point toward where gravity accelerates bodies (earthCenter if on earth)
      void setCenter( const Vec3& center );

      /// \return the direction/magnitude
      Real getGravity( ) const;


      /// \return the direction/magnitude
      Vec3 getCenter( ) const;

      /**
      Given \p position, a gravity acceleration will be calculated and returned.
      \param position - The position at which the acceleration will be given.
      \return The calculated acceleration at position.
      */
      virtual Vec3 calculateGravity( const Vec3& position ) const override;

    public:
      /// Internal use
      virtual Task* createRigidBodyTask() override;

      /// Internal use
      virtual Task* createParticleSystemTask() override;

      AGXSTREAM_DECLARE_SERIALIZABLE(agx::PointGravityField);


    protected:
      virtual ~PointGravityField();

    private:
      agxData::ValueRefT<Real>  m_gravity;
      agxData::ValueRefT<Vec3>  m_center;
  };

  //////////////////// IMPLEMENTATION ////////////////
  inline Vec3 PointGravityField::calculateGravity( const Vec3& position) const
  {
    Vec3 pos = m_center->get() - position;
    pos.normalize();

    return pos * m_gravity->get();
  }

  inline Real PointGravityField::getGravity() const
  {
    return m_gravity->get();
  }

  inline Vec3 PointGravityField::getCenter() const
  {
    return m_center->get();
  }
} //namespace agx






#endif

