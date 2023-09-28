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

#ifndef AGX_SleepThreshold_H
#define AGX_SleepThreshold_H

#include <agx/agxPhysics_export.h>
#include <agxStream/Serializable.h>

namespace agx
{

  /**
  */
  class AGXPHYSICS_EXPORT SleepThreshold : public agx::Referenced, public virtual agxStream::Serializable
  {
  public:

    /**
    \param velocity - Linear velocity threshold for sleeping
    \param acceleration - Linear acceleration threshold for sleeping
    \param angularVelocity - Angular velocity threshold for sleeping
    \param angularAcceleration - Angular acceleration threshold for sleeping
    */
    inline SleepThreshold( float velocity=float(0.05),
                           float acceleration=float(0.2),
                           float angularVelocity=float(1E10),
                           float angularAcceleration=1E10
                         );

    /// \return the threshold of the linear velocity magnitude when a body should be put to sleep
    inline float getVelocity() const;

    /// \return the threshold of the angular velocity magnitude when a body should be put to sleep
    inline float getAngularVelocity() const;

    /// \return the threshold of the linear acceleration magnitude when a body should be put to sleep
    inline float getAcceleration() const;

    /// \return the threshold of the angular acceleration magnitude when a body should be put to sleep
    inline float getAngularAcceleration() const;


    // Squared
    /// \return the threshold of the linear velocity magnitude when a body should be put to sleep
    inline float getVelocity2() const;

    /// \return the threshold of the angular velocity magnitude when a body should be put to sleep
    inline float getAngularVelocity2() const;

    /// \return the threshold of the linear acceleration magnitude when a body should be put to sleep
    inline float getAcceleration2() const;

    /// \return the threshold of the angular acceleration magnitude when a body should be put to sleep
    inline float getAngularAcceleration2() const;

    // Setters
    /// Set the threshold of the linear velocity magnitude when a body should be put to sleep
    inline void setVelocity( float vel );

    /// Set the threshold of the angular velocity magnitude when a body should be put to sleep
    inline void setAngularVelocity( float vel );

    /// set the threshold of the linear acceleration magnitude when a body should be put to sleep
    inline void setAcceleration( float acc );

    /// Set the threshold of the angular acceleration magnitude when a body should be put to sleep
    inline void setAngularAcceleration( float acc );

    AGXSTREAM_DECLARE_SERIALIZABLE( agx::SleepThreshold );

  protected:



    float m_velocityThreshold;
    float m_accelerationThreshold;
    float m_angularVelocityThreshold;
    float m_angularAccelerationThreshold;

    // float m_velocityThreshold2;
    // float m_accelerationThreshold2;
    // float m_angularVelocityThreshold2;
    // float m_angularAccelerationThreshold2;

  };

  typedef agx::ref_ptr<SleepThreshold> SleepThresholdRef;


  typedef SleepThreshold MergeSplitThreshold;

  typedef agx::ref_ptr<MergeSplitThreshold> MergeSplitThresholdRef;

  class AGXPHYSICS_EXPORT AutoSleepThreshold : public SleepThreshold
  {
  public:

    AutoSleepThreshold( float velocity=float(0.05),
                               float acceleration=1E10,
                               float angularVelocity=float(0.05),
                               float angularAcceleration=1E10,
                               float time=float(0.8) );

    /**
    Set the time-window during which the body must be below velocity/acceleration threshold before it is put to sleep.
    \param time - time in seconds
    */
    inline void setTime( float time );

    /**
    Get the time-window during which the body must be below velocity/acceleration threshold before it is put to sleep.
    \return time in seconds
    */
    inline float getTime() const;

    AGXSTREAM_DECLARE_SERIALIZABLE( agx::AutoSleepThreshold );

  protected:
    float m_time;
  };

  typedef agx::ref_ptr<AutoSleepThreshold> AutoSleepThresholdRef;


  // Implementation
  inline SleepThreshold::SleepThreshold( float velocity, float acceleration, float angularVelocity, float angularAcceleration ) :
    m_velocityThreshold(velocity), m_accelerationThreshold(acceleration), m_angularVelocityThreshold( angularVelocity ), m_angularAccelerationThreshold(angularAcceleration)
  {
    // m_velocityThreshold2 = m_velocityThreshold*m_velocityThreshold;
    // m_accelerationThreshold2=m_accelerationThreshold*m_accelerationThreshold;
    // m_angularVelocityThreshold2=m_angularVelocityThreshold*m_angularVelocityThreshold;
    // m_angularAccelerationThreshold2=m_angularAccelerationThreshold*m_angularAccelerationThreshold;
  }

  inline float SleepThreshold::getVelocity() const { return m_velocityThreshold;          }
  inline float SleepThreshold::getVelocity2() const { return m_velocityThreshold * m_velocityThreshold;          }

  inline void SleepThreshold::setVelocity( float vel ) { m_velocityThreshold = vel; }
  inline void SleepThreshold::setAcceleration( float acc ) { m_accelerationThreshold = acc; }
  inline void SleepThreshold::setAngularAcceleration( float acc ) { m_angularAccelerationThreshold = acc; }

  inline float SleepThreshold::getAngularAcceleration( ) const { return m_angularAccelerationThreshold; }
  inline float SleepThreshold::getAngularAcceleration2( ) const { return m_angularAccelerationThreshold * m_angularAccelerationThreshold; }

  inline float SleepThreshold::getAcceleration( ) const { return m_accelerationThreshold; }
  inline float SleepThreshold::getAcceleration2( ) const { return m_accelerationThreshold * m_accelerationThreshold; }

  inline float SleepThreshold::getAngularVelocity() const { return m_angularVelocityThreshold; }
  inline float SleepThreshold::getAngularVelocity2() const { return m_angularVelocityThreshold * m_angularVelocityThreshold; }

  inline void SleepThreshold::setAngularVelocity( float vel ) {  m_angularVelocityThreshold = vel; }

  inline void AutoSleepThreshold::setTime( float time ) {  m_time = time;  }
  inline float AutoSleepThreshold::getTime( ) const {  return m_time;  }

  inline std::ostream& operator <<(std::ostream& out, const SleepThreshold& t)
  {
    out << "Vt: " << t.getVelocity() << " AVt: " << t.getAngularVelocity() << " At: " << t.getAcceleration() << " AAt: " << t.getAngularAcceleration() << std::endl;
    return out;
  }
}

#endif /* AGX_SleepThreshold_H */
