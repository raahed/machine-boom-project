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

#ifndef AGX_AUTOSLEEP_H
#define AGX_AUTOSLEEP_H

#include <agx/agxPhysics_export.h>
#include <agx/Referenced.h>
#include <agx/SleepThreshold.h>
#include <agxCollide/Space.h>
#include <agxCollide/agxcollide_vector_types.h>
#include <agxData/Buffer.h>

namespace agx
{

  class DynamicsSystem;

  /**
  Class that will investigate bodies to see if they are resting, that is if
  linear/angular velocity and linear/angular acceleration is less than the corresponding
  threshold values.

  Global thresholds can be set on this class.
  Individual threshold can be set on each body using RigidBody::setAutoSleep()/getAutoSleep()
  */
  class AGXPHYSICS_EXPORT AutoSleep : public agx::Referenced
  {
  public:


    /// Default constructor
    AutoSleep( agx::DynamicsSystem *system );

    /**
    \return true if auto disable is enabled.
    */
    bool getEnable() const;

    /**
    \return true if auto disable is enabled.
    */
    bool isEnabled() const;

    /**
    Specify if auto-disable of dynamic Bodies should be enabled or not.
    Auto-disable is a feature that will depending on constrained system and bodies velocities
    put non-moving objects to sleep. This will affect performance in a positive way.
    \param enable - If true, auto-disable is enabled.
    */
    void setEnable( bool enable );

    /**
    \return a pointer to a AutoSleepThreshold object which holds the acceleration/velocity threshold settings
    */
    AutoSleepThreshold *getThreshold() { return m_threshold; }

    /**
    \return a const pointer to a AutoSleepThreshold object which holds the acceleration/velocity threshold settings
    */
    const AutoSleepThreshold *getThreshold() const { return m_threshold; }

  protected:
    virtual ~AutoSleep();
    friend class DynamicsSystem;

    /// Investigate if bodies should be awaken based on separations and contacts
    void updateSleepState( const agxCollide::SeparationPairVector& separations,
      const agxCollide::GeometryContactPtrVector& contacts );


    /// Investigate if bodies should be put to sleep/awaken
    void updateSleepState();
    RigidBodyPtrVector& getBodyConnections(RigidBody *body);

    bool m_enabled;
    DynamicsSystem *m_system;
    agx::ref_ptr<AutoSleepThreshold> m_threshold;
    RigidBodyPtrVector m_bodyConnections;
  };

  typedef agx::ref_ptr<AutoSleep> AutoSleepRef;
}

#endif /* _AGX_AutoSleep_H_ */
