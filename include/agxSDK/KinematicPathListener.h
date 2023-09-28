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

#ifndef AGXSDK_KINEMATICPATHLISTENER_H
#define AGXSDK_KINEMATICPATHLISTENER_H

#include <agxSDK/agxSDK.h>
#include <agxSDK/StepEventListener.h>
#include <agx/RigidBody.h>
#include <string>



namespace agxSDK
{
  /**
  Class that will update a rigid body's transformation and velocities
  based on a given path of transformations and time stamps.
  */
  class AGXPHYSICS_EXPORT KinematicPathListener : public StepEventListener
  {
  public:

    enum PathEndBehavior {
      STAND_STILL,
      CONTINUE_LAST,
      CYCLE
    };

    /// Help struct for storing transformation data per time step.
    struct PathConfiguration
    {
      PathConfiguration( agx::Real setTime, const agx::Vec3& setPosition, const agx::Quat& setRotation)
        : position(setPosition), time(setTime), rotation(setRotation) {}
      PathConfiguration() {}
      agx::Vec3 position;
      agx::Real time;
      agx::Quat rotation;
    };

    typedef agx::Vector<PathConfiguration> PathConfigurationVector;

    /**
    Constructs a KinematicPathListener from a file defining the path.
    \param fileName The file storing the data.
    In the file, each column should contain 8 numbers (time stamp, position, rotation).
    Lines can be commented out with '#'.
    \param rigidBody Pointer to the rigid body which should be moved.
    Its MotionControl should be KINEMATICS.
    \param pathEndBehavior What should happen at end of path?
    */
    KinematicPathListener( const std::string& fileName,
      agx::RigidBody* rigidBody, PathEndBehavior pathEndBehavior = STAND_STILL );


    /**
    Constructs a KinematicPathListener from a file defining the path.
    \param configurations A vector of previously created PathConfigurations.
    \param rigidBody Pointer to the rigid body which should be moved.
    Its MotionControl should be KINEMATICS.
    \param pathEndBehavior What should happen at end of path?
    */
    KinematicPathListener( const agx::Vector<PathConfiguration>& configurations,
      agx::RigidBody* rigidBody, PathEndBehavior pathEndBehavior = STAND_STILL );

    /// Sets a time offset to the path.
    void setTimeOffset(agx::Real);

    /// Sets a time offset to the path.
    agx::Real getTimeOffset() const;

    /// Inherited from StepEventListener. Manages the velocity updates here.
    virtual void preCollide(const agx::TimeStamp& /*time*/);

    /// Returns the index of the current path configuration (0 at beginning).
    agx::UInt32 getCurrentIndex() const;

    /// Returns the path configurations.
    const PathConfigurationVector& getPathConfigurations() const;

  private:
    void validateBody();
    void validatePath();
    void validatePathEndBehavior();

  protected:
    virtual ~KinematicPathListener();

  protected:
    agx::observer_ptr<agx::RigidBody> m_body;
    bool m_valid;
    PathEndBehavior m_pathEndBehavior;
    agx::Real m_cycleAccTime;
    PathConfigurationVector m_configurations;
    agx::UInt32 m_index;
    agx::Real m_timeOffset;
  };


  typedef agx::ref_ptr<KinematicPathListener> KinematicPathListenerRef;




}

#endif /* AGXSDK_KINEMATICPATHLISTENER_H */
