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

#ifndef AGXMODEL_SURFACEVELOCITYCONVEYORBELT_H
#define AGXMODEL_SURFACEVELOCITYCONVEYORBELT_H

#include <agxModel/export.h>
#include <agxCollide/Geometry.h>
#include <agxCollide/Contacts.h>
#include <agxSDK/StepEventListener.h>
#include <agx/Plane.h>

#include <agx/debug.h>
#include <agx/Vector.h>
#include <agx/Vec3.h>

namespace agxCollide
{
  class Trimesh;
}


namespace agxSDK
{
  class Simulation;
}

namespace agxModel
{
  AGX_DECLARE_POINTER_TYPES(SurfaceVelocityConveyorBelt);

  /**
  This class models a conveyor belt using surface velocity.
  Only the surface behavior of an ideal conveyor belt is modeled here,
  not inner elasticity, motor torque, ...
  agxCollide::Geometry's surface velocity has the limitation that it is
  only defined in one direction (relative to the geometry).
  SurfaceVelocityConveyorBelt extends this behavior by taking a vector of points
  which are supposed to represent connected line segments or a closed loop along the conveyor belt (e.g. the mid line).
  Surface velocity is adapted after each point, pointing to the next one.
  The last point does not automatically point to the first one -
  in order to get a closed loop, the first point has to be added at the end again explicitly.

  SurfaceVelocityConveyorBelt inherits from agxCollide::Geometry and can be used instead of one.
  */
  class CALLABLE AGXMODEL_EXPORT SurfaceVelocityConveyorBelt : public agxCollide::Geometry
  {
    public:

      typedef agx::Vector<SurfaceVelocityConveyorBelt *> SurfaceVelocityConveyorBeltPtrVector;

      /**
      Constructor, based on list of points.
      The points are supposed to represent connected line segments or a closed loop along the conveyor belt (e.g. the mid line).
      */
      SurfaceVelocityConveyorBelt(const agx::Vec3Vector& points);

      /**
      Calculates the surface velocity based on the location of the contact point.
      Inherited from agxCollide::Geometry.
      \param point - ContactPoint in world coordinates that can be used to calculate the surface velocity.
      \param index - Which geometry in the contact is this one? Valid values: 0 or 1.
      The normal in 'point' points in the direction that 0 has to move to leave the contact with 0.
      \return the calculated velocity.
      */
      virtual agx::Vec3f calculateSurfaceVelocity( const agxCollide::LocalContactPoint& point, size_t index ) const override;

      /**
      Returns the vector of points. Assumed to be connected line segments or a closed loop along the conveyor belt (e.g. the mid line).
      Will not change during conveyor belt life time.
      */
      const agx::Vec3Vector& getPoints() const;

      /**
      Returns the vector of unit-length directions from point to point.
      Size will be one less than size of getPoints() or zero.
      Assumed to point within connected line segments or a closed loop along the conveyor belt (e.g. the mid line).
      Will not change during conveyor belt life time.
      */
      const agx::Vec3Vector& getUnitDirs() const;

      /// Sets the speed of the conveyor belt. Negative speed is allowed and understood as the opposite direction.
      void setSpeed(agx::Real speed);

      /// Gets the speed of the conveyor belt. Negative speed is allowed and understood as the opposite direction.
      agx::Real getSpeed() const;

      /**
      Enables debug rendering of conveyor belt.
      Comes at a computational cost (mainly due to special case rendering), recommended only for debugging.
      Will be updated each simulation step (so moving the conveyor belt while having a paused simulation
      will not update graphics; only the simulation step will update the graphics).
      \param sim The simulation that will trigger the graphics updates.
      \return Was enabling the debug rendering successful?
      */
      bool enableDebugRendering(agxSDK::Simulation* sim);

      /**
      Disables the debug rendering.
      \return Was disabling the debug rendering successful?
      */
      bool disableDebugRendering();

      /// Is debug rendering enabled?
      bool getEnableDebugRendering() const;

      /**
      This method will do a linear search in Simulation and return a pointer to the specified SurfaceVelocityConveyorBelt
      \param simulation - The instance of the simulation in which we are searching
      \param name - name of the SurfaceVelocityConveyorBelt we are searching for
      \returns pointer to the found SurfaceVelocityConveyorBelt or null if not found.
      */
      static SurfaceVelocityConveyorBelt* find(agxSDK::Simulation* simulation, const agx::Name& name );

      /**
      This method will do a linear search in Simulation and return a pointer to the specified SurfaceVelocityConveyorBelt
      \param simulation - The instance of the simulation in which we are searching
      \param uuid - uuid of the SurfaceVelocityConveyorBelt we are searching for
      \returns pointer to the found SurfaceVelocityConveyorBelt or null if not found.
      */
      static SurfaceVelocityConveyorBelt* find(agxSDK::Simulation* simulation, const agx::Uuid& uuid);

      /**
      This method will do a linear search in Simulation and return a vector with pointers to all matching SurfaceVelocityConveyorBelt's
      \param simulation - The instance of the simulation in which we are searching
      \param name - name of the SurfaceVelocityConveyorBelt's we are searching for
      \returns vector containing pointers to all matching SurfaceVelocityConveyorBelt's or null if not found.
      */
      static SurfaceVelocityConveyorBeltPtrVector findAll(agxSDK::Simulation* simulation, const agx::Name& name);


      AGXSTREAM_DECLARE_SERIALIZABLE( agxModel::SurfaceVelocityConveyorBelt );

    protected:
      virtual ~SurfaceVelocityConveyorBelt();
      SurfaceVelocityConveyorBelt();

    private:
      agx::Vec3Vector m_points;
      agx::Vec3Vector m_unitDirs;
      agx::Real m_speed;
      bool m_enableDebugRendering;
      agxSDK::StepEventListenerRef m_debugRenderListener;
  };
}

#endif
