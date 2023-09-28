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

#ifndef AGXRENDER_RENDERABLE_H
#define AGXRENDER_RENDERABLE_H


#include <agx/agxPhysics_export.h>
#include <agx/Referenced.h>
#include <agx/Referenced.h>

namespace agxSDK
{
  class Simulation;
}

namespace agxRender
{

  class RenderManager;

  AGX_DECLARE_POINTER_TYPES( Renderable );

  /// Inherit from this class and implement the render() method to make it it self during debug rendering.
  class AGXPHYSICS_EXPORT Renderable : public agx::Referenced
  {
    public:
      Renderable() : m_simulation( nullptr ), m_enable( true ) {}

      /// Pure virtual method that should implemented to render a derived class during debug rendering.
      virtual void render( agxRender::RenderManager* mgr ) = 0;

      /**
      \return true if this debug Renderable is enabled - otherwise false
      */
      bool getEnable() const;

      /**
      \return true if this debug Renderable is enabled - otherwise false
      */
      inline bool isEnabled() const { return getEnable(); }

      /**
      Enable/disable this Renderable. If disabled, no calls to render will be made.
      */
      void setEnable( bool enable );

    protected:

      /// Destructor
      virtual ~Renderable();

      friend class RenderManager;

      /// \return a pointer to the associated simulation
      agxSDK::Simulation* getSimulation();

      /// \return a pointer to the associated simulation
      const agxSDK::Simulation* getSimulation() const;

      /// Set the associated simulation
      void setSimulation( agxSDK::Simulation* simulation );

      /// Called when this Renderable is added to a RenderManager
      virtual void addNotification() {}

      /// Called when this Renderable is removed from a RenderManager
      virtual void removeNotification() {}

    private:
      agxSDK::Simulation* m_simulation;
      bool m_enable;
  };
}

#endif
