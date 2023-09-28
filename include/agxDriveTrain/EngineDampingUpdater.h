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

#include <agxSDK/StepEventListener.h>
#include <agxDriveTrain/Engine.h>


namespace agxDriveTrain
{

  AGX_DECLARE_POINTER_TYPES(EngineDampingUpdater);

  /**
   * The engine provided by the power line implementation is based on Euler
   * integration and is unstable in the presence of large derivatives in the
   * power curve. The EngineDampingUpdater is an attempt to stabilize unwanted
   * oscillations by dynamically update the engine body's mass and velocity
   * damping based on the engine's current angular velocity and the derivative
   * of the power curve.
   */
  class AGXMODEL_EXPORT EngineDampingUpdater : public agxSDK::StepEventListener
  {
    public:
      EngineDampingUpdater(agxDriveTrain::Engine* engine);
      virtual void pre(const agx::TimeStamp&) override;
      virtual void post(const agx::TimeStamp&) override;

      AGXSTREAM_DECLARE_SERIALIZABLE(agxDriveTrain::EngineDampingUpdater);

    protected:
      EngineDampingUpdater();
      virtual ~EngineDampingUpdater() {}

    private:
      agxDriveTrain::EngineRef m_engine;
      agx::Real m_defaultMass;
  };
}

