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
#include <agxModel/Controller1D.h>
#include <agxModel/export.h>


namespace agxModel {

  AGX_DECLARE_POINTER_TYPES(ControllerHandler);
  
  /**
  This is a PID handler that collects the Plant measured Process Variable (PV) and execute PID update() to set a
  new Manipulated Variable (MV) for the Plant.
  */
  class AGXMODEL_EXPORT ControllerHandler : public agxSDK::StepEventListener
  {
  public:
    AGX_DECLARE_POINTER_TYPES(Plant);
    /**
    The Plant (aka the Physical System) is added to the ControllHandler in the Control System.
    The Plant has a method for reading the current Plant Variable (PV) and a
    method for setting the Manipulated Variable (MV), also known as the Control Variable, controlling the plant.
    To implement your own plant extend this base class and override the two methods getProcessVariable 
    and setManipulatedVariable. Generally getProcessVariable reads a parameter in the simulation and 
    setManipulatedVariable change e.g. a target speed on a constraint.
    */
    class AGXMODEL_EXPORT Plant : public virtual agx::Referenced
    {
    public:
      Plant() = default;

      /// Create a plant 
      Plant(std::function<agx::Real(const agx::TimeStamp&)> getProcessVariable, std::function<void(agx::Real)>setManipulatedVariable)
      {
        m_getProcessVariable = getProcessVariable;
        m_setManipulatedVariable = setManipulatedVariable;
      };

      /// \return The current measured Process Variable (PV). This is called in the post step of the ControllerHandler.
      virtual agx::Real getProcessVariable(const agx::TimeStamp& time) { return m_getProcessVariable ? m_getProcessVariable(time) : 0; };

      /**
      * Set the Manipulated Variable(MV) controlling the Plant, also known as Control Variable.
      * This is called in the post step of the ControllerHandler.
      */
      virtual void setManipulatedVariable(agx::Real manipulatedVariable) { if (m_setManipulatedVariable) m_setManipulatedVariable(manipulatedVariable); };

    private:
      std::function<agx::Real(const agx::TimeStamp&)> m_getProcessVariable;
      std::function<void(agx::Real)> m_setManipulatedVariable;
    };

  public:

    ControllerHandler(Controller1D* controller, Plant* plant);

    Controller1DRef getController() const;

    PlantRef getPlant() const;

    /**
    Read Plant measured Process Variable (PV) and execute PID update to get an 
    updated Manipulated Variable and set the new Manipulated Variable to the Plant.
    \param time - the current simulation time
    */
    virtual void post(const agx::TimeStamp& time) override;

  private:
    Controller1DRef m_controller;
    PlantRef m_plant;

  };

}

