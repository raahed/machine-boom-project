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

#ifndef AGXOSG_SIMULATIONTIMEPRINTER_H
#define AGXOSG_SIMULATIONTIMEPRINTER_H

#include <agxOSG/export.h>
#include <agxOSG/HudTextManager.h>
#include <agxSDK/GuiEventListener.h>


namespace agxOSG
{
  /**
  SimulationTimePrinter is a class that inherits from agxSDK::GuiEventListener and
  prints the simulation time to the hud.
  */
  class AGXOSG_EXPORT SimulationTimePrinter : public agxSDK::GuiEventListener
  {
    public:
      SimulationTimePrinter(agxOSG::HudTextManager* hudTextManager);

      /// Updates the HUD given current simulation time.
      void updateHud();

      /// Inherited from agxSDK::GuiEventListener.
      virtual void update(float, float) override;

    protected:
      virtual ~SimulationTimePrinter();

    protected:
      HudTextManagerObserver m_hudTextManager;
      HudTextManager::ReferencedTextRef m_text;
      agx::Real m_time;
  };

}



#endif


