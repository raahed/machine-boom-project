/*
Copyright 2007-2023. Algoryx Simulation AB.

All AGX source code, intellectual property, documentation, sample code,
tutorials, scene files and technical white papers, is copyrighted, proprietary
and confidential material of Algoryx Simulation AB. You may not download, read,
store, distribute, publish, copy or otherwise disseminate, use or expose this
material unless having a written signed agreement with Algoryx Simulation AB, or having been
advised so by Algoryx Simulation AB for a time limited evaluation, or having purchased a
valid commercial license from Algoryx Simulation AB.

Algoryx Simulation AB disclaims all responsibilities for loss or damage caused
from using this software, unless otherwise stated in written agreements with
Algoryx Simulation AB.
*/

#ifndef AGXOSG_PRESSURE_GENERATOR_H
#define AGXOSG_PRESSURE_GENERATOR_H

#include <agx/Referenced.h>

#include <agxOSG/export.h>

namespace agxOSG
{
  class PressureAtlas;

  AGX_DECLARE_POINTER_TYPES(PressureGenerator);
  AGX_DECLARE_VECTOR_TYPES(PressureGenerator);

  /**
   * Base class for types that want to supply force updates to a PressureAtlas.
   * It's not strictly required to inherit from this in order to send force
   * updates, but it makes it possible to register the PressureGenerator with a
   * PressureAtlas and let the atlas be the owner of the PressureGenerator
   * instance.
   */
  class AGXOSG_EXPORT PressureGenerator : public agx::Referenced
  {
  public:
    PressureGenerator(agxOSG::PressureAtlas* atlas);
    agxOSG::PressureAtlas* getAtlas();

  protected:
    virtual ~PressureGenerator() {}

  private:
    agxOSG::PressureAtlas* m_atlas;
  };
}

#endif
