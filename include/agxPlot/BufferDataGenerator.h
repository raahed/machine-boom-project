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

#ifndef AGXPLOT_BUFFERDATAGENERATOR_H
#define AGXPLOT_BUFFERDATAGENERATOR_H

#include <agxPlot/DataGenerator.h>
#include <agx/Uuid.h>

namespace agxSDK
{
  class Simulation;
}

namespace agxPlot
{
  class AGXPHYSICS_EXPORT BufferDataGenerator : public DataGenerator
  {
  public:
    BufferDataGenerator(agx::Uuid entityId, const agx::String& dataPath, agxSDK::Simulation* simulation);

    virtual DataGenerator::Result getValue();
  private:
    agx::Uuid m_entityId;
    agx::String m_dataPath;
    agxSDK::Simulation* m_simulation;
  };
}

#endif