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

#ifndef AGX_OCTAVE_DUMPER_H
#define AGX_OCTAVE_DUMPER_H


#include <agx/Task.h>
#include <agx/TaskTemplate.h>
#include <agx/IndexRange.h>
#include <agx/Parameter.h>
#include <agx/Vector.h>
#include <agxIO/Events.h>
#include <agx/Bound.h>
#include <agx/String.h>

namespace agx
{
  class OctaveDumper : public SerialTask
  {
  public:
    
    OctaveDumper(Device* device) : SerialTask("OctaveDumper", device) {}
    virtual void run() { agxAbort1("TODO"); }
    
  #if 0
    OctaveDumper(Device* device);
    virtual ~OctaveDumper();
    virtual void run();

  private:
    Val<agx::Bound1u> m_rangeParameter;
    Val<agx::Bool> m_shouldPrintParameter;
    agx::Bound1u m_range;
    size_t m_rangeLength;
    size_t m_fileSuffix;
        
    void writeArguments();
    void writeArgument(AbstractArgument* argument);
  #endif
  };
}

#endif

