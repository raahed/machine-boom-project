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
#ifndef AGXOSG_SIMULATIONSERIALIZER_H
#define AGXOSG_SIMULATIONSERIALIZER_H

#include <agxSDK/StepEventListener.h>
#include <agxOSG/ExampleApplication.h>

namespace agxOSG
{

  /// class for serializing a simulation into a series for files on disk  0001_filename etc.
  class AGXOSG_EXPORT SimulationSerializer : public agxSDK::SimulationSerializer
  {
    public:

      /**
      Default constructor, sets interval to 30hz and filename to saved_scene
      \param application - Pointer to an ExampleApplication.
      */
      SimulationSerializer( agxOSG::ExampleApplication* application );

    protected:

      void write( const agx::TimeStamp& t ) ;

      virtual ~SimulationSerializer();

    private:
      agx::observer_ptr<agxOSG::ExampleApplication> m_app;
  };

  typedef agx::ref_ptr<SimulationSerializer> SimulationSerializerRef;
}
#endif
