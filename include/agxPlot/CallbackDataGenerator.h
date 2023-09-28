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

#ifndef AGXPLOT_CALLBACKDATAGENERATOR_H
#define AGXPLOT_CALLBACKDATAGENERATOR_H

#include <agxPlot/DataGenerator.h>

#include <agxSDK/EventListener.h>

namespace agxPlot
{
  class AGXPHYSICS_EXPORT DataListener : public agxSDK::EventListener
  {
    public:
      DataListener();

      virtual DataGenerator::Result getValue();

    protected:
      /// Destructor
      virtual ~DataListener();
  };
  AGX_DECLARE_POINTER_TYPES(DataListener);



  class CallbackDataGenerator : public DataGenerator
  {
  public:
    CallbackDataGenerator(DataListener* callback);

    virtual DataGenerator::Result getValue();

  protected:
    /// Destructor
    virtual ~CallbackDataGenerator();

  private:
    DataListenerRef m_callback;
  };
}

#endif
