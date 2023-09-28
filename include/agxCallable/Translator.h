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


#ifndef AGXCALLABLE_TRANSLATOR_H
#define AGXCALLABLE_TRANSLATOR_H


#include <agxCallable/export.h>
#include <agx/String.h>


namespace agxCallable
{
  class AGXCALLABLE_EXPORT Translator
  {
  public:

    static agx::String translateMethod(const agx::String& objectType, const agx::String& method, bool setter, agx::String *finalType = nullptr);

  private:

    static agx::String translateComponent(agx::String& objectType, const agx::String& component, bool setter);

  };
}

// Include guard
#endif
