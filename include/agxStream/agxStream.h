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

#include <agx/agxCore_export.h>

#include <agx/Vector.h>
#include <agx/agx.h>

#ifdef _MSC_VER
# pragma warning(push)
# pragma warning( disable : 4275 ) //  warning C4275: non dll-interface class
#endif


/**
\namespace agxStream
\brief This namespace contain classes for streaming classes into archives, ASCII, binary for storage (serialization).
*/
namespace agxStream
{
  class Serializable;

  class AGXCORE_EXPORT SerializablePtrVector : public agx::Vector<Serializable *>  {
  public:
    SerializablePtrVector() {}
    void deallocate();
    ~SerializablePtrVector() {}

  };
}

#ifdef _MSC_VER
# pragma warning(pop)
#endif

