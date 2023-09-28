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



#include <agx/String.h>
#include <agx/Referenced.h>

namespace agxStream
{

  class Serializable;

  /// Class for listening to the restore of Serializable objects
  class AGXCORE_EXPORT RestoreListener : public agx::Referenced
  {
  public:
    RestoreListener();

    /**
    Virtual method called for each restored object during the de-serialization process.
    \param obj - Pointer to a newly restored object
    \param className - The name of the class for this restored object.
    */
    virtual void restore( Serializable *obj, const agx::String& className ) = 0;

  protected:
    virtual ~RestoreListener();
  };

  typedef agx::ref_ptr<RestoreListener> RestoreListenerRef;

} // namespace

